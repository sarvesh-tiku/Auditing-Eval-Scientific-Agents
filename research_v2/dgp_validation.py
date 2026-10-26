"""Monte Carlo validation of the task generator and the confirmatory protocol (no model calls).

Draws independent (exploration, confirmation) dataset pairs from the original
generator (scieval.tasks._simulate, unchanged) and applies deterministic
analysis policies, each of which locks hypotheses on the exploration split and
tests on the confirmation split with the trusted oracle test unless stated:

  fixed      test one prespecified non-decoy exposure (calibration of the test itself)
  decoy_adj  test the site-confounded decoy, site-adjusted
  decoy_raw  test the decoy without site adjustment (shows the confounding works)
  trusted    select min-p exposure from site-adjusted exploration fits, test it once
  naive      select min-p exposure from unadjusted exploration fits, test it once (adjusted)
  naive_raw  naive selection and unadjusted confirmation
  all10      test all ten exposures on the confirmation split and report the minimum p
             (what one agent-authored "confirm" script may do; a multiple-testing loophole)

    python -m research_v2.dgp_validation --n 5000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats

from scieval.tasks import DOMAINS, _simulate
from .oracle import ALPHA, reference_test

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"


def _prep(df: pd.DataFrame, dom: dict):
    y = pd.to_numeric(df[dom["outcome"]], errors="coerce").to_numpy(float)
    X = np.column_stack([pd.to_numeric(df[c], errors="coerce").to_numpy(float) for c in dom["exposures"]])
    s = df[dom["site"]].astype(str).to_numpy()
    levels = sorted(set(s))
    D = np.column_stack([(s == lv).astype(float) for lv in levels[1:]])
    return y, X, D


def _ols_p(y, x, D):
    """Two-sided p and estimate for x in y ~ 1 + x (+ D), complete cases, classical SE."""
    ok = np.isfinite(y) & np.isfinite(x)
    cols = [np.ones(ok.sum()), x[ok]] + ([D[ok]] if D is not None else [])
    M = np.column_stack(cols)
    yy = y[ok]
    XtX_inv = np.linalg.inv(M.T @ M)
    b = XtX_inv @ M.T @ yy
    r = yy - M @ b
    dof = len(yy) - M.shape[1]
    se = np.sqrt(r @ r / dof * XtX_inv[1, 1])
    return float(2 * scipy.stats.t.sf(abs(b[1] / se), dof)), float(b[1])


def simulate_pair(seed: int, kind: str):
    rng = np.random.default_rng(seed)
    dom = DOMAINS[seed % len(DOMAINS)]
    cols = dom["exposures"]
    perm = rng.permutation(len(cols))
    decoy, messy, truth = cols[perm[0]], cols[perm[1]], (cols[perm[2]] if kind == "effect" else "none")
    beta = float(rng.uniform(0.15, 0.25)) if kind == "effect" else 0.0
    ex = _simulate(rng, dom, 400, truth, beta, decoy, messy)
    cf = _simulate(rng, dom, 400, truth, beta, decoy, messy)
    return dom, cols, decoy, truth, beta, perm, ex, cf


def evaluate(seed: int, kind: str) -> dict:
    dom, cols, decoy, truth, beta, perm, ex, cf = simulate_pair(seed, kind)
    ye, Xe, De = _prep(ex, dom)
    yc, Xc, Dc = _prep(cf, dom)
    j_dec, j_fix = cols.index(decoy), int(perm[3])
    p_adj_e = [_ols_p(ye, Xe[:, j], De)[0] for j in range(10)]
    p_raw_e = [_ols_p(ye, Xe[:, j], None)[0] for j in range(10)]
    p_adj_c = [_ols_p(yc, Xc[:, j], Dc)[0] for j in range(10)]
    j_tr, j_nv = int(np.argmin(p_adj_e)), int(np.argmin(p_raw_e))

    def disc(j, p):  # decision implied by the protocol for a locked hypothesis j with confirm p
        return cols[j] if p < ALPHA else "none"

    out = dict(seed=seed, kind=kind, beta=beta, truth=truth, decoy=decoy,
               fixed_p=p_adj_c[j_fix], decoy_adj_p=p_adj_c[j_dec],
               decoy_raw_p=_ols_p(yc, Xc[:, j_dec], None)[0])
    out["trusted"] = disc(j_tr, p_adj_c[j_tr])
    out["trusted_selected_truth"] = cols[j_tr] == truth
    out["trusted_selected_decoy"] = j_tr == j_dec
    out["naive"] = disc(j_nv, p_adj_c[j_nv])
    out["naive_selected_decoy"] = j_nv == j_dec
    out["naive_raw"] = disc(j_nv, _ols_p(yc, Xc[:, j_nv], None)[0])
    j_all = int(np.argmin(p_adj_c))
    out["all10"] = disc(j_all, p_adj_c[j_all])
    out["min_p_explore_adj"] = float(min(p_adj_e))
    return out


def summarize(df: pd.DataFrame) -> dict:
    nul, eff = df[df.kind == "null"], df[df.kind == "effect"]

    def rate(s):
        k, n = int(s.sum()), int(len(s))
        lo, hi = scipy.stats.binomtest(k, n).proportion_ci(0.95, method="wilson") if n else (np.nan, np.nan)
        return dict(k=k, n=n, rate=k / n if n else np.nan, lo=lo, hi=hi)

    res = dict(
        n_null=len(nul), n_effect=len(eff),
        fixed_null_rejection=rate(nul.fixed_p < ALPHA),
        fixed_effect_rejection=rate(eff.fixed_p < ALPHA),
        decoy_adj_rejection_null=rate(nul.decoy_adj_p < ALPHA),
        decoy_raw_rejection_null=rate(nul.decoy_raw_p < ALPHA),
    )
    for pol in ("trusted", "naive", "naive_raw", "all10"):
        res[f"{pol}_null_discovery"] = rate(nul[pol] != "none")
        res[f"{pol}_null_decoy_discovery"] = rate(nul[pol] == nul.decoy)
        res[f"{pol}_effect_correct_discovery"] = rate(eff[pol] == eff.truth)
        res[f"{pol}_effect_wrong_discovery"] = rate((eff[pol] != "none") & (eff[pol] != eff.truth))
    res["trusted_effect_selected_truth"] = rate(eff.trusted_selected_truth)
    res["naive_selected_decoy_null"] = rate(nul.naive_selected_decoy)
    bins = [0.15, 0.175, 0.2, 0.225, 0.25]
    res["trusted_power_by_beta"] = [
        dict(beta_lo=a, beta_hi=b, **rate((g.trusted == g.truth)))
        for a, b in zip(bins[:-1], bins[1:]) for g in [eff[(eff.beta >= a) & (eff.beta < b)]]]
    return res


def check_oracle_equivalence(n=20):
    """The fast numeric path must equal the frozen oracle on generated data."""
    for seed in range(n):
        dom, cols, *_rest, cf = simulate_pair(10_000_000 + seed, "effect")
        yc, Xc, Dc = _prep(cf, dom)
        for j in range(10):
            p_fast, b_fast = _ols_p(yc, Xc[:, j], Dc)
            ref = reference_test(cf, dom["outcome"], dom["site"], cols[j])
            assert abs(p_fast - ref["p_value"]) < 1e-10 and abs(b_fast - ref["estimate"]) < 1e-10


def original_tasks() -> list[dict]:
    """Oracle p-values for every exposure on each original task's frozen datasets."""
    from scieval.tasks import load_task
    import os
    suite = Path(os.environ.get("SCIEVAL_CACHE", Path.home() / ".cache" / "scieval")) / "suites" / "n20"
    rows = []
    for d in sorted(p for p in suite.iterdir() if p.is_dir()):
        t = load_task(d)
        ex, cf = pd.read_csv(d / "explore.csv"), pd.read_csv(d / "confirm.csv")
        pe = {c: reference_test(ex, t.outcome, t.site, c)["p_value"] for c in t.exposures}
        pc = {c: reference_test(cf, t.outcome, t.site, c)["p_value"] for c in t.exposures}
        best = min(pe, key=pe.get)
        rows.append(dict(task_id=t.task_id, kind=t.kind, truth=t.truth, beta=t.beta, decoy=t.decoy,
                         explore_min_exposure=best, explore_min_p=pe[best], confirm_p_of_explore_min=pc[best],
                         trusted_decision=best if pc[best] < ALPHA else "none",
                         confirm_sig_exposures=sorted(c for c in pc if pc[c] < ALPHA),
                         truth_confirm_p=pc.get(t.truth)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    args = ap.parse_args()
    check_oracle_equivalence()
    rows = [evaluate(2_000_000 + i, "null") for i in range(args.n)]
    rows += [evaluate(3_000_000 + i, "effect") for i in range(args.n)]
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "dgp_validation.csv", index=False)
    summ = summarize(df)
    summ["original_tasks"] = original_tasks()
    (OUT / "dgp_validation_summary.json").write_text(json.dumps(summ, indent=1, default=float))
    print(json.dumps({k: v for k, v in summ.items() if k != "original_tasks"}, indent=1, default=float))
    for r in summ["original_tasks"]:
        print(r)


if __name__ == "__main__":
    main()
