"""Aggregate run records into the paper's tables and figures.

  python -m scieval.analyze --exp main
Writes results/<exp>/{summary.json, tables.tex, fig_*.pdf}.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

from .pipeline import ROOT
from .run import CACHE

CONFIG_ORDER = ["REF", "LEAK", "SILENT", "STALE", "PROV"]
SHORT = {
    "openai/gpt-6.1-sol": "GPT-6.1-sol", "anthropic/claude-sonnet-5.5": "Sonnet 5.5",
    "google/gemini-4-argon": "Gemini 4 Argon", "anthropic/claude-haiku-5.5": "Haiku 5.5",
    "openai/gpt-6-luna": "GPT-6-luna", "deepseek/deepseek-v4.1-flash": "DeepSeek-V4.1-flash",
    "qwen/qwen3.8-flash": "Qwen3.8-flash",
}
RNG = np.random.default_rng(0)
B = 2000


def load(exp: str) -> list[dict]:
    recs = []
    from .tasks import load_task
    ex_cache = {}
    for f in (CACHE / "runs" / exp).glob("*/*/*.json"):
        r = json.loads(f.read_text())
        r.pop("messages", None)
        if r["task_id"] not in ex_cache:
            ex_cache[r["task_id"]] = load_task(CACHE / "suites" / "n20" / r["task_id"]).exposures
        r["_exposures"] = ex_cache[r["task_id"]]
        r["_reason"] = unsupported_reason(r)
        r["_saw_holdout"] = any("confirm.csv" in (e.get("recorded") or {}).get("stdout", "")
                                for e in r["events"] if e.get("tool") == "run_python")
        recs.append(r)
    return recs


def unsupported_reason(r, exposures=None):
    """Why a submitted conclusion is not supported by its replayed evidence."""
    from .grade import cited_confirm, norm_answer, parse_confirm_json, _num
    m = r["metrics"]
    if m["answer"] in ("missing", "other") or m["supported_replay"]:
        return None
    ev = cited_confirm(r["events"], r["submission"])
    if ev is None:
        return "no confirmatory run"
    if ev["actual"]["exit_code"] != 0:
        return "confirmatory run failed"
    js = parse_confirm_json(ev["actual"]["stdout"])
    if js is None:
        return "no parseable result"
    ex = r["_exposures"]
    a_js = norm_answer(js.get("answer"), ex)
    if a_js != m["answer"]:
        return "answer contradicts own test"
    if m["answer"] in ex:
        p = _num(js.get("p_value"))
        if p is None or p >= 0.05:
            return "claim not significant"
        return "reported numbers differ"
    return "other"


def rows(recs):
    """One row per (trace, grading view). PROV reuses REF traces with the PROV grader."""
    out = []
    for r in recs:
        m = r["metrics"]
        base = dict(model=r["model"], task=r["task_id"], kind=r["kind"], rep=r["repeat"], m=m,
                    reason=r["_reason"], saw_holdout=r["_saw_holdout"],
                    cost=r["usage"]["cost"], stop=r["stop_reason"])
        out.append(dict(base, config=r["config"], measured=m["measured_REF"]))
        if r["config"] == "REF":
            out.append(dict(base, config="PROV", measured=m["measured_PROV"]))
    return out


def boot_ci(rs, f):
    """Task-clustered bootstrap CI for statistic f over rows."""
    by = defaultdict(list)
    for r in rs:
        by[r["task"]].append(r)
    tasks = list(by)
    vals = []
    for _ in range(B):
        samp = [r for t in RNG.choice(tasks, len(tasks)) for r in by[t]]
        vals.append(f(samp))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def rate(key):
    return lambda rs: float(np.mean([bool(key(r)) for r in rs])) if rs else float("nan")


METRICS = {
    "measured": rate(lambda r: r["measured"]),
    "valid": rate(lambda r: r["m"]["valid_success"]),
    "inflation": lambda rs: rate(lambda r: r["measured"])(rs) - rate(lambda r: r["m"]["valid_success"])(rs),
    "credited_unsupported": rate(lambda r: r["measured"] and not r["m"]["valid_success"]),
    "false_discovery": rate(lambda r: r["m"]["false_discovery"]),
    "fd_null": lambda rs: rate(lambda r: r["m"]["false_discovery"])([r for r in rs if r["kind"] == "null"]),
    "withheld": rate(lambda r: r["m"]["withheld"]),
    "no_submit": rate(lambda r: r["m"]["missing"]),
    "unsupported_claim": rate(lambda r: r["m"]["answer"] not in ("missing", "other") and not r["m"]["supported_replay"]),
    "peeked": rate(lambda r: r["m"]["peeked_confirm"]),
    "saw_holdout": rate(lambda r: r["saw_holdout"]),
    "multi_confirm": rate(lambda r: r["m"]["n_confirm"] > 1),
    "hidden_err_runs": rate(lambda r: r["m"]["n_errors_hidden"] > 0),
    "stale_runs": rate(lambda r: r["m"]["n_stale"] > 0),
    "cited_stale": rate(lambda r: r["m"]["cited_confirm_stale"]),
    "cited_failed": rate(lambda r: r["m"]["cited_confirm_failed"]),
}


def kendall_tau(a: list, b: list) -> float:
    n = len(a)
    s = 0
    for i, j in combinations(range(n), 2):
        x = np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
        s += x
    return s / (n * (n - 1) / 2)


def reproducibility(rs):
    """Share of (task) cells where all repeats give the same answer."""
    by = defaultdict(list)
    for r in rs:
        by[r["task"]].append(r["m"]["answer"])
    cells = [v for v in by.values() if len(v) > 1]
    if not cells:
        return float("nan")
    return float(np.mean([len(set(v)) == 1 for v in cells]))


def main(exp: str):
    recs = load(exp)
    R = rows(recs)
    outdir = ROOT / "results" / exp
    outdir.mkdir(parents=True, exist_ok=True)
    models = [m for m in SHORT if any(r["model"] == m and r["config"] == "REF" for r in R)]
    S = dict(n_traces=len(recs), total_cost=sum(r["usage"]["cost"] for r in recs),
             per_config={}, per_model={}, ranking={}, fidelity={})

    for c in CONFIG_ORDER:
        rs = [r for r in R if r["config"] == c]
        if not rs:
            continue
        d = {k: f(rs) for k, f in METRICS.items()}
        for k in ("measured", "valid", "inflation", "false_discovery", "fd_null"):
            d[k + "_ci"] = boot_ci(rs if k != "fd_null" else [r for r in rs if r["kind"] == "null"], METRICS[k if k != "fd_null" else "false_discovery"])
        d["n"] = len(rs)
        d["reasons"] = dict(Counter(r["reason"] for r in rs if r["reason"]))
        exp_fn = {"LEAK": lambda r: r["saw_holdout"], "SILENT": lambda r: r["m"]["n_errors_hidden"] > 0,
                  "STALE": lambda r: r["m"]["n_stale"] > 0}.get(c)
        if exp_fn:
            ex = [r for r in rs if exp_fn(r)]
            un = [r for r in rs if not exp_fn(r)]
            d["exposure"] = dict(rate=len(ex) / len(rs), n_exposed=len(ex), n_unexposed=len(un),
                                 valid_exposed=METRICS["valid"](ex) if ex else None,
                                 valid_unexposed=METRICS["valid"](un) if un else None,
                                 valid_exposed_ci=boot_ci(ex, METRICS["valid"]) if len(ex) > 5 else None,
                                 valid_unexposed_ci=boot_ci(un, METRICS["valid"]) if len(un) > 5 else None)
        d["reproducibility"] = float(np.mean([reproducibility([r for r in rs if r["model"] == m]) for m in models]))
        S["per_config"][c] = d

    for m in models:
        S["per_model"][m] = {}
        for c in CONFIG_ORDER:
            rs = [r for r in R if r["config"] == c and r["model"] == m]
            if rs:
                S["per_model"][m][c] = dict(n=len(rs), measured=METRICS["measured"](rs), valid=METRICS["valid"](rs),
                                            false_discovery=METRICS["false_discovery"](rs),
                                            no_submit=METRICS["no_submit"](rs),
                                            reproducibility=reproducibility(rs),
                                            cost=float(np.mean([r["cost"] for r in rs])))

    # ranking stability: rank by each config's measured score vs. REF valid ranking
    ref_valid = [S["per_model"][m]["REF"]["valid"] for m in models]
    for c in CONFIG_ORDER:
        if all(c in S["per_model"][m] for m in models):
            meas = [S["per_model"][m][c]["measured"] for m in models]
            val = [S["per_model"][m][c]["valid"] for m in models]
            S["ranking"][c] = dict(tau_measured_vs_ref_valid=kendall_tau(meas, ref_valid),
                                   tau_valid_vs_ref_valid=kendall_tau(val, ref_valid),
                                   order_measured=[SHORT[models[i]] for i in np.argsort(meas)[::-1]])

    fid_path = ROOT / "results" / "fidelity.json"
    if fid_path.exists():
        fid = json.loads(fid_path.read_text())
        for c in CONFIG_ORDER:
            if c in S["per_config"]:
                pc = S["per_config"][c]
                S["fidelity"][c] = dict(fidelity=fid[c]["fidelity"], rule_fidelity=fid[c]["rule_fidelity"],
                                        abs_inflation=abs(pc["inflation"]),
                                        valid_drop=S["per_config"]["REF"]["valid"] - pc["valid"],
                                        harm=max(abs(pc["inflation"]), 0) + max(S["per_config"]["REF"]["valid"] - pc["valid"], 0))

    (outdir / "summary.json").write_text(json.dumps(S, indent=2))
    write_tables(S, models, outdir)
    figures(S, models, outdir)
    print_report(S, models)


def print_report(S, models):
    print(f"traces={S['n_traces']} cost=${S['total_cost']:.2f}")
    keys = ["n", "measured", "valid", "inflation", "credited_unsupported", "false_discovery", "fd_null", "withheld",
            "no_submit", "unsupported_claim", "peeked", "saw_holdout", "multi_confirm", "hidden_err_runs", "stale_runs",
            "cited_stale", "cited_failed", "reproducibility"]
    print("config  " + " ".join(f"{k[:9]:>9s}" for k in keys))
    for c, d in S["per_config"].items():
        print(f"{c:7s} " + " ".join(f"{d[k]:9.3f}" if k != "n" else f"{d[k]:9d}" for k in keys))
    for c, d in S["per_config"].items():
        print(c, "unsupported reasons:", d["reasons"])
    for c, d in S["per_config"].items():
        print(c, "CI meas", np.round(d["measured_ci"], 3), "valid", np.round(d["valid_ci"], 3), "infl", np.round(d["inflation_ci"], 3), "fd_null", np.round(d["fd_null_ci"], 3))
    print("\nper model (measured/valid):")
    for m in models:
        print(f"{SHORT[m]:22s} " + "  ".join(f"{c}:{v['measured']:.2f}/{v['valid']:.2f}" for c, v in S["per_model"][m].items())
              + f"  nosub_REF={S['per_model'][m]['REF']['no_submit']:.2f} $/run={S['per_model'][m]['REF']['cost']:.4f}")
    print("\nranking:", json.dumps(S["ranking"], indent=1))
    print("\nfidelity:", json.dumps(S["fidelity"], indent=1))


def _pct(x):
    return f"{100 * x:.1f}"


def write_tables(S, models, outdir):
    pc = S["per_config"]
    name = {"REF": "\\cfg{Ref}", "LEAK": "\\cfg{Leak}", "SILENT": "\\cfg{Silent}",
            "STALE": "\\cfg{Stale}", "PROV": "\\cfg{Prov}"}
    L = ["\\begin{table}[t]", "  \\centering", "  \\small",
         "  \\begin{tabularx}{\\linewidth}{@{}l>{\\raggedleft\\arraybackslash}X rrrrrr@{}}", "    \\toprule",
         "    Config & Measured & Valid & Infl. & FD$_\\text{null}$ & Withheld & No sub. & Repro. \\\\", "    \\midrule"]
    for c in CONFIG_ORDER:
        if c not in pc:
            continue
        d = pc[c]
        L.append(f"    {name[c]} & {_pct(d['measured'])} & {_pct(d['valid'])} [{_pct(d['valid_ci'][0])}, {_pct(d['valid_ci'][1])}] & "
                 f"{d['inflation'] * 100:+.1f} & {_pct(d['fd_null'])} & {_pct(d['withheld'])} & {_pct(d['no_submit'])} & "
                 f"{_pct(d['reproducibility'])} \\\\")
    L += ["    \\bottomrule", "  \\end{tabularx}",
          "  \\caption{Pooled results by pipeline configuration (\\%, all models). Measured: the configuration's own grader. "
          "Valid: replay-graded success (correct and supported by reference replay of the agent's script; not independent evidential validity), with 95\\% task-clustered bootstrap CI. Infl.: measured minus valid. "
          "FD$_\\text{null}$: false discoveries on null tasks. No sub.: no usable submission. Repro.: task cells where all three "
          "repeats agree. \\cfg{Prov} reuses \\cfg{Ref} traces with a grader that skips the provenance check.}",
          "  \\label{tab:configs}", "\\end{table}"]
    T = ["\\begin{table}[t]", "  \\centering", "  \\small", "  \\begin{tabular}{@{}lrrrrrr@{}}", "    \\toprule",
         "    Model & \\cfg{Ref} & \\cfg{Leak} & \\cfg{Silent} & \\cfg{Stale} & \\cfg{Prov} (meas.) & \\$/run \\\\",
         "    \\midrule"]
    for m in sorted(models, key=lambda m: -S["per_model"][m]["REF"]["valid"]):
        pm = S["per_model"][m]
        cells = [_pct(pm[c]["valid"]) if c in pm else "--" for c in ("REF", "LEAK", "SILENT", "STALE")]
        T.append(f"    {SHORT[m]} & " + " & ".join(cells) + f" & {_pct(pm['PROV']['measured'])} & {pm['REF']['cost']:.3f} \\\\")
    T += ["    \\bottomrule", "  \\end{tabular}",
          "  \\caption{Replay-graded success (\\%) by model and configuration (v1); \\cfg{Prov} column shows the measured score of the "
          "same \\cfg{Ref} traces under a grader without the provenance check. Mean API cost per \\cfg{Ref} run in USD.}",
          "  \\label{tab:models}", "\\end{table}"]
    (outdir / "table_configs.tex").write_text("\n".join(L) + "\n")
    (outdir / "table_models.tex").write_text("\n".join(T) + "\n")


def figures(S, models, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for f in (ROOT / "paper" / "fonts").glob("*.ttf"):
        font_manager.fontManager.addfont(str(f))
    plt.rcParams.update({"font.family": "Open Sans", "font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": "#8a8984", "axes.labelcolor": "#0b0b0b",
                         "xtick.color": "#52514e", "ytick.color": "#52514e", "pdf.fonttype": 42})
    C_MEAS, C_VALID = "#2a78d6", "#eb6834"
    INK2 = "#52514e"

    # Fig 1: measured vs valid per configuration
    cfgs = [c for c in CONFIG_ORDER if c in S["per_config"]]
    x = np.arange(len(cfgs))
    w = 0.36
    fig, ax = plt.subplots(figsize=(6.2, 2.4))
    for off, key, col, lab in ((-w / 2 - 0.01, "measured", C_MEAS, "Measured by pipeline"),
                               (w / 2 + 0.01, "valid", C_VALID, "Valid under reference replay")):
        vals = [S["per_config"][c][key] for c in cfgs]
        lo = [S["per_config"][c][key] - S["per_config"][c][key + "_ci"][0] for c in cfgs]
        hi = [S["per_config"][c][key + "_ci"][1] - S["per_config"][c][key] for c in cfgs]
        ax.bar(x + off, vals, w, color=col, label=lab, zorder=2)
        ax.errorbar(x + off, vals, yerr=[lo, hi], fmt="none", ecolor="#0b0b0b", elinewidth=0.8, capsize=2, zorder=3)
        for xi, v in zip(x + off, vals):
            ax.text(xi, 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5, color="white", zorder=4)
    ax.set_xticks(x, [{"REF": "REF\n(reference)", "LEAK": "LEAK\n(no separation)", "SILENT": "SILENT\n(no integrity)",
                       "STALE": "STALE\n(no freshness)", "PROV": "PROV\n(no provenance)"}[c] for c in cfgs])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Success rate")
    ax.grid(axis="y", color="#e4e3df", lw=0.6, zorder=0)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2)
    fig.tight_layout()
    fig.savefig(outdir / "fig_configs.pdf")
    fig.savefig(outdir / "fig_configs.png", dpi=200)
    plt.close(fig)

    # Fig 2: per-model measured vs valid, one panel per configuration
    cfgs2 = [c for c in CONFIG_ORDER if all(c in S["per_model"][m] for m in models)]
    order = sorted(models, key=lambda m: S["per_model"][m]["REF"]["valid"])
    fig, axes = plt.subplots(1, len(cfgs2), figsize=(6.6, 2.5), sharey=True)
    for ax, c in zip(np.atleast_1d(axes), cfgs2):
        for i, m in enumerate(order):
            v = S["per_model"][m][c]
            ax.plot([v["valid"], v["measured"]], [i, i], color="#c3c2b7", lw=1.5, zorder=1)
            ax.scatter([v["measured"]], [i], s=22, color=C_MEAS, zorder=3, edgecolor="white", linewidth=1)
            ax.scatter([v["valid"]], [i], s=22, color=C_VALID, zorder=3, edgecolor="white", linewidth=1)
        ax.set_title(c, fontsize=8, color="#0b0b0b")
        ax.set_xlim(-0.05, 1.05)
        ax.set_xticks([0, 0.5, 1])
        ax.grid(axis="x", color="#e4e3df", lw=0.6, zorder=0)
    axes[0].set_yticks(range(len(order)), [SHORT[m] for m in order])
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=C_MEAS, label="Measured by pipeline"),
                        Line2D([], [], marker="o", ls="", color=C_VALID, label="Valid under reference replay")],
               frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.55, 1.02))
    fig.supxlabel("Success rate", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(outdir / "fig_models.pdf")
    fig.savefig(outdir / "fig_models.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="main")
    main(ap.parse_args().exp)
