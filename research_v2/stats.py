"""Task-cluster bootstrap, rank statistics, and influence analyses shared by the v1 and v2 analyses.

The independent sampling unit is the generated task. Every bootstrap replicate
resamples task IDs with replacement within the null/effect strata and keeps all
model x repeat x condition rows of each selected task. Intervals are percentile
intervals over B = 20,000 replicates.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

B = 20_000
SEED = 20261025


def boot_index(tasks: pd.DataFrame, b: int = B, seed: int = SEED) -> np.ndarray:
    """tasks: one row per task with columns task_id, kind. Returns (b, n_tasks) positional indices."""
    rng = np.random.default_rng(seed)
    cols = []
    for _, g in tasks.reset_index(drop=True).groupby("kind"):
        pos = g.index.to_numpy()
        cols.append(pos[rng.integers(0, len(pos), size=(b, len(pos)))])
    return np.concatenate(cols, axis=1)


def per_task(df: pd.DataFrame, tasks: pd.DataFrame, col: str) -> tuple[np.ndarray, np.ndarray]:
    g = df.groupby("task_id")[col].agg(["sum", "count"]).reindex(tasks.task_id).fillna(0)
    return g["sum"].to_numpy(float), g["count"].to_numpy(float)


def ratio_ci(s: np.ndarray, n: np.ndarray, idx: np.ndarray) -> dict:
    est = s.sum() / n.sum()
    reps = s[idx].sum(1) / n[idx].sum(1)
    lo, hi = np.percentile(reps, [2.5, 97.5])
    p_two = min(1.0, 2 * min((reps <= 0).mean(), (reps >= 0).mean()))
    return dict(est=float(est), lo=float(lo), hi=float(hi), boot_p=float(p_two), n=int(n.sum()))


def mean_ci(df: pd.DataFrame, tasks: pd.DataFrame, col: str, idx: np.ndarray) -> dict:
    d = df.dropna(subset=[col]).copy()
    d[col] = d[col].astype(float)
    return ratio_ci(*per_task(d, tasks, col), idx)


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def tau_b(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Kendall tau-b between rows of x and rows of y (shape (r, k)) with tie handling."""
    x, y = np.atleast_2d(x), np.atleast_2d(y)
    i, j = np.triu_indices(x.shape[1], 1)
    sx, sy = np.sign(x[:, i] - x[:, j]), np.sign(y[:, i] - y[:, j])
    num = (sx * sy).sum(1)
    den = np.sqrt((sx != 0).sum(1) * (sy != 0).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def model_scores(df: pd.DataFrame, tasks: pd.DataFrame, models: list[str], col: str, idx: np.ndarray):
    """Point and bootstrap per-model means of col; returns (point (k,), reps (b, k))."""
    S, N = [], []
    for m in models:
        s, n = per_task(df[df.model == m].assign(**{col: df[df.model == m][col].astype(float)}), tasks, col)
        S.append(s)
        N.append(n)
    S, N = np.array(S), np.array(N)  # (k, T)
    point = S.sum(1) / N.sum(1)
    reps = np.stack([S[:, idx].sum(2) / N[:, idx].sum(2)], 0)[0].T  # (b, k)
    return point, reps


def rank_analysis(ref_df, alt_df, tasks, models, ref_col, alt_col, idx) -> dict:
    p_ref, r_ref = model_scores(ref_df, tasks, models, ref_col, idx)
    p_alt, r_alt = model_scores(alt_df, tasks, models, alt_col, idx)
    tau_point = float(tau_b(p_ref, p_alt)[0])
    taus = tau_b(r_ref, r_alt)
    pairs = {}
    for a, b in itertools.combinations(range(len(models)), 2):
        pairs[f"{models[a]}>{models[b]}"] = dict(
            ref=float((r_ref[:, a] > r_ref[:, b]).mean()), alt=float((r_alt[:, a] > r_alt[:, b]).mean()),
            ref_point=float(np.sign(p_ref[a] - p_ref[b])), alt_point=float(np.sign(p_alt[a] - p_alt[b])))
    flips = [k for k, v in pairs.items() if v["ref_point"] * v["alt_point"] < 0]
    return dict(tau_b=tau_point, tau_b_lo=float(np.nanpercentile(taus, 2.5)),
                tau_b_hi=float(np.nanpercentile(taus, 97.5)), tau_b_median=float(np.nanmedian(taus)),
                p_identical_order=float((tau_b(r_ref, r_alt) == 1).mean()),
                point_ref=dict(zip(models, map(float, p_ref))), point_alt=dict(zip(models, map(float, p_alt))),
                pairwise=pairs, flipped_pairs=flips)


def leave_one_out(df: pd.DataFrame, by: str, fn) -> pd.DataFrame:
    rows = []
    for v in sorted(df[by].unique()):
        rows.append(dict(left_out=v, **fn(df[df[by] != v])))
    return pd.DataFrame(rows)


def signflip_p(s: np.ndarray, n: np.ndarray, b: int = B, seed: int = SEED) -> float:
    """Two-sided task-level paired sign-flip randomization test of H0: no condition effect.

    s_i is task i's sum of paired differences (condition minus reference) and n_i its number of
    pairs. Under H0 the condition labels within each task are exchangeable, so each task's summed
    difference is symmetric about zero; flipping whole-task signs respects within-task dependence.
    Returns (1 + #{|T*| >= |T|}) / (b + 1), with T = sum(s) / sum(n).
    """
    rng = np.random.default_rng(seed + 1)
    t = abs(s.sum() / n.sum())
    eps = rng.choice([-1.0, 1.0], size=(b, len(s)))
    tstar = np.abs(eps @ s) / n.sum()
    return float((1 + (tstar >= t - 1e-12).sum()) / (b + 1))
