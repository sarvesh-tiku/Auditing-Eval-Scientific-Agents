"""Paper figures in a conventional ML-paper style.

  python -m scieval.figures --exp main
Reads results/<exp>/summary.json and writes fig_main.pdf, fig_rank.pdf.
"""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from .analyze import CONFIG_ORDER, SHORT
from .pipeline import ROOT

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

TAB = plt.get_cmap("tab10").colors
MODEL_COLOR = {  # fixed per model across all figures
    "openai/gpt-6.1-sol": TAB[0], "anthropic/claude-sonnet-5.5": TAB[1], "google/gemini-4-argon": TAB[2],
    "anthropic/claude-haiku-5.5": TAB[3], "openai/gpt-6-luna": TAB[4], "deepseek/deepseek-v4.1-flash": TAB[5],
    "qwen/qwen3.8-flash": TAB[6],
}
MODEL_MARK = {"openai/gpt-6.1-sol": "o", "anthropic/claude-sonnet-5.5": "s", "google/gemini-4-argon": "^",
              "anthropic/claude-haiku-5.5": "D", "openai/gpt-6-luna": "v", "deepseek/deepseek-v4.1-flash": "P",
              "qwen/qwen3.8-flash": "X"}
CFG_LABEL = {"REF": "Ref", "LEAK": "Leak", "SILENT": "Silent", "STALE": "Stale", "PROV": "Prov"}


def fig_main(S, out):
    P = json.loads((out / "paired_full.json").read_text())
    F = S["fidelity"]
    models = [m for m in MODEL_COLOR if m in S["per_model"]]
    cfgs = ["LEAK", "SILENT", "STALE", "PROV"]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.45), gridspec_kw=dict(width_ratios=[1.0, 1.15, 1.0], wspace=0.42))

    def err(ax, x, d, **kw):
        v, lo, hi = 100 * d["est"], 100 * d["lo"], 100 * d["hi"]
        ax.errorbar(x, v, yerr=[[v - lo], [hi - v]], ms=4.5, capsize=2.2, elinewidth=1, lw=0, **kw)

    # (a) paired change vs. Ref, with each pipeline's aggregate fidelity score
    ax = axes[0]
    for i, c in enumerate(cfgs):
        err(ax, i - 0.13, P[c]["all_measured"], fmt="o", color=TAB[0], mec=TAB[0], mfc=TAB[0], mew=1.1)
        err(ax, i + 0.13, P[c]["all_valid"], fmt="s", color=TAB[3], mec=TAB[3], mfc="white", mew=1.1)
    ax.axhline(0, color="0.55", lw=0.8, ls="--", zorder=0)
    ax.set_xticks(range(len(cfgs)), [f"{CFG_LABEL[c]}\n({F[c]['fidelity']:.2f})" for c in cfgs], fontsize=7.5)
    ax.set_xlim(-0.5, len(cfgs) - 0.5)
    ax.set_ylabel("Change vs. Ref (pp)")
    ax.set_title("(a) Effect of each defect")

    # (b) per-model valid success across environment configurations
    ax = axes[1]
    env = ["REF", "LEAK", "SILENT", "STALE"]
    xe = np.arange(len(env))
    jit = np.linspace(-0.06, 0.06, len(models))  # small offsets so overlapping lines stay visible
    for m, dj in zip(models, jit):
        v = [S["per_model"][m][c]["valid"] * 100 for c in env]
        ax.plot(xe + dj, v, color=MODEL_COLOR[m], marker=MODEL_MARK[m], ms=4, lw=1.3, mec="white", mew=0.5)
    ax.set_xticks(xe, [CFG_LABEL[c] for c in env])
    ax.set_ylabel("Valid success (%)")
    ax.set_title("(b) Valid success by model")
    ax.set_xlim(-0.25, len(env) - 0.75)

    # (c) frontier vs. efficient models (valid change; Prov: measured change)
    ax = axes[2]
    for i, c in enumerate(cfgs):
        f = "measured" if c == "PROV" else "valid"
        err(ax, i - 0.13, P[c][f"frontier_{f}"], fmt="^", color="0.15", mec="0.15", mfc="0.15", mew=1.1)
        err(ax, i + 0.13, P[c][f"efficient_{f}"], fmt="v", color="0.15", mec="0.15", mfc="white", mew=1.1)
    ax.axhline(0, color="0.55", lw=0.8, ls="--", zorder=0)
    ax.set_xticks(range(len(cfgs)), [CFG_LABEL[c] + ("\n(meas.)" if c == "PROV" else "\n(valid)") for c in cfgs], fontsize=7.5)
    ax.set_xlim(-0.5, len(cfgs) - 0.5)
    ax.set_ylabel("Change vs. Ref (pp)")
    ax.set_title("(c) By model tier")
    lo = min(axes[0].get_ylim()[0], ax.get_ylim()[0])
    hi = max(axes[0].get_ylim()[1], ax.get_ylim()[1])
    for a_ in (axes[0], ax):
        a_.set_ylim(lo, hi)

    h = [Line2D([], [], marker="o", color=TAB[0], ls="", ms=4.5, label="Measured"),
         Line2D([], [], marker="s", color=TAB[3], ls="", ms=4.5, mfc="white", mew=1.1, label="Valid"),
         Line2D([], [], marker="^", color="0.15", ls="", ms=4.5, label="Frontier models"),
         Line2D([], [], marker="v", color="0.15", ls="", ms=4.5, mfc="white", mew=1.1, label="Efficient models")]
    h += [Line2D([], [], color=MODEL_COLOR[m], marker=MODEL_MARK[m], ms=4, lw=1.3, label=SHORT[m]) for m in models]
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=6, frameon=False,
               handlelength=1.6, columnspacing=1.1)
    fig.savefig(out / "fig_main.pdf")
    fig.savefig(out / "fig_main.png", dpi=220)
    plt.close(fig)


CFG_STYLE = {  # (color, marker, filled)
    "REF": ("black", "o", False), "LEAK": (TAB[4], "v", True), "SILENT": (TAB[1], "s", True),
    "STALE": (TAB[2], "D", True), "PROV": (TAB[0], "o", True),
}


def fig_models(S, out):
    models = [m for m in MODEL_COLOR if m in S["per_model"]]
    pm = S["per_model"]
    order = sorted(models, key=lambda m: pm[m]["REF"]["valid"])
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw=dict(width_ratios=[1.35, 1.0], wspace=0.55))

    # (a) where each model lands under each pipeline
    ax = axes[0]
    series = [("REF", "valid", "Ref (valid)"), ("LEAK", "valid", "Leak"), ("SILENT", "valid", "Silent"),
              ("STALE", "valid", "Stale"), ("PROV", "measured", "Prov (measured)")]
    offs = np.linspace(-0.2, 0.2, len(series))
    for i, m in enumerate(order):
        vals = [pm[m][c][k] * 100 for c, k, _ in series if c in pm[m]]
        ax.plot([min(vals), max(vals)], [i, i], color="0.82", lw=2.2, zorder=1, solid_capstyle="round")
    for (c, k, lab), off in zip(series, offs):
        col, mk, filled = CFG_STYLE[c]
        xs = [pm[m][c][k] * 100 for m in order]
        tau = S["ranking"].get(c, {}).get("tau_measured_vs_ref_valid" if k == "measured" else "tau_valid_vs_ref_valid")
        label = lab if c == "REF" else f"{lab}, $\\tau$={tau:.2f}"
        ax.scatter(xs, np.arange(len(order)) + off, marker=mk, s=22 if c != "REF" else 34, zorder=3 if c != "REF" else 4,
                   color=col if filled else "white", edgecolor=col, linewidth=1.0, label=label)
    ax.set_yticks(range(len(order)), [SHORT[m] for m in order])
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlabel("Success rate (%)")
    ax.set_title("(a) Per-model score under each pipeline")
    ax.legend(loc="upper center", bbox_to_anchor=(0.42, -0.2), ncol=3, frameon=False, handletextpad=0.3,
              columnspacing=0.9)

    # (b) exposure vs. harm
    ax = axes[1]
    pc = S["per_config"]
    cfgs = [c for c in ("LEAK", "SILENT", "STALE") if c in pc and "exposure" in pc[c]]
    what = {"LEAK": "saw held-out file", "SILENT": "error hidden", "STALE": "stale output"}
    y = np.arange(len(cfgs))[::-1]
    for yi, c in zip(y, cfgs):
        e = pc[c]["exposure"]
        for key, mk, fill, dy in (("valid_unexposed", "o", "white", 0.13), ("valid_exposed", "o", "0.15", -0.13)):
            v = e[key]
            if v is None:
                continue
            ci = e.get(key + "_ci")
            xerr = None if ci is None else [[100 * (v - ci[0])], [100 * (ci[1] - v)]]
            ax.errorbar(100 * v, yi + dy, xerr=xerr, fmt=mk, ms=4.5, color="0.15", mfc=fill, mec="0.15", mew=1.1,
                        elinewidth=0.9, capsize=2, zorder=3)
        ax.annotate(f"{100 * e['rate']:.0f}% of runs", (0.0, yi - 0.36), xycoords=("axes fraction", "data"),
                    xytext=(4, 0), textcoords="offset points", fontsize=7, color="0.35", va="center")
    ax.set_yticks(y, [f"{CFG_LABEL[c]}\n({what[c]})" for c in cfgs])
    ax.set_ylim(-0.6, len(cfgs) - 0.4)
    ax.set_xlabel("Valid success (%)")
    ax.set_title("(b) Exposure to the defect")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color="0.15", mfc="white", mew=1.1, label="Not exposed"),
                       Line2D([], [], marker="o", ls="", color="0.15", label="Exposed")],
              loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
    fig.savefig(out / "fig_models.pdf")
    fig.savefig(out / "fig_models.png", dpi=220)
    plt.close(fig)


def fig_rank(S, out):
    models = [m for m in MODEL_COLOR if m in S["per_model"]]
    cols = [("REF", "valid", "Ref (valid)"), ("LEAK", "measured", "Leak"), ("SILENT", "measured", "Silent"),
            ("STALE", "measured", "Stale"), ("PROV", "measured", "Prov")]
    cols = [c for c in cols if all(c[0] in S["per_model"][m] for m in models)]
    fig, ax = plt.subplots(figsize=(3.6, 2.5))
    ranks = {}
    for j, (c, key, _) in enumerate(cols):
        scores = np.array([S["per_model"][m][c][key] for m in models])
        order = (-scores).argsort(kind="stable")
        r = np.empty(len(models))
        r[order] = np.arange(1, len(models) + 1)
        for v in np.unique(scores):  # average ranks for ties
            idx = scores == v
            r[idx] = r[idx].mean()
        ranks[j] = r
    for i, m in enumerate(models):
        y = [ranks[j][i] for j in range(len(cols))]
        ax.plot(range(len(cols)), y, color=MODEL_COLOR[m], marker=MODEL_MARK[m], ms=4.5, lw=1.4, mec="white", mew=0.5)
    # end labels outside the frame; spread tied ranks so labels never overlap
    ends = sorted(((ranks[len(cols) - 1][i], m) for i, m in enumerate(models)), key=lambda t: t[0])
    gap = 0.62
    lab_y = []
    for k, (y, _) in enumerate(ends):
        lab_y.append(max(y, lab_y[-1] + gap) if lab_y else y)
    shift = max(0.0, lab_y[-1] - (len(models) + 0.3))
    lab_y = [v - shift for v in lab_y]
    import matplotlib.transforms as mtransforms
    LABEL_GAP_PT = 11  # space between the right spine and the labels
    tr = mtransforms.blended_transform_factory(ax.transAxes, ax.transData) + \
        mtransforms.ScaledTranslation(LABEL_GAP_PT / 72, 0, fig.dpi_scale_trans)
    for (y, m), ly in zip(ends, lab_y):
        ax.text(1.0, ly, SHORT[m], transform=tr, va="center", ha="left", fontsize=7.5,
                color=MODEL_COLOR[m], clip_on=False)
    ax.set_xticks(range(len(cols)), [c[2] for c in cols])
    ax.set_yticks(range(1, len(models) + 1))
    ax.set_ylim(len(models) + 0.5, 0.5)
    ax.set_xlim(-0.3, len(cols) - 0.7)
    ax.set_ylabel("Rank (1 = best)")
    ax.set_title("Model rank under each pipeline")
    fig.savefig(out / "fig_rank.pdf")
    fig.savefig(out / "fig_rank.png", dpi=220)
    plt.close(fig)


def main(exp):
    out = ROOT / "results" / exp
    S = json.loads((out / "summary.json").read_text())
    fig_main(S, out)
    fig_models(S, out)
    fig_rank(S, out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="main")
    main(ap.parse_args().exp)
