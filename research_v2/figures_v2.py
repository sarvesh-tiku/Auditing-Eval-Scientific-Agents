"""Main quantitative figure, generated from frozen result files only.

    python -m research_v2.figures_v2     # writes paper/figures/fig_effects.pdf
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent / "paper" / "figures"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Open Sans", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 7.6, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none",
})
GRADE, ENV = "#CC79A7", "#D55E00"
REL = {"contradicts_evidence": ("contradicts own test", "#D55E00"),
       "consistent_label_mismatch": ("consistent; label off-format", "#56B4E9"),
       "failed_or_missing_confirm": ("script failed", "#999999")}
SHORT = {"anthropic/claude-haiku-5.5": "Haiku 5.5", "anthropic/claude-sonnet-5.5": "Sonnet 5.5",
         "deepseek/deepseek-v4.1-flash": "DeepSeek-V4.1-fl.", "google/gemini-4-argon": "Gemini 4 Argon",
         "openai/gpt-6-luna": "GPT-6-luna", "openai/gpt-6.1-sol": "GPT-6.1-sol", "qwen/qwen3.8-flash": "Qwen3.8-flash"}


def main():
    s1 = json.loads((HERE / "results" / "v1" / "summary_v1.json").read_text())
    s2 = json.loads((HERE / "results" / "v2" / "summary_v2.json").read_text())
    nc = pd.read_csv(HERE / "results" / "v1" / "prov_newly_credited.csv")
    try:
        from matplotlib import font_manager
        for f in (HERE.parent / "paper" / "fonts").glob("*.ttf"):
            font_manager.fontManager.addfont(str(f))
    except Exception:
        pass

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.3), gridspec_kw=dict(width_ratios=[1.2, 1], wspace=0.55))

    # (a) forest plot of paired effects
    e1, e2 = s1["estimands"], s2["estimands"]
    rows = [("v1: replay-graded success", None, None, None),
            ("Provenance (regrade)", e1["PROV_same_trace"], GRADE, "D"),
            ("Separation", e1["LEAK_fresh_valid"], ENV, "o"),
            ("Integrity", e1["SILENT_fresh_valid"], ENV, "o"),
            ("Freshness", e1["STALE_fresh_valid"], ENV, "o"),
            ("v2: joint valid success", None, None, None),
            ("Provenance (regrade)", e2["PROV_GRADE_same_trace"], GRADE, "D"),
            ("Separation", e2["SEP_READ_fresh_joint_valid"], ENV, "o"),
            ("Integrity", e2["INT_MASK_fresh_joint_valid"], ENV, "o"),
            ("Freshness", e2["FRESH_CACHE_fresh_joint_valid"], ENV, "o")]
    y = list(range(len(rows)))[::-1]
    labels = []
    for yy, (lab, e, col, mk) in zip(y, rows):
        if e is None:
            labels.append(lab)
            continue
        labels.append(lab)
        ax1.plot([100 * e["lo"], 100 * e["hi"]], [yy, yy], color=col, lw=1.4, solid_capstyle="round")
        ax1.plot(100 * e["est"], yy, marker=mk, color=col, ms=4.6, mec="white", mew=0.6, zorder=3)
    ax1.axvline(0, color="black", lw=0.6)
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels)
    ax1.tick_params(axis="y", length=0, pad=2)
    for t, (lab, e, *_r) in zip(ax1.get_yticklabels(), rows):
        if e is None:
            t.set_fontweight("bold")
    ax1.spines["left"].set_visible(False)
    ax1.set_xlim(-12, 10)
    ax1.set_xlabel("Paired change vs. reference (pp)")
    ax1.set_title("(a) Effect of each fault", loc="left", x=-0.62, fontweight="bold")

    # (b) composition of the 19 newly credited original REF traces by model
    order = nc.model.value_counts().index.tolist()
    left = pd.Series(0.0, index=order)
    for rel, (lab, col) in REL.items():
        vals = nc[nc.relation == rel].model.value_counts().reindex(order).fillna(0)
        ax2.barh([SHORT[m] for m in order], vals.values, left=left.values, color=col, label=lab,
                 height=0.62, edgecolor="white", linewidth=0.6)
        left += vals
    ax2.invert_yaxis()
    ax2.set_xlabel("Number of reference traces")
    ax2.set_title("(b) v1 traces credited only without provenance", loc="left", x=-0.45, fontweight="bold")
    ax2.xaxis.get_major_locator().set_params(integer=True)
    ax2.set_xlim(0, 14.5); ax2.set_xticks([0, 2, 4, 6, 8, 10])
    ax2.legend(frameon=False, loc="lower right", handlelength=1.0, borderaxespad=0.1)
    ax2.tick_params(axis="y", length=0)
    fig.savefig(PAPER / "fig_effects.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(HERE / "figures" / "fig_effects.png", dpi=220, bbox_inches="tight", pad_inches=0.02)


if __name__ == "__main__":
    main()
