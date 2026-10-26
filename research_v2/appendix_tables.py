"""Supplementary tables and figures for the technical appendix, generated from frozen records.

    python -m research_v2.appendix_tables     # writes paper/figures/app_*.tex, app_*.pdf
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import matplotlib
matplotlib.rcParams["hatch.color"] = "#ffffff"
matplotlib.rcParams["hatch.linewidth"] = 0.8
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scieval.grade import cited_confirm
from .figures_v2 import SHORT

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent / "paper" / "figures"
RES = HERE / "results"
CACHE = Path(os.environ.get("SCIEVAL_CACHE", Path.home() / ".cache" / "scieval"))
CORRECTION = re.compile(r"bonferroni|holm|multipletests|fdr|\*\s*len\(|\*\s*10\b|0\.05\s*/\s*(len|10)|alpha\s*/", re.I)
MODEL_ORDER = ["openai/gpt-6.1-sol", "anthropic/claude-sonnet-5.5", "google/gemini-4-argon",
               "anthropic/claude-haiku-5.5", "openai/gpt-6-luna", "deepseek/deepseek-v4.1-flash", "qwen/qwen3.8-flash"]


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


def write(name, body):
    (PAPER / name).write_text(body)


def taxonomy(df, audit):
    def cat(x):
        if not x.has_confirm:
            return None
        if x.n_fits == 0 or x.n_tested_exposures == 0:
            return "ambiguous"
        if x.n_tested_exposures == 1:
            return "single"
        return "separate" if x.multi_fit_distinct else "multivariable"

    df = df.copy()
    df["cat"] = df.apply(cat, axis=1)
    hc = df[df.cat.notna()].copy()
    argmin, corr = [], []
    for _, x in hc.iterrows():
        a = audit[(x.model, x.config, x.task_id, x["repeat"])]
        ps = {}
        for f in a.get("fit_log") or []:
            for g in f.get("regressors", []):
                if g["exposure"] not in (None, "?") and g["p"] is not None:
                    ps[g["exposure"]] = min(ps.get(g["exposure"], 1.0), g["p"])
        argmin.append(len(ps) > 1 and x.decision == "discover" and x.answer == min(ps, key=ps.get))
        rec = json.loads((CACHE / "runs" / "main" / x.model.replace("/", "__") / x.config /
                          f"{x.task_id}_r{x["repeat"]}.json").read_text())
        ev = cited_confirm(rec["events"], rec["submission"])
        corr.append(bool(CORRECTION.search(ev["code"])) if ev else False)
    hc["minp_selected"] = argmin
    hc["correction_text"] = corr
    names = {"single": "Single candidate", "multivariable": "Several candidates, one model structure",
             "separate": "Separate per-candidate fits", "ambiguous": "No identifiable fit"}
    rows, out = [], {}
    for c in ["single", "multivariable", "separate", "ambiguous"]:
        g = hc[hc.cat == c]
        disc = g[g.decision == "discover"]
        nd = disc[disc.kind == "null"]
        out[c] = dict(n=len(g), share=len(g) / len(hc), discover=len(disc), null_disc=len(nd),
                      minp_selected=int(g.minp_selected.sum()), correction=int(g.correction_text.sum()))
        rows.append(f"{names[c]} & {len(g)} & {pct(len(g) / len(hc))} & {len(disc)} & {len(nd)} & "
                    f"{int(g.minp_selected.sum())} & {int(g.correction_text.sum())}\\\\")
    write("app_taxonomy.tex", r"""\begin{table}[h]
\centering
\caption{\textbf{v1 confirmation scripts by inferential structure} (all 1,665 cited confirmation
scripts across \cfg{Ref}, \cfg{Leak}, \cfg{Silent}, \cfg{Stale}; unit: run). \emph{Null disc.}: discoveries
on null tasks (all false). \emph{Min-$p$}: discoveries whose reported exposure is the smallest-$p$
candidate among those fitted (only defined with $\ge 2$ candidates). \emph{Correction}: script text
contains a multiplicity-correction pattern (Bonferroni, Holm, \texttt{multipletests}, FDR, $\alpha/m$);
a lexical indicator, not a verified procedure.}
\label{tab:taxonomy}
\footnotesize
\begin{tabular}{lrrrrrr}
\toprule
Structure & Scripts & \% & Discoveries & Null disc. & Min-$p$ & Correction\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    by_model = pd.crosstab(hc.model, hc.cat).reindex(MODEL_ORDER)
    rows = []
    for m, r in by_model.iterrows():
        n = r.sum()
        rows.append(f"{SHORT[m]} & {n} & " + " & ".join(pct(r.get(c, 0) / n, 0) for c in
                                                        ["single", "multivariable", "separate", "ambiguous"]) + r"\\")
    write("app_taxonomy_models.tex", r"""\begin{table}[h]
\centering
\caption{\textbf{Share of v1 confirmation scripts (\%) in each structure, by model} (unit: run;
denominator: runs with a cited confirmation, all four v1 configurations).}
\label{tab:taxmodels}
\small
\begin{tabular}{lrrrrr}
\toprule
Model & Scripts & Single & Multivariable & Separate fits & No fit\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    return out


def dominant_tasks(df):
    rows = []
    for t in ["ecology-null-1011", "clinical-null-1009"]:
        g = df[(df.task_id == t) & (df.decision == "discover")]
        for m in MODEL_ORDER:
            gm = g[g.model == m]
            tot = int((df.task_id == t).sum() / 7)
            named = ", ".join(f"{e.replace('_', chr(92) + '_')} ({k})" for e, k in gm.answer.value_counts().items()) or "---"
            rows.append(f"\\texttt{{{t}}} & {SHORT[m]} & {len(gm)}/{tot} & {named}\\\\")
        rows.append(r"\midrule")
    write("app_dominant.tex", r"""\begin{table}[h]
\centering
\caption{\textbf{Discoveries on the two dominant v1 null tasks, by model} (unit: run; 12 runs per
model and task = 4 configurations $\times$ 3 repeats). Every run on a task shares one confirmation split.}
\label{tab:dominant}
\small
\begin{tabular}{llrl}
\toprule
Task & Model & Discoveries & Exposures named (count)\\
\midrule
""" + "\n".join(rows[:-1]) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


def power_table():
    d = pd.read_csv(RES / "dgp_validation.csv")
    eff = d[d.kind == "effect"]
    edges = np.linspace(0.15, 0.25, 9)
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        g = eff[(eff.beta >= a) & ((eff.beta < b) | (b == edges[-1]))]
        rows.append(f"$[{a:.4f}, {b:.4f})$ & {len(g)} & {pct((g.trusted == g.truth).mean())} & "
                    f"{pct(g.trusted_selected_truth.mean())} & {pct((g.naive == g.truth).mean())} & "
                    f"{pct((g.all10 == g.truth).mean())}\\\\")
    write("app_power.tex", r"""\begin{table}[h]
\centering
\caption{\textbf{Detection by effect size} (Monte Carlo; 5,000 independent effect-task pairs; unit:
simulated dataset pair; $n=400$ per split). \emph{Reference}: correct discovery by the reference agent;
\emph{Selected truth}: its exploration screen locks the true exposure; \emph{Naive}: unadjusted screen,
adjusted confirmation; \emph{All ten}: minimum-$p$ over ten confirmation tests.}
\label{tab:power}
\small
\begin{tabular}{lrrrrr}
\toprule
$\beta$ bin & Pairs & Reference (\%) & Selected truth (\%) & Naive (\%) & All ten (\%)\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


def v2_models():
    d = pd.read_csv(RES / "v2" / "v2_per_run.csv")
    d = d[d.model != "reference"]
    rows = []
    for m in ["openai/gpt-6.1-sol", "anthropic/claude-haiku-5.5", "openai/gpt-6-luna", "deepseek/deepseek-v4.1-flash"]:
        g = d[d.model == m]
        cells = [pct(g[g.condition == c].joint_valid_success.mean()) for c in ["REF", "SEP_READ", "INT_MASK", "FRESH_CACHE"]]
        r = g[g.condition == "REF"]
        rows.append(f"{SHORT[m]} & " + " & ".join(cells) + f" & {pct(r.credited_PROV_GRADE.mean())} & "
                    f"{pct(g.protocol_valid.mean())} & {g.calls.mean():.1f} & {g.cost.mean():.4f}\\\\")
    write("app_v2_models.tex", r"""\begin{table}[h]
\centering
\caption{\textbf{v2 joint valid success (\%) by model and condition} (unit: run; 40 tasks per cell, one
repeat). \emph{Prov.}: answer-only credit on the same \cfg{Ref} traces. \emph{Protocol}: share of all 160
runs of the model with a valid protocol. \emph{Calls}: mean model calls per run. \emph{\$/run}: mean
recorded OpenRouter cost.}
\label{tab:v2models}
\small
\begin{tabular}{lrrrrrrrr}
\toprule
Model & \cfg{Ref} & \cfg{Sep\_read} & \cfg{Int\_mask} & \cfg{Fresh\_cache} & Prov. & Protocol & Calls & \$/run\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


def loto_table():
    lt = pd.read_csv(RES / "v1" / "task_influence.csv")
    s1 = json.loads((RES / "v1" / "summary_v1.json").read_text())
    full = {"PROV": s1["estimands"]["PROV_same_trace"]["est"],
            **{c: s1["estimands"][f"{c}_fresh_valid"]["est"] for c in ["LEAK", "SILENT", "STALE"]}}
    rows = []
    for c in ["PROV", "LEAK", "SILENT", "STALE"]:
        dev = (lt[c] - full[c]).abs()
        top = lt.loc[dev.sort_values(ascending=False).index[:2]]
        tops = "; ".join(f"\\texttt{{{r.left_out}}} ({100 * r[c]:+.1f})" for _, r in top.iterrows())
        rows.append(f"\\cfg{{{c.title()}}} & ${100 * full[c]:+.1f}$ & $[{100 * lt[c].min():+.1f}, {100 * lt[c].max():+.1f}]$ & {tops}\\\\")
    write("app_loto.tex", r"""\begin{table}[h]
\centering
\caption{\textbf{v1 leave-one-task-out sensitivity} (pp; unit: task; 20 refits per estimand). \emph{Most
influential}: the two tasks whose removal moves the estimate most, with the estimate after removal.}
\label{tab:loto}
\small
\begin{tabular}{lrrl}
\toprule
Estimand & Full & LOTO range & Most influential tasks (estimate without)\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


def lomo_table():
    lm = pd.read_csv(RES / "v1" / "model_influence.csv")
    rows = []
    for _, r in lm.set_index("left_out").reindex(MODEL_ORDER).reset_index().iterrows():
        rows.append(f"{SHORT[r.left_out]} & " + " & ".join(f"${100 * r[c]:+.1f}$" for c in ["PROV", "LEAK", "SILENT", "STALE"]) + r"\\")
    write("app_lomo.tex", r"""\begin{table}[h]
\centering
\caption{\textbf{v1 leave-one-model-out estimates} (pp; each row omits one model and recomputes the
paired estimand over the remaining six).}
\label{tab:lomo}
\small
\begin{tabular}{lrrrr}
\toprule
Model left out & \cfg{Prov} & \cfg{Leak} & \cfg{Silent} & \cfg{Stale}\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


def fig_manifest():
    s1 = json.loads((RES / "v1" / "summary_v1.json").read_text())
    s2 = json.loads((RES / "v2" / "summary_v2.json").read_text())
    rows = []
    for r in s1["fault_manifestation"]:
        rows.append((f"{r['config']} (v1)", r["runs"], r["triggered"], r["visible"], r["used"]))
    for r in s2["fault_manifestation"]:
        used = r.get("used", r.get("last_pre_lock_output_faulty"))
        rows.append((f"{r['condition']} (v2)", r["runs"], r["triggered"], r["visible"], used))
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    y = np.arange(len(rows))[::-1]
    cols = ["#C8C8C8", "#56B4E9", "#0072B2", "#D55E00"]
    labs = ["scheduled", "triggered", "visible to agent", "used (operational)"]
    for k in range(4):
        ax.barh(y + (1.5 - k) * 0.19, [100 * r[k + 1] / r[1] for r in rows], height=0.18, color=cols[k], label=labs[k])
    for yy, r in zip(y, rows):
        ax.text(101, yy, f"n={r[1]}", va="center", fontsize=6.5, color="#555555")
    ax.set_yticks(y)
    ax.set_yticklabels([r[0].replace("_", r"\_") if False else r[0] for r in rows])
    ax.set_xlim(0, 112)
    ax.set_xlabel("Share of faulted runs (%)")
    ax.legend(frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.45, 1.0), fontsize=6.6)
    fig.savefig(PAPER / "app_manifest.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def fig_pairwise():
    s1 = json.loads((RES / "v1" / "summary_v1.json").read_text())
    pw = s1["rankings"]["PROV"]["pairwise"]
    models = sorted(s1["rankings"]["PROV"]["point_ref"], key=lambda m: -s1["rankings"]["PROV"]["point_ref"][m])
    k = len(models)
    fig, axes = plt.subplots(2, 1, figsize=(4.4, 7.0), gridspec_kw=dict(hspace=0.12))
    short = {"anthropic/claude-haiku-5.5": "Haiku", "anthropic/claude-sonnet-5.5": "Sonnet", "deepseek/deepseek-v4.1-flash": "DeepSeek", "google/gemini-4-argon": "Gemini", "openai/gpt-6-luna": "Luna", "openai/gpt-6.1-sol": "Sol", "qwen/qwen3.8-flash": "Qwen"}
    for ax, which, title in zip(axes, ["ref", "alt"], ["Reference grader (replay-graded success)", "Answer-only grader (\\textsc{prov})"]):
        M = np.full((k, k), np.nan)
        for i, a in enumerate(models):
            for j, b in enumerate(models):
                if i == j:
                    continue
                if f"{a}>{b}" in pw:
                    M[i, j] = pw[f"{a}>{b}"][which]
                else:
                    M[i, j] = 1 - pw[f"{b}>{a}"][which] if f"{b}>{a}" in pw else np.nan
        im = ax.imshow(M, vmin=0, vmax=1, cmap="RdBu_r")
        for i in range(k):
            for j in range(k):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                            color="white" if abs(M[i, j] - 0.5) > 0.35 else "black")
        ax.set_xticks(range(k))
        ax.set_yticks(range(k))
        ax.set_xticklabels([short[m] for m in models] if which == "alt" else [], rotation=35, ha="right", fontsize=8)
        ax.set_yticklabels([short[m] for m in models], fontsize=8)
        ax.set_title(title.replace("\\textsc{prov}", "PROV"), fontsize=8.5, fontweight="bold")
        for s in ax.spines.values():
            s.set_visible(False)
    cb = fig.colorbar(im, ax=axes, fraction=0.04, pad=0.03)
    cb.ax.tick_params(labelsize=7)
    cb.set_label("P(row model scores strictly higher)", fontsize=8)
    fig.savefig(PAPER / "app_pairwise.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main():
    from matplotlib import font_manager
    for f in (HERE.parent / "paper" / "fonts").glob("*.ttf"):
        font_manager.fontManager.addfont(str(f))
    df = pd.read_csv(RES / "v1_per_run.csv")
    df["kind"] = df.task_id.str.split("-").str[1]
    audit = {(r["model"], r["config"], r["task_id"], r["repeat"]): r
             for r in map(json.loads, open(HERE / "frozen" / "confirm_audit.jsonl"))}
    out = dict(taxonomy=taxonomy(df, audit))
    dominant_tasks(df)
    power_table()
    v2_models()
    loto_table()
    lomo_table()
    fig_manifest()
    fig_pairwise()
    fig_tec()
    (RES / "appendix_extras.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))



def fig_tec():
    """Truth x evidence cells for reference traces, with credit under the two graders."""
    v1 = pd.read_csv(RES / "v1_per_run.csv")
    v1 = v1[v1.config == "REF"].copy()
    v1["T"] = v1.truth_correct.map({True: "correct", False: "incorrect"}).fillna(
        v1.decision.map({"inconclusive": "abstain", "missing": "no submission"}))
    v1["E"] = v1.independent_evidence.map({True: "supported", False: "unsupported"}).fillna("undetermined")
    v1 = v1.rename(columns={"measured_REF": "cr", "measured_PROV": "cp"})
    v2 = pd.read_csv(RES / "v2" / "v2_per_run.csv")
    v2 = v2[(v2.model != "reference") & (v2.condition == "REF")].copy()
    v2["T"] = v2.truth_correct.map({True: "correct", False: "incorrect"}).fillna("abstain")
    v2["E"] = v2.independent_evidence_supported.map({True: "supported", False: "unsupported"}).fillna("undetermined")
    v2 = v2.rename(columns={"credited_REF": "cr", "credited_PROV_GRADE": "cp"})
    rows = ["correct", "incorrect", "abstain", "no submission"]
    cols = ["supported", "unsupported", "undetermined"]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7))
    for ax, d, title in zip(axes, [v1, v2], [f"v1 reference traces (n={len(v1)})", f"v2 reference runs (n={len(v2)})"]):
        tot = len(d)
        for i, t in enumerate(rows):
            for j, e in enumerate(cols):
                g = d[(d["T"] == t) & (d["E"] == e)]
                n = len(g)
                shade = 0.08 + 0.7 * (n / tot) ** 0.5 if n else 0
                ax.add_patch(plt.Rectangle((j, -i), 1, 1, facecolor=(0.0, 0.447, 0.698, shade) if n else "#FAFAFA",
                                           edgecolor="white", lw=1.5,
                                           hatch="////" if (n and e == "undetermined") else None))
                if n:
                    ax.text(j + 0.5, -i + 0.62, f"{n}", ha="center", va="center", fontsize=9.5, fontweight="bold",
                            color="white" if shade > 0.45 else "black")
                    ax.text(j + 0.5, -i + 0.25, f"{int(g.cr.sum())} / {int(g.cp.sum())}", ha="center",
                            va="center", fontsize=7, color="white" if shade > 0.45 else "#333333")
        ax.set_xlim(0, 3)
        ax.set_ylim(-3, 1)
        ax.set_xticks([0.5, 1.5, 2.5])
        ax.set_xticklabels(["supported", "unsupported", "undetermined\n(hatched)"], fontsize=7.5)
        ax.set_yticks([0.5, -0.5, -1.5, -2.5])
        ax.set_yticklabels(["true", "false", "abstain", "no submission"], fontsize=7.5)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title, fontsize=8.5, fontweight="bold"); ax.set_xlabel("oracle evidence", fontsize=7.5)
    fig.tight_layout(w_pad=2.5)
    fig.savefig(PAPER / "app_tec.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    main()
