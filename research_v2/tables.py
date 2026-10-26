"""LaTeX tables for the paper, generated from frozen summaries only.

    python -m research_v2.tables      # writes paper/figures/table_*.tex
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent / "paper" / "figures"


def ci(e, holm=False):
    s = f"${100 * e['est']:+.1f}$ & $[{100 * e['lo']:+.1f}, {100 * e['hi']:+.1f}]$"
    return s


def pct(x):
    return f"{100 * x:.0f}\\%"


def main():
    s1 = json.loads((HERE / "results" / "v1" / "summary_v1.json").read_text())
    s2 = json.loads((HERE / "results" / "v2" / "summary_v2.json").read_text())
    e1, e2 = s1["estimands"], s2["estimands"]
    fm1 = {r["config"]: r for r in s1["fault_manifestation"]}
    fm2 = {r["condition"]: r for r in s2["fault_manifestation"]}
    lomo1, lomo2 = s1["leave_one_model_out"], s2["lomo"]

    def rng(a, b):
        return f"$[{100 * a:+.1f}, {100 * b:+.1f}]$"

    rows = []
    rows.append(r"\multicolumn{6}{l}{\textit{Original protocol: agent-authored confirm script; 20 tasks, 7 models, 3 repeats}}\\")
    rows.append(r"\cfg{Prov} & same-trace regrade & " + ci(e1["PROV_same_trace"]) + r" & --- & " +
                rng(lomo1["PROV"]["min"], lomo1["PROV"]["max"]) + r"\\")
    names1 = {"LEAK": "\\cfg{Leak}", "SILENT": "\\cfg{Silent}", "STALE": "\\cfg{Stale}"}
    for c in ("LEAK", "SILENT", "STALE"):
        f = fm1[c]
        rows.append(f"{names1[c]} & fresh runs & " + ci(e1[f"{c}_fresh_valid"]) +
                    f" & {pct(f['visible'] / f['runs'])} & " + rng(lomo1[c]["min"], lomo1[c]["max"]) + r"\\")
    rows.append(r"\midrule")
    rows.append(r"\multicolumn{6}{l}{\textit{Trusted one-shot confirmation: " + f"{s2['n_tasks']}" +
                r" new tasks, 4 models, 1 repeat}}\\")
    rows.append(r"\cfg{Prov\_grade} & same-trace regrade & " + ci(e2["PROV_GRADE_same_trace"]) + r" & --- & " +
                rng(*lomo2["PROV"]) + r"\\")
    names2 = {"SEP_READ": "\\cfg{Sep\\_read}", "INT_MASK": "\\cfg{Int\\_mask}", "FRESH_CACHE": "\\cfg{Fresh\\_cache}"}
    for c in ("SEP_READ", "INT_MASK", "FRESH_CACHE"):
        f = fm2[c]
        rows.append(f"{names2[c]} & fresh runs & " + ci(e2[f"{c}_fresh_joint_valid"]) +
                    f" & {pct(f['visible'] / f['runs'])} & " + rng(*lomo2[c]) + r"\\")
    tex = (r"""\begin{table}[t]
\centering
\caption{Paired change vs.\ the reference pipeline (pp; 95\% task-cluster bootstrap CI,
$B=20{,}000$). Regrade rows change only credit on identical traces; fresh rows change replay-graded success (v1) or joint
valid success (v2), which are different constructs. Fault $p$-values: Holm-adjusted task-level sign-flip tests
(v1: 0.023, 0.006, 0.006; v2: all $\ge 0.56$ raw, 1.0 adjusted).
\emph{Visible}: share of faulted runs in which a faulty observation reached the agent.
\emph{LOMO}: range when each model is left out.}
\label{tab:main}
\small
\begin{tabular}{llrrrr}
\toprule
Fault & Comparison & $\Delta$ & 95\% CI & Visible & LOMO range\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    (PAPER / "table_main.tex").write_text(tex)

    # v1 decomposition of REF traces: truth x independent evidence, and agent evidence
    m = s1["ref_matrix_by_decision"]
    print(json.dumps(m, indent=1))


if __name__ == "__main__":
    main()
