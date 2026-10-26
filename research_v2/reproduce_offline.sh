#!/usr/bin/env bash
# Regenerate every statistic, table, and figure from frozen records. No API keys, no network, no model calls.
#   bash research_v2/reproduce_offline.sh            (from the repository root)
set -euo pipefail
PY=${PY:-.venv/bin/python}
export SCIEVAL_CACHE=${SCIEVAL_CACHE:-$HOME/.cache/scieval}
# 1. Restore frozen run records and task suites if the cache is empty.
if [ ! -d "$SCIEVAL_CACHE/runs/main" ]; then mkdir -p "$SCIEVAL_CACHE/runs" && tar -xzf research_v2/frozen/runs_v1_main.tar.gz -C "$SCIEVAL_CACHE/runs"; fi
if [ ! -d "$SCIEVAL_CACHE/suites/v2_40" ]; then mkdir -p "$SCIEVAL_CACHE/suites" && tar -xzf research_v2/frozen/suites.tar.gz -C "$SCIEVAL_CACHE/suites"; fi
# 2. Tests for the oracle, the confirmation protocol, the validator, and fault isolation.
$PY -m pytest research_v2/tests -q
# 3. Generator / protocol validation (10,000 simulated dataset pairs).
$PY -m research_v2.dgp_validation --n 5000 > /dev/null
# 4. Original experiment: optional re-execution of all cited confirm scripts (slow, macOS sandbox);
#    the frozen output research_v2/frozen/confirm_audit.jsonl is used otherwise.
if [ "${REAUDIT:-0}" = "1" ]; then $PY -m research_v2.audit_existing; fi
$PY -m research_v2.decompose_v1 > /dev/null
$PY -m research_v2.analysis_v1 > /dev/null
# 5. v2 experiment (frozen run records in research_v2/frozen/runs_v2/v2main).
$PY -m research_v2.analysis_v2 > /dev/null
# 6. Figures and tables, then the paper.
$PY -m research_v2.figures_v2
$PY -m research_v2.tables
$PY -m research_v2.appendix_tables > /dev/null
(cd paper/figures/src && xelatex -interaction=nonstopmode -halt-on-error pipeline.tex > /dev/null && cp pipeline.pdf .. && pdftocairo -svg pipeline.pdf ../pipeline.svg)
(cd paper && latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex > /dev/null)
echo "done: paper/main.pdf"
