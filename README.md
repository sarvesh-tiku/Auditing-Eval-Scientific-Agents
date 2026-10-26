# When the Pipeline Is Wrong: Auditing the Evaluation of Scientific Agents

Sarvesh Tiku, Gauri Sharma (Georgia Institute of Technology). Agent Evaluation Science Symposium, Fall 2026.

A controlled audit of how defects in scientific-agent evaluation pipelines change measured success.
Agents test hypotheses on synthetic datasets with known ground truth inside a reference pipeline that
enforces four rules (data separation, execution integrity, result freshness, evidence provenance).
Each defective configuration disables one rule while tasks and data stay fixed.

## Layout

| Path | Contents |
|---|---|
| `scieval/tasks.py` | Synthetic hypothesis-testing tasks with known generating processes |
| `scieval/pipeline.py` | Sandboxed reference pipeline and the LEAK / SILENT / STALE / PROV defects |
| `scieval/agent.py` | Tool-calling agent loop over OpenRouter, plus scripted mock agents |
| `scieval/grade.py` | Pipeline grading and replay-based validity |
| `scieval/run.py` | Runs models across configurations (resumable, with a spending cap) |
| `scieval/probes.py` | Conformance probes and the aggregate pipeline-fidelity score |
| `scieval/replay_fixed.py` | Replays reference traces through each defective pipeline |
| `scieval/analyze.py`, `scieval/paired.py` | Aggregate metrics, paired contrasts, LaTeX tables |
| `scieval/figures.py` | Paper figures |
| `results/` | Summary JSON, tables, and figures from the main experiment |
| `paper/` | AES LaTeX source (`latexmk -xelatex main.tex`) |

## Reproduce

```sh
uv venv ~/.venvs/scieval --python 3.12 && ln -s ~/.venvs/scieval .venv
uv pip install --python .venv/bin/python numpy scipy pandas statsmodels openai matplotlib python-dotenv
echo 'OPENROUTER_API_KEY=sk-or-...' > .env   # gitignored

.venv/bin/python -m scieval.probes                                   # fidelity scores
.venv/bin/python -m scieval.run --models mock:careful,mock:sloppy --tasks 8 --exp smoke
.venv/bin/python -m scieval.run --tasks 20 --repeats 3 --exp main --budget 40 \
  --models openai/gpt-6.1-sol,anthropic/claude-sonnet-5.5,google/gemini-4-argon,anthropic/claude-haiku-5.5,openai/gpt-6-luna,deepseek/deepseek-v4.1-flash,qwen/qwen3.8-flash
.venv/bin/python -m scieval.analyze --exp main
.venv/bin/python -m scieval.paired
.venv/bin/python -m scieval.replay_fixed --exp main
.venv/bin/python -m scieval.figures --exp main
```

Agent code runs under a macOS `sandbox-exec` profile (no network, writes confined to the run workspace).
Run traces are written to `~/.cache/scieval/runs/` (about 1,680 JSON files for the main experiment).
The main experiment cost $35.60 in OpenRouter credit.

## Version 2: truth / evidence / credit audit (`research_v2/`)

| Path | Contents |
|---|---|
| `research_v2/oracle.py` | Frozen statistical oracle (OLS, outcome ~ exposure + site, two-sided, alpha 0.05) and hash-bound evidence records |
| `research_v2/harness.py`, `agent2.py`, `run2.py` | Trusted one-shot `confirm(exposure)` protocol, validator, single-boundary faults (SEP_READ, INT_MASK, FRESH_CACHE), runner with pinned endpoints and a spending cap |
| `research_v2/faults/contracts/` | Machine-readable fault contracts |
| `research_v2/audit_runner.py`, `audit_existing.py` | Instrumented re-execution of every original agent-written confirm script |
| `research_v2/decompose_v1.py`, `analysis_v1.py`, `analysis_v2.py`, `stats.py` | Truth/evidence/credit decomposition, stratified task-cluster bootstrap (B = 20,000), LOMO/LOTO, tau-b, Holm |
| `research_v2/dgp_validation.py` | 10,000-pair Monte Carlo validation of the generator and protocol |
| `research_v2/frozen/` | Frozen inputs: original run records, task suites, v2 run records, task manifest, OpenRouter metadata |
| `research_v2/results/` | All derived tables and summaries |
| `research_v2/tests/` | 15 tests for the oracle, protocol, validator, and fault isolation |
| `research_v2/appendix_tables.py`, `figures_v2.py`, `tables.py` | Every paper table and data figure, generated from frozen results |
| `paper/figures/src/pipeline.tex` | Editable TikZ source of the pipeline figure (also `pipeline.pdf`, `pipeline.svg`) |

Offline reproduction (no API key, no network): `bash research_v2/reproduce_offline.sh`.
The v2 experiment (640 runs, 4 models, 40 new tasks) cost $2.94.
