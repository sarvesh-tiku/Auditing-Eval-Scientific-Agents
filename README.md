# When the Pipeline Is Wrong: Auditing the Evaluation of Scientific Agents

**Sarvesh Tiku, Gauri Sharma** — Georgia Institute of Technology
Agent Evaluation Science (AES) Symposium, Fall 2026

When a scientific-agent benchmark marks a conclusion correct, it conflates three things: whether the
conclusion is **true**, whether the agent's experiment **supports** it, and whether the grader **credited**
it for that reason. This repository contains the code, frozen run records, and analyses for a study that
measures these quantities separately in hypothesis-testing tasks with known generating processes.

## Summary

- **Setup.** Agents explore a synthetic observational dataset, lock one candidate exposure, confirm it on a
  held-out split, and submit a decision. Ground truth is known by construction, and a frozen statistical
  oracle recomputes the confirmatory evidence independently of the agent.
- **Original protocol (v1).** 1,680 runs of seven models on 20 tasks, in which agents wrote their own
  confirmation scripts. Instrumented re-execution shows that 42% of single confirmation calls fit more than
  one candidate hypothesis on held-out data. Of the 19 reference traces credited only by an answer-only
  grader, 8 contradict the agent's own significant test and 11 were rejected by the reference grader for
  reasons other than missing evidence.
- **Trusted protocol (v2).** The harness runs one locked, server-side test and the decision must cite an
  immutable evidence record. Across 640 runs of four models on 40 new tasks, every confirmed decision
  matched its evidence, yet an answer-only grader still credited untested "no discovery" decisions.
- **Faults.** Data-separation, execution-integrity, result-freshness, and evidence-provenance faults are
  injected one boundary at a time, and each fault's trigger, visibility, and use are logged.
- **Implication.** Scientific-agent benchmarks should execute confirmatory analyses in the harness, bind
  decisions to evidence records, and report truth, evidential support, and credit as distinct quantities.

## Repository structure

```
scieval/                 Original (v1) harness
  tasks.py               Synthetic tasks with known generating processes
  pipeline.py            Sandboxed pipeline and the LEAK / SILENT / STALE / PROV configurations
  agent.py               Tool-calling agent loop and scripted mock agents
  grade.py               v1 grader and replay-based support check
  run.py                 Experiment runner (resumable)
  replay_fixed.py        Fixed-action replay of reference traces
  probes.py              Conformance probes
  analyze.py, paired.py, figures.py

research_v2/             Trusted-confirmation (v2) protocol and all analyses in the paper
  oracle.py              Frozen statistical oracle and hash-bound evidence records
  harness.py             confirm(exposure) protocol, validator, fault operators
  agent2.py, run2.py     v2 agent loop and runner
  faults/contracts/      Machine-readable fault contracts
  audit_runner.py        Instrumentation of agent-written confirmation scripts
  audit_existing.py      Re-execution of every v1 confirmation script
  decompose_v1.py        Truth / evidence / credit decomposition of v1 runs
  analysis_v1.py         v1 estimands, sensitivity, rankings
  analysis_v2.py         v2 estimands, manifestation, rankings
  stats.py               Task-cluster bootstrap, sign-flip test, Kendall tau-b, Holm
  dgp_validation.py      Monte Carlo validation of the generator (10,000 dataset pairs)
  tables.py, appendix_tables.py, figures_v2.py
  tests/                 Unit tests for the oracle, protocol, validator, and faults
  frozen/                Frozen inputs: run records, task suites, task manifest, model metadata
  results/               Derived tables and summaries
  reproduce_offline.sh   Regenerates every statistic, table, and figure

paper/                   AES LaTeX source; figure sources in paper/figures/src/
results/                 Original v1 summaries
```

## Reproducing the paper

All results, tables, and figures can be regenerated from the frozen records without network access or
model calls:

```sh
uv venv ~/.venvs/scieval --python 3.12 && ln -s ~/.venvs/scieval .venv
uv pip install --python .venv/bin/python numpy scipy pandas statsmodels matplotlib pytest openai python-dotenv
bash research_v2/reproduce_offline.sh
```

The script restores the frozen archives, runs the unit tests, regenerates the Monte Carlo validation and
both analyses, rebuilds all figures and tables, and compiles `paper/main.pdf` (requires XeLaTeX).
Re-executing the v1 confirmation scripts under instrumentation is optional (`REAUDIT=1`, macOS only).

## Running new experiments

New agent runs require an OpenRouter key in a local `.env` file (`OPENROUTER_API_KEY=...`; gitignored).
Agent code executes under a macOS `sandbox-exec` profile with no network access.

```sh
# v2 trusted protocol: deterministic reference agent, then language-model agents
.venv/bin/python -m research_v2.run2 --models reference --exp v2main
.venv/bin/python -m research_v2.run2 --exp v2main \
  --models openai/gpt-6.1-sol,anthropic/claude-haiku-5.5,openai/gpt-6-luna,deepseek/deepseek-v4.1-flash
.venv/bin/python -m research_v2.analysis_v2

# v1 original protocol
.venv/bin/python -m scieval.run --tasks 20 --repeats 3 --exp main --models <comma-separated model ids>
.venv/bin/python -m scieval.analyze --exp main
```

Live model runs are not bit-for-bit reproducible even with pinned endpoints and seeds, which is why the
records used in the paper are frozen in `research_v2/frozen/`.
