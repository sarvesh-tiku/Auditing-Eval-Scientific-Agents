"""Truth / evidence / decision / credit decomposition of the original 1,680 runs (no model calls).

Inputs: the original run records (read-only) and research_v2/frozen/confirm_audit.jsonl.
Output: research_v2/results/v1_per_run.csv, one row per original run.

Definitions (original runs; the agent wrote its own confirm script):
  decision           discover (an exposure named) | none | inconclusive | missing
  truth_correct      discover(x): x is the true exposure; none: task is null;
                     inconclusive / missing: undetermined (reported separately)
  agent_evidence     the original replay check: the cited agent-authored confirm script, re-run
                     under reference semantics, prints the submitted answer (and, for a discovery,
                     the submitted estimate and p < .05)
  tested_exposures   candidate exposures that entered any model fitted by that script (instrumented)
  hypothesis         discover(x): x; none: the single tested exposure if exactly one, else unrecoverable
  independent_p      frozen oracle test of the hypothesis on the confirmation split
  independent_evidence  discover(x): independent_p < .05; none: independent_p >= .05
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from scieval.grade import norm_answer, parse_confirm_json
from scieval.tasks import load_task
from .oracle import ALPHA, reference_test

HERE = Path(__file__).resolve().parent
CACHE = Path(os.environ.get("SCIEVAL_CACHE", Path.home() / ".cache" / "scieval"))
RUNS = CACHE / "runs" / "main"
SUITE = CACHE / "suites" / "n20"
OUT = HERE / "results" / "v1_per_run.csv"


def main():
    audit = {(r["model"], r["config"], r["task_id"], r["repeat"]): r
             for r in map(json.loads, open(HERE / "frozen" / "confirm_audit.jsonl"))}
    tasks, oracle = {}, {}
    for d in SUITE.iterdir():
        if d.is_dir():
            t = load_task(d)
            tasks[t.task_id] = t
            cf = pd.read_csv(d / "confirm.csv")
            oracle[t.task_id] = {c: reference_test(cf, t.outcome, t.site, c) for c in t.exposures}
    rows = []
    for f in sorted(RUNS.glob("*/*/*.json")):
        rec = json.loads(f.read_text())
        t = tasks[rec["task_id"]]
        m = rec["metrics"]
        a = audit[(rec["model"], rec["config"], rec["task_id"], rec["repeat"])]
        ans = m["answer"]
        decision = "discover" if ans in t.exposures else ans if ans in ("none", "inconclusive") else "missing"
        tested, fits, multi_fit_distinct = [], 0, False
        if a.get("fit_log"):
            per_fit = []
            for fit in a["fit_log"]:
                ex = sorted({x["exposure"] for x in fit.get("regressors", []) if x["exposure"] not in (None, "?")})
                if "regressors" in fit:
                    fits += 1
                    per_fit.append(ex)
                tested += ex
            tested = sorted(set(tested))
            multi_fit_distinct = len({tuple(x) for x in per_fit if x}) > 1
        hyp = None
        if decision == "discover":
            hyp = ans
        elif decision == "none" and len(tested) == 1:
            hyp = tested[0]
        ip = oracle[t.task_id][hyp]["p_value"] if hyp else None
        ind = None
        if decision == "discover":
            ind = ip < ALPHA
        elif decision == "none" and hyp is not None:
            ind = ip >= ALPHA
        js = a.get("recorded_json") or {}
        agent_printed = norm_answer(js.get("answer"), t.exposures) if js else None
        printed_p = js.get("p_value") if js else None
        try:
            printed_p = float(printed_p)
        except (TypeError, ValueError):
            printed_p = None
        truth_correct = None
        if decision == "discover":
            truth_correct = ans == t.truth
        elif decision == "none":
            truth_correct = t.truth == "none"
        rows.append(dict(
            model=rec["model"], config=rec["config"], task_id=t.task_id, kind=t.kind, truth=t.truth,
            repeat=rec["repeat"], decision=decision, answer=ans, truth_correct=truth_correct,
            v1_correct=m["correct"], agent_evidence=m["supported_replay"], agent_evidence_recorded=m["supported_recorded"],
            has_confirm=a["has_confirm"], confirm_failed=m["cited_confirm_failed"],
            agent_printed_answer=agent_printed, agent_printed_p=printed_p,
            submission_matches_printed=(agent_printed == ans) if agent_printed else None,
            n_fits=fits, n_tested_exposures=len(tested), tested_exposures=";".join(tested),
            multi_fit_distinct=multi_fit_distinct, hypothesis=hyp, independent_p=ip,
            independent_evidence=ind, valid_success=m["valid_success"],
            **{f"measured_{c}": m[f"measured_{c}"] for c in ("REF", "LEAK", "SILENT", "STALE", "PROV")},
            n_errors_actual=m["n_errors_actual"], n_errors_hidden=m["n_errors_hidden"], n_stale=m["n_stale"],
            cited_confirm_stale=m["cited_confirm_stale"], peeked_confirm=m["peeked_confirm"],
            n_confirm=m["n_confirm"], n_confirm_refused=m["n_confirm_refused"],
            heldout_listed=any(e["tool"] == "run_python" and "confirm.csv" in e["recorded"]["stdout"]
                               for e in rec["events"] if e.get("recorded")),
            stop_reason=rec["stop_reason"], cost=rec["usage"]["cost"],
        ))
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(df.groupby("config")[["truth_correct", "agent_evidence", "independent_evidence", "valid_success"]]
          .agg(lambda s: s.dropna().astype(float).mean()).round(3))


if __name__ == "__main__":
    main()
