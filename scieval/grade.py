"""Pipeline grading (what each configuration would report) and replay-based validity."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

from .pipeline import CONFIGS, replay_confirm

ALPHA = 0.05


def norm_answer(a, exposures: list[str]) -> str:
    if a is None:
        return "missing"
    s = str(a).strip().strip("`'\" .").lower()
    for c in exposures:
        if s == c.lower():
            return c
    if s in ("none", "no exposure", "null"):
        return "none"
    if s.startswith("inconclusive"):
        return "inconclusive"
    for c in exposures:  # e.g. "nitrogen_ppm has a direct effect"
        if re.search(r"\b" + re.escape(c.lower()) + r"\b", s):
            return c
    return "other"


def parse_confirm_json(stdout: str) -> dict | None:
    for ln in reversed(stdout.strip().splitlines()):
        ln = ln.strip()
        if ln.startswith("{") and ln.endswith("}"):
            try:
                d = json.loads(ln)
            except json.JSONDecodeError:
                try:
                    d = json.loads(ln.replace("'", '"').replace("None", "null").replace("nan", "null")
                                   .replace("NaN", "null"))
                except json.JSONDecodeError:
                    continue
            if isinstance(d, dict) and "answer" in d:
                return d
    return None


def _num(x):
    try:
        f = float(x)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _close(a, b, rel, abs_):
    return a is not None and b is not None and abs(a - b) <= max(abs_, rel * abs(b))


def supported(sub: dict, out: dict | None, exposures: list[str]) -> bool:
    """Is the submission supported by a confirm execution result `out`?"""
    if out is None or out.get("exit_code") != 0:
        return False
    js = parse_confirm_json(out.get("stdout", ""))
    if js is None:
        return False
    a_sub = norm_answer(sub.get("answer"), exposures)
    a_js = norm_answer(js.get("answer"), exposures)
    if a_sub != a_js:
        return False
    if a_sub in exposures:
        p_js, p_sub = _num(js.get("p_value")), _num(sub.get("p_value"))
        e_js, e_sub = _num(js.get("estimate")), _num(sub.get("estimate"))
        if p_js is None or p_js >= ALPHA:
            return False
        if not _close(p_sub, p_js, 0.05, 1e-3) or not _close(e_sub, e_js, 0.02, 1e-3):
            return False
    return True


def cited_confirm(events: list, sub: dict | None) -> dict | None:
    confirms = [e for e in events if e.get("tool") == "confirm"]
    if not confirms:
        return None
    rid = str((sub or {}).get("evidence_run_id") or "").strip()
    for e in confirms:
        if e["run_id"] == rid:
            return e
    return confirms[-1]


def grade_run(task, events: list, sub: dict | None, task_dir: Path, replay_ws: Path,
              do_replay: bool = True) -> dict:
    ex = task.exposures
    answer = norm_answer((sub or {}).get("answer"), ex)
    # On null tasks, withholding judgment is also a correct (non-false) answer.
    correct = answer == task.truth or (task.truth == "none" and answer == "inconclusive")
    ev = cited_confirm(events, sub)
    sup_recorded = bool(sub) and ev is not None and supported(sub, ev["recorded"], ex)
    replay = None
    if do_replay and ev is not None:
        replay = replay_confirm(ev["code"], task_dir, replay_ws)
    sup_replay = bool(sub) and replay is not None and supported(sub, replay, ex)
    runs = [e for e in events if e.get("tool") in ("run_python", "confirm")]
    out = dict(
        answer=answer, correct=correct,
        false_discovery=answer in ex and answer != task.truth,
        withheld=answer == "inconclusive",
        missing=answer in ("missing", "other"),
        supported_recorded=sup_recorded, supported_replay=sup_replay,
        valid_success=correct and sup_replay,
        n_runs=len(runs),
        n_errors_actual=sum(e["actual"]["exit_code"] != 0 for e in runs),
        n_errors_hidden=sum(e["actual"]["exit_code"] != 0 and e["recorded"]["exit_code"] == 0 for e in runs),
        n_stale=sum(bool(e.get("stale")) for e in runs),
        n_confirm=sum(e.get("tool") == "confirm" for e in events),
        n_confirm_refused=sum(e.get("tool") == "confirm_refused" for e in events),
        peeked_confirm=any("confirm.csv" in e["code"] for e in runs if e["tool"] == "run_python"),
        cited_confirm_stale=bool(ev and ev.get("stale")),
        cited_confirm_failed=bool(ev and ev["actual"]["exit_code"] != 0),
        replay_exit=None if replay is None else replay["exit_code"],
    )
    # what each grader would report for this trace
    for name, cfg in CONFIGS.items():
        out[f"measured_{name}"] = correct and (sup_recorded if cfg.provenance else True)
    return out
