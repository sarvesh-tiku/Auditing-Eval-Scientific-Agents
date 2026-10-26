"""Fixed-action replay of reference traces through defective pipelines (no API calls).

Holding the agent's recorded actions fixed isolates the evaluation discrepancy a
defect introduces from any change in agent behaviour: the same action sequence is
executed by each configuration and scored by that configuration's own grader.

  python -m scieval.replay_fixed --exp main
"""
from __future__ import annotations

import argparse
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .grade import cited_confirm, grade_run
from .pipeline import CONFIGS, ROOT, Pipeline
from .run import CACHE
from .tasks import load_task

SUITE = CACHE / "suites" / "n20"


def replay_trace(rec: dict, cfg: str) -> dict:
    task_dir = SUITE / rec["task_id"]
    task = load_task(task_dir)
    ws = CACHE / "ws" / ("fx_" + uuid.uuid4().hex[:10])
    pipe = Pipeline(CONFIGS[cfg], task_dir, ws)
    # map recorded run ids to the replayed ones (they coincide unless refusals interleave)
    id_map = {}
    for e in rec["events"]:
        if e["tool"] == "run_python":
            pipe.run_python(e["code"])
            id_map[e["run_id"]] = pipe.events[-1]["run_id"]
        elif e["tool"] == "confirm":
            pipe.confirm(e["code"])
            id_map[e["run_id"]] = pipe.events[-1]["run_id"]
    sub = rec["submission"]
    if sub is not None:
        sub = dict(sub, evidence_run_id=id_map.get(str(sub.get("evidence_run_id")), sub.get("evidence_run_id")))
    m = grade_run(task, pipe.events, sub, task_dir, ws.parent / (ws.name + "_rp"), do_replay=False)
    import shutil
    shutil.rmtree(ws, ignore_errors=True)
    return dict(model=rec["model"], task=rec["task_id"], kind=rec["kind"], rep=rec["repeat"], config=cfg,
                measured=m["measured_REF"], orig_measured=rec["metrics"]["measured_REF"],
                orig_valid=rec["metrics"]["valid_success"], n_stale=m["n_stale"],
                n_hidden=m["n_errors_hidden"], cited_stale=m["cited_confirm_stale"])


def main(exp: str):
    recs = []
    for f in (CACHE / "runs" / exp).glob("*/REF/*.json"):
        r = json.loads(f.read_text())
        r.pop("messages", None)
        recs.append(r)
    jobs = [(r, c) for r in recs for c in ("LEAK", "SILENT", "STALE")]
    with ThreadPoolExecutor(8) as ex:
        out = list(ex.map(lambda j: replay_trace(*j), jobs))
    S = {}
    for c in ("LEAK", "SILENT", "STALE"):
        rs = [o for o in out if o["config"] == c]
        S[c] = dict(n=len(rs), measured=float(np.mean([o["measured"] for o in rs])),
                    valid=float(np.mean([o["orig_valid"] for o in rs])),
                    flipped_up=float(np.mean([o["measured"] and not o["orig_valid"] for o in rs])),
                    flipped_down=float(np.mean([not o["measured"] and o["orig_valid"] for o in rs])),
                    any_stale=float(np.mean([o["n_stale"] > 0 for o in rs])),
                    cited_stale=float(np.mean([o["cited_stale"] for o in rs])),
                    any_hidden=float(np.mean([o["n_hidden"] > 0 for o in rs])))
        print(c, {k: round(v, 3) for k, v in S[c].items()})
    od = ROOT / "results" / exp
    od.mkdir(parents=True, exist_ok=True)
    (od / "fixed_replay.json").write_text(json.dumps(dict(summary=S, rows=out), indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="main")
    main(ap.parse_args().exp)
