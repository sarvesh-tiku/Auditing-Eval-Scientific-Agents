"""Re-execute every cited confirm script from the original experiment under instrumentation.

For each of the 1,680 original runs this records which candidate exposures the
agent's confirmatory script actually entered into a fitted model, the JSON it
printed, and whether re-execution reproduces the originally recorded output.
No model calls are made; the original run records are read, never modified.

    python -m research_v2.audit_existing            # writes research_v2/frozen/confirm_audit.jsonl
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from scieval.grade import cited_confirm, parse_confirm_json
from scieval.pipeline import PYTHON, SANDBOX_PROFILE
from scieval.tasks import load_task

HERE = Path(__file__).resolve().parent
CACHE = Path(os.environ.get("SCIEVAL_CACHE", Path.home() / ".cache" / "scieval"))
RUNS = CACHE / "runs" / "main"
SUITE = CACHE / "suites" / "n20"
OUT = HERE / "frozen" / "confirm_audit.jsonl"


def run_instrumented(code: str, task, task_dir: Path) -> dict:
    ws = CACHE / "ws_audit" / uuid.uuid4().hex[:12]
    (ws / "confirm" / "data").mkdir(parents=True)
    shutil.copy(task_dir / "explore.csv", ws / "confirm" / "data" / "explore.csv")
    shutil.copy(task_dir / "confirm.csv", ws / "confirm" / "data" / "confirm.csv")
    (ws / "agent.py").write_text(code)
    (ws / "meta.json").write_text(json.dumps(dict(exposures=task.exposures)))
    (ws / "sandbox.sb").write_text(SANDBOX_PROFILE)
    env = {"PATH": "/usr/bin:/bin", "HOME": str(ws), "MPLCONFIGDIR": str(ws / ".mpl"),
           "PYTHONDONTWRITEBYTECODE": "1", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    try:
        p = subprocess.run(["sandbox-exec", "-f", str(ws / "sandbox.sb"), "-D", f"WS={ws}", PYTHON,
                            str(HERE / "audit_runner.py"), str(ws / "agent.py"), str(ws / "meta.json"),
                            str(ws / "log.json")],
                           cwd=ws / "confirm", capture_output=True, text=True, timeout=120, env=env)
        res = dict(exit_code=p.returncode, stdout=p.stdout, stderr=p.stderr[-2000:])
    except subprocess.TimeoutExpired:
        res = dict(exit_code=-9, stdout="", stderr="timeout")
    try:
        res["log"] = json.loads((ws / "log.json").read_text())
    except Exception:
        res["log"] = None
    shutil.rmtree(ws, ignore_errors=True)
    return res


def audit_one(path: Path) -> dict:
    rec = json.loads(path.read_text())
    task_dir = SUITE / rec["task_id"]
    task = load_task(task_dir)
    ev = cited_confirm(rec["events"], rec["submission"])
    out = dict(model=rec["model"], config=rec["config"], task_id=rec["task_id"], repeat=rec["repeat"],
               has_confirm=ev is not None)
    if ev is None:
        return out
    r = run_instrumented(ev["code"], task, task_dir)
    js_new = parse_confirm_json(r["stdout"])
    js_old = parse_confirm_json(ev["actual"].get("stdout", ""))
    out.update(cited_run_id=ev["run_id"], replay_exit=r["exit_code"], replay_json=js_new,
               recorded_json=js_old, replay_matches_recorded=(js_new == js_old),
               fit_log=r["log"], stderr_tail=r["stderr"][-300:] if r["exit_code"] else "")
    return out


def main():
    paths = sorted(p for cfg in ("REF", "LEAK", "SILENT", "STALE") for p in RUNS.glob(f"*/{cfg}/*.json"))
    print(len(paths), "records", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = 0
    with ThreadPoolExecutor(max_workers=7) as ex, open(OUT, "w") as fh:
        for row in ex.map(audit_one, paths):
            fh.write(json.dumps(row) + "\n")
            done += 1
            if done % 100 == 0:
                print(done, flush=True)


if __name__ == "__main__":
    main()
