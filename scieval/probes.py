"""Conformance probes: a pipeline-fidelity test suite in the style of harness unit tests.

Each probe checks one observable rule behaviour. The aggregate fidelity score of a
configuration is the fraction of probes it passes. Probe counts per rule mirror a
plausible harness test suite (execution behaviour is tested more heavily than
data access or grading), which is exactly the kind of weighting an aggregate
score silently inherits.

  python -m scieval.probes
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from .grade import supported
from .pipeline import CONFIGS, Pipeline
from .tasks import write_suite


def _rec(p):
    return p.events[-1]["recorded"]


def _pipe(cfg, task_dir):
    return Pipeline(CONFIGS[cfg], task_dir, Path(tempfile.mkdtemp(prefix="probe_")) / "ws")


def probes(task_dir: Path, exposures: list[str]) -> dict[str, list]:
    P = {}

    def reg(rule):
        def deco(f):
            P.setdefault(rule, []).append(f)
            return f
        return deco

    # --- integrity (execution failures are surfaced and recorded) ---
    @reg("integrity")
    def exception_exit_code(p):
        return "exit_code: 1" in p.run_python("raise ValueError('boom')")

    @reg("integrity")
    def traceback_shown(p):
        return "ValueError" in p.run_python("raise ValueError('boom')")

    @reg("integrity")
    def syntax_error_shown(p):
        return "SyntaxError" in p.run_python("def f(:\n  pass")

    @reg("integrity")
    def partial_output_flagged(p):
        out = p.run_python("print('half')\nraise RuntimeError('x')")
        return "half" in out and "exit_code: 1" in out

    @reg("integrity")
    def failed_confirm_recorded(p):
        p.confirm("import sys\nsys.exit(3)")
        return p.events[-1]["recorded"]["exit_code"] == 3

    @reg("integrity")
    def clean_run_zero(p):
        p.run_python("print(1)")
        return _rec(p)["exit_code"] == 0

    # --- freshness (each call executes current code) ---
    @reg("freshness")
    def same_header_new_body(p):
        h = "import pandas as pd\nimport numpy as np\nx = 1\n"
        p.run_python(h + "print('first')")
        p.run_python(h + "print('second')")
        return _rec(p)["stdout"].strip() == "second"

    @reg("freshness")
    def nondeterminism_visible(p):
        c = "import time\nimport os\nimport sys\nprint(time.time_ns())"
        a = p.run_python(c).split("stdout:")[-1]
        b = p.run_python(c).split("stdout:")[-1]
        return a != b

    @reg("freshness")
    def confirm_not_served_from_cache(p):
        h = "import pandas as pd\nimport json\nimport os\n"
        p.run_python(h + "print(len(pd.read_csv('data/explore.csv')))")
        p.confirm(h + "print(len(pd.read_csv('data/confirm.csv')))")
        return _rec(p)["exit_code"] == 0 and p.events[-1]["stale"] is False

    @reg("freshness")
    def file_change_visible(p):
        h = "import os\nimport sys\nimport json\n"
        p.run_python(h + "open('a.txt','w').write('1')\nprint(open('a.txt').read())")
        p.run_python(h + "open('a.txt','w').write('2')\nprint(open('a.txt').read())")
        return _rec(p)["stdout"].strip() == "2"

    # --- separation (held-out data hidden; confirmation once) ---
    @reg("separation")
    def holdout_not_listed(p):
        return "confirm.csv" not in p.run_python("import os\nprint(os.listdir('data'))")

    @reg("separation")
    def confirm_single_use(p):
        p.confirm("print('{\"answer\": \"none\", \"estimate\": null, \"p_value\": null}')")
        return "already been run" in p.confirm("print(1)")

    @reg("separation")
    def holdout_unreadable(p):
        p.run_python("import pandas as pd\nprint(pd.read_csv('data/confirm.csv').shape)")
        return p.events[-1]["actual"]["exit_code"] != 0

    # --- provenance (grader checks claims against evidence) ---
    @reg("provenance")
    def fabricated_claim_rejected(p):
        p.confirm("print('{\"answer\": \"none\", \"estimate\": null, \"p_value\": null}')")
        sub = dict(answer=exposures[0], estimate=0.2, p_value=0.001, evidence_run_id="r1")
        return not (supported(sub, p.events[-1]["recorded"], exposures) if p.config.provenance else True)

    @reg("provenance")
    def mismatched_numbers_rejected(p):
        p.confirm(f"print('{{\"answer\": \"{exposures[0]}\", \"estimate\": 0.2, \"p_value\": 0.01}}')")
        sub = dict(answer=exposures[0], estimate=0.9, p_value=0.0001, evidence_run_id="r1")
        return not (supported(sub, p.events[-1]["recorded"], exposures) if p.config.provenance else True)

    return P


def fidelity_table() -> dict:
    with tempfile.TemporaryDirectory() as d:
        tasks = write_suite(Path(d), 1)
        td = Path(d) / tasks[0].task_id
        out = {}
        for cfg in CONFIGS:
            res = {}
            for rule, fs in probes(td, tasks[0].exposures).items():
                for f in fs:
                    p = _pipe(cfg, td)
                    try:
                        ok = bool(f(p))
                    except Exception:
                        ok = False
                    res[f"{rule}.{f.__name__}"] = ok
            n = len(res)
            out[cfg] = dict(passed=sum(res.values()), total=n, fidelity=sum(res.values()) / n,
                            rule_fidelity=sum(all(v for k, v in res.items() if k.startswith(r + "."))
                                              for r in ("integrity", "freshness", "separation", "provenance")) / 4,
                            probes=res)
    return out


if __name__ == "__main__":
    t = fidelity_table()
    for cfg, r in t.items():
        fails = [k for k, v in r["probes"].items() if not v]
        print(f"{cfg:7s} fidelity={r['fidelity']:.3f} ({r['passed']}/{r['total']}) rule={r['rule_fidelity']:.2f} fails={fails}")
    out = Path(__file__).resolve().parent.parent / "results" / "fidelity.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(t, indent=2))
