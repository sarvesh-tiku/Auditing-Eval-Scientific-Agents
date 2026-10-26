"""Run the v2 experiment: fresh agents in REF and the three isolated environment faults.

    python -m research_v2.run2 --models mock --tasks 4 --exp smoke2
    python -m research_v2.run2 --models anthropic/claude-haiku-5.5,openai/gpt-6-luna --exp v2main --budget 5

Runs are written one JSON file per run (resumable; existing files are skipped),
conditions are executed in a randomized order within each task x model x repeat
block, and new runs stop being scheduled once recorded spend reaches --budget.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import shutil
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

from scieval.pipeline import ROOT
from scieval.tasks import load_task, make_task
from .agent2 import run_llm_agent, run_reference_agent
from .harness import CONDITIONS, ENV_CONDITIONS, Harness, fault_manifestation, validate
from .oracle import ORACLE_VERSION, environment

HERE = Path(__file__).resolve().parent
CACHE = Path(os.environ.get("SCIEVAL_CACHE", Path.home() / ".cache" / "scieval"))
SUITE_V2 = CACHE / "suites" / "v2_40"
RUNS_V2 = HERE / "frozen" / "runs_v2"
GENERATOR_VERSION = "scieval.tasks v1 (unchanged), seeds 5000-5039"

# Pinned endpoints and decoding settings; parameters are sent only where the
# model's OpenRouter metadata lists them as supported (frozen/openrouter_models_*.json).
MODEL_SETTINGS = {
    "anthropic/claude-haiku-5.5": dict(provider="anthropic", reasoning="low", seed=False, temperature=False),
    "openai/gpt-6-luna": dict(provider="openai", reasoning="low", seed=True, temperature=False),
    "openai/gpt-6.1-sol": dict(provider="openai", reasoning="low", seed=True, temperature=False),
    "qwen/qwen3.8-flash": dict(provider="alibaba", reasoning="low", seed=True, temperature=True),
    "deepseek/deepseek-v4.1-flash": dict(provider="deepinfra/fp8", reasoning="low", seed=True, temperature=True),
}


def request_settings(model: str, rep: int) -> dict:
    s = MODEL_SETTINGS[model]
    extra = {"provider": {"order": [s["provider"]], "allow_fallbacks": False, "require_parameters": True},
             "reasoning": {"effort": s["reasoning"]}}
    params = {}
    if s["seed"]:
        params["seed"] = 10_000 + rep
    if s["temperature"]:
        params["temperature"] = 0
    return dict(extra_body=extra, params=params, pinned_provider=s["provider"])


def v2_kind(i: int) -> str:
    # blocks of four consecutive seeds cover all four surface framings (seed % 4)
    return "effect" if (i % 8) < 4 else "null"


def write_suite_v2(n: int = 40) -> list[Path]:
    SUITE_V2.mkdir(parents=True, exist_ok=True)
    dirs = []
    for i in range(n):
        t, ex, cf = make_task(5000 + i, v2_kind(i))
        d = SUITE_V2 / t.task_id
        d.mkdir(exist_ok=True)
        if not (d / "task.json").exists():
            ex.to_csv(d / "explore.csv", index=False)
            cf.to_csv(d / "confirm.csv", index=False)
            (d / "task.json").write_text(json.dumps(asdict(t), indent=2))
        dirs.append(d)
    manifest = []
    for d in dirs:
        t = load_task(d)
        manifest.append(dict(task_id=t.task_id, generator_seed=t.seed, truth_type=t.kind,
                             true_exposure=None if t.truth == "none" else t.truth, beta=t.beta, decoy=t.decoy,
                             surface_framing=t.domain, generator_version=GENERATOR_VERSION,
                             exploration_dataset_sha256=hashlib.sha256((d / "explore.csv").read_bytes()).hexdigest(),
                             confirmation_dataset_sha256=hashlib.sha256((d / "confirm.csv").read_bytes()).hexdigest()))
    (HERE / "frozen" / "task_manifest_v2.json").write_text(json.dumps(manifest, indent=1))
    return dirs


def slug(m: str) -> str:
    return m.replace("/", "__").replace(":", "_")


async def one(client, model, cond, task_dir, rep, out_path, sem):
    task = load_task(task_dir)
    ws = CACHE / "ws2" / uuid.uuid4().hex[:12]
    h = Harness(CONDITIONS[cond], task, task_dir, ws)
    t0 = time.time()
    settings = None
    if model == "reference":
        res = await asyncio.to_thread(run_reference_agent, h, task)
    else:
        settings = request_settings(model, rep)
        res = await run_llm_agent(client, model, settings, h, task.prompt, sem)
    shutil.rmtree(ws, ignore_errors=True)
    if res["stop_reason"] == "api_error":
        print(f"[{model} {cond} {task.task_id} r{rep}] API ERROR, not saved", flush=True)
        return None
    val = validate(task, h.events, h.submission, h.records)
    rec = dict(model=model, condition=cond, task_id=task.task_id, kind=task.kind, truth=task.truth, repeat=rep,
               settings=settings, oracle_version=ORACLE_VERSION, environment=environment(),
               submission=h.submission, validation=val, fault=fault_manifestation(cond, h.events),
               evidence_records=h.records, usage=res["usage"], calls_meta=res["calls_meta"],
               stop_reason=res["stop_reason"], wall=round(time.time() - t0, 1), started=t0,
               events=h.events, messages=res["messages"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rec, default=str))
    print(f"[{model} {cond} {task.task_id} r{rep}] dec={val['decision']} exp={val['exposure']} "
          f"locked={val['locked_exposure']} truth_ok={val['truth_correct']} ev_ok={val['independent_evidence_supported']} "
          f"joint={val['joint_valid_success']} ${res['usage']['cost']:.4f} {rec['wall']}s {res['stop_reason']}", flush=True)
    return rec


async def main(args):
    load_dotenv(ROOT / ".env")
    task_dirs = write_suite_v2(40)[: args.tasks]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    conds = args.conditions.split(",") if args.conditions else ENV_CONDITIONS
    client = None
    if any(m != "reference" for m in models):
        from openai import AsyncOpenAI
        client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"],
                             max_retries=1)
    rng = random.Random(20261025)
    jobs = []
    for rep in range(args.repeats):
        for td in task_dirs:
            for model in rng.sample(models, len(models)):
                for cond in rng.sample(conds, len(conds)):  # randomized order within the block
                    p = RUNS_V2 / args.exp / slug(model) / cond / f"{td.name}_r{rep}.json"
                    if not p.exists():
                        jobs.append((model, cond, td, rep, p))
    print(f"{len(jobs)} runs to do", flush=True)
    sem = asyncio.Semaphore(args.concurrency)
    runner = asyncio.Semaphore(args.concurrency)
    prior = sum(json.loads(p.read_text())["usage"]["cost"] for p in (RUNS_V2 / args.exp).rglob("*.json")) \
        if (RUNS_V2 / args.exp).exists() else 0.0
    spent = [prior]
    print(f"prior spend in this experiment: ${prior:.3f}", flush=True)

    async def guarded(j):
        async with runner:
            if spent[0] >= args.budget:
                return None
            try:
                rec = await one(client, *j[:4], j[4], sem)
                if rec:
                    spent[0] += rec["usage"]["cost"]
                    print(f"--- spent ${spent[0]:.3f} / ${args.budget:.2f}", flush=True)
                return rec
            except Exception as e:
                print(f"FAILED {j[:4]}: {type(e).__name__}: {e}", flush=True)

    await asyncio.gather(*(guarded(j) for j in jobs))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--conditions", default="")
    ap.add_argument("--tasks", type=int, default=40)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--exp", default="v2main")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--budget", type=float, default=5.0)
    asyncio.run(main(ap.parse_args()))
