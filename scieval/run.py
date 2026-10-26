"""Run agents across pipeline configurations.

  python -m scieval.run --models mock --tasks 8 --repeats 1 --exp smoke
  python -m scieval.run --models openai/gpt-5-mini,anthropic/claude-haiku-4.5 --tasks 20 --repeats 3 --exp main
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv

from .agent import run_llm_agent, run_mock_agent
from .grade import grade_run
from .pipeline import CONFIGS, ENV_CONFIGS, ROOT, Pipeline
from .tasks import load_task, write_suite

CACHE = Path(os.environ.get("SCIEVAL_CACHE", Path.home() / ".cache" / "scieval"))


def slug(model: str) -> str:
    return model.replace("/", "__").replace(":", "_")


async def one(client, model, cfg_name, task_dir, rep, out_path, sem, exec_sem):
    task = load_task(task_dir)
    ws = CACHE / "ws" / uuid.uuid4().hex[:12]
    pipe = Pipeline(CONFIGS[cfg_name], task_dir, ws)
    t0 = time.time()
    if model.startswith("mock"):
        kind = model.split(":")[1] if ":" in model else "careful"
        res = await asyncio.to_thread(run_mock_agent, pipe, task, kind)
    else:
        res = await run_llm_agent(client, model, pipe, task.prompt, sem)
    async with exec_sem:
        metrics = await asyncio.to_thread(grade_run, task, pipe.events, pipe.submission, task_dir,
                                          CACHE / "ws" / (ws.name + "_replay"))
    if res["stop_reason"] == "api_error":  # leave unsaved so a rerun retries it
        print(f"[{model} {cfg_name} {task.task_id} r{rep}] API ERROR, not saved", flush=True)
        import shutil
        shutil.rmtree(ws, ignore_errors=True)
        return None
    rec = dict(model=model, config=cfg_name, task_id=task.task_id, kind=task.kind, truth=task.truth,
               repeat=rep, submission=pipe.submission, metrics=metrics, usage=res["usage"],
               stop_reason=res["stop_reason"], wall=round(time.time() - t0, 1),
               events=pipe.events, messages=res["messages"])
    import shutil
    shutil.rmtree(ws, ignore_errors=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rec, default=str))
    m = metrics
    print(f"[{model} {cfg_name} {task.task_id} r{rep}] ans={m['answer']} correct={m['correct']} "
          f"meas={m['measured_REF']} valid={m['valid_success']} stale={m['n_stale']} "
          f"hidden_err={m['n_errors_hidden']} peek={m['peeked_confirm']} ${res['usage']['cost']:.4f} "
          f"{rec['wall']}s {res['stop_reason']}", flush=True)
    return rec


async def main(args):
    load_dotenv(ROOT / ".env")
    suite_dir = CACHE / "suites" / f"n{args.tasks}"
    if not suite_dir.exists():
        write_suite(suite_dir, args.tasks)
    task_dirs = sorted(d for d in suite_dir.iterdir() if d.is_dir())[: args.tasks]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    configs = args.configs.split(",") if args.configs else ENV_CONFIGS
    client = None
    if any(not m.startswith("mock") for m in models):
        from openai import AsyncOpenAI
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise SystemExit("OPENROUTER_API_KEY not set (put it in .env at the repo root)")
        client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=key, max_retries=2)
    sem = asyncio.Semaphore(args.concurrency)
    exec_sem = asyncio.Semaphore(8)
    jobs = []
    # repeat-major order: if the budget runs out, completed cells stay balanced
    for rep in range(args.repeats):
        for td in task_dirs:
            for model in models:
                for cfg in configs:
                    p = CACHE / "runs" / args.exp / slug(model) / cfg / f"{td.name}_r{rep}.json"
                    if p.exists() and not args.overwrite:
                        continue
                    jobs.append((model, cfg, td, rep, p))
    print(f"{len(jobs)} runs to do", flush=True)
    runner = asyncio.Semaphore(args.concurrency)

    spent = [0.0]

    async def guarded(j):
        async with runner:
            if spent[0] >= args.budget:
                return None
            try:
                rec = await one(client, *j[:4], j[4], sem, exec_sem)
                if rec:
                    spent[0] += rec["usage"]["cost"]
                    if len(str(int(spent[0] * 10))) and int(spent[0] * 4) != int((spent[0] - rec["usage"]["cost"]) * 4):
                        print(f"--- spent ${spent[0]:.2f} of ${args.budget:.2f} budget", flush=True)
                return rec
            except Exception as e:
                print(f"FAILED {j[:4]}: {type(e).__name__}: {e}", flush=True)

    await asyncio.gather(*(guarded(j) for j in jobs))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--configs", default="")
    ap.add_argument("--tasks", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--exp", default="main")
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--budget", type=float, default=1e9, help="stop launching runs after this many USD")
    asyncio.run(main(ap.parse_args()))
