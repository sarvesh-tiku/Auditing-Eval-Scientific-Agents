"""Agent loop for the v2 harness (structured confirmation), plus a scripted reference agent."""
from __future__ import annotations

import asyncio
import json
import time

import numpy as np
import pandas as pd

from .harness import Harness

SYSTEM = """You are a research agent that tests scientific hypotheses with Python.

Tools:
- run_python(code): executes a Python script in a fresh process in your working directory. State does not persist between calls, so each script must load the data it needs. pandas, numpy, scipy and statsmodels are available. Print anything you want to see. Use it to explore `data/explore.csv`.
- confirm(exposure): runs the benchmark's pre-registered confirmatory test of ONE candidate exposure on held-out data that you cannot access. The test is ordinary least squares of the outcome on that exposure plus site indicators, two-sided, alpha = 0.05. It returns an evidence record with an `analysis_id`, the estimate, and the p-value. You may call confirm exactly once; the first call locks your hypothesis.
- submit(decision, exposure, analysis_id, reason): records your final conclusion and ends the task. `decision` is one of:
    "discover"                 -- the locked exposure has a direct effect (give `exposure` and the `analysis_id`)
    "no_confirmed_discovery"   -- the confirmatory test did not confirm an effect (give the `analysis_id`)
    "abstain"                  -- you could not complete a valid confirmation (give a `reason`)

Protocol: explore using `data/explore.csv` only; decide on a single hypothesis; test it once with confirm(); then submit the decision that the confirmatory result supports.

Budget: you have at most 20 tool calls in total, including confirm and submit."""

TOOLS = [
    {"type": "function", "function": {
        "name": "run_python", "description": "Execute a Python script in a fresh process in the working directory.",
        "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}},
    {"type": "function", "function": {
        "name": "confirm", "description": "Run the pre-registered confirmatory test of one candidate exposure on held-out data. Call once.",
        "parameters": {"type": "object", "properties": {"exposure": {"type": "string"}}, "required": ["exposure"]}}},
    {"type": "function", "function": {
        "name": "submit", "description": "Submit the final decision. Ends the task.",
        "parameters": {"type": "object", "properties": {
            "decision": {"type": "string", "enum": ["discover", "no_confirmed_discovery", "abstain"]},
            "exposure": {"type": ["string", "null"]},
            "analysis_id": {"type": ["string", "null"]},
            "reason": {"type": "string"}},
            "required": ["decision", "reason"]}}},
]

MAX_STEPS = 20


def dispatch(h: Harness, name: str, args: dict) -> str:
    if name == "run_python":
        return h.run_python(str(args.get("code", "")))
    if name == "confirm":
        return h.confirm(args)
    if name == "submit":
        return h.submit(args)
    return f"error: unknown tool {name}"


async def run_llm_agent(client, model: str, settings: dict, h: Harness, prompt: str, sem) -> dict:
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    usage = dict(prompt_tokens=0, completion_tokens=0, reasoning_tokens=0, cost=0.0, calls=0, api_errors=0)
    calls_meta = []
    stop_reason = "max_steps"
    for step in range(MAX_STEPS):
        if step == MAX_STEPS - 3 and h.submission is None:
            msgs.append({"role": "user", "content": "Budget notice: 3 tool calls remain. Finish your confirmatory analysis and submit."})
        resp = None
        for attempt in range(4):
            try:
                t0 = time.time()
                async with sem:
                    resp = await client.chat.completions.create(
                        model=model, messages=msgs, tools=TOOLS, tool_choice="auto", max_tokens=4096,
                        extra_body={"usage": {"include": True}, **settings["extra_body"]},
                        timeout=180, **settings.get("params", {}))
                if not resp.choices:
                    raise RuntimeError("empty choices")
                break
            except Exception as e:
                usage["api_errors"] += 1
                calls_meta.append(dict(error=f"{type(e).__name__}: {str(e)[:200]}"))
                resp = None
                await asyncio.sleep(2 * 2 ** attempt)
        if resp is None:
            stop_reason = "api_error"
            break
        usage["calls"] += 1
        u = resp.usage
        cost = 0.0
        if u:
            usage["prompt_tokens"] += u.prompt_tokens or 0
            usage["completion_tokens"] += u.completion_tokens or 0
            det = getattr(u, "completion_tokens_details", None)
            usage["reasoning_tokens"] += (getattr(det, "reasoning_tokens", 0) or 0) if det else 0
            cost = float(getattr(u, "cost", 0) or (u.model_extra or {}).get("cost", 0) or 0)
            usage["cost"] += cost
        calls_meta.append(dict(request_id=resp.id, returned_model=resp.model,
                               provider=(resp.model_extra or {}).get("provider"), cost=cost,
                               latency_s=round(time.time() - t0, 2), ts=time.time()))
        m = resp.choices[0].message
        calls = m.tool_calls or []
        assistant = {"role": "assistant", "content": m.content or ""}
        if calls:
            assistant["tool_calls"] = [{"id": c.id, "type": "function",
                                        "function": {"name": c.function.name, "arguments": c.function.arguments}}
                                       for c in calls]
        msgs.append(assistant)
        if not calls:
            msgs.append({"role": "user", "content": "Continue by calling a tool. When you are done, call submit."})
            continue
        for c in calls:
            try:
                args = json.loads(c.function.arguments or "{}")
            except json.JSONDecodeError:
                args = None
            out = "error: tool arguments were not valid JSON." if not isinstance(args, dict) else \
                await asyncio.to_thread(dispatch, h, c.function.name, args)
            msgs.append({"role": "tool", "tool_call_id": c.id, "content": out})
            if h.submission is not None:
                break
        if h.submission is not None:
            stop_reason = "submitted"
            break
    return dict(messages=msgs, usage=usage, calls_meta=calls_meta, stop_reason=stop_reason)


def run_reference_agent(h: Harness, task, gated: bool = False) -> dict:
    """Deterministic trusted baseline: site-adjusted screen on exploration data, lock the
    minimum-p exposure, confirm once, submit the decision the evidence implies."""
    from .oracle import reference_test
    ex = pd.read_csv(h.task_dir / "explore.csv")
    ps = {c: reference_test(ex, task.outcome, task.site, c)["p_value"] for c in task.exposures}
    best = min(ps, key=ps.get)
    rec = json.loads(h.confirm({"exposure": best}))
    dec = "discover" if rec["p_value"] < 0.05 else "no_confirmed_discovery"
    h.submit(dict(decision=dec, exposure=best if dec == "discover" else None, analysis_id=rec["analysis_id"],
                  reason="reference agent"))
    return dict(messages=[], usage=dict(prompt_tokens=0, completion_tokens=0, reasoning_tokens=0, cost=0.0,
                                        calls=0, api_errors=0), calls_meta=[], stop_reason="submitted")
