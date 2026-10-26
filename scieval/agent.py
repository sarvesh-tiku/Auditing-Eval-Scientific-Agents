"""Agent loop over OpenRouter (OpenAI-compatible tool calling), plus scripted mock agents."""
from __future__ import annotations

import asyncio
import json
import os
import time

from .pipeline import Pipeline

SYSTEM = """You are a research agent that tests scientific hypotheses with Python.

Tools:
- run_python(code): executes a Python script in a fresh process in your working directory. State does not persist between calls, so each script must load the data it needs. pandas, numpy, scipy and statsmodels are available. Print anything you want to see.
- confirm(code): runs your pre-specified confirmatory analysis on held-out data. The held-out data are at `data/confirm.csv` (same columns as `data/explore.csv`). You may call confirm exactly once. The script must print exactly one line of JSON of the form {"answer": <exposure name, "none", or "inconclusive">, "estimate": <float or null>, "p_value": <float or null>}.
- submit(answer, estimate, p_value, evidence_run_id, rationale): records your final conclusion. `evidence_run_id` is the run_id of the confirm call that supports it. Submitting ends the task.

Protocol: explore using `data/explore.csv` only; decide on a single hypothesis; test it once with confirm() at alpha = 0.05; then submit a conclusion that is supported by the confirmatory result. Answer "none" if you conclude no exposure has a direct effect, or "inconclusive" if the evidence does not support a conclusion.

Budget: you have at most 20 tool calls in total, including confirm and submit."""

TOOLS = [
    {"type": "function", "function": {
        "name": "run_python", "description": "Execute a Python script in a fresh process in the working directory.",
        "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}},
    {"type": "function", "function": {
        "name": "confirm", "description": "Run the confirmatory analysis script on held-out data (data/confirm.csv). Call once.",
        "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}},
    {"type": "function", "function": {
        "name": "submit", "description": "Submit the final conclusion. Ends the task.",
        "parameters": {"type": "object", "properties": {
            "answer": {"type": "string", "description": "Exposure name, 'none', or 'inconclusive'."},
            "estimate": {"type": ["number", "null"]},
            "p_value": {"type": ["number", "null"]},
            "evidence_run_id": {"type": ["string", "null"]},
            "rationale": {"type": "string"}},
            "required": ["answer", "rationale"]}}},
]

MAX_STEPS = 20


def _dispatch(pipe: Pipeline, name: str, args: dict) -> str:
    if name == "run_python":
        return pipe.run_python(str(args.get("code", "")))
    if name == "confirm":
        return pipe.confirm(str(args.get("code", "")))
    if name == "submit":
        return pipe.submit(str(args.get("answer", "")), args.get("estimate"), args.get("p_value"),
                           args.get("evidence_run_id"), str(args.get("rationale", "")))
    return f"error: unknown tool {name}"


async def run_llm_agent(client, model: str, pipe: Pipeline, prompt: str, sem: asyncio.Semaphore) -> dict:
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    usage = dict(prompt_tokens=0, completion_tokens=0, cost=0.0, calls=0, api_errors=0)
    stop_reason = "max_steps"
    for step in range(MAX_STEPS):
        if step == MAX_STEPS - 3 and pipe.submission is None:
            msgs.append({"role": "user", "content": "Budget notice: 3 tool calls remain. Finish your confirmatory analysis and submit."})
        resp = None
        for attempt in range(4):
            try:
                async with sem:
                    resp = await client.chat.completions.create(
                        model=model, messages=msgs, tools=TOOLS, tool_choice="auto",
                        max_tokens=4096, extra_body={"usage": {"include": True}}, timeout=180)
                if not resp.choices:
                    raise RuntimeError("empty choices")
                break
            except Exception as e:  # rate limits, provider errors
                usage["api_errors"] += 1
                resp = None
                await asyncio.sleep(2 * 2 ** attempt)
        if resp is None:
            stop_reason = "api_error"
            break
        usage["calls"] += 1
        if resp.usage:
            usage["prompt_tokens"] += resp.usage.prompt_tokens or 0
            usage["completion_tokens"] += resp.usage.completion_tokens or 0
            usage["cost"] += float(getattr(resp.usage, "cost", 0) or (resp.usage.model_extra or {}).get("cost", 0) or 0)
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
            if not isinstance(args, dict):
                out = "error: tool arguments were not valid JSON."
            else:
                out = await asyncio.to_thread(_dispatch, pipe, c.function.name, args)
            msgs.append({"role": "tool", "tool_call_id": c.id, "content": out})
            if pipe.submission is not None:
                break
        if pipe.submission is not None:
            stop_reason = "submitted"
            break
    return dict(messages=msgs, usage=usage, stop_reason=stop_reason)


# ---------------------------------------------------------------------------
# Scripted mock agents (for testing the harness without API calls)

_ADJ = """import pandas as pd, statsmodels.formula.api as smf, json
df = pd.read_csv('data/{f}.csv')
for c in {cols}:
    df[c] = pd.to_numeric(df[c], errors='coerce')
df = df.dropna()
"""


def run_mock_agent(pipe: Pipeline, task, kind: str = "careful") -> dict:
    cols = task.exposures
    pipe.run_python("import os\nprint(os.listdir('data'))")
    if kind == "sloppy":
        # naive correlation: errors on string columns
        pipe.run_python("import pandas as pd\ndf = pd.read_csv('data/explore.csv')\nprint(df.corr()['%s'])" % task.outcome)
    code = _ADJ.format(f="explore", cols=cols + [task.outcome]) + (
        "ps = {}\n"
        f"for c in {cols}:\n"
        f"    m = smf.ols('{task.outcome} ~ ' + c + ' + C({task.site})', data=df).fit()\n"
        "    ps[c] = (float(m.params[c]), float(m.pvalues[c]))\n"
        "best = min(ps, key=lambda c: ps[c][1])\nprint(best, ps[best][0], ps[best][1])\n")
    out = pipe.run_python(code)
    best, p = None, 1.0
    for ln in out.splitlines():
        if ln.split(" ")[0] in cols:
            best = ln.split(" ")[0]
            p = float(ln.split(" ")[2])
    if best is None:
        best = cols[0]
    target = best if p < 0.05 / len(cols) else None
    if target:
        conf = _ADJ.format(f="confirm", cols=cols + [task.outcome]) + (
            f"m = smf.ols('{task.outcome} ~ {target} + C({task.site})', data=df).fit()\n"
            f"p = float(m.pvalues['{target}'])\n"
            f"print(json.dumps({{'answer': '{target}' if p < 0.05 else 'none', 'estimate': float(m.params['{target}']), 'p_value': p}}))\n")
    else:
        conf = "import json\nprint(json.dumps({'answer': 'none', 'estimate': None, 'p_value': None}))\n"
    out = pipe.confirm(conf)
    rid = out.splitlines()[0].split(": ")[1]
    js = None
    for ln in out.splitlines():
        if ln.startswith("{"):
            js = json.loads(ln)
    if js is None:
        js = {"answer": "inconclusive", "estimate": None, "p_value": None}
    pipe.submit(js["answer"], js["estimate"], js["p_value"], rid, "mock")
    return dict(messages=[], usage=dict(prompt_tokens=0, completion_tokens=0, cost=0.0, calls=0, api_errors=0),
                stop_reason="submitted")
