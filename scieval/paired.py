"""Paired REF-vs-defect contrasts (same model, task, repeat), false-discovery breakdown, tier splits.

  python -m scieval.paired
"""
import json, numpy as np, collections
from scieval.analyze import load, rows
from scieval.tasks import load_task
from scieval.run import CACHE
from scieval.pipeline import ROOT
recs = load("main"); R = rows(recs)
key = lambda r: (r["model"], r["task"], r["rep"])
by = {c: {key(r): r for r in R if r["config"] == c} for c in ["REF","LEAK","SILENT","STALE","PROV"]}
tasks = sorted({r["task"] for r in R}); rng = np.random.default_rng(1)
def paired(c, field_ref, field_c, models=None):
    ks = [k for k in by["REF"] if k in by[c] and (models is None or k[0] in models)]
    d = {t: [] for t in tasks}
    for k in ks:
        a = by[c][k]; b = by["REF"][k]
        va = a["measured"] if field_c == "measured" else a["m"]["valid_success"]
        vb = b["measured"] if field_ref == "measured" else b["m"]["valid_success"]
        d[k[1]].append(float(va) - float(vb))
    est = np.mean([x for v in d.values() for x in v])
    bs = []
    for _ in range(4000):
        samp = rng.choice(tasks, len(tasks))
        bs.append(np.mean([x for t in samp for x in d[t]]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p = 2 * min(np.mean(np.array(bs) <= 0), np.mean(np.array(bs) >= 0))
    return len(ks), est, lo, hi, p
out = {}
for c in ["LEAK", "SILENT", "STALE"]:
    out[c] = paired(c, "valid", "valid"); print(c, "Δvalid vs REF", [round(x, 4) for x in out[c]])
out["PROV"] = paired("PROV", "valid", "measured"); print("PROV measured − REF valid", [round(x, 4) for x in out["PROV"]])
# per-model paired valid deltas (no CI)
for m in sorted({r["model"] for r in R}):
    s = []
    for c in ["LEAK", "SILENT", "STALE"]:
        ks = [k for k in by["REF"] if k[0] == m and k in by[c]]
        s.append(round(np.mean([by[c][k]["m"]["valid_success"] - by["REF"][k]["m"]["valid_success"] for k in ks]) * 100, 1))
    print(m, s)
# false discoveries: decoy vs other
ex = {}
for r in recs:
    if r["metrics"]["false_discovery"]:
        t = load_task(CACHE / "suites" / "n20" / r["task_id"])
        ex.setdefault(r["config"], collections.Counter())["decoy" if r["metrics"]["answer"] == t.decoy else "other"] += 1
print("FD breakdown", {k: dict(v) for k, v in ex.items()})
# strong vs weak tier
tiers = {"frontier": ["openai/gpt-6.1-sol", "anthropic/claude-sonnet-5.5", "google/gemini-4-argon"]}
for name, ms in [("frontier", tiers["frontier"]), ("efficient", [m for m in {r["model"] for r in R} if m not in tiers["frontier"]])]:
    for c in ["REF", "LEAK", "SILENT", "STALE", "PROV"]:
        rs = [r for r in R if r["config"] == c and r["model"] in ms]
        print(name, c, "meas", round(np.mean([r["measured"] for r in rs]), 3), "valid", round(np.mean([r["m"]["valid_success"] for r in rs]), 3))
FRONTIER = ["openai/gpt-6.1-sol", "anthropic/claude-sonnet-5.5", "google/gemini-4-argon"]
EFFICIENT = [m for m in {r["model"] for r in R} if m not in FRONTIER]
full = {}
for c in ["LEAK", "SILENT", "STALE", "PROV"]:
    full[c] = {}
    for tier, ms in (("all", None), ("frontier", FRONTIER), ("efficient", EFFICIENT)):
        for f in ("measured", "valid"):
            n, est, lo, hi, p = paired(c, "valid" if f == "measured" else "valid", f, ms) if c == "PROV" or f == "valid" \
                else paired(c, "measured", "measured", ms)
            full[c][f"{tier}_{f}"] = dict(n=n, est=est, lo=lo, hi=hi, p=p)
json.dump(full, open(ROOT / "results" / "main" / "paired_full.json", "w"), indent=1)
json.dump({k: dict(n=v[0], est=v[1], lo=v[2], hi=v[3], p=v[4]) for k, v in out.items()},
          open(ROOT / "results" / "main" / "paired.json", "w"), indent=1)
