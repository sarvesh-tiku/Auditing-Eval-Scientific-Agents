"""Analysis of the v2 experiment (trusted one-shot confirmation; 40 new task seeds).

    python -m research_v2.analysis_v2      # writes research_v2/results/v2/*
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .stats import B, boot_index, holm, signflip_p, leave_one_out, mean_ci, per_task, rank_analysis, ratio_ci

HERE = Path(__file__).resolve().parent
RUNS = HERE / "frozen" / "runs_v2" / "v2main"
OUT = HERE / "results" / "v2"
FAULTS = ["SEP_READ", "INT_MASK", "FRESH_CACHE"]


def load() -> pd.DataFrame:
    rows = []
    for f in sorted(RUNS.glob("*/*/*.json")):
        r = json.loads(f.read_text())
        v, fm = r["validation"], r["fault"]
        rows.append(dict(model=r["model"], condition=r["condition"], task_id=r["task_id"], kind=r["kind"],
                         truth=r["truth"], repeat=r["repeat"], cost=r["usage"]["cost"], calls=r["usage"]["calls"],
                         stop_reason=r["stop_reason"],
                         providers=";".join(sorted({str(c.get("provider")) for c in r["calls_meta"] if "provider" in c})),
                         returned_models=";".join(sorted({str(c.get("returned_model")) for c in r["calls_meta"] if "returned_model" in c})),
                         n_python=sum(e["tool"] == "run_python" for e in r["events"]),
                         n_errors=sum(e["tool"] == "run_python" and e["actual"]["exit_code"] != 0 for e in r["events"]),
                         **{k: v[k] for k in ("validation_status", "decision", "exposure", "locked_exposure", "locked_p",
                                              "implied_decision", "protocol_valid", "truth_correct",
                                              "independent_evidence_supported", "decision_consistent", "abstained",
                                              "joint_valid_success", "false_discovery", "credited_REF",
                                              "credited_PROV_GRADE", "n_confirm", "n_confirm_rejected",
                                              "heldout_accessed")},
                         **{f"fault_{k}": fm.get(k) for k in ("triggered", "visible", "dependent_use",
                                                             "last_pre_lock_output_faulty", "n_fault_events",
                                                             "heldout_read_before_lock", "later_unaffected_run")}))
    df = pd.DataFrame(rows)
    df["kind"] = df.task_id.str.split("-").str[1]
    return df


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_df = load()
    all_df.to_csv(OUT / "v2_per_run.csv", index=False)
    base = all_df[all_df.model == "reference"]
    df = all_df[all_df.model != "reference"].copy()
    # keep only complete task x model blocks (all four conditions present)
    full = df.groupby(["model", "task_id", "repeat"]).condition.nunique()
    keep = full[full == 4].index
    df = df.set_index(["model", "task_id", "repeat"]).loc[keep].reset_index()
    tasks = df[["task_id", "kind"]].drop_duplicates().sort_values("task_id").reset_index(drop=True)
    idx = boot_index(tasks)
    res = dict(n_runs=len(df), n_tasks=len(tasks), models=sorted(df.model.unique()), bootstrap_B=B,
               total_cost=float(all_df.cost.sum()),
               cost_by_model=df.groupby("model").cost.sum().round(4).to_dict(),
               providers=df.groupby("model").providers.agg(lambda s: sorted(set(";".join(s).split(";")))).to_dict(),
               returned_models=df.groupby("model").returned_models.agg(lambda s: sorted(set(";".join(s).split(";")))).to_dict(),
               stop_reasons=df.stop_reason.value_counts().to_dict())
    ref = df[df.condition == "REF"].copy()

    # reference (deterministic) agent on the same tasks
    bref = base[base.condition == "REF"]
    res["reference_agent"] = dict(
        joint_valid=float(bref.joint_valid_success.mean()),
        null_discoveries=int((bref[bref.kind == "null"].decision == "discover").sum()),
        effect_detected=int(((bref.kind == "effect") & bref.truth_correct.astype(bool)).sum()),
        n=len(bref), identical_across_conditions=bool(base.groupby("task_id").decision.nunique().max() == 1))

    # decomposition on REF
    ref["truth_s"] = ref.truth_correct.map({True: "correct", False: "incorrect"}).fillna("abstain/undetermined")
    ref["ev_s"] = ref.independent_evidence_supported.map({True: "supported", False: "unsupported"}).fillna("undetermined")
    res["ref_matrix"] = {f"{a}|{b}": int(v) for (a, b), v in ref.groupby(["truth_s", "ev_s"]).size().items()}
    res["ref_decisions"] = ref.decision.value_counts(dropna=False).to_dict()
    res["ref_by_model"] = ref.groupby("model")[["joint_valid_success", "credited_PROV_GRADE", "protocol_valid",
                                                "decision_consistent"]].mean().round(4).to_dict()
    res["decision_consistency"] = dict(
        all=float(df.decision_consistent.dropna().astype(float).mean()),
        by_condition=df.groupby("condition").decision_consistent.apply(lambda s: float(s.dropna().astype(float).mean())).to_dict())
    agree = ref.merge(bref[["task_id", "decision", "exposure"]], on="task_id", suffixes=("", "_base"))
    res["agreement_with_reference_agent"] = float(((agree.decision == agree.decision_base) &
                                                   ((agree.decision != "discover") | (agree.exposure == agree.exposure_base))).mean())
    res["same_lock_as_reference_agent"] = float((ref.merge(bref[["task_id", "locked_exposure"]], on="task_id",
                                                           suffixes=("", "_b")).pipe(lambda d: d.locked_exposure == d.locked_exposure_b)).mean())

    # estimands
    ref["prov"] = ref.credited_PROV_GRADE.astype(float) - ref.credited_REF.astype(float)
    est = {"PROV_GRADE_same_trace": mean_ci(ref, tasks, "prov", idx),
           "REF_joint_valid": mean_ci(ref.assign(j=ref.joint_valid_success.astype(float)), tasks, "j", idx)}
    a = ref.set_index(["model", "task_id", "repeat"]).joint_valid_success.astype(float)
    pv = {}
    for c in FAULTS:
        b = df[df.condition == c].set_index(["model", "task_id", "repeat"]).joint_valid_success.astype(float)
        d = (b - a).dropna().rename("diff").reset_index()
        est[f"{c}_fresh_joint_valid"] = ratio_ci(*per_task(d, tasks, "diff"), idx)
        est[f"{c}_fresh_joint_valid"]["perm_p"] = signflip_p(*per_task(d, tasks, "diff"))
        pv[c] = est[f"{c}_fresh_joint_valid"]["perm_p"]
    for c, p in holm(pv).items():
        est[f"{c}_fresh_joint_valid"]["holm_p"] = p
    res["estimands"] = est
    res["by_condition"] = df.groupby("condition").agg(
        joint_valid=("joint_valid_success", "mean"), prov_credit=("credited_PROV_GRADE", "mean"),
        abstain=("abstained", "mean"), protocol_valid=("protocol_valid", "mean"),
        null_fd=("false_discovery", lambda s: float(s[df.loc[s.index, "kind"] == "null"].mean())),
        calls=("calls", "mean"), n_errors=("n_errors", "mean")).round(4).to_dict()

    nc = ref[ref.credited_PROV_GRADE & ~ref.credited_REF]
    res["prov_newly_credited"] = nc[["model", "task_id", "decision", "exposure", "locked_exposure", "locked_p",
                                     "protocol_valid"]].to_dict("records")

    def eff(sub):
        r = sub[sub.condition == "REF"]
        out = {"PROV": float((r.credited_PROV_GRADE.astype(float) - r.credited_REF.astype(float)).mean())}
        a2 = r.set_index(["model", "task_id", "repeat"]).joint_valid_success.astype(float)
        for c in FAULTS:
            b2 = sub[sub.condition == c].set_index(["model", "task_id", "repeat"]).joint_valid_success.astype(float)
            out[c] = float((b2 - a2).dropna().mean())
        return out
    lomo, loto = leave_one_out(df, "model", eff), leave_one_out(df, "task_id", eff)
    lomo.to_csv(OUT / "model_influence.csv", index=False)
    loto.to_csv(OUT / "task_influence.csv", index=False)
    res["lomo"] = {c: [float(lomo[c].min()), float(lomo[c].max())] for c in ["PROV"] + FAULTS}
    res["loto"] = {c: [float(loto[c].min()), float(loto[c].max())] for c in ["PROV"] + FAULTS}
    res["per_model"] = {m: eff(g) for m, g in df.groupby("model")}

    # false discoveries on independent null seeds
    nul = ref[ref.kind == "null"]
    fdt = nul.groupby("task_id").false_discovery.sum()
    res["false_discoveries_ref"] = dict(n=int(nul.false_discovery.sum()), runs=len(nul), tasks_with_fd=int((fdt > 0).sum()),
                                        n_null_tasks=int(len(fdt)), max_per_task=int(fdt.max()),
                                        evidence_supported=int(nul[nul.false_discovery].independent_evidence_supported.fillna(False).astype(bool).sum()),
                                        reference_agent_fd_tasks=sorted(bref[(bref.kind == "null") & (bref.decision == "discover")].task_id))

    # fault manifestation
    fm = []
    for c in FAULTS:
        g = df[df.condition == c]
        row = dict(condition=c, runs=len(g), triggered=int(g.fault_triggered.sum()), visible=int(g.fault_visible.sum()))
        if c == "SEP_READ":
            row.update(used=int(g.heldout_accessed.sum()), used_before_lock=int(g.fault_heldout_read_before_lock.fillna(False).astype(bool).sum()))
        else:
            row.update(last_pre_lock_output_faulty=int(g.fault_last_pre_lock_output_faulty.fillna(False).astype(bool).sum()),
                       mean_fault_events=float(g.fault_n_fault_events.mean()))
        trig = g[g.fault_triggered.astype(bool)]
        row["joint_valid_triggered"] = float(trig.joint_valid_success.mean()) if len(trig) else None
        row["joint_valid_not_triggered"] = float(g[~g.fault_triggered.astype(bool)].joint_valid_success.mean()) if (~g.fault_triggered.astype(bool)).any() else None
        fm.append(row)
    res["fault_manifestation"] = fm
    pd.DataFrame(fm).to_csv(OUT / "fault_manifestation_v2.csv", index=False)
    res["confirm_rejections"] = df.groupby("condition").n_confirm_rejected.sum().to_dict()

    models = sorted(df.model.unique())
    ranks = {"PROV_GRADE": rank_analysis(ref, ref, tasks, models, "joint_valid_success", "credited_PROV_GRADE", idx)}
    for c in FAULTS:
        ranks[c] = rank_analysis(ref, df[df.condition == c], tasks, models, "joint_valid_success", "joint_valid_success", idx)
    res["rankings"] = ranks
    (OUT / "summary_v2.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: v for k, v in res.items() if k != "rankings"}, indent=1, default=str))
    for c, v in ranks.items():
        print(c, round(v["tau_b"], 3), round(v["tau_b_lo"], 3), round(v["tau_b_hi"], 3), v["flipped_pairs"], v["point_ref"], v["point_alt"])


if __name__ == "__main__":
    main()
