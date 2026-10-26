"""Zero-cost re-analysis of the original experiment (1,680 runs; 420 REF traces regraded under PROV).

    python -m research_v2.analysis_v1      # writes research_v2/results/v1/*
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .stats import B, boot_index, holm, signflip_p, leave_one_out, mean_ci, per_task, rank_analysis, ratio_ci

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "v1"
REPO = HERE.parent
ENV = ["LEAK", "SILENT", "STALE"]
DOMINANT_NULL = ["clinical-null-1009", "ecology-null-1011"]


def load():
    df = pd.read_csv(HERE / "results" / "v1_per_run.csv")
    df["kind"] = df.task_id.str.split("-").str[1]  # "null" would otherwise parse as NA
    tasks = df[["task_id", "kind"]].drop_duplicates().sort_values("task_id").reset_index(drop=True)
    return df, tasks


def paired(df, cfg, col_ref="valid_success", col_cfg="valid_success"):
    a = df[df.config == "REF"].set_index(["model", "task_id", "repeat"])[col_ref].astype(float)
    b = df[df.config == cfg].set_index(["model", "task_id", "repeat"])[col_cfg].astype(float)
    d = (b - a).dropna().rename("diff").reset_index()
    return d


def prov_relation(r) -> str:
    if r.confirm_failed or not r.has_confirm:
        return "failed_or_missing_confirm"
    p = r.agent_printed_p
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "no_printed_p"
    return "contradicts_evidence" if p < 0.05 else "consistent_label_mismatch"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df, tasks = load()
    idx = boot_index(tasks)
    res = dict(n_runs=len(df), n_tasks=len(tasks), bootstrap_B=B)
    ref = df[df.config == "REF"].copy()

    # ---------------- A. truth / evidence / credit decomposition on REF traces
    ref["truth"] = ref.truth_correct.map({True: "correct", False: "incorrect"}).fillna("undetermined")
    ref["ind"] = ref.independent_evidence.map({True: "supported", False: "unsupported"}).fillna("undetermined")
    ref["agent"] = ref.agent_evidence.map({True: "supported", False: "unsupported"})
    res["ref_decisions"] = ref.decision.value_counts().to_dict()
    res["ref_matrix_independent"] = pd.crosstab(ref.truth, ref.ind).to_dict()
    res["ref_matrix_agent"] = pd.crosstab(ref.truth, ref.agent).to_dict()
    res["ref_matrix_by_decision"] = {str(k): v for k, v in
                                     pd.crosstab([ref.decision, ref.truth], [ref.agent, ref.ind]).stack([0, 1]).to_dict().items() if v}

    nc = ref[ref.measured_PROV & ~ref.measured_REF].copy()
    nc["relation"] = nc.apply(prov_relation, axis=1)
    res["prov_newly_credited"] = dict(
        n=len(nc), by_relation=nc.relation.value_counts().to_dict(), by_model=nc.model.value_counts().to_dict(),
        by_task=nc.task_id.value_counts().to_dict(), by_decision=nc.decision.value_counts().to_dict(),
        by_model_relation={f"{m}|{r}": int(v) for (m, r), v in nc.groupby(["model", "relation"]).size().items()})
    nc[["model", "task_id", "repeat", "decision", "answer", "agent_printed_answer", "agent_printed_p",
        "hypothesis", "independent_p", "n_tested_exposures", "confirm_failed", "relation"]] \
        .to_csv(OUT / "prov_newly_credited.csv", index=False)
    unsup = ref[~ref.agent_evidence & ref.decision.ne("missing")].copy()
    unsup["relation"] = unsup.apply(prov_relation, axis=1)
    res["ref_unsupported_by_agent_evidence"] = dict(n=len(unsup), by_relation=unsup.relation.value_counts().to_dict(),
                                                    truth_correct=int(unsup.v1_correct.sum()))

    # ---------------- evidence audit: agent-authored vs independent evidence (all configs)
    det = df[df.independent_evidence.notna() & df.has_confirm]
    agree = (det.agent_evidence == det.independent_evidence.astype(bool))
    dis = det[~agree]
    res["evidence_audit"] = dict(
        n_runs=len(df), n_with_confirm=int(df.has_confirm.sum()),
        n_analyzable=len(det), n_unrecoverable=int((df.has_confirm & df.independent_evidence.isna()).sum()),
        unrecoverable_by_decision=df[df.has_confirm & df.independent_evidence.isna()].decision.value_counts().to_dict(),
        agreement=float(agree.mean()),
        agent_yes_independent_no=int((dis.agent_evidence & ~dis.independent_evidence.astype(bool)).sum()),
        agent_no_independent_yes=int((~dis.agent_evidence & dis.independent_evidence.astype(bool)).sum()),
        disagreements_credited_by_ref_grader=int(dis.measured_REF.sum()),
        disagreements_by_config=dis.config.value_counts().to_dict(),
        disagreement_relation=dis.apply(prov_relation, axis=1).value_counts().to_dict(),
        discover_agree=float(agree[det.decision == "discover"].mean()),
        none_agree=float(agree[det.decision == "none"].mean()),
    )
    single = df[(df.n_tested_exposures == 1) & df.agent_printed_p.notna() & df.hypothesis.notna()]
    res["evidence_audit"]["single_test_significance_agreement"] = dict(
        n=len(single), agree=float(((single.agent_printed_p < .05) == (single.independent_p < .05)).mean()))
    dis.to_csv(OUT / "evidence_disagreements.csv", index=False)

    # multi-exposure confirmation scripts
    hc = df[df.has_confirm]
    res["multi_exposure_confirm"] = dict(
        overall=float((hc.n_tested_exposures > 1).mean()),
        separate_fits=float(hc.multi_fit_distinct.mean()),
        no_fit_recorded=float((hc.n_fits == 0).mean()),
        by_model=hc.groupby("model").apply(lambda g: dict(any=float((g.n_tested_exposures > 1).mean()),
                                                          separate=float(g.multi_fit_distinct.mean()))).to_dict(),
        by_config=hc.groupby("config").apply(lambda g: float((g.n_tested_exposures > 1).mean())).to_dict(),
    )
    disc = df[df.decision == "discover"]
    res["fd_by_confirm_type"] = disc.groupby(disc.multi_fit_distinct).apply(
        lambda g: dict(n=len(g), fd_rate=float((~g.truth_correct.astype(bool)).mean()))).to_dict()

    # ---------------- B. primary and secondary estimands
    ref["prov_infl"] = ref.measured_PROV.astype(float) - ref.measured_REF.astype(float)
    est = {"PROV_same_trace": mean_ci(ref, tasks, "prov_infl", idx)}
    fixed = pd.DataFrame(json.loads((REPO / "results" / "main" / "fixed_replay.json").read_text())["rows"])
    fixed["task_id"] = fixed["task"]
    for cfg in ENV:
        d = paired(df, cfg)
        est[f"{cfg}_fresh_valid"] = ratio_ci(*per_task(d, tasks, "diff"), idx)
        est[f"{cfg}_fresh_valid"]["perm_p"] = signflip_p(*per_task(d, tasks, "diff"))
        dm = paired(df, cfg, "measured_REF", f"measured_{cfg}")
        est[f"{cfg}_fresh_measured"] = ratio_ci(*per_task(dm, tasks, "diff"), idx)
        f = fixed[fixed.config == cfg].copy()
        f["diff"] = f.measured.astype(float) - f.orig_measured.astype(float)
        est[f"{cfg}_fixed_action"] = ratio_ci(*per_task(f, tasks, "diff"), idx)
    hp = holm({c: est[f"{c}_fresh_valid"]["perm_p"] for c in ENV})
    for c in ENV:
        est[f"{c}_fresh_valid"]["holm_p"] = hp[c]
    res["estimands"] = est

    def effects(sub):
        out = {}
        r = sub[sub.config == "REF"]
        out["PROV"] = float((r.measured_PROV.astype(float) - r.measured_REF.astype(float)).mean())
        for cfg in ENV:
            out[cfg] = float(paired(sub, cfg)["diff"].mean())
        return out

    lomo = leave_one_out(df, "model", effects)
    loto = leave_one_out(df, "task_id", effects)
    lomo.to_csv(OUT / "model_influence.csv", index=False)
    loto.to_csv(OUT / "task_influence.csv", index=False)
    res["leave_one_model_out"] = {c: dict(min=float(lomo[c].min()), max=float(lomo[c].max()),
                                          argmax_influence=str(lomo.loc[(lomo[c] - effects(df)[c]).abs().idxmax(), "left_out"]))
                                  for c in ["PROV"] + ENV}
    res["leave_one_task_out"] = {c: dict(min=float(loto[c].min()), max=float(loto[c].max()),
                                         sign_changes=int((np.sign(loto[c]) != np.sign(effects(df)[c])).sum()),
                                         argmax_influence=str(loto.loc[(loto[c] - effects(df)[c]).abs().idxmax(), "left_out"]))
                                 for c in ["PROV"] + ENV}
    excl = df[~df.task_id.isin(DOMINANT_NULL)]
    tasks_ex = tasks[~tasks.task_id.isin(DOMINANT_NULL)].reset_index(drop=True)
    idx_ex = boot_index(tasks_ex)
    r_ex = excl[excl.config == "REF"].copy()
    r_ex["prov_infl"] = r_ex.measured_PROV.astype(float) - r_ex.measured_REF.astype(float)
    res["excluding_dominant_null_tasks"] = dict(
        PROV=mean_ci(r_ex, tasks_ex, "prov_infl", idx_ex),
        **{cfg: ratio_ci(*per_task(paired(excl, cfg), tasks_ex, "diff"), idx_ex) for cfg in ENV})
    per_model = {}
    for m in sorted(df.model.unique()):
        sub = df[df.model == m]
        per_model[m] = dict(n_prov_new=int((sub[sub.config == "REF"].measured_PROV & ~sub[sub.config == "REF"].measured_REF).sum()),
                            **{k: round(v, 4) for k, v in effects(sub).items()})
    res["per_model_effects"] = per_model

    # ---------------- C. rankings
    models = sorted(df.model.unique())
    ranks = {"PROV": rank_analysis(ref, ref, tasks, models, "valid_success", "measured_PROV", idx)}
    for cfg in ENV:
        ranks[cfg] = rank_analysis(ref, df[df.config == cfg], tasks, models, "valid_success", f"measured_{cfg}", idx)
    res["rankings"] = ranks
    pd.DataFrame([dict(config=c, tau_b=v["tau_b"], lo=v["tau_b_lo"], hi=v["tau_b_hi"],
                       p_identical=v["p_identical_order"], flipped=";".join(v["flipped_pairs"]))
                  for c, v in ranks.items()]).to_csv(OUT / "rank_bootstrap.csv", index=False)

    # ---------------- D. false-discovery concentration
    dgp = json.loads((HERE / "results" / "dgp_validation_summary.json").read_text())["original_tasks"]
    base = {t["task_id"]: t for t in dgp}
    nul = df[df.kind == "null"]
    fd_rows = []
    for tid, g in nul.groupby("task_id"):
        fds = g[g.decision == "discover"]
        fd_rows.append(dict(task_id=tid, runs=len(g), false_discoveries=len(fds),
                            fd_exposures=";".join(sorted(fds.answer.unique())),
                            fd_independently_supported=int(fds.independent_evidence.fillna(False).astype(bool).sum()),
                            trusted_baseline=base[tid]["trusted_decision"],
                            confirm_sig_exposures=";".join(base[tid]["confirm_sig_exposures"]),
                            explore_min_exposure=base[tid]["explore_min_exposure"],
                            explore_min_p=base[tid]["explore_min_p"],
                            confirm_p_of_explore_min=base[tid]["confirm_p_of_explore_min"]))
    fdt = pd.DataFrame(fd_rows).sort_values("false_discoveries", ascending=False)
    fdt.to_csv(OUT / "fd_by_task.csv", index=False)
    total_fd = int(fdt.false_discoveries.sum())
    nul_ex = nul[~nul.task_id.isin(DOMINANT_NULL)]
    res["false_discoveries"] = dict(
        total=total_fd, null_runs=len(nul), rate=total_fd / len(nul),
        top2=int(fdt.false_discoveries.head(2).sum()), top2_tasks=fdt.task_id.head(2).tolist(),
        independently_supported=int(fdt.fd_independently_supported.sum()),
        trusted_baseline_false_discoveries_on_original_nulls=int(sum(1 for t in dgp if t["kind"] == "null" and t["trusted_decision"] != "none")),
        rate_excluding_top2=float((nul_ex.decision == "discover").mean()),
        by_config=nul.groupby("config").apply(lambda g: float((g.decision == "discover").mean())).to_dict(),
        by_config_excluding_top2=nul_ex.groupby("config").apply(lambda g: float((g.decision == "discover").mean())).to_dict(),
        decoy_named=int((nul.decision == "discover").sum() - 0) and int(sum(
            1 for _, r in nul[nul.decision == "discover"].iterrows() if r.answer == base[r.task_id].get("decoy"))),
    )

    # ---------------- E. fault manifestation (original environment defects)
    fm = []
    for cfg, g in df[df.config.isin(ENV)].groupby("config"):
        row = dict(config=cfg, runs=len(g))
        if cfg == "LEAK":
            row.update(triggered=len(g), visible=int(g.heldout_listed.sum()), used=int(g.peeked_confirm.sum()),
                       repeated_confirm=int((g.n_confirm > 1).sum()))
        elif cfg == "SILENT":
            row.update(triggered=int((g.n_errors_actual > 0).sum()), visible=int((g.n_errors_hidden > 0).sum()),
                       used=int(g.confirm_failed.sum()))
        else:
            row.update(triggered=int((g.n_stale > 0).sum()), visible=int((g.n_stale > 0).sum()),
                       used=int(g.cited_confirm_stale.sum()))
        fm.append(row)
    pd.DataFrame(fm).to_csv(OUT / "fault_manifestation_v1.csv", index=False)
    res["fault_manifestation"] = fm

    (OUT / "summary_v1.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: res[k] for k in ("prov_newly_credited", "evidence_audit", "estimands",
                                          "leave_one_model_out", "leave_one_task_out",
                                          "excluding_dominant_null_tasks", "false_discoveries", "fault_manifestation")},
                     indent=1, default=str))
    for c, v in ranks.items():
        print(c, round(v["tau_b"], 3), round(v["tau_b_lo"], 3), round(v["tau_b_hi"], 3), v["p_identical_order"], v["flipped_pairs"])


if __name__ == "__main__":
    main()
