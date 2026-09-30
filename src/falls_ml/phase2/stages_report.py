"""Stages S14-S16: feature consensus, reporting (tables, figures, Hebrew management summary, scientific summary) and the non-identifying
share package (planning/EXPERIMENT_PLAN.md §7, §11; REVIEW_DECISIONS G-08). Every number printed comes from a committed table; nothing is
typed in. Wording rules: never "significant", no causal language, validation numbers labelled confirmatory or descriptive.

The three categories are never mixed (reviews/ASTRA_FINAL_RESOLUTION.md): HISTORICAL_BASELINE_15 (the benchmark), SAFE_DISCOVERY (primary;
proven provenance only) and EXPLORATORY_UNRESOLVED_SENSITIVITY (separate tables and sections). The management waterfall reports every step
on the SAME VALIDATION rows at matched capacities."""

from __future__ import annotations

import getpass
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from falls_ml.phase2 import WATERMARK, WATERMARK_HE
from falls_ml.phase2 import durable as D
from falls_ml.phase2.context import Ctx
from falls_ml.phase2.featuresets import ALL_SAFE, EXPL_ALL, EXPL_B15_SAFE, EXPLORATORY, HISTORICAL, SAFE_DISCOVERY
from falls_ml.phase2.stages_models import BENCHMARK, SAFE_REF, config_set, configurations, lasso_config, selection
from falls_ml.phase2.state import Phase2Stop

BANNED = ("significant", "significantly", "excellent", "breakthrough", "production ready", "production-ready", "clinically proven", "causes", "caused by",
          "מובהק", "מצוין", "מעולה", "פריצת דרך", "מוכח קלינית", "מוכן לייצור", "גורם לנפילה", "גורמים לנפילה")
SHARE_TABLES = {   # share file -> (artifact source, suppression rule); rows whose small cells would be re-derivable are suppressed whole
    "OPERATIONAL_CAPACITY.csv": ("OPERATIONAL_CAPACITY.csv", "confusion"),
    "THRESHOLD_RESULTS.csv": ("THRESHOLD_RESULTS.csv", "confusion"),
    "SUBGROUP_SUMMARY.csv": ("SUBGROUP_SUMMARY.csv", "subgroup"),
    "CALIBRATION_CURVES.csv": ("CALIBRATION_CURVES.csv", "calibration"),
    "DOMAIN_INCREMENTAL_GAIN.csv": ("DOMAIN_INCREMENTAL_GAIN.csv", "capacity_counts"),
}
CONFUSION = ("tp", "fp", "fn", "tn")
CONFUSION_DERIVED = ("sensitivity", "specificity", "ppv", "npv", "lift", "false_alerts", "false_alert_pct")
CATEGORY_HE = {HISTORICAL: "היסטורי (15 המשתנים)", SAFE_DISCOVERY: "בטוח – תזמון מאומת", EXPLORATORY: "חקרני – תזמון לא מאומת"}
BENCHMARK_LABEL = "BASELINE_15 refit (15 historical predictors, Phase 2 protocol, TRAIN only; not the recalibrated Phase 1 model)"
SCOPE = ("Scope: exploratory prediction of RECORDED falls/fractures within 180 days in the D00_CLEAN selected cohort (inclusion depended on "
         "information after the prediction time); VALIDATION was reused in Phase 1. Classifications below are pre-declared point-estimate "
         "screens, not formal superiority or non-inferiority tests.")


def _fmt(v: Any, d: int = 3) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "–"
    return "–" if not np.isfinite(f) else f"{f:.{d}f}"


def _pct(v: Any, d: int = 1) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "–"
    return "–" if not np.isfinite(f) else f"{100 * f:.{d}f}%"


def _signed(v: Any, d: int = 0) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "–"
    return "–" if not np.isfinite(f) else f"{f:+.{d}f}"


def _art(ctx: Ctx, name: str) -> pd.DataFrame:
    p = ctx.run.artifacts / name
    return pd.read_csv(p) if p.exists() and p.stat().st_size > 1 else pd.DataFrame()


def cap_tag(c: float) -> str:
    return f"top{round(100 * float(c)):g}"


# ============================================================================ S14 consensus (SAFE features) + exploratory table (UNRESOLVED)
RECOMMENDATION_TEXT = {
    "STRONG": "carry forward (SAFE; consistent incremental evidence) - confirm on a new temporal holdout",
    "MODERATE": "carry forward as a candidate (SAFE; moderate evidence)",
    "WEAK": "not prioritised (weak evidence; may be redundant with other features - not shown to be useless)",
    "NONE": "no incremental evidence in this analysis (association may be carried by other features - not shown to be useless)",
}


def s14_consensus(ctx: Ctx) -> None:
    st = ctx.run.stage("14", "consensus", ("13",))
    if st.complete_record():
        return
    c = ctx.cfg["consensus"]

    def fn(tmp: Path, seed: int) -> dict[str, Any]:
        er = _art(ctx, "02_ENGINEERED_FEATURE_REGISTRY.csv").set_index("feature")
        uni = _art(ctx, "03_UNIVARIATE_SCREEN.csv").set_index("feature")
        q = _art(ctx, "05_FEATURE_QUALITY.csv").set_index("feature")
        stab = _art(ctx, "STABILITY.csv")
        shap = _art(ctx, "SHAP_SUMMARY.csv")
        perm = _art(ctx, "PERMUTATION_IMPORTANCE.csv")
        abl = _art(ctx, "ABLATION_RESULTS.csv")
        oofc = _art(ctx, "OOF_MODEL_COMPARISON.csv").set_index("config")
        cfgs = configurations(ctx)

        def final(config: str) -> dict[str, Any]:
            if config not in cfgs or cfgs[config]["failed"]:
                return {}
            return ctx.run.stage(*cfgs[config]["stage"]).result(cfgs[config]["final"])

        lasso_cfg = lasso_config(ctx, ALL_SAFE)
        lasso_sel = set(final(lasso_cfg).get("selected") or [])
        enet_sel = set(final(f"ENET:{ALL_SAFE}").get("selected") or [])
        xgb_imp = {k: v for k, v in (final(f"XGB_TUNED:{ALL_SAFE}").get("importance") or {}).items() if v > 0}
        xgb_rank = {f: i + 1 for i, f in enumerate(sorted(xgb_imp, key=lambda k: (-xgb_imp[k], k)))}
        expl_sel = set(final(lasso_config(ctx, EXPL_ALL)).get("selected") or [])
        stab_all = stab[stab["feature_set"] == ALL_SAFE].set_index("feature") if len(stab) else pd.DataFrame()
        shap_i = shap.set_index("feature") if len(shap) else pd.DataFrame()
        perm_i = perm.set_index("feature") if len(perm) else pd.DataFrame()
        ind = abl[abl["analysis"] == "individual"].set_index("removed") if len(abl) and "analysis" in abl else pd.DataFrame()
        loo = abl[abl["analysis"] == "leave_one_domain_out"].set_index("removed") if len(abl) and "analysis" in abl else pd.DataFrame()
        in_all = set(ctx.fs().sets[ALL_SAFE]["features"])

        def uni_ev(name: str) -> tuple[float, float, float, float, bool]:
            u = uni.loc[name] if name in uni.index else None
            smd, auc = (u["smd"], u["univariate_auroc"]) if u is not None else (np.nan, np.nan)
            lo, hi = (u["univariate_auroc_ci_low"], u["univariate_auroc_ci_high"]) if u is not None else (np.nan, np.nan)
            U = bool((pd.notna(smd) and abs(smd) >= float(c["univariate_min_abs_smd"])) or
                     (pd.notna(auc) and not float(c["univariate_auroc_band"][0]) <= auc <= float(c["univariate_auroc_band"][1])))
            return smd, auc, lo, hi, U

        def caution(f: Any) -> list[str]:
            out = []
            if f.missing == "not_assessed" or f.form_indicator:
                out.append("assessment-conditional: also reflects who was assessed (care process); transport depends on assessment practice")
            if any(k in (f.text or "").lower() for k in ("latest", "previous", "worse", "current", "at index", "group")) or f.op == "copy" and \
                    any(k in " ".join(f.inputs).lower() for k in ("group", "worsened", "prev", "status", "at_index")):
                out.append("status / latest-value field: the record date bounds the record, not later updates of its value (warehouse to confirm; Astra F-10)")
            if "NAME_ONLY" in str(er.at[f.name, "meaning_status"]) or "UNKNOWN" in str(er.at[f.name, "meaning_status"]):
                out.append("meaning not documented (SME review)")
            clus = q.at[f.name, "redundancy_cluster_safe"] if f.name in q.index and "redundancy_cluster_safe" in q.columns else ""
            if isinstance(clus, str) and clus.strip():
                out.append(f"redundancy cluster {clus} (representative {q.at[f.name, 'representative_safe']})")
            return out

        rows, xrows = [], []
        for f in ctx.cat.features:
            if f.name not in er.index:
                continue
            elig = er.at[f.name, "eligibility"]
            smd, auc, lo, hi, U = uni_ev(f.name)
            miss = float(er.at[f.name, "missing_pct_train"])
            if elig == "ELIGIBLE":
                stab_v = float(stab_all.at[f.name, "selection_frequency"]) if len(stab_all) and f.name in stab_all.index else np.nan
                sign_v = float(stab_all.at[f.name, "sign_positive_share"]) if len(stab_all) and f.name in stab_all.index and pd.notna(stab_all.at[f.name, "sign_positive_share"]) else np.nan
                # L: LASSO selection backed by LASSO stability (the stability refits are LASSO only; an elastic-net-only selection is reported
                # but not certified - Astra F-13)
                L = bool(f.name in lasso_sel and (np.isnan(stab_v) or stab_v >= float(c["linear_min_stability"])))
                shap_rank = int(shap_i.at[f.name, "rank"]) if len(shap_i) and f.name in shap_i.index else None
                perm_v = float(perm_i.at[f.name, "delta_ap_mean"]) if len(perm_i) and f.name in perm_i.index else np.nan
                # I: SHAP rank of the promoted XGBoost; a positive permutation loss alone is weak support only (Astra F-14)
                I = bool(shap_rank is not None and shap_rank <= int(c["shap_top_rank"]))
                I_weak = bool(np.isfinite(perm_v) and perm_v > 0)
                dom_cfg = lasso_config(ctx, f"SAFE_BASE_PLUS_{f.domain}") if f"SAFE_BASE_PLUS_{f.domain}" in ctx.fs().sets else None
                dom_gain = (float(oofc.at[dom_cfg, "oof_delta_ap_vs_safe_base"]) if dom_cfg and dom_cfg in oofc.index
                            and pd.notna(oofc.at[dom_cfg, "oof_delta_ap_vs_safe_base"]) else np.nan)
                A_dom = bool(np.isfinite(dom_gain) and dom_gain >= float(c["domain_min_delta_ap"]))
                ind_v = float(ind.at[f.name, "delta_ap_full_minus_ablated"]) if len(ind) and f.name in ind.index and "delta_ap_full_minus_ablated" in ind else np.nan
                ind_lo = float(ind.at[f.name, "delta_ap_ci_low_full_minus_ablated"]) if len(ind) and f.name in ind.index and "delta_ap_ci_low_full_minus_ablated" in ind else np.nan
                ind_hi = float(ind.at[f.name, "delta_ap_ci_high_full_minus_ablated"]) if len(ind) and f.name in ind.index and "delta_ap_ci_high_full_minus_ablated" in ind else np.nan
                A_feat = bool(np.isfinite(ind_v) and ind_v >= float(c["ablation_min_delta_ap"]))
                loo_v = float(loo.at[f.domain, "delta_ap_full_minus_ablated"]) if len(loo) and f.domain in loo.index and "delta_ap_full_minus_ablated" in loo else np.nan
                if A_feat and (L or I):
                    rating = "STRONG"
                elif ((L or I) and (U or A_dom)) or A_feat:
                    rating = "MODERATE"
                elif U or L or I or I_weak or A_dom or f.name in enet_sel:
                    rating = "WEAK"
                else:
                    rating = "NONE"
                caut = caution(f)
                if f.name in enet_sel and f.name not in lasso_sel:
                    caut.append("selected by elastic net only (not covered by the LASSO stability refits)")
                if f.name not in in_all:
                    rep = q.at[f.name, "representative_safe"] if f.name in q.index and "representative_safe" in q.columns else ""
                    caut.append(f"not in ALL_REVIEWED_SAFE: redundant with {rep} (|Spearman| >= {ctx.cfg['screening']['redundancy_abs_spearman']})")
                rows.append({
                    "category": SAFE_DISCOVERY, "feature": f.name, "text": f.text, "source": er.at[f.name, "sources"], "domain": f.domain,
                    "efalls_vs_native": "eFalls concept" if f.mapping_class.startswith("EFALLS") else "Meuhedet-native", "efalls_concept": f.efalls_concept or "",
                    "provenance_status": er.at[f.name, "provenance_status"], "missing_pct_train": miss, "coverage_pct_train": round(100.0 - miss, 3),
                    "univariate_smd": smd, "univariate_auroc": auc, "univariate_auroc_ci_low": lo, "univariate_auroc_ci_high": hi, "evidence_U_univariate": U,
                    "in_all_reviewed_safe": f.name in in_all, "lasso_selected": f.name in lasso_sel, "lasso_stability": stab_v, "lasso_sign_positive_share": sign_v,
                    "elasticnet_selected": f.name in enet_sel, "evidence_L_linear": L, "xgb_gain_rank": xgb_rank.get(f.name),
                    "xgb_used": f.name in xgb_imp, "xgb_shap_rank": shap_rank, "xgb_permutation_delta_ap": perm_v, "evidence_I_importance": I,
                    "domain_oof_delta_ap_vs_safe_base": dom_gain, "evidence_A_domain": A_dom, "domain_unique_oof_delta_ap_loo": loo_v,
                    "ablation_delta_ap_full_minus_ablated": ind_v, "ablation_delta_ap_ci_low": ind_lo, "ablation_delta_ap_ci_high": ind_hi,
                    "evidence_A_feature": A_feat, "evidence_I_weak_permutation_only": I_weak, "recommendation": rating,
                    "evidence_label": "exploratory predictive evidence (not causal; conditional on the other features)",
                    "final_recommendation": RECOMMENDATION_TEXT[rating],
                    "interpretation_caution": "; ".join(caut) or "none flagged"})
            elif elig == "ELIGIBLE_EXPLORATORY":
                sig = bool(U or f.name in expl_sel)
                xrows.append({
                    "category": EXPLORATORY, "feature": f.name, "text": f.text, "source": er.at[f.name, "sources"], "domain": f.domain,
                    "efalls_vs_native": "eFalls concept" if f.mapping_class.startswith("EFALLS") else "Meuhedet-native",
                    "provenance_status": "UNRESOLVED", "why_unresolved": er.at[f.name, "eligibility_reason"], "missing_pct_train": miss,
                    "coverage_pct_train": round(100.0 - miss, 3), "univariate_smd": smd, "univariate_auroc": auc, "evidence_U_univariate": U,
                    "selected_in_exploratory_lasso": f.name in expl_sel,
                    "exploratory_status": ("EXPLORATORY SIGNAL - needs proven record dates in the next extract before any use" if sig
                                           else "no exploratory signal in this analysis"),
                    "interpretation_caution": "timing UNRESOLVED: never part of a SAFE model; " + ("; ".join(caution(f)) or "")})
        t = pd.DataFrame(rows)
        order = {"STRONG": 0, "MODERATE": 1, "WEAK": 2, "NONE": 3}
        if len(t):
            t = t.sort_values(["recommendation", "ablation_delta_ap_full_minus_ablated", "feature"], key=lambda s: s.map(order) if s.name == "recommendation" else s,
                              ascending=[True, False, True]).reset_index(drop=True)
        x = pd.DataFrame(xrows)
        if len(x):
            x = x.sort_values(["selected_in_exploratory_lasso", "feature"], ascending=[False, True]).reset_index(drop=True)
        D.write_csv(tmp / "FEATURE_CONSENSUS.csv", t)
        D.write_csv(tmp / "EXPLORATORY_UNRESOLVED_FEATURES.csv", x)
        return {"n_safe_features": int(len(t)), "n_unresolved_features": int(len(x)),
                "ratings": t["recommendation"].value_counts().to_dict() if len(t) else {}}

    res = st.item("consensus", fn)
    outs = [D.write_bytes(ctx.run.artifacts / n, (st.path("consensus") / n).read_bytes()) for n in ("FEATURE_CONSENSUS.csv", "EXPLORATORY_UNRESOLVED_FEATURES.csv")]
    st.finalize(outs, res)


# ============================================================================ S15 report
def _model_comparison(ctx: Ctx) -> pd.DataFrame:
    oofc = _art(ctx, "OOF_MODEL_COMPARISON.csv")
    val = _art(ctx, "VALIDATION_MODEL_COMPARISON.csv")
    keep = ["config", "category", "family", "feature_set", "set_kind", "n_baseline_features", "n_new_features_offered", "n_features_used", "failed",
            "qualifies_vs_safe_base"]
    t = oofc[[c for c in keep if c in oofc.columns] + [c for c in oofc.columns if c.startswith("oof_")]]
    if len(val):
        t = t.merge(val[[c for c in val.columns if c.startswith("val_") or c in ("config", "role")]], on="config", how="left")
    t = t.copy()
    t.insert(1, "basis", "TRAIN outer-fold OOF (selection basis) + VALIDATION (one-shot; confirmatory only for the pre-declared comparisons); TEST not used")
    cat_order = {HISTORICAL: 0, SAFE_DISCOVERY: 1, EXPLORATORY: 2}
    return t.sort_values(["category", "config"], key=lambda s: s.map(cat_order) if s.name == "category" else s).reset_index(drop=True)


def _capacity_index(cap: pd.DataFrame) -> dict[tuple[str, str], pd.Series]:
    out: dict[tuple[str, str], pd.Series] = {}
    for _, r in cap.iterrows():
        out[(str(r["config"]), cap_tag(float(r["capacity"])))] = r
    return out


def _metrics_row(ctx: Ctx, cfg: str, oofc: pd.DataFrame, val: pd.DataFrame, capx: dict[tuple[str, str], pd.Series], caps: list[float]) -> dict[str, Any]:
    fs = ctx.fs()
    s = fs.sets[config_set(cfg)]
    row: dict[str, Any] = {"model": cfg, "category": s["category"], "feature_set": config_set(cfg),
                           "n_predictors_offered": int(s["n_new_features"]) + int(s["n_baseline_features"])}
    if cfg in oofc.index:
        o = oofc.loc[cfg]
        for k in ("ap", "auroc", "brier", "cal_slope", "citl"):
            row[f"oof_{k}"] = o.get(f"oof_{k}")
        for k in ("delta_ap", "delta_ap_ci_low", "delta_ap_ci_high"):
            row[f"oof_{k}_vs_benchmark"] = o.get(f"oof_{k}")
        row["oof_delta_ap_vs_safe_base"] = o.get("oof_delta_ap_vs_safe_base")
    if len(val) and cfg in val.index:
        v = val.loc[cfg]
        for k in ("ap", "auroc", "brier", "scaled_brier", "logloss", "cal_slope", "cal_intercept", "citl", "oe", "n", "events"):
            row[f"val_{k}"] = v.get(f"val_{k}")
        for k in ("delta_ap", "delta_ap_ci_low", "delta_ap_ci_high", "delta_auroc", "delta_brier", "delta_capture_principal",
                  "delta_capture_principal_ci_low", "delta_capture_principal_ci_high"):
            row[f"val_{k}_vs_benchmark"] = v.get(f"val_{k}")
        for k in ("delta_ap", "delta_ap_ci_low", "delta_ap_ci_high"):
            row[f"val_{k}_vs_safe_base"] = v.get(f"val_{k}_vs_safe_base")
        for c in caps:
            tag = cap_tag(c)
            r = capx.get((cfg, tag))
            b = capx.get((BENCHMARK, tag))
            if r is None:
                continue
            row.update({f"val_flagged_{tag}": r["n_flagged"], f"val_falls_identified_{tag}": r["tp"], f"val_capture_{tag}": r["sensitivity"],
                        f"val_ppv_{tag}": r["ppv"], f"val_false_alerts_{tag}": r["fp"],
                        f"val_additional_falls_vs_benchmark_{tag}": (float(r["tp"]) - float(b["tp"])) if b is not None else np.nan,
                        f"val_additional_falls_vs_benchmark_{tag}_ci_low": v.get(f"val_delta_falls_{tag}_ci_low", np.nan) if cfg != BENCHMARK else np.nan,
                        f"val_additional_falls_vs_benchmark_{tag}_ci_high": v.get(f"val_delta_falls_{tag}_ci_high", np.nan) if cfg != BENCHMARK else np.nan,
                        f"val_fewer_false_alerts_vs_benchmark_{tag}": (float(b["fp"]) - float(r["fp"])) if b is not None else np.nan})
    return row


def _domain_table(ctx: Ctx, sel: dict[str, Any]) -> pd.DataFrame:
    """The management waterfall (every step on the SAME VALIDATION rows), one-domain additions and leave-one-domain-out."""
    fs = ctx.fs()
    oofc = _art(ctx, "OOF_MODEL_COMPARISON.csv").set_index("config")
    val = _art(ctx, "VALIDATION_MODEL_COMPARISON.csv")
    val = val.set_index("config") if len(val) else val
    capx = _capacity_index(_art(ctx, "OPERATIONAL_CAPACITY.csv"))
    caps = [float(c) for c in ctx.cfg["metrics"]["capacities"]]
    abl = _art(ctx, "ABLATION_RESULTS.csv")
    loo = abl[abl["analysis"] == "leave_one_domain_out"].set_index("removed") if len(abl) and "analysis" in abl else pd.DataFrame()
    dl = {d: ctx.cat.domains[d]["label"] for d in ctx.cat.domains}
    steps: list[tuple[str, str, str, int | None]] = [(BENCHMARK_LABEL, BENCHMARK, "", None),
                                                     ("SAFE_BASE (SAFE subset of BASELINE_15)", SAFE_REF, "", len(fs.sets["SAFE_BASE"]["baseline"]))]
    added_by_step: dict[str, list[str]] = {}
    prev_feats: set[str] = set()
    for w in fs.design.get("waterfall", []):
        added = " + ".join(dl[d] for d in w.get("domains", [w["added"]])) if w["added"] == "OTHER_NATIVE" else dl.get(w["added"], w["added"])
        label = f"+ {added}" + (" = ALL_SAFE_LINEAR (LASSO)" if w["set"] == ALL_SAFE else "")
        feats = set(fs.sets[w["set"]]["features"])
        added_by_step[label] = sorted(feats - prev_feats)
        prev_feats = feats
        steps.append((label, lasso_config(ctx, w["set"]), w["added"], int(w.get("n_safe_features_added", 0))))
    for label, cfg in (("ALL_SAFE_ELASTIC_NET", f"ENET:{ALL_SAFE}"), ("ALL_SAFE_XGBOOST (tuned)", f"XGB_TUNED:{ALL_SAFE}")):
        if cfg in oofc.index:
            steps.append((label, cfg, "", 0))
    if sel["recommended"] not in [c for _, c, _, _ in steps]:
        steps.append((f"RECOMMENDED SAFE MODEL ({sel['recommended']})", sel["recommended"], "", None))
    rows = []
    prev = None
    principal = cap_tag(float(ctx.cfg["selection"]["principal_capacity"]))
    for k, (label, cfg, dom, n_added) in enumerate(steps):
        r = {"analysis": "management_waterfall", "step": k, "label": label, "domain_added": dom, "n_safe_features_added": n_added,
             "no_safe_feature_in_domain": bool(n_added == 0 and dom not in ("", None)),
             "features_added": "; ".join(added_by_step.get(label, [])) if label in added_by_step else "",
             **_metrics_row(ctx, cfg, oofc, val, capx, caps)}
        if prev is not None and "val_ap" in r and "val_ap" in prev:
            r["val_delta_ap_vs_previous_step"] = float(r["val_ap"]) - float(prev["val_ap"])
            if f"val_capture_{principal}" in r and f"val_capture_{principal}" in prev:
                r[f"val_delta_capture_{principal}_vs_previous_step"] = float(r[f"val_capture_{principal}"]) - float(prev[f"val_capture_{principal}"])
        rows.append(r)
        if k <= len(fs.design.get("waterfall", [])) + 1:
            prev = r
    for d in ctx.cat.domains:
        name = f"SAFE_BASE_PLUS_{d}"
        if name in fs.sets:
            r = {"analysis": "one_domain_added", "step": None, "label": f"SAFE_BASE + {dl[d]}", "domain_added": d,
                 "n_safe_features_added": int(fs.sets[name]["n_new_features"]), "features_added": "; ".join(fs.sets[name]["features"]),
                 **_metrics_row(ctx, lasso_config(ctx, name), oofc, val, capx, caps)}
            if len(loo) and d in loo.index:
                minus = f"{ALL_SAFE}_MINUS_{d}"
                if minus in fs.sets:
                    r["features_removed_in_loo"] = "; ".join(sorted(set(fs.sets[ALL_SAFE]["features"]) - set(fs.sets[minus]["features"])))
                r["unique_contribution_oof_delta_ap_loo"] = loo.at[d, "delta_ap_full_minus_ablated"]
                r["unique_contribution_oof_delta_ap_loo_ci_low"] = loo.at[d, "delta_ap_ci_low_full_minus_ablated"]
                r["unique_contribution_oof_delta_ap_loo_ci_high"] = loo.at[d, "delta_ap_ci_high_full_minus_ablated"]
            rows.append(r)
    return pd.DataFrame(rows)


def _funnel(ctx: Ctx) -> pd.DataFrame:
    from falls_ml.phase2.accounting import funnel_rows
    from falls_ml.phase2.stages_data import baseline_status

    reg = _art(ctx, "01_COLUMN_REGISTRY.csv")
    er = _art(ctx, "02_ENGINEERED_FEATURE_REGISTRY.csv")
    cons = _art(ctx, "FEATURE_CONSENSUS.csv")
    xf = _art(ctx, "EXPLORATORY_UNRESOLVED_FEATURES.csv")
    rows = funnel_rows(reg, er, ctx.fs(), baseline_status(ctx))
    for r in ("STRONG", "MODERATE", "WEAK", "NONE"):
        rows.append({"step": f"SAFE features rated {r} (consensus)", "n": int((cons["recommendation"] == r).sum()) if len(cons) else 0,
                     "category": SAFE_DISCOVERY, "note": RECOMMENDATION_TEXT[r]})
    rows.append({"step": "UNRESOLVED features with an exploratory signal", "n": int(xf["exploratory_status"].str.startswith("EXPLORATORY SIGNAL").sum()) if len(xf) else 0,
                 "category": EXPLORATORY, "note": "never part of a SAFE model; need proven record dates"})
    return pd.DataFrame(rows)


SHORTLIST_COLUMNS = ["feature", "text", "source", "domain", "efalls_vs_native", "provenance_status", "missing_pct_train", "coverage_pct_train",
                     "univariate_smd", "univariate_auroc", "univariate_auroc_ci_low", "univariate_auroc_ci_high", "lasso_selected", "elasticnet_selected",
                     "xgb_used", "xgb_gain_rank", "xgb_permutation_delta_ap", "xgb_shap_rank", "lasso_stability", "lasso_sign_positive_share",
                     "domain_oof_delta_ap_vs_safe_base", "domain_unique_oof_delta_ap_loo", "ablation_delta_ap_full_minus_ablated",
                     "ablation_delta_ap_ci_low", "ablation_delta_ap_ci_high", "interpretation_caution", "recommendation", "final_recommendation"]


def _headline(ctx: Ctx, sel: dict[str, Any], dom: pd.DataFrame) -> list[dict[str, Any]]:
    """The management question per capacity: recommended SAFE model vs the historical benchmark on the same VALIDATION rows."""
    if not len(dom):
        return []
    w = dom[dom["analysis"] == "management_waterfall"].set_index("model")
    rec = sel["recommended"]
    if rec not in w.index or BENCHMARK not in w.index:
        return []
    r, b = w.loc[rec], w.loc[BENCHMARK]
    r = r.iloc[0] if isinstance(r, pd.DataFrame) else r
    b = b.iloc[0] if isinstance(b, pd.DataFrame) else b
    n = float(r.get("val_n") or np.nan)
    principal = cap_tag(float(ctx.cfg["selection"]["principal_capacity"]))
    min_cell = int(ctx.cfg["screening"]["min_cell"])
    out = []
    for c in ctx.cfg["metrics"]["capacities"]:
        tag = cap_tag(float(c))
        if f"val_falls_identified_{tag}" not in r:
            continue
        small = _cap_small(r, tag, min_cell) or _cap_small(b, tag, min_cell)
        add = float(r[f"val_falls_identified_{tag}"]) - float(b[f"val_falls_identified_{tag}"])
        row = {"capacity": float(c), "tag": tag, "flagged": r[f"val_flagged_{tag}"], "falls_new": r[f"val_falls_identified_{tag}"],
               "falls_benchmark": b[f"val_falls_identified_{tag}"], "additional": add, "per_10000": add / n * 10000 if n else np.nan,
               "fa_new": r[f"val_false_alerts_{tag}"], "fa_benchmark": b[f"val_false_alerts_{tag}"],
               "ci_low": r.get(f"val_additional_falls_vs_benchmark_{tag}_ci_low"), "ci_high": r.get(f"val_additional_falls_vs_benchmark_{tag}_ci_high"),
               "events": r.get("val_events"), "n": n, "suppressed": bool(small)}
        if small:   # a confusion cell below the minimum: no count at this capacity leaves the PC
            row.update({k: None for k in ("falls_new", "falls_benchmark", "additional", "per_10000", "fa_new", "fa_benchmark", "ci_low", "ci_high")})
        out.append(row)
    return out


def _cnt(v: Any) -> str:
    return "suppressed (<10)" if v is None or (isinstance(v, float) and not np.isfinite(v)) else str(int(v))


def _figures(ctx: Ctx, figs: Path, dom: pd.DataFrame, mc: pd.DataFrame, sel: dict[str, Any], head: list[dict[str, Any]]) -> list[str]:
    from falls_ml.modelreport import figures as F

    made = []
    wm = "EXPLORATORY – PHASE 2 (TRAIN OOF / VALIDATION) – NOT A FINAL CLAIM"
    # 1 funnel (raw columns -> SAFE discovery)
    fun = _funnel(ctx)
    keep = ["raw columns in the extract (contract)", "engineered candidate features (catalogue)", "engineered features: SAFE",
            "SAFE new features entering the discovery (ALL_REVIEWED_SAFE)", "SAFE features rated STRONG (consensus)", "SAFE features rated MODERATE (consensus)"]
    steps = [(r.step, int(r.n), "") for r in fun.itertuples() if r.step in keep]
    if steps:
        fig = F.funnel(steps, title="Column / feature funnel (SAFE discovery)", lang="en")
        made += list(F.save(fig, figs, "01_feature_funnel", wm).values())
    # 2 management waterfall on the SAME VALIDATION rows: AP and capture at the principal capacity
    principal = cap_tag(float(ctx.cfg["selection"]["principal_capacity"]))
    wf = dom[dom["analysis"] == "management_waterfall"] if len(dom) else dom
    if len(wf) and "val_ap" in wf:
        fig = F._fig(11, 7.2)
        ax1, ax2 = fig.add_subplot(211), fig.add_subplot(212)
        x = np.arange(len(wf))
        colors = [F.MUTED if c == HISTORICAL else F.BLUE for c in wf["category"]]
        for ax in (ax1, ax2):
            F._style(ax)
        ax1.bar(x, wf["val_ap"], color=colors, width=0.62)
        ax1.plot(x, wf["oof_ap"], "o", color=F.ORANGE, ms=5, label="TRAIN out-of-fold AP (selection basis)")
        ax1.set_ylabel("AP = PR-AUC (VALIDATION)")
        ax1.set_xticks(x, [""] * len(x))
        ax1.legend(fontsize=8, frameon=False, loc="upper left")
        ax1.set_title("Where the improvement comes from: every step on the same VALIDATION rows (grey = historical benchmark, blue = SAFE only)",
                      loc="left", fontsize=10)
        col = f"val_capture_{principal}"
        if col in wf:
            mc_ = int(ctx.cfg["screening"]["min_cell"])
            vals = [np.nan if _cap_small(r, principal, mc_) else 100 * float(r[col]) for _, r in wf.iterrows()]
            ax2.bar(x, vals, color=colors, width=0.62)
            for i, v in enumerate(vals):
                if not np.isfinite(v):
                    ax2.text(i, 0, "suppressed\n(<10)", ha="center", va="bottom", fontsize=6.5, color=F.INK2)
            ax2.set_ylabel(f"% of recorded events captured,\ntop {float(ctx.cfg['selection']['principal_capacity']):.0%}")
        ax2.set_xticks(x, [_short_label(r) for _, r in wf.iterrows()], rotation=30, ha="right", fontsize=8)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        made += list(F.save(fig, figs, "02_domain_waterfall", wm).values())
    # 3/4 AUROC and PR-AUC comparison: waterfall end points + shortlist + exploratory (labelled by category)
    show = list(dict.fromkeys([BENCHMARK, SAFE_REF, *sel["shortlist"], lasso_config(ctx, ALL_SAFE), f"ENET:{ALL_SAFE}", f"XGB_TUNED:{ALL_SAFE}",
                               lasso_config(ctx, EXPL_B15_SAFE), lasso_config(ctx, EXPL_ALL)]))
    sub = mc[mc["config"].isin(show)].drop_duplicates("config")
    sub = sub.set_index("config").loc[[c for c in show if c in set(sub["config"])]].reset_index()
    for metric, stem, title in (("auroc", "03_auroc_comparison", "AUROC"), ("ap", "04_ap_comparison", "AP (average precision = step-wise PR-AUC)")):
        if not len(sub):
            break
        fig = F._fig(9.5, 0.5 * len(sub) + 1.8)
        ax = fig.add_subplot(111)
        F._style(ax)
        y = np.arange(len(sub))
        ax.barh(y - 0.18, sub[f"oof_{metric}"], height=0.36, color=F.BLUE, label="TRAIN out-of-fold")
        if f"val_{metric}" in sub:
            ax.barh(y + 0.18, sub[f"val_{metric}"], height=0.36, color=F.ORANGE, label="VALIDATION")
        ax.set_yticks(y, [f"{c}  [{k}]" for c, k in zip(sub["config"], sub["category"])], fontsize=7.5)
        ax.invert_yaxis()
        ax.set_xlabel(title)
        ax.legend(fontsize=8, frameon=False, loc="lower right")
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        made += list(F.save(fig, figs, stem, wm).values())
    # 5 calibration (VALIDATION, shortlist)
    cal = _art(ctx, "CALIBRATION_CURVES.csv")
    smooth = _art(ctx, "CALIBRATION_SMOOTH.csv")
    if len(cal):   # smooth curve with a 95% band (restricted cubic spline) + decile points, shortlist only (Astra F-11)
        mc_ = int(ctx.cfg["screening"]["min_cell"])
        fig = F._fig(7, 5.4)
        ax = fig.add_subplot(111)
        F._style(ax)
        lim = 0.0
        for i, (cfg, g) in enumerate(cal.groupby("config", sort=True)):
            col = F.SERIES[i % len(F.SERIES)]
            g = g[~((g["events"] > 0) & (g["events"] < mc_)) & ~((g["n"] - g["events"] > 0) & (g["n"] - g["events"] < mc_))]
            ax.plot(g["mean_predicted"], g["observed_rate"], "o", ms=3.5, color=col)
            sm = smooth[smooth["config"] == cfg] if len(smooth) else smooth
            if len(sm):
                ax.plot(sm["predicted"], sm["observed_smooth"], lw=1.5, color=col, label=cfg)
                ax.fill_between(sm["predicted"], sm["ci_low"], sm["ci_high"], color=col, alpha=0.12, lw=0)
                lim = max(lim, float(sm["predicted"].max()))
            lim = max(lim, float(g["mean_predicted"].max()) if len(g) else 0.0)
        ax.plot([0, lim * 1.05], [0, lim * 1.05], color=F.MUTED, lw=1, ls="--")
        ax.set_xlabel("predicted risk")
        ax.set_ylabel("observed rate")
        ax.set_title("Calibration, VALIDATION: smooth curve (95% band) + risk deciles; no recalibration", loc="left", fontsize=10)
        ax.legend(fontsize=7, frameon=False)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        made += list(F.save(fig, figs, "05_calibration", wm).values())
    # 6 consensus (SAFE features only)
    cons = _art(ctx, "FEATURE_CONSENSUS.csv")
    if len(cons):
        top = cons[cons["recommendation"].isin(["STRONG", "MODERATE", "WEAK"])].head(25)
        cols = ["evidence_U_univariate", "evidence_L_linear", "evidence_I_importance", "evidence_A_domain", "evidence_A_feature"]
        if len(top):
            fig = F._fig(8, 0.32 * len(top) + 1.8)
            ax = fig.add_subplot(111)
            F._style(ax)
            for j, col in enumerate(cols):
                on = top[col].astype(bool).to_numpy()
                ax.scatter(np.full(on.sum(), j), np.flatnonzero(on), s=60, color=F.BLUE)
                ax.scatter(np.full((~on).sum(), j), np.flatnonzero(~on), s=20, color=F.GRID)
            ax.set_xticks(range(len(cols)), ["univariate", "linear", "importance", "domain gain", "feature ablation"], fontsize=8)
            ax.set_yticks(range(len(top)), [f"{a} [{b}]" for a, b in zip(top["feature"], top["recommendation"])], fontsize=7)
            ax.invert_yaxis()
            ax.set_title("SAFE feature consensus (evidence classes; not a vote)", loc="left", fontsize=11)
            fig.tight_layout(rect=(0, 0.04, 1, 1))
            made += list(F.save(fig, figs, "06_feature_consensus", wm).values())
    # 7 capture and 8 false alerts at matched capacity (shortlist, VALIDATION)
    cap = _art(ctx, "OPERATIONAL_CAPACITY.csv")
    capsh = cap[cap["config"].isin(sel["shortlist"])] if len(cap) else cap
    if len(capsh):   # capacities with a small confusion cell are not drawn (same rule as the shared tables)
        mc_ = int(ctx.cfg["screening"]["min_cell"])
        capsh = capsh[~capsh.apply(lambda r: any(_small(r[c], mc_) for c in CONFUSION), axis=1)]
    if len(capsh):
        fig = F._fig(8, 4.2)
        ax = fig.add_subplot(111)
        F._style(ax)
        for i, (cfg, g) in enumerate(capsh.groupby("config", sort=True)):
            ax.plot(100 * g["capacity"], 100 * g["sensitivity"], marker="o", color=F.SERIES[i % len(F.SERIES)], label=cfg)
        ax.set_xlabel("% of the population flagged (capacity)")
        ax.set_ylabel("% of recorded falls captured")
        ax.set_title("Top-N capacity capture, VALIDATION (same rows, same capacity)", loc="left", fontsize=11)
        ax.legend(fontsize=7, frameon=False)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        made += list(F.save(fig, figs, "07_top_risk_capture", wm).values())
        fig = F._fig(9, 4.0)
        ax = fig.add_subplot(111)
        F._style(ax)
        fr_list = [float(c) for c in ctx.cfg["metrics"]["capacities"] if float(c) >= 0.03]
        labels = sorted(capsh["config"].unique())
        wdt = 0.8 / max(1, len(fr_list))
        for j, fr in enumerate(fr_list):
            vals = [float(m["fp"].iloc[0]) if len(m := capsh[(capsh["config"] == c) & np.isclose(capsh["capacity"], fr)]) else np.nan for c in labels]
            ax.bar(np.arange(len(labels)) + (j - (len(fr_list) - 1) / 2) * wdt, vals, width=wdt, color=F.SERIES[j % len(F.SERIES)], label=f"top {fr:.0%}")
        ax.set_xticks(range(len(labels)), labels, rotation=15, ha="right", fontsize=8)
        ax.set_ylabel("false alerts (flagged, no recorded fall)")
        ax.set_title("False alerts at matched capacity, VALIDATION", loc="left", fontsize=11)
        ax.legend(fontsize=8, frameon=False)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        made += list(F.save(fig, figs, "08_false_alerts_matched_capacity", wm).values())
    # 9 model families on the same SAFE feature set
    fam = mc[mc["feature_set"] == ALL_SAFE]
    if len(fam):
        fig = F._fig(8, 3.8)
        ax = fig.add_subplot(111)
        F._style(ax)
        x = np.arange(len(fam))
        ax.bar(x - 0.2, fam["oof_ap"], width=0.4, color=F.BLUE, label="AP, TRAIN OOF")
        if "val_ap" in fam:
            ax.bar(x + 0.2, fam["val_ap"], width=0.4, color=F.ORANGE, label="AP, VALIDATION")
        ax.set_xticks(x, list(fam["family"]), fontsize=9)
        ax.set_title("Model comparison on the same SAFE feature set (ALL_REVIEWED_SAFE)", loc="left", fontsize=11)
        ax.legend(fontsize=8, frameon=False)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        made += list(F.save(fig, figs, "09_model_families", wm).values())
    # 10 SHAP summary of the promoted XGBoost: mean |SHAP| bars + p05-p95 ranges (aggregated; no individual points)
    shap = _art(ctx, "SHAP_SUMMARY.csv")
    if len(shap):
        top = shap.head(20).iloc[::-1]
        fig = F._fig(10, 0.32 * len(top) + 1.8)
        ax1, ax2 = fig.add_subplot(121), fig.add_subplot(122)
        for ax in (ax1, ax2):
            F._style(ax)
        y = np.arange(len(top))
        ax1.barh(y, top["mean_abs_shap"], color=F.BLUE)
        ax1.set_yticks(y, list(top["feature"]), fontsize=7)
        ax1.set_title("mean |SHAP| (log-odds)", fontsize=10, loc="left")
        ax2.hlines(y, top["p05"], top["p95"], color=F.MUTED, lw=2)
        ax2.hlines(y, top["p25"], top["p75"], color=F.BLUE, lw=5)
        ax2.plot(top["p50"], y, "|", color=F.INK, ms=8)
        ax2.axvline(0, color=F.GRID)
        ax2.set_yticks(y, [""] * len(y))
        ax2.set_title("SHAP distribution: p5-p95, p25-p75, median", fontsize=10, loc="left")
        fig.suptitle("Promoted XGBoost: global SHAP summary (TRAIN sample; not causal)", x=0.01, ha="left", fontsize=11)
        fig.tight_layout(rect=(0, 0.04, 1, 0.95))
        made += list(F.save(fig, figs, "10_xgb_shap_summary", wm).values())
    # 11 the management question: additional recorded falls identified at the same capacity (recommended SAFE model vs BASELINE_15)
    if head:
        fig = F._fig(8, 3.8)
        ax = fig.add_subplot(111)
        F._style(ax)
        x = np.arange(len(head))
        head = [h for h in head if not h.get("suppressed")]
        x = np.arange(len(head))
        vals = [float(h["additional"]) for h in head]
        ax.bar(x, vals, color=[F.BLUE if v >= 0 else F.ORANGE for v in vals], width=0.6)
        for i, h in enumerate(head):
            ax.text(i, vals[i], f"{vals[i]:+.0f}", ha="center", va="bottom" if vals[i] >= 0 else "top", fontsize=9, color=F.INK)
        ax.axhline(0, color=F.GRID)
        ax.set_xticks(x, [f"top {h['capacity']:.0%}\n({int(h['flagged'])} flagged)" for h in head], fontsize=8)
        ax.set_ylabel("additional recorded falls identified")
        ax.set_title(f"Same capacity, same VALIDATION rows: {sel['recommended']} vs BASELINE_15", loc="left", fontsize=10)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        made += list(F.save(fig, figs, "11_additional_falls_same_capacity", wm).values())
    return made


def _death_sentence(facts: dict[str, Any]) -> str:
    """The outcome's death semantics as measured on TRAIN + VALIDATION (Astra F-05), in one sentence."""
    a = facts.get("label_death_audit") or {}
    if not a.get("available"):
        return "- Death semantics of the label could not be audited (death columns absent)."
    n = int(a.get("deaths_within_window") or 0)
    if n == 0:
        return ("- No labelled TRAIN / VALIDATION member died within the 180-day window: the warehouse label censors (excludes) such members, so "
                "the cohort contains no death within the window; a death after the window does not change the label.")
    return (f"- {n} labelled TRAIN / VALIDATION members died within the window; {a.get('events_among_deaths_within_window')} of them carry a "
            f"recorded event (a recorded fall/fracture stays an event when death follows) and {a.get('non_events_among_deaths_within_window')} are "
            "non-events (death before any recorded event).")


def _short_label(r: Any) -> str:
    """A compact axis label of a waterfall row (the full label stays in the tables)."""
    m, lab = str(r["model"]), str(r["label"])
    if m == BENCHMARK:
        return "BASELINE_15 refit"
    if m == SAFE_REF:
        return "SAFE_BASE"
    if "ALL_SAFE_LINEAR" in lab:
        return "+ other domains = ALL_SAFE_LINEAR"
    if lab.startswith("ALL_SAFE_") or lab.startswith("RECOMMENDED"):
        return lab.split(" (")[0]
    return lab if len(lab) <= 32 else lab[:30] + "…"


def _md_table(header: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header), *["| " + " | ".join(r) + " |" for r in rows]]


def _scientific_summary(ctx: Ctx, sel: dict[str, Any], mc: pd.DataFrame, dom: pd.DataFrame, fun: pd.DataFrame, head: list[dict[str, Any]],
                        contr: pd.DataFrame | None = None) -> str:
    facts = ctx.facts()
    conf = json.loads((ctx.run.artifacts / "VALIDATION_CONFIRMATION.json").read_text(encoding="utf-8"))
    opened = json.loads((ctx.run.out / "VALIDATION_OPENED.json").read_text(encoding="utf-8"))
    cons = _art(ctx, "FEATURE_CONSENSUS.csv")
    xf = _art(ctx, "EXPLORATORY_UNRESOLVED_FEATURES.csv")
    rr = mc.set_index("config")
    rec = sel["recommended"]
    caps = [float(c) for c in ctx.cfg["metrics"]["capacities"]]
    principal = float(ctx.cfg["selection"]["principal_capacity"])
    p = facts["partitions"]
    L = ["# Phase 2 scientific summary", "", f"**{WATERMARK}**", ""]
    if facts.get("synthetic"):
        L += ["> SYNTHETIC DATA – SOFTWARE TEST ONLY, NOT SCIENTIFIC RESULTS", ""]
    if not ctx.run.plan.get("frozen_production_config"):
        L += ["> NOT THE FROZEN PRODUCTION CONFIGURATION (test / development run)", ""]
    L += [f"*{SCOPE}*", "", f"Frozen configuration: FINAL_EXPERIMENT_CONFIG.json sha256 `{str(ctx.run.plan.get('final_experiment_config_sha256'))[:16]}…`; "
          f"TRAIN {p['train']['n_rows']:,} rows / {p['train']['n_events']:,} events; VALIDATION {p['validation']['n_rows']:,} / {p['validation']['n_events']:,}; "
          f"TEST ({facts['test_rows_dropped']:,} rows) dropped before any Phase 2 computation and never read.", "",
          "## 1. Three categories (never mixed)", "",
          "- **HISTORICAL_BASELINE_15** – the previous benchmark's 15 EXTENDED predictors (provenance as found), REFITTED on TRAIN in the Phase 2 "
          "protocol (`LASSO:BASELINE_15`). It is not the Phase 1 model, which was recalibrated on these VALIDATION outcomes; its Phase 1 "
          "metrics are in the reference run's reports and are not re-scored here.",
          "- **SAFE_DISCOVERY** (primary) – only features, new and baseline, whose prediction-time provenance is proven SAFE (D-00 rules unchanged).",
          "- **EXPLORATORY_UNRESOLVED_SENSITIVITY** – a separate LASSO sensitivity that also offers UNRESOLVED sources (never UNSAFE ones).", "",
          "## 2. Column and feature accounting (exact counts, this extract)", "", *_md_table(["Step", "n", "Category"],
                                                                                          [[r.step, str(r.n), r.category or ""] for r in fun.itertuples()]), "",
          "## 3. Benchmark: BASELINE_15 refit (Phase 2 protocol, TRAIN only)", ""]
    b = rr.loc[BENCHMARK]
    L += _md_table(["Metric", "TRAIN out-of-fold", "VALIDATION"], [["AP (average precision = step-wise PR-AUC)", _fmt(b.get("oof_ap")), _fmt(b.get("val_ap"))],
                                                                  ["AUROC", _fmt(b.get("oof_auroc")), _fmt(b.get("val_auroc"))],
                                                                  ["Brier", _fmt(b.get("oof_brier"), 4), _fmt(b.get("val_brier"), 4)],
                                                                  [f"capture at top {principal:.0%}", _pct(b.get(f"oof_capture@{principal:g}")), _pct(b.get(f"val_capture@{principal:g}"))]])
    L += ["", "## 4. Where the improvement comes from (management waterfall; every step on the same VALIDATION rows)", ""]
    wf = dom[dom["analysis"] == "management_waterfall"] if len(dom) else dom
    hdr = ["Step", "Category", "AP (PR-AUC)", "AUROC", "Brier", "slope", "CITL", *[f"capture {c:.0%}" for c in caps if c >= 0.03],
           f"PPV {principal:.0%}", f"false alerts {principal:.0%}", "Δ AP vs BASELINE_15 refit [95% CI]"]
    tag = cap_tag(principal)
    mc_ = int(ctx.cfg["screening"]["min_cell"])
    L += _md_table(hdr, [[str(r["label"]), str(r["category"]), _fmt(r.get("val_ap")), _fmt(r.get("val_auroc")), _fmt(r.get("val_brier"), 4),
                          _fmt(r.get("val_cal_slope"), 2), _fmt(r.get("val_citl"), 2),
                          *[("supp." if _cap_small(r, cap_tag(c), mc_) else _pct(r.get(f"val_capture_{cap_tag(c)}"))) for c in caps if c >= 0.03],
                          "supp." if _cap_small(r, tag, mc_) else _pct(r.get(f"val_ppv_{tag}")),
                          "supp." if _cap_small(r, tag, mc_) else _fmt(r.get(f"val_false_alerts_{tag}"), 0),
                          f"{_fmt(r.get('val_delta_ap_vs_benchmark'), 4)} [{_fmt(r.get('val_delta_ap_ci_low_vs_benchmark'), 4)}, {_fmt(r.get('val_delta_ap_ci_high_vs_benchmark'), 4)}]"]
                         for _, r in wf.iterrows()])
    L += ["", "The waterfall order was declared before any result; a domain step is order-dependent (see the one-domain additions and "
          "leave-one-domain-out rows of DOMAIN_INCREMENTAL_GAIN.csv). A step marked `no_safe_feature_in_domain` added nothing because that "
          "domain has no SAFE feature in this extract.", "", "## 5. The management question", ""]
    if head:
        L += [f"If Meuhedet can intervene on the same X% of members, how many additional recorded future falls/fractures (180 days) does the "
              f"recommended SAFE model ({rec}) identify compared with the BASELINE_15 refit, on the same {int(head[0]['n']):,} VALIDATION members "
              f"({int(head[0]['events'])} recorded events)?", ""]
        L += _md_table(["Capacity", "Flagged", "Recorded events identified: recommended", "... : BASELINE_15 refit", "Additional [95% CI]",
                        "per 10,000 comparable D00_CLEAN patients [95% CI]", "False alerts: recommended / BASELINE_15 refit"],
                       [[f"top {h['capacity']:.0%}", str(int(h["flagged"])), _cnt(h["falls_new"]), _cnt(h["falls_benchmark"]),
                         (_signed(h["additional"]) if not h.get("suppressed") else "suppressed (<10)")
                         + (f" [{_signed(h['ci_low'])} to {_signed(h['ci_high'])}]" if h.get("ci_low") is not None and pd.notna(h["ci_low"]) else ""),
                         _signed(h["per_10000"], 1) + (f" [{_signed(h['ci_low'] / h['n'] * 10000, 1)} to {_signed(h['ci_high'] / h['n'] * 10000, 1)}]"
                                                       if h.get("ci_low") is not None and pd.notna(h["ci_low"]) else ""),
                         f"{_cnt(h['fa_new'])} / {_cnt(h['fa_benchmark'])}"] for h in head])
        L += ["", "\"Identified\" = recorded events among the flagged members, not falls prevented, and not an estimate for all members. Intervals: "
              "paired patient bootstrap of the frozen predictions (same resampled patients for both models, top-N recomputed in every replicate; "
              "they do not account for the model search).", ""]
    L += ["## 6. Frozen selection (TRAIN only) and the one-shot VALIDATION", "",
          f"- Recommended SAFE model (frozen before VALIDATION was opened): **{rec}** – {sel['reason']}",
          f"- Shortlist: {', '.join(sel['shortlist'])}",
          f"- VALIDATION was opened once, after the selection was frozen ({opened['n_configs_to_score']} frozen configurations scored; the time and "
          "the frozen-selection sha256 are in VALIDATION_OPENED.json).",
          f"- Selection outcome: **{sel.get('selection_outcome')}**.",
          f"- Confirmation vs SAFE_BASE (pre-declared exploratory screen on reused VALIDATION): **{conf['confirmation']}** – {conf['detail']}. "
          "Wording: promising, not established.",
          f"- Point-estimate class vs the BASELINE_15 refit: TRAIN OOF **{conf.get('benchmark_class_oof')}**, VALIDATION "
          f"**{conf.get('benchmark_class_validation')}** (MEETS_SUPERIORITY_RULE / WITHIN_NONINFERIORITY_MARGINS / OUTSIDE_MARGINS; not formal tests).",
          "- Nesting: every supervised step (fractional polynomials, medians, standardisation, lambda / l1_ratio, XGBoost trials and tree counts) is "
          "fitted inside the outer training folds; the inner lambda search is conditional on the outer-training-fold FP choice for age / "
          "polypharmacy, and the label-blind decisions (redundancy clusters, thermometer levels, NULL-pattern groups) were made once on all TRAIN "
          "rows; OOF results are conditional on them. Inner folds are common to every candidate of a tuning scope.", ""]
    contr = contr if contr is not None else pd.DataFrame()
    if len(contr):
        L += ["## 6b. Attribution contrasts (frozen before VALIDATION)", ""]
        L += _md_table(["Contrast", "New vs reference", "OOF ΔAP [95% CI]", "VALIDATION ΔAP [95% CI]", f"VALIDATION Δ events identified at top {principal:.0%} [95% CI]"],
                       [[r.contrast, f"{r.new} vs {r.reference}", f"{_fmt(r.oof_delta_ap, 4)} [{_fmt(r.oof_delta_ap_ci_low, 4)}, {_fmt(r.oof_delta_ap_ci_high, 4)}]",
                         f"{_fmt(getattr(r, 'val_delta_ap', None), 4)} [{_fmt(getattr(r, 'val_delta_ap_ci_low', None), 4)}, {_fmt(getattr(r, 'val_delta_ap_ci_high', None), 4)}]",
                         f"{_signed(getattr(r, 'val_delta_falls_' + cap_tag(principal), None))} [{_signed(getattr(r, 'val_delta_falls_' + cap_tag(principal) + '_ci_low', None))}, "
                         f"{_signed(getattr(r, 'val_delta_falls_' + cap_tag(principal) + '_ci_high', None))}]"] for r in contr.itertuples()])
        L += ["", "FEATURE_EXPANSION isolates the SAFE new features (same model family); MODEL_FAMILY_PIPELINE isolates gradient boosting on the same "
              "SAFE feature set (a pipeline comparison: encodings and missing-value handling differ by design).", ""]

    L += ["## 7. Model families on the same SAFE feature set", "",
          "XGBoost: stage 0 (fixed defaults) and stage 1 (15 Optuna trials per tuning scope); the refinement stage 2 is disabled in the frozen "
          "configuration (Astra F-06) - deeper tuning is a future experiment only if stage 1 shows value.", ""]
    fam = mc[mc["feature_set"] == ALL_SAFE]
    L += _md_table(["Configuration", "OOF AP", "OOF AUROC", "VAL AP", "VAL AUROC", "VAL slope", "VAL CITL"],
                   [[r.config, _fmt(r.oof_ap), _fmt(r.oof_auroc), _fmt(getattr(r, "val_ap", None)), _fmt(getattr(r, "val_auroc", None)),
                     _fmt(getattr(r, "val_cal_slope", None), 2), _fmt(getattr(r, "val_citl", None), 2)] for r in fam.itertuples()])
    L += ["", "## 8. SAFE feature shortlist (STRONG / MODERATE)", ""]
    if len(cons):
        top = cons[cons["recommendation"].isin(["STRONG", "MODERATE"])].head(20)
        L += _md_table(["Feature", "Domain", "eFalls / native", "Rating", "LASSO / EN / XGB", "stability", "ablation ΔAP [95% CI]", "Caution"],
                       [[r.feature, r.domain, r.efalls_vs_native, r.recommendation,
                         f"{'Y' if r.lasso_selected else '-'} / {'Y' if r.elasticnet_selected else '-'} / {'Y' if r.xgb_used else '-'}", _fmt(r.lasso_stability, 2),
                         f"{_fmt(r.ablation_delta_ap_full_minus_ablated, 4)} [{_fmt(r.ablation_delta_ap_ci_low, 4)}, {_fmt(r.ablation_delta_ap_ci_high, 4)}]",
                         r.interpretation_caution] for r in top.itertuples()] or [["(none)", "", "", "", "", "", "", ""]])
    L += ["", "## 9. EXPLORATORY_UNRESOLVED_SENSITIVITY (separate; not a SAFE result)", ""]
    ex = [c for c in (lasso_config(ctx, EXPL_B15_SAFE), lasso_config(ctx, EXPL_ALL)) if c in rr.index]
    L += _md_table(["Configuration", "OOF AP", "OOF Δ vs BASELINE_15 refit", "VAL AP (descriptive)", "VAL Δ vs BASELINE_15 refit"],
                   [[c, _fmt(rr.at[c, "oof_ap"]), _fmt(rr.at[c, "oof_delta_ap"], 4), _fmt(rr.at[c, "val_ap"] if "val_ap" in rr else None),
                     _fmt(rr.at[c, "val_delta_ap"] if "val_delta_ap" in rr else None, 4)] for c in ex])
    if len(xf):
        sig = xf[xf["exploratory_status"].str.startswith("EXPLORATORY SIGNAL")]
        L += ["", f"UNRESOLVED features with an exploratory signal ({len(sig)} of {len(xf)}): " + (", ".join(sig["feature"].head(20)) or "none") +
              ". They need proven record dates in the next extract before any use."]
    L += ["", "## 10. Sensitivity and robustness", ""]
    na = lasso_config(ctx, f"{ALL_SAFE}_WITHOUT_EXPLICIT_ASSESSMENT_FLAGS")
    full = lasso_config(ctx, ALL_SAFE)
    if na in rr.index and full in rr.index:
        L.append(f"- Without the explicit form 'assessed' indicators (other NULL indicators stay): OOF AP {_fmt(rr.at[na, 'oof_ap'])} vs {_fmt(rr.at[full, 'oof_ap'])} with them "
                 f"(VALIDATION, descriptive: {_fmt(rr.at[na, 'val_ap'] if 'val_ap' in rr else None)} vs {_fmt(rr.at[full, 'val_ap'] if 'val_ap' in rr else None)}).")
    abl = _art(ctx, "ABLATION_RESULTS.csv")
    if len(abl) and "delta_ap_full_minus_ablated" in abl:
        ind = abl[abl["analysis"] == "individual"].sort_values("delta_ap_full_minus_ablated", ascending=False)
        loo = abl[abl["analysis"] == "leave_one_domain_out"].sort_values("delta_ap_full_minus_ablated", ascending=False)
        if len(ind):
            tot = ind["delta_ap_full_minus_ablated"].clip(lower=0).sum()
            share = float(ind["delta_ap_full_minus_ablated"].clip(lower=0).iloc[0] / tot) if tot > 0 else float("nan")
            L.append(f"- Largest single-feature ablation: {ind.iloc[0]['removed']} (ΔAP {_fmt(ind.iloc[0]['delta_ap_full_minus_ablated'], 4)}; "
                     f"{_pct(share)} of the summed positive individual effects).")
        if len(loo):
            L.append("- Leave-one-domain-out (unique contribution after the other SAFE domains): " + "; ".join(
                f"{r.removed} {_fmt(r.delta_ap_full_minus_ablated, 4)}" for r in loo.itertuples()))
    L += ["", "## 11. Recommendation", "",
          f"Carry **{rec}** (SAFE_DISCOVERY) to a new, preferably temporal, holdout" + (" – no SAFE configuration met the advancement rule over "
                                                                                        "SAFE_BASE." if rec == SAFE_REF else "."),
          "Every feature of it has proven prediction-time provenance in this extract. The historical BASELINE_15 contains predictors whose timing is "
          "UNRESOLVED; its comparison is a benchmark, not a deployable alternative, until the extract proves their record dates.", "",
          "## 12. Next scientific step (freeze before the holdout outcomes are read; Astra A-35)", "",
          "- cohort, index date(s), horizon, endpoint, death and incomplete-follow-up handling; source extraction with Event_Date < Index_Date (no row removal)",
          "- sources, timing rules, feature definitions, missingness processing and coding (this catalogue + FEATURE_SETS.json)",
          "- model family, hyper-parameters, fitting / refitting procedure and training-data cutoff (SELECTION_FROZEN.json)",
          "- any recalibration procedure and the independent data used to fit it (never fitted and assessed on the same holdout)",
          f"- primary metric (AP = step-wise PR-AUC), principal capacity (top {principal:.0%}), the advancement rule and the code version",
          "- report holdout prevalence and case mix; show the original model's performance before any updating", "",
          "## 13. Limitations", "",
          "- Exploratory 180-day fall/fracture label (not the eFalls outcome); deaths are non-events; censored rows excluded.",
          "- The reference cohort removed members with index-day records (D-00): inclusion depends on information unavailable at prediction time.",
          "- VALIDATION was reused in earlier work (Phase 1 recalibration); here it is opened once, confirmatory for the pre-declared comparisons only.",
          "- Nurse and MEFI assessments also reflect who was assessed (care process); transport depends on assessment practice.",
          "- Selected features are statistical associations with the recorded outcome; no causal interpretation is made.",
          "- A record date bounds the record, not later updates of a status / latest-value field (warehouse to confirm).",
          _death_sentence(facts)]
    return "\n".join(L) + "\n"


def _management_he(ctx: Ctx, sel: dict[str, Any], mc: pd.DataFrame, dom: pd.DataFrame, head: list[dict[str, Any]]) -> str:
    conf = json.loads((ctx.run.artifacts / "VALIDATION_CONFIRMATION.json").read_text(encoding="utf-8"))
    facts = ctx.facts()
    rec = sel["recommended"]
    principal = float(ctx.cfg["selection"]["principal_capacity"])
    caps = [float(c) for c in ctx.cfg["metrics"]["capacities"] if float(c) >= 0.03]
    L = ["# סיכום להנהלה – שלב 2: משתנים חדשים וסוגי מודלים", "", f"**{WATERMARK_HE}**", ""]
    if facts.get("synthetic"):
        L += ["> נתונים סינתטיים – בדיקת תוכנה בלבד, לא תוצאות מדעיות", ""]
    L += ["*היקף: חיזוי חקרני של נפילות/שברים רשומים תוך 180 יום בקוהורטה D00_CLEAN (שנבחרה גם לפי מידע מאוחר לזמן החיזוי); סט האימות שימש כבר "
          "בשלב 1. הסיווגים הם כללי סף על אומדנים נקודתיים שנקבעו מראש – לא מבחני עליונות או אי-נחיתות פורמליים.*", "",
          "## השאלה", "", "אם מאוחדת יכולה להתערב אצל אותו אחוז מהמבוטחים – כמה נפילות עתידיות רשומות (בחצי השנה הבאה) נוספות מזהה הגישה החדשה?", ""]
    if head:
        hp = next((h for h in head if abs(h["capacity"] - principal) < 1e-9), head[0])
        if hp.get("suppressed"):
            L += [f"**בקיבולת של {hp['capacity']:.0%}:** המספרים מוסתרים (תא קטן מ-10).", ""]
        else:
            L += [f"**בקיבולת של {hp['capacity']:.0%} ({int(hp['flagged'])} מבוטחים מסומנים מתוך {int(hp['n']):,} בסט האימות):** המודל הבטוח המומלץ מזהה "
                  f"{int(hp['falls_new'])} נפילות רשומות לעומת {int(hp['falls_benchmark'])} ב-15 המשתנים ההיסטוריים (שהותאמו מחדש על סט האימון) – "
                  f"הפרש של {_signed(hp['additional'])}"
                  + (f" (טווח 95%: {_signed(hp['ci_low'])} עד {_signed(hp['ci_high'])})" if hp.get("ci_low") is not None and pd.notna(hp["ci_low"]) else "")
                  + f", כלומר כ-{_signed(hp['per_10000'], 1)} לכל 10,000 מטופלים דומים בקוהורטה.", ""]
        L += ["| קיבולת (אחוז מסומנים) | מסומנים | זוהו – מודל חדש | זוהו – 15 המשתנים | נוספות (טווח 95%) | לכל 10,000 מטופלים דומים | התראות שווא (חדש / היסטורי) |",
              "|---|---|---|---|---|---|---|"]
        L += [f"| {h['capacity']:.0%} | {int(h['flagged'])} | {_cnt(h['falls_new'])} | {_cnt(h['falls_benchmark'])} | "
              + ("מוסתר" if h.get("suppressed") else _signed(h["additional"]) + (f" ({_signed(h['ci_low'])} עד {_signed(h['ci_high'])})"
                                                                                   if h.get("ci_low") is not None and pd.notna(h["ci_low"]) else ""))
              + f" | {_signed(h['per_10000'], 1)} | {_cnt(h['fa_new'])} / {_cnt(h['fa_benchmark'])} |" for h in head]
        L += ["", "\"זוהו\" = נפילות רשומות בקרב המסומנים, לא נפילות שנמנעו.", ""]
    L += ["## מאיפה הגיע השיפור (כל שלב על אותם מבוטחים בסט האימות)", "",
          "| שלב | קטגוריה | AP (PR-AUC) | AUROC | Brier | שיפוע כיול | " + " | ".join(f"זיהוי ב-{c:.0%}" for c in caps)
          + f" | PPV ב-{principal:.0%} | התראות שווא ב-{principal:.0%} |", "|" + "---|" * (7 + len(caps))]
    labels = {d: ctx.cat.domains[d]["label_he"] for d in ctx.cat.domains}
    tag = cap_tag(principal)
    wf = dom[dom["analysis"] == "management_waterfall"] if len(dom) else dom
    for _, r in wf.iterrows():
        lab = str(r["label"])
        if r["model"] == BENCHMARK:
            lab = "15 המשתנים ההיסטוריים – הותאמו מחדש על סט האימון (נקודת ההשוואה)"
        elif r["model"] == SAFE_REF:
            lab = "הבסיס הבטוח (משתני הבסיס שתזמונם מאומת)"
        elif str(r.get("domain_added") or "") in labels:
            lab = f"+ {labels[str(r['domain_added'])]}" + (" (אין משתנה בטוח בתחום)" if bool(r.get("no_safe_feature_in_domain")) else "")
        elif str(r.get("domain_added") or "") == "OTHER_NATIVE":
            lab = "+ שאר התחומים = כל המשתנים הבטוחים (מודל ליניארי)"
        elif lab.startswith("ALL_SAFE_ELASTIC_NET"):
            lab = "כל המשתנים הבטוחים – Elastic Net"
        elif lab.startswith("ALL_SAFE_XGBOOST"):
            lab = "כל המשתנים הבטוחים – XGBoost"
        elif lab.startswith("RECOMMENDED"):
            lab = f"המודל הבטוח המומלץ ({r['model']})"
        mc_ = int(ctx.cfg["screening"]["min_cell"])
        cells = ["מוסתר" if _cap_small(r, cap_tag(c), mc_) else _pct(r.get(f"val_capture_{cap_tag(c)}")) for c in caps]
        small_p = _cap_small(r, tag, mc_)
        L.append(f"| {lab} | {CATEGORY_HE.get(str(r['category']), r['category'])} | {_fmt(r.get('val_ap'))} | {_fmt(r.get('val_auroc'))} | "
                 f"{_fmt(r.get('val_brier'), 4)} | {_fmt(r.get('val_cal_slope'), 2)} | " + " | ".join(cells)
                 + f" | {'מוסתר' if small_p else _pct(r.get(f'val_ppv_{tag}'))} | {'מוסתר' if small_p else _fmt(r.get(f'val_false_alerts_{tag}'), 0)} |")
    L += ["", "הבחירה נעשתה מראש בתוך סט האימון בלבד והוקפאה; סט האימות נפתח פעם אחת בלבד לאחר ההקפאה, ולא שימש לכוונון.", "",
          "## ההמלצה", "",
          (f"להמשיך לבדיקה על תקופת זמן חדשה עם המודל הבטוח: **{rec}**." if rec != SAFE_REF else
           "אף צירוף של משתנים בטוחים לא עמד בכלל ההתקדמות שנקבע מראש מעל הבסיס הבטוח."),
          f"סטטוס הבדיקה בסט האימות (מול הבסיס הבטוח; סינון חקרני): **{conf['confirmation']}** (מבטיח – לא מוכח).",
          f"מול 15 המשתנים ההיסטוריים: **{conf.get('benchmark_class_validation')}** בסט האימות (MEETS_SUPERIORITY_RULE = עומד בכלל העדיפות, "
          "WITHIN_NONINFERIORITY_MARGINS = בתוך שולי אי-הנחיתות, OUTSIDE_MARGINS = מחוץ לשוליים; אומדן נקודתי בלבד).", "",
          "## ניתוח חקרני נפרד (לא חלק מההמלצה)", "",
          "משתנים שתזמונם לא אומת (ללא תאריך רשומה בחילוץ) נבדקו רק בניתוח רגישות נפרד, ומופיעים בקובץ EXPLORATORY_UNRESOLVED_FEATURES.csv. "
          "הם לא ייכנסו למודל לפני שהחילוץ הבא יוכיח את תאריכי הרשומה שלהם.", "",
          "## הסתייגויות", "",
          "- תוצא חקרני (נפילה או שבר רשומים תוך 180 יום), לא התוצא של eFalls.",
          "- הערכות אחות ו-MEFI משקפות גם את מי שנבדק (תהליך טיפול) – תלוי בנוהלי ההערכה.",
          "- משתנים נבחרים הם קשר סטטיסטי עם התוצא, לא סיבה.",
          "- נדרש תיקוף על תקופת זמן מאוחרת יותר לפני כל שימוש קליני."]
    return "\n".join(L) + "\n"


def _check_wording(text: str, where: str) -> None:
    low = text.lower()
    hits = [w for w in BANNED if re.search(rf"(?<![\w]){re.escape(w)}(?![\w])", low)]
    if hits:
        raise Phase2Stop("WORDING", f"{where} contains banned wording", hits)


def s15_report(ctx: Ctx) -> None:
    st = ctx.run.stage("15", "report", ("14",))
    if st.complete_record():
        return
    sel = selection(ctx)
    figs = ctx.run.out / "figures"

    def fn(tmp: Path, seed: int) -> dict[str, Any]:
        mc = _model_comparison(ctx)
        dom = _domain_table(ctx, sel)
        fun = _funnel(ctx)
        head = _headline(ctx, sel, dom)
        cons = _art(ctx, "FEATURE_CONSENSUS.csv")
        short = cons[cons["recommendation"].isin(["STRONG", "MODERATE"])][[c for c in SHORTLIST_COLUMNS if c in cons.columns]] if len(cons) else cons
        show = list(dict.fromkeys([*sel["shortlist"], *dom.loc[dom["analysis"] == "management_waterfall", "model"]])) if len(dom) else list(sel["shortlist"])
        cal_rows = []
        cb = _art(ctx, "CALIBRATION_BOOTSTRAP.csv")
        cb = cb.set_index("config") if len(cb) else cb
        for r in mc.itertuples():
            if r.config in show:
                row = {"config": r.config, "category": r.category, **{k: getattr(r, k, None) for k in (
                    "oof_citl", "oof_cal_slope", "oof_cal_intercept", "oof_oe", "oof_brier", "oof_scaled_brier", "oof_logloss", "val_citl", "val_cal_slope",
                    "val_cal_intercept", "val_oe", "val_brier", "val_scaled_brier", "val_logloss")}}
                if len(cb) and r.config in cb.index:
                    for k in ("cal_slope", "cal_intercept", "citl", "oe", "brier", "scaled_brier"):
                        row[f"val_{k}_ci_low"], row[f"val_{k}_ci_high"] = cb.at[r.config, f"{k}_ci_low"], cb.at[r.config, f"{k}_ci_high"]
                cal_rows.append(row)
        definitions = {"scaled_brier": "1 - Brier / [p_eval (1 - p_eval)], p_eval = prevalence of the evaluated rows", "oe": "sum(y) / sum(p)",
                       "citl": "a in logit P(y=1) = a + 1 x logit(p) (fixed-slope offset model)", "cal_slope/intercept": "logit P(y=1) = a + b logit(p)",
                       "intervals": "percentile 95%, patient bootstrap of the frozen predictions"}
        co, cv = _art(ctx, "ATTRIBUTION_CONTRASTS_OOF.csv"), _art(ctx, "ATTRIBUTION_CONTRASTS_VALIDATION.csv")
        contr = co.merge(cv[[c for c in cv.columns if c.startswith("val_") or c == "contrast"]], on="contrast", how="left") if len(co) and len(cv) else co
        D.write_csv(tmp / "ATTRIBUTION_CONTRASTS.csv", contr)
        D.write_csv(tmp / "MODEL_COMPARISON.csv", mc)
        D.write_csv(tmp / "DOMAIN_INCREMENTAL_GAIN.csv", dom)
        D.write_csv(tmp / "COLUMN_FUNNEL.csv", fun)
        D.write_csv(tmp / "FEATURE_SHORTLIST.csv", short)
        D.write_csv(tmp / "CALIBRATION_SUMMARY.csv", pd.DataFrame(cal_rows).assign(definitions=json.dumps(definitions)))
        D.write_json(tmp / "MANAGEMENT_HEADLINE.json", {"recommended": sel["recommended"], "benchmark": BENCHMARK, "rows": head})
        sci = _scientific_summary(ctx, sel, mc, dom, fun, head, contr)
        he = _management_he(ctx, sel, mc, dom, head)
        _check_wording(sci, "SCIENTIFIC_SUMMARY.md")
        _check_wording(he, "MANAGEMENT_SUMMARY_HE.md")
        D.write_str(tmp / "SCIENTIFIC_SUMMARY.md", sci)
        D.write_str(tmp / "MANAGEMENT_SUMMARY_HE.md", he)
        return {"n_configs": int(len(mc))}

    st.item("report", fn)
    outs = [D.write_bytes(ctx.run.artifacts / n, (st.path("report") / n).read_bytes())
            for n in ("MODEL_COMPARISON.csv", "DOMAIN_INCREMENTAL_GAIN.csv", "COLUMN_FUNNEL.csv", "FEATURE_SHORTLIST.csv", "CALIBRATION_SUMMARY.csv",
                      "ATTRIBUTION_CONTRASTS.csv", "MANAGEMENT_HEADLINE.json", "SCIENTIFIC_SUMMARY.md", "MANAGEMENT_SUMMARY_HE.md")]
    if not (figs / ".done").exists():
        tmpf = ctx.run.out / f".tmp-figures-a{ctx.run.attempt}"
        if tmpf.exists():
            shutil.rmtree(tmpf)
        dom = _art(ctx, "DOMAIN_INCREMENTAL_GAIN.csv")
        head = json.loads((ctx.run.artifacts / "MANAGEMENT_HEADLINE.json").read_text(encoding="utf-8"))["rows"]
        made = _figures(ctx, tmpf, dom, _art(ctx, "MODEL_COMPARISON.csv"), sel, head)
        D.write_json(tmpf / ".done", {"files": made})
        if figs.exists():
            D.rename_dir(figs, ctx.run.out / "stages" / f"_old_figures_a{ctx.run.attempt}")
        D.rename_dir(tmpf, figs)
    outs += [p for p in sorted(figs.iterdir()) if p.is_file() and p.name != ".done"]
    st.finalize(outs, {"n_figures": len(outs)})


# ============================================================================ S16 share
def _small(v: Any, min_cell: int) -> bool:
    x = pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]
    return bool(pd.notna(x) and 0 < x < min_cell)


def _suppress(df: pd.DataFrame, rule: str, min_cell: int) -> pd.DataFrame:
    """Small-cell suppression for shared tables. A confusion-matrix row with any cell in 1..min_cell-1 loses all four cells AND every rate
    derived from them (otherwise n_flagged - fp or ppv x n_flagged gives the hidden count back); calibration groups lose their observed rate
    and interval when events or non-events are small; subgroup rows lose their counts when small (their metrics are reported only >= 10 events)."""
    out = df.copy().astype(object)
    tag = f"<{min_cell}"
    bench_small: set[str] = set()
    if rule == "capacity_counts" and "model" in out.columns:
        for _, row in out[out["model"] == BENCHMARK].iterrows():
            bench_small |= {t for t in _cap_tags(out.columns) if _cap_small(row, t, min_cell)}
    for i, row in out.iterrows():
        if rule == "confusion":
            if any(_small(row.get(c), min_cell) for c in CONFUSION):
                for c in (*CONFUSION, *CONFUSION_DERIVED):
                    if c in out.columns:
                        out.at[i, c] = tag if c in CONFUSION else "suppressed"
        elif rule == "calibration":
            n, ev = pd.to_numeric(row.get("n"), errors="coerce"), pd.to_numeric(row.get("events"), errors="coerce")
            if _small(ev, min_cell) or _small(n - ev, min_cell) or _small(n, min_cell):
                for c in ("events", "observed_rate", "ci_low", "ci_high"):
                    if c in out.columns:
                        out.at[i, c] = tag if c == "events" else "suppressed"
        elif rule == "capacity_counts":
            for tag in _cap_tags(out.columns):
                if _cap_small(row, tag, min_cell) or tag in bench_small:
                    for c in _cap_cols(tag, own=_cap_small(row, tag, min_cell)):
                        if c in out.columns:
                            out.at[i, c] = "suppressed"
        elif rule == "subgroup":
            for c in ("n", "events"):
                if c in out.columns and _small(row.get(c), min_cell):
                    out.at[i, c] = tag
            n, ev = pd.to_numeric(row.get("n"), errors="coerce"), pd.to_numeric(row.get("events"), errors="coerce")
            if pd.notna(n) and pd.notna(ev) and _small(n - ev, min_cell):
                out.at[i, "events"] = tag
    return out


def _cap_tags(columns: Any) -> list[str]:
    return [c[len("val_falls_identified_"):] for c in columns if str(c).startswith("val_falls_identified_")]


def _cap_small(row: Any, tag: str, min_cell: int) -> bool:
    """Any cell of the capacity confusion matrix of this row (tp, fp, fn, tn at capacity ``tag``) in 1..min_cell-1."""
    tp, k = pd.to_numeric(row.get(f"val_falls_identified_{tag}"), errors="coerce"), pd.to_numeric(row.get(f"val_flagged_{tag}"), errors="coerce")
    n, ev = pd.to_numeric(row.get("val_n"), errors="coerce"), pd.to_numeric(row.get("val_events"), errors="coerce")
    if not (pd.notna(tp) and pd.notna(k)):
        return False
    cells = [tp, k - tp] + ([ev - tp, n - ev - (k - tp)] if pd.notna(n) and pd.notna(ev) else [])
    return any(_small(c, min_cell) for c in cells)


def _cap_cols(tag: str, *, own: bool) -> list[str]:
    diff = [f"val_additional_falls_vs_benchmark_{tag}", f"val_additional_falls_vs_benchmark_{tag}_ci_low", f"val_additional_falls_vs_benchmark_{tag}_ci_high",
            f"val_fewer_false_alerts_vs_benchmark_{tag}"]
    return [*diff, f"val_falls_identified_{tag}", f"val_capture_{tag}", f"val_ppv_{tag}", f"val_false_alerts_{tag}"] if own else diff


def _identifiers(ctx: Ctx) -> set[str]:
    ids: set[str] = set()
    raw = pd.read_csv(ctx.src, usecols=["Customer_Full_ID", "Snapshot_Key"], dtype="string", keep_default_na=False) \
        if ctx.src.suffix.lower() == ".csv" else pd.read_parquet(ctx.src, columns=["Customer_Full_ID", "Snapshot_Key"])
    for col in raw.columns:
        ids |= {str(v) for v in raw[col].dropna().astype(str) if len(str(v)) >= 6 and str(v).upper() != "NULL"}
    ids |= set(ctx.work()["research_id"].astype(str))
    return ids


def _pepper(ctx: Ctx) -> str:
    from falls_ml.phase2.cohort import read_reference_pepper

    pepper, _ = read_reference_pepper(ctx.src, ctx.ref_dir, ctx.pepper_file, None)
    return pepper


def _path_hits(text: str, needles: list[str]) -> list[str]:
    low = text.lower()
    hits = [n for n in needles if n and n.lower() in low]
    if re.search(r"[a-z]:\\users\\", low) or re.search(r"/users/[^/\s]+/", low) or re.search(r"/home/[^/\s]+/", low):
        hits.append("absolute user path")
    return hits


def s16_share(ctx: Ctx) -> None:
    from falls_ml.eda.runner import privacy_scan

    st = ctx.run.stage("16", "share", ("15",))
    if st.complete_record():
        return
    out = ctx.run.out
    share = out / "share"
    if share.exists():
        rec = st.complete_record()
        if rec is None:   # an interrupted earlier build: never reuse it
            D.rename_dir(share, out / "stages" / f"_incomplete_share_a{ctx.run.attempt}")
    tmp = out / f".tmp-share-a{ctx.run.attempt}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    min_cell = int(ctx.cfg["screening"]["min_cell"])
    A = ctx.run.artifacts
    top = {"COLUMN_FUNNEL.csv": "COLUMN_FUNNEL.csv", "FEATURE_REGISTRY.csv": "02_ENGINEERED_FEATURE_REGISTRY.csv", "FEATURE_QUALITY.csv": "05_FEATURE_QUALITY.csv",
           "FEATURE_SHORTLIST.csv": "FEATURE_SHORTLIST.csv", "FEATURE_CONSENSUS.csv": "FEATURE_CONSENSUS.csv",
           "EXPLORATORY_UNRESOLVED_FEATURES.csv": "EXPLORATORY_UNRESOLVED_FEATURES.csv", "MODEL_COMPARISON.csv": "MODEL_COMPARISON.csv",
           "ABLATION_RESULTS.csv": "ABLATION_RESULTS.csv", "CALIBRATION_SUMMARY.csv": "CALIBRATION_SUMMARY.csv",
           "MANAGEMENT_SUMMARY_HE.md": "MANAGEMENT_SUMMARY_HE.md", "SCIENTIFIC_SUMMARY.md": "SCIENTIFIC_SUMMARY.md",
           "MANAGEMENT_HEADLINE.json": "MANAGEMENT_HEADLINE.json", "VALIDATION_CONFIRMATION.json": "VALIDATION_CONFIRMATION.json",
           "ATTRIBUTION_CONTRASTS.csv": "ATTRIBUTION_CONTRASTS.csv"}
    for dst, src in top.items():
        if (A / src).exists():
            shutil.copyfile(A / src, tmp / dst)
    for name in ("FINAL_EXPERIMENT_CONFIG.json", "FINAL_EXPERIMENT_CONFIG.sha256", "SELECTION_FROZEN.json", "VALIDATION_OPENED.json"):
        if (out / name).exists():
            shutil.copyfile(out / name, tmp / name)
    for dst, (src, rule) in SHARE_TABLES.items():
        if (A / src).exists():
            D.write_csv(tmp / dst, _suppress(pd.read_csv(A / src), rule, min_cell))
    (tmp / "tables").mkdir()
    for name in ("01_COLUMN_REGISTRY.csv", "03_UNIVARIATE_SCREEN.csv", "04_REDUNDANCY_CLUSTERS.csv", "OOF_MODEL_COMPARISON.csv",
                 "VALIDATION_MODEL_COMPARISON.csv", "VALIDATION_CONFIRMATION.json", "STABILITY.csv", "PERMUTATION_IMPORTANCE.csv", "SHAP_SUMMARY.csv",
                 "SHAP_INTERACTIONS.csv", "XGB_TRIALS.csv", "FITS_LASSO.csv", "FITS_ENET.csv", "FITS_XGB.csv", "FEATURE_SETS.json", "COHORT_FACTS.json",
                 "ENVIRONMENT.json", "CALIBRATION_SMOOTH.csv", "CALIBRATION_BOOTSTRAP.csv"):
        if (A / name).exists():
            shutil.copyfile(A / name, tmp / "tables" / name)
    shutil.copytree(out / "figures", tmp / "figures", ignore=shutil.ignore_patterns(".done"))
    for name in ("validation_evaluation_registry.jsonl",):
        if (out / name).exists():
            shutil.copyfile(out / name, tmp / "tables" / name.replace(".jsonl", ".json.txt"))
    from falls_ml.paths import resolve_path

    rev = resolve_path("reviews")
    (tmp / "reviews").mkdir()
    for n in ("ASTRA_METHOD_REVIEW.md", "AGY_ENGINEERING_REVIEW.md", "REVIEW_DECISIONS.md", "ASTRA_FINAL_PARAMETER_REVIEW.md", "ASTRA_FINAL_RESOLUTION.md",
              "FINAL_RUN_READINESS.md"):
        if (rev / n).exists():   # the reviews quote illustrative user-profile paths; masked in the shared copy only
            text = rev.joinpath(n).read_text(encoding="utf-8")
            masked = re.sub(r"(?i)\b[a-z]:\\users\\[^\s`'\"]*", "<example user-profile path>", text)
            masked = re.sub(r"/(?:Users|home)/[^\s`'\"/]+/", "<example user-profile path>/", masked)
            if masked != text:
                masked = "> Shared copy: example user-profile paths in the text are masked.\n\n" + masked
            D.write_str(tmp / "reviews" / n, masked)
    from falls_ml.phase2.state import write_timings

    write_timings(out)                      # the current attempt's items are included (the run-level file is refreshed again at the end)
    if (out / "RUN_TIMINGS.csv").exists():
        shutil.copyfile(out / "RUN_TIMINGS.csv", tmp / "RUN_TIMINGS.csv")
    audit = json.loads((out / "RESUME_AUDIT.json").read_text(encoding="utf-8"))
    for a in audit["attempts"]:
        if a.get("attempt") == ctx.run.attempt and a.get("status") == "RUNNING":
            a["status"] = "COMPLETE (this attempt built the share package)"
    D.write_json(tmp / "RESUME_AUDIT.json", audit)
    plan = json.loads((out / "PHASE2_PLAN.json").read_text(encoding="utf-8"))
    facts = ctx.facts()
    manifest = {"watermark": WATERMARK, "falls_ml_version": __import__("falls_ml").__version__, "plan_sha256": plan["plan_sha256"],
                "code_sha256_at_start": plan["code_sha256_at_start"], "code_sha256_now": ctx.run.code_sha, "input_name": plan["plan"]["input_name"],
                "input_sha256": plan["plan"]["input_sha256"], "reference_folder": plan["plan"]["reference_folder"], "config_sha256": plan["plan"]["config_sha256"],
                "catalogue_sha256": plan["plan"]["catalogue_sha256"], "feature_sets_sha256": ctx.fs().sha256, "seed": plan["plan"]["seed"],
                "threads": plan["plan"]["threads"], "split": {"strategy": facts["split_strategy"], "seed": facts["split_seed"],
                                                               "train_rows_sha256": facts.get("train_rows_sha256"),
                                                               "validation_rows_sha256": facts.get("validation_rows_sha256"),
                                                               "test_rows_sha256": facts["test_rows_sha256"], "test_rows_dropped": facts["test_rows_dropped"],
                                                               "test_outcomes_read": facts["test_outcomes_read"]},
                "environment": json.loads((A / "ENVIRONMENT.json").read_text(encoding="utf-8")), "stages": sorted(ctx.run.state.get("completed_stages", [])),
                "selection_frozen_sha256": D.sha256_file(out / "SELECTION_FROZEN.json"), "synthetic": facts.get("synthetic", False),
                "final_experiment_config_sha256": plan["plan"].get("final_experiment_config_sha256"),
                "frozen_production_config": plan["plan"].get("frozen_production_config"),
                "validation_opened": json.loads((out / "VALIDATION_OPENED.json").read_text(encoding="utf-8")) if (out / "VALIDATION_OPENED.json").exists() else None,
                "categories": ["HISTORICAL_BASELINE_15", "SAFE_DISCOVERY", "EXPLORATORY_UNRESOLVED_SENSITIVITY"]}
    D.write_json(tmp / "RUN_MANIFEST.json", manifest)
    state = {**ctx.run.state, "status": "COMPLETE"}
    D.write_json(tmp / "RUN_STATE_FINAL.json", state)
    D.write_str(tmp / "README.md", _readme(manifest))
    # ---- privacy + path scan; the folder becomes share/ only when clean (G-08b)
    scan = privacy_scan(tmp, _identifiers(ctx), _pepper(ctx))
    needles = [str(ctx.run.out.resolve()), str(ctx.ref_dir.resolve()), str(ctx.src.resolve()), os.environ.get("USERNAME", ""), getpass.getuser()]
    needles = [n for n in needles if n and len(n) >= 3]
    path_hits = []
    for p in sorted(tmp.rglob("*")):
        if p.is_file() and p.suffix.lower() in (".csv", ".md", ".json", ".txt", ".svg"):
            h = _path_hits(p.read_text(encoding="utf-8", errors="ignore"), needles)
            if h:
                path_hits.append(f"{p.relative_to(tmp).as_posix()}: {sorted(set(h))}")
    if not scan["passed"] or path_hits:
        raise Phase2Stop("PRIVACY_SCAN", "the share package failed the privacy / path scan; nothing was published to share/",
                         [*scan["hits"][:10], *path_hits[:10]])
    D.write_json(tmp / "PRIVACY_SCAN.json", {**scan, "path_scan": "passed", "hits": []})
    from falls_ml.phase2.state import crash_point

    crash_point("S16_share", "share", "before_rename")
    D.rename_dir(tmp, share)
    outs = sorted(p for p in share.rglob("*") if p.is_file())
    st.finalize(outs, {"n_files": len(outs), "privacy_scan": "passed"})


def _readme(m: dict[str, Any]) -> str:
    return "\n".join([
        "# Phase 2 share package", "", f"**{m['watermark']}**", "",
        *(["> SYNTHETIC DATA – SOFTWARE TEST ONLY", ""] if m.get("synthetic") else []),
        *(["> NOT THE FROZEN PRODUCTION CONFIGURATION", ""] if not m.get("frozen_production_config") else []),
        "This folder is the only folder to zip for review. It holds aggregate results only (no member ids, no pseudonyms, no row-level data, no "
        "pepper, no local paths); small cells (< 10) are suppressed; every text file passed the privacy and path scan (PRIVACY_SCAN.json).", "",
        "Three categories are never mixed: HISTORICAL_BASELINE_15 (benchmark), SAFE_DISCOVERY (primary; proven provenance only) and "
        "EXPLORATORY_UNRESOLVED_SENSITIVITY (separate files / sections).", "",
        "| File | Content |", "|---|---|",
        "| MANAGEMENT_SUMMARY_HE.md | Hebrew management summary: additional falls at the same capacity; what each domain added |",
        "| SCIENTIFIC_SUMMARY.md | accounting, benchmark, waterfall, management question, selection, validation, families, shortlist, exploratory, robustness |",
        "| FINAL_EXPERIMENT_CONFIG.json / .sha256 | the frozen effective configuration of this run |",
        "| SELECTION_FROZEN.json / VALIDATION_OPENED.json / VALIDATION_CONFIRMATION.json | frozen TRAIN selection; one-shot VALIDATION opening; pre-declared confirmation |",
        "| COLUMN_FUNNEL.csv | exact column / feature accounting per category |",
        "| FEATURE_REGISTRY.csv / FEATURE_QUALITY.csv | engineered features with provenance contract / data-quality and redundancy per universe |",
        "| FEATURE_CONSENSUS.csv / FEATURE_SHORTLIST.csv | SAFE features: evidence per feature / STRONG + MODERATE shortlist |",
        "| EXPLORATORY_UNRESOLVED_FEATURES.csv | UNRESOLVED features (exploratory only; never in a SAFE model) |",
        "| MODEL_COMPARISON.csv | every configuration (category column): TRAIN out-of-fold and VALIDATION metrics, deltas with paired-bootstrap CIs |",
        "| DOMAIN_INCREMENTAL_GAIN.csv | management waterfall (same VALIDATION rows, matched capacities), one-domain additions, leave-one-domain-out |",
        "| MANAGEMENT_HEADLINE.json | additional recorded events identified at the same capacity (recommended SAFE model vs the BASELINE_15 refit) |",
        "| ATTRIBUTION_CONTRASTS.csv | frozen contrasts: SAFE feature expansion (LASSO) and model family (XGBoost vs best linear, same SAFE set) |",
        "| ABLATION_RESULTS.csv | leave-one-domain-out and individual ablations (TRAIN out-of-fold, paired) |",
        "| OPERATIONAL_CAPACITY.csv / THRESHOLD_RESULTS.csv | capacity 1/3/5/10/20% and probability thresholds (VALIDATION) |",
        "| CALIBRATION_SUMMARY.csv / CALIBRATION_CURVES.csv | calibration indices and grouped curves (no recalibration) |", "| SUBGROUP_SUMMARY.csv | descriptive subgroups |",
        "| tables/ | registries, screens, clusters, fits, XGBoost trials, stability, SHAP, feature sets, validation registry |",
        "| figures/ | slide-ready figures (PNG + SVG) |", "| reviews/ | Astra / AGY reviews, review decisions, final parameter review and resolution, readiness |",
        "| RUN_MANIFEST.json / RUN_STATE_FINAL.json / RESUME_AUDIT.json / RUN_TIMINGS.csv | provenance, final state, every attempt, time per stage |", "",
        f"Final configuration sha256 {str(m.get('final_experiment_config_sha256'))[:16]}…, plan sha256 {m['plan_sha256'][:16]}…, input sha256 "
        f"{m['input_sha256'][:16]}…, test partition dropped before any computation ({m['split']['test_rows_dropped']} rows; outcomes never read).", ""])
