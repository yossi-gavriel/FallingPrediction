"""Phase 3 stages S15 (reports) and S16 (privacy-safe share) for BOTH outcomes:

- Outcome A (GO): the time-contract verification, feature eligibility (verified / attested / bounded), availability assumptions, the models, the frozen selection on the ADVERSE bound, the management question at 10% capacity (additional
  recorded falls, adverse and favourable bounds), the domains that add measurable information, the reused-VALIDATION check, and the DWH list for
  what could not be recovered;
- Outcome B (NO_GO): recovery evidence and the complete DWH remediation requirements; no model was fitted.

Every number printed comes from a committed table; wording rules as in Phase 2 (never "significant", no causal language; banned words checked)."""

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

from falls_ml.phase2 import durable as D
from falls_ml.phase2.stages_report import BANNED, _path_hits
from falls_ml.phase2.state import Phase2Stop
from falls_ml.phase3 import PHASE3_VERSION, WATERMARK, WATERMARK_HE
from falls_ml.phase3.context import P3Ctx
from falls_ml.phase3.recovery import RECOVERED_BOUNDED, RECOVERED_EXACT

REF = "LASSO:P3_BASE"
MIN_CELL = 10
DOMAIN_HE = {"COGNITION": "קוגניציה", "MEDICATION": "תרופות", "FRAILTY": "שבריריות (MEFI)", "FUNCTION_MOBILITY": "תפקוד וניידות",
             "PRIOR_FALLS_DETAIL": "נפילות קודמות (פירוט)", "NURSE_FALL_RISK": "הערכת סיכון נפילה של אחות", "UTILISATION": "שימוש בשירותי בריאות",
             "HOME_ENVIRONMENT": "סביבת הבית", "SOCIAL_SUPPORT": "תמיכה חברתית", "ADDITIONAL_COMORBIDITY": "תחלואה נוספת", "BASELINE_15": "15 המשתנים ההיסטוריים"}
CLASS_HE = {"SAFE_VERIFIED": "זמין – תזמון נבדק", "SAFE_VERIFIED_BOUNDED": "זמין – תזמון נבדק, עם גבולות", "SAFE_ATTESTED": "זמין – לפי אישור מחסן הנתונים",
            "NOT_RECOVERABLE_FUTURE_RECORDS": "לא זמין – רשומות אחרי יום החיזוי", "UNRESOLVED": "תזמון לא נתמך (חקרני בלבד)",
            "NOT_RECOVERABLE_FORBIDDEN": "אסור (מידע שלאחר נקודת החיזוי)", "INELIGIBLE_DATA": "מעט מדי נתונים"}


def _art(ctx: P3Ctx, name: str) -> pd.DataFrame:
    p = ctx.run.artifacts / name
    return pd.read_csv(p) if p.exists() and p.stat().st_size > 1 else pd.DataFrame()


def _j(ctx: P3Ctx, name: str) -> dict[str, Any]:
    p = ctx.run.artifacts / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _f(v: Any, d: int = 3) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "–"
    return "–" if not np.isfinite(x) else f"{x:.{d}f}"


def _s(v: Any, d: int = 3) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "–"
    return "–" if not np.isfinite(x) else f"{x:+.{d}f}"


def _pct(v: Any, d: int = 1) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "–"
    return "–" if not np.isfinite(x) else f"{100 * x:.{d}f}%"


def _cnt(v: Any) -> str:
    try:
        x = int(v)
    except (TypeError, ValueError):
        return "–"
    return f"<{MIN_CELL}" if 0 < x < MIN_CELL else f"{x:,}"


def check_wording(text: str, where: str) -> None:
    low = text.lower()
    hits = [w for w in BANNED if re.search(rf"(?<![\w]){re.escape(w)}(?![\w])", low)]
    if hits:
        raise Phase2Stop("WORDING", f"{where} contains banned wording", hits)


def _md(header: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header), *["| " + " | ".join(str(c) for c in r) + " |" for r in rows]]


# ============================================================================ derived tables
def recovery_summary(reg: pd.DataFrame) -> pd.DataFrame:
    t = reg.groupby(["domain", "phase3_class"]).size().unstack(fill_value=0)
    t["n_features"] = t.sum(axis=1)
    t["n_newly_recovered"] = reg.groupby("domain")["newly_recovered"].sum().reindex(t.index).fillna(0).astype(int)
    p2 = reg.assign(p2_safe=reg["phase2_eligibility"] == "ELIGIBLE").groupby("domain")["p2_safe"].sum()
    t["n_phase2_rule_eligible"] = p2.reindex(t.index).fillna(0).astype(int)
    return t.reset_index()


def domain_table(ctx: P3Ctx) -> pd.DataFrame:
    oof = _art(ctx, "OOF_MODEL_COMPARISON.csv")
    boot = _art(ctx, "OOF_BOOTSTRAP_VS_P3_BASE.csv")
    val = _art(ctx, "VALIDATION_MODEL_COMPARISON.csv")
    abl = _art(ctx, "ABLATION_RESULTS.csv")
    if oof.empty:
        return pd.DataFrame()
    b = boot.set_index("config") if len(boot) else pd.DataFrame()
    v = val.set_index("config") if len(val) else pd.DataFrame()
    rows = []
    sel = oof[(oof["category"] == "PRIMARY_FULL") & (oof["family"] == "LASSO") & oof["set_kind"].isin(["main", "domain_add", "waterfall"])]
    for _, r in sel.iterrows():
        c = r["config"]
        row = {"analysis": {"domain_add": "one_domain_added_to_P3_BASE", "waterfall": "cumulative_waterfall", "main": "all_recovered"}[r["set_kind"]],
               "config": c, "domain": str(r["feature_set"]).replace("P3_BASE_PLUS_", "") if r["set_kind"] == "domain_add" else "",
               "n_new_features": r.get("n_new_features_offered"), "oof_adverse_ap": r.get("adverse_ap"), "oof_favourable_ap": r.get("favourable_ap"),
               "oof_adverse_delta_ap": r.get("adverse_delta_ap"), "oof_favourable_delta_ap": r.get("favourable_delta_ap"),
               "oof_adverse_delta_capture10_pp": 100 * float(r.get("adverse_delta_capture_principal", np.nan)),
               "oof_favourable_delta_capture10_pp": 100 * float(r.get("favourable_delta_capture_principal", np.nan)),
               "oof_adverse_additional_falls_top10": r.get("adverse_additional_falls_principal"),
               "oof_favourable_additional_falls_top10": r.get("favourable_additional_falls_principal"),
               "n_unknown_rows_train": r.get("n_unknown_rows")}
        if len(b) and c in b.index:
            row.update({"oof_adverse_delta_ap_ci_low": b.at[c, "adverse_delta_ap_ci_low"], "oof_adverse_delta_ap_ci_high": b.at[c, "adverse_delta_ap_ci_high"],
                        "oof_adverse_delta_falls_top10_ci_low": b.at[c, "adverse_delta_falls_top10_ci_low"] if "adverse_delta_falls_top10_ci_low" in b else np.nan,
                        "oof_adverse_delta_falls_top10_ci_high": b.at[c, "adverse_delta_falls_top10_ci_high"] if "adverse_delta_falls_top10_ci_high" in b else np.nan})
        if len(v) and c in v.index:
            row.update({"val_adverse_delta_ap": v.at[c, "adverse_delta_ap"], "val_adverse_delta_capture10_pp": 100 * float(v.at[c, "adverse_delta_capture_principal"])})
        rows.append(row)
    if len(abl) and "analysis" in abl:
        for _, r in abl[abl["analysis"] == "leave_one_domain_out"].iterrows():
            rows.append({"analysis": "leave_one_domain_out_from_all_recovered", "config": r.get("set"), "domain": r.get("removed"),
                         "oof_adverse_delta_ap": r.get("adverse_delta_ap_full_minus_ablated"), "oof_favourable_delta_ap": r.get("favourable_delta_ap_full_minus_ablated"),
                         "oof_adverse_delta_ap_ci_low": r.get("adverse_delta_ap_ci_low_full_minus_ablated"),
                         "oof_adverse_delta_ap_ci_high": r.get("adverse_delta_ap_ci_high_full_minus_ablated")})
    return pd.DataFrame(rows)


def headline(ctx: P3Ctx, sel: dict[str, Any]) -> dict[str, Any]:
    oof = _art(ctx, "OOF_MODEL_COMPARISON.csv").set_index("config")
    val = _art(ctx, "VALIDATION_MODEL_COMPARISON.csv")
    val = val.set_index("config") if len(val) else val
    facts = ctx.facts()
    n_train = int(facts["phase3_partitions"]["train"]["n_rows"])
    pop = int(facts["full_labeled_rows"])
    rec = sel["recommended"]
    out: dict[str, Any] = {"recommended": rec, "reference": REF, "selection_outcome": sel["selection_outcome"], "capacity": 0.10,
                           "population_full_labeled": pop, "train_rows": n_train, "train_events": int(facts["phase3_partitions"]["train"]["n_events"])}
    for name, key in ((REF, "reference"), (rec, "recommended")):
        if name in oof.index:
            r = oof.loc[name]
            out[f"{key}_oof_adverse_capture10"] = float(r["adverse_capture@0.1"])
            out[f"{key}_oof_favourable_capture10"] = float(r["favourable_capture@0.1"])
    if rec in oof.index and rec != REF:
        r = oof.loc[rec]
        out["oof_additional_falls_top10_adverse"] = int(r["adverse_additional_falls_principal"])
        out["oof_additional_falls_top10_favourable"] = int(r["favourable_additional_falls_principal"])
        out["scaled_to_population_adverse"] = round(float(r["adverse_additional_falls_principal"]) * pop / n_train)
        out["scaled_to_population_favourable"] = round(float(r["favourable_additional_falls_principal"]) * pop / n_train)
        b = _art(ctx, "OOF_BOOTSTRAP_VS_P3_BASE.csv")
        if len(b) and rec in set(b["config"]):
            bb = b.set_index("config").loc[rec]
            out["oof_additional_falls_top10_adverse_ci"] = [float(bb.get("adverse_delta_falls_top10_ci_low", np.nan)), float(bb.get("adverse_delta_falls_top10_ci_high", np.nan))]
        if len(val) and rec in val.index:
            out["val_additional_falls_top10_adverse"] = int(val.at[rec, "adverse_additional_falls_principal"])
            out["val_additional_falls_top10_favourable"] = int(val.at[rec, "favourable_additional_falls_principal"])
            out["val_false_alerts_top10_adverse"] = int(val.at[rec, "adverse_false_alerts@principal"])
            out["val_false_alerts_top10_reference_favourable"] = int(val.at[REF, "favourable_false_alerts@principal"]) if REF in val.index else None
    return out


# ============================================================================ figures
def figures(ctx: P3Ctx, out: Path) -> list[str]:
    from falls_ml.modelreport.figures import BLUE, GREEN, INK2, MUTED, ORANGE, RED, T, _fig, _style, save

    made: list[str] = []
    reg = _art(ctx, "FEATURE_RECOVERY.csv")
    cat = reg[reg["kind"] == "catalogue"]
    order = [d for d in ctx.cat.domains if d in set(cat["domain"])]
    classes = [("SAFE_VERIFIED", GREEN), ("SAFE_VERIFIED_BOUNDED", BLUE), ("SAFE_ATTESTED", "#eda100"), ("NOT_RECOVERABLE_FUTURE_RECORDS", ORANGE),
               ("UNRESOLVED", MUTED), ("NOT_RECOVERABLE_FORBIDDEN", RED), ("INELIGIBLE_DATA", INK2)]
    for lang in ("en", "he"):
        fig = _fig(9, 5.2)
        ax = fig.add_subplot(111)
        _style(ax)
        left = np.zeros(len(order))
        for cl, col in classes:
            vals = np.array([int(((cat["domain"] == d) & (cat["phase3_class"] == cl)).sum()) for d in order])
            if vals.sum():
                ax.barh(range(len(order)), vals, left=left, color=col, label=T(CLASS_HE[cl], lang) if lang == "he" else cl.replace("_", " ").lower())
                left += vals
        ax.set_yticks(range(len(order)), [T(DOMAIN_HE.get(d, d), lang) if lang == "he" else ctx.cat.domains[d]["label"] for d in order])
        ax.invert_yaxis()
        ax.set_xlabel(T("מספר משתנים", lang) if lang == "he" else "engineered features")
        ax.set_title(T("זמינות משתנים לפי תחום קליני", lang) if lang == "he" else "Feature eligibility by clinical domain (corrected time contract)", fontsize=11)
        ax.legend(fontsize=7, loc="lower right")
        made += list(save(fig, out, f"01_recovery_by_domain_{lang}", WATERMARK if lang == "en" else "PHASE 3", lang=lang).values())
    dom = domain_table(ctx)
    d1 = dom[dom["analysis"] == "one_domain_added_to_P3_BASE"] if len(dom) else dom
    if len(d1):
        for lang in ("en", "he"):
            fig = _fig(9, 0.6 + 0.55 * len(d1))
            ax = fig.add_subplot(111)
            _style(ax)
            yv = np.arange(len(d1))
            lo, hi = d1["oof_adverse_delta_capture10_pp"].to_numpy(dtype=float), d1["oof_favourable_delta_capture10_pp"].to_numpy(dtype=float)
            ax.hlines(yv, lo, hi, color=BLUE, linewidth=6, alpha=0.5)
            ax.plot(lo, yv, "o", color=ORANGE, label=T("גבול שמרני", lang) if lang == "he" else "adverse bound")
            ax.plot(hi, yv, "o", color=GREEN, label=T("גבול אופטימי", lang) if lang == "he" else "favourable bound")
            ax.axvline(0, color=INK2, linewidth=1)
            ax.set_yticks(yv, [T(DOMAIN_HE.get(d, d), lang) if lang == "he" else ctx.cat.domains.get(d, {}).get("label", d) for d in d1["domain"]])
            ax.invert_yaxis()
            ax.set_xlabel(T("שינוי בשיעור הנפילות שזוהו ב-10% העליונים (נקודות אחוז)", lang) if lang == "he" else
                          "change in recorded falls identified in the top 10% vs P3_BASE (percentage points, TRAIN out-of-fold)")
            ax.legend(fontsize=8)
            made += list(save(fig, out, f"02_domain_added_value_{lang}", WATERMARK if lang == "en" else "PHASE 3", lang=lang).values())
    cap = _art(ctx, "OOF_OPERATIONAL_CAPACITY.csv")
    sel = json.loads((ctx.run.out / "SELECTION_FROZEN.json").read_text(encoding="utf-8")) if (ctx.run.out / "SELECTION_FROZEN.json").exists() else None
    if len(cap) and sel:
        fig = _fig(8, 4.6)
        ax = fig.add_subplot(111)
        _style(ax)
        for c, col in ((REF, INK2), (sel["recommended"], BLUE)):
            for mode, ls in (("adverse", "-"), ("favourable", "--")):
                t = cap[(cap["config"] == c) & (cap["bound"] == mode)]
                if len(t):
                    ax.plot(100 * t["capacity"], 100 * t["sensitivity"], ls, marker="o", color=col, label=f"{c} ({mode})")
        ax.set_xlabel("capacity: % of patients flagged")
        ax.set_ylabel("% of recorded falls identified (TRAIN out-of-fold)")
        ax.legend(fontsize=7)
        made += list(save(fig, out, "03_capture_by_capacity", WATERMARK).values())
    return made


# ============================================================================ texts
def _evaluation_history_lines(ctx: P3Ctx) -> list[str]:
    eh = _j(ctx, "EVALUATION_HISTORY.json")
    if not eh:
        return []
    return [f"- TEST: {eh['TEST']['status']} ({eh['TEST']['recorded_releases_of_the_reference_test_rows']} recorded releases of the reference TEST rows); "
            f"{eh['TEST']['phase3_use']}.",
            f"- VALIDATION: {eh['VALIDATION']['status']}; {eh['VALIDATION']['phase3_use']}.",
            f"- TRAIN: {eh['TRAIN']['phase3_use']}.", f"- {eh['conclusion']}"]


def _tc(ctx: P3Ctx) -> dict[str, Any]:
    return _j(ctx, "TIME_CONTRACT_VERIFICATION.json")


def _time_contract_lines(ctx: P3Ctx) -> list[str]:
    tc = _tc(ctx)
    if not tc:
        return []
    v1, v2, v3 = tc.get("V1_label_excludes_index_day", {}), tc.get("V2_window_end", {}), tc.get("V3_no_post_index_records", {})
    v4, v6, v7 = tc.get("V4_days_since_consistency", {}), tc.get("V6_boundary_episode", {}), tc.get("V7_proxy_episode_audit", {})
    on = {c: v.get("n_on_index") for c, v in (v3.get("by_column") or {}).items() if v.get("n_on_index")}
    after = {c: v.get("n_after_index") for c, v in (v3.get("by_column") or {}).items() if v.get("n_after_index")}
    rows = [["V1 outcome excludes Index_Date (HARD)", "PASS" if v1.get("passed") else "FAIL",
             f"{_cnt(v1.get('positives_event_on_or_before_index'))} of {_cnt(v1.get('positives'))} positive labels with an event on/before Index_Date; "
             f"minimum days to event {v1.get('min_days_to_event')}"],
            ["V2 window end = Index_Date + 180 (HARD)", "PASS" if v2.get("passed") else "FAIL", str(v2.get("label_end_minus_index_days"))],
            ["V3 no predictor record after Index_Date (attestation)", "PASS" if v3.get("passed") else "FAIL",
             f"records after Index_Date: {({c: _cnt(n) for c, n in after.items()}) or 'none'}; ON Index_Date (history now): {({c: _cnt(n) for c, n in on.items()}) or 'none'}"],
            ["V4 Days_Since_* consistent", "PASS" if v4.get("passed") else "WARN", str({k: v for k, v in (v4.get("by_column") or {}).items() if v.get("n_mismatch") or v.get("n_negative")}) or "consistent"],
            ["V6 boundary episodes (TRAIN)", "FLAG" if v6.get("investigation") else "PASS", f"day +1 events among patients with an index-day record {v6.get('day1_events_among_them')} "
             f"of {_cnt(v6.get('train_rows_with_index_day_record'))} rows; rate ratio {v6.get('rate_ratio')}"],
            ["V7 fall-recency proxy / episode audit (TRAIN)", "FLAG" if v7.get("investigation") else "PASS", str(v7.get("interpretation"))]]
    return ["## 1. The corrected time contract and its verification on this extract", "",
            "The DWH developer explained that the extract is built for prediction at the END of Index_Date (predictors include records through 2025-01-01) "
            "and that the 180-day outcome starts on 2025-01-02. Phases 1-2 assumed the START of the index day and therefore treated index-day records as "
            "future information (the D-00 exclusions). Phase 3 adopts the corrected contract only because the read-only checks below confirm it:", "",
            *_md(["check", "result", "evidence (TRAIN + VALIDATION; TEST never read)"], rows), "",
            "Event timing is not information availability: a record dated before the index day may still have been entered later (late documentation, "
            "backdated registry memberships, claims lag, scores computed at extraction). The extract has no entry timestamps, so availability is an explicit "
            "assumption per source (FEATURE_RECOVERY.csv columns availability_risk / availability_assumption); features whose declared risk is HIGH or "
            "UNKNOWN leave the pre-declared LOW_RISK sensitivity.", ""]


def _reconciliation_lines(ctx: P3Ctx) -> list[str]:
    rc = _j(ctx, "COHORT_FACTS.json").get("cohort_reconciliation") or {}
    if not rc:
        return []
    return ["## 2. Cohort reconciliation (exact)", "",
            *_md(["step", "rows"], [["extract rows", f"{rc.get('extract_rows'):,}"], ["on the index date", f"{rc.get('rows_on_index_date'):,}"],
                                    ["eligible on the index date", f"{rc.get('eligible_on_index_date'):,}"],
                                    ["without a usable 180-day label (censored)", f"{rc.get('label_null_eligible'):,} {rc.get('label_null_by_reason')}"],
                                    ["FULL_LABELED (Phase 3 population)", f"{rc.get('full_labeled'):,}"], ["of which D00_CLEAN (Phases 1-2)", f"{rc.get('d00_clean'):,}"],
                                    ["of which re-included D-00 rows", f"{rc.get('d00_rows'):,} (reference build removed {rc.get('reference_rows_removed_for_d00')})"]]), "",
            f"All reconciliation checks passed: {rc.get('passed')}. {rc.get('residual_selection_note')}.", ""]


def _pair_delta(oof: pd.DataFrame, new: str, base: str) -> tuple[float, float]:
    """Adverse ΔAP and adverse Δcapture@10% of ``new`` vs ``base`` from the bounded bundles (new adverse - base favourable)."""
    if new not in set(oof["config"]) or base not in set(oof["config"]):
        return np.nan, np.nan
    a, b = oof.set_index("config").loc[new], oof.set_index("config").loc[base]
    return float(a["adverse_ap"] - b["favourable_ap"]), float(a["adverse_capture@0.1"] - b["favourable_capture@0.1"])


def robustness(ctx: P3Ctx) -> dict[str, Any]:
    """Pre-declared robustness statements of the FEATURE-EXPANSION gain (LASSO): does it survive without attested sources, without HIGH / UNKNOWN
    availability-risk sources, and without the fall-recency features? Same patients, same rules; point estimates on the adverse bound."""
    oof = _art(ctx, "OOF_MODEL_COMPARISON.csv")
    if oof.empty:
        return {}
    sel = ctx.cfg["selection"]
    need_cap = float(sel["min_delta_capture_principal_pp"]) / 100.0
    out: dict[str, Any] = {}
    for key, new, base in (("primary", "LASSO:P3_ALL_RECOVERED", REF), ("verified_only", "LASSO:P3_VERIFIED_ALL", "LASSO:P3_VERIFIED_BASE"),
                           ("low_availability_risk", "LASSO:P3_LOWRISK_ALL", "LASSO:P3_LOWRISK_BASE"),
                           ("without_fall_recency", "LASSO:P3_ALL_NO_FALL_RECENCY", REF)):
        dap, dcap = _pair_delta(oof, new, base)
        out[key] = {"new": new, "base": base, "adverse_delta_ap": dap, "adverse_delta_capture10": dcap,
                    "meets_min_capture_gain": bool(np.isfinite(dcap) and dcap >= need_cap)}
    p = out["primary"]["meets_min_capture_gain"]
    out["verdict"] = {
        "attested_sources": ("ROBUST" if out["verified_only"]["meets_min_capture_gain"] else "DEPENDS_ON_ATTESTED_SOURCES") if p else "NO_PRIMARY_GAIN",
        "availability_risk": ("ROBUST" if out["low_availability_risk"]["meets_min_capture_gain"] else "DEPENDS_ON_HIGH_RISK_SOURCES") if p else "NO_PRIMARY_GAIN",
        "fall_recency": ("ROBUST" if out["without_fall_recency"]["meets_min_capture_gain"] else "DEPENDS_ON_FALL_RECENCY") if p else "NO_PRIMARY_GAIN",
        "rule": f"a gain is ROBUST to a restriction when the restricted pair still meets the pre-declared +{sel['min_delta_capture_principal_pp']} pp capture@10% "
                "(adverse bound); otherwise the gain is NOT ESTABLISHED until that information is confirmed"}
    return out


def scientific_summary(ctx: P3Ctx) -> str:
    facts = ctx.facts()
    feas = _j(ctx, "FEASIBILITY_DECISION.json")
    reg = _art(ctx, "FEATURE_RECOVERY.csv")
    tim = _art(ctx, "SOURCE_TIMING.csv")
    dfe = _art(ctx, "DOMAIN_FEASIBILITY.csv")
    p2 = _j(ctx, "PHASE2_RULE_ON_PHASE2_POPULATION.json")
    go = feas.get("decision") == "GO"
    P = facts["phase3_partitions"]
    L = [f"# Phase 3 scientific summary – corrected time contract {'and FULL extended modelling' if go else '(no modelling: NO_GO)'}", "", f"**{WATERMARK}**", "",
         *(["> SYNTHETIC DATA – SOFTWARE TEST ONLY", ""] if facts.get("synthetic") else []),
         f"falls_ml Phase 3 {PHASE3_VERSION}. Outcome **{feas.get('outcome')}** ({feas.get('decision')}).", "",
         *_time_contract_lines(ctx), *_reconciliation_lines(ctx),
         "## 3. Population and evaluation history", "",
         f"- FULL_LABELED: TRAIN {P['train']['n_rows']:,} rows ({P['train']['n_events']:,} events; {P['train']['n_d00_rows']:,} re-included D-00 rows), VALIDATION "
         f"{P['validation']['n_rows']:,} rows ({P['validation']['n_events']:,} events). TEST ({facts['test_rows_dropped']:,} reference rows + "
         f"{facts['extra_test_rows_dropped']:,} TEST-assigned D-00 rows) was dropped in memory and never read.",
         "- Every direct comparison (baseline vs expanded) is on the SAME FULL_LABELED patients with the same rules. D00_CLEAN results of Phases 1-2 are "
         "historical reference only; the bridge columns restrict the SAME predictions to the D00_CLEAN patients.",
         "- Re-including the D-00 rows removes the selection on index-day records; it does not remove every selection: FULL_LABELED still requires a usable "
         "label (censoring, death, leaving the HMO) and the cohort eligibility of the VIEW.",
         *_evaluation_history_lines(ctx), "",
         "## 4. Source timing (record dates only; no outcome)", ""]
    rows = []
    for _, t in tim.iterrows():
        if str(t.get("record_date_column") or "") and str(t.get("root_cause")) not in ("NO_RECORD_DATE", "POST_INDEX"):
            rows.append([t["source"], t["record_date_column"], t.get("historical_semantics", ""), _cnt(t.get("n_on_index")), _cnt(t.get("n_after_index")), t.get("root_cause", "")])
    L += _md(["source", "record date", "semantics", "rows ON the index day (history)", "rows AFTER it", "status"], rows) + [""]
    L += ["## 5. Feature eligibility (93 engineered features + the 15 historical predictors)", ""]
    if len(reg):
        cl = reg.groupby(["kind", "phase3_class"]).size().unstack(fill_value=0)
        L += _md(["kind", *cl.columns], [[k, *[str(int(v)) for v in cl.loc[k]]] for k in cl.index]) + [""]
        el = reg[reg["phase3_class"].isin(["SAFE_VERIFIED", "SAFE_VERIFIED_BOUNDED", "SAFE_ATTESTED"])]
        rk = el.groupby(["phase3_class", "availability_risk"]).size().unstack(fill_value=0)
        L += ["Declared information-availability risk of the eligible features (event timing verified or attested; availability assumed):", "",
              *_md(["class", *rk.columns], [[k, *[str(int(v)) for v in rk.loc[k]]] for k in rk.index]), ""]
    if p2:
        L += [f"Phase 2 rule recomputed with the unchanged Phase 2 code on the Phase 2 population: {json.dumps(p2.get('catalogue_eligibility_counts'))}; "
              f"historical predictors SAFE under that rule: {p2.get('n_baseline_safe')}. Compare with the Phase 2 preflight printout.", ""]
    nr = reg[reg["newly_recovered"].astype(bool)] if len(reg) else reg
    L += [f"Newly eligible (not usable under the Phase 2 rule): {len(nr)} feature(s).", ""]
    L += ["## 6. Scientific feasibility (pre-declared, before any model)", "", f"Rule: {feas.get('rule')}", ""]
    if len(dfe):
        L += _md(["domain", "priority", "eligible (verified / attested / bounded)", "newly eligible items", "KNOWN informative TRAIN rows", "events", "GO"],
                 [[r["domain"], "yes" if r["priority_domain"] else "no", f"{int(r['n_eligible'])} ({int(r['n_verified'])} / {int(r['n_attested'])} / {int(r['n_bounded'])})",
                   int(r["n_newly_eligible_items"]), _cnt(r["n_known_informative_train"]), _cnt(r["events_known_informative_train"]),
                   "GO" if r["domain_go"] else "–"] for _, r in dfe.iterrows()]) + [""]
    L += [f"**Decision: {feas.get('decision')}** (GO domains: {', '.join(feas.get('go_domains') or []) or 'none'}).", ""]
    if go:
        L += _modelling_sections(ctx)
    else:
        L += ["## 7. Outcome B", "", "No additional clinical domain met the pre-declared feasibility rule on this extract, so no model was fitted. "
              "The warehouse actions are in DWH_REMEDIATION.md / tables/DWH_REMEDIATION_REQUIREMENTS.csv.", ""]
    L += ["## Limitations", "",
          "- Exploratory 180-day RECORDED fall/fracture label (not the eFalls outcome); censored rows excluded; capture = recorded events identified, not falls prevented.",
          "- Internal development estimates only (nested CV inside TRAIN) plus a descriptive check on a REUSED partition; no independent or prospective validation.",
          "- The corrected time contract rests on the DWH developer's explanation verified by V1-V9; SAFE_ATTESTED sources rest additionally on the statement that "
          "no event after Index_Date enters the extract; information availability (entry lag) is an assumption for every source (Q-P3-02).",
          "- Rows with a post-index or unbounded value (if any) are fitted out and evaluated with bounds (XGBoost bounds are outer bounds).",
          "- Assessment-conditional nurse / MEFI fields also reflect who was assessed (care process); no causal interpretation.", "",
          "## Next scientific step (freeze list for an independent test)", "",
          "- A new extract with index dates AFTER the current label windows under the same written time contract, with entry timestamps or record dates for "
          "the attested sources (DWH_REMEDIATION.md).",
          "- Freeze now: the eligible feature definitions, the selected configuration and its hyper-parameters, the 10% capacity rule; test once.", ""]
    return "\n".join(L) + "\n"


def _modelling_sections(ctx: P3Ctx) -> list[str]:
    sel = json.loads((ctx.run.out / "SELECTION_FROZEN.json").read_text(encoding="utf-8"))
    oof = _art(ctx, "OOF_MODEL_COMPARISON.csv")
    head = headline(ctx, sel)
    dom = domain_table(ctx)
    vchk = _j(ctx, "VALIDATION_CHECK.json")
    rob = robustness(ctx)
    prox = _art(ctx, "PROXY_ABLATION_OOF.csv")
    pdc = _j(ctx, "PROXY_DOMINANCE_CHECK.json")
    L = ["## 7. Models (TRAIN nested grouped CV; outer out-of-fold; identical FULL_LABELED patients; adverse / favourable bounds)", ""]
    show = oof[~oof["failed"].astype(bool)].sort_values(["category", "set_kind", "config"]) if len(oof) else oof
    L += _md(["config", "category", "AP adv / fav", "AUROC adv / fav", "capture@10% adv / fav", "ΔAP vs P3_BASE (adv)", "Δcapture@10% pp (adv)", "UNKNOWN rows"],
             [[r["config"], r["category"], f"{_f(r['adverse_ap'])} / {_f(r['favourable_ap'])}", f"{_f(r['adverse_auroc'])} / {_f(r['favourable_auroc'])}",
               f"{_pct(r['adverse_capture@0.1'])} / {_pct(r['favourable_capture@0.1'])}", _s(r["adverse_delta_ap"]),
               _s(100 * float(r["adverse_delta_capture_principal"]), 1), _cnt(r.get("n_unknown_rows"))] for _, r in show.iterrows()]) + [""]
    L += ["## 8. Frozen selection (adverse bound; VALIDATION not read)", "", f"Outcome: **{sel['selection_outcome']}** – {sel['reason']}.",
          f"Shortlist: {', '.join(sel['shortlist'])}.", ""]
    L += ["## 9. The management question: same 10% capacity, same patients", ""]
    if head.get("oof_additional_falls_top10_adverse") is not None:
        ci = head.get("oof_additional_falls_top10_adverse_ci") or [np.nan, np.nan]
        L += [f"TRAIN out-of-fold ({head['train_rows']:,} patients, {head['train_events']:,} recorded falls): at the same 10% capacity the recommended model "
              f"identifies at least **{head['oof_additional_falls_top10_adverse']:+d}** more recorded falls than P3_BASE (adverse bound; 95% bootstrap interval "
              f"{_s(ci[0], 0)} to {_s(ci[1], 0)}) and at most {head['oof_additional_falls_top10_favourable']:+d} (favourable bound). Scaled to the "
              f"{head['population_full_labeled']:,}-patient population per 180 days: {head['scaled_to_population_adverse']:+d} to "
              f"{head['scaled_to_population_favourable']:+d} (a scaling, not a separate estimate).", ""]
        if head.get("val_additional_falls_top10_adverse") is not None:
            L += [f"Reused VALIDATION (descriptive): {head['val_additional_falls_top10_adverse']:+d} to {head['val_additional_falls_top10_favourable']:+d} "
                  f"additional recorded falls at 10%; check: {vchk.get('check')} – {vchk.get('detail')}.", ""]
    else:
        L += ["No configuration met the pre-declared rule over P3_BASE on the adverse bound: the expanded information did not identify a robust number of "
              "additional recorded falls at 10% capacity in this extract.", ""]
    if rob:
        L += ["## 10. Robustness of the feature-expansion gain (LASSO; pre-declared restrictions; identical patients)", "",
              *_md(["restriction", "pair", "ΔAP (adv)", "Δcapture@10% pp (adv)", "meets the +1 pp rule"],
                   [[k, f"{v['new']} vs {v['base']}", _s(v["adverse_delta_ap"]), _s(100 * v["adverse_delta_capture10"], 1), "yes" if v["meets_min_capture_gain"] else "no"]
                    for k, v in rob.items() if k != "verdict"]), "",
              f"Verdicts: attested sources **{rob['verdict']['attested_sources']}**, availability risk **{rob['verdict']['availability_risk']}**, fall recency "
              f"**{rob['verdict']['fall_recency']}** ({rob['verdict']['rule']}).", ""]
    L += ["## 11. The Phase 2 PROXY_DOMINANCE finding (falls_days_since_last)", "",
          "Phase 2 stopped because falls_days_since_last held 89% of the XGBoost permutation AP loss. Phase 3 checks whether that signal is history or one "
          "episode counted twice (V7: Last_Fall_Date / Days_Since_Last_Fall / Index_Date / Next_Fall_Date_180D, TIME_CONTRACT_VERIFICATION.json) and refits "
          "without the fall-recency features on the same patients:", ""]
    if len(prox):
        L += _md(["family", "full vs without fall recency", "ΔAP adv (95% CI)", "Δ recorded falls @10% adv (95% CI)"],
                 [[r["family"], f"{r['full']} vs {r['without_fall_recency']}",
                   f"{_s(r.get('adverse_delta_ap_full_minus_ablated'))} ({_s(r.get('adverse_delta_ap_ci_low_full_minus_ablated'))} to {_s(r.get('adverse_delta_ap_ci_high_full_minus_ablated'))})",
                   f"{_s(r.get('adverse_delta_falls_top10_full_minus_ablated'), 0)} ({_s(r.get('adverse_delta_falls_top10_ci_low_full_minus_ablated'), 0)} to "
                   f"{_s(r.get('adverse_delta_falls_top10_ci_high_full_minus_ablated'), 0)})"] for _, r in prox.iterrows()]) + [""]
    if pdc:
        L += [f"Dominance in the tuned XGBoost: {pdc.get('dominance') or 'none'}; V7 flagged: {pdc.get('v7_investigation')}; stop raised on: {pdc.get('stopped_on') or 'none'}. "
              "A dominant clinical predictor is not by itself leakage; the V7 audit and the ablation above are the evidence.", ""]
    L += ["## 12. Which clinical domains add measurable information (P3_BASE + one domain; LASSO)", ""]
    d1 = dom[dom["analysis"] == "one_domain_added_to_P3_BASE"] if len(dom) else dom
    if len(d1):
        L += _md(["domain", "ΔAP adv (95% CI)", "ΔAP fav", "Δcapture@10% pp adv / fav", "additional falls @10% adv / fav"],
                 [[r["domain"], f"{_s(r['oof_adverse_delta_ap'])} ({_s(r.get('oof_adverse_delta_ap_ci_low'))} to {_s(r.get('oof_adverse_delta_ap_ci_high'))})",
                   _s(r["oof_favourable_delta_ap"]), f"{_s(r['oof_adverse_delta_capture10_pp'], 1)} / {_s(r['oof_favourable_delta_capture10_pp'], 1)}",
                   f"{_s(r['oof_adverse_additional_falls_top10'], 0)} / {_s(r['oof_favourable_additional_falls_top10'], 0)}"] for _, r in d1.iterrows()]) + [""]
    L += ["Leave-one-domain-out and individual ablations: ABLATION_RESULTS.csv. Stability: STABILITY.csv. XGBoost permutation importance and SHAP (KNOWN rows): "
          "PERMUTATION_IMPORTANCE.csv, SHAP_SUMMARY.csv. Importance is predictive association conditional on the other features, not causation.", ""]
    return L


def management_he(ctx: P3Ctx) -> str:
    facts = ctx.facts()
    feas = _j(ctx, "FEASIBILITY_DECISION.json")
    reg = _art(ctx, "FEATURE_RECOVERY.csv")
    tc = _tc(ctx)
    go = feas.get("decision") == "GO"
    cat = reg[reg["kind"] == "catalogue"] if len(reg) else reg
    n_ver = int(cat["phase3_class"].isin(["SAFE_VERIFIED", "SAFE_VERIFIED_BOUNDED"]).sum()) if len(cat) else 0
    n_att = int((cat["phase3_class"] == "SAFE_ATTESTED").sum()) if len(cat) else 0
    L = ["# שלב 3 – סיכום להנהלה", "", f"**{WATERMARK_HE}**", "", *(["> נתונים סינתטיים – בדיקת תוכנה בלבד", ""] if facts.get("synthetic") else []),
         "## מה השתנה", "",
         "- לפי הסבר מפתח מחסן הנתונים, התמצית בנויה לחיזוי בסוף יום 1.1.2025: המידע הקליני כולל את היום הזה, והתוצא (נפילה בתוך 180 יום) מתחיל ב-2.1.2025.",
         f"- הבדיקות על הנתונים: נפילות ב-1.1 אינן בתוצא – {'אושר' if (tc.get('V1_label_excludes_index_day') or {}).get('passed') else 'לא אושר'}; "
         f"אין רשומות אחרי יום החיזוי – {'אושר' if (tc.get('V3_no_post_index_records') or {}).get('passed') else 'לא אושר'}.",
         f"- לכן נכללו גם {facts['d00_rows_in_full_labeled']:,} המטופלים שהוצאו בשלבים הקודמים (סה\"כ {facts['full_labeled_rows']:,}).", "",
         "## מה זמין עכשיו", "",
         f"- {n_ver} משתנים שתזמונם נבדק ברמת המטופל, ו-{n_att} משתנים שנשענים על אישור מחסן הנתונים (ללא תאריך ברמת המטופל, למשל תרופות ורישומים).",
         "- לגבי כל מקור מתועדת הנחת זמינות: רשומה עם תאריך מוקדם עשויה להיות מוזנת מאוחר יותר. מקורות בסיכון גבוה (למשל רישומי מחלות) נבדקים גם בניתוח ללא אותם מקורות.", ""]
    dfe = _art(ctx, "DOMAIN_FEASIBILITY.csv")
    if len(dfe):
        L += ["| תחום | זמינים | חדשים | החלטה |", "|---|---|---|---|"]
        L += [f"| {DOMAIN_HE.get(r['domain'], r['domain'])} | {int(r['n_eligible'])} | {int(r['n_newly_eligible_items'])} | {'להמשיך' if r['domain_go'] else '–'} |"
              for _, r in dfe.iterrows()]
        L += [""]
    if not go:
        L += ["## החלטה: לא ממשיכים לאימון (תוצאה B)", "", "- הבדיקות לא אפשרו ניסוי מורחב מבוסס. לא אומן מודל. הפעולות הנדרשות במחסן הנתונים: DWH_REMEDIATION.md.", ""]
    else:
        sel = json.loads((ctx.run.out / "SELECTION_FROZEN.json").read_text(encoding="utf-8"))
        h = headline(ctx, sel)
        rob = robustness(ctx)
        L += ["## החלטה: הומשך לאימון המורחב (תוצאה A)", ""]
        if h.get("oof_additional_falls_top10_adverse") is not None:
            L += [f"- באותה קיבולת (10% מהמטופלים) ועל אותם מטופלים, המודל המורחב זיהה לפחות {h['oof_additional_falls_top10_adverse']:+d} ועד "
                  f"{h['oof_additional_falls_top10_favourable']:+d} נפילות רשומות נוספות לעומת מודל הבסיס (אימות צולב פנימי, {h['train_rows']:,} מטופלים).",
                  f"- בהיקף האוכלוסייה (כ-{h['population_full_labeled']:,} מטופלים, 180 יום): בערך {h['scaled_to_population_adverse']:+d} עד "
                  f"{h['scaled_to_population_favourable']:+d} (הערכה בקנה מידה)."]
            if rob:
                v = rob["verdict"]
                L += [f"- יציבות: ללא מקורות מאושרים-בלבד – {'נשמר' if v['attested_sources'] == 'ROBUST' else 'לא נשמר'}; ללא מקורות בסיכון זמינות גבוה – "
                      f"{'נשמר' if v['availability_risk'] == 'ROBUST' else 'לא נשמר'}; ללא משתני נפילה אחרונה – {'נשמר' if v['fall_recency'] == 'ROBUST' else 'לא נשמר'}."]
            L += [""]
        else:
            L += ["- אף תצורה מורחבת לא עמדה בכלל שנקבע מראש בתרחיש השמרני.", ""]
        dom = domain_table(ctx)
        d1 = dom[dom["analysis"] == "one_domain_added_to_P3_BASE"] if len(dom) else dom
        if len(d1):
            L += ["## מה כל תחום הוסיף (תוספת לבסיס, 10% העליונים, נקודות אחוז)", "", "| תחום | שמרני | אופטימי |", "|---|---|---|"]
            L += [f"| {DOMAIN_HE.get(r['domain'], r['domain'])} | {_s(r['oof_adverse_delta_capture10_pp'], 1)} | {_s(r['oof_favourable_delta_capture10_pp'], 1)} |"
                  for _, r in d1.iterrows()]
            L += [""]
    L += ["## הסתייגויות", "", "- אומדני פיתוח פנימיים; אין סט נתונים שלא נבדק. נדרש תיקוף על תקופה מאוחרת יותר לפני כל שימוש קליני.",
          "- תוצא חקרני: נפילה או שבר רשומים תוך 180 יום (לא התוצא של eFalls). זיהוי נפילה רשומה אינו מניעת נפילה.",
          "- המשתנה 'ימים מאז הנפילה האחרונה' נבדק במיוחד (ממצא שלב 2) – ראו הסיכום המדעי.",
          "- משתנים שנבחרו מבטאים קשר סטטיסטי, לא סיבה.", ""]
    return "\n".join(L) + "\n"


def scientific_he(ctx: P3Ctx) -> str:
    facts = ctx.facts()
    feas = _j(ctx, "FEASIBILITY_DECISION.json")
    reg = _art(ctx, "FEATURE_RECOVERY.csv")
    tc = _tc(ctx)
    L = ["# שלב 3 – סיכום מדעי", "", f"**{WATERMARK_HE}**", "",
         "## חוזה הזמן המתוקן", "",
         "חיזוי בסוף יום האינדקס (1.1.2025): משתנים מבוססים על רשומות עד יום זה כולל; התוצא מתחיל ביום שלמחרת. בשלבים 1–2 הונח חיזוי בתחילת היום, "
         "ולכן רשומות מיום האינדקס סווגו כמידע עתידי. שלב 3 מאמץ את החוזה המתוקן רק לאחר בדיקות על הנתונים (ללא קריאת סט הבדיקה):", "",
         f"- V1 נפילות ביום האינדקס אינן בתוצא: {'עבר' if (tc.get('V1_label_excludes_index_day') or {}).get('passed') else 'נכשל'}",
         f"- V2 סוף חלון התוצא = יום האינדקס + 180: {'עבר' if (tc.get('V2_window_end') or {}).get('passed') else 'נכשל'}",
         f"- V3 אין רשומות אחרי יום האינדקס: {'עבר' if (tc.get('V3_no_post_index_records') or {}).get('passed') else 'נכשל'}",
         f"- V7 בדיקת 'ימים מאז הנפילה האחרונה' (אירוע אחד שנספר פעמיים?): {(tc.get('V7_proxy_episode_audit') or {}).get('interpretation')}", "",
         "## תזמון אירוע לעומת זמינות מידע", "",
         "אישור מחסן הנתונים מתייחס לתאריכי האירועים. זמינות המידע בפועל (הזנה מאוחרת, רישום רטרואקטיבי, עיכוב חשבונות) אינה ניתנת לבדיקה בתמצית; "
         "לכל מקור הוגדרו הנחה ורמת סיכון. משתנים 'מאושרים' (ללא תאריך ברמת המטופל) מדווחים בנפרד, ונבדק ניתוח רגישות ללא מקורות בסיכון גבוה.", "",
         "## אוכלוסייה", "", f"- FULL_LABELED: {facts['full_labeled_rows']:,} מטופלים (כולל {facts['d00_rows_in_full_labeled']:,} שהוצאו קודם). השוואות – על אותם מטופלים בלבד.",
         "- סט הבדיקה אינו נטען; סט האימות שימש בעבר ומדווח כבדיקה תיאורית בלבד.", "", "## זמינות משתנים", ""]
    if len(reg):
        t = reg.groupby("phase3_class").size()
        L += ["| סיווג | משתנים |", "|---|---|", *[f"| {CLASS_HE.get(k, k)} | {int(v)} |" for k, v in t.items()], ""]
    L += [f"## החלטת היתכנות: {feas.get('decision')}", "", f"תחומים שעברו: {', '.join(DOMAIN_HE.get(d, d) for d in feas.get('go_domains') or []) or 'אין'}.", ""]
    if feas.get("decision") == "GO" and (ctx.run.out / "SELECTION_FROZEN.json").exists():
        sel = json.loads((ctx.run.out / "SELECTION_FROZEN.json").read_text(encoding="utf-8"))
        L += [f"## בחירה קפואה (לפני פתיחת סט האימות): {sel['selection_outcome']}", "", f"מודל מומלץ: {sel['recommended']}.", ""]
    L += ["## מגבלות", "", "- אומדני פיתוח פנימיים; אין תיקוף עצמאי או פרוספקטיבי.", "- זמינות המידע היא הנחה לכל מקור (אין חותמות זמן הזנה).",
          "- פעולות נדרשות במחסן הנתונים: DWH_REMEDIATION.md.", ""]
    return "\n".join(L) + "\n"


def dwh_report(ctx: P3Ctx) -> str:
    rem = _art(ctx, "DWH_REMEDIATION_REQUIREMENTS.csv")
    feas = _j(ctx, "FEASIBILITY_DECISION.json")
    L = ["# DWH requirements and questions (Phase 3)", "", f"**{WATERMARK}**", "",
         f"Decision of this run: **{feas.get('decision')}** (outcome {feas.get('outcome')}). Written for both outcomes.", "",
         "## Questions to confirm in writing", "",
         "- **Q-P3-01 (time contract):** predictors include records through the END of Index_Date; the 180-day outcome starts on Index_Date + 1 day and "
         "excludes falls on Index_Date; no clinical event dated after Index_Date enters any column (please confirm per source, including medications, "
         "registries, the comorbidity group and benefit status).",
         "- **Q-P3-02 (availability):** are values built from EVENT dates or from ENTRY / LOAD dates? Can a record dated before Index_Date have been entered after "
         "it (late nurse documentation, backdated registry memberships, reimbursement claims, laboratory imports, MEFI batch computation)?",
         "- **Q-P3-03 (fall episodes):** is one injury episode (e.g. an ED visit followed by admission or follow-up) de-duplicated before it becomes the 'last "
         "fall' and the 'next fall'? What is the event definition of Next_Fall_Date_180D?", "",
         "## Per source", ""]
    for _, r in rem.iterrows() if len(rem) else []:
        L += [f"### {r['source']} – {r['source_text']}", "",
              f"- Evidence kind: {r['evidence_kind']}; semantics: {r.get('historical_semantics', '')}; record date: {r.get('record_date_column') or 'none'}; status: "
              f"{r.get('root_cause', '')} (rows ON the index day {_cnt(r.get('n_on_index'))}, AFTER it {_cnt(r.get('n_after_index'))}).",
              f"- Domains: {r['domain_labels']}{' (priority)' if r['priority_domain'] else ''}.",
              f"- Not eligible ({int(r['n_features_blocked'])}): {r['features_blocked'] or '–'}.",
              f"- Eligible with bounds ({int(r['n_features_bounded'])}): {r['features_bounded'] or '–'}.",
              f"- Eligible on the attestation only ({int(r['n_features_attested_only'])}): {r['features_attested_only'] or '–'}.",
              f"- **Required action:** {r['required_fix']}", ""]
    L += ["## בעברית (תקציר)", "", "- נדרש אישור בכתב לחוזה הזמן (חיזוי בסוף יום האינדקס; תוצא מהיום שלמחרת; אין אירועים אחרי יום האינדקס).",
          "- נדרש לברר אם הערכים נבנו לפי תאריך אירוע או תאריך הזנה, ואם אירוע נפילה אחד עשוי להיספר גם כנפילה קודמת וגם כתוצא.",
          "- למקורות ללא תאריך (תרופות, מעבדה, רישומים, מדד תחלואה) יש להוסיף את תאריך הרשומה התורמת האחרונה.", ""]
    return "\n".join(L) + "\n"


# ============================================================================ S15
def s15_report(ctx: P3Ctx) -> None:
    st = ctx.run.stage("15", "report", ("14",) if ctx.go() else ("05",))
    if st.complete_record():
        return

    def fn(tmp: Path, seed: int) -> dict[str, Any]:
        reg = _art(ctx, "FEATURE_RECOVERY.csv")
        D.write_csv(tmp / "RECOVERY_SUMMARY.csv", recovery_summary(reg))
        texts = {"SCIENTIFIC_SUMMARY.md": scientific_summary(ctx), "SCIENTIFIC_SUMMARY_HE.md": scientific_he(ctx), "MANAGEMENT_SUMMARY_HE.md": management_he(ctx),
                 "DWH_REMEDIATION.md": dwh_report(ctx)}
        if ctx.go():
            D.write_csv(tmp / "DOMAIN_INCREMENTAL_VALUE.csv", domain_table(ctx))
            sel = json.loads((ctx.run.out / "SELECTION_FROZEN.json").read_text(encoding="utf-8"))
            D.write_json(tmp / "MANAGEMENT_HEADLINE.json", headline(ctx, sel))
            D.write_json(tmp / "ROBUSTNESS.json", robustness(ctx))
        for n, t in texts.items():
            check_wording(t, n)
            D.write_str(tmp / n, t)
        return {"go": ctx.go(), "files": sorted(p.name for p in tmp.iterdir())}

    res = st.item("report", fn)
    outs = [D.write_bytes(ctx.run.artifacts / n, (st.path("report") / n).read_bytes()) for n in res["files"]]
    figs = ctx.run.out / "figures"
    if not (figs / ".done").exists():
        tmpf = ctx.run.out / f".tmp-figures-a{ctx.run.attempt}"
        if tmpf.exists():
            shutil.rmtree(tmpf)
        made = figures(ctx, tmpf)
        D.write_json(tmpf / ".done", {"files": made})
        if figs.exists():
            D.rename_dir(figs, ctx.run.out / "stages" / f"_old_figures_a{ctx.run.attempt}")
        D.rename_dir(tmpf, figs)
    outs += [p for p in sorted(figs.iterdir()) if p.is_file() and p.name != ".done"]
    st.finalize(outs, {"go": res["go"]})


# ============================================================================ S16 share (small cells suppressed; privacy + path scan fails closed)
COUNT_COLUMNS = ("n_on_index", "n_after_index", "n_recorded_without_date", "n_unknown", "n_unknown_train", "n_on_or_after_train", "n_on_or_after_validation",
                 "n_on_or_after_in_d00_clean", "n_on_or_after_outside_d00_clean", "n_train_rows_on_or_after", "events_on_or_after", "n_train_rows_other",
                 "events_other", "n_known_informative_train", "events_known_informative_train", "n_unknown_rows", "n_unknown_heldout",
                 "n_known_observed", "n_known_observed_train", "n_known_informative", "n_with_date", "n_before_index", "n_null_date")
DERIVED = {"n_unknown": ("unknown_share_population", "unknown_share_informative", "unknown_pct"), "events_on_or_after": ("outcome_rate_on_or_after_pct", "rate_ratio"),
           "n_train_rows_on_or_after": ("outcome_rate_on_or_after_pct", "rate_ratio"), "events_other": ("outcome_rate_other_pct", "rate_ratio")}


def suppress_counts(df: pd.DataFrame, min_cell: int = MIN_CELL) -> pd.DataFrame:
    out = df.copy().astype(object)
    for c in [c for c in COUNT_COLUMNS if c in out.columns]:
        x = pd.to_numeric(out[c], errors="coerce")
        small = (x > 0) & (x < min_cell)
        out.loc[small, c] = f"<{min_cell}"
        for d in DERIVED.get(c, ()):
            if d in out.columns:
                out.loc[small, d] = "suppressed"
    return out


COUNT_KEY = re.compile(r"(^n_|_n$|events|rows|positives|index_day_falls|^le_\d+d$|with_label|recorded)")


def suppress_json(obj: Any, key: str = "", min_cell: int = MIN_CELL, inherited: bool = False) -> Any:
    """Counts 1..min_cell-1 under count-like keys (or inside a count-like dictionary such as counts by reason) become '<min_cell>' (recursively);
    shares / rates / flags are kept."""
    hit = inherited or bool(COUNT_KEY.search(key or "")) or key.endswith(("_by_reason", "_counts", "by_guard_column"))
    if isinstance(obj, dict):
        return {k: suppress_json(v, str(k), min_cell, hit and not isinstance(v, dict)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [suppress_json(v, key, min_cell, inherited) for v in obj]
    if isinstance(obj, (int, np.integer)) and not isinstance(obj, bool) and 0 < int(obj) < min_cell and (inherited or COUNT_KEY.search(key or "")):
        return f"<{min_cell}"
    return obj


def s16_share(ctx: P3Ctx) -> None:
    from falls_ml.eda.runner import privacy_scan
    from falls_ml.phase2.stages_report import _identifiers, _pepper, _suppress
    from falls_ml.phase2.state import crash_point, write_timings

    st = ctx.run.stage("16", "share", ("15",))
    if st.complete_record():
        return
    out = ctx.run.out
    share = out / "share"
    if share.exists():
        D.rename_dir(share, out / "stages" / f"_incomplete_share_a{ctx.run.attempt}")
    tmp = out / f".tmp-share-a{ctx.run.attempt}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    A = ctx.run.artifacts
    for n in ("SCIENTIFIC_SUMMARY.md", "SCIENTIFIC_SUMMARY_HE.md", "MANAGEMENT_SUMMARY_HE.md", "DWH_REMEDIATION.md", "MANAGEMENT_HEADLINE.json",
              "FEASIBILITY_DECISION.json", "EVALUATION_HISTORY.json", "SCHEMA_VERIFICATION.json", "PHASE2_RULE_ON_PHASE2_POPULATION.json", "VALIDATION_CHECK.json",
              "FEATURE_SETS.json", "ENVIRONMENT.json", "ROBUSTNESS.json", "PROXY_DOMINANCE_CHECK.json"):
        if (A / n).exists():
            shutil.copyfile(A / n, tmp / n)
    for n in ("TIME_CONTRACT_VERIFICATION.json", "COHORT_FACTS.json"):
        if (A / n).exists():
            D.write_json(tmp / n, suppress_json(json.loads((A / n).read_text(encoding="utf-8"))))
    for n in ("PHASE3_FINAL_EXPERIMENT_CONFIG.json", "PHASE3_FINAL_EXPERIMENT_CONFIG.sha256", "SELECTION_FROZEN.json", "VALIDATION_OPENED.json"):
        if (out / n).exists():
            shutil.copyfile(out / n, tmp / n)
    (tmp / "tables").mkdir()
    for n in ("FEATURE_RECOVERY.csv", "SOURCE_TIMING.csv", "DWH_REMEDIATION_REQUIREMENTS.csv", "DOMAIN_FEASIBILITY.csv", "SELECTION_BIAS_DIAGNOSTIC.csv",
              "FEATURE_COVERAGE.csv", "RECOVERY_SUMMARY.csv", "OOF_MODEL_COMPARISON.csv", "OOF_BOOTSTRAP_VS_P3_BASE.csv", "VALIDATION_MODEL_COMPARISON.csv",
              "VALIDATION_BOOTSTRAP_VS_P3_BASE.csv", "DOMAIN_INCREMENTAL_VALUE.csv", "ABLATION_RESULTS.csv", "STABILITY.csv", "PERMUTATION_IMPORTANCE.csv",
              "SHAP_SUMMARY.csv", "UNIVARIATE_SCREEN.csv", "REDUNDANCY_CLUSTERS.csv", "FITS_LASSO.csv", "FITS_ENET.csv", "FITS_XGB.csv", "XGB_TRIALS.csv",
              "PROXY_ABLATION_OOF.csv"):
        if (A / n).exists():
            t = pd.read_csv(A / n)
            if n == "UNIVARIATE_SCREEN.csv":
                t = t  # already suppressed at source (Phase 2 screen)
            D.write_csv(tmp / "tables" / n, suppress_counts(t))
    for n, rule in (("OOF_OPERATIONAL_CAPACITY.csv", "confusion"), ("VALIDATION_OPERATIONAL_CAPACITY.csv", "confusion"), ("CALIBRATION_CURVES.csv", "calibration"),
                    ("SUBGROUP_SUMMARY.csv", "subgroup")):
        if (A / n).exists() and (A / n).stat().st_size > 1:
            D.write_csv(tmp / "tables" / n, _suppress(pd.read_csv(A / n), rule, MIN_CELL))
    if (out / "figures").exists():
        shutil.copytree(out / "figures", tmp / "figures", ignore=shutil.ignore_patterns(".done"))
    from falls_ml.paths import resolve_path

    from falls_ml.errors import ConfigError

    for doc in ("planning/PHASE3_DESIGN.md", "planning/PHASE3_TIME_CONTRACT_AUDIT.md"):
        try:
            src_doc = resolve_path(doc)
        except ConfigError:
            continue
        (tmp / "docs").mkdir(exist_ok=True)
        shutil.copyfile(src_doc, tmp / "docs" / Path(doc).name)
    write_timings(out)
    if (out / "RUN_TIMINGS.csv").exists():
        shutil.copyfile(out / "RUN_TIMINGS.csv", tmp / "RUN_TIMINGS.csv")
    audit = json.loads((out / "RESUME_AUDIT.json").read_text(encoding="utf-8"))
    D.write_json(tmp / "RESUME_AUDIT.json", audit)
    plan = json.loads((out / "PHASE3_PLAN.json").read_text(encoding="utf-8"))
    facts = ctx.facts()
    manifest = {"watermark": WATERMARK, "phase3_version": PHASE3_VERSION, "falls_ml_version": __import__("falls_ml").__version__,
                "outcome": _j(ctx, "FEASIBILITY_DECISION.json").get("outcome"), "decision": _j(ctx, "FEASIBILITY_DECISION.json").get("decision"),
                "plan_sha256": plan["plan_sha256"], "code_sha256_at_start": plan["code_sha256_at_start"], "code_sha256_now": ctx.run.code_sha,
                "input_sha256": plan["plan"]["input_sha256"], "final_experiment_config_sha256": plan["plan"].get("final_experiment_config_sha256"),
                "frozen_production_config": plan["plan"].get("frozen_production_config"), "seed": plan["plan"]["seed"], "threads": plan["plan"]["threads"],
                "population": {"full_labeled_rows": facts["full_labeled_rows"], "partitions": facts["phase3_partitions"]},
                "test": {"reference_rows_dropped": facts["test_rows_dropped"], "d00_rows_dropped": facts["extra_test_rows_dropped"], "outcomes_read": False},
                "synthetic": facts.get("synthetic", False), "stages": sorted(ctx.run.state.get("completed_stages", []))}
    D.write_json(tmp / "RUN_MANIFEST.json", manifest)
    D.write_str(tmp / "README.md", _readme(manifest))
    scan = privacy_scan(tmp, _identifiers(ctx), _pepper(ctx))
    needles = [str(ctx.run.out.resolve()), str(ctx.ref_dir.resolve()), str(ctx.src.resolve()), os.environ.get("USERNAME", ""), getpass.getuser()]
    needles = [n for n in needles if n and len(n) >= 3]
    path_hits = []
    for p in sorted(tmp.rglob("*")):
        if p.is_file() and p.suffix.lower() in (".csv", ".md", ".json", ".txt", ".svg"):
            h = _path_hits(p.read_text(encoding="utf-8", errors="ignore"), needles)
            if h:
                path_hits.append(f"{p.relative_to(tmp).as_posix()}: {sorted(set(h))}")
    forbidden_files = [p.relative_to(tmp).as_posix() for p in tmp.rglob("*") if p.is_file() and p.suffix.lower() in (".parquet", ".pkl", ".npz", ".sqlite", ".jsonl")]
    row_level = [p.relative_to(tmp).as_posix() for p in tmp.rglob("*.csv") if "research_id" in pd.read_csv(p, nrows=0).columns]
    if not scan["passed"] or path_hits or forbidden_files or row_level:
        raise Phase2Stop("PRIVACY_SCAN", "the share package failed the privacy / path / file-type scan; nothing was published to share/",
                         [*scan["hits"][:10], *path_hits[:10], *[f"forbidden file type: {f}" for f in forbidden_files], *[f"row-level column: {f}" for f in row_level]])
    D.write_json(tmp / "PRIVACY_SCAN.json", {**scan, "path_scan": "passed", "file_type_scan": "passed", "row_level_scan": "passed", "hits": []})
    crash_point("S16_share", "share", "before_rename")
    D.rename_dir(tmp, share)
    outs = sorted(p for p in share.rglob("*") if p.is_file())
    st.finalize(outs, {"n_files": len(outs), "privacy_scan": "passed"})


def _readme(m: dict[str, Any]) -> str:
    return "\n".join([
        "# Phase 3 share package", "", f"**{m['watermark']}**", "", *(["> SYNTHETIC DATA – SOFTWARE TEST ONLY", ""] if m.get("synthetic") else []),
        *(["> NOT THE FROZEN PRODUCTION CONFIGURATION", ""] if not m.get("frozen_production_config") else []),
        f"Outcome **{m.get('outcome')}** ({m.get('decision')}). Aggregate results only: no identifiers, no pseudonyms, no row-level data, no model objects, "
        "no pepper, no raw logs, no local paths; counts 1-9 are suppressed; PRIVACY_SCAN.json records the fail-closed scan.", "",
        "| File | Content |", "|---|---|",
        "| MANAGEMENT_SUMMARY_HE.md | Hebrew management summary |", "| SCIENTIFIC_SUMMARY.md / SCIENTIFIC_SUMMARY_HE.md | scientific summary (English / Hebrew) |",
        "| DWH_REMEDIATION.md, tables/DWH_REMEDIATION_REQUIREMENTS.csv | what the warehouse must correct, per source |",
        "| FEASIBILITY_DECISION.json, tables/DOMAIN_FEASIBILITY.csv | the pre-declared GO / NO_GO decision per clinical domain |",
        "| tables/FEATURE_RECOVERY.csv, tables/SOURCE_TIMING.csv, tables/FEATURE_COVERAGE.csv | row-level recovery evidence |",
        "| tables/SELECTION_BIAS_DIAGNOSTIC.csv | why deleting patients with on/after-index records would bias the sample (descriptive, TRAIN) |",
        "| EVALUATION_HISTORY.json | which partitions were examined before; no untouched holdout exists |",
        "| tables/OOF_*.csv, SELECTION_FROZEN.json | TRAIN nested-CV results with adverse / favourable bounds; the frozen selection |",
        "| tables/VALIDATION_*.csv, VALIDATION_OPENED.json, VALIDATION_CHECK.json | one-shot descriptive check on the REUSED validation partition |",
        "| MANAGEMENT_HEADLINE.json, tables/DOMAIN_INCREMENTAL_VALUE.csv | additional recorded falls at 10% capacity; domain value |",
        "| tables/STABILITY.csv, ABLATION_RESULTS.csv, PERMUTATION_IMPORTANCE.csv, SHAP_SUMMARY.csv | robustness and importance |",
        "| PHASE3_FINAL_EXPERIMENT_CONFIG.json / .sha256 | the frozen configuration of this run |", "| figures/ | PNG + SVG figures |",
        "| RUN_MANIFEST.json, RESUME_AUDIT.json, RUN_TIMINGS.csv | provenance, attempts, time per stage |", ""])
