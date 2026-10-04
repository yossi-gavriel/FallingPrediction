"""Phase 4 reports (English scientific / Hebrew management summary), figures, and the aggregate-only share folder with a fail-closed privacy scan.

Share rule: aggregate results only - no identifier, no row key, no row-level prediction, no raw data, no model object, no local path; counts
1-9 suppressed (and every rate that would give them back)."""

from __future__ import annotations

import json
import math
import os
import getpass
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from falls_ml.artifacts import utc_now
from falls_ml.phase2 import durable as D
from falls_ml.phase2.state import Phase2Stop
from falls_ml.phase4 import PHASE4_VERSION, WATERMARK, WATERMARK_HE
from falls_ml.phase4.common import MODEL_MANIFEST, PRED_MANIFEST, dirs, read_json
from falls_ml.phase4.metrics import pct

MIN_CELL = 10
COUNT_RE = re.compile(r"(^n$|^n_|_n$|events|_rows|^rows|positives|negatives|index_day_fall|recorded|censored_or|n_recent|scored|usable_rows|"
                      r"falls_captured|^tp$|^fp$|^fn$|^tn$|false_alerts|n_selected|n_population|events_population|in_both|new_in|not_in|cohort_|eligible)")
ROW_LEVEL_COLUMNS = {"key", "research_id", "Customer_Full_ID", "Snapshot_Key", "member_id"}


def _small(v: Any) -> bool:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return False
    return math.isfinite(x) and 0 < x < MIN_CELL


def suppress_obj(obj: Any, key: str = "") -> Any:
    """Counts 1-9 under count-like keys become '<10' (recursively); min_/max_/share/rate values are kept."""
    if isinstance(obj, dict):
        hit = key.endswith(("_by_reason", "_counts", "index_dates", "definition_version"))
        return {k: (suppress_obj(v, str(k)) if not hit or isinstance(v, dict) else ("<10" if _small(v) and not isinstance(v, bool) else v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [suppress_obj(v, key) for v in obj]
    if isinstance(obj, (int, np.integer)) and not isinstance(obj, bool) and not key.startswith(("min_", "max_")) and COUNT_RE.search(key) and _small(obj):
        return "<10"
    return obj


def suppress_capacity(t: pd.DataFrame) -> pd.DataFrame:
    out = t.copy().astype(object)
    for i, r in out.iterrows():
        if any(_small(r.get(c)) for c in ("tp", "fp", "fn", "tn", "n_selected")):
            for c in ("tp", "fp", "fn", "tn", "falls_captured", "false_alerts", "n_selected"):
                if c in out.columns:
                    out.at[i, c] = "<10"
            for c in ("capture", "ppv", "lift", "pct_selected"):
                if c in out.columns:
                    out.at[i, c] = "suppressed"
    return out


def suppress_calibration(t: pd.DataFrame) -> pd.DataFrame:
    out = t.copy().astype(object)
    for i, r in out.iterrows():
        n, ev = pd.to_numeric(r.get("n"), errors="coerce"), pd.to_numeric(r.get("events"), errors="coerce")
        if _small(ev) or _small(n - ev) or _small(n):
            for c in ("events", "observed_rate", "ci_low", "ci_high"):
                if c in out.columns:
                    out.at[i, c] = "<10" if c == "events" else "suppressed"
    return out


def suppress_comparison(t: pd.DataFrame) -> pd.DataFrame:
    out = t.copy().astype(object)
    for i, r in out.iterrows():
        for c in ("n", "events"):
            if c in out.columns and _small(r.get(c)):
                out.at[i, c] = "<10"
        for c in [c for c in out.columns if c.startswith(("falls_captured_", "n_selected_"))]:
            if _small(r.get(c)) or _small(pd.to_numeric(r.get("events"), errors="coerce") - pd.to_numeric(r.get(c.replace("n_selected_", "falls_captured_")), errors="coerce")
                                          if c.startswith("falls_captured_") else False):
                tag = c.split("_")[-1]
                for cc in (f"falls_captured_{tag}", f"n_selected_{tag}"):
                    out.at[i, cc] = "<10"
                for cc in [x for x in out.columns if x.endswith(tag) and x.startswith(("capture_", "ppv_", "lift_"))]:
                    out.at[i, cc] = "suppressed"
    return out


# ============================================================================ figures
def figures(out_dir: Path, res: dict[str, Any], mm: dict[str, Any], est: dict[str, Any]) -> list[str]:
    from falls_ml.modelreport.figures import BLUE, GREEN, MUTED, ORANGE, T, _fig, _style, save

    made: list[str] = []
    cal = res["calibration"]
    cols = [BLUE, ORANGE, GREEN, MUTED]
    if len(cal):
        for lang in ("en", "he"):
            fig = _fig(6.2, 5.4)
            ax = fig.add_subplot(111)
            _style(ax)
            mx = 0.0
            for j, (c, g) in enumerate(cal.groupby("model", sort=False)):
                ok = (pd.to_numeric(g["events"], errors="coerce") >= MIN_CELL) & ((pd.to_numeric(g["n"]) - pd.to_numeric(g["events"], errors="coerce")) >= MIN_CELL)
                g = g[ok]
                ax.plot(g["mean_predicted"], g["observed_rate"], "o-", color=cols[j % 4], label=c.split(":")[1], ms=4)
                mx = max(mx, float(np.nanmax(g[["mean_predicted", "observed_rate"]].to_numpy(dtype=float))) if len(g) else 0.0)
            ax.plot([0, mx * 1.05], [0, mx * 1.05], "--", color=MUTED, lw=1)
            ax.set_xlabel(T("סיכון חזוי ממוצע (עשירון)", lang) if lang == "he" else "mean predicted risk (decile)")
            ax.set_ylabel(T("שיעור נפילות בפועל", lang) if lang == "he" else "observed 180-day rate")
            ax.set_title(T("כיול 2026 של המודלים המוקפאים", lang) if lang == "he" else "2026 calibration of the frozen 2025 models", fontsize=11)
            ax.legend(fontsize=8)
            made += list(save(fig, out_dir, f"01_calibration_2026_{lang}", WATERMARK if lang == "en" else "PHASE 4", lang=lang).values())
    cap = res["capacity"]
    prim = mm["primary"]
    capp = cap[(cap["model"] == prim) & (cap["population"] == "ALL_2026_USABLE") & cap["rule"].str.startswith("top")] if len(cap) else cap
    e3 = ((est.get("models") or {}).get(prim) or {}).get("adverse") or {}
    if len(capp):
        capp = capp[capp["assignment"].isin(["exact", "adverse"])]
        for lang in ("en", "he"):
            fig = _fig(6.6, 4.4)
            ax = fig.add_subplot(111)
            _style(ax)
            xs = np.arange(len(capp))
            v26 = capp["capture"].to_numpy(dtype=float)
            v25 = np.array([e3.get(f"capture_top{round(100 * q):g}", np.nan) for q in capp["capacity"]], dtype=float)
            ax.bar(xs - 0.18, 100 * v25, 0.36, color=MUTED, label=T("2025 (אומדן פנימי)", lang) if lang == "he" else "2025 internal estimate")
            ax.bar(xs + 0.18, 100 * v26, 0.36, color=BLUE, label=T("2026 (תיקוף זמני)", lang) if lang == "he" else "2026 temporal validation")
            ax.set_xticks(xs, [f"{100 * q:g}%" for q in capp["capacity"]])
            ax.set_xlabel(T("קיבולת (אחוז המטופלים המסומנים)", lang) if lang == "he" else "capacity (share of patients flagged)")
            ax.set_ylabel(T("אחוז מהנפילות שנתפסו", lang) if lang == "he" else "% of recorded falls captured")
            ax.set_title(T(f"{prim.split(':')[1]}: נפילות שנתפסו", lang) if lang == "he" else f"{prim}: recorded falls captured", fontsize=11)
            ax.legend(fontsize=8)
            made += list(save(fig, out_dir, f"02_capture_2025_vs_2026_{lang}", WATERMARK if lang == "en" else "PHASE 4", lang=lang).values())
    return made


# ============================================================================ summaries
def _isnum(z: Any) -> bool:
    try:
        return math.isfinite(float(z))
    except (TypeError, ValueError):
        return False


def _fmt(metric: str, v: Any, lo: Any = None, hi: Any = None, *, change: bool = False) -> str:
    """Counts as integers; capture / PPV / prevalence as percentages (changes in percentage points); other metrics with 3 decimals."""
    if not _isnum(v):
        return "–"
    if metric in ("n", "events"):
        return f"{int(round(float(v))):+d}" if change else str(int(round(float(v))))
    if metric.startswith(("capture", "ppv")) or metric == "prevalence":
        if change:
            return f"{100 * float(v):+.1f} pp"
        return pct(v, 1) + (f" ({pct(lo, 1)}–{pct(hi, 1)})" if _isnum(lo) and _isnum(hi) else "")
    if change:
        return f"{float(v):+.3f}"
    return f"{float(v):.3f}" + (f" ({float(lo):.3f}–{float(hi):.3f})" if _isnum(lo) and _isnum(hi) else "")


def _row(res: dict[str, Any], model: str, pop: str = "ALL_2026_USABLE") -> dict[str, Any]:
    t = res["comparison"]
    m = t[(t["model"] == model) & (t["population"] == pop) & t["assignment"].isin(["exact", "adverse"])]
    return m.iloc[0].to_dict() if len(m) else {}


def _ci(r: dict[str, Any], k: str, digits: int = 3) -> str:
    v, lo, hi = r.get(k), r.get(f"{k}_ci_low"), r.get(f"{k}_ci_high")
    try:
        return f"{float(v):.{digits}f} ({float(lo):.{digits}f}–{float(hi):.{digits}f})"
    except (TypeError, ValueError):
        return "–"


def _cip(r: dict[str, Any], k: str) -> str:
    v, lo, hi = r.get(k), r.get(f"{k}_ci_low"), r.get(f"{k}_ci_high")
    return f"{pct(v)} ({pct(lo)}–{pct(hi)})" if v is not None else "–"


def summary_en(res: dict[str, Any], oc: dict[str, Any], mm: dict[str, Any], est: dict[str, Any], overlap: dict[str, Any], man: dict[str, Any], cfg: Any) -> str:
    prim = mm["primary"]
    r = _row(res, prim)
    L = ["# Phase 4 – temporal validation 2026 of the frozen Phase 3 models", "", f"**{WATERMARK}**", "",
         "## What was done", "",
         f"- The models frozen in Phase 3 (developed on Index_Date {cfg['development_index_date']}) were applied unchanged to the 2026 snapshot "
         f"(Index_Date {cfg['index_date']}). Primary: **{prim}**; pre-declared secondary: {', '.join(c for c in mm['models'] if c != prim) or 'none'}"
         + (f"; excluded before scoring by the pre-declared preflight rule: {', '.join(mm.get('excluded_models', {}))}" if mm.get("excluded_models") else "") + ".",
         *[f"- {c}: artifact mode **{m['mode']}** ({m['mode_text']})." for c, m in mm["models"].items()],
         "- No 2026 outcome was used for training, tuning, feature selection, thresholds, calibration, model choice, imputation or schema adaptation. "
         f"The predictions were frozen and hashed before the outcomes were opened (predictions sha256 `{man['predictions_sha256'][:16]}…`).",
         "- Capacity rule: the top 1% / 5% / 10% of the evaluated 2026 patients by predicted risk (the frozen Phase 3 rule); the absolute 2025 cut-offs "
         "are reported as a secondary operational view.", "",
         "## Cohort and overlap", "",
         f"- 2026 scored (eligible) patients: {overlap.get('n_2026_cohort_eligible')}; usable outcome population: "
         f"{oc['checks']['O6_censoring_usable_population']['usable_rows']} ({pct(oc['checks']['O6_censoring_usable_population']['usable_share'])}); "
         f"censored / unlabelled: {oc['checks']['O6_censoring_usable_population']['censored_or_unlabelled_rows']} (reported, excluded from every metric).",
         f"- Also in the 2025 cohort: {overlap.get('n_in_both')} ({overlap.get('pct_2026_previously_in_2025')}% of 2026); new in 2026: {overlap.get('n_new_in_2026')}; "
         f"2025 patients not in 2026: {overlap.get('n_2025_not_in_2026')}.",
         "- Because patients overlap, this is a **temporal validation of a later snapshot**, not an independent external population.", "",
         "## Outcome contract 2026", "", f"- {oc['contract']}: **PASSED** (no index-day event counted as outcome; window end Index_Date + 180; "
                                         "positives within follow-up; label / date consistency). See OUTCOME_CONTRACT_2026.json, including the episode audit.", "",
         f"## Primary model {prim} on 2026 (95% CI, patient bootstrap)", "",
         "| Metric | 2026 |", "|---|---|",
         f"| AUROC | {_ci(r, 'auroc')} |", f"| Average precision (PR-AUC) | {_ci(r, 'ap')} |", f"| Brier score | {_ci(r, 'brier', 4)} |",
         f"| Brier skill vs development prevalence | {_ci(r, 'brier_skill_vs_development_prevalence')} |", f"| Log loss | {_ci(r, 'logloss', 4)} |",
         f"| Calibration intercept | {_ci(r, 'cal_intercept')} |", f"| Calibration slope | {_ci(r, 'cal_slope')} |",
         *[f"| Capture@{t} / PPV@{t} / Lift@{t} | {_cip(r, f'capture_top{t}')} / {_cip(r, f'ppv_top{t}')} / {_ci(r, f'lift_top{t}', 2)} |" for t in ("1", "5", "10")],
         f"| Event prevalence (usable) | {pct(r.get('prevalence'), 2)} ({r.get('events')} of {r.get('n')}) |", ""]
    v = res["vs_phase3"]
    if len(v):
        vp = v[v["model"] == prim]
        L += ["## 2025 (Phase 3 internal estimate) vs 2026 (temporal validation)", "",
              "| Metric | Phase 3 internal (95% CI) | 2026 (95% CI) | Change | Interpretation |", "|---|---|---|---|---|"]
        for _, x in vp.iterrows():
            L.append(f"| {x['metric_label']} | {_fmt(x['metric'], x['phase3_internal_estimate'], x.get('phase3_ci_low'), x.get('phase3_ci_high'))} | "
                     f"{_fmt(x['metric'], x['temporal_2026'], x.get('temporal_2026_ci_low'), x.get('temporal_2026_ci_high'))} | "
                     f"{_fmt(x['metric'], x['absolute_change'], change=True)} | {x['interpretation']} |")
        L += ["", f"Rule (pre-declared): {cfg['comparison']['rule']}", "",
              "The Phase 3 number is an INTERNAL development estimate (nested cross-validation inside 2025 TRAIN); the 2026 number is the first "
              "evaluation of the frozen model on a later snapshot. A drop is expected (optimism, case-mix, coding drift) and is not by itself a model failure.", ""]
    L += ["## Secondary frozen models", ""]
    for c in mm["models"]:
        if c == prim:
            continue
        rr = _row(res, c)
        L.append(f"- {c}: AUROC {_ci(rr, 'auroc')}, AP {_ci(rr, 'ap')}, Brier {_ci(rr, 'brier', 4)}, Capture@10% {_cip(rr, 'capture_top10')}.")
    pv = res["paired_vs_primary"]
    if len(pv):
        for _, x in pv.iterrows():
            L.append(f"  - {x['model']} − {x['reference']} (conservative bound, paired patient bootstrap): ΔAP {x.get('adverse_delta_ap', float('nan')):.4f} "
                     f"({x.get('adverse_delta_ap_ci_low', float('nan')):.4f}–{x.get('adverse_delta_ap_ci_high', float('nan')):.4f}); "
                     f"Δ falls@10% {x.get('adverse_delta_falls_top10', float('nan')):.0f}.")
    L += ["", "## Pre-declared overlap subgroups (same model, same flags)", ""]
    t = res["comparison"]
    for sub in ("A_SEEN_IN_2025", "B_NEW_IN_2026"):
        s = t[(t["model"] == prim) & (t["population"] == sub)]
        if len(s) and "note" in s.columns and isinstance(s.iloc[0].get("note"), str):
            L.append(f"- {sub}: {s.iloc[0]['note']}")
        elif len(s):
            x = s.iloc[0].to_dict()
            L.append(f"- {sub}: n {x.get('n')}, events {x.get('events')}; AUROC {_ci(x, 'auroc')}, AP {_ci(x, 'ap')}, calibration slope {_ci(x, 'cal_slope')}.")
    L += ["", "## Limitations", "",
          "- Exploratory outcome: a RECORDED fall / fracture within 180 days (not the eFalls outcome). A recorded fall is not prevented by a prediction.",
          "- Overlapping patients: the 2026 cohort is not independent of 2025; report the new-patient subgroup B for the cleanest view.",
          "- Censored patients (death / leaving within the window) have no usable label and are excluded, as in development.",
          "- New 2026 predictors (NEW_2026_FEATURES_CATALOGUE.csv) were catalogued only: to use them, backfill them for 2025, develop on 2025, and "
          "validate once on a still-unseen later snapshot.", "",
          "Files: TEMPORAL_MODEL_COMPARISON.csv, TEMPORAL_VS_PHASE3.csv, TEMPORAL_CAPACITY.csv, TEMPORAL_CALIBRATION.csv, PAIRED_VS_PRIMARY.csv, "
          "COHORT_OVERLAP.json, OUTCOME_CONTRACT_2026.json, PHASE4_PREFLIGHT.md, SCHEMA_COMPARISON_2025_2026.csv, PREDICTOR_SHIFT.csv, "
          "NEW_2026_FEATURES_CATALOGUE.csv, FROZEN_MODELS.json, RUN_MANIFEST.json, PRIVACY_SCAN.json, figures/.", ""]
    return "\n".join(L)


HE_METRIC = {"capture_top10": "נפילות שנתפסו ב-10% העליונים", "ppv_top10": "דיוק בקרב המסומנים (10%)", "lift_top10": "פי כמה מהשיעור הכללי (10%)",
             "auroc": "יכולת הבחנה (AUROC)", "ap": "דיוק ממוצע (AP)", "brier": "ציון Brier (נמוך = טוב)", "prevalence": "שיעור נפילות",
             "n": "מספר מטופלים", "events": "מספר נפילות"}
HE_INTERP = {"WITHIN UNCERTAINTY": "בתוך טווח אי-הוודאות", "CI-SUPPORTED CHANGE": "שינוי נתמך סטטיסטית", "descriptive": "תיאורי", "NO INTERVAL": "–"}


def summary_he(res: dict[str, Any], oc: dict[str, Any], mm: dict[str, Any], est: dict[str, Any], overlap: dict[str, Any], cfg: Any) -> str:
    prim = mm["primary"]
    r = _row(res, prim)
    v = res["vs_phase3"]
    vp = v[v["model"] == prim].set_index("metric") if len(v) else pd.DataFrame()

    def g(k: str, col: str) -> Any:
        return vp.at[k, col] if k in vp.index else None

    L = ["# שלב 4 – תיקוף זמני 2026 – סיכום להנהלה", "", f"**{WATERMARK_HE}**", "",
         "## מה נעשה", "",
         f"- המודל שהוקפא בשלב 3 (פותח על נתוני 1.1.2025) הופעל ללא שום שינוי על תמונת המצב של 1.1.2026. המודל הראשי: {prim.split(':')[1]}.",
         "- התחזיות הוקפאו ונחתמו (SHA-256) לפני שנפתחו התוצאות של 2026. שום החלטה (משתנים, ספים, כיול, בחירת מודל) לא התקבלה על סמך 2026.", "",
         "## האוכלוסייה", "",
         f"- {overlap.get('n_2026_cohort_eligible')} מטופלים זכאים ב-2026; מהם {float(overlap.get('pct_2026_previously_in_2025') or 0):.1f}% היו גם באוכלוסיית 2025. "
         "לכן זהו תיקוף זמני (תמונת מצב מאוחרת יותר), לא אוכלוסייה בלתי תלויה.",
         f"- אוכלוסיית ההערכה (עם תוצא ידוע): {oc['checks']['O6_censoring_usable_population']['usable_rows']} "
         f"({pct(oc['checks']['O6_censoring_usable_population']['usable_share'])}); שיעור נפילות רשומות תוך 180 יום: {pct(r.get('prevalence'), 2)}.", "",
         "## התוצאה העיקרית", "",
         f"- כשמסמנים 10% מהמטופלים בעלי הסיכון הגבוה ביותר, נתפסות {_cip(r, 'capture_top10')} מהנפילות הרשומות"
         + (f" (באומדן הפנימי של 2025: {pct(g('capture_top10', 'phase3_internal_estimate'))})." if g("capture_top10", "phase3_internal_estimate") is not None else "."),
         f"- מתוך המסומנים ב-10%, {_cip(r, 'ppv_top10')} נפלו בפועל (פי {_ci(r, 'lift_top10', 1)} מהשיעור הכללי).",
         f"- ב-1% העליונים: נתפסות {_cip(r, 'capture_top1')} מהנפילות; ב-5%: {_cip(r, 'capture_top5')}.",
         f"- יכולת הבחנה (AUROC): {_ci(r, 'auroc')}" + (f" (2025: {g('auroc', 'phase3_internal_estimate'):.3f})." if g("auroc", "phase3_internal_estimate") is not None else "."),
         f"- כיול: שיפוע {_ci(r, 'cal_slope')}, חיתוך {_ci(r, 'cal_intercept')} (שיפוע 1 וחיתוך 0 = כיול מושלם).", "",
         "## 2025 מול 2026 (המודל הראשי)", "",
         "| מדד | 2025 (אומדן פנימי) | 2026 (תיקוף זמני) | שינוי | פרשנות |", "|---|---|---|---|---|",
         *[f"| {HE_METRIC.get(k, k)} | {_fmt(k, g(k, 'phase3_internal_estimate'), g(k, 'phase3_ci_low'), g(k, 'phase3_ci_high'))} | "
           f"{_fmt(k, g(k, 'temporal_2026'), g(k, 'temporal_2026_ci_low'), g(k, 'temporal_2026_ci_high'))} | {_fmt(k, g(k, 'absolute_change'), change=True).replace(' pp', ' נק׳ אחוז')} | "
           f"{HE_INTERP.get(str(g(k, 'interpretation')), '')} |" for k in ("capture_top10", "ppv_top10", "lift_top10", "auroc", "ap", "brier", "prevalence", "n", "events")
           if k in vp.index], "",
         "## איך לקרוא את ההשוואה ל-2025", "",
         "- המספר של 2025 הוא אומדן פיתוח פנימי; המספר של 2026 הוא הבדיקה הראשונה של המודל המוקפא על שנה מאוחרת יותר. ירידה מסוימת צפויה.",
         "- הבדל נחשב משמעותי רק כאשר רווחי הסמך (95%) של שתי השנים אינם חופפים.", "",
         "## הסתייגויות", "",
         "- תוצא חקרני: נפילה או שבר רשומים תוך 180 יום (לא התוצא של eFalls). זיהוי נפילה אינו מניעתה.",
         "- חלק מהמטופלים משותפים לשתי השנים; תת-הקבוצה של מטופלים חדשים מדווחת בנפרד בדוח המדעי.",
         "- משתנים חדשים של 2026 (סחרחורת, הליכה, ראייה, שמיעה ועוד) קוטלגו בלבד. כדי להשתמש בהם: להשלים אותם לשנת 2025, לפתח על 2025, ולתקף פעם אחת על תמונת מצב עתידית.", ""]
    return "\n".join(L)


def stopped_summaries(oc: dict[str, Any]) -> tuple[str, str]:
    en = "\n".join(["# Phase 4 – STOPPED before reporting model performance", "", f"**{WATERMARK}**", "",
                    "The 2026 outcome contract failed. No model performance was computed or reported.", "", *[f"- {h}" for h in oc["hard_failures"]], "",
                    "See OUTCOME_CONTRACT_2026.json and ask the DWH team to correct the 2026 outcome before any evaluation.", ""])
    he = "\n".join(["# שלב 4 – נעצר לפני דיווח ביצועים", "", f"**{WATERMARK_HE}**", "",
                    "בדיקת הגדרת התוצא של 2026 נכשלה, ולכן לא חושבו ולא דווחו ביצועי מודל.", "", *[f"- {h}" for h in oc["hard_failures"]], "",
                    "יש לפנות לצוות מחסן הנתונים לתיקון הגדרת התוצא לפני כל הערכה.", ""])
    return en, he


# ============================================================================ write reports / share
def write_reports(out: Path, *, res: dict[str, Any], oc: dict[str, Any], mm: dict[str, Any], est: dict[str, Any], overlap: dict[str, Any], man: dict[str, Any],
                  cfg: Any) -> None:
    ev = dirs(out)["evaluation"]
    D.write_csv(ev / "TEMPORAL_MODEL_COMPARISON.csv", res["comparison"])
    D.write_csv(ev / "TEMPORAL_CAPACITY.csv", res["capacity"])
    D.write_csv(ev / "TEMPORAL_CALIBRATION.csv", res["calibration"])
    D.write_csv(ev / "TEMPORAL_VS_PHASE3.csv", res["vs_phase3"])
    D.write_csv(ev / "PAIRED_VS_PRIMARY.csv", res["paired_vs_primary"])
    D.write_str(ev / "PHASE4_TEMPORAL_SUMMARY.md", summary_en(res, oc, mm, est, overlap, man, cfg))
    D.write_str(ev / "PHASE4_TEMPORAL_SUMMARY_HE.md", summary_he(res, oc, mm, est, overlap, cfg))
    figures(ev / "figures", res, mm, est)


def _identifiers(src: Path, out: Path) -> set[str]:
    from falls_ml.phase4.sealed import read_ids, read_header, sealed_map

    ids: set[str] = set()
    raw = read_ids(src, {}, extra=())
    for c in raw.columns:
        ids |= {str(v) for v in raw[c].dropna().astype(str) if len(str(v)) >= 6 and str(v).upper() != "NULL"}
    pf = dirs(out)["sealed"] / "BLIND_PREDICTIONS.parquet"
    if pf.is_file():
        ids |= set(pd.read_parquet(pf, columns=["key"])["key"].astype(str))
    return ids


def write_share(out: Path, *, L: dict[str, Any], cfg: Any, man: dict[str, Any], stopped: dict[str, Any] | None, src: Path) -> dict[str, Any]:
    from falls_ml.eda.runner import privacy_scan
    from falls_ml.phase2.stages_report import _path_hits

    d = dirs(out)
    tmp = out / ".tmp-share"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    pf, ev = d["preflight"], d["evaluation"]
    shutil.copyfile(pf / "PHASE4_PREFLIGHT.md", tmp / "PHASE4_PREFLIGHT.md")
    for n in ("SCHEMA_COMPARISON_2025_2026.csv", "PREDICTOR_SHIFT.csv", "NEW_2026_FEATURES_CATALOGUE.csv"):
        if (pf / n).is_file():
            shutil.copyfile(pf / n, tmp / n)
    D.write_json(tmp / "COHORT_OVERLAP.json", suppress_obj(read_json(d["sealed"] / "COHORT_OVERLAP.json")))
    D.write_json(tmp / "OUTCOME_CONTRACT_2026.json", suppress_obj(read_json(ev / "OUTCOME_CONTRACT_2026.json")))
    mm = read_json(d["frozen"] / MODEL_MANIFEST)
    D.write_json(tmp / "FROZEN_MODELS.json", {"primary": mm["primary"], "excluded_models": mm.get("excluded_models", {}),
                                              "models": {c: {k: v for k, v in m.items() if k not in ("artifact",)} for c, m in mm["models"].items()},
                                              "feature_list_sha256": mm["feature_list_sha256"], "phase3": mm["phase3"], "phase3_digest": mm["phase3_digest"]})
    if stopped is None:
        D.write_csv(tmp / "TEMPORAL_MODEL_COMPARISON.csv", suppress_comparison(pd.read_csv(ev / "TEMPORAL_MODEL_COMPARISON.csv")))
        D.write_csv(tmp / "TEMPORAL_CAPACITY.csv", suppress_capacity(pd.read_csv(ev / "TEMPORAL_CAPACITY.csv")))
        D.write_csv(tmp / "TEMPORAL_CALIBRATION.csv", suppress_calibration(pd.read_csv(ev / "TEMPORAL_CALIBRATION.csv")))
        for n in ("TEMPORAL_VS_PHASE3.csv", "PAIRED_VS_PRIMARY.csv", "PHASE4_TEMPORAL_SUMMARY.md", "PHASE4_TEMPORAL_SUMMARY_HE.md"):
            if (ev / n).is_file() and (ev / n).stat().st_size > 1:
                shutil.copyfile(ev / n, tmp / n)
        if (ev / "figures").exists():
            shutil.copytree(ev / "figures", tmp / "figures", ignore=shutil.ignore_patterns(".done"))
    else:
        en, he = stopped_summaries(stopped)
        D.write_str(tmp / "PHASE4_TEMPORAL_SUMMARY.md", en)
        D.write_str(tmp / "PHASE4_TEMPORAL_SUMMARY_HE.md", he)
    opened, _ = D.read_jsonl(ev / "OUTCOMES_OPENED.jsonl")
    D.write_json(tmp / "RUN_MANIFEST.json", {"watermark": WATERMARK, "phase4_version": PHASE4_VERSION, "falls_ml_version": __import__("falls_ml").__version__,
                                             "status": "STOPPED_OUTCOME_CONTRACT" if stopped else "EVALUATION_COMPLETE", "created_at": utc_now(),
                                             "input_2026": {k: man["input_2026"][k] for k in ("name", "sha256")}, "input_2025": {k: man["input_2025"][k] for k in ("name", "sha256")},
                                             "predictions_sha256": man["predictions_sha256"], "prediction_manifest_sha256": D.sha256_file(d["sealed"] / PRED_MANIFEST),
                                             "frozen_model_manifest_sha256": man["frozen_model_manifest_sha256"], "feature_list_sha256": man["feature_list_sha256"],
                                             "input_schema_sha256": man["input_schema_sha256"], "code_sha256_scoring": man["code_sha256"],
                                             "config_sha256": man["config_sha256"], "phase3_digest": man["phase3_digest"], "scored_at": man["created_at"],
                                             "outcomes_opened": [o.get("ts") for o in opened], "n_scored": man["n_scored"],
                                             "artifact_modes": {c: m["mode"] for c, m in mm["models"].items()}, "sealed_columns_never_read_before_evaluation": len(man["sealed_columns_never_read"])})
    ids = _identifiers(src, out)
    scan = privacy_scan(tmp, ids, "")
    needles = [str(out.resolve()), str(src.resolve()), str(src.resolve().parent), os.environ.get("USERNAME", ""), getpass.getuser()]
    needles = [n for n in needles if n and len(n) >= 3]
    path_hits, row_level = [], []
    for p in sorted(tmp.rglob("*")):
        if p.is_file() and p.suffix.lower() in (".csv", ".md", ".json", ".txt", ".svg"):
            h = _path_hits(p.read_text(encoding="utf-8", errors="ignore"), needles)
            if h:
                path_hits.append(f"{p.relative_to(tmp).as_posix()}: {sorted(set(h))}")
        if p.is_file() and p.suffix.lower() == ".csv":
            t = pd.read_csv(p)
            if set(t.columns) & ROW_LEVEL_COLUMNS or len(t) > 2000:
                row_level.append(p.relative_to(tmp).as_posix())
    forbidden = [p.relative_to(tmp).as_posix() for p in tmp.rglob("*") if p.is_file() and p.suffix.lower() in (".parquet", ".pkl", ".npz", ".sqlite", ".jsonl", ".joblib")]
    if not scan["passed"] or path_hits or forbidden or row_level:
        shutil.rmtree(tmp)
        raise Phase2Stop("PRIVACY_SCAN", "the share package failed the privacy / path / file-type / row-level scan; nothing was published to share/",
                         [*scan["hits"][:10], *path_hits[:10], *[f"forbidden file type: {f}" for f in forbidden], *[f"row-level table: {f}" for f in row_level]])
    res = {**scan, "path_scan": "passed", "file_type_scan": "passed", "row_level_scan": "passed", "hits": [], "min_cell": MIN_CELL,
           "rule": "aggregate only: no identifiers, row keys, row-level predictions, raw data, model objects or local paths; counts 1-9 suppressed"}
    D.write_json(tmp / "PRIVACY_SCAN.json", res)
    share = d["share"]
    if share.exists():
        D.rename_dir(share, out / f".old-share-{utc_now().replace(':', '').replace('-', '')[:15]}")
    D.rename_dir(tmp, share)
    return res
