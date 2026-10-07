"""The Phase 5 OPERATING DASHBOARD and the exact 3% capacity report, built from the COMPLETED run - analysis / reporting only.

    falls_ml meuhedet-phase5-dashboard --out <completed Phase 5 folder> [--input <the 2026 extract>]

reads the committed outer-OOF predictions (integrity-checked by their COMPLETE.json hashes, never changed), computes the outer-fold capacity
curves, the exact 3% operating point with a paired bootstrap, the "what drives the model" tables from the existing explanation / stability
outputs, writes the aggregate tables + the standalone HTML dashboard into share/ next to the existing results and re-runs the privacy scan.
No model is fitted, tuned or refitted; no CV is run; the existing Phase 5 results and their files stay byte-identical (the previous share is
kept in work/dashboard/ before share/ is republished).
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from falls_ml.artifacts import utc_now
from falls_ml.phase2 import durable as D
from falls_ml.phase2.state import Phase2Stop
from falls_ml.phase5 import DESIGN_LABEL, PHASE5_VERSION, SYNTHETIC_WATERMARK
from falls_ml.phase5.capacity import (CAPTURE_TARGETS, COMPARISONS, PRESETS, PRIMARY_SETS, TARGET, bootstrap_tables, capacity_bootstrap, curve,
                                      extended_grid, feature_drivers, fine_grid, fold_rows, grid_step, load_primary_oof, n_selected, rank_model,
                                      safe_points)
from falls_ml.phase5.config import FAMILIES, PRIMARY_FAMILY, SET_ALL, SET_OLD, SET_SAFE

DASHBOARD_FILE = "PHASE5_OPERATING_DASHBOARD.html"
NEW_SHARE_FILES = (DASHBOARD_FILE, "CAPACITY_CURVE_FINE.csv", "CAPACITY_CURVE_EXTENDED.csv", "CAPACITY_CURVE_POOLED_DESCRIPTIVE.csv",
                   "TOP3_CAPACITY_PRIMARY.csv", "TOP3_CAPACITY_BY_FOLD.csv", "TOP3_CAPACITY_COMPARISON.csv", "TOP3_CAPACITY_BOOTSTRAP.csv",
                   "TOP3_CAPACITY_SUMMARY_HE.md", "CAPTURE_TARGET_CAPACITY.csv", "FEATURE_DRIVERS.csv")
CAP_QUESTION = "אם ניתן להתערב רק אצל 3% מהמטופלים, כמה מהנפילות נתפוס?"
SECONDARY_HEADING = "## ניתוח משני: כ-70% רגישות (הניתוח הראשי המקורי של שלב 5 – ללא שינוי)"
HISTORICAL_HEADING_51 = "## ביקורת היסטורית בלבד: כלל ה-70% רגישות של שלב 5 2.2.0 – אינו תוצאת שלב 5.1 ואינו כלל הצלחה"
ORIGINAL_MARK = "<!-- phase5-original-management-summary -->"
SET_HE = {SET_OLD: "OLD (ישן)", SET_ALL: "OLD + כל החדשים הכשירים", SET_SAFE: "OLD + חדשים בטוחים"}
FAM_HE = {"LASSO": "LASSO", "ENET": "Elastic Net", "XGB": "XGBoost"}
METHOD_NOTES = [
    "חישוב ראשי (לפי קפלים חיצוניים): בכל קיבולת c נקבע מספר ההתערבויות הכולל T = עיגול(c × N); T מחולק בין הקפלים החיצוניים באופן יחסי לגודלם "
    "בשיטת 'השארית הגדולה ביותר' (דטרמיניסטית, סכום מדויק); בכל קפל נבחרים המטופלים עם הסיכון הגבוה ביותר לפי המודל של אותו קפל; הספירות "
    "מסוכמות. סולמות הסתברות של מודלים חיצוניים שונים אינם מושווים זה לזה.",
    "תצוגת OOF מאוחד: דירוג גלובלי של כל התחזיות יחד – לתכנון / תיאור בלבד, לא האומדן הראשי.",
    "בקיבולת קבועה כל המודלים בוחרים אותו מספר מטופלים, ולכן כל נפילה נוספת שנתפסת היא בדיוק התערבות מיותרת אחת פחות.",
    "ב-3% חושב רווח סמך מזווג (bootstrap של מטופלים, אותם משקלים לכל המודלים); בכל דגימה הקיבולת חולקה מחדש בין הקפלים והבחירה חושבה מחדש בתוך כל קפל. "
    "רווחי הסמך מותנים במודלים שאומנו.",
    "אין כאן אימון, כיוונון או CV חדשים: התחזיות הן תחזיות ה-OOF החיצוניות השמורות של הריצה שהושלמה.",
    "פרטיות: רק ספירות מצטברות לכל נקודת קיבולת; נקודה שבה תא כלשהו (TP / FP / FN / TN) הוא 1–9 אצל מודל כלשהו אינה מוצגת; אין מזהים, אין תחזיות "
    "ברמת מטופל ואין תאריכים.",
    "חשיבות פיצ'רים היא אסוציאציה עם תחזית המודל ואינה סיבתית.",
    "תיקוף צולב מקונן פנימי על תמונת מצב אחת (2026) – לא תיקוף חיצוני."]


def _round(df: pd.DataFrame, d: int = 5) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].round(d)
    return out


def _r(v: Any, d: int = 5) -> Any:
    return None if v is None or (isinstance(v, float) and not math.isfinite(v)) else round(float(v), d)


# ============================================================================ the computation (read-only)
def build_operating(ctx: Any, plan: dict[str, Any], *, n_boot: int, seed: int, schema: Any = None, log: Any = print) -> dict[str, Any]:
    from falls_ml.phase5.explain import importance_tables, stability_table

    t0 = time.time()
    y = ctx.y.astype(np.int64)
    outer = ctx.outer.astype(int)
    N, E = len(y), int(y.sum())
    oof = load_primary_oof(ctx, plan)
    if not any(fam == PRIMARY_FAMILY for fam, _ in oof):
        raise Phase2Stop("NO_PRIMARY_OOF", f"the {PRIMARY_FAMILY} PRIMARY units of this folder are not complete: nothing to build the dashboard from")
    ranked = {k: rank_model(k[0], k[1], y, outer, p) for k, p in oof.items()}
    grid = fine_grid(N)
    fine = curve(y, ranked, grid)
    safe, unsafe = safe_points(fine)
    ext = curve(y, ranked, extended_grid())
    ext_safe, _ = safe_points(ext)
    pooled = curve(y, ranked, grid, pooled=True)
    pooled_safe, _ = safe_points(pooled)
    T3 = n_selected(TARGET, N)
    log(f"capacity curves: {len(grid)} points x {len(ranked)} models (outer-fold primary + pooled descriptive), {len(safe)} shareable")
    byfold = pd.DataFrame([r for rk in ranked.values() for r in fold_rows(y, rk, T3)])
    log(f"paired bootstrap at {TARGET / 10:.1f}% ({n_boot} replicates, the selection re-computed within every outer fold in every replicate)")
    boot = capacity_bootstrap(y, outer, ranked, TARGET, n_boot=n_boot, seed=seed,
                              progress=lambda b, n: log(f"  bootstrap {b}/{n}") if b % 1000 == 0 else None)
    cmp3, dist3 = bootstrap_tables(y, ranked, boot)
    wins = []
    for r in cmp3.itertuples():
        a, b = r.comparison.split(" minus ")[1], r.comparison.split(" minus ")[0]
        fa = byfold[(byfold.family == r.family) & (byfold.feature_set == a)].set_index("outer_fold")["tp"]
        fb = byfold[(byfold.family == r.family) & (byfold.feature_set == b)].set_index("outer_fold")["tp"]
        wins.append((int((fb > fa).sum()), int((fb < fa).sum()), int(len(fa))))
    if len(cmp3):
        cmp3["folds_new_more_falls"] = [w[0] for w in wins]
        cmp3["folds_new_fewer_falls"] = [w[1] for w in wins]
        cmp3["n_folds"] = [w[2] for w in wins]
    # Phase 5.1 (3.x plans): "what drives the model" describes the ADMISSIBLE comparator (OLD_PLUS_NEW_SAFE); a 2.x plan keeps its ALL set
    s_all = plan["alias"][plan.get("comparator_set", SET_ALL)] if str(plan.get("phase5_major", "2")) == "3" else plan["alias"][SET_ALL]
    fams = [f for f in FAMILIES if f in plan.get("families", list(FAMILIES))]
    perm, shap = importance_tables(ctx, fams, [s_all])
    stab = stability_table(ctx, fams, s_all, perm)
    reg = pd.read_csv(ctx.out / "work" / "REGISTRY.csv")
    drivers = feature_drivers(perm, shap, stab, reg, s_all, schema)
    capture = _capture_rows(fine[fine.capacity_permille.isin(safe)], ext[ext.capacity_permille.isin(ext_safe)], ranked)
    return {"N": N, "E": E, "grid": grid, "grid_step": grid_step(N), "fine": fine, "safe": safe, "unsafe": unsafe, "ext": ext, "ext_safe": ext_safe,
            "pooled": pooled, "pooled_safe": pooled_safe, "T3": T3, "byfold": byfold, "cmp3": cmp3, "dist3": dist3, "boot_n": n_boot, "seed": seed,
            "drivers": drivers, "capture": capture, "models": sorted(ranked), "schema": schema, "seconds": round(time.time() - t0, 1),
            "three_pct_safe": TARGET in safe, "safe_same": plan["alias"].get(SET_SAFE) == plan["alias"].get(SET_ALL),
            "no_new": plan["alias"].get(SET_ALL) == plan["alias"].get(SET_OLD), "folds": int(outer.max()) + 1}


def _capture_rows(fine: pd.DataFrame, ext: pd.DataFrame, ranked: dict[tuple[str, str], Any]) -> pd.DataFrame:
    both = pd.concat([fine, ext], ignore_index=True).sort_values("capacity_permille")
    rows = []
    for fam, s in ranked:
        g = both[(both.family == fam) & (both.feature_set == s)]
        for t in CAPTURE_TARGETS:
            h = g[g.sensitivity >= t - 1e-12]
            if not len(h):
                rows.append({"family": fam, "feature_set": s, "target_fall_capture": t, "required_capacity_pct": None, "note": "not reached on the computed grid"})
                continue
            r = h.iloc[0]
            rows.append({"family": fam, "feature_set": s, "target_fall_capture": t, "required_capacity_pct": r.capacity_pct, "selected_total": r.selected_total,
                         "tp": r.tp, "fp": r.fp, "sensitivity": r.sensitivity, "ppv": r.ppv,
                         "note": "smallest grid capacity (0.1% steps to 20%, 0.5% beyond) whose outer-fold capacity selection reaches the target; secondary"})
    return pd.DataFrame(rows)


# ============================================================================ the share outputs
CELL_COLS = ("tp", "fp", "fn", "tn", "a_tp", "b_tp", "a_fp", "b_fp", "falls_total")
KEEP_COLS = ("family", "feature_set", "comparison", "role", "capacity_permille", "capacity_pct", "population", "selected_total", "actual_intervention_pct",
             "outer_fold", "fold_patients", "selected", "method", "ties_split_at_boundary", "note", "target_fall_capture", "n_folds")


def _suppress(df: pd.DataFrame) -> pd.DataFrame:
    """A row with any patient cell of 1-9 (TP / FP / FN / TN, a model's TP / FP, a fold's falls) shows '<10' for those cells and 'suppressed' for
    every other count, rate and difference of that row (any of them would give the small cell back); a cell is never shown in isolation."""
    from falls_ml.phase5.report import _small_cell

    out = _round(df).astype(object)
    cells = [c for c in CELL_COLS if c in out.columns]
    if not cells or not len(out):
        return out
    bad = pd.Series(False, index=out.index)
    for c in cells:
        bad |= out[c].map(_small_cell).astype(bool)
    for i in out.index[bad]:
        for c in out.columns:
            if c in KEEP_COLS:
                continue
            out.at[i, c] = "<10" if (c in cells and _small_cell(out.at[i, c])) else "suppressed"
    return out


def write_operating(tmp: Path, A: dict[str, Any], plan: dict[str, Any], *, synthetic: bool) -> list[str]:
    fine, ext, pooled = A["fine"], A["ext"], A["pooled"]
    keep = lambda t, pts: t[t.capacity_permille.isin(pts)].drop(columns=["cells_safe"])  # noqa: E731
    D.write_csv(tmp / "CAPACITY_CURVE_FINE.csv", _suppress(keep(fine, A["safe"])))
    D.write_csv(tmp / "CAPACITY_CURVE_EXTENDED.csv", _suppress(keep(ext, A["ext_safe"])))
    D.write_csv(tmp / "CAPACITY_CURVE_POOLED_DESCRIPTIVE.csv", _suppress(keep(pooled, A["pooled_safe"])))
    p3 = fine[fine.capacity_permille == TARGET].drop(columns=["cells_safe"])
    D.write_csv(tmp / "TOP3_CAPACITY_PRIMARY.csv", _suppress(p3))
    D.write_csv(tmp / "TOP3_CAPACITY_BY_FOLD.csv", _suppress(A["byfold"]))
    cmp3 = A["cmp3"].rename(columns={"tp_a": "a_tp", "tp_b": "b_tp", "fp_a": "a_fp", "fp_b": "b_fp"})
    D.write_csv(tmp / "TOP3_CAPACITY_COMPARISON.csv", _suppress(cmp3))
    D.write_csv(tmp / "TOP3_CAPACITY_BOOTSTRAP.csv", _round(A["dist3"]))
    D.write_csv(tmp / "CAPTURE_TARGET_CAPACITY.csv", _suppress(A["capture"]))
    D.write_csv(tmp / "FEATURE_DRIVERS.csv", _round(A["drivers"]) if len(A["drivers"]) else pd.DataFrame(columns=["family", "feature"]))
    D.write_str(tmp / "TOP3_CAPACITY_SUMMARY_HE.md", top3_summary_he(A, plan, synthetic))
    from falls_ml.phase5.dashboard_html import render

    D.write_str(tmp / DASHBOARD_FILE, render(dashboard_data(A, plan, synthetic)))
    return list(NEW_SHARE_FILES)


def dashboard_data(A: dict[str, Any], plan: dict[str, Any], synthetic: bool) -> dict[str, Any]:
    dom_he = {k: str((v or {}).get("label_he", k)) for k, v in (getattr(A.get("schema"), "domains", None) or {}).items()}
    fams = [f for f in FAMILIES if any(k[0] == f for k in A["models"])]
    sets = [s for s in PRIMARY_SETS]

    def tps(t: pd.DataFrame, pts: list[int]) -> dict[str, dict[str, list[int]]]:
        out: dict[str, dict[str, list[int]]] = {}
        for fam in fams:
            for s in sets:
                g = t[(t.family == fam) & (t.feature_set == s)].set_index("capacity_permille")
                if len(g):
                    out.setdefault(fam, {})[s] = [int(g.at[pm, "tp"]) for pm in pts]
        return out

    top3: dict[str, dict[str, Any]] = {}
    if A["three_pct_safe"]:
        for r in A["cmp3"].itertuples():
            b = r.comparison.split(" minus ")[0]
            top3.setdefault(r.family, {})[b] = {"d": int(r.delta_falls_captured), "lo": _r(r.delta_falls_captured_ci_low, 1),
                                                 "hi": _r(r.delta_falls_captured_ci_high, 1), "folds_better": int(r.folds_new_more_falls)}
    drv: dict[str, Any] = {}
    t = A["drivers"]
    for fam in fams:
        g = t[t.family == fam] if len(t) else pd.DataFrame()
        if not len(g):
            continue

        def row(r: Any) -> dict[str, Any]:
            if fam in ("LASSO", "ENET") and pd.notna(getattr(r, "selection_frequency", np.nan)):
                st = ("לא נבחר באף דגימה (מקדם 0)" if r.selection_frequency == 0 else
                      f"נבחר ב-{100 * r.selection_frequency:.0f}% מהדגימות" + (f"; עקביות סימן {100 * r.sign_consistency:.0f}%" if pd.notna(r.sign_consistency) else ""))
            elif fam == "XGB" and pd.notna(getattr(r, "top10_frequency", np.nan)):
                st = f"בעשירייה הראשונה ב-{100 * r.top10_frequency:.0f}% מהדגימות"
            elif pd.notna(getattr(r, "permutation_folds_in_top10", np.nan)):
                st = f"בעשירייה הראשונה ב-{int(r.permutation_folds_in_top10)} קפלים"
            else:
                st = ""
            return {"rank": int(r.rank_overall), "feature": str(r.feature), "label": str(r.label_he), "importance": _r(r.importance, 5),
                    "stability": st, "stable": bool(r.stable) if pd.notna(r.stable) else False, "direction": str(r.direction or ""),
                    "domain": dom_he.get(str(r.domain or ""), str(r.domain or "")), "status": str(r.new_status or "")}
        drv[fam] = {"measure": str(g["importance_measure"].iloc[0]),
                    "old": [row(r) for r in g[g.origin == "OLD"].head(10).itertuples()],
                    "new": [row(r) for r in g[g.origin == "NEW"].itertuples()]}
    start = min(A["safe"]) if A["safe"] else None
    default = TARGET if TARGET in A["safe"] else (min(A["safe"], key=lambda k: abs(k - TARGET)) if A["safe"] else TARGET)
    notes = list(METHOD_NOTES)
    if A["unsafe"]:
        notes.append(f"נקודות קיבולת {', '.join(f'{k / 10:.1f}%' for k in A['unsafe'][:12])}{' ...' if len(A['unsafe']) > 12 else ''} אינן מוצגות: "
                     f"תא אחד לפחות קטן מ-10 מטופלים. העקומה המשותפת מתחילה ב-{start / 10:.1f}%." if start else "אין נקודת קיבולת בטוחה לשיתוף.")
    if A["grid_step"] > 1:
        notes.append(f"צעד הקיבולת הוגדל ל-{A['grid_step'] / 10:.1f}% כי באוכלוסייה של {A['N']:,} מטופלים צעד של 0.1% הוא פחות מ-10 מטופלים.")
    return {"n": A["N"], "events": A["E"], "sets": sets, "families": fams, "primary_family": PRIMARY_FAMILY,
            "grid": A["safe"], "grid_step": A["grid_step"], "ext_grid": A["ext_safe"], "pooled_grid": A["pooled_safe"],
            "tp": tps(A["fine"], A["safe"]), "tp_ext": tps(A["ext"], A["ext_safe"]), "tp_pooled": tps(A["pooled"], A["pooled_safe"]),
            "presets": list(PRESETS), "target_permille": TARGET, "default_permille": default, "capture_targets": list(CAPTURE_TARGETS),
            "top3": top3, "n_boot": A["boot_n"], "folds": A["folds"], "drivers": drv, "synthetic": bool(synthetic),
            "synthetic_text": f"{SYNTHETIC_WATERMARK} – נתונים סינתטיים (בדיקת תוכנה); המספרים אינם תוצאה מדעית.",
            "subtitle": (f"שלב 5 – {DESIGN_LABEL}. {A['N']:,} מטופלים, {A['E']:,} נפילות ב-180 יום אחרי 01/01/2026. "
                         f"מודל ראשי: Elastic Net. תחזיות OOF חיצוניות של הריצה שהושלמה – ללא אימון מחדש."),
            "method_notes": notes}


# ============================================================================ the Hebrew texts
def _pct(v: Any, d: int = 1) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    return "—" if not math.isfinite(x) else f"{100 * x:.{d}f}%"


def _num(v: Any) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    return "—" if not math.isfinite(x) else f"{x:,.0f}"


def _sg(v: Any, f: Any = _num) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    return "—" if not math.isfinite(x) else (("+" if x > 0 else ("−" if x < 0 else "±")) + f(abs(x)))


def _pp(v: Any) -> str:
    return _sg(v, lambda x: f"{100 * x:.2f} נ\"א")


def _cells(A: dict[str, Any], fam: str, s: str) -> dict[str, Any] | None:
    f = A["fine"]
    g = f[(f.family == fam) & (f.feature_set == s) & (f.capacity_permille == TARGET)]
    return _suppress(g).iloc[0].to_dict() if len(g) else None


def capacity_section_he(A: dict[str, Any], plan: dict[str, Any]) -> list[str]:
    fam = PRIMARY_FAMILY
    N, E, T3 = A["N"], A["E"], A["T3"]
    L = [f"## עמוד ראשון: {CAP_QUESTION}", "",
         f"- **אוכלוסייה:** {_num(N)} מטופלים; **נפילות** ב-180 הימים שאחרי יום המדד: {_num(E)} ({_pct(E / N if N else float('nan'))}).",
         f"- **3% מהאוכלוסייה = {_num(T3)} מטופלים** שנבחרים להתערבות (הסיכון הגבוה ביותר לפי המודל, בכל קפל חיצוני בנפרד).", "",
         f"### {FAM_HE[fam]} (המודל הראשי) ב-3%", "",
         "| מערך פיצ'רים | נפילות שנתפסו | שיעור תפיסה (Recall) | PPV | התערבויות מיותרות |", "|---|---|---|---|---|"]
    for s in PRIMARY_SETS:
        c = _cells(A, fam, s)
        if c is None:
            continue
        L.append(f"| {SET_HE[s]} | {_num(c['tp'])} מתוך {_num(E)} | {_pct(c['sensitivity'])} | {_pct(c['ppv'])} | {_num(c['fp'])} |")
    L.append("")
    cm = A["cmp3"]
    for _, r in cm[cm.family == fam].iterrows():
        b = r["comparison"].split(" minus ")[0]
        lab = "כל החדשים הכשירים" if b == SET_ALL else "חדשים בטוחים בלבד (בדיקה מחמירה)"
        L.append(f"- **{lab} מול OLD:** {_sg(r['delta_falls_captured'])} נפילות שנתפסו (רווח סמך 95%: {_sg(r['delta_falls_captured_ci_low'])} עד "
                 f"{_sg(r['delta_falls_captured_ci_high'])}) = {_sg(-r['delta_falls_captured'])} התערבויות מיותרות; שינוי ב-Recall {_pp(r['delta_sensitivity'])}, "
                 f"שינוי ב-PPV {_pp(r['delta_ppv'])}; יותר נפילות ב-{int(r['folds_new_more_falls'])} מתוך {int(r['n_folds'])} הקפלים החיצוניים.")
    L += ["", "בקיבולת קבועה מספר ההתערבויות זהה בכל המודלים – כל נפילה נוספת שנתפסת היא בדיוק התערבות מיותרת אחת פחות.", ""]
    sec = []
    for f2 in FAMILIES:
        if f2 == fam:
            continue
        r = cm[(cm.family == f2) & (cm.comparison == f"{SET_ALL} minus {SET_OLD}")]
        if len(r):
            r = r.iloc[0]
            sec.append(f"{FAM_HE[f2]}: {_sg(r['delta_falls_captured'])} נפילות (רווח סמך {_sg(r['delta_falls_captured_ci_low'])} עד "
                       f"{_sg(r['delta_falls_captured_ci_high'])})")
    if sec:
        L += [f"- משפחות משניות (כל החדשים מול OLD ב-3%, לא קובעות): {'; '.join(sec)}.", ""]
    if not A["three_pct_safe"]:
        L += ["- חלק מהתאים ב-3% קטנים מ-10 מטופלים ולכן הוסתרו ('<10').", ""]
    L += [f"**לוח מחוונים אינטראקטיבי:** `{DASHBOARD_FILE}` – לפתוח בכרום / אדג' (ללא אינטרנט); ניתן לבחור כל קיבולת בין 0.5% ל-20%, "
          "משפחת מודל ומערך פיצ'רים. פירוט מלא: `TOP3_CAPACITY_SUMMARY_HE.md`, `TOP3_CAPACITY_COMPARISON.csv`, `CAPACITY_CURVE_FINE.csv`.",
          "", "שיטה: חישוב קיבולת לפי קפלים חיצוניים (לא סף הסתברות מאוחד); תחזיות ה-OOF החיצוניות של הריצה שהושלמה – אין אימון, כיוונון או CV חדשים.", ""]
    return L


def compose_management(original: str, A: dict[str, Any], plan: dict[str, Any]) -> str:
    """The 3% capacity question FIRST; the original ~70% sensitivity summary follows unchanged as the SECONDARY analysis."""
    text = original.replace("\r\n", "\n")
    head, sep, body = text.partition("\n## עמוד ראשון")
    if not sep:
        head, body = "", text
    else:
        body = body.split("\n", 1)[1] if "\n" in body else ""
    heading = HISTORICAL_HEADING_51 if str(plan.get("phase5_major", "2")) == "3" else SECONDARY_HEADING
    return "\n".join([head.rstrip("\n"), "", *capacity_section_he(A, plan), heading, ORIGINAL_MARK, body.lstrip("\n")]).rstrip("\n") + "\n"


def original_management(composed: str) -> str:
    head, sep, _ = composed.partition(f"\n## עמוד ראשון: {CAP_QUESTION}")
    _, mark, body = composed.partition(ORIGINAL_MARK + "\n")
    if not sep or not mark:
        return composed
    return head.rstrip("\n") + "\n\n## עמוד ראשון\n" + body


def top3_summary_he(A: dict[str, Any], plan: dict[str, Any], synthetic: bool) -> str:
    N, E, T3 = A["N"], A["E"], A["T3"]
    L = ["# קיבולת התערבות של 3% – דו\"ח מדויק (שלב 5)", ""]
    if synthetic:
        L += [f"> **{SYNTHETIC_WATERMARK}** – נתונים סינתטיים (בדיקת תוכנה); המספרים אינם תוצאה מדעית.", ""]
    L += [f"**השאלה:** {CAP_QUESTION}", "",
          f"- אוכלוסייה {_num(N)}, נפילות {_num(E)}; 3% = {_num(T3)} מטופלים (חלוקה בין {A['folds']} הקפלים החיצוניים לפי גודלם, 'השארית הגדולה ביותר').",
          f"- {DESIGN_LABEL}; תחזיות OOF חיצוניות שמורות – אין אימון מחדש.", "",
          "## כל המשפחות והמערכים ב-3% (חישוב לפי קפלים – ראשי)", "",
          "| משפחה | מערך | נבחרו | נפילות שנתפסו | Recall | PPV | התערבויות מיותרות | נפילות שהוחמצו | Lift | NNI |", "|---|---|---|---|---|---|---|---|---|---|"]
    f = _suppress(A["fine"][A["fine"].capacity_permille == TARGET])
    for r in f.itertuples():
        L.append(f"| {FAM_HE[r.family]}{' (ראשי)' if r.family == PRIMARY_FAMILY else ''} | {SET_HE[r.feature_set]} | {_num(r.selected_total)} | {_num(r.tp)} | "
                 f"{_pct(r.sensitivity)} | {_pct(r.ppv)} | {_num(r.fp)} | {_num(r.fn)} | {r.lift if isinstance(r.lift, str) else f'{r.lift:.2f}'} | "
                 f"{r.nni_per_captured_fall if isinstance(r.nni_per_captured_fall, str) else f'{r.nni_per_captured_fall:.1f}'} |")
    L += ["", f"## ההבדל, bootstrap מזווג ({A['boot_n']:,} דגימות; הבחירה חושבה מחדש בכל קפל בכל דגימה)", "",
          "| משפחה | השוואה | Δ נפילות שנתפסו [95%] | Δ Recall [95%] | Δ PPV [95%] | Δ התערבויות מיותרות [95%] | קפלים עם יותר נפילות |",
          "|---|---|---|---|---|---|---|"]
    for r in A["cmp3"].itertuples():
        L.append(f"| {FAM_HE[r.family]} | {r.comparison} | {_sg(r.delta_falls_captured)} [{_sg(r.delta_falls_captured_ci_low)}, {_sg(r.delta_falls_captured_ci_high)}] | "
                 f"{_pp(r.delta_sensitivity)} [{_pp(r.delta_sensitivity_ci_low)}, {_pp(r.delta_sensitivity_ci_high)}] | {_pp(r.delta_ppv)} "
                 f"[{_pp(r.delta_ppv_ci_low)}, {_pp(r.delta_ppv_ci_high)}] | {_sg(r.delta_false_interventions)} [{_sg(r.delta_false_interventions_ci_low)}, "
                 f"{_sg(r.delta_false_interventions_ci_high)}] | {r.folds_new_more_falls}/{r.n_folds} |")
    L += ["", "## פרשנות", "",
          "- בקיבולת קבועה Δ התערבויות מיותרות = −Δ נפילות שנתפסו, ו-Δ PPV = Δ נפילות / מספר הנבחרים.",
          "- רווח סמך שכולו מעל 0 בנפילות שנתפסו = עדות שהמידע החדש תופס יותר נפילות באותה קיבולת; רווח שחוצה את 0 = לא הוכח הבדל.",
          "- רווחי הסמך מותנים במודלים שאומנו (bootstrap של תחזיות ה-OOF), כמו בניתוח ה-70%.",
          "- ניתוח ה-70% רגישות המקורי (PRIMARY_70_SENSITIVITY_COMPARISON.csv) נשאר ללא שינוי כניתוח משני.",
          "", "## שיטה ופרטיות", "", *[f"- {t}" for t in METHOD_NOTES], "",
          "קבצים: TOP3_CAPACITY_PRIMARY.csv, TOP3_CAPACITY_BY_FOLD.csv, TOP3_CAPACITY_COMPARISON.csv, TOP3_CAPACITY_BOOTSTRAP.csv, CAPACITY_CURVE_FINE.csv, "
          f"{DASHBOARD_FILE}."]
    return "\n".join(L) + "\n"


# ============================================================================ the standalone command
def units_digest(out: Path) -> str:
    """sha256 over every committed unit / explanation file record (COMPLETE.json): proves the run's results were not touched."""
    h = hashlib.sha256()
    for p in sorted([*(out / "work" / "units").glob("*/COMPLETE.json"), *(out / "work" / "explain").glob("*/COMPLETE.json")]):
        h.update(p.parent.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def _find_input(out: Path, plan: dict[str, Any], input_path: str | Path | None) -> Path:
    from falls_ml.data.dataset import sha256_file

    want = plan["input"]
    cands = [Path(input_path)] if input_path else [out.parent / want["name"], out / want["name"]]
    for c in cands:
        if c.is_file() and c.stat().st_size == int(want["bytes"]) and sha256_file(c) == want["sha256"]:
            return c
    if input_path:
        raise Phase2Stop("INPUT_MISMATCH", "the --input file is not the 2026 extract of this Phase 5 run (sha256 differs)",
                         ["the identifier scan of share/ needs the run's own input file"])
    raise Phase2Stop("INPUT_NEEDED", f"the run's input file {want['name']} was not found next to the output folder",
                     ["add --input \"<path to the 2026 extract of this run>\" (it is read only for the identifier scan of share/)"])


def run_dashboard(out_dir: str | Path, *, input_path: str | Path | None = None, n_boot: int = 2000, log: Any = print) -> dict[str, Any]:
    from falls_ml.phase5.engine import Ctx
    from falls_ml.phase5.models import DeviceState
    from falls_ml.phase5.report import publish

    out = Path(out_dir)
    w = out / "work"
    if not (w / "PLAN.json").is_file():
        raise Phase2Stop("NO_RUN", f"{out.name} holds no Phase 5 run (work/PLAN.json missing)")
    plan = json.loads((w / "PLAN.json").read_text(encoding="utf-8"))
    if str(plan.get("phase5_version", "")).split(".")[0] not in ("2", "3"):
        raise Phase2Stop("PLAN_VERSION", f"this folder was made by Phase 5 {plan.get('phase5_version')}; the dashboard needs a Phase 5 2.x or 3.x run")
    probs = [f"{name} differs from the plan" for name, key in (("ANALYSIS_FRAME.parquet", "frame_sha256"), ("FOLDS.parquet", "folds_sha256"))
             if D.sha256_file(w / name) != plan[key]]
    if probs:
        raise Phase2Stop("WORK_INTEGRITY", "the work folder does not match its plan", probs)
    share = out / "share"
    if not (share / "RUN_MANIFEST.json").is_file() or not (share / "MANAGEMENT_SUMMARY_HE.md").is_file():
        raise Phase2Stop("NO_SHARE", "the completed run's share/ folder (RUN_MANIFEST.json, MANAGEMENT_SUMMARY_HE.md) is missing: finish the run first")
    src = _find_input(out, plan, input_path)
    before = units_digest(out)
    frame = pd.read_parquet(w / "ANALYSIS_FRAME.parquet")
    z = np.load(w / "Y_FOLDS.npz")
    canon = {plan["alias"][k]: v for k, v in plan["sets"].items() if plan["alias"][k] == k}
    ctx = Ctx(out=out, frame=frame, y=z["y"].astype(int), meta=json.loads((w / "META.json").read_text(encoding="utf-8")),
              sets={k: canon[plan["alias"][k]] for k in plan["sets"]}, outer=z["outer"].astype(int), cfg=None, seed=int(plan["seed"]), jobs=1,
              state=DeviceState("cpu"), log=log)
    schema = None
    try:
        from falls_ml.phase5.schema import load_v21_schema

        schema = load_v21_schema("configs/meuhedet/phase5_v21_schema.yaml")
    except Exception:  # noqa: BLE001 - Hebrew labels only
        schema = None
    log(f"Phase 5 operating dashboard: {plan['n']:,} patients, {plan['events']:,} events; reading the committed outer-OOF predictions (no fitting)")
    A = build_operating(ctx, plan, n_boot=int(n_boot), seed=int(plan["seed"]) + 41, schema=schema, log=log)
    dd = w / "dashboard"
    dd.mkdir(parents=True, exist_ok=True)
    orig_path = dd / "MANAGEMENT_SUMMARY_HE_ORIGINAL.md"
    current = (share / "MANAGEMENT_SUMMARY_HE.md").read_text(encoding="utf-8")
    if not orig_path.is_file():
        D.write_str(orig_path, original_management(current))
    original = orig_path.read_text(encoding="utf-8")
    tmp = out / ".tmp-share"
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(share, tmp, ignore=shutil.ignore_patterns("PRIVACY_SCAN.json"))
    files = write_operating(tmp, A, plan, synthetic=bool(plan.get("synthetic")))
    D.write_str(tmp / "MANAGEMENT_SUMMARY_HE.md", compose_management(original, A, plan))
    man = json.loads((tmp / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    man["operating_capacity_dashboard"] = manifest_block(A, plan, files, before)
    D.write_json(tmp / "RUN_MANIFEST.json", man)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup = dd / f"share_before_dashboard_{stamp}"
    k = 1
    while backup.exists():
        k += 1
        backup = dd / f"share_before_dashboard_{stamp}_{k}"
    shutil.copytree(share, backup)
    res = publish(out, tmp, src, frame)
    after = units_digest(out)
    if after != before:
        raise Phase2Stop("RESULTS_CHANGED", "the committed Phase 5 units changed while the dashboard was built (they must be read-only)")
    D.append_jsonl(out / "logs" / "dashboard_runs.jsonl", {"ts": utc_now(), "falls_ml_version": __import__("falls_ml").__version__, "n_boot": int(n_boot),
                                                            "seconds": A["seconds"], "units_digest": after, "files": files})
    return {"status": "DASHBOARD_COMPLETE", "exit_code": 0, "files": files, "privacy_passed": bool(res["passed"]), "seconds": A["seconds"],
            "three_pct_safe": A["three_pct_safe"], "units_unchanged": True, "share": str(share)}


def manifest_block(A: dict[str, Any], plan: dict[str, Any], files: list[str], digest: str) -> dict[str, Any]:
    return {"created_at": utc_now(), "falls_ml_version": __import__("falls_ml").__version__, "phase5_version": PHASE5_VERSION,
            "method_primary": "OUTER_FOLD_CAPACITY: T = round(c x N) allocated across outer folds by largest remainder, top-k within each fold, counts summed",
            "method_descriptive": "POOLED_OOF (global ranking of the concatenated OOF risks) - planning / descriptive only",
            "grid": {"min_pct": GRID_TXT[0], "max_pct": GRID_TXT[1], "step_pct": A["grid_step"] / 10, "exact_3pct_selected": A["T3"],
                     "shareable_start_pct": (min(A["safe"]) / 10) if A["safe"] else None, "suppressed_points_pct": [k / 10 for k in A["unsafe"]]},
            "bootstrap": {"n": A["boot_n"], "seed": A["seed"], "paired": True, "selection_recomputed_per_replicate": True},
            "no_model_fitted": True, "source": "committed outer-OOF predictions of the PRIMARY units (integrity-checked)",
            "units_digest_before_and_after": digest, "files": files, "seconds": A["seconds"]}


GRID_TXT = (0.5, 20.0)


def add_to_report(tmp: Path, ctx: Any, plan: dict[str, Any], *, n_boot: int, synthetic: bool, log: Any = print) -> list[str]:
    """Used by the full report (build_reports): the same outputs inside a fresh share, the 3% question first in the management summary."""
    schema = getattr(getattr(ctx, "cfg", None), "schema", None)
    A = build_operating(ctx, plan, n_boot=n_boot, seed=int(plan["seed"]) + 41, schema=schema, log=log)
    files = write_operating(tmp, A, plan, synthetic=synthetic)
    mp = tmp / "MANAGEMENT_SUMMARY_HE.md"
    D.write_str(mp, compose_management(mp.read_text(encoding="utf-8"), A, plan))
    man = json.loads((tmp / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    man["operating_capacity_dashboard"] = manifest_block(A, plan, files, units_digest(ctx.out))
    D.write_json(tmp / "RUN_MANIFEST.json", man)
    return files
