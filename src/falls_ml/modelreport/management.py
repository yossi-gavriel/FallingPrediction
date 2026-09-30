"""Hebrew management reporting: MANAGEMENT_MODEL_REPORT_HE.html, MANAGEMENT_MODEL_SUMMARY_HE.md, EXECUTIVE_ONE_PAGER_HE.html.

Every sentence with a number is generated from the run artifacts. Conclusions are stated only when the artifacts support them (for
example "discriminates" only when the lower 95% bound of AUROC exceeds 0.5); wording is descriptive ("קשור לתחזית המודל", never
"גורם לנפילה") and never uses hype ("excellent", "production ready", "clinically proven", "breakthrough" or their Hebrew equivalents).
The primary model is the pre-declared EXTENDED analysis, never the one with the best test result.
"""

from __future__ import annotations

import base64
import html
from typing import Any

import numpy as np
import pandas as pd

from falls_ml.modelreport import figures as F

BANNED = ("excellent", "production ready", "production-ready", "clinically proven", "breakthrough", "מצוין", "מעולה", "פריצת דרך", "מוכח קלינית", "מוכן לייצור",
          "גורם לנפילה", "גורמים לנפילה")
WATERMARK_HE = "תוצאה חקרנית: תוצא נפילה תוך 180 יום – לא שחזור של מודל eFalls"
SYNTHETIC_HE = "נתונים סינתטיים – בדיקת תוכנה בלבד, לא תוצאות מדעיות"
ROADMAP_HE = ["מודל חקרני נוכחי", "ניתוח נתונים מלא (EDA)", "ניתוחי רגישות בטוחי-תזמון", "תיקון התזמון במחסן הנתונים",
              "משתנים נוספים ממאוחדת", "תיקוף זמני (כמה נקודות זמן)", "תיקוף חיצוני / פרוספקטיבי", "רק אז: שקילת שימוש קליני"]


def _esc(v: Any) -> str:
    return html.escape("" if v is None else str(v))


def _pct(x: Any, d: int = 1) -> str:
    try:
        return f"{100.0 * float(x):.{d}f}%"
    except (TypeError, ValueError):
        return "לא זמין"


def _est(d: dict[str, Any] | None) -> tuple[float | None, float | None, float | None]:
    d = d or {}
    return d.get("estimate"), d.get("ci_low"), d.get("ci_high")


def _run(rs: Any, key: str) -> Any:
    return next((r for r in rs.runs if r.run.key == key), None)


def _label_he(rs: Any, key: str) -> str:
    return ((rs.labels.get("analyses") or {}).get(key) or {}).get("he", key)


def _fname(rs: Any, f: str) -> str:
    from falls_ml.modelreport.artifacts import feature_label

    return feature_label(rs.labels, f, "he")


def _stable_features(rr: Any, min_freq: float = 0.8) -> pd.DataFrame:
    fd = rr.tables.get("feature_dictionary")
    if fd is None or not len(fd):
        return pd.DataFrame()
    d = fd[fd["selected_at_lambda_star"].astype(bool)].copy()
    d["freq"] = pd.to_numeric(d["bootstrap_selection_pct"], errors="coerce")
    d = d.dropna(subset=["freq"]).sort_values("freq", ascending=False)
    return d[d["freq"] >= 100 * min_freq]


def _permutation(rr: Any) -> pd.DataFrame:
    """Permutation importance of the primary run (computed by the run on VALIDATION: mean drop in AUROC when one predictor's values are
    shuffled, SD over repeats). Read from the run's own artifact; nothing is refitted."""
    imp = rr.run.csv("feature_importance.csv")
    if not len(imp) or "permutation_importance_mean" not in imp.columns:
        return pd.DataFrame()
    d = imp.dropna(subset=["permutation_importance_mean"]).copy()
    d["mean"] = d["permutation_importance_mean"].astype(float)
    d["sd"] = pd.to_numeric(d.get("permutation_importance_std"), errors="coerce").fillna(0.0) if "permutation_importance_std" in d else 0.0
    return d.sort_values("mean", ascending=False)[["feature", "mean", "sd"]].reset_index(drop=True)


IMPORTANCE_NOTE_HE = ["משתנה יכול להיות יציב מאוד (נבחר כמעט תמיד) ועדיין להוסיף מעט מידע ייחודי לחיזוי.",
                      "משתנים שקשורים זה לזה חולקים מידע: כשמערבבים אחד מהם, האחר \"מפצה\" חלקית, ולכן התרומה של כל אחד נראית קטנה יותר.",
                      "התרומה נמדדת עבור המודל הזה בלבד (על קבוצת התיקוף); במודל אחר או באוכלוסייה אחרת היא יכולה להיות שונה.",
                      "תרומה לחיזוי אינה סיבתיות: משתנה עם תרומה גדולה קשור לתחזית המודל; אין כאן טענה סיבתית."]


def _sign(rr: Any, feature: str) -> float:
    coef = rr.run.csv("coefficients.csv")
    if not len(coef):
        return 0.0
    c = coef[(coef["feature"] == feature) & coef["selected"].astype(bool)]
    if "relative_to_reference" in c.columns:
        c = c[~c["relative_to_reference"].astype(bool)]
    if not len(c):
        return 0.0
    return float(c.loc[c["standardized_coefficient"].abs().idxmax(), "standardized_coefficient"])


# ============================================================================ figures
def management_figures(rs: Any, figs: Any) -> None:
    rr = rs.primary
    if rr is None:
        return
    run = rr.run
    wm = "EXPLORATORY 180-DAY OUTCOME – NOT EFALLS REPRODUCTION"
    steps_he = ["כל השורות בקובץ", "בתאריך התחזית", "עומדים בתנאי ההכללה", "עם מעקב שמיש ל-180 יום", "אחרי החרגת בעיית התזמון (D-00)", "אוכלוסיית המודל"]
    steps = [(steps_he[i], s[1], "") for i, s in enumerate(rr.info.get("funnel", []))]
    if steps:
        steps[-1] = (steps[-1][0], steps[-1][1], f"{int(run.build.get('n_events') or 0):,} נפילות")
        rs.figures["he_funnel"] = F.save(F.funnel(steps, title="על כמה מטופלים נבנה המודל", lang="he"), figs, "he_funnel", wm, lang="he")
    dec = rr.tables.get("deciles")
    if dec is not None:
        rs.figures["he_deciles"] = F.save(F.deciles(dec, lang="he"), figs, "he_deciles", wm, lang="he")
        rs.figures["he_calibration"] = F.save(_calibration_he(dec), figs, "he_calibration", wm, lang="he")
    if "roc" in rr.tables:
        a, lo, hi = _est(run.perf("test").get("auroc"))
        lab = f"{_label_he(rs, run.key)}: AUROC {a:.2f}" if a is not None else _label_he(rs, run.key)
        rs.figures["he_roc"] = F.save(F.roc([(lab, rr.tables["roc"], F.ORANGE)], lang="he", title="עד כמה המודל מבדיל בין מי שנפל למי שלא נפל"),
                                      figs, "he_roc", wm, lang="he")
    st = _stable_features(rr, 0.0).head(10)
    if len(st):
        vals = st["freq"].astype(float).tolist()
        signs = [_sign(rr, f) for f in st["canonical_feature"]]
        rs.figures["he_features"] = F.save(F.hbars([_fname(rs, f) for f in st["canonical_feature"]], vals, lang="he", fmt="{:.0f}%",
                                                   title="יציבות הבחירה: באיזו עקביות המודל משאיר כל משתנה",
                                                   xlabel="אחוז ההרצות החוזרות שבהן המשתנה נשאר במודל (כתום: קשור לסיכון חזוי גבוה יותר) – זו אינה מידת החשיבות",
                                                   colors=[F.ORANGE if s > 0 else F.BLUE for s in signs], ref=80.0), figs, "he_features", wm, lang="he")
    perm = _permutation(rr).head(10)
    if len(perm):
        rs.figures["he_permutation"] = F.save(F.hbars([_fname(rs, f) for f in perm["feature"]], perm["mean"].tolist(), lang="he", fmt="{:.3f}",
                                                      title="תרומה לחיזוי המודל: כמה נפגעת ההבחנה כשמשבשים את המשתנה",
                                                      xlabel="ירידה ב-AUROC בקבוצת התיקוף כשמערבבים את ערכי המשתנה (קו שגיאה: סטיית תקן בין חזרות)",
                                                      colors=[F.BLUE] * len(perm), err=perm["sd"].tolist()), figs, "he_permutation", wm, lang="he")
    if rs.d00:
        fig = _d00_matrix_he(rs)
        if fig is not None:
            rs.figures["he_d00_matrix"] = F.save(fig, figs, "he_d00_matrix", wm, lang="he")
        fig = _d00_top10_he(rs)
        if fig is not None:
            rs.figures["he_d00_top10"] = F.save(fig, figs, "he_d00_top10", wm, lang="he")
    if "lift" in rr.tables and "gains" in rr.tables:
        rs.figures["he_lift"] = F.save(F.gains_lift(rr.tables["gains"], rr.tables["lift"], lang="he"), figs, "he_lift", wm, lang="he")
    strict, ext = _run(rs, "STRICT"), _run(rs, "EXTENDED")
    if strict is not None and ext is not None:
        rs.figures["he_strict_vs_extended"] = F.save(_strict_vs_extended(rs, strict, ext), figs, "he_strict_vs_extended", wm, lang="he")
    rs.figures["he_roadmap"] = F.save(F.roadmap(ROADMAP_HE, 0, lang="he"), figs, "he_roadmap", wm, lang="he")


def _calibration_he(dec: pd.DataFrame) -> Any:
    fig = F._fig(6.2, 5.4)
    ax = fig.add_subplot(111)
    F._style(ax)
    x = 100 * pd.to_numeric(dec["mean_predicted"], errors="coerce")
    y = 100 * pd.to_numeric(dec["observed_rate"], errors="coerce")
    hi = float(np.nanmax([x.max(), y.max()])) * 1.1 if len(dec) else 1.0
    ax.plot([0, hi], [0, hi], color=F.INK2, ls="--", lw=1, label=F.rtl("התאמה מושלמת"))
    ax.scatter(x, y, s=60, color=F.ORANGE, zorder=3, label=F.rtl("עשירוני סיכון"))
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_xlabel(F.rtl("סיכון חזוי ממוצע (%)"))
    ax.set_ylabel(F.rtl("שיעור נפילות בפועל (%)"))
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_title(F.rtl("האם הסיכון החזוי תואם את המציאות"), fontsize=11, loc="right")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def _strict_vs_extended(rs: Any, strict: Any, ext: Any) -> Any:
    fig = F._fig(7.5, 3.4)
    ax = fig.add_subplot(111)
    F._style(ax)
    rows = [(strict, F.BLUE), (ext, F.ORANGE)]
    for i, (rr, col) in enumerate(rows):
        a, lo, hi = _est(rr.run.perf("test").get("auroc"))
        if a is None:
            continue
        ax.errorbar([a], [i], xerr=[[a - lo], [hi - a]] if lo is not None else None, fmt="o", color=col, ms=9, capsize=4, lw=2)
        ax.text(a, i + 0.22, f"{a:.3f}", ha="center", fontsize=9, color=F.INK)
    ax.set_yticks([0, 1], [F.rtl(f"{_label_he(rs, 'STRICT')} – {len(strict.run.features)} משתנים"), F.rtl(f"{_label_he(rs, 'EXTENDED')} – {len(ext.run.features)} משתנים")])
    ax.set_ylim(-0.6, 1.7)
    ax.axvline(0.5, color=F.MUTED, ls="--", lw=1)
    ax.set_xlabel(F.rtl("יכולת הבחנה AUROC (רווח סמך 95%) – אותם מטופלים בקבוצת הבדיקה"))
    ax.set_title(F.rtl("מה קרה כשהוספנו משתנים זמינים ממאוחדת"), fontsize=11, loc="right")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


# ============================================================================ D-00 sensitivity (read from d00_sensitivity.json)
def _d00_rows(rs: Any) -> list[tuple[str, dict[str, Any] | None, str]]:
    """(Hebrew label, cell record or None for pending, colour) in the pre-declared display order."""
    d = rs.d00
    cells = d.get("cells") or {}
    sets = d.get("feature_sets") or {}
    prim = next((v for v in sets.values() if v.get("role") == "primary"), None)
    sec = next((v for v in sets.values() if v.get("role") == "secondary"), None)
    rows: list[tuple[str, dict[str, Any] | None, str]] = [("המודל הנוכחי (EXTENDED, אוכלוסייה נקייה)", cells.get("FULL_EXTENDED_D00_CLEAN"), F.BLUE)]
    if prim:
        rows.append(("כל המטופלים, רק משתנים שהוכחו בטוחים", cells.get(f"{prim['SAFE_EXTENDED']['name']}_FULL"), F.ORANGE))
    if sec:
        rows.append(("כל המטופלים, בטוחים + לא מוכרעים", cells.get(f"{sec['SAFE_EXTENDED']['name']}_FULL"), F.ORANGE))
    rows += [("בלי נפילות קודמות", cells.get("EXTENDED_NO_FALLS"), F.AQUA), ("בלי קושי בהליכה", cells.get("EXTENDED_NO_MOBILITY"), F.AQUA),
             ("בלי שניהם", cells.get("EXTENDED_NO_FALLS_NO_MOBILITY"), F.AQUA), ("אחרי תיקון ה-DWH (טרם הורץ)", None, F.MUTED)]
    return rows


def _d00_matrix_he(rs: Any) -> Any:
    rows = _d00_rows(rs)
    if not any(r for _l, r, _c in rows if r and r.get("auroc") is not None):
        return None
    fig = F._fig(8.4, 0.55 * len(rows) + 1.6)
    ax = fig.add_subplot(111)
    F._style(ax)
    lows = [r["auroc_ci"][0] for _l, r, _c in rows if r and r.get("auroc_ci") and r["auroc_ci"][0] is not None]
    x0 = min([0.48] + [v - 0.02 for v in lows])
    for i, (lab, r, col) in enumerate(rows):
        if r is None or r.get("auroc") is None:
            ax.text((x0 + 1.0) / 2, i, F.rtl("ממתין לתיקון מקור הנתונים – אין מספר" if r is None else "לא הורץ"), va="center", ha="center",
                    fontsize=9, color=F.INK2, bbox={"boxstyle": "round,pad=0.3", "fc": F.SURFACE, "ec": F.MUTED, "ls": "--"})
            continue
        a, ci = r.get("auroc"), r.get("auroc_ci")
        ax.errorbar([a], [i], xerr=[[a - ci[0]], [ci[1] - a]] if ci and ci[0] is not None else None, fmt="o", color=col, ms=8, capsize=4, lw=2)
        ax.text(ci[1] if ci and ci[1] is not None else a, i, f"  {a:.3f}", va="center", fontsize=9, color=F.INK)
    ax.set_yticks(range(len(rows)), [F.rtl(r[0]) for r in rows], fontsize=9)
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.axvline(0.5, color=F.MUTED, ls="--", lw=1)
    ax.set_xlim(x0, 1.0)
    ax.set_xlabel(F.rtl("יכולת הבחנה AUROC (רווח סמך 95%) בקבוצת הבדיקה; כתום = כל המטופלים (אוכלוסייה אחרת, לא השוואה ישירה)"))
    ax.set_title(F.rtl("בדיקת רגישות לבעיית התזמון: כמה יכולת נשארת"), fontsize=11, loc="right")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def _d00_capture(rs: Any, cell: dict[str, Any] | None, top: float = 10.0) -> tuple[float, float, float] | None:
    if not cell:
        return None
    risk = pd.DataFrame(rs.d00.get("risk_concentration") or [])
    if not len(risk):
        return None
    r = risk[(risk["model"] == cell.get("serving_cell")) & (risk["top_pct"] == top)]
    if not len(r) or pd.isna(r["pct_of_all_falls_captured"].iloc[0]):
        return None
    r = r.iloc[0]
    return float(r["pct_of_all_falls_captured"]), float(r["capture_ci_low"]), float(r["capture_ci_high"])


def _d00_top10_he(rs: Any) -> Any:
    rows = [(lab, _d00_capture(rs, r), col) for lab, r, col in _d00_rows(rs)]
    if not any(v for _l, v, _c in rows):
        return None
    fig = F._fig(8.4, 0.55 * len(rows) + 1.6)
    ax = fig.add_subplot(111)
    F._style(ax)
    for i, (lab, v, col) in enumerate(rows):
        if v is None:
            ax.text(50, i, F.rtl("אין מספר (טרם הורץ או תא קטן מדי)"), va="center", ha="center", fontsize=9, color=F.INK2,
                    bbox={"boxstyle": "round,pad=0.3", "fc": F.SURFACE, "ec": F.MUTED, "ls": "--"})
            continue
        ax.barh(i, v[0], color=col, height=0.6)
        ax.errorbar([v[0]], [i], xerr=[[v[0] - v[1]], [v[2] - v[0]]], fmt="none", ecolor=F.INK2, capsize=3, lw=1)
        ax.text(v[2] + 1, i, f"{v[0]:.0f}%", va="center", fontsize=9, color=F.INK)
    ax.axvline(10, color=F.MUTED, ls="--", lw=1)
    ax.set_yticks(range(len(rows)), [F.rtl(r[0]) for r in rows], fontsize=9)
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.set_xlim(0, 100)
    ax.set_xlabel(F.rtl("אחוז מכלל הנפילות שנמצאו בעשירון הסיכון העליון (רווח סמך 95%); הקו המקווקו = 10% (ללא ריכוז)"))
    ax.set_title(F.rtl("האם 10% בסיכון הגבוה ביותר עדיין מרכזים את הנפילות"), fontsize=11, loc="right")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def _d00_section(rs: Any) -> list[str]:
    d = rs.d00
    if not d:
        return []
    cells = d.get("cells") or {}
    ab = {r["compared"]: r for r in ((d.get("comparisons") or {}).get("ablations") or [])}
    ref = cells.get("FULL_EXTENDED_D00_CLEAN") or {}
    out = ["<h2>בדיקת רגישות לבעיית התזמון (D-00)</h2>",
           "<p>בחלק מהשורות נרשמו אבחנה או נפילה ביום התחזית עצמו. המודל הנוכחי נבנה בלי השורות האלה. כדי לבדוק כמה התוצאה תלויה בזה, הרצנו "
           "מודלים נוספים שנקבעו מראש: (1) על כל המטופלים עם תוצא, רק עם משתנים שמקורם הוכח כבטוח; (2) המודל הנוכחי בלי נפילות קודמות, בלי קושי "
           "בהליכה, ובלי שניהם – על אותם מטופלים ואותה קבוצת בדיקה. המודל שיורץ אחרי תיקון מחסן הנתונים מוצג כממתין, בלי מספר.</p>",
           _img(rs, "he_d00_matrix", "יכולת ההבחנה של המודל הנוכחי, של המודלים הבטוחים על כל המטופלים ושל ניסויי ההסרה"),
           _img(rs, "he_d00_top10", "ריכוז הנפילות בעשירון הסיכון העליון בכל אחד מהמודלים")]
    for cell, he in (("EXTENDED_NO_FALLS", "בלי נפילות קודמות"), ("EXTENDED_NO_MOBILITY", "בלי קושי בהליכה"), ("EXTENDED_NO_FALLS_NO_MOBILITY", "בלי שניהם")):
        r = ab.get(cell)
        if r and r.get("delta_auroc") is not None and r.get("delta_auroc_ci_low") is not None:
            clear = "ירידה ברורה" if r["delta_auroc_ci_high"] < 0 else ("עלייה" if r["delta_auroc_ci_low"] > 0 else "לא הוכח הבדל ברור")
            out.append(f"<div class='box'>{he}: AUROC {r['compared_auroc']:.3f} לעומת {r['reference_auroc']:.3f} במודל הנוכחי – הפרש "
                       f"{r['delta_auroc']:+.3f} (רווח סמך 95%: {r['delta_auroc_ci_low']:+.3f} עד {r['delta_auroc_ci_high']:+.3f}); {clear}. "
                       "אותם מטופלים ואותה קבוצת בדיקה (השוואה מזווגת).</div>")
    out.append("<p class='box'>ההשוואה בין \"כל המטופלים\" לבין המודל הנוכחי אינה השוואה ישירה: אוכלוסייה אחרת וקבוצת בדיקה אחרת. "
               "הפירוט המלא, כולל איזה משתנה הוכח בטוח ומה לא ניתן להכריע מהקובץ, בדוח D00_SENSITIVITY_REPORT.html ובסיכום D00_SENSITIVITY_SUMMARY_HE.md.</p>")
    return out


# ============================================================================ conclusions
def conclusions_he(rs: Any) -> list[str]:
    out: list[str] = []
    rr = rs.primary
    if rr is None:
        return ["לא נמצאה הרצה שהושלמה; אין מסקנות."]
    run = rr.run
    name = _label_he(rs, run.key)
    a, lo, hi = _est(run.perf("test").get("auroc"))
    if a is not None and lo is not None:
        if lo > 0.5:
            out.append(f"ל{name} יש יכולת מדידה להבחין בין מטופלים בסיכון גבוה יותר לנמוך יותר: AUROC {a:.2f} (רווח סמך 95%: {lo:.2f}–{hi:.2f}). "
                       f"כלומר, כשמשווים מטופל אחד שנפל למטופל אחד שלא נפל, המודל נותן סיכון גבוה יותר למי שנפל בכ-{100 * a:.0f}% מההשוואות.")
        else:
            out.append(f"לא הוכחה יכולת הבחנה של {name}: רווח הסמך של AUROC ({lo:.2f}–{hi:.2f}) כולל את 0.5 (ניחוש מקרי).")
    strict = _run(rs, "STRICT")
    pair = next((p for p in rs.paired if strict is not None and p.get("compared") == strict.run.display), None)
    if pair and pair.get("auroc_estimate") is not None and run.key == "EXTENDED":
        d, dl, dh = -pair["auroc_estimate"], -pair["auroc_ci_high"], -pair["auroc_ci_low"]
        if dl > 0:
            out.append(f"הוספת המשתנים המורחבים (EXTENDED) שיפרה את ההבחנה לעומת המודל הבסיסי (STRICT) על אותם מטופלים: תוספת AUROC של {d:.3f} "
                       f"(רווח סמך 95%: {dl:.3f}–{dh:.3f}).")
        elif dh < 0:
            out.append(f"המודל המורחב הבחין פחות טוב מהמודל הבסיסי על אותם מטופלים: הפרש AUROC {d:.3f} (רווח סמך 95%: {dl:.3f}–{dh:.3f}).")
        else:
            out.append(f"לא נמצא הבדל ברור בהבחנה בין המודל המורחב לבסיסי על אותם מטופלים: הפרש AUROC {d:.3f} (רווח סמך 95%: {dl:.3f}–{dh:.3f}).")
    perm = _permutation(rr)
    if len(perm) and float(perm["mean"].iloc[0]) > 0:
        top = perm[perm["mean"] > 0].head(3)
        out.append("התרומה הגדולה ביותר לחיזוי המודל (הירידה בהבחנה כשמשבשים את המשתנה): "
                   + ", ".join(f"{_fname(rs, f)} ({m:.3f})" for f, m in zip(top["feature"], top["mean"]))
                   + ". זו תרומה לחיזוי של המודל הזה – לא סיבה לנפילה; יציבות הבחירה מוצגת בנפרד.")
    else:
        st = _stable_features(rr, 0.8)
        if len(st):
            out.append("המשתנים שנשארו במודל ביותר מ-80% מההרצות החוזרות (יציבות, לא חשיבות): "
                       + ", ".join(_fname(rs, f) for f in st["canonical_feature"].head(5)) + ". זהו קשר סטטיסטי לתחזית, לא סיבה לנפילה.")
    lift = rr.tables.get("lift")
    if lift is not None and len(lift):
        r = lift[lift["top_pct"] == 10.0]
        if len(r) and pd.notna(r["capture_ci_low"].iloc[0]):
            cap, clo = float(r["pct_of_all_falls_captured"].iloc[0]), float(r["capture_ci_low"].iloc[0])
            if clo > 10.0:
                out.append(f"קבוצה קטנה יחסית מרכזת חלק גדול יחסית מהנפילות: 10% המטופלים בסיכון החזוי הגבוה ביותר כללו {cap:.0f}% מהנפילות שאירעו "
                           f"(רווח סמך 95%: {clo:.0f}%–{float(r['capture_ci_high'].iloc[0]):.0f}%), פי {float(r['lift'].iloc[0]):.1f} מהשיעור באוכלוסייה.")
            else:
                out.append(f"10% המטופלים בסיכון הגבוה ביותר כללו {cap:.0f}% מהנפילות; לא הוכח ריכוז גבוה מהצפוי באקראי.")
    s, slo, shi = _est(run.perf("test").get("calibration_slope"))
    o, olo, ohi = _est(run.perf("test").get("oe_ratio"))
    if s is not None and slo is not None and o is not None and olo is not None:
        if slo <= 1 <= shi and olo <= 1 <= ohi:
            out.append(f"הסיכון החזוי תואם בממוצע את שיעור הנפילות בפועל (יחס נצפה/חזוי {o:.2f}, שיפוע כיול {s:.2f}; רווחי הסמך כוללים את הערך האידיאלי 1).")
        else:
            out.append(f"יש סטייה בכיול התחזיות (יחס נצפה/חזוי {o:.2f}, שיפוע כיול {s:.2f}); יש לבדוק כיול מחדש לפני כל שימוש.")
    ab = {r["compared"]: r for r in (((rs.d00 or {}).get("comparisons") or {}).get("ablations") or [])}
    nf = ab.get("EXTENDED_NO_FALLS")
    if nf and nf.get("delta_auroc") is not None and nf.get("delta_auroc_ci_low") is not None:
        out.append(f"בלי מידע על נפילות קודמות, על אותם מטופלים, AUROC הוא {nf['compared_auroc']:.2f} לעומת {nf['reference_auroc']:.2f} "
                   f"(הפרש {nf['delta_auroc']:+.3f}, רווח סמך 95%: {nf['delta_auroc_ci_low']:+.3f} עד {nf['delta_auroc_ci_high']:+.3f}).")
    out.append("כל התוצאות חקרניות: תוצא של 180 יום במאוחדת (לא התוצא של eFalls), נקודת זמן אחת וחלוקה פנימית; נדרשים תיקוף זמני, תיקוף חיצוני ותיקון "
               "התזמון במחסן הנתונים לפני כל שקילה של שימוש קליני.")
    return out[:6] if len(out) <= 6 else out[:5] + [out[-1]]


# ============================================================================ HTML / MD
CSS_HE = """
body{margin:0;font-family:Arial,'Segoe UI',Helvetica,sans-serif;background:#fcfcfb;color:#0b0b0b;line-height:1.6;font-size:16px}
.banner{background:#b00020;color:#fff;font-weight:700;text-align:center;padding:10px;font-size:15px}
.synthetic{background:#6b21a8;color:#fff;font-weight:700;text-align:center;padding:6px}
main{max-width:1000px;margin:0 auto;padding:20px 28px 70px}
h1{font-size:26px}h2{font-size:20px;border-bottom:2px solid #e4e3df;padding-bottom:4px;margin-top:36px}
.concl{background:#fff;border:1px solid #e4e3df;border-right:6px solid #2a78d6;border-radius:6px;padding:12px 18px}
.concl li{margin:6px 0}
.box{background:#f3f2ee;border-radius:6px;padding:10px 14px;margin:10px 0;font-size:15px}
.limit{background:#fff3cd;border-right:5px solid #d39e00;padding:10px 14px}
figure{margin:12px 0}figure img{max-width:100%;border:1px solid #e4e3df;border-radius:4px;background:#fff}figcaption{font-size:13px;color:#52514e}
.kpi{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:10px;margin:12px 0}.kpi div{background:#fff;border:1px solid #e4e3df;border-radius:6px;padding:10px}
.kpi b{display:block;font-size:22px}.kpi span{font-size:13px;color:#52514e}
.ltr{direction:ltr;unicode-bidi:embed}
"""


def _img(rs: Any, key: str, caption: str) -> str:
    fig = rs.figures.get(key)
    if not fig:
        return ""
    p = rs.out_dir / "figures" / fig["png"]
    if not p.exists():
        return ""
    data = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"<figure><img alt='{_esc(caption)}' src='data:image/png;base64,{data}'><figcaption>{_esc(caption)}</figcaption></figure>"


def _page(title: str, body: str, rs: Any, extra_css: str = "") -> str:
    return ("<!DOCTYPE html><html lang='he' dir='rtl'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{_esc(title)}</title><style>{CSS_HE}{extra_css}</style></head><body>"
            f"<div class='banner'>{_esc(WATERMARK_HE)} <span class='ltr'>| EXPLORATORY 180-DAY OUTCOME – NOT EFALLS REPRODUCTION</span></div>"
            + (f"<div class='synthetic'>{SYNTHETIC_HE} | SYNTHETIC DATA</div>" if rs.synthetic else "") + f"<main>{body}</main></body></html>")


def _facts(rs: Any) -> dict[str, Any]:
    rr = rs.primary
    run = rr.run
    coh = run.metrics.get("cohort") or {}
    perf = run.perf("test")
    lift = rr.tables.get("lift")
    top = {}
    if lift is not None:
        for _, r in lift.iterrows():
            top[float(r["top_pct"])] = r
    return {"name": _label_he(rs, run.key), "n": coh.get("n_rows"), "patients": coh.get("n_patients"), "events": coh.get("n_events"),
            "prev": coh.get("prevalence"), "n_test": perf.get("n"), "auroc": _est(perf.get("auroc")), "pr": _est(perf.get("pr_auc")),
            "slope": _est(perf.get("calibration_slope")), "oe": _est(perf.get("oe_ratio")), "top": top, "n_feat": len(run.features),
            "d00": int(run.build.get("n_rows_timing_violation") or 0), "index_date": ", ".join(run.build.get("index_dates") or [])}


def _limitations(rs: Any, f: dict[str, Any]) -> list[str]:
    items = ["התוצא הוא נפילה או שבר תוך 180 יום לפי נתוני מאוחדת – תוצא חקרני, לא התוצא של eFalls (פנייה למיון או אשפוז בשל נפילה תוך 12 חודשים).",
             "זה אינו שחזור מדויק של מודל eFalls: חלק מהמשתנים הותאמו בקירוב (APPROXIMATE) וחלק אינם זמינים.",
             f"נקודת זמן אחת ({f['index_date']}) וחלוקה אקראית פנימית: עדיין אין תיקוף לאורך זמן ואין תיקוף חיצוני.",
             "תיקון התזמון במחסן הנתונים (DWH) עדיין ממתין: חלק מהרשומות כוללות מידע מיום התחזית עצמו."]
    if f["d00"]:
        items.append(f"בגלל בעיית התזמון (D-00) הוחרגו {f['d00']:,} שורות מאוכלוסיית המודל; לכן התוצאה מתארת את האוכלוסייה הנקייה בלבד.")
    items.append("המשמעות העסקית של חלק מהשדות טרם אושרה על ידי מומחי תוכן, והמודל לא נבחן בשימוש קליני.")
    return items


def render_management_html(rs: Any) -> str:
    if rs.primary is None:
        return _page("דוח הנהלה", "<p>לא נמצאה הרצה שהושלמה.</p>", rs)
    f = _facts(rs)
    a, alo, ahi = f["auroc"]
    b = [f"<h1>מודל חקרני לחיזוי נפילות – דוח להנהלה</h1>",
         "<h2>מסקנות עיקריות</h2><ul class='concl'>" + "".join(f"<li>{_esc(c)}</li>" for c in rs.conclusions_he) + "</ul>",
         "<h2>מה ניסינו לחזות?</h2>",
         "<p>המטרה: להעריך לכל מבוטח בגיל 65 ומעלה את הסיכון לנפילה (או שבר) ב-180 הימים שאחרי תאריך התחזית, רק על סמך מידע שהיה ידוע לפני "
         "תאריך זה – גיל, מין, תרופות, רשמי מחלות ונפילות קודמות. זהו מודל לחקר ולמידה, לא כלי קליני.</p>",
         "<h2>על כמה מטופלים?</h2>",
         f"<p>אוכלוסיית המודל כוללת {f['n']:,} מבוטחים, מתוכם {f['events']:,} נפלו תוך 180 יום ({_pct(f['prev'])}). המודל אומן על חלק מהם ונבדק על "
         f"קבוצת בדיקה נפרדת של {f['n_test']:,} מבוטחים שהמודל לא ראה בזמן הבנייה.</p>",
         _img(rs, "he_funnel", "מקובץ הנתונים ועד אוכלוסיית המודל"),
         "<h2>עד כמה המודל מצליח להבדיל בין סיכון גבוה לנמוך?</h2>",
         "<p>מחלקים את המבוטחים בקבוצת הבדיקה לעשר קבוצות שוות לפי הסיכון שהמודל חזה. אם המודל מבדיל היטב, שיעור הנפילות בפועל עולה מקבוצה לקבוצה.</p>",
         _img(rs, "he_deciles", "שיעור הנפילות בפועל בכל עשירון של סיכון חזוי (קבוצת הבדיקה)")]
    if a is not None:
        b.append(f"<div class='box'>מדד AUROC הוא {a:.2f} (רווח סמך 95%: {alo:.2f}–{ahi:.2f}). בשפה פשוטה: אם בוחרים באקראי מבוטח אחד שנפל ומבוטח אחד "
                 f"שלא נפל, המודל נותן סיכון גבוה יותר למי שנפל בכ-{100 * a:.0f}% מהמקרים (50% = ניחוש מקרי, 100% = הבחנה מושלמת).</div>")
    p, plo, phi = f["pr"]
    if p is not None:
        b.append(f"<div class='box'>נפילות תוך 180 יום אינן שכיחות ({_pct(f['prev'])}). לכן חשוב לבדוק גם: מבין מי שהמודל מסמן כבסיכון, כמה באמת נופלים? "
                 f"מדד PR-AUC הוא {p:.3f} (רווח סמך 95%: {plo:.3f}–{phi:.3f}), לעומת {f['prev']:.3f} אם היינו בוחרים באקראי.</div>")
    b.append(_img(rs, "he_roc", "עקומת ROC: ככל שהעקומה רחוקה יותר מהקו האלכסוני, ההבחנה טובה יותר"))
    b.append("<h2>האם התחזיות מכוילות?</h2><p>מודל מכויל הוא מודל שבו \"סיכון של 10%\" אכן מתאים לכ-10% נפילות בפועל. הנקודות צריכות להיות קרובות לקו האלכסוני.</p>")
    s, slo, shi = f["slope"]
    o, olo, ohi = f["oe"]
    if o is not None:
        b.append(f"<div class='box'>יחס נצפה/חזוי: {o:.2f} (1 = התאמה מלאה בממוצע; רווח סמך {olo:.2f}–{ohi:.2f}). שיפוע כיול: {s:.2f} (1 = אידיאלי; "
                 f"רווח סמך {slo:.2f}–{shi:.2f}).</div>")
    b.append(_img(rs, "he_calibration", "סיכון חזוי מול שיעור נפילות בפועל, לפי עשירוני סיכון"))
    b.append("<h2>אילו משתנים בולטים במודל?</h2><p>שתי שאלות שונות, שני תרשימים שונים:</p>"
             "<h3>1. יציבות הבחירה</h3><p>באיזו עקביות המודל (LASSO) משאיר את המשתנה כשבונים אותו מחדש שוב ושוב על מדגמים חוזרים. כתום = קשור לתחזית "
             "סיכון גבוהה יותר; כחול = נמוכה יותר. כמה משתנים יכולים להישאר ב-100% מההרצות – זה לא אומר שהם חשובים באותה מידה.</p>")
    b.append(_img(rs, "he_features", "יציבות הבחירה: אחוז ההרצות החוזרות שבהן המשתנה נשאר במודל"))
    b.append("<h3>2. תרומה לחיזוי המודל</h3><p>כמה יכולת ההבחנה של המודל נפגעת כשמשבשים (מערבבים באקראי) את ערכי המשתנה, על קבוצת התיקוף. "
             "ככל שהעמודה ארוכה יותר, המודל נשען יותר על המידע במשתנה הזה.</p>")
    b.append(_img(rs, "he_permutation", "תרומה לחיזוי המודל: ירידה ב-AUROC כשמשבשים את המשתנה"))
    b.append("<div class='box'><b>איך לקרוא את שני התרשימים יחד:</b><ul>" + "".join(f"<li>{_esc(x)}</li>" for x in IMPORTANCE_NOTE_HE) + "</ul></div>")
    b.append("<h2>האם אפשר לרכז מאמץ באוכלוסייה בסיכון גבוה?</h2>")
    for k in (5.0, 10.0, 20.0):
        r = f["top"].get(k)
        if r is not None and pd.notna(r["pct_of_all_falls_captured"]):
            b.append(f"<div class='box'>{k:g}% המבוטחים בסיכון החזוי הגבוה ביותר: כללו {float(r['pct_of_all_falls_captured']):.0f}% מכלל הנפילות "
                     f"(רווח סמך 95%: {float(r['capture_ci_low']):.0f}%–{float(r['capture_ci_high']):.0f}%). שיעור הנפילות בקבוצה זו: "
                     f"{_pct(r['observed_prevalence'])} – פי {float(r['lift']):.1f} מהממוצע.</div>")
    b.append(_img(rs, "he_lift", "כמה מהנפילות נמצאות בקבוצות הסיכון העליונות (קבוצת הבדיקה, לאחר שהמודל הוקפא)"))
    strict, ext = _run(rs, "STRICT"), _run(rs, "EXTENDED")
    if strict is not None and ext is not None:
        b.append("<h2>STRICT מול EXTENDED</h2><p>המודל הבסיסי (STRICT) כולל רק משתנים שההתאמה שלהם להגדרות eFalls גבוהה. המודל המורחב (EXTENDED) מוסיף משתנים "
                 "זמינים במאוחדת שההתאמה שלהם מקורבת – למשל נפילות קודמות, מספר תרופות וקושי בהליכה. שני המודלים נבדקו על אותם מבוטחים.</p>")
        b.append(_img(rs, "he_strict_vs_extended", "יכולת ההבחנה של שני המודלים על אותה קבוצת בדיקה"))
    b += _d00_section(rs)
    b.append("<h2>מה מגביל את התוצאה כרגע?</h2><div class='limit'><ul>" + "".join(f"<li>{_esc(x)}</li>" for x in _limitations(rs, f)) + "</ul></div>")
    b.append("<h2>מה השלב הבא?</h2>" + _img(rs, "he_roadmap", "מפת הדרכים: מהמודל החקרני ועד שקילת שימוש קליני"))
    b.append("<p class='box'>כל המספרים חושבו על קבוצת בדיקה שהמודל לא ראה בזמן הבנייה, ורק לאחר שהמודל הוקפא. הדוח מתאר את המודל ואינו משמש לכוונון מחדש שלו.</p>")
    return _page("מודל חקרני לחיזוי נפילות – דוח הנהלה", "".join(b), rs)


def render_management_md(rs: Any) -> str:
    if rs.primary is None:
        return "# סיכום להנהלה\n\nלא נמצאה הרצה שהושלמה.\n"
    f = _facts(rs)
    L = ["<div dir=\"rtl\">", "", "# מודל חקרני לחיזוי נפילות – סיכום להנהלה", "", f"**{WATERMARK_HE}**", ""]
    if rs.synthetic:
        L += [f"**{SYNTHETIC_HE}**", ""]
    L += ["## מסקנות עיקריות", ""] + [f"- {c}" for c in rs.conclusions_he] + [""]
    L += ["## במספרים", "", f"- אוכלוסיית המודל: {f['n']:,} מבוטחים; נפלו תוך 180 יום: {f['events']:,} ({_pct(f['prev'])}).",
          f"- קבוצת בדיקה נפרדת: {f['n_test']:,} מבוטחים.", f"- המודל העיקרי (מוגדר מראש): {f['name']}, {f['n_feat']} משתנים."]
    a, alo, ahi = f["auroc"]
    if a is not None:
        L.append(f"- יכולת הבחנה (AUROC): {a:.2f} (רווח סמך 95%: {alo:.2f}–{ahi:.2f}).")
    for k in (5.0, 10.0, 20.0):
        r = f["top"].get(k)
        if r is not None and pd.notna(r["pct_of_all_falls_captured"]):
            L.append(f"- {k:g}% בסיכון הגבוה ביותר כללו {float(r['pct_of_all_falls_captured']):.0f}% מהנפילות (פי {float(r['lift']):.1f} מהממוצע).")
    perm = _permutation(rs.primary).head(5)
    if len(perm):
        L += ["", "## תרומה לחיזוי המודל (לא יציבות, לא סיבתיות)", ""] + [f"- {_fname(rs, x)}: {m:.3f}" for x, m in zip(perm["feature"], perm["mean"])]
        L += [""] + [f"- {x}" for x in IMPORTANCE_NOTE_HE]
    if rs.d00:
        L += ["", "## בדיקת רגישות לבעיית התזמון (D-00)", ""]
        for lab, r, _c in _d00_rows(rs):
            if r is None:
                L.append(f"- {lab}: ממתין – אין מספר.")
            elif r.get("auroc") is not None:
                ci = r.get("auroc_ci") or [None, None]
                L.append(f"- {lab}: AUROC {r['auroc']:.3f}" + (f" ({ci[0]:.3f}–{ci[1]:.3f})" if ci[0] is not None else ""))
        L.append("- השוואה בין \"כל המטופלים\" לאוכלוסייה הנקייה אינה השוואה ישירה (אוכלוסייה וקבוצת בדיקה שונות).")
    L += ["", "## מגבלות", ""] + [f"- {x}" for x in _limitations(rs, f)]
    L += ["", "## השלב הבא", "", " ← ".join(ROADMAP_HE), "", "תרשימים: figures/he_*.png (ו-SVG). הדוח המלא: MANAGEMENT_MODEL_REPORT_HE.html.", "", "</div>", ""]
    return "\n".join(L)


ONE_PAGER_CSS = """
@page{size:A4;margin:12mm}
@media print{.banner,.synthetic{-webkit-print-color-adjust:exact;print-color-adjust:exact}main{padding:0}}
main{max-width:190mm}body{font-size:13px}h1{font-size:20px;margin:6px 0}h2{font-size:15px;margin:12px 0 4px;border:none}
.two{display:grid;grid-template-columns:1fr 1fr;gap:12px}figure img{max-height:70mm;object-fit:contain}.kpi b{font-size:18px}
"""


def _d00_one_pager(rs: Any) -> list[str]:
    if not rs.d00:
        return []
    parts = []
    for lab, r, _c in _d00_rows(rs):
        if r is not None and r.get("auroc") is not None:
            parts.append(f"{lab}: {r['auroc']:.2f}")
    if not parts:
        return []
    return ["<h2>בדיקת רגישות לבעיית התזמון (AUROC)</h2><p>" + _esc("; ".join(parts)) + ". המודל שאחרי תיקון מחסן הנתונים – ממתין, ללא מספר. "
            "\"כל המטופלים\" היא אוכלוסייה אחרת (לא השוואה ישירה).</p>"]


def render_one_pager(rs: Any) -> str:
    if rs.primary is None:
        return _page("תקציר מנהלים", "<p>לא נמצאה הרצה שהושלמה.</p>", rs, ONE_PAGER_CSS)
    f = _facts(rs)
    a, alo, ahi = f["auroc"]
    p, _plo, _phi = f["pr"]
    o, _olo, _ohi = f["oe"]
    t10 = f["top"].get(10.0)
    kpis = [("מבוטחים באוכלוסיית המודל", f"{f['n']:,}"), ("נפלו תוך 180 יום", _pct(f["prev"])),
            ("יכולת הבחנה AUROC", f"{a:.2f} ({alo:.2f}–{ahi:.2f})" if a is not None else "לא זמין"),
            ("PR-AUC (מקרי = שיעור הנפילות)", f"{p:.3f} מול {f['prev']:.3f}" if p is not None else "לא זמין"),
            ("יחס נצפה/חזוי (1 = מכויל)", f"{o:.2f}" if o is not None else "לא זמין")]
    if t10 is not None and pd.notna(t10["pct_of_all_falls_captured"]):
        kpis.append(("נפילות בעשירון הסיכון העליון", f"{float(t10['pct_of_all_falls_captured']):.0f}%"))
    perm = _permutation(rs.primary)
    perm = perm[perm["mean"] > 0].head(5) if len(perm) else perm
    if len(perm):
        signals = ", ".join(_fname(rs, x) for x in perm["feature"]) + " (לפי הירידה בהבחנה כשמשבשים את המשתנה; יציבות הבחירה אינה מידת חשיבות)"
    else:
        st = _stable_features(rs.primary, 0.8)
        signals = ", ".join(_fname(rs, x) for x in st["canonical_feature"].head(5)) if len(st) else "אין משתנה שנשאר ביותר מ-80% מההרצות"
    b = ["<h1>חיזוי נפילות בגיל 65+ – תקציר מנהלים (עמוד אחד)</h1>",
         "<h2>מטרה</h2><p>להעריך מראש את הסיכון לנפילה תוך 180 יום, ממידע שידוע לפני תאריך התחזית, כדי ללמוד אם אפשר למקד מאמצי מניעה.</p>",
         f"<h2>המודל העיקרי הנוכחי (מוגדר מראש)</h2><p>{_esc(f['name'])} – {f['n_feat']} משתנים; נבדק על {f['n_test']:,} מבוטחים שלא שימשו לבנייתו.</p>",
         "<div class='kpi'>" + "".join(f"<div><b class='ltr'>{_esc(v)}</b><span>{_esc(k)}</span></div>" for k, v in kpis) + "</div>",
         f"<h2>התרומה הגדולה ביותר לחיזוי המודל (לא סיבות)</h2><p>{_esc(signals)}</p>",
         *_d00_one_pager(rs),
         "<div class='two'><div>" + _img(rs, "he_lift", "ריכוז הנפילות בקבוצות הסיכון העליונות") + "</div><div><h2>מגבלות עיקריות</h2><ul>"
         + "".join(f"<li>{_esc(x)}</li>" for x in _limitations(rs, f)[:5]) + "</ul></div></div>",
         "<h2>השלב הבא</h2><p>תיקון התזמון במחסן הנתונים, הרחבת המשתנים ממאוחדת ותיקוף לאורך זמן – לפני כל שקילה של שימוש קליני.</p>"]
    return _page("חיזוי נפילות – תקציר מנהלים", "".join(b), rs, ONE_PAGER_CSS)
