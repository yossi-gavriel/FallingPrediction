"""Figures of the model reports. Every figure is aggregate, carries the watermark and is saved twice: PNG (200 dpi) and SVG, so it can be
pasted into PowerPoint, an e-mail or a paper. Technical figures use English and canonical names; management figures use Hebrew.

Hebrew in matplotlib: glyphs come from the bundled DejaVu Sans (identical on Windows). matplotlib 3.11 (the locked version) reorders
right-to-left text itself; :func:`native_bidi` measures that once and :func:`rtl` reorders manually only on an older matplotlib.
"""

from __future__ import annotations

import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

if "matplotlib.pyplot" not in sys.modules:
    matplotlib.use("Agg")

from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

DPI = 200
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"
SERIES = (BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED)
INK, INK2, GRID, SURFACE, MUTED = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#b9b8b3"
_HEB = re.compile(r"[֐-׿]")
_RUN = re.compile(r"([֐-׿]+(?:['\"״׳][֐-׿]+)*)|([A-Za-z0-9](?:[A-Za-z0-9.,:/%+_\-]*[A-Za-z0-9%+])?)|(\s+)|(.)")
_MIRROR = str.maketrans("()[]{}<>", ")(][}{><")


@lru_cache(maxsize=1)
def native_bidi() -> bool:
    """True when this matplotlib lays out right-to-left text itself (3.11+ shapes text with bidi reordering): measured, not assumed."""
    try:
        from matplotlib._text_helpers import layout
        from matplotlib.font_manager import FontProperties, findfont, get_font

        font = get_font(findfont(FontProperties(family="DejaVu Sans")))
        font.set_size(12, 72)
        xs = {g.char: g.x for g in layout("\u05d0\u05d1", font)}
        return xs["\u05d0"] > xs["\u05d1"]
    except Exception:  # noqa: BLE001 - private API changed: fall back to manual reordering
        return False


def rtl(text: str) -> str:
    """Hebrew for display: unchanged when matplotlib reorders right-to-left text itself; otherwise the visual order for a left-to-right
    renderer (runs of Hebrew reversed, Latin / numeric runs kept, run order reversed, brackets mirrored). No-op for text without Hebrew."""
    if not text or not _HEB.search(text) or native_bidi():
        return text
    out = []
    for line in str(text).split("\n"):
        runs = []
        for heb, ltr, space, other in _RUN.findall(line):
            if heb:
                runs.append(heb[::-1])
            elif ltr:
                runs.append(ltr)
            elif space:
                runs.append(space)
            else:
                runs.append(other.translate(_MIRROR))
        out.append("".join(reversed(runs)))
    return "\n".join(out)


def T(text: str, lang: str) -> str:
    return rtl(text) if lang == "he" else text


def _fig(w: float, h: float) -> Figure:
    fig = Figure(figsize=(w, h), facecolor=SURFACE)
    FigureCanvasAgg(fig)
    return fig


def _style(ax: Any) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


def save(fig: Figure, out_dir: Path, stem: str, watermark: str, lang: str = "en", note: str = "") -> dict[str, str]:
    """Write <stem>.png and <stem>.svg; returns their file names."""
    fig.text(0.005, 0.002, watermark + (f"  |  {note}" if note else ""), fontsize=7, color=INK2, ha="left", va="bottom")
    if lang == "he":   # separate line, right-aligned: mixing the two scripts in one line scrambles the bidi order
        fig.text(0.995, 0.002, rtl("תוצאה חקרנית: נפילה תוך 180 יום – לא שחזור של eFalls"), fontsize=7, color=INK2, ha="right", va="bottom")
    out_dir.mkdir(parents=True, exist_ok=True)
    png, svg = out_dir / f"{stem}.png", out_dir / f"{stem}.svg"
    fig.savefig(png, dpi=DPI, format="png", metadata={"Software": None}, bbox_inches="tight", facecolor=SURFACE)
    fig.savefig(svg, format="svg", metadata={"Date": None}, bbox_inches="tight", facecolor=SURFACE)
    return {"png": png.name, "svg": svg.name}


# ============================================================================ cohort
def funnel(steps: list[tuple[str, int, str]], *, title: str, lang: str) -> Figure:
    """steps: (label, rows, annotation)."""
    fig = _fig(10, 0.55 * len(steps) + 1.3)
    ax = fig.add_subplot(111)
    _style(ax)
    vals = [s[1] for s in steps]
    ax.barh(range(len(steps)), vals, color=[BLUE] * (len(steps) - 1) + [ORANGE], height=0.66)
    for i, (_lab, v, ann) in enumerate(steps):
        ax.text(v, i, f"  {v:,}" + (f"  ({ann})" if ann and lang == "en" else ""), va="center", fontsize=9, color=INK)
    # Hebrew annotations go into the row label: a number followed by Hebrew in one text object reorders unpredictably
    ax.set_yticks(range(len(steps)), [T(s[0] + (f" ({s[2]})" if s[2] and lang == "he" else ""), lang) for s in steps], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, max(vals) * 1.45 if vals else 1)
    ax.set_title(T(title, lang), fontsize=12, loc="right" if lang == "he" else "left", color=INK)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def partitions(parts: dict[str, dict[str, Any]], *, lang: str) -> Figure:
    names = list(parts)
    fig = _fig(8, 3.2)
    ax = fig.add_subplot(111)
    _style(ax)
    ev = np.array([parts[p]["n_events"] for p in names], dtype=float)
    n = np.array([parts[p]["n_rows"] for p in names], dtype=float)
    ax.barh(names, n - ev, color=BLUE, label=T("no fall within 180 days" if lang == "en" else "ללא נפילה ב-180 יום", lang), height=0.6)
    ax.barh(names, ev, left=n - ev, color=ORANGE, label=T("fall within 180 days" if lang == "en" else "נפילה ב-180 יום", lang), height=0.6)
    for i, p in enumerate(names):
        ax.text(n[i], i, f"  {int(n[i]):,} rows, {int(ev[i]):,} events ({100 * ev[i] / max(n[i], 1):.1f}%)", va="center", fontsize=8.5, color=INK)
    ax.set_xlim(0, n.max() * 1.55)
    ax.invert_yaxis()
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("Train / validation / test partitions (patient-grouped, stratified)", fontsize=11, loc="left")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


# ============================================================================ LASSO
def cv_curve(cv: pd.DataFrame, grid_info: dict[str, Any]) -> Figure:
    fig = _fig(9, 4.8)
    ax = fig.add_subplot(111)
    _style(ax)
    lam, m, se = cv["lambda"].to_numpy(), cv["cv_mean_deviance"].to_numpy(), cv["cv_se"].to_numpy()
    ax.fill_between(lam, m - se, m + se, color=BLUE, alpha=0.15, lw=0, label="± 1 SE across the 10 folds")
    ax.plot(lam, m, color=BLUE, lw=2, label="mean held-out deviance")
    ax.set_xscale("log")
    k = int(np.flatnonzero(cv["selected"].astype(bool).to_numpy())[0])
    kmin = int(np.nanargmin(m))
    ax.axvline(lam[k], color=ORANGE, lw=2, label=f"selected lambda* = {lam[k]:.3g} ({int(cv['n_nonzero'].iloc[k])} non-zero)")
    ax.scatter([lam[kmin]], [m[kmin]], s=60, color=INK, zorder=4, label=f"minimum of the mean curve (index {kmin})")
    top = ax.twiny()
    top.set_xscale("log")
    top.set_xlim(ax.get_xlim())
    ticks = np.unique(np.linspace(0, len(lam) - 1, 8).astype(int))
    top.set_xticks(lam[ticks], [str(int(v)) for v in cv["n_nonzero"].to_numpy()[ticks]], fontsize=8, color=INK2)
    top.set_xlabel("number of non-zero coefficients", fontsize=8, color=INK2)
    ax.invert_xaxis()
    top.invert_xaxis()
    ax.set_xlabel("lambda (log scale; strong penalty on the left, weaker to the right - same direction as the coefficient-path figure)")
    ax.set_ylabel("cross-validated mean deviance (lower is better)")
    notes = []
    if grid_info.get("cv_minimum_identified") is False:
        notes.append("CV MINIMUM NOT IDENTIFIED: the curve is flat around lambda*; nearby lambdas are nearly equivalent (D-14 argmin used)")
    if grid_info.get("at_grid_end"):
        notes.append("lambda* IS ON THE GRID BOUNDARY: the true minimum may lie beyond the grid")
    elif grid_info.get("near_grid_end"):
        notes.append(f"lambda* is {grid_info.get('grid_points_to_end')} grid points from the weak-penalty end of the grid")
    if notes:
        ax.text(0.01, 0.98, "\n".join(notes), transform=ax.transAxes, va="top", fontsize=8.5, color="#8a2c0b",
                bbox={"boxstyle": "round", "fc": "#fff3cd", "ec": "#d39e00"})
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("LASSO 10-fold cross-validation curve", fontsize=11, loc="left", pad=28)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def coef_paths(path: pd.DataFrame, lam_star: float, labels: dict[str, str]) -> Figure:
    fig = _fig(10, 5.6)
    ax = fig.add_subplot(111)
    _style(ax)
    wide = path.pivot(index="lambda", columns="design_column", values="coef_standardized").sort_index()
    at = wide.loc[wide.index[np.argmin(np.abs(wide.index - lam_star))]]
    ranked = at.abs().sort_values(ascending=False)
    colored = [c for c in ranked.index if ranked[c] > 0][:8]
    for c in wide.columns:
        if c not in colored:
            ax.plot(wide.index, wide[c], color=MUTED, lw=1)
    ends = {c: float(wide[c].iloc[0]) for c in colored}
    span = (max(ends.values()) - min(ends.values())) if ends else 1.0
    gap = max(span, float(np.nanmax(np.abs(wide.to_numpy()))) or 1.0) * 0.045
    placed: dict[str, float] = {}
    for c in sorted(ends, key=ends.get):          # spread the direct labels so they never overlap
        y = ends[c]
        if placed and y - max(placed.values()) < gap:
            y = max(placed.values()) + gap
        placed[c] = y
    for i, c in enumerate(colored):
        ax.plot(wide.index, wide[c], color=SERIES[i], lw=2)
        ax.text(wide.index.min(), placed[c], "  " + labels.get(c, c), fontsize=7.5, color=INK, va="center")
    ax.axvline(lam_star, color=INK, lw=1.2, ls="--")
    ax.text(lam_star, ax.get_ylim()[1], " lambda*", fontsize=8, color=INK, va="top")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("lambda (log scale; decreasing penalty to the right, so predictors enter from left to right)")
    ax.set_ylabel("coefficient on the standardised scale (per 1 SD)")
    ax.set_title(f"LASSO coefficient paths on TRAIN (coloured: the {len(colored)} largest at lambda*; grey: others)", fontsize=11, loc="left")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def fp_panel(effects: dict[str, pd.DataFrame], forms: dict[str, dict[str, Any]], labels: dict[str, str]) -> Figure | None:
    if not effects:
        return None
    fig = _fig(5.2 * len(effects), 4.2)
    axs = fig.subplots(1, len(effects), squeeze=False)[0]
    for ax, (var, eff) in zip(axs, effects.items()):
        _style(ax)
        ax.plot(eff[var], eff["log_odds_vs_median"], color=BLUE, lw=2.2)
        ax.axhline(0, color=INK2, lw=0.8)
        f = forms.get(var, {})
        stab = f.get("share_with_final_form")
        ax.set_title(f"{labels.get(var, var)}\nchosen form {f.get('final_form') or f.get('powers', '?')}"
                     + (f"; same form in {100 * stab:.0f}% of {f.get('replicates')} bootstrap refits" if stab is not None else ""), fontsize=9, loc="left")
        ax.set_xlabel(f"{var} (TRAIN p1-p99)")
        ax.set_ylabel("change in log-odds vs the median value")
    fig.suptitle("Fitted shape of continuous predictors (fractional-polynomial choice; association, not cause)", fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    return fig


def learning(lc: pd.DataFrame, info: dict[str, Any]) -> Figure:
    d = lc[lc["status"] == "fitted"]
    fig = _fig(11, 4.2)
    a1, a2 = fig.subplots(1, 2)
    for ax in (a1, a2):
        _style(ax)
    a1.plot(d["n_train"], d["train_auroc_apparent"], marker="o", color=ORANGE, lw=2, label="training subset (apparent, optimistic)")
    a1.errorbar(d["n_train"], d["validation_auroc"], yerr=[d["validation_auroc"] - d["validation_auroc_ci_low"], d["validation_auroc_ci_high"] - d["validation_auroc"]],
                marker="o", color=BLUE, lw=2, capsize=3, label="validation (95% bootstrap CI)")
    a1.set_xlabel("training rows used")
    a1.set_ylabel("AUROC")
    a1.legend(frameon=False, fontsize=8)
    a1.set_title("Discrimination vs training size", fontsize=10, loc="left")
    a2.plot(d["n_train"], d["train_brier_apparent"], marker="o", color=ORANGE, lw=2, label="training subset (apparent)")
    a2.plot(d["n_train"], d["validation_brier"], marker="o", color=BLUE, lw=2, label="validation")
    a2.set_xlabel("training rows used")
    a2.set_ylabel("Brier score (lower is better)")
    a2.legend(frameon=False, fontsize=8)
    a2.set_title("Accuracy of predicted risks vs training size", fontsize=10, loc="left")
    fig.suptitle("Learning curve (nested TRAIN subsets, evaluated on VALIDATION; the test set is not used)"
                 + (" — REDUCED-COMPUTATION MODE" if info.get("mode") == "reduced" else ""), fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    return fig


# ============================================================================ discrimination / calibration
def roc(curves: list[tuple[str, pd.DataFrame, str]], *, lang: str, title: str) -> Figure:
    """curves: (label incl. AUROC / CI, points, colour)."""
    fig = _fig(6.2, 5.6)
    ax = fig.add_subplot(111)
    _style(ax)
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--", label=T("no discrimination (chance)" if lang == "en" else "ללא יכולת הבחנה (מקרי)", lang))
    for lab, pts, col in curves:
        ax.plot(pts["fpr"], pts["tpr"], color=col, lw=2.2, label=T(lab, lang))
    ax.set_xlabel(T("1 - specificity (share of non-fallers flagged)" if lang == "en" else "שיעור מי שלא נפלו וסומנו כבסיכון", lang))
    ax.set_ylabel(T("sensitivity (share of fallers flagged)" if lang == "en" else "שיעור מי שנפלו וסומנו כבסיכון", lang))
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title(T(title, lang), fontsize=11, loc="right" if lang == "he" else "left")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def pr(curves: list[tuple[str, pd.DataFrame, str]], prevalence: float, *, title: str) -> Figure:
    fig = _fig(6.2, 5.6)
    ax = fig.add_subplot(111)
    _style(ax)
    ax.axhline(prevalence, color=MUTED, lw=1.2, ls="--", label=f"no-skill reference = event prevalence {100 * prevalence:.1f}%")
    for lab, pts, col in curves:
        ax.plot(pts["recall"], pts["precision"], color=col, lw=2.2, label=lab)
    ax.set_xlabel("recall = sensitivity (share of fallers flagged)")
    ax.set_ylabel("precision = PPV (share of flagged who fell)")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_title(title, fontsize=11, loc="left")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def risk_distribution(hist: pd.DataFrame) -> Figure:
    fig = _fig(9, 4.2)
    ax = fig.add_subplot(111)
    _style(ax)
    for cls, col, lab in ((0, BLUE, "no fall within 180 days"), (1, ORANGE, "fall within 180 days")):
        d = hist[hist["outcome"] == cls]
        ax.bar(d["bin_low"], 100 * d["share"], width=(d["bin_high"] - d["bin_low"]) * 0.95, align="edge", color=col, alpha=0.55, label=lab)
    ax.set_xlabel("predicted 180-day risk")
    ax.set_ylabel("% of the group")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Predicted-risk distributions of future non-fallers and fallers (HELD-OUT TEST; the overlap is what the model cannot separate)",
                 fontsize=10, loc="left")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def calibration(cal: pd.DataFrame, stats: dict[str, str], p_hist: pd.DataFrame) -> Figure:
    fig = _fig(7.2, 7.6)
    gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 1])
    ax = fig.add_subplot(gs[0])
    _style(ax)
    hi = 0.0
    for variant, col in (("uncalibrated", BLUE), ("recalibrated", ORANGE)):
        d = cal[(cal["variant"] == variant) & (cal["kind"] == "grouped")]
        if d.empty:
            continue
        hi = max(hi, float(d["mean_predicted"].max()), float(d["observed_rate"].max()))
        ax.errorbar(d["mean_predicted"], d["observed_rate"], yerr=[d["observed_rate"] - d["ci_low"], d["ci_high"] - d["observed_rate"]], fmt="o",
                    color=col, ms=5, capsize=2, lw=1.2, label=variant)
    hi = hi * 1.1 or 1.0
    ax.plot([0, hi], [0, hi], color=INK2, lw=1, ls="--", label="ideal (predicted = observed)")
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_xlabel("mean predicted risk in the group")
    ax.set_ylabel("observed fall rate (95% CI)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.text(0.98, 0.03, "\n".join(f"{k}: {v}" for k, v in stats.items()).replace(", ", "\n    "), transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7.5, color=INK, bbox={"boxstyle": "round", "fc": "white", "ec": GRID})
    ax.set_title("Calibration on the HELD-OUT TEST partition (groups of predicted risk)", fontsize=10.5, loc="left")
    ax2 = fig.add_subplot(gs[1])
    _style(ax2)
    tot = p_hist.groupby("bin_low", as_index=False)["n"].sum()
    width = float(p_hist["bin_high"].iloc[0] - p_hist["bin_low"].iloc[0]) if len(p_hist) else 0.01
    ax2.bar(tot["bin_low"], tot["n"], width=width * 0.95, align="edge", color=MUTED)
    ax2.set_xlim(0, hi)
    ax2.set_xlabel("predicted risk (distribution; bins below min_cell hidden)")
    ax2.set_ylabel("rows")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def thresholds_plot(tt: pd.DataFrame) -> Figure:
    fig = _fig(10, 4.4)
    a1, a2 = fig.subplots(1, 2)
    for ax in (a1, a2):
        _style(ax)
    for col, c, lab in (("sensitivity", ORANGE, "sensitivity"), ("specificity", BLUE, "specificity"), ("ppv", AQUA, "PPV"), ("npv", VIOLET, "NPV")):
        a1.plot(tt["threshold"], pd.to_numeric(tt[col], errors="coerce"), color=c, lw=2, label=lab)
    a1.set_xlabel("risk threshold (no threshold is recommended)")
    a1.set_ylim(0, 1.02)
    a1.legend(frameon=False, fontsize=8)
    a1.set_title("Operating characteristics by threshold", fontsize=10, loc="left")
    a2.plot(tt["threshold"], pd.to_numeric(tt["pct_flagged"], errors="coerce"), color=BLUE, lw=2, label="% of population flagged")
    a2.plot(tt["threshold"], pd.to_numeric(tt["pct_falls_captured"], errors="coerce"), color=ORANGE, lw=2, label="% of future falls captured")
    a2.set_xlabel("risk threshold")
    a2.set_ylabel("%")
    a2.legend(frameon=False, fontsize=8)
    a2.set_title("Workload vs falls captured", fontsize=10, loc="left")
    fig.suptitle("HELD-OUT TEST, model frozen; exploratory - thresholds shown, none chosen", fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    return fig


def gains_lift(gains: pd.DataFrame, lift: pd.DataFrame, *, lang: str) -> Figure:
    fig = _fig(11, 4.6)
    a1, a2 = fig.subplots(1, 2)
    for ax in (a1, a2):
        _style(ax)
    a1.plot(gains["pct_population"], gains["pct_falls_captured"], color=ORANGE, lw=2.4, label=T("model ranking" if lang == "en" else "דירוג לפי המודל", lang))
    a1.plot([0, 100], [0, 100], color=MUTED, ls="--", lw=1.2, label=T("random selection" if lang == "en" else "בחירה אקראית", lang))
    a1.set_xlabel(T("% of patients selected, highest predicted risk first" if lang == "en" else "אחוז המטופלים שנבחרו (מהסיכון הגבוה)", lang))
    a1.set_ylabel(T("% of all future falls in the selected group" if lang == "en" else "אחוז מכלל הנפילות בקבוצה שנבחרה", lang))
    a1.legend(frameon=False, fontsize=8, loc="lower right")
    a1.set_title(T("Cumulative gains" if lang == "en" else "כמה מהנפילות נמצאות בקבוצת הסיכון העליונה", lang), fontsize=10, loc="right" if lang == "he" else "left")
    d = lift.dropna(subset=["pct_of_all_falls_captured"])
    x = np.arange(len(d))
    a2.bar(x - 0.2, d["top_pct"], width=0.38, color=MUTED, label=T("% of population selected" if lang == "en" else "אחוז מהאוכלוסייה", lang))
    cap = d["pct_of_all_falls_captured"].astype(float)
    err = [cap - d["capture_ci_low"].astype(float), d["capture_ci_high"].astype(float) - cap]
    a2.bar(x + 0.2, cap, width=0.38, color=ORANGE, yerr=err, capsize=3, label=T("% of all future falls captured (95% CI)" if lang == "en" else "אחוז מכלל הנפילות (רווח סמך 95%)", lang))
    for i, (c, l, top) in enumerate(zip(cap, d["lift"], d["capture_ci_high"].astype(float))):
        a2.text(x[i] + 0.2, (top if np.isfinite(top) else c) + 1.0, f"{c:.0f}%\n×{float(l):.1f}", ha="center", va="bottom", fontsize=8, color=INK)
    a2.set_xticks(x, [T(f"top {v:g}%" if lang == "en" else f"{v:g}% העליונים", lang) for v in d["top_pct"]])
    a2.legend(frameon=False, fontsize=8, loc="upper left")
    a2.set_title(T("Risk concentration (lift = × population rate)" if lang == "en" else "ריכוז הסיכון (× פי כמה מהשיעור באוכלוסייה)", lang), fontsize=10,
                 loc="right" if lang == "he" else "left")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def deciles(dec: pd.DataFrame, *, lang: str) -> Figure:
    fig = _fig(9, 4.4)
    ax = fig.add_subplot(111)
    _style(ax)
    obs = 100 * pd.to_numeric(dec["observed_rate"], errors="coerce")
    pred = 100 * pd.to_numeric(dec["mean_predicted"], errors="coerce")
    ax.bar(dec["risk_group"], obs, color=ORANGE, width=0.7, label=T("observed fall rate" if lang == "en" else "שיעור נפילות בפועל", lang))
    ax.plot(dec["risk_group"], pred, color=INK, marker="o", lw=1.6, label=T("average predicted risk" if lang == "en" else "סיכון חזוי ממוצע", lang))
    for g, v in zip(dec["risk_group"], obs):
        if np.isfinite(v):
            ax.text(g, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(range(1, 11), [T(s, lang) for s in (["1\nlowest"] + [str(i) for i in range(2, 10)] + ["10\nhighest"] if lang == "en"
                                                     else ["1\nהנמוך"] + [str(i) for i in range(2, 10)] + ["10\nהגבוה"])])
    ax.set_xlabel(T("tenth of predicted risk" if lang == "en" else "עשירון הסיכון החזוי", lang))
    ax.set_ylabel(T("% who fell within 180 days" if lang == "en" else "אחוז שנפלו תוך 180 יום", lang))
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_title(T("Fall rate by tenth of predicted risk (held-out test)" if lang == "en" else "שיעור הנפילות לפי עשירוני סיכון (קבוצת הבדיקה)", lang), fontsize=11,
                 loc="right" if lang == "he" else "left")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


# ============================================================================ features
def hbars(names: list[str], values: list[float], *, title: str, xlabel: str, lang: str, colors: list[str] | None = None,
          err: list[float] | None = None, ref: float | None = None, fmt: str = "{:.2f}") -> Figure:
    fig = _fig(9, 0.38 * len(names) + 1.6)
    ax = fig.add_subplot(111)
    _style(ax)
    y = np.arange(len(names))
    ax.barh(y, values, color=colors or BLUE, height=0.65, xerr=err, capsize=2 if err else 0, error_kw={"ecolor": INK2, "elinewidth": 1})
    for i, v in enumerate(values):
        e = abs(float(err[i])) if err else 0.0   # the label sits beyond the error bar, never on it
        ax.text(v + e if v >= 0 else v - e, i, " " + fmt.format(v) if v >= 0 else fmt.format(v) + " ", va="center", ha="left" if v >= 0 else "right",
                fontsize=8, color=INK)
    if ref is not None:
        ax.axvline(ref, color=INK2, lw=1, ls="--")
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(y, [T(n, lang) for n in names], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(T(xlabel, lang))
    ax.set_title(T(title, lang), fontsize=11, loc="right" if lang == "he" else "left")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def subgroup_plot(sg: pd.DataFrame) -> Figure | None:
    d = sg.dropna(subset=["auroc"]) if "auroc" in sg.columns else pd.DataFrame()
    if d.empty:
        return None
    fig = _fig(9, 0.34 * len(d) + 1.8)
    ax = fig.add_subplot(111)
    _style(ax)
    y = np.arange(len(d))
    a = d["auroc"].astype(float)
    ax.errorbar(a, y, xerr=[a - d["auroc_ci_low"].astype(float), d["auroc_ci_high"].astype(float) - a], fmt="o", color=BLUE, capsize=2)
    ax.axvline(0.5, color=MUTED, ls="--")
    ax.set_yticks(y, [f"{s}: {lv} (n {n}, events {e})" for s, lv, n, e in zip(d["subgroup"], d["level"], d["n"], d["events"])], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("AUROC (Hanley-McNeil 95% CI) - exploratory; a computed metric does not establish subgroup validity or fairness")
    ax.set_title("Performance by subgroup (HELD-OUT TEST; groups with < 10 events or non-events not shown)", fontsize=10, loc="left")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def comparison(table: pd.DataFrame, *, lang: str) -> Figure:
    """Dot + CI panels per metric; rows = analyses; unpaired rows marked."""
    metrics = [("auroc", "AUROC"), ("pr_auc", "PR-AUC"), ("brier", "Brier"), ("calibration_slope", "calibration slope"), ("oe_ratio", "O:E")]
    fig = _fig(15, 0.5 * len(table) + 2.2)
    axs = fig.subplots(1, len(metrics), sharey=True)
    y = np.arange(len(table))
    for ax, (m, lab) in zip(axs, metrics):
        _style(ax)
        est = pd.to_numeric(table[f"{m}"], errors="coerce")
        lo = pd.to_numeric(table[f"{m}_ci_low"], errors="coerce")
        hi = pd.to_numeric(table[f"{m}_ci_high"], errors="coerce")
        colors = [BLUE if p else ORANGE for p in table["paired_with_reference"]]
        for i in range(len(table)):
            if np.isfinite(est.iloc[i]):
                ax.errorbar(est.iloc[i], y[i], xerr=[[est.iloc[i] - lo.iloc[i]], [hi.iloc[i] - est.iloc[i]]] if np.isfinite(lo.iloc[i]) else None,
                            fmt="o", color=colors[i], capsize=2)
        if m in ("calibration_slope", "oe_ratio"):
            ax.axvline(1.0, color=MUTED, ls="--")
        ax.set_title(lab, fontsize=10)
    axs[0].set_yticks(y, [T(s, lang) for s in table["label"]], fontsize=8.5)
    axs[0].invert_yaxis()
    fig.suptitle("Side-by-side comparison on each analysis' own held-out test partition. Blue = same test rows as the reference (paired); "
                 "orange = different population: NOT A DIRECT PAIRED MODEL COMPARISON", fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 0.9))
    return fig


def roadmap(steps: list[str], current: int, *, lang: str) -> Figure:
    """Vertical timeline (one box per step, top to bottom): robust to long labels in any language."""
    n = len(steps)
    fig = _fig(7.5, 0.62 * n + 0.8)
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(n, -0.4)
    for i, step in enumerate(steps):
        col = ORANGE if i == current else (BLUE if i < current else "#e8e7e3")
        ax.text(0.5, i + 0.3, T(f"{i + 1}. {step}" if lang == "en" else f"{step} .{i + 1}", lang), ha="center", va="center", fontsize=10,
                color="white" if i <= current else INK, bbox={"boxstyle": "round,pad=0.45", "fc": col, "ec": "none"})
        if i < n - 1:
            ax.annotate("", xy=(0.5, i + 1.02), xytext=(0.5, i + 0.62), arrowprops={"arrowstyle": "->", "color": INK2, "lw": 1.2})
    note = "current position" if lang == "en" else "המיקום הנוכחי"
    ax.text(0.97, current + 0.3, T(note, lang), ha="right", va="center", fontsize=8.5, color=ORANGE)
    return fig
