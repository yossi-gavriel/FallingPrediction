"""EDA figures (PNG, Agg canvas, no global pyplot state). Every figure is aggregate and carries the watermark and its analysis basis.

Privacy: no scatter of individual rows; histograms hide bins with 1..min_cell-1 rows; distribution curves are drawn from percentiles
p1..p99 (never from individual values); box plots draw p25-p75 boxes with p5-p95 whiskers and no outlier points.
Colours (reference data-viz palette): blue / orange / aqua for up to three series, a blue-grey-red diverging scale for correlations,
status colours only for data-quality severity (always labelled).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

if "matplotlib.pyplot" not in sys.modules:
    matplotlib.use("Agg")

from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

DPI = 100
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
STATUS = {"CRITICAL": "#d03b3b", "WARNING": "#fab219", "INFO": "#86b6ef"}
DIVERGING = LinearSegmentedColormap.from_list("eda_div", ["#d03b3b", "#f0efec", "#2a78d6"])
SEQUENTIAL = LinearSegmentedColormap.from_list("eda_seq", ["#f0efec", "#86b6ef", "#2a78d6", "#104281"])


def _fig(w: float, h: float) -> Figure:
    fig = Figure(figsize=(w, h), facecolor=SURFACE)
    FigureCanvasAgg(fig)
    return fig


def _style(ax: Any) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6, axis="both")
    ax.set_axisbelow(True)
    ax.title.set_color(INK)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def _save(fig: Figure, path: Path, watermark: str, basis: str, dpi: int = DPI) -> Path:
    fig.text(0.01, 0.003, f"{watermark}  |  basis: {basis}  |  aggregate; small cells hidden", fontsize=7, color=INK2, ha="left", va="bottom")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, format="png", metadata={"Software": None}, bbox_inches="tight", facecolor=SURFACE)
    return path


def _pctl(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.linspace(0.01, 0.99, 99)
    return np.quantile(x, p), p


def _hist(ax: Any, x: np.ndarray, min_cell: int, color: str) -> int:
    """Histogram with bins of 1..min_cell-1 rows hidden; returns the number of hidden bins."""
    if x.size == 0:
        return 0
    uniq = np.unique(x)
    if uniq.size <= 40 and np.allclose(uniq, np.round(uniq)):
        edges = np.arange(uniq.min() - 0.5, uniq.max() + 1.5, 1.0)
    else:
        lo, hi = np.quantile(x, [0.0, 1.0])
        edges = np.linspace(lo, hi if hi > lo else lo + 1, 41)
    counts, edges = np.histogram(x, bins=edges)
    hide = (counts > 0) & (counts < min_cell)
    shown = np.where(hide, 0, counts)
    ax.bar(edges[:-1], shown, width=np.diff(edges) * 0.9, align="edge", color=color, edgecolor=SURFACE, linewidth=0.5)
    return int(hide.sum())


def numeric_panel(path: Path, name: str, full: np.ndarray, cohort: np.ndarray, train_fall: np.ndarray | None, train_no: np.ndarray | None, *,
                  watermark: str, min_cell: int, subtitle: str = "") -> Path:
    """Histogram (full extract), percentile curves (full vs modelling cohort), boxes (p5-p95 whiskers), TRAIN outcome-stratified curves."""
    fig = _fig(11, 6.6)
    axs = fig.subplots(2, 2)
    fig.suptitle(f"{name}" + (f" — {subtitle}" if subtitle else ""), fontsize=11, color=INK, x=0.01, ha="left")
    ax = axs[0, 0]
    _style(ax)
    hidden = _hist(ax, full, min_cell, SERIES[0])
    ax.set_title(f"histogram, full extract (n observed {full.size}; bins < {min_cell} rows hidden: {hidden})", fontsize=9, loc="left")
    ax.set_ylabel("rows")
    ax = axs[0, 1]
    _style(ax)
    for x, lab, col in ((full, "full extract", SERIES[0]), (cohort, "modelling cohort", SERIES[1])):
        if x.size >= min_cell:
            q, p = _pctl(x)
            ax.plot(q, p, color=col, linewidth=2, label=lab, drawstyle="steps-post")
    ax.set_title("distribution curve from percentiles p1..p99", fontsize=9, loc="left")
    ax.set_ylabel("cumulative share")
    ax.legend(fontsize=8, frameon=False)
    ax = axs[1, 0]
    _style(ax)
    stats = []
    for x, lab in ((full, "full extract"), (cohort, "modelling cohort")):
        if x.size >= min_cell:
            p5, q1, med, q3, p95 = np.quantile(x, [0.05, 0.25, 0.5, 0.75, 0.95])
            stats.append({"label": lab, "whislo": p5, "q1": q1, "med": med, "q3": q3, "whishi": p95, "fliers": []})
    if stats:
        try:   # matplotlib >= 3.10: orientation; older: vert
            b = ax.bxp(stats, orientation="horizontal", showfliers=False, patch_artist=True, widths=0.5)
        except TypeError:
            b = ax.bxp(stats, vert=False, showfliers=False, patch_artist=True, widths=0.5)
        for patch, col in zip(b["boxes"], SERIES):
            patch.set_facecolor(col)
            patch.set_alpha(0.55)
            patch.set_edgecolor(col)
        for med in b["medians"]:
            med.set_color(INK)
    ax.set_title("box = p25-p75, line = median, whiskers = p5-p95 (no individual points)", fontsize=9, loc="left")
    ax = axs[1, 1]
    _style(ax)
    if train_fall is not None and train_no is not None and train_fall.size >= min_cell and train_no.size >= min_cell:
        for x, lab, col in ((train_no, "no fall within 180 days", SERIES[0]), (train_fall, "fall within 180 days", SERIES[1])):
            q, p = _pctl(x)
            ax.plot(q, p, color=col, linewidth=2, label=f"{lab} (n {x.size})", drawstyle="steps-post")
        ax.legend(fontsize=8, frameon=False)
        ax.set_title("TRAIN ONLY: by outcome (descriptive, univariate)", fontsize=9, loc="left")
    else:
        ax.text(0.5, 0.5, "TRAIN outcome view not available\n(split not built or too few observed values)", ha="center", va="center", color=INK2, fontsize=9,
                transform=ax.transAxes)
        ax.set_title("TRAIN ONLY: by outcome", fontsize=9, loc="left")
    ax.set_ylabel("cumulative share")
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    return _save(fig, path, watermark, "FULL_EXTRACT / MODELLING_COHORT (unsupervised) + TRAIN_ONLY (outcome panel)", dpi=72)


def categorical_panel(path: Path, name: str, levels: pd.DataFrame, train: pd.DataFrame | None, *, watermark: str) -> Path:
    """Left: level % in the modelling cohort (NULL = its own level). Right: TRAIN prevalence per level with its 95% CI."""
    fig = _fig(11, max(3.0, 0.32 * len(levels) + 1.6))
    axs = fig.subplots(1, 2)
    fig.suptitle(name, fontsize=11, color=INK, x=0.01, ha="left")
    ax = axs[0]
    _style(ax)
    lv = levels.copy()
    lv["pct_num"] = pd.to_numeric(lv["pct"], errors="coerce").fillna(0.0)
    ax.barh(range(len(lv)), lv["pct_num"], color=SERIES[0], height=0.7)
    ax.set_yticks(range(len(lv)), [str(v) for v in lv["level"]], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("% of modelling cohort (suppressed levels drawn as 0)")
    ax.set_title("level distribution (unsupervised)", fontsize=9, loc="left")
    ax = axs[1]
    _style(ax)
    if train is not None and len(train):
        t = train.copy()
        t["p"] = pd.to_numeric(t["prevalence"], errors="coerce")
        t["lo"] = pd.to_numeric(t["prevalence_ci_low"], errors="coerce")
        t["hi"] = pd.to_numeric(t["prevalence_ci_high"], errors="coerce")
        order = {str(v): i for i, v in enumerate(lv["level"])}
        t["y"] = t["level"].map(lambda v: order.get(str(v), np.nan))
        t = t.dropna(subset=["y", "p"])
        ax.errorbar(100 * t["p"], t["y"], xerr=[100 * (t["p"] - t["lo"]), 100 * (t["hi"] - t["p"])], fmt="o", color=SERIES[1], ms=6, capsize=2, lw=1.5)
        ax.set_yticks(range(len(lv)), [str(v) for v in lv["level"]], fontsize=8)
        ax.set_ylim(len(lv) - 0.5, -0.5)
        ax.set_xlabel("% with a fall within 180 days (Wilson 95% CI)")
        ax.set_title("TRAIN ONLY: outcome prevalence per level (descriptive)", fontsize=9, loc="left")
    else:
        ax.text(0.5, 0.5, "TRAIN view not available", ha="center", va="center", color=INK2, transform=ax.transAxes)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    return _save(fig, path, watermark, "MODELLING_COHORT (left) + TRAIN_ONLY (right)", dpi=80)


def cohort_funnel(path: Path, flow: pd.DataFrame, *, watermark: str) -> Path:
    f = flow[flow["branch"] == "PRIMARY"].reset_index(drop=True)
    safe = flow[flow["branch"] != "PRIMARY"]
    labels = [str(s).strip() for s in f["stage"]] + [str(s) for s in safe["stage"]]
    rows = list(f["rows"]) + list(safe["rows"])
    cols = [SERIES[0]] * len(f) + [SERIES[1]] * len(safe)
    fig = _fig(11, 0.42 * len(labels) + 1.4)
    ax = fig.add_subplot(111)
    _style(ax)
    ax.barh(range(len(labels)), rows, color=cols, height=0.68)
    for i, (r, ev) in enumerate(zip(rows, list(f["events"]) + list(safe["events"]))):
        evs = "" if ev is None or ev == "" or (isinstance(ev, float) and np.isnan(ev)) else (f"{int(ev):,}" if isinstance(ev, (int, float, np.integer)) else str(ev))
        ax.text(r, i, f"  {r:,} rows" + (f", {evs} events" if evs else ""), va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(labels)), [textwrap(s, 70) for s in labels], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, max(rows) * 1.35 if rows else 1)
    ax.set_xlabel("rows")
    ax.set_title("Cohort funnel (blue: primary modelling cohort; orange: SAFE-ALL-ROWS sensitivity branch)", fontsize=10, loc="left")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return _save(fig, path, watermark, "FULL_EXTRACT -> MODELLING_COHORT")


def textwrap(s: str, width: int) -> str:
    import textwrap as tw

    return "\n".join(tw.wrap(s, width)) or s


def grouped_distribution(path: Path, prof: pd.DataFrame, section: str, variable: str, title: str, *, watermark: str) -> Path | None:
    d = prof[(prof["section"] == section) & (prof["variable"] == variable)].copy()
    if d.empty:
        return None
    d["pct_num"] = pd.to_numeric(d["pct"], errors="coerce").fillna(0.0)
    bases = list(dict.fromkeys(d["basis"]))
    levels = list(dict.fromkeys(d["level"]))
    fig = _fig(10, 3.8)
    ax = fig.add_subplot(111)
    _style(ax)
    w = 0.8 / max(len(bases), 1)
    for i, b in enumerate(bases):
        sub = d[d["basis"] == b].set_index("level")
        ax.bar(np.arange(len(levels)) + i * w, [sub["pct_num"].get(lv, 0.0) for lv in levels], width=w * 0.92, color=SERIES[i % len(SERIES)], label=b)
    ax.set_xticks(np.arange(len(levels)) + w * (len(bases) - 1) / 2, levels, fontsize=8)
    ax.set_ylabel("% of rows")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title(title, fontsize=10, loc="left")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return _save(fig, path, watermark, ", ".join(bases))


def prevalence_forest(path: Path, prev: pd.DataFrame, overall: float | None, *, watermark: str, title: str, basis: str) -> Path | None:
    d = prev.copy()
    d["p"] = pd.to_numeric(d["prevalence"], errors="coerce")
    d = d.dropna(subset=["p"])
    if d.empty:
        return None
    d["lo"] = pd.to_numeric(d["ci_low"], errors="coerce")
    d["hi"] = pd.to_numeric(d["ci_high"], errors="coerce")
    labels = [f"{s}: {lv}" for s, lv in zip(d["stratifier"], d["level"])]
    fig = _fig(11, 0.24 * len(d) + 1.6)
    ax = fig.add_subplot(111)
    _style(ax)
    y = np.arange(len(d))
    ax.errorbar(100 * d["p"], y, xerr=[100 * (d["p"] - d["lo"]), 100 * (d["hi"] - d["p"])], fmt="o", color=SERIES[0], ms=5, capsize=2, lw=1.2)
    if overall is not None:
        ax.axvline(100 * overall, color=SERIES[1], lw=1.5, ls="--", label=f"overall {100 * overall:.2f}%")
        ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.set_yticks(y, [textwrap(s, 80) for s in labels], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("% with a fall within 180 days (Wilson 95% CI)")
    ax.set_title(title, fontsize=10, loc="left")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return _save(fig, path, watermark, basis)


def missing_by_column(path: Path, miss: pd.DataFrame, classes: tuple[str, ...], *, watermark: str, top: int = 80) -> Path | None:
    d = miss.copy()
    d["pct"] = pd.to_numeric(d["missing_or_sentinel_pct"], errors="coerce").fillna(0.0)
    d = d[d["pct"] > 0].sort_values("pct", ascending=False).head(top)
    if d.empty:
        return None
    counts = pd.DataFrame({k: pd.to_numeric(d[f"n_{k.lower()}"], errors="coerce").fillna(0.0) for k in classes})
    tot = counts.sum(axis=1).replace(0, np.nan)
    shares = counts.div(tot, axis=0).fillna(0.0).mul(d["pct"].to_numpy(), axis=0)
    fig = _fig(11, 0.19 * len(d) + 1.8)
    ax = fig.add_subplot(111)
    _style(ax)
    left = np.zeros(len(d))
    for i, k in enumerate(classes):
        ax.barh(range(len(d)), shares[k], left=left, color=SERIES[i % len(SERIES)], height=0.75, label=k.replace("_", " ").lower(),
                edgecolor=SURFACE, linewidth=0.4)
        left += shares[k].to_numpy()
    ax.set_yticks(range(len(d)), list(d["column"]), fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of rows missing (or declared sentinel), split by missingness class")
    ax.legend(fontsize=7, frameon=False, ncol=4, loc="lower right")
    ax.set_title(f"Missingness by column, full extract (top {len(d)} columns with any missing value)", fontsize=10, loc="left")
    fig.tight_layout(rect=(0, 0.01, 1, 1))
    return _save(fig, path, watermark, "FULL_EXTRACT")


def bars(path: Path, labels: list[str], values: list[float], *, title: str, xlabel: str, watermark: str, basis: str, color: str = SERIES[0]) -> Path:
    fig = _fig(10, 0.3 * len(labels) + 1.5)
    ax = fig.add_subplot(111)
    _style(ax)
    ax.barh(range(len(labels)), values, color=color, height=0.7)
    for i, v in enumerate(values):
        ax.text(v, i, f" {v:.1f}", va="center", fontsize=7, color=INK)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=10, loc="left")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return _save(fig, path, watermark, basis)


def heatmap(path: Path, mat: pd.DataFrame, *, title: str, watermark: str, basis: str, diverging: bool = True, label: str = "") -> Path | None:
    if mat.empty:
        return None
    k = len(mat)
    size = min(max(6.0, 0.13 * k + 3), 22)
    fig = _fig(size + 1.5, size)
    ax = fig.add_subplot(111)
    im = ax.imshow(mat.to_numpy(dtype=float), cmap=DIVERGING if diverging else SEQUENTIAL, vmin=-1 if diverging else 0, vmax=1, interpolation="nearest")
    fs = 6 if k <= 60 else (4.5 if k <= 110 else 3.5)
    ax.set_xticks(range(k), list(mat.columns), rotation=90, fontsize=fs)
    ax.set_yticks(range(k), list(mat.index), fontsize=fs)
    ax.set_title(title, fontsize=10, loc="left", color=INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    cb.set_label(label, fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.01, 1, 1))
    return _save(fig, path, watermark, basis)


def pattern_matrix(path: Path, patterns: pd.DataFrame, sources: list[str], *, watermark: str) -> Path | None:
    """UpSet-style view: which sources are entirely missing together, and how many rows show each pattern."""
    d = patterns[patterns["pattern"] != "(other patterns, merged)"].reset_index(drop=True)
    if d.empty:
        return None
    fig = _fig(11, 0.36 * len(d) + 2.4)
    gs = fig.add_gridspec(1, 2, width_ratios=[2.2, 1])
    ax = fig.add_subplot(gs[0])
    ax.set_facecolor(SURFACE)
    for i, (_, r) in enumerate(d.iterrows()):
        for j, s in enumerate(sources):
            miss = int(r.get(s) or 0) == 1
            ax.scatter(j, i, s=60, color=SERIES[1] if miss else GRID, zorder=3)
    ax.set_xticks(range(len(sources)), [s.replace("_", " ").lower() for s in sources], rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(d)), [f"pattern {i + 1}" for i in range(len(d))], fontsize=7)
    ax.set_ylim(len(d) - 0.5, -0.5)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_title("orange = every predictor column of this source is NULL on the row", fontsize=9, loc="left", color=INK)
    ax2 = fig.add_subplot(gs[1])
    _style(ax2)
    ax2.barh(range(len(d)), pd.to_numeric(d["n_rows"], errors="coerce").fillna(0), color=SERIES[0], height=0.6)
    ax2.set_ylim(len(d) - 0.5, -0.5)
    ax2.set_yticks([])
    ax2.set_xlabel("rows (modelling cohort)")
    fig.suptitle("Source-level missingness patterns (patterns with < min_cell rows merged, not shown)", fontsize=10, x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    return _save(fig, path, watermark, "MODELLING_COHORT")


def histogram_share(path: Path, share: pd.Series, *, title: str, xlabel: str, watermark: str, basis: str, min_cell: int) -> Path:
    fig = _fig(10, 3.6)
    ax = fig.add_subplot(111)
    _style(ax)
    x = share.dropna().to_numpy(dtype=float)
    counts, edges = np.histogram(x, bins=np.linspace(0, 1, 41))
    shown = np.where((counts > 0) & (counts < min_cell), 0, counts)
    ax.bar(edges[:-1] * 100, shown, width=np.diff(edges) * 100 * 0.9, align="edge", color=SERIES[0])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("rows")
    ax.set_title(title, fontsize=10, loc="left")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return _save(fig, path, watermark, basis)


def smd_dots(path: Path, d: pd.DataFrame, *, value_cols: list[tuple[str, str]], label_col: str, title: str, watermark: str, basis: str,
             band: float | None = 0.1, xlabel: str = "standardised mean difference") -> Path | None:
    d = d.copy()
    if d.empty:
        return None
    fig = _fig(11, 0.22 * len(d) + 1.8)
    ax = fig.add_subplot(111)
    _style(ax)
    y = np.arange(len(d))
    if band:
        ax.axvspan(-band, band, color=GRID, alpha=0.6, lw=0)
    ax.axvline(0, color=INK2, lw=0.8)
    for i, (col, lab) in enumerate(value_cols):
        ax.scatter(pd.to_numeric(d[col], errors="coerce"), y + (i - (len(value_cols) - 1) / 2) * 0.18, s=26, color=SERIES[i], label=lab, zorder=3)
    ax.set_yticks(y, [textwrap(str(v), 70) for v in d[label_col]], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    if len(value_cols) > 1:
        ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.set_title(title, fontsize=10, loc="left")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return _save(fig, path, watermark, basis)


def forest_ratio(path: Path, d: pd.DataFrame, *, label_col: str, title: str, watermark: str, basis: str) -> Path | None:
    d = d.copy()
    d["pr"] = pd.to_numeric(d["prevalence_ratio_1_vs_0"], errors="coerce")
    d = d.dropna(subset=["pr"])
    if d.empty:
        return None
    d["lo"] = pd.to_numeric(d["pr_ci_low"], errors="coerce")
    d["hi"] = pd.to_numeric(d["pr_ci_high"], errors="coerce")
    fig = _fig(11, 0.22 * len(d) + 1.8)
    ax = fig.add_subplot(111)
    _style(ax)
    y = np.arange(len(d))
    ax.errorbar(d["pr"], y, xerr=[d["pr"] - d["lo"], d["hi"] - d["pr"]], fmt="o", color=SERIES[0], ms=5, capsize=2, lw=1.2)
    ax.axvline(1.0, color=INK2, lw=0.8)
    ax.set_xscale("log")
    ax.set_yticks(y, [textwrap(str(v), 70) for v in d[label_col]], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("prevalence ratio, flag = 1 vs flag = 0 (Katz 95% CI, log scale)")
    ax.set_title(title, fontsize=10, loc="left")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return _save(fig, path, watermark, basis)


def temporal_bars(path: Path, t: pd.DataFrame, *, watermark: str) -> Path | None:
    d = t[t["kind"] == "date column"].copy()
    if d.empty:
        return None
    for c in ("n_before_index", "n_equal_index", "n_after_index"):
        d[c + "_n"] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)
    tot = (d["n_before_index_n"] + d["n_equal_index_n"] + d["n_after_index_n"]).replace(0, np.nan)
    fig = _fig(11, 0.3 * len(d) + 1.8)
    ax = fig.add_subplot(111)
    _style(ax)
    left = np.zeros(len(d))
    for c, lab, col in (("n_before_index_n", "before Index_Date", SERIES[0]), ("n_equal_index_n", "equal to Index_Date", SERIES[3]),
                        ("n_after_index_n", "after Index_Date", SERIES[7])):
        v = (100 * d[c] / tot).fillna(0.0).to_numpy()
        ax.barh(range(len(d)), v, left=left, color=col, height=0.7, label=lab, edgecolor=SURFACE, linewidth=0.4)
        left += v
    ax.set_yticks(range(len(d)), [f"{c} [{r}]" for c, r in zip(d["column"], d["leakage_risk"])], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of observed values (declared sentinels excluded)")
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="lower right")
    ax.set_title("Every date column against Index_Date (full extract); label in brackets = leakage-risk class", fontsize=10, loc="left")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return _save(fig, path, watermark, "FULL_EXTRACT")


def provenance_graph(path: Path, prov: pd.DataFrame, *, watermark: str) -> Path | None:
    """Record date -> aggregate column -> canonical feature -> model sets, for the mapped (current) features."""
    d = prov[prov["canonical_feature"].astype(str) != ""].copy()
    if d.empty:
        return None
    cols = [list(dict.fromkeys(d["record_date_column"])), list(dict.fromkeys(d["aggregate_column"])), list(dict.fromkeys(d["canonical_feature"])),
            ["STRICT", "EXTENDED", "SAFE_ALL_ROWS"]]
    status = {r["record_date_column"]: r["source_timing_status"] for _, r in d.iterrows()}
    colour = {"PROVEN_INDEX_DAY_OR_FUTURE_RECORDS": STATUS["CRITICAL"], "CLEAN_IN_THIS_EXTRACT": "#0ca30c", "PARTIAL_EVIDENCE_ONLY": STATUS["WARNING"],
              "UNVERIFIABLE_NO_RECORD_DATE": "#b9b8b3"}
    n = max(len(c) for c in cols)
    fig = _fig(13, 0.42 * n + 2)
    ax = fig.add_subplot(111)
    ax.set_facecolor(SURFACE)
    ax.axis("off")
    xs = [0.0, 1.0, 2.0, 3.0]
    pos: dict[tuple[int, str], tuple[float, float]] = {}
    for k, names in enumerate(cols):
        for i, nm in enumerate(names):
            y = (i + 0.5) * n / len(names)
            pos[(k, nm)] = (xs[k], y)
    edges = set()
    for _, r in d.iterrows():
        edges.add(((0, r["record_date_column"]), (1, r["aggregate_column"])))
        for feat in str(r["canonical_feature"]).split("; "):
            edges.add(((1, r["aggregate_column"]), (2, feat)))
            for m in str(r["models"]).split("; "):
                if m in cols[3]:
                    edges.add(((2, feat), (3, m)))
    for a, b in edges:
        (x0, y0), (x1, y1) = pos[a], pos[b]
        ax.plot([x0 + 0.08, x1 - 0.08], [y0, y1], color=GRID, lw=1.0, zorder=1)
    for (k, nm), (x, y) in pos.items():
        fc = colour.get(status.get(nm, ""), SERIES[0]) if k == 0 else (SERIES[0] if k < 3 else SERIES[1])
        ax.text(x, y, nm, ha="center", va="center", fontsize=7.5, color="white" if k else INK, zorder=3,
                bbox={"boxstyle": "round,pad=0.3", "fc": fc, "ec": "none"})
    for k, head in enumerate(["source record date (timing evidence)", "aggregate column (VIEW)", "canonical feature", "model / analysis"]):
        ax.text(xs[k], -0.6, head, ha="center", fontsize=9, color=INK, weight="bold")
    ax.set_xlim(-0.6, 3.6)
    ax.set_ylim(n + 0.3, -1.0)
    ax.text(-0.55, n + 0.1, "record-date colour: red = index-day / future records proven (D-00); green = clean in this extract; "
            "yellow = partial (sufficient-only) evidence; grey = no record date (unverifiable)", fontsize=7.5, color=INK2)
    return _save(fig, path, watermark, "DEFINITIONS + D-00 evidence of this extract")


def dq_summary(path: Path, dq: pd.DataFrame, *, watermark: str) -> Path | None:
    d = dq[dq["status"] == "FINDING"]
    if d.empty:
        return None
    tab = d.groupby(["category", "severity"]).size().unstack(fill_value=0)
    fig = _fig(10, 0.36 * len(tab) + 1.6)
    ax = fig.add_subplot(111)
    _style(ax)
    left = np.zeros(len(tab))
    for sev in ("CRITICAL", "WARNING", "INFO"):
        if sev in tab.columns:
            ax.barh(range(len(tab)), tab[sev], left=left, color=STATUS[sev], height=0.65, label=sev.lower(), edgecolor=SURFACE)
            left += tab[sev].to_numpy()
    ax.set_yticks(range(len(tab)), list(tab.index), fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("number of checks with findings")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.set_title("Data-quality findings by category and severity", fontsize=10, loc="left")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return _save(fig, path, watermark, "FULL_EXTRACT")


def safe_name(col: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in col)


def nice(x: float) -> str:
    return f"{x:.3g}" if math.isfinite(x) else "n/a"
