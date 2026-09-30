"""Report plots (architecture §3.4 ``plots/``; calibration curves D-17; decision curves §7.1, D-21).

Figures are drawn on stand-alone ``matplotlib.figure.Figure`` objects rendered by the Agg canvas, so no
global pyplot state or GUI backend is touched. Files are PNG at 150 dpi with the ``Software`` metadata
removed, which keeps output byte-identical for identical inputs. Every function returns the written path.
Invalid inputs raise ``ValueError``; nothing is silently dropped (removed non-finite rows are logged).
"""

from __future__ import annotations

import json
import math
import sys
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

if "matplotlib.pyplot" not in sys.modules:
    # Headless Agg backend before any pyplot import (no Tk/GUI probing on Windows servers or over RDP). When a caller
    # already imported pyplot (e.g. a notebook), its backend is left alone: switching would close the caller's figures.
    matplotlib.use("Agg")

from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import MaxNLocator, NullFormatter  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

from falls_ml.evaluation.metrics import UndefinedMetricWarning, auroc, grouped_calibration, pr_auc, smoothed_calibration  # noqa: E402
from falls_ml.logging_utils import get_logger  # noqa: E402

log = get_logger(__name__)

DPI = 150
OUTCOME_LABEL = "fall/fracture within 12 months"
ODDS_RATIO_TICKS = (0.1, 0.2, 0.33, 0.5, 0.67, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0)


# ---------------------------------------------------------------------- helpers
def _figure(width: float = 7.0, height: float = 5.0) -> Figure:
    fig = Figure(figsize=(width, height))
    FigureCanvasAgg(fig)
    return fig


def _save(fig: Figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, format="png", metadata={"Software": None}, bbox_inches="tight")
    return path


def _require_columns(df: pd.DataFrame, columns: Sequence[str], what: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{what}: missing columns {missing}")


def _outcome_and_risk(y: Any, p: Any, label: str) -> tuple[np.ndarray, np.ndarray]:
    yy = np.asarray(y, dtype=float)
    pp = np.asarray(p, dtype=float)
    if yy.ndim != 1 or yy.shape != pp.shape or yy.size == 0:
        raise ValueError(f"{label}: y and p must be non-empty 1-D arrays of equal length")
    if not np.isin(yy, (0.0, 1.0)).all():
        raise ValueError(f"{label}: outcome must be 0/1")
    if not (np.isfinite(pp).all() and (pp >= 0).all() and (pp <= 1).all()):
        raise ValueError(f"{label}: predicted risks must be finite and lie in [0, 1]")
    if yy.min() == yy.max():
        raise ValueError(f"{label}: both outcome classes are required")
    return yy.astype(np.int64), pp


def _check_curves(curves: Mapping[str, tuple[Any, Any]]) -> None:
    if not curves:
        raise ValueError("at least one (y, p) curve is required")


def _finite_rows(df: pd.DataFrame, columns: Sequence[str], what: str) -> pd.DataFrame:
    values = df[list(columns)].apply(pd.to_numeric, errors="coerce")
    keep = np.isfinite(values.to_numpy(dtype=float)).all(axis=1)
    if not keep.all():
        log.info("plot_rows_without_values", extra_fields={"plot": what, "n_removed": int((~keep).sum()), "n_total": len(df)})
    return df.loc[keep]


def _no_data(fig: Figure, ax: Any, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True, transform=ax.transAxes)
    ax.set_axis_off()


def _as_bool(series: pd.Series) -> pd.Series:
    """Booleans from bool columns or CSV round-tripped strings ("True"/"False")."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype("string").str.lower().isin(["true", "1", "1.0"]).astype(bool)


# ---------------------------------------------------------------------- discrimination
def plot_roc(curves: Mapping[str, tuple[Any, Any]], path: str | Path, title: str) -> Path:
    """ROC curves, one per label, with AUROC in the legend."""
    _check_curves(curves)
    fig = _figure(6.5, 6.0)
    ax = fig.add_subplot()
    for label, (y, p) in curves.items():
        yy, pp = _outcome_and_risk(y, p, label)
        fpr, tpr, _ = roc_curve(yy, pp)
        ax.plot(fpr, tpr, lw=1.8, label=f"{label} (AUROC {auroc(yy, pp):.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1, label="No discrimination (AUROC 0.5)")
    ax.set(xlim=(0, 1), ylim=(0, 1.01), title=title,
           xlabel="1 − specificity (false-positive rate)", ylabel="Sensitivity (true-positive rate)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_precision_recall(curves: Mapping[str, tuple[Any, Any]], path: str | Path, title: str,
                          prevalence: float | None = None) -> Path:
    """Precision–recall curves with PR-AUC in the legend and the no-skill line at the outcome prevalence."""
    _check_curves(curves)
    fig = _figure(6.5, 6.0)
    ax = fig.add_subplot()
    for label, (y, p) in curves.items():
        yy, pp = _outcome_and_risk(y, p, label)
        precision, recall, _ = precision_recall_curve(yy, pp)
        ax.step(recall, precision, where="post", lw=1.8, label=f"{label} (PR-AUC {pr_auc(yy, pp):.3f})")
    if prevalence is not None:
        if not 0.0 <= prevalence <= 1.0:
            raise ValueError("prevalence must lie in [0, 1]")
        ax.axhline(prevalence, ls="--", color="grey", lw=1, label=f"No skill (prevalence {prevalence:.3f})")
    ax.set(xlim=(0, 1), ylim=(0, 1.01), title=title,
           xlabel="Recall (sensitivity)", ylabel="Precision (positive predictive value)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, path)


# ---------------------------------------------------------------------- calibration and clinical utility
def plot_calibration(grouped: pd.DataFrame, smoothed: pd.DataFrame | None, path: str | Path, title: str,
                     predictions: Any | None = None) -> Path:
    """Grouped observed-vs-predicted points (with CIs), LOWESS curve, ideal diagonal and a risk histogram (D-17)."""
    _require_columns(grouped, ["mean_predicted", "observed_rate"], "grouped calibration")
    pp = None
    if predictions is not None:
        pp = np.asarray(predictions, dtype=float)
        if pp.ndim != 1 or pp.size == 0 or not np.isfinite(pp).all():
            raise ValueError("predictions must be a non-empty finite 1-D array")
    fig = _figure(6.5, 7.0)
    grid = fig.add_gridspec(2, 1, height_ratios=(4, 1), hspace=0.08)
    ax = fig.add_subplot(grid[0])
    points = _finite_rows(grouped, ["mean_predicted", "observed_rate"], "calibration_grouped")
    upper = float(np.nanmax([points["mean_predicted"].max(), points["observed_rate"].max(), 0.01])) if len(points) else 1.0
    if smoothed is not None and len(smoothed):
        _require_columns(smoothed, ["mean_predicted", "observed_rate"], "smoothed calibration")
        curve = _finite_rows(smoothed, ["mean_predicted", "observed_rate"], "calibration_smoothed")
        if len(curve):
            ax.plot(curve["mean_predicted"], curve["observed_rate"], color="tab:blue", lw=1.8, label="LOWESS (frac 0.75)")
            upper = max(upper, float(curve["mean_predicted"].max()), float(curve["observed_rate"].max()))
        else:
            ax.plot([], [], color="tab:blue", label="LOWESS undefined for these risks")
    upper = min(1.0, max(upper, float(pp.max()) if pp is not None else 0.0) * 1.05)
    ax.plot([0, upper], [0, upper], ls="--", color="grey", lw=1, label="Ideal calibration")
    if len(points):
        yerr = None
        if {"ci_low", "ci_high"} <= set(points.columns):
            low = (points["observed_rate"] - pd.to_numeric(points["ci_low"], errors="coerce")).clip(lower=0).fillna(0)
            high = (pd.to_numeric(points["ci_high"], errors="coerce") - points["observed_rate"]).clip(lower=0).fillna(0)
            yerr = np.vstack([low.to_numpy(float), high.to_numpy(float)])
        ax.errorbar(points["mean_predicted"], points["observed_rate"], yerr=yerr, fmt="o", ms=4, color="tab:orange",
                    ecolor="tab:orange", elinewidth=0.8, capsize=2, label=f"Risk groups (n={len(points)}, 95% CI)")
    ax.set(xlim=(0, upper), ylim=(0, upper), title=title, ylabel=f"Observed proportion ({OUTCOME_LABEL})")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    hist = fig.add_subplot(grid[1], sharex=ax)
    if pp is not None:
        hist.hist(pp, bins=50, range=(0, upper), color="grey")
        hist.set_ylabel("Patients")
    else:
        hist.text(0.5, 0.5, "Risk distribution not supplied", ha="center", va="center", transform=hist.transAxes)
        hist.set_yticks([])
    hist.set_xlabel("Predicted 12-month risk")
    ax.tick_params(labelbottom=False)
    return _save(fig, path)


def plot_calibration_before_after(y: Any, p_uncalibrated: Any, p_recalibrated: Any, path: str | Path, title: str,
                                  n_groups: int = 10) -> Path:
    """Calibration of the uncalibrated and recalibrated risks on the same rows: grouped points and LOWESS for both (D-17).

    ``n_groups`` equal-size risk groups per variant; a LOWESS curve that is undefined for a variant is named in the legend.
    """
    yy, pu = _outcome_and_risk(y, p_uncalibrated, "calibration before recalibration")
    _, pr = _outcome_and_risk(yy, p_recalibrated, "calibration after recalibration")
    fig = _figure(6.5, 6.0)
    ax = fig.add_subplot()
    upper = 0.01
    for label, pp, color, marker in (("Uncalibrated", pu, "tab:blue", "o"), ("Recalibrated", pr, "tab:red", "s")):
        points = grouped_calibration(yy, pp, n_groups=n_groups)
        ax.plot(points["mean_predicted"], points["observed_rate"], marker, ms=4, color=color, label=f"{label}: risk groups (n={n_groups})")
        upper = max(upper, float(points["mean_predicted"].max()), float(points["observed_rate"].max()))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UndefinedMetricWarning)  # an undefined LOWESS is shown in the legend instead
            curve = smoothed_calibration(yy, pp)
        curve = curve.loc[np.isfinite(curve["observed_rate"].to_numpy(dtype=float))]
        if len(curve):
            ax.plot(curve["mean_predicted"], curve["observed_rate"], color=color, lw=1.8, label=f"{label}: LOWESS (frac 0.75)")
            upper = max(upper, float(curve["mean_predicted"].max()), float(curve["observed_rate"].max()))
        else:
            log.info("plot_lowess_undefined", extra_fields={"plot": "calibration_before_after", "variant": label})
            ax.plot([], [], color=color, lw=1.8, label=f"{label}: LOWESS undefined for these risks")
    upper = min(1.0, upper * 1.05)
    ax.plot([0, upper], [0, upper], ls="--", color="grey", lw=1, label="Ideal calibration")
    ax.set(xlim=(0, upper), ylim=(0, upper), title=title, xlabel="Predicted 12-month risk",
           ylabel=f"Observed proportion ({OUTCOME_LABEL})")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_decision_curve(dc: pd.DataFrame, path: str | Path, title: str,
                        highlight: tuple[float, float] | None = (0.10, 0.25)) -> Path:
    """Net benefit of the model(s) vs treat-all and treat-none; ``highlight`` shades the threshold region of interest."""
    _require_columns(dc, ["threshold", "net_benefit_model", "net_benefit_treat_all"], "decision curve")
    if dc.empty:
        raise ValueError("decision curve: no rows")
    groups = list(dc.groupby("variant", sort=False)) if "variant" in dc.columns and dc["variant"].nunique() > 1 else [("Model", dc)]
    for label, part in groups:
        if part["threshold"].duplicated().any():
            raise ValueError(f"decision curve {label!r}: duplicated thresholds (filter decision_curve.csv to one split first)")
    fig = _figure(7.0, 5.0)
    ax = fig.add_subplot()
    ymax = 0.0
    for label, part in groups:
        part = part.sort_values("threshold")
        ax.plot(part["threshold"], part["net_benefit_model"], lw=1.8, label=str(label))
        ymax = max(ymax, float(np.nanmax(part["net_benefit_model"])))
    first = groups[0][1].sort_values("threshold")
    ax.plot(first["threshold"], first["net_benefit_treat_all"], color="grey", lw=1.2, label="Treat all")
    ax.axhline(0.0, color="black", lw=1.0, ls="--", label="Treat none")
    ymax = max(ymax, float(np.nanmax(first["net_benefit_treat_all"])), 1e-3) * 1.1
    if highlight is not None:
        lo, hi = highlight
        ax.axvspan(lo, hi, color="tab:green", alpha=0.12, label=f"Thresholds of interest ({lo:.0%}–{hi:.0%})")
    ax.set(xlim=(float(dc["threshold"].min()), float(dc["threshold"].max())), ylim=(-0.25 * ymax, ymax), title=title,
           xlabel="Risk threshold for intervention", ylabel="Net benefit (true positives per patient, harm-adjusted)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_risk_distribution(y: Any, p: Any, path: str | Path, title: str) -> Path:
    """Histograms of predicted risk for patients with and without the outcome (proportion within group)."""
    yy, pp = _outcome_and_risk(y, p, "risk distribution")
    fig = _figure(7.0, 4.5)
    ax = fig.add_subplot()
    bins = np.linspace(0.0, max(float(pp.max()), 1e-3), 41)
    for value, label, color in ((0, "No " + OUTCOME_LABEL, "tab:blue"), (1, OUTCOME_LABEL.capitalize(), "tab:red")):
        group = pp[yy == value]
        ax.hist(group, bins=bins, weights=np.full(group.size, 1.0 / group.size), alpha=0.55, color=color,
                label=f"{label} (n={group.size})")
    ax.set(title=title, xlabel="Predicted 12-month risk", ylabel="Proportion of patients in group")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, path)


# ---------------------------------------------------------------------- features
def plot_feature_importance(fi: pd.DataFrame, path: str | Path, title: str, top_n: int = 15) -> Path:
    """Top ``top_n`` features by mean permutation importance with ± SD error bars."""
    _require_columns(fi, ["feature", "permutation_importance_mean"], "feature importance")
    fig = _figure(7.0, max(3.0, 0.35 * top_n + 1.5))
    ax = fig.add_subplot()
    data = _finite_rows(fi, ["permutation_importance_mean"], "feature_importance")
    if data.empty:
        _no_data(fig, ax, "No permutation importance values available")
        ax.set_title(title)
        return _save(fig, path)
    top = data.sort_values("permutation_importance_mean", ascending=False, kind="stable").head(top_n).iloc[::-1]
    std = pd.to_numeric(top["permutation_importance_std"], errors="coerce").fillna(0.0) if "permutation_importance_std" in top else None
    ax.barh(top["feature"].astype(str), top["permutation_importance_mean"], xerr=std, color="tab:blue", ecolor="black", capsize=2)
    ax.axvline(0.0, color="black", lw=0.8)
    ax.set(title=title, xlabel="Drop in AUROC when the feature is randomly permuted (mean ± SD)")
    ax.grid(axis="x", alpha=0.3)
    return _save(fig, path)


def plot_coefficients(coef: pd.DataFrame, path: str | Path, title: str, top_n: int = 25) -> Path:
    """Odds ratios of selected (non-zero) coefficients, ranked by |standardised coefficient| when available.

    Rows re-expressed relative to published reference levels (``relative_to_reference``, D-10), the ``_intercept`` row and
    Stata-style omitted columns (``omitted``) are not plotted.

    If standardised coefficients are missing the ranking uses |coefficient| on the original scale and the
    figure says so (magnitudes are then not comparable across predictors with different units).
    """
    _require_columns(coef, ["coefficient"], "coefficients")
    label_col = "design_column" if "design_column" in coef.columns else "feature"
    _require_columns(coef, [label_col], "coefficients")
    if "relative_to_reference" in coef.columns:  # D-10 re-expressed rows duplicate the fitted design columns
        coef = coef.loc[~_as_bool(coef["relative_to_reference"])]
    if "omitted" in coef.columns:
        coef = coef.loc[~_as_bool(coef["omitted"])]
    coef = coef.loc[coef[label_col].astype(str) != "_intercept"]
    fig = _figure(7.0, max(3.0, 0.32 * top_n + 1.8))
    ax = fig.add_subplot()
    data = _finite_rows(coef, ["coefficient"], "coefficients")
    data = data.loc[data["coefficient"].astype(float) != 0.0]
    if data.empty:
        _no_data(fig, ax, "No non-zero coefficients (no predictor selected)")
        ax.set_title(title)
        return _save(fig, path)
    standardized = pd.to_numeric(data["standardized_coefficient"], errors="coerce") if "standardized_coefficient" in data else None
    use_std = standardized is not None and standardized.notna().all()
    rank_key = standardized.abs() if use_std else data["coefficient"].astype(float).abs()
    top = data.assign(_rank=rank_key.to_numpy()).sort_values("_rank", ascending=False, kind="stable").head(top_n).iloc[::-1]
    odds = np.exp(top["coefficient"].astype(float))
    colors = np.where(odds >= 1.0, "tab:red", "tab:blue")
    ax.barh(top[label_col].astype(str), odds - 1.0, left=1.0, color=colors)
    ax.axvline(1.0, color="black", lw=0.8)
    ax.set_xscale("log")
    lo, hi = ax.get_xlim()
    ticks = [t for t in ODDS_RATIO_TICKS if lo <= t <= hi]
    ax.set_xticks(ticks, labels=[f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_formatter(NullFormatter())
    note = ("ranked by |standardised coefficient|" if use_std else
            "UNSTANDARDISED: ranked by |coefficient| on the original scale; magnitudes not comparable across units")
    ax.set(title=f"{title}\n({len(data)} selected; top {len(top)} shown; {note})",
           xlabel="Odds ratio (log scale; >1 higher risk, <1 lower risk)")
    ax.title.set_fontsize(9)
    ax.grid(axis="x", alpha=0.3)
    return _save(fig, path)


def plot_selection_stability(stability: pd.DataFrame, path: str | Path, title: str, top_n: int = 30,
                             selection_threshold: float = 0.8) -> Path:
    """Bootstrap selection frequency of the ``top_n`` most frequently selected features (robust ones highlighted)."""
    _require_columns(stability, ["feature", "selection_frequency"], "feature stability")
    fig = _figure(7.0, max(3.0, 0.3 * top_n + 1.5))
    ax = fig.add_subplot()
    data = _finite_rows(stability, ["selection_frequency"], "selection_stability")
    if data.empty:
        _no_data(fig, ax, "No selection-frequency values available")
        ax.set_title(title)
        return _save(fig, path)
    top = data.sort_values("selection_frequency", ascending=False, kind="stable").head(top_n).iloc[::-1]
    robust = _as_bool(top["robust"]) if "robust" in top.columns else pd.Series(False, index=top.index)
    ax.barh(top["feature"].astype(str), top["selection_frequency"], color=np.where(robust, "tab:green", "tab:grey"))
    line = ax.axvline(selection_threshold, color="black", ls="--", lw=1, label=f"Selection threshold {selection_threshold:.0%}")
    handles = [line, Patch(color="tab:green", label="Robust (frequently selected, stable sign)"),
               Patch(color="tab:grey", label="Not robust")]
    ax.set(xlim=(0, 1), title=title, xlabel="Proportion of bootstrap refits that selected the feature")
    ax.legend(handles=handles, loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    return _save(fig, path)


def plot_hyperparameter_search(results: pd.DataFrame, path: str | Path, title: str) -> Path:
    """Validation objective per trial and the AUROC/Brier trade-off, with the selected trial highlighted."""
    _require_columns(results, ["trial", "objective", "auroc", "brier"], "hyperparameter results")
    if results.empty:
        raise ValueError("hyperparameter results: no trials")
    data = results.sort_values("trial", kind="stable")
    selected = _as_bool(data["selected"]) if "selected" in data.columns else pd.Series(False, index=data.index)
    fig = _figure(11.0, 4.8)
    left, right = fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)
    left.plot(data["trial"], data["objective"], "o", ms=5, color="tab:blue", label="Trial")
    right.scatter(data["brier"], data["auroc"], s=18, color="tab:blue", label="Trial")
    if selected.any():
        best = data.loc[selected]
        left.plot(best["trial"], best["objective"], "*", ms=14, color="tab:red", label="Selected")
        right.scatter(best["brier"], best["auroc"], marker="*", s=180, color="tab:red", label="Selected")
        if "params_json" in best.columns:
            params = json.loads(str(best["params_json"].iloc[0]))
            fig.text(0.5, -0.02, "Selected parameters: " + ", ".join(f"{k}={v}" for k, v in params.items()),
                     ha="center", fontsize=8)
    left.set(xlabel="Trial", ylabel="Composite objective (higher is better)", title="Objective by trial")
    right.set(xlabel="Validation Brier score (lower is better)", ylabel="Validation AUROC (higher is better)",
              title="Discrimination vs accuracy")
    left.xaxis.set_major_locator(MaxNLocator(integer=True))
    right.xaxis.set_major_locator(MaxNLocator(nbins=5))
    for ax in (left, right):
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    return _save(fig, path)


def plot_lasso_cv_path(cv: pd.DataFrame, path: str | Path, title: str) -> Path:
    """Cross-validated mean deviance (± SE band) against ln(λ), with the selected λ* marked (D-14)."""
    _require_columns(cv, ["lambda", "cv_mean_deviance", "cv_se"], "LASSO CV results")
    data = _finite_rows(cv, ["lambda", "cv_mean_deviance"], "lasso_cv_path")
    positive = data["lambda"].astype(float) > 0
    if not positive.all():
        log.info("plot_rows_without_values", extra_fields={"plot": "lasso_cv_path_lambda_le_0", "n_removed": int((~positive).sum()),
                                                            "n_total": len(cv)})
    data = data.loc[positive].sort_values("lambda")
    if data.empty:
        raise ValueError("LASSO CV results: no finite (lambda > 0, deviance) rows")
    x = np.log(data["lambda"].astype(float).to_numpy())
    dev = data["cv_mean_deviance"].astype(float).to_numpy()
    se = pd.to_numeric(data["cv_se"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    fig = _figure(7.0, 5.0)
    ax = fig.add_subplot()
    ax.fill_between(x, dev - se, dev + se, color="tab:blue", alpha=0.2, label="± 1 standard error")
    ax.plot(x, dev, color="tab:blue", lw=1.8, label="Mean held-out deviance")
    if "selected" in data.columns and _as_bool(data["selected"]).any():
        star = data.loc[_as_bool(data["selected"])].iloc[0]
        nonzero = f", {int(star['n_nonzero'])} non-zero" if "n_nonzero" in star and not pd.isna(star["n_nonzero"]) else ""
        ax.axvline(math.log(float(star["lambda"])), color="tab:red", ls="--", lw=1.2,
                   label=f"λ* = {float(star['lambda']):.3g}{nonzero}")
    ax.set(title=title, xlabel="ln(λ)  (larger λ = stronger penalty, fewer predictors)",
           ylabel="Cross-validated deviance per observation")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, path)
