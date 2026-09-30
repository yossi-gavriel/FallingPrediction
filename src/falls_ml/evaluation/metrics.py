"""Performance measures for binary risk predictions (spec §7.1, D-17, D-21; formulas in validation_methods.md §2–4).

Conventions
- ``y`` is a 1-D array of 0/1 outcomes, ``p`` the predicted risks in [0, 1], ``lp`` the linear predictor
  (log-odds). Where a measure needs a linear predictor and ``lp`` is ``None``, ``logit(p)`` is used.
- Invalid inputs raise ``ValueError``. Measures that are undefined for valid inputs (a single outcome class,
  a logistic fit whose maximum-likelihood estimate does not exist, ...) return ``NaN`` and emit an
  :class:`UndefinedMetricWarning`; they never silently return a number.
- Every classification rule is "positive if p >= threshold" (Vickers 2008).
"""

from __future__ import annotations

import math
import re
import warnings
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit as _scipy_expit
from scipy.stats import norm, rankdata
from statsmodels.nonparametric.smoothers_lowess import lowess

DEFAULT_THRESHOLDS: tuple[float, ...] = (0.10, 0.15, 0.20, 0.25)
LOGISTIC_MAX_ITER = 100
LOGISTIC_TOL = 1e-10
LOGISTIC_MAX_ABS_COEF = 1e6  # a coefficient beyond this signals (quasi-)separation: the MLE does not exist
SINGULAR_COND = 1e12
NET_BENEFIT_METRIC = re.compile(r"^net_benefit@(?P<t>[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)$")


class UndefinedMetricWarning(RuntimeWarning):
    """A measure is undefined for the given (valid) data and was returned as NaN."""


# ---------------------------------------------------------------------- input validation
def _as_outcome(y: Any) -> np.ndarray:
    arr = np.asarray(y)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"y must be a non-empty 1-D array, got shape {arr.shape}")
    if arr.dtype == bool:
        return arr.astype(np.float64)
    if arr.dtype.kind not in "iuf":
        raise ValueError(f"y must be numeric 0/1, got dtype {arr.dtype}")
    out = arr.astype(np.float64)
    if not np.all((out == 0.0) | (out == 1.0)):
        raise ValueError("y must contain only 0 and 1 (no missing values)")
    return out


def _as_float(x: Any, n: int, name: str) -> np.ndarray:
    arr = np.asarray(x)
    if arr.dtype.kind not in "biuf":
        raise ValueError(f"{name} must be numeric, got dtype {arr.dtype}")
    arr = arr.astype(np.float64)
    if arr.ndim != 1 or arr.size != n:
        raise ValueError(f"{name} must be a 1-D array of length {n}, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _as_probability(p: Any, n: int, name: str = "p") -> np.ndarray:
    arr = _as_float(p, n, name)
    if np.any((arr < 0.0) | (arr > 1.0)):
        raise ValueError(f"{name} must lie in [0, 1]")
    return arr


def _validate_threshold(t: float) -> float:
    t = float(t)
    if not 0.0 < t < 1.0:
        raise ValueError(f"thresholds must lie in (0, 1), got {t}")
    return t


def _check_y_p(y: Any, p: Any) -> tuple[np.ndarray, np.ndarray]:
    yy = _as_outcome(y)
    return yy, _as_probability(p, yy.size)


def _check_y_lp(y: Any, lp: Any) -> tuple[np.ndarray, np.ndarray]:
    yy = _as_outcome(y)
    return yy, _as_float(lp, yy.size, "lp")


def _warn(message: str) -> None:
    warnings.warn(message, UndefinedMetricWarning, stacklevel=3)


def _single_class(y: np.ndarray) -> bool:
    s = y.sum()
    return s == 0 or s == y.size


def _none_if_nan(x: float | None) -> float | None:
    return None if x is None or not math.isfinite(x) else float(x)


# ---------------------------------------------------------------------- link functions
def logit(p: Any, eps: float = 1e-12) -> np.ndarray | float:
    """log(p / (1 - p)) with p clipped to [eps, 1 - eps] so that risks of exactly 0 or 1 stay finite."""
    arr = np.asarray(p, dtype=np.float64)
    if np.any(np.isnan(arr)) or np.any((arr < 0.0) | (arr > 1.0)):
        raise ValueError("logit requires probabilities in [0, 1]")
    q = np.clip(arr, eps, 1.0 - eps)
    out = np.log(q) - np.log1p(-q)
    return float(out) if out.ndim == 0 else out


def expit(x: Any) -> np.ndarray | float:
    """Inverse logit 1 / (1 + exp(-x))."""
    out = _scipy_expit(np.asarray(x, dtype=np.float64))
    return float(out) if np.ndim(out) == 0 else out


# ---------------------------------------------------------------------- discrimination
def _auroc(y: np.ndarray, p: np.ndarray) -> float:
    n1 = y.sum()
    n0 = y.size - n1
    if n1 == 0 or n0 == 0:
        return math.nan
    ranks = rankdata(p, method="average")
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auroc(y: Any, p: Any) -> float:
    """C-statistic = Mann–Whitney probability of concordance, ties counted 1/2 (spec §7.1)."""
    yy, pp = _check_y_p(y, p)
    if _single_class(yy):
        _warn("AUROC is undefined when y contains a single class")
        return math.nan
    return _auroc(yy, pp)


def _delong(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Fast midrank DeLong (Sun & Xu 2014): returns (AUROC, variance)."""
    pos, neg = p[y == 1], p[y == 0]
    m, n = pos.size, neg.size
    if m < 2 or n < 2:
        return math.nan, math.nan
    tx = rankdata(pos, method="average")
    ty = rankdata(neg, method="average")
    tz = rankdata(np.concatenate([pos, neg]), method="average")
    auc = (tz[:m].sum() - m * (m + 1) / 2) / (m * n)
    v10 = (tz[:m] - tx) / n            # per-event placement values
    v01 = 1.0 - (tz[m:] - ty) / m      # per-non-event placement values
    return float(auc), float(np.var(v10, ddof=1) / m + np.var(v01, ddof=1) / n)


def auroc_delong_variance(y: Any, p: Any) -> float:
    """DeLong variance of the AUROC (D-21). NaN if there are fewer than 2 events or 2 non-events."""
    yy, pp = _check_y_p(y, p)
    _, var = _delong(yy, pp)
    if math.isnan(var):
        _warn("DeLong variance needs at least two events and two non-events")
    return var


def auroc_logit_se(y: Any, p: Any) -> float:
    """SE of logit(AUROC) by the delta method: sqrt(Var_DeLong) / (C(1 - C)) (D-16, D-21)."""
    yy, pp = _check_y_p(y, p)
    auc, var = _delong(yy, pp)
    if math.isnan(var) or auc <= 0.0 or auc >= 1.0:
        _warn("logit(AUROC) SE is undefined (too few events/non-events or AUROC of 0 or 1)")
        return math.nan
    return math.sqrt(var) / (auc * (1.0 - auc))


def _pr_auc(y: np.ndarray, p: np.ndarray) -> float:
    n_pos = y.sum()
    if n_pos == 0:
        return math.nan
    order = np.argsort(-p, kind="stable")
    ys, ps = y[order], p[order]
    last_of_tie = np.r_[np.flatnonzero(np.diff(ps) != 0), ys.size - 1]
    tp = np.cumsum(ys)[last_of_tie]
    precision = tp / (last_of_tie + 1)
    recall = tp / n_pos
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def pr_auc(y: Any, p: Any) -> float:
    """Average precision: sum_k (R_k - R_{k-1}) P_k over distinct thresholds (step-wise PR-AUC)."""
    yy, pp = _check_y_p(y, p)
    if yy.sum() == 0:
        _warn("PR-AUC is undefined when there are no events")
        return math.nan
    return _pr_auc(yy, pp)


# ---------------------------------------------------------------------- accuracy and calibration
def _brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def brier(y: Any, p: Any) -> float:
    """Brier score mean((p - y)^2)."""
    return _brier(*_check_y_p(y, p))


def _fit_logistic(y: np.ndarray, x: np.ndarray | None, offset: np.ndarray | None) -> tuple[np.ndarray, np.ndarray] | None:
    """ML fit of logit P(y=1) = b0 [+ b1 x] [+ offset] by damped Newton–Raphson.

    Returns (coefficients, covariance = inverse Fisher information), or ``None`` when the MLE does not exist
    (single outcome class, separation, singular information).
    """
    if _single_class(y):
        return None
    n = y.size
    design = np.ones((n, 1)) if x is None else np.column_stack([np.ones(n), x])
    off = np.zeros(n) if offset is None else offset
    beta = np.zeros(design.shape[1])
    beta[0] = math.log(y.mean() / (1 - y.mean())) - (float(off.mean()) if x is None else 0.0)

    def loglik(b: np.ndarray) -> float:
        eta = design @ b + off
        return float(np.sum(y * eta - np.logaddexp(0.0, eta)))

    ll = loglik(beta)
    for _ in range(LOGISTIC_MAX_ITER):
        mu = _scipy_expit(design @ beta + off)
        info = (design.T * (mu * (1.0 - mu))) @ design
        if np.linalg.cond(info) > SINGULAR_COND:
            return None
        step = np.linalg.solve(info, design.T @ (y - mu))
        scale = 1.0
        while True:
            candidate = beta + scale * step
            ll_new = loglik(candidate)
            if ll_new >= ll - 1e-12 * abs(ll) or scale < 1e-10:
                break
            scale /= 2.0
        beta, ll = candidate, ll_new
        if np.max(np.abs(beta)) > LOGISTIC_MAX_ABS_COEF:
            return None
        if np.max(np.abs(scale * step)) < LOGISTIC_TOL * (1.0 + np.max(np.abs(beta))):
            mu = _scipy_expit(design @ beta + off)
            info = (design.T * (mu * (1.0 - mu))) @ design
            if np.linalg.cond(info) > SINGULAR_COND:
                return None
            return beta, np.linalg.inv(info)
    return None


def _slope_intercept(y: np.ndarray, lp: np.ndarray) -> tuple[float, float, float, float]:
    fit = _fit_logistic(y, lp, None)
    if fit is None:
        return math.nan, math.nan, math.nan, math.nan
    beta, cov = fit
    return float(beta[0]), float(beta[1]), math.sqrt(cov[0, 0]), math.sqrt(cov[1, 1])


def calibration_slope_intercept(y: Any, lp: Any) -> tuple[float, float, float, float]:
    """GLM logit P(y=1) = a + b·LP (spec §7.1). Returns (a, b, SE(a), SE(b)), Wald SEs from the information."""
    yy, ll = _check_y_lp(y, lp)
    out = _slope_intercept(yy, ll)
    if math.isnan(out[1]):
        _warn("calibration slope is undefined (single class, separation or constant LP)")
    return out


def _citl(y: np.ndarray, lp: np.ndarray) -> tuple[float, float]:
    fit = _fit_logistic(y, None, lp)
    if fit is None:
        return math.nan, math.nan
    beta, cov = fit
    return float(beta[0]), math.sqrt(cov[0, 0])


def citl(y: Any, lp: Any) -> tuple[float, float]:
    """Calibration-in-the-large: a in logit P(y=1) = a + 1·LP (LP as offset). Returns (a, SE(a)) (spec §7.1)."""
    yy, ll = _check_y_lp(y, lp)
    out = _citl(yy, ll)
    if math.isnan(out[0]):
        _warn("CITL is undefined when y contains a single class")
    return out


def _oe_ratio(y: np.ndarray, p: np.ndarray) -> float:
    expected = p.sum()
    return math.nan if expected == 0 else float(y.sum() / expected)


def oe_ratio(y: Any, p: Any) -> float:
    """Observed / expected events: sum(y) / sum(p) (spec §7.1)."""
    yy, pp = _check_y_p(y, p)
    if pp.sum() == 0:
        _warn("O/E is undefined when every predicted risk is 0")
        return math.nan
    return _oe_ratio(yy, pp)


def log_oe_se(y: Any, p: Any) -> float:
    """SE of ln(O/E) = sqrt((1 - observed_rate) / n_events) (Riley 2021 eq 4; D-21)."""
    yy, _ = _check_y_p(y, p)
    events = yy.sum()
    if events == 0:
        _warn("SE of ln(O/E) is undefined when there are no events")
        return math.nan
    return math.sqrt((1.0 - events / yy.size) / events)


def grouped_calibration(y: Any, p: Any, n_groups: int = 20, *, alpha: float = 0.05) -> pd.DataFrame:
    """Observed vs expected in equal-size groups by rank of predicted risk (D-17).

    Rows are ordered by predicted risk and split into ``n_groups`` consecutive groups whose sizes differ by at
    most one; tied risks keep their input order (stable sort), so a tie block may straddle two groups (unlike
    Stata ``xtile``). CI: Wald interval for a proportion, truncated to [0, 1] (pmcalplot).
    """
    yy, pp = _check_y_p(y, p)
    if isinstance(n_groups, bool) or not isinstance(n_groups, (int, np.integer)) or n_groups < 1:
        raise ValueError("n_groups must be a positive integer")
    if yy.size < n_groups:
        raise ValueError(f"grouped calibration needs at least n_groups={n_groups} rows, got {yy.size}")
    order = np.argsort(pp, kind="stable")
    z = norm.ppf(1 - alpha / 2)
    rows = []
    for g, idx in enumerate(np.array_split(order, n_groups), start=1):
        obs, n = float(yy[idx].mean()), int(idx.size)
        half = z * math.sqrt(obs * (1 - obs) / n)
        rows.append({"group": g, "mean_predicted": float(pp[idx].mean()), "observed_rate": obs, "n": n,
                     "ci_low": max(0.0, obs - half), "ci_high": min(1.0, obs + half)})
    return pd.DataFrame(rows).astype({"group": "int64", "n": "int64"})


def _check_lowess_frac(frac: float) -> float:
    if not 0.0 < frac <= 1.0:
        raise ValueError("frac must lie in (0, 1]")
    return float(frac)


def _lowess_problem(p: np.ndarray, frac: float) -> str | None:
    """Why LOWESS is undefined for these (valid) risks, or None.

    statsmodels uses the k = int(frac·n) nearest points; if one tied risk value has >= k rows its neighbourhood
    radius is 0 and the fit silently degenerates (division by zero, e.g. 1.0 where the true rate is 0.575).
    """
    if np.ptp(p) == 0.0:
        return "smoothed calibration needs at least two distinct predicted risks"
    k = min(p.size, max(2, int(frac * p.size + 1e-10)))
    if np.unique(p, return_counts=True)[1].max() >= k:
        return f"LOWESS undefined: a single tied risk value covers the whole {k}-point smoothing window"
    return None


def _lowess_fitted(y: np.ndarray, p: np.ndarray, frac: float) -> np.ndarray | None:
    """LOWESS fitted value at each p, or None (with an UndefinedMetricWarning) when LOWESS is undefined."""
    problem = _lowess_problem(p, frac)
    if problem is not None:
        warnings.warn(problem, UndefinedMetricWarning, stacklevel=3)
        return None
    return lowess(y, p, frac=frac, it=0, delta=0.01 * float(np.ptp(p)), return_sorted=False)


def smoothed_calibration(y: Any, p: Any, frac: float = 0.75, n_points: int = 100) -> pd.DataFrame:
    """LOWESS calibration curve (statsmodels, local linear, it=0, delta = 1% of the risk range; D-17).

    The curve is evaluated on ``n_points`` equally spaced risks between min(p) and max(p) by linear
    interpolation of the fitted values. Values are not clipped to [0, 1]. Labelled "LOWESS"; not R loess.
    If LOWESS is undefined (all risks equal, or one tied value fills the smoothing window) ``observed_rate``
    is NaN and an :class:`UndefinedMetricWarning` is emitted.
    """
    yy, pp = _check_y_p(y, p)
    frac = _check_lowess_frac(frac)
    if isinstance(n_points, bool) or not isinstance(n_points, (int, np.integer)) or n_points < 2:
        raise ValueError("n_points must be an integer >= 2")
    grid = np.linspace(pp.min(), pp.max(), n_points)
    fitted = _lowess_fitted(yy, pp, frac)
    if fitted is None:
        return pd.DataFrame({"mean_predicted": grid, "observed_rate": np.full(n_points, np.nan)})
    xs, first = np.unique(pp, return_index=True)
    return pd.DataFrame({"mean_predicted": grid, "observed_rate": np.interp(grid, xs, fitted[first])})


def calibration_indices(y: Any, p: Any, frac: float = 0.75) -> dict[str, float]:
    """ICI, E50 and E90: mean, median and 90th percentile of |p_i - LOWESS(p_i)| (Austin & Steyerberg; D-17).

    NaN (with an :class:`UndefinedMetricWarning`) when LOWESS is undefined for these risks.
    """
    yy, pp = _check_y_p(y, p)
    fitted = _lowess_fitted(yy, pp, _check_lowess_frac(frac))
    if fitted is None:
        return {"ici": math.nan, "e50": math.nan, "e90": math.nan}
    diff = np.abs(pp - fitted)
    return {"ici": float(diff.mean()), "e50": float(np.median(diff)), "e90": float(np.quantile(diff, 0.9))}


# ---------------------------------------------------------------------- thresholds and clinical utility
def net_benefit_from_counts(tp: float, fp: float, n: float, threshold: float) -> float:
    """NB(t) = TP/N - FP/N · t / (1 - t) (Vickers 2008; spec §7.1). Counts may be non-integer (per-1,000 tables)."""
    t = _validate_threshold(threshold)
    if n <= 0:
        raise ValueError("n must be positive")
    return tp / n - fp / n * t / (1.0 - t)


def net_benefit_treat_all(prevalence: float, threshold: float) -> float:
    """Treat-all NB = phi - (1 - phi) · t / (1 - t)."""
    t = _validate_threshold(threshold)
    if not 0.0 <= prevalence <= 1.0:
        raise ValueError("prevalence must lie in [0, 1]")
    return prevalence - (1.0 - prevalence) * t / (1.0 - t)


def _counts(y: np.ndarray, p: np.ndarray, t: float) -> tuple[int, int, int, int]:
    pos = p >= t
    tp = int(np.sum(pos & (y == 1)))
    fp = int(np.sum(pos & (y == 0)))
    fn = int(y.sum()) - tp
    tn = y.size - tp - fp - fn
    return tp, fp, tn, fn


def _net_benefit(y: np.ndarray, p: np.ndarray, t: float) -> float:
    tp, fp, _, _ = _counts(y, p, t)
    return net_benefit_from_counts(tp, fp, y.size, t)


def net_benefit(y: Any, p: Any, threshold: float) -> float:
    """Model net benefit at one risk threshold (positive if p >= t)."""
    yy, pp = _check_y_p(y, p)
    return _net_benefit(yy, pp, _validate_threshold(threshold))


def _ratio(num: int, den: int) -> float | None:
    return None if den == 0 else num / den


def threshold_metrics(y: Any, p: Any, thresholds: Iterable[float] = DEFAULT_THRESHOLDS) -> list[dict[str, Any]]:
    """Confusion counts and accuracy at each threshold (positive if p >= t); undefined ratios are ``None``."""
    yy, pp = _check_y_p(y, p)
    rows = []
    for t in (_validate_threshold(t) for t in thresholds):
        tp, fp, tn, fn = _counts(yy, pp, t)
        rows.append({"threshold": t, "sensitivity": _ratio(tp, tp + fn), "specificity": _ratio(tn, tn + fp),
                     "ppv": _ratio(tp, tp + fp), "npv": _ratio(tn, tn + fn), "f1": _ratio(2 * tp, 2 * tp + fp + fn),
                     "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                     "net_benefit": net_benefit_from_counts(tp, fp, yy.size, t)})
    return rows


def decision_curve(y: Any, p: Any, thresholds: Sequence[float]) -> pd.DataFrame:
    """Decision curve: model, treat-all and treat-none NB, and standardised NB = NB / phi (spec §7.1)."""
    yy, pp = _check_y_p(y, p)
    ts = np.array([_validate_threshold(t) for t in thresholds], dtype=np.float64)
    prevalence = float(yy.mean())
    model = np.array([_net_benefit(yy, pp, t) for t in ts])
    odds = ts / (1.0 - ts)
    standardized = model / prevalence if prevalence > 0 else np.full(ts.size, np.nan)
    return pd.DataFrame({"threshold": ts, "net_benefit_model": model,
                         "net_benefit_treat_all": prevalence - (1.0 - prevalence) * odds,
                         "net_benefit_treat_none": np.zeros(ts.size), "standardized_net_benefit": standardized})


# ---------------------------------------------------------------------- scalar metric registry (bootstrap, tuning)
SCALAR_METRICS: tuple[str, ...] = ("auroc", "pr_auc", "brier", "calibration_slope", "calibration_intercept", "citl", "oe_ratio")


def evaluate_prepared(name: str, y: np.ndarray, p: np.ndarray, lp: np.ndarray) -> float:
    """Scalar metric on arrays returned by :func:`prepare_inputs` (or subsets of them); no validation, NaN if undefined.

    ``name`` must already be validated with :func:`parse_metric_name`. Used in resampling loops.
    """
    if name == "auroc":
        return _auroc(y, p)
    if name == "pr_auc":
        return _pr_auc(y, p)
    if name == "brier":
        return _brier(y, p)
    if name == "calibration_slope":
        return _slope_intercept(y, lp)[1]
    if name == "calibration_intercept":
        return _slope_intercept(y, lp)[0]
    if name == "citl":
        return _citl(y, lp)[0]
    if name == "oe_ratio":
        return _oe_ratio(y, p)
    return _net_benefit(y, p, parse_metric_name(name)[1])


def evaluate_many_prepared(names: Sequence[str], y: np.ndarray, p: np.ndarray, lp: np.ndarray) -> list[float]:
    """Like :func:`evaluate_prepared` for several names, fitting the slope/intercept GLM at most once."""
    slope_fit: tuple[float, float, float, float] | None = None
    values = []
    for name in names:
        if name in ("calibration_slope", "calibration_intercept"):
            slope_fit = slope_fit or _slope_intercept(y, lp)
            values.append(slope_fit[1] if name == "calibration_slope" else slope_fit[0])
        else:
            values.append(evaluate_prepared(name, y, p, lp))
    return values


def parse_metric_name(name: str) -> tuple[str, float | None]:
    """Validate a scalar metric name; ``net_benefit@<t>`` returns ("net_benefit", t)."""
    if name in SCALAR_METRICS:
        return name, None
    match = NET_BENEFIT_METRIC.match(name)
    if match is None:
        raise ValueError(f"unknown metric {name!r}; expected one of {SCALAR_METRICS} or 'net_benefit@<threshold>'")
    return "net_benefit", _validate_threshold(float(match.group("t")))


def prepare_predictions(p: Any, lp: Any | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Validate predictions without outcomes; lp defaults to logit(p)."""
    arr = np.asarray(p)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"p must be a non-empty 1-D array, got shape {arr.shape}")
    pp = _as_probability(arr, arr.size)
    ll = np.asarray(logit(pp), dtype=np.float64) if lp is None else _as_float(lp, pp.size, "lp")
    return pp, ll


def prepare_inputs(y: Any, p: Any, lp: Any | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate (y, p, lp) once; lp defaults to logit(p)."""
    yy = _as_outcome(y)
    if np.ndim(p) != 1 or np.size(p) != yy.size:
        raise ValueError(f"p must be a 1-D array of length {yy.size}, got shape {np.shape(p)}")
    pp, ll = prepare_predictions(p, lp)
    return yy, pp, ll


def compute_metric(name: str, y: Any, p: Any, lp: Any | None = None) -> float:
    """Scalar metric by name (see :data:`SCALAR_METRICS`, or ``net_benefit@<t>``); NaN if undefined."""
    parse_metric_name(name)
    return evaluate_prepared(name, *prepare_inputs(y, p, lp))


# ---------------------------------------------------------------------- summary
def performance_summary(y: Any, p: Any, lp: Any | None = None,
                        thresholds: Iterable[float] = DEFAULT_THRESHOLDS) -> dict[str, Any]:
    """Point estimates with the ``metrics.json`` performance keys (ARTIFACT_SCHEMAS); undefined values are ``None``."""
    yy, pp, ll = prepare_inputs(y, p, lp)
    out: dict[str, Any] = {"n": int(yy.size), "n_events": int(yy.sum())}
    for name, value in zip(SCALAR_METRICS, evaluate_many_prepared(SCALAR_METRICS, yy, pp, ll)):
        out[name] = _none_if_nan(value)
    out["mean_predicted"] = float(pp.mean())
    out["observed_rate"] = float(yy.mean())
    if _lowess_problem(pp, 0.75) is None:
        out["calibration_indices"] = calibration_indices(yy, pp)
    else:
        out["calibration_indices"] = {"ici": None, "e50": None, "e90": None}
    out["thresholds"] = threshold_metrics(yy, pp, thresholds)
    undefined = [k for k in SCALAR_METRICS if out[k] is None]
    if out["calibration_indices"]["ici"] is None:
        undefined.append("calibration_indices")
    if undefined:
        warnings.warn(f"undefined performance measures returned as None: {undefined}", UndefinedMetricWarning, stacklevel=2)
    return out
