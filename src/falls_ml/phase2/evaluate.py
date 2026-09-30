"""Metrics with fixed implementations (planning/EXPERIMENT_PLAN.md §8) and the paired patient bootstrap for differences between frozen
predictions on the same rows. AP = step-wise average precision (``evaluation.metrics.pr_auc``, sklearn-equivalent, ties grouped)."""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import pandas as pd

from falls_ml.evaluation.metrics import _auroc, _citl, _pr_auc, _slope_intercept, calibration_slope_intercept, citl, grouped_calibration, logit

EPS = 1e-15


def logloss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def cap_tag(c: float) -> str:
    """Column tag of a capacity: 0.1 -> 'top10'."""
    return f"top{round(100 * float(c)):g}"


def top_mask(p: np.ndarray, frac: float) -> np.ndarray:
    """The ceil(frac * n) highest risks; ties keep row order (identical rule for every model on the same rows)."""
    k = max(1, math.ceil(frac * len(p)))
    m = np.zeros(len(p), dtype=bool)
    m[np.argsort(-p, kind="mergesort")[:k]] = True
    return m


def capture(y: np.ndarray, p: np.ndarray, frac: float) -> float:
    ev = y.sum()
    return float(y[top_mask(p, frac)].sum() / ev) if ev else math.nan


def bundle(y: np.ndarray, p: np.ndarray, *, capacities: tuple[float, ...] = (), principal: float | None = None) -> dict[str, Any]:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    n, ev = len(y), float(y.sum())
    prev = ev / n if n else math.nan
    lp = logit(np.clip(p, 1e-12, 1 - 1e-12))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a, b, _, _ = calibration_slope_intercept(y, lp)
        c, _ = citl(y, lp)
    brier = float(np.mean((p - y) ** 2))
    brier_null = prev * (1 - prev)
    out = {"n": n, "events": int(ev), "prevalence": prev, "ap": float(_pr_auc(y, p)), "auroc": float(_auroc(y, p)), "logloss": logloss(y, p),
           "brier": brier, "scaled_brier": 1 - brier / brier_null if brier_null > 0 else math.nan, "citl": c, "cal_intercept": a, "cal_slope": b,
           "oe": float(y.sum() / p.sum()) if p.sum() > 0 else math.nan, "mean_predicted": float(p.mean())}
    for f in capacities:
        out[f"capture@{f:g}"] = capture(y, p, f)
    if principal is not None:
        m = top_mask(p, principal)
        out["ppv@principal"] = float(y[m].mean())
        out["false_alerts@principal"] = int((y[m] == 0).sum())
    return out


def capacity_table(y: np.ndarray, p: np.ndarray, fractions: tuple[float, ...]) -> pd.DataFrame:
    y = np.asarray(y, dtype=int)
    n, ev = len(y), int(y.sum())
    rows = []
    for f in fractions:
        m = top_mask(np.asarray(p, dtype=float), f)
        tp, fp = int(y[m].sum()), int((y[m] == 0).sum())
        fn, tn = ev - tp, (n - ev) - fp
        k = int(m.sum())
        rows.append({"capacity": f, "n_flagged": k, "pct_population": 100.0 * k / n, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                     "sensitivity": tp / ev if ev else math.nan, "specificity": tn / (n - ev) if n - ev else math.nan, "ppv": tp / k if k else math.nan,
                     "npv": tn / (tn + fn) if tn + fn else math.nan, "lift": (tp / k) / (ev / n) if k and ev else math.nan,
                     "false_alerts": fp, "false_alert_pct": 100.0 * fp / k if k else math.nan})
    return pd.DataFrame(rows)


THRESHOLDS = tuple(round(x, 3) for x in (*np.arange(0.01, 0.10, 0.01), *np.arange(0.10, 0.51, 0.025)))


def threshold_table(y: np.ndarray, p: np.ndarray, thresholds: tuple[float, ...] = THRESHOLDS) -> pd.DataFrame:
    y = np.asarray(y, dtype=int)
    n, ev = len(y), int(y.sum())
    rows = []
    for t in thresholds:
        m = np.asarray(p) >= t
        tp, fp = int(y[m].sum()), int((y[m] == 0).sum())
        fn, tn = ev - tp, (n - ev) - fp
        k = int(m.sum())
        rows.append({"threshold": t, "n_flagged": k, "pct_population": 100.0 * k / n, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                     "sensitivity": tp / ev if ev else math.nan, "specificity": tn / (n - ev) if n - ev else math.nan,
                     "ppv": tp / k if k else math.nan, "npv": tn / (tn + fn) if tn + fn else math.nan})
    return pd.DataFrame(rows)


def calibration_table(y: np.ndarray, p: np.ndarray, groups: int) -> pd.DataFrame:
    t = grouped_calibration(np.asarray(y, dtype=float), np.asarray(p, dtype=float), n_groups=groups)
    t["events"] = (t["observed_rate"] * t["n"]).round().astype(int)
    return t


DELTA_METRICS = ("ap", "auroc", "logloss", "brier")


def paired_bootstrap(y: np.ndarray, p_ref: np.ndarray, p_new: np.ndarray, *, n_boot: int, seed: int, principal: float,
                     capacities: tuple[float, ...] = ()) -> dict[str, Any]:
    """Differences new - reference on the same rows (one row per patient -> an ordinary paired patient bootstrap: the SAME resampled patients
    for both models, top-ceil(q x n) membership recomputed in every replicate). Percentile 95% intervals; the share of replicates with a
    difference > 0 is reported as such (not a posterior probability). ``capacities`` adds the difference in recorded events identified
    (true positives) at each capacity (``delta_falls_<topN>``)."""
    y = np.asarray(y, dtype=float)
    a, b = np.asarray(p_ref, dtype=float), np.asarray(p_new, dtype=float)
    caps = tuple(float(c) for c in capacities)

    def stats(yy: np.ndarray, pa: np.ndarray, pb: np.ndarray) -> np.ndarray:
        out = [_pr_auc(yy, pb) - _pr_auc(yy, pa), _auroc(yy, pb) - _auroc(yy, pa), logloss(yy, pb) - logloss(yy, pa),
               float(np.mean((pb - yy) ** 2) - np.mean((pa - yy) ** 2))]
        ev, m = yy.sum(), len(yy)
        # identical rule to top_mask (stable descending order, ceil(q x n) rows): one sort per model, prefix sums for every capacity
        ca = np.cumsum(yy[np.argsort(-pa, kind="mergesort")])
        cb = np.cumsum(yy[np.argsort(-pb, kind="mergesort")])

        def tp(cum: np.ndarray, q: float) -> float:
            return float(cum[max(1, math.ceil(q * m)) - 1])

        k = max(1, math.ceil(principal * m))
        out.append(float((tp(cb, principal) - tp(ca, principal)) / ev) if ev else math.nan)
        out.append(float((k - tp(cb, principal)) - (k - tp(ca, principal))))
        for c in caps:
            out.append(tp(cb, c) - tp(ca, c))
        return np.array(out)

    point = stats(y, a, b)
    rng = np.random.default_rng(seed)
    n = len(y)
    reps = np.empty((n_boot, point.size))
    for r in range(n_boot):
        idx = rng.integers(0, n, n)
        reps[r] = stats(y[idx], a[idx], b[idx])
    names = ["delta_ap", "delta_auroc", "delta_logloss", "delta_brier", "delta_capture_principal", "delta_false_alerts_principal",
             *[f"delta_falls_{cap_tag(c)}" for c in caps]]
    out: dict[str, Any] = {"n_boot": n_boot, "seed": seed}
    for j, nm in enumerate(names):
        col = reps[:, j][np.isfinite(reps[:, j])]
        out[nm] = float(point[j])
        out[f"{nm}_ci_low"] = float(np.percentile(col, 2.5)) if col.size else math.nan
        out[f"{nm}_ci_high"] = float(np.percentile(col, 97.5)) if col.size else math.nan
        out[f"{nm}_share_gt0"] = float(np.mean(col > 0)) if col.size else math.nan
    return out


def calibration_bootstrap(y: np.ndarray, p: np.ndarray, *, n_boot: int, seed: int) -> dict[str, Any]:
    """Percentile 95% intervals (patient bootstrap) of the calibration indices of one frozen prediction vector (Astra F-11):
    calibration slope and intercept (logit P = a + b LP), CITL (LP as offset, slope fixed at 1), O:E = sum(y)/sum(p), Brier and scaled Brier
    = 1 - Brier / [p_eval (1 - p_eval)] with p_eval the prevalence of the evaluated rows."""
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    lp = np.log(p) - np.log1p(-p)

    def stats(yy: np.ndarray, pp: np.ndarray, ll: np.ndarray) -> list[float]:
        a, b, _, _ = _slope_intercept(yy, ll)
        c, _ = _citl(yy, ll)
        prev = yy.mean()
        brier = float(np.mean((pp - yy) ** 2))
        return [b, a, c, float(yy.sum() / pp.sum()) if pp.sum() > 0 else math.nan, brier,
                1 - brier / (prev * (1 - prev)) if 0 < prev < 1 else math.nan]

    names = ["cal_slope", "cal_intercept", "citl", "oe", "brier", "scaled_brier"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        point = stats(y, p, lp)
        rng = np.random.default_rng(seed)
        n = len(y)
        reps = np.array([stats(y[i], p[i], lp[i]) for i in (rng.integers(0, n, n) for _ in range(n_boot))], dtype=float)
    out: dict[str, Any] = {"n_boot": n_boot}
    for j, nm in enumerate(names):
        col = reps[:, j][np.isfinite(reps[:, j])]
        out[nm] = point[j]
        out[f"{nm}_ci_low"] = float(np.percentile(col, 2.5)) if col.size else math.nan
        out[f"{nm}_ci_high"] = float(np.percentile(col, 97.5)) if col.size else math.nan
    return out


def _rcs_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Restricted (natural) cubic spline basis (Harrell), linear beyond the outer knots; columns: x, then k-2 non-linear terms."""
    k = len(knots)
    t = knots
    cols = [x]
    norm = (t[-1] - t[0]) ** 2
    for j in range(k - 2):
        term = (np.clip(x - t[j], 0, None) ** 3 - np.clip(x - t[-2], 0, None) ** 3 * (t[-1] - t[j]) / (t[-1] - t[-2])
                + np.clip(x - t[-1], 0, None) ** 3 * (t[-2] - t[j]) / (t[-1] - t[-2])) / norm
        cols.append(term)
    return np.column_stack(cols)


def smooth_calibration(y: np.ndarray, p: np.ndarray, *, n_knots: int = 4, n_grid: int = 50) -> pd.DataFrame:
    """Smooth calibration curve with a 95% confidence band (Astra F-11): logistic regression of the outcome on a restricted cubic spline of
    the logit of the prediction (knots at the 5/35/65/95th percentiles for 4 knots), evaluated on a grid of predicted risks between the 1st
    and 99th percentile. Aggregate: no row-level value is returned. An empty frame when the fit is not identifiable."""
    import statsmodels.api as sm

    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    lp = np.log(p) - np.log1p(-p)
    probs = {3: (10, 50, 90), 4: (5, 35, 65, 95), 5: (5, 27.5, 50, 72.5, 95)}[int(n_knots)]
    knots = np.percentile(lp, probs)
    if len(np.unique(knots)) < len(knots) or y.sum() < 10 or (len(y) - y.sum()) < 10:
        return pd.DataFrame(columns=["predicted", "observed_smooth", "ci_low", "ci_high"])
    X = sm.add_constant(_rcs_basis(lp, knots), has_constant="add")
    grid_p = np.percentile(p, np.linspace(1, 99, n_grid))
    grid_lp = np.log(grid_p) - np.log1p(-grid_p)
    G = sm.add_constant(_rcs_basis(grid_lp, knots), has_constant="add")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sm.GLM(y, X, family=sm.families.Binomial()).fit()
            fr = res.get_prediction(G).summary_frame(alpha=0.05)
    except Exception:  # noqa: BLE001 - separation / singular information: no curve rather than a wrong one
        return pd.DataFrame(columns=["predicted", "observed_smooth", "ci_low", "ci_high"])
    return pd.DataFrame({"predicted": grid_p, "observed_smooth": fr["mean"].to_numpy(), "ci_low": fr["mean_ci_lower"].to_numpy(),
                         "ci_high": fr["mean_ci_upper"].to_numpy()})
