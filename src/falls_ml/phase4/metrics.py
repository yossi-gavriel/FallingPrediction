"""Temporal-validation metrics with the fixed Phase 2 / 3 implementations (AUROC = Mann-Whitney, AP = step-wise average precision, calibration
slope / intercept = GLM y ~ LP, CITL with LP offset) and percentile 95% intervals from a patient bootstrap (one row per patient; the same
resampled patients for every metric). Capacity rule: the ceil(q x n) highest risks, ties in row order (``phase2.evaluate.top_mask``) - the
frozen Phase 3 rule, never a threshold chosen on 2026."""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import pandas as pd

from falls_ml.evaluation.metrics import _auroc, _citl, _pr_auc, _slope_intercept
from falls_ml.phase2.evaluate import cap_tag, logloss

CI_METRICS = ("auroc", "ap", "brier", "brier_skill_vs_development_prevalence", "logloss", "cal_intercept", "cal_slope", "citl", "oe")


def _point(y: np.ndarray, p: np.ndarray, caps: tuple[float, ...], dev_prev: float) -> dict[str, float]:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    lp = np.log(p) - np.log1p(-p)
    n, ev = len(y), float(y.sum())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a, b, _, _ = _slope_intercept(y, lp)
        c, _ = _citl(y, lp)
    brier = float(np.mean((p - y) ** 2))
    ref_dev = float(np.mean((dev_prev - y) ** 2))
    prev = ev / n if n else math.nan
    out = {"auroc": float(_auroc(y, p)), "ap": float(_pr_auc(y, p)), "brier": brier,
           "brier_skill_vs_development_prevalence": 1 - brier / ref_dev if ref_dev > 0 else math.nan,
           "brier_skill_vs_observed_prevalence": 1 - brier / (prev * (1 - prev)) if 0 < prev < 1 else math.nan,
           "logloss": logloss(y, p), "cal_intercept": a, "cal_slope": b, "citl": c, "oe": float(ev / p.sum()) if p.sum() > 0 else math.nan}
    order = np.argsort(-p, kind="mergesort")
    cum = np.cumsum(y[order])
    for q in caps:
        k = max(1, math.ceil(q * n))
        tp = float(cum[k - 1])
        out[f"capture_{cap_tag(q)}"] = tp / ev if ev else math.nan
        out[f"ppv_{cap_tag(q)}"] = tp / k
        out[f"lift_{cap_tag(q)}"] = (tp / k) / prev if prev else math.nan
        out[f"n_selected_{cap_tag(q)}"] = k
        out[f"falls_captured_{cap_tag(q)}"] = int(tp)
    return out


def metrics_with_ci(y: np.ndarray, p: np.ndarray, *, caps: tuple[float, ...], dev_prev: float, n_boot: int, seed: int) -> dict[str, Any]:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    point = _point(y, p, caps, dev_prev)
    names = [*CI_METRICS, *[f"{m}_{cap_tag(q)}" for q in caps for m in ("capture", "ppv", "lift")]]
    rng = np.random.default_rng(seed)
    n = len(y)
    reps = np.full((n_boot, len(names)), np.nan)
    for r in range(n_boot):
        i = rng.integers(0, n, n)
        if y[i].sum() == 0 or y[i].sum() == n:
            continue
        pr = _point(y[i], p[i], caps, dev_prev)
        reps[r] = [pr[k] for k in names]
    out: dict[str, Any] = {"n": int(n), "events": int(y.sum()), "prevalence": float(y.mean()) if n else math.nan, "mean_predicted": float(p.mean()) if n else math.nan,
                           "n_boot": int(n_boot), "bootstrap": "patient (one row per patient), percentile 95%"}
    out.update(point)
    for j, k in enumerate(names):
        col = reps[:, j][np.isfinite(reps[:, j])]
        out[f"{k}_ci_low"] = float(np.percentile(col, 2.5)) if col.size else math.nan
        out[f"{k}_ci_high"] = float(np.percentile(col, 97.5)) if col.size else math.nan
    return out


def capacity_rows(y: np.ndarray, p: np.ndarray, caps: tuple[float, ...], *, model: str, population: str, assignment: str,
                  mask_from: np.ndarray | None = None, cutoffs: dict[float, float] | None = None) -> list[dict[str, Any]]:
    """Operational table. ``mask_from``: flags computed on the FULL scored population (subgroups keep the same flags - no new threshold).
    ``cutoffs``: frozen absolute 2025 risk cut-offs (secondary view)."""
    from falls_ml.phase2.evaluate import top_mask

    y = np.asarray(y, dtype=int)
    n, ev = len(y), int(y.sum())
    prev = ev / n if n else math.nan
    rows = []
    for q in caps:
        if cutoffs is not None:
            m = np.asarray(p) >= cutoffs[q]
            rule = f"frozen 2025 cut-off {cutoffs[q]:.6f} (top {100 * q:g}% of 2025 TRAIN)"
        elif mask_from is not None:
            m = mask_from[q]
            rule = f"top {100 * q:g}% of the full 2026 scored population (flags unchanged in the subgroup)"
        else:
            m = top_mask(np.asarray(p, dtype=float), q)
            rule = f"top {100 * q:g}% of the evaluated 2026 population"
        k, tp = int(m.sum()), int(y[m].sum())
        fp = k - tp
        rows.append({"model": model, "population": population, "assignment": assignment, "capacity": q, "rule": rule, "n_population": n, "events_population": ev,
                     "n_selected": k, "pct_selected": 100.0 * k / n if n else math.nan, "tp": tp, "fp": fp, "fn": ev - tp, "tn": (n - ev) - fp,
                     "falls_captured": tp, "capture": tp / ev if ev else math.nan, "ppv": tp / k if k else math.nan,
                     "lift": (tp / k) / prev if k and prev else math.nan, "false_alerts": fp})
    return rows


def calibration_rows(y: np.ndarray, p: np.ndarray, groups: int, *, model: str, population: str, assignment: str) -> list[dict[str, Any]]:
    from falls_ml.phase2.evaluate import calibration_table

    t = calibration_table(np.asarray(y, dtype=float), np.asarray(p, dtype=float), groups)
    t.insert(0, "assignment", assignment)
    t.insert(0, "population", population)
    t.insert(0, "model", model)
    return t.to_dict("records")


def pct(x: Any, digits: int = 1) -> str:
    """0.835 -> '83.5%' (management tables); NaN -> '–'."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "–"
    return "–" if not math.isfinite(v) else f"{100 * v:.{digits}f}%"


def interpret(p3: tuple[float, float, float], t26: tuple[float, float, float]) -> str:
    """CI-SUPPORTED only when the two 95% intervals do not overlap (pre-declared, conservative)."""
    _, lo3, hi3 = p3
    _, lo6, hi6 = t26
    if not all(math.isfinite(v) for v in (lo3, hi3, lo6, hi6)):
        return "NO INTERVAL"
    return "CI-SUPPORTED CHANGE" if (hi6 < lo3 or lo6 > hi3) else "WITHIN UNCERTAINTY"


def comparison_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows)
