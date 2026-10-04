"""Operating points: the HIGHEST probability threshold reaching a target sensitivity, the confusion counts at a threshold, threshold tables and
the business-aligned tuning objective.

A patient is flagged when its predicted risk is >= the threshold (ties are flagged together, never split by row order).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def _distinct(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Distinct risks in decreasing order with the cumulative flagged count and true positives when flagging every risk >= that value."""
    order = np.argsort(-p, kind="mergesort")
    ps, ys = p[order], y[order]
    last = np.r_[np.flatnonzero(np.diff(ps) != 0), len(ps) - 1]
    return ps[last], last + 1, np.cumsum(ys)[last]


def counts(y: np.ndarray, flag: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=float)
    flag = np.asarray(flag, dtype=bool)
    tp = int((flag & (y == 1)).sum())
    fp = int((flag & (y == 0)).sum())
    fn = int((~flag & (y == 1)).sum())
    tn = int((~flag & (y == 0)).sum())
    return derive(tp, fp, fn, tn)


def derive(tp: int, fp: int, fn: int, tn: int) -> dict[str, Any]:
    n, P, flagged = tp + fp + fn + tn, tp + fn, tp + fp
    prev = P / n if n else math.nan
    ppv = tp / flagged if flagged else math.nan
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": n, "events": P, "flagged": flagged,
            "sensitivity": tp / P if P else math.nan, "specificity": tn / (tn + fp) if (tn + fp) else math.nan, "ppv": ppv,
            "npv": tn / (tn + fn) if (tn + fn) else math.nan, "false_alert_share": fp / flagged if flagged else math.nan,
            "fpr": fp / (fp + tn) if (fp + tn) else math.nan, "flagged_share": flagged / n if n else math.nan,
            "flagged_per_capture": flagged / tp if tp else math.nan, "lift": ppv / prev if prev and not math.isnan(ppv) else math.nan,
            "false_alerts_per_10000": 1e4 * fp / n if n else math.nan, "flagged_per_10000": 1e4 * flagged / n if n else math.nan,
            "captured_per_10000": 1e4 * tp / n if n else math.nan}


def operating_point(y: np.ndarray, p: np.ndarray, target: float) -> dict[str, Any]:
    """The highest threshold whose flagged set reaches sensitivity >= target (infeasible only without events or with non-finite risks)."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    P = float(y.sum())
    if P == 0 or not np.isfinite(p).all() or not len(p):
        return {"threshold": math.nan, "feasible": False, "degenerate": True, "target": target, **derive(0, 0, int(P), int(len(y) - P))}
    vals, k, tp = _distinct(y, p)
    hit = np.flatnonzero(tp / P >= target - 1e-12)
    j = int(hit[0])
    t = float(vals[j])
    out = counts(y, p >= t)
    out.update({"threshold": t, "feasible": True, "degenerate": bool(out["flagged"] == len(p)), "target": target})
    return out


def thresholds_for(y: np.ndarray, p: np.ndarray, targets: list[float]) -> dict[str, float]:
    return {f"{t:.2f}": operating_point(y, p, t)["threshold"] for t in targets}


def threshold_table(y: np.ndarray, p: np.ndarray, *, grid_step: float | None = None) -> pd.DataFrame:
    """Every distinct threshold (local, exhaustive) or, with ``grid_step``, the thresholds at which the flagged share first reaches each multiple
    of the step (shareable: each row adds at least step x n patients, never one patient)."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    vals, k, tp = _distinct(y, p)
    n, P = len(y), float(y.sum())
    if grid_step:
        want = np.arange(grid_step, 1.0 + 1e-9, grid_step)
        idx = sorted({int(np.searchsorted(k / n, w - 1e-12)) for w in want if np.searchsorted(k / n, w - 1e-12) < len(k)})
        vals, k, tp = vals[idx], k[idx], tp[idx]
    fp = k - tp
    rows = [{"threshold": float(t), **derive(int(a), int(b), int(P - a), int(n - P - b))} for t, a, b in zip(vals, tp, fp)]
    return pd.DataFrame(rows)


def objective(y: np.ndarray, p: np.ndarray, *, target: float, complexity: float, tie: float) -> dict[str, Any]:
    """The pre-declared tuning objective on INNER out-of-fold predictions: minimise the share flagged at the highest threshold reaching the
    target sensitivity; ties (at ``tie`` resolution) -> lower false-alert share -> lower share flagged -> higher AP -> lower Brier -> simpler."""
    from falls_ml.phase5.metrics import fast_ap

    op = operating_point(y, p, target)
    ap = fast_ap(y, p)
    brier = float(np.mean((p - y) ** 2)) if len(p) else math.nan
    big = 9.0e9                                       # an infeasible / undefined component sorts last (finite: JSON-safe)
    fin = (lambda v: float(v) if v is not None and math.isfinite(v) else big)
    if not op["feasible"]:
        key = (big, big, big, big, big, fin(complexity))
    else:
        r = (lambda v: round(v / tie) * tie if math.isfinite(v) else big)
        key = (r(op["flagged_share"]), r(op["false_alert_share"]), fin(op["flagged_share"]), fin(-ap), fin(brier), fin(complexity))
    return {"key": key, "flagged_share": op["flagged_share"], "false_alert_share": op["false_alert_share"], "ppv": op["ppv"],
            "sensitivity": op["sensitivity"], "fpr": op["fpr"], "threshold": op["threshold"], "feasible": op["feasible"], "degenerate": op["degenerate"],
            "ap": ap, "brier": brier, "complexity": complexity}
