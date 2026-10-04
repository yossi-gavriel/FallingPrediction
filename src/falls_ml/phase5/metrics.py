"""Phase 5 metrics: the fixed Phase 2-4 implementations for point estimates (AUROC = Mann-Whitney, AP = step-wise average precision, calibration
slope / intercept = GLM y ~ LP, CITL with LP offset) and a fast weighted form of AUROC / AP / Brier for the PAIRED patient bootstrap (multinomial
patient weights; the same resampled patients for both models of a comparison - their predictions are on the same patients)."""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import pandas as pd

from falls_ml.evaluation.metrics import _auroc, _citl, _pr_auc, _slope_intercept
from falls_ml.phase2.evaluate import logloss
from falls_ml.phase5.thresholds import counts, derive, operating_point


def fast_ap(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    return float(_pr_auc(y, np.asarray(p, dtype=float))) if y.sum() else math.nan


def point_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    lp = np.log(p) - np.log1p(-p)
    n, ev = len(y), float(y.sum())
    prev = ev / n if n else math.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a, b, _, _ = _slope_intercept(y, lp)
        c, _ = _citl(y, lp)
    brier = float(np.mean((p - y) ** 2))
    return {"n": n, "events": int(ev), "prevalence": prev, "mean_predicted": float(p.mean()) if n else math.nan, "auroc": float(_auroc(y, p)),
            "ap": float(_pr_auc(y, p)) if ev else math.nan, "brier": brier, "brier_skill": 1 - brier / (prev * (1 - prev)) if 0 < prev < 1 else math.nan,
            "logloss": logloss(y, p), "calibration_intercept": c, "calibration_slope": b, "calibration_glm_intercept": a,
            "oe_ratio": float(ev / p.sum()) if p.sum() > 0 else math.nan}


def calibration_table(y: np.ndarray, p: np.ndarray, groups: int = 10) -> pd.DataFrame:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    order = np.argsort(p, kind="mergesort")
    rows = []
    for g, ix in enumerate(np.array_split(order, groups), start=1):
        if not len(ix):
            continue
        rows.append({"group": g, "n": int(len(ix)), "events": int(y[ix].sum()), "mean_predicted": float(p[ix].mean()), "observed_rate": float(y[ix].mean()),
                     "min_predicted": float(p[ix].min()), "max_predicted": float(p[ix].max())})
    return pd.DataFrame(rows)


# ============================================================================ fast weighted metrics for the bootstrap
class _Sorted:
    def __init__(self, y: np.ndarray, p: np.ndarray):
        self.order = np.argsort(-p, kind="mergesort")
        ps = p[self.order]
        self.ends = np.r_[np.flatnonzero(np.diff(ps) != 0), len(ps) - 1]
        self.ys = y[self.order]

    def stats(self, w: np.ndarray, targets: tuple[float, ...]) -> dict[str, float]:
        ws = w[self.order]
        w1 = np.cumsum(ws * self.ys)[self.ends]
        wk = np.cumsum(ws)[self.ends]
        W1, W = float(w1[-1]), float(wk[-1])
        W0 = W - W1
        g1 = np.diff(np.r_[0.0, w1])
        w0c = wk - w1
        g0 = np.diff(np.r_[0.0, w0c])
        out: dict[str, float] = {}
        if W1 <= 0 or W0 <= 0:
            out.update({"auroc": math.nan, "ap": math.nan})
        else:
            out["auroc"] = float(np.sum(g1 * (W0 - w0c + g0 / 2.0)) / (W1 * W0))
            with np.errstate(divide="ignore", invalid="ignore"):
                out["ap"] = float(np.sum((g1 / W1) * np.where(wk > 0, w1 / wk, 0.0)))
        for t in targets:
            if W1 <= 0:
                out[f"desc_fas_{t:.2f}"] = math.nan
                out[f"desc_flagged_{t:.2f}"] = math.nan
                continue
            j = int(np.flatnonzero(w1 / W1 >= t - 1e-12)[0])
            out[f"desc_fas_{t:.2f}"] = float((wk[j] - w1[j]) / wk[j]) if wk[j] > 0 else math.nan
            out[f"desc_flagged_{t:.2f}"] = float(wk[j] / W) if W > 0 else math.nan
        return out


OP_KEYS = ("sensitivity", "ppv", "false_alert_share", "flagged_share", "fpr", "specificity", "false_alerts_per_10000", "flagged_per_10000", "flagged", "fn",
           "captured_per_10000", "tp", "fp")


def _weighted_counts(w: np.ndarray, y: np.ndarray, flag: np.ndarray) -> dict[str, Any]:
    tp = float((w * (y == 1) * flag).sum())
    fp = float((w * (y == 0) * flag).sum())
    fn = float((w * (y == 1) * ~flag).sum())
    tn = float((w * (y == 0) * ~flag).sum())
    return derive(tp, fp, fn, tn)  # type: ignore[arg-type]


def model_summary(y: np.ndarray, p: np.ndarray, flag: np.ndarray | None, *, targets: tuple[float, ...] = (0.70,)) -> dict[str, Any]:
    out = point_metrics(y, p)
    for t in targets:
        op = operating_point(y, p, t)
        out[f"desc_fas_{t:.2f}"] = op["false_alert_share"]
        out[f"desc_flagged_{t:.2f}"] = op["flagged_share"]
    if flag is not None:
        out.update({f"op_{k}": v for k, v in counts(y, flag).items()})
    return out


def paired_bootstrap(y: np.ndarray, pa: np.ndarray, pb: np.ndarray, fa: np.ndarray | None, fb: np.ndarray | None, *, n_boot: int, seed: int,
                     targets: tuple[float, ...] = (0.70,)) -> dict[str, Any]:
    """Point differences B - A and percentile 95% intervals from the paired patient bootstrap (identical weights for A and B)."""
    y = np.asarray(y, dtype=float)
    pa, pb = np.asarray(pa, dtype=float), np.asarray(pb, dtype=float)
    n = len(y)
    sa, sb = _Sorted(y, pa), _Sorted(y, pb)
    keys_rank = ["auroc", "ap", *[f"desc_fas_{t:.2f}" for t in targets], *[f"desc_flagged_{t:.2f}" for t in targets]]
    full_a, full_b = model_summary(y, pa, fa, targets=targets), model_summary(y, pb, fb, targets=targets)
    point = {}
    for k in [*keys_rank, "brier"]:
        point[k] = full_b[k] - full_a[k]
    if fa is not None and fb is not None:
        for k in OP_KEYS:
            point[f"op_{k}"] = full_b[f"op_{k}"] - full_a[f"op_{k}"]
    rng = np.random.default_rng(seed)
    names = list(point)
    reps = np.full((n_boot, len(names)), np.nan)
    for r in range(n_boot):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        a, b = sa.stats(w, targets), sb.stats(w, targets)
        W = w.sum()
        row = {k: b[k] - a[k] for k in keys_rank}
        row["brier"] = float((w * (pb - y) ** 2).sum() / W - (w * (pa - y) ** 2).sum() / W)
        if fa is not None and fb is not None:
            ca, cb = _weighted_counts(w, y, fa), _weighted_counts(w, y, fb)
            for k in OP_KEYS:
                row[f"op_{k}"] = cb[k] - ca[k]
        reps[r] = [row[k] for k in names]
    out: dict[str, Any] = {"n": n, "events": int(y.sum()), "n_boot": int(n_boot), "seed": int(seed),
                           "bootstrap": "paired patient bootstrap (multinomial weights, identical for both models), percentile 95%"}
    for j, k in enumerate(names):
        col = reps[:, j][np.isfinite(reps[:, j])]
        out[f"delta_{k}"] = point[k]
        out[f"delta_{k}_ci_low"] = float(np.percentile(col, 2.5)) if col.size else math.nan
        out[f"delta_{k}_ci_high"] = float(np.percentile(col, 97.5)) if col.size else math.nan
        out[f"delta_{k}_share_below_0"] = float((col < 0).mean()) if col.size else math.nan
    out["a"] = full_a
    out["b"] = full_b
    return out


def single_bootstrap(y: np.ndarray, p: np.ndarray, flag: np.ndarray | None, *, n_boot: int, seed: int, targets: tuple[float, ...] = (0.70,)) -> dict[str, Any]:
    """Percentile 95% intervals of one model's discrimination, Brier and operational counts (patient bootstrap)."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    n = len(y)
    s = _Sorted(y, p)
    rng = np.random.default_rng(seed)
    names = ["auroc", "ap", "brier", *([f"op_{k}" for k in ("sensitivity", "ppv", "false_alert_share", "flagged_share", "fpr")] if flag is not None else [])]
    reps = np.full((n_boot, len(names)), np.nan)
    for r in range(n_boot):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        st = s.stats(w, targets)
        row = {"auroc": st["auroc"], "ap": st["ap"], "brier": float((w * (p - y) ** 2).sum() / w.sum())}
        if flag is not None:
            c = _weighted_counts(w, y, flag)
            row.update({f"op_{k}": c[k] for k in ("sensitivity", "ppv", "false_alert_share", "flagged_share", "fpr")})
        reps[r] = [row[k] for k in names]
    out = {}
    for j, k in enumerate(names):
        col = reps[:, j][np.isfinite(reps[:, j])]
        out[f"{k}_ci_low"] = float(np.percentile(col, 2.5)) if col.size else math.nan
        out[f"{k}_ci_high"] = float(np.percentile(col, 97.5)) if col.size else math.nan
    return out
