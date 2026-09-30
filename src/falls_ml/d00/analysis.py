"""Aggregate computations of the D-00 sensitivity report: risk concentration, paired differences on identical test rows, subset performance,
learning-curve increments and the pre-declared plateau rule. Inputs are outcome / predicted-risk arrays held in memory; every output is an
aggregate with small cells suppressed."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from falls_ml.evaluation.metrics import auroc, brier, pr_auc
from falls_ml.seeding import rng_for

SUPPRESSED = "<min_cell"


def _r(v: Any, d: int = 4) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return round(f, d) if math.isfinite(f) else None


def cell(n: int, min_cell: int) -> int | str:
    n = int(n)
    return n if n == 0 or n >= min_cell else SUPPRESSED


def top_mask(p: np.ndarray, frac: float) -> np.ndarray:
    """The ceil(frac * n) highest predicted risks (stable order: ties keep the row order, identical for every model on the same rows)."""
    k = max(1, math.ceil(frac * len(p)))
    order = np.argsort(-p, kind="mergesort")
    m = np.zeros(len(p), dtype=bool)
    m[order[:k]] = True
    return m


def risk_concentration(y: np.ndarray, p: np.ndarray, *, fractions: tuple[float, ...], min_cell: int, n_boot: int = 1000, seed: int = 20260923,
                       label: str = "") -> pd.DataFrame:
    """Highest-risk top x% of the evaluation partition: N, % of population, observed falls, % of all falls captured, observed prevalence,
    lift over the population prevalence; 95% intervals from a row bootstrap (percentile). Exploratory operational characterisation - no
    threshold is chosen."""
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=float)
    n, events = len(y), int(y.sum())
    base = events / n if n else float("nan")
    g = rng_for(seed, f"risk_concentration::{label}")
    boots: dict[float, list[tuple[float, float, float]]] = {f: [] for f in fractions}
    for _ in range(n_boot):
        i = g.integers(0, n, n)
        yy, pp = y[i], p[i]
        ev = int(yy.sum())
        if ev == 0:
            continue
        for f in fractions:
            m = top_mask(pp, f)
            prev = float(yy[m].mean())
            boots[f].append((float(yy[m].sum()) / ev, prev, prev / (ev / n)))
    rows = []
    for f in fractions:
        m = top_mask(p, f)
        k, cap = int(m.sum()), int(y[m].sum())
        b = np.array(boots[f]) if boots[f] else np.empty((0, 3))
        small = k < min_cell or 0 < cap < min_cell or 0 < k - cap < min_cell
        q = (lambda j, lo: None if small or not len(b) else float(np.quantile(b[:, j], lo)))
        rows.append({"model": label, "top_pct": _r(100 * f, 1), "n_selected": cell(k, min_cell), "pct_of_population": _r(100.0 * k / n if n else None, 2),
                     "falls_in_group": cell(cap, min_cell),
                     "pct_of_all_falls_captured": None if small else _r(100.0 * cap / events if events else None, 2),
                     "capture_ci_low": _r(100 * q(0, 0.025), 2) if q(0, 0.025) is not None else None,
                     "capture_ci_high": _r(100 * q(0, 0.975), 2) if q(0, 0.975) is not None else None,
                     "observed_prevalence": None if small else _r(cap / k), "prevalence_ci_low": _r(q(1, 0.025)), "prevalence_ci_high": _r(q(1, 0.975)),
                     "lift": None if small else _r((cap / k) / base if base else None, 3), "lift_ci_low": _r(q(2, 0.025), 3), "lift_ci_high": _r(q(2, 0.975), 3),
                     "population_n": n, "population_falls": cell(events, min_cell), "population_prevalence": _r(base), "n_boot": n_boot})
    return pd.DataFrame(rows)


def paired_comparison(y: np.ndarray, pa: np.ndarray, pb: np.ndarray, *, fractions: tuple[float, ...] = (0.05, 0.10, 0.20), n_boot: int = 1000,
                      seed: int = 20260923, label: str = "") -> dict[str, Any]:
    """B minus A on IDENTICAL rows: AUROC, PR-AUC, Brier and the share of all falls captured in the top x% of each model's own ranking;
    95% percentile intervals from one shared row bootstrap (the pairing is kept inside every replicate)."""
    y = np.asarray(y, dtype=np.int64)
    pa, pb = np.asarray(pa, dtype=float), np.asarray(pb, dtype=float)

    def stats(yy: np.ndarray, a: np.ndarray, b: np.ndarray) -> dict[str, float]:
        ev = yy.sum()
        out = {"auroc": auroc(yy, b) - auroc(yy, a), "pr_auc": pr_auc(yy, b) - pr_auc(yy, a), "brier": brier(yy, b) - brier(yy, a)}
        for f in fractions:
            out[f"capture_top{int(round(100 * f))}"] = 100.0 * (yy[top_mask(b, f)].sum() - yy[top_mask(a, f)].sum()) / ev if ev else float("nan")
        return out

    est = stats(y, pa, pb)
    g = rng_for(seed, f"paired::{label}")
    bs: dict[str, list[float]] = {k: [] for k in est}
    n = len(y)
    for _ in range(n_boot):
        i = g.integers(0, n, n)
        if y[i].min() == y[i].max():
            continue
        for k, v in stats(y[i], pa[i], pb[i]).items():
            bs[k].append(v)
    return {k: {"estimate": _r(v, 4), "ci_low": _r(np.quantile(bs[k], 0.025), 4) if bs[k] else None,
                "ci_high": _r(np.quantile(bs[k], 0.975), 4) if bs[k] else None} for k, v in est.items()} | {"n_rows": n, "n_boot": n_boot}


def subset_performance(y: np.ndarray, p: np.ndarray, mask: np.ndarray, *, min_cell: int, min_events: int = 10) -> dict[str, Any]:
    """Descriptive performance of one model on a subset of its own test rows (AUROC with a Hanley-McNeil interval when both classes have
    >= min_events rows). Aggregate only."""
    from falls_ml.eda.common import auroc as auc_ci

    m = np.asarray(mask, dtype=bool)
    n, e = int(m.sum()), int(np.asarray(y)[m].sum())
    out: dict[str, Any] = {"n": cell(n, min_cell), "events": cell(e, min_cell),
                           "prevalence": None if (0 < e < min_cell or 0 < n - e < min_cell) else _r(e / n if n else None),
                           "mean_predicted": _r(float(np.mean(np.asarray(p)[m])) if n else None)}
    if e >= min_events and n - e >= min_events:
        a, lo, hi = auc_ci(np.asarray(p)[m], np.asarray(y)[m])
        out.update({"auroc": _r(a, 3), "auroc_ci_low": _r(lo, 3), "auroc_ci_high": _r(hi, 3)})
    else:
        out["auroc"] = None
        out["note"] = f"AUROC not reported: fewer than {min_events} events or non-events"
    return out


def learning_increments(y_val: np.ndarray, preds: dict[float, np.ndarray], intervals: list[tuple[float, float]], *, n_boot: int = 1000,
                        seed: int = 20260923, label: str = "") -> list[dict[str, Any]]:
    """Validation-AUROC gain between two training fractions on the SAME validation rows (paired bootstrap interval)."""
    y = np.asarray(y_val, dtype=np.int64)
    out = []
    g = rng_for(seed, f"learning_increments::{label}")
    for a, b in intervals:
        if a not in preds or b not in preds:
            out.append({"from_fraction": a, "to_fraction": b, "delta_validation_auroc": None, "ci_low": None, "ci_high": None, "note": "a fit is missing"})
            continue
        pa, pb = preds[a], preds[b]
        d = auroc(y, pb) - auroc(y, pa)
        bs = []
        for _ in range(n_boot):
            i = g.integers(0, len(y), len(y))
            if y[i].min() != y[i].max():
                bs.append(auroc(y[i], pb[i]) - auroc(y[i], pa[i]))
        out.append({"from_fraction": a, "to_fraction": b, "delta_validation_auroc": _r(d, 4), "ci_low": _r(np.quantile(bs, 0.025), 4) if bs else None,
                    "ci_high": _r(np.quantile(bs, 0.975), 4) if bs else None, "paired": True})
    return out


def plateau_statement(increments: list[dict[str, Any]], *, max_gain: float, name: str) -> dict[str, Any]:
    """The pre-declared plateau rule (configs/meuhedet/d00_sensitivity.yaml): 'approaching a plateau' only if the last gain (80->100%) is below
    max_gain, its 95% interval includes 0, and it is not larger than the previous gain. Never 'more data will not help'."""
    by = {(round(r["from_fraction"], 2), round(r["to_fraction"], 2)): r for r in increments}
    last, prev = by.get((0.8, 1.0)), by.get((0.6, 0.8))
    caveat = ("One snapshot, one internal split: this describes adding more patients drawn from the same snapshot distribution. It cannot "
              "establish that more data (other dates, other populations, richer records) would not help.")
    if not last or last.get("delta_validation_auroc") is None or last.get("ci_low") is None:
        return {"model": name, "plateau_suggested": None, "statement": f"{name}: the increments could not be estimated. " + caveat}
    d, lo, hi = last["delta_validation_auroc"], last["ci_low"], last["ci_high"]
    small = abs(d) < max_gain and lo <= 0 <= hi
    shrinking = prev is None or prev.get("delta_validation_auroc") is None or d <= prev["delta_validation_auroc"] + 1e-12
    if small and shrinking:
        s = (f"{name}: going from 80% to 100% of the training patients changed validation AUROC by {d:+.4f} (95% CI {lo:+.4f} to {hi:+.4f}), "
             f"below the pre-declared {max_gain:.2f} and not larger than the previous step. The curve appears to be approaching a plateau, suggesting that "
             "richer predictors may currently provide more value than simply adding more patients drawn from the same snapshot distribution.")
    else:
        s = (f"{name}: the 80% to 100% step changed validation AUROC by {d:+.4f} (95% CI {lo:+.4f} to {hi:+.4f}); the pre-declared plateau rule "
             f"(gain < {max_gain:.2f}, interval including 0, not larger than the previous step) is not met, so no plateau is claimed.")
    return {"model": name, "plateau_suggested": bool(small and shrinking), "statement": s + " " + caveat}


def share_of_gain(y: np.ndarray, p_base: np.ndarray, p_full: np.ndarray, p_ablated: np.ndarray, *, n_boot: int = 1000, seed: int = 20260923,
                  label: str = "") -> dict[str, Any]:
    """Share of the base -> full AUROC gain that is lost when the full model is refitted without some predictors, on IDENTICAL rows:
    (AUROC_full - AUROC_ablated) / (AUROC_full - AUROC_base), with a joint percentile bootstrap (one resample for all three models).
    Reported only when the gain's own 95% interval excludes 0 (a ratio over a gain that may be zero is not interpretable)."""
    y = np.asarray(y, dtype=np.int64)
    gain = auroc(y, p_full) - auroc(y, p_base)
    loss = auroc(y, p_full) - auroc(y, p_ablated)
    g = rng_for(seed, f"share_of_gain::{label}")
    gains, shares = [], []
    for _ in range(n_boot):
        i = g.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        ga = auroc(y[i], p_full[i]) - auroc(y[i], p_base[i])
        gains.append(ga)
        if ga > 0:
            shares.append((auroc(y[i], p_full[i]) - auroc(y[i], p_ablated[i])) / ga)
    glo, ghi = (float(np.quantile(gains, 0.025)), float(np.quantile(gains, 0.975))) if gains else (None, None)
    defined = glo is not None and glo > 0 and gain > 0
    return {"gain": _r(gain, 4), "gain_ci": [_r(glo, 4), _r(ghi, 4)], "loss": _r(loss, 4), "defined": bool(defined),
            "share": _r(loss / gain, 3) if defined else None,
            "share_ci": [_r(np.quantile(shares, 0.025), 3), _r(np.quantile(shares, 0.975), 3)] if defined and shares else [None, None],
            "rule": "reported only when the base -> full gain has a 95% interval above 0; joint row bootstrap over identical test rows"}


def two_proportions(e1: int, n1: int, e2: int, n2: int) -> dict[str, Any]:
    """Difference of two proportions (1 minus 2) with a Wald 95% interval - descriptive only."""
    if not n1 or not n2:
        return {"difference": None, "ci": [None, None], "differs": None}
    p1, p2 = e1 / n1, e2 / n2
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    lo, hi = p1 - p2 - 1.96 * se, p1 - p2 + 1.96 * se
    return {"difference": _r(p1 - p2, 5), "ci": [_r(lo, 5), _r(hi, 5)], "differs": bool(lo > 0 or hi < 0)}
