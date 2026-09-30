"""Percentile bootstrap CIs for performance measures and paired model differences (D-11, D-21).

Resampling is patient-level (or cluster-level when ``cluster`` is given: whole clusters are drawn with
replacement, so every row of a drawn cluster enters the resample). Random numbers come only from
``rng_for(seed, component)``. Resamples containing a single outcome class are skipped and counted; a
replicate whose measure is otherwise undefined (e.g. a non-existent calibration-slope MLE) is invalid for
that measure only. If fewer than ``MIN_VALID_FRACTION`` of the requested replicates are valid, the CI is
``None`` and a warning is raised.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np
import pandas as pd

from falls_ml.evaluation.metrics import (
    SCALAR_METRICS,
    UndefinedMetricWarning,
    evaluate_many_prepared,
    evaluate_prepared,
    parse_metric_name,
    prepare_inputs,
)
from falls_ml.logging_utils import get_logger
from falls_ml.seeding import rng_for

log = get_logger(__name__)

MIN_VALID_FRACTION = 0.8


def _cluster_codes(cluster: Any, n_rows: int) -> np.ndarray:
    """Integer cluster codes (order of first appearance); missing cluster ids are rejected, never pooled."""
    arr = np.asarray(cluster)
    if arr.ndim != 1 or arr.size != n_rows:
        raise ValueError(f"cluster must be a 1-D array of length {n_rows}, got shape {arr.shape}")
    if pd.isna(arr).any():
        raise ValueError(f"cluster contains {int(pd.isna(arr).sum())} missing ids")
    return pd.factorize(arr)[0]


def _resample_indices(n_rows: int, n: int, rng: np.random.Generator, codes: np.ndarray | None) -> Iterator[np.ndarray]:
    """Yield ``n`` row-index arrays; a cluster drawn k times contributes its rows k times."""
    if codes is None:
        for _ in range(n):
            yield rng.integers(0, n_rows, size=n_rows)
        return
    n_clusters = int(codes.max()) + 1
    rows = np.arange(n_rows)
    for _ in range(n):
        draws = np.bincount(rng.integers(0, n_clusters, size=n_clusters), minlength=n_clusters)
        yield np.repeat(rows, draws[codes])


def _single_class(y: np.ndarray) -> bool:
    events = y.sum()
    return events == 0 or events == y.size


def _check_settings(n: int, alpha: float = 0.05) -> None:
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n (number of bootstrap replicates) must be a positive integer")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")


def _finite_or_none(x: float) -> float | None:
    return float(x) if math.isfinite(x) else None


def _percentile_ci(values: np.ndarray, n: int, alpha: float, label: str, component: str) -> tuple[float | None, float | None]:
    valid = values[np.isfinite(values)]
    if valid.size < MIN_VALID_FRACTION * n:
        message = f"bootstrap CI for {label} not reported: {valid.size}/{n} valid replicates (< {MIN_VALID_FRACTION:.0%})"
        log.warning("bootstrap_ci_insufficient_valid", extra_fields={"metric": label, "component": component,
                                                                    "n_valid": int(valid.size), "n_requested": n})
        warnings.warn(message, UndefinedMetricWarning, stacklevel=3)
        return None, None
    low, high = np.quantile(valid, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)


def bootstrap_replicates(y: Any, p: Any, lp: Any | None = None, *, n: int, seed: int, component: str,
                         cluster: Any | None = None, metrics: Sequence[str] = SCALAR_METRICS) -> pd.DataFrame:
    """Replicate estimates (one row per requested replicate, one column per metric; NaN = invalid).

    Column ``single_class`` flags skipped single-class resamples.
    """
    _check_settings(n)
    names = list(dict.fromkeys(metrics))
    for name in names:
        parse_metric_name(name)
    yy, pp, ll = prepare_inputs(y, p, lp)
    out = np.full((n, len(names)), np.nan)
    single = np.zeros(n, dtype=bool)
    codes = None if cluster is None else _cluster_codes(cluster, yy.size)
    for b, idx in enumerate(_resample_indices(yy.size, n, rng_for(seed, component), codes)):
        yb = yy[idx]
        if _single_class(yb):
            single[b] = True
            continue
        pb, lb = pp[idx], ll[idx]
        out[b] = evaluate_many_prepared(names, yb, pb, lb)
    frame = pd.DataFrame(out, columns=names)
    frame["single_class"] = single
    return frame


def bootstrap_metric_cis(y: Any, p: Any, lp: Any | None = None, *, n: int, seed: int, component: str,
                         cluster: Any | None = None, metrics: Sequence[str] = SCALAR_METRICS,
                         alpha: float = 0.05) -> dict[str, dict[str, float | None]]:
    """Point estimate (full data) and percentile ``1 - alpha`` bootstrap CI for each metric (D-21).

    Returns ``{metric: {"estimate", "ci_low", "ci_high"}}``; undefined values are ``None``.
    """
    _check_settings(n, alpha)
    yy, pp, ll = prepare_inputs(y, p, lp)
    reps = bootstrap_replicates(yy, pp, ll, n=n, seed=seed, component=component, cluster=cluster, metrics=metrics)
    names = list(dict.fromkeys(metrics))
    result: dict[str, dict[str, float | None]] = {}
    for name, estimate in zip(names, evaluate_many_prepared(names, yy, pp, ll)):
        low, high = _percentile_ci(reps[name].to_numpy(), n, alpha, name, component)
        result[name] = {"estimate": _finite_or_none(estimate), "ci_low": low, "ci_high": high}
    log.info("bootstrap_cis", extra_fields={
        "component": component, "n_requested": n, "clustered": cluster is not None,
        "n_single_class_skipped": int(reps["single_class"].sum()),
        "n_valid": {name: int(np.isfinite(reps[name]).sum()) for name in result}})
    return result


def paired_bootstrap_difference(y: Any, p_a: Any, p_b: Any, *, metric: str, n: int, seed: int, component: str,
                                lp_a: Any | None = None, lp_b: Any | None = None, cluster: Any | None = None,
                                alpha: float = 0.05) -> dict[str, float | int | None]:
    """Difference metric(A) - metric(B) with a percentile CI from identical resamples for both models.

    ``metric`` is a scalar metric name or ``net_benefit@<threshold>``. Returns
    ``{"estimate", "ci_low", "ci_high", "n_valid"}``.
    """
    _check_settings(n, alpha)
    parse_metric_name(metric)
    yy, pa, la = prepare_inputs(y, p_a, lp_a)
    _, pb, lb = prepare_inputs(yy, p_b, lp_b)
    diffs = np.full(n, np.nan)
    n_single = 0
    codes = None if cluster is None else _cluster_codes(cluster, yy.size)
    for b, idx in enumerate(_resample_indices(yy.size, n, rng_for(seed, component), codes)):
        yr = yy[idx]
        if _single_class(yr):
            n_single += 1
            continue
        diffs[b] = evaluate_prepared(metric, yr, pa[idx], la[idx]) - evaluate_prepared(metric, yr, pb[idx], lb[idx])
    estimate = evaluate_prepared(metric, yy, pa, la) - evaluate_prepared(metric, yy, pb, lb)
    low, high = _percentile_ci(diffs, n, alpha, f"{metric} difference", component)
    n_valid = int(np.isfinite(diffs).sum())
    log.info("paired_bootstrap_difference", extra_fields={"component": component, "metric": metric, "n_requested": n,
                                                           "n_single_class_skipped": n_single, "n_valid": n_valid})
    return {"estimate": _finite_or_none(estimate), "ci_low": low, "ci_high": high, "n_valid": n_valid}
