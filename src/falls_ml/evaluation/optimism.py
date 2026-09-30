"""Harrell's enhanced bootstrap optimism correction (spec §7.2, D-11).

Apparent performance: the model developed on the whole frame, evaluated on the same frame. Replicate ``b``:
resample clusters (patients) with replacement to the same number of clusters, repeat the entire development
process (``fit_fn``) on the resample, and compute optimism_b = performance in the resample - performance of that
model in the original frame. corrected = apparent - mean optimism (B = 25 published protocol, B = 200 Meuhedet).

Summaries per metric use the replicates where optimism is finite: ``optimism_mcse`` = SD(optimism_b)/sqrt(B) and
the Monte Carlo 95% interval of the mean optimism, mean ± t_{B-1, 0.975} · MCSE. Single-class resamples are
recorded failures; more than 10% failures raise :class:`~falls_ml.evaluation.stability.ResamplingError`. A LASSO replicate whose
full-data solve did not converge gets the pre-declared convergence retry (:mod:`falls_ml.evaluation.convergence_retry`) first.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from falls_ml.errors import DegenerateFitError
from falls_ml.errors import ConfigError
from falls_ml.evaluation.metrics import evaluate_many_prepared, parse_metric_name, prepare_inputs
from falls_ml.evaluation.stability import (
    check_positive_int,
    checked_proba,
    is_single_class,
    outcome_array,
    record_failure,
    resample_clusters,
)
from falls_ml.logging_utils import get_logger
from falls_ml.pipeline import FittedPipeline
from falls_ml.seeding import int_seed_for, rng_for

log = get_logger(__name__)

OPTIMISM_COLUMNS = ["metric", "apparent", "mean_optimism", "optimism_mcse", "optimism_ci_low", "optimism_ci_high", "corrected",
                    "n_bootstrap"]
DEFAULT_METRICS = ("auroc", "calibration_slope", "citl", "oe_ratio", "brier")


def _performance(pipeline: FittedPipeline, df: pd.DataFrame, y: np.ndarray, metrics: Sequence[str]) -> np.ndarray:
    """Metric values (NaN if undefined) of ``pipeline`` on ``df``; LP defaults to logit(p) for non-linear models."""
    yy, pp, lp = prepare_inputs(y, checked_proba(pipeline, df), pipeline.linear_predictor(df))
    return np.asarray(evaluate_many_prepared(metrics, yy, pp, lp), dtype=np.float64)


def harrell_optimism(df: pd.DataFrame, fit_fn: Callable[[pd.DataFrame, int], FittedPipeline], *, n_bootstrap: int, seed: int,
                     cluster_column: str, spec: Any, metrics: Sequence[str] = DEFAULT_METRICS,
                     retry_log: list[dict[str, Any]] | None = None) -> pd.DataFrame:
    """Optimism-corrected performance, one row per metric (columns :data:`OPTIMISM_COLUMNS`; spec §7.2, D-11).

    ``retry_log`` (optional) receives one row per convergence retry of a replicate fit."""
    from falls_ml.evaluation.convergence_retry import fit_replicate

    n_bootstrap = check_positive_int(n_bootstrap, "n_bootstrap")
    metrics = list(metrics)
    if not metrics or len(set(metrics)) != len(metrics):
        raise ConfigError("metrics must be a non-empty list without duplicates")
    for name in metrics:
        try:
            parse_metric_name(name)
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
    if cluster_column not in df.columns:
        raise ConfigError(f"cluster column {cluster_column!r} is not in the frame")
    outcome = spec.outcome.name
    y = outcome_array(df, outcome)

    apparent = _performance(fit_fn(df, seed), df, y, metrics)
    optimism = np.full((n_bootstrap, len(metrics)), np.nan)
    failures: list[dict[str, Any]] = []
    for b in range(n_bootstrap):
        sample = resample_clusters(df, cluster_column, rng_for(seed, f"optimism:resample:{b}"))
        y_sample = outcome_array(sample, outcome)
        if is_single_class(y_sample):
            record_failure(failures, n_bootstrap, replicate=b, reason="single outcome class in bootstrap sample", component="optimism")
            continue
        try:
            model = fit_replicate(fit_fn, sample, int_seed_for(seed, f"optimism:fit:{b}"), component="optimism", replicate=b, retry_log=retry_log)
        except DegenerateFitError as exc:     # recorded failure, bounded by MAX_FAILURE_FRACTION (as in stability)
            record_failure(failures, n_bootstrap, replicate=b, reason=f"degenerate fit: {exc}", component="optimism")
            continue
        optimism[b] = _performance(model, sample, y_sample, metrics) - _performance(model, df, y, metrics)

    rows = []
    for j, name in enumerate(metrics):
        values = optimism[:, j][np.isfinite(optimism[:, j])]
        k = values.size
        mean = float(values.mean()) if k else math.nan
        mcse = float(values.std(ddof=1) / math.sqrt(k)) if k > 1 else math.nan
        half = float(student_t.ppf(0.975, k - 1)) * mcse if k > 1 else math.nan
        n_invalid = n_bootstrap - len(failures) - k
        if n_invalid:
            log.warning("optimism_metric_undefined_in_replicates", extra_fields={"metric": name, "n_invalid": n_invalid})
        rows.append({"metric": name, "apparent": float(apparent[j]), "mean_optimism": mean, "optimism_mcse": mcse,
                     "optimism_ci_low": mean - half, "optimism_ci_high": mean + half,
                     "corrected": float(apparent[j]) - mean, "n_bootstrap": k})
    log.info("harrell_optimism_done", extra_fields={"n_requested": n_bootstrap, "n_failed": len(failures), "metrics": metrics})
    return pd.DataFrame(rows, columns=OPTIMISM_COLUMNS)
