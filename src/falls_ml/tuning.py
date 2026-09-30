"""Bounded hyperparameter search: fit on train, score on validation (architecture §4 step 4; D-19 §1 T_val role).

Only L4 alternative models are tuned (the orchestrator refuses tuning for eFalls-labelled experiments). Each
trial fits on the training partition only and is scored on the validation partition with *uncalibrated*
predictions. The composite objective is a weighted sum of validation metrics; higher is better, so loss-type
metrics (Brier, calibration errors) must carry non-positive weights. Ties select the earliest trial.
"""

from __future__ import annotations

import itertools
import json
import math
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from falls_ml.errors import ConfigError, DatasetValidationError, FallsMLError, LeakageError
from falls_ml.evaluation import metrics as M
from falls_ml.features.spec import FeatureSpec
from falls_ml.logging_utils import get_logger
from falls_ml.seeding import rng_for

if TYPE_CHECKING:
    from falls_ml.pipeline import FittedPipeline

log = get_logger(__name__)

MAX_TRIALS = 500
MAX_GRID_SIZE = 1_000_000
RANDOM_COMPONENT = "tuning_random_search"
NET_BENEFIT_PREFIX = "net_benefit_at_"
#: objective metric name -> sign its weight must have (+1: higher is better, -1: lower is better)
METRIC_DIRECTIONS = {"auroc": 1, "pr_auc": 1, "brier": -1, "calibration_slope_abs_error": -1, "citl_abs": -1}
RESULT_COLUMNS = ["trial", "params_json", "objective", "auroc", "brier", "calibration_slope", "citl", "selected"]


# ---------------------------------------------------------------------- search space
def _validated_space(space: Mapping[str, Sequence[Any]]) -> tuple[list[str], list[list[Any]]]:
    if not isinstance(space, Mapping) or not space:
        raise ConfigError(f"search space must be a non-empty mapping {{param: [values]}}, got {space!r}")
    keys = sorted(space)
    values: list[list[Any]] = []
    for key in keys:
        options = space[key]
        if not isinstance(key, str) or isinstance(options, str) or not isinstance(options, Sequence) or not options:
            raise ConfigError(f"search space {key!r}: expected a non-empty list of values, got {options!r}")
        for value in options:
            scalar = value is None or isinstance(value, (bool, int, str)) or (isinstance(value, float) and math.isfinite(value))
            if not scalar:
                raise ConfigError(f"search space {key!r}: value {value!r} must be null, bool, int, finite float or string")
        encoded = [json.dumps(v) for v in options]
        if len(set(encoded)) != len(encoded):
            raise ConfigError(f"search space {key!r}: duplicated values {list(options)}")
        values.append(list(options))
    return keys, values


def expand_search_space(space: Mapping[str, Sequence[Any]], method: str = "grid", n_iter: int | None = None,
                        seed: int = 0) -> list[dict[str, Any]]:
    """Trial parameter dicts (keys sorted).

    ``grid``: the full cartesian product in ``itertools.product`` order over sorted keys.
    ``random``: ``min(n_iter, grid size)`` distinct grid points drawn without replacement with
    ``rng_for(seed, 'tuning_random_search')``, in draw order. Both are bounded by ``MAX_TRIALS``.
    """
    keys, values = _validated_space(space)
    n_grid = math.prod(len(v) for v in values)
    if n_grid > MAX_GRID_SIZE:
        raise ConfigError(f"search space has {n_grid} combinations (> {MAX_GRID_SIZE}); reduce it")
    if method == "grid":
        if n_grid > MAX_TRIALS:
            raise ConfigError(f"grid search would run {n_grid} trials (> {MAX_TRIALS}); use method 'random' or reduce the space")
        return [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    if method == "random":
        if isinstance(n_iter, bool) or not isinstance(n_iter, (int, np.integer)) or n_iter < 1:
            raise ConfigError(f"random search requires n_iter >= 1, got {n_iter!r}")
        n = min(int(n_iter), n_grid)
        if n > MAX_TRIALS:
            raise ConfigError(f"random search would run {n} trials (> {MAX_TRIALS})")
        drawn = rng_for(seed, RANDOM_COMPONENT).choice(n_grid, size=n, replace=False)
        return [_grid_point(int(i), keys, values) for i in drawn]
    raise ConfigError(f"tuning method must be 'grid' or 'random', got {method!r}")


def _grid_point(i: int, keys: list[str], values: list[list[Any]]) -> dict[str, Any]:
    """The ``i``-th element of ``itertools.product(*values)`` (last key varies fastest)."""
    point: dict[str, Any] = {}
    for key, options in zip(reversed(keys), reversed(values)):
        i, r = divmod(i, len(options))
        point[key] = options[r]
    return {key: point[key] for key in keys}


# ---------------------------------------------------------------------- objective
def _net_benefit_threshold(name: str) -> float | None:
    if not name.startswith(NET_BENEFIT_PREFIX):
        return None
    try:
        t = float(name[len(NET_BENEFIT_PREFIX):])
    except ValueError as exc:
        raise ConfigError(f"objective metric {name!r}: threshold is not a number") from exc
    if not 0.0 < t < 1.0:
        raise ConfigError(f"objective metric {name!r}: threshold must lie in (0, 1)")
    return t


def _validate_weights(weights: Mapping[str, float]) -> None:
    if not isinstance(weights, Mapping) or not weights:
        raise ConfigError("objective weights must be a non-empty mapping {metric: weight}")
    for name, w in weights.items():
        if not isinstance(name, str):
            raise ConfigError(f"objective metric names must be strings, got {name!r}")
        direction = 1 if _net_benefit_threshold(name) is not None else METRIC_DIRECTIONS.get(name)
        if direction is None:
            raise ConfigError(f"unknown objective metric {name!r}; allowed {sorted(METRIC_DIRECTIONS)} or {NET_BENEFIT_PREFIX}<t>")
        if isinstance(w, bool) or not isinstance(w, (int, float)) or not math.isfinite(w):
            raise ConfigError(f"objective weight for {name!r} must be a finite number, got {w!r}")
        if w * direction < 0:
            raise ConfigError(f"objective weight for {name!r} has the wrong sign: higher objective must be better "
                              f"({'non-negative' if direction > 0 else 'non-positive'} weight required)")
    if all(w == 0 for w in weights.values()):
        raise ConfigError("objective weights are all zero")


def composite_objective(metrics: Mapping[str, float], weights: Mapping[str, float]) -> float:
    """``sum(weight * metric)`` over the weighted metrics (higher is better).

    Unknown metric names and metrics absent from ``metrics`` raise ``ConfigError``. A weighted metric that is
    undefined for the data (NaN, e.g. the calibration slope of a constant predictor) makes the objective NaN.
    """
    _validate_weights(weights)
    terms = []
    for name in sorted(weights):
        if name not in metrics or metrics[name] is None:
            raise ConfigError(f"objective metric {name!r} is missing from the evaluated metrics")
        value = float(metrics[name])
        if not math.isfinite(value):
            return math.nan
        terms.append(float(weights[name]) * value)
    return math.fsum(terms)


def _validation_metrics(y: np.ndarray, p: np.ndarray, lp: np.ndarray, weights: Mapping[str, float]) -> dict[str, float]:
    """Metrics recorded per trial plus any extra metric the objective weights (net benefit: positive if p >= t)."""
    _, slope, _, _ = M.calibration_slope_intercept(y, lp)
    citl, _ = M.citl(y, lp)
    values = {"auroc": float(M.auroc(y, p)), "brier": float(M.brier(y, p)), "calibration_slope": float(slope),
              "citl": float(citl), "calibration_slope_abs_error": abs(float(slope) - 1.0), "citl_abs": abs(float(citl))}
    if "pr_auc" in weights:
        values["pr_auc"] = float(M.pr_auc(y, p))
    for name in weights:
        t = _net_benefit_threshold(name)
        if t is not None:
            values[name] = float(M.net_benefit(y, p, t))
    return values


# ---------------------------------------------------------------------- search
def _check_partitions(train_df: pd.DataFrame, validation_df: pd.DataFrame, spec: FeatureSpec) -> np.ndarray:
    y_col = spec.outcome.name
    for label, df in (("train", train_df), ("validation", validation_df)):
        if len(df) == 0:
            raise DatasetValidationError(f"tuning: {label} partition is empty")
        if y_col not in df.columns:
            raise DatasetValidationError(f"tuning: {label} partition lacks outcome column {y_col!r}")
    y = validation_df[y_col]
    if y.isna().any() or not set(pd.unique(y)) <= {0, 1} or y.nunique() < 2:
        raise DatasetValidationError("tuning: validation outcome must be 0/1 with both classes present")
    key = [spec.identifier_columns[0], spec.index_column]
    if set(key) <= set(train_df.columns) and set(key) <= set(validation_df.columns):
        if _row_keys(train_df, key).isin(_row_keys(validation_df, key)).any():
            raise LeakageError("tuning: training and validation partitions share rows")
    return y.to_numpy(dtype=np.int64)


def _row_keys(df: pd.DataFrame, key: list[str]) -> pd.Series:
    return df[key[0]].astype("str") + "|" + df[key[1]].astype("str")


def run_search(train_df: pd.DataFrame, validation_df: pd.DataFrame, *,
               fit_fn: Callable[[pd.DataFrame, dict[str, Any]], FittedPipeline], space: Mapping[str, Sequence[Any]],
               method: str, n_iter: int | None, seed: int, weights: Mapping[str, float],
               spec: FeatureSpec) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate every trial and return ``(best_params, hyperparameter_results)`` (ARTIFACT_SCHEMAS columns).

    ``fit_fn(train_df, params)`` must fit preprocessing and model on the rows it receives; it is only ever
    given ``train_df``. Validation rows are used for uncalibrated prediction and scoring only. Trials whose
    objective is undefined (NaN) are recorded and logged but never selected; if no trial has a defined
    objective the search fails.
    """
    _validate_weights(weights)
    trials = expand_search_space(space, method, n_iter, seed)
    y_val = _check_partitions(train_df, validation_df, spec)
    rows: list[dict[str, Any]] = []
    for trial, params in enumerate(trials):
        # Fresh shallow copies (copy-on-write): a fit_fn or pipeline that adds or overwrites columns cannot carry
        # state into later trials or back into the caller's partitions.
        pipeline = fit_fn(train_df.copy(deep=False), dict(params))
        p = np.asarray(pipeline.predict_proba(validation_df.copy(deep=False)), dtype=np.float64)
        if p.shape != y_val.shape or not np.all((p >= 0.0) & (p <= 1.0)):
            raise FallsMLError(f"tuning trial {trial}: predictions must be {y_val.shape[0]} probabilities in [0, 1]")
        lp = pipeline.linear_predictor(validation_df.copy(deep=False))
        lp = M.logit(p) if lp is None else np.asarray(lp, dtype=np.float64)
        if lp.shape != y_val.shape or not np.isfinite(lp).all():
            raise FallsMLError(f"tuning trial {trial}: linear predictor must hold {y_val.shape[0]} finite values")
        values = _validation_metrics(y_val, p, lp, weights)
        objective = composite_objective(values, weights)
        rows.append({"trial": trial, "params_json": json.dumps(params, sort_keys=True), "objective": objective,
                     **{k: values[k] for k in ("auroc", "brier", "calibration_slope", "citl")}})
        if math.isnan(objective):
            log.warning("tuning_trial_objective_undefined", extra_fields={"trial": trial, "params": params, **values})
        else:
            log.info("tuning_trial", extra_fields={"trial": trial, "params": params, "objective": objective})
    objectives = np.array([r["objective"] for r in rows], dtype=np.float64)
    if np.isnan(objectives).all():
        raise FallsMLError(f"tuning: no trial produced a defined objective for weights {dict(weights)}")
    best = int(np.nanargmax(objectives))  # first maximum among defined objectives: ties select the earliest trial
    results = pd.DataFrame([{**r, "selected": r["trial"] == best} for r in rows], columns=RESULT_COLUMNS)
    results = results.astype({"trial": "int64", "params_json": "str", "objective": "float64", "auroc": "float64",
                              "brier": "float64", "calibration_slope": "float64", "citl": "float64", "selected": "bool"})
    log.info("tuning_selected", extra_fields={"method": method, "n_trials": len(rows), "best_trial": best,
                                              "best_params": trials[best], "objective": rows[best]["objective"]})
    return dict(trials[best]), results
