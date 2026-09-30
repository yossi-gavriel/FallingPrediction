"""Bootstrap model stability (spec §7.2 "Model stability", D-11, D-13; Riley & Collins 2023).

Each replicate resamples whole clusters (patients) with replacement to the original number of clusters and
repeats the entire development process through ``fit_fn``. Duplicated rows keep their original
``research_id``, so patient-grouped CV folds inside the replicate keep duplicates together (D-11).
Bootstrap models are applied to the ORIGINAL training rows (S3 l.513) and compared with the reference model:

- selection frequency, coefficient mean/median/SD and sign stability per design column (linear models);
  a design column absent from a replicate's design (e.g. a different FP power, D-13) counts as not selected;
  ``raw_feature_selection_frequency`` is the share of replicates in which ANY design column of the raw feature
  was selected (e.g. any FP form);
- per-individual MAPE = mean_b |p_bi - p_i| and classification instability at thresholds (share of
  bootstrap models classifying an individual differently from the reference, "positive if p >= t");
- optionally, permutation importance of every bootstrap model on a deterministic subsample of the rows.

Random numbers come only from ``rng_for(seed, "stability:...")``; replicate ``b`` is identical whatever the
number of replicates. Single-class bootstrap samples are recorded failures; if more than
:data:`MAX_FAILURE_FRACTION` of the replicates fail, :class:`ResamplingError` is raised. A LASSO replicate whose full-data solve
did not converge gets the pre-declared convergence retry (:mod:`falls_ml.evaluation.convergence_retry`) before it can count as failed.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from falls_ml.errors import ConfigError, DatasetValidationError, DegenerateFitError, FallsMLError
from falls_ml.evaluation.importance import (
    NUMERICAL_ZERO,
    PERMUTATION_METRICS,
    permutation_importance_grouped,
    standardized_coefficients,
)
from falls_ml.logging_utils import get_logger
from falls_ml.pipeline import FittedPipeline
from falls_ml.seeding import int_seed_for, rng_for

log = get_logger(__name__)

MAX_FAILURE_FRACTION = 0.10
FEATURE_STABILITY_COLUMNS = [
    "feature", "raw_feature", "n_bootstrap", "n_selected", "selection_frequency", "selection_frequency_mcse",
    "raw_feature_selection_frequency", "coef_mean", "coef_median", "coef_sd", "sign_stability", "permutation_importance_mean",
    "permutation_importance_std", "robust"]
INSTABILITY_COLUMNS = ["threshold", "n_bootstrap", "mean_classification_instability", "max_classification_instability",
                       "mape_mean", "mape_mean_mcse", "mape_median"]
PERMUTATION_KEYS = {"features", "n_repeats", "metric", "max_rows"}


class ResamplingError(FallsMLError):
    """Too many bootstrap replicates could not be used (e.g. single-class resamples)."""


@dataclass(frozen=True)
class StabilityResult:
    """Stability summaries. ``n_bootstrap`` counts the successful replicates summarised (requested = n_bootstrap + n_failed).

    ``bootstrap_coefficients``: one row per successful replicate (index = replicate number), one column per
    design column (0 = not selected); empty for non-linear models. ``mape_individual``: mean_b |p_bi - p_i|
    for every row of the original training frame, in row order.
    """

    feature_stability: pd.DataFrame
    instability: pd.DataFrame
    bootstrap_coefficients: pd.DataFrame
    mape_individual: np.ndarray
    n_bootstrap: int
    n_failed: int
    failures: list[dict[str, Any]] = field(default_factory=list)
    convergence_retries: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------- shared resampling helpers
def check_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ConfigError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def resample_clusters(df: pd.DataFrame, cluster_column: str, rng: np.random.Generator) -> pd.DataFrame:
    """Draw as many clusters as ``df`` holds, with replacement; a cluster drawn k times contributes its rows k times.

    Identifiers are kept unchanged (D-11); the returned frame has a fresh RangeIndex. Missing cluster ids are refused.
    """
    if cluster_column not in df.columns:
        raise ConfigError(f"cluster column {cluster_column!r} is not in the frame")
    if len(df) == 0:
        raise DatasetValidationError("cannot bootstrap an empty frame")
    codes, uniques = pd.factorize(df[cluster_column], sort=True)
    n_missing = int((codes < 0).sum())
    if n_missing:
        raise DatasetValidationError(f"cluster column {cluster_column!r} has {n_missing} missing values; "
                                     "bootstrap clusters must be defined")
    k = len(uniques)
    draws = np.bincount(rng.integers(0, k, size=k), minlength=k)
    return df.iloc[np.repeat(np.arange(len(df)), draws[codes])].reset_index(drop=True)


def is_single_class(y: np.ndarray) -> bool:
    events = int(np.sum(y))
    return events == 0 or events == y.size


def outcome_array(df: pd.DataFrame, outcome: str) -> np.ndarray:
    if outcome not in df.columns:
        raise DatasetValidationError(f"outcome column {outcome!r} is not in the frame")
    y = df[outcome].to_numpy(dtype="float64", na_value=np.nan)
    if not np.isin(y, (0.0, 1.0)).all():
        raise DatasetValidationError(f"outcome column {outcome!r} must contain only 0/1 (no missing values)")
    return y.astype(np.int64)


def record_failure(failures: list[dict[str, Any]], n_requested: int, *, replicate: int, reason: str, component: str) -> None:
    """Append a failure and raise :class:`ResamplingError` once failures exceed MAX_FAILURE_FRACTION of the replicates."""
    failures.append({"replicate": replicate, "reason": reason})
    log.warning("bootstrap_replicate_failed", extra_fields={"component": component, "replicate": replicate, "reason": reason})
    if len(failures) > MAX_FAILURE_FRACTION * n_requested:
        raise ResamplingError(f"{component}: {len(failures)} of {n_requested} bootstrap replicates failed "
                              f"(> {MAX_FAILURE_FRACTION:.0%}); last reason: {reason}")


def checked_proba(pipeline: FittedPipeline, df: pd.DataFrame) -> np.ndarray:
    p = np.asarray(pipeline.predict_proba(df), dtype=np.float64)
    if p.shape != (len(df),) or not np.all(np.isfinite(p)) or np.any((p < 0.0) | (p > 1.0)):
        raise FallsMLError(f"pipeline returned invalid probabilities (shape {p.shape}, expected ({len(df)},) in [0, 1])")
    return p


def _sd(values: np.ndarray) -> float:
    return float(np.std(values, ddof=1)) if values.size > 1 else math.nan


# ---------------------------------------------------------------------- stability
def _coefficients(pipeline: FittedPipeline) -> tuple[pd.Series, dict[str, str]] | None:
    """(coefficient per design column, design column -> raw feature) for linear models; None otherwise."""
    if not pipeline.model.is_linear:
        return None
    table = pipeline.model.get_feature_importance()
    names = table["feature"].astype(str).tolist()
    coefs = pd.Series(table["coefficient"].to_numpy(dtype=np.float64), index=names)
    if not np.all(np.isfinite(coefs.to_numpy())):
        raise FallsMLError(f"linear model {pipeline.model.name} returned non-finite coefficients")
    return coefs, {c: pipeline.preprocessor.raw_feature_of(c) for c in names}


def _raw_features(pipeline: FittedPipeline) -> list[str]:
    pre = pipeline.preprocessor
    return list(dict.fromkeys(pre.raw_feature_of(c) for c in pre.design_columns()))


def _permutation_settings(permutation: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Validated permutation settings; ``features`` None means all spec predictors (resolved by the caller)."""
    if permutation is None:
        return None
    unknown = sorted(set(permutation) - PERMUTATION_KEYS)
    if unknown:
        raise ConfigError(f"permutation settings: unknown keys {unknown} (allowed {sorted(PERMUTATION_KEYS)})")
    features = permutation.get("features")
    features = None if features is None else list(features)
    if features is not None and (not features or len(set(features)) != len(features)):
        raise ConfigError("permutation features must be a non-empty list without duplicates")
    metric = permutation.get("metric", "auroc")
    if metric not in PERMUTATION_METRICS:
        raise ConfigError(f"permutation metric must be one of {sorted(PERMUTATION_METRICS)}, got {metric!r}")
    max_rows = permutation.get("max_rows")
    return {"features": features, "n_repeats": check_positive_int(permutation.get("n_repeats", 5), "permutation n_repeats"),
            "metric": metric, "max_rows": None if max_rows is None else check_positive_int(max_rows, "permutation max_rows")}


def _coefficient_summary(coefs: pd.DataFrame, raw_of: dict[str, str], selection_threshold: float,
                         sign_threshold: float) -> pd.DataFrame:
    rows = []
    b = len(coefs)
    raws = [raw_of[c] for c in coefs.columns]
    selected = coefs.to_numpy(dtype=np.float64) != 0.0  # replicates x design columns
    raw_frequency = {raw: float(selected[:, [j for j, r in enumerate(raws) if r == raw]].any(axis=1).mean())
                     for raw in dict.fromkeys(raws)}
    for column in coefs.columns:
        values = coefs[column].to_numpy(dtype=np.float64)
        nonzero = values[values != 0.0]
        freq = nonzero.size / b
        sign = math.nan if nonzero.size == 0 else max(int((nonzero > 0).sum()), int((nonzero < 0).sum())) / nonzero.size
        rows.append({"feature": column, "raw_feature": raw_of[column], "n_bootstrap": b, "n_selected": int(nonzero.size),
                     "selection_frequency": freq, "selection_frequency_mcse": math.sqrt(freq * (1.0 - freq) / b),
                     "raw_feature_selection_frequency": raw_frequency[raw_of[column]],
                     "coef_mean": float(values.mean()), "coef_median": float(np.median(values)), "coef_sd": _sd(values),
                     "sign_stability": sign,
                     "robust": bool(freq >= selection_threshold and not math.isnan(sign) and sign >= sign_threshold)})
    return pd.DataFrame(rows, columns=[c for c in FEATURE_STABILITY_COLUMNS if not c.startswith("permutation")])


def bootstrap_stability(train_df: pd.DataFrame, fit_fn: Callable[[pd.DataFrame, int], FittedPipeline], *, n_bootstrap: int,
                        seed: int, cluster_column: str, thresholds: Sequence[float], selection_threshold: float = 0.8,
                        sign_stability_threshold: float = 0.9, reference_pipeline: FittedPipeline | None = None,
                        permutation: Mapping[str, Any] | None = None, spec: Any = None,
                        retry_log: list[dict[str, Any]] | None = None) -> StabilityResult:
    """Bootstrap stability of the development process (spec §7.2, D-11).

    ``reference_pipeline`` defaults to ``fit_fn(train_df, seed)``; ``spec`` (outcome name, predictors) defaults to
    the reference pipeline's spec. ``permutation`` = {'features', 'n_repeats', 'metric', 'max_rows'} adds the
    mean/SD over bootstrap models of their permutation importance (on at most ``max_rows`` deterministic rows).

    ``feature_stability`` has one row per design column for linear models and one row per raw feature (coefficient
    and selection measures NaN) otherwise; ``raw_feature_selection_frequency`` is identical on all rows of a raw feature.
    ``robust`` = selection_frequency >= selection_threshold and sign_stability >= sign_stability_threshold (and permutation mean - 2 SD > 0 when available). For non-linear models it is NA, or
    False when the permutation criterion is available and fails. ``retry_log`` (optional) receives one row per convergence retry.
    """
    from falls_ml.evaluation.convergence_retry import fit_replicate

    n_bootstrap = check_positive_int(n_bootstrap, "n_bootstrap")
    thr = np.asarray(list(thresholds), dtype=np.float64)
    if thr.size == 0 or thr.ndim != 1 or not np.all((thr > 0.0) & (thr < 1.0)) or np.unique(thr).size != thr.size:
        raise ConfigError(f"thresholds must be a non-empty sequence of distinct values in (0, 1), got {list(thresholds)}")
    for name, value in (("selection_threshold", selection_threshold), ("sign_stability_threshold", sign_stability_threshold)):
        if not 0.0 <= float(value) <= 1.0:
            raise ConfigError(f"{name} must lie in [0, 1], got {value}")
    if cluster_column not in train_df.columns:
        raise ConfigError(f"cluster column {cluster_column!r} is not in the training frame")

    perm = _permutation_settings(permutation)

    reference = reference_pipeline if reference_pipeline is not None else fit_fn(train_df, seed)
    spec = spec if spec is not None else reference.spec
    if perm is not None and perm["features"] is None:
        perm["features"] = spec.predictor_names()
    outcome = spec.outcome.name
    outcome_array(train_df, outcome)
    n = len(train_df)
    p_ref = checked_proba(reference, train_df)
    ref_positive = p_ref[None, :] >= thr[:, None]

    perm_df = perm_y = None
    if perm is not None:
        rows = np.arange(n)
        if perm["max_rows"] is not None and n > perm["max_rows"]:
            rows = np.sort(rng_for(seed, "stability:permutation_rows").choice(n, size=perm["max_rows"], replace=False))
        perm_df = train_df.iloc[rows]
        perm_y = outcome_array(perm_df, outcome)
        if perm["metric"] == "auroc" and is_single_class(perm_y):
            raise ConfigError("permutation rows contain a single outcome class; AUROC permutation importance is undefined")

    reference_coefs = _coefficients(reference)
    linear = reference_coefs is not None
    columns: dict[str, str] = dict(reference_coefs[1]) if linear else {}
    coef_rows: dict[int, pd.Series] = {}
    perm_means: list[pd.Series] = []
    abs_diff_sum = np.zeros(n)
    disagreements = np.zeros((thr.size, n))
    replicate_mape: list[float] = []
    residue: dict[str, int] = {}  # design column -> replicates with a floating-point-residue coefficient
    failures: list[dict[str, Any]] = []
    retries: list[dict[str, Any]] = []

    for b in range(n_bootstrap):
        sample = resample_clusters(train_df, cluster_column, rng_for(seed, f"stability:resample:{b}"))
        if is_single_class(outcome_array(sample, outcome)):
            record_failure(failures, n_bootstrap, replicate=b, reason="single outcome class in bootstrap sample",
                           component="stability")
            continue
        fit_seed = int_seed_for(seed, f"stability:fit:{b}")
        try:
            model = fit_replicate(fit_fn, sample, fit_seed, component="stability", replicate=b, retry_log=retries)
        except DegenerateFitError as exc:     # e.g. separation created by resampling; recorded, bounded by MAX_FAILURE_FRACTION
            record_failure(failures, n_bootstrap, replicate=b, reason=f"degenerate fit: {exc}", component="stability")
            continue
        p_b = checked_proba(model, train_df)
        diff = np.abs(p_b - p_ref)
        abs_diff_sum += diff
        replicate_mape.append(float(diff.mean()))
        disagreements += (p_b[None, :] >= thr[:, None]) != ref_positive
        if linear:
            extracted = _coefficients(model)
            if extracted is None:
                raise FallsMLError("bootstrap model is non-linear but the reference model is linear")
            coef_rows[b] = extracted[0]
            for column, raw in extracted[1].items():
                columns.setdefault(column, raw)
            standardized = standardized_coefficients(model.model, extracted[0].index.tolist())
            if standardized is not None:
                for column in extracted[0].index[(extracted[0].to_numpy() != 0.0) & (np.abs(standardized) < NUMERICAL_ZERO)]:
                    residue[column] = residue.get(column, 0) + 1
        if perm is not None:
            table = permutation_importance_grouped(model, perm_df, perm_y, features=perm["features"], n_repeats=perm["n_repeats"],
                                                   seed=int_seed_for(seed, f"stability:permutation:{b}"), metric=perm["metric"])
            perm_means.append(table.set_index("feature")["permutation_importance_mean"])
        diagnostics = model.model.fit_diagnostics()
        log.info("stability_replicate", extra_fields={"replicate": b, "n_rows": len(sample), "fit_seed": fit_seed,
                                                     "lambda_star": diagnostics.get("lambda_star")})

    n_ok = len(replicate_mape)
    if retry_log is not None:
        retry_log.extend(retries)
    if residue:
        log.warning("stability_coefficients_at_numerical_zero", extra_fields={
            "replicates_per_design_column": residue, "tolerance": NUMERICAL_ZERO,
            "note": "counted as selected (coefficient != 0), inflating selection frequency; the adapter should return exact zeros"})
    coefs = pd.DataFrame({c: [s.get(c, 0.0) for s in coef_rows.values()] for c in columns},
                         index=pd.Index(list(coef_rows), name="replicate"), dtype=np.float64)
    if linear:
        feature_stability = _coefficient_summary(coefs, columns, selection_threshold, sign_stability_threshold)
    else:
        raws = _raw_features(reference)
        feature_stability = pd.DataFrame({"feature": raws, "raw_feature": raws, "n_bootstrap": n_ok, "n_selected": pd.NA,
                                          "selection_frequency": math.nan, "selection_frequency_mcse": math.nan,
                                          "raw_feature_selection_frequency": math.nan, "coef_mean": math.nan,
                                          "coef_median": math.nan, "coef_sd": math.nan, "sign_stability": math.nan,
                                          "robust": pd.NA})
    feature_stability["n_selected"] = feature_stability["n_selected"].astype("Int64")
    feature_stability["robust"] = feature_stability["robust"].astype("boolean")
    feature_stability["permutation_importance_mean"] = math.nan
    feature_stability["permutation_importance_std"] = math.nan
    if perm_means:
        matrix = pd.concat(perm_means, axis=1)
        mean = feature_stability["raw_feature"].map(matrix.mean(axis=1))
        sd = feature_stability["raw_feature"].map(matrix.std(axis=1, ddof=1))  # NaN with a single replicate
        feature_stability["permutation_importance_mean"] = mean.astype("float64")
        feature_stability["permutation_importance_std"] = sd.astype("float64")
        available = mean.notna() & sd.notna()
        passes = (mean - 2.0 * sd) > 0.0
        feature_stability["robust"] = feature_stability["robust"].where(~available, feature_stability["robust"] & passes)
    feature_stability = feature_stability[FEATURE_STABILITY_COLUMNS]

    mape_individual = abs_diff_sum / n_ok
    instability_i = disagreements / n_ok
    reps = np.asarray(replicate_mape)
    mape_mcse = float(np.std(reps, ddof=1) / math.sqrt(n_ok)) if n_ok > 1 else math.nan
    instability = pd.DataFrame({"threshold": thr, "n_bootstrap": n_ok,
                                "mean_classification_instability": instability_i.mean(axis=1),
                                "max_classification_instability": instability_i.max(axis=1),
                                "mape_mean": float(mape_individual.mean()), "mape_mean_mcse": mape_mcse,
                                "mape_median": float(np.median(mape_individual))})[INSTABILITY_COLUMNS]
    log.info("bootstrap_stability_done", extra_fields={"n_requested": n_bootstrap, "n_ok": n_ok, "n_failed": len(failures),
                                                      "linear": linear, "mape_mean": float(mape_individual.mean()),
                                                      "n_convergence_retries": len(retries),
                                                      "n_converged_on_retry": sum(r["accepted"] for r in retries)})
    return StabilityResult(feature_stability=feature_stability, instability=instability, bootstrap_coefficients=coefs,
                           mape_individual=mape_individual, n_bootstrap=n_ok, n_failed=len(failures), failures=failures,
                           convergence_retries=retries)
