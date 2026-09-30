"""Coefficient tables and grouped permutation importance (architecture §4 step 7; ARTIFACT_SCHEMAS
``coefficients.csv`` and ``feature_importance.csv``; spec D-10, D-14).

Rules
- Raw coefficient magnitudes are never compared across differently scaled design columns: ``rank`` in the
  coefficient table uses |standardized coefficient| and is NA when the model reports no standardized scale.
- Permutation importance is computed for every model at the level of the raw (spec) feature: the raw column is
  permuted and the full fitted pipeline (preprocessing + model [+ calibrator]) is re-run, so all design columns
  derived from one feature (indicators, FP terms, missing flags) move together. Positive = important.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from falls_ml.errors import ConfigError, FallsMLError
from falls_ml.evaluation.metrics import auroc, brier, prepare_inputs
from falls_ml.logging_utils import get_logger
from falls_ml.pipeline import FittedPipeline
from falls_ml.seeding import rng_for

log = get_logger(__name__)

COEFFICIENT_COLUMNS = ["design_column", "feature", "coefficient", "standardized_coefficient", "odds_ratio", "abs_coefficient",
                       "selected", "direction", "rank", "relative_to_reference"]
PERMUTATION_COLUMNS = ["feature", "permutation_importance_mean", "permutation_importance_std", "baseline_metric", "metric"]
FEATURE_IMPORTANCE_COLUMNS = ["feature", "model", "importance_rank", "coefficient", "odds_ratio", "permutation_importance_mean",
                              "permutation_importance_std", "shap_mean_abs", "selection_frequency"]
#: metric -> (function, +1 if higher is better else -1)
PERMUTATION_METRICS = {"auroc": (auroc, 1.0), "brier": (brier, -1.0)}
#: a non-zero coefficient with |standardized coefficient| below this is floating-point residue (e.g. at the KKT
#: boundary of exactly collinear all-levels indicators, D-10). It is logged, never re-coded: selected = coefficient != 0.
NUMERICAL_ZERO = 1e-12


class ImportanceError(FallsMLError):
    """Feature importance cannot be computed (undefined baseline metric, inconsistent model output)."""


# ---------------------------------------------------------------------- coefficients
def standardized_coefficients(model: Any, design: list[str]) -> np.ndarray | None:
    """Standardized coefficients aligned to ``design`` from ``fit_diagnostics()`` (D-14), or None if not reported."""
    reported = model.fit_diagnostics().get("standardized_coefficients")
    if reported is None:
        return None
    if isinstance(reported, Mapping):
        reported = pd.Series(reported, dtype=np.float64)
    if isinstance(reported, pd.Series):
        if set(map(str, reported.index)) != set(design) or len(reported) != len(design):
            raise ImportanceError(f"{model.name}: standardized coefficients do not match the design columns")
        values = reported.set_axis(list(map(str, reported.index))).reindex(design).to_numpy(dtype=np.float64)
    else:
        values = np.asarray(reported, dtype=np.float64)
        if values.shape != (len(design),):
            raise ImportanceError(f"{model.name}: {values.shape} standardized coefficients for {len(design)} design columns")
    if not np.all(np.isfinite(values)):
        raise ImportanceError(f"{model.name}: non-finite standardized coefficients")
    return values


def empty_coefficient_table() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in COEFFICIENT_COLUMNS})


def coefficient_table(pipeline: FittedPipeline) -> pd.DataFrame:
    """``coefficients.csv`` rows (one per design column) for linear models; header-only frame otherwise.

    Coefficients are on the design scale (``relative_to_reference`` False). ``rank`` orders |standardized
    coefficient| (ties share the smallest rank) and is NA when no standardized scale is reported.
    """
    model = pipeline.model
    if not model.is_linear:
        return empty_coefficient_table()
    native = model.get_feature_importance()
    design = native["feature"].astype(str).tolist()
    coef = native["coefficient"].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(coef)):
        raise ImportanceError(f"{model.name}: non-finite coefficients")
    standardized = standardized_coefficients(model, design)
    if standardized is None:
        std_col = np.full(len(design), np.nan)
        rank = pd.array([pd.NA] * len(design), dtype="Int64")
    else:
        std_col = standardized
        rank = pd.Series(np.abs(standardized)).rank(ascending=False, method="min").astype("Int64").array
        residue = [c for c, b, z in zip(design, coef, standardized) if b != 0.0 and abs(z) < NUMERICAL_ZERO]
        if residue:
            log.warning("coefficients_at_numerical_zero", extra_fields={
                "model": model.name, "design_columns": residue, "tolerance": NUMERICAL_ZERO,
                "note": "reported as selected (coefficient != 0); the adapter should return exact zeros"})
    return pd.DataFrame({
        "design_column": design,
        "feature": [pipeline.preprocessor.raw_feature_of(c) for c in design],
        "coefficient": coef,
        "standardized_coefficient": std_col,
        "odds_ratio": np.exp(coef),
        "abs_coefficient": np.abs(coef),
        "selected": pd.array(coef != 0.0, dtype="boolean"),
        "direction": np.where(coef > 0, "increases_risk", np.where(coef < 0, "decreases_risk", "none")),
        "rank": rank,
        "relative_to_reference": False,
    })[COEFFICIENT_COLUMNS]


# ---------------------------------------------------------------------- permutation importance
def _predict(pipeline: FittedPipeline, df: pd.DataFrame, calibrated: bool) -> np.ndarray:
    return pipeline.predict_proba(df, calibrated=True) if calibrated else pipeline.predict_proba(df)


def permutation_importance_grouped(pipeline: FittedPipeline, df: pd.DataFrame, y: Any, *, features: Sequence[str], n_repeats: int,
                                   seed: int, metric: str = "auroc", calibrated: bool = False) -> pd.DataFrame:
    """Permutation importance per raw feature: baseline metric minus permuted metric (AUROC), or permuted minus
    baseline (Brier), so positive = important. Mean and SD (ddof = 0, as scikit-learn) over ``n_repeats``.

    Repeat ``r`` of feature ``f`` uses ``rng_for(seed, f'perm:{f}:{r}')``; ``df`` is never modified.
    """
    if metric not in PERMUTATION_METRICS:
        raise ConfigError(f"permutation metric must be one of {sorted(PERMUTATION_METRICS)}, got {metric!r}")
    if isinstance(n_repeats, bool) or not isinstance(n_repeats, (int, np.integer)) or n_repeats < 1:
        raise ConfigError(f"n_repeats must be a positive integer, got {n_repeats!r}")
    features = list(features)
    if not features or len(set(features)) != len(features):
        raise ConfigError("features must be a non-empty list without duplicates")
    not_predictors = sorted(set(features) - set(pipeline.spec.predictor_names()))
    if not_predictors:
        raise ConfigError(f"permutation importance is defined for spec predictors only; got {not_predictors}")
    missing = sorted(set(features) - set(df.columns))
    if missing:
        raise ConfigError(f"features not in the evaluation frame: {missing}")
    score, direction = PERMUTATION_METRICS[metric]
    yy, pp, _ = prepare_inputs(y, _predict(pipeline, df, calibrated))
    if metric == "auroc" and (yy.sum() == 0 or yy.sum() == yy.size):
        raise ImportanceError("AUROC permutation importance needs both outcome classes in the evaluation rows")
    baseline = score(yy, pp)
    rows = []
    for feature in features:
        original = df[feature].array
        drops = np.empty(n_repeats)
        for r in range(n_repeats):
            permuted = df.copy(deep=False)
            permuted[feature] = original.take(rng_for(seed, f"perm:{feature}:{r}").permutation(len(df)))
            drops[r] = direction * (baseline - score(yy, _predict(pipeline, permuted, calibrated)))
        rows.append({"feature": feature, "permutation_importance_mean": float(drops.mean()),
                     "permutation_importance_std": float(drops.std(ddof=0)), "baseline_metric": float(baseline), "metric": metric})
    log.info("permutation_importance_done", extra_fields={"metric": metric, "n_features": len(features), "n_repeats": n_repeats,
                                                         "n_rows": len(df), "baseline": float(baseline), "calibrated": calibrated})
    return pd.DataFrame(rows, columns=PERMUTATION_COLUMNS)


# ---------------------------------------------------------------------- per-raw-feature table
def build_feature_importance_table(pipeline: FittedPipeline, permutation_df: pd.DataFrame, stability_df: pd.DataFrame | None = None,
                                   *, model_name: str) -> pd.DataFrame:
    """``feature_importance.csv``: one row per raw feature of ``permutation_df``, ranked by permutation mean.

    ``coefficient``/``odds_ratio`` only when the raw feature maps to exactly one design column of a linear model;
    ``selection_frequency`` = max over the feature's design columns in ``stability_df``; ``shap_mean_abs`` NaN.
    """
    absent = [c for c in PERMUTATION_COLUMNS[:3] if c not in permutation_df.columns]
    if absent:
        raise ConfigError(f"permutation table lacks columns {absent}")
    if permutation_df["feature"].duplicated().any():
        raise ConfigError("permutation table has duplicated features")
    table = permutation_df[PERMUTATION_COLUMNS[:3]].reset_index(drop=True)
    table.insert(1, "model", model_name)

    coefs = coefficient_table(pipeline)
    one_to_one = coefs.groupby("feature").filter(lambda g: len(g) == 1).set_index("feature")
    table["coefficient"] = table["feature"].map(one_to_one["coefficient"]).astype("float64")
    table["odds_ratio"] = table["feature"].map(one_to_one["odds_ratio"]).astype("float64")
    table["shap_mean_abs"] = math.nan
    table["selection_frequency"] = math.nan
    if stability_df is not None and len(stability_df):
        per_raw = stability_df.groupby("raw_feature")["selection_frequency"].max()
        table["selection_frequency"] = table["feature"].map(per_raw).astype("float64")
    table["importance_rank"] = table["permutation_importance_mean"].rank(ascending=False, method="min").astype("Int64")
    table = table.sort_values(["importance_rank", "feature"], na_position="last", kind="stable").reset_index(drop=True)
    return table[FEATURE_IMPORTANCE_COLUMNS]
