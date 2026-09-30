"""Fitted pipeline = Preprocessor + ModelAdapter (+ optional Recalibrator).

This is the single object used for evaluation, resampling analyses (stability, optimism, IECV)
and production inference, so the exact same preprocessing runs everywhere.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from falls_ml.config import ExperimentConfig
from falls_ml.errors import LeakageError
from falls_ml.features.preprocessing import Preprocessor
from falls_ml.features.spec import FeatureSpec
from falls_ml.models.base import ModelAdapter
from falls_ml.models.registry import get_adapter_class

FitFunction = Callable[[pd.DataFrame, int], "FittedPipeline"]


@dataclass
class FittedPipeline:
    spec: FeatureSpec
    preprocessor: Preprocessor
    model: ModelAdapter
    calibrator: Any | None = None  # falls_ml.evaluation.calibration.Recalibrator

    def design(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.preprocessor.transform(df)

    def predict_proba(self, df: pd.DataFrame, *, calibrated: bool = False) -> np.ndarray:
        X = self.design(df)
        p = self.model.predict_proba(X)
        if calibrated:
            if self.calibrator is None:
                raise ValueError("calibrated predictions requested but no calibrator is attached")
            p = self.calibrator.transform(p, lp=self.model.linear_predictor(X))
        return p

    def linear_predictor(self, df: pd.DataFrame) -> np.ndarray | None:
        return self.model.linear_predictor(self.design(df))


def model_params_from_config(config: ExperimentConfig) -> dict[str, Any]:
    params = dict(config.model.params)
    if config.model.published_equation is not None:
        pe = config.model.published_equation
        # published options come only from model.published_equation (config validation forbids model.params here)
        params.update({"config": pe.config, "sex_parameterisation": pe.sex_parameterisation,
                       "unavailable_predictors": list(pe.unavailable_predictors), "unavailable_fill": pe.unavailable_fill,
                       "zero_low_support_predictors": pe.zero_low_support_predictors, "feature_spec": config.dataset.feature_spec})
    if config.model.name == "lasso_logistic_cv":
        params.setdefault("cv_folds", config.validation.cv_folds)
    return params


def row_keys(df: pd.DataFrame, spec: FeatureSpec) -> set[tuple[Any, Any]]:
    """(research_id, index_date) keys identifying modelling-dataset rows."""
    return set(zip(df[spec.identifier_columns[0]], df[spec.index_column]))


def fit_pipeline(train_df: pd.DataFrame, spec: FeatureSpec, config: ExperimentConfig, *, random_state: int,
                 model_params: dict[str, Any] | None = None, forbidden_row_keys: set[tuple[Any, Any]] | None = None) -> FittedPipeline:
    """Fit preprocessing and model on ``train_df`` only.

    ``forbidden_row_keys`` (optional) is a leakage guard: the orchestrator passes the (research_id, index_date)
    keys of the validation and test partitions, and fitting refuses any overlap.
    """
    id_col = spec.identifier_columns[0]
    if forbidden_row_keys:
        leaked = row_keys(train_df, spec) & forbidden_row_keys
        if leaked:
            raise LeakageError(f"{len(leaked)} validation/test rows were passed to fit_pipeline")
    y = train_df[spec.outcome.name].to_numpy(dtype=np.int64)
    pre = Preprocessor(spec, config.preprocessing.representation, fp=config.preprocessing.fractional_polynomial,
                       bmi_obese_cutpoint=config.preprocessing.bmi_obese_cutpoint, random_state=random_state)
    X = pre.fit(train_df, y).transform(train_df)
    adapter_cls = get_adapter_class(config.model.name)
    if adapter_cls.representation != config.preprocessing.representation:
        from falls_ml.errors import ConfigError
        raise ConfigError(f"model {config.model.name} requires representation {adapter_cls.representation!r}, "
                          f"config has {config.preprocessing.representation!r}")
    params = dict(model_params if model_params is not None else model_params_from_config(config))
    if config.model.name == "lasso_logistic_cv":
        # spec 6.2: the unpenalised refit uses one published reference level per categorical variable
        params.setdefault("refit_reference_levels", pre.reference_levels())
    model = adapter_cls(params, random_state=random_state)
    groups = train_df[id_col].to_numpy() if id_col in train_df.columns else None
    model.fit(X, y, groups=groups)
    return FittedPipeline(spec=spec, preprocessor=pre, model=model)
