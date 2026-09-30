"""Production inference: ``predict_risk`` scores rows with a verified model bundle (architecture §1.4, §5).

Rows are validated in inference mode (D-00: ``predictor_max_record_date < index_date``), transformed by the
bundle's fitted preprocessor (never refitted) and scored by the bundle's model. By default the bundle's served variant
is returned: recalibrated exactly when the bundle carries a recalibrator (D-19); a caller override is flagged per row.
Risk categories are exposed only when their thresholds were approved (M-12, B-03). Contributions of linear models are
on the uncalibrated log-odds scale.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from falls_ml.artifacts import utc_now
from falls_ml.bundle import LoadedBundle
from falls_ml.data.schema import PREDICTOR_MAX_DATE, validate_modeling_dataset
from falls_ml.errors import ConfigError, DatasetValidationError, PreprocessingMismatchError
from falls_ml.features.spec import FeatureSpec
from falls_ml.logging_utils import get_logger
from falls_ml.pipeline import FittedPipeline

log = get_logger(__name__)

AGE_FLAG_ABOVE = 95.0  # D-06: no cap, but ages above 95 lie beyond the published development data (Fig. S3.1)
SYNTHETIC_SOURCE = "synthetic_fixture"
FLAG_CATEGORY_NOT_APPROVED = "risk_category_not_approved"
FLAG_NO_CONTRIBUTIONS = "contributions_not_available_for_model"
FLAG_SERVED_VARIANT_OVERRIDDEN = "served_variant_overridden"


@dataclass(frozen=True)
class PredictionResult:
    research_id: str | None
    model_version: str
    prediction_timestamp: str
    risk_12m: float
    risk_category: str | None
    top_contributing_features: tuple[dict[str, Any], ...]
    data_quality_flags: tuple[str, ...]
    calibrated: bool


# ---------------------------------------------------------------------- input
def _as_frame(input_features: pd.DataFrame | Mapping[str, Any] | Sequence[Mapping[str, Any]], spec: FeatureSpec) -> pd.DataFrame:
    """DataFrame input is used as is; record input gets dtypes from the spec (dates parsed, numeric all-null columns)."""
    if isinstance(input_features, pd.DataFrame):
        return input_features.copy(deep=False)
    records = [input_features] if isinstance(input_features, Mapping) else list(input_features)
    if not all(isinstance(r, Mapping) for r in records):
        raise DatasetValidationError("predict_risk input must be a DataFrame, a mapping or a sequence of mappings")
    df = pd.DataFrame.from_records(records)
    try:
        for column in (spec.index_column, PREDICTOR_MAX_DATE):
            if column in df.columns:
                df[column] = pd.to_datetime(df[column])
        for f in spec.features:
            if f.name in df.columns and not f.is_categorical and df[f.name].dtype == object:
                df[f.name] = pd.to_numeric(df[f.name])
    except (ValueError, TypeError) as exc:
        raise DatasetValidationError(f"predict_risk record input cannot be typed: {exc}") from exc
    return df


# ---------------------------------------------------------------------- contributions
def is_indicator_column(pipeline: FittedPipeline, column: str) -> bool:
    """True for 0/1 design columns (binary features, category levels, missingness indicators)."""
    feature = pipeline.spec.get(pipeline.preprocessor.raw_feature_of(column))
    return feature.is_binary or feature.is_categorical or "=" in column or column.endswith("__missing")


def reference_design_row(bundle: LoadedBundle) -> pd.Series:
    """Contribution reference: 0 for indicator columns, the training mean for continuous columns."""
    pipeline = bundle.pipeline
    means = bundle.reference_profile.get("design_column_means") or {}
    values = {}
    for column in pipeline.preprocessor.design_columns():
        if is_indicator_column(pipeline, column):
            values[column] = 0.0
        elif column in means and means[column] is not None:
            values[column] = float(means[column])
        else:
            raise PreprocessingMismatchError(f"reference profile has no training mean for continuous design column {column!r}")
    return pd.Series(values, dtype="float64")


def linear_contributions(design: pd.DataFrame, bundle: LoadedBundle) -> pd.DataFrame | None:
    """Per-row ``coefficient × (x − reference)`` for linear models; ``None`` for non-linear models.

    Each row sums to ``LP(x) − LP(reference)``. This is verified row by row, so one row never affects another:
    a row whose contributions do not reproduce the adapter's linear predictor is set to NaN (not trustworthy).
    """
    model = bundle.pipeline.model
    return _contributions(design, model.linear_predictor(design) if model.is_linear else None, bundle)


def _contributions(design: pd.DataFrame, lp: np.ndarray | None, bundle: LoadedBundle) -> pd.DataFrame | None:
    model = bundle.pipeline.model
    if not model.is_linear or lp is None:
        return None
    coefs = model.get_feature_importance().set_index("feature")["coefficient"].reindex(design.columns)
    if coefs.isna().any():
        log.warning("linear_contributions_unavailable", extra_fields={"model": model.name, "reason": "coefficient table incomplete"})
        return None
    reference = reference_design_row(bundle).reindex(design.columns)
    lp_reference = float(np.asarray(model.linear_predictor(reference.to_frame().T), dtype="float64")[0])
    values = (design.to_numpy(dtype="float64") - reference.to_numpy()) * coefs.to_numpy(dtype="float64")
    target = np.asarray(lp, dtype="float64") - lp_reference
    consistent = np.isclose(values.sum(axis=1), target, rtol=1e-9, atol=1e-9)
    if not consistent.all():
        log.warning("linear_contributions_inconsistent", extra_fields={"model": model.name, "n_rows": int((~consistent).sum())})
        values[~consistent] = np.nan
    return pd.DataFrame(values, index=design.index, columns=design.columns)


def _top_features(row: np.ndarray, columns: list[str], raw_of: list[str], top_k: int) -> tuple[dict[str, Any], ...]:
    if np.isnan(row).any():  # contributions not verified for this row
        return ()
    order = np.argsort(-np.abs(row), kind="stable")
    out = []
    for j in order[:top_k]:
        if row[j] == 0.0:
            break
        out.append({"feature": columns[j], "raw_feature": raw_of[j], "contribution": float(row[j]),
                    "direction": "increases_risk" if row[j] > 0 else "decreases_risk"})
    return tuple(out)


# ---------------------------------------------------------------------- categories and flags
def _risk_categories(risk: np.ndarray, config: Mapping[str, Any]) -> list[str | None]:
    """Approved categories on lower-closed intervals [c_(i-1), c_i); ``None`` when not approved (M-12)."""
    if not config.get("approved"):
        return [None] * risk.size
    cutpoints = np.asarray(config["cutpoints"], dtype="float64")
    labels = list(config["labels"])
    if cutpoints.size == 0 or len(labels) != cutpoints.size + 1 or np.any(np.diff(cutpoints) <= 0) \
            or np.any((cutpoints <= 0) | (cutpoints >= 1)):
        raise ConfigError(f"approved risk categories need strictly increasing cutpoints in (0, 1) and one more label: {config}")
    return [labels[i] for i in np.searchsorted(cutpoints, risk, side="right")]


def _row_flags(df: pd.DataFrame, bundle: LoadedBundle) -> list[list[str]]:
    n = len(df)
    flags: list[list[str]] = [[] for _ in range(n)]

    def add(mask: np.ndarray, name: str) -> None:
        for i in np.flatnonzero(mask):
            flags[i].append(name)

    for column, name in (("bmi_value", "bmi_missing"), ("alcohol_category", "alcohol_missing"), ("smoking_status", "smoking_missing")):
        if column in df.columns:
            add(df[column].isna().to_numpy(), name)
    if "age_years" in df.columns:
        add(df["age_years"].to_numpy(dtype="float64") > AGE_FLAG_ABOVE, "age_above_95")
    poly_max = bundle.reference_profile.get("polypharmacy_max")
    if "polypharmacy_count_120d" in df.columns and poly_max is not None:
        add(df["polypharmacy_count_120d"].to_numpy(dtype="float64") > float(poly_max), "polypharmacy_above_training_max")

    extra = bundle.metadata.get("extra_metadata") or {}
    model_params = (bundle.metadata.get("model") or {}).get("params") or {}
    constant: list[str] = []
    if extra.get("unavailable_predictors") or model_params.get("unavailable_predictors"):  # M-11
        constant.append("unavailable_predictors_fixed_zero")
    if extra.get("mappings_clinically_validated") is not True or not all(f.clinically_validated for f in bundle.feature_spec.features):
        constant.append("mappings_not_clinically_validated")
    if (bundle.metadata.get("training_dataset") or {}).get("source") == SYNTHETIC_SOURCE:
        constant.append("synthetic_training_data")
    if not (bundle.metadata.get("risk_categories") or {}).get("approved"):
        constant.append(FLAG_CATEGORY_NOT_APPROVED)
    if not bundle.pipeline.model.is_linear:
        constant.append(FLAG_NO_CONTRIBUTIONS)
    return [row + constant for row in flags]


# ---------------------------------------------------------------------- public API
def _use_calibration(bundle: LoadedBundle, use_calibration: bool | None) -> tuple[bool, bool]:
    """(calibrated, overridden). ``None`` follows the bundle: calibrated iff it carries a recalibrator (served variant)."""
    calibrator = bundle.pipeline.calibrator
    served_calibrated = calibrator is not None and calibrator.method != "none"
    if use_calibration is None:
        return served_calibrated, False
    if not isinstance(use_calibration, bool):
        raise ConfigError(f"use_calibration must be None (follow the bundle), True or False, got {use_calibration!r}")
    if use_calibration and not served_calibrated:
        raise ConfigError("use_calibration=True but the bundle has no recalibrator (its served variant is "
                          f"{bundle.metadata.get('served_variant')!r}); recalibrated risks cannot be produced")
    return use_calibration, use_calibration != served_calibrated


def predict_risk(input_features: pd.DataFrame | Mapping[str, Any] | Sequence[Mapping[str, Any]], bundle: LoadedBundle, *,
                 include_research_id: bool = True, use_calibration: bool | None = None, now: datetime | str | None = None,
                 top_k: int = 5) -> list[PredictionResult]:
    """12-month fall-risk predictions, one :class:`PredictionResult` per input row (in input order).

    ``use_calibration=None`` serves the bundle's variant. An explicit value that differs from it adds the
    ``served_variant_overridden`` flag to every row; ``True`` on a bundle without a recalibrator raises ``ConfigError``.
    """
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
        raise ConfigError(f"top_k must be an integer >= 0, got {top_k!r}")
    calibrated, overridden = _use_calibration(bundle, use_calibration)
    spec, pipeline = bundle.feature_spec, bundle.pipeline
    df = _as_frame(input_features, spec).reset_index(drop=True)
    validate_modeling_dataset(df, spec, mode="inference")

    design = pipeline.preprocessor.transform(df)  # fitted at training; never refitted here
    model = pipeline.model
    risk = model.predict_proba(design)
    lp = model.linear_predictor(design)
    if calibrated:
        risk = pipeline.calibrator.transform(risk, lp=lp)
    risk = np.asarray(risk, dtype="float64")
    if risk.shape != (len(df),) or not np.all(np.isfinite(risk)):
        raise PreprocessingMismatchError(f"model returned invalid risks (shape {risk.shape}, finite={bool(np.all(np.isfinite(risk)))})")

    categories = _risk_categories(risk, bundle.metadata.get("risk_categories") or {})
    flags = _row_flags(df, bundle)
    contributions = _contributions(design, lp, bundle)
    contribution_rows = None if contributions is None else contributions.to_numpy()
    if model.is_linear:
        missing = np.ones(len(df), dtype=bool) if contribution_rows is None else np.isnan(contribution_rows).any(axis=1)
        flags = [row + [FLAG_NO_CONTRIBUTIONS] if missing[i] else row for i, row in enumerate(flags)]
    if overridden:
        flags = [row + [FLAG_SERVED_VARIANT_OVERRIDDEN] for row in flags]
    columns = list(design.columns)
    raw_of = [pipeline.preprocessor.raw_feature_of(c) for c in columns]

    id_col = spec.identifier_columns[0] if spec.identifier_columns else None
    ids = df[id_col].astype(object).tolist() if include_research_id and id_col in df.columns else [None] * len(df)
    timestamp = now.isoformat() if isinstance(now, datetime) else (now or utc_now())
    results = [
        PredictionResult(
            research_id=None if ids[i] is None or pd.isna(ids[i]) else str(ids[i]),
            model_version=bundle.model_version,
            prediction_timestamp=str(timestamp),
            risk_12m=float(risk[i]),
            risk_category=categories[i],
            top_contributing_features=() if contribution_rows is None else _top_features(contribution_rows[i], columns, raw_of, top_k),
            data_quality_flags=tuple(flags[i]),
            calibrated=calibrated,
        )
        for i in range(len(df))
    ]
    log.info("predictions_made", extra_fields={"n_rows": len(df), "model_version": bundle.model_version, "calibrated": calibrated,
                                               "served_variant_overridden": overridden,
                                               "categories_exposed": categories[0] is not None if categories else False})
    return results
