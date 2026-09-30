"""Strict schema validation of the modelling dataset (architecture §3.1, spec §5.9, §13.3, D-00).

Timing (D-00): prediction at the start of the index day; predictors use records strictly before
index_date; outcome events satisfy index_date <= date <= index_date + 1 year - 1 day.

Validation never repairs data. Every violation is collected and raised together as a
``DatasetValidationError`` so data engineering sees the full list in one pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from falls_ml.errors import DatasetValidationError
from falls_ml.features.spec import FeatureDefinition, FeatureSpec

Mode = Literal["training", "inference"]

VERSION_COLUMNS = ("dataset_version", "mapping_version", "source")
PREDICTOR_MAX_DATE = "predictor_max_record_date"
OUTCOME_EVENT_DATE = "outcome_first_event_date"


@dataclass(frozen=True)
class ValidationReport:
    n_rows: int
    mode: str
    warnings: tuple[str, ...] = field(default_factory=tuple)


def _is_string_like(s: pd.Series) -> bool:
    return pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype)


def _check_feature(f: FeatureDefinition, s: pd.Series, spec: FeatureSpec, problems: list[str]) -> None:
    n_null = int(s.isna().sum())
    if f.dtype == "binary":
        if not (pd.api.types.is_integer_dtype(s) or pd.api.types.is_bool_dtype(s)):
            problems.append(f"{f.name}: binary feature must have integer/bool dtype, got {s.dtype}")
            return
        if n_null:
            problems.append(f"{f.name}: {n_null} null values in binary feature (absent must be materialised as 0)")
        bad = set(pd.unique(s.dropna().astype("int64"))) - {0, 1}
        if bad:
            problems.append(f"{f.name}: illegal binary values {sorted(bad)[:5]}")
    elif f.dtype in {"float", "count", "float_nullable"}:
        if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            problems.append(f"{f.name}: numeric feature has non-numeric dtype {s.dtype}")
            return
        if f.dtype == "count" and not pd.api.types.is_integer_dtype(s):
            problems.append(f"{f.name}: count feature must have integer dtype, got {s.dtype}")
        if not f.nullable and n_null:
            problems.append(f"{f.name}: {n_null} null values in non-nullable feature")
        values = s.dropna().astype("float64")
        if len(values) and not np.isfinite(values.to_numpy()).all():
            problems.append(f"{f.name}: non-finite values present")
        vr = f.valid_range or {}
        lo, hi = vr.get("min"), vr.get("max")
        if f.name == "age_years":
            lo = max(spec.age_min, lo if lo is not None else spec.age_min)
        if lo is not None and (values < lo).any():
            problems.append(f"{f.name}: {int((values < lo).sum())} values below minimum {lo}")
        if hi is not None and (values > hi).any():
            problems.append(f"{f.name}: {int((values > hi).sum())} values above maximum {hi}")
    elif f.is_categorical:
        if not _is_string_like(s):
            problems.append(f"{f.name}: categorical feature must be string-typed, got {s.dtype}")
            return
        if not f.nullable and n_null:
            problems.append(f"{f.name}: {n_null} null values in non-nullable categorical feature")
        observed = set(map(str, pd.unique(s.dropna())))
        bad = sorted(observed - set(f.levels))
        if bad:
            problems.append(f"{f.name}: undeclared levels {bad[:10]} (allowed {list(f.levels)})")


def validate_modeling_dataset(
    df: pd.DataFrame,
    spec: FeatureSpec,
    *,
    mode: Mode = "training",
    expected_versions: dict[str, str] | None = None,
) -> ValidationReport:
    """Validate ``df`` against ``spec``. Raises ``DatasetValidationError`` listing every problem."""
    problems: list[str] = []
    warnings: list[str] = []
    cols = list(df.columns)

    dup_cols = sorted({c for c in cols if cols.count(c) > 1})
    if dup_cols:
        raise DatasetValidationError("Duplicate column names", [str(dup_cols)])

    predictors = spec.predictor_names()
    required = [spec.index_column, PREDICTOR_MAX_DATE, *predictors]
    if mode == "training":
        # outcome_first_event_date is required so the outcome-window definition (D-00/D-09) is always checked
        required = [*spec.identifier_columns, *required, spec.outcome.name, OUTCOME_EVENT_DATE, *VERSION_COLUMNS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        problems.append(f"missing required columns: {missing}")

    declared = set(required) | set(spec.identifier_columns) | set(spec.metadata_columns) | set(spec.provenance_columns)
    declared |= {spec.outcome.name}
    extra = sorted(set(cols) - declared)
    if extra:
        problems.append(f"undeclared columns present (must be declared in the feature spec or removed): {extra}")

    if len(df) == 0:
        problems.append("dataset has zero rows")

    # identifiers / index
    idx = spec.index_column
    if idx in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df[idx]):
            problems.append(f"{idx}: must be datetime64 dtype, got {df[idx].dtype}")
        elif df[idx].isna().any():
            problems.append(f"{idx}: {int(df[idx].isna().sum())} null index dates")
    key = [c for c in (*spec.identifier_columns, idx) if c in df.columns]
    if mode == "training" and len(key) == len(spec.identifier_columns) + 1:
        for c in spec.identifier_columns:
            if df[c].isna().any():
                problems.append(f"{c}: null identifiers")
        n_dup = int(df.duplicated(subset=key).sum())
        if n_dup:
            problems.append(f"{n_dup} duplicated rows on {key}")

    # leakage guard: predictors may only use records strictly before the index date (D-00)
    if PREDICTOR_MAX_DATE in df.columns and idx in df.columns and pd.api.types.is_datetime64_any_dtype(df[idx]):
        pm = df[PREDICTOR_MAX_DATE]
        if not pd.api.types.is_datetime64_any_dtype(pm):
            problems.append(f"{PREDICTOR_MAX_DATE}: must be datetime64 dtype, got {pm.dtype}")
        else:
            n_future = int((pm >= df[idx]).sum())
            if n_future:
                problems.append(f"LEAKAGE: {n_future} rows have {PREDICTOR_MAX_DATE} on or after {idx}")

    # outcome
    oc = spec.outcome.name
    if mode == "training" and oc in df.columns:
        y = df[oc]
        if not (pd.api.types.is_integer_dtype(y) or pd.api.types.is_bool_dtype(y)):
            problems.append(f"{oc}: outcome must be integer/bool dtype, got {y.dtype}")
        else:
            if y.isna().any():
                problems.append(f"{oc}: {int(y.isna().sum())} null outcomes")
            bad = set(pd.unique(y.dropna().astype("int64"))) - {0, 1}
            if bad:
                problems.append(f"{oc}: illegal outcome values {sorted(bad)}")
            elif len(df) and y.nunique() < 2:
                warnings.append(f"{oc}: only one outcome class present")
        if OUTCOME_EVENT_DATE in df.columns and idx in df.columns and pd.api.types.is_datetime64_any_dtype(df[idx]):
            ev = df[OUTCOME_EVENT_DATE]
            if not pd.api.types.is_datetime64_any_dtype(ev):
                problems.append(f"{OUTCOME_EVENT_DATE}: must be datetime64 dtype")
            elif pd.api.types.is_integer_dtype(y) or pd.api.types.is_bool_dtype(y):
                pos = y.astype("int64") == 1
                if (pos & ev.isna()).any():
                    problems.append(f"{OUTCOME_EVENT_DATE}: missing for {int((pos & ev.isna()).sum())} positive outcomes")
                if (~pos & ev.notna()).any():
                    problems.append(f"{OUTCOME_EVENT_DATE}: present for {int((~pos & ev.notna()).sum())} negative outcomes")
                end = spec.outcome.window_end(df[idx])
                outside = ev.notna() & ((ev < df[idx]) | (ev > end))
                if outside.any():
                    problems.append(f"LEAKAGE/DEFINITION: {int(outside.sum())} outcome event dates outside [index, index + {spec.outcome.horizon_label} - 1d]")

    # predictors
    for f in spec.features:
        if f.name in df.columns:
            _check_feature(f, df[f.name], spec, problems)

    # versions
    if mode == "training":
        for c in VERSION_COLUMNS:
            if c in df.columns:
                vals = pd.unique(df[c].dropna())
                if len(vals) != 1 or df[c].isna().any():
                    problems.append(f"{c}: must hold exactly one non-null constant value, found {list(vals)[:5]}")
                elif expected_versions and c in expected_versions and str(vals[0]) != str(expected_versions[c]):
                    problems.append(f"{c}: value {vals[0]!r} differs from manifest {expected_versions[c]!r}")

    if problems:
        raise DatasetValidationError(f"Modelling dataset failed {mode} validation ({len(problems)} problems)", problems)
    return ValidationReport(n_rows=len(df), mode=mode, warnings=tuple(warnings))
