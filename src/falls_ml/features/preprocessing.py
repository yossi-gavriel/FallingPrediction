"""The single preprocessing implementation: raw modelling-dataset rows → float64 design matrix.

Architecture §3.3. ``fit`` sees training rows only and stores nothing but the training row count,
FP forms (D-13) and medians of nullable extension numeric features. Category levels come from the
feature spec and are never learned from data. ``transform`` reads only predictor columns (never
identifiers, dates or outcomes) and raises ``DatasetValidationError`` on schema violations (§5.9).

Representations:
- ``efalls_published``: published terms and reference levels (§5.3–5.8, D-02, D-04).
- ``efalls_fp_all_levels``: FP terms + all-levels indicators for the retrained LASSO (D-10, D-13).
- ``efalls_reference_coded``: published continuous forms, reference-coded categoricals.
- ``efalls_raw``: raw age and polypharmacy count + all-levels indicators (tree models).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from falls_ml.config import FractionalPolynomialSection
from falls_ml.errors import ConfigError, DatasetValidationError, NotFittedError, PreprocessingMismatchError
from falls_ml.features.fractional_polynomial import FPFittingError, FPForm, select_fp_forms
from falls_ml.models.separation import SeparationError, stata_logit_omissions
from falls_ml.features.spec import FeatureDefinition, FeatureSpec
from falls_ml.features.transforms import (
    BMI_LEVELS,
    BMI_OVERWEIGHT_FROM,
    MISSING_LEVEL,
    SMOKING_LEVELS,
    alcohol_level,
    bmi_category,
    log_polypharmacy,
    smoking_all_levels,
    smoking_published,
)
from falls_ml.logging_utils import get_logger

log = get_logger(__name__)

REPRESENTATIONS = ("efalls_published", "efalls_fp_all_levels", "efalls_reference_coded", "efalls_raw")
STATE_VERSION = "falls_ml.preprocessor/1"

AGE, SEX, POLYPHARMACY, BMI, SMOKING, ALCOHOL = (
    "age_years", "sex", "polypharmacy_count_120d", "bmi_value", "smoking_status", "alcohol_category")
#: eFalls non-binary predictors in design order (continuous first, as in Table S3.2).
EFALLS_NON_BINARY = (AGE, POLYPHARMACY, SEX, BMI, SMOKING, ALCOHOL)
N_EFALLS_BINARIES = 72  # D-12: 78 candidates = 6 non-binary + 72 binary
POLYPHARMACY_LOG_COLUMN = "polypharmacy_log_p1_div10"
SEX_LEVELS = ("female", "male")
ALCOHOL_REFERENCE = "lower_risk"
BMI_REFERENCE = "overweight"
#: Reference levels for re-expressing coefficients (D-01 Table S3.2 sex reference, D-02, D-04, D-10).
REFERENCE_LEVELS = {"sex": "male", "bmi_category": BMI_REFERENCE, "smoking_status": "never", "alcohol_category": ALCOHOL_REFERENCE}
_REFERENCE_FEATURE = {"sex": SEX, "bmi_category": BMI, "smoking_status": SMOKING, "alcohol_category": ALCOHOL}
#: Stata-published FP forms for ``fp.mode == fixed_published`` (§5.3, §5.5).
FIXED_PUBLISHED_FORMS = {AGE: ((1.0,), 0.0, 1.0), POLYPHARMACY: ((0.0,), 1.0, 10.0)}

_EXPECTED = {  # name -> (dtype, levels or None)
    AGE: ("float", None), SEX: ("categorical", SEX_LEVELS), POLYPHARMACY: ("count", None),
    BMI: ("float_nullable", BMI_LEVELS), SMOKING: ("categorical_nullable", SMOKING_LEVELS), ALCOHOL: ("categorical_nullable", None),
}
# (continuous style, categorical style) per representation
_STYLES = {
    "efalls_published": ("published", "published"),
    "efalls_reference_coded": ("published", "reference"),
    "efalls_raw": ("raw", "all_levels"),
    "efalls_fp_all_levels": ("fp", "all_levels"),
}


@dataclass(frozen=True)
class _Block:
    raw_feature: str
    columns: tuple[str, ...]
    compute: Callable[[pd.DataFrame], np.ndarray]  # frame -> (n, len(columns)) float64


def _numeric(s: pd.Series) -> np.ndarray:
    return s.to_numpy(dtype="float64", na_value=np.nan)


def _indicators(values: pd.Series, levels: list[str]) -> np.ndarray:
    obj = values.to_numpy(dtype=object)
    return np.column_stack([(obj == level).astype("float64") for level in levels])


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


class Preprocessor:
    """Fit on training rows; transform any rows into the design matrix of ``representation``."""

    def __init__(self, spec: FeatureSpec, representation: str, *, fp: FractionalPolynomialSection | None = None,
                 bmi_obese_cutpoint: float = 30.0, random_state: int = 0):
        if representation not in REPRESENTATIONS:
            raise ConfigError(f"Unknown preprocessing representation {representation!r}; allowed {list(REPRESENTATIONS)}")
        if not float(bmi_obese_cutpoint) > BMI_OVERWEIGHT_FROM:
            raise ConfigError(f"bmi_obese_cutpoint must be > {BMI_OVERWEIGHT_FROM}, got {bmi_obese_cutpoint}")
        self.spec = spec
        self.representation = representation
        self.fp = fp if fp is not None else FractionalPolynomialSection()
        self.bmi_obese_cutpoint = float(bmi_obese_cutpoint)
        self.random_state = int(random_state)
        self._features = {f.name: f for f in spec.features}
        self._check_spec()
        # fitted state
        self.n_train_rows_: int | None = None
        self.fp_forms_: dict[str, FPForm] = {}
        self.medians_: dict[str, float] = {}
        self.design_columns_: list[str] | None = None
        self._raw_of: dict[str, str] = {}

    # ------------------------------------------------------------------ configuration checks
    def _check_spec(self) -> None:
        for name, (dtype, levels) in _EXPECTED.items():
            f = self._features.get(name)
            if f is None:
                continue
            if f.dtype != dtype or (levels is not None and tuple(f.levels) != levels):
                raise ConfigError(f"{name}: expected dtype {dtype} with levels {levels}, spec has {f.dtype} {f.levels}")
        if ALCOHOL in self._features and ALCOHOL_REFERENCE not in self._features[ALCOHOL].levels:
            raise ConfigError(f"{ALCOHOL}: levels must include the reference {ALCOHOL_REFERENCE!r}")
        for f in self.spec.features:
            if f.is_categorical and MISSING_LEVEL in f.levels:
                raise ConfigError(f"{f.name}: {MISSING_LEVEL!r} must not be a declared level (derived from nulls)")
            if (f.name not in _EXPECTED and f.dtype == "categorical_nullable" and f.missing_rule == "reference_level"
                    and f.reference_level not in f.levels):
                raise ConfigError(f"{f.name}: missing_rule reference_level requires a declared reference_level")
        if self.representation == "efalls_published":
            n_binaries = sum(f.is_binary and f.exact_efalls_baseline and f.layer == "L1_published" for f in self.spec.features)
            missing = [n for n in EFALLS_NON_BINARY if n not in self._features]
            if missing or n_binaries != N_EFALLS_BINARIES:
                raise ConfigError(f"efalls_published requires the full eFalls feature set (missing {missing}, "
                                  f"{n_binaries} of {N_EFALLS_BINARIES} published binaries)")
        if self.representation == "efalls_fp_all_levels":
            if self.fp.mode not in ("select", "fixed_published", "linear"):
                raise ConfigError(f"fractional_polynomial.mode {self.fp.mode!r} not supported")
            if not self.fp.variables:
                raise ConfigError("efalls_fp_all_levels requires at least one fractional_polynomial variable")
            for v in self.fp.variables:
                f = self._features.get(v)
                if f is None or f.dtype not in ("float", "count"):
                    raise ConfigError(f"FP variable {v!r} must be a non-nullable float/count feature of the spec")
                if self.fp.mode == "fixed_published" and v not in FIXED_PUBLISHED_FORMS:
                    raise ConfigError(f"fp mode fixed_published defines forms only for {sorted(FIXED_PUBLISHED_FORMS)}, not {v!r}")

    # ------------------------------------------------------------------ validation
    def _validate(self, df: pd.DataFrame) -> None:
        """Schema checks needed by the transforms (§5.9); every problem is reported at once."""
        cols = list(df.columns)
        missing = [f.name for f in self.spec.features if f.name not in df.columns]
        if missing:
            raise DatasetValidationError("Preprocessor input is missing predictor columns", [str(missing)])
        dupes = sorted({c for c in cols if cols.count(c) > 1 and c in self._features})
        if dupes:
            raise DatasetValidationError("Preprocessor input has duplicated predictor columns", [str(dupes)])
        problems: list[str] = []
        for f in self.spec.features:
            s = df[f.name]
            n_null = int(s.isna().sum())
            if f.is_categorical:
                if not f.nullable and n_null:
                    problems.append(f"{f.name}: {n_null} null values in non-nullable categorical feature")
                observed = set(s.dropna().astype(object).tolist())
                bad = sorted(map(str, observed - set(f.levels)))
                if bad:
                    problems.append(f"{f.name}: undeclared levels {bad[:10]} (allowed {list(f.levels)})")
                continue
            if not (pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s)):
                problems.append(f"{f.name}: expected a numeric dtype, got {s.dtype}")
                continue
            values = _numeric(s)
            if not f.nullable and n_null:
                problems.append(f"{f.name}: {n_null} null values in non-nullable feature")
            if np.isinf(values).any():
                problems.append(f"{f.name}: infinite values present")
            if f.is_binary and not np.isin(values[~np.isnan(values)], (0.0, 1.0)).all():
                problems.append(f"{f.name}: binary feature has values other than 0/1")
            if f.dtype == "count" and (values < 0).any():
                problems.append(f"{f.name}: {int((values < 0).sum())} negative counts")
        if problems:
            raise DatasetValidationError(f"Preprocessor input failed validation ({len(problems)} problems)", problems)

    # ------------------------------------------------------------------ design blocks
    def _blocks(self, continuous: str, categorical: str, *, exclude: tuple[str, ...] = ()) -> list[_Block]:
        """Design blocks in deterministic order: eFalls non-binary terms, then remaining features in spec order."""
        order = [n for n in EFALLS_NON_BINARY if n in self._features]
        order += [f.name for f in self.spec.features if f.name not in EFALLS_NON_BINARY]
        blocks: list[_Block] = []
        for name in order:
            if name in exclude:
                continue
            f = self._features[name]
            if continuous == "fp" and name in self.fp_forms_:
                form = self.fp_forms_[name]
                blocks.append(_Block(name, tuple(form.column_names()),
                                     lambda d, n=name, fm=form: fm.transform(_numeric(d[n]))))
            elif name == POLYPHARMACY and continuous == "published":
                blocks.append(_Block(name, (POLYPHARMACY_LOG_COLUMN,),
                                     lambda d: log_polypharmacy(_numeric(d[POLYPHARMACY])).reshape(-1, 1)))
            elif name in (SEX, BMI, SMOKING, ALCOHOL):
                blocks.append(self._efalls_categorical_block(f, categorical))
            else:
                blocks.append(self._generic_block(f, categorical))
        return blocks

    def _efalls_categorical_block(self, f: FeatureDefinition, style: str) -> _Block:
        all_levels = style == "all_levels"
        if f.name == SEX:
            levels = list(SEX_LEVELS) if all_levels else ["male"]
            return _Block(SEX, tuple(f"sex={lv}" for lv in levels), lambda d: _indicators(d[SEX], levels))
        if f.name == BMI:
            levels = [lv for lv in BMI_LEVELS if all_levels or lv != BMI_REFERENCE]
            cut = self.bmi_obese_cutpoint
            return _Block(BMI, tuple(f"bmi_category={lv}" for lv in levels),
                          lambda d: _indicators(bmi_category(d[BMI], cut), levels))
        if f.name == SMOKING:
            if style == "published":
                return _Block(SMOKING, ("smoking=current",), lambda d: _indicators(smoking_published(d[SMOKING]), ["current"]))
            levels = [lv for lv in SMOKING_LEVELS if all_levels or lv != "never"]
            return _Block(SMOKING, tuple(f"smoking_status={lv}" for lv in levels),
                          lambda d: _indicators(smoking_all_levels(d[SMOKING]), levels))
        declared = list(f.levels)
        levels = [lv for lv in declared if all_levels or lv != ALCOHOL_REFERENCE] + [MISSING_LEVEL]
        return _Block(ALCOHOL, tuple(f"alcohol_category={lv}" for lv in levels),
                      lambda d: _indicators(alcohol_level(d[ALCOHOL], declared), levels))

    def _generic_block(self, f: FeatureDefinition, style: str) -> _Block:
        name = f.name
        if f.dtype in ("binary", "float", "count"):
            return _Block(name, (name,), lambda d: _numeric(d[name]).reshape(-1, 1))
        if f.dtype == "float_nullable":
            def impute(d: pd.DataFrame) -> np.ndarray:
                v = _numeric(d[name])
                miss = np.isnan(v)
                return np.column_stack([np.where(miss, self.medians_[name], v), miss.astype("float64")])
            return _Block(name, (name, f"{name}__missing"), impute)
        levels = list(f.levels)[1:] if style == "reference" else list(f.levels)
        fill_reference = f.dtype == "categorical_nullable" and f.missing_rule == "reference_level"
        if f.dtype == "categorical_nullable" and not fill_reference:
            levels.append(MISSING_LEVEL)

        def encode(d: pd.DataFrame) -> np.ndarray:
            s = d[name]
            filled = s.astype(object).where(s.notna(), f.reference_level if fill_reference else MISSING_LEVEL)
            return _indicators(filled, levels) if levels else np.empty((len(d), 0))
        return _Block(name, tuple(f"{name}={lv}" for lv in levels), encode)

    # ------------------------------------------------------------------ fit / transform
    def fit(self, df: pd.DataFrame, y: Any = None) -> Preprocessor:
        """Fit on training rows only. ``y`` is required only for ``fp.mode == select`` (D-13)."""
        self.design_columns_ = None  # a failed (re)fit leaves the preprocessor unfitted, never half-updated
        self._validate(df)
        if len(df) == 0:
            raise DatasetValidationError("Preprocessor.fit received zero rows")
        medians: dict[str, float] = {}
        for f in self.spec.features:
            if f.dtype == "float_nullable" and f.name != BMI:
                values = _numeric(df[f.name])
                if np.isnan(values).all():
                    raise DatasetValidationError(f"{f.name}: all training values are null; no median for imputation")
                medians[f.name] = float(np.nanmedian(values))
        self.medians_ = medians
        self.fp_forms_ = self._fit_fp_forms(df, y) if self.representation == "efalls_fp_all_levels" else {}
        self.n_train_rows_ = len(df)
        self._finalize()
        log.info("preprocessor_fitted", extra_fields={
            "representation": self.representation, "n_train_rows": self.n_train_rows_,
            "n_design_columns": len(self.design_columns_ or []), "fingerprint": self.fingerprint()})
        return self

    def _fit_fp_forms(self, df: pd.DataFrame, y: Any) -> dict[str, FPForm]:
        variables = [f.name for f in self.spec.features if f.name in self.fp.variables]
        if self.fp.mode == "fixed_published":
            return {v: FPForm(v, *FIXED_PUBLISHED_FORMS[v], selection_table={"mode": "fixed_published"}) for v in variables}
        if self.fp.mode == "linear":
            return {v: FPForm(v, (1.0,), 0.0, 1.0, selection_table={"mode": "linear"}) for v in variables}
        if y is None:
            raise ConfigError("fractional_polynomial.mode=select requires the training outcome y")
        yv = np.asarray(y, dtype="float64").ravel()
        if yv.size != len(df) or not np.isin(yv, (0.0, 1.0)).all():
            raise DatasetValidationError(f"FP selection outcome must be {len(df)} values in {{0, 1}}")
        # FP stage uses reference-coded categoricals (D-10) plus all other non-FP columns.
        other_blocks = self._blocks("fp", "reference", exclude=tuple(variables))
        other = np.column_stack([np.empty((len(df), 0)), *(b.compute(df) for b in other_blocks)])
        other_names = [c for b in other_blocks for c in b.columns]
        # D-13: constant columns and perfect predictors are omitted as Stata ``logit`` does (noted, never silent).
        try:
            omissions = stata_logit_omissions(other, yv, other_names)
        except SeparationError as exc:
            raise FPFittingError(f"FP selection: {exc}") from exc
        if omissions.any:
            log.warning("fp_selection_stata_omissions", extra_fields=omissions.to_dict())
        rows = omissions.kept_rows
        other = other[np.ix_(rows, list(omissions.kept_columns))]
        forms = select_fp_forms({v: _numeric(df[v])[rows] for v in variables}, other, yv[rows], powers=self.fp.powers,
                                max_degree=self.fp.max_degree, alpha=self.fp.alpha, max_cycles=self.fp.max_cycles)
        return {v: dataclasses.replace(forms[v], selection_table={**forms[v].selection_table,
                                                                  "stata_logit_omissions": omissions.to_dict()})
                for v in variables}

    def _finalize(self) -> None:
        continuous, categorical = _STYLES[self.representation]
        columns: list[str] = []
        raw_of: dict[str, str] = {}
        for block in self._blocks(continuous, categorical):
            columns.extend(block.columns)
            raw_of.update(dict.fromkeys(block.columns, block.raw_feature))
        if len(set(columns)) != len(columns):
            raise ConfigError(f"Design column names collide: {sorted({c for c in columns if columns.count(c) > 1})}")
        self.design_columns_ = columns
        self._raw_of = raw_of

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Design matrix (float64, deterministic column order, index preserved). Reads predictor columns only."""
        self._check_fitted()
        self._validate(df)
        continuous, categorical = _STYLES[self.representation]
        parts = [b.compute(df) for b in self._blocks(continuous, categorical)]
        matrix = np.column_stack([np.empty((len(df), 0)), *parts]).astype("float64", copy=False)
        return pd.DataFrame(matrix, index=df.index, columns=list(self.design_columns_ or []))

    # ------------------------------------------------------------------ queries
    def _check_fitted(self) -> None:
        if self.design_columns_ is None:
            raise NotFittedError("Preprocessor has not been fitted")

    def design_columns(self) -> list[str]:
        self._check_fitted()
        return list(self.design_columns_ or [])

    def raw_feature_of(self, design_column: str) -> str:
        """Spec feature name that produced ``design_column``."""
        self._check_fitted()
        try:
            return self._raw_of[design_column]
        except KeyError as exc:
            raise ConfigError(f"{design_column!r} is not a design column of this preprocessor") from exc

    def reference_levels(self) -> dict[str, str]:
        """Reference levels for re-expressing coefficients (D-10); extension categoricals use their first level."""
        refs = {key: level for key, level in REFERENCE_LEVELS.items() if _REFERENCE_FEATURE[key] in self._features}
        for f in self.spec.features:
            if f.is_categorical and f.name not in _EXPECTED:
                refs[f.name] = f.levels[0]
        return refs

    # ------------------------------------------------------------------ identity and persistence
    def _fingerprint_payload(self) -> dict[str, Any]:
        return {
            "state_version": STATE_VERSION,
            "feature_spec_sha256": self.spec.content_sha256,
            "representation": self.representation,
            "bmi_obese_cutpoint": self.bmi_obese_cutpoint,
            "fp_forms": {v: {"powers": list(f.powers), "shift": f.shift, "scale": f.scale} for v, f in self.fp_forms_.items()},
            "medians": self.medians_,
            "design_columns": self.design_columns_,
        }

    def fingerprint(self) -> str:
        """SHA-256 of the canonical JSON of everything that determines ``transform`` output."""
        self._check_fitted()
        return hashlib.sha256(_canonical_json(self._fingerprint_payload()).encode("utf-8")).hexdigest()

    def state(self) -> dict[str, Any]:
        """JSON-serialisable fitted state (includes FP selection tables for audit)."""
        self._check_fitted()
        fp = {k: list(v) if isinstance(v, tuple) else v for k, v in dataclasses.asdict(self.fp).items()}
        return {
            "state_version": STATE_VERSION,
            "feature_spec": {"name": self.spec.name, "version": self.spec.version, "sha256": self.spec.content_sha256},
            "representation": self.representation,
            "bmi_obese_cutpoint": self.bmi_obese_cutpoint,
            "random_state": self.random_state,
            "fp": fp,
            "n_train_rows": self.n_train_rows_,
            "fp_forms": {v: f.to_dict() for v, f in self.fp_forms_.items()},
            "medians": dict(self.medians_),
            "design_columns": list(self.design_columns_ or []),
            "fingerprint": self.fingerprint(),
        }

    @classmethod
    def from_state(cls, spec: FeatureSpec, state: dict[str, Any]) -> Preprocessor:
        """Rebuild a fitted preprocessor; refuses a different spec or a state whose fingerprint does not verify."""
        if state.get("state_version") != STATE_VERSION:
            raise PreprocessingMismatchError(f"Unsupported preprocessor state version {state.get('state_version')!r}")
        if state["feature_spec"]["sha256"] != spec.content_sha256:
            raise PreprocessingMismatchError("Preprocessor state was fitted with a different feature spec")
        fp_raw = state["fp"]
        fp = FractionalPolynomialSection(**{k: tuple(v) if isinstance(v, list) else v for k, v in fp_raw.items()})
        pre = cls(spec, state["representation"], fp=fp, bmi_obese_cutpoint=state["bmi_obese_cutpoint"],
                  random_state=state["random_state"])
        pre.n_train_rows_ = int(state["n_train_rows"])
        pre.fp_forms_ = {v: FPForm.from_dict(d) for v, d in state["fp_forms"].items()}
        pre.medians_ = {k: float(v) for k, v in state["medians"].items()}
        pre._finalize()
        if pre.design_columns_ != list(state["design_columns"]) or pre.fingerprint() != state["fingerprint"]:
            raise PreprocessingMismatchError("Preprocessor state does not reproduce its design columns / fingerprint")
        return pre
