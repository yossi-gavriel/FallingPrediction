"""Fixed published eFalls equation (spec §5.10, D-01, D-07, D-20, M-11).

``p = expit(LP)`` with ``LP = intercept + sex term + Σ β·x`` over the ``efalls_published`` design columns:

* ``age_years`` (decimal years, identity), ``polypharmacy_log_p1_div10`` (= ln((P + 1) / 10), computed by the
  Preprocessor), ``sex=male``;
* ``bmi_category=<underweight|normal|obese|missing>`` (reference overweight), ``smoking=current`` (reference
  ex/never), ``alcohol_category=<harmful|higher_risk|previous_higher_risk_or_harmful|zero|missing>`` (reference
  lower risk);
* the 72 binary predictors in feature-spec order (the 10 not retained by LASSO have coefficient 0).

The sex/intercept conflict (D-01) is expressed as ``intercept = intercept_female`` and
``β[sex=male] = intercept_male − intercept_female`` for each of the four sex parameterisations. No parameter is
ever estimated: ``fit`` only validates the design matrix, and labels are ignored.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np
import pandas as pd
import yaml
from scipy.special import expit

from falls_ml.errors import BundleIntegrityError, ConfigError, PreprocessingMismatchError
from falls_ml.features.spec import FeatureSpec, load_feature_spec
from falls_ml.logging_utils import get_logger
from falls_ml.models.base import ModelAdapter, coefficient_importance
from falls_ml.paths import resolve_path

log = get_logger(__name__)

SEX_PARAMETERISATIONS = ("lp_c_box_s3_1", "lp_a_table_s3_2", "lp_b_label_swap", "lp_d2_numeric_swap")
PRIMARY_VARIANT = "lp_c_box_s3_1"
CO_REPORTED_VARIANT = "lp_a_table_s3_2"
DEFAULT_FEATURE_SPEC = "configs/features/efalls_v1.yaml"
UNAVAILABLE_FILLS = ("zero", "sail_prevalence")
PARTIAL_LABEL = "efalls_partial_scoring"
PUBLISHED_LABEL = "efalls_published_scoring"

AGE_COLUMN = "age_years"
POLYPHARMACY_COLUMN = "polypharmacy_log_p1_div10"
SEX_COLUMN = "sex=male"
BMI_LEVELS = ("underweight", "normal", "obese", "missing")
ALCOHOL_LEVELS = ("harmful", "higher_risk", "previous_higher_risk_or_harmful", "zero", "missing")
POLYPHARMACY_TRANSFORM = "log((P + 1) / 10)"
_MIN_LOG_POLYPHARMACY = math.log(0.1) - 1e-12  # value at P = 0, with round-off slack

#: Non-binary spec features → the design columns that carry them (in design order).
NON_BINARY_TERMS: dict[str, tuple[str, ...]] = {
    "age_years": (AGE_COLUMN,),
    "polypharmacy_count_120d": (POLYPHARMACY_COLUMN,),
    "sex": (SEX_COLUMN,),
    "bmi_value": tuple(f"bmi_category={level}" for level in BMI_LEVELS),
    "smoking_status": ("smoking=current",),
    "alcohol_category": tuple(f"alcohol_category={level}" for level in ALCOHOL_LEVELS),
}
#: Categorical indicator columns whose SAIL proportions live in ``development_distribution_sail.proportions``.
CATEGORICAL_INDICATORS = NON_BINARY_TERMS["bmi_value"] + NON_BINARY_TERMS["smoking_status"] + NON_BINARY_TERMS["alcohol_category"]

COVERAGE_METHOD = ("M-11 approximation: sum over available terms of beta^2 * var / sum over all terms; indicator var = p(1 - p) "
                   "with SAIL p (sex=male: 1 - p_female); age var = SD^2; polypharmacy log-term var = normal approximation "
                   "from the IQR; covariances ignored; published (unmodified) coefficients")


def design_columns(binary_names: Sequence[str]) -> list[str]:
    """Design columns of the ``efalls_published`` representation, in the order the adapter requires."""
    return [c for columns in NON_BINARY_TERMS.values() for c in columns] + list(binary_names)


def find_project_file(path: str | Path, *, anchor: str | Path | None = None) -> Path:
    """Resolve a declared project file with :func:`falls_ml.paths.resolve_path` (the single path rule); must be a file."""
    resolved = resolve_path(path, anchor=anchor)
    if not resolved.is_file():
        raise ConfigError(f"Project file {str(path)!r} resolved to {resolved}, which is not a file")
    return resolved


# ---------------------------------------------------------------------- validation helpers
def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ConfigError(f"{where}: expected a finite number, got {value!r}")
    return float(value)


def _proportion(value: Any, where: str) -> float:
    p = _number(value, where)
    if not 0.0 < p < 1.0:
        raise ConfigError(f"{where}: proportion must lie in (0, 1), got {p}")
    return p


def _exact_keys(mapping: Mapping[str, Any], expected: Sequence[str], where: str) -> None:
    if set(mapping) != set(expected):
        raise ConfigError(f"{where}: expected keys {sorted(expected)}, got {sorted(mapping)}")


# ---------------------------------------------------------------------- published coefficients
@dataclass(frozen=True)
class PublishedEfallsCoefficients:
    """Published coefficients (S2 Table S3.2), sex parameterisations (D-01), recalibration reference (D-07) and
    the SAIL development distribution summaries used by the M-11 coverage metric. Treat mappings as read-only."""

    age: float
    log_polypharmacy: float
    bmi: Mapping[str, float]
    smoking_current: float
    alcohol: Mapping[str, float]
    binary: Mapping[str, float]
    sex_parameterisations: Mapping[str, Mapping[str, float]]
    recalibration_reference: Mapping[str, Any]
    development_distribution_sail: Mapping[str, Any]
    config_sha256: str

    @classmethod
    def from_yaml(cls, path: str | Path, *, anchor: str | Path | None = None) -> Self:
        """Load ``configs/models/efalls_published.yaml`` (resolved with :func:`find_project_file` from ``anchor``)."""
        resolved = find_project_file(path, anchor=anchor)
        content = resolved.read_bytes()
        try:
            raw = yaml.safe_load(content)
            terms = raw["terms"]
            problems = [msg for ok, msg in (
                (raw["link"] == "logit", "link must be 'logit'"),
                (terms["age_years"]["transform"] == "identity", "age_years transform must be 'identity'"),
                (terms["polypharmacy_count_120d"]["transform"] == POLYPHARMACY_TRANSFORM,
                 f"polypharmacy transform must be {POLYPHARMACY_TRANSFORM!r}"),
                (terms["bmi_category"]["reference"] == "overweight", "bmi_category reference must be 'overweight'"),
                (terms["smoking"]["reference"] == "ex_never", "smoking reference must be 'ex_never'"),
                (terms["alcohol_category"]["reference"] == "lower_risk", "alcohol_category reference must be 'lower_risk'"),
            ) if not ok]
            if problems:
                raise ConfigError(f"{resolved}: {problems}")
            _exact_keys(terms["smoking"]["levels"], ["current"], f"{resolved}: terms.smoking.levels")
            flat = {
                "age": terms["age_years"]["coefficient"],
                "log_polypharmacy": terms["polypharmacy_count_120d"]["coefficient"],
                "bmi": terms["bmi_category"]["levels"],
                "smoking_current": terms["smoking"]["levels"]["current"],
                "alcohol": terms["alcohol_category"]["levels"],
                "binary": terms["binary"],
                "sex_parameterisations": raw["sex_parameterisations"],
                "recalibration_reference": raw["recalibration_reference_connected_bradford"],
                "development_distribution_sail": raw["development_distribution_sail"],
                "config_sha256": hashlib.sha256(content).hexdigest(),
            }
        except (KeyError, TypeError, yaml.YAMLError) as exc:
            raise ConfigError(f"{resolved}: malformed published coefficients file ({exc!r})") from exc
        coefficients = cls.from_dict(flat)
        log.info("published_efalls_coefficients_loaded",
                 extra_fields={"path": str(resolved), "config_sha256": coefficients.config_sha256,
                               "n_binary": len(coefficients.binary)})
        return coefficients

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Validated construction from :meth:`to_dict` output (no YAML needed)."""
        try:
            _exact_keys(data["bmi"], BMI_LEVELS, "bmi")
            _exact_keys(data["alcohol"], ALCOHOL_LEVELS, "alcohol")
            _exact_keys(data["sex_parameterisations"], SEX_PARAMETERISATIONS, "sex_parameterisations")
            if not data["binary"]:
                raise ConfigError("binary: no binary coefficients")
            rec = data["recalibration_reference"]
            if rec["applies_to"] not in SEX_PARAMETERISATIONS:
                raise ConfigError(f"recalibration_reference.applies_to {rec['applies_to']!r} is not a sex parameterisation")
            dist = dict(data["development_distribution_sail"])
            proportions = dist["proportions"]
            _exact_keys(proportions, ("sex=female", *CATEGORICAL_INDICATORS), "development_distribution_sail.proportions")
            for key in ("age_years_sd", "polypharmacy_log_term_sd_approx"):
                dist[key] = _number(dist[key], f"development_distribution_sail.{key}")
                if dist[key] <= 0:
                    raise ConfigError(f"development_distribution_sail.{key} must be positive")
            dist["proportions"] = {k: _proportion(v, f"proportions[{k}]") for k, v in proportions.items()}
            return cls(
                age=_number(data["age"], "age"),
                log_polypharmacy=_number(data["log_polypharmacy"], "log_polypharmacy"),
                bmi={level: _number(data["bmi"][level], f"bmi[{level}]") for level in BMI_LEVELS},
                smoking_current=_number(data["smoking_current"], "smoking_current"),
                alcohol={level: _number(data["alcohol"][level], f"alcohol[{level}]") for level in ALCOHOL_LEVELS},
                binary={str(k): _number(v, f"binary[{k}]") for k, v in data["binary"].items()},
                sex_parameterisations={
                    v: {side: _number(data["sex_parameterisations"][v][side], f"{v}.{side}")
                        for side in ("intercept_female", "intercept_male")}
                    for v in SEX_PARAMETERISATIONS},
                recalibration_reference={
                    "alpha": _number(rec["alpha"], "recalibration_reference.alpha"),
                    "beta": _number(rec["beta"], "recalibration_reference.beta"),
                    "applies_to": rec["applies_to"], "use": str(rec.get("use", "")), "source": str(rec.get("source", ""))},
                development_distribution_sail=dist,
                config_sha256=str(data["config_sha256"]),
            )
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise ConfigError(f"malformed published eFalls coefficients: missing or invalid entry {exc!r}") from exc

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable copy (floats round-trip exactly through JSON)."""
        dist = dict(self.development_distribution_sail)
        dist["proportions"] = dict(dist["proportions"])
        return {"age": self.age, "log_polypharmacy": self.log_polypharmacy, "bmi": dict(self.bmi),
                "smoking_current": self.smoking_current, "alcohol": dict(self.alcohol), "binary": dict(self.binary),
                "sex_parameterisations": {k: dict(v) for k, v in self.sex_parameterisations.items()},
                "recalibration_reference": dict(self.recalibration_reference),
                "development_distribution_sail": dist, "config_sha256": self.config_sha256}

    def design_coefficients(self, variant: str) -> tuple[float, dict[str, float]]:
        """``(intercept, {design column: β})`` for a sex parameterisation (D-01).

        intercept = intercept_female; β[sex=male] = intercept_male − intercept_female.
        """
        if variant not in SEX_PARAMETERISATIONS:
            raise ConfigError(f"Unknown sex_parameterisation {variant!r}; allowed: {list(SEX_PARAMETERISATIONS)}")
        sp = self.sex_parameterisations[variant]
        coefs = {AGE_COLUMN: self.age, POLYPHARMACY_COLUMN: self.log_polypharmacy,
                 SEX_COLUMN: sp["intercept_male"] - sp["intercept_female"]}
        coefs.update({f"bmi_category={level}": beta for level, beta in self.bmi.items()})
        coefs["smoking=current"] = self.smoking_current
        coefs.update({f"alcohol_category={level}": beta for level, beta in self.alcohol.items()})
        coefs.update(self.binary)
        return sp["intercept_female"], coefs


def published_linear_predictor(X: pd.DataFrame, intercept: float, coefs: Mapping[str, float]) -> np.ndarray:
    """``LP = intercept + Σ β_j x_j`` (spec §5.10). ``X`` columns must be exactly the keys of ``coefs``."""
    missing = [c for c in coefs if c not in X.columns]
    extra = [c for c in X.columns if c not in coefs]
    duplicated = sorted({str(c) for c in X.columns[X.columns.duplicated()]})
    if missing or extra or duplicated:
        raise PreprocessingMismatchError(
            f"design/coefficient mismatch (missing={missing[:5]}, extra={extra[:5]}, duplicated={duplicated[:5]})")
    values = _float_matrix(X)
    return float(intercept) + values @ np.array([coefs[c] for c in X.columns], dtype=np.float64)


def _float_matrix(X: pd.DataFrame) -> np.ndarray:
    # Object/string columns would be silently coerced ("1" -> 1.0) by to_numpy; the design must already be numeric.
    non_numeric = [str(c) for c, dtype in X.dtypes.items()
                   if not (pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype))]
    if non_numeric:
        raise PreprocessingMismatchError(f"design matrix has non-numeric columns {non_numeric[:5]}")
    try:
        values = X.to_numpy(dtype=np.float64, na_value=np.nan)
    except (TypeError, ValueError) as exc:
        raise PreprocessingMismatchError(f"design matrix is not numeric: {exc}") from exc
    if not np.isfinite(values).all():
        bad = ~np.isfinite(values)
        cols = [str(c) for c, b in zip(X.columns, bad.any(axis=0)) if b]
        raise PreprocessingMismatchError(f"design matrix has {int(bad.sum())} missing/non-finite values in columns {cols[:5]}")
    return values


# ---------------------------------------------------------------------- feature metadata + coverage (M-11)
@dataclass(frozen=True)
class PublishedFeatureMetadata:
    """Feature-spec facts the adapter needs (binary order, SAIL prevalence, development support, M-11 rule).

    Serialised inside ``adapter.json`` so that loading a saved adapter never needs the YAML files.
    """

    binary_names: tuple[str, ...]
    sail_proportion: Mapping[str, float]
    development_support: Mapping[str, str]
    mandatory_for_efalls_label: tuple[str, ...]
    coverage_threshold: float
    feature_spec_sha256: str

    def __post_init__(self) -> None:
        names = list(self.binary_names)
        if not names or len(set(names)) != len(names):
            raise ConfigError("feature metadata: binary names must be non-empty and unique")
        _exact_keys(self.sail_proportion, names, "feature metadata sail_proportion")
        _exact_keys(self.development_support, names, "feature metadata development_support")
        for name in names:
            _proportion(self.sail_proportion[name], f"{name}.published_prevalence.sail_proportion")
            if self.development_support[name] not in {"low", "adequate"}:
                raise ConfigError(f"{name}.development_support must be 'low' or 'adequate'")
        if not self.mandatory_for_efalls_label:
            raise ConfigError("cohort.mandatory_for_efalls_label is empty; the M-11 hard rule would be silently disabled")
        unknown = sorted(set(self.mandatory_for_efalls_label) - set(self.feature_terms()))
        if unknown:
            raise ConfigError(f"cohort.mandatory_for_efalls_label names unknown features {unknown}")
        if not 0.0 < _number(self.coverage_threshold, "cohort.coverage_threshold") <= 1.0:
            raise ConfigError("cohort.coverage_threshold must lie in (0, 1]")

    @classmethod
    def from_feature_spec(cls, spec: FeatureSpec) -> Self:
        """Extract metadata from a loaded feature spec (binary ``published_prevalence``/``development_support``)."""
        missing = [name for name in NON_BINARY_TERMS if name not in spec.predictor_names()]
        if missing:
            raise ConfigError(f"feature spec {spec.name} lacks non-binary eFalls predictors {missing}")
        binaries = [spec.get(name) for name in spec.binary_names()]
        try:
            proportions = {f.name: f.raw["published_prevalence"]["sail_proportion"] for f in binaries}
            support = {f.name: f.raw["development_support"] for f in binaries}
        except (KeyError, TypeError) as exc:
            raise ConfigError(f"feature spec {spec.name}: binary predictor lacks published_prevalence/development_support ({exc!r})") from exc
        cohort = spec.cohort
        if not isinstance(cohort, Mapping) or not cohort:
            raise ConfigError(f"feature spec {spec.name}: missing cohort section (M-11 rule and coverage threshold)")
        mandatory = cohort.get("mandatory_for_efalls_label")
        if not isinstance(mandatory, list):
            raise ConfigError(f"feature spec {spec.name}: cohort.mandatory_for_efalls_label must be a list (M-11 hard rule)")
        return cls(binary_names=tuple(f.name for f in binaries), sail_proportion=proportions, development_support=support,
                   mandatory_for_efalls_label=tuple(mandatory),
                   coverage_threshold=cohort.get("coverage_threshold"), feature_spec_sha256=spec.content_sha256)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        try:
            return cls(binary_names=tuple(data["binary_names"]), sail_proportion=dict(data["sail_proportion"]),
                       development_support=dict(data["development_support"]),
                       mandatory_for_efalls_label=tuple(data["mandatory_for_efalls_label"]),
                       coverage_threshold=data["coverage_threshold"], feature_spec_sha256=str(data["feature_spec_sha256"]))
        except (KeyError, TypeError) as exc:
            raise ConfigError(f"malformed published feature metadata: {exc!r}") from exc

    def to_dict(self) -> dict[str, Any]:
        return {"binary_names": list(self.binary_names), "sail_proportion": dict(self.sail_proportion),
                "development_support": dict(self.development_support),
                "mandatory_for_efalls_label": list(self.mandatory_for_efalls_label),
                "coverage_threshold": self.coverage_threshold, "feature_spec_sha256": self.feature_spec_sha256}

    def feature_terms(self) -> dict[str, tuple[str, ...]]:
        """Spec feature name → design columns (binary features map to themselves)."""
        return {**NON_BINARY_TERMS, **{name: (name,) for name in self.binary_names}}


def coverage_report(unavailable: Iterable[str], coefficients: PublishedEfallsCoefficients,
                    feature_spec: FeatureSpec | PublishedFeatureMetadata, *, variant: str = PRIMARY_VARIANT) -> dict[str, Any]:
    """M-11 feature-coverage report for a list of unavailable spec features (binary or non-binary names).

    ``lp_variance_share_available`` = Σ_available β²·var / Σ_all β²·var (see ``COVERAGE_METHOD``). The run is
    labelled ``efalls_partial_scoring`` if a mandatory predictor is unavailable or the share is below the threshold.
    ``variant`` only sets the ``sex=male`` coefficient shown in the unavailable table (β² is variant-invariant).
    """
    meta = feature_spec if isinstance(feature_spec, PublishedFeatureMetadata) else PublishedFeatureMetadata.from_feature_spec(feature_spec)
    names = list(unavailable)
    terms = meta.feature_terms()
    unknown = sorted({n for n in names if n not in terms})
    if unknown:
        raise ConfigError(f"Unavailable predictors not in the feature spec: {unknown}")
    if len(set(names)) != len(names):
        raise ConfigError(f"Duplicate unavailable predictors: {names}")
    _, coefs = coefficients.design_coefficients(variant)
    dist = coefficients.development_distribution_sail
    prevalence: dict[str, float | None] = {AGE_COLUMN: None, POLYPHARMACY_COLUMN: None,
                                           SEX_COLUMN: 1.0 - dist["proportions"]["sex=female"],
                                           **{c: dist["proportions"][c] for c in CATEGORICAL_INDICATORS},
                                           **meta.sail_proportion}
    variance = {c: (p * (1.0 - p) if p is not None else math.nan) for c, p in prevalence.items()}
    variance[AGE_COLUMN] = dist["age_years_sd"] ** 2
    variance[POLYPHARMACY_COLUMN] = dist["polypharmacy_log_term_sd_approx"] ** 2
    columns = design_columns(meta.binary_names)
    unavailable_columns = [c for n in names for c in terms[n]]
    excluded = set(unavailable_columns)
    contribution = {c: coefs[c] ** 2 * variance[c] for c in columns}
    total = math.fsum(contribution[c] for c in columns)
    share = math.fsum(contribution[c] for c in columns if c not in excluded) / total
    mandatory_unavailable = [m for m in meta.mandatory_for_efalls_label if m in names]
    label = PARTIAL_LABEL if mandatory_unavailable or share < meta.coverage_threshold else PUBLISHED_LABEL
    rows = [{"feature": c, "coefficient": coefs[c], "sail_prevalence": prevalence[c],
             "expected_mean_lp_shift": -coefs[c] * prevalence[c] if prevalence[c] is not None else None}
            for c in unavailable_columns]
    return {"lp_variance_share_available": share, "threshold": meta.coverage_threshold,
            "mandatory_unavailable": mandatory_unavailable, "effective_experiment_label": label,
            "efalls_feature_coverage": "incomplete" if names else "complete", "unavailable_predictors": rows,
            "variant": variant, "method": COVERAGE_METHOD}


# ---------------------------------------------------------------------- adapter
class PublishedEfallsModel(ModelAdapter):
    """Fixed published eFalls equation (§5.10) with the D-01 sex parameterisations, M-11 fills and D-20 option.

    params:
      config: path to ``configs/models/efalls_published.yaml`` (required)
      sex_parameterisation: one of ``SEX_PARAMETERISATIONS`` (required; no default, D-01 item 7)
      unavailable_predictors: binary predictor names that cannot be recorded (columns must be all zero; default [])
      unavailable_fill: ``zero`` (primary, M-11) or ``sail_prevalence`` (adds Σ β·p_SAIL of those terms to the intercept)
      zero_low_support_predictors: set β = 0 for binaries with ``development_support == 'low'`` (D-20; default False)
      feature_spec: feature spec path (default ``configs/features/efalls_v1.yaml``); ``config`` is resolved from it
    """

    name = "efalls_published"
    representation = "efalls_published"
    is_linear = True
    requires_fit = False

    _PARAM_KEYS = frozenset({"config", "sex_parameterisation", "unavailable_predictors", "unavailable_fill",
                             "zero_low_support_predictors", "feature_spec"})

    def __init__(self, params: Mapping[str, Any] | None = None, *, random_state: int = 0,
                 coefficients: PublishedEfallsCoefficients | None = None,
                 feature_metadata: PublishedFeatureMetadata | None = None):
        """``coefficients``/``feature_metadata`` bypass the YAML files (used by :meth:`load`); pass both or neither."""
        super().__init__(params, random_state=random_state)
        self._normalise_params()
        if coefficients is None and feature_metadata is None:
            spec_path = find_project_file(self.params["feature_spec"])
            coefficients = PublishedEfallsCoefficients.from_yaml(self.params["config"], anchor=spec_path)
            feature_metadata = PublishedFeatureMetadata.from_feature_spec(load_feature_spec(spec_path))
        elif coefficients is None or feature_metadata is None:
            raise ConfigError("pass both coefficients and feature_metadata, or neither")
        self._configure(coefficients, feature_metadata)

    # ------------------------------------------------------------------ configuration
    def _normalise_params(self) -> None:
        p = self.params
        unknown = sorted(set(p) - self._PARAM_KEYS)
        if unknown:
            raise ConfigError(f"{self.name}: unknown params {unknown} (allowed: {sorted(self._PARAM_KEYS)})")
        if not p.get("config"):
            raise ConfigError(f"{self.name}: params.config (published coefficients YAML) is required")
        if p.get("sex_parameterisation") not in SEX_PARAMETERISATIONS:
            raise ConfigError(f"{self.name}: params.sex_parameterisation must be stated explicitly as one of "
                              f"{list(SEX_PARAMETERISATIONS)} (D-01; no default), got {p.get('sex_parameterisation')!r}")
        unavailable = p.get("unavailable_predictors", [])
        if isinstance(unavailable, str) or not isinstance(unavailable, (list, tuple)) \
                or not all(isinstance(u, str) for u in unavailable) or len(set(unavailable)) != len(unavailable):
            raise ConfigError(f"{self.name}: unavailable_predictors must be a list of unique names, got {unavailable!r}")
        fill = p.get("unavailable_fill", "zero")
        if fill not in UNAVAILABLE_FILLS:
            raise ConfigError(f"{self.name}: unavailable_fill must be one of {list(UNAVAILABLE_FILLS)}, got {fill!r}")
        zero_low = p.get("zero_low_support_predictors", False)
        if not isinstance(zero_low, bool):
            raise ConfigError(f"{self.name}: zero_low_support_predictors must be true/false")
        p.update(unavailable_predictors=list(unavailable), unavailable_fill=fill, zero_low_support_predictors=zero_low,
                 feature_spec=str(p.get("feature_spec", DEFAULT_FEATURE_SPEC)), config=str(p["config"]))

    def _configure(self, coefficients: PublishedEfallsCoefficients, metadata: PublishedFeatureMetadata) -> None:
        if set(coefficients.binary) != set(metadata.binary_names):
            diff = sorted(set(coefficients.binary) ^ set(metadata.binary_names))
            raise ConfigError(f"{self.name}: binary predictors differ between coefficients and feature spec: {diff[:10]}")
        unavailable = self.params["unavailable_predictors"]
        not_binary = [u for u in unavailable if u not in metadata.binary_names]
        if not_binary:
            raise ConfigError(f"{self.name}: unavailable_predictors must be binary spec predictors; got {not_binary} "
                              "(age, sex, polypharmacy and categorical terms cannot be zero-filled)")
        self.coefficients = coefficients
        self.feature_metadata = metadata
        self._sex_parameterisation: str = self.params["sex_parameterisation"]
        self.design_columns: list[str] = design_columns(metadata.binary_names)
        self.low_support_zeroed: list[str] = ([n for n in metadata.binary_names if metadata.development_support[n] == "low"]
                                              if self.params["zero_low_support_predictors"] else [])
        self._unavailable_idx = [self.design_columns.index(u) for u in unavailable]
        self._indicator_idx = [i for i, c in enumerate(self.design_columns) if c not in (AGE_COLUMN, POLYPHARMACY_COLUMN)]
        self._equations = {v: self._equation(v) for v in SEX_PARAMETERISATIONS}
        report = self.coverage()
        fields = {"sex_parameterisation": self.sex_parameterisation, "config_sha256": coefficients.config_sha256,
                  "unavailable_predictors": unavailable, "unavailable_fill": self.params["unavailable_fill"],
                  "unavailable_intercept_shift": self.unavailable_intercept_shift, "low_support_zeroed": self.low_support_zeroed,
                  "lp_variance_share_available": report["lp_variance_share_available"],
                  "effective_experiment_label": report["effective_experiment_label"]}
        log.info("published_efalls_configured", extra_fields=fields)
        if report["effective_experiment_label"] == PARTIAL_LABEL:
            log.warning("efalls_partial_scoring", extra_fields={"mandatory_unavailable": report["mandatory_unavailable"],
                                                                "lp_variance_share_available": report["lp_variance_share_available"]})

    @property
    def sex_parameterisation(self) -> str:
        """The configured D-01 sex parameterisation (read-only; fixed at construction)."""
        return self._sex_parameterisation

    @property
    def unavailable_intercept_shift(self) -> float:
        """Σ β·p_SAIL over unavailable terms when ``unavailable_fill == 'sail_prevalence'`` (after D-20 zeroing), else 0."""
        if self.params["unavailable_fill"] != "sail_prevalence":
            return 0.0
        zeroed = set(self.low_support_zeroed)
        return math.fsum(0.0 if u in zeroed else self.coefficients.binary[u] * self.feature_metadata.sail_proportion[u]
                         for u in self.params["unavailable_predictors"])

    def _equation(self, variant: str) -> tuple[float, dict[str, float], np.ndarray]:
        intercept, coefs = self.coefficients.design_coefficients(variant)
        for name in self.low_support_zeroed:
            coefs[name] = 0.0
        intercept += self.unavailable_intercept_shift
        return intercept, coefs, np.array([coefs[c] for c in self.design_columns], dtype=np.float64)

    def equation(self, variant: str | None = None) -> tuple[float, dict[str, float]]:
        """Effective ``(intercept, coefficients)`` after fill and low-support options (default: configured variant)."""
        intercept, coefs, _ = self._equations[self._variant(variant)]
        return intercept, dict(coefs)

    def _variant(self, variant: str | None) -> str:
        variant = self.sex_parameterisation if variant is None else variant
        if variant not in SEX_PARAMETERISATIONS:
            raise ConfigError(f"Unknown sex_parameterisation {variant!r}; allowed: {list(SEX_PARAMETERISATIONS)}")
        return variant

    # ------------------------------------------------------------------ design validation
    def _design_matrix(self, X: pd.DataFrame) -> np.ndarray:
        if not isinstance(X, pd.DataFrame):
            raise PreprocessingMismatchError(f"{self.name}: expected a design DataFrame, got {type(X).__name__}")
        if list(X.columns) != self.design_columns:
            missing = [c for c in self.design_columns if c not in X.columns]
            extra = [c for c in X.columns if c not in self.design_columns]
            raise PreprocessingMismatchError(f"{self.name}: design columns differ from representation efalls_published "
                                             f"(missing={missing[:5]}, extra={extra[:5]}, order_changed={not missing and not extra})")
        values = _float_matrix(X)
        not_binary = [self.design_columns[i] for i in self._indicator_idx
                      if not ((values[:, i] == 0.0) | (values[:, i] == 1.0)).all()]
        if not_binary:
            raise PreprocessingMismatchError(f"{self.name}: indicator columns with values outside {{0, 1}}: {not_binary[:5]}")
        # A category has at most one active non-reference level (all zero = reference level; §5.6, §5.8).
        for feature in ("bmi_value", "alcohol_category"):
            idx = [self.design_columns.index(c) for c in NON_BINARY_TERMS[feature]]
            n_bad = int((values[:, idx].sum(axis=1) > 1.0).sum())
            if n_bad:
                raise PreprocessingMismatchError(f"{self.name}: {n_bad} rows with more than one active {feature} level")
        # ln((P + 1) / 10) >= ln(0.1) for every count P >= 0 (§5.5); smaller values mean a wrong or negative input.
        n_bad = int((values[:, self.design_columns.index(POLYPHARMACY_COLUMN)] < _MIN_LOG_POLYPHARMACY).sum())
        if n_bad:
            raise PreprocessingMismatchError(f"{self.name}: {n_bad} rows with {POLYPHARMACY_COLUMN} < ln(0.1) (impossible count)")
        nonzero = [self.design_columns[i] for i in self._unavailable_idx if values[:, i].any()]
        if nonzero:
            raise ConfigError(f"{self.name}: predictors declared unavailable have non-zero values: {nonzero} (M-11)")
        return values

    # ------------------------------------------------------------------ core
    def fit(self, X: pd.DataFrame, y: np.ndarray | None = None, *, groups: np.ndarray | None = None) -> Self:
        """No estimation (fixed coefficients); validates the design, ignores labels and records ``feature_names_``."""
        self._design_matrix(X)
        self.feature_names_ = list(self.design_columns)
        log.info("published_efalls_fit_noop", extra_fields={"n_rows": len(X), "sex_parameterisation": self.sex_parameterisation})
        return self

    def linear_predictor(self, X: pd.DataFrame) -> np.ndarray:
        intercept, _, beta = self._equations[self.sex_parameterisation]
        return intercept + self._design_matrix(X) @ beta

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return expit(self.linear_predictor(X))

    def score_all_variants(self, X: pd.DataFrame) -> pd.DataFrame:
        """``lp_<variant>``, ``risk_<variant>`` for all four sex parameterisations (D-01), same fill/support options."""
        values = self._design_matrix(X)
        out: dict[str, np.ndarray] = {}
        for variant in SEX_PARAMETERISATIONS:
            intercept, _, beta = self._equations[variant]
            lp = intercept + values @ beta
            out[f"lp_{variant}"] = lp
            out[f"risk_{variant}"] = expit(lp)
        return pd.DataFrame(out, index=X.index)

    def coverage(self) -> dict[str, Any]:
        """M-11 coverage report for the configured unavailable predictors."""
        return coverage_report(self.params["unavailable_predictors"], self.coefficients, self.feature_metadata,
                               variant=self.sex_parameterisation)

    def recalibration_reference(self) -> dict[str, Any]:
        """Connected Bradford recalibration expit(alpha + beta·LP_C) — regression-test reference only (D-07)."""
        rec = self.coefficients.recalibration_reference
        return {"alpha": rec["alpha"], "beta": rec["beta"], "applies_to": rec["applies_to"],
                "note": f"Regression-test reference only; never applied to Meuhedet (D-07). Source: {rec['source']}"}

    def get_feature_importance(self) -> pd.DataFrame:
        _, _, beta = self._equations[self.sex_parameterisation]
        return coefficient_importance(self.design_columns, beta, standardized_scale=False)

    def fit_diagnostics(self) -> dict[str, Any]:
        intercept, _, _ = self._equations[self.sex_parameterisation]
        return {"published_equation": {"sex_parameterisation": self.sex_parameterisation, "intercept": intercept,
                                       "unavailable_fill": self.params["unavailable_fill"],
                                       "unavailable_intercept_shift": self.unavailable_intercept_shift,
                                       "low_support_zeroed": list(self.low_support_zeroed),
                                       "config_sha256": self.coefficients.config_sha256, "coverage": self.coverage()}}

    # ------------------------------------------------------------------ persistence
    def _equation_sha256(self) -> str:
        """Digest of the persisted coefficients and feature metadata (detects edits to ``adapter.json``)."""
        payload = {"coefficients": self.coefficients.to_dict(), "feature_metadata": self.feature_metadata.to_dict()}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def save(self, directory: Path) -> None:
        self._write_meta(Path(directory), extra={
            "design_columns": self.design_columns, "config_sha256": self.coefficients.config_sha256,
            "feature_spec_sha256": self.feature_metadata.feature_spec_sha256, "equation_sha256": self._equation_sha256(),
            "coefficients": self.coefficients.to_dict(), "feature_metadata": self.feature_metadata.to_dict()})

    @classmethod
    def load(cls, directory: Path) -> Self:
        """Rebuild from ``adapter.json`` alone (no YAML access); predictions are bit-identical."""
        try:
            meta = cls._read_meta(Path(directory))
            if meta["adapter"] != cls.name or meta["representation"] != cls.representation:
                raise BundleIntegrityError(f"adapter.json belongs to {meta['adapter']!r}/{meta['representation']!r}, "
                                           f"not {cls.name!r}/{cls.representation!r}")
            model = cls(meta["params"], random_state=meta["random_state"],
                        coefficients=PublishedEfallsCoefficients.from_dict(meta["coefficients"]),
                        feature_metadata=PublishedFeatureMetadata.from_dict(meta["feature_metadata"]))
            consistent = (model.design_columns == meta["design_columns"]
                          and model.coefficients.config_sha256 == meta["config_sha256"]
                          and model._equation_sha256() == meta["equation_sha256"]
                          and meta["feature_names"] in (None, model.design_columns))
        except (OSError, ValueError, KeyError, TypeError, ConfigError) as exc:
            raise BundleIntegrityError(f"{cls.name}: cannot load adapter from {directory}: {exc}") from exc
        if not consistent:
            raise BundleIntegrityError(f"{cls.name}: adapter.json is inconsistent (design columns, feature names, "
                                       "config hash or equation digest)")
        model.feature_names_ = meta["feature_names"]
        return model
