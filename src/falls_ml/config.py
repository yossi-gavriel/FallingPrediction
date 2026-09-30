"""Experiment configuration: YAML → frozen, validated dataclasses.

Unknown keys are errors (typos must not silently fall back to defaults). Every important
experimental decision is visible in the YAML file and copied verbatim into the run directory.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, get_args, get_origin, get_type_hints

import yaml

from falls_ml.errors import ConfigError

if typing.TYPE_CHECKING:
    from falls_ml.features.spec import FeatureSpec

ExperimentKind = Literal["efalls_published_scoring", "efalls_retrained", "efalls_retrained_reduced", "alternative_model", "ablation_member"]
#: experiment kinds labelled eFalls (purity rules apply); the reduced kind is never a full eFalls reproduction
EFALLS_KINDS = frozenset({"efalls_published_scoring", "efalls_retrained", "efalls_retrained_reduced"})
#: kinds that retrain with the published eFalls learning process (D-10, D-13, D-14)
RETRAINED_KINDS = frozenset({"efalls_retrained", "efalls_retrained_reduced"})
REDUCED_KIND = "efalls_retrained_reduced"
#: predictor that every reduced eFalls predictor set must keep (age is the dominant published term)
REDUCED_REQUIRED_FEATURES = ("age_years",)
EFALLS_CANDIDATE_COUNT = 78  # published eFalls candidate predictors: 72 binary + 6 non-binary (spec §5, Appendix B)
SplitStrategy = Literal["temporal", "group_holdout", "patient_grouped_random"]
CalibrationMethod = Literal["none", "intercept_only", "logistic_intercept_slope"]
SexParameterisation = Literal["lp_c_box_s3_1", "lp_a_table_s3_2", "lp_b_label_swap", "lp_d2_numeric_swap"]
SubgroupVariable = Literal["sex", "bmi_category", "age_band"]
#: LASSO params that keep the published learning process (spec D-14); anything else is not "efalls_retrained"
RETRAINED_LASSO_PARAMS = frozenset({"n_lambda", "selection", "lambda_min_ratio", "tol", "stop", "cv_tolerance", "cv_confirm",
                                    "refit_unpenalized", "fold_standardization"})
#: experiment.name becomes part of a run directory name: portable file-name characters only (Windows forbids : * ? " < > | \ /
#: and a trailing dot or space); derived names (``<name>__<model>``, ablation steps) must stay valid too
EXPERIMENT_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}")

LAYERS = ("L1_published", "L2_assumption", "L3a_meuhedet_mapping", "L3b_meuhedet_predictor", "L4_alternative")


@dataclass(frozen=True)
class ExperimentSection:
    name: str
    kind: ExperimentKind
    description: str
    layers: tuple[str, ...]


@dataclass(frozen=True)
class DatasetSection:
    feature_spec: str
    path: str | None = None
    feature_spec_extensions: tuple[str, ...] = ()
    require_scientific_use: bool = False


@dataclass(frozen=True)
class PublishedEquationSection:
    """Experiment A options (spec D-01, D-20, M-11)."""

    config: str
    sex_parameterisation: SexParameterisation
    score_all_variants: bool = True
    unavailable_predictors: tuple[str, ...] = ()
    unavailable_fill: Literal["zero", "sail_prevalence"] = "zero"
    zero_low_support_predictors: bool = False


@dataclass(frozen=True)
class ModelSection:
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    published_equation: PublishedEquationSection | None = None


@dataclass(frozen=True)
class FractionalPolynomialSection:
    mode: Literal["select", "fixed_published", "linear"] = "select"
    max_degree: int = 2
    alpha: float = 0.05
    powers: tuple[float, ...] = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0)
    max_cycles: int = 5
    variables: tuple[str, ...] = ("age_years", "polypharmacy_count_120d")


@dataclass(frozen=True)
class PreprocessingSection:
    representation: Literal["efalls_published", "efalls_fp_all_levels", "efalls_reference_coded", "efalls_raw"]
    features: tuple[str, ...] | None = None
    fractional_polynomial: FractionalPolynomialSection = field(default_factory=FractionalPolynomialSection)
    bmi_obese_cutpoint: float = 30.0


@dataclass(frozen=True)
class TemporalSplit:
    train_end: str
    validation_end: str
    embargo_outcome_windows: bool = True


@dataclass(frozen=True)
class GroupHoldoutSplit:
    group_column: str
    validation_groups: tuple[str, ...]
    test_groups: tuple[str, ...]


@dataclass(frozen=True)
class RandomSplit:
    validation_fraction: float = 0.2
    test_fraction: float = 0.2
    stratify_outcome: bool = True


@dataclass(frozen=True)
class ValidationSection:
    strategy: SplitStrategy
    seed: int
    cv_folds: int = 10
    patients_disjoint: bool = True
    temporal: TemporalSplit | None = None
    group_holdout: GroupHoldoutSplit | None = None
    patient_grouped_random: RandomSplit | None = None
    limitation_note: str = ""
    #: documented outcome ascertainment/claims lag; the dataset freeze date must be >= last outcome window end + lag (D-19)
    outcome_lag_days: int = 0


@dataclass(frozen=True)
class ObjectiveSection:
    weights: dict[str, float] = field(default_factory=lambda: {"auroc": 1.0, "brier": -1.0, "calibration_slope_abs_error": -0.25})


@dataclass(frozen=True)
class TuningSection:
    enabled: bool = False
    method: Literal["grid", "random"] = "grid"
    n_iter: int = 20
    search_space: dict[str, list[Any]] | None = None
    objective: ObjectiveSection = field(default_factory=ObjectiveSection)


@dataclass(frozen=True)
class CalibrationSection:
    enabled: bool = True
    method: CalibrationMethod = "logistic_intercept_slope"
    #: serve recalibrated risks from the bundle and judge eligibility on them. None = by kind:
    #: False for efalls_published_scoring (serve the published equation), True otherwise.
    serve_recalibrated: bool | None = None

    def serves_recalibrated(self, kind: str) -> bool:
        if not self.enabled or self.method == "none":
            return False
        return (kind != "efalls_published_scoring") if self.serve_recalibrated is None else self.serve_recalibrated


@dataclass(frozen=True)
class BootstrapSection:
    n: int = 1000
    seed: int = 20240314
    cluster_column: str = "research_id"


@dataclass(frozen=True)
class EvaluationSection:
    thresholds: tuple[float, ...] = (0.10, 0.15, 0.20, 0.25)
    decision_curve_thresholds: tuple[float, float, float] = (0.01, 0.50, 0.01)
    calibration_groups: int = 20
    bootstrap: BootstrapSection = field(default_factory=BootstrapSection)
    subgroups: tuple[SubgroupVariable, ...] = ("sex", "bmi_category")
    permutation_importance_repeats: int = 10
    cluster_column_for_heterogeneity: str | None = None


@dataclass(frozen=True)
class StabilitySection:
    enabled: bool = True
    n_bootstrap: int = 200
    selection_threshold: float = 0.8
    sign_stability_threshold: float = 0.9
    permutation_repeats: int = 0          # 0 disables permutation-importance stability (cost: B x features x repeats)
    permutation_max_rows: int = 20000


@dataclass(frozen=True)
class OptimismSection:
    enabled: bool = False
    n_bootstrap: int = 25


@dataclass(frozen=True)
class IecvSection:
    enabled: bool = False
    cluster_column: str | None = None


@dataclass(frozen=True)
class AnalysisSection:
    stability: StabilitySection = field(default_factory=StabilitySection)
    optimism: OptimismSection = field(default_factory=OptimismSection)
    iecv: IecvSection = field(default_factory=IecvSection)


@dataclass(frozen=True)
class EligibilitySection:
    min_auroc: float = 0.70
    calibration_slope_range: tuple[float, float] = (0.8, 1.2)
    max_abs_citl: float = 0.2


@dataclass(frozen=True)
class RiskCategoriesSection:
    approved: bool = False
    cutpoints: tuple[float, ...] = ()
    labels: tuple[str, ...] = ()
    approval_reference: str = ""


@dataclass(frozen=True)
class ReportingSection:
    enabled: bool = True
    eligibility: EligibilitySection = field(default_factory=EligibilitySection)
    risk_categories: RiskCategoriesSection = field(default_factory=RiskCategoriesSection)
    top_n_features: int = 10


@dataclass(frozen=True)
class OutputSection:
    runs_dir: str = "runs"


@dataclass(frozen=True)
class ExperimentConfig:
    experiment: ExperimentSection
    dataset: DatasetSection
    model: ModelSection
    preprocessing: PreprocessingSection
    validation: ValidationSection
    tuning: TuningSection = field(default_factory=TuningSection)
    calibration: CalibrationSection = field(default_factory=CalibrationSection)
    evaluation: EvaluationSection = field(default_factory=EvaluationSection)
    analysis: AnalysisSection = field(default_factory=AnalysisSection)
    reporting: ReportingSection = field(default_factory=ReportingSection)
    output: OutputSection = field(default_factory=OutputSection)
    source_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d.pop("source_path", None)
        return _plain(d)

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, default=str).encode("utf-8")).hexdigest()

    def with_overrides(self, **sections: Any) -> ExperimentConfig:
        return dataclasses.replace(self, **sections)


# ---------------------------------------------------------------------- building
def _plain(o: Any) -> Any:
    if isinstance(o, dict):
        return {k: _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    return o


def _coerce(tp: Any, value: Any, where: str) -> Any:
    origin = get_origin(tp)
    args = get_args(tp)
    if value is None:
        if type(None) in args or tp is Any:
            return None
        raise ConfigError(f"{where}: value required")
    if origin is Literal:
        if value not in args:
            raise ConfigError(f"{where}: {value!r} not in {list(args)}")
        return value
    if origin is typing.Union or origin is types.UnionType:
        non_none = [a for a in args if a is not type(None)]
        errors = []
        for a in non_none:
            try:
                return _coerce(a, value, where)
            except ConfigError as exc:
                errors.append(str(exc))
        raise ConfigError(f"{where}: {value!r} does not match {tp} ({'; '.join(errors)})")
    if dataclasses.is_dataclass(tp):
        if not isinstance(value, dict):
            raise ConfigError(f"{where}: expected a mapping")
        return _build(tp, value, where)
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{where}: expected a list")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce(args[0], v, f"{where}[{i}]") for i, v in enumerate(value))
        if len(args) != len(value):
            raise ConfigError(f"{where}: expected {len(args)} items")
        return tuple(_coerce(a, v, f"{where}[{i}]") for i, (a, v) in enumerate(zip(args, value)))
    if origin is dict:
        if not isinstance(value, dict):
            raise ConfigError(f"{where}: expected a mapping")
        return dict(value)
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{where}: expected a number")
        return float(value)
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{where}: expected an integer")
        return value
    if tp is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{where}: expected true/false")
        return value
    if tp is str:
        if not isinstance(value, (str, int, float)) or isinstance(value, bool):
            raise ConfigError(f"{where}: expected a string")
        return str(value)
    if tp is Any:
        return value
    raise ConfigError(f"{where}: unsupported config type {tp}")


def _build(cls: type, data: dict[str, Any], where: str) -> Any:
    hints = get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - set(fields))
    if unknown:
        raise ConfigError(f"{where}: unknown keys {unknown} (allowed: {sorted(fields)})")
    kwargs = {}
    for name, f in fields.items():
        if name in data:
            kwargs[name] = _coerce(hints[name], data[name], f"{where}.{name}")
        elif f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
            raise ConfigError(f"{where}: missing required key {name!r}")
    return cls(**kwargs)


def validate_experiment_name(name: str) -> None:
    """Raise ``ConfigError`` unless ``name`` is safe as part of a directory name on Windows, macOS and Linux."""
    if not isinstance(name, str) or not EXPERIMENT_NAME_PATTERN.fullmatch(name) or name.endswith("."):
        raise ConfigError(f"experiment.name {name!r} is not a portable directory-name part: use 1-200 characters from "
                          "A-Z a-z 0-9 _ . - starting with a letter or digit and not ending with '.' (it is used in run directory names; "
                          "Windows forbids : * ? \" < > | \\ / and trailing dots)")


def _validate_semantics(cfg: ExperimentConfig) -> None:
    exp, model, pre, val = cfg.experiment, cfg.model, cfg.preprocessing, cfg.validation
    validate_experiment_name(exp.name)
    bad_layers = [x for x in exp.layers if x not in LAYERS]
    if bad_layers:
        raise ConfigError(f"experiment.layers: unknown layers {bad_layers}")
    if exp.kind == "efalls_published_scoring":
        if model.name != "efalls_published" or model.published_equation is None:
            raise ConfigError("efalls_published_scoring requires model.name=efalls_published and model.published_equation")
        if pre.representation != "efalls_published":
            raise ConfigError("efalls_published_scoring requires preprocessing.representation=efalls_published")
        if "L4_alternative" in exp.layers:
            raise ConfigError("the published eFalls experiment cannot include L4 alternative-model layers")
    if model.name == "efalls_published" and exp.kind != "efalls_published_scoring":
        raise ConfigError("model efalls_published may only be used by experiment kind efalls_published_scoring")
    if cfg.dataset.feature_spec_extensions and "L3b_meuhedet_predictor" not in exp.layers:
        raise ConfigError("configs that load feature_spec_extensions (Meuhedet predictors) must declare layer L3b_meuhedet_predictor")
    if exp.kind in EFALLS_KINDS and "L3b_meuhedet_predictor" in exp.layers:
        raise ConfigError("eFalls-labelled experiments cannot include L3b Meuhedet predictors (spec §1.2)")
    if exp.kind in RETRAINED_KINDS:
        if model.name != "lasso_logistic_cv":
            raise ConfigError(f"{exp.kind} must use model lasso_logistic_cv (published learning process)")
        if cfg.dataset.feature_spec_extensions:
            raise ConfigError(f"{exp.kind} cannot use feature-spec extensions (Meuhedet predictors contaminate the baseline)")
    strategy_section = {"temporal": val.temporal, "group_holdout": val.group_holdout, "patient_grouped_random": val.patient_grouped_random}
    if val.strategy != "patient_grouped_random" and strategy_section[val.strategy] is None:
        raise ConfigError(f"validation.strategy={val.strategy} requires validation.{val.strategy} section")
    if val.strategy == "patient_grouped_random" and not val.limitation_note:
        raise ConfigError("validation.strategy=patient_grouped_random requires validation.limitation_note justifying a non-temporal split")
    efalls_kind = exp.kind in EFALLS_KINDS
    if exp.kind == "efalls_published_scoring":
        pe = model.published_equation
        overlap = sorted(set(model.params) & {f.name for f in dataclasses.fields(PublishedEquationSection)} | ({"config"} & set(model.params)))
        if model.params:
            raise ConfigError(f"efalls_published_scoring: model.params must be empty; set published options in model.published_equation "
                              f"(found {sorted(model.params)}; overlapping {overlap})")
        if pe.sex_parameterisation != "lp_c_box_s3_1" or not pe.score_all_variants:
            raise ConfigError("efalls_published_scoring must use sex_parameterisation lp_c_box_s3_1 (primary, D-01) with score_all_variants=true; "
                              "the other readings are always co-reported in the same run")
        if cfg.dataset.feature_spec_extensions or pre.features is not None:
            raise ConfigError("efalls_published_scoring cannot use feature-spec extensions or a feature subset")
    if exp.kind == "efalls_retrained" and pre.features is not None:
        raise ConfigError("efalls_retrained uses all 78 eFalls candidates (D-12); preprocessing.features must be unset "
                          "(a declared subset of the eFalls candidates is experiment kind efalls_retrained_reduced)")
    if exp.kind == REDUCED_KIND:
        _validate_reduced_declaration(cfg)
    if exp.kind in RETRAINED_KINDS:
        if pre.representation != "efalls_fp_all_levels":
            raise ConfigError(f"{exp.kind} requires preprocessing.representation=efalls_fp_all_levels (D-10)")
        if pre.fractional_polynomial.mode not in {"select", "fixed_published"}:
            raise ConfigError(f"{exp.kind} requires fractional_polynomial.mode select or fixed_published (D-13)")
        bad = sorted(set(model.params) - RETRAINED_LASSO_PARAMS - {"cv_folds"})
        if bad:
            raise ConfigError(f"{exp.kind}: LASSO params {bad} are not part of the published learning process (D-14)")
        if model.params.get("selection", "min") != "min":
            raise ConfigError(f"{exp.kind} requires the lambda-min CV rule (D-14); run 1se as a separately named experiment")
        if val.cv_folds != 10 or model.params.get("cv_folds", 10) != 10:
            raise ConfigError(f"{exp.kind} requires 10-fold CV (published learning process)")
    if pre.features is not None and pre.representation == "efalls_fp_all_levels":
        outside = [v for v in pre.fractional_polynomial.variables if v not in pre.features]
        if outside:
            raise ConfigError(f"preprocessing.fractional_polynomial.variables {outside} are not in preprocessing.features; "
                              "list only available continuous predictors as FP variables")
    if efalls_kind and cfg.tuning.enabled:
        raise ConfigError("tuning is not allowed for eFalls-labelled experiments")
    if val.strategy == "temporal" and val.temporal is not None and not val.temporal.embargo_outcome_windows:
        if efalls_kind:
            raise ConfigError("eFalls-labelled experiments require the temporal outcome-window embargo (D-19)")
        if not val.limitation_note:
            raise ConfigError("disabling the outcome-window embargo requires validation.limitation_note")
    lo, hi, step = cfg.evaluation.decision_curve_thresholds
    if not (0 < lo < hi < 1 and step > 0):
        raise ConfigError("evaluation.decision_curve_thresholds must be (start, stop, step) with 0 < start < stop < 1 and step > 0")
    for name, value, minimum in (("evaluation.bootstrap.n", cfg.evaluation.bootstrap.n, 2), ("evaluation.calibration_groups", cfg.evaluation.calibration_groups, 2),
                                 ("validation.cv_folds", val.cv_folds, 2), ("analysis.stability.n_bootstrap", cfg.analysis.stability.n_bootstrap, 2),
                                 ("analysis.optimism.n_bootstrap", cfg.analysis.optimism.n_bootstrap, 2),
                                 ("evaluation.permutation_importance_repeats", cfg.evaluation.permutation_importance_repeats, 1),
                                 ("validation.outcome_lag_days", val.outcome_lag_days, 0)):
        if value < minimum:
            raise ConfigError(f"{name} must be >= {minimum}, got {value}")
    if cfg.analysis.iecv.enabled and not cfg.analysis.iecv.cluster_column:
        raise ConfigError("analysis.iecv.enabled requires analysis.iecv.cluster_column")
    from falls_ml.models.registry import get_adapter_class
    adapter_repr = get_adapter_class(model.name).representation
    if adapter_repr != pre.representation:
        raise ConfigError(f"model {model.name} requires preprocessing.representation={adapter_repr!r}, config has {pre.representation!r}")
    rc = cfg.reporting.risk_categories
    if rc.approved and (not rc.cutpoints or len(rc.labels) != len(rc.cutpoints) + 1 or not rc.approval_reference):
        raise ConfigError("reporting.risk_categories approved=true requires cutpoints, len(labels)=len(cutpoints)+1 and approval_reference")
    if any(not 0 < t < 1 for t in cfg.evaluation.thresholds):
        raise ConfigError("evaluation.thresholds must lie in (0, 1)")


def _validate_reduced_declaration(cfg: ExperimentConfig) -> None:
    """Spec-free rules of ``efalls_retrained_reduced``; the spec-level rules are :func:`check_feature_declaration`."""
    features = cfg.preprocessing.features
    if not features:
        raise ConfigError("efalls_retrained_reduced requires preprocessing.features: an explicit, non-empty list of the eFalls "
                          "baseline predictors available in the data (use kind efalls_retrained for all 78 candidates)")
    dupes = sorted({f for f in features if features.count(f) > 1})
    if dupes:
        raise ConfigError(f"efalls_retrained_reduced: preprocessing.features lists duplicates {dupes}")
    missing = [f for f in REDUCED_REQUIRED_FEATURES if f not in features]
    if missing:
        raise ConfigError(f"efalls_retrained_reduced: preprocessing.features must include {missing}")
    if "L4_alternative" in cfg.experiment.layers:
        raise ConfigError("efalls_retrained_reduced cannot include L4 alternative-model layers (published learning process only)")


def check_feature_declaration(cfg: ExperimentConfig, spec: FeatureSpec) -> None:
    """Spec-level feature rules checked before any data are read or a run directory is created.

    ``efalls_retrained_reduced``: ``preprocessing.features`` must be a STRICT subset of the eFalls baseline candidates
    (``exact_efalls_baseline: true``) of ``spec`` and include ``age_years``. All candidates -> use ``efalls_retrained``.
    Other kinds: declared features must exist in ``spec``.
    """
    features = cfg.preprocessing.features
    if cfg.experiment.kind in EFALLS_KINDS:
        n_candidates = sum(1 for f in spec.features if f.exact_efalls_baseline)
        if spec.subset_of is None and n_candidates != EFALLS_CANDIDATE_COUNT:
            raise ConfigError(f"{cfg.experiment.kind}: feature spec {spec.name} {spec.version} declares {n_candidates} eFalls baseline "
                              f"candidates, the published candidate set has {EFALLS_CANDIDATE_COUNT} (spec §5); the base feature spec "
                              "appears edited, so eFalls coverage and labels would be wrong")
    if features is None:
        if cfg.experiment.kind == REDUCED_KIND:
            raise ConfigError("efalls_retrained_reduced requires preprocessing.features")
        return
    known = set(spec.predictor_names())
    unknown = [f for f in features if f not in known]
    if unknown:
        raise ConfigError(f"preprocessing.features lists predictors not declared in feature spec {spec.name} {spec.version}: {unknown}")
    if cfg.experiment.kind != REDUCED_KIND:
        return
    candidates = [f.name for f in spec.features if f.exact_efalls_baseline]
    not_efalls = [f for f in features if f not in set(candidates)]
    if not_efalls:
        raise ConfigError(f"efalls_retrained_reduced: preprocessing.features {not_efalls} are not eFalls baseline candidates "
                          "(exact_efalls_baseline: true); Meuhedet predictors belong in alternative_model/ablation experiments")
    if set(features) == set(candidates):
        raise ConfigError(f"efalls_retrained_reduced: preprocessing.features lists all {len(candidates)} eFalls baseline candidates; "
                          "this is the full eFalls predictor set, so use experiment kind efalls_retrained (preprocessing.features unset)")
    missing = [f for f in REDUCED_REQUIRED_FEATURES if f not in features]
    if missing:
        raise ConfigError(f"efalls_retrained_reduced: preprocessing.features must include {missing}")


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Experiment config not found: {path}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    cfg = _build(ExperimentConfig, raw, path.name)
    cfg = dataclasses.replace(cfg, source_path=str(path))
    _validate_semantics(cfg)
    return cfg


def experiment_config_from_dict(raw: dict[str, Any], name: str = "<dict>") -> ExperimentConfig:
    cfg = _build(ExperimentConfig, raw, name)
    _validate_semantics(cfg)
    return cfg
