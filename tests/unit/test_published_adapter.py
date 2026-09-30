"""Unit tests for the fixed published eFalls adapter (spec §5.10, D-01, D-07, D-20, M-11)."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from scipy.special import expit

from falls_ml.errors import BundleIntegrityError, ConfigError, PreprocessingMismatchError
from falls_ml.features.spec import load_feature_spec, load_feature_spec_from_payloads
from falls_ml.models import published_efalls as pe
from falls_ml.models.base import IMPORTANCE_COLUMNS
from falls_ml.models.published_efalls import (
    ALCOHOL_LEVELS,
    BMI_LEVELS,
    SEX_PARAMETERISATIONS,
    PublishedEfallsCoefficients,
    PublishedEfallsModel,
    PublishedFeatureMetadata,
    coverage_report,
    find_project_file,
)
from falls_ml.models.registry import get_adapter_class
from falls_ml.seeding import rng_for

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = "configs/models/efalls_published.yaml"
FEATURE_SPEC = "configs/features/efalls_v1.yaml"
LOW_SUPPORT = ["dressing_and_grooming_problems", "meal_preparation_problems", "medication_management",
               "motor_neurone_disease", "washing_and_bathing"]


@pytest.fixture(scope="module")
def coefficients() -> PublishedEfallsCoefficients:
    return PublishedEfallsCoefficients.from_yaml(CONFIG)


@pytest.fixture(scope="module")
def metadata() -> PublishedFeatureMetadata:
    return PublishedFeatureMetadata.from_feature_spec(load_feature_spec(REPO_ROOT / FEATURE_SPEC))


@pytest.fixture()
def make_model(coefficients, metadata):
    def _make(**params) -> PublishedEfallsModel:
        full = {"config": CONFIG, "sex_parameterisation": "lp_c_box_s3_1", **params}
        return PublishedEfallsModel(full, coefficients=coefficients, feature_metadata=metadata)
    return _make


def synthetic_design(columns: list[str], n: int = 60, seed: int = 11) -> pd.DataFrame:
    """Deterministic SYNTHETIC efalls_published design matrix (software test data only)."""
    rng = rng_for(seed, "tests.unit.published_adapter")
    data: dict[str, np.ndarray] = {
        "age_years": rng.uniform(65.0, 100.0, n),
        "polypharmacy_log_p1_div10": np.log((rng.integers(0, 40, n) + 1.0) / 10.0),
        "sex=male": rng.integers(0, 2, n).astype(np.int8),
    }
    bmi = rng.integers(0, len(BMI_LEVELS) + 1, n)  # last code = reference (overweight)
    data.update({f"bmi_category={level}": (bmi == i).astype(np.int8) for i, level in enumerate(BMI_LEVELS)})
    data["smoking=current"] = rng.integers(0, 2, n).astype(np.int8)
    alcohol = rng.integers(0, len(ALCOHOL_LEVELS) + 1, n)
    data.update({f"alcohol_category={level}": (alcohol == i).astype(np.int8) for i, level in enumerate(ALCOHOL_LEVELS)})
    for name in columns[len(data):]:
        data[name] = (rng.random(n) < 0.2).astype(np.int8)
    return pd.DataFrame(data)[columns]


# ---------------------------------------------------------------------- construction and D-01 variants
@pytest.mark.parametrize(("variant", "intercept", "sex_male"), [
    ("lp_c_box_s3_1", -6.258167, -0.303708),
    ("lp_a_table_s3_2", -6.258167, 0.303708),
    ("lp_b_label_swap", -5.954459, -0.303708),
    ("lp_d2_numeric_swap", -6.561875, 0.303708),
])
def test_variant_intercepts(coefficients, variant, intercept, sex_male):
    got_intercept, coefs = coefficients.design_coefficients(variant)
    assert got_intercept == pytest.approx(intercept, abs=1e-12)
    assert coefs["sex=male"] == pytest.approx(sex_male, abs=1e-12)
    assert len(coefs) == 85 and sum(v == 0.0 for v in coefs.values()) == 10


def test_real_yaml_initialisation_and_registry():
    cls = get_adapter_class("efalls_published")
    model = cls({"config": CONFIG, "sex_parameterisation": "lp_a_table_s3_2"})
    assert cls is PublishedEfallsModel and model.is_linear and not model.requires_fit
    assert model.design_columns[:3] == ["age_years", "polypharmacy_log_p1_div10", "sex=male"]
    assert model.design_columns[13:] == load_feature_spec(REPO_ROOT / FEATURE_SPEC).binary_names()
    assert len(model.design_columns) == 85
    assert model.coefficients.config_sha256 == hashlib.sha256((REPO_ROOT / CONFIG).read_bytes()).hexdigest()
    assert model.get_params() == {"config": CONFIG, "sex_parameterisation": "lp_a_table_s3_2", "unavailable_predictors": [],
                                  "unavailable_fill": "zero", "zero_low_support_predictors": False, "feature_spec": FEATURE_SPEC}


@pytest.mark.parametrize("params", [
    {"config": CONFIG},
    {"config": CONFIG, "sex_parameterisation": None},
    {"config": CONFIG, "sex_parameterisation": "lp_x"},
    {"sex_parameterisation": "lp_c_box_s3_1"},
    {"config": CONFIG, "sex_parameterisation": "lp_c_box_s3_1", "unavailable_fill": "mean"},
    {"config": CONFIG, "sex_parameterisation": "lp_c_box_s3_1", "zero_low_support_predictors": "yes"},
    {"config": CONFIG, "sex_parameterisation": "lp_c_box_s3_1", "unavailable_predictors": "falls"},
    {"config": CONFIG, "sex_parameterisation": "lp_c_box_s3_1", "sex_parameterization": "lp_c_box_s3_1"},
])
def test_invalid_params_raise(params):
    with pytest.raises(ConfigError):
        PublishedEfallsModel(params)


@pytest.mark.parametrize("unavailable", [["age_years"], ["sex"], ["not_a_feature"], ["falls", "falls"]])
def test_unavailable_must_be_unique_binary_predictors(make_model, unavailable):
    with pytest.raises(ConfigError):
        make_model(unavailable_predictors=unavailable)


def test_find_project_file_follows_the_shared_path_rule(tmp_path, monkeypatch):
    """falls_ml.paths.resolve_path: anchor's project root, then the package's repository root, then the CWD."""
    project = tmp_path / "project"
    local = project / CONFIG
    local.parent.mkdir(parents=True)
    local.write_text("x: 1\n", encoding="utf-8")
    only_in_cwd = "configs/models/only_in_this_project.yaml"
    (project / only_in_cwd).write_text("x: 2\n", encoding="utf-8")
    monkeypatch.chdir(project)
    assert find_project_file(CONFIG) == (REPO_ROOT / CONFIG).resolve()          # a CWD copy never shadows the repository
    assert find_project_file(CONFIG, anchor=local) == local.resolve()          # the declaring file's project wins
    assert find_project_file(only_in_cwd) == (project / only_in_cwd).resolve()  # last resort: the CWD
    assert find_project_file(REPO_ROOT / CONFIG) == REPO_ROOT / CONFIG
    with pytest.raises(ConfigError, match="not found"):
        find_project_file("configs/models/does_not_exist.yaml")
    with pytest.raises(ConfigError, match="not a file"):
        find_project_file("configs/models")


def test_adapter_resolves_the_coefficients_file_from_the_feature_spec_project(tmp_path, monkeypatch):
    project = tmp_path / "project"
    for relative in (CONFIG, FEATURE_SPEC):
        (project / relative).parent.mkdir(parents=True, exist_ok=True)
    (project / FEATURE_SPEC).write_bytes((REPO_ROOT / FEATURE_SPEC).read_bytes())
    (project / CONFIG).write_text((REPO_ROOT / CONFIG).read_text(encoding="utf-8") + "\n# project copy\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    model = PublishedEfallsModel({"config": CONFIG, "sex_parameterisation": "lp_c_box_s3_1",
                                  "feature_spec": str(project / FEATURE_SPEC)})
    assert model.coefficients.config_sha256 == hashlib.sha256((project / CONFIG).read_bytes()).hexdigest()
    assert model.coefficients.config_sha256 != hashlib.sha256((REPO_ROOT / CONFIG).read_bytes()).hexdigest()


def test_feature_metadata_uses_the_spec_cohort_without_reading_files(monkeypatch):
    raw = yaml.safe_load((REPO_ROOT / FEATURE_SPEC).read_text(encoding="utf-8"))
    spec = load_feature_spec_from_payloads(raw, source_paths=["configs/features/no_longer_on_disk.yaml"])

    def _no_files(*args, **kwargs):
        raise AssertionError("feature metadata must come from FeatureSpec.cohort, not from YAML files")
    monkeypatch.setattr(pe, "find_project_file", _no_files)
    monkeypatch.setattr(pe, "resolve_path", _no_files)
    metadata = PublishedFeatureMetadata.from_feature_spec(spec)
    assert list(metadata.mandatory_for_efalls_label) == raw["cohort"]["mandatory_for_efalls_label"]
    assert metadata.coverage_threshold == raw["cohort"]["coverage_threshold"]


def test_sex_parameterisation_is_read_only(make_model):
    model = make_model(sex_parameterisation="lp_a_table_s3_2")
    assert model.sex_parameterisation == "lp_a_table_s3_2"
    with pytest.raises(AttributeError):
        model.sex_parameterisation = "lp_c_box_s3_1"


def test_malformed_coefficients_file_raises(tmp_path):
    raw = yaml.safe_load((REPO_ROOT / CONFIG).read_text(encoding="utf-8"))
    del raw["sex_parameterisations"]["lp_b_label_swap"]
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError):
        PublishedEfallsCoefficients.from_yaml(path)
    path.write_text("terms: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        PublishedEfallsCoefficients.from_yaml(path)


def test_m11_mandatory_rule_cannot_be_silently_disabled(tmp_path, metadata):
    raw = yaml.safe_load((REPO_ROOT / FEATURE_SPEC).read_text(encoding="utf-8"))
    del raw["cohort"]["mandatory_for_efalls_label"]
    path = tmp_path / "efalls_no_mandatory.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    with pytest.raises(ConfigError, match="mandatory_for_efalls_label"):
        PublishedFeatureMetadata.from_feature_spec(load_feature_spec(path))
    with pytest.raises(ConfigError, match="mandatory_for_efalls_label"):
        PublishedFeatureMetadata.from_dict({**metadata.to_dict(), "mandatory_for_efalls_label": []})


# ---------------------------------------------------------------------- design validation
def test_fit_is_noop_and_ignores_labels(make_model):
    model = make_model()
    X = synthetic_design(model.design_columns)
    before = model.predict_proba(X)
    y = rng_for(3, "labels").integers(0, 2, len(X))
    assert model.fit(X, y) is model and model.feature_names_ == model.design_columns
    np.testing.assert_array_equal(model.predict_proba(X), before)
    assert before.shape == (len(X),)
    np.testing.assert_allclose(before, expit(model.linear_predictor(X)), rtol=0, atol=0)


def test_wrong_column_order_raises(make_model):
    model = make_model()
    X = synthetic_design(model.design_columns)
    cols = list(X.columns)
    cols[1], cols[0] = cols[0], cols[1]
    with pytest.raises(PreprocessingMismatchError, match="order_changed=True"):
        model.fit(X[cols])
    with pytest.raises(PreprocessingMismatchError):
        model.predict_proba(X.drop(columns=["falls"]))


def test_non_finite_and_non_indicator_values_raise(make_model):
    model = make_model()
    X = synthetic_design(model.design_columns)
    with_nan = X.assign(age_years=X["age_years"].where(X.index != 3))
    with pytest.raises(PreprocessingMismatchError, match="non-finite"):
        model.linear_predictor(with_nan)
    with pytest.raises(PreprocessingMismatchError, match="outside"):
        model.linear_predictor(X.assign(falls=X["falls"] * 2))


def test_non_numeric_design_columns_raise(make_model):
    """String/object columns must not be silently coerced to floats ("1" -> 1.0)."""
    model = make_model()
    X = synthetic_design(model.design_columns)
    with pytest.raises(PreprocessingMismatchError, match="non-numeric"):
        model.linear_predictor(X.astype({"falls": "str"}))
    with pytest.raises(PreprocessingMismatchError, match="non-numeric"):
        model.predict_proba(X.astype({"age_years": object}))
    with pytest.raises(PreprocessingMismatchError, match="non-numeric"):
        pe.published_linear_predictor(X.astype({"falls": "str"}), -6.0, dict.fromkeys(X.columns, 0.1))
    np.testing.assert_array_equal(model.linear_predictor(X.astype({"falls": bool})), model.linear_predictor(X))


def test_published_linear_predictor_rejects_duplicated_columns():
    X = pd.DataFrame([[1.0, 1.0, 2.0]], columns=["a", "a", "b"])  # membership checks alone would double-count "a"
    with pytest.raises(PreprocessingMismatchError, match="duplicated"):
        pe.published_linear_predictor(X, 0.0, {"a": 1.0, "b": 1.0})
    np.testing.assert_array_equal(pe.published_linear_predictor(X.iloc[:, 1:], -1.0, {"b": 0.5, "a": 2.0}), [2.0])


@pytest.mark.parametrize("levels", [("bmi_category=underweight", "bmi_category=normal"),
                                    ("alcohol_category=harmful", "alcohol_category=missing")])
def test_more_than_one_active_category_level_raises(make_model, levels):
    model = make_model()
    X = synthetic_design(model.design_columns)
    bad = X.copy()
    bad.loc[5, list(levels)] = 1
    with pytest.raises(PreprocessingMismatchError, match="more than one active"):
        model.fit(bad)
    with pytest.raises(PreprocessingMismatchError, match="more than one active"):
        model.score_all_variants(bad)


def test_polypharmacy_log_term_below_published_minimum_raises(make_model):
    model = make_model()
    X = synthetic_design(model.design_columns)
    floor = X.assign(polypharmacy_log_p1_div10=float(np.log((0 + 1) / 10)))  # P = 0 is the smallest valid value
    model.linear_predictor(floor)
    with pytest.raises(PreprocessingMismatchError, match="ln\\(0.1\\)"):
        model.linear_predictor(X.assign(polypharmacy_log_p1_div10=np.log((-0.5 + 1) / 10)))  # impossible P = -0.5


def test_design_from_preprocessor_matches_adapter(coefficients, metadata):
    """Cross-unit contract: the ``efalls_published`` Preprocessor output is exactly the adapter design (RT-02/03/04)."""
    from falls_ml.features.preprocessing import Preprocessor

    spec = load_feature_spec(REPO_ROOT / FEATURE_SPEC)
    row = {**dict.fromkeys(spec.binary_names(), 0), "dementia": 1, "liver_problems": 1, "osteoporosis": 1}
    df = pd.DataFrame([row, row])
    df = df.astype(dict.fromkeys(spec.binary_names(), "int8")).assign(
        age_years=np.array([89.0, 89.0]), sex=pd.Series(["female", "male"], dtype="object"),
        polypharmacy_count_120d=np.array([8, 8], dtype=np.int64), bmi_value=np.array([17.0, 17.0]),
        smoking_status=pd.Series([None, None], dtype="object"),
        alcohol_category=pd.Series(["previous_higher_risk_or_harmful"] * 2, dtype="object"))
    X = Preprocessor(spec, "efalls_published").fit(df).transform(df)
    for variant, expected in (("lp_c_box_s3_1", [-1.408453, -1.712161]), ("lp_a_table_s3_2", [-1.408453, -1.104745])):
        model = PublishedEfallsModel({"config": CONFIG, "sex_parameterisation": variant},
                                     coefficients=coefficients, feature_metadata=metadata)
        assert list(X.columns) == model.design_columns
        np.testing.assert_allclose(model.fit(X).linear_predictor(X), expected, rtol=0, atol=1e-6)


def test_unavailable_predictor_with_nonzero_data_raises(make_model):
    model = make_model(unavailable_predictors=["falls"])
    X = synthetic_design(model.design_columns)
    assert X["falls"].any()
    with pytest.raises(ConfigError, match="falls"):
        model.fit(X)
    with pytest.raises(ConfigError, match="falls"):
        model.predict_proba(X)
    model.fit(X.assign(falls=0))


# ---------------------------------------------------------------------- M-11 fill, D-20 low support
def test_sail_prevalence_fill_shifts_lp(make_model, coefficients, metadata):
    unavailable = ["falls", "asthma", "housebound"]
    zero = make_model(unavailable_predictors=unavailable)
    filled = make_model(unavailable_predictors=unavailable, unavailable_fill="sail_prevalence")
    X = synthetic_design(zero.design_columns).assign(**{u: 0 for u in unavailable})
    expected_shift = sum(coefficients.binary[u] * metadata.sail_proportion[u] for u in unavailable)
    assert expected_shift > 0
    np.testing.assert_allclose(filled.linear_predictor(X) - zero.linear_predictor(X), expected_shift, rtol=0, atol=1e-12)
    for variant in SEX_PARAMETERISATIONS:
        diff = filled.score_all_variants(X)[f"lp_{variant}"] - zero.score_all_variants(X)[f"lp_{variant}"]
        np.testing.assert_allclose(diff, expected_shift, rtol=0, atol=1e-12)


def test_low_support_zeroing(make_model, coefficients):
    default = make_model()
    zeroed = make_model(zero_low_support_predictors=True)
    assert zeroed.low_support_zeroed == LOW_SUPPORT and default.low_support_zeroed == []
    X = synthetic_design(default.design_columns).assign(**{name: 1 for name in LOW_SUPPORT})
    expected = sum(coefficients.binary[name] for name in LOW_SUPPORT)
    np.testing.assert_allclose(default.linear_predictor(X) - zeroed.linear_predictor(X), expected, rtol=0, atol=1e-12)
    for variant in SEX_PARAMETERISATIONS:  # D-20 zeroing applies to every scored sex parameterisation
        diff = default.score_all_variants(X)[f"lp_{variant}"] - zeroed.score_all_variants(X)[f"lp_{variant}"]
        np.testing.assert_allclose(diff, expected, rtol=0, atol=1e-12)
    table = zeroed.get_feature_importance().set_index("feature")
    assert (table.loc[LOW_SUPPORT, "coefficient"] == 0.0).all()
    assert not table.loc[LOW_SUPPORT, "selected"].any()
    _, published = coefficients.design_coefficients("lp_c_box_s3_1")
    assert published["medication_management"] == pytest.approx(0.8030273)  # source object is not mutated


def test_low_support_zeroing_removes_fill_contribution(make_model, coefficients, metadata):
    model = make_model(unavailable_predictors=["falls", "medication_management"], unavailable_fill="sail_prevalence",
                       zero_low_support_predictors=True)
    assert model.unavailable_intercept_shift == pytest.approx(coefficients.binary["falls"] * metadata.sail_proportion["falls"], abs=1e-15)


# ---------------------------------------------------------------------- coverage (M-11)
def _independent_total_variance(raw_model: dict, spec) -> float:
    dist = raw_model["development_distribution_sail"]
    terms = raw_model["terms"]
    total = terms["age_years"]["coefficient"] ** 2 * dist["age_years_sd"] ** 2
    total += terms["polypharmacy_count_120d"]["coefficient"] ** 2 * dist["polypharmacy_log_term_sd_approx"] ** 2
    p_male = 1 - dist["proportions"]["sex=female"]
    total += 0.303708 ** 2 * p_male * (1 - p_male)
    for group, prefix in (("bmi_category", "bmi_category"), ("smoking", "smoking"), ("alcohol_category", "alcohol_category")):
        for level, beta in terms[group]["levels"].items():
            p = dist["proportions"][f"{prefix}={level}"]
            total += beta ** 2 * p * (1 - p)
    for f in spec.features:
        if f.is_binary:
            p = f.raw["published_prevalence"]["sail_proportion"]
            total += terms["binary"][f.name] ** 2 * p * (1 - p)
    return total


def test_coverage_complete(make_model):
    report = make_model().coverage()
    assert report["lp_variance_share_available"] == 1.0
    assert report["effective_experiment_label"] == "efalls_published_scoring"
    assert report["efalls_feature_coverage"] == "complete"
    assert report["mandatory_unavailable"] == [] and report["unavailable_predictors"] == []
    assert report["threshold"] == 0.9


def test_coverage_mandatory_falls_is_partial(make_model, coefficients, metadata):
    report = make_model(unavailable_predictors=["falls"]).coverage()
    assert report["mandatory_unavailable"] == ["falls"]
    assert report["effective_experiment_label"] == "efalls_partial_scoring"
    assert report["lp_variance_share_available"] >= 0.9  # partial because mandatory, not because of the share
    p = metadata.sail_proportion["falls"]
    raw = yaml.safe_load((REPO_ROOT / CONFIG).read_text(encoding="utf-8"))
    total = _independent_total_variance(raw, load_feature_spec(REPO_ROOT / FEATURE_SPEC))
    assert report["lp_variance_share_available"] == pytest.approx(1 - 0.3009161 ** 2 * p * (1 - p) / total, abs=1e-12)
    assert report["unavailable_predictors"] == [{"feature": "falls", "coefficient": 0.3009161, "sail_prevalence": p,
                                                 "expected_mean_lp_shift": -0.3009161 * p}]


def test_coverage_rare_low_weight_predictor_keeps_label(make_model):
    report = make_model(unavailable_predictors=["peptic_ulcer_disease"]).coverage()
    assert 0.9 <= report["lp_variance_share_available"] < 1.0
    assert report["effective_experiment_label"] == "efalls_published_scoring"
    assert report["efalls_feature_coverage"] == "incomplete"


def test_coverage_threshold_rule_and_non_binary_features(coefficients, metadata):
    spec = load_feature_spec(REPO_ROOT / FEATURE_SPEC)
    from_spec = coverage_report(["bmi_value"], coefficients, spec)
    assert from_spec == coverage_report(["bmi_value"], coefficients, metadata)
    assert [r["feature"] for r in from_spec["unavailable_predictors"]] == [f"bmi_category={lvl}" for lvl in BMI_LEVELS]
    age = coverage_report(["age_years"], coefficients, metadata)
    assert age["lp_variance_share_available"] < 0.9 and age["mandatory_unavailable"] == ["age_years"]
    assert age["unavailable_predictors"][0]["expected_mean_lp_shift"] is None
    # below-threshold share alone forces the partial label (no mandatory predictor involved)
    strict = PublishedFeatureMetadata.from_dict({**metadata.to_dict(), "coverage_threshold": 0.99})
    low = coverage_report(["bmi_value"], coefficients, strict)
    assert low["mandatory_unavailable"] == [] and low["effective_experiment_label"] == "efalls_partial_scoring"
    with pytest.raises(ConfigError):
        coverage_report(["unknown"], coefficients, metadata)


# ---------------------------------------------------------------------- outputs and persistence
def test_score_all_variants_columns_and_d01_relations(make_model):
    model = make_model(sex_parameterisation="lp_a_table_s3_2")
    X = synthetic_design(model.design_columns)
    scores = model.score_all_variants(X)
    assert list(scores.columns) == [f"{kind}_{v}" for v in SEX_PARAMETERISATIONS for kind in ("lp", "risk")]
    assert scores.index.equals(X.index)
    np.testing.assert_array_equal(scores["lp_lp_a_table_s3_2"].to_numpy(), model.linear_predictor(X))
    np.testing.assert_allclose(scores["risk_lp_c_box_s3_1"], expit(scores["lp_lp_c_box_s3_1"]), rtol=0, atol=1e-15)
    np.testing.assert_allclose(scores["lp_lp_b_label_swap"] - scores["lp_lp_c_box_s3_1"], 0.303708, atol=1e-9)
    np.testing.assert_allclose(scores["lp_lp_d2_numeric_swap"] - scores["lp_lp_a_table_s3_2"], -0.303708, atol=1e-9)


def test_importance_columns(make_model):
    table = make_model(sex_parameterisation="lp_a_table_s3_2").get_feature_importance()
    assert list(table.columns) == IMPORTANCE_COLUMNS
    assert list(table["feature"]) == make_model().design_columns
    sex = table.set_index("feature").loc["sex=male"]
    assert sex["coefficient"] == pytest.approx(0.303708, abs=1e-12) and sex["direction"] == "increases_risk"
    assert table["native_importance"].isna().all()


def test_recalibration_reference(make_model):
    ref = make_model().recalibration_reference()
    assert (ref["alpha"], ref["beta"], ref["applies_to"]) == (-0.423, 1.25, "lp_c_box_s3_1")
    assert "D-07" in ref["note"]


def test_save_load_identity_without_yaml(make_model, tmp_path, monkeypatch):
    model = make_model(sex_parameterisation="lp_d2_numeric_swap", unavailable_predictors=["falls"],
                       unavailable_fill="sail_prevalence", zero_low_support_predictors=True)
    X = synthetic_design(model.design_columns).assign(falls=0)
    model.fit(X)
    model.save(tmp_path / "adapter")
    saved = json.loads((tmp_path / "adapter" / "adapter.json").read_text(encoding="utf-8"))
    assert saved["config_sha256"] == model.coefficients.config_sha256 and len(saved["coefficients"]["binary"]) == 72

    def _no_files(*args, **kwargs):
        raise AssertionError("load must not read YAML files")
    monkeypatch.setattr(pe, "find_project_file", _no_files)
    monkeypatch.setattr(pe, "load_feature_spec", _no_files)
    loaded = PublishedEfallsModel.load(tmp_path / "adapter")
    np.testing.assert_array_equal(loaded.predict_proba(X), model.predict_proba(X))
    pd.testing.assert_frame_equal(loaded.score_all_variants(X), model.score_all_variants(X))
    assert loaded.get_params() == model.get_params() and loaded.feature_names_ == model.feature_names_
    assert loaded.coverage() == model.coverage()
    assert loaded.coefficients == model.coefficients and loaded.feature_metadata == model.feature_metadata


def test_load_rejects_tampered_adapter(make_model, tmp_path):
    model = make_model()
    model.save(tmp_path)
    original = json.loads((tmp_path / "adapter.json").read_text(encoding="utf-8"))

    def _tampered(edit) -> None:
        meta = json.loads(json.dumps(original))
        edit(meta)
        (tmp_path / "adapter.json").write_text(json.dumps(meta), encoding="utf-8")
        with pytest.raises(BundleIntegrityError):
            PublishedEfallsModel.load(tmp_path)

    _tampered(lambda m: m["coefficients"]["bmi"].pop("obese"))                        # structural
    _tampered(lambda m: m["coefficients"]["binary"].update(falls=3.0))                # value edit, still well-formed
    _tampered(lambda m: m["feature_metadata"]["sail_proportion"].update(falls=0.5))   # coverage input edit
    _tampered(lambda m: m.update(feature_names=m["design_columns"][::-1]))
    _tampered(lambda m: m.pop("equation_sha256"))
    with pytest.raises(BundleIntegrityError):
        PublishedEfallsModel.load(tmp_path / "missing")


def test_intercept_shift_zero_by_default(make_model):
    assert make_model(unavailable_predictors=["falls"]).unavailable_intercept_shift == 0.0
    assert math.isclose(make_model().fit_diagnostics()["published_equation"]["intercept"], -6.258167, abs_tol=1e-12)
