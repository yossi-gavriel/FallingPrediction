"""Unit tests for production inference: ``predict_risk`` risks, contributions, categories and flags (M-12, D-00, D-06)."""

from __future__ import annotations

import dataclasses
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from falls_ml.bundle import LoadedBundle, load_bundle, save_bundle
from falls_ml.config import ExperimentConfig, experiment_config_from_dict, load_experiment_config
from falls_ml.data.dataset import DatasetManifest
from falls_ml.errors import ConfigError, DatasetValidationError
from falls_ml.evaluation.calibration import LogisticRecalibrator
from falls_ml.features.spec import FeatureSpec, load_feature_spec
from falls_ml.inference import PredictionResult, _risk_categories, linear_contributions, predict_risk, reference_design_row
from falls_ml.monitoring import build_reference_profile
from falls_ml.pipeline import FittedPipeline, fit_pipeline
from falls_ml.seeding import rng_for

REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "configs" / "features" / "efalls_v1.yaml"
FEATURES = ["age_years", "sex", "polypharmacy_count_120d", "bmi_value", "smoking_status", "alcohol_category",
            "falls", "dementia", "fracture"]
ALCOHOL = ["harmful", "higher_risk", "lower_risk", "previous_higher_risk_or_harmful", "zero"]
CREATED = "2026-09-15T00:00:00+00:00"
NOW = "2026-09-15T08:30:00+00:00"


def make_frame(spec: FeatureSpec, n: int, component: str, *, poly_max: int = 24) -> pd.DataFrame:
    rng = rng_for(20260915, component)
    index = pd.Timestamp("2019-01-01") + pd.to_timedelta(rng.integers(0, 720, n), unit="D")
    df = pd.DataFrame({
        "research_id": pd.Series([f"{component}-{i}" for i in range(n)], dtype="str"),
        "index_date": index,
        "predictor_max_record_date": index - pd.Timedelta(days=1),
        "age_years": rng.uniform(65.0, 95.0, n),
        "sex": pd.Series(rng.choice(["female", "male"], n), dtype="str"),
        "polypharmacy_count_120d": rng.integers(0, poly_max + 1, n).astype("int64"),
        "bmi_value": np.where(rng.random(n) < 0.2, np.nan, rng.uniform(16.0, 42.0, n)),
        "smoking_status": pd.Series(rng.choice(np.array(["never", "ex", "current", None], dtype=object), n), dtype="str"),
        "alcohol_category": pd.Series(rng.choice(np.array([*ALCOHOL, None], dtype=object), n), dtype="str"),
    })
    for name in spec.binary_names():
        df[name] = rng.binomial(1, 0.2, n).astype("int8")
    eta = -6.0 + 0.06 * df["age_years"] + 0.4 * np.log((df["polypharmacy_count_120d"] + 1) / 10) + 0.8 * df["falls"]
    df["outcome_12m"] = rng.binomial(1, expit(eta.to_numpy())).astype("int8")
    return df


def make_config(model: str = "logistic_unpenalized", representation: str = "efalls_reference_coded",
                params: dict[str, Any] | None = None, risk_categories: dict[str, Any] | None = None) -> ExperimentConfig:
    raw: dict[str, Any] = {
        "experiment": {"name": "inference_test", "kind": "alternative_model", "description": "unit test",
                       "layers": ["L1_published", "L4_alternative"]},
        "dataset": {"feature_spec": str(SPEC_PATH)},
        "model": {"name": model, "params": params or {}},
        "preprocessing": {"representation": representation},
        "validation": {"strategy": "patient_grouped_random", "seed": 1, "limitation_note": "unit test"},
    }
    if risk_categories is not None:
        raw["reporting"] = {"risk_categories": risk_categories}
    return experiment_config_from_dict(raw)


def build_bundle(directory: Path, pipeline: FittedPipeline, config: ExperimentConfig, train: pd.DataFrame, *,
                 extra: dict[str, Any] | None = None) -> LoadedBundle:
    spec = pipeline.spec
    manifest = DatasetManifest(dataset_version="test-v1", mapping_version="mapping-v0", source="synthetic_fixture",
                               feature_spec_name=spec.name, feature_spec_version=spec.version,
                               feature_spec_sha256=spec.content_sha256, data_file="modeling_dataset.parquet",
                               data_sha256="0" * 64, n_rows=len(train), n_patients=len(train), index_date_min="2019-01-01",
                               index_date_max="2020-12-31", outcome_prevalence=None, created_utc=CREATED,
                               generator="unit-test", scientific_use_allowed=False)
    save_bundle(directory, pipeline, experiment_config=config, dataset_manifest=manifest,
                reference_profile=build_reference_profile(train, spec, pipeline), model_version="model-v1",
                created_utc=CREATED, extra_metadata=extra if extra is not None else {"mappings_clinically_validated": False})
    return load_bundle(directory)


@pytest.fixture(scope="module")
def spec() -> FeatureSpec:
    return load_feature_spec(SPEC_PATH).subset(FEATURES)


@pytest.fixture(scope="module")
def train(spec: FeatureSpec) -> pd.DataFrame:
    return make_frame(spec, 1200, "train", poly_max=20)


@pytest.fixture(scope="module")
def rows(spec: FeatureSpec) -> pd.DataFrame:
    return make_frame(spec, 200, "rows", poly_max=20)


@pytest.fixture(scope="module")
def linear_pipeline(spec: FeatureSpec, train: pd.DataFrame) -> FittedPipeline:
    fitted = fit_pipeline(train, spec, make_config(), random_state=3)
    fitted.calibrator = LogisticRecalibrator(alpha=0.2, beta=0.9)
    return fitted


@pytest.fixture(scope="module")
def linear_bundle(tmp_path_factory: pytest.TempPathFactory, linear_pipeline: FittedPipeline, train: pd.DataFrame) -> LoadedBundle:
    return build_bundle(tmp_path_factory.mktemp("linear") / "model", linear_pipeline, make_config(), train)


# ---------------------------------------------------------------------- risks
def test_risks_match_the_training_pipeline(linear_bundle, linear_pipeline, rows):
    assert linear_bundle.served_variant == "recalibrated"
    calibrated = predict_risk(rows, linear_bundle, now=NOW)  # default: the bundle's served variant
    uncalibrated = predict_risk(rows, linear_bundle, use_calibration=False, now=NOW)
    assert len(calibrated) == len(rows) and all(isinstance(r, PredictionResult) for r in calibrated)
    assert np.array_equal([r.risk_12m for r in calibrated], linear_pipeline.predict_proba(rows, calibrated=True))
    assert np.array_equal([r.risk_12m for r in uncalibrated], linear_pipeline.predict_proba(rows))
    assert all(r.calibrated for r in calibrated) and not any(r.calibrated for r in uncalibrated)
    assert not any("served_variant_overridden" in r.data_quality_flags for r in calibrated)
    assert all("served_variant_overridden" in r.data_quality_flags for r in uncalibrated)
    explicit = predict_risk(rows, linear_bundle, use_calibration=True, now=NOW)  # explicit but equal to the served variant
    assert explicit == calibrated
    first = calibrated[0]
    assert first.research_id == "rows-0" and first.model_version == "model-v1" and first.prediction_timestamp == NOW


def test_research_id_withheld_when_not_allowed(linear_bundle, rows):
    assert all(r.research_id is None for r in predict_risk(rows.head(3), linear_bundle, include_research_id=False))
    no_ids = rows.head(3).drop(columns=["research_id"])
    assert all(r.research_id is None for r in predict_risk(no_ids, linear_bundle))


def test_datetime_now_is_serialised(linear_bundle, rows):
    now = datetime(2026, 9, 15, 9, 0, tzinfo=timezone.utc)
    assert predict_risk(rows.head(1), linear_bundle, now=now)[0].prediction_timestamp == "2026-09-15T09:00:00+00:00"


def test_mapping_input_matches_dataframe_input(linear_bundle, rows):
    row = rows.iloc[[0]].reset_index(drop=True)
    row.loc[0, "bmi_value"] = np.nan
    record = {k: (None if pd.isna(v) else v) for k, v in row.iloc[0].to_dict().items()}
    record["index_date"] = str(row.loc[0, "index_date"].date())
    record["predictor_max_record_date"] = str(row.loc[0, "predictor_max_record_date"].date())
    from_frame = predict_risk(row, linear_bundle, now=NOW)[0]
    from_mapping = predict_risk(record, linear_bundle, now=NOW)[0]
    from_sequence = predict_risk([record, record], linear_bundle, now=NOW)
    two_rows = predict_risk(pd.concat([row, row], ignore_index=True), linear_bundle, now=NOW)  # same batch size: BLAS-exact
    assert from_mapping == from_frame and from_sequence == two_rows
    assert "bmi_missing" in from_mapping.data_quality_flags


def test_post_index_predictor_records_are_rejected(linear_bundle, rows):
    leaky = rows.head(2).copy()
    leaky.loc[:, "predictor_max_record_date"] = leaky["index_date"]
    with pytest.raises(DatasetValidationError, match="LEAKAGE"):
        predict_risk(leaky, linear_bundle)


# ---------------------------------------------------------------------- contributions
def test_contributions_sum_to_lp_minus_reference_lp(linear_bundle, rows):
    model = linear_bundle.pipeline.model
    design = linear_bundle.pipeline.design(rows)
    contributions = linear_contributions(design, linear_bundle)
    reference = reference_design_row(linear_bundle).to_frame().T[design.columns]
    expected = model.linear_predictor(design) - model.linear_predictor(reference)[0]
    np.testing.assert_allclose(contributions.sum(axis=1).to_numpy(), expected, rtol=0, atol=1e-10)
    means = linear_bundle.reference_profile["design_column_means"]
    assert reference.loc[:, "age_years"].item() == means["age_years"] and reference.loc[:, "falls"].item() == 0.0
    assert reference.loc[:, "sex=male"].item() == 0.0 and reference.loc[:, "polypharmacy_log_p1_div10"].item() == means["polypharmacy_log_p1_div10"]


def test_top_contributing_features_are_ranked_by_absolute_contribution(linear_bundle, rows):
    results = predict_risk(rows, linear_bundle, top_k=3, now=NOW)
    contributions = linear_contributions(linear_bundle.pipeline.design(rows), linear_bundle)
    for i, result in enumerate(results):
        top = result.top_contributing_features
        assert 0 < len(top) <= 3
        values = [t["contribution"] for t in top]
        assert values == sorted(values, key=abs, reverse=True)
        expected = contributions.iloc[i].abs().sort_values(ascending=False, kind="stable").head(len(top))
        assert [t["feature"] for t in top] == list(expected.index)
        for t in top:
            assert t["contribution"] == contributions.iloc[i][t["feature"]]
            assert t["raw_feature"] == linear_bundle.pipeline.preprocessor.raw_feature_of(t["feature"])
            assert t["direction"] == ("increases_risk" if t["contribution"] > 0 else "decreases_risk")
    assert predict_risk(rows.head(1), linear_bundle, top_k=0)[0].top_contributing_features == ()


def test_published_equation_contributions_with_unavailable_predictors(tmp_path):
    full = load_feature_spec(SPEC_PATH)
    config = load_experiment_config(REPO / "configs" / "experiments" / "fixture" / "efalls_published_scoring.yaml")
    equation = dataclasses.replace(config.model.published_equation, unavailable_predictors=("abdominal_pain",),
                                   unavailable_fill="sail_prevalence")
    config = config.with_overrides(model=dataclasses.replace(config.model, published_equation=equation))
    train, rows = make_frame(full, 300, "pub_train"), make_frame(full, 50, "pub_rows")
    train.loc[:, "abdominal_pain"], rows.loc[:, "abdominal_pain"] = np.int8(0), np.int8(0)
    pipeline = fit_pipeline(train, full, config, random_state=1)
    bundle = build_bundle(tmp_path / "model", pipeline, config, train)  # the flag must not rely on caller extra_metadata
    design = bundle.pipeline.design(rows)
    reference = reference_design_row(bundle).to_frame().T[design.columns]
    expected = bundle.pipeline.model.linear_predictor(design) - bundle.pipeline.model.linear_predictor(reference)[0]
    np.testing.assert_allclose(linear_contributions(design, bundle).sum(axis=1).to_numpy(), expected, rtol=0, atol=1e-10)
    result = predict_risk(rows.head(1), bundle)[0]
    assert "unavailable_predictors_fixed_zero" in result.data_quality_flags and len(result.top_contributing_features) == 5


def test_non_linear_model_reports_no_contributions(tmp_path, spec, train, rows):
    config = make_config("random_forest", "efalls_raw", {"n_estimators": 20, "min_samples_leaf": 20})
    bundle = build_bundle(tmp_path / "model", fit_pipeline(train, spec, config, random_state=5), config, train)
    results = predict_risk(rows.head(5), bundle)
    assert all(r.top_contributing_features == () for r in results)
    assert all("contributions_not_available_for_model" in r.data_quality_flags for r in results)
    assert not any(r.calibrated for r in results)
    assert linear_contributions(bundle.pipeline.design(rows), bundle) is None


def test_bundle_without_recalibrator_serves_uncalibrated_risks(tmp_path, spec, train, rows):
    pipeline = fit_pipeline(train, spec, make_config(), random_state=3)
    bundle = build_bundle(tmp_path / "model", pipeline, make_config(), train)
    assert bundle.served_variant == "uncalibrated"
    results = predict_risk(rows.head(5), bundle, now=NOW)
    assert np.array_equal([r.risk_12m for r in results], pipeline.predict_proba(rows.head(5)))
    assert not any(r.calibrated or "served_variant_overridden" in r.data_quality_flags for r in results)
    assert predict_risk(rows.head(5), bundle, use_calibration=False, now=NOW) == results
    with pytest.raises(ConfigError, match="no recalibrator"):
        predict_risk(rows.head(5), bundle, use_calibration=True)
    with pytest.raises(ConfigError, match="use_calibration"):
        predict_risk(rows.head(5), bundle, use_calibration="yes")


# ---------------------------------------------------------------------- categories and flags
def test_risk_category_is_none_unless_approved(tmp_path, linear_bundle, linear_pipeline, train, rows):
    unapproved = predict_risk(rows, linear_bundle)
    assert all(r.risk_category is None and "risk_category_not_approved" in r.data_quality_flags for r in unapproved)

    categories = {"approved": True, "cutpoints": [0.2, 0.3], "labels": ["low", "medium", "high"],
                  "approval_reference": "CLIN-2026-001"}
    config = make_config(risk_categories=categories)
    approved_bundle = build_bundle(tmp_path / "model", linear_pipeline, config, train)
    approved = predict_risk(rows, approved_bundle)
    risks = np.array([r.risk_12m for r in approved])
    expected = np.select([risks < 0.2, risks < 0.3], ["low", "medium"], "high")
    assert [r.risk_category for r in approved] == expected.tolist()
    assert len(set(expected)) == 3
    assert not any("risk_category_not_approved" in r.data_quality_flags for r in approved)


def test_data_quality_flags(linear_bundle, rows):
    df = rows.head(4).copy().reset_index(drop=True)
    df.loc[:, ["bmi_value"]] = [np.nan, 25.0, 25.0, 25.0]
    df.loc[:, "alcohol_category"] = pd.Series([None, "zero", "zero", "zero"], dtype="str")
    df.loc[:, "smoking_status"] = pd.Series(["never", None, "never", "never"], dtype="str")
    df.loc[:, "age_years"] = [70.0, 70.0, 95.5, 95.0]
    poly_max = int(linear_bundle.reference_profile["polypharmacy_max"])
    df.loc[:, "polypharmacy_count_120d"] = np.array([0, 1, 2, poly_max + 1], dtype="int64")
    flags = [set(r.data_quality_flags) for r in predict_risk(df, linear_bundle)]
    constant = {"mappings_not_clinically_validated", "synthetic_training_data", "risk_category_not_approved"}
    assert flags[0] == constant | {"bmi_missing", "alcohol_missing"}
    assert flags[1] == constant | {"smoking_missing"}
    assert flags[2] == constant | {"age_above_95"}
    assert flags[3] == constant | {"polypharmacy_above_training_max"}


# ---------------------------------------------------------------------- reviewer additions
def test_approved_categories_are_lower_closed_like_threshold_metrics():
    """A risk equal to a cutpoint belongs to the upper category (``positive if p >= t`` in metrics, M-12)."""
    config = {"approved": True, "cutpoints": [0.2, 0.3], "labels": ["low", "medium", "high"]}
    risks = np.array([0.0, np.nextafter(0.2, 0.0), 0.2, 0.3, 1.0])
    assert _risk_categories(risks, config) == ["low", "low", "medium", "high", "high"]
    assert _risk_categories(risks, {"approved": False, "cutpoints": [0.2]}) == [None] * 5


def test_row_results_do_not_depend_on_the_rest_of_the_batch(linear_bundle, rows):
    batch = predict_risk(rows.head(8), linear_bundle, now=NOW)
    for i, result in enumerate(batch):
        alone = predict_risk(rows.iloc[[i]], linear_bundle, now=NOW)[0]
        assert alone.top_contributing_features == result.top_contributing_features
        assert alone.data_quality_flags == result.data_quality_flags and alone.risk_category == result.risk_category
        assert alone.risk_12m == pytest.approx(result.risk_12m, rel=1e-12, abs=1e-15)


def test_unverifiable_contributions_are_withheld_for_that_row_only(linear_bundle, rows, monkeypatch):
    model = linear_bundle.pipeline.model
    original = model.linear_predictor

    def shifted(X: pd.DataFrame) -> np.ndarray:  # an adapter whose LP disagrees with its coefficients on one row
        lp = np.array(original(X), dtype="float64")
        if len(X) > 1:
            lp[0] += 0.5
        return lp

    monkeypatch.setattr(model, "linear_predictor", shifted)
    results = predict_risk(rows.head(3), linear_bundle, now=NOW)
    assert results[0].top_contributing_features == () and "contributions_not_available_for_model" in results[0].data_quality_flags
    for result in results[1:]:
        assert result.top_contributing_features and "contributions_not_available_for_model" not in result.data_quality_flags
