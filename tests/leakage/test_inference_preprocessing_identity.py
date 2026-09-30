"""Leakage tests: production inference reuses the training preprocessing exactly and never fits anything.

Architecture §1 principle 4: one preprocessing implementation, persisted in the bundle and fingerprinted. The design
matrix built from a loaded bundle must equal the training design bit for bit, and neither ``load_bundle``,
``predict_risk`` nor drift monitoring may refit the preprocessor (FP forms, imputation medians), the model or the
recalibrator on production rows. D-00: rows whose predictor records reach the index date are refused before any
transformation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml
from scipy.special import expit

from falls_ml.bundle import LoadedBundle, load_bundle, save_bundle
from falls_ml.config import ExperimentConfig, experiment_config_from_dict
from falls_ml.data.dataset import DatasetManifest
from falls_ml.errors import DatasetValidationError
from falls_ml.evaluation.calibration import LogisticRecalibrator, Recalibrator
from falls_ml.features import preprocessing
from falls_ml.features import spec as spec_module
from falls_ml.features.preprocessing import Preprocessor
from falls_ml.features.spec import FeatureSpec, load_feature_spec
from falls_ml.inference import predict_risk
from falls_ml.models.lasso_cv import LassoLogisticCV
from falls_ml.monitoring import build_reference_profile, drift_report
from falls_ml.pipeline import FittedPipeline, fit_pipeline
from falls_ml.seeding import rng_for

REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "configs" / "features" / "efalls_v1.yaml"
FEATURES = ["age_years", "sex", "polypharmacy_count_120d", "bmi_value", "smoking_status", "alcohol_category",
            "falls", "dementia", "fracture", "egfr"]
ALCOHOL = ["harmful", "higher_risk", "lower_risk", "previous_higher_risk_or_harmful", "zero"]
CREATED = "2026-09-15T00:00:00+00:00"


def make_frame(spec: FeatureSpec, n: int, component: str, *, age: tuple[float, float] = (65.0, 95.0),
               egfr_mean: float = 60.0) -> pd.DataFrame:
    rng = rng_for(20260915, component)
    index = pd.Timestamp("2019-01-01") + pd.to_timedelta(rng.integers(0, 720, n), unit="D")
    df = pd.DataFrame({
        "research_id": pd.Series([f"{component}-{i}" for i in range(n)], dtype="str"),
        "index_date": index,
        "predictor_max_record_date": index - pd.Timedelta(days=1),
        "age_years": rng.uniform(*age, n),
        "sex": pd.Series(rng.choice(["female", "male"], n), dtype="str"),
        "polypharmacy_count_120d": rng.integers(0, 30, n).astype("int64"),
        "bmi_value": np.where(rng.random(n) < 0.2, np.nan, rng.uniform(16.0, 42.0, n)),
        "smoking_status": pd.Series(rng.choice(np.array(["never", "ex", "current", None], dtype=object), n), dtype="str"),
        "alcohol_category": pd.Series(rng.choice(np.array([*ALCOHOL, None], dtype=object), n), dtype="str"),
        "egfr": np.where(rng.random(n) < 0.3, np.nan, rng.normal(egfr_mean, 10.0, n)),
    })
    for name in spec.binary_names():
        df[name] = rng.binomial(1, 0.2, n).astype("int8")
    eta = -7.0 + 0.07 * df["age_years"] + 0.5 * np.log((df["polypharmacy_count_120d"] + 1) / 10) + 0.8 * df["falls"]
    df["outcome_12m"] = rng.binomial(1, expit(eta.to_numpy())).astype("int8")
    return df


def make_config() -> ExperimentConfig:
    return experiment_config_from_dict({
        "experiment": {"name": "identity_test", "kind": "alternative_model", "description": "leakage test",
                       "layers": ["L1_published", "L3b_meuhedet_predictor", "L4_alternative"]},
        "dataset": {"feature_spec": str(SPEC_PATH)},
        "model": {"name": "lasso_logistic_cv", "params": {"n_lambda": 30, "cv_folds": 5}},
        "preprocessing": {"representation": "efalls_fp_all_levels", "fractional_polynomial": {"mode": "select"}},
        "validation": {"strategy": "patient_grouped_random", "seed": 1, "limitation_note": "leakage test"},
    })


@pytest.fixture(scope="module")
def spec(tmp_path_factory: pytest.TempPathFactory) -> FeatureSpec:
    ext = {"feature_set": {"name": "identity_ext", "version": "0.0.1", "layer": "L3b_meuhedet_predictor"},
           "features": [{"name": "egfr", "concept": "eGFR", "dtype": "float_nullable", "missing_rule": "missing_category",
                         "layer": "L3b_meuhedet_predictor", "exact_efalls_baseline": False,
                         "clinically_validated": False, "group": "labs"}]}
    path = tmp_path_factory.mktemp("ext") / "identity_ext.yaml"
    path.write_text(yaml.safe_dump(ext), encoding="utf-8")
    return load_feature_spec(SPEC_PATH, extensions=[path]).subset(FEATURES)


@pytest.fixture(scope="module")
def train(spec: FeatureSpec) -> pd.DataFrame:
    return make_frame(spec, 1500, "train")


@pytest.fixture(scope="module")
def pipeline(spec: FeatureSpec, train: pd.DataFrame) -> FittedPipeline:
    fitted = fit_pipeline(train, spec, make_config(), random_state=11)
    fitted.calibrator = LogisticRecalibrator(alpha=0.1, beta=0.95)
    return fitted


@pytest.fixture(scope="module")
def bundle_dir(tmp_path_factory: pytest.TempPathFactory, pipeline: FittedPipeline, train: pd.DataFrame, spec: FeatureSpec) -> Path:
    manifest = DatasetManifest(dataset_version="test-v1", mapping_version="mapping-v0", source="synthetic_fixture",
                               feature_spec_name=spec.name, feature_spec_version=spec.version,
                               feature_spec_sha256=spec.content_sha256, data_file="modeling_dataset.parquet",
                               data_sha256="0" * 64, n_rows=len(train), n_patients=len(train), index_date_min="2019-01-01",
                               index_date_max="2020-12-31", outcome_prevalence=None, created_utc=CREATED,
                               generator="unit-test", scientific_use_allowed=False)
    return save_bundle(tmp_path_factory.mktemp("identity") / "model", pipeline, experiment_config=make_config(),
                       dataset_manifest=manifest, reference_profile=build_reference_profile(train, spec, pipeline),
                       model_version="model-v1", created_utc=CREATED)


@pytest.fixture
def forbid_fitting(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("inference/monitoring attempted to fit")

    monkeypatch.setattr(Preprocessor, "fit", forbidden)
    monkeypatch.setattr(Preprocessor, "_fit_fp_forms", forbidden)
    monkeypatch.setattr(preprocessing, "select_fp_forms", forbidden)
    monkeypatch.setattr(LassoLogisticCV, "fit", forbidden)
    monkeypatch.setattr(Recalibrator, "fit", forbidden)


def test_fitted_state_is_non_trivial(pipeline):
    """Guards the other tests: the preprocessor really holds data-dependent state (FP forms, medians)."""
    assert set(pipeline.preprocessor.fp_forms_) == {"age_years", "polypharmacy_count_120d"}
    assert set(pipeline.preprocessor.medians_) == {"egfr"}


def test_inference_design_equals_training_design_exactly(bundle_dir, pipeline, train, spec, forbid_fitting):
    loaded = load_bundle(bundle_dir)
    pre, original = loaded.pipeline.preprocessor, pipeline.preprocessor
    assert pre.fingerprint() == original.fingerprint()
    assert pre.state() == original.state()
    shifted = make_frame(spec, 500, "shifted", age=(80.0, 105.0), egfr_mean=30.0)
    for rows in (train, shifted):
        pd.testing.assert_frame_equal(pre.transform(rows), original.transform(rows), check_exact=True)
    results = predict_risk(shifted, loaded)
    assert np.array_equal([r.risk_12m for r in results], pipeline.predict_proba(shifted, calibrated=True))


def test_loaded_bundle_needs_no_feature_spec_files(bundle_dir, pipeline, spec, forbid_fitting, monkeypatch):
    """The training spec (base + extension + subset) travels inside the bundle; production never re-reads YAML."""
    def no_yaml(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("load_bundle read a feature spec YAML file")

    monkeypatch.setattr(spec_module, "_read_yaml", no_yaml)
    loaded = load_bundle(bundle_dir)
    assert loaded.feature_spec.content_sha256 == spec.content_sha256 and loaded.feature_spec.predictor_names() == spec.predictor_names()
    assert loaded.served_variant == "recalibrated"
    shifted = make_frame(spec, 300, "no_yaml", age=(80.0, 105.0), egfr_mean=30.0)
    pd.testing.assert_frame_equal(loaded.pipeline.design(shifted), pipeline.design(shifted), check_exact=True)
    assert [r.risk_12m for r in predict_risk(shifted, loaded)] == pipeline.predict_proba(shifted, calibrated=True).tolist()


def test_inference_and_monitoring_never_fit(bundle_dir, pipeline, train, spec, forbid_fitting):
    loaded = load_bundle(bundle_dir)
    fingerprint = loaded.pipeline.preprocessor.fingerprint()
    state = loaded.pipeline.preprocessor.state()
    shifted = make_frame(spec, 500, "never_fit", age=(80.0, 105.0), egfr_mean=30.0)
    predict_risk(shifted, loaded)
    report = drift_report(loaded.reference_profile, shifted, loaded.feature_spec, pipeline=loaded.pipeline,
                          labels=shifted["outcome_12m"].to_numpy())
    assert report.overall_status == "alert"  # drift is reported, nothing is adapted
    assert loaded.pipeline.preprocessor.fingerprint() == fingerprint
    assert loaded.pipeline.preprocessor.state() == state
    pd.testing.assert_frame_equal(loaded.pipeline.design(train), pipeline.design(train), check_exact=True)


def test_rows_with_predictor_records_on_the_index_date_are_refused_before_transform(bundle_dir, train, monkeypatch):
    loaded: LoadedBundle = load_bundle(bundle_dir)

    def no_transform(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("transform ran before D-00 validation")

    monkeypatch.setattr(Preprocessor, "transform", no_transform)
    leaky = train.head(5).assign(predictor_max_record_date=train["index_date"].head(5))
    with pytest.raises(DatasetValidationError, match="LEAKAGE"):
        predict_risk(leaky, loaded)


def test_inference_ignores_outcome_columns(bundle_dir, train):
    """Outcome values present in production rows cannot influence a prediction."""
    loaded = load_bundle(bundle_dir)
    rows = train.head(50)
    flipped = rows.assign(outcome_12m=(1 - rows["outcome_12m"]).astype("int8"))
    assert [r.risk_12m for r in predict_risk(rows, loaded)] == [r.risk_12m for r in predict_risk(flipped, loaded)]
    assert [r.risk_12m for r in predict_risk(rows.drop(columns=["outcome_12m"]), loaded)] == [r.risk_12m for r in predict_risk(rows, loaded)]
