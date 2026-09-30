"""Unit tests for drift monitoring: reference profile, PSI, drift statuses, performance, writers; never retrains."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from falls_ml.config import ExperimentConfig, experiment_config_from_dict
from falls_ml.errors import ConfigError, DatasetValidationError
from falls_ml.evaluation.calibration import LogisticRecalibrator, Recalibrator
from falls_ml.features.preprocessing import Preprocessor
from falls_ml.features.spec import FeatureSpec, load_feature_spec
from falls_ml.models.logistic import UnpenalizedLogistic
from falls_ml.monitoring import (
    FEATURE_DRIFT_COLUMNS,
    RECOMMENDATION,
    SYNTHETIC_WATERMARK,
    build_reference_profile,
    drift_report,
    population_stability_index,
    write_drift_report,
)
from falls_ml.pipeline import FittedPipeline, fit_pipeline
from falls_ml.seeding import rng_for

REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "configs" / "features" / "efalls_v1.yaml"
FEATURES = ["age_years", "sex", "polypharmacy_count_120d", "bmi_value", "smoking_status", "alcohol_category",
            "falls", "dementia", "fracture"]
ALCOHOL = ["harmful", "higher_risk", "lower_risk", "previous_higher_risk_or_harmful", "zero"]


def make_frame(spec: FeatureSpec, n: int, component: str, *, age: tuple[float, float] = (65.0, 95.0), falls_rate: float = 0.2,
               bmi_missing: float = 0.2, intercept: float = -6.0) -> pd.DataFrame:
    rng = rng_for(20260915, component)
    index = pd.Timestamp("2019-01-01") + pd.to_timedelta(rng.integers(0, 720, n), unit="D")
    df = pd.DataFrame({
        "research_id": pd.Series([f"{component}-{i}" for i in range(n)], dtype="str"),
        "index_date": index,
        "predictor_max_record_date": index - pd.Timedelta(days=1),
        "age_years": rng.uniform(*age, n),
        "sex": pd.Series(rng.choice(["female", "male"], n), dtype="str"),
        "polypharmacy_count_120d": rng.integers(0, 25, n).astype("int64"),
        "bmi_value": np.where(rng.random(n) < bmi_missing, np.nan, rng.uniform(16.0, 42.0, n)),
        "smoking_status": pd.Series(rng.choice(np.array(["never", "ex", "current", None], dtype=object), n), dtype="str"),
        "alcohol_category": pd.Series(rng.choice(np.array([*ALCOHOL, None], dtype=object), n), dtype="str"),
        "source": pd.Series(["synthetic_fixture"] * n, dtype="str"),
    })
    for name in spec.binary_names():
        df[name] = rng.binomial(1, falls_rate if name == "falls" else 0.2, n).astype("int8")
    eta = intercept + 0.06 * df["age_years"] + 0.4 * np.log((df["polypharmacy_count_120d"] + 1) / 10) + 0.8 * df["falls"]
    df["outcome_12m"] = rng.binomial(1, expit(eta.to_numpy())).astype("int8")
    return df


def make_config() -> ExperimentConfig:
    return experiment_config_from_dict({
        "experiment": {"name": "monitoring_test", "kind": "alternative_model", "description": "unit test",
                       "layers": ["L1_published", "L4_alternative"]},
        "dataset": {"feature_spec": str(SPEC_PATH)},
        "model": {"name": "logistic_unpenalized"},
        "preprocessing": {"representation": "efalls_reference_coded"},
        "validation": {"strategy": "patient_grouped_random", "seed": 1, "limitation_note": "unit test"},
    })


@pytest.fixture(scope="module")
def spec() -> FeatureSpec:
    return load_feature_spec(SPEC_PATH).subset(FEATURES)


@pytest.fixture(scope="module")
def train(spec: FeatureSpec) -> pd.DataFrame:
    return make_frame(spec, 1500, "train")


@pytest.fixture(scope="module")
def pipeline(spec: FeatureSpec, train: pd.DataFrame) -> FittedPipeline:
    fitted = fit_pipeline(train, spec, make_config(), random_state=3)
    fitted.calibrator = LogisticRecalibrator(alpha=0.0, beta=1.0)
    return fitted


@pytest.fixture(scope="module")
def profile(train: pd.DataFrame, spec: FeatureSpec, pipeline: FittedPipeline) -> dict[str, Any]:
    return build_reference_profile(train, spec, pipeline)


# ---------------------------------------------------------------------- reference profile
def test_reference_profile_summarises_training_rows(profile, train, spec, pipeline):
    assert json.loads(json.dumps(profile)) == profile  # JSON-serialisable as written into the bundle
    assert profile["n_rows"] == len(train) and profile["feature_spec"]["sha256"] == spec.content_sha256
    features = profile["features"]
    assert set(features) == set(spec.predictor_names())
    assert features["falls"]["kind"] == "binary" and features["falls"]["prevalence"] == pytest.approx(train["falls"].mean())
    alcohol = features["alcohol_category"]["frequencies"]
    assert list(alcohol) == [*ALCOHOL, "missing"] and sum(alcohol.values()) == pytest.approx(1.0)
    assert alcohol["missing"] == pytest.approx(train["alcohol_category"].isna().mean())
    bmi = features["bmi_value"]
    assert bmi["missing_rate"] == pytest.approx(train["bmi_value"].isna().mean())
    assert sum(bmi["proportions"]) + bmi["missing_rate"] == pytest.approx(1.0)
    age_edges = features["age_years"]["quantile_edges"]
    assert len(age_edges) == 9 and train["age_years"].min() < age_edges[0] and age_edges[-1] < train["age_years"].max()
    assert features["age_years"]["proportions"] == pytest.approx([0.1] * 10, abs=1e-3)
    assert profile["polypharmacy_max"] == train["polypharmacy_count_120d"].max()
    design_means = pipeline.design(train).mean(axis=0)
    assert profile["design_column_means"] == pytest.approx(design_means.to_dict())
    assert sum(profile["predictions"]["proportions"]) == pytest.approx(1.0)
    assert profile["predictions"]["mean"] == pytest.approx(pipeline.predict_proba(train).mean())
    assert profile["outcome_prevalence"] == pytest.approx(train["outcome_12m"].mean())
    assert profile["synthetic_training_data"] is True


# ---------------------------------------------------------------------- PSI
def test_population_stability_index_formula():
    assert population_stability_index([0.25, 0.75], [0.25, 0.75]) == 0.0
    expected = (0.5 - 0.25) * math.log(0.5 / 0.25) + (0.5 - 0.75) * math.log(0.5 / 0.75)
    assert population_stability_index([0.25, 0.75], [0.5, 0.5]) == pytest.approx(expected, rel=1e-12)
    assert math.isfinite(population_stability_index([0.0, 1.0], [0.5, 0.5]))  # epsilon smoothing of empty bins
    with pytest.raises(ValueError):
        population_stability_index([0.5, 0.5], [1.0])


# ---------------------------------------------------------------------- drift statuses
def test_drift_is_ok_on_the_reference_data(profile, train, spec, pipeline):
    report = drift_report(profile, train, spec, pipeline=pipeline, labels=train["outcome_12m"].to_numpy())
    assert list(report.feature_drift.columns) == FEATURE_DRIFT_COLUMNS
    assert report.feature_drift["psi"].max() < 1e-6 and set(report.feature_drift["status"]) == {"ok"}
    assert report.prediction_drift["psi"] < 1e-6 and report.prediction_drift["status"] == "ok"
    assert report.performance["prevalence_status"] == "ok"
    assert report.overall_status == "ok"
    assert report.recommendation == RECOMMENDATION


def test_shifted_data_raises_alerts(profile, spec, pipeline):
    shifted = make_frame(spec, 800, "shifted", age=(80.0, 110.0), falls_rate=0.7, bmi_missing=0.2)
    report = drift_report(profile, shifted, spec, pipeline=pipeline)
    status = report.feature_drift.set_index("feature")["status"]
    assert status["age_years"] == "alert" and status["falls"] == "alert"
    assert status["sex"] == "ok"
    assert report.prediction_drift["status"] == "alert"
    assert report.prediction_drift["mean_current"] > report.prediction_drift["mean_reference"]
    assert report.performance is None
    assert report.overall_status == "alert"
    assert report.recommendation == RECOMMENDATION


def test_missing_rate_change_warns(profile, spec):
    current = make_frame(spec, 1500, "train", bmi_missing=0.2)
    current.loc[current.index[:150], "bmi_value"] = np.nan  # ~+8 percentage points missing, same observed distribution
    report = drift_report(profile, current, spec, missing_rate_abs_warn=0.05)
    bmi = report.feature_drift.set_index("feature").loc["bmi_value"]
    assert bmi["missing_rate_current"] - bmi["missing_rate_reference"] == pytest.approx(0.08, abs=0.02)
    assert bmi["psi"] < 0.1  # the warning comes from the missing-rate rule, not from PSI
    assert bmi["status"] == "warn" and report.overall_status == "warn"
    assert report.prediction_drift is None and report.performance is None


def test_performance_with_labels_reports_prevalence_discrimination_and_calibration(profile, spec, pipeline):
    current = make_frame(spec, 1500, "labelled", intercept=-4.5)
    y = current["outcome_12m"].to_numpy()
    report = drift_report(profile, current, spec, pipeline=pipeline, labels=y)
    perf = report.performance
    assert perf["variant"] == "recalibrated" and perf["n"] == len(current) and perf["n_events"] == int(y.sum())
    assert perf["prevalence_current"] == pytest.approx(y.mean()) and perf["prevalence_current"] > perf["prevalence_reference"]
    assert perf["prevalence_status"] in {"warn", "alert"}
    assert 0.5 < perf["auroc"] < 1.0 and perf["oe_ratio"] > 1.0 and perf["citl"] > 0.2
    assert perf["calibration_status"] == "warn" and report.overall_status in {"warn", "alert"}
    periods = [p["period"] for p in perf["by_period"]]
    assert periods == sorted(periods) and periods[0] == "2019Q1" and sum(p["n"] for p in perf["by_period"]) == len(current)


def test_small_cells_report_null_metrics(profile, spec, pipeline):
    current = make_frame(spec, 60, "small")
    report = drift_report(profile, current, spec, pipeline=pipeline, labels=np.zeros(60, dtype=int))
    assert report.performance["auroc"] is None and report.performance["calibration_status"] == "not_assessed"


def test_monitoring_never_fits(profile, spec, pipeline, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("monitoring must never fit or retrain")

    monkeypatch.setattr(Preprocessor, "fit", forbidden)
    monkeypatch.setattr(UnpenalizedLogistic, "fit", forbidden)
    monkeypatch.setattr(Recalibrator, "fit", forbidden)
    fingerprint = pipeline.preprocessor.fingerprint()
    current = make_frame(spec, 400, "never_fit", age=(75.0, 100.0))
    report = drift_report(profile, current, spec, pipeline=pipeline, labels=current["outcome_12m"].to_numpy())
    assert report.recommendation == RECOMMENDATION
    assert pipeline.preprocessor.fingerprint() == fingerprint


# ---------------------------------------------------------------------- errors and writers
def test_invalid_inputs_fail_loudly(profile, spec, pipeline, train):
    with pytest.raises(ConfigError, match="requires the bundle pipeline"):
        drift_report(profile, train, spec, labels=train["outcome_12m"].to_numpy())
    with pytest.raises(ConfigError, match="different feature spec"):
        drift_report(profile, train, load_feature_spec(SPEC_PATH).subset(FEATURES[:-1]))
    with pytest.raises(DatasetValidationError, match="labels"):
        drift_report(profile, train, spec, pipeline=pipeline, labels=np.full(len(train), 2))
    leaky = train.head(20).assign(predictor_max_record_date=train["index_date"].head(20))
    with pytest.raises(DatasetValidationError, match="LEAKAGE"):
        drift_report(profile, leaky, spec)


def test_write_drift_report(tmp_path, profile, spec, pipeline, train):
    report = drift_report(profile, train, spec, pipeline=pipeline, labels=train["outcome_12m"].to_numpy())
    paths = write_drift_report(report, tmp_path / "drift")
    assert [p.name for p in paths] == ["drift_report.json", "feature_drift.csv", "drift_report.md"]
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    assert payload["overall_status"] == "ok" and payload["recommendation"] == RECOMMENDATION
    assert len(payload["feature_drift"]) == len(FEATURES) and payload["performance"]["by_period"]
    assert list(pd.read_csv(paths[1]).columns) == FEATURE_DRIFT_COLUMNS
    markdown = paths[2].read_text(encoding="utf-8")
    assert markdown.startswith(SYNTHETIC_WATERMARK) and RECOMMENDATION in markdown and "| age_years |" in markdown


# ---------------------------------------------------------------------- reviewer additions
def test_profile_must_belong_to_the_pipeline_and_spec(profile, spec, pipeline, train):
    config = experiment_config_from_dict({**make_config().to_dict(), "preprocessing": {"representation": "efalls_raw"},
                                          "model": {"name": "random_forest", "params": {"n_estimators": 10}}})
    forest = fit_pipeline(train, spec, config, random_state=3)
    with pytest.raises(ConfigError, match="different pipeline"):
        drift_report(profile, train, spec, pipeline=forest)
    unrecorded = {k: v for k, v in profile.items() if k != "feature_spec"}
    with pytest.raises(ConfigError, match="different feature spec"):
        drift_report(unrecorded, train, spec)


# ---------------------------------------------------------------------- reference performance (AUROC drop, prevalence change)
def _reference_performance(**overrides: Any) -> dict[str, Any]:
    return {"auroc": 0.75, "calibration_slope": 1.0, "citl": 0.0, "oe_ratio": 1.0, "prevalence": 0.2, **overrides}


def _train_report(train, spec, pipeline, reference_performance, **kwargs):
    profile = build_reference_profile(train, spec, pipeline, reference_performance=reference_performance)
    return drift_report(profile, train, spec, pipeline=pipeline, labels=train["outcome_12m"].to_numpy(), **kwargs)


def test_reference_profile_stores_validated_reference_performance(train, spec, pipeline, profile):
    assert profile["reference_performance"] is None
    stored = build_reference_profile(train, spec, pipeline, reference_performance=_reference_performance(
        auroc=np.float64(0.8), citl=None, split="test", variant="recalibrated"))["reference_performance"]
    assert stored == {**_reference_performance(auroc=0.8, citl=None), "split": "test", "variant": "recalibrated"}
    assert json.loads(json.dumps(stored)) == stored
    for bad in ({k: v for k, v in _reference_performance().items() if k != "oe_ratio"}, _reference_performance(aurok=0.7),
                _reference_performance(auroc=True), _reference_performance(auroc=float("nan")), _reference_performance(auroc=1.5),
                _reference_performance(prevalence=0.0), _reference_performance(split=1), [0.7]):
        with pytest.raises(ConfigError, match="reference_performance"):
            build_reference_profile(train, spec, pipeline, reference_performance=bad)
    with pytest.raises(ConfigError, match="reference_performance"):  # a hand-edited profile is refused too
        drift_report({**profile, "reference_performance": {"auroc": 0.7}}, train, spec)


@pytest.mark.parametrize(("drop", "expected"), [(0.0, "ok"), (0.019, "ok"), (0.02, "warn"), (0.049, "warn"), (0.05, "alert"), (0.2, "alert")])
def test_auroc_drop_against_reference_performance_sets_status(train, spec, pipeline, profile, drop, expected):
    current_auroc = drift_report(profile, train, spec, pipeline=pipeline, labels=train["outcome_12m"].to_numpy()).performance["auroc"]
    prevalence = float(train["outcome_12m"].mean())
    report = _train_report(train, spec, pipeline, _reference_performance(auroc=current_auroc + drop, prevalence=prevalence))
    perf = report.performance
    assert perf["auroc_reference"] == pytest.approx(current_auroc + drop) and perf["auroc_drop"] == pytest.approx(drop, abs=1e-12)
    assert perf["discrimination_status"] == expected and perf["prevalence_status"] == "ok" and perf["calibration_status"] == "ok"
    assert report.overall_status == expected  # features, predictions, prevalence and calibration are all ok on the reference rows
    assert report.recommendation == RECOMMENDATION


def test_auroc_drop_thresholds_are_parameters_and_validated(train, spec, pipeline, profile):
    current_auroc = drift_report(profile, train, spec, pipeline=pipeline, labels=train["outcome_12m"].to_numpy()).performance["auroc"]
    reference = _reference_performance(auroc=current_auroc + 0.03, prevalence=float(train["outcome_12m"].mean()))
    report = _train_report(train, spec, pipeline, reference, auroc_drop_warn=0.01, auroc_drop_alert=0.03)
    assert report.performance["discrimination_status"] == "alert" and report.settings["auroc_drop_alert"] == 0.03
    for kwargs in ({"auroc_drop_warn": 0.05, "auroc_drop_alert": 0.02}, {"auroc_drop_warn": 0.0}, {"prevalence_abs_warn": 0.0},
                   {"prevalence_rel_warn": -0.1}):
        with pytest.raises(ConfigError, match="performance thresholds"):
            drift_report(profile, train, spec, **kwargs)


def test_discrimination_not_assessed_without_reference_or_events(train, spec, pipeline, profile):
    perf = drift_report(profile, train, spec, pipeline=pipeline, labels=train["outcome_12m"].to_numpy()).performance
    assert perf["auroc_reference"] is None and perf["discrimination_status"] == "not_assessed"
    assert perf["prevalence_reference_source"] == "training_rows"
    small = make_frame(spec, 60, "small")
    with_reference = build_reference_profile(train, spec, pipeline, reference_performance=_reference_performance())
    perf = drift_report(with_reference, small, spec, pipeline=pipeline, labels=np.zeros(60, dtype=int)).performance
    assert perf["auroc"] is None and perf["auroc_drop"] is None and perf["discrimination_status"] == "not_assessed"


@pytest.mark.parametrize(("shift", "expected"), [(0.0, "ok"), (0.019, "ok"), (0.02, "warn"), (-0.02, "warn"), (0.05, "warn")])
def test_absolute_prevalence_change_warns(train, spec, pipeline, profile, shift, expected):
    prevalence = float(train["outcome_12m"].mean())
    current_auroc = drift_report(profile, train, spec, pipeline=pipeline, labels=train["outcome_12m"].to_numpy()).performance["auroc"]
    report = _train_report(train, spec, pipeline, _reference_performance(auroc=current_auroc, prevalence=prevalence + shift))
    perf = report.performance
    assert perf["prevalence_reference_source"] == "reference_performance"
    assert perf["prevalence_abs_change"] == pytest.approx(abs(shift), abs=1e-12)
    assert perf["prevalence_psi"] < 0.1  # the warning comes from the change rule, not from PSI
    assert perf["prevalence_status"] == expected and report.overall_status == expected


def test_relative_prevalence_change_warns_for_rare_outcomes(profile, spec, pipeline, train):
    labels = np.zeros(len(train), dtype=int)
    labels[:78] = 1  # current prevalence 0.052
    for reference_prevalence, expected in ((0.04, "warn"), (0.045, "ok")):  # relative change 30% (abs 0.012) vs 15.6%
        reference = build_reference_profile(train, spec, pipeline, reference_performance=_reference_performance(
            auroc=None, prevalence=reference_prevalence))
        perf = drift_report(reference, train, spec, pipeline=pipeline, labels=labels).performance
        assert perf["prevalence_abs_change"] < 0.02
        assert perf["prevalence_rel_change"] == pytest.approx(0.012 / 0.04 if expected == "warn" else 0.007 / 0.045)
        assert perf["prevalence_status"] == expected


def test_report_writers_include_reference_comparison(tmp_path, train, spec, pipeline, profile):
    current_auroc = drift_report(profile, train, spec, pipeline=pipeline, labels=train["outcome_12m"].to_numpy()).performance["auroc"]
    report = _train_report(train, spec, pipeline, _reference_performance(auroc=current_auroc + 0.1,
                                                                        prevalence=float(train["outcome_12m"].mean())))
    paths = write_drift_report(report, tmp_path / "drift")
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    assert payload["overall_status"] == "alert" and payload["performance"]["discrimination_status"] == "alert"
    assert payload["performance"]["reference_performance"]["auroc"] == pytest.approx(current_auroc + 0.1)
    markdown = paths[2].read_text(encoding="utf-8")
    assert "discrimination status **alert**" in markdown and RECOMMENDATION in markdown
