"""End-to-end integration test on a tiny deterministic SYNTHETIC fixture (software test only, spec §13.2).

Proves the definition-of-done mechanics: one entry point per algorithm, all run artifacts, the published
equation and the retrained LASSO kept apart, comparison and best-features reports, bundle → predict_risk
identical to in-run test predictions, exact reproduction from run artifacts, and drift monitoring.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from falls_ml.bundle import load_bundle
from falls_ml.cli import main as cli_main
from falls_ml.config import experiment_config_from_dict
from falls_ml.data.dataset import ModelingDataset
from falls_ml.data.synthetic import generate_synthetic_modeling_dataset
from falls_ml.experiment import reproduce, run_ablation, run_experiment
from falls_ml.features.spec import load_feature_spec
from falls_ml.inference import predict_risk
from falls_ml.monitoring import drift_report
from falls_ml.reporting.best_features import build_best_features
from falls_ml.reporting.compare import build_comparison

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_CONFIGS = ROOT / "configs" / "experiments" / "fixture"
REQUIRED_ARTIFACTS = [
    "config.yaml", "dataset_manifest.json", "environment.json", "metrics.json", "splits.csv", "cv_results.csv",
    "hyperparameter_results.csv", "coefficients.csv", "feature_importance.csv", "feature_stability.csv", "calibration.csv",
    "confusion_matrices.csv", "decision_curve.csv", "subgroup_metrics.csv", "predictions_validation.parquet",
    "predictions_test.parquet", "report.md", "report.html", "model/bundle.json", "run_inputs.json", "run_log.jsonl",
    "RUN_COMPLETE.json",
]

pytestmark = pytest.mark.slow


def _config(name: str, dataset_dir: Path, **overrides) -> object:
    raw = yaml.safe_load((FIXTURE_CONFIGS / f"{name}.yaml").read_text(encoding="utf-8"))
    raw["dataset"]["path"] = str(dataset_dir)
    raw["dataset"]["feature_spec"] = str(ROOT / "configs" / "features" / "efalls_v1.yaml")
    if raw["model"].get("published_equation"):
        raw["model"]["published_equation"]["config"] = str(ROOT / "configs" / "models" / "efalls_published.yaml")
    raw["evaluation"]["bootstrap"]["n"] = 30
    raw["evaluation"]["permutation_importance_repeats"] = 1
    raw["analysis"]["stability"]["n_bootstrap"] = 3
    for section, values in overrides.items():
        raw[section].update(values)
    return experiment_config_from_dict(raw, name=name)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    base = tmp_path_factory.mktemp("e2e")
    data = base / "fixture"
    generate_synthetic_modeling_dataset(data, n_patients=900, index_dates=("2018-04-01", "2019-04-01", "2021-04-01"),
                                        feature_spec_path=ROOT / "configs" / "features" / "efalls_v1.yaml")
    spec = load_feature_spec(ROOT / "configs" / "features" / "efalls_v1.yaml")
    dataset = ModelingDataset.load(data, spec)
    runs = base / "runs"
    results = {
        "published": run_experiment(None, dataset, _config("efalls_published_scoring", data), runs_dir=runs,
                                    created_utc="2026-09-15T00:00:00+00:00"),
        "lasso": run_experiment(None, dataset, _config("efalls_retrained_lasso_fixed_fp", data,
                                                       model={"params": {"n_lambda": 20, "selection": "min"}}),
                                runs_dir=runs, created_utc="2026-09-15T00:00:01+00:00"),
        "rf": run_experiment(None, dataset, _config("random_forest", data, tuning={"enabled": False},
                                                    model={"params": {"n_estimators": 50}}),
                             runs_dir=runs, created_utc="2026-09-15T00:00:02+00:00"),
    }
    return {"base": base, "data": data, "spec": spec, "dataset": dataset, "runs": runs, "results": results}


@pytest.fixture(scope="module")
def reduced_world(world):
    """The reduced fixture config (efalls_retrained_reduced, 66 of 78 predictors) on a subset fixture with the world's rows."""
    base = world["base"]
    data = base / "fixture_reduced"
    features = list(yaml.safe_load((FIXTURE_CONFIGS / "efalls_retrained_reduced.yaml").read_text(encoding="utf-8"))["preprocessing"]["features"])
    generate_synthetic_modeling_dataset(data, n_patients=900, index_dates=("2018-04-01", "2019-04-01", "2021-04-01"),
                                        feature_spec_path=ROOT / "configs" / "features" / "efalls_v1.yaml", features=features)
    config = _config("efalls_retrained_reduced", data, model={"params": {"n_lambda": 20, "selection": "min"}},
                     preprocessing={"fractional_polynomial": {"mode": "fixed_published"}})
    runs = base / "runs_reduced"
    result = run_experiment(None, None, config, runs_dir=runs, created_utc="2026-09-15T00:00:03+00:00")
    return {"data": data, "features": features, "runs": runs, "result": result}


def test_reduced_efalls_run_reports_coverage_and_is_never_full_efalls(reduced_world, world, tmp_path):
    result = reduced_world["result"]
    missing = [a for a in REQUIRED_ARTIFACTS if not (result.run_dir / a).exists()]
    assert not missing, missing
    m = result.metrics
    cov = m["efalls_coverage"]
    assert m["experiment"]["kind"] == "efalls_retrained_reduced" and m["feature_set"]["n_features"] == 66
    assert (cov["n_available"], cov["n_total"], cov["coverage_pct"], cov["is_full_efalls_feature_set"]) == (66, 78, 84.6, False)
    assert cov["mandatory_unavailable"] == [] and m["production_readiness"]["unavailable_predictors"] == 12
    assert world["results"]["lasso"].metrics["efalls_coverage"]["n_available"] == 78
    assert world["results"]["published"].metrics["efalls_coverage"]["coverage_pct"] == 100.0
    assert world["results"]["rf"].metrics["efalls_coverage"] is None
    assert m["split"]["test_rows_sha256"] == world["results"]["lasso"].metrics["split"]["test_rows_sha256"], "same rows as the full fixture"
    md = (result.run_dir / "report.md").read_text(encoding="utf-8")
    for text in ("| Available eFalls predictors | 66 / 78 |", "Available eFalls predictors: 66 / 78", "Coverage percentage: 84.6%",
                 "REDUCED eFalls predictor set – NOT a full eFalls reproduction",
                 "Local retraining on a REDUCED eFalls predictor set (NOT a full eFalls reproduction)", "bmi_value, smoking_status"):
        assert text in md, text
    subgroups = pd.read_csv(result.run_dir / "subgroup_metrics.csv")
    assert set(subgroups["subgroup_variable"]) == {"sex", "age_band"}, "bmi_value is unavailable, so no BMI subgroup"
    # bundle -> predict_risk equals the in-run served predictions; the bundle's spec is the subset spec
    bundle = load_bundle(result.run_dir / "model")
    assert bundle.feature_spec.is_subset and len(bundle.feature_spec.features) == 66
    assert bundle.metadata["extra_metadata"]["efalls_coverage"]["n_available"] == 66
    dataset = ModelingDataset.load(reduced_world["data"], world["spec"], features=reduced_world["features"])
    preds = pd.read_parquet(result.run_dir / "predictions_test.parquet")
    merged = dataset.frame.merge(preds[["research_id", "index_date", "risk_served"]], on=["research_id", "index_date"])
    scored = predict_risk(merged.drop(columns=["risk_served"]), bundle)
    np.testing.assert_allclose([r.risk_12m for r in scored], merged["risk_served"].to_numpy(), rtol=1e-12, atol=1e-15)
    # CLI scoring of reduced rows from CSV
    csv = tmp_path / "reduced_rows.csv"
    merged.drop(columns=["risk_served"]).head(30).to_csv(csv, index=False)
    assert cli_main(["predict", "--model", str(result.run_dir / "model"), "--input", str(csv), "--out", str(tmp_path / "out.csv")]) == 0
    np.testing.assert_allclose(pd.read_csv(tmp_path / "out.csv")["risk_12m"].to_numpy(), merged["risk_served"].to_numpy()[:30], rtol=1e-12)
    # comparison: own category, never counted as the eFalls retraining
    table = build_comparison(reduced_world["runs"], world["base"] / "reports_reduced", n_bootstrap=20)
    assert table["category"].tolist() == ["reduced_local_retraining"] and not table["is_efalls_retrained"].any()
    assert table["efalls_coverage"].tolist() == ["66/78"]
    # exact reproduction selects the subset spec from the run's dataset manifest
    report = reproduce(result.run_dir)
    assert report["identical"], report["differences"]


def test_every_run_is_self_contained(world):
    for result in world["results"].values():
        missing = [a for a in REQUIRED_ARTIFACTS if not (result.run_dir / a).exists()]
        assert not missing, (result.run_id, missing)
        assert any((result.run_dir / "plots").glob("*.png"))
        metrics = json.loads((result.run_dir / "metrics.json").read_text(encoding="utf-8"))
        assert metrics["synthetic_fixture"] is True and metrics["eligibility"]["eligible_for_further_validation"] is False
        assert (result.run_dir / "report.md").read_text(encoding="utf-8").startswith("SYNTHETIC FIXTURE")


def test_published_and_retrained_experiments_are_distinct(world):
    pub = world["results"]["published"].metrics
    lasso = world["results"]["lasso"].metrics
    assert pub["experiment"]["kind"] == "efalls_published_scoring" and pub["published_scoring"]["primary"] == "lp_c_box_s3_1"
    assert set(pub["performance"]["test"]) >= {"lp_c_box_s3_1", "lp_a_table_s3_2", "lp_b_label_swap", "lp_d2_numeric_swap"}
    assert lasso["experiment"]["kind"] == "efalls_retrained" and lasso["published_scoring"] is None
    assert lasso["lasso"]["lambda_star"] > 0
    # identical test rows across experiments (D-19 common test)
    assert pub["split"]["test_rows_sha256"] == lasso["split"]["test_rows_sha256"]
    # the published equation is served as published; recalibration is reported, not silently served
    assert pub["served_variant"] == pub["primary_variant"] == "lp_c_box_s3_1" and pub["transportability_variant"] == "lp_c_box_s3_1"
    assert "lp_c_box_s3_1+recalibrated" in pub["performance"]["test"]
    assert set(pub["performance"]) == {"train", "validation", "test", "full_cohort"}
    assert not any(v.endswith("+recalibrated") for v in pub["performance"]["full_cohort"]), "no in-sample recalibration on the full cohort"
    assert lasso["served_variant"] == "recalibrated" and lasso["eligibility"]["judged_on_variant"] == "recalibrated"
    assert {"train", "validation", "test"} <= set(lasso["performance"])
    assert lasso["lasso"]["selection_rule"] == "min" and lasso["lasso"]["lambda_max"] > lasso["lasso"]["lambda_star"]
    for name in ("lasso", "rf"):
        comparison = world["results"][name].metrics["comparison_to_efalls"]
        assert comparison["reference_run_id"] == world["results"]["published"].run_id
        assert comparison["reference_variant"] == "lp_c_box_s3_1"
    # analytically: LP_B and LP_D2 are CITL shifts of LP_C and LP_A (same discrimination)
    t = pub["performance"]["test"]
    assert t["lp_b_label_swap"]["auroc"]["estimate"] == pytest.approx(t["lp_c_box_s3_1"]["auroc"]["estimate"], abs=1e-12)
    assert t["lp_d2_numeric_swap"]["auroc"]["estimate"] == pytest.approx(t["lp_a_table_s3_2"]["auroc"]["estimate"], abs=1e-12)


def test_disclosure_artifacts(world):
    lasso_dir = world["results"]["lasso"].run_dir
    coefs = pd.read_csv(lasso_dir / "coefficients.csv")
    assert {"omitted", "relative_to_reference"} <= set(coefs.columns)
    assert (coefs["design_column"] == "_intercept").sum() == 2, "intercept on the design scale and re-expressed (D-10)"
    assert (lasso_dir / "fp_selection.csv").exists()
    metrics = world["results"]["lasso"].metrics
    assert metrics["missingness"]["smoking_status"]["handling"].startswith("missing merged into 'never'")
    assert set(metrics["missingness"]["bmi_value"]["missing_rate_by_split"]) == {"train", "validation", "test"}
    pub = world["results"]["published"].metrics
    assert pub["missingness"]["smoking_status"]["handling"].startswith("missing -> ex/never")
    assert metrics["timing"]["total_train_seconds"] >= metrics["timing"]["final_fit_seconds"] > 0
    inputs = json.loads((lasso_dir / "run_inputs.json").read_text(encoding="utf-8"))
    assert len(inputs["files"]["feature_spec"][0]["sha256"]) == 64
    manifest = json.loads((lasso_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert Path(manifest["dataset_path_used"]) == world["data"]
    registry = (world["runs"] / "test_evaluation_registry.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(registry) >= 3


def test_no_row_leaks_between_partitions(world):
    splits = pd.read_csv(world["results"]["lasso"].run_dir / "splits.csv")
    key = splits["research_id"].astype(str) + "|" + splits["index_date"].astype(str)
    assert not key.duplicated().any()
    assert set(splits["split"]) == {"train", "validation", "test"}


def test_bundle_predictions_match_in_run_test_predictions(world):
    for name in ("published", "lasso", "rf"):
        run_dir = world["results"][name].run_dir
        bundle = load_bundle(run_dir / "model")
        splits = pd.read_csv(run_dir / "splits.csv", parse_dates=["index_date"])
        test_keys = splits.loc[splits["split"] == "test", ["research_id", "index_date"]]
        frame = world["dataset"].frame.merge(test_keys, on=["research_id", "index_date"], how="inner")
        preds = pd.read_parquet(run_dir / "predictions_test.parquet")
        merged = frame.merge(preds[["research_id", "index_date", "risk_served"]], on=["research_id", "index_date"])
        results = predict_risk(merged.drop(columns=["risk_served"]), bundle)
        np.testing.assert_allclose([r.risk_12m for r in results], merged["risk_served"].to_numpy(), rtol=1e-12, atol=1e-15)
        assert bundle.metadata["served_variant"] == world["results"][name].metrics["served_variant"]
        assert all(r.risk_category is None for r in results), "risk categories must stay hidden until approved"
        assert all("synthetic_training_data" in r.data_quality_flags for r in results)


def test_run_reproduces_exactly_from_artifacts(world):
    report = reproduce(world["results"]["lasso"].run_dir)  # dataset located from the run's own manifest
    assert report["identical"], report["differences"]
    reproduced = list((world["base"] / "runs_reproduced").glob(f"*/{'RUN_COMPLETE.json'}"))
    assert len(reproduced) == 1, "reproductions never land in the master runs directory"


def test_reproduce_finds_the_dataset_after_the_project_folder_moved(tmp_path):
    """dataset_manifest.json records a project-relative POSIX dataset path next to the absolute one and reproduce prefers it, so
    a handoff folder copied to another machine, drive letter or parent folder still reproduces without --dataset."""
    project = tmp_path / "Falls Research" / "proj"  # short: keeps test outputs far from the Windows 260-character limit
    (project / "configs").mkdir(parents=True)
    data = project / "data" / "fixtures" / "tiny_v1"
    generate_synthetic_modeling_dataset(data, n_patients=900, index_dates=("2018-04-01", "2019-04-01", "2021-04-01"),
                                        feature_spec_path=ROOT / "configs" / "features" / "efalls_v1.yaml")
    result = run_experiment(None, None, _config("efalls_published_scoring", data), runs_dir=project / "runs",
                            created_utc="2026-09-15T00:00:05+00:00")
    manifest = json.loads((result.run_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset_path_relative"] == "data/fixtures/tiny_v1" and Path(manifest["dataset_path_used"]) == data.resolve()
    moved = tmp_path / "moved" / "proj"
    shutil.move(str(project), str(moved))
    assert not Path(manifest["dataset_path_used"]).exists()
    report = reproduce(moved / "runs" / result.run_id)
    assert report["identical"], report["differences"]


def test_model_override_creates_a_separately_named_experiment(world):
    result = run_experiment("elastic_net_logistic", world["dataset"], _config("efalls_retrained_lasso_fixed_fp", world["data"]),
                            runs_dir=world["base"] / "override_runs")
    assert result.metrics["experiment"]["name"] == "efalls_retrained_lasso_fixed_fp__elastic_net_logistic"
    assert result.metrics["model"]["name"] == "elastic_net_logistic"
    assert result.metrics["experiment"]["kind"] == "alternative_model"
    assert "L4_alternative" in result.metrics["experiment"]["layers"]


def test_ablation_runs_nested_steps_on_identical_rows(tmp_path):
    ext = ROOT / "configs" / "features" / "meuhedet_enhanced_example.yaml"
    data = tmp_path / "enhanced"
    generate_synthetic_modeling_dataset(data, n_patients=600, index_dates=("2018-04-01", "2019-04-01", "2021-04-01"),
                                        feature_spec_path=ROOT / "configs" / "features" / "efalls_v1.yaml", extension_spec_paths=(ext,))
    # penalised base model: an unpenalised fit of ~80 columns on this tiny fixture separates (fails loudly by design)
    base = yaml.safe_load((FIXTURE_CONFIGS / "elastic_net_logistic.yaml").read_text(encoding="utf-8"))
    base["tuning"]["enabled"] = False
    base["evaluation"]["bootstrap"]["n"] = 10
    base["evaluation"]["permutation_importance_repeats"] = 1
    base["analysis"]["stability"]["n_bootstrap"] = 2
    (tmp_path / "base.yaml").write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
    (tmp_path / "ablation.yaml").write_text(yaml.safe_dump({
        "name": "t", "base_experiment_config": str(tmp_path / "base.yaml"), "extension_specs": [str(ext)],
        "steps": [{"name": "+ cognition", "groups": ["cognition"]}]}), encoding="utf-8")
    table = run_ablation(tmp_path / "ablation.yaml", data, runs_dir=tmp_path / "runs_ablation", out_dir=tmp_path / "ablation")
    assert len(table) == 2
    assert (tmp_path / "ablation" / "ablation_results.csv").exists() and (tmp_path / "ablation" / "ablation.md").exists()


def test_cli_predict_from_csv_in_foreign_working_directory(world, tmp_path, monkeypatch):
    run_dir = world["results"]["lasso"].run_dir
    frame = world["dataset"].frame.head(40).copy()
    frame["research_id"] = [f"{i:06d}" for i in range(len(frame))]  # zero-padded identifiers must survive CSV
    frame.loc[frame.index[:5], "smoking_status"] = pd.NA  # empty categorical cells are missing, not the text 'nan'
    csv = tmp_path / "input.csv"
    frame.to_csv(csv, index=False)
    monkeypatch.chdir(tmp_path)
    assert cli_main(["predict", "--model", str(run_dir / "model"), "--input", str(csv), "--out", "out.csv"]) == 0
    out = pd.read_csv(tmp_path / "out.csv", dtype={"research_id": str})
    assert out["research_id"].tolist() == frame["research_id"].tolist()
    expected = predict_risk(frame, load_bundle(run_dir / "model"))
    np.testing.assert_allclose(out["risk_12m"].to_numpy(), [r.risk_12m for r in expected], rtol=1e-12)
    with pytest.raises(SystemExit):
        cli_main(["predict", "--model", str(run_dir / "model"), "--input", str(csv), "--out", "out.txt"])


def test_comparison_and_best_features(world):
    out = world["base"] / "reports"
    table = build_comparison(world["runs"], out)
    assert {"scientific_reproduction", "local_retraining", "experimental_algorithm"} <= set(table["category"])
    assert table["is_efalls_published"].any() and table["is_efalls_retrained"].any()
    best = build_best_features(world["runs"], out)
    assert {"published_efalls_coefficient", "retrained_selected"} <= set(best.columns)
    assert (out / "best_features.md").read_text(encoding="utf-8").startswith("SYNTHETIC FIXTURE")


def test_monitoring_reports_without_retraining(world):
    run_dir = world["results"]["lasso"].run_dir
    bundle = load_bundle(run_dir / "model")
    shifted = world["dataset"].frame.assign(polypharmacy_count_120d=lambda d: d["polypharmacy_count_120d"] + 15)
    report = drift_report(bundle.reference_profile, shifted, world["spec"], pipeline=bundle.pipeline)
    assert report.overall_status in {"warn", "alert"}
    assert "No automatic retraining" in report.recommendation
