"""Unit tests for the contract modules: feature spec, schema validation, dataset manifest, config, artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from falls_ml.artifacts import create_run_directory, source_tree_sha256, write_json
from falls_ml.config import experiment_config_from_dict, load_experiment_config
from falls_ml.data.dataset import ModelingDataset, write_modeling_dataset
from falls_ml.data.schema import validate_modeling_dataset
from falls_ml.errors import ConfigError, DatasetValidationError
from falls_ml.features.spec import load_feature_spec
from falls_ml.paths import resolve_path

ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = ROOT / "configs" / "features" / "efalls_v1.yaml"


@pytest.fixture(scope="module")
def spec():
    return load_feature_spec(SPEC_PATH)


def _frame(spec, n: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "research_id": [f"r{i}" for i in range(n)],
        "index_date": pd.to_datetime(["2018-04-01"] * n),
        "predictor_max_record_date": pd.to_datetime(["2018-03-31"] * n),
        "outcome_12m": np.array([0, 1] * (n // 2), dtype="int8"),
        "outcome_first_event_date": pd.to_datetime([None, "2019-03-31"] * (n // 2)),
        "age_years": np.linspace(65.0, 99.0, n),
        "sex": ["female", "male"] * (n // 2),
        "polypharmacy_count_120d": rng.integers(0, 20, n).astype("int64"),
        "bmi_value": [22.0, np.nan, 31.0, 17.0, 26.0, 40.0][:n],
        "smoking_status": ["never", None, "current", "ex", None, "never"][:n],
        "alcohol_category": [None, "zero", None, "harmful", None, None][:n],
    })
    for b in spec.binary_names():
        df[b] = np.zeros(n, dtype="int8")
    return df


def _versions(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(dataset_version="v1", mapping_version="m1", source="synthetic_fixture")


class TestFeatureSpec:
    def test_counts_and_layers(self, spec):
        assert len(spec.features) == 78
        assert len(spec.binary_names()) == 72
        assert sum(1 for f in spec.features if f.retained_in_published_model) == 62
        assert spec.is_pure_efalls

    def test_outcome_window_end_calendar_year_minus_one_day(self, spec):
        assert spec.outcome.window_end(pd.Timestamp("2018-04-01")) == pd.Timestamp("2019-03-31")

    def test_subset_rejects_unknown(self, spec):
        with pytest.raises(ConfigError):
            spec.subset(["age_years", "not_a_feature"])

    def test_unknown_feature_key_rejected(self, tmp_path):
        raw = yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))
        raw["features"][0]["missing_rul"] = "forbid"
        bad = tmp_path / "bad.yaml"
        bad.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        with pytest.raises(ConfigError, match="unknown keys"):
            load_feature_spec(bad)

    def test_extension_cannot_claim_baseline(self, tmp_path, spec):
        ext = tmp_path / "ext.yaml"
        ext.write_text(json.dumps({"feature_set": {"name": "x", "version": "0"}, "features": [
            {"name": "mini_cog", "concept": "c", "dtype": "binary", "layer": "L1_published", "missing_rule": "absent_is_zero",
             "exact_efalls_baseline": True, "clinically_validated": False, "group": "cognition"}]}), encoding="utf-8")
        with pytest.raises(ConfigError):
            load_feature_spec(SPEC_PATH, [ext])


class TestSchema:
    def test_valid_frame_passes(self, spec):
        assert validate_modeling_dataset(_versions(_frame(spec)), spec).n_rows == 6

    @pytest.mark.parametrize("mutate, message", [
        (lambda d: d.drop(columns=["falls"]), "missing required columns"),
        (lambda d: d.assign(extra=1), "undeclared columns"),
        (lambda d: pd.concat([d, d.iloc[[0]]], ignore_index=True), "duplicated rows"),
        (lambda d: d.assign(outcome_12m=d["outcome_12m"].replace({1: 2})), "illegal outcome"),
        (lambda d: d.assign(age_years=d["age_years"] - 10), "below minimum"),
        (lambda d: d.assign(falls=d["falls"].astype("float64")), "integer/bool dtype"),
        (lambda d: d.assign(sex=["female", "unknown"] * 3), "undeclared levels"),
        (lambda d: d.assign(predictor_max_record_date=d["index_date"]), "LEAKAGE"),
        (lambda d: d.assign(outcome_first_event_date=pd.to_datetime([None, "2019-04-01"] * 3)), "outside [index"),
        (lambda d: d.assign(polypharmacy_count_120d=pd.array([None] + [1] * 5, dtype="Int64")), "null values"),
        (lambda d: d.drop(columns=["outcome_first_event_date"]), "missing required columns"),
    ])
    def test_violations_fail_loudly(self, spec, mutate, message):
        with pytest.raises(DatasetValidationError) as exc:
            validate_modeling_dataset(_versions(mutate(_frame(spec))), spec)
        assert any(message in p for p in exc.value.problems), exc.value.problems

    def test_inference_mode_does_not_require_outcome(self, spec):
        df = _frame(spec).drop(columns=["outcome_12m", "outcome_first_event_date", "research_id"])
        assert validate_modeling_dataset(df, spec, mode="inference").n_rows == 6


class TestDataset:
    def test_write_load_and_hash_verification(self, spec, tmp_path):
        m = write_modeling_dataset(_frame(spec), tmp_path, spec, dataset_version="v1", mapping_version="m1",
                                   source="synthetic_fixture", generator="test", scientific_use_allowed=False, created_utc="2026-01-01T00:00:00+00:00")
        ds = ModelingDataset.load(tmp_path, spec)
        assert len(ds) == 6 and ds.manifest.data_sha256 == m.data_sha256
        view = ds.frame
        view["age_years"] = 0.0
        assert (ds.frame["age_years"] > 0).all(), "callers must not be able to mutate the dataset"
        with (tmp_path / m.data_file).open("ab") as fh:
            fh.write(b"tamper")
        with pytest.raises(DatasetValidationError, match="hash"):
            ModelingDataset.load(tmp_path, spec)

    def test_datasets_are_immutable(self, spec, tmp_path):
        kwargs = dict(dataset_version="v1", mapping_version="m1", source="synthetic_fixture", generator="test",
                      scientific_use_allowed=False, created_utc="2026-01-01T00:00:00+00:00")
        write_modeling_dataset(_frame(spec), tmp_path, spec, **kwargs)
        with pytest.raises(DatasetValidationError, match="immutable"):
            write_modeling_dataset(_frame(spec), tmp_path, spec, **kwargs)

    def test_changed_definitions_with_same_version_are_detected(self, spec, tmp_path):
        data = tmp_path / "data"
        write_modeling_dataset(_frame(spec), data, spec, dataset_version="v1", mapping_version="m1", source="synthetic_fixture",
                               generator="test", scientific_use_allowed=False, created_utc="2026-01-01T00:00:00+00:00")
        raw = yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))
        raw["features"][0]["concept"] = raw["features"][0]["concept"] + " (edited)"
        edited = tmp_path / "configs" / "features" / "efalls_v1.yaml"
        edited.parent.mkdir(parents=True)
        edited.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        with pytest.raises(DatasetValidationError) as exc:
            ModelingDataset.load(data, load_feature_spec(edited))
        assert any("feature spec content" in p for p in exc.value.problems)


class TestConfig:
    def _raw(self):
        return {"experiment": {"name": "x", "kind": "efalls_retrained", "description": "d", "layers": ["L1_published"]},
                "dataset": {"feature_spec": str(SPEC_PATH)}, "model": {"name": "lasso_logistic_cv"},
                "preprocessing": {"representation": "efalls_fp_all_levels"},
                "validation": {"strategy": "temporal", "seed": 1, "temporal": {"train_end": "2018-12-31", "validation_end": "2019-12-31"}}}

    def test_unknown_key_rejected(self):
        raw = self._raw()
        raw["validation"]["sede"] = 3
        with pytest.raises(ConfigError, match="unknown keys"):
            experiment_config_from_dict(raw)

    def test_published_scoring_requires_explicit_sex_parameterisation(self):
        raw = self._raw()
        raw["experiment"]["kind"] = "efalls_published_scoring"
        raw["model"] = {"name": "efalls_published", "published_equation": {"config": "configs/models/efalls_published.yaml"}}
        raw["preprocessing"]["representation"] = "efalls_published"
        with pytest.raises(ConfigError, match="sex_parameterisation"):
            experiment_config_from_dict(raw)

    def test_retrained_cannot_use_extensions(self):
        raw = self._raw()
        raw["dataset"]["feature_spec_extensions"] = ["configs/features/meuhedet_enhanced_example.yaml"]
        with pytest.raises(ConfigError):
            experiment_config_from_dict(raw)

    def _published_raw(self, **published):
        raw = self._raw()
        raw["experiment"]["kind"] = "efalls_published_scoring"
        raw["model"] = {"name": "efalls_published", "published_equation": {
            "config": "configs/models/efalls_published.yaml", "sex_parameterisation": "lp_c_box_s3_1", "score_all_variants": True, **published}}
        raw["preprocessing"]["representation"] = "efalls_published"
        return raw

    def test_published_scoring_valid_minimal(self):
        assert experiment_config_from_dict(self._published_raw()).experiment.kind == "efalls_published_scoring"

    @pytest.mark.parametrize("change, message", [
        (lambda r: r["model"].update(params={"sex_parameterisation": "lp_a_table_s3_2"}), "model.params must be empty"),
        (lambda r: r["model"]["published_equation"].update(sex_parameterisation="lp_a_table_s3_2"), "lp_c_box_s3_1"),
        (lambda r: r["model"]["published_equation"].update(score_all_variants=False), "score_all_variants"),
        (lambda r: r["preprocessing"].update(features=["age_years", "sex"]), "feature subset"),
        (lambda r: r.update(tuning={"enabled": True}), "tuning is not allowed"),
        (lambda r: r["validation"]["temporal"].update(embargo_outcome_windows=False), "embargo"),
    ])
    def test_published_scoring_purity(self, change, message):
        raw = self._published_raw()
        change(raw)
        with pytest.raises(ConfigError, match=message):
            experiment_config_from_dict(raw)

    @pytest.mark.parametrize("change, message", [
        (lambda r: r["model"].update(params={"C": 1.0}), "not part of the published learning process"),
        (lambda r: r["model"].update(params={"selection": "1se"}), "lambda-min"),
        (lambda r: r["validation"].update(cv_folds=5), "10-fold"),
        (lambda r: r["preprocessing"].update(features=["age_years"]), "must be unset"),
        (lambda r: r.update(tuning={"enabled": True}), "tuning is not allowed"),
        (lambda r: r["experiment"].update(layers=["L1_published", "L3b_meuhedet_predictor"]), "L3b"),
    ])
    def test_retrained_purity(self, change, message):
        raw = self._raw()
        change(raw)
        with pytest.raises(ConfigError, match=message):
            experiment_config_from_dict(raw)

    def test_serves_recalibrated_defaults(self):
        pub = experiment_config_from_dict(self._published_raw())
        lasso = experiment_config_from_dict(self._raw())
        assert pub.calibration.serves_recalibrated(pub.experiment.kind) is False, "published equation is served as published by default"
        assert lasso.calibration.serves_recalibrated(lasso.experiment.kind) is True

    def test_config_hash_independent_of_source_location(self, tmp_path):
        source = ROOT / "configs" / "experiments" / "fixture" / "efalls_retrained_lasso.yaml"
        copy = tmp_path / "elsewhere.yaml"
        copy.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        assert load_experiment_config(source).sha256() == load_experiment_config(copy).sha256()

    def test_random_split_requires_limitation_note(self):
        raw = self._raw()
        raw["validation"] = {"strategy": "patient_grouped_random", "seed": 1}
        with pytest.raises(ConfigError, match="limitation_note"):
            experiment_config_from_dict(raw)

    def test_risk_categories_need_approval_details(self):
        raw = self._raw()
        raw["reporting"] = {"risk_categories": {"approved": True, "cutpoints": [0.1]}}
        with pytest.raises(ConfigError):
            experiment_config_from_dict(raw)

    @pytest.mark.parametrize("path", sorted((ROOT / "configs" / "experiments").rglob("*.yaml")), ids=lambda p: p.name)
    def test_repository_configs_load(self, path):
        if "ablation" in path.name:
            pytest.skip("ablation config has its own schema")
        cfg = load_experiment_config(path)
        assert len(cfg.sha256()) == 64


class TestPaths:
    def test_relative_paths_resolve_from_any_working_directory(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert resolve_path("configs/features/efalls_v1.yaml") == SPEC_PATH.resolve()
        assert resolve_path("configs/features/efalls_v1.yaml", anchor=tmp_path / "x.yaml") == SPEC_PATH.resolve()

    def test_missing_path_lists_candidates(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ConfigError, match="tried"):
            resolve_path("configs/features/nope.yaml")


class TestArtifacts:
    def test_run_directory_unique_and_json_sanitised(self, tmp_path):
        run = create_run_directory(tmp_path, "exp", "model", config_sha256="a", dataset_sha256="b", created_utc="2026-09-14T00:00:00+00:00")
        assert run.run_id.startswith("2026-09-14_exp_model_") and (run.path / "plots").is_dir()
        with pytest.raises(FileExistsError):
            create_run_directory(tmp_path, "exp", "model", config_sha256="a", dataset_sha256="b", created_utc="2026-09-14T00:00:00+00:00")
        write_json(run.file("m.json"), {"x": float("nan"), "y": np.float64(1.5)})
        assert json.loads(run.file("m.json").read_text(encoding="utf-8")) == {"x": None, "y": 1.5}
        assert len(source_tree_sha256()) == 64
