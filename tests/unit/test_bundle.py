"""Unit tests for the model bundle: save/load round trip, SHA-256 integrity, preprocessing identity (architecture §1.4)."""

from __future__ import annotations

import json
import pickle
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml
from scipy.special import expit

from falls_ml.bundle import (
    BUNDLE_FILE,
    FEATURE_SPEC_FILE,
    PREPROCESSOR_FILE,
    load_bundle,
    save_bundle,
)
from falls_ml.config import ExperimentConfig, experiment_config_from_dict
from falls_ml.data.dataset import DatasetManifest, sha256_file
from falls_ml.errors import BundleIntegrityError, ConfigError, PreprocessingMismatchError
from falls_ml.evaluation.calibration import LogisticRecalibrator
from falls_ml.features import spec as spec_module
from falls_ml.features.spec import FeatureSpec, feature_spec_payloads, load_feature_spec
from falls_ml.monitoring import build_reference_profile
from falls_ml.pipeline import FittedPipeline, fit_pipeline
from falls_ml.seeding import rng_for

REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "configs" / "features" / "efalls_v1.yaml"
FEATURES = ["age_years", "sex", "polypharmacy_count_120d", "bmi_value", "smoking_status", "alcohol_category",
            "falls", "dementia", "fracture"]
ALCOHOL = ["harmful", "higher_risk", "lower_risk", "previous_higher_risk_or_harmful", "zero"]
CREATED = "2026-09-15T00:00:00+00:00"


def make_frame(spec: FeatureSpec, n: int, component: str) -> pd.DataFrame:
    rng = rng_for(20260915, component)
    index = pd.Timestamp("2019-01-01") + pd.to_timedelta(rng.integers(0, 720, n), unit="D")
    df = pd.DataFrame({
        "research_id": pd.Series([f"{component}-{i}" for i in range(n)], dtype="str"),
        "index_date": index,
        "predictor_max_record_date": index - pd.Timedelta(days=1),
        "age_years": rng.uniform(65.0, 95.0, n),
        "sex": pd.Series(rng.choice(["female", "male"], n), dtype="str"),
        "polypharmacy_count_120d": rng.integers(0, 25, n).astype("int64"),
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
                params: dict[str, Any] | None = None) -> ExperimentConfig:
    return experiment_config_from_dict({
        "experiment": {"name": "bundle_test", "kind": "alternative_model", "description": "unit test",
                       "layers": ["L1_published", "L4_alternative"]},
        "dataset": {"feature_spec": str(SPEC_PATH)},
        "model": {"name": model, "params": params or {}},
        "preprocessing": {"representation": representation},
        "validation": {"strategy": "patient_grouped_random", "seed": 1, "limitation_note": "unit test"},
    })


def make_manifest(spec: FeatureSpec, n_rows: int) -> DatasetManifest:
    return DatasetManifest(dataset_version="test-v1", mapping_version="mapping-v0", source="synthetic_fixture",
                           feature_spec_name=spec.name, feature_spec_version=spec.version,
                           feature_spec_sha256=spec.content_sha256, data_file="modeling_dataset.parquet",
                           data_sha256="0" * 64, n_rows=n_rows, n_patients=n_rows, index_date_min="2019-01-01",
                           index_date_max="2020-12-31", outcome_prevalence=None, created_utc=CREATED,
                           generator="unit-test", scientific_use_allowed=False)


def write_bundle(directory: Path, pipeline: FittedPipeline, config: ExperimentConfig, train: pd.DataFrame,
                 extra: dict[str, Any] | None = None) -> Path:
    profile = build_reference_profile(train, pipeline.spec, pipeline)
    return save_bundle(directory, pipeline, experiment_config=config, dataset_manifest=make_manifest(pipeline.spec, len(train)),
                       reference_profile=profile, model_version="model-v1", created_utc=CREATED,
                       extra_metadata=extra if extra is not None else {"mappings_clinically_validated": False})


def rewrite_bundle_json(bundle_dir: Path, edit) -> None:
    meta = json.loads((bundle_dir / BUNDLE_FILE).read_text(encoding="utf-8"))
    edit(meta)
    (bundle_dir / BUNDLE_FILE).write_text(json.dumps(meta), encoding="utf-8")


def rewrite_listed_file(bundle_dir: Path, relative: str, content: bytes) -> None:
    """Replace a bundle file *and* its manifest hash (simulates a consistent but different bundle)."""
    (bundle_dir / relative).write_bytes(content)
    meta = json.loads((bundle_dir / BUNDLE_FILE).read_text(encoding="utf-8"))
    meta["files"][relative] = sha256_file(bundle_dir / relative)
    (bundle_dir / BUNDLE_FILE).write_text(json.dumps(meta), encoding="utf-8")


@pytest.fixture(scope="module")
def spec() -> FeatureSpec:
    return load_feature_spec(SPEC_PATH).subset(FEATURES)


@pytest.fixture(scope="module")
def train(spec: FeatureSpec) -> pd.DataFrame:
    return make_frame(spec, 1200, "train")


@pytest.fixture(scope="module")
def new_rows(spec: FeatureSpec) -> pd.DataFrame:
    return make_frame(spec, 300, "new")


@pytest.fixture(scope="module")
def pipeline(spec: FeatureSpec, train: pd.DataFrame) -> FittedPipeline:
    fitted = fit_pipeline(train, spec, make_config(), random_state=3)
    fitted.calibrator = LogisticRecalibrator(alpha=0.2, beta=0.9)
    return fitted


@pytest.fixture(scope="module")
def template_bundle(tmp_path_factory: pytest.TempPathFactory, pipeline: FittedPipeline, train: pd.DataFrame) -> Path:
    return write_bundle(tmp_path_factory.mktemp("template") / "model", pipeline, make_config(), train)


@pytest.fixture
def bundle_dir(tmp_path: Path, template_bundle: Path) -> Path:
    """A private copy of the saved bundle that a test may tamper with."""
    return Path(shutil.copytree(template_bundle, tmp_path / "model"))


# ---------------------------------------------------------------------- round trip
def test_save_load_predictions_and_design_are_identical(bundle_dir, pipeline, new_rows):
    loaded = load_bundle(bundle_dir)
    pd.testing.assert_frame_equal(loaded.pipeline.design(new_rows), pipeline.design(new_rows), check_exact=True)
    for calibrated in (False, True):
        assert np.array_equal(loaded.pipeline.predict_proba(new_rows, calibrated=calibrated),
                              pipeline.predict_proba(new_rows, calibrated=calibrated))
    assert loaded.pipeline.preprocessor.fingerprint() == pipeline.preprocessor.fingerprint()
    assert loaded.feature_spec.content_sha256 == pipeline.spec.content_sha256  # subset spec rebuilt from embedded payloads
    assert loaded.feature_spec.subset_of == pipeline.spec.subset_of and loaded.feature_spec.cohort == pipeline.spec.cohort
    assert loaded.model_version == "model-v1"
    assert loaded.warnings == ()


def test_bundle_metadata_records_provenance(bundle_dir, pipeline):
    meta = load_bundle(bundle_dir).metadata
    pre = pipeline.preprocessor
    assert meta["model"] == {"name": "logistic_unpenalized", "version": "model-v1", "params": {"maxiter": 100},
                             "is_linear": True, "representation": "efalls_reference_coded"}
    assert meta["experiment"]["layers"] == ["L1_published", "L4_alternative"]
    assert meta["config_sha256"] == make_config().sha256() and meta["config"] == make_config().to_dict()
    assert meta["mapping_version"] == "mapping-v0" and meta["dataset_version"] == "test-v1"
    assert meta["training_dataset"]["source"] == "synthetic_fixture"
    assert meta["preprocessing"]["fingerprint"] == pre.fingerprint()
    assert meta["preprocessing"]["design_columns"] == pre.design_columns()
    assert meta["preprocessing"]["selected_design_columns"] == pre.design_columns()  # unpenalised: all non-zero
    assert meta["preprocessing"]["category_levels"]["sex"] == ["female", "male"]
    assert meta["risk_categories"]["approved"] is False
    assert meta["calibration"]["method"] == "logistic_intercept_slope"
    assert meta["calibration"]["params"] == {"alpha": 0.2, "beta": 0.9}
    assert meta["served_variant"] == "recalibrated" and load_bundle(bundle_dir).served_variant == "recalibrated"
    assert set(meta["code_version"]) == {"git_commit", "source_tree_sha256"}
    assert "scikit-learn" in meta["environment"]["packages"]
    assert meta["created_utc"] == CREATED and meta["extra_metadata"] == {"mappings_clinically_validated": False}
    listed = set(meta["files"])
    on_disk = {p.relative_to(bundle_dir).as_posix() for p in bundle_dir.rglob("*") if p.is_file()} - {BUNDLE_FILE}
    assert listed == on_disk
    assert {PREPROCESSOR_FILE, "calibrator.json", "reference_profile.json", FEATURE_SPEC_FILE, "adapter/adapter.json"} <= listed
    identity = json.loads((bundle_dir / FEATURE_SPEC_FILE).read_text(encoding="utf-8"))
    assert identity["predictor_names"] == pipeline.spec.predictor_names() and identity["sha256"] == pipeline.spec.content_sha256
    assert identity["payloads"] == json.loads(json.dumps(feature_spec_payloads(pipeline.spec)))
    assert identity["payloads"]["subset"] == pipeline.spec.predictor_names() and identity["payloads"]["extensions"] == []


def test_preprocessor_is_pickled_with_protocol_5(bundle_dir):
    import pickletools

    ops = list(pickletools.genops((bundle_dir / PREPROCESSOR_FILE).read_bytes()))
    assert ops[0][0].name == "PROTO" and ops[0][1] == 5


def test_missing_calibrator_round_trips_as_none(tmp_path, spec, train, new_rows):
    fitted = fit_pipeline(train, spec, make_config(), random_state=3)
    loaded = load_bundle(write_bundle(tmp_path / "model", fitted, make_config(), train))
    assert json.loads((tmp_path / "model" / "calibrator.json").read_text(encoding="utf-8"))["method"] == "none"
    assert loaded.pipeline.calibrator is None and loaded.served_variant == "uncalibrated"
    assert np.array_equal(loaded.pipeline.predict_proba(new_rows), fitted.predict_proba(new_rows))


def test_non_linear_model_round_trip(tmp_path, spec, train, new_rows):
    config = make_config("random_forest", "efalls_raw", {"n_estimators": 20, "min_samples_leaf": 20})
    fitted = fit_pipeline(train, spec, config, random_state=5)
    loaded = load_bundle(write_bundle(tmp_path / "model", fitted, config, train))
    assert np.array_equal(loaded.pipeline.predict_proba(new_rows), fitted.predict_proba(new_rows))
    assert loaded.metadata["preprocessing"]["selected_design_columns"] == fitted.preprocessor.design_columns()


def test_refuses_to_write_into_non_empty_directory(tmp_path, pipeline, train):
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "stale.txt").write_text("x", encoding="utf-8")
    with pytest.raises(BundleIntegrityError, match="non-empty"):
        write_bundle(tmp_path / "model", pipeline, make_config(), train)


# ---------------------------------------------------------------------- integrity
@pytest.mark.parametrize("relative", [PREPROCESSOR_FILE, "adapter/adapter.json", "calibrator.json", "reference_profile.json"])
def test_tampered_byte_raises_before_unpickling(bundle_dir, relative, monkeypatch):
    path = bundle_dir / relative
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0x01
    path.write_bytes(bytes(data))

    def no_unpickling(*args, **kwargs):
        raise AssertionError("pickle.load called before integrity verification")

    monkeypatch.setattr(pickle, "load", no_unpickling)
    with pytest.raises(BundleIntegrityError, match="SHA-256 mismatch"):
        load_bundle(bundle_dir)


def test_missing_and_unexpected_files_raise(bundle_dir):
    (bundle_dir / "calibrator.json").unlink()
    (bundle_dir / "adapter" / "extra.bin").write_bytes(b"\x00")
    with pytest.raises(BundleIntegrityError) as excinfo:
        load_bundle(bundle_dir)
    assert "missing file: calibrator.json" in str(excinfo.value)
    assert "unexpected file: adapter/extra.bin" in str(excinfo.value)


def test_file_manager_metadata_is_ignored_but_other_unlisted_files_are_not(bundle_dir, pipeline, new_rows):
    """Windows Explorer / macOS Finder drop desktop.ini, Thumbs.db, .DS_Store into folders; they are never read by load_bundle."""
    (bundle_dir / "desktop.ini").write_bytes(b"[.ShellClassInfo]\r\n")
    (bundle_dir / "adapter" / "THUMBS.DB").write_bytes(b"\x00")
    (bundle_dir / ".DS_Store").write_bytes(b"\x00")
    loaded = load_bundle(bundle_dir)
    assert np.array_equal(loaded.pipeline.predict_proba(new_rows), pipeline.predict_proba(new_rows))
    (bundle_dir / "calibrator-DESKTOP-AB12CD.json").write_bytes(b"{}")
    with pytest.raises(BundleIntegrityError, match="unexpected file: calibrator-DESKTOP-AB12CD.json.*conflict copies"):
        load_bundle(bundle_dir)


def test_directory_without_manifest_raises(tmp_path):
    with pytest.raises(BundleIntegrityError, match="not a model bundle"):
        load_bundle(tmp_path)


# ---------------------------------------------------------------------- preprocessing identity
def test_modified_preprocessor_state_raises_mismatch(bundle_dir):
    with (bundle_dir / PREPROCESSOR_FILE).open("rb") as handle:
        pre = pickle.load(handle)
    pre.bmi_obese_cutpoint = 32.0  # changes transform output without changing the design columns
    rewrite_listed_file(bundle_dir, PREPROCESSOR_FILE, pickle.dumps(pre, protocol=5))
    with pytest.raises(PreprocessingMismatchError, match="fingerprint"):
        load_bundle(bundle_dir)


def test_load_never_reads_feature_spec_yaml(tmp_path, template_bundle, pipeline, new_rows, monkeypatch):
    """The bundle is self-contained: the spec is rebuilt from feature_spec.json, from any working directory."""
    bundle_dir = Path(shutil.copytree(template_bundle, tmp_path / "elsewhere" / "model"))
    identity = json.loads((bundle_dir / FEATURE_SPEC_FILE).read_text(encoding="utf-8"))
    identity["source_paths"] = ["configs/features/does_not_exist.yaml"]  # provenance only
    rewrite_listed_file(bundle_dir, FEATURE_SPEC_FILE, json.dumps(identity).encode("utf-8"))

    def no_yaml(*args, **kwargs):
        raise AssertionError("load_bundle must not read feature spec YAML files")

    monkeypatch.setattr(spec_module, "_read_yaml", no_yaml)
    monkeypatch.setattr(spec_module, "load_feature_spec", no_yaml)
    monkeypatch.chdir(tmp_path / "elsewhere")
    loaded = load_bundle(bundle_dir)
    assert loaded.feature_spec.content_sha256 == pipeline.spec.content_sha256
    assert np.array_equal(loaded.pipeline.predict_proba(new_rows, calibrated=True), pipeline.predict_proba(new_rows, calibrated=True))


def test_edited_embedded_feature_spec_raises_mismatch(bundle_dir, pipeline):
    identity = json.loads((bundle_dir / FEATURE_SPEC_FILE).read_text(encoding="utf-8"))
    identity["payloads"]["base"]["features"][0]["concept"] += " (edited)"  # consistent file hash, different content
    rewrite_listed_file(bundle_dir, FEATURE_SPEC_FILE, json.dumps(identity).encode("utf-8"))
    with pytest.raises(PreprocessingMismatchError, match="does not reproduce the training SHA-256"):
        load_bundle(bundle_dir)
    with pytest.raises(PreprocessingMismatchError, match="does not reproduce"):  # an expected spec does not bypass the check
        load_bundle(bundle_dir, expected_feature_spec=pipeline.spec)


@pytest.mark.parametrize("edit", [lambda i: i.pop("payloads"), lambda i: i["payloads"].pop("base"),
                                  lambda i: i["payloads"].update(subset=["not_a_feature"])])
def test_missing_or_malformed_embedded_payloads_raise(bundle_dir, edit):
    identity = json.loads((bundle_dir / FEATURE_SPEC_FILE).read_text(encoding="utf-8"))
    edit(identity)
    rewrite_listed_file(bundle_dir, FEATURE_SPEC_FILE, json.dumps(identity).encode("utf-8"))
    with pytest.raises(BundleIntegrityError, match="rebuildable feature spec"):
        load_bundle(bundle_dir)


def test_expected_feature_spec_is_an_extra_check(bundle_dir, pipeline):
    loaded = load_bundle(bundle_dir, expected_feature_spec=pipeline.spec)
    assert loaded.feature_spec.content_sha256 == pipeline.spec.content_sha256


def test_expected_feature_spec_with_different_hash_raises(bundle_dir):
    other = load_feature_spec(SPEC_PATH).subset(FEATURES[:-1])
    with pytest.raises(PreprocessingMismatchError, match="expected_feature_spec"):
        load_bundle(bundle_dir, expected_feature_spec=other)


def test_library_version_differences_become_warnings(bundle_dir):
    meta = json.loads((bundle_dir / BUNDLE_FILE).read_text(encoding="utf-8"))
    meta["environment"]["packages"]["numpy"] = "0.0.1"
    (bundle_dir / BUNDLE_FILE).write_text(json.dumps(meta), encoding="utf-8")
    loaded = load_bundle(bundle_dir)
    assert any("numpy bundle=0.0.1" in w for w in loaded.warnings)


# ---------------------------------------------------------------------- reviewer additions
def test_tampered_feature_spec_identity_raises(bundle_dir):
    path = bundle_dir / FEATURE_SPEC_FILE
    path.write_text(path.read_text(encoding="utf-8").replace('"name"', '"name" ', 1), encoding="utf-8")
    with pytest.raises(BundleIntegrityError, match=f"SHA-256 mismatch: {FEATURE_SPEC_FILE}"):
        load_bundle(bundle_dir)


def test_subset_spec_keeping_every_predictor_reloads(tmp_path, train, new_rows):
    """``FeatureSpec.subset`` of all predictors is the full spec itself (idempotent, root-based digest) and reloads."""
    raw = yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))
    raw["features"] = [f for f in raw["features"] if f["name"] in FEATURES]
    base_path = tmp_path / "efalls_small.yaml"
    base_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    base = load_feature_spec(base_path)
    spec_all = base.subset(base.predictor_names())
    assert spec_all.predictor_names() == base.predictor_names() and spec_all.content_sha256 == base.content_sha256
    assert spec_all.subset_of is None
    fitted = fit_pipeline(train, spec_all, make_config(), random_state=3)
    loaded = load_bundle(write_bundle(tmp_path / "model", fitted, make_config(), train))
    assert loaded.feature_spec.content_sha256 == spec_all.content_sha256
    assert np.array_equal(loaded.pipeline.predict_proba(new_rows), fitted.predict_proba(new_rows))


def test_hand_edited_risk_category_approval_is_refused(bundle_dir):
    """bundle.json is outside its own hash manifest; category exposure (M-12) must not be switched on by editing it."""
    meta = json.loads((bundle_dir / BUNDLE_FILE).read_text(encoding="utf-8"))
    meta["risk_categories"] = {"approved": True, "cutpoints": [0.2], "labels": ["low", "high"], "approval_reference": "x"}
    (bundle_dir / BUNDLE_FILE).write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(BundleIntegrityError, match="risk_categories differ"):
        load_bundle(bundle_dir)
    meta["config"]["reporting"]["risk_categories"] = meta["risk_categories"]  # edit the config too: its hash no longer matches
    (bundle_dir / BUNDLE_FILE).write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(BundleIntegrityError, match="config_sha256"):
        load_bundle(bundle_dir)


def test_reference_profile_of_another_pipeline_is_refused(tmp_path, spec, pipeline, train):
    other_spec = load_feature_spec(SPEC_PATH).subset(FEATURES[:-1])
    other = fit_pipeline(train, other_spec, make_config(), random_state=3)
    with pytest.raises(PreprocessingMismatchError, match="reference_profile"):
        save_bundle(tmp_path / "model", pipeline, experiment_config=make_config(), dataset_manifest=make_manifest(spec, len(train)),
                    reference_profile=build_reference_profile(train, other_spec, other), model_version="v", created_utc=CREATED)
    assert not (tmp_path / "model").exists()  # refused before anything is written


def test_extension_spec_bundle_is_self_contained(tmp_path, train, new_rows, monkeypatch):
    ext = {"feature_set": {"name": "bundle_ext", "version": "0.0.1", "layer": "L3b_meuhedet_predictor"},
           "features": [{"name": "egfr", "concept": "eGFR", "dtype": "float_nullable", "missing_rule": "missing_category",
                         "layer": "L3b_meuhedet_predictor", "exact_efalls_baseline": False, "clinically_validated": False,
                         "group": "labs"}]}
    ext_path = tmp_path / "bundle_ext.yaml"
    ext_path.write_text(yaml.safe_dump(ext), encoding="utf-8")
    spec = load_feature_spec(SPEC_PATH, extensions=[ext_path]).subset([*FEATURES, "egfr"])
    rows = {"train": train, "new": new_rows}
    for name, df in rows.items():
        rows[name] = df.assign(egfr=np.where(np.arange(len(df)) % 4 == 0, np.nan, 40.0 + np.arange(len(df)) % 30))
    fitted = fit_pipeline(rows["train"], spec, make_config(), random_state=3)
    directory = write_bundle(tmp_path / "model", fitted, make_config(), rows["train"])
    ext_path.unlink()  # the extension YAML is gone: loading must not need it
    loaded = load_bundle(directory)
    assert loaded.feature_spec.content_sha256 == spec.content_sha256 and loaded.feature_spec.name == "efalls_v1+bundle_ext"
    assert np.array_equal(loaded.pipeline.predict_proba(rows["new"]), fitted.predict_proba(rows["new"]))


def test_served_variant_follows_extra_metadata_and_must_agree_with_the_calibrator(tmp_path, spec, pipeline, train):
    config = make_config()
    published_like = write_bundle(tmp_path / "a", pipeline, config, train, extra={"served_variant": "lp_c_box_s3_1+recalibrated"})
    assert load_bundle(published_like).served_variant == "lp_c_box_s3_1+recalibrated"
    with pytest.raises(ConfigError, match="contradicts the pipeline"):
        write_bundle(tmp_path / "b", pipeline, config, train, extra={"served_variant": "uncalibrated"})
    uncalibrated = fit_pipeline(train, spec, config, random_state=3)
    with pytest.raises(ConfigError, match="contradicts the pipeline"):
        write_bundle(tmp_path / "c", uncalibrated, config, train, extra={"served_variant": "recalibrated"})
    assert not (tmp_path / "b").exists() and not (tmp_path / "c").exists()
    assert load_bundle(write_bundle(tmp_path / "d", uncalibrated, config, train,
                                    extra={"served_variant": "lp_c_box_s3_1"})).served_variant == "lp_c_box_s3_1"


def test_hand_edited_or_missing_served_variant_is_refused(bundle_dir):
    """bundle.json is outside its own hash manifest; served_variant must agree with the hashed calibrator.json."""
    rewrite_bundle_json(bundle_dir, lambda m: m.update(served_variant="uncalibrated"))
    with pytest.raises(BundleIntegrityError, match="served_variant 'uncalibrated' contradicts"):
        load_bundle(bundle_dir)
    rewrite_bundle_json(bundle_dir, lambda m: m.pop("served_variant"))
    with pytest.raises(BundleIntegrityError, match="served_variant"):
        load_bundle(bundle_dir)
