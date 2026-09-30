"""CLI dataset commands: ``build-dataset`` (CSV/Parquet extract -> immutable dataset), ``validate-dataset`` spec selection and
``make-fixture --features-from-config``. SYNTHETIC test data only; no database is involved anywhere."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from falls_ml.cli import main as cli_main
from falls_ml.cli import spec_csv_dtypes
from falls_ml.data.dataset import ModelingDataset
from falls_ml.data.synthetic import generate_synthetic_modeling_dataset
from falls_ml.errors import ConfigError, DatasetValidationError
from falls_ml.features.spec import load_feature_spec
from tests.helpers.fake_run import reduced_fixture_features

ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = ROOT / "configs" / "features" / "efalls_v1.yaml"
REDUCED_CONFIG = ROOT / "configs" / "experiments" / "fixture" / "efalls_retrained_reduced.yaml"
VERSION_COLUMNS = ("dataset_version", "mapping_version", "source")


@pytest.fixture(scope="module")
def spec():
    return load_feature_spec(SPEC_PATH)


@pytest.fixture(scope="module")
def extract(tmp_path_factory, spec) -> pd.DataFrame:
    """A flat extract (no version columns) from a tiny SYNTHETIC fixture, with zero-padded research ids and practice ids."""
    base = tmp_path_factory.mktemp("extract")
    generate_synthetic_modeling_dataset(base / "fixture", n_patients=120, feature_spec_path=SPEC_PATH)
    df = ModelingDataset.load(base / "fixture", spec).frame.drop(columns=list(VERSION_COLUMNS))
    ids = {rid: f"{i:07d}" for i, rid in enumerate(sorted(df["research_id"].unique()))}
    df["research_id"] = df["research_id"].map(ids)
    df["practice_id"] = df["practice_id"].astype("string").str.replace("P", "0", regex=False)
    assert df["research_id"].str.startswith("0").all()
    return df.reset_index(drop=True)


def _run(capsys, *argv: str) -> dict:
    assert cli_main(list(argv)) == 0
    return json.loads(capsys.readouterr().out)


def _build_args(source: Path, out: Path, *extra: str) -> list[str]:
    return ["build-dataset", "--input", str(source), "--out", str(out), "--dataset-version", "extract-2026-09",
            "--mapping-version", "meuhedet_v0", "--source", "test_extract", "--data-freeze-date", "2023-12-31", *extra]


def _write_csv(df: pd.DataFrame, path: Path) -> Path:
    df.to_csv(path, index=False, date_format="%Y-%m-%d")
    return path


# ============================================================================ build-dataset
def test_build_dataset_from_csv_preserves_ids_and_types(tmp_path, capsys, extract, spec):
    source = _write_csv(extract, tmp_path / "extract.csv")
    out = tmp_path / "dataset"
    result = _run(capsys, *_build_args(source, out, "--feature-spec", str(SPEC_PATH)))
    ds = ModelingDataset.load(out, spec)
    frame = ds.frame
    assert frame["research_id"].tolist() == extract["research_id"].tolist() and frame["research_id"].str.len().eq(7).all()
    assert frame["practice_id"].dropna().str.startswith("0").all(), "string metadata keeps leading zeros"
    for column in ("sex", "smoking_status", "alcohol_category"):
        assert not frame[column].astype("string").eq("nan").any()
    pd.testing.assert_series_equal(frame["smoking_status"].isna(), extract["smoking_status"].isna(), check_names=False)
    np.testing.assert_allclose(frame["bmi_value"].to_numpy(dtype=float), extract["bmi_value"].to_numpy(dtype=float), equal_nan=True)
    assert (frame["index_date"] == extract["index_date"]).all()
    m = ds.manifest
    assert (m.dataset_version, m.mapping_version, m.source, m.data_freeze_date) == ("extract-2026-09", "meuhedet_v0", "test_extract", "2023-12-31")
    assert m.scientific_use_allowed is False and m.generator == "falls_ml build-dataset" and m.n_rows == len(extract)
    audit = m.audit["build_dataset"]
    assert audit["input_file_name"] == "extract.csv" and len(audit["input_sha256"]) == 64 and audit["database_connections"] == "none"
    assert audit["scientific_use_approval_reference"] is None and "feature_spec_subset" not in m.audit
    assert result["feature_spec"]["is_subset"] is False and result["manifest"]["data_sha256"] == m.data_sha256


def test_build_dataset_from_parquet(tmp_path, capsys, extract, spec):
    source = tmp_path / "extract.parquet"
    extract.to_parquet(source, index=False)
    _run(capsys, *_build_args(source, tmp_path / "dataset"))
    ds = ModelingDataset.load(tmp_path / "dataset", spec)
    assert ds.frame["research_id"].tolist() == extract["research_id"].tolist() and len(ds) == len(extract)


def test_build_dataset_refuses_non_empty_out(tmp_path, extract):
    source = _write_csv(extract, tmp_path / "extract.csv")
    out = tmp_path / "dataset"
    out.mkdir()
    (out / "something.txt").write_text("x", encoding="utf-8")
    with pytest.raises(SystemExit, match="not an empty directory"):
        cli_main(_build_args(source, out))
    assert sorted(p.name for p in out.iterdir()) == ["something.txt"]


def test_scientific_use_needs_an_approval_reference(tmp_path, capsys, extract, spec):
    source = _write_csv(extract, tmp_path / "extract.csv")
    with pytest.raises(SystemExit, match="requires --approval-reference"):
        cli_main(_build_args(source, tmp_path / "a", "--scientific-use-allowed"))
    with pytest.raises(SystemExit, match="requires --approval-reference"):
        cli_main(_build_args(source, tmp_path / "a", "--scientific-use-allowed", "--approval-reference", "  "))
    with pytest.raises(SystemExit, match="only recorded together with --scientific-use-allowed"):
        cli_main(_build_args(source, tmp_path / "a", "--approval-reference", "IRB-1"))
    assert not (tmp_path / "a").exists()
    _run(capsys, *_build_args(source, tmp_path / "b", "--scientific-use-allowed", "--approval-reference", "HELSINKI-0042-26"))
    m = ModelingDataset.load(tmp_path / "b", spec).manifest
    assert m.scientific_use_allowed is True and m.audit["build_dataset"]["scientific_use_approval_reference"] == "HELSINKI-0042-26"


@pytest.mark.parametrize("value", ["2023-13-01", "31/12/2023", "2023-12-31T00:00"])
def test_build_dataset_requires_a_calendar_freeze_date(tmp_path, extract, value):
    source = _write_csv(extract.head(5), tmp_path / "extract.csv")
    args = _build_args(source, tmp_path / "dataset")
    args[args.index("--data-freeze-date") + 1] = value
    with pytest.raises(SystemExit, match="--data-freeze-date"):
        cli_main(args)


def test_build_dataset_validates_strictly(tmp_path, extract):
    bad = extract.assign(unexpected_column=1)
    with pytest.raises(DatasetValidationError, match="failed training validation") as exc:
        cli_main(_build_args(_write_csv(bad, tmp_path / "bad.csv"), tmp_path / "dataset"))
    assert any("unexpected_column" in p for p in exc.value.problems)
    duplicated = pd.concat([extract, extract.head(1)], ignore_index=True)
    with pytest.raises(DatasetValidationError) as exc:
        cli_main(_build_args(_write_csv(duplicated, tmp_path / "dup.csv"), tmp_path / "dataset2"))
    assert any("duplicated rows" in p for p in exc.value.problems)
    with pytest.raises(SystemExit, match="does not exist"):
        cli_main(_build_args(tmp_path / "missing.csv", tmp_path / "dataset3"))


def test_build_dataset_with_declared_subset(tmp_path, capsys, extract, spec):
    features = reduced_fixture_features()
    absent = [n for n in spec.predictor_names() if n not in features]
    source = _write_csv(extract.drop(columns=absent), tmp_path / "reduced.csv")
    out = tmp_path / "reduced_dataset"
    result = _run(capsys, *_build_args(source, out, "--features-from-config", str(REDUCED_CONFIG)))
    subset = spec.subset(features)
    assert result["feature_spec"]["sha256"] == subset.content_sha256 and result["feature_spec"]["absent_predictors"] == absent
    ds = ModelingDataset.load(out, spec, features=features)
    assert ds.spec.content_sha256 == subset.content_sha256 and ds.manifest.audit["feature_spec_subset"]["n_predictors"] == 66
    # a full extract is not silently cut down to the declared subset
    with pytest.raises(DatasetValidationError) as exc:
        cli_main(_build_args(_write_csv(extract, tmp_path / "full.csv"), tmp_path / "x", "--features-from-config", str(REDUCED_CONFIG)))
    assert any("undeclared columns" in p and "bmi_value" in p for p in exc.value.problems)


def test_csv_dtypes_come_from_the_spec(spec):
    header = ["research_id", "index_date", "predictor_max_record_date", "outcome_first_event_date", "sex", "age_years", "falls",
              "practice_id", "death_date", "dataset_version"]
    dtypes, dates = spec_csv_dtypes(spec, header)
    assert dtypes == {"research_id": "string", "sex": "string", "practice_id": "string", "dataset_version": "string"}
    assert dates == ["index_date", "predictor_max_record_date", "outcome_first_event_date", "death_date"]


# ============================================================================ validate-dataset
def test_validate_dataset_selects_full_or_subset_spec(tmp_path, capsys, spec):
    features = reduced_fixture_features()
    generate_synthetic_modeling_dataset(tmp_path / "full", n_patients=120, feature_spec_path=SPEC_PATH)
    generate_synthetic_modeling_dataset(tmp_path / "reduced", n_patients=120, feature_spec_path=SPEC_PATH, features=features)
    auto = _run(capsys, "validate-dataset", "--dataset", str(tmp_path / "reduced"))
    assert auto["valid"] and auto["feature_spec"]["is_subset"] and auto["feature_spec"]["n_predictors"] == 66
    assert auto["feature_spec"]["root_sha256"] == spec.content_sha256 and auto["features_from_config"] is None
    assert _run(capsys, "validate-dataset", "--dataset", str(tmp_path / "full"))["feature_spec"]["is_subset"] is False
    declared = _run(capsys, "validate-dataset", "--dataset", str(tmp_path / "reduced"), "--features-from-config", str(REDUCED_CONFIG))
    assert declared["feature_spec"]["sha256"] == spec.subset(features).content_sha256
    full_with_config = _run(capsys, "validate-dataset", "--dataset", str(tmp_path / "full"), "--features-from-config", str(REDUCED_CONFIG))
    assert full_with_config["feature_spec"]["sha256"] == spec.content_sha256
    retrained = ROOT / "configs" / "experiments" / "fixture" / "efalls_retrained_lasso.yaml"
    with pytest.raises(DatasetValidationError, match="different feature spec"):
        cli_main(["validate-dataset", "--dataset", str(tmp_path / "reduced"), "--features-from-config", str(retrained)])
    with pytest.raises(SystemExit, match="do not combine"):
        cli_main(["validate-dataset", "--dataset", str(tmp_path / "reduced"), "--features-from-config", str(REDUCED_CONFIG),
                  "--feature-spec", str(SPEC_PATH)])


# ============================================================================ make-fixture
def test_make_fixture_restricted_to_config_features(tmp_path, capsys, spec):
    features = reduced_fixture_features()
    out = tmp_path / "fixture"
    manifest = _run(capsys, "make-fixture", "--out", str(out), "--n-patients", "120", "--features-from-config", str(REDUCED_CONFIG))
    subset = spec.subset(features)
    assert manifest["feature_spec_sha256"] == subset.content_sha256 and manifest["source"] == "synthetic_fixture"
    frame = pd.read_parquet(out / "modeling_dataset.parquet")
    assert not set(spec.predictor_names()) - set(features) & set(frame.columns) and set(features) <= set(frame.columns)
    full = tmp_path / "full"
    generate_synthetic_modeling_dataset(full, n_patients=120, feature_spec_path=SPEC_PATH)
    assert len(frame) == len(pd.read_parquet(full / "modeling_dataset.parquet")), "same rows as the unrestricted fixture"
    model = json.loads((out / "synthetic_generating_model.json").read_text(encoding="utf-8"))
    assert model["written_feature_subset"]["n_predictors"] == 66


def test_make_fixture_refuses_invalid_reduced_declaration(tmp_path, spec):
    raw = yaml.safe_load(REDUCED_CONFIG.read_text(encoding="utf-8"))
    raw["preprocessing"]["features"] = spec.predictor_names()
    raw["dataset"]["feature_spec"] = str(SPEC_PATH)
    config = tmp_path / "all78.yaml"
    config.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    with pytest.raises(ConfigError, match="use experiment kind efalls_retrained"):
        cli_main(["make-fixture", "--out", str(tmp_path / "fixture"), "--n-patients", "50", "--features-from-config", str(config)])
    assert not (tmp_path / "fixture").exists()
