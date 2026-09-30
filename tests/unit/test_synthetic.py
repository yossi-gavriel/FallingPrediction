"""Deterministic SYNTHETIC fixtures (spec §13.2). SYNTHETIC — NOT SCIENTIFIC EVIDENCE."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from falls_ml.data.dataset import ModelingDataset
from falls_ml.data.synthetic import (
    COEFFICIENT_RANGES,
    GENERATING_MODEL_FILE,
    SYNTHETIC_LABEL,
    generate_synthetic_event_tables,
    generate_synthetic_modeling_dataset,
)
from falls_ml.dataeng.derive import DerivationRules, derive_modeling_frame
from falls_ml.errors import ConfigError
from falls_ml.features.spec import load_feature_spec

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "configs" / "features" / "efalls_v1.yaml"
EXTENSION = ROOT / "configs" / "features" / "meuhedet_enhanced_example.yaml"
SPEC = load_feature_spec(BASE)
INDEX = pd.Timestamp("2018-04-01")


@pytest.fixture(scope="module")
def fixture_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("synthetic_default")
    generate_synthetic_modeling_dataset(directory)
    return directory


@pytest.fixture(scope="module")
def raw_tables() -> dict[str, pd.DataFrame]:
    return generate_synthetic_event_tables(400, seed=11, spec=SPEC)


def test_regeneration_is_byte_identical(fixture_dir: Path, tmp_path: Path) -> None:
    again = generate_synthetic_modeling_dataset(tmp_path / "again")
    first = json.loads((fixture_dir / "manifest.json").read_text(encoding="utf-8"))
    assert again.data_sha256 == first["data_sha256"]
    assert (tmp_path / "again" / GENERATING_MODEL_FILE).read_bytes() == (fixture_dir / GENERATING_MODEL_FILE).read_bytes()
    other = generate_synthetic_modeling_dataset(tmp_path / "other_seed", n_patients=300, seed=7)
    assert other.data_sha256 != again.data_sha256


def test_loads_with_watermark_multiple_index_dates_and_plausible_prevalence(fixture_dir: Path) -> None:
    ds = ModelingDataset.load(fixture_dir, SPEC)
    m = ds.manifest
    assert m.source == "synthetic_fixture" and m.scientific_use_allowed is False and m.notes == SYNTHETIC_LABEL
    assert m.created_utc == "2026-09-14T00:00:00+00:00" and m.dataset_version == "synthetic-1.0.0"
    frame = ds.frame
    assert sorted(frame["index_date"].dt.strftime("%Y-%m-%d").unique()) == ["2018-04-01", "2019-04-01", "2020-04-01", "2021-04-01"]
    assert frame.groupby("research_id")["index_date"].nunique().max() == 4
    assert 0.03 <= m.outcome_prevalence <= 0.20
    assert set(frame["deprivation_group"].unique()) == {"1", "2", "3", "4", "5", "missing"}
    assert frame["practice_id"].nunique() == 20 and frame["site_id"].nunique() == 4
    assert frame["bmi_value"].isna().any() and frame["alcohol_category"].isna().any()
    assert frame["death_date"].notna().any()


def test_generating_model_is_arbitrary_and_differs_from_published(fixture_dir: Path) -> None:
    model = json.loads((fixture_dir / GENERATING_MODEL_FILE).read_text(encoding="utf-8"))
    assert model["label"] == SYNTHETIC_LABEL and "NOT tuned" in model["purpose"]
    assert 1 <= len(model["binary_coefficients"]) <= 10  # a few strong effects only
    for name, value in [*model["terms"].items(), *(("binary", v) for v in model["binary_coefficients"].values())]:
        lo, hi = COEFFICIENT_RANGES[name]
        assert lo <= value <= hi, name
    probs = np.array(list(model["condition_onset_probability"].values()))
    assert len(probs) == 72 and probs.min() >= 0.01 and probs.max() <= 0.25


def test_coefficient_ranges_exclude_every_published_counterpart_for_any_seed() -> None:
    """Spec §13.2: 'deliberately different from Table S3.2' must hold structurally, not only for the default seed."""
    root = yaml.safe_load((ROOT / "configs" / "models" / "efalls_published.yaml").read_text(encoding="utf-8"))
    pub, sex = root["terms"], root["sex_parameterisations"]["lp_c_box_s3_1"]
    counterparts = {
        "age_per_decade_from_75": [10 * pub["age_years"]["coefficient"]],
        "male": [abs(sex["intercept_male"] - sex["intercept_female"])],
        "bmi_missing": [pub["bmi_category"]["levels"]["missing"]],
        "bmi_at_least_30": [pub["bmi_category"]["levels"]["obese"]],
        "smoking_current": [pub["smoking"]["levels"]["current"]],
        "alcohol_higher_risk_or_harmful": [pub["alcohol_category"]["levels"][k] for k in ("harmful", "higher_risk")],
        "binary": list(pub["binary"].values()),
    }
    for name, values in counterparts.items():
        lo, hi = COEFFICIENT_RANGES[name]
        assert all(v < lo - 0.15 or v > hi + 0.15 for v in values), (name, values)
    assert set(COEFFICIENT_RANGES) - set(counterparts) == {"polypharmacy_count_per_paragraph"}  # raw count, not log term


def test_raw_tables_contain_leakage_traps(raw_tables: dict[str, pd.DataFrame]) -> None:
    cond = raw_tables["conditions"]
    assert (cond["record_date"] >= INDEX).any()  # post-index records
    backfilled = (cond["record_date"] < INDEX) & (cond["available_date"] >= INDEX)
    assert backfilled.any()
    bmi = raw_tables["measurements"].query("kind == 'bmi'")["value"]
    assert ((bmi < 10) | (bmi > 80)).any()
    rx = raw_tables["prescriptions"]
    assert rx["bnf_chapter"].isin([20, 21, 22, 23]).any() and (~rx["is_drug"]).any()
    enc = raw_tables["encounters"]
    assert (enc["admission_method"] == "elective").any()
    assert (enc["spell_start_date"] < enc["event_date"]).any()  # continuation spells
    assert raw_tables["persons"]["research_id"].duplicated().any()  # membership gaps
    assert set(raw_tables["lifestyle"]["kind"]) == {"smoking", "alcohol", "alcohol_units_week"}


def test_derived_outcome_equals_generated_outcome(raw_tables: dict[str, pd.DataFrame]) -> None:
    frame, dlog = derive_modeling_frame(raw_tables, raw_tables["index_table"], SPEC, DerivationRules())
    truth = raw_tables["synthetic_truth"]
    merged = frame.merge(truth, on=["research_id", "index_date"], suffixes=("", "_true"), validate="one_to_one")
    assert len(merged) == len(frame) == len(truth)
    assert (merged["outcome_12m"] == merged["outcome_12m_true"]).all()
    assert merged["outcome_first_event_date"].equals(merged["outcome_first_event_date_true"])
    audit = dlog.outcome_audit.groupby("status")["n_records"].sum()
    assert {"qualifying", "admission_not_non_elective", "spell_started_before_index"} <= set(audit.index)


def test_extension_features_get_synthetic_signal_and_load(tmp_path: Path) -> None:
    manifest = generate_synthetic_modeling_dataset(tmp_path, n_patients=400, extension_spec_paths=(EXTENSION,))
    spec = load_feature_spec(BASE, [EXTENSION])
    frame = ModelingDataset.load(tmp_path, spec).frame
    assert manifest.feature_spec_sha256 == spec.content_sha256
    assert frame["mini_cog_score"].between(0, 5).any() and frame["mini_cog_score"].isna().any()
    assert set(frame["nurse_fall_risk_high"].unique()) == {0, 1}
    model = json.loads((tmp_path / GENERATING_MODEL_FILE).read_text(encoding="utf-8"))
    assert set(model["extension_coefficients"]) == {"mini_cog_score", "nurse_fall_risk_high", "uses_walking_aid"}


def test_overlapping_outcome_windows_are_rejected() -> None:
    with pytest.raises(ConfigError, match="non-overlapping"):
        generate_synthetic_event_tables(50, seed=1, spec=SPEC, index_dates=("2018-04-01", "2018-10-01"))
