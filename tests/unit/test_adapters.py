"""External-dataset adapter: rename and recode only (spec §13.1)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from falls_ml.data.adapters import ExternalDatasetAdapter
from falls_ml.data.dataset import ModelingDataset
from falls_ml.data.schema import VERSION_COLUMNS, validate_modeling_dataset
from falls_ml.data.synthetic import generate_synthetic_modeling_dataset
from falls_ml.errors import ConfigError, DatasetValidationError
from falls_ml.features.spec import load_feature_spec

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "configs" / "adapters" / "example_external_dataset.yaml"
SPEC = load_feature_spec(ROOT / "configs" / "features" / "efalls_v1.yaml")


def small_adapter(**overrides) -> ExternalDatasetAdapter:
    kwargs = dict(name="t", source="unit", version="1", column_map={"ID": "research_id", "SEX": "sex", "DT": "index_date",
                                                                      "FLAG": "falls"},
                  recode_map={"sex": {"1": "male", "2": "female", "9": None}, "falls": {"Y": 1, "N": 0}},
                  drop_columns=("NOISE",), date_columns=("index_date",))
    kwargs.update(overrides)
    return ExternalDatasetAdapter(**kwargs)


def source_frame() -> pd.DataFrame:
    return pd.DataFrame({"ID": ["a", "b", "c"], "SEX": [1.0, 2.0, np.nan], "DT": ["2018-04-01", "2019-04-01", None],
                         "FLAG": ["Y", "N", "N"], "NOISE": [0, 0, 0]})


def test_apply_renames_parses_dates_and_recodes() -> None:
    out = small_adapter().apply(source_frame())
    assert list(out.columns) == ["research_id", "sex", "index_date", "falls"]
    assert out["sex"].tolist()[:2] == ["male", "female"] and pd.isna(out["sex"].iloc[2])
    assert pd.api.types.is_datetime64_any_dtype(out["index_date"]) and pd.isna(out["index_date"].iloc[2])
    assert out["falls"].dtype == "int64" and out["falls"].tolist() == [1, 0, 0]


@pytest.mark.parametrize(("mutate", "match"), [
    (lambda df: df.assign(EXTRA=1), "unmapped source columns"),
    (lambda df: df.drop(columns="FLAG"), "absent from the data"),
    (lambda df: df.drop(columns="NOISE"), "absent from the data"),
    (lambda df: df.assign(SEX=[1, 3, 4]), r"unmapped values \['3', '4'\]"),
    (lambda df: df.assign(DT=["2018-04-01", "not a date", None]), "not parseable as dates"),
])
def test_apply_fails_loudly_on_mismatches(mutate, match) -> None:
    with pytest.raises(DatasetValidationError, match=match):
        small_adapter().apply(mutate(source_frame()))


@pytest.mark.parametrize("overrides", [
    {"column_map": {"A": "sex", "B": "sex"}, "recode_map": {}, "date_columns": (), "drop_columns": ()},
    {"recode_map": {"age_years": {"1": 1}}},
    {"date_columns": ("sex",)},
    {"drop_columns": ("ID",)},
    {"recode_map": {"falls": {"Y": 1, "N": "no"}}},
])
def test_invalid_declarations_raise_config_error(overrides) -> None:
    with pytest.raises(ConfigError):
        small_adapter(**overrides)


def test_fingerprint_is_stable_and_version_sensitive() -> None:
    assert small_adapter().sha256() == small_adapter().sha256()
    assert small_adapter().sha256() != small_adapter(version="2").sha256()


def test_from_yaml_rejects_unknown_keys(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("adapter: {name: x, source: y, version: '1'}\ncolumn_map: {A: research_id}\nrecodes: {}\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown keys"):
        ExternalDatasetAdapter.from_yaml(bad)


def test_example_config_maps_every_modelling_column() -> None:
    adapter = ExternalDatasetAdapter.from_yaml(EXAMPLE)
    expected = {*SPEC.identifier_columns, SPEC.index_column, SPEC.outcome.name, "predictor_max_record_date",
                "outcome_first_event_date", *SPEC.predictor_names(), *SPEC.metadata_columns}
    assert set(adapter.column_map.values()) == expected
    assert "example" in adapter.name and adapter.version == "0.0.1"


def test_example_adapter_round_trips_a_sail_like_extract(tmp_path: Path) -> None:
    generate_synthetic_modeling_dataset(tmp_path, n_patients=200)
    original = ModelingDataset.load(tmp_path, SPEC).frame.drop(columns=list(VERSION_COLUMNS))
    adapter = ExternalDatasetAdapter.from_yaml(EXAMPLE)
    inverse_columns = {v: k for k, v in adapter.column_map.items()}
    source = original.copy()
    for column, mapping in adapter.recode_map.items():
        inverse = {v: k for k, v in mapping.items() if v is not None}
        source[column] = source[column].map(lambda v, inv=inverse: None if pd.isna(v) else inv[v]).astype(object)
    for column in adapter.date_columns:
        source[column] = source[column].dt.strftime("%Y-%m-%d")
    source = source.rename(columns=inverse_columns).assign(WOB="1950-01-01", LSOA2011_CD="W01000001")

    adapted = adapter.apply(source)
    pd.testing.assert_frame_equal(adapted[original.columns], original, check_dtype=False)
    validate_modeling_dataset(adapted.assign(outcome_12m=adapted["outcome_12m"].astype("int8"), dataset_version="x",
                                             mapping_version=f"{adapter.name}-{adapter.version}", source=adapter.source), SPEC)
