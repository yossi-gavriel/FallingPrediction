"""Meuhedet wide-table adapter (Phase 1): contract, mapping manifest, adapter rules (NULL never silently zero, D-00 timing,
cohort filter, exploratory outcome), aggregate audit privacy, CLI commands and the baseline freeze. SYNTHETIC data only."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import replace
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from falls_ml.cli import main as cli_main
from falls_ml.config import check_feature_declaration, load_experiment_config
from falls_ml.data.dataset import ModelingDataset
from falls_ml.data.meuhedet_audit import SUPPRESSED, audit_wide_extract, write_audit
from falls_ml.data.meuhedet_synthetic import generate_synthetic_wide_extract, write_synthetic_wide_extract
from falls_ml.data.meuhedet_wide import (DEFAULT_CONTRACT, DEFAULT_MAPPING, MeuhedetWideDatasetAdapter, build_meuhedet_dataset,
                                         column_inventory_table, load_wide_contract, load_wide_mapping, mapping_table,
                                         read_wide_extract, validate_wide_contract)
from falls_ml.errors import ConfigError, DatasetValidationError
from falls_ml.features.spec import load_feature_spec, load_feature_spec_from_payloads

ROOT = Path(__file__).resolve().parents[2]
EFALLS_SPEC = ROOT / "configs" / "features" / "efalls_v1.yaml"
EXPLORATORY_SPEC = ROOT / "configs" / "features" / "efalls_v1__outcome_fall_next_180d_exploratory.yaml"
REAL_CONFIG = ROOT / "configs" / "experiments" / "meuhedet" / "phase1_180d_exploratory_efalls_reduced.yaml"
FIXTURE_CONFIG = ROOT / "configs" / "experiments" / "fixture" / "meuhedet_180d_exploratory_efalls_reduced.yaml"
INDEX_DATE = "2025-01-01"


@pytest.fixture(scope="module")
def contract():
    return load_wide_contract(DEFAULT_CONTRACT)


@pytest.fixture(scope="module")
def spec():
    return load_feature_spec(EFALLS_SPEC)


@pytest.fixture(scope="module")
def exploratory_spec():
    return load_feature_spec(EXPLORATORY_SPEC)


@pytest.fixture(scope="module")
def mapping(spec, contract):
    return load_wide_mapping(DEFAULT_MAPPING, spec=spec, contract=contract)


@pytest.fixture(scope="module")
def wide(contract) -> pd.DataFrame:
    return generate_synthetic_wide_extract(1500, seed=7, contract=contract)


@pytest.fixture(scope="module")
def adapter(mapping, contract, exploratory_spec):
    return MeuhedetWideDatasetAdapter(mapping, contract, exploratory_spec)


# ------------------------------------------------------------------ contract (column inventory / datatype contract)
def test_contract_covers_all_221_columns_with_one_role_each(contract) -> None:
    assert len(contract.columns) == 221 and len(set(contract.names)) == 221
    roles = {c.role for c in contract.columns}
    assert roles == {"IDENTIFIER", "COHORT_ELIGIBILITY", "QA_CONTROL", "EFALLS_BASELINE_FEATURE", "MEUHEDET_ENHANCED_FEATURE", "LABEL", "FORBIDDEN_LEAKAGE"}
    for c in contract.columns:
        if c.role not in {"EFALLS_BASELINE_FEATURE", "MEUHEDET_ENHANCED_FEATURE"}:
            assert c.model_dtype == "never", c.name
        if c.role == "MEUHEDET_ENHANCED_FEATURE":
            assert c.group
    # identifiers stay strings and never enter a model; labels and forbidden columns never do either
    assert contract.get("Customer_Full_ID").pandas_dtype == "string" and not contract.get("Customer_Full_ID").predictor_allowed
    assert all(not contract.get(n).predictor_allowed for n in ("Fall_Next_180D_Ind", "Audit_Only_Adif_Status", "New_Registry_30D",
                                                                  "External_Care_Count_365D", "Followup_End_Date", "Is_Deceased_Ind"))
    # unknown timing blocks a predictor until confirmed
    assert not contract.get("Hospitalization_Count_365D").predictor_allowed
    assert contract.get("Age_At_Index").pandas_dtype == "Int64" and contract.get("Age_At_Index").model_dtype == "float64"


def test_contract_loader_rejects_bad_rules(tmp_path) -> None:
    raw = yaml.safe_load(Path(DEFAULT_CONTRACT).read_text(encoding="utf-8"))
    raw["columns"]["Fall_Next_180D_Ind"]["model"] = "int8_binary"  # a label entering the model matrix
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="can never enter a model matrix"):
        load_wide_contract(p)


# ------------------------------------------------------------------ mapping manifest
def test_mapping_denominator_is_derived_from_the_spec_and_coverage_is_explicit(mapping, spec, exploratory_spec) -> None:
    cov = mapping.coverage(spec)
    assert cov["n_expected"] == sum(f.exact_efalls_baseline for f in spec.features) == len(mapping.features)
    assert cov["n_available"] == cov["n_exact"] + cov["n_high_confidence"] + cov["n_approximate"] == 15
    assert cov["n_exact"] == 0 and cov["n_unavailable"] == cov["n_expected"] - 15
    assert cov["mandatory_unavailable"] == ["fracture", "fragility_fracture"]
    assert "NOT exact eFalls reproduction" in cov["label"]
    assert mapping.coverage(exploratory_spec)["n_expected"] == cov["n_expected"]
    assert mapping.clinically_validated is False and "PROPOSED" in mapping.status
    # every row carries the eFalls definition and the three types
    table = mapping_table(mapping)
    assert len(table) == cov["n_expected"]
    assert table["efalls_definition"].notna().all()
    inc = table[table["include_in_baseline"]]
    assert (inc["raw_dtype"] != "").all() and (inc["canonical_dtype"] != "").all() and (inc["model_dtype"] != "").all()


def test_mapping_loader_rejects_silent_equivalence_and_role_violations(tmp_path, spec, contract) -> None:
    raw = yaml.safe_load(Path(DEFAULT_MAPPING).read_text(encoding="utf-8"))
    bad = tmp_path / "m.yaml"
    # a nurse-assessed column (MEUHEDET_ENHANCED_FEATURE) cannot back an eFalls baseline binary
    raw["features"]["visual_impairment"] = {"mapping_status": "APPROXIMATE", "mapping_quality": "APPROXIMATE", "op": "any_positive",
                                            "source_columns": ["Vision_Impairment_Nurse_Ind"], "missing_rule": "absent_is_zero", "include_in_baseline": True}
    bad.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="Vision_Impairment_Nurse_Ind has role MEUHEDET_ENHANCED_FEATURE"):
        load_wide_mapping(bad, spec=spec, contract=contract)
    raw = yaml.safe_load(Path(DEFAULT_MAPPING).read_text(encoding="utf-8"))
    raw["features"]["polypharmacy_count_120d"].pop("absent_indicator")
    bad.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="requires absent_indicator"):
        load_wide_mapping(bad, spec=spec, contract=contract)
    raw = yaml.safe_load(Path(DEFAULT_MAPPING).read_text(encoding="utf-8"))
    del raw["features"]["weight_loss"]
    bad.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="without a manifest row: \\['weight_loss'\\]"):
        load_wide_mapping(bad, spec=spec, contract=contract)


def test_experiment_configs_declare_exactly_the_mapping_baseline(mapping) -> None:
    for path in (REAL_CONFIG, FIXTURE_CONFIG):
        cfg = load_experiment_config(path)
        spec = load_feature_spec(cfg.dataset.feature_spec)
        check_feature_declaration(cfg, spec)
        assert cfg.experiment.kind == "efalls_retrained_reduced"
        assert list(cfg.preprocessing.features) == mapping.baseline_features
        assert "NOT an eFalls reproduction" in cfg.experiment.description
        assert cfg.validation.strategy == "patient_grouped_random" and cfg.validation.limitation_note
    assert load_experiment_config(REAL_CONFIG).dataset.require_scientific_use is True


# ------------------------------------------------------------------ exploratory outcome spec
def test_exploratory_spec_keeps_the_78_predictors_and_declares_a_day_window(spec, exploratory_spec) -> None:
    assert [f.raw for f in exploratory_spec.features] == [f.raw for f in spec.features]
    assert exploratory_spec.outcome.name == "fall_next_180d" and exploratory_spec.outcome.horizon_days == 181
    assert not exploratory_spec.outcome.is_published_efalls and spec.outcome.is_published_efalls
    assert exploratory_spec.outcome.window_end(pd.Timestamp(INDEX_DATE)) == pd.Timestamp("2025-06-30")
    assert spec.outcome.window_end(pd.Timestamp(INDEX_DATE)) == pd.Timestamp("2025-12-31")
    assert exploratory_spec.content_sha256 != spec.content_sha256 and exploratory_spec.is_pure_efalls
    check = subprocess.run([sys.executable, str(ROOT / "tools" / "generate_exploratory_outcome_spec.py"), "--check"], capture_output=True, text=True, encoding="utf-8")
    assert check.returncode == 0, check.stdout + check.stderr


def test_spec_loader_rejects_ambiguous_or_published_day_windows(spec) -> None:
    raw = yaml.safe_load(EXPLORATORY_SPEC.read_text(encoding="utf-8"))
    raw["outcome"]["layer"] = "L1_published"
    with pytest.raises(ConfigError, match="cannot be tagged L1_published"):
        load_feature_spec_from_payloads(raw)
    raw = yaml.safe_load(EXPLORATORY_SPEC.read_text(encoding="utf-8"))
    raw["outcome"]["window"]["horizon_years"] = 1
    with pytest.raises(ConfigError, match="exactly one of"):
        load_feature_spec_from_payloads(raw)


# ------------------------------------------------------------------ synthetic extract and reading
def test_synthetic_extract_matches_the_contract_and_csv_round_trips(wide, contract, tmp_path) -> None:
    assert list(wide.columns) == contract.names
    for c in contract.columns:
        assert str(wide[c.name].dtype) == c.pandas_dtype, c.name
    rep = validate_wide_contract(wide, contract, strict=True)
    assert rep.wrong_type_total == 0 and rep.problems == []
    csv = write_synthetic_wide_extract(tmp_path / "w.csv", n_rows=300, seed=3)
    pq = write_synthetic_wide_extract(tmp_path / "w.parquet", n_rows=300, seed=3)
    assert "NULL" in csv.read_text(encoding="utf-8")
    a, wa = read_wide_extract(csv, contract)
    b, wb = read_wide_extract(pq, contract)
    assert wa == {} and wb == {}
    pd.testing.assert_frame_equal(a, b)


def test_reading_reports_values_that_do_not_fit_the_contract_type(wide, contract, tmp_path) -> None:
    df = wide.head(50).copy()
    df["Age_At_Index"] = df["Age_At_Index"].astype("string")
    df.loc[df.index[0], "Age_At_Index"] = "seventy"
    df.loc[df.index[1], "Age_At_Index"] = "70.5"
    csv = tmp_path / "bad.csv"
    df.to_csv(csv, index=False, na_rep="NULL")
    out, wrong = read_wide_extract(csv, contract)
    assert wrong == {"Age_At_Index": 2} and out["Age_At_Index"].isna().sum() == 2
    rep = validate_wide_contract(out, contract, wrong_type=wrong, strict_columns=["Age_At_Index"], strict=False)
    assert any("Age_At_Index: 2 values cannot be represented as int" in p for p in rep.problems)


# ------------------------------------------------------------------ adapter
def test_adapter_builds_the_canonical_frame_with_explicit_counts(adapter, wide, mapping, exploratory_spec) -> None:
    frame, rep = adapter.apply(wide, index_date=INDEX_DATE)
    assert rep.n_rows_input == len(wide) and rep.synthetic
    real_like = wide.copy()
    real_like["Snapshot_Key"] = real_like["Snapshot_Key"].str.replace("SYN_", "", regex=False)
    assert adapter.apply(real_like, index_date=INDEX_DATE)[1].synthetic is False
    assert rep.n_rows_other_index_dates + rep.n_rows_ineligible + rep.n_rows_label_null + rep.n_rows_final == len(wide)
    assert rep.n_rows_timing_violation == 0 and rep.n_patients_final == rep.n_rows_final
    expected = {"research_id", "index_date", "predictor_max_record_date", "fall_next_180d", "outcome_first_event_date", "death_date",
                "followup_end_date", *mapping.baseline_features}
    assert set(frame.columns) == expected
    assert frame["research_id"].dtype == "string" and frame["index_date"].dt.normalize().eq(pd.Timestamp(INDEX_DATE)).all()
    assert (frame["predictor_max_record_date"] < frame["index_date"]).all()
    assert set(frame["sex"].unique()) == {"male", "female"} and rep.sex_code_descriptions == {"1": ["זכר"], "2": ["נקבה"]}
    for name in ("falls", "dementia", "diabetes_mellitus", "severe_mental_illness"):
        assert frame[name].dtype == "int8" and set(frame[name].unique()) <= {0, 1}
    assert frame["polypharmacy_count_120d"].dtype == "int64" and frame["age_years"].dtype == "float64"
    # NULL -> 0 only via the declared eFalls rule and always counted
    eligible = wide[(wide["Index_Date"] == pd.Timestamp(INDEX_DATE)) & (wide["Is_Eligible_Cohort"] == 1) & wide["Fall_Next_180D_Ind"].notna()]
    n_med_null = int((eligible["Distinct_Active_Substance_Count"].isna() & (eligible["Medication_Missing_Ind"] == 1)).sum())
    assert rep.null_to_zero["polypharmacy_count_120d"] == n_med_null > 0
    assert rep.null_to_zero["severe_mental_illness"] == int(eligible["Registry_SMI_Level"].isna().sum())
    assert rep.source_absent_rows["dementia"] == int((eligible["Registry_Missing_Ind"] == 1).sum())
    assert (frame["diabetes_mellitus"].to_numpy() == ((eligible["Registry_Diabetes_T1_Ind"] > 0) | (eligible["Registry_Diabetes_T2_Ind"] > 0)).to_numpy()).all()
    assert rep.coverage["n_available"] == 15 and list(rep.outcome_window_days_observed) == ["180"]


def _eligible_rows(wide: pd.DataFrame) -> pd.Index:
    return wide.index[(wide["Index_Date"] == pd.Timestamp(INDEX_DATE)) & (wide["Is_Eligible_Cohort"] == 1) & wide["Fall_Next_180D_Ind"].notna()]


def test_adapter_never_converts_null_to_zero_without_a_declared_rule(adapter, wide) -> None:
    df = wide.copy()
    rows = _eligible_rows(df)[:3]
    df.loc[rows, "Distinct_Active_Substance_Count"] = pd.NA  # NULL count while Medication_Missing_Ind == 0
    df.loc[rows, "Medication_Missing_Ind"] = 0
    with pytest.raises(DatasetValidationError) as exc:
        adapter.apply(df, index_date=INDEX_DATE)
    assert any("polypharmacy_count_120d: 3 NULL values" in p and "not explained by Medication_Missing_Ind" in p for p in exc.value.problems)
    df = wide.copy()
    df.loc[rows, "Registry_Dementia_Ind"] = pd.NA
    with pytest.raises(DatasetValidationError) as exc:
        adapter.apply(df, index_date=INDEX_DATE)
    assert any("DDL NOT NULL" in p for p in exc.value.problems)


def test_adapter_rejects_unknown_sex_codes_view_leakage_flags_and_window_mismatch(adapter, wide) -> None:
    rows = _eligible_rows(wide)[:2]
    df = wide.copy()
    df.loc[rows, "Gender_Code"] = 3
    with pytest.raises(DatasetValidationError) as exc:
        adapter.apply(df, index_date=INDEX_DATE)
    assert any("Gender_Code: 2 values outside allowed" in p for p in exc.value.problems)
    df = wide.copy()
    df.loc[rows, "Leakage_Check_Ind"] = 1
    with pytest.raises(DatasetValidationError, match="contract"):
        adapter.apply(df, index_date=INDEX_DATE)
    df = wide.copy()
    df["Label_End_180D"] = df["Index_Date"] + pd.Timedelta(days=179)
    with pytest.raises(DatasetValidationError) as exc:
        adapter.apply(df, index_date=INDEX_DATE)
    assert any("window_days_including_index" in p for p in exc.value.problems)
    df = wide.copy().drop(columns=["Registry_COPD_Ind"])
    with pytest.raises(DatasetValidationError, match="missing contract columns"):
        adapter.apply(df, index_date=INDEX_DATE)
    df = wide.copy()
    df["Fall_Next_365D_Ind"] = 0
    with pytest.raises(DatasetValidationError, match="not in the contract"):
        adapter.apply(df, index_date=INDEX_DATE)


def test_adapter_enforces_d00_timing_and_counts_dropped_rows(mapping, contract, exploratory_spec) -> None:
    df = generate_synthetic_wide_extract(600, seed=11, contract=contract, n_index_day_falls=4)
    adapter = MeuhedetWideDatasetAdapter(mapping, contract, exploratory_spec)
    with pytest.raises(DatasetValidationError) as exc:
        adapter.apply(df, index_date=INDEX_DATE)
    assert any("D-00 timing: 4 rows" in p and "Last_Fall_Date" in p for p in exc.value.problems)
    frame, rep = adapter.apply(df, index_date=INDEX_DATE, index_day_records="drop_rows")
    assert rep.n_rows_timing_violation == 4 and rep.timing_violations_by_column == {"Last_Fall_Date": 4}
    assert (frame["predictor_max_record_date"] < frame["index_date"]).all()
    assert any("4 rows dropped" in w for w in rep.warnings)


def test_adapter_requires_the_exploratory_spec(mapping, contract, spec) -> None:
    with pytest.raises(ConfigError, match="not the mapping outcome"):
        MeuhedetWideDatasetAdapter(mapping, contract, spec)


# ------------------------------------------------------------------ dates: the full SQL date range, declared sentinels, malformed values
def _csv_with_dates(wide: pd.DataFrame, path: Path, column: str, values: list) -> Path:
    df = wide.head(len(values)).copy()
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = df[c].dt.strftime("%Y-%m-%d")
    df[column] = pd.Series(values, index=df.index, dtype="string")
    df.to_csv(path, index=False, na_rep="NULL")
    return path


def test_date_columns_keep_the_full_sql_date_range_and_report_offending_values(wide, contract, tmp_path) -> None:
    from falls_ml.data.meuhedet_wide import DATE_DTYPE, read_wide_extract_report

    assert DATE_DTYPE == "datetime64[us]" and contract.get("Index_Date").pandas_dtype == DATE_DTYPE
    scd = contract.get("Population_Record_To_Date")
    assert scd.sentinels == ("2999-12-31",) and scd.sentinel_means and scd.sentinel_modeling_rule == "keep"
    values = ["2999-12-31", "2025-03-01", None, "2025-13-45", "31/12/2999", "9999-12-31", "2025-01-01 00:00:00.000"]  # 31/12/2999 = the same sentinel, DD/MM
    csv = _csv_with_dates(wide, tmp_path / "dates.csv", "Population_Record_To_Date", values)
    r = read_wide_extract_report(csv, contract)
    col = r.frame["Population_Record_To_Date"]
    assert str(col.dtype) == DATE_DTYPE
    assert col.iloc[0] == pd.Timestamp("2999-12-31") and col.iloc[1] == pd.Timestamp("2025-03-01") and col.iloc[5] == pd.Timestamp("9999-12-31")
    assert col.iloc[6] == pd.Timestamp("2025-01-01") and pd.isna(col.iloc[2]) and pd.isna(col.iloc[3])
    assert col.iloc[4] == pd.Timestamp("2999-12-31")  # 31/12/2999 read with the declared %d/%m/%Y layout
    assert r.wrong_type == {"Population_Record_To_Date": 1}
    assert r.wrong_type_examples == {"Population_Record_To_Date": ["2025-13-45"]}
    assert r.date_formats["Population_Record_To_Date"] == {"%Y-%m-%d": 4, "%d/%m/%Y": 1}
    dr = r.date_range["Population_Record_To_Date"]
    assert dr["n_beyond_ns_range"] == 3 and dr["declared_sentinels"] == {"2999-12-31": 2} and dr["undeclared_beyond_ns_range_values"] == {"9999-12-31": 1}
    assert dr["min"] == "2025-01-01" and dr["max"] == "9999-12-31" and dr["sentinel_modeling_rule"] == "keep"
    # day arithmetic is exact across the whole range and unchanged around the index date
    assert (col.iloc[0] - pd.Timestamp(INDEX_DATE)).days == (date(2999, 12, 31) - date(2025, 1, 1)).days
    assert ((r.frame["Label_End_180D"] - r.frame["Index_Date"]).dt.days == 180).all()
    # the report names column, offending value and expected semantic type - never an identifier
    rep = validate_wide_contract(r.frame, contract, wrong_type=r.wrong_type, wrong_type_examples=r.wrong_type_examples,
                                 strict_columns=["Population_Record_To_Date"], strict=False)
    text = "\n".join(rep.problems + rep.warnings)
    assert "Population_Record_To_Date: 1 values cannot be represented as date" in text and "2025-13-45" in text
    assert "expected one of the declared source date formats ['%Y-%m-%d', '%d/%m/%Y']" in text
    assert "9999-12-31" in text and "not declared sentinels" in text
    assert not any(cid in text for cid in r.frame["Customer_Full_ID"])
    assert rep.date_range["Population_Record_To_Date"]["declared_sentinels"] == {"2999-12-31": 2}
    assert rep.to_dict()["wrong_type_examples"] == {"Population_Record_To_Date": ["2025-13-45"]}


def test_excel_dates_are_parsed_from_cells_not_inferred(wide, contract, tmp_path) -> None:
    from falls_ml.data.meuhedet_wide import read_wide_extract_report

    df = wide.head(5).copy()
    for c in df.columns:
        if str(df[c].dtype) == "Int64":
            df[c] = df[c].astype(object).where(df[c].notna(), None)
    df["Population_Record_To_Date"] = pd.Series([datetime(2999, 12, 31), datetime(2025, 3, 1), None, "not a date", date(2025, 6, 30)],
                                                index=df.index, dtype=object)
    xlsx = tmp_path / "dates.xlsx"
    df.to_excel(xlsx, index=False, engine="openpyxl")
    r = read_wide_extract_report(xlsx, contract)
    col = r.frame["Population_Record_To_Date"]
    assert str(col.dtype) == "datetime64[us]" and col.iloc[0] == pd.Timestamp("2999-12-31") and col.iloc[4] == pd.Timestamp("2025-06-30")
    assert pd.isna(col.iloc[2]) and pd.isna(col.iloc[3]) and r.wrong_type == {"Population_Record_To_Date": 1}
    assert r.wrong_type_examples == {"Population_Record_To_Date": ["not a date"]} and "datetime" in r.cell_types["Population_Record_To_Date"]
    assert r.date_range["Population_Record_To_Date"]["declared_sentinels"] == {"2999-12-31": 1}


def test_adapter_keeps_declared_sentinels_and_stops_on_far_future_dates_in_modelling_columns(adapter, wide) -> None:
    assert (wide["Population_Record_To_Date"] == pd.Timestamp("2999-12-31")).sum() > 0  # the fixture carries the declared SCD sentinel
    frame, rep = adapter.apply(wide, index_date=INDEX_DATE)
    for col in ("index_date", "predictor_max_record_date", "outcome_first_event_date", "death_date", "followup_end_date"):
        assert str(frame[col].dtype) == "datetime64[ns]", col
    assert rep.contract["date_range"]["Population_Record_To_Date"]["declared_sentinels"]["2999-12-31"] > 0
    assert set(rep.sentinel_to_null) == {"followup_end_date"}   # only the column whose contract rule says to_null; the SCD date keeps its value
    assert not any("Population_Record_To_Date" in w for w in rep.warnings)
    assert (frame["index_date"] - frame["predictor_max_record_date"]).dt.days.min() >= 1
    ev = frame["outcome_first_event_date"].dropna()
    assert (ev - frame.loc[ev.index, "index_date"]).dt.days.between(0, 180).all()
    rows = _eligible_rows(wide)[:3]
    ids = list(wide.loc[rows, "Customer_Full_ID"])
    # an undeclared far-future date in a column the build maps -> stop, naming column / value / expected type only
    df = wide.copy()
    df.loc[rows, "Death_Censor_Date"] = pd.Timestamp("2999-12-31")   # Death_Censor_Date declares no sentinel
    with pytest.raises(DatasetValidationError) as exc:
        adapter.apply(df, index_date=INDEX_DATE)
    text = "\n".join(exc.value.problems)
    assert "Death_Censor_Date" in text and "{'2999-12-31': 3}" in text and "semantic type date" in text and "sentinel" in text
    assert not any(cid in text for cid in ids)
    # an undeclared far-future date in an enhanced column the build does not use -> reported and kept, the build continues
    df = wide.copy()
    df.loc[rows, "Last_Visit_Date"] = pd.Timestamp("2999-12-31")
    frame2, rep2 = adapter.apply(df, index_date=INDEX_DATE)
    assert len(frame2) == len(frame) and any("Last_Visit_Date" in w and "2999-12-31" in w and "not declared sentinels" in w for w in rep2.warnings)
    assert rep2.contract["date_range"]["Last_Visit_Date"]["undeclared_beyond_ns_range_values"] == {"2999-12-31": 3}


def test_modelling_layer_cast_applies_only_column_specific_sentinel_rules(contract, tmp_path) -> None:
    from falls_ml.data.meuhedet_wide import BuildReport, to_modeling_dates

    s = pd.Series(np.array(["2999-12-31", "2025-01-01", "NaT", "2999-12-31"], dtype="datetime64[us]"))
    keep = contract.get("Population_Record_To_Date")
    problems, rep = [], BuildReport(index_date=INDEX_DATE)
    out = to_modeling_dates(s, source="Population_Record_To_Date", canonical="x", column=keep, problems=problems, rep=rep)
    assert str(out.dtype) == "datetime64[ns]" and out.isna().tolist() == [True, False, True, True] and out.iloc[1] == pd.Timestamp("2025-01-01")
    assert len(problems) == 1 and "2 dates {'2999-12-31': 2}" in problems[0] and "datetime64[ns]" in problems[0] and "declared sentinel" in problems[0]
    assert "expected semantic type date" in problems[0] and rep.sentinel_to_null == {}
    problems, rep = [], BuildReport(index_date=INDEX_DATE)
    out = to_modeling_dates(s, source="Population_Record_To_Date", canonical="x", column=replace(keep, sentinel_modeling_rule="to_null"),
                            problems=problems, rep=rep)
    assert problems == [] and rep.sentinel_to_null == {"x": 2} and out.isna().tolist() == [True, False, True, True]
    assert any("to_null" in w and "2999-12-31" in w for w in rep.warnings)
    problems, rep = [], BuildReport(index_date=INDEX_DATE)
    to_modeling_dates(s, source="Last_Visit_Date", canonical="y", column=contract.get("Last_Visit_Date"), problems=problems, rep=rep)
    assert len(problems) == 1 and "not a declared sentinel" in problems[0]
    problems, rep = [], BuildReport(index_date=INDEX_DATE)
    same = to_modeling_dates(s.iloc[1:3], source="Index_Date", canonical="index_date", column=contract.get("Index_Date"), problems=problems, rep=rep)
    assert problems == [] and same.tolist()[0] == pd.Timestamp("2025-01-01") and str(same.dtype) == "datetime64[ns]"
    # contract rules: sentinels only on date columns, always with a meaning, ISO dates, rule from the closed list
    raw = yaml.safe_load((ROOT / DEFAULT_CONTRACT).read_text(encoding="utf-8"))
    cases = (("Age_At_Index", {"sentinels": ["2999-12-31"], "sentinel_means": "x"}, "only allowed on date"),
             ("Last_Visit_Date", {"sentinels": ["2999-12-31"]}, "need sentinel_means"),
             ("Last_Visit_Date", {"sentinels": ["31/12/2999"], "sentinel_means": "x"}, "not an ISO date"),
             ("Last_Visit_Date", {"sentinels": ["2999-12-31"], "sentinel_means": "x", "sentinel_modeling_rule": "drop"}, "sentinel_modeling_rule"),
             ("Last_Visit_Date", {"sentinel_means": "x"}, "without sentinels"))
    for i, (column, extra, match) in enumerate(cases):
        bad = yaml.safe_load(yaml.safe_dump(raw))
        bad["columns"][column].update(extra)
        p = tmp_path / f"contract_{i}.yaml"
        p.write_text(yaml.safe_dump(bad, sort_keys=False, allow_unicode=True), encoding="utf-8")
        with pytest.raises(ConfigError, match=match):
            load_wide_contract(p)


def test_source_dates_are_read_only_with_the_declared_formats(wide, contract, tmp_path) -> None:
    """DD/MM/YYYY exports (the real Meuhedet CSV) and ISO exports are both read; the layout is declared, never inferred."""
    from falls_ml.data.meuhedet_wide import read_wide_extract_report

    assert contract.date_formats == ("%Y-%m-%d", "%d/%m/%Y")
    values = ["01/02/2024", "31/12/2024", "2024-02-01", "2999-12-31", None, "2025-13-45", "45000", "02-30-2024", "01/02/2024 14:30:00"]
    csv = _csv_with_dates(wide, tmp_path / "formats.csv", "Population_Record_To_Date", values)
    col = read_wide_extract_report(csv, contract).frame["Population_Record_To_Date"]
    assert col.iloc[0] == pd.Timestamp("2024-02-01")    # 01/02/2024 is always 1 February, never 2 January
    assert col.iloc[1] == pd.Timestamp("2024-12-31") and col.iloc[2] == pd.Timestamp("2024-02-01")
    assert col.iloc[3] == pd.Timestamp("2999-12-31") and col.iloc[8] == pd.Timestamp("2024-02-01 14:30:00")
    assert [pd.isna(col.iloc[i]) for i in (4, 5, 6, 7)] == [True] * 4   # NULL and three malformed values
    r = read_wide_extract_report(csv, contract)
    assert r.wrong_type == {"Population_Record_To_Date": 3}
    assert r.wrong_type_examples == {"Population_Record_To_Date": ["2025-13-45", "45000", "02-30-2024"]}
    assert r.date_formats["Population_Record_To_Date"] == {"%d/%m/%Y": 3, "%Y-%m-%d": 2}
    assert r.declared_date_formats == ("%Y-%m-%d", "%d/%m/%Y")
    assert r.to_dict()["declared_date_formats"] == ["%Y-%m-%d", "%d/%m/%Y"]
    rep = validate_wide_contract(r.frame, contract, wrong_type=r.wrong_type, wrong_type_examples=r.wrong_type_examples,
                                 date_formats=r.date_formats, strict=False)
    assert any("mixes source date layouts" in w for w in rep.warnings)
    assert rep.date_formats["Population_Record_To_Date"] == {"%d/%m/%Y": 3, "%Y-%m-%d": 2}


def test_contract_refuses_ambiguous_or_incomplete_date_formats(tmp_path) -> None:
    from falls_ml.data.meuhedet_wide import check_date_formats

    assert check_date_formats(["%Y-%m-%d", "%d/%m/%Y"]) == []
    assert any("ambiguous" in p for p in check_date_formats(["%d/%m/%Y", "%m/%d/%Y"]))     # day/month inference is impossible by construction
    assert any("year, month and day" in p for p in check_date_formats(["%d/%m/%y"]))
    assert check_date_formats([]) and any("at least one" in p for p in check_date_formats([]))
    raw = yaml.safe_load((ROOT / DEFAULT_CONTRACT).read_text(encoding="utf-8"))
    for i, formats in (("amb", ["%d/%m/%Y", "%m/%d/%Y"]), ("bad", ["%Y/%d"]), ("empty", [])):
        bad = yaml.safe_load(yaml.safe_dump(raw))
        bad["contract"]["date_formats"] = formats
        p = tmp_path / f"contract_fmt_{i}.yaml"
        p.write_text(yaml.safe_dump(bad, sort_keys=False, allow_unicode=True), encoding="utf-8")
        with pytest.raises(ConfigError, match="date_formats"):
            load_wide_contract(p)


def test_a_ddmmyyyy_extract_builds_exactly_like_the_iso_extract(tmp_path, exploratory_spec, mapping) -> None:
    iso = write_synthetic_wide_extract(tmp_path / "iso.csv", n_rows=400, seed=9, other_index_date_share=0.0)
    ddmm = write_synthetic_wide_extract(tmp_path / "ddmm.csv", n_rows=400, seed=9, other_index_date_share=0.0, csv_date_format="%d/%m/%Y")
    assert "01/01/2025" in ddmm.read_text(encoding="utf-8") and "2025-01-01" not in ddmm.read_text(encoding="utf-8")
    a, wa = read_wide_extract(iso, contract := load_wide_contract(DEFAULT_CONTRACT))
    b, wb = read_wide_extract(ddmm, contract)
    assert wa == wb == {} 
    pd.testing.assert_frame_equal(a, b)   # same data, only the source layout differs
    _, rep = build_meuhedet_dataset(ddmm, tmp_path / "ds", index_date=INDEX_DATE, dataset_version="ddmm", data_freeze_date=None)
    assert rep.n_rows_final > 0 and rep.outcome_window_days_observed == {"180": rep.n_rows_final}


def test_multiple_index_dates_are_listed_as_parsed_dates_with_the_index_date_instruction(contract, tmp_path, capsys) -> None:
    """The real pattern 01/01/2024, 01/02/2024, 01/03/2024 ... must be reported as 2024-01-01, 2024-02-01, 2024-03-01."""
    from falls_ml.data.meuhedet_wide import read_wide_extract
    from falls_ml.meuhedet_explore import infer_index_date

    df = generate_synthetic_wide_extract(300, seed=13, contract=contract)
    months = [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-01"), pd.Timestamp("2024-03-01")]
    df["Index_Date"] = pd.Series([months[i % 3] for i in range(len(df))], index=df.index, dtype="datetime64[us]")
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].dt.strftime("%d/%m/%Y")
    csv = tmp_path / "months.csv"
    out.to_csv(csv, index=False, na_rep="NULL")
    assert "01/01/2024" in csv.read_text(encoding="utf-8") and "01/03/2024" in csv.read_text(encoding="utf-8")
    frame, wrong = read_wide_extract(csv, contract)
    assert wrong == {} and sorted(str(d.date()) for d in frame["Index_Date"].dropna().unique()) == ["2024-01-01", "2024-02-01", "2024-03-01"]
    with pytest.raises(DatasetValidationError) as exc:
        infer_index_date(frame)
    assert "3 index dates" in str(exc.value) and "--index-date" in str(exc.value)
    listed = "\n".join(exc.value.problems)
    assert "2024-01-01: 100 rows" in listed and "2024-02-01: 100 rows" in listed and "2024-03-01: 100 rows" in listed
    assert "for example: --index-date 2024-01-01" in listed
    assert cli_main(["meuhedet-explore", "--input", str(csv), "--out", str(tmp_path / "explore")]) == 2
    err = capsys.readouterr().err
    assert "STOPPED" in err and "2024-02-01: 100 rows" in err and "--index-date 2024-01-01" in err


def test_open_followup_sentinel_is_declared_kept_raw_and_read_as_open_follow_up(adapter, wide, contract, tmp_path) -> None:
    """Followup_End_Date = 2999-12-31 means OPEN_FOLLOWUP_NO_KNOWN_END: it passes the contract, keeps its raw value, and reaches
    the modelling layer as NULL (this column's declared value for an open follow-up) - never as a follow-up that ended in 2999."""
    col = contract.get("Followup_End_Date")
    assert col.sentinels == ("2999-12-31",) and col.sentinel_modeling_rule == "to_null"
    assert "OPEN_FOLLOWUP_NO_KNOWN_END" in col.sentinel_means and col.null_means == "follow-up open"
    open_rows = _eligible_rows(wide)[:4]
    df = wide.copy()
    df.loc[open_rows, "Followup_End_Date"] = pd.Timestamp("2999-12-31")   # open follow-up, sentinel form
    df.loc[open_rows, ["Is_Censored_180D", "Is_Deceased_Ind"]] = 0
    df.loc[open_rows, "Has_Full_Followup_180D"] = 1
    # 1. contract layer: the raw value is kept and the column passes validation (no undeclared far-future dates)
    rep = validate_wide_contract(df, contract, strict_columns=["Followup_End_Date"], strict=True)
    assert rep.date_range["Followup_End_Date"]["declared_sentinels"]["2999-12-31"] >= 4
    assert rep.date_range["Followup_End_Date"]["undeclared_beyond_ns_range_values"] == {}
    assert (df.loc[open_rows, "Followup_End_Date"] == pd.Timestamp("2999-12-31")).all()
    # 2. modelling layer: NULL = open follow-up, counted and warned, the rows are kept and nothing else moves
    frame, build = adapter.apply(df, index_date=INDEX_DATE)
    base, _ = adapter.apply(wide, index_date=INDEX_DATE)
    n_sentinel = int((df.loc[df.index.isin(_eligible_rows(df)), "Followup_End_Date"] == pd.Timestamp("2999-12-31")).sum())
    assert build.sentinel_to_null == {"followup_end_date": n_sentinel} and n_sentinel >= 4
    assert any("OPEN_FOLLOWUP_NO_KNOWN_END" in w and "2999-12-31" in w for w in build.warnings)
    assert len(frame) == len(base) and str(frame["followup_end_date"].dtype) == "datetime64[ns]"
    assert not (frame["followup_end_date"].dropna() > pd.Timestamp("2262-01-01")).any()  # never "follow-up ended in 2999"
    pd.testing.assert_series_equal(frame["fall_next_180d"], base["fall_next_180d"])
    pd.testing.assert_series_equal(frame["death_date"], base["death_date"])
    # rows whose follow-up is open are exactly the rows with a NULL canonical follow-up end
    is_open = (df.loc[_eligible_rows(df), "Followup_End_Date"].isna() | (df.loc[_eligible_rows(df), "Followup_End_Date"] == pd.Timestamp("2999-12-31")))
    assert int(frame["followup_end_date"].isna().sum()) == int(is_open.sum())
    # 3. end to end through the immutable dataset: the build succeeds and records the conversion
    csv = write_synthetic_wide_extract(tmp_path / "open.csv", n_rows=400, seed=17, other_index_date_share=0.0)
    manifest, report = build_meuhedet_dataset(csv, tmp_path / "ds_open", index_date=INDEX_DATE, dataset_version="open-followup",
                                              data_freeze_date=None)
    assert report.sentinel_to_null.get("followup_end_date", 0) > 0
    audit = manifest.to_dict()["audit"]["meuhedet_build"]
    assert audit["sentinel_to_null"]["followup_end_date"] == report.sentinel_to_null["followup_end_date"]
    assert audit["contract_report"]["date_range"]["Followup_End_Date"]["sentinel_modeling_rule"] == "to_null"
    assert audit["contract"]["name"] == "meuhedet_wide_v1" and audit["contract_report"]["date_formats"]["Followup_End_Date"]
    saved = pd.read_parquet(next((tmp_path / "ds_open").glob("*.parquet")))
    assert saved["followup_end_date"].isna().any() and (saved["followup_end_date"].dropna() < pd.Timestamp("2262-01-01")).all()


def test_the_2999_sentinel_is_declared_only_where_the_business_meaning_is_known(contract) -> None:
    """Narrow scan: exactly two date columns declare 2999-12-31, and the contract layer never assumes it for the others."""
    declared = {c.name: c.sentinels for c in contract.columns if c.sentinels}
    assert declared == {"Population_Record_To_Date": ("2999-12-31",), "Followup_End_Date": ("2999-12-31",)}
    others = [c.name for c in contract.columns if c.is_date and not c.sentinels]
    assert len(others) == 23 and "Death_Censor_Date" in others and "Next_Fall_Date_180D" in others
    # an undeclared column keeps reporting the value instead of assuming a meaning for it
    s = pd.Series(np.array(["2999-12-31", "2025-01-01"], dtype="datetime64[us]"))
    from falls_ml.data.meuhedet_wide import date_range_report
    dr = date_range_report(s, contract.get("Death_Censor_Date"))
    assert dr["undeclared_beyond_ns_range_values"] == {"2999-12-31": 1} and dr["declared_sentinels"] == {} and dr["sentinel_means"] is None


# ------------------------------------------------------------------ build + load through the framework
def test_build_writes_an_immutable_validated_dataset(tmp_path, exploratory_spec, mapping) -> None:
    csv = write_synthetic_wide_extract(tmp_path / "wide.csv", n_rows=800, seed=5)
    manifest, rep = build_meuhedet_dataset(csv, tmp_path / "ds", index_date=INDEX_DATE, dataset_version="synthetic-wide-test",
                                           data_freeze_date="2025-09-01")
    assert manifest.source == "synthetic_fixture" and rep.synthetic and manifest.scientific_use_allowed is False  # SYN_ snapshot keys
    assert manifest.notes.startswith("SYNTHETIC DATA")
    assert manifest.mapping_version == f"{mapping.name}-{mapping.version}"
    build = manifest.audit["meuhedet_build"]
    assert build["mapping"]["sha256"] == mapping.content_sha256 and build["exploratory_outcome"]["NOT_EFALLS_OUTCOME"] is True
    assert build["database_connections"] == "none" and build["index_date"] == INDEX_DATE
    ds = ModelingDataset.load(tmp_path / "ds", exploratory_spec, features=mapping.baseline_features)
    assert len(ds) == rep.n_rows_final and ds.spec.is_subset
    with pytest.raises(DatasetValidationError, match="immutable"):
        build_meuhedet_dataset(csv, tmp_path / "ds", index_date=INDEX_DATE, dataset_version="x", data_freeze_date="2025-09-01")


# ------------------------------------------------------------------ audit privacy and content
def test_audit_is_aggregate_non_identifying_and_suppresses_small_cells(wide, contract, mapping, exploratory_spec, tmp_path) -> None:
    rep = audit_wide_extract(wide, contract, mapping, exploratory_spec, index_date=INDEX_DATE, min_cell=10)
    j, m = write_audit(rep, tmp_path / "audit")
    text = j.read_text(encoding="utf-8") + m.read_text(encoding="utf-8")
    assert not re.search(r"S\d{10}|EV180-\d|EV30-\d", text)  # no member ids, no event ids
    assert rep["columns"]["Customer_Full_ID"].keys() >= {"n_unique"} and "distribution" not in rep["columns"]["Customer_Full_ID"]
    assert rep["leakage_audit"]["result"] == "PASS" and rep["outcomes"]["efalls_365d_outcome"]["exists"] is False
    assert rep["outcomes"]["efalls_365d_outcome"]["verdict"].startswith("BLOCKED")
    assert rep["efalls_coverage"]["n_available"] == 15 and rep["efalls_coverage"]["mandatory_unavailable"] == ["fracture", "fragility_fracture"]
    assert rep["datatype_audit"]["wrong_type_total"] == 0
    assert "MMSE_Score" in rep["columns"] and rep["columns"]["MMSE_Score"]["pct_null"] == 100.0
    assert rep["overview"]["gender_code_x_desc"]["1"]["זכר"] > 0
    strict = audit_wide_extract(wide, contract, mapping, exploratory_spec, index_date=INDEX_DATE, min_cell=10_000)
    assert SUPPRESSED in json.dumps(strict["cohort_profile"]["sex"], ensure_ascii=False)
    # a dropped label column makes the audit say so rather than crash
    partial = audit_wide_extract(wide.drop(columns=["Fall_Next_30D_Ind"]), contract, mapping, exploratory_spec, index_date=INDEX_DATE)
    assert "fall_next_30d" not in partial["outcomes"] and partial["columns"]["Fall_Next_30D_Ind"] == {"present": False}


def test_tables_render_every_row(mapping, contract) -> None:
    assert len(column_inventory_table(contract)) == 221
    t = mapping_table(mapping)
    assert set(t["mapping_status"]) <= {"AVAILABLE", "APPROXIMATE", "UNAVAILABLE"}
    assert t.loc[t["canonical_feature"] == "fracture", "mandatory"].item() is True


# ------------------------------------------------------------------ CLI and baseline freeze
def test_cli_commands_run_end_to_end_without_training(tmp_path, capsys, monkeypatch) -> None:
    monkeypatch.chdir(ROOT)
    assert cli_main(["meuhedet-make-fixture", "--out", str(tmp_path / "w.parquet"), "--n-rows", "500", "--seed", "9"]) == 0
    assert cli_main(["meuhedet-audit", "--input", str(tmp_path / "w.parquet"), "--out", str(tmp_path / "audit"), "--index-date", INDEX_DATE]) == 0
    assert '"leakage_result": "PASS"' in capsys.readouterr().out
    assert (tmp_path / "audit" / "wide_extract_audit.md").exists()
    assert cli_main(["meuhedet-build", "--input", str(tmp_path / "w.parquet"), "--out", str(tmp_path / "ds"), "--index-date", INDEX_DATE,
                     "--dataset-version", "synthetic-wide-cli", "--data-freeze-date", "2025-09-01", "--report", str(tmp_path / "build.json")]) == 0
    summary = json.loads((tmp_path / "build.json").read_text(encoding="utf-8"))
    assert summary["manifest"]["source"] == "synthetic_fixture" and "NOT eFalls reproduction" in summary["label"]
    with pytest.raises(SystemExit, match="approval-reference"):
        cli_main(["meuhedet-build", "--input", str(tmp_path / "w.parquet"), "--out", str(tmp_path / "ds2"), "--index-date", INDEX_DATE,
                  "--dataset-version", "v", "--data-freeze-date", "2025-09-01", "--scientific-use-allowed"])


def test_freeze_baseline_records_hashes_and_refuses_overwrite(tmp_path) -> None:
    from falls_ml.baseline import freeze_baseline

    run = tmp_path / "run"
    run.mkdir()
    (run / "RUN_COMPLETE.json").write_text('{"run_id": "r1"}', encoding="utf-8")
    metrics = {"run_id": "r1", "experiment": {"name": "e", "kind": "efalls_retrained_reduced"}, "feature_set": {"sha256": "abc"},
               "outcome": {"name": "fall_next_180d", "horizon": "181d", "published_efalls_outcome": False},
               "lasso": {"lambda_star": 0.01}, "performance": {"test": {"uncalibrated": {"auroc": {"estimate": 0.7}}}},
               "split": {"test_rows_sha256": "t" * 64}, "efalls_coverage": {"label": "Reduced"}, "limitations": [], "warnings": []}
    (run / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (run / "config.yaml").write_text(yaml.safe_dump({"model": {"name": "lasso_logistic_cv"}, "preprocessing": {}, "validation": {}, "dataset": {}}), encoding="utf-8")
    (run / "dataset_manifest.json").write_text(json.dumps({"dataset_version": "d", "data_sha256": "h" * 64, "n_rows": 10, "audit": {"meuhedet_build": {
        "index_date": INDEX_DATE, "mapping": {"sha256": "m" * 64}, "contract": {"sha256": "c" * 64}}}}), encoding="utf-8")
    pd.DataFrame({"design_column": ["age_years"], "coefficient": [0.1]}).to_csv(run / "coefficients.csv", index=False)
    rec = freeze_baseline(run, "EFALLS_BASELINE_TEST_V1", baselines_dir=tmp_path / "baselines")
    saved = json.loads((tmp_path / "baselines" / "EFALLS_BASELINE_TEST_V1" / "baseline.json").read_text(encoding="utf-8"))
    assert saved["selected_lambda"] == 0.01 and saved["split"]["test_rows_sha256"] == "t" * 64 and saved["index_date"] == INDEX_DATE
    assert saved["mapping_manifest"]["sha256"] == "m" * 64 and saved["coefficients"][0]["design_column"] == "age_years"
    assert set(saved["files"]) == {"config.yaml", "metrics.json", "coefficients.csv", "dataset_manifest.json"}
    assert rec["baseline_dir"].endswith("EFALLS_BASELINE_TEST_V1")
    with pytest.raises(DatasetValidationError, match="immutable"):
        freeze_baseline(run, "EFALLS_BASELINE_TEST_V1", baselines_dir=tmp_path / "baselines")


@pytest.mark.slow
def test_exploratory_run_is_labelled_not_efalls(tmp_path, monkeypatch) -> None:
    from falls_ml.experiment import run_experiment

    monkeypatch.chdir(ROOT)
    csv = write_synthetic_wide_extract(tmp_path / "wide.parquet", n_rows=2500, seed=21)
    build_meuhedet_dataset(csv, tmp_path / "ds", index_date=INDEX_DATE, dataset_version="synthetic-wide-run", data_freeze_date="2025-09-01")
    cfg = yaml.safe_load(FIXTURE_CONFIG.read_text(encoding="utf-8"))
    cfg["output"]["runs_dir"] = str(tmp_path / "runs")
    cfg["evaluation"]["bootstrap"]["n"] = 10
    cfg["analysis"]["stability"]["n_bootstrap"] = 5
    cfg["analysis"]["optimism"]["n_bootstrap"] = 2
    cfg["evaluation"]["permutation_importance_repeats"] = 1
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    result = run_experiment(None, tmp_path / "ds", cfg_path)
    metrics = json.loads((Path(result.run_dir) / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["outcome"]["published_efalls_outcome"] is False and metrics["outcome"]["name"] == "fall_next_180d"
    assert any("EXPLORATORY OUTCOME" in x for x in metrics["limitations"]) and any("REDUCED eFalls" in x for x in metrics["limitations"])
    assert any("fracture" in x for x in metrics["warnings"]) and metrics["efalls_coverage"]["n_available"] == 15
    report = (Path(result.run_dir) / "report.md").read_text(encoding="utf-8")
    assert "EXPLORATORY OUTCOME" in report


# ------------------------------------------------------------------ Excel input and the one-command exploratory flow
def test_excel_extract_is_read_from_the_contract_not_from_excel_types(contract, tmp_path) -> None:
    from falls_ml.data.meuhedet_wide import read_wide_extract_report

    xlsx = write_synthetic_wide_extract(tmp_path / "w.xlsx", n_rows=200, seed=4)
    pq = write_synthetic_wide_extract(tmp_path / "w.parquet", n_rows=200, seed=4)
    r = read_wide_extract_report(xlsx, contract)
    assert r.source_format == "excel" and r.sheet == "0" and r.wrong_type == {}
    b, _ = read_wide_extract(pq, contract)
    pd.testing.assert_frame_equal(r.frame, b)
    assert r.conversions["Age_At_Index"]["contract_dtype"] == "Int64" and r.conversions["Age_At_Index"]["excel_cell_types"]
    assert r.frame["Customer_Full_ID"].dtype == "string" and r.frame["Index_Date"].dtype == "datetime64[us]"
    assert (r.frame["Population_Record_To_Date"] == pd.Timestamp("2999-12-31")).sum() > 0  # Excel datetime cells keep the SCD sentinel
    # named sheet and an unknown sheet
    named = read_wide_extract_report(xlsx, contract, sheet="V_Falls_Prediction_Wide_1")
    assert named.sheet == "V_Falls_Prediction_Wide_1" and len(named.frame) == 200
    with pytest.raises(DatasetValidationError, match="cannot read sheet"):
        read_wide_extract_report(xlsx, contract, sheet="nope")


def test_excel_numeric_identifiers_fail_loudly_and_are_never_repaired(contract, tmp_path) -> None:
    xlsx = write_synthetic_wide_extract(tmp_path / "bad.xlsx", n_rows=50, seed=4, excel_numeric_ids=True)
    with pytest.raises(DatasetValidationError) as exc:
        read_wide_extract(xlsx, contract)
    assert any("Customer_Full_ID" in p and "Excel converted the identifier" in p for p in exc.value.problems)
    assert not any("S00" in p for p in exc.value.problems)  # no identifier values in the message


def test_feature_sets_are_derived_from_mapping_quality(mapping) -> None:
    sets = mapping.feature_sets()
    hc = [f.canonical for f in mapping.features if f.include_in_baseline and f.quality == "HIGH_CONFIDENCE"]
    approx = [f.canonical for f in mapping.features if f.include_in_baseline and f.quality == "APPROXIMATE"]
    assert sets["strict"] == hc and set(sets["extended"]) == set(hc) | set(approx) and sets["extended"] == mapping.baseline_features
    assert "age_years" in sets["strict"] and "polypharmacy_count_120d" not in sets["strict"]


def test_build_with_a_feature_set_and_pseudonymised_ids(tmp_path, exploratory_spec, mapping) -> None:
    csv = write_synthetic_wide_extract(tmp_path / "wide.csv", n_rows=500, seed=6)
    m1, r1 = build_meuhedet_dataset(csv, tmp_path / "strict", index_date=INDEX_DATE, dataset_version="s", data_freeze_date=None,
                                    feature_set="strict", id_pepper="pepper-1")
    m2, r2 = build_meuhedet_dataset(csv, tmp_path / "extended", index_date=INDEX_DATE, dataset_version="e", data_freeze_date=None,
                                    feature_set="extended", id_pepper="pepper-1")
    strict = ModelingDataset.load(tmp_path / "strict", exploratory_spec, features=mapping.feature_sets()["strict"]).frame
    ext = ModelingDataset.load(tmp_path / "extended", exploratory_spec, features=mapping.feature_sets()["extended"]).frame
    assert set(strict.columns) < set(ext.columns) and "falls" not in strict.columns and "falls" in ext.columns
    assert r1.pseudonymised and (strict["research_id"] == ext["research_id"]).all() and not strict["research_id"].str.startswith("S").any()
    assert strict["research_id"].str.len().eq(20).all()
    assert m1.audit["meuhedet_build"]["feature_set"]["label"] == "strict" and m1.audit["meuhedet_build"]["identifiers"]["pseudonymised"] is True
    assert "pepper-1" not in json.dumps(m1.audit)  # the salt itself is never recorded
    with pytest.raises(ConfigError, match="subset of the included mappings"):
        MeuhedetWideDatasetAdapter(mapping, load_wide_contract(DEFAULT_CONTRACT), exploratory_spec, features=["fracture"])


def test_infer_index_date_requires_a_single_snapshot(wide) -> None:
    from falls_ml.meuhedet_explore import infer_index_date

    with pytest.raises(DatasetValidationError, match="2 index dates"):
        infer_index_date(wide)
    one = wide[wide["Index_Date"] == pd.Timestamp(INDEX_DATE)]
    assert infer_index_date(one) == INDEX_DATE


@pytest.mark.slow
def test_meuhedet_explore_runs_end_to_end_from_excel(tmp_path, monkeypatch) -> None:
    from falls_ml.meuhedet_explore import RUN_LABELS, explore
    from falls_ml.reporting.report import exploratory_outcome_banner

    monkeypatch.chdir(ROOT)
    xlsx = write_synthetic_wide_extract(tmp_path / "falls_extract.xlsx", n_rows=1600, seed=41, other_index_date_share=0.0)
    template = yaml.safe_load((ROOT / "configs" / "experiments" / "meuhedet" / "explore_180d_template.yaml").read_text(encoding="utf-8"))
    template["evaluation"]["bootstrap"]["n"] = 10
    template["evaluation"]["permutation_importance_repeats"] = 1
    template["analysis"]["stability"]["n_bootstrap"] = 4
    template["analysis"]["optimism"]["n_bootstrap"] = 2
    tpl = tmp_path / "template.yaml"
    tpl.write_text(yaml.safe_dump(template, sort_keys=False), encoding="utf-8")
    # a fixed pepper makes the research ids - and therefore the patient-grouped split - reproducible in the test
    res = explore(xlsx, template_path=tpl, fast=False, id_pepper="pepper-e2e")
    out = res.out_dir
    assert out == tmp_path / "falls_extract_explore" and res.index_date == INDEX_DATE
    for fs, label in RUN_LABELS.items():
        assert res.metrics[fs]["experiment"]["name"] == label and res.metrics[fs]["outcome"]["published_efalls_outcome"] is False
        assert (out / f"feature_report_{fs}.md").exists() and (out / f"feature_report_{fs}.csv").exists()
        report = (res.runs[fs] / "report.md").read_text(encoding="utf-8")
        assert "EXPLORATORY 180-DAY OUTCOME" in report and "NOT EFALLS REPRODUCTION" in report
        assert exploratory_outcome_banner(res.metrics[fs]) == "EXPLORATORY 180-DAY OUTCOME (fall_next_180d) – NOT EFALLS REPRODUCTION"
    assert res.comparison["same_test_rows"] and res.metrics["strict"]["split"]["test_rows_sha256"] == res.metrics["extended"]["split"]["test_rows_sha256"]
    assert res.comparison["runs"]["strict"]["n_predictors"] == 9 and res.comparison["runs"]["extended"]["n_predictors"] == 15
    fr = res.feature_reports["extended"]
    assert set(fr["canonical_efalls_feature"]) == set(res.comparison["runs"]["extended"]["predictors"])
    assert {"mapping_quality", "missing_pct", "distribution", "lasso_coefficient", "odds_ratio", "non_zero_coefficient", "selection_frequency",
            "permutation_importance"} <= set(fr.columns)
    summary = (out / "SUMMARY.md").read_text(encoding="utf-8")
    assert "EXPLORATORY — NOT EFALLS REPRODUCTION" in summary and "STRICT" in summary and "EXTENDED" in summary and "Most stable predictors" in summary
    # privacy: no member ids or the pepper in any shared artifact
    pepper = (out / "id_pepper.txt").read_text(encoding="utf-8").strip()
    shared = "".join(p.read_text(encoding="utf-8", errors="ignore") for p in out.rglob("*") if p.suffix in {".md", ".json", ".csv", ".yaml", ".html", ".jsonl"}
                     and p.name != "id_pepper.txt")
    assert not re.search(r"\bS\d{10}\b", shared) and pepper not in shared
    for fs in RUN_LABELS:
        preds = pd.read_parquet(res.runs[fs] / "predictions_test.parquet")
        assert not preds["research_id"].astype(str).str.startswith("S").any()
    with pytest.raises(DatasetValidationError, match="not empty"):
        explore(xlsx, template_path=tpl)
