"""The real-data path of ``meuhedet-explore`` (v0.5.0): D-00 timing diagnostic, multi-snapshot panel fixtures, snapshot audit and
temporal design, adequacy gate, reproducible pepper, readiness verdicts, STRICT/EXTENDED identical rows, patient-disjoint
sensitivity, privacy of every shareable report. SYNTHETIC data only."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from falls_ml.adequacy import EPP_STOP, INSUFFICIENT, READY, assess_adequacy, max_design_columns
from falls_ml.cli import main as cli_main
from falls_ml.config import load_experiment_config
from falls_ml.data.dataset import ModelingDataset
from falls_ml.data.meuhedet_snapshots import EXCLUDED, USABLE, choose_temporal_design, snapshot_audit
from falls_ml.data.meuhedet_synthetic import generate_synthetic_wide_extract, generate_synthetic_wide_panel, write_synthetic_wide_extract
from falls_ml.data.meuhedet_timing import render_timing_markdown, timing_diagnostic
from falls_ml.data.meuhedet_wide import (DEFAULT_CONTRACT, DEFAULT_MAPPING, PREDICTION_TIME, MeuhedetWideDatasetAdapter, load_wide_contract,
                                         load_wide_mapping)
from falls_ml.errors import ConfigError, DatasetValidationError
from falls_ml.features.spec import load_feature_spec
from falls_ml.meuhedet_explore import READY_TEXT, RUN_LABELS, SENSITIVITY_LABELS, explore, resolve_id_pepper
from falls_ml.splitting import make_split

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "configs" / "experiments" / "meuhedet" / "explore_180d_template.yaml"
ID_RE = re.compile(r"\bS\d{10}\b")   # synthetic Customer_Full_ID pattern


@pytest.fixture(scope="module")
def contract():
    return load_wide_contract(DEFAULT_CONTRACT)


@pytest.fixture(scope="module")
def mapping(contract):
    return load_wide_mapping(DEFAULT_MAPPING, contract=contract)


@pytest.fixture(scope="module")
def spec(mapping):
    return load_feature_spec(mapping.exploratory_spec_path)


@pytest.fixture(scope="module")
def panel(contract) -> pd.DataFrame:
    """20 monthly snapshots, freeze-censored tail, planted same-day + future diagnosis and fall records on the first snapshot."""
    return generate_synthetic_wide_panel(500, n_snapshots=20, first_index_date="2024-01-01", seed=5, contract=contract, data_freeze_date="2025-12-31",
                                         n_index_day_dx=25, n_index_day_falls=3, n_future_dx=4, n_future_falls=2)


def small_template(tmp_path: Path) -> Path:
    tpl = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
    tpl["evaluation"]["bootstrap"]["n"] = 10
    tpl["evaluation"]["permutation_importance_repeats"] = 1
    tpl["analysis"]["stability"]["n_bootstrap"] = 4
    tpl["analysis"]["optimism"]["n_bootstrap"] = 2
    p = tmp_path / "template.yaml"
    p.write_text(yaml.safe_dump(tpl, sort_keys=False), encoding="utf-8")
    return p


def shared_text(out: Path) -> str:
    return "".join(p.read_text(encoding="utf-8", errors="ignore") for p in out.rglob("*")
                   if p.suffix in {".md", ".json", ".csv", ".yaml", ".html", ".jsonl", ".txt"} and p.name != "id_pepper.txt" and "datasets" not in p.parts)


# ------------------------------------------------------------------ fixtures: the panel is realistic and contract-valid
def test_panel_fixture_repeats_patients_censors_past_freeze_and_plants_timing_defects(panel, contract) -> None:
    from falls_ml.data.meuhedet_wide import validate_wide_contract

    assert validate_wide_contract(panel, contract, strict=False).problems == []
    dates = sorted(panel["Index_Date"].dt.date.unique())
    assert len(dates) == 20 and panel["Customer_Full_ID"].nunique() == 500 and len(panel) > 500 * 15
    assert not panel.duplicated(subset=["Customer_Full_ID", "Index_Date"]).any() and panel["Snapshot_Key"].is_unique
    one = panel[panel["Customer_Full_ID"] == "S0000000003"].sort_values("Index_Date")
    assert one["Gender_Code"].nunique() == 1 and one["Registry_Dementia_Ind"].nunique() == 1 and one["Age_At_Index"].is_monotonic_increasing
    assert (one["Gait_Disorder_Since_Study_Start_Ind"].diff().dropna() >= 0).all()   # monotone since-study-start flag
    last = panel[panel["Index_Date"].dt.date == dates[-1]]
    assert (last["Has_Full_180D_Label"] == 0).all() and last["Fall_Next_180D_Ind"].isna().mean() > 0.8 and (last["Fall_Next_180D_Ind"].dropna() == 1).all()
    first = panel[panel["Index_Date"].dt.date == dates[0]]
    assert int((first["Last_Dx_Date"] == first["Index_Date"]).sum()) == 25 and int((first["Last_Dx_Date"] > first["Index_Date"]).sum()) == 4
    assert int((first["Last_Fall_Date"] == first["Index_Date"]).sum()) == 3 and int((first["Last_Fall_Date"] > first["Index_Date"]).sum()) == 2
    assert (panel["Population_Record_To_Date"] == pd.Timestamp("2999-12-31")).any() and (panel["Followup_End_Date"] == pd.Timestamp("2999-12-31")).any()
    # the single-snapshot generator is byte-for-byte what it was (seeded expectations elsewhere stay valid)
    single = generate_synthetic_wide_extract(300, seed=7, contract=contract)
    assert single.shape == (300, 221) and int(single["Fall_Next_180D_Ind"].sum()) == 18


# ------------------------------------------------------------------ D-00 timing diagnostic
def test_timing_diagnostic_classifies_same_day_and_future_records_and_traces_predictors(panel, contract, mapping) -> None:
    rep = timing_diagnostic(panel, contract, mapping)
    v = rep["verdict"]
    assert rep["prediction_time"] == PREDICTION_TIME == "START_OF_INDEX_DAY" and rep["d00_guard_columns"] == ["Last_Fall_Date", "Last_Dx_Date"]
    assert v["root_cause"] == "A_AND_B" and v["root_cause_by_column"] == {"Last_Fall_Date": "A_AND_B", "Last_Dx_Date": "A_AND_B"}
    assert v["affected_efalls_features_by_feature_set"] == {"strict": [], "extended": ["falls", "mobility_problems"]}
    assert v["python_repair_possible"] is False and any("Event_Date < Index_Date" in line for line in v["dwh_correction"])
    dx = rep["columns"]["Last_Dx_Date"]
    assert dx["n_on_index"] == 25 and dx["n_after_index"] == 4 and dx["positive_offset_days"]["buckets"]["0 (index day)"] == 25
    assert dx["positive_offset_days"]["min"] == 0 and dx["positive_offset_days"]["max"] >= 1
    assert dx["days_since_companion"]["column"] == "Days_Since_Last_Diagnosis" and dx["days_since_companion"]["on_index_rows"]["days_since_0"] == 25
    assert dx["days_since_companion"]["after_index_rows"]["days_since_negative"] == 4 and "(C, not D)" in dx["root_cause_text"]
    assert set(dx["related_predictors"]["phase2_columns"]) >= {"Diagnosis_Count_180D", "Diagnosis_Count_365D", "Distinct_Diagnosis_Codes_365D",
                                                                "Chronic_Diagnosis_Count_365D", "Days_Since_Last_Diagnosis"}
    assert [f["feature"] for f in dx["related_predictors"]["efalls_features"]] == ["mobility_problems"]
    falls = rep["columns"]["Last_Fall_Date"]
    assert set(falls["related_predictors"]["phase2_columns"]) >= {"Prior_Fall_Count_30D", "Prior_Fall_Count_90D", "Prior_Fall_Count_180D",
                                                                   "Prior_Fall_Count_365D", "Prior_Fall_Count_Since_Study_Start", "Days_Since_Last_Fall"}
    assert list(v["violating_eligible_rows_by_index_date"]) == ["2024-01-01"] and "label_prevalence" in dx
    md = render_timing_markdown(rep)
    assert "DWH / source correction required" in md and not ID_RE.search(md) and "Days_Since_Last_Diagnosis" in md


def test_timing_diagnostic_is_clean_on_a_clean_panel_and_same_day_only_is_cause_a(contract, mapping) -> None:
    clean = generate_synthetic_wide_panel(200, n_snapshots=3, seed=6, contract=contract)
    assert timing_diagnostic(clean, contract, mapping)["verdict"]["root_cause"] == "CLEAN"
    same_day = generate_synthetic_wide_panel(200, n_snapshots=3, seed=6, contract=contract, n_index_day_dx=10)
    rep = timing_diagnostic(same_day, contract, mapping)
    assert rep["verdict"]["root_cause"] == "A_SAME_DAY_INCLUSION" and rep["columns"]["Last_Dx_Date"]["positive_offset_days"]["max"] == 0
    assert rep["columns"]["Last_Fall_Date"]["root_cause"] == "CLEAN" and rep["columns"]["Last_Fall_Date"]["dwh_correction"] == []


def test_adapter_refuses_true_future_and_same_day_records_under_start_of_day_rule(panel, contract, mapping, spec) -> None:
    adapter = MeuhedetWideDatasetAdapter(mapping, contract, spec, features=mapping.feature_sets()["extended"])
    with pytest.raises(DatasetValidationError, match="D-00 timing"):
        adapter.apply(panel, index_date="2024-01-01")
    frame, rep = adapter.apply(panel, index_date="2024-01-01", index_day_records="drop_rows")
    assert rep.n_rows_timing_violation > 0 and rep.timing_violations_by_column["Last_Dx_Date"] >= 25 and rep.timing_violations_by_column["Last_Fall_Date"] >= 3
    assert rep.timing_violations_by_index_date == {"2024-01-01": rep.n_rows_timing_violation}
    assert (frame["predictor_max_record_date"] < frame["index_date"]).all()   # what survives is strictly pre-index
    # a clean snapshot of the same panel keeps every eligible labelled row
    frame2, rep2 = adapter.apply(panel, index_date="2024-02-01")
    assert rep2.n_rows_timing_violation == 0 and len(frame2) == rep2.n_rows_final


def test_mapping_declares_the_record_date_sources_and_the_start_of_day_contract(mapping, contract, spec, tmp_path) -> None:
    assert {f.canonical: f.record_date_column for f in mapping.features if f.record_date_column} == {"falls": "Last_Fall_Date", "mobility_problems": "Last_Dx_Date"}
    assert mapping.cohort["prediction_time"] == "START_OF_INDEX_DAY" and mapping.cohort["predictor_record_rule"] == "source_event_date < Index_Date"
    raw = yaml.safe_load(Path(DEFAULT_MAPPING).read_text(encoding="utf-8"))
    weak = json.loads(json.dumps(raw))
    weak["cohort"]["predictor_record_rule"] = "source_event_date <= Index_Date"
    p = tmp_path / "weak.yaml"
    p.write_text(yaml.safe_dump(weak, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ConfigError, match="cannot be weakened"):
        load_wide_mapping(p, contract=contract)
    mismatch = json.loads(json.dumps(raw))
    mismatch["cohort"]["predictor_max_record_date"]["columns"] = ["Last_Fall_Date"]
    p.write_text(yaml.safe_dump(mismatch, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ConfigError, match="record_date_column"):
        load_wide_mapping(p, contract=contract)


# ------------------------------------------------------------------ snapshot audit and temporal design
def test_snapshot_audit_excludes_partially_observed_snapshots_and_chooses_embargo_aware_boundaries(panel, contract, mapping, spec, tmp_path) -> None:
    rows, totals = snapshot_audit(panel, contract, mapping, data_freeze_date="2025-12-31", index_day_records="drop_rows")
    assert totals["n_snapshots"] == 20 and totals["n_usable_snapshots"] == 19 and totals["unique_patients"] == 500
    last = rows[-1]
    assert last.status == EXCLUDED and "Has_Full_180D_Label = 0" in last.reason and "censored share" in last.reason and "after the data freeze" in last.reason
    assert rows[0].n_d00_violations >= 28 and all(r.n_d00_violations == 0 for r in rows[1:])
    assert rows[0].n_modelling_events == rows[0].n_events - sum(1 for _ in ()) or rows[0].n_modelling_rows < rows[0].n_labelled
    design = choose_temporal_design(rows, horizon_days=totals["horizon_days"])
    dates = [pd.Timestamp(r.index_date) for r in rows if r.status == USABLE]
    train_end, val_end = pd.Timestamp(design.train_end), pd.Timestamp(design.validation_end)
    assert train_end < val_end < dates[-1] and design.expected["test"]["snapshots"] >= 1
    first_val = min(d for d in dates if d > train_end)
    first_test = min(d for d in dates if d > val_end)
    for r in rows:
        d = pd.Timestamp(r.index_date)
        if r.partition == "train":
            assert d <= train_end and d + pd.Timedelta(days=180) < first_val
        elif r.partition == "train (embargoed)":
            assert d <= train_end and d + pd.Timedelta(days=180) >= first_val
        elif r.partition == "validation":
            assert train_end < d <= val_end and d + pd.Timedelta(days=180) < first_test
        elif r.partition == "test":
            assert d > val_end
    # the simulated embargo equals what falls_ml.splitting does on the built dataset
    usable = [r.index_date for r in rows if r.status == USABLE]
    from falls_ml.data.meuhedet_wide import build_meuhedet_dataset_from_frame
    build_meuhedet_dataset_from_frame(panel, tmp_path / "ds", index_dates=usable, dataset_version="v", data_freeze_date="2025-12-31", mapping=mapping,
                                      contract=contract, spec=spec, input_name="x", input_sha256="0" * 64, features=mapping.feature_sets()["extended"],
                                      feature_set_label="extended", id_pepper="p", index_day_records="drop_rows")
    ds = ModelingDataset.load(tmp_path / "ds", spec, features=mapping.feature_sets()["extended"])
    tpl = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
    tpl["preprocessing"]["features"] = mapping.feature_sets()["extended"]
    tpl["preprocessing"]["fractional_polynomial"]["variables"] = ["age_years", "polypharmacy_count_120d"]
    tpl["validation"] = {"strategy": "temporal", "seed": 42, "cv_folds": 10, "patients_disjoint": False, "outcome_lag_days": 0, "limitation_note": design.limitation_note,
                         "temporal": {"train_end": design.train_end, "validation_end": design.validation_end, "embargo_outcome_windows": True}}
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(tpl, sort_keys=False), encoding="utf-8")
    cfg = load_experiment_config(cfg_path)
    plan = make_split(ds.frame, ds.spec, cfg.validation)
    for part in ("train", "validation", "test"):
        assert len(plan.indices()[part]) == design.expected[part]["rows_after_embargo"]
    f = ds.frame
    assert f.iloc[plan.train_idx]["index_date"].max() < f.iloc[plan.validation_idx]["index_date"].min() < f.iloc[plan.test_idx]["index_date"].min()
    # repeated patients across partitions are expected (temporal repeated-risk) and are counted, not hidden
    ids = {p: set(f.iloc[i]["research_id"]) for p, i in plan.indices().items()}
    assert ids["train"] & ids["test"]
    # adequacy is READY on this panel and INSUFFICIENT on the tiny one
    a = assess_adequacy(f, ds.spec, plan, cfg)
    assert a.verdict == READY and a.partitions["test"]["classes_present"] == [0, 1] and a.thresholds["train_events_required"] == max(20, EPP_STOP * max_design_columns(ds.spec, cfg))


def test_temporal_design_refuses_when_the_embargo_leaves_no_validation(contract, mapping) -> None:
    short = generate_synthetic_wide_panel(300, n_snapshots=8, seed=9, contract=contract)
    rows, totals = snapshot_audit(short, contract, mapping)
    with pytest.raises(DatasetValidationError, match="embargo"):
        choose_temporal_design(rows, horizon_days=totals["horizon_days"])


# ------------------------------------------------------------------ pepper: automatic, local, reusable
def test_pepper_is_created_once_next_to_the_input_and_reused(tmp_path) -> None:
    src = tmp_path / "extract.csv"
    src.write_text("x\n", encoding="utf-8")
    out1, out2 = tmp_path / "o1", tmp_path / "o2"
    out1.mkdir(), out2.mkdir()
    p1, info1 = resolve_id_pepper(src, out1)
    p2, info2 = resolve_id_pepper(src, out2)
    assert p1 == p2 and info1["source"].startswith("created") and info2["source"].startswith("reused") and info1["sha256"] == info2["sha256"]
    assert (tmp_path / "extract.csv.id_pepper.txt").is_file() and p1 not in json.dumps({k: v for k, v in info1.items()})
    p3, info3 = resolve_id_pepper(src, out2, explicit="other")
    assert p3 == "other" and info3["warnings"]
    other = tmp_path / "other.csv"
    other.write_text("y\n", encoding="utf-8")
    assert resolve_id_pepper(other, out1)[0] != p1


# ------------------------------------------------------------------ readiness (audit-only) verdicts through the CLI
def test_audit_only_stops_on_d00_then_is_ready_with_drop_rows_and_reproduces_the_split(tmp_path, capsys, monkeypatch) -> None:
    monkeypatch.chdir(ROOT)
    csv = write_synthetic_wide_extract(tmp_path / "wide.csv", n_rows=2500, seed=11, other_index_date_share=0.0, csv_date_format="%d/%m/%Y", n_index_day_falls=5)
    raw = pd.read_csv(csv, dtype=str, keep_default_na=False)
    idx = raw.index[(raw["Is_Eligible_Cohort"] == "1") & (raw["Last_Dx_Date"] != "NULL")][:40]
    raw.loc[idx, "Last_Dx_Date"] = raw.loc[idx, "Index_Date"]
    raw.loc[idx, "Days_Since_Last_Diagnosis"] = "0"
    raw.to_csv(csv, index=False, lineterminator="\n")
    assert cli_main(["meuhedet-explore", "--input", str(csv), "--audit-only", "--out", str(tmp_path / "a")]) == 2
    err = capsys.readouterr().err
    assert "STOPPED" in err and "D-00 timing" in err and "timing_diagnostic.md" in err and "drop_rows" in err
    readiness = (tmp_path / "a" / "READINESS.md").read_text(encoding="utf-8")
    assert readiness.startswith("# STOPPED") and "EXPLORATORY 180-DAY OUTCOME" in readiness
    assert not (tmp_path / "a" / "runs").exists()
    assert cli_main(["meuhedet-explore", "--input", str(csv), "--audit-only", "--index-day-records", "drop_rows", "--out", str(tmp_path / "b")]) == 0
    out = capsys.readouterr().out
    assert READY_TEXT in out and "runs" not in json.loads(out[out.index("{"):])["runs"]
    r1 = json.loads((tmp_path / "b" / "readiness.json").read_text(encoding="utf-8"))
    assert r1["verdict"] == READY_TEXT and r1["timing_verdict"]["root_cause"] == "A_SAME_DAY_INCLUSION" and r1["adequacy"]["extended"]["verdict"] == READY
    assert (tmp_path / "wide.csv.id_pepper.txt").is_file()
    assert cli_main(["meuhedet-explore", "--input", str(csv), "--audit-only", "--index-day-records", "drop_rows", "--out", str(tmp_path / "c")]) == 0
    r2 = json.loads((tmp_path / "c" / "readiness.json").read_text(encoding="utf-8"))
    assert r2["split_audit"]["test_rows_sha256"] == r1["split_audit"]["test_rows_sha256"] and r2["pepper_sha256"] == r1["pepper_sha256"]
    assert {k: v["rows_sha256"] for k, v in r2["split_audit"]["partitions"].items()} == {k: v["rows_sha256"] for k, v in r1["split_audit"]["partitions"].items()}
    # every shareable file is free of member ids and of the pepper value
    pepper = (tmp_path / "b" / "id_pepper.txt").read_text(encoding="utf-8").strip()
    text = shared_text(tmp_path / "b")
    assert not ID_RE.search(text) and pepper not in text and r1["pepper_sha256"] in text


def test_audit_only_stops_with_insufficient_events_before_any_fit(tmp_path, capsys, monkeypatch) -> None:
    monkeypatch.chdir(ROOT)
    tiny = write_synthetic_wide_extract(tmp_path / "tiny.csv", n_rows=150, seed=3, other_index_date_share=0.0)
    assert cli_main(["meuhedet-explore", "--input", str(tiny), "--out", str(tmp_path / "t")]) == 2
    err = capsys.readouterr().err
    assert INSUFFICIENT in err and "events" in err and not (tmp_path / "t" / "runs").exists() and not (tmp_path / "t" / "SUMMARY.md").exists()
    assert (tmp_path / "t" / "READINESS.md").read_text(encoding="utf-8").startswith("# STOPPED — " + INSUFFICIENT)


def test_multiple_index_dates_without_a_choice_stop_and_offer_both_modes(tmp_path, capsys, monkeypatch) -> None:
    monkeypatch.chdir(ROOT)
    p = write_synthetic_wide_extract(tmp_path / "panel.csv", n_rows=200, seed=4, n_snapshots=3, index_date="2024-01-01")
    assert cli_main(["meuhedet-explore", "--input", str(p), "--audit-only", "--out", str(tmp_path / "m")]) == 2
    err = capsys.readouterr().err
    assert "3 index dates" in err and "--index-date 2024-01-01" in err and "--all-index-dates" in err
    with pytest.raises(ConfigError, match="not both"):
        explore(p, out_dir=tmp_path / "m2", index_date="2024-01-01", all_index_dates=True)


# ------------------------------------------------------------------ end to end (slow): all snapshots, temporal + patient-disjoint
@pytest.mark.slow
def test_all_snapshots_explore_trains_temporal_and_patient_disjoint_on_identical_rows(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(ROOT)
    panel = write_synthetic_wide_extract(tmp_path / "panel.csv", n_rows=700, seed=8, n_snapshots=20, index_date="2024-01-01", data_freeze_date="2025-12-31",
                                         n_index_day_dx=30, csv_date_format="%d/%m/%Y")
    res = explore(panel, out_dir=tmp_path / "out", all_index_dates=True, audit_only=False, data_freeze_date="2025-12-31", index_day_records="drop_rows",
                  template_path=small_template(tmp_path), fast=False)
    assert res.ready and res.mode == "all_snapshots" and res.design is not None
    out = res.out_dir
    # identical rows and test partition for STRICT and EXTENDED, temporal ordering, embargo counted
    assert res.comparison["same_test_rows"] and res.metrics["strict"]["split"]["test_rows_sha256"] == res.metrics["extended"]["split"]["test_rows_sha256"] == res.split_audit["test_rows_sha256"]
    for fs in ("strict", "extended"):
        m = res.metrics[fs]
        assert m["split"]["strategy"] == "temporal" and m["split"]["embargo_removed"]["train"] > 0 and m["experiment"]["name"] == RUN_LABELS[fs]
        s = pd.read_csv(res.runs[fs] / "splits.csv", parse_dates=["index_date"])
        assert s[s.split == "train"]["index_date"].max() < s[s.split == "validation"]["index_date"].min() < s[s.split == "test"]["index_date"].min()
        assert s[s.split == "test"]["index_date"].min() > pd.Timestamp(res.design.validation_end)
    ds_s = pd.read_parquet(res.datasets["strict"] / "modeling_dataset.parquet")
    ds_e = pd.read_parquet(res.datasets["extended"] / "modeling_dataset.parquet")
    assert len(ds_s) == len(ds_e) and (ds_s["research_id"].to_numpy() == ds_e["research_id"].to_numpy()).all() and (ds_s["fall_next_180d"].to_numpy() == ds_e["fall_next_180d"].to_numpy()).all()
    assert not ds_e.duplicated(subset=["research_id", "index_date"]).any()
    # temporal test performance by snapshot with the INSUFFICIENT EVENTS rule
    bs = res.by_snapshot["extended"]
    assert set(bs["status"]) <= {"ok", "INSUFFICIENT EVENTS"} and (bs.loc[bs.status == "ok", "n_events"] >= 10).all()
    assert (out / "temporal_test_by_snapshot.md").exists() and (out / "audit" / "snapshot_audit.md").exists() and (out / "split_audit.md").exists()
    # patient-disjoint sensitivity: no test patient in train/validation, same rows, both feature sets on identical test rows
    assert set(res.sensitivity_runs) == {"strict", "extended"} and res.sensitivity_comparison["same_test_rows"]
    for fs in ("strict", "extended"):
        s = pd.read_csv(res.sensitivity_runs[fs] / "splits.csv")
        by = {k: set(v) for k, v in s.groupby("split")["research_id"]}
        assert not (by["test"] & (by["train"] | by["validation"])) and res.sensitivity_metrics[fs]["experiment"]["name"] == SENSITIVITY_LABELS[fs]
        assert len(s) == len(ds_e)   # every row of the dataset is assigned (no embargo in the patient-grouped design)
    # reports: watermark, funnel, hashes, no identifiers / pepper
    summary = (out / "SUMMARY.md").read_text(encoding="utf-8")
    assert "NOT EFALLS REPRODUCTION" in summary and "Temporal design (primary" in summary and "Patient-disjoint sensitivity" in summary and "Row funnel" in summary
    assert "D-00 timing" in summary and "Known mapping limitations" in summary and "significan" not in summary.lower()
    pepper = (out / "id_pepper.txt").read_text(encoding="utf-8").strip()
    text = shared_text(out)
    assert not ID_RE.search(text) and pepper not in text
    fr = pd.read_csv(out / "feature_report_extended.csv")
    assert {"efalls_concept", "feature_sets", "non_zero_coefficient", "selection_frequency", "mapping_quality"} <= set(fr.columns)
    assert (out / "READINESS.md").read_text(encoding="utf-8").startswith("# READY TO TRAIN")
