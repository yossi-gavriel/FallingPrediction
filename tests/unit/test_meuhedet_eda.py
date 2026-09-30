"""EDA layer (falls_ml meuhedet-eda). Synthetic fixtures only - no real data."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from falls_ml.cli import main as cli_main
from falls_ml.data.meuhedet_synthetic import write_synthetic_wide_extract
from falls_ml.data.meuhedet_wide import DEFAULT_CONTRACT, DEFAULT_MAPPING, build_meuhedet_dataset_from_frame, load_wide_contract, load_wide_mapping, read_wide_extract_report
from falls_ml.eda.common import read_csv, wilson
from falls_ml.eda.dictionary import DEFAULT_DICTIONARY, load_data_dictionary
from falls_ml.eda.missingness import declared_class, missingness_outcome
from falls_ml.eda.runner import run_eda
from falls_ml.eda.supervised import TrainPartition
from falls_ml.errors import ConfigError, DatasetValidationError, LeakageError
from falls_ml.features.spec import load_feature_spec

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "configs" / "experiments" / "meuhedet" / "explore_180d_template.yaml"
WM = "EXPLORATORY 180-DAY OUTCOME – NOT EFALLS REPRODUCTION"
ID_RE = re.compile(r"\bS\d{10}\b")
REQUIRED = ["column_profile.csv", "data_dictionary.csv", "missingness.csv", "data_quality_findings.csv", "numeric_profile.csv", "categorical_profile.csv",
            "table1.csv", "correlation_pairs.csv", "feature_provenance.csv", "temporal_audit.csv", "current_15_feature_dictionary.csv",
            "phase2_candidate_features.csv", "cohort_flow.csv", "outcome_prevalence.csv", "split_balance.csv", "univariate_association_train.csv",
            "missingness_outcome_train.csv", "table1_train.csv", "redundancy_clusters.csv", "data_quality_checks.csv"]


@pytest.fixture(scope="module")
def contract():
    return load_wide_contract(DEFAULT_CONTRACT)


@pytest.fixture(scope="module")
def mapping(contract):
    return load_wide_mapping(DEFAULT_MAPPING, contract=contract)


# ------------------------------------------------------------------ definitions
def test_dictionary_covers_every_contract_column_and_refuses_gaps(contract, tmp_path) -> None:
    d = load_data_dictionary(DEFAULT_DICTIONARY, contract)
    assert list(d.columns) == contract.names and len(d.columns) == 221
    assert {v["status"] for v in d.columns.values()} <= {"DOCUMENTED", "NAME_ONLY", "UNKNOWN"}
    assert d.record_date("Prior_Fall_Count_365D") == "Last_Fall_Date" and d.record_date("Distinct_Active_Substance_Count") is None
    raw = yaml.safe_load((ROOT / DEFAULT_DICTIONARY).read_text(encoding="utf-8"))
    del raw["columns"]["Vitamin_D_Ind"]
    bad = tmp_path / "dict.yaml"
    bad.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ConfigError, match="Vitamin_D_Ind"):
        load_data_dictionary(bad, contract)


def test_null_classes_follow_the_declared_meaning_never_zero(contract) -> None:
    assert declared_class(contract, "Mobility_Score") == "NOT_MEASURED"
    assert declared_class(contract, "Days_Since_Last_Fall") == "TRUE_CLINICAL_ABSENCE"
    assert declared_class(contract, "Prescription_Fill_Ratio_365D") == "NOT_APPLICABLE"
    assert declared_class(contract, "Distinct_Active_Substance_Count") == "SOURCE_UNAVAILABLE"
    assert declared_class(contract, "Age_At_Index") == "TECHNICAL_FAILURE"
    assert declared_class(contract, "Last_Hosp_Date") == "TRUE_CLINICAL_ABSENCE"   # via its Days_Since companion's declaration


def test_train_partition_refuses_validation_and_test_rows() -> None:
    f = pd.DataFrame({"x": [1, 2, 3], "Fall_Next_180D_Ind": [0, 1, 0]})
    with pytest.raises(LeakageError):
        TrainPartition(f, np.array([0, 1, 0]), pd.Series(["train", "test", "train"]))
    tp = TrainPartition.from_assignment(f, pd.Series(["train", "validation", "train"], dtype="string"), "Fall_Next_180D_Ind")
    assert tp.n == 2
    with pytest.raises(TypeError):
        missingness_outcome(f, ["x"], None, None, 10)   # a plain frame is not accepted for a target-aware analysis


def test_wilson_interval() -> None:
    p, lo, hi = wilson(10, 100)
    assert p == 0.1 and 0.05 < lo < 0.1 < hi < 0.18


# ------------------------------------------------------------------ end to end
def _split_test_ids(csv: Path, tmp: Path, mapping, contract) -> tuple[set[str], str, str]:
    """The split the EDA must reproduce, computed independently: (Customer_Full_ID of test rows, test sha256, pepper)."""
    from falls_ml.config import load_experiment_config
    from falls_ml.data.dataset import ModelingDataset
    from falls_ml.meuhedet_explore import _write_config, resolve_id_pepper
    from falls_ml.splitting import make_split

    spec = load_feature_spec(mapping.exploratory_spec_path)
    ext = mapping.feature_sets()["extended"]
    read = read_wide_extract_report(csv, contract)
    tmp.mkdir(parents=True, exist_ok=True)
    pepper, _info = resolve_id_pepper(csv, tmp, explicit=None, pepper_file=None)
    build_meuhedet_dataset_from_frame(read.frame, tmp / "ds", index_dates=["2025-01-01"], dataset_version="t", data_freeze_date=None, mapping=mapping,
                                      contract=contract, spec=spec, input_name=csv.name, input_sha256="x", features=ext, feature_set_label="extended",
                                      id_pepper=pepper, index_day_records="drop_rows")
    cfg = load_experiment_config(_write_config(TEMPLATE, tmp / "c.yaml", name="t", description="t", features=ext, fast=True, spec=spec))
    ds = ModelingDataset.load(tmp / "ds", spec, features=ext)
    plan = make_split(ds.frame, ds.spec, cfg.validation)
    test_pseudo = set(ds.frame.iloc[plan.test_idx]["research_id"].astype(str))
    ids = set(read.frame["Customer_Full_ID"].astype(str))
    test_ids = {i for i in ids if hashlib.sha256(f"{pepper}|{i}".encode()).hexdigest()[:20] in test_pseudo}
    return test_ids, plan.test_rows_sha256(ds.frame, ds.spec), pepper


@pytest.fixture(scope="module")
def eda_run(tmp_path_factory, mapping, contract):
    tmp = tmp_path_factory.mktemp("eda")
    csv = write_synthetic_wide_extract(tmp / "wide.csv", n_rows=2500, seed=5, other_index_date_share=0.0, csv_date_format="%d/%m/%Y", n_index_day_falls=4)
    raw = pd.read_csv(csv, dtype=str, keep_default_na=False)
    elig = (raw["Is_Eligible_Cohort"] == "1") & raw["Fall_Next_180D_Ind"].isin(["0", "1"])
    dx = raw.index[elig & (raw["Last_Dx_Date"] != "NULL")][:30]
    raw.loc[dx, "Last_Dx_Date"] = raw.loc[dx, "Index_Date"]
    raw.loc[raw.index[raw["Last_Hosp_Date"] != "NULL"][:12], "Last_Hosp_Date"] = "00:00.0"
    raw.loc[raw.index[:11], "Visit_Count_30D"] = "-1"
    raw.loc[raw.index[40:52], "Snapshot_Confidence"] = "BAD"
    raw.to_csv(csv, index=False, lineterminator="\n")
    test_ids, test_sha, pepper = _split_test_ids(csv, tmp / "split", mapping, contract)
    # leak detector: a column that equals the outcome on TEST rows only (0 everywhere else); the split does not read it
    raw = pd.read_csv(csv, dtype=str, keep_default_na=False)
    on_test = raw["Customer_Full_ID"].isin(test_ids)
    raw["Vitamin_D_Ind"] = np.where(on_test & (raw["Fall_Next_180D_Ind"] == "1"), "1", "0")
    raw.to_csv(csv, index=False, lineterminator="\n")
    res = run_eda(csv, out_dir=tmp / "out", index_day_records="drop_rows", min_cell=5)
    return {"res": res, "out": tmp / "out", "csv": csv, "test_sha": test_sha, "pepper": pepper, "n_dx": len(dx), "n_test_ids": len(test_ids), "tmp": tmp}


def test_eda_writes_every_output_with_watermark_and_basis(eda_run) -> None:
    out, res = eda_run["out"], eda_run["res"]
    for name in REQUIRED:
        assert (out / "eda" / name).exists(), name
        first = (out / "eda" / name).read_text(encoding="utf-8").splitlines()[0]
        assert first.startswith(WM) and "basis:" in first
    for name in ("REAL_DATA_EDA_REPORT.html", "REAL_DATA_EDA_SUMMARY.md", "DATA_QUALITY_REPORT.md"):
        text = (out / name).read_text(encoding="utf-8")
        assert WM in text and "SYNTHETIC" in text
    assert len(list((out / "eda" / "plots").rglob("*.png"))) > 30
    for stage in ("RAW EXTRACT", "CONTRACT CHECK", "FULL EDA", "LEAKAGE / TIMING AUDIT", "COHORT BUILD", "SPLIT", "TRAIN-ONLY SUPERVISED EDA", "ADEQUACY"):
        assert res.stages[stage]["status"] == "DONE", (stage, res.stages[stage])
    assert res.stages["MODELLING"]["status"] == "NEXT"
    dd = read_csv(out / "eda" / "data_dictionary.csv")
    assert len(dd) == 221 and int(dd["used_by_extended"].sum()) == 16 and int(dd["used_by_strict"].sum()) == 10
    assert len(read_csv(out / "eda" / "current_15_feature_dictionary.csv")) == 15
    p2 = read_csv(out / "eda" / "phase2_candidate_features.csv")
    assert len(p2) > 40 and p2["trained"].str.startswith("NO").all()


def test_eda_split_is_the_modelling_split_and_test_rows_never_reach_target_aware_tables(eda_run) -> None:
    out, res = eda_run["out"], eda_run["res"]
    man = json.loads((out / "eda_manifest.json").read_text(encoding="utf-8"))
    assert man["split"]["test_rows_sha256"] == eda_run["test_sha"]
    uni = read_csv(out / "eda" / "univariate_association_train.csv").set_index("column")
    # Vitamin_D_Ind equals the outcome on test rows and is 0 elsewhere: on TRAIN it is constant, so no association can appear
    v = uni.loc["Vitamin_D_Ind"]
    assert (pd.isna(v["smd_fall_vs_no_fall"]) or abs(float(v["smd_fall_vs_no_fall"])) < 1e-9) and float(v["univariate_auroc"]) == 0.5
    cat = read_csv(out / "eda" / "categorical_outcome_train.csv")
    assert set(pd.to_numeric(cat.loc[cat["column"] == "Vitamin_D_Ind", "level"]).astype(int)) == {0}
    t1 = read_csv(out / "eda" / "table1_train.csv")
    n_train = man["split"]["partitions"]["train"]["n_rows"]
    assert t1.loc[t1["variable"] == "N (rows)", "total"].astype(int).iloc[0] == n_train


def test_eda_finds_the_planted_defects_and_the_d00_rows(eda_run) -> None:
    out = eda_run["out"]
    dq = read_csv(out / "eda" / "data_quality_findings.csv")
    b01 = dq[(dq["check_id"] == "B01") & (dq["columns"] == "Last_Hosp_Date")]
    assert len(b01) == 1 and "99:99.9" in b01["detail"].iloc[0] and "00:00.0" in b01["detail"].iloc[0]
    assert ((dq["check_id"] == "D01") & (dq["columns"] == "Visit_Count_30D")).any()
    assert ((dq["check_id"] == "B03") & (dq["columns"] == "Snapshot_Confidence")).any()
    ta = read_csv(out / "eda" / "temporal_audit.csv").set_index("column")
    assert ta.at["Last_Dx_Date", "leakage_risk"] == "PROVEN_INDEX_DAY_OR_FUTURE_RECORDS"
    flow = read_csv(out / "eda" / "cohort_flow.csv")
    d00 = flow[flow["stage"].str.startswith("D-00")]
    assert int(d00["rows_removed_at_this_step"].iloc[0]) >= eda_run["n_dx"]
    assert all(c["agree"] for c in eda_run["res"].tables["cohort_flow"].attrs["build_crosscheck"])
    miss = read_csv(out / "eda" / "missingness.csv").set_index("column")
    assert int(miss.at["Last_Hosp_Date", "n_technical_failure"]) == 12
    assert miss.at["Mobility_Score", "dominant_class"] == "NOT_MEASURED"


def test_eda_outputs_hold_no_identifiers_pseudonyms_or_pepper(eda_run) -> None:
    out = eda_run["out"]
    text = "".join(p.read_text(encoding="utf-8", errors="ignore") for p in out.rglob("*") if p.suffix in {".csv", ".md", ".json", ".html"})
    text = re.sub(r"data:image/png;base64,[A-Za-z0-9+/=]+", "", text)
    assert not ID_RE.search(text) and "SYN_" not in text and eda_run["pepper"] not in text
    raw = pd.read_csv(eda_run["csv"], dtype=str, keep_default_na=False)
    pseudo = {hashlib.sha256(f"{eda_run['pepper']}|{i}".encode()).hexdigest()[:20] for i in raw["Customer_Full_ID"]}
    assert not (set(re.findall(r"[0-9a-f]{20}", text)) & pseudo)
    man = json.loads((out / "eda_manifest.json").read_text(encoding="utf-8"))
    assert man["privacy"]["identifier_scan"]["passed"] is True and not (out / "id_pepper.txt").exists()
    assert man["split"]["temporary_dataset_deleted"] is True


def test_eda_cli_refuses_a_non_empty_output_and_d00_without_policy(eda_run, tmp_path) -> None:
    assert cli_main(["meuhedet-eda", "--input", str(eda_run["csv"]), "--out", str(eda_run["out"])]) == 2
    res = run_eda(eda_run["csv"], out_dir=tmp_path / "fail_policy", make_plots=False)
    assert res.stages["COHORT BUILD"]["status"] == "STOPPED" and "drop_rows" in res.stages["COHORT BUILD"]["detail"]
    assert res.stages["TRAIN-ONLY SUPERVISED EDA"]["status"] == "NOT RUN" and "univariate_association_train" not in res.tables
    assert (tmp_path / "fail_policy" / "REAL_DATA_EDA_REPORT.html").exists()


@pytest.mark.slow
def test_eda_reproduces_a_completed_reference_run_and_never_writes_into_it(eda_run, tmp_path) -> None:
    from falls_ml.meuhedet_explore import explore
    from falls_ml.meuhedet_sensitivity import directory_digest

    tpl = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
    tpl["evaluation"]["bootstrap"]["n"] = 20
    tpl["evaluation"]["permutation_importance_repeats"] = 1
    tpl["analysis"]["stability"]["n_bootstrap"] = 4
    tpl["analysis"]["optimism"]["n_bootstrap"] = 2
    (tmp_path / "t.yaml").write_text(yaml.safe_dump(tpl, sort_keys=False), encoding="utf-8")
    ref = tmp_path / "explore"
    explore(eda_run["csv"], out_dir=ref, index_day_records="drop_rows", template_path=tmp_path / "t.yaml", fast=False, model_report="off")
    before = directory_digest(ref)["combined_sha256"]
    res = run_eda(eda_run["csv"], out_dir=tmp_path / "eda", reference_dir=ref, make_plots=False)
    man = json.loads((tmp_path / "eda" / "eda_manifest.json").read_text(encoding="utf-8"))
    ref_split = json.loads((ref / "split_audit.json").read_text(encoding="utf-8"))
    assert man["split"]["matches_reference"] is True and man["split"]["test_rows_sha256"] == ref_split["test_rows_sha256"]
    assert man["reference"]["integrity"]["unchanged"] is True and directory_digest(ref)["combined_sha256"] == before
    assert res.policy == "drop_rows" and res.stages["SPLIT"]["detail"].endswith("= reference run")
    cur = read_csv(tmp_path / "eda" / "eda" / "current_15_feature_dictionary.csv")
    assert cur["lasso_coefficient_extended"].notna().all() and not cur["lasso_coefficient_extended"].astype(str).str.startswith("not available").any()
    foreign = tmp_path / "foreign_pepper.txt"
    foreign.write_text("# test\nnot-the-reference-pepper\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="pepper"):
        run_eda(eda_run["csv"], out_dir=tmp_path / "eda2", reference_dir=ref, id_pepper_file=foreign, make_plots=False)
    other = tmp_path / "other.csv"
    other.write_text(eda_run["csv"].read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="sha256"):
        run_eda(other, out_dir=tmp_path / "eda3", reference_dir=ref, make_plots=False)
    assert cli_main(["meuhedet-eda", "--input", str(eda_run["csv"]), "--reference", str(ref), "--out", str(ref / "eda_inside")]) == 2
