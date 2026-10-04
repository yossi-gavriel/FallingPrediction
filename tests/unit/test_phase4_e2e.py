"""Phase 4 end-to-end on SYNTHETIC data (slow): a synthetic Phase 3 run (frozen LASSO fits) on 2025 and a synthetic 2026 snapshot.

The brief's required proofs:
1. the score command cannot access labels             test_blind_score_never_reads_a_sealed_column_and_ignores_outcome_values
2. evaluation cannot run before frozen hashes exist    test_evaluation_refuses_before_scoring
3. a changed prediction file / hash blocks evaluation  test_changed_frozen_files_block_evaluation
4. a 2026-only feature cannot enter a frozen model     test_a_2026_only_feature_cannot_enter_a_frozen_model
5. a schema mismatch hard-stops                        test_schema_mismatch_hard_stops
6. an outcome-contract violation hard-stops            test_outcome_contract_violation_stops_before_any_metric
7. the same patient in 2025 / 2026 is counted right    test_patient_overlap_is_counted_correctly
8. the share contains no patient-level data            test_share_contains_no_patient_level_data
plus: mode B re-fit = persisted model, write-once scoring, read-only status, idempotent evaluation, the Phase 3 folder is never written.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

pytestmark = pytest.mark.slow
ROOT = Path(__file__).resolve().parents[2]
PEPPER = "synthetic-phase2-test-pepper-03"
NEW_COLS = ["Dizziness_Vertigo_Ind", "Cataract_Dx_Date", "Opiate_Registry_Ind", "Fall_Next_365D_Ind"]


def _tree_digest(d: Path) -> dict[str, str]:
    return {p.relative_to(d).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.rglob("*")) if p.is_file()}


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    from falls_ml.data.meuhedet_synthetic import generate_synthetic_wide_extract
    from falls_ml.phase3.runner import run_phase3
    from falls_ml.phase3.synthetic import plant_phase3_history, write_phase3_csv
    from falls_ml.phase4.synthetic import make_2026

    base = tmp_path_factory.mktemp("p4")
    w = base / "w25"
    w.mkdir()
    df = generate_synthetic_wide_extract(3500, seed=11, index_date="2025-01-01", n_index_day_falls=4)
    df, _ = plant_phase3_history(df, seed=3, same_day={"MiniCog_Date": 0.02, "Get_Up_And_Go_Date": 0.015, "Last_Visit_Date": 0.08},
                                 undated={"Home_Safety_Assessment_Date": 0.01})
    x25 = write_phase3_csv(df, w / "x.csv", n_unreadable={"Get_Up_And_Go_Date": 3})
    (w / "x.csv.id_pepper.txt").write_text(f"# falls_ml pseudonym\n{PEPPER}\n", encoding="utf-8", newline="\n")
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    subprocess.run([sys.executable, "-m", "falls_ml", "meuhedet-explore", "--input", str(x25), "--out", str(w / "explore"), "--index-date", "2025-01-01",
                    "--index-day-records", "drop_rows", "--model-report", "off"], check=True, env=env, cwd=ROOT, capture_output=True)
    cfg = yaml.safe_load((ROOT / "configs/meuhedet/phase3.yaml").read_text(encoding="utf-8"))
    p = cfg["phase3"]
    p["cv"] = {"outer_folds": 3, "inner_folds_linear": 3, "inner_folds_xgb": 3}
    p["lasso"]["n_lambda"] = 15
    p["eligibility"]["min_observed_train_rows"] = 20
    p["feasibility"].update({"min_domain_known_informative_train_rows": 50, "min_domain_known_informative_train_events": 5})
    for k, v in (("features", "configs/meuhedet/phase2_features.yaml"), ("recovery", "configs/meuhedet/phase3_recovery.yaml"),
                 ("d00_config", "configs/meuhedet/d00_sensitivity.yaml"), ("time_contract", "configs/meuhedet/phase3_time_contract.yaml")):
        p[k] = str(ROOT / v)
    cfg3 = w / "cfg3.yaml"
    cfg3.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8", newline="\n")
    c4 = yaml.safe_load((ROOT / "configs/meuhedet/phase4.yaml").read_text(encoding="utf-8"))
    c4["phase4"]["metrics"]["bootstrap_n"] = 200
    cfg4 = w / "cfg4.yaml"
    cfg4.write_text(yaml.safe_dump(c4, sort_keys=False, allow_unicode=True), encoding="utf-8", newline="\n")
    p3 = base / "p3"
    res = run_phase3(x25, w / "explore", out_dir=p3, config_path=cfg3, allow_unfrozen_config=True, stop_after="s07")
    assert res["status"] == "STOPPED_AFTER_s07" and res["decision"] == "GO"
    d26, facts = make_2026(df)
    y26 = write_phase3_csv(d26, base / "y26.csv")
    return {"base": base, "x25": x25, "y26": y26, "df25": df, "d26": d26, "facts": facts, "p3": p3, "cfg3": cfg3, "cfg4": cfg4,
            "p3_digest": _tree_digest(p3)}


def _pf(world: dict[str, Any], out: Path, csv: Path | None = None, p3: Path | None = None) -> int:
    from falls_ml.phase4.preflight import run_preflight

    return run_preflight(csv or world["y26"], p3 or world["p3"], out_dir=out, phase3_config=world["cfg3"], config_path=world["cfg4"],
                         allow_unfrozen_phase3=True, expected_definition_version="V21")


def _sc(world: dict[str, Any], out: Path, csv: Path | None = None) -> dict[str, Any]:
    from falls_ml.phase4.score import run_score

    return run_score(csv or world["y26"], world["x25"], world["p3"], out_dir=out, phase3_config=world["cfg3"], config_path=world["cfg4"],
                     allow_unfrozen_phase3=True, expected_definition_version="V21")


def _ev(world: dict[str, Any], out: Path, csv: Path | None = None) -> dict[str, Any]:
    from falls_ml.phase4.evaluate import run_evaluate

    return run_evaluate(csv or world["y26"], out_dir=out, config_path=world["cfg4"], phase3_config=world["cfg3"])


def _pred(out: Path) -> pd.DataFrame:
    return pd.read_parquet(out / "sealed" / "BLIND_PREDICTIONS.parquet")


@pytest.fixture(scope="module")
def main_run(world: dict[str, Any]) -> dict[str, Any]:
    from falls_ml.phase4 import sealed

    out = world["base"] / "A"
    sealed.READ_LOG.clear()
    assert _pf(world, out) == 0
    s = _sc(world, out)
    reads = list(sealed.READ_LOG)
    e = _ev(world, out)
    return {"out": out, "score": s, "eval": e, "reads": reads}


def _variant(world: dict[str, Any], name: str, fn: Any) -> Path:
    df = fn(pd.read_csv(world["y26"], dtype="string", keep_default_na=False))
    p = world["base"] / name
    df.to_csv(p, index=False, lineterminator="\n")
    return p


# ============================================================================ 1
def test_blind_score_never_reads_a_sealed_column_and_ignores_outcome_values(world: dict[str, Any], main_run: dict[str, Any]) -> None:
    from falls_ml.phase4.config import load_phase4_config
    from falls_ml.phase4.sealed import read_header, sealed_map
    from falls_ml.data.meuhedet_wide import load_wide_contract

    sm = sealed_map(read_header(world["y26"]), load_wide_contract(), load_phase4_config(world["cfg4"]))
    assert {"Fall_Next_180D_Ind", "Next_Fall_Date_180D", "Label_Reason_180D", "Followup_End_Date", "Fall_Next_365D_Ind"} <= set(sm)
    read_2026 = [cols for name, cols in main_run["reads"] if name == world["y26"].name]
    assert read_2026, "the 2026 file was read through the sealed reader"
    assert not {c for cols in read_2026 for c in cols} & set(sm), "a sealed column was requested before evaluation"
    man = json.loads((main_run["out"] / "sealed" / "BLIND_PREDICTIONS_MANIFEST.json").read_text(encoding="utf-8"))
    assert man["outcomes_read"] is False and set(man["sealed_columns_never_read"]) == set(sm)

    # the same file with every sealed column replaced by garbage gives byte-identical predictions
    rng = np.random.default_rng(0)

    def poison(df: pd.DataFrame) -> pd.DataFrame:
        for c in sm:
            df[c] = rng.choice(["NULL", "1", "0", "2030-01-01", "x"], len(df))
        return df

    pz = _variant(world, "y26_poisoned.csv", poison)
    out = world["base"] / "P"
    assert _pf(world, out, pz) == 0
    _sc(world, out, pz)
    a, b = _pred(main_run["out"]), _pred(out)
    assert a.drop(columns=["key"]).equals(b.drop(columns=["key"]))


# ============================================================================ 2
def test_evaluation_refuses_before_scoring(world: dict[str, Any]) -> None:
    from falls_ml.phase2.state import Phase2Stop

    out = world["base"] / "N"
    assert _pf(world, out) == 0
    with pytest.raises(Phase2Stop) as e:
        _ev(world, out)
    assert e.value.gate == "NOT_SCORED"
    assert not (out / "evaluation" / "OUTCOMES_OPENED.jsonl").exists() and not (out / "share").exists()


# ============================================================================ 3
def _scored_copy(src: Path, dst: Path) -> Path:
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("evaluation", "share", ".tmp-share", ".old-share-*"))
    for p in dst.rglob("*"):
        if p.is_file():
            os.chmod(p, stat.S_IWRITE | stat.S_IREAD)
    return dst


def test_changed_frozen_files_block_evaluation(world: dict[str, Any], main_run: dict[str, Any]) -> None:
    from falls_ml.phase2.state import Phase2Stop

    def tamper_pred(o: Path) -> None:
        t = _pred(o)
        c = [x for x in t.columns if x.startswith("lo__")][0]
        t.loc[0, c] = t.loc[0, c] + 0.01
        t.to_parquet(o / "sealed" / "BLIND_PREDICTIONS.parquet", index=False)

    def tamper_manifest(o: Path) -> None:
        p = o / "frozen" / "FROZEN_MODEL_MANIFEST.json"
        p.write_text(p.read_text(encoding="utf-8").replace('"primary": "LASSO:P3_BASE"', '"primary": "LASSO:P3_ALL_RECOVERED"'), encoding="utf-8")

    def tamper_model(o: Path) -> None:
        m = sorted((o / "frozen" / "models").glob("*.pkl"))[0]
        m.write_bytes(m.read_bytes() + b"\0")

    def tamper_hash_record(o: Path) -> None:
        p = o / "sealed" / "BLIND_PREDICTIONS_MANIFEST.json"
        j = json.loads(p.read_text(encoding="utf-8"))
        j["predictions_sha256"] = "0" * 64
        p.write_text(json.dumps(j), encoding="utf-8")

    for k, fn in enumerate((tamper_pred, tamper_manifest, tamper_model, tamper_hash_record)):
        o = _scored_copy(main_run["out"], world["base"] / f"T{k}")
        fn(o)
        with pytest.raises(Phase2Stop) as e:
            _ev(world, o)
        assert e.value.gate == "FROZEN_HASH_MISMATCH", (fn.__name__, e.value.gate)
        assert not (o / "evaluation" / "OUTCOMES_OPENED.jsonl").exists(), fn.__name__


# ============================================================================ 4
def test_a_2026_only_feature_cannot_enter_a_frozen_model(world: dict[str, Any], main_run: dict[str, Any]) -> None:
    flist = json.loads((main_run["out"] / "frozen" / "FEATURE_LIST.json").read_text(encoding="utf-8"))
    used = {f for m in flist.values() for f in [*m["new"], *m["baseline"]]}
    assert not used & set(NEW_COLS)
    cat = pd.read_csv(main_run["out"] / "preflight" / "NEW_2026_FEATURES_CATALOGUE.csv").set_index("column")
    assert cat.at["Dizziness_Vertigo_Ind", "status"].startswith("CATALOGUED ONLY") and bool(cat.at["Fall_Next_365D_Ind", "sealed"])
    # the strongly outcome-related new column changes nothing: predictions without the new columns are identical
    nonew = _variant(world, "y26_nonew.csv", lambda df: df.drop(columns=NEW_COLS))
    out = world["base"] / "W"
    assert _pf(world, out, nonew) == 0
    _sc(world, out, nonew)
    assert _pred(main_run["out"]).drop(columns=["key"]).equals(_pred(out).drop(columns=["key"]))
    # a frozen feature list edited to include a new column is refused (the Phase 3 commit record no longer matches)
    p3t = world["base"] / "p3_tampered"
    shutil.copytree(world["p3"], p3t)
    for p in p3t.rglob("*"):
        if p.is_file():
            os.chmod(p, stat.S_IWRITE | stat.S_IREAD)
    fsj = next(p3t.glob("stages/S06_feature_sets/items/*/FEATURE_SETS.json"))
    j = json.loads(fsj.read_text(encoding="utf-8"))
    j["sets"]["P3_BASE"]["features"].append("Dizziness_Vertigo_Ind")
    fsj.write_text(json.dumps(j), encoding="utf-8")
    assert _pf(world, world["base"] / "W2", p3=p3t) == 2


# ============================================================================ 5
def test_schema_mismatch_hard_stops(world: dict[str, Any], capsys: pytest.CaptureFixture[str]) -> None:
    from falls_ml.phase2.state import Phase2Stop

    flist = json.loads((world["base"] / "A" / "frozen" / "FEATURE_LIST.json").read_text(encoding="utf-8"))
    from falls_ml.phase4.common import load_definitions
    from falls_ml.phase4.features import feature_inputs

    L = load_definitions(world["cfg3"], world["cfg4"])
    f0 = flist["LASSO:P3_BASE"]["baseline"][0]
    col = feature_inputs(f0, L["cat"], L["mapping"])[0]
    def duplicate(df: pd.DataFrame) -> pd.DataFrame:
        el = df.index[(df["Is_Eligible_Cohort"] == "1") & df["Index_Date"].isin(["2026-01-01", "01/01/2026"])]
        df.loc[el[1], "Customer_Full_ID"] = df.loc[el[0], "Customer_Full_ID"]
        return df

    cases = {"absent_input": lambda df: df.drop(columns=[col]),
             "new_gender_code": lambda df: df.assign(Gender_Code=df["Gender_Code"].where(df.index % 7 != 0, "9")),
             "duplicate_patient": duplicate}
    for name, fn in cases.items():
        csv = _variant(world, f"y26_{name}.csv", fn)
        out = world["base"] / f"S_{name}"
        capsys.readouterr()
        assert _pf(world, out, csv) == 2, name
        assert capsys.readouterr().out.rstrip().endswith("STOP - TEMPORAL VALIDATION NOT DEFENSIBLE"), name
        with pytest.raises(Phase2Stop) as e:
            _sc(world, out, csv)
        assert e.value.gate in ("PREFLIGHT_NOT_SAFE", "NO_PREFLIGHT"), name
        assert not (out / "sealed").exists()


# ============================================================================ 6
def test_outcome_contract_violation_stops_before_any_metric(world: dict[str, Any]) -> None:
    from falls_ml.phase2.state import Phase2Stop

    def index_day_positives(df: pd.DataFrame) -> pd.DataFrame:        # an index-day event counted as the future outcome (contract O1)
        pos = df.index[df["Fall_Next_180D_Ind"] == "1"][:15]
        df.loc[pos, "Next_Fall_Date_180D"] = "2026-01-01"
        df.loc[pos, "Days_To_Next_Fall_180D"] = "0"
        return df

    bad = _variant(world, "y26_bad_outcome.csv", index_day_positives)
    out = world["base"] / "C"
    assert _pf(world, out, bad) == 0
    _sc(world, out, bad)
    with pytest.raises(Phase2Stop) as e:
        _ev(world, out, bad)
    assert e.value.gate == "OUTCOME_CONTRACT_FAILED"
    share = out / "share"
    oc = json.loads((share / "OUTCOME_CONTRACT_2026.json").read_text(encoding="utf-8"))
    assert oc["passed"] is False and any(h.startswith("O1") for h in oc["hard_failures"])
    assert not (share / "TEMPORAL_MODEL_COMPARISON.csv").exists() and not (out / "evaluation" / "TEMPORAL_MODEL_COMPARISON.csv").exists()
    assert json.loads((share / "PRIVACY_SCAN.json").read_text(encoding="utf-8"))["passed"]


# ============================================================================ 7
def test_patient_overlap_is_counted_correctly(world: dict[str, Any], main_run: dict[str, Any]) -> None:
    ov = json.loads((main_run["out"] / "sealed" / "COHORT_OVERLAP.json").read_text(encoding="utf-8"))
    f = world["facts"]
    assert ov["n_in_both"] == f["n_overlap_planted"] == 700
    assert ov["n_2026_cohort_eligible"] == f["n_2026_eligible"] and ov["n_2025_cohort_full_labeled"] == f["n_2025_full_labeled"]
    assert ov["n_2025_eligible_on_index_date"] == f["n_2025_eligible"]
    assert ov["n_new_in_2026"] == ov["n_2026_cohort_eligible"] - 700 and ov["n_2025_not_in_2026"] == ov["n_2025_cohort_full_labeled"] - 700
    assert int(_pred(main_run["out"])["seen_in_2025_cohort"].sum()) == 700 and ov["computed_before_outcomes"] is True


# ============================================================================ 8
def test_share_contains_no_patient_level_data(world: dict[str, Any], main_run: dict[str, Any]) -> None:
    share = main_run["out"] / "share"
    need = {"PHASE4_TEMPORAL_SUMMARY_HE.md", "PHASE4_TEMPORAL_SUMMARY.md", "TEMPORAL_MODEL_COMPARISON.csv", "TEMPORAL_CAPACITY.csv", "TEMPORAL_CALIBRATION.csv",
            "COHORT_OVERLAP.json", "SCHEMA_COMPARISON_2025_2026.csv", "PREDICTOR_SHIFT.csv", "OUTCOME_CONTRACT_2026.json", "NEW_2026_FEATURES_CATALOGUE.csv",
            "RUN_MANIFEST.json", "PRIVACY_SCAN.json", "TEMPORAL_VS_PHASE3.csv"}
    names = {p.name for p in share.iterdir()}
    assert need <= names and (share / "figures").is_dir()
    assert json.loads((share / "PRIVACY_SCAN.json").read_text(encoding="utf-8"))["passed"]
    files = [p for p in share.rglob("*") if p.is_file()]
    assert not [p for p in files if p.suffix.lower() in (".parquet", ".pkl", ".npz", ".jsonl", ".sqlite")]
    ids = set(world["d26"]["Customer_Full_ID"].astype(str)) | set(world["df25"]["Customer_Full_ID"].astype(str)) | set(_pred(main_run["out"])["key"])
    ids |= set(world["d26"]["Snapshot_Key"].astype(str))
    for p in files:
        if p.suffix.lower() in (".csv", ".md", ".json", ".svg"):
            text = p.read_text(encoding="utf-8", errors="ignore")
            assert not [i for i in ids if i in text], p.name
            assert str(world["base"]) not in text and "/Users/" not in text, p.name
        if p.suffix.lower() == ".csv":
            t = pd.read_csv(p)
            assert len(t) < 500 and not {"key", "Customer_Full_ID", "Snapshot_Key", "research_id"} & set(t.columns), p.name
    comp = pd.read_csv(share / "TEMPORAL_MODEL_COMPARISON.csv")
    assert set(comp["model"]) == {"LASSO:P3_BASE", "LASSO:P3_VERIFIED_ALL", "LASSO:P3_ALL_RECOVERED"}
    vs = pd.read_csv(share / "TEMPORAL_VS_PHASE3.csv")
    assert {"auroc", "ap", "brier", "capture_top10", "ppv_top10", "lift_top10", "prevalence", "n", "events"} <= set(vs.loc[vs["model"] == "LASSO:P3_BASE", "metric"])


# ============================================================================ further guarantees
def test_mode_b_refit_reproduces_the_persisted_phase3_model(world: dict[str, Any], tmp_path: Path) -> None:
    from falls_ml.phase4.phase3_source import Phase3Source

    src = Phase3Source(world["p3"], allow_unfrozen=True)
    for s in ("P3_BASE", "P3_VERIFIED_ALL"):
        a = src.persisted_model(s, "07_lasso")
        b = src.refit(s, config_path=world["cfg3"], workdir=tmp_path / s)
        assert b["mode"] == "B" and a["coefficients_sha256"] == b["coefficients_sha256"], s


def test_scoring_is_write_once_status_is_read_only_and_evaluation_idempotent(world: dict[str, Any], main_run: dict[str, Any]) -> None:
    from falls_ml.phase2.state import Phase2Stop
    from falls_ml.phase4.status import phase4_status

    out = main_run["out"]
    before = _tree_digest(out)
    assert phase4_status(out) == 0
    assert _tree_digest(out) == before
    with pytest.raises(Phase2Stop) as e:
        _sc(world, out)
    assert e.value.gate == "ALREADY_SCORED"
    assert _pf(world, out) == 2
    r = _ev(world, out)
    assert "already evaluated" in r.get("note", "")
    assert main_run["score"]["modes"] == {"LASSO:P3_BASE": "A", "LASSO:P3_VERIFIED_ALL": "A", "LASSO:P3_ALL_RECOVERED": "A"}


def test_zz_the_phase3_folder_was_never_written(world: dict[str, Any], main_run: dict[str, Any]) -> None:
    assert _tree_digest(world["p3"]) == world["p3_digest"]
