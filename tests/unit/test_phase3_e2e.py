"""Phase 3 end to end on tiny SYNTHETIC extracts (slow): a reference explore run, then ``meuhedet-phase3`` with a reduced configuration.

Proves: the time-contract verification and cohort reconciliation; the GO path (feature sets, models on identical FULL_LABELED patients, frozen
selection before a write-once VALIDATION_OPENED.json, proxy ablation, robustness verdicts, reports, share with fail-closed privacy / path / file-type
scans and no row-level file); the preflight fits nothing and writes nothing; an abrupt kill (os._exit) inside an XGBoost trial and before the
share rename, resumed, gives byte-identical final tables; a contradicted time contract (index-day outcomes) stops before any model; an extract
whose sources carry post-index records everywhere ends in Outcome B (NO_GO, DWH report, no model)."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.slow
PEPPER = "synthetic-phase2-test-pepper-03"


def _xgb_ok() -> bool:
    try:
        import optuna  # noqa: F401
        import xgboost  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _world(base: Path, mode: str, seed: int = 3) -> dict[str, Any]:
    from falls_ml.data.meuhedet_synthetic import generate_synthetic_wide_extract
    from falls_ml.phase3.synthetic import plant_phase3_history, write_phase3_csv

    w = base / mode
    w.mkdir(parents=True)
    df = generate_synthetic_wide_extract(3500, seed=11, index_date="2025-01-01", n_index_day_falls=4)
    unr: dict[str, int] = {}
    if mode == "go":
        df, _ = plant_phase3_history(df, seed=seed, same_day={"MiniCog_Date": 0.02, "Get_Up_And_Go_Date": 0.015, "Last_Visit_Date": 0.08},
                                     undated={"Home_Safety_Assessment_Date": 0.01})
        unr = {"Get_Up_And_Go_Date": 3}
    elif mode == "contradict":
        # seed 4: the small Phase 1 reference split under seed 3 is degenerate for recalibration (synthetic artefact, not a Phase 3 path)
        df, _ = plant_phase3_history(df, seed=4, index_day_outcomes=True)
    elif mode == "nogo":
        fut = {c: 0.4 for c in ("MiniCog_Date", "Get_Up_And_Go_Date", "Falls_Risk_Assessment_Date", "Home_Safety_Assessment_Date", "Mobility_Assessment_Date",
                                "MEFI_From_Date", "Last_Visit_Date", "Last_Assessment_Date")}
        df, _ = plant_phase3_history(df, seed=seed, future=fut)
    csv = write_phase3_csv(df, w / "x.csv", n_unreadable=unr)
    (w / "x.csv.id_pepper.txt").write_text(f"# falls_ml pseudonym\n{PEPPER}\n", encoding="utf-8", newline="\n")
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    subprocess.run([sys.executable, "-m", "falls_ml", "meuhedet-explore", "--input", str(csv), "--out", str(w / "explore"), "--index-date", "2025-01-01",
                    "--index-day-records", "drop_rows", "--model-report", "off"], check=True, env=env, cwd=ROOT, capture_output=True)
    cfg = yaml.safe_load((ROOT / "configs/meuhedet/phase3.yaml").read_text(encoding="utf-8"))
    p = cfg["phase3"]
    p["cv"] = {"outer_folds": 3, "inner_folds_linear": 3, "inner_folds_xgb": 3}
    p["xgb"]["stage1"].update({"n_trials": 3, "n_startup_trials": 2})
    p["xgb"].update({"n_estimators_max": 120, "early_stopping_rounds": 20})
    p["lasso"]["n_lambda"] = 15
    p["enet"].update({"n_lambda": 12, "l1_ratios": [0.5, 0.9]})
    p["stability"]["n_replicates"] = 2
    p["ablation"]["individual_top_n"] = 2
    p["metrics"]["bootstrap_n"] = 30
    p["explain"].update({"shap_sample_rows": 600, "permutation_repeats": 1})
    p["eligibility"]["min_observed_train_rows"] = 20
    p["feasibility"].update({"min_domain_known_informative_train_rows": 50, "min_domain_known_informative_train_events": 5})
    for k, v in (("features", "configs/meuhedet/phase2_features.yaml"), ("recovery", "configs/meuhedet/phase3_recovery.yaml"),
                 ("d00_config", "configs/meuhedet/d00_sensitivity.yaml"), ("time_contract", "configs/meuhedet/phase3_time_contract.yaml")):
        p[k] = str(ROOT / v)
    (w / "cfg.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8", newline="\n")
    return {"w": w, "csv": csv, "ref": w / "explore", "cfg": w / "cfg.yaml", "env": env}


@pytest.fixture(scope="module")
def base(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if not _xgb_ok():
        pytest.skip("xgboost / optuna runtime not available")
    return tmp_path_factory.mktemp("p3")


@pytest.fixture(scope="module")
def go(base: Path) -> dict[str, Any]:
    return _world(base, "go")


def _cli(world: dict[str, Any], out: Path, *extra: str, crash: str | None = None, cmd: str = "meuhedet-phase3") -> subprocess.CompletedProcess[str]:
    env = dict(world["env"])
    if crash:
        env["FALLS_ML_PHASE2_CRASH_AT"] = crash
    args = [sys.executable, "-m", "falls_ml", cmd, "--out", str(out)]
    if cmd == "meuhedet-phase3":
        args += ["--input", str(world["csv"]), "--reference", str(world["ref"]), "--config", str(world["cfg"]), "--allow-unfrozen-config", *extra]
    return subprocess.run(args, env=env, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")


def _complete(world: dict[str, Any], out: Path, *first: str, crash: str | None = None) -> subprocess.CompletedProcess[str]:
    r = _cli(world, out, *first, crash=crash)
    while r.returncode == 2 and "INVESTIGATION STOP" in r.stderr:
        gate = r.stderr.split("[", 1)[1].split("]", 1)[0]
        r = _cli(world, out, "--resume", "--accept-gate", gate, "--reason", "synthetic test data (planted signal)", crash=crash)
    return r


@pytest.fixture(scope="module")
def run_a(go: dict[str, Any]) -> Path:
    out = go["w"] / "A"
    r = _complete(go, out)
    assert r.returncode == 0, r.stderr[-4000:]
    return out


def test_preflight_fits_nothing_and_writes_nothing(go: dict[str, Any]) -> None:
    out = go["w"] / "never"
    r = _cli(go, out, "--preflight")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
    lines = [x for x in r.stdout.splitlines() if x.strip()]
    assert lines[-1] == "SAFE TO START FULL RUN"
    assert "V1 outcome excludes Index_Date" in r.stdout and "SCIENTIFIC FORECAST: GO" in r.stdout and "V9 cohort reconciliation" in r.stdout
    assert not out.exists()


def test_go_run_outputs_share_and_invariants(run_a: Path) -> None:
    share = run_a / "share"
    for f in ("README.md", "RUN_MANIFEST.json", "PRIVACY_SCAN.json", "SCIENTIFIC_SUMMARY.md", "SCIENTIFIC_SUMMARY_HE.md", "MANAGEMENT_SUMMARY_HE.md",
              "DWH_REMEDIATION.md", "FEASIBILITY_DECISION.json", "TIME_CONTRACT_VERIFICATION.json", "COHORT_FACTS.json", "SELECTION_FROZEN.json",
              "VALIDATION_OPENED.json", "ROBUSTNESS.json", "PHASE3_FINAL_EXPERIMENT_CONFIG.json", "tables/FEATURE_RECOVERY.csv", "tables/OOF_MODEL_COMPARISON.csv",
              "tables/PROXY_ABLATION_OOF.csv", "tables/DOMAIN_FEASIBILITY.csv", "tables/DWH_REMEDIATION_REQUIREMENTS.csv", "docs/PHASE3_TIME_CONTRACT_AUDIT.md"):
        assert (share / f).is_file(), f
    assert json.loads((share / "PRIVACY_SCAN.json").read_text(encoding="utf-8"))["passed"]
    assert not [p for p in share.rglob("*") if p.suffix in (".parquet", ".pkl", ".npz", ".sqlite", ".jsonl")]
    for p in share.rglob("*.csv"):
        assert "research_id" not in pd.read_csv(p, nrows=0).columns, p
    m = json.loads((share / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    assert m["decision"] == "GO" and m["test"]["outcomes_read"] is False and m["synthetic"] is True and m["frozen_production_config"] is False
    tc = json.loads((share / "TIME_CONTRACT_VERIFICATION.json").read_text(encoding="utf-8"))
    assert tc["hard_passed"] and tc["contract"]["prediction_time"] == "END_OF_INDEX_DAY"
    fs = json.loads((run_a / "artifacts" / "FEATURE_SETS.json").read_text(encoding="utf-8"))
    reg = pd.read_csv(run_a / "artifacts" / "FEATURE_RECOVERY.csv").set_index("feature")
    for name, s in fs["sets"].items():
        if s["category"] == "PRIMARY_FULL":
            assert all(reg.at[f, "phase3_class"] in ("SAFE_VERIFIED", "SAFE_VERIFIED_BOUNDED", "SAFE_ATTESTED") for f in s["features"] + s["baseline"]), name
        if s["category"] == "SENSITIVITY_VERIFIED_ONLY":
            assert all(reg.at[f, "phase3_class"] in ("SAFE_VERIFIED", "SAFE_VERIFIED_BOUNDED") for f in s["features"] + s["baseline"]), name
    assert "P3_ALL_NO_FALL_RECENCY" in fs["sets"] and "falls_days_since_last" not in fs["sets"]["P3_ALL_NO_FALL_RECENCY"]["features"]
    oof = pd.read_csv(run_a / "artifacts" / "OOF_MODEL_COMPARISON.csv")
    assert oof["n"].nunique() == 1 if "n" in oof else True
    assert oof.loc[~oof["failed"].astype(bool), "adverse_n"].nunique() == 1     # every configuration on the SAME patients
    ok = oof[~oof["failed"].astype(bool)]
    assert (ok["adverse_ap"] <= ok["favourable_ap"] + 1e-12).all()
    prox = pd.read_csv(run_a / "artifacts" / "PROXY_ABLATION_OOF.csv")
    assert set(prox["family"]) == {"LASSO", "XGB_DEFAULT"} and prox["same_rows"].all()
    for text in ("SCIENTIFIC_SUMMARY.md", "MANAGEMENT_SUMMARY_HE.md"):
        t = (share / text).read_text(encoding="utf-8")
        assert "significant" not in t.lower() and "מובהק" not in t


def test_fit_rows_never_hold_unknown_values(run_a: Path) -> None:
    fits = pd.read_csv(run_a / "artifacts" / "FITS_LASSO.csv")
    cf = json.loads((run_a / "artifacts" / "COHORT_FACTS.json").read_text(encoding="utf-8"))
    n_train = cf["phase3_partitions"]["train"]["n_rows"]
    assert fits.loc[fits["fold"] == "final", "n_fit_rows"].max() <= n_train
    reg = pd.read_csv(run_a / "artifacts" / "FEATURE_RECOVERY.csv")
    assert reg["n_unknown"].sum() > 0          # the planted undated / unreadable rows exist ...
    assert fits.loc[fits["fold"] == "final", "n_fit_rows"].max() < n_train     # ... and are never fitted


def test_status_is_read_only(run_a: Path, go: dict[str, Any]) -> None:
    before = {str(p): p.stat().st_mtime_ns for p in run_a.rglob("*") if p.is_file()}
    r = _cli(go, run_a, cmd="meuhedet-phase3-status")
    assert r.returncode == 0 and "Phase 3 run status" in r.stdout and "GO" in r.stdout
    assert before == {str(p): p.stat().st_mtime_ns for p in run_a.rglob("*") if p.is_file()}


def test_kill_and_resume_give_identical_tables(run_a: Path, go: dict[str, Any]) -> None:
    out = go["w"] / "B"
    r = _cli(go, out, crash="S09_xgb|XGB1_outer1__t01|item_start")
    assert r.returncode != 0
    r = _complete(go, out, "--resume", crash="S16_share|share|before_rename")
    assert r.returncode != 0 and not (out / "share").exists()
    r = _complete(go, out, "--resume")
    assert r.returncode == 0, r.stderr[-3000:]
    for t in ("OOF_MODEL_COMPARISON.csv", "OOF_BOOTSTRAP_VS_P3_BASE.csv", "PROXY_ABLATION_OOF.csv", "FEATURE_RECOVERY.csv", "VALIDATION_MODEL_COMPARISON.csv"):
        assert _sha(run_a / "artifacts" / t) == _sha(out / "artifacts" / t), t
    audit = json.loads((out / "RESUME_AUDIT.json").read_text(encoding="utf-8"))
    assert len(audit["attempts"]) >= 3


def test_contradicted_time_contract_stops_before_any_model(base: Path) -> None:
    w = _world(base, "contradict")
    out = w["w"] / "C"
    r = _cli(w, out)
    assert r.returncode == 2 and "TIME_CONTRACT_CONTRADICTED" in r.stderr
    assert not (out / "stages" / "S07_lasso").exists() and not (out / "stages" / "S06_feature_sets").exists()
    pf = _cli(w, w["w"] / "never", "--preflight")
    assert pf.returncode == 2 and "NOT SAFE TO START FULL RUN" in pf.stdout


def test_post_index_records_everywhere_end_in_outcome_b(base: Path) -> None:
    w = _world(base, "nogo")
    out = w["w"] / "D"
    r = _complete(w, out)
    assert r.returncode == 0, r.stderr[-3000:]
    res = json.loads(r.stdout[r.stdout.index("{"):])
    assert res["status"] == "COMPLETE_OUTCOME_B_NO_GO"
    assert not (out / "stages" / "S07_lasso").exists()
    share = out / "share"
    assert (share / "DWH_REMEDIATION.md").is_file() and json.loads((share / "PRIVACY_SCAN.json").read_text(encoding="utf-8"))["passed"]
    feas = json.loads((share / "FEASIBILITY_DECISION.json").read_text(encoding="utf-8"))
    assert feas["decision"] == "NO_GO"
    tc = json.loads((share / "TIME_CONTRACT_VERIFICATION.json").read_text(encoding="utf-8"))
    assert tc["attestation_holds"] is False
    reg = pd.read_csv(share / "tables" / "FEATURE_RECOVERY.csv")
    assert not reg["phase3_class"].eq("SAFE_ATTESTED").any()        # the attestation is withdrawn when post-index records exist
