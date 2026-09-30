"""Phase 2 end to end on a tiny SYNTHETIC extract (slow): a completed reference explore run, then ``meuhedet-phase2`` with a reduced config.

Proves the funnel, the category invariants (SAFE_DISCOVERY = SAFE only), the frozen selection before a write-once VALIDATION_OPENED.json, the
share package and its privacy scan, the preflight (no fit, nothing written), the read-only status command, and abrupt kills (os._exit, like
a power loss) inside an Optuna trial, inside the one-shot validation and before the share rename, each resumed with byte-identical final
tables, no completed trial rerun, no committed file changed and the frozen configuration unchanged. Also: an accepted code change is logged."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.slow
PEPPER = "synthetic-phase2-test-pepper-03"      # pinned: the random-pepper split of this tiny extract can be degenerate for the reference fit


def _xgb_ok() -> bool:
    try:
        import optuna  # noqa: F401
        import xgboost  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    if not _xgb_ok():
        pytest.skip("xgboost / optuna runtime not available")
    from falls_ml.data.meuhedet_synthetic import generate_synthetic_wide_extract

    w = tmp_path_factory.mktemp("p2")
    df = generate_synthetic_wide_extract(3500, seed=11, index_date="2025-01-01", n_index_day_falls=4)
    idx = pd.Timestamp("2025-01-01")
    m = (df["Index_Date"] == idx) & (df["Is_Eligible_Cohort"] == 1) & df["Home_Safety_Assessment_Date"].notna() & df["Fall_Next_180D_Ind"].notna()
    df.loc[df.index[np.flatnonzero(m.to_numpy())[:10]], "Home_Safety_Assessment_Date"] = np.datetime64(idx, "us")
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].dt.strftime("%Y-%m-%d")
    csv = w / "x.csv"
    out.to_csv(csv, index=False, na_rep="NULL", lineterminator="\n")
    (w / "x.csv.id_pepper.txt").write_text(f"# falls_ml pseudonym\n{PEPPER}\n", encoding="utf-8", newline="\n")
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    ref = w / "explore"
    subprocess.run([sys.executable, "-m", "falls_ml", "meuhedet-explore", "--input", str(csv), "--out", str(ref), "--index-date", "2025-01-01",
                    "--index-day-records", "drop_rows", "--model-report", "off"], check=True, env=env, cwd=ROOT, capture_output=True)
    cfg = yaml.safe_load((ROOT / "configs/meuhedet/phase2.yaml").read_text(encoding="utf-8"))
    p = cfg["phase2"]
    p["cv"] = {"outer_folds": 3, "inner_folds_linear": 3, "inner_folds_xgb": 3}
    p["xgb"]["stage1"].update({"n_trials": 3, "n_startup_trials": 2})
    p["xgb"]["stage2"].update({"n_trials": 2, "condition_min_ap_gain_vs_best_linear": -1.0})
    p["xgb"].update({"n_estimators_max": 150, "early_stopping_rounds": 20})
    p["lasso"]["n_lambda"] = 25
    p["enet"].update({"n_lambda": 20, "l1_ratios": [0.5, 0.9]})
    p["stability"]["n_replicates"] = 2
    p["ablation"]["individual_top_n"] = 2
    p["metrics"]["bootstrap_n"] = 40
    p["explain"].update({"shap_sample_rows": 800, "permutation_repeats": 1})
    p["eligibility"]["min_observed_train_rows"] = 20
    p["features"] = str(ROOT / "configs/meuhedet/phase2_features.yaml")
    p["d00_config"] = str(ROOT / "configs/meuhedet/d00_sensitivity.yaml")
    cfg_path = w / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8", newline="\n")
    return {"w": w, "csv": csv, "ref": ref, "cfg": cfg_path, "env": env}


def _cli(world: dict[str, Any], out: Path, *extra: str, crash: str | None = None, cmd: str = "meuhedet-phase2") -> subprocess.CompletedProcess[str]:
    env = dict(world["env"])
    if crash:
        env["FALLS_ML_PHASE2_CRASH_AT"] = crash
    return subprocess.run([sys.executable, "-m", "falls_ml", cmd, "--input", str(world["csv"]), "--reference", str(world["ref"]),
                           "--out", str(out), "--config", str(world["cfg"]), "--allow-unfrozen-config", *extra], env=env, cwd=ROOT,
                          capture_output=True, text=True, encoding="utf-8", errors="replace")


def _complete(world: dict[str, Any], out: Path, *first: str, crash: str | None = None) -> subprocess.CompletedProcess[str]:
    r = _cli(world, out, *first, crash=crash)
    while r.returncode == 2 and "INVESTIGATION STOP" in r.stderr:   # synthetic planted signals can trip plausibility gates: accept, recorded
        gate = r.stderr.split("[", 1)[1].split("]", 1)[0]
        r = _cli(world, out, "--resume", "--accept-gate", gate, "--reason", "synthetic test data (planted signal)", crash=crash)
    return r


def committed(out: Path) -> dict[str, str]:
    res = {}
    for rec_path in sorted((out / "stages").glob("*/items/*.COMPLETE.json")):
        rec = json.loads(rec_path.read_text(encoding="utf-8"))
        d = rec_path.parent / rec_path.name[: -len(".COMPLETE.json")]
        res[str(rec_path.relative_to(out))] = _sha(rec_path)
        for rel in rec["files"]:
            res[str((d / rel).relative_to(out))] = _sha(d / rel)
    return res


@pytest.fixture(scope="module")
def run_a(world: dict[str, Any]) -> Path:
    out = world["w"] / "A"
    r = _complete(world, out)
    assert r.returncode == 0, r.stderr[-3000:]
    return out


def test_full_run_outputs_and_share(run_a: Path) -> None:
    share = run_a / "share"
    for f in ("README.md", "RUN_MANIFEST.json", "RUN_STATE_FINAL.json", "FINAL_EXPERIMENT_CONFIG.json", "FINAL_EXPERIMENT_CONFIG.sha256",
              "SELECTION_FROZEN.json", "VALIDATION_OPENED.json", "VALIDATION_CONFIRMATION.json", "COLUMN_FUNNEL.csv", "FEATURE_REGISTRY.csv",
              "FEATURE_QUALITY.csv", "FEATURE_SHORTLIST.csv", "FEATURE_CONSENSUS.csv", "EXPLORATORY_UNRESOLVED_FEATURES.csv", "MODEL_COMPARISON.csv",
              "DOMAIN_INCREMENTAL_GAIN.csv", "ABLATION_RESULTS.csv", "OPERATIONAL_CAPACITY.csv", "THRESHOLD_RESULTS.csv", "CALIBRATION_SUMMARY.csv",
              "SUBGROUP_SUMMARY.csv", "MANAGEMENT_SUMMARY_HE.md", "MANAGEMENT_HEADLINE.json", "SCIENTIFIC_SUMMARY.md", "RESUME_AUDIT.json",
              "RUN_TIMINGS.csv", "PRIVACY_SCAN.json", "reviews/REVIEW_DECISIONS.md"):
        assert (share / f).is_file(), f
    assert json.loads((share / "PRIVACY_SCAN.json").read_text(encoding="utf-8"))["passed"]
    figs = {p.stem for p in (share / "figures").glob("*.png")}
    assert {"01_feature_funnel", "02_domain_waterfall", "03_auroc_comparison", "04_ap_comparison", "05_calibration", "07_top_risk_capture",
            "08_false_alerts_matched_capacity", "09_model_families", "11_additional_falls_same_capacity"} <= figs
    manifest = json.loads((share / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["split"]["test_outcomes_read"] is False and manifest["synthetic"] is True
    assert manifest["frozen_production_config"] is False       # the reduced test configuration is flagged, never passed off as the frozen one
    assert (share / "FINAL_EXPERIMENT_CONFIG.sha256").read_text(encoding="utf-8").split()[0] == _sha(share / "FINAL_EXPERIMENT_CONFIG.json")
    for p in share.rglob("*.csv"):   # no row-level file in share
        assert "research_id" not in pd.read_csv(p, nrows=0).columns, p
    dom = pd.read_csv(share / "DOMAIN_INCREMENTAL_GAIN.csv")
    wf = dom[dom["analysis"] == "management_waterfall"]
    assert list(wf["model"][:2]) == ["LASSO:BASELINE_15", "LASSO:SAFE_BASE"] and "XGB_TUNED:ALL_REVIEWED_SAFE" in set(wf["model"])
    for c in ("val_ap", "val_auroc", "val_brier", "val_cal_slope", "val_capture_top3", "val_capture_top5", "val_capture_top10", "val_capture_top20",
              "val_ppv_top10", "val_false_alerts_top10", "val_additional_falls_vs_benchmark_top10"):
        assert c in wf.columns, c
    assert wf["val_n"].nunique() == 1          # every step on the same VALIDATION rows


def test_category_invariants_safe_sets_hold_safe_features_only(run_a: Path) -> None:
    er = pd.read_csv(run_a / "artifacts" / "02_ENGINEERED_FEATURE_REGISTRY.csv").set_index("feature")
    assert er.at["form_home_safety_assessed", "eligibility"] == "INELIGIBLE_UNSAFE"      # planted index-day records -> UNSAFE, never fitted
    fs = json.loads((run_a / "artifacts" / "FEATURE_SETS.json").read_text(encoding="utf-8"))
    bstat = fs["design"]["baseline_provenance"]
    unsafe = set(er.index[er["eligibility"] == "INELIGIBLE_UNSAFE"])
    unresolved = set(er.index[er["eligibility"] == "ELIGIBLE_EXPLORATORY"])
    assert unresolved, "the synthetic extract must contain UNRESOLVED features for this test to mean anything"
    seen_unresolved_in_exploratory = False
    for name, s in fs["sets"].items():
        assert not unsafe & set(s["features"]), name
        if s["category"] == "SAFE_DISCOVERY":
            assert all(er.at[f, "eligibility"] == "ELIGIBLE" for f in s["features"]), name
            assert all(bstat[f] == "SAFE" for f in s["baseline"]), name
        if s["category"] == "EXPLORATORY_UNRESOLVED_SENSITIVITY":
            seen_unresolved_in_exploratory |= bool(unresolved & set(s["features"]))
    assert seen_unresolved_in_exploratory
    assert fs["sets"]["BASELINE_15"]["category"] == "HISTORICAL_BASELINE_15" and fs["sets"]["ALL_REVIEWED_SAFE"]["category"] == "SAFE_DISCOVERY"
    cons = pd.read_csv(run_a / "share" / "FEATURE_CONSENSUS.csv")
    assert set(cons["provenance_status"]) == {"SAFE"}
    xf = pd.read_csv(run_a / "share" / "EXPLORATORY_UNRESOLVED_FEATURES.csv")
    assert set(xf["provenance_status"]) == {"UNRESOLVED"}
    oof = pd.read_csv(run_a / "artifacts" / "OOF_MODEL_COMPARISON.csv")
    assert not oof.loc[oof["qualifies_vs_safe_base"].astype(bool), "category"].ne("SAFE_DISCOVERY").any()
    # XGBoost never ingests identifiers / metadata / labels: every tree column is a catalogue feature or a BASELINE_15 predictor of its set
    allowed = set(er.index) | set(fs["sets"]["BASELINE_15"]["baseline"]) | {"sex_female"}
    for rec in (run_a / "stages" / "S08_xgb" / "items").glob("*.COMPLETE.json"):
        cols = json.loads(rec.read_text(encoding="utf-8"))["result"].get("columns")
        if cols:
            assert set(cols) <= allowed, (rec.name, sorted(set(cols) - allowed))
            safe_cols = [c for c in cols if c in er.index]
            if "ALL_REVIEWED_SAFE" in rec.name or rec.name.startswith("XGB1_"):
                assert all(er.at[c, "eligibility"] == "ELIGIBLE" for c in safe_cols), rec.name


def test_selection_frozen_before_validation_opened_once(run_a: Path) -> None:
    sel = json.loads((run_a / "SELECTION_FROZEN.json").read_text(encoding="utf-8"))
    opened = json.loads((run_a / "VALIDATION_OPENED.json").read_text(encoding="utf-8"))
    assert sel["benchmark"] == "LASSO:BASELINE_15" and sel["discovery_reference"] == "LASSO:SAFE_BASE"
    assert opened["frozen_selection_sha256"] == _sha(run_a / "SELECTION_FROZEN.json")
    assert opened["final_experiment_config_sha256"] == _sha(run_a / "FINAL_EXPERIMENT_CONFIG.json")
    assert sorted(opened["descriptive_configs"]) == sorted(sel["descriptive_configs"])
    assert {f["config"] for f in opened["confirmatory_finalists"]} == set(sel["shortlist"])
    assert sel["frozen_at"] <= opened["opened_at"]
    reg = [json.loads(line) for line in (run_a / "validation_evaluation_registry.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len({r["selection_sha256"] for r in reg}) == 1
    ev = [json.loads(line) for line in (run_a / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    t_sel = next(e["ts"] for e in ev if e["stage"] == "S09_select" and e["action"] == "stage")
    t_val = next(e["ts"] for e in ev if e["stage"] == "S13_validation" and e["action"] == "item")
    assert t_sel <= t_val
    facts = json.loads((run_a / "artifacts" / "COHORT_FACTS.json").read_text(encoding="utf-8"))
    assert facts["test_outcomes_read"] is False and facts["test_rows_dropped"] > 0
    work = pd.read_parquet(next((run_a / "stages" / "S01_cohort" / "items").glob("cohort/trainval.parquet")))
    assert set(work["partition"]) == {"train", "validation"}      # TEST rows never reach any stage


def test_status_is_read_only(world: dict[str, Any], run_a: Path) -> None:
    before = {str(p.relative_to(run_a)): (_sha(p), p.stat().st_mtime_ns) for p in run_a.rglob("*") if p.is_file()}
    r = subprocess.run([sys.executable, "-m", "falls_ml", "meuhedet-phase2-status", "--out", str(run_a)], env=world["env"], cwd=ROOT,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stderr
    assert "completed stages  : 17 of 17" in r.stdout and "share             : READY" in r.stdout
    after = {str(p.relative_to(run_a)): (_sha(p), p.stat().st_mtime_ns) for p in run_a.rglob("*") if p.is_file()}
    assert before == after


def test_preflight_fits_nothing_and_writes_nothing(world: dict[str, Any]) -> None:
    out = world["w"] / "PREFLIGHT_OUT"
    r = _cli(world, out, "--preflight")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
    assert r.stdout.rstrip().splitlines()[-1] == "SAFE TO START FULL RUN"
    for needle in ("SAFE_BASE predictors", "SAFE new features entering the discovery", "XGBoost stage 1", "xgb_optuna.sqlite", "RUN_STATE.json",
                   "category invariant", "plan sha256 the run will freeze"):
        assert needle in r.stdout, needle
    assert not out.exists()
    bad = _cli(world, out, "--preflight", "--resume")
    assert bad.returncode == 2


def test_kills_inside_trial_validation_and_share_then_resume_are_identical(world: dict[str, Any], run_a: Path) -> None:
    out = world["w"] / "B"
    snapshots: list[dict[str, str]] = []
    first = True
    for crash in ("S08_xgb|XGB1_outer1__t01|before_record", "S13_validation|validation|before_record", "S16_share|share|before_rename"):
        r = _complete(world, out, *([] if first else ["--resume"]), crash=crash)
        first = False
        assert r.returncode == 137, (crash, r.returncode, r.stderr[-2000:])
        snapshots.append(committed(out))
        fc = _sha(out / "FINAL_EXPERIMENT_CONFIG.json")
    r = _complete(world, out, "--resume")
    assert r.returncode == 0, r.stderr[-3000:]
    now = committed(out)
    for snap in snapshots:   # nothing committed before a kill ever changed afterwards
        assert all(now.get(k) == v for k, v in snap.items())
    assert _sha(out / "FINAL_EXPERIMENT_CONFIG.json") == fc
    # the Optuna study resumed: every (study, trial) is committed once and its trial item was never recomputed
    ledger = [json.loads(line) for line in (out / "checkpoints" / "xgb_trials.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    keys = [(r["study"], r["trial_index"]) for r in ledger]
    assert len(keys) == len(set(keys))
    t0 = "stages/S08_xgb/items/XGB1_outer1__t00.COMPLETE.json".replace("/", os.sep)
    assert t0 in snapshots[0] and now[t0] == snapshots[0][t0]      # committed before the kill, never recomputed
    assert list((out / "stages" / "S08_xgb" / "_incomplete").rglob("XGB1_outer1__t01"))
    # the validation was re-read only for the identical frozen selection
    reg = [json.loads(line) for line in (out / "validation_evaluation_registry.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len({x["selection_sha256"] for x in reg}) == 1 and len(reg) >= 2
    assert json.loads((out / "share" / "PRIVACY_SCAN.json").read_text(encoding="utf-8"))["passed"]
    for f in ("MODEL_COMPARISON.csv", "FEATURE_CONSENSUS.csv", "DOMAIN_INCREMENTAL_GAIN.csv", "OPERATIONAL_CAPACITY.csv", "SCIENTIFIC_SUMMARY.md",
              "MANAGEMENT_SUMMARY_HE.md", "FEATURE_SHORTLIST.csv", "MANAGEMENT_HEADLINE.json"):
        assert (out / "share" / f).read_bytes() == (run_a / "share" / f).read_bytes(), f
    audit = json.loads((out / "RESUME_AUDIT.json").read_text(encoding="utf-8"))
    assert [a["status"] for a in audit["attempts"]][-1] == "COMPLETE" and len(audit["attempts"]) >= 4


def test_resume_refuses_a_changed_plan(world: dict[str, Any], run_a: Path, tmp_path: Path) -> None:
    cfg = yaml.safe_load(Path(world["cfg"]).read_text(encoding="utf-8"))
    cfg["phase2"]["stability"]["n_replicates"] = 5
    p = tmp_path / "changed.yaml"
    p.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8", newline="\n")
    r = subprocess.run([sys.executable, "-m", "falls_ml", "meuhedet-phase2", "--input", str(world["csv"]), "--reference", str(world["ref"]),
                        "--out", str(run_a), "--config", str(p), "--allow-unfrozen-config", "--resume"], env=world["env"], cwd=ROOT,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert r.returncode == 2 and "PLAN_CHANGED" in r.stderr


def test_code_change_needs_explicit_acceptance_and_is_logged(world: dict[str, Any]) -> None:
    out = world["w"] / "C"
    r = _complete(world, out, crash="S06_lasso|LASSO__BASELINE_15__outer0|item_start")
    assert r.returncode == 137
    # simulate a run started with other code (the stored code hash differs from the current one)
    plan = json.loads((out / "PHASE2_PLAN.json").read_text(encoding="utf-8"))
    plan["code_sha256_at_start"] = "0" * 64
    (out / "PHASE2_PLAN.json").write_text(json.dumps(plan), encoding="utf-8")
    audit = json.loads((out / "RESUME_AUDIT.json").read_text(encoding="utf-8"))
    for a in audit["attempts"]:
        a["code_sha256"] = "0" * 64
    (out / "RESUME_AUDIT.json").write_text(json.dumps(audit), encoding="utf-8")
    r = _cli(world, out, "--resume", crash="S06_lasso|LASSO__BASELINE_15__outer0|item_start")
    assert r.returncode == 2 and "CODE_CHANGED" in r.stderr
    bad = _cli(world, out, "--accept-code-change", "no resume flag")
    assert bad.returncode == 2
    r = _cli(world, out, "--resume", "--accept-code-change", "test: code patched between attempts", crash="S06_lasso|LASSO__BASELINE_15__outer1|item_start")
    assert r.returncode == 137
    gates = [json.loads(line) for line in (out / "logs" / "gates.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert any(g["gate"] == "CODE_CHANGED" and g["decision"] == "ACCEPTED" and g["reason"].startswith("test:") for g in gates)
    audit = json.loads((out / "RESUME_AUDIT.json").read_text(encoding="utf-8"))
    assert audit["attempts"][-1]["accept_code_change"] == "test: code patched between attempts"


def test_accept_gate_requires_resume_and_reason(world: dict[str, Any]) -> None:
    out = world["w"] / "D"
    assert _cli(world, out, "--accept-gate", "IMPLAUSIBLE_GAIN_OOF", "--reason", "x").returncode == 2
    assert _cli(world, out, "--resume", "--accept-gate", "IMPLAUSIBLE_GAIN_OOF").returncode == 2
    assert not out.exists()
