"""Phase 5 end-to-end on SYNTHETIC V21 extracts (slow): the whole overnight command in a small configuration (3 x 3 folds, tiny budgets).

Brief AB proofs covered here:
 4  the same outer folds are reused for OLD and NEW (and every other set)     test_same_outer_folds_for_every_family_and_set
 9  no true extra signal -> no guaranteed improvement                        test_null_new_feature_does_not_show_a_robust_gain
 10 a planted useful new feature improves the operational metric             test_planted_new_feature_lowers_the_false_alert_burden
 12 an interrupted run resumes without repeating finished work              test_interrupted_run_resumes_without_recomputing
 14 the share folder holds no patient-level data                             test_share_is_aggregate_only
 15 Windows-safe paths (a folder name with spaces; no absolute path shared)  (every test: the work folder has spaces; share path scan)
 17 a deterministic rerun gives identical folds and model-selection decisions test_deterministic_rerun
 2  a tampered plan with an ineligible / post-index feature hard-stops         test_tampered_feature_set_hard_stops_on_resume
 E  an outcome-contract violation stops before any model is fitted            test_outcome_contract_violation_stops_before_training
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.slow
ROOT = Path(__file__).resolve().parents[2]
OV = {"cv": {"outer_folds": 3, "inner_folds": 3}, "eligibility": {"min_known_observed_rows": 20},
      "modes": {"quick": {"lasso": {"n_lambda": 8}, "enet": {"n_lambda": 6, "l1_ratios": [0.2, 0.8]}, "xgb": {"n_trials": 3, "n_startup_trials": 2},
                          "bootstrap_n": 300, "stability_linear": 3, "stability_xgb": 2, "permutation_repeats": 1, "shap_rows": 500}}}
REQUIRED_SHARE = ["MANAGEMENT_SUMMARY_HE.md", "SCIENTIFIC_SUMMARY_HE.md", "SCIENTIFIC_SUMMARY.md", "PRIMARY_70_SENSITIVITY_COMPARISON.csv", "THRESHOLD_TRADEOFF.csv",
                  "MODEL_COMPARISON.csv", "DOMAIN_INCREMENTAL_VALUE.csv", "ABLATION_RESULTS.csv", "OOF_MODEL_COMPARISON.csv", "OOF_PREDICTION_SUMMARY.csv",
                  "CALIBRATION.csv", "SUBGROUP_SUMMARY.csv", "FEATURE_ELIGIBILITY.csv", "NEW_FEATURE_CATALOGUE.csv", "FEATURE_STABILITY.csv",
                  "PERMUTATION_IMPORTANCE.csv", "SHAP_SUMMARY.csv", "COHORT_FACTS_2026.json", "OUTCOME_CONTRACT_2026.json", "RUN_MANIFEST.json",
                  "RUN_TIMINGS.csv", "ENVIRONMENT.json", "PRIVACY_SCAN.json"]


def _run(src: Path, out: Path, **kw: Any) -> dict[str, Any]:
    from falls_ml.phase5.runner import run_phase5

    return run_phase5(src, out, mode="quick", device="auto", jobs=2, resume=True, overrides=OV, synthetic=True, **kw)


def _units(out: Path) -> dict[str, dict[str, Any]]:
    res = {}
    for d in sorted((out / "work" / "units").iterdir()):
        if (d / "COMPLETE.json").is_file():
            r = json.loads((d / "result.json").read_text(encoding="utf-8"))
            a = np.load(d / "arrays.npz")
            res[d.name] = {"result": r, "test_idx": a["test_idx"], "p_test": a["p_test"]}
    return res


def _digest(d: Path) -> dict[str, tuple[int, str]]:
    return {p.relative_to(d).as_posix(): (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(d.rglob("*")) if p.is_file()}


@pytest.fixture(scope="module")
def worlds(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    from falls_ml.phase5.synthetic import make_v21, write_v21_csv

    base = tmp_path_factory.mktemp("p5 e2e") / "work folder with spaces"
    base.mkdir()
    out = {"base": base}
    for sc in ("planted", "null"):
        d = base / f"in {sc}"
        d.mkdir()
        df, _ = make_v21(3000, scenario=sc, seed=41, signal=0.45)      # planted: P(new=1) 51% for fallers vs 6% (univariate AUROC ~0.72 < 0.80)
        out[sc] = write_v21_csv(df, d / f"v21 {sc}.csv")
        out[f"df_{sc}"] = df
    return out


@pytest.fixture(scope="module")
def planted(worlds: dict[str, Any]) -> dict[str, Any]:
    out = worlds["base"] / "out planted"
    r = _run(worlds["planted"], out)
    return {"out": out, "res": r}


# ============================================================================ the planted run
def test_planted_run_completes_with_every_output(planted: dict[str, Any]) -> None:
    out = planted["out"]
    assert planted["res"]["status"] == "COMPLETE" and planted["res"]["exit_code"] == 0
    share = out / "share"
    for n in REQUIRED_SHARE:
        assert (share / n).is_file(), n
    assert len(list((share / "figures").glob("*.png"))) >= 12
    for n in ("RUN_STATUS.json", "RUN_TIMINGS.csv", "OVERNIGHT_PROGRESS.log"):
        assert (out / n).is_file()
    st = json.loads((out / "RUN_STATUS.json").read_text(encoding="utf-8"))
    assert st["status"] == "COMPLETE" and st["heartbeat"] and st["progress"]["PRIMARY"]["done"] == st["progress"]["PRIMARY"]["total"]
    assert "SYNTHETIC" in (share / "MANAGEMENT_SUMMARY_HE.md").read_text(encoding="utf-8")
    man = json.loads((share / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    assert man["design"] == "INTERNAL NESTED CROSS-VALIDATION ON 2026 SNAPSHOT" and man["report_kind"] == "FINAL"
    assert (out / "preflight" / "PHASE5_PREFLIGHT.md").read_text(encoding="utf-8").rstrip().endswith("SAFE TO MODEL")


def test_planted_new_feature_lowers_the_false_alert_burden(planted: dict[str, Any]) -> None:
    t = pd.read_csv(planted["out"] / "share" / "PRIMARY_70_SENSITIVITY_COMPARISON.csv")
    d = t[t["row"] == "DELTA_NEW_MINUS_OLD"].set_index("family")
    for fam in ("LASSO", "ENET", "XGB"):
        assert float(d.loc[fam, "false_alert_share"]) < 0, fam
        assert float(d.loc[fam, "delta_false_alert_share_ci_high"]) < 0, fam          # paired interval below 0
        assert float(d.loc[fam, "delta_ap"]) > 0
    v = t[t["row"] == "VERDICT"].set_index("family")["verdict"]
    assert set(v) <= {"NEW_FEATURES_OPERATIONALLY_USEFUL", "PROMISING_BUT_NOT_ROBUST"} and "NEW_FEATURES_OPERATIONALLY_USEFUL" in set(v)
    dom = pd.read_csv(planted["out"] / "share" / "DOMAIN_INCREMENTAL_VALUE.csv")
    dd = dom[(dom["domain"] == "NEW_DIAGNOSIS") & (dom["status"] == "COMPLETE")]
    assert len(dd) == 3 and (dd["delta_false_alert_share"] < 0).all()


def test_same_outer_folds_for_every_family_and_set(planted: dict[str, Any]) -> None:
    u = _units(planted["out"])
    by_fold: dict[str, list[np.ndarray]] = {}
    for name, r in u.items():
        if r["result"]["outer"] >= 0:
            by_fold.setdefault(str(r["result"]["outer"]), []).append(r["test_idx"])
    assert set(by_fold) == {"0", "1", "2"}
    for k, arrs in by_fold.items():
        assert len(arrs) > 20
        for a in arrs[1:]:
            assert np.array_equal(a, arrs[0]), f"fold {k} differs between units"
    allt = np.concatenate([by_fold[k][0] for k in sorted(by_fold)])
    assert len(allt) == len(np.unique(allt))
    plan = json.loads((planted["out"] / "work" / "PLAN.json").read_text(encoding="utf-8"))
    assert len(allt) == plan["n"]


def test_share_is_aggregate_only(planted: dict[str, Any], worlds: dict[str, Any]) -> None:
    share = planted["out"] / "share"
    scan = json.loads((share / "PRIVACY_SCAN.json").read_text(encoding="utf-8"))
    assert scan["passed"] and scan["path_scan"] == "passed" and scan["row_level_scan"] == "passed" and scan["file_type_scan"] == "passed"
    ids = set(worlds["df_planted"]["Customer_Full_ID"].astype(str))
    keys = set(pd.read_parquet(planted["out"] / "work" / "ANALYSIS_FRAME.parquet", columns=["row_key"])["row_key"])
    for p in share.rglob("*"):
        if p.is_dir():
            continue
        assert p.suffix.lower() in (".csv", ".md", ".json", ".png", ".txt"), p.name
        if p.suffix.lower() in (".csv", ".md", ".json", ".txt"):
            text = p.read_text(encoding="utf-8")
            assert not any(i in text for i in list(ids)[:400]), p.name
            assert not any(k in text for k in list(keys)[:400]), p.name
            assert "work folder with spaces" not in text and str(worlds["base"]) not in text
        if p.suffix.lower() == ".csv" and p.stat().st_size > 2:
            t = pd.read_csv(p)
            assert len(t) <= 2000 and not {"row_key", "Customer_Full_ID"} & set(t.columns), p.name
    assert (planted["out"] / "work" / "analysis" / "OOF_PREDICTIONS_LOCAL.parquet").is_file()     # row-level kept locally only


def test_report_only_rebuilds_share_without_fitting(planted: dict[str, Any], worlds: dict[str, Any]) -> None:
    from falls_ml.phase5.runner import run_phase5

    units_before = _digest(planted["out"] / "work" / "units")
    r = run_phase5(worlds["planted"], planted["out"], mode="quick", report_only=True, overrides=OV, synthetic=True)
    assert r["status"] == "REPORT_COMPLETE" and r["report"]["privacy_passed"]
    assert _digest(planted["out"] / "work" / "units") == units_before


def test_status_and_estimate(planted: dict[str, Any], worlds: dict[str, Any], tmp_path: Path) -> None:
    from falls_ml.phase5.config import load_phase5_config
    from falls_ml.phase5.estimate import estimate, estimate_text
    from falls_ml.phase5.status import phase5_status

    s = phase5_status(planted["out"])
    assert s["status"] in ("COMPLETE", "REPORT_COMPLETE") and "PRIMARY" in s["text"] and "failures: 0" in s["text"]
    cfg = load_phase5_config(mode="overnight")
    e = estimate(worlds["planted"], tmp_path / "est", cfg, 2)
    assert e["total_hours"] > 0 and e["shape"]["n"] > 1000 and "no model was fitted" in estimate_text(e)
    assert (tmp_path / "est" / "ESTIMATE.json").is_file()


def test_tampered_feature_set_hard_stops_on_resume(planted: dict[str, Any], worlds: dict[str, Any]) -> None:
    from falls_ml.phase2.state import Phase2Stop

    copy = worlds["base"] / "out tampered"
    shutil.copytree(planted["out"], copy)
    pp = copy / "work" / "PLAN.json"
    plan = json.loads(pp.read_text(encoding="utf-8"))
    plan["sets"]["OLD_PLUS_NEW_SAFE"] = [*plan["sets"]["OLD_PLUS_NEW_SAFE"], "new_syncope_ind"]
    pp.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(Phase2Stop) as e:
        _run(worlds["planted"], copy)
    assert e.value.gate == "X_LEAKAGE"


# ============================================================================ the null run, interrupted and resumed
def test_interrupted_run_resumes_without_recomputing(worlds: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    import falls_ml.phase5.runner as R

    out = worlds["base"] / "out null"
    real = R.run_unit
    calls = {"n": 0}

    def interrupting(ctx: Any, spec: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 7:
            raise KeyboardInterrupt                      # Ctrl+C in the middle of the night
        return real(ctx, spec)

    monkeypatch.setattr(R, "run_unit", interrupting)
    r1 = _run(worlds["null"], out)
    assert r1["status"] == "INTERRUPTED" and r1["exit_code"] == 130
    assert json.loads((out / "RUN_STATUS.json").read_text(encoding="utf-8"))["status"] == "INTERRUPTED"
    done_before = {k: v for k, v in _digest(out / "work" / "units").items()}
    n_done = sum(1 for k in done_before if k.endswith("COMPLETE.json"))
    assert n_done == 6
    monkeypatch.setattr(R, "run_unit", real)
    r2 = _run(worlds["null"], out, max_items=4)
    assert r2["status"] == "PAUSED_TEST_LIMIT"
    r3 = _run(worlds["null"], out)
    assert r3["status"] == "COMPLETE"
    after = _digest(out / "work" / "units")
    for k, v in done_before.items():
        if k.split("/")[0] in {x.split("/")[0] for x in done_before if x.endswith("COMPLETE.json")}:
            assert after[k] == v, f"a finished unit was recomputed: {k}"
    t = pd.read_csv(out / "RUN_TIMINGS.csv")
    done = t[t["status"] == "done"]
    assert done["item"].is_unique, "an item was computed twice"
    st = json.loads((out / "RUN_STATUS.json").read_text(encoding="utf-8"))
    assert st["sessions"] == 3


def test_null_new_feature_does_not_show_a_robust_gain(worlds: dict[str, Any]) -> None:
    out = worlds["base"] / "out null"
    if not (out / "share" / "PRIMARY_70_SENSITIVITY_COMPARISON.csv").is_file():
        pytest.skip("depends on test_interrupted_run_resumes_without_recomputing")
    t = pd.read_csv(out / "share" / "PRIMARY_70_SENSITIVITY_COMPARISON.csv")
    v = t[t["row"] == "VERDICT"].set_index("family")["verdict"]
    assert "NEW_FEATURES_OPERATIONALLY_USEFUL" not in set(v), v.to_dict()
    man = json.loads((out / "share" / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    assert man["overall_answer"] in ("NO", "UNCERTAIN")
    # the patient bootstrap holds the fitted models fixed: under the null a noise feature can still give a narrow interval below 0 (model noise);
    # the pre-declared rule therefore also requires the improvement in EVERY outer fold - no family meets both here
    vr = t[t["row"] == "VERDICT"].set_index("family")
    assert not vr["criterion_2_paired_ci_below_zero_and_every_outer_fold_improves"].astype(str).eq("True").any()


# ============================================================================ determinism, contract stop
def test_deterministic_rerun(planted: dict[str, Any], worlds: dict[str, Any]) -> None:
    out2 = worlds["base"] / "out planted rerun"
    plan1 = json.loads((planted["out"] / "work" / "PLAN.json").read_text(encoding="utf-8"))
    n_primary = sum(1 for k in _units(planted["out"]) if k.startswith("PRIMARY__"))
    r = _run(worlds["planted"], out2, max_items=n_primary)
    assert r["status"] == "PAUSED_TEST_LIMIT"
    plan2 = json.loads((out2 / "work" / "PLAN.json").read_text(encoding="utf-8"))
    assert plan1["folds_sha256"] == plan2["folds_sha256"] and plan1["frame_sha256"] == plan2["frame_sha256"] and plan1["sets"] == plan2["sets"]
    u1, u2 = _units(planted["out"]), _units(out2)
    prim = [k for k in u2 if k.startswith("PRIMARY__")]
    assert len(prim) == n_primary
    for k in prim:
        a, b = u1[k]["result"], u2[k]["result"]
        assert a["config"] == b["config"] and a["thresholds"] == b["thresholds"], k
        assert np.array_equal(u1[k]["p_test"], u2[k]["p_test"]), k


def test_outcome_contract_violation_stops_before_training(worlds: dict[str, Any]) -> None:
    from falls_ml.phase5.synthetic import make_v21, write_v21_csv

    d = worlds["base"] / "in contract"
    d.mkdir()
    df, _ = make_v21(3000, scenario="planted", seed=41)
    y = df["Fall_Next_180D_Ind"]
    pos = df.index[(y == 1).fillna(False)][:40]
    idx = pd.Timestamp("2026-01-01")
    df.loc[pos, "Next_Fall_Date_180D"] = np.datetime64(idx, "us")          # positive labels for falls ON the index day (contract violation)
    df.loc[pos, "Days_To_Next_Fall_180D"] = 0
    src = write_v21_csv(df, d / "v21 contract.csv")
    out = worlds["base"] / "out contract"
    r = _run(src, out)
    assert r["status"] == "STOPPED_PREFLIGHT" and r["exit_code"] == 2
    assert not (out / "work" / "units").exists() and not (out / "work" / "PLAN.json").exists()
    md = (out / "preflight" / "PHASE5_PREFLIGHT.md").read_text(encoding="utf-8")
    assert md.rstrip().endswith("STOP - 2026 REDEVELOPMENT NOT DEFENSIBLE") and "O1" in md
    oc = json.loads((out / "preflight" / "OUTCOME_CONTRACT_2026.json").read_text(encoding="utf-8"))
    assert oc["passed"] is False


def test_duplicate_patient_rows_stop(worlds: dict[str, Any]) -> None:
    d = worlds["base"] / "in dup"
    d.mkdir()
    df = worlds["df_planted"].copy()
    el = df.index[(df["Is_Eligible_Cohort"] == 1) & (df["Index_Date"] == pd.Timestamp("2026-01-01"))]
    df.loc[el[1], "Customer_Full_ID"] = df.loc[el[0], "Customer_Full_ID"]
    from falls_ml.phase5.synthetic import write_v21_csv

    src = write_v21_csv(df, d / "v21 dup.csv")
    r = _run(src, worlds["base"] / "out dup")
    assert r["status"] == "STOPPED_PREFLIGHT"
