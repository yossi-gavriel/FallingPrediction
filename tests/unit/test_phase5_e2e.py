"""Phase 5 end-to-end on SYNTHETIC extracts with the EXACT authoritative V21 header (slow): the whole overnight command in a small configuration
(3 x 3 folds, tiny budgets).

Brief section 22 acceptance proofs covered here:
 7  a tampered plan with an ineligible / removed feature hard-stops           test_tampered_feature_set_hard_stops_on_resume
 8  the same folds are used for OLD and NEW (every family and set)            test_same_outer_folds_for_every_family_and_set
 11 null new features do not produce a USEFUL verdict                        test_null_new_features_do_not_give_a_useful_verdict
 12 a planted useful new feature improves the operational endpoint           test_planted_new_feature_lowers_the_false_alert_burden
 14 an interrupted run resumes without repeating finished work              test_interrupted_run_resumes_without_recomputing
 15 the share folder holds no patient-level data                             test_share_is_aggregate_only
 +  Windows-safe paths (folder names with spaces; no absolute path shared), a deterministic rerun, the outcome contract and duplicate-ID stops
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
OV = {"cv": {"outer_folds": 3, "inner_folds": 3}, "eligibility": {"min_known_observed_rows": 20}, "negative_controls": {"seeds": 2},
      "modes": {"quick": {"lasso": {"n_lambda": 8}, "enet": {"n_lambda": 6, "l1_ratios": [0.2, 0.8]}, "xgb": {"n_trials": 3, "n_startup_trials": 2},
                          "bootstrap_n": 300, "stability_linear": 3, "stability_xgb": 2, "permutation_repeats": 1, "shap_rows": 500}}}
REQUIRED_SHARE = ["MANAGEMENT_SUMMARY_HE.md", "SCIENTIFIC_SUMMARY_HE.md", "SCIENTIFIC_SUMMARY.md", "PRIMARY_70_SENSITIVITY_COMPARISON.csv", "THRESHOLD_TRADEOFF.csv",
                  "MODEL_COMPARISON.csv", "DOMAIN_INCREMENTAL_VALUE.csv", "ABLATION_RESULTS.csv", "OOF_MODEL_COMPARISON.csv", "CALIBRATION.csv",
                  "SUBGROUP_SUMMARY.csv", "FEATURE_ELIGIBILITY.csv", "NEW_FEATURE_CATALOGUE.csv", "ALL_V21_COLUMN_CLASSIFICATION.csv", "SCHEMA_DIFF_V1_V21.csv",
                  "FEATURE_STABILITY.csv", "PERMUTATION_IMPORTANCE.csv", "COHORT_FACTS_2026.json", "OUTCOME_CONTRACT_2026.json",
                  # Phase 5.1 (ENET only, no SHAP): the negative controls and the historical-audit verdict
                  "NEGATIVE_CONTROLS.csv", "NEGATIVE_CONTROLS_RESULT.json", "HISTORICAL_VERDICT_2_2_0.json",
                  "RUN_MANIFEST.json", "RUN_TIMINGS.csv", "ENVIRONMENT.json", "PRIVACY_SCAN.json",     # brief section 24
                  "REMOVED_V1_COLUMNS.csv", "RENAMED_OR_CHANGED_COLUMNS.csv", "V21_UNDECLARED_COLUMNS.csv", "CAPACITY_CURVE.csv", "OOF_PREDICTION_SUMMARY.csv",
                  "OUTER_FOLD_RESULTS.csv",
                  # the operating-capacity dashboard and the exact 3% report
                  "PHASE5_OPERATING_DASHBOARD.html", "CAPACITY_CURVE_FINE.csv", "TOP3_CAPACITY_PRIMARY.csv", "TOP3_CAPACITY_BY_FOLD.csv",
                  "TOP3_CAPACITY_COMPARISON.csv", "TOP3_CAPACITY_BOOTSTRAP.csv", "TOP3_CAPACITY_SUMMARY_HE.md", "FEATURE_DRIVERS.csv"]
PRIMARY_CMP = "OLD vs OLD_PLUS_ALL_NEW_ELIGIBLE"


def _run(src: Path, out: Path, controls: bool = False, **kw: Any) -> dict[str, Any]:
    from falls_ml.phase5.runner import run_phase5

    return run_phase5(src, out, mode="quick", device="auto", jobs=2, resume=True, overrides=OV, synthetic=True, negative_controls=controls,
                      negative_control_seeds=2, **kw)


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
    r = _run(worlds["planted"], out, controls=True)
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
    assert man["primary_family"] == "ENET" and man["families"] == ["ENET"] and man["primary_prepost_arms"] == ["OLD", "OLD_PLUS_NEW_SAFE"]
    assert man["historical_primary_comparison_2_2_0"] == ["OLD", "OLD_PLUS_ALL_NEW_ELIGIBLE"] and man["negative_controls"]["passed"] is True
    assert man["experiment"].startswith("PHASE 5.1") and man["folds_source"].startswith("generated")
    cl = pd.read_csv(share / "ALL_V21_COLUMN_CLASSIFICATION.csv")
    assert len(cl) == 224 and cl["x_use"].astype(str).str.len().gt(3).all() and not (cl["class"] == "REQUIRES_SEMANTIC_REVIEW").any()
    sets = json.loads((share / "FEATURE_SETS.json").read_text(encoding="utf-8"))
    assert {"OLD", "OLD_PLUS_ALL_NEW_ELIGIBLE", "OLD_PLUS_NEW_SAFE"} <= set(sets["sets"])
    ab = pd.read_csv(share / "ABLATION_RESULTS.csv")
    assert set(ab["family"]) == {"ENET"} and set(ab["ablation"]) == {"NO_NEW_REGISTRY"} and set(ab["role"]) == {"SECONDARY_DIAGNOSTIC"}
    assert set(ab["base_set"]) == {"OLD_PLUS_NEW_SAFE"}
    u = _units(out)
    assert {r["result"]["family"] for r in u.values()} == {"ENET"}                                   # ENET only: no LASSO / XGB unit exists
    assert all(r["result"].get("features_effective") for r in u.values())
    nc = pd.read_csv(share / "NEGATIVE_CONTROLS.csv")
    assert len(nc) == 2 and (nc["auroc"] < 0.62).all() and (nc["recall_top3"] < 0.12).all()         # permuted labels: chance-level (small synthetic folds)
    assert (out / "work" / "FORENSIC_UNIVARIATE_AUROC_BY_FOLD.csv").is_file() and not (share / "FORENSIC_UNIVARIATE_AUROC_BY_FOLD.csv").exists()
    hv = json.loads((share / "HISTORICAL_VERDICT_2_2_0.json").read_text(encoding="utf-8"))
    assert "HISTORICAL AUDIT" in hv["note"] and "ENET" in hv["decisions"]
    mg = (share / "MANAGEMENT_SUMMARY_HE.md").read_text(encoding="utf-8")
    for needle in ("מטופלים שנותחו", "נפילות", "פיצ'רים חדשים אמיתיים", "ALL_NEW", "התראות שווא שנחסכו", "PPV", "האם הפיצ'רים הנוספים של V21"):
        assert needle in mg, needle


def test_planted_new_feature_lowers_the_false_alert_burden(planted: dict[str, Any]) -> None:
    t = pd.read_csv(planted["out"] / "share" / "PRIMARY_70_SENSITIVITY_COMPARISON.csv")
    t = t[t["comparison"] == PRIMARY_CMP]
    d = t[t["row"] == "DELTA_NEW_MINUS_OLD"].set_index("family")
    for fam in ("ENET",):
        assert float(d.loc[fam, "false_alert_share"]) < 0, fam
        assert float(d.loc[fam, "delta_false_alert_share_ci_high"]) < 0, fam          # paired interval below 0
        assert float(d.loc[fam, "false_alerts_avoided_per_10000"]) > 0, fam
        assert float(d.loc[fam, "delta_ap"]) > 0
    v = t[t["row"] == "VERDICT"].set_index("family")
    assert v.loc["ENET", "verdict"] == "NEW_FEATURES_OPERATIONALLY_USEFUL" and v.loc["ENET", "role"] == "PRIMARY"
    man = json.loads((planted["out"] / "share" / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    assert man["historical_overall_answer_2_2_0_rule"] == "YES" and "overall_answer" not in man       # historical audit only, no 5.1 verdict
    dom = pd.read_csv(planted["out"] / "share" / "DOMAIN_INCREMENTAL_VALUE.csv")
    assert len(dom) == 0                                                                               # Phase 5.1: no DOMAIN units
    p3 = pd.read_csv(planted["out"] / "share" / "TOP3_CAPACITY_PRIMARY.csv")
    assert set(p3["family"]) == {"ENET"} and set(p3["feature_set"]) == {"OLD", "OLD_PLUS_ALL_NEW_ELIGIBLE", "OLD_PLUS_NEW_SAFE"}


def test_same_outer_folds_for_every_family_and_set(planted: dict[str, Any]) -> None:
    u = _units(planted["out"])
    by_fold: dict[str, list[np.ndarray]] = {}
    for name, r in u.items():
        if r["result"]["outer"] >= 0:
            by_fold.setdefault(str(r["result"]["outer"]), []).append(r["test_idx"])
    assert set(by_fold) == {"0", "1", "2"}
    for k, arrs in by_fold.items():
        assert len(arrs) >= 4                                   # ENET x 3 primary sets + the secondary contrast
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
        assert p.suffix.lower() in (".csv", ".md", ".json", ".png", ".txt", ".html"), p.name
        if p.suffix.lower() in (".csv", ".md", ".json", ".txt", ".html"):
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


FIT_FUNCTIONS = (("falls_ml.phase5.engine", "run_unit"), ("falls_ml.phase5.engine", "tune_and_fit"), ("falls_ml.phase5.engine", "_tune_linear"),
                 ("falls_ml.phase5.engine", "_tune_xgb"), ("falls_ml.phase5.engine", "fit_linear"), ("falls_ml.phase5.engine", "fit_xgb"),
                 ("falls_ml.phase5.models", "fit_linear"), ("falls_ml.phase5.models", "fit_xgb"), ("falls_ml.phase5.models", "linear_path"),
                 ("falls_ml.phase5.models", "linear_path_task"), ("falls_ml.phase5.models", "xgb_inner_fold"), ("falls_ml.phase5.models", "_train"),
                 ("falls_ml.phase5.explain", "run_fold_explain"), ("falls_ml.phase5.explain", "run_stability"),
                 ("falls_ml.phase5.explain", "permutation_importance"), ("falls_ml.phase5.explain", "shap_importance"),
                 ("falls_ml.phase5.runner", "_execute"), ("falls_ml.phase5.runner", "run_preflight"), ("falls_ml.phase2.enet", "enet_logistic_path"),
                 ("xgboost", "train"), ("optuna", "create_study"))


def _block_fitting(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    import importlib

    calls: list[str] = []
    for mod, name in FIT_FUNCTIONS:
        m = importlib.import_module(mod)
        if hasattr(m, name):
            def boom(*a: Any, _n: str = f"{mod}.{name}", **k: Any) -> Any:
                calls.append(_n)
                raise AssertionError(f"model-fitting function {_n} was called by the dashboard")
            monkeypatch.setattr(m, name, boom)
    return calls


def test_operating_dashboard_in_the_final_report(planted: dict[str, Any]) -> None:
    from falls_ml.phase5.dashboard import CAP_QUESTION, SECONDARY_HEADING

    share = planted["out"] / "share"
    mg = (share / "MANAGEMENT_SUMMARY_HE.md").read_text(encoding="utf-8")
    assert mg.index(CAP_QUESTION) < mg.index(SECONDARY_HEADING) < mg.index("האם הפיצ'רים הנוספים של V21") and mg.count(CAP_QUESTION) == 1
    c = pd.read_csv(share / "TOP3_CAPACITY_COMPARISON.csv")
    e = c[(c["family"] == "ENET") & (c["comparison"] == "OLD_PLUS_ALL_NEW_ELIGIBLE minus OLD")].iloc[0]
    assert e["role"] == "PRIMARY" and float(e["delta_falls_captured"]) > 0 and float(e["delta_false_interventions"]) == -float(e["delta_falls_captured"])
    p3 = pd.read_csv(share / "TOP3_CAPACITY_PRIMARY.csv")
    assert set(p3["capacity_pct"]) == {3.0} and len(p3) == 3 and (p3["method"] == "OUTER_FOLD_CAPACITY_PRIMARY").all()
    fine = pd.read_csv(share / "CAPACITY_CURVE_FINE.csv")
    assert fine["capacity_pct"].max() <= 20.0 and 3.0 in set(fine["capacity_pct"]) and len(fine) <= 2000
    man = json.loads((share / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    assert man["operating_capacity_dashboard"]["no_model_fitted"] is True
    html = (share / "PHASE5_OPERATING_DASHBOARD.html").read_text(encoding="utf-8")
    assert "INTERVENTION CAPACITY" in html and "מה מניע את המודל?" in html and "<script src" not in html


def test_dashboard_command_never_fits_and_keeps_the_results(planted: dict[str, Any], worlds: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """The work-PC command on a COMPLETED run made WITHOUT the dashboard (as by 0.12.1): no model-fitting function is ever invoked, the committed
    units and OOF predictions are byte-identical, the earlier share files are unchanged, the 70% analysis stays as the secondary section."""
    from falls_ml.cli import main
    from falls_ml.phase2.state import Phase2Stop
    from falls_ml.phase5.dashboard import CAP_QUESTION, NEW_SHARE_FILES, SECONDARY_HEADING, original_management, run_dashboard

    out = worlds["base"] / "out dashboard command"
    shutil.copytree(planted["out"], out)
    share = out / "share"
    for n in NEW_SHARE_FILES:                                       # make it look like a folder finished by 0.12.1
        (share / n).unlink(missing_ok=True)
    mp = share / "MANAGEMENT_SUMMARY_HE.md"
    mp.write_text(original_management(mp.read_text(encoding="utf-8")), encoding="utf-8")
    man = json.loads((share / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    man.pop("operating_capacity_dashboard", None)
    (share / "RUN_MANIFEST.json").write_text(json.dumps(man), encoding="utf-8")
    old_share = {p.name: p.read_bytes() for p in share.iterdir() if p.is_file() and p.name not in ("MANAGEMENT_SUMMARY_HE.md", "RUN_MANIFEST.json",
                                                                                                     "PRIVACY_SCAN.json")}
    units, expl = _digest(out / "work" / "units"), _digest(out / "work" / "explain")
    oof = (out / "work" / "analysis" / "OOF_PREDICTIONS_LOCAL.parquet").read_bytes()
    calls = _block_fitting(monkeypatch)
    with pytest.raises(Phase2Stop) as e:                            # the input is not next to the output folder here
        run_dashboard(out, n_boot=50)
    assert e.value.gate == "INPUT_NEEDED"
    rc = main(["meuhedet-phase5-dashboard", "--out", str(out), "--input", str(worlds["planted"]), "--bootstrap", "200"])
    assert rc == 0 and calls == []
    assert _digest(out / "work" / "units") == units and _digest(out / "work" / "explain") == expl
    assert (out / "work" / "analysis" / "OOF_PREDICTIONS_LOCAL.parquet").read_bytes() == oof
    for n, b in old_share.items():
        assert (share / n).read_bytes() == b, n                    # every earlier result file is byte-identical
    for n in NEW_SHARE_FILES:
        assert (share / n).is_file(), n
    scan = json.loads((share / "PRIVACY_SCAN.json").read_text(encoding="utf-8"))
    assert scan["passed"] and scan["files_scanned"] >= len(old_share)
    mg = mp.read_text(encoding="utf-8")
    assert mg.index(CAP_QUESTION) < mg.index(SECONDARY_HEADING) < mg.index("האם הפיצ'רים הנוספים של V21")
    ids = set(worlds["df_planted"]["Customer_Full_ID"].astype(str))
    keys = set(pd.read_parquet(out / "work" / "ANALYSIS_FRAME.parquet", columns=["row_key"])["row_key"])
    html = (share / "PHASE5_OPERATING_DASHBOARD.html").read_text(encoding="utf-8")
    assert not any(i in html for i in ids) and not any(k in html for k in keys) and str(worlds["base"]) not in html
    boot = pd.read_csv(share / "TOP3_CAPACITY_BOOTSTRAP.csv")
    assert set(boot["n_boot"]) == {200} and len(boot) <= 2000
    r2 = run_dashboard(out, input_path=worlds["planted"], n_boot=50)        # idempotent: one 3% section, the original summary kept once
    mg2 = mp.read_text(encoding="utf-8")
    assert r2["status"] == "DASHBOARD_COMPLETE" and mg2.count(CAP_QUESTION) == 1 and mg2.count(SECONDARY_HEADING) == 1 and calls == []
    assert any(p.name.startswith("share_before_dashboard_") for p in (out / "work" / "dashboard").iterdir())


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
    plan["sets"]["OLD_PLUS_ALL_NEW_ELIGIBLE"] = [*plan["sets"]["OLD_PLUS_ALL_NEW_ELIGIBLE"], "hypertension"]      # removed in V21
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


def test_null_new_features_do_not_give_a_useful_verdict(worlds: dict[str, Any]) -> None:
    out = worlds["base"] / "out null"
    if not (out / "share" / "PRIMARY_70_SENSITIVITY_COMPARISON.csv").is_file():
        pytest.skip("depends on test_interrupted_run_resumes_without_recomputing")
    t = pd.read_csv(out / "share" / "PRIMARY_70_SENSITIVITY_COMPARISON.csv")
    v = t[(t["row"] == "VERDICT") & (t["comparison"] == PRIMARY_CMP)].set_index("family")
    assert v.loc["ENET", "verdict"] != "NEW_FEATURES_OPERATIONALLY_USEFUL", v["verdict"].to_dict()
    man = json.loads((out / "share" / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    assert man["overall_answer"] in ("NO", "UNCERTAIN")
    # the patient bootstrap holds the fitted models fixed: under the null a noise feature can still give a narrow interval below 0; the
    # pre-declared rule therefore also requires the improvement in EVERY outer fold and without the questionable predictors
    assert "NEW_FEATURES_OPERATIONALLY_USEFUL" not in set(v["verdict"]), v["verdict"].to_dict()


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
    assert md.rstrip().endswith("STOP - REVIEW REQUIRED") and "O1" in md
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


# ============================================================================ Phase 5.1: PRE / POST on a synthetic 2.2.0 PRE, and the control gate
def test_prepost_rehearsal_with_a_synthetic_pre_run(worlds: dict[str, Any]) -> None:
    from falls_ml.phase2.state import Phase2Stop
    from falls_ml.phase5.engine import outer_folds
    from falls_ml.phase5.prerun import pre_digest
    from falls_ml.phase5.runner import run_phase5
    from tests.unit.test_phase51_repair import make_pre_folder

    pf = worlds["base"] / "out pf for pre"
    r = run_phase5(worlds["planted"], pf, mode="quick", preflight_only=True, overrides=OV, synthetic=True, jobs=2)
    assert r["status"] == "PREFLIGHT_COMPLETE"
    frame = pd.read_parquet(pf / "work" / "ANALYSIS_FRAME.parquet", columns=["row_key"])
    z = np.load(pf / "work" / "Y_FOLDS.npz")
    plan0 = json.loads((pf / "work" / "PLAN.json").read_text(encoding="utf-8"))
    keys, y = frame["row_key"].to_numpy(), z["y"].astype(int)
    outer = outer_folds(y, 3, 2468)                                                   # other folds than the seed gives: adoption must show
    assert not np.array_equal(outer, z["outer"])
    pre = worlds["base"] / "pre 2.2.0 synthetic"
    sets = {k: plan0["sets"][k] for k in ("OLD", "OLD_PLUS_ALL_NEW_ELIGIBLE", "OLD_PLUS_NEW_SAFE")}
    info = make_pre_folder(pre, keys=keys, y=y, outer=outer, input_sha=hashlib.sha256(worlds["planted"].read_bytes()).hexdigest(),
                           input_name=worlds["planted"].name, sets=sets)
    digest = pre_digest(pre)
    out = worlds["base"] / "out post with pre"
    r = _run(worlds["planted"], out, controls=True, pre_run=pre)
    assert r["status"] == "COMPLETE" and r["report"]["prepost"] if "report" in r else r["status"] == "COMPLETE"
    plan = json.loads((out / "work" / "PLAN.json").read_text(encoding="utf-8"))
    assert plan["folds_source"].startswith("PRE") and np.array_equal(np.load(out / "work" / "Y_FOLDS.npz")["outer"], outer)
    assert pre_digest(pre) == digest                                                   # the PRE folder was never written to
    share = out / "share"
    for n in ("TOP3_PRE_POST_HEADLINE.csv", "PRE_POST_PAIRED.csv", "PRE_POST_CONTRAST.csv", "PRE_POST_CORRECTION.csv", "PRE_POST_MEMBERSHIP.csv",
              "PRE_POST_HISTORICAL_70_RULE.csv", "PRECISION_PLANNING.csv"):
        assert (share / n).is_file(), n
    H = pd.read_csv(share / "TOP3_PRE_POST_HEADLINE.csv")
    prim = H[H["role"] == "PRIMARY"]
    assert set(prim["arm"]) == {"OLD", "ADMISSIBLE"} and set(prim["version"]) == {"PRE 2.2.0", "POST 5.1", "DELTA POST minus PRE"}
    T = int(prim[prim["version"] == "POST 5.1"]["selected_total"].iloc[0])
    assert T == round(0.03 * plan["n"] + 1e-9) and (prim["selected_total"].astype(int) == T).all()
    t3 = info["top3"].set_index("feature_set")
    for arm, s_ in (("OLD", "OLD"), ("ADMISSIBLE", "OLD_PLUS_NEW_SAFE")):
        row = prim[(prim["arm"] == arm) & (prim["version"] == "PRE 2.2.0")].iloc[0]
        assert int(row["captured_falls"]) == int(t3.loc[s_, "tp"])                   # the PRE row IS the verified PRE Top-3% result
        d = prim[(prim["arm"] == arm) & (prim["version"] == "DELTA POST minus PRE")].iloc[0]
        post = prim[(prim["arm"] == arm) & (prim["version"] == "POST 5.1")].iloc[0]
        assert int(d["captured_falls"]) == int(post["captured_falls"]) - int(row["captured_falls"])
        assert float(d["delta_captured_falls_ci_low"]) <= float(d["captured_falls"]) <= float(d["delta_captured_falls_ci_high"])
        assert int(d["false_interventions"]) == -int(d["captured_falls"])
    for name in ("MANAGEMENT_SUMMARY_HE.md", "SCIENTIFIC_SUMMARY.md"):
        txt = (share / name).read_text(encoding="utf-8")
        assert txt.index("Top 3%") < txt.index("## ")                                  # the headline comes before every other section
        assert "QUARANTINED" in txt or "בהסגר" in txt                                  # the CCI quarantine is stated
    mem = pd.read_csv(share / "PRE_POST_MEMBERSHIP.csv")
    assert (mem[mem["feature"] == "new_deficit_count_proxy"]["reason"].astype(str).str.contains("R-1")).all()
    assert (mem[mem["feature"] == "com_cci_group"]["reason"].astype(str).str.contains("R-4")).all()
    c = pd.read_csv(share / "PRE_POST_CONTRAST.csv")
    assert set(c["role"]) >= {"PRIMARY_CONTRAST", "AUDIT_ONLY", "SECONDARY_DIAGNOSTIC"}
    man = json.loads((share / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    assert man["pre_run"]["verified"] and man["pre_run"]["folder_name"] == pre.name and str(pre) not in json.dumps(man)
    # a PRE folder changed after the preflight stops the report
    (pre / "share" / "TOP3_CAPACITY_PRIMARY.csv").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(Phase2Stop) as e:
        run_phase5(worlds["planted"], out, mode="quick", report_only=True, overrides=OV, synthetic=True, pre_run=pre, jobs=2)
    assert e.value.gate == "PRE_RUN_MODIFIED"


def test_negative_control_failure_is_a_hard_stop(worlds: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    import falls_ml.phase5.controls as C
    from falls_ml.phase2.state import Phase2Stop
    from falls_ml.phase5.runner import run_phase5

    monkeypatch.setattr(C, "permute_within_folds", lambda y, outer, seed: np.asarray(y).astype(int).copy())   # a broken permutation = information leaks
    out = worlds["base"] / "out control failure"
    with pytest.raises(Phase2Stop) as e:
        _run(worlds["planted"], out, controls=True)
    assert e.value.gate == "NEGATIVE_CONTROL_FAILED"
    res = json.loads((out / "work" / "NEGATIVE_CONTROLS_RESULT.json").read_text(encoding="utf-8"))
    assert res["passed"] is False and res["mean_auroc"] > 0.55
    assert not any(d.name.startswith("PRIMARY__") for d in (out / "work" / "units").iterdir()) if (out / "work" / "units").is_dir() else True
    monkeypatch.undo()
    with pytest.raises(Phase2Stop) as e:                                                # the failed folder can neither fit nor report
        run_phase5(worlds["planted"], out, mode="quick", resume=True, overrides=OV, synthetic=True, jobs=2)
    assert e.value.gate == "NEGATIVE_CONTROL_FAILED"
    with pytest.raises(Phase2Stop) as e:
        run_phase5(worlds["planted"], out, mode="quick", report_only=True, overrides=OV, synthetic=True, jobs=2)
    assert e.value.gate == "NEGATIVE_CONTROL_FAILED"
