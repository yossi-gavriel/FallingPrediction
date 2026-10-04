"""Phase 5 contract tests (fast, synthetic data only).

Brief AB proofs covered here (the rest in tests/unit/test_phase5_e2e.py):
 1  outcome / future fields cannot enter X                      test_outcome_and_future_fields_never_enter_x
 2  post-index predictor leakage hard-stops                     test_post_index_or_ineligible_feature_in_x_hard_stops
 3  an unsafe new feature is excluded                           test_unsafe_leaky_and_undeclared_new_features_are_excluded
 5-8 inner selection never sees the outer labels                test_unit_choices_do_not_depend_on_outer_labels (LASSO, ENET, XGB)
 7  the 70% threshold comes from inner OOF only                 test_threshold_is_the_highest_reaching_the_target / ..._outer_labels
 11 the paired patient bootstrap works                          test_paired_bootstrap_*
 13 a GPU failure falls back to CPU                             test_gpu_failure_falls_back_to_cpu
 15 Windows paths                                               test_windows_paths_are_detected_in_share_text / CLI help (utf-8)
 18 Phase 2 / 3 / 4 files unchanged                             test_phase2_3_4_files_are_unchanged
"""

from __future__ import annotations

import hashlib
import inspect
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]


# ============================================================================ fixtures
@pytest.fixture(scope="module")
def prepared(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    from falls_ml.phase3.runner import load_all
    from falls_ml.phase4 import sealed
    from falls_ml.phase4.common import input_identity
    from falls_ml.phase5.config import load_phase5_config
    from falls_ml.phase5.data import prepare
    from falls_ml.phase5.runner import build_sets
    from falls_ml.phase5.synthetic import make_v21, write_v21_csv

    d = tmp_path_factory.mktemp("p5c")
    df, facts = make_v21(2500, scenario="planted", seed=31)
    csv = write_v21_csv(df, d / "v21.csv")
    cfg = load_phase5_config(mode="quick", overrides={"eligibility": {"min_known_observed_rows": 20}})
    L = load_all(cfg["phase3_config"])
    sealed.READ_LOG.clear()
    P = prepare(csv, cfg, L, input_info=input_identity(csv))
    reads = list(sealed.READ_LOG)
    S = build_sets(P.registry, P.meta, cfg)
    return {"P": P, "cfg": cfg, "L": L, "csv": csv, "df": df, "reads": reads, "S": S, "dir": d}


def _small_ctx(prepared: dict[str, Any], out: Path, y: np.ndarray | None = None, overrides: dict[str, Any] | None = None) -> Any:
    from falls_ml.phase5.config import load_phase5_config
    from falls_ml.phase5.engine import Ctx, outer_folds
    from falls_ml.phase5.models import DeviceState

    P, S = prepared["P"], prepared["S"]
    ov = {"eligibility": {"min_known_observed_rows": 20}, "cv": {"outer_folds": 3, "inner_folds": 3},
          "modes": {"quick": {"lasso": {"n_lambda": 6}, "enet": {"n_lambda": 4, "l1_ratios": [0.5]}, "xgb": {"n_trials": 3, "n_startup_trials": 2}}}}
    cfg = load_phase5_config(mode="quick", overrides={**ov, **(overrides or {})})
    order = np.argsort(P.frame["row_key"].to_numpy(), kind="mergesort")
    frame = P.frame.iloc[order].reset_index(drop=True)
    yy = P.y[order].astype(int) if y is None else y
    sets = {k: v for k, v in S["sets"].items() if k in ("OLD", "OLD_PLUS_NEW_SAFE")}
    folds = outer_folds(P.y[order].astype(int), cfg.outer_folds, 7)
    return Ctx(out=out, frame=frame, y=yy, meta=P.meta, sets=sets, outer=folds, cfg=cfg, seed=99, jobs=2, state=DeviceState("cpu"))


# ============================================================================ 1-3 X sealing and eligibility
def test_outcome_and_future_fields_never_enter_x(prepared: dict[str, Any]) -> None:
    P, S = prepared["P"], prepared["S"]
    sealed = P.sealed
    for c in ("Fall_Next_180D_Ind", "Next_Fall_Date_180D", "Days_To_Next_Fall_180D", "Followup_End_Date", "Is_Censored_180D", "Fall_Next_365D_Ind",
              "Hospital_Discharge_Date", "Label_Reason_180D", "Fall_Next_30D_Ind"):
        assert c in sealed, c
    x_reads = [cols for name, cols in prepared["reads"] if "Customer_Full_ID" in cols and "Is_Eligible_Cohort" in cols and len(cols) > 10]
    assert x_reads, "the X reader was not used"
    for cols in x_reads:
        assert not set(cols) & set(sealed), "a sealed outcome / future column was requested by the X reader"
    assert not set(P.frame.columns) & set(sealed)
    reg = P.registry.set_index("feature")
    for name, feats in S["sets"].items():
        for f in feats:
            raw = [c.strip() for c in str(reg.loc[f, "raw_columns"]).split(";")]
            assert not set(raw) & set(sealed), (name, f)
    cat = P.catalogue.set_index("raw_column")
    assert cat.loc["Fall_Next_365D_Ind", "eligibility"] == "INELIGIBLE_LEAKAGE"
    assert cat.loc["Hospital_Discharge_Date", "eligibility"] == "INELIGIBLE_LEAKAGE"


def test_sealed_reader_refuses_an_outcome_column(prepared: dict[str, Any]) -> None:
    from falls_ml.phase2.state import Phase2Stop
    from falls_ml.phase4.sealed import read_columns

    with pytest.raises(Phase2Stop) as e:
        read_columns(prepared["csv"], ["Customer_Full_ID", "Fall_Next_180D_Ind"], prepared["P"].sealed, prepared["L"]["contract"])
    assert e.value.gate == "SEALED_COLUMN_REQUESTED"


def test_unsafe_leaky_and_undeclared_new_features_are_excluded(prepared: dict[str, Any]) -> None:
    P, S = prepared["P"], prepared["S"]
    reg = P.registry.set_index("feature")
    assert reg.loc["new_syncope_ind", "class"] == "INELIGIBLE_TIMING"            # post-index records (outcome-related) beyond the gates
    assert reg.loc["new_gait_abnormality_ind", "class"] == "INELIGIBLE_LEAKAGE"   # single-feature AUROC >= 0.80
    assert reg.loc["new_dizziness_ind", "class"] == "SAFE_VERIFIED"
    allx = {f for v in S["sets"].values() for f in v}
    assert "new_syncope_ind" not in allx and "new_gait_abnormality_ind" not in allx
    und = set(P.undeclared["column"])
    assert "Mystery_Score_V21" in und and "Mystery_Score_V21" not in " ".join(allx)
    assert S["sets"]["OLD"] == [f for f in S["sets"]["OLD"] if P.meta[f]["origin"] == "OLD"]
    assert "new_dizziness_ind" in S["sets"]["OLD_PLUS_NEW_SAFE"] and "new_dizziness_ind" not in S["sets"]["OLD"]
    # MEFI columns of the brief are 2025 contract columns -> OLD, never NEW
    olds = P.catalogue[P.catalogue["old_or_new"] == "OLD_2025_CONTRACT"]["raw_column"].tolist()
    assert "MEFI_Group_At_Index" in olds


def test_registry_attested_and_low_risk_sets(prepared: dict[str, Any]) -> None:
    P, S = prepared["P"], prepared["S"]
    reg = P.registry.set_index("feature")
    assert reg.loc["new_registry_opiate_ind", "class"] == "SAFE_ATTESTED"
    assert reg.loc["new_registry_opiate_ind", "availability_risk"] == "HIGH"
    assert "new_registry_opiate_ind" in S["sets"]["OLD_PLUS_NEW_SAFE"]
    assert "new_registry_opiate_ind" not in S["sets"]["OLD_PLUS_NEW_VERIFIED_ONLY"]
    assert "new_registry_opiate_ind" not in S["sets"]["OLD_PLUS_NEW_LOW_AVAILABILITY_RISK"]


def test_post_index_or_ineligible_feature_in_x_hard_stops(prepared: dict[str, Any]) -> None:
    from falls_ml.phase2.state import Phase2Stop
    from falls_ml.phase5.runner import x_guard

    P, S, cfg = prepared["P"], prepared["S"], prepared["cfg"]
    x_guard(S["sets"], P.registry, S["kinds"], P.sealed, list(P.frame.columns), cfg)          # the real sets pass
    bad = {**S["sets"], "OLD_PLUS_NEW_SAFE": [*S["sets"]["OLD_PLUS_NEW_SAFE"], "new_syncope_ind"]}
    with pytest.raises(Phase2Stop) as e:
        x_guard(bad, P.registry, S["kinds"], P.sealed, list(P.frame.columns), cfg)
    assert e.value.gate == "X_LEAKAGE"
    with pytest.raises(Phase2Stop):
        x_guard(S["sets"], P.registry, S["kinds"], P.sealed, [*P.frame.columns, "Fall_Next_180D_Ind"], cfg)
    old_with_new = {**S["sets"], "OLD": [*S["sets"]["OLD"], "new_dizziness_ind"]}
    with pytest.raises(Phase2Stop):
        x_guard(old_with_new, P.registry, S["kinds"], P.sealed, list(P.frame.columns), cfg)


def test_unknown_cells_take_the_no_record_state() -> None:
    from falls_ml.phase5.data import _no_record_state

    assert _no_record_state("binary", "unexpected") == 0.0
    assert _no_record_state("count", "source_absent") == 0.0
    assert math.isnan(_no_record_state("days", "not_assessed"))
    assert math.isnan(_no_record_state("binary", "no_event"))
    assert _no_record_state("binary", "no_event", op="present") == 0.0


# ============================================================================ thresholds, objective, decision rule
def test_threshold_is_the_highest_reaching_the_target() -> None:
    from falls_ml.phase5.thresholds import operating_point

    y = np.array([1, 1, 1, 0, 1, 0, 0, 1, 0, 0])
    p = np.array([.9, .8, .7, .65, .6, .5, .4, .3, .2, .1])
    op = operating_point(y, p, 0.70)            # 5 events: >= 3.5 -> 4 captured at threshold 0.6
    assert op["threshold"] == pytest.approx(0.6) and op["tp"] == 4 and op["fp"] == 1 and op["flagged"] == 5
    assert op["sensitivity"] == pytest.approx(0.8) and op["false_alert_share"] == pytest.approx(0.2)
    ties = np.array([.9, .5, .5, .5, .5, .5, .1, .1, .1, .1])
    opt = operating_point(y, ties, 0.70)
    assert opt["flagged"] == 6                  # ties at the threshold are flagged together
    assert not operating_point(np.zeros(5), np.ones(5), 0.7)["feasible"]


def test_objective_prefers_fewer_flagged_then_fewer_false_alerts() -> None:
    from falls_ml.phase5.thresholds import objective

    rng = np.random.default_rng(1)
    y = (rng.random(4000) < 0.05).astype(float)
    good = y * 1.5 + rng.normal(0, 1, 4000)
    bad = y * 0.3 + rng.normal(0, 1, 4000)
    og = objective(y, 1 / (1 + np.exp(-good)), target=0.7, complexity=5, tie=1e-4)
    ob = objective(y, 1 / (1 + np.exp(-bad)), target=0.7, complexity=1, tie=1e-4)
    assert og["key"] < ob["key"] and og["flagged_share"] < ob["flagged_share"]
    assert all(math.isfinite(v) for v in og["key"])


def test_decision_rule() -> None:
    from falls_ml.phase5.analysis import decide, overall
    from falls_ml.phase5.config import load_phase5_config

    cfg = load_phase5_config()

    def cmp(d: float, hi: float, slope: float = 1.0, b_lo: float = -0.001, folds: list[float] | None = None) -> dict[str, Any]:
        m = {"calibration_slope": slope, "calibration_intercept": 0.05}
        return {"delta_op_false_alert_share": d, "delta_op_false_alert_share_ci_high": hi, "delta_desc_fas_0.70": d, "delta_brier_ci_low": b_lo,
                "a": {"calibration_slope": 1.0, "calibration_intercept": 0.05}, "b": m, "fold_delta_false_alert_share": folds or [d] * 5}

    assert decide(cmp(-0.05, -0.01), cmp(-0.04, 0.01), cmp(-0.03, 0.01), no_new=False, ver_same=False, low_same=False, cfg=cfg)["verdict"] == \
        "NEW_FEATURES_OPERATIONALLY_USEFUL"
    assert decide(cmp(-0.05, 0.01), None, None, no_new=False, ver_same=True, low_same=True, cfg=cfg)["verdict"] == "PROMISING_BUT_NOT_ROBUST"
    # a "significant" bootstrap interval that one outer fold contradicts is model noise, not a robust gain
    assert decide(cmp(-0.05, -0.01, folds=[-0.06, -0.07, 0.01, -0.05, -0.04]), None, None, no_new=False, ver_same=True, low_same=True,
                  cfg=cfg)["verdict"] == "PROMISING_BUT_NOT_ROBUST"
    # the gain disappears without the attested / high-risk predictors -> not robust
    assert decide(cmp(-0.05, -0.01), cmp(0.0, 0.02), cmp(-0.02, 0.0), no_new=False, ver_same=False, low_same=False, cfg=cfg)["verdict"] == \
        "PROMISING_BUT_NOT_ROBUST"
    assert decide(cmp(-0.05, -0.01, slope=0.5), None, None, no_new=False, ver_same=True, low_same=True, cfg=cfg)["verdict"] == "PROMISING_BUT_NOT_ROBUST"
    assert decide(cmp(0.01, 0.03), None, None, no_new=False, ver_same=True, low_same=True, cfg=cfg)["verdict"] == "NO_ROBUST_OPERATIONAL_GAIN"
    assert decide(None, None, None, no_new=True, ver_same=True, low_same=True, cfg=cfg)["verdict"] == "NO_ELIGIBLE_NEW_FEATURES"
    U, P_, N = "NEW_FEATURES_OPERATIONALLY_USEFUL", "PROMISING_BUT_NOT_ROBUST", "NO_ROBUST_OPERATIONAL_GAIN"
    assert overall({"LASSO": U, "ENET": U, "XGB": N}) == "YES"
    assert overall({"LASSO": U, "ENET": P_, "XGB": N}) == "UNCERTAIN"
    assert overall({"LASSO": N, "ENET": N, "XGB": N}) == "NO"


# ============================================================================ 11 paired bootstrap
def test_fast_weighted_metrics_equal_the_reference_implementation() -> None:
    from falls_ml.evaluation.metrics import _auroc, _pr_auc
    from falls_ml.phase5.metrics import _Sorted

    rng = np.random.default_rng(3)
    y = (rng.random(3000) < 0.08).astype(float)
    p = np.round(rng.random(3000) * 0.5 + 0.3 * y, 3)          # ties on purpose
    s = _Sorted(y, p).stats(np.ones(len(y)), (0.7,))
    assert s["auroc"] == pytest.approx(_auroc(y, p), abs=1e-12)
    assert s["ap"] == pytest.approx(_pr_auc(y, p), abs=1e-12)
    w = np.bincount(rng.integers(0, 3000, 3000), minlength=3000).astype(float)
    rep = np.repeat(np.arange(3000), w.astype(int))
    s2 = _Sorted(y, p).stats(w, (0.7,))
    assert s2["auroc"] == pytest.approx(_auroc(y[rep], p[rep]), abs=1e-10)
    assert s2["ap"] == pytest.approx(_pr_auc(y[rep], p[rep]), abs=1e-10)


def test_paired_bootstrap_identical_models_give_zero_and_a_better_model_a_negative_interval() -> None:
    from falls_ml.phase5.metrics import paired_bootstrap
    from falls_ml.phase5.thresholds import operating_point

    rng = np.random.default_rng(5)
    y = (rng.random(5000) < 0.05).astype(float)
    pa = 1 / (1 + np.exp(-(y * 0.8 + rng.normal(0, 1, 5000) - 3)))
    fa = pa >= operating_point(y, pa, 0.7)["threshold"]
    r = paired_bootstrap(y, pa, pa.copy(), fa, fa.copy(), n_boot=200, seed=1)
    assert r["delta_op_false_alert_share"] == 0 and r["delta_op_false_alert_share_ci_low"] == 0 and r["delta_op_false_alert_share_ci_high"] == 0
    assert r["delta_auroc_ci_low"] == 0 and r["delta_auroc_ci_high"] == 0
    pb = 1 / (1 + np.exp(-(y * 2.5 + rng.normal(0, 1, 5000) - 3)))
    fb = pb >= operating_point(y, pb, 0.7)["threshold"]
    r2 = paired_bootstrap(y, pa, pb, fa, fb, n_boot=300, seed=2)
    assert r2["delta_op_false_alert_share_ci_high"] < 0 and r2["delta_ap_ci_low"] > 0 and r2["delta_op_false_alerts_per_10000"] < 0
    r3 = paired_bootstrap(y, pa, pb, fa, fb, n_boot=300, seed=2)
    assert r3["delta_op_false_alert_share_ci_high"] == r2["delta_op_false_alert_share_ci_high"]     # seeded, reproducible


# ============================================================================ 5-8 the outer labels never influence the inner choices
@pytest.mark.parametrize("family", ["LASSO", "ENET", "XGB"])
def test_unit_choices_do_not_depend_on_outer_labels(prepared: dict[str, Any], tmp_path: Path, family: str) -> None:
    from falls_ml.phase5.engine import UnitSpec, load_result, run_unit, tune_and_fit

    assert "y_te" not in inspect.signature(tune_and_fit).parameters        # the unit never receives the holdout outcome
    c1 = _small_ctx(prepared, tmp_path / "a")
    spec = UnitSpec(uid=f"PRIMARY|{family}|OLD_PLUS_NEW_SAFE|outer0", stage="PRIMARY", family=family, setname="OLD_PLUS_NEW_SAFE", outer=0)
    r1 = run_unit(c1, spec)
    y2 = c1.y.copy()
    te = c1.outer == 0
    y2[te] = 1 - y2[te]                                                    # every holdout label flipped
    c2 = _small_ctx(prepared, tmp_path / "b", y=y2)
    r2 = run_unit(c2, spec)
    js = __import__("json").dumps
    assert r1["config"] == r2["config"] and r1["thresholds"] == r2["thresholds"]
    assert js(r1["inner_objective"], sort_keys=True, default=str) == js(r2["inner_objective"], sort_keys=True, default=str)
    a1, a2 = load_result(c1, spec)["arrays"], load_result(c2, spec)["arrays"]
    assert np.array_equal(a1["inner_oof"], a2["inner_oof"]) and np.array_equal(a1["p_test"], a2["p_test"]) and np.array_equal(a1["test_idx"], a2["test_idx"])
    t = r1["thresholds"]["0.70"]
    assert t == pytest.approx(__import__("falls_ml.phase5.thresholds", fromlist=["x"]).operating_point(
        c1.y[a1["train_idx"]].astype(float), a1["inner_oof"], 0.70)["threshold"])                       # 70% threshold = inner OOF only


def test_committed_unit_is_never_recomputed(prepared: dict[str, Any], tmp_path: Path) -> None:
    from falls_ml.phase5.engine import UnitSpec, is_complete, run_unit, unit_dir

    c = _small_ctx(prepared, tmp_path)
    spec = UnitSpec(uid="PRIMARY|LASSO|OLD|outer1", stage="PRIMARY", family="LASSO", setname="OLD", outer=1)
    run_unit(c, spec)
    d = unit_dir(c, spec)
    before = {p.name: (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest()) for p in d.iterdir()}
    assert is_complete(c, spec)
    run_unit(c, spec)
    assert before == {p.name: (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest()) for p in d.iterdir()}
    (d / "result.json").write_text("{}", encoding="utf-8")                  # a damaged commit is detected
    assert not is_complete(c, spec)


# ============================================================================ 13 GPU -> CPU
def test_gpu_failure_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    import xgboost

    from falls_ml.phase5.config import load_phase5_config
    from falls_ml.phase5.models import DeviceState, xgb_inner_fold
    from falls_ml.phase5.resources import resolve_device

    real = xgboost.train
    calls: list[str] = []

    def fake(params: dict[str, Any], *a: Any, **k: Any) -> Any:
        calls.append(params["device"])
        if params["device"] != "cpu":
            raise xgboost.core.XGBoostError("CUDA error: no CUDA-capable device is detected (simulated)")
        return real(params, *a, **k)

    monkeypatch.setattr(xgboost, "train", fake)
    cfg = load_phase5_config()
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 5)).astype(np.float32)
    y = (X[:, 0] + rng.normal(size=600) > 1).astype(float)
    logs: list[str] = []
    st = DeviceState("cuda", log=logs.append)
    params = {"max_depth": 3, "learning_rate": 0.1, "min_child_weight": 1.0, "subsample": 1.0, "colsample_bytree": 1.0, "reg_alpha": 0.0, "reg_lambda": 1.0, "gamma": 0.0}
    r = xgb_inner_fold(X, y, X[:50], params, cfg, seed=1, nthread=1, state=st)
    assert len(r["pred"]) == 50 and st.device == "cpu" and st.fallbacks and calls[:2] == ["cuda", "cpu"]
    assert any("GPU FAILURE" in m for m in logs)
    xgb_inner_fold(X, y, X[:50], params, cfg, seed=2, nthread=1, state=st)
    assert calls[-1] == "cpu"                                               # the rest of the run stays on CPU
    monkeypatch.setattr(xgboost, "train", real)
    dev = resolve_device("gpu")
    assert dev["device"] in ("cpu", "cuda") and dev["reason"]


# ============================================================================ resources, config, status
def test_default_jobs_never_take_every_core(monkeypatch: pytest.MonkeyPatch) -> None:
    from falls_ml.phase5 import resources
    from falls_ml.phase5.config import load_phase5_config

    cfg = load_phase5_config()
    for n, want in ((1, 1), (2, 1), (8, 4), (16, 9), (64, 12)):
        monkeypatch.setattr(resources, "logical_cpus", lambda n=n: n)
        assert resources.default_jobs(cfg) == want


def test_config_is_validated_and_predeclared() -> None:
    from falls_ml.errors import ConfigError
    from falls_ml.phase5.config import load_phase5_config

    cfg = load_phase5_config()
    assert cfg["families"] == ["LASSO", "ENET", "XGB"] and cfg.primary_sensitivity == 0.70 and cfg.outer_folds == 5 and cfg.inner_folds == 5
    assert cfg["xgb"]["fixed"]["scale_pos_weight"] == 1.0
    with pytest.raises(ConfigError):
        load_phase5_config(overrides={"families": ["LASSO", "ENET", "XGB", "RF"]})
    with pytest.raises(ConfigError):
        load_phase5_config(overrides={"operating": {"primary_sensitivity": 0.6}})
    assert load_phase5_config(overrides={"cv": {"outer_folds": 3}}).sha256 != cfg.sha256


def test_status_is_read_only_and_reports_a_missing_run(tmp_path: Path) -> None:
    from falls_ml.phase5.status import phase5_status

    r = phase5_status(tmp_path)
    assert r["status"] == "NOT_STARTED" and not any(tmp_path.iterdir())


# ============================================================================ privacy
def test_share_publication_fails_closed_on_an_identifier(prepared: dict[str, Any], tmp_path: Path) -> None:
    from falls_ml.phase2.state import Phase2Stop
    from falls_ml.phase5.report import publish

    tmp = tmp_path / ".tmp-share"
    tmp.mkdir()
    some_id = str(prepared["df"]["Customer_Full_ID"].iloc[5])
    (tmp / "x.md").write_text(f"patient {some_id}\n", encoding="utf-8")
    with pytest.raises(Phase2Stop) as e:
        publish(tmp_path, tmp, prepared["csv"], prepared["P"].frame)
    assert e.value.gate == "PRIVACY_SCAN" and not (tmp_path / "share").exists()
    tmp.mkdir()
    pd.DataFrame({"row_key": ["a"], "p": [0.1]}).to_csv(tmp / "t.csv", index=False)
    with pytest.raises(Phase2Stop):
        publish(tmp_path, tmp, prepared["csv"], prepared["P"].frame)
    tmp.mkdir()
    (tmp / "m.pkl").write_bytes(b"x")
    with pytest.raises(Phase2Stop):
        publish(tmp_path, tmp, prepared["csv"], prepared["P"].frame)


def test_small_cells_and_their_rates_are_suppressed() -> None:
    from falls_ml.phase5.report import suppress, suppress_grid

    t = suppress(pd.DataFrame({"n": [500, 500], "op_tp": [5, 50], "op_ppv": [0.1, 0.2], "n_features": [3, 3], "captured_falls": [5, 40]}))
    assert t.loc[0, "op_tp"] == "<10" and t.loc[0, "op_ppv"] == "suppressed" and t.loc[0, "n_features"] == 3 and t.loc[1, "op_ppv"] == 0.2
    g = suppress_grid(pd.DataFrame({"threshold": [.5, .1], "tp": [3, 40], "fp": [20, 300], "fn": [37, 0], "tn": [400, 120], "sensitivity": [.07, 1.0]}))
    assert g.loc[0, "tp"] == "<10" and g.loc[0, "sensitivity"] == "suppressed" and g.loc[1, "sensitivity"] == 1.0


# ============================================================================ 15 Windows paths / CLI
def test_windows_paths_are_detected_in_share_text() -> None:
    from falls_ml.phase2.stages_report import _path_hits

    win = "C:" + "\\" + "\\".join(["Users", "someone", "Downloads", "out5", "share"])       # built at run time: no literal path in the repository
    mac = "/" + "/".join(["Users", "someone", "data"])
    assert _path_hits(f"see {win}", [])
    assert _path_hits(f"see {mac}", [])
    assert not _path_hits("aggregate table, no path", [])


def test_cli_help_lists_the_phase5_commands_utf8() -> None:
    env = {**__import__("os").environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONIOENCODING": "utf-8"}
    r = subprocess.run([sys.executable, "-m", "falls_ml", "meuhedet-phase5", "--help"], capture_output=True, text=True, encoding="utf-8", env=env, cwd=ROOT)
    assert r.returncode == 0
    for flag in ("--mode", "--device", "--jobs", "--resume", "--preflight-only", "--estimate", "--status", "--report-only"):
        assert flag in r.stdout
    r2 = subprocess.run([sys.executable, "-m", "falls_ml", "meuhedet-phase5-synthetic", "--help"], capture_output=True, text=True, encoding="utf-8", env=env,
                        cwd=ROOT)
    assert r2.returncode == 0 and "--scenario" in r2.stdout


def test_out_folder_guards(tmp_path: Path) -> None:
    from falls_ml.phase2.state import Phase2Stop
    from falls_ml.phase5.runner import guard_out

    for m in ("PHASE2_PLAN.json", "PHASE3_PLAN.json", "PHASE4_PLAN.json"):
        d = tmp_path / m.split("_")[0]
        d.mkdir()
        (d / m).write_text("{}", encoding="utf-8")
        with pytest.raises(Phase2Stop) as e:
            guard_out(d, None, False)
        assert e.value.gate == "OUT_IS_EARLIER_PHASE"
    src = tmp_path / "x.csv"
    src.write_text("a\n", encoding="utf-8")
    with pytest.raises(Phase2Stop):
        guard_out(tmp_path, src, False)


# ============================================================================ 18 Phase 2 / 3 / 4 unchanged
@pytest.mark.parametrize("manifest", ["PHASE2_0.8.1_PROTECTED.sha256", "PHASE3_0.9.0_PROTECTED.sha256", "PHASE4_0.10.0_PROTECTED.sha256"])
def test_phase2_3_4_files_are_unchanged(manifest: str) -> None:
    lines = [ln for ln in (ROOT / "configs/meuhedet" / manifest).read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]
    assert len(lines) > 10
    bad = []
    for ln in lines:
        digest, rel = ln.split("  ", 1)
        p = ROOT / rel
        if not p.is_file() or hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest() != digest:
            bad.append(rel)
    assert not bad, f"protected earlier-phase files changed: {bad[:10]}"
