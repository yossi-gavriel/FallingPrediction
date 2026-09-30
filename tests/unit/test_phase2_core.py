"""Phase 2 building blocks (synthetic data only): durable writes, the item/stage commit protocol and recovery, gates, the catalogue loader,
the elastic-net solver, screening / redundancy, feature sets and design encodings, metrics, the deterministic resumable Optuna study and
the validation seal."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from falls_ml.phase2 import durable as D
from falls_ml.phase2.state import STOP_HARD, STOP_INVESTIGATION, EventLog, Phase2Stop, Run, item_seed, safe_item_id


def _run(tmp_path: Path, attempt: int = 1, accepted: dict[str, str] | None = None) -> Run:
    out = tmp_path / "run"
    out.mkdir(parents=True, exist_ok=True)
    return Run(out=out, plan={}, plan_sha="p" * 64, code_sha="c" * 64, attempt=attempt, seed=7, events=EventLog(out / "logs" / "events.jsonl", attempt),
               accepted_gates=accepted or {}, readonly=False)


def xgb_available() -> bool:
    try:
        import xgboost  # noqa: F401
        return True
    except Exception:  # noqa: BLE001 - a missing OpenMP runtime raises XGBoostError, not ImportError
        return False


# ============================================================================ durable
def test_durable_write_replace_and_jsonl_torn_tail(tmp_path: Path) -> None:
    p = D.write_json(tmp_path / "a.json", {"x": 1})
    D.write_json(p, {"x": 2})
    assert json.loads(p.read_text(encoding="utf-8"))["x"] == 2
    assert not list(tmp_path.glob(".a.json.tmp-*"))
    log = tmp_path / "l.jsonl"
    D.append_jsonl(log, {"i": 1})
    D.append_jsonl(log, {"i": 2})
    with open(log, "a", encoding="utf-8") as fh:
        fh.write('{"i": 3, "trunc')            # power loss in the middle of an append
    recs, tail = D.read_jsonl(log)
    assert [r["i"] for r in recs] == [1, 2] and tail is not None
    moved = D.repair_jsonl(log, tmp_path / "l.torn")
    assert moved and (tmp_path / "l.torn").read_text(encoding="utf-8").startswith('{"i": 3')
    recs2, tail2 = D.read_jsonl(log)
    assert [r["i"] for r in recs2] == [1, 2] and tail2 is None


def test_synced_folder_detection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert D.synced_folder(tmp_path) is None
    assert D.synced_folder(tmp_path / "OneDrive - Meuhedet" / "runs") is not None
    monkeypatch.setenv("OneDrive", str(tmp_path / "od"))
    assert D.synced_folder(tmp_path / "od" / "x") is not None


def test_rename_dir_refuses_existing_target(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    with pytest.raises(FileExistsError):
        D.rename_dir(tmp_path / "a", tmp_path / "b")


# ============================================================================ item / stage protocol
def test_item_commit_and_reuse(tmp_path: Path) -> None:
    run = _run(tmp_path)
    st = run.stage("06", "lasso")
    calls: list[int] = []

    def fn(tmp: Path, seed: int) -> dict[str, Any]:
        calls.append(seed)
        D.write_json(tmp / "x.json", {"v": 1})
        return {"value": 3}

    r1 = st.item("A__1", fn)
    assert r1 == {"value": 3} and (st.items_dir / "A__1.COMPLETE.json").is_file()
    rec = json.loads((st.items_dir / "A__1.COMPLETE.json").read_text(encoding="utf-8"))
    assert rec["files"]["x.json"] == D.sha256_file(st.items_dir / "A__1" / "x.json") and rec["seed"] == item_seed(7, "S06_lasso", "A__1")
    # a new attempt reuses the committed item without calling fn
    run2 = _run(tmp_path, attempt=2)
    assert run2.stage("06", "lasso").item("A__1", fn) == {"value": 3}
    assert len(calls) == 1


def test_recovery_moves_incomplete_work_aside_and_recomputes(tmp_path: Path) -> None:
    run = _run(tmp_path)
    st = run.stage("06", "lasso")
    st.items_dir.mkdir(parents=True)
    (st.items_dir / ".tmp-B-a1").mkdir()                       # crash mid-item
    (st.items_dir / ".tmp-B-a1" / "partial.npz").write_bytes(b"\x00\x01")
    (st.items_dir / "C").mkdir()                               # crash after the rename, before the commit record
    (st.items_dir / "C" / "oof.npz").write_bytes(b"x")
    (st.items_dir / "E").mkdir()                               # a torn commit record
    (st.items_dir / "E.COMPLETE.json").write_text('{"files": {"a": ', encoding="utf-8", newline="\n")
    run2 = _run(tmp_path, attempt=2)
    st2 = run2.stage("06", "lasso")
    assert not st2.done("C") and not st2.done("E")
    aside = st2.dir / "_incomplete" / "attempt-2"
    assert (aside / ".tmp-B-a1" / "partial.npz").exists() and (aside / "C" / "oof.npz").exists() and (aside / "E.COMPLETE.json").exists()
    assert st2.item("C", lambda tmp, seed: {"ok": True}) == {"ok": True}


def test_changed_committed_file_is_a_hard_stop(tmp_path: Path) -> None:
    run = _run(tmp_path)
    st = run.stage("06", "lasso")
    st.item("A", lambda tmp, seed: (D.write_str(tmp / "t.txt", "one"), {"r": 1})[1])
    p = st.items_dir / "A" / "t.txt"
    os.chmod(p, 0o644)
    p.write_text("two", encoding="utf-8", newline="\n")
    with pytest.raises(Phase2Stop) as e:
        _run(tmp_path, attempt=2).stage("06", "lasso").done("A")
    assert e.value.gate == "COMPLETED_ARTIFACT_CHANGED" and e.value.kind == STOP_HARD


def test_stage_hash_chain(tmp_path: Path) -> None:
    run = _run(tmp_path)
    s1 = run.stage("01", "cohort")
    f1 = D.write_str(run.artifacts / "one.txt", "a")
    s1.finalize([f1])
    s2 = run.stage("02", "registry", ("01",))
    f2 = D.write_str(run.artifacts / "two.txt", "b")
    s2.finalize([f2])
    assert _run(tmp_path, 2).stage("02", "registry", ("01",)).complete_record() is not None
    rec = json.loads((run.out / "STAGE_01_COMPLETE.json").read_text(encoding="utf-8"))
    rec["summary"] = {"tampered": True}
    (run.out / "STAGE_01_COMPLETE.json").write_text(json.dumps(rec), encoding="utf-8", newline="\n")
    with pytest.raises(Phase2Stop, match="upstream"):
        _run(tmp_path, 3).stage("02", "registry", ("01",)).complete_record()


def test_gate_stops_unless_accepted(tmp_path: Path) -> None:
    run = _run(tmp_path)
    run.gate("IMPLAUSIBLE_GAIN_OOF", False, "never")
    with pytest.raises(Phase2Stop) as e:
        run.gate("IMPLAUSIBLE_GAIN_OOF", True, "jump", ["config: X"])
    assert e.value.kind == STOP_INVESTIGATION
    ok = _run(tmp_path, 2, {"IMPLAUSIBLE_GAIN_OOF": "investigated: synthetic planted signal"})
    ok.gate("IMPLAUSIBLE_GAIN_OOF", True, "jump")
    recs, _ = D.read_jsonl(ok.out / "logs" / "gates.jsonl")
    assert [r["decision"] for r in recs] == ["STOPPED", "ACCEPTED"]


def test_item_ids_are_bounded_and_safe() -> None:
    long = "LASSO__" + "X" * 200 + "__outer3"
    s = safe_item_id(long)
    assert len(s) <= 90 and s != safe_item_id(long + "y")
    assert safe_item_id("a:b/c") == "a_b_c"


def test_crash_point_is_inert_without_the_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    from falls_ml.phase2.state import crash_point

    monkeypatch.delenv("FALLS_ML_PHASE2_CRASH_AT", raising=False)
    crash_point("S06_lasso", "A", "item_start")          # must not exit


# ============================================================================ catalogue / config
@pytest.fixture(scope="module")
def loaded() -> dict[str, Any]:
    from falls_ml.data.meuhedet_wide import load_wide_contract, load_wide_mapping
    from falls_ml.eda.dictionary import load_data_dictionary

    m = load_wide_mapping()
    c = load_wide_contract(m.contract_path)
    d = load_data_dictionary("configs/meuhedet/wide_v1_data_dictionary.yaml", c)
    return {"mapping": m, "contract": c, "dictionary": d}


def test_catalogue_loads_and_covers_every_predictor_column(loaded: dict[str, Any]) -> None:
    from falls_ml.phase2.config import load_catalogue, load_phase2_config

    cat = load_catalogue("configs/meuhedet/phase2_features.yaml", loaded["contract"], loaded["dictionary"], loaded["mapping"])
    assert len(cat.domains) == 10 and cat.waterfall[-1] == "OTHER_NATIVE"
    assert all(f.form is not None for f in cat.features if f.missing == "not_assessed")
    assert {cat.indicator_of(s) for s in cat.forms} <= set(cat.names)
    assert not set(cat.quarantine) & {c for f in cat.features for c in f.inputs}
    cfg = load_phase2_config()
    assert float(cfg["selection"]["principal_capacity"]) == 0.10


def test_catalogue_rejects_quarantined_or_unknown_inputs(loaded: dict[str, Any], tmp_path: Path) -> None:
    import yaml

    from falls_ml.errors import ConfigError
    from falls_ml.phase2.config import load_catalogue

    raw = yaml.safe_load(Path("configs/meuhedet/phase2_features.yaml").read_text(encoding="utf-8"))
    raw["features"].append({"name": "bad_q", "domain": "SOCIAL_SUPPORT", "op": "copy", "inputs": ["Siudi_Status"], "kind": "binary", "linear": "none",
                            "missing": "unexpected", "mapping_class": "MEUHEDET_NATIVE"})
    raw["features"].append({"name": "bad_x", "domain": "SOCIAL_SUPPORT", "op": "copy", "inputs": ["Fall_Next_180D_Ind"], "kind": "binary",
                            "linear": "none", "missing": "unexpected", "mapping_class": "MEUHEDET_NATIVE"})
    p = tmp_path / "cat.yaml"
    p.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8", newline="\n")
    with pytest.raises(ConfigError) as e:
        load_catalogue(p, loaded["contract"], loaded["dictionary"], loaded["mapping"])
    assert "quarantined" in str(e.value) and "LABEL" in str(e.value)


# ============================================================================ elastic net
def _lin_data(n: int = 3000, p: int = 8, seed: int = 1) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    X[:, 1] = X[:, 0] + 0.05 * rng.normal(size=n)
    lp = -2.5 + 0.9 * X[:, 0] - 0.6 * X[:, 2]
    y = (rng.random(n) < 1 / (1 + np.exp(-lp))).astype(float)
    return pd.DataFrame(X, columns=[f"x{i}" for i in range(p)]), y


def test_enet_with_l1_ratio_one_is_the_lasso_path() -> None:
    from falls_ml.models.lasso_cv import lambda_grid, lambda_max, lasso_logistic_path, standardize
    from falls_ml.phase2.enet import enet_logistic_path

    X, y = _lin_data()
    Z, _, _ = standardize(X.to_numpy())
    g = lambda_grid(lambda_max(Z, y), 30, 1e-3)
    a, b = lasso_logistic_path(Z, y, g), enet_logistic_path(Z, y, g, 1.0)
    assert np.array_equal(a.beta, b.beta) and np.array_equal(a.intercept, b.intercept)


def test_enet_cv_is_deterministic_and_fixed_refit_reproduces() -> None:
    from falls_ml.phase2.enet import ElasticNetLogisticCV

    X, y = _lin_data()
    m1 = ElasticNetLogisticCV(l1_ratios=(0.5, 0.9), cv_folds=4, n_lambda=30, random_state=3, lambda_min_ratio=1e-3).fit(X, y)
    m2 = ElasticNetLogisticCV(l1_ratios=(0.5, 0.9), cv_folds=4, n_lambda=30, random_state=3, lambda_min_ratio=1e-3).fit(X, y)
    assert np.array_equal(m1.coef_, m2.coef_) and m1.l1_ratio_ in (0.5, 0.9)
    m3 = ElasticNetLogisticCV(n_lambda=30, fixed=(m1.l1_ratio_, m1.lambda_), lambda_min_ratio=1e-3).fit(X, y)
    assert np.allclose(m3.coef_, m1.coef_, atol=1e-6)
    assert m1.coef_[0] > 0 > m1.coef_[2]


# ============================================================================ screening / redundancy
def test_rank_spearman_matches_scipy_on_complete_data() -> None:
    from scipy.stats import spearmanr

    from falls_ml.phase2.screen import rank_spearman

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(500, 4)), columns=list("abcd"))
    X["b"] = X["a"] + 0.3 * rng.normal(size=500)
    c, n = rank_spearman(X)
    assert np.isclose(c.loc["a", "b"], spearmanr(X["a"], X["b"]).statistic) and int(n.loc["a", "b"]) == 500
    X.loc[:99, "c"] = np.nan
    c2, n2 = rank_spearman(X)
    assert int(n2.loc["a", "c"]) == 400 and np.isfinite(c2.loc["a", "c"])


def test_redundancy_clusters_are_anchored_not_chained() -> None:
    from falls_ml.phase2.screen import redundancy_clusters

    names = ["a", "b", "c"]
    corr = pd.DataFrame([[1, 0.95, 0.5], [0.95, 1, 0.95], [0.5, 0.95, 1]], index=names, columns=names, dtype=float)
    cl, rep = redundancy_clusters(corr, threshold=0.9, missing_pct={"a": 0, "b": 0, "c": 0}, order={"a": 0, "b": 1, "c": 2}, baseline=set())
    assert rep == {"a": "a", "b": "a"} and "c" not in rep       # c is |0.5| from the representative: never chained in
    cl2, rep2 = redundancy_clusters(corr, threshold=0.9, missing_pct={"a": 5, "b": 0, "c": 0}, order={"a": 0, "b": 1, "c": 2}, baseline={"c"})
    assert rep2["b"] == "c"                                          # a BASELINE_15 feature is always the representative


# ============================================================================ design encodings
def test_linear_encodings_and_duplicate_guard() -> None:
    from falls_ml.phase2.design import _encode

    x = np.array([0, 1, 2, 3, np.nan])
    cols, mat = _encode("s", {"linear": "thermometer", "kind": "ordinal", "levels_used": [1, 2, 3]}, x)
    assert cols == ["s__ge1", "s__ge2", "s__ge3"] and mat.tolist() == [[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1], [0, 0, 0]]
    cols, mat = _encode("f", {"linear": "fall_recency_bands", "kind": "days", "bands": [90, 180, 365]}, np.array([10, 100, 200, 400, np.nan]))
    assert cols == ["f__le90", "f__91_180", "f__181_365", "f__gt365"] and mat.sum(axis=1).tolist() == [1, 1, 1, 1, 0]
    cols, mat = _encode("b", {"linear": "none", "kind": "binary"}, np.array([1, 0, np.nan]))
    assert mat[:, 0].tolist() == [1, 0, 0]                          # NULL is never 1; its state comes from the covering indicator


# ============================================================================ metrics
def test_capacity_table_counts_and_identical_bootstrap() -> None:
    from falls_ml.phase2.evaluate import capacity_table, paired_bootstrap

    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 1])
    p = np.linspace(1, 0.1, 10)
    t = capacity_table(y, p, (0.1, 0.3)).set_index("capacity")
    assert t.at[0.1, "tp"] == 1 and t.at[0.3, "tp"] == 2 and t.at[0.3, "fp"] == 1 and t.at[0.3, "fn"] == 1 and t.at[0.3, "tn"] == 6
    b = paired_bootstrap(y, p, p, n_boot=50, seed=1, principal=0.3)
    assert b["delta_ap"] == 0 and b["delta_ap_ci_low"] == 0 and b["delta_ap_ci_high"] == 0


# ============================================================================ Optuna: deterministic after a crash
@pytest.mark.skipif(not xgb_available(), reason="xgboost runtime not available on this machine")
def test_study_resumes_with_identical_suggestions(tmp_path: Path) -> None:
    pytest.importorskip("optuna")
    from falls_ml.phase2.xgb_tuning import TrialStore, run_study

    space = {"max_depth": {"type": "int", "low": 1, "high": 3}, "min_child_weight": {"type": "float", "low": 5.0, "high": 30.0, "log": True}}

    def objective(params: dict[str, Any], seed: int, tmp: Path) -> dict[str, Any]:
        return {"value": (params["max_depth"] - 2) ** 2 + np.log(params["min_child_weight"]) / 10, "ap": 0.1, "n_trees": 10, "best_iterations": [9]}

    ra = _run(tmp_path / "a")
    full = run_study(ra.stage("08", "xgb"), TrialStore(ra), name="S", space=space, n_trials=6, n_startup=2, seed_base=5, objective=objective, meta={})
    rb = _run(tmp_path / "b")
    part = run_study(rb.stage("08", "xgb"), TrialStore(rb), name="S", space=space, n_trials=3, n_startup=2, seed_base=5, objective=objective, meta={})
    rb2 = _run(tmp_path / "b", attempt=2)                             # "restart": fresh storage, rebuilt from the committed trials
    rest = run_study(rb2.stage("08", "xgb"), TrialStore(rb2), name="S", space=space, n_trials=6, n_startup=2, seed_base=5, objective=objective, meta={})
    assert [r["params"] for r in full] == [r["params"] for r in rest] and len(part) == 3
    assert (rb2.checkpoints / "_previous").is_dir()
    led, _ = D.read_jsonl(rb2.checkpoints / "xgb_trials.jsonl")
    assert sorted(r["trial_index"] for r in led) == list(range(6))


# ============================================================================ share suppression
def test_share_suppression_hides_rederivable_small_cells() -> None:
    from falls_ml.phase2.stages_report import _suppress

    cap = pd.DataFrame({"capacity": [0.01, 0.1], "n_flagged": [30, 300], "tp": [9, 60], "fp": [21, 240], "fn": [181, 130], "tn": [2693, 2474],
                        "sensitivity": [0.047, 0.3], "ppv": [0.3, 0.2], "false_alerts": [21, 240]})
    out = _suppress(cap, "confusion", 10)
    assert out.loc[0, ["tp", "fp", "fn", "tn"]].tolist() == ["<10"] * 4 and out.loc[0, "ppv"] == "suppressed" and out.loc[0, "false_alerts"] == "suppressed"
    assert out.loc[1, "tp"] == 60 and out.loc[0, "n_flagged"] == 30            # n_flagged is not outcome-derived
    cal = pd.DataFrame({"n": [100, 100], "events": [3, 40], "observed_rate": [0.03, 0.4], "ci_low": [0.0, 0.3], "ci_high": [0.06, 0.5]})
    oc = _suppress(cal, "calibration", 10)
    assert oc.loc[0, "events"] == "<10" and oc.loc[0, "observed_rate"] == "suppressed" and oc.loc[1, "observed_rate"] == 0.4


def test_durable_write_replaces_a_read_only_uncommitted_output(tmp_path: Path) -> None:
    p = D.write_str(tmp_path / "o.csv", "a")
    D.mark_readonly(p)
    D.write_str(p, "b")                                   # interrupted finalisation left it read-only; the rewrite must succeed
    assert p.read_text(encoding="utf-8") == "b"
