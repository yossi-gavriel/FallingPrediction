"""Phase 5.1 (Phase 5 3.0.0) repair-only correction - focused tests for every changed methodological boundary
(docs/phase6/EXPERIMENT1_IMPLEMENTATION_CONTRACT.md):

 R-1  no outcome-dependent eligibility decision anywhere                   test_r1_no_numeric_exclusion_and_preflight_reads_no_label_for_eligibility
 R-2  per-fold coverage gate is label-free                                  test_r2_per_fold_coverage_gate_reads_predictors_only
 R-3  lambda grid anchored inside each inner training fold                  test_r3_inner_grid_anchored_on_inner_training_rows_only
 R-4  CCI quarantined by default; branches only via a registered override   test_r4_cci_quarantine_default_and_override_branches
 R-5  explicit `other` level for learned-level codes                         test_r5_other_level_for_learned_codes
 R-6  forensic AUROC never changes membership                               test_r6_forensic_diagnostic_reports_and_never_excludes
 R-8/9 ENET only, FINAL for three sets, one secondary contrast               test_r8_r9_plan_units_enet_only_three_finals_single_contrast
 R-10 PRE verification a-f and failure cases                                test_r10_pre_verification_passes_and_each_failure_stops
 R-12 negative controls: frozen folds, fixed thresholds, hard stop           test_r12_negative_controls_permute_within_frozen_folds_and_gate
 R-13 version guards                                                        test_r13_v2_settings_and_v2_plans_are_refused
 N-4  inner validation labels never touch the inner design / grid           test_n4_inner_validation_labels_do_not_touch_design_or_grid
 +    real data requires --pre-run before anything is read                  test_real_data_requires_pre_run
 +    no Experiment-2 logic is reachable                                    test_no_experiment2_logic_reachable
"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tests.unit.test_phase5_contract import _prepare, _small_ctx

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "falls_ml" / "phase5"


@pytest.fixture(scope="module")
def prepared(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    from falls_ml.phase5.runner import build_sets

    r = _prepare(tmp_path_factory.mktemp("p51"), traps=("leaky_new", "weak_proxy", "strong_legit"))
    r["S"] = build_sets(r["P"].registry, r["P"].meta, r["cfg"])
    return r


# ============================================================================ R-1
def test_r1_no_numeric_exclusion_and_preflight_reads_no_label_for_eligibility(prepared: dict[str, Any]) -> None:
    import falls_ml.phase5.data as data_mod

    P, S = prepared["P"], prepared["S"]
    reg = P.registry.set_index("feature")
    assert "INELIGIBLE_LEAKAGE" not in set(reg["class"])
    assert reg["univariate_auroc"].isna().all()
    src = inspect.getsource(data_mod.prepare)
    assert "roc_auc" not in src and "_univariate_auroc" not in src          # the preflight computes no outcome-dependent statistic for eligibility
    assert not hasattr(data_mod, "_univariate_auroc")
    # the strong legitimate predictor and the weak proxy are RETAINED (admissible by provenance / timing), the composite is legacy-ALL only
    assert "new_gait_abnormality_ind" in S["sets"]["OLD_PLUS_NEW_SAFE"] and "new_osteoporosis_ind" in S["sets"]["OLD_PLUS_NEW_SAFE"]
    assert "new_deficit_count_proxy" not in S["sets"]["OLD_PLUS_NEW_SAFE"] and "new_deficit_count_proxy" in S["sets"]["OLD_PLUS_ALL_NEW_ELIGIBLE"]
    cfg = prepared["cfg"]
    assert "leakage_univariate_auroc" not in cfg["eligibility"]


# ============================================================================ R-2
def test_r2_per_fold_coverage_gate_reads_predictors_only() -> None:
    from falls_ml.phase5.engine import coverage_gate

    assert "y" not in inspect.signature(coverage_gate).parameters
    X = pd.DataFrame({"a": np.r_[np.ones(30), np.zeros(70)], "rare": np.r_[np.ones(5), np.full(95, np.nan)], "const": np.ones(100)})
    keep, dropped = coverage_gate(X, ["a", "rare", "const"], 10)
    assert keep == ["a"] and set(dropped) == {"rare", "const"}
    assert "5 known" in dropped["rare"] and "constant" in dropped["const"]


# ============================================================================ R-3 and N-4
def _toy(seed: int = 3, n: int = 400) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n), "b": (rng.random(n) < 0.3).astype(float)})
    y = (rng.random(n) < 1 / (1 + np.exp(-(0.8 * X["x1"] + 0.5 * X["b"] - 2.0)))).astype(float).to_numpy()
    meta = {"x1": {"kind": "continuous", "linear": "none"}, "x2": {"kind": "continuous", "linear": "none"}, "b": {"kind": "binary", "linear": "none"}}
    return X, y, meta


def test_r3_inner_grid_anchored_on_inner_training_rows_only() -> None:
    from falls_ml.phase5.models import linear_path, ratio_grid

    X, y, meta = _toy()
    spec = {"lambda_min_ratio": 1e-3, "tol": 1e-7, "max_iter": 10000}
    tr, va = np.arange(0, 300), np.arange(300, 400)
    r1 = linear_path(X.iloc[tr], y[tr], X.iloc[va], list(X.columns), meta, l1_ratio=0.5, n_lambda=8, lambda_min_ratio=1e-3, spec=spec)
    Xva2 = X.iloc[va].copy()
    Xva2["x1"] = Xva2["x1"] * 50 + 10                                                      # extreme validation predictors
    r2 = linear_path(X.iloc[tr], y[tr], Xva2, list(X.columns), meta, l1_ratio=0.5, n_lambda=8, lambda_min_ratio=1e-3, spec=spec)
    assert r1["lambda_max"] == r2["lambda_max"] and np.array_equal(r1["lambdas"], r2["lambdas"])      # the grid comes from the training rows only
    assert np.allclose(r1["lambdas"], r1["lambda_max"] * ratio_grid(8, 1e-3))
    r3 = linear_path(X.iloc[tr[:200]], y[tr[:200]], X.iloc[va], list(X.columns), meta, l1_ratio=0.5, n_lambda=8, lambda_min_ratio=1e-3, spec=spec)
    assert r3["lambda_max"] != r1["lambda_max"]                                             # a different training fold anchors its own lambda_max


def test_n4_inner_validation_labels_do_not_touch_design_or_grid() -> None:
    from falls_ml.phase5.models import linear_path

    X, y, meta = _toy(seed=5)
    spec = {"lambda_min_ratio": 1e-3, "tol": 1e-7, "max_iter": 10000}
    tr, va = np.arange(0, 300), np.arange(300, 400)
    assert "yva" not in inspect.signature(linear_path).parameters                          # validation labels cannot even be passed
    r1 = linear_path(X.iloc[tr], y[tr], X.iloc[va], list(X.columns), meta, l1_ratio=0.5, n_lambda=6, lambda_min_ratio=1e-3, spec=spec)
    r2 = linear_path(X.iloc[tr], y[tr], X.iloc[va], list(X.columns), meta, l1_ratio=0.5, n_lambda=6, lambda_min_ratio=1e-3, spec=spec)
    assert np.array_equal(r1["preds"], r2["preds"]) and r1["lambda_max"] == r2["lambda_max"]


def test_r3_unit_records_inner_and_outer_anchoring(prepared: dict[str, Any], tmp_path: Path) -> None:
    from falls_ml.phase5.engine import UnitSpec, run_unit

    c = _small_ctx(prepared, tmp_path / "u")
    spec = UnitSpec(uid="PRIMARY|ENET|OLD|outer0", stage="PRIMARY", family="ENET", setname="OLD", outer=0)
    r = run_unit(c, spec)
    assert "inner" in r["grid_anchoring"] and "outer" in r["grid_anchoring"]
    inner = r["lambda_max_inner"][str(r["config"]["l1_ratio"])]
    assert len(inner) == c.cfg.inner_folds and len(set(inner)) > 1                     # each inner fold anchored its own lambda_max
    assert r["config"]["lambda"] == pytest.approx(r["lambda_max_outer"] * r["config"]["lambda_ratio"])
    assert set(r["features_effective"]) <= set(c.sets["OLD"]) and r["n_features_effective"] == len(r["features_effective"])
    t = pd.read_csv(c.units_dir / spec.folder / "trials.csv")
    assert "lambda_ratio" in t.columns and "lambda" not in t.columns                   # candidates are identified by the dimensionless ratio


# ============================================================================ R-4
def test_r4_cci_quarantine_default_and_override_branches(prepared: dict[str, Any], tmp_path: Path) -> None:
    from falls_ml.errors import ConfigError
    from falls_ml.phase5.config import load_phase5_config
    from falls_ml.phase5.data import apply_catalogue_overrides

    P, S, cfg = prepared["P"], prepared["S"], prepared["cfg"]
    reg = P.registry.set_index("feature")
    assert cfg["feature_overrides"]["com_cci_group"]["branch"] == "quarantine"
    assert reg.loc["com_cci_group", "class"] == "INELIGIBLE_SEMANTICS" and "QUARANTINED" in str(reg.loc["com_cci_group", "reason"])
    assert reg.loc["com_cci_group", "override"] == "quarantine"
    assert "com_cci_group" not in {f for v in S["sets"].values() for f in v}
    assert P.meta["com_cci_group"]["kind"] == "continuous"                              # nothing inferred: the catalogue copy is untouched under quarantine
    cat = prepared["L"]["cat"]
    assert cat.get("com_cci_group").kind == "continuous"                                 # the loaded Phase 3 catalogue object is never mutated
    o = apply_catalogue_overrides(cat, {"com_cci_group": {"branch": "ordinal", "levels": [1, 2, 3, 4]}})
    f = o.get("com_cci_group")
    assert f.kind == "ordinal" and f.linear == "thermometer" and f.levels == (1.0, 2.0, 3.0, 4.0)
    o = apply_catalogue_overrides(cat, {"com_cci_group": {"branch": "nominal", "levels": [0, 1, 2]}})
    f = o.get("com_cci_group")
    assert f.kind == "categorical" and f.linear == "onehot" and f.levels == (0.0, 1.0, 2.0)
    assert cat.get("com_cci_group").kind == "continuous"
    for bad in ({"feature_overrides": {"com_cci_group": {"branch": "ordinal"}}},                     # ordinal needs the documented levels
                {"feature_overrides": {"com_cci_group": {"branch": "continuous"}}},                  # no such branch
                {"feature_overrides": {"com_cci_group": {"branch": "quarantine", "reason": ""}}}):   # quarantine needs its reason
        with pytest.raises(ConfigError):
            load_phase5_config(overrides=bad)
    # an ordinal registration flows through the preflight (meta) - order only, no spacing
    r = _prepare(tmp_path, n=2500)
    assert r["P"].meta["com_cci_group"]["kind"] == "continuous" and r["P"].registry.set_index("feature").loc["com_cci_group", "class"] == "INELIGIBLE_SEMANTICS"


# ============================================================================ R-5
def test_r5_other_level_for_learned_codes() -> None:
    from falls_ml.phase5.design import LinearDesign

    meta = {"c": {"kind": "categorical", "linear": "onehot", "missing": "no_event"}, "d": {"kind": "categorical", "linear": "onehot", "levels": [1, 2, 3], "missing": "no_event"}}
    tr = pd.DataFrame({"c": [1.0] * 15 + [2.0] * 15 + [7.0] * 3 + [np.nan] * 7, "d": [1.0] * 20 + [2.0] * 10 + [3.0] * 10})
    d = LinearDesign(["c", "d"], meta).fit(tr)
    assert d.levels_["c"] == [1.0, 2.0] and "c__other" in d.columns_ and "c__eq2" in d.columns_      # learned levels: rare code 7 -> other
    assert not any(col.startswith("d__other") for col in d.columns_)                                 # declared levels: no other column
    te = pd.DataFrame({"c": [1.0, 2.0, 7.0, 9.0, np.nan], "d": [1.0, 2.0, 3.0, 2.0, 1.0]})
    names, M = d._raw(te)
    j = names.index("c__other")
    assert list(M[:, j]) == [0.0, 0.0, 1.0, 1.0, 0.0]                                                 # rare and unseen codes -> other; missing -> 0 (+ NA)
    assert "c__na" in names and M[4, names.index("c__na")] == 1.0


# ============================================================================ R-6
def test_r6_forensic_diagnostic_reports_and_never_excludes(prepared: dict[str, Any], tmp_path: Path) -> None:
    from falls_ml.phase5.engine import UnitSpec, run_unit

    c = _small_ctx(prepared, tmp_path / "f")
    spec = UnitSpec(uid="PRIMARY|ENET|OLD_PLUS_ALL_NEW_ELIGIBLE|outer1", stage="PRIMARY", family="ENET", setname="OLD_PLUS_ALL_NEW_ELIGIBLE", outer=1)
    r = run_unit(c, spec)
    fa = r["forensic_auroc"]
    assert fa["new_deficit_count_proxy"] >= 0.80 and "new_deficit_count_proxy" in r["forensic_flagged"]
    assert "new_deficit_count_proxy" in r["features_effective"]                     # flagged, REPORTED, never excluded
    assert fa["new_gait_abnormality_ind"] >= 0.80 and "new_gait_abnormality_ind" in r["features_effective"]   # a strong legitimate predictor is retained
    assert 0.55 <= fa["new_osteoporosis_ind"] < 0.80                                  # the weak proxy sits below any numeric gate: no gate exists
    assert r["forensic_warn_at"] == 0.80


# ============================================================================ R-8 / R-9
def test_r8_r9_plan_units_enet_only_three_finals_single_contrast(prepared: dict[str, Any]) -> None:
    from falls_ml.phase5.runner import plan_units

    S, cfg = prepared["S"], prepared["cfg"]
    plan = {"cv": {"outer_folds": 3}, "families": ["ENET"], **S}
    u = plan_units(plan, cfg)
    assert {x.family for st in u for x in u[st]} == {"ENET"}
    assert len(u["PRIMARY"]) == 9 and len(u["FINAL"]) == 3 and u["DOMAIN"] == []
    assert len(u["ABLATION"]) == 3 and all(x.setname == "OLD_PLUS_NEW_SAFE__NO_NEW_REGISTRY" for x in u["ABLATION"])
    assert S["ablations"]["NO_NEW_REGISTRY"]["base_set"] == "OLD_PLUS_NEW_SAFE" and S["ablations"]["NO_NEW_REGISTRY"]["role"] == "SECONDARY_DIAGNOSTIC"
    assert set(S["ablations"]["NO_NEW_REGISTRY"]["removed"]) <= set(S["sets"]["OLD_PLUS_NEW_SAFE"])
    assert all(f.startswith("new_registry_") for f in S["ablations"]["NO_NEW_REGISTRY"]["removed"])


# ============================================================================ R-10: a synthetic Phase 5 2.2.0 PRE folder consistent with a POST context
def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def make_pre_folder(pre: Path, *, keys: np.ndarray, y: np.ndarray, outer: np.ndarray, input_sha: str, input_name: str, sets: dict[str, list[str]],
                    seed: int = 7, version: str = "2.2.0") -> dict[str, Any]:
    """A completed 2.2.0-layout PRE run made from a POST cohort: plan, folds, labels, committed ENET units with synthetic OOF risks, a FINAL
    manifest and the Top-3% primary table computed with the same capacity arithmetic (what the 0.12.3 dashboard wrote)."""
    from falls_ml.phase5.capacity import curve, rank_model
    from falls_ml.phase5.config import SET_ALL, SET_OLD, SET_SAFE

    rng = np.random.default_rng(seed)
    work, share = pre / "work", pre / "share"
    (work / "units").mkdir(parents=True)
    share.mkdir()
    n = len(y)
    K = int(outer.max()) + 1
    pd.DataFrame({"row_key": keys, "outer_fold": outer}).to_parquet(work / "FOLDS.parquet", index=False)
    np.savez(work / "Y_FOLDS.npz", y=y, outer=outer)
    pd.DataFrame({"row_key": keys}).to_parquet(work / "ANALYSIS_FRAME.parquet", index=False)
    alias = {SET_OLD: SET_OLD, SET_ALL: SET_ALL, SET_SAFE: SET_SAFE}
    preds = {}
    for i, s in enumerate((SET_OLD, SET_ALL, SET_SAFE)):
        p = 1 / (1 + np.exp(-(rng.normal(size=n) + (1.2 + 0.3 * i) * y - 2.5)))
        preds[s] = p
        for k in range(K):
            d = work / "units" / f"PRIMARY__ENET__{s}__outer{k}"
            d.mkdir()
            te = np.flatnonzero(outer == k)
            tr = np.flatnonzero(outer != k)
            np.savez(d / "arrays.npz", train_idx=tr, test_idx=te, inner_oof=p[tr], p_test=p[te])
            thr = float(np.quantile(p[tr], 0.6))
            (d / "result.json").write_text(json.dumps({"uid": f"PRIMARY|ENET|{s}|outer{k}", "stage": "PRIMARY", "family": "ENET", "set": s, "outer": k,
                                                       "config": {"lambda": 0.01, "l1_ratio": 0.5, "lambda_index": 20}, "thresholds": {"0.70": thr, "0.50": thr * 1.5},
                                                       "n_test": int(len(te)), "seconds": 1.0}), encoding="utf-8")
            (d / "model.pkl").write_bytes(b"SYNTHETIC PRE MODEL CANARY - NEVER LOADED")
            files = {q.name: _sha(q) for q in d.iterdir() if q.is_file()}
            (d / "COMPLETE.json").write_text(json.dumps({"files": files}), encoding="utf-8")
    plan = {"phase5_version": version, "synthetic": True, "seed": seed, "n": n, "events": int(y.sum()), "cv": {"outer_folds": K, "inner_folds": 3},
            "alias": alias, "sets": sets, "kinds": {s: "PRIMARY" for s in alias}, "frame_sha256": _sha(work / "ANALYSIS_FRAME.parquet"),
            "folds_sha256": _sha(work / "FOLDS.parquet"), "input": {"name": input_name, "sha256": input_sha, "bytes": 0}, "code_sha256": "pre-code",
            "config_sha256": "pre-config", "falls_ml_version": "0.12.3", "mode": "quick"}
    (work / "PLAN.json").write_text(json.dumps(plan), encoding="utf-8")
    (pre / "RUN_STATUS.json").write_text(json.dumps({"status": "COMPLETE"}), encoding="utf-8")
    (share / "RUN_MANIFEST.json").write_text(json.dumps({"report_kind": "FINAL", "phase5_version": version}), encoding="utf-8")
    ranked = {("ENET", s): rank_model("ENET", s, y.astype(np.int64), outer, p) for s, p in preds.items()}
    t3 = curve(y.astype(np.int64), ranked, [30]).drop(columns=["cells_safe"])
    t3.to_csv(share / "TOP3_CAPACITY_PRIMARY.csv", index=False)
    rows = [{"feature": f, "class": "SAFE_VERIFIED", "in_OLD": f in sets[SET_OLD], "in_OLD_PLUS_ALL_NEW_ELIGIBLE": f in sets[SET_ALL],
             "in_OLD_PLUS_NEW_SAFE": f in sets[SET_SAFE]} for f in sets[SET_ALL]]
    rows = [r for r in rows if r["feature"] != "new_deficit_count_proxy"]                     # one row per feature, as a real 2.2.0 table has
    rows.append({"feature": "new_deficit_count_proxy", "class": "INELIGIBLE_LEAKAGE", "in_OLD": False, "in_OLD_PLUS_ALL_NEW_ELIGIBLE": False, "in_OLD_PLUS_NEW_SAFE": False})
    pd.DataFrame(rows).to_csv(share / "FEATURE_ELIGIBILITY.csv", index=False)
    return {"plan": plan, "preds": preds, "top3": t3}


@pytest.fixture(scope="module")
def pre_world(prepared: dict[str, Any], tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    from falls_ml.phase5.engine import outer_folds

    P, S = prepared["P"], prepared["S"]
    order = np.argsort(P.frame["row_key"].to_numpy(), kind="mergesort")
    keys = P.frame["row_key"].to_numpy()[order]
    y = P.y[order].astype(int)
    outer = outer_folds(y, 3, 12345)                                                   # NOT the seed the POST would use: adoption must be visible
    sets = {k: v for k, v in S["sets"].items() if k in ("OLD", "OLD_PLUS_ALL_NEW_ELIGIBLE", "OLD_PLUS_NEW_SAFE")}
    base = tmp_path_factory.mktemp("pre")
    pre = base / "pre run 2.2.0"
    info = make_pre_folder(pre, keys=keys, y=y, outer=outer, input_sha=_sha(prepared["csv"]), input_name=prepared["csv"].name, sets=sets)
    return {"pre": pre, "keys": keys, "y": y, "outer": outer, "base": base, **info}


def test_r10_pre_verification_passes_and_each_failure_stops(pre_world: dict[str, Any], prepared: dict[str, Any]) -> None:
    import shutil

    from falls_ml.phase2.state import Phase2Stop
    from falls_ml.phase5.prerun import check_unmodified, pre_digest, verify_pre_run

    pre, keys, y, outer = pre_world["pre"], pre_world["keys"], pre_world["y"], pre_world["outer"]
    sha = _sha(prepared["csv"])
    V = verify_pre_run(pre, post_input_sha256=sha, post_row_keys=keys, post_y=y, log=lambda m: None)
    assert V["pre"]["verified"] and np.array_equal(V["outer"], outer) and len(V["pre"]["top3_reproduced"]) == 3
    assert all(r["reproduced"] for r in V["pre"]["top3_reproduced"]) and V["pre"]["folder_name"] == pre.name
    assert "digest_sha256" in V["pre"] and pre_digest(pre) == V["pre"]["digest_sha256"]
    # the POST cohort order may differ: the adopted folds follow the row keys, not the file order
    perm = np.random.default_rng(1).permutation(len(keys))
    V2 = verify_pre_run(pre, post_input_sha256=sha, post_row_keys=keys[perm], post_y=y[perm], log=lambda m: None)
    assert np.array_equal(V2["outer"], outer[perm])

    def failing(mutate: Any, gate: str = "PRE_VERIFICATION_FAILED") -> str:
        copy = pre_world["base"] / f"pre copy {np.random.default_rng().integers(1e9)}"
        shutil.copytree(pre, copy)
        mutate(copy)
        with pytest.raises(Phase2Stop) as e:
            verify_pre_run(copy, post_input_sha256=sha, post_row_keys=keys, post_y=y, log=lambda m: None)
        assert e.value.gate == gate
        return e.value.message

    def rewrite_plan(copy: Path, **kw: Any) -> None:
        pp = copy / "work" / "PLAN.json"
        plan = json.loads(pp.read_text(encoding="utf-8"))
        plan.update(kw)
        pp.write_text(json.dumps(plan), encoding="utf-8")

    assert "input" in failing(lambda c: rewrite_plan(c, input={"name": "x", "sha256": "0" * 64, "bytes": 1}))                    # (b) different extract
    assert "Phase 5" in failing(lambda c: rewrite_plan(c, phase5_version="3.0.0"))                                                # (a) a 3.x folder is no PRE
    assert "not complete" in failing(lambda c: (c / "RUN_STATUS.json").write_text(json.dumps({"status": "INTERRUPTED"}), encoding="utf-8"))
    assert "fold hash" in failing(lambda c: rewrite_plan(c, folds_sha256="f" * 64))                                               # (d) fold file vs plan hash
    assert "COMPLETE.json" in failing(lambda c: (c / "work" / "units" / "PRIMARY__ENET__OLD__outer1" / "COMPLETE.json").unlink())  # (e) a missing unit
    with pytest.raises(Phase2Stop) as e:                                                                                          # (c) labels differ
        verify_pre_run(pre, post_input_sha256=sha, post_row_keys=keys, post_y=1 - y, log=lambda m: None)
    assert "labels" in e.value.message
    with pytest.raises(Phase2Stop) as e:                                                                                          # (c) cohort differs
        verify_pre_run(pre, post_input_sha256=sha, post_row_keys=keys[:-5], post_y=y[:-5], log=lambda m: None)
    assert "COHORT_MISMATCH" in e.value.message

    def tamper_top3(c: Path) -> None:
        t = pd.read_csv(c / "share" / "TOP3_CAPACITY_PRIMARY.csv")
        t.loc[t["feature_set"] == "OLD", "tp"] = t.loc[t["feature_set"] == "OLD", "tp"] + 1
        t.to_csv(c / "share" / "TOP3_CAPACITY_PRIMARY.csv", index=False)

    assert "NOT reproduced" in failing(tamper_top3)                                                                                # (f) the table must reproduce
    assert "TOP3_CAPACITY_PRIMARY" in failing(lambda c: (c / "share" / "TOP3_CAPACITY_PRIMARY.csv").unlink())                     # (f) the table must exist

    def tamper_unit(c: Path) -> None:                                                                                             # (f) via the arrays, not the csv
        d = c / "work" / "units" / "PRIMARY__ENET__OLD_PLUS_NEW_SAFE__outer0"
        a = np.load(d / "arrays.npz")
        p = np.full(len(a["p_test"]), 0.5)                     # constant risks: the top-k becomes the first k rows (ties) - a different selection
        np.savez(d / "arrays.npz", train_idx=a["train_idx"], test_idx=a["test_idx"], inner_oof=a["inner_oof"], p_test=p)
        files = {q.name: _sha(q) for q in d.iterdir() if q.is_file() and q.name != "COMPLETE.json"}
        (d / "COMPLETE.json").write_text(json.dumps({"files": files}), encoding="utf-8")

    failing(tamper_unit)
    # the digest re-check: any later change of the PRE folder stops the report
    digest = pre_digest(pre)
    copy = pre_world["base"] / "pre modified later"
    shutil.copytree(pre, copy)
    assert pre_digest(copy) == digest
    (copy / "share" / "TOP3_CAPACITY_PRIMARY.csv").write_text("family\n", encoding="utf-8")
    with pytest.raises(Phase2Stop) as e:
        check_unmodified(copy, digest)
    assert e.value.gate == "PRE_RUN_MODIFIED"


def test_r10_preflight_adopts_pre_folds_and_refuses_the_out_folder(pre_world: dict[str, Any], prepared: dict[str, Any], tmp_path: Path) -> None:
    from falls_ml.phase2.state import Phase2Stop
    from falls_ml.phase5.runner import run_phase5

    ov = {"eligibility": {"min_known_observed_rows": 20}, "cv": {"outer_folds": 3, "inner_folds": 3}}
    out = tmp_path / "post"
    r = run_phase5(prepared["csv"], out, mode="quick", preflight_only=True, overrides=ov, synthetic=True, pre_run=pre_world["pre"], jobs=1)
    assert r["status"] == "PREFLIGHT_COMPLETE"
    plan = json.loads((out / "work" / "PLAN.json").read_text(encoding="utf-8"))
    assert plan["folds_source"].startswith("PRE") and plan["pre_run"]["verified"] and plan["phase5_major"] == "3"
    folds = pd.read_parquet(out / "work" / "FOLDS.parquet")
    assert np.array_equal(folds["outer_fold"].to_numpy(), pre_world["outer"]) and list(folds["row_key"]) == list(pre_world["keys"])
    assert plan["pre_run"]["folds_sha256"] == pre_world["plan"]["folds_sha256"]
    md = (out / "preflight" / "PHASE5_PREFLIGHT.md").read_text(encoding="utf-8")
    assert "P10" in md and "ADOPTED" in md and md.rstrip().endswith("SAFE TO MODEL")
    assert str(pre_world["pre"]) not in json.dumps(plan)                                 # the folder NAME is recorded, never a local path
    with pytest.raises(Phase2Stop) as e:
        run_phase5(prepared["csv"], out, mode="quick", preflight_only=True, overrides=ov, synthetic=True, pre_run=out, jobs=1)
    assert e.value.gate == "PRE_RUN_IS_OUT"
    # a mismatching PRE at resume time (different folder name than the verified one) is refused
    other = tmp_path / "other pre"
    other.mkdir()
    with pytest.raises(Phase2Stop) as e:
        run_phase5(prepared["csv"], out, mode="quick", preflight_only=True, overrides=ov, synthetic=True, pre_run=other, jobs=1)
    assert e.value.gate == "PLAN_MISMATCH"


def test_real_data_requires_pre_run(prepared: dict[str, Any], tmp_path: Path) -> None:
    from falls_ml.phase2.state import Phase2Stop
    from falls_ml.phase4 import sealed
    from falls_ml.phase5.runner import run_phase5

    out = tmp_path / "real"
    r = run_phase5(prepared["csv"], out, mode="quick", preflight_only=True, synthetic=False, jobs=1,
                   overrides={"eligibility": {"min_known_observed_rows": 20}})
    assert r["status"] == "STOPPED_PREFLIGHT" and r["exit_code"] == 2
    md = (out / "preflight" / "PHASE5_PREFLIGHT.md").read_text(encoding="utf-8")
    assert "PRE_RUN_REQUIRED" in md and md.rstrip().endswith("STOP - REVIEW REQUIRED")
    assert not (out / "work" / "PLAN.json").exists() and not (out / "work" / "units").exists()   # no folds, no plan, nothing fitted
    with pytest.raises(Phase2Stop) as e:                                                            # and the modelling command refuses too
        run_phase5(prepared["csv"], out, mode="quick", synthetic=False, jobs=1, overrides={"eligibility": {"min_known_observed_rows": 20}})
    assert e.value.gate == "PREFLIGHT_REQUIRED"
    assert sealed.READ_LOG is not None


# ============================================================================ R-12
def test_r12_negative_controls_permute_within_frozen_folds_and_gate(prepared: dict[str, Any], tmp_path: Path) -> None:
    from falls_ml.errors import ConfigError
    from falls_ml.phase2.state import Phase2Stop
    from falls_ml.phase5.config import NEGATIVE_CONTROL_LIMITS, load_phase5_config
    from falls_ml.phase5.controls import permute_within_folds, require_passed

    y = np.r_[np.ones(20), np.zeros(80), np.ones(10), np.zeros(90)].astype(int)
    outer = np.r_[np.zeros(100), np.ones(100)].astype(int)
    yp = permute_within_folds(y, outer, 3)
    assert yp[:100].sum() == 20 and yp[100:].sum() == 10 and not np.array_equal(yp, y)     # fold prevalences preserved, labels shuffled within folds
    assert np.array_equal(permute_within_folds(y, outer, 3), yp)                            # deterministic per seed
    assert NEGATIVE_CONTROL_LIMITS == {"max_mean_auroc": 0.55, "max_mean_recall_top3": 0.05}
    for bad in ({"negative_controls": {"max_mean_auroc": 0.60}}, {"negative_controls": {"max_mean_recall_top3": 0.10}}, {"negative_controls": {"seeds": 0}}):
        with pytest.raises(ConfigError):
            load_phase5_config(overrides=bad)
    out = tmp_path / "nc"
    (out / "work").mkdir(parents=True)
    with pytest.raises(Phase2Stop) as e:                                                   # real data: no controls -> nothing is fitted or reported
        require_passed(out, synthetic=False, n_required=10)
    assert e.value.gate == "NEGATIVE_CONTROLS_REQUIRED"
    assert require_passed(out, synthetic=True, n_required=10)["passed"] is None            # a synthetic run may skip them
    (out / "work" / "NEGATIVE_CONTROLS_RESULT.json").write_text(json.dumps({"passed": False, "n_seeds": 10, "mean_auroc": 0.7, "mean_recall_top3": 0.1}), encoding="utf-8")
    with pytest.raises(Phase2Stop) as e:
        require_passed(out, synthetic=False, n_required=10)
    assert e.value.gate == "NEGATIVE_CONTROL_FAILED"
    (out / "work" / "NEGATIVE_CONTROLS_RESULT.json").write_text(json.dumps({"passed": True, "n_seeds": 3, "mean_auroc": 0.5, "mean_recall_top3": 0.03}), encoding="utf-8")
    with pytest.raises(Phase2Stop) as e:                                                   # fewer seeds than registered: not enough on real data
        require_passed(out, synthetic=False, n_required=10)
    assert e.value.gate == "NEGATIVE_CONTROLS_REQUIRED"


# ============================================================================ R-13
def test_r13_v2_settings_and_v2_plans_are_refused(tmp_path: Path) -> None:
    from falls_ml.errors import ConfigError
    from falls_ml.phase2.state import Phase2Stop
    from falls_ml.phase5.config import load_phase5_config
    from falls_ml.phase5.runner import Monitor, _verify_plan

    with pytest.raises(ConfigError):
        load_phase5_config(overrides={"version": 2})
    cfg = load_phase5_config()
    with pytest.raises(Phase2Stop) as e:
        _verify_plan({"phase5_version": "2.2.0", "config_sha256": cfg.sha256}, cfg, None, None, Monitor(tmp_path, 30.0), report_only=False)
    assert e.value.gate == "PHASE5_VERSION_MISMATCH"


# ============================================================================ no Experiment-2 logic reachable
def test_no_experiment2_logic_reachable() -> None:
    from falls_ml.phase5.config import load_phase5_config

    import ast

    names: set[str] = set()
    for p in sorted(SRC.glob("*.py")):                                                       # identifiers only (docstrings may NAME what is excluded)
        for node in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.arg):
                names.add(node.arg)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
    blob = "\n".join(sorted(names)).lower()
    for token in ("one_se", "shortlist", "logloss_top3", "objective_logloss", "near_tie", "policy_b", "recall_top3_selection", "additive_basis", "spline",
                  "catboost", "lightgbm", "gower", "kmedoids", "pam_cluster", "select_by_top3", "top3_objective"):
        assert token not in blob, token
    cfg = load_phase5_config()
    assert cfg["families"] == ["ENET"]
    assert cfg.budget["enet"] == {"n_lambda": 40, "l1_ratios": [0.1, 0.25, 0.5, 0.75, 0.9]}          # the 2.2.0 candidate space, unchanged
    assert 1.0 not in cfg.budget["enet"]["l1_ratios"]
    assert cfg["domain_families"] == [] and list(cfg["ablations"]["blocks"]) == ["NO_NEW_REGISTRY"]
    from falls_ml.phase5.engine import _tune_linear
    from falls_ml.phase5.thresholds import objective

    s = inspect.getsource(_tune_linear)
    assert "objective(" in s and s.count("fit_linear(") == 1                                  # one selection policy, one refit per unit
    assert "flagged_share" in inspect.getsource(objective)                                   # the 2.2.0 70%-sensitivity objective
