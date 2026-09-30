"""Fast checks of the final Phase 2 readiness changes (reviews/ASTRA_FINAL_RESOLUTION.md): the frozen configuration is current and enforced,
the three categories never mix (SAFE_DISCOVERY = SAFE only), VALIDATION_OPENED.json is write-once, the status command is read-only, the CLI
refuses gate / code-change acceptance outside a resume, and shared capacity tables suppress small cells."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def loaded() -> dict[str, Any]:
    from falls_ml.d00.dependency import load_d00_config
    from falls_ml.data.meuhedet_wide import load_wide_contract, load_wide_mapping
    from falls_ml.eda.dictionary import load_data_dictionary
    from falls_ml.phase2.config import load_catalogue, load_phase2_config

    cfg = load_phase2_config()
    mapping = load_wide_mapping()
    contract = load_wide_contract(mapping.contract_path)
    dictionary = load_data_dictionary("configs/meuhedet/wide_v1_data_dictionary.yaml", contract)
    d00 = load_d00_config(cfg["d00_config"], contract=contract, dictionary=dictionary)
    cat = load_catalogue(cfg["features"], contract, dictionary, mapping)
    return {"cfg": cfg, "mapping": mapping, "contract": contract, "dictionary": dictionary, "d00": d00, "cat": cat}


def test_frozen_final_config_is_current_and_hashed(loaded: dict[str, Any]) -> None:
    from falls_ml.phase2.final_config import FROZEN_PATH, build_final_config, sha256_of, sha_line, to_bytes

    fc = build_final_config(loaded["cfg"], loaded["cat"], contract=loaded["contract"], dictionary=loaded["dictionary"], mapping=loaded["mapping"],
                            d00=loaded["d00"])
    # falls_ml 0.9.0 (Phase 3) carries the Phase 2 freeze of the 0.8.1 package unchanged: every scientific value must still match it; only the
    # package version differs (Phase 2 runs only from its own 0.8.1 installation - see test_phase2_production_run_is_refused_in_this_package)
    assert fc["falls_ml_version"] != "0.8.1"
    fc["falls_ml_version"] = "0.8.1"
    frozen = (ROOT / FROZEN_PATH).read_bytes()
    assert frozen == to_bytes(fc), "configs/meuhedet/FINAL_EXPERIMENT_CONFIG.json is stale: run tools/freeze_phase2_config.py after a reviewed change"
    assert (ROOT / FROZEN_PATH).with_suffix(".sha256").read_text(encoding="utf-8") == sha_line(sha256_of(fc))
    x = fc["model_families"]["XGBOOST"]["fixed_parameters"]
    for k in ("objective", "eval_metric", "tree_method", "learning_rate", "subsample", "colsample_bytree", "gamma", "reg_alpha", "max_delta_step",
              "grow_policy", "scale_pos_weight", "max_bin"):
        assert k in x, k
    assert fc["categories"]["SAFE_DISCOVERY"]["allowed_status"] == ["SAFE"]
    assert "UNSAFE" not in fc["categories"]["EXPLORATORY_UNRESOLVED_SENSITIVITY"]["allowed_status"]


def test_runner_refuses_an_unfrozen_production_configuration(loaded: dict[str, Any], tmp_path: Path) -> None:
    from falls_ml.phase2.config import load_phase2_config
    from falls_ml.phase2.runner import effective_final_config
    from falls_ml.phase2.state import Phase2Stop

    raw = yaml.safe_load((ROOT / "configs/meuhedet/phase2.yaml").read_text(encoding="utf-8"))
    raw["phase2"]["xgb"]["stage1"]["n_trials"] = 14
    raw["phase2"]["features"] = str(ROOT / "configs/meuhedet/phase2_features.yaml")
    raw["phase2"]["d00_config"] = str(ROOT / "configs/meuhedet/d00_sensitivity.yaml")
    p = tmp_path / "changed.yaml"
    p.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8")
    cfg = load_phase2_config(p)
    kw = {k: loaded[k] for k in ("contract", "dictionary", "mapping", "d00")}
    with pytest.raises(Phase2Stop) as e:
        effective_final_config(cfg, loaded["cat"], allow_unfrozen=False, **kw)
    assert e.value.gate == "FROZEN_CONFIG_MISMATCH"
    _, _, frozen = effective_final_config(cfg, loaded["cat"], allow_unfrozen=True, **kw)
    assert frozen is False


def test_phase2_production_run_is_refused_in_this_package(loaded: dict[str, Any]) -> None:
    """Isolation (Phase 3 package, falls_ml 0.9.0): the reviewed Phase 2 freeze belongs to falls_ml 0.8.1, so this package can never start or
    resume a production Phase 2 run - the running experiment stays on its own installation."""
    from falls_ml.phase2.runner import effective_final_config
    from falls_ml.phase2.state import Phase2Stop

    kw = {k: loaded[k] for k in ("contract", "dictionary", "mapping", "d00")}
    with pytest.raises(Phase2Stop) as e:
        effective_final_config(loaded["cfg"], loaded["cat"], allow_unfrozen=False, **kw)
    assert e.value.gate == "FROZEN_CONFIG_MISMATCH"


def test_config_loader_guards_the_category_invariants(tmp_path: Path) -> None:
    from falls_ml.errors import ConfigError
    from falls_ml.phase2.config import load_phase2_config

    base = yaml.safe_load((ROOT / "configs/meuhedet/phase2.yaml").read_text(encoding="utf-8"))
    for mutate in (lambda r: r["eligibility"]["categories"].update({"SAFE_DISCOVERY": ["SAFE", "UNRESOLVED"]}),
                   lambda r: r["eligibility"]["categories"].update({"EXPLORATORY_UNRESOLVED_SENSITIVITY": ["SAFE", "UNSAFE"]}),
                   lambda r: r["xgb"]["fixed"].update({"scale_pos_weight": 45.0}),
                   lambda r: r["selection"].update({"candidates": "EXPLORATORY_UNRESOLVED_SENSITIVITY"})):
        raw = json.loads(json.dumps(base))
        mutate(raw["phase2"])
        p = tmp_path / "bad.yaml"
        p.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8")
        with pytest.raises(ConfigError):
            load_phase2_config(p)


def _registry(cat: Any, status: dict[str, str]) -> pd.DataFrame:
    elig = {"SAFE": "ELIGIBLE", "UNRESOLVED": "ELIGIBLE_EXPLORATORY", "UNSAFE": "INELIGIBLE_UNSAFE"}
    return pd.DataFrame([{"feature": f.name, "eligibility": elig[status.get(f.name, "SAFE")]} for f in cat.features])


def test_feature_sets_never_mix_categories(loaded: dict[str, Any]) -> None:
    from falls_ml.phase2.featuresets import (ALL_SAFE, EXPL_ALL, EXPLORATORY, HISTORICAL, SAFE_BASE, SAFE_DISCOVERY, build_feature_sets,
                                             check_category_invariants)

    cat = loaded["cat"]
    status = {f.name: ("UNRESOLVED" if f.domain == "MEDICATION" else "UNSAFE" if f.domain == "HOME_ENVIRONMENT" else "SAFE") for f in cat.features}
    er = _registry(cat, status)
    baseline = ["age_years", "sex", "falls", "dementia", "polypharmacy_count_120d"]
    bstat = {"age_years": "SAFE", "sex": "SAFE", "falls": "SAFE", "dementia": "UNRESOLVED", "polypharmacy_count_120d": "UNRESOLVED"}
    design = {"features": {}, "na_groups": {}}
    fs = build_feature_sets(cat, er, {SAFE_DISCOVERY: {}, EXPLORATORY: {}}, baseline_features=baseline, baseline_status=bstat, design=design)
    assert check_category_invariants(fs, er, bstat, baseline) == []
    assert fs.sets["BASELINE_15"]["category"] == HISTORICAL and fs.sets[SAFE_BASE]["baseline"] == ["age_years", "sex", "falls"]
    for name, s in fs.sets.items():
        assert not {f for f in s["features"] if status[f] == "UNSAFE"}, name
        if s["category"] == SAFE_DISCOVERY:
            assert all(status[f] == "SAFE" for f in s["features"]) and all(bstat[b] == "SAFE" for b in s["baseline"]), name
    assert any(status[f] == "UNRESOLVED" for f in fs.sets[EXPL_ALL]["features"])
    assert "SAFE_BASE_PLUS_MEDICATION" not in fs.sets and ALL_SAFE not in fs.aliases and SAFE_BASE not in fs.aliases
    step = next(w for w in fs.design["waterfall"] if w["added"] == "MEDICATION")
    assert step["n_safe_features_added"] == 0                        # a domain without SAFE features adds nothing (reported as such)
    # a violation is caught
    fs.sets[ALL_SAFE]["features"].append(next(f for f, v in status.items() if v == "UNRESOLVED"))
    assert check_category_invariants(fs, er, bstat, baseline)


class _Run:
    def __init__(self, out: Path):
        from falls_ml.phase2.state import EventLog

        self.out, self.attempt, self.plan_sha, self.code_sha = out, 1, "p" * 64, "c" * 64
        self.plan = {"final_experiment_config_sha256": "f" * 64}
        self.events = EventLog(out / "logs" / "events.jsonl", 1)


class _Ctx:
    def __init__(self, out: Path):
        self.run = _Run(out)

    def fs(self) -> Any:
        return type("FS", (), {"sha256": "s" * 64})()


def test_validation_opened_is_write_once(tmp_path: Path) -> None:
    from falls_ml.phase2.stages_models import open_validation
    from falls_ml.phase2.state import Phase2Stop

    ctx = _Ctx(tmp_path)
    sel = {"sha256": "a" * 64, "shortlist": ["LASSO:SAFE_BASE", "LASSO:BASELINE_15", "LASSO:ALL_REVIEWED_SAFE"], "recommended": "LASSO:ALL_REVIEWED_SAFE",
           "descriptive_configs": ["LASSO:SAFE_BASE", "LASSO:BASELINE_15", "LASSO:ALL_REVIEWED_SAFE"]}
    first = open_validation(ctx, sel)                     # type: ignore[arg-type]
    before = (tmp_path / "VALIDATION_OPENED.json").read_bytes()
    again = open_validation(ctx, sel)                     # type: ignore[arg-type]
    assert again["opened_at"] == first["opened_at"] and (tmp_path / "VALIDATION_OPENED.json").read_bytes() == before
    with pytest.raises(Phase2Stop) as e:
        open_validation(ctx, {**sel, "sha256": "b" * 64})    # type: ignore[arg-type]
    assert e.value.gate == "VALIDATION_REOPEN_REFUSED"
    assert {f["role"] for f in first["confirmatory_finalists"]} == {"discovery_reference", "benchmark", "recommended"}


def test_status_on_a_partial_run_is_read_only(tmp_path: Path) -> None:
    from falls_ml.phase2.status import phase2_status

    out = tmp_path / "run"
    (out / "logs").mkdir(parents=True)
    (out / "checkpoints").mkdir()
    (out / "PHASE2_PLAN.json").write_text(json.dumps({"plan": {"config": {"cv": {"outer_folds": 5}, "xgb": {"stage1": {"n_trials": 15}}}}}), encoding="utf-8")
    (out / "RUN_STATE.json").write_text(json.dumps({"status": "RUNNING", "stage": "S08_xgb", "current_item": "XGB1_outer2__t06", "attempt": 2,
                                                    "pid": 999999999, "started_at": "2026-09-29T00:00:00+00:00"}), encoding="utf-8")
    (out / "STAGE_00_COMPLETE.json").write_text("{}", encoding="utf-8")
    (out / "logs" / "events.jsonl").write_text('{"ts": "2026-09-29T00:10:00+00:00", "stage": "S08_xgb", "action": "item", "status": "DONE", '
                                               '"item": "XGB1_outer2__t05", "metrics": {"value": 0.08}}\n{"torn', encoding="utf-8")
    (out / "checkpoints" / "xgb_trials.jsonl").write_text("".join(json.dumps({"study": "XGB1_outer0", "trial_index": i}) + "\n" for i in range(15)),
                                                          encoding="utf-8")
    (out / "INVESTIGATION_IMPLAUSIBLE_GAIN_OOF.md").write_text("# x", encoding="utf-8")
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in out.rglob("*") if p.is_file()}
    lines: list[str] = []
    assert phase2_status(out, printer=lines.append) == 0
    text = "\n".join(lines)
    assert "NOT RUNNING" in text and "XGB1_outer2 trial 7" in text and "stage 1 15 of 90" in text and "IMPLAUSIBLE_GAIN_OOF (OPEN" in text
    after = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in out.rglob("*") if p.is_file()}
    assert before == after and not (out / "logs" / "events.jsonl.torn").exists()


def test_cli_refuses_acceptance_outside_a_resume(tmp_path: Path) -> None:
    base = [sys.executable, "-m", "falls_ml", "meuhedet-phase2", "--input", str(tmp_path / "x.csv"), "--reference", str(tmp_path), "--out", str(tmp_path / "o")]
    env = {"PYTHONPATH": str(ROOT / "src")}
    for extra in (["--accept-gate", "IMPLAUSIBLE_GAIN_OOF", "--reason", "r"], ["--accept-code-change", "r"], ["--preflight", "--resume"]):
        r = subprocess.run([*base, *extra], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
                           env={**__import__("os").environ, **env})
        assert r.returncode == 2, (extra, r.stderr)
    assert not (tmp_path / "o").exists()


def test_share_capacity_suppression_hides_small_counts() -> None:
    from falls_ml.phase2.stages_report import _suppress

    df = pd.DataFrame([{"model": "LASSO:BASELINE_15", "val_n": 1000, "val_events": 40, "val_flagged_top1": 10, "val_falls_identified_top1": 4,
                        "val_capture_top1": 0.1, "val_ppv_top1": 0.4, "val_false_alerts_top1": 6, "val_additional_falls_vs_benchmark_top1": 0,
                        "val_flagged_top10": 100, "val_falls_identified_top10": 20, "val_capture_top10": 0.5, "val_ppv_top10": 0.2,
                        "val_false_alerts_top10": 80, "val_additional_falls_vs_benchmark_top10": 0},
                       {"model": "LASSO:ALL_REVIEWED_SAFE", "val_n": 1000, "val_events": 40, "val_flagged_top1": 10, "val_falls_identified_top1": 10,
                        "val_capture_top1": 0.25, "val_ppv_top1": 1.0, "val_false_alerts_top1": 0, "val_additional_falls_vs_benchmark_top1": 6,
                        "val_flagged_top10": 100, "val_falls_identified_top10": 25, "val_capture_top10": 0.625, "val_ppv_top10": 0.25,
                        "val_false_alerts_top10": 75, "val_additional_falls_vs_benchmark_top10": 5}])
    out = _suppress(df, "capacity_counts", 10)
    assert out.at[0, "val_falls_identified_top1"] == "suppressed" and out.at[1, "val_additional_falls_vs_benchmark_top1"] == "suppressed"
    assert out.at[1, "val_falls_identified_top10"] == 25 and out.at[1, "val_additional_falls_vs_benchmark_top10"] == 5


def test_benchmark_class_rule() -> None:
    from falls_ml.phase2.stages_models import benchmark_class

    kw = {"max_ap_loss": 0.010, "max_cap_loss_pp": 1.0}
    assert benchmark_class(0.02, 2.0, superior=True, **kw) == "MEETS_SUPERIORITY_RULE"
    assert benchmark_class(-0.005, -0.5, superior=False, **kw) == "WITHIN_NONINFERIORITY_MARGINS"
    assert benchmark_class(-0.02, 0.0, superior=False, **kw) == "OUTSIDE_MARGINS"
    assert benchmark_class(np.nan, 0.0, superior=False, **kw) == "OUTSIDE_MARGINS"


def test_xgb_native_early_stopping_rules() -> None:
    """Astra F-01: num_boost_round = the ceiling; the inner held-out fold is scored with the trees up to the best iteration (the native API
    returns the LAST model); the final fit uses the fixed tree count without early stopping; the stage-2 depth range never contains 0."""
    xgb = pytest.importorskip("xgboost")
    from falls_ml.phase2.fitting import _dmatrix, stratified_folds, xgb_fit, xgb_inner_cv, xgb_params

    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.normal(size=(1500, 4)), columns=list("abcd"))
    y = (rng.random(1500) < 1 / (1 + np.exp(-(-2.5 + X["a"].to_numpy())))).astype(float)
    cfg = yaml.safe_load((ROOT / "configs/meuhedet/phase2.yaml").read_text(encoding="utf-8"))["phase2"]
    params = xgb_params(cfg, {"max_depth": 2, "min_child_weight": 5.0, "reg_lambda": 5.0}, seed=1, nthread=1)
    cv = xgb_inner_cv(X, y, params, n_folds=3, seed=9, n_max=400, early_stopping=15)
    folds = stratified_folds(y, 3, 9, "xgb_inner_folds")
    te = folds == 0
    bst = xgb.train(params, _dmatrix(X.loc[~te], y[~te]), num_boost_round=400, evals=[(_dmatrix(X.loc[te], y[te]), "h")], early_stopping_rounds=15,
                    verbose_eval=False)
    assert bst.num_boosted_rounds() > bst.best_iteration + 1          # the returned model holds trees after the best iteration
    p_best = bst.predict(_dmatrix(X.loc[te]), iteration_range=(0, bst.best_iteration + 1))
    ll = float(-np.mean(y[te] * np.log(p_best) + (1 - y[te]) * np.log(1 - p_best)))
    assert cv["folds"][0]["best_iteration"] == bst.best_iteration and abs(cv["folds"][0]["logloss"] - ll) < 1e-9
    final = xgb_fit(X, y, params, cv["n_trees"])
    assert final.num_boosted_rounds() == cv["n_trees"]
    from falls_ml.phase2.stages_models import _local_space

    from types import SimpleNamespace

    ctx = SimpleNamespace(cfg={"xgb": cfg["xgb"]})
    for best_depth in (1, 2, 3):
        sp = _local_space(ctx, {"params": {"max_depth": best_depth, "min_child_weight": 10.0, "reg_lambda": 5.0}})   # type: ignore[arg-type]
        assert sp["max_depth"]["low"] >= 1 and sp["max_depth"]["high"] <= 3


def test_death_label_audit() -> None:
    from falls_ml.phase2.cohort import death_label_audit

    idx = pd.Timestamp("2025-01-01")
    n = 400
    df = pd.DataFrame({"Index_Date": [idx] * n, "Fall_Next_180D_Ind": [0] * n, "Is_Deceased_Ind": [0] * n, "Death_Censor_Date": [pd.NaT] * n,
                       "Label_Reason_180D": ["NEGATIVE_FULL_FOLLOWUP"] * n})
    df.loc[:59, "Is_Deceased_Ind"] = 1
    df.loc[:59, "Death_Censor_Date"] = idx + pd.Timedelta(days=90)
    a = death_label_audit(df, "Fall_Next_180D_Ind", "Index_Date")
    assert a["deaths_within_window"] == 60 and a["overwrite_suspected"] is True       # deaths never carry an event -> suspicious
    df.loc[:14, "Fall_Next_180D_Ind"] = 1
    b = death_label_audit(df, "Fall_Next_180D_Ind", "Index_Date")
    assert b["overwrite_suspected"] is False and b["events_among_deaths_within_window"] == 15 and b["events_kept_when_death_follows"] is True
    df.loc[:59, "Is_Deceased_Ind"] = 0
    c = death_label_audit(df, "Fall_Next_180D_Ind", "Index_Date")
    assert c["deaths_within_window"] == 0 and c["overwrite_suspected"] is False


def test_config_hashes_ignore_crlf_line_endings(tmp_path: Path) -> None:
    """A CRLF copy of an unchanged settings / catalogue file (e.g. re-saved on Windows) keeps its hash, so the frozen check still matches."""
    from falls_ml.phase2.config import _sha

    for rel in ("configs/meuhedet/phase2.yaml", "configs/meuhedet/phase2_features.yaml"):
        src = ROOT / rel
        crlf = tmp_path / src.name
        crlf.write_bytes(src.read_bytes().replace(b"\n", b"\r\n"))
        assert _sha(crlf) == _sha(src)
