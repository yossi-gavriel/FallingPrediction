"""Dashboard safety through the actual CLI, using hand-built SYNTHETIC artifacts.

No solver or training fixture runs here. The model bytes deliberately cannot be
unpickled: a dashboard must aggregate stored OOF and explanation outputs only.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import re
import sqlite3
import sys
from types import ModuleType
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot(paths: list[Path]) -> dict[str, tuple[int, str]]:
    return {str(p): (p.stat().st_mtime_ns, _sha(p)) for p in paths}


def _commit(directory: Path) -> None:
    files = {p.name: _sha(p) for p in directory.iterdir() if p.is_file()}
    (directory / "COMPLETE.json").write_text(json.dumps({"files": files}), encoding="utf-8")


@pytest.fixture
def completed_dashboard(tmp_path: Path) -> dict[str, Any]:
    from falls_ml.phase5.capacity import PRIMARY_SETS
    from falls_ml.phase5.config import FAMILIES, SET_ALL

    out = tmp_path / "synthetic completed run with spaces"
    work, share = out / "work", out / "share"
    work.mkdir(parents=True)
    share.mkdir()
    rng = np.random.default_rng(39)
    n, folds = 4000, 3
    y = (np.arange(n) % 8 == 0).astype(int)
    outer = np.arange(n) % folds
    keys = [f"SYNTHETICKEY{i:08d}" for i in range(n)]
    ids = [f"SYNTHETICID{i:08d}" for i in range(n)]
    frame = pd.DataFrame({"row_key": keys, "old_feature": rng.normal(size=n), "new_feature": rng.normal(size=n)})
    frame.to_parquet(work / "ANALYSIS_FRAME.parquet", index=False)
    pd.DataFrame({"outer": outer}).to_parquet(work / "FOLDS.parquet", index=False)
    np.savez(work / "Y_FOLDS.npz", y=y, outer=outer)
    (work / "META.json").write_text("{}", encoding="utf-8")
    pd.DataFrame([
        {"feature": "old_feature", "origin": "OLD", "domain": "OLD", "in_OLD_PLUS_NEW_SAFE": True},
        {"feature": "new_feature", "origin": "NEW_V21", "domain": "NEW", "in_OLD_PLUS_NEW_SAFE": True},
    ]).to_csv(work / "REGISTRY.csv", index=False)
    src = tmp_path / "synthetic extract.csv"
    pd.DataFrame({"Customer_Full_ID": ids}).to_csv(src, index=False)
    plan = {"phase5_version": "2.2.0", "synthetic": True, "seed": 39, "n": n, "events": int(y.sum()),
            "cv": {"outer_folds": folds}, "alias": {s: s for s in PRIMARY_SETS},
            "sets": {s: ["old_feature", "new_feature"] for s in PRIMARY_SETS},
            "frame_sha256": _sha(work / "ANALYSIS_FRAME.parquet"), "folds_sha256": _sha(work / "FOLDS.parquet"),
            "input": {"name": src.name, "bytes": src.stat().st_size, "sha256": _sha(src)}}
    (work / "PLAN.json").write_text(json.dumps(plan), encoding="utf-8")
    predictions: dict[tuple[str, str], np.ndarray] = {}
    for family in FAMILIES:
        base = rng.random(n) + 0.03 * y
        for i, feature_set in enumerate(PRIMARY_SETS):
            # Deliberately shift fold 0's probability scale. A pooled ranking
            # must differ from the primary proportional within-fold selection.
            p = (base + 0.005 * i * y) / 1.5 + 0.3 * (outer == 0)
            predictions[family, feature_set] = p
            for fold in range(folds):
                directory = work / "units" / f"PRIMARY__{family}__{feature_set}__outer{fold}"
                directory.mkdir(parents=True)
                test = np.flatnonzero(outer == fold)
                np.savez(directory / "arrays.npz", test_idx=test, p_test=p[test])
                (directory / "result.json").write_text("{}", encoding="utf-8")
                (directory / "model.pkl").write_bytes(b"SYNTHETIC MODEL CANARY; NEVER LOAD OR FIT")
                _commit(directory)
        directory = work / "explain" / f"FOLDS__{family}__{SET_ALL}"
        directory.mkdir(parents=True)
        pd.DataFrame([{"outer": fold, "feature": feat, "drop_ap": 0.02 + 0.01 * i, "drop_auroc": 0.01}
                      for fold in range(folds) for i, feat in enumerate(("old_feature", "new_feature"))]).to_csv(
                          directory / "permutation_folds.csv", index=False)
        if family == "XGB":
            pd.DataFrame([{"outer": fold, "feature": feat, "mean_abs_shap": 0.02 + 0.01 * i}
                          for fold in range(folds) for i, feat in enumerate(("old_feature", "new_feature"))]).to_csv(
                              directory / "shap_folds.csv", index=False)
        _commit(directory)
        directory = work / "explain" / f"STABILITY__{family}__{SET_ALL}"
        directory.mkdir(parents=True)
        pd.DataFrame([{"replicate": rep, "feature": feat, "coefficient": 0.1 + 0.1 * i, "selected": True,
                       "rank": i + 1, "drop_ap": 0.02}
                      for rep in range(2) for i, feat in enumerate(("old_feature", "new_feature"))]).to_csv(
                          directory / "replicates.csv", index=False)
        _commit(directory)
    analysis = work / "analysis"
    analysis.mkdir()
    pd.DataFrame({"row_key": keys, "y": y, "outer_fold": outer}).to_parquet(analysis / "OOF_PREDICTIONS_LOCAL.parquet", index=False)
    # Real SQLite format, synthetic contents only, plus lifecycle sidecars.
    checkpoint = out / "checkpoints"
    checkpoint.mkdir()
    database = checkpoint / "xgb_optuna.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE synthetic_canary (value TEXT)")
        connection.execute("INSERT INTO synthetic_canary VALUES ('dashboard must never open storage')")
    for suffix in ("-wal", "-shm", "-journal"):
        Path(str(database) + suffix).write_bytes(b"SYNTHETIC SQLITE SIDECAR CANARY")
    (share / "RUN_MANIFEST.json").write_text(json.dumps({"report_kind": "FINAL", "synthetic": True}), encoding="utf-8")
    (share / "MANAGEMENT_SUMMARY_HE.md").write_text("# SYNTHETIC completed report\n\n## עמוד ראשון\nOriginal aggregate analysis.\n", encoding="utf-8")
    (share / "PRIMARY_70_SENSITIVITY_COMPARISON.csv").write_text("family,aggregate_note\nENET,original analysis\n", encoding="utf-8")
    protected = [src, *[p for p in work.rglob("*") if p.is_file()], *[p for p in checkpoint.iterdir() if p.is_file()]]
    return {"out": out, "src": src, "protected": protected, "protected_dirs": [work / "units", work / "explain", analysis, checkpoint],
            "ids": ids, "keys": keys, "y": y, "outer": outer, "predictions": predictions}


def _deny_training_and_storage(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []
    forbidden = {
        "falls_ml.phase5.engine": ("run_unit", "tune_and_fit", "_tune_linear", "_tune_xgb", "fit_linear", "fit_xgb", "load_model"),
        "falls_ml.phase5.models": ("fit_linear", "fit_xgb", "linear_path", "linear_path_task", "xgb_inner_fold", "_train"),
        "falls_ml.phase5.explain": ("run_fold_explain", "run_stability", "permutation_importance", "shap_importance", "load_model"),
        "falls_ml.phase5.runner": ("run_phase5", "_execute", "run_preflight"),
        "falls_ml.phase2.enet": ("enet_logistic_path",),
        "falls_ml.phase2.xgb_tuning": ("run_study",),
        "falls_ml.phase2.durable": ("read_pickle",),
        "optuna": ("create_study", "load_study", "get_all_study_summaries", "delete_study", "copy_study"),
        "sqlalchemy": ("create_engine",),
        "sqlalchemy.engine.create": ("create_engine",),
        "sqlite3": ("connect",),
        "sqlite3.dbapi2": ("connect",),
    }

    def deny(label: str) -> Any:
        def fail(*args: Any, **kwargs: Any) -> Any:
            calls.append(label)
            raise AssertionError(f"dashboard attempted training, model load, or tuning storage: {label}")
        return fail

    # Reporting must work without loading a native training library. Trap all
    # model construction/training/prediction entry points even if OpenMP is absent.
    xgb = ModuleType("xgboost")
    for name in ("train", "cv", "Booster", "DMatrix", "XGBClassifier", "XGBRegressor"):
        setattr(xgb, name, deny(f"xgboost.{name}"))
    monkeypatch.setitem(sys.modules, "xgboost", xgb)
    for module_name, names in forbidden.items():
        module = importlib.import_module(module_name)
        for name in names:
            monkeypatch.setattr(module, name, deny(f"{module_name}.{name}"))
    for module_name, class_name, methods in (
        ("optuna.storages", "RDBStorage", ("__init__",)),
        ("optuna.storages", "InMemoryStorage", ("__init__",)),
        ("optuna.study", "Study", ("optimize", "ask", "tell")),
        ("falls_ml.phase2.xgb_tuning", "TrialStore", ("__init__", "fresh", "append")),
        ("falls_ml.phase5.models", "FittedModel", ("predict",)),
        ("falls_ml.phase5.design", "LinearDesign", ("fit",)),
        ("falls_ml.phase5.design", "TreeDesign", ("fit",)),
    ):
        cls = getattr(importlib.import_module(module_name), class_name)
        for method in methods:
            monkeypatch.setattr(cls, method, deny(f"{module_name}.{class_name}.{method}"))
    return calls


def test_dashboard_cli_never_fits_opens_storage_or_mutates_inputs(completed_dashboard: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    from falls_ml.cli import main
    from falls_ml.phase5.capacity import allocate, n_selected

    fixture = completed_dashboard
    out, share = fixture["out"], fixture["out"] / "share"
    before = _snapshot(fixture["protected"])
    inventory = {p for directory in fixture["protected_dirs"] for p in directory.rglob("*") if p.is_file()}
    legacy = (share / "PRIMARY_70_SENSITIVITY_COMPARISON.csv").read_bytes()
    calls = _deny_training_and_storage(monkeypatch)
    args = ["meuhedet-phase5-dashboard", "--out", str(out), "--input", str(fixture["src"]), "--bootstrap", "20"]
    for _ in range(2):
        assert main(args) == 0
        assert calls == []
        assert _snapshot(fixture["protected"]) == before
        assert {p for directory in fixture["protected_dirs"] for p in directory.rglob("*") if p.is_file()} == inventory
        assert (share / "PRIMARY_70_SENSITIVITY_COMPARISON.csv").read_bytes() == legacy
    # Also compare against independent brute-force selection, not dashboard helpers.
    total = n_selected(30, len(fixture["y"]))
    fold_counts = allocate(total, np.bincount(fixture["outer"]))
    primary = pd.read_csv(share / "TOP3_CAPACITY_PRIMARY.csv")
    assert len(primary) == 9 and set(primary["method"]) == {"OUTER_FOLD_CAPACITY_PRIMARY"}
    assert set(primary["selected_total"]) == {total}
    for row in primary.itertuples():
        p = fixture["predictions"][row.family, row.feature_set]
        selected = []
        for fold, count in enumerate(fold_counts):
            indices = np.flatnonzero(fixture["outer"] == fold)
            selected.extend(indices[np.argsort(-p[indices], kind="mergesort")[:count]])
        assert int(row.tp) == int(fixture["y"][selected].sum())
    manifest = json.loads((share / "RUN_MANIFEST.json").read_text(encoding="utf-8"))["operating_capacity_dashboard"]
    assert manifest["no_model_fitted"] is True and "largest remainder" in manifest["method_primary"]
    scan = json.loads((share / "PRIVACY_SCAN.json").read_text(encoding="utf-8"))
    assert scan["passed"] and scan["path_scan"] == scan["row_level_scan"] == scan["file_type_scan"] == "passed"
    html = (share / "PHASE5_OPERATING_DASHBOARD.html").read_text(encoding="utf-8")
    assert not any(value in html for value in [*fixture["ids"], *fixture["keys"], str(out), str(fixture["src"])])
    assert not re.search(r"<script[^>]+src=|<link[^>]+href=|@import|url\(|\bfetch\(|XMLHttpRequest|WebSocket", html)
    assert 'method: "fold"' in html and 'method === "pooled"' in html
    assert set(pd.read_csv(share / "FEATURE_DRIVERS.csv")["family"]) == {"LASSO", "ENET", "XGB"}


def test_dashboard_rejects_tampered_input_without_mutation(completed_dashboard: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    from falls_ml.phase2.state import Phase2Stop
    from falls_ml.phase5.dashboard import run_dashboard

    fixture = completed_dashboard
    calls = _deny_training_and_storage(monkeypatch)
    fixture["src"].write_text("Customer_Full_ID\nSYNTHETIC_WRONG_FILE\n", encoding="utf-8")
    before = _snapshot([*fixture["protected"], *[p for p in (fixture["out"] / "share").iterdir() if p.is_file()]])
    with pytest.raises(Phase2Stop) as error:
        run_dashboard(fixture["out"], input_path=fixture["src"], n_boot=20)
    assert error.value.gate == "INPUT_MISMATCH"
    assert calls == [] and _snapshot([Path(p) for p in before]) == before
