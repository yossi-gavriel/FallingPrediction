"""Synthetic Optuna studies: connections must close before an immediate restart.

Retain real DBAPI connections (and exception tracebacks) so garbage collection
cannot hide the lifecycle bug. These tests do not import or fit XGBoost.
"""
from __future__ import annotations

import gc
import sqlite3
from pathlib import Path

import pytest

from falls_ml.phase2 import durable as D
from falls_ml.phase2.state import EventLog, Run
from falls_ml.phase2.xgb_tuning import TrialStore, run_study


def _run(root: Path, attempt: int = 1) -> Run:
    root.mkdir(parents=True, exist_ok=True)
    return Run(out=root, plan={}, plan_sha="p" * 64, code_sha="c" * 64,
               attempt=attempt, seed=7, events=EventLog(root / "logs/events.jsonl", attempt),
               accepted_gates={}, readonly=False)


SPACE = {"depth": {"type": "int", "low": 1, "high": 4},
         "weight": {"type": "float", "low": 1.0, "high": 30.0, "log": True}}


def _objective(params: dict, seed: int, tmp: Path) -> dict:
    return {"value": (params["depth"] - 2) ** 2 + params["weight"] / 100,
            "objective_seed": seed}


def _study(run: Run, count: int, objective=_objective, name: str = "S", store=None) -> list[dict]:
    return run_study(run.stage("08", "xgb"), store or TrialStore(run), name=name,
                     space=SPACE, n_trials=count, n_startup=2, seed_base=5,
                     objective=objective, meta={})


@pytest.fixture(params=[False, True], ids=["rollback-journal", "wal"])
def connections(request, tmp_path):
    pytest.importorskip("optuna")
    from sqlalchemy import event
    from sqlalchemy.engine import Engine

    opened = []

    def retain(connection, record):
        databases = connection.execute("PRAGMA database_list").fetchall()
        if any(str(tmp_path) in row[2] for row in databases):
            opened.append(connection)
            if request.param:
                connection.execute("PRAGMA journal_mode=WAL").fetchone()

    event.listen(Engine, "connect", retain)
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield opened
    finally:
        event.remove(Engine, "connect", retain)
        for connection in opened:
            connection.close()
        if enabled:
            gc.enable()


def _assert_closed(connections):
    assert connections, "the test must observe real SQLite connections"
    for connection in connections:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connection.execute("SELECT 1")


def test_study_releases_connections_before_return(tmp_path, connections):
    run = _run(tmp_path / "run")
    assert len(_study(run, 2)) == 2
    _assert_closed(connections)
    database = run.checkpoints / "xgb_optuna.sqlite"
    archived = database.with_suffix(".archived")
    database.replace(archived)  # immediate, including on Windows; no retry or GC
    archived.replace(database)
    assert len(_study(run, 3, name="other")) == 3
    _assert_closed(connections)


def test_objective_failure_releases_connections_before_resume(tmp_path, connections):
    run = _run(tmp_path / "run")

    def interrupted(params, seed, tmp):
        raise RuntimeError("synthetic interruption")

    with pytest.raises(RuntimeError, match="synthetic interruption") as error:
        _study(run, 2, objective=interrupted)
    assert error.traceback  # keep the study/trial stack alive while checking closure
    _assert_closed(connections)
    restarted = _run(tmp_path / "run", attempt=2)
    assert len(_study(restarted, 2)) == 2
    assert (restarted.checkpoints / "_previous/xgb_optuna.sqlite.attempt-2").is_file()
    _assert_closed(connections)


def test_storage_initialization_failure_releases_connections(tmp_path, connections, monkeypatch):
    from optuna.storages._rdb.storage import _VersionManager

    def incompatible(self):
        raise RuntimeError("synthetic incompatible storage schema")

    run = _run(tmp_path / "run")
    with monkeypatch.context() as patch:
        patch.setattr(_VersionManager, "check_table_schema_compatibility", incompatible)
        with pytest.raises(RuntimeError, match="synthetic incompatible storage schema") as error:
            _study(run, 2)
    assert error.traceback
    _assert_closed(connections)
    restarted = _run(tmp_path / "run", attempt=2)
    assert len(_study(restarted, 2)) == 2
    _assert_closed(connections)


@pytest.mark.parametrize("boundary", ["tell", "ledger"])
def test_post_commit_failure_closes_storage_and_reconciles(tmp_path, connections, monkeypatch, boundary):
    import optuna

    full = _study(_run(tmp_path / "full"), 3)
    run = _run(tmp_path / "interrupted")
    store = TrialStore(run)

    def interrupted(*args, **kwargs):
        raise RuntimeError("synthetic post-commit interruption")

    with monkeypatch.context() as patch:
        if boundary == "tell":
            patch.setattr(optuna.study.Study, "tell", interrupted)
        else:
            patch.setattr(store, "append", interrupted)
        with pytest.raises(RuntimeError, match="synthetic post-commit interruption") as error:
            _study(run, 3, store=store)
    assert error.traceback
    assert run.stage("08", "xgb").done("S__t00")
    _assert_closed(connections)
    resumed = _run(tmp_path / "interrupted", attempt=2)
    assert _study(resumed, 3) == full
    ledger, tail = D.read_jsonl(resumed.checkpoints / "xgb_trials.jsonl")
    assert tail is None
    assert [row["trial_index"] for row in ledger] == [0, 1, 2]
    assert ledger[0]["reconciled"] is True
    _assert_closed(connections)


def test_failed_archive_does_not_mark_store_fresh(tmp_path, monkeypatch):
    run = _run(tmp_path / "run")
    store = TrialStore(run)
    store.sqlite.write_bytes(b"synthetic archive canary")

    def locked(source, destination):
        raise PermissionError("synthetic external lock")

    monkeypatch.setattr("falls_ml.phase2.xgb_tuning.os.replace", locked)
    for _ in range(2):
        with pytest.raises(PermissionError, match="synthetic external lock"):
            store.fresh()  # a failed archive must never be silently treated as fresh
    assert store.sqlite.read_bytes() == b"synthetic archive canary"


def test_repeated_restart_matches_uninterrupted_results(tmp_path, connections):
    full = _study(_run(tmp_path / "full"), 6)
    _assert_closed(connections)
    for attempt, count in enumerate([1, 3, 4, 6], start=1):
        run = _run(tmp_path / "resumed", attempt=attempt)
        resumed = _study(run, count)
        assert resumed == full[:count]  # params, suggestions, values AND item seeds
        _assert_closed(connections)
        if attempt > 1:
            assert (run.checkpoints / f"_previous/xgb_optuna.sqlite.attempt-{attempt}").is_file()
    ledger, tail = D.read_jsonl(run.checkpoints / "xgb_trials.jsonl")
    assert tail is None
    assert [row["trial_index"] for row in ledger] == list(range(6))
    assert _study(run, 6) == full  # already complete: no refit or new trials
    _assert_closed(connections)
