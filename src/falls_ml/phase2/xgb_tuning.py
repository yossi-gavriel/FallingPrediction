"""Persistent, deterministic Optuna studies for XGBoost (planning/ARCHITECTURE_PHASE2.md §7 as amended by REVIEW_DECISIONS G-03a / G-06).

- Every trial is a committed item (``<study>__tNN``): its commit record holds the parameters, the inner-CV result and the elapsed time.
  The records are the truth for each trial.
- ``checkpoints/xgb_trials.jsonl`` is the append-only audit ledger (one line per completed trial; reconciled with the records on resume;
  a torn tail is moved to a sidecar).
- ``checkpoints/xgb_optuna.sqlite`` is Optuna's storage for this attempt. It is a derived cache: at the first use in an attempt the previous
  file is moved to ``checkpoints/_previous/`` and every study is rebuilt from the committed trials (``add_trial``), so a trial interrupted by a
  crash leaves no trace in the sampler's history.
- The sampler is re-created before every ``ask`` with ``seed = f(study seed, trial index)``, so trial k gets the same suggestion after a
  restart as in an uninterrupted run (same completed history, same seed).
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from falls_ml.artifacts import utc_now
from falls_ml.phase2 import durable as D
from falls_ml.phase2.state import Run, Stage, safe_item_id


def _optuna() -> Any:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return optuna


def trial_seed(seed_base: int, study: str, t: int) -> int:
    return int(hashlib.sha256(f"{seed_base}|{study}|{t}".encode()).hexdigest()[:8], 16) % (2**31 - 1)


def distributions(space: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Optuna distributions of the free parameters and the fixed ones (a collapsed local range becomes a fixed value)."""
    optuna = _optuna()
    dist, fixed = {}, {}
    for k, s in space.items():
        lo, hi = s["low"], s["high"]
        if lo >= hi:
            fixed[k] = int(lo) if s["type"] == "int" else float(lo)
        elif s["type"] == "int":
            dist[k] = optuna.distributions.IntDistribution(int(lo), int(hi))
        else:
            dist[k] = optuna.distributions.FloatDistribution(float(lo), float(hi), log=bool(s.get("log", False)))
    return dist, fixed


class TrialStore:
    """The SQLite cache paths and ledger of one attempt; owns no live connections."""

    def __init__(self, run: Run):
        self.run = run
        self.dir = run.checkpoints
        self.dir.mkdir(parents=True, exist_ok=True)
        self.sqlite = self.dir / "xgb_optuna.sqlite"
        self.ledger = self.dir / "xgb_trials.jsonl"
        self._fresh = False
        self._rebuilt: set[str] = set()

    @property
    def url(self) -> str:
        return "sqlite:///" + self.sqlite.resolve().as_posix()

    def fresh(self) -> None:
        if self._fresh:
            return
        prev = self.dir / "_previous"
        for suffix in ("", "-journal", "-wal", "-shm"):
            p = Path(str(self.sqlite) + suffix)
            if p.exists():
                prev.mkdir(exist_ok=True)
                os.replace(p, prev / f"{p.name}.attempt-{self.run.attempt}")
        tail = D.repair_jsonl(self.ledger, self.dir / f"xgb_trials.jsonl.torn-attempt-{self.run.attempt}")
        if tail is not None:
            self.run.events("S08_xgb", "ledger_torn_tail_moved", "RECOVERED")
        self._fresh = True

    def ledger_keys(self) -> set[str]:
        records, _ = D.read_jsonl(self.ledger)
        return {f"{r.get('study')}|{r.get('trial_index')}" for r in records}

    def append(self, rec: dict[str, Any]) -> None:
        D.append_jsonl(self.ledger, rec)


@contextmanager
def _study_storage(url: str) -> Iterator[Any]:
    """One study invocation owns one RDB engine, including its exception path.

    String URLs passed to create/load_study create independent RDBStorage
    engines each time. Passing the same explicit storage keeps ownership here.
    Optuna accesses storage on this thread; objectives do not receive it.
    Remove its scoped session before disposing the pool so checked-out as well
    as idle DBAPI connections close before a subsequent attempt archives SQLite.
    """
    class OwnedRDBStorage(_optuna().storages.RDBStorage):
        def __init__(self) -> None:
            try:
                super().__init__(url=url)
            except BaseException:
                # Schema/version initialization can fail after opening SQLite.
                # Keep ownership even when the constructor never returns.
                self.close()
                raise

        def close(self) -> None:
            try:
                if getattr(self, "scoped_session", None) is not None:
                    self.remove_session()
            finally:
                if getattr(self, "engine", None) is not None:
                    self.engine.dispose()

    storage = OwnedRDBStorage()
    try:
        yield storage
    finally:
        storage.close()


def run_study(stage: Stage, store: TrialStore, *, name: str, space: dict[str, dict[str, Any]], n_trials: int, n_startup: int, seed_base: int,
              objective: Callable[[dict[str, Any], int, Path], dict[str, Any]], meta: dict[str, Any]) -> list[dict[str, Any]]:
    """Run (or resume) study ``name`` to ``n_trials`` completed trials; returns the trial results in trial order."""
    optuna = _optuna()
    store.fresh()
    dist, fixed = distributions(space)
    ids = [f"{name}__t{t:02d}" for t in range(n_trials)]
    done = [stage.done(i) for i in ids]
    k = 0
    while k < len(done) and done[k]:
        k += 1
    if any(done[k:]):
        raise RuntimeError(f"study {name}: committed trials are not a prefix (trial order broken)")
    results = [stage.result(i) for i in ids[:k]]
    records = stage.records()
    logged = store.ledger_keys()
    for t, r in enumerate(results):          # reconcile the ledger with the committed trials
        if f"{name}|{t}" not in logged:
            store.append({**_ledger_row(name, t, r, meta, records.get(safe_item_id(ids[t]), {})), "reconciled": True})
    if k == n_trials:
        return results
    with _study_storage(store.url) as storage:
        if name not in store._rebuilt:
            study = optuna.create_study(study_name=name, storage=storage, direction="minimize", load_if_exists=True)
            if len(study.trials) == 0:
                for r in results:
                    study.add_trial(optuna.trial.create_trial(params=r["suggested"], distributions=dist, value=float(r["value"]),
                                                              user_attrs={"trial_index": r["trial_index"]}))
            store._rebuilt.add(name)
        for t in range(k, n_trials):
            sampler = optuna.samplers.TPESampler(seed=trial_seed(seed_base, name, t), n_startup_trials=int(n_startup), multivariate=True)
            study = optuna.load_study(study_name=name, storage=storage, sampler=sampler)
            trial = study.ask(dist)
            suggested = dict(trial.params)
            params = {**fixed, **suggested}

            def fn(tmp: Path, seed: int, params: dict[str, Any] = params, suggested: dict[str, Any] = suggested, t: int = t) -> dict[str, Any]:
                out = objective(params, seed, tmp)
                return {**out, "trial_index": t, "params": params, "suggested": suggested, "study": name}

            r = stage.item(ids[t], fn)
            study.tell(trial, float(r["value"]))
            store.append(_ledger_row(name, t, r, meta, stage.records().get(safe_item_id(ids[t]), {})))
            results.append(r)
    return results


def _ledger_row(name: str, t: int, r: dict[str, Any], meta: dict[str, Any], rec: dict[str, Any]) -> dict[str, Any]:
    return {"ts": utc_now(), "study": name, "trial_index": t, "status": "COMPLETE", "params": r.get("params"), "value_inner_cv_logloss": r.get("value"),
            "inner_cv_ap": r.get("ap"), "n_trees_median": r.get("n_trees"), "best_iterations": r.get("best_iterations"), "seed": rec.get("seed"),
            "elapsed_s": rec.get("elapsed_s"), "code_sha256": rec.get("code_sha256"), "attempt": rec.get("attempt"), **meta}


def best_trial(results: list[dict[str, Any]]) -> dict[str, Any]:
    return min(results, key=lambda r: (float(r["value"]), int(r["trial_index"]), str(r.get("study", ""))))
