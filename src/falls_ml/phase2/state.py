"""Run state, item/stage commit protocol, event log and gates (planning/ARCHITECTURE_PHASE2.md §5 as amended by reviews/REVIEW_DECISIONS.md).

Item protocol (G-01): an item is computed in ``items/.tmp-<item>-a<attempt>/``; every file is written durably; the directory is renamed to
``items/<item>/``; LAST, the commit record ``items/<item>.COMPLETE.json`` (sha256 of every file, the small JSON result, seed, code and plan
hashes, elapsed time) is written atomically. An item is complete iff its commit record parses and every listed file verifies. Anything else
(temp directories, directories without a record, torn records) is moved - never read, never deleted - to ``_incomplete/attempt-<n>/``.
A record whose files no longer match their hashes is a HARD stop (a completed artifact changed).

Stage protocol: when all items of a stage are complete, the stage writes its outputs durably and then ``STAGE_<nn>_COMPLETE.json`` with the
output hashes and the hashes of the upstream stage records (a hash chain).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from falls_ml.artifacts import utc_now
from falls_ml.errors import FallsMLError
from falls_ml.phase2 import durable as D

STOP_HARD = "HARD"
STOP_INVESTIGATION = "INVESTIGATION"
MAX_ITEM_ID = 90


class Phase2Stop(FallsMLError):
    """The run stops. HARD: the cause must be fixed. INVESTIGATION: continue only with ``--accept-gate <gate> --reason ...`` (recorded)."""

    def __init__(self, gate: str, message: str, details: list[str] | None = None, *, kind: str = STOP_HARD, evidence: dict[str, Any] | None = None):
        self.gate, self.kind, self.details, self.evidence = gate, kind, list(details or []), dict(evidence or {})
        super().__init__(f"{kind} STOP [{gate}]: {message}" + "".join(f"\n  - {d}" for d in self.details))
        self.message = message


CRASH_ENV = "FALLS_ML_PHASE2_CRASH_AT"


def crash_point(stage: str, item: str, point: str) -> None:
    """Rehearsal instrumentation ONLY: when FALLS_ML_PHASE2_CRASH_AT="<stage>|<item>|<point>" matches, the process is killed on the spot
    (os._exit - no cleanup, like a power loss). Points: item_start, before_rename, before_record. Inert unless the variable is set."""
    want = os.environ.get(CRASH_ENV)
    if want and want == f"{stage}|{item}|{point}":
        os._exit(137)


def item_seed(plan_seed: int, stage: str, item: str) -> int:
    """Seed of one item: a function of the plan seed, the stage and the item only (independent of order and interruptions)."""
    return int(hashlib.sha256(f"{plan_seed}|{stage}|{item}".encode()).hexdigest()[:8], 16)


def safe_item_id(item: str) -> str:
    """File-system safe, bounded item id (Windows MAX_PATH)."""
    s = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in item)
    if len(s) > MAX_ITEM_ID:
        s = s[: MAX_ITEM_ID - 13] + "__" + hashlib.sha256(item.encode()).hexdigest()[:11]
    return s


class EventLog:
    """Append-only ``logs/events.jsonl`` (fsync per event)."""

    def __init__(self, path: Path, attempt: int):
        self.path, self.attempt = path, attempt

    def __call__(self, stage: str, action: str, status: str, *, item: str | None = None, duration_s: float | None = None,
                 metrics: dict[str, Any] | None = None, error: str | None = None, **extra: Any) -> None:
        rec = {"ts": utc_now(), "attempt": self.attempt, "stage": stage, "item": item, "action": action, "status": status,
               "duration_s": None if duration_s is None else round(float(duration_s), 3), "metrics": metrics or {}, "error": error, **extra}
        D.append_jsonl(self.path, rec)


@dataclass
class Run:
    out: Path
    plan: dict[str, Any]
    plan_sha: str
    code_sha: str
    attempt: int
    seed: int
    events: EventLog
    accepted_gates: dict[str, str] = field(default_factory=dict)
    readonly: bool = True
    state: dict[str, Any] = field(default_factory=dict)
    _stages: dict[str, Stage] = field(default_factory=dict)

    # ---------------------------------------------------------------- layout
    @property
    def stages_dir(self) -> Path:
        return self.out / "stages"

    @property
    def work(self) -> Path:
        return self.out / "work"

    @property
    def artifacts(self) -> Path:
        return self.out / "artifacts"

    @property
    def checkpoints(self) -> Path:
        return self.out / "checkpoints"

    def stage(self, sid: str, name: str, upstream: tuple[str, ...] = ()) -> Stage:
        if sid not in self._stages:
            self._stages[sid] = Stage(self, sid, name, upstream)
        return self._stages[sid]

    def item_seed(self, stage: str, item: str) -> int:
        return item_seed(self.seed, stage, item)

    # ---------------------------------------------------------------- RUN_STATE.json (a progress summary; the records are the truth)
    def checkpoint(self, **kw: Any) -> None:
        self.state.update(kw)
        self.state["last_checkpoint"] = utc_now()
        D.write_json(self.out / "RUN_STATE.json", self.state)

    # ---------------------------------------------------------------- gates
    def gate(self, gate: str, triggered: bool, message: str, details: list[str] | None = None, evidence: dict[str, Any] | None = None) -> None:
        """An investigation stop: pause unless the user accepted this gate for this run (``--accept-gate``); acceptance is recorded."""
        if not triggered:
            return
        if gate in self.accepted_gates:
            self.events("gates", "gate_accepted", "CONTINUE", gate=gate, reason=self.accepted_gates[gate], message=message, details=details or [])
            D.append_jsonl(self.out / "logs" / "gates.jsonl", {"ts": utc_now(), "attempt": self.attempt, "gate": gate, "decision": "ACCEPTED",
                                                              "reason": self.accepted_gates[gate], "message": message, "details": details or []})
            return
        D.append_jsonl(self.out / "logs" / "gates.jsonl", {"ts": utc_now(), "attempt": self.attempt, "gate": gate, "decision": "STOPPED",
                                                          "message": message, "details": details or []})
        raise Phase2Stop(gate, message, details, kind=STOP_INVESTIGATION, evidence=evidence)


class Stage:
    def __init__(self, run: Run, sid: str, name: str, upstream: tuple[str, ...]):
        self.run, self.sid, self.name, self.upstream = run, sid, name, upstream
        self.dir = run.stages_dir / f"S{sid}_{name}"
        self.items_dir = self.dir / "items"
        self.marker_path = run.out / f"STAGE_{sid}_COMPLETE.json"
        self._verified: dict[str, dict[str, Any]] = {}
        self._recovered = False

    @property
    def label(self) -> str:
        return f"S{self.sid}_{self.name}"

    # ---------------------------------------------------------------- recovery (on first touch in an attempt)
    def recover(self) -> None:
        """Verify every committed item once; move incomplete work aside (G-01, G-04a, G-09-1)."""
        if self._recovered:
            return
        self._recovered = True
        if not self.items_dir.exists():
            return
        aside = self.dir / "_incomplete" / f"attempt-{self.run.attempt}"
        entries = sorted(self.items_dir.iterdir(), key=lambda p: p.name)
        records = {p.name[: -len(".COMPLETE.json")]: p for p in entries if p.is_file() and p.name.endswith(".COMPLETE.json")}
        for p in entries:
            if p.is_file() and p.name.endswith(".COMPLETE.json"):
                continue
            name = p.name
            committed = p.is_dir() and not name.startswith(".tmp-") and name in records
            if committed:
                continue
            aside.mkdir(parents=True, exist_ok=True)
            D.rename_dir(p, aside / name) if p.is_dir() else os.replace(p, aside / name)
            self.run.events(self.label, "incomplete_moved_aside", "RECOVERED", item=name, path=str((aside / name).relative_to(self.run.out)))
        for item, rec_path in records.items():
            rec = self._read_record(rec_path)
            if rec is None:   # torn / unparseable record -> the item is incomplete (recomputed)
                aside.mkdir(parents=True, exist_ok=True)
                os.replace(rec_path, aside / rec_path.name)
                if (self.items_dir / item).exists():
                    D.rename_dir(self.items_dir / item, aside / item)
                self.run.events(self.label, "torn_record_moved_aside", "RECOVERED", item=item)
                continue
            self._verify(item, rec)
            self._verified[item] = rec

    @staticmethod
    def _read_record(path: Path) -> dict[str, Any] | None:
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            return None
        return rec if isinstance(rec, dict) and "files" in rec and "result" in rec else None

    def _verify(self, item: str, rec: dict[str, Any]) -> None:
        d = self.items_dir / item
        problems = []
        for rel, digest in rec["files"].items():
            p = d / rel
            if not p.is_file():
                problems.append(f"{rel}: missing")
            elif D.sha256_file(p) != digest:
                problems.append(f"{rel}: content changed")
        if problems:
            raise Phase2Stop("COMPLETED_ARTIFACT_CHANGED", f"committed item {self.label}/{item} no longer matches its commit record", problems)

    # ---------------------------------------------------------------- items
    def done(self, item: str) -> bool:
        self.recover()
        return safe_item_id(item) in self._verified

    def result(self, item: str) -> dict[str, Any]:
        self.recover()
        return self._verified[safe_item_id(item)]["result"]

    def path(self, item: str) -> Path:
        return self.items_dir / safe_item_id(item)

    def item(self, item: str, fn: Callable[[Path, int], dict[str, Any]]) -> dict[str, Any]:
        """Run ``fn(tmp_dir, seed) -> small JSON result`` unless the item is committed; commit it; return its result."""
        self.recover()
        iid = safe_item_id(item)
        if iid in self._verified:
            return self._verified[iid]["result"]
        seed = self.run.item_seed(self.label, item)
        tmp = self.items_dir / f".tmp-{iid}-a{self.run.attempt}"
        tmp.mkdir(parents=True, exist_ok=False)
        self.run.checkpoint(stage=self.label, substage=item.split("__")[0], current_item=item, status="RUNNING")
        t0 = time.perf_counter()
        crash_point(self.label, item, "item_start")
        try:
            result = fn(tmp, seed)
        except Exception as exc:
            self.run.events(self.label, "item", "FAILED", item=item, duration_s=time.perf_counter() - t0, error=f"{type(exc).__name__}: {exc}"[:2000])
            raise
        elapsed = time.perf_counter() - t0
        files = {str(p.relative_to(tmp)).replace("\\", "/"): D.sha256_file(p) for p in sorted(tmp.rglob("*")) if p.is_file()}
        final = self.items_dir / iid
        crash_point(self.label, item, "before_rename")
        D.rename_dir(tmp, final)
        crash_point(self.label, item, "before_record")
        rec = {"stage": self.label, "item": item, "item_id": iid, "files": files, "result": result, "seed": seed, "code_sha256": self.run.code_sha,
               "plan_sha256": self.run.plan_sha, "attempt": self.run.attempt, "elapsed_s": round(elapsed, 3), "finished_at": utc_now()}
        D.write_json(self.items_dir / f"{iid}.COMPLETE.json", rec)
        if self.run.readonly:
            for rel in files:
                D.mark_readonly(final / rel)
        self._verified[iid] = rec
        small = {k: v for k, v in result.items() if isinstance(v, (int, float, str, bool)) or v is None} if isinstance(result, dict) else {}
        self.run.events(self.label, "item", "DONE", item=item, duration_s=elapsed, metrics=dict(list(small.items())[:12]))
        self.run.checkpoint(stage=self.label, substage=None, current_item=None, status="RUNNING")
        return result

    def records(self) -> dict[str, dict[str, Any]]:
        self.recover()
        return dict(self._verified)

    # ---------------------------------------------------------------- stage completion
    def complete_record(self) -> dict[str, Any] | None:
        """The verified stage record, or None if the stage is not complete. A record whose outputs changed is a HARD stop."""
        if not self.marker_path.exists():
            return None
        rec = self._read_record_stage(self.marker_path)
        if rec is None:
            aside = self.dir / "_incomplete" / f"attempt-{self.run.attempt}"
            aside.mkdir(parents=True, exist_ok=True)
            os.replace(self.marker_path, aside / self.marker_path.name)
            self.run.events(self.label, "torn_stage_record_moved_aside", "RECOVERED")
            return None
        problems = []
        for rel, digest in rec["outputs"].items():
            p = self.run.out / rel
            if not p.is_file():
                problems.append(f"{rel}: missing")
            elif D.sha256_file(p) != digest:
                problems.append(f"{rel}: content changed")
        for sid, digest in rec.get("upstream", {}).items():
            up = self.run.out / f"STAGE_{sid}_COMPLETE.json"
            if not up.is_file() or D.sha256_file(up) != digest:
                problems.append(f"upstream STAGE_{sid}_COMPLETE.json changed or missing")
        if problems:
            raise Phase2Stop("COMPLETED_ARTIFACT_CHANGED", f"stage {self.label} no longer matches its completion record", problems)
        return rec

    @staticmethod
    def _read_record_stage(path: Path) -> dict[str, Any] | None:
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            return None
        return rec if isinstance(rec, dict) and "outputs" in rec else None

    def finalize(self, outputs: list[Path], summary: dict[str, Any] | None = None) -> dict[str, Any]:
        outs = {}
        for p in outputs:
            rel = str(Path(p).relative_to(self.run.out)).replace("\\", "/")
            outs[rel] = D.sha256_file(Path(p))
            if self.run.readonly:
                D.mark_readonly(Path(p))
        upstream = {}
        for sid in self.upstream:
            up = self.run.out / f"STAGE_{sid}_COMPLETE.json"
            if not up.is_file():
                raise Phase2Stop("STAGE_ORDER", f"{self.label} finalised before upstream stage {sid}")
            upstream[sid] = D.sha256_file(up)
        items = self.records()
        rec = {"stage": self.label, "sid": self.sid, "outputs": outs, "upstream": upstream, "n_items": len(items),
               "items_elapsed_s": round(sum(float(r.get("elapsed_s") or 0) for r in items.values()), 3),
               "item_code_sha256": sorted({str(r.get("code_sha256")) for r in items.values()}), "code_sha256": self.run.code_sha,
               "plan_sha256": self.run.plan_sha, "attempt": self.run.attempt, "finished_at": utc_now(), "summary": summary or {}}
        D.write_json(self.marker_path, rec)
        self.run.events(self.label, "stage", "COMPLETE", duration_s=rec["items_elapsed_s"], metrics={k: v for k, v in (summary or {}).items()
                                                                                                       if isinstance(v, (int, float, str, bool))})
        done = list(self.run.state.get("completed_stages", []))
        if self.label not in done:
            done.append(self.label)
        self.run.checkpoint(stage=self.label, completed_stages=done, current_item=None)
        return rec


def write_timings(out: Path) -> None:
    """RUN_TIMINGS.csv from the item commit records: time per stage and attempt (which stages consumed the time)."""
    rows = []
    for rec_path in sorted((out / "stages").glob("*/items/*.COMPLETE.json")):
        try:
            r = json.loads(rec_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        rows.append({"stage": r.get("stage"), "attempt": r.get("attempt"), "elapsed_s": float(r.get("elapsed_s") or 0.0)})
    if not rows:
        return
    t = pd.DataFrame(rows).groupby(["stage", "attempt"], as_index=False).agg(n_items=("elapsed_s", "size"), total_s=("elapsed_s", "sum"),
                                                                             max_item_s=("elapsed_s", "max"))
    t["total_min"] = (t["total_s"] / 60).round(2)
    D.write_csv(out / "RUN_TIMINGS.csv", t.round(3))
