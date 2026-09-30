"""``falls_ml meuhedet-phase2-status --out <run folder>``: a READ-ONLY progress view of a running or stopped Phase 2 run.

It opens files for reading only, never creates, moves, repairs or locks anything (it does not use the Stage recovery code), and prints only
aggregate progress: stage, item, attempt, times, XGBoost trial counts, the latest small metrics, frozen selection / validation status, gates
and whether the process is alive. Safe to run as often as wanted while the job runs (a file being replaced is simply retried).
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STAGES = ("S00_preflight", "S01_cohort", "S02_registry", "S03_engineer", "S04_screen", "S05_feature_sets", "S06_lasso", "S07_enet", "S08_xgb",
          "S09_select", "S10_stability", "S11_ablation", "S12_explain", "S13_validation", "S14_consensus", "S15_report", "S16_share")


def _read_text(p: Path, tries: int = 5) -> str | None:
    for _ in range(tries):
        try:
            with open(p, encoding="utf-8") as fh:
                return fh.read()
        except FileNotFoundError:
            return None
        except (PermissionError, OSError):
            time.sleep(0.2)
    return None


def _read_json(p: Path) -> Any:
    t = _read_text(p)
    if t is None:
        return None
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        return None


def _jsonl(p: Path) -> list[dict[str, Any]]:
    t = _read_text(p)
    out = []
    for line in (t or "").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue          # a line being appended right now
    return out


def _ts(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        d = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _ago(d: datetime | None, now: datetime) -> str:
    if d is None:
        return "unknown"
    s = int((now - d).total_seconds())
    h, rem = divmod(max(0, s), 3600)
    m, sec = divmod(rem, 60)
    return f"{h}h {m:02d}m {sec:02d}s"


def pid_alive(pid: int | None) -> bool | None:
    """True / False, or None when it cannot be told (no pid recorded)."""
    if not pid:
        return None
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, int(pid))      # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return kernel32.GetLastError() == 5                        # access denied = exists
        code = ctypes.c_ulong()
        try:
            kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        finally:
            kernel32.CloseHandle(handle)
        return code.value == 259                                         # STILL_ACTIVE
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def phase2_status(out_dir: str | Path, *, printer: Callable[[str], None] = print) -> int:
    out = Path(out_dir)
    p = printer
    plan = _read_json(out / "PHASE2_PLAN.json")
    if plan is None:
        p(f"No Phase 2 run in {out} (PHASE2_PLAN.json not found).")
        return 1
    now = datetime.now(timezone.utc)
    state = _read_json(out / "RUN_STATE.json") or {}
    audit = _read_json(out / "RESUME_AUDIT.json") or {"attempts": []}
    events = _jsonl(out / "logs" / "events.jsonl")
    done = [s for s in STAGES if (out / f"STAGE_{s[1:3]}_COMPLETE.json").is_file()]
    status = state.get("status", "?")
    alive = pid_alive(state.get("pid")) if status == "RUNNING" else None
    p("Phase 2 run status (read-only)")
    p(f"  status            : {status}" + ("" if alive is None else f" (process {'ALIVE' if alive else 'NOT RUNNING - interrupted; use the resume command'})"))
    p(f"  attempt           : {state.get('attempt')} (resumes so far: {state.get('resume_count')})")
    p(f"  completed stages  : {len(done)} of {len(STAGES)}; last completed: {done[-1] if done else 'none'}")
    p(f"  current stage     : {state.get('stage')}  substage: {state.get('substage')}  item: {state.get('current_item')}")
    p(f"  run started       : {state.get('started_at')} ({_ago(_ts(state.get('started_at')), now)} ago)")
    p(f"  this attempt      : {state.get('attempt_started_at')} ({_ago(_ts(state.get('attempt_started_at')), now)} ago)")
    p(f"  last checkpoint   : {_ago(_ts(state.get('last_checkpoint')), now)} ago")
    last = events[-1] if events else {}
    p(f"  last event        : {_ago(_ts(last.get('ts')), now)} ago - {last.get('stage')} {last.get('action')} {last.get('item') or ''} {last.get('status') or ''}")
    # XGBoost trials
    ledger = _jsonl(out / "checkpoints" / "xgb_trials.jsonl")
    x = (plan.get("plan", {}).get("config") or {}).get("xgb") or {}
    if ledger or "S08" in str(state.get("stage")):
        per: dict[str, int] = {}
        for r in ledger:
            per[str(r.get("study"))] = per.get(str(r.get("study")), 0) + 1
        k = int(((plan.get("plan", {}).get("config") or {}).get("cv") or {}).get("outer_folds", 5)) + 1
        n1 = int((x.get("stage1") or {}).get("n_trials", 0))
        s1 = sum(v for s, v in per.items() if s.startswith("XGB1_"))
        s2 = sum(v for s, v in per.items() if s.startswith("XGB2_"))
        p(f"  XGBoost trials    : stage 1 {s1} of {n1 * k}; stage 2 {s2} (runs only if stage 1 beats the best SAFE linear model)")
        cur = str(state.get("current_item") or "")
        if cur.startswith(("XGB1_", "XGB2_")) and "__t" in cur:
            p(f"  current trial     : {cur.split('__t')[0]} trial {int(cur.split('__t')[1]) + 1}")
    # latest small metrics
    metr = [e for e in events if e.get("action") == "item" and e.get("status") == "DONE" and e.get("metrics")]
    if metr:
        e = metr[-1]
        shown = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in e["metrics"].items() if isinstance(v, (int, float, str, bool))}
        p(f"  latest item result: {e.get('stage')} {e.get('item')} {json.dumps(shown)[:300]}")
    sel = _read_json(out / "SELECTION_FROZEN.json")
    if sel:
        p(f"  frozen selection  : recommended {sel.get('recommended')}; benchmark class (TRAIN OOF) {sel.get('benchmark_class_oof')}")
    vo = _read_json(out / "VALIDATION_OPENED.json")
    p(f"  VALIDATION opened : {'yes, at ' + str(vo.get('opened_at')) if vo else 'no (sealed)'}")
    conf = _read_json(out / "artifacts" / "VALIDATION_CONFIRMATION.json")
    if conf:
        p(f"  confirmation      : {conf.get('confirmation')}; benchmark class (VALIDATION) {conf.get('benchmark_class_validation')}")
    # gates
    gates = sorted(out.glob("INVESTIGATION_*.md"))
    if (out / "STOPPED.md").is_file():
        p("  HARD STOP         : STOPPED.md present - read it; fix the cause; then run the resume command")
    for g in gates:
        name = g.stem.replace("INVESTIGATION_", "")
        accepted = any(r.get("gate") == name and r.get("decision") == "ACCEPTED" for r in _jsonl(out / "logs" / "gates.jsonl"))
        p(f"  investigation     : {name} ({'accepted with a recorded reason' if accepted else 'OPEN - review ' + g.name + ' before deciding'})")
    if not gates and not (out / "STOPPED.md").is_file():
        p("  gates             : none fired")
    scan = _read_json(out / "share" / "PRIVACY_SCAN.json")
    p(f"  share             : {'READY (privacy scan passed)' if scan and scan.get('passed') else 'not built yet'}")
    # time per stage
    tim = []
    for s in STAGES:
        total, n = 0.0, 0
        for rp in (out / "stages" / s / "items").glob("*.COMPLETE.json") if (out / "stages" / s / "items").is_dir() else []:
            r = _read_json(rp)
            if r:
                total += float(r.get("elapsed_s") or 0)
                n += 1
        if n:
            tim.append(f"{s} {total / 60:.1f} min ({n} items)")
    if tim:
        p("  time per stage    : " + "; ".join(tim))
    p(f"  attempts          : " + "; ".join(f"#{a.get('attempt')} {a.get('status')}" for a in audit.get("attempts", [])))
    return 0
