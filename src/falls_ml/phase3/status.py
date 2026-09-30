"""``falls_ml meuhedet-phase3-status --out <run folder>``: READ-ONLY progress of a Phase 3 run. Opens files for reading only (never creates,
moves, repairs or locks anything; does not use the Stage recovery code) and prints aggregate progress only."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from falls_ml.phase2.status import _ago, _jsonl, _read_json, _ts, pid_alive

STAGES = ("S00_preflight", "S01_cohort", "S02_timing", "S03_recovery", "S04_reconstruct", "S05_feasibility", "S06_feature_sets", "S07_lasso", "S08_enet",
          "S09_xgb", "S10_select", "S11_stability", "S12_ablation", "S13_explain", "S14_validation", "S15_report", "S16_share")


def phase3_status(out_dir: str | Path, *, printer: Callable[[str], None] = print) -> int:
    out = Path(out_dir)
    p = printer
    plan = _read_json(out / "PHASE3_PLAN.json")
    if plan is None:
        p(f"No Phase 3 run in {out} (PHASE3_PLAN.json not found).")
        return 1
    now = datetime.now(timezone.utc)
    state = _read_json(out / "RUN_STATE.json") or {}
    audit = _read_json(out / "RESUME_AUDIT.json") or {"attempts": []}
    events = _jsonl(out / "logs" / "events.jsonl")
    done = [s for s in STAGES if (out / f"STAGE_{s[1:3]}_COMPLETE.json").is_file()]
    status = state.get("status", "?")
    alive = pid_alive(state.get("pid")) if status == "RUNNING" else None
    feas = _read_json(out / "artifacts" / "FEASIBILITY_DECISION.json") or {}
    planned = len(STAGES) if feas.get("decision") != "NO_GO" else 8
    p("Phase 3 run status (read-only)")
    p(f"  status            : {status}" + ("" if alive is None else f" (process {'ALIVE' if alive else 'NOT RUNNING - interrupted; use the resume command'})"))
    p(f"  attempt           : {state.get('attempt')} (resumes so far: {state.get('resume_count')})")
    p(f"  completed stages  : {len(done)} of {planned}; last completed: {done[-1] if done else 'none'}")
    p(f"  current stage     : {state.get('stage')}  item: {state.get('current_item')}")
    p(f"  run started       : {state.get('started_at')} ({_ago(_ts(state.get('started_at')), now)} ago); last checkpoint {_ago(_ts(state.get('last_checkpoint')), now)} ago")
    last = events[-1] if events else {}
    p(f"  last event        : {_ago(_ts(last.get('ts')), now)} ago - {last.get('stage')} {last.get('action')} {last.get('item') or ''} {last.get('status') or ''}")
    if feas:
        p(f"  feasibility       : {feas.get('decision')} -> Outcome {feas.get('outcome')}; GO domains: {', '.join(feas.get('go_domains') or []) or 'none'}")
    else:
        p("  feasibility       : not decided yet (S05)")
    ledger = _jsonl(out / "checkpoints" / "xgb_trials.jsonl")
    if ledger:
        k = int(((plan.get("plan", {}).get("config") or {}).get("cv") or {}).get("outer_folds", 5)) + 1
        n1 = int((((plan.get("plan", {}).get("config") or {}).get("xgb") or {}).get("stage1") or {}).get("n_trials", 0))
        p(f"  XGBoost trials    : {len(ledger)} of {n1 * k}")
    sel = _read_json(out / "SELECTION_FROZEN.json")
    if sel:
        p(f"  frozen selection  : {sel.get('selection_outcome')}; recommended {sel.get('recommended')}")
    vo = _read_json(out / "VALIDATION_OPENED.json")
    p(f"  VALIDATION opened : {'yes, at ' + str(vo.get('opened_at')) if vo else 'no (sealed)'}")
    if (out / "STOPPED.md").is_file():
        p("  HARD STOP         : STOPPED.md present - read it; fix the cause; then run the resume command")
    for g in sorted(out.glob("INVESTIGATION_*.md")):
        name = g.stem.replace("INVESTIGATION_", "")
        accepted = any(r.get("gate") == name and r.get("decision") == "ACCEPTED" for r in _jsonl(out / "logs" / "gates.jsonl"))
        p(f"  investigation     : {name} ({'accepted with a recorded reason' if accepted else 'OPEN - review ' + g.name})")
    scan = _read_json(out / "share" / "PRIVACY_SCAN.json")
    p(f"  share             : {'READY (privacy scan passed) - send back ONLY this folder' if scan and scan.get('passed') else 'not built yet'}")
    tim = []
    for s in STAGES:
        d = out / "stages" / s / "items"
        recs = [_read_json(rp) for rp in d.glob("*.COMPLETE.json")] if d.is_dir() else []
        recs = [r for r in recs if r]
        if recs:
            tim.append(f"{s} {sum(float(r.get('elapsed_s') or 0) for r in recs) / 60:.1f} min ({len(recs)} items)")
    if tim:
        p("  time per stage    : " + "; ".join(tim))
    p("  attempts          : " + "; ".join(f"#{a.get('attempt')} {a.get('status')}" for a in audit.get("attempts", [])))
    if json.dumps(state).count("COMPLETE_OUTCOME"):
        p(f"  result            : {state.get('status')}")
    return 0
