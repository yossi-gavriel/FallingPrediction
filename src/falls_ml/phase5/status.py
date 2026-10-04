"""``meuhedet-phase5 --status``: read-only progress of a Phase 5 folder (never writes, never starts work)."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def phase5_status(out: str | Path) -> dict[str, Any]:
    out = Path(out)
    st_path = out / "RUN_STATUS.json"
    if not st_path.is_file():
        return {"status": "NOT_STARTED", "text": f"no Phase 5 run in this folder (RUN_STATUS.json missing)"}
    st = json.loads(st_path.read_text(encoding="utf-8"))
    plan_p = out / "work" / "PLAN.json"
    plan = json.loads(plan_p.read_text(encoding="utf-8")) if plan_p.is_file() else {}
    hb = st.get("heartbeat")
    age = None
    if hb:
        try:
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(hb)).total_seconds()
        except ValueError:
            age = None
    alive = st.get("status") == "RUNNING" and age is not None and age < 3 * 60
    lines = [f"Phase 5 status: {st.get('status')}" + (" (heartbeat {:.0f} s ago)".format(age) if age is not None else ""),
             f"mode {st.get('mode', plan.get('mode', '?'))}; jobs {st.get('jobs', '?')}; XGBoost device {(st.get('device') or {}).get('device', '?')}"]
    if st.get("status") == "RUNNING" and not alive:
        lines.append("WARNING: no heartbeat for more than 3 minutes - the process is probably not running any more (power cut / closed window): "
                     "restart the same command with --resume")
    if plan:
        lines.append(f"cohort: {plan.get('n')} patients, {plan.get('events')} events; folds {plan.get('cv')}")
    pr = st.get("progress") or {}
    for k in ("PRIMARY", "FINAL", "DOMAIN", "ABLATION", "EXPLAIN", "STABILITY"):
        if k in pr:
            lines.append(f"  {k:<10} {pr[k]['done']:>4} / {pr[k]['total']:<4} complete")
    if "outer_folds_complete" in pr:
        lines.append(f"  outer folds with every primary model complete: {pr['outer_folds_complete']} / {pr['outer_folds_total']}")
    cur = st.get("current") or {}
    if cur:
        lines.append(f"current: {cur.get('stage')} {cur.get('item', '')}" + (f" - trial {cur.get('trial')} / {cur.get('n_trials')}" if cur.get("trial") else ""))
    tot = float(st.get("elapsed_seconds_total") or 0)
    lines.append(f"elapsed (all sessions): {tot / 3600:.2f} h; sessions {st.get('sessions', 1)}")
    if st.get("eta_seconds") is not None:
        lines.append(f"estimated remaining: {float(st['eta_seconds']) / 3600:.1f} h (around {st.get('eta_at')})")
    lc = st.get("last_checkpoint") or {}
    if lc:
        lines.append(f"last checkpoint: {lc.get('item')} at {lc.get('at')}")
    fl = st.get("failures") or []
    lines.append(f"failures: {len(fl)}; retries: {st.get('retries', 0)}" + ("".join(f"\n  - {f.get('item')}: {f.get('error')}" for f in fl[:10])))
    if st.get("device", {}) and (st.get("device") or {}).get("device") == "cpu" and (st.get("device") or {}).get("requested") in ("auto", "gpu"):
        lines.append(f"device note: {(st.get('device') or {}).get('reason')}")
    if st.get("status") in ("COMPLETE", "COMPLETE_WITH_FAILURES", "REPORT_COMPLETE"):
        lines.append(f"results: {out / 'share'} (aggregate only)")
    if st.get("status") == "STOPPED":
        lines.append(f"stopped: [{st.get('stop_gate')}] {st.get('stop_message')}")
    if st.get("status") == "STOPPED_PREFLIGHT":
        lines.append("STOP - 2026 REDEVELOPMENT NOT DEFENSIBLE: see preflight/PHASE5_PREFLIGHT.md")
    return {"status": st.get("status"), "alive": alive, "heartbeat_age_seconds": age, "text": "\n".join(lines), "raw": st, "checked_at": time.strftime("%Y-%m-%d %H:%M:%S")}
