"""``falls_ml meuhedet-phase4-status``: READ-ONLY progress of a Phase 4 folder (never writes, moves or locks anything; never reads an outcome)."""

from __future__ import annotations

import json
from pathlib import Path

from falls_ml.phase4.common import MODEL_MANIFEST, PLAN, PRED_MANIFEST, SCORE_DONE, dirs


def _j(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def phase4_status(out_dir: str | Path) -> int:
    out = Path(out_dir)
    d = dirs(out)
    if not (out / PLAN).is_file():
        print(f"{out.name}: no Phase 4 plan - run meuhedet-phase4-preflight first")
        return 1
    pre = _j(d["preflight"] / "PREFLIGHT_RESULT.json")
    print(f"Phase 4 folder: {out.name}")
    print(f"  1. preflight : {pre.get('verdict', 'missing')}  ({pre.get('created_at', '')})")
    for c in pre.get("checks", []):
        if c["status"] != "OK":
            print(f"       {c['status']:4s} {c['id']} {c['title']}")
    if pre.get("excluded_models"):
        print(f"       secondary models excluded (pre-declared rule): {', '.join(pre['excluded_models'])}")
    done = d["sealed"] / SCORE_DONE
    if done.is_file():
        man = _j(d["sealed"] / PRED_MANIFEST)
        mm = _j(d["frozen"] / MODEL_MANIFEST)
        print(f"  2. blind score: COMPLETE ({man.get('created_at', '')}); {man.get('n_scored')} patients; outcomes read: {man.get('outcomes_read')}")
        print(f"       predictions sha256 {str(man.get('predictions_sha256'))[:16]}…; frozen model manifest {str(man.get('frozen_model_manifest_sha256'))[:16]}…")
        for c, m in (mm.get("models") or {}).items():
            print(f"       {c}: mode {m.get('mode')} ({m.get('mode_text')})")
    else:
        print("  2. blind score: not done")
    ev = d["evaluation"]
    if (ev / "EVALUATION_COMPLETE.json").is_file():
        e = _j(ev / "EVALUATION_COMPLETE.json")
        print(f"  3. evaluation: COMPLETE ({e.get('finished_at', '')}); outcome contract {e.get('outcome_contract')}; privacy scan passed {e.get('privacy_scan')}")
    elif (ev / "STOPPED.md").is_file():
        print("  3. evaluation: STOPPED - the 2026 outcome contract failed (no performance reported); see evaluation/OUTCOME_CONTRACT_2026.json")
    elif (ev / "OUTCOMES_OPENED.jsonl").is_file():
        print("  3. evaluation: outcomes opened, not finished - run meuhedet-phase4-evaluate again (deterministic)")
    else:
        print("  3. evaluation: not started (outcomes sealed)")
    share = d["share"]
    print(f"  share: {'ready - send ONLY this folder: ' + share.name if (share / 'PRIVACY_SCAN.json').is_file() else 'not yet'}")
    return 0
