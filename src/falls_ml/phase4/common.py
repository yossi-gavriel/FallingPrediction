"""Phase 4 plumbing: the output-folder layout, guards, the plan record and the event log."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from falls_ml.artifacts import utc_now
from falls_ml.phase2 import durable as D
from falls_ml.phase2.state import Phase2Stop

PLAN = "PHASE4_PLAN.json"
SCORE_DONE = "BLIND_SCORE_COMPLETE.txt"
PRED_FILE = "BLIND_PREDICTIONS.parquet"
PRED_MANIFEST = "BLIND_PREDICTIONS_MANIFEST.json"
MODEL_MANIFEST = "FROZEN_MODEL_MANIFEST.json"
DEFAULT_PHASE3_CONFIG = "configs/meuhedet/phase3.yaml"


def dirs(out: Path) -> dict[str, Path]:
    return {"preflight": out / "preflight", "frozen": out / "frozen", "sealed": out / "sealed", "evaluation": out / "evaluation", "share": out / "share",
            "logs": out / "logs"}


def guard_out(out: Path, *, phase3_out: Path, inputs: list[Path], allow_synced_folder: bool) -> None:
    o = out.resolve()
    p3 = phase3_out.resolve()
    if o == p3 or p3 in o.parents or o in p3.parents:
        raise Phase2Stop("OUT_OVERLAPS_PHASE3", "--out must be a NEW folder, separate from (not inside or around) the Phase 3 output folder")
    if (out / "PHASE2_PLAN.json").exists() or (out / "PHASE3_PLAN.json").exists():
        raise Phase2Stop("OUT_IS_EARLIER_PHASE", "--out holds a Phase 2 / Phase 3 run: Phase 4 never writes into an earlier phase's folder")
    for i in inputs:
        if i.resolve().parent == o:
            raise Phase2Stop("OUT_HOLDS_INPUT", f"--out must not be the folder of the input {i.name}")
    synced = D.synced_folder(out.parent if not out.exists() else out)
    if synced and not allow_synced_folder:
        raise Phase2Stop("SYNCED_FOLDER", f"--out is inside a synchronised folder ({synced})", ["choose a local folder, or pass --allow-synced-folder (recorded)"])


def event(out: Path, command: str, action: str, status: str, **kw: Any) -> None:
    D.append_jsonl(out / "logs" / "events.jsonl", {"ts": utc_now(), "command": command, "action": action, "status": status, **kw})


def read_json(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def input_identity(p: Path) -> dict[str, Any]:
    from falls_ml.data.dataset import sha256_file

    if not p.is_file():
        raise Phase2Stop("INPUT_MISSING", f"input file not found: {p.name}")
    return {"name": p.name, "sha256": sha256_file(p), "bytes": p.stat().st_size}


def load_definitions(phase3_config: str | Path, phase4_config: str | Path) -> dict[str, Any]:
    from falls_ml.phase3.runner import load_all
    from falls_ml.phase4.config import load_phase4_config

    L = load_all(phase3_config)
    L["cfg4"] = load_phase4_config(phase4_config)
    return L


def code_sha() -> str:
    from falls_ml.artifacts import source_tree_sha256

    return source_tree_sha256()


def sha_json(obj: Any) -> str:
    return D.sha256_text(json.dumps(obj, sort_keys=True, default=str))
