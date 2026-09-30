"""Freeze the Phase 3 scientific configuration: write configs/meuhedet/PHASE3_FINAL_EXPERIMENT_CONFIG.json and its .sha256 from the current
settings, recovery rules, time contract, the unchanged Phase 2 catalogue and the result-relevant code constants. The runner and the preflight refuse
any other effective configuration. Never touches the Phase 2 freeze (configs/meuhedet/FINAL_EXPERIMENT_CONFIG.json).

Usage: python tools/freeze_phase3_config.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    from falls_ml.phase3.final_config import FROZEN_PATH, build_final_config, sha256_of, sha_line, to_bytes
    from falls_ml.phase3.runner import load_all

    L = load_all()
    fc = build_final_config(L["cfg"], L["rules"], L["cat"], contract=L["contract"], dictionary=L["dictionary"], mapping=L["mapping"], d00=L["d00"], tc=L["tc"])
    out = ROOT / FROZEN_PATH
    out.write_bytes(to_bytes(fc))
    out.with_suffix(".sha256").write_text(sha_line(sha256_of(fc)), encoding="utf-8", newline="\n")
    print(f"{FROZEN_PATH}: sha256 {sha256_of(fc)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
