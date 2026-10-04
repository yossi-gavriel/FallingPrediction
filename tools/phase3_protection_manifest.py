"""Write configs/meuhedet/PHASE3_0.9.0_PROTECTED.sha256: the sha256 of every file the delivered Phase 3 (falls_ml 0.9.0) experiment executes or
reads (its package, the shared modules it imports - which also define the classes of its persisted model objects - its configurations and frozen
file). The Phase 4 test suite verifies that none of them changed, so the frozen Phase 3 models load and re-fit with exactly the delivered code.
Excluded by design: src/falls_ml/cli.py (Phase 4 only ADDS commands) and src/falls_ml/__init__.py (package version). Line endings are
normalised (LF) so a Windows checkout gives the same hashes.

Usage: python tools/phase3_protection_manifest.py   (run once, on the unchanged 0.9.0 tree)
"""
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDE = {"src/falls_ml/cli.py", "src/falls_ml/__init__.py"}
TARGET = "configs/meuhedet/PHASE3_0.9.0_PROTECTED.sha256"


def files() -> list[str]:
    out = []
    for p in sorted((ROOT / "src" / "falls_ml").rglob("*.py")):
        rel = p.relative_to(ROOT).as_posix()
        if "/phase4/" in rel or "__pycache__" in rel or rel in EXCLUDE:
            continue
        out.append(rel)
    for name in ("phase3.yaml", "phase3_recovery.yaml", "phase3_time_contract.yaml", "PHASE3_FINAL_EXPERIMENT_CONFIG.json",
                 "PHASE3_FINAL_EXPERIMENT_CONFIG.sha256", "phase2.yaml", "phase2_features.yaml", "FINAL_EXPERIMENT_CONFIG.json",
                 "FINAL_EXPERIMENT_CONFIG.sha256", "d00_sensitivity.yaml", "wide_v1_columns.yaml", "wide_v1_data_dictionary.yaml",
                 "wide_v1_efalls_mapping.yaml", "feature_labels_he.yaml", "PHASE2_0.8.1_PROTECTED.sha256"):
        out.append(f"configs/meuhedet/{name}")
    out += ["tools/freeze_phase3_config.py", "requirements.lock"]
    return out


def main() -> int:
    lines = ["# sha256 (LF-normalised) of the files the delivered Phase 3 (falls_ml 0.9.0) experiment executes or reads; verified by",
             "# tests/unit/test_phase4_contract.py::test_phase3_files_are_byte_identical_to_the_delivered_0_9_0_package"]
    for rel in files():
        data = (ROOT / rel).read_bytes().replace(b"\r\n", b"\n")
        lines.append(f"{hashlib.sha256(data).hexdigest()}  {rel}")
    (ROOT / TARGET).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(lines) - 2} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
