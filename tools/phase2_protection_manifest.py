"""Write configs/meuhedet/PHASE2_0.8.1_PROTECTED.sha256: the sha256 of every file the running Phase 2 (falls_ml 0.8.1) experiment executes or reads
(its package, the shared modules it imports, its configurations and frozen file). The Phase 3 test suite verifies that none of them changed.
Excluded by design: src/falls_ml/cli.py (Phase 3 only ADDS two commands) and src/falls_ml/__init__.py (package version). Line endings are
normalised (LF) so a Windows checkout gives the same hashes.

Usage: python tools/phase2_protection_manifest.py   (run once, on the unchanged 0.8.1 tree)
"""
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDE = {"src/falls_ml/cli.py", "src/falls_ml/__init__.py"}


def files() -> list[str]:
    out = []
    for p in sorted((ROOT / "src" / "falls_ml").rglob("*.py")):
        rel = p.relative_to(ROOT).as_posix()
        if "/phase3/" in rel or "__pycache__" in rel or rel in EXCLUDE:
            continue
        out.append(rel)
    for name in ("phase2.yaml", "phase2_features.yaml", "FINAL_EXPERIMENT_CONFIG.json", "FINAL_EXPERIMENT_CONFIG.sha256", "d00_sensitivity.yaml",
                 "wide_v1_columns.yaml", "wide_v1_data_dictionary.yaml", "wide_v1_efalls_mapping.yaml", "feature_labels_he.yaml"):
        out.append(f"configs/meuhedet/{name}")
    out += ["tools/freeze_phase2_config.py", "requirements.lock"]
    return out


def main() -> int:
    lines = ["# sha256 (LF-normalised) of the files the running Phase 2 (falls_ml 0.8.1) experiment executes or reads; verified by",
             "# tests/unit/test_phase3_contract.py::test_phase2_files_are_byte_identical_to_the_running_0_8_1_package"]
    for rel in files():
        data = (ROOT / rel).read_bytes().replace(b"\r\n", b"\n")
        lines.append(f"{hashlib.sha256(data).hexdigest()}  {rel}")
    (ROOT / "configs/meuhedet/PHASE2_0.8.1_PROTECTED.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(lines) - 2} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
