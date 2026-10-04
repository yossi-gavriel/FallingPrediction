"""Write configs/meuhedet/PHASE4_0.10.0_PROTECTED.sha256: the sha256 of every file of the delivered (frozen, unused) Phase 4 package
(falls_ml 0.10.0) - its package, settings, runbook, design, tests, builder and the Phase 3 protection manifest it verifies. The Phase 5 test suite
verifies that none of them changed (Phase 4 stays intact for possible future use). The shared modules Phase 4 imports are covered by
PHASE3_0.9.0_PROTECTED.sha256 (verified by the same suite). Line endings are normalised (LF).

Usage: python tools/phase4_protection_manifest.py   (run once, on the unchanged 0.10.0 tree)
"""
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = "configs/meuhedet/PHASE4_0.10.0_PROTECTED.sha256"


def files() -> list[str]:
    out = [p.relative_to(ROOT).as_posix() for p in sorted((ROOT / "src" / "falls_ml" / "phase4").glob("*.py"))]
    out += ["configs/meuhedet/phase4.yaml", "configs/meuhedet/PHASE3_0.9.0_PROTECTED.sha256", "docs/meuhedet/PHASE4_WORK_PC_RUNBOOK.md",
            "planning/PHASE4_DESIGN.md", "tests/unit/test_phase4_contract.py", "tests/unit/test_phase4_e2e.py", "tools/build_phase4_package.py",
            "tools/phase3_protection_manifest.py"]
    return out


def main() -> int:
    lines = ["# sha256 (LF-normalised) of the delivered Phase 4 (falls_ml 0.10.0) files; verified by",
             "# tests/unit/test_phase5_contract.py::test_phase2_3_4_files_are_unchanged"]
    for rel in files():
        data = (ROOT / rel).read_bytes().replace(b"\r\n", b"\n")
        lines.append(f"{hashlib.sha256(data).hexdigest()}  {rel}")
    (ROOT / TARGET).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(lines) - 2} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
