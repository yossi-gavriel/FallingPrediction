"""Build the independent, mail-safe Phase 5 Windows package from the handoff build (tools/build_handoff.py).

    python tools/build_handoff.py --dist <dist> --force
    python tools/build_phase5_package.py --handoff <dist>/falls_ml_handoff --out <dist>

Writes <out>/falls_ml_phase5_<version>_mailsafe.zip: one top folder falls_ml_phase5_<version>/ holding the complete package (its own setup, lock
files, configuration, tests, docs); every *.py and *.cmd is stored as *.py.txt / *.cmd.txt (mail filters block them);
RESTORE_FILES.py.txt renames them back and verifies EVERY file against PACKAGE_MANIFEST.txt (sha256). The package never contains data, outputs,
environments or model objects. Deterministic: sorted entries and fixed timestamps. Standard library only.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
from pathlib import Path

EPOCH = (2026, 1, 1, 0, 0, 0)
RENAME = (".py", ".cmd")

RESTORE = '''"""Restore the mail-safe Phase 5 package: rename *.py.txt / *.cmd.txt back and verify every file against PACKAGE_MANIFEST.txt (sha256).

Usage (CMD, inside this folder):   python RESTORE_FILES.py.txt
Exit code 0 = every file restored and verified; 1 = a file is missing or differs (do not use the package; request a new copy).
"""
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SELF = Path(__file__).resolve()


def main() -> int:
    n = 0
    for src in sorted(ROOT.rglob("*.txt")):
        if src.resolve() == SELF or not (src.name.endswith(".py.txt") or src.name.endswith(".cmd.txt")):
            continue
        dst = src.with_suffix("")
        if dst.exists():
            dst.unlink()
        src.replace(dst)
        n += 1
    bad = []
    lines = (ROOT / "PACKAGE_MANIFEST.txt").read_text(encoding="utf-8").splitlines()
    for line in lines:
        if not line.strip():
            continue
        digest, rel = line.split("  ", 1)
        p = ROOT / rel
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            bad.append(rel)
    print(f"restored {n} files; verified {len(lines) - len(bad)} of {len(lines)} against PACKAGE_MANIFEST.txt")
    if bad:
        print("FAILED - missing or different: " + ", ".join(bad[:20]))
        return 1
    print("PACKAGE VERIFIED - next: setup_windows.cmd (see PHASE5_WINDOWS_README.txt)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--handoff", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    root = a.handoff
    version = (root / "VERSION").read_text(encoding="utf-8").strip()
    top = f"falls_ml_phase5_{version}"
    manifest = [ln for ln in (root / "PACKAGE_MANIFEST.txt").read_text(encoding="utf-8").splitlines() if ln.strip()]
    listed = {ln.split("  ", 1)[1] for ln in manifest}
    files = sorted(p for p in root.rglob("*") if p.is_file())
    zpath = a.out / f"{top}_mailsafe.zip"
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:
            rel = p.relative_to(root).as_posix()
            if rel != "PACKAGE_MANIFEST.txt" and rel not in listed:
                raise SystemExit(f"{rel} is not in PACKAGE_MANIFEST.txt")
            arc = f"{top}/{rel}" + (".txt" if p.suffix in RENAME else "")
            info = zipfile.ZipInfo(arc, EPOCH)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, p.read_bytes())
        info = zipfile.ZipInfo(f"{top}/RESTORE_FILES.py.txt", EPOCH)
        info.compress_type = zipfile.ZIP_DEFLATED
        z.writestr(info, RESTORE.encode("utf-8"))
        readme = (root / "docs" / "meuhedet" / "PHASE5_WORK_PC_RUNBOOK.md").read_bytes()
        info = zipfile.ZipInfo(f"{top}/PHASE5_WINDOWS_README.txt", EPOCH)
        info.compress_type = zipfile.ZIP_DEFLATED
        z.writestr(info, readme)
    sha = hashlib.sha256(zpath.read_bytes()).hexdigest()
    (a.out / f"SHA256SUMS_phase5_{version}.txt").write_text(f"{sha}  {zpath.name}\n", encoding="utf-8", newline="\n")
    print(f"{zpath}: {len(files)} package files; sha256 {sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
