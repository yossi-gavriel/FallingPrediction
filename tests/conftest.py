"""Verify historical source freezes with the one reviewed 0.12.3 lifecycle patch.

Historical manifests are immutable. The user authorized a necessary source fix
in the shared Phase 2 tuning module; accept only that exact reviewed fingerprint,
for its exact old fingerprint and path, plus the corresponding frozen test guard.
Every other protected file stays frozen.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATCH_PATH = "src/falls_ml/phase2/xgb_tuning.py"
ORIGINAL_SHA = "60617bff8d1389242c227e7629a74f646d86d64521e254f87f774f6b2a129da1"
ORIGINAL_SHAS = {
    PATCH_PATH: ORIGINAL_SHA,
    "tests/unit/test_phase4_contract.py": "20385302f5ddf17b3c691a5ce38787f5a77f0610dc1a4e995abb7bca2e854798",
}


@pytest.fixture
def protected_source_matches():
    manifest = ROOT / "configs/meuhedet/SOURCE_PATCH_EXCEPTION_0.12.3.sha256"
    entries = [line.split("  ", 1) for line in manifest.read_text(encoding="utf-8").splitlines()
               if line.strip() and not line.startswith("#")]
    patches = {rel: digest for digest, rel in entries}
    assert len(entries) == len(patches) == 2 and set(patches) == set(ORIGINAL_SHAS)

    def matches(rel: str, expected: str, data: bytes) -> bool:
        actual = hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()
        if actual == expected:
            return True
        return (rel in ORIGINAL_SHAS and expected == ORIGINAL_SHAS[rel]
                and (ROOT / "VERSION").read_text(encoding="utf-8").strip() == "0.12.3"
                and actual == patches[rel])

    return matches
