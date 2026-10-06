"""The narrow lifecycle exception must not permit unrelated protected changes."""
from __future__ import annotations

import hashlib
from pathlib import Path


def test_reviewed_exception_is_bound_to_exact_path_and_baseline(protected_source_matches):
    root = Path(__file__).resolve().parents[2]
    rel = "src/falls_ml/phase2/xgb_tuning.py"
    original = "60617bff8d1389242c227e7629a74f646d86d64521e254f87f774f6b2a129da1"
    data = (root / rel).read_bytes()
    assert protected_source_matches(rel, original, data)
    assert not protected_source_matches(rel, original, data + b"\n# unreviewed change\n")
    assert not protected_source_matches("src/falls_ml/phase2/enet.py", original, data)
    assert not protected_source_matches(rel, "0" * 64, data)
    unchanged = b"unchanged historical source\n"
    assert protected_source_matches("any/frozen/file.py", hashlib.sha256(unchanged).hexdigest(), unchanged)
