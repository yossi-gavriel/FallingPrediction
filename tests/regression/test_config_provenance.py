"""Config provenance (spec §16): the generated YAML configs are reproducible byte-for-byte from the source tables."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.helpers.subprocesses import utf8_env

REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATOR = REPO_ROOT / "tools/generate_efalls_configs.py"
SOURCES = ("tools/source_table_s3_2.csv", "tools/source_table_s3_1_binary_predictors.json")
GENERATED = ("configs/features/efalls_v1.yaml", "configs/models/efalls_published.yaml", "configs/mappings/meuhedet_v0.yaml")


def test_generated_configs_are_byte_identical(tmp_path):
    for rel in SOURCES:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / rel, tmp_path / rel)
    result = subprocess.run(
        [sys.executable, str(GENERATOR), "--coefficients", str(tmp_path / SOURCES[0]), "--predictors", str(tmp_path / SOURCES[1]),
         "--project-root", str(tmp_path)],
        cwd=tmp_path, capture_output=True, text=True, encoding="utf-8", errors="replace", env=utf8_env(), timeout=120, check=False)
    assert result.returncode == 0, result.stderr
    for rel in GENERATED:
        generated, committed = tmp_path / rel, REPO_ROOT / rel
        assert generated.is_file(), f"generator did not write {rel}"
        if generated.read_bytes() != committed.read_bytes():
            pytest.fail(f"{rel} differs from the output of tools/generate_efalls_configs.py (edit only via reviewed change)")
