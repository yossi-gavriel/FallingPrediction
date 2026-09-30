"""Packaging and repository hygiene: versions, the data-schema template, the handoff builder and its content scanners."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
import tomllib
import zipfile
from pathlib import Path

import pytest

import falls_ml
from falls_ml.data.schema import VERSION_COLUMNS
from falls_ml.features.spec import load_feature_spec

ROOT = Path(__file__).resolve().parents[2]


def _load_tool(name: str):
    spec = importlib.util.spec_from_file_location(f"_tool_{name}", ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses need the module registered while the class body runs
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def builder():
    return _load_tool("build_handoff")


@pytest.fixture(scope="module")
def schema_tool():
    return _load_tool("generate_data_schema")


@pytest.fixture(scope="module")
def spec():
    return load_feature_spec(ROOT / "configs/features/efalls_v1.yaml")


# ------------------------------------------------------------------------------------------------ repository state
def test_version_is_consistent():
    version = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    assert version == pyproject == falls_ml.__version__


def test_example_schema_is_up_to_date(schema_tool):
    assert schema_tool.check(ROOT) == []


def test_example_schema_describes_every_column(schema_tool, spec):
    with (ROOT / "data/example_schema.csv").open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert tuple(rows[0]) == schema_tool.COLUMNS
    by_role: dict[str, list[str]] = {}
    for r in rows:
        by_role.setdefault(r["role"], []).append(r["column"])
    assert by_role["predictor"] == spec.predictor_names()
    assert len(rows) == len({r["column"] for r in rows})
    training = schema_tool.required_columns(spec, "training")
    scoring = schema_tool.required_columns(spec, "inference")
    for r in rows:
        assert (r["required_for_training"] == "yes") == (r["column"] in training), r["column"]
        assert (r["required_for_scoring"] == "yes") == (r["column"] in scoring), r["column"]
        assert r["missing_value_behaviour"], r["column"]
    predictors = {r["column"]: r for r in rows if r["role"] == "predictor"}
    assert predictors["sex"]["allowed_values"] == "female|male"
    assert predictors["age_years"]["allowed_values"] == "65.0..120.0"
    assert all(predictors[f.name]["allowed_values"] == "0|1" for f in spec.features if f.is_binary)


def test_header_only_template_matches_validator_and_fixture(schema_tool, spec):
    lines = (ROOT / "data/example_header_only.csv").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    header = lines[0].split(",")
    assert header == [c for c in schema_tool.required_columns(spec, "training") if c not in VERSION_COLUMNS]
    manifest = json.loads((ROOT / "data/fixtures/synthetic_v1/manifest.json").read_text(encoding="utf-8"))
    import pyarrow.parquet as pq

    fixture_columns = set(pq.read_schema(ROOT / "data/fixtures/synthetic_v1" / manifest["data_file"]).names)
    assert set(header) <= fixture_columns


def test_windows_handoff_starts_with_quick_start(builder):
    text = (ROOT / "WINDOWS_HANDOFF.md").read_text(encoding="utf-8")
    assert text.startswith(builder.HANDOFF_START)


def _repo_text_files(builder) -> list[str]:
    rels = [p.name for p in ROOT.iterdir() if p.is_file() and p.suffix in {".md", ".cmd"}]
    for top in ("src", "configs", "docs", "tests", "tools", "scripts"):
        for p in sorted((ROOT / top).rglob("*")):
            rel = p.relative_to(ROOT).as_posix()
            if p.is_file() and builder.excluded_reason(rel) is None and p.suffix not in builder.BINARY_SUFFIXES:
                rels.append(rel)
    return rels


def test_no_machine_specific_paths_in_repository(builder):
    findings = [f for rel in _repo_text_files(builder)
                for f in builder.scan_text(rel, (ROOT / rel).read_text(encoding="utf-8"), kinds=("path",))]
    assert not findings, "\n".join(map(str, findings))


def test_no_credentials_or_connection_strings_in_repository(builder):
    findings = [f for rel in _repo_text_files(builder)
                for f in builder.scan_text(rel, (ROOT / rel).read_text(encoding="utf-8"), kinds=("credential",))]
    assert not findings, "\n".join(map(str, findings))


# ------------------------------------------------------------------------------------------------ builder on a fake tree
def _write(path: Path, data: bytes | str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data.encode("utf-8") if isinstance(data, str) else data)
    return path


def _fixture(root: Path, name: str = "synthetic_v1", *, source: str = "synthetic_fixture", scientific: bool = False) -> None:
    data = b"PAR1 not really parquet PAR1"
    d = root / "data/fixtures" / name
    _write(d / "modeling_dataset.parquet", data)
    _write(d / "manifest.json", json.dumps({"source": source, "scientific_use_allowed": scientific, "data_file": "modeling_dataset.parquet",
                                            "data_sha256": hashlib.sha256(data).hexdigest()}))


@pytest.fixture
def fake_tree(tmp_path):
    root = tmp_path / "project root"
    _write(root / "VERSION", "9.8.7\n")
    _write(root / "pyproject.toml", '[project]\nname = "falls_ml"\nversion = "9.8.7"\n')
    _write(root / "src/falls_ml/__init__.py", '"""x"""\n\n__version__ = "9.8.7"\n')
    _write(root / "src/falls_ml/cli.py", "def main():\n    return 0\n")
    _write(root / "src/falls_ml/__pycache__/cli.cpython-313.pyc", b"\x00\x01")
    _write(root / "src/falls_ml.egg-info/PKG-INFO", "Name: falls_ml\n")
    _write(root / "tests/.pytest_cache/v/cache/lastfailed", "{}\n")
    _write(root / "tests/runs_scratch/run.txt", "generated\n")
    _write(root / "tests/test_x.py", "def test_x():\n    assert True\n")
    _write(root / "docs/.DS_Store", b"\x00")
    _write(root / "docs/guide.md", "# Guide\n")
    _write(root / "configs/a.yaml", "a: 1\n")
    _write(root / "scripts/posix/setup.sh", "#!/usr/bin/env bash\necho ok\n")
    _write(root / "tools/tool.py", "print('ok')\n")
    _write(root / "README.md", "# readme\n")
    _write(root / "setup_windows.cmd", b"@echo off\r\necho ok\r\nexit /b 0\r\n")
    _write(root / "runs/2026-01-01_x/metrics.json", "{}\n")          # top-level output: never collected
    _write(root / "data/meuhedet/v1/modeling_dataset.parquet", b"x")   # not in the include list: never collected
    _fixture(root)
    return root


def test_collect_excludes_caches_and_generated_outputs(builder, fake_tree):
    plan = builder.collect(fake_tree)
    assert plan.problems == []
    assert "src/falls_ml/cli.py" in plan.files and "setup_windows.cmd" in plan.files
    assert "data/fixtures/synthetic_v1/modeling_dataset.parquet" in plan.files
    for rel in plan.files:
        for bad in ("__pycache__", ".egg-info", ".pytest_cache", "runs", ".DS_Store", "meuhedet"):
            assert bad not in rel, rel
    assert builder.excluded_reason("src/falls_ml/x.pyc") and builder.excluded_reason("runs_ablation/a/b.json")
    assert builder.excluded_reason("demo_outputs/baseline/report.md") and builder.excluded_reason("src/pkg.egg-info/PKG-INFO")
    assert builder.excluded_reason("src/falls_ml/reporting/report.py") is None


def test_generated_marker_inside_source_tree_fails(builder, fake_tree):
    _write(fake_tree / "docs/run/.falls_ml_generated", "")
    assert any("generated-output marker" in p for p in builder.collect(fake_tree).problems)


def test_line_ending_rules(builder, fake_tree):
    assert builder.check_line_endings("run_demo.cmd", b"@echo off\r\nexit /b 0\r\n") == []
    assert builder.check_line_endings("run_demo.cmd", b"@echo off\nexit /b 0\n")
    assert builder.check_line_endings("run_demo.cmd", "echo caf\u00e9\r\n".encode("utf-8"))
    assert builder.check_line_endings("src/a.py", b"x = 1\r\n")
    assert builder.check_line_endings("configs/a.yaml", b"a: 1\r\n")
    assert builder.check_line_endings("src/a.py", b"x = 1\n") == []
    _write(fake_tree / "setup_windows.cmd", b"@echo off\necho ok\n")
    _write(fake_tree / "configs/b.yaml", b"b: 2\r\n")
    problems = builder.validate(fake_tree).problems
    assert any("setup_windows.cmd: CMD scripts must use CRLF" in p for p in problems)
    assert any("configs/b.yaml: CR/CRLF" in p for p in problems)


def test_scanner_finds_planted_credentials_and_paths(builder):
    planted = {
        "aws_access_key_id": "key = AKIA" + "ABCDEFGHIJKLMNOP",
        "password_assignment": "pass" + "word=hunter2",
        "credential_literal": 'auth_to' + 'ken = "abcdefgh12345678"',
        "private_key_block": "-----BEGIN RSA PRIV" + "ATE KEY-----",
        "connection_string_server": "Ser" + "ver=db01;Database=falls;",
        "database_url": "postgres" + "ql://reader@db.example.org/falls",
        "jdbc_url": "jd" + "bc:sql" + "server://db01",
        "macos_user_home": "see /Us" + "ers/alice/project/file.md",
        "macos_private_tmp": "/priv" + "ate/tmp/work",
        "windows_user_profile": "C:\\Us" + "ers\\alice\\Desktop",
    }
    for rule, text in planted.items():
        assert rule in {f.rule for f in builder.scan_text("docs/x.md", text)}, rule
    clean = ["token = hashlib.sha256(data).hexdigest()", "https://example.org/home/page", r"C:\Projects\Falls Research\falls_ml_handoff",
             "set HTTPS_PROXY=http://proxy.example.org:8080", "no credentials are needed", "import pymssql is forbidden"]
    for text in clean:
        assert builder.scan_text("docs/x.md", text) == [], text


def test_validate_fails_on_planted_token_file(builder, fake_tree):
    _write(fake_tree / "configs/connection.yaml", "db:\n  aws_key: AKIA" + "QRSTUVWXYZ012345\n")
    problems = builder.validate(fake_tree).problems
    assert any("configs/connection.yaml:2: credential/aws_access_key_id" in p for p in problems)


@pytest.mark.parametrize(("source", "scientific", "message"), [
    ("meuhedet_dwh", False, "only 'synthetic_fixture' may be packaged"),
    ("synthetic_fixture", True, "scientific_use_allowed must be false"),
])
def test_non_synthetic_fixture_parquet_is_refused(builder, fake_tree, source, scientific, message):
    _fixture(fake_tree, "synthetic_enhanced_v1", source=source, scientific=scientific)
    problems = builder.validate(fake_tree).problems
    assert any(message in p for p in problems), problems


def test_data_files_outside_fixtures_are_refused(builder, fake_tree):
    _write(fake_tree / "docs/extract.parquet", b"PAR1")
    _write(fake_tree / "tests/patients.csv", "research_id,age\nr1,70\n")
    _write(fake_tree / "tools/model.pkl", b"\x80\x04")
    problems = "\n".join(builder.validate(fake_tree).problems)
    assert "docs/extract.parquet: data file outside data/fixtures" in problems
    assert "tests/patients.csv: data file outside data/fixtures" in problems
    assert "tools/model.pkl" in problems
    fixture = fake_tree / "data/fixtures/synthetic_v1/modeling_dataset.parquet"
    fixture.write_bytes(b"modified")
    assert any("sha256 differs from manifest" in p for p in builder.validate(fake_tree).problems)


def test_missing_required_files_fail_unless_allowed(builder, fake_tree, tmp_path):
    with pytest.raises(builder.BuildError, match="missing required files"):
        builder.build(fake_tree, tmp_path / "dist", check_only=True)
    assert builder.build(fake_tree, tmp_path / "dist", check_only=True, allow_missing=True)["version"] == "9.8.7"
    assert not (tmp_path / "dist").exists()


def test_build_is_reproducible_and_force_replaces_only_its_outputs(builder, fake_tree, tmp_path, monkeypatch):
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1767225600")  # reproducible builds pin the timestamp explicitly
    first = builder.build(fake_tree, tmp_path / "dist1", allow_missing=True)
    second = builder.build(fake_tree, tmp_path / "dist2", allow_missing=True)
    zip1, zip2 = Path(first["zip"]), Path(second["zip"])
    assert zip1.name == "falls_ml_handoff_9.8.7.zip" and zip1.read_bytes() == zip2.read_bytes()

    folder = tmp_path / "dist1/falls_ml_handoff"
    manifest_lines = (folder / "PACKAGE_MANIFEST.txt").read_text(encoding="utf-8").splitlines()
    paths = [line.split("  ", 1)[1] for line in manifest_lines]
    assert paths == sorted(paths) and "PACKAGE_MANIFEST.txt" not in paths
    for line in manifest_lines:
        digest, rel = line.split("  ", 1)
        assert hashlib.sha256((folder / rel).read_bytes()).hexdigest() == digest
    with zipfile.ZipFile(zip1) as zf:
        names = zf.namelist()
        assert names == sorted(names) and all(n.startswith("falls_ml_handoff/") for n in names)
        assert "falls_ml_handoff/PACKAGE_MANIFEST.txt" in names
        assert {i.date_time for i in zf.infolist()} == {(2026, 1, 1, 0, 0, 0)}
        assert zf.read("falls_ml_handoff/setup_windows.cmd") == (fake_tree / "setup_windows.cmd").read_bytes()
    sums = (tmp_path / "dist1/SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()
    assert sums[0] == f"{hashlib.sha256(zip1.read_bytes()).hexdigest()}  falls_ml_handoff_9.8.7.zip"
    assert f"{hashlib.sha256(b'9.8.7' + bytes([10])).hexdigest()}  falls_ml_handoff/VERSION" in sums

    with pytest.raises(builder.BuildError, match="already exists"):
        builder.build(fake_tree, tmp_path / "dist1", allow_missing=True)
    keep = _write(tmp_path / "dist1/unrelated.txt", "keep me\n")
    monkeypatch.delenv("SOURCE_DATE_EPOCH", raising=False)
    third = builder.build(fake_tree, tmp_path / "dist1", allow_missing=True, force=True)
    assert keep.read_text(encoding="utf-8") == "keep me\n"
    assert Path(third["zip"]).read_bytes() != zip2.read_bytes()  # default timestamps are the build time
    with zipfile.ZipFile(third["zip"]) as zf:
        assert min(i.date_time for i in zf.infolist()) > (2026, 1, 1, 0, 0, 0)
    with pytest.raises(builder.BuildError, match="contains the source project"):
        builder._prepare_dist(tmp_path, "9.8.7", True, tmp_path / "falls_ml_handoff" / "project")


# ------------------------------------------------------------------------------------------------ line endings
def test_gitattributes_protects_line_endings_and_is_packaged(builder):
    lines = [ln for ln in (ROOT / ".gitattributes").read_text(encoding="utf-8").splitlines() if ln and not ln.startswith("#")]
    assert lines == ["* text=auto eol=lf", "*.cmd text eol=crlf", "*.bat text eol=crlf", "*.parquet binary", "*.whl binary", "*.png binary",
                     "*.zip binary"]
    assert ".gitattributes" in builder.TOP_FILES and ".gitattributes" in builder.REQUIRED_FILES
    assert ".gitattributes" in builder.collect(ROOT).files
    assert builder.check_line_endings(".gitattributes", (ROOT / ".gitattributes").read_bytes()) == []


def test_no_text_file_is_exempt_from_lf(builder):
    assert builder.LINE_ENDING_EXEMPT == {}
    assert b"\r" not in (ROOT / "tools/source_table_s3_2.csv").read_bytes()
