"""Build the self-contained Windows handoff package ``dist/falls_ml_handoff/`` and its reproducible zip.

Standard library only (shutil/zipfile/hashlib/re); it never imports falls_ml or a third-party package.

What it does, in order (nothing is written until every check has passed):
  1. collects files from an explicit include list (top-level files including .gitattributes, source trees, data templates,
     synthetic fixtures)
     and skips caches and generated outputs (.venv, __pycache__, *.egg-info, runs*, reports, logs, demo_outputs, ...);
  2. checks: required files present; VERSION == pyproject version == falls_ml.__version__; .cmd files CRLF + ASCII and
     every other text file LF; no machine-specific absolute paths; no credentials or database connection strings;
     no data file outside data/fixtures (documented allowlist below); every parquet belongs to a synthetic fixture
     manifest (source synthetic_fixture, scientific_use_allowed false, hash matches); lock files in the hash-pinned format;
  3. copies the files to ``<dist>/falls_ml_handoff/``, writes ``PACKAGE_MANIFEST.txt`` ("<sha256>  <path>", sorted),
     ``<dist>/falls_ml_handoff_<VERSION>.zip`` (sorted entries, ZIP_DEFLATED, fixed timestamps from SOURCE_DATE_EPOCH or
     2026-01-01) and ``<dist>/SHA256SUMS.txt``.

Usage:
  python tools/build_handoff.py                  # build into dist/ (refuses to overwrite)
  python tools/build_handoff.py --force          # replace dist/falls_ml_handoff and the zip a previous build produced
  python tools/build_handoff.py --check-only     # run every check, write nothing
  python tools/build_handoff.py --dist <dir> --allow-missing   # development builds while required files are missing
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
import time
import tomllib
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "falls_ml_handoff"
MANIFEST_NAME = "PACKAGE_MANIFEST.txt"
SUMS_NAME = "SHA256SUMS.txt"
DEFAULT_EPOCH = 1767225600  # 2026-01-01T00:00:00Z
ZIP_MIN_EPOCH = 315532800   # 1980-01-01T00:00:00Z, the earliest timestamp a zip entry can hold

TOP_FILES = ("README.md", "WINDOWS_HANDOFF.md", "VERSION", "pyproject.toml", "requirements.lock", "requirements-build.lock",
             "setup_windows.cmd", "verify_installation.cmd", "run_demo.cmd", "run_full_demo.cmd",
             "prepare_offline_package.cmd", "clean_demo_outputs.cmd",
             ".gitattributes")  # keeps LF text / CRLF .cmd line endings if the folder is later put under git on Windows
TREES = ("src", "configs", "docs", "tests", "tools", "scripts")
DATA_FILES = ("data/README.md", "data/example_schema.csv", "data/example_header_only.csv")
FIXTURES = ("data/fixtures/synthetic_v1", "data/fixtures/synthetic_enhanced_v1", "data/fixtures/synthetic_reduced_v1")
HANDOFF_SCRIPTS = ("common", "check_python", "setup_steps", "verify", "smoke", "demo", "clean_outputs", "prepare_offline",
                   "environment_report")
POSIX_SCRIPTS = ("setup", "verify", "run_demo", "run_full_demo", "prepare_offline", "clean_demo_outputs")
SUMS_FILES = ("requirements.lock", "requirements-build.lock", "configs/features/efalls_v1.yaml", "configs/models/efalls_published.yaml",
              "configs/experiments/efalls_published_scoring.yaml", "configs/experiments/efalls_retrained_lasso.yaml",
              "configs/experiments/efalls_retrained_reduced.yaml", "configs/experiments/fixture/efalls_published_scoring.yaml",
              "configs/experiments/fixture/efalls_retrained_lasso.yaml", "configs/experiments/fixture/efalls_retrained_reduced.yaml",
              "setup_windows.cmd", "WINDOWS_HANDOFF.md", "VERSION", MANIFEST_NAME)
REQUIRED_FILES = (
    *TOP_FILES, *DATA_FILES, *(f"{d}/manifest.json" for d in FIXTURES),
    "docs/DEPENDENCIES.md", "docs/ARCHITECTURE.md", "docs/ARTIFACT_SCHEMAS.md", "docs/EFALLS_REPRODUCTION_SPEC.md",
    "src/falls_ml/__init__.py", "src/falls_ml/cli.py", "tools/build_handoff.py", "tools/generate_data_schema.py",
    "scripts/handoff/lockfile.py", *(f"scripts/handoff/{n}.py" for n in HANDOFF_SCRIPTS), *(f"scripts/posix/{n}.sh" for n in POSIX_SCRIPTS),
    *(s for s in SUMS_FILES if s != MANIFEST_NAME and s not in TOP_FILES),
)
#: WINDOWS_HANDOFF.md must start with exactly this text.
HANDOFF_START = ("1. Copy this folder to the work computer.\n2. Open CMD in this directory.\n3. Run:\n   setup_windows.cmd\n"
                 "4. Wait for:\n   INSTALLATION SUCCESSFUL\n5. Run:\n   run_demo.cmd\n")

# ------------------------------------------------------------------------------------------------ exclusion rules
EXCLUDED_DIRS = frozenset({
    ".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".ipynb_checkpoints", ".tox", ".git", ".hg",
    ".svn", ".idea", ".vscode", "htmlcov", "build", "dist", "reports", "logs", "demo_outputs", "setup_logs", "offline_packages",
    "00_working", "outputs", "presentation", "scratch", "scratchpad", "tmp",
})
EXCLUDED_DIR_GLOBS = ("runs*", "baselines*", "*.egg-info", "*.dist-info")
EXCLUDED_FILE_GLOBS = ("*.pyc", "*.pyo", ".DS_Store", "Thumbs.db", "desktop.ini", "*.swp", "*~", "*.tmp", "*.bak", "*.orig",
                       "*.log", ".coverage", "environment_report.txt", MANIFEST_NAME)
#: files whose presence means generated output leaked into a source tree (the build fails instead of skipping them)
GENERATED_MARKERS = (".falls_ml_generated", "SYNTHETIC_DATA_NOT_SCIENTIFIC_RESULTS.txt")

CRLF_SUFFIXES = frozenset({".cmd", ".bat"})
TEXT_SUFFIXES = frozenset({".py", ".yaml", ".yml", ".md", ".csv", ".json", ".toml", ".lock", ".txt", ".sh", ".cfg", ".ini", ".typed", ""})
BINARY_SUFFIXES = frozenset({".parquet"})
ALLOWED_SUFFIXES = CRLF_SUFFIXES | TEXT_SUFFIXES | BINARY_SUFFIXES
DATA_SUFFIXES = frozenset({".parquet", ".csv", ".tsv", ".json", ".jsonl", ".feather", ".arrow", ".pkl", ".pickle", ".joblib", ".npy",
                           ".npz", ".h5", ".hdf5", ".xlsx", ".xls", ".sav", ".dta", ".sas7bdat", ".db", ".sqlite", ".sqlite3",
                           ".rds", ".rdata"})
#: data-like files allowed outside data/fixtures, each with the reason it is not individual-level data
DATA_FILE_ALLOWLIST = {
    "data/example_schema.csv": "column template generated from the feature spec by tools/generate_data_schema.py; no data rows",
    "data/example_header_only.csv": "header row only; the build checks it has exactly one line",
    "tools/source_table_s3_2.csv": "published aggregate coefficients (Archer et al. 2024 Table S3.2, CC BY 4.0); no individual data",
    "tools/source_table_s3_1_binary_predictors.json": "published predictor list (Table S3.1 + eFI2 rules); no individual data",
    "docs/meuhedet/tables/efalls_meuhedet_mapping.csv": "eFalls -> wide-table column mapping table generated from configs/meuhedet/*.yaml by "
                                                        "tools/generate_meuhedet_docs.py; one row per eFalls predictor, no data rows",
    "docs/meuhedet/tables/wide_v1_column_inventory.csv": "column inventory (role/type/NULL meaning) generated from the wide-table contract by "
                                                         "tools/generate_meuhedet_docs.py; one row per VIEW column, no data rows",
}
#: LF rule exemptions: path -> justification. Keep empty unless a CR/CRLF text file is provably required.
LINE_ENDING_EXEMPT: dict[str, str] = {}

# ------------------------------------------------------------------------------------------------ content scanners
# Patterns are written so that their own source text does not match (the builder scans itself).
_NOT_URL = r"(?<![\w.:/-])"
PATH_RULES = {
    "macos_user_home": _NOT_URL + r"/Users/[A-Za-z0-9._-]+",
    "linux_user_home": _NOT_URL + r"/home/[A-Za-z0-9._-]+",
    "macos_private_tmp": _NOT_URL + r"/private/(?:tmp|var)(?![\w-])",
    "macos_var_folders": _NOT_URL + r"/var/folders/[A-Za-z0-9_]",
    "tmp_claude_scratch": _NOT_URL + r"/tmp/claude-\d+",
    "homebrew_prefix": _NOT_URL + r"/opt/home(?:brew)(?![\w-])",
    "windows_user_profile": r"(?i)(?<![\w])[a-z]:(?:\\{1,2}|/)Users(?:\\{1,2}|/)[A-Za-z0-9._-]",
    "dev_project_path": r"(?i)dev[/\\-]Falling[ _-]prediction",
}
CREDENTIAL_RULES = {
    "password_assignment": r"(?i)\b(?:password|passwd|pwd)\s*[=:]",
    "credential_literal": r"(?i)\b[\w-]*(?:secre[t]|token|api[_-]?key|apikey|access[_-]?key)[\w-]*\s*[=:]\s*[\"'][^\"'\s]{8,}[\"']",
    "secret_word": r"(?i)\bsecret\b",
    "api_key_word": r"(?i)\bapi_key\b",
    "private_key_block": r"-----BEGIN[A-Z ]*PRIVATE KEY-----",
    "aws_access_key_id": r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b",
    "github_token": r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})",
    "slack_token": r"\bxox[baprs]-[A-Za-z0-9-]{10,}",
    "api_secret_key_prefix": r"\bsk-(?:ant-|proj-)?[A-Za-z0-9_-]{20,}",
    "connection_string_server": r"(?<![\w.])Server\s*=",
    "connection_string_data_source": r"(?i)\bData Source\s*=",
    "connection_string_pair": r"(?i)(?<![\w.])(?:initial catalog|trusted_connection|user id|uid)\s*=\s*[^\s;'\"=]+\s*;",
    "jdbc_url": r"(?i)\bjdbc:",
    "database_url": r"(?i)\b(?:postgres(?:ql)?|mysql|mariadb|oracle|ms(?:sql)|sqlserver|mongodb(?:\+srv)?|redshift|snowflake|teradata)"
                    r"(?:\+\w+)?://",
    "mssql_word": r"(?i)\bmssql\b",
    "url_with_credentials": r"\b[a-z][a-z0-9+.-]*://[^/\s:@'\"]+:[^/\s@'\"]+@[\w.-]+",
}
_COMPILED = {name: (kind, re.compile(rx)) for kind, rules in (("path", PATH_RULES), ("credential", CREDENTIAL_RULES)) for name, rx in rules.items()}
#: documented false positives: (relative path, rule) -> justification. Keep empty unless a hit is provably harmless.
SCAN_ALLOWLIST: dict[tuple[str, str], str] = {}


class BuildError(Exception):
    """The package cannot be built; the message lists every problem."""


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    rule: str
    kind: str
    excerpt: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.kind}/{self.rule}: {self.excerpt}"


@dataclass
class Plan:
    root: Path
    files: list[str] = field(default_factory=list)      # relative posix paths, sorted
    missing: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    version: str | None = None


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ collection
def excluded_reason(rel: str) -> str | None:
    """Why ``rel`` (posix, relative to the project root) is skipped, or None when it is packaged."""
    parts = PurePosixPath(rel).parts
    for d in parts[:-1]:
        if d in EXCLUDED_DIRS or any(fnmatch.fnmatch(d, g) for g in EXCLUDED_DIR_GLOBS):
            return f"excluded directory {d!r}"
    name = parts[-1]
    if any(fnmatch.fnmatch(name, g) for g in EXCLUDED_FILE_GLOBS):
        return f"excluded file pattern ({name})"
    return None


def _walk(root: Path, top: str, plan: Plan) -> Iterable[str]:
    base = root / top
    for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
        rel_dir = Path(dirpath).relative_to(root).as_posix()
        keep = []
        for d in sorted(dirnames):
            rel = f"{rel_dir}/{d}"
            if (root / rel).is_symlink():
                plan.problems.append(f"{rel}: symbolic links are not packaged (replace with a real directory)")
            elif excluded_reason(f"{rel}/x") is not None:
                plan.skipped.append(rel + "/")
            else:
                keep.append(d)
        dirnames[:] = keep
        for fname in sorted(filenames):
            rel = f"{rel_dir}/{fname}"
            if (root / rel).is_symlink():
                plan.problems.append(f"{rel}: symbolic links are not packaged")
                continue
            if excluded_reason(rel) is not None:
                plan.skipped.append(rel)
                continue
            yield rel


def collect(root: Path) -> Plan:
    """Explicit include list -> sorted file list, missing required files and structural problems (no content checks)."""
    plan = Plan(root=root)
    files: set[str] = set()
    for rel in (*TOP_FILES, *DATA_FILES):
        p = root / rel
        if p.is_symlink():
            plan.problems.append(f"{rel}: symbolic links are not packaged")
        elif p.is_file():
            files.add(rel)
    for top in (*TREES, *FIXTURES):
        if (root / top).is_dir():
            files.update(_walk(root, top, plan))
        else:
            plan.missing.append(top + "/")
    for rel in sorted(files):
        name = PurePosixPath(rel).name
        if name in GENERATED_MARKERS:
            plan.problems.append(f"{rel}: generated-output marker inside a source tree (move generated outputs to demo_outputs/)")
        suffix = PurePosixPath(rel).suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            plan.problems.append(f"{rel}: file type {suffix!r} is not allowed in the package")
    plan.files = sorted(files)
    plan.missing += [rel for rel in REQUIRED_FILES if rel not in files]
    return plan


# ------------------------------------------------------------------------------------------------ checks
def check_versions(root: Path, plan: Plan) -> None:
    paths = {"VERSION": root / "VERSION", "pyproject.toml": root / "pyproject.toml", "__init__": root / "src/falls_ml/__init__.py"}
    if not all(p.is_file() for p in paths.values()):
        return  # reported as missing
    version = paths["VERSION"].read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        plan.problems.append(f"VERSION: {version!r} is not MAJOR.MINOR.PATCH")
    py = tomllib.loads(paths["pyproject.toml"].read_text(encoding="utf-8")).get("project", {}).get("version")
    m = re.search(r"^__version__\s*=\s*[\"']([^\"']+)[\"']", paths["__init__"].read_text(encoding="utf-8"), re.MULTILINE)
    init = m.group(1) if m else None
    if not (version == py == init):
        plan.problems.append(f"version mismatch: VERSION={version!r}, pyproject.toml={py!r}, falls_ml.__version__={init!r}")
    plan.version = version


def check_line_endings(rel: str, data: bytes) -> list[str]:
    """.cmd/.bat: CRLF only and pure ASCII. Other text files: LF only (see LINE_ENDING_EXEMPT)."""
    suffix = PurePosixPath(rel).suffix.lower()
    problems = []
    if suffix in CRLF_SUFFIXES:
        if data.count(b"\n") != data.count(b"\r\n") or data.count(b"\r") != data.count(b"\r\n"):
            problems.append(f"{rel}: CMD scripts must use CRLF line endings only")
        if any(b > 127 for b in data):
            problems.append(f"{rel}: CMD scripts must be pure ASCII")
    elif suffix in TEXT_SUFFIXES and b"\r" in data and rel not in LINE_ENDING_EXEMPT:
        problems.append(f"{rel}: CR/CRLF line endings found; text files must use LF")
    return problems


def scan_text(rel: str, text: str, kinds: Iterable[str] = ("path", "credential")) -> list[Finding]:
    """Machine-specific absolute paths and credential/connection-string patterns in ``text`` (allowlist applied)."""
    wanted = set(kinds)
    out = []
    for number, line in enumerate(text.splitlines(), start=1):
        for name, (kind, rx) in _COMPILED.items():
            if kind in wanted and (m := rx.search(line)) and (rel, name) not in SCAN_ALLOWLIST:
                start = max(0, m.start() - 30)
                out.append(Finding(rel, number, name, kind, line[start:m.end() + 30].strip()))
    return out


def check_data_file(root: Path, rel: str) -> list[str]:
    """Data files only inside data/fixtures (or allowlisted); parquet only with a matching synthetic manifest."""
    p = PurePosixPath(rel)
    suffix = p.suffix.lower()
    if suffix not in DATA_SUFFIXES:
        return []
    if suffix in {".pkl", ".pickle", ".joblib"}:
        return [f"{rel}: pickled objects are never packaged"]
    in_fixture = len(p.parts) == 4 and "/".join(p.parts[:3]) in FIXTURES
    if not in_fixture:
        if rel in DATA_FILE_ALLOWLIST and suffix != ".parquet":
            if rel == "data/example_header_only.csv" and len((root / rel).read_bytes().splitlines()) != 1:
                return [f"{rel}: must contain exactly one header row and no data"]
            return []
        return [f"{rel}: data file outside data/fixtures (real or extracted data must never be packaged)"]
    if suffix != ".parquet":
        return [] if suffix == ".json" else [f"{rel}: only parquet + json files belong in a synthetic fixture"]
    manifest_path = root / p.parent / "manifest.json"
    if not manifest_path.is_file():
        return [f"{rel}: parquet without a manifest.json in the same directory"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"{p.parent}/manifest.json: invalid JSON ({exc})"]
    problems = []
    if manifest.get("source") != "synthetic_fixture":
        problems.append(f"{rel}: manifest source is {manifest.get('source')!r}, only 'synthetic_fixture' may be packaged")
    if manifest.get("scientific_use_allowed") is not False:
        problems.append(f"{rel}: manifest scientific_use_allowed must be false for packaged data")
    if manifest.get("data_file") != p.name:
        problems.append(f"{rel}: manifest data_file is {manifest.get('data_file')!r}")
    elif manifest.get("data_sha256") != sha256_file(root / rel):
        problems.append(f"{rel}: sha256 differs from manifest data_sha256")
    return problems


def check_lock_files(root: Path, plan: Plan) -> None:
    helper = root / "scripts/handoff/lockfile.py"
    locks = [n for n in ("requirements.lock", "requirements-build.lock") if (root / n).is_file()]
    if not locks:
        return
    if not helper.is_file():
        plan.warnings.append("scripts/handoff/lockfile.py missing: lock-file format not checked")
        return
    spec = importlib.util.spec_from_file_location("_handoff_lockfile", helper)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their module during class creation
    try:
        spec.loader.exec_module(module)
        for name in locks:
            text = (root / name).read_text(encoding="utf-8")
            try:
                module.parse_lock_text(text, source=name)
            except module.LockFileError as exc:
                plan.problems.append(f"{name}: {exc}")
            header = [line for line in text.splitlines() if line.startswith("#")]
            if not any("3.13" in line for line in header):
                plan.problems.append(f"{name}: header comments must state the target Python 3.13")
    finally:
        sys.modules.pop(spec.name, None)
    if (root / "requirements-lock.txt").exists():
        plan.warnings.append("requirements-lock.txt still exists (superseded by requirements.lock; it is not packaged)")


def validate(root: Path) -> Plan:
    """Collect and run every content check. Never writes."""
    plan = collect(root)
    check_versions(root, plan)
    findings: list[Finding] = []
    for rel in plan.files:
        data = (root / rel).read_bytes()
        plan.problems += check_line_endings(rel, data)
        plan.problems += check_data_file(root, rel)
        if PurePosixPath(rel).suffix.lower() in BINARY_SUFFIXES:
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            plan.problems.append(f"{rel}: not valid UTF-8 ({exc.reason} at byte {exc.start})")
            continue
        findings += scan_text(rel, text)
        if rel == "WINDOWS_HANDOFF.md" and not text.replace("\r\n", "\n").startswith(HANDOFF_START):
            plan.problems.append("WINDOWS_HANDOFF.md: must start with the five-step quick start block")
    plan.problems += [f"{f} (see SCAN_ALLOWLIST in tools/build_handoff.py for documented exceptions)" for f in findings]
    check_lock_files(root, plan)
    return plan


# ------------------------------------------------------------------------------------------------ writing
def zip_epoch() -> int:
    raw = os.environ.get("SOURCE_DATE_EPOCH")
    if raw is None or raw.strip() == "":
        # build time, not a fixed date: identical fixed timestamps across versions would let Python reuse stale
        # __pycache__ bytecode when a newer package is extracted over an older one (pyc validation uses mtime + size)
        return max(int(time.time()), ZIP_MIN_EPOCH)
    try:
        value = int(raw)
    except ValueError as exc:
        raise BuildError(f"SOURCE_DATE_EPOCH must be an integer, got {raw!r}") from exc
    return max(value, ZIP_MIN_EPOCH)


def write_zip(folder: Path, files: list[str], zip_path: Path, epoch: int) -> None:
    """Deterministic zip: sorted entries under ``PACKAGE/``, fixed timestamp and permissions, ZIP_DEFLATED."""
    stamp = time.gmtime(epoch)[:6]
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for rel in sorted(files):
            info = zipfile.ZipInfo(f"{PACKAGE}/{rel}", date_time=stamp)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = ((0o100755 if rel.endswith(".sh") else 0o100644) & 0xFFFF) << 16
            zf.writestr(info, (folder / rel).read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def _prepare_dist(dist: Path, version: str, force: bool, root: Path) -> None:
    folder, zip_path = dist / PACKAGE, dist / f"{PACKAGE}_{version}.zip"
    source = root.resolve()
    if folder.resolve() == source or folder.resolve() in source.parents:
        raise BuildError(f"refusing to build into {folder}: it contains the source project {source}; choose another --dist")
    previous = []
    if folder.exists() or folder.is_symlink():
        if folder.is_symlink() or not folder.is_dir():
            raise BuildError(f"{folder} exists and is not a directory; remove it manually")
        prev_version = (folder / "VERSION").read_text(encoding="utf-8").strip() if (folder / "VERSION").is_file() else None
        if prev_version and re.fullmatch(r"\d+\.\d+\.\d+", prev_version):
            previous.append(dist / f"{PACKAGE}_{prev_version}.zip")
    existing = [p for p in (folder, zip_path) if p.exists()]
    if existing and not force:
        raise BuildError("output already exists (use --force to replace): " + ", ".join(map(str, existing)))
    if force:
        if folder.exists():
            shutil.rmtree(folder)
            print(f"removed {folder}")
        for z in dict.fromkeys([zip_path, *previous]):
            if z.is_file():
                z.unlink()
                print(f"removed {z}")


def build(root: Path = ROOT, dist: Path | None = None, *, force: bool = False, allow_missing: bool = False,
          check_only: bool = False) -> dict[str, object]:
    """Validate, then write the folder, manifest, zip and checksums. Raises BuildError listing every problem."""
    root = root.resolve()
    dist = (dist if dist is not None else root / "dist").resolve()
    plan = validate(root)
    for w in plan.warnings:
        print(f"WARNING: {w}")
    if plan.missing:
        header = "missing required files" + (" (--allow-missing: building an INCOMPLETE package)" if allow_missing else "")
        text = header + ":\n" + "\n".join(f"  - {m}" for m in plan.missing)
        if allow_missing:
            print(f"WARNING: {text}")
        else:
            plan.problems.insert(0, text)
    if plan.version is None and not plan.problems:
        plan.problems.append("VERSION / pyproject.toml / src/falls_ml/__init__.py are required to name the package")
    if plan.problems:
        raise BuildError(f"handoff package check failed ({len(plan.problems)} problems):\n" + "\n".join(f"- {p}" for p in plan.problems))
    for top in TREES + ("data",):
        if dist == root / top or (root / top) in dist.parents:
            raise BuildError(f"--dist {dist} lies inside the packaged tree {top}/")
    summary: dict[str, object] = {"files": len(plan.files), "version": plan.version, "skipped": len(plan.skipped),
                                  "missing": list(plan.missing)}
    if check_only:
        print(f"CHECK OK: {len(plan.files)} files would be packaged ({len(plan.skipped)} cache/generated paths skipped)")
        return summary
    version = str(plan.version)
    dist.mkdir(parents=True, exist_ok=True)
    _prepare_dist(dist, version, force, root)
    folder, zip_path = dist / PACKAGE, dist / f"{PACKAGE}_{version}.zip"
    try:
        lines, total = [], 0
        for rel in plan.files:
            src, dst = root / rel, folder / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            digest = sha256_file(dst)
            if digest != sha256_file(src):
                raise BuildError(f"{rel}: copy differs from source (file changed during the build?)")
            lines.append(f"{digest}  {rel}")
            total += dst.stat().st_size
        (folder / MANIFEST_NAME).write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
        packaged = [*plan.files, MANIFEST_NAME]
        write_zip(folder, packaged, zip_path, zip_epoch())
        zip_sha = sha256_file(zip_path)
        sums = [f"{zip_sha}  {zip_path.name}"]
        for rel in SUMS_FILES:
            if (folder / rel).is_file():
                sums.append(f"{sha256_file(folder / rel)}  {PACKAGE}/{rel}")
            else:
                print(f"WARNING: {rel} not in package; omitted from {SUMS_NAME}")
        (dist / SUMS_NAME).write_bytes(("\n".join(sums) + "\n").encode("utf-8"))
    except BaseException:  # never leave a half-written package or checksums that describe it
        shutil.rmtree(folder, ignore_errors=True)
        zip_path.unlink(missing_ok=True)
        (dist / SUMS_NAME).unlink(missing_ok=True)
        raise
    summary.update({"folder": str(folder), "zip": str(zip_path), "zip_sha256": zip_sha, "zip_bytes": zip_path.stat().st_size,
                    "total_bytes": total + (folder / MANIFEST_NAME).stat().st_size, "files": len(packaged)})
    print("========================================")
    print("HANDOFF PACKAGE BUILT" + (" (INCOMPLETE: --allow-missing)" if plan.missing else ""))
    print("========================================")
    print(f"Version:      {version}")
    print(f"Files:        {summary['files']} ({summary['total_bytes'] / 1e6:.1f} MB); skipped {len(plan.skipped)} cache/generated paths")
    print(f"Folder:       {folder}")
    print(f"Zip:          {zip_path} ({summary['zip_bytes'] / 1e6:.1f} MB)")
    print(f"Zip SHA-256:  {zip_sha}")
    print(f"Checksums:    {dist / SUMS_NAME}")
    print(f"Verify on Windows: certutil -hashfile {zip_path.name} SHA256")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build dist/falls_ml_handoff and its reproducible zip.")
    ap.add_argument("--root", type=Path, default=ROOT, help="project root to package (default: this repository)")
    ap.add_argument("--dist", type=Path, help="output directory (default: <root>/dist)")
    ap.add_argument("--force", action="store_true", help="replace <dist>/falls_ml_handoff and the zip a previous build produced")
    ap.add_argument("--allow-missing", action="store_true", help="build even if required files are missing (development only)")
    ap.add_argument("--check-only", action="store_true", help="run every check and write nothing")
    a = ap.parse_args(argv)
    try:
        build(a.root, a.dist, force=a.force, allow_missing=a.allow_missing, check_only=a.check_only)
    except BuildError as exc:
        print(f"BUILD FAILED\n{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
