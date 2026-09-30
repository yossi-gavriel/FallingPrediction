"""Run directories and reproducibility metadata (architecture §3.4)."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

TRACKED_PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "statsmodels", "matplotlib", "PyYAML", "pyarrow")


def package_root() -> Path:
    return Path(__file__).resolve().parent


def source_tree_sha256(root: Path | None = None) -> str:
    """SHA-256 over the relative POSIX paths + contents of all ``*.py`` files in the package, identical on every OS.

    Files are ordered by case-sensitive path components (the POSIX ``sorted(Path)`` order, which Windows would make
    case-insensitive), names are hashed with ``/`` separators and CRLF line endings are hashed as LF, so a Windows copy or a
    ``core.autocrlf`` checkout of the same code gives the same digest as macOS/Linux.
    """
    root = root or package_root()
    h = hashlib.sha256()
    for p in sorted(root.rglob("*.py"), key=lambda q: q.relative_to(root).parts):
        if "__pycache__" in p.parts:
            continue
        h.update(p.relative_to(root).as_posix().encode("utf-8"))
        h.update(p.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def definition_file_sha256(path: Path) -> str:
    """SHA-256 of a text definition file (feature spec, extension, published equation YAML) with CRLF hashed as LF, so a
    Windows editor re-save or a ``core.autocrlf`` checkout of unchanged content keeps the hash. Identical to the raw-bytes
    SHA-256 for LF files."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str] | None:
    """Run git with UTF-8 decoding (git output is UTF-8, never the Windows ANSI code page); None if git is unusable."""
    try:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                              stdin=subprocess.DEVNULL, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def git_commit(cwd: Path | None = None) -> str | None:
    """HEAD commit (``-dirty`` when the work tree has changes) of the repository whose top level is ``cwd``.

    Default ``cwd`` is the project root of the installed source tree (``src/falls_ml`` -> project). A commit is reported only
    when git's top level IS that directory, so a handoff folder copied inside an unrelated repository never records that
    repository's commit.
    """
    root = Path(cwd) if cwd is not None else package_root().parents[1]
    if not root.is_dir():
        return None
    top = _git(["rev-parse", "--show-toplevel"], root)
    if top is None or top.returncode != 0 or not top.stdout.strip():
        return None
    if Path(top.stdout.strip()).resolve() != root.resolve():
        return None
    out = _git(["rev-parse", "HEAD"], root)
    if out is None or out.returncode != 0:
        return None
    dirty = _git(["status", "--porcelain"], root)
    return out.stdout.strip() + ("-dirty" if dirty is None or dirty.returncode != 0 or dirty.stdout.strip() else "")


def code_version() -> dict[str, str | None]:
    return {"git_commit": git_commit(), "source_tree_sha256": source_tree_sha256()}


def environment_info() -> dict[str, Any]:
    versions = {}
    for pkg in TRACKED_PACKAGES:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = None
    return {"python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
            "packages": versions, "code_version": code_version()}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, float) and not np.isfinite(o):
        return None
    raise TypeError(f"not JSON serialisable: {type(o)}")


REPLACE_ATTEMPTS = 10


def replace_file(src: Path, dst: Path, *, windows: bool | None = None, attempts: int = REPLACE_ATTEMPTS,
                 sleep: Callable[[float], None] = time.sleep) -> None:
    """``os.replace`` that tolerates transient Windows sharing violations.

    On Windows, antivirus scanners, the search indexer, OneDrive or a viewer holding ``dst``/``src`` open make ``os.replace``
    fail with ``PermissionError`` (WinError 5/32) for a moment; retry with a short back-off (0.05 s doubling to 0.5 s) and
    re-raise with both paths when it persists. Elsewhere a single ``os.replace`` (errors propagate unchanged).
    """
    windows = sys.platform == "win32" if windows is None else windows
    delay = 0.05
    for attempt in range(1, (attempts if windows else 1) + 1):
        try:
            os.replace(src, dst)
            return
        except PermissionError as exc:
            if not windows:
                raise
            if attempt == attempts:
                raise PermissionError(exc.errno, f"could not replace {dst} with {src} after {attempts} attempts (file in use by "
                                                 f"another process, e.g. antivirus, OneDrive or an open viewer): {exc.strerror}") from exc
            sleep(delay)
            delay = min(delay * 2, 0.5)


def write_json(path: Path, obj: Any) -> Path:
    """Write JSON atomically (temp file + rename): readers never see a half-written file. UTF-8, LF line endings on every OS."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_sanitize(obj), indent=2, default=_json_default, allow_nan=False)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    try:
        replace_file(tmp, path)
    except OSError:
        tmp.unlink(missing_ok=True)
        raise
    return path


def _sanitize(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize(v) for v in o]
    if isinstance(o, float) and not np.isfinite(o):
        return None
    if isinstance(o, np.floating) and not np.isfinite(o):
        return None
    return o


def write_yaml(path: Path, obj: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(obj, sort_keys=False, allow_unicode=True), encoding="utf-8", newline="\n")
    return path


def write_csv(path: Path, df: pd.DataFrame) -> Path:
    """UTF-8 CSV with LF line endings on every OS (pandas would otherwise use ``os.linesep``, CRLF on Windows)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    return path


def write_parquet(path: Path, df: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


@dataclass(frozen=True)
class RunDirectory:
    path: Path
    run_id: str
    created_utc: str

    @property
    def plots(self) -> Path:
        return self.path / "plots"

    @property
    def model(self) -> Path:
        return self.path / "model"

    def file(self, name: str) -> Path:
        return self.path / name


RUN_LABEL_MAX_CHARS = 60


def run_label(experiment_name: str, model_name: str) -> str:
    """Name part of a run id: ``<experiment>_<model>``, or just ``<experiment>`` when it already contains the model name
    (model overrides ``<name>__<model>``, ablation steps). Capped at :data:`RUN_LABEL_MAX_CHARS` characters (the hash token keeps
    ids unique) so run directories stay far below the Windows 260-character path limit."""
    label = experiment_name if model_name and model_name in experiment_name else f"{experiment_name}_{model_name}"
    if len(label) > RUN_LABEL_MAX_CHARS:
        label = label[:RUN_LABEL_MAX_CHARS].rstrip("_.-")
    return label


def create_run_directory(runs_dir: Path, experiment_name: str, model_name: str, *, config_sha256: str,
                         dataset_sha256: str, created_utc: str | None = None) -> RunDirectory:
    created = created_utc or utc_now()
    # An explicit timestamp gives a deterministic id (tests, fixtures); otherwise a nanosecond clock reading keeps
    # repeated runs of one config within the same second distinct.
    uniquifier = "" if created_utc else f"|{time.time_ns()}"
    token = hashlib.sha256(f"{config_sha256}|{dataset_sha256}|{source_tree_sha256()}|{created}{uniquifier}".encode()).hexdigest()[:8]
    date = created[:10]
    run_id = f"{date}_{run_label(experiment_name, model_name)}_{token}"
    path = Path(runs_dir) / run_id
    if path.exists():
        raise FileExistsError(f"Run directory already exists: {path}")
    (path / "plots").mkdir(parents=True)
    (path / "model").mkdir()
    return RunDirectory(path=path, run_id=run_id, created_utc=created)
