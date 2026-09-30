"""Durable file operations for a run that must survive power loss (planning/ARCHITECTURE_PHASE2.md §5, reviews/REVIEW_DECISIONS.md G-01..G-09).

Every write goes temp file -> flush -> fsync (FlushFileBuffers on Windows) -> close -> atomic replace -> fsync of the parent directory (POSIX;
Windows has no directory fsync, the commit-record marker written last covers it). Renames and replaces retry transient Windows sharing
violations (antivirus, indexer, OneDrive). Nothing here deletes a file that holds results.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import pickle
import stat
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from falls_ml.artifacts import _json_default, _sanitize, replace_file

RENAME_ATTEMPTS = 10


def fsync_dir(path: Path) -> None:
    """Persist a directory entry change (POSIX). Windows cannot open a directory for fsync; there the marker protocol covers it."""
    if sys.platform == "win32":
        return
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _tmp(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp-{os.getpid()}")


def write_bytes(path: Path, data: bytes) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp(path)
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    if path.exists() and not os.access(path, os.W_OK):
        # an output left read-only by an interrupted stage finalisation (no stage record yet): Windows cannot replace a read-only file.
        # Committed outputs are never rewritten (a stage with a verified record is skipped), so clearing the bit here is safe.
        os.chmod(path, stat.S_IREAD | stat.S_IWRITE)
    try:
        replace_file(tmp, path)
    except OSError:
        tmp.unlink(missing_ok=True)
        raise
    fsync_dir(path.parent)
    return path


def write_str(path: Path, text: str) -> Path:
    return write_bytes(path, text.encode("utf-8"))


def json_text(obj: Any) -> str:
    return json.dumps(_sanitize(obj), indent=2, default=_json_default, allow_nan=False, ensure_ascii=False) + "\n"


def write_json(path: Path, obj: Any) -> Path:
    return write_bytes(path, json_text(obj).encode("utf-8"))


def write_csv(path: Path, df: pd.DataFrame) -> Path:
    buf = io.StringIO()
    df.to_csv(buf, index=False, lineterminator="\n")
    return write_bytes(path, buf.getvalue().encode("utf-8"))


def write_parquet(path: Path, df: pd.DataFrame) -> Path:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return write_bytes(path, buf.getvalue())


def write_npz(path: Path, **arrays: np.ndarray) -> Path:
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return write_bytes(path, buf.getvalue())


def write_pickle(path: Path, obj: Any) -> Path:
    """Local checkpoint of fitted objects (never shared; loaded only by the same code version, checked by the plan/code hash)."""
    return write_bytes(path, pickle.dumps(obj, protocol=5))


def read_pickle(path: Path) -> Any:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rename_dir(src: Path, dst: Path, *, attempts: int = RENAME_ATTEMPTS, sleep: Callable[[float], None] = time.sleep) -> None:
    """Rename a directory (the destination must not exist); retries Windows sharing violations with back-off."""
    if dst.exists():
        raise FileExistsError(f"refusing to rename {src} onto an existing {dst}")
    delay = 0.05
    for attempt in range(1, attempts + 1):
        try:
            os.rename(src, dst)
            fsync_dir(dst.parent)
            return
        except PermissionError:
            if attempt == attempts:
                raise
            sleep(delay)
            delay = min(delay * 2, 0.5)


def mark_readonly(path: Path) -> None:
    """Read-only attribute on a completed artifact (guards against an accidental save from Excel; G-02c)."""
    try:
        os.chmod(path, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
    except OSError:
        pass


def append_jsonl(path: Path, obj: Any) -> None:
    """Append one JSON line and fsync (an interrupted append leaves at most one torn last line; see :func:`repair_jsonl`)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(_sanitize(obj), default=_json_default, allow_nan=False, ensure_ascii=False) + "\n"
    with open(path, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(line)
        fh.flush()
        os.fsync(fh.fileno())


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], str | None]:
    """Records of a JSONL file and its torn tail (text after the last parseable line), if any."""
    if not path.exists():
        return [], None
    raw = path.read_bytes().decode("utf-8", errors="replace")
    records: list[dict[str, Any]] = []
    good_end = 0
    pos = 0
    for line in raw.splitlines(keepends=True):
        pos += len(line)
        if not line.endswith("\n"):
            break
        try:
            records.append(json.loads(line))
            good_end = pos
        except json.JSONDecodeError:
            break
    tail = raw[good_end:]
    return records, (tail if tail.strip() else None)


def repair_jsonl(path: Path, sidecar: Path) -> str | None:
    """Move a torn tail (G-09-4) to ``sidecar`` (kept for audit) and rewrite the file with its valid lines. Returns the tail or None."""
    records, tail = read_jsonl(path)
    if tail is None:
        return None
    write_str(sidecar, tail)
    write_str(path, "".join(json.dumps(r, ensure_ascii=False, default=_json_default) + "\n" for r in records))
    return tail


SYNC_MARKERS = ("onedrive", "dropbox", "google drive", "googledrive", "icloud", "box sync")


def synced_folder(path: Path) -> str | None:
    """The sync client that seems to own ``path`` (OneDrive and similar rewrite / lock files while a run writes them; G-02a)."""
    resolved = str(Path(path).resolve()).lower()
    for var in ("OneDrive", "OneDriveCommercial", "OneDriveConsumer"):
        root = os.environ.get(var)
        if root and resolved.startswith(str(Path(root).resolve()).lower()):
            return f"{var} ({root})"
    for part in Path(resolved).parts:
        for m in SYNC_MARKERS:
            if m in part:
                return f"path component {part!r}"
    return None
