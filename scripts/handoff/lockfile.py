"""Stdlib-only reader for the hash-pinned lock files requirements.lock and requirements-build.lock.

It runs before any dependency is installed, so it imports only the standard library (Python 3.9+ syntax).

Lock format (pip requirements syntax, deliberately restricted). One logical requirement per line; a trailing
backslash continues the line. Full-line comments start with '#'; blank lines are ignored. Accepted:
    <name>==<version>[ ; sys_platform == "win32" | ; sys_platform != "win32"] --hash=sha256:<64 hex> [--hash=...]
Everything else is rejected: options (-r, --index-url, ...), URLs, extras, other markers, other hash algorithms,
unpinned or unhashed requirements, duplicate names or hashes, a continuation that runs into a comment or EOF.

API
    Requirement(name, version, hashes, marker)
        frozen dataclass; name is PEP 503 canonical ("pyyaml", "typing-extensions"); version is the pinned string;
        hashes is a sorted tuple of lowercase sha256 hex digests (without the "sha256:" prefix);
        marker is None, MARKER_WIN32 ('sys_platform == "win32"') or MARKER_NOT_WIN32 ('sys_platform != "win32"').
    LockFileError(ValueError)                        any format violation; the message names the file and line.
    parse_lock(path) -> list[Requirement]            parse a lock file (UTF-8, LF or CRLF); file order preserved.
    parse_lock_text(text, source) -> list[Requirement]
    applies(req, platform=sys.platform) -> bool      evaluate the marker for a sys.platform value ("win32", "darwin", ...).
    write_evaluated_requirements(path, dest, sys_platform) -> list[Requirement]
        write a marker-free requirements file holding only the requirements of lock `path` that apply to
        `sys_platform` (hashes unchanged, one line each); returns those requirements. Needed for cross-platform
        `pip download --platform ...`, because pip evaluates markers against the host, not the target.
    sha256_file(path) -> str                         lowercase hex sha256 of the file's bytes.
    canonical_name(name) -> str                      PEP 503 normalisation.
    parse_wheel_filename(filename) -> WheelName      (name, version, build, python_tags, abi_tags, platform_tags).
    format_requirement(req, with_marker=True) -> str single-line pip requirement text.
"""
from __future__ import annotations

import hashlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

MARKER_WIN32 = 'sys_platform == "win32"'
MARKER_NOT_WIN32 = 'sys_platform != "win32"'
ALLOWED_MARKERS = (MARKER_WIN32, MARKER_NOT_WIN32)

_NAME = r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?"
_VERSION = r"[0-9][0-9A-Za-z.+!]*"
_LINE_RE = re.compile(
    r"^(?P<name>" + _NAME + r")==(?P<version>" + _VERSION + r")"
    r"(?:\s*;\s*(?P<marker>sys_platform (?:==|!=) \"win32\"))?"
    r"(?P<hashes>(?:\s+--hash=sha256:[0-9a-f]{64})+)\s*$")
_HASH_RE = re.compile(r"--hash=sha256:([0-9a-f]{64})")


class LockFileError(ValueError):
    """A lock file violates the restricted hash-pinned format."""


@dataclass(frozen=True)
class Requirement:
    name: str
    version: str
    hashes: Tuple[str, ...]
    marker: Optional[str] = None


@dataclass(frozen=True)
class WheelName:
    name: str
    version: str
    build: Optional[str]
    python_tags: Tuple[str, ...]
    abi_tags: Tuple[str, ...]
    platform_tags: Tuple[str, ...]


def canonical_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _logical_lines(text: str, source: str) -> List[Tuple[int, str]]:
    out: List[Tuple[int, str]] = []
    buf: List[str] = []
    start = 0
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.rstrip()
        stripped = line.strip()
        if buf and (not stripped or stripped.startswith("#")):
            raise LockFileError("%s:%d: line continuation from line %d runs into a blank or comment line" % (source, number, start))
        if not buf and (not stripped or stripped.startswith("#")):
            continue
        if not buf:
            start = number
        if line.endswith("\\"):
            buf.append(line[:-1].strip())
            continue
        buf.append(stripped)
        out.append((start, " ".join(part for part in buf if part)))
        buf = []
    if buf:
        raise LockFileError("%s:%d: line continuation runs into end of file" % (source, start))
    return out


def parse_lock_text(text: str, source: str = "<text>") -> List[Requirement]:
    if text.startswith("﻿"):
        raise LockFileError("%s: byte-order mark not allowed" % source)
    reqs: List[Requirement] = []
    seen = {}
    for number, line in _logical_lines(text, source):
        match = _LINE_RE.match(line)
        if match is None:
            raise LockFileError("%s:%d: not a hash-pinned requirement of the form "
                                "'name==version [; sys_platform ==|!= \"win32\"] --hash=sha256:<hex> ...': %r" % (source, number, line))
        hashes = _HASH_RE.findall(match.group("hashes"))
        if len(set(hashes)) != len(hashes):
            raise LockFileError("%s:%d: duplicate hash for %s" % (source, number, match.group("name")))
        name = canonical_name(match.group("name"))
        if name in seen:
            raise LockFileError("%s:%d: duplicate requirement %s (first on line %d)" % (source, number, name, seen[name]))
        seen[name] = number
        reqs.append(Requirement(name=name, version=match.group("version"), hashes=tuple(sorted(hashes)), marker=match.group("marker")))
    if not reqs:
        raise LockFileError("%s: no requirements found" % source)
    return reqs


def parse_lock(path) -> List[Requirement]:
    path = Path(path)
    try:
        text = path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise LockFileError("%s: not valid UTF-8 (%s)" % (path, exc)) from exc
    return parse_lock_text(text, source=str(path))


def applies(req: Requirement, platform: str = sys.platform) -> bool:
    if req.marker is None:
        return True
    if req.marker == MARKER_WIN32:
        return platform == "win32"
    if req.marker == MARKER_NOT_WIN32:
        return platform != "win32"
    raise LockFileError("unsupported marker %r on %s" % (req.marker, req.name))


def format_requirement(req: Requirement, with_marker: bool = True) -> str:
    marker = " ; %s" % req.marker if (with_marker and req.marker) else ""
    return "%s==%s%s %s" % (req.name, req.version, marker, " ".join("--hash=sha256:" + h for h in req.hashes))


def write_evaluated_requirements(path, dest, sys_platform: str) -> List[Requirement]:
    reqs = [r for r in parse_lock(path) if applies(r, sys_platform)]
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Marker-evaluated copy of %s for sys_platform=%s (scripts/handoff/lockfile.py). Do not edit." % (Path(path).name, sys_platform)]
    lines += [format_requirement(r, with_marker=False) for r in reqs]
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return reqs


def parse_wheel_filename(filename: str) -> WheelName:
    base = Path(filename).name
    if not base.endswith(".whl"):
        raise ValueError("not a wheel filename: %r" % base)
    parts = base[:-4].split("-")
    if len(parts) not in (5, 6) or not all(parts):
        raise ValueError("malformed wheel filename: %r" % base)
    build = parts[2] if len(parts) == 6 else None
    return WheelName(name=canonical_name(parts[0]), version=parts[1], build=build, python_tags=tuple(parts[-3].split(".")),
                     abi_tags=tuple(parts[-2].split(".")), platform_tags=tuple(parts[-1].split(".")))
