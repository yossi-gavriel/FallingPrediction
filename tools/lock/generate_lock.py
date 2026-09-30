"""Generate the hash-pinned dependency locks requirements.lock and requirements-build.lock (developer tool).

Runs on the development machine only (macOS arm64, CPython 3.13, internet access to the package index). It uses pip
and pip's vendored `packaging`; nothing here is needed at install time.

Procedure (see tools/lock/README.md and docs/DEPENDENCIES.md):
  1. Roots: [project].dependencies + [project.optional-dependencies].test from pyproject.toml (runtime lock) and
     pip, setuptools>=69, wheel (build lock).
  2. Pins: the versions installed in the verified environment (--pins-python, default .venv's interpreter) are kept
     unchanged. A package that is not installed there (platform-only packages such as colorama/tzdata on Windows,
     setuptools/wheel) keeps its pin from the existing lock when that still satisfies every specifier, otherwise the
     newest version with a compatible wheel is taken (--upgrade ignores existing lock pins).
  3. For each target, every pinned wheel is downloaded (pip download --only-binary=:all: --no-deps), its METADATA
     Requires-Dist is evaluated for the target environment (several patch/OS-release variants, which must agree),
     recursively, including requested extras.
       windows-amd64-cp313: --platform win_amd64 --python-version 3.13 --implementation cp --abi cp313
       windows-amd64-cp311: --platform win_amd64 --python-version 3.11 --implementation cp --abi cp311
       macos-arm64-cp313:   the running interpreter (must be CPython 3.13 on macOS arm64)
     A pinned version must therefore have wheels for CPython 3.11 AND 3.13 on Windows (the work computers run 3.11).
  4. A package needed on only one target gets '; sys_platform == "win32"' or '; sys_platform != "win32"'.
     Every requirement lists the sha256 of each downloaded wheel (all targets). Shared packages must have identical
     pins in both locks.
  5. The files are written only when their body changed; the "Generated" date is kept otherwise (idempotent).
     --check compares without writing (exit 1 when a lock is out of date).

Usage:
  .venv/bin/python tools/lock/generate_lock.py --cache-dir <dir> [--pins-python <python>] [--upgrade] [--check] [--date YYYY-MM-DD]
"""
from __future__ import annotations

import argparse
import datetime as dt
import email.parser
import json
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
import zipfile
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

from pip._vendor.packaging.markers import default_environment
from pip._vendor.packaging.requirements import Requirement as PkgRequirement
from pip._vendor.packaging.specifiers import SpecifierSet
from pip._vendor.packaging.utils import canonicalize_name
from pip._vendor.packaging.version import Version

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "handoff"))
import lockfile  # noqa: E402  (stdlib-only lock parser shared with the installer)

RUNTIME_LOCK, BUILD_LOCK = ROOT / "requirements.lock", ROOT / "requirements-build.lock"
BUILD_ROOTS = ("pip", "setuptools>=69", "wheel")
PYTHON_SERIES = "3.13"          # the development/verification interpreter (--pins-python)
SUPPORTED_SERIES = ("3.11", "3.13")  # every locked wheel must exist for these CPython series on Windows
WIN313 = "windows-amd64-cp313"
WIN311 = "windows-amd64-cp311"
MAC = "macos-arm64-cp313"
WIN = WIN313  # kept for messages about the primary Windows target


def _win_pip_args(series: str) -> tuple[str, ...]:
    return ("--platform", "win_amd64", "--python-version", series, "--implementation", "cp", "--abi", f"cp{series.replace('.', '')}")


class LockError(RuntimeError):
    """Resolution or download failure; always fatal."""


@dataclass
class Target:
    name: str
    sys_platform: str
    pip_args: tuple[str, ...]
    envs: list[dict[str, str]]
    py_minor: int = 13  # CPython 3.<py_minor> the wheels must be compatible with


@dataclass
class Node:
    name: str
    version: str
    wheel: Path
    requires: list[PkgRequirement]
    extras: set[str] = field(default_factory=set)
    required_by: set[str] = field(default_factory=set)


def _windows_envs(series: str = PYTHON_SERIES) -> list[dict[str, str]]:
    envs = []
    for patch in (f"{series}.0", f"{series}.4" if series == "3.11" else f"{series}.13"):
        for release, version in (("10", "10.0.19045"), ("11", "10.0.26100")):
            envs.append({"implementation_name": "cpython", "implementation_version": patch, "os_name": "nt", "platform_machine": "AMD64",
                         "platform_release": release, "platform_system": "Windows", "platform_version": version,
                         "python_full_version": patch, "platform_python_implementation": "CPython", "python_version": series,
                         "sys_platform": "win32"})
    return envs


def _mac_envs() -> list[dict[str, str]]:
    host = dict(default_environment())
    if not (host["sys_platform"] == "darwin" and host["platform_machine"] == "arm64" and host["implementation_name"] == "cpython"
            and host["python_version"] == PYTHON_SERIES and sys.maxsize > 2**32):
        raise LockError(f"the {MAC} target is resolved with the running interpreter, which must be 64-bit CPython {PYTHON_SERIES} "
                        f"on macOS arm64; got {host['implementation_name']} {host['python_full_version']} {host['sys_platform']} {host['platform_machine']}")
    return [host, {**host, "python_full_version": "3.13.0", "implementation_version": "3.13.0"}]


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    """Run a Python/pip child with UTF-8 on both ends of the pipe (Windows would otherwise use the ANSI code page)."""
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    if proc.returncode != 0:
        raise LockError(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}")
    return proc


def installed_pins(python: str) -> dict[str, str]:
    info = json.loads(_run([python, "-c", "import json,sys,platform; print(json.dumps([sys.implementation.name, list(sys.version_info[:3]), sys.maxsize > 2**32]))"]).stdout)
    if info[0] != "cpython" or info[1][:2] != [3, 13] or not info[2]:
        raise LockError(f"--pins-python must be 64-bit CPython {PYTHON_SERIES}; got {info}")
    rows = json.loads(_run([python, "-m", "pip", "list", "--format=json", "--exclude-editable", "--disable-pip-version-check"]).stdout)
    return {canonicalize_name(r["name"]): r["version"] for r in rows}


def pins_python_version(python: str) -> str:
    return _run([python, "-c", "import platform; print(platform.python_version())"]).stdout.strip()


def existing_pins(path: Path) -> dict[str, str]:
    return {r.name: r.version for r in lockfile.parse_lock(path)} if path.exists() else {}


def wheel_ok(wheel: lockfile.WheelName, target: Target) -> bool:
    tag = f"cp3{target.py_minor}"
    def abi3_ok(t: str) -> bool:  # a cp3XY-abi3 wheel runs on CPython 3.XY and newer
        return t.startswith("cp3") and t[2:].isdigit() and int(t[3:] or 0) <= target.py_minor
    py = any(t in ("py3", f"py3{target.py_minor}", tag) for t in wheel.python_tags) \
        or ("abi3" in wheel.abi_tags and any(abi3_ok(t) for t in wheel.python_tags))
    abi = any(t in ("none", "abi3", tag) for t in wheel.abi_tags)
    if target.sys_platform == "win32":
        plat = any(t in ("any", "win_amd64") for t in wheel.platform_tags)
    else:
        plat = any(t == "any" or (t.startswith("macosx_") and (t.endswith("_arm64") or t.endswith("_universal2"))) for t in wheel.platform_tags)
    return py and abi and plat


class Downloader:
    def __init__(self, cache: Path):
        self.cache = cache

    def _pip_download(self, spec: str, target: Target) -> Path:
        tdir = self.cache / target.name
        tdir.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix="dl-", dir=self.cache))
        try:
            _run([sys.executable, "-m", "pip", "download", "--disable-pip-version-check", "--no-input", "--no-deps", "--only-binary=:all:",
                  "--dest", str(tmp), *target.pip_args, spec])
            files = sorted(tmp.iterdir())
            if len(files) != 1 or files[0].suffix != ".whl":
                raise LockError(f"pip download {spec} for {target.name} produced {[f.name for f in files]} (expected one wheel)")
            final = tdir / files[0].name
            if final.exists():
                if lockfile.sha256_file(final) != lockfile.sha256_file(files[0]):
                    raise LockError(f"cached wheel {final} differs from a fresh download; delete the cache directory and rerun")
            else:
                shutil.move(str(files[0]), final)
            return final
        finally:
            shutil.rmtree(tmp)

    def wheel(self, name: str, version: str, target: Target) -> Path:
        tdir = self.cache / target.name
        cached = [p for p in sorted(tdir.glob("*.whl")) if (w := lockfile.parse_wheel_filename(p.name)).name == name and w.version == version] if tdir.exists() else []
        path = cached[0] if len(cached) == 1 else self._pip_download(f"{name}=={version}", target)
        self._check(path, name, version, target)
        return path

    def newest(self, name: str, specifier: SpecifierSet, target: Target) -> str:
        path = self._pip_download(f"{name}{specifier}", target)
        self._check(path, name, None, target)
        return lockfile.parse_wheel_filename(path.name).version

    @staticmethod
    def _check(path: Path, name: str, version: str | None, target: Target) -> None:
        w = lockfile.parse_wheel_filename(path.name)
        if w.name != name or (version is not None and Version(w.version) != Version(version)) or not wheel_ok(w, target):
            raise LockError(f"wheel {path.name} is not a {target.name}-compatible wheel of {name} {version or ''}")


def read_metadata(wheel: Path) -> tuple[list[PkgRequirement], str | None]:
    with zipfile.ZipFile(wheel) as zf:
        names = [n for n in zf.namelist() if n.count("/") == 1 and n.endswith(".dist-info/METADATA")]
        if len(names) != 1:
            raise LockError(f"{wheel.name}: expected one .dist-info/METADATA, found {names}")
        msg = email.parser.Parser().parsestr(zf.read(names[0]).decode("utf-8"))
    return [PkgRequirement(r) for r in msg.get_all("Requires-Dist") or []], msg.get("Requires-Python")


def _marker_true(req: PkgRequirement, extra: str, target: Target) -> bool:
    if req.marker is None:
        return True
    results = {req.marker.evaluate({**env, "extra": extra}) for env in target.envs}
    if len(results) != 1:
        raise LockError(f"marker {req.marker} evaluates differently across {target.name} environment variants; lock cannot be platform-exact")
    return results.pop()


def resolve(roots: list[str], target: Target, pins: dict[str, str], preferred: dict[str, str], dl: Downloader,
            fixed: dict[str, str]) -> dict[str, Node]:
    """Resolve the closure of `roots` for one target. `fixed` pins choices already made for another target."""
    forced: dict[str, SpecifierSet] = defaultdict(SpecifierSet)
    for _attempt in range(10):
        nodes: dict[str, Node] = {}
        constraints: dict[str, SpecifierSet] = defaultdict(SpecifierSet)
        work = deque((PkgRequirement(r), "<root>") for r in roots if _marker_true(PkgRequirement(r), "", target))   # root markers per target
        restart = False
        while work:
            req, source = work.popleft()
            name = canonicalize_name(req.name)
            constraints[name] &= req.specifier
            node = nodes.get(name)
            if node is not None:
                node.required_by.add(source)
                if not req.specifier.contains(node.version, prereleases=True):
                    if name in pins or name in fixed:
                        raise LockError(f"{target.name}: {source} requires {req}, but {name}=={node.version} is pinned")
                    forced[name] &= req.specifier
                    restart = True
                    break
                new = set(req.extras) - node.extras
                node.extras |= new
                work.extend((d, name) for d in node.requires for e in sorted(new) if _marker_true(d, e, target))
                continue
            spec = constraints[name] & forced[name]
            if name in pins or name in fixed:
                version = pins.get(name) or fixed[name]
                if not spec.contains(version, prereleases=True):
                    raise LockError(f"{target.name}: pinned {name}=={version} violates {spec} (required by {source})")
            elif name in preferred and spec.contains(preferred[name], prereleases=True):
                version = preferred[name]
            else:
                version = dl.newest(name, spec, target)
            wheel = dl.wheel(name, version, target)
            requires, requires_python = read_metadata(wheel)
            if requires_python and not all(SpecifierSet(requires_python).contains(e["python_full_version"]) for e in target.envs):
                raise LockError(f"{name}=={version} Requires-Python {requires_python} excludes {target.name}")
            node = nodes[name] = Node(name, version, wheel, requires, set(req.extras), {source})
            work.extend((d, name) for d in requires for e in ["", *sorted(node.extras)] if _marker_true(d, e, target))
        if not restart:
            return nodes
    raise LockError(f"{target.name}: resolution did not converge (forced constraints {dict(forced)})")


@dataclass
class Locked:
    name: str
    version: str
    hashes: list[str]
    marker: str | None
    required_by: dict[str, list[str]]


def combine(per_target: dict[str, dict[str, Node]]) -> list[Locked]:
    names = sorted(set().union(*per_target.values()))
    out = []
    for name in names:
        present = {t: nodes[name] for t, nodes in per_target.items() if name in nodes}
        versions = {n.version for n in present.values()}
        if len(versions) != 1:
            raise LockError(f"{name}: different versions across targets {versions}")
        windows = {t for t in present if t.startswith("windows-")}
        if len(present) == len(per_target):
            marker = None
        elif set(present) == windows:
            marker = lockfile.MARKER_WIN32
        elif not windows:
            marker = lockfile.MARKER_NOT_WIN32
        else:
            raise LockError(f"{name}: present on {sorted(present)} only; no marker covers that combination")
        hashes = sorted({lockfile.sha256_file(n.wheel) for n in present.values()})
        out.append(Locked(name, versions.pop(), hashes, marker, {t: sorted(n.required_by) for t, n in present.items()}))
    return out


def render_body(locked: list[Locked], root_label: str) -> str:
    lines = []
    for item in locked:
        marker = f" ; {item.marker}" if item.marker else ""
        lines.append(f"{item.name}=={item.version}{marker} \\")
        lines.extend(f"    --hash=sha256:{h}" + (" \\" if i < len(item.hashes) - 1 else "") for i, h in enumerate(item.hashes))
        via = sorted({s for srcs in item.required_by.values() for s in srcs})
        lines.append("    # via " + ", ".join(root_label if s == "<root>" else s for s in via))
    return "\n".join(lines) + "\n"


def render_header(kind: str, tested: str, date: str) -> str:
    what = ("runtime and test dependencies of falls_ml ([project].dependencies + [project.optional-dependencies].test)"
            if kind == "runtime" else "build tools for the editable install of falls_ml (pip, setuptools, wheel and their dependencies)")
    return "\n".join([
        f"# Hash-pinned lock: {what}.",
        f"# Python: CPython {' or '.join(SUPPORTED_SERIES)}, 64-bit (development and verification interpreter: {tested}).",
        f"# Target platforms: {WIN313} and {WIN311} (Windows 10/11 x64, win_amd64), {MAC} (macOS arm64, CPython {PYTHON_SERIES}).",
        "# Markers: only '; sys_platform == \"win32\"' and '; sys_platform != \"win32\"' are used.",
        f"# Generated: {date}",
        "# Generated by: python tools/lock/generate_lock.py",
        "# Do not edit by hand; regeneration and hash verification are described in docs/DEPENDENCIES.md.",
        "",
    ]) + "\n"


def split_existing(path: Path) -> tuple[str | None, str | None]:
    """Return (generated date, body) of an existing lock file."""
    if not path.exists():
        return None, None
    text = path.read_text(encoding="utf-8")
    date = next((ln.split(":", 1)[1].strip() for ln in text.splitlines() if ln.startswith("# Generated:")), None)
    head, sep, body = text.partition("\n\n")
    return date, body if sep else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache-dir", type=Path, default=Path(tempfile.gettempdir()) / "falls_ml_lock_wheel_cache",
                    help="wheel download cache (one subdirectory per target)")
    ap.add_argument("--pins-python", default=None, help="interpreter whose installed versions are kept (default: .venv)")
    ap.add_argument("--upgrade", action="store_true", help="ignore pins of existing locks for packages not installed in --pins-python")
    ap.add_argument("--check", action="store_true", help="do not write; exit 1 when a lock file is out of date")
    ap.add_argument("--date", default=None, help="Generated date for changed files (default: today, UTC)")
    args = ap.parse_args(argv)

    venv_python = ROOT / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    pins_python = args.pins_python or (str(venv_python) if venv_python.exists() else sys.executable)
    pins = installed_pins(pins_python)
    tested = pins_python_version(pins_python)
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    runtime_roots = list(pyproject["dependencies"]) + list(pyproject.get("optional-dependencies", {}).get("test", []))
    targets = [Target(WIN313, "win32", _win_pip_args("3.13"), _windows_envs("3.13"), 13),
               Target(WIN311, "win32", _win_pip_args("3.11"), _windows_envs("3.11"), 11),
               Target(MAC, "darwin", (), _mac_envs(), 13)]
    dl = Downloader(args.cache_dir.resolve())

    results: dict[str, list[Locked]] = {}
    for kind, roots in (("runtime", runtime_roots), ("build", list(BUILD_ROOTS))):
        preferred = {} if args.upgrade else {**existing_pins(RUNTIME_LOCK), **existing_pins(BUILD_LOCK)}
        for other in results.values():  # shared packages must carry the same pin in both locks
            preferred.update({item.name: item.version for item in other})
        per_target: dict[str, dict[str, Node]] = {}
        fixed: dict[str, str] = {}
        for target in targets:
            per_target[target.name] = resolve(roots, target, pins, preferred, dl, fixed)
            fixed.update({n: node.version for n, node in per_target[target.name].items() if n not in pins})
        results[kind] = combine(per_target)
    runtime_names = {i.name: i for i in results["runtime"]}
    for item in results["build"]:
        other = runtime_names.get(item.name)
        if other is not None and (other.version, other.hashes, other.marker) != (item.version, item.hashes, item.marker):
            raise LockError(f"{item.name} differs between requirements.lock and requirements-build.lock")

    locked_names = {i.name for items in results.values() for i in items}
    unlocked = sorted(set(pins) - locked_names)
    stale = 0
    today = args.date or dt.datetime.now(dt.timezone.utc).date().isoformat()
    for kind, path in (("runtime", RUNTIME_LOCK), ("build", BUILD_LOCK)):
        body = render_body(results[kind], "pyproject.toml" if kind == "runtime" else "build tools")
        old_date, old_body = split_existing(path)
        changed = body != old_body
        text = render_header(kind, tested, today if changed or old_date is None else old_date) + body
        parsed = lockfile.parse_lock_text(text, source=path.name)  # the written file must satisfy the installer's parser
        assert [r.name for r in parsed] == [i.name for i in results[kind]]
        if args.check:
            stale += int(changed)
            print(f"{path.name}: {'OUT OF DATE' if changed else 'up to date'}")
        elif changed or path.read_text(encoding="utf-8") != text:
            path.write_bytes(text.encode("utf-8"))
            print(f"wrote {path.name} ({len(parsed)} requirements)")
        else:
            print(f"{path.name} unchanged ({len(parsed)} requirements)")
    print("\npackage | version | marker | required by (per target) | sha256 count")
    for kind in ("runtime", "build"):
        for item in results[kind]:
            via = "; ".join(f"{t}: {', '.join(('pyproject.toml' if kind == 'runtime' else 'build tools') if s == '<root>' else s for s in srcs)}" for t, srcs in item.required_by.items())
            print(f"[{kind}] {item.name} | {item.version} | {item.marker or '-'} | {via} | {len(item.hashes)}")
    if unlocked:
        print(f"\nnote: installed in {pins_python} but not required by any lock root: {', '.join(unlocked)}")
    return 1 if stale else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except LockError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
