"""Build (or verify) the offline wheel bundle for setup_windows.cmd --offline. Standard library only.

Usage:
  python scripts/handoff/prepare_offline.py [--target windows-amd64-cp313|windows-amd64-cp311|current] [--dest offline_packages] [--force]
  python scripts/handoff/prepare_offline.py --verify-only [--dest offline_packages]

Run it on any computer with internet access (Windows, macOS or Linux; any CPython 3.9+ with pip for the Windows
target, CPython 3.13 64-bit on a lock target platform for --target current). It downloads exactly the wheels pinned in
requirements-build.lock and requirements.lock with the running interpreter's pip (pip honours PIP_INDEX_URL,
HTTPS_PROXY and pip.ini/pip.conf):
  windows-amd64-cp313: pip download --only-binary=:all: --no-deps --require-hashes --platform win_amd64
                       --python-version 3.13 --implementation cp --abi cp313 -r <marker-evaluated lock copy>
  windows-amd64-cp311: the same with --python-version 3.11 --abi cp311 (work computers running CPython 3.11)
                       (pip evaluates markers against the host, so the lock is first evaluated for sys_platform=win32)
  current:             the same for the running interpreter's own platform.
Child pip processes run with UTF-8 I/O and without PIP_USER / PIP_TARGET / PIP_PREFIX (index, proxy and pip.ini settings
are kept). Every downloaded file is checked against the lock hashes, then OFFLINE_MANIFEST.json is written last:
  {"format": 1, "created_utc", "target": {"name", "platform_tags", "python_version", "implementation", "abi"},
   "requirements_lock_sha256", "build_lock_sha256", "files": {"<wheel filename>": "<sha256>"}}
A non-empty --dest is refused; --force replaces a previous bundle but removes only the files listed in its manifest
(anything else in the folder is an error). --verify-only re-checks an existing bundle against the locks.

Functions used by other helpers: verify_bundle(dest, runtime_lock, build_lock, sys_platform) -> list[str] (problems).
Exit codes: 0 success, 1 failure, 2 usage error.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lockfile  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_NAME = "OFFLINE_MANIFEST.json"
STAGING_NAME = ".partial_download"
WINDOWS_TARGETS = {"windows-amd64-cp313": "3.13", "windows-amd64-cp311": "3.11"}
WINDOWS_TARGET = "windows-amd64-cp313"  # default (the work computer's series is chosen with --target)
SUPPORTED_SERIES = ("3.11", "3.13")


def _win_pip_args(series: str) -> list:
    return ["--platform", "win_amd64", "--python-version", series, "--implementation", "cp", "--abi", "cp" + series.replace(".", "")]
#: pip settings that redirect or break an install target; cleared for child pip processes (index, proxy and pip.ini settings stay)
PIP_TARGET_SETTINGS = ("PIP_USER", "PIP_TARGET", "PIP_PREFIX")


class OfflineError(RuntimeError):
    """Fatal problem preparing or verifying the offline bundle."""


def child_env() -> Dict[str, str]:
    """Environment for child pip processes: UTF-8 I/O (never the Windows ANSI code page), no PYTHONPATH/PYTHONHOME leak,
    no install-target overrides. PIP_INDEX_URL, PIP_EXTRA_INDEX_URL, proxies and pip.ini/pip.conf are inherited unchanged."""
    env = dict(os.environ)
    for name in ("PYTHONHOME", "PYTHONPATH") + PIP_TARGET_SETTINGS:
        env.pop(name, None)
    env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PIP_DISABLE_PIP_VERSION_CHECK="1", PIP_NO_INPUT="1")
    return env


def _host_platform_tag() -> str:
    return sysconfig.get_platform().replace("-", "_").replace(".", "_")


def target_spec(name: str) -> dict:
    """Manifest target block, marker platform and pip download flags for a target name."""
    if name in WINDOWS_TARGETS:
        series = WINDOWS_TARGETS[name]
        return {"manifest": {"name": name, "platform_tags": ["win_amd64"], "python_version": series, "implementation": "cp",
                             "abi": "cp" + series.replace(".", "")},
                "sys_platform": "win32", "pip_args": _win_pip_args(series)}
    if name == "current":
        machine = platform.machine().lower()
        supported = (sys.platform == "win32" and machine in ("amd64", "x86_64")) or (sys.platform == "darwin" and machine == "arm64")
        series = "%d.%d" % sys.version_info[:2]
        if sys.implementation.name != "cpython" or series not in SUPPORTED_SERIES or sys.maxsize <= 2**32 or not supported:
            raise OfflineError("--target current needs 64-bit CPython 3.11 or 3.13 on Windows x64 or macOS arm64 (the lock holds hashes only for "
                               "those platforms); running %s %s on %s/%s. Use --target %s from any computer instead."
                               % (sys.implementation.name, platform.python_version(), sys.platform, platform.machine(), WINDOWS_TARGET))
        return {"manifest": {"name": "current", "platform_tags": [_host_platform_tag()], "python_version": series, "implementation": "cp",
                             "abi": "cp" + series.replace(".", "")},
                "sys_platform": sys.platform, "pip_args": []}
    raise OfflineError("unknown target %r (expected one of %s, or current)" % (name, ", ".join(sorted(WINDOWS_TARGETS))))


def manifest_sys_platform(target: dict) -> str:
    """sys.platform a bundle is valid for, derived from its manifest target block."""
    if target.get("name") in WINDOWS_TARGETS:
        return "win32"
    tags = target.get("platform_tags") or []
    if any(t.startswith("win") for t in tags):
        return "win32"
    if any(t.startswith("macosx") for t in tags):
        return "darwin"
    raise OfflineError("manifest target %r does not name a supported platform" % (target,))


def applicable_requirements(runtime_lock: Path, build_lock: Path, sys_platform: str) -> Dict[str, List[lockfile.Requirement]]:
    """Canonical name -> applicable requirement(s) from both locks; shared names must pin the same version."""
    out: Dict[str, List[lockfile.Requirement]] = {}
    for path in (build_lock, runtime_lock):
        for req in lockfile.parse_lock(path):
            if not lockfile.applies(req, sys_platform):
                continue
            prior = out.setdefault(req.name, [])
            if prior and prior[0].version != req.version:
                raise OfflineError("%s is pinned to %s and %s in the two lock files" % (req.name, prior[0].version, req.version))
            prior.append(req)
    return out


def wheel_compatible(wheel: lockfile.WheelName, target_name: str) -> bool:
    if target_name != WINDOWS_TARGET:
        return True  # pip selected it for the running interpreter; the lock hashes are the authority
    py = any(t in ("py3", "py313", "cp313") for t in wheel.python_tags) or ("abi3" in wheel.abi_tags and any(t.startswith("cp3") for t in wheel.python_tags))
    return py and any(t in ("none", "abi3", "cp313") for t in wheel.abi_tags) and any(t in ("any", "win_amd64") for t in wheel.platform_tags)


def check_wheels(folder: Path, reqs: Dict[str, List[lockfile.Requirement]], target_name: str, ignore=(MANIFEST_NAME,)) -> Dict[str, str]:
    """Verify `folder` holds exactly one lock-matching wheel per requirement; return {filename: sha256}."""
    files: Dict[str, str] = {}
    seen: Dict[str, str] = {}
    problems: List[str] = []
    for entry in sorted(folder.iterdir()):
        if entry.name in ignore:
            continue
        if not entry.is_file() or entry.suffix != ".whl":
            problems.append("unexpected entry %s" % entry.name)
            continue
        try:
            wheel = lockfile.parse_wheel_filename(entry.name)
        except ValueError as exc:
            problems.append(str(exc))
            continue
        digest = lockfile.sha256_file(entry)
        pinned = reqs.get(wheel.name)
        if not pinned:
            problems.append("%s is not an applicable requirement of the locks" % entry.name)
        elif wheel.version.lower() != pinned[0].version.lower():
            problems.append("%s: version %s but the lock pins %s" % (entry.name, wheel.version, pinned[0].version))
        elif not all(digest in r.hashes for r in pinned):
            problems.append("%s: sha256 %s is not listed in the lock (corrupted download or different index file)" % (entry.name, digest))
        elif not wheel_compatible(wheel, target_name):
            problems.append("%s is not compatible with %s" % (entry.name, target_name))
        elif wheel.name in seen:
            problems.append("two wheels for %s: %s and %s" % (wheel.name, seen[wheel.name], entry.name))
        else:
            seen[wheel.name] = entry.name
            files[entry.name] = digest
    problems += ["missing wheel for %s==%s" % (name, rs[0].version) for name, rs in sorted(reqs.items()) if name not in seen]
    if problems:
        raise OfflineError("offline bundle check failed:\n  " + "\n  ".join(problems))
    return files


def verify_bundle(dest: Path, runtime_lock: Path = ROOT / "requirements.lock", build_lock: Path = ROOT / "requirements-build.lock",
                  sys_platform: Optional[str] = None) -> List[str]:
    """Problems with an existing bundle (empty list = valid for the locks and for `sys_platform`, default the host)."""
    dest = Path(dest)
    manifest_path = dest / MANIFEST_NAME
    if not manifest_path.is_file():
        return ["%s not found" % manifest_path]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return ["%s is unreadable: %s" % (manifest_path, exc)]
    problems: List[str] = []
    if manifest.get("format") != 1 or not isinstance(manifest.get("files"), dict) or not isinstance(manifest.get("target"), dict):
        return ["%s has an unsupported format" % manifest_path]
    if manifest.get("requirements_lock_sha256") != lockfile.sha256_file(runtime_lock):
        problems.append("bundle was prepared for a different requirements.lock (sha256 mismatch); rerun prepare_offline_package")
    if manifest.get("build_lock_sha256") != lockfile.sha256_file(build_lock):
        problems.append("bundle was prepared for a different requirements-build.lock (sha256 mismatch); rerun prepare_offline_package")
    try:
        bundle_platform = manifest_sys_platform(manifest["target"])
        wanted = sys_platform or sys.platform
        if bundle_platform != wanted:
            problems.append("bundle target %s is for %s, this computer is %s" % (manifest["target"].get("name"), bundle_platform, wanted))
        found = check_wheels(dest, applicable_requirements(runtime_lock, build_lock, bundle_platform), manifest["target"].get("name", ""))
        if found != manifest["files"]:
            problems.append("files on disk differ from %s" % MANIFEST_NAME)
    except (OfflineError, lockfile.LockFileError) as exc:
        problems.append(str(exc))
    return problems


def planned_removals(dest: Path, force: bool) -> List[Path]:
    """Files to delete before writing a bundle into `dest` (non-destructive check; raises when dest is not usable)."""
    if dest.exists() and not dest.is_dir():
        raise OfflineError("%s exists and is not a directory" % dest)
    entries = sorted(p.name for p in dest.iterdir()) if dest.exists() else []
    if not entries:
        return []
    if not force:
        raise OfflineError("%s is not empty. Use --force to replace a bundle previously created by this script, or choose another --dest." % dest)
    manifest_path = dest / MANIFEST_NAME
    if not manifest_path.is_file():
        raise OfflineError("%s is not empty and has no %s; --force only replaces a bundle created by this script. "
                           "Delete or empty the folder yourself (it may be a partial download)." % (dest, MANIFEST_NAME))
    try:
        listed = json.loads(manifest_path.read_text(encoding="utf-8"))["files"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise OfflineError("cannot read the file list of %s: %s" % (manifest_path, exc)) from exc
    if not isinstance(listed, dict):
        raise OfflineError("%s: 'files' is not an object; refusing to delete anything" % manifest_path)
    bad = [n for n in listed if n != Path(n).name or n in (".", "..") or not n.endswith(".whl")]
    if bad:
        raise OfflineError("%s lists invalid file names %s; refusing to delete anything" % (manifest_path, bad))
    unknown = [n for n in entries if n != MANIFEST_NAME and n not in listed]
    if unknown:
        raise OfflineError("%s contains entries not listed in %s: %s. Nothing was deleted." % (dest, MANIFEST_NAME, ", ".join(unknown)))
    return [dest / n for n in sorted(listed) if (dest / n).is_file()] + [manifest_path]


def _pip_download(requirements: Path, staging: Path, pip_args: List[str]) -> None:
    cmd = [sys.executable, "-m", "pip", "download", "--disable-pip-version-check", "--no-input", "--require-hashes", "--only-binary=:all:",
           "--no-deps", "--dest", str(staging)] + pip_args + ["-r", str(requirements)]
    print("> " + " ".join('"%s"' % c if " " in c else c for c in cmd), flush=True)
    code = subprocess.call(cmd, env=child_env())
    if code != 0:
        raise OfflineError("pip download failed with exit code %d. Check the network, proxy (HTTPS_PROXY) or package index "
                           "(PIP_INDEX_URL) settings; a hash error means the index served a file that differs from the lock." % code)


def prepare(target_name: str, dest: Path, force: bool, runtime_lock: Path, build_lock: Path) -> dict:
    spec = target_spec(target_name)
    for path in (runtime_lock, build_lock):
        if not path.is_file():
            raise OfflineError("lock file not found: %s" % path)
    reqs = applicable_requirements(runtime_lock, build_lock, spec["sys_platform"])
    removals = planned_removals(dest, force)
    probe = subprocess.run([sys.executable, "-m", "pip", "--version"], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           universal_newlines=True, encoding="utf-8", errors="replace", env=child_env())
    if probe.returncode != 0:
        raise OfflineError("pip is not available for %s:\n%s" % (sys.executable, probe.stdout))
    print("Using %s" % probe.stdout.strip(), flush=True)
    for path in removals:
        path.unlink()
    if removals:
        print("Removed the previous bundle (%d files listed in its %s)." % (len(removals) - 1, MANIFEST_NAME), flush=True)
    dest.mkdir(parents=True, exist_ok=True)
    staging = dest / STAGING_NAME
    staging.mkdir()
    try:
        with tempfile.TemporaryDirectory(prefix="falls_ml_offline_") as tmp:
            for lock in (build_lock, runtime_lock):
                evaluated = Path(tmp) / (lock.name + ".evaluated.txt")
                lockfile.write_evaluated_requirements(lock, evaluated, spec["sys_platform"])
                _pip_download(evaluated, staging, spec["pip_args"])
        files = check_wheels(staging, reqs, spec["manifest"]["name"], ignore=())
        for name in sorted(files):
            os.replace(str(staging / name), str(dest / name))
    finally:
        if staging.exists():  # created above inside a verified-empty folder, so everything in it is ours
            shutil.rmtree(str(staging))
    manifest = {"format": 1, "created_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "target": spec["manifest"],
                "requirements_lock_sha256": lockfile.sha256_file(runtime_lock), "build_lock_sha256": lockfile.sha256_file(build_lock),
                "files": dict(sorted(files.items()))}
    tmp_manifest = dest / (MANIFEST_NAME + ".tmp")
    tmp_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(str(tmp_manifest), str(dest / MANIFEST_NAME))
    return manifest


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Prepare or verify the offline wheel bundle (offline_packages).")
    running = "windows-amd64-cp%d%d" % sys.version_info[:2]
    default_target = running if sys.platform == "win32" and running in WINDOWS_TARGETS else WINDOWS_TARGET
    ap.add_argument("--target", default=default_target, choices=[*sorted(WINDOWS_TARGETS), "current"],
                    help="windows-amd64-cp311 or windows-amd64-cp313 (default: this Windows computer's Python series, "
                         "otherwise windows-amd64-cp313)")
    ap.add_argument("--dest", default=str(ROOT / "offline_packages"), help="bundle folder (default: <project>/offline_packages)")
    ap.add_argument("--force", action="store_true", help="replace a bundle previously created by this script")
    ap.add_argument("--verify-only", action="store_true", help="only verify an existing bundle against the locks and this computer")
    ap.add_argument("--lock", default=str(ROOT / "requirements.lock"), help=argparse.SUPPRESS)
    ap.add_argument("--build-lock", default=str(ROOT / "requirements-build.lock"), help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    dest, runtime_lock, build_lock = Path(args.dest).resolve(), Path(args.lock).resolve(), Path(args.build_lock).resolve()
    try:
        if args.verify_only:
            problems = verify_bundle(dest, runtime_lock, build_lock)
            if problems:
                raise OfflineError("offline bundle %s is not usable:\n  %s" % (dest, "\n  ".join(problems)))
            print("[OK] offline bundle %s matches the lock files" % dest)
            return 0
        manifest = prepare(args.target, dest, args.force, runtime_lock, build_lock)
    except (OfflineError, lockfile.LockFileError, OSError) as exc:
        print("\nERROR: %s" % exc, file=sys.stderr)
        return 1
    size = sum((dest / name).stat().st_size for name in manifest["files"])
    print("")
    print("[OK] Offline bundle ready: %d wheels, %.1f MB, target %s" % (len(manifest["files"]), size / 1e6, manifest["target"]["name"]))
    print("     Folder:   %s" % dest)
    print("     Manifest: %s" % (dest / MANIFEST_NAME))
    print("")
    print("Next steps:")
    if manifest["target"]["name"] == WINDOWS_TARGET:
        print("  1. Copy this folder into the project folder on the Windows computer as 'offline_packages'")
        print("     (next to setup_windows.cmd), together with the same requirements.lock and requirements-build.lock.")
        print("  2. In CMD, in the project folder, run:  setup_windows.cmd --offline")
    else:
        print("  1. Keep this folder as 'offline_packages' in the project folder of a computer of the same platform.")
        print("  2. Windows (CMD):  setup_windows.cmd --offline")
        print("     macOS (POSIX):  bash scripts/posix/setup.sh --offline")
    print("  To prepare the bundle again (for example after the lock files changed), re-run this command with --force;")
    print("  it replaces only the files listed in %s. If setup reports a bundle for another platform, use" % MANIFEST_NAME)
    print("  --target %s --force for a Windows computer." % WINDOWS_TARGET)
    return 0


if __name__ == "__main__":
    sys.exit(main())
