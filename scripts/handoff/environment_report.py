"""Write environment_report.txt: OS, interpreter, pip, installed distributions, project version, locks. Standard library only.

Usage: <venv python> scripts/handoff/environment_report.py [--out environment_report.txt] [--mode online|offline|unknown]

Run it with the virtual-environment interpreter so the installed distributions are the ones listed. Environment
variables are never written; the report only says whether PIP_INDEX_URL / HTTPS_PROXY-style settings and the pip
install-target overrides PIP_USER / PIP_TARGET / PIP_PREFIX (cleared by the installer) are set (yes/no).
"""
from __future__ import annotations

import argparse
import importlib.metadata
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

FLAGGED_SETTINGS = ("PIP_INDEX_URL", "PIP_EXTRA_INDEX_URL", "PIP_FIND_LINKS", "PIP_CERT", "HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY")


def _run(cmd: list[str], cwd: Path | None = None) -> tuple[int, str]:
    try:
        proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", errors="replace", timeout=120, env=common.subprocess_env())
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 127, str(exc)
    return proc.returncode, proc.stdout.strip()


def os_lines() -> list[str]:
    lines = [f"platform: {platform.platform()}", f"system: {platform.system()} {platform.release()} ({platform.version()})",
             f"machine: {platform.machine()}"]
    if sys.platform == "win32":
        release, version, csd, ptype = platform.win32_ver()
        lines.append(f"windows: release={release} version={version} service_pack={csd or '-'} type={ptype}")
        lines.append(f"windows edition: {platform.win32_edition() or 'not available'}")
        lines.append(f"long paths enabled: {common.windows_long_paths_enabled()}")
    elif sys.platform == "darwin":
        lines.append(f"macos: {platform.mac_ver()[0]}")
    return lines


def python_lines() -> list[str]:
    rc, pip_version = _run([sys.executable, "-m", "pip", "--version"])
    pip_text = pip_version if rc == 0 else f"not available ({(pip_version.splitlines() or [f'exit code {rc}'])[-1]})"
    return [f"executable: {sys.executable}", f"version: {platform.python_version()} ({platform.python_implementation()})",
            f"bits: {64 if sys.maxsize > 2 ** 32 else 32}", f"prefix: {sys.prefix}", f"base prefix: {sys.base_prefix}",
            f"in virtual environment: {'yes' if sys.prefix != sys.base_prefix else 'no'}", f"pip: {pip_text}"]


def distribution_lines() -> list[str]:
    seen: dict[str, str] = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata["Name"]
        if name:
            seen.setdefault(re.sub(r"[-_.]+", "-", name).lower(), f"{name}=={dist.version}")
    return sorted(seen.values(), key=str.lower)


def project_lines(root: Path) -> list[str]:
    version_file = root / "VERSION"
    project_version = version_file.read_text(encoding="utf-8").strip() if version_file.is_file() else "VERSION file missing"
    rc, out = _run([sys.executable, "-c", "import falls_ml, sys; sys.stdout.write(falls_ml.__version__ + '|' + falls_ml.__file__)"], cwd=root)
    if rc == 0 and "|" in out:
        package_version, package_file = out.split("|", 1)
    else:
        package_version, package_file = f"import failed ({out.splitlines()[-1] if out else rc})", "-"
    git = "not available"
    if (root / ".git").exists() and shutil.which("git"):
        rc, commit = _run(["git", "-C", str(root), "rev-parse", "HEAD"])
        git = commit if rc == 0 and re.fullmatch(r"[0-9a-f]{40}", commit) else "not available"
    return [f"project folder: {root}", f"VERSION file: {project_version}", f"falls_ml.__version__: {package_version}",
            f"falls_ml imported from: {package_file}", f"git commit: {git}"]


def lock_lines(root: Path) -> list[str]:
    import lockfile

    lines = []
    for path in common.lock_paths(root):
        lines.append(f"{path.name} sha256: {lockfile.sha256_file(path) if path.is_file() else 'missing'}")
    manifest = root / common.OFFLINE_DIR / common.OFFLINE_MANIFEST
    lines.append(f"offline bundle manifest: {'present' if manifest.is_file() else 'absent'}")
    return lines


def settings_lines() -> list[str]:
    present = {k.upper() for k in os.environ}
    return [f"{name} set: {'yes' if name in present else 'no'}" for name in FLAGGED_SETTINGS]


def pip_target_lines() -> list[str]:
    """PIP_USER / PIP_TARGET / PIP_PREFIX: set in the user's environment or not (yes/no). The installer clears them before any
    pip process runs, so the names come from FALLS_ML_CLEARED_PIP_SETTINGS (recorded by the entry point) as well."""
    cleared = set(common.cleared_pip_settings())
    return [f"{name} set: {'yes' if name in cleared else 'no'}" for name in common.PIP_TARGET_SETTINGS]


def build_report(root: Path, mode: str) -> str:
    sections = [
        ("falls_ml environment report", [f"generated (UTC): {common.utc_now()}", f"install mode: {mode}",
                                         "values of environment variables are deliberately not recorded"]),
        ("Operating system", os_lines()),
        ("Python", python_lines()),
        ("Project", project_lines(root)),
        ("Lock files", lock_lines(root)),
        ("Network-related settings (set: yes/no only)", settings_lines()),
        ("pip install-target settings (set: yes/no only; always cleared for the installer's pip processes)", pip_target_lines()),
        ("Installed distributions", distribution_lines()),
    ]
    out = []
    for title, lines in sections:
        out += [title, "-" * len(title), *lines, ""]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    common.configure_console()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(common.PROJECT_ROOT / common.ENVIRONMENT_REPORT))
    ap.add_argument("--mode", default="unknown", choices=["online", "offline", "unknown"])
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.write_text(build_report(common.PROJECT_ROOT, args.mode), encoding="utf-8")
    print(f"environment report written: {out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
