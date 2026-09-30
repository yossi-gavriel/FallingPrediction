"""One rule for resolving file paths written in configs, manifests and run artifacts.

Paths in YAML files are *declared* relative to the project root (the directory containing ``configs/``).
They are resolved at use time, never baked into config hashes, so a config hash is identical on every
machine:

1. absolute paths are used as given;
2. otherwise, relative to the project root found by walking up from ``anchor`` (the YAML file, run
   directory or bundle that declared the path) to the first directory containing ``configs/``;
3. otherwise, relative to the installed package's repository root (same walk from this module);
4. otherwise, relative to the current working directory.

The first existing candidate wins; if none exists a ``ConfigError`` lists every candidate tried.
"""

from __future__ import annotations

from pathlib import Path

from falls_ml.errors import ConfigError


def project_root(start: str | Path | None) -> Path | None:
    """First ancestor of ``start`` (inclusive) that contains a ``configs`` directory, or None."""
    if start is None:
        return None
    p = Path(start).resolve()
    if p.is_file():
        p = p.parent
    for candidate in (p, *p.parents):
        if (candidate / "configs").is_dir():
            return candidate
    return None


def portable_path(declared: str | Path) -> str:
    """A declared path as recorded in artifacts: relative paths with ``/`` separators (identical on every OS, and still
    resolvable by :func:`resolve_path` on Windows); absolute paths unchanged."""
    p = Path(declared)
    return str(p) if p.is_absolute() else p.as_posix()


def project_relative_posix(path: str | Path, anchor: str | Path | None) -> str | None:
    """``path`` relative to the project root found from ``anchor`` (``/`` separators), or None when there is no project root
    or ``path`` lies outside it. Lets artifacts point at project files after the project folder is moved or copied."""
    root = project_root(anchor)
    if root is None:
        return None
    try:
        return Path(path).resolve().relative_to(root).as_posix()
    except ValueError:
        return None


def resolve_path(declared: str | Path, *, anchor: str | Path | None = None, must_exist: bool = True) -> Path:
    """Resolve a declared path (see module docstring). ``anchor`` is where the path was declared."""
    declared = Path(declared)
    if declared.is_absolute():
        if must_exist and not declared.exists():
            raise ConfigError(f"Path not found: {declared}")
        return declared
    candidates: list[Path] = []
    for root in (project_root(anchor), project_root(Path(__file__))):
        if root is not None and root / declared not in candidates:
            candidates.append(root / declared)
    candidates.append(Path.cwd() / declared)
    for c in candidates:
        if c.exists():
            return c.resolve()
    if not must_exist:
        return candidates[0]
    raise ConfigError(f"Path {str(declared)!r} not found; tried: {[str(c) for c in candidates]}")
