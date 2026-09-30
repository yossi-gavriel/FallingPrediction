"""Child-process helpers for tests: UTF-8 on both ends of the pipe, on every OS.

On Windows a piped Python child writes the ANSI code page (cp1252/cp1255) unless told otherwise, and ``text=True`` without
``encoding`` decodes with that code page in the parent. Tests therefore force UTF-8 in the child (``PYTHONUTF8``,
``PYTHONIOENCODING``, via :func:`utf8_env`) and decode with ``text=True, encoding="utf-8", errors="replace"`` in the parent.
"""

from __future__ import annotations

import os


def utf8_env(**extra: str) -> dict[str, str]:
    """Copy of ``os.environ`` with UTF-8 mode and UTF-8 standard streams for a Python child process."""
    return {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", **extra}
