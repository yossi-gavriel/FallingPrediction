"""Structured (JSON-lines) logging.

Usage:
    log = get_logger(__name__)
    log.info("split_created", extra_fields={"n_train": 100})
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if fields:
            payload.update(fields)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, sort_keys=False)


class StructuredLogger(logging.LoggerAdapter):
    """Logger adapter accepting ``extra_fields`` as a dict of structured values."""

    def process(self, msg: Any, kwargs: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        fields = kwargs.pop("extra_fields", None) or {}
        kwargs["extra"] = {"fields": fields}
        return msg, kwargs


def get_logger(name: str) -> StructuredLogger:
    logger = logging.getLogger(name)
    return StructuredLogger(logger, {})


def configure_logging(level: int = logging.INFO, run_log_path: Path | None = None) -> None:
    """Configure root logging to stderr (JSON) and optionally a run-level JSON-lines file."""
    root = logging.getLogger("falls_ml")
    root.setLevel(level)
    root.propagate = False
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(JsonFormatter())
    root.addHandler(stream)
    if run_log_path is not None:
        run_log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(run_log_path, encoding="utf-8")
        fh.setFormatter(JsonFormatter())
        root.addHandler(fh)


@contextmanager
def run_log(run_log_path: Path) -> Iterator[None]:
    """Attach a JSON-lines file handler for one run and always detach it afterwards (no cross-run leakage)."""
    root = logging.getLogger("falls_ml")
    if root.level == logging.NOTSET or root.level > logging.INFO:
        root.setLevel(logging.INFO)
    run_log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(run_log_path, encoding="utf-8")
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    try:
        yield
    finally:
        root.removeHandler(handler)
        handler.close()
