"""External-dataset adapter: rename and recode an eFalls-compatible table to the modelling schema (spec §13.1).

An adapter renames and recodes only; it never computes features from raw events (that is the data engineering
of the source environment). Adapters are declared per source, versioned, and fingerprinted (``sha256``) so the
mapping version recorded in the dataset manifest identifies the exact mapping used.

Order of operations in :meth:`ExternalDatasetAdapter.apply`: drop declared columns, rename, parse dates, recode.
``recode_map`` and ``date_columns`` use canonical (renamed) column names. Recode keys are matched on their string
form (integral floats as integers, so 1.0 matches key "1"); a recode target of ``null`` means missing.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from falls_ml.errors import ConfigError, DatasetValidationError
from falls_ml.logging_utils import get_logger

log = get_logger(__name__)

_TOP_LEVEL_KEYS = {"adapter", "column_map", "recode_map", "drop_columns", "date_columns"}


def _key(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _target_dtype(column: str, targets: list[Any]) -> str:
    values = [v for v in targets if v is not None]
    if all(isinstance(v, str) for v in values):
        return "str"
    if all(isinstance(v, int) and not isinstance(v, bool) for v in values):
        return "int"
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
        return "float64"
    raise ConfigError(f"recode_map[{column!r}] mixes target value types {sorted({type(v).__name__ for v in values})}")


@dataclass(frozen=True)
class ExternalDatasetAdapter:
    """Declarative column/recode mapping from one external source to the canonical modelling columns."""

    name: str
    source: str
    column_map: Mapping[str, str]
    recode_map: Mapping[str, Mapping[str, Any]]
    drop_columns: tuple[str, ...] = ()
    date_columns: tuple[str, ...] = ()
    version: str = "0.0.0"

    def __post_init__(self) -> None:
        object.__setattr__(self, "column_map", {str(k): str(v) for k, v in dict(self.column_map).items()})
        object.__setattr__(self, "recode_map", {str(c): {_key(k): v for k, v in dict(m).items()}
                                                for c, m in dict(self.recode_map).items()})
        object.__setattr__(self, "drop_columns", tuple(map(str, self.drop_columns)))
        object.__setattr__(self, "date_columns", tuple(map(str, self.date_columns)))
        problems = []
        if not self.name or not self.source:
            problems.append("name and source are required")
        targets = list(self.column_map.values())
        dupes = sorted({t for t in targets if targets.count(t) > 1})
        if dupes:
            problems.append(f"several source columns map to {dupes}")
        overlap = sorted(set(self.drop_columns) & set(self.column_map))
        if overlap:
            problems.append(f"columns both mapped and dropped: {overlap}")
        for label, names in (("recode_map", self.recode_map), ("date_columns", self.date_columns)):
            unknown = sorted(set(names) - set(targets))
            if unknown:
                problems.append(f"{label} refers to columns that are not column_map targets: {unknown}")
        both = sorted(set(self.recode_map) & set(self.date_columns))
        if both:
            problems.append(f"columns cannot be both recoded and parsed as dates: {both}")
        if problems:
            raise ConfigError(f"Invalid external dataset adapter {self.name!r}: " + "; ".join(problems))
        for column, mapping in self.recode_map.items():
            _target_dtype(column, list(mapping.values()))

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExternalDatasetAdapter:
        path = Path(path)
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except FileNotFoundError as exc:
            raise ConfigError(f"Adapter config not found: {path}") from exc
        unknown = sorted(set(raw) - _TOP_LEVEL_KEYS)
        if unknown or "adapter" not in raw or "column_map" not in raw:
            raise ConfigError(f"{path}: needs 'adapter' and 'column_map'; unknown keys {unknown}")
        meta = raw["adapter"]
        missing = [k for k in ("name", "source", "version") if not meta.get(k)]
        if missing:
            raise ConfigError(f"{path}: adapter section missing {missing}")
        return cls(name=str(meta["name"]), source=str(meta["source"]), version=str(meta["version"]),
                   column_map=raw["column_map"], recode_map=raw.get("recode_map") or {},
                   drop_columns=tuple(raw.get("drop_columns") or ()), date_columns=tuple(raw.get("date_columns") or ()))

    def sha256(self) -> str:
        """Fingerprint of the full mapping declaration (record it with the mapping version)."""
        payload = {"name": self.name, "source": self.source, "version": self.version, "column_map": self.column_map,
                   "recode_map": self.recode_map, "drop_columns": self.drop_columns, "date_columns": self.date_columns}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename, parse dates and recode ``df``. Raises ``DatasetValidationError`` listing every mismatch."""
        columns = [str(c) for c in df.columns]
        problems = []
        dupes = sorted({c for c in columns if columns.count(c) > 1})
        if dupes:
            problems.append(f"duplicate source columns {dupes}")
        unmapped = [c for c in columns if c not in self.column_map and c not in self.drop_columns]
        if unmapped:
            problems.append(f"unmapped source columns (map or declare in drop_columns): {unmapped}")
        missing = [c for c in [*self.column_map, *self.drop_columns] if c not in columns]
        if missing:
            problems.append(f"declared source columns absent from the data: {missing}")
        if problems:
            raise DatasetValidationError(f"Adapter {self.name} {self.version}: source columns do not match", problems)

        out = df.drop(columns=list(self.drop_columns)).rename(columns=self.column_map)
        for column in self.date_columns:
            parsed = pd.to_datetime(out[column], errors="coerce")
            bad = out[column].notna() & parsed.isna()
            if bad.any():
                problems.append(f"{column}: {int(bad.sum())} values not parseable as dates, e.g. {out.loc[bad, column].iloc[0]!r}")
            out[column] = parsed.astype("datetime64[ns]")
        for column, mapping in self.recode_map.items():
            present = out[column].notna().to_numpy()
            keys = [_key(v) for v in out.loc[present, column]]
            bad_values = sorted(set(keys) - set(mapping))
            if bad_values:
                problems.append(f"{column}: unmapped values {bad_values[:20]}")
                continue
            values = np.full(len(out), None, dtype=object)
            values[present] = [mapping[k] for k in keys]
            recoded = pd.Series(values, index=out.index)
            dtype = _target_dtype(column, list(mapping.values()))
            if dtype == "int":
                dtype = "Int64" if recoded.isna().any() else "int64"
            out[column] = recoded.astype(dtype)
        if problems:
            raise DatasetValidationError(f"Adapter {self.name} {self.version}: values do not match", problems)
        log.info("external_dataset_adapted", extra_fields={"adapter": self.name, "version": self.version,
                                                           "source": self.source, "n_rows": len(out),
                                                           "adapter_sha256": self.sha256()})
        return out
