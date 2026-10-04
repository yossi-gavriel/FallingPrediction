"""The sealed reader: the ONLY way Phase 4 preflight / scoring touch the 2026 extract.

A column is SEALED when it is in the brief's outcome list, has a sealed contract role (LABEL, FORBIDDEN_LEAKAGE) or contract timing post_index
(follow-up, censoring, death), or - for a column that is not in the 2025 contract - when its name looks like outcome / future information.
Sealed columns are never requested from the file (``usecols``): their values never enter memory before ``meuhedet-phase4-evaluate`` verified the
frozen prediction hashes. Every read is logged in :data:`READ_LOG` (column names only), so the tests can prove it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from falls_ml.phase2.state import Phase2Stop

READ_LOG: list[tuple[str, tuple[str, ...]]] = []          # (file name, columns requested) - names only, for the sealing tests
UNREADABLE_PREFIX = "__unreadable__"                       # identical to the Phase 3 marker (row is UNKNOWN for that source)
SUPPORTED = (".csv", ".txt", ".parquet")


def read_header(src: Path) -> list[str]:
    suffix = src.suffix.lower()
    if suffix not in SUPPORTED:
        raise Phase2Stop("INPUT_FORMAT", f"{src.name}: Phase 4 reads .csv (SSMS export with NULL literals) or .parquet only")
    if suffix == ".parquet":
        import pyarrow.parquet as pq

        cols = list(pq.read_schema(src).names)
    else:
        cols = list(pd.read_csv(src, nrows=0, dtype="string", encoding="utf-8").columns)
    cols = [str(c).strip() for c in cols]
    dup = sorted({c for c in cols if cols.count(c) > 1})
    if dup:
        raise Phase2Stop("SCHEMA_DUPLICATE_COLUMNS", f"{src.name}: duplicated column names {dup[:10]}")
    return cols


def sealed_map(header: list[str], contract: Any, cfg: Any) -> dict[str, str]:
    """column -> why it is sealed (only columns present in the file, plus the brief's list for documentation)."""
    roles = set(cfg["sealed_contract_roles"])
    pats = [re.compile(p, re.IGNORECASE) for p in cfg["sealed_name_patterns"]]
    names = set(contract.names)
    out: dict[str, str] = {}
    for c in header:
        if c in set(cfg["sealed_columns"]):
            out[c] = "BRIEF_OUTCOME_COLUMN"
        elif c in names:
            cc = contract.get(c)
            if cc.role in roles:
                out[c] = f"CONTRACT_ROLE_{cc.role}"
            elif cc.timing == "post_index":
                out[c] = "CONTRACT_TIMING_POST_INDEX"
        elif any(p.search(c) for p in pats):
            out[c] = "NEW_COLUMN_NAME_LOOKS_LIKE_OUTCOME_OR_FUTURE"
    return out


@dataclass
class SealedRead:
    raw: pd.DataFrame                     # string cells exactly as read (contract NULL literals)
    frame: pd.DataFrame                   # contract columns cast to the contract types; other columns kept as text
    wrong_type: dict[str, int] = field(default_factory=dict)
    not_allowed: dict[str, int] = field(default_factory=dict)
    unreadable: pd.DataFrame | None = None            # per-cell masks (column -> bool) of contract columns with a cell problem
    date_formats: dict[str, dict[str, int]] = field(default_factory=dict)


def read_columns(src: Path, columns: list[str], sealed: dict[str, str], contract: Any) -> SealedRead:
    """Read ONLY ``columns`` (never a sealed one) and cast the contract columns with the validated Phase 1 coercion (declared date layouts,
    sentinels, no inference)."""
    from falls_ml.data.meuhedet_wide import NA_VALUES, coerce_wide_types_report

    bad = sorted(set(columns) & set(sealed))
    if bad:
        raise Phase2Stop("SEALED_COLUMN_REQUESTED", f"internal guard: a sealed column was requested before the frozen hashes were verified: {bad[:10]}")
    cols = list(dict.fromkeys(columns))
    READ_LOG.append((src.name, tuple(cols)))
    if src.suffix.lower() == ".parquet":
        raw = pd.read_parquet(src, columns=cols)
        raw = raw.astype("string")
    else:
        raw = pd.read_csv(src, usecols=cols, dtype="string", na_values=list(NA_VALUES), keep_default_na=False, encoding="utf-8")
    raw.columns = [str(c).strip() for c in raw.columns]
    raw = raw[cols]
    frame, rep = coerce_wide_types_report(raw, contract)
    names = set(contract.names)
    unr = pd.DataFrame(index=raw.index)
    not_allowed: dict[str, int] = {}
    for c in [c for c in cols if c in names]:
        cc = contract.get(c)
        m = np.zeros(len(raw), dtype=bool)
        if int(rep.wrong_type.get(c, 0)):
            m |= (raw[c].notna() & frame[c].isna()).to_numpy(dtype=bool)
        if cc.allowed is not None:
            s = frame[c]
            if cc.is_integer:
                ok = s.isna() | s.astype("Float64").isin([float(v) for v in cc.allowed])
            elif cc.pandas_dtype == "float64":
                ok = s.isna() | s.astype("float64").isin([float(v) for v in cc.allowed])
            else:
                ok = s.isna() | s.astype("string").isin([str(v) for v in cc.allowed])
            na = ~ok.fillna(False).to_numpy(dtype=bool)
            if na.any():
                not_allowed[c] = int(na.sum())
            m |= na
        if m.any():
            unr[c] = m
    return SealedRead(raw=raw, frame=frame, wrong_type=dict(rep.wrong_type), not_allowed=not_allowed, unreadable=unr, date_formats=dict(rep.date_formats))


def read_ids(src: Path, sealed: dict[str, str], *, id_col: str = "Customer_Full_ID", extra: tuple[str, ...] = ("Index_Date", "Is_Eligible_Cohort")) -> pd.DataFrame:
    """Identifier + eligibility columns only (overlap, keys, privacy scan): kept in memory, never written."""
    header = read_header(src)
    cols = [c for c in (id_col, "Snapshot_Key", *extra) if c in header]
    bad = sorted(set(cols) & set(sealed))
    if bad:
        raise Phase2Stop("SEALED_COLUMN_REQUESTED", f"internal guard: {bad}")
    READ_LOG.append((src.name, tuple(cols)))
    if src.suffix.lower() == ".parquet":
        return pd.read_parquet(src, columns=cols).astype("string")
    from falls_ml.data.meuhedet_wide import NA_VALUES

    return pd.read_csv(src, usecols=cols, dtype="string", na_values=list(NA_VALUES), keep_default_na=False, encoding="utf-8")[cols]
