"""Meuhedet wide-table adapter (Phase 1): data contract, eFalls mapping manifest and the canonical modelling frame.

    V_Falls_Prediction_Wide_1 (221 columns, immutable research VIEW)
        -> read_wide_extract / validate_wide_contract      (configs/meuhedet/wide_v1_columns.yaml: types, roles, NULL semantics)
        -> MeuhedetWideDatasetAdapter.apply                (configs/meuhedet/wide_v1_efalls_mapping.yaml: eFalls concepts only)
        -> canonical modelling frame (spec §13.3) written by build_meuhedet_dataset via write_modeling_dataset

The ML code never sees the raw 221-column schema. Nothing is inferred from the data: every dtype, allowed value and NULL rule
comes from the contract; every predictor comes from the mapping manifest; NULL is converted to 0 only where the manifest
declares the eFalls absent_is_zero rule (and every such conversion is counted in the build report). No database access.
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
from collections import Counter
from datetime import date, datetime
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from falls_ml.data.schema import OUTCOME_EVENT_DATE, PREDICTOR_MAX_DATE
from falls_ml.errors import ConfigError, DatasetValidationError, LeakageError
from falls_ml.features.spec import FeatureSpec, load_feature_spec
from falls_ml.logging_utils import get_logger

log = get_logger(__name__)

ROLES = ("IDENTIFIER", "COHORT_ELIGIBILITY", "QA_CONTROL", "EFALLS_BASELINE_FEATURE", "MEUHEDET_ENHANCED_FEATURE", "LABEL",
         "FORBIDDEN_LEAKAGE")
PREDICTOR_ROLES = frozenset({"EFALLS_BASELINE_FEATURE", "MEUHEDET_ENHANCED_FEATURE"})
SEMANTIC_TYPES = ("identifier", "date", "binary", "continuous", "count", "categorical", "ordinal", "quality_control", "label")
MODEL_DTYPES = ("never", "float64", "int8_binary", "onehot_categorical", "ordinal_float64")
TIMINGS = ("pre_index", "at_index", "post_index", "unknown", "na")
PREDICTOR_TIMINGS = frozenset({"pre_index", "at_index"})
MAPPING_STATUS = ("AVAILABLE", "APPROXIMATE", "UNAVAILABLE")
MAPPING_QUALITY = ("EXACT", "HIGH_CONFIDENCE", "APPROXIMATE", "UNAVAILABLE")
OPS = ("copy_numeric", "recode", "any_positive", "count")
INT_SQL = ("int", "smallint", "tinyint", "bit")
NA_VALUES = ("NULL", "null", "Null", "", "NaN", "nan", "N/A")
DATE_DTYPE = "datetime64[us]"           # contract layer: the full SQL date range (0001-01-01 .. 9999-12-31), e.g. the 2999-12-31 sentinel
MODELING_DATE_DTYPE = "datetime64[ns]"  # canonical modelling dataset (day-precision datetime64[ns], as everywhere in the framework)
NS_MIN_DAY = np.datetime64("1677-09-22", "us")  # first / last whole day representable in datetime64[ns]
NS_MAX_DAY = np.datetime64("2262-04-11", "us")
NS_RANGE_TEXT = "1677-09-22..2262-04-11"
SENTINEL_MODELING_RULES = ("keep", "to_null")
_FRACTION_RE = re.compile(r"(\.\d{6})\d+")  # SQL datetime2 exports carry 7 fractional digits; python datetimes keep 6
_TIME_RE = re.compile(r"^(\d{1,2}):(\d{2})(?::(\d{2})(?:\.(\d{1,6}))?)?$")
_TZ_RE = re.compile(r"(Z|[+-]\d{2}:?\d{2})$")
DEFAULT_DATE_FORMATS = ("%Y-%m-%d", "%d/%m/%Y")  # trial order; declared in the contract (contract.date_formats), never inferred
NATIVE_DATES = "native date cells (no text parsing)"  # Excel/Parquet columns that already carry real dates
SOURCE_NAME = "meuhedet_wide_v1"
DEFAULT_CONTRACT = "configs/meuhedet/wide_v1_columns.yaml"
DEFAULT_MAPPING = "configs/meuhedet/wide_v1_efalls_mapping.yaml"


def _sha256_json(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _sql_base(sql: str) -> str:
    return str(sql).split("(")[0].strip().lower()


# ============================================================================ contract
@dataclass(frozen=True)
class ColumnContract:
    name: str
    sql: str
    nullable: bool
    role: str
    semantic: str
    model_dtype: str
    timing: str
    allowed: tuple[Any, ...] | None = None
    null_means: str | None = None
    group: str | None = None
    note: str = ""
    sentinels: tuple[str, ...] = ()            # declared business sentinel dates (ISO), e.g. 2999-12-31 = open-ended SCD row
    sentinel_means: str | None = None          # their documented meaning (required when sentinels are declared)
    sentinel_modeling_rule: str = "keep"       # keep = the value stays a date (build stops if the modelling layer cannot hold it); to_null = NULL there, counted

    @property
    def base_type(self) -> str:
        return _sql_base(self.sql)

    @property
    def is_integer(self) -> bool:
        return self.base_type in INT_SQL

    @property
    def is_date(self) -> bool:
        return self.base_type in ("date", "datetime")

    @property
    def is_text(self) -> bool:
        return self.base_type in ("varchar", "nvarchar", "char")

    @property
    def pandas_dtype(self) -> str:
        if self.is_integer:
            return "Int64"
        if self.base_type in ("decimal", "numeric", "float"):
            return "float64"
        if self.is_date:
            return DATE_DTYPE
        return "string"

    @property
    def predictor_allowed(self) -> bool:
        return self.role in PREDICTOR_ROLES and self.timing in PREDICTOR_TIMINGS and self.model_dtype != "never"


@dataclass(frozen=True)
class WideContract:
    name: str
    version: str
    source_view: str
    definition_version: str
    date_formats: tuple[str, ...]
    columns: tuple[ColumnContract, ...]
    content_sha256: str
    raw: Mapping[str, Any] = field(default_factory=dict, compare=False, hash=False, repr=False)

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.columns]

    def get(self, name: str) -> ColumnContract:
        for c in self.columns:
            if c.name == name:
                return c
        raise ConfigError(f"Column {name!r} is not in contract {self.name} {self.version}")

    def by_role(self, role: str) -> list[ColumnContract]:
        return [c for c in self.columns if c.role == role]


def check_date_formats(formats: Iterable[str]) -> list[str]:
    """Validate the declared source date formats: each must be a strptime format carrying year, month and day, and no two of them
    may read the same text as different dates (that would make the order decide the meaning, i.e. day/month inference)."""
    formats = list(formats)
    problems: list[str] = []
    if not formats:
        return ["contract.date_formats is empty; declare at least one source date format (e.g. %Y-%m-%d)"]
    usable: list[str] = []
    for f in formats:
        missing = [t for t in ("%Y", "%m", "%d") if t not in f]
        if missing:
            problems.append(f"contract.date_formats {f!r}: a source date format needs {missing} (year, month and day are always explicit)")
            continue
        try:
            probe = datetime(2024, 2, 1)
            if datetime.strptime(probe.strftime(f), f) != probe:
                problems.append(f"contract.date_formats {f!r}: does not round-trip a date")
                continue
        except (ValueError, TypeError) as exc:
            problems.append(f"contract.date_formats {f!r}: not a usable strptime format ({exc})")
            continue
        usable.append(f)
    probe = datetime(2024, 2, 1)   # day 1, month 2: the classic DD/MM vs MM/DD ambiguity
    for a in usable:
        text = probe.strftime(a)
        for b in usable:
            if b == a:
                continue
            try:
                other = datetime.strptime(text, b)
            except ValueError:
                continue
            if other != probe:
                problems.append(f"contract.date_formats {a!r} and {b!r} are ambiguous: {text!r} would be {probe.date()} under the first "
                                f"and {other.date()} under the second; declare only one reading of a source layout")
    return problems


def load_wide_contract(path: str | Path = DEFAULT_CONTRACT) -> WideContract:
    """Load and validate the column contract (every column has a role, a semantic type and a model dtype; rules enforced)."""
    from falls_ml.paths import resolve_path

    p = resolve_path(path)
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Wide-table contract not found: {p}") from exc
    if not isinstance(raw, dict) or set(raw) != {"contract", "columns"}:
        raise ConfigError(f"{p}: expected top-level keys 'contract' and 'columns'")
    meta, cols = raw["contract"], raw["columns"]
    problems: list[str] = []
    out: list[ColumnContract] = []
    for name, d in cols.items():
        d = dict(d or {})
        missing = [k for k in ("sql", "nullable", "role", "sem", "model", "timing") if k not in d]
        if missing:
            problems.append(f"{name}: missing keys {missing}")
            continue
        unknown = sorted(set(d) - {"sql", "nullable", "role", "sem", "model", "timing", "allowed", "null_means", "group", "note",
                                   "sentinels", "sentinel_means", "sentinel_modeling_rule"})
        if unknown:
            problems.append(f"{name}: unknown keys {unknown}")
        c = ColumnContract(name=str(name), sql=str(d["sql"]), nullable=bool(d["nullable"]), role=str(d["role"]), semantic=str(d["sem"]),
                           model_dtype=str(d["model"]), timing=str(d["timing"]),
                           allowed=tuple(d["allowed"]) if d.get("allowed") is not None else None,
                           null_means=None if d.get("null_means") is None else str(d["null_means"]),
                           group=None if d.get("group") is None else str(d["group"]), note=str(d.get("note", "")),
                           sentinels=tuple(str(v) for v in (d.get("sentinels") or [])),
                           sentinel_means=None if d.get("sentinel_means") is None else str(d["sentinel_means"]),
                           sentinel_modeling_rule=str(d.get("sentinel_modeling_rule", "keep")))
        if c.role not in ROLES:
            problems.append(f"{name}: role {c.role!r} not in {ROLES}")
        if c.semantic not in SEMANTIC_TYPES:
            problems.append(f"{name}: sem {c.semantic!r} not in {SEMANTIC_TYPES}")
        if c.model_dtype not in MODEL_DTYPES:
            problems.append(f"{name}: model {c.model_dtype!r} not in {MODEL_DTYPES}")
        if c.timing not in TIMINGS:
            problems.append(f"{name}: timing {c.timing!r} not in {TIMINGS}")
        if c.role not in PREDICTOR_ROLES and c.model_dtype != "never":
            problems.append(f"{name}: role {c.role} can never enter a model matrix (model must be 'never')")
        if c.role in PREDICTOR_ROLES and c.model_dtype != "never" and c.timing not in PREDICTOR_TIMINGS | {"unknown"}:
            problems.append(f"{name}: predictor with timing {c.timing} (post-index knowledge) is not allowed")
        if c.role == "MEUHEDET_ENHANCED_FEATURE" and not c.group:
            problems.append(f"{name}: MEUHEDET_ENHANCED_FEATURE needs a Phase-2 group")
        if c.semantic == "binary" and c.allowed is not None and set(c.allowed) - {0, 1}:
            problems.append(f"{name}: binary column allows {list(c.allowed)}")
        if c.sentinels:
            if not c.is_date:
                problems.append(f"{name}: sentinels are only allowed on date/datetime columns")
            for v in c.sentinels:
                try:
                    date.fromisoformat(v)
                except ValueError:
                    problems.append(f"{name}: sentinel {v!r} is not an ISO date (YYYY-MM-DD)")
            if not c.sentinel_means:
                problems.append(f"{name}: sentinels need sentinel_means (their documented business meaning)")
        elif d.get("sentinel_means") is not None or "sentinel_modeling_rule" in d:
            problems.append(f"{name}: sentinel_means / sentinel_modeling_rule without sentinels")
        if c.sentinel_modeling_rule not in SENTINEL_MODELING_RULES:
            problems.append(f"{name}: sentinel_modeling_rule {c.sentinel_modeling_rule!r} not in {SENTINEL_MODELING_RULES}")
        out.append(c)
    declared = meta["date_formats"] if "date_formats" in meta else DEFAULT_DATE_FORMATS   # declared [] is an error, not "use the default"
    date_formats = tuple(str(f) for f in (declared if isinstance(declared, (list, tuple)) else [declared]))
    problems.extend(check_date_formats(date_formats))
    names = [c.name for c in out]
    if int(meta.get("n_columns", len(names))) != len(names):
        problems.append(f"contract.n_columns={meta.get('n_columns')} but {len(names)} columns are declared")
    if problems:
        raise ConfigError(f"Invalid wide-table contract {p}: " + "; ".join(problems[:20]))
    return WideContract(name=str(meta["name"]), version=str(meta["version"]), source_view=str(meta.get("source_view", "")),
                        definition_version=str(meta.get("definition_version", "")), date_formats=date_formats, columns=tuple(out),
                        content_sha256=_sha256_json(raw), raw=raw)


# ============================================================================ extract reading and contract validation
@dataclass
class ContractReport:
    n_rows: int
    n_columns: int
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    observed_dtypes: dict[str, str] = field(default_factory=dict)
    wrong_type: dict[str, int] = field(default_factory=dict)          # values that could not be coerced to the contract type
    null_counts: dict[str, int] = field(default_factory=dict)
    invalid_values: dict[str, int] = field(default_factory=dict)      # outside `allowed`
    non_null_violations: dict[str, int] = field(default_factory=dict)  # NULL in a DDL NOT NULL column
    wrong_type_examples: dict[str, list[str]] = field(default_factory=dict)  # date columns: the offending raw values (dates are never identifiers)
    date_range: dict[str, dict[str, Any]] = field(default_factory=dict)    # date columns holding values beyond the nanosecond range
    date_formats: dict[str, dict[str, int]] = field(default_factory=dict)  # date column -> {declared source format: values read with it}

    @property
    def wrong_type_total(self) -> int:
        return int(sum(self.wrong_type.values()))

    def to_dict(self) -> dict[str, Any]:
        return {"n_rows": self.n_rows, "n_columns": self.n_columns, "problems": list(self.problems), "warnings": list(self.warnings),
                "wrong_type_total": self.wrong_type_total, "wrong_type": dict(self.wrong_type), "invalid_values": dict(self.invalid_values),
                "non_null_violations": dict(self.non_null_violations), "wrong_type_examples": {k: list(v) for k, v in self.wrong_type_examples.items()},
                "date_range": {k: dict(v) for k, v in self.date_range.items()}, "date_formats": {k: dict(v) for k, v in self.date_formats.items()}}


def _parse_date_cell(v: Any, formats: Iterable[str]) -> tuple[datetime | None, str | None]:
    """One raw cell -> (naive python datetime in the SQL date range 0001..9999, the declared format that read it) or (None, None)
    for NULL. Text is read only with the formats declared in the contract, tried in order; the first that fits wins and the loader
    guarantees no two of them read the same text differently, so nothing is inferred: 01/02/2024 under %d/%m/%Y is always 1 February.
    A value no declared format fits raises ValueError - no day/month guessing, no Excel serial numbers, nothing repaired."""
    if v is None or v is pd.NaT or v is pd.NA or (isinstance(v, float) and np.isnan(v)):
        return None, None
    if isinstance(v, np.datetime64):
        return (None, None) if np.isnat(v) else (v.astype("datetime64[us]").item(), NATIVE_DATES)
    if isinstance(v, datetime):  # pd.Timestamp included
        return (v.replace(tzinfo=None) if v.tzinfo is not None else v), NATIVE_DATES
    if isinstance(v, date):
        return datetime(v.year, v.month, v.day), NATIVE_DATES
    if not isinstance(v, str):
        raise ValueError(f"cell of type {type(v).__name__} is not a date")
    txt = _FRACTION_RE.sub(r"\1", v.strip())
    if not txt:
        return None, None
    date_text, time_text = txt, ""
    for sep in ("T", " "):
        if sep in txt:
            date_text, _, time_text = txt.partition(sep)
            break
    time_text = _TZ_RE.sub("", time_text.strip()).strip()   # an offset is dropped, never shifted (the VIEW holds local dates)
    h = m = sec = micro = 0
    if time_text:
        parts = _TIME_RE.match(time_text)
        if parts is None:
            raise ValueError(f"{time_text!r} is not a HH:MM[:SS[.ffffff]] time")
        h, m = int(parts.group(1)), int(parts.group(2))
        sec = int(parts.group(3) or 0)
        micro = int((parts.group(4) or "").ljust(6, "0") or 0)
    for fmt in formats:
        try:
            d = datetime.strptime(date_text.strip(), fmt)
        except ValueError:
            continue
        return d.replace(hour=h, minute=m, second=sec, microsecond=micro), fmt
    raise ValueError(f"{date_text!r} does not match any declared date format {list(formats)}")


def _coerce_date_column(s: pd.Series, formats: Iterable[str]) -> tuple[pd.Series, pd.Series, list[str], dict[str, int]]:
    """Contract-driven date parsing -> (datetime64[us] series covering the full SQL date range, mask of observed values that could
    not be parsed, up to five distinct offending values as text, how many values each declared format read). Each distinct raw
    value is parsed once."""
    formats = list(formats)
    if pd.api.types.is_datetime64_any_dtype(s):
        parsed = s.dt.tz_localize(None) if getattr(s.dt, "tz", None) is not None else s
        used = {NATIVE_DATES: int(parsed.notna().sum())} if parsed.notna().any() else {}
        return parsed.astype(DATE_DTYPE), pd.Series(False, index=s.index), [], used
    codes, uniques = pd.factorize(s, use_na_sentinel=True)
    n_u = max(len(uniques), 1)
    parsed_u = np.full(n_u, np.datetime64("NaT", "us"), dtype="datetime64[us]")
    fmt_u: list[str | None] = [None] * n_u
    bad_u = np.zeros(n_u, dtype=bool)
    examples: list[str] = []
    for i, v in enumerate(uniques):
        try:
            d, fmt = _parse_date_cell(v, formats)
        except (ValueError, TypeError, OverflowError):
            bad_u[i] = True
            if len(examples) < 5:
                examples.append(str(v))
            continue
        if d is not None:
            parsed_u[i] = np.datetime64(d, "us")
            fmt_u[i] = fmt
    observed = codes >= 0
    pos = np.where(observed, codes, 0)
    out = np.where(observed, parsed_u[pos], np.datetime64("NaT", "us")).astype("datetime64[us]")
    bad = observed & bad_u[pos]
    used: dict[str, int] = {}
    if observed.any():
        seen = pd.Series([fmt_u[i] for i in pos[observed]], dtype="object")
        used = {str(k): int(v) for k, v in seen.dropna().value_counts().items()}
    return pd.Series(out, index=s.index), pd.Series(bad, index=s.index), examples, used


def date_range_report(s: pd.Series, c: ColumnContract) -> dict[str, Any]:
    """Aggregate date audit of one contract-typed date column: observed min/max, the values beyond the nanosecond range (which
    the modelling layer cannot hold) split into declared sentinels and undeclared far-future dates. Values are dates, never ids."""
    arr = s.to_numpy(dtype="datetime64[us]")
    valid = ~np.isnat(arr)
    days = arr.astype("datetime64[D]")
    beyond = valid & ((arr < NS_MIN_DAY) | (arr > NS_MAX_DAY))
    vals, counts = np.unique(days[beyond], return_counts=True)
    found = {str(v): int(n) for v, n in zip(vals, counts)}
    declared = {v: int((days == np.datetime64(v, "D")).sum()) for v in c.sentinels}
    return {"dtype": DATE_DTYPE, "n_observed": int(valid.sum()), "min": str(days[valid].min()) if valid.any() else None,
            "max": str(days[valid].max()) if valid.any() else None, "n_beyond_ns_range": int(beyond.sum()), "beyond_ns_range_values": found,
            "undeclared_beyond_ns_range_values": {k: n for k, n in found.items() if k not in set(c.sentinels)},
            "declared_sentinels": declared, "sentinel_means": c.sentinel_means,
            "sentinel_modeling_rule": c.sentinel_modeling_rule if c.sentinels else None}


@dataclass
class CoercionReport:
    wrong_type: dict[str, int] = field(default_factory=dict)                  # values that could not be represented (now NULL)
    wrong_type_examples: dict[str, list[str]] = field(default_factory=dict)   # date columns only: offending raw values (never identifiers)
    date_range: dict[str, dict[str, Any]] = field(default_factory=dict)      # every date column with observed values
    date_formats: dict[str, dict[str, int]] = field(default_factory=dict)    # date column -> {declared format: values read with it}


def coerce_wide_types_report(raw: pd.DataFrame, contract: WideContract) -> tuple[pd.DataFrame, CoercionReport]:
    """Cast every contract column to its declared pandas dtype (no inference). Dates are read with the contract's declared source
    formats (``contract.date_formats``, e.g. %Y-%m-%d and %d/%m/%Y) into ``datetime64[us]`` (the whole SQL date range, so
    2999-12-31 stays 2999-12-31); values that cannot be represented become NULL and are counted as wrong_type, with the offending
    raw values and the format that read each column reported."""
    out = pd.DataFrame(index=raw.index)
    rep = CoercionReport()
    for c in contract.columns:
        if c.name not in raw.columns:
            continue
        s = raw[c.name]
        observed = s.notna()
        if c.is_integer:
            if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
                txt = s.astype("string").str.strip()
                txt = txt.replace({"True": "1", "False": "0", "true": "1", "false": "0"})
                num = pd.to_numeric(txt, errors="coerce")
            else:
                num = pd.to_numeric(s, errors="coerce")
            num = num.astype("Float64")
            bad = observed & (num.isna() | (num != num.round()))
            out[c.name] = num.where(~bad).round().astype("Int64")
        elif c.is_date:
            parsed, bad, examples, used = _coerce_date_column(s, contract.date_formats)
            out[c.name] = parsed
            if examples:
                rep.wrong_type_examples[c.name] = examples
            if used:
                rep.date_formats[c.name] = used
            if parsed.notna().any():
                rep.date_range[c.name] = date_range_report(parsed, c)
        elif c.pandas_dtype == "float64":
            num = pd.to_numeric(s if pd.api.types.is_numeric_dtype(s) else s.astype("string").str.strip(), errors="coerce")
            bad = observed & num.isna()
            out[c.name] = num.astype("float64")
        else:
            txt = s.astype("string").str.strip()
            txt = txt.mask(txt == "")
            bad = pd.Series(False, index=raw.index)
            out[c.name] = txt
        if int(bad.sum()):
            rep.wrong_type[c.name] = int(bad.sum())
    for extra in [c for c in raw.columns if c not in out.columns]:
        out[extra] = raw[extra]
    return out, rep


def coerce_wide_types(raw: pd.DataFrame, contract: WideContract) -> tuple[pd.DataFrame, dict[str, int]]:
    """:func:`coerce_wide_types_report` returning only the frame and the per-column wrong-type counts."""
    frame, rep = coerce_wide_types_report(raw, contract)
    return frame, rep.wrong_type


@dataclass
class ExtractRead:
    """An extract cast to the contract types plus everything the reader had to decide (no inference is trusted)."""

    frame: pd.DataFrame
    wrong_type: dict[str, int]
    source_format: str
    sheet: str | None = None
    cell_types: dict[str, dict[str, int]] = field(default_factory=dict)   # Excel/object columns: python cell types seen
    conversions: dict[str, dict[str, str]] = field(default_factory=dict)  # column -> {source_dtype, contract_dtype}
    warnings: list[str] = field(default_factory=list)
    wrong_type_examples: dict[str, list[str]] = field(default_factory=dict)  # date columns: offending raw values (never identifiers)
    date_range: dict[str, dict[str, Any]] = field(default_factory=dict)      # per date column: min/max, beyond-ns-range values, sentinels
    date_formats: dict[str, dict[str, int]] = field(default_factory=dict)    # per date column: which declared source format read the values
    declared_date_formats: tuple[str, ...] = ()                              # the contract's source date formats, in trial order

    def to_dict(self) -> dict[str, Any]:
        return {"source_format": self.source_format, "sheet": self.sheet, "n_rows": len(self.frame), "wrong_type": dict(self.wrong_type),
                "wrong_type_examples": {k: list(v) for k, v in self.wrong_type_examples.items()},
                "cell_types": {k: dict(v) for k, v in self.cell_types.items()}, "conversions": {k: dict(v) for k, v in self.conversions.items()},
                "date_range": {k: dict(v) for k, v in self.date_range.items()},
                "declared_date_formats": list(self.declared_date_formats), "date_formats": {k: dict(v) for k, v in self.date_formats.items()},
                "warnings": list(self.warnings)}


EXCEL_SUFFIXES = frozenset({".xlsx", ".xlsm", ".xls"})


def _excel_cell_types(raw: pd.DataFrame) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for c in raw.columns:
        s = raw[c]
        if s.dtype == object:
            counts = Counter(type(v).__name__ for v in s if v is not None and not (isinstance(v, float) and np.isnan(v)))
            out[str(c)] = dict(counts)
    return out


def _normalise_object_cells(raw: pd.DataFrame, contract: WideContract) -> pd.DataFrame:
    """Excel gives object columns with mixed python types. Text columns: integral numbers become their integer text (a '1' stored
    as the number 1), datetimes their ISO date; identifiers are checked before this (numbers there are an error, never repaired)."""
    out = raw.copy()
    for c in contract.columns:
        if c.name not in out.columns or out[c.name].dtype != object or not c.is_text:
            continue
        def _fix(v: Any) -> Any:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return None
            if isinstance(v, bool):
                return str(int(v))
            if isinstance(v, (int, np.integer)):
                return str(int(v))
            if isinstance(v, (float, np.floating)):
                return str(int(v)) if float(v).is_integer() else str(v)
            if hasattr(v, "isoformat"):
                return v.isoformat()[:10]
            return str(v)
        out[c.name] = out[c.name].map(_fix)
    return out


def read_wide_extract_report(path: str | Path, contract: WideContract, *, encoding: str = "utf-8", sep: str = ",",
                             sheet: str | None = None) -> ExtractRead:
    """Read one local extract file (.xlsx first sheet by default, .parquet, or .csv with 'NULL' literals) and cast it to the
    contract types. Excel cell types are never trusted: every column is re-typed from the contract and every decision reported;
    date text is read only with the contract's declared formats (``contract.date_formats``), so 01/02/2024 has one meaning.
    An identifier stored by Excel as a number (lost leading zeros / scientific notation) fails loudly and is never repaired."""
    p = Path(path)
    if not p.is_file():
        raise DatasetValidationError(f"Extract file not found: {p}")
    suffix = p.suffix.lower()
    cell_types: dict[str, dict[str, int]] = {}
    sheet_used: str | None = None
    if suffix == ".parquet":
        raw = pd.read_parquet(p)
        fmt = "parquet"
    elif suffix in {".csv", ".txt", ".tsv"}:
        try:
            raw = pd.read_csv(p, dtype="string", na_values=list(NA_VALUES), keep_default_na=False, encoding=encoding, sep=sep)
        except UnicodeDecodeError as exc:
            raise DatasetValidationError(f"{p}: not {encoding}-encoded ({exc}); SSMS exports are often UTF-16 or cp1255 - pass --encoding") from exc
        fmt = "csv"
    elif suffix in EXCEL_SUFFIXES:
        try:
            import openpyxl  # noqa: F401  (engine; listed in requirements.lock)
        except ImportError as exc:
            raise DatasetValidationError("Reading Excel needs the openpyxl package (in requirements.lock); re-run the installer") from exc
        target: int | str = 0 if sheet is None else (int(sheet) if str(sheet).isdigit() else str(sheet))
        try:
            raw = pd.read_excel(p, sheet_name=target, dtype=object, na_values=list(NA_VALUES), keep_default_na=False, engine="openpyxl")
        except ValueError as exc:
            raise DatasetValidationError(f"{p}: cannot read sheet {target!r} ({exc})") from exc
        sheet_used = str(target)
        fmt = "excel"
        raw.columns = [str(c).strip() for c in raw.columns]
        cell_types = _excel_cell_types(raw)
        problems = []
        for c in contract.columns:
            if c.role == "IDENTIFIER" and c.name in raw.columns:
                numeric = sum(n for t, n in cell_types.get(c.name, {}).items() if t in ("int", "float", "Decimal"))
                if numeric:
                    problems.append(f"{c.name}: {numeric} of {int(raw[c.name].notna().sum())} cells are numbers - Excel converted the identifier "
                                    "(leading zeros / scientific notation are lost); export the column as text and re-run. Not repaired.")
        if problems:
            raise DatasetValidationError(f"{p.name}: identifier columns were altered by Excel", problems)
        raw = _normalise_object_cells(raw, contract)
    else:
        raise DatasetValidationError(f"{p}: unsupported extract format {p.suffix!r} (use .xlsx, .parquet or .csv)")
    raw.columns = [str(c).strip() for c in raw.columns]
    source_dtypes = {str(c): str(raw[c].dtype) for c in raw.columns}
    frame, coerced = coerce_wide_types_report(raw, contract)
    conversions = {c.name: {"source_dtype": source_dtypes[c.name], "excel_cell_types": cell_types.get(c.name), "contract_dtype": c.pandas_dtype,
                            "sql_type": c.sql} for c in contract.columns if c.name in frame.columns}
    return ExtractRead(frame=frame, wrong_type=coerced.wrong_type, source_format=fmt, sheet=sheet_used, cell_types=cell_types, conversions=conversions,
                       wrong_type_examples=coerced.wrong_type_examples, date_range=coerced.date_range, date_formats=coerced.date_formats,
                       declared_date_formats=tuple(contract.date_formats))


def read_wide_extract(path: str | Path, contract: WideContract, *, encoding: str = "utf-8", sep: str = ",",
                      sheet: str | None = None) -> tuple[pd.DataFrame, dict[str, int]]:
    """Read one local extract file (.xlsx / .parquet / .csv) and cast it to the contract types; see :func:`read_wide_extract_report`."""
    r = read_wide_extract_report(path, contract, encoding=encoding, sep=sep, sheet=sheet)
    return r.frame, r.wrong_type


def validate_wide_contract(df: pd.DataFrame, contract: WideContract, *, wrong_type: Mapping[str, int] | None = None,
                           wrong_type_examples: Mapping[str, Iterable[str]] | None = None,
                           date_formats: Mapping[str, Mapping[str, int]] | None = None, strict_columns: Iterable[str] = (),
                           strict: bool = True) -> ContractReport:
    """Check ``df`` against the contract: exact column set, coercion failures, DDL NOT NULL, allowed values, VIEW constants.

    Violations in ``strict_columns`` (columns the build reads) are problems; elsewhere they are warnings with counts.
    With ``strict`` a problem raises ``DatasetValidationError`` listing everything at once."""
    rep = ContractReport(n_rows=len(df), n_columns=len(df.columns))
    cols = list(df.columns)
    missing = [c for c in contract.names if c not in cols]
    extra = [c for c in cols if c not in set(contract.names)]
    if missing:
        rep.problems.append(f"missing contract columns: {missing[:15]}{' ...' if len(missing) > 15 else ''}")
    if extra:
        rep.problems.append(f"columns not in the contract (the VIEW changed? update the contract explicitly): {extra[:15]}")
    dup = sorted({c for c in cols if cols.count(c) > 1})
    if dup:
        rep.problems.append(f"duplicate columns {dup}")
    strict_set = set(strict_columns)
    wrong_type = dict(wrong_type or {})
    for c in contract.columns:
        if c.name not in df.columns:
            continue
        s = df[c.name]
        rep.observed_dtypes[c.name] = str(s.dtype)
        n_null = int(s.isna().sum())
        rep.null_counts[c.name] = n_null
        if c.name in wrong_type:
            rep.wrong_type[c.name] = int(wrong_type[c.name])
            msg = f"{c.name}: {wrong_type[c.name]} values cannot be represented as {c.sql} (now NULL)"
            examples = list((wrong_type_examples or {}).get(c.name) or [])
            if examples and c.semantic != "identifier":
                rep.wrong_type_examples[c.name] = examples
                expected = (f"one of the declared source date formats {list(contract.date_formats)}" if c.is_date
                            else f"semantic type {c.semantic}")
                msg += f"; expected {expected}; offending values e.g. {examples}"
            (rep.problems if c.name in strict_set else rep.warnings).append(msg)
        if c.is_date and s.notna().any():
            used = dict((date_formats or {}).get(c.name) or {})
            if used:
                rep.date_formats[c.name] = used
                text_formats = [f for f in used if f != NATIVE_DATES]
                if len(text_formats) > 1:
                    rep.warnings.append(f"{c.name}: the extract mixes source date layouts {used}; each value was read with the declared "
                                        f"format that fits it (no guessing), but a single export should use one layout")
            dr = date_range_report(s, c)
            if dr["n_beyond_ns_range"]:
                rep.date_range[c.name] = dr
                undeclared = dr["undeclared_beyond_ns_range_values"]
                if undeclared:
                    msg = (f"{c.name}: {sum(undeclared.values())} dates beyond the pandas nanosecond range ({NS_RANGE_TEXT}) that are not declared "
                           f"sentinels: {undeclared} (semantic type {c.semantic}, SQL {c.sql}); kept unchanged in the contract layer ({DATE_DTYPE}); "
                           "declare `sentinels` + `sentinel_means` for this column in the contract if these are business sentinels")
                    (rep.problems if c.name in strict_set else rep.warnings).append(msg)
        if not c.nullable and n_null:
            rep.non_null_violations[c.name] = n_null
            msg = f"{c.name}: {n_null} NULL values in a DDL NOT NULL column"
            (rep.problems if c.name in strict_set else rep.warnings).append(msg)
        if c.allowed is not None and s.notna().any():
            vals = s.dropna()
            if c.is_integer:
                allowed = {int(v) for v in c.allowed}
                bad = ~vals.astype("int64").isin(allowed)
            elif c.pandas_dtype == "float64":
                bad = ~vals.astype("float64").isin({float(v) for v in c.allowed})
            else:
                allowed_s = {str(v) for v in c.allowed}
                bad = ~vals.astype("string").isin(allowed_s)
            n_bad = int(bad.sum())
            if n_bad:
                rep.invalid_values[c.name] = n_bad
                examples = sorted(map(str, pd.unique(vals[bad.to_numpy()])))[:5]
                msg = f"{c.name}: {n_bad} values outside allowed {list(c.allowed)} (e.g. {examples})"
                (rep.problems if c.name in strict_set else rep.warnings).append(msg)
    if "Leakage_Check_Ind" in df.columns:
        n_leak = int((df["Leakage_Check_Ind"].fillna(0) != 0).sum())
        if n_leak:
            rep.problems.append(f"Leakage_Check_Ind: {n_leak} rows flagged by the VIEW's own leakage check (VIEW bug)")
    if "Definition_Version" in df.columns and contract.definition_version:
        seen = sorted(map(str, pd.unique(df["Definition_Version"].dropna())))
        if seen and seen != [contract.definition_version]:
            rep.problems.append(f"Definition_Version {seen} differs from the contract ({contract.definition_version})")
    if strict and rep.problems:
        raise DatasetValidationError(f"Extract violates the wide-table contract {contract.name} {contract.version} "
                                     f"({len(rep.problems)} problems)", rep.problems)
    return rep


# ============================================================================ mapping manifest
@dataclass(frozen=True)
class FeatureMapping:
    canonical: str
    status: str
    quality: str
    source_columns: tuple[str, ...]
    include_in_baseline: bool
    op: str | None = None
    codes: Mapping[Any, Any] | None = None
    null_is_absent: bool = False
    absent_indicator: str | None = None
    record_date_column: str | None = None    # "last record" date of the event source behind the predictor (D-00 guard / tracing)
    same_day_evidence_column: str | None = None   # date column proving an index-day record of the source (sufficient only; SAFE-ALL-ROWS scope)
    raw_dtype: str = ""
    canonical_dtype: str = ""
    model_dtype: str = ""
    missing_rule: str = ""
    transformation: str = ""
    lookback_window: str = ""
    allowed_values: Any = None
    notes: str = ""
    phase2_candidate: tuple[str, ...] = ()
    efalls: Mapping[str, Any] = field(default_factory=dict, compare=False, hash=False)

    @property
    def available(self) -> bool:
        return self.status != "UNAVAILABLE"


@dataclass(frozen=True)
class WideMapping:
    name: str
    version: str
    feature_spec_path: str
    exploratory_spec_path: str
    contract_path: str
    status: str
    clinically_validated: bool
    identity: Mapping[str, str]
    cohort: Mapping[str, Any]
    outcome: Mapping[str, Any]
    metadata: Mapping[str, str]
    open_questions: Mapping[str, str]
    features: tuple[FeatureMapping, ...]
    content_sha256: str
    spec_sha256: str
    raw: Mapping[str, Any] = field(default_factory=dict, compare=False, hash=False, repr=False)

    def get(self, name: str) -> FeatureMapping:
        for f in self.features:
            if f.canonical == name:
                return f
        raise ConfigError(f"{name!r} is not in mapping {self.name} {self.version}")

    @property
    def baseline_features(self) -> list[str]:
        return [f.canonical for f in self.features if f.include_in_baseline]

    def feature_sets(self) -> dict[str, list[str]]:
        """Exploratory feature sets derived from the manifest (never hard-coded): ``strict`` = included HIGH_CONFIDENCE (and EXACT)
        mappings only; ``extended`` = strict + APPROXIMATE. Spec order is preserved."""
        strict = [f.canonical for f in self.features if f.include_in_baseline and f.quality in ("EXACT", "HIGH_CONFIDENCE")]
        extended = [f.canonical for f in self.features if f.include_in_baseline and f.quality in ("EXACT", "HIGH_CONFIDENCE", "APPROXIMATE")]
        return {"strict": strict, "extended": extended}

    def coverage(self, spec: FeatureSpec) -> dict[str, Any]:
        """eFalls coverage of the baseline; the denominator is derived from ``spec`` (exact_efalls_baseline predictors)."""
        candidates = [f.name for f in spec.features if f.exact_efalls_baseline]
        retained = {f.name for f in spec.features if f.exact_efalls_baseline and f.retained_in_published_model is True}
        mandatory = list(spec.cohort.get("mandatory_for_efalls_label") or [])
        by_name = {f.canonical: f for f in self.features}
        available = [c for c in candidates if by_name[c].include_in_baseline]
        unavailable = [c for c in candidates if not by_name[c].include_in_baseline]
        q = {lvl: [c for c in available if by_name[c].quality == lvl] for lvl in MAPPING_QUALITY if lvl != "UNAVAILABLE"}
        return {"feature_spec": spec.name, "feature_spec_version": spec.version, "feature_spec_sha256": spec.content_sha256,
                "n_expected": len(candidates), "n_available": len(available),
                "coverage_pct": round(100.0 * len(available) / len(candidates), 1) if candidates else 0.0,
                "n_exact": len(q["EXACT"]), "n_high_confidence": len(q["HIGH_CONFIDENCE"]), "n_approximate": len(q["APPROXIMATE"]),
                "n_unavailable": len(unavailable), "available": available, "exact": q["EXACT"], "high_confidence": q["HIGH_CONFIDENCE"],
                "approximate": q["APPROXIMATE"], "unavailable": unavailable,
                "mandatory": mandatory, "mandatory_unavailable": [m for m in mandatory if m not in set(available)],
                "n_published_retained_available": sum(c in retained for c in available), "n_published_retained_total": len(retained),
                "label": "Reduced eFalls-compatible feature set - NOT exact eFalls reproduction"}


def feature_input_columns(f: FeatureMapping) -> dict[str, tuple[str, ...]]:
    """The extract columns the adapter reads for one feature, by use (the single source of truth for the builder and the D-00 dependency graph):
    ``value`` - the columns the canonical value is computed from (``MeuhedetWideDatasetAdapter._feature`` reads exactly these);
    ``validation`` - read only to check / count explicit NULL handling (the absent indicator), never part of the value;
    ``row_evidence`` - record-date / same-day evidence columns that can exclude a row under a D-00 guard scope (never part of the value)."""
    return {"value": tuple(f.source_columns), "validation": (f.absent_indicator,) if f.absent_indicator else (),
            "row_evidence": tuple(c for c in (f.record_date_column, f.same_day_evidence_column) if c)}


PREDICTION_TIME = "START_OF_INDEX_DAY"
#: D-00 guard scopes of a build: "cohort" = the mapping's cohort guard columns (one shared cohort for STRICT / EXTENDED);
#: "built_predictors" = the record_date_column / same_day_evidence_column of the predictors actually built (SAFE-ALL-ROWS sensitivity)
TIMING_SCOPES = ("cohort", "built_predictors")
PREDICTOR_RECORD_RULE = "source_event_date < Index_Date"
_FEATURE_KEYS = frozenset({"mapping_status", "mapping_quality", "source_columns", "op", "codes", "null_is_absent", "absent_indicator", "record_date_column",
                           "same_day_evidence_column",
                           "raw_dtype", "canonical_dtype", "model_dtype", "missing_rule", "transformation", "lookback_window",
                           "allowed_values", "include_in_baseline", "notes", "phase2_candidate"})


def load_wide_mapping(path: str | Path = DEFAULT_MAPPING, *, spec: FeatureSpec | None = None,
                      contract: WideContract | None = None) -> WideMapping:
    """Load the mapping manifest and check it against the feature spec (denominator) and the column contract."""
    from falls_ml.paths import resolve_path

    p = resolve_path(path)
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Mapping manifest not found: {p}") from exc
    for key in ("mapping", "identity", "cohort", "outcome", "metadata", "features"):
        if key not in raw:
            raise ConfigError(f"{p}: missing top-level key {key!r}")
    meta = raw["mapping"]
    spec = spec or load_feature_spec(meta["feature_spec"])
    contract = contract or load_wide_contract(meta.get("contract", DEFAULT_CONTRACT))
    by_spec = {f.name: f for f in spec.features if f.exact_efalls_baseline}
    problems: list[str] = []
    feats: list[FeatureMapping] = []
    for name, d in raw["features"].items():
        d = dict(d or {})
        unknown = sorted(set(d) - _FEATURE_KEYS)
        if unknown:
            problems.append(f"{name}: unknown keys {unknown}")
        if name not in by_spec:
            problems.append(f"{name}: not an eFalls candidate of {spec.name} {spec.version}")
            continue
        sd = by_spec[name]
        status, quality = str(d.get("mapping_status")), str(d.get("mapping_quality"))
        sources = tuple(str(c) for c in (d.get("source_columns") or ()))
        include = bool(d.get("include_in_baseline", False))
        if status not in MAPPING_STATUS or quality not in MAPPING_QUALITY:
            problems.append(f"{name}: mapping_status {status!r} / mapping_quality {quality!r} invalid")
        if (status == "UNAVAILABLE") != (quality == "UNAVAILABLE"):
            problems.append(f"{name}: status and quality disagree on availability")
        if status == "UNAVAILABLE" and (sources or include):
            problems.append(f"{name}: UNAVAILABLE features cannot have source columns or be included")
        if include and status == "UNAVAILABLE":
            problems.append(f"{name}: included but unavailable")
        op = d.get("op")
        if status != "UNAVAILABLE":
            if op not in OPS:
                problems.append(f"{name}: op {op!r} not in {OPS}")
            if not sources:
                problems.append(f"{name}: available mapping without source columns")
            if str(d.get("missing_rule")) != sd.missing_rule:
                problems.append(f"{name}: missing_rule {d.get('missing_rule')!r} differs from the spec rule {sd.missing_rule!r}")
            if op == "recode":
                targets = {str(v) for v in (d.get("codes") or {}).values()}
                if not targets or not targets <= set(sd.levels):
                    problems.append(f"{name}: recode targets {sorted(targets)} must be spec levels {list(sd.levels)}")
            if op == "count" and not d.get("absent_indicator"):
                problems.append(f"{name}: op count requires absent_indicator (explicit absent_is_zero)")
            if sd.dtype == "binary" and op != "any_positive":
                problems.append(f"{name}: binary eFalls predictors use op any_positive")
        for c in sources:
            try:
                cc = contract.get(c)
            except ConfigError:
                problems.append(f"{name}: source column {c} not in the contract")
                continue
            if cc.role != "EFALLS_BASELINE_FEATURE" or cc.timing not in PREDICTOR_TIMINGS:
                problems.append(f"{name}: source column {c} has role {cc.role} / timing {cc.timing} (needs EFALLS_BASELINE_FEATURE, pre/at index)")
        ind = d.get("absent_indicator")
        if ind and (ind not in contract.names or contract.get(ind).role != "QA_CONTROL"):
            problems.append(f"{name}: absent_indicator {ind} must be a QA_CONTROL contract column")
        rdc = d.get("record_date_column")
        if rdc is not None:
            if rdc not in contract.names or not contract.get(rdc).is_date or contract.get(rdc).timing != "pre_index":
                problems.append(f"{name}: record_date_column {rdc} must be a pre_index date column of the contract")
            if status == "UNAVAILABLE":
                problems.append(f"{name}: record_date_column on an UNAVAILABLE mapping")
        sde = d.get("same_day_evidence_column")
        if sde is not None:
            if sde not in contract.names or not contract.get(sde).is_date or contract.get(sde).timing not in PREDICTOR_TIMINGS:
                problems.append(f"{name}: same_day_evidence_column {sde} must be a pre/at-index date column of the contract")
            if status == "UNAVAILABLE":
                problems.append(f"{name}: same_day_evidence_column on an UNAVAILABLE mapping")
        for c in d.get("phase2_candidate") or ():
            if c not in contract.names or contract.get(c).role != "MEUHEDET_ENHANCED_FEATURE":
                problems.append(f"{name}: phase2_candidate {c} must be a MEUHEDET_ENHANCED_FEATURE contract column")
        tw = sd.raw.get("time_window") or {}
        feats.append(FeatureMapping(
            canonical=name, status=status, quality=quality, source_columns=sources, include_in_baseline=include, op=op,
            codes=dict(d["codes"]) if d.get("codes") else None, null_is_absent=bool(d.get("null_is_absent", False)),
            absent_indicator=None if ind is None else str(ind), record_date_column=None if rdc is None else str(rdc),
            same_day_evidence_column=None if sde is None else str(sde), raw_dtype=str(d.get("raw_dtype", "")),
            canonical_dtype=str(d.get("canonical_dtype", "")), model_dtype=str(d.get("model_dtype", "")),
            missing_rule=str(d.get("missing_rule", sd.missing_rule)), transformation=str(d.get("transformation", "")),
            lookback_window=str(d.get("lookback_window", "")), allowed_values=d.get("allowed_values"), notes=str(d.get("notes", "")),
            phase2_candidate=tuple(str(c) for c in (d.get("phase2_candidate") or ())),
            efalls={"concept": sd.concept, "dtype": sd.dtype, "missing_rule": sd.missing_rule, "efalls_term": sd.efalls_term,
                    "retained_in_published_model": sd.retained_in_published_model, "window": (tw.get("window") or tw.get("type")),
                    "non_code_rule": tw.get("non_code_rule"), "efalls_source": sd.raw.get("efalls_source"),
                    "mandatory": name in (spec.cohort.get("mandatory_for_efalls_label") or [])}))
    missing = sorted(set(by_spec) - {f.canonical for f in feats})
    if missing:
        problems.append(f"eFalls candidates without a manifest row: {missing}")
    # the baseline source columns are exactly the contract's EFALLS_BASELINE_FEATURE columns (no orphan, no silent use)
    used = {c for f in feats if f.include_in_baseline for c in f.source_columns}
    declared = {c.name for c in contract.by_role("EFALLS_BASELINE_FEATURE")}
    if used != declared:
        problems.append(f"contract EFALLS_BASELINE_FEATURE columns {sorted(declared ^ used)} are not exactly the columns used by included mappings")
    cohort = raw["cohort"]
    if cohort.get("prediction_time") != PREDICTION_TIME or cohort.get("predictor_record_rule") != PREDICTOR_RECORD_RULE:
        problems.append(f"cohort: prediction_time must be {PREDICTION_TIME!r} and predictor_record_rule {PREDICTOR_RECORD_RULE!r} "
                        "(D-00 start-of-index-day contract; it cannot be weakened in the manifest)")
    guard_cols = sorted(cohort.get("predictor_max_record_date", {}).get("columns") or [])
    declared_rdc = sorted({f.record_date_column for f in feats if f.include_in_baseline and f.record_date_column})
    if guard_cols != declared_rdc:
        problems.append(f"cohort.predictor_max_record_date.columns {guard_cols} must equal the record_date_column values of the included "
                        f"mappings {declared_rdc} (the D-00 guard covers exactly the event sources the baseline reads)")
    out = raw["outcome"]
    for key in ("canonical_name", "feature_spec", "label_column", "event_date_column", "window_end_column", "label_reason_column",
                "window_days_including_index", "censored_rows"):
        if key not in out:
            problems.append(f"outcome: missing {key}")
    if problems:
        raise ConfigError(f"Invalid Meuhedet mapping manifest {p}: " + "; ".join(problems[:25]))
    feats.sort(key=lambda f: list(by_spec).index(f.canonical))
    return WideMapping(name=str(meta["name"]), version=str(meta["version"]), feature_spec_path=str(meta["feature_spec"]),
                       exploratory_spec_path=str(out["feature_spec"]), contract_path=str(meta.get("contract", DEFAULT_CONTRACT)),
                       status=str(meta.get("status", "")), clinically_validated=bool(meta.get("clinically_validated", False)),
                       identity=dict(raw["identity"]), cohort=dict(raw["cohort"]), outcome=dict(out), metadata=dict(raw["metadata"]),
                       open_questions=dict(raw.get("open_questions") or {}), features=tuple(feats),
                       content_sha256=_sha256_json(raw), spec_sha256=spec.content_sha256, raw=raw)


def mapping_table(mapping: WideMapping) -> pd.DataFrame:
    """One row per eFalls candidate with every manifest field plus the spec definition (the deliverable mapping table)."""
    rows = []
    for f in mapping.features:
        e = f.efalls
        rows.append({"canonical_feature": f.canonical, "efalls_definition": e.get("concept"), "efalls_term": e.get("efalls_term"),
                     "efalls_window": json.dumps(e.get("window"), default=str) if isinstance(e.get("window"), dict) else e.get("window"),
                     "efalls_non_code_rule": e.get("non_code_rule"), "mandatory": e.get("mandatory"),
                     "retained_in_published_model": e.get("retained_in_published_model"),
                     "meuhedet_source_column": "; ".join(f.source_columns), "record_date_column": f.record_date_column or "",
                     "same_day_evidence_column": f.same_day_evidence_column or "",
                     "mapping_status": f.status, "mapping_quality": f.quality,
                     "raw_dtype": f.raw_dtype, "canonical_dtype": f.canonical_dtype or e.get("dtype"), "model_dtype": f.model_dtype,
                     "missing_rule": f.missing_rule, "transformation": f.transformation, "lookback_window": f.lookback_window,
                     "allowed_values": json.dumps(f.allowed_values) if f.allowed_values is not None else "",
                     "include_in_baseline": f.include_in_baseline, "phase2_candidate": "; ".join(f.phase2_candidate), "notes": f.notes})
    return pd.DataFrame(rows)


def column_inventory_table(contract: WideContract) -> pd.DataFrame:
    rows = [{"column": c.name, "sql_type": c.sql, "ddl_nullable": c.nullable, "role": c.role, "semantic_type": c.semantic,
             "model_dtype": c.model_dtype, "timing": c.timing, "allowed_values": json.dumps(list(c.allowed)) if c.allowed else "",
             "null_means": c.null_means or "", "phase2_group": c.group or "", "predictor_allowed": c.predictor_allowed,
             "sentinels": "; ".join(c.sentinels), "sentinel_means": c.sentinel_means or "", "note": c.note}
            for c in contract.columns]
    return pd.DataFrame(rows)


# ============================================================================ adapter
@dataclass
class BuildReport:
    index_date: str                                                   # the snapshot, or "<first>..<last> (<n> snapshots)" for a multi-snapshot build
    n_rows_input: int = 0
    n_rows_other_index_dates: int = 0
    other_index_dates: dict[str, int] = field(default_factory=dict)
    index_dates: list[str] = field(default_factory=list)              # every snapshot the build keeps (one entry for a single-snapshot build)
    rows_by_index_date: dict[str, int] = field(default_factory=dict)  # final modelling rows per snapshot
    events_by_index_date: dict[str, int] = field(default_factory=dict)
    timing_violations_by_index_date: dict[str, int] = field(default_factory=dict)  # D-00 rows (failed or dropped) per snapshot
    n_rows_ineligible: int = 0
    ineligible_by_reason: dict[str, int] = field(default_factory=dict)
    n_rows_label_null: int = 0
    label_null_by_reason: dict[str, int] = field(default_factory=dict)
    n_rows_timing_violation: int = 0
    timing_violations_by_column: dict[str, int] = field(default_factory=dict)
    timing_scope: str = "cohort"                                      # cohort | built_predictors (TIMING_SCOPES)
    timing_guard_columns: list[str] = field(default_factory=list)     # the date columns whose on/after-index values excluded rows in THIS build
    cohort_guard_rows_retained: int = 0                               # built_predictors scope: rows kept although a cohort guard column is on/after index
    cohort_guard_rows_retained_by_column: dict[str, int] = field(default_factory=dict)
    n_rows_final: int = 0
    n_patients_final: int = 0
    n_events: int = 0
    outcome_prevalence: float | None = None
    outcome_window_days_observed: dict[str, int] = field(default_factory=dict)
    null_to_zero: dict[str, int] = field(default_factory=dict)
    sentinel_to_null: dict[str, int] = field(default_factory=dict)   # declared sentinel dates -> NULL in the modelling layer (contract rule to_null only)
    source_absent_rows: dict[str, int] = field(default_factory=dict)
    sex_code_descriptions: dict[str, list[str]] = field(default_factory=dict)
    birth_date_suspect_rows: int = 0
    index_day_fall_rows: int = 0
    deceased_rows: int = 0
    contract: dict[str, Any] = field(default_factory=dict)
    coverage: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    pseudonymised: bool = False
    synthetic: bool = False   # every Snapshot_Key starts with SYN_ (falls_ml.data.meuhedet_synthetic); never true for VIEW exports

    def to_dict(self) -> dict[str, Any]:
        return {k: (dict(v) if isinstance(v, dict) else list(v) if isinstance(v, list) else v) for k, v in self.__dict__.items()}


def to_modeling_dates(s: pd.Series, *, source: str, canonical: str, column: ColumnContract | None, problems: list[str],
                      rep: BuildReport) -> pd.Series:
    """Contract-layer dates (datetime64[us], the full SQL range) -> modelling-layer dates (datetime64[ns]). A value beyond the
    nanosecond range is never converted silently: a declared sentinel whose contract rule is ``sentinel_modeling_rule: to_null``
    becomes NULL and is counted; anything else stops the build naming the column, the value and the expected type (never an id)."""
    arr = s.to_numpy(dtype="datetime64[us]")
    beyond = ~np.isnat(arr) & ((arr < NS_MIN_DAY) | (arr > NS_MAX_DAY))
    if beyond.any():
        vals, counts = np.unique(arr[beyond].astype("datetime64[D]"), return_counts=True)
        found = {str(v): int(n) for v, n in zip(vals, counts)}
        declared = set(column.sentinels) if column is not None else set()
        undeclared = {k: n for k, n in found.items() if k not in declared}
        if column is not None and not undeclared and column.sentinel_modeling_rule == "to_null":
            rep.sentinel_to_null[canonical] = int(beyond.sum())
            rep.warnings.append(f"{source} -> {canonical}: {int(beyond.sum())} declared sentinel dates {found} ({column.sentinel_means}) set to NULL "
                                "in the modelling layer by the contract rule sentinel_modeling_rule: to_null")
        else:
            hint = (f"declared sentinel ({column.sentinel_means}), kept in the contract layer; the contract rule sentinel_modeling_rule: to_null "
                    "would map it to NULL in the modelling layer" if column is not None and not undeclared else
                    "not a declared sentinel: declare `sentinels` + `sentinel_means` (and a sentinel_modeling_rule) for the column in the contract "
                    "if it is a business sentinel, otherwise correct the extract")
            problems.append(f"{source} -> {canonical}: {int(beyond.sum())} dates {found} cannot be represented in the modelling layer "
                            f"({MODELING_DATE_DTYPE}, {NS_RANGE_TEXT}); expected semantic type date; {hint}")
        s = s.where(~beyond)
    return s.astype(MODELING_DATE_DTYPE)


class MeuhedetWideDatasetAdapter:
    """Wide table -> canonical eFalls modelling frame (baseline predictors of the mapping manifest only)."""

    def __init__(self, mapping: WideMapping, contract: WideContract, spec: FeatureSpec, *, features: Iterable[str] | None = None,
                 id_pepper: str | None = None, timing_scope: str = "cohort", forbidden_columns: Iterable[str] | None = None) -> None:
        """``features``: the eFalls predictors to build (default: every included mapping); must be included mappings.
        ``id_pepper``: when given, research_id = sha256(pepper | Customer_Full_ID)[:20] (row linkage without the real identifier).
        ``timing_scope``: ``cohort`` applies the mapping's D-00 guard columns (one shared cohort for every feature set);
        ``built_predictors`` applies the record_date_column and same_day_evidence_column of the predictors built here only
        (SAFE-ALL-ROWS sensitivity: rows are excluded exactly when the file proves an index-day record of a source a built predictor reads).
        ``forbidden_columns``: extract columns no built feature may read (value or validation input); a feature that does raises
        ``LeakageError`` before anything is read (the D-00 sensitivity passes every column of a source proven unsafe)."""
        if spec.outcome.name != mapping.outcome["canonical_name"]:
            raise ConfigError(f"feature spec outcome {spec.outcome.name!r} is not the mapping outcome {mapping.outcome['canonical_name']!r} "
                              f"(use {mapping.exploratory_spec_path})")
        want = int(mapping.outcome["window_days_including_index"])
        if spec.outcome.horizon_days != want:
            raise ConfigError(f"feature spec outcome window {spec.outcome.horizon_label} differs from the mapping window ({want} days)")
        self.mapping, self.contract, self.spec = mapping, contract, spec
        wanted = list(features) if features is not None else mapping.baseline_features
        bad = [f for f in wanted if f not in set(mapping.baseline_features)]
        if not wanted or bad:
            raise ConfigError(f"features must be a non-empty subset of the included mappings {mapping.baseline_features}; got {wanted} (bad: {bad})")
        self.feature_names = [f.canonical for f in mapping.features if f.canonical in set(wanted)]
        self.id_pepper = id_pepper
        if timing_scope not in TIMING_SCOPES:
            raise ConfigError(f"timing_scope must be one of {TIMING_SCOPES}, got {timing_scope!r}")
        self.timing_scope = timing_scope
        built = [f for f in mapping.features if f.canonical in set(self.feature_names)]
        self.forbidden_columns = frozenset(forbidden_columns or ())
        if self.forbidden_columns:
            reads = {f.canonical: sorted(set(feature_input_columns(f)["value"]) | set(feature_input_columns(f)["validation"])) for f in built}
            bad = {k: [c for c in v if c in self.forbidden_columns] for k, v in reads.items()}
            bad = {k: v for k, v in bad.items() if v}
            if bad:
                raise LeakageError("a feature of this build reads a column of a source proven unsafe for this cohort (D-00)",
                                   [f"{k} reads {v}" for k, v in bad.items()])
        if timing_scope == "cohort":
            self.timing_guard_columns = list(mapping.cohort["predictor_max_record_date"]["columns"])
        else:
            cols = [f.record_date_column for f in built if f.record_date_column] + [f.same_day_evidence_column for f in built if f.same_day_evidence_column]
            self.timing_guard_columns = sorted(set(cols))
        self.baseline_spec = spec.subset(self.feature_names)

    # -------------------------------------------------------------- helpers
    def used_columns(self) -> list[str]:
        m = self.mapping
        cols = [m.identity["research_id"], m.identity["index_date"], *m.cohort["filters"], *m.cohort.get("required_constant", {}),
                *m.cohort["predictor_max_record_date"]["columns"], *self.timing_guard_columns, m.outcome["label_column"], m.outcome["event_date_column"],
                m.outcome["window_end_column"], m.outcome["label_reason_column"], *m.metadata.values()]
        for f in m.features:
            if f.canonical in set(self.feature_names):
                inputs = feature_input_columns(f)
                cols.extend(inputs["value"])
                cols.extend(inputs["validation"])
                if f.same_day_evidence_column:
                    cols.append(f.same_day_evidence_column)
        for extra in ("Gender_Desc", "Birth_Date_Suspect_Ind", "Fall_On_Index_Date_Ind", "Is_Deceased_Ind", "Snapshot_Key"):
            if extra in self.contract.names:
                cols.append(extra)
        seen: list[str] = []
        for c in cols:
            if c not in seen:
                seen.append(c)
        return seen

    def _feature(self, f: FeatureMapping, df: pd.DataFrame, rep: BuildReport, problems: list[str]) -> pd.Series:
        n = len(df)
        if f.absent_indicator and f.absent_indicator in df.columns:
            rep.source_absent_rows[f.canonical] = int((df[f.absent_indicator].fillna(0) == 1).sum())
        if f.op == "copy_numeric":
            s = df[f.source_columns[0]]
            if s.isna().any():
                problems.append(f"{f.canonical}: {int(s.isna().sum())} NULL values in {f.source_columns[0]} (missing_rule forbid)")
            return s.astype("Float64").astype("float64")
        if f.op == "recode":
            src = f.source_columns[0]
            s = df[src]
            codes = {str(int(k)) if isinstance(k, (int, np.integer)) else str(k): v for k, v in (f.codes or {}).items()}
            keys = s.astype("string")
            keys = keys.where(s.isna(), keys.str.replace(r"\.0$", "", regex=True))
            if s.isna().any():
                problems.append(f"{f.canonical}: {int(s.isna().sum())} NULL values in {src} (missing_rule forbid)")
            unknown = sorted(set(keys.dropna().unique()) - set(codes))
            if unknown:
                problems.append(f"{f.canonical}: codes {unknown[:10]} of {src} are not declared in the mapping ({codes})")
            out = keys.map(codes).astype("string")
            if src == "Gender_Code" and "Gender_Desc" in df.columns:
                tab = df.groupby(s.astype("string"), observed=True)["Gender_Desc"].agg(lambda v: sorted(map(str, pd.unique(v.dropna()))))
                rep.sex_code_descriptions = {str(k): list(v) for k, v in tab.items()}
                multi = {k: v for k, v in rep.sex_code_descriptions.items() if len(v) > 1}
                if multi:
                    problems.append(f"sex: Gender_Code maps to several Gender_Desc values {multi}")
            return out
        if f.op == "any_positive":
            acc = np.zeros(n, dtype=bool)
            filled = 0
            for src in f.source_columns:
                s = df[src]
                nulls = s.isna()
                if nulls.any():
                    if not f.null_is_absent:
                        problems.append(f"{f.canonical}: {int(nulls.sum())} NULL values in {src} and null_is_absent is false")
                    filled += int(nulls.sum())
                if (s.dropna() < 0).any():
                    problems.append(f"{f.canonical}: negative values in {src}")
                acc |= (s.fillna(0) > 0).to_numpy()
            if filled:
                rep.null_to_zero[f.canonical] = filled
            return pd.Series(acc.astype("int8"), index=df.index)
        if f.op == "count":
            src = f.source_columns[0]
            s = df[src]
            nulls = s.isna()
            ind = df[f.absent_indicator].fillna(0) == 1
            unexplained = nulls & ~ind
            if unexplained.any():
                problems.append(f"{f.canonical}: {int(unexplained.sum())} NULL values in {src} not explained by {f.absent_indicator} = 1")
            if (s.dropna() < 0).any():
                problems.append(f"{f.canonical}: negative counts in {src}")
            if int((nulls & ind).sum()):
                rep.null_to_zero[f.canonical] = int((nulls & ind).sum())
            return s.fillna(0).astype("int64")
        raise ConfigError(f"{f.canonical}: unsupported op {f.op!r}")

    # -------------------------------------------------------------- main
    def apply(self, wide: pd.DataFrame, *, index_date: str | None = None, index_dates: Iterable[str] | None = None,
              index_day_records: str | None = None, wrong_type: Mapping[str, int] | None = None,
              wrong_type_examples: Mapping[str, Iterable[str]] | None = None,
              date_formats: Mapping[str, Mapping[str, int]] | None = None) -> tuple[pd.DataFrame, BuildReport]:
        """Validate the extract, select the cohort (Is_Eligible_Cohort = 1 on one ``index_date`` or on every snapshot listed in
        ``index_dates``), map the eFalls predictors and the exploratory outcome. Every rule is applied row by row against the row's
        own Index_Date, so a multi-snapshot build is the union of the single-snapshot builds. Returns the canonical frame
        (unvalidated against the spec: ``write_modeling_dataset`` does that)."""
        m, c = self.mapping, self.contract
        idx_col, id_col = m.identity["index_date"], m.identity["research_id"]
        policy = index_day_records or m.cohort.get("index_day_records", "fail")
        if policy not in ("fail", "drop_rows"):
            raise ConfigError(f"index_day_records must be fail or drop_rows, got {policy!r}")
        if (index_date is None) == (index_dates is None):
            raise ConfigError("give exactly one of index_date (one snapshot) or index_dates (several snapshots)")
        targets = sorted({pd.Timestamp(d).normalize() for d in ([index_date] if index_date is not None else list(index_dates))})
        if not targets:
            raise ConfigError("index_dates is empty")
        label = str(targets[0].date()) if len(targets) == 1 else f"{targets[0].date()}..{targets[-1].date()} ({len(targets)} snapshots)"
        rep = BuildReport(index_date=label, n_rows_input=len(wide), index_dates=[str(t.date()) for t in targets])
        contract_report = validate_wide_contract(wide, c, wrong_type=wrong_type, wrong_type_examples=wrong_type_examples,
                                                 date_formats=date_formats, strict_columns=self.used_columns(), strict=True)
        rep.contract = contract_report.to_dict()
        rep.warnings.extend(contract_report.warnings)
        problems: list[str] = []

        dates = wide[idx_col].dt.normalize()
        on_date = dates.isin(targets)
        others = dates[~on_date]
        rep.n_rows_other_index_dates = int((~on_date).sum())
        rep.other_index_dates = {str(k.date()) if pd.notna(k) else "NULL": int(v) for k, v in others.value_counts(dropna=False).items()}
        df = wide.loc[on_date]
        if df.empty:
            raise DatasetValidationError(f"No rows with {idx_col} in {rep.index_dates}", [f"index dates present: {rep.other_index_dates}"])
        for col, value in m.cohort["filters"].items():
            keep = df[col].fillna(-1) == value
            rep.n_rows_ineligible += int((~keep).sum())
            reasons = df.loc[~keep, "Exclusion_Reason"].fillna("NULL").astype(str).value_counts() if "Exclusion_Reason" in df.columns else {}
            rep.ineligible_by_reason = {str(k): int(v) for k, v in dict(reasons).items()}
            df = df.loc[keep]
        for col, value in (m.cohort.get("required_constant") or {}).items():
            seen = sorted(map(str, pd.unique(df[col].dropna())))
            if seen != [str(value)]:
                problems.append(f"{col} must be {value!r} for every eligible row, found {seen[:5]}")
        if "Snapshot_Key" in df.columns:
            keys = df["Snapshot_Key"].astype("string")
            rep.synthetic = bool(len(keys) and keys.notna().all() and keys.str.startswith("SYN_").all())
            n_dup_key = int(df["Snapshot_Key"].dropna().duplicated().sum())
            if n_dup_key:
                problems.append(f"Snapshot_Key: {n_dup_key} duplicated technical row keys among eligible rows (the VIEW should be unique)")
        n_dup_patient = int(df.duplicated(subset=[id_col, idx_col]).sum())
        if n_dup_patient:
            problems.append(f"{n_dup_patient} duplicated {id_col} x {idx_col} rows among eligible rows (one row per patient and index date)")

        # ---- outcome (exploratory): NULL label = censored -> dropped and counted
        label = df[m.outcome["label_column"]]
        null_label = label.isna()
        rep.n_rows_label_null = int(null_label.sum())
        reasons = df.loc[null_label, m.outcome["label_reason_column"]].fillna("NULL").astype(str).value_counts()
        rep.label_null_by_reason = {str(k): int(v) for k, v in reasons.items()}
        df = df.loc[~null_label]
        if df.empty:
            raise DatasetValidationError("No eligible rows with a non-null label", [])
        y = df[m.outcome["label_column"]].astype("int64")
        bad_y = set(pd.unique(y)) - {0, 1}
        if bad_y:
            problems.append(f"{m.outcome['label_column']}: illegal values {sorted(bad_y)}")
        ev = df[m.outcome["event_date_column"]]
        idx = df[idx_col].dt.normalize()
        end_col = df[m.outcome["window_end_column"]]
        observed = (end_col.dt.normalize() - idx).dt.days
        rep.outcome_window_days_observed = {str(k): int(v) for k, v in observed.value_counts(dropna=False).items()}
        want = int(m.outcome["window_days_including_index"]) - 1
        if observed.notna().any() and (observed.dropna() != want).any():
            problems.append(f"{m.outcome['window_end_column']} - {idx_col} is {sorted(observed.dropna().unique())[:5]} days, the mapping "
                            f"declares window_days_including_index = {want + 1}; set it to the VIEW's window before building")
        pos = y == 1
        if (pos & ev.isna()).any():
            problems.append(f"{m.outcome['event_date_column']}: missing for {int((pos & ev.isna()).sum())} positive labels")
        if (~pos & ev.notna()).any():
            problems.append(f"{m.outcome['event_date_column']}: present for {int((~pos & ev.notna()).sum())} negative labels")
        outside = ev.notna() & ((ev.dt.normalize() < idx) | (ev.dt.normalize() > idx + pd.Timedelta(days=want)))
        if outside.any():
            problems.append(f"{m.outcome['event_date_column']}: {int(outside.sum())} event dates outside the label window")

        # ---- D-00 timing: predictor record dates strictly before the index date. The guard columns are the cohort's (scope
        # "cohort") or those of the built predictors (scope "built_predictors"); the cohort guard columns are always measured.
        cohort_cols = list(m.cohort["predictor_max_record_date"]["columns"])
        pm_cols = list(self.timing_guard_columns)
        rep.timing_scope, rep.timing_guard_columns = self.timing_scope, list(pm_cols)
        if pm_cols:
            pm = df[pm_cols].apply(lambda s: s.dt.normalize() if pd.api.types.is_datetime64_any_dtype(s) else s).max(axis=1, skipna=True)
            pm = pm.where(pm.notna(), idx - pd.Timedelta(days=1))
        else:
            pm = idx - pd.Timedelta(days=1)   # no built predictor reads a dated source: nothing to guard, recorded as such
        viol = pm >= idx
        for col in pm_cols:
            n_col = int((df[col].dt.normalize() >= idx).sum())
            if n_col:
                rep.timing_violations_by_column[col] = n_col
        rep.n_rows_timing_violation = int(viol.sum())
        if viol.any():
            rep.timing_violations_by_index_date = {str(k.date()): int(v) for k, v in idx[viol].value_counts().sort_index().items()}
            if policy == "fail":
                problems.append(f"D-00 timing: {int(viol.sum())} rows have predictor record dates on/after the index date "
                                f"{rep.timing_violations_by_column} (index_day_records=fail; drop_rows removes and counts them)")
            else:
                rep.warnings.append(f"D-00 timing: {int(viol.sum())} rows dropped for predictor record dates on/after the index date "
                                    f"{rep.timing_violations_by_column} (guard scope {self.timing_scope}: {pm_cols})")
                df, y, ev, idx, pm = df.loc[~viol], y.loc[~viol], ev.loc[~viol], idx.loc[~viol], pm.loc[~viol]
        if self.timing_scope != "cohort":
            retained = pd.Series(False, index=df.index)
            for col in cohort_cols:
                if col in pm_cols:
                    continue
                hit = df[col].dt.normalize() >= idx
                if hit.any():
                    rep.cohort_guard_rows_retained_by_column[col] = int(hit.sum())
                retained |= hit
            rep.cohort_guard_rows_retained = int(retained.sum())
            if rep.cohort_guard_rows_retained:
                rep.warnings.append(f"D-00 scope {self.timing_scope}: {rep.cohort_guard_rows_retained} rows kept although a cohort guard column is on/after "
                                    f"the index date {rep.cohort_guard_rows_retained_by_column}; no built predictor reads those sources")

        # ---- canonical frame
        out = pd.DataFrame(index=df.index)
        ids = df[id_col].astype("string")
        if self.id_pepper:
            pepper = self.id_pepper
            ids = ids.map(lambda v: hashlib.sha256(f"{pepper}|{v}".encode("utf-8")).hexdigest()[:20] if pd.notna(v) else v).astype("string")
            rep.pseudonymised = True
        out[self.spec.identifier_columns[0]] = ids
        out[self.spec.index_column] = to_modeling_dates(idx, source=idx_col, canonical=self.spec.index_column, column=c.get(idx_col),
                                                        problems=problems, rep=rep)
        out[PREDICTOR_MAX_DATE] = to_modeling_dates(pm, source=" | ".join(pm_cols), canonical=PREDICTOR_MAX_DATE, column=None, problems=problems, rep=rep)
        out[self.spec.outcome.name] = y.astype("int8")
        out[OUTCOME_EVENT_DATE] = to_modeling_dates(ev.dt.normalize(), source=m.outcome["event_date_column"], canonical=OUTCOME_EVENT_DATE,
                                                    column=c.get(m.outcome["event_date_column"]), problems=problems, rep=rep)
        for f in self.mapping.features:
            if f.canonical in set(self.feature_names):
                out[f.canonical] = self._feature(f, df, rep, problems)
        for canonical, src in m.metadata.items():
            out[canonical] = (to_modeling_dates(df[src].dt.normalize(), source=src, canonical=canonical, column=c.get(src), problems=problems, rep=rep)
                              if pd.api.types.is_datetime64_any_dtype(df[src]) else df[src])
        if "Birth_Date_Suspect_Ind" in df.columns:
            rep.birth_date_suspect_rows = int((df["Birth_Date_Suspect_Ind"].fillna(0) == 1).sum())
        if "Fall_On_Index_Date_Ind" in df.columns:
            rep.index_day_fall_rows = int((df["Fall_On_Index_Date_Ind"].fillna(0) == 1).sum())
        if "Is_Deceased_Ind" in df.columns:
            rep.deceased_rows = int((df["Is_Deceased_Ind"].fillna(0) == 1).sum())
        if problems:
            raise DatasetValidationError(f"Meuhedet wide table -> canonical dataset failed ({len(problems)} problems)", problems)
        out = out.reset_index(drop=True)
        rep.n_rows_final = len(out)
        rep.n_patients_final = int(out[self.spec.identifier_columns[0]].nunique())
        rep.n_events = int(out[self.spec.outcome.name].sum())
        by_date = out.groupby(out[self.spec.index_column].dt.strftime("%Y-%m-%d"))[self.spec.outcome.name]
        rep.rows_by_index_date = {str(k): int(v) for k, v in by_date.size().items()}
        rep.events_by_index_date = {str(k): int(v) for k, v in by_date.sum().items()}
        rep.outcome_prevalence = rep.n_events / len(out) if len(out) else None
        rep.coverage = {**self.mapping.coverage(self.spec), "features_built": list(self.feature_names)}
        log.info("meuhedet_wide_adapted", extra_fields={"index_date": str(index_date), "n_rows": len(out), "n_events": rep.n_events,
                                                        "mapping_sha256": self.mapping.content_sha256, "contract_sha256": self.contract.content_sha256})
        return out, rep


def build_meuhedet_dataset_from_frame(wide: pd.DataFrame, out_dir: str | Path, *, index_date: str | None = None, dataset_version: str,
                                      data_freeze_date: str | None, mapping: WideMapping, contract: WideContract, spec: FeatureSpec,
                                      input_name: str, input_sha256: str, wrong_type: Mapping[str, int] | None = None,
                                      wrong_type_examples: Mapping[str, Iterable[str]] | None = None,
                                      date_formats: Mapping[str, Mapping[str, int]] | None = None, features: Iterable[str] | None = None, feature_set_label: str = "baseline", id_pepper: str | None = None,
                                      scientific_use_allowed: bool = False, approval_reference: str | None = None, notes: str = "",
                                      index_day_records: str | None = None, read_report: Mapping[str, Any] | None = None,
                                      index_dates: Iterable[str] | None = None, timing_scope: str = "cohort",
                                      forbidden_columns: Iterable[str] | None = None) -> tuple[Any, BuildReport]:
    """Already-read extract frame -> immutable canonical modelling dataset + build report (used by build and explore).
    ``index_date`` keeps one snapshot; ``index_dates`` keeps several (the multi-snapshot temporal design)."""
    from falls_ml.data.dataset import write_modeling_dataset

    adapter = MeuhedetWideDatasetAdapter(mapping, contract, spec, features=features, id_pepper=id_pepper, timing_scope=timing_scope,
                                         forbidden_columns=forbidden_columns)
    frame, report = adapter.apply(wide, index_date=index_date, index_dates=index_dates, index_day_records=index_day_records,
                                  wrong_type=wrong_type, wrong_type_examples=wrong_type_examples, date_formats=date_formats)
    audit = {"meuhedet_build": {**report.to_dict(), "contract_report": dict(report.contract),   # the checks; "contract" below is the identity
                                "input_file_name": input_name, "input_sha256": input_sha256,
                                "feature_set": {"label": feature_set_label, "features": list(adapter.feature_names)},
                                "timing": {"scope": adapter.timing_scope, "guard_columns": list(adapter.timing_guard_columns),
                                           "cohort_guard_columns": list(mapping.cohort["predictor_max_record_date"]["columns"])},
                                "read": dict(read_report or {}),
                                "identifiers": {"pseudonymised": bool(id_pepper), "research_id_rule": "sha256(pepper | Customer_Full_ID)[:20]" if id_pepper
                                                else "Customer_Full_ID copied (not shared outside the data environment)"},
                                "mapping": {"name": mapping.name, "version": mapping.version, "sha256": mapping.content_sha256,
                                            "status": mapping.status, "clinically_validated": mapping.clinically_validated},
                                "contract": {"name": contract.name, "version": contract.version, "sha256": contract.content_sha256},
                                "exploratory_outcome": {"name": spec.outcome.name, "horizon": spec.outcome.horizon_label,
                                                        "concept": spec.outcome.raw.get("concept"), "label_column": mapping.outcome["label_column"],
                                                        "NOT_EFALLS_OUTCOME": True},
                                "scientific_use_approval_reference": approval_reference if scientific_use_allowed else None,
                                "database_connections": "none"}}
    source = "synthetic_fixture" if report.synthetic else SOURCE_NAME
    if report.synthetic and scientific_use_allowed:
        raise DatasetValidationError("a synthetic wide-table extract can never be allowed for scientific use", [])
    note_prefix = "SYNTHETIC DATA - NOT SCIENTIFIC RESULTS; " if report.synthetic else ""
    manifest = write_modeling_dataset(frame, out_dir, adapter.baseline_spec, dataset_version=dataset_version,
                                      mapping_version=f"{mapping.name}-{mapping.version}", source=source,
                                      generator="falls_ml meuhedet-build", scientific_use_allowed=scientific_use_allowed,
                                      notes=note_prefix + (notes or "Meuhedet 180-day exploratory outcome - NOT eFalls reproduction"),
                                      audit=audit, data_freeze_date=data_freeze_date)
    return manifest, report


def build_meuhedet_dataset(extract: str | Path, out_dir: str | Path, *, index_date: str, dataset_version: str, data_freeze_date: str | None,
                           mapping_path: str | Path = DEFAULT_MAPPING, contract_path: str | Path | None = None,
                           scientific_use_allowed: bool = False, approval_reference: str | None = None, notes: str = "",
                           index_day_records: str | None = None, encoding: str = "utf-8", sep: str = ",", sheet: str | None = None,
                           feature_set: str | None = None, id_pepper: str | None = None) -> tuple[Any, BuildReport]:
    """Extract file (.xlsx/.parquet/.csv) -> immutable canonical modelling dataset + build report. No database access.
    ``feature_set``: None = every included mapping; ``strict`` / ``extended`` = the manifest-derived exploratory sets."""
    from falls_ml.data.dataset import sha256_file

    mapping = load_wide_mapping(mapping_path)
    contract = load_wide_contract(contract_path or mapping.contract_path)
    spec = load_feature_spec(mapping.exploratory_spec_path)
    read = read_wide_extract_report(extract, contract, encoding=encoding, sep=sep, sheet=sheet)
    features = None if feature_set is None else mapping.feature_sets()[feature_set]
    return build_meuhedet_dataset_from_frame(read.frame, out_dir, index_date=index_date, dataset_version=dataset_version,
                                             data_freeze_date=data_freeze_date, mapping=mapping, contract=contract, spec=spec,
                                             input_name=Path(extract).name, input_sha256=sha256_file(Path(extract)), wrong_type=read.wrong_type,
                                             wrong_type_examples=read.wrong_type_examples, date_formats=read.date_formats, features=features, feature_set_label=feature_set or "baseline", id_pepper=id_pepper,
                                             scientific_use_allowed=scientific_use_allowed, approval_reference=approval_reference, notes=notes,
                                             index_day_records=index_day_records, read_report=read.to_dict())
