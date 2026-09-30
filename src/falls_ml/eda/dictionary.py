"""Explicit data dictionary of the wide extract: contract (types, roles, timing) + dictionary YAML (domain, source, meaning) + mapping
(STRICT / EXTENDED use, mapping confidence, Phase-2 candidacy) + the D-00 evidence measured on this extract (availability at the
prediction time). Meanings are never invented: undocumented ones say NAME_ONLY / UNKNOWN and NEEDS SME REVIEW."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from falls_ml.data.meuhedet_wide import PREDICTOR_ROLES, WideContract, WideMapping
from falls_ml.errors import ConfigError

DEFAULT_DICTIONARY = "configs/meuhedet/wide_v1_data_dictionary.yaml"
MEANING_STATUS = ("DOCUMENTED", "NAME_ONLY", "UNKNOWN")
STATUS_TEXT = {"DOCUMENTED": "documented in the S2T/DDL notes (confirm with the SME)",
               "NAME_ONLY": "name only - business definition NEEDS SME REVIEW",
               "UNKNOWN": "UNKNOWN / NEEDS SME REVIEW"}
#: availability of a column's value at the prediction time (start of the index day)
AVAILABILITY_TEXT = {
    "YES_RECORD_DATE_CLEAN": "declared pre/at index; its source's record date is before the index day on every eligible row of this extract",
    "YES_BUT_INDEX_DAY_RECORDS_PROVEN": "declared pre/at index, BUT its source's record date is on/after the index day on some eligible rows (D-00): "
                                        "on those rows the value includes information not available at the start of the index day",
    "YES_PARTIAL_EVIDENCE": "state at index; only sufficient-only same-day evidence exists (First_Registry_Date, Q-M-11)",
    "YES_DECLARED_UNVERIFIABLE": "declared pre/at index by the contract; the extract carries no record date for its source, so the timing cannot be verified",
    "UNKNOWN_TIMING": "timing not confirmed by data engineering (retrospective / billing lag): blocked as a predictor",
    "NO_POST_INDEX": "post-index information (follow-up, censoring, outcome): never available at prediction time",
    "NO_FORBIDDEN": "forbidden by the contract (retrospective, unreliable or audit-only): never a predictor",
    "NOT_APPLICABLE": "technical / identifier / quality field (not a clinical value)",
}


@dataclass(frozen=True)
class DataDictionary:
    name: str
    version: str
    path: str
    sha256: str
    domains: dict[str, str]
    sources: dict[str, dict[str, Any]]
    columns: dict[str, dict[str, Any]]
    raw: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    def domain(self, col: str) -> str:
        return self.columns[col]["domain"]

    def domain_label(self, col: str) -> str:
        return self.domains[self.columns[col]["domain"]]

    def source(self, col: str) -> str:
        return self.columns[col]["source"]

    def record_date(self, col: str) -> str | None:
        """The record-date column bounding the column's source in time (column override > source default)."""
        entry = self.columns[col]
        if "record_date" in entry:
            return entry["record_date"]
        return self.sources[entry["source"]].get("record_date")


def load_data_dictionary(path: str | Path, contract: WideContract) -> DataDictionary:
    """Load and validate: one entry per contract column (no more, no less), known domains / sources / statuses."""
    from falls_ml.paths import resolve_path

    p = resolve_path(path)
    text = p.read_text(encoding="utf-8")
    raw = yaml.safe_load(text)
    head, cols = raw.get("dictionary") or {}, raw.get("columns") or {}
    problems: list[str] = []
    missing = [c for c in contract.names if c not in cols]
    extra = [c for c in cols if c not in set(contract.names)]
    if missing:
        problems.append(f"columns missing from the dictionary: {missing}")
    if extra:
        problems.append(f"dictionary columns not in the contract: {extra}")
    domains, sources = dict(head.get("domains") or {}), dict(head.get("sources") or {})
    for name, e in cols.items():
        if e.get("domain") not in domains:
            problems.append(f"{name}: unknown domain {e.get('domain')!r}")
        if e.get("source") not in sources:
            problems.append(f"{name}: unknown source {e.get('source')!r}")
        if e.get("status") not in MEANING_STATUS:
            problems.append(f"{name}: status must be one of {MEANING_STATUS}")
        if not str(e.get("meaning") or "").strip():
            problems.append(f"{name}: empty meaning (write UNKNOWN instead)")
        rd = e.get("record_date", sources.get(e.get("source"), {}).get("record_date"))
        if rd is not None and (rd not in set(contract.names) or not contract.get(rd).is_date):
            problems.append(f"{name}: record_date {rd!r} is not a date column of the contract")
    if problems:
        raise ConfigError(f"data dictionary {p} is invalid ({len(problems)} problems): " + "; ".join(problems[:20]))
    return DataDictionary(name=str(head.get("name")), version=str(head.get("version")), path=str(path),
                          sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(), domains=domains, sources=sources,
                          columns={c: dict(cols[c]) for c in contract.names}, raw=raw)


def column_usage(mapping: WideMapping) -> dict[str, dict[str, Any]]:
    """Contract column -> how the current analyses use it: features reading it, STRICT / EXTENDED membership, mapping quality, D-00
    guard / same-day evidence role, and the eFalls concepts for which it is a Phase-2 candidate."""
    sets = mapping.feature_sets()
    guard = list(mapping.cohort["predictor_max_record_date"]["columns"])
    out: dict[str, dict[str, Any]] = {}

    def entry(col: str) -> dict[str, Any]:
        return out.setdefault(col, {"features": [], "strict": False, "extended": False, "quality": [], "guard_for": [], "evidence_for": [], "phase2_for": []})

    for f in mapping.features:
        if f.include_in_baseline:
            for c in f.source_columns:
                e = entry(c)
                e["features"].append(f.canonical)
                e["strict"] |= f.canonical in sets["strict"]
                e["extended"] |= f.canonical in sets["extended"]
                e["quality"].append(f.quality)
            if f.record_date_column:
                entry(f.record_date_column)["guard_for"].append(f.canonical)
            if f.same_day_evidence_column:
                entry(f.same_day_evidence_column)["evidence_for"].append(f.canonical)
        for c in f.phase2_candidate:
            entry(c)["phase2_for"].append(f.canonical)
    for c in guard:
        entry(c)
    return out


def availability(col: str, contract: WideContract, dictionary: DataDictionary, on_after: dict[str, int]) -> str:
    """Availability code of a column at the prediction time; ``on_after`` = eligible rows whose date column is on/after the index day."""
    c = contract.get(col)
    if c.role == "IDENTIFIER" or c.timing == "na":
        return "NOT_APPLICABLE"
    if c.role == "LABEL" or c.timing == "post_index":
        return "NO_POST_INDEX"
    if c.role == "FORBIDDEN_LEAKAGE":
        return "NO_FORBIDDEN"
    if c.timing == "unknown":
        return "UNKNOWN_TIMING"
    rd = dictionary.record_date(col)
    if rd is None:
        return "YES_DECLARED_UNVERIFIABLE"
    if dictionary.source(col) == "REGISTRIES":
        return "YES_PARTIAL_EVIDENCE" if not on_after.get(rd) else "YES_BUT_INDEX_DAY_RECORDS_PROVEN"
    return "YES_BUT_INDEX_DAY_RECORDS_PROVEN" if on_after.get(rd) else "YES_RECORD_DATE_CLEAN"


def dictionary_table(contract: WideContract, dictionary: DataDictionary, mapping: WideMapping, on_after: dict[str, int]) -> pd.DataFrame:
    """``data_dictionary.csv``: one row per contract column (no data values)."""
    usage = column_usage(mapping)
    rows = []
    for c in contract.columns:
        d = dictionary.columns[c.name]
        u = usage.get(c.name, {})
        avail = availability(c.name, contract, dictionary, on_after)
        rd = dictionary.record_date(c.name)
        rows.append({
            "column": c.name, "sql_type": c.sql, "declared_type": c.semantic, "pandas_dtype": c.pandas_dtype, "nullable": c.nullable,
            "domain": dictionary.domains[d["domain"]], "domain_key": d["domain"], "source": d["source"], "source_text": dictionary.sources[d["source"]]["text"],
            "meaning": d["meaning"], "meaning_status": d["status"], "meaning_status_text": STATUS_TEXT[d["status"]],
            "contract_role": c.role, "contract_timing": c.timing, "model_dtype": c.model_dtype, "phase2_group": c.group or "",
            "allowed_values": "" if c.allowed is None else ", ".join(map(str, c.allowed)), "null_means": c.null_means or "",
            "declared_sentinels": ", ".join(c.sentinels), "record_date_column": rd or "",
            "available_at_prediction_time": avail, "availability_text": AVAILABILITY_TEXT[avail],
            "predictor_role": c.role in PREDICTOR_ROLES,
            "used_by_strict": bool(u.get("strict")), "used_by_extended": bool(u.get("extended")),
            "efalls_features": "; ".join(u.get("features", [])), "mapping_confidence": "; ".join(sorted(set(u.get("quality", [])))),
            "d00_guard_for": "; ".join(u.get("guard_for", [])), "same_day_evidence_for": "; ".join(u.get("evidence_for", [])),
            "phase2_candidate_for_efalls": "; ".join(u.get("phase2_for", [])), "contract_note": c.note,
        })
    return pd.DataFrame(rows)
