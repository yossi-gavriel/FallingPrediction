"""The D-00 feature-dependency graph: every modelling feature traced to the extract columns the builder reads, their data-dictionary
sources, the declared VIEW-level derivations (followed transitively) and the record dates that bound each source in time; the D-00 status
(SAFE / UNSAFE / UNRESOLVED) of every feature per analysis cohort; and the feature sets derived from those statuses.

Nothing here reads an outcome VALUE: the cohorts use label availability (a non-NULL 180-day label = the cohort definition of the reference
run), and the evidence is counts of record dates on / after the index day. The test partition is not involved at all. Safety is never
inferred from a feature or column name: the edges come from the mapping manifest (``feature_input_columns``, the function the builder
itself uses), the data dictionary (column -> source -> record date) and the declared derivations of ``configs/meuhedet/d00_sensitivity.yaml``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from falls_ml.data.meuhedet_wide import PREDICTOR_ROLES, WideContract, WideMapping, feature_input_columns
from falls_ml.errors import ConfigError

DEFAULT_D00_CONFIG = "configs/meuhedet/d00_sensitivity.yaml"
STATUSES = ("SAFE", "UNRESOLVED", "UNSAFE")          # ascending severity
EVIDENCE_KINDS = ("STATIC_ATTRIBUTE", "COMPLETE_RECORD_DATE", "SUFFICIENT_ONLY_RECORD_DATE", "NO_RECORD_DATE", "POST_INDEX", "NOT_A_PREDICTOR_SOURCE")
COHORTS = ("FULL_LABELED", "D00_CLEAN")
STATUS_TEXT = {
    "SAFE": "the complete provenance shows that no record dated on/after the index day can enter the value in this cohort",
    "UNSAFE": "the extract proves, on at least one row of this cohort, that a record dated on/after the index day entered a source the feature reads",
    "UNRESOLVED": "the extract can neither prove nor exclude an index-day record in a source the feature reads",
}
KIND_TEXT = {
    "STATIC_ATTRIBUTE": "fixed member attribute or a value computed only from it and the index date (no event record)",
    "COMPLETE_RECORD_DATE": "record-date column bounding every record behind the source's values",
    "SUFFICIENT_ONLY_RECORD_DATE": "first-entry date: proves an index-day record when on/after the index day, cannot exclude one otherwise",
    "NO_RECORD_DATE": "no record date in the extract",
    "POST_INDEX": "follow-up / outcome information (after the prediction time)",
    "NOT_A_PREDICTOR_SOURCE": "technical / cohort bookkeeping source",
}


def worst(statuses: list[str]) -> str:
    return max(statuses, key=STATUSES.index) if statuses else "UNRESOLVED"


# ============================================================================ configuration
@dataclass(frozen=True)
class D00Config:
    name: str
    version: str
    path: str
    sha256: str
    source_evidence: dict[str, str]
    derived_from: dict[str, dict[str, Any]]
    open_questions: dict[str, str]
    plan: dict[str, Any]
    header: dict[str, Any]

    @property
    def standards(self) -> dict[str, dict[str, Any]]:
        return dict(self.plan["evidence_standards"])

    @property
    def primary_standard(self) -> str:
        return next(k for k, v in self.standards.items() if v.get("role") == "primary")


def load_d00_config(path: str | Path = DEFAULT_D00_CONFIG, *, contract: WideContract, dictionary: Any) -> D00Config:
    """Load and validate against the contract and the data dictionary: every dictionary source has an evidence kind, derivations name
    contract columns, the plan names known cohorts, feature sets and standards."""
    from falls_ml.paths import resolve_path

    p = resolve_path(path)
    text = p.read_text(encoding="utf-8")
    raw = yaml.safe_load(text) or {}
    head = raw.get("d00_sensitivity") or {}
    ev = {str(k): str(v) for k, v in (raw.get("source_evidence") or {}).items()}
    problems: list[str] = []
    missing = [s for s in dictionary.sources if s not in ev]
    if missing:
        problems.append(f"source_evidence misses data-dictionary sources {missing} (every source needs a declared evidence kind)")
    unknown = {k: v for k, v in ev.items() if v not in EVIDENCE_KINDS or k not in dictionary.sources}
    if unknown:
        problems.append(f"source_evidence entries with an unknown source or kind: {unknown} (kinds {EVIDENCE_KINDS})")
    derived: dict[str, dict[str, Any]] = {}
    for col, d in (raw.get("derived_from") or {}).items():
        d = d if isinstance(d, dict) else {"from": list(d), "evidence": ""}
        frm = [str(c) for c in d.get("from") or []]
        bad = [c for c in [col, *frm] if c not in set(contract.names)]
        if bad or not frm:
            problems.append(f"derived_from {col}: unknown contract columns {bad} or empty 'from'")
        derived[str(col)] = {"from": frm, "evidence": str(d.get("evidence") or "")}
    plan = raw.get("analysis_plan") or {}
    stds = plan.get("evidence_standards") or {}
    if sum(1 for v in stds.values() if v.get("role") == "primary") != 1:
        problems.append("analysis_plan.evidence_standards needs exactly one role: primary standard")
    for k, v in stds.items():
        if not set(v.get("admit") or []) <= set(STATUSES) or "UNSAFE" in (v.get("admit") or []) or "SAFE" not in (v.get("admit") or []):
            problems.append(f"evidence standard {k}: admit must contain SAFE, may add UNRESOLVED, never UNSAFE")
    for cell in plan.get("matrix") or []:
        if cell.get("cohort") not in COHORTS or cell.get("feature_set") not in ("STRICT_SAFE", "SAFE_EXTENDED", "FULL_EXTENDED"):
            problems.append(f"matrix cell {cell}: unknown cohort or feature set")
        if cell.get("feature_set") == "FULL_EXTENDED" and cell.get("cohort") != "D00_CLEAN":
            problems.append("FULL_EXTENDED may only be used on D00_CLEAN until the DWH correction")
    for ab in plan.get("ablations") or []:
        if not ab.get("remove"):
            problems.append(f"ablation {ab.get('cell')}: empty remove list")
    if problems:
        raise ConfigError(f"D-00 configuration {p} is invalid: " + "; ".join(problems))
    return D00Config(name=str(head.get("name")), version=str(head.get("version")), path=str(path), sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                     source_evidence=ev, derived_from=derived, open_questions={str(k): " ".join(str(v).split()) for k, v in (raw.get("open_questions") or {}).items()},
                     plan=plan, header=head)


# ============================================================================ cohorts (no outcome values)
@dataclass
class CohortMasks:
    """Row masks over the READ extract frame. FULL_LABELED = rows on the index date that pass the mapping's cohort filters and carry a
    usable (non-NULL) label; D00_CLEAN = FULL_LABELED minus rows whose D-00 guard columns are on/after the index date. The label's
    value is never read - only its availability, exactly as the reference build defines its cohort."""

    index_date: str
    idx: pd.Series
    masks: dict[str, pd.Series]
    guard_columns: list[str]
    counts: dict[str, int] = field(default_factory=dict)

    def __getitem__(self, name: str) -> pd.Series:
        return self.masks[name]


def cohort_masks(frame: pd.DataFrame, mapping: WideMapping, index_date: str) -> CohortMasks:
    idx_col = mapping.identity["index_date"]
    idx = frame[idx_col].dt.normalize()
    on = idx == pd.Timestamp(index_date).normalize()
    base = on.copy()
    for col, value in mapping.cohort["filters"].items():
        base &= frame[col].fillna(-1) == value
    labelled = base & frame[mapping.outcome["label_column"]].notna()
    guard = list(mapping.cohort["predictor_max_record_date"]["columns"])
    hit = pd.Series(False, index=frame.index)
    for col in guard:
        hit |= frame[col].dt.normalize() >= idx
    clean = labelled & ~hit
    counts = {"rows_on_index_date": int(on.sum()), "eligible": int(base.sum()), "FULL_LABELED": int(labelled.sum()), "D00_CLEAN": int(clean.sum()),
              "d00_rows_in_full_labeled": int((labelled & hit).sum())}
    for col in guard:
        counts[f"d00_rows_by_{col}"] = int((labelled & (frame[col].dt.normalize() >= idx)).sum())
    return CohortMasks(index_date=str(pd.Timestamp(index_date).date()), idx=idx, masks={"FULL_LABELED": labelled, "D00_CLEAN": clean},
                       guard_columns=guard, counts=counts)


# ============================================================================ per-column evidence and status
def _recorded(s: pd.Series) -> pd.Series:
    """True where a column carries a recorded value (a positive count / flag, or any non-NULL non-numeric value)."""
    if pd.api.types.is_numeric_dtype(s):
        out = s.notna() & (s.fillna(0) != 0)
    elif pd.api.types.is_datetime64_any_dtype(s):
        out = s.notna()
    else:
        out = s.notna() & (s.astype("string").fillna("").str.strip() != "")
    return out.fillna(False).astype(bool)


@dataclass
class Graph:
    """The dependency graph and everything derived from it (JSON-serialisable via ``to_dict``)."""

    config: D00Config
    index_date: str
    cohort_counts: dict[str, int]
    columns: dict[str, dict[str, Any]]          # every column reached (value/validation inputs, derivations, predictor-role columns)
    features: dict[str, dict[str, Any]]         # canonical modelling feature -> provenance, status per cohort, explanation
    feature_sets: dict[str, dict[str, Any]]     # standard -> {STRICT_SAFE: {...}, SAFE_EXTENDED: {...}}
    forbidden_columns: dict[str, dict[str, list[str]]]   # standard -> cohort -> columns no built feature may read
    source_evidence: dict[str, dict[str, Any]]  # source -> cohort -> evidence
    requested: list[dict[str, Any]]
    sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"config": {"name": self.config.name, "version": self.config.version, "sha256": self.config.sha256, "path": self.config.path},
                "index_date": self.index_date, "cohort_counts": self.cohort_counts, "status_definitions": STATUS_TEXT, "evidence_kinds": KIND_TEXT,
                "features": self.features, "columns": self.columns, "feature_sets": self.feature_sets, "forbidden_columns": self.forbidden_columns,
                "source_evidence": self.source_evidence, "requested_trace": self.requested, "open_questions": self.config.open_questions,
                "graph_sha256": self.sha256,
                "rules": {"outcome_values_used": False, "test_partition_used": False, "names_used_for_safety": False,
                          "propagation": "a feature's status is the worst status over every column the builder reads for it (value and validation "
                                         "inputs), each followed through declared derivations; UNSAFE > UNRESOLVED > SAFE"}}


def _source_evidence(frame: pd.DataFrame, cohorts: CohortMasks, dictionary: Any, config: D00Config) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for src, meta in dictionary.sources.items():
        kind = config.source_evidence[src]
        date_col = meta.get("record_date")
        entry: dict[str, Any] = {"kind": kind, "kind_text": KIND_TEXT[kind], "record_date_column": date_col,
                                 "record_date_meaning_status": dictionary.columns.get(date_col, {}).get("status") if date_col else None,
                                 "record_date_in_extract": bool(date_col and date_col in frame.columns and pd.api.types.is_datetime64_any_dtype(frame[date_col]))}
        for coh in COHORTS:
            m = cohorts[coh]
            e: dict[str, Any] = {"n_rows": int(m.sum())}
            if entry["record_date_in_extract"]:
                off = (frame[date_col].dt.normalize() - cohorts.idx).dt.days
                e.update({"n_with_date": int((m & off.notna()).sum()), "n_on_index": int((m & (off == 0)).sum()), "n_after_index": int((m & (off > 0)).sum())})
                e["n_on_or_after"] = e["n_on_index"] + e["n_after_index"]
            entry[coh] = e
        out[src] = entry
    return out


def _column_status(col: str, coh: str, frame: pd.DataFrame, cohorts: CohortMasks, contract: WideContract, dictionary: Any, config: D00Config,
                   src_ev: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Status of one column's OWN source in one cohort (derivations are combined by the caller)."""
    c = contract.get(col)
    d = dictionary.columns[col]
    src = d["source"]
    kind = config.source_evidence[src]
    date_col = dictionary.record_date(col)
    base = {"column": col, "source": src, "domain": d["domain"], "meaning": d["meaning"], "meaning_status": d["status"], "contract_role": c.role,
            "contract_timing": c.timing, "evidence_kind": kind, "record_date_column": date_col}
    if c.role == "FORBIDDEN_LEAKAGE":
        return {**base, "status": "UNSAFE", "reason": "role FORBIDDEN_LEAKAGE in the contract (never available as a predictor)"}
    if c.timing == "post_index":
        return {**base, "status": "UNSAFE", "reason": "contract timing post_index (known only after the prediction time)"}
    if kind == "POST_INDEX":
        return {**base, "status": "UNSAFE", "reason": f"source {src} holds follow-up / outcome information"}
    if c.timing == "unknown":
        return {**base, "status": "UNRESOLVED", "reason": "contract timing unknown (not confirmed by data engineering)"}
    if kind == "STATIC_ATTRIBUTE":
        if d["status"] == "DOCUMENTED":
            return {**base, "status": "SAFE", "reason": f"source {src}: {KIND_TEXT[kind]}; meaning DOCUMENTED ({d['meaning']})"}
        return {**base, "status": "UNRESOLVED", "reason": f"source {src} is static, but the column's meaning is {d['status']} (not documented)"}
    if kind in ("NO_RECORD_DATE", "NOT_A_PREDICTOR_SOURCE"):
        return {**base, "status": "UNRESOLVED", "reason": f"source {src}: {KIND_TEXT[kind]} - an index-day record can neither be proven nor excluded"}
    # dated sources
    if not date_col or date_col not in frame.columns or not pd.api.types.is_datetime64_any_dtype(frame[date_col]):
        return {**base, "status": "UNRESOLVED", "reason": f"the record-date column {date_col!r} of source {src} is not in the extract"}
    m = cohorts[coh]
    off = (frame[date_col].dt.normalize() - cohorts.idx).dt.days
    n_on, n_after = int((m & (off == 0)).sum()), int((m & (off > 0)).sum())
    ev = {"n_rows": int(m.sum()), "n_on_index": n_on, "n_after_index": n_after}
    if n_on + n_after:
        return {**base, **ev, "status": "UNSAFE",
                "reason": f"{date_col} is on/after the index day on {n_on + n_after} rows of {coh} ({n_on} on the index day, {n_after} after): "
                          f"records the model may not see entered source {src} on those rows"}
    if kind == "SUFFICIENT_ONLY_RECORD_DATE":
        return {**base, **ev, "status": "UNRESOLVED",
                "reason": f"{date_col} is before the index day on every row of {coh}, but it is a first-entry date: a later same-day entry is invisible"}
    date_status = dictionary.columns.get(date_col, {}).get("status")
    if date_status != "DOCUMENTED":
        return {**base, **ev, "status": "UNRESOLVED", "reason": f"no on/after-index {date_col} in {coh}, but its meaning is {date_status} (bound not documented)"}
    if col != date_col and c.role in PREDICTOR_ROLES:   # QA indicators (e.g. an absent indicator = 1) are not records
        unbounded = int((m & frame[date_col].isna() & _recorded(frame[col])).sum()) if col in frame.columns else 0
        ev["n_recorded_without_date"] = unbounded
        if unbounded:
            return {**base, **ev, "status": "UNRESOLVED",
                    "reason": f"{unbounded} rows of {coh} carry a recorded {col} value but no {date_col}: the bound does not cover them"}
    return {**base, **ev, "status": "SAFE", "reason": f"{date_col} (DOCUMENTED record date of {src}) is before the index day on every row of {coh}"}


def _trace(col: str, coh: str, memo: dict[tuple[str, str], dict[str, Any]], stack: tuple[str, ...], **ctx: Any) -> dict[str, Any]:
    """Status of a column including its declared derivations, followed transitively (cycle-safe)."""
    key = (col, coh)
    if key in memo:
        return memo[key]
    own = _column_status(col, coh, **ctx)
    chain: list[dict[str, Any]] = []
    statuses = [own["status"]]
    for parent in ctx["config"].derived_from.get(col, {}).get("from", []):
        if parent in stack:
            chain.append({"column": parent, "status": "UNRESOLVED", "reason": "derivation cycle"})
            statuses.append("UNRESOLVED")
            continue
        t = _trace(parent, coh, memo, (*stack, col), **ctx)
        chain.append({"column": parent, "status": t["status"], "source": t["source"], "via": t.get("derived_chain", [])})
        statuses.append(t["status"])
    res = {**own, "own_status": own["status"], "status": worst(statuses), "derived_chain": chain,
           "derived_evidence": ctx["config"].derived_from.get(col, {}).get("evidence", "")}
    if res["status"] != own["status"]:
        res["reason"] = own["reason"] + f"; inherits {res['status']} from its declared derivation {[c['column'] for c in chain]}"
    memo[key] = res
    return res


def _reached_sources(t: dict[str, Any]) -> list[str]:
    out = [t["source"]]
    for c in t.get("derived_chain", []):
        if c.get("source"):
            out.append(c["source"])
        for v in c.get("via", []):
            if v.get("source"):
                out.append(v["source"])
    return list(dict.fromkeys(out))


def build_graph(frame: pd.DataFrame, cohorts: CohortMasks, *, mapping: WideMapping, contract: WideContract, dictionary: Any, config: D00Config,
                design_columns: dict[str, list[str]] | None = None, requested: tuple[str, ...] = ()) -> Graph:
    """Trace every included mapping feature and every predictor-role contract column; derive statuses and feature sets."""
    ctx = {"frame": frame, "cohorts": cohorts, "contract": contract, "dictionary": dictionary, "config": config,
           "src_ev": _source_evidence(frame, cohorts, dictionary, config)}
    memo: dict[tuple[str, str], dict[str, Any]] = {}
    sets = mapping.feature_sets()
    features: dict[str, dict[str, Any]] = {}
    for f in mapping.features:
        if not f.include_in_baseline:
            continue
        inputs = feature_input_columns(f)
        entry: dict[str, Any] = {
            "feature": f.canonical, "mapping_quality": f.quality, "op": f.op, "transformation": f.transformation, "lookback_window": f.lookback_window,
            "strict": f.canonical in sets["strict"], "extended": f.canonical in sets["extended"],
            "value_columns": list(inputs["value"]), "validation_columns": list(inputs["validation"]), "row_evidence_columns": list(inputs["row_evidence"]),
            "design_columns": list((design_columns or {}).get(f.canonical, [])),
            "mapping_record_date_column": f.record_date_column, "mapping_same_day_evidence_column": f.same_day_evidence_column,
            "evidence_code": ("falls_ml.data.meuhedet_wide.feature_input_columns -> MeuhedetWideDatasetAdapter._feature (value: "
                              f"{list(inputs['value'])}; validation: {list(inputs['validation'])}); mapping {mapping.name} {mapping.version} "
                              f"features.{f.canonical}; data dictionary {dictionary.name} {dictionary.version}; D-00 config {config.name} {config.version}"),
        }
        for coh in COHORTS:
            traced = [_trace(c, coh, memo, (), **ctx) for c in [*inputs["value"], *inputs["validation"]]]
            st = worst([t["status"] for t in traced])
            drivers = [t for t in traced if t["status"] == st]
            entry[coh] = {"status": st, "columns": {t["column"]: t["status"] for t in traced},
                          "explanation": "; ".join(dict.fromkeys(t["reason"] for t in drivers))}
        traced_full = [_trace(c, "FULL_LABELED", memo, (), **ctx) for c in [*inputs["value"], *inputs["validation"]]]
        srcs = list(dict.fromkeys(s for t in traced_full for s in _reached_sources(t)))
        entry["sources"] = srcs
        entry["domains"] = sorted({dictionary.columns[c]["domain"] for c in [*inputs["value"], *inputs["validation"]]})
        entry["record_date_fields"] = sorted({dictionary.sources[s].get("record_date") for s in srcs if dictionary.sources[s].get("record_date")})
        entry["intermediate_aggregate"] = "; ".join(f"{c}: {dictionary.columns[c]['meaning']} ({dictionary.columns[c]['status']})" for c in inputs["value"])
        entry["same_day_data_may_enter"] = _same_day_text(entry, traced_full)
        # cross-check: the mapping's declared D-00 columns vs the dictionary's record dates of the traced sources (no silent disagreement)
        declared = {c for c in (f.record_date_column, f.same_day_evidence_column) if c}
        entry["mapping_vs_dictionary"] = ("agree" if declared <= set(entry["record_date_fields"]) and (declared or not any(
            config.source_evidence[s] in ("COMPLETE_RECORD_DATE", "SUFFICIENT_ONLY_RECORD_DATE") for s in srcs))
            else f"DIFFER: mapping declares {sorted(declared) or 'none'}, dictionary sources give {entry['record_date_fields'] or 'none'} "
                 "(the graph uses the dictionary; the mapping declaration should be reviewed)")
        features[f.canonical] = entry
    # every predictor-role column + requested names (Phase-2 candidates, source columns)
    col_names = [c.name for c in contract.columns if c.role in PREDICTOR_ROLES]
    for name in requested:
        if name in set(contract.names) and name not in col_names:
            col_names.append(name)
    columns: dict[str, dict[str, Any]] = {}
    for col in dict.fromkeys([*col_names, *[c for e in features.values() for c in e["value_columns"] + e["validation_columns"]]]):
        t_full, t_clean = _trace(col, "FULL_LABELED", memo, (), **ctx), _trace(col, "D00_CLEAN", memo, (), **ctx)
        used_by = [k for k, e in features.items() if col in e["value_columns"] or col in e["validation_columns"]]
        columns[col] = {"column": col, "source": t_full["source"], "domain": t_full["domain"], "meaning_status": t_full["meaning_status"],
                        "contract_role": t_full["contract_role"], "contract_timing": t_full["contract_timing"], "evidence_kind": t_full["evidence_kind"],
                        "record_date_column": t_full["record_date_column"], "derived_from": [c["column"] for c in t_full.get("derived_chain", [])],
                        "used_by_features": used_by, "FULL_LABELED": {"status": t_full["status"], "reason": t_full["reason"]},
                        "D00_CLEAN": {"status": t_clean["status"], "reason": t_clean["reason"]},
                        "evidence": {k: t_full.get(k) for k in ("n_rows", "n_on_index", "n_after_index", "n_recorded_without_date") if k in t_full}}
    feature_sets = derive_feature_sets(features, mapping, config)
    forbidden = {std: {coh: sorted(c for c, e in columns.items() if e[coh]["status"] not in set(config.standards[std]["admit"])) for coh in COHORTS}
                 for std in config.standards}
    req = []
    for name in requested:
        if name in features:
            e = features[name]
            req.append({"name": name, "kind": "modelling feature", "FULL_LABELED": e["FULL_LABELED"]["status"], "D00_CLEAN": e["D00_CLEAN"]["status"],
                        "resolution": e["FULL_LABELED"]["explanation"]})
        elif name in columns:
            e = columns[name]
            req.append({"name": name, "kind": "source column of " + ", ".join(e["used_by_features"]) if e["used_by_features"] else "Phase-2 / unused column",
                        "FULL_LABELED": e["FULL_LABELED"]["status"], "D00_CLEAN": e["D00_CLEAN"]["status"], "resolution": e["FULL_LABELED"]["reason"]})
        else:
            req.append({"name": name, "kind": "unknown", "FULL_LABELED": "n/a", "D00_CLEAN": "n/a", "resolution": "not a feature and not a contract column"})
    g = Graph(config=config, index_date=cohorts.index_date, cohort_counts=dict(cohorts.counts), columns=columns, features=features,
              feature_sets=feature_sets, forbidden_columns=forbidden, source_evidence=ctx["src_ev"], requested=req)
    g.sha256 = hashlib.sha256(repr(sorted((k, v["FULL_LABELED"]["status"], v["D00_CLEAN"]["status"]) for k, v in features.items())).encode()
                              + repr(sorted((k, v["FULL_LABELED"]["status"]) for k, v in columns.items())).encode()).hexdigest()
    return g


def _same_day_text(entry: dict[str, Any], traced: list[dict[str, Any]]) -> str:
    st = entry["FULL_LABELED"]["status"]
    if st == "UNSAFE":
        by_date = {t["record_date_column"]: int(t.get("n_on_index", 0)) + int(t.get("n_after_index", 0)) for t in traced
                   if t["status"] == "UNSAFE" and "n_on_index" in t}
        proof = ", ".join(f"{k} on/after the index day on {v} FULL_LABELED rows" for k, v in by_date.items()) or "declared post-index / forbidden"
        return f"YES - proven in FULL_LABELED ({proof}); in D00_CLEAN: {entry['D00_CLEAN']['status']}"
    if st == "SAFE":
        kinds = {t["evidence_kind"] for t in traced}
        return "NO - static attribute / index-date derived" if kinds <= {"STATIC_ATTRIBUTE"} else "NO - record date before the index day on every row"
    return "POSSIBLE - cannot be proven or excluded from the extract"


def derive_feature_sets(features: dict[str, dict[str, Any]], mapping: WideMapping, config: D00Config) -> dict[str, dict[str, Any]]:
    """STRICT_SAFE and SAFE_EXTENDED per evidence standard, from the FULL_LABELED statuses (never a hand-written list). A STRICT feature
    that the standard does not admit is never silently dropped under the STRICT name: the set is renamed (…_SUBSET) and the deviation
    is recorded feature by feature."""
    sets = mapping.feature_sets()
    strict, extended = sets["strict"], sets["extended"]
    out: dict[str, dict[str, Any]] = {}
    for std, spec in config.standards.items():
        admit = set(spec["admit"])
        s_keep = [f for f in strict if features[f]["FULL_LABELED"]["status"] in admit]
        s_drop = {f: features[f]["FULL_LABELED"]["status"] for f in strict if f not in s_keep}
        ext_only = [f for f in extended if f not in set(strict)]
        e_add = [f for f in ext_only if features[f]["FULL_LABELED"]["status"] in admit]
        e_drop = {f: features[f]["FULL_LABELED"]["status"] for f in ext_only if f not in e_add}
        primary = spec.get("role") == "primary"
        s_name = ("STRICT_SAFE" if primary else "STRICT_SAFE_OR_UNRESOLVED") + ("_SUBSET" if s_drop else "")
        e_name = "SAFE_EXTENDED" if primary else "EXTENDED_SAFE_OR_UNRESOLVED"
        e_feats = [f for f in extended if f in set(s_keep) | set(e_add)]
        out[std] = {"standard": std, "role": spec.get("role"), "admit": sorted(admit, key=STATUSES.index),
                    "STRICT_SAFE": {"name": s_name, "features": s_keep, "identical_to_strict": not s_drop,
                                    "deviation": {f: f"removed from STRICT: FULL_LABELED status {st} - {features[f]['FULL_LABELED']['explanation']}" for f, st in s_drop.items()},
                                    "rule": f"STRICT predictors whose FULL_LABELED status is in {sorted(admit)}"},
                    "SAFE_EXTENDED": {"name": e_name, "features": e_feats, "added_to_strict_safe": e_add,
                                      "excluded_extended_predictors": {f: f"FULL_LABELED status {st} - {features[f]['FULL_LABELED']['explanation']}" for f, st in e_drop.items()},
                                      "identical_to_strict_safe": not e_add,
                                      "rule": f"{s_name} + EXTENDED-only predictors whose FULL_LABELED status is in {sorted(admit)}"},
                    "FULL_EXTENDED": {"name": "FULL_EXTENDED", "features": list(extended), "rule": "the reference EXTENDED set (D00_CLEAN only)"}}
    return out
