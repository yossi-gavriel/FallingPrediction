"""The AUTHORITATIVE V21 / 2026 schema (``configs/meuhedet/phase5_v21_schema.yaml`` + the embedded VIEW definition file) and the EXACT,
programmatic V1 -> V21 schema diff of the real extract header.

The diff is computed at every preflight from three column lists - the V1 contract (``wide_v1_columns.yaml``, 221 columns), the authoritative V21
schema (224 columns) and the header of the file actually read - never from a hand-written list of "new variables". The reviewed schema supplies only the
SEMANTIC conclusions the names cannot give: whether a shared column kept its meaning (evidence DOCUMENTED_MATCH / CLARIFIED_NO_CONFLICT / CHANGED),
and the definition, timing basis and provenance of every column V1 did not have.

Every column of the extract gets exactly one class:

    OLD_UNCHANGED               V1 predictor column, same meaning in V21
    OLD_CHANGED_DEFINITION      V1 predictor column whose V21 definition is a different quantity (handled explicitly, never silently as OLD)
    RENAMED_OR_REPLACED         V21 column whose LINEAGE to a removed V1 column is PROVEN (V1 and V21 SQL, registry IDs, source and logic compared)
                                with a MATERIAL_DEFINITION_CHANGE (a proven TRUE_RENAME / CORRECTED_LABEL stops for review: OLD inputs are never aliased)
    NEW_CANDIDATE_PREDICTOR     V21 predictor column that V1 did not have
    METADATA_OR_ADMIN           cohort / QA / provenance / record-date-check column (V1 roles QA_CONTROL, COHORT_ELIGIBILITY at or before the index)
    OUTCOME_OR_FUTURE_FORBIDDEN label, follow-up, censoring, post-index or V1 FORBIDDEN_LEAKAGE column
    IDENTIFIER                  member / row identifier
    REQUIRES_SEMANTIC_REVIEW    a column the authoritative schema does not define (or a shared predictor column without a review): STOP before training

A rename is NEVER inferred from a column position or from one name replacing another: the reviewed ``lineage`` section classifies every examined
(removed V1 column, V21 column) pair as TRUE_RENAME_SAME_SEMANTICS / CORRECTED_LABEL_SAME_SOURCE / OLD_REMOVED_NEW_ADDED / MATERIAL_DEFINITION_CHANGE,
and a same-lineage class is accepted only with ``lineage_proven: true`` and the V1 SQL evidence. Unproven -> OLD_REMOVED_NEW_ADDED: the V1 column is
removed and the V21 column is a genuinely new predictor (NEW_CANDIDATE_PREDICTOR).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from falls_ml.errors import ConfigError
from falls_ml.paths import resolve_path

OLD_UNCHANGED, OLD_CHANGED, RENAMED = "OLD_UNCHANGED", "OLD_CHANGED_DEFINITION", "RENAMED_OR_REPLACED"
NEW_CAND, META, FORBIDDEN, IDENT, REVIEW = ("NEW_CANDIDATE_PREDICTOR", "METADATA_OR_ADMIN", "OUTCOME_OR_FUTURE_FORBIDDEN", "IDENTIFIER",
                                            "REQUIRES_SEMANTIC_REVIEW")
CLASSES = (OLD_UNCHANGED, OLD_CHANGED, RENAMED, NEW_CAND, META, FORBIDDEN, IDENT, REVIEW)
EVIDENCE = ("DOCUMENTED_MATCH", "CLARIFIED_NO_CONFLICT", "CHANGED")
NEW_ROLES = ("PREDICTOR", META, IDENT, FORBIDDEN)
KINDS = ("binary", "count", "days", "ordinal", "categorical", "continuous")
TIMINGS = ("record_date", "attested", "uncertain")
PROVENANCE = ("DEFENSIBLE", "UNVALIDATED_CODES", "EXPERIMENTAL_COMPOSITE")
TRUE_RENAME, CORRECTED_LABEL, REMOVED_ADDED, MATERIAL_CHANGE = ("TRUE_RENAME_SAME_SEMANTICS", "CORRECTED_LABEL_SAME_SOURCE", "OLD_REMOVED_NEW_ADDED",
                                                               "MATERIAL_DEFINITION_CHANGE")
LINEAGE_CLASSES = (TRUE_RENAME, CORRECTED_LABEL, REMOVED_ADDED, MATERIAL_CHANGE)
SAME_LINEAGE = (TRUE_RENAME, CORRECTED_LABEL, MATERIAL_CHANGE)     # each needs lineage_proven: true + the V1 SQL evidence
SAME_SOURCE = (TRUE_RENAME, CORRECTED_LABEL)                       # each also needs identical registry IDs and source table
LINEAGE_FIELDS = ("v21_column", "class", "lineage_proven", "v1_sql_expression", "v1_registry_ids", "v1_source_table", "v1_logic", "v21_sql_expression",
                  "v21_registry_ids", "v21_source_table", "v21_logic", "comparison", "conclusion")
V1_EVIDENCE = ("v1_sql_expression", "v1_registry_ids", "v1_source_table")
MISSING = ("no_event", "not_assessed", "unexpected")
PREDICTOR_ROLES = ("EFALLS_BASELINE_FEATURE", "MEUHEDET_ENHANCED_FEATURE")
SELECT_RE = re.compile(r"^\s*,?\[(\w+)\]\s*--\s*(.*)$", re.M)


def _sha_text(p: Path) -> str:
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


@dataclass(frozen=True)
class NewPredictor:
    column: str
    domain: str
    source: str
    kind: str
    missing: str
    timing: str
    provenance: str
    definition: str
    text: str
    text_he: str
    record_date: str | None = None
    replaces: str | None = None
    timing_reason: str = ""
    levels: tuple[float, ...] | None = None


@dataclass(frozen=True)
class Lineage:
    v1_column: str
    v21_column: str
    klass: str
    proven: bool
    evidence: dict[str, str]

    def v1_evidence(self) -> str:
        return " | ".join(f"{k[3:]}: {self.evidence[k]}" for k in ("v1_sql_expression", "v1_registry_ids", "v1_source_table", "v1_logic"))

    def v21_evidence(self) -> str:
        return " | ".join(f"{k[4:]}: {self.evidence[k]}" for k in ("v21_sql_expression", "v21_registry_ids", "v21_source_table", "v21_logic"))


def _documented(v: Any) -> bool:
    s = str(v or "").strip().upper()
    return bool(s) and not s.startswith(("NOT AVAILABLE", "NOT DOCUMENTED", "UNKNOWN"))


@dataclass(frozen=True)
class V21Schema:
    path: str
    sha256: str
    definition_path: str
    definition_sha256: str
    header: tuple[str, ...]
    meaning_he: dict[str, str]
    columns: dict[str, dict[str, Any]]
    domains: dict[str, dict[str, str]]
    availability: dict[str, dict[str, str]]
    bridges: dict[str, dict[str, Any]]
    caveats: tuple[str, ...]
    predictors: dict[str, NewPredictor]
    source_view: str
    lineage: dict[str, Lineage]             # removed V1 column -> its examined V21 counterpart and the proven-or-not lineage class

    def risk_of(self, p: NewPredictor) -> tuple[str, str]:
        a = self.availability.get(p.source) or {}
        return str(a.get("risk", "UNKNOWN")), str(a.get("assumption", "no availability declaration"))


def parse_definition(text: str) -> tuple[list[str], dict[str, str]]:
    """(the header line, column -> the Hebrew business definition of its SELECT comment)."""
    lines = text.replace("\r\n", "\n").split("\n")
    header = lines[0].split()
    meaning = {c: m.strip() for c, m in SELECT_RE.findall(text)}
    return header, meaning


def load_v21_schema(path: str | Path) -> V21Schema:
    p = resolve_path(path)
    raw = (yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("phase5_v21_schema") or {}
    problems: list[str] = []
    dp = resolve_path(raw.get("definition_file", ""))
    if not dp.is_file():
        raise ConfigError(f"{p}: the V21 definition file {raw.get('definition_file')!r} is missing")
    dsha = _sha_text(dp)
    if dsha != raw.get("definition_sha256"):
        problems.append(f"definition file sha256 {dsha[:16]} differs from the reviewed {str(raw.get('definition_sha256'))[:16]}")
    header, meaning = parse_definition(dp.read_text(encoding="utf-8"))
    select = list(meaning)
    cols = raw.get("columns") or {}
    if header != select:
        problems.append("the definition file header and its SELECT list differ")
    if list(cols) != header:
        problems.append(f"the reviewed columns differ from the authoritative header (missing {[c for c in header if c not in cols][:10]}, "
                        f"extra {[c for c in cols if c not in header][:10]}, or another order)")
    if len(set(header)) != len(header):
        problems.append("duplicate column names in the authoritative header")
    if int(raw.get("n_columns", -1)) != len(header):
        problems.append(f"n_columns {raw.get('n_columns')} != {len(header)}")
    domains = dict(raw.get("domains") or {})
    avail = dict(raw.get("availability") or {})
    preds: dict[str, NewPredictor] = {}
    for c, e in cols.items():
        e = e or {}
        has_old, has_new = "v1_equivalence" in e, "new" in e
        if has_old == has_new:
            problems.append(f"{c}: exactly one of v1_equivalence / new is required")
            continue
        if has_old and e["v1_equivalence"] not in EVIDENCE:
            problems.append(f"{c}: v1_equivalence must be one of {EVIDENCE}")
        if has_new:
            n = e["new"] or {}
            if n.get("role") not in NEW_ROLES:
                problems.append(f"{c}: new.role must be one of {NEW_ROLES}")
                continue
            if n["role"] != "PREDICTOR":
                if not n.get("reason"):
                    problems.append(f"{c}: a non-predictor new column needs a reason")
                continue
            bad = [k for k, allowed in (("kind", KINDS), ("timing", TIMINGS), ("provenance", PROVENANCE), ("missing", MISSING)) if n.get(k) not in allowed]
            if bad:
                problems.append(f"{c}: invalid {bad}")
            if n.get("domain") not in domains:
                problems.append(f"{c}: unknown domain {n.get('domain')!r}")
            if n.get("source") not in avail:
                problems.append(f"{c}: source {n.get('source')!r} has no availability declaration")
            if n.get("timing") == "record_date" and n.get("record_date") not in header:
                problems.append(f"{c}: record_date {n.get('record_date')!r} is not a V21 column")
            if n.get("timing") == "uncertain" and not n.get("timing_reason"):
                problems.append(f"{c}: uncertain timing needs timing_reason")
            if n.get("kind") == "ordinal" and not n.get("levels"):
                problems.append(f"{c}: ordinal needs levels")
            preds[c] = NewPredictor(column=c, domain=str(n.get("domain")), source=str(n.get("source")), kind=str(n.get("kind")), missing=str(n.get("missing")),
                                    timing=str(n.get("timing")), provenance=str(n.get("provenance")), definition=str(n.get("definition", "")),
                                    text=str(n.get("text", c)), text_he=str(n.get("text_he", n.get("text", c))), record_date=n.get("record_date"),
                                    replaces=n.get("replaces"), timing_reason=str(n.get("timing_reason", "")),
                                    levels=tuple(float(x) for x in n["levels"]) if n.get("levels") else None)
    lineage: dict[str, Lineage] = {}
    if list(raw.get("lineage_classes") or LINEAGE_CLASSES) != list(LINEAGE_CLASSES):
        problems.append(f"lineage_classes must be {list(LINEAGE_CLASSES)}")
    for v1c, spec in (raw.get("lineage") or {}).items():
        spec = spec or {}
        miss = [k for k in LINEAGE_FIELDS if k not in spec or (k != "lineage_proven" and not str(spec[k]).strip())]
        if miss:
            problems.append(f"lineage {v1c}: missing {miss}")
            continue
        k, v21c, proven = spec["class"], spec["v21_column"], spec["lineage_proven"]
        if k not in LINEAGE_CLASSES:
            problems.append(f"lineage {v1c}: class {k!r} not in {LINEAGE_CLASSES}")
        if not isinstance(proven, bool):
            problems.append(f"lineage {v1c}: lineage_proven must be true / false")
        if v1c in header:
            problems.append(f"lineage {v1c}: the V1 column is still in V21 (lineage is only for a removed V1 column)")
        if v21c not in preds:
            problems.append(f"lineage {v1c}: {v21c!r} is not a new V21 predictor")
        if k in SAME_LINEAGE:
            if proven is not True:
                problems.append(f"lineage {v1c}: {k} requires lineage_proven: true")
            undoc = [f for f in V1_EVIDENCE if not _documented(spec[f])]
            if undoc:
                problems.append(f"lineage {v1c}: {k} requires the V1 evidence {undoc} (a rename is never inferred from a position or a name)")
        if k in SAME_SOURCE and (str(spec["v1_registry_ids"]).strip() != str(spec["v21_registry_ids"]).strip()
                                 or str(spec["v1_source_table"]).strip() != str(spec["v21_source_table"]).strip()):
            problems.append(f"lineage {v1c}: {k} requires identical V1 / V21 registry IDs and source table")
        lineage[v1c] = Lineage(v1_column=v1c, v21_column=str(v21c), klass=str(k), proven=bool(proven),
                               evidence={f: str(spec[f]) for f in LINEAGE_FIELDS if f not in ("v21_column", "class", "lineage_proven")})
    by_v21 = {g.v21_column: g for g in lineage.values()}
    for c, pr in preds.items():
        g = by_v21.get(c)
        if pr.replaces and not (g and g.v1_column == pr.replaces and g.klass in SAME_LINEAGE and g.proven):
            problems.append(f"{c}: replaces {pr.replaces!r} without a PROVEN same-lineage entry in lineage (renames are never inferred)")
        if g and g.klass in SAME_LINEAGE and pr.replaces != g.v1_column:
            problems.append(f"{c}: lineage {g.v1_column} is {g.klass}: declare replaces: {g.v1_column}")
    for b, spec in (raw.get("bridges") or {}).items():
        if b in header:
            problems.append(f"bridge {b}: the column exists in V21 (a bridge is only for a removed V1 column)")
        if "value" not in (spec or {}) or not (spec or {}).get("reason"):
            problems.append(f"bridge {b}: value and reason required")
    if problems:
        raise ConfigError(f"{p}: invalid V21 schema: " + "; ".join(problems))
    return V21Schema(path=str(p), sha256=_sha_text(p), definition_path=str(dp), definition_sha256=dsha, header=tuple(header), meaning_he=meaning,
                     columns={c: dict(e or {}) for c, e in cols.items()}, domains=domains, availability=avail, bridges=dict(raw.get("bridges") or {}),
                     caveats=tuple(str(x) for x in raw.get("global_caveats") or []), predictors=preds, source_view=str(raw.get("source_view", "")),
                     lineage=lineage)


# ============================================================================ the diff
@dataclass
class SchemaDiff:
    classes: dict[str, str]                 # extract column -> one of CLASSES
    reasons: dict[str, str]
    table: pd.DataFrame                     # SCHEMA_DIFF_V1_V21
    classification: pd.DataFrame            # ALL_V21_COLUMN_CLASSIFICATION (one row per extract column; x_use filled by the preflight)
    removed: pd.DataFrame                   # REMOVED_V1_COLUMNS
    renamed_changed: pd.DataFrame           # RENAMED_OR_CHANGED_COLUMNS
    counts: dict[str, Any]
    unresolved: list[str]
    missing_from_extract: list[str]         # authoritative V21 columns the extract does not have
    extra_in_extract: list[str]             # extract columns the authoritative schema does not have
    header_matches: bool


def _v1_base(cc: Any) -> tuple[str, str]:
    if cc.role == "IDENTIFIER":
        return IDENT, "V1 role IDENTIFIER"
    if cc.role in ("LABEL", "FORBIDDEN_LEAKAGE") or cc.timing == "post_index":
        return FORBIDDEN, f"V1 role {cc.role}, timing {cc.timing}: outcome / follow-up / post-index or declared leakage risk"
    if cc.role in ("QA_CONTROL", "COHORT_ELIGIBILITY"):
        return META, f"V1 role {cc.role}: cohort / QA / provenance, never a predictor"
    return OLD_UNCHANGED, f"V1 role {cc.role}"


def schema_diff(header: list[str], schema: V21Schema, contract: Any, dictionary: Any, *, sealed_patterns: list[str],
                feature_inputs: dict[str, list[str]]) -> SchemaDiff:
    """The exact diff of the extract header against the V1 contract and the authoritative V21 schema (no row is read)."""
    pats = [re.compile(p, re.IGNORECASE) for p in sealed_patterns]
    v1 = list(contract.names)
    v1s, auth, hdr = set(v1), set(schema.header), set(header)
    used_by: dict[str, list[str]] = {}
    for f, ins in feature_inputs.items():
        for c in ins:
            used_by.setdefault(c, []).append(f)
    successors = {p.replaces: p.column for p in schema.predictors.values() if p.replaces}     # PROVEN same-lineage pairs only
    lin_v1 = schema.lineage
    lin_v21 = {g.v21_column: g for g in lin_v1.values()}
    classes: dict[str, str] = {}
    reasons: dict[str, str] = {}
    rows = []
    for c in header:
        e = schema.columns.get(c)
        if c in v1s:
            cc = contract.get(c)
            base, why = _v1_base(cc)
            if base == OLD_UNCHANGED:
                if e is None or "v1_equivalence" not in e:
                    cls, why = REVIEW, f"{why}; shared V1 predictor column without a review in the authoritative V21 schema"
                elif e["v1_equivalence"] == "CHANGED":
                    cls, why = OLD_CHANGED, f"{why}; V21 definition differs: {e.get('note', '')}"
                else:
                    cls, why = OLD_UNCHANGED, f"{why}; V1 -> V21 equivalence {e['v1_equivalence']}" + (f" ({e['note']})" if e.get("note") else "")
            else:
                cls = base
                if e is None:
                    why += "; not in the authoritative V21 schema (extract has an extra V1 column)"
                elif e.get("note"):
                    why += f"; {e['note']}"
        elif e is not None and "new" in e:
            n = e["new"]
            if n["role"] == "PREDICTOR":
                p = schema.predictors[c]
                g = lin_v21.get(c)
                if p.replaces and p.replaces in v1s and p.replaces not in hdr and g is not None and g.klass in SAME_SOURCE:
                    cls, why = REVIEW, (f"PROVEN {g.klass} of the removed V1 column {p.replaces}: the Phase 3 feature would be reproducible under the V21 "
                                        "name, which this package never does silently (OLD inputs are not aliased) - review required")
                elif p.replaces and p.replaces in v1s and p.replaces not in hdr:
                    cls, why = RENAMED, (f"PROVEN lineage {g.klass if g else '?'} of the removed V1 column {p.replaces} (V1 / V21 SQL, registry IDs, source and "
                                         f"logic compared): {p.definition}")
                else:
                    cls, why = NEW_CAND, f"V21 predictor not in V1: {p.definition}"
                    if p.replaces:
                        why += f" (declared predecessor {p.replaces} is still in the extract: not treated as a rename)"
                    if g is not None and g.klass == REMOVED_ADDED:
                        why += (f" [lineage to the removed V1 column {g.v1_column} examined: OLD_REMOVED_NEW_ADDED, "
                                f"{'proven different' if g.proven else 'equivalence NOT proven'} - a genuinely new predictor, never a rename]")
            else:
                cls, why = n["role"], f"new V21 column: {n.get('reason', '')}"
        elif any(pt.search(c) for pt in pats):
            cls, why = FORBIDDEN, "not in the authoritative V21 schema; the name looks like an outcome / future / follow-up field (sealed from X)"
        else:
            cls, why = REVIEW, "not in the authoritative V21 schema: meaning, timing and provenance unknown"
        classes[c], reasons[c] = cls, why
    for c in sorted(v1s | auth | hdr, key=lambda x: (schema.header.index(x) if x in auth else 10_000 + (v1.index(x) if x in v1s else 0), x)):
        in1, inA, inE = c in v1s, c in auth, c in hdr
        status = ("UNCHANGED_NAME" if in1 and inA else "NEW_IN_V21" if inA else "REMOVED_IN_V21" if in1 else "EXTRACT_ONLY")
        cc = contract.get(c) if in1 else None
        e = schema.columns.get(c) or {}
        dm = dictionary.columns.get(c) if (in1 and c in dictionary.columns) else None
        g = lin_v1.get(c) or lin_v21.get(c)
        rows.append({"column": c, "in_v1_contract": in1, "in_v21_authoritative": inA, "in_extract": inE, "name_status": status,
                     "v21_position": schema.header.index(c) + 1 if inA else None, "v1_role": cc.role if cc else "", "v1_sql_type": cc.sql if cc else "",
                     "v1_timing": cc.timing if cc else "", "v1_meaning_status": (dm or {}).get("status", ""),
                     "v1_meaning": (dm or {}).get("meaning", ""), "v1_to_v21_equivalence": e.get("v1_equivalence", ""),
                     "class_in_extract": classes.get(c, ""), "successor_in_v21": successors.get(c, ""),
                     "predecessor_in_v1": (schema.predictors[c].replaces or "") if c in schema.predictors else "",
                     "review_note": e.get("note", "") or (e.get("new") or {}).get("reason", "") or (e.get("new") or {}).get("definition", ""),
                     "phase3_features_reading_it": "; ".join(used_by.get(c, [])),
                     "lineage_pair": (f"{g.v1_column} (V1) -> {g.v21_column} (V21)" if g else ""), "lineage_class": g.klass if g else "",
                     "lineage_proven": (g.proven if g else ""), "lineage_v1_evidence": g.v1_evidence() if g else "",
                     "lineage_v21_evidence": g.v21_evidence() if g else "", "lineage_comparison": g.evidence["comparison"] if g else "",
                     "lineage_conclusion": g.evidence["conclusion"] if g else ""})
    table = pd.DataFrame(rows)
    cl = []
    for c in header:
        e = schema.columns.get(c) or {}
        p = schema.predictors.get(c)
        cl.append({"position_in_extract": header.index(c) + 1, "column": c, "class": classes[c], "reason": reasons[c],
                   "in_v1_contract": c in v1s, "in_v21_authoritative": c in auth, "v1_to_v21_equivalence": e.get("v1_equivalence", ""),
                   "domain": p.domain if p else "", "kind": p.kind if p else "", "timing_basis": (p.timing if p else ""),
                   "provenance": p.provenance if p else "", "replaces_v1_column": (p.replaces or "") if p else "",
                   "v21_definition_he": schema.meaning_he.get(c, ""), "phase3_features_reading_it": "; ".join(used_by.get(c, []))})
    classification = pd.DataFrame(cl)
    rem = []
    for c in v1:
        if c in hdr:
            continue
        cc = contract.get(c)
        dm = dictionary.columns.get(c) if c in dictionary.columns else None
        feats = used_by.get(c, [])
        succ = successors.get(c, "")
        g = lin_v1.get(c)
        bridged = c in schema.bridges
        lin_txt = (f"; lineage to V21 {g.v21_column} examined: {g.klass} ({'proven' if g.proven else 'NOT proven'})" if g else "")
        if bridged:
            cons = f"bridged: {schema.bridges[c]['reason']}"
        elif feats:
            cons = (f"Phase 3 feature(s) {feats} cannot be reproduced with the same meaning -> excluded from OLD (INELIGIBLE_DATA)" + lin_txt
                    + (f"; {g.v21_column} is a genuinely new V21 predictor (NEW_CANDIDATE_PREDICTOR)" if g and g.klass == REMOVED_ADDED else ""))
        else:
            cons = "not read by any Phase 3 feature: no effect on OLD" + lin_txt
        rem.append({"column": c, "v1_role": cc.role, "v1_timing": cc.timing, "v1_meaning": (dm or {}).get("meaning", ""),
                    "v1_note": getattr(cc, "note", "") or "", "successor_in_v21": succ, "lineage_v21_column": g.v21_column if g else "",
                    "lineage_class": g.klass if g else "", "lineage_proven": (g.proven if g else ""), "bridged": bridged,
                    "phase3_features_affected": "; ".join(feats), "consequence": cons})
    removed = pd.DataFrame(rem, columns=["column", "v1_role", "v1_timing", "v1_meaning", "v1_note", "successor_in_v21", "lineage_v21_column", "lineage_class",
                                         "lineage_proven", "bridged", "phase3_features_affected", "consequence"])
    rc = []
    for c in header:
        if classes[c] == RENAMED:
            p = schema.predictors[c]
            dm = dictionary.columns.get(p.replaces) if p.replaces in dictionary.columns else None
            g = lin_v21[c]
            rc.append({"v21_column": c, "change": RENAMED, "v1_column": p.replaces, "lineage_class": g.klass, "lineage_proven": g.proven,
                       "v1_meaning": (dm or {}).get("meaning", ""), "v1_note": getattr(contract.get(p.replaces), "note", "") or "", "v21_definition": p.definition,
                       "phase3_features_of_v1_column": "; ".join(used_by.get(p.replaces, [])),
                       "handling": f"PROVEN {g.klass}: the V1 Phase 3 feature is not reproducible (NOT in OLD); the V21 column is a NEW candidate under the "
                                   "ordinary eligibility rules"})
        elif c in lin_v21 and classes[c] == NEW_CAND:
            g = lin_v21[c]
            dm = dictionary.columns.get(g.v1_column) if g.v1_column in dictionary.columns else None
            rc.append({"v21_column": c, "change": f"NOT_A_RENAME ({g.klass})", "v1_column": g.v1_column, "lineage_class": g.klass, "lineage_proven": g.proven,
                       "v1_meaning": (dm or {}).get("meaning", ""), "v1_note": getattr(contract.get(g.v1_column), "note", "") or "",
                       "v21_definition": schema.predictors[c].definition, "phase3_features_of_v1_column": "; ".join(used_by.get(g.v1_column, [])),
                       "handling": f"lineage examined and {'proven different' if g.proven else 'NOT proven'}: {g.evidence['conclusion']}"})
        elif classes[c] == OLD_CHANGED:
            dm = dictionary.columns.get(c) if c in dictionary.columns else None
            feats = used_by.get(c, [])
            rc.append({"v21_column": c, "change": OLD_CHANGED, "v1_column": c, "v1_meaning": (dm or {}).get("meaning", ""),
                       "v1_note": getattr(contract.get(c), "note", "") or "", "v21_definition": (schema.columns.get(c) or {}).get("note", ""),
                       "phase3_features_of_v1_column": "; ".join(feats),
                       "handling": ("Phase 3 feature(s) kept with the V21 definition, flagged OLD_CHANGED_DEFINITION, present in EVERY feature set (cannot create "
                                    "an OLD vs NEW difference)" if feats else "not read by any Phase 3 feature: never in X")})
    renamed_changed = pd.DataFrame(rc, columns=["v21_column", "change", "v1_column", "lineage_class", "lineage_proven", "v1_meaning", "v1_note", "v21_definition",
                                                "phase3_features_of_v1_column", "handling"])
    unresolved = [c for c in header if classes[c] == REVIEW]
    vc = pd.Series(list(classes.values())).value_counts().to_dict()
    shared = [c for c in header if c in v1s]
    counts = {"v1_columns": len(v1), "v21_authoritative_columns": len(schema.header), "extract_columns": len(header),
              "shared_names": len(shared), "unchanged_columns": int(sum(classes[c] != OLD_CHANGED for c in shared)),
              "removed_v1_columns": int(len(removed)), "new_columns": int(sum(c not in v1s for c in header)),
              "renamed_or_replaced": int(vc.get(RENAMED, 0)), "changed_definition": int(vc.get(OLD_CHANGED, 0)),
              "new_candidate_predictors": int(vc.get(NEW_CAND, 0)), "unresolved_requires_semantic_review": len(unresolved),
              "lineage_pairs_examined": len(lin_v1), "lineage_by_class": {k: int(sum(g.klass == k for g in lin_v1.values())) for k in LINEAGE_CLASSES},
              "by_class": {k: int(vc.get(k, 0)) for k in CLASSES}}
    return SchemaDiff(classes=classes, reasons=reasons, table=table, classification=classification, removed=removed, renamed_changed=renamed_changed,
                      counts=counts, unresolved=unresolved, missing_from_extract=[c for c in schema.header if c not in hdr],
                      extra_in_extract=[c for c in header if c not in auth], header_matches=list(header) == list(schema.header))
