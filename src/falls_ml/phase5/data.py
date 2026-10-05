"""The 2026 analysis data: the exact V1 -> V21 schema diff, cohort, outcome, the OLD (Phase 3) feature universe and every NEW V21 predictor with its
timing class and feature-set membership.

X and the outcome are read SEPARATELY. X columns come only through the sealed reader (``falls_ml.phase4.sealed.read_columns``), which refuses any
sealed column (outcome / label / follow-up / censoring / identifier / future-looking names, and every column the schema diff classifies as
OUTCOME_OR_FUTURE_FORBIDDEN or IDENTIFIER); the outcome columns are read by the outcome reader and only ever become ``y`` and the outcome-contract
audit. Nothing here fits a model.

OLD    the Phase 3 universe (the 93-feature catalogue + BASELINE_15) rebuilt with the UNCHANGED Phase 1-3 code (adapter, ``phase2.engineer.compute``,
       ``phase3.recovery.build_recovery`` under the END-OF-INDEX-DAY contract, attestation re-checked on 2026 by V3). A feature whose V1 input was
       removed in V21 is not reproducible (INELIGIBLE_DATA); a removed VALIDATION-only input is bridged as the schema declares (Prior_Fall_Missing_Ind).
NEW    every V21 column the schema diff classes NEW_CANDIDATE_PREDICTOR or RENAMED_OR_REPLACED - exact names from the authoritative schema, never a
       hand-written short list: timing from its declared basis (row-level record date / attestation / uncertain), semantics from its declared type,
       coverage, the univariate leakage safety screen.
Timing classes: SAFE_VERIFIED, SAFE_BOUNDED, SAFE_ATTESTED, UNCERTAIN_TIMING, INELIGIBLE_TIMING, INELIGIBLE_SEMANTICS, INELIGIBLE_DATA, INELIGIBLE_LEAKAGE.
Sets: OLD = OLD features with a SAFE class; OLD_PLUS_ALL_NEW_ELIGIBLE adds every new predictor with a SAFE class or UNCERTAIN_TIMING (plausibly
known at the index day, not future-derived); OLD_PLUS_NEW_SAFE adds only SAFE-class new predictors with a DEFENSIBLE provenance.
A cell whose record is dated AFTER Index_Date (UNKNOWN at prediction time) or that cannot be read takes the feature's NO-RECORD state, so no
missingness pattern can carry the future record.
"""

from __future__ import annotations

import dataclasses
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from falls_ml.phase2.state import Phase2Stop
from falls_ml.phase5.schema import FORBIDDEN, IDENT, NEW_CAND, OLD_CHANGED, RENAMED, REVIEW, SchemaDiff, schema_diff

ID_COL = "Customer_Full_ID"
ID_COLS = ("Customer_Full_ID", "Index_Date", "Is_Eligible_Cohort")      # read for the key, cohort and duplicate checks only - never a feature
QA_COLS = ("Leakage_Check_Ind",)
UNREADABLE_PREFIX = "__unreadable__"

SAFE_VERIFIED, SAFE_BOUNDED, SAFE_ATTESTED = "SAFE_VERIFIED", "SAFE_BOUNDED", "SAFE_ATTESTED"
UNCERTAIN = "UNCERTAIN_TIMING"
I_TIMING, I_SEMANTICS, I_DATA, I_LEAKAGE = "INELIGIBLE_TIMING", "INELIGIBLE_SEMANTICS", "INELIGIBLE_DATA", "INELIGIBLE_LEAKAGE"
SAFE = (SAFE_VERIFIED, SAFE_BOUNDED, SAFE_ATTESTED)
P3_TO_BRIEF = {"SAFE_VERIFIED": SAFE_VERIFIED, "SAFE_VERIFIED_BOUNDED": SAFE_BOUNDED, "SAFE_ATTESTED": SAFE_ATTESTED,
               "NOT_RECOVERABLE_FUTURE_RECORDS": I_TIMING, "UNRESOLVED": I_TIMING, "NOT_RECOVERABLE_FORBIDDEN": I_LEAKAGE, "INELIGIBLE_DATA": I_DATA}
BASELINE_KIND = {"age_years": ("continuous", "none"), "polypharmacy_count_120d": ("count", "log1p")}
LINEAR_OF_KIND = {"binary": "none", "count": "log1p", "days": "log1p", "ordinal": "thermometer", "categorical": "onehot", "continuous": "none"}


def row_key(member_id: str) -> str:
    """Outcome-blind, order-independent key of a patient (the fold assignment sorts on it). Local only, never shared."""
    return hashlib.sha256(f"phase5-row|{member_id}".encode("utf-8")).hexdigest()[:24]


def feature_name(col: str) -> str:
    return "new_" + re.sub(r"[^0-9a-z]+", "_", col.lower()).strip("_")


# ============================================================================ sealing (X side)
def x_sealed_map(header: list[str], contract: Any, cfg: Any, diff: SchemaDiff | None = None) -> dict[str, str]:
    """column -> why it can never enter X (and is never requested by the X reader)."""
    xs = cfg["x_sealing"]
    roles = set(xs["contract_roles"])
    pats = [re.compile(p, re.IGNORECASE) for p in xs["name_patterns"]]
    sch = set(xs.get("schema_classes") or [])
    names = set(contract.names)
    out: dict[str, str] = {}
    for c in header:
        if c in set(xs["columns"]):
            out[c] = "BRIEF_OUTCOME_OR_FOLLOWUP_COLUMN"
        elif diff is not None and diff.classes.get(c) in sch and c not in ID_COLS:
            out[c] = f"V21_SCHEMA_CLASS_{diff.classes[c]}"
        elif c in names:
            cc = contract.get(c)
            if cc.role in roles:
                out[c] = f"CONTRACT_ROLE_{cc.role}"
            elif cc.timing == "post_index":
                out[c] = "CONTRACT_TIMING_POST_INDEX"
        elif (diff is None or c not in diff.classes or diff.classes[c] == REVIEW) and any(p.search(c) for p in pats):
            out[c] = "NEW_COLUMN_NAME_LOOKS_LIKE_OUTCOME_OR_FUTURE"
    return out


# ============================================================================ the analysis data
@dataclass
class Prepared:
    frame: pd.DataFrame                       # usable cohort rows: row_key + every candidate feature value (float) + subgroup columns
    y: np.ndarray
    registry: pd.DataFrame                    # one row per candidate feature (OLD universe + NEW V21) with its class and set membership
    catalogue: pd.DataFrame                   # NEW_FEATURE_CATALOGUE rows (every extract column the V1 contract does not have)
    undeclared: pd.DataFrame                  # REQUIRES_SEMANTIC_REVIEW columns (aggregate profile)
    meta: dict[str, dict[str, Any]]           # feature -> encoding / domain / class / availability / origin
    sealed: dict[str, str]
    diff: SchemaDiff | None = None
    checks: list[dict[str, Any]] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)
    outcome: dict[str, Any] = field(default_factory=dict)
    input: dict[str, Any] = field(default_factory=dict)

    @property
    def safe(self) -> bool:
        return not any(c["status"] == "STOP" for c in self.checks)

    def add(self, cid: str, title: str, status: str, details: list[str]) -> None:
        self.checks.append({"id": cid, "title": title, "status": status, "details": list(details)})


def baseline_names(mapping: Any) -> list[str]:
    return [f.canonical for f in mapping.features if f.include_in_baseline]


def universe_inputs(L: dict[str, Any]) -> dict[str, list[str]]:
    """Every raw column a Phase 3 universe feature reads (value + validation)."""
    from falls_ml.phase4.features import feature_inputs

    cat, mapping = L["cat"], L["mapping"]
    return {n: feature_inputs(n, cat, mapping) for n in [*cat.names, *baseline_names(mapping)]}


def value_inputs(L: dict[str, Any]) -> dict[str, list[str]]:
    """The raw columns a Phase 3 universe feature's VALUE is computed from (an absent indicator counts only where the adapter needs it: op count)."""
    from falls_ml.data.meuhedet_wide import feature_input_columns

    cat, mapping = L["cat"], L["mapping"]
    out = {n: list(cat.get(n).inputs) for n in cat.names}
    for f in mapping.features:
        if f.include_in_baseline:
            i = feature_input_columns(f)
            out[f.canonical] = [*i["value"], *(i["validation"] if f.op == "count" else ())]
    return out


def _parse_dates(s: pd.Series, formats: tuple[str, ...]) -> tuple[pd.Series, np.ndarray]:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s, np.zeros(len(s), dtype=bool)
    raw = s.astype("string").str.strip()
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[us]")
    todo = raw.notna() & (raw != "")
    for fmt in formats:
        if not todo.any():
            break
        got = pd.to_datetime(raw[todo], format=fmt, errors="coerce")
        ok = got.notna()
        out.loc[got.index[ok]] = got[ok].astype("datetime64[us]")
        todo.loc[got.index[ok]] = False
    unreadable = todo.to_numpy(dtype=bool)
    return out, unreadable


def _no_record_state(kind: str, missing: str, op: str = "copy") -> float:
    """The value a patient with nothing recorded has (used for an UNKNOWN / unreadable cell)."""
    if op == "present":
        return 0.0
    if missing in ("not_assessed", "no_event"):
        return np.nan
    return 0.0 if kind in ("binary", "count") else np.nan


def _univariate_auroc(x: np.ndarray, y: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    v = np.where(np.isfinite(x), x, (np.nanmin(x) - 1.0) if np.isfinite(x).any() else 0.0)
    if len(np.unique(v)) < 2 or len(np.unique(y)) < 2:
        return 0.5
    a = float(roc_auc_score(y, v))
    return max(a, 1.0 - a)


def prepare(src: Path, cfg: Any, L: dict[str, Any], *, input_info: dict[str, Any]) -> Prepared:
    from falls_ml.phase3.timecontract import predictor_date_columns
    from falls_ml.phase4.evaluate import outcome_contract, read_outcomes
    from falls_ml.phase4.sealed import read_columns, read_header

    contract, dictionary, cat = L["contract"], L["dictionary"], L["cat"]
    schema = cfg.schema
    idx = pd.Timestamp(cfg["index_date"])
    header = read_header(src)
    uinp = universe_inputs(L)
    vinp = value_inputs(L)
    diff = schema_diff(header, schema, contract, dictionary, sealed_patterns=list(cfg["x_sealing"]["name_patterns"]), feature_inputs=uinp)
    sealed = x_sealed_map(header, contract, cfg, diff)
    P = Prepared(frame=pd.DataFrame(), y=np.array([]), registry=pd.DataFrame(), catalogue=pd.DataFrame(), undeclared=pd.DataFrame(), meta={},
                 sealed=sealed, diff=diff, input=input_info)
    P.facts["schema"] = {**diff.counts, "header_matches_authoritative_v21": diff.header_matches, "missing_from_extract": diff.missing_from_extract,
                         "extra_in_extract": diff.extra_in_extract, "unresolved": diff.unresolved, "v21_schema_sha256": schema.sha256,
                         "v21_definition_sha256": schema.definition_sha256}
    P.facts["sealed"] = {"n_sealed": len(sealed), "by_reason": pd.Series(list(sealed.values())).value_counts().to_dict() if sealed else {},
                         "columns": sorted(sealed)}
    # ---- P0: the exact V1 -> V21 schema diff (STOP when a column's meaning is unresolved)
    c = diff.counts
    det = [f"V1 contract {c['v1_columns']} columns; authoritative V21 schema {c['v21_authoritative_columns']}; this extract {c['extract_columns']}"
           + (" (header IDENTICAL to the authoritative V21 header)" if diff.header_matches else " (header DIFFERS from the authoritative V21 header)"),
           f"unchanged {c['unchanged_columns']}; removed {c['removed_v1_columns']}; new {c['new_columns']}; renamed / replaced {c['renamed_or_replaced']}; "
           f"changed definition {c['changed_definition']}; new candidate predictors {c['new_candidate_predictors']}",
           f"classes: {c['by_class']}"]
    if diff.missing_from_extract:
        det.append(f"authoritative V21 columns absent from the extract: {diff.missing_from_extract[:15]} (features reading them are INELIGIBLE_DATA)")
    if diff.unresolved:
        det.append(f"REQUIRES_SEMANTIC_REVIEW: {diff.unresolved[:20]} - columns whose meaning / timing / provenance the authoritative schema does not "
                   "define; no clinical column may be ignored silently: review them and add them to configs/meuhedet/phase5_v21_schema.yaml")
    P.add("P0", "exact V1 -> V21 schema diff (every extract column classified)", "STOP" if diff.unresolved else ("OK" if diff.header_matches else "WARN"), det)
    P.undeclared = _undeclared(diff)
    P.catalogue = _catalogue(diff, pd.DataFrame(), schema)
    if diff.unresolved:
        return P
    oc = cfg["outcome_contract"]
    miss = [c for c in (ID_COL, "Index_Date", "Is_Eligible_Cohort", oc["label_column"]) if c not in header]
    P.add("P1", "required identifier / eligibility / label columns present", "STOP" if miss else "OK",
          [f"absent: {miss}"] if miss else ["Customer_Full_ID, Index_Date, Is_Eligible_Cohort and the label column are present"])
    if miss:
        return P

    # ---- the OLD universe: which features are reproducible under the 2026 schema
    bridges = {b: s for b, s in schema.bridges.items() if b not in header}
    present = set(header) | set(bridges)
    absent = {f: [c for c in ins if c not in present] for f, ins in vinp.items()}
    absent = {f: v for f, v in absent.items() if v}
    sealed_in = {f: [c for c in ins if c in sealed] for f, ins in uinp.items()}
    sealed_in = {f: v for f, v in sealed_in.items() if v}
    if sealed_in:
        P.add("P2", "no Phase 3 universe feature reads an outcome / future column", "STOP",
              [f"{f}: reads sealed {v}" for f, v in list(sealed_in.items())[:20]])
        return P
    changed_in = {f: [c for c in ins if diff.classes.get(c) == OLD_CHANGED] for f, ins in uinp.items()}
    changed_in = {f: v for f, v in changed_in.items() if v}
    policy = cfg["eligibility"]["old_changed_definition_policy"]
    P.add("P2", "Phase 3 universe under V21: no outcome / future input; removed / changed inputs handled explicitly", "OK",
          [f"{len(sealed)} columns sealed from X: {P.facts['sealed']['by_reason']}", "no universe feature reads a sealed column",
           f"not reproducible (V1 input removed in V21): {sorted(absent)}",
           f"bridged removed validation inputs: {sorted(bridges)}",
           f"inputs with a CHANGED V21 definition: {changed_in or 'none'} (policy {policy})"])
    base_all = baseline_names(L["mapping"])
    base_ok = [b for b in base_all if b not in absent and not (policy == "exclude" and b in changed_in)]
    cat_ok = tuple(f for f in cat.features if f.name not in absent and not (policy == "exclude" and f.name in changed_in))

    # ---- NEW V21 predictors: exact columns of the authoritative schema present in the extract
    new_cols = [c for c in header if diff.classes.get(c) in (NEW_CAND, RENAMED)]
    # ---- columns read for X: universe inputs, their record dates, V3 dates, QA, ids, new predictors and their record dates, unresolved-free extras
    cols: list[str] = [c for c in ID_COLS if c in header]
    for f, ins in uinp.items():
        if f in absent:
            continue
        for c in ins:
            if c in header and c not in cols:
                cols.append(c)
            rd = dictionary.record_date(c) if c in dictionary.columns else None
            if rd and rd in header and rd not in sealed and rd not in cols:
                cols.append(rd)
    v3_cols = [c for c in predictor_date_columns(dictionary, L["d00"], L["tc"], header) if c not in sealed]
    rec_cols = [schema.predictors[c].record_date for c in new_cols if schema.predictors[c].timing == "record_date"]
    for c in [*v3_cols, *QA_COLS, *new_cols, *rec_cols]:
        if c and c in header and c not in sealed and c not in cols:
            cols.append(c)
    P.facts["timing_check_columns"] = sorted({c for c in [*v3_cols, *rec_cols] if c in header})
    read = read_columns(src, cols, sealed, contract)
    fr = read.frame
    for b, spec in bridges.items():
        fr[b] = pd.array([spec["value"]] * len(fr), dtype="Int64")
    # ---- cohort
    elig = ((fr["Index_Date"].dt.normalize() == idx).fillna(False) & (pd.to_numeric(fr["Is_Eligible_Cohort"], errors="coerce") == 1).fillna(False)).to_numpy()
    n_el = int(elig.sum())
    ids = fr.loc[elig, ID_COL].astype("string")
    dup = int(ids.dropna().duplicated().sum())
    n_null_id = int(ids.isna().sum())
    P.facts["cohort"] = {"extract_rows": int(len(fr)), "rows_on_index_date": int((fr["Index_Date"].dt.normalize() == idx).fillna(False).sum()),
                         "eligible_on_index_date": n_el, "duplicated_ids_among_eligible": dup, "null_ids_among_eligible": n_null_id,
                         "distinct_index_dates": int(fr["Index_Date"].dt.normalize().nunique(dropna=False))}
    ccfg = cfg["cohort"]
    st = "STOP" if n_el < int(ccfg["min_eligible_rows"]) or ((dup or n_null_id) and ccfg["stop_if_duplicate_ids"]) else "OK"
    P.add("P3", f"cohort on Index_Date {cfg['index_date']}: eligible rows, one snapshot per patient", st,
          [f"{len(fr)} extract rows; {n_el} eligible on {cfg['index_date']} (Is_Eligible_Cohort = 1)",
           f"{dup} duplicated Customer_Full_ID and {n_null_id} NULL ids among them (any -> HARD STOP: one row per patient at the index date)"])
    if dup or n_null_id:
        return P
    # ---- QA: the VIEW's own leakage flag (partial by definition - never taken as proof of safety)
    qa, qst = [], "OK"
    if "Leakage_Check_Ind" in fr.columns:
        nl = int((pd.to_numeric(fr.loc[elig, "Leakage_Check_Ind"], errors="coerce").fillna(0) != 0).sum())
        qa.append(f"Leakage_Check_Ind non-zero on {nl} eligible rows (the VIEW checks four latest dates only - never taken as proof of safety)")
        qst = "STOP" if nl else qst
    else:
        qa.append("Leakage_Check_Ind absent (Phase 5 relies on its own sealing and timing gates)")
    qa.append("Definition_Version is not a V21 column: the extract is identified by its exact header against the authoritative V21 schema (P0)")
    P.add("P4", "the VIEW's own leakage flag", qst, qa)
    if not P.safe:
        return P

    # ---- outcome: read separately, audited by the contract, never part of X
    ocfg = {"outcome_contract": oc, "index_date": cfg["index_date"]}
    O = read_outcomes(src, contract, ocfg)
    if len(O) != len(fr) or not (O[ID_COL].astype("string").fillna("").to_numpy() == fr[ID_COL].astype("string").fillna("").to_numpy()).all():
        raise Phase2Stop("OUTCOME_ALIGNMENT", "the outcome rows do not align with the predictor rows (file read twice gave different rows)")
    Oe = O.loc[elig].reset_index(drop=True)
    contract_res = outcome_contract(Oe, ocfg)
    contract_res["extra_audit"] = _outcome_extra(Oe, oc, idx)
    fz = _followup_zero_tolerance(Oe, oc)
    contract_res["checks"]["O4_positive_after_followup_end"] = fz
    contract_res["hard_failures"] = [h for h in contract_res["hard_failures"] if not str(h).startswith("O4")]
    if not fz["passed"]:
        contract_res["hard_failures"].append(fz["message"])
    contract_res["passed"] = not contract_res["hard_failures"]
    P.facts["followup_end_audit"] = {k: fz[k] for k in ("positives", "positives_event_after_followup_end", "pct_of_positives", "rule")}
    P.outcome = contract_res
    yv = pd.to_numeric(Oe[oc["label_column"]], errors="coerce").to_numpy(dtype=float)
    usable = np.isin(yv, (0.0, 1.0))
    P.add("P5", "2026 outcome contract (before any model is fitted)", "OK" if contract_res["passed"] else "STOP",
          [*(contract_res["hard_failures"] or ["O1-O6 passed: outcome strictly after the index day, within Index_Date + 180, consistent dates, enough usable "
                                              "events; NO positive after the personal Followup_End_Date (zero tolerance)"]),
           f"positives with the event after Followup_End_Date: {fz['count_text']} of {fz['positives']} ({fz['pct_text']}; zero tolerance - labels "
           "never altered, patients never excluded)",
           f"usable labelled patients {int(usable.sum())}, events {int((yv == 1).sum())}, censored / unlabelled {int((~usable).sum())} (labels never rewritten)"])
    if not contract_res["passed"]:
        return P

    fe = fr.loc[elig].reset_index(drop=True).loc[usable].reset_index(drop=True)
    y = yv[usable].astype(int)
    unr = read.unreadable.loc[elig].reset_index(drop=True).loc[usable].reset_index(drop=True) if read.unreadable is not None else pd.DataFrame(index=fe.index)
    n = len(fe)
    # ---- V3 attestation on 2026 (record dates only, the unchanged Phase 3 rule over the Phase 3 predictor record dates)
    from falls_ml.phase4.features import v3_check

    v3 = v3_check(fe, v3_cols, cfg["index_date"])
    P.facts["v3_attestation"] = {"passed": bool(v3["passed"]), "rows_with_any_post_index_record": int(v3["rows_with_any_post_index_record"]),
                                 "columns_checked": len(v3["by_column"]),
                                 "columns_with_post_index_records": sorted(c for c, v in v3["by_column"].items() if v["n_after_index"])}
    P.add("P6", "V3: no predictor record dated after Index_Date (the DWH attestation, re-checked on 2026)", "OK" if v3["passed"] else "WARN",
          [f"{len(v3['by_column'])} record-date columns checked; rows with any post-index record: {v3['rows_with_any_post_index_record']}",
           "attestation HOLDS: attested features (OLD and NEW) may be SAFE_ATTESTED; uncertain-timing NEW predictors may be UNCERTAIN_TIMING"
           if v3["passed"] else
           f"attestation WITHDRAWN: every attested / uncertain feature becomes INELIGIBLE_TIMING (post-index records in "
           f"{P.facts['v3_attestation']['columns_with_post_index_records'][:8]})"])

    # ---- OLD universe values + Phase 3 classes
    old_vals, old_reg = _old_universe(fe, unr, L, base_ok, cat_ok, cfg, v3_passed=bool(v3["passed"]))
    reg_rows = []
    meta: dict[str, dict[str, Any]] = {}
    for _, r in old_reg.iterrows():
        f = str(r["feature"])
        kind, lin, lev, miss_sem, op, dom = _old_encoding(f, cat, r)
        chg = changed_in.get(f, [])
        reg_rows.append({"feature": f, "raw_columns": "; ".join(uinp[f]), "origin": "OLD_PHASE3_UNIVERSE", "domain": dom, "kind": kind,
                         "v21_class": OLD_CHANGED if chg else "OLD_UNCHANGED", "class": r["class"],
                         "reason": r["reason"] + (f"; V21 CHANGED the definition of {chg}: kept with the V21 definition (policy {policy})" if chg else ""),
                         "availability_risk": r.get("availability_risk", "UNKNOWN"), "availability_assumption": r.get("availability_assumption", ""),
                         "n_unknown_cells": int(r.get("n_unknown", 0)), "phase3_class": r.get("phase3_class", ""), "provenance": r.get("provenance", ""),
                         "new_provenance": ""})
        meta[f] = {"kind": kind, "linear": lin, "levels": lev, "missing": miss_sem, "op": op, "domain": dom, "origin": "OLD"}
    for f, cs in absent.items():
        dom = cat.get(f).domain if f in cat.names else "BASELINE_15"
        lin = [schema.lineage[c] for c in cs if c in schema.lineage]
        lin_txt = "; ".join(f"lineage to V21 {g.v21_column}: {g.klass} ({'proven' if g.proven else 'equivalence NOT proven'})" for g in lin)
        reg_rows.append({"feature": f, "raw_columns": "; ".join(uinp[f]), "origin": "OLD_PHASE3_UNIVERSE", "domain": dom, "kind": "", "v21_class": "REMOVED_IN_V21",
                         "class": I_DATA, "reason": f"V1 input column(s) {cs} removed in V21: not reproducible with an equivalent meaning"
                         + (f" ({lin_txt}: the V21 field is a separate, genuinely new candidate, never relabelled as this OLD feature)" if lin else ""),
                         "availability_risk": "", "availability_assumption": "", "n_unknown_cells": 0, "phase3_class": "", "provenance": "", "new_provenance": ""})

    # ---- NEW V21 predictors
    new_vals, new_reg = _new_candidates(fe, cfg, new_cols, diff, idx=idx, v3_passed=bool(v3["passed"]), gates=L["rules"].gates,
                                        date_formats=tuple(contract.date_formats))
    for r in new_reg:
        m = r.pop("_meta")
        reg_rows.append(r)
        meta[r["feature"]] = m
    values = pd.concat([old_vals, new_vals], axis=1)
    # ---- coverage + the leakage safety screen (usable cohort; the screen can only EXCLUDE)
    el = cfg["eligibility"]
    lim_auc = float(el["leakage_univariate_auroc"])
    screened = set(el["all_new_classes"]) | set(el["safe_classes"])
    reg = pd.DataFrame(reg_rows)
    reg["univariate_auroc"] = np.nan
    reg["n_known_observed"] = 0
    reg["missing_pct"] = np.nan
    reg["prevalence_or_median"] = ""
    minobs = int(el["min_known_observed_rows"])
    for i, r in reg.iterrows():
        f = r["feature"]
        if f not in values.columns:
            continue
        x = values[f].to_numpy(dtype=float)
        obs = np.isfinite(x)
        reg.at[i, "n_known_observed"] = int(obs.sum())
        nm = int((~obs).sum())
        reg.at[i, "missing_pct"] = round(100.0 * float((~obs).mean()), 2) if not (0 < nm < 10 or 0 < int(obs.sum()) < 10) else np.nan
        reg.at[i, "prevalence_or_median"] = _dist_text(x, meta[f]["kind"])
        if r["class"] in screened:
            if int(obs.sum()) < minobs or len(np.unique(x[obs])) <= 1:
                reg.at[i, "class"] = I_DATA
                reg.at[i, "reason"] = f"{int(obs.sum())} KNOWN non-NULL rows (< {minobs}) or constant"
                continue
            a = _univariate_auroc(x, y)
            reg.at[i, "univariate_auroc"] = round(a, 4)
            if a >= lim_auc:
                reg.at[i, "class"] = I_LEAKAGE
                reg.at[i, "reason"] = (f"single-feature AUROC {a:.3f} >= {lim_auc:g}: implausibly strong alone (future / outcome-derived information "
                                       "suspected) - excluded, DWH review required")
    # ---- set membership with a reason for every inclusion / exclusion
    reg["in_OLD"], reg["in_OLD_PLUS_ALL_NEW_ELIGIBLE"], reg["in_OLD_PLUS_NEW_SAFE"], reg["set_reason"] = False, False, False, ""
    for i, r in reg.iterrows():
        a, s, why = _membership(r, el)
        if r["origin"] == "OLD_PHASE3_UNIVERSE":
            reg.at[i, "in_OLD"] = a
            reg.at[i, "in_OLD_PLUS_ALL_NEW_ELIGIBLE"] = a
            reg.at[i, "in_OLD_PLUS_NEW_SAFE"] = a
        else:
            reg.at[i, "in_OLD_PLUS_ALL_NEW_ELIGIBLE"] = a
            reg.at[i, "in_OLD_PLUS_NEW_SAFE"] = s
        reg.at[i, "set_reason"] = why
    for f in list(meta):
        rr = reg.loc[reg["feature"] == f].iloc[0]
        meta[f]["class"] = str(rr["class"])
        meta[f]["availability_risk"] = str(rr["availability_risk"])
    P.registry = reg
    # ---- local analysis frame (row-level, never shared)
    keys = fe[ID_COL].astype("string").map(row_key)
    frame = pd.DataFrame({"row_key": keys.to_numpy()})
    keep = [f for f in values.columns if f in meta]
    frame = pd.concat([frame, values[keep].reset_index(drop=True)], axis=1)
    frame = pd.concat([frame, _subgroups(fe, values, meta)], axis=1)
    P.frame, P.y, P.meta = frame, y, {f: meta[f] for f in keep}
    P.catalogue = _catalogue(diff, reg, schema)
    P.facts["usable"] = {"n": int(n), "events": int(y.sum()), "non_events": int(n - y.sum()), "prevalence": float(y.mean()) if n else 0.0}
    nw = reg[reg["origin"] == "NEW_V21"]
    n_old_ok = int(reg["in_OLD"].sum())
    P.facts["new_predictors"] = {"genuine_new_predictors": int(len(nw)), "all_new_eligible": int(nw["in_OLD_PLUS_ALL_NEW_ELIGIBLE"].sum()),
                                 "new_safe": int(nw["in_OLD_PLUS_NEW_SAFE"].sum()),
                                 "excluded_from_all_new": nw.loc[~nw["in_OLD_PLUS_ALL_NEW_ELIGIBLE"].astype(bool), ["raw_columns", "class", "set_reason"]]
                                 .rename(columns={"raw_columns": "column"}).to_dict("records"),
                                 "all_new_only": nw.loc[nw["in_OLD_PLUS_ALL_NEW_ELIGIBLE"].astype(bool) & ~nw["in_OLD_PLUS_NEW_SAFE"].astype(bool),
                                                        ["raw_columns", "class", "set_reason"]].rename(columns={"raw_columns": "column"}).to_dict("records"),
                                 "new_non_predictor_columns": [{"column": c, "class": k, "reason": diff.reasons[c]} for c, k in diff.classes.items()
                                                               if c not in set(contract.names) and k not in (NEW_CAND, RENAMED)]}
    P.add("P7", "feature eligibility (timing, provenance, semantics, coverage, leakage) and set membership", "OK",
          [f"OLD (Phase 3 universe) eligible: {n_old_ok} of {int((reg['origin'] == 'OLD_PHASE3_UNIVERSE').sum())}",
           f"genuine new V21 predictors: {len(nw)}; in OLD_PLUS_ALL_NEW_ELIGIBLE {int(nw['in_OLD_PLUS_ALL_NEW_ELIGIBLE'].sum())}; in OLD_PLUS_NEW_SAFE "
           f"{int(nw['in_OLD_PLUS_NEW_SAFE'].sum())}; every exclusion has a reason (FEATURE_ELIGIBILITY.csv, NEW_FEATURE_CATALOGUE.csv)",
           f"classes: {reg['class'].value_counts().to_dict()}"])
    return P


def _membership(r: Any, el: dict[str, Any]) -> tuple[bool, bool, str]:
    """(in the ALL_NEW-style set, in the SAFE set, reason). OLD features: the first flag is OLD membership (all three sets)."""
    k = str(r["class"])
    safe, alln = set(el["safe_classes"]), set(el["all_new_classes"])
    if r["origin"] == "OLD_PHASE3_UNIVERSE":
        if k in safe:
            return True, True, f"OLD: Phase 3 universe feature, class {k} (in all three feature sets)"
        return False, False, f"excluded from OLD (and therefore from every set): {k} - {r['reason']}"
    prov = str(r.get("new_provenance", ""))
    if k in alln:
        if k in safe and prov in set(el["safe_provenance"]):
            return True, True, f"NEW: {k}, provenance {prov} -> OLD_PLUS_ALL_NEW_ELIGIBLE and OLD_PLUS_NEW_SAFE"
        why = ("timing uncertain at the index day" if k == UNCERTAIN else f"provenance {prov} (not verifiably new / validated information)")
        return True, False, f"NEW: {k} -> OLD_PLUS_ALL_NEW_ELIGIBLE only; excluded from OLD_PLUS_NEW_SAFE: {why}"
    return False, False, f"NEW predictor excluded from every set: {k} - {r['reason']}"


# ============================================================================ helpers
def _old_encoding(f: str, cat: Any, r: Any) -> tuple[str, str, list[float] | None, str, str, str]:
    if f in cat.names:
        d = cat.get(f)
        return d.kind, d.linear, list(d.levels) if d.levels else None, d.missing, d.op, d.domain
    kind, lin = BASELINE_KIND.get(f, ("binary", "none"))
    return kind, lin, None, "unexpected", "copy", "BASELINE_15"


def _old_universe(fe: pd.DataFrame, unr: pd.DataFrame, L: dict[str, Any], base: list[str], cat_feats: tuple[Any, ...], cfg: Any, *,
                  v3_passed: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    from falls_ml.data.meuhedet_wide import MeuhedetWideDatasetAdapter
    from falls_ml.phase2.engineer import compute
    from falls_ml.phase3.cohort import canonical_baseline
    from falls_ml.phase3.recovery import build_recovery
    from falls_ml.phase4.features import feature_inputs

    mapping, contract, cat = L["mapping"], L["contract"], L["cat"]
    frame = fe.copy()
    for c in unr.columns:
        frame[UNREADABLE_PREFIX + c] = unr[c].to_numpy(dtype=bool)
    adapter = MeuhedetWideDatasetAdapter(mapping, contract, L["espec"], features=base)
    canon, problems = canonical_baseline(adapter, mapping, frame, base)
    for b in base:
        frame[b] = canon[b].to_numpy()
    cat2 = dataclasses.replace(cat, features=cat_feats)
    values = pd.DataFrame({f.name: compute(f, frame).to_numpy() for f in cat_feats}, index=frame.index)
    rec = build_recovery(frame, values, catalogue=cat2, mapping=mapping, baseline=base, contract=contract, dictionary=L["dictionary"], d00=L["d00"],
                         rules=L["rules"], train_mask=np.ones(len(frame), dtype=bool), min_observed_train=int(cfg["eligibility"]["min_known_observed_rows"]),
                         tc=L["tc"], attestation_holds=v3_passed)
    reg = rec.registry.copy()
    unknown = rec.unknown.copy()
    lim = float(cfg["eligibility"]["max_unreadable_share"])
    cols_out: dict[str, np.ndarray] = {}
    rows = []
    for _, r in reg.iterrows():
        f = str(r["feature"])
        if f in cat.names:
            x = values[f].to_numpy(dtype=float).copy()
            d = cat.get(f)
            kind, miss, op = d.kind, d.missing, d.op
        else:
            s = canon[f]
            x = ((s.astype(str) == "female").astype(float) if f == "sex" else pd.to_numeric(s, errors="coerce")).to_numpy(dtype=float).copy()
            kind, miss, op = BASELINE_KIND.get(f, ("binary", "none"))[0], "unexpected", "copy"
        unk = unknown[f].to_numpy(dtype=bool) if f in unknown.columns else np.zeros(len(frame), dtype=bool)
        bad = np.zeros(len(frame), dtype=bool)
        for c in feature_inputs(f, cat, mapping):
            if UNREADABLE_PREFIX + c in frame.columns:
                bad |= frame[UNREADABLE_PREFIX + c].to_numpy(dtype=bool)
        mask = unk | bad
        x[mask] = _no_record_state(kind, miss, op)
        cols_out[f] = x
        klass = P3_TO_BRIEF.get(str(r["phase3_class"]), I_TIMING)
        reason = str(r["phase3_reason"])
        if bad.mean() > lim and klass in SAFE:
            klass, reason = I_DATA, f"{100 * bad.mean():.2f}% unreadable / not-allowed input cells (> {100 * lim:g}%)"
        rows.append({"feature": f, "class": klass, "reason": reason, "phase3_class": r["phase3_class"], "availability_risk": r["availability_risk"],
                     "availability_assumption": r["availability_assumption"], "n_unknown": int(mask.sum()),
                     "provenance": f"sources {r['sources']}; record dates {r['record_dates'] or 'none'}; evidence {r['evidence_class']}"})
    return pd.DataFrame(cols_out, index=frame.index), pd.DataFrame(rows)


def _new_candidates(fe: pd.DataFrame, cfg: Any, new_cols: list[str], diff: SchemaDiff, *, idx: pd.Timestamp, v3_passed: bool, gates: dict[str, Any],
                    date_formats: tuple[str, ...]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    schema = cfg.schema
    lim_unr = float(cfg["eligibility"]["max_unreadable_share"])
    fmts = date_formats
    n = len(fe)
    vals = pd.DataFrame(index=fe.index)
    reg: list[dict[str, Any]] = []
    for col in new_cols:
        p = schema.predictors[col]
        name = feature_name(col)
        risk, assumption = schema.risk_of(p)
        raw = fe[col].astype("string") if col in fe.columns else pd.Series(pd.NA, index=fe.index, dtype="string")
        num = pd.to_numeric(raw.str.strip(), errors="coerce").to_numpy(dtype=float)
        present = (raw.notna() & (raw.str.strip() != "")).to_numpy()
        unreadable = present & ~np.isfinite(num)
        if p.kind == "binary":
            unreadable |= np.isfinite(num) & ~np.isin(num, (0.0, 1.0))
        elif p.kind == "ordinal" and p.levels:
            unreadable |= np.isfinite(num) & ~np.isin(num, np.asarray(p.levels, dtype=float))
        elif p.kind == "count":
            unreadable |= np.isfinite(num) & ((num < 0) | (num != np.floor(num)))
        x = np.where(unreadable, np.nan, num)
        unknown = np.zeros(n, dtype=bool)
        tbasis = p.timing
        meta = {"kind": p.kind, "linear": LINEAR_OF_KIND[p.kind], "levels": list(p.levels) if p.levels else None, "missing": p.missing, "op": "copy",
                "domain": p.domain, "origin": "NEW", "provenance": p.provenance, "v21_class": diff.classes[col]}
        if unreadable.mean() > lim_unr:
            klass, reason = I_SEMANTICS, (f"{100 * unreadable.mean():.2f}% of the cells do not fit the declared type {p.kind} (> {100 * lim_unr:g}%): "
                                          "semantics not as the V21 schema declares")
        elif p.timing == "record_date":
            rd = p.record_date
            if rd in fe.columns:
                d, dbad = _parse_dates(fe[rd], fmts)
                off = (d.dt.normalize() - idx).dt.days
                rec = present & np.isfinite(num) & (num != 0)
                unknown = (off > 0).fillna(False).to_numpy() | (rec & d.isna().to_numpy()) | dbad
                tbasis = f"record_date {rd} (latest record of the source, row-level)"
                inf = int((np.isfinite(x) & (x != 0) & ~unknown).sum())
                nu = int(unknown.sum())
                sp, si = nu / max(1, n), nu / max(1, nu + inf)
                if sp > float(gates["max_unknown_share_population"]) or si > float(gates["max_unknown_share_informative"]):
                    klass = I_TIMING
                    reason = f"post-index / undated source records on {nu} rows ({100 * sp:.2f}% of the cohort, {100 * si:.1f}% of informative rows): Phase 3 gates G2 / G3"
                elif nu:
                    klass, reason = SAFE_BOUNDED, f"{nu} rows with a post-index / undated source record take the no-record state (within the Phase 3 gates)"
                else:
                    klass, reason = SAFE_VERIFIED, f"every row verified on or before the index day ({rd})"
            else:
                klass, reason = (SAFE_ATTESTED, f"record date {rd} absent from the extract: eligible on the DWH attestation (V3 passed)") if v3_passed else \
                    (I_TIMING, f"record date {rd} absent and the attestation is withdrawn (V3)")
        elif p.timing == "attested":
            klass, reason = ((SAFE_ATTESTED, "status at the index day without a row-level date: eligible on the DWH attestation, re-checked on 2026 by V3")
                             if v3_passed else (I_TIMING, "no row-level date and the attestation is withdrawn: V3 found post-index records"))
        else:
            klass, reason = ((UNCERTAIN, f"plausibly known at the index day and not future-derived, but timing not bounded: {p.timing_reason}")
                             if v3_passed else (I_TIMING, f"uncertain timing ({p.timing_reason}) and the attestation is withdrawn (V3)"))
        x = np.where(unknown | unreadable, _no_record_state(p.kind, p.missing), x)
        vals[name] = x
        meta["timing_basis"] = tbasis
        reg.append({"feature": name, "raw_columns": col, "origin": "NEW_V21", "domain": p.domain, "kind": p.kind, "v21_class": diff.classes[col],
                    "class": klass, "reason": reason, "availability_risk": risk, "availability_assumption": assumption,
                    "n_unknown_cells": int((unknown | unreadable).sum()), "phase3_class": "", "new_provenance": p.provenance,
                    "provenance": f"V21 schema {col}: {p.definition}" + (f" (replaces V1 {p.replaces})" if diff.classes[col] == RENAMED else "")
                    + f"; timing {tbasis}", "_meta": meta})
    return vals, reg


def _catalogue(diff: SchemaDiff, reg: pd.DataFrame, schema: Any) -> pd.DataFrame:
    """NEW_FEATURE_CATALOGUE: one row per extract column the V1 contract does not have (+ every removed V1 column, for completeness)."""
    rows = []
    byraw = {str(r["raw_columns"]): r for _, r in reg.iterrows() if r["origin"] == "NEW_V21"} if len(reg) else {}
    for c, cls in diff.classes.items():
        if bool(diff.table.loc[diff.table["column"] == c, "in_v1_contract"].iloc[0]):
            continue
        p = schema.predictors.get(c)
        r = byraw.get(c)
        e = (schema.columns.get(c) or {}).get("new") or {}
        rows.append({"raw_column": c, "v21_class": cls, "engineered_feature": r["feature"] if r is not None else "", "domain": p.domain if p else "",
                     "old_or_new": "NEW_V21" if cls != RENAMED else f"RENAMED_FROM_V1 {p.replaces}",
                     "semantic_status": (f"DEFINED BY THE V21 SCHEMA ({p.provenance})" if p else (e.get("reason") or diff.reasons[c])),
                     "timing_status": (r["class"] if r is not None else ("SEALED" if cls in (FORBIDDEN, IDENT) else "NOT_A_PREDICTOR")),
                     "availability_status": f"{r['availability_risk']}: {r['availability_assumption']}" if r is not None else "",
                     "missingness": (f"{r['missing_pct']}%" if r is not None and pd.notna(r["missing_pct"]) else ""),
                     "prevalence_or_distribution": r["prevalence_or_median"] if r is not None else "",
                     "eligibility": r["class"] if r is not None else cls,
                     "in_OLD_PLUS_ALL_NEW_ELIGIBLE": bool(r["in_OLD_PLUS_ALL_NEW_ELIGIBLE"]) if r is not None else False,
                     "in_OLD_PLUS_NEW_SAFE": bool(r["in_OLD_PLUS_NEW_SAFE"]) if r is not None else False,
                     "inclusion_or_exclusion_reason": r["set_reason"] if r is not None else diff.reasons[c],
                     "provenance": r["provenance"] if r is not None else "V21 schema", "v21_definition_he": schema.meaning_he.get(c, ""),
                     "notes": p.text if p else ""})
    for _, r in diff.removed.iterrows():
        rows.append({"raw_column": r["column"], "v21_class": "REMOVED_IN_V21", "engineered_feature": "", "domain": "", "old_or_new": "V1_ONLY",
                     "semantic_status": r["v1_meaning"], "timing_status": "", "availability_status": "", "missingness": "", "prevalence_or_distribution": "",
                     "eligibility": "NOT_IN_EXTRACT", "in_OLD_PLUS_ALL_NEW_ELIGIBLE": False, "in_OLD_PLUS_NEW_SAFE": False,
                     "inclusion_or_exclusion_reason": r["consequence"], "provenance": "V1 contract", "v21_definition_he": "", "notes": ""})
    return pd.DataFrame(rows)


def _undeclared(diff: SchemaDiff) -> pd.DataFrame:
    rows = []
    for c in diff.unresolved:
        rows.append({"column": c, "class": REVIEW, "reason": diff.reasons[c]})
    return pd.DataFrame(rows, columns=["column", "class", "reason"])


def _dist_text(x: np.ndarray, kind: str) -> str:
    obs = x[np.isfinite(x)]
    if not len(obs):
        return "no observed value"
    if 0 < len(obs) < 10:
        return "suppressed (<10 observed)"
    if kind == "binary":
        k = int((obs == 1).sum())
        if 0 < k < 10 or 0 < len(obs) - k < 10:
            return "suppressed (a cell < 10)"
        return f"{100 * float((obs == 1).mean()):.1f}% = 1 among observed"
    if kind == "categorical":
        return f"{len(np.unique(obs))} distinct codes"
    q = np.percentile(obs, [25, 50, 75])
    return f"median {q[1]:g} (IQR {q[0]:g}-{q[2]:g})"


def _followup_zero_tolerance(o: pd.DataFrame, oc: dict[str, Any]) -> dict[str, Any]:
    """O4 (Phase 5, zero tolerance): ANY positive whose event lies after the personal Followup_End_Date stops the run (aggregate count / percent only).

    V21 labels a fall even after the personal follow-up end by construction; Phase 5 neither relabels nor excludes such patients - it stops for review.
    A NULL / 2999 sentinel Followup_End_Date is open follow-up (V21: NULL = no end candidate). The column itself is required.
    """
    from falls_ml.phase4.evaluate import SENTINEL_YEAR

    rule = ("zero tolerance: any positive with its event after the personal Followup_End_Date -> STOP - REVIEW REQUIRED (no tolerance, no relabelling, "
            "no exclusion; aggregate count and percentage only)")
    y = pd.to_numeric(o[oc["label_column"]], errors="coerce") if oc["label_column"] in o else pd.Series(np.nan, index=o.index)
    pos = (y == 1).fillna(False).to_numpy()
    npos = int(pos.sum())
    if oc["followup_end_column"] not in o or oc["event_date_column"] not in o:
        return {"passed": False, "positives": npos, "positives_event_after_followup_end": 0, "pct_of_positives": 0.0, "rule": rule, "count_text": "n/a",
                "pct_text": "n/a",
                "message": f"O4: {oc['followup_end_column']} / {oc['event_date_column']} not in the extract: positives after the follow-up end cannot be ruled out"}
    fe = o[oc["followup_end_column"]]
    ev = o[oc["event_date_column"]].dt.normalize()
    real = fe.notna() & (fe.dt.year < SENTINEL_YEAR)
    after = pos & (real & (ev > fe.dt.normalize())).fillna(False).to_numpy()
    n = int(after.sum())
    pct = 100.0 * n / npos if npos else 0.0
    small = 0 < n < 10                                       # the share small-cell rule: 1-9 is reported as '<10' (and the % as an upper bound)
    count_text = "<10" if small else str(n)
    pct_text = (f"<{100.0 * 10 / npos:.2f}%" if small else f"{pct:.2f}%") if npos else "0.00%"
    return {"passed": n == 0, "positives": npos, "positives_event_after_followup_end": n, "pct_of_positives": pct_text if small else round(pct, 4),
            "count_text": count_text, "pct_text": pct_text, "limit": 0, "rule": rule, "positives_followup_end_open": int((pos & ~real.to_numpy()).sum()),
            "message": (f"O4: {count_text} positive(s) ({pct_text} of {npos} positives) have the event after the personal Followup_End_Date "
                        "(zero tolerance) - STOP, review the V21 label definition; labels are not altered and no patient is excluded")}


def _outcome_extra(o: pd.DataFrame, oc: dict[str, Any], idx: pd.Timestamp) -> dict[str, Any]:
    """Aggregate audit items not covered by O1-O7: events on/before the index day among all labels, beyond the nominal horizon, class balance."""
    y = pd.to_numeric(o[oc["label_column"]], errors="coerce")
    ev = o[oc["event_date_column"]].dt.normalize() if oc["event_date_column"] in o else pd.Series(pd.NaT, index=o.index)
    dte = (ev - idx).dt.days
    n = int(y.isin([0, 1]).sum())
    return {"labelled_rows": n, "events": int((y == 1).sum()), "non_events": int((y == 0).sum()), "censored_or_unlabelled": int((~y.isin([0, 1])).sum()),
            "event_dates_on_or_before_index_any_label": int((dte <= 0).fillna(False).sum()),
            "event_dates_beyond_nominal_horizon_any_label": int((dte > int(oc["window_days"])).fillna(False).sum()),
            "prevalence": float((y == 1).sum() / n) if n else None}


def _subgroups(fe: pd.DataFrame, values: pd.DataFrame, meta: dict[str, Any]) -> pd.DataFrame:
    out = pd.DataFrame(index=values.index)
    age = values["age_years"].to_numpy(dtype=float) if "age_years" in values else np.full(len(values), np.nan)
    out["sg_age"] = np.select([age < 65, age < 75, age < 85, age >= 85], ["<65", "65-74", "75-84", "85+"], default="unknown")
    sx = values["sex"].to_numpy(dtype=float) if "sex" in values else np.full(len(values), np.nan)
    out["sg_sex"] = np.where(sx == 1, "female", np.where(sx == 0, "male", "unknown"))
    pf = values["falls"].to_numpy(dtype=float) if "falls" in values else np.full(len(values), np.nan)
    out["sg_prior_fall"] = np.where(pf == 1, "yes", np.where(pf == 0, "no", "unknown"))
    g = values["frail_mefi_group"].to_numpy(dtype=float) if "frail_mefi_group" in values else np.full(len(values), np.nan)
    out["sg_mefi_group"] = np.where(np.isfinite(g), np.char.add("MEFI ", np.nan_to_num(g).astype(int).astype(str)), "not assessed")
    fa = values["form_mefi_assessed"].to_numpy(dtype=float) if "form_mefi_assessed" in values else np.full(len(values), np.nan)
    out["sg_frailty_assessed"] = np.where(fa == 1, "assessed", np.where(fa == 0, "not assessed", "unknown"))
    forms = [f for f in values.columns if f.startswith("form_")]
    anyf = values[forms].fillna(0).to_numpy(dtype=float).max(axis=1) if forms else np.zeros(len(values))
    out["sg_data_coverage"] = np.where(anyf > 0, "any nurse / MEFI assessment", "no assessment recorded")
    return out
