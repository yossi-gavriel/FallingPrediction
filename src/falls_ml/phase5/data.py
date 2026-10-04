"""The 2026 analysis data: cohort, outcome, the OLD (Phase 3) feature universe and the NEW (V21) candidates with their eligibility classes.

X and the outcome are read SEPARATELY. X columns come only through the sealed reader (``falls_ml.phase4.sealed.read_columns``), which refuses any
sealed column (outcome / label / follow-up / censoring / death / future-looking names); the outcome columns are read by the outcome reader and only
ever become ``y`` and the outcome-contract audit. Nothing here fits a model.

OLD    the Phase 3 universe (the 93-feature catalogue + BASELINE_15) rebuilt with the UNCHANGED Phase 1-3 code (adapter, ``phase2.engineer.compute``,
       ``phase3.recovery.build_recovery`` under the corrected END-OF-INDEX-DAY contract, attestation re-checked on 2026 by V3).
NEW    a 2026 column that is not in the 2025 contract and matches the pre-declared V21 catalogue; timing from its record date / day count / the
       attestation; semantics from its declared type; coverage; the univariate leakage screen.
Classes (brief §D): SAFE_VERIFIED, SAFE_BOUNDED, SAFE_ATTESTED, INELIGIBLE_TIMING, INELIGIBLE_SEMANTICS, INELIGIBLE_DATA, INELIGIBLE_LEAKAGE.
A cell whose record is dated AFTER Index_Date (UNKNOWN at prediction time) or that cannot be read takes the feature's NO-RECORD state (0 for
indicators / counts that have no NULL state, NULL otherwise): the post-index record is never used, and such rows look exactly like patients with
nothing recorded (no missingness pattern can carry the future record).
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

ID_COL = "Customer_Full_ID"
ID_COLS = ("Customer_Full_ID", "Snapshot_Key", "Index_Date", "Is_Eligible_Cohort")
QA_COLS = ("Leakage_Check_Ind", "Definition_Version")
UNREADABLE_PREFIX = "__unreadable__"
STOPWORDS = {"ind", "dx", "flag"}

SAFE_VERIFIED, SAFE_BOUNDED, SAFE_ATTESTED = "SAFE_VERIFIED", "SAFE_BOUNDED", "SAFE_ATTESTED"
I_TIMING, I_SEMANTICS, I_DATA, I_LEAKAGE = "INELIGIBLE_TIMING", "INELIGIBLE_SEMANTICS", "INELIGIBLE_DATA", "INELIGIBLE_LEAKAGE"
P3_TO_BRIEF = {"SAFE_VERIFIED": SAFE_VERIFIED, "SAFE_VERIFIED_BOUNDED": SAFE_BOUNDED, "SAFE_ATTESTED": SAFE_ATTESTED,
               "NOT_RECOVERABLE_FUTURE_RECORDS": I_TIMING, "UNRESOLVED": I_TIMING, "NOT_RECOVERABLE_FORBIDDEN": I_LEAKAGE, "INELIGIBLE_DATA": I_DATA}
BASELINE_KIND = {"age_years": ("continuous", "none"), "polypharmacy_count_120d": ("count", "log1p")}


def row_key(member_id: str) -> str:
    """Outcome-blind, order-independent key of a patient (the fold assignment sorts on it). Local only, never shared."""
    return hashlib.sha256(f"phase5-row|{member_id}".encode("utf-8")).hexdigest()[:24]


def tokens(name: str) -> frozenset[str]:
    return frozenset(t for t in re.split(r"[_\W]+", name.lower()) if t and t not in STOPWORDS)


# ============================================================================ sealing (X side)
def x_sealed_map(header: list[str], contract: Any, cfg: Any) -> dict[str, str]:
    """column -> why it can never enter X (and is never requested by the X reader)."""
    xs = cfg["x_sealing"]
    roles = set(xs["contract_roles"])
    pats = [re.compile(p, re.IGNORECASE) for p in xs["name_patterns"]]
    names = set(contract.names)
    out: dict[str, str] = {}
    for c in header:
        if c in set(xs["columns"]):
            out[c] = "BRIEF_OUTCOME_OR_FOLLOWUP_COLUMN"
        elif c in names:
            cc = contract.get(c)
            if cc.role in roles:
                out[c] = f"CONTRACT_ROLE_{cc.role}"
            elif cc.timing == "post_index":
                out[c] = "CONTRACT_TIMING_POST_INDEX"
        elif any(p.search(c) for p in pats):
            out[c] = "NEW_COLUMN_NAME_LOOKS_LIKE_OUTCOME_OR_FUTURE"
    return out


# ============================================================================ the analysis data
@dataclass
class Prepared:
    frame: pd.DataFrame                       # usable cohort rows: row_key + every candidate feature value (float) + subgroup columns
    y: np.ndarray
    registry: pd.DataFrame                    # one row per candidate feature (OLD universe + NEW V21) with its class
    catalogue: pd.DataFrame                   # NEW_FEATURE_CATALOGUE rows (every 2026 column outside the Phase 3 universe)
    undeclared: pd.DataFrame
    meta: dict[str, dict[str, Any]]           # feature -> encoding / domain / class / availability / origin
    sealed: dict[str, str]
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
    from falls_ml.phase4.features import feature_inputs

    cat, mapping = L["cat"], L["mapping"]
    return {n: feature_inputs(n, cat, mapping) for n in [*cat.names, *baseline_names(mapping)]}


def _parse_dates(s: pd.Series, formats: tuple[str, ...]) -> tuple[pd.Series, np.ndarray]:
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
    from falls_ml.phase3.timecontract import availability_of, predictor_date_columns
    from falls_ml.phase4.evaluate import outcome_contract, read_outcomes
    from falls_ml.phase4.sealed import read_columns, read_header

    contract, mapping, dictionary, cat = L["contract"], L["mapping"], L["dictionary"], L["cat"]
    idx = pd.Timestamp(cfg["index_date"])
    header = read_header(src)
    sealed = x_sealed_map(header, contract, cfg)
    P = Prepared(frame=pd.DataFrame(), y=np.array([]), registry=pd.DataFrame(), catalogue=pd.DataFrame(), undeclared=pd.DataFrame(), meta={},
                 sealed=sealed, input=input_info)
    P.facts["sealed"] = {"n_sealed": len(sealed), "by_reason": pd.Series(list(sealed.values())).value_counts().to_dict() if sealed else {},
                         "columns": sorted(sealed)}
    oc = cfg["outcome_contract"]
    miss = [c for c in (*ID_COLS[:1], "Index_Date", "Is_Eligible_Cohort", oc["label_column"]) if c not in header]
    P.add("P1", "required identifier / eligibility / label columns present", "STOP" if miss else "OK",
          [f"absent: {miss}"] if miss else ["Customer_Full_ID, Index_Date, Is_Eligible_Cohort and the label column are present"])
    if miss:
        return P

    # ---- the OLD universe: which features are reproducible under the 2026 schema
    uinp = universe_inputs(L)
    absent = {f: [c for c in ins if c not in header] for f, ins in uinp.items()}
    absent = {f: v for f, v in absent.items() if v}
    sealed_in = {f: [c for c in ins if c in sealed] for f, ins in uinp.items()}
    sealed_in = {f: v for f, v in sealed_in.items() if v}
    if sealed_in:
        P.add("P2", "no Phase 3 universe feature reads an outcome / future column", "STOP",
              [f"{f}: reads sealed {v}" for f, v in list(sealed_in.items())[:20]])
        return P
    P.add("P2", "no Phase 3 universe feature reads an outcome / future column", "OK",
          [f"{len(sealed)} columns sealed from X: {P.facts['sealed']['by_reason']}", "no universe feature reads a sealed column"])
    base_all = baseline_names(mapping)
    base_ok = [b for b in base_all if b not in absent]
    cat_ok = tuple(f for f in cat.features if f.name not in absent)

    # ---- the V21 matches (columns outside the 2025 contract)
    names = set(contract.names)
    new_cols = [c for c in header if c not in names]
    matches, ambiguous, v21_old = _match_v21(cfg.v21, header, names, new_cols)
    # ---- columns read for X: universe inputs, their record dates, V3 dates, QA, ids, every non-sealed new column (profiled)
    cols: list[str] = [c for c in ID_COLS if c in header]
    for f, ins in uinp.items():
        if f in absent:
            continue
        for c in ins:
            if c not in cols:
                cols.append(c)
            rd = dictionary.record_date(c) if c in dictionary.columns else None
            if rd and rd in header and rd not in sealed and rd not in cols:
                cols.append(rd)
    v3_cols = [c for c in predictor_date_columns(dictionary, L["d00"], L["tc"], header) if c not in sealed]
    for c in [*v3_cols, *QA_COLS, *[c for c in new_cols if c not in sealed]]:
        if c in header and c not in cols:
            cols.append(c)
    purchase_cols = [c for c in cfg["eligibility"]["medication_purchase_date_columns"] if c in header and c not in sealed]
    for c in purchase_cols:
        if c not in cols:
            cols.append(c)
    read = read_columns(src, cols, sealed, contract)
    fr = read.frame
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
           f"{dup} duplicated Customer_Full_ID and {n_null_id} NULL ids among them (any -> STOP: one row per patient is required; with repeated rows a "
           "grouped split would be needed)"])
    # ---- QA: definition version and the VIEW's own leakage flag (not sufficient on its own)
    qa, qst = [], "OK"
    if "Definition_Version" in fr.columns:
        vals = fr.loc[elig, "Definition_Version"].astype("string").fillna("NULL").value_counts().to_dict()
        P.facts["definition_version"] = {str(k): int(v) for k, v in vals.items()}
        want = cfg["expected_definition_version"]
        if set(vals) != {want}:
            qa.append(f"Definition_Version on eligible rows {vals}: expected only {want!r} - the extracted object is NOT the intended V21 definition")
            qst = "STOP"
        else:
            qa.append(f"Definition_Version = {want!r} on every eligible row (the extracted object matches the intended definition)")
    else:
        qa.append("Definition_Version absent: the extracted VIEW definition cannot be confirmed")
        qst = "STOP"
    if "Leakage_Check_Ind" in fr.columns:
        nl = int((pd.to_numeric(fr.loc[elig, "Leakage_Check_Ind"], errors="coerce").fillna(0) != 0).sum())
        qa.append(f"Leakage_Check_Ind non-zero on {nl} eligible rows (the VIEW's own partial check - never taken as proof of safety)")
        qst = "STOP" if nl else qst
    else:
        qa.append("Leakage_Check_Ind absent (the VIEW's partial check cannot be confirmed; Phase 5 relies on its own sealing and timing gates)")
    P.add("P4", "extract definition (V21) and the VIEW's leakage flag", qst, qa)
    if not P.safe:
        return P

    # ---- outcome: read separately, audited by the contract, never part of X
    ocfg = {"outcome_contract": oc, "index_date": cfg["index_date"]}
    O = read_outcomes(src, contract, ocfg)
    if len(O) != len(fr) or not (O[ID_COL].astype("string").fillna("") .to_numpy() == fr[ID_COL].astype("string").fillna("").to_numpy()).all():
        raise Phase2Stop("OUTCOME_ALIGNMENT", "the outcome rows do not align with the predictor rows (file read twice gave different rows)")
    Oe = O.loc[elig].reset_index(drop=True)
    contract_res = outcome_contract(Oe, ocfg)
    contract_res["extra_audit"] = _outcome_extra(Oe, oc, idx)
    P.outcome = contract_res
    yv = pd.to_numeric(Oe[oc["label_column"]], errors="coerce").to_numpy(dtype=float)
    usable = np.isin(yv, (0.0, 1.0))
    P.add("P5", "2026 outcome contract (before any model is fitted)", "OK" if contract_res["passed"] else "STOP",
          [*(contract_res["hard_failures"] or ["O1-O6 passed: outcome strictly after the index day, within Index_Date + 180, consistent dates, enough usable events"]),
           f"usable labelled patients {int(usable.sum())}, events {int((yv == 1).sum())}, censored / unlabelled {int((~usable).sum())}"])
    if not contract_res["passed"]:
        return P

    fe = fr.loc[elig].reset_index(drop=True).loc[usable].reset_index(drop=True)
    y = yv[usable].astype(int)
    unr = read.unreadable.loc[elig].reset_index(drop=True).loc[usable].reset_index(drop=True) if read.unreadable is not None else pd.DataFrame(index=fe.index)
    n = len(fe)
    # ---- V3 attestation on 2026 (record dates only, the unchanged Phase 3 rule over the Phase 3 predictor record dates). A V21 record date never
    # withdraws the attestation of the OLD universe: its post-index records make ITS OWN feature UNKNOWN / INELIGIBLE_TIMING (gates G2 / G3), and
    # the attested V21 features additionally require that no declared V21 record date holds a post-index record.
    v21_dates: dict[str, tuple[str, pd.Series, np.ndarray]] = {}
    for col, (f, _) in matches.items():
        for dc in _date_candidates(f, col):
            if dc in fe.columns and dc not in sealed and dc not in names:
                d, bad = _parse_dates(fe[dc], tuple(contract.date_formats))
                v21_dates[col] = (dc, d, bad)
                break
    from falls_ml.phase4.features import v3_check

    v3 = v3_check(fe, v3_cols, cfg["index_date"])
    v21_post = {}
    for col, (dc, d, _) in v21_dates.items():
        off = (d.dt.normalize() - idx).dt.days
        after = (off > 0).fillna(False).to_numpy()
        v21_post[dc] = {"n_after_index": int(after.sum()), "n_on_index": int((off == 0).fillna(False).sum()), "max_days_after": int(off[after].max()) if after.any() else 0}
    post_domains = sorted({matches[col][0].domain for col, (dc, _, _) in v21_dates.items() if v21_post[dc]["n_after_index"]})
    v21_ok = not post_domains
    P.facts["v3_attestation"] = {"passed": bool(v3["passed"]), "rows_with_any_post_index_record": int(v3["rows_with_any_post_index_record"]),
                                 "columns_checked": len(v3["by_column"]),
                                 "columns_with_post_index_records": sorted(c for c, v in v3["by_column"].items() if v["n_after_index"]),
                                 "v21_record_dates": v21_post, "v21_domains_with_post_index_dates": post_domains}
    P.add("P6", "V3: no predictor record dated after Index_Date (the DWH attestation, re-checked on 2026)", "OK" if v3["passed"] and v21_ok else "WARN",
          [f"{len(v3['by_column'])} Phase 3 record-date columns checked; rows with any post-index record: {v3['rows_with_any_post_index_record']}",
           "attestation HOLDS for the Phase 3 universe: its attested features may be SAFE_ATTESTED" if v3["passed"] else
           f"attestation WITHDRAWN: every attested OLD feature becomes INELIGIBLE_TIMING (post-index records in {P.facts['v3_attestation']['columns_with_post_index_records'][:8]})",
           f"V21 record dates: {len(v21_post)} found; post-index records in {sorted(k for k, v in v21_post.items() if v['n_after_index'])} -> "
           + ("attested V21 features stay eligible" if v3["passed"] and v21_ok else
              f"attested V21 features of the domains {post_domains} (or of every domain, if V3 failed) become INELIGIBLE_TIMING")])

    # ---- OLD universe values + Phase 3 classes
    old_vals, old_reg = _old_universe(fe, unr, L, base_ok, cat_ok, cfg, v3_passed=bool(v3["passed"]))
    reg_rows = []
    meta: dict[str, dict[str, Any]] = {}
    for _, r in old_reg.iterrows():
        f = str(r["feature"])
        kind, lin, lev, miss_sem, op, dom = _old_encoding(f, cat, r)
        reg_rows.append({"feature": f, "raw_columns": "; ".join(uinp[f]), "origin": "OLD_PHASE3_UNIVERSE", "domain": dom, "kind": kind,
                         "class": r["class"], "reason": r["reason"], "availability_risk": r.get("availability_risk", "UNKNOWN"),
                         "availability_assumption": r.get("availability_assumption", ""), "n_unknown_cells": int(r.get("n_unknown", 0)),
                         "phase3_class": r.get("phase3_class", ""), "provenance": r.get("provenance", "")})
        meta[f] = {"kind": kind, "linear": lin, "levels": lev, "missing": miss_sem, "op": op, "domain": dom, "origin": "OLD",
                   "process": bool(op == "present" or f in set(cfg["ablations"]["NO_ASSESSMENT_PROCESS"]["features"]))}
    for f, cs in absent.items():
        dom = cat.get(f).domain if f in cat.names else "BASELINE_15"
        reg_rows.append({"feature": f, "raw_columns": "; ".join(uinp[f]), "origin": "OLD_PHASE3_UNIVERSE", "domain": dom, "kind": "",
                         "class": I_DATA, "reason": f"input column(s) absent from the 2026 extract: {cs} (not reproducible under the 2026 schema)",
                         "availability_risk": "", "availability_assumption": "", "n_unknown_cells": 0, "phase3_class": "", "provenance": ""})

    # ---- NEW V21 candidates
    new_vals, new_reg, cat_rows, undeclared = _new_candidates(fe, unr, cfg, contract, uinp, sealed, header, new_cols, matches, ambiguous, v21_old,
                                                              v21_dates, v3_passed=bool(v3["passed"]), post_domains=set(post_domains), purchase_cols=purchase_cols, idx=idx,
                                                              availability_of=availability_of, L=L)
    for r in new_reg:
        reg_rows.append(r)
        if r["feature"] in new_vals:
            m = r.pop("_meta")
            meta[r["feature"]] = m
        else:
            r.pop("_meta", None)
    values = pd.concat([old_vals, new_vals], axis=1)
    # ---- leakage screen + coverage (usable cohort; the screen can only EXCLUDE)
    lim_auc = float(cfg["eligibility"]["leakage_univariate_auroc"])
    reg = pd.DataFrame(reg_rows)
    reg["univariate_auroc"] = np.nan
    reg["n_known_observed"] = 0
    reg["missing_pct"] = np.nan
    reg["prevalence_or_median"] = ""
    minobs = int(cfg["eligibility"]["min_known_observed_rows"])
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
        if r["class"] in (SAFE_VERIFIED, SAFE_BOUNDED, SAFE_ATTESTED):
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
    for f in list(meta):
        meta[f]["class"] = str(reg.loc[reg["feature"] == f, "class"].iloc[0])
        meta[f]["availability_risk"] = str(reg.loc[reg["feature"] == f, "availability_risk"].iloc[0])
    P.registry = reg
    # ---- local analysis frame (row-level, never shared)
    keys = fe[ID_COL].astype("string").map(row_key)
    frame = pd.DataFrame({"row_key": keys.to_numpy()})
    keep = [f for f in values.columns if f in meta]
    frame = pd.concat([frame, values[keep].reset_index(drop=True)], axis=1)
    frame = pd.concat([frame, _subgroups(fe, values, meta)], axis=1)
    P.frame, P.y, P.meta = frame, y, {f: meta[f] for f in keep}
    if len(cat_rows):
        cat_rows = cat_rows.copy()
        for i, r in cat_rows.iterrows():
            f = r["engineered_feature"]
            if f and f in set(reg["feature"]):
                rr = reg.loc[reg["feature"] == f].iloc[0]
                cat_rows.at[i, "eligibility"] = rr["class"]
                cat_rows.at[i, "exclusion_reason"] = "" if rr["class"] in (SAFE_VERIFIED, SAFE_BOUNDED, SAFE_ATTESTED) else rr["reason"]
                cat_rows.at[i, "missingness"] = f"{rr['missing_pct']}%" if pd.notna(rr["missing_pct"]) else ""
                cat_rows.at[i, "prevalence_or_distribution"] = rr["prevalence_or_median"]
    P.catalogue, P.undeclared = cat_rows, undeclared
    P.facts["usable"] = {"n": int(n), "events": int(y.sum()), "non_events": int(n - y.sum()), "prevalence": float(y.mean()) if n else 0.0}
    n_old_ok = int(((reg["origin"] == "OLD_PHASE3_UNIVERSE") & reg["class"].isin(cfg["eligibility"]["primary_classes"])).sum())
    n_new_ok = int(((reg["origin"] == "NEW_V21") & reg["class"].isin(cfg["eligibility"]["primary_classes"])).sum())
    P.add("P7", "feature eligibility (timing, provenance, semantics, coverage, leakage)", "OK",
          [f"OLD (Phase 3 universe) eligible under the Phase 3 primary rule: {n_old_ok} of {int((reg['origin'] == 'OLD_PHASE3_UNIVERSE').sum())}",
           f"NEW V21 candidates eligible: {n_new_ok} of {int((reg['origin'] == 'NEW_V21').sum())} declared matches; "
           f"{len(undeclared)} undeclared new columns (INELIGIBLE_SEMANTICS, listed for review)",
           f"classes: {reg['class'].value_counts().to_dict()}"])
    return P


# ============================================================================ helpers
def _match_v21(v21: Any, header: list[str], contract_names: set[str], new_cols: list[str]) -> tuple[dict[str, tuple[Any, str]], dict[str, list[str]], dict[str, str]]:
    """column -> (declared entry, how it matched); ambiguous entries; entries whose column is a 2025 contract column (OLD)."""
    lower = {c.lower(): c for c in header}
    by_tok: dict[frozenset[str], list[str]] = {}
    for c in new_cols:
        by_tok.setdefault(tokens(c), []).append(c)
    out: dict[str, tuple[Any, str]] = {}
    amb: dict[str, list[str]] = {}
    old: dict[str, str] = {}
    for f in v21.features:
        found = None
        how = ""
        for nm in (f.column, *f.aliases):
            if nm.lower() in lower:
                found, how = lower[nm.lower()], ("exact name" if nm == f.column else f"declared alias {nm}")
                break
        if found is None:
            cands = sorted({c for nm in (f.column, *f.aliases) for c in by_tok.get(tokens(nm), [])})
            if len(cands) == 1:
                found, how = cands[0], "same name tokens (case / '_' / Ind / Dx ignored) - CONFIRM"
            elif len(cands) > 1:
                amb[f.column] = cands
        if found is None:
            continue
        if found in contract_names:
            old[f.column] = found
            continue
        if found in out:
            amb[f.column] = [found]
            continue
        out[found] = (f, how)
    return out, amb, old


def _date_candidates(f: Any, col: str) -> list[str]:
    stem = re.sub(r"(_Dx)?_Ind$", "", col)
    return list(dict.fromkeys([*f.record_date, f"{stem}_Date", f"{stem}_Dx_Date", f"Last_{stem}_Date", f"First_{stem}_Date", f"{stem}_Diagnosis_Date"]))


def _old_encoding(f: str, cat: Any, r: Any) -> tuple[str, str, list[float] | None, str, str, str]:
    if f in cat.names:
        d = cat.get(f)
        lin = d.linear
        return d.kind, lin, list(d.levels) if d.levels else None, d.missing, d.op, d.domain
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
        if bad.mean() > lim and klass in (SAFE_VERIFIED, SAFE_BOUNDED, SAFE_ATTESTED):
            klass, reason = I_DATA, f"{100 * bad.mean():.2f}% unreadable / not-allowed input cells (> {100 * lim:g}%)"
        rows.append({"feature": f, "class": klass, "reason": reason, "phase3_class": r["phase3_class"], "availability_risk": r["availability_risk"],
                     "availability_assumption": r["availability_assumption"], "n_unknown": int(mask.sum()),
                     "provenance": f"sources {r['sources']}; record dates {r['record_dates'] or 'none'}; evidence {r['evidence_class']}"})
    return pd.DataFrame(cols_out, index=frame.index), pd.DataFrame(rows)


def _new_candidates(fe: pd.DataFrame, unr: pd.DataFrame, cfg: Any, contract: Any, uinp: dict[str, list[str]], sealed: dict[str, str], header: list[str],
                    new_cols: list[str], matches: dict[str, tuple[Any, str]], ambiguous: dict[str, list[str]], v21_old: dict[str, str],
                    v21_dates: dict[str, tuple[str, pd.Series, np.ndarray]], *, v3_passed: bool, post_domains: set[str], purchase_cols: list[str], idx: pd.Timestamp,
                    availability_of: Any, L: dict[str, Any]) -> tuple[pd.DataFrame, list[dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    v21 = cfg.v21
    gates = L["rules"].gates
    lim_unr = float(cfg["eligibility"]["max_unreadable_share"])
    n = len(fe)
    vals = pd.DataFrame(index=fe.index)
    reg: list[dict[str, Any]] = []
    cat_rows: list[dict[str, Any]] = []
    universe_cols = {c for ins in uinp.values() for c in ins}
    date_cols_used = {dc for dc, _, _ in v21_dates.values()}
    purchase_after = 0
    for c in purchase_cols:
        d, _ = _parse_dates(fe[c].astype("string"), tuple(contract.date_formats)) if not pd.api.types.is_datetime64_any_dtype(fe[c]) else (fe[c], None)
        purchase_after += int(((d.dt.normalize() - idx).dt.days > 0).fillna(False).sum())
    for col, (f, how) in matches.items():
        name = "new_" + re.sub(r"[^0-9a-z]+", "_", col.lower()).strip("_")
        risk, assumption = v21.risk_of(f)
        raw = fe[col].astype("string") if col in fe.columns else pd.Series(pd.NA, index=fe.index, dtype="string")
        num = pd.to_numeric(raw.str.strip(), errors="coerce").to_numpy(dtype=float)
        present = (raw.notna() & (raw.str.strip() != "")).to_numpy()
        unreadable = present & ~np.isfinite(num)
        if f.kind == "binary":
            unreadable |= np.isfinite(num) & ~np.isin(num, (0.0, 1.0))
        elif f.kind in ("ordinal", "categorical") and f.levels:
            unreadable |= np.isfinite(num) & ~np.isin(num, np.asarray(f.levels, dtype=float))
        elif f.kind == "count":
            unreadable |= np.isfinite(num) & ((num < 0) | (num != np.floor(num)))
        x = np.where(unreadable, np.nan, num)
        unknown = np.zeros(n, dtype=bool)
        klass, reason, tbasis = SAFE_VERIFIED, "", f.timing
        dom_meta = {"kind": f.kind, "linear": {"binary": "none", "count": "log1p", "days": "log1p", "ordinal": "thermometer", "categorical": "onehot",
                                               "continuous": "none"}[f.kind],
                    "levels": list(f.levels) if f.levels else None, "missing": f.missing, "op": "copy", "domain": f.domain, "origin": "NEW",
                    "process": f.process}
        timing_status = ""
        if unreadable.mean() > lim_unr:
            klass, reason = I_SEMANTICS, f"{100 * unreadable.mean():.2f}% of the cells do not fit the declared type {f.kind} (> {100 * lim_unr:g}%): semantics not as declared"
            timing_status = "NOT_ASSESSED"
        elif f.timing == "forbidden":
            klass, reason, timing_status = I_TIMING, "declared timing concern that cannot be checked", "FORBIDDEN_BY_DECLARATION"
        else:
            if f.timing == "record_date" and col in v21_dates:
                dc, d, dbad = v21_dates[col]
                off = (d.dt.normalize() - idx).dt.days
                rec = present & np.isfinite(num) & (num != 0)
                unknown = (off > 0).fillna(False).to_numpy() | (rec & d.isna().to_numpy()) | dbad
                tbasis = f"record_date {dc}"
            elif f.timing == "days_value":
                unknown = np.isfinite(x) & (x < 0)
                tbasis = "days_value (negative = post-index record)"
            elif f.timing == "record_date":
                tbasis = "record_date declared but no date column found -> attested"
            if tbasis.startswith("record_date ") or f.timing == "days_value":
                inf = int((np.isfinite(x) & (x != 0) & ~unknown).sum())
                nu = int(unknown.sum())
                sp, si = nu / max(1, n), nu / max(1, nu + inf)
                if sp > float(gates["max_unknown_share_population"]) or si > float(gates["max_unknown_share_informative"]):
                    klass = I_TIMING
                    reason = f"post-index / undated records on {nu} rows ({100 * sp:.2f}% of the cohort, {100 * si:.1f}% of informative rows): Phase 3 gates G2 / G3"
                    timing_status = "POST_INDEX_RECORDS"
                elif nu:
                    klass, reason, timing_status = SAFE_BOUNDED, f"{nu} rows with a post-index / undated record take the no-record state (within the Phase 3 gates)", "ROW_VERIFIED_BOUNDED"
                else:
                    klass, reason, timing_status = SAFE_VERIFIED, "every row verified on or before the index day", "ROW_VERIFIED"
            else:
                if v3_passed and f.domain not in post_domains:
                    klass, reason, timing_status = SAFE_ATTESTED, "no row-level date: eligible on the DWH attestation, re-checked on 2026 by V3", "ATTESTED_V3_PASSED"
                else:
                    klass, reason, timing_status = I_TIMING, ("no row-level date and the attestation is withdrawn: " + ("V3 found post-index records"
                                                               if not v3_passed else f"post-index V21 record dates in its domain {f.domain}")), "ATTESTATION_WITHDRAWN"
            if f.domain == "NEW_MEDICATION_SAFE" and purchase_after:
                klass, reason, timing_status = I_TIMING, f"medication purchase dates after Index_Date on {purchase_after} rows (brief concern 1-2)", "PURCHASE_AFTER_INDEX"
        x = np.where(unknown | unreadable, _no_record_state(f.kind, f.missing), x)
        vals[name] = x
        dom_meta["timing_basis"] = tbasis
        reg.append({"feature": name, "raw_columns": col, "origin": "NEW_V21", "domain": f.domain, "kind": f.kind, "class": klass, "reason": reason,
                    "availability_risk": risk, "availability_assumption": assumption, "n_unknown_cells": int((unknown | unreadable).sum()), "phase3_class": "",
                    "provenance": f"V21 catalogue entry {f.column} ({how}); timing {tbasis}", "_meta": dom_meta})
        cat_rows.append({"raw_column": col, "engineered_feature": name, "domain": f.domain, "old_or_new": "NEW_V21", "semantic_status":
                         ("DECLARED_TYPE_MISMATCH" if klass == I_SEMANTICS else f"DECLARED ({how})"), "timing_status": timing_status or tbasis,
                         "availability_status": f"{risk}: {assumption}", "missingness": "", "prevalence_or_distribution": "", "eligibility": klass,
                         "exclusion_reason": "" if klass in (SAFE_VERIFIED, SAFE_BOUNDED, SAFE_ATTESTED) else reason,
                         "provenance": f"2026 column outside the 2025 contract; matched to the V21 catalogue ({how})", "notes": f.text})
    # ---- every other 2026 column outside the Phase 3 universe (catalogued, never a predictor)
    und = []
    for c in header:
        if c in matches or c in universe_cols or c in ID_COLS or c in QA_COLS or c in date_cols_used:
            continue
        in25 = c in set(contract.names)
        if c in sealed:
            cat_rows.append({"raw_column": c, "engineered_feature": "", "domain": "SEALED_OUTCOME_OR_FUTURE", "old_or_new": "2025_CONTRACT" if in25 else "NEW_2026",
                             "semantic_status": sealed[c], "timing_status": "SEALED", "availability_status": "", "missingness": "", "prevalence_or_distribution": "",
                             "eligibility": I_LEAKAGE, "exclusion_reason": f"outcome / follow-up / future column ({sealed[c]}): never read into X",
                             "provenance": "x_sealing", "notes": ""})
            continue
        if in25:
            continue          # a 2025 contract column outside the Phase 3 universe: documented in the Phase 2/3 dispositions (not a V21 addition)
        s = fe[c].astype("string") if c in fe.columns else pd.Series(pd.NA, index=fe.index, dtype="string")
        num = pd.to_numeric(s.str.strip(), errors="coerce")
        nn = int(s.notna().sum())
        small = 0 < nn < 10 or 0 < n - nn < 10
        prof = {"missing_pct": (round(100.0 * float(s.isna().mean()), 2) if not small else "suppressed") if n else None,
                "numeric_share": round(float(num.notna().sum() / max(1, nn)), 3) if not small else "suppressed",
                "n_distinct": int(s.nunique(dropna=True)) if not small else "suppressed"}
        amb = next((k for k, v in ambiguous.items() if c in v), "")
        why = (f"ambiguous match for the V21 entry {amb} ({ambiguous[amb]})" if amb else "not in the pre-declared V21 catalogue: semantics not verified")
        und.append({"column": c, "suggested_domain": _suggest_domain(c), **prof, "reason": why})
        cat_rows.append({"raw_column": c, "engineered_feature": "", "domain": _suggest_domain(c), "old_or_new": "NEW_2026", "semantic_status": "UNDECLARED",
                         "timing_status": "NOT_ASSESSED", "availability_status": "UNKNOWN", "missingness": f"{prof['missing_pct']}%",
                         "prevalence_or_distribution": "suppressed" if small else (f"{prof['n_distinct']} distinct values" if prof["n_distinct"] >= 10
                                                                                   else "fewer than 10 distinct values"), "eligibility": I_SEMANTICS,
                         "exclusion_reason": why, "provenance": "2026 column outside the 2025 contract", "notes": "never a predictor in this run"})
    for fcol, c25 in v21_old.items():
        cat_rows.append({"raw_column": c25, "engineered_feature": "", "domain": "OLD", "old_or_new": "OLD_2025_CONTRACT", "semantic_status": "2025 contract column",
                         "timing_status": "Phase 3 rules", "availability_status": "", "missingness": "", "prevalence_or_distribution": "",
                         "eligibility": "SEE_FEATURE_ELIGIBILITY" if c25 in universe_cols else "OUTSIDE_PHASE3_UNIVERSE",
                         "exclusion_reason": "" if c25 in universe_cols else "2025 column represented by another Phase 3 feature or not a predictor role",
                         "provenance": f"V21 catalogue entry {fcol} is a 2025 column: OLD, not new", "notes": "the brief listed it as new; it belongs to the Phase 3 universe"})
    return vals, reg, pd.DataFrame(cat_rows), pd.DataFrame(und, columns=["column", "suggested_domain", "missing_pct", "numeric_share", "n_distinct", "reason"])


def _suggest_domain(c: str) -> str:
    s = c.lower()
    for key, dom in (("mefi", "NEW_FRAILTY_MEFI"), ("frail", "NEW_FRAILTY_MEFI"), ("registry", "NEW_REGISTRY"), ("dx", "NEW_DIAGNOSIS"),
                     ("diag", "NEW_DIAGNOSIS"), ("purchas", "NEW_MEDICATION_SAFE"), ("drug", "NEW_MEDICATION_SAFE"), ("med", "NEW_MEDICATION_SAFE"),
                     ("visit", "NEW_UTILISATION_SAFE"), ("admis", "NEW_UTILISATION_SAFE"), ("cogn", "NEW_COGNITION"), ("gait", "NEW_FUNCTION_MOBILITY"),
                     ("mobil", "NEW_FUNCTION_MOBILITY"), ("adl", "NEW_FUNCTION_MOBILITY")):
        if key in s:
            return dom
    return "UNASSIGNED"


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
    q = np.percentile(obs, [25, 50, 75])
    return f"median {q[1]:g} (IQR {q[0]:g}-{q[2]:g})"


def _outcome_extra(o: pd.DataFrame, oc: dict[str, Any], idx: pd.Timestamp) -> dict[str, Any]:
    """Aggregate audit items of brief §E not covered by O1-O7: events on/before the index day among all labels, beyond the nominal horizon,
    class balance."""
    y = pd.to_numeric(o[oc["label_column"]], errors="coerce")
    ev = o[oc["event_date_column"]].dt.normalize() if oc["event_date_column"] in o else pd.Series(pd.NaT, index=o.index)
    dte = (ev - idx).dt.days
    n = int(y.isin([0, 1]).sum())
    return {"labelled_rows": n, "events": int((y == 1).sum()), "non_events": int((y == 0).sum()),
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
