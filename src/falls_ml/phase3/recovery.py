"""Feature eligibility under the corrected Phase 3 time contract (planning/PHASE3_TIME_CONTRACT_AUDIT.md, configs/meuhedet/phase3_time_contract.yaml).

Prediction at the END of Index_Date: a predictor may use records dated ON OR BEFORE Index_Date (``source_event_date <= Index_Date``); the outcome
starts the next day. Evidence is the Phase 1-2 D-00 evidence (contract roles / timing, data-dictionary sources and record dates, d00 evidence kinds),
read from record dates and values only - never an outcome.

Per row and feature:
    KNOWN    the value was available at prediction time: a documented static attribute; or the source's record date is ON OR BEFORE Index_Date;
             or the source has no record and no value was recorded
    UNKNOWN  a record of the source is dated AFTER Index_Date (genuine future information, if the extract holds any), a value was recorded
             without its record date, or a cell could not be read (e.g. '00:00.0')
Per feature (classes):
    SAFE_VERIFIED           every input has a row-level record date (or is a documented static attribute) and every analysed row is KNOWN
    SAFE_VERIFIED_BOUNDED   as above with a few UNKNOWN rows (within the gates): those rows stay in the population and are evaluated with bounds
    SAFE_ATTESTED           an input has no row-level date (medications, prescriptions, laboratories, registries, comorbidity / social status,
                            undocumented static meanings, values the date does not bound): eligible because the DWH states that no event after
                            Index_Date enters the extract AND check V3 found no post-index record in any dated source; withdrawn otherwise
    NOT_RECOVERABLE_FUTURE_RECORDS  too many rows carry post-index records (gates G2 / G3)
    UNRESOLVED              timing cannot be supported (unknown contract timing, a documented post-index date source, or the attestation withdrawn)
    NOT_RECOVERABLE_FORBIDDEN       post-index / label / forbidden information: never a predictor
    INELIGIBLE_DATA         fewer than the minimum KNOWN observed TRAIN rows, or constant
Nothing is repaired: an UNKNOWN value is never imputed or re-coded, and its patient is never removed from the analysis population.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from falls_ml.data.meuhedet_wide import PREDICTOR_ROLES, feature_input_columns
from falls_ml.phase3.config import ROW_RECOVERABLE_KIND, RecoveryRules

# column-level evidence classes
ROW = "ROW_LEVEL"
STATIC = "STATIC_DOCUMENTED"
ATTESTED = "ATTESTED"
UNRES = "UNRESOLVED_TIMING"
EXCLUDED = "EXCLUDED_FORBIDDEN"
SEVERITY = {ROW: 0, STATIC: 0, ATTESTED: 1, UNRES: 2, EXCLUDED: 3}

# feature classes
SAFE_VERIFIED = "SAFE_VERIFIED"
SAFE_BOUNDED = "SAFE_VERIFIED_BOUNDED"
SAFE_ATTESTED = "SAFE_ATTESTED"
NR_FUTURE = "NOT_RECOVERABLE_FUTURE_RECORDS"
UNRESOLVED = "UNRESOLVED"
NR_FORBIDDEN = "NOT_RECOVERABLE_FORBIDDEN"
NR_DATA = "INELIGIBLE_DATA"
ELIGIBLE = (SAFE_VERIFIED, SAFE_BOUNDED, SAFE_ATTESTED)      # the primary (full) standard
VERIFIED = (SAFE_VERIFIED, SAFE_BOUNDED)                      # the verified-only sensitivity standard
# backward-compatible names used by the stages
RECOVERED = ELIGIBLE
RECOVERED_EXACT, RECOVERED_BOUNDED, NR_NO_ROW_DATE = SAFE_VERIFIED, SAFE_BOUNDED, UNRESOLVED
CLASS_TEXT = {
    SAFE_VERIFIED: "row-level record dates (or documented static attribute): available at prediction time on every analysed row",
    SAFE_BOUNDED: "row-level record dates; a few rows carry a post-index record: they stay in the population and are evaluated with bounds",
    SAFE_ATTESTED: "no row-level record date: eligible on the DWH statement 'no event after Index_Date' confirmed by check V3 (not row-verified)",
    NR_FUTURE: "too many rows with post-index records (gates G2 / G3): DWH correction needed",
    UNRESOLVED: "timing cannot be supported (unknown contract timing, a post-index date source, or the attestation withdrawn): exploratory only",
    NR_FORBIDDEN: "post-index / label / forbidden information: never a predictor",
    NR_DATA: "fewer than the minimum KNOWN observed TRAIN rows, or constant: not analysable",
}
INCREASING_OPS = ("copy", "any_positive", "flag_ge")
UNREADABLE_PREFIX = "__unreadable__"


def _recorded(s: pd.Series) -> np.ndarray:
    """A recorded value: a non-zero number or any non-empty non-numeric value (the Phase 2 D-00 definition)."""
    if pd.api.types.is_numeric_dtype(s):
        return (s.notna() & (s.fillna(0) != 0)).to_numpy(dtype=bool)
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.notna().to_numpy(dtype=bool)
    return (s.notna() & (s.astype("string").fillna("").str.strip() != "")).to_numpy(dtype=bool)


@dataclass
class ColumnEvidence:
    column: str
    source: str
    evidence_kind: str
    klass: str
    reason: str
    record_date_column: str | None = None
    unknown: np.ndarray | None = None
    record_present: np.ndarray | None = None
    n_on_index: int = 0
    n_after_index: int = 0
    n_recorded_without_date: int = 0


class RowRecovery:
    """Memoised column-level evidence on the analysis rows (TRAIN + VALIDATION of the Phase 3 population)."""

    def __init__(self, frame: pd.DataFrame, *, contract: Any, dictionary: Any, d00: Any, rules: RecoveryRules, tc: Any, attestation_holds: bool):
        self.frame, self.contract, self.dictionary, self.d00, self.rules, self.tc = frame, contract, dictionary, d00, rules, tc
        self.attest = bool(attestation_holds)
        self.idx = frame["Index_Date"].dt.normalize()
        self._memo: dict[str, ColumnEvidence] = {}

    def _attested(self, base: dict[str, Any], why: str) -> ColumnEvidence:
        if self.attest:
            return ColumnEvidence(**base, klass=ATTESTED, reason=f"{why}; eligible on the DWH attestation (check V3 passed)")
        return ColumnEvidence(**base, klass=UNRES, reason=f"{why}; the attestation is WITHDRAWN (check V3 found post-index records)")

    def column(self, col: str) -> ColumnEvidence:
        if col in self._memo:
            return self._memo[col]
        c = self.contract.get(col)
        d = self.dictionary.columns[col]
        src = d["source"]
        kind = self.d00.source_evidence[src]
        date_col = self.dictionary.record_date(col)
        base = {"column": col, "source": src, "evidence_kind": kind, "record_date_column": date_col}
        if c.role in ("FORBIDDEN_LEAKAGE", "LABEL", "IDENTIFIER") or c.timing == "post_index" or kind == "POST_INDEX":
            ev = ColumnEvidence(**base, klass=EXCLUDED, reason=f"contract role {c.role} / timing {c.timing} / source {src}: never a predictor")
        elif col in self.tc.documented_as_of:
            ev = self._attested(base, f"documented as-of Index_Date: {self.tc.documented_as_of[col]}")
        elif src in self.tc.excluded_date_sources:
            ev = ColumnEvidence(**base, klass=UNRES, reason=f"source {src}: {self.tc.excluded_date_sources[src]}")
        elif c.timing == "unknown":
            ev = ColumnEvidence(**base, klass=UNRES, reason="contract timing unknown (not confirmed as available at prediction time)")
        elif kind == "STATIC_ATTRIBUTE":
            ev = (ColumnEvidence(**base, klass=STATIC, reason=f"static attribute of {src}, meaning DOCUMENTED") if d["status"] == "DOCUMENTED" else
                  self._attested(base, f"static source {src} (availability not in doubt) but the meaning is {d['status']}"))
        elif kind != ROW_RECOVERABLE_KIND:
            ev = (self._attested(base, f"source {src}: evidence kind {kind} (no row-level record date)") if kind in ("NO_RECORD_DATE", "SUFFICIENT_ONLY_RECORD_DATE")
                  else ColumnEvidence(**base, klass=UNRES, reason=f"source {src}: evidence kind {kind}"))
        elif not date_col or date_col not in self.frame.columns or not pd.api.types.is_datetime64_any_dtype(self.frame[date_col]):
            ev = self._attested(base, f"record-date column {date_col!r} of {src} is not in the extract")
        elif self.dictionary.columns.get(date_col, {}).get("status") != "DOCUMENTED":
            ev = self._attested(base, f"the meaning of the record date {date_col} is not DOCUMENTED (no row-level bound)")
        elif col in self.rules.not_bounded:
            ev = self._attested(base, f"not bounded by {date_col}: {self.rules.not_bounded[col]}")
        else:
            off = (self.frame[date_col].dt.normalize() - self.idx).dt.days
            after = (off > 0).to_numpy(dtype=bool)
            present = self.frame[date_col].notna().to_numpy(dtype=bool)
            unknown = after.copy()
            n_wo = 0
            if col != date_col and c.role in PREDICTOR_ROLES and col in self.frame.columns:
                rec = _recorded(self.frame[col])
                if col in self.rules.absence_indicators:
                    rec &= ~(pd.to_numeric(self.frame[col], errors="coerce") == self.rules.absence_indicators[col]).to_numpy(dtype=bool)
                wo = ~present & rec
                n_wo = int(wo.sum())
                unknown |= wo
            n_unr = 0
            for c_ in {col, date_col}:          # a cell the validated reader could not read (e.g. '00:00.0') is never taken as absent
                um = self.frame.get(UNREADABLE_PREFIX + c_)
                if um is not None:
                    unr = um.to_numpy(dtype=bool)
                    n_unr += int((unr & ~unknown).sum())
                    unknown |= unr
            ev = ColumnEvidence(**base, klass=ROW, unknown=unknown, record_present=present,
                                reason=f"row level: {date_col} (DOCUMENTED record date of {src}) <= Index_Date -> KNOWN; after it -> UNKNOWN"
                                       + (f"; {n_wo} rows carry a value without a record date -> UNKNOWN" if n_wo else "")
                                       + (f"; {n_unr} rows with an unreadable cell -> UNKNOWN" if n_unr else ""),
                                n_on_index=int((off == 0).sum()), n_after_index=int((off > 0).sum()), n_recorded_without_date=n_wo)
        self._memo[col] = ev
        return ev

    def feature(self, name: str, value_inputs: tuple[str, ...], other_inputs: tuple[str, ...] = (), *, op: str = "copy",
                observed: pd.Series | None = None) -> dict[str, Any]:
        evs = [self.column(c) for c in (*value_inputs, *other_inputs)]
        worst = max(evs, key=lambda e: SEVERITY[e.klass]).klass if evs else STATIC
        out: dict[str, Any] = {"feature": name, "sources": sorted({e.source for e in evs}), "record_dates": sorted({e.record_date_column for e in evs if e.record_date_column}),
                               "evidence_kinds": sorted({e.evidence_kind for e in evs}), "reasons": [f"{e.column}: {e.reason}" for e in evs], "klass": worst,
                               "unknown": None, "record_present": None, "monotone_upper_bound": False,
                               "n_on_index": sum(e.n_on_index for e in evs), "n_after_index": sum(e.n_after_index for e in evs),
                               "n_recorded_without_date": sum(e.n_recorded_without_date for e in evs)}
        if worst in (EXCLUDED, UNRES):
            return out
        rows = [e for e in evs if e.klass == ROW]
        if rows:
            n = len(self.frame)
            unk, pres = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
            for e in rows:
                unk |= e.unknown
                pres |= e.record_present
            mono = bool(value_inputs) and all(c in self.rules.upper_bound_columns for c in value_inputs) and op in INCREASING_OPS
            if mono and observed is not None:
                obs = pd.to_numeric(observed, errors="coerce").to_numpy(dtype=float)
                unk &= ~(obs == 0.0)
                out["monotone_upper_bound"] = True
            out.update({"unknown": unk, "record_present": pres | unk})
        return out


def root_cause(n_on: int, n_after: int) -> str:
    if n_after == 0:
        return "CLEAN_UNDER_CORRECTED_CONTRACT" if n_on else "CLEAN"
    return "POST_INDEX_RECORDS"


ROOT_CAUSE_TEXT = {
    "CLEAN": "no record on or after the index day",
    "CLEAN_UNDER_CORRECTED_CONTRACT": "records dated ON the index day only: history at the end-of-day prediction time (they were the Phase 1-2 D-00 "
                                      "flags under the start-of-day assumption)",
    "POST_INDEX_RECORDS": "records dated AFTER the index day: genuine future information for those rows (contradicts the DWH statement)",
}


def source_timing(frame: pd.DataFrame, *, dictionary: Any, d00: Any, rules: RecoveryRules, partition: pd.Series, bridge: pd.Series) -> pd.DataFrame:
    """Per data-dictionary source with a record date: rows before / on / after the index day, NULL dates, offsets after it, the root cause and the
    historical semantics (TRAIN + VALIDATION; aggregate counts only)."""
    idx = frame["Index_Date"].dt.normalize()
    rows = []
    for src, meta in dictionary.sources.items():
        dc = meta.get("record_date")
        sem = rules.source_semantics.get(src, {})
        r: dict[str, Any] = {"source": src, "source_text": meta.get("text"), "evidence_kind": d00.source_evidence.get(src), "record_date_column": dc or "",
                             "historical_semantics": sem.get("semantics", ""), "semantics_text": sem.get("text", ""), "n_rows": int(len(frame))}
        if dc and dc in frame.columns and pd.api.types.is_datetime64_any_dtype(frame[dc]):
            off = (frame[dc].dt.normalize() - idx).dt.days
            on, after = off == 0, off > 0
            r.update({"n_with_date": int(off.notna().sum()), "n_before_index": int((off < 0).sum()), "n_on_index": int(on.sum()), "n_after_index": int(after.sum()),
                      "n_null_date": int(off.isna().sum()), "max_days_after_index": int(off[after].max()) if after.any() else 0,
                      "root_cause": root_cause(int(on.sum()), int(after.sum())),
                      "n_on_index_in_d00_clean": int((on.to_numpy() & bridge.to_numpy(dtype=bool)).sum()),
                      "n_on_index_outside_d00_clean": int((on.to_numpy() & ~bridge.to_numpy(dtype=bool)).sum())})
            for part in ("train", "validation"):
                r[f"n_after_index_{part}"] = int((after.to_numpy() & (partition == part).to_numpy()).sum())
            r["root_cause_text"] = ROOT_CAUSE_TEXT[r["root_cause"]]
        else:
            r.update({"root_cause": "NO_RECORD_DATE" if d00.source_evidence.get(src) != "POST_INDEX" else "POST_INDEX",
                      "root_cause_text": "no row-level record date in the extract"})
        rows.append(r)
    return pd.DataFrame(rows)


@dataclass
class RecoveryResult:
    registry: pd.DataFrame
    unknown: pd.DataFrame
    upper_bound: pd.DataFrame
    facts: dict[str, Any] = field(default_factory=dict)


def _gate(n_unknown: int, n_known_obs_train: int, n_known_informative: int, n_rows: int, n_unique: int, klass: str, gates: dict[str, float],
          min_obs: int) -> tuple[str, str]:
    if klass == EXCLUDED:
        return NR_FORBIDDEN, "post-index / forbidden input"
    if klass == UNRES:
        return UNRESOLVED, "timing of an input cannot be supported"
    need = max(int(gates["min_known_informative_train"]), int(min_obs))
    if n_known_obs_train < need or n_unique <= 1:
        return NR_DATA, f"{n_known_obs_train} KNOWN observed TRAIN rows (< {need}) or constant"
    if n_unknown:
        sp = n_unknown / n_rows
        si = n_unknown / max(1, n_unknown + n_known_informative)
        if sp > float(gates["max_unknown_share_population"]):
            return NR_FUTURE, f"G2: {n_unknown} rows with post-index / unbounded records = {100 * sp:.2f}% of the population"
        if si > float(gates["max_unknown_share_informative"]):
            return NR_FUTURE, f"G3: {100 * si:.2f}% of the informative rows carry post-index / unbounded records"
    if klass == ATTESTED:
        return SAFE_ATTESTED, "no row-level date for an input; DWH attestation confirmed by V3" + (f"; {n_unknown} UNKNOWN rows bounded" if n_unknown else "")
    if n_unknown:
        return SAFE_BOUNDED, f"{n_unknown} UNKNOWN rows ({100 * n_unknown / n_rows:.3f}% of the population): bounded"
    return SAFE_VERIFIED, "every analysed row KNOWN"


def build_recovery(frame: pd.DataFrame, values: pd.DataFrame, *, catalogue: Any, mapping: Any, baseline: list[str], contract: Any, dictionary: Any,
                   d00: Any, rules: RecoveryRules, train_mask: np.ndarray, min_observed_train: int, tc: Any, attestation_holds: bool,
                   phase2_eligibility: dict[str, str] | None = None) -> RecoveryResult:
    """Classify every catalogue feature and every BASELINE_15 predictor. Labels are never read."""
    rr = RowRecovery(frame, contract=contract, dictionary=dictionary, d00=d00, rules=rules, tc=tc, attestation_holds=attestation_holds)
    n = len(frame)
    rows, unknown, ub = [], {}, {}
    items: list[tuple[str, str, tuple[str, ...], tuple[str, ...], str, pd.Series, dict[str, Any]]] = []
    for f in catalogue.features:
        items.append((f.name, f.domain, f.inputs, (), f.op, values[f.name], {"kind": "catalogue", "mapping_class": f.mapping_class, "form": f.form or "",
                                                                             "form_indicator": f.form_indicator, "text": f.text}))
    for f in mapping.features:
        if f.canonical in baseline:
            inp = feature_input_columns(f)
            s = frame[f.canonical]
            s = (s.astype(str) == "female").astype(float) if f.canonical == "sex" else pd.to_numeric(s, errors="coerce")
            items.append((f.canonical, "BASELINE_15", tuple(inp["value"]), tuple(inp["validation"]), "any_positive" if f.op == "any_positive" else "copy",
                          s, {"kind": "baseline", "mapping_class": f"EFALLS_{f.quality}", "form": "", "form_indicator": False, "text": f.canonical}))
    from falls_ml.phase3.timecontract import availability_of, worst_risk

    for name, domain, vin, oin, op, obs, meta in items:
        ev = rr.feature(name, vin, oin, op=op, observed=obs)
        avs = [availability_of(tc, c, dictionary.source(c)) for c in vin or oin]
        risk = worst_risk([a["risk"] for a in avs])
        x = pd.to_numeric(obs, errors="coerce").to_numpy(dtype=float)
        unk = ev["unknown"] if ev["unknown"] is not None else np.zeros(n, dtype=bool)
        present = ev["record_present"] if ev["record_present"] is not None else np.ones(n, dtype=bool)
        kobs = train_mask & ~unk & np.isfinite(x)
        n_unique = int(len(np.unique(x[kobs]))) if kobs.any() else 0
        n_inf = int((~unk & present).sum())
        klass, why = _gate(int(unk.sum()), int(kobs.sum()), n_inf, n, n_unique, ev["klass"], rules.gates, min_observed_train)
        if ev["unknown"] is not None:
            unknown[name] = unk
            if ev["monotone_upper_bound"]:
                ub[name] = x
        rows.append({"feature": name, "domain": domain, **meta, "inputs": "; ".join((*vin, *oin)), "sources": "; ".join(ev["sources"]),
                     "record_dates": "; ".join(ev["record_dates"]), "evidence_kinds": "; ".join(ev["evidence_kinds"]), "evidence_class": ev["klass"],
                     "monotone_upper_bound": ev["monotone_upper_bound"], "n_rows": n, "n_unknown": int(unk.sum()), "n_unknown_train": int((unk & train_mask).sum()),
                     "unknown_share_population": float(unk.mean()) if n else 0.0, "n_known_informative": n_inf,
                     "unknown_share_informative": float(unk.sum() / max(1, unk.sum() + n_inf)), "n_known_observed_train": int(kobs.sum()),
                     "n_unique_known_train": n_unique, "n_on_index": ev["n_on_index"], "n_after_index": ev["n_after_index"],
                     "n_recorded_without_date": ev["n_recorded_without_date"], "phase3_class": klass, "phase3_reason": why, "phase3_class_text": CLASS_TEXT.get(klass, ""),
                     "verified": klass in VERIFIED, "availability_risk": risk,
                     "availability_assumption": " | ".join(dict.fromkeys(a["assumption"] for a in avs)),
                     "evidence": " | ".join(ev["reasons"]), "phase2_eligibility": (phase2_eligibility or {}).get(name, "")})
    reg = pd.DataFrame(rows)
    p2 = reg["phase2_eligibility"]
    reg["newly_recovered"] = reg["phase3_class"].isin(ELIGIBLE) & (p2 != "ELIGIBLE") & (p2 != "")
    facts = {"n_features": int(len(reg)), "classes": reg["phase3_class"].value_counts().to_dict(),
             "classes_catalogue": reg.loc[reg["kind"] == "catalogue", "phase3_class"].value_counts().to_dict(),
             "classes_baseline": reg.loc[reg["kind"] == "baseline", "phase3_class"].value_counts().to_dict(),
             "n_newly_recovered": int(reg["newly_recovered"].sum()), "attestation_holds": bool(attestation_holds), "labels_read": False}
    return RecoveryResult(registry=reg, unknown=pd.DataFrame(unknown, index=frame.index), upper_bound=pd.DataFrame(ub, index=frame.index), facts=facts)


def remediation_rows(timing: pd.DataFrame, registry: pd.DataFrame, *, rules: RecoveryRules, priority_domains: list[str],
                     catalogue_domains: dict[str, dict[str, str]]) -> pd.DataFrame:
    """One row per source that blocks a feature, holds post-index records, or is eligible only on the attestation: the DWH action."""
    rows = []
    tim = timing.set_index("source")
    for src in tim.index:
        t = tim.loc[src]
        feats = [r for _, r in registry.iterrows() if src in str(r["sources"]).split("; ")]
        if not feats:
            continue
        blocked = [r for r in feats if r["phase3_class"] in (NR_FUTURE, UNRESOLVED)]
        bounded = [r for r in feats if r["phase3_class"] == SAFE_BOUNDED]
        attested = [r for r in feats if r["phase3_class"] == SAFE_ATTESTED]
        n_after = t.get("n_after_index")
        n_after = int(n_after) if n_after is not None and pd.notna(n_after) else 0
        if not blocked and not bounded and not attested and n_after == 0:
            continue
        kind = str(t["evidence_kind"])
        key = ("POST_INDEX" if kind == "POST_INDEX" else "NOT_BOUNDED" if any("not bounded by" in str(r["evidence"]) for r in feats)
               else ROW_RECOVERABLE_KIND if kind == ROW_RECOVERABLE_KIND else kind)
        doms = sorted({str(r["domain"]) for r in feats})
        rows.append({"source": src, "source_text": t["source_text"], "evidence_kind": kind, "historical_semantics": t.get("historical_semantics", ""),
                     "record_date_column": t.get("record_date_column", ""), "root_cause": t.get("root_cause", ""), "n_on_index": t.get("n_on_index"),
                     "n_after_index": t.get("n_after_index"), "n_features_blocked": len(blocked), "features_blocked": "; ".join(r["feature"] for r in blocked),
                     "n_features_bounded": len(bounded), "features_bounded": "; ".join(r["feature"] for r in bounded),
                     "n_features_attested_only": len(attested), "features_attested_only": "; ".join(r["feature"] for r in attested),
                     "domains": "; ".join(doms), "priority_domain": any(d in priority_domains for d in doms),
                     "domain_labels": "; ".join(catalogue_domains.get(d, {}).get("label", d) for d in doms),
                     "required_fix": rules.remediation.get(key, rules.remediation.get(ROW_RECOVERABLE_KIND, "")),
                     "effect_of_fix": "blocked features become eligible, bounded ones exact, attested ones row-verified" if key != "POST_INDEX" else "none"})
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["priority_domain", "n_features_blocked", "source"], ascending=[False, False, True]).reset_index(drop=True)
    return out
