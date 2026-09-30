"""Temporal / leakage EDA: every date column against Index_Date, every source without a record date, and the provenance chain

    source concept (dictionary source) -> aggregate column (contract) -> canonical feature (mapping) -> model(s) that consume it.

The start-of-index-day rule is the framework's (source_event_date < Index_Date). A record date on the index day or later proves that
the aggregates of that source include information not available at the prediction time on those rows. Nothing is repaired.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from falls_ml.data.meuhedet_timing import DAYS_SINCE_COMPANION
from falls_ml.data.meuhedet_wide import PREDICTOR_ROLES, WideContract, WideMapping
from falls_ml.eda.common import SUPPRESSED, cell, fnum, numeric
from falls_ml.eda.dictionary import DataDictionary, column_usage

RISK_TEXT = {
    "PROVEN_INDEX_DAY_OR_FUTURE_RECORDS": "record dates on/after the index day exist among eligible rows: every aggregate of this source carries "
                                          "information unavailable at the prediction time on those rows (leakage risk)",
    "CLEAN_IN_THIS_EXTRACT": "every observed record date of eligible rows is before the index day",
    "PARTIAL_EVIDENCE_ONLY": "sufficient-only evidence (first registry entry); a later same-day entry is undetectable",
    "TIMING_UNKNOWN": "as-of timing not confirmed (billing lag / retrospective): blocked as a predictor source",
    "POST_INDEX_BY_DESIGN": "post-index by design (outcome / follow-up); never a predictor",
    "UNVERIFIABLE_NO_RECORD_DATE": "the extract carries no record date for this source: its timing is declared, not verifiable",
    "TECHNICAL_DATE": "technical / SCD / window date (not a clinical record date)",
}
OFFSET_BUCKETS = (("before: >365 days", -np.inf, -366), ("before: 91-365", -365, -91), ("before: 31-90", -90, -31), ("before: 8-30", -30, -8),
                  ("before: 1-7", -7, -1), ("index day (0)", 0, 0), ("after: 1-7", 1, 7), ("after: 8-30", 8, 30), ("after: >30", 31, np.inf))


def _models(features: list[str], sets: dict[str, list[str]], safe: list[str]) -> list[str]:
    out = []
    if any(f in sets["strict"] for f in features):
        out.append("STRICT")
    if any(f in sets["extended"] for f in features):
        out.append("EXTENDED")
    if any(f in safe for f in features):
        out.append("SAFE_ALL_ROWS")
    return out


def temporal_audit(frame: pd.DataFrame, contract: WideContract, dictionary: DataDictionary, mapping: WideMapping, *, index: pd.Series,
                   eligible: pd.Series, labelled: pd.Series, safe_features: list[str], min_cell: int) -> pd.DataFrame:
    """``temporal_audit.csv``: one row per date column (all of them, predictor source or not) + one row per source without a record date."""
    sets = mapping.feature_sets()
    usage = column_usage(mapping)
    by_source: dict[str, list[str]] = {}
    for c in contract.columns:
        by_source.setdefault(dictionary.source(c.name), []).append(c.name)
    rows: list[dict[str, Any]] = []
    record_dates = {dictionary.record_date(c.name) for c in contract.columns} - {None}
    for c in contract.columns:
        if not c.is_date or c.name not in frame.columns:
            continue
        s = frame[c.name].dt.normalize()
        sent = s.isin([pd.Timestamp(v) for v in c.sentinels]) if c.sentinels else pd.Series(False, index=s.index)
        obs = s.notna() & index.notna() & ~sent
        off = (s - index).dt.days.where(obs)
        before, on, after = obs & (off < 0), obs & (off == 0), obs & (off > 0)
        src = dictionary.source(c.name)
        derived = [x for x in by_source.get(src, []) if x != c.name and contract.get(x).role in PREDICTOR_ROLES and not contract.get(x).is_date]
        if dictionary.record_date(c.name) != c.name:   # a date column that is not the source's record date (e.g. MMSE date inside NURSE_COGNITION)
            derived = [x for x in derived if dictionary.record_date(x) == c.name]
        feats = sorted({f for x in derived for f in usage.get(x, {}).get("features", [])} | set(usage.get(c.name, {}).get("guard_for", []))
                       | set(usage.get(c.name, {}).get("evidence_for", [])))
        affected_elig = int(((on | after) & eligible).sum())
        if c.role == "LABEL" or c.timing == "post_index":
            risk = "POST_INDEX_BY_DESIGN"
        elif c.timing == "unknown" or c.role == "FORBIDDEN_LEAKAGE":
            risk = "TIMING_UNKNOWN"
        elif c.name not in record_dates and c.role not in PREDICTOR_ROLES:
            risk = "TECHNICAL_DATE"
        elif src == "REGISTRIES":
            risk = "PROVEN_INDEX_DAY_OR_FUTURE_RECORDS" if affected_elig else "PARTIAL_EVIDENCE_ONLY"
        else:
            risk = "PROVEN_INDEX_DAY_OR_FUTURE_RECORDS" if affected_elig else "CLEAN_IN_THIS_EXTRACT"
        q = off.dropna()
        r: dict[str, Any] = {"column": c.name, "kind": "date column", "source": src, "domain": dictionary.domain_label(c.name), "contract_role": c.role,
                             "contract_timing": c.timing, "is_source_record_date": c.name in record_dates,
                             "n_observed": cell(int(obs.sum()), min_cell), "n_declared_sentinel": cell(int(sent.sum()), min_cell),
                             "n_before_index": cell(int(before.sum()), min_cell), "n_equal_index": cell(int(on.sum()), min_cell),
                             "n_after_index": cell(int(after.sum()), min_cell),
                             "pct_equal_index": fnum(100.0 * on.sum() / obs.sum(), 3) if obs.sum() and cell(int(on.sum()), min_cell) != SUPPRESSED else (SUPPRESSED if obs.sum() else None),
                             "pct_after_index": fnum(100.0 * after.sum() / obs.sum(), 3) if obs.sum() and cell(int(after.sum()), min_cell) != SUPPRESSED else (SUPPRESSED if obs.sum() else None),
                             "eligible_on_or_after_index": cell(affected_elig, min_cell),
                             "labelled_equal_index": cell(int((on & labelled).sum()), min_cell), "labelled_after_index": cell(int((after & labelled).sum()), min_cell),
                             "offset_days_p1": fnum(q.quantile(0.01), 1) if len(q) else None, "offset_days_p5": fnum(q.quantile(0.05), 1) if len(q) else None,
                             "offset_days_median": fnum(q.median(), 1) if len(q) else None, "offset_days_p95": fnum(q.quantile(0.95), 1) if len(q) else None,
                             "offset_days_p99": fnum(q.quantile(0.99), 1) if len(q) else None}
        for label, lo, hi in OFFSET_BUCKETS:
            r[f"offset {label}"] = cell(int(((q >= lo) & (q <= hi)).sum()), min_cell)
        comp = DAYS_SINCE_COMPANION.get(c.name)
        if comp and comp in frame.columns:
            ds = numeric(frame[comp])
            both = obs & ds.notna()
            r["days_since_companion"] = comp
            r["companion_consistent_share"] = fnum(float((both & (ds == -off)).sum()) / max(int(both.sum()), 1), 4) if both.any() else None
        r.update({"derived_aggregate_columns": "; ".join(derived), "canonical_features": "; ".join(feats),
                  "models": "; ".join(_models(feats, sets, safe_features)), "leakage_risk": risk, "leakage_risk_text": RISK_TEXT[risk]})
        rows.append(r)
    for src, meta in dictionary.sources.items():
        if meta.get("record_date") is not None:
            continue
        cols = [x for x in by_source.get(src, []) if contract.get(x).role in PREDICTOR_ROLES and not contract.get(x).is_date]
        if not cols:
            continue
        timings = {contract.get(x).timing for x in cols}
        feats = sorted({f for x in cols for f in usage.get(x, {}).get("features", [])})
        risk = "TIMING_UNKNOWN" if "unknown" in timings else "UNVERIFIABLE_NO_RECORD_DATE"
        rows.append({"column": f"(no record date: {src})", "kind": "source without a record date", "source": src, "domain": "",
                     "contract_role": "", "contract_timing": "; ".join(sorted(timings)), "derived_aggregate_columns": "; ".join(cols),
                     "canonical_features": "; ".join(feats), "models": "; ".join(_models(feats, sets, safe_features)),
                     "leakage_risk": risk, "leakage_risk_text": RISK_TEXT[risk] + f" ({meta['text']})"})
    return pd.DataFrame(rows)


def provenance_table(contract: WideContract, dictionary: DataDictionary, mapping: WideMapping, temporal: pd.DataFrame,
                     provenance: dict[str, Any] | None, safe_features: list[str]) -> pd.DataFrame:
    """``feature_provenance.csv``: source concept -> aggregate column -> canonical feature -> model(s), for every predictor-role column
    and every column read by a mapped feature, with the timing status of the source."""
    sets = mapping.feature_sets()
    usage = column_usage(mapping)
    risk_by_date = {r["column"]: r["leakage_risk"] for _, r in temporal.iterrows() if r["kind"] == "date column"}
    risk_by_source = {r["source"]: r["leakage_risk"] for _, r in temporal.iterrows() if r["kind"] != "date column"}
    klass = {e["feature"]: e for e in (provenance or {}).get("features", [])}
    rows = []
    for c in contract.columns:
        u = usage.get(c.name, {})
        if c.role not in PREDICTOR_ROLES and not u.get("features"):
            continue
        if c.is_date and not u.get("features"):
            continue
        rd = dictionary.record_date(c.name)
        src = dictionary.source(c.name)
        status = risk_by_date.get(rd, "") if rd else risk_by_source.get(src, "UNVERIFIABLE_NO_RECORD_DATE")
        if c.timing == "unknown" or c.role == "FORBIDDEN_LEAKAGE":
            status = "TIMING_UNKNOWN"
        feats = u.get("features", [])
        models = _models(feats, sets, safe_features)
        if not models and u.get("phase2_for"):
            models = ["PHASE2_CANDIDATE (not trained)"]
        rows.append({"source_concept": src, "source_text": dictionary.sources[src]["text"], "record_date_column": rd or "(none)",
                     "source_timing_status": status, "aggregate_column": c.name, "domain": dictionary.domain_label(c.name), "contract_role": c.role,
                     "contract_timing": c.timing, "canonical_feature": "; ".join(feats),
                     "mapping_quality": "; ".join(sorted({mapping.get(f).quality for f in feats})), "op": "; ".join(sorted({mapping.get(f).op or '' for f in feats})),
                     "models": "; ".join(models) or "none (not used by any current analysis)",
                     "provenance_class": "; ".join(sorted({klass[f]["provenance_class"] for f in feats if f in klass})),
                     "safe_all_rows": "; ".join(sorted({klass[f]["safe_all_rows"] for f in feats if f in klass})),
                     "efalls_phase2_candidate_for": "; ".join(u.get("phase2_for", []))})
    return pd.DataFrame(rows)
