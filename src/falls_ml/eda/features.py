"""Feature-level EDA tables.

``current_feature_dictionary``  the current EXTENDED predictors (STRICT is a subset), every field derived from the mapping YAML, the
                                contract, the data dictionary, the D-00 evidence of this extract and - when a completed reference run is
                                given - its final LASSO coefficient, bootstrap selection frequency and permutation importance.
``phase2_catalogue``            Meuhedet-native predictor columns outside the eFalls baseline, profiled for Phase 2. They are NOT added
                                to any model here and nothing is trained on them.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from falls_ml.data.meuhedet_wide import PREDICTOR_ROLES, WideContract, WideMapping
from falls_ml.eda.common import SUPPRESSED, fnum, med_iqr, numeric
from falls_ml.eda.dictionary import DataDictionary, column_usage
from falls_ml.eda.missingness import declared_class
from falls_ml.eda.profile import constant_status

NOT_AVAILABLE = "not available (no completed reference run given: --reference)"


def _canonical_distribution(s: pd.Series, model_dtype: str) -> str:
    v = s.dropna()
    if v.empty:
        return "all missing"
    if model_dtype == "int8_binary":
        return f"prevalence {100.0 * float(numeric(v).mean()):.2f}%"
    if model_dtype == "onehot_categorical":
        return "; ".join(f"{k}: {100.0 * p:.1f}%" for k, p in v.astype(str).value_counts(normalize=True).items())
    return med_iqr(numeric(v)) + f" (mean {numeric(v).mean():.2f})"


def current_feature_dictionary(mapping: WideMapping, contract: WideContract, dictionary: DataDictionary, *, cohort_raw: pd.DataFrame,
                               canonical: pd.DataFrame | None, build: Any | None, timing: dict[str, Any], provenance: dict[str, Any] | None,
                               reference_report: pd.DataFrame | None, reference_strict_report: pd.DataFrame | None) -> pd.DataFrame:
    sets = mapping.feature_sets()
    cols = timing.get("columns", {})
    klass = {e["feature"]: e for e in (provenance or {}).get("features", [])}
    ref = reference_report.set_index("canonical_efalls_feature") if reference_report is not None and len(reference_report) else None
    ref_s = reference_strict_report.set_index("canonical_efalls_feature") if reference_strict_report is not None and len(reference_strict_report) else None
    rows = []
    for name in sets["extended"]:
        f = mapping.get(name)
        src = list(f.source_columns)
        if f.record_date_column:
            e = cols.get(f.record_date_column, {})
            n_aff = e.get("n_affected_eligible_labelled", e.get("n_affected_eligible", 0))
            d00 = (f"YES - {f.record_date_column} on/after the index day on {n_aff} eligible labelled rows ({e.get('root_cause')})" if n_aff
                   else f"NO - {f.record_date_column} before the index day on every eligible row")
            temporal = f"event source; last-record date {f.record_date_column}"
        elif f.same_day_evidence_column:
            e = cols.get(f.same_day_evidence_column, {})
            n_aff = e.get("n_affected_eligible_labelled", e.get("n_affected_eligible", 0))
            d00 = (f"PARTIAL EVIDENCE - {f.same_day_evidence_column} on/after the index day on {n_aff} eligible labelled rows" if n_aff
                   else f"NO EVIDENCE - {f.same_day_evidence_column} before the index day on every eligible row (sufficient-only evidence, Q-M-11)")
            temporal = f"state at index (registry); same-day evidence {f.same_day_evidence_column}"
        else:
            d00 = "NOT VERIFIABLE - the extract carries no record date for this source"
            temporal = "state at index; no record date in the extract"
        src_missing = "; ".join(f"{c}: {100.0 * cohort_raw[c].isna().mean():.2f}%" for c in src if c in cohort_raw.columns)
        conv = int((getattr(build, "null_to_zero", {}) or {}).get(name, 0)) if build is not None else None
        r: dict[str, Any] = {
            "canonical_feature": name, "exact_source_columns": "; ".join(src), "op": f.op, "source_domain": "; ".join(sorted({dictionary.domain_label(c) for c in src})),
            "source_system": "; ".join(sorted({dictionary.sources[dictionary.source(c)]["text"] for c in src})),
            "business_meaning": " | ".join(f"{c}: {dictionary.columns[c]['meaning']} [{dictionary.columns[c]['status']}]" for c in src),
            "efalls_concept": (f.efalls or {}).get("concept") or name, "strict": name in sets["strict"], "extended": True,
            "mapping_quality": f.quality, "temporal_source": temporal, "source_missing_pct_modelling_cohort": src_missing,
            "absent_is_zero_conversions": conv if conv is not None else "",
            "distribution_modelling_cohort": _canonical_distribution(canonical[name], f.model_dtype) if canonical is not None and name in canonical.columns else "",
            "d00_affected": d00, "provenance_class": klass.get(name, {}).get("provenance_class", ""),
            "safe_all_rows": klass.get(name, {}).get("safe_all_rows", ""),
            "known_mapping_limitation": "; ".join(x for x in (f.notes, f.transformation) if x),
        }
        for tag, table in (("extended", ref), ("strict", ref_s)):
            if tag == "strict" and name not in sets["strict"]:
                continue
            if table is None or name not in table.index:
                r[f"lasso_coefficient_{tag}"] = NOT_AVAILABLE if table is None else "n/a"
                continue
            t = table.loc[name]
            r[f"lasso_coefficient_{tag}"] = t.get("lasso_coefficient")
            r[f"odds_ratio_{tag}"] = t.get("odds_ratio")
            r[f"non_zero_{tag}"] = t.get("non_zero_coefficient")
            r[f"bootstrap_selection_frequency_{tag}"] = t.get("selection_frequency")
            r[f"permutation_importance_{tag}"] = t.get("permutation_importance")
        rows.append(r)
    return pd.DataFrame(rows)


def phase2_catalogue(mapping: WideMapping, contract: WideContract, dictionary: DataDictionary, *, cohort_raw: pd.DataFrame, temporal: pd.DataFrame,
                     univariate: pd.DataFrame | None, spearman: pd.DataFrame | None, clusters: pd.DataFrame | None,
                     availability: dict[str, str], missing_classes: pd.DataFrame) -> pd.DataFrame:
    """Predictor-role, non-date columns not read by the EXTENDED set: profile + readiness category. Not trained."""
    usage = column_usage(mapping)
    sets = mapping.feature_sets()
    ext_sources = {c for f in sets["extended"] for c in mapping.get(f).source_columns}
    risk_by_date = {r["column"]: r["leakage_risk"] for _, r in temporal.iterrows() if r["kind"] == "date column"}
    risk_by_source = {r["source"]: r["leakage_risk"] for _, r in temporal.iterrows() if r["kind"] != "date column"}
    uni = univariate.set_index("column") if univariate is not None and len(univariate) else None
    cluster_of: dict[str, str] = {}
    if clusters is not None and len(clusters):
        for _, r in clusters.iterrows():
            for m in str(r["columns"]).split("; "):
                cluster_of[m] = r["cluster"]
    miss = missing_classes.set_index("column") if len(missing_classes) else None
    rows = []
    for c in contract.columns:
        if c.role not in PREDICTOR_ROLES or c.is_date or c.name in ext_sources or c.name not in cohort_raw.columns:
            continue
        s = cohort_raw[c.name]
        status, _share = constant_status(s)
        d = dictionary.columns[c.name]
        rd = dictionary.record_date(c.name)
        src_status = risk_by_date.get(rd, "") if rd else risk_by_source.get(d["source"], "UNVERIFIABLE_NO_RECORD_DATE")
        miss_pct = 100.0 * float(s.isna().mean()) if len(s) else None
        klass = declared_class(contract, c.name)
        if c.semantic == "binary":
            dist = f"prevalence {100.0 * float(numeric(s).mean()):.2f}% of observed" if s.notna().any() else "all missing"
        elif c.semantic in ("categorical", "ordinal"):
            dist = "; ".join(f"{k}: {100.0 * p:.1f}%" for k, p in s.astype("string").fillna("NULL").value_counts(normalize=True).head(6).items())
        else:
            dist = med_iqr(numeric(s))
        best, best_r = "", None
        if spearman is not None and c.name in spearman.columns:
            cand = [x for x in ext_sources if x in spearman.columns]
            if cand:
                vals = spearman.loc[c.name, cand].abs()
                if vals.notna().any():
                    best, best_r = str(vals.idxmax()), float(vals.max())
        if c.timing == "unknown":
            ready = "BLOCKED - timing unknown (billing lag / retrospective); needs DWH confirmation of as-of semantics"
        elif status in ("EMPTY", "CONSTANT"):
            ready = f"NOT USABLE - {status.lower()} in this extract"
        elif d["status"] == "UNKNOWN":
            ready = "NEEDS SME - meaning / codes unknown"
        elif src_status == "PROVEN_INDEX_DAY_OR_FUTURE_RECORDS":
            ready = f"CANDIDATE AFTER DWH FIX - its source's record date ({rd}) is on/after the index day on some rows (D-00)"
        elif klass == "NOT_MEASURED" and (miss_pct or 0) > 25:
            ready = "CANDIDATE - assessment-conditional (NULL = not assessed): needs an explicit not-assessed design, never NULL -> 0"
        elif src_status == "UNVERIFIABLE_NO_RECORD_DATE":
            ready = "CANDIDATE - timing declared by the contract, not verifiable from the extract"
        elif status == "NEAR_CONSTANT":
            ready = "CANDIDATE (SPARSE) - near-constant; unstable estimates"
        else:
            ready = "CANDIDATE"
        u = uni.loc[c.name] if uni is not None and c.name in uni.index else None
        rows.append({"column": c.name, "domain": dictionary.domain_label(c.name), "source": d["source"], "phase2_group": c.group or "",
                     "efalls_concept_candidate": "; ".join(usage.get(c.name, {}).get("phase2_for", [])), "meaning": d["meaning"],
                     "meaning_status": d["status"], "contract_timing": c.timing, "record_date_column": rd or "(none)", "source_timing_status": src_status,
                     "available_at_prediction_time": availability.get(c.name, ""), "missing_pct_modelling_cohort": fnum(miss_pct, 2),
                     "declared_null_class": klass, "dominant_missing_class": miss.at[c.name, "dominant_class"] if miss is not None and c.name in miss.index else "",
                     "distribution_modelling_cohort": dist, "constant_status": status,
                     "train_smd_fall_vs_no_fall": None if u is None else u.get("smd_fall_vs_no_fall"),
                     "train_univariate_auroc": None if u is None else u.get("univariate_auroc"),
                     "most_correlated_extended_source": best, "abs_spearman_with_it": fnum(best_r, 3), "redundancy_cluster": cluster_of.get(c.name, ""),
                     "readiness": ready, "trained": "NO - catalogue only (Phase 2)", "contract_note": c.note})
    out = pd.DataFrame(rows)
    return out.sort_values(["readiness", "domain", "column"]).reset_index(drop=True) if len(out) else out
