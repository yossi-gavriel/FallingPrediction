"""S02 column registry (all 221 contract columns) and the provenance status of every column and engineered feature.

Provenance uses the UNCHANGED Phase 1 D-00 rules (``falls_ml.d00.dependency._trace``) evaluated on the TRAIN + VALIDATION rows of the
reference cohort: record dates only, never an outcome value, never the test partition. A feature takes the worst status of its inputs.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from falls_ml.d00.dependency import STATUS_TEXT, CohortMasks, _trace, worst
from falls_ml.phase2.config import Catalogue

COHORT = "PHASE2_TRAIN_VALIDATION"
FORBIDDEN_ROLES = ("IDENTIFIER", "LABEL", "FORBIDDEN_LEAKAGE")
METADATA_ROLES = ("QA_CONTROL", "COHORT_ELIGIBILITY")
MAPPING_QUALITY_CLASS = {"HIGH_CONFIDENCE": "EFALLS_HIGH_CONFIDENCE", "APPROXIMATE": "EFALLS_APPROXIMATE", "EXACT": "EFALLS_HIGH_CONFIDENCE"}


def phase2_cohort(frame: pd.DataFrame, index_date: str) -> CohortMasks:
    idx = frame["Index_Date"].dt.normalize()
    mask = pd.Series(True, index=frame.index)
    return CohortMasks(index_date=index_date, idx=idx, masks={COHORT: mask}, guard_columns=[], counts={COHORT: int(len(frame))})


class Provenance:
    """Memoised D-00 status of contract columns on the Phase 2 rows."""

    def __init__(self, frame: pd.DataFrame, index_date: str, *, contract: Any, dictionary: Any, d00_config: Any):
        self.cohorts = phase2_cohort(frame, index_date)
        self.ctx = {"frame": frame, "cohorts": self.cohorts, "contract": contract, "dictionary": dictionary, "config": d00_config, "src_ev": {}}
        self.memo: dict[tuple[str, str], dict[str, Any]] = {}

    def column(self, col: str) -> dict[str, Any]:
        return _trace(col, COHORT, self.memo, (), **self.ctx)

    def feature(self, inputs: tuple[str, ...]) -> dict[str, Any]:
        per = {c: self.column(c) for c in inputs}
        status = worst([t["status"] for t in per.values()])
        reasons = [f"{c}: {t['status']} - {t['reason']}" for c, t in per.items()]
        dates = sorted({t["record_date_column"] for t in per.values() if t.get("record_date_column")})
        kinds = sorted({t["evidence_kind"] for t in per.values()})
        n_on = sum(int(t.get("n_on_index") or 0) for t in per.values())
        n_after = sum(int(t.get("n_after_index") or 0) for t in per.values())
        return {"status": status, "reasons": reasons, "record_dates": dates, "evidence_kinds": kinds, "n_on_index": n_on, "n_after_index": n_after}


def mapping_usage(mapping: Any) -> dict[str, dict[str, Any]]:
    usage: dict[str, dict[str, Any]] = {}
    for f in mapping.features:
        if f.include_in_baseline:
            for c in f.source_columns:
                u = usage.setdefault(c, {"baseline": [], "quality": [], "phase2_for": []})
                u["baseline"].append(f.canonical)
                u["quality"].append(f.quality)
        for c in getattr(f, "phase2_candidate", ()) or ():
            usage.setdefault(c, {"baseline": [], "quality": [], "phase2_for": []})["phase2_for"].append(f.canonical)
    return usage


def column_registry(profile: pd.DataFrame, prov: Provenance, *, contract: Any, dictionary: Any, mapping: Any, catalogue: Catalogue,
                    min_observed_train: int) -> pd.DataFrame:
    usage = mapping_usage(mapping)
    prof = profile.set_index("column")
    rows = []
    for c in contract.columns:
        d = dictionary.columns[c.name]
        u = usage.get(c.name, {"baseline": [], "quality": [], "phase2_for": []})
        feeds = catalogue.reading(c.name)
        t = prov.column(c.name)
        pr = prof.loc[c.name]
        efalls = sorted({*u["baseline"], *u["phase2_for"], *(catalogue.get(f).efalls_concept for f in feeds if catalogue.get(f).efalls_concept)})
        # mapping class
        if c.role in FORBIDDEN_ROLES or c.timing == "post_index":
            klass = "FORBIDDEN"
        elif c.role in METADATA_ROLES:
            klass = "NON_PREDICTIVE_METADATA"
        elif u["baseline"]:
            klass = MAPPING_QUALITY_CLASS.get(u["quality"][0], "EFALLS_APPROXIMATE")
        elif any(catalogue.get(f).mapping_class.startswith("EFALLS") for f in feeds):
            klass = "EFALLS_APPROXIMATE"
        else:
            klass = "MEUHEDET_NATIVE"
        # disposition (first rule that applies)
        if c.role == "IDENTIFIER":
            disp, why = "INELIGIBLE_FORBIDDEN", "identifier (never a predictor, never printed)"
        elif c.role == "LABEL":
            disp, why = "INELIGIBLE_FORBIDDEN", "outcome / label information"
        elif c.role == "FORBIDDEN_LEAKAGE":
            disp, why = "INELIGIBLE_FORBIDDEN", f"forbidden by the contract: {c.note}"
        elif c.timing == "post_index":
            disp, why = "INELIGIBLE_FORBIDDEN", "known only after the prediction time"
        elif c.role in METADATA_ROLES:
            disp, why = "NON_PREDICTIVE_METADATA", "data-quality / cohort bookkeeping column (not a predictor)"
        elif c.name in catalogue.quarantine:
            disp, why = "NEEDS_SME", catalogue.quarantine[c.name]
        elif c.timing == "unknown":
            disp, why = "INELIGIBLE_FORBIDDEN", "timing unknown: not confirmed as available before the index day"
        elif u["baseline"]:
            disp, why = "IN_BASELINE_15", f"source of BASELINE_15 feature(s) {u['baseline']}; D-00 status {t['status']}"
        elif c.name in catalogue.represented_by:
            disp, why = "REPRESENTED_BY", f"represented by {catalogue.represented_by[c.name]}"
        elif c.is_date:
            disp, why = ("RECORD_DATE", f"record date (timing evidence) feeding {feeds}" if feeds else "record date used only as D-00 timing evidence")
        elif int(pr["n_observed_train"]) < min_observed_train:
            disp, why = "INELIGIBLE_DATA", f"{int(pr['n_observed_train'])} observed TRAIN rows (< {min_observed_train})"
        elif pr["constant_status"] in ("EMPTY", "CONSTANT"):
            disp, why = "INELIGIBLE_DATA", f"{str(pr['constant_status']).lower()} in this extract"
        elif t["status"] == "UNSAFE":
            disp, why = "INELIGIBLE_UNSAFE", f"candidate after DWH fix: {t['reason']}"
        elif t["status"] == "UNRESOLVED":
            disp, why = "ELIGIBLE_EXPLORATORY", t["reason"]
        else:
            disp, why = "ELIGIBLE", t["reason"]
        rows.append({
            "raw_column": c.name, "domain": dictionary.domain_label(c.name), "source_system": d["source"], "source_text": dictionary.sources[d["source"]]["text"],
            "data_type": c.sql, "semantic_type": c.semantic, "contract_role": c.role, "business_meaning": d["meaning"], "meaning_status": d["status"],
            "missing_pct": float(pr["missing_pct"]), "unique_count": int(pr["unique_count"]), "constant_status": pr["constant_status"],
            "near_constant": pr["constant_status"] == "NEAR_CONSTANT", "n_observed_train": int(pr["n_observed_train"]),
            "source_timestamp_field": dictionary.record_date(c.name) or "", "availability_at_prediction_time": c.timing,
            "provenance_status": t["status"], "provenance_reason": t["reason"],
            "leakage_reason": t["reason"] if t["status"] == "UNSAFE" else ("" if disp != "INELIGIBLE_FORBIDDEN" else why),
            "rows_record_on_index_day": t.get("n_on_index"), "rows_record_after_index_day": t.get("n_after_index"),
            "used_in_extended_15": bool(u["baseline"]), "extended_15_features": "; ".join(u["baseline"]), "efalls_concept": "; ".join(efalls),
            "mapping_class": klass, "proposed_engineered_features": "; ".join(feeds), "final_discovery_eligibility": disp, "eligibility_reason": why,
            "contract_note": c.note, "null_means": c.null_means or ""})
    return pd.DataFrame(rows)


def status_definitions() -> dict[str, str]:
    return dict(STATUS_TEXT)
