"""Exact provenance / funnel accounting (FINAL readiness brief §3), shared by the preflight (printed before any fit) and COLUMN_FUNNEL.csv.

Every number is a count of the committed registries and feature sets of THIS extract; nothing is estimated. The categories are never mixed:
each row names the category it belongs to.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from falls_ml.phase2.featuresets import ALL_SAFE, BASELINE, EXPL_ALL, EXPLORATORY, HISTORICAL, SAFE_BASE, SAFE_DISCOVERY, FeatureSets

DISPOSITION_TEXT = {
    "INELIGIBLE_FORBIDDEN": "forbidden (identifier, label / follow-up, contract leakage, post-index or unknown timing)",
    "NON_PREDICTIVE_METADATA": "metadata / non-predictive (QA and cohort bookkeeping)",
    "RECORD_DATE": "record dates (timing evidence only; enter features only through days-since / assessed indicators)",
    "NEEDS_SME": "quarantined: undocumented codes (needs SME review; never an input)",
    "REPRESENTED_BY": "duplicate / represented by another feature (not used separately)",
    "IN_BASELINE_15": "source columns of the BASELINE_15 predictors",
    "INELIGIBLE_DATA": "empty / constant / too few observed TRAIN rows",
    "INELIGIBLE_UNSAFE": "timing proven UNSAFE (record on/after the index day): candidate after DWH fix",
    "ELIGIBLE_EXPLORATORY": "UNRESOLVED timing (exploratory sensitivity only)",
    "ELIGIBLE": "SAFE (proven prediction-time provenance)",
}


def funnel_rows(reg: pd.DataFrame, er: pd.DataFrame, fs: FeatureSets, baseline_status: dict[str, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(step: str, n: int, category: str = "", note: str = "") -> None:
        rows.append({"step": step, "n": int(n), "category": category, "note": note})

    add("raw columns in the extract (contract)", len(reg), note="every column has one disposition in 01_COLUMN_REGISTRY.csv")
    for st in ("SAFE", "UNRESOLVED", "UNSAFE"):
        add(f"raw columns with D-00 status {st}", int((reg["provenance_status"] == st).sum()), note="status of the column's own source on TRAIN+VALIDATION")
    for disp, text in DISPOSITION_TEXT.items():
        add(f"raw columns: {disp}", int((reg["final_discovery_eligibility"] == disp).sum()), note=text)
    bl = list(fs.sets[BASELINE]["baseline"])
    add("BASELINE_15 predictors", len(bl), HISTORICAL, f"built from {int((reg['final_discovery_eligibility'] == 'IN_BASELINE_15').sum())} source columns")
    for st in ("SAFE", "UNRESOLVED", "UNSAFE"):
        names = [f for f in bl if baseline_status.get(f) == st]
        add(f"BASELINE_15 predictors with status {st}", len(names), HISTORICAL, ", ".join(names))
    add("engineered candidate features (catalogue)", len(er), note="93 declared before any data was seen")
    for elig, text in (("ELIGIBLE", "SAFE"), ("ELIGIBLE_EXPLORATORY", "UNRESOLVED"), ("INELIGIBLE_UNSAFE", "UNSAFE (candidate after DWH fix)"),
                       ("INELIGIBLE_DATA", "empty / constant / too few observed TRAIN rows")):
        add(f"engineered features: {text}", int((er["eligibility"] == elig).sum()))
    safe_base = list(fs.sets[SAFE_BASE]["baseline"])
    add("SAFE_BASE predictors (SAFE subset of BASELINE_15)", len(safe_base), SAFE_DISCOVERY, ", ".join(safe_base))
    pruned = fs.design.get("pruned_out", {})
    pruned_safe = pruned.get(SAFE_DISCOVERY, []) if isinstance(pruned, dict) else []
    add("SAFE engineered features removed as redundant (|Spearman| >= threshold with a representative)", len(pruned_safe), SAFE_DISCOVERY, ", ".join(pruned_safe))
    add("SAFE new features entering the discovery (ALL_REVIEWED_SAFE)", int(fs.sets[ALL_SAFE]["n_new_features"]), SAFE_DISCOVERY,
        "incl. form 'assessed' indicators")
    add("predictors in the largest SAFE model (SAFE_BASE + new)", int(fs.sets[ALL_SAFE]["n_new_features"]) + len(safe_base), SAFE_DISCOVERY)
    ex = fs.sets[EXPL_ALL]
    reg_e = er.set_index("feature")
    n_unres = sum(1 for f in ex["features"] if f in reg_e.index and reg_e.at[f, "eligibility"] == "ELIGIBLE_EXPLORATORY")
    add("new features offered in EXPLORATORY_ALL_REVIEWED (SAFE + UNRESOLVED)", int(ex["n_new_features"]), EXPLORATORY,
        f"{n_unres} UNRESOLVED; the historical BASELINE_15 is its base")
    for cat in (HISTORICAL, SAFE_DISCOVERY, EXPLORATORY):
        names = [s for s in fs.fitted() if fs.sets[s]["category"] == cat and fs.sets[s]["kind"] != "loo"]
        add(f"feature sets fitted (LASSO; distinct specifications) - {cat}", len(names), cat)
    return rows
