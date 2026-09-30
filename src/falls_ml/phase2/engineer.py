"""S03 feature engineering: row-wise, parameter-free operations from the catalogue (nothing fitted, no outcome read) and the engineered-feature
registry with a predictor-level provenance contract (reviews/REVIEW_DECISIONS.md A-21). NULL is never turned into 0: "not measured" stays
NULL and is encoded later by the form's single "assessed" indicator."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from falls_ml.phase2.config import Catalogue, FeatureDef
from falls_ml.phase2.registry import Provenance

BASELINE_TRACK = {"SAFE": "ELIGIBLE", "UNRESOLVED": "ELIGIBLE_EXPLORATORY", "UNSAFE": "INELIGIBLE_UNSAFE"}


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("float64")


def compute(f: FeatureDef, df: pd.DataFrame) -> pd.Series:
    if f.op == "copy":
        return _num(df[f.inputs[0]])
    if f.op in ("flag_lt", "flag_ge"):
        s = _num(df[f.inputs[0]])
        hit = s < f.threshold if f.op == "flag_lt" else s >= f.threshold
        return hit.astype("float64").where(s.notna())
    if f.op == "any_positive":
        m = pd.concat([_num(df[c]) for c in f.inputs], axis=1)
        return (m > 0).any(axis=1).astype("float64").where(m.notna().any(axis=1))
    if f.op == "days_since":
        d = df[f.inputs[0]]
        return (df["Index_Date"].dt.normalize() - d.dt.normalize()).dt.days.astype("float64")
    if f.op == "present":
        return df[f.inputs[0]].notna().astype("float64")
    raise ValueError(f"unknown op {f.op}")


def engineer(frame: pd.DataFrame, catalogue: Catalogue) -> pd.DataFrame:
    """Engineered values for every catalogue feature (float64, NaN kept), in catalogue order, aligned with ``frame``."""
    return pd.DataFrame({f.name: compute(f, frame).to_numpy() for f in catalogue.features}, index=frame.index)


def _lookback(meanings: list[str]) -> str:
    text = " ".join(meanings).lower()
    for token in ("30 days", "90 days", "180 days", "365 days", "since 2022", "at index"):
        if token in text:
            return token
    return "latest record before the index day (window not documented)" if "last" in text or "latest" in text else "not documented"


def engineered_registry(values: pd.DataFrame, train_mask: np.ndarray, prov: Provenance, *, catalogue: Catalogue, contract: Any,
                        dictionary: Any, min_observed_train: int) -> pd.DataFrame:
    rows = []
    for f in catalogue.features:
        s = values[f.name]
        s_tr = s[train_mask]
        obs = s_tr.dropna()
        pv = prov.feature(f.inputs)
        n_unique = int(obs.nunique())
        meanings = [dictionary.columns[c]["meaning"] for c in f.inputs]
        null_means = sorted({contract.get(c).null_means or "" for c in f.inputs} - {""})
        unexpected = int(s.isna().sum()) if f.missing == "unexpected" else 0
        if int(len(obs)) < min_observed_train:
            elig, why = "INELIGIBLE_DATA", f"{len(obs)} observed TRAIN rows (< {min_observed_train})"
        elif n_unique <= 1:
            elig, why = "INELIGIBLE_DATA", "constant on the observed TRAIN rows"
        else:
            elig = BASELINE_TRACK[pv["status"]]
            why = "; ".join(pv["reasons"])
            if pv["status"] == "UNSAFE":
                why = "candidate after DWH fix - " + why
        form_date = catalogue.forms[f.form]["date"] if f.form else ""
        rows.append({
            "feature": f.name, "domain": f.domain, "domain_label": catalogue.domains[f.domain]["label"], "text": f.text, "inputs": "; ".join(f.inputs),
            "sources": "; ".join(sorted({dictionary.source(c) for c in f.inputs})), "derivation": f"{f.op}({', '.join(f.inputs)})" + (f" threshold {f.threshold:g}" if f.threshold is not None else ""),
            "kind": f.kind, "linear_encoding": f.linear, "levels": "" if f.levels is None else "; ".join(f"{x:g}" for x in f.levels),
            "missing_semantics": f.missing, "null_means_contract": "; ".join(null_means), "assessment_form": f.form or "", "form_record_date": form_date,
            "is_form_indicator": f.form_indicator, "mapping_class": f.mapping_class, "efalls_concept": f.efalls_concept or "",
            "lookback": _lookback(meanings), "availability_delay": "none declared" if not any("bill" in m.lower() for m in meanings) else "billing lag possible",
            "meaning_status": "; ".join(sorted({dictionary.columns[c]["status"] for c in f.inputs})),
            "provenance_status": pv["status"], "timing_evidence": "; ".join(pv["evidence_kinds"]), "timing_bound_columns": "; ".join(pv["record_dates"]),
            "timing_assumption": ("record date(s) " + ", ".join(pv["record_dates"]) + " bound every contributing record (source-level declaration, Q-P2-01)")
            if pv["record_dates"] else "no record date in the extract: timing declared by the contract only",
            "rows_record_on_index_day": pv["n_on_index"], "rows_record_after_index_day": pv["n_after_index"],
            "n_observed_train": int(len(obs)), "missing_pct_train": round(100.0 * float(s_tr.isna().mean()), 3), "n_unique_train": n_unique,
            "n_unexpected_null": unexpected, "eligibility": elig, "eligibility_reason": why, "catalogue_order": f.order})
    return pd.DataFrame(rows)
