"""Missingness analysis. A NULL is never read as zero here: every NULL cell is assigned to one class, per column:

    TRUE_CLINICAL_ABSENCE  the declared meaning of NULL is the absence of the event / state (no prior fall, no hospitalisation, ...)
    NOT_MEASURED           not assessed / not asked / not graded / not computed
    SOURCE_UNAVAILABLE     the member has no data in the source at all (the VIEW's source-absent flag is set on that row, or the
                           declared NULL meaning says "no ... data")
    NOT_APPLICABLE         the value is undefined by construction (ratio with a zero denominator, no previous assessment, eligible
                           row without an exclusion reason, open follow-up, censored label)
    TECHNICAL_FAILURE      a value was present in the file but could not be parsed into the contract type, or NULL in a column the
                           DDL declares NOT NULL, or NULL where the contract says the value is mandatory
    SENTINEL               a declared sentinel date (e.g. 2999-12-31): present in the file but meaning "no value" (not a NULL cell)
    UNKNOWN                NULL with no declared meaning (needs SME review)

Row-level evidence takes precedence over the declared meaning: a NULL on a row where the event does not exist (no outcome event, not
deceased, no fall) is NOT_APPLICABLE; a NULL on a row whose record source is absent is SOURCE_UNAVAILABLE, and on a row never assessed
in an assessment source (nurse forms, MEFI) NOT_MEASURED. Outcome association of missingness is computed on TRAIN only.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from falls_ml.data.meuhedet_wide import PREDICTOR_ROLES, WideContract
from falls_ml.eda.common import SUPPRESSED, cell, fnum, prevalence_ratio
from falls_ml.eda.dictionary import DataDictionary

CLASSES = ("TRUE_CLINICAL_ABSENCE", "NOT_MEASURED", "SOURCE_UNAVAILABLE", "NOT_APPLICABLE", "TECHNICAL_FAILURE", "SENTINEL", "UNKNOWN")
#: dictionary source -> (row-level "source absent" condition column, value meaning absent, class of the NULLs on such rows). A member with
#: no record at all in an ASSESSMENT source (nurse forms, MEFI) was not assessed: NOT_MEASURED; in a record source (medication, falls,
#: registries, invoices, diagnoses, visits) the member has no data there: SOURCE_UNAVAILABLE.
SOURCE_ABSENT = {
    "FALL_EVENTS": ("Prior_Fall_Missing_Ind", 1, "SOURCE_UNAVAILABLE"), "MEDICATION_EXPOSURE": ("Medication_Missing_Ind", 1, "SOURCE_UNAVAILABLE"),
    "PRESCRIPTIONS": ("Medication_Missing_Ind", 1, "SOURCE_UNAVAILABLE"), "EXTERNAL_CARE_INVOICES": ("External_Care_Missing_Ind", 1, "SOURCE_UNAVAILABLE"),
    "HOSPITALISATIONS": ("External_Care_Missing_Ind", 1, "SOURCE_UNAVAILABLE"), "REGISTRIES": ("Registry_Missing_Ind", 1, "SOURCE_UNAVAILABLE"),
    "DIAGNOSES": ("Has_Source_Diagnosis_Ind", 0, "SOURCE_UNAVAILABLE"), "VISITS": ("Has_Source_Visits_Ind", 0, "SOURCE_UNAVAILABLE"),
    "MEFI": ("Has_Source_Frailty_Ind", 0, "NOT_MEASURED"),
    "NURSE_FALLS_RISK_999_52": ("Has_Source_Assessment_Ind", 0, "NOT_MEASURED"), "NURSE_GET_UP_AND_GO_999_79": ("Has_Source_Assessment_Ind", 0, "NOT_MEASURED"),
    "NURSE_FUNCTION_999_175": ("Has_Source_Assessment_Ind", 0, "NOT_MEASURED"), "NURSE_COGNITION": ("Has_Source_Assessment_Ind", 0, "NOT_MEASURED"),
    "NURSE_HOME_SAFETY_999_139": ("Has_Source_Assessment_Ind", 0, "NOT_MEASURED"), "NURSE_ASSESSMENT_SUMMARY": ("Has_Source_Assessment_Ind", 0, "NOT_MEASURED"),
}
_RULES = (  # declared null meaning (lower case, substring) -> class; first match wins
    ("invalid row", "TECHNICAL_FAILURE"), ("unknown", "UNKNOWN"),
    ("no medication data", "SOURCE_UNAVAILABLE"), ("no fall data", "SOURCE_UNAVAILABLE"), ("no assessment source data", "SOURCE_UNAVAILABLE"),
    ("not assessed", "NOT_MEASURED"), ("not asked", "NOT_MEASURED"), ("not graded", "NOT_MEASURED"), ("not computed", "NOT_MEASURED"),
    ("never assessed", "NOT_MEASURED"),
    ("ratio undefined", "NOT_APPLICABLE"), ("no previous assessment", "NOT_APPLICABLE"), ("eligible", "NOT_APPLICABLE"),
    ("follow-up open", "NOT_APPLICABLE"), ("censored", "NOT_APPLICABLE"),
    ("no prior fall", "TRUE_CLINICAL_ABSENCE"), ("no diagnosis", "TRUE_CLINICAL_ABSENCE"), ("no visit", "TRUE_CLINICAL_ABSENCE"),
    ("no hospitalisation", "TRUE_CLINICAL_ABSENCE"), ("no registry", "TRUE_CLINICAL_ABSENCE"), ("not in the smi registry", "TRUE_CLINICAL_ABSENCE"),
    ("no risk code", "TRUE_CLINICAL_ABSENCE"), ("no nursing-care status", "TRUE_CLINICAL_ABSENCE"), ("no fall", "TRUE_CLINICAL_ABSENCE"),
)
THRESHOLDS = (99.0, 90.0, 50.0, 25.0)
#: columns defined only when another column says the event exists: a NULL on the other rows is NOT_APPLICABLE by construction
#: (column -> (condition column, value meaning "the event does not exist"))
CONDITIONAL = {
    **{f"{c}_{h}": (f"Fall_Next_{h}_Ind", 0) for h in ("30D", "180D") for c in ("Next_Fall_Date", "Days_To_Next_Fall", "Next_Fall_Event_ID", "Next_Fall_Confidence")},
    "Death_Censor_Date": ("Is_Deceased_Ind", 0), "Death_Date_Gap_Days": ("Is_Deceased_Ind", 0),
    "Last_Fall_Confidence": ("Prior_Fall_Since_Study_Start_Ind", 0),
}


def declared_class(contract: WideContract, col: str) -> str:
    """Class of a NULL from the contract's declared NULL meaning; a last-record date without one takes the declaration of its
    'days since' companion column (both describe the same last record); NOT NULL columns -> TECHNICAL_FAILURE; else UNKNOWN."""
    from falls_ml.data.meuhedet_timing import DAYS_SINCE_COMPANION

    c = contract.get(col)
    text = (c.null_means or "").lower()
    if not text and col in DAYS_SINCE_COMPANION and DAYS_SINCE_COMPANION[col] in contract.names:
        text = (contract.get(DAYS_SINCE_COMPANION[col]).null_means or "").lower()
    for key, klass in _RULES:
        if key in text:
            return klass
    if not c.nullable:
        return "TECHNICAL_FAILURE"
    return "UNKNOWN"


def classify_nulls(frame: pd.DataFrame, col: str, contract: WideContract, dictionary: DataDictionary,
                   failure_masks: dict[str, pd.Series], wrong_type: dict[str, int], mask: pd.Series | None = None) -> dict[str, int]:
    """Counts per class for one column over ``mask`` rows (default all)."""
    s = frame[col]
    m = pd.Series(True, index=frame.index) if mask is None else mask
    null = s.isna() & m
    counts = dict.fromkeys(CLASSES, 0)
    c = contract.get(col)
    if c.sentinels and c.is_date:
        counts["SENTINEL"] = int((s.isin([pd.Timestamp(v) for v in c.sentinels]) & m).sum())
    if not null.any():
        return counts
    remaining = null.copy()
    unlocated = 0
    if col in failure_masks:
        tech = remaining & failure_masks[col]
        counts["TECHNICAL_FAILURE"] += int(tech.sum())
        remaining &= ~tech
    elif wrong_type.get(col) and mask is None:
        unlocated = min(int(wrong_type[col]), int(remaining.sum()))   # counts only (non-CSV input): the parse failures are among the NULLs
        counts["TECHNICAL_FAILURE"] += unlocated
    cond = CONDITIONAL.get(col)
    if cond and cond[0] in frame.columns:
        na = (frame[cond[0]] == cond[1]).fillna(False).astype(bool) & remaining
        if col.startswith(("Next_Fall_", "Days_To_Next_Fall_")):   # a censored (NULL) label also has no event date
            na |= frame[cond[0]].isna() & remaining
        counts["NOT_APPLICABLE"] += int(na.sum())
        remaining &= ~na
    absent = SOURCE_ABSENT.get(dictionary.source(col))
    if absent and absent[0] in frame.columns and absent[0] != col:
        a = (frame[absent[0]] == absent[1]).fillna(False).astype(bool) & remaining
        counts[absent[2]] += int(a.sum())
        remaining &= ~a
    declared = int(remaining.sum()) - unlocated
    if declared < 0 and absent:   # the unlocated failures sat on source-absent rows
        counts[absent[2]] += declared
        declared = 0
    counts[declared_class(contract, col)] += declared
    return counts


def missingness_table(frame: pd.DataFrame, contract: WideContract, dictionary: DataDictionary, bases: dict[str, pd.Series],
                      failure_masks: dict[str, pd.Series], wrong_type: dict[str, int], min_cell: int) -> pd.DataFrame:
    """``missingness.csv``: per column, missing count / % per basis, declared NULL meaning, class breakdown (full extract), threshold band."""
    n = len(frame)
    rows = []
    for c in contract.columns:
        if c.name not in frame.columns:
            continue
        s = frame[c.name]
        k = int(s.isna().sum())
        classes = classify_nulls(frame, c.name, contract, dictionary, failure_masks, wrong_type)
        pct = 100.0 * k / n if n else 0.0
        eff = k + classes["SENTINEL"]
        r: dict[str, Any] = {"column": c.name, "domain": dictionary.domain_label(c.name), "source": dictionary.source(c.name), "contract_role": c.role,
                             "declared_type": c.semantic, "nullable_in_ddl": c.nullable, "declared_null_meaning": c.null_means or "",
                             "declared_null_class": declared_class(contract, c.name), "missing_count": cell(k, min_cell),
                             "missing_pct": fnum(pct, 2) if cell(k, min_cell) != SUPPRESSED else SUPPRESSED,
                             "missing_or_sentinel_pct": fnum(100.0 * eff / n, 2) if n and cell(eff, min_cell) != SUPPRESSED else (SUPPRESSED if n else None)}
        for bname, mask in bases.items():
            mm = int(mask.sum())
            kk = int(s[mask].isna().sum())
            r[f"missing_pct_{bname.lower()}"] = (fnum(100.0 * kk / mm, 2) if cell(kk, min_cell) != SUPPRESSED else SUPPRESSED) if mm else None
        for klass in CLASSES:
            r[f"n_{klass.lower()}"] = cell(classes[klass], min_cell)
        nonzero = {kl: v for kl, v in classes.items() if v}
        r["dominant_class"] = max(nonzero, key=nonzero.get) if nonzero else ""
        r["band"] = next((f">{t:g}%" for t in THRESHOLDS if pct > t), "0%" if k == 0 else "<=25%")
        absent = SOURCE_ABSENT.get(dictionary.source(c.name))
        r["source_absent_flag"] = f"{absent[0]} == {absent[1]} -> {absent[2]}" if absent else ""
        rows.append(r)
    return pd.DataFrame(rows)


def by_domain(miss: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dom, g in miss.groupby("domain", sort=True):
        pcts = pd.to_numeric(g["missing_pct"], errors="coerce").fillna(0.0)
        rows.append({"domain": dom, "n_columns": len(g), "mean_missing_pct": fnum(pcts.mean(), 2), "median_missing_pct": fnum(pcts.median(), 2),
                     "n_columns_gt_25pct": int((pcts > 25).sum()), "n_columns_gt_50pct": int((pcts > 50).sum()),
                     "n_columns_gt_90pct": int((pcts > 90).sum()), "n_columns_gt_99pct": int((pcts > 99).sum()),
                     "n_columns_complete": int((pcts == 0).sum())})
    return pd.DataFrame(rows)


def predictor_columns_for_missingness(contract: WideContract, frame: pd.DataFrame) -> list[str]:
    return [c.name for c in contract.columns if c.name in frame.columns and c.role in PREDICTOR_ROLES and not c.is_date]


def co_missingness(frame: pd.DataFrame, cols: list[str], *, lo: float = 0.005, hi: float = 0.995) -> tuple[pd.DataFrame, list[str]]:
    """Phi correlation of the missing indicators of partially missing columns, in hierarchical-cluster order."""
    ind = frame[cols].isna()
    share = ind.mean()
    keep = [c for c in cols if lo <= share[c] <= hi]
    if len(keep) < 2:
        return pd.DataFrame(), keep
    m = ind[keep].to_numpy(dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        phi = np.corrcoef(m, rowvar=False)
    phi = np.nan_to_num(phi, nan=0.0)
    order = keep
    try:
        from scipy.cluster.hierarchy import leaves_list, linkage
        from scipy.spatial.distance import squareform

        dist = np.clip(1.0 - phi, 0.0, 2.0)
        np.fill_diagonal(dist, 0.0)
        order = [keep[i] for i in leaves_list(linkage(squareform(dist, checks=False), method="average"))]
    except (ImportError, ValueError):
        pass
    mat = pd.DataFrame(phi, index=keep, columns=keep).loc[order, order]
    return mat, order


def co_missing_pairs(frame: pd.DataFrame, mat: pd.DataFrame, dictionary: DataDictionary, min_cell: int, threshold: float = 0.8) -> pd.DataFrame:
    rows = []
    cols = list(mat.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            phi = float(mat.at[a, b])
            if abs(phi) < threshold:
                continue
            both = int((frame[a].isna() & frame[b].isna()).sum())
            either = int((frame[a].isna() | frame[b].isna()).sum())
            rows.append({"column_a": a, "column_b": b, "phi_missing_indicators": fnum(phi, 4), "jaccard": fnum(both / either, 4) if either else None,
                         "n_both_missing": cell(both, min_cell), "same_source": dictionary.source(a) == dictionary.source(b),
                         "source_a": dictionary.source(a), "source_b": dictionary.source(b)})
    return pd.DataFrame(rows).sort_values("phi_missing_indicators", ascending=False).reset_index(drop=True) if rows else pd.DataFrame(
        columns=["column_a", "column_b", "phi_missing_indicators", "jaccard", "n_both_missing", "same_source", "source_a", "source_b"])


def source_block_patterns(frame: pd.DataFrame, contract: WideContract, dictionary: DataDictionary, mask: pd.Series, min_cell: int,
                          top: int = 15) -> tuple[pd.DataFrame, list[str]]:
    """Row-level pattern of 'every predictor column of this source is NULL', per source, over ``mask`` rows; patterns < min_cell merged."""
    sources: dict[str, list[str]] = {}
    for c in contract.columns:
        if c.name in frame.columns and c.role in PREDICTOR_ROLES and not c.is_date:
            sources.setdefault(dictionary.source(c.name), []).append(c.name)
    blocks = {src: frame.loc[mask, cols].isna().all(axis=1) for src, cols in sources.items() if frame.loc[mask, cols].isna().all(axis=1).any()}
    names = sorted(blocks)
    if not names:
        return pd.DataFrame(), []
    key = pd.DataFrame(blocks)[names].astype(int).astype(str).agg("".join, axis=1)
    vc = key.value_counts()
    rows, other = [], 0
    n = int(mask.sum())
    for pat, k in vc.items():
        if len(rows) >= top or k < min_cell:
            other += int(k)
            continue
        rows.append({"pattern": pat, "n_rows": int(k), "pct": fnum(100.0 * k / n, 2), **{src: int(ch) for src, ch in zip(names, pat)},
                     "n_sources_missing": pat.count("1")})
    if other:
        rows.append({"pattern": "(other patterns, merged)", "n_rows": cell(other, min_cell), "pct": fnum(100.0 * other / n, 2) if other >= min_cell else SUPPRESSED,
                     **{src: None for src in names}, "n_sources_missing": None})
    return pd.DataFrame(rows), names


def row_missingness(frame: pd.DataFrame, cols: list[str], mask: pd.Series) -> dict[str, Any]:
    """Distribution of the per-row share of missing predictor columns (aggregate only)."""
    share = frame.loc[mask, cols].isna().mean(axis=1)
    q = share.quantile([0.5, 0.9, 0.99]) if len(share) else pd.Series(dtype=float)
    thr = float(q.get(0.99, 1.0)) if len(share) else 1.0
    return {"n_rows": int(mask.sum()), "n_columns": len(cols), "median_share": fnum(q.get(0.5), 4), "p90_share": fnum(q.get(0.9), 4),
            "p99_share": fnum(thr, 4), "n_rows_above_50pct": int((share > 0.5).sum()), "n_rows_above_75pct": int((share > 0.75).sum()),
            "n_rows_above_p99": int((share > thr).sum()), "share_values": share}


def missingness_outcome(train: Any, cols: list[str], contract: WideContract, dictionary: DataDictionary, min_cell: int) -> pd.DataFrame:
    """TRAIN only: outcome prevalence among rows where the column is missing vs observed (descriptive)."""
    from falls_ml.eda.supervised import TrainPartition

    if not isinstance(train, TrainPartition):
        raise TypeError("missingness_outcome accepts only a TrainPartition (target-aware analyses use TRAIN only)")
    f, y = train.frame, train.y
    n = len(f)
    rows = []
    for col in cols:
        miss = f[col].isna().to_numpy()
        k = int(miss.sum())
        if k == 0 or k == n or k / n < 0.01:
            continue
        a, c = int(y[miss].sum()), int(y[~miss].sum())
        pr, lo, hi = prevalence_ratio(a, k, c, n - k)
        sup = any(0 < v < min_cell for v in (a, k - a, c, (n - k) - c))
        rows.append({"column": col, "domain": dictionary.domain_label(col), "declared_null_class": declared_class(contract, col),
                     "n_missing": cell(k, min_cell), "events_missing": cell(a, min_cell), "prevalence_missing": SUPPRESSED if sup else fnum(a / k, 4),
                     "n_observed": cell(n - k, min_cell), "events_observed": cell(c, min_cell), "prevalence_observed": SUPPRESSED if sup else fnum(c / (n - k), 4),
                     "prevalence_ratio_missing_vs_observed": None if sup else fnum(pr, 3), "pr_ci_low": None if sup else fnum(lo, 3),
                     "pr_ci_high": None if sup else fnum(hi, 3), "basis": "TRAIN_ONLY"})
    return pd.DataFrame(rows)
