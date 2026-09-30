"""Systematic data-quality checks on the whole extract (unsupervised; nothing is repaired).

Each check is recorded with a status: FINDING (rows affected), PASS (checked, nothing found) or NOT_APPLICABLE (columns absent / not
checkable), a severity (CRITICAL = the build or the science is affected; WARNING = to resolve or explain before modelling; INFO =
to know), the affected row count (small counts suppressed) and a recommendation. Consistency rules between columns are EXPECTED
relations read from the column names and the S2T notes: a violation is a question for the DWH / SME, not a proven error.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from falls_ml.data.meuhedet_timing import DAYS_SINCE_COMPANION
from falls_ml.data.meuhedet_wide import PREDICTOR_ROLES, WideContract, WideMapping
from falls_ml.eda.common import FULL_EXTRACT, SUPPRESSED, TRAIN_ONLY, cell, fnum, numeric
from falls_ml.eda.profile import FAR_FUTURE, TIME_ONLY_RE, TOO_EARLY, constant_status, is_identifier

SEVERITIES = ("CRITICAL", "WARNING", "INFO")
REGISTRY_FLAGS = ("Registry_Dementia_Ind", "Registry_Home_Confined_Ind", "Registry_Suicide_Attempt_Ind", "Registry_Heart_Disease_Ind",
                  "Registry_Diabetes_T1_Ind", "Registry_Diabetes_T2_Ind", "Registry_Chronic_Renal_Failure_Ind", "Registry_Kidneys_Ind",
                  "Registry_Liver_Ind", "Registry_COPD_Ind", "Registry_Asthma_Ind", "Registry_Blood_Pressure_Ind", "Registry_Transplant_Ind",
                  "Registry_Stoma_Ind")
NURSE_FIELDS = ("Falls_Risk_Assessment_Score", "Instability_Dizziness_Ind", "Fear_Of_Falling_Ind", "Balance_Disorder_Ind", "Uses_Walking_Aid_Ind",
                "Mobility_Score", "ADL_Worst_Domain_Score", "MiniCog_Score", "Home_Safety_Risk_Count")
SOURCE_FLAGS = {"Has_Source_Visits_Ind": "Visit_Count_365D", "Has_Source_Diagnosis_Ind": "Diagnosis_Count_365D",
                "Has_Source_Medication_Ind": "Distinct_Active_Substance_Count", "Has_Source_Registry_Ind": "Active_Registry_Count",
                "Has_Source_Frailty_Ind": "MEFI_Group_At_Index", "Has_Source_External_Care_Ind": "External_Care_Count_365D_Visible"}


@dataclass
class Findings:
    n_rows: int
    min_cell: int
    rows: list[dict[str, Any]] = field(default_factory=list)

    def add(self, check_id: str, category: str, severity: str, title: str, columns: list[str] | str, affected: Any, *, detail: str = "",
            recommendation: str = "", basis: str = FULL_EXTRACT, denominator: int | None = None) -> None:
        """``affected``: boolean mask, an int count, or None (= not applicable)."""
        cols = columns if isinstance(columns, str) else "; ".join(columns)
        if affected is None:
            status, n = "NOT_APPLICABLE", None
        else:
            n = int(affected.sum()) if isinstance(affected, (pd.Series, np.ndarray)) else int(affected)
            status = "FINDING" if n > 0 else "PASS"
        d = denominator if denominator is not None else self.n_rows
        c = None if n is None else cell(n, self.min_cell)
        self.rows.append({"check_id": check_id, "category": category, "severity": severity if status == "FINDING" else "", "status": status,
                          "title": title, "columns": cols, "n_rows": c,
                          "pct_rows": (fnum(100.0 * n / d, 3) if (n is not None and d and c != SUPPRESSED) else (SUPPRESSED if c == SUPPRESSED else None)),
                          "basis": basis, "detail": detail, "recommendation": recommendation if status == "FINDING" else ""})

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def _has(f: pd.DataFrame, *cols: str) -> bool:
    return all(c in f.columns for c in cols)


def _num(f: pd.DataFrame, c: str) -> pd.Series:
    return numeric(f[c])


def _age_group_bounds(v: str) -> tuple[float, float] | None:
    m = re.match(r"^\s*(\d{1,3})\s*[-–]\s*(\d{1,3})\s*$", v)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.match(r"^\s*(\d{1,3})\s*\+\s*$", v)
    if m:
        return float(m.group(1)), float("inf")
    return None


def data_quality(frame: pd.DataFrame, contract: WideContract, mapping: WideMapping, *, index: pd.Series, wrong_type: dict[str, int],
                 patterns: dict[str, dict[str, Any]], contract_report: Any, min_cell: int, eligible: pd.Series,
                 corr_pairs: pd.DataFrame | None = None, window_days: int = 180) -> pd.DataFrame:
    """Run every check; returns all checks (``status`` FINDING / PASS / NOT_APPLICABLE)."""
    f, n = frame, len(frame)
    F = Findings(n, min_cell)
    id_col, idx_col = mapping.identity["research_id"], mapping.identity["index_date"]

    # ---------------- A. columns
    for c in contract.columns:
        if c.name not in f.columns:
            F.add("A01", "column", "WARNING", "contract column absent from the file", c.name, 1, detail="the VIEW has 221 columns; this one was not exported",
                  recommendation="re-export the full VIEW")
            continue
        status, share = constant_status(f[c.name])
        predictor = c.role in PREDICTOR_ROLES
        if status == "EMPTY":
            expected = "empty in all rows" in (c.note or "")
            F.add("A02", "column", "INFO" if expected or not predictor else "WARNING", "column is empty (all NULL)", c.name, n,
                  detail="expected per S2T" if expected else "no value in any row", recommendation="drop from analysis; ask the DWH whether it is populated")
        elif status == "CONSTANT" and not is_identifier(contract, c.name):
            intended = c.allowed is not None and len(c.allowed) == 1
            F.add("A03", "column", "INFO" if (intended or not predictor) else "WARNING", "column is constant", c.name, 0 if intended else n,
                  detail=f"one distinct non-null value (share {share:.3f})", recommendation="no information for modelling; confirm with the DWH")
        elif status == "NEAR_CONSTANT" and predictor:
            k = int(f[c.name].notna().sum() * (1 - share))
            F.add("A04", "column", "INFO", "near-constant predictor column (top value >= 99.5% of non-null values)", c.name, k,
                  detail=f"minority values: {cell(k, min_cell)} rows", recommendation="sparse: unstable estimates, perfect-separation risk; consider grouping or exclusion (decide on TRAIN)")
    # duplicate columns (identical values and NULL pattern), constant / empty columns excluded
    cand = [c.name for c in contract.columns if c.name in f.columns and not is_identifier(contract, c.name) and f[c.name].nunique(dropna=True) > 1]
    sig: dict[tuple[Any, ...], list[str]] = {}
    for c in cand:
        s = f[c]
        key = (int(s.notna().sum()), str(s.dtype), int(pd.util.hash_pandas_object(s.astype("string").fillna("<NA>"), index=False).sum() % (2 ** 61)))
        sig.setdefault(key, []).append(c)
    parent: dict[str, str] = {}

    def root(x: str) -> str:
        while parent.get(x, x) != x:
            x = parent[x]
        return x

    for group in sig.values():
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                if root(a) != root(b) and f[a].astype("string").fillna("<NA>").equals(f[b].astype("string").fillna("<NA>")):
                    parent[root(b)] = root(a)
    groups: dict[str, list[str]] = {}
    for c in cand:
        if c in parent or any(parent.get(x) == c for x in parent):
            groups.setdefault(root(c), []).append(c)
    for members in groups.values():
        if len(members) > 1:
            F.add("A05", "column", "WARNING", "identical columns (same values and NULL pattern in every row)", members, n,
                  detail=f"{len(members)} columns carry the same content", recommendation="keep one; confirm with the DWH that they are meant to be identical")
    if not any(len(m) > 1 for m in groups.values()):
        F.add("A05", "column", "WARNING", "identical columns (same values and NULL pattern)", "all non-constant columns", 0)
    if corr_pairs is not None and len(corr_pairs):
        red = corr_pairs[corr_pairs["redundancy"].isin(["CANDIDATE_DUPLICATE_DEFINITION", "HIGHLY_REDUNDANT"])]
        for _, r in red.iterrows():
            F.add("A06", "column", "INFO", f"highly redundant pair ({r['redundancy']})", [r["column_a"], r["column_b"]], 1, basis=TRAIN_ONLY,
                  detail=f"Spearman {r['spearman']} / Cramér's V {r['cramers_v']} on TRAIN", recommendation="do not enter both into one model without a reason; see correlation_pairs.csv")
    else:
        F.add("A06", "column", "INFO", "highly redundant pairs", "predictor columns", None if corr_pairs is None else 0, basis=TRAIN_ONLY,
              detail="needs the TRAIN correlation stage" if corr_pairs is None else "")
    # ---------------- B. types, parsing, contract values
    for col, k in sorted(wrong_type.items()):
        if k:
            pats = "; ".join(f"{p}: {v}" for p, v in (patterns.get(col) or {}).items())
            time_only = any(re.match(r"^9{1,2}:99(:99)?(\.9+)?$", p) for p in (patterns.get(col) or {}))
            F.add("B01", "parsing", "WARNING", "values that could not be parsed into the contract type (now NULL, never repaired)", col, k,
                  detail=f"raw text shapes (digits->9, letters->a): {pats or 'n/a (non-CSV input)'}" + ("; time-only text such as 00:00.0 = an Excel-formatted date/time exported as text" if time_only else ""),
                  recommendation="re-export the column as ISO dates (yyyy-mm-dd) / plain numbers; do not guess the values")
    if not any(wrong_type.values()):
        F.add("B01", "parsing", "WARNING", "values that could not be parsed into the contract type", "all columns", 0)
    text_cols = [c.name for c in contract.columns if c.name in f.columns and c.is_text and not is_identifier(contract, c.name)]
    malformed = {c: int(f[c].dropna().astype(str).str.match(TIME_ONLY_RE).sum()) for c in text_cols}
    for c, k in malformed.items():
        if k:
            F.add("B02", "parsing", "WARNING", "malformed text: time-only strings such as 00:00.0 in a text column", c, k,
                  recommendation="Excel reformatted the cells; re-export from SQL as text")
    if not any(malformed.values()):
        F.add("B02", "parsing", "WARNING", "malformed text (time-only strings such as 00:00.0)", "all text columns", 0)
    inv = dict(getattr(contract_report, "invalid_values", {}) or {})
    for col, k in sorted(inv.items()):
        if k:
            F.add("B03", "contract", "WARNING", "values outside the contract's allowed list (out-of-contract categorical values)", col, k,
                  detail=f"allowed {list(contract.get(col).allowed or [])}", recommendation="ask the DWH for the code meaning; never recode silently")
    if not any(inv.values()):
        F.add("B03", "contract", "WARNING", "values outside the contract's allowed lists", "coded columns", 0)
    nnv = dict(getattr(contract_report, "non_null_violations", {}) or {})
    for col, k in sorted(nnv.items()):
        if k:
            F.add("B04", "contract", "WARNING", "NULL in a column the DDL declares NOT NULL", col, k, recommendation="VIEW / export defect: ask the DWH")
    if not any(nnv.values()):
        F.add("B04", "contract", "WARNING", "NULL in NOT NULL columns", "NOT NULL columns", 0)
    # ---------------- C. dates
    for c in contract.columns:
        if c.name not in f.columns or not c.is_date:
            continue
        s = f[c.name]
        sent = s.isin([pd.Timestamp(v) for v in c.sentinels]) if c.sentinels else pd.Series(False, index=s.index)
        F.add("C01", "dates", "WARNING", "date before 1900-01-01 (impossible for this population)", c.name, s < TOO_EARLY,
              recommendation="VIEW default value? ask the DWH; never used as a real date")
        F.add("C02", "dates", "WARNING", "undeclared far-future date (>= 2100): possible undeclared sentinel", c.name, (s >= FAR_FUTURE) & ~sent,
              recommendation="declare it as a sentinel with its business meaning (like Q-M-08 / Q-M-09) or have it corrected")
        if c.sentinels:
            F.add("C03", "dates", "INFO", f"declared sentinel {', '.join(c.sentinels)} ({c.sentinel_means or ''})".strip(), c.name, sent,
                  detail=f"modelling rule {c.sentinel_modeling_rule}", recommendation="kept as declared; not a real date")
    for col, companion in DAYS_SINCE_COMPANION.items():
        if not _has(f, col, companion, idx_col):
            continue
        ds = _num(f, companion)
        d = f[col].dt.normalize()
        F.add("C04", "dates", "WARNING", "negative 'days since' value (record dated after Index_Date)", companion, ds < 0,
              recommendation="the VIEW computed it from a future record: D-00 / snapshot defect")
        both = ds.notna() & d.notna()
        mism = both & (ds != (index - d).dt.days)
        F.add("C05", "dates", "INFO", f"{companion} differs from Index_Date - {col}", [companion, col], mism, denominator=int(both.sum()),
              detail="the day count and the date disagree", recommendation="ask which one is authoritative")
    seq = [("Population_Record_From_Date", "Population_Record_To_Date", "SCD validity start after its end"),
           ("Obs_Window_Start", idx_col, "observation window starts on/after Index_Date")]
    for a, b, title in seq:
        if _has(f, a, b):
            bad = (f[a] > f[b]) if a != "Obs_Window_Start" else (f[a].dt.normalize() >= f[b].dt.normalize())
            F.add("C06", "temporal sequence", "WARNING", f"impossible temporal sequence: {title}", [a, b], bad.fillna(False),
                  recommendation="VIEW defect: ask the DWH")
    if _has(f, "Population_Record_From_Date", idx_col):
        F.add("C07", "temporal sequence", "INFO", "population SCD row starts after Index_Date", ["Population_Record_From_Date", idx_col],
              (f["Population_Record_From_Date"].dt.normalize() > index).fillna(False), recommendation="the member's population row was not valid at the index date?")
    if _has(f, "Death_Censor_Date", idx_col):
        F.add("C08", "temporal sequence", "WARNING", "death date before Index_Date", ["Death_Censor_Date", idx_col],
              (f["Death_Censor_Date"].dt.normalize() < index).fillna(False), recommendation="a deceased member cannot have a snapshot: VIEW / source defect")
    if _has(f, "Followup_End_Date", idx_col):
        c = contract.get("Followup_End_Date")
        sent = f["Followup_End_Date"].isin([pd.Timestamp(v) for v in c.sentinels]) if c.sentinels else False
        F.add("C09", "temporal sequence", "WARNING", "follow-up ends before Index_Date", ["Followup_End_Date", idx_col],
              ((f["Followup_End_Date"].dt.normalize() < index) & ~sent).fillna(False), recommendation="VIEW defect: ask the DWH")
    for h, w in (("30D", 30), ("180D", window_days)):
        end, ev = f"Label_End_{h}", f"Next_Fall_Date_{h}"
        if _has(f, end, idx_col):
            days = (f[end].dt.normalize() - index).dt.days
            F.add("C10", "temporal sequence", "WARNING", f"{end} - Index_Date differs from {w} days", [end, idx_col], (days.notna() & (days != w)),
                  recommendation="outcome window of unexpected length: confirm the label definition")
        if _has(f, ev, end, idx_col):
            e = f[ev].dt.normalize()
            F.add("C11", "outcome", "WARNING", f"{ev} outside [Index_Date, {end}]", [ev, idx_col, end],
                  (e.notna() & ((e < index) | (e > f[end].dt.normalize()))).fillna(False), recommendation="label defect: an event outside its window")
    # ---------------- D. numeric plausibility
    for c in contract.columns:
        if c.name in f.columns and c.semantic == "count" and not c.is_date and not c.is_text:
            F.add("D01", "numeric", "WARNING", "negative counter", c.name, _num(f, c.name) < 0, recommendation="impossible count: VIEW defect")
    if "Age_At_Index" in f.columns:
        age = _num(f, "Age_At_Index")
        F.add("D02", "numeric", "CRITICAL", "implausible age (< 0 or > 120)", "Age_At_Index", (age < 0) | (age > 120),
              recommendation="impossible; ask the DWH (default birth dates?)")
        F.add("D03", "numeric", "WARNING", "eligible row with age < 65 (cohort is 65+)", ["Age_At_Index", "Is_Eligible_Cohort"], eligible & (age < 65),
              basis=f"{FULL_EXTRACT} (eligible rows)", denominator=int(eligible.sum()), recommendation="eligibility / age defect")
        if "Age_Group" in f.columns:
            g = f["Age_Group"].astype("string")
            bounds = g.dropna().map(_age_group_bounds)
            if len(bounds) and bounds.notna().mean() >= 0.9:
                lo = g.map(lambda v: (_age_group_bounds(v) or (np.nan, np.nan))[0] if pd.notna(v) else np.nan).astype("float64")
                hi = g.map(lambda v: (_age_group_bounds(v) or (np.nan, np.nan))[1] if pd.notna(v) else np.nan).astype("float64")
                F.add("D04", "consistency", "WARNING", "Age_Group does not contain Age_At_Index", ["Age_Group", "Age_At_Index"],
                      (age.notna() & lo.notna() & ((age < lo) | (age > hi))), recommendation="two derivations of age disagree")
            else:
                F.add("D04", "consistency", "WARNING", "Age_Group vs Age_At_Index", ["Age_Group", "Age_At_Index"], None, detail="Age_Group format not recognised")
    if "Birth_Date_Suspect_Ind" in f.columns:
        F.add("D05", "numeric", "INFO", "default (suspect) birth date: age unreliable", "Birth_Date_Suspect_Ind", _num(f, "Birth_Date_Suspect_Ind") == 1,
              recommendation="kept in the primary cohort; a sensitivity analysis can exclude them")
    for col in ("Prescription_Fill_Ratio_365D", "Lab_Performed_Ratio_365D"):
        if col in f.columns:
            r = _num(f, col)
            F.add("D06", "numeric", "INFO", "ratio outside [0, 1]", col, (r < 0) | (r > 1), recommendation="confirm the ratio definition (numerator can exceed denominator?)")
    # ---------------- E. flag / count consistency (expected relations)
    def monotone(prefix: str, windows: list[str], title: str) -> None:
        cols = [f"{prefix}{w}" for w in windows]
        if not _has(f, *cols):
            return
        bad = pd.Series(False, index=f.index)
        for a, b in zip(cols, cols[1:]):
            bad |= (_num(f, a) > _num(f, b)).fillna(False)
        F.add("E01", "consistency", "WARNING", f"{title}: a shorter window holds more events than a longer one", cols, bad,
              recommendation="window counts must be nested; ask the DWH")
    monotone("Prior_Fall_Count_", ["30D", "90D", "180D", "365D"], "prior-fall counts")
    monotone("Visit_Count_", ["30D", "90D", "180D", "365D"], "visit counts")
    monotone("Diagnosis_Count_", ["180D", "365D"], "diagnosis counts")
    monotone("External_Care_Count_", ["30D", "90D", "180D", "365D"], "external-care counts")
    monotone("Hospitalization_Count_", ["180D", "365D"], "hospitalisation counts")
    if _has(f, "External_Care_Count_365D", "External_Care_Count_365D_Visible", "External_Care_Hidden_By_Billing_Lag_365D"):
        tot = _num(f, "External_Care_Count_365D")
        F.add("E02", "consistency", "INFO", "External_Care_Count_365D != Visible + Hidden_By_Billing_Lag (S2T identity)",
              ["External_Care_Count_365D", "External_Care_Count_365D_Visible", "External_Care_Hidden_By_Billing_Lag_365D"],
              (tot != _num(f, "External_Care_Count_365D_Visible") + _num(f, "External_Care_Hidden_By_Billing_Lag_365D")).fillna(False),
              recommendation="confirm the S2T identity")
    if _has(f, "Prior_Fall_Since_Study_Start_Ind", "Prior_Fall_Count_365D"):
        flag = _num(f, "Prior_Fall_Since_Study_Start_Ind")
        F.add("E03", "consistency", "WARNING", "no fall since 2022 but falls counted in the last 365 days", ["Prior_Fall_Since_Study_Start_Ind", "Prior_Fall_Count_365D"],
              ((flag == 0) & (_num(f, "Prior_Fall_Count_365D") > 0) & (index >= pd.Timestamp("2023-01-01"))).fillna(False),
              recommendation="the since-2022 flag must cover the 365-day window for index dates from 2023; ask the DWH")
    if _has(f, "Prior_Fall_Since_Study_Start_Ind", "Prior_Fall_Count_Since_Study_Start"):
        flag, cnt = _num(f, "Prior_Fall_Since_Study_Start_Ind"), _num(f, "Prior_Fall_Count_Since_Study_Start")
        F.add("E04", "consistency", "WARNING", "since-2022 fall flag disagrees with the since-2022 fall count",
              ["Prior_Fall_Since_Study_Start_Ind", "Prior_Fall_Count_Since_Study_Start"], (cnt.notna() & ((flag == 1) != (cnt > 0))).fillna(False),
              recommendation="the flag should equal count > 0")
    if "Prior_Fall_Missing_Ind" in f.columns:
        miss = _num(f, "Prior_Fall_Missing_Ind") == 1
        cols = [c for c in ("Prior_Fall_Since_Study_Start_Ind", "Prior_Fall_Count_365D") if c in f.columns]
        any_fall = pd.concat([_num(f, c) > 0 for c in cols], axis=1).any(axis=1) if cols else pd.Series(False, index=f.index)
        F.add("E05", "consistency", "WARNING", "fall source marked absent but falls recorded", ["Prior_Fall_Missing_Ind", *cols], (miss & any_fall).fillna(False),
              recommendation="source-absent flag contradicts the counts")
    if _has(f, "Last_Fall_Date", "Days_Since_Last_Fall"):
        F.add("E06", "consistency", "INFO", "Last_Fall_Date and Days_Since_Last_Fall are not NULL together", ["Last_Fall_Date", "Days_Since_Last_Fall"],
              f["Last_Fall_Date"].isna() != f["Days_Since_Last_Fall"].isna(), recommendation="both describe the same last fall")
    if _has(f, "Polypharmacy_5Plus_Ind", "Distinct_Active_Substance_Count"):
        cnt = _num(f, "Distinct_Active_Substance_Count")
        p5 = _num(f, "Polypharmacy_5Plus_Ind")
        F.add("E07", "consistency", "INFO", "Polypharmacy_5Plus_Ind disagrees with Distinct_Active_Substance_Count >= 5",
              ["Polypharmacy_5Plus_Ind", "Distinct_Active_Substance_Count"], (cnt.notna() & p5.notna() & ((p5 == 1) != (cnt >= 5))).fillna(False),
              recommendation="the flag may use another substance definition; confirm")
    if _has(f, "Polypharmacy_5Plus_Ind", "Polypharmacy_10Plus_Ind"):
        F.add("E08", "consistency", "WARNING", "10+ substances flagged without the 5+ flag", ["Polypharmacy_10Plus_Ind", "Polypharmacy_5Plus_Ind"],
              ((_num(f, "Polypharmacy_10Plus_Ind") == 1) & (_num(f, "Polypharmacy_5Plus_Ind") == 0)).fillna(False), recommendation="nested flags disagree")
    if _has(f, "Medication_Missing_Ind", "Distinct_Active_Substance_Count"):
        F.add("E09", "consistency", "WARNING", "medication source marked absent but substances counted", ["Medication_Missing_Ind", "Distinct_Active_Substance_Count"],
              ((_num(f, "Medication_Missing_Ind") == 1) & (_num(f, "Distinct_Active_Substance_Count") > 0)).fillna(False), recommendation="source flag contradicts the count")
    flags = [c for c in REGISTRY_FLAGS if c in f.columns]
    if flags and "Registry_Missing_Ind" in f.columns:
        anyreg = pd.concat([_num(f, c) == 1 for c in flags], axis=1).any(axis=1)
        F.add("E10", "consistency", "WARNING", "registry source marked absent but a registry flag is set", ["Registry_Missing_Ind", "Registry_*_Ind"],
              ((_num(f, "Registry_Missing_Ind") == 1) & anyreg).fillna(False), recommendation="source flag contradicts the registry flags")
    if flags and "Active_Registry_Count" in f.columns:
        nflags = pd.concat([(_num(f, c) == 1).astype(int) for c in flags], axis=1).sum(axis=1)
        F.add("E11", "consistency", "INFO", "Active_Registry_Count smaller than the number of registry flags set", ["Active_Registry_Count", "Registry_*_Ind"],
              (_num(f, "Active_Registry_Count") < nflags).fillna(False), recommendation="the count should cover at least the flagged registries")
    if _has(f, "Distinct_Registry_Type_Count", "Active_Registry_Count"):
        F.add("E12", "consistency", "INFO", "more distinct registry types than active registries", ["Distinct_Registry_Type_Count", "Active_Registry_Count"],
              (_num(f, "Distinct_Registry_Type_Count") > _num(f, "Active_Registry_Count")).fillna(False), recommendation="confirm both definitions")
    if _has(f, "MEFI_Group_At_Index", "Frailty_Not_Assessed_Ind"):
        g, na = f["MEFI_Group_At_Index"], _num(f, "Frailty_Not_Assessed_Ind")
        F.add("E13", "consistency", "WARNING", "MEFI group present although frailty 'not assessed' (or absent although assessed)",
              ["MEFI_Group_At_Index", "Frailty_Not_Assessed_Ind"], ((g.notna() & (na == 1)) | (g.isna() & (na == 0))).fillna(False),
              recommendation="the not-assessed flag should mirror the NULL group")
    if _has(f, "MEFI_Worsened_Ind", "MEFI_Prev_Group", "MEFI_Group_At_Index"):
        w, prev, cur = _num(f, "MEFI_Worsened_Ind"), _num(f, "MEFI_Prev_Group"), _num(f, "MEFI_Group_At_Index")
        F.add("E14", "consistency", "INFO", "MEFI_Worsened_Ind disagrees with current group > previous group", ["MEFI_Worsened_Ind", "MEFI_Prev_Group", "MEFI_Group_At_Index"],
              (w.notna() & prev.notna() & cur.notna() & ((w == 1) != (cur > prev))).fillna(False), recommendation="confirm the direction of 'worsened' (1 low .. 4 high)")
    for code, desc in (("Gender_Code", "Gender_Desc"), ("MEFI_Group_At_Index", "MEFI_Group_Desc")):
        if _has(f, code, desc):
            pairs = f[[code, desc]].dropna().astype("string").drop_duplicates()
            many = int((pairs.groupby(code)[desc].nunique() > 1).sum() + (pairs.groupby(desc)[code].nunique() > 1).sum())
            F.add("E15", "consistency", "WARNING", f"{code} and {desc} are not one-to-one", [code, desc], many, detail="count = codes or labels with several partners",
                  recommendation="code and label disagree; ask the DWH")
    nurse = [c for c in NURSE_FIELDS if c in f.columns]
    if nurse and "Assessment_Not_Performed_Ind" in f.columns:
        anyv = f[nurse].notna().any(axis=1)
        F.add("E16", "consistency", "WARNING", "nurse values present although 'assessment not performed'", ["Assessment_Not_Performed_Ind", *nurse],
              ((_num(f, "Assessment_Not_Performed_Ind") == 1) & anyv).fillna(False), recommendation="the not-performed flag contradicts the values")
    if _has(f, "Has_Repeat_Assessment_Ind", "Distinct_Assessment_Dates_365D"):
        F.add("E17", "consistency", "INFO", "repeat-assessment flag with fewer than 2 assessment dates in 365 days", ["Has_Repeat_Assessment_Ind", "Distinct_Assessment_Dates_365D"],
              ((_num(f, "Has_Repeat_Assessment_Ind") == 1) & (_num(f, "Distinct_Assessment_Dates_365D") < 2)).fillna(False),
              recommendation="the flag may use a longer window; confirm")
    if _has(f, "Home_Safety_Risk_Count", "Home_Safety_Questions_Answered"):
        F.add("E18", "consistency", "WARNING", "more home-safety risks than questions answered", ["Home_Safety_Risk_Count", "Home_Safety_Questions_Answered"],
              (_num(f, "Home_Safety_Risk_Count") > _num(f, "Home_Safety_Questions_Answered")).fillna(False), recommendation="impossible; ask the DWH")
    for flag, value_col in SOURCE_FLAGS.items():
        if _has(f, flag, value_col):
            v = f[value_col]
            has_value = (numeric(v) > 0) if not pd.api.types.is_string_dtype(v) else v.notna()
            F.add("E19", "consistency", "INFO", f"{flag} = 0 but {value_col} shows data", [flag, value_col], ((_num(f, flag) == 0) & has_value).fillna(False),
                  recommendation="coverage flag contradicts the source values")
    hs = [c for c in SOURCE_FLAGS if c in f.columns] + [c for c in ("Has_Source_Assessment_Ind",) if c in f.columns]
    if hs and "Distinct_Sources_With_Data" in f.columns:
        F.add("E20", "consistency", "INFO", "Distinct_Sources_With_Data differs from the sum of Has_Source_* flags", ["Distinct_Sources_With_Data", "Has_Source_*"],
              (_num(f, "Distinct_Sources_With_Data") != pd.concat([_num(f, c) for c in hs], axis=1).sum(axis=1)).fillna(False),
              recommendation="the count may include other sources; confirm")
    # ---------------- F. outcome contradictions
    for h in ("30D", "180D"):
        lab, ev, reason, dtn, hc = f"Fall_Next_{h}_Ind", f"Next_Fall_Date_{h}", f"Label_Reason_{h}", f"Days_To_Next_Fall_{h}", f"Fall_Next_{h}_HighConf_Ind"
        if _has(f, lab, ev):
            y = _num(f, lab)
            F.add("F01", "outcome", "CRITICAL", f"{lab} = 1 without an event date", [lab, ev], ((y == 1) & f[ev].isna()).fillna(False), recommendation="label defect")
            F.add("F02", "outcome", "CRITICAL", f"{lab} = 0 with an event date", [lab, ev], ((y == 0) & f[ev].notna()).fillna(False), recommendation="label defect")
        if _has(f, lab, reason):
            y, rs = _num(f, lab), f[reason].astype("string").str.upper()
            bad = (rs.str.startswith("POSITIVE") & (y != 1)) | (rs.str.startswith("NEGATIVE") & (y != 0)) | (rs.str.startswith("CENSORED") & y.notna())
            F.add("F03", "outcome", "WARNING", f"{reason} disagrees with {lab}", [reason, lab], bad.fillna(False), recommendation="label and its reason disagree")
        if _has(f, dtn, ev, idx_col):
            d = _num(f, dtn)
            both = d.notna() & f[ev].notna()
            F.add("F04", "outcome", "INFO", f"{dtn} differs from {ev} - Index_Date", [dtn, ev], both & (d != (f[ev].dt.normalize() - index).dt.days),
                  denominator=int(both.sum()), recommendation="two derivations of the event time disagree")
        if _has(f, hc, lab):
            F.add("F05", "outcome", "WARNING", f"high-confidence label = 1 while {lab} = 0", [hc, lab], ((_num(f, hc) == 1) & (_num(f, lab) == 0)).fillna(False),
                  recommendation="a high-confidence event must also be an event")
    if _has(f, "Fall_Next_30D_Ind", "Fall_Next_180D_Ind"):
        F.add("F06", "outcome", "CRITICAL", "30-day event but no 180-day event", ["Fall_Next_30D_Ind", "Fall_Next_180D_Ind"],
              ((_num(f, "Fall_Next_30D_Ind") == 1) & (_num(f, "Fall_Next_180D_Ind") == 0)).fillna(False), recommendation="nested windows disagree")
    if _has(f, "Is_Censored_180D", "Fall_Next_180D_Ind"):
        F.add("F07", "outcome", "INFO", "censored within 180 days but a 180-day label is present", ["Is_Censored_180D", "Fall_Next_180D_Ind"],
              ((_num(f, "Is_Censored_180D") == 1) & f["Fall_Next_180D_Ind"].notna()).fillna(False),
              recommendation="positives observed before censoring can be labelled; negatives cannot - check Label_Reason_180D")
    if _has(f, "Fall_On_Index_Date_Ind", "Fall_Next_180D_Ind"):
        F.add("F08", "outcome", "WARNING", "fall on the index day but the 180-day label (window includes the index day) is 0",
              ["Fall_On_Index_Date_Ind", "Fall_Next_180D_Ind"], ((_num(f, "Fall_On_Index_Date_Ind") == 1) & (_num(f, "Fall_Next_180D_Ind") == 0)).fillna(False),
              recommendation="the index-day event should be in the outcome window")
    # ---------------- G. keys and provenance
    if _has(f, id_col, idx_col):
        F.add("G01", "keys", "CRITICAL", "duplicated patient x Index_Date rows", [id_col, idx_col], f.duplicated(subset=[id_col, idx_col], keep="first"),
              recommendation="one row per patient and snapshot is required (the build stops)")
        per = f.groupby(f[id_col])[idx_col].nunique()
        F.add("G02", "keys", "INFO", "patients with rows on several snapshots (repeated measures)", id_col, int((per > 1).sum()), denominator=int(len(per)),
              detail=f"{int(len(per))} distinct patients", recommendation="patient-grouped splitting is required (already enforced)")
    if "Snapshot_Key" in f.columns:
        F.add("G03", "keys", "CRITICAL", "duplicated Snapshot_Key", "Snapshot_Key", f["Snapshot_Key"].notna() & f["Snapshot_Key"].duplicated(keep="first"),
              recommendation="technical key must be unique")
    if "Definition_Version" in f.columns:
        F.add("G04", "provenance", "CRITICAL", "Definition_Version is not constant V1.0", "Definition_Version", (f["Definition_Version"].astype("string") != "V1.0").fillna(True),
              recommendation="mixed VIEW versions in one file")
    if "Leakage_Check_Ind" in f.columns:
        F.add("G05", "provenance", "CRITICAL", "Leakage_Check_Ind != 0 (VIEW self-check failed)", "Leakage_Check_Ind", (_num(f, "Leakage_Check_Ind") != 0).fillna(False),
              recommendation="VIEW bug: the build stops")
    if "Multiple_Valid_Population_Rows_Ind" in f.columns:
        F.add("G06", "provenance", "WARNING", "several population SCD rows valid at Index_Date", "Multiple_Valid_Population_Rows_Ind",
              _num(f, "Multiple_Valid_Population_Rows_Ind") == 1, recommendation="SCD duplication: confirm which row the VIEW used")
    if "Snapshot_Confidence" in f.columns:
        F.add("G07", "provenance", "INFO", "rows with Snapshot_Confidence = LOW", "Snapshot_Confidence", f["Snapshot_Confidence"].astype("string") == "LOW",
              recommendation="rule not documented; report by confidence level")
    if "Unsettled_Visit_Counter_Ind" in f.columns:
        F.add("G08", "provenance", "INFO", "visit counters inside the 3-day settlement period", "Unsettled_Visit_Counter_Ind", _num(f, "Unsettled_Visit_Counter_Ind") == 1,
              recommendation="train/serve skew: production counters may differ")
    out = F.frame()
    sev = {s: i for i, s in enumerate(SEVERITIES)}
    out["_s"] = out["severity"].map(sev).fillna(9)
    out["_st"] = out["status"].map({"FINDING": 0, "NOT_APPLICABLE": 1, "PASS": 2})
    return out.sort_values(["_st", "_s", "check_id"]).drop(columns=["_s", "_st"]).reset_index(drop=True)
