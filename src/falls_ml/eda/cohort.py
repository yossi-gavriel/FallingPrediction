"""Population / cohort EDA: the cohort funnel, the population profile and the descriptive outcome prevalence.

The funnel is computed from the extract with the same rules the build applies, and cross-checked against the build report of the
actual canonical build (the two must agree; a disagreement is reported, never hidden). Outcome prevalence by stratum is POST-HOC
DESCRIPTIVE on the whole modelling cohort (test rows included): it describes the population and is never used for model choice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from falls_ml.eda.common import (ELIGIBLE_LABELLED, FULL_EXTRACT, MODELLING_COHORT, POSTHOC_DESCRIPTIVE, SUPPRESSED, cell, fnum, numeric,
                                 quantiles, small, wilson)

AGE_BINS = [-np.inf, 65, 70, 75, 80, 85, 90, 95, np.inf]
AGE_LABELS = ["<65", "65-69", "70-74", "75-79", "80-84", "85-89", "90-94", "95+"]
SENIORITY_BINS = [-np.inf, 12, 60, 120, 240, np.inf]
SENIORITY_LABELS = ["<1 year", "1-5 years", "5-10 years", "10-20 years", "20+ years"]
SUBSTANCE_BINS = [-np.inf, 0.5, 4.5, 9.5, np.inf]
SUBSTANCE_LABELS = ["0", "1-4", "5-9", "10+"]


@dataclass
class CohortMasks:
    """Row masks over the raw extract (aligned to its index)."""

    on_date: pd.Series
    eligible: pd.Series        # eligible on the index date
    labelled: pd.Series        # eligible on the index date with a usable 180-day label
    d00: pd.Series             # labelled rows whose cohort guard dates are on/after the index day
    evidence: pd.Series        # labelled rows whose same-day evidence (First_Registry_Date) is on/after the index day
    cohort: pd.Series          # modelling cohort (rows with a partition)
    safe: pd.Series            # SAFE-ALL-ROWS rows (labelled minus evidence rows)
    partition: pd.Series       # train / validation / test / <NA>


def age_band(s: pd.Series) -> pd.Series:
    return pd.cut(numeric(s), AGE_BINS, right=False, labels=AGE_LABELS).astype("string")


def cohort_flow(frame: pd.DataFrame, masks: CohortMasks, *, id_col: str, label_col: str, index_date: str, policy: str,
                guard_cols: list[str], evidence_cols: list[str], build: Any | None, safe_build: Any | None, min_cell: int) -> pd.DataFrame:
    """``cohort_flow.csv``: every funnel step with rows, patients, events (where the label is usable) and the rows removed."""
    y = numeric(frame[label_col])

    def step(k: int, name: str, mask: pd.Series, removed: int | None, why: str, *, labelled: bool = False, branch: str = "PRIMARY") -> dict[str, Any]:
        n = int(mask.sum())
        ev = int((y[mask] == 1).sum()) if labelled else None
        return {"step": k, "branch": branch, "stage": name, "rows": n, "patients": int(frame.loc[mask, id_col].nunique()),
                "events": "" if ev is None else cell(ev, min_cell), "prevalence": fnum(ev / n, 4) if (labelled and n and not small(ev, min_cell)) else None,
                "rows_removed_at_this_step": None if removed is None else cell(removed, min_cell), "reason": why}

    all_rows = pd.Series(True, index=frame.index)
    rows = [step(0, "rows in the extract file", all_rows, None, f"{frame[id_col].nunique()} distinct patients; "
                 f"{frame['Index_Date'].dt.normalize().nunique() if 'Index_Date' in frame.columns else '?'} snapshots")]
    rows.append(step(1, f"rows on the index date {index_date}", masks.on_date, int((~masks.on_date).sum()), "other snapshots are not modelled in this run"))
    excl = frame.loc[masks.on_date & ~masks.eligible, "Exclusion_Reason"].astype("string").fillna("NULL").value_counts() if "Exclusion_Reason" in frame.columns else pd.Series(dtype=int)
    rows.append(step(2, "eligible (Is_Eligible_Cohort = 1)", masks.eligible, int((masks.on_date & ~masks.eligible).sum()),
                     "; ".join(f"{k}: {cell(v, min_cell)}" for k, v in excl.items()) or "none"))
    reasons = frame.loc[masks.eligible & ~masks.labelled, "Label_Reason_180D"].astype("string").fillna("NULL").value_counts() if "Label_Reason_180D" in frame.columns else pd.Series(dtype=int)
    rows.append(step(3, "usable 180-day label (Fall_Next_180D_Ind not NULL)", masks.labelled, int((masks.eligible & ~masks.labelled).sum()),
                     "no usable label: " + ("; ".join(f"{k}: {cell(v, min_cell)}" for k, v in reasons.items()) or "none"), labelled=True))
    d00_by = {c: int((masks.labelled & (frame[c].dt.normalize() >= frame["Index_Date"].dt.normalize())).sum()) for c in guard_cols if c in frame.columns}
    if policy == "drop_rows":
        rows.append(step(4, "D-00: rows with a cohort guard date on/after the index day excluded", masks.labelled & ~masks.d00, int(masks.d00.sum()),
                         "start-of-index-day rule (source_event_date < Index_Date); by column: " + "; ".join(f"{k}: {cell(v, min_cell)}" for k, v in d00_by.items()),
                         labelled=True))
    else:
        rows.append(step(4, f"D-00 policy {policy}: {cell(int(masks.d00.sum()), min_cell)} violating rows STOP the build", masks.labelled & ~masks.d00,
                         int(masks.d00.sum()), "rerun with the reference run (--reference) or --index-day-records drop_rows to build the cohort", labelled=True))
    rows.append(step(5, "modelling cohort (train + validation + test)", masks.cohort, None, "the split is drawn on these rows", labelled=True))
    for k, part in enumerate(("train", "validation", "test"), start=6):
        rows.append(step(k, f"  {part} partition", (masks.partition == part).fillna(False), None, "patient-grouped, stratified split (reference config)", labelled=True))
    ev_by = {c: int((masks.labelled & (frame[c].dt.normalize() >= frame["Index_Date"].dt.normalize())).sum()) for c in evidence_cols if c in frame.columns}
    rows.append(step(4, "SAFE-ALL-ROWS: rows with same-day registry evidence excluded (sensitivity branch)", masks.safe, int(masks.evidence.sum()),
                     "no exclusion for Last_Dx_Date / Last_Fall_Date (their predictors are removed instead); evidence by column: "
                     + ("; ".join(f"{k}: {cell(v, min_cell)}" for k, v in ev_by.items()) or "none"), labelled=True, branch="SAFE_ALL_ROWS"))
    out = pd.DataFrame(rows)
    checks = []
    if build is not None:
        checks.append(("modelling cohort", int(masks.cohort.sum()), int(build.n_rows_final)))
    if safe_build is not None:
        checks.append(("SAFE-ALL-ROWS", int(masks.safe.sum()), int(safe_build.n_rows_final)))
    out.attrs["build_crosscheck"] = [{"what": w, "eda_rows": a, "build_rows": b, "agree": a == b} for w, a, b in checks]
    return out


def population_profile(frame: pd.DataFrame, masks: CohortMasks, *, label_col: str, min_cell: int) -> pd.DataFrame:
    """Long table of population distributions (section, basis, variable, level, n, pct) for the three population bases."""
    bases = {FULL_EXTRACT: pd.Series(True, index=frame.index), ELIGIBLE_LABELLED: masks.labelled, MODELLING_COHORT: masks.cohort}
    rows: list[dict[str, Any]] = []

    def dist(section: str, variable: str, s: pd.Series, basis: str, mask: pd.Series) -> None:
        n = int(mask.sum())
        vc = s[mask].astype("string").fillna("NULL").value_counts().sort_index()
        for level, k in vc.items():
            kk = cell(k, min_cell)
            rows.append({"section": section, "basis": basis, "variable": variable, "level": str(level), "n": kk,
                         "pct": fnum(100.0 * k / n, 2) if n and kk != SUPPRESSED else (SUPPRESSED if kk == SUPPRESSED else None)})

    def quant(section: str, variable: str, s: pd.Series, basis: str, mask: pd.Series) -> None:
        v = numeric(s[mask])
        q = quantiles(v)
        for k, val in {"n_observed": int(v.notna().sum()), "mean": fnum(v.mean(), 3), **q}.items():
            rows.append({"section": section, "basis": basis, "variable": variable, "level": k, "n": None, "pct": None, "value": val})

    for basis, mask in bases.items():
        if "Age_At_Index" in frame.columns:
            dist("age", "age band", age_band(frame["Age_At_Index"]), basis, mask)
            quant("age", "Age_At_Index", frame["Age_At_Index"], basis, mask)
        if "Gender_Code" in frame.columns:
            dist("sex", "Gender_Code", frame["Gender_Code"], basis, mask)
        if "Birth_Date_Suspect_Ind" in frame.columns:
            dist("suspect birth dates", "Birth_Date_Suspect_Ind", frame["Birth_Date_Suspect_Ind"], basis, mask)
        if "HMO_Seniority_Months" in frame.columns:
            dist("HMO seniority", "seniority band", pd.cut(numeric(frame["HMO_Seniority_Months"]), SENIORITY_BINS, right=False,
                                                           labels=SENIORITY_LABELS).astype("string"), basis, mask)
            quant("HMO seniority", "HMO_Seniority_Months", frame["HMO_Seniority_Months"], basis, mask)
        for col in ("Has_Full_365D_Lookback", "Has_Minimum_History"):
            if col in frame.columns:
                dist("history availability", col, frame[col], basis, mask)
        for col in ("Available_History_Days", "Observed_History_Days"):
            if col in frame.columns:
                quant("history availability", col, frame[col], basis, mask)
    on = masks.on_date
    elig = masks.eligible
    for col, section in (("Is_Eligible_Cohort", "eligibility"), ("Exclusion_Reason", "exclusions"), ("Index_Date_Status", "eligibility"),
                         ("Is_Active_For_Modeling", "eligibility")):
        if col in frame.columns:
            dist(section, col, frame[col], "ON_INDEX_DATE", on)
    for col, section in (("Label_Reason_180D", "labels"), ("Fall_Next_180D_Ind", "labels"), ("Is_Censored_180D", "censoring"),
                         ("Has_Full_Followup_180D", "censoring"), ("Followup_End_Reason", "censoring"), ("Is_Deceased_Ind", "censoring"),
                         ("Death_Identification_Status", "censoring")):
        if col in frame.columns:
            dist(section, col, frame[col], "ELIGIBLE_ON_INDEX_DATE", elig)
    if {"Followup_End_Date", "Index_Date", "Label_End_180D"} <= set(frame.columns):
        end = frame["Followup_End_Date"].where(frame["Followup_End_Date"] < pd.Timestamp("2100-01-01"))
        stop = pd.concat([end.dt.normalize(), frame["Label_End_180D"].dt.normalize()], axis=1).min(axis=1)
        days = (stop - frame["Index_Date"].dt.normalize()).dt.days
        quant("follow-up", "days of follow-up within the 180-day window (min of follow-up end, label end)", days, "ELIGIBLE_ON_INDEX_DATE", elig)
    return pd.DataFrame(rows)


def outcome_prevalence(frame: pd.DataFrame, masks: CohortMasks, *, label_col: str, min_cell: int) -> pd.DataFrame:
    """``outcome_prevalence.csv``: prevalence (Wilson 95% CI) overall and by stratum. Strata on the modelling cohort are POST-HOC
    DESCRIPTIVE (test rows included) - they describe the population; they are not an input to any modelling decision."""
    y = numeric(frame[label_col])
    rows: list[dict[str, Any]] = []

    def add(basis: str, stratifier: str, level: str, mask: pd.Series) -> None:
        m = mask.fillna(False).astype(bool)
        n = int(m.sum())
        k = int((y[m] == 1).sum())
        sup = small(n, min_cell) or small(k, min_cell) or small(n - k, min_cell)
        p, lo, hi = wilson(k, n)
        rows.append({"basis": basis, "stratifier": stratifier, "level": level, "n": cell(n, min_cell), "events": cell(k, min_cell),
                     "non_events": cell(n - k, min_cell), "prevalence": SUPPRESSED if sup else fnum(p, 4),
                     "ci_low": None if sup else fnum(lo, 4), "ci_high": None if sup else fnum(hi, 4)})

    add(ELIGIBLE_LABELLED, "overall", "all eligible rows with a usable label", masks.labelled)
    add(ELIGIBLE_LABELLED, "D-00 status", "clean (kept in the modelling cohort)", masks.labelled & ~masks.d00)
    add(ELIGIBLE_LABELLED, "D-00 status", "cohort guard date on/after the index day (excluded)", masks.d00)
    add(ELIGIBLE_LABELLED, "SAFE-ALL-ROWS", "SAFE-ALL-ROWS rows", masks.safe)
    add(MODELLING_COHORT, "overall", "modelling cohort", masks.cohort)
    for part in ("train", "validation", "test"):
        add(MODELLING_COHORT, "partition", part, (masks.partition == part).fillna(False))
    c = masks.cohort
    strata: list[tuple[str, pd.Series]] = []
    if "Age_At_Index" in frame.columns:
        strata.append(("age band", age_band(frame["Age_At_Index"])))
    if "Gender_Code" in frame.columns:
        strata.append(("sex (Gender_Code)", frame["Gender_Code"].astype("string")))
    for col, name in (("Prior_Fall_Since_Study_Start_Ind", "prior fall since 2022 [falls]"), ("Gait_Disorder_Since_Study_Start_Ind", "difficulty walking 719.7 [mobility_problems]"),
                      ("Mobility_Score", "nurse mobility score (NULL = not assessed)"), ("Uses_Walking_Aid_Ind", "uses walking aid (NULL = not assessed)"),
                      ("Registry_Home_Confined_Ind", "housebound registry"), ("MEFI_Group_At_Index", "MEFI frailty group (NULL = not assessed)"),
                      ("Birth_Date_Suspect_Ind", "suspect birth date")):
        if col in frame.columns:
            strata.append((name, frame[col].astype("string")))
    if "Prior_Fall_Count_365D" in frame.columns:
        strata.append(("fall/fracture events in 365 days", pd.cut(numeric(frame["Prior_Fall_Count_365D"]), [-np.inf, 0.5, 1.5, np.inf], labels=["0", "1", "2+"]).astype("string")))
    if "Distinct_Active_Substance_Count" in frame.columns:
        strata.append(("distinct active substances (medication burden)", pd.cut(numeric(frame["Distinct_Active_Substance_Count"]), SUBSTANCE_BINS,
                                                                                labels=SUBSTANCE_LABELS).astype("string")))
    for stratifier, s in strata:
        for level in sorted(s[c].fillna("NULL").unique(), key=str):
            add(POSTHOC_DESCRIPTIVE, stratifier, str(level), c & (s.fillna("NULL") == level))
    registries = [("Registry_Dementia_Ind", "dementia"), ("Registry_Home_Confined_Ind", "home-confined"), ("Registry_Chronic_Renal_Failure_Ind", "chronic renal failure"),
                  ("Registry_COPD_Ind", "COPD"), ("Registry_Asthma_Ind", "asthma"), ("Registry_Blood_Pressure_Ind", "hypertension"),
                  ("Registry_Liver_Ind", "liver"), ("Registry_Heart_Disease_Ind", "heart disease"), ("Registry_Suicide_Attempt_Ind", "suicide attempt"),
                  ("Registry_Kidneys_Ind", "kidneys"), ("Registry_Transplant_Ind", "transplant"), ("Registry_Stoma_Ind", "stoma")]
    if {"Registry_Diabetes_T1_Ind", "Registry_Diabetes_T2_Ind"} <= set(frame.columns):
        dia = (numeric(frame["Registry_Diabetes_T1_Ind"]).fillna(0) > 0) | (numeric(frame["Registry_Diabetes_T2_Ind"]).fillna(0) > 0)
        add(POSTHOC_DESCRIPTIVE, "registry: diabetes (T1 or T2)", "in registry", c & dia)
        add(POSTHOC_DESCRIPTIVE, "registry: diabetes (T1 or T2)", "not in registry", c & ~dia)
    if "Registry_SMI_Level" in frame.columns:
        smi = numeric(frame["Registry_SMI_Level"]).fillna(0) >= 1
        add(POSTHOC_DESCRIPTIVE, "registry: SMI level >= 1", "in registry", c & smi)
        add(POSTHOC_DESCRIPTIVE, "registry: SMI level >= 1", "not in registry", c & ~smi)
    for col, name in registries:
        if col in frame.columns:
            flag = numeric(frame[col]) == 1
            add(POSTHOC_DESCRIPTIVE, f"registry: {name}", "in registry", c & flag)
            add(POSTHOC_DESCRIPTIVE, f"registry: {name}", "not in registry", c & ~flag)
    return pd.DataFrame(rows)
