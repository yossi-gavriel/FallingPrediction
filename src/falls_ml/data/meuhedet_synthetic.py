"""SYNTHETIC wide-table extract with the 221 columns of V_Falls_Prediction_Wide_1 (software tests and demos only).

Every column is generated from the contract (type, allowed values, NULL semantics); a handful of columns carry a planted
signal so the exploratory pipeline has something to learn. Nothing here resembles real Meuhedet distributions:
SYNTHETIC DATA - NOT SCIENTIFIC RESULTS.

``generate_synthetic_wide_extract`` writes one snapshot (plus a few rows on an older one); ``generate_synthetic_wide_panel``
writes a monthly panel: the same patients on every Index_Date (static traits fixed, time-varying blocks redrawn, "since study
start" flags monotone), patient churn, labels censored past a data-freeze date, and planted timing defects (same-day and
true-future diagnosis / fall records) for the D-00 tests.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from falls_ml.data.meuhedet_wide import DEFAULT_CONTRACT, WideContract, load_wide_contract

NAT = np.datetime64("NaT", "us")
EXCLUSIONS = ("INSIDE_BURN_IN_PERIOD", "AGE_BELOW_65", "INSUFFICIENT_HISTORY", "NOT_ACTIVE")


def _dates(rng: np.random.Generator, lo: pd.Timestamp, hi: pd.Timestamp, n: int) -> np.ndarray:
    span = max(int((hi - lo).days), 1)
    return (lo + pd.to_timedelta(rng.integers(0, span, n), unit="D")).to_numpy(dtype="datetime64[us]")


def _mask(values: np.ndarray, present: np.ndarray, dtype: str = "Int64") -> pd.Series:
    s = pd.Series(values)
    if dtype == "Int64":
        s = s.astype("Int64")
    return s.where(pd.Series(present), other=pd.NA if dtype == "Int64" else np.nan)


def generate_synthetic_wide_extract(n_rows: int = 3000, *, seed: int = 20260917, index_date: str = "2025-01-01",
                                    contract: WideContract | None = None, other_index_date_share: float = 0.1,
                                    n_index_day_falls: int = 0) -> pd.DataFrame:
    """A deterministic 221-column extract: ~85% eligible rows on ``index_date`` (+ some rows on another snapshot), planted
    outcome signal (age, prior falls, dementia, mobility, walking aid, MEFI), assessment-conditional nurse blocks (NULL =
    not assessed), source-missing indicators, censored labels and QA columns."""
    contract = contract or load_wide_contract(DEFAULT_CONTRACT)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)  # 221 column inserts; copied once at the end
        return _generate(int(n_rows), np.random.default_rng(seed), pd.Timestamp(index_date), contract, other_index_date_share, n_index_day_falls)


def _generate(n: int, rng: np.random.Generator, idx: pd.Timestamp, contract: WideContract, other_index_date_share: float,
              n_index_day_falls: int, *, patients: dict[str, Any] | None = None, months: int = 0, state: dict[str, np.ndarray] | None = None,
              n_index_day_dx: int = 0, n_future_dx: int = 0, n_future_falls: int = 0, present: np.ndarray | None = None) -> pd.DataFrame:
    """One snapshot. With ``patients`` (see :func:`_draw_patients`) the static traits come from that table (``months`` after its
    first snapshot) and ``state`` carries the monotone "since study start" flags between snapshots; without it every draw is
    fresh (the single-snapshot stream is unchanged). ``n_index_day_dx`` / ``n_future_dx`` / ``n_future_falls`` plant timing defects:
    a last diagnosis (fall) dated on the index day, or strictly after it (Days_Since_* negative, as an ETL bug would produce)."""
    alt = idx - pd.DateOffset(years=1)
    on_alt = rng.random(n) < other_index_date_share
    fixed = patients is not None
    plantable = ~on_alt if present is None else (~on_alt & np.asarray(present, dtype=bool))   # rows that stay in the output
    index_dates = np.where(on_alt, np.datetime64(alt, "us"), np.datetime64(idx, "us")).astype("datetime64[us]")
    row_idx = pd.Series(index_dates)

    df = pd.DataFrame({"Customer_Full_ID": [f"S{i:010d}" for i in range(n)], "Index_Date": index_dates})
    # "SYN_" marks synthetic rows: the adapter records such datasets as synthetic_fixture (SYNTHETIC DATA - NOT SCIENTIFIC RESULTS)
    df["Snapshot_Key"] = [f"SYN_{pd.Timestamp(d).strftime('%Y%m%d')}_{i:07d}" for i, d in enumerate(index_dates)]
    df["Customer_Id_Type_Prefix"] = "1"
    df["Definition_Version"] = "V1.0"
    hist = patients["hist0"] + months * 30 if fixed else rng.integers(200, 12000, n)
    df["Observed_History_Days"] = hist
    df["Has_Full_365D_Lookback"] = (hist >= 365).astype(int)
    df["Has_Full_30D_Label"] = 1
    df["Has_Full_180D_Label"] = 1
    age = patients["age0"] + months // 12 if fixed else np.clip(rng.normal(76, 7.5, n).round(), 62, 101).astype(int)
    eligible = (rng.random(n) < 0.9) & (age >= 65) & (hist >= 365)
    df["Is_Eligible_Cohort"] = eligible.astype(int)
    reasons = np.where(age < 65, "AGE_BELOW_65", np.where(hist < 365, "INSUFFICIENT_HISTORY",
                                                          rng.choice(EXCLUSIONS, n)))
    df["Exclusion_Reason"] = pd.Series(np.where(eligible, None, reasons), dtype="string")
    df["Population_Record_From_Date"] = patients["from_date"] if fixed else _dates(rng, idx - pd.DateOffset(years=20), idx - pd.DateOffset(years=1), n)
    # SCD validity end: the open-ended sentinel 2999-12-31 (declared in the contract) for current rows, NULL for a few, a real
    # closing date after the index for closed rows - exercises the full SQL date range end to end
    to_kind = rng.random(n)
    closed = _dates(rng, idx + pd.Timedelta(days=1), idx + pd.Timedelta(days=400), n)
    df["Population_Record_To_Date"] = pd.Series(np.where(to_kind < 0.90, np.datetime64("2999-12-31", "us"), np.where(to_kind < 0.95, NAT, closed)),
                                                dtype="datetime64[us]")
    df["Population_Current_Ind"] = (to_kind < 0.95).astype(int)
    df["Population_Record_Count"] = 1
    df["Multiple_Valid_Population_Rows_Ind"] = 0
    df["Obs_Window_Start"] = (row_idx - pd.DateOffset(years=1)).to_numpy()
    df["Label_End_30D"] = (row_idx + pd.Timedelta(days=30)).to_numpy()
    df["Label_End_180D"] = (row_idx + pd.Timedelta(days=180)).to_numpy()
    df["Is_Active_For_Modeling"] = eligible.astype(int)
    df["Index_Date_Status"] = np.where(eligible, "ACTIVE", "INSIDE_BURN_IN_PERIOD")
    df["Age_At_Index"] = age
    df["Age_Group"] = pd.cut(age, [0, 64, 69, 74, 79, 84, 89, 200], labels=["<65", "65-69", "70-74", "75-79", "80-84", "85-89", "90+"]).astype(str)
    df["Birth_Date_Suspect_Ind"] = patients["birth_suspect"] if fixed else (rng.random(n) < 0.1).astype(int)
    sex = patients["sex"] if fixed else rng.choice([1, 2], n, p=[0.45, 0.55])
    df["Gender_Code"] = sex
    df["Gender_Desc"] = np.where(sex == 1, "זכר", "נקבה")
    df["Available_History_Days"] = hist
    df["Has_Minimum_History"] = (hist >= 365).astype(int)
    df["HMO_Seniority_Months"] = (hist // 30).astype(int)

    # ---- planted risk structure
    prior_fall_365 = rng.poisson(0.12, n)
    if state is not None:   # monotone: a fall since study start stays on record in every later snapshot
        state["fall_ever"] = state["fall_ever"] | (prior_fall_365 > 0) | (rng.random(n) < 0.02)
        prior_since_start = state["fall_ever"].astype(int)
    else:
        prior_since_start = ((prior_fall_365 > 0) | (rng.random(n) < 0.08)).astype(int)
    dementia = patients["dementia"] if fixed else (rng.random(n) < 0.06 + 0.004 * np.clip(age - 65, 0, 35)).astype(int)
    assessed = rng.random(n) < 0.3
    mobility = np.where(assessed, rng.choice([0, 1, 2, 3], n, p=[0.55, 0.25, 0.12, 0.08]), 0)
    walking_aid = np.where(assessed, (rng.random(n) < 0.35).astype(int), 0)
    mefi_assessed = patients["mefi_assessed"] if fixed else rng.random(n) < 0.6
    mefi = patients["mefi"] if fixed else np.where(mefi_assessed, rng.choice([1, 2, 3, 4], n, p=[0.45, 0.3, 0.17, 0.08]), 0)
    lp = (-3.6 + 0.045 * (age - 75) + 0.9 * (prior_fall_365 > 0) + 0.5 * prior_since_start + 0.6 * dementia
          + 0.35 * mobility + 0.4 * walking_aid + 0.3 * np.clip(mefi - 1, 0, 3) + 0.2 * (sex == 2))
    p180 = 1 / (1 + np.exp(-lp))
    y180 = rng.random(n) < p180
    y30 = y180 & (rng.random(n) < 0.25)
    censored = rng.random(n) < 0.03
    deceased = censored & (rng.random(n) < 0.5)

    # follow-up end: a real date for censored rows; for open follow-up the source uses either NULL or the declared 2999-12-31 sentinel
    open_sentinel = ~censored & (rng.random(n) < 0.6)
    df["Followup_End_Date"] = pd.Series(np.where(censored, _dates(rng, idx + pd.Timedelta(days=1), idx + pd.Timedelta(days=170), n),
                                                 np.where(open_sentinel, np.datetime64("2999-12-31", "us"), NAT)), dtype="datetime64[us]")
    df["Followup_End_Reason"] = np.where(deceased, "DEATH", np.where(censored, "LEFT_HMO", "STATUS_END"))
    df["Has_Full_Followup_30D"] = 1
    df["Has_Full_Followup_180D"] = (~censored).astype(int)
    df["Is_Censored_30D"] = 0
    df["Is_Censored_180D"] = censored.astype(int)
    df["Is_Deceased_Ind"] = deceased.astype(int)
    df["Leave_Join_Ind"] = _mask(np.where(censored & ~deceased, 5, -9), np.ones(n, bool))
    df["Death_Identification_Status"] = np.where(deceased, "DECEASED", "NOT_DECEASED")
    df["Death_Censor_Date"] = df["Followup_End_Date"].where(pd.Series(deceased))
    df["Death_Date_Gap_Days"] = _mask(np.zeros(n, int), deceased)
    df["Abroad_Ind"] = _mask((rng.random(n) < 0.01).astype(int), np.ones(n, bool))

    # ---- prior falls
    fall_missing = rng.random(n) < 0.02
    df["Prior_Fall_Count_30D"] = np.minimum(prior_fall_365, rng.binomial(1, 0.15, n))
    df["Prior_Fall_Count_90D"] = np.minimum(prior_fall_365, rng.binomial(1, 0.3, n) + df["Prior_Fall_Count_30D"])
    df["Prior_Fall_Count_180D"] = np.minimum(prior_fall_365, df["Prior_Fall_Count_90D"] + rng.binomial(1, 0.3, n))
    df["Prior_Fall_Count_365D"] = prior_fall_365
    df["Prior_Fall_Since_Study_Start_Ind"] = prior_since_start
    df["Prior_Fall_Count_Since_Study_Start"] = _mask(prior_since_start * (prior_fall_365 + rng.poisson(0.3, n)), ~fall_missing)
    had_fall = prior_since_start == 1
    last_fall = np.where(had_fall, _dates(rng, idx - pd.DateOffset(years=3), idx - pd.Timedelta(days=1), n), NAT)
    if n_index_day_falls:
        pick = np.flatnonzero(eligible & plantable & ~censored & ~fall_missing)[:n_index_day_falls]  # rows that reach the timing check
        last_fall[pick] = np.datetime64(idx, "us")
        had_fall[pick] = True
    if n_future_falls:   # true future leakage: the "last fall" lies after the index day (an ETL/snapshot defect, never a business rule)
        pick = np.flatnonzero(eligible & plantable & ~censored & ~fall_missing)[n_index_day_falls:n_index_day_falls + n_future_falls]
        last_fall[pick] = (idx + pd.to_timedelta(rng.integers(1, 40, len(pick)), unit="D")).to_numpy(dtype="datetime64[us]")
        had_fall[pick] = True
    df["Last_Fall_Date"] = pd.Series(last_fall, dtype="datetime64[us]")
    df["Days_Since_Last_Fall"] = _mask((row_idx - pd.Series(last_fall)).dt.days.to_numpy(na_value=0), had_fall)
    df["Last_Fall_Confidence"] = pd.Series(np.where(had_fall, rng.choice(["HIGH", "MEDIUM", "LOW"], n, p=[0.7, 0.2, 0.1]), None), dtype="string")
    df["Fall_On_Index_Date_Ind"] = pd.Series(last_fall).dt.normalize().eq(row_idx).fillna(False).astype(int).to_numpy()
    df["Prior_Fall_Missing_Ind"] = fall_missing.astype(int)
    df.loc[fall_missing, ["Prior_Fall_Count_30D", "Prior_Fall_Count_90D", "Prior_Fall_Count_180D", "Prior_Fall_Count_365D",
                          "Prior_Fall_Since_Study_Start_Ind", "Fall_On_Index_Date_Ind"]] = 0

    # ---- diagnoses / visits
    dx = rng.poisson(9, n)
    df["Diagnosis_Count_180D"] = rng.binomial(dx, 0.55)
    df["Diagnosis_Count_365D"] = dx
    df["Distinct_Diagnosis_Codes_365D"] = rng.binomial(dx, 0.7)
    df["Chronic_Diagnosis_Count_365D"] = rng.binomial(dx, 0.3)
    gait = (rng.random(n) < 0.03) | ((mobility >= 2) & (rng.random(n) < 0.3))
    if state is not None:
        state["gait_ever"] = state["gait_ever"] | gait
        gait = state["gait_ever"]
    df["Gait_Disorder_Since_Study_Start_Ind"] = gait.astype(int)
    has_dx = dx > 0
    last_dx = np.where(has_dx, _dates(rng, idx - pd.DateOffset(years=1), idx - pd.Timedelta(days=1), n), NAT)
    if n_index_day_dx or n_future_dx:
        candidates = np.flatnonzero(eligible & plantable & ~censored & has_dx)
        same = candidates[:n_index_day_dx]
        future = candidates[n_index_day_dx:n_index_day_dx + n_future_dx]
        last_dx[same] = np.datetime64(idx, "us")   # same-day inclusion (window rule <= Index_Date)
        last_dx[future] = (idx + pd.to_timedelta(rng.integers(1, 40, len(future)), unit="D")).to_numpy(dtype="datetime64[us]")
    df["Last_Dx_Date"] = pd.Series(last_dx, dtype="datetime64[us]")
    df["Days_Since_Last_Diagnosis"] = _mask((row_idx - pd.Series(last_dx)).dt.days.to_numpy(na_value=0), has_dx)
    visits = rng.poisson(12, n)
    df["Visit_Count_30D"] = rng.binomial(visits, 0.1)
    df["Visit_Count_90D"] = rng.binomial(visits, 0.28)
    df["Visit_Count_180D"] = rng.binomial(visits, 0.5)
    df["Visit_Count_365D"] = visits
    has_visit = visits > 0
    last_visit = np.where(has_visit, _dates(rng, idx - pd.DateOffset(years=1), idx - pd.Timedelta(days=1), n), NAT)
    df["Last_Visit_Date"] = pd.Series(last_visit, dtype="datetime64[us]")
    df["Days_Since_Last_Visit"] = _mask((row_idx - pd.Series(last_visit)).dt.days.to_numpy(na_value=0), has_visit)
    med_missing = rng.random(n) < 0.04
    rx_w = rng.poisson(20, n)
    rx_p = rng.binomial(rx_w, 0.8)
    df["Rx_Written_365D"] = _mask(rx_w, ~med_missing)
    df["Rx_Purchased_365D"] = _mask(rx_p, ~med_missing)
    df["Prescription_Fill_Ratio_365D"] = np.where(~med_missing & (rx_w > 0), rx_p / np.maximum(rx_w, 1), np.nan)
    lab_r = rng.poisson(4, n)
    lab_p = rng.binomial(lab_r, 0.85)
    df["Lab_Referred_365D"] = _mask(lab_r, np.ones(n, bool))
    df["Lab_Performed_365D"] = _mask(lab_p, np.ones(n, bool))
    df["Lab_Performed_Ratio_365D"] = np.where(lab_r > 0, lab_p / np.maximum(lab_r, 1), np.nan)
    df["Unsettled_Visit_Counter_Ind"] = (rng.random(n) < 0.02).astype(int)

    # ---- nurse blocks (assessment-conditional): NULL = not assessed
    def nurse_flag(p: float) -> pd.Series:
        return _mask((rng.random(n) < p).astype(int), assessed)

    df["Falls_Risk_Assessment_Score"] = _mask(rng.choice([1, 2, 3], n, p=[0.5, 0.3, 0.2]), assessed)
    assess_date = np.where(assessed, _dates(rng, idx - pd.DateOffset(years=1), idx - pd.Timedelta(days=1), n), NAT)
    df["Falls_Risk_Assessment_Date"] = pd.Series(assess_date, dtype="datetime64[us]")
    df["Fall_Self_Report_Value"] = _mask(rng.choice([0, 1, 2], n, p=[0.6, 0.3, 0.1]), assessed)
    df["Fall_Self_Report_Date"] = pd.Series(assess_date, dtype="datetime64[us]")
    df["Fall_Count_Self_Reported"] = np.nan
    df["Fall_Required_Treatment_Ind"] = nurse_flag(0.2)
    df["Instability_Dizziness_Ind"] = nurse_flag(0.3)
    df["Fear_Of_Falling_Ind"] = nurse_flag(0.35)
    df["Confined_To_Chair_Bed_Ind"] = _mask((mobility == 3).astype(int), assessed)
    df["Balance_Disorder_Ind"] = nurse_flag(0.3)
    df["Shuffling_Gait_Ind"] = nurse_flag(0.15)
    df["Hesitant_Gait_Ind"] = nurse_flag(0.2)
    df["Uses_Walking_Aid_Ind"] = _mask(walking_aid, assessed)
    df["Prior_Falls_Nurse_Ind"] = _mask(np.where(assessed, np.maximum(prior_since_start, (rng.random(n) < 0.1).astype(int)), 0), assessed)
    df["Muscle_Weakness_Ind"] = nurse_flag(0.3)
    df["Cognitive_Impairment_Nurse_Ind"] = _mask(np.maximum(dementia, (rng.random(n) < 0.1).astype(int)), assessed)
    df["On_Medications_Nurse_Ind"] = nurse_flag(0.9)
    df["Unsafe_Environment_Ind"] = nurse_flag(0.15)
    df["Unsuitable_Footwear_Ind"] = nurse_flag(0.1)
    df["Missing_Sensory_Aids_Ind"] = nurse_flag(0.1)
    df["Unsuitable_Bed_Chair_Height_Ind"] = nurse_flag(0.08)
    df["Vision_Impairment_Nurse_Ind"] = nurse_flag(0.25)
    df["Orthostatic_Hypotension_Ind"] = nurse_flag(0.1)
    df["Polypharmacy_Nurse_Ind"] = nurse_flag(0.5)
    df["No_Primary_Caregiver_Ind"] = nurse_flag(0.2)
    df["Get_Up_And_Go_Date"] = pd.Series(assess_date, dtype="datetime64[us]")
    df["ADL_Worst_Domain_Score"] = _mask(np.minimum(4, mobility + rng.choice([0, 1], n)), assessed)
    df["ADL_Domains_Assessed"] = _mask(rng.integers(3, 7, n), assessed)
    df["Transfer_Dependency_Score"] = _mask(np.minimum(4, mobility + rng.choice([0, 0, 1], n)), assessed)
    df["Mobility_Score"] = _mask(mobility, assessed)
    df["Vision_Score"] = _mask(rng.choice([0, 1, 2], n, p=[0.7, 0.27, 0.03]), assessed)
    df["Hearing_Score"] = _mask(rng.choice([0, 1, 2], n, p=[0.7, 0.28, 0.02]), assessed)
    df["Mobility_Assessment_Date"] = pd.Series(assess_date, dtype="datetime64[us]")
    df["MMSE_Score"] = np.nan
    df["MMSE_Date"] = pd.Series([NAT] * n, dtype="datetime64[us]")
    cog = assessed & (rng.random(n) < 0.5)
    df["MiniCog_Score"] = np.where(cog, np.clip(rng.normal(3.6 - 1.5 * dementia, 1.1, n).round(), 0, 5), np.nan)
    df["MiniCog_Date"] = pd.Series(np.where(cog, assess_date, NAT), dtype="datetime64[us]")
    df["Fall_Location_At_Home_Ind"] = nurse_flag(0.6)
    df["Fall_Count_Home_Safety"] = np.nan
    df["Orthostatic_BP_Measured_Ind"] = nurse_flag(0.4)
    df["Slow_Rising_Ind"] = nurse_flag(0.5)
    df["Vitamin_D_Ind"] = nurse_flag(0.3)
    df["Home_Safety_Risk_Count"] = _mask(rng.poisson(1.2, n), assessed)
    df["Home_Safety_Questions_Answered"] = _mask(rng.integers(5, 12, n), assessed)
    df["Home_Safety_Assessment_Date"] = pd.Series(assess_date, dtype="datetime64[us]")
    df["Unmapped_Answer_Count"] = _mask(rng.poisson(0.1, n), assessed)
    df["Assessment_Count_365D"] = _mask(np.where(assessed, rng.integers(1, 4, n), 0), np.ones(n, bool))
    df["Last_Assessment_Date"] = pd.Series(assess_date, dtype="datetime64[us]")
    df["Days_Since_Last_Assessment"] = _mask((row_idx - pd.Series(assess_date)).dt.days.to_numpy(na_value=0), assessed)
    df["Distinct_Assessment_Dates_365D"] = _mask(np.where(assessed, rng.integers(1, 3, n), 0), np.ones(n, bool))
    df["Has_Repeat_Assessment_Ind"] = _mask((df["Distinct_Assessment_Dates_365D"].fillna(0) > 1).astype(int).to_numpy(), assessed)
    df["Assessment_Not_Performed_Ind"] = (~assessed).astype(int)

    # ---- medication
    substances = rng.poisson(6.5, n)
    df["Distinct_Active_Substance_Count"] = _mask(substances, ~med_missing)
    df["Polypharmacy_5Plus_Ind"] = _mask((substances >= 5).astype(int), ~med_missing)
    df["Polypharmacy_10Plus_Ind"] = _mask((substances >= 10).astype(int), ~med_missing)
    df["Narcotic_Drug_Count"] = _mask(rng.binomial(1, 0.08, n), ~med_missing)
    df["Prescribed_Not_Collected_365D"] = _mask(rx_w - rx_p, ~med_missing)
    df["Credited_Prescriptions_365D"] = _mask(rng.poisson(0.3, n), ~med_missing)
    df["Approx_Matched_365D"] = _mask(rng.poisson(0.5, n), ~med_missing)
    df["Bought_Outside_Meuhedet_365D"] = _mask(rng.poisson(0.4, n), ~med_missing)
    df["Atc_Unmatched_365D"] = _mask(rng.poisson(0.2, n), ~med_missing)
    df["Future_Dated_365D"] = _mask(np.zeros(n, int), ~med_missing)
    df["Fallback_Exposure_365D"] = _mask(rng.poisson(0.3, n), ~med_missing)
    df["Total_DDD_Amount"] = np.where(~med_missing, (substances * rng.gamma(2, 120, n)).round(2), np.nan)
    df["Medication_Missing_Ind"] = med_missing.astype(int)

    # ---- external care / hospitalisation
    ext_missing = rng.random(n) < 0.03
    ext = rng.poisson(1.5, n)
    hidden = rng.binomial(ext, 0.1)
    df["External_Care_Count_30D"] = rng.binomial(ext, 0.1)
    df["External_Care_Count_90D"] = rng.binomial(ext, 0.28)
    df["External_Care_Count_180D"] = rng.binomial(ext, 0.5)
    df["External_Care_Count_365D"] = ext
    df["External_Care_Count_365D_Visible"] = ext - hidden
    df["External_Care_Hidden_By_Billing_Lag_365D"] = hidden
    df["Max_Invoice_Lag_365D"] = _mask(rng.integers(0, 120, n), ext > 0)
    has_ext = ext > 0
    last_ext = np.where(has_ext, _dates(rng, idx - pd.DateOffset(years=1), idx - pd.Timedelta(days=1), n), NAT)
    df["Last_Ext_Date"] = pd.Series(last_ext, dtype="datetime64[us]")
    df["Days_Since_Last_External_Care"] = _mask((row_idx - pd.Series(last_ext)).dt.days.to_numpy(na_value=0), has_ext)
    hosp = rng.poisson(0.25, n)
    df["Hospitalization_Count_180D"] = rng.binomial(hosp, 0.5)
    df["Hospitalization_Count_365D"] = hosp
    df["Total_Hosp_Days_365D"] = _mask(hosp * rng.integers(1, 9, n), np.ones(n, bool))
    df["Last_Hosp_Length"] = _mask(rng.integers(1, 9, n), hosp > 0)
    last_hosp = np.where(hosp > 0, _dates(rng, idx - pd.DateOffset(years=1), idx - pd.Timedelta(days=1), n), NAT)
    df["Last_Hosp_Date"] = pd.Series(last_hosp, dtype="datetime64[us]")
    df["Days_Since_Last_Hospitalization"] = _mask((row_idx - pd.Series(last_hosp)).dt.days.to_numpy(na_value=0), hosp > 0)
    df["Invoice_Src_Rows_365D"] = _mask(ext + hosp, np.ones(n, bool))
    df["External_Care_Missing_Ind"] = ext_missing.astype(int)

    # ---- registries (population-wide)
    reg_missing = patients["reg_missing"] if fixed else rng.random(n) < 0.01
    reg = {"Registry_Dementia_Ind": dementia, "Registry_Home_Confined_Ind": (rng.random(n) < 0.04 + 0.02 * (mobility >= 2)).astype(int),
           "Registry_Suicide_Attempt_Ind": (rng.random(n) < 0.004).astype(int), "Registry_Heart_Disease_Ind": (rng.random(n) < 0.22).astype(int),
           "Registry_Diabetes_T1_Ind": (rng.random(n) < 0.01).astype(int), "Registry_Diabetes_T2_Ind": (rng.random(n) < 0.28).astype(int),
           "Registry_Chronic_Renal_Failure_Ind": (rng.random(n) < 0.07).astype(int), "Registry_Kidneys_Ind": (rng.random(n) < 0.1).astype(int),
           "Registry_Liver_Ind": (rng.random(n) < 0.02).astype(int), "Registry_COPD_Ind": (rng.random(n) < 0.09).astype(int),
           "Registry_Asthma_Ind": (rng.random(n) < 0.07).astype(int), "Registry_Blood_Pressure_Ind": (rng.random(n) < 0.6).astype(int),
           "Registry_Transplant_Ind": (rng.random(n) < 0.003).astype(int), "Registry_Stoma_Ind": (rng.random(n) < 0.004).astype(int)}
    if fixed:
        reg = {k: patients["registries"][k] for k in reg}
    for k, v in reg.items():
        reg[k] = np.where(reg_missing, 0, v)
    smi = patients["smi"] if fixed else np.where(rng.random(n) < 0.05, rng.choice([1, 2, 3, 4], n), 0)
    active = sum(reg.values()) + (smi > 0)
    df["Active_Registry_Count"] = active
    df["Distinct_Registry_Type_Count"] = active
    df["Registry_Dementia_Ind"] = reg["Registry_Dementia_Ind"]
    df["Registry_Home_Confined_Ind"] = reg["Registry_Home_Confined_Ind"]
    df["Registry_SMI_Level"] = _mask(smi, (smi > 0) & ~reg_missing)
    for k in ("Registry_Suicide_Attempt_Ind", "Registry_Heart_Disease_Ind", "Registry_Diabetes_T1_Ind", "Registry_Diabetes_T2_Ind",
              "Registry_Chronic_Renal_Failure_Ind", "Registry_Kidneys_Ind", "Registry_Liver_Ind", "Registry_COPD_Ind", "Registry_Asthma_Ind",
              "Registry_Blood_Pressure_Ind", "Registry_Transplant_Ind", "Registry_Stoma_Ind"):
        df[k] = reg[k]
    has_reg = active > 0
    first_reg = np.where(has_reg, _dates(rng, idx - pd.DateOffset(years=15), idx - pd.Timedelta(days=30), n), NAT)
    df["Longest_Registry_Duration_Days"] = _mask((row_idx - pd.Series(first_reg)).dt.days.to_numpy(na_value=0), has_reg)
    df["Registry_Distinct_Status_Values"] = _mask(rng.integers(1, 3, n), has_reg)
    df["Days_Since_First_Registry"] = df["Longest_Registry_Duration_Days"]
    df["Registry_Unknown_Dictionary_Count"] = 0
    df["New_Registry_30D"] = rng.binomial(1, 0.01, n)
    df["New_Registry_90D"] = df["New_Registry_30D"] + rng.binomial(1, 0.02, n)
    df["First_Registry_Date"] = pd.Series(first_reg, dtype="datetime64[us]")
    df["Registry_Missing_Ind"] = reg_missing.astype(int)

    # ---- MEFI
    df["MEFI_Group_At_Index"] = _mask(mefi, mefi_assessed)
    df["MEFI_Group_Desc"] = pd.Series(np.where(mefi_assessed, np.array(["", "LOW", "MEDIUM", "HIGH", "VERY_HIGH"])[mefi], None), dtype="string")
    mefi_from = np.where(mefi_assessed, _dates(rng, idx - pd.DateOffset(years=2), idx - pd.Timedelta(days=1), n), NAT)
    df["MEFI_From_Date"] = pd.Series(mefi_from, dtype="datetime64[us]")
    df["MEFI_Assessed_Ind"] = mefi_assessed.astype(int)
    days_in = (row_idx - pd.Series(mefi_from)).dt.days.to_numpy(na_value=0)
    df["Days_In_Current_MEFI_Group"] = _mask(days_in, mefi_assessed)
    df["Days_Since_MEFI_Assessment"] = _mask(days_in, mefi_assessed)
    df["MEFI_Assessment_Count"] = np.where(mefi_assessed, rng.integers(1, 5, n), 0)
    df["MEFI_Ever_Assessed_Ind"] = mefi_assessed.astype(int)
    prev_known = mefi_assessed & (df["MEFI_Assessment_Count"] > 1)
    prev = np.clip(mefi - rng.choice([-1, 0, 0, 1], n), 1, 4)
    df["MEFI_Prev_Group"] = _mask(prev, prev_known)
    df["MEFI_Worsened_Ind"] = _mask((mefi > prev).astype(int), prev_known)
    df["Frailty_Not_Assessed_Ind"] = (~mefi_assessed).astype(int)

    # ---- comorbidity / social
    df["CCI_Group"] = _mask(rng.choice([0, 1, 2, 3], n, p=[0.35, 0.3, 0.2, 0.15]), rng.random(n) < 0.95)
    df["Malnutrition_Grade"] = _mask(rng.choice([0, 1, 2], n, p=[0.8, 0.15, 0.05]), assessed)
    df["Siudi_Status"] = _mask(rng.choice([1, 2, 3], n), rng.random(n) < 0.12)
    df["Customer_Risk_Code"] = _mask(rng.choice([1, 2, 3, 4], n), rng.random(n) < 0.7)
    df["Income_Support"] = _mask((rng.random(n) < 0.15).astype(int), rng.random(n) < 0.98)
    df["Audit_Only_Adif_Status"] = _mask(rng.choice([0, 1], n), rng.random(n) < 0.5)
    df["Audit_Only_Si_Status"] = _mask(rng.choice([0, 1], n), rng.random(n) < 0.5)
    for k, present in (("Has_Source_Visits_Ind", has_visit), ("Has_Source_Diagnosis_Ind", has_dx), ("Has_Source_Assessment_Ind", assessed),
                       ("Has_Source_Medication_Ind", ~med_missing), ("Has_Source_External_Care_Ind", ~ext_missing),
                       ("Has_Source_Registry_Ind", ~reg_missing), ("Has_Source_Frailty_Ind", mefi_assessed)):
        df[k] = np.asarray(present).astype(int)
    df["Distinct_Sources_With_Data"] = df[[c for c in df.columns if c.startswith("Has_Source_")]].sum(axis=1)
    df["Snapshot_Confidence"] = np.where(df["Distinct_Sources_With_Data"] >= 6, "HIGH", np.where(df["Distinct_Sources_With_Data"] >= 4, "MEDIUM", "LOW"))
    df["Leakage_Check_Ind"] = 0

    # ---- labels
    days_to_180 = rng.integers(0, 181, n)
    days_to_30 = rng.integers(0, 31, n)
    lab180 = np.where(censored, pd.NA, y180.astype(int))
    lab30 = np.where(censored, pd.NA, y30.astype(int))
    df["Fall_Next_30D_Ind"] = pd.array(lab30, dtype="Int64")
    ev30 = np.where(y30 & ~censored, (row_idx + pd.to_timedelta(days_to_30, unit="D")).to_numpy(), NAT)
    df["Next_Fall_Date_30D"] = pd.Series(ev30, dtype="datetime64[us]")
    df["Days_To_Next_Fall_30D"] = _mask(days_to_30, y30 & ~censored)
    df["Next_Fall_Event_ID_30D"] = pd.Series(np.where(y30 & ~censored, [f"EV30-{i}" for i in range(n)], None), dtype="string")
    conf30 = np.where(y30 & ~censored, "HIGH", "")
    df["Next_Fall_Confidence_30D"] = pd.Series(conf30, dtype="string").mask(pd.Series(conf30) == "")
    df["Label_Reason_30D"] = np.where(censored, "CENSORED", np.where(y30, "POSITIVE", "NEGATIVE_FULL_FOLLOWUP"))
    df["Fall_Next_180D_Ind"] = pd.array(lab180, dtype="Int64")
    ev180 = np.where(y180 & ~censored, (row_idx + pd.to_timedelta(np.where(y30, days_to_30, days_to_180), unit="D")).to_numpy(), NAT)
    df["Next_Fall_Date_180D"] = pd.Series(ev180, dtype="datetime64[us]")
    df["Days_To_Next_Fall_180D"] = _mask(np.where(y30, days_to_30, days_to_180), y180 & ~censored)
    df["Next_Fall_Event_ID_180D"] = pd.Series(np.where(y180 & ~censored, [f"EV180-{i}" for i in range(n)], None), dtype="string")
    conf180 = np.where(y180 & ~censored, rng.choice(["HIGH", "MEDIUM"], n, p=[0.85, 0.15]), "")
    df["Next_Fall_Confidence_180D"] = pd.Series(conf180, dtype="string").mask(pd.Series(conf180) == "")
    df["Label_Reason_180D"] = np.where(censored, np.where(deceased, "CENSORED_DEATH", "CENSORED"), np.where(y180, "POSITIVE", "NEGATIVE_FULL_FOLLOWUP"))
    df["Fall_Next_30D_HighConf_Ind"] = pd.array(np.where(censored, pd.NA, (y30 & (conf30 == "HIGH")).astype(int)), dtype="Int64")
    df["Fall_Next_180D_HighConf_Ind"] = pd.array(np.where(censored, pd.NA, (y180 & (conf180 == "HIGH")).astype(int)), dtype="Int64")

    ordered = df[contract.names]
    for c in contract.columns:
        if c.is_integer:
            ordered[c.name] = ordered[c.name].astype("Int64")
        elif c.is_text:
            ordered[c.name] = ordered[c.name].astype("string")
    return ordered


def _draw_patients(rng: np.random.Generator, n: int, first_index: pd.Timestamp, n_snapshots: int, churn: float) -> dict[str, Any]:
    """Static per-patient traits drawn once for a panel, plus the snapshot range each patient is present in (churn = share of
    patients joining after the first snapshot or leaving before the last)."""
    age0 = np.clip(rng.normal(76, 7.5, n).round(), 62, 101).astype(int)
    reg = {"Registry_Home_Confined_Ind": (rng.random(n) < 0.05).astype(int), "Registry_Suicide_Attempt_Ind": (rng.random(n) < 0.004).astype(int),
           "Registry_Heart_Disease_Ind": (rng.random(n) < 0.22).astype(int), "Registry_Diabetes_T1_Ind": (rng.random(n) < 0.01).astype(int),
           "Registry_Diabetes_T2_Ind": (rng.random(n) < 0.28).astype(int), "Registry_Chronic_Renal_Failure_Ind": (rng.random(n) < 0.07).astype(int),
           "Registry_Kidneys_Ind": (rng.random(n) < 0.1).astype(int), "Registry_Liver_Ind": (rng.random(n) < 0.02).astype(int),
           "Registry_COPD_Ind": (rng.random(n) < 0.09).astype(int), "Registry_Asthma_Ind": (rng.random(n) < 0.07).astype(int),
           "Registry_Blood_Pressure_Ind": (rng.random(n) < 0.6).astype(int), "Registry_Transplant_Ind": (rng.random(n) < 0.003).astype(int),
           "Registry_Stoma_Ind": (rng.random(n) < 0.004).astype(int)}
    dementia = (rng.random(n) < 0.06 + 0.004 * np.clip(age0 - 65, 0, 35)).astype(int)
    reg["Registry_Dementia_Ind"] = dementia
    mefi_assessed = rng.random(n) < 0.6
    joins = np.where(rng.random(n) < churn, rng.integers(1, max(n_snapshots, 2), n), 0)
    leaves = np.where(rng.random(n) < churn, rng.integers(0, max(n_snapshots - 1, 1), n), n_snapshots - 1)
    leaves = np.maximum(leaves, joins)
    return {"age0": age0, "sex": rng.choice([1, 2], n, p=[0.45, 0.55]), "birth_suspect": (rng.random(n) < 0.1).astype(int),
            "hist0": rng.integers(200, 12000, n), "from_date": _dates(rng, first_index - pd.DateOffset(years=20), first_index - pd.DateOffset(years=1), n),
            "dementia": dementia, "registries": reg, "reg_missing": rng.random(n) < 0.01,
            "smi": np.where(rng.random(n) < 0.05, rng.choice([1, 2, 3, 4], n), 0), "mefi_assessed": mefi_assessed,
            "mefi": np.where(mefi_assessed, rng.choice([1, 2, 3, 4], n, p=[0.45, 0.3, 0.17, 0.08]), 0),
            "first_snapshot": joins, "last_snapshot": leaves}


def _censor_past_freeze(df: pd.DataFrame, freeze: pd.Timestamp) -> pd.DataFrame:
    """Rows whose label window ends after the data-freeze date: the VIEW cannot know the full window. An event already observed
    before the freeze stays a positive label (as a partially observed window would report it); every other row is censored
    (label NULL, Is_Censored = 1, Has_Full_*_Label = 0, Label_Reason CENSORED_END_OF_DATA) - the partial-observability pattern the
    snapshot audit must exclude."""
    idx = df["Index_Date"].astype("datetime64[us]")
    for h, days in (("30D", 30), ("180D", 180)):
        incomplete = (idx + pd.Timedelta(days=days)) > np.datetime64(freeze, "us")
        if not incomplete.any():
            continue
        ev = df[f"Next_Fall_Date_{h}"].astype("datetime64[us]")
        seen = incomplete & ev.notna() & (ev <= np.datetime64(freeze, "us"))
        drop = incomplete & ~seen
        df.loc[incomplete, f"Has_Full_{h}_Label"] = 0
        df.loc[incomplete, f"Has_Full_Followup_{h}"] = 0
        df.loc[drop, f"Fall_Next_{h}_Ind"] = pd.NA
        df.loc[drop, f"Fall_Next_{h}_HighConf_Ind"] = pd.NA
        df.loc[drop, f"Is_Censored_{h}"] = 1
        df.loc[drop, f"Label_Reason_{h}"] = "CENSORED_END_OF_DATA"
        for c in (f"Next_Fall_Date_{h}", f"Days_To_Next_Fall_{h}", f"Next_Fall_Event_ID_{h}", f"Next_Fall_Confidence_{h}"):
            df.loc[drop, c] = pd.NaT if c.startswith("Next_Fall_Date") else pd.NA
    return df


def generate_synthetic_wide_panel(n_patients: int = 1500, index_dates: Sequence[str] | None = None, *, n_snapshots: int = 12,
                                  first_index_date: str = "2024-01-01", seed: int = 20260922, contract: WideContract | None = None,
                                  data_freeze_date: str | None = None, churn: float = 0.05, n_index_day_dx: int = 0, n_index_day_falls: int = 0,
                                  n_future_dx: int = 0, n_future_falls: int = 0) -> pd.DataFrame:
    """A deterministic monthly panel: ``n_patients`` members repeated on every snapshot (``index_dates`` or ``n_snapshots`` monthly
    dates from ``first_index_date``), one row per member and Index_Date, static traits fixed, "since study start" flags monotone,
    ``churn`` of the patients joining late or leaving early. With ``data_freeze_date`` the label windows ending after it are
    censored (:func:`_censor_past_freeze`). The timing defects are planted on the *first* snapshot only, so the D-00 diagnostic sees
    both affected and clean snapshots. SYNTHETIC DATA - NOT SCIENTIFIC RESULTS."""
    contract = contract or load_wide_contract(DEFAULT_CONTRACT)
    if index_dates is None:
        first = pd.Timestamp(first_index_date)
        index_dates = [str((first + pd.DateOffset(months=k)).date()) for k in range(int(n_snapshots))]
    dates = [pd.Timestamp(d) for d in index_dates]
    if len(dates) != len(set(dates)) or dates != sorted(dates):
        raise ValueError("index_dates must be distinct and ascending")
    rng = np.random.default_rng(seed)
    patients = _draw_patients(rng, int(n_patients), dates[0], len(dates), float(churn))
    state = {"fall_ever": np.zeros(int(n_patients), dtype=bool), "gait_ever": np.zeros(int(n_patients), dtype=bool)}
    frames = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
        for k, idx in enumerate(dates):
            months = (idx.year - dates[0].year) * 12 + idx.month - dates[0].month
            present = (patients["first_snapshot"] <= k) & (k <= patients["last_snapshot"])
            snap = _generate(int(n_patients), rng, idx, contract, 0.0, n_index_day_falls if k == 0 else 0, patients=patients, months=months,
                             state=state, n_index_day_dx=n_index_day_dx if k == 0 else 0, n_future_dx=n_future_dx if k == 0 else 0,
                             n_future_falls=n_future_falls if k == 0 else 0, present=present)
            frames.append(snap.loc[present])
    panel = pd.concat(frames, ignore_index=True)
    if data_freeze_date is not None:
        panel = _censor_past_freeze(panel, pd.Timestamp(data_freeze_date))
    return panel


def write_synthetic_wide_extract(path: str | Path, *, n_rows: int = 3000, seed: int = 20260917, index_date: str = "2025-01-01",
                                 csv_null_literal: str = "NULL", csv_date_format: str = "%Y-%m-%d", excel_numeric_ids: bool = False,
                                 sheet_name: str = "V_Falls_Prediction_Wide_1", **kwargs) -> Path:
    """Write the synthetic extract as .parquet, .csv (SQL-Server-style 'NULL' literals) or .xlsx (empty cells for NULL, dates as
    Excel dates). ``excel_numeric_ids`` stores Customer_Full_ID as numbers, imitating an Excel export that lost the text type;
    ``csv_date_format`` writes the CSV dates in one of the contract's declared source layouts (e.g. "%d/%m/%Y" as SSMS with an
    Israeli locale exports them)."""
    path = Path(path)
    if "index_dates" in kwargs or "n_snapshots" in kwargs:   # monthly panel (repeated patients); n_rows = patients per snapshot
        df = generate_synthetic_wide_panel(n_rows, seed=seed, first_index_date=index_date, **kwargs)
    else:
        df = generate_synthetic_wide_extract(n_rows, seed=seed, index_date=index_date, **kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    elif path.suffix.lower() in {".xlsx", ".xlsm"}:
        out = df.copy()
        if excel_numeric_ids:
            out["Customer_Full_ID"] = [int(v[1:]) for v in out["Customer_Full_ID"]]
        for c in out.columns:
            if str(out[c].dtype) == "Int64":
                out[c] = out[c].astype(object).where(out[c].notna(), None)
        out.to_excel(path, index=False, sheet_name=sheet_name, engine="openpyxl")
    else:
        out = df.copy()
        for c in out.columns:
            if pd.api.types.is_datetime64_any_dtype(out[c]):
                out[c] = out[c].dt.strftime(csv_date_format)
        out.to_csv(path, index=False, na_rep=csv_null_literal, encoding="utf-8", lineterminator="\n")
    return path
