"""SYNTHETIC Phase 3 extracts (software tests and rehearsals only - SYNTHETIC DATA, NOT SCIENTIFIC RESULTS).

Starts from the existing 221-column generator (``falls_ml.data.meuhedet_synthetic``) and plants the historical-timing situations Phase 3 must
handle, with the outcome-dependence that makes deleting patients a biased selection:

- ``same_day``   record dates moved to the index day (overwritten pre-index history), preferentially for patients with a later recorded fall;
- ``future``     record dates moved 1-60 days after the index day (the latest record as of the extraction, not as of the index date);
- ``undated``    recorded item values whose record date is missing (timing not bounded);
- ``unreadable`` record-date cells written as ``00:00.0`` (an Excel time-only artefact the validated reader cannot read);
- D-00 rows      diagnosis dates on the index day (the rows D00_CLEAN removed), so FULL_LABELED is larger than D00_CLEAN;
- the label      follows the corrected contract (events from Index_Date + 1 day) unless ``index_day_outcomes=True`` (a contradicting extract).
The CSV mixes the declared date layouts YYYY-MM-DD and DD/MM/YYYY (never ambiguous under the contract) and keeps 2999-12-31 sentinels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

FORM_DATES = {"MiniCog_Date": ["MiniCog_Score"], "Falls_Risk_Assessment_Date": ["Falls_Risk_Assessment_Score", "Instability_Dizziness_Ind", "Fear_Of_Falling_Ind"],
              "Get_Up_And_Go_Date": ["Uses_Walking_Aid_Ind", "Balance_Disorder_Ind", "Muscle_Weakness_Ind"],
              "Home_Safety_Assessment_Date": ["Home_Safety_Risk_Count", "Fall_Location_At_Home_Ind"], "Mobility_Assessment_Date": ["ADL_Worst_Domain_Score", "Mobility_Score"],
              "MEFI_From_Date": ["MEFI_Group_At_Index"], "Last_Visit_Date": [], "Last_Assessment_Date": []}


def _pick(rng: np.random.Generator, pool: np.ndarray, y: np.ndarray, share: float, enrich: float) -> np.ndarray:
    """Rows of ``pool`` (a share of it), with probability proportional to 1 + enrich * y (outcome-dependent overwriting)."""
    k = int(round(share * len(pool)))
    if k <= 0 or not len(pool):
        return np.array([], dtype=int)
    w = 1.0 + enrich * y[pool]
    return np.sort(rng.choice(pool, size=min(k, len(pool)), replace=False, p=w / w.sum()))


def plant_phase3_history(df: pd.DataFrame, *, index_date: str = "2025-01-01", seed: int = 7, same_day: dict[str, float] | None = None,
                         future: dict[str, float] | None = None, undated: dict[str, float] | None = None, n_unreadable: dict[str, int] | None = None,
                         d00_dx_share: float = 0.03, enrich: float = 6.0, index_day_outcomes: bool = False,
                         episode_duplicates: int = 0) -> tuple[pd.DataFrame, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    out = df.copy()
    idx = pd.Timestamp(index_date)
    if not index_day_outcomes:          # corrected contract: an event ON the index day is history, never the outcome -> first event from day 1
        for h in ("180D", "30D"):
            ev, dd = f"Next_Fall_Date_{h}", f"Days_To_Next_Fall_{h}"
            day0 = (out[ev].dt.normalize() == idx).fillna(False).to_numpy()
            out.loc[day0, ev] = out.loc[day0, ev] + pd.Timedelta(days=1)
            out.loc[day0, dd] = pd.to_numeric(out.loc[day0, dd]) + 1
    elig = ((out["Index_Date"] == idx) & (out["Is_Eligible_Cohort"] == 1) & out["Fall_Next_180D_Ind"].notna()).to_numpy()
    y = out["Fall_Next_180D_Ind"].fillna(0).to_numpy(dtype=float)
    log: dict[str, Any] = {"same_day": {}, "future": {}, "undated": {}, "unreadable": {}}
    for col, share in (same_day or {}).items():
        pool = np.flatnonzero(elig & out[col].notna().to_numpy())
        rows = _pick(rng, pool, y, share, enrich)
        out.loc[out.index[rows], col] = np.datetime64(idx, "us")
        from falls_ml.data.meuhedet_timing import DAYS_SINCE_COMPANION

        if DAYS_SINCE_COMPANION.get(col) in out:      # keep the VIEW's derived day count consistent with the moved date
            out.loc[out.index[rows], DAYS_SINCE_COMPANION[col]] = 0
        log["same_day"][col] = int(len(rows))
    for col, share in (future or {}).items():
        pool = np.flatnonzero(elig & out[col].notna().to_numpy() & (out[col] < idx).to_numpy())
        rows = _pick(rng, pool, y, share, enrich)
        out.loc[out.index[rows], col] = (idx + pd.to_timedelta(rng.integers(1, 61, len(rows)), unit="D")).to_numpy(dtype="datetime64[us]")
        log["future"][col] = int(len(rows))
    for col, share in (undated or {}).items():
        items = FORM_DATES.get(col) or []
        pool = np.flatnonzero(elig & out[col].notna().to_numpy() & (out[items].notna().any(axis=1).to_numpy() if items else True))
        rows = _pick(rng, pool, y, share, 0.0)
        out.loc[out.index[rows], col] = pd.NaT
        log["undated"][col] = int(len(rows))
    if d00_dx_share:
        pool = np.flatnonzero(elig & out["Last_Dx_Date"].notna().to_numpy())
        rows = _pick(rng, pool, y, d00_dx_share, 2.0)
        out.loc[out.index[rows], "Last_Dx_Date"] = np.datetime64(idx, "us")
        if "Days_Since_Last_Diagnosis" in out:
            out.loc[out.index[rows], "Days_Since_Last_Diagnosis"] = 0
        log["d00_dx_rows"] = int(len(rows))
    if episode_duplicates:             # one injury episode recorded as the last fall AND as the next event (a proxy artefact V7 must flag)
        elig2 = ((out["Index_Date"] == idx) & (out["Is_Eligible_Cohort"] == 1) & (out["Fall_Next_180D_Ind"] == 1)).to_numpy()
        rows = np.flatnonzero(elig2)[: int(episode_duplicates)]
        last = idx - pd.to_timedelta(rng.integers(1, 5, len(rows)), unit="D")
        out.loc[out.index[rows], "Last_Fall_Date"] = last.to_numpy(dtype="datetime64[us]")
        out.loc[out.index[rows], "Days_Since_Last_Fall"] = (idx - last).days.to_numpy()
        out.loc[out.index[rows], "Prior_Fall_Since_Study_Start_Ind"] = 1
        nxt = idx + pd.to_timedelta(rng.integers(1, 4, len(rows)), unit="D")
        out.loc[out.index[rows], "Next_Fall_Date_180D"] = nxt.to_numpy(dtype="datetime64[us]")
        out.loc[out.index[rows], "Days_To_Next_Fall_180D"] = (nxt - idx).days.to_numpy()
        log["episode_duplicates"] = int(len(rows))
    log["unreadable"] = dict(n_unreadable or {})
    return out, log


def write_phase3_csv(df: pd.DataFrame, path: Path, *, seed: int = 11, n_unreadable: dict[str, int] | None = None, ddmm_share: float = 0.5) -> Path:
    """CSV like the real SSMS export: 'NULL' literals, dates as YYYY-MM-DD or DD/MM/YYYY (per column, declared layouts), a few '00:00.0' cells."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    idx = pd.Timestamp("2025-01-01")
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            fmt = "%d/%m/%Y" if rng.random() < ddmm_share else "%Y-%m-%d"
            out[c] = out[c].dt.strftime(fmt)
    for col, k in (n_unreadable or {}).items():
        elig = ((df["Index_Date"] == idx) & (df["Is_Eligible_Cohort"] == 1) & df["Fall_Next_180D_Ind"].notna() & df[col].notna()).to_numpy()
        rows = np.flatnonzero(elig)[: int(k)]
        out.loc[out.index[rows], col] = "00:00.0"
    out.to_csv(path, index=False, na_rep="NULL", lineterminator="\n")
    return path
