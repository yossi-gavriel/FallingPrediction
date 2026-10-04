"""Synthetic 2026 snapshot for the Phase 4 tests (never real data): the Phase 1 synthetic wide extract on Index_Date 2026-01-01 with the corrected
contract, a known number of patients shared with a 2025 synthetic cohort, and new V21-style columns (one strongly outcome-related predictor, a
new date column, a name-sealed outcome column) that must never reach a frozen model."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def make_2026(df25: pd.DataFrame, *, n_rows: int = 3500, seed: int = 12, n_overlap: int = 700, index_date: str = "2026-01-01",
              index_day_outcomes: bool = False, definition_version: str = "V21") -> tuple[pd.DataFrame, dict[str, Any]]:
    from falls_ml.data.meuhedet_synthetic import generate_synthetic_wide_extract
    from falls_ml.phase3.synthetic import plant_phase3_history

    df = generate_synthetic_wide_extract(n_rows, seed=seed, index_date=index_date, n_index_day_falls=4)
    df, _ = plant_phase3_history(df, index_date=index_date, seed=seed, index_day_outcomes=index_day_outcomes)
    df = df.reset_index(drop=True)
    df["Customer_Full_ID"] = [f"T{i:010d}" for i in range(len(df))]
    idx25 = pd.Timestamp("2025-01-01")
    full25 = df25.loc[(df25["Index_Date"] == idx25) & (df25["Is_Eligible_Cohort"] == 1) & df25["Fall_Next_180D_Ind"].notna(), "Customer_Full_ID"].astype(str).tolist()
    elig_any25 = set(df25.loc[(df25["Index_Date"] == idx25) & (df25["Is_Eligible_Cohort"] == 1), "Customer_Full_ID"].astype(str))
    idx = pd.Timestamp(index_date)
    el26 = np.flatnonzero(((df["Index_Date"] == idx) & (df["Is_Eligible_Cohort"] == 1)).to_numpy())
    rng = np.random.default_rng(seed)
    k = min(n_overlap, len(full25), len(el26))
    chosen = rng.choice(np.array(full25, dtype=object), size=k, replace=False)
    df.loc[df.index[el26[:k]], "Customer_Full_ID"] = chosen
    y = df["Fall_Next_180D_Ind"].fillna(0).to_numpy(dtype=float)
    # new V21-style columns (2026 only): catalogued, never used
    df["Dizziness_Vertigo_Ind"] = ((y + rng.normal(0, 0.25, len(df))) > 0.5).astype(int)          # strongly outcome-related on purpose
    df["Cataract_Dx_Date"] = pd.NaT
    m = rng.random(len(df)) < 0.2
    df.loc[m, "Cataract_Dx_Date"] = idx - pd.to_timedelta(rng.integers(1, 2000, int(m.sum())), unit="D")
    df["Opiate_Registry_Ind"] = (rng.random(len(df)) < 0.05).astype(int)
    df["Fall_Next_365D_Ind"] = df["Fall_Next_180D_Ind"]                                            # sealed by name ("next")
    df["Definition_Version"] = definition_version
    facts = {"n_overlap_planted": int(k), "n_2026_eligible": int(len(el26)), "n_2025_full_labeled": len(set(full25)), "n_2025_eligible": len(elig_any25)}
    return df, facts
