"""SYNTHETIC 2026 / V21 extracts for the Phase 5 tests and the smoke run (never real data - SYNTHETIC DATA, NOT SCIENTIFIC RESULTS).

The Phase 1 synthetic wide extract on Index_Date 2026-01-01 under the corrected contract, with V21-style columns:

    Dizziness_Ind            the NEW signal: outcome-related in the "planted" scenario, pure noise in the "null" scenario
    Registry_Opiate_Ind      noise (attested registry status)
    Cataract_Ind             noise
    Syncope_Ind + Syncope_Date   UNSAFE: many records dated AFTER Index_Date, preferentially for the patients who fall (must be INELIGIBLE_TIMING)
    Gait_Abnormality_Ind     LEAKY: almost the label itself (must be INELIGIBLE_LEAKAGE)
    Fall_Next_365D_Ind       future outcome (sealed by name; never read into X)
    Hospital_Discharge_Date  post-index care information (sealed by name)
    Mystery_Score_V21        undeclared (INELIGIBLE_SEMANTICS)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCENARIOS = ("planted", "null")


def make_v21(n_rows: int = 4000, *, seed: int = 26, scenario: str = "planted", index_date: str = "2026-01-01", definition_version: str = "V21",
             signal: float = 0.32, base_rate: float = 0.06, index_day_outcomes: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
    from falls_ml.data.meuhedet_synthetic import generate_synthetic_wide_extract
    from falls_ml.phase3.synthetic import plant_phase3_history

    if scenario not in SCENARIOS:
        raise ValueError(f"scenario must be one of {SCENARIOS}")
    df = generate_synthetic_wide_extract(n_rows, seed=seed, index_date=index_date, n_index_day_falls=4)
    df, _ = plant_phase3_history(df, index_date=index_date, seed=seed, index_day_outcomes=index_day_outcomes)
    df = df.reset_index(drop=True)
    df["Customer_Full_ID"] = [f"V{i:010d}" for i in range(len(df))]
    df["Definition_Version"] = definition_version
    rng = np.random.default_rng(seed + 1)
    idx = pd.Timestamp(index_date)
    n = len(df)
    y = df["Fall_Next_180D_Ind"].fillna(0).to_numpy(dtype=float)
    p = base_rate + (signal * y if scenario == "planted" else 0.0)
    df["Dizziness_Ind"] = (rng.random(n) < p).astype(int)
    df["Registry_Opiate_Ind"] = (rng.random(n) < 0.05).astype(int)
    df["Cataract_Ind"] = (rng.random(n) < 0.12).astype(int)
    syn = rng.random(n) < 0.10 + 0.30 * y
    df["Syncope_Ind"] = syn.astype(int)
    late = syn & (rng.random(n) < 0.15 + 0.6 * y)
    d = idx - pd.to_timedelta(rng.integers(1, 1500, n), unit="D")
    d = d.where(~late, idx + pd.to_timedelta(rng.integers(1, 120, n), unit="D"))
    df["Syncope_Date"] = pd.Series(np.where(syn, d.to_numpy(dtype="datetime64[us]"), np.datetime64("NaT", "us")), dtype="datetime64[us]")
    df["Gait_Abnormality_Ind"] = np.where(rng.random(n) < 0.03, 1 - y, y).astype(int)
    df["Fall_Next_365D_Ind"] = df["Fall_Next_180D_Ind"]
    dis = rng.random(n) < 0.05
    df["Hospital_Discharge_Date"] = pd.Series(np.where(dis, (idx + pd.to_timedelta(rng.integers(1, 90, n), unit="D")).to_numpy(dtype="datetime64[us]"),
                                                       np.datetime64("NaT", "us")), dtype="datetime64[us]")
    df["Mystery_Score_V21"] = rng.integers(0, 5, n)
    facts = {"scenario": scenario, "n_rows": n, "planted_feature": "Dizziness_Ind" if scenario == "planted" else None,
             "unsafe_feature": "Syncope_Ind", "leaky_feature": "Gait_Abnormality_Ind", "sealed_new": ["Fall_Next_365D_Ind", "Hospital_Discharge_Date"],
             "undeclared": ["Mystery_Score_V21"]}
    return df, facts


def write_v21_csv(df: pd.DataFrame, path: Path, *, seed: int = 11) -> Path:
    """CSV like the real SSMS export ('NULL' literals, the declared date layouts), via the Phase 3 synthetic writer."""
    from falls_ml.phase3.synthetic import write_phase3_csv

    return write_phase3_csv(df, Path(path), seed=seed)
