"""SYNTHETIC 2026 / V21 extracts for the Phase 5 tests and the smoke run (never real data - SYNTHETIC DATA, NOT SCIENTIFIC RESULTS).

The Phase 1 synthetic wide extract on Index_Date 2026-01-01 under the corrected contract, converted to the EXACT authoritative V21 header (224 columns,
the order of ``configs/meuhedet/phase5_v21_schema.yaml``): the 20 removed V1 columns are dropped and the new V21 columns added (the COVID-19 /
dialysis / immunosuppression registries are independent new noise columns - their lineage to the removed V1 registries is NOT proven):

    Dizziness_Ind                 the NEW signal: outcome-related in the "planted" scenario, pure noise in the "null" scenario
    other diagnosis / registry    noise (diagnosis flags only where a diagnosis record exists: Last_Dx_Date bounds them)
    Registry_*_SubCode            raw sub-codes of the smoking / obesity registries (noise)
    Deficit_Count_Proxy           a count of the NEW flags (UNCERTAIN_TIMING by the schema)

Optional traps (``traps``):
    "unknown_column"   an extra clinical-looking column the authoritative schema does not define  -> REQUIRES_SEMANTIC_REVIEW -> STOP
    "future_column"    an extra future-outcome column (Fall_Next_365D_Ind)                        -> sealed by name, never read into X
    "leaky_new"        Deficit_Count_Proxy almost equal to the label                             -> INELIGIBLE_LEAKAGE (excluded)
    "post_index_dx"    Last_Dx_Date after the index day on many diagnosed fallers                -> diagnosis features UNKNOWN / INELIGIBLE_TIMING
    "missing_column"   one authoritative V21 column (Tremor_Ind) absent from the extract         -> WARN; the feature is not available
    "positive_after_followup"  a few positives whose event lies after their personal Followup_End_Date -> O4 zero tolerance -> STOP
    Phase 5.1 boundary traps (docs/phase6/EXPERIMENT1_IMPLEMENTATION_CONTRACT.md N-1):
    "forbidden_lineage"  an UNDECLARED predictor-looking column (Fall_Risk_Flag_Ind) that is in fact outcome-derived -> REQUIRES_SEMANTIC_REVIEW
                         -> STOP (fails closed; it never enters X) - unknown fields are never admitted on the strength of a name
    "weak_proxy"         Osteoporosis_Ind replaced by a NOISY outcome proxy (univariate AUROC ~0.65): an admissible-looking column that no numeric
                         gate catches -> RETAINED (documents that provenance, not a number, is the defence; the forensic value is reported < 0.80)
    "strong_legit"       Gait_Abnormality_Ind planted as a STRONG legitimate predictor (AUROC ~0.86) -> RETAINED, forensic WARN, no exclusion
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCENARIOS = ("planted", "null")
TRAPS = ("unknown_column", "future_column", "leaky_new", "post_index_dx", "missing_column", "positive_after_followup", "forbidden_lineage", "weak_proxy",
         "strong_legit")
DX_PREV = {"Dizziness_Ind": None, "Gait_Abnormality_Ind": 0.04, "Syncope_Ind": 0.03, "Tremor_Ind": 0.03, "Cataract_Ind": 0.15, "Hearing_Loss_Dx_Ind": 0.08,
           "Vision_Impairment_Dx_Ind": 0.06, "Osteoporosis_Ind": 0.10, "Parkinsonism_Ind": 0.02, "Stroke_Dx_Ind": 0.05}
REG_PREV = {"Registry_Smoking_Ind": 0.08, "Registry_Obesity_Ind": 0.15, "Registry_Oncology_Ind": 0.06, "Registry_IBD_Ind": 0.02, "Registry_Opiate_Ind": 0.04,
            "Registry_Severe_Function_Ind": 0.02, "Registry_Corona_Ind": 0.05, "Registry_Dialysis_Ind": 0.01, "Registry_Immunosuppressant_Ind": 0.02}


def v21_header() -> list[str]:
    from falls_ml.phase5.schema import load_v21_schema

    return list(load_v21_schema("configs/meuhedet/phase5_v21_schema.yaml").header)


def make_v21(n_rows: int = 4000, *, seed: int = 26, scenario: str = "planted", index_date: str = "2026-01-01", signal: float = 0.32,
             base_rate: float = 0.06, index_day_outcomes: bool = False, traps: tuple[str, ...] = ()) -> tuple[pd.DataFrame, dict[str, Any]]:
    from falls_ml.data.meuhedet_synthetic import generate_synthetic_wide_extract
    from falls_ml.phase3.synthetic import plant_phase3_history

    if scenario not in SCENARIOS:
        raise ValueError(f"scenario must be one of {SCENARIOS}")
    bad = [t for t in traps if t not in TRAPS]
    if bad:
        raise ValueError(f"unknown traps {bad}; allowed {TRAPS}")
    df = generate_synthetic_wide_extract(n_rows, seed=seed, index_date=index_date, n_index_day_falls=4)
    df, _ = plant_phase3_history(df, index_date=index_date, seed=seed, index_day_outcomes=index_day_outcomes)
    df = df.reset_index(drop=True)
    df["Customer_Full_ID"] = [f"V{i:010d}" for i in range(len(df))]
    rng = np.random.default_rng(seed + 1)
    idx = pd.Timestamp(index_date)
    n = len(df)
    y = pd.to_numeric(df["Fall_Next_180D_Ind"], errors="coerce").fillna(0).to_numpy(dtype=float)
    has_dx = pd.to_datetime(df["Last_Dx_Date"], errors="coerce").notna().to_numpy()
    df["Diagnosis_Source_Absent_Ind"] = (~has_dx).astype(int)
    for c, prev in DX_PREV.items():
        p = (base_rate + (signal * y if scenario == "planted" else 0.0)) if prev is None else np.full(n, prev)
        df[c] = ((rng.random(n) < p) & has_dx).astype(int)
    for c, prev in REG_PREV.items():
        df[c] = (rng.random(n) < prev).astype(int)
    df["Registry_Smoking_SubCode"] = np.where(df["Registry_Smoking_Ind"] == 1, rng.integers(1, 4, n), np.nan)
    df["Registry_Obesity_SubCode"] = np.where(df["Registry_Obesity_Ind"] == 1, rng.integers(1, 3, n), np.nan)
    # the proxy sums only NEW flags here, so the "null" world carries no information beyond OLD (the real proxy also re-uses OLD registry / fall
    # flags - its own domain test and ablation isolate that)
    df["Deficit_Count_Proxy"] = sum(pd.to_numeric(df[c], errors="coerce").fillna(0).clip(0, 1) for c in [*DX_PREV, *REG_PREV]).astype(int)
    facts: dict[str, Any] = {"scenario": scenario, "n_rows": n, "planted_feature": "Dizziness_Ind" if scenario == "planted" else None, "traps": list(traps)}
    if "leaky_new" in traps:
        df["Deficit_Count_Proxy"] = np.where(rng.random(n) < 0.03, 1 - y, y).astype(int)
        facts["leaky_feature"] = "Deficit_Count_Proxy"
    if "weak_proxy" in traps:                       # P(x=1|y=1)=0.45, P(x=1|y=0)=0.15 -> univariate AUROC ~0.65 (below any numeric gate)
        df["Osteoporosis_Ind"] = ((rng.random(n) < np.where(y == 1, 0.45, 0.15)) & has_dx).astype(int)
        facts["weak_proxy_feature"] = "Osteoporosis_Ind"
    if "strong_legit" in traps:                     # P(x=1|y=1)=0.80, P(x=1|y=0)=0.08 -> univariate AUROC ~0.86 (legitimate, admissible, strong)
        df["Gait_Abnormality_Ind"] = ((rng.random(n) < np.where(y == 1, 0.80, 0.08)) & has_dx).astype(int)
        facts["strong_legit_feature"] = "Gait_Abnormality_Ind"
    if "post_index_dx" in traps:
        ld = pd.to_datetime(df["Last_Dx_Date"], errors="coerce")
        late = has_dx & (rng.random(n) < 0.10 + 0.6 * y)
        df["Last_Dx_Date"] = ld.where(~late, idx + pd.to_timedelta(rng.integers(1, 120, n), unit="D"))
        facts["post_index_rows"] = int(late.sum())
    if "positive_after_followup" in traps:
        ev = pd.to_datetime(df["Next_Fall_Date_180D"], errors="coerce")
        cand = np.flatnonzero((y == 1) & ev.notna().to_numpy() & ((ev - idx).dt.days > 2).to_numpy())[:7]
        fe = pd.to_datetime(df["Followup_End_Date"], errors="coerce")
        fe.iloc[cand] = ev.iloc[cand] - pd.Timedelta(days=1)
        df["Followup_End_Date"] = fe
        facts["positives_after_followup"] = int(len(cand))
    cols = v21_header()
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"synthetic V21 builder lacks columns {missing}")
    out = df[cols].copy()
    if "missing_column" in traps:
        out = out.drop(columns=["Tremor_Ind"])
        facts["missing_column"] = "Tremor_Ind"
    if "unknown_column" in traps:
        out["Balance_Clinic_Referral_Ind"] = (rng.random(n) < 0.1).astype(int)
        facts["unknown_column"] = "Balance_Clinic_Referral_Ind"
    if "future_column" in traps:
        out["Fall_Next_365D_Ind"] = df["Fall_Next_180D_Ind"]
        facts["future_column"] = "Fall_Next_365D_Ind"
    if "forbidden_lineage" in traps:                # outcome-derived under a predictor-like, undeclared name: must fail closed (semantic review STOP)
        out["Fall_Risk_Flag_Ind"] = np.where(rng.random(n) < 0.02, 1 - y, y).astype(int)
        facts["forbidden_lineage_column"] = "Fall_Risk_Flag_Ind"
    return out, facts


def write_v21_csv(df: pd.DataFrame, path: Path, *, seed: int = 11) -> Path:
    """CSV like the real SSMS export ('NULL' literals, the declared date layouts), via the Phase 3 synthetic writer."""
    from falls_ml.phase3.synthetic import write_phase3_csv

    return write_phase3_csv(df, Path(path), seed=seed)
