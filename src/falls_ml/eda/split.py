"""Split EDA: train / validation / test compared on size, events, age, sex, the EXTENDED predictors and key missingness.

A diagnostic of the deterministic split (the reference configuration), never a reason to re-split. Predictor distributions are compared
between partitions without outcome stratification (standardised mean differences train vs validation and train vs test); the only
outcome figures are each partition's event count and prevalence, which the split audit already reports.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from falls_ml.eda.common import SPLIT_DIAGNOSTIC, SUPPRESSED, cell, fnum, med_iqr, numeric, small, smd

IMBALANCE_SMD = 0.1
PARTS = ("train", "validation", "test")


def _variables(raw: pd.DataFrame, canonical_names: list[str], sources: dict[str, list[str]]) -> list[tuple[str, pd.Series, str]]:
    out: list[tuple[str, pd.Series, str]] = []
    if "Age_At_Index" in raw.columns:
        out.append(("age (years)", numeric(raw["Age_At_Index"]), "continuous"))
    if "Gender_Code" in raw.columns:
        g = raw["Gender_Code"].astype("string")
        top = g.value_counts().index
        if len(top) > 1:
            out.append((f"sex: Gender_Code = {top[1]}", (g == top[1]).astype("float64").where(g.notna()), "binary"))
    for name in canonical_names:
        cols = [c for c in sources.get(name, []) if c in raw.columns]
        if not cols or name in ("age_years", "sex"):
            continue
        if len(cols) == 1:
            v = numeric(raw[cols[0]])
        else:
            v = pd.concat([numeric(raw[c]).fillna(0) > 0 for c in cols], axis=1).any(axis=1).astype("float64")
        if cols[0] == "Registry_SMI_Level":   # any_positive of the level: NULL = not in the registry (explicit absent_is_zero)
            v = (v.fillna(0) >= 1).astype("float64")
        kind = "binary" if set(v.dropna().unique()) <= {0.0, 1.0} else "continuous"
        out.append((f"{name} ({'; '.join(cols)})", v, kind))
    for col, label in (("Assessment_Not_Performed_Ind", "nurse assessment NOT performed"), ("Frailty_Not_Assessed_Ind", "frailty NOT assessed"),
                       ("Medication_Missing_Ind", "no medication data"), ("Registry_Missing_Ind", "no registry data"),
                       ("Birth_Date_Suspect_Ind", "suspect birth date")):
        if col in raw.columns:
            out.append((label, numeric(raw[col]), "binary"))
    for col in ("HMO_Seniority_Months", "Visit_Count_365D", "Diagnosis_Count_365D"):
        if col in raw.columns:
            out.append((col, numeric(raw[col]), "continuous"))
    return out


def split_balance(raw: pd.DataFrame, partition: pd.Series, *, label_col: str, canonical_names: list[str], sources: dict[str, list[str]],
                  min_cell: int) -> pd.DataFrame:
    """``split_balance.csv``: per variable and partition the distribution, and the SMD of validation / test against train."""
    parts = {p: (partition == p).fillna(False).to_numpy(dtype=bool) for p in PARTS}
    y = numeric(raw[label_col])
    rows: list[dict[str, Any]] = []
    head = {"variable": "rows", "kind": "count"}
    ev = {"variable": "events (fall within 180 days)", "kind": "count"}
    prev = {"variable": "prevalence", "kind": "proportion"}
    for p, m in parts.items():
        n, k = int(m.sum()), int((y[m] == 1).sum())
        head[p], ev[p] = n, cell(k, min_cell)
        prev[p] = fnum(k / n, 4) if n and not small(k, min_cell) else SUPPRESSED
    rows += [head, ev, prev]
    for label, v, kind in _variables(raw, canonical_names, sources):
        r: dict[str, Any] = {"variable": label, "kind": kind}
        for p, m in parts.items():
            x = v[m]
            if kind == "binary":
                k = int((x == 1).sum())
                r[p] = SUPPRESSED if small(k, min_cell) else f"{100.0 * k / max(int(x.notna().sum()), 1):.2f}%"
            else:
                r[p] = med_iqr(x)
            r[f"{p}_missing_pct"] = fnum(100.0 * float(x.isna().mean()), 2) if len(x) else None
        xt = v[parts["train"]].to_numpy(dtype="float64")
        for p in ("validation", "test"):
            s = smd(v[parts[p]].to_numpy(dtype="float64"), xt)
            r[f"smd_{p}_vs_train"] = fnum(s, 3)
        worst = max((abs(r[f"smd_{p}_vs_train"]) for p in ("validation", "test") if r[f"smd_{p}_vs_train"] is not None), default=None)
        r["imbalance_flag"] = bool(worst is not None and worst > IMBALANCE_SMD)
        rows.append(r)
    out = pd.DataFrame(rows)
    out["basis"] = SPLIT_DIAGNOSTIC
    return out
