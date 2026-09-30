"""Design matrices fitted inside each fold (planning/EXPERIMENT_PLAN.md §3).

Linear: the BASELINE_15 block comes from the unchanged eFalls preprocessor (``efalls_fp_all_levels``: FP selection for age and polypharmacy,
fitted on the fold's training rows with its outcome), followed by the Phase 2 block (label-blind encodings frozen in FEATURE_SETS.json;
fold medians for NULL fills) and the na__ indicators of the NULL-pattern groups. Trees: raw values, NULL kept.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from falls_ml.config import FractionalPolynomialSection
from falls_ml.features.preprocessing import Preprocessor

FP_VARIABLES = ("age_years", "polypharmacy_count_120d")


def _encode(name: str, spec: dict[str, Any], x: np.ndarray) -> tuple[list[str], np.ndarray]:
    """Columns for one Phase 2 feature before NULL filling (NULL stays NaN in continuous columns, 0 in indicator-type columns)."""
    obs = np.isfinite(x)
    lin = spec["linear"]
    if lin == "thermometer":
        levels = spec.get("levels_used") or []
        cols = [f"{name}__ge{lv:g}" for lv in levels]
        mat = np.column_stack([np.where(obs, (x >= lv).astype(float), 0.0) for lv in levels]) if levels else np.empty((len(x), 0))
        return cols, mat
    if lin == "fall_recency_bands":
        b = list(spec.get("bands") or [90, 180, 365])
        edges = [-np.inf, *b, np.inf]
        labels = [f"le{b[0]}", *[f"{lo + 1}_{hi}" for lo, hi in zip(b[:-1], b[1:])], f"gt{b[-1]}"]
        xx = np.where(obs, x, np.nan)
        with np.errstate(invalid="ignore"):
            mat = np.column_stack([((xx > lo) & (xx <= hi)).astype(float) for lo, hi in zip(edges[:-1], edges[1:])])
        return [f"{name}__{lab}" for lab in labels], mat
    if spec["kind"] == "binary":
        return [name], np.where(obs, (x == 1.0).astype(float), 0.0)[:, None]
    v = np.log1p(np.clip(x, 0.0, None)) if lin == "log1p" else x.astype(float)
    return [name], v[:, None]


class LinearDesign:
    def __init__(self, new: list[str], baseline: list[str], design: dict[str, Any], efalls_spec: Any):
        self.new, self.baseline = list(new), list(baseline)
        self.fspec = {f: design["features"][f] for f in self.new}
        groups = design.get("na_groups", {})
        self.na_groups = {g: [m for m in v["members"] if m in set(self.new)] for g, v in groups.items()}
        self.na_groups = {g: m for g, m in self.na_groups.items() if m}
        self.efalls_spec = efalls_spec.subset(self.baseline) if self.baseline else None
        self.pre: Preprocessor | None = None
        self.medians_: dict[str, float] = {}
        self.columns_: list[str] = []
        self.duplicates_: dict[str, str] = {}
        self.feature_of_: dict[str, str] = {}

    def fit(self, frame: pd.DataFrame, y: np.ndarray) -> LinearDesign:
        if self.efalls_spec is not None:
            fpv = tuple(v for v in FP_VARIABLES if v in self.baseline)
            if fpv:
                self.pre = Preprocessor(self.efalls_spec, "efalls_fp_all_levels", fp=FractionalPolynomialSection(mode="select", variables=fpv))
            else:
                self.pre = Preprocessor(self.efalls_spec, "efalls_reference_coded")
            self.pre.fit(frame[self.baseline], np.asarray(y))
        self.medians_ = {}
        for f in self.new:
            s = self.fspec[f]
            if s["fill"] == "median":
                cols, mat = _encode(f, s, frame[f].to_numpy(dtype=float))
                v = mat[:, 0]
                self.medians_[f] = float(np.nanmedian(v)) if np.isfinite(v).any() else 0.0
        self.duplicates_ = {}
        X = self.transform(frame)
        # exact duplicate columns (e.g. two forms always assessed together) make the penalised solution non-unique and stall coordinate
        # descent: keep the first, record the rest (label-blind numerical safeguard)
        seen: dict[bytes, str] = {}
        for col in X.columns:
            key = np.ascontiguousarray(X[col].to_numpy(dtype=np.float64)).tobytes()
            if key in seen:
                self.duplicates_[col] = seen[key]
            else:
                seen[key] = col
        self.columns_ = [c for c in X.columns if c not in self.duplicates_]
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        parts, names = [], []
        feature_of: dict[str, str] = {}
        if self.pre is not None:
            B = self.pre.transform(frame[self.baseline])
            parts.append(B.to_numpy())
            names += list(B.columns)
            feature_of.update({c: self.pre.raw_feature_of(c) for c in B.columns})
        for f in self.new:
            s = self.fspec[f]
            cols, mat = _encode(f, s, frame[f].to_numpy(dtype=float))
            if s["fill"] == "median" and mat.shape[1]:
                mat = np.where(np.isfinite(mat), mat, self.medians_.get(f, 0.0))
            parts.append(mat)
            names += cols
            feature_of.update({c: f for c in cols})
        for g, members in self.na_groups.items():
            parts.append(frame[members[0]].isna().to_numpy(dtype=float)[:, None])
            names.append(g)
            feature_of[g] = g
        self.feature_of_ = feature_of
        X = np.column_stack(parts) if parts else np.empty((len(frame), 0))
        out = pd.DataFrame(X, index=frame.index, columns=names)
        return out[self.columns_] if self.columns_ else out


def tree_matrix(frame: pd.DataFrame, new: list[str], baseline: list[str]) -> pd.DataFrame:
    """Raw values for XGBoost: BASELINE_15 as numbers (sex -> female 0/1), Phase 2 features as engineered (NULL kept)."""
    cols: dict[str, np.ndarray] = {}
    for f in baseline:
        s = frame[f]
        cols["sex_female" if f == "sex" else f] = (s.astype(str) == "female").to_numpy(dtype=float) if f == "sex" else pd.to_numeric(s).to_numpy(dtype=float)
    for f in new:
        cols[f] = frame[f].to_numpy(dtype=float)
    return pd.DataFrame(cols, index=frame.index)


def tree_feature_of(column: str) -> str:
    return "sex" if column == "sex_female" else column
