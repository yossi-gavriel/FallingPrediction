"""Design matrices learned on TRAINING rows only (every inner fold and every outer refit fits its own): the same rules for every feature set and
every model family. Nothing is learned from the rows a design is applied to.

Linear (LASSO / elastic net):
    binary          1 if the value is 1, else 0
    thermometer     ordinal levels: one column ">= level" per level above the lowest
    fall bands      days since the last fall: <=90 / 91-180 / 181-365 / >365 (reference: no prior fall)
    onehot          declared categorical levels (reference: the first level); a categorical without declared levels (raw, unvalidated codes)
                    learns its levels on the TRAINING rows (codes seen >= 10 times; others fall into the reference)
    log1p / none    log(1 + max(x, 0)) or the value; NULL -> the TRAINING median
    + one "<feature>__na" indicator per feature with any NULL in the training rows (NULL = not assessed / nothing recorded is information,
      never silently 0); exact duplicate and constant training columns are dropped (label-blind numerical safeguard); every column is then
      standardised with the TRAINING mean / sd.
Trees (XGBoost): the engineered values with NULL kept (native missing-value handling); declared categorical levels one-hot (raw codes without declared
levels are kept as one numeric column - a tree can split on them).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

FALL_BANDS = (90, 180, 365)
MIN_LEVEL_COUNT = 10


def _encode(f: str, m: dict[str, Any], x: np.ndarray, levels: list[float] | None = None) -> tuple[list[str], np.ndarray, bool]:
    """(column names, matrix before NULL filling, needs a median fill)."""
    obs = np.isfinite(x)
    lin, kind = m.get("linear", "none"), m.get("kind", "continuous")
    if lin == "thermometer":
        lv = sorted(float(v) for v in (m.get("levels") or []))[1:]
        if not lv:
            return [f], np.where(obs, x, np.nan)[:, None], True
        return [f"{f}__ge{v:g}" for v in lv], np.column_stack([np.where(obs, (x >= v).astype(float), 0.0) for v in lv]), False
    if lin == "fall_recency_bands":
        edges = [-np.inf, *FALL_BANDS, np.inf]
        labels = [f"le{FALL_BANDS[0]}", *[f"{a + 1}_{b}" for a, b in zip(FALL_BANDS[:-1], FALL_BANDS[1:])], f"gt{FALL_BANDS[-1]}"]
        xx = np.where(obs, x, np.nan)
        with np.errstate(invalid="ignore"):
            mat = np.column_stack([((xx > lo) & (xx <= hi)).astype(float) for lo, hi in zip(edges[:-1], edges[1:])])
        return [f"{f}__{lab}" for lab in labels], mat, False
    if lin == "onehot":
        lv = [float(v) for v in (levels if levels is not None else (m.get("levels") or []))][1:]
        return [f"{f}__eq{v:g}" for v in lv], np.column_stack([np.where(obs, (x == v).astype(float), 0.0) for v in lv]) if lv else np.empty((len(x), 0)), False
    if kind == "binary":
        return [f], np.where(obs, (x == 1.0).astype(float), 0.0)[:, None], False
    v = np.log1p(np.clip(x, 0.0, None)) if lin == "log1p" else x.astype(float)
    return [f], np.where(obs, v, np.nan)[:, None], True


class LinearDesign:
    def __init__(self, features: list[str], meta: dict[str, dict[str, Any]]):
        self.features = list(features)
        self.meta = {f: meta[f] for f in self.features}
        self.medians_: dict[str, float] = {}
        self.na_: list[str] = []
        self.keep_: list[int] = []
        self.mean_: np.ndarray | None = None
        self.sd_: np.ndarray | None = None
        self.columns_: list[str] = []
        self.feature_of_: dict[str, str] = {}
        self.levels_: dict[str, list[float]] = {}

    def _raw(self, X: pd.DataFrame) -> tuple[list[str], np.ndarray]:
        names, parts = [], []
        for f in self.features:
            x = X[f].to_numpy(dtype=float)
            cols, mat, fill = _encode(f, self.meta[f], x, self.levels_.get(f))
            if fill and mat.shape[1]:
                mat = np.where(np.isfinite(mat), mat, self.medians_.get(f, 0.0))
            names += cols
            parts.append(mat)
        for f in self.na_:
            names.append(f"{f}__na")
            parts.append((~np.isfinite(X[f].to_numpy(dtype=float))).astype(float)[:, None])
        return names, (np.column_stack(parts) if parts else np.empty((len(X), 0)))

    def fit(self, X: pd.DataFrame) -> LinearDesign:
        self.medians_, self.na_, self.levels_ = {}, [], {}
        for f in self.features:
            x = X[f].to_numpy(dtype=float)
            m = self.meta[f]
            if m.get("linear") == "onehot" and not m.get("levels"):
                v, cnt = np.unique(x[np.isfinite(x)], return_counts=True)
                self.levels_[f] = [float(a) for a, k in zip(v, cnt) if k >= MIN_LEVEL_COUNT]
            cols, mat, fill = _encode(f, m, x, self.levels_.get(f))
            if fill and mat.shape[1]:
                v = mat[:, 0]
                self.medians_[f] = float(np.nanmedian(v)) if np.isfinite(v).any() else 0.0
            if (~np.isfinite(x)).any():
                self.na_.append(f)
        names, M = self._raw(X)
        keep, seen = [], set()
        for j in range(M.shape[1]):
            col = M[:, j]
            if np.nanmax(col) == np.nanmin(col):
                continue
            key = np.ascontiguousarray(col).tobytes()
            if key in seen:
                continue
            seen.add(key)
            keep.append(j)
        self.keep_ = keep
        K = M[:, keep]
        self.mean_ = K.mean(axis=0)
        sd = K.std(axis=0)
        self.sd_ = np.where(sd > 0, sd, 1.0)
        self.columns_ = [names[j] for j in keep]
        self.feature_of_ = {c: (c.split("__", 1)[0]) for c in self.columns_}
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        _, M = self._raw(X)
        return (M[:, self.keep_] - self.mean_) / self.sd_


class TreeDesign:
    def __init__(self, features: list[str], meta: dict[str, dict[str, Any]]):
        self.features = list(features)
        self.meta = {f: meta[f] for f in self.features}
        self.columns_: list[str] = []
        self.feature_of_: dict[str, str] = {}

    def fit(self, X: pd.DataFrame) -> TreeDesign:
        cols = []
        for f in self.features:
            m = self.meta[f]
            if m.get("linear") == "onehot" and m.get("levels"):
                cols += [f"{f}__eq{float(v):g}" for v in (m.get("levels") or [])]
            else:
                cols.append(f)
        self.columns_ = cols
        self.feature_of_ = {c: c.split("__", 1)[0] for c in cols}
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        parts = []
        for f in self.features:
            m = self.meta[f]
            x = X[f].to_numpy(dtype=float)
            if m.get("linear") == "onehot" and m.get("levels"):
                lv = [float(v) for v in (m.get("levels") or [])]
                parts += [np.where(np.isfinite(x), (x == v).astype(float), np.nan) for v in lv]
            else:
                parts.append(x)
        return np.column_stack(parts).astype(np.float32) if parts else np.empty((len(X), 0), dtype=np.float32)
