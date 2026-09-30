"""Stata ``logit``-style handling of constant columns and perfect predictors in unpenalised logistic fits.

Stata's ``logit`` (used by the eFalls authors for the unpenalised FP-selection model and the unpenalised
refit) does not fail on these cases; it notes and omits them:

- a column that is constant in the fitting rows is omitted (collinear with the intercept);
- an indicator whose value ``1`` (or ``0``) occurs only with one outcome class "predicts success/failure
  perfectly": the column is dropped together with the observations it predicts perfectly.

The procedure is repeated until nothing changes, because dropping observations can create new cases.
Every omission is returned (and must be logged/recorded by the caller) — nothing is silent (spec D-13).
Only indicator-like columns (values within {0, 1}) are checked for perfect prediction; separation involving
continuous columns still surfaces as a fitting error in the calling solver.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from falls_ml.errors import DegenerateFitError


class SeparationError(DegenerateFitError):
    """After Stata-style omissions the fitting rows contain a single outcome class."""


@dataclass(frozen=True)
class OmissionReport:
    kept_columns: tuple[int, ...]
    kept_rows: np.ndarray                      # boolean mask over the input rows
    constant_columns: tuple[str, ...] = ()
    perfect_predictors: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    @property
    def n_rows_dropped(self) -> int:
        return int((~self.kept_rows).sum())

    def to_dict(self) -> dict[str, Any]:
        return {"constant_columns": list(self.constant_columns), "perfect_predictors": [dict(p) for p in self.perfect_predictors],
                "n_rows_dropped": self.n_rows_dropped}

    @property
    def any(self) -> bool:
        return bool(self.constant_columns or self.perfect_predictors)


def stata_logit_omissions(X: np.ndarray, y: np.ndarray, names: Sequence[str]) -> OmissionReport:
    """Return the columns and rows Stata ``logit`` would keep for ``y ~ X`` (see module docstring)."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    if X.ndim != 2 or X.shape[0] != y.shape[0] or X.shape[1] != len(names):
        raise ValueError(f"stata_logit_omissions: incompatible shapes X={X.shape}, y={y.shape}, names={len(names)}")
    rows = np.ones(X.shape[0], dtype=bool)
    active = list(range(X.shape[1]))
    constant: list[str] = []
    perfect: list[dict[str, Any]] = []
    changed = True
    while changed:
        changed = False
        if np.unique(y[rows]).size < 2:
            raise SeparationError("after omitting perfect predictors the fitting rows contain a single outcome class")
        for j in list(active):
            col = X[rows, j]
            values = np.unique(col)
            if values.size <= 1:
                active.remove(j)
                constant.append(str(names[j]))
                changed = True
                continue
            if values.size == 2 and set(values.tolist()) <= {0.0, 1.0}:
                yr = y[rows]
                for level in (1.0, 0.0):
                    at_level = col == level
                    outcomes = np.unique(yr[at_level])
                    if outcomes.size == 1:
                        idx = np.flatnonzero(rows)[at_level]
                        rows[idx] = False
                        active.remove(j)
                        perfect.append({"column": str(names[j]), "level": int(level), "predicts_outcome": int(outcomes[0]),
                                        "n_rows_dropped": int(idx.size)})
                        changed = True
                        break
                if changed:
                    break
        # restart the scan after any change so later checks see the reduced rows
    return OmissionReport(kept_columns=tuple(active), kept_rows=rows, constant_columns=tuple(constant), perfect_predictors=tuple(perfect))
