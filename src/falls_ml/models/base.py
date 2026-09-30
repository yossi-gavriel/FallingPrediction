"""Model adapter contract (architecture §3.2).

Every algorithm — including the fixed published eFalls equation — implements this interface, so
``run_experiment`` never branches on the algorithm. Adapters receive the design matrix produced by
the fitted ``Preprocessor`` (a DataFrame with named columns) and never see identifiers, dates or
outcome columns other than ``y``.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar, Self

import numpy as np
import pandas as pd

from falls_ml.errors import NotFittedError

IMPORTANCE_COLUMNS = ["feature", "coefficient", "odds_ratio", "abs_coefficient", "selected", "direction", "native_importance"]


class ModelAdapter(ABC):
    #: registry name, e.g. "lasso_logistic_cv"
    name: ClassVar[str]
    #: Preprocessor representation consumed by this adapter
    representation: ClassVar[str]
    #: True for models whose predictions are expit(linear predictor)
    is_linear: ClassVar[bool] = False
    #: False only for fixed published equations
    requires_fit: ClassVar[bool] = True

    def __init__(self, params: Mapping[str, Any] | None = None, *, random_state: int = 0):
        self.params: dict[str, Any] = dict(params or {})
        self.random_state = int(random_state)
        self.feature_names_: list[str] | None = None

    # ------------------------------------------------------------------ core
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: np.ndarray, *, groups: np.ndarray | None = None) -> Self: ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return P(outcome = 1) with shape (n,)."""

    def linear_predictor(self, X: pd.DataFrame) -> np.ndarray | None:
        return None

    @abstractmethod
    def get_feature_importance(self) -> pd.DataFrame:
        """Native feature importance with columns ``IMPORTANCE_COLUMNS`` (NaN where not applicable)."""

    def fit_diagnostics(self) -> dict[str, Any]:
        return {}

    def get_params(self) -> dict[str, Any]:
        return dict(self.params)

    # ------------------------------------------------------------------ persistence
    @abstractmethod
    def save(self, directory: Path) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, directory: Path) -> Self: ...

    @classmethod
    def default_search_space(cls) -> dict[str, list[Any]] | None:
        return None

    # ------------------------------------------------------------------ helpers for subclasses
    def _check_fitted(self) -> None:
        if self.feature_names_ is None:
            raise NotFittedError(f"{self.name} has not been fitted")

    def _check_columns(self, X: pd.DataFrame) -> None:
        self._check_fitted()
        if list(X.columns) != self.feature_names_:
            missing = [c for c in self.feature_names_ if c not in X.columns]  # type: ignore[union-attr]
            extra = [c for c in X.columns if c not in self.feature_names_]  # type: ignore[operator]
            from falls_ml.errors import PreprocessingMismatchError
            raise PreprocessingMismatchError(
                f"{self.name}: design matrix columns differ from training (missing={missing[:5]}, extra={extra[:5]}, or order changed)")

    def _write_meta(self, directory: Path, extra: Mapping[str, Any] | None = None) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        meta = {"adapter": self.name, "representation": self.representation, "params": self.params,
                "random_state": self.random_state, "feature_names": self.feature_names_, **(extra or {})}
        (directory / "adapter.json").write_text(json.dumps(meta, indent=2, default=_json_default), encoding="utf-8", newline="\n")

    @staticmethod
    def _read_meta(directory: Path) -> dict[str, Any]:
        return json.loads((directory / "adapter.json").read_text(encoding="utf-8"))


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def empty_importance(features: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"feature": features, "coefficient": np.nan, "odds_ratio": np.nan, "abs_coefficient": np.nan,
                         "selected": pd.array([pd.NA] * len(features), dtype="boolean"), "direction": "",
                         "native_importance": np.nan})[IMPORTANCE_COLUMNS]


def coefficient_importance(features: list[str], coefs: np.ndarray, *, standardized_scale: bool) -> pd.DataFrame:
    """Coefficient table for linear models. ``native_importance`` is |coef| only when features share a scale."""
    coefs = np.asarray(coefs, dtype=float)
    df = pd.DataFrame({
        "feature": features,
        "coefficient": coefs,
        "odds_ratio": np.exp(coefs),
        "abs_coefficient": np.abs(coefs),
        "selected": pd.array(coefs != 0.0, dtype="boolean"),
        "direction": np.where(coefs > 0, "increases_risk", np.where(coefs < 0, "decreases_risk", "none")),
        "native_importance": np.abs(coefs) if standardized_scale else np.nan,
    })
    return df[IMPORTANCE_COLUMNS]
