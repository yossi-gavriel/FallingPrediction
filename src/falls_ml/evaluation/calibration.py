"""Model recalibration (spec §7.4, D-07, D-19; validation_methods.md §10).

Recalibrators are fitted on validation data only (the orchestrator enforces which rows they see) and act on
the linear predictor LP (``logit(p)`` when no LP is supplied):

- ``none``: p' = p.
- ``intercept_only``: logit p' = LP + delta, delta = CITL (offset GLM). Apparent CITL after updating is 0.
- ``logistic_intercept_slope``: logit p' = alpha + beta·LP (GLM y ~ LP). Apparent slope 1, CITL 0, O/E 1.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Self

import numpy as np

from falls_ml.errors import ConfigError, FallsMLError, NotFittedError
from falls_ml.evaluation.metrics import calibration_slope_intercept, citl, expit, prepare_inputs, prepare_predictions


class RecalibrationError(FallsMLError):
    """A recalibration model cannot be estimated from the supplied data."""


class Recalibrator(ABC):
    """Maps uncalibrated risks (and LP) to recalibrated risks."""

    method: ClassVar[str]
    param_names: ClassVar[tuple[str, ...]]

    def __init__(self) -> None:
        self.fitted_on_n: int | None = None
        self._params: dict[str, float] | None = None

    @property
    def is_fitted(self) -> bool:
        return self._params is not None

    def fit(self, p: Any, y: Any, lp: Any | None = None) -> Self:
        """Estimate parameters from (p, y[, lp]); ``lp`` defaults to logit(p)."""
        yy, _, ll = prepare_inputs(y, p, lp)
        self._params = self._estimate(yy, ll)
        self.fitted_on_n = int(yy.size)
        return self

    def transform(self, p: Any, lp: Any | None = None) -> np.ndarray:
        """Recalibrated risks for (p[, lp])."""
        if not self.is_fitted:
            raise NotFittedError(f"{type(self).__name__} must be fitted before transform")
        return self._apply(*prepare_predictions(p, lp))

    def params(self) -> dict[str, float]:
        if not self.is_fitted:
            raise NotFittedError(f"{type(self).__name__} is not fitted")
        return dict(self._params)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable state: ``{"method", "params", "fitted_on_n"}`` (calibrator.json)."""
        return {"method": self.method, "params": self.params(), "fitted_on_n": self.fitted_on_n}

    @abstractmethod
    def _estimate(self, y: np.ndarray, lp: np.ndarray) -> dict[str, float]: ...

    @abstractmethod
    def _apply(self, p: np.ndarray, lp: np.ndarray) -> np.ndarray: ...


class NoRecalibration(Recalibrator):
    """Identity mapping; needs no data, so it counts as fitted from construction."""

    method = "none"
    param_names = ()

    def __init__(self) -> None:
        super().__init__()
        self._params = {}

    def _estimate(self, y: np.ndarray, lp: np.ndarray) -> dict[str, float]:
        return {}

    def _apply(self, p: np.ndarray, lp: np.ndarray) -> np.ndarray:
        return p.copy()


class InterceptOnlyRecalibrator(Recalibrator):
    """Recalibration-in-the-large: logit p' = LP + delta with delta = CITL."""

    method = "intercept_only"
    param_names = ("delta",)

    def _estimate(self, y: np.ndarray, lp: np.ndarray) -> dict[str, float]:
        delta, _ = citl(y, lp)
        if math.isnan(delta):
            raise RecalibrationError("intercept-only recalibration needs both outcome classes in the fitting data")
        return {"delta": delta}

    def _apply(self, p: np.ndarray, lp: np.ndarray) -> np.ndarray:
        return np.asarray(expit(lp + self._params["delta"]))  # type: ignore[index]


class LogisticRecalibrator(Recalibrator):
    """Logistic recalibration logit p' = alpha + beta·LP; construct with ``alpha`` and ``beta`` to fix them."""

    method = "logistic_intercept_slope"
    param_names = ("alpha", "beta")

    def __init__(self, alpha: float | None = None, beta: float | None = None) -> None:
        super().__init__()
        if (alpha is None) != (beta is None):
            raise ValueError("give both alpha and beta, or neither")
        if alpha is not None and beta is not None:
            if not (math.isfinite(alpha) and math.isfinite(beta)):
                raise ValueError("alpha and beta must be finite")
            self._params = {"alpha": float(alpha), "beta": float(beta)}

    def _estimate(self, y: np.ndarray, lp: np.ndarray) -> dict[str, float]:
        alpha, beta, _, _ = calibration_slope_intercept(y, lp)
        if math.isnan(beta):
            raise RecalibrationError("logistic recalibration failed: the GLM y ~ LP has no finite MLE "
                                     "(single outcome class, separation or constant LP)")
        return {"alpha": alpha, "beta": beta}

    def _apply(self, p: np.ndarray, lp: np.ndarray) -> np.ndarray:
        prm = self._params
        return np.asarray(expit(prm["alpha"] + prm["beta"] * lp))  # type: ignore[index]


RECALIBRATORS: dict[str, type[Recalibrator]] = {
    cls.method: cls for cls in (NoRecalibration, InterceptOnlyRecalibrator, LogisticRecalibrator)
}


def make_recalibrator(method: str) -> Recalibrator:
    """Unfitted recalibrator for a config ``calibration.method``."""
    if method not in RECALIBRATORS:
        raise ConfigError(f"unknown recalibration method {method!r}; expected one of {sorted(RECALIBRATORS)}")
    return RECALIBRATORS[method]()


def recalibrator_from_dict(d: dict[str, Any]) -> Recalibrator:
    """Rebuild a fitted recalibrator from :meth:`Recalibrator.to_dict` output."""
    method = d.get("method")
    rec = make_recalibrator(str(method))
    params = d.get("params") or {}
    if set(params) != set(rec.param_names):
        raise ConfigError(f"recalibrator {method!r} needs params {sorted(rec.param_names)}, got {sorted(params)}")
    values = {k: float(v) for k, v in params.items()}
    if not all(math.isfinite(v) for v in values.values()):
        raise ConfigError(f"recalibrator {method!r} has non-finite params {values}")
    if rec.param_names:
        rec._params = values
    fitted_on_n = d.get("fitted_on_n")
    rec.fitted_on_n = None if fitted_on_n is None else int(fitted_on_n)
    return rec
