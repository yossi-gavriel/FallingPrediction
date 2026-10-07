"""The three pre-declared model families and their fitting primitives (no algorithm is added after results are seen).

LASSO   logistic regression, L1 penalty: the Phase 1-3 proximal-Newton / coordinate-descent solver (``phase2.enet.enet_logistic_path``,
        unpenalised intercept, standardised columns) along a logarithmic lambda grid with warm starts
ENET    the same solver with the glmnet elastic-net penalty, an l1_ratio x lambda grid
XGB     gradient-boosted trees (hist), Optuna TPE search with a per-trial deterministic sampler seed; n_estimators by early stopping on an
        early-stopping split carved out of the INNER training rows only
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from falls_ml.phase5.design import LinearDesign, TreeDesign


def lambda_grid(A: np.ndarray, y: np.ndarray, l1_ratio: float, n: int, min_ratio: float) -> np.ndarray:
    """glmnet grid: lambda_max of the (training) design down to min_ratio x lambda_max, logarithmic."""
    lmax = float(np.max(np.abs(A.T @ (y - y.mean()))) / len(y)) / float(l1_ratio) if A.shape[1] else 1.0
    lmax = lmax if lmax > 0 else 1.0
    return lmax * np.logspace(0.0, math.log10(float(min_ratio)), int(n))


def _path(A: np.ndarray, y: np.ndarray, lambdas: np.ndarray, l1_ratio: float, spec: dict[str, Any]) -> Any:
    from falls_ml.phase2.enet import enet_logistic_path

    return enet_logistic_path(A, y.astype(float), np.asarray(lambdas, dtype=float), float(l1_ratio), tol=float(spec["tol"]), max_iter=int(spec["max_iter"]))


def ratio_grid(n: int, min_ratio: float) -> np.ndarray:
    """The dimensionless lambda grid (1 ... min_ratio, logarithmic): a candidate is identified by its INDEX on this grid, never by an absolute
    lambda, so every inner fold and the outer refit anchor the same recipe on their OWN training rows (Phase 5.1 R-3)."""
    return np.logspace(0.0, math.log10(float(min_ratio)), int(n))


def linear_path(Xtr: pd.DataFrame, ytr: np.ndarray, Xva: pd.DataFrame, features: list[str], meta: dict[str, Any], *, l1_ratio: float,
                n_lambda: int, lambda_min_ratio: float, spec: dict[str, Any]) -> dict[str, Any]:
    """One inner fold, one l1_ratio: the design AND the lambda grid are fitted / anchored on the inner-TRAINING rows only (lambda_max of this
    fold's own design, Phase 5.1 R-3); predictions on the validation rows for every grid index, non-zero counts, convergence."""
    from scipy.special import expit

    d = LinearDesign(features, meta).fit(Xtr)
    A, B = d.transform(Xtr), d.transform(Xva)
    lambdas = lambda_grid(A, np.asarray(ytr, dtype=float), l1_ratio, int(n_lambda), float(lambda_min_ratio))
    path = _path(A, ytr, lambdas, l1_ratio, spec)
    eta = path.intercept[None, :] + B @ path.beta.T
    return {"preds": expit(eta), "nnz": (path.beta != 0).sum(axis=1).astype(float), "not_converged": int((~path.converged).sum()),
            "lambda_max": float(lambdas[0]), "lambdas": lambdas}


def linear_path_task(Xarr: np.ndarray, cols: list[str], a: np.ndarray, v: np.ndarray, ytr: np.ndarray, meta: dict[str, Any], l1_ratio: float,
                     n_lambda: int, lambda_min_ratio: float, spec: dict[str, Any]) -> dict[str, Any]:
    """Process-pool entry point (the coordinate-descent loop holds the GIL): the training matrix arrives memory-mapped. Only the inner-training
    rows ``a`` reach the design fit and the grid anchor; the validation rows ``v`` are transformed and predicted only."""
    X = pd.DataFrame(np.asarray(Xarr), columns=cols)
    return linear_path(X.iloc[a], np.asarray(ytr)[a], X.iloc[v], cols, meta, l1_ratio=l1_ratio, n_lambda=n_lambda, lambda_min_ratio=lambda_min_ratio,
                       spec=spec)


@dataclass
class FittedModel:
    family: str
    config: dict[str, Any]
    features: list[str]
    design: Any
    model: Any
    n_estimators: int | None = None
    info: dict[str, Any] = field(default_factory=dict)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        A = self.design.transform(X[self.features])
        if self.family == "XGB":
            import xgboost as xgb

            return self.model.predict(xgb.DMatrix(A, missing=np.nan))
        from scipy.special import expit

        return expit(self.model["intercept"] + A @ self.model["coef"])

    def coefficients(self) -> pd.DataFrame | None:
        if self.family == "XGB":
            return None
        return pd.DataFrame({"column": self.design.columns_, "feature": [self.design.feature_of_[c] for c in self.design.columns_],
                             "coefficient_standardised": self.model["coef"]})


def fit_linear(X: pd.DataFrame, y: np.ndarray, features: list[str], meta: dict[str, Any], *, l1_ratio: float, lambdas: np.ndarray, k: int,
               spec: dict[str, Any], family: str) -> FittedModel:
    """Refit on the training rows along the grid down to the chosen lambda (warm starts, as in the inner folds)."""
    d = LinearDesign(features, meta).fit(X)
    A = d.transform(X)
    path = _path(A, np.asarray(y, dtype=float), np.asarray(lambdas[: k + 1], dtype=float), l1_ratio, spec)
    coef = path.beta[k].copy()
    return FittedModel(family=family, config={"lambda": float(lambdas[k]), "l1_ratio": float(l1_ratio)}, features=list(features), design=d,
                       model={"intercept": float(path.intercept[k]), "coef": coef},
                       info={"not_converged": int(not path.converged[k]), "nnz": int(np.count_nonzero(coef))})


# ============================================================================ XGBoost
class DeviceState:
    """The XGBoost device for the rest of the run; a GPU failure switches it to CPU once (logged, never stops the run)."""

    def __init__(self, device: str, log: Any = None):
        self.device = device
        self.fallbacks: list[str] = []
        self.log = log

    def fallback(self, why: str) -> None:
        if self.device != "cpu":
            self.fallbacks.append(why)
            if self.log:
                self.log(f"GPU FAILURE -> switching XGBoost to CPU for the rest of the run: {why}")
            self.device = "cpu"


def xgb_params(cfg: Any, params: dict[str, Any], *, seed: int, nthread: int, device: str) -> dict[str, Any]:
    fx = dict(cfg["xgb"]["fixed"])
    p = {"objective": fx["objective"], "eval_metric": fx["eval_metric"], "tree_method": fx["tree_method"], "max_bin": int(fx["max_bin"]),
         "scale_pos_weight": float(fx["scale_pos_weight"]), "seed": int(seed) % (2**31 - 1), "nthread": int(nthread), "device": device, "verbosity": 0,
         "max_depth": int(params["max_depth"]), "eta": float(params["learning_rate"]), "min_child_weight": float(params["min_child_weight"]),
         "subsample": float(params["subsample"]), "colsample_bytree": float(params["colsample_bytree"]), "alpha": float(params["reg_alpha"]),
         "lambda": float(params["reg_lambda"]), "gamma": float(params["gamma"])}
    return p


def _train(p: dict[str, Any], dtr: Any, rounds: int, evals: list[Any] | None, es: int | None, state: DeviceState) -> Any:
    import xgboost as xgb

    for attempt in (1, 2):
        p = {**p, "device": state.device}
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                b = xgb.train(p, dtr, num_boost_round=rounds, evals=evals or [], early_stopping_rounds=es, verbose_eval=False)
            if p["device"] != "cpu" and any("changed from GPU to CPU" in str(x.message) for x in w):
                raise RuntimeError("XGBoost silently changed the device from GPU to CPU")
            return b
        except Exception as exc:  # noqa: BLE001 - a GPU problem must not stop the overnight run
            if state.device != "cpu" and attempt == 1:
                state.fallback(f"{type(exc).__name__}: {str(exc)[:200]}")
                continue
            raise
    raise RuntimeError("unreachable")


def xgb_inner_fold(Atr: np.ndarray, ytr: np.ndarray, Ava: np.ndarray, params: dict[str, Any], cfg: Any, *, seed: int, nthread: int,
                   state: DeviceState) -> dict[str, Any]:
    """Train on the inner training rows minus an early-stopping split (inner rows only), predict the inner validation rows."""
    import xgboost as xgb
    from sklearn.model_selection import StratifiedShuffleSplit

    xc = cfg["xgb"]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=float(xc["early_stopping_fraction"]), random_state=int(seed) % (2**31 - 1))
    fit_i, es_i = next(sss.split(np.zeros(len(ytr)), ytr))
    dfit = xgb.DMatrix(Atr[fit_i], label=ytr[fit_i], missing=np.nan)
    des = xgb.DMatrix(Atr[es_i], label=ytr[es_i], missing=np.nan)
    p = xgb_params(cfg, params, seed=seed, nthread=nthread, device=state.device)
    b = _train(p, dfit, int(xc["n_estimators_max"]), [(des, "es")], int(xc["early_stopping_rounds"]), state)
    best = int(b.best_iteration) + 1
    pred = b.predict(xgb.DMatrix(Ava, missing=np.nan), iteration_range=(0, best))
    return {"pred": pred, "best_iter": best}


def fit_xgb(X: pd.DataFrame, y: np.ndarray, features: list[str], meta: dict[str, Any], params: dict[str, Any], n_estimators: int, cfg: Any, *, seed: int,
            nthread: int, state: DeviceState) -> FittedModel:
    import xgboost as xgb

    d = TreeDesign(features, meta).fit(X)
    A = d.transform(X)
    p = xgb_params(cfg, params, seed=seed, nthread=nthread, device=state.device)
    b = _train(p, xgb.DMatrix(A, label=y, missing=np.nan), int(n_estimators), None, None, state)
    b.set_param({"device": "cpu", "nthread": int(nthread)})       # prediction / explanation on CPU, whatever the training device
    return FittedModel(family="XGB", config=dict(params), features=list(features), design=d, model=b, n_estimators=int(n_estimators),
                       info={"device": state.device})
