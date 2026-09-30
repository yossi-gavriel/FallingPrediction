"""Model fitting for the nested grouped CV (planning/EXPERIMENT_PLAN.md §6): folds, LASSO / elastic net on the fold's linear design, and
XGBoost with early stopping inside inner folds. Every function is deterministic given its seed; nothing reads VALIDATION or TEST rows."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from falls_ml.evaluation.metrics import pr_auc
from falls_ml.models.lasso_cv import LassoLogisticCV
from falls_ml.phase2.design import LinearDesign, tree_feature_of
from falls_ml.phase2.enet import ElasticNetLogisticCV
from falls_ml.seeding import rng_for

LASSO, ENET, XGB = "LASSO", "ENET", "XGB"


def stratified_folds(y: np.ndarray, k: int, seed: int, component: str) -> np.ndarray:
    """Fold labels 0..k-1, outcome-stratified (one row per patient, so patient-grouped). Deterministic in (seed, component)."""
    y = np.asarray(y).astype(int)
    rng = rng_for(seed, component)
    folds = np.empty(len(y), dtype=np.int64)
    for cls in (0, 1):
        idx = np.flatnonzero(y == cls)
        idx = idx[rng.permutation(len(idx))]
        folds[idx] = np.arange(len(idx)) % k
    return folds


# ============================================================================ linear
@dataclass
class LinearFit:
    family: str
    design: LinearDesign
    model: Any
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self.design.transform(frame))

    def hyper(self) -> dict[str, Any]:
        d = self.model.diagnostics_ if hasattr(self.model, "diagnostics_") else {}
        return {"lambda": d.get("lambda_star"), "l1_ratio": d.get("l1_ratio", 1.0)}

    def coefficients(self) -> pd.DataFrame:
        coef = np.asarray(self.model.coef_)
        std = np.asarray(self.model.coef_standardized_)
        cols = self.design.columns_
        return pd.DataFrame({"design_column": cols, "feature": [self.design.feature_of_.get(c, c) for c in cols], "coefficient": coef,
                             "standardized_coefficient": std, "selected": coef != 0.0})

    def selected_features(self) -> list[str]:
        t = self.coefficients()
        return sorted(set(t.loc[t["selected"] & ~t["feature"].str.startswith("na__"), "feature"]))


def fit_linear(family: str, frame: pd.DataFrame, y: np.ndarray, *, new: list[str], baseline: list[str], design: dict[str, Any], efalls_spec: Any,
               cfg: Any, seed: int, fixed: dict[str, float] | None = None, model_param_overrides: dict[str, Any] | None = None,
               groups: np.ndarray | None = None) -> LinearFit:
    d = LinearDesign(new, baseline, design, efalls_spec).fit(frame, y)
    X = d.transform(frame)
    lin = cfg["lasso"]
    inner = int(cfg["cv"]["inner_folds_linear"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if family == LASSO:
            params = {"cv_folds": inner, "n_lambda": int(lin["n_lambda"]), "lambda_min_ratio": float(lin["lambda_min_ratio"]),
                      "selection": str(lin["selection"]), "tol": float(lin["tol"]),
                      "max_iter": int(lin["max_iter"]), "refit_unpenalized": False,
                      "fixed_lambda": None if not fixed else float(fixed["lambda"]), **(model_param_overrides or {})}
            model = LassoLogisticCV(params, random_state=int(seed)).fit(X, np.asarray(y), groups=groups)
        elif family == ENET:
            en = cfg["enet"]
            model = ElasticNetLogisticCV(l1_ratios=tuple(en["l1_ratios"]), cv_folds=inner, n_lambda=int(en["n_lambda"]), tol=float(lin["tol"]),
                                         max_iter=int(lin["max_iter"]), random_state=int(seed), lambda_min_ratio=float(lin["lambda_min_ratio"]),
                                         fixed=None if not fixed else (float(fixed["l1_ratio"]), float(fixed["lambda"]))).fit(X, np.asarray(y), groups=groups)
        else:
            raise ValueError(family)
    diag = dict(model.diagnostics_)
    return LinearFit(family=family, design=d, model=model, diagnostics={k: v for k, v in diag.items() if isinstance(v, (int, float, str, bool)) or v is None})


# ============================================================================ XGBoost
def xgb_params(cfg: Any, tuned: dict[str, Any], *, seed: int, nthread: int) -> dict[str, Any]:
    x = cfg["xgb"]
    p = {**{k: v for k, v in x["fixed"].items() if v != "auto"}, **tuned, "seed": int(seed) % (2**31 - 1), "nthread": int(nthread), "verbosity": 0}
    if "max_depth" in p:
        p["max_depth"] = int(p["max_depth"])
    return p


def _dmatrix(X: pd.DataFrame, y: np.ndarray | None = None) -> Any:
    import xgboost as xgb

    return xgb.DMatrix(X.to_numpy(dtype=np.float32), label=None if y is None else np.asarray(y, dtype=np.float32),
                       feature_names=list(X.columns), missing=np.nan)


def _logloss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def xgb_inner_cv(X: pd.DataFrame, y: np.ndarray, params: dict[str, Any], *, n_folds: int, seed: int, n_max: int, early_stopping: int) -> dict[str, Any]:
    """Inner grouped CV: train on k-1 folds with ``num_boost_round = n_max`` and early stopping on the held-out fold (log loss); the held-out
    fold is scored with the trees up to the best iteration only (``iteration_range = (0, best_iteration + 1)``; the native API returns the
    LAST model, Astra F-01); objective = mean held-out log loss. ``seed`` fixes the fold assignment (the booster seed is in ``params``)."""
    import xgboost as xgb

    y = np.asarray(y, dtype=float)
    folds = stratified_folds(y, n_folds, seed, "xgb_inner_folds")
    res = []
    for f in range(n_folds):
        te = folds == f
        dtr, dte = _dmatrix(X.loc[~te], y[~te]), _dmatrix(X.loc[te], y[te])
        bst = xgb.train(params, dtr, num_boost_round=int(n_max), evals=[(dte, "held_out")], early_stopping_rounds=int(early_stopping), verbose_eval=False)
        best = int(bst.best_iteration)
        p = bst.predict(dte, iteration_range=(0, best + 1))
        res.append({"fold": f, "best_iteration": best, "n_trees": best + 1, "logloss": _logloss(y[te], p), "ap": float(pr_auc(y[te], p))})
    return {"folds": res, "mean_logloss": float(np.mean([r["logloss"] for r in res])), "mean_ap": float(np.mean([r["ap"] for r in res])),
            "n_trees": int(np.median([r["n_trees"] for r in res]))}


def xgb_fit(X: pd.DataFrame, y: np.ndarray, params: dict[str, Any], n_trees: int) -> Any:
    import xgboost as xgb

    return xgb.train(params, _dmatrix(X, y), num_boost_round=int(n_trees), verbose_eval=False)


def xgb_predict(booster: Any, X: pd.DataFrame) -> np.ndarray:
    return booster.predict(_dmatrix(X)).astype(np.float64)


def xgb_gain_importance(booster: Any) -> dict[str, float]:
    return {tree_feature_of(k): float(v) for k, v in booster.get_score(importance_type="total_gain").items()}
