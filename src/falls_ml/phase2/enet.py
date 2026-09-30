"""Elastic-net logistic regression with grouped-CV selection of (l1_ratio, lambda) (Phase 2, planning/EXPERIMENT_PLAN.md §6).

The solver is the Phase 1 LASSO proximal-Newton / covariance-mode coordinate descent (``falls_ml.models.lasso_cv``) with glmnet's
elastic-net penalty ``lambda * (a * |b|_1 + (1 - a) / 2 * |b|_2^2)`` on standardised columns (unpenalised intercept):
the coordinate update is ``S(g, lambda * a) / (H_jj + lambda * (1 - a))`` and ``lambda_max = max_j |Z_j'(y - ybar)| / (N a)``. With
``a = 1`` it reduces to the LASSO (verified by tests). Folds, grid ratio, within-fold standardisation and the CV-minimum rule are those of
the LASSO; l1_ratio and lambda are chosen jointly by the minimum inner-CV mean deviance (= 2 x log loss).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit

from falls_ml.logging_utils import get_logger
from falls_ml.models.lasso_cv import (ETA_CLIP, MAX_HALVINGS, MIN_WEIGHT, NUMERICAL_ZERO, LassoConvergenceError, LassoError, _mean_deviance,
                                      _weighted_gram, assign_folds, cv_select_lambda, lambda_grid, standardize)
from falls_ml.seeding import rng_for

log = get_logger(__name__)
FOLD_COMPONENT = "enet_cv_folds"


def _objective(eta: np.ndarray, y: np.ndarray, beta: np.ndarray, lam: float, a: float) -> float:
    return float(np.mean(np.logaddexp(0.0, eta) - y * eta) + lam * (a * np.abs(beta).sum() + 0.5 * (1.0 - a) * float(beta @ beta)))


def _cd_pass(H: np.ndarray, diag: np.ndarray, r: np.ndarray, beta: np.ndarray, l1: float, l2: float, idx: np.ndarray) -> float:
    max_change = 0.0
    for j in idx:
        hjj = diag[j]
        bj = beta[j]
        if hjj <= 0.0:
            new = 0.0
        else:
            g = r[j] + hjj * bj
            den = hjj + l2
            new = (g - l1) / den if g > l1 else (g + l1) / den if g < -l1 else 0.0
        if new != bj:
            delta = new - bj
            r -= delta * H[j]
            beta[j] = new
            max_change = max(max_change, abs(delta))
    return max_change


def _cd_quadratic(H: np.ndarray, c: np.ndarray, l1: float, l2: float, beta: np.ndarray, tol: float, max_passes: int) -> tuple[np.ndarray, int, bool]:
    beta = beta.copy()
    r = c - H @ beta
    diag = np.diag(H).copy()
    all_idx = np.arange(H.shape[0])
    passes = 0
    while passes < max_passes:
        change = _cd_pass(H, diag, r, beta, l1, l2, all_idx)
        passes += 1
        if change <= tol * max(1.0, float(np.max(np.abs(beta), initial=0.0))):
            return beta, passes, True
        while passes < max_passes:
            change = _cd_pass(H, diag, r, beta, l1, l2, np.flatnonzero(beta))
            passes += 1
            if change <= tol * max(1.0, float(np.max(np.abs(beta), initial=0.0))):
                break
    return beta, passes, False


def _fit_single_lambda(Z: np.ndarray, y: np.ndarray, lam: float, a: float, b0: float, beta: np.ndarray, *, tol: float,
                       max_iter: int) -> tuple[float, np.ndarray, int, bool]:
    n = Z.shape[0]
    l1, l2 = lam * a, lam * (1.0 - a)
    eta = b0 + Z @ beta
    obj = _objective(eta, y, beta, lam, a)
    passes = 0
    while passes < max_iter:
        prob = expit(np.clip(eta, -ETA_CLIP, ETA_CLIP))
        w = np.maximum(prob * (1.0 - prob), MIN_WEIGHT)
        v = w / n
        s = v.sum()
        vz = (w * eta + (y - prob)) / n
        zbar = vz.sum() / s
        m = (Z.T @ v) / s
        H = _weighted_gram(Z, v) - s * np.outer(m, m)
        c = Z.T @ vz - s * m * zbar
        new_beta, used, inner_ok = _cd_quadratic(H, c, l1, l2, beta, tol, max_iter - passes)
        passes += used
        d_beta = new_beta - beta
        d_b0 = (zbar - m @ new_beta) - b0
        newton_step = max(abs(d_b0), float(np.max(np.abs(d_beta), initial=0.0)))
        step = 1.0
        for _ in range(MAX_HALVINGS):
            cb, cb0 = beta + step * d_beta, b0 + step * d_b0
            ce = cb0 + Z @ cb
            co = _objective(ce, y, cb, lam, a)
            if co <= obj + 1e-12 * abs(obj):
                break
            step *= 0.5
        else:
            return b0, beta, passes, False
        b0, beta, eta, obj = cb0, cb, ce, co
        if inner_ok and newton_step <= tol * max(1.0, float(np.max(np.abs(beta), initial=0.0))):
            return b0, beta, passes, True
    return b0, beta, passes, False


@dataclass(frozen=True)
class EnetPath:
    lambdas: np.ndarray
    beta: np.ndarray
    intercept: np.ndarray
    converged: np.ndarray


def enet_logistic_path(Z: np.ndarray, y: np.ndarray, lambdas: np.ndarray, a: float, *, tol: float = 1e-7, max_iter: int = 10000) -> EnetPath:
    """Elastic-net path on standardised columns along a non-increasing lambda grid (warm starts)."""
    if not 0.0 < a <= 1.0:
        raise LassoError(f"l1_ratio must be in (0, 1], got {a}")
    Z = np.ascontiguousarray(Z, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ybar = float(y.mean())
    if ybar in (0.0, 1.0):
        raise LassoError("y contains a single class")
    lmax = float(np.max(np.abs(Z.T @ (y - ybar))) / y.shape[0]) / a
    null_b0 = float(np.log(ybar / (1.0 - ybar)))
    p = Z.shape[1]
    betas = np.zeros((lambdas.size, p))
    b0s = np.empty(lambdas.size)
    conv = np.ones(lambdas.size, dtype=bool)
    b0, beta = null_b0, np.zeros(p)
    for k, lam in enumerate(lambdas):
        if lam >= lmax * (1.0 - 1e-9):
            b0, beta = null_b0, np.zeros(p)
        else:
            b0, beta, _, conv[k] = _fit_single_lambda(Z, y, float(lam), float(a), b0, beta, tol=tol, max_iter=max_iter)
            beta = np.where(np.abs(beta) < NUMERICAL_ZERO, 0.0, beta)
        betas[k], b0s[k] = beta, b0
    return EnetPath(lambdas=lambdas, beta=betas, intercept=b0s, converged=conv)


def _grid(Z: np.ndarray, y: np.ndarray, a: float, n_lambda: int, ratio: float) -> np.ndarray:
    lmax = float(np.max(np.abs(Z.T @ (y - y.mean()))) / y.shape[0]) / a
    return lambda_grid(lmax, n_lambda, ratio)


class ElasticNetLogisticCV:
    """Joint grouped-CV choice of l1_ratio and lambda; penalised coefficients at the chosen pair are the served model."""

    name = "elastic_net_logistic_cv"

    def __init__(self, *, l1_ratios: tuple[float, ...] = (0.1, 0.5, 0.9), cv_folds: int = 10, n_lambda: int = 100, tol: float = 1e-7,
                 max_iter: int = 10000, random_state: int = 0, fixed: tuple[float, float] | None = None, lambda_min_ratio: float | None = None):
        self.l1_ratios = tuple(float(a) for a in l1_ratios)
        self.cv_folds, self.n_lambda, self.tol, self.max_iter = int(cv_folds), int(n_lambda), float(tol), int(max_iter)
        self.random_state = int(random_state)
        self.fixed = fixed
        self.lambda_min_ratio = lambda_min_ratio
        self.diagnostics_: dict[str, Any] = {}

    def fit(self, X: pd.DataFrame, y: np.ndarray, *, groups: np.ndarray | None = None) -> ElasticNetLogisticCV:
        names = [str(c) for c in X.columns]
        Xa = X.to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(Xa)):
            raise LassoError("elastic net: design contains non-finite values")
        yv = np.asarray(y, dtype=np.float64)
        n, p = Xa.shape
        Z, mean, sd = standardize(Xa)
        ratio = float(self.lambda_min_ratio) if self.lambda_min_ratio is not None else (1e-4 if p < n else 1e-2)
        solver = {"tol": self.tol, "max_iter": self.max_iter}
        cv_table = []
        if self.fixed is None:
            folds = assign_folds(n, self.cv_folds, rng_for(self.random_state, FOLD_COMPONENT), groups)
            best = None
            for a in self.l1_ratios:
                grid = _grid(Z, yv, a, self.n_lambda, ratio)
                dev = np.zeros(grid.size)
                for f in range(self.cv_folds):
                    te = folds == f
                    Zt, mu, s = standardize(Xa[~te])
                    Zv = (Xa[te] - mu) / s
                    path = enet_logistic_path(Zt, yv[~te], grid, a, **solver)
                    eta = path.intercept[None, :] + Zv @ path.beta.T
                    dev += (2.0 * (np.logaddexp(0.0, eta) - yv[te, None] * eta)).sum(axis=0)
                cv = dev / n
                k, identified = cv_select_lambda(cv)
                cv_table.append({"l1_ratio": a, "lambda_index": int(k), "lambda": float(grid[k]), "cv_mean_deviance": float(cv[k]),
                                 "minimum_identified": bool(identified)})
                if best is None or cv[k] < best[2]:
                    best = (a, grid, float(cv[k]), k)
            a, grid, _, k = best
        else:
            a, lam = float(self.fixed[0]), float(self.fixed[1])
            g = _grid(Z, yv, a, self.n_lambda, ratio)
            grid = np.append(g[g > lam], lam)
            k = grid.size - 1
        full = enet_logistic_path(Z, yv, grid, a, **solver)
        if not full.converged[k]:
            raise LassoConvergenceError(f"elastic net: full-data solve at l1_ratio={a}, lambda={grid[k]:.6g} did not converge",
                                        lambda_star=float(grid[k]), lambda_star_index=int(k), tol=self.tol, max_iter=self.max_iter)
        beta_std = full.beta[k].copy()          # zero-variance columns are all-zero in Z (standardize), so their coefficient stays 0
        self.feature_names_ = names
        self.coef_ = beta_std / sd
        self.intercept_ = float(full.intercept[k] - np.sum(self.coef_ * mean))
        self.coef_standardized_ = beta_std
        self.l1_ratio_, self.lambda_ = float(a), float(grid[k])
        self.diagnostics_ = {"l1_ratio": self.l1_ratio_, "lambda_star": self.lambda_, "lambda_star_index": int(k), "n_lambda": int(grid.size),
                             "lambda_at_grid_boundary": bool(self.fixed is None and k == grid.size - 1), "n_selected": int(np.count_nonzero(beta_std)),
                             "cv": cv_table, "cv_folds": self.cv_folds, "fixed": self.fixed is not None}
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if [str(c) for c in X.columns] != self.feature_names_:
            raise LassoError("elastic net: design columns differ from the fitted ones")
        return expit(self.intercept_ + X.to_numpy(dtype=np.float64) @ self.coef_)
