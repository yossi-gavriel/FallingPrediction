"""Stata-equivalent LASSO logistic regression with K-fold CV selection of lambda (spec §6.2, D-10, D-11, D-14; RT-15).

Objective (Stata 17 ``lasso logit``, spec §6.2), on columns standardised to mean 0 / SD 1 (ddof = 0)::

    Q(b0, beta) = (1/N) * sum_i [ -y_i * eta_i + ln(1 + exp(eta_i)) ] + lambda * sum_j |beta_j|,   eta = b0 + Z beta

The intercept is unpenalised. The solver is a proximal-Newton (IRLS) outer loop whose weighted least-squares
sub-problem is solved by covariance-mode cyclic coordinate descent with soft-thresholding and warm starts
along a decreasing lambda grid. Lambda is chosen by minimising the K-fold CV mean held-out deviance
(-2 log-likelihood per observation), with Stata's ``cvtolerance`` identification rule recorded (D-14).

Scaling note: Stata's lambda corresponds to scikit-learn's ``C = 1 / (N * lambda)`` on identically
standardised data (spec §6.2 "Python mapping").
"""

from __future__ import annotations

import warnings
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np
import pandas as pd
from scipy.linalg import qr
from scipy.special import expit
from statsmodels.discrete.discrete_model import Logit

from falls_ml.errors import DegenerateFitError, BundleIntegrityError, ConfigError, FallsMLError
from falls_ml.logging_utils import get_logger
from falls_ml.models.base import ModelAdapter, coefficient_importance
from falls_ml.seeding import rng_for

log = get_logger(__name__)

ETA_CLIP = 35.0
MIN_WEIGHT = 1e-10
ZERO_VARIANCE_RTOL = 1e-10
COLLINEARITY_TOL = 1e-8
LAMBDA_MAX_RTOL = 1e-9  # absorbs BLAS rounding in the score so lambda_grid()[0] reproduces the exact null solution
MAX_HALVINGS = 40
FOLD_COMPONENT = "lasso_cv_folds"

DEFAULT_PARAMS: dict[str, Any] = {
    "cv_folds": 10,
    "n_lambda": 100,
    "lambda_min_ratio": None,
    "tol": 1e-7,
    "max_iter": 10000,
    "stop": 1e-5,
    "cv_tolerance": 1e-3,
    "cv_confirm": 5,
    "selection": "min",
    "fixed_lambda": None,
    "refit_unpenalized": True,
    "refit_reference_levels": None,
    "fold_standardization": "within_fold",
    "full_path_max_iter": None,
}


#: standardized-scale coefficients below this magnitude are floating-point zeros (spec D-14)
NUMERICAL_ZERO = 1e-12


class LassoError(FallsMLError):
    """Invalid input to, or failure of, the LASSO logistic solver / CV procedure."""


class LassoConvergenceError(LassoError, DegenerateFitError):
    """The full-data solve at the CV-selected lambda did not converge (typically a near-unpenalised solution on a quasi-separated
    resample). The main fit fails loudly; inside bootstrap loops it is a recorded, bounded replicate failure (DegenerateFitError).

    Carries the selected lambda (value and grid index), the tolerance and the iteration limit of the failed full-data solve, so the
    bootstrap convergence-retry policy (``falls_ml.evaluation.convergence_retry``) can verify that a retry solved the same problem."""

    def __init__(self, message: str, *, lambda_star: float | None = None, lambda_star_index: int | None = None, tol: float | None = None,
                 max_iter: int | None = None):
        super().__init__(message)
        self.lambda_star, self.lambda_star_index, self.tol, self.max_iter = lambda_star, lambda_star_index, tol, max_iter


# ============================================================================ numerical building blocks
def _column_moments(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Column means, ddof=0 SDs and a zero-variance mask (SD negligible relative to the column scale)."""
    mean = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)
    zero = sd <= ZERO_VARIANCE_RTOL * np.maximum(1.0, np.abs(mean))
    return mean, sd, zero


def standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardise columns to mean 0 and SD 1 with divisor N (ddof = 0), as Stata ``lasso`` does (§6.2).

    Zero-variance columns get ``sd = 1`` and an all-zero standardised column, so their coefficient is 0.
    Returns ``(Z, mean, sd)``.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise LassoError(f"standardize expects a 2-D array, got shape {X.shape}")
    mean, sd, zero = _column_moments(X)
    sd = np.where(zero, 1.0, sd)
    Z = (X - mean) / sd
    Z[:, zero] = 0.0
    return Z, mean, sd


def lambda_max(Z: np.ndarray, y: np.ndarray) -> float:
    """Smallest lambda at which all penalised coefficients are zero: max_j |Z_j^T (y - ybar)| / N (§6.2)."""
    Z = np.asarray(Z, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if Z.ndim != 2 or Z.shape[1] == 0 or Z.shape[0] != y.shape[0]:
        raise LassoError(f"lambda_max: incompatible shapes Z={Z.shape}, y={y.shape}")
    return float(np.max(np.abs(Z.T @ (y - y.mean()))) / y.shape[0])


def lambda_grid(lmax: float, n_lambda: int = 100, ratio: float = 1e-4) -> np.ndarray:
    """Stata log-spaced grid: ln lambda_i = ((i-1)/(n-1)) ln(ratio) + ln(lmax), i = 1..n (§6.2).

    Stata's default ratio is 1e-4 when p < N and 1e-2 otherwise; the caller chooses.
    """
    if not np.isfinite(lmax) or lmax <= 0:
        raise LassoError(f"lambda_max must be positive and finite, got {lmax}")
    if int(n_lambda) < 1:
        raise LassoError(f"n_lambda must be >= 1, got {n_lambda}")
    if not 0 < ratio <= 1:
        raise LassoError(f"lambda ratio must be in (0, 1], got {ratio}")
    n_lambda = int(n_lambda)
    if n_lambda == 1:
        return np.array([float(lmax)])
    return float(lmax) * np.power(float(ratio), np.arange(n_lambda) / (n_lambda - 1))


def _mean_deviance(eta: np.ndarray, y: np.ndarray) -> float:
    """In-sample mean deviance -2 * loglik / N (saturated loglik = 0 for binary y)."""
    return float(2.0 * np.mean(np.logaddexp(0.0, eta) - y * eta))


def _objective(eta: np.ndarray, y: np.ndarray, beta: np.ndarray, lam: float) -> float:
    return float(np.mean(np.logaddexp(0.0, eta) - y * eta) + lam * np.abs(beta).sum())


def _weighted_gram(Z: np.ndarray, v: np.ndarray, chunk: int = 65536) -> np.ndarray:
    """Z^T diag(v) Z (v > 0), computed in row chunks to bound memory; A^T A form lets BLAS use syrk."""
    p = Z.shape[1]
    G = np.zeros((p, p))
    for start in range(0, Z.shape[0], chunk):
        A = Z[start:start + chunk] * np.sqrt(v[start:start + chunk])[:, None]
        G += A.T @ A
    return G


def _cd_pass(H: np.ndarray, diag: np.ndarray, r: np.ndarray, beta: np.ndarray, lam: float, idx: np.ndarray) -> float:
    """One cyclic coordinate-descent pass over ``idx`` for 0.5 b'Hb - c'b + lam |b|_1; r = c - H b kept in sync."""
    max_change = 0.0
    for j in idx:
        hjj = diag[j]
        bj = beta[j]
        if hjj <= 0.0:
            new = 0.0
        else:
            g = r[j] + hjj * bj
            new = (g - lam) / hjj if g > lam else (g + lam) / hjj if g < -lam else 0.0
        if new != bj:
            delta = new - bj
            r -= delta * H[j]
            beta[j] = new
            max_change = max(max_change, abs(delta))
    return max_change


def _cd_quadratic(H: np.ndarray, c: np.ndarray, lam: float, beta: np.ndarray, tol: float,
                  max_passes: int) -> tuple[np.ndarray, int, bool]:
    """Covariance-mode CD with an active-set strategy: iterate on nonzero coefficients, confirm with full passes."""
    beta = beta.copy()
    r = c - H @ beta
    diag = np.diag(H).copy()
    all_idx = np.arange(H.shape[0])
    passes = 0
    while passes < max_passes:
        change = _cd_pass(H, diag, r, beta, lam, all_idx)
        passes += 1
        if change <= tol * max(1.0, float(np.max(np.abs(beta), initial=0.0))):
            return beta, passes, True
        while passes < max_passes:
            active = np.flatnonzero(beta)
            change = _cd_pass(H, diag, r, beta, lam, active)
            passes += 1
            if change <= tol * max(1.0, float(np.max(np.abs(beta), initial=0.0))):
                break
    return beta, passes, False


def _fit_single_lambda(Z: np.ndarray, y: np.ndarray, lam: float, b0: float, beta: np.ndarray, *, tol: float,
                       max_iter: int) -> tuple[float, np.ndarray, int, bool]:
    """Proximal-Newton (IRLS) solve at one lambda from a warm start. Returns (b0, beta, cd_passes, converged)."""
    n = Z.shape[0]
    eta = b0 + Z @ beta
    obj = _objective(eta, y, beta, lam)
    passes = 0
    while passes < max_iter:
        prob = expit(np.clip(eta, -ETA_CLIP, ETA_CLIP))
        w = np.maximum(prob * (1.0 - prob), MIN_WEIGHT)
        v = w / n
        s = v.sum()
        vz = (w * eta + (y - prob)) / n                  # v_i * z_i, working response z = eta + (y - p) / w
        zbar = vz.sum() / s
        m = (Z.T @ v) / s                                # weighted column means
        H = _weighted_gram(Z, v) - s * np.outer(m, m)    # weighted Gram of weighted-centred columns
        c = Z.T @ vz - s * m * zbar
        new_beta, used, inner_converged = _cd_quadratic(H, c, lam, beta, tol, max_iter - passes)
        passes += used
        d_beta = new_beta - beta
        d_b0 = (zbar - m @ new_beta) - b0
        # Convergence is judged on the full proximal-Newton step, not the damped one: a tiny accepted step
        # far from the optimum must not count as convergence.
        newton_step = max(abs(d_b0), float(np.max(np.abs(d_beta), initial=0.0)))
        step = 1.0
        for _ in range(MAX_HALVINGS):                    # backtracking guards against non-monotone Newton steps
            cand_beta = beta + step * d_beta
            cand_b0 = b0 + step * d_b0
            cand_eta = cand_b0 + Z @ cand_beta
            cand_obj = _objective(cand_eta, y, cand_beta, lam)
            if cand_obj <= obj + 1e-12 * abs(obj):
                break
            step *= 0.5
        else:                                            # no descent found: stop and report non-convergence
            log.warning("lasso_line_search_failed", extra_fields={"lambda": lam, "newton_step": newton_step})
            return b0, beta, passes, False
        b0, beta, eta, obj = cand_b0, cand_beta, cand_eta, cand_obj
        if inner_converged and newton_step <= tol * max(1.0, float(np.max(np.abs(beta), initial=0.0))):
            return b0, beta, passes, True
    return b0, beta, passes, False


@dataclass(frozen=True)
class PathResult:
    """Solution path on standardised columns (§6.2)."""

    lambdas: np.ndarray            # (n_lambda,)
    beta: np.ndarray               # (n_lambda, p) standardised-scale coefficients
    intercept: np.ndarray          # (n_lambda,)
    mean_deviance: np.ndarray      # (n_lambda,) in-sample -2 loglik / N
    n_iter: np.ndarray             # (n_lambda,) coordinate-descent passes used
    converged: np.ndarray          # (n_lambda,) bool
    lambda_stop_index: int | None  # first index where Stata's stop() rule is met (recorded; path not truncated)

    @property
    def n_nonzero(self) -> np.ndarray:
        return (self.beta != 0.0).sum(axis=1)


def lasso_logistic_path(Z: np.ndarray, y: np.ndarray, lambdas: np.ndarray, *, tol: float = 1e-7,
                        max_iter: int = 10000, stop: float = 1e-5) -> PathResult:
    """Fit the Stata ``lasso logit`` objective along a non-increasing lambda grid with warm starts (§6.2, RT-15).

    ``Z`` must already be standardised. Convergence per lambda: the inner CD has converged and the full
    proximal-Newton step satisfies max |d(b0, beta)| <= tol * max(1, max|beta|); a failed line search is
    reported as non-converged. ``max_iter`` bounds coordinate-descent passes per lambda. For
    lambda >= lambda_max the exact solution (beta = 0, b0 = logit(ybar)) is used. ``lambda_stop_index`` is
    the first k >= 1 with a nonzero coefficient where (dev[k-1] - dev[k]) / dev[k-1] < ``stop``.
    """
    Z = np.ascontiguousarray(Z, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    lambdas = np.asarray(lambdas, dtype=np.float64)
    if Z.ndim != 2 or y.ndim != 1 or Z.shape[0] != y.shape[0]:
        raise LassoError(f"lasso_logistic_path: incompatible shapes Z={Z.shape}, y={y.shape}")
    if not np.all(np.isfinite(Z)):
        raise LassoError(f"lasso_logistic_path: Z contains {int((~np.isfinite(Z)).sum())} non-finite values")
    if lambdas.ndim != 1 or lambdas.size == 0 or not np.all(np.isfinite(lambdas)) or np.any(lambdas <= 0):
        raise LassoError("lambdas must be a non-empty 1-D array of positive finite values")
    if np.any(np.diff(lambdas) > 0):
        raise LassoError("lambdas must be non-increasing (warm starts run from lambda_max downwards)")
    if not np.all(np.isin(y, (0.0, 1.0))):
        raise LassoError("y must be binary 0/1")
    ybar = float(y.mean())
    if ybar in (0.0, 1.0):
        raise LassoError("y contains a single class; the logistic LASSO is undefined")

    n_lambda, p = lambdas.size, Z.shape[1]
    lmax = lambda_max(Z, y)
    null_b0 = float(np.log(ybar / (1.0 - ybar)))
    betas = np.zeros((n_lambda, p))
    intercepts = np.empty(n_lambda)
    deviances = np.empty(n_lambda)
    n_iter = np.zeros(n_lambda, dtype=np.int64)
    converged = np.ones(n_lambda, dtype=bool)
    b0, beta = null_b0, np.zeros(p)
    for k, lam in enumerate(lambdas):
        if lam >= lmax * (1.0 - LAMBDA_MAX_RTOL):
            b0, beta = null_b0, np.zeros(p)
        else:
            b0, beta, n_iter[k], converged[k] = _fit_single_lambda(Z, y, float(lam), b0, beta, tol=tol, max_iter=max_iter)
            # Round-off at the KKT boundary (e.g. all-levels indicators collinear with the intercept) leaves ~1e-16
            # instead of an exact zero; snap to zero so "selected" means a real coefficient (documented tolerance).
            beta = np.where(np.abs(beta) < NUMERICAL_ZERO, 0.0, beta)
        betas[k] = beta
        intercepts[k] = b0
        deviances[k] = _mean_deviance(b0 + Z @ beta, y)

    stop_index = None
    for k in range(1, n_lambda):
        if np.any(betas[k] != 0.0) and (deviances[k - 1] <= 0.0 or (deviances[k - 1] - deviances[k]) / deviances[k - 1] < stop):
            stop_index = k
            break
    if not converged.all():
        log.warning("lasso_path_not_converged", extra_fields={"n_not_converged": int((~converged).sum()),
                                                                "max_iter": max_iter, "tol": tol})
    return PathResult(lambdas=lambdas, beta=betas, intercept=intercepts, mean_deviance=deviances, n_iter=n_iter,
                      converged=converged, lambda_stop_index=stop_index)


def cv_select_lambda(cv_mean_deviance: np.ndarray, *, tol: float = 1e-3, confirm: int = 5) -> tuple[int, bool]:
    """CV-minimum rule (§6.2, D-14). Returns ``(k, identified)`` with k = argmin (largest lambda among ties).

    The minimum is *identified* (Stata ``cvtolerance``) iff at least ``confirm`` later (smaller-lambda)
    entries j exceed it by a relative difference (cv[j] - cv[k]) / cv[k] >= ``tol``. D-14: when not
    identified the argmin is still used and ``identified`` is False.
    """
    cv = np.asarray(cv_mean_deviance, dtype=np.float64)
    if cv.ndim != 1 or cv.size == 0 or not np.all(np.isfinite(cv)):
        raise LassoError("cv_mean_deviance must be a non-empty 1-D array of finite values")
    k = int(np.argmin(cv))
    if cv[k] <= 0:
        return k, False
    later = cv[k + 1:]
    identified = int(np.sum((later - cv[k]) / cv[k] >= tol)) >= int(confirm)
    return k, bool(identified)


def assign_folds(n_rows: int, n_folds: int, rng: np.random.Generator, groups: np.ndarray | None = None) -> np.ndarray:
    """Deterministic fold labels in 0..K-1 (D-11, D-14).

    Without groups: a random permutation split into K near-equal folds. With groups: whole groups go to
    one fold (shuffled, largest first, each to the currently smallest fold by rows), so bootstrap
    duplicates sharing an id never straddle folds.
    """
    if n_folds < 2:
        raise LassoError(f"cv_folds must be >= 2, got {n_folds}")
    if groups is None:
        if n_rows < n_folds:
            raise LassoError(f"{n_rows} rows cannot fill {n_folds} folds")
        folds = np.empty(n_rows, dtype=np.int64)
        folds[rng.permutation(n_rows)] = np.arange(n_rows) % n_folds
        return folds
    groups = np.asarray(groups)
    if groups.shape != (n_rows,):
        raise LassoError(f"groups must have shape ({n_rows},), got {groups.shape}")
    codes, uniques = pd.factorize(groups, use_na_sentinel=True)
    if np.any(codes < 0):
        raise LassoError(f"groups contain {int(np.sum(codes < 0))} missing ids")
    n_groups = len(uniques)
    if n_groups < n_folds:
        raise LassoError(f"{n_groups} groups cannot fill {n_folds} folds")
    sizes = np.bincount(codes, minlength=n_groups)
    order = rng.permutation(n_groups)
    order = order[np.argsort(-sizes[order], kind="stable")]
    load = [0] * n_folds
    group_fold = np.empty(n_groups, dtype=np.int64)
    for g in order.tolist():
        f = min(range(n_folds), key=load.__getitem__)
        group_fold[g] = f
        load[f] += int(sizes[g])
    return group_fold[codes]


def _unpenalized_refit(X: np.ndarray, y: np.ndarray, names: list[str], selected: np.ndarray,
                       reference_levels: Mapping[str, str] | None) -> pd.DataFrame:
    """Ordinary logistic refit on LASSO-selected columns (§6.2 'Unpenalised refit'); Wald 95% CIs for ORs.

    Reference-level columns (``'<factor>=<reference>'``) are excluded, then exact collinearity is removed by
    unpivoted QR of [constant, columns] in design order: the first column whose |R_jj| is negligible relative
    to its norm is dropped and the QR repeated, so the constant is always kept and later columns are dropped.
    """
    ref_cols = {f"{p}={r}" for p, r in (reference_levels or {}).items()}
    kept = [int(j) for j in np.flatnonzero(selected) if names[j] not in ref_cols]
    status = {names[j]: "reference_level" for j in np.flatnonzero(selected) if names[j] in ref_cols}
    while True:
        design = np.column_stack([np.ones(X.shape[0]), X[:, kept]])
        R = qr(design, mode="r", check_finite=False)[0]
        r_diag = np.zeros(design.shape[1])
        r_diag[:min(R.shape)] = np.abs(np.diag(R))
        dependent = np.flatnonzero(r_diag <= COLLINEARITY_TOL * np.linalg.norm(design, axis=0))
        if dependent.size == 0:
            break
        if dependent[0] == 0:
            raise LassoError("unpenalised refit: constant column is degenerate")
        status[names[kept.pop(int(dependent[0]) - 1)]] = "dropped_collinear"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            res = Logit(y, design).fit(method="newton", maxiter=100, disp=False)
        except np.linalg.LinAlgError as exc:
            raise LassoError(f"unpenalised refit failed: {exc}") from exc
    messages = sorted({str(w.message) for w in caught})
    if messages or not res.mle_retvals.get("converged", False):
        log.warning("lasso_unpenalized_refit_warnings",
                    extra_fields={"warnings": messages, "converged": bool(res.mle_retvals.get("converged", False))})
    params, bse = np.asarray(res.params), np.asarray(res.bse)
    ci = np.asarray(res.conf_int(alpha=0.05))
    terms = ["_cons"] + [names[j] for j in kept]
    converged = bool(res.mle_retvals.get("converged", False))
    with np.errstate(over="ignore"):   # separated terms give huge coefficients; their OR is reported as inf, not a warning
        odds = np.exp(params), np.exp(ci[:, 0]), np.exp(ci[:, 1])
    status_col = np.where(np.isfinite(odds[0]) & np.isfinite(odds[2]) & converged, "estimated", "not_estimable_separation_or_nonconvergence")
    table = pd.DataFrame({
        "term": terms, "coefficient": params, "se": bse, "z": np.asarray(res.tvalues), "p_value": np.asarray(res.pvalues),
        "odds_ratio": odds[0], "or_ci_low": odds[1], "or_ci_high": odds[2], "status": status_col,
    })
    if status:
        extra = pd.DataFrame({"term": list(status), "status": list(status.values())})
        table = pd.concat([table, extra], ignore_index=True)
    return table


# ============================================================================ adapter
class LassoLogisticCV(ModelAdapter):
    """LASSO logistic regression with CV-selected lambda on all-levels FP design (§6.2, D-10, D-14).

    Params (defaults in ``DEFAULT_PARAMS``): cv_folds, n_lambda, lambda_min_ratio (None = Stata default),
    tol, max_iter, stop, cv_tolerance, cv_confirm, selection ('min' | '1se'), fixed_lambda (skip CV),
    refit_unpenalized, refit_reference_levels ({factor: reference level} for the refit),
    fold_standardization ('within_fold' primary | 'full_sample' sensitivity), full_path_max_iter (None = max_iter; a larger limit for the
    full-data path only, used by the bootstrap convergence-retry policy - the CV fold paths, hence lambda selection, keep max_iter).
    """

    name = "lasso_logistic_cv"
    representation = "efalls_fp_all_levels"
    is_linear = True

    def __init__(self, params: Mapping[str, Any] | None = None, *, random_state: int = 0):
        super().__init__(params, random_state=random_state)
        unknown = sorted(set(self.params) - set(DEFAULT_PARAMS))
        if unknown:
            raise ConfigError(f"{self.name}: unknown params {unknown}; allowed {sorted(DEFAULT_PARAMS)}")
        self.params = {**DEFAULT_PARAMS, **self.params}
        self._validate_params()
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None
        self.coef_standardized_: np.ndarray | None = None
        self.intercept_standardized_: float | None = None
        self.mean_: np.ndarray | None = None
        self.sd_: np.ndarray | None = None
        self.cv_results_: pd.DataFrame | None = None
        self.unpenalized_refit_: pd.DataFrame | None = None
        self.diagnostics_: dict[str, Any] = {}

    def _validate_params(self) -> None:
        p = self.params

        def is_int(key: str, minimum: int) -> bool:
            value = p[key]
            return isinstance(value, (int, np.integer)) and not isinstance(value, bool) and value >= minimum

        def is_real(value: Any, low: float, high: float = np.inf, *, low_open: bool) -> bool:
            if isinstance(value, bool):
                return False
            try:                                          # numeric strings allowed: PyYAML reads `1e-7` as str
                value = float(value)
            except (TypeError, ValueError):
                return False
            return bool(np.isfinite(value)) and (value > low if low_open else value >= low) and value <= high

        problems = []
        for key, minimum in (("cv_folds", 2), ("n_lambda", 1), ("max_iter", 1), ("cv_confirm", 0)):
            if not is_int(key, minimum):
                problems.append(f"{key} must be an integer >= {minimum}")
        if p["full_path_max_iter"] is not None and not is_int("full_path_max_iter", 1):
            problems.append("full_path_max_iter must be an integer >= 1 or None")
        if not is_real(p["tol"], 0.0, low_open=True):
            problems.append("tol must be a finite number > 0")
        for key in ("stop", "cv_tolerance"):
            if not is_real(p[key], 0.0, low_open=False):
                problems.append(f"{key} must be a finite number >= 0")
        if p["lambda_min_ratio"] is not None and not is_real(p["lambda_min_ratio"], 0.0, 1.0, low_open=True):
            problems.append("lambda_min_ratio must be in (0, 1] or None")
        if p["fixed_lambda"] is not None and not is_real(p["fixed_lambda"], 0.0, low_open=True):
            problems.append("fixed_lambda must be a finite number > 0 or None")
        if p["selection"] not in ("min", "1se"):
            problems.append("selection must be 'min' or '1se'")
        if p["fold_standardization"] not in ("within_fold", "full_sample"):
            problems.append("fold_standardization must be 'within_fold' or 'full_sample'")
        if not isinstance(p["refit_unpenalized"], bool):
            problems.append("refit_unpenalized must be a bool")
        if p["refit_reference_levels"] is not None and not isinstance(p["refit_reference_levels"], Mapping):
            problems.append("refit_reference_levels must be a mapping {factor: reference_level} or None")
        if problems:
            raise ConfigError(f"{self.name}: invalid params: " + "; ".join(problems))

    # ------------------------------------------------------------------ fitting
    def fit(self, X: pd.DataFrame, y: np.ndarray, *, groups: np.ndarray | None = None) -> Self:
        Xa, yv, names = self._validate_inputs(X, y)
        p = self.params
        n, n_cols = Xa.shape
        Z, mean, sd = standardize(Xa)
        zero_mask = ~Z.any(axis=0)                       # standardize() zeroes exactly the zero-variance columns
        zero_variance = [names[j] for j in np.flatnonzero(zero_mask)]
        if zero_variance:
            log.warning("lasso_zero_variance_columns", extra_fields={"columns": zero_variance})
        lmax = lambda_max(Z, yv)
        ratio = float(p["lambda_min_ratio"]) if p["lambda_min_ratio"] is not None else (1e-4 if n_cols < n else 1e-2)
        solver = {"tol": float(p["tol"]), "max_iter": int(p["max_iter"]), "stop": float(p["stop"])}
        # The CV fold paths always use max_iter, so lambda selection never depends on full_path_max_iter; the latter only lets the
        # full-data solve iterate longer (bootstrap convergence-retry policy). None = max_iter.
        full_max_iter = int(p["full_path_max_iter"]) if p["full_path_max_iter"] is not None else int(p["max_iter"])
        full_solver = {**solver, "max_iter": full_max_iter}
        grid = lambda_grid(lmax, int(p["n_lambda"]), ratio)

        cv_mean = cv_se = None
        identified: bool | None = None
        cv_converged: bool | None = None
        fold_sizes: list[int] = []
        if p["fixed_lambda"] is None:
            folds = assign_folds(n, int(p["cv_folds"]), rng_for(self.random_state, FOLD_COMPONENT), groups)
            cv_mean, cv_se, fold_sizes, cv_converged = self._cross_validate(Xa, Z, yv, folds, grid, solver)
            full = lasso_logistic_path(Z, yv, grid, **full_solver)
            k_min, identified = cv_select_lambda(cv_mean, tol=float(p["cv_tolerance"]), confirm=int(p["cv_confirm"]))
            if not identified:
                log.warning("lasso_cv_minimum_not_identified",
                            extra_fields={"lambda_index": k_min, "lambda": float(grid[k_min]), "rule": "D-14 argmin used"})
            k_star = k_min if p["selection"] == "min" else int(np.flatnonzero(cv_mean <= cv_mean[k_min] + cv_se[k_min])[0])
        else:
            fixed = float(p["fixed_lambda"])
            grid = np.append(grid[grid > fixed], fixed)
            full = lasso_logistic_path(Z, yv, grid, **full_solver)
            k_star = grid.size - 1

        if not full.converged[k_star]:
            raise LassoConvergenceError(f"{self.name}: full-data solve at the selected lambda {grid[k_star]:.6g} did not converge "
                                        f"(tol={p['tol']}, max_iter={full_max_iter}); refusing to report unconverged coefficients",
                                        lambda_star=float(grid[k_star]), lambda_star_index=int(k_star), tol=float(p["tol"]), max_iter=full_max_iter)
        if cv_converged is False:
            log.warning("lasso_cv_fold_paths_not_converged", extra_fields={"max_iter": p["max_iter"], "tol": p["tol"]})
        beta_std = full.beta[k_star].copy()
        beta_std[zero_mask] = 0.0
        coef = beta_std / sd
        selected = coef != 0.0
        refit = (_unpenalized_refit(Xa, yv, names, selected, p["refit_reference_levels"])
                 if p["refit_unpenalized"] and selected.any() else None)
        self.feature_names_ = names
        self.mean_, self.sd_ = mean, sd
        self.coef_standardized_ = beta_std
        self.intercept_standardized_ = float(full.intercept[k_star])
        self.coef_ = coef
        self.intercept_ = float(self.intercept_standardized_ - np.sum(coef * mean))
        self.unpenalized_refit_ = refit
        n_cv = grid.size
        self.cv_results_ = pd.DataFrame({
            "lambda": grid,
            "cv_mean_deviance": cv_mean if cv_mean is not None else np.full(n_cv, np.nan),
            "cv_se": cv_se if cv_se is not None else np.full(n_cv, np.nan),
            "n_nonzero": full.n_nonzero.astype(np.int64),
            "selected": np.arange(n_cv) == k_star,
        })
        self.diagnostics_ = {
            "lambda_star": float(grid[k_star]), "lambda_star_index": int(k_star), "lambda_max": lmax,
            "lambda_ratio": ratio, "n_lambda": int(n_cv), "cv_folds": int(p["cv_folds"]),
            "cv_criterion": "mean_deviance", "selection": p["selection"], "cv_minimum_identified": identified,
            "fold_sizes": fold_sizes, "fold_standardization": p["fold_standardization"],
            "lambda_stop_index": full.lambda_stop_index, "path_converged": bool(full.converged.all()),
            "cv_paths_converged": cv_converged, "lambda_star_converged": bool(full.converged[k_star]),
            "full_path_max_iter": full_max_iter, "tol": float(p["tol"]),
            "n_selected": int(selected.sum()), "seed": self.random_state, "zero_variance_columns": zero_variance,
        }
        log.info("lasso_cv_fitted", extra_fields={k: v for k, v in self.diagnostics_.items() if k != "zero_variance_columns"})
        return self

    def _cross_validate(self, Xa: np.ndarray, Z: np.ndarray, y: np.ndarray, folds: np.ndarray, grid: np.ndarray,
                        solver: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, list[int], bool]:
        """K-fold held-out deviance on the absolute full-data grid (D-14: within-fold standardisation primary).

        cv_mean_deviance = total held-out deviance / N; cv_se = SD (ddof = 1) of per-fold mean deviances / sqrt(K).
        Also returns whether every fold path converged.
        """
        n_folds = int(self.params["cv_folds"])
        dev_sum = np.zeros((n_folds, grid.size))
        fold_sizes = np.bincount(folds, minlength=n_folds)
        all_converged = True
        for f in range(n_folds):
            test = folds == f
            train = ~test
            if self.params["fold_standardization"] == "within_fold":
                Z_train, mu, sd = standardize(Xa[train])
                Z_test = (Xa[test] - mu) / sd
            else:
                Z_train, Z_test = Z[train], Z[test]
            if np.unique(y[train]).size < 2:
                raise LassoError(f"CV fold {f}: training rows contain a single outcome class")
            path = lasso_logistic_path(Z_train, y[train], grid, **solver)
            all_converged = all_converged and bool(path.converged.all())
            eta = path.intercept[None, :] + Z_test @ path.beta.T
            dev_sum[f] = (2.0 * (np.logaddexp(0.0, eta) - y[test, None] * eta)).sum(axis=0)
        cv_mean = dev_sum.sum(axis=0) / y.shape[0]
        cv_se = (dev_sum / fold_sizes[:, None]).std(axis=0, ddof=1) / np.sqrt(n_folds)
        return cv_mean, cv_se, fold_sizes.astype(int).tolist(), all_converged

    def _validate_inputs(self, X: pd.DataFrame, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
        if not isinstance(X, pd.DataFrame):
            raise LassoError(f"{self.name}: X must be a pandas DataFrame")
        names = [str(c) for c in X.columns]
        if len(set(names)) != len(names) or X.shape[1] == 0:
            raise LassoError(f"{self.name}: design columns must be non-empty and unique")
        Xa = self._to_array(X)
        yv = np.asarray(y)
        if yv.shape != (X.shape[0],):
            raise LassoError(f"{self.name}: y must have shape ({X.shape[0]},), got {yv.shape}")
        yv = yv.astype(np.float64)
        if not np.all(np.isin(yv, (0.0, 1.0))) or np.unique(yv).size < 2:
            raise LassoError(f"{self.name}: y must be binary 0/1 with both classes present")
        return Xa, yv, names

    def _to_array(self, X: pd.DataFrame) -> np.ndarray:
        bad = [c for c in X.columns if not (pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c]))]
        if bad:
            raise LassoError(f"{self.name}: non-numeric design columns {bad[:5]}")
        Xa = X.to_numpy(dtype=np.float64, na_value=np.nan)
        n_bad = int((~np.isfinite(Xa)).sum())
        if n_bad:
            raise LassoError(f"{self.name}: design matrix contains {n_bad} missing or non-finite values")
        return Xa

    # ------------------------------------------------------------------ prediction and reporting
    def linear_predictor(self, X: pd.DataFrame) -> np.ndarray:
        self._check_columns(X)
        return self.intercept_ + self._to_array(X) @ self.coef_

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return expit(self.linear_predictor(X))

    def get_feature_importance(self) -> pd.DataFrame:
        self._check_fitted()
        table = coefficient_importance(self.feature_names_, self.coef_, standardized_scale=False)
        table["native_importance"] = np.abs(self.coef_standardized_)
        return table

    def fit_diagnostics(self) -> dict[str, Any]:
        self._check_fitted()
        return {
            **self.diagnostics_,
            "cv_results": self.cv_results_.copy(),
            "standardized_coefficients": pd.Series(self.coef_standardized_, index=self.feature_names_, name="standardized_coefficient"),
            "standardized_intercept": self.intercept_standardized_,
            "intercept": self.intercept_,
            "unpenalized_refit": None if self.unpenalized_refit_ is None else self.unpenalized_refit_.copy(),
        }

    def coefficients_relative_to_reference(self, reference_levels: Mapping[str, str]) -> tuple[float, pd.Series]:
        """Re-express all-levels factor coefficients relative to reference levels (D-10).

        For factor prefix ``p`` with reference ``r``: every ``'p=<level>'`` coefficient has ``coef['p=r']``
        subtracted and the intercept gains it. Predictions are unchanged when exactly one level indicator
        of each factor equals 1 in every row (all-levels coding).
        """
        self._check_fitted()
        coefs = pd.Series(self.coef_, index=self.feature_names_, name="coefficient")
        intercept = float(self.intercept_)
        for prefix, ref in reference_levels.items():
            level_cols = [c for c in self.feature_names_ if c.startswith(f"{prefix}=")]
            ref_col = f"{prefix}={ref}"
            if ref_col not in level_cols:
                raise ConfigError(f"{self.name}: reference column {ref_col!r} not in the design (levels: {level_cols})")
            shift = float(coefs[ref_col])
            coefs.loc[level_cols] = coefs.loc[level_cols] - shift
            intercept += shift
        return intercept, coefs

    # ------------------------------------------------------------------ persistence
    def save(self, directory: Path) -> None:
        self._check_fitted()
        directory = Path(directory)
        self._write_meta(directory, extra={"diagnostics": self.diagnostics_})
        np.savez(directory / "coefficients.npz", coef=self.coef_, intercept=np.array([self.intercept_]),
                 coef_standardized=self.coef_standardized_, intercept_standardized=np.array([self.intercept_standardized_]),
                 mean=self.mean_, sd=self.sd_)
        self.cv_results_.to_csv(directory / "cv_results.csv", index=False, encoding="utf-8", lineterminator="\n")
        if self.unpenalized_refit_ is not None:
            self.unpenalized_refit_.to_csv(directory / "unpenalized_refit.csv", index=False, encoding="utf-8", lineterminator="\n")

    @classmethod
    def load(cls, directory: Path) -> Self:
        directory = Path(directory)
        try:
            meta = cls._read_meta(directory)
            with open(directory / "coefficients.npz", "rb") as handle, np.load(handle, allow_pickle=False) as npz:
                arrays = {key: npz[key] for key in npz.files}
            cv_results = pd.read_csv(directory / "cv_results.csv")
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            raise BundleIntegrityError(f"{cls.name}: cannot read adapter files in {directory}: {exc}") from exc
        if not isinstance(meta, dict) or meta.get("adapter") != cls.name:
            raise BundleIntegrityError(f"{directory} does not hold a {cls.name!r} adapter")
        missing = sorted({"params", "random_state", "feature_names"} - set(meta))
        missing += sorted({"coef", "intercept", "coef_standardized", "intercept_standardized", "mean", "sd"} - set(arrays))
        if missing:
            raise BundleIntegrityError(f"{cls.name}: adapter files in {directory} lack {missing}")
        model = cls(meta["params"], random_state=meta["random_state"])
        names = list(meta["feature_names"])
        for key in ("coef", "coef_standardized", "mean", "sd"):
            if arrays[key].shape != (len(names),):
                raise BundleIntegrityError(f"{cls.name}: {key} has shape {arrays[key].shape}, expected ({len(names)},)")
        model.feature_names_ = names
        model.coef_ = arrays["coef"]
        model.intercept_ = float(arrays["intercept"][0])
        model.coef_standardized_ = arrays["coef_standardized"]
        model.intercept_standardized_ = float(arrays["intercept_standardized"][0])
        model.mean_, model.sd_ = arrays["mean"], arrays["sd"]
        model.cv_results_ = cv_results
        refit_path = directory / "unpenalized_refit.csv"
        model.unpenalized_refit_ = pd.read_csv(refit_path) if refit_path.exists() else None
        model.diagnostics_ = dict(meta.get("diagnostics", {}))
        return model
