"""Stata-compatible fractional polynomials: forms, logistic deviance and ``mfp`` closed-test selection.

Spec §6.1 and D-13: FP2 maximum over the Stata powers set, closed-test function selection
(FP2 vs linear on 3 df, then FP2 vs FP1 on 2 df) with backfitting, ``select(1)`` (all variables
forced in), in an unpenalised logistic model with all other candidates. Automatic pre-scaling
applies only when a non-linear form is selected; a linear form stays on the raw scale.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations_with_replacement
from typing import Any

import numpy as np
from numpy.linalg import LinAlgError
from scipy.linalg import cho_factor, cho_solve
from scipy.special import expit
from scipy.stats import chi2

from falls_ml.errors import ConfigError, FallsMLError, DegenerateFitError
from falls_ml.features.transforms import fp_auto_scaling, fp_basis, fp_column_names, normalize_fp_powers
from falls_ml.logging_utils import get_logger

log = get_logger(__name__)

LINEAR: tuple[float, ...] = (1.0,)

#: Newton–Raphson settings for the unpenalised logistic fit (D-13).
_MAX_ITER = 100
_TOL = 1e-10
#: Step-halving gives up once the trial step would move no linear predictor by more than this (logit units).
#: Halving is bounded by step size, not a fixed count: after an overshoot on a rare level the next Newton step
#: can be ~1e12 logits although the MLE is finite.
_MIN_ETA_MOVE = 1e-10
#: At deviance convergence, a remaining Newton step that still moves some linear predictor by more than this
#: (logit units) means the likelihood has no finite maximum, i.e. (quasi-)complete separation. Measured in
#: eta units so the check does not depend on column scaling (separated rows move by ~1 per step; converged
#: fits by < 1e-7).
_SEPARATION_ETA_STEP = 1e-2
#: Cholesky pivots below this ratio (cond(A'WA) >~ 1e12) switch to the ridge-stabilised solve.
_CHOLESKY_MIN_PIVOT_RATIO = 1e-6
#: Ridge (relative to the largest diagonal of A'WA) for numerically singular A'WA: rank-deficient designs, or
#: rows saturated after an overshoot. Keeps the step an accurate descent direction; the MLE is unchanged.
_RIDGE = 1e-10


class FPFittingError(DegenerateFitError):
    """Unpenalised logistic fit failed (non-convergence or separation); stops FP selection (D-13)."""


# ---------------------------------------------------------------------- form
@dataclass(frozen=True)
class FPForm:
    """A fitted FP transformation: columns = FP basis of ``(x + shift) / scale`` (§6.1, D-13)."""

    variable: str
    powers: tuple[float, ...]
    shift: float
    scale: float
    selection_table: dict[str, Any] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "powers", normalize_fp_powers(self.powers))
        object.__setattr__(self, "shift", float(self.shift))
        object.__setattr__(self, "scale", float(self.scale))
        if not (np.isfinite(self.shift) and np.isfinite(self.scale) and self.scale > 0):
            raise ConfigError(f"FPForm {self.variable}: shift must be finite and scale > 0")
        if self.is_linear and (self.shift != 0.0 or self.scale != 1.0):
            raise ConfigError(f"FPForm {self.variable}: a linear form uses the raw scale (shift 0, scale 1; D-13)")

    @property
    def is_linear(self) -> bool:
        return self.powers == LINEAR

    def transform(self, x: Any) -> np.ndarray:
        """FP basis matrix (n, m) for raw values ``x``."""
        return fp_basis((np.asarray(x, dtype="float64") + self.shift) / self.scale, self.powers)

    def column_names(self) -> list[str]:
        return fp_column_names(self.variable, self.powers)

    def to_dict(self) -> dict[str, Any]:
        return {"variable": self.variable, "powers": list(self.powers), "shift": self.shift, "scale": self.scale,
                "selection_table": dict(self.selection_table)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> FPForm:
        return cls(variable=str(d["variable"]), powers=tuple(d["powers"]), shift=d["shift"], scale=d["scale"],
                   selection_table=dict(d.get("selection_table", {})))


# ---------------------------------------------------------------------- logistic deviance
def _deviance(eta: np.ndarray, y: np.ndarray) -> float:
    return 2.0 * float(np.sum(np.logaddexp(0.0, eta) - y * eta))


def _newton_step(A: np.ndarray, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Newton step (A'WA)^-1 A'(y - p) by Cholesky; a numerically singular A'WA gets a tiny relative ridge."""
    p = expit(eta)
    w = p * (1.0 - p)
    hessian = (A * w[:, None]).T @ A
    gradient = A.T @ (y - p)
    try:
        factor = cho_factor(hessian, check_finite=False)
        diag = np.abs(np.diag(factor[0]))
        if diag.min() > _CHOLESKY_MIN_PIVOT_RATIO * diag.max():
            return cho_solve(factor, gradient, check_finite=False)
    except LinAlgError:
        pass
    ridge = _RIDGE * float(np.max(np.diag(hessian)))
    try:
        factor = cho_factor(hessian + ridge * np.eye(hessian.shape[0]), check_finite=False)
    except LinAlgError as exc:
        raise FPFittingError(f"logistic fit: information matrix cannot be factorised ({exc})") from exc
    return cho_solve(factor, gradient, check_finite=False)


def logistic_fit_deviance(X: Any, y: Any, offset: Any = None) -> tuple[float, np.ndarray]:
    """Unpenalised logistic regression with intercept; returns ``(deviance, coef)`` (D-13, RT-14).

    ``coef[0]`` is the intercept, ``coef[1:]`` match the columns of ``X`` (shape (n, k), k >= 0).
    Newton–Raphson with step-halving on internally standardised columns; convergence when a full step
    changes the deviance by |Δdeviance| / (|deviance| + 0.1) < 1e-10. Separation is declared when, at
    convergence, a Newton step would still move some linear predictor by > 0.01 (scale-invariant).
    Rank-deficient designs give the correct deviance with one (non-unique) coefficient solution. Raises
    ``FPFittingError`` on non-convergence within 100 iterations or on (quasi-)complete separation.
    """
    yv = np.asarray(y, dtype="float64").ravel()
    n = yv.size
    Xm = np.asarray(X, dtype="float64")
    if Xm.ndim == 1:
        Xm = Xm.reshape(-1, 1)
    off = np.zeros(n) if offset is None else np.asarray(offset, dtype="float64").ravel()
    if Xm.ndim != 2 or Xm.shape[0] != n or off.size != n:
        raise FPFittingError("logistic fit: X, y and offset must have the same number of rows")
    if not (np.isfinite(Xm).all() and np.isfinite(yv).all() and np.isfinite(off).all()):
        raise FPFittingError("logistic fit: non-finite values in X, y or offset")
    if not np.isin(yv, (0.0, 1.0)).all():
        raise FPFittingError("logistic fit: y must be 0/1")
    ybar = yv.mean()
    if ybar in (0.0, 1.0):
        raise FPFittingError("logistic fit: outcome has a single class (separation)")

    mean = Xm.mean(axis=0)
    sd = Xm.std(axis=0)
    sd[sd == 0] = 1.0
    A = np.column_stack([np.ones(n), (Xm - mean) / sd])
    beta = np.zeros(A.shape[1])
    beta[0] = np.log(ybar / (1.0 - ybar))
    eta = A @ beta + off
    dev = _deviance(eta, yv)
    for _ in range(_MAX_ITER):
        step = _newton_step(A, eta, yv)
        largest = float(np.abs(A @ step).max())
        t = 1.0
        while True:
            new_beta = beta + t * step
            new_eta = A @ new_beta + off
            new_dev = _deviance(new_eta, yv)
            if np.isfinite(new_dev) and new_dev <= dev + 1e-12 * abs(dev):
                break
            t *= 0.5
            if not t * largest >= _MIN_ETA_MOVE:  # also stops on a non-finite step
                raise FPFittingError("logistic fit: step-halving failed to decrease the deviance")
        rel_change = abs(dev - new_dev) / (abs(new_dev) + 0.1)
        beta, eta, dev = new_beta, new_eta, new_dev
        if rel_change < _TOL and t == 1.0:  # a tiny halved step is not evidence of convergence
            eta_step = np.abs(A @ _newton_step(A, eta, yv))
            if eta_step.max() > _SEPARATION_ETA_STEP:
                raise FPFittingError("logistic fit: coefficients diverge (complete or quasi-complete separation); "
                                     f"remaining Newton step moves the linear predictor by {eta_step.max():.3g}")
            slopes = beta[1:] / sd
            return dev, np.concatenate([[beta[0] - slopes @ mean], slopes])
    raise FPFittingError(f"logistic fit: no convergence within {_MAX_ITER} iterations")


# ---------------------------------------------------------------------- closed test
def _closed_test_pvalues(dev_linear: float, dev_fp1: float, dev_fp2: float) -> tuple[float, float]:
    p_nonlinear = float(chi2.sf(max(dev_linear - dev_fp2, 0.0), 3))
    p_simplify = float(chi2.sf(max(dev_fp1 - dev_fp2, 0.0), 2))
    return p_nonlinear, p_simplify


def closed_test_decision(dev_linear: float, dev_fp1: float, dev_fp2: float, alpha: float) -> str:
    """``mfp`` closed test for FP2 max (§6.1): FP2 vs linear (χ² 3 df), then FP2 vs FP1 (χ² 2 df)."""
    if not 0.0 < alpha < 1.0:
        raise ConfigError(f"alpha must lie in (0, 1), got {alpha}")
    if not np.isfinite([dev_linear, dev_fp1, dev_fp2]).all():  # NaN p-values would silently pick a non-linear form
        raise FPFittingError(f"closed test: non-finite deviances {(dev_linear, dev_fp1, dev_fp2)}")
    p_nonlinear, p_simplify = _closed_test_pvalues(dev_linear, dev_fp1, dev_fp2)
    if p_nonlinear >= alpha:
        return "linear"
    return "fp1" if p_simplify >= alpha else "fp2"


# ---------------------------------------------------------------------- selection
def _best(candidates: list[tuple[tuple[float, ...], float]]) -> tuple[tuple[float, ...], float]:
    best = candidates[0]
    for cand in candidates[1:]:
        if cand[1] < best[1]:  # strict: ties keep the first candidate in powers order
            best = cand
    return best


def _select_one(base: np.ndarray, x: np.ndarray, z: np.ndarray, y: np.ndarray, powers: tuple[float, ...],
                max_degree: int, alpha: float) -> dict[str, Any]:
    """Closed-test selection for one variable given the other columns ``base`` (D-13)."""

    def deviance(cols: np.ndarray) -> float:
        return logistic_fit_deviance(np.column_stack([base, cols]), y)[0]

    dev_lin = deviance(x.reshape(-1, 1))
    fp1_powers, dev_fp1 = _best([((p,), deviance(fp_basis(z, (p,)))) for p in powers])
    if max_degree == 2:
        fp2_powers, dev_fp2 = _best([(pair, deviance(fp_basis(z, pair)))
                                     for pair in combinations_with_replacement(powers, 2)])
        p_nonlinear, p_simplify = _closed_test_pvalues(dev_lin, dev_fp1, dev_fp2)
        decision = closed_test_decision(dev_lin, dev_fp1, dev_fp2, alpha)
        selected = {"linear": LINEAR, "fp1": fp1_powers, "fp2": fp2_powers}[decision]
        fp2_record: dict[str, Any] = {"fp2_powers": list(fp2_powers), "deviance_fp2": dev_fp2, "p_fp2_vs_fp1": p_simplify}
    else:  # FP1 maximum: FP1 vs linear on 1 df
        p_nonlinear = float(chi2.sf(max(dev_lin - dev_fp1, 0.0), 1))
        decision = "fp1" if p_nonlinear < alpha else "linear"
        selected = fp1_powers if decision == "fp1" else LINEAR
        fp2_record = {"fp2_powers": None, "deviance_fp2": None, "p_fp2_vs_fp1": None}
    return {"deviance_linear": dev_lin, "fp1_powers": list(fp1_powers), "deviance_fp1": dev_fp1, **fp2_record,
            "p_nonlinear": p_nonlinear, "decision": decision, "selected_powers": list(selected)}


def select_fp_forms(
    continuous: Mapping[str, Any],
    other_design: Any,
    y: Any,
    *,
    powers: Sequence[float],
    max_degree: int = 2,
    alpha: float = 0.05,
    max_cycles: int = 5,
) -> dict[str, FPForm]:
    """``mfp``-style FP selection with backfitting (D-13, RT-14). Deterministic.

    All variables are forced in (select = 1). Variables are processed in ascending order of the LR
    p-value for dropping each from the all-linear model (ties by name). Each cycle re-selects every
    variable given the current forms of the others plus ``other_design``; cycling stops when a full
    cycle changes nothing or after ``max_cycles`` (a warning is logged and recorded).
    """
    if max_degree not in (1, 2):
        raise ConfigError(f"max_degree must be 1 or 2 (FP2 maximum, D-13), got {max_degree}")
    if not 0.0 < alpha < 1.0:
        raise ConfigError(f"alpha must lie in (0, 1), got {alpha}")
    if max_cycles < 1:
        raise ConfigError("max_cycles must be >= 1")
    pw = normalize_fp_powers(sorted({float(p) for p in powers}))
    if not continuous:
        raise ConfigError("select_fp_forms: no continuous variables given")

    yv = np.asarray(y, dtype="float64").ravel()
    n = yv.size
    xs: dict[str, np.ndarray] = {}
    for name in sorted(continuous):
        x = np.asarray(continuous[name], dtype="float64").ravel()
        if x.size != n or not np.isfinite(x).all():
            raise FPFittingError(f"FP variable {name!r}: expected {n} finite values")
        if np.ptp(x) == 0:
            raise FPFittingError(f"FP variable {name!r} is constant in the fitting data")
        xs[name] = x

    other = np.asarray(other_design if other_design is not None else np.empty((n, 0)), dtype="float64").reshape(n, -1)
    zero_var = np.flatnonzero(np.ptp(other, axis=0) == 0) if other.shape[1] else np.empty(0, dtype=int)
    if zero_var.size:
        log.info("fp_zero_variance_columns_dropped",
                 extra_fields={"n_dropped": int(zero_var.size), "column_indices": zero_var.tolist()})
        other = np.delete(other, zero_var, axis=1)
    full_linear = np.column_stack([np.ones(n), other, *xs.values()])
    rank = int(np.linalg.matrix_rank(full_linear))
    if rank < full_linear.shape[1]:
        log.warning("fp_design_rank_deficient", extra_fields={"rank": rank, "n_columns": full_linear.shape[1]})

    # xorder(+): ascending LR p-value for dropping each variable from the all-linear model. The p-value
    # underflows to 0 for strong predictors, so ties are broken by the (monotone, 1 df) LR statistic, then name.
    dev_full, _ = logistic_fit_deviance(full_linear[:, 1:], yv)
    xorder_stat: dict[str, float] = {}
    xorder_p: dict[str, float] = {}
    for name in xs:
        others = [xs[u] for u in xs if u != name]
        dev_without, _ = logistic_fit_deviance(np.column_stack([other, *others]) if others else other, yv)
        xorder_stat[name] = max(dev_without - dev_full, 0.0)
        xorder_p[name] = float(chi2.sf(xorder_stat[name], 1))
    order = sorted(xs, key=lambda v: (xorder_p[v], -xorder_stat[v], v))

    scaling = {v: fp_auto_scaling(xs[v]) for v in order}
    z = {v: (xs[v] + scaling[v][0]) / scaling[v][1] for v in order}

    def columns(v: str, form: tuple[float, ...]) -> np.ndarray:
        return xs[v].reshape(-1, 1) if form == LINEAR else fp_basis(z[v], form)

    forms: dict[str, tuple[float, ...]] = {v: LINEAR for v in order}
    records: dict[str, dict[str, Any]] = {}
    converged = False
    cycles_run = 0
    for cycle in range(1, max_cycles + 1):
        cycles_run = cycle
        changed = False
        for v in order:
            base = np.column_stack([other, *(columns(u, forms[u]) for u in order if u != v)])
            record = _select_one(base, xs[v], z[v], yv, pw, max_degree, alpha)
            records[v] = {"cycle": cycle, **record}
            selected = tuple(record["selected_powers"])
            if selected != forms[v]:
                forms[v] = selected
                changed = True
        if not changed:
            converged = True
            break
    if not converged:
        log.warning("fp_selection_not_converged",
                    extra_fields={"max_cycles": max_cycles, "forms": {v: list(f) for v, f in forms.items()}})

    result: dict[str, FPForm] = {}
    for rank_pos, v in enumerate(order, start=1):
        table = {
            "method": "mfp_closed_test", "alpha": alpha, "max_degree": max_degree, "powers_searched": list(pw),
            "n_obs": n, "n_other_columns": int(other.shape[1]), "n_zero_variance_dropped": int(zero_var.size),
            "xorder_rank": rank_pos, "xorder_p_value": xorder_p[v], "xorder_lr_statistic": xorder_stat[v],
            "cycles_run": cycles_run, "converged": converged,
            **records[v],
        }
        shift, scale = (0.0, 1.0) if forms[v] == LINEAR else scaling[v]
        result[v] = FPForm(variable=v, powers=forms[v], shift=shift, scale=scale, selection_table=table)
    log.info("fp_selection_done", extra_fields={
        "forms": {v: {"powers": list(f.powers), "shift": f.shift, "scale": f.scale} for v, f in result.items()},
        "cycles_run": cycles_run, "converged": converged})
    return result
