"""Random-effects meta-analysis of cluster-level performance (spec §7.2, §7.5, D-16, D-21; RT-16).

Model: y_j ~ N(mu_j, v_j), mu_j ~ N(mu, tau^2).

- tau^2 by REML, maximised over tau^2 >= 0 (see :func:`reml_tau2`; the Viechtbauer 2005 fixed-point
  iteration is not used because it can cycle when within-cluster variances are very unequal).
- CI: Hartung–Knapp–Sidik–Jonkman, mu ± t_{k-1} · sqrt(q* / sum w), q = sum w (y - mu)^2 / (k - 1) with
  RE weights w = 1 / (v + tau^2). D-16 uses the *modified* HKSJ q* = max(1, q) by default; the unmodified
  q* = q is Stata's ``se(khartung)`` default. ``hksj=False`` gives the Wald z interval.
- Prediction interval (k >= 3): mu ± t_{k-2} · sqrt(tau^2 + 1 / sum w) with the conventional REML variance.
- I^2 = tau^2 / (tau^2 + s^2), s^2 the Higgins–Thompson typical within-cluster variance (Stata convention).

Pooling scales (D-16, Snell 2018): AUROC on the logit scale, O/E on the log scale, calibration slope and
CITL on the original scale; see :data:`POOLING_SCALES` for the variance each measure expects.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit
from scipy.stats import norm, t as student_t

from falls_ml.errors import FallsMLError
from falls_ml.evaluation.metrics import (
    auroc,
    auroc_delong_variance,
    calibration_slope_intercept,
    citl,
    log_oe_se,
    oe_ratio,
    prepare_inputs,
)

REML_MAX_BRACKET_DOUBLINGS = 200
REML_GRID_POINTS = 256

# measure -> (pooling scale, what the supplied per-cluster variance must describe)
POOLING_SCALES: dict[str, tuple[str, str]] = {
    "auroc": ("logit", "variance of the AUROC on its original scale (e.g. DeLong); converted by the delta method"),
    "oe_ratio": ("log", "variance of ln(O/E), e.g. log_oe_se(y, p) ** 2"),
    "calibration_slope": ("identity", "variance of the slope (GLM Wald SE squared)"),
    "citl": ("identity", "variance of CITL (GLM Wald SE squared)"),
}


class MetaAnalysisError(FallsMLError):
    """The meta-analysis cannot be computed (e.g. REML did not converge)."""


@dataclass(frozen=True)
class MetaResult:
    """Pooled result. ``mu``/CI/PI are on the measure's original scale; ``se``, ``tau2``, ``tau`` and
    ``q_hksj`` are on the pooling ``scale``. ``pi_low``/``pi_high`` are ``None`` when k < 3."""

    mu: float
    se: float
    tau2: float
    tau: float
    i2: float
    ci_low: float
    ci_high: float
    pi_low: float | None
    pi_high: float | None
    k: int
    q_hksj: float
    method: str
    scale: str = "identity"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_vector(x: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _reml_score(tau2: float, y: np.ndarray, v: np.ndarray) -> float:
    """Twice the derivative of the REML log-likelihood in tau^2: sum w^2 (y - mu)^2 - sum w + sum w^2 / sum w."""
    w = 1.0 / (v + tau2)
    sw = float(np.sum(w))
    mu = float(np.sum(w * y)) / sw
    return float(np.sum(w**2 * (y - mu) ** 2)) - sw + float(np.sum(w**2)) / sw


def _reml_loglik(tau2: float, y: np.ndarray, v: np.ndarray) -> float:
    """REML log-likelihood up to a constant: -(sum ln(v + tau^2) + ln sum w + sum w (y - mu)^2) / 2."""
    w = 1.0 / (v + tau2)
    sw = float(np.sum(w))
    mu = float(np.sum(w * y)) / sw
    return -0.5 * (float(np.sum(np.log(v + tau2))) + math.log(sw) + float(np.sum(w * (y - mu) ** 2)))


def reml_tau2(estimates: np.ndarray, variances: np.ndarray) -> float:
    """REML estimate of the between-cluster variance tau^2 (>= 0); inputs as validated by :func:`random_effects_meta`.

    The restricted likelihood can be multimodal when within-cluster variances are very unequal, so every local
    maximum is located (tau^2 = 0 if the score there is <= 0; each +/- sign change of the score on a geometric
    grid up to a bound where the score is negative, refined by Brent's method) and the best one is returned.
    """
    y, v = estimates, variances
    upper = 10.0 * max(float(np.max(v)), float(np.var(y)))  # beyond ~10 max(v) the weights are nearly equal: unimodal
    for _ in range(REML_MAX_BRACKET_DOUBLINGS):
        if _reml_score(upper, y, v) < 0.0:
            break
        upper *= 2.0
    else:
        raise MetaAnalysisError("REML tau^2: could not bracket the maximum of the restricted likelihood")
    grid = np.r_[0.0, np.geomspace(1e-3 * float(np.min(v)), upper, REML_GRID_POINTS)]
    scores = [_reml_score(float(t), y, v) for t in grid]
    candidates = [0.0] if scores[0] <= 0.0 else []
    for a, b, sa, sb in zip(grid[:-1], grid[1:], scores[:-1], scores[1:]):
        if sa > 0.0 >= sb:
            root = float(b) if sb == 0.0 else brentq(_reml_score, a, b, args=(y, v), xtol=1e-15,
                                                    rtol=4 * np.finfo(float).eps, maxiter=1000)
            candidates.append(float(root))
    return max(candidates, key=lambda t: _reml_loglik(t, y, v))


def random_effects_meta(estimates: Sequence[float] | np.ndarray, variances: Sequence[float] | np.ndarray, *,
                        alpha: float = 0.05, pi_alpha: float | None = None, hksj: bool = True,
                        hksj_modified: bool = True) -> MetaResult:
    """REML random-effects meta-analysis with HKSJ CI and t_{k-2} prediction interval (D-16).

    ``pi_alpha`` sets the prediction-interval level separately (default ``alpha``).
    """
    y = _as_vector(estimates, "estimates")
    v = _as_vector(variances, "variances")
    if y.size != v.size:
        raise ValueError("estimates and variances must have the same length")
    if y.size < 2:
        raise ValueError("random-effects meta-analysis needs at least 2 clusters")
    if np.any(v <= 0):
        raise ValueError("variances must be positive")
    for a in (alpha, alpha if pi_alpha is None else pi_alpha):
        if not 0.0 < a < 1.0:
            raise ValueError("alpha and pi_alpha must lie in (0, 1)")
    k = int(y.size)
    tau2 = reml_tau2(y, v)
    w = 1.0 / (v + tau2)
    mu = float(np.sum(w * y) / np.sum(w))
    se = math.sqrt(1.0 / np.sum(w))
    q = float(np.sum(w * (y - mu) ** 2) / (k - 1))
    if hksj:
        q_used = max(1.0, q) if hksj_modified else q
        half = float(student_t.ppf(1 - alpha / 2, k - 1)) * math.sqrt(q_used) * se
        method = "REML+HKSJ-modified" if hksj_modified else "REML+HKSJ"
    else:
        half = float(norm.ppf(1 - alpha / 2)) * se
        method = "REML+Wald"
    pi_low = pi_high = None
    if k >= 3:
        pa = alpha if pi_alpha is None else pi_alpha
        pi_half = float(student_t.ppf(1 - pa / 2, k - 2)) * math.sqrt(tau2 + se**2)
        pi_low, pi_high = mu - pi_half, mu + pi_half
    w0 = 1.0 / v
    s2 = float((k - 1) * np.sum(w0) / (np.sum(w0) ** 2 - np.sum(w0**2)))
    i2 = float(tau2 / (tau2 + s2))
    return MetaResult(mu=mu, se=se, tau2=tau2, tau=math.sqrt(tau2), i2=i2, ci_low=mu - half, ci_high=mu + half,
                      pi_low=pi_low, pi_high=pi_high, k=k, q_hksj=q, method=method)


def _forward(measure: str, values: np.ndarray, variances: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scale = POOLING_SCALES[measure][0]
    if scale == "logit":
        if np.any((values <= 0) | (values >= 1)):
            raise ValueError(f"{measure} values must lie in (0, 1) for logit pooling")
        return np.log(values / (1 - values)), variances / (values * (1 - values)) ** 2
    if scale == "log":
        if np.any(values <= 0):
            raise ValueError(f"{measure} values must be positive for log pooling")
        return np.log(values), variances
    return values, variances


def _back(scale: str, x: float | None) -> float | None:
    if x is None or scale == "identity":
        return x
    return float(expit(x)) if scale == "logit" else float(np.exp(x))  # no OverflowError for extreme bounds


def pool_measure(values: Sequence[float] | np.ndarray, variances: Sequence[float] | np.ndarray, measure: str,
                 **kwargs: Any) -> MetaResult:
    """Pool a cluster-level measure on its D-16 scale and back-transform mu, CI and PI.

    ``variances`` must be as described in :data:`POOLING_SCALES`; ``kwargs`` go to :func:`random_effects_meta`.
    """
    if measure not in POOLING_SCALES:
        raise ValueError(f"unknown measure {measure!r}; expected one of {sorted(POOLING_SCALES)}")
    vals, var = _as_vector(values, "values"), _as_vector(variances, "variances")
    if vals.size != var.size:
        raise ValueError("values and variances must have the same length")
    scale = POOLING_SCALES[measure][0]
    res = random_effects_meta(*_forward(measure, vals, var), **kwargs)
    return MetaResult(mu=_back(scale, res.mu), se=res.se, tau2=res.tau2, tau=res.tau, i2=res.i2,  # type: ignore[arg-type]
                      ci_low=_back(scale, res.ci_low), ci_high=_back(scale, res.ci_high),  # type: ignore[arg-type]
                      pi_low=_back(scale, res.pi_low), pi_high=_back(scale, res.pi_high), k=res.k,
                      q_hksj=res.q_hksj, method=res.method, scale=scale)


def measure_with_variance(measure: str, y: Any, p: Any, lp: Any | None = None) -> tuple[float, float]:
    """One cluster's (value, variance) in the form :func:`pool_measure` expects (D-21 within-cluster SEs).

    NaN values signal an undefined measure (the caller must exclude and count such clusters, D-16).
    """
    yy, pp, ll = prepare_inputs(y, p, lp)
    if measure == "auroc":
        return auroc(yy, pp), auroc_delong_variance(yy, pp)
    if measure == "oe_ratio":
        return oe_ratio(yy, pp), log_oe_se(yy, pp) ** 2
    if measure == "calibration_slope":
        _, slope, _, slope_se = calibration_slope_intercept(yy, ll)
        return slope, slope_se**2
    if measure == "citl":
        est, se = citl(yy, ll)
        return est, se**2
    raise ValueError(f"unknown measure {measure!r}; expected one of {sorted(POOLING_SCALES)}")
