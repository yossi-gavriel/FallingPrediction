"""Unit tests for REML random-effects meta-analysis with HKSJ CI and prediction interval (D-16; RT-16)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import optimize, stats

from falls_ml.evaluation import metrics as M
from falls_ml.evaluation.meta_analysis import (
    MetaResult,
    measure_with_variance,
    pool_measure,
    random_effects_meta,
    reml_tau2,
)

RT16_TOL = 1e-3

# Stata 19 [META] meta summarize Examples 1/5/6 (pupil IQ, first 10 studies); SEs reconstructed from printed 95% CIs.
STATA_ES = np.array([0.030, 0.120, -0.140, 1.180, 0.260, -0.060, -0.020, -0.320, 0.270, 0.800])
STATA_LB = np.array([-0.215, -0.168, -0.467, 0.449, -0.463, -0.262, -0.222, -0.751, -0.051, 0.308])
STATA_UB = np.array([0.275, 0.408, 0.187, 1.911, 0.983, 0.142, 0.182, 0.111, 0.591, 1.292])
STATA_VAR = ((STATA_UB - STATA_LB) / (2 * stats.norm.ppf(0.975))) ** 2


def _neg_reml(tau2: float, y: np.ndarray, v: np.ndarray) -> float:
    w = 1 / (v + tau2)
    mu = np.sum(w * y) / np.sum(w)
    return 0.5 * (np.sum(np.log(v + tau2)) + np.log(np.sum(w)) + np.sum(w * (y - mu) ** 2))


@pytest.mark.parametrize("modified", [True, False])
def test_rt16_reproduces_stata_meta_summarize_example(modified: bool) -> None:
    # Stata: tau2 0.0754, theta 0.1335309, SE 0.1061617, KH CI -0.1413358 to 0.4083976, 90% PI [-0.414, 0.681].
    # Stata's se(khartung) is the unmodified HKSJ; here q = 1.31 > 1, so the D-16 modified variant is identical.
    res = random_effects_meta(STATA_ES, STATA_VAR, pi_alpha=0.10, hksj_modified=modified)
    assert res.q_hksj > 1
    assert res.tau2 == pytest.approx(0.0754, abs=RT16_TOL)
    assert res.mu == pytest.approx(0.1335309, abs=RT16_TOL)
    assert res.se == pytest.approx(0.1061617, abs=RT16_TOL)
    assert (res.ci_low, res.ci_high) == pytest.approx((-0.1413358, 0.4083976), abs=RT16_TOL)
    assert (res.pi_low, res.pi_high) == pytest.approx((-0.414, 0.681), abs=RT16_TOL)
    assert res.k == 10 and res.method == ("REML+HKSJ-modified" if modified else "REML+HKSJ")
    # tighter internal agreement with the verified reference implementation (meta_reml.py)
    assert res.tau2 == pytest.approx(0.07537093, abs=1e-7)
    assert math.sqrt(res.q_hksj) * res.se == pytest.approx(0.1215065, abs=1e-5)  # Stata KH SE (SEs rebuilt from 3-dp CIs)


def test_reml_tau2_maximises_restricted_likelihood() -> None:
    rng = np.random.default_rng(4)
    y = rng.normal(0.5, 0.3, 12) + rng.normal(0, 0.1, 12)
    v = rng.uniform(0.005, 0.03, 12)
    opt = optimize.minimize_scalar(_neg_reml, bounds=(0, 5), args=(y, v), method="bounded", options={"xatol": 1e-12})
    assert reml_tau2(y, v) == pytest.approx(opt.x, abs=1e-7)


@pytest.mark.parametrize("seed", [5019, 1329])
def test_reml_tau2_is_the_global_restricted_likelihood_maximum_for_very_unequal_variances(seed: int) -> None:
    # seed 5019 (k = 30): the Viechtbauer fixed-point iteration cycles between 0 and 3.9e-4 and never converges;
    # seed 1329 (k = 8): the restricted likelihood is bimodal (tau2 = 0 and ~0.0061) and a DerSimonian-Laird-started
    # fixed-point iteration stops at the inferior local maximum 0.0061.
    rng = np.random.default_rng(seed)
    k = 30 if seed == 5019 else 8
    v = np.exp(rng.uniform(-7 if seed == 5019 else -8, 0, k))
    y = rng.normal(0, np.sqrt(v + (1e-4 if seed == 5019 else 0.01)))
    tau2 = reml_tau2(y, v)
    grid = np.r_[0.0, np.geomspace(1e-10, 1.0, 4000)]
    best = min(_neg_reml(t, y, v) for t in grid)
    assert _neg_reml(tau2, y, v) <= best + 1e-10
    assert random_effects_meta(y, v).tau2 == tau2


def test_homogeneous_equal_variance_hand_calculation_and_modified_hksj() -> None:
    y = np.array([0.98, 1.0, 1.02, 1.0])
    v = np.full(4, 0.01)
    res = random_effects_meta(y, v)
    assert res.tau2 == 0.0 and res.tau == 0.0 and res.i2 == 0.0
    assert res.mu == pytest.approx(1.0) and res.se == pytest.approx(math.sqrt(0.01 / 4))
    q = np.sum((y - 1.0) ** 2 / 0.01) / 3
    assert res.q_hksj == pytest.approx(q) and q < 1
    t3 = stats.t.ppf(0.975, 3)
    assert res.ci_high - res.mu == pytest.approx(t3 * res.se)  # modified: max(1, q) = 1
    unmod = random_effects_meta(y, v, hksj_modified=False)
    assert unmod.ci_high - unmod.mu == pytest.approx(t3 * math.sqrt(q) * res.se)
    wald = random_effects_meta(y, v, hksj=False)
    assert wald.ci_high - wald.mu == pytest.approx(stats.norm.ppf(0.975) * res.se) and wald.method == "REML+Wald"
    assert res.pi_high - res.mu == pytest.approx(stats.t.ppf(0.975, 2) * math.sqrt(res.se**2))


def test_i2_equals_tau2_share_for_equal_variances() -> None:
    res = random_effects_meta(STATA_ES, np.full(10, 0.02))
    assert res.i2 == pytest.approx(res.tau2 / (res.tau2 + 0.02))


def test_small_k_and_invalid_inputs() -> None:
    two = random_effects_meta([0.1, 0.3], [0.01, 0.02])
    assert two.pi_low is None and two.pi_high is None and two.k == 2
    for est, var in (([0.1], [0.01]), ([0.1, 0.2], [0.01, 0.0]), ([0.1, np.nan], [0.01, 0.01]), ([0.1, 0.2], [0.01])):
        with pytest.raises(ValueError):
            random_effects_meta(est, var)


def test_pool_measure_scales_and_back_transformation() -> None:
    c = np.array([0.70, 0.74, 0.78, 0.72, 0.76])
    var_c = np.array([2e-4, 3e-4, 1e-4, 2.5e-4, 1.5e-4])
    pooled = pool_measure(c, var_c, "auroc")
    direct = random_effects_meta(np.log(c / (1 - c)), var_c / (c * (1 - c)) ** 2)
    assert pooled.scale == "logit"
    assert pooled.mu == pytest.approx(M.expit(direct.mu)) and pooled.tau2 == pytest.approx(direct.tau2)
    assert (pooled.ci_low, pooled.pi_high) == pytest.approx((M.expit(direct.ci_low), M.expit(direct.pi_high)))

    oe = np.array([0.8, 1.1, 1.3, 0.9])
    var_log = np.array([0.01, 0.02, 0.015, 0.01])
    pooled_oe = pool_measure(oe, var_log, "oe_ratio", alpha=0.1)
    direct_oe = random_effects_meta(np.log(oe), var_log, alpha=0.1)
    assert pooled_oe.scale == "log" and pooled_oe.ci_high == pytest.approx(math.exp(direct_oe.ci_high))

    slopes = np.array([0.9, 1.1, 1.0])
    assert pool_measure(slopes, [0.01, 0.02, 0.01], "calibration_slope") == random_effects_meta(slopes, [0.01, 0.02, 0.01])
    with pytest.raises(ValueError):
        pool_measure([0.7, 1.0], [0.01, 0.01], "auroc")
    with pytest.raises(ValueError):
        pool_measure([0.7, 0.8], [0.01, 0.01], "brier")
    # extreme log-scale bounds back-transform to inf instead of raising OverflowError
    with np.errstate(over="ignore"):
        wide = pool_measure([1e-300, 1.0, 1e300], [1.0, 1.0, 1.0], "oe_ratio", pi_alpha=0.01)
    assert wide.pi_high == math.inf and wide.pi_low == 0.0


def test_measure_with_variance_uses_d21_within_cluster_ses() -> None:
    rng = np.random.default_rng(1)
    lp = rng.normal(-1.0, 1.0, 400)
    y = (rng.random(400) < M.expit(lp)).astype(int)
    p = np.asarray(M.expit(lp))
    assert measure_with_variance("auroc", y, p) == (M.auroc(y, p), M.auroc_delong_variance(y, p))
    assert measure_with_variance("oe_ratio", y, p) == (M.oe_ratio(y, p), M.log_oe_se(y, p) ** 2)
    slope = M.calibration_slope_intercept(y, lp)
    assert measure_with_variance("calibration_slope", y, p, lp) == (slope[1], slope[3] ** 2)
    est, se = M.citl(y, lp)
    assert measure_with_variance("citl", y, p, lp) == (est, se**2)
    assert isinstance(pool_measure(*zip(*[measure_with_variance("auroc", y[i::4], p[i::4]) for i in range(4)]), "auroc"),
                      MetaResult)
