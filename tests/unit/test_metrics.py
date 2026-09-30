"""Unit tests for performance measures (spec §7.1, D-17, D-21): hand calculations on tiny arrays, ties and edge cases."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from sklearn.metrics import average_precision_score
from statsmodels.nonparametric.smoothers_lowess import lowess

from falls_ml.evaluation import metrics as M
from falls_ml.evaluation.metrics import UndefinedMetricWarning

SUMMARY_KEYS = {"n", "n_events", "auroc", "pr_auc", "brier", "calibration_slope", "calibration_intercept", "citl",
                "oe_ratio", "mean_predicted", "observed_rate", "calibration_indices", "thresholds"}
THRESHOLD_KEYS = {"threshold", "sensitivity", "specificity", "ppv", "npv", "f1", "tp", "fp", "tn", "fn", "net_benefit"}


def _synthetic(n: int = 4000, seed: int = 7, intercept: float = 0.0, slope: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    lp = rng.normal(-2.0, 1.0, n)
    y = (rng.random(n) < M.expit(intercept + slope * lp)).astype(np.int64)
    return y, np.asarray(M.expit(lp)), lp


def _pairwise_auc(y: np.ndarray, p: np.ndarray) -> float:
    pos, neg = p[y == 1][:, None], p[y == 0][None, :]
    return float(np.mean((pos > neg) + 0.5 * (pos == neg)))


def _brute_delong_variance(y: np.ndarray, p: np.ndarray) -> float:
    pos, neg = p[y == 1], p[y == 0]
    psi = (pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])
    return float(np.var(psi.mean(axis=1), ddof=1) / pos.size + np.var(psi.mean(axis=0), ddof=1) / neg.size)


# ---------------------------------------------------------------------- links
def test_logit_expit_round_trip_and_clipping() -> None:
    p = np.array([0.0, 1e-3, 0.5, 0.9, 1.0])
    lp = M.logit(p)
    assert np.all(np.isfinite(lp))
    assert lp[0] == pytest.approx(math.log(1e-12), rel=1e-9)
    assert M.logit(0.5) == 0.0 and isinstance(M.logit(0.5), float)
    np.testing.assert_allclose(M.expit(lp[1:-1]), p[1:-1], rtol=1e-12)
    assert M.expit(0.0) == 0.5
    with pytest.raises(ValueError):
        M.logit(np.array([0.2, 1.2]))
    with pytest.raises(ValueError):
        M.logit(np.array([np.nan]))


# ---------------------------------------------------------------------- AUROC / DeLong / PR-AUC
def test_auroc_hand_calculation_with_ties() -> None:
    # events 0.35, 0.8, 0.35; non-events 0.1, 0.4, 0.35 -> concordance (1.5 + 1.5 + 3) / 9
    y = np.array([0, 0, 1, 1, 0, 1])
    p = np.array([0.1, 0.4, 0.35, 0.8, 0.35, 0.35])
    assert M.auroc(y, p) == pytest.approx(6 / 9, abs=1e-12)
    assert M.auroc(y, np.full(6, 0.3)) == 0.5


def test_auroc_matches_pairwise_definition_on_heavily_tied_data() -> None:
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 300)
    p = rng.integers(0, 8, 300) / 10
    assert M.auroc(y, p) == pytest.approx(_pairwise_auc(y, p), abs=1e-12)


def test_auroc_single_class_is_nan_with_warning() -> None:
    with pytest.warns(UndefinedMetricWarning):
        assert math.isnan(M.auroc(np.zeros(5), np.linspace(0.1, 0.5, 5)))


def test_delong_variance_hand_calculation() -> None:
    # placements: events V10 = [0.5, 1], non-events V01 = [1, 0.5]; var = 0.125/2 + 0.125/2
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.35, 0.8])
    assert M.auroc(y, p) == 0.75
    assert M.auroc_delong_variance(y, p) == pytest.approx(0.125, abs=1e-12)
    assert M.auroc_logit_se(y, p) == pytest.approx(math.sqrt(0.125) / (0.75 * 0.25), abs=1e-12)


def test_fast_delong_matches_brute_force_with_ties() -> None:
    rng = np.random.default_rng(11)
    y = rng.integers(0, 2, 250)
    p = np.round(rng.random(250), 1)
    assert M.auroc_delong_variance(y, p) == pytest.approx(_brute_delong_variance(y, p), rel=1e-10)


def test_delong_needs_two_of_each_class() -> None:
    with pytest.warns(UndefinedMetricWarning):
        assert math.isnan(M.auroc_delong_variance(np.array([0, 0, 1]), np.array([0.1, 0.2, 0.3])))
    with pytest.warns(UndefinedMetricWarning):  # perfect separation: logit(1) undefined
        assert math.isnan(M.auroc_logit_se(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.3, 0.4])))


def test_pr_auc_hand_calculations() -> None:
    # (R, P) steps: (0.5, 1), (0.5, 0.5), (1, 2/3), (1, 0.5) -> 0.5·1 + 0.5·2/3
    assert M.pr_auc(np.array([1, 0, 1, 0]), np.array([0.9, 0.8, 0.7, 0.1])) == pytest.approx(0.5 + 1 / 3, abs=1e-12)
    # tied top scores form one threshold: (R, P) = (0.5, 0.5), (1, 2/3)
    assert M.pr_auc(np.array([1, 0, 1]), np.array([0.5, 0.5, 0.2])) == pytest.approx(0.25 + 1 / 3, abs=1e-12)


def test_pr_auc_matches_sklearn_average_precision_with_ties() -> None:
    rng = np.random.default_rng(5)
    y = rng.integers(0, 2, 400)
    p = rng.integers(0, 20, 400) / 20
    assert M.pr_auc(y, p) == pytest.approx(average_precision_score(y, p), abs=1e-12)
    with pytest.warns(UndefinedMetricWarning):
        assert math.isnan(M.pr_auc(np.zeros(4), np.full(4, 0.2)))


# ---------------------------------------------------------------------- Brier, GLM calibration, O/E
def test_brier_hand_calculation() -> None:
    assert M.brier(np.array([0, 1]), np.array([0.2, 0.6])) == pytest.approx(0.1, abs=1e-15)


def test_calibration_slope_intercept_matches_statsmodels_glm() -> None:
    y, _, lp = _synthetic(intercept=-0.3, slope=1.2)
    a, b, se_a, se_b = M.calibration_slope_intercept(y, lp)
    fit = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit(tol=1e-12)
    np.testing.assert_allclose([a, b], fit.params, atol=1e-7)
    np.testing.assert_allclose([se_a, se_b], fit.bse, rtol=1e-6)


def test_calibration_slope_intercept_two_group_hand_calculation() -> None:
    # Two LP values form a saturated model: logit(ybar_g) = a + b x_g. Group 1: x = -2, 2/10 events (logit -ln 4);
    # group 2: x = 0, 4/8 events (logit 0) -> b = ln 2, a = 0. Var(logit ybar_g) = 1 / (n_g ybar_g (1 - ybar_g)) =
    # 0.625 and 0.5 -> Var(b) = (0.625 + 0.5) / 2^2, Var(a) = (0^2 * 0.625 + (-2)^2 * 0.5) / 2^2.
    y = np.r_[np.ones(2), np.zeros(8), np.ones(4), np.zeros(4)]
    lp = np.r_[np.full(10, -2.0), np.zeros(8)]
    a, b, se_a, se_b = M.calibration_slope_intercept(y, lp)
    assert (a, b) == pytest.approx((0.0, math.log(2)), abs=1e-10)
    assert (se_a, se_b) == pytest.approx((math.sqrt(0.5), math.sqrt(1.125 / 4)), abs=1e-10)


def test_citl_matches_statsmodels_offset_glm_and_closed_form() -> None:
    y, _, lp = _synthetic(intercept=0.4)
    est, se = M.citl(y, lp)
    fit = sm.GLM(y, np.ones((y.size, 1)), family=sm.families.Binomial(), offset=lp).fit(tol=1e-12)
    assert est == pytest.approx(fit.params[0], abs=1e-7)
    assert se == pytest.approx(fit.bse[0], rel=1e-6)
    mu = M.expit(est + lp)
    assert se == pytest.approx(1 / math.sqrt(np.sum(mu * (1 - mu))), rel=1e-9)
    # zero offset: CITL = logit(ybar), SE = 1 / sqrt(n ybar (1 - ybar))
    est0, se0 = M.citl(np.array([1, 0, 0, 0]), np.zeros(4))
    assert est0 == pytest.approx(math.log(1 / 3), abs=1e-9)
    assert se0 == pytest.approx(1 / math.sqrt(4 * 0.25 * 0.75), abs=1e-9)


@pytest.mark.parametrize(("y", "lp"), [
    (np.array([0, 0, 1, 1]), np.array([-2.0, -1.0, 1.0, 2.0])),   # complete separation
    (np.array([0, 1, 0, 1]), np.zeros(4)),                         # constant LP
    (np.ones(4), np.array([-1.0, 0.0, 1.0, 2.0])),                 # single class
])
def test_calibration_slope_undefined_cases_are_nan(y: np.ndarray, lp: np.ndarray) -> None:
    with pytest.warns(UndefinedMetricWarning):
        out = M.calibration_slope_intercept(y, lp)
    assert all(math.isnan(v) for v in out)


def test_citl_single_class_is_nan() -> None:
    with pytest.warns(UndefinedMetricWarning):
        assert all(math.isnan(v) for v in M.citl(np.zeros(3), np.zeros(3)))


def test_oe_ratio_and_log_oe_se_hand_calculation() -> None:
    y, p = np.array([1, 0, 0, 1]), np.full(4, 0.25)
    assert M.oe_ratio(y, p) == 2.0
    assert M.log_oe_se(y, p) == pytest.approx(math.sqrt(0.5 / 2), abs=1e-15)
    with pytest.warns(UndefinedMetricWarning):
        assert math.isnan(M.log_oe_se(np.zeros(4), p))
    with pytest.warns(UndefinedMetricWarning):
        assert math.isnan(M.oe_ratio(y, np.zeros(4)))


# ---------------------------------------------------------------------- calibration curves
def test_grouped_calibration_hand_calculation_and_truncated_wald_ci() -> None:
    p = np.array([0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.05])
    y = np.array([1, 0, 0, 1, 0, 1, 0, 1, 1, 0])
    out = M.grouped_calibration(y, p, n_groups=3)
    assert list(out.columns) == ["group", "mean_predicted", "observed_rate", "n", "ci_low", "ci_high"]
    assert out["group"].tolist() == [1, 2, 3] and out["n"].tolist() == [4, 3, 3]
    # sorted risks: [.05 .1 .2 .3] y [0 0 0 1]; [.4 .5 .6] y [0 1 0]; [.7 .8 .9] y [1 1 1]
    np.testing.assert_allclose(out["mean_predicted"], [0.1625, 0.5, 0.8], atol=1e-12)
    np.testing.assert_allclose(out["observed_rate"], [0.25, 1 / 3, 1.0], atol=1e-12)
    half = 1.959963984540054 * math.sqrt(0.25 * 0.75 / 4)
    assert (out.loc[0, "ci_low"], out.loc[0, "ci_high"]) == pytest.approx((0.0, 0.25 + half))  # truncated at 0
    assert (out.loc[2, "ci_low"], out.loc[2, "ci_high"]) == (1.0, 1.0)
    assert out["group"].dtype == np.int64 and out["n"].dtype == np.int64


def test_grouped_calibration_equal_sizes_with_ties_and_errors() -> None:
    p = np.full(40, 0.2)
    out = M.grouped_calibration(np.r_[np.ones(10), np.zeros(30)], p, n_groups=20)
    assert out["n"].tolist() == [2] * 20
    with pytest.raises(ValueError):
        M.grouped_calibration(np.array([0, 1]), np.array([0.1, 0.2]), n_groups=3)


def test_smoothed_calibration_evaluates_statsmodels_lowess_on_a_grid() -> None:
    y, p, _ = _synthetic(n=3000)
    out = M.smoothed_calibration(y, p, n_points=50)
    assert list(out.columns) == ["mean_predicted", "observed_rate"] and len(out) == 50
    assert out["mean_predicted"].iloc[0] == p.min() and out["mean_predicted"].iloc[-1] == p.max()
    ref = lowess(y, p, frac=0.75, it=0, delta=0.01 * np.ptp(p))
    xs, first = np.unique(ref[:, 0], return_index=True)
    np.testing.assert_allclose(out["observed_rate"], np.interp(out["mean_predicted"], xs, ref[first, 1]), atol=1e-12)


def test_calibration_indices_against_direct_lowess_and_perfect_calibration() -> None:
    y, p, _ = _synthetic(n=20000, seed=1)
    idx = M.calibration_indices(y, p)
    fitted = lowess(y, p, frac=0.75, it=0, delta=0.01 * np.ptp(p), return_sorted=False)
    diff = np.abs(p - fitted)
    assert idx == pytest.approx({"ici": diff.mean(), "e50": np.median(diff), "e90": np.quantile(diff, 0.9)})
    assert idx["ici"] < 0.01  # well calibrated by construction
    y_mis, p_mis, _ = _synthetic(n=20000, seed=1, intercept=1.0)
    assert M.calibration_indices(y_mis, p_mis)["ici"] > 5 * idx["ici"]
    with pytest.warns(UndefinedMetricWarning, match="distinct"):  # constant risk: valid input, undefined curve
        assert all(math.isnan(v) for v in M.calibration_indices(np.array([0, 1, 0]), np.full(3, 0.3)).values())
    with pytest.raises(ValueError):
        M.calibration_indices(y, p, frac=0.0)


def test_lowess_reproduces_a_linear_truth_exactly() -> None:
    # 50 tied risk values k/100, each with 100 rows of which exactly k are events: the group means lie on y = p, so the
    # local-linear fit (equal weights within a tie) is the identity -> curve = diagonal, ICI = E50 = E90 = 0.
    ks = np.arange(5, 55)
    p = np.repeat(ks / 100, 100)
    y = np.concatenate([np.r_[np.ones(k), np.zeros(100 - k)] for k in ks])
    idx = M.calibration_indices(y, p)
    assert max(idx.values()) < 1e-12
    curve = M.smoothed_calibration(y, p, n_points=7)
    np.testing.assert_allclose(curve["observed_rate"], curve["mean_predicted"], atol=1e-12)
    np.testing.assert_allclose(curve["mean_predicted"], np.linspace(0.05, 0.54, 7), atol=1e-15)


# ---------------------------------------------------------------------- thresholds, net benefit, decision curve
def test_threshold_metrics_hand_calculation_boundary_is_positive() -> None:
    y = np.array([1, 1, 0, 0, 1, 0])
    p = np.array([0.9, 0.2, 0.3, 0.1, 0.5, 0.6])
    row = M.threshold_metrics(y, p, [0.5])[0]
    assert set(row) == THRESHOLD_KEYS
    assert (row["tp"], row["fp"], row["tn"], row["fn"]) == (2, 1, 2, 1)  # p = 0.5 counts as positive
    for key in ("sensitivity", "specificity", "ppv", "npv", "f1"):
        assert row[key] == pytest.approx(2 / 3)
    assert row["net_benefit"] == pytest.approx(2 / 6 - 1 / 6 * 1.0)


def test_threshold_metrics_undefined_ratios_are_none() -> None:
    row = M.threshold_metrics(np.array([1, 0, 1]), np.array([0.1, 0.2, 0.3]), [0.95])[0]
    assert (row["tp"], row["fp"], row["ppv"], row["f1"]) == (0, 0, None, 0.0)
    row0 = M.threshold_metrics(np.zeros(3), np.array([0.1, 0.2, 0.3]), [0.95])[0]
    assert row0["sensitivity"] is None and row0["f1"] is None and row0["specificity"] == 1.0
    with pytest.raises(ValueError):
        M.threshold_metrics(np.zeros(3), np.full(3, 0.2), [1.0])


def test_net_benefit_and_decision_curve_hand_calculation() -> None:
    y = np.array([1, 1, 0, 0, 1, 0, 0, 0])
    p = np.array([0.9, 0.2, 0.3, 0.1, 0.5, 0.6, 0.05, 0.25])
    # t = 0.25: positives 0.9(1) 0.3(0) 0.5(1) 0.6(0) 0.25(0) -> TP 2, FP 3
    assert M.net_benefit(y, p, 0.25) == pytest.approx(2 / 8 - 3 / 8 * (1 / 3))
    dc = M.decision_curve(y, p, [0.25, 0.5])
    assert list(dc.columns) == ["threshold", "net_benefit_model", "net_benefit_treat_all", "net_benefit_treat_none",
                                "standardized_net_benefit"]
    phi = 3 / 8
    np.testing.assert_allclose(dc["net_benefit_treat_all"], [phi - (1 - phi) / 3, phi - (1 - phi)])
    np.testing.assert_allclose(dc["net_benefit_model"], [M.net_benefit(y, p, 0.25), 2 / 8 - 1 / 8])
    assert (dc["net_benefit_treat_none"] == 0).all()
    np.testing.assert_allclose(dc["standardized_net_benefit"], dc["net_benefit_model"] / phi)
    assert M.net_benefit_treat_all(phi, 0.25) == pytest.approx(dc["net_benefit_treat_all"][0])


# ---------------------------------------------------------------------- registry and summary
def test_compute_metric_names() -> None:
    y, p, lp = _synthetic(n=500)
    assert M.compute_metric("auroc", y, p) == M.auroc(y, p)
    assert M.compute_metric("calibration_intercept", y, p, lp) == M.calibration_slope_intercept(y, lp)[0]
    assert M.compute_metric("net_benefit@0.1", y, p) == M.net_benefit(y, p, 0.1)
    assert M.parse_metric_name("net_benefit@0.25") == ("net_benefit", 0.25)
    for bad in ("accuracy", "net_benefit@1.5", "net_benefit@"):
        with pytest.raises(ValueError):
            M.compute_metric(bad, y, p)


def test_performance_summary_keys_values_and_default_lp() -> None:
    y, p, lp = _synthetic(n=2000)
    s = M.performance_summary(y, p, thresholds=(0.1, 0.2))
    assert set(s) == SUMMARY_KEYS
    assert s["n"] == 2000 and s["n_events"] == int(y.sum())
    assert s["auroc"] == M.auroc(y, p) and s["oe_ratio"] == M.oe_ratio(y, p)
    assert s["calibration_slope"] == pytest.approx(M.calibration_slope_intercept(y, lp)[1], abs=1e-9)
    assert s["citl"] == pytest.approx(M.citl(y, lp)[0], abs=1e-9)
    assert [r["threshold"] for r in s["thresholds"]] == [0.1, 0.2]
    assert set(s["calibration_indices"]) == {"ici", "e50", "e90"}
    json.dumps(s, allow_nan=False)


def test_performance_summary_single_class_gives_none_and_warns() -> None:
    with pytest.warns(UndefinedMetricWarning):
        s = M.performance_summary(np.zeros(30), np.linspace(0.01, 0.3, 30))
    assert s["auroc"] is None and s["pr_auc"] is None and s["calibration_slope"] is None and s["citl"] is None
    assert s["brier"] is not None
    json.dumps(s, allow_nan=False)


@pytest.mark.parametrize(("y", "p"), [
    (np.array([0, 2]), np.array([0.1, 0.2])),
    (np.array([0.0, np.nan]), np.array([0.1, 0.2])),
    (np.array([0, 1]), np.array([0.1, 1.2])),
    (np.array([0, 1]), np.array([0.1, np.nan])),
    (np.array([0, 1, 1]), np.array([0.1, 0.2])),
    (np.array([]), np.array([])),
    (np.array(["0", "1"]), np.array([0.1, 0.2])),
    (np.array([[0, 1]]), np.array([[0.1, 0.2]])),
])
def test_invalid_inputs_raise_value_error(y: np.ndarray, p: np.ndarray) -> None:
    with pytest.raises(ValueError):
        M.brier(y, p)
    with pytest.raises(ValueError):
        M.performance_summary(y, p)


def test_accepts_pandas_series_and_booleans() -> None:
    y = pd.Series([True, False, True, False])
    p = pd.Series([0.8, 0.3, 0.6, 0.1], index=[10, 11, 12, 13])
    assert M.auroc(y, p) == 1.0
    with pytest.raises(ValueError):
        M.calibration_slope_intercept(y, np.array([0.1, np.inf, 0.2, 0.3]))


def test_lowess_tie_groups_that_fill_the_smoothing_window_are_undefined() -> None:
    # 80 of 100 rows share one risk: statsmodels would return a degenerate fit (radius 0, e.g. 1.0 where the true
    # rate is ~0.5), so the curve and indices are NaN with a warning instead of a silently wrong number.
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 100)
    p = np.r_[np.full(80, 0.2), np.linspace(0.3, 0.9, 20)]
    with pytest.warns(UndefinedMetricWarning, match="tied risk"):
        curve = M.smoothed_calibration(y, p)
    assert len(curve) == 100 and curve["observed_rate"].isna().all() and curve["mean_predicted"].notna().all()
    with pytest.warns(UndefinedMetricWarning, match="tied risk"):
        assert all(math.isnan(v) for v in M.calibration_indices(y, p).values())
    with pytest.warns(UndefinedMetricWarning, match="calibration_indices"):
        assert M.performance_summary(y, p)["calibration_indices"] == {"ici": None, "e50": None, "e90": None}
    # boundary: statsmodels uses k = int(0.75 * 100) = 75 neighbours; a 74-row tie is still a proper fit
    p74 = np.r_[np.full(74, 0.2), np.linspace(0.3, 0.9, 26)]
    fitted = lowess(y, p74, frac=0.75, it=0, delta=0.01 * np.ptp(p74), return_sorted=False)
    assert fitted[0] == pytest.approx(y[:74].mean(), abs=1e-12)
    assert set(M.calibration_indices(y, p74)) == {"ici", "e50", "e90"}
    with pytest.warns(UndefinedMetricWarning, match="tied risk"):
        M.calibration_indices(y, np.r_[np.full(75, 0.2), np.linspace(0.3, 0.9, 25)])


def test_smoothed_calibration_argument_validation() -> None:
    y, p, _ = _synthetic(n=200)
    for kwargs in ({"frac": 0.0}, {"frac": 1.5}, {"n_points": 1}, {"n_points": True}):
        with pytest.raises(ValueError):
            M.smoothed_calibration(y, p, **kwargs)
