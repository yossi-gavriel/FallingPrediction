"""Unit tests for FP forms, the logistic deviance fit and mfp closed-test selection (§6.1, D-13, RT-14)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import statsmodels.api as sm
from scipy.special import expit
from scipy.stats import chi2

import falls_ml.features.fractional_polynomial as fpmod
from falls_ml.errors import ConfigError, FallsMLError
from falls_ml.features.fractional_polynomial import (
    FPForm,
    FPFittingError,
    closed_test_decision,
    logistic_fit_deviance,
    select_fp_forms,
)
from falls_ml.features.transforms import fp_auto_scaling, fp_basis
from falls_ml.seeding import rng_for

POWERS = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0)
SEED = 20260914


def _binary_outcome(rng: np.random.Generator, eta: np.ndarray) -> np.ndarray:
    return rng.binomial(1, expit(eta)).astype(float)


# ---------------------------------------------------------------------- FPForm
def test_fp_form_transform_names_and_round_trip() -> None:
    form = FPForm("polypharmacy_count_120d", (0,), shift=1.0, scale=10.0, selection_table={"decision": "fp1"})
    x = np.array([0, 9, 61])
    np.testing.assert_allclose(form.transform(x)[:, 0], np.log((x + 1) / 10))
    assert form.column_names() == ["polypharmacy_count_120d__fp_p0"]
    restored = FPForm.from_dict(json.loads(json.dumps(form.to_dict())))
    assert restored == form and restored.powers == (0.0,)


def test_fp_form_linear_uses_raw_scale_d13() -> None:
    form = FPForm("age_years", (1,), 0.0, 1.0)
    np.testing.assert_array_equal(form.transform([65.5, 90.0]), [[65.5], [90.0]])
    with pytest.raises(ConfigError):
        FPForm("age_years", (1,), 0.0, 10.0)
    with pytest.raises(ConfigError):
        FPForm("x", (0,), 1.0, 0.0)


# ---------------------------------------------------------------------- logistic deviance
def _design(n: int = 4000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = rng_for(SEED, "test_fp_deviance")
    X = np.column_stack([rng.uniform(65, 95, n), rng.binomial(1, 0.3, n), rng.normal(size=n)])
    offset = rng.normal(scale=0.2, size=n)
    y = _binary_outcome(rng, -4.5 + 0.04 * X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2])
    return X, y, offset


def test_logistic_fit_deviance_matches_statsmodels_rt14() -> None:
    X, y, offset = _design()
    for off in (None, offset):
        dev, coef = logistic_fit_deviance(X, y, offset=off)
        ref = sm.GLM(y, sm.add_constant(X), family=sm.families.Binomial(), offset=off).fit(tol=1e-12)
        assert abs(dev - ref.deviance) < 1e-6
        np.testing.assert_allclose(coef, ref.params, atol=1e-6)


def test_logistic_fit_deviance_intercept_only_and_fp_basis_columns() -> None:
    X, y, _ = _design()
    null_ref = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial()).fit(tol=1e-12)
    assert abs(logistic_fit_deviance(np.empty((len(y), 0)), y)[0] - null_ref.deviance) < 1e-6
    B = fp_basis(X[:, 0] / 10, (-2, -2))  # poorly scaled, highly correlated columns
    ref = sm.GLM(y, sm.add_constant(B), family=sm.families.Binomial()).fit(tol=1e-13, maxiter=200)
    assert abs(logistic_fit_deviance(B, y)[0] - ref.deviance) < 1e-6


def test_logistic_fit_deviance_rank_deficient_design_keeps_deviance() -> None:
    X, y, _ = _design()
    dev, _ = logistic_fit_deviance(X, y)
    dev_dup, coef_dup = logistic_fit_deviance(np.column_stack([X, X[:, 1], 2 * X[:, 1] + 1]), y)
    assert abs(dev - dev_dup) < 1e-6 and np.isfinite(coef_dup).all()


def test_logistic_fit_deviance_separation_raises() -> None:
    rng = rng_for(SEED, "test_fp_separation")
    x = rng.normal(size=2000)
    with pytest.raises(FPFittingError, match="separation"):  # complete
        logistic_fit_deviance(x, (x > 0).astype(float))
    rare = np.zeros(2000)
    rare[:5] = 1
    y = _binary_outcome(rng, -1 + 0.5 * x)
    y[:5] = 0
    with pytest.raises(FPFittingError, match="separation"):  # quasi-complete (rare binary, no events)
        logistic_fit_deviance(np.column_stack([x, rare]), y)
    with pytest.raises(FPFittingError, match="single class"):
        logistic_fit_deviance(x, np.zeros(2000))
    assert issubclass(FPFittingError, FallsMLError)


@pytest.mark.parametrize("scale", [1.0, 1e3, 1e6])
def test_separation_detection_does_not_depend_on_column_scale(scale: float) -> None:
    """D-13: separation must stop FP selection whatever the units of the separating column."""
    rng = rng_for(SEED, "test_fp_separation_scale")
    n = 3000
    x = rng.normal(size=n)
    y = _binary_outcome(rng, -1 + 0.5 * x)
    rare = np.zeros(n)
    rare[:5] = 1
    y[:5] = 0
    with pytest.raises(FPFittingError, match="separation"):  # quasi-complete: rare indicator coded 0/scale
        logistic_fit_deviance(np.column_stack([x, rare * scale]), y)
    g = rng.integers(-1, 2, n).astype(float)
    y3 = np.where(g < 0, 0.0, np.where(g > 0, 1.0, rng.binomial(1, 0.5, n)))
    with pytest.raises(FPFittingError, match="separation"):  # quasi-complete on a 3-valued variable in large units
        logistic_fit_deviance(np.column_stack([x, g * scale]), y3)


@pytest.mark.parametrize("seed", [83, 85, 97])
def test_rare_level_with_mixed_outcomes_converges(seed: int) -> None:
    """A rare indicator with one event and one non-event has a finite MLE; undamped Newton overshoots it into
    saturation and step-halving cannot recover (regression test for 'step-halving failed')."""
    rng = rng_for(seed, "rare_mixed_search")
    n = 2000
    X = (rng.random((n, 8)) < rng.uniform(0.02, 0.3, 8)).astype(float)
    y = rng.binomial(1, expit(-3.0 + X @ rng.normal(scale=1.5, size=8))).astype(float)
    X[:, 0] = 0
    X[:2, 0] = 1
    y[0], y[1] = 1.0, 0.0
    X = X[:, X.std(axis=0) > 0]
    dev, coef = logistic_fit_deviance(X, y)
    ref = sm.GLM(y, sm.add_constant(X), family=sm.families.Binomial()).fit(tol=1e-12, maxiter=200)
    assert abs(dev - ref.deviance) < 1e-6
    np.testing.assert_allclose(coef, ref.params, atol=1e-5)


def test_numerically_singular_information_matrix_converges() -> None:
    """After an overshoot the rare level's rows saturate and A'WA has condition ~1e16; a least-squares step there
    was not a descent direction ('step-halving failed'). The ridge-stabilised step must reach the finite MLE."""
    rng = rng_for(35, "fuzz5")
    n, k = int(rng.integers(300, 20000)), int(rng.integers(1, 40))
    X = (rng.random((n, k)) < rng.uniform(0.0005, 0.3, k)).astype(float)
    X[:, 0] = 0
    X[:2, 0] = 1
    continuous = rng.normal(size=(n, 2)) * rng.choice([1e-3, 1, 1e3])
    y = _binary_outcome(rng, -2.5 + X @ rng.normal(size=k))
    y[0], y[1] = 1.0, 0.0
    design = np.column_stack([X[:, X.std(axis=0) > 0], continuous])
    dev, _ = logistic_fit_deviance(design, y)
    ref = sm.GLM(y, sm.add_constant(design), family=sm.families.Binomial()).fit(tol=1e-12, maxiter=500)
    assert abs(dev - ref.deviance) < 1e-6 and np.abs(ref.params).max() < 20  # finite MLE, same optimum


@pytest.mark.parametrize("scale", [1e-4, 1e4])
def test_logistic_fit_deviance_is_scale_invariant_without_false_separation(scale: float) -> None:
    X, y, _ = _design()
    dev, coef = logistic_fit_deviance(X, y)
    dev_scaled, coef_scaled = logistic_fit_deviance(X * scale, y)
    assert abs(dev - dev_scaled) < 1e-6
    np.testing.assert_allclose(coef_scaled[1:] * scale, coef[1:], rtol=1e-6)


def test_logistic_fit_deviance_non_convergence_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    X, y, _ = _design()
    monkeypatch.setattr(fpmod, "_MAX_ITER", 1)
    with pytest.raises(FPFittingError, match="no convergence"):
        logistic_fit_deviance(X, y)


# ---------------------------------------------------------------------- closed test (RT-14, injected deviances)
def test_closed_test_decision_injected_deviances() -> None:
    crit3 = chi2.isf(0.05, 3)  # 7.8147
    crit2 = chi2.isf(0.05, 2)  # 5.9915
    assert closed_test_decision(1000.0 + crit3 - 0.01, 1000.5, 1000.0, 0.05) == "linear"
    assert closed_test_decision(1000.0 + crit3 + 0.01, 1000.0 + crit2 - 0.01, 1000.0, 0.05) == "fp1"
    assert closed_test_decision(1000.0 + crit3 + 0.01, 1000.0 + crit2 + 0.01, 1000.0, 0.05) == "fp2"
    # p(FP2 vs linear) = chi2.sf(20, 3) = 1.7e-4; p(FP2 vs FP1) = chi2.sf(10, 2) = 6.7e-3
    assert closed_test_decision(1020.0, 1010.0, 1000.0, 0.01) == "fp2"
    assert closed_test_decision(1020.0, 1010.0, 1000.0, 0.005) == "fp1"
    assert closed_test_decision(1020.0, 1010.0, 1000.0, 1e-4) == "linear"
    assert closed_test_decision(999.999999, 999.9999995, 1000.0, 0.05) == "linear"  # numerical negatives
    with pytest.raises(ConfigError):
        closed_test_decision(1.0, 1.0, 1.0, 0.0)
    for bad in [(np.nan, 1000.0, 999.0), (1000.0, np.inf, 990.0), (1000.0, 999.0, np.nan)]:
        with pytest.raises(FPFittingError, match="non-finite"):  # NaN p-values must not select a non-linear form
            closed_test_decision(*bad, 0.05)


# ---------------------------------------------------------------------- selection (RT-14)
def _count_like(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.integers(0, 61, n).astype(float)


def test_select_recovers_log_truth_rt14() -> None:
    rng = rng_for(SEED, "test_fp_log_truth")
    n = 20000
    x = _count_like(rng, n)
    y = _binary_outcome(rng, -3.0 + 0.8 * np.log(x + 1))
    form = select_fp_forms({"poly": x}, None, y, powers=POWERS)["poly"]
    assert form.powers == (0.0,)
    assert (form.shift, form.scale) == fp_auto_scaling(x) == (1.0, 10.0)
    assert form.selection_table["decision"] == "fp1" and form.selection_table["converged"] is True


def test_select_keeps_linear_truth_on_raw_scale_rt14() -> None:
    rng = rng_for(SEED, "test_fp_linear_truth")
    n = 20000
    age = rng.uniform(65, 95, n)
    form = select_fp_forms({"age": age}, None, _binary_outcome(rng, -5.0 + 0.04 * age), powers=POWERS)["age"]
    assert form.powers == (1.0,) and (form.shift, form.scale) == (0.0, 1.0)
    assert form.selection_table["decision"] == "linear"
    assert form.selection_table["cycles_run"] == 1


def test_select_detects_strong_nonlinearity_rt14() -> None:
    rng = rng_for(SEED, "test_fp_u_shape")
    n = 20000
    x = rng.uniform(0.5, 3.0, n)
    form = select_fp_forms({"x": x}, np.empty((n, 0)), _binary_outcome(rng, -1.0 + 2.0 * (x - 1.75) ** 2),
                           powers=POWERS)["x"]
    assert form.selection_table["decision"] == "fp2" and len(form.powers) == 2
    assert (form.shift, form.scale) == fp_auto_scaling(x)
    assert form.selection_table["p_nonlinear"] < 1e-10


def test_select_backfitting_with_covariates_is_deterministic() -> None:
    rng = rng_for(SEED, "test_fp_backfit")
    n = 20000
    poly = _count_like(rng, n)
    age = rng.uniform(65, 95, n)
    binary = rng.binomial(1, 0.3, n).astype(float)
    other = np.column_stack([binary, np.ones(n), np.zeros(n)])  # two zero-variance columns are dropped
    y = _binary_outcome(rng, -6.0 + 0.04 * age + 0.8 * np.log(poly + 1) + 0.3 * binary)
    kwargs = dict(powers=POWERS, max_degree=2, alpha=0.05, max_cycles=5)
    forms = select_fp_forms({"poly": poly, "age": age}, other, y, **kwargs)
    again = select_fp_forms({"age": age, "poly": poly}, other, y, **kwargs)
    assert {v: f.to_dict() for v, f in forms.items()} == {v: f.to_dict() for v, f in again.items()}
    assert forms["poly"].powers == (0.0,) and forms["age"].powers == (1.0,)
    table = forms["age"].selection_table
    assert table["n_zero_variance_dropped"] == 2 and table["n_other_columns"] == 1
    assert table["converged"] is True and table["cycles_run"] == 2
    assert sorted(f.selection_table["xorder_rank"] for f in forms.values()) == [1, 2]
    json.dumps({v: f.to_dict() for v, f in forms.items()})


def test_select_searches_8_fp1_and_36_fp2_candidates_on_scaled_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    """§6.1: FP2 max = 8 FP1 + 36 FP2 models (repeated powers included), all on the pre-scaled variable."""
    rng = rng_for(SEED, "test_fp_candidates")
    n = 2000
    x = _count_like(rng, n)
    y = _binary_outcome(rng, -2.0 + 0.02 * x)
    calls: list[tuple[tuple[float, ...], float]] = []

    def recording_basis(z: np.ndarray, powers: tuple[float, ...]) -> np.ndarray:
        calls.append((tuple(float(p) for p in powers), float(np.min(z))))
        return fp_basis(z, powers)

    monkeypatch.setattr(fpmod, "fp_basis", recording_basis)
    form = select_fp_forms({"x": x}, None, y, powers=POWERS, max_cycles=1)["x"]
    searched = [powers for powers, _ in calls]
    assert sorted(p for p in searched if len(p) == 1) == [(p,) for p in POWERS]
    fp2 = [p for p in searched if len(p) == 2]
    assert len(fp2) == len(set(fp2)) == 36 and {(p, p) for p in POWERS} <= set(fp2)
    assert all(z_min == pytest.approx(0.1) for _, z_min in calls)  # (x + 1) / 10, x >= 0 (RT-13)
    assert form.selection_table["cycles_run"] == 1


def test_select_recovers_repeated_power_truth() -> None:
    rng = rng_for(SEED, "test_fp_repeated_truth")
    n = 20000
    x = rng.uniform(0.05, 10.0, n)  # wide range: on narrow ranges (-0.5, 0.5) is near-equivalent to (0, 0)
    ln = np.log(x)
    form = select_fp_forms({"x": x}, None, _binary_outcome(rng, -1.0 + 0.5 * ln + 0.6 * ln**2), powers=POWERS)["x"]
    assert (form.shift, form.scale) == (0.0, 1.0) and form.selection_table["decision"] == "fp2"
    assert form.powers == (0.0, 0.0) and form.column_names() == ["x__fp_p0", "x__fp_p0_ln"]


def test_xorder_ties_from_underflowing_p_values_use_lr_statistic() -> None:
    """With p-values underflowing to 0, the more significant variable must still come first (not name order)."""
    rng = rng_for(SEED, "test_fp_xorder")
    n = 20000
    strong = rng.uniform(0, 10, n)
    weak = rng.uniform(0, 10, n)
    y = _binary_outcome(rng, -6.0 + 0.9 * strong + 0.5 * weak)
    forms = select_fp_forms({"z_strong": strong, "a_weaker": weak}, None, y, powers=POWERS, max_degree=1, max_cycles=1)
    tables = {v: f.selection_table for v, f in forms.items()}
    assert tables["z_strong"]["xorder_p_value"] == tables["a_weaker"]["xorder_p_value"] == 0.0
    assert tables["z_strong"]["xorder_lr_statistic"] > tables["a_weaker"]["xorder_lr_statistic"]
    assert tables["z_strong"]["xorder_rank"] == 1 and tables["a_weaker"]["xorder_rank"] == 2


def test_select_records_non_convergence_of_cycles() -> None:
    rng = rng_for(SEED, "test_fp_log_truth")
    n = 20000
    x = _count_like(rng, n)
    y = _binary_outcome(rng, -3.0 + 0.8 * np.log(x + 1))
    form = select_fp_forms({"poly": x}, None, y, powers=POWERS, max_cycles=1)["poly"]
    assert form.selection_table["converged"] is False and form.selection_table["cycles_run"] == 1


def test_select_errors() -> None:
    rng = rng_for(SEED, "test_fp_errors")
    n = 3000
    x = rng.uniform(1, 5, n)
    y = _binary_outcome(rng, -1 + 0.2 * x)
    with pytest.raises(FPFittingError, match="constant"):
        select_fp_forms({"x": np.full(n, 2.0)}, None, y, powers=POWERS)
    separating = np.zeros(n)
    separating[np.flatnonzero(y == 0)[:3]] = 1
    with pytest.raises(FPFittingError, match="separation"):
        select_fp_forms({"x": x}, separating[:, None], y, powers=POWERS)
    with pytest.raises(ConfigError):
        select_fp_forms({"x": x}, None, y, powers=POWERS, max_degree=3)
    with pytest.raises(ConfigError):
        select_fp_forms({}, None, y, powers=POWERS)
