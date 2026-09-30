"""Unit tests for the Stata-equivalent LASSO logistic solver and CV adapter (spec §6.2, D-10, D-11, D-14; RT-15)."""

from __future__ import annotations

import json
import time
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from statsmodels.discrete.discrete_model import Logit

from falls_ml.errors import BundleIntegrityError, ConfigError, NotFittedError, PreprocessingMismatchError
from falls_ml.models import lasso_cv
from falls_ml.models.base import IMPORTANCE_COLUMNS
from falls_ml.models.lasso_cv import (
    LassoError,
    LassoLogisticCV,
    assign_folds,
    cv_select_lambda,
    lambda_grid,
    lambda_max,
    lasso_logistic_path,
    standardize,
    _unpenalized_refit,
)
from falls_ml.models.registry import get_adapter_class
from falls_ml.seeding import rng_for

FACTORS = {"bmi": ["underweight", "normal", "overweight", "obese", "missing"], "smoking": ["never", "ex", "current"]}


def make_design(n: int, n_binary: int = 6, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """Synthetic all-levels design (binary + continuous + one-hot factors) with a logistic outcome."""
    rng = np.random.default_rng(seed)
    cols: dict[str, np.ndarray] = {f"b{j}": (rng.random(n) < 0.05 + 0.03 * (j % 10)).astype(np.float64) for j in range(n_binary)}
    cols["age_years"] = rng.uniform(65, 95, n)
    cols["log_poly"] = np.log((rng.poisson(5, n) + 1) / 10)
    codes = {}
    for factor, levels in FACTORS.items():
        codes[factor] = rng.integers(0, len(levels), n)
        for i, level in enumerate(levels):
            cols[f"{factor}={level}"] = (codes[factor] == i).astype(np.float64)
    X = pd.DataFrame(cols)
    eta = (-5.0 + 0.05 * cols["age_years"] + 0.4 * cols["log_poly"] + 0.8 * cols["b0"] - 0.6 * cols["b1"]
           + 0.5 * cols["b2"] + np.array([0.6, 0.2, 0.0, 0.1, -0.2])[codes["bmi"]])
    y = (rng.random(n) < expit(eta)).astype(np.int64)
    return X, y


def kkt_violation(Z: np.ndarray, y: np.ndarray, lam: float, b0: float, beta: np.ndarray) -> float:
    g = Z.T @ (expit(b0 + Z @ beta) - y) / y.size
    nz = beta != 0
    stationarity = np.abs(g[nz] + lam * np.sign(beta[nz]))
    subgradient = np.abs(g[~nz]) - lam
    intercept = abs(float(np.mean(expit(b0 + Z @ beta) - y)))
    return float(max(np.max(stationarity, initial=0.0), np.max(subgradient, initial=0.0), intercept))


@pytest.fixture(scope="module")
def data() -> tuple[pd.DataFrame, np.ndarray]:
    return make_design(1200, seed=1)


@pytest.fixture(scope="module")
def fitted(data) -> LassoLogisticCV:
    X, y = data
    return LassoLogisticCV({"cv_folds": 5, "n_lambda": 40}, random_state=7).fit(X, y)


# ------------------------------------------------------------------ standardisation, grid, lambda_max
def test_standardize_uses_ddof0_and_neutralises_zero_variance():
    X = np.array([[1.0, 5.0, 2.0], [2.0, 5.0, 4.0], [3.0, 5.0, 9.0]])
    Z, mean, sd = standardize(X)
    np.testing.assert_allclose(mean, X.mean(axis=0))
    np.testing.assert_allclose(sd[[0, 2]], X[:, [0, 2]].std(axis=0, ddof=0))
    assert sd[1] == 1.0 and np.all(Z[:, 1] == 0.0)
    np.testing.assert_allclose(Z[:, [0, 2]].std(axis=0, ddof=0), 1.0)


def test_lambda_grid_is_stata_log_spaced():
    grid = lambda_grid(2.0, 100, 1e-4)
    assert grid.size == 100 and grid[0] == 2.0
    np.testing.assert_allclose(grid[-1], 2e-4, rtol=1e-12)
    np.testing.assert_allclose(np.diff(np.log(grid)), np.log(1e-4) / 99, rtol=1e-10)
    with pytest.raises(LassoError):
        lambda_grid(0.0)


def test_default_ratio_depends_on_p_versus_n():
    X, y = make_design(12, n_binary=6, seed=5)                      # p = 16 >= n = 12
    y = np.array([0, 1] * 6)
    X.loc[:, "b0"] = np.arange(12, dtype=float)
    model = LassoLogisticCV({"cv_folds": 2, "n_lambda": 5, "refit_unpenalized": False}).fit(X, y)
    assert model.fit_diagnostics()["lambda_ratio"] == 1e-2
    X2, y2 = make_design(300, seed=5)
    model2 = LassoLogisticCV({"cv_folds": 3, "n_lambda": 5, "refit_unpenalized": False}).fit(X2, y2)
    assert model2.fit_diagnostics()["lambda_ratio"] == 1e-4


def test_lambda_max_zeroes_all_coefficients_and_smaller_lambda_selects(data):
    X, y = data
    Z, _, _ = standardize(X.to_numpy())
    lmax = lambda_max(Z, y)
    path = lasso_logistic_path(Z, y, np.array([lmax, 0.99 * lmax]))
    assert np.all(path.beta[0] == 0.0)
    assert path.intercept[0] == pytest.approx(np.log(y.mean() / (1 - y.mean())), abs=1e-12)
    assert path.n_nonzero[1] >= 1


# ------------------------------------------------------------------ solver correctness (RT-15)
def test_kkt_conditions_hold_along_path(data):
    X, y = data
    Z, _, _ = standardize(X.to_numpy())
    grid = lambda_grid(lambda_max(Z, y), 60, 1e-4)
    path = lasso_logistic_path(Z, y, grid)
    assert path.converged.all()
    for k in (5, 20, 40, 59):
        assert kkt_violation(Z, y, grid[k], path.intercept[k], path.beta[k]) < 1e-5
    assert np.all(np.diff(path.n_nonzero[:10]) >= 0)
    assert np.all(np.diff(path.mean_deviance) <= 1e-12)


def test_agrees_with_sklearn_saga():
    rng = np.random.default_rng(3)
    n, p = 800, 8
    X = rng.normal(size=(n, p)) * rng.uniform(0.5, 5, p) + rng.uniform(-3, 3, p)
    X[:, 1] += 0.6 * X[:, 0]
    y = (rng.random(n) < expit(-1 + X @ rng.normal(0, 0.15, p))).astype(float)
    Z, _, _ = standardize(X)
    lams = np.array([0.5, 0.2, 0.05, 0.01]) * lambda_max(Z, y)
    path = lasso_logistic_path(Z, y, lams)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for k, lam in enumerate(lams):
            ref = LogisticRegression(C=1.0 / (n * lam), l1_ratio=1.0, solver="saga", tol=1e-10, max_iter=200_000,
                                     random_state=0).fit(Z, y)
            np.testing.assert_allclose(path.beta[k], ref.coef_[0], atol=1e-3)
            assert path.intercept[k] == pytest.approx(ref.intercept_[0], abs=1e-3)


def test_lambda_stop_index_follows_relative_deviance_rule(data):
    X, y = data
    Z, _, _ = standardize(X.to_numpy())
    grid = lambda_grid(lambda_max(Z, y), 100, 1e-4)
    path = lasso_logistic_path(Z, y, grid, stop=1e-5)
    k = path.lambda_stop_index
    assert k is not None and path.beta.shape[0] == 100          # recorded, not truncated
    rel = -np.diff(path.mean_deviance) / path.mean_deviance[:-1]
    assert rel[k - 1] < 1e-5
    assert np.all((rel[:k - 1] >= 1e-5) | (path.n_nonzero[1:k] == 0))


def test_path_rejects_invalid_inputs():
    Z = np.ones((4, 2))
    with pytest.raises(LassoError):
        lasso_logistic_path(Z, np.zeros(4), np.array([0.1]))
    with pytest.raises(LassoError):
        lasso_logistic_path(Z, np.array([0, 1, 0, 1.0]), np.array([0.1, 0.2]))
    Z_nan = np.array([[0.0, 1.0], [1.0, np.nan], [0.0, 0.0], [1.0, 1.0]])
    with pytest.raises(LassoError, match="non-finite"):
        lasso_logistic_path(Z_nan, np.array([0, 1, 0, 1.0]), np.array([0.1]))


def _rejecting_objective(accept_after: int | None):
    """Fake objective: every line-search candidate is worse than the incumbent, except (optionally) the
    ``accept_after``-th halving of each Newton step, which is accepted as a tiny damped step."""
    state = {"obj": 0.0, "calls": -1}

    def objective(eta, y, beta, lam):
        state["calls"] += 1
        if state["calls"] == 0:
            return state["obj"]
        if accept_after is not None and state["calls"] % (accept_after + 1) == 0:
            state["obj"] -= 1.0
            return state["obj"]
        return state["obj"] + 1.0
    return objective


@pytest.mark.parametrize("accept_after", [None, 30])
def test_failed_or_heavily_damped_line_search_is_not_reported_as_converged(data, monkeypatch, accept_after):
    """A tiny (or no) accepted step far from the optimum must not satisfy the convergence test."""
    X, y = data
    Z, _, _ = standardize(X.to_numpy())
    lam = 0.3 * lambda_max(Z, y)
    monkeypatch.setattr(lasso_cv, "_objective", _rejecting_objective(accept_after))
    path = lasso_logistic_path(Z, y, np.array([lam]), max_iter=40)
    assert not path.converged[0]


# ------------------------------------------------------------------ CV rule
@pytest.mark.parametrize(
    ("cv", "expected"),
    [
        ([1.10, 1.00, 1.002, 1.003, 1.004, 1.005, 1.006], (1, True)),       # five later points >= 1e-3 above
        ([1.10, 1.00, 1.002, 1.003, 1.004, 1.005, 1.0005], (1, False)),     # only four exceed the tolerance
        ([1.10, 1.05, 1.02, 1.00], (3, False)),                             # minimum at the end of the grid
        ([1.00, 1.00, 1.01, 1.01, 1.01, 1.01, 1.01], (0, True)),            # ties resolve to the largest lambda
    ],
)
def test_cv_select_lambda_cases(cv, expected):
    assert cv_select_lambda(np.array(cv), tol=1e-3, confirm=5) == expected


def test_cv_select_lambda_rejects_non_finite():
    with pytest.raises(LassoError):
        cv_select_lambda(np.array([1.0, np.nan]))


# ------------------------------------------------------------------ folds (D-11, D-14)
def test_ungrouped_folds_near_equal_and_deterministic():
    a = assign_folds(103, 10, rng_for(3, "lasso_cv_folds"))
    b = assign_folds(103, 10, rng_for(3, "lasso_cv_folds"))
    np.testing.assert_array_equal(a, b)
    sizes = np.bincount(a, minlength=10)
    assert sizes.max() - sizes.min() <= 1 and sizes.sum() == 103


def test_grouped_folds_never_split_a_group():
    rng = np.random.default_rng(0)
    ids = np.array([f"p{i}" for i in rng.integers(0, 150, 400)], dtype=object)   # bootstrap-like duplicates
    folds = assign_folds(ids.size, 10, rng_for(11, "lasso_cv_folds"), groups=ids)
    per_group = pd.DataFrame({"id": ids, "fold": folds}).groupby("id")["fold"].nunique()
    assert per_group.max() == 1
    sizes = np.bincount(folds, minlength=10)
    assert sizes.min() > 0 and sizes.max() - sizes.min() <= pd.Series(ids).value_counts().max()
    np.testing.assert_array_equal(folds, assign_folds(ids.size, 10, rng_for(11, "lasso_cv_folds"), groups=ids))
    with pytest.raises(LassoError):
        assign_folds(3, 2, rng_for(0, "x"), groups=np.array(["a", None, "b"], dtype=object))


# ------------------------------------------------------------------ adapter
def test_fit_is_deterministic_and_reports_diagnostics(data, fitted):
    X, y = data
    again = LassoLogisticCV({"cv_folds": 5, "n_lambda": 40}, random_state=7).fit(X, y)
    d1, d2 = fitted.fit_diagnostics(), again.fit_diagnostics()
    assert d1["lambda_star"] == d2["lambda_star"]
    pd.testing.assert_frame_equal(d1["cv_results"], d2["cv_results"])
    assert list(d1["cv_results"].columns) == ["lambda", "cv_mean_deviance", "cv_se", "n_nonzero", "selected"]
    assert d1["cv_results"]["selected"].sum() == 1
    k = int(np.argmin(d1["cv_results"]["cv_mean_deviance"]))
    assert d1["lambda_star"] == d1["cv_results"]["lambda"].iloc[k]
    for key in ("lambda_max", "lambda_ratio", "cv_minimum_identified", "fold_sizes", "standardized_coefficients",
                "intercept", "unpenalized_refit", "n_selected", "seed", "zero_variance_columns"):
        assert key in d1
    assert sum(d1["fold_sizes"]) == len(y) and d1["seed"] == 7
    assert d1["n_selected"] == int((fitted.coef_ != 0).sum()) > 0
    assert d1["path_converged"] and d1["cv_paths_converged"]
    other_seed = LassoLogisticCV({"cv_folds": 5, "n_lambda": 40, "refit_unpenalized": False}, random_state=8).fit(X, y)
    assert not np.array_equal(other_seed.cv_results_["cv_mean_deviance"], d1["cv_results"]["cv_mean_deviance"])


def test_unidentified_cv_minimum_uses_argmin_and_is_flagged(data, monkeypatch):
    """D-14: with no identified minimum, lambda* is still the CV argmin, flagged False and warned."""
    X, y = data
    events = []
    monkeypatch.setattr(lasso_cv.log, "warning", lambda event, **kwargs: events.append(event))
    model = LassoLogisticCV({"cv_folds": 5, "n_lambda": 40, "cv_confirm": 41, "refit_unpenalized": False}, random_state=7).fit(X, y)
    d = model.fit_diagnostics()
    assert d["cv_minimum_identified"] is False
    assert "lasso_cv_minimum_not_identified" in events
    cv = d["cv_results"]
    assert d["lambda_star_index"] == int(np.argmin(cv["cv_mean_deviance"]))
    assert d["lambda_star"] == cv["lambda"].iloc[d["lambda_star_index"]]


def test_unconverged_selected_solution_fails_loudly(data):
    X, y = data
    with pytest.raises(LassoError, match="did not converge"):
        LassoLogisticCV({"fixed_lambda": 1e-3, "max_iter": 1, "refit_unpenalized": False}).fit(X, y)


def test_cv_curve_matches_manual_within_fold_computation(data, fitted):
    """CV mean deviance = held-out -2 loglik / N with training-fold-only standardisation (D-14)."""
    X, y = data
    Xa = X.to_numpy()
    grid = fitted.cv_results_["lambda"].to_numpy()
    folds = assign_folds(len(y), 5, rng_for(7, "lasso_cv_folds"))
    dev = np.zeros((5, grid.size))
    for f in range(5):
        tr, te = folds != f, folds == f
        Z_tr, mu, sd = standardize(Xa[tr])
        path = lasso_logistic_path(Z_tr, y[tr], grid)
        eta = path.intercept + ((Xa[te] - mu) / sd) @ path.beta.T
        dev[f] = -2.0 * (y[te, None] * eta - np.logaddexp(0.0, eta)).sum(axis=0)
    np.testing.assert_allclose(fitted.cv_results_["cv_mean_deviance"], dev.sum(axis=0) / len(y), rtol=1e-12)
    per_fold = dev / np.bincount(folds)[:, None]
    np.testing.assert_allclose(fitted.cv_results_["cv_se"], per_fold.std(axis=0, ddof=1) / np.sqrt(5), rtol=1e-10)
    null_dev = -2.0 * np.mean(y * np.log(y.mean()) + (1 - y) * np.log(1 - y.mean()))
    assert fitted.cv_results_["cv_mean_deviance"].iloc[0] == pytest.approx(null_dev, rel=0.01)


def test_raw_coefficients_reproduce_standardized_predictions(data, fitted):
    X, y = data
    Z = (X.to_numpy() - fitted.mean_) / fitted.sd_
    lp_std = fitted.intercept_standardized_ + Z @ fitted.coef_standardized_
    np.testing.assert_allclose(fitted.linear_predictor(X), lp_std, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(fitted.predict_proba(X), expit(lp_std), rtol=1e-10)


def test_final_coefficients_equal_full_data_path_at_lambda_star(data, fitted):
    X, y = data
    Z, _, _ = standardize(X.to_numpy())
    grid = fitted.cv_results_["lambda"].to_numpy()
    k = int(np.flatnonzero(fitted.cv_results_["selected"])[0])
    path = lasso_logistic_path(Z, y, grid)
    np.testing.assert_allclose(fitted.coef_standardized_, path.beta[k], atol=1e-12)


def test_grouped_fit_and_fold_standardization_options(data):
    X, y = data
    groups = np.repeat(np.arange(len(y) // 2), 2)
    within = LassoLogisticCV({"cv_folds": 4, "n_lambda": 20, "refit_unpenalized": False}, random_state=1).fit(X, y, groups=groups)
    full = LassoLogisticCV({"cv_folds": 4, "n_lambda": 20, "refit_unpenalized": False, "fold_standardization": "full_sample"},
                           random_state=1).fit(X, y, groups=groups)
    assert all(size % 2 == 0 for size in within.fit_diagnostics()["fold_sizes"])
    assert full.fit_diagnostics()["fold_standardization"] == "full_sample"

    # full_sample sensitivity: training folds reuse the full-sample standardisation (manual recomputation)
    Z, _, _ = standardize(X.to_numpy())
    grid = full.cv_results_["lambda"].to_numpy()
    folds = assign_folds(len(y), 4, rng_for(1, "lasso_cv_folds"), groups=groups)
    dev = np.zeros(grid.size)
    for f in range(4):
        tr, te = folds != f, folds == f
        path = lasso_logistic_path(Z[tr], y[tr], grid)
        eta = path.intercept + Z[te] @ path.beta.T
        dev += -2.0 * (y[te, None] * eta - np.logaddexp(0.0, eta)).sum(axis=0)
    np.testing.assert_allclose(full.cv_results_["cv_mean_deviance"], dev / len(y), rtol=1e-12)
    assert not np.allclose(full.cv_results_["cv_mean_deviance"], within.cv_results_["cv_mean_deviance"], rtol=1e-9, atol=0)


@pytest.mark.parametrize("mode", ["within_fold", "full_sample"])
def test_column_constant_within_a_training_fold(data, mode):
    """A rare indicator whose ones all sit in one held-out fold is constant in the other training folds."""
    X, y = data
    folds = assign_folds(len(y), 5, rng_for(2, "lasso_cv_folds"))
    rare = np.zeros(len(y))
    rare[np.flatnonzero(folds == 3)[:2]] = 1.0
    model = LassoLogisticCV({"cv_folds": 5, "n_lambda": 30, "refit_unpenalized": False, "fold_standardization": mode},
                            random_state=2).fit(X.assign(rare=rare), y)
    d = model.fit_diagnostics()
    assert np.isfinite(d["cv_results"]["cv_mean_deviance"]).all() and d["cv_paths_converged"]
    assert d["zero_variance_columns"] == []


def test_one_se_rule_selects_larger_or_equal_lambda(data, fitted):
    X, y = data
    one_se = LassoLogisticCV({"cv_folds": 5, "n_lambda": 40, "selection": "1se"}, random_state=7).fit(X, y)
    assert one_se.fit_diagnostics()["lambda_star"] >= fitted.fit_diagnostics()["lambda_star"]


def test_fixed_lambda_skips_cv(data):
    X, y = data
    model = LassoLogisticCV({"fixed_lambda": 0.002, "refit_unpenalized": False}).fit(X, y)
    d = model.fit_diagnostics()
    assert d["lambda_star"] == 0.002 and d["cv_minimum_identified"] is None and d["fold_sizes"] == []
    assert d["cv_results"]["cv_mean_deviance"].isna().all()


def test_reference_reexpression_preserves_predictions(data, fitted):
    X, _ = data
    refs = {"bmi": "overweight", "smoking": "never"}
    intercept, coefs = fitted.coefficients_relative_to_reference(refs)
    assert coefs["bmi=overweight"] == 0.0 and coefs["smoking=never"] == 0.0
    np.testing.assert_allclose(intercept + X.to_numpy() @ coefs.to_numpy(), fitted.linear_predictor(X), atol=1e-10)
    with pytest.raises(ConfigError):
        fitted.coefficients_relative_to_reference({"bmi": "enormous"})


def test_unpenalized_refit_matches_statsmodels_and_drops_collinearity(data):
    X, y = data
    names = list(X.columns)
    all_selected = np.ones(len(names), dtype=bool)                     # all levels selected => exact collinearity
    refit = _unpenalized_refit(X.to_numpy(), y.astype(float), names, all_selected, None)
    dropped = refit.loc[refit["status"] == "dropped_collinear", "term"].tolist()
    assert dropped == ["bmi=missing", "smoking=current"]              # last dependent column in design order
    est = refit[refit["status"] == "estimated"]
    cols = [t for t in est["term"] if t != "_cons"]
    ref = Logit(y, np.column_stack([np.ones(len(y)), X[cols].to_numpy()])).fit(disp=False)
    np.testing.assert_allclose(est["coefficient"].to_numpy(), ref.params, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(est["or_ci_low"].to_numpy(), np.exp(ref.conf_int()[:, 0]), rtol=1e-6)

    refs = {"bmi": "overweight", "smoking": "never"}
    table = _unpenalized_refit(X.to_numpy(), y.astype(float), names, all_selected, refs).set_index("term")
    assert table.loc["bmi=overweight", "status"] == "reference_level"
    assert table.loc["bmi=missing", "status"] == "estimated"
    assert (table["status"] == "dropped_collinear").sum() == 0

    model = LassoLogisticCV({"fixed_lambda": 1e-4, "refit_reference_levels": refs}).fit(X, y)
    adapter_refit = model.fit_diagnostics()["unpenalized_refit"]
    assert set(adapter_refit["term"]) == {"_cons"} | {c for c, b in zip(names, model.coef_) if b != 0}
    assert np.isfinite(adapter_refit.loc[adapter_refit["status"] == "estimated", "se"]).all()


def test_zero_variance_column_gets_zero_coefficient_and_is_reported(data):
    X, y = data
    X = X.assign(constant_code=1.0)
    model = LassoLogisticCV({"cv_folds": 3, "n_lambda": 20}).fit(X, y)
    d = model.fit_diagnostics()
    assert d["zero_variance_columns"] == ["constant_code"]
    assert model.coef_[-1] == 0.0 and d["standardized_coefficients"]["constant_code"] == 0.0
    assert np.isfinite(model.predict_proba(X)).all()


def test_feature_importance_uses_standardized_magnitude(fitted):
    imp = fitted.get_feature_importance()
    assert list(imp.columns) == IMPORTANCE_COLUMNS
    np.testing.assert_allclose(imp["native_importance"], np.abs(fitted.coef_standardized_))
    np.testing.assert_allclose(imp["coefficient"], fitted.coef_)


def test_save_load_roundtrip(tmp_path, data, fitted):
    X, _ = data
    fitted.save(tmp_path / "adapter")
    assert {"adapter.json", "coefficients.npz", "cv_results.csv"} <= {p.name for p in (tmp_path / "adapter").iterdir()}
    loaded = LassoLogisticCV.load(tmp_path / "adapter")
    np.testing.assert_array_equal(loaded.predict_proba(X), fitted.predict_proba(X))
    assert loaded.fit_diagnostics()["lambda_star"] == fitted.fit_diagnostics()["lambda_star"]
    assert loaded.params == fitted.params
    pd.testing.assert_frame_equal(loaded.cv_results_, fitted.cv_results_, check_exact=False, rtol=1e-15)
    meta = (tmp_path / "adapter" / "adapter.json")
    meta.write_text(meta.read_text(encoding="utf-8").replace('"adapter": "lasso_logistic_cv"', '"adapter": "other"'), encoding="utf-8")
    with pytest.raises(BundleIntegrityError):
        LassoLogisticCV.load(tmp_path / "adapter")


def test_load_rejects_corrupt_bundles_with_typed_error(tmp_path, fitted):
    directory = tmp_path / "adapter"
    fitted.save(directory)
    npz = directory / "coefficients.npz"
    npz.write_bytes(npz.read_bytes()[: len(npz.read_bytes()) // 2])
    with pytest.raises(BundleIntegrityError):
        LassoLogisticCV.load(directory)

    fitted.save(directory)
    meta = json.loads((directory / "adapter.json").read_text(encoding="utf-8"))
    del meta["params"]
    (directory / "adapter.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(BundleIntegrityError, match="params"):
        LassoLogisticCV.load(directory)


def test_contract_errors(data, fitted):
    X, y = data
    assert get_adapter_class("lasso_logistic_cv") is LassoLogisticCV
    with pytest.raises(ConfigError):
        LassoLogisticCV({"folds": 10})
    with pytest.raises(ConfigError):
        LassoLogisticCV({"selection": "max"})
    for bad in ({"tol": float("nan")}, {"tol": "abc"}, {"stop": float("inf")}, {"max_iter": None}, {"cv_folds": True},
                {"cv_folds": 10.0}, {"fixed_lambda": -1.0}, {"lambda_min_ratio": 2.0}, {"refit_unpenalized": "yes"}):
        with pytest.raises(ConfigError):
            LassoLogisticCV(bad)
    assert LassoLogisticCV({"tol": "1e-7"}).params["tol"] == "1e-7"          # PyYAML reads 1e-7 as a string
    with pytest.raises(NotFittedError):
        LassoLogisticCV().predict_proba(X)
    with pytest.raises(PreprocessingMismatchError):
        fitted.predict_proba(X[X.columns[::-1]])
    with pytest.raises(LassoError):
        LassoLogisticCV().fit(X.assign(b0=np.where(np.arange(len(y)) == 0, np.nan, X["b0"])), y)
    with pytest.raises(LassoError):
        LassoLogisticCV().fit(X, np.zeros(len(y)))


def test_efalls_scale_path_and_cv_runtime():
    """n = 5000, p = 90 full path plus 10-fold CV within 60 s."""
    rng = np.random.default_rng(42)
    X, _ = make_design(5000, n_binary=80, seed=42)
    assert X.shape == (5000, 90)
    eta = -3.0 + X.iloc[:, :20].to_numpy() @ rng.normal(0, 0.4, 20)
    y = (rng.random(5000) < expit(eta)).astype(np.int64)
    start = time.perf_counter()
    model = LassoLogisticCV({"refit_unpenalized": False}, random_state=3).fit(X, y)
    elapsed = time.perf_counter() - start
    assert elapsed < 60, f"path + 10-fold CV took {elapsed:.1f} s"
    d = model.fit_diagnostics()
    assert d["n_lambda"] == 100 and len(d["fold_sizes"]) == 10 and d["path_converged"]


def test_non_convergence_is_a_bounded_degenerate_failure_inside_resampling_loops() -> None:
    """A non-converged full-data solve fails the main fit loudly (LassoError) but is a DegenerateFitError for the bootstrap
    stability/optimism loops, which record and bound such replicate failures instead of aborting the run."""
    from falls_ml.errors import DegenerateFitError
    from falls_ml.models.lasso_cv import LassoConvergenceError, LassoError

    assert issubclass(LassoConvergenceError, LassoError) and issubclass(LassoConvergenceError, DegenerateFitError)
    assert not issubclass(LassoError, DegenerateFitError)
