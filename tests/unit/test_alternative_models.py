"""Unit tests for the L4 alternative adapters: unpenalised and elastic-net logistic, random forest, HGB (arch. §3.2)."""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import pytest
import sklearn
import statsmodels.api as sm
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from falls_ml.errors import BundleIntegrityError, ConfigError, DatasetValidationError, NotFittedError, PreprocessingMismatchError
from falls_ml.models.hist_gbm import HistGradientBoostingModel
from falls_ml.models.logistic import ElasticNetLogistic, ModelFitError, UnpenalizedLogistic
from falls_ml.models.random_forest import RandomForestModel
from falls_ml.seeding import rng_for


def make_design(n: int = 800, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = rng_for(seed, "test_alternative_models_design")
    X = pd.DataFrame({"age_years": rng.uniform(65.0, 95.0, n), "falls": rng.binomial(1, 0.3, n),
                      "dementia": rng.binomial(1, 0.2, n), "polypharmacy": rng.poisson(5.0, n)}).astype("float64")
    eta = -6.0 + 0.06 * X["age_years"] + 0.8 * X["falls"] - 0.5 * X["dementia"] + 0.05 * X["polypharmacy"]
    return X, rng.binomial(1, expit(eta.to_numpy())).astype(np.int64)


@pytest.fixture(scope="module")
def data() -> tuple[pd.DataFrame, np.ndarray]:
    return make_design()


# ---------------------------------------------------------------------- unpenalised logistic
class TestUnpenalizedLogistic:
    def test_matches_statsmodels_glm_and_reports_wald_table(self, data):
        X, y = data
        model = UnpenalizedLogistic().fit(X, y)
        glm = sm.GLM(y, sm.add_constant(X.to_numpy()), family=sm.families.Binomial()).fit()
        np.testing.assert_allclose([model.intercept_, *model.coef_], glm.params, atol=1e-6)
        table = model.fit_diagnostics()["coefficients"]
        assert table["term"].tolist() == ["intercept", *X.columns]
        np.testing.assert_allclose(table["se"], glm.bse, rtol=1e-6)
        np.testing.assert_allclose(table["ci_low"], table["coefficient"] - 1.959963984540054 * table["se"], rtol=1e-9)
        np.testing.assert_allclose(table["or_ci_high"], np.exp(table["ci_high"]), rtol=1e-12)
        assert model.fit_diagnostics()["converged"] is True

    def test_importance_uses_standardised_coefficients(self, data):
        X, y = data
        model = UnpenalizedLogistic().fit(X, y)
        imp = model.get_feature_importance()
        np.testing.assert_allclose(imp["native_importance"], np.abs(model.coef_ * X.std(ddof=0).to_numpy()))
        np.testing.assert_allclose(imp["odds_ratio"], np.exp(model.coef_))

    def test_collinear_design_raises(self, data):
        X, y = data
        with pytest.raises(ModelFitError, match="singular"):
            UnpenalizedLogistic().fit(X.assign(copy=X["falls"]), y)

    def test_constant_column_is_omitted_like_stata(self, data):
        X, y = data
        model = UnpenalizedLogistic().fit(X.assign(copy=1.0), y)
        omissions = model.fit_diagnostics()["stata_logit_omissions"]
        assert omissions["constant_columns"] == ["copy"] and omissions["n_rows_dropped"] == 0
        assert model.coef_[list(model.feature_names_).index("copy")] == 0.0
        reference = UnpenalizedLogistic().fit(X, y)
        np.testing.assert_allclose(model.predict_proba(X.assign(copy=1.0)), reference.predict_proba(X), rtol=1e-10)

    def test_perfect_predictor_indicator_is_dropped_with_its_rows(self, data):
        X, y = data
        rare = np.zeros(len(X))
        negatives = np.flatnonzero(np.asarray(y) == 0)[:5]
        rare[negatives] = 1.0                       # indicator 1 only among non-events: predicts failure perfectly
        model = UnpenalizedLogistic().fit(X.assign(rare=rare), y)
        pp = model.fit_diagnostics()["stata_logit_omissions"]["perfect_predictors"]
        assert pp == [{"column": "rare", "level": 1, "predicts_outcome": 0, "n_rows_dropped": 5}]
        assert model.fit_diagnostics()["n_obs"] == len(X) - 5

    def test_perfect_separation_raises(self, data):
        X, _ = data
        with pytest.raises(ModelFitError):
            UnpenalizedLogistic().fit(X, X["falls"].to_numpy(dtype=np.int64))

    def test_non_convergence_raises(self, data):
        with pytest.raises(ModelFitError):
            UnpenalizedLogistic({"maxiter": 1}).fit(*data)

    @pytest.mark.parametrize("params", [{"alpha": 1.0}, {"maxiter": 0}, {"maxiter": True}])
    def test_invalid_params_raise(self, params):
        with pytest.raises(ConfigError):
            UnpenalizedLogistic(params)

    def test_invalid_inputs_raise(self, data):
        X, y = data
        with pytest.raises(DatasetValidationError):
            UnpenalizedLogistic().fit(X, np.zeros_like(y))
        with pytest.raises(PreprocessingMismatchError):
            UnpenalizedLogistic().fit(X.assign(age_years=np.nan), y)
        with pytest.raises(NotFittedError):
            UnpenalizedLogistic().predict_proba(X)

    def test_save_load_keeps_wald_table_and_diagnostics(self, data, tmp_path):
        model = UnpenalizedLogistic().fit(*data)
        model.save(tmp_path)
        loaded = UnpenalizedLogistic.load(tmp_path)
        original, restored = model.fit_diagnostics(), loaded.fit_diagnostics()
        pd.testing.assert_frame_equal(restored.pop("coefficients"), original.pop("coefficients"), check_exact=True)
        assert restored == original

    def test_load_refuses_other_adapter(self, data, tmp_path):
        UnpenalizedLogistic().fit(*data).save(tmp_path)
        with pytest.raises(BundleIntegrityError):
            ElasticNetLogistic.load(tmp_path)


# ---------------------------------------------------------------------- elastic net
class TestElasticNetLogistic:
    def test_defaults_and_search_space(self):
        assert ElasticNetLogistic().params == {"C": 1.0, "l1_ratio": 0.5}
        assert ElasticNetLogistic.default_search_space() == {"C": [0.001, 0.01, 0.1, 1.0, 10.0], "l1_ratio": [0.1, 0.5, 0.9]}

    def test_equals_saga_on_standardised_design_without_future_warnings(self, data):
        X, y = data
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            warnings.simplefilter("error", DeprecationWarning)
            model = ElasticNetLogistic({"C": 0.05, "l1_ratio": 0.7}, random_state=5).fit(X, y)
            Z = (X - X.mean()) / X.std(ddof=0)
            reference = LogisticRegression(C=0.05, l1_ratio=0.7, solver="saga", tol=1e-6, max_iter=20000, random_state=5).fit(Z.to_numpy(), y)
        np.testing.assert_allclose(model.predict_proba(X), reference.predict_proba(Z.to_numpy())[:, 1], atol=1e-7)
        np.testing.assert_allclose(model.coef_standardized_, reference.coef_[0], atol=1e-12)
        assert abs(model.fit_diagnostics()["intercept_refit_shift"]) < 1e-6
        np.testing.assert_allclose(model.get_feature_importance()["native_importance"], np.abs(reference.coef_[0]), atol=1e-12)

    @pytest.mark.parametrize("l1_ratio", [0.1, 0.5, 0.9])
    def test_solution_satisfies_elastic_net_kkt_conditions(self, data, l1_ratio):
        """Independent of scikit-learn: the fit minimises sum(logloss) + (1 - r)/(2C)||w||^2 + r/C ||w||_1 on the
        ddof=0-standardised design with an unpenalised intercept (score equation sum(y - p) = 0)."""
        X, y = data
        Z = ((X - X.mean()) / X.std(ddof=0)).to_numpy()
        C = 1.5 * l1_ratio / np.abs(Z.T @ (y - y.mean())).max()  # just below lambda_max: some coefficients stay 0
        model = ElasticNetLogistic({"C": C, "l1_ratio": l1_ratio}).fit(X, y)
        w = model.coef_standardized_
        assert (w == 0).any() and (w != 0).any()
        p = expit(model.intercept_standardized_ + Z @ w)
        score = Z.T @ (y - p)
        l1, l2 = l1_ratio / C, (1.0 - l1_ratio) / C
        np.testing.assert_allclose(score[w != 0], l2 * w[w != 0] + l1 * np.sign(w[w != 0]), rtol=0, atol=1e-3)
        assert np.all(np.abs(score[w == 0]) <= l1 + 1e-3)
        assert abs(np.sum(y - p)) < 1e-8
        np.testing.assert_allclose(model.predict_proba(X), p, rtol=0, atol=1e-12)

    def test_weak_penalty_approaches_maximum_likelihood(self, data):
        X, y = data
        enet = ElasticNetLogistic({"C": 1e4, "l1_ratio": 0.5}).fit(X, y)
        mle = UnpenalizedLogistic().fit(X, y)
        np.testing.assert_allclose(enet.coef_, mle.coef_, atol=2e-3)

    def test_strong_lasso_penalty_selects_nothing(self, data):
        X, y = data
        model = ElasticNetLogistic({"C": 1e-4, "l1_ratio": 1.0}).fit(X, y)
        assert not model.get_feature_importance()["selected"].any()
        assert model.fit_diagnostics()["n_selected"] == 0
        np.testing.assert_allclose(model.predict_proba(X), y.mean(), rtol=1e-10)  # null model predicts the prevalence

    def test_rescaling_a_column_leaves_predictions_unchanged(self, data):
        X, y = data
        base = ElasticNetLogistic({"C": 0.1}).fit(X, y)
        days = ElasticNetLogistic({"C": 0.1}).fit(X.assign(age_years=X["age_years"] * 365.25), y)
        np.testing.assert_allclose(days.predict_proba(X.assign(age_years=X["age_years"] * 365.25)), base.predict_proba(X), atol=1e-8)
        np.testing.assert_allclose(days.coef_standardized_, base.coef_standardized_, atol=1e-8)

    def test_constant_column_gets_zero_coefficient(self, data):
        X, y = data
        model = ElasticNetLogistic().fit(X.assign(constant=1.0), y)
        assert model.coef_[-1] == 0.0 and model.fit_diagnostics()["constant_columns"] == ["constant"]

    def test_non_convergence_raises(self, data, monkeypatch):
        monkeypatch.setattr(ElasticNetLogistic, "MAX_ITER", 1)
        with pytest.raises(ModelFitError, match="converge"):
            ElasticNetLogistic().fit(*data)

    @pytest.mark.parametrize("params", [{"C": 0.0}, {"C": -1.0}, {"C": "1"}, {"C": float("inf")}, {"l1_ratio": 1.5},
                                        {"l1_ratio": None}, {"penalty": "l1"}])
    def test_invalid_params_raise(self, params):
        with pytest.raises(ConfigError):
            ElasticNetLogistic(params)

    def test_saved_json_is_standard(self, data, tmp_path):
        ElasticNetLogistic().fit(*data).save(tmp_path)
        meta = json.loads((tmp_path / "adapter.json").read_text(encoding="utf-8"), parse_constant=lambda c: pytest.fail(c))
        assert meta["adapter"] == "elastic_net_logistic" and len(meta["coef"]) == 4


# ---------------------------------------------------------------------- tree ensembles
class TestRandomForest:
    def test_defaults_importance_and_space(self, data):
        X, y = data
        model = RandomForestModel(random_state=3).fit(X, y)
        params = model.estimator_.get_params()
        assert {k: params[k] for k in ("n_estimators", "max_depth", "min_samples_leaf", "max_features", "n_jobs", "random_state")} == \
            {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 50, "max_features": "sqrt", "n_jobs": 1, "random_state": 3}
        imp = model.get_feature_importance()
        np.testing.assert_allclose(imp["native_importance"], model.estimator_.feature_importances_)
        assert imp["native_importance"].sum() == pytest.approx(1.0)
        assert imp["coefficient"].isna().all() and imp["selected"].isna().all()
        assert RandomForestModel.default_search_space() == {"n_estimators": [300], "max_depth": [4, 8, None],
                                                            "min_samples_leaf": [20, 100], "max_features": ["sqrt", 0.33]}

    def test_seed_changes_the_forest(self, data):
        X, y = data
        a = RandomForestModel({"n_estimators": 20}, random_state=1).fit(X, y).predict_proba(X)
        b = RandomForestModel({"n_estimators": 20}, random_state=2).fit(X, y).predict_proba(X)
        assert not np.array_equal(a, b)

    def test_invalid_params_raise(self, data):
        with pytest.raises(ConfigError, match="unknown"):
            RandomForestModel({"n_jobs": 4})
        with pytest.raises(ConfigError, match="invalid"):
            RandomForestModel({"n_estimators": 5, "min_samples_leaf": 0}).fit(*data)

    def test_load_refuses_other_sklearn_version_and_corrupt_files(self, data, tmp_path):
        RandomForestModel({"n_estimators": 5}).fit(*data).save(tmp_path)
        meta_path = tmp_path / "adapter.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta["sklearn_version"] == sklearn.__version__
        meta_path.write_text(json.dumps({**meta, "sklearn_version": "0.0.1"}), encoding="utf-8")
        with pytest.raises(BundleIntegrityError, match="scikit-learn"):
            RandomForestModel.load(tmp_path)
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        with pytest.raises(BundleIntegrityError):
            HistGradientBoostingModel.load(tmp_path)
        (tmp_path / "estimator.pkl").write_bytes(b"not a pickle")
        with pytest.raises(BundleIntegrityError):
            RandomForestModel.load(tmp_path)
        (tmp_path / "estimator.pkl").write_bytes(b"cfalls_ml_no_such_module\nForest\n.")  # pickle naming a missing class
        with pytest.raises(BundleIntegrityError):
            RandomForestModel.load(tmp_path)
        (tmp_path / "estimator.pkl").write_bytes(b"csklearn.ensemble\nNoSuchForest\n.")
        with pytest.raises(BundleIntegrityError):
            RandomForestModel.load(tmp_path)


class TestHistGradientBoosting:
    def test_defaults_no_early_stopping_and_nan_importance(self, data):
        X, y = data
        model = HistGradientBoostingModel({"max_iter": 30}, random_state=4).fit(X, y)
        params = model.estimator_.get_params()
        expected = {"learning_rate": 0.05, "max_iter": 30, "max_leaf_nodes": 31, "max_depth": None, "min_samples_leaf": 100,
                    "l2_regularization": 0.0, "max_features": 1.0, "early_stopping": False, "random_state": 4}
        assert {k: params[k] for k in expected} == expected
        assert model.fit_diagnostics()["n_iter"] == 30
        assert model.get_feature_importance()["native_importance"].isna().all()
        assert HistGradientBoostingModel().params["max_iter"] == 300

    def test_search_space_and_fixed_early_stopping(self):
        assert HistGradientBoostingModel.default_search_space() == {
            "learning_rate": [0.03, 0.1], "max_iter": [200, 500], "max_leaf_nodes": [15, 31], "min_samples_leaf": [50, 200],
            "l2_regularization": [0.0, 1.0], "max_features": [0.5, 1.0]}
        with pytest.raises(ConfigError, match="unknown"):
            HistGradientBoostingModel({"early_stopping": True})
