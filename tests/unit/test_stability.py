"""Unit tests for bootstrap model stability (spec §7.2, D-11, D-13; ARTIFACT_SCHEMAS feature_stability.csv,
instability.csv). Synthetic data and small stub pipelines; one test uses the real fit_pipeline."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression

from falls_ml.config import experiment_config_from_dict
from falls_ml.errors import ConfigError, DatasetValidationError, FallsMLError
from falls_ml.evaluation import stability as stability_module
from falls_ml.evaluation.stability import (
    FEATURE_STABILITY_COLUMNS,
    INSTABILITY_COLUMNS,
    bootstrap_stability,
    resample_clusters,
)
from falls_ml.features.spec import load_feature_spec
from falls_ml.models.base import ModelAdapter, coefficient_importance, empty_importance
from falls_ml.pipeline import FittedPipeline, fit_pipeline
from falls_ml.seeding import rng_for

REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "configs" / "features" / "efalls_v1.yaml"
SEED = 20260915
FEATURES = ["x_strong", "x_noise1", "x_noise2"]


# ---------------------------------------------------------------------- stubs
class StubSpec:
    outcome = SimpleNamespace(name="y")
    identifier_columns = ("research_id",)

    def __init__(self, predictors: list[str]):
        self._predictors = list(predictors)

    def predictor_names(self) -> list[str]:
        return list(self._predictors)


class IdentityPreprocessor:
    def __init__(self, columns: list[str]):
        self.columns = list(columns)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[self.columns].astype(float)

    def raw_feature_of(self, column: str) -> str:
        return column.split("^")[0]

    def design_columns(self) -> list[str]:
        return list(self.columns)


class StubAdapter(ModelAdapter):
    representation = "stub"

    def save(self, directory):
        raise NotImplementedError

    @classmethod
    def load(cls, directory):
        raise NotImplementedError


class L1Logistic(StubAdapter):
    """Small L1-penalised logistic regression (liblinear) used as a stand-in for the LASSO development process."""

    name = "l1_stub"
    is_linear = True

    def fit(self, X, y, *, groups=None):
        self.estimator = LogisticRegression(C=self.params.get("C", 0.05), l1_ratio=1.0, solver="liblinear",
                                            random_state=0).fit(X.to_numpy(), y)
        self.feature_names_ = list(X.columns)
        return self

    def linear_predictor(self, X):
        return self.estimator.decision_function(X.to_numpy())

    def predict_proba(self, X):
        return expit(self.linear_predictor(X))

    def get_feature_importance(self):
        return coefficient_importance(self.feature_names_, self.estimator.coef_[0], standardized_scale=False)

    def fit_diagnostics(self):
        return {"lambda_star": 1.0 / self.params.get("C", 0.05)}


class L1NonLinear(L1Logistic):
    name = "l1_nonlinear_stub"
    is_linear = False

    def get_feature_importance(self):
        return empty_importance(self.feature_names_)


class MeanRateModel(StubAdapter):
    """Intercept-only model (risk = observed event rate); column 'x' gets coefficient 0."""

    name = "mean_rate_stub"
    is_linear = True

    def fit(self, X, y, *, groups=None):
        self.intercept = logit(y.mean())
        self.feature_names_ = ["x"]
        return self

    def linear_predictor(self, X):
        return np.full(len(X), self.intercept)

    def predict_proba(self, X):
        return expit(self.linear_predictor(X))

    def get_feature_importance(self):
        return coefficient_importance(["x"], np.array([0.0]), standardized_scale=False)


class BinaryRateModel(StubAdapter):
    """Linear model reproducing the event rate within each level of one binary column (row-order independent)."""

    name = "rate_stub"
    is_linear = True

    def fit(self, X, y, *, groups=None):
        x = X["x"].to_numpy()
        r0, r1 = y[x == 0].sum() / (x == 0).sum(), y[x == 1].sum() / (x == 1).sum()
        self.intercept, self.coef = logit(r0), logit(r1) - logit(r0)
        self.feature_names_ = ["x"]
        return self

    def linear_predictor(self, X):
        return self.intercept + self.coef * X["x"].to_numpy()

    def predict_proba(self, X):
        return expit(self.linear_predictor(X))

    def get_feature_importance(self):
        return coefficient_importance(["x"], np.array([self.coef]), standardized_scale=False)


class ScriptedLinear(StubAdapter):
    """Linear stub whose coefficients are dictated by the test (``params`` = {design column: coefficient}); constant risk."""

    name = "scripted_stub"
    is_linear = True

    def fit(self, X, y, *, groups=None):
        self.feature_names_ = list(self.params)
        return self

    def linear_predictor(self, X):
        return np.full(len(X), -0.5)

    def predict_proba(self, X):
        return expit(self.linear_predictor(X))

    def get_feature_importance(self):
        return coefficient_importance(self.feature_names_, np.array(list(self.params.values()), dtype=float), standardized_scale=False)


def make_fit_fn(model_cls=L1Logistic, columns=FEATURES, params=None, calls: list | None = None):
    def fit_fn(df: pd.DataFrame, seed: int) -> FittedPipeline:
        if calls is not None:
            calls.append({"ids": df["research_id"].to_numpy(), "seed": seed, "n": len(df)})
        pre = IdentityPreprocessor(columns)
        model = model_cls(params, random_state=seed).fit(pre.transform(df), df["y"].to_numpy())
        return FittedPipeline(spec=StubSpec(sorted({c.split("^")[0] for c in columns})), preprocessor=pre, model=model)
    return fit_fn


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    n = 300
    rng = rng_for(SEED, "test_stability_frame")
    df = pd.DataFrame({"research_id": pd.Series([f"P{i:04d}" for i in range(n)], dtype="str"),
                       **{c: rng.normal(size=n) for c in FEATURES}})
    df["y"] = rng.binomial(1, expit(-0.3 + 1.2 * df["x_strong"].to_numpy())).astype("int8")
    return df


def run(frame, fit_fn=None, **kwargs):
    args = {"n_bootstrap": 40, "seed": 3, "cluster_column": "research_id", "thresholds": (0.3, 0.5), **kwargs}
    return bootstrap_stability(frame, fit_fn or make_fit_fn(), **args)


# ---------------------------------------------------------------------- resampling
class TestResampleClusters:
    def test_whole_clusters_with_original_ids_and_same_number_of_draws(self):
        sizes = [1, 2, 3] * 10
        df = pd.DataFrame({"research_id": np.repeat([f"P{i:02d}" for i in range(30)], sizes), "v": np.arange(sum(sizes))})
        sample = resample_clusters(df, "research_id", rng_for(1, "test"))
        assert isinstance(sample.index, pd.RangeIndex) and list(sample.columns) == ["research_id", "v"]
        original_sizes = df.groupby("research_id").size()
        counts = sample.groupby("research_id").size()
        assert set(counts.index) <= set(original_sizes.index)
        multiplicity = counts / original_sizes[counts.index]
        assert (multiplicity == multiplicity.round()).all() and multiplicity.sum() == 30
        for rid, group in sample.groupby("research_id"):  # duplicated rows are exact copies of the cluster's rows
            assert sorted(group["v"]) == sorted(np.repeat(df.loc[df["research_id"] == rid, "v"], int(multiplicity[rid])))

    def test_missing_cluster_ids_and_unknown_column_are_refused(self):
        df = pd.DataFrame({"research_id": ["a", None, "b"], "y": [0, 1, 0]})
        with pytest.raises(DatasetValidationError):
            resample_clusters(df, "research_id", rng_for(1, "test"))
        with pytest.raises(ConfigError):
            resample_clusters(df, "patient", rng_for(1, "test"))


# ---------------------------------------------------------------------- stability summaries
class TestBootstrapStability:
    def test_selection_frequency_strong_versus_noise(self, frame):
        result = run(frame)
        fs = result.feature_stability.set_index("feature")
        assert list(result.feature_stability.columns) == FEATURE_STABILITY_COLUMNS
        assert result.n_bootstrap == 40 and result.n_failed == 0 and (fs["n_bootstrap"] == 40).all()
        assert fs.loc["x_strong", "selection_frequency"] == 1.0 and bool(fs.loc["x_strong", "robust"])
        assert fs.loc["x_strong", "sign_stability"] == 1.0 and fs.loc["x_strong", "coef_mean"] > 0.3
        for noise in ("x_noise1", "x_noise2"):
            assert fs.loc[noise, "selection_frequency"] <= 0.2 and not bool(fs.loc[noise, "robust"])
        coefs = result.bootstrap_coefficients
        assert list(coefs.columns) == FEATURES and len(coefs) == 40
        for column in FEATURES:
            values = coefs[column].to_numpy()
            f = np.mean(values != 0)
            assert fs.loc[column, "n_selected"] == np.count_nonzero(values)
            assert fs.loc[column, "selection_frequency_mcse"] == pytest.approx(np.sqrt(f * (1 - f) / 40))
            assert fs.loc[column, "coef_sd"] == pytest.approx(np.std(values, ddof=1))  # zeros included
            assert fs.loc[column, "coef_median"] == pytest.approx(np.median(values))
        assert fs["permutation_importance_mean"].isna().all()

    def test_selection_sign_stability_and_robust_arithmetic_with_scripted_coefficients(self, frame):
        script = {"a": [1.0] * 8 + [-1.0, 0.0],   # selected 9/10, modal sign 8/9 < 0.9 -> not robust
                  "b": [2.0] * 8 + [0.0, 0.0],    # selection exactly 0.8, sign 1.0 -> robust (>= boundary)
                  "c": [1.0] * 9 + [-1.0],        # selection 1.0, sign exactly 0.9 -> robust (>= boundary)
                  "d": [0.0] * 10,                # never selected -> sign stability NaN, not robust
                  "e": [-3.0] * 7 + [0.0] * 3}    # selection 0.7 < 0.8 -> not robust
        columns = list(script)
        replicates = iter([{c: values[b] for c, values in script.items()} for b in range(10)])
        df = frame.assign(**dict.fromkeys(columns, 0.0))

        def pipeline(coefs):
            return FittedPipeline(spec=StubSpec(columns), preprocessor=IdentityPreprocessor(columns), model=ScriptedLinear(coefs).fit(None, None))

        result = bootstrap_stability(df, lambda sample, seed: pipeline(next(replicates)), n_bootstrap=10, seed=4,
                                     cluster_column="research_id", thresholds=[0.3, 0.5],
                                     reference_pipeline=pipeline(dict.fromkeys(columns, 1.0)))
        fs = result.feature_stability.set_index("feature")
        assert fs.index.tolist() == columns and result.n_bootstrap == 10
        expected = {"a": (9, 8 / 9, False), "b": (8, 1.0, True), "c": (10, 0.9, True), "d": (0, np.nan, False), "e": (7, 1.0, False)}
        for column, (n_selected, sign, robust) in expected.items():
            values = np.array(script[column])
            row = fs.loc[column]
            f = n_selected / 10
            assert row["n_selected"] == n_selected and row["selection_frequency"] == pytest.approx(f, abs=1e-15)
            assert row["raw_feature_selection_frequency"] == row["selection_frequency"]  # one design column per raw feature
            assert row["selection_frequency_mcse"] == pytest.approx(np.sqrt(f * (1 - f) / 10), abs=1e-15)
            assert row["sign_stability"] == pytest.approx(sign, abs=1e-15, nan_ok=True)
            assert row["coef_mean"] == pytest.approx(values.mean(), abs=1e-15)
            assert row["coef_median"] == pytest.approx(np.median(values), abs=1e-15)
            assert row["coef_sd"] == pytest.approx(values.std(ddof=1), abs=1e-15)
            assert bool(row["robust"]) is robust
        np.testing.assert_array_equal(result.bootstrap_coefficients.to_numpy(), np.column_stack([script[c] for c in columns]))
        assert (result.instability[["mape_mean", "max_classification_instability"]] == 0.0).all().all()

    def test_floating_point_residue_coefficients_are_counted_per_contract_and_flagged(self, frame, monkeypatch):
        class Standardized(ScriptedLinear):
            def fit_diagnostics(self):
                return {"standardized_coefficients": pd.Series(self.params)}

        warnings_logged: list[tuple[str, dict]] = []
        monkeypatch.setattr(stability_module, "log", SimpleNamespace(
            info=lambda *a, **k: None, warning=lambda event, extra_fields=None: warnings_logged.append((event, extra_fields))))
        replicates = iter([{"a": 0.5, "b": 1e-17}, {"a": 0.5, "b": 0.0}, {"a": 0.4, "b": -2e-16}])

        def pipeline(coefs):
            return FittedPipeline(spec=StubSpec(["a", "b"]), preprocessor=IdentityPreprocessor(["a", "b"]),
                                  model=Standardized(coefs).fit(None, None))

        result = bootstrap_stability(frame.assign(a=0.0, b=0.0), lambda sample, seed: pipeline(next(replicates)), n_bootstrap=3,
                                     seed=4, cluster_column="research_id", thresholds=[0.3],
                                     reference_pipeline=pipeline({"a": 1.0, "b": 1.0}))
        assert result.feature_stability.set_index("feature").loc["b", "n_selected"] == 2  # contract: selected = coefficient != 0
        flagged = dict(warnings_logged)["stability_coefficients_at_numerical_zero"]
        assert flagged["replicates_per_design_column"] == {"b": 2}

    def test_invalid_permutation_settings_fail_before_any_fit(self, frame):
        calls: list[dict[str, Any]] = []
        for permutation in ({"metric": "pr_auc"}, {"features": []}, {"features": ["x_strong", "x_strong"]}, {"n_repeats": 0},
                            {"max_rows": 0}):
            with pytest.raises(ConfigError):
                run(frame, make_fit_fn(calls=calls), n_bootstrap=3, permutation=permutation)
        assert calls == []
        one_class = frame.assign(y=np.r_[np.ones(10, dtype="int8"), np.zeros(len(frame) - 10, dtype="int8")])
        reference = make_fit_fn()(one_class, 0)
        with pytest.raises(ConfigError, match="single outcome class"):  # AUROC undefined on the permutation rows
            run(one_class, make_fit_fn(calls=calls), n_bootstrap=3, reference_pipeline=reference,
                permutation={"features": FEATURES, "n_repeats": 1, "max_rows": 1})
        assert calls == []

    def test_mape_is_zero_for_deterministic_fit_on_identical_clusters(self):
        block = pd.DataFrame({"x": [0, 0, 1, 1, 1], "y": [0, 1, 1, 1, 0]})
        df = pd.concat([block.assign(research_id=f"P{i:02d}") for i in range(20)], ignore_index=True)
        result = bootstrap_stability(df, make_fit_fn(BinaryRateModel, ["x"]), n_bootstrap=15, seed=1,
                                     cluster_column="research_id", thresholds=[0.4, 0.6])
        assert list(result.instability.columns) == INSTABILITY_COLUMNS
        assert np.all(result.mape_individual == 0.0) and result.mape_individual.shape == (len(df),)
        assert (result.instability[["mean_classification_instability", "max_classification_instability", "mape_mean",
                                    "mape_mean_mcse", "mape_median"]] == 0.0).all().all()
        assert result.feature_stability.loc[0, "coef_sd"] == pytest.approx(0.0, abs=1e-12)
        assert result.feature_stability.loc[0, "selection_frequency"] == 1.0

    def test_mape_and_classification_instability_match_direct_computation(self, frame):
        reference = make_fit_fn()(frame, 0)
        predictions = []
        base = make_fit_fn()

        def spy(df, seed):
            pipeline = base(df, seed)
            predictions.append(pipeline.predict_proba(frame))
            return pipeline

        thresholds = (0.3, 0.45, 0.6)
        result = run(frame, spy, n_bootstrap=25, reference_pipeline=reference, thresholds=thresholds)
        P, p_ref = np.vstack(predictions), reference.predict_proba(frame)
        assert len(predictions) == 25  # the reference is not refitted when given
        np.testing.assert_allclose(result.mape_individual, np.abs(P - p_ref).mean(axis=0), rtol=0, atol=1e-15)
        per_rep = np.abs(P - p_ref).mean(axis=1)
        inst = result.instability.set_index("threshold")
        for t in thresholds:
            disagree = ((P >= t) != (p_ref >= t)).mean(axis=0)
            assert inst.loc[t, "mean_classification_instability"] == pytest.approx(disagree.mean(), abs=1e-15)
            assert inst.loc[t, "max_classification_instability"] == pytest.approx(disagree.max(), abs=1e-15)
            assert inst.loc[t, "mape_mean"] == pytest.approx(per_rep.mean(), abs=1e-15)
            assert inst.loc[t, "mape_mean_mcse"] == pytest.approx(per_rep.std(ddof=1) / np.sqrt(25), abs=1e-15)
            assert inst.loc[t, "mape_median"] == pytest.approx(np.median(np.abs(P - p_ref).mean(axis=0)), abs=1e-15)
        assert inst["mean_classification_instability"].max() > 0

    def test_fit_fn_gets_resamples_with_original_ids_and_results_are_deterministic(self, frame):
        calls: list[dict[str, Any]] = []
        first = run(frame, make_fit_fn(calls=calls), n_bootstrap=10)
        assert calls[0]["seed"] == 3 and calls[0]["n"] == len(frame)  # reference = fit_fn(train_df, seed)
        original = set(frame["research_id"])
        for call in calls[1:]:
            assert set(call["ids"]) <= original and pd.Series(call["ids"]).duplicated().any()
        assert len({call["seed"] for call in calls[1:]}) == 10
        again = run(frame, n_bootstrap=10)
        pd.testing.assert_frame_equal(first.feature_stability, again.feature_stability)
        pd.testing.assert_frame_equal(first.instability, again.instability)

    def test_design_columns_absent_from_a_replicate_count_as_not_selected(self, frame):
        df = frame.assign(**{"x_strong^2": frame["x_strong"] ** 2})

        def fit_fn(sample, seed):  # the "FP" term chosen for x_strong alternates between replicates (D-13)
            term = "x_strong" if seed % 2 == 0 else "x_strong^2"
            return make_fit_fn(columns=[term, "x_noise1"], params={"C": 1.0})(sample, seed)

        result = run(df, fit_fn, n_bootstrap=20, reference_pipeline=make_fit_fn(columns=["x_strong", "x_noise1"])(df, 0))
        fs = result.feature_stability.set_index("feature")
        assert fs.index.tolist() == ["x_strong", "x_noise1", "x_strong^2"]
        assert fs.loc["x_strong^2", "raw_feature"] == "x_strong"
        assert fs.loc["x_strong", "selection_frequency"] + fs.loc["x_strong^2", "selection_frequency"] == pytest.approx(1.0)
        assert (result.bootstrap_coefficients[["x_strong", "x_strong^2"]] == 0).sum(axis=1).eq(1).all()
        # some form of x_strong is selected in every replicate, although each form alone is selected about half the time
        assert fs.loc["x_strong", "raw_feature_selection_frequency"] == fs.loc["x_strong^2", "raw_feature_selection_frequency"] == 1.0
        assert fs.loc["x_noise1", "raw_feature_selection_frequency"] == fs.loc["x_noise1", "selection_frequency"]

    def test_raw_feature_selection_frequency_counts_any_design_column_once_per_replicate(self, frame):
        script = {"a^1": [1.0, 0.0, 1.0, 0.0], "a^2": [0.0, 0.0, -1.0, 2.0], "b": [0.0, 1.0, 0.0, 0.0], "c": [0.0] * 4}
        columns = list(script)
        replicates = iter([{c: values[b] for c, values in script.items()} for b in range(4)])
        df = frame.assign(**dict.fromkeys(columns, 0.0))

        def pipeline(coefs):
            return FittedPipeline(spec=StubSpec(["a", "b", "c"]), preprocessor=IdentityPreprocessor(columns),
                                  model=ScriptedLinear(coefs).fit(None, None))

        result = bootstrap_stability(df, lambda sample, seed: pipeline(next(replicates)), n_bootstrap=4, seed=4,
                                     cluster_column="research_id", thresholds=[0.5],
                                     reference_pipeline=pipeline(dict.fromkeys(columns, 1.0)))
        fs = result.feature_stability.set_index("feature")
        assert fs["selection_frequency"].to_dict() == {"a^1": 0.5, "a^2": 0.5, "b": 0.25, "c": 0.0}
        assert fs["raw_feature_selection_frequency"].to_dict() == {"a^1": 0.75, "a^2": 0.75, "b": 0.25, "c": 0.0}

    def test_permutation_importance_enters_robustness(self, frame):
        perm = {"features": FEATURES, "n_repeats": 2, "metric": "auroc", "max_rows": 200}
        result = run(frame, n_bootstrap=10, permutation=perm)
        fs = result.feature_stability.set_index("feature")
        assert fs.loc["x_strong", "permutation_importance_mean"] - 2 * fs.loc["x_strong", "permutation_importance_std"] > 0
        assert bool(fs.loc["x_strong", "robust"])
        assert abs(fs.loc["x_noise1", "permutation_importance_mean"]) < 0.02 and not bool(fs.loc["x_noise1", "robust"])
        with pytest.raises(ConfigError):
            run(frame, n_bootstrap=2, permutation={"features": FEATURES, "repeats": 2})

    def test_non_linear_model_has_one_row_per_raw_feature_without_coefficients(self, frame):
        perm = {"features": ["x_strong", "x_noise1"], "n_repeats": 1}
        result = run(frame, make_fit_fn(L1NonLinear), n_bootstrap=5, permutation=perm)
        fs = result.feature_stability
        assert list(fs.columns) == FEATURE_STABILITY_COLUMNS and fs["feature"].tolist() == FEATURES
        assert fs[["selection_frequency", "raw_feature_selection_frequency", "coef_mean", "coef_sd", "sign_stability"]].isna().all().all()
        assert fs["n_selected"].isna().all() and result.bootstrap_coefficients.empty
        assert np.isnan(fs.loc[2, "permutation_importance_mean"])  # not requested
        assert fs.loc[0, "permutation_importance_mean"] > 0 and not fs["robust"].fillna(False).any()
        assert result.instability["mape_mean"].iloc[0] > 0

    def test_single_class_resamples_are_recorded_failures(self):
        df = pd.DataFrame({"research_id": [f"P{i:03d}" for i in range(60)], "x": [1, 0] * 30, "y": [1, 0, 1, 1] + [0] * 56})
        calls: list[dict[str, Any]] = []
        # 3 event clusters of 60: P(resample without an event) = (57/60)^60 ~ 4.6%, so a few of the 100 resamples fail
        result = bootstrap_stability(df, make_fit_fn(MeanRateModel, ["x"], calls=calls), n_bootstrap=100, seed=3,
                                     cluster_column="research_id", thresholds=[0.1])
        assert result.n_failed == len(result.failures) >= 1
        assert result.n_bootstrap + result.n_failed == 100 and len(calls) == 1 + result.n_bootstrap
        assert all(f["reason"] == "single outcome class in bootstrap sample" for f in result.failures)
        assert (result.feature_stability["n_bootstrap"] == result.n_bootstrap).all()

    def test_more_than_ten_percent_failures_raise_and_other_errors_propagate(self):
        df = pd.DataFrame({"research_id": [f"P{i:02d}" for i in range(20)], "x": [1, 0] * 10, "y": [1] + [0] * 19})
        fit_fn = make_fit_fn(MeanRateModel, ["x"])
        reference = fit_fn(df, 0)
        with pytest.raises(FallsMLError, match="bootstrap replicates failed"):
            bootstrap_stability(df, fit_fn, n_bootstrap=20, seed=1, cluster_column="research_id", thresholds=[0.1],
                                reference_pipeline=reference)

        def broken(sample, seed):
            raise ValueError("boom")
        with pytest.raises(ValueError, match="boom"):
            bootstrap_stability(df.assign(y=[1, 0] * 10), broken, n_bootstrap=3, seed=1, cluster_column="research_id",
                                thresholds=[0.1], reference_pipeline=reference)

    @pytest.mark.parametrize("kwargs", [{"n_bootstrap": 0}, {"thresholds": []}, {"thresholds": [1.0]}, {"thresholds": [0.2, 0.2]},
                                        {"selection_threshold": 1.5}, {"cluster_column": "patient"}])
    def test_invalid_arguments(self, frame, kwargs):
        with pytest.raises(ConfigError):
            run(frame, **kwargs)


# ---------------------------------------------------------------------- real pipeline
def test_real_fit_pipeline_bootstrap_stability():
    spec = load_feature_spec(SPEC_PATH).subset(["age_years", "sex", "polypharmacy_count_120d", "falls", "dementia"])
    n = 800
    rng = rng_for(SEED, "test_stability_real")
    ids = [f"R{i:04d}" for i in range(n // 2)]
    df = pd.DataFrame({
        "research_id": pd.Series(np.repeat(ids, 2), dtype="str"),  # two index dates per patient
        "index_date": np.tile(pd.to_datetime(["2017-04-01", "2018-04-01"]), n // 2),
        "age_years": rng.uniform(65.0, 100.0, n),
        "sex": pd.Series(rng.choice(["female", "male"], n), dtype="str"),
        "polypharmacy_count_120d": rng.poisson(6.0, n).astype("int64"),
        "falls": rng.binomial(1, 0.25, n).astype("int8"),
        "dementia": rng.binomial(1, 0.2, n).astype("int8"),
    })
    df["outcome_12m"] = rng.binomial(1, expit(-1.5 + 0.04 * (df["age_years"] - 80.0) + 1.5 * df["falls"]).to_numpy()).astype("int8")
    config = experiment_config_from_dict({
        "experiment": {"name": "unit", "kind": "alternative_model", "description": "unit test", "layers": ["L4_alternative"]},
        "dataset": {"feature_spec": str(SPEC_PATH)}, "model": {"name": "logistic_unpenalized"},
        "preprocessing": {"representation": "efalls_reference_coded"},
        "validation": {"strategy": "patient_grouped_random", "seed": 1, "limitation_note": "unit test"}})

    def fit_fn(train: pd.DataFrame, seed: int) -> FittedPipeline:
        return fit_pipeline(train, spec, config, random_state=seed)

    result = bootstrap_stability(df, fit_fn, n_bootstrap=4, seed=9, cluster_column="research_id", thresholds=(0.1, 0.25),
                                 spec=spec, permutation={"features": ["falls", "dementia"], "n_repeats": 1, "max_rows": 300})
    reference = fit_fn(df, 9)
    fs = result.feature_stability
    assert fs["feature"].tolist() == reference.preprocessor.design_columns()
    assert fs["raw_feature"].tolist() == [reference.preprocessor.raw_feature_of(c) for c in fs["feature"]]
    assert (fs["selection_frequency"] == 1.0).all() and (fs["raw_feature_selection_frequency"] == 1.0).all() and result.n_failed == 0
    assert fs.set_index("feature").loc["falls", "permutation_importance_mean"] > 0
    assert result.instability["threshold"].tolist() == [0.1, 0.25] and 0 < result.instability["mape_mean"].iloc[0] < 0.2
