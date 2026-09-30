"""Unit tests for Harrell's bootstrap optimism correction (spec §7.2, D-11). Synthetic data and stub pipelines."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit
from scipy.stats import t as student_t
from sklearn.linear_model import LogisticRegression

from falls_ml.errors import ConfigError, FallsMLError
from falls_ml.evaluation import metrics as M
from falls_ml.evaluation.optimism import OPTIMISM_COLUMNS, harrell_optimism
from falls_ml.models.base import ModelAdapter, coefficient_importance
from falls_ml.pipeline import FittedPipeline
from falls_ml.seeding import rng_for

SPEC = SimpleNamespace(outcome=SimpleNamespace(name="y"), identifier_columns=("research_id",),
                       predictor_names=lambda: ["x0"])


class ColumnsPreprocessor:
    def __init__(self, columns):
        self.columns = list(columns)

    def transform(self, df):
        return df[self.columns].astype(float)

    def raw_feature_of(self, column):
        return column

    def design_columns(self):
        return list(self.columns)


class Logistic(ModelAdapter):
    """Near-unpenalised logistic regression (lbfgs, C = 1e4)."""

    name = "logistic_stub"
    representation = "stub"
    is_linear = True

    def fit(self, X, y, *, groups=None):
        self.estimator = LogisticRegression(C=1e4, max_iter=5000).fit(X.to_numpy(), y)
        self.feature_names_ = list(X.columns)
        return self

    def linear_predictor(self, X):
        return self.estimator.decision_function(X.to_numpy())

    def predict_proba(self, X):
        return expit(self.linear_predictor(X))

    def get_feature_importance(self):
        return coefficient_importance(self.feature_names_, self.estimator.coef_[0], standardized_scale=False)

    def save(self, directory):
        raise NotImplementedError

    @classmethod
    def load(cls, directory):
        raise NotImplementedError


class FixedModel(Logistic):
    """LP = -0.5 + x0, whatever the training data."""

    name = "fixed_stub"

    def fit(self, X, y, *, groups=None):
        self.feature_names_ = list(X.columns)
        return self

    def linear_predictor(self, X):
        return -0.5 + X["x0"].to_numpy()


class ConstantModel(FixedModel):
    name = "constant_stub"

    def fit(self, X, y, *, groups=None):
        self.rate = float(np.mean(y))
        self.feature_names_ = list(X.columns)
        return self

    def linear_predictor(self, X):
        return np.full(len(X), logit(self.rate))


def fit_fn_for(model_cls, columns=("x0",), samples=None):
    def fit_fn(df, seed):
        if samples is not None:
            samples.append(df)
        pre = ColumnsPreprocessor(columns)
        model = model_cls(random_state=seed).fit(pre.transform(df), df["y"].to_numpy())
        return FittedPipeline(spec=SPEC, preprocessor=pre, model=model)
    return fit_fn


def make_frame(n: int, n_noise: int = 0, component: str = "frame", rows_per_patient: int = 1) -> pd.DataFrame:
    rng = rng_for(20260915, f"test_optimism_{component}")
    df = pd.DataFrame({"research_id": np.repeat([f"P{i:04d}" for i in range(n // rows_per_patient)], rows_per_patient),
                       "x0": rng.normal(size=n), **{f"noise{j}": rng.normal(size=n) for j in range(n_noise)}})
    df["y"] = rng.binomial(1, expit(-0.5 + df["x0"].to_numpy())).astype("int8")
    return df


def test_optimism_arithmetic_against_hand_computation():
    df = make_frame(400, rows_per_patient=2)
    samples: list[pd.DataFrame] = []
    metrics = ("auroc", "calibration_slope", "citl", "oe_ratio", "brier")
    table = harrell_optimism(df, fit_fn_for(FixedModel, samples=samples), n_bootstrap=12, seed=5, cluster_column="research_id",
                             spec=SPEC, metrics=metrics)
    assert list(table.columns) == OPTIMISM_COLUMNS and table["metric"].tolist() == list(metrics)
    assert len(samples) == 13 and samples[0] is df  # apparent model first, then one fit per replicate

    def perf(frame):
        y, lp = frame["y"].to_numpy(), -0.5 + frame["x0"].to_numpy()
        p = expit(lp)
        return {"auroc": M.auroc(y, p), "calibration_slope": M.calibration_slope_intercept(y, lp)[1], "citl": M.citl(y, lp)[0],
                "oe_ratio": M.oe_ratio(y, p), "brier": M.brier(y, p)}

    original = perf(df)
    replicates = [perf(sample) for sample in samples[1:]]
    for sample in samples[1:]:  # patient-level resamples: 200 whole patients drawn, original ids kept
        assert set(sample["research_id"]) <= set(df["research_id"]) and (sample.groupby("research_id").size() % 2 == 0).all()
        assert len(sample) == len(df) and sample["research_id"].nunique() < df["research_id"].nunique()  # with replacement
    rows = table.set_index("metric")
    for name in metrics:
        opt = np.array([r[name] - original[name] for r in replicates])
        mcse = opt.std(ddof=1) / np.sqrt(12)
        half = student_t.ppf(0.975, 11) * mcse
        row = rows.loc[name]
        assert row["apparent"] == pytest.approx(original[name], abs=1e-12)
        assert row["mean_optimism"] == pytest.approx(opt.mean(), abs=1e-12)
        assert row["optimism_mcse"] == pytest.approx(mcse, abs=1e-12)
        expected_ci = (opt.mean() - half, opt.mean() + half)
        assert (row["optimism_ci_low"], row["optimism_ci_high"]) == pytest.approx(expected_ci, abs=1e-12)
        assert row["corrected"] == pytest.approx(original[name] - opt.mean(), abs=1e-12)
        assert row["n_bootstrap"] == 12


def test_overfitted_development_process_shows_positive_optimism():
    df = make_frame(150, n_noise=20, component="overfit")
    columns = ["x0", *[f"noise{j}" for j in range(20)]]
    table = harrell_optimism(df, fit_fn_for(Logistic, columns), n_bootstrap=20, seed=1, cluster_column="research_id", spec=SPEC)
    rows = table.set_index("metric")
    assert rows.loc["auroc", "mean_optimism"] > 0.02 and rows.loc["auroc", "corrected"] < rows.loc["auroc", "apparent"]
    assert rows.loc["calibration_slope", "apparent"] == pytest.approx(1.0, abs=0.01)  # MLE on the development data
    assert rows.loc["calibration_slope", "mean_optimism"] > 0.1 and rows.loc["brier", "mean_optimism"] < 0
    assert rows.loc["auroc", "optimism_ci_low"] < rows.loc["auroc", "mean_optimism"] < rows.loc["auroc", "optimism_ci_high"]


def test_deterministic_for_a_fixed_seed():
    df = make_frame(200, n_noise=3, component="determinism")
    columns = ["x0", "noise0", "noise1", "noise2"]
    run = lambda seed: harrell_optimism(df, fit_fn_for(Logistic, columns), n_bootstrap=5, seed=seed, cluster_column="research_id",
                                        spec=SPEC, metrics=("auroc", "brier"))
    pd.testing.assert_frame_equal(run(4), run(4))
    assert not run(4).equals(run(5))


def test_metric_undefined_in_every_replicate_is_reported_as_nan():
    df = make_frame(200, component="constant")
    table = harrell_optimism(df, fit_fn_for(ConstantModel), n_bootstrap=5, seed=1, cluster_column="research_id", spec=SPEC,
                             metrics=("calibration_slope", "oe_ratio"))
    rows = table.set_index("metric")
    assert rows.loc["calibration_slope", "n_bootstrap"] == 0 and np.isnan(rows.loc["calibration_slope", "corrected"])
    assert rows.loc["oe_ratio", "n_bootstrap"] == 5 and np.isfinite(rows.loc["oe_ratio", "corrected"])


def test_too_many_single_class_resamples_raise():
    df = pd.DataFrame({"research_id": [f"P{i:02d}" for i in range(20)], "x0": np.linspace(-1, 1, 20), "y": [1] + [0] * 19})
    with pytest.raises(FallsMLError, match="bootstrap replicates failed"):
        harrell_optimism(df, fit_fn_for(FixedModel), n_bootstrap=20, seed=1, cluster_column="research_id", spec=SPEC)


@pytest.mark.parametrize("kwargs", [{"metrics": ("auroc", "c_index")}, {"metrics": ()}, {"n_bootstrap": 0},
                                    {"cluster_column": "patient"}])
def test_invalid_arguments(kwargs):
    args = {"n_bootstrap": 2, "seed": 1, "cluster_column": "research_id", "spec": SPEC, **kwargs}
    with pytest.raises(ConfigError):
        harrell_optimism(make_frame(50, component="invalid"), fit_fn_for(FixedModel), **args)
