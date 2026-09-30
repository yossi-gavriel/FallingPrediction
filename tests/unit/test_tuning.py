"""Unit tests for bounded hyperparameter search and the composite objective (architecture §4 step 4, D-19 §1)."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from falls_ml.errors import ConfigError, DatasetValidationError, FallsMLError
from falls_ml.evaluation import metrics as M
from falls_ml.features.spec import FeatureSpec, load_feature_spec
from falls_ml.seeding import rng_for
from falls_ml.tuning import MAX_TRIALS, RESULT_COLUMNS, composite_objective, expand_search_space, run_search

REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "configs" / "features" / "efalls_v1.yaml"
WEIGHTS = {"auroc": 1.0, "brier": -1.0, "calibration_slope_abs_error": -0.25}


@pytest.fixture(scope="module")
def spec() -> FeatureSpec:
    return load_feature_spec(SPEC_PATH)


def partition(n: int, prefix: str, index_date: str) -> pd.DataFrame:
    """Rows with a single 'signal' column and outcome ~ Bernoulli(expit(-1 + signal))."""
    rng = rng_for(20260915, f"test_tuning_{prefix}")
    signal = rng.normal(size=n)
    return pd.DataFrame({"research_id": pd.Series([f"{prefix}-{i}" for i in range(n)], dtype="str"),
                         "index_date": pd.Timestamp(index_date), "signal": signal,
                         "outcome_12m": rng.binomial(1, expit(-1.0 + signal)).astype("int8")})


@pytest.fixture(scope="module")
def train() -> pd.DataFrame:
    return partition(500, "TRAIN", "2018-04-01")


@pytest.fixture(scope="module")
def validation() -> pd.DataFrame:
    return partition(3000, "VAL", "2019-04-01")


class StubPipeline:
    """Duck-typed FittedPipeline with LP = -1 + slope * signal."""

    def __init__(self, params: dict[str, Any], *, linear: bool = True):
        self.slope = float(params.get("slope", 1.0))
        self.linear = linear

    def _lp(self, df: pd.DataFrame) -> np.ndarray:
        return -1.0 + self.slope * df["signal"].to_numpy()

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return expit(self._lp(df))

    def linear_predictor(self, df: pd.DataFrame) -> np.ndarray | None:
        return self._lp(df) if self.linear else None


def search(train, validation, spec, *, space, method="grid", n_iter=None, weights=WEIGHTS, linear=True, fit_fn=None):
    fit_fn = fit_fn or (lambda df, params: StubPipeline(params, linear=linear))
    return run_search(train, validation, fit_fn=fit_fn, space=space, method=method, n_iter=n_iter, seed=7,
                      weights=weights, spec=spec)


# ---------------------------------------------------------------------- search-space expansion
class TestExpandSearchSpace:
    SPACE = {"max_depth": [4, 8, None], "C": [0.1, 1.0], "max_features": ["sqrt", 0.33]}

    def test_grid_is_cartesian_product_over_sorted_keys(self):
        trials = expand_search_space(self.SPACE, "grid")
        keys = sorted(self.SPACE)
        assert trials == [dict(zip(keys, combo)) for combo in itertools.product(*(self.SPACE[k] for k in keys))]
        assert all(list(t) == keys for t in trials) and len(trials) == 12

    def test_random_is_distinct_deterministic_and_bounded_by_grid(self):
        grid = expand_search_space(self.SPACE, "grid")
        a = expand_search_space(self.SPACE, "random", n_iter=5, seed=3)
        assert a == expand_search_space(self.SPACE, "random", n_iter=5, seed=3)
        assert a != expand_search_space(self.SPACE, "random", n_iter=5, seed=4)
        assert len(a) == 5 and len({json.dumps(t, sort_keys=True) for t in a}) == 5
        assert all(t in grid for t in a)
        full = expand_search_space(self.SPACE, "random", n_iter=100, seed=3)
        assert sorted(map(json.dumps, full)) == sorted(map(json.dumps, grid))

    def test_trial_count_is_bounded(self):
        big = {f"p{i}": list(range(5)) for i in range(4)}  # 625 combinations
        assert 625 > MAX_TRIALS
        with pytest.raises(ConfigError, match="random"):
            expand_search_space(big, "grid")
        assert len(expand_search_space(big, "random", n_iter=10, seed=0)) == 10
        with pytest.raises(ConfigError):
            expand_search_space(big, "random", n_iter=MAX_TRIALS + 1, seed=0)

    @pytest.mark.parametrize("space, method, n_iter", [
        ({}, "grid", None),
        ({"C": []}, "grid", None),
        ({"C": "abc"}, "grid", None),
        ({"C": [1.0, 1.0]}, "grid", None),
        ({"C": [[1, 2]]}, "grid", None),
        ({"C": [float("nan")]}, "grid", None),
        ({"C": [1.0]}, "bayesian", None),
        ({"C": [1.0]}, "random", 0),
        ({"C": [1.0]}, "random", None),
    ])
    def test_invalid_inputs_raise(self, space, method, n_iter):
        with pytest.raises(ConfigError):
            expand_search_space(space, method, n_iter, seed=0)


# ---------------------------------------------------------------------- objective
class TestCompositeObjective:
    METRICS = {"auroc": 0.75, "pr_auc": 0.3, "brier": 0.12, "calibration_slope_abs_error": 0.2, "citl_abs": 0.05,
               "net_benefit_at_0.1": 0.01}

    def test_weighted_sum_higher_is_better(self):
        assert composite_objective(self.METRICS, WEIGHTS) == pytest.approx(0.75 - 0.12 - 0.25 * 0.2)
        worse = {**self.METRICS, "brier": 0.2}
        assert composite_objective(worse, WEIGHTS) < composite_objective(self.METRICS, WEIGHTS)
        weights = {"pr_auc": 2.0, "citl_abs": -1.0, "net_benefit_at_0.1": 10.0}
        assert composite_objective(self.METRICS, weights) == pytest.approx(0.6 - 0.05 + 0.1)

    def test_undefined_metric_gives_nan(self):
        assert np.isnan(composite_objective({**self.METRICS, "calibration_slope_abs_error": float("nan")}, WEIGHTS))

    @pytest.mark.parametrize("metrics, weights, match", [
        (METRICS, {"accuracy": 1.0}, "unknown"),
        (METRICS, {"net_benefit_at_0.2": 1.0}, "missing"),
        (METRICS, {"net_benefit_at_abc": 1.0}, "not a number"),
        (METRICS, {"net_benefit_at_1.5": 1.0}, r"\(0, 1\)"),
        ({**METRICS, "auroc": None}, {"auroc": 1.0}, "missing"),
        (METRICS, {"brier": 1.0}, "wrong sign"),
        (METRICS, {"auroc": -1.0}, "wrong sign"),
        (METRICS, {"auroc": 0.0}, "all zero"),
        (METRICS, {"auroc": True}, "finite number"),
        (METRICS, {}, "non-empty"),
    ])
    def test_invalid_raise(self, metrics, weights, match):
        with pytest.raises(ConfigError, match=match):
            composite_objective(metrics, weights)


# ---------------------------------------------------------------------- search
class TestRunSearch:
    def test_selects_best_trial_and_matches_schema(self, train, validation, spec):
        with pytest.warns(M.UndefinedMetricWarning):
            best, results = search(train, validation, spec, space={"slope": [0.0, 0.5, 1.0, 3.0]})
        assert best == {"slope": 1.0}
        assert list(results.columns) == RESULT_COLUMNS
        assert results["trial"].tolist() == [0, 1, 2, 3]
        assert results["selected"].dtype == bool and results["selected"].sum() == 1
        assert np.isnan(results["objective"].iloc[0])  # constant predictor: calibration slope undefined, never selected
        assert json.loads(results.loc[results["selected"], "params_json"].item()) == best
        y = validation["outcome_12m"].to_numpy()
        p = expit(-1.0 + validation["signal"].to_numpy())
        row = results.iloc[2]
        assert row["auroc"] == pytest.approx(M.auroc(y, p))
        assert row["brier"] == pytest.approx(M.brier(y, p))
        assert row["calibration_slope"] == pytest.approx(M.calibration_slope_intercept(y, np.log(p / (1 - p)))[1])
        assert row["citl"] == pytest.approx(M.citl(y, np.log(p / (1 - p)))[0])
        assert row["objective"] == pytest.approx(row["auroc"] - row["brier"] - 0.25 * abs(row["calibration_slope"] - 1))

    def test_ties_select_earliest_trial(self, train, validation, spec):
        best, results = search(train, validation, spec, space={"slope": [1.0], "unused": ["c", "a", "b"]})
        assert results["objective"].nunique() == 1
        assert results["selected"].tolist() == [True, False, False]
        assert best == {"slope": 1.0, "unused": "c"}

    def test_non_linear_pipelines_use_logit_of_risk(self, train, validation, spec):
        _, linear = search(train, validation, spec, space={"slope": [0.5, 2.0]})
        _, nonlinear = search(train, validation, spec, space={"slope": [0.5, 2.0]}, linear=False)
        pd.testing.assert_frame_equal(linear, nonlinear, check_exact=False, rtol=1e-8)

    def test_random_method_runs_expanded_trials_in_order(self, train, validation, spec):
        space = {"slope": [0.1, 0.25, 0.5, 1.0, 2.0, 3.0]}
        _, results = search(train, validation, spec, space=space, method="random", n_iter=3)
        expected = expand_search_space(space, "random", n_iter=3, seed=7)
        assert [json.loads(s) for s in results["params_json"]] == expected

    def test_net_benefit_objective(self, train, validation, spec):
        best, results = search(train, validation, spec, space={"slope": [0.1, 1.0]}, weights={"net_benefit_at_0.2": 1.0})
        y, signal = validation["outcome_12m"].to_numpy(), validation["signal"].to_numpy()
        assert results["objective"].tolist() == pytest.approx([M.net_benefit(y, expit(-1.0 + s * signal), 0.2) for s in (0.1, 1.0)])
        assert best == {"slope": 1.0}

    def test_all_objectives_undefined_raises(self, train, validation, spec):
        with pytest.warns(M.UndefinedMetricWarning), pytest.raises(FallsMLError, match="no trial"):
            search(train, validation, spec, space={"slope": [0.0]})

    def test_weights_are_validated_before_any_fit(self, train, validation, spec):
        def fit_fn(df, params):
            raise AssertionError("fit_fn must not be called with invalid weights")

        with pytest.raises(ConfigError, match="unknown"):
            search(train, validation, spec, space={"slope": [1.0]}, weights={"accuracy": 1.0}, fit_fn=fit_fn)

    def test_single_class_validation_raises(self, train, validation, spec):
        one_class = validation.assign(outcome_12m=np.zeros(len(validation), dtype="int8"))
        with pytest.raises(DatasetValidationError, match="both classes"):
            search(train, one_class, spec, space={"slope": [1.0]})

    def test_invalid_predictions_raise(self, train, validation, spec):
        class Broken(StubPipeline):
            def predict_proba(self, df):
                return np.full(len(df) - 1, 0.5)

        with pytest.raises(FallsMLError, match="probabilities"):
            search(train, validation, spec, space={"slope": [1.0]}, fit_fn=lambda df, params: Broken(params))

    def test_invalid_linear_predictor_raises(self, train, validation, spec):
        class BrokenLP(StubPipeline):
            def linear_predictor(self, df):
                return np.zeros(len(df) + 1)

        with pytest.raises(FallsMLError, match="linear predictor"):
            search(train, validation, spec, space={"slope": [1.0]}, fit_fn=lambda df, params: BrokenLP(params))
