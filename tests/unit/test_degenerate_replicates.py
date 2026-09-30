"""Resampling analyses record degenerate replicate fits (D-11) instead of aborting; the primary fit still fails loudly."""

import numpy as np
import pandas as pd
import pytest

from falls_ml.errors import DegenerateFitError
from falls_ml.evaluation.stability import ResamplingError, bootstrap_stability


class _Model:
    is_linear = False
    requires_fit = True

    def get_feature_importance(self):
        return pd.DataFrame({"feature": ["x"], "coefficient": [np.nan], "odds_ratio": [np.nan], "abs_coefficient": [np.nan],
                             "selected": pd.array([pd.NA], dtype="boolean"), "direction": [""], "native_importance": [np.nan]})

    def fit_diagnostics(self):
        return {}


class _Preprocessor:
    def design_columns(self):
        return ["x"]

    def raw_feature_of(self, column):
        return column


class _Pipeline:
    def __init__(self):
        self.model = _Model()
        self.preprocessor = _Preprocessor()

    def predict_proba(self, df, calibrated=False):
        return np.full(len(df), 0.3)


def _frame(n=60):
    rng = np.random.default_rng(0)
    return pd.DataFrame({"research_id": [f"p{i}" for i in range(n)], "outcome_12m": rng.integers(0, 2, n).astype("int8"),
                         "x": rng.normal(size=n)})


def _spec():
    class Outcome:
        name = "outcome_12m"
    class Spec:
        outcome = Outcome()
        identifier_columns = ("research_id",)
        def predictor_names(self):
            return ["x"]
    return Spec()


def test_occasional_degenerate_replicate_is_recorded():
    calls = {"n": 0}

    def fit_fn(df, seed):
        calls["n"] += 1
        if calls["n"] == 3:
            raise DegenerateFitError("separation in replicate")
        return _Pipeline()

    result = bootstrap_stability(_frame(), fit_fn, n_bootstrap=20, seed=1, cluster_column="research_id", thresholds=(0.2,),
                                 reference_pipeline=_Pipeline(), spec=_spec())
    assert result.n_failed == 1 and "degenerate fit" in result.failures[0]["reason"]


def test_too_many_degenerate_replicates_raise():
    def fit_fn(df, seed):
        raise DegenerateFitError("always separated")

    with pytest.raises(ResamplingError):
        bootstrap_stability(_frame(), fit_fn, n_bootstrap=10, seed=1, cluster_column="research_id", thresholds=(0.2,),
                            reference_pipeline=_Pipeline(), spec=_spec())
