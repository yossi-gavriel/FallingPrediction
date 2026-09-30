"""Stata ``logit``-style omissions (spec D-13): constant columns and perfect predictors are dropped and reported."""

import numpy as np
import pytest

from falls_ml.models.separation import SeparationError, stata_logit_omissions


def test_no_omissions_on_ordinary_data():
    rng = np.random.default_rng(1)
    X = rng.integers(0, 2, size=(400, 3)).astype(float)
    y = rng.integers(0, 2, size=400).astype(float)
    report = stata_logit_omissions(X, y, ["a", "b", "c"])
    assert not report.any and report.kept_columns == (0, 1, 2) and report.kept_rows.all()


def test_constant_and_perfect_predictors_are_reported_iteratively():
    y = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 0], dtype=float)
    const = np.ones(10)
    perfect = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)      # 1 only with y=0
    ordinary = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 1], dtype=float)
    report = stata_logit_omissions(np.column_stack([const, perfect, ordinary]), y, ["const", "perfect", "ordinary"])
    assert report.constant_columns == ("const",)
    assert report.perfect_predictors == ({"column": "perfect", "level": 1, "predicts_outcome": 0, "n_rows_dropped": 2},)
    assert report.kept_columns == (2,) and report.n_rows_dropped == 2 and not report.kept_rows[:2].any()


def test_single_class_after_omission_raises():
    y = np.array([1, 1, 0, 0], dtype=float)
    x = np.array([1, 1, 0, 0], dtype=float)
    with pytest.raises(SeparationError):
        stata_logit_omissions(x[:, None], y, ["x"])


def test_continuous_columns_are_not_checked_for_perfect_prediction():
    y = np.array([0, 0, 1, 1], dtype=float)
    x = np.array([0.1, 0.2, 0.9, 1.3])
    report = stata_logit_omissions(x[:, None], y, ["x"])
    assert not report.any
