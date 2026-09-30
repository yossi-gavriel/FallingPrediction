"""Report plots write readable, non-empty, deterministic PNG files (architecture §3.4 plots/)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from falls_ml.evaluation import metrics as M
from falls_ml.reporting import plots as P
from falls_ml.seeding import rng_for

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@pytest.fixture(scope="module")
def yp() -> tuple[np.ndarray, np.ndarray]:
    rng = rng_for(7, "test_plots")
    x = rng.normal(size=400)
    y = (rng.random(400) < expit(-1.5 + x)).astype(np.int64)
    return y, expit(-1.5 + 0.9 * x)


def _assert_png(returned, expected) -> None:
    assert returned == expected
    data = expected.read_bytes()
    assert data[:8] == PNG_MAGIC and len(data) > 2000


def test_roc_and_precision_recall(tmp_path, yp):
    y, p = yp
    curves = {"model A": (y, p), "model B": (y, np.clip(p * 0.8, 0, 1))}
    _assert_png(P.plot_roc(curves, tmp_path / "roc.png", "ROC"), tmp_path / "roc.png")
    _assert_png(P.plot_precision_recall(curves, tmp_path / "pr.png", "PR", prevalence=float(y.mean())), tmp_path / "pr.png")


def test_calibration_with_and_without_histogram(tmp_path, yp):
    y, p = yp
    grouped, smoothed = M.grouped_calibration(y, p), M.smoothed_calibration(y, p)
    _assert_png(P.plot_calibration(grouped, smoothed, tmp_path / "cal.png", "Calibration", predictions=p), tmp_path / "cal.png")
    undefined = smoothed.assign(observed_rate=np.nan)
    _assert_png(P.plot_calibration(grouped, undefined, tmp_path / "cal2.png", "Calibration"), tmp_path / "cal2.png")


def test_calibration_before_after(tmp_path, yp):
    y, p = yp
    recalibrated = expit(0.2 + 1.1 * np.log(p / (1 - p)))
    path = tmp_path / "before_after.png"
    _assert_png(P.plot_calibration_before_after(y, p, recalibrated, path, "Before / after"), path)
    assert P.plot_calibration_before_after(y, p, recalibrated, tmp_path / "again.png", "Before / after").read_bytes() == path.read_bytes()
    constant = np.full(y.size, 0.2)  # LOWESS undefined: still plotted, named in the legend
    _assert_png(P.plot_calibration_before_after(y, p, constant, tmp_path / "constant.png", "Before / after"), tmp_path / "constant.png")
    with pytest.raises(ValueError, match="equal length"):
        P.plot_calibration_before_after(y, p[:-1], recalibrated, tmp_path / "x.png", "Before / after")


def test_decision_curve_single_and_multiple_variants(tmp_path, yp):
    y, p = yp
    grid = np.round(np.arange(0.01, 0.51, 0.01), 10)
    dc = M.decision_curve(y, p, grid)
    _assert_png(P.plot_decision_curve(dc, tmp_path / "dc.png", "Decision curve"), tmp_path / "dc.png")
    both = pd.concat([dc.assign(variant="uncalibrated"), M.decision_curve(y, p * 0.9, grid).assign(variant="recalibrated")])
    _assert_png(P.plot_decision_curve(both, tmp_path / "dc2.png", "Decision curve", highlight=None), tmp_path / "dc2.png")


def test_risk_distribution(tmp_path, yp):
    y, p = yp
    _assert_png(P.plot_risk_distribution(y, p, tmp_path / "risk.png", "Risk"), tmp_path / "risk.png")


def test_feature_importance_and_coefficients(tmp_path):
    fi = pd.DataFrame({"feature": [f"f{i}" for i in range(20)], "permutation_importance_mean": np.linspace(0.05, -0.01, 20),
                       "permutation_importance_std": 0.003})
    _assert_png(P.plot_feature_importance(fi, tmp_path / "fi.png", "Importance"), tmp_path / "fi.png")
    coef = pd.DataFrame({"design_column": [f"c{i}" for i in range(30)], "coefficient": np.r_[np.linspace(-0.5, 0.8, 25), np.zeros(5)],
                         "standardized_coefficient": np.nan})
    _assert_png(P.plot_coefficients(coef, tmp_path / "coef.png", "Coefficients"), tmp_path / "coef.png")
    none_selected = coef.assign(coefficient=0.0)
    _assert_png(P.plot_coefficients(none_selected, tmp_path / "coef0.png", "Coefficients"), tmp_path / "coef0.png")


def test_selection_stability_hyperparameters_and_lasso_path(tmp_path):
    stability = pd.DataFrame({"feature": [f"c{i}" for i in range(40)], "selection_frequency": np.linspace(1, 0, 40),
                              "robust": ["True"] * 10 + ["False"] * 30})
    _assert_png(P.plot_selection_stability(stability, tmp_path / "stab.png", "Stability"), tmp_path / "stab.png")
    trials = pd.DataFrame({"trial": range(5), "params_json": [json.dumps({"max_depth": d}) for d in range(5)], "objective": [0.1, 0.3, 0.2, 0.25, 0.15],
                           "auroc": [0.70, 0.74, 0.72, 0.73, 0.71], "brier": [0.11, 0.10, 0.105, 0.102, 0.108], "selected": [False, True, False, False, False]})
    _assert_png(P.plot_hyperparameter_search(trials, tmp_path / "hp.png", "Search"), tmp_path / "hp.png")
    lam = np.geomspace(0.1, 1e-4, 30)
    cv = pd.DataFrame({"lambda": lam, "cv_mean_deviance": 0.8 + (np.log10(lam) + 2.5) ** 2 * 0.01, "cv_se": 0.01,
                       "n_nonzero": np.arange(30), "selected": np.arange(30) == 14})
    _assert_png(P.plot_lasso_cv_path(cv, tmp_path / "lasso.png", "LASSO CV"), tmp_path / "lasso.png")


def test_png_output_is_deterministic(tmp_path, yp):
    y, p = yp
    a = P.plot_roc({"m": (y, p)}, tmp_path / "a.png", "ROC").read_bytes()
    b = P.plot_roc({"m": (y, p)}, tmp_path / "b.png", "ROC").read_bytes()
    assert a == b


def test_invalid_inputs_fail_loudly(tmp_path, yp):
    y, p = yp
    with pytest.raises(ValueError, match="at least one"):
        P.plot_roc({}, tmp_path / "x.png", "ROC")
    with pytest.raises(ValueError, match="both outcome classes"):
        P.plot_roc({"m": (np.zeros(10), np.full(10, 0.1))}, tmp_path / "x.png", "ROC")
    with pytest.raises(ValueError, match="missing columns"):
        P.plot_feature_importance(pd.DataFrame({"feature": ["a"]}), tmp_path / "x.png", "FI")
    with pytest.raises(ValueError, match="no finite"):
        P.plot_lasso_cv_path(pd.DataFrame({"lambda": [0.1], "cv_mean_deviance": [np.nan], "cv_se": [0.1]}), tmp_path / "x.png", "L")


def test_decision_curve_with_several_splits_per_curve_raises(tmp_path, yp):
    """decision_curve.csv holds every split: plotting it unfiltered would join curves from different partitions."""
    y, p = yp
    grid = np.round(np.arange(0.01, 0.51, 0.01), 10)
    dc = M.decision_curve(y, p, grid)
    both_splits = pd.concat([dc.assign(split="validation", variant="uncalibrated"), dc.assign(split="test", variant="uncalibrated")])
    with pytest.raises(ValueError, match="duplicated thresholds"):
        P.plot_decision_curve(both_splits, tmp_path / "dc.png", "Decision curve")


def test_coefficients_ignore_rows_relative_to_reference(tmp_path):
    """D-10 re-expressed rows duplicate design columns; the plot must be identical to plotting the fitted rows only."""
    fitted = pd.DataFrame({"design_column": ["a", "b", "c"], "coefficient": [0.4, -0.3, 0.2], "standardized_coefficient": [0.8, -0.6, 0.4],
                           "relative_to_reference": False})
    extra = fitted.assign(coefficient=[0.9, 0.1, -0.5], relative_to_reference=True)
    both = P.plot_coefficients(pd.concat([fitted, extra]).astype({"relative_to_reference": str}), tmp_path / "both.png", "C").read_bytes()
    assert both == P.plot_coefficients(fitted, tmp_path / "fitted.png", "C").read_bytes()
    intercept = pd.DataFrame([{"design_column": "_intercept", "coefficient": -2.0, "standardized_coefficient": np.nan,
                               "relative_to_reference": False}])
    omitted = fitted.iloc[:1].assign(design_column="dropped", coefficient=0.0, omitted=True)
    with_extras = pd.concat([fitted.assign(omitted=False), intercept.assign(omitted=False), omitted])
    assert P.plot_coefficients(with_extras, tmp_path / "extras.png", "C").read_bytes() == both
