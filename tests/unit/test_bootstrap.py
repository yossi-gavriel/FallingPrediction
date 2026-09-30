"""Unit tests for percentile (cluster) bootstrap CIs and paired differences (D-11, D-21)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from falls_ml.evaluation import metrics as M
from falls_ml.evaluation.bootstrap import (
    MIN_VALID_FRACTION,
    bootstrap_metric_cis,
    bootstrap_replicates,
    paired_bootstrap_difference,
)
from falls_ml.evaluation.metrics import SCALAR_METRICS, UndefinedMetricWarning


def _data(n: int = 600, seed: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    lp = rng.normal(-1.5, 1.0, n)
    y = (rng.random(n) < M.expit(lp)).astype(np.int64)
    return y, np.asarray(M.expit(lp)), lp


def test_estimates_equal_point_metrics_and_cis_bracket_them() -> None:
    y, p, lp = _data()
    out = bootstrap_metric_cis(y, p, lp, n=200, seed=1, component="ci:test")
    assert set(out) == set(SCALAR_METRICS)
    summary = M.performance_summary(y, p, lp=lp)
    for name, entry in out.items():
        assert set(entry) == {"estimate", "ci_low", "ci_high"}
        assert entry["estimate"] == pytest.approx(summary[name], abs=1e-12)
        assert entry["ci_low"] < entry["estimate"] < entry["ci_high"]


def test_percentiles_are_computed_from_the_replicates() -> None:
    y, p, _ = _data()
    reps = bootstrap_replicates(y, p, n=150, seed=4, component="c", metrics=("auroc", "brier"))
    assert len(reps) == 150 and list(reps.columns) == ["auroc", "brier", "single_class"]
    out = bootstrap_metric_cis(y, p, n=150, seed=4, component="c", metrics=("auroc", "brier"), alpha=0.1)
    low, high = np.quantile(reps["brier"], [0.05, 0.95])
    assert (out["brier"]["ci_low"], out["brier"]["ci_high"]) == pytest.approx((low, high), abs=1e-15)


def test_deterministic_for_seed_and_component() -> None:
    y, p, _ = _data()
    a = bootstrap_metric_cis(y, p, n=50, seed=9, component="x", metrics=("auroc",))
    b = bootstrap_metric_cis(y, p, n=50, seed=9, component="x", metrics=("auroc",))
    c = bootstrap_metric_cis(y, p, n=50, seed=9, component="y", metrics=("auroc",))
    assert a == b
    assert a["auroc"]["ci_low"] != c["auroc"]["ci_low"]


def test_cluster_bootstrap_resamples_whole_clusters() -> None:
    # Every row duplicated inside its own cluster: a cluster resample is the row resample of the de-duplicated
    # data with the same draws, each drawn row appearing twice, and these measures are invariant to that.
    y, p, lp = _data(n=300)
    metrics = ("auroc", "pr_auc", "brier", "calibration_slope", "citl", "oe_ratio", "net_benefit@0.2")
    rows = bootstrap_replicates(y, p, lp, n=40, seed=3, component="k", metrics=metrics)
    order = np.repeat(np.arange(300), 2)
    clustered = bootstrap_replicates(y[order], p[order], lp[order], n=40, seed=3, component="k",
                                     metrics=metrics, cluster=order)
    np.testing.assert_allclose(clustered.to_numpy(dtype=float), rows.to_numpy(dtype=float), rtol=1e-7, atol=1e-10)


def test_cluster_bootstrap_is_wider_for_strongly_clustered_data() -> None:
    rng = np.random.default_rng(0)
    cluster = np.repeat(np.arange(30), 20)
    shift = rng.normal(0, 1.0, 30)[cluster]
    p = np.asarray(M.expit(-1.0 + rng.normal(0, 0.3, cluster.size)))
    y = (rng.random(cluster.size) < M.expit(-1.0 + shift)).astype(int)
    naive = bootstrap_metric_cis(y, p, n=300, seed=1, component="w", metrics=("oe_ratio",))["oe_ratio"]
    clus = bootstrap_metric_cis(y, p, n=300, seed=1, component="w", metrics=("oe_ratio",), cluster=cluster)["oe_ratio"]
    assert clus["ci_high"] - clus["ci_low"] > 1.5 * (naive["ci_high"] - naive["ci_low"])


def test_single_class_resamples_are_skipped_counted_and_ci_suppressed() -> None:
    y = np.zeros(12, dtype=int)
    y[0] = 1  # P(no event in a resample) = (11/12)^12 ≈ 0.35
    p = np.linspace(0.05, 0.6, 12)
    reps = bootstrap_replicates(y, p, n=400, seed=5, component="s", metrics=("brier",))
    n_single = int(reps["single_class"].sum())
    assert 0.25 * 400 < n_single < 0.45 * 400
    assert reps.loc[reps["single_class"], "brier"].isna().all()
    assert np.isfinite(reps.loc[~reps["single_class"], "brier"]).all()
    assert n_single > (1 - MIN_VALID_FRACTION) * 400
    with pytest.warns(UndefinedMetricWarning, match="valid replicates"):
        out = bootstrap_metric_cis(y, p, n=400, seed=5, component="s", metrics=("brier",))
    assert out["brier"]["estimate"] == pytest.approx(M.brier(y, p))
    assert out["brier"]["ci_low"] is None and out["brier"]["ci_high"] is None


def test_paired_difference_uses_identical_resamples() -> None:
    y, p_a, _ = _data(n=500)
    rng = np.random.default_rng(8)
    p_b = np.clip(p_a + rng.normal(0, 0.15, p_a.size), 0.001, 0.999)
    res = paired_bootstrap_difference(y, p_a, p_b, metric="auroc", n=200, seed=6, component="d")
    assert set(res) == {"estimate", "ci_low", "ci_high", "n_valid"}
    assert res["estimate"] == pytest.approx(M.auroc(y, p_a) - M.auroc(y, p_b))
    ra = bootstrap_replicates(y, p_a, n=200, seed=6, component="d", metrics=("auroc",))["auroc"].to_numpy()
    rb = bootstrap_replicates(y, p_b, n=200, seed=6, component="d", metrics=("auroc",))["auroc"].to_numpy()
    assert (res["ci_low"], res["ci_high"]) == pytest.approx(tuple(np.quantile(ra - rb, [0.025, 0.975])), abs=1e-15)
    assert res["n_valid"] == 200 and res["ci_low"] > 0  # noise degrades discrimination


def test_paired_difference_identical_models_and_net_benefit_metric() -> None:
    y, p, lp = _data()
    same = paired_bootstrap_difference(y, p, p, metric="calibration_slope", n=50, seed=1, component="z", lp_a=lp, lp_b=lp)
    assert (same["estimate"], same["ci_low"], same["ci_high"], same["n_valid"]) == (0.0, 0.0, 0.0, 50)
    nb = paired_bootstrap_difference(y, p, np.full(p.size, 0.5), metric="net_benefit@0.2", n=50, seed=1, component="z")
    assert nb["estimate"] == pytest.approx(M.net_benefit(y, p, 0.2) - M.net_benefit_treat_all(y.mean(), 0.2))


def test_invalid_arguments_raise() -> None:
    y, p, _ = _data(n=50)
    with pytest.raises(ValueError):
        bootstrap_metric_cis(y, p, n=10, seed=1, component="a", metrics=("accuracy",))
    with pytest.raises(ValueError):
        bootstrap_metric_cis(y, p, n=0, seed=1, component="a")
    with pytest.raises(ValueError):
        paired_bootstrap_difference(y, p, p[:-1], metric="auroc", n=10, seed=1, component="a")
    with pytest.raises(ValueError):
        bootstrap_metric_cis(y, p, n=10, seed=1, component="a", cluster=np.arange(10))
    # missing cluster ids must not be pooled silently into one pseudo-cluster
    ids = np.array([f"c{i % 7}" for i in range(50)], dtype=object)
    ids[3] = None
    with pytest.raises(ValueError, match="missing"):
        bootstrap_metric_cis(y, p, n=10, seed=1, component="a", cluster=ids)
    with pytest.raises(ValueError, match="missing"):
        paired_bootstrap_difference(y, p, p, metric="brier", n=10, seed=1, component="a",
                                    cluster=np.r_[np.arange(49.0), np.nan])
    assert math.isfinite(bootstrap_metric_cis(y, p, n=10, seed=1, component="a", metrics=("brier",))["brier"]["ci_low"])
