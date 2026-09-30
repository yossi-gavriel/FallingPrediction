"""Ablation summary: paired deltas vs baseline on identical rows, and mismatch errors (D-19 §5, D-21)."""

from __future__ import annotations

import inspect
import re

import numpy as np
import pandas as pd
import pytest

from falls_ml.evaluation.bootstrap import paired_bootstrap_difference
from falls_ml.evaluation.metrics import compute_metric
from falls_ml.reporting.ablation import BOOTSTRAP_COMPONENT, summarize_ablation
from falls_ml.reporting.report import SYNTHETIC_BANNER
from tests.helpers.fake_run import make_fake_run

N_BOOT = 60


def _runs(root):
    base = make_fake_run(root, kind="ablation_member", model="logistic_unpenalized", seed=10, experiment="abl__baseline")
    cog = make_fake_run(root, kind="ablation_member", model="logistic_unpenalized", seed=11, experiment="abl__cognition",
                        extra_features={"mini_cog_score": 0.2}, signal=1.4)
    mob = make_fake_run(root, kind="ablation_member", model="random_forest", seed=12, experiment="abl__mobility",
                        extra_features={"mini_cog_score": 0.2, "mobility_aid": 0.1}, signal=1.5)
    return base, cog, mob


def _pred(run):
    return pd.read_parquet(run / "predictions_test.parquet").sort_values("research_id").reset_index(drop=True)


def test_deltas_match_metric_differences(tmp_path):
    base, cog, mob = _runs(tmp_path / "runs")
    df = summarize_ablation(base, [("+ cognition", cog), ("+ mobility", mob)], tmp_path / "out", n_bootstrap=N_BOOT, seed=3)
    assert list(df["step"]) == ["Baseline", "+ cognition", "+ mobility"]
    b, c, m = _pred(base), _pred(cog), _pred(mob)
    y = b["outcome"].to_numpy()
    for metric in ("auroc", "brier", "calibration_slope", "citl"):
        base_value = compute_metric(metric, y, b["risk_uncalibrated"], b["linear_predictor"])
        cog_value = compute_metric(metric, y, c["risk_uncalibrated"], c["linear_predictor"])
        mob_value = compute_metric(metric, y, m["risk_uncalibrated"])  # non-linear step: calibration on logit(risk)
        assert df.loc[0, metric] == pytest.approx(base_value, abs=1e-12)
        assert df.loc[1, f"{metric}_delta"] == pytest.approx(cog_value - base_value, abs=1e-12)
        assert df.loc[2, f"{metric}_delta"] == pytest.approx(mob_value - base_value, abs=1e-12)
        assert np.isnan(df.loc[0, f"{metric}_delta"])
        assert df.loc[1, f"{metric}_delta_ci_low"] < df.loc[1, f"{metric}_delta_ci_high"]
    expected = paired_bootstrap_difference(y, c["risk_uncalibrated"], b["risk_uncalibrated"], metric="auroc", n=N_BOOT, seed=3,
                                           component=BOOTSTRAP_COMPONENT, lp_a=c["linear_predictor"], lp_b=b["linear_predictor"],
                                           cluster=b["research_id"].astype("string").to_numpy())
    assert df.loc[1, "auroc_delta_ci_low"] == pytest.approx(expected["ci_low"]) and df.loc[1, "auroc_delta_ci_high"] == pytest.approx(expected["ci_high"])
    assert df.loc[2, "auroc_change_vs_previous_step"] == pytest.approx(df.loc[2, "auroc"] - df.loc[1, "auroc"])
    assert list(df["added_features"]) == ["", "mini_cog_score", "mobility_aid"]
    csv = pd.read_csv(tmp_path / "out" / "ablation_results.csv")
    assert list(csv["step"]) == list(df["step"])
    md = (tmp_path / "out" / "ablation.md").read_text(encoding="utf-8")
    assert md.startswith(SYNTHETIC_BANNER + "\n")
    assert re.search(r"^- Baseline AUROC \d\.\d{3} ", md, flags=re.M)
    assert re.search(r"^- \+ cognition [+-]\d\.\d{3} \(-?\d\.\d{3}–-?\d\.\d{3}\) AUROC; Brier .*calibration slope .*CITL", md, flags=re.M)
    assert "Baseline model: abl__baseline (logistic_unpenalized; experiment kind ablation_member), run " + base.name in md
    assert "Dataset: version synthetic_v1 (data SHA-256 " in md and "not the published eFalls equation" in md
    assert list(df["model"]) == ["logistic_unpenalized", "logistic_unpenalized", "random_forest"]
    again = summarize_ablation(base, [("+ cognition", cog), ("+ mobility", mob)], tmp_path / "out2", n_bootstrap=N_BOOT, seed=3)
    pd.testing.assert_frame_equal(df, again)


def test_different_rows_raise(tmp_path):
    base = make_fake_run(tmp_path, kind="ablation_member", model="logistic_unpenalized", seed=1, experiment="a")
    other = make_fake_run(tmp_path, kind="ablation_member", model="logistic_unpenalized", seed=2, experiment="b", cohort_seed=99)
    with pytest.raises(ValueError, match="test rows differ"):
        summarize_ablation(base, [("+ x", other)], tmp_path / "out", n_bootstrap=N_BOOT)


def test_different_outcomes_raise(tmp_path):
    base = make_fake_run(tmp_path, kind="ablation_member", model="logistic_unpenalized", seed=1, experiment="a")
    other = make_fake_run(tmp_path, kind="ablation_member", model="logistic_unpenalized", seed=2, experiment="b")
    pred = pd.read_parquet(other / "predictions_test.parquet")
    outcome = pred["outcome"].to_numpy().copy()
    outcome[0] = 1 - outcome[0]
    pred.assign(outcome=outcome).to_parquet(other / "predictions_test.parquet", index=False)
    with pytest.raises(ValueError, match="outcomes differ"):
        summarize_ablation(base, [("+ x", other)], tmp_path / "out", n_bootstrap=N_BOOT)


def test_duplicated_rows_raise(tmp_path):
    base = make_fake_run(tmp_path, kind="ablation_member", model="logistic_unpenalized", seed=1, experiment="a")
    pred = pd.read_parquet(base / "predictions_test.parquet")
    pd.concat([pred, pred.iloc[:1]]).to_parquet(base / "predictions_test.parquet", index=False)
    with pytest.raises(ValueError, match="duplicated"):
        summarize_ablation(base, [], tmp_path / "out", n_bootstrap=N_BOOT)


def test_mixing_synthetic_and_real_runs_raises(tmp_path):
    base = make_fake_run(tmp_path, kind="ablation_member", model="logistic_unpenalized", seed=1, experiment="a", synthetic=True)
    other = make_fake_run(tmp_path, kind="ablation_member", model="logistic_unpenalized", seed=2, experiment="b", synthetic=False)
    with pytest.raises(ValueError, match="mix synthetic"):
        summarize_ablation(base, [("+ x", other)], tmp_path / "out", n_bootstrap=N_BOOT)


def test_real_data_ablation_has_no_banner(tmp_path):
    base = make_fake_run(tmp_path, kind="ablation_member", model="logistic_unpenalized", seed=1, experiment="a", synthetic=False)
    summarize_ablation(base, [], tmp_path / "out", n_bootstrap=N_BOOT)
    assert SYNTHETIC_BANNER not in (tmp_path / "out" / "ablation.md").read_text(encoding="utf-8")


def test_default_bootstrap_resamples():
    assert inspect.signature(summarize_ablation).parameters["n_bootstrap"].default == 1000
