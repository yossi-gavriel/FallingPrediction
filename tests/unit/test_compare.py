"""Master comparison: categories, highlights, comparability, paired differences and the synthetic/real guard (spec §1.2, D-19 §5)."""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

import pandas as pd
import pytest

from falls_ml.evaluation.bootstrap import paired_bootstrap_difference
from falls_ml.evaluation.metrics import compute_metric
from falls_ml.reporting.compare import CATEGORIES, PAIRED_COLUMNS, PAIRED_COMPONENT, build_comparison
from falls_ml.reporting.report import SYNTHETIC_BANNER
from tests.helpers.fake_run import make_fake_run

MARGINS = {"auroc_margin": 0.2, "brier_margin": 0.05}
N_BOOT = 40


def _build(runs_dir: Path, out: Path, **kwargs) -> pd.DataFrame:
    return build_comparison(runs_dir, out, n_bootstrap=N_BOOT, **kwargs)


def _edit_metrics(run: Path, edit) -> None:
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    edit(metrics)
    (run / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")


@pytest.fixture(scope="module")
def comparison(tmp_path_factory) -> tuple[pd.DataFrame, Path, dict[str, Path]]:
    root = tmp_path_factory.mktemp("runs")
    runs = {
        "published": make_fake_run(root, kind="efalls_published_scoring", model="efalls_published", seed=1),
        "retrained": make_fake_run(root, kind="efalls_retrained", model="lasso_logistic_cv", seed=2, served_recalibrated=True),
        "logistic": make_fake_run(root, kind="alternative_model", model="logistic_unpenalized", seed=3, omitted=("dementia",)),
        # most discriminating non-ablation comparable model, but badly calibrated (slope far below 0.8)
        "forest": make_fake_run(root, kind="alternative_model", model="random_forest", seed=4, signal=2.5, slope_multiplier=2.5),
        "enhanced": make_fake_run(root, kind="alternative_model", model="elastic_net_logistic", seed=7, experiment="elastic_net_enhanced",
                                  extra_features={"mini_cog_score": 0.2}),
        # ablation member: discriminates best of all but must never be highlighted or recommended
        "ablation": make_fake_run(root, kind="ablation_member", model="logistic_unpenalized", seed=5, experiment="abl__cognition",
                                  extra_features={"mini_cog_score": 0.2}, signal=8.0),
        # different test rows: must never be highlighted even though it discriminates best
        "other_rows": make_fake_run(root, kind="alternative_model", model="hist_gradient_boosting", seed=6, signal=5.0,
                                    test_rows_sha256="different"),
    }
    (root / "not_a_run").mkdir()
    unfinished = make_fake_run(root, kind="alternative_model", model="random_forest", seed=8, experiment="crashed_rf", signal=9.0)
    (unfinished / "RUN_COMPLETE.json").unlink()
    runs["unfinished"] = unfinished
    out = tmp_path_factory.mktemp("reports")
    return _build(root, out, **MARGINS), out, runs


def _row(df: pd.DataFrame, model: str, variant: str | None = None) -> pd.Series:
    rows = df.loc[(df["model"] == model) & (df["kind"] != "ablation_member") & ((df["variant"] == variant) if variant else True)]
    assert len(rows) == 1
    return rows.iloc[0]


def test_required_columns_and_categories(comparison):
    df, out, _ = comparison
    required = ["model", "experiment", "kind", "category", "run_id", "variant", "AUROC", "PR_AUC", "Brier", "calibration_slope",
                "calibration_intercept", "citl", "oe_ratio", "sensitivity_at_threshold", "specificity_at_threshold", "threshold",
                "number_of_features", "best_params", "train_time", "inference_time", "dataset_version", "split_strategy",
                "test_rows_sha256", "synthetic_fixture", "effective_experiment_label", "feature_set_name", "pure_efalls"]
    assert list(df.columns[: len(required)]) == required
    enhanced = df["kind"].eq("alternative_model") & df["pure_efalls"].eq(False)
    assert (df.loc[~enhanced, "category"] == df.loc[~enhanced, "kind"].map(CATEGORIES)).all()
    assert df.loc[enhanced, "experiment"].tolist() == ["elastic_net_enhanced"]
    assert set(df["category"]) == {"scientific_reproduction", "local_retraining", "experimental_algorithm", "meuhedet_enhanced", "ablation"}
    published = df.loc[df["kind"] == "efalls_published_scoring"]
    assert list(published["variant"]) == ["lp_c_box_s3_1", "lp_a_table_s3_2"]
    assert set(published["effective_experiment_label"]) == {"efalls_published_scoring"}
    assert (out / "comparison.csv").is_file() and pd.read_csv(out / "comparison.csv").shape == df.shape


def test_only_completed_runs_are_loaded(comparison):
    df, out, runs = comparison
    assert runs["unfinished"].name not in set(df["run_id"])
    skipped = (out / "comparison.md").read_text(encoding="utf-8").split("## Skipped directories")[1]
    assert f"{runs['unfinished'].name} (incomplete: no RUN_COMPLETE.json)" in skipped
    assert "not_a_run (incomplete: no RUN_COMPLETE.json, no metrics.json)" in skipped


def test_run_complete_must_match_metrics(tmp_path):
    run = make_fake_run(tmp_path, kind="alternative_model", model="logistic_unpenalized", seed=1)
    (run / "RUN_COMPLETE.json").write_text(json.dumps({"run_id": "someone_else", "completed_utc": "x"}), encoding="utf-8")
    with pytest.raises(ValueError, match="RUN_COMPLETE.json run_id"):
        _build(tmp_path, tmp_path / "out")


def test_rows_use_served_variant_and_explicit_published_variants(comparison):
    df, _, runs = comparison
    retrained = _row(df, "lasso_logistic_cv")
    metrics = json.loads((runs["retrained"] / "metrics.json").read_text(encoding="utf-8"))
    assert retrained["variant"] == metrics["primary_variant"] == metrics["served_variant"] == "recalibrated"
    assert retrained["AUROC"] == metrics["performance"]["test"]["recalibrated"]["auroc"]["estimate"]
    assert retrained["Brier"] != metrics["performance"]["test"]["uncalibrated"]["brier"]["estimate"]
    published = json.loads((runs["published"] / "metrics.json").read_text(encoding="utf-8"))["performance"]["test"]
    for variant in ("lp_c_box_s3_1", "lp_a_table_s3_2"):
        assert _row(df, "efalls_published", variant)["Brier"] == published[variant]["brier"]["estimate"]


def test_published_rows_ignore_served_recalibration_and_missing_co_reported_warns(tmp_path):
    run = make_fake_run(tmp_path, kind="efalls_published_scoring", model="efalls_published", seed=1, served_recalibrated=True)
    _edit_metrics(run, lambda m: [m["performance"]["test"].pop(v) for v in ("lp_a_table_s3_2", "lp_a_table_s3_2+recalibrated")])
    df = _build(tmp_path, tmp_path / "out")
    assert df["variant"].tolist() == ["lp_c_box_s3_1"]
    md = (tmp_path / "out" / "comparison.md").read_text(encoding="utf-8")
    assert "> **WARNING: " + run.name + ": the co-reported variant lp_a_table_s3_2 was not scored on the test split" in md
    _edit_metrics(run, lambda m: m["performance"]["test"].pop("lp_c_box_s3_1"))
    with pytest.raises(ValueError, match="published primary variant"):
        _build(tmp_path, tmp_path / "out")


def test_threshold_metrics_taken_from_run(comparison):
    df, _, _ = comparison
    row = _row(df, "logistic_unpenalized")
    assert row["threshold"] == pytest.approx(0.10)
    assert 0 <= row["sensitivity_at_threshold"] <= 1 and 0 <= row["specificity_at_threshold"] <= 1


def test_number_of_features_counts_non_zero_raw_predictors(comparison, tmp_path):
    df, _, _ = comparison
    assert _row(df, "lasso_logistic_cv")["number_of_features"] == 9  # anxiety has coefficient 0; bmi's two columns count once
    assert _row(df, "logistic_unpenalized")["number_of_features"] == 9  # dementia omitted (Stata-style); intercept not counted
    assert _row(df, "random_forest")["number_of_features"] == 10  # tree model: every predictor of the feature set
    run = make_fake_run(tmp_path, kind="efalls_retrained", model="lasso_logistic_cv", seed=2)
    coef = pd.read_csv(run / "coefficients.csv")
    extra = pd.DataFrame([{**coef.iloc[0].to_dict(), "design_column": "anxiety=relative", "feature": "anxiety", "coefficient": 0.4,
                           "relative_to_reference": True}])
    pd.concat([coef, extra]).to_csv(run / "coefficients.csv", index=False)
    assert _build(tmp_path, tmp_path / "out")["number_of_features"].item() == 9
    (run / "coefficients.csv").write_text(",".join(coef.columns) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="coefficients.csv"):
        _build(tmp_path, tmp_path / "out")


def test_train_time_is_total_with_fallback_for_old_runs(comparison, tmp_path):
    df, _, runs = comparison
    timing = json.loads((runs["forest"] / "metrics.json").read_text(encoding="utf-8"))["timing"]
    assert _row(df, "random_forest")["train_time"] == timing["total_train_seconds"] == timing["tuning_seconds"] + timing["final_fit_seconds"]
    run = make_fake_run(tmp_path, kind="alternative_model", model="logistic_unpenalized", seed=1)
    _edit_metrics(run, lambda m: m.update(timing={"fit_seconds": 3.5, "predict_seconds_per_1000_rows": 0.02}))
    assert _build(tmp_path, tmp_path / "out")["train_time"].item() == 3.5


def test_comparable_flag(comparison):
    df, _, _ = comparison
    assert not _row(df, "hist_gradient_boosting")["comparable"]
    assert df.loc[df["model"] != "hist_gradient_boosting", "comparable"].all()


def test_highlights_exclude_ablation(comparison):
    df, _, _ = comparison
    assert df["is_efalls_published"].tolist() == (df["kind"] == "efalls_published_scoring").tolist()
    assert df["is_efalls_retrained"].tolist() == (df["kind"] == "efalls_retrained").tolist()
    for flag in ("best_discrimination", "best_calibrated", "simplest_within_margin"):
        assert df[flag].sum() == 1
    ablation = df.loc[df["category"] == "ablation"].iloc[0]
    assert ablation["comparable"] and ablation["AUROC"] == df.loc[df["comparable"], "AUROC"].max()
    candidates = df.loc[df["comparable"] & (df["category"] != "ablation") & (df["variant_role"] == "primary")]
    assert df.loc[df["best_discrimination"], "run_id"].item() == candidates.loc[candidates["AUROC"].idxmax(), "run_id"]
    forest = _row(df, "random_forest")
    assert forest["best_discrimination"] and forest["calibration_slope"] < 0.8
    error = (candidates["calibration_slope"] - 1).abs() + candidates["citl"].abs()
    assert df.loc[df["best_calibrated"]].index.item() == error.idxmin()
    published = _row(df, "efalls_published", "lp_c_box_s3_1")
    assert published["AUROC"] >= candidates["AUROC"].max() - MARGINS["auroc_margin"] and 0.8 <= published["calibration_slope"] <= 1.2
    assert published["net_benefit_at_threshold"] > max(published["net_benefit_treat_all_at_threshold"], 0)
    assert published["simplest_within_margin"]
    assert not df.loc[~df["comparable"] | (df["variant"] == "lp_a_table_s3_2") | (df["category"] == "ablation"),
                      ["best_discrimination", "best_calibrated", "simplest_within_margin"]].any().any()


def test_markdown_and_html_sections_and_recommendation(comparison):
    _, out, runs = comparison
    md = (out / "comparison.md").read_text(encoding="utf-8")
    html = (out / "comparison.html").read_text(encoding="utf-8")
    assert md.startswith(SYNTHETIC_BANNER + "\n")
    order = ["## Scientific reproduction", "## Local retraining", "## Experimental algorithms", "## Meuhedet-enhanced experimental algorithms",
             "## Ablation members", "## Paired differences vs published eFalls (primary)", "## Highlights", "## Recommendation",
             "## Runs not comparable", "## Skipped directories"]
    positions = [md.index(h) for h in order]
    assert positions == sorted(positions)
    header = next(line for line in md.splitlines() if line.startswith("| Experiment | Model |"))
    for column in ("Calibration intercept", "Best params", "Train time (s)", "Inference (s per 1000 rows)"):
        assert f"| {column} |" in header
        assert f"<th>{column}</th>" in html
    assert '"max_depth": 4' in md.split("## Experimental algorithms")[1].split("## Meuhedet")[0]
    recommendation = md.split("## Recommendation")[1].split("## Runs not comparable")[0]
    assert "never chosen on AUROC alone" in recommendation
    assert "net benefit at the candidate threshold greater than both treat-all and treat-none" in recommendation
    assert "descriptive evidence only; they are not ranking criteria" in recommendation
    # fake runs fail their own eligibility check, so no model may be "recommended"
    assert "No model is recommended" in recommendation
    assert "would favour: efalls_published_scoring (efalls_published, lp_c_box_s3_1)" in recommendation
    assert "Recommended for further validation" not in recommendation
    assert "abl__cognition" not in recommendation
    assert re.search(r"highest-AUROC model, random_forest .* was not chosen because calibration slope", recommendation)
    for word in ("stability", "interpretability", "net benefit", "simplest"):
        assert word in recommendation.lower()
    assert runs["other_rows"].name in md.split("## Runs not comparable")[1]
    assert "<script" not in html and "http" not in html
    assert not re.search(r"\bnan\b", md, flags=re.I)


def test_paired_differences_vs_published_primary(comparison):
    df, out, runs = comparison
    paired = pd.read_csv(out / "comparison_paired_differences.csv")
    assert list(paired.columns) == PAIRED_COLUMNS
    assert set(paired["metric"]) == {"auroc", "brier", "calibration_slope", "citl", "net_benefit@0.1"}
    assert (paired["reference_variant"] == "lp_c_box_s3_1").all() and (paired["n_bootstrap"] == N_BOOT).all()
    compared = set(zip(paired["run_id"], paired["variant"]))
    published = runs["published"].name
    assert compared == {(published, "lp_c_box_s3_1+recalibrated"), (runs["retrained"].name, "recalibrated"),
                        (runs["logistic"].name, "uncalibrated"), (runs["forest"].name, "uncalibrated"),
                        (runs["enhanced"].name, "uncalibrated"), (runs["ablation"].name, "uncalibrated")}
    assert len(paired) == 5 * len(compared)  # non-comparable runs and the co-reported variant are not compared

    ref = pd.read_parquet(runs["published"] / "predictions_test.parquet").sort_values("research_id").reset_index(drop=True)
    other = pd.read_parquet(runs["retrained"] / "predictions_test.parquet").sort_values("research_id").reset_index(drop=True)
    y = ref["outcome"].to_numpy()
    row = paired.loc[(paired["run_id"] == runs["retrained"].name) & (paired["metric"] == "brier")].iloc[0]
    served, reference = other["risk_recalibrated"].to_numpy(), ref["risk_lp_c_box_s3_1"].to_numpy()
    assert row["estimate"] == pytest.approx(compute_metric("brier", y, served) - compute_metric("brier", y, reference), abs=1e-12)
    expected = paired_bootstrap_difference(y, served, reference, metric="brier", n=N_BOOT, seed=1, component=PAIRED_COMPONENT,
                                           cluster=ref["research_id"].astype("string").to_numpy())
    assert row["ci_low"] == pytest.approx(expected["ci_low"]) and row["ci_high"] == pytest.approx(expected["ci_high"])
    section = (out / "comparison.md").read_text(encoding="utf-8").split("## Paired differences vs published eFalls (primary)")[1].split("## Highlights")[0]
    assert f"run {published}" in section and "comparison_paired_differences.csv" in section
    assert re.search(r"^\| efalls_retrained_lasso \(lasso_logistic_cv\) \| recalibrated \| [+-]\d\.\d{3} \([+-]\d\.\d{3} to [+-]\d\.\d{3}\) \|",
                     section, flags=re.M)


def test_paired_differences_require_identical_rows_and_data(tmp_path):
    make_fake_run(tmp_path, kind="efalls_published_scoring", model="efalls_published", seed=1)
    run = make_fake_run(tmp_path, kind="alternative_model", model="logistic_unpenalized", seed=2)
    other_data = make_fake_run(tmp_path, kind="alternative_model", model="random_forest", seed=3, data_sha256="e" * 64)
    _build(tmp_path, tmp_path / "out")
    paired = pd.read_csv(tmp_path / "out" / "comparison_paired_differences.csv")
    assert other_data.name not in set(paired["run_id"]) and run.name in set(paired["run_id"])
    pred = pd.read_parquet(run / "predictions_test.parquet")
    pred.iloc[1:].to_parquet(run / "predictions_test.parquet", index=False)  # same hash claimed, different rows on disk
    with pytest.raises(ValueError, match="test rows differ"):
        _build(tmp_path, tmp_path / "out")


def test_no_paired_differences_without_published_reference(tmp_path):
    make_fake_run(tmp_path, kind="efalls_published_scoring", model="efalls_published", seed=1, effective_experiment_label="efalls_partial_scoring")
    make_fake_run(tmp_path, kind="alternative_model", model="logistic_unpenalized", seed=2)
    _build(tmp_path, tmp_path / "out")
    assert pd.read_csv(tmp_path / "out" / "comparison_paired_differences.csv").empty
    assert "contains no paired confidence intervals" in (tmp_path / "out" / "comparison.md").read_text(encoding="utf-8")


def test_net_benefit_is_a_recommendation_criterion(tmp_path):
    published = make_fake_run(tmp_path, kind="efalls_published_scoring", model="efalls_published", seed=1, eligible=True)
    make_fake_run(tmp_path, kind="efalls_retrained", model="lasso_logistic_cv", seed=2, eligible=True)

    def no_benefit(m):
        for t in m["performance"]["test"]["lp_c_box_s3_1"]["thresholds"]:
            t["net_benefit"] = -0.01
    _edit_metrics(published, no_benefit)
    df = _build(tmp_path, tmp_path / "out", **MARGINS)
    assert not _row(df, "efalls_published", "lp_c_box_s3_1")["simplest_within_margin"]
    assert _row(df, "lasso_logistic_cv")["simplest_within_margin"]
    recommendation = (tmp_path / "out" / "comparison.md").read_text(encoding="utf-8").split("## Recommendation")[1]
    assert "net benefit at threshold 0.10 (-0.0100) does not exceed both treat-all" in recommendation
    assert "(criterion met)" in recommendation


def test_no_recommendation_when_criteria_fail(tmp_path):
    make_fake_run(tmp_path, kind="efalls_published_scoring", model="efalls_published", seed=1, slope_multiplier=3.0)
    make_fake_run(tmp_path, kind="alternative_model", model="random_forest", seed=2, signal=2.5, slope_multiplier=3.0)
    df = _build(tmp_path, tmp_path / "out")
    assert not df["simplest_within_margin"].any()
    assert "no model is recommended" in (tmp_path / "out" / "comparison.md").read_text(encoding="utf-8")


def test_mixing_synthetic_and_real_runs_raises(tmp_path):
    make_fake_run(tmp_path, kind="efalls_published_scoring", model="efalls_published", seed=1, synthetic=True)
    make_fake_run(tmp_path, kind="alternative_model", model="logistic_unpenalized", seed=2, synthetic=False)
    with pytest.raises(ValueError, match="synthetic"):
        _build(tmp_path, tmp_path / "out")


def test_majority_reference_without_published_run_and_data_file(tmp_path):
    make_fake_run(tmp_path, kind="alternative_model", model="logistic_unpenalized", seed=1, test_rows_sha256="a")
    make_fake_run(tmp_path, kind="alternative_model", model="random_forest", seed=2, test_rows_sha256="a")
    make_fake_run(tmp_path, kind="alternative_model", model="elastic_net_logistic", seed=3, test_rows_sha256="a", dataset_version="v2")
    make_fake_run(tmp_path, kind="alternative_model", model="hist_gradient_boosting", seed=4, test_rows_sha256="b")
    df = _build(tmp_path, tmp_path / "out", threshold=0.2)
    assert dict(zip(df["model"], df["comparable"])) == {"logistic_unpenalized": True, "random_forest": True,
                                                        "elastic_net_logistic": False, "hist_gradient_boosting": False}
    assert (df["threshold"] == 0.2).all()


def test_empty_runs_dir_raises(tmp_path):
    with pytest.raises(ValueError, match="no completed runs"):
        _build(tmp_path, tmp_path / "out")


def test_partial_scoring_warning_is_prominent(tmp_path):
    make_fake_run(tmp_path, kind="efalls_published_scoring", model="efalls_published", seed=1,
                  effective_experiment_label="efalls_partial_scoring")
    df = _build(tmp_path, tmp_path / "out")
    assert set(df["effective_experiment_label"]) == {"efalls_partial_scoring"}
    md = (tmp_path / "out" / "comparison.md").read_text(encoding="utf-8")
    assert md.index("labelled efalls_partial_scoring; not a validation of eFalls") < md.index("## Scientific reproduction")


def test_runs_without_test_row_hash_are_never_comparable(tmp_path):
    """No reference can be established without test-row hashes: every row is non-comparable and nothing is recommended."""
    make_fake_run(tmp_path, kind="alternative_model", model="logistic_unpenalized", seed=1, test_rows_sha256=None)
    make_fake_run(tmp_path, kind="alternative_model", model="random_forest", seed=2, test_rows_sha256=None)
    df = _build(tmp_path, tmp_path / "out")
    assert not df["comparable"].any()
    assert not df[["best_discrimination", "best_calibrated", "simplest_within_margin"]].any().any()
    assert "no recommendation is made" in (tmp_path / "out" / "comparison.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("threshold", [0.0, 1.0, 1.5])
def test_threshold_outside_unit_interval_raises(tmp_path, threshold):
    make_fake_run(tmp_path, kind="alternative_model", model="logistic_unpenalized", seed=1)
    with pytest.raises(ValueError, match="threshold must lie in"):
        _build(tmp_path, tmp_path / "out", threshold=threshold)


def test_missing_auroc_ci_is_labelled(tmp_path):
    run = make_fake_run(tmp_path, kind="alternative_model", model="logistic_unpenalized", seed=1)
    _edit_metrics(run, lambda m: m["performance"]["test"]["uncalibrated"]["auroc"].update(ci_low=None, ci_high=None))
    _build(tmp_path, tmp_path / "out")
    estimate = json.loads((run / "metrics.json").read_text(encoding="utf-8"))["performance"]["test"]["uncalibrated"]["auroc"]["estimate"]
    assert f"| {estimate:.3f} (CI not available) |" in (tmp_path / "out" / "comparison.md").read_text(encoding="utf-8")


def test_recommendation_explains_equally_simple_better_calibrated_choice(tmp_path):
    make_fake_run(tmp_path, kind="alternative_model", model="random_forest", seed=1, experiment="rf_a")
    make_fake_run(tmp_path, kind="alternative_model", model="random_forest", seed=1, experiment="rf_b", signal=1.3, slope_multiplier=0.8)
    df = _build(tmp_path, tmp_path / "out", auroc_margin=0.2, brier_margin=0.05, slope_range=(0.5, 1.5))
    best, chosen = df.loc[df["best_discrimination"]].iloc[0], df.loc[df["simplest_within_margin"]].iloc[0]
    assert best["experiment"] == "rf_b" and chosen["experiment"] == "rf_a"
    text = (tmp_path / "out" / "comparison.md").read_text(encoding="utf-8")
    assert "equally simple but less well calibrated" in text and "less simple to operate" not in text


def test_recommendation_only_for_eligible_runs(tmp_path):
    make_fake_run(tmp_path, kind="efalls_published_scoring", model="efalls_published", seed=1, eligible=True)
    make_fake_run(tmp_path, kind="efalls_retrained", model="lasso_logistic_cv", seed=2, eligible=True)
    out = tmp_path / "reports"
    _build(tmp_path, out)
    recommendation = (out / "comparison.md").read_text(encoding="utf-8").split("## Recommendation")[1]
    assert "Recommended for further validation:" in recommendation and "No model is recommended" not in recommendation


def test_identical_candidates_are_reported_as_a_tie(tmp_path):
    make_fake_run(tmp_path, kind="efalls_retrained", model="lasso_logistic_cv", seed=2, experiment="b_lasso", eligible=True)
    make_fake_run(tmp_path, kind="efalls_retrained", model="lasso_logistic_cv", seed=2, experiment="a_lasso", eligible=True)
    make_fake_run(tmp_path, kind="efalls_published_scoring", model="efalls_published", seed=1, signal=0.2, eligible=True)
    out = tmp_path / "reports"
    _build(tmp_path, out)
    recommendation = (out / "comparison.md").read_text(encoding="utf-8").split("## Recommendation")[1]
    assert "less well calibrated" not in recommendation


def test_default_bootstrap_resamples():
    assert inspect.signature(build_comparison).parameters["n_bootstrap"].default == 1000
