"""Per-run report: section order, watermark, self-contained HTML, result-type wording (spec §1.1/§1.2, D-01, M-11)."""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from falls_ml.artifacts import write_yaml
from falls_ml.config import load_experiment_config
from falls_ml.reporting.report import (
    EXECUTIVE_SUMMARY, NOT_AVAILABLE, SECTION_TITLES, SYNTHETIC_BANNER, build_run_document, config_sha256, render_run_report,
)
from tests.helpers.fake_run import PROJECT_ROOT, make_fake_run


@pytest.fixture(scope="module")
def runs(tmp_path_factory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("runs")
    out = {
        "published": make_fake_run(root, kind="efalls_published_scoring", model="efalls_published", seed=1, with_plots=True),
        "retrained": make_fake_run(root, kind="efalls_retrained", model="lasso_logistic_cv", seed=2, with_plots=True),
        "alternative": make_fake_run(root, kind="alternative_model", model="random_forest", seed=3, with_plots=True),
    }
    for run in out.values():
        render_run_report(run)
    return out


def _md(run: Path) -> str:
    return (run / "report.md").read_text(encoding="utf-8")


def _html(run: Path) -> str:
    return (run / "report.html").read_text(encoding="utf-8")


@pytest.mark.parametrize("name", ["published", "retrained", "alternative"])
def test_sections_in_required_order(runs, name):
    md, html = _md(runs[name]), _html(runs[name])
    titles = [EXECUTIVE_SUMMARY, *SECTION_TITLES]
    md_headings = re.findall(r"^## (.+)$", md, flags=re.M)
    html_headings = re.findall(r"<h2>(.+?)</h2>", html)
    assert md_headings == titles
    assert [h.replace("&amp;", "&") for h in html_headings] == titles


@pytest.mark.parametrize("name", ["published", "retrained", "alternative"])
def test_synthetic_banner_starts_every_report(runs, name):
    assert _md(runs[name]).startswith(SYNTHETIC_BANNER + "\n")
    assert not re.search(r"\bnan\b|<NA>|\| None \|", _md(runs[name]))
    html = _html(runs[name])
    assert html.index(SYNTHETIC_BANNER) < html.index("<h1>")


def test_real_data_report_has_no_banner_and_suppresses_small_cells(tmp_path):
    run = make_fake_run(tmp_path, kind="alternative_model", model="logistic_unpenalized", seed=9, synthetic=False, n_test=60)
    md = _md(Path(render_run_report(run)[0]).parent)
    assert SYNTHETIC_BANNER not in md
    errors = md.split("## 13. Error analysis")[1].split("## 14.")[0]
    assert "<10" in errors and "M-13" in errors


@pytest.mark.parametrize("name", ["published", "retrained", "alternative"])
def test_html_is_self_contained_with_embedded_images(runs, name):
    run, html = runs[name], _html(runs[name])
    assert not re.search(r"(src|href)\s*=\s*[\"']?(https?:|//|file:)", html, flags=re.I)
    assert "<link" not in html and "<script" not in html and "@import" not in html
    sources = re.findall(r'<img alt="[^"]*" src="([^"]+)"', html)
    md_images = re.findall(r"!\[[^\]]*\]\((plots/[^)]+\.png)\)", _md(run))
    assert md_images and len(sources) == len(md_images)
    for src, rel in zip(sources, md_images):
        assert src.startswith("data:image/png;base64,")
        assert base64.b64decode(src.split(",", 1)[1]) == (run / rel).read_bytes()


def test_published_report_states_transportability_and_variants(runs):
    md = _md(runs["published"])
    metrics = json.loads((runs["published"] / "metrics.json").read_text(encoding="utf-8"))
    assert "TRANSPORTABILITY of the ORIGINAL published eFalls equation" in md
    assert "LOCAL RETRAINING" not in md and "EXPERIMENTAL ALGORITHM" not in md
    perf = md.split("## 9. Test performance")[1].split("## 10.")[0]
    full, shift = perf.split("### CITL-shift variants (O/E and CITL only)")
    for variant in ("lp_c_box_s3_1", "lp_a_table_s3_2"):
        e = metrics["performance"]["test"][variant]["auroc"]
        assert f"| {variant} |" in full and f"{e['estimate']:.3f} ({e['ci_low']:.3f}–{e['ci_high']:.3f})" in full
    assert "| Variant | O/E | CITL |" in shift
    assert re.search(r"^\| lp_b_label_swap \| [^|]+ \| [^|]+ \|$", shift, flags=re.M)
    assert re.search(r"^\| lp_d2_numeric_swap \| [^|]+ \| [^|]+ \|$", shift, flags=re.M)
    assert "lp_b_label_swap" not in full
    assert "Effective experiment label | efalls_published_scoring" in md
    assert "Share of development LP variance available | 1.000" in md
    assert "DESCRIPTIVE ONLY" in md.split("## 10. Calibration")[1].split("## 11.")[0]


def test_partial_scoring_label_displayed_prominently(tmp_path):
    run = make_fake_run(tmp_path, kind="efalls_published_scoring", model="efalls_published", seed=5,
                        effective_experiment_label="efalls_partial_scoring")
    md_path, html_path = render_run_report(run)
    md, html = md_path.read_text(encoding="utf-8"), html_path.read_text(encoding="utf-8")
    warning = "> **EFFECTIVE EXPERIMENT LABEL: efalls_partial_scoring."
    assert md.index(warning) < md.index(f"## {EXECUTIVE_SUMMARY}")
    summary = md.split(f"## {EXECUTIVE_SUMMARY}")[1].split("## 1.")[0]
    assert warning in summary and "| Effective experiment label | efalls_partial_scoring |" in summary
    assert "NOT a validation of eFalls" in summary
    assert "| Type of result | Scientific reproduction: ORIGINAL published eFalls equation (transportability); labelled " \
           "efalls_partial_scoring: NOT a validation of eFalls (M-11) |" in summary
    assert "| Improved over the eFalls baseline? | Not applicable: this run scores the published equation with incomplete coverage" in summary
    assert re.search(r'<p class="warning">EFFECTIVE EXPERIMENT LABEL: efalls_partial_scoring', html)
    assert "| Unavailable predictor |" in md and "| falls |" in md


def test_retrained_and_alternative_wording_is_not_confused(runs):
    retrained, alternative = _md(runs["retrained"]), _md(runs["alternative"])
    assert "coefficients were fitted locally" in retrained
    assert "TRANSPORTABILITY of the ORIGINAL" not in retrained
    assert "EXPERIMENTAL ALGORITHM" in alternative and "experimental algorithm" in alternative.split("## 15.")[1]
    assert "coefficients were fitted locally" not in alternative and "TRANSPORTABILITY of the ORIGINAL" not in alternative
    assert "lambda* = " in retrained.split(f"## {EXECUTIVE_SUMMARY}")[1].split("## 1.")[0]


def test_executive_summary_contents_and_formatting(runs):
    md = _md(runs["alternative"])
    metrics = json.loads((runs["alternative"] / "metrics.json").read_text(encoding="utf-8"))
    summary = md.split(f"## {EXECUTIVE_SUMMARY}")[1].split("## 1.")[0]
    for label in ("Experiment name", "Algorithm", "Data version", "Number of patients", "Outcomes", "Main AUROC", "PR-AUC",
                  "Brier score", "Calibration status", "Best hyperparameters", "Top 10 features", "Improved over the eFalls baseline?",
                  "Eligible for further validation?"):
        assert f"| {label}" in summary
    brier = metrics["performance"]["test"]["uncalibrated"]["brier"]
    assert f"{brier['estimate']:.3f} ({brier['ci_low']:.3f}–{brier['ci_high']:.3f})" in summary
    assert "max_depth=4" in summary and ", ".join(metrics["features"]["top_features"][:10]) in summary


def test_limitations_always_include_required_items(runs, tmp_path):
    run = make_fake_run(tmp_path, kind="alternative_model", model="logistic_unpenalized", seed=4)
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    metrics.pop("limitations")
    (run / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    limitations = _md(Path(render_run_report(run)[0]).parent).split("## 16. Limitations")[1].split("## 17.")[0]
    for text in ("no temporal validation", "D-19", "D-01", "Clinical code mappings", "Unavailable predictors", "Split: temporal"):
        assert text in limitations
    assert NOT_AVAILABLE in limitations


def test_missing_optional_inputs_are_reported_not_available(tmp_path):
    run = make_fake_run(tmp_path, kind="alternative_model", model="random_forest", seed=6)
    for name in ("feature_importance.csv", "subgroup_metrics.csv", "hyperparameter_results.csv", "config.yaml"):
        (run / name).unlink()
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    for key in ("missingness", "production_readiness"):
        metrics.pop(key)
    (run / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    md = _md(Path(render_run_report(run)[0]).parent)
    section = lambda n: md.split(SECTION_TITLES[n - 1])[1].split("\n## ")[0]  # noqa: E731
    assert NOT_AVAILABLE in section(4) and NOT_AVAILABLE in section(11) and NOT_AVAILABLE in section(17)
    assert f"ROC curve: {NOT_AVAILABLE}" in section(9)
    assert f"Config SHA-256 | {NOT_AVAILABLE}" in md


def test_missing_metrics_json_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        render_run_report(tmp_path)


def test_text_is_escaped_in_html(tmp_path):
    run = make_fake_run(tmp_path, kind="alternative_model", model="logistic_unpenalized", seed=8)
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    metrics["experiment"]["name"] = "<script>alert('x')</script>"
    (run / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    html = render_run_report(run)[1].read_text(encoding="utf-8")
    assert "<script>" not in html and "&lt;script&gt;" in html


def test_config_hash_matches_experiment_config(tmp_path):
    cfg = load_experiment_config(PROJECT_ROOT / "configs" / "experiments" / "fixture" / "efalls_retrained_lasso.yaml")
    write_yaml(tmp_path / "config.yaml", cfg.to_dict())
    assert config_sha256(tmp_path) == cfg.sha256()


def test_document_is_deterministic(runs):
    assert build_run_document(runs["published"]) == build_run_document(runs["published"])


def test_published_report_shows_sex_stratified_metrics_for_c_and_a(runs):
    """D-01 item 5: sex-stratified C, slope, CITL and O/E for the primary and co-reported variants."""
    perf = _md(runs["published"]).split("## 9. Test performance")[1].split("## 10.")[0]
    section = perf.split("### Sex-stratified performance")[1].split("###")[0]
    sub = pd.read_csv(runs["published"] / "subgroup_metrics.csv")
    for variant in ("lp_c_box_s3_1", "lp_a_table_s3_2"):
        for sex in ("female", "male"):
            row = sub.loc[(sub["variant"] == variant) & (sub["subgroup_level"] == sex)].iloc[0]
            assert f"| {variant} | {sex} | {row['n']} | {row['n_events']} | {row['auroc']:.3f} | {row['calibration_slope']:.3f} | " \
                   f"{row['citl']:.3f} | {row['oe_ratio']:.3f} |" in section
    assert "lp_b_label_swap" not in section
    lp_a = sub.loc[sub["variant"] == "lp_a_table_s3_2"]
    assert not np.allclose(lp_a["citl"], sub.loc[sub["variant"] == "lp_c_box_s3_1", "citl"])  # the table is not a copy of C


def test_calibration_parameters_are_rounded(runs):
    model = _md(runs["published"]).split("## 6. Model")[1].split("## 7.")[0]
    assert "| Recalibration parameters | recalibration of lp_c_box_s3_1: alpha 0.100, beta 0.900; recalibration of lp_a_table_s3_2: " \
           "alpha 0.100, beta 0.900;" in model


def test_unavailable_predictors_fall_back_to_model_params(tmp_path):
    """metrics.json may lack the per-predictor M-11 rows; the report must not claim that nothing is unavailable."""
    run = make_fake_run(tmp_path, kind="efalls_published_scoring", model="efalls_published", seed=5,
                        effective_experiment_label="efalls_partial_scoring")
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    metrics["published_scoring"]["unavailable_predictors"] = []
    (run / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    md = render_run_report(run)[0].read_text(encoding="utf-8")
    assert "No published predictor was declared unavailable" not in md
    assert "Published predictors declared unavailable: falls" in md.split("## 3. Feature definitions")[1].split("## 4.")[0]
    assert "Unavailable predictors: falls" in md.split("## 16. Limitations")[1].split("## 17.")[0]


def test_split_section_reports_disabled_embargo(tmp_path):
    run = make_fake_run(tmp_path, kind="alternative_model", model="random_forest", seed=6)
    cfg = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["validation"]["temporal"]["embargo_outcome_windows"] is True
    on = _md(Path(render_run_report(run)[0]).parent).split("## 5. Split methodology")[1].split("## 6.")[0]
    assert "Outcome-window embargo on" in on and "DISABLED" not in on
    cfg["validation"]["temporal"]["embargo_outcome_windows"] = False
    (run / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    off = _md(Path(render_run_report(run)[0]).parent).split("## 5. Split methodology")[1].split("## 6.")[0]
    assert "> **Temporal validation" in off and "embargo (D-19) was DISABLED" in off and "Outcome-window embargo on" not in off


def test_comparison_to_efalls_shows_reference_primary_and_co_reported(tmp_path):
    published = make_fake_run(tmp_path, kind="efalls_published_scoring", model="efalls_published", seed=1)
    run = make_fake_run(tmp_path, kind="alternative_model", model="random_forest", seed=2)
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    metrics["comparison_to_efalls"] = {"reference_run_id": published.name, "reference_variant": "lp_c_box_s3_1", "delta_auroc_test": 0.01,
                                       "note": "identical rows"}
    (run / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    md = render_run_report(run)[0].read_text(encoding="utf-8")
    section = md.split("## 14. Comparison to eFalls")[1].split("## 15.")[0]
    summary = md.split(f"## {EXECUTIVE_SUMMARY}")[1].split("## 1.")[0]
    for text in (section, summary):  # 3(f): a point estimate; CIs exist only in the master comparison
        assert "point estimate only; paired patient-level bootstrap CIs are produced by the master comparison" in text
        assert "reports/comparison_paired_differences.csv, not in this report" in text
    assert "AUROC +0.010 vs published eFalls (lp_c_box_s3_1, run " in summary
    ref = json.loads((published / "metrics.json").read_text(encoding="utf-8"))["performance"]["test"]
    for variant in ("lp_c_box_s3_1", "lp_a_table_s3_2"):
        e = ref[variant]["auroc"]
        assert re.search(rf"\| Published eFalls \({variant}; [^|]+\| {e['estimate']:.3f} \({e['ci_low']:.3f}–{e['ci_high']:.3f}\)", section)


def test_missing_co_reported_variant_is_flagged(tmp_path):
    run = make_fake_run(tmp_path, kind="efalls_published_scoring", model="efalls_published", seed=7)
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    for variant in ("lp_a_table_s3_2", "lp_a_table_s3_2+recalibrated"):
        metrics["performance"]["test"].pop(variant)
    (run / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    perf = render_run_report(run)[0].read_text(encoding="utf-8").split("## 9. Test performance")[1].split("## 10.")[0]
    assert "> **Sex parameterisations not scored on the test split: lp_a_table_s3_2." in perf


def _edit(run: Path, edit) -> None:
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    edit(metrics)
    (run / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")


def _section(md: str, number: int) -> str:
    return md.split(f"## {SECTION_TITLES[number - 1]}")[1].split("\n## ")[0]


def test_served_and_transportability_variants_are_explained(runs, tmp_path):
    summary = _md(runs["published"]).split(f"## {EXECUTIVE_SUMMARY}")[1].split("## 1.")[0]
    assert "Served variant: the prediction the saved model bundle returns by default" in summary
    assert "Unless a recalibration is served, the served prediction IS the published equation" in summary
    assert "| Served prediction (what the model bundle returns) | lp_c_box_s3_1: the published eFalls equation itself (no recalibration is served) |" in summary
    assert "Transportability of the unmodified published equation" not in summary
    assert "Transportability variant" not in _md(runs["retrained"])

    run = make_fake_run(tmp_path, kind="efalls_published_scoring", model="efalls_published", seed=3, served_recalibrated=True)
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    md = render_run_report(run)[0].read_text(encoding="utf-8")
    summary = md.split(f"## {EXECUTIVE_SUMMARY}")[1].split("## 1.")[0]
    assert "served prediction variant lp_c_box_s3_1+recalibrated" in summary
    assert "a LOCAL RECALIBRATION of the published equation" in summary
    served, transport = (metrics["performance"]["test"][v] for v in ("lp_c_box_s3_1+recalibrated", "lp_c_box_s3_1"))
    assert f"| Main AUROC (0.5 = chance, 1 = perfect) | {served['auroc']['estimate']:.3f} (" in summary
    assert f"| Brier score (lower is better) | {served['brier']['estimate']:.3f} (" in summary
    assert (f"| Transportability of the unmodified published equation (lp_c_box_s3_1) | AUROC {transport['auroc']['estimate']:.3f} (" in summary
            and f"calibration slope {transport['calibration_slope']['estimate']:.3f}" in summary)
    assert "recalibrated on validation split; served by the model bundle" in _section(md, 9)


def test_missingness_shows_handling_and_split_rates(runs):
    section = _section(_md(runs["alternative"]), 4)
    assert "| Predictor | Missing (all rows) | Missing (train) | Missing (validation) | Missing (test) | Missing-data rule | " \
           "Handling in the fitted preprocessing |" in section
    assert "| bmi_value | 30.0% | 32.0% | 29.0% | 25.0% | missing_category | separate 'missing' category level |" in section


def test_model_section_shows_effective_params_fit_notes_and_lasso_tables(runs, tmp_path):
    lasso = _section(_md(runs["retrained"]), 6)
    assert '| Effective parameters (configured plus adapter defaults) | {"lambda_min_ratio": 0.0001, "selection": "min", "tol": 1e-07} |' in lasso
    for row in ("| Intercept | -2.000000 |", "| Selection rule | min |", "| lambda_max | 0.1 |", "| Non-zero coefficients (n_selected) | 10 |"):
        assert row in lasso
    assert "| Stata-style omissions in the final fit (constant columns, perfect predictors and the rows they predict) | none |" in lasso
    refit = lasso.split("### Unpenalised refit of the LASSO-selected columns: DEMONSTRATION ONLY (spec §6.2)")[1].split("###")[0]
    assert "demonstration refit" in refit and re.search(r"^\| _cons \| -2\.000 \| 0\.100 \| 1\.00 \| 0\.3000 \| 0\.135 \(0\.500–2\.000\) \| estimated \|$",
                                                         refit, flags=re.M)
    fp = lasso.split("### Fractional-polynomial selection (D-13)")[1]
    assert "| polypharmacy_count_120d | fp1 | [0.0] | 1.000 | 10.000 | 812.40 / 805.10 / 804.90 | 0.0200 | 0.9000 | 2 | yes | none |" in fp
    assert "| _intercept |" not in lasso
    assert "Unpenalised refit" not in _section(_md(runs["alternative"]), 6)

    run = make_fake_run(tmp_path, kind="alternative_model", model="logistic_unpenalized", seed=4, omitted=("dementia",))
    model = _section(render_run_report(run)[0].read_text(encoding="utf-8"), 6)
    assert '"perfect_predictors": [{"column": "dementia", "n_rows_dropped": 12}]' in model
    assert "> **Omitted design columns (coefficient forced to 0, Stata-style): dementia**" in model
    assert "| Model intercept (coefficients.csv) | -2.000000 |" in model


def test_calibration_section_shows_both_variants_and_before_after_plot(runs):
    md = _md(runs["retrained"])
    section = _section(md, 10)
    metrics = json.loads((runs["retrained"] / "metrics.json").read_text(encoding="utf-8"))
    assert "| Measure (test split) | uncalibrated (served) | recalibrated (not served) |" in section
    slopes = [metrics["performance"]["test"][v]["calibration_slope"]["estimate"] for v in ("uncalibrated", "recalibrated")]
    assert re.search(rf"^\| Calibration slope \| {slopes[0]:.3f} \([^|]+\| {slopes[1]:.3f} \(", section, flags=re.M)
    assert f"| Calibration status | {metrics['calibration_status']} | not assessed separately (not served) |" in section
    assert "| Recalibration parameters | recalibration of uncalibrated: alpha 0.100, beta 0.900 |" in section
    assert "![Calibration before and after recalibration (grouped points and LOWESS)](plots/calibration_before_after.png)" in section
    assert (runs["retrained"] / "plots" / "calibration_before_after.png").is_file()


def test_registry_count_and_not_assessed_status(tmp_path):
    run = make_fake_run(tmp_path, kind="alternative_model", model="logistic_unpenalized", seed=4)

    def edit(m):
        m["test_evaluation_registry"].update(prior_evaluations_same_test_rows=2, prior_config_sha256=["aa", "bb"])
        m["calibration_status"] = "not_assessed"
    _edit(run, edit)
    md = render_run_report(run)[0].read_text(encoding="utf-8")
    split = _section(md, 5)
    assert "| Prior evaluations of these test rows (test-evaluation registry, D-19 §4) | 2 (purpose of this run: train; prior config " \
           "hashes: aa, bb;" in split
    assert "> **The locked test rows were evaluated 2 time(s) before this run." in split
    summary = md.split(f"## {EXECUTIVE_SUMMARY}")[1].split("## 1.")[0]
    assert "| Calibration status | not assessed (calibration slope or calibration-in-the-large could not be estimated) (slope" in summary
    assert "not_assessed" not in md


def test_heterogeneity_tables_in_error_analysis(tmp_path):
    run = make_fake_run(tmp_path, kind="alternative_model", model="logistic_unpenalized", seed=4, synthetic=False)
    pd.DataFrame({"cluster": ["a", "b"], "n": [120, 7], "n_events": [30, 3], "auroc": [0.71, 0.66], "auroc_se": [0.03, 0.08]}).to_csv(
        run / "heterogeneity_clusters.csv", index=False)
    pd.DataFrame({"metric": ["auroc"], "pooled": [0.70], "tau2": [0.001], "prediction_interval_low": [0.62]}).to_csv(
        run / "heterogeneity_pooled.csv", index=False)
    section = _section(render_run_report(run)[0].read_text(encoding="utf-8"), 13)
    assert "### Heterogeneity: test metrics per cluster (D-16)" in section and "| b | <10 | <10 | 0.660 | 0.080 |" in section
    assert "### Heterogeneity: random-effects pooled summary (D-16)" in section and "| auroc | 0.700 | 0.001 | 0.620 |" in section


def test_run_warnings_are_prominent(tmp_path):
    run = make_fake_run(tmp_path, kind="alternative_model", model="logistic_unpenalized", seed=4)
    _edit(run, lambda m: m.update(warnings=["toileting_problems omitted as a perfect predictor (12 rows dropped)"]))
    md, html = (p.read_text(encoding="utf-8") for p in render_run_report(run))
    warning = "> **RUN WARNINGS (1): the pipeline recorded the following warnings; read them before using any result.**"
    assert md.startswith(SYNTHETIC_BANNER) and md.index(warning) < md.index(f"## {EXECUTIVE_SUMMARY}")
    assert md.index("- toileting_problems omitted as a perfect predictor") < md.index(f"## {EXECUTIVE_SUMMARY}")
    assert '<p class="warning">RUN WARNINGS (1)' in html
