"""Reduced eFalls predictor set (experiment kind ``efalls_retrained_reduced``): config rules, root-based feature subsets,
dataset spec selection, the ``efalls_coverage`` block, report/comparison/best-features wording and published strictness.

SYNTHETIC test data only.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from falls_ml.bundle import feature_spec_identity, rebuild_feature_spec
from falls_ml.config import check_feature_declaration, experiment_config_from_dict, load_experiment_config
from falls_ml.data.dataset import ModelingDataset
from falls_ml.data.schema import validate_modeling_dataset
from falls_ml.data.synthetic import generate_synthetic_modeling_dataset
from falls_ml.errors import ConfigError, DatasetValidationError
from falls_ml.experiment import (REDUCED_WARNING, _preflight, _resolve_dataset, efalls_coverage, required_dataset_columns,
                                 run_experiment)
from falls_ml.features.preprocessing import Preprocessor
from falls_ml.features.spec import load_feature_spec
from falls_ml.reporting.best_features import build_best_features
from falls_ml.reporting.compare import CATEGORIES, build_comparison
from falls_ml.reporting.report import render_run_report
from tests.helpers.fake_run import make_fake_run, reduced_fixture_features

ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = ROOT / "configs" / "features" / "efalls_v1.yaml"
FIXTURE_CONFIGS = ROOT / "configs" / "experiments" / "fixture"
REDUCED_CONFIG = FIXTURE_CONFIGS / "efalls_retrained_reduced.yaml"
TEMPLATE_CONFIG = ROOT / "configs" / "experiments" / "efalls_retrained_reduced.yaml"
MANDATORY = ["age_years", "sex", "polypharmacy_count_120d", "falls", "fracture", "fragility_fracture", "dementia"]
ILLUSTRATIVE_UNAVAILABLE = ["bmi_value", "smoking_status", "alcohol_category", "environment_problems", "housebound",
                            "meal_preparation_problems", "medication_management", "problems_managing_finances", "requirement_for_care",
                            "shopping_problems", "social_vulnerability", "washing_and_bathing"]


@pytest.fixture(scope="module")
def spec():
    return load_feature_spec(SPEC_PATH)


def _raw(name: str = "efalls_retrained_reduced") -> dict:
    raw = yaml.safe_load((FIXTURE_CONFIGS / f"{name}.yaml").read_text(encoding="utf-8"))
    raw["dataset"]["feature_spec"] = str(SPEC_PATH)
    if raw["model"].get("published_equation"):
        raw["model"]["published_equation"]["config"] = str(ROOT / "configs" / "models" / "efalls_published.yaml")
    return raw


def _cfg(raw: dict, dataset: Path | None = None):
    raw = copy.deepcopy(raw)
    if dataset is not None:
        raw["dataset"]["path"] = str(dataset)
    return experiment_config_from_dict(raw, name="test")


@pytest.fixture(scope="module")
def datasets(tmp_path_factory):
    """Tiny SYNTHETIC fixtures with identical rows: the full eFalls spec and the reduced fixture config's subset."""
    base = tmp_path_factory.mktemp("reduced_data")
    features = reduced_fixture_features()
    kwargs = dict(n_patients=260, feature_spec_path=SPEC_PATH)
    generate_synthetic_modeling_dataset(base / "full", **kwargs)
    generate_synthetic_modeling_dataset(base / "reduced", features=features, **kwargs)
    return {"full": base / "full", "reduced": base / "reduced", "features": features}


# ============================================================================ config validation
class TestConfig:
    def test_fixture_and_template_configs_are_valid_reduced_declarations(self, spec):
        for path in (REDUCED_CONFIG, TEMPLATE_CONFIG):
            cfg = load_experiment_config(path)
            check_feature_declaration(cfg, spec)
            features = cfg.preprocessing.features
            assert cfg.experiment.kind == "efalls_retrained_reduced" and cfg.model.name == "lasso_logistic_cv"
            assert len(features) == 66 and set(MANDATORY) <= set(features)
            assert [f for f in spec.predictor_names() if f not in features] == ILLUSTRATIVE_UNAVAILABLE
            assert "bmi_category" not in cfg.evaluation.subgroups, "bmi_value is unavailable in the example"
            text = path.read_text(encoding="utf-8")
            assert "ILLUSTRATIVE" in text and "TO_BE_MAPPED" in text and "configs/mappings/meuhedet_v0.yaml" in text
            assert "all except" not in yaml.safe_load(text)["preprocessing"], "features are listed explicitly"
        fixture, template = load_experiment_config(REDUCED_CONFIG), load_experiment_config(TEMPLATE_CONFIG)
        assert fixture.dataset.path == "data/fixtures/synthetic_reduced_v1" and fixture.dataset.require_scientific_use is False
        assert template.dataset.path is None and template.dataset.require_scientific_use is True
        assert fixture.output.runs_dir == template.output.runs_dir == "runs"

    @pytest.mark.parametrize("change, message", [
        (lambda r: r["preprocessing"].pop("features"), "requires preprocessing.features"),
        (lambda r: r["preprocessing"].update(features=[]), "requires preprocessing.features"),
        (lambda r: r["preprocessing"].update(features=r["preprocessing"]["features"] + ["falls"]), "duplicates"),
        (lambda r: r["preprocessing"].update(features=[f for f in r["preprocessing"]["features"] if f != "age_years"]), "age_years"),
        (lambda r: r["preprocessing"].update(features=[f for f in r["preprocessing"]["features"] if f != "polypharmacy_count_120d"]),
         "fractional_polynomial.variables"),
        (lambda r: r["model"].update(name="logistic_unpenalized", params={}), "must use model lasso_logistic_cv"),
        (lambda r: r["model"].update(params={"C": 1.0}), "not part of the published learning process"),
        (lambda r: r["model"].update(params={"selection": "1se"}), "lambda-min"),
        (lambda r: r["validation"].update(cv_folds=5), "10-fold"),
        (lambda r: r["preprocessing"]["fractional_polynomial"].update(mode="linear"), "select or fixed_published"),
        (lambda r: r.update(tuning={"enabled": True}), "tuning is not allowed"),
        (lambda r: r["validation"]["temporal"].update(embargo_outcome_windows=False), "embargo"),
        (lambda r: r["experiment"].update(layers=["L1_published", "L3b_meuhedet_predictor"]), "L3b"),
        (lambda r: r["experiment"].update(layers=["L1_published", "L4_alternative"]), "L4"),
        (lambda r: r["dataset"].update(feature_spec_extensions=["configs/features/meuhedet_enhanced_example.yaml"]), "extensions"),
    ])
    def test_reduced_rules(self, change, message):
        raw = _raw()
        change(raw)
        with pytest.raises(ConfigError, match=message):
            experiment_config_from_dict(raw)

    def test_full_retraining_with_a_subset_points_to_the_reduced_kind(self):
        raw = _raw("efalls_retrained_lasso")
        raw["preprocessing"]["features"] = reduced_fixture_features()
        with pytest.raises(ConfigError, match="efalls_retrained_reduced"):
            experiment_config_from_dict(raw)

    def test_all_78_candidates_are_not_a_reduced_set(self, spec, tmp_path, datasets):
        raw = _raw()
        raw["preprocessing"]["features"] = spec.predictor_names()
        cfg = _cfg(raw, datasets["full"])
        with pytest.raises(ConfigError, match="use experiment kind efalls_retrained"):
            check_feature_declaration(cfg, spec)
        runs = tmp_path / "runs"
        with pytest.raises(ConfigError, match="use experiment kind efalls_retrained"):
            run_experiment(None, None, cfg, runs_dir=runs)
        assert not runs.exists(), "the check runs before a run directory is created"

    def test_unknown_or_non_efalls_names_fail_before_any_data_are_read(self, spec, tmp_path):
        raw = _raw()
        raw["preprocessing"]["features"] = [*raw["preprocessing"]["features"], "mini_cog_score"]
        cfg = _cfg(raw, tmp_path / "does_not_exist")
        with pytest.raises(ConfigError, match="mini_cog_score"):
            run_experiment(None, None, cfg, runs_dir=tmp_path / "runs")
        assert not (tmp_path / "runs").exists()

    def test_model_override_leaves_the_efalls_label(self):
        from falls_ml.experiment import _resolve_config

        cfg = _resolve_config(load_experiment_config(REDUCED_CONFIG), "random_forest")
        assert cfg.experiment.kind == "alternative_model" and cfg.preprocessing.features == load_experiment_config(REDUCED_CONFIG).preprocessing.features


# ============================================================================ feature subsets
class TestSubset:
    def test_subset_is_idempotent_and_root_based(self, spec):
        a = reduced_fixture_features()
        b = [f for f in a if f not in {"falls", "dementia"}]
        sub = spec.subset(a)
        assert sub.is_subset and sub.subset_of == spec.content_sha256 and sub.content_sha256 != spec.content_sha256
        assert sub.subset(a) is sub and sub.subset(a).content_sha256 == sub.content_sha256
        assert sub.subset(b).content_sha256 == spec.subset(b).content_sha256
        assert sub.subset(b).subset_of == spec.content_sha256
        assert spec.subset(list(reversed(a))).content_sha256 == sub.content_sha256, "order of names does not matter"
        assert sub.predictor_names() == [n for n in spec.predictor_names() if n in set(a)], "spec order preserved"

    def test_subset_with_every_predictor_is_the_full_spec(self, spec):
        assert spec.subset(spec.predictor_names()) is spec
        assert spec.subset(reversed(spec.predictor_names())).content_sha256 == spec.content_sha256

    @pytest.mark.parametrize("names, message", [([], "non-empty"), (["age_years", "age_years"], "Duplicated"),
                                                (["age_years", "not_a_feature"], "Unknown")])
    def test_invalid_subsets_fail(self, spec, names, message):
        with pytest.raises(ConfigError, match=message):
            spec.subset(names)

    def test_names_outside_a_subset_are_unknown(self, spec):
        with pytest.raises(ConfigError, match="bmi_value"):
            spec.subset(reduced_fixture_features()).subset(["age_years", "bmi_value"])

    def test_bundle_rebuilds_subset_of_subset(self, spec):
        nested = spec.subset(reduced_fixture_features()).subset(["age_years", "sex", "polypharmacy_count_120d", "falls"])
        identity = json.loads(json.dumps(feature_spec_identity(nested)))
        assert rebuild_feature_spec(identity).content_sha256 == nested.content_sha256


# ============================================================================ dataset spec selection
class TestDatasetSelection:
    def test_fixture_rows_equal_the_full_fixture(self, datasets, spec):
        full = ModelingDataset.load(datasets["full"], spec)
        reduced = ModelingDataset.load(datasets["reduced"], spec, features=datasets["features"])
        assert reduced.spec.content_sha256 == spec.subset(datasets["features"]).content_sha256
        assert reduced.manifest.feature_spec_sha256 == reduced.spec.content_sha256
        pd.testing.assert_frame_equal(full.frame[reduced.frame.columns], reduced.frame)
        audit = reduced.manifest.audit["feature_spec_subset"]
        assert audit["absent_predictors"] == ILLUSTRATIVE_UNAVAILABLE and audit["root_feature_spec_sha256"] == spec.content_sha256

    def test_declared_features_accept_full_or_exact_subset(self, datasets, spec):
        assert ModelingDataset.load(datasets["full"], spec, features=datasets["features"]).spec is spec
        assert ModelingDataset.load(datasets["reduced"], spec, features=datasets["features"]).spec.is_subset

    def test_other_subset_fails_listing_both_expected_hashes(self, datasets, spec):
        other = [f for f in datasets["features"] if f != "falls"]
        with pytest.raises(DatasetValidationError) as exc:
            ModelingDataset.load(datasets["reduced"], spec, features=other)
        problems = "\n".join(exc.value.problems)
        assert spec.content_sha256 in problems and spec.subset(other).content_sha256 in problems
        assert "feature spec content" in problems

    def test_full_spec_without_declaration_rejects_subset_dataset(self, datasets, spec):
        with pytest.raises(DatasetValidationError, match="different feature spec") as exc:
            ModelingDataset.load(datasets["reduced"], spec)
        assert any(spec.content_sha256 in p for p in exc.value.problems)

    def test_subset_inference_confirms_the_manifest_hash(self, datasets, spec):
        inferred = ModelingDataset.load(datasets["reduced"], spec, infer_subset=True)
        assert inferred.spec.content_sha256 == spec.subset(datasets["features"]).content_sha256
        assert ModelingDataset.load(datasets["full"], spec, infer_subset=True).spec is spec

    def test_resolve_dataset_for_reduced_config(self, datasets, spec):
        cfg = _cfg(_raw(), datasets["reduced"])
        ds = _resolve_dataset(None, cfg, spec)
        assert ds.spec.is_subset
        _preflight(cfg, ds)  # sex, age_years and research_id are present; bmi_value is not needed
        assert _resolve_dataset(ModelingDataset.load(datasets["full"], spec), cfg, spec).spec is spec

    def test_loaded_dataset_with_other_spec_lists_expected_hashes(self, datasets, spec):
        other = ModelingDataset.load(datasets["reduced"], spec, infer_subset=True)
        cfg = _cfg(_raw("efalls_retrained_lasso"))
        with pytest.raises(DatasetValidationError) as exc:
            _resolve_dataset(other, cfg, spec)
        assert any(spec.content_sha256 in p for p in exc.value.problems)


# ============================================================================ preflight columns
class TestPreflightColumns:
    def test_only_needed_columns_are_required(self):
        reduced = load_experiment_config(REDUCED_CONFIG)
        assert required_dataset_columns(reduced) == {"research_id", "sex", "age_years"}
        published = load_experiment_config(FIXTURE_CONFIGS / "efalls_published_scoring.yaml")
        assert required_dataset_columns(published) == {"research_id", "sex", "bmi_value", "age_years"}
        raw = _raw()
        raw["evaluation"]["subgroups"] = ["age_band"]
        raw["analysis"]["iecv"] = {"enabled": True, "cluster_column": "practice_id"}
        raw["evaluation"]["cluster_column_for_heterogeneity"] = "site_id"
        assert required_dataset_columns(_cfg(raw)) == {"research_id", "age_years", "practice_id", "site_id"}

    def test_bmi_subgroup_on_dataset_without_bmi_fails_before_run_directory(self, datasets, tmp_path):
        raw = _raw()
        raw["evaluation"]["subgroups"] = ["sex", "bmi_category"]
        runs = tmp_path / "runs"
        with pytest.raises(ConfigError, match="bmi_value"):
            run_experiment(None, None, _cfg(raw, datasets["reduced"]), runs_dir=runs)
        assert not runs.exists()


# ============================================================================ coverage block
class TestCoverage:
    def test_reduced_fixture_coverage(self, spec):
        cov = efalls_coverage(spec, reduced_fixture_features(), kind="efalls_retrained_reduced")
        assert (cov["n_available"], cov["n_total"], cov["coverage_pct"]) == (66, 78, 84.6)
        assert cov["unavailable"] == ILLUSTRATIVE_UNAVAILABLE and len(cov["available"]) == 66
        assert (cov["n_published_retained_available"], cov["n_published_retained_total"]) == (56, 62)
        assert cov["mandatory_unavailable"] == [] and cov["is_full_efalls_feature_set"] is False
        assert cov["label"].startswith(REDUCED_WARNING)

    def test_percentage_rounding_and_mandatory(self, spec):
        available = [n for n in spec.predictor_names() if n not in {"dementia", "falls", "anxiety"}]
        cov = efalls_coverage(spec, available, kind="efalls_retrained_reduced")
        assert cov["n_available"] == 75 and cov["coverage_pct"] == round(100 * 75 / 78, 1) == 96.2
        assert cov["mandatory_unavailable"] == ["falls", "dementia"]
        assert cov["n_published_retained_available"] == 60, "anxiety is not retained in the published model"

    def test_full_and_published_coverage(self, spec):
        full = efalls_coverage(spec, spec.predictor_names(), kind="efalls_retrained")
        assert (full["n_available"], full["coverage_pct"], full["is_full_efalls_feature_set"], full["unavailable"]) == (78, 100.0, True, [])
        published = efalls_coverage(spec, [n for n in spec.predictor_names() if n != "abdominal_pain"], kind="efalls_published_scoring")
        assert published["n_available"] == 77 and published["unavailable"] == ["abdominal_pain"] and "M-11" in published["label"]


# ============================================================================ reporting
def test_reduced_report_shows_coverage_and_warnings(tmp_path):
    run = make_fake_run(tmp_path, kind="efalls_retrained_reduced", model="lasso_logistic_cv", seed=11, served_recalibrated=True)
    md = render_run_report(run)[0].read_text(encoding="utf-8")
    html = (run / "report.html").read_text(encoding="utf-8")
    for text in ("| Available eFalls predictors | 66 / 78 |", "| Coverage percentage | 84.6% |", "Available eFalls predictors: 66 / 78",
                 "Coverage percentage: 84.6%", f"Unavailable eFalls predictors: {', '.join(ILLUSTRATIVE_UNAVAILABLE)}",
                 "Local retraining on a REDUCED eFalls predictor set (NOT a full eFalls reproduction)", REDUCED_WARNING,
                 "### eFalls predictor coverage", "| Published-retained predictors available (non-zero published coefficient) | 56 / 62 |"):
        assert text in md, text
    executive = md.split("## Executive Summary")[1].split("## 1. Dataset")[0]
    assert f"> **{REDUCED_WARNING}" in executive and "| Available eFalls predictors | 66 / 78 |" in executive
    assert md.index(REDUCED_WARNING) < md.index("## Executive Summary"), "warning is shown before the summary"
    assert "Available eFalls predictors: 66 / 78" in html and "mandatory eFalls predictors are unavailable" not in md
    assert "Local retraining: eFalls predictors and learning process" not in md, "never labelled as the full retraining"


def test_mandatory_unavailable_predictors_are_warned(tmp_path):
    run = make_fake_run(tmp_path, kind="efalls_retrained_reduced", model="lasso_logistic_cv", seed=12,
                        unavailable_efalls=("dementia", "bmi_value"))
    md = render_run_report(run)[0].read_text(encoding="utf-8")
    assert "WARNING: mandatory eFalls predictors are unavailable (cohort.mandatory_for_efalls_label): dementia" in md
    assert "| Mandatory eFalls predictors unavailable | dementia |" in md and "| dementia | YES |" in md
    assert "| Available eFalls predictors | 76 / 78 |" in md and "| Coverage percentage | 97.4% |" in md


def test_full_efalls_reports_show_full_coverage(tmp_path):
    for kind, model in (("efalls_published_scoring", "efalls_published"), ("efalls_retrained", "lasso_logistic_cv")):
        md = render_run_report(make_fake_run(tmp_path, kind=kind, model=model, seed=13))[0].read_text(encoding="utf-8")
        assert "| Available eFalls predictors | 78 / 78 |" in md and "| Coverage percentage | 100.0% |" in md
        assert REDUCED_WARNING not in md
    md = render_run_report(make_fake_run(tmp_path, kind="alternative_model", model="random_forest", seed=14))[0].read_text(encoding="utf-8")
    assert "Available eFalls predictors" not in md


def test_comparison_has_reduced_category_and_coverage_column(tmp_path):
    runs = tmp_path / "runs"
    make_fake_run(runs, kind="efalls_published_scoring", model="efalls_published", seed=1)
    make_fake_run(runs, kind="efalls_retrained", model="lasso_logistic_cv", seed=2, served_recalibrated=True)
    reduced = make_fake_run(runs, kind="efalls_retrained_reduced", model="lasso_logistic_cv", seed=3, served_recalibrated=True)
    make_fake_run(runs, kind="alternative_model", model="random_forest", seed=4)
    df = build_comparison(runs, tmp_path / "reports", n_bootstrap=20)
    assert CATEGORIES["efalls_retrained_reduced"] == "reduced_local_retraining"
    row = df.loc[df["kind"] == "efalls_retrained_reduced"].iloc[0]
    assert row["category"] == "reduced_local_retraining" and bool(row["is_efalls_retrained"]) is False
    assert row["efalls_coverage"] == "66/78" and row["run_id"] == reduced.name
    assert set(df.loc[df["kind"].isin(["efalls_published_scoring", "efalls_retrained"]), "efalls_coverage"]) == {"78/78"}
    assert df.loc[df["kind"] == "alternative_model", "efalls_coverage"].isna().all()
    assert df.loc[df["kind"] == "efalls_retrained", "is_efalls_retrained"].all()
    md = (tmp_path / "reports" / "comparison.md").read_text(encoding="utf-8")
    title = "## Local retraining on a reduced eFalls predictor set (not a full eFalls reproduction)"
    order = ["## Local retraining of eFalls", title, "## Experimental algorithms"]
    assert [md.index(h) for h in order] == sorted(md.index(h) for h in order)
    section = md.split(title)[1].split("## Experimental algorithms")[0]
    assert "| eFalls coverage |" in section and "| 66/78 |" in section and "efalls_retrained_reduced" in section
    assert f"WARNING: {reduced.name} use a REDUCED eFalls predictor set" in md
    assert "is_efalls_retrained: efalls_retrained_lasso (lasso_logistic_cv)" in md
    assert pd.read_csv(tmp_path / "reports" / "comparison.csv")["efalls_coverage"].dropna().tolist().count("66/78") == 1


def test_best_features_never_uses_reduced_run_as_efalls_lasso_evidence(tmp_path):
    runs = tmp_path / "runs"
    make_fake_run(runs, kind="efalls_published_scoring", model="efalls_published", seed=1)
    reduced = make_fake_run(runs, kind="efalls_retrained_reduced", model="lasso_logistic_cv", seed=3)
    table = build_best_features(runs, tmp_path / "reports", ablation_dir=tmp_path / "no_ablation")
    assert table["retrained_lasso_coefficient"].isna().all() and table["selection_frequency"].isna().all()
    assert table["perm_lasso_logistic_cv_mean"].notna().any(), "still a local model for permutation importance"
    md = (tmp_path / "reports" / "best_features.md").read_text(encoding="utf-8")
    assert "(no local eFalls retraining run)" in md.split("## 4.")[1].split("## 5.")[0]
    assert f"WARNING: run {reduced.name} (lasso_logistic_cv) uses a REDUCED eFalls predictor set" in md

    retrained = make_fake_run(runs, kind="efalls_retrained", model="lasso_logistic_cv", seed=2)
    table = build_best_features(runs, tmp_path / "reports2", ablation_dir=tmp_path / "no_ablation")
    md = (tmp_path / "reports2" / "best_features.md").read_text(encoding="utf-8")
    assert f"lasso_logistic_cv: run {retrained.name} (efalls_retrained;" in md and reduced.name in md.split("ignored")[1]
    assert table["retrained_lasso_coefficient"].notna().any()


# ============================================================================ published and full retraining stay strict
def test_published_scoring_on_reduced_dataset_fails_loudly(datasets, spec, tmp_path):
    runs = tmp_path / "runs"
    with pytest.raises(DatasetValidationError, match="different feature spec"):
        run_experiment(None, datasets["reduced"], _cfg(_raw("efalls_published_scoring")), runs_dir=runs)
    reduced = ModelingDataset.load(datasets["reduced"], spec, infer_subset=True)
    with pytest.raises(DatasetValidationError, match="different feature spec"):
        run_experiment(None, reduced, _cfg(_raw("efalls_published_scoring")), runs_dir=runs)
    assert not runs.exists()
    # the published representation itself refuses rows missing predictor columns
    with pytest.raises(DatasetValidationError) as exc:
        validate_modeling_dataset(reduced.frame, spec)
    assert any("missing required columns" in p and "bmi_value" in p for p in exc.value.problems)
    pre = Preprocessor(spec, "efalls_published")
    with pytest.raises(DatasetValidationError, match="missing predictor columns"):
        pre.fit(reduced.frame)
    with pytest.raises(ConfigError, match="full eFalls feature set"):
        Preprocessor(reduced.spec, "efalls_published")
    raw = _raw("efalls_published_scoring")
    raw["preprocessing"]["features"] = datasets["features"]
    with pytest.raises(ConfigError, match="feature subset"):
        experiment_config_from_dict(raw)


def test_full_retraining_rejects_subset_dataset(datasets, spec, tmp_path):
    runs = tmp_path / "runs"
    with pytest.raises(DatasetValidationError, match="different feature spec"):
        run_experiment(None, datasets["reduced"], _cfg(_raw("efalls_retrained_lasso")), runs_dir=runs)
    with pytest.raises(DatasetValidationError, match="different feature spec"):
        run_experiment(None, ModelingDataset.load(datasets["reduced"], spec, infer_subset=True), _cfg(_raw("efalls_retrained_lasso")),
                       runs_dir=runs)
    assert not runs.exists()


def test_edited_base_spec_cannot_shrink_the_efalls_candidate_denominator(tmp_path):
    import yaml as _yaml

    from falls_ml.config import check_feature_declaration, load_experiment_config
    from falls_ml.errors import ConfigError as _ConfigError
    from falls_ml.features.spec import load_feature_spec as _load

    root = Path(__file__).resolve().parents[2]
    raw = _yaml.safe_load((root / "configs/features/efalls_v1.yaml").read_text(encoding="utf-8"))
    raw["features"] = [f for f in raw["features"] if f["name"] != "falls"]
    edited = tmp_path / "configs" / "features" / "efalls_v1.yaml"
    edited.parent.mkdir(parents=True)
    edited.write_text(_yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    cfg = load_experiment_config(root / "configs/experiments/fixture/efalls_retrained_lasso.yaml")
    with pytest.raises(_ConfigError, match="appears edited"):
        check_feature_declaration(cfg, _load(edited))


def test_paired_reference_requires_the_unmodified_published_equation():
    from falls_ml.reporting.compare import _unmodified_published

    base = {"experiment": {"kind": "efalls_published_scoring"},
            "published_scoring": {"unavailable_predictors": [], "unavailable_fill": "zero", "low_support_zeroed": False}}
    assert _unmodified_published(base)
    assert not _unmodified_published({**base, "published_scoring": {**base["published_scoring"], "unavailable_predictors": ["falls"]}})
    assert not _unmodified_published({**base, "published_scoring": {**base["published_scoring"], "low_support_zeroed": True}})
    assert not _unmodified_published({**base, "experiment": {"kind": "efalls_retrained"}})
