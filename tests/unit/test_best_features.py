"""Best-features table answers each question with evidence from the correct source (published, retrained, local models)."""

from __future__ import annotations

import pandas as pd
import pytest
import yaml

from falls_ml.errors import ConfigError
from falls_ml.reporting.ablation import summarize_ablation
from falls_ml.reporting.best_features import build_best_features
from falls_ml.reporting.report import SYNTHETIC_BANNER
from tests.helpers.fake_run import PROJECT_ROOT, make_fake_run

QUESTIONS = ("## 1. Which features are strongest?", "## 2. Which features are most stable?",
             "## 3. Which features were selected by the published eFalls LASSO?", "## 4. Which features were important in the local LASSO?",
             "## 5. Which features were important in nonlinear models?", "## 6. Which features had little or no value?",
             "## 7. Which Meuhedet-specific features improve beyond pure eFalls?")


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    runs, abl, out = (tmp_path_factory.mktemp(n) for n in ("runs", "abl", "out"))
    make_fake_run(runs, kind="efalls_published_scoring", model="efalls_published", seed=1)
    retrained = make_fake_run(runs, kind="efalls_retrained", model="lasso_logistic_cv", seed=2)
    duplicate = make_fake_run(runs, kind="efalls_retrained", model="lasso_logistic_cv", seed=3, experiment="efalls_retrained_lasso_fixed_fp")
    make_fake_run(runs, kind="alternative_model", model="random_forest", seed=4)
    make_fake_run(runs, kind="alternative_model", model="hist_gradient_boosting", seed=5)
    member = make_fake_run(runs, kind="ablation_member", model="logistic_unpenalized", seed=6, experiment="abl__x")
    base = make_fake_run(abl, kind="ablation_member", model="logistic_unpenalized", seed=7, experiment="abl__baseline")
    step = make_fake_run(abl, kind="ablation_member", model="logistic_unpenalized", seed=8, experiment="abl__cognition",
                         extra_features={"mini_cog_score": 0.25}, signal=1.5)
    summarize_ablation(base, [("+ cognition", step)], out / "ablation", n_bootstrap=40)  # the default ablation_dir
    incomplete = make_fake_run(runs, kind="alternative_model", model="elastic_net_logistic", seed=9)
    (incomplete / "RUN_COMPLETE.json").unlink()
    df = build_best_features(runs, out)
    return df.set_index("feature"), (out / "best_features.md").read_text(encoding="utf-8"), {"retrained": retrained, "duplicate": duplicate,
                                                                            "member": member}


def test_published_columns(result):
    df, _, _ = result
    assert len(df) == 79  # 78 published candidates + one Meuhedet-specific ablation feature
    assert df.loc["falls", "published_efalls_coefficient"] == pytest.approx(0.3009161)
    assert bool(df.loc["falls", "selected_in_published_lasso"])
    assert df.loc["anxiety", "published_efalls_coefficient"] == 0.0 and not bool(df.loc["anxiety", "selected_in_published_lasso"])
    assert pd.isna(df.loc["bmi_value", "published_efalls_coefficient"]) and "bmi_category=underweight=+0.4896735" in df.loc["bmi_value", "published_efalls_terms"]
    assert "D-01" in df.loc["sex", "published_efalls_terms"]
    assert not bool(df.loc["mini_cog_score", "is_published_efalls_feature"]) and not bool(df.loc["mini_cog_score", "selected_in_published_lasso"])


def test_retrained_and_stability_columns(result):
    df, _, _ = result
    assert bool(df.loc["falls", "retrained_selected"]) and not bool(df.loc["anxiety", "retrained_selected"])
    assert df.loc["anxiety", "retrained_lasso_coefficient"] == 0.0
    assert pd.isna(df.loc["bmi_value", "retrained_lasso_coefficient"])  # two design columns
    assert df.loc["falls", "selection_frequency"] == pytest.approx(1.0) and bool(df.loc["falls", "robust"])
    # any-form selection: bmi_value's two design columns are each selected in 25% of refits, the raw feature in 43.75%
    assert df.loc["bmi_value", "selection_frequency"] == pytest.approx(1 - 0.75**2)
    assert not bool(df.loc["anxiety", "robust"])
    assert pd.isna(df.loc["seizures", "selection_frequency"])


def test_importance_columns_and_flags(result):
    df, _, _ = result
    for model in ("efalls_published", "lasso_logistic_cv", "random_forest", "hist_gradient_boosting"):
        assert {f"perm_{model}_mean", f"perm_{model}_std", f"importance_rank_{model}"} <= set(df.columns)
    assert "perm_logistic_unpenalized_mean" not in df.columns  # ablation members are excluded
    assert "perm_elastic_net_logistic_mean" not in df.columns  # incomplete runs are excluded
    assert df.loc["falls", "importance_rank_random_forest"] == 1
    assert bool(df.loc["falls", "important_in_nonlinear"]) and not bool(df.loc["anxiety", "important_in_nonlinear"])
    assert bool(df.loc["anxiety", "little_or_no_value"]) and not bool(df.loc["falls", "little_or_no_value"])
    assert pd.isna(df.loc["seizures", "little_or_no_value"])  # no local evidence at all
    assert pd.isna(df.loc["seizures", "important_in_nonlinear"])  # absent from the nonlinear models: unknown, not "unimportant"
    assert "+ cognition" in df.loc["mini_cog_score", "meuhedet_incremental_value"]
    assert pd.isna(df.loc["falls", "meuhedet_incremental_value"])


def test_markdown_answers_questions_with_evidence(result):
    _, md, runs = result
    assert md.startswith(SYNTHETIC_BANNER + "\n")
    positions = [md.index(q) for q in QUESTIONS]
    assert positions == sorted(positions)
    strongest = md.split("strongest?")[1].split("## 2.")[0]
    assert strongest.index("| falls |") < strongest.index("| anxiety |")
    assert "across 3 locally fitted model(s) (hist_gradient_boosting, lasso_logistic_cv, random_forest;" in strongest
    assert "Importance efalls_published" not in strongest
    assert "Baseline: logistic_unpenalized, run " in md.split("## 7.")[1]
    assert "WARNING: no ablation_results.csv" not in md
    assert "published coefficients: " + str(PROJECT_ROOT / "configs" / "models" / "efalls_published.yaml") in md
    assert "and no support in the local LASSO: anxiety." in md
    assert "Not selected (coefficient 0): anxiety" in md
    assert "| + cognition | mini_cog_score |" in md
    assert runs["retrained"].name in md and runs["duplicate"].name in md and runs["member"].name in md


def test_without_ablation_results(tmp_path):
    make_fake_run(tmp_path / "runs", kind="efalls_published_scoring", model="efalls_published", seed=1)
    df = build_best_features(tmp_path / "runs", tmp_path / "out")
    assert df["meuhedet_incremental_value"].isna().all() and df["retrained_selected"].isna().all()
    md = (tmp_path / "out" / "best_features.md").read_text(encoding="utf-8")
    assert f"> **WARNING: no ablation_results.csv in {tmp_path / 'out' / 'ablation'}; question 7" in md
    assert md.index("WARNING: no ablation_results.csv") < md.index("## 1.")
    assert "no local eFalls retraining run" in md and "no nonlinear model run" in md


def test_explicit_ablation_dir(tmp_path):
    make_fake_run(tmp_path / "runs", kind="efalls_published_scoring", model="efalls_published", seed=1)
    base = make_fake_run(tmp_path / "abl", kind="ablation_member", model="logistic_unpenalized", seed=7, experiment="abl__baseline")
    step = make_fake_run(tmp_path / "abl", kind="ablation_member", model="logistic_unpenalized", seed=8, experiment="abl__cognition",
                         extra_features={"mini_cog_score": 0.25})
    summarize_ablation(base, [("+ cognition", step)], tmp_path / "somewhere", n_bootstrap=20)
    df = build_best_features(tmp_path / "runs", tmp_path / "out", ablation_dir=tmp_path / "somewhere")
    assert "+ cognition" in df.set_index("feature").loc["mini_cog_score", "meuhedet_incremental_value"]


def test_published_coefficients_come_from_the_published_run_config(tmp_path):
    run = make_fake_run(tmp_path / "runs", kind="efalls_published_scoring", model="efalls_published", seed=1)
    raw = yaml.safe_load((PROJECT_ROOT / "configs" / "models" / "efalls_published.yaml").read_text(encoding="utf-8"))
    raw["terms"]["binary"]["falls"] = 0.5
    edited = tmp_path / "edited_coefficients.yaml"
    edited.write_text(yaml.safe_dump(raw), encoding="utf-8")
    cfg = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    cfg["model"]["published_equation"]["config"] = str(edited)
    (run / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    df = build_best_features(tmp_path / "runs", tmp_path / "out", published_config="does/not/exist.yaml")
    assert df.set_index("feature").loc["falls", "published_efalls_coefficient"] == pytest.approx(0.5)
    assert f"published coefficients: {edited} (model.published_equation.config of run {run.name})" in (tmp_path / "out" / "best_features.md").read_text(encoding="utf-8")
    cfg["model"]["published_equation"]["config"] = "configs/models/missing.yaml"
    (run / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ConfigError, match="not found"):
        build_best_features(tmp_path / "runs", tmp_path / "out")


def test_evidence_from_different_dataset_versions_is_flagged(tmp_path):
    make_fake_run(tmp_path / "runs", kind="efalls_retrained", model="lasso_logistic_cv", seed=1)
    make_fake_run(tmp_path / "runs", kind="alternative_model", model="random_forest", seed=2, dataset_version="v2")
    build_best_features(tmp_path / "runs", tmp_path / "out")
    md = (tmp_path / "out" / "best_features.md").read_text(encoding="utf-8")
    assert "WARNING: the evidence comes from runs on different dataset versions or test rows" in md
    assert "dataset synthetic_v1" in md and "dataset v2" in md
