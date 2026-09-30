"""Deterministic fake run directories that follow ``docs/ARTIFACT_SCHEMAS.md`` (SYNTHETIC test data only).

Metrics are computed from simulated predictions with the real metric functions, so every artifact of a fake run
is internally consistent. Bootstrap CIs are replaced by a fixed ±0.02 band to keep tests fast.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.special import expit

from falls_ml.artifacts import write_csv, write_json, write_parquet
from falls_ml.evaluation import metrics as M
from falls_ml.seeding import rng_for

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_CONFIGS = PROJECT_ROOT / "configs" / "experiments" / "fixture"
SPEC_PATH = PROJECT_ROOT / "configs" / "features" / "efalls_v1.yaml"
EFALLS_KINDS = ("efalls_published_scoring", "efalls_retrained", "efalls_retrained_reduced")
PUBLISHED_VARIANTS = ("lp_c_box_s3_1", "lp_a_table_s3_2", "lp_b_label_swap", "lp_d2_numeric_swap")
METRIC_NAMES = ("auroc", "pr_auc", "brier", "calibration_slope", "calibration_intercept", "citl", "oe_ratio")
THRESHOLDS = (0.10, 0.15, 0.20, 0.25)
LINEAR_MODELS = {"efalls_published", "lasso_logistic_cv", "logistic_unpenalized", "elastic_net_logistic"}

#: raw feature -> (strength, design columns). "anxiety" carries no signal (published coefficient 0.0).
FEATURES: dict[str, tuple[float, tuple[str, ...]]] = {
    "falls": (0.30, ("falls",)),
    "age_years": (0.25, ("age_years",)),
    "fracture": (0.20, ("fracture",)),
    "polypharmacy_count_120d": (0.15, ("polypharmacy_log_p1_div10",)),
    "housebound": (0.12, ("housebound",)),
    "sex": (0.10, ("sex=male",)),
    "dementia": (0.10, ("dementia",)),
    "osteoporosis": (0.08, ("osteoporosis",)),
    "bmi_value": (0.05, ("bmi_category=underweight", "bmi_category=obese")),
    "anxiety": (0.0, ("anxiety",)),
}


def _run_id(experiment: str, model: str, seed: int) -> str:
    token = hashlib.sha256(f"{experiment}|{model}|{seed}".encode()).hexdigest()[:8]
    return f"2026-09-15_{experiment}_{model}_{token}"


def _cohort(cohort_seed: int, n: int, split: str) -> tuple[pd.DataFrame, np.ndarray]:
    rng = rng_for(cohort_seed, f"fake_cohort:{split}")
    x = rng.normal(size=n)
    y = (rng.random(n) < expit(-2.0 + x)).astype(np.int8)
    frame = pd.DataFrame({"research_id": [f"P{cohort_seed}_{split[0]}{i:05d}" for i in range(n)],
                          "index_date": pd.to_datetime("2022-01-01" if split == "test" else "2021-01-01"),
                          "outcome": y, "male": rng.random(n) < 0.45})
    return frame, x


def _lp(x: np.ndarray, seed: int, component: str, signal: float, citl_shift: float, slope_multiplier: float) -> np.ndarray:
    z = x + rng_for(seed, component).normal(size=x.size) / signal
    return -2.0 + citl_shift + slope_multiplier * z / (1.0 + 1.0 / signal**2)


def _entry(y: np.ndarray, p: np.ndarray, lp: np.ndarray) -> dict[str, Any]:
    summary = M.performance_summary(y, p, lp=lp, thresholds=THRESHOLDS)
    entry = {k: v for k, v in summary.items() if k not in METRIC_NAMES}
    for name in METRIC_NAMES:
        est = summary[name]
        entry[name] = {"estimate": est, "ci_low": None if est is None else est - 0.02, "ci_high": None if est is None else est + 0.02}
    return entry


def _variants(kind: str, lp: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    if kind == "efalls_published_scoring":
        base = {"lp_c_box_s3_1": lp, "lp_a_table_s3_2": lp + 0.15, "lp_b_label_swap": lp + 0.303708,
                "lp_d2_numeric_swap": lp + 0.15 - 0.303708}
    else:
        base = {"uncalibrated": lp}
    out = {v: (expit(l), l) for v, l in base.items()}
    for v, l in base.items():
        recal = 0.1 + 0.9 * l
        out["recalibrated" if kind != "efalls_published_scoring" else f"{v}+recalibrated"] = (expit(recal), recal)
    return out


def reduced_fixture_features() -> list[str]:
    """Available predictors declared by the reduced fixture config (66 of 78)."""
    raw = yaml.safe_load((FIXTURE_CONFIGS / "efalls_retrained_reduced.yaml").read_text(encoding="utf-8"))
    return list(raw["preprocessing"]["features"])


def fake_efalls_coverage(kind: str, *, unavailable: Sequence[str] = ()) -> dict[str, Any] | None:
    """``metrics.efalls_coverage`` for eFalls kinds, computed from the real feature spec (None for other kinds)."""
    if kind not in EFALLS_KINDS:
        return None
    from falls_ml.experiment import efalls_coverage
    from falls_ml.features.spec import load_feature_spec

    spec = load_feature_spec(SPEC_PATH)
    if kind == "efalls_retrained_reduced" and not unavailable:
        available = reduced_fixture_features()
    else:
        available = [n for n in spec.predictor_names() if n not in set(unavailable)]
    return efalls_coverage(spec, available, kind=kind)


def _config(kind: str, model: str, experiment: str) -> dict[str, Any]:
    name = {"efalls_published_scoring": "efalls_published_scoring.yaml", "efalls_retrained": "efalls_retrained_lasso.yaml",
            "efalls_retrained_reduced": "efalls_retrained_reduced.yaml"}.get(kind, f"{model}.yaml")
    raw = yaml.safe_load((FIXTURE_CONFIGS / name).read_text(encoding="utf-8"))
    raw["experiment"]["kind"], raw["experiment"]["name"] = kind, experiment
    return raw


def make_fake_run(runs_dir: str | Path, *, kind: str, model: str, seed: int, synthetic: bool = True,
                  test_rows_sha256: str = "abc", dataset_version: str = "synthetic_v1", experiment: str | None = None,
                  effective_experiment_label: str = "efalls_published_scoring", signal: float = 1.0, citl_shift: float = 0.0,
                  slope_multiplier: float = 1.0, extra_features: Mapping[str, float] | None = None, cohort_seed: int = 0,
                  n_test: int = 600, with_plots: bool = False, tuning: bool | None = None,
                  eligible: bool = False, served_recalibrated: bool = False, data_sha256: str | None = None,
                  omitted: Sequence[str] = (), unavailable_efalls: Sequence[str] = ()) -> Path:
    """Write a complete fake run directory (ending with ``RUN_COMPLETE.json``) and return its path.

    ``served_recalibrated`` makes the bundle serve the recalibrated variant (``primary_variant`` = ``served_variant``).
    ``data_sha256`` defaults to a hash of ``dataset_version``; ``omitted`` lists design columns forced to 0 (Stata-style).
    ``unavailable_efalls`` sets the eFalls predictors missing from ``efalls_coverage`` (reduced runs default to the
    reduced fixture config's illustrative list).
    """
    experiment = experiment or (kind if kind in {"efalls_published_scoring", "efalls_retrained_reduced"} else
                                "efalls_retrained_lasso" if kind == "efalls_retrained" else model)
    run_id = _run_id(experiment, model, seed)
    run = Path(runs_dir) / run_id
    (run / "plots").mkdir(parents=True)
    (run / "model").mkdir()
    published = kind == "efalls_published_scoring"
    fitted_linear = model in LINEAR_MODELS and not published
    is_partial = published and effective_experiment_label == "efalls_partial_scoring"
    coverage = fake_efalls_coverage(kind, unavailable=tuple(unavailable_efalls) or (("falls",) if is_partial else ()))
    tuning = (kind == "alternative_model" and model in {"random_forest", "hist_gradient_boosting", "elastic_net_logistic"}) if tuning is None else tuning
    features = {**{f: s for f, (s, _) in FEATURES.items()}, **dict(extra_features or {})}
    design = {**{f: cols for f, (_, cols) in FEATURES.items()}, **{f: (f,) for f in (extra_features or {})}}
    noise = rng_for(seed, f"fake_noise:{experiment}")

    performance: dict[str, dict[str, Any]] = {}
    predictions: dict[str, pd.DataFrame] = {}
    for split, n in (("validation", n_test // 2), ("test", n_test)):
        frame, x = _cohort(cohort_seed, n, split)
        lp = _lp(x, seed, f"fake_lp:{experiment}:{split}", signal, citl_shift, slope_multiplier)
        variants = _variants(kind, lp)
        y = frame["outcome"].to_numpy(dtype=np.int64)
        performance[split] = {v: _entry(y, p, l) for v, (p, l) in variants.items()}
        base = "lp_c_box_s3_1" if published else "uncalibrated"
        recal = f"{base}+recalibrated" if published else "recalibrated"
        pred = frame[["research_id", "index_date", "outcome"]].assign(
            risk_uncalibrated=variants[base][0], risk_recalibrated=variants[recal][0],
            linear_predictor=variants[base][1] if model in LINEAR_MODELS else np.nan)
        if published:
            for v in PUBLISHED_VARIANTS:
                pred[f"lp_{v}"], pred[f"risk_{v}"] = variants[v][1], variants[v][0]
        predictions[split] = pred
        write_parquet(run / f"predictions_{split}.parquet", pred)
    test_pred = predictions["test"]
    uncalibrated = "lp_c_box_s3_1" if published else "uncalibrated"
    primary = (f"{uncalibrated}+recalibrated" if published else "recalibrated") if served_recalibrated else uncalibrated
    risk_column = "risk_recalibrated" if served_recalibrated else "risk_uncalibrated"
    y_test, p_test = test_pred["outcome"].to_numpy(dtype=np.int64), test_pred[risk_column].to_numpy()

    # ---- feature tables
    coef_rows = []
    for f, cols in design.items():
        for j, col in enumerate(cols):
            c = 0.0 if features[f] == 0.0 and model == "lasso_logistic_cv" else features[f] * (1 - 0.5 * j) + noise.normal(0, 0.005)
            c = 0.0 if col in omitted else c
            coef_rows.append({"design_column": col, "feature": f, "coefficient": c, "standardized_coefficient": np.nan if published else 2 * c,
                              "odds_ratio": np.exp(c), "abs_coefficient": abs(c), "selected": c != 0.0,
                              "direction": "increases_risk" if c > 0 else "decreases_risk" if c < 0 else "none",
                              "rank": 0, "relative_to_reference": False, "omitted": col in omitted})
    coef = pd.DataFrame(coef_rows)
    coef["rank"] = coef["abs_coefficient"].rank(ascending=False, method="first").astype(int)
    intercept = pd.DataFrame([{"design_column": "_intercept", "feature": "_intercept", "coefficient": -2.0, "standardized_coefficient": np.nan,
                               "odds_ratio": np.exp(-2.0), "abs_coefficient": 2.0, "selected": True, "direction": "none", "rank": 0,
                               "relative_to_reference": False, "omitted": False}])
    write_csv(run / "coefficients.csv", pd.concat([coef, intercept], ignore_index=True) if model in LINEAR_MODELS else coef.iloc[0:0])

    perm = {f: features[f] * (1.5 if model in {"random_forest", "hist_gradient_boosting"} else 1.0) * 0.1 + noise.normal(0, 0.0005) for f in features}
    fi = pd.DataFrame({"feature": list(perm), "model": model, "permutation_importance_mean": list(perm.values()),
                       "permutation_importance_std": 0.002, "shap_mean_abs": np.nan})
    fi["importance_rank"] = fi["permutation_importance_mean"].rank(ascending=False, method="first").astype(int)
    single = coef.groupby("feature")["coefficient"].agg(["count", "first"])
    fi["coefficient"] = fi["feature"].map(lambda f: single.loc[f, "first"] if model in LINEAR_MODELS and single.loc[f, "count"] == 1 else np.nan)
    fi["odds_ratio"] = np.exp(fi["coefficient"])
    stab_rows = []
    for f, cols in design.items():
        for col in cols:
            freq = min(1.0, 0.1 + 3 * features[f])
            sign = 1.0 if features[f] >= 0.08 else 0.6
            stab_rows.append({"feature": col, "raw_feature": f, "n_bootstrap": 20, "n_selected": int(round(freq * 20)), "selection_frequency": freq,
                              "selection_frequency_mcse": 0.02, "raw_feature_selection_frequency": 1 - (1 - freq) ** len(cols),
                              "coef_mean": features[f], "coef_median": features[f], "coef_sd": 0.02,
                              "sign_stability": sign, "permutation_importance_mean": perm[f], "permutation_importance_std": 0.002,
                              "robust": freq >= 0.8 and sign >= 0.9})
    stability = pd.DataFrame(stab_rows)
    fi["selection_frequency"] = fi["feature"].map(stability.groupby("raw_feature")["selection_frequency"].max()) if fitted_linear else np.nan
    write_csv(run / "feature_importance.csv", fi[["feature", "model", "importance_rank", "coefficient", "odds_ratio", "permutation_importance_mean",
                                                  "permutation_importance_std", "shap_mean_abs", "selection_frequency"]])
    write_csv(run / "feature_stability.csv", stability if fitted_linear else stability.iloc[0:0])
    inst = pd.DataFrame({"threshold": THRESHOLDS, "n_bootstrap": 20, "mean_classification_instability": 0.05,
                         "max_classification_instability": 0.3, "mape_mean": 0.01, "mape_mean_mcse": 0.001, "mape_median": 0.008})
    write_csv(run / "instability.csv", inst if fitted_linear else inst.iloc[0:0])

    lambdas = np.geomspace(0.1, 1e-4, 20)
    cv = pd.DataFrame({"lambda": lambdas, "cv_mean_deviance": 0.8 + (np.log10(lambdas) + 2.5) ** 2 * 0.01, "cv_se": 0.01,
                       "n_nonzero": np.linspace(0, len(coef), 20).astype(int), "selected": np.arange(20) == 9})
    write_csv(run / "cv_results.csv", cv if model == "lasso_logistic_cv" else cv.iloc[0:0])
    if model == "lasso_logistic_cv":
        selected_cols = coef.loc[coef["coefficient"] != 0.0]
        write_csv(run / "unpenalized_refit.csv", pd.DataFrame({
            "term": ["_cons", *selected_cols["design_column"]], "coefficient": [-2.0, *selected_cols["coefficient"]], "se": 0.1, "z": 1.0,
            "p_value": 0.3, "odds_ratio": np.exp([-2.0, *selected_cols["coefficient"]]), "or_ci_low": 0.5, "or_ci_high": 2.0,
            "status": "estimated"}))
        write_csv(run / "fp_selection.csv", pd.DataFrame([{
            "variable": "polypharmacy_count_120d", "powers": "[0.0]", "shift": 1.0, "scale": 10.0, "decision": "fp1", "deviance_linear": 812.4,
            "fp1_powers": "[0.0]", "deviance_fp1": 805.1, "fp2_powers": "[0.0, 0.0]", "deviance_fp2": 804.9, "p_nonlinear": 0.02,
            "p_fp2_vs_fp1": 0.9, "cycles_run": 2, "converged": True, "omissions_json": "{}"}]))
    trials = pd.DataFrame({"trial": range(4), "params_json": [json.dumps({"max_depth": d}) for d in (2, 4, 6, 8)],
                           "objective": [0.60, 0.64, 0.62, 0.61], "auroc": [0.70, 0.73, 0.72, 0.71], "brier": [0.11, 0.10, 0.105, 0.108],
                           "calibration_slope": [0.9, 1.0, 1.1, 1.2], "citl": 0.0, "selected": [False, True, False, False]})
    write_csv(run / "hyperparameter_results.csv", trials if tuning else trials.iloc[0:0])

    grouped = M.grouped_calibration(y_test, p_test).assign(kind="grouped")
    smooth = M.smoothed_calibration(y_test, p_test).assign(kind="smoothed", group=np.nan, n=np.nan, ci_low=np.nan, ci_high=np.nan)
    write_csv(run / "calibration.csv", pd.concat([grouped, smooth]).assign(split="test", variant=primary))
    write_csv(run / "confusion_matrices.csv", pd.DataFrame([{"split": "test", "variant": primary, **{k: t[k] for k in (
        "threshold", "tp", "fp", "tn", "fn", "sensitivity", "specificity", "ppv", "npv", "f1")}} for t in performance["test"][primary]["thresholds"]]))
    dc = M.decision_curve(y_test, p_test, np.round(np.arange(0.01, 0.505, 0.01), 10))
    write_csv(run / "decision_curve.csv", dc.assign(split="test", variant=primary))
    male = _cohort(cohort_seed, n_test, "test")[0]["male"].to_numpy()
    test_variants = _variants(kind, _lp(_cohort(cohort_seed, n_test, "test")[1], seed, f"fake_lp:{experiment}:test", signal, citl_shift,
                                        slope_multiplier))
    write_csv(run / "subgroup_metrics.csv", pd.DataFrame([
        {"split": "test", "variant": variant, "subgroup_variable": "sex", "subgroup_level": level, "n": int(mask.sum()),
         "n_events": int(y_test[mask].sum()), "auroc": M.auroc(y_test[mask], vp[mask]), "brier": M.brier(y_test[mask], vp[mask]),
         "calibration_slope": M.calibration_slope_intercept(y_test[mask], vlp[mask])[1], "citl": M.citl(y_test[mask], vlp[mask])[0],
         "oe_ratio": M.oe_ratio(y_test[mask], vp[mask])}
        for variant, (vp, vlp) in test_variants.items() for level, mask in (("female", ~male), ("male", male))]))
    write_csv(run / "splits.csv", test_pred[["research_id", "index_date"]].assign(split="test"))

    # ---- metadata files
    source = "synthetic_fixture" if synthetic else "meuhedet_dwh"
    manifest = {"dataset_version": dataset_version, "mapping_version": "fixture_v0", "source": source, "feature_spec_name": "efalls_v1",
                "feature_spec_version": "1.0.0", "feature_spec_sha256": "f" * 64, "data_file": "modeling_dataset.parquet",
                "data_sha256": data_sha256 or hashlib.sha256(dataset_version.encode()).hexdigest(), "n_rows": 3 * n_test, "n_patients": 2 * n_test, "index_date_min": "2020-01-01",
                "index_date_max": "2022-01-01", "outcome_prevalence": float(y_test.mean()), "created_utc": "2026-09-15T00:00:00+00:00",
                "generator": "tests.helpers.fake_run", "scientific_use_allowed": not synthetic, "notes": ""}
    (run / "config.yaml").write_text(yaml.safe_dump(_config(kind, model, experiment), sort_keys=False), encoding="utf-8")
    write_json(run / "dataset_manifest.json", manifest)
    write_json(run / "environment.json", {"python": "3.13", "packages": {}})
    (run / "run_log.jsonl").write_text('{"event": "run_started"}\n', encoding="utf-8")
    write_json(run / "model" / "bundle.json", {"model_name": model, "run_id": run_id})

    def status(variant: str) -> str:
        e = performance["test"][variant]
        slope, citl = e["calibration_slope"]["estimate"], e["citl"]["estimate"]
        if slope is None or citl is None:
            return "not_assessed"
        return "adequate" if 0.8 <= slope <= 1.2 and abs(citl) <= 0.2 else "miscalibrated"

    top = fi.sort_values("permutation_importance_mean", ascending=False)["feature"].head(10).tolist()
    by_split = {s: {"n_rows": len(predictions[s]), "n_patients": len(predictions[s]), "n_events": int(predictions[s]["outcome"].sum()),
                    "prevalence": float(predictions[s]["outcome"].mean()), "index_date_min": "2022-01-01", "index_date_max": "2022-01-01"}
                for s in predictions}
    metrics = {
        "schema_version": "1.0", "run_id": run_id, "created_utc": "2026-09-15T00:00:00+00:00",
        "experiment": {"name": experiment, "kind": kind, "layers": ["L1_published", "L2_assumption", "L3a_meuhedet_mapping"]
                       + (["L4_alternative"] if kind in {"alternative_model", "ablation_member"} else []), "description": f"fake {kind} run"},
        "model": {"name": model, "params": {"sex_parameterisation": "lp_c_box_s3_1", "unavailable_predictors": ["falls"] if is_partial else []}
                  if published else {}, "best_params": {"max_depth": 4} if tuning else None, "is_linear": model in LINEAR_MODELS,
                  "effective_params": {"max_depth": 4 if tuning else 3, "n_estimators": 100} if model in {"random_forest", "hist_gradient_boosting"}
                  else {"lambda_min_ratio": 1e-4, "selection": "min", "tol": 1e-7} if model == "lasso_logistic_cv" else {},
                  "resampling_params": {"max_depth": 4} if tuning else {},
                  "fit_notes": {"stata_logit_omissions": {"constant_columns": [], "perfect_predictors": [{"column": c, "n_rows_dropped": 12}
                                                                                                         for c in omitted],
                                                          "n_rows_dropped": 12 * len(omitted)} if model in LINEAR_MODELS and not published else {},
                                "fp_selection_omissions": {}}},
        "dataset": manifest, "synthetic_fixture": synthetic, "scientific_use_allowed": not synthetic,
        "feature_set": {"name": "efalls_v1", "version": "1.0.0", "sha256": "f" * 64, "pure_efalls": not extra_features, "n_features": len(features)},
        "efalls_coverage": coverage,
        "cohort": {"n_rows": 3 * n_test, "n_patients": 2 * n_test, "n_events": int(3 * n_test * y_test.mean()), "prevalence": float(y_test.mean()),
                   "by_split": by_split},
        "split": {"strategy": "temporal", "description": "fake temporal split", "limitation_note": "", "seed": 42,
                  "test_rows_sha256": test_rows_sha256, "embargo_removed": {}, "patient_overlap_removed": {}},
        "primary_variant": primary, "served_variant": primary, "transportability_variant": "lp_c_box_s3_1" if published else None,
        "performance": performance,
        "calibration": {"method": "logistic_intercept_slope", "fitted_on": "validation",
                        "params": {v: {"alpha": 0.1, "beta": 0.9} for v in (PUBLISHED_VARIANTS if published else ("uncalibrated",))}},
        "calibration_status": status(primary), "calibration_status_uncalibrated": status(uncalibrated),
        "features": {"n_design_columns": len(coef), "n_selected": int((coef["coefficient"] != 0).sum()) if model in LINEAR_MODELS else None,
                     "top_features": top},
        "lasso": {"lambda_star": float(lambdas[9]), "lambda_max": float(lambdas[0]), "lambda_ratio": 1e-3, "n_lambda": 20, "cv_folds": 10,
                  "cv_criterion": "mean_deviance", "selection_rule": "min", "cv_minimum_identified": True,
                  "n_selected": int((coef["coefficient"] != 0).sum()), "intercept": -2.0, "converged": True, "seed": 0}
        if model == "lasso_logistic_cv" else None,
        "hyperparameter_search": {"enabled": bool(tuning), "method": "random", "n_trials": 4 if tuning else 0, "objective": {"auroc": 1.0}},
        "timing": {"tuning_seconds": 2.0 if tuning else 0.0, "final_fit_seconds": 0.5 if model in LINEAR_MODELS else 5.0,
                   "total_train_seconds": (0.5 if model in LINEAR_MODELS else 5.0) + (2.0 if tuning else 0.0),
                   "predict_seconds_per_1000_rows": 0.01},
        "test_evaluation_registry": {"registry_file": "../test_evaluation_registry.jsonl", "prior_evaluations_same_test_rows": 0,
                                     "prior_config_sha256": [], "purpose": "train"},
        "eligibility": {"eligible_for_further_validation": eligible, "reasons": [] if eligible else ["synthetic fixture"]},
        "comparison_to_efalls": {"reference_run_id": None, "reference_variant": "lp_c_box_s3_1", "delta_auroc_test": None, "candidates": [],
                                 "note": "point estimate; paired bootstrap CIs are in reports/comparison_paired_differences.csv"},
        "published_scoring": {
            "sex_parameterisations": list(PUBLISHED_VARIANTS), "primary": "lp_c_box_s3_1", "co_reported": "lp_a_table_s3_2",
            "configured_variant": "lp_c_box_s3_1", "effective_experiment_label": effective_experiment_label,
            "coverage": {"lp_variance_share_available": 0.82 if is_partial else 1.0, "mandatory_unavailable": ["falls"] if is_partial else [],
                         "threshold": 0.9},
            "unavailable_fill": "zero", "low_support_zeroed": False,
            "unavailable_predictors": [{"feature": "falls", "coefficient": 0.3009161, "sail_prevalence": 0.2,
                                        "expected_mean_lp_shift": -0.06}] if is_partial else [],
            "efalls_feature_coverage": "incomplete" if is_partial else "complete",
            "sex_specific_citl_lp_c": {"male_minus_female": 0.05, "ci_low": -0.1, "ci_high": 0.2,
                                       "note": "descriptive only; mixes true sex-specific miscalibration (D-01)"},
            "sex_specific_oe": {v: {"female": 1.0, "male": 0.9} for v in PUBLISHED_VARIANTS},
        } if published else None,
        "internal_validation": {"optimism": None, "iecv": None, "heterogeneity": None},
        "missingness": {"bmi_value": {"missing_rate": 0.3, "missing_rate_by_split": {"train": 0.32, "validation": 0.29, "test": 0.25},
                                      "rule": "missing_category", "handling": "separate 'missing' category level"},
                        "falls": {"missing_rate": 0.0, "missing_rate_by_split": {"train": 0.0, "validation": 0.0, "test": 0.0},
                                  "rule": "absent_is_zero", "handling": "absent code counted as 0"}},
        "limitations": ["SYNTHETIC FIXTURE: these results are software test output, not scientific evidence."] if synthetic else [],
        "production_readiness": {"bundle_saved": True, "thresholds_approved": False, "mappings_clinically_validated": not synthetic,
                                 "unavailable_predictors": len(coverage["unavailable"]) if coverage is not None else 0},
        "warnings": [], "code_version": {"git_commit": None, "source_tree_sha256": "c" * 64},
        "artifacts": {"report_html": "report.html", "report_md": "report.md", "model_bundle": "model/"},
    }
    write_json(run / "metrics.json", metrics)
    if with_plots:
        _plots(run, test_pred, risk_column, coef, fi, stability if fitted_linear else None, trials if tuning else None,
               cv if model == "lasso_logistic_cv" else None, grouped, smooth, dc)
    write_json(run / "RUN_COMPLETE.json", {"run_id": run_id, "completed_utc": "2026-09-15T00:00:01+00:00"})
    return run


def _plots(run: Path, pred: pd.DataFrame, risk_column: str, coef: pd.DataFrame, fi: pd.DataFrame, stability: pd.DataFrame | None,
           trials: pd.DataFrame | None, cv: pd.DataFrame | None, grouped: pd.DataFrame, smooth: pd.DataFrame, dc: pd.DataFrame) -> None:
    from falls_ml.reporting import plots as P

    y, p = pred["outcome"].to_numpy(dtype=np.int64), pred[risk_column].to_numpy()
    d = run / "plots"
    P.plot_calibration_before_after(y, pred["risk_uncalibrated"].to_numpy(), pred["risk_recalibrated"].to_numpy(),
                                    d / "calibration_before_after.png", "Calibration before and after recalibration")
    P.plot_roc({"model": (y, p)}, d / "roc.png", "ROC")
    P.plot_precision_recall({"model": (y, p)}, d / "precision_recall.png", "PR", prevalence=float(y.mean()))
    P.plot_calibration(grouped, smooth, d / "calibration.png", "Calibration", predictions=p)
    P.plot_decision_curve(dc, d / "decision_curve.png", "Decision curve")
    P.plot_risk_distribution(y, p, d / "risk_distribution.png", "Risk")
    P.plot_feature_importance(fi, d / "feature_importance.png", "Importance")
    if not np.isnan(pred["linear_predictor"]).all():
        P.plot_coefficients(coef, d / "coefficients.png", "Coefficients")
    if stability is not None:
        P.plot_selection_stability(stability, d / "selection_stability.png", "Stability")
    if trials is not None:
        P.plot_hyperparameter_search(trials, d / "hyperparameter_search.png", "Search")
    if cv is not None:
        P.plot_lasso_cv_path(cv, d / "lasso_cv_path.png", "LASSO CV")
