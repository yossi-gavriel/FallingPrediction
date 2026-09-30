# Run artifact schemas (contract between `experiment.py`, reporting, comparison and inference)

This document defines every file in a run directory (`runs/<run_id>/`). Writers must produce exactly these columns and keys; readers may rely on them.

**Conventions**
- Missing or non-applicable numeric values are JSON `null` / empty CSV cells.
- Probabilities are in [0, 1].
- `split` ∈ {`train`, `validation`, `test`, `full_cohort`}. `performance.train` is apparent (in-sample) performance, reported for transparency only.
- `variant` names a prediction variant:
  - `uncalibrated`, `recalibrated`;
  - for published eFalls scoring, `<sex_parameterisation>` (e.g. `lp_c_box_s3_1`) and `<sex_parameterisation>+recalibrated`.
- Portability (Windows 10/11 and macOS/Linux produce identical artifacts): text artifacts (JSON, YAML, CSV, Markdown, HTML, JSON lines) are UTF-8 with LF line endings on every OS; relative paths recorded in JSON/YAML use `/` separators (absolute paths are recorded as given by the OS).
- Run id: `<YYYY-MM-DD>_<label>_<token>`. `label` is `<experiment.name>_<model.name>`, or just `<experiment.name>` when it already contains the model name (model overrides `<name>__<model>`, ablation steps), truncated to 60 characters; `token` is the first 8 hex digits of `sha256(config_sha256|data_sha256|source_tree_sha256|created_utc[|clock])` and keeps truncated ids unique. `experiment.name` must match `[A-Za-z0-9][A-Za-z0-9_.-]{0,199}` without a trailing dot (it becomes part of a directory name; Windows forbids `: * ? " < > | \ /`).

## `metrics.json`

```jsonc
{
  "schema_version": "1.2",                        // 1.2 adds efalls_coverage
  "run_id": "2026-09-14_efalls_retrained_lasso_lasso_logistic_cv_ab12cd34",
  "created_utc": "...",
  "experiment": {"name": "...", "kind": "efalls_retrained", "layers": ["L1_published", "L2_assumption"], "description": "..."},
                                                   // kind: efalls_published_scoring | efalls_retrained | efalls_retrained_reduced
                                                   //       | alternative_model | ablation_member
  "model": {"name": "lasso_logistic_cv", "params": {}, "best_params": {}, "effective_params": {}, "is_linear": true,
            "resampling_params": {},                // params used by stability / optimism / IECV refits (= final fit params)
            "fit_notes": {"stata_logit_omissions": {}, "fp_selection_omissions": {}}},
  "dataset": {/* verbatim DatasetManifest */},
  "outcome": {"name": "outcome_12m", "horizon": "1y", "layer": "L1_published", "published_efalls_outcome": true,
              "concept": "...", "status": "..."},  // exploratory outcomes (horizon in days, e.g. "181d") set published_efalls_outcome false
                                                   // and add the EXPLORATORY OUTCOME limitation/warning
  "synthetic_fixture": true,                       // true if manifest.source == "synthetic_fixture"
  "scientific_use_allowed": false,
  "feature_set": {"name": "efalls_v1", "version": "1.0.0", "sha256": "...", "pure_efalls": true, "n_features": 78},
                                                   // sha256 of the spec used: the full spec or FeatureSpec.subset (root-based digest)
  "efalls_coverage": {                             // efalls_published_scoring, efalls_retrained, efalls_retrained_reduced; null otherwise
    "n_available": 66, "n_total": 78,              // exact_efalls_baseline candidates of the root feature spec
    "coverage_pct": 84.6,                          // round(100 * n_available / n_total, 1)
    "available": ["age_years", "..."], "unavailable": ["bmi_value", "..."],   // spec order; published scoring: unavailable =
                                                   //   model.published_equation.unavailable_predictors (fixed at 0, M-11)
    "n_published_retained_available": 56, "n_published_retained_total": 62,  // retained_in_published_model: true
    "mandatory_unavailable": [],                   // cohort.mandatory_for_efalls_label members not available (report warns)
    "is_full_efalls_feature_set": false,
    "label": "REDUCED eFalls predictor set – NOT a full eFalls reproduction (66/78 eFalls predictors available)"
  },
  "cohort": {
    "n_rows": 0, "n_patients": 0, "n_events": 0, "prevalence": 0.0,
    "by_split": {"train": {"n_rows": 0, "n_patients": 0, "n_events": 0, "prevalence": 0.0,
                           "index_date_min": "YYYY-MM-DD", "index_date_max": "YYYY-MM-DD"},
                 "validation": {}, "test": {}}
  },
  "split": {"strategy": "temporal", "description": "...", "limitation_note": "", "seed": 42,
            "test_rows_sha256": "...",          // sha256 of sorted (research_id, index_date) of the test partition; comparisons require equality (D-19)
            "embargo_removed": {}, "patient_overlap_removed": {}},
  "primary_variant": "recalibrated",               // = served_variant: the prediction the bundle serves; used for the
                                                   //   executive summary, eligibility, thresholds, plots and comparison rows
  "served_variant": "recalibrated",                // uncalibrated | recalibrated | <sex variant> | <sex variant>+recalibrated
  "transportability_variant": "lp_c_box_s3_1",     // published scoring only: uncalibrated published equation (null otherwise)
  "performance": {
    "<split>": {
      "<variant>": {
        "n": 0, "n_events": 0,
        "auroc": {"estimate": 0.0, "ci_low": 0.0, "ci_high": 0.0},
        "pr_auc": {"estimate": 0.0, "ci_low": 0.0, "ci_high": 0.0},
        "brier": {"estimate": 0.0, "ci_low": 0.0, "ci_high": 0.0},
        "calibration_slope": {"estimate": 0.0, "ci_low": 0.0, "ci_high": 0.0},
        "calibration_intercept": {"estimate": 0.0, "ci_low": 0.0, "ci_high": 0.0},  // free-slope intercept a in logit p = a + b*LP
        "citl": {"estimate": 0.0, "ci_low": 0.0, "ci_high": 0.0},                   // intercept with LP offset
        "oe_ratio": {"estimate": 0.0, "ci_low": 0.0, "ci_high": 0.0},
        "mean_predicted": 0.0, "observed_rate": 0.0,
        "calibration_indices": {"ici": 0.0, "e50": 0.0, "e90": 0.0},
        "thresholds": [{"threshold": 0.1, "sensitivity": 0.0, "specificity": 0.0, "ppv": 0.0, "npv": 0.0, "f1": 0.0,
                        "tp": 0, "fp": 0, "tn": 0, "fn": 0, "net_benefit": 0.0}]
      }
    }
  },
  "calibration": {"method": "logistic_intercept_slope", "fitted_on": "validation", "params": {"<variant>": {"alpha": 0.0, "beta": 1.0}}},
  "calibration_status": "adequate|miscalibrated|not_assessed",   // judged on test split, served variant; not_assessed if slope/CITL undefined
  "calibration_status_uncalibrated": "adequate|miscalibrated|not_assessed",
  "features": {"n_design_columns": 0, "n_selected": 0, "top_features": ["..."]},
  "lasso": {"lambda_star": 0.0, "lambda_max": 0.0, "lambda_ratio": 0.0001, "n_lambda": 100, "cv_folds": 10,
            "cv_criterion": "mean_deviance", "selection_rule": "min", "cv_minimum_identified": true, "n_selected": 0,
            "intercept": 0.0, "converged": true, "seed": 0},  // or null
  "hyperparameter_search": {"enabled": false, "method": "grid", "n_trials": 0, "objective": {}},
  "timing": {"tuning_seconds": 0.0, "final_fit_seconds": 0.0, "total_train_seconds": 0.0, "predict_seconds_per_1000_rows": 0.0},
  "test_evaluation_registry": {"registry_file": "../test_evaluation_registry.jsonl", "prior_evaluations_same_test_rows": 0,
                               "prior_config_sha256": [], "purpose": "train|reproduce"},
  "eligibility": {"eligible_for_further_validation": false, "reasons": ["..."]},
  "comparison_to_efalls": {"reference_run_id": null, "reference_variant": "lp_c_box_s3_1", "delta_auroc_test": null,
                           "candidates": [], "note": "point estimate; paired bootstrap CIs are in reports/comparison_paired_differences.csv"},
  "published_scoring": {                            // only for kind == efalls_published_scoring, else null
    "sex_parameterisations": ["lp_c_box_s3_1", "lp_a_table_s3_2", "lp_b_label_swap", "lp_d2_numeric_swap"],
    "primary": "lp_c_box_s3_1",
    "co_reported": "lp_a_table_s3_2",
    "effective_experiment_label": "efalls_published_scoring|efalls_partial_scoring",   // M-11 hard rule
    "coverage": {"lp_variance_share_available": 1.0, "mandatory_unavailable": [], "threshold": 0.90},
    "unavailable_fill": "zero|sail_prevalence",
    "low_support_zeroed": false,
    "unavailable_predictors": [{"feature": "...", "coefficient": 0.0, "sail_prevalence": 0.0, "expected_mean_lp_shift": 0.0}],
    "efalls_feature_coverage": "complete|incomplete",
    "sex_specific_citl_lp_c": {"male_minus_female": 0.0, "ci_low": 0.0, "ci_high": 0.0, "note": "descriptive only; mixes true sex-specific miscalibration (D-01)"},
    "sex_specific_oe": {"<variant>": {"female": 0.0, "male": 0.0}}
  },
  "internal_validation": {"optimism": null, "iecv": null, "heterogeneity": null},
  "missingness": {"<feature>": {"missing_rate": 0.0, "missing_rate_by_split": {"train": 0.0, "validation": 0.0, "test": 0.0},
                                "rule": "absent_is_zero|missing_category|reference_level|forbid",
                                "handling": "plain-language description of what the fitted preprocessing did (e.g. 'training median 3.0 + missing indicator')"}},
  "limitations": ["..."],
  "production_readiness": {"bundle_saved": true, "thresholds_approved": false, "mappings_clinically_validated": false,
                           "unavailable_predictors": 0},                     // count = len(efalls_coverage.unavailable) (0 for non-eFalls kinds)
  "warnings": ["..."],
  "code_version": {"git_commit": null, "source_tree_sha256": "..."},
                                                   // source_tree_sha256: sha256 over the package's *.py files ordered by case-sensitive
                                                   // path components, each as its "/"-separated relative path + content with CRLF hashed
                                                   // as LF, so the same code hashes identically on Windows and macOS/Linux (unchanged
                                                   // from earlier versions for LF files on POSIX). git_commit is reported only when git's
                                                   // top level is the project folder itself (else null)
  "artifacts": {"report_html": "report.html", "report_md": "report.md", "model_bundle": "model/"}
}
```

## CSV / Parquet files

| File | Columns |
|---|---|
| `splits.csv` | `research_id, index_date, split` |
| `cv_results.csv` | `lambda, cv_mean_deviance, cv_se, n_nonzero, selected` (LASSO only; empty file with header otherwise) |
| `hyperparameter_results.csv` | `trial, params_json, objective, auroc, brier, calibration_slope, citl, selected` |
| `coefficients.csv` | `design_column, feature, coefficient, standardized_coefficient, odds_ratio, abs_coefficient, selected, direction, rank, relative_to_reference, omitted` (linear models; header only otherwise). A row with `design_column = "_intercept"` holds the model intercept. `omitted = true` marks Stata-style omissions (coefficient forced to 0) |
| `unpenalized_refit.csv` | LASSO only: `term, coefficient, se, z, p_value, odds_ratio, or_ci_low, or_ci_high, status` (spec §6.2 demonstration refit) |
| `fp_selection.csv` | Retrained eFalls only: `variable, powers, shift, scale, decision, deviance_linear, deviance_fp1, fp1_powers, deviance_fp2, fp2_powers, p_nonlinear, p_fp2_vs_fp1, cycles_run, converged, omissions_json` |
| `heterogeneity_clusters.csv`, `heterogeneity_pooled.csv` | If `evaluation.cluster_column_for_heterogeneity` is set: per-cluster test metrics with SEs and the random-effects pooled summary (D-16) |
| `RUN_COMPLETE.json` | Written last: `{"run_id", "completed_utc"}`. Readers (comparison, reference lookup, reproduce) ignore run directories without it |
| `feature_importance.csv` | `feature, model, importance_rank, coefficient, odds_ratio, permutation_importance_mean, permutation_importance_std, shap_mean_abs, selection_frequency` — one row per **raw feature** (spec name); `coefficient`/`odds_ratio` filled only when the raw feature maps to exactly one design column |
| `feature_stability.csv` | `feature, raw_feature, n_bootstrap, n_selected, selection_frequency, selection_frequency_mcse, raw_feature_selection_frequency, coef_mean, coef_median, coef_sd, sign_stability, permutation_importance_mean, permutation_importance_std, robust` — one row per design column for linear models (`feature` = design column). `raw_feature_selection_frequency` = share of replicates in which ANY design column of the raw feature was non-zero (e.g. any FP form) |
| `calibration.csv` | `split, variant, kind (grouped|smoothed), group, mean_predicted, observed_rate, n, ci_low, ci_high` |
| `confusion_matrices.csv` | `split, variant, threshold, tp, fp, tn, fn, sensitivity, specificity, ppv, npv, f1` |
| `decision_curve.csv` | `split, variant, threshold, net_benefit_model, net_benefit_treat_all, net_benefit_treat_none, standardized_net_benefit` |
| `subgroup_metrics.csv` | `split, variant, subgroup_variable, subgroup_level, n, n_events, auroc, brier, calibration_slope, citl, oe_ratio` (cells with < 10 events → metrics null) |
| `instability.csv` | `threshold, n_bootstrap, mean_classification_instability, max_classification_instability, mape_mean, mape_mean_mcse, mape_median` |
| `predictions_validation.parquet`, `predictions_test.parquet` | `research_id, index_date, outcome, risk_uncalibrated, risk_recalibrated, linear_predictor` (+ `lp_<variant>`, `risk_<variant>` for published scoring) |

## Provenance files

| File | Content |
|---|---|
| `config.yaml` | The resolved experiment config (`ExperimentConfig.to_dict()`) |
| `dataset_manifest.json` | The dataset's `DatasetManifest` plus `dataset_path_used` (absolute directory as resolved on the machine that ran it) and `dataset_path_relative` (the same directory relative to the project root that contains the run directory, `/` separators; `null` when the dataset lies outside that project). `reproduce` without `--dataset` tries `dataset_path_relative` first (resolved from the run directory, so a moved or copied project folder still works), then `dataset_path_used` |
| `run_inputs.json` | `config_source`, `dataset_dir`, `dataset_sha256`, `feature_spec_sha256`, `purpose`, and `files` {role: [{`declared` (as written in the config, relative paths with `/`), `resolved` (absolute), `sha256`}]} for the feature spec, extensions and published-equation files. `sha256` is taken with CRLF hashed as LF (identical to the plain file hash for LF files), so an editor or `git core.autocrlf` re-save of unchanged content does not block `reproduce`; any other change does |
| `environment.json` | Python, platform, machine, tracked package versions and `code_version` |

## Modelling dataset manifest (`manifest.json`)

`DatasetManifest` keys (unknown keys are refused): `dataset_version, mapping_version, source, feature_spec_name, feature_spec_version, feature_spec_sha256, data_file, data_sha256, n_rows, n_patients, index_date_min, index_date_max, outcome_prevalence, created_utc, generator, scientific_use_allowed, notes, audit, data_freeze_date`.

- `feature_spec_sha256` identifies the spec the data were validated with: the full spec, or a subset spec (`FeatureSpec.subset`). It decides how a dataset loads: without declared features only the full spec is accepted; an experiment with `preprocessing.features` accepts the full spec or exactly that subset; `validate-dataset` without `--features-from-config` infers the subset from the predictor columns and accepts it only if its SHA-256 equals the manifest. A mismatch lists every expected SHA-256.
- `audit.feature_spec_subset` (subset datasets, informational): `root_feature_spec_sha256, subset_feature_spec_sha256, n_predictors, n_root_predictors, predictors, absent_predictors`.
- `audit.build_dataset` (`falls_ml build-dataset`): `input_file_name, input_sha256, input_rows, feature_spec_sha256, features_from_config, scientific_use_allowed, scientific_use_approval_reference` (required when scientific use is allowed), `built_utc, database_connections: "none"`.
- `audit.meuhedet_build` (`falls_ml meuhedet-build`, source `meuhedet_wide_v1`): the `BuildReport` (`index_date, n_rows_input, n_rows_other_index_dates, other_index_dates, n_rows_ineligible, ineligible_by_reason, n_rows_label_null, label_null_by_reason, n_rows_timing_violation, timing_violations_by_column, n_rows_final, n_patients_final, n_events, outcome_prevalence, outcome_window_days_observed, null_to_zero` (per predictor: NULL→0 conversions by the declared eFalls rule), `sentinel_to_null` (declared sentinel dates set to NULL in the modelling layer by a column-specific contract rule), `source_absent_rows, sex_code_descriptions, birth_date_suspect_rows, index_day_fall_rows, deceased_rows, contract_report` (the contract checks: `wrong_type_examples`, `date_range` with the declared sentinels and their modelling rule, `date_formats`), `coverage` (eFalls coverage with quality counts), `warnings`) plus `input_file_name, input_sha256, mapping {name, version, sha256, status, clinically_validated}, contract {name, version, sha256}, exploratory_outcome {name, horizon, concept, label_column, NOT_EFALLS_OUTCOME: true}, scientific_use_approval_reference, database_connections: "none"`.

## `meuhedet-explore` results directory

`<input name>_explore/` (or `--out`). Everything except `datasets/`, `id_pepper.txt` and `runs/*/predictions_*.parquet` is aggregate and shareable.

- `READINESS.md` / `readiness.json`: **READY TO TRAIN** or **STOPPED — <reason>** with `details`, `next` (what to do), `stages` (input, contract, d00_timing, snapshots, pepper, cohort, split, adequacy, training), `pepper_sha256`, `split_audit`, `adequacy`, `snapshot_totals`, `timing_verdict`. Written on every run, also when the run stops; `--audit-only` ends here.
- `audit/`: `wide_extract_audit.md/.json` (M-13 aggregate audit), `input_read_report.json` (source format, per-column conversions, `wrong_type_examples`, `date_range`, `declared_date_formats`, `date_formats`), `timing_diagnostic.md/.json` (D-00: per pre-index record-date column `n_before_index / n_on_index / n_after_index`, `pct_on_or_after_of_observed`, `positive_offset_days {min, median, p90, p99, max, buckets}`, `affected_rows_by_index_date`, `affected_eligible_rows_by_index_date`, `label_prevalence {affected, clean}`, `days_since_companion` consistency, `root_cause` ∈ {CLEAN, A_SAME_DAY_INCLUSION, B_FUTURE_RECORDS, A_AND_B}, `related_predictors {efalls_features, phase2_columns}`, `dwh_correction`; `verdict {n_eligible_rows_violating_d00, root_cause, affected_efalls_features_by_feature_set, python_repair_possible: false, dwh_correction, policy}`), `snapshot_audit.md/.json/.csv` (one row per Index_Date: `n_rows, n_patients, n_eligible, n_eligible_patients, n_labelled, n_censored, n_label_invalid, n_events, prevalence, n_d00_violations, n_full_window, n_labelled_without_full_window, window_end, status USABLE|EXCLUDED, reason, partition, n_modelling_rows, n_modelling_events`; `totals` with snapshot rows, unique patients, eligible rows and patients, events, prevalence, mean snapshots per patient; `temporal_design {train_end, validation_end, horizon_days, embargo_days, partition_by_snapshot, expected, embargo_removed, rule, limitation_note}` in `--all-index-dates` mode).
- `<input>.id_pepper.txt` next to the input (created on the first run, reused afterwards; local only) and `id_pepper.txt` (copy; local only). Reports carry only `pepper_sha256`.
- `datasets/strict|extended/` (immutable modelling datasets on identical rows; research ids = sha256(pepper | Customer_Full_ID)[:20]; `manifest.audit.meuhedet_build` gains `index_dates`, `rows_by_index_date`, `events_by_index_date`, `timing_violations_by_index_date`).
- `configs/<RUN_LABEL>.yaml`: the concrete experiment configs (`MEUHEDET_EFALLS_STRICT|EXTENDED_180D_EXPLORATORY`, and in `--all-index-dates` mode `..._PATIENT_DISJOINT`).
- `split_audit.md/.json`: `strategy, description, limitation_note, metadata {embargo_removed, patient_overlap_removed}, partitions {train|validation|test: n_rows, n_patients, n_events, prevalence, index_date_min/max, n_snapshots, rows_sha256}, test_rows_sha256, patients_in_several_partitions, patients_in_any_two_partitions, n_rows_total, n_patients_total, mean_snapshots_per_patient, rows_used_by_split, rows_excluded_by_split, reproducibility {input_sha256, pepper_sha256, pepper_source, contract_sha256, mapping_sha256, feature_spec_sha256, split_seed, config_sha256 {strict, extended}, dataset_sha256 {strict, extended}, rule}`.
- `adequacy_strict|extended.md/.json`: `verdict READY|INSUFFICIENT EVENTS FOR EXPLORATORY MODEL FIT, reasons, warnings, partitions {n_rows, n_patients, n_events, n_non_events, prevalence, classes_present}, predictors {constant_in_train, extremely_sparse_binary_in_train, support}, thresholds, riley_reference (information only), total`.
- `runs/` (the two primary runs + test-evaluation registry); `runs_patient_disjoint/` (the sensitivity runs, `--all-index-dates` mode).
- `feature_report_strict|extended.md/.csv`: `canonical_efalls_feature, efalls_concept, meuhedet_source_column, mapping_quality, feature_sets, semantic_dtype, model_dtype, missing_pct, distribution, lasso_coefficient, odds_ratio, non_zero_coefficient, selection_frequency, permutation_importance`.
- `comparison.md/.json` (same-test-rows STRICT vs EXTENDED: N, events, prevalence, predictors, AUROC/PR-AUC/Brier with CIs, calibration intercept/slope before and after recalibration, O:E, non-zero columns/predictors, lambda, `split {strategy, test_rows_sha256}`); `comparison_patient_disjoint.md/.json` (the same for the sensitivity runs); `temporal_test_by_snapshot.md` + `temporal_test_by_snapshot_<set>.csv` (`index_date, n, n_events, prevalence, status ok|INSUFFICIENT EVENTS, auroc, brier, oe_ratio, calibration_slope, citl, served_variant`).
- `SUMMARY.md`: readiness, mode, row funnel (supplied → other/excluded snapshots → ineligible → no usable label → D-00 dropped → used), split with per-partition hashes and the pepper fingerprint, temporal design, adequacy, STRICT / EXTENDED metrics, temporal test performance by snapshot, patient-disjoint sensitivity, stable predictors, data-quality findings, known mapping limitations, scientific status.

## `meuhedet-sensitivity` results directory (v0.5.1)

`<reference>_sensitivity/` (or `--out`), written next to a completed single-snapshot `meuhedet-explore` run that is read, hashed before and after
(`reference_integrity.json {n_files, combined_sha256_before, combined_sha256_after, unchanged, changed_files}`) and never written to.

- `feature_provenance.md/.json`: `d00_guard_columns`, `violating_guard_columns`, per included predictor `features[] {feature, efalls_concept, mapping_quality,
  feature_sets, op, source_columns, source_groups, contract_timing, record_date_column, same_day_evidence_column, d00_evidence {column, kind
  last_record_date|same_day_evidence_sufficient_only, n_on_index, n_after_index, n_affected_eligible, root_cause}, provenance_class ∈
  {EXCLUDED_PROVEN_SAME_DAY_SOURCE, EVENT_SOURCE_CLEAN_IN_THIS_EXTRACT, DERIVED_FROM_IMMUTABLE_ATTRIBUTES, STATE_AT_INDEX_WITH_PARTIAL_EVIDENCE,
  STATE_AT_INDEX_UNVERIFIABLE}, safe_all_rows included|excluded, reason}`, `safe_all_rows_feature_set`, `excluded_from_safe_all_rows`, `safe_row_rule`,
  `phase2_columns_of_implicated_sources[]`, `requested_trace[] {name, kind, resolution}`.
- `audit/timing_diagnostic.md/.json`: as for `meuhedet-explore`; since v0.5.1 the per-column table also covers at-index date columns of predictor roles
  (`First_Registry_Date`, `MEFI_From_Date`; entry gains `timing`, `same_day_evidence_for`) and the report gains `same_day_evidence_columns`.
- `datasets/<analysis>/`: `ablation_no_<feature>` / `ablation_no_<f1>_no_<f2>` (reference cohort rebuilt with `timing_scope: cohort`, verified row by row and
  column by column against the reference dataset) and `safe_all_rows` (`timing_scope: built_predictors`; `manifest.audit.meuhedet_build` gains
  `timing_scope, timing_guard_columns, cohort_guard_rows_retained, cohort_guard_rows_retained_by_column` and `feature_set.timing {scope, guard_columns, cohort_guard_columns}`).
- `configs/<RUN_LABEL>.yaml` (`MEUHEDET_EFALLS_EXTENDED_180D_ABLATION_NO_*`, `MEUHEDET_EFALLS_SAFE_ALL_ROWS_180D_SENSITIVITY`; the reference EXTENDED config is
  the template, so model, seed, proportions and resampling are identical), `split_audit_<analysis>.md/.json` (as `split_audit` + `reference_test_rows_sha256`),
  `adequacy_<analysis>.md/.json`, `feature_report_<analysis>.md/.csv`, `runs/` (one run per analysis).
- `convergence_audit.md/.json`: per run (reference STRICT / EXTENDED and every new run) `final_fit {converged_at_lambda_star, grid {lambda_star, lambda_max,
  lambda_ratio, n_lambda, lambda_star_index, grid_points_to_end, at_grid_end, near_grid_end, lambda_stop_index, beyond_stata_stop, cv_minimum_identified,
  cv_paths_converged, path_converged, n_selected}, warnings {event: count}, unpenalised_refit}`, `replicates {stability_warnings, optimism_warnings,
  post_fit_warnings, path_not_converged_lambdas_per_event, perfect_predictor_omissions_by_column, replicate_failures, fp_form_instability {variable: {final_form,
  replicates, share_with_final_form, forms}}, resampling, run_warnings}`, `verdict ∈ {VALID, VALID WITH QUALIFICATIONS, QUESTIONABLE - lambda* on the grid boundary,
  NOT ASSESSABLE}`, `qualifications[]`, `solver_or_grid_change_warranted[]`, `warning_meaning`.
- `sensitivity_comparison.json` / `SENSITIVITY_COMPARISON.md`: `records {reference_strict, reference_extended, ablation_*, safe_all_rows: {state, label, run_id,
  n_rows, n_patients, n_events, event_prevalence, n_predictors, predictors, predictors_excluded {name: why}, split {train|validation|test: n_rows, n_patients,
  n_events}, test_rows_sha256, auroc(+_ci), pr_auc(+_ci), brier(+_ci), calibration_slope_uncalibrated(+_ci), calibration_intercept_uncalibrated(+_ci),
  citl_uncalibrated(+_ci), oe_ratio_uncalibrated(+_ci), calibration_slope_recalibrated, citl_recalibrated, lambda_star, selected_design_columns,
  selected_raw_features, stability_top, convergence_verdict, final_fit_warnings, replicate_warnings, timing_related_exclusion {n_rows_excluded, columns,
  cohort_guard_rows_retained, text}, run_warnings}}`, `skipped`, `feature_set_effect[] {a, b, same_test_rows, auroc_a, auroc_b, delta}` (paired, hash-verified),
  `population_effect {text, rows, partner, same_predictors, paired}` (unpaired), `final_dwh_fixed {status NOT RUN ..., population, predictors, requirement,
  answers}`, `mapping`, `provenance`.
- `REAL_DATA_EXPLORATORY_REPORT.md`: the plain-language report (14 numbered sections: population, prediction date, outcome, D-00, why rows were excluded,
  leakage, STRICT, EXTENDED, SAFE-ALL-ROWS + ablations, features that matter predictively, warnings, justified / not-yet-justified conclusions, next steps
  with DWH access); `SENSITIVITY_README.md`.
- Never share: `datasets/`, `id_pepper.txt`, `runs/*/predictions_*.parquet`.

## `meuhedet-eda` results directory (v0.6.0)

Aggregate only (no identifier, pseudonym, pepper or row; counts 1..min_cell-1 read `<min_cell`). Every CSV starts with one watermark line
(`<watermark> | basis: <BASIS> | ...`; read with `skiprows=1`). Bases: FULL_EXTRACT (every row, no outcome), ELIGIBLE_LABELLED, MODELLING_COHORT
(no outcome), TRAIN_ONLY (target-aware; the only basis allowed to inform modelling decisions), POSTHOC_DESCRIPTIVE_FULL_COHORT (outcome-stratified
incl. test rows; never for model choice), SPLIT_DIAGNOSTIC, DEFINITIONS.

| file | basis | content |
|---|---|---|
| `REAL_DATA_EDA_REPORT.html` | all | self-contained report (sections 1-14, filterable tables, embedded figures) |
| `REAL_DATA_EDA_SUMMARY.md` | all | plain-language summary: contents, good, poor quality, univariate signal (TRAIN), missing, next |
| `DATA_QUALITY_REPORT.md` | FULL_EXTRACT | every data-quality finding by severity, checks passed / not applicable |
| `eda_manifest.json` | - | input sha256, definition hashes, stages, split (test hash; `matches_reference`), file bases + sha256, privacy scan |
| `eda/data_dictionary.csv` | DEFINITIONS | column, types, domain, source, meaning + status, role, timing, availability at prediction time, STRICT/EXTENDED use |
| `eda/column_profile.csv` | FULL_EXTRACT | counts, missing % per basis, uniqueness, constant status, top values, quantiles, dates, sentinels, parse failures |
| `eda/numeric_profile.csv`, `eda/categorical_profile.csv` | FULL_EXTRACT + MODELLING_COHORT | distributions, zero inflation, skewness, extremes, transformations to investigate; level counts |
| `eda/missingness*.csv` | see basis column | classes (true absence / not measured / source unavailable / not applicable / technical failure / sentinel / unknown), by domain, co-missingness, patterns, outcome association (TRAIN) |
| `eda/data_quality_checks.csv`, `eda/data_quality_findings.csv` | FULL_EXTRACT | check_id, category, severity, status, columns, n_rows, detail, recommendation |
| `eda/cohort_flow.csv`, `eda/population_profile.csv`, `eda/outcome_prevalence.csv` | as stated per row | funnel (+ SAFE-ALL-ROWS branch; cross-checked with the build), population, prevalence with Wilson CIs |
| `eda/numeric_outcome_train.csv`, `eda/categorical_outcome_train.csv`, `eda/univariate_association_train.csv`, `eda/correlation_pairs.csv`, `eda/redundancy_clusters.csv`, `eda/table1_train.csv` | TRAIN_ONLY | target-aware tables (no p-values) |
| `eda/table1.csv` | POSTHOC_DESCRIPTIVE | Table 1 of the modelling population by future fall, with SMD |
| `eda/temporal_audit.csv`, `eda/feature_provenance.csv`, `eda/timing_diagnostic.json` | FULL_EXTRACT / DEFINITIONS | every date column vs Index_Date, sources without record dates, leakage-risk class; source -> column -> feature -> model |
| `eda/current_15_feature_dictionary.csv`, `eda/phase2_candidate_features.csv` | DEFINITIONS + MODELLING_COHORT (+ TRAIN) | current predictors (with reference coefficients when `--reference`); Phase-2 catalogue (not trained) |
| `eda/split_balance.csv` | SPLIT_DIAGNOSTIC | partitions compared, SMD vs train, imbalance flag |

## `model-report` results directory (v0.6.0)

Written by `falls_ml model-report` and at the end of `meuhedet-explore` / `meuhedet-sensitivity` (`<out>/reports`). Never written into a run.

| file | content |
|---|---|
| `MODEL_TRAINING_REPORT_<ANALYSIS>.html` | per run: cohort, LASSO CV curve / coefficient paths / FP shapes (TRAIN), learning curve (TRAIN / VALIDATION), discrimination, calibration, thresholds, lift, features, subgroups, error analysis, uncertainty (HELD-OUT TEST, model frozen) |
| `MODEL_COMPARISON_REPORT.html` | all analyses side by side; paired bootstrap differences for identical test rows only; NOT A DIRECT PAIRED MODEL COMPARISON otherwise |
| `MANAGEMENT_MODEL_REPORT_HE.html`, `MANAGEMENT_MODEL_SUMMARY_HE.md`, `EXECUTIVE_ONE_PAGER_HE.html` | Hebrew management reports; conclusions generated from the artifacts |
| `figures/*.png` + `*.svg`, `FIGURE_INDEX.md` | every figure at 200 dpi and as vector; `he_*` = Hebrew management versions |
| `tables/<ANALYSIS>_{coefficient_path,learning_curve,thresholds_test,lift_test,risk_deciles_test,subgroups_test,feature_dictionary}.csv`, `tables/model_comparison.csv`, `tables/paired_differences.csv` | the numbers behind the figures |
| `model_report_manifest.json` | runs, primary analysis (pre-declared order, never chosen by test), test-set rule, coefficient-path reproduction check, privacy scan |

## `meuhedet-d00` results directory (v0.7.0; `--resume` v0.7.1)

`<out>/share/` (the only folder to share; every CSV starts with a watermark line - `pd.read_csv(path, skiprows=1)`):

| file | content |
|---|---|
| `D00_FEATURE_DEPENDENCY.md/.csv/.json` | per feature: source columns (builder inputs), validation columns, domain, source, intermediate VIEW aggregate, record-date fields, same-day possibility, STRICT/EXTENDED membership, D-00 status in FULL_LABELED and D00_CLEAN, explanation, code/config evidence, design columns; per predictor-role column; derived feature sets; forbidden columns |
| `ANALYSIS_PLAN.json` | the frozen plan (cells, cohorts, predictor lists, aliases, reused reference runs), `plan_sha256`; contains no result |
| `D00_SENSITIVITY_REPORT.html` | technical report (English): D-00 problem, graph, sets, matrix, paired feature-set effects, unpaired population effects, risk concentration, learning curves, warnings, DWH requirements |
| `D00_SENSITIVITY_SUMMARY_HE.md` | plain-language Hebrew answers to the twelve questions |
| `LASSO_WARNINGS_AUDIT.md/.json` | final fits vs bootstrap replicates vs learning-curve refits; lambda sensitivity on VALIDATION; materiality |
| `d00_sensitivity.json` | machine-readable summary (cells, comparisons, risk concentration, learning curve, feature statuses) |
| `tables/sensitivity_matrix.csv` | one row per cell (status, serving run, rows, events, test AUROC / PR-AUC / Brier / calibration with CIs, paired flag) |
| `tables/ablation_paired.csv`, `feature_set_effects_paired.csv` | paired differences (compared minus reference) with bootstrap CIs, each model's calibration and top-x% capture |
| `tables/population_effects_unpaired.csv`, `population_d00_decomposition.csv` | same predictors on the two cohorts (NOT paired); FULL_LABELED models on the D00-clean vs D-00 subsets of their own test rows |
| `tables/risk_concentration.csv` | top 1/2/5/10/20%: N, % population, falls, % captured, prevalence, lift, bootstrap CIs (small cells suppressed) |
| `tables/learning_curve_<M>.csv`, `learning_curve_increments_<M>.csv`, `d00_column_dependency.csv` | learning curve with per-refit diagnostics; paired increments |
| `figures/d00_*.png/.svg`, `split/`, `adequacy/`, `management/` (model-report folder with the D-00 section), `reference_integrity.json`, `d00_manifest.json` | |

`<out>/datasets/`, `<out>/runs/` (row-level predictions), `<out>/configs/` and `<out>/id_pepper.txt` stay local. Run names:
`MEUHEDET_EFALLS_D00_<CELL>_180D_SENSITIVITY`.

**v0.7.1 `--resume`**: `share/RESUME_LOG.json` = `resumed, plan_sha256, falls_ml_version_now, source_tree_sha256_now, cells` (per run cell:
`action` REUSED_COMPLETED_RUN | REFITTED (...) | RUN (not reached ...), `run_id`, `incomplete_runs_ignored`, for reused runs
`source_tree_sha256_of_run` and `bootstrap_convergence_retry_policy_in_effect`, for refitted runs `earlier_failures` and the retry summary),
`incomplete_attempts` (`run_id, cell, test_set_released, failure {error, bootstrap_replicate_failed_events, ...}, handling`),
`completed_runs_not_in_plan`, `earlier_attempt_files_unchanged`, `integrity` (`n_files_checked, changed_files, registry_only_appended,
registry_entries_added, unchanged`). `reference_integrity.json` gains `earlier_attempt`; `d00_manifest.json` gains `resume`; cell status
`REUSED (completed in an earlier attempt of this D-00 run; not refitted, test set not re-evaluated)`. `LASSO_WARNINGS_AUDIT.json`:
`new_runs` covers every sensitivity run with its `origin`; `new_run_replicates` (per cell: replicates, `convergence_retry_policy_in_effect`,
`convergence_retries` {component: {outcome: n}}, `replicate_failures`).

## Bootstrap convergence retry (v0.7.1, every run)

`bootstrap_convergence_retries.csv` (one row per replicate fit whose full-data LASSO solve did not converge at the configured limit; empty when
none): `component` (stability | optimism), `replicate`, `fit_seed`, `policy_version`, `outcome` (CONVERGED_ON_RETRY | FAILED_ON_RETRY |
REJECTED_ON_RETRY | NOT_RETRIED), `accepted`, `tol`, `initial_max_iter`, `retry_max_iter` (10 x initial), `lambda_star`, `lambda_star_index`,
`retry_lambda_star`, `retry_lambda_star_index`, `retry_converged_at_lambda_star`, `initial_error`, `retry_error`. `metrics.json` ->
`bootstrap_convergence_retry` = `policy` (text of the pre-declared rule), `n_initial_convergence_failures`, `n_converged_on_retry`,
`n_still_failed`, `by_component`, `records`. A replicate that still fails is recorded in the stability / optimism failures as before (reason
text ends with `[convergence retry at full_path_max_iter=...: FAILED_ON_RETRY - ...]`). LASSO parameter `full_path_max_iter` (default None =
`max_iter`) and diagnostics `lambda_star_converged`, `full_path_max_iter`, `tol`.

## Frozen baselines (`baselines/<NAME>/baseline.json`)

Written once by `falls_ml freeze-baseline` (the directory is immutable): `baseline_name, frozen_utc, note, run_id, run_dir, experiment, dataset` (version, mapping version, source, data sha256, rows, patients, index dates, prevalence, freeze date, scientific-use flag, feature-spec sha256), `index_date, feature_spec, mapping_manifest, wide_contract, outcome, model_config` (model, preprocessing, validation, calibration), `lasso, selected_lambda, coefficients` (rows of `coefficients.csv`), `metrics_test, calibration, split` (incl. `test_rows_sha256`), `efalls_coverage, limitations, warnings, code_version, files` (sha256 of the copied run artifacts: config, metrics, coefficients, dataset manifest, importance/stability/FP/CV/refit tables, splits, report.md).

## Plots (`plots/`)

`roc.png`, `precision_recall.png`, `calibration.png` (served variant), `calibration_before_after.png` (uncalibrated vs recalibrated, when a recalibration was fitted), `decision_curve.png`, `risk_distribution.png`, `feature_importance.png`, `coefficients.png` (linear), `selection_stability.png`, `hyperparameter_search.png` (if tuning), `lasso_cv_path.png` (LASSO).

## `model/` bundle

| File | Content |
|---|---|
| `bundle.json` | Model name/version, experiment, `served_variant` (must agree with the presence of a calibrator), config dict + `config_sha256`, training dataset manifest, feature spec name/version/sha256, mapping version, preprocessor fingerprint, selected design columns, category levels, risk-category config (with `approved`), calibration params, code version, library versions, created UTC, `files`: SHA-256 of every other file in `model/` (keys are `/`-separated relative paths). Unlisted file-manager metadata (`desktop.ini`, `Thumbs.db`, `.DS_Store`, case-insensitive, any depth) is ignored with a warning because it is never read; any other unlisted file (e.g. a OneDrive conflict copy `calibrator-<COMPUTER>.json`) fails verification. Not signed: `load_bundle` verifies file hashes before unpickling, that the stored config reproduces `config_sha256`, and that the risk-category block equals the one derived from that config |
| `feature_spec.json` | Feature spec name, version, sha256, source paths, predictor names, category levels, and `payloads` (full base and extension spec content plus subset list). `load_bundle` rebuilds the spec from the payloads and verifies its sha256; it never reads YAML, so bundles are self-contained. Subset digests are root-based (`sha256({"base": <root spec sha256>, "subset": [names in spec order]})`): a subset of a subset equals the direct subset, and a "subset" keeping every predictor is the full spec itself. `bundle.json` `extra_metadata.efalls_coverage` repeats the coverage summary for eFalls kinds. A reduced-set bundle scores rows with exactly its predictor columns (extra predictor columns are undeclared and refused) |
| `preprocessor.pkl` | Fitted `Preprocessor` |
| `adapter/` | Adapter-specific files written by `ModelAdapter.save` |
| `calibrator.json` | Recalibration method and parameters when the served variant is recalibrated; otherwise method `none` |
| `reference_profile.json` | Training reference distributions for monitoring and contribution references (`design_column_means`); `pipeline` {model, preprocessor_fingerprint}; `reference_performance` {auroc, calibration_slope, citl, oe_ratio, prevalence, split, variant} of the served variant on validation, used for AUROC-drop and prevalence-change rules |

## Master comparison outputs (`reports/`)

| File | Content |
|---|---|
| `comparison.csv|md|html` | One row per completed run (`RUN_COMPLETE.json` required; other directories are listed as skipped with a reason). Non-published runs contribute their served variant; published scoring contributes `lp_c_box_s3_1` (primary) and `lp_a_table_s3_2` (co-reported). Columns include `category` (`scientific_reproduction`, `local_retraining`, `reduced_local_retraining` for `efalls_retrained_reduced` (own section "Local retraining on a reduced eFalls predictor set (not a full eFalls reproduction)"; `is_efalls_retrained` false), `experimental_algorithm`, `meuhedet_enhanced` for alternative models with `pure_efalls` false, `ablation`), `efalls_coverage` (`"X/78"` for eFalls kinds, empty otherwise; column "eFalls coverage" in the Markdown/HTML tables), `feature_set_name`, `pure_efalls`, `data_sha256`, AUROC, PR_AUC, Brier, calibration_slope, calibration_intercept, sensitivity/specificity at threshold, number_of_features (raw predictors with non-zero weight; all predictors for tree models), best_params, train_time (total), inference_time. Runs are comparable only with identical `test_rows_sha256` **and** `data_sha256`. Ablation members never enter highlights or the recommendation |
| `comparison_paired_differences.csv` | Paired patient-level bootstrap differences vs the published eFalls primary variant on identical test rows: `run_id, variant, reference_variant, metric (auroc|brier|calibration_slope|citl|net_benefit@t), estimate, ci_low, ci_high, n_bootstrap` |
| `best_features.csv|md` | Feature evidence; Meuhedet incremental value read from `--ablation-dir` (default `reports/ablation`, a missing summary is a visible warning); published coefficients read from the published run's own `model.published_equation.config`. Local eFalls LASSO evidence (coefficients, stability) comes only from `efalls_retrained`; an `efalls_retrained_reduced` run contributes permutation importance as a local model and is flagged with a warning |
| `ablation/ablation_results.csv|ablation.md` | One row per ablation step (`step`, `model`, run, test metrics, paired bootstrap Δ vs the baseline step); `ablation.md` states the baseline model, run, feature set, dataset version, data sha and test-row sha |

## Phase 2 (`falls_ml meuhedet-phase2`, v0.8.1 FINAL)

Everything below lives in `--out`. Only `share\` is for review; `stages\` and `work\` hold row-level files and never leave the computer.

| Path | Content |
|---|---|
| `PHASE2_PLAN.json` | frozen plan: config, catalogue / contract / dictionary / mapping / D-00 config hashes, `final_experiment_config_sha256`, `frozen_production_config`, input sha256, reference folder name, threads, seed; `plan_sha256`; `code_sha256_at_start` |
| `FINAL_EXPERIMENT_CONFIG.json` / `.sha256` | the effective configuration of this run (byte copy of `configs/meuhedet/FINAL_EXPERIMENT_CONFIG.json` for the production run), read-only, verified on every resume |
| `RUN_STATE.json` | progress summary: stage, substage, completed stages, current item, dataset / feature-registry / config / final-config / code hashes, seed, started_at, attempt_started_at, last_checkpoint, status (RUNNING / COMPLETE / STOPPED / INVESTIGATION / FAILED), resume_count, attempt, threads, pid (read by `meuhedet-phase2-status`) |
| `RESUME_AUDIT.json` | every attempt: start / end, resumed, code sha256, accepted gates with reasons, `--accept-code-change` reason, status, stop gate |
| `STAGE_<nn>_COMPLETE.json` | stage record: sha256 of every stage output, sha256 of the upstream stage records (hash chain), items, elapsed, code / plan sha256 |
| `stages\S<nn>_<name>\items\<item>\` + `<item>.COMPLETE.json` | item outputs and their commit record (files -> sha256, small JSON result, seed, code / plan sha256, attempt, elapsed) |
| `stages\S<nn>_<name>\_incomplete\attempt-<n>\` | work interrupted by a crash, moved aside, never read |
| `checkpoints\xgb_optuna.sqlite`, `xgb_trials.jsonl`, `_previous\` | Optuna storage of this attempt (rebuilt from the committed trials), the append-only trial ledger, earlier storages |
| `logs\events.jsonl`, `logs\gates.jsonl` | append-only event log (ts, attempt, stage, item, action, status, duration, metrics, error) and gate decisions |
| `RUN_TIMINGS.csv` | stage, attempt, n_items, total_s, max_item_s, total_min (from the commit records) |
| `SELECTION_FROZEN.json` | selection outcome, recommended SAFE configuration, shortlist, discovery reference (LASSO:SAFE_BASE), benchmark (LASSO:BASELINE_15 refit) + point-estimate class, attribution contrasts, qualifying candidates, every configuration to be scored, reason, hashes; written before VALIDATION is read |
| `VALIDATION_OPENED.json` | write-once record of the one-time opening of VALIDATION: timestamp, frozen-selection sha256, final-config sha256, plan / code / feature-set sha256, confirmatory finalists with roles, descriptive configurations |
| `validation_evaluation_registry.jsonl` | the one-time opening of VALIDATION (selection sha256; re-reading after a crash only for the same selection) |
| `artifacts\01_COLUMN_REGISTRY.csv` | one row per contract column: raw_column, domain, source_system, data_type, business_meaning (+ status), missing_pct, unique_count, constant_status / near_constant, source_timestamp_field, availability_at_prediction_time, provenance_status (+ reason, index-day / later row counts), leakage_reason, used_in_extended_15, efalls_concept, mapping_class, proposed_engineered_features, final_discovery_eligibility (+ reason) |
| `artifacts\02_ENGINEERED_FEATURE_REGISTRY.csv` | one row per engineered feature: domain, inputs, sources, derivation, kind, linear encoding, missing semantics, assessment form, mapping class, eFalls concept, look-back, availability delay, meaning status, provenance status, timing evidence / bound columns / assumption, TRAIN observed / missing / unique, eligibility (+ reason) |
| `artifacts\03_UNIVARIATE_SCREEN.csv` | TRAIN only: n observed, missing %, distribution, outcome rates, prevalence ratio (CI), SMD, univariate AUROC (CI), sparse / separation flags (cells < 10 suppressed) |
| `artifacts\04_REDUNDANCY_CLUSTERS.csv` / `05_FEATURE_QUALITY.csv` | representative-anchored |Spearman| >= 0.90 clusters (TRAIN, label-blind) per universe (SAFE / exploratory) / per-feature quality, representatives and pooled-set membership per universe |
| `artifacts\FEATURE_SETS.json` | the frozen sets (features, baseline, category, kind), aliases (within a category), the label-blind design spec, the SAFE waterfall steps, SAFE_BASE, baseline provenance |
| `artifacts\COHORT_FACTS.json` | partitions, hashes, TEST dropped (count + sha256 only), `label_death_audit` (TRAIN+VALIDATION deaths inside the window, events among them, label reasons; small cells suppressed) |
| `artifacts\FITS_LASSO.csv`, `FITS_ENET.csv`, `FITS_XGB.csv`, `XGB_TRIALS.csv` | every fit: λ, l1_ratio, grid-boundary flag, CV minimum identified, design columns, duplicates dropped, selected count, retries, elapsed / every Optuna trial |
| `artifacts\OOF_MODEL_COMPARISON.csv` | TRAIN outer-OOF metrics of every configuration (category column), Δ vs the BASELINE_15 refit with paired-bootstrap CIs, Δ vs SAFE_BASE for SAFE configurations, qualification flag |
| `artifacts\ATTRIBUTION_CONTRASTS_OOF.csv` / `_VALIDATION.csv` / `ATTRIBUTION_CONTRASTS.csv` | FEATURE_EXPANSION and MODEL_FAMILY_PIPELINE paired contrasts (OOF, VALIDATION, merged) |
| `artifacts\VALIDATION_MODEL_COMPARISON.csv`, `VALIDATION_CONFIRMATION.json` | validation metrics (role: benchmark / discovery_reference / recommended / shortlist / descriptive; category), Δ with CIs incl. recorded events identified per capacity; the pre-declared confirmation status and point-estimate benchmark classes |
| `artifacts\CALIBRATION_BOOTSTRAP.csv` / `CALIBRATION_SMOOTH.csv` | calibration slope / intercept / CITL / O:E / Brier / scaled Brier with bootstrap intervals; smooth calibration curve (restricted cubic spline, 95% band) for the shortlist and the management waterfall |
| `artifacts\STABILITY.csv`, `ABLATION_RESULTS.csv`, `PERMUTATION_IMPORTANCE.csv`, `SHAP_SUMMARY.csv`, `SHAP_INTERACTIONS.csv`, `FEATURE_CONSENSUS.csv` (SAFE features), `EXPLORATORY_UNRESOLVED_FEATURES.csv` (UNRESOLVED features) | as named (plan §7) |
| `artifacts\MODEL_COMPARISON.csv`, `DOMAIN_INCREMENTAL_GAIN.csv` (management waterfall with features added, one-domain additions, leave-one-domain-out with features removed; every capacity: flagged, events identified, capture, PPV, false alerts, additional events vs the BASELINE_15 refit with CIs), `COLUMN_FUNNEL.csv` (exact accounting per category), `FEATURE_SHORTLIST.csv`, `CALIBRATION_SUMMARY.csv`, `MANAGEMENT_HEADLINE.json`, `OPERATIONAL_CAPACITY.csv`, `THRESHOLD_RESULTS.csv`, `CALIBRATION_CURVES.csv`, `SUBGROUP_SUMMARY.csv`, `SCIENTIFIC_SUMMARY.md`, `MANAGEMENT_SUMMARY_HE.md` | report tables and summaries |
| `figures\` | 01 funnel, 02 management waterfall (same VALIDATION rows), 03 AUROC, 04 AP, 05 calibration (smooth + deciles), 06 consensus, 07 capture, 08 false alerts at matched capacity, 09 model families, 10 XGBoost SHAP summary, 11 additional events at the same capacity (PNG + SVG) |
| `share\` | README, RUN_MANIFEST, RUN_STATE_FINAL, RESUME_AUDIT, RUN_TIMINGS, PRIVACY_SCAN, the user-specified tables (counts < 10 suppressed), `tables\`, `figures\`, `reviews\` |
