"""Single public entry point: ``run_experiment`` (architecture §4).

``run_experiment`` only orchestrates small tested components. The same function runs the published eFalls
equation (Experiment A), the locally retrained eFalls LASSO (Experiment B) and every alternative model;
adding an algorithm means writing an adapter, not changing this module.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from falls_ml.artifacts import (RunDirectory, code_version, create_run_directory, definition_file_sha256, environment_info, utc_now,
                                write_csv, write_json, write_parquet, write_yaml)
from falls_ml.config import (EFALLS_KINDS, REDUCED_KIND, ExperimentConfig, check_feature_declaration, experiment_config_from_dict,
                             load_experiment_config)
from falls_ml.data.dataset import ModelingDataset
from falls_ml.errors import ConfigError, DatasetValidationError, LeakageError
from falls_ml.features.spec import FeatureSpec, load_feature_spec
from falls_ml.logging_utils import get_logger, run_log
from falls_ml.paths import portable_path, project_relative_posix, resolve_path
from falls_ml.pipeline import FittedPipeline, fit_pipeline, model_params_from_config, row_keys
from falls_ml.seeding import int_seed_for

log = get_logger(__name__)

PUBLISHED_PRIMARY = "lp_c_box_s3_1"
PUBLISHED_CO_REPORTED = "lp_a_table_s3_2"
PUBLISHED_VARIANTS = ("lp_c_box_s3_1", "lp_a_table_s3_2", "lp_b_label_swap", "lp_d2_numeric_swap")
METRIC_NAMES = ("auroc", "pr_auc", "brier", "calibration_slope", "calibration_intercept", "citl", "oe_ratio")
REGISTRY_FILE = "test_evaluation_registry.jsonl"
COMPLETE_MARKER = "RUN_COMPLETE.json"
#: dataset column each configured subgroup variable is derived from (preflight requires only these)
SUBGROUP_SOURCE_COLUMNS = {"sex": "sex", "bmi_category": "bmi_value", "age_band": "age_years"}
REDUCED_WARNING = "REDUCED eFalls predictor set – NOT a full eFalls reproduction"
EXPLORATORY_OUTCOME_WARNING = "EXPLORATORY OUTCOME – NOT an eFalls reproduction"


@dataclass(frozen=True)
class ExperimentResult:
    run_id: str
    run_dir: Path
    metrics: dict[str, Any]
    config: ExperimentConfig


# ============================================================================ public API
def run_experiment(
    model_name: str | None,
    dataset: ModelingDataset | str | Path | None,
    experiment_config: ExperimentConfig | str | Path,
    *,
    runs_dir: str | Path | None = None,
    created_utc: str | None = None,
    purpose: str = "train",
    allow_test_reevaluation: bool = False,
) -> ExperimentResult:
    """Run one experiment end to end and return the run directory and metrics.

    Args:
        model_name: registry model name; ``None`` uses ``experiment_config.model.name``. A different name creates a
            separately named experiment ``<name>__<model>`` using that adapter's representation (config checks re-run).
        dataset: a loaded ``ModelingDataset`` or a dataset directory; ``None`` uses ``experiment_config.dataset.path``.
        experiment_config: config object or YAML path.
        runs_dir: output root (default ``experiment_config.output.runs_dir``).
        created_utc: fixed timestamp (tests); default now.
        purpose: ``train`` or ``reproduce`` (recorded in the test-evaluation registry, D-19 §4).
        allow_test_reevaluation: explicitly allow re-evaluating the same test rows with an already-evaluated config
            on a scientific dataset (recorded).
    """
    if purpose not in {"train", "reproduce"}:
        raise ConfigError(f"purpose must be 'train' or 'reproduce', got {purpose!r}")
    config = _resolve_config(experiment_config, model_name)
    anchor = config.source_path
    spec = load_feature_spec(config.dataset.feature_spec, config.dataset.feature_spec_extensions, anchor=anchor)
    check_feature_declaration(config, spec)  # preflight, part 1: spec-level feature rules before any data are read
    ds = _resolve_dataset(dataset, config, spec)
    _preflight(config, ds)
    run = create_run_directory(Path(runs_dir or config.output.runs_dir), config.experiment.name, config.model.name,
                               config_sha256=config.sha256(), dataset_sha256=ds.manifest.data_sha256, created_utc=created_utc)
    with run_log(run.file("run_log.jsonl")):
        log.info("run_started", extra_fields={"run_id": run.run_id, "experiment": config.experiment.name, "model": config.model.name})
        try:
            metrics = _Orchestrator(config, spec, ds, run, purpose=purpose, allow_test_reevaluation=allow_test_reevaluation).execute()
        except Exception:
            log.error("run_failed", extra_fields={"run_id": run.run_id}, exc_info=True)
            raise
        log.info("run_finished", extra_fields={"run_id": run.run_id})
    return ExperimentResult(run_id=run.run_id, run_dir=run.path, metrics=metrics, config=config)


# ============================================================================ helpers
def _resolve_config(experiment_config: ExperimentConfig | str | Path, model_name: str | None) -> ExperimentConfig:
    config = experiment_config if isinstance(experiment_config, ExperimentConfig) else load_experiment_config(experiment_config)
    if model_name is None or model_name == config.model.name:
        return config
    from falls_ml.models.registry import get_adapter_class

    adapter = get_adapter_class(model_name)
    raw = config.to_dict()
    raw["experiment"]["name"] = f"{config.experiment.name}__{model_name}"
    raw["model"] = {"name": model_name, "params": {}, "published_equation": raw["model"].get("published_equation")
                    if model_name == "efalls_published" else None}
    raw["preprocessing"]["representation"] = adapter.representation
    if config.experiment.kind in EFALLS_KINDS and model_name not in {"efalls_published", "lasso_logistic_cv"}:
        raw["experiment"]["kind"] = "alternative_model"
        raw["experiment"]["layers"] = sorted(set(raw["experiment"]["layers"]) | {"L4_alternative"})
    return dataclasses.replace(experiment_config_from_dict(raw, name=f"override:{model_name}"), source_path=config.source_path)


def _resolve_dataset(dataset: ModelingDataset | str | Path | None, config: ExperimentConfig, spec: FeatureSpec) -> ModelingDataset:
    """Load (or check) the dataset. With ``preprocessing.features`` the dataset may be built with the full spec or with
    exactly the declared subset spec; the manifest ``feature_spec_sha256`` decides (anything else fails loudly)."""
    features = config.preprocessing.features
    if isinstance(dataset, ModelingDataset):
        allowed = {spec.content_sha256: "full feature spec"}
        if features is not None:
            allowed.setdefault(spec.subset(features).content_sha256, f"declared subset ({len(features)} predictors)")
        if dataset.spec.content_sha256 not in allowed:
            raise DatasetValidationError("Dataset was loaded with a different feature spec than the experiment config",
                                         [f"dataset feature spec sha256 {dataset.spec.content_sha256}",
                                          *[f"expected {label}: {sha}" for sha, label in allowed.items()]])
        return dataset
    if dataset is not None:
        return ModelingDataset.load(Path(dataset), spec, features=features)
    if config.dataset.path is None:
        raise ConfigError("No dataset given (argument or experiment_config.dataset.path)")
    return ModelingDataset.load(resolve_path(config.dataset.path, anchor=config.source_path), spec, features=features)


def required_dataset_columns(config: ExperimentConfig) -> set[str]:
    """Non-predictor columns a run reads: bootstrap/IECV/heterogeneity clusters, subgroup sources, ``sex`` for published scoring."""
    needed = {config.evaluation.bootstrap.cluster_column, *(SUBGROUP_SOURCE_COLUMNS[v] for v in config.evaluation.subgroups)}
    if config.analysis.iecv.enabled:
        needed.add(config.analysis.iecv.cluster_column)
    if config.evaluation.cluster_column_for_heterogeneity:
        needed.add(config.evaluation.cluster_column_for_heterogeneity)
    if config.experiment.kind == "efalls_published_scoring":
        needed.add("sex")  # sex-specific calibration of the published equation (D-01)
    return needed


def _preflight(config: ExperimentConfig, ds: ModelingDataset) -> None:
    """Checks that must pass before a run directory is created (fail fast, no orphan runs).

    Part 1 (``check_feature_declaration``: reduced eFalls feature rules) runs in ``run_experiment`` before the dataset is read.
    """
    m = ds.manifest
    if config.dataset.require_scientific_use and not m.scientific_use_allowed:
        raise DatasetValidationError(f"Dataset {m.dataset_version} is not approved for scientific use (source={m.source}); "
                                     "set require_scientific_use=false only for software tests")
    frame_cols = set(ds.frame.columns)
    missing = sorted(c for c in required_dataset_columns(config) if c not in frame_cols)
    if missing:
        raise ConfigError(f"configured columns not present in the dataset: {missing} (bootstrap/IECV/heterogeneity cluster columns; "
                          f"evaluation.subgroups sources {SUBGROUP_SOURCE_COLUMNS}); remove the subgroup or cluster setting "
                          "or provide the column")
    # D-19 §3 label maturity: every outcome window must be closed (plus lag) at the data freeze date.
    last_window_end = ds.spec.outcome.window_end(pd.Timestamp(ds.frame[ds.spec.index_column].max()))
    required_freeze = last_window_end + pd.Timedelta(days=config.validation.outcome_lag_days)
    if m.data_freeze_date is None:
        if m.scientific_use_allowed:
            raise DatasetValidationError("scientific datasets must record data_freeze_date (D-19 §3 label maturity)")
    elif pd.Timestamp(m.data_freeze_date) < required_freeze:
        raise DatasetValidationError(f"outcome windows are not mature: last window ends {last_window_end.date()} (+{config.validation.outcome_lag_days}d lag) "
                                     f"but data were frozen on {m.data_freeze_date} (D-19 §3)")


def _estimate(v: dict[str, Any] | None) -> float | None:
    return None if v is None else v.get("estimate")


def _age_band(age: pd.Series) -> pd.Series:
    return pd.cut(age, bins=[0, 75, 85, np.inf], right=False, labels=["65-74", "75-84", "85+"]).astype("string")


def _retry_summary(retries: list[dict[str, Any]]) -> dict[str, Any]:
    from falls_ml.evaluation.convergence_retry import retry_summary

    return retry_summary(retries)


def _read_complete_metrics(run_dir: Path) -> dict[str, Any] | None:
    """metrics.json of a completed run, or None (incomplete/unreadable runs are skipped with a warning)."""
    if not (run_dir / COMPLETE_MARKER).exists():
        return None
    try:
        return json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("unreadable_sibling_run", extra_fields={"run_dir": str(run_dir), "error": str(exc)})
        return None


class _Orchestrator:
    """Executes the run steps; kept as one class so state flows explicitly between steps."""

    def __init__(self, config: ExperimentConfig, spec: FeatureSpec, dataset: ModelingDataset, run: RunDirectory, *,
                 purpose: str, allow_test_reevaluation: bool):
        self.config, self.spec, self.dataset, self.run = config, spec, dataset, run
        self.purpose, self.allow_test_reevaluation = purpose, allow_test_reevaluation
        self.seed = config.validation.seed
        self.full_spec = spec
        self.id_col = spec.identifier_columns[0]
        self.y_col = spec.outcome.name
        self.warnings: list[str] = []
        self.is_published = config.experiment.kind == "efalls_published_scoring"
        self.serves_recalibrated = config.calibration.serves_recalibrated(config.experiment.kind)
        self.frozen_params: dict[str, Any] = model_params_from_config(config)
        self.convergence_retries: list[dict[str, Any]] = []   # bootstrap convergence-retry records (stability + optimism)

    # ------------------------------------------------------------------ main
    def execute(self) -> dict[str, Any]:
        from falls_ml.splitting import TestSetGuard, make_split

        cfg, run = self.config, self.run
        write_yaml(run.file("config.yaml"), cfg.to_dict())
        directory = self.dataset.directory
        write_json(run.file("dataset_manifest.json"), {**self.dataset.manifest.to_dict(),
                                                        "dataset_path_used": str(directory) if directory else None,
                                                        "dataset_path_relative": project_relative_posix(directory, run.path) if directory else None})
        write_json(run.file("run_inputs.json"), self._run_inputs())
        write_json(run.file("environment.json"), environment_info())

        frame = self.dataset.frame
        if cfg.preprocessing.features is not None:
            self.spec = self.spec.subset(cfg.preprocessing.features)
        coverage = self._efalls_coverage()
        if cfg.experiment.kind == REDUCED_KIND and coverage["mandatory_unavailable"]:
            self.warnings.append(f"{REDUCED_WARNING}: mandatory eFalls predictors unavailable (cohort.mandatory_for_efalls_label): "
                                 f"{coverage['mandatory_unavailable']}")
        if not self.spec.outcome.is_published_efalls:
            self.warnings.append(self._exploratory_outcome_note())
        plan = make_split(frame, self.spec, cfg.validation)
        write_csv(run.file("splits.csv"), plan.assignments(frame, self.spec))
        train_df = frame.iloc[plan.train_idx].reset_index(drop=True)
        val_df = frame.iloc[plan.validation_idx].reset_index(drop=True)
        test_keys = row_keys(frame.iloc[plan.test_idx], self.spec)
        guard = TestSetGuard(frame.iloc[plan.test_idx].reset_index(drop=True))
        self._check_disjoint_rows(row_keys(train_df, self.spec), row_keys(val_df, self.spec), test_keys)
        self.forbidden_for_fit = row_keys(val_df, self.spec) | test_keys
        split_info = {"strategy": plan.strategy, "description": plan.description, "limitation_note": plan.limitation_note,
                      "seed": self.seed, "test_rows_sha256": plan.test_rows_sha256(frame, self.spec), **plan.metadata}
        prior_evaluations = self._check_test_registry(split_info["test_rows_sha256"])  # fail before any fitting

        # hyperparameter search (train -> validation only)
        t0 = time.perf_counter()
        best_params, tuning_df = self._tune(train_df, val_df)
        tuning_seconds = time.perf_counter() - t0
        write_csv(run.file("hyperparameter_results.csv"), tuning_df)
        if best_params is not None:
            self.frozen_params = best_params

        # final fit on training data only
        t1 = time.perf_counter()
        pipeline = fit_pipeline(train_df, self.spec, cfg, random_state=int_seed_for(self.seed, "final_fit"),
                                model_params=self.frozen_params, forbidden_row_keys=self.forbidden_for_fit)
        final_fit_seconds = time.perf_counter() - t1
        self._assert_fitted_on(pipeline, train_df)
        if self.is_published and pipeline.model.params.get("sex_parameterisation") != PUBLISHED_PRIMARY:
            raise ConfigError("published scoring model does not use the primary sex parameterisation (D-01)")

        # calibration assessment + recalibration on validation only
        recalibrators = self._fit_recalibrators(self._prediction_variants(pipeline, val_df), val_df)
        if self.serves_recalibrated:
            pipeline.calibrator = recalibrators[self._base_variant()]

        # feature analysis on training / validation data (never test)
        coef_df, stability, perm_df = self._feature_analysis(pipeline, train_df, val_df)
        internal = self._internal_validation(train_df)
        self._record_convergence_retries()

        # ---- model selection is frozen: record in the registry, then release the test set exactly once
        registry = self._register_test_evaluation(split_info["test_rows_sha256"], prior_evaluations)
        test_df = guard.release("final evaluation after model selection and calibration were frozen")
        t2 = time.perf_counter()
        _ = pipeline.predict_proba(test_df, calibrated=pipeline.calibrator is not None)
        predict_seconds_per_1000 = (time.perf_counter() - t2) / max(len(test_df), 1) * 1000

        performance, tables = {}, {"calibration": [], "confusion": [], "decision": [], "subgroup": []}
        # "train" is APPARENT (in-sample, optimistic) performance, reported for transparency only (user item 6).
        for split_name, df in (("train", train_df), ("validation", val_df), ("test", test_df)):
            performance[split_name] = self._evaluate_split(split_name, pipeline, df, recalibrators, tables)
        if self.is_published:
            # descriptive whole-cohort view of the published equation; recalibrated variants excluded (in-sample)
            performance["full_cohort"] = self._evaluate_split("full_cohort", pipeline, frame, {}, tables)
        internal["heterogeneity"] = self._heterogeneity(pipeline, test_df, recalibrators)
        self._write_predictions(pipeline, val_df, recalibrators, "predictions_validation.parquet")
        self._write_predictions(pipeline, test_df, recalibrators, "predictions_test.parquet")
        write_csv(run.file("calibration.csv"), pd.concat(tables["calibration"], ignore_index=True))
        write_csv(run.file("confusion_matrices.csv"), pd.DataFrame(tables["confusion"]))
        write_csv(run.file("decision_curve.csv"), pd.concat(tables["decision"], ignore_index=True))
        write_csv(run.file("subgroup_metrics.csv"), pd.DataFrame(tables["subgroup"]))

        self._save_bundle(pipeline, train_df, performance)
        timing = {"tuning_seconds": tuning_seconds, "final_fit_seconds": final_fit_seconds,
                  "total_train_seconds": tuning_seconds + final_fit_seconds, "predict_seconds_per_1000_rows": predict_seconds_per_1000}
        metrics = self._assemble_metrics(pipeline, performance, split_info, frame, plan, best_params, timing, coef_df, perm_df,
                                         internal, recalibrators, test_df, train_df, val_df, registry)
        write_json(run.file("metrics.json"), metrics)
        self._plots(pipeline, test_df, recalibrators, coef_df, stability, tuning_df)
        if cfg.reporting.enabled:
            from falls_ml.reporting.report import render_run_report
            render_run_report(run.path)
        write_json(run.file(COMPLETE_MARKER), {"run_id": run.run_id, "completed_utc": utc_now()})
        return metrics

    # ------------------------------------------------------------------ inputs and guards
    def _run_inputs(self) -> dict[str, Any]:
        """Resolved paths and content hashes of every definition file the run depends on (reproducibility)."""
        anchor = self.config.source_path
        files: dict[str, Any] = {}
        declared = [("feature_spec", self.config.dataset.feature_spec), *[("feature_spec_extension", e) for e in self.config.dataset.feature_spec_extensions]]
        if self.config.model.published_equation is not None:
            declared.append(("published_equation", self.config.model.published_equation.config))
        for role, path in declared:
            resolved = resolve_path(path, anchor=anchor)
            files.setdefault(role, []).append({"declared": portable_path(path), "resolved": str(resolved),
                                               "sha256": definition_file_sha256(resolved)})
        return {"config_source": anchor, "dataset_dir": str(self.dataset.directory) if self.dataset.directory else None,
                "dataset_sha256": self.dataset.manifest.data_sha256, "feature_spec_sha256": self.dataset.spec.content_sha256,
                "files": files, "purpose": self.purpose}

    @staticmethod
    def _check_disjoint_rows(*key_sets: set[tuple[Any, Any]]) -> None:
        names = ("train", "validation", "test")
        for i in range(len(key_sets)):
            for j in range(i + 1, len(key_sets)):
                if key_sets[i] & key_sets[j]:
                    raise LeakageError(f"{names[i]} and {names[j]} partitions share {len(key_sets[i] & key_sets[j])} rows")

    def _assert_fitted_on(self, pipeline: FittedPipeline, train_df: pd.DataFrame) -> None:
        n_fit = getattr(pipeline.preprocessor, "n_train_rows_", None)
        if n_fit is not None and n_fit != len(train_df):
            raise LeakageError(f"preprocessor fitted on {n_fit} rows but training partition has {len(train_df)}")

    def _registry_path(self) -> Path:
        return self.run.path.parent / REGISTRY_FILE

    def _check_test_registry(self, test_sha: str) -> list[dict[str, Any]]:
        """D-19 section 4: one test evaluation per pre-registered config. Returns prior evaluations of these test rows."""
        prior = []
        if self._registry_path().exists():
            for line in self._registry_path().read_text(encoding="utf-8").splitlines():
                if line.strip():
                    entry = json.loads(line)
                    if entry.get("data_sha256") == self.dataset.manifest.data_sha256 and entry.get("test_rows_sha256") == test_sha:
                        prior.append(entry)
        same_config = [e for e in prior if e.get("config_sha256") == self.config.sha256() and e.get("purpose") == "train"]
        if same_config and self.purpose == "train":
            message = (f"test rows already evaluated {len(same_config)} time(s) with this exact config "
                       f"({[e['run_id'] for e in same_config]}); D-19 section 4 allows one evaluation per pre-registered config")
            if self.dataset.manifest.scientific_use_allowed and not self.allow_test_reevaluation:
                raise LeakageError(message + " (pass allow_test_reevaluation=True to override; the override is recorded)")
            self.warnings.append(message)
        return prior

    def _register_test_evaluation(self, test_sha: str, prior: list[dict[str, Any]]) -> dict[str, Any]:
        """Append-only registry entry, written before the test partition is released."""
        config_sha = self.config.sha256()
        entry = {"data_sha256": self.dataset.manifest.data_sha256, "test_rows_sha256": test_sha, "config_sha256": config_sha,
                 "run_id": self.run.run_id, "purpose": self.purpose, "allow_test_reevaluation": self.allow_test_reevaluation,
                 "released_utc": utc_now()}
        with self._registry_path().open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(entry) + "\n")
        return {"registry_file": REGISTRY_FILE, "prior_evaluations_same_test_rows": len(prior),
                "prior_config_sha256": sorted({e.get("config_sha256") for e in prior if e.get("config_sha256") != config_sha}),
                "purpose": self.purpose, "allow_test_reevaluation": self.allow_test_reevaluation}

    # ------------------------------------------------------------------ tuning
    def _tune(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> tuple[dict[str, Any] | None, pd.DataFrame]:
        from falls_ml.models.registry import get_adapter_class

        cols = ["trial", "params_json", "objective", "auroc", "brier", "calibration_slope", "citl", "selected"]
        cfg = self.config
        if not cfg.tuning.enabled:
            return None, pd.DataFrame(columns=cols)
        space = cfg.tuning.search_space or get_adapter_class(cfg.model.name).default_search_space()
        if not space:
            raise ConfigError(f"tuning enabled but model {cfg.model.name} has no search space")
        from falls_ml.tuning import run_search

        base = model_params_from_config(cfg)
        seed = int_seed_for(self.seed, "tuning")

        def fit_fn(df: pd.DataFrame, params: dict[str, Any]) -> FittedPipeline:
            return fit_pipeline(df, self.spec, cfg, random_state=seed, model_params={**base, **params},
                                forbidden_row_keys=self.forbidden_for_fit)

        best, results = run_search(train_df, val_df, fit_fn=fit_fn, space=space, method=cfg.tuning.method, n_iter=cfg.tuning.n_iter,
                                   seed=seed, weights=cfg.tuning.objective.weights, spec=self.spec)
        return {**base, **best}, results

    # ------------------------------------------------------------------ predictions and variants
    def _base_variant(self) -> str:
        return PUBLISHED_PRIMARY if self.is_published else "uncalibrated"

    def _served_variant(self) -> str:
        if not self.serves_recalibrated:
            return self._base_variant()
        return f"{PUBLISHED_PRIMARY}+recalibrated" if self.is_published else "recalibrated"

    def _prediction_variants(self, pipeline: FittedPipeline, df: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray | None]]:
        """Return {variant: (risk, lp)} of uncalibrated predictions."""
        X = pipeline.design(df)
        if self.is_published:
            scored = pipeline.model.score_all_variants(X)
            return {v: (scored[f"risk_{v}"].to_numpy(), scored[f"lp_{v}"].to_numpy()) for v in PUBLISHED_VARIANTS}
        return {"uncalibrated": (pipeline.model.predict_proba(X), pipeline.model.linear_predictor(X))}

    def _fit_recalibrators(self, variants: dict[str, tuple[np.ndarray, np.ndarray | None]], val_df: pd.DataFrame) -> dict[str, Any]:
        from falls_ml.evaluation.calibration import make_recalibrator

        cal = self.config.calibration
        if not cal.enabled or cal.method == "none":
            return {}
        y = val_df[self.y_col].to_numpy(dtype=np.int64)
        return {v: make_recalibrator(cal.method).fit(p, y, lp=lp) for v, (p, lp) in variants.items()}

    def _variant_predictions(self, pipeline: FittedPipeline, df: pd.DataFrame, recalibrators: dict[str, Any]) -> dict[str, tuple[np.ndarray, np.ndarray | None]]:
        from falls_ml.evaluation.metrics import logit

        base = self._prediction_variants(pipeline, df)
        out = dict(base)
        for v, rec in recalibrators.items():
            p, lp = base[v]
            p_recal = rec.transform(p, lp=lp)
            out["recalibrated" if not self.is_published else f"{v}+recalibrated"] = (p_recal, logit(p_recal))
        return out

    # ------------------------------------------------------------------ evaluation
    def _evaluate_split(self, split_name: str, pipeline: FittedPipeline, df: pd.DataFrame, recalibrators: dict[str, Any],
                        tables: dict[str, list]) -> dict[str, Any]:
        from falls_ml.evaluation import metrics as M
        from falls_ml.evaluation.bootstrap import bootstrap_metric_cis

        ev = self.config.evaluation
        y = df[self.y_col].to_numpy(dtype=np.int64)
        cluster = df[ev.bootstrap.cluster_column].to_numpy()
        lo, hi, step = ev.decision_curve_thresholds
        dc_grid = np.round(np.arange(lo, hi + step / 2, step), 10)
        out: dict[str, Any] = {}
        for variant, (p, lp) in self._variant_predictions(pipeline, df, recalibrators).items():
            summary = M.performance_summary(y, p, lp=lp, thresholds=ev.thresholds)
            cis = bootstrap_metric_cis(y, p, lp, n=ev.bootstrap.n, seed=ev.bootstrap.seed, component=f"ci:{split_name}:{variant}",
                                       cluster=cluster, metrics=METRIC_NAMES)
            entry = {k: summary[k] for k in summary if k not in METRIC_NAMES}
            for name in METRIC_NAMES:
                entry[name] = cis.get(name, {"estimate": summary.get(name), "ci_low": None, "ci_high": None})
            out[variant] = entry
            grouped = M.grouped_calibration(y, p, n_groups=ev.calibration_groups).assign(kind="grouped")
            smooth = M.smoothed_calibration(y, p).assign(kind="smoothed", group=np.nan, n=np.nan, ci_low=np.nan, ci_high=np.nan)
            tables["calibration"].append(pd.concat([grouped, smooth], ignore_index=True).assign(split=split_name, variant=variant))
            for row in summary["thresholds"]:
                tables["confusion"].append({"split": split_name, "variant": variant, **{k: row[k] for k in
                                            ("threshold", "tp", "fp", "tn", "fn", "sensitivity", "specificity", "ppv", "npv", "f1")}})
            tables["decision"].append(M.decision_curve(y, p, dc_grid).assign(split=split_name, variant=variant))
            tables["subgroup"].extend(self._subgroups(split_name, variant, df, y, p, lp))
        return out

    def _subgroups(self, split_name: str, variant: str, df: pd.DataFrame, y: np.ndarray, p: np.ndarray, lp: np.ndarray | None) -> list[dict]:
        from falls_ml.evaluation import metrics as M
        from falls_ml.features.transforms import bmi_category

        derived = pd.DataFrame(index=df.index)
        configured = set(self.config.evaluation.subgroups)
        if "sex" in configured:
            derived["sex"] = df["sex"]
        if "bmi_category" in configured:
            derived["bmi_category"] = bmi_category(df["bmi_value"], obese_cutpoint=self.config.preprocessing.bmi_obese_cutpoint)
        if "age_band" in configured:
            derived["age_band"] = _age_band(df["age_years"])
        rows = []
        for var in self.config.evaluation.subgroups:
            for level in sorted(derived[var].dropna().astype(str).unique()):
                mask = (derived[var].astype(str) == level).to_numpy()
                n_ev = int(y[mask].sum())
                row = {"split": split_name, "variant": variant, "subgroup_variable": var, "subgroup_level": level,
                       "n": int(mask.sum()), "n_events": n_ev, "auroc": None, "brier": None, "calibration_slope": None, "citl": None, "oe_ratio": None}
                if 10 <= n_ev < mask.sum():
                    lp_eff = lp[mask] if lp is not None else M.logit(p[mask])
                    row.update({"auroc": M.auroc(y[mask], p[mask]), "brier": M.brier(y[mask], p[mask]),
                                "calibration_slope": M.calibration_slope_intercept(y[mask], lp_eff)[1],
                                "citl": M.citl(y[mask], lp_eff)[0], "oe_ratio": M.oe_ratio(y[mask], p[mask])})
                rows.append(row)
        return rows

    def _heterogeneity(self, pipeline: FittedPipeline, test_df: pd.DataFrame, recalibrators: dict[str, Any]) -> list[dict] | None:
        """Cluster-level test performance of the served variant pooled by random-effects meta-analysis (spec 7.2, D-16, D-21)."""
        cluster_col = self.config.evaluation.cluster_column_for_heterogeneity
        if not cluster_col:
            return None
        from falls_ml.evaluation.meta_analysis import measure_with_variance, pool_measure

        p, lp = self._variant_predictions(pipeline, test_df, recalibrators)[self._served_variant()]
        y = test_df[self.y_col].to_numpy(dtype=np.int64)
        clusters = test_df[cluster_col].astype("string").fillna("missing").to_numpy()
        rows = []
        measures = ("auroc", "calibration_slope", "citl", "oe_ratio")
        for c in sorted(set(clusters)):
            mask = clusters == c
            n_ev = int(y[mask].sum())
            row: dict[str, Any] = {"cluster": c, "n": int(mask.sum()), "n_events": n_ev, "included": 10 <= n_ev < mask.sum()}
            if row["included"]:
                for m in measures:
                    value, var = measure_with_variance(m, y[mask], p[mask], None if lp is None else lp[mask])
                    row[m], row[f"{m}_variance"] = value, var
                row["included"] = all(np.isfinite([row[m] for m in measures] + [row[f"{m}_variance"] for m in measures]))
            rows.append(row)
        per = pd.DataFrame(rows)
        write_csv(self.run.file("heterogeneity_clusters.csv"), per)
        inc = per[per["included"]]
        excluded_share = 1 - inc["n"].sum() / max(per["n"].sum(), 1)
        if excluded_share > 0.2:
            self.warnings.append(f"heterogeneity: clusters with < 10 events hold {excluded_share:.0%} of test rows; consider a coarser cluster (D-16)")
        pooled = []
        if len(inc) >= 2:
            for m in measures:
                res = pool_measure(inc[m].to_numpy(dtype=float), inc[f"{m}_variance"].to_numpy(dtype=float), m)
                pooled.append({"measure": m, **res.to_dict()})
        write_csv(self.run.file("heterogeneity_pooled.csv"), pd.DataFrame(pooled))
        return pooled

    def _write_predictions(self, pipeline: FittedPipeline, df: pd.DataFrame, recalibrators: dict[str, Any], name: str) -> None:
        preds = self._variant_predictions(pipeline, df, recalibrators)
        out = pd.DataFrame({"research_id": df[self.id_col].to_numpy(), "index_date": df[self.spec.index_column].to_numpy(),
                            "outcome": df[self.y_col].to_numpy(dtype=np.int8)})
        base = self._base_variant()
        out["risk_uncalibrated"] = preds[base][0]
        recal_key = "recalibrated" if not self.is_published else f"{base}+recalibrated"
        out["risk_recalibrated"] = preds[recal_key][0] if recal_key in preds else np.nan
        out["risk_served"] = preds[self._served_variant()][0]
        out["linear_predictor"] = preds[base][1] if preds[base][1] is not None else np.nan
        if self.is_published:
            for v in PUBLISHED_VARIANTS:
                out[f"lp_{v}"], out[f"risk_{v}"] = preds[v][1], preds[v][0]
        write_parquet(self.run.file(name), out)

    # ------------------------------------------------------------------ feature analysis
    def _fit_fn(self):
        cfg, spec, params, forbidden = self.config, self.spec, dict(self.frozen_params), self.forbidden_for_fit

        def fit_fn(df: pd.DataFrame, seed: int, *, model_param_overrides: dict[str, Any] | None = None) -> FittedPipeline:
            # resampling refits use the frozen (tuned) parameters of the final model; tuning is not repeated per replicate.
            # model_param_overrides is used only by the bootstrap convergence-retry policy (full_path_max_iter).
            return fit_pipeline(df, spec, cfg, random_state=seed, model_params={**params, **(model_param_overrides or {})},
                                forbidden_row_keys=forbidden)
        return fit_fn

    def _feature_analysis(self, pipeline: FittedPipeline, train_df: pd.DataFrame, val_df: pd.DataFrame):
        from falls_ml.evaluation.importance import build_feature_importance_table, coefficient_table, permutation_importance_grouped

        cfg = self.config
        coef_df = self._coefficients_with_intercept_and_omissions(pipeline, coefficient_table(pipeline))
        write_csv(self.run.file("coefficients.csv"), coef_df)
        diag = pipeline.model.fit_diagnostics() if pipeline.model.requires_fit else {}
        cv = diag.get("cv_results") if isinstance(diag, dict) else None
        write_csv(self.run.file("cv_results.csv"), cv if isinstance(cv, pd.DataFrame)
                  else pd.DataFrame(columns=["lambda", "cv_mean_deviance", "cv_se", "n_nonzero", "selected"]))
        refit = diag.get("unpenalized_refit") if isinstance(diag, dict) else None
        if isinstance(refit, pd.DataFrame):
            write_csv(self.run.file("unpenalized_refit.csv"), refit)
        self._write_fp_selection(pipeline)

        stability = None
        st = cfg.analysis.stability
        if st.enabled and pipeline.model.requires_fit:
            from falls_ml.evaluation.stability import bootstrap_stability
            permutation = ({"features": self.spec.predictor_names(), "n_repeats": st.permutation_repeats, "metric": "auroc",
                            "max_rows": st.permutation_max_rows} if st.permutation_repeats > 0 else None)
            stability = bootstrap_stability(train_df, self._fit_fn(), n_bootstrap=st.n_bootstrap, seed=int_seed_for(self.seed, "stability"),
                                            cluster_column=self.id_col, thresholds=cfg.evaluation.thresholds,
                                            selection_threshold=st.selection_threshold, sign_stability_threshold=st.sign_stability_threshold,
                                            reference_pipeline=pipeline, permutation=permutation, spec=self.spec,
                                            retry_log=self.convergence_retries)
            write_csv(self.run.file("feature_stability.csv"), stability.feature_stability)
            write_csv(self.run.file("instability.csv"), stability.instability)
            if stability.n_failed:
                self.warnings.append(f"stability: {stability.n_failed} bootstrap replicate(s) failed and were recorded "
                                     f"({sorted({f['reason'][:80] for f in stability.failures})})")
        else:
            reason = "published equation has no fitting process" if not pipeline.model.requires_fit else "disabled in config"
            self.warnings.append(f"feature stability not computed: {reason}")
            from falls_ml.evaluation.stability import FEATURE_STABILITY_COLUMNS, INSTABILITY_COLUMNS
            write_csv(self.run.file("feature_stability.csv"), pd.DataFrame(columns=FEATURE_STABILITY_COLUMNS))
            write_csv(self.run.file("instability.csv"), pd.DataFrame(columns=INSTABILITY_COLUMNS))

        y_val = val_df[self.y_col].to_numpy(dtype=np.int64)
        perm_df = permutation_importance_grouped(pipeline, val_df, y_val, features=self.spec.predictor_names(),
                                                 n_repeats=cfg.evaluation.permutation_importance_repeats,
                                                 seed=int_seed_for(self.seed, "permutation"), metric="auroc")
        fi = build_feature_importance_table(pipeline, perm_df, stability.feature_stability if stability else None, model_name=cfg.model.name)
        write_csv(self.run.file("feature_importance.csv"), fi)
        return coef_df, stability, perm_df

    def _record_convergence_retries(self) -> None:
        """bootstrap_convergence_retries.csv (one row per initial convergence failure of a replicate, with its retry outcome)."""
        from falls_ml.evaluation.convergence_retry import retry_table

        write_csv(self.run.file("bootstrap_convergence_retries.csv"), retry_table(self.convergence_retries))
        if self.convergence_retries:
            ok = sum(1 for r in self.convergence_retries if r["accepted"])
            self.warnings.append(f"bootstrap convergence retry: {len(self.convergence_retries)} replicate fit(s) did not converge at the "
                                 f"configured iteration limit; {ok} converged on the pre-declared retry (same resample, seed and lambda; "
                                 f"higher full-data iteration limit) and {len(self.convergence_retries) - ok} still failed and were counted as "
                                 "failed replicates")

    def _coefficients_with_intercept_and_omissions(self, pipeline: FittedPipeline, coef_df: pd.DataFrame) -> pd.DataFrame:
        """coefficients.csv per ARTIFACT_SCHEMAS: D-10 re-expression (LASSO), intercept row and Stata-style omission flags."""
        if coef_df.empty:
            return coef_df.assign(relative_to_reference=pd.Series(dtype=bool), omitted=pd.Series(dtype=bool))
        coef_df = coef_df.assign(relative_to_reference=False)
        omitted_terms: set[str] = set()
        intercept = None
        if pipeline.model.requires_fit:
            diag = pipeline.model.fit_diagnostics()
            intercept = diag.get("intercept")
            table = diag.get("coefficients")
            if isinstance(table, pd.DataFrame) and "omitted" in table:
                omitted_terms = set(table.loc[table["omitted"].astype(bool), "term"].astype(str))
            omissions = diag.get("stata_logit_omissions") or {}
            if omissions.get("constant_columns") or omissions.get("perfect_predictors"):
                self.warnings.append(f"Stata-style omissions in the final fit: constant columns {omissions.get('constant_columns')}, "
                                     f"perfect predictors {[p['column'] for p in omissions.get('perfect_predictors', [])]} "
                                     f"({omissions.get('n_rows_dropped', 0)} training rows dropped; coefficients fixed at 0)")
        else:
            intercept = pipeline.model.fit_diagnostics()["published_equation"]["intercept"]
        coef_df["omitted"] = coef_df["design_column"].astype(str).isin(omitted_terms)
        frames = [coef_df]
        if self.config.model.name == "lasso_logistic_cv":
            rel_intercept, rel = pipeline.model.coefficients_relative_to_reference(pipeline.preprocessor.reference_levels())
            extra = coef_df.copy()
            extra["coefficient"] = extra["design_column"].map(rel)
            extra["odds_ratio"] = np.exp(extra["coefficient"])
            extra["abs_coefficient"] = extra["coefficient"].abs()
            extra["relative_to_reference"] = True
            frames.append(extra)
            frames.append(pd.DataFrame([{"design_column": "_intercept", "feature": "_intercept", "coefficient": rel_intercept,
                                         "relative_to_reference": True, "omitted": False}]))
        if intercept is not None:
            frames.insert(1, pd.DataFrame([{"design_column": "_intercept", "feature": "_intercept", "coefficient": float(intercept),
                                            "relative_to_reference": False, "omitted": False}]))
        return pd.concat(frames, ignore_index=True)

    def _write_fp_selection(self, pipeline: FittedPipeline) -> None:
        forms = getattr(pipeline.preprocessor, "fp_forms_", None) or {}
        if not forms:
            return
        rows = []
        for variable, form in forms.items():
            t = dict(form.selection_table or {})
            omissions = t.pop("stata_logit_omissions", None)
            rows.append({"variable": variable, "powers": list(form.powers), "shift": form.shift, "scale": form.scale,
                         **{k: t.get(k) for k in ("mode", "decision", "deviance_linear", "deviance_fp1", "fp1_powers", "deviance_fp2", "fp2_powers",
                                                  "p_nonlinear", "p_fp2_vs_fp1", "cycles_run", "converged")},
                         "omissions_json": json.dumps(omissions) if omissions else None})
            if omissions and (omissions.get("constant_columns") or omissions.get("perfect_predictors")):
                note = (f"FP selection (Stata logit omissions): constant {omissions.get('constant_columns')}, "
                        f"perfect predictors {[p['column'] for p in omissions.get('perfect_predictors', [])]}")
                if note not in self.warnings:
                    self.warnings.append(note)
        write_csv(self.run.file("fp_selection.csv"), pd.DataFrame(rows))

    def _internal_validation(self, train_df: pd.DataFrame) -> dict[str, Any]:
        cfg = self.config
        out: dict[str, Any] = {"optimism": None, "iecv": None, "heterogeneity": None}
        if cfg.analysis.optimism.enabled and not self.is_published:
            from falls_ml.evaluation.optimism import harrell_optimism
            opt = harrell_optimism(train_df, self._fit_fn(), n_bootstrap=cfg.analysis.optimism.n_bootstrap,
                                   seed=int_seed_for(self.seed, "optimism"), cluster_column=self.id_col, spec=self.spec,
                                   retry_log=self.convergence_retries)
            write_csv(self.run.file("optimism.csv"), opt)
            out["optimism"] = opt.to_dict(orient="records")
        if cfg.analysis.iecv.enabled and not self.is_published:
            from falls_ml.evaluation.iecv import internal_external_cv
            per, pooled = internal_external_cv(train_df, self._fit_fn(), cluster_column=cfg.analysis.iecv.cluster_column,
                                               spec=self.spec, seed=int_seed_for(self.seed, "iecv"))
            write_csv(self.run.file("iecv_clusters.csv"), per)
            write_csv(self.run.file("iecv_pooled.csv"), pooled)
            out["iecv"] = pooled.to_dict(orient="records")
        return out

    # ------------------------------------------------------------------ bundle
    def _save_bundle(self, pipeline: FittedPipeline, train_df: pd.DataFrame, performance: dict[str, Any]) -> None:
        from falls_ml.bundle import save_bundle
        from falls_ml.monitoring import build_reference_profile

        val = performance["validation"][self._served_variant()]
        reference_performance = {k: _estimate(val.get(k)) for k in ("auroc", "calibration_slope", "citl", "oe_ratio")}
        reference_performance.update(prevalence=val.get("observed_rate"), split="validation", variant=self._served_variant())
        profile = build_reference_profile(train_df, self.spec, pipeline, reference_performance=reference_performance)
        extra = {"unavailable_predictors": list(self.config.model.published_equation.unavailable_predictors) if self.is_published else [],
                 "mappings_clinically_validated": bool(self.dataset.manifest.scientific_use_allowed), "run_id": self.run.run_id,
                 "served_variant": self._served_variant()}
        coverage = self._efalls_coverage()
        if coverage is not None:
            extra["efalls_coverage"] = {k: coverage[k] for k in ("n_available", "n_total", "coverage_pct", "unavailable",
                                                                  "is_full_efalls_feature_set", "label")}
        save_bundle(self.run.model, pipeline, experiment_config=self.config, dataset_manifest=self.dataset.manifest,
                    reference_profile=profile, model_version=self.run.run_id, created_utc=self.run.created_utc, extra_metadata=extra)

    # ------------------------------------------------------------------ metrics.json
    def _calibration_status(self, entry: dict[str, Any]) -> str:
        rep = self.config.reporting.eligibility
        slope, citl = _estimate(entry.get("calibration_slope")), _estimate(entry.get("citl"))
        if slope is None or citl is None:
            return "not_assessed"
        ok = rep.calibration_slope_range[0] <= slope <= rep.calibration_slope_range[1] and abs(citl) <= rep.max_abs_citl
        return "adequate" if ok else "miscalibrated"

    def _missingness(self, pipeline: FittedPipeline, frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
        """Missing rate per split and how the preprocessor handled missing values (no silent imputation)."""
        from falls_ml.features.preprocessing import ALCOHOL, BMI, SMOKING

        published_style = self.config.preprocessing.representation == "efalls_published"
        medians = getattr(pipeline.preprocessor, "medians_", {}) or {}
        out = {}
        for f in self.spec.features:
            if f.missing_rule == "forbid":
                handling = "missing values not allowed (dataset validation fails)"
            elif f.missing_rule == "absent_is_zero":
                handling = "no record = 0 (absence of a code/prescription is absence of the predictor)"
            elif f.name == BMI:
                handling = "no valid BMI -> 'missing' BMI category indicator"
            elif f.name == SMOKING:
                handling = ("missing -> ex/never reference (published term, D-04)" if published_style
                            else "missing merged into 'never' (D-04, D-10)")
            elif f.name == ALCOHOL:
                handling = "missing -> 'missing' alcohol category indicator (D-05)"
            elif f.dtype == "float_nullable":
                median = medians.get(f.name)
                handling = (f"training-set median {median:.4g} + '{f.name}__missing' indicator" if median is not None
                            else "training-set median + missing indicator")
            elif f.is_categorical and f.nullable:
                handling = (f"missing -> reference level {f.reference_level!r}" if f.missing_rule == "reference_level"
                            else "missing -> separate 'missing' level indicator")
            else:
                handling = f"rule {f.missing_rule}"
            out[f.name] = {"missing_rate": float(pd.concat([v[f.name] for v in frames.values()]).isna().mean()),
                           "missing_rate_by_split": {k: float(v[f.name].isna().mean()) for k, v in frames.items()},
                           "rule": f.missing_rule, "handling": handling}
        return out

    def _assemble_metrics(self, pipeline, performance, split_info, frame, plan, best_params, timing, coef_df, perm_df, internal,
                          recalibrators, test_df, train_df, val_df, registry) -> dict[str, Any]:
        cfg, m = self.config, self.dataset.manifest
        served = self._served_variant()
        test_served = performance["test"][served]
        coverage = self._efalls_coverage()
        rep = cfg.reporting.eligibility
        status = self._calibration_status(test_served)
        auc = _estimate(test_served["auroc"])
        r3 = lambda v: "n/a" if v is None else f"{v:.3f}"  # noqa: E731
        reasons = []
        if m.source == "synthetic_fixture" or not m.scientific_use_allowed:
            reasons.append("dataset is not approved for scientific use (synthetic fixture or unvalidated mappings)")
        if auc is None or auc < rep.min_auroc:
            reasons.append(f"test AUROC {r3(auc)} below minimum {rep.min_auroc:.2f}")
        if status != "adequate":
            reasons.append(f"test calibration {status} (slope {r3(_estimate(test_served['calibration_slope']))}, CITL {r3(_estimate(test_served['citl']))}; "
                           f"required slope {rep.calibration_slope_range[0]}-{rep.calibration_slope_range[1]}, |CITL| <= {rep.max_abs_citl})")
        diag = pipeline.model.fit_diagnostics() if pipeline.model.requires_fit else {}
        plain = coef_df[(~coef_df["relative_to_reference"].astype(bool)) & (coef_df["design_column"] != "_intercept")] if len(coef_df) else coef_df
        n_selected = int((plain["coefficient"] != 0).sum()) if len(plain) else None
        top = perm_df.sort_values("permutation_importance_mean", ascending=False)["feature"].head(cfg.reporting.top_n_features).tolist()
        cohort = {"n_rows": len(frame), "n_patients": int(frame[self.id_col].nunique()), "n_events": int(frame[self.y_col].sum()),
                  "prevalence": float(frame[self.y_col].mean()), "by_split": plan.summary(frame, self.spec)}
        fit_notes = {"stata_logit_omissions": diag.get("stata_logit_omissions") if isinstance(diag, dict) else None,
                     "fp_selection_omissions": {v: f.selection_table.get("stata_logit_omissions")
                                                for v, f in (getattr(pipeline.preprocessor, "fp_forms_", None) or {}).items()}}
        effective = {k: v for k, v in pipeline.model.get_params().items() if not isinstance(v, (pd.DataFrame, np.ndarray))}
        metrics: dict[str, Any] = {
            "schema_version": "1.2", "run_id": self.run.run_id, "created_utc": self.run.created_utc,
            "experiment": {"name": cfg.experiment.name, "kind": cfg.experiment.kind, "layers": list(cfg.experiment.layers),
                           "description": cfg.experiment.description},
            "model": {"name": cfg.model.name, "params": model_params_from_config(cfg), "best_params": best_params, "effective_params": effective,
                      "resampling_params": dict(self.frozen_params), "is_linear": pipeline.model.is_linear, "fit_notes": fit_notes},
            "dataset": m.to_dict(), "synthetic_fixture": m.source == "synthetic_fixture", "scientific_use_allowed": m.scientific_use_allowed,
            "feature_set": {"name": self.spec.name, "version": self.spec.version, "sha256": self.spec.content_sha256,
                            "pure_efalls": self.spec.is_pure_efalls, "n_features": len(self.spec.features)},
            "outcome": self._outcome_record(),
            "efalls_coverage": coverage,
            "cohort": cohort, "split": split_info,
            "primary_variant": served, "served_variant": served,
            "transportability_variant": PUBLISHED_PRIMARY if self.is_published else None,
            "performance": performance,
            "calibration": {"method": cfg.calibration.method if cfg.calibration.enabled else "none", "fitted_on": "validation",
                            "served": self.serves_recalibrated, "params": {v: r.params() for v, r in recalibrators.items()}},
            "calibration_status": status,
            "calibration_status_uncalibrated": self._calibration_status(performance["test"][self._base_variant()]),
            "features": {"n_design_columns": len(pipeline.preprocessor.design_columns()), "n_selected": n_selected, "top_features": top},
            "lasso": ({"lambda_star": diag.get("lambda_star"), "lambda_max": diag.get("lambda_max"), "lambda_ratio": diag.get("lambda_ratio"),
                       "n_lambda": len(diag["cv_results"]), "cv_folds": diag.get("cv_folds", cfg.validation.cv_folds),
                       "cv_criterion": "mean_deviance", "selection_rule": pipeline.model.get_params().get("selection", "min"),
                       "cv_minimum_identified": diag.get("cv_minimum_identified"), "cv_paths_converged": diag.get("cv_paths_converged"),
                       "n_selected": diag.get("n_selected"), "intercept": diag.get("intercept"), "seed": diag.get("seed")}
                      if cfg.model.name == "lasso_logistic_cv" else None),
            "hyperparameter_search": {"enabled": cfg.tuning.enabled, "method": cfg.tuning.method,
                                      "n_trials": int(len(pd.read_csv(self.run.file("hyperparameter_results.csv")))) if cfg.tuning.enabled else 0,
                                      "objective": cfg.tuning.objective.weights,
                                      "note": "resampling analyses reuse the tuned parameters; tuning is not repeated inside replicates"},
            "timing": timing,
            "test_evaluation_registry": registry,
            "eligibility": {"eligible_for_further_validation": not reasons, "reasons": reasons, "judged_on_variant": served},
            "comparison_to_efalls": self._comparison_to_efalls(auc, split_info["test_rows_sha256"]),
            "published_scoring": self._published_block(pipeline, test_df, recalibrators) if self.is_published else None,
            "internal_validation": internal,
            "bootstrap_convergence_retry": _retry_summary(self.convergence_retries),
            "missingness": self._missingness(pipeline, {"train": train_df, "validation": val_df, "test": test_df}),
            "limitations": self._limitations(plan, m),
            "production_readiness": {"bundle_saved": True, "served_variant": served, "thresholds_approved": cfg.reporting.risk_categories.approved,
                                     "mappings_clinically_validated": bool(m.scientific_use_allowed),
                                     "unavailable_predictors": len(coverage["unavailable"]) if coverage is not None else 0},
            "warnings": self.warnings, "code_version": code_version(),
            "artifacts": {"report_html": "report.html", "report_md": "report.md", "model_bundle": "model/", "complete_marker": COMPLETE_MARKER},
        }
        return metrics

    def _exploratory_outcome_note(self) -> str:
        o = self.spec.outcome
        return (f"{EXPLORATORY_OUTCOME_WARNING}: outcome {o.name!r} ({o.horizon_label} window; layer {o.raw.get('layer')}) is not the "
                f"published eFalls outcome (ED attendance or hospital admission with fall or fracture within 12 months); "
                f"{o.raw.get('status', 'exploratory outcome')}.")

    def _outcome_record(self) -> dict[str, Any]:
        o = self.spec.outcome
        return {"name": o.name, "horizon": o.horizon_label, "layer": o.raw.get("layer"), "published_efalls_outcome": o.is_published_efalls,
                "concept": o.raw.get("concept"), "status": o.raw.get("status", "published eFalls outcome" if o.is_published_efalls else "exploratory")}

    def _efalls_coverage(self) -> dict[str, Any] | None:
        """``metrics.efalls_coverage`` for eFalls-labelled kinds (None otherwise); see :func:`efalls_coverage`."""
        kind = self.config.experiment.kind
        if kind not in EFALLS_KINDS:
            return None
        if self.is_published:
            unavailable = set(self.config.model.published_equation.unavailable_predictors)
            available = [f.name for f in self.full_spec.features if f.exact_efalls_baseline and f.name not in unavailable]
        else:
            available = self.spec.predictor_names()
        return efalls_coverage(self.full_spec, available, kind=kind)

    def _comparison_to_efalls(self, auc: float | None, test_sha: str) -> dict[str, Any]:
        """Point estimate vs the published eFalls primary variant on identical data and test rows (no hidden CIs)."""
        data_sha = self.dataset.manifest.data_sha256
        candidates = []
        for run_dir in sorted(p.parent for p in self.run.path.parent.glob(f"*/{COMPLETE_MARKER}")):
            if run_dir == self.run.path:
                continue
            other = _read_complete_metrics(run_dir)
            if not other or (other.get("experiment") or {}).get("kind") != "efalls_published_scoring":
                continue
            ps = other.get("published_scoring") or {}
            test_perf = ((other.get("performance") or {}).get("test") or {}).get(PUBLISHED_PRIMARY)
            if (other.get("split", {}).get("test_rows_sha256") == test_sha and other.get("dataset", {}).get("data_sha256") == data_sha
                    and ps.get("effective_experiment_label") == "efalls_published_scoring" and not ps.get("low_support_zeroed")
                    and ps.get("unavailable_fill") in (None, "zero") and test_perf):
                candidates.append((other["created_utc"], other["run_id"], test_perf["auroc"]["estimate"]))
        if not candidates:
            return {"reference_run_id": None, "reference_variant": PUBLISHED_PRIMARY, "delta_auroc_test": None, "candidates": [],
                    "note": "no completed published-eFalls run on identical data and test rows was found in this runs directory"}
        _, ref_id, ref_auc = sorted(candidates)[-1]
        return {"reference_run_id": ref_id, "reference_variant": PUBLISHED_PRIMARY,
                "delta_auroc_test": None if auc is None or ref_auc is None else auc - ref_auc,
                "candidates": [c[1] for c in sorted(candidates)],
                "note": "point estimate vs the latest matching published-eFalls run (primary sex parameterisation); "
                        "paired bootstrap CIs are produced by `falls_ml compare` (comparison_paired_differences.csv)"}

    def _published_block(self, pipeline: FittedPipeline, test_df: pd.DataFrame, recalibrators: dict[str, Any]) -> dict[str, Any]:
        import statsmodels.api as sm

        from falls_ml.evaluation import metrics as M

        pe = self.config.model.published_equation
        cov = pipeline.model.coverage()
        y = test_df[self.y_col].to_numpy(dtype=np.int64)
        preds = self._variant_predictions(pipeline, test_df, recalibrators)
        male = (test_df["sex"] == "male").to_numpy()
        sex_oe = {v: {"female": M.oe_ratio(y[~male], p[~male]) if (~male).any() else None,
                      "male": M.oe_ratio(y[male], p[male]) if male.any() else None} for v, (p, _) in preds.items()}
        sex_citl = None
        lp_c = preds[PUBLISHED_PRIMARY][1]
        if male.any() and (~male).any():
            fit = sm.GLM(y, sm.add_constant(male.astype(float)), family=sm.families.Binomial(), offset=lp_c).fit()
            ci = fit.conf_int()[1]
            sex_citl = {"male_minus_female": float(fit.params[1]), "ci_low": float(ci[0]), "ci_high": float(ci[1]),
                        "note": "descriptive only; mixes true sex-specific miscalibration (D-01)"}
        return {"sex_parameterisations": list(PUBLISHED_VARIANTS), "primary": PUBLISHED_PRIMARY, "co_reported": PUBLISHED_CO_REPORTED,
                "configured_variant": pe.sex_parameterisation, "effective_experiment_label": cov["effective_experiment_label"],
                "coverage": {"lp_variance_share_available": cov["lp_variance_share_available"],
                             "mandatory_unavailable": cov["mandatory_unavailable"], "threshold": cov.get("threshold")},
                "unavailable_fill": pe.unavailable_fill, "low_support_zeroed": pe.zero_low_support_predictors,
                "unavailable_predictors": cov.get("unavailable_predictors", []),
                "efalls_feature_coverage": cov.get("efalls_feature_coverage", "complete" if not pe.unavailable_predictors else "incomplete"),
                "served_prediction": ("published equation (uncalibrated)" if not self.serves_recalibrated
                                      else "recalibrated published equation (recalibration fitted on the validation split)"),
                "sex_specific_citl_lp_c": sex_citl, "sex_specific_oe": sex_oe}

    def _limitations(self, plan, manifest) -> list[str]:
        lims = ["eFalls performed no temporal validation; temporal Meuhedet results have no published comparator (spec D-19).",
                "The published eFalls sex term is internally inconsistent; results depend on the sex parameterisation (spec D-01).",
                "Several eFalls definitions (time windows, measurement rules, polypharmacy unit, alcohol) are documented assumptions (spec §9)."]
        if manifest.source == "synthetic_fixture":
            lims.insert(0, "SYNTHETIC FIXTURE: these results are software test output, not scientific evidence.")
        if not manifest.scientific_use_allowed:
            lims.append("Clinical code mappings are not clinically validated (spec B-02).")
        if plan.limitation_note:
            lims.append(f"Split limitation: {plan.limitation_note}")
        if self.config.tuning.enabled:
            lims.append("Hyperparameters were tuned on the validation split; resampling analyses reuse them without re-tuning (optimism may be understated).")
        if self.is_published and self.config.model.published_equation.unavailable_predictors:
            lims.append("Some published predictors are unavailable in this data and were handled per M-11.")
        if not self.spec.outcome.is_published_efalls:
            lims.insert(1 if manifest.source == "synthetic_fixture" else 0, self._exploratory_outcome_note())
        coverage = self._efalls_coverage()
        if self.config.experiment.kind == REDUCED_KIND and coverage is not None:
            lims.insert(1 if manifest.source == "synthetic_fixture" else 0,
                        f"{REDUCED_WARNING}: only {coverage['n_available']} of {coverage['n_total']} eFalls candidate predictors "
                        f"({coverage['coverage_pct']:.1f}%) were available; unavailable: {', '.join(coverage['unavailable'])}.")
            if coverage["mandatory_unavailable"]:
                lims.append(f"Mandatory eFalls predictors unavailable (cohort.mandatory_for_efalls_label): "
                            f"{', '.join(coverage['mandatory_unavailable'])}.")
        return lims

    # ------------------------------------------------------------------ plots
    def _plots(self, pipeline, test_df, recalibrators, coef_df, stability, tuning_df) -> None:
        from falls_ml.evaluation import metrics as M
        from falls_ml.reporting import plots as P

        plots = self.run.plots
        y = test_df[self.y_col].to_numpy(dtype=np.int64)
        preds = self._variant_predictions(pipeline, test_df, recalibrators)
        label = self.config.model.name
        curves = {f"{label} [{v}]": (y, p) for v, (p, _) in preds.items() if not v.startswith(("lp_b", "lp_d2"))}
        P.plot_roc(curves, plots / "roc.png", f"ROC curve — test set ({label})")
        P.plot_precision_recall(curves, plots / "precision_recall.png", f"Precision–recall — test set ({label})", prevalence=float(y.mean()))
        served = self._served_variant()
        p0 = preds[served][0]
        P.plot_calibration(M.grouped_calibration(y, p0, n_groups=self.config.evaluation.calibration_groups), M.smoothed_calibration(y, p0),
                           plots / "calibration.png", f"Calibration — test set ({label}, served: {served})", predictions=p0)
        recal_key = "recalibrated" if not self.is_published else f"{PUBLISHED_PRIMARY}+recalibrated"
        if recal_key in preds:
            P.plot_calibration_before_after(y, preds[self._base_variant()][0], preds[recal_key][0], plots / "calibration_before_after.png",
                                            f"Calibration before and after recalibration (fitted on validation) — test set ({label})")
        lo, hi, step = self.config.evaluation.decision_curve_thresholds
        P.plot_decision_curve(M.decision_curve(y, p0, np.round(np.arange(lo, hi + step / 2, step), 10)), plots / "decision_curve.png",
                              f"Decision curve — test set ({label}, served: {served})")
        P.plot_risk_distribution(y, p0, plots / "risk_distribution.png", f"Predicted 12-month risk by outcome — test set ({label}, {served})")
        fi = pd.read_csv(self.run.file("feature_importance.csv"))
        P.plot_feature_importance(fi, plots / "feature_importance.png", f"Permutation importance (validation AUROC drop) — {label}")
        if len(coef_df):
            plain = coef_df[(~coef_df["relative_to_reference"].astype(bool)) & (coef_df["design_column"] != "_intercept")]
            P.plot_coefficients(plain, plots / "coefficients.png", f"Coefficients (odds ratios) — {label}")
        if stability is not None and len(stability.feature_stability):
            P.plot_selection_stability(stability.feature_stability, plots / "selection_stability.png", f"Feature selection stability — {label}")
        if len(tuning_df):
            P.plot_hyperparameter_search(tuning_df, plots / "hyperparameter_search.png", f"Hyperparameter search (validation) — {label}")
        cv = pd.read_csv(self.run.file("cv_results.csv"))
        if len(cv):
            P.plot_lasso_cv_path(cv, plots / "lasso_cv_path.png", "LASSO 10-fold cross-validation (mean deviance)")


# ============================================================================ ablation and reproduction
def run_ablation(ablation_config_path: str | Path, dataset: str | Path | None = None, *, runs_dir: str | Path | None = None,
                 out_dir: str | Path = "reports/ablation") -> pd.DataFrame:
    """Nested feature-group ablation (user requirement 20): baseline eFalls features, then cumulative groups.

    YAML: {base_experiment_config: path, extension_specs: [paths], steps: [{name: '+ cognition', groups: [cognition]}, ...]}
    Every step reuses the base config's partitions, outcome and seed; only the feature set changes. Runs default to
    ``<output.runs_dir>_ablation`` so they never mix with the master eFalls comparison.
    """
    from falls_ml.reporting.ablation import summarize_ablation

    ablation_path = Path(ablation_config_path)
    raw = yaml.safe_load(ablation_path.read_text(encoding="utf-8"))
    unknown = sorted(set(raw) - {"base_experiment_config", "extension_specs", "steps", "name"})
    if unknown:
        raise ConfigError(f"ablation config: unknown keys {unknown}")
    base = load_experiment_config(resolve_path(raw["base_experiment_config"], anchor=ablation_path))
    if base.experiment.kind in EFALLS_KINDS:
        raise ConfigError("ablation base must be an alternative_model or ablation_member config (eFalls experiments stay pure)")
    ext = tuple(_ablation_extension_path(e, ablation_path, base.source_path) for e in raw.get("extension_specs", ()))
    full_spec = load_feature_spec(base.dataset.feature_spec, ext, anchor=base.source_path)
    baseline_features = [f.name for f in full_spec.features if f.exact_efalls_baseline]
    runs_root = Path(runs_dir) if runs_dir else Path(f"{base.output.runs_dir}_ablation")
    runs: list[tuple[str, Path]] = []
    cumulative = list(baseline_features)
    base_raw = base.to_dict()
    base_raw["dataset"]["feature_spec_extensions"] = list(ext)
    base_raw["experiment"]["kind"] = "ablation_member"
    steps = [{"name": "eFalls baseline", "groups": []}, *raw["steps"]]
    for step in steps:
        groups = step.get("groups", [])
        unknown_groups = sorted(set(groups) - set(full_spec.groups()))
        if unknown_groups:
            raise ConfigError(f"ablation step {step['name']!r}: unknown groups {unknown_groups}")
        cumulative += [f.name for f in full_spec.features if f.group in groups and f.name not in cumulative]
        step_raw = json.loads(json.dumps(base_raw))
        step_raw["experiment"]["name"] = f"{base.experiment.name}__{_slug(step['name'])}"
        step_raw["experiment"]["layers"] = sorted(set(step_raw["experiment"]["layers"]) | {"L3b_meuhedet_predictor"})
        step_raw["preprocessing"]["features"] = list(cumulative)
        step_cfg = dataclasses.replace(experiment_config_from_dict(step_raw, name=step["name"]), source_path=base.source_path)
        result = run_experiment(None, dataset, step_cfg, runs_dir=runs_root)
        runs.append((step["name"], result.run_dir))
    return summarize_ablation(runs[0][1], runs[1:], Path(out_dir))


def _ablation_extension_path(declared: str, ablation_path: Path, base_source: str | Path | None) -> str:
    """Extension-spec path written into the step configs. The declared relative path (``/`` separators) when it names the same
    file from the base config's location as from the ablation config, so the step config hashes (and the test-evaluation
    registry keys) do not depend on the machine or project folder; otherwise the absolute resolved path."""
    resolved = resolve_path(declared, anchor=ablation_path)
    if Path(declared).is_absolute():
        return str(resolved)
    from_base = resolve_path(declared, anchor=base_source, must_exist=False)
    return portable_path(declared) if from_base.exists() and from_base.resolve() == resolved.resolve() else str(resolved)


def reproduce(run_dir: str | Path, *, dataset: str | Path | None = None, runs_dir: str | Path | None = None, tolerance: float = 1e-9) -> dict[str, Any]:
    """Re-run an experiment from its saved artifacts and verify definitions, data and results.

    Refuses to run when the dataset, feature spec or published-equation file content differs from the original run.
    The reproduced run goes to ``<runs root>_reproduced`` by default so it never duplicates the master comparison.
    """
    run_dir = Path(run_dir).resolve()
    original = _read_complete_metrics(run_dir)
    if original is None:
        raise ConfigError(f"{run_dir} is not a completed run (missing {COMPLETE_MARKER} or unreadable metrics.json)")
    config = dataclasses.replace(experiment_config_from_dict(yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8")),
                                                             name=str(run_dir / "config.yaml")), source_path=str(run_dir / "config.yaml"))
    manifest = json.loads((run_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    inputs = json.loads((run_dir / "run_inputs.json").read_text(encoding="utf-8"))
    ds_path = dataset
    if ds_path is None and manifest.get("dataset_path_relative"):
        # project-relative path first: still valid after the project folder was moved or copied to another machine
        candidate = resolve_path(manifest["dataset_path_relative"], anchor=run_dir, must_exist=False)
        ds_path = candidate if candidate.exists() else None
    ds_path = ds_path or manifest.get("dataset_path_used")
    if ds_path is None:
        raise ConfigError("cannot determine the dataset directory of the original run; pass dataset=")
    problems = []
    for role, entries in inputs.get("files", {}).items():
        for entry in entries:
            current = resolve_path(entry["declared"], anchor=run_dir)
            if definition_file_sha256(current) != entry["sha256"]:
                problems.append(f"{role} {entry['declared']} changed since the original run")
    if problems:
        raise ConfigError("definitions changed; exact reproduction impossible: " + "; ".join(problems))
    spec = load_feature_spec(config.dataset.feature_spec, config.dataset.feature_spec_extensions, anchor=run_dir)
    ds = ModelingDataset.load(ds_path, spec, features=config.preprocessing.features)
    if ds.manifest.data_sha256 != manifest["data_sha256"]:
        raise DatasetValidationError("dataset hash differs from the original run; exact reproduction impossible")
    out_root = Path(runs_dir) if runs_dir else Path(f"{run_dir.parent}_reproduced")
    new = run_experiment(None, ds, config, runs_dir=out_root, purpose="reproduce")
    diffs = []
    for split, variants in original["performance"].items():
        for variant, vals in variants.items():
            for key in METRIC_NAMES:
                for part in ("estimate", "ci_low", "ci_high"):
                    a, b = vals[key][part], new.metrics["performance"][split][variant][key][part]
                    if (a is None) != (b is None) or (a is not None and abs(a - b) > tolerance):
                        diffs.append({"split": split, "variant": variant, "metric": f"{key}.{part}", "original": a, "reproduced": b})
    for key in (("split", "test_rows_sha256"), ("feature_set", "sha256"), ("model", "best_params"), ("lasso", "lambda_star")):
        a = (original.get(key[0]) or {}).get(key[1])
        b = (new.metrics.get(key[0]) or {}).get(key[1])
        if a != b:
            diffs.append({"metric": ".".join(key), "original": a, "reproduced": b})
    pred_a = hashlib.sha256(pd.read_parquet(run_dir / "predictions_test.parquet").to_csv(index=False, lineterminator="\n").encode()).hexdigest()
    pred_b = hashlib.sha256(pd.read_parquet(new.run_dir / "predictions_test.parquet").to_csv(index=False, lineterminator="\n").encode()).hexdigest()
    if pred_a != pred_b:
        diffs.append({"metric": "predictions_test.parquet", "original": pred_a, "reproduced": pred_b})
    report = {"original_run_id": original["run_id"], "reproduced_run_id": new.run_id, "tolerance": tolerance,
              "identical": not diffs, "differences": diffs,
              "code_version_original": original.get("code_version"), "code_version_reproduced": new.metrics.get("code_version")}
    write_json(new.run_dir / "reproduction_report.json", report)
    return report


def efalls_coverage(spec: FeatureSpec, available: Iterable[str], *, kind: str) -> dict[str, Any]:
    """eFalls predictor coverage (``metrics.efalls_coverage``) of ``available`` predictor names against ``spec``.

    ``spec`` is the root (full) feature spec: candidates are its ``exact_efalls_baseline`` predictors (78 for efalls_v1),
    published-retained predictors have ``retained_in_published_model: true`` (62) and mandatory predictors are
    ``cohort.mandatory_for_efalls_label``.
    """
    names = set(available)
    candidates = [f.name for f in spec.features if f.exact_efalls_baseline]
    retained = [f.name for f in spec.features if f.exact_efalls_baseline and f.retained_in_published_model is True]
    avail = [c for c in candidates if c in names]
    unavailable = [c for c in candidates if c not in names]
    n_total = len(candidates)
    full = not unavailable
    if kind == REDUCED_KIND:
        label = f"{REDUCED_WARNING} ({len(avail)}/{n_total} eFalls predictors available)"
    elif full:
        label = f"Full eFalls predictor set ({len(avail)}/{n_total})"
    else:
        label = f"Published eFalls equation with {len(avail)}/{n_total} predictors available (unavailable predictors handled per M-11)"
    return {"n_available": len(avail), "n_total": n_total, "coverage_pct": round(100.0 * len(avail) / n_total, 1) if n_total else 0.0,
            "available": avail, "unavailable": unavailable,
            "n_published_retained_available": sum(r in names for r in retained), "n_published_retained_total": len(retained),
            "mandatory_unavailable": [c for c in (spec.cohort.get("mandatory_for_efalls_label") or []) if c not in names],
            "is_full_efalls_feature_set": full, "label": label}


def _slug(text: str) -> str:
    import re
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", text.lower())).strip("_") or "step"

