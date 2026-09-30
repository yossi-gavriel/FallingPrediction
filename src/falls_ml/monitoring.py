"""Drift monitoring against the training reference profile. Monitoring never retrains (architecture §2, §5).

``build_reference_profile`` summarises the training rows (aggregates only; numeric quantile edges exclude the
minimum and maximum) and stores the held-out reference performance of the served prediction when given.
``drift_report`` compares new rows with that profile: missing-rate changes, feature and category drift (population
stability index, PSI), prediction drift and, when outcomes are supplied, prevalence drift, discrimination (AUROC drop
against the reference performance) and calibration overall and by calendar quarter. It only reports: any retraining
must go through the explicit, approved retraining workflow.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from falls_ml.artifacts import write_csv, write_json
from falls_ml.data.schema import validate_modeling_dataset
from falls_ml.errors import ConfigError, DatasetValidationError
from falls_ml.evaluation import metrics as M
from falls_ml.features.spec import FeatureDefinition, FeatureSpec
from falls_ml.features.transforms import MISSING_LEVEL
from falls_ml.logging_utils import get_logger
from falls_ml.pipeline import FittedPipeline

log = get_logger(__name__)

PROFILE_SCHEMA_VERSION = "1.0"
PSI_EPSILON = 1e-4
MIN_EVENTS_FOR_METRICS = 10  # cells with < 10 events or non-events report null metrics (ARTIFACT_SCHEMAS subgroups)
CALIBRATION_SLOPE_RANGE = (0.8, 1.2)  # mirrors reporting.eligibility defaults
MAX_ABS_CITL = 0.2
SYNTHETIC_WATERMARK = "SYNTHETIC FIXTURE — SOFTWARE TEST OUTPUT, NOT SCIENTIFIC EVIDENCE"
RECOMMENDATION = ("No automatic retraining is performed. Review the drift report; any retraining must follow the explicit, "
                  "approved retraining workflow.")
FEATURE_DRIFT_COLUMNS = ["feature", "kind", "psi", "missing_rate_reference", "missing_rate_current", "status"]
REFERENCE_PERFORMANCE_KEYS = ("auroc", "calibration_slope", "citl", "oe_ratio", "prevalence")
REFERENCE_PERFORMANCE_OPTIONAL_KEYS = ("split", "variant")  # descriptive provenance, e.g. "test" / "recalibrated"
_SEVERITY = {"ok": 0, "warn": 1, "alert": 2}
_TOLERANCE = 1e-12  # threshold comparisons are inclusive; absorbs floating-point noise in differences such as 0.81 - 0.76


# ============================================================================ helpers
def population_stability_index(expected: Any, actual: Any, *, epsilon: float = PSI_EPSILON) -> float:
    """PSI = Σ (a − e)·ln(a / e) over bins; proportions are floored at ``epsilon`` and renormalised."""
    e = np.maximum(np.asarray(expected, dtype="float64"), epsilon)
    a = np.maximum(np.asarray(actual, dtype="float64"), epsilon)
    if e.shape != a.shape or e.ndim != 1 or e.size == 0:
        raise ValueError(f"PSI needs two equally long, non-empty proportion vectors, got {e.shape} and {a.shape}")
    e, a = e / e.sum(), a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def _kind(f: FeatureDefinition) -> str:
    return "binary" if f.is_binary else "categorical" if f.is_categorical else "numeric"


def _bin_proportions(values: np.ndarray, edges: list[float]) -> list[float]:
    """Proportions of all rows per bin ``(-inf, e1), [e1, e2), …, [ek, inf)``; missing values are not counted."""
    present = values[~np.isnan(values)]
    counts = np.bincount(np.searchsorted(np.asarray(edges, dtype="float64"), present, side="right"), minlength=len(edges) + 1)
    return (counts / values.size).tolist()


def _interior_quantile_edges(values: np.ndarray, n_bins: int) -> list[float]:
    present = values[~np.isnan(values)]
    if present.size == 0:
        return []
    return np.unique(np.quantile(present, np.linspace(0.0, 1.0, n_bins + 1)[1:-1])).tolist()


def _category_frequencies(s: pd.Series, f: FeatureDefinition) -> dict[str, float]:
    obj = s.astype(object)
    freq = {level: float((obj == level).mean()) for level in f.levels}
    freq[MISSING_LEVEL] = float(s.isna().mean())
    return freq


def _feature_distribution(f: FeatureDefinition, s: pd.Series, edges: list[float] | None = None, n_bins: int = 10) -> dict[str, Any]:
    kind = _kind(f)
    out: dict[str, Any] = {"kind": kind, "missing_rate": float(s.isna().mean())}
    if kind == "binary":
        out["prevalence"] = float(s.to_numpy(dtype="float64", na_value=np.nan).mean())
    elif kind == "categorical":
        out["frequencies"] = _category_frequencies(s, f)
    else:
        values = s.to_numpy(dtype="float64", na_value=np.nan)
        out["quantile_edges"] = _interior_quantile_edges(values, n_bins) if edges is None else list(edges)
        out["proportions"] = _bin_proportions(values, out["quantile_edges"])
    return out


def _distribution_vector(d: Mapping[str, Any], categories: list[str] | None = None) -> list[float]:
    """Proportions whose bins sum to 1: binary {0, 1}; categories incl. missing; numeric bins + a missing bin."""
    if d["kind"] == "binary":
        return [1.0 - d["prevalence"], d["prevalence"]]
    if d["kind"] == "categorical":
        return [d["frequencies"][k] for k in (categories or list(d["frequencies"]))]
    return [*d["proportions"], d["missing_rate"]]


def _status(psi: float | None, psi_warn: float, psi_alert: float, extra_warn: bool = False) -> str:
    if psi is not None and math.isfinite(psi) and psi >= psi_alert:
        return "alert"
    if (psi is not None and math.isfinite(psi) and psi >= psi_warn) or extra_warn:
        return "warn"
    return "ok"


def _uncalibrated_risk(pipeline: FittedPipeline, design: pd.DataFrame) -> np.ndarray:
    return np.asarray(pipeline.model.predict_proba(design), dtype="float64")


def _check_reference_performance(performance: Any) -> dict[str, Any] | None:
    """Validated copy of a reference performance block (``None`` passes through). Metric values may be null (undefined)."""
    if performance is None:
        return None
    if not isinstance(performance, Mapping):
        raise ConfigError(f"reference_performance must be a mapping, got {type(performance).__name__}")
    missing = [k for k in REFERENCE_PERFORMANCE_KEYS if k not in performance]
    unknown = sorted(set(performance) - set(REFERENCE_PERFORMANCE_KEYS) - set(REFERENCE_PERFORMANCE_OPTIONAL_KEYS))
    if missing or unknown:
        raise ConfigError(f"reference_performance needs keys {list(REFERENCE_PERFORMANCE_KEYS)} (optional "
                          f"{list(REFERENCE_PERFORMANCE_OPTIONAL_KEYS)}); missing {missing}, unknown {unknown}")
    out: dict[str, Any] = {}
    for key in REFERENCE_PERFORMANCE_KEYS:
        value = performance[key]
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating))
                                  or not math.isfinite(value)):
            raise ConfigError(f"reference_performance[{key!r}] must be a finite number or null, got {value!r}")
        out[key] = None if value is None else float(value)
    if out["auroc"] is not None and not 0.0 <= out["auroc"] <= 1.0:
        raise ConfigError(f"reference_performance['auroc'] must lie in [0, 1], got {out['auroc']}")
    if out["prevalence"] is not None and not 0.0 < out["prevalence"] < 1.0:
        raise ConfigError(f"reference_performance['prevalence'] must lie in (0, 1), got {out['prevalence']}")
    for key in REFERENCE_PERFORMANCE_OPTIONAL_KEYS:
        if key in performance:
            if not isinstance(performance[key], str):
                raise ConfigError(f"reference_performance[{key!r}] must be a string, got {performance[key]!r}")
            out[key] = performance[key]
    return out


# ============================================================================ reference profile
def build_reference_profile(train_df: pd.DataFrame, spec: FeatureSpec, pipeline: FittedPipeline, *, n_bins: int = 10,
                            reference_performance: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Training reference distributions for monitoring and inference (aggregates only; nothing is fitted).

    ``reference_performance`` = held-out performance of the served prediction (keys ``REFERENCE_PERFORMANCE_KEYS``,
    optionally ``split``/``variant``); ``drift_report`` compares labelled monitoring data with it.
    """
    if isinstance(n_bins, bool) or not isinstance(n_bins, int) or n_bins < 2:
        raise ConfigError(f"n_bins must be an integer >= 2, got {n_bins!r}")
    reference_performance = _check_reference_performance(reference_performance)
    if len(train_df) == 0:
        raise DatasetValidationError("build_reference_profile received zero rows")
    missing = [f.name for f in spec.features if f.name not in train_df.columns]
    if missing:
        raise DatasetValidationError("Reference rows are missing predictor columns", [str(missing)])
    features = {f.name: _feature_distribution(f, train_df[f.name], n_bins=n_bins) for f in spec.features}
    design = pipeline.design(train_df)  # the fitted preprocessor; nothing is refitted
    risk = _uncalibrated_risk(pipeline, design)
    risk_edges = np.unique(np.quantile(risk, np.linspace(0.0, 1.0, 11)[1:-1])).tolist()
    outcome = spec.outcome.name
    poly = "polypharmacy_count_120d"
    profile = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "n_rows": int(len(train_df)),
        "n_bins": n_bins,
        "feature_spec": {"name": spec.name, "version": spec.version, "sha256": spec.content_sha256},
        "pipeline": {"model": pipeline.model.name, "preprocessor_fingerprint": pipeline.preprocessor.fingerprint()},
        "features": features,
        "predictions": {"kind": "risk_uncalibrated", "decile_edges": risk_edges,
                        "proportions": _bin_proportions(risk, risk_edges), "mean": float(risk.mean())},
        "outcome_prevalence": float(train_df[outcome].mean()) if outcome in train_df.columns else None,
        "reference_performance": reference_performance,
        "design_column_means": {c: float(v) for c, v in design.mean(axis=0).items()},
        "polypharmacy_max": float(train_df[poly].max()) if poly in train_df.columns else None,
        "synthetic_training_data": bool("source" in train_df.columns and (train_df["source"] == "synthetic_fixture").any()),
    }
    log.info("reference_profile_built", extra_fields={"n_rows": profile["n_rows"], "n_features": len(features)})
    return profile


# ============================================================================ drift report
@dataclass(frozen=True)
class DriftReport:
    feature_drift: pd.DataFrame
    prediction_drift: dict[str, Any] | None
    performance: dict[str, Any] | None
    overall_status: str
    recommendation: str = RECOMMENDATION
    settings: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        rows = self.feature_drift.astype(object).where(self.feature_drift.notna(), None).to_dict(orient="records")
        return {"overall_status": self.overall_status, "recommendation": self.recommendation, "settings": self.settings,
                "feature_drift": rows, "prediction_drift": self.prediction_drift, "performance": self.performance}


def _metric_block(y: np.ndarray, p: np.ndarray, lp: np.ndarray | None) -> dict[str, Any]:
    n, n_events = int(y.size), int(y.sum())
    out: dict[str, Any] = {"n": n, "n_events": n_events, "observed_rate": float(y.mean()) if n else None,
                           "mean_predicted": float(p.mean()) if n else None,
                           "auroc": None, "citl": None, "calibration_slope": None, "oe_ratio": None}
    if n_events < MIN_EVENTS_FOR_METRICS or n - n_events < MIN_EVENTS_FOR_METRICS:
        return out
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", M.UndefinedMetricWarning)  # undefined values are reported as null
        values = {"auroc": M.auroc(y, p), "citl": M.citl(y, lp if lp is not None else M.logit(p))[0],
                  "calibration_slope": M.calibration_slope_intercept(y, lp if lp is not None else M.logit(p))[1],
                  "oe_ratio": M.oe_ratio(y, p)}
    out.update({k: (float(v) if math.isfinite(v) else None) for k, v in values.items()})
    return out


def _discrimination(reference: Mapping[str, Any] | None, auroc: float | None, drop_warn: float, drop_alert: float) -> dict[str, Any]:
    ref = None if reference is None else reference["auroc"]
    if ref is None or auroc is None:
        return {"auroc_reference": ref, "auroc_drop": None, "discrimination_status": "not_assessed"}
    drop = ref - auroc
    status = "alert" if drop >= drop_alert - _TOLERANCE else "warn" if drop >= drop_warn - _TOLERANCE else "ok"
    return {"auroc_reference": ref, "auroc_drop": drop, "discrimination_status": status}


def _prevalence(reference_profile: Mapping[str, Any], reference: Mapping[str, Any] | None, current: float, psi_warn: float,
                psi_alert: float, abs_warn: float, rel_warn: float) -> dict[str, Any]:
    """Prevalence drift against the reference performance prevalence (else the training prevalence): PSI and change rules."""
    if reference is not None and reference["prevalence"] is not None:
        ref, source = reference["prevalence"], "reference_performance"
    else:
        ref, source = reference_profile.get("outcome_prevalence"), "training_rows"
    if ref is None:
        return {"prevalence_reference": None, "prevalence_reference_source": None, "prevalence_current": current,
                "prevalence_psi": None, "prevalence_abs_change": None, "prevalence_rel_change": None, "prevalence_status": "ok"}
    psi = population_stability_index([1 - ref, ref], [1 - current, current])
    abs_change = abs(current - ref)
    rel_change = abs_change / ref if ref > 0 else None
    changed = abs_change >= abs_warn - _TOLERANCE or (rel_change is not None and rel_change >= rel_warn - _TOLERANCE)
    return {"prevalence_reference": ref, "prevalence_reference_source": source, "prevalence_current": current,
            "prevalence_psi": psi, "prevalence_abs_change": abs_change, "prevalence_rel_change": rel_change,
            "prevalence_status": _status(psi, psi_warn, psi_alert, extra_warn=changed)}


def _performance(reference_profile: Mapping[str, Any], reference: Mapping[str, Any] | None, df: pd.DataFrame,
                 design: pd.DataFrame, spec: FeatureSpec, pipeline: FittedPipeline, labels: Any,
                 thresholds: Mapping[str, float]) -> dict[str, Any]:
    y = np.asarray(labels)
    if y.shape != (len(df),) or not np.isin(y, (0, 1)).all():
        raise DatasetValidationError(f"labels must be {len(df)} binary 0/1 outcomes aligned with current_df")
    y = y.astype("int64")
    p = _uncalibrated_risk(pipeline, design)
    lp = pipeline.model.linear_predictor(design)
    calibrated = pipeline.calibrator is not None and pipeline.calibrator.method != "none"
    if calibrated:
        p, lp = np.asarray(pipeline.calibrator.transform(p, lp=lp), dtype="float64"), None
    out = {"variant": "recalibrated" if calibrated else "uncalibrated", **_metric_block(y, p, lp)}

    slope, citl = out["calibration_slope"], out["citl"]
    miscalibrated = (slope is not None and not CALIBRATION_SLOPE_RANGE[0] <= slope <= CALIBRATION_SLOPE_RANGE[1]) \
        or (citl is not None and abs(citl) > MAX_ABS_CITL)
    out.update({"reference_performance": reference,
                **_prevalence(reference_profile, reference, float(y.mean()), thresholds["psi_warn"], thresholds["psi_alert"],
                              thresholds["prevalence_abs_warn"], thresholds["prevalence_rel_warn"]),
                **_discrimination(reference, out["auroc"], thresholds["auroc_drop_warn"], thresholds["auroc_drop_alert"]),
                "calibration_status": "warn" if miscalibrated else "ok" if slope is not None else "not_assessed"})
    periods = df[spec.index_column].dt.to_period("Q").astype(str).to_numpy()
    out["by_period"] = [{"period": period, **_metric_block(y[periods == period], p[periods == period],
                                                          None if lp is None else lp[periods == period])}
                        for period in sorted(set(periods))]
    return out


def drift_report(reference_profile: Mapping[str, Any], current_df: pd.DataFrame, spec: FeatureSpec, *,
                 pipeline: FittedPipeline | None = None, labels: Any = None, psi_warn: float = 0.1, psi_alert: float = 0.25,
                 missing_rate_abs_warn: float = 0.05, auroc_drop_warn: float = 0.02, auroc_drop_alert: float = 0.05,
                 prevalence_abs_warn: float = 0.02, prevalence_rel_warn: float = 0.25) -> DriftReport:
    """Compare ``current_df`` with the training reference profile. Reports only; never fits or retrains anything.

    With ``labels``: AUROC drop versus ``reference_performance`` >= ``auroc_drop_alert`` is an alert and >=
    ``auroc_drop_warn`` a warning; an absolute prevalence change >= ``prevalence_abs_warn`` or a relative change >=
    ``prevalence_rel_warn`` is a warning. Both enter ``overall_status``.
    """
    if not 0 < psi_warn < psi_alert or not 0 <= missing_rate_abs_warn <= 1:
        raise ConfigError("drift thresholds need 0 < psi_warn < psi_alert and 0 <= missing_rate_abs_warn <= 1")
    if not 0 < auroc_drop_warn < auroc_drop_alert <= 1 or not 0 < prevalence_abs_warn < 1 or not 0 < prevalence_rel_warn:
        raise ConfigError("performance thresholds need 0 < auroc_drop_warn < auroc_drop_alert <= 1, 0 < prevalence_abs_warn < 1 "
                          "and prevalence_rel_warn > 0")
    if labels is not None and pipeline is None:
        raise ConfigError("performance monitoring with labels requires the bundle pipeline")
    profile_spec = (reference_profile.get("feature_spec") or {}).get("sha256")
    if profile_spec != spec.content_sha256:
        raise ConfigError("reference profile was built with a different feature spec (SHA-256 mismatch or not recorded)")
    reference_performance = _check_reference_performance(reference_profile.get("reference_performance"))
    if pipeline is not None:
        recorded = reference_profile.get("pipeline") or {}
        current = {"model": pipeline.model.name, "preprocessor_fingerprint": pipeline.preprocessor.fingerprint()}
        if recorded != current:
            raise ConfigError(f"reference profile was built with a different pipeline ({recorded} vs {current}); "
                              "prediction and performance drift would be compared with the wrong baseline")
    reference = reference_profile.get("features") or {}
    unknown = sorted(set(spec.predictor_names()) ^ set(reference))
    if unknown:
        raise ConfigError(f"reference profile and feature spec declare different features: {unknown[:10]}")
    validate_modeling_dataset(current_df, spec, mode="inference")
    df = current_df.reset_index(drop=True)

    rows = []
    for f in spec.features:
        ref = reference[f.name]
        if ref["kind"] != _kind(f):
            raise ConfigError(f"{f.name}: reference profile kind {ref['kind']!r} differs from spec kind {_kind(f)!r}")
        cur = _feature_distribution(f, df[f.name], edges=ref.get("quantile_edges"))
        categories = list(ref["frequencies"]) if ref["kind"] == "categorical" else None
        psi = population_stability_index(_distribution_vector(ref, categories), _distribution_vector(cur, categories))
        delta_missing = abs(cur["missing_rate"] - ref["missing_rate"])
        rows.append({"feature": f.name, "kind": ref["kind"], "psi": psi, "missing_rate_reference": ref["missing_rate"],
                     "missing_rate_current": cur["missing_rate"],
                     "status": _status(psi, psi_warn, psi_alert, extra_warn=delta_missing > missing_rate_abs_warn)})
    feature_drift = pd.DataFrame(rows, columns=FEATURE_DRIFT_COLUMNS)

    prediction_drift, design = None, None
    if pipeline is not None:
        ref_pred = reference_profile.get("predictions")
        if not ref_pred:
            raise ConfigError("reference profile has no prediction distribution")
        design = pipeline.design(df)  # the bundle's fitted preprocessor; never refitted
        risk = _uncalibrated_risk(pipeline, design)
        proportions = _bin_proportions(risk, ref_pred["decile_edges"])
        psi = population_stability_index(ref_pred["proportions"], proportions)
        prediction_drift = {"kind": ref_pred["kind"], "psi": psi, "mean_reference": ref_pred["mean"],
                            "mean_current": float(risk.mean()), "proportions_current": proportions,
                            "status": _status(psi, psi_warn, psi_alert)}

    thresholds = {"psi_warn": psi_warn, "psi_alert": psi_alert, "auroc_drop_warn": auroc_drop_warn,
                  "auroc_drop_alert": auroc_drop_alert, "prevalence_abs_warn": prevalence_abs_warn,
                  "prevalence_rel_warn": prevalence_rel_warn}
    performance = None if labels is None else _performance(reference_profile, reference_performance, df, design, spec, pipeline,
                                                           labels, thresholds)

    statuses = feature_drift["status"].tolist()
    if prediction_drift is not None:
        statuses.append(prediction_drift["status"])
    if performance is not None:
        statuses += [performance["prevalence_status"], "warn" if performance["calibration_status"] == "warn" else "ok",
                     performance["discrimination_status"] if performance["discrimination_status"] in _SEVERITY else "ok"]
    overall = max(statuses, key=_SEVERITY.__getitem__, default="ok")
    settings = {"n_rows_current": int(len(df)), "n_rows_reference": reference_profile.get("n_rows"), **thresholds,
                "missing_rate_abs_warn": missing_rate_abs_warn, "psi_epsilon": PSI_EPSILON,
                "synthetic_training_data": bool(reference_profile.get("synthetic_training_data", False))}
    log.info("drift_report", extra_fields={"overall_status": overall, "n_rows": len(df),
                                           "n_features_alert": int((feature_drift["status"] == "alert").sum()),
                                           "n_features_warn": int((feature_drift["status"] == "warn").sum())})
    return DriftReport(feature_drift=feature_drift, prediction_drift=prediction_drift, performance=performance,
                       overall_status=overall, recommendation=RECOMMENDATION, settings=settings)


# ============================================================================ writing
def _fmt(value: Any) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def _markdown(report: DriftReport) -> str:
    lines = []
    if report.settings.get("synthetic_training_data"):
        lines += [SYNTHETIC_WATERMARK, ""]
    lines += ["# Drift monitoring report", "", f"**Overall status:** {report.overall_status}", "",
              f"**Recommendation:** {report.recommendation}", "", "## Feature drift", "",
              "| " + " | ".join(FEATURE_DRIFT_COLUMNS) + " |", "|" + "---|" * len(FEATURE_DRIFT_COLUMNS)]
    lines += ["| " + " | ".join(_fmt(row[c]) for c in FEATURE_DRIFT_COLUMNS) + " |"
              for row in report.to_dict()["feature_drift"]]
    lines += ["", "## Prediction drift", ""]
    if report.prediction_drift is None:
        lines.append("Not assessed (no model pipeline supplied).")
    else:
        pdr = report.prediction_drift
        lines.append(f"PSI {_fmt(pdr['psi'])}; mean risk reference {_fmt(pdr['mean_reference'])}, current "
                     f"{_fmt(pdr['mean_current'])}; status **{pdr['status']}**.")
    lines += ["", "## Performance", ""]
    if report.performance is None:
        lines.append("Not assessed (no outcome labels supplied).")
    else:
        perf = report.performance
        keys = ["n", "n_events", "auroc", "citl", "calibration_slope", "oe_ratio"]
        lines += [f"Variant: {perf['variant']}; prevalence reference {_fmt(perf['prevalence_reference'])} "
                  f"({perf['prevalence_reference_source'] or 'not recorded'}), current {_fmt(perf['prevalence_current'])} "
                  f"(status **{perf['prevalence_status']}**); AUROC reference {_fmt(perf['auroc_reference'])}, drop "
                  f"{_fmt(perf['auroc_drop'])} (discrimination status **{perf['discrimination_status']}**); calibration status "
                  f"**{perf['calibration_status']}**.", "", "| period | " + " | ".join(keys) + " |", "|" + "---|" * (len(keys) + 1),
                  "| overall | " + " | ".join(_fmt(perf[k]) for k in keys) + " |"]
        lines += ["| " + row["period"] + " | " + " | ".join(_fmt(row[k]) for k in keys) + " |" for row in perf["by_period"]]
    return "\n".join(lines) + "\n"


def write_drift_report(report: DriftReport, out_dir: str | Path) -> list[Path]:
    """Write ``drift_report.json``, ``feature_drift.csv`` and ``drift_report.md``; returns the three paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = [write_json(out / "drift_report.json", report.to_dict()),
             write_csv(out / "feature_drift.csv", report.feature_drift)]
    md = out / "drift_report.md"
    md.write_text(_markdown(report), encoding="utf-8", newline="\n")
    paths.append(md)
    return paths
