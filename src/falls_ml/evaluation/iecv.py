"""Internal–external cross-validation by cluster (spec §7.3, D-15, D-16, D-21).

Each cycle repeats the full development process (``fit_fn``) on all clusters but one and validates on the
omitted cluster; rows with a missing cluster value form their own cluster ``'missing'`` (as the published
missing-WIMD group). Within-cluster SEs follow D-21: DeLong variance of C on the logit scale by the delta method,
GLM Wald SEs for the calibration slope and CITL, sqrt((1 - φ)/O) for ln(O/E).

D-16 cluster rule: a cluster enters the meta-analysis only with >= ``min_events`` events, >= 1 non-event and a
converged calibration slope. Excluded clusters are kept in the per-cluster table with their reasons; if they
hold > 20% of the rows a warning is logged and a row with ``measure == 'note'`` (message in ``scale``, ``k`` =
number of excluded clusters) is added to the pooled table. Pooling uses
:func:`~falls_ml.evaluation.meta_analysis.pool_measure` (REML τ², modified HKSJ CI, t_{k-2} prediction interval;
logit C, log O/E, slope and CITL on the original scale); a measure with < 2 poolable clusters has NaN estimates.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from falls_ml.errors import ConfigError, FallsMLError, LeakageError
from falls_ml.evaluation.meta_analysis import POOLING_SCALES, pool_measure
from falls_ml.evaluation.metrics import (
    UndefinedMetricWarning,
    auroc,
    auroc_logit_se,
    calibration_slope_intercept,
    citl,
    log_oe_se,
    logit,
    oe_ratio,
)
from falls_ml.evaluation.stability import check_positive_int, checked_proba, is_single_class, outcome_array
from falls_ml.logging_utils import get_logger
from falls_ml.pipeline import FittedPipeline
from falls_ml.seeding import int_seed_for

log = get_logger(__name__)

MISSING_CLUSTER = "missing"
MAX_EXCLUDED_ROW_SHARE = 0.20
PER_CLUSTER_COLUMNS = ["cluster", "n", "n_events", "auroc", "auroc_se_logit", "calibration_slope", "calibration_slope_se", "citl",
                       "citl_se", "oe_ratio", "log_oe_se", "included", "exclusion_reason"]
POOLED_COLUMNS = ["measure", "scale", "mu", "ci_low", "ci_high", "pi_low", "pi_high", "tau2", "i2", "k"]
#: pooled measure -> (value column, SE column); variances are passed to pool_measure as it documents
POOLED_MEASURES = {"auroc": ("auroc", "auroc_se_logit"), "calibration_slope": ("calibration_slope", "calibration_slope_se"),
                   "citl": ("citl", "citl_se"), "oe_ratio": ("oe_ratio", "log_oe_se")}


class IECVError(FallsMLError):
    """Internal–external cross-validation cannot be run on the given clusters."""


def _cluster_labels(df: pd.DataFrame, cluster_column: str) -> tuple[np.ndarray, list[str]]:
    """(label per row, clusters): clusters in sorted value order with ``'missing'`` (null values) last.

    Distinct values whose labels coincide (e.g. ``1`` and ``'1'``, or a literal ``'missing'`` next to nulls) are
    refused rather than silently merged.
    """
    if cluster_column not in df.columns:
        raise ConfigError(f"IECV cluster column {cluster_column!r} is not in the frame")
    codes, uniques = pd.factorize(df[cluster_column], sort=True)
    names = [str(u) for u in uniques]
    has_missing = bool((codes < 0).any())
    if len(set(names)) != len(names) or (has_missing and MISSING_CLUSTER in names):
        raise ConfigError(f"IECV cluster column {cluster_column!r}: distinct cluster values share a label "
                          f"(duplicated string forms, or a literal {MISSING_CLUSTER!r} cluster next to null values)")
    clusters = names + ([MISSING_CLUSTER] if has_missing else [])
    labels = np.asarray(clusters, dtype=object)[codes]  # code -1 (null) selects the last entry, 'missing'
    return labels, clusters


def _check_patients_disjoint(df: pd.DataFrame, labels: np.ndarray, spec: Any, cluster_column: str) -> None:
    """D-15: a patient must not appear in the held-out cluster and in the training clusters of the same cycle."""
    id_col = spec.identifier_columns[0] if spec.identifier_columns else None
    if id_col is None or id_col == cluster_column or id_col not in df.columns:
        return
    n_clusters = pd.DataFrame({"id": df[id_col].to_numpy(), "cluster": labels}).groupby("id")["cluster"].nunique()
    straddling = n_clusters[n_clusters > 1]
    if len(straddling):
        raise LeakageError(f"{len(straddling)} patients have rows in more than one {cluster_column!r} cluster "
                           f"(e.g. {list(straddling.index[:5])}); IECV clusters must be patient-disjoint (D-15)")


def _evaluate_cluster(y: np.ndarray, p: np.ndarray, lp: np.ndarray | None, min_events: int) -> dict[str, Any]:
    lp_eff = logit(p) if lp is None else np.asarray(lp, dtype=np.float64)
    n_events = int(y.sum())
    with warnings.catch_warnings():  # undefined measures are NaN and recorded through the exclusion rule
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        _, slope, _, slope_se = calibration_slope_intercept(y, lp_eff)
        citl_value, citl_se = citl(y, lp_eff)
        row = {"n": int(y.size), "n_events": n_events, "auroc": auroc(y, p), "auroc_se_logit": auroc_logit_se(y, p),
               "calibration_slope": slope, "calibration_slope_se": slope_se, "citl": citl_value, "citl_se": citl_se,
               "oe_ratio": oe_ratio(y, p), "log_oe_se": log_oe_se(y, p)}
    reasons = []
    if n_events < min_events:
        reasons.append(f"fewer than {min_events} events ({n_events})")
    if n_events == y.size:
        reasons.append("no non-events")
    if not (math.isfinite(slope) and math.isfinite(slope_se)):
        reasons.append("calibration slope did not converge")
    row.update(included=not reasons, exclusion_reason="; ".join(reasons))
    return row


def _pooled_row(per_cluster: pd.DataFrame, measure: str) -> dict[str, Any]:
    value_col, se_col = POOLED_MEASURES[measure]
    inc = per_cluster[per_cluster["included"]]
    values = inc[value_col].to_numpy(dtype=np.float64)
    se = inc[se_col].to_numpy(dtype=np.float64)
    variances = (se * values * (1.0 - values)) ** 2 if measure == "auroc" else se**2  # auroc: DeLong variance, original scale
    valid = np.isfinite(values) & np.isfinite(variances) & (variances > 0)
    if measure == "auroc":
        valid &= (values > 0) & (values < 1)
    elif measure == "oe_ratio":
        valid &= values > 0
    if (~valid).any():
        log.warning("iecv_measure_undefined_in_included_clusters", extra_fields={
            "measure": measure, "clusters": inc.loc[~valid, "cluster"].tolist()})
    k = int(valid.sum())
    row: dict[str, Any] = dict.fromkeys(POOLED_COLUMNS, math.nan) | {"measure": measure, "scale": POOLING_SCALES[measure][0], "k": k}
    if k < 2:
        log.warning("iecv_measure_not_pooled", extra_fields={"measure": measure, "k": k, "reason": "fewer than 2 clusters"})
        return row
    res = pool_measure(values[valid], variances[valid], measure)
    return row | {"mu": res.mu, "ci_low": res.ci_low, "ci_high": res.ci_high,
                  "pi_low": math.nan if res.pi_low is None else res.pi_low,
                  "pi_high": math.nan if res.pi_high is None else res.pi_high, "tau2": res.tau2, "i2": res.i2}


def internal_external_cv(df: pd.DataFrame, fit_fn: Callable[[pd.DataFrame, int], FittedPipeline], *, cluster_column: str, spec: Any,
                         min_events: int = 10, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """IECV: (per-cluster table :data:`PER_CLUSTER_COLUMNS`, pooled table :data:`POOLED_COLUMNS`) (spec §7.3, D-16)."""
    min_events = check_positive_int(min_events, "min_events")
    outcome = spec.outcome.name
    y_all = outcome_array(df, outcome)
    labels, clusters = _cluster_labels(df, cluster_column)
    if len(clusters) < 2:
        raise IECVError(f"IECV needs at least 2 clusters in {cluster_column!r}, found {clusters}")
    _check_patients_disjoint(df, labels, spec, cluster_column)

    rows = []
    for cluster in clusters:
        held_out = labels == cluster
        y_train = y_all[~held_out]
        if is_single_class(y_train):
            raise IECVError(f"IECV training clusters without {cluster!r} contain a single outcome class")
        test = df.loc[held_out]
        pipeline = fit_fn(df.loc[~held_out], int_seed_for(seed, f"iecv:fit:{cluster}"))
        row = _evaluate_cluster(y_all[held_out], checked_proba(pipeline, test), pipeline.linear_predictor(test), min_events)
        rows.append({"cluster": cluster, **row})
        log.info("iecv_cluster_done", extra_fields={"cluster": cluster, "n": row["n"], "n_events": row["n_events"],
                                                   "included": row["included"], "exclusion_reason": row["exclusion_reason"]})
    per_cluster = pd.DataFrame(rows, columns=PER_CLUSTER_COLUMNS)
    per_cluster["included"] = per_cluster["included"].astype(bool)

    pooled_rows = [_pooled_row(per_cluster, measure) for measure in POOLED_MEASURES]
    excluded = per_cluster[~per_cluster["included"]]
    share = float(excluded["n"].sum()) / len(df)
    if share > MAX_EXCLUDED_ROW_SHARE:
        message = (f"excluded clusters hold {share:.1%} of rows (> {MAX_EXCLUDED_ROW_SHARE:.0%}): "
                   f"{len(excluded)} of {len(per_cluster)} clusters excluded; D-16 recommends a coarser cluster unit")
        log.warning("iecv_excluded_share_high", extra_fields={"share": share, "excluded_clusters": excluded["cluster"].tolist()})
        pooled_rows.append(dict.fromkeys(POOLED_COLUMNS, math.nan) | {"measure": "note", "scale": message, "k": len(excluded)})
    pooled = pd.DataFrame(pooled_rows, columns=POOLED_COLUMNS)
    pooled["k"] = pooled["k"].astype("Int64")
    log.info("iecv_done", extra_fields={"n_clusters": len(per_cluster), "n_included": int(per_cluster["included"].sum()),
                                       "excluded_row_share": share})
    return per_cluster, pooled
