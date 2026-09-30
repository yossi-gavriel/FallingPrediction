"""Master comparison of all runs on identical test rows (spec §1.1, §1.2, D-19 §5; ARTIFACT_SCHEMAS ``metrics.json``).

Only completed runs (``RUN_COMPLETE.json`` present) are read; other directories are listed as skipped. One row per
run: the served prediction (``metrics.primary_variant`` = ``served_variant``) on the test split. The published eFalls
scoring contributes the fixed published equation explicitly: primary ``lp_c_box_s3_1`` and, when scored, co-reported
``lp_a_table_s3_2`` (D-01). Runs are grouped into scientific reproduction, local retraining, local retraining on a
reduced eFalls predictor set (never a full eFalls reproduction), experimental algorithms, Meuhedet-enhanced experimental
algorithms (feature set not pure eFalls) and ablation members, which are never mixed in the narrative. Rows whose test rows or data file differ from the published eFalls reference (or the majority when no
published run exists) are flagged ``comparable=False``. Highlights and the recommendation use comparable primary rows
outside the ablation. Paired patient-level bootstrap differences against the published primary variant are written
to ``comparison_paired_differences.csv``.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from falls_ml.evaluation.bootstrap import paired_bootstrap_difference
from falls_ml.logging_utils import get_logger
from falls_ml.models.published_efalls import CO_REPORTED_VARIANT, PRIMARY_VARIANT
from falls_ml.reporting.ablation import check_same_rows, load_test_predictions
from falls_ml.reporting.report import (
    PARTIAL_LABEL, SYNTHETIC_BANNER, ArtifactError, Block, Heading, Paragraph, as_bool, bullets, dig, estimate, fmt, fmt_ci,
    read_optional_csv, table, write_document,
)

log = get_logger(__name__)

PUBLISHED_KIND = "efalls_published_scoring"
REDUCED_KIND = "efalls_retrained_reduced"
CATEGORIES = {PUBLISHED_KIND: "scientific_reproduction", "efalls_retrained": "local_retraining", REDUCED_KIND: "reduced_local_retraining",
              "alternative_model": "experimental_algorithm", "ablation_member": "ablation"}
CATEGORY_TEXT = {
    "scientific_reproduction": ("Scientific reproduction: ORIGINAL published eFalls equation",
                                "Transportability of the fixed published equation; primary lp_c_box_s3_1 and co-reported "
                                "lp_a_table_s3_2 (D-01). A run labelled efalls_partial_scoring is not a validation of eFalls (M-11)."),
    "local_retraining": ("Local retraining of eFalls",
                         "eFalls predictors and published learning process with coefficients fitted locally."),
    "reduced_local_retraining": ("Local retraining on a reduced eFalls predictor set (not a full eFalls reproduction)",
                                 "Published eFalls learning process with coefficients fitted locally on a declared SUBSET of the eFalls "
                                 "candidate predictors (see the eFalls coverage column). REDUCED eFalls predictor set – NOT a full "
                                 "eFalls reproduction; never reported as eFalls."),
    "experimental_algorithm": ("Experimental algorithms", "Alternative algorithms (layer L4) on the pure eFalls feature set; not eFalls."),
    "meuhedet_enhanced": ("Meuhedet-enhanced experimental algorithms",
                          "Alternative algorithms whose feature set adds Meuhedet-specific predictors (layer L3b); not eFalls."),
    "ablation": ("Ablation members", "Experimental algorithms with added feature groups; not eFalls. Never highlighted or recommended here "
                                     "(see the ablation report)."),
}
SIMPLICITY_ORDER = ("efalls_published", "logistic_unpenalized", "lasso_logistic_cv", "elastic_net_logistic", "random_forest",
                    "hist_gradient_boosting")
INTERPRETABILITY = {"efalls_published": "high (fixed published equation)", "logistic_unpenalized": "high (coefficients)",
                    "lasso_logistic_cv": "high (sparse coefficients)", "elastic_net_logistic": "high (penalised coefficients)",
                    "random_forest": "low (permutation importance only)", "hist_gradient_boosting": "low (permutation importance only)"}
METRIC_KEYS = {"AUROC": "auroc", "PR_AUC": "pr_auc", "Brier": "brier", "calibration_slope": "calibration_slope",
               "calibration_intercept": "calibration_intercept", "citl": "citl", "oe_ratio": "oe_ratio"}
COLUMNS = ["model", "experiment", "kind", "category", "run_id", "variant", "AUROC", "PR_AUC", "Brier", "calibration_slope",
           "calibration_intercept", "citl", "oe_ratio", "sensitivity_at_threshold", "specificity_at_threshold", "threshold",
           "number_of_features", "best_params", "train_time", "inference_time", "dataset_version", "split_strategy",
           "test_rows_sha256", "synthetic_fixture", "effective_experiment_label", "feature_set_name", "pure_efalls"]
EXTRA_COLUMNS = ["variant_role", "data_sha256", "AUROC_ci_low", "AUROC_ci_high", "net_benefit_at_threshold",
                 "net_benefit_treat_all_at_threshold", "robust_features", "eligible_for_further_validation", "comparable",
                 "is_efalls_published", "is_efalls_retrained", "best_calibrated", "best_discrimination", "simplest_within_margin",
                 "efalls_coverage"]
PAIRED_COLUMNS = ["run_id", "variant", "reference_variant", "metric", "estimate", "ci_low", "ci_high", "n_bootstrap"]
PAIRED_METRICS = ("auroc", "brier", "calibration_slope", "citl")
PAIRED_COMPONENT = "comparison_paired_vs_published_efalls"
_TOL = 1e-12


def load_run_metrics(runs_dir: str | Path) -> tuple[list[tuple[Path, dict[str, Any]]], list[str]]:
    """``([(run_dir, metrics)], skipped)`` for completed runs (``RUN_COMPLETE.json`` present) in sorted order.

    ``skipped`` names every other directory with the reason. ``ArtifactError`` when a completed run has no readable
    ``metrics.json`` or its run id disagrees with ``RUN_COMPLETE.json``.
    """
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"runs directory not found: {runs_dir}")
    runs, skipped = [], []
    for d in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        if not (d / "RUN_COMPLETE.json").is_file():
            skipped.append(f"{d.name} (incomplete: no RUN_COMPLETE.json" + ("" if (d / "metrics.json").is_file() else ", no metrics.json") + ")")
            continue
        try:
            m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
            done = json.loads((d / "RUN_COMPLETE.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactError(f"{d.name}: completed run has unreadable metrics.json or RUN_COMPLETE.json ({exc})") from exc
        if done.get("run_id") != m.get("run_id"):
            raise ArtifactError(f"{d.name}: RUN_COMPLETE.json run_id {done.get('run_id')!r} != metrics.json run_id {m.get('run_id')!r}")
        runs.append((d, m))
    if skipped:
        log.info("incomplete_runs_skipped", extra_fields={"runs_dir": str(runs_dir), "skipped": skipped})
    return runs, skipped


def check_not_mixed(runs: list[tuple[Path, dict[str, Any]]]) -> bool:
    """Return the common synthetic flag; ``ArtifactError`` (a ValueError) if synthetic and real runs are mixed."""
    flags = {bool(m.get("synthetic_fixture")) for _, m in runs}
    if len(flags) > 1:
        synthetic = [d.name for d, m in runs if m.get("synthetic_fixture")]
        raise ArtifactError(f"synthetic fixture runs cannot be compared with real-data runs (synthetic: {synthetic})")
    return flags.pop() if flags else False


def _variants(run_dir: Path, m: dict[str, Any], warnings: list[str]) -> list[tuple[str, str, dict[str, Any]]]:
    """``[(variant, role, test entry)]``: published scoring uses the fixed equation explicitly, other runs the served variant."""
    test = dig(m, "performance", "test", default={})
    if m["experiment"]["kind"] == PUBLISHED_KIND:
        if PRIMARY_VARIANT not in test:
            raise ArtifactError(f"{run_dir.name}: test performance for the published primary variant {PRIMARY_VARIANT!r} is missing")
        out = [(PRIMARY_VARIANT, "primary", test[PRIMARY_VARIANT])]
        if CO_REPORTED_VARIANT in test:
            out.append((CO_REPORTED_VARIANT, "co_reported", test[CO_REPORTED_VARIANT]))
        else:
            warnings.append(f"{m.get('run_id') or run_dir.name}: the co-reported variant {CO_REPORTED_VARIANT} was not scored on the test "
                            "split, so only the primary row is shown; D-01 requires both.")
        return out
    variant = m.get("primary_variant")
    if not variant or variant not in test:
        raise ArtifactError(f"{run_dir.name}: test performance for the served primary variant {variant!r} is missing")
    return [(str(variant), "primary", test[variant])]


def _category(run_dir: Path, m: dict[str, Any]) -> str:
    kind = dig(m, "experiment", "kind")
    if kind not in CATEGORIES:
        raise ArtifactError(f"{run_dir.name}: unknown experiment kind {kind!r}")
    if kind == "alternative_model" and dig(m, "feature_set", "pure_efalls") is False:
        return "meuhedet_enhanced"
    return CATEGORIES[kind]


def _resolve_threshold(entries: list[dict[str, Any]], threshold: float | None) -> float:
    if threshold is not None:
        if not 0.0 < float(threshold) < 1.0:
            raise ArtifactError(f"threshold must lie in (0, 1), got {threshold}")
        return float(threshold)
    sets = [{round(float(t["threshold"]), 10) for t in e.get("thresholds") or []} for e in entries]
    common = set.intersection(*sets) if sets else set()
    if not common:
        raise ArtifactError("no decision threshold is shared by all runs; pass threshold= explicitly")
    return min(common)


def _at_threshold(entry: dict[str, Any], t: float) -> dict[str, Any]:
    for row in entry.get("thresholds") or []:
        if abs(float(row["threshold"]) - t) < 1e-9:
            return row
    return {}


def _robust_count(run_dir: Path) -> float:
    fs = read_optional_csv(run_dir / "feature_stability.csv")
    return float("nan") if fs is None or "robust" not in fs.columns else float(as_bool(fs["robust"]).sum())


def _n_features(run_dir: Path, m: dict[str, Any]) -> float:
    """Raw predictors with non-zero weight (linear models, from coefficients.csv); all predictors for tree models."""
    if dig(m, "model", "is_linear") is False:
        return _num(dig(m, "feature_set", "n_features"))
    coef = read_optional_csv(run_dir / "coefficients.csv")
    if coef is None:
        raise ArtifactError(f"{run_dir.name}: linear model without coefficient rows in coefficients.csv")
    keep = (coef["design_column"].astype(str) != "_intercept") & (pd.to_numeric(coef["coefficient"], errors="coerce").fillna(0.0) != 0.0)
    for flag in ("relative_to_reference", "omitted"):
        if flag in coef.columns:
            keep &= ~as_bool(coef[flag])
    return float(coef.loc[keep, "feature"].nunique())


def _coverage_text(m: dict[str, Any]) -> str | None:
    """``"X/78"`` from ``metrics.efalls_coverage`` (eFalls-labelled runs); None otherwise."""
    cov = m.get("efalls_coverage")
    if not cov or cov.get("n_available") is None or cov.get("n_total") is None:
        return None
    return f"{int(cov['n_available'])}/{int(cov['n_total'])}"


def _train_time(m: dict[str, Any]) -> float:
    total = dig(m, "timing", "total_train_seconds")
    return _num(total if total is not None else dig(m, "timing", "fit_seconds"))


def _num(x: Any) -> float:
    return float("nan") if x is None else float(x)


def _build_rows(runs: list[tuple[Path, dict[str, Any]]], threshold: float | None) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    selected: list[tuple[Path, dict[str, Any], str, str, dict[str, Any]]] = []
    for run_dir, m in runs:
        _category(run_dir, m)
        selected += [(run_dir, m, variant, role, entry) for variant, role, entry in _variants(run_dir, m, warnings)]
    t = _resolve_threshold([e for *_, e in selected], threshold)
    rows = []
    for run_dir, m, variant, role, e in selected:
        at = _at_threshold(e, t)
        if not at:
            log.warning("comparison_threshold_missing", extra_fields={"run_id": m.get("run_id"), "threshold": t})
        prevalence = e.get("observed_rate")
        row = {"model": dig(m, "model", "name"), "experiment": dig(m, "experiment", "name"), "kind": m["experiment"]["kind"],
               "category": _category(run_dir, m), "run_id": m.get("run_id") or run_dir.name, "variant": variant,
               **{col: _num(estimate(e.get(key))) for col, key in METRIC_KEYS.items()},
               "sensitivity_at_threshold": _num(at.get("sensitivity")), "specificity_at_threshold": _num(at.get("specificity")),
               "threshold": t, "number_of_features": _n_features(run_dir, m),
               "best_params": json.dumps(dig(m, "model", "best_params") or {}, sort_keys=True, default=str),
               "train_time": _train_time(m), "inference_time": _num(dig(m, "timing", "predict_seconds_per_1000_rows")),
               "dataset_version": dig(m, "dataset", "dataset_version"), "split_strategy": dig(m, "split", "strategy"),
               "test_rows_sha256": dig(m, "split", "test_rows_sha256"), "synthetic_fixture": bool(m.get("synthetic_fixture")),
               "effective_experiment_label": dig(m, "published_scoring", "effective_experiment_label"),
               "feature_set_name": dig(m, "feature_set", "name"), "pure_efalls": dig(m, "feature_set", "pure_efalls"),
               "variant_role": role, "data_sha256": dig(m, "dataset", "data_sha256"),
               "AUROC_ci_low": _num(dig(e, "auroc", "ci_low")), "AUROC_ci_high": _num(dig(e, "auroc", "ci_high")),
               "net_benefit_at_threshold": _num(at.get("net_benefit")),
               "net_benefit_treat_all_at_threshold": float("nan") if prevalence is None else prevalence - (1 - prevalence) * t / (1 - t),
               "robust_features": _robust_count(run_dir),
               "eligible_for_further_validation": dig(m, "eligibility", "eligible_for_further_validation"),
               "efalls_coverage": _coverage_text(m), "_run_dir": str(run_dir),
               "_unmodified_published": _unmodified_published(m)}
        rows.append(row)
    df = pd.DataFrame(rows)
    order = {c: i for i, c in enumerate(CATEGORY_TEXT)}
    df = df.assign(_o=df["category"].map(order), _r=df["variant_role"] != "primary")
    return df.sort_values(["_o", "experiment", "run_id", "_r"], kind="stable").drop(columns=["_o", "_r"]).reset_index(drop=True), warnings


def _flag_comparable(df: pd.DataFrame) -> pd.Series:
    pairs = [(sha, data) if isinstance(sha, str) and isinstance(data, str) else None
             for sha, data in zip(df["test_rows_sha256"], df["data_sha256"])]
    published = [p for p, k in zip(pairs, df["kind"]) if k == PUBLISHED_KIND and p is not None]
    known = published or [p for p in pairs if p is not None]
    reference = Counter(known).most_common(1)[0][0] if known else None
    # rows without test-row or data hash can never be shown to share the test rows (D-19 §5)
    flags = pd.Series([p is not None and p == reference for p in pairs], index=df.index, dtype=bool)
    if not flags.all():
        log.warning("comparison_non_comparable_runs", extra_fields={"run_ids": df.loc[~flags, "run_id"].tolist(),
                                                                    "reference": None if reference is None else list(reference)})
    return flags


def _calibration_error(df: pd.DataFrame) -> pd.Series:
    return (df["calibration_slope"] - 1.0).abs() + df["citl"].abs()


def _simplicity(model: str) -> int:
    return SIMPLICITY_ORDER.index(model) if model in SIMPLICITY_ORDER else len(SIMPLICITY_ORDER)


def _candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Rows eligible for highlights and the recommendation: comparable primary rows outside the ablation."""
    return df.loc[df["comparable"] & (df["variant_role"] == "primary") & (df["category"] != "ablation")]


def _failures(row: pd.Series, best_auroc: float, best_brier: float, auroc_margin: float, brier_margin: float,
              slope_range: tuple[float, float]) -> list[str]:
    out = []
    if not row["AUROC"] >= best_auroc - auroc_margin - _TOL:
        out.append(f"AUROC {fmt(row['AUROC'])} is more than {auroc_margin} below the best ({fmt(best_auroc)})")
    if not row["Brier"] <= best_brier + brier_margin + _TOL:
        out.append(f"Brier score {fmt(row['Brier'], 4)} is more than {brier_margin} above the best ({fmt(best_brier, 4)})")
    if not slope_range[0] <= row["calibration_slope"] <= slope_range[1]:
        out.append(f"calibration slope {fmt(row['calibration_slope'])} is outside [{slope_range[0]}, {slope_range[1]}]")
    nb, nb_all = row["net_benefit_at_threshold"], row["net_benefit_treat_all_at_threshold"]
    if math.isnan(nb) or math.isnan(nb_all) or not nb > max(nb_all, 0.0):
        out.append(f"net benefit at threshold {row['threshold']:.2f} ({fmt(nb, 4)}) does not exceed both treat-all ({fmt(nb_all, 4)}) "
                   "and treat-none (0)")
    return out


def _highlight(df: pd.DataFrame, auroc_margin: float, brier_margin: float, slope_range: tuple[float, float]) -> pd.DataFrame:
    df = df.assign(is_efalls_published=df["kind"] == PUBLISHED_KIND, is_efalls_retrained=df["kind"] == "efalls_retrained",
                   best_calibrated=False, best_discrimination=False, simplest_within_margin=False)
    cand = _candidates(df)
    if cand["AUROC"].notna().any():
        df.loc[cand["AUROC"].idxmax(), "best_discrimination"] = True
    cal = _calibration_error(cand)
    if cal.notna().any():
        df.loc[cal.idxmin(), "best_calibrated"] = True
    if cand["AUROC"].notna().any() and cand["Brier"].notna().any():
        best_auroc, best_brier = cand["AUROC"].max(), cand["Brier"].min()
        ok = cand.loc[[not _failures(r, best_auroc, best_brier, auroc_margin, brier_margin, slope_range) for _, r in cand.iterrows()]]
        if not ok.empty:
            ranked = ok.assign(_s=ok["model"].map(_simplicity), _c=_calibration_error(ok), _a=-ok["AUROC"])
            df.loc[ranked.sort_values(["_s", "_c", "_a", "run_id"], kind="stable").index[0], "simplest_within_margin"] = True
    return df


# ============================================================================ paired differences
def _unmodified_published(m: dict[str, Any]) -> bool:
    """True for a published-scoring run of the unmodified equation: no unavailable predictors, default fill, no D-20 zeroing."""
    ps = m.get("published_scoring") or {}
    return (m["experiment"]["kind"] == PUBLISHED_KIND and not ps.get("unavailable_predictors")
            and ps.get("unavailable_fill") in (None, "zero") and not ps.get("low_support_zeroed"))


def _paired_reference(df: pd.DataFrame) -> pd.Series | None:
    """The comparable UNMODIFIED published run (efalls_published_scoring label, primary variant, full predictor coverage,
    default options); the latest by creation time (run id as tie-break) if several."""
    ref = df.loc[df["comparable"] & (df["kind"] == PUBLISHED_KIND) & (df["variant"] == PRIMARY_VARIANT)
                 & (df["effective_experiment_label"] == PUBLISHED_KIND) & df["_unmodified_published"].astype(bool)]
    ref = ref.sort_values("run_id", kind="stable")
    if len(ref) > 1:
        log.warning("comparison_several_published_references", extra_fields={"run_ids": ref["run_id"].tolist(),
                                                                             "chosen": ref["run_id"].iloc[-1]})
    return None if ref.empty else ref.iloc[-1]


def _served_risk_column(row: pd.Series, m: dict[str, Any]) -> str:
    if row["kind"] == PUBLISHED_KIND:
        return f"risk_{row['variant']}"
    served = m.get("served_variant") or m.get("primary_variant")
    return "risk_recalibrated" if "recalibrated" in str(served) else "risk_uncalibrated"


def _paired_differences(df: pd.DataFrame, metrics_by_dir: dict[str, dict[str, Any]], n_bootstrap: int,
                        seed: int) -> tuple[pd.DataFrame, pd.Series | None]:
    """Run minus published primary on identical test rows, for every other comparable primary row (and recalibrated A)."""
    ref = _paired_reference(df)
    if ref is None:
        log.warning("comparison_paired_differences_not_computed", extra_fields={"reason": "no comparable published eFalls run"})
        return pd.DataFrame(columns=PAIRED_COLUMNS), None
    t = float(df["threshold"].iloc[0])
    names = [*PAIRED_METRICS, f"net_benefit@{t:g}"]
    ref_dir, ref_m = ref["_run_dir"], metrics_by_dir[ref["_run_dir"]]
    targets: list[tuple[str, str, str, str]] = []
    recalibrated = f"{PRIMARY_VARIANT}+recalibrated"
    configured = ref_m.get("transportability_variant") or dig(ref_m, "published_scoring", "configured_variant")
    if dig(ref_m, "performance", "test", recalibrated) is not None and configured == PRIMARY_VARIANT:
        targets.append((ref["run_id"], recalibrated, ref_dir, "risk_recalibrated"))
    same = df.loc[df["comparable"] & (df["variant_role"] == "primary") & (df["_run_dir"] != ref_dir)
                  & (df["test_rows_sha256"] == ref["test_rows_sha256"]) & (df["data_sha256"] == ref["data_sha256"])]
    targets += [(r["run_id"], r["variant"], r["_run_dir"], _served_risk_column(r, metrics_by_dir[r["_run_dir"]])) for _, r in same.iterrows()]
    base = load_test_predictions(Path(ref_dir), [f"risk_{PRIMARY_VARIANT}"])
    y = base["outcome"].to_numpy(dtype=np.int64)
    cluster = base["research_id"].astype("string").to_numpy()
    p_ref = base[f"risk_{PRIMARY_VARIANT}"].to_numpy(dtype=float)
    rows = []
    for run_id, variant, run_dir, column in targets:
        pred = load_test_predictions(Path(run_dir), [column])
        check_same_rows(base, pred, f"paired comparison of {run_id} ({variant}) with published eFalls {PRIMARY_VARIANT}")
        p = pred[column].to_numpy(dtype=float)
        for metric in names:
            d = paired_bootstrap_difference(y, p, p_ref, metric=metric, n=n_bootstrap, seed=seed, component=PAIRED_COMPONENT, cluster=cluster)
            rows.append({"run_id": run_id, "variant": variant, "reference_variant": PRIMARY_VARIANT, "metric": metric,
                         "estimate": _num(d["estimate"]), "ci_low": _num(d["ci_low"]), "ci_high": _num(d["ci_high"]), "n_bootstrap": n_bootstrap})
    return pd.DataFrame(rows, columns=PAIRED_COLUMNS), ref


def build_comparison(runs_dir: str | Path, out_dir: str | Path, *, threshold: float | None = None, auroc_margin: float = 0.01,
                     brier_margin: float = 0.002, slope_range: tuple[float, float] = (0.8, 1.2), n_bootstrap: int = 1000,
                     seed: int = 1) -> pd.DataFrame:
    """Write ``comparison.csv|md|html`` and ``comparison_paired_differences.csv``; return the comparison table.

    ``threshold`` defaults to the smallest decision threshold reported by every run; it is also the net-benefit
    threshold of the paired differences (``n_bootstrap`` patient-clustered resamples, ``seed``). ``ValueError`` when
    synthetic and real runs are mixed or no completed run exists. Directories without ``RUN_COMPLETE.json`` are
    skipped and listed.
    """
    runs, skipped = load_run_metrics(runs_dir)
    if not runs:
        raise ArtifactError(f"no completed runs (RUN_COMPLETE.json) in {runs_dir}")
    if skipped:
        log.warning("comparison_incomplete_runs_skipped", extra_fields={"runs_dir": str(runs_dir), "skipped": skipped})
    synthetic = check_not_mixed(runs)
    df, warnings = _build_rows(runs, threshold)
    df["comparable"] = _flag_comparable(df)
    df = _highlight(df, auroc_margin, brier_margin, slope_range)
    paired, reference = _paired_differences(df, {str(d): m for d, m in runs}, n_bootstrap, seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table_df = df[COLUMNS + EXTRA_COLUMNS]
    table_df.to_csv(out_dir / "comparison.csv", index=False, encoding="utf-8", lineterminator="\n")
    paired.to_csv(out_dir / "comparison_paired_differences.csv", index=False, encoding="utf-8", lineterminator="\n")
    blocks = _document(table_df, paired, reference, warnings, skipped, synthetic, auroc_margin, brier_margin, slope_range, n_bootstrap, seed)
    write_document(blocks, out_dir / "comparison.md", out_dir / "comparison.html", title="Model comparison")
    log.info("comparison_written", extra_fields={"out_dir": str(out_dir), "n_rows": len(df), "n_skipped": len(skipped),
                                                 "n_paired_rows": len(paired)})
    return table_df


# ============================================================================ document
def _label(row: pd.Series) -> str:
    variant = f", {row['variant']}" if row["kind"] == PUBLISHED_KIND else ""
    reduced = f"; REDUCED eFalls predictor set {row['efalls_coverage']}" if row["kind"] == REDUCED_KIND else ""
    return f"{row['experiment']} ({row['model']}{variant}{reduced})"


def _highlight_text(row: pd.Series) -> str:
    names = [n for n in ("best_discrimination", "best_calibrated", "simplest_within_margin") if row[n]]
    return ", ".join(names) or "-"


def _params_text(best_params: str) -> str:
    return "none" if best_params in ("{}", "null") else best_params


def _category_table(part: pd.DataFrame) -> Block:
    t = float(part["threshold"].iloc[0])
    rows = []
    for _, r in part.iterrows():
        label = r["effective_experiment_label"] if isinstance(r["effective_experiment_label"], str) else "-"
        auroc = fmt_ci({"estimate": _none_if_nan(r["AUROC"]), "ci_low": _none_if_nan(r["AUROC_ci_low"]),
                        "ci_high": _none_if_nan(r["AUROC_ci_high"])})
        coverage = r["efalls_coverage"] if isinstance(r["efalls_coverage"], str) else "-"
        rows.append((r["experiment"], r["model"], r["variant"], label, coverage, auroc, fmt(r["PR_AUC"]), fmt(r["Brier"]), fmt(r["calibration_slope"]),
                     fmt(r["calibration_intercept"]), fmt(r["citl"]), fmt(r["oe_ratio"]), fmt(r["sensitivity_at_threshold"]),
                     fmt(r["specificity_at_threshold"]), fmt(r["net_benefit_at_threshold"], 4), fmt(_int_or_none(r["number_of_features"])),
                     fmt(_int_or_none(r["robust_features"])), _params_text(r["best_params"]), fmt(r["train_time"], 2),
                     fmt(r["inference_time"], 4), fmt(bool(r["comparable"])), _highlight_text(r)))
    return table(("Experiment", "Model", "Variant", "Effective label", "eFalls coverage", "AUROC (95% CI)", "PR-AUC", "Brier", "Slope",
                  "Calibration intercept",
                  "CITL", "O/E", f"Sens@{t:.2f}", f"Spec@{t:.2f}", f"NB@{t:.2f}", "Features (non-zero raw predictors)", "Robust features",
                  "Best params", "Train time (s)", "Inference (s per 1000 rows)", "Comparable", "Highlights"), rows)


def _none_if_nan(x: float) -> float | None:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


def _int_or_none(x: float) -> int | None:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else int(x)


def _recommendation(df: pd.DataFrame, synthetic: bool, auroc_margin: float, brier_margin: float,
                    slope_range: tuple[float, float]) -> list[Block]:
    cand = _candidates(df)
    blocks: list[Block] = []
    if synthetic:
        blocks.append(Paragraph("SYNTHETIC FIXTURE: this recommendation only demonstrates the decision logic; it is not scientific evidence.",
                                "warning"))
    blocks.append(Paragraph(
        "Candidates are comparable runs (served prediction; published eFalls primary variant) outside the ablation. A candidate "
        f"meets the criteria when its AUROC is within {auroc_margin} of the best candidate, its Brier score within {brier_margin} of "
        f"the best, its calibration slope within [{slope_range[0]}, {slope_range[1]}], and it is clinically useful: net benefit at "
        "the candidate threshold greater than both treat-all and treat-none (0). The highest-AUROC model is never chosen on AUROC "
        "alone: among candidates meeting all criteria the operationally simplest (fixed order: " + ", ".join(SIMPLICITY_ORDER) + ") is "
        "preferred, then the smaller calibration error |slope-1|+|CITL|, then the higher AUROC. Stability (robust features across "
        "bootstrap refits) and interpretability are shown as descriptive evidence only; they are not ranking criteria."))
    if cand.empty or cand["AUROC"].isna().all():
        return blocks + [Paragraph("No comparable run with test metrics is available; no recommendation is made.")]
    best_auroc, best_brier = cand["AUROC"].max(), cand["Brier"].min()
    blocks.append(table(("Model", "AUROC", "Brier", "Calibration error |slope-1|+|CITL|", "NB minus treat-all NB", "Criteria not met",
                         "Simplicity rank", "Robust features (descriptive)", "Interpretability (descriptive)", "Run eligibility check"),
                        [(_label(r), fmt(r["AUROC"]), fmt(r["Brier"], 4), fmt(abs(r["calibration_slope"] - 1) + abs(r["citl"])),
                          fmt(r["net_benefit_at_threshold"] - r["net_benefit_treat_all_at_threshold"], 4),
                          "; ".join(_failures(r, best_auroc, best_brier, auroc_margin, brier_margin, slope_range)) or "none",
                          str(_simplicity(r["model"]) + 1), fmt(_int_or_none(r["robust_features"])), INTERPRETABILITY.get(r["model"], "unknown"),
                          fmt(r["eligible_for_further_validation"])) for _, r in cand.iterrows()]))
    best = df.loc[df["best_discrimination"]].iloc[0]
    rec = df.loc[df["simplest_within_margin"]]
    if rec.empty:
        calibrated = df.loc[df["best_calibrated"]]
        return blocks + [Paragraph(
            "No comparable model meets all criteria, so no model is recommended for further validation on this evidence. "
            f"Best discrimination: {_label(best)} (AUROC {fmt(best['AUROC'])}); best calibrated: "
            f"{_label(calibrated.iloc[0]) if not calibrated.empty else 'not available'}. See the criteria not met in the table above.")]
    r = rec.iloc[0]
    eligible = r["eligible_for_further_validation"] is None or bool(r["eligible_for_further_validation"])
    lead = ("Recommended for further validation" if eligible else
            "No model is recommended: the preferred candidate failed its own eligibility check (see its report). "
            "If those issues were resolved, the decision logic would favour")
    text = [f"{lead}: {_label(r)}. Test AUROC {fmt(r['AUROC'])}, Brier {fmt(r['Brier'], 4)}, calibration "
            f"slope {fmt(r['calibration_slope'])}, CITL {fmt(r['citl'])} and observed/expected ratio {fmt(r['oe_ratio'])} (1 means predicted "
            f"risks match the observed rate on average; above 1, risks are under-predicted)."]
    if best["run_id"] != r["run_id"] or best["variant"] != r["variant"]:
        reasons = _failures(best, best_auroc, best_brier, auroc_margin, brier_margin, slope_range)
        cal_best = abs(best["calibration_slope"] - 1) + abs(best["citl"])
        cal_rec = abs(r["calibration_slope"] - 1) + abs(r["citl"])
        if _simplicity(best["model"]) > _simplicity(r["model"]):
            tie = "less simple to operate and interpret"
        elif math.isclose(cal_best, cal_rec, rel_tol=1e-9, abs_tol=1e-12) and math.isclose(best["AUROC"], r["AUROC"], abs_tol=1e-12):
            tie = "tied on every criterion (identical metrics); the order of experiment names decided"
        else:
            tie = "equally simple but less well calibrated (|slope-1|+|CITL|)"
        text.append(f"The highest-AUROC model, {_label(best)} (AUROC {fmt(best['AUROC'])}), was not chosen because "
                    + ("; ".join(reasons) if reasons else
                       f"its AUROC advantage ({best['AUROC'] - r['AUROC']:+.3f}) is within the {auroc_margin} margin and it is {tie}") + ".")
    text.append(f"Clinical usefulness at threshold {r['threshold']:.2f}: net benefit {r['net_benefit_at_threshold']:.4f} vs treat-all "
                f"{r['net_benefit_treat_all_at_threshold']:.4f} and treat-none 0 (criterion met).")
    text.append(f"Descriptive evidence (not used for ranking): interpretability {INTERPRETABILITY.get(r['model'], 'unknown')}; "
                + (f"{_int_or_none(r['robust_features'])} robust features in bootstrap refits." if not math.isnan(r["robust_features"])
                   else "stability not assessed for this model (for example, fixed published coefficients have no fitting process)."))
    if r["effective_experiment_label"] == PARTIAL_LABEL:
        text.append("Caution: this run is labelled efalls_partial_scoring and is not a validation of eFalls (M-11).")
    return blocks + [Paragraph(" ".join(text))]


def _signed_ci(row: pd.Series | None, dp: int) -> str:
    if row is None or pd.isna(row["estimate"]):
        return "not available"
    ci = "CI not available" if pd.isna(row["ci_low"]) or pd.isna(row["ci_high"]) else f"{row['ci_low']:+.{dp}f} to {row['ci_high']:+.{dp}f}"
    return f"{row['estimate']:+.{dp}f} ({ci})"


def _paired_section(df: pd.DataFrame, paired: pd.DataFrame, ref: pd.Series | None, n_bootstrap: int, seed: int) -> list[Block]:
    blocks: list[Block] = [Heading("Paired differences vs published eFalls (primary)")]
    if ref is None:
        return blocks + [Paragraph(f"Not computed: no comparable published eFalls run labelled {PUBLISHED_KIND} with the primary variant "
                                   f"{PRIMARY_VARIANT} is available. This comparison therefore contains no paired confidence intervals.", "note")]
    t = float(df["threshold"].iloc[0])
    nb_metric = f"net_benefit@{t:g}"
    blocks.append(Paragraph(
        f"Each row is a run minus the published eFalls primary variant ({PRIMARY_VARIANT}, run {ref['run_id']}) on identical test rows "
        f"(test rows SHA-256 {ref['test_rows_sha256']}, data SHA-256 {ref['data_sha256']}); rows are matched on (research_id, index_date) "
        "and must agree exactly, outcomes included. The compared prediction is what each run's bundle serves (the recalibrated risk "
        "when recalibration is served); the reference is the unmodified published equation. 95% percentile CIs come from "
        f"{n_bootstrap} paired patient-level bootstrap resamples (clustered by research_id, seed {seed}). A positive AUROC or net "
        "benefit difference and a negative Brier difference favour the run; slope and CITL differences are not directional (ideal "
        "slope 1, CITL 0). Machine-readable: comparison_paired_differences.csv."))
    if paired.empty:
        return blocks + [Paragraph("No other comparable run shares these test rows and data file.", "note")]
    names = df.drop_duplicates("run_id").set_index("run_id")
    rows = []
    for (run_id, variant), part in paired.groupby(["run_id", "variant"], sort=False):
        by_metric = {r["metric"]: r for _, r in part.iterrows()}
        who = names.loc[run_id]
        rows.append((f"{who['experiment']} ({who['model']})", variant, _signed_ci(by_metric.get("auroc"), 3), _signed_ci(by_metric.get("brier"), 4),
                     _signed_ci(by_metric.get("calibration_slope"), 3), _signed_ci(by_metric.get("citl"), 3),
                     _signed_ci(by_metric.get(nb_metric), 4)))
    return blocks + [table(("Run", "Variant", "Δ AUROC (95% CI)", "Δ Brier (95% CI)", "Δ Calibration slope (95% CI)", "Δ CITL (95% CI)",
                            f"Δ Net benefit @{t:.2f} (95% CI)"), rows)]


def _document(df: pd.DataFrame, paired: pd.DataFrame, reference: pd.Series | None, warnings: list[str], skipped: list[str],
              synthetic: bool, auroc_margin: float, brier_margin: float, slope_range: tuple[float, float], n_bootstrap: int,
              seed: int) -> list[Block]:
    comparable = df.loc[df["comparable"]]
    sha, data = (comparable["test_rows_sha256"].iloc[0], comparable["data_sha256"].iloc[0]) if len(comparable) else ("none", "none")
    blocks: list[Block] = [Paragraph(SYNTHETIC_BANNER, "banner")] if synthetic else []
    blocks += [Heading("Model comparison", 1)]
    partial = df.loc[df["effective_experiment_label"] == PARTIAL_LABEL]
    if not partial.empty:
        blocks.append(Paragraph(f"WARNING: {', '.join(sorted(set(partial['run_id'])))} labelled efalls_partial_scoring; "
                                "not a validation of eFalls (M-11).", "warning"))
    reduced = df.loc[df["kind"] == REDUCED_KIND]
    if not reduced.empty:
        blocks.append(Paragraph(f"WARNING: {', '.join(sorted(set(reduced['run_id'])))} use a REDUCED eFalls predictor set – NOT a full "
                                "eFalls reproduction (eFalls coverage "
                                f"{', '.join(sorted(set(reduced['efalls_coverage'].dropna().astype(str)))) or 'not available'}); "
                                "they are reported in their own section and never as eFalls.", "warning"))
    blocks += [Paragraph(f"WARNING: {w}", "warning") for w in warnings]
    blocks.append(Paragraph(
        f"All metrics are on the test split. Non-published runs show the prediction their bundle serves (primary_variant); the "
        f"published eFalls run shows the fixed published equation ({PRIMARY_VARIANT} primary, {CO_REPORTED_VARIANT} co-reported). A run "
        f"is comparable only if its test rows (SHA-256 {sha}) and data file (SHA-256 {data}) match the published eFalls reference (or "
        f"the majority when no published run exists); only comparable primary rows outside the ablation are highlighted. Threshold "
        f"{df['threshold'].iloc[0]:.2f} is a candidate, not a clinically approved cut-off (M-12). eFalls performed no temporal "
        "validation, so temporal results have no published comparator (D-19)."))
    for category, (title, statement) in CATEGORY_TEXT.items():
        part = df.loc[df["category"] == category]
        blocks += [Heading(title), Paragraph(statement)]
        blocks.append(_category_table(part) if not part.empty else Paragraph("No runs in this category.", "note"))
    blocks += _paired_section(df, paired, reference, n_bootstrap, seed)
    marks = [f"{name}: {', '.join(_label(r) for _, r in df.loc[df[name]].iterrows()) or 'none'}"
             for name in ("is_efalls_published", "is_efalls_retrained", "best_discrimination", "best_calibrated", "simplest_within_margin")]
    blocks += [Heading("Highlights"), bullets(marks), Heading("Recommendation")]
    blocks += _recommendation(df, synthetic, auroc_margin, brier_margin, slope_range)
    excluded = df.loc[~df["comparable"]]
    blocks += [Heading("Runs not comparable with the reference"),
               bullets(f"{_label(r)} [{r['run_id']}]: test rows {r['test_rows_sha256']}, data {r['data_sha256']} (dataset {r['dataset_version']})"
                       for _, r in excluded.iterrows()) if not excluded.empty else Paragraph("None.", "note"),
               Heading("Skipped directories (incomplete runs: no RUN_COMPLETE.json)"),
               bullets(skipped) if skipped else Paragraph("None.", "note")]
    return blocks
