"""Nested feature-group ablation summary (spec §1.2 L3b additions, D-19 §5 common test, D-21 paired bootstrap).

Every step is compared with the baseline on identical test rows: the ``(research_id, index_date)`` sets and the
outcomes must match exactly, otherwise ``ArtifactError`` (a ``ValueError``). Predictions are the uncalibrated risks
(``risk_uncalibrated``) so that feature groups, not recalibration, explain the differences. Differences are step
minus baseline, with patient-level paired bootstrap CIs. :func:`load_test_predictions` and :func:`check_same_rows`
are shared with the master comparison.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from falls_ml.evaluation.bootstrap import paired_bootstrap_difference
from falls_ml.evaluation.metrics import compute_metric
from falls_ml.logging_utils import get_logger
from falls_ml.reporting.report import (
    SYNTHETIC_BANNER, ArtifactError, Block, Heading, Paragraph, bullets, dig, fmt, read_optional_csv, table, write_document,
)

log = get_logger(__name__)

METRICS = ("auroc", "brier", "calibration_slope", "citl")
METRIC_LABELS = {"auroc": "AUROC", "brier": "Brier", "calibration_slope": "calibration slope", "citl": "CITL"}
KEY = ["research_id", "index_date"]
BOOTSTRAP_COMPONENT = "ablation_paired_test"


def load_test_predictions(run: Path, columns: Sequence[str]) -> pd.DataFrame:
    """``predictions_test.parquet`` sorted by its ``(research_id, index_date)`` key (column ``_key``).

    ``FileNotFoundError`` without the file; ``ArtifactError`` for missing columns or duplicated keys.
    """
    path = run / "predictions_test.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"predictions_test.parquet not found in {run}")
    pred = pd.read_parquet(path)
    missing = [c for c in dict.fromkeys((*KEY, "outcome", *columns)) if c not in pred.columns]
    if missing:
        raise ArtifactError(f"{run.name}: predictions_test.parquet lacks columns {missing}")
    key = pred["research_id"].astype("string") + "|" + pd.to_datetime(pred["index_date"]).dt.strftime("%Y-%m-%d")
    if key.duplicated().any():
        raise ArtifactError(f"{run.name}: {int(key.duplicated().sum())} duplicated (research_id, index_date) rows")
    return pred.assign(_key=key).sort_values("_key", kind="stable").reset_index(drop=True)


def _linear_predictor(pred: pd.DataFrame, name: str) -> np.ndarray | None:
    lp = pred["linear_predictor"].to_numpy(dtype=float)
    finite = np.isfinite(lp)
    if finite.all():
        return lp
    if not finite.any():
        return None  # non-linear model: calibration uses logit(risk)
    raise ArtifactError(f"{name}: linear_predictor is missing for {int((~finite).sum())} of {lp.size} rows")


def check_same_rows(base: pd.DataFrame, other: pd.DataFrame, label: str) -> None:
    """``ArtifactError`` unless both loaded prediction frames hold exactly the same keys and outcomes."""
    if len(base) != len(other) or not np.array_equal(base["_key"].to_numpy(), other["_key"].to_numpy()):
        only_base = len(set(base["_key"]) - set(other["_key"]))
        only_other = len(set(other["_key"]) - set(base["_key"]))
        raise ArtifactError(f"{label}: test rows differ from the reference "
                            f"({only_base} rows only in the reference, {only_other} only in the compared run)")
    diff = int((base["outcome"].to_numpy(dtype=np.int64) != other["outcome"].to_numpy(dtype=np.int64)).sum())
    if diff:
        raise ArtifactError(f"{label}: outcomes differ from the reference in {diff} rows")


def _features(run: Path) -> list[str] | None:
    fi = read_optional_csv(run / "feature_importance.csv")
    return None if fi is None else [str(f) for f in fi["feature"]]


def _metrics_json(run: Path) -> dict[str, Any]:
    path = run / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def summarize_ablation(baseline_run: str | Path, step_runs: list[tuple[str, Path]], out_dir: str | Path, *,
                       n_bootstrap: int = 1000, seed: int = 1) -> pd.DataFrame:
    """Write ``ablation_results.csv`` and ``ablation.md``; return one row per run (baseline first)."""
    baseline_run = Path(baseline_run)
    runs = [("Baseline", baseline_run), *((str(n), Path(p)) for n, p in step_runs)]
    preds = [load_test_predictions(path, ("risk_uncalibrated", "linear_predictor")) for _, path in runs]
    for (name, _), pred in zip(runs[1:], preds[1:]):
        check_same_rows(preds[0], pred, f"ablation step {name!r} vs baseline")
    y = preds[0]["outcome"].to_numpy(dtype=np.int64)
    cluster = preds[0]["research_id"].astype("string").to_numpy()
    base_p, base_lp = preds[0]["risk_uncalibrated"].to_numpy(dtype=float), _linear_predictor(preds[0], "Baseline")
    metas = [_metrics_json(path) for _, path in runs]
    flags = {bool(meta.get("synthetic_fixture")) for meta in metas}
    if len(flags) > 1:
        raise ArtifactError("ablation runs mix synthetic fixture and real-data runs: "
                            f"{[name for (name, _), meta in zip(runs, metas) if meta.get('synthetic_fixture')]} are synthetic")
    synthetic = flags == {True}
    rows: list[dict[str, Any]] = []
    previous_features = None
    for i, ((name, path), pred, meta) in enumerate(zip(runs, preds, metas)):
        p, lp = pred["risk_uncalibrated"].to_numpy(dtype=float), _linear_predictor(pred, name)
        features = _features(path)
        added = ("" if i == 0 else "not available" if features is None or previous_features is None
                 else ";".join(f for f in features if f not in set(previous_features)))
        row: dict[str, Any] = {"step": name, "run_id": meta.get("run_id", path.name), "model": dig(meta, "model", "name"),
                               "added_features": added, "n": int(y.size), "n_events": int(y.sum())}
        for metric in METRICS:
            row[metric] = compute_metric(metric, y, p, lp)
            if i == 0:
                row.update({f"{metric}_delta": np.nan, f"{metric}_delta_ci_low": np.nan, f"{metric}_delta_ci_high": np.nan})
                continue
            d = paired_bootstrap_difference(y, p, base_p, metric=metric, n=n_bootstrap, seed=seed, component=BOOTSTRAP_COMPONENT,
                                            lp_a=lp, lp_b=base_lp, cluster=cluster)
            row.update({f"{metric}_delta": _nan(d["estimate"]), f"{metric}_delta_ci_low": _nan(d["ci_low"]),
                        f"{metric}_delta_ci_high": _nan(d["ci_high"])})
        row["auroc_change_vs_previous_step"] = np.nan if i == 0 else row["auroc"] - rows[-1]["auroc"]
        rows.append(row)
        previous_features = features
    df = pd.DataFrame(rows)
    df["n_bootstrap"], df["seed"] = n_bootstrap, seed
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "ablation_results.csv", index=False, encoding="utf-8", lineterminator="\n")
    write_document(_document(df, metas[0], synthetic, n_bootstrap, seed), out_dir / "ablation.md", None, title="Feature-group ablation")
    log.info("ablation_summary_written", extra_fields={"out_dir": str(out_dir), "n_steps": len(step_runs), "n_rows_test": int(y.size)})
    return df


def _nan(x: float | None) -> float:
    return np.nan if x is None else float(x)


def _delta(row: pd.Series, metric: str, dp: int = 3) -> str:
    est, low, high = row[f"{metric}_delta"], row[f"{metric}_delta_ci_low"], row[f"{metric}_delta_ci_high"]
    if pd.isna(est):
        return "not available"
    ci = "CI not available" if pd.isna(low) or pd.isna(high) else f"{low:.{dp}f}–{high:.{dp}f}"
    return f"{est:+.{dp}f} ({ci})"


def _baseline_text(base: pd.Series, meta: dict[str, Any]) -> str:
    ds, fs = meta.get("dataset") or {}, meta.get("feature_set") or {}
    return (f"Baseline model: {dig(meta, 'experiment', 'name', default='not available')} ({base['model'] or 'model not available'}; "
            f"experiment kind {dig(meta, 'experiment', 'kind', default='not available')}), run {base['run_id']}, feature set "
            f"{fs.get('name', 'not available')} {fs.get('version', '')} (pure eFalls: {fmt(fs.get('pure_efalls'))}). Dataset: version "
            f"{ds.get('dataset_version', 'not available')} (data SHA-256 {ds.get('data_sha256', 'not available')}), test rows "
            f"SHA-256 {dig(meta, 'split', 'test_rows_sha256', default='not available')}, {base['n']} test rows with {base['n_events']} "
            "outcomes. This baseline is the ablation's own fitted model on these rows, not the published eFalls equation and not "
            "the standalone eFalls retraining.")


def _document(df: pd.DataFrame, base_meta: dict[str, Any], synthetic: bool, n_bootstrap: int, seed: int) -> list[Block]:
    base = df.iloc[0]
    lines = [f"Baseline AUROC {base['auroc']:.3f} (Brier {base['brier']:.4f}, calibration slope {fmt(base['calibration_slope'])}, "
             f"CITL {fmt(base['citl'])})"]
    for _, r in df.iloc[1:].iterrows():
        lines.append(f"{r['step']} {_delta(r, 'auroc')} AUROC; Brier {_delta(r, 'brier', 4)}; calibration slope "
                     f"{_delta(r, 'calibration_slope')}; CITL {_delta(r, 'citl')}; added features: {r['added_features'] or 'none'}")
    blocks: list[Block] = [Paragraph(SYNTHETIC_BANNER, "banner")] if synthetic else []
    blocks += [Heading("Feature-group ablation", 1), Paragraph(_baseline_text(base, base_meta)), Paragraph(
        "Each step adds feature groups cumulatively to the baseline feature set. Added groups are Meuhedet-specific "
        "additions (layer L3b), so every step is an experimental algorithm, not eFalls. All steps use identical test rows "
        "and outcomes and uncalibrated predictions. Differences are step minus baseline with paired patient-level bootstrap "
        f"95% CIs ({n_bootstrap} resamples, seed {seed}). Lower Brier is better; ideal calibration slope is 1 and ideal CITL is 0."),
        bullets(lines),
        table(("Step", "Run", "Model", "AUROC", "AUROC change vs baseline", "AUROC change vs previous step", "Brier", "Brier change",
               "Calibration slope", "Slope change", "CITL", "CITL change", "Added features"),
              [(r["step"], r["run_id"], fmt(r["model"]), fmt(r["auroc"]), _delta(r, "auroc"), fmt(r["auroc_change_vs_previous_step"]), fmt(r["brier"], 4),
                _delta(r, "brier", 4), fmt(r["calibration_slope"]), _delta(r, "calibration_slope"), fmt(r["citl"]), _delta(r, "citl"),
                r["added_features"] or "-") for _, r in df.iterrows()])]
    return blocks
