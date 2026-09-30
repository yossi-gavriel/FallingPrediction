"""Computations behind the model reports. Every result is aggregate.

Test-set discipline (D-19 §4): anything that could influence model development uses TRAIN / VALIDATION only - the coefficient path
(TRAIN design matrix, the run's own lambda grid), the learning curve (nested TRAIN subsets, VALIDATION evaluation; test rows are
refused by the fit guard) and the primary error analysis (VALIDATION). The test partition is read only from the predictions the
frozen model already produced after the run released it once; those figures are labelled HELD-OUT TEST (model frozen) and are a
description of the final model, never an input to retuning.
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from falls_ml.errors import DatasetValidationError, LeakageError
from falls_ml.evaluation.metrics import auroc, brier, pr_auc
from falls_ml.logging_utils import get_logger
from falls_ml.modelreport.artifacts import Run
from falls_ml.seeding import int_seed_for, rng_for

log = get_logger(__name__)

LIFT_FRACTIONS = (0.01, 0.02, 0.05, 0.10, 0.20)
LEARNING_FRACTIONS = (0.10, 0.20, 0.40, 0.60, 0.80, 1.00)
THRESHOLDS = tuple(round(x, 3) for x in (*np.arange(0.01, 0.10, 0.01), *np.arange(0.10, 0.51, 0.025)))
MIN_EVENTS_METRIC = 10          # the framework's floor for reporting a metric on a subset
SUPPRESSED = "<min_cell"


def cell(n: int, min_cell: int) -> int | str:
    n = int(n)
    return n if n == 0 or n >= min_cell else SUPPRESSED


def _r(v: Any, d: int = 4) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return round(f, d) if math.isfinite(f) else None


@dataclass
class Frames:
    """TRAIN / VALIDATION rows of a run's canonical dataset (test rows are held back: only their keys are kept)."""

    spec: Any
    config: Any
    train: pd.DataFrame
    validation: pd.DataFrame
    test_keys: set[tuple[str, str]]
    frame_features: list[str]


def _keys(df: pd.DataFrame, spec: Any) -> pd.Series:
    return df[spec.identifier_columns[0]].astype(str) + "|" + pd.to_datetime(df[spec.index_column]).dt.strftime("%Y-%m-%d")


def load_frames(run: Run, search: list[Path]) -> Frames:
    """The run's dataset partitioned by its own splits.csv; verified against the run's recorded test-partition hash."""
    from falls_ml.config import load_experiment_config
    from falls_ml.data.dataset import ModelingDataset
    from falls_ml.features.spec import load_feature_spec

    ds_dir = run.dataset_dir(search)
    if ds_dir is None:
        raise DatasetValidationError(f"{run.display}: the run's canonical dataset was not found (needed locally for the coefficient path and learning curve)")
    cfg = load_experiment_config(run.run_dir / "config.yaml")
    spec = load_feature_spec(cfg.dataset.feature_spec, cfg.dataset.feature_spec_extensions, anchor=run.run_dir)
    feats = list(cfg.preprocessing.features or [])
    ds = ModelingDataset.load(ds_dir, spec, features=feats or None)
    sub = spec.subset(feats) if feats else spec
    splits = run.csv("splits.csv")
    if splits.empty:
        raise DatasetValidationError(f"{run.display}: splits.csv missing")
    lookup = dict(zip(splits["research_id"].astype(str) + "|" + pd.to_datetime(splits["index_date"]).dt.strftime("%Y-%m-%d"), splits["split"].astype(str)))
    frame = ds.frame
    part = _keys(frame, sub).map(lookup)
    test_keys = sorted(_keys(frame[part == "test"], sub))
    sha = hashlib.sha256("\n".join(test_keys).encode("utf-8")).hexdigest()
    if run.test_sha and sha != run.test_sha:
        raise LeakageError(f"{run.display}: the dataset + splits.csv do not reproduce the run's test partition", [run.test_sha, sha])
    tk = {tuple(k.split("|")) for k in test_keys}
    return Frames(spec=sub, config=cfg, train=frame[part == "train"].reset_index(drop=True), validation=frame[part == "validation"].reset_index(drop=True),
                  test_keys=tk, frame_features=feats)


def coefficient_path(run: Run, frames: Frames) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Standardised-scale coefficients along the run's own lambda grid, on the run's TRAIN design matrix (the saved preprocessor);
    checked against the saved model at lambda*."""
    from falls_ml.bundle import load_bundle
    from falls_ml.models.lasso_cv import lasso_logistic_path, standardize

    bundle = load_bundle(run.run_dir / "model")
    model = bundle.pipeline.model
    if not hasattr(model, "coef_standardized_"):
        raise DatasetValidationError(f"{run.display}: the served model is not a LASSO (no coefficient path)")
    X = bundle.pipeline.preprocessor.transform(frames.train)
    Xa = X[list(model.feature_names_)].to_numpy(dtype=np.float64)
    y = frames.train[frames.spec.outcome.name].to_numpy(dtype=np.float64)
    Z, _mean, sd = standardize(Xa)
    cv = run.csv("cv_results.csv")
    grid = cv["lambda"].to_numpy(dtype=np.float64)
    p = model.params
    path = lasso_logistic_path(Z, y, grid, tol=float(p["tol"]), max_iter=int(p["max_iter"]), stop=float(p["stop"]))
    k = int(np.flatnonzero(cv["selected"].astype(bool).to_numpy())[0])
    diff = float(np.max(np.abs(path.beta[k] - np.asarray(model.coef_standardized_)))) if path.beta.shape[1] else 0.0
    rows = [{"lambda": float(grid[i]), "design_column": c, "coef_standardized": float(path.beta[i, j])}
            for i in range(grid.size) for j, c in enumerate(model.feature_names_)]
    info = {"lambda_star": float(grid[k]), "lambda_star_index": k, "n_lambda": int(grid.size), "reproduces_saved_model": diff < 1e-5,
            "max_abs_difference_at_lambda_star": diff, "n_train_rows": int(len(y)), "path_converged_share": float(path.converged.mean()),
            "entry_order": _entry_order(path.beta, list(model.feature_names_), grid)}
    return pd.DataFrame(rows), info


def _entry_order(beta: np.ndarray, names: list[str], grid: np.ndarray) -> list[dict[str, Any]]:
    out = []
    for j, n in enumerate(names):
        nz = np.flatnonzero(beta[:, j] != 0.0)
        if nz.size:
            out.append({"design_column": n, "enters_at_lambda": float(grid[nz[0]]), "enters_at_index": int(nz[0])})
    return sorted(out, key=lambda d: d["enters_at_index"])


@contextmanager
def captured_warnings() -> Iterator[Counter]:
    """Count the WARNING events the falls_ml loggers emit inside the block (e.g. lasso_cv_minimum_not_identified during one refit)."""
    counter: Counter = Counter()

    class _Counter(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            counter[record.getMessage()] += 1

    handler = _Counter(level=logging.WARNING)
    root = logging.getLogger("falls_ml")
    root.addHandler(handler)
    try:
        yield counter
    finally:
        root.removeHandler(handler)


def learning_curve(run: Run, frames: Frames, *, mode: str = "full", fractions: tuple[float, ...] = LEARNING_FRACTIONS, n_boot: int = 200,
                   seed: int = 20260923, predictions_out: dict[float, np.ndarray] | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Refit the run's pipeline (same config, same preprocessing / FP rules, same LASSO procedure) on nested, outcome-stratified,
    patient-grouped TRAIN subsets; evaluate on the subset (apparent) and on the whole VALIDATION partition. Test rows are forbidden in
    every fit. ``mode='reduced'`` keeps everything except the 10-fold CV: lambda is fixed at the full run's lambda* (documented).
    Every row records the refit's own LASSO diagnostics and the warnings it emitted; ``predictions_out`` (optional) receives the VALIDATION
    predictions per fraction, kept in memory only (paired increments)."""
    from falls_ml.pipeline import fit_pipeline, model_params_from_config, row_keys

    if mode not in ("full", "reduced"):
        raise ValueError("learning-curve mode must be full or reduced")
    spec, cfg = frames.spec, frames.config
    id_col, y_col = spec.identifier_columns[0], spec.outcome.name
    forbidden = row_keys(frames.validation, spec) | {(k[0], pd.Timestamp(k[1])) for k in frames.test_keys}
    params = model_params_from_config(cfg)
    lam_star = float((run.metrics.get("lasso") or {}).get("lambda_star") or 0.0)
    if mode == "reduced":
        if not lam_star:
            raise ValueError("reduced mode needs the run's lambda*")
        params["fixed_lambda"] = lam_star
    params["refit_unpenalized"] = False    # descriptive refit is not needed for a learning curve
    pat = frames.train.groupby(id_col)[y_col].max()
    rng = rng_for(seed, f"learning_curve::{run.key}")
    order = {cls: list(pd.Index(pat.index[pat == cls])[rng.permutation(int((pat == cls).sum()))]) for cls in (0, 1)}
    yv = frames.validation[y_col].to_numpy(dtype=np.int64)
    rows = []
    for f in fractions:
        chosen = set(order[0][: math.ceil(f * len(order[0]))]) | set(order[1][: math.ceil(f * len(order[1]))])
        sub = frames.train[frames.train[id_col].isin(chosen)].reset_index(drop=True)
        ys = sub[y_col].to_numpy(dtype=np.int64)
        r: dict[str, Any] = {"fraction": f, "n_train": int(len(sub)), "n_train_events": int(ys.sum()), "mode": mode}
        t0 = time.perf_counter()
        wc: Counter = Counter()
        try:
            with captured_warnings() as wc:
                pipe = fit_pipeline(sub, spec, cfg, random_state=int_seed_for(seed, f"learning_curve_fit::{f}"), model_params=params,
                                    forbidden_row_keys=forbidden)
            pt, pv = pipe.predict_proba(sub), pipe.predict_proba(frames.validation)
            if predictions_out is not None:
                predictions_out[f] = np.asarray(pv, dtype=float)
            diag = pipe.model.fit_diagnostics() if hasattr(pipe.model, "fit_diagnostics") else {}
            k_star, n_lam = diag.get("lambda_star_index"), diag.get("n_lambda")
            if mode == "reduced":   # lambda fixed at lambda*: no CV, and the path ends at lambda* by construction - grid flags do not apply
                r.update({"cv_minimum_identified": None, "lambda_star_index": None, "n_lambda": None, "lambda_star_near_grid_end": None,
                          "path_converged": diag.get("path_converged"), "cv_paths_converged": None})
            else:
                r.update({"cv_minimum_identified": diag.get("cv_minimum_identified"), "lambda_star_index": k_star, "n_lambda": n_lam,
                          "lambda_star_near_grid_end": bool(k_star is not None and n_lam and int(k_star) >= int(n_lam) - 5) if k_star is not None else None,
                          "path_converged": diag.get("path_converged"), "cv_paths_converged": diag.get("cv_paths_converged")})
            boot = []
            g = rng_for(seed, f"learning_curve_boot::{f}")
            for _ in range(n_boot):
                i = g.integers(0, len(yv), len(yv))
                if yv[i].min() != yv[i].max():
                    boot.append(auroc(yv[i], pv[i]))
            r.update({"train_auroc_apparent": _r(auroc(ys, pt)), "validation_auroc": _r(auroc(yv, pv)),
                      "validation_auroc_ci_low": _r(np.quantile(boot, 0.025)) if boot else None, "validation_auroc_ci_high": _r(np.quantile(boot, 0.975)) if boot else None,
                      "train_brier_apparent": _r(brier(ys, pt), 5), "validation_brier": _r(brier(yv, pv), 5),
                      "lambda_star": _r(diag.get("lambda_star"), 6), "n_selected": diag.get("n_selected"), "status": "fitted"})
        except Exception as exc:   # a small subset can be degenerate (too few events): recorded, never hidden
            r.update({"status": f"fit failed: {type(exc).__name__}: {str(exc)[:160]}"})
        r["warnings"] = "; ".join(f"{k} x{v}" for k, v in sorted(wc.items())) or "none"
        r["seconds"] = round(time.perf_counter() - t0, 1)
        rows.append(r)
    info = {"mode": mode, "fractions": list(fractions), "n_validation": int(len(yv)), "n_validation_events": int(yv.sum()),
            "reduced_note": (f"REDUCED-COMPUTATION MODE: lambda fixed at the full run's lambda* = {lam_star:.6g} (no 10-fold CV per subset); preprocessing, "
                             "FP selection, predictors and the LASSO solver are unchanged") if mode == "reduced" else "",
            "rule": "nested outcome-stratified patient subsets of TRAIN (seeded permutation, each larger subset contains the smaller); "
                    "evaluation on the whole VALIDATION partition; uncalibrated model probabilities (the recalibration is fitted on validation, so it "
                    "cannot be used to evaluate on validation); the test partition is never read", "seed": seed}
    return pd.DataFrame(rows), info


def threshold_table(y: np.ndarray, p: np.ndarray, min_cell: int, thresholds: tuple[float, ...] = THRESHOLDS) -> pd.DataFrame:
    """Operating characteristics across risk thresholds (no threshold is chosen here)."""
    n, events = len(y), int(y.sum())
    rows = []
    for t in thresholds:
        flag = p >= t
        tp, fp = int((flag & (y == 1)).sum()), int((flag & (y == 0)).sum())
        fn, tn = events - tp, (n - events) - fp
        nf = tp + fp
        small = 0 < nf < min_cell or 0 < tp < min_cell
        rows.append({"threshold": t, "n_flagged": cell(nf, min_cell), "pct_flagged": None if small else _r(100.0 * nf / n, 2),
                     "falls_captured": cell(tp, min_cell), "pct_falls_captured": None if small else _r(100.0 * tp / events if events else None, 2),
                     "sensitivity": None if small else _r(tp / events if events else None), "specificity": _r(tn / (n - events) if n - events else None),
                     "ppv": None if small or not nf else _r(tp / nf), "npv": _r(tn / (tn + fn) if tn + fn else None)})
    return pd.DataFrame(rows)


def _top_mask(p: np.ndarray, frac: float) -> np.ndarray:
    k = max(1, math.ceil(frac * len(p)))
    order = np.argsort(-p, kind="mergesort")
    m = np.zeros(len(p), dtype=bool)
    m[order[:k]] = True
    return m


def lift_table(y: np.ndarray, p: np.ndarray, *, min_cell: int, n_boot: int = 1000, seed: int = 20260923,
               fractions: tuple[float, ...] = LIFT_FRACTIONS) -> pd.DataFrame:
    """Highest-risk top x%: patients selected, observed prevalence, falls captured (n, % of all), lift = prevalence / overall prevalence;
    95% intervals from a row bootstrap of the evaluation partition."""
    n, events = len(y), int(y.sum())
    base = events / n if n else float("nan")
    g = rng_for(seed, "lift")
    boots: dict[float, list[tuple[float, float]]] = {f: [] for f in fractions}
    for _ in range(n_boot):
        i = g.integers(0, n, n)
        yy, pp = y[i], p[i]
        ev = yy.sum()
        if ev == 0:
            continue
        for f in fractions:
            m = _top_mask(pp, f)
            boots[f].append((yy[m].sum() / ev, (yy[m].mean() / (ev / n)) if ev else np.nan))
    rows = []
    for f in fractions:
        m = _top_mask(p, f)
        k, cap = int(m.sum()), int(y[m].sum())
        b = np.array(boots[f]) if boots[f] else np.empty((0, 2))
        small = 0 < cap < min_cell or k < min_cell
        rows.append({"top_fraction": f, "top_pct": _r(100 * f, 1), "n_selected": cell(k, min_cell), "falls_in_group": cell(cap, min_cell),
                     "observed_prevalence": None if small else _r(cap / k), "pct_of_all_falls_captured": None if small else _r(100.0 * cap / events if events else None, 2),
                     "capture_ci_low": None if small or not len(b) else _r(100 * np.quantile(b[:, 0], 0.025), 2),
                     "capture_ci_high": None if small or not len(b) else _r(100 * np.quantile(b[:, 0], 0.975), 2),
                     "lift": None if small else _r((cap / k) / base if base else None, 3),
                     "lift_ci_low": None if small or not len(b) else _r(np.nanquantile(b[:, 1], 0.025), 3),
                     "lift_ci_high": None if small or not len(b) else _r(np.nanquantile(b[:, 1], 0.975), 3),
                     "population_prevalence": _r(base)})
    return pd.DataFrame(rows)


def gains_curve(y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    order = np.argsort(-p, kind="mergesort")
    cum = np.cumsum(y[order]) / max(int(y.sum()), 1)
    share = np.arange(1, len(y) + 1) / len(y)
    idx = np.unique(np.clip(np.round(np.linspace(0.01, 1.0, 100) * len(y)).astype(int) - 1, 0, len(y) - 1))
    return pd.DataFrame({"pct_population": 100 * share[idx], "pct_falls_captured": 100 * cum[idx]})


def risk_deciles(y: np.ndarray, p: np.ndarray, min_cell: int, q: int = 10) -> pd.DataFrame:
    """Observed fall rate and mean predicted risk per tenth of predicted risk (1 = lowest)."""
    order = np.argsort(p, kind="mergesort")
    groups = np.empty(len(p), dtype=int)
    groups[order] = np.floor(np.arange(len(p)) * q / len(p)).astype(int) + 1
    rows = []
    for g in range(1, q + 1):
        m = groups == g
        k, e = int(m.sum()), int(y[m].sum())
        small = 0 < e < min_cell or 0 < k - e < min_cell
        rows.append({"risk_group": g, "n": cell(k, min_cell), "events": cell(e, min_cell), "observed_rate": None if small else _r(e / k if k else None),
                     "mean_predicted": _r(p[m].mean() if k else None), "min_predicted": _r(p[m].min() if k else None), "max_predicted": _r(p[m].max() if k else None)})
    return pd.DataFrame(rows)


def risk_histogram(y: np.ndarray, p: np.ndarray, min_cell: int, bins: int = 40) -> pd.DataFrame:
    hi = float(np.quantile(p, 0.995)) if len(p) else 1.0
    edges = np.linspace(0.0, max(hi, 1e-6), bins + 1)
    rows = []
    for cls in (0, 1):
        c, _ = np.histogram(np.clip(p[y == cls], 0, edges[-1]), bins=edges)
        for i, v in enumerate(c):
            rows.append({"outcome": cls, "bin_low": edges[i], "bin_high": edges[i + 1], "n": int(v) if (v == 0 or v >= min_cell) else 0,
                         "share": (v / max(int((y == cls).sum()), 1)) if (v == 0 or v >= min_cell) else 0.0, "suppressed": bool(0 < v < min_cell)})
    return pd.DataFrame(rows)


def roc_points(y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y, p)
    keep = np.unique(np.linspace(0, len(fpr) - 1, min(len(fpr), 300)).astype(int))
    return pd.DataFrame({"fpr": fpr[keep], "tpr": tpr[keep]})


def pr_points(y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    from sklearn.metrics import precision_recall_curve

    prec, rec, _ = precision_recall_curve(y, p)
    keep = np.unique(np.linspace(0, len(prec) - 1, min(len(prec), 300)).astype(int))
    return pd.DataFrame({"recall": rec[keep], "precision": prec[keep]})


def subgroup_performance(df: pd.DataFrame, y: np.ndarray, p: np.ndarray, groups: dict[str, pd.Series], min_cell: int) -> pd.DataFrame:
    """Exploratory performance per subgroup; AUROC only with >= 10 events and >= 10 non-events, calibration slope with >= 20 events."""
    from falls_ml.evaluation.metrics import calibration_slope_intercept, logit

    rows = []
    for gname, s in groups.items():
        for level in sorted(s.dropna().astype(str).unique()):
            m = (s.astype(str) == level).to_numpy()
            n, e = int(m.sum()), int(y[m].sum())
            r: dict[str, Any] = {"subgroup": gname, "level": level, "n": cell(n, min_cell), "events": cell(e, min_cell),
                                 "observed_prevalence": None if (0 < e < min_cell or 0 < n - e < min_cell) else _r(e / n if n else None),
                                 "mean_predicted_risk": _r(p[m].mean() if n else None), "oe_ratio": _r(e / p[m].sum() if n and p[m].sum() else None, 3)}
            if e >= MIN_EVENTS_METRIC and (n - e) >= MIN_EVENTS_METRIC:
                from falls_ml.eda.common import auroc as auc_ci

                a, lo, hi = auc_ci(p[m], y[m])
                r.update({"auroc": _r(a, 3), "auroc_ci_low": _r(lo, 3), "auroc_ci_high": _r(hi, 3)})
                if e >= 20:
                    try:
                        slope = calibration_slope_intercept(y[m], logit(p[m]))[0]
                        r["calibration_slope"] = _r(slope, 3)
                    except Exception:   # noqa: BLE001 - a degenerate subgroup fit is reported as not estimable
                        r["calibration_slope"] = None
                r["reliability"] = "estimated (exploratory)" if e >= 50 else "few events: wide uncertainty"
            else:
                r["reliability"] = f"NOT REPORTED: fewer than {MIN_EVENTS_METRIC} events or non-events"
            rows.append(r)
    return pd.DataFrame(rows)


def error_analysis(df: pd.DataFrame, y: np.ndarray, p: np.ndarray, features: list[str], spec: Any, min_cell: int, *,
                   high_q: float = 0.8, low_q: float = 0.5) -> pd.DataFrame:
    """Aggregate profile of four groups defined by DESCRIPTIVE risk cut-points of this partition (top 20% = higher predicted risk,
    bottom 50% = lower predicted risk; not clinical thresholds): higher-risk non-fallers, lower-risk fallers, higher-risk fallers,
    lower-risk non-fallers. Groups below min_cell are not profiled."""
    hi_cut, lo_cut = float(np.quantile(p, high_q)), float(np.quantile(p, low_q))
    groups = {"higher-risk non-fallers (false alarms)": (p >= hi_cut) & (y == 0), "lower-risk fallers (missed)": (p <= lo_cut) & (y == 1),
              "higher-risk fallers (identified)": (p >= hi_cut) & (y == 1), "lower-risk non-fallers (correctly low)": (p <= lo_cut) & (y == 0)}
    rows = []
    for gname, m in groups.items():
        n = int(m.sum())
        r: dict[str, Any] = {"group": gname, "n": cell(n, min_cell), "mean_predicted_risk": _r(p[m].mean()) if n >= min_cell else None}
        if n >= min_cell:
            for f in features:
                if f not in df.columns:
                    continue
                s = df.loc[m, f]
                dt = spec.get(f).dtype if hasattr(spec, "get") else ""
                if str(dt) == "binary" or set(pd.to_numeric(s, errors="coerce").dropna().unique()) <= {0, 1}:
                    k = int((pd.to_numeric(s, errors="coerce") == 1).sum())
                    r[f] = SUPPRESSED if 0 < k < min_cell else f"{100.0 * k / n:.1f}%"
                elif str(dt).startswith("categorical") or s.dtype == object or str(s.dtype) == "string":
                    top = s.astype(str).value_counts()
                    r[f] = "; ".join(f"{lv} {100.0 * c / n:.0f}%" for lv, c in top.head(2).items() if c >= min_cell)
                else:
                    v = pd.to_numeric(s, errors="coerce")
                    r[f] = f"{v.median():.1f} [{v.quantile(0.25):.1f}-{v.quantile(0.75):.1f}]"
        rows.append(r)
    out = pd.DataFrame(rows)
    out.attrs["cut_points"] = {"higher_risk_from": hi_cut, "lower_risk_up_to": lo_cut}
    return out


def paired_differences(y: np.ndarray, pa: np.ndarray, pb: np.ndarray, *, n_boot: int = 1000, seed: int = 20260923) -> dict[str, Any]:
    """B - A on the same rows: AUROC, PR-AUC, Brier with row-bootstrap 95% intervals."""
    g = rng_for(seed, "paired")
    base = {"auroc": auroc(y, pb) - auroc(y, pa), "pr_auc": pr_auc(y, pb) - pr_auc(y, pa), "brier": brier(y, pb) - brier(y, pa)}
    bs: dict[str, list[float]] = {k: [] for k in base}
    n = len(y)
    for _ in range(n_boot):
        i = g.integers(0, n, n)
        if y[i].min() == y[i].max():
            continue
        bs["auroc"].append(auroc(y[i], pb[i]) - auroc(y[i], pa[i]))
        bs["pr_auc"].append(pr_auc(y[i], pb[i]) - pr_auc(y[i], pa[i]))
        bs["brier"].append(brier(y[i], pb[i]) - brier(y[i], pa[i]))
    return {k: {"estimate": _r(v, 4), "ci_low": _r(np.quantile(bs[k], 0.025), 4) if bs[k] else None,
                "ci_high": _r(np.quantile(bs[k], 0.975), 4) if bs[k] else None} for k, v in base.items()} | {"n_boot": n_boot, "n_rows": n}


def fp_effect(run: Run, frames: Frames, variable: str) -> pd.DataFrame | None:
    """Partial effect of an FP-transformed continuous predictor on the log-odds (relative to its TRAIN median), over TRAIN p1..p99,
    computed with the saved preprocessor and coefficients (all other predictors fixed): the fitted shape, not a causal effect."""
    from falls_ml.bundle import load_bundle

    bundle = load_bundle(run.run_dir / "model")
    pipe = bundle.pipeline
    if variable not in frames.train.columns:
        return None
    x = pd.to_numeric(frames.train[variable], errors="coerce").dropna()
    if x.nunique() < 3:
        return None
    grid = np.unique(np.linspace(float(x.quantile(0.01)), float(x.quantile(0.99)), 60))
    base_row = frames.train.iloc[[0]].copy()
    rows = pd.concat([base_row] * len(grid), ignore_index=True)
    rows[variable] = grid.astype(frames.train[variable].dtype if str(frames.train[variable].dtype) != "Int64" else "float64")
    lp = pipe.model.linear_predictor(pipe.design(rows))
    ref = pd.concat([base_row], ignore_index=True)
    ref[variable] = float(x.median())
    lp0 = float(pipe.model.linear_predictor(pipe.design(ref))[0])
    return pd.DataFrame({variable: grid, "log_odds_vs_median": lp - lp0})
