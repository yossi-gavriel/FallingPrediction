"""S04 cheap exhaustive screening on TRAIN only (planning/EXPERIMENT_PLAN.md §4): per-candidate univariate description and signal,
Spearman redundancy clusters (label-blind) and data-quality flags. No candidate is removed for weak univariate association."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from falls_ml.eda.common import auroc as uni_auroc
from falls_ml.eda.common import prevalence_ratio, smd

ELIGIBLE = ("ELIGIBLE", "ELIGIBLE_EXPLORATORY")


def _r(v: Any, d: int = 4) -> float | None:
    return None if v is None or not np.isfinite(v) else round(float(v), d)


def _suppress(n: int, min_cell: int) -> int | str:
    return n if n == 0 or n >= min_cell else f"<{min_cell}"


def univariate_screen(values: pd.DataFrame, y: np.ndarray, registry: pd.DataFrame, *, min_cell: int, sparse_min_events: int,
                      baseline: pd.DataFrame | None = None) -> pd.DataFrame:
    """TRAIN rows only. ``values``: engineered candidates; ``baseline``: canonical BASELINE_15 values (reference rows)."""
    y = np.asarray(y, dtype=int)
    reg = registry.set_index("feature")
    cols: list[tuple[str, pd.Series, str]] = [(f, values[f], "candidate") for f in values.columns]
    if baseline is not None:
        for f in baseline.columns:
            s = baseline[f]
            s = (s.astype(str) == "female").astype(float) if f == "sex" else pd.to_numeric(s, errors="coerce").astype(float)
            cols.append((f, s, "BASELINE_15"))
    rows = []
    for name, s, role in cols:
        x = s.to_numpy(dtype=float)
        obs = np.isfinite(x)
        n_obs = int(obs.sum())
        yo = y[obs]
        uniq = np.unique(x[obs])
        binary = set(uniq.tolist()) <= {0.0, 1.0}
        e_obs, e_mis = int(yo.sum()), int(y[~obs].sum())
        row: dict[str, Any] = {"feature": name, "role": role, "eligibility": reg.at[name, "eligibility"] if name in reg.index else "BASELINE_15",
                               "domain": reg.at[name, "domain"] if name in reg.index else "BASELINE_15", "n_train": int(len(x)),
                               "n_observed": _suppress(n_obs, min_cell), "missing_pct": round(100.0 * (1 - n_obs / len(x)), 3) if len(x) else None,
                               "outcome_rate_observed_pct": _r(100.0 * yo.mean(), 3) if n_obs >= min_cell and e_obs >= min_cell else None,
                               "outcome_rate_missing_pct": _r(100.0 * y[~obs].mean(), 3) if (~obs).sum() >= min_cell and e_mis >= min_cell else None}
        if binary and n_obs:
            n1, n0 = int((x[obs] == 1).sum()), int((x[obs] == 0).sum())
            a, c = int(yo[x[obs] == 1].sum()), int(yo[x[obs] == 0].sum())
            pr = prevalence_ratio(a, n1, c, n0) if n1 and n0 else (None, None, None)
            ok = min(n1, n0) >= min_cell and min(a, c) >= min_cell and min(n1 - a, n0 - c) >= min_cell
            row.update({"distribution": f"prevalence {100.0 * n1 / n_obs:.2f}% of observed", "n_level_1": _suppress(n1, min_cell),
                        "events_level_1": _suppress(a, min_cell), "outcome_rate_level_1_pct": _r(100.0 * a / n1, 3) if ok else None,
                        "outcome_rate_level_0_pct": _r(100.0 * c / n0, 3) if ok else None, "prevalence_ratio": _r(pr[0], 3) if ok else None,
                        "prevalence_ratio_ci_low": _r(pr[1], 3) if ok else None, "prevalence_ratio_ci_high": _r(pr[2], 3) if ok else None,
                        "sparse_flag": bool(min(a, c) < sparse_min_events), "separation_flag": bool(a == 0 or c == 0 or a == n1 or c == n0)})
        elif n_obs:
            q = np.nanpercentile(x[obs], [1, 25, 50, 75, 99])
            row.update({"distribution": f"median {q[2]:.3g} [IQR {q[1]:.3g}-{q[3]:.3g}; p1 {q[0]:.3g}, p99 {q[4]:.3g}]", "sparse_flag": False,
                        "separation_flag": False})
        s_d = smd(x[obs & (y == 1)], x[obs & (y == 0)]) if n_obs else None
        auc, lo, hi = uni_auroc(x, y) if n_obs else (None, None, None)
        row.update({"smd": _r(s_d, 4), "univariate_auroc": _r(auc, 4), "univariate_auroc_ci_low": _r(lo, 4), "univariate_auroc_ci_high": _r(hi, 4),
                    "abs_auroc_deviation": _r(abs(auc - 0.5), 4) if auc is not None else None, "n_unique_observed": int(len(uniq))})
        rows.append(row)
    return pd.DataFrame(rows)


def rank_spearman(X: pd.DataFrame, min_pairs: int = 100) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Spearman correlation as pairwise-complete Pearson correlation of each column's ranks (ranks over that column's observed rows).
    Returns (correlation, pairwise-complete n). Vectorised: O(n p^2)."""
    A = X.to_numpy(dtype=float)
    M = np.isfinite(A).astype(float)
    R = np.zeros_like(A)
    for j in range(A.shape[1]):
        ok = M[:, j] > 0
        if ok.any():
            R[ok, j] = rankdata(A[ok, j])
    n = M.T @ M
    sx = R.T @ M                   # sx[i, j] = sum of rank_i over rows where both i and j are observed
    sxx = (R * R).T @ M
    sxy = R.T @ R
    with np.errstate(divide="ignore", invalid="ignore"):
        cov = sxy - sx * sx.T / n
        vi = sxx - sx * sx / n
        vj = vi.T
        corr = cov / np.sqrt(vi * vj)
    corr[(n < min_pairs) | ~np.isfinite(corr)] = np.nan
    np.fill_diagonal(corr, 1.0)
    names = list(X.columns)
    return pd.DataFrame(corr, index=names, columns=names), pd.DataFrame(n, index=names, columns=names)


def redundancy_clusters(corr: pd.DataFrame, *, threshold: float, missing_pct: dict[str, float], order: dict[str, int],
                        baseline: set[str]) -> tuple[pd.DataFrame, dict[str, str]]:
    """Representative-anchored clusters (no chaining): features are visited in priority order (BASELINE_15 features first, then the lowest
    missingness, then catalogue order, then name); each unassigned feature becomes a representative and takes every unassigned feature whose
    |Spearman| with IT is >= threshold. Label-blind. Returns (clusters table, feature -> representative for every clustered feature)."""
    names = list(corr.columns)
    C = corr.to_numpy().copy()
    pos = {n: i for i, n in enumerate(names)}
    ranked = sorted(names, key=lambda f: (f not in baseline, missing_pct.get(f, 0.0), order.get(f, 10**6), f))
    unassigned = set(names)
    rows, rep_of = [], {}
    k = 0
    for rep in ranked:
        if rep not in unassigned:
            continue
        unassigned.discard(rep)
        i = pos[rep]
        members = [g for g in ranked if g in unassigned and np.isfinite(C[i, pos[g]]) and abs(C[i, pos[g]]) >= threshold]
        if not members:
            continue
        for g in members:
            unassigned.discard(g)
        k += 1
        allm = [rep, *members]
        vals = [abs(C[i, pos[g]]) for g in members]
        for m in allm:
            rep_of[m] = rep
        rows.append({"cluster": f"R{k:02d}", "n_members": len(allm), "representative": rep, "members": "; ".join(allm),
                     "min_abs_spearman_with_representative": round(float(min(vals)), 3), "max_abs_spearman_with_representative": round(float(max(vals)), 3),
                     "contains_baseline_feature": any(m in baseline for m in allm),
                     "rule": "representative-anchored (no chaining); representative = BASELINE_15 feature if present, else lowest missingness, "
                             "then catalogue order, then name (label-blind)"})
    return pd.DataFrame(rows), rep_of
