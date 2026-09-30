"""Shared helpers of the EDA layer: analysis bases, small-cell suppression, watermarked CSV output, descriptive statistics.

Every statistic here is descriptive. Nothing returns a p-value: the EDA never ranks or selects predictors by significance.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SUPPRESSED = "<min_cell"

#: analysis bases (every output file declares one; see eda_manifest.json)
FULL_EXTRACT = "FULL_EXTRACT"                  # every row of the file (all snapshots, eligible or not): unsupervised only
ELIGIBLE_LABELLED = "ELIGIBLE_LABELLED"        # eligible rows on the index date with a usable 180-day label (before D-00)
MODELLING_COHORT = "MODELLING_COHORT"          # rows of the modelling cohort (train + validation + test): unsupervised only
TRAIN_ONLY = "TRAIN_ONLY"                      # target-aware analyses that may inform modelling decisions
POSTHOC_DESCRIPTIVE = "POSTHOC_DESCRIPTIVE_FULL_COHORT"   # outcome-stratified, includes test rows: never used for model choice
SPLIT_DIAGNOSTIC = "SPLIT_DIAGNOSTIC"          # partition comparison without outcome stratification
DEFINITIONS = "DEFINITIONS"                    # derived from the contract / mapping / dictionary only (no data)
BASIS_TEXT = {
    FULL_EXTRACT: "every row of the extract file (all snapshots, eligible or not); unsupervised (no outcome used)",
    ELIGIBLE_LABELLED: "eligible rows on the index date with a usable 180-day label (before the D-00 exclusion)",
    MODELLING_COHORT: "rows of the modelling cohort (train + validation + test); unsupervised (no outcome used)",
    TRAIN_ONLY: "TRAIN partition only; target-aware; the only basis allowed to inform modelling decisions",
    POSTHOC_DESCRIPTIVE: "POST-HOC DESCRIPTIVE: outcome-stratified on the full cohort INCLUDING test rows; never used for model or feature choice",
    SPLIT_DIAGNOSTIC: "train / validation / test compared without outcome stratification (a diagnostic; never a reason to re-split)",
    DEFINITIONS: "derived from the contract, mapping and dictionary files only",
}
UNIVARIATE_BANNER = "UNIVARIATE DESCRIPTIVE ASSOCIATION — NOT CAUSAL, NOT MULTIVARIATE IMPORTANCE"


def cell(n: Any, min_cell: int) -> int | str:
    """A count for a shareable table: 0 and counts >= min_cell as is, 1..min_cell-1 suppressed."""
    n = int(n)
    return n if n == 0 or n >= min_cell else SUPPRESSED


def small(n: Any, min_cell: int) -> bool:
    n = int(n)
    return 0 < n < min_cell


def write_csv(df: pd.DataFrame, path: Path, *, watermark: str, basis: str) -> Path:
    """CSV with one leading watermark line (read back with ``pd.read_csv(path, skiprows=1)``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    head = f"{watermark} | basis: {basis} | aggregate, non-identifying; cells < min_cell suppressed\n"
    path.write_text(head + df.to_csv(index=False, lineterminator="\n"), encoding="utf-8", newline="\n")
    return path


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, skiprows=1)


def write_json(obj: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default, ensure_ascii=False), encoding="utf-8", newline="\n")
    return path


def _json_default(v: Any) -> Any:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, (pd.Timestamp,)):
        return str(v.date())
    if isinstance(v, Path):
        return v.name
    return str(v)


def fnum(v: Any, digits: int = 3) -> float | None:
    """Rounded finite float or None."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return round(f, digits) if math.isfinite(f) else None


def numeric(s: pd.Series) -> pd.Series:
    """float64 view of a numeric / boolean / nullable column (NA -> NaN)."""
    if pd.api.types.is_datetime64_any_dtype(s):
        raise TypeError("dates are not numeric")
    return pd.to_numeric(s, errors="coerce").astype("float64")


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float | None, float | None, float | None]:
    """Proportion with its Wilson score 95% interval."""
    if n <= 0:
        return None, None, None
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, centre - half), min(1.0, centre + half)


def prevalence_ratio(a: int, n1: int, c: int, n0: int, z: float = 1.959964) -> tuple[float | None, float | None, float | None]:
    """Risk (prevalence) ratio of group 1 vs group 0 with the Katz log interval; None when a cell is empty."""
    if n1 <= 0 or n0 <= 0 or a <= 0 or c <= 0:
        return None, None, None
    pr = (a / n1) / (c / n0)
    se = math.sqrt(max(1 / a - 1 / n1 + 1 / c - 1 / n0, 0.0))
    return pr, math.exp(math.log(pr) - z * se), math.exp(math.log(pr) + z * se)


def smd(x1: np.ndarray, x0: np.ndarray) -> float | None:
    """Standardised mean difference (group 1 - group 0) / pooled SD (sqrt of the mean of the two variances); works for 0/1 too."""
    x1, x0 = x1[np.isfinite(x1)], x0[np.isfinite(x0)]
    if len(x1) < 2 or len(x0) < 2:
        return None
    v = (np.var(x1, ddof=1) + np.var(x0, ddof=1)) / 2.0
    if v <= 0:
        return 0.0 if np.mean(x1) == np.mean(x0) else None
    return float((np.mean(x1) - np.mean(x0)) / math.sqrt(v))


def auroc(x: np.ndarray, y: np.ndarray) -> tuple[float | None, float | None, float | None]:
    """Univariate concordance of a numeric value with a 0/1 outcome (Mann-Whitney), Hanley-McNeil 95% interval; NaN excluded."""
    from scipy.stats import rankdata

    ok = np.isfinite(x)
    x, y = x[ok], y[ok]
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return None, None, None
    r = rankdata(x)
    auc = (r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    q1, q2 = auc / (2 - auc), 2 * auc * auc / (1 + auc)
    var = (auc * (1 - auc) + (n1 - 1) * (q1 - auc * auc) + (n0 - 1) * (q2 - auc * auc)) / (n1 * n0)
    se = math.sqrt(max(var, 0.0))
    return float(auc), max(0.0, auc - 1.959964 * se), min(1.0, auc + 1.959964 * se)


def cramers_v(a: pd.Series, b: pd.Series) -> float | None:
    """Cramér's V (no continuity correction) over rows where both are observed."""
    from scipy.stats import chi2_contingency

    ok = a.notna() & b.notna()
    if ok.sum() < 2:
        return None
    tab = pd.crosstab(a[ok].astype(str), b[ok].astype(str))
    if min(tab.shape) < 2:
        return None
    chi2 = chi2_contingency(tab.to_numpy(), correction=False)[0]
    return float(math.sqrt(chi2 / (tab.to_numpy().sum() * (min(tab.shape) - 1))))


def quantiles(v: pd.Series) -> dict[str, float | None]:
    x = v.dropna().astype("float64")
    if x.empty:
        return {k: None for k in ("p1", "p5", "p25", "p50", "p75", "p95", "p99")}
    q = x.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {f"p{int(round(p * 100))}": fnum(q[p], 4) for p in q.index}


def med_iqr(x: pd.Series) -> str:
    v = x.dropna().astype("float64")
    if v.empty:
        return "n/a"
    return f"{v.median():.1f} [{v.quantile(0.25):.1f}–{v.quantile(0.75):.1f}]"


def pct(n: int, d: int, digits: int = 1) -> str:
    return "n/a" if d <= 0 else f"{100.0 * n / d:.{digits}f}%"
