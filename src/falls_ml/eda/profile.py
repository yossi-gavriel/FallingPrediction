"""Unsupervised column profiling of the whole extract (no outcome is used here).

``column_profile``      one row per contract column: type, counts, missingness, uniqueness, constant status, top values, numeric
                        quantiles, date range, sentinels, parse failures and their text patterns, out-of-contract values.
``numeric_profile``     numeric / count / score columns: distribution shape, zero inflation, extremes and transformations TO INVESTIGATE
                        (reported only; nothing is transformed here).
``categorical_profile`` level counts of binary / ordinal / categorical columns, NULL shown as its own level, sparse levels flagged.

Identifier values are never listed; counts 1..min_cell-1 are suppressed.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from falls_ml.data.meuhedet_wide import NA_VALUES, WideContract
from falls_ml.eda.common import SUPPRESSED, cell, fnum, numeric, quantiles
from falls_ml.eda.dictionary import DataDictionary

NEAR_CONSTANT_SHARE = 0.995
FAR_FUTURE = pd.Timestamp("2100-01-01")
TOO_EARLY = pd.Timestamp("1900-01-01")
TIME_ONLY_RE = re.compile(r"^\s*\d{1,2}:\d{2}(:\d{2})?(\.\d+)?\s*$")
CATEGORICAL_SEMS = ("binary", "ordinal", "categorical", "label", "quality_control")
NUMERIC_SEMS = ("count", "continuous", "ordinal")


def is_identifier(contract: WideContract, col: str) -> bool:
    c = contract.get(col)
    return c.role == "IDENTIFIER" or c.semantic == "identifier"


def text_pattern(v: str) -> str:
    """Privacy-safe shape of a raw text value: digits -> 9, letters (Latin / Hebrew) -> a, other characters kept, runs of 'a' capped."""
    s = re.sub(r"\d", "9", v.strip())
    s = re.sub(r"[A-Za-z֐-׿]", "a", s)
    return re.sub(r"a{4,}", "aaa…", s)[:24] or "(blank)"


def failure_patterns(path: Path, source_format: str, frame: pd.DataFrame, wrong_type: dict[str, int], *, encoding: str, sep: str,
                     min_cell: int) -> tuple[dict[str, dict[str, Any]], dict[str, pd.Series]]:
    """Shapes of the raw values that could not be parsed (they are NULL in the typed frame) and the cell masks, CSV input only
    (Excel / Parquet: counts only). Returns ({column: {pattern: count}}, {column: mask})."""
    cols = [c for c, n in wrong_type.items() if n]
    if not cols or source_format != "csv":
        return {}, {}
    raw = pd.read_csv(path, dtype="string", na_values=list(NA_VALUES), keep_default_na=False, encoding=encoding, sep=sep, usecols=cols)
    raw.columns = [str(c).strip() for c in raw.columns]
    patterns: dict[str, dict[str, Any]] = {}
    masks: dict[str, pd.Series] = {}
    for c in cols:
        if c not in raw.columns:
            continue
        txt = raw[c].str.strip()
        bad = txt.notna() & (txt != "") & frame[c].isna().to_numpy()
        masks[c] = pd.Series(bad.to_numpy(), index=frame.index)
        counts = txt[bad].map(text_pattern).value_counts()
        patterns[c] = {str(k): cell(v, min_cell) for k, v in counts.head(8).items()}
    return patterns, masks


def _top_values(s: pd.Series, min_cell: int, n: int, top: int = 8) -> str:
    vc = s.astype("string").fillna("NULL").value_counts()
    parts = []
    for k, v in vc.head(top).items():
        c = cell(v, min_cell)
        parts.append(f"{k}: {c}" + (f" ({100.0 * v / n:.1f}%)" if c != SUPPRESSED else ""))
    if len(vc) > top:
        parts.append(f"... {len(vc) - top} more levels")
    return "; ".join(parts)


def constant_status(s: pd.Series) -> tuple[str, float | None]:
    nn = s.dropna()
    if nn.empty:
        return "EMPTY", None
    share = float(nn.value_counts(normalize=True).iloc[0])
    if nn.nunique() <= 1:
        return "CONSTANT", share
    if share >= NEAR_CONSTANT_SHARE:
        return "NEAR_CONSTANT", share
    return "VARIABLE", share


def column_profile(frame: pd.DataFrame, contract: WideContract, dictionary: DataDictionary, *, index: pd.Series, wrong_type: dict[str, int],
                   patterns: dict[str, dict[str, Any]], invalid_values: dict[str, int], bases: dict[str, pd.Series], min_cell: int,
                   dict_table: pd.DataFrame) -> pd.DataFrame:
    """One row per contract column on the FULL extract (+ missing % per basis). ``index`` = the row's Index_Date (normalised)."""
    n = len(frame)
    dt = dict_table.set_index("column")
    rows = []
    for c in contract.columns:
        name = c.name
        base = {"column": name, "domain": dt.at[name, "domain"], "declared_type": c.semantic, "sql_type": c.sql, "contract_role": c.role,
                "inferred_pandas_dtype": str(frame[name].dtype) if name in frame.columns else "", "present_in_file": name in frame.columns}
        if name not in frame.columns:
            rows.append({**base, "status": "ABSENT_FROM_FILE"})
            continue
        s = frame[name]
        nn = int(s.notna().sum())
        miss = n - nn
        status, share = constant_status(s)
        r: dict[str, Any] = {**base, "row_count": n, "non_null_count": cell(nn, min_cell), "missing_count": cell(miss, min_cell),
                             "missing_pct": fnum(100.0 * miss / n, 2) if cell(miss, min_cell) != SUPPRESSED else SUPPRESSED,
                             "unique_count": int(s.nunique(dropna=True)), "status": status,
                             "top_value_share_of_non_null": fnum(share, 4)}
        for bname, mask in bases.items():
            m = int(mask.sum())
            k = int(s[mask].isna().sum())
            r[f"missing_pct_{bname.lower()}"] = (fnum(100.0 * k / m, 2) if cell(k, min_cell) != SUPPRESSED else SUPPRESSED) if m else None
        if is_identifier(contract, name):
            r["top_values"] = "(identifier - values never shown)"
            r["duplicate_values"] = int(s.dropna().duplicated().sum())
        elif c.is_date:
            d = s.dropna()
            r["date_min"] = str(d.min().date()) if len(d) else None
            r["date_max"] = str(d.max().date()) if len(d) else None
            sent = s.isin([pd.Timestamp(v) for v in c.sentinels]) if c.sentinels else pd.Series(False, index=s.index)
            r["declared_sentinel_count"] = cell(int(sent.sum()), min_cell)
            r["undeclared_far_future_count"] = cell(int(((s >= FAR_FUTURE) & ~sent).sum()), min_cell)
            r["before_1900_count"] = cell(int((s < TOO_EARLY).sum()), min_cell)
            if c.role != "LABEL" and c.timing in ("pre_index", "at_index", "unknown"):
                r["on_or_after_index_count"] = cell(int(((s.dt.normalize() >= index) & ~sent).sum()), min_cell)
        elif c.is_text:
            r["top_values"] = _top_values(s, min_cell, n)
            r["time_only_text_count"] = cell(int(s.dropna().astype(str).str.match(TIME_ONLY_RE).sum()), min_cell)
        else:
            v = numeric(s)
            q = quantiles(v)
            x = v.dropna()
            r.update({"min": fnum(x.min(), 4) if len(x) else None, "max": fnum(x.max(), 4) if len(x) else None,
                      "mean": fnum(x.mean(), 4) if len(x) else None, "sd": fnum(x.std(ddof=1), 4) if len(x) > 1 else None,
                      "median": q["p50"], "iqr": fnum(q["p75"] - q["p25"], 4) if q["p75"] is not None else None, **q,
                      "zero_count": cell(int((x == 0).sum()), min_cell), "negative_count": cell(int((x < 0).sum()), min_cell)})
            if c.semantic in CATEGORICAL_SEMS or c.allowed is not None or s.nunique() <= 12:
                r["top_values"] = _top_values(s, min_cell, n)
        r["parse_failures"] = cell(int(wrong_type.get(name, 0)), min_cell)
        r["parse_failure_patterns"] = "; ".join(f"{k}: {v}" for k, v in (patterns.get(name) or {}).items())
        r["out_of_contract_values"] = cell(int(invalid_values.get(name, 0)), min_cell)
        if c.allowed is not None and not c.is_date:
            allowed = {str(int(a)) if c.is_integer else str(a) for a in c.allowed}
            seen = s.dropna().astype("string")
            bad = seen[~seen.isin(allowed)].value_counts()
            r["out_of_contract_levels"] = "; ".join(f"{k}: {cell(v, min_cell)}" for k, v in bad.head(5).items())
        r.update({"available_at_prediction_time": dt.at[name, "available_at_prediction_time"], "used_by_strict": dt.at[name, "used_by_strict"],
                  "used_by_extended": dt.at[name, "used_by_extended"], "mapping_confidence": dt.at[name, "mapping_confidence"],
                  "meaning_status": dt.at[name, "meaning_status"]})
        rows.append(r)
    return pd.DataFrame(rows)


def numeric_columns(contract: WideContract, frame: pd.DataFrame) -> list[str]:
    """Numeric / count / score columns worth a distribution analysis (identifiers, dates, text and labels excluded)."""
    out = []
    for c in contract.columns:
        if c.name not in frame.columns or c.is_date or c.is_text or c.role in ("IDENTIFIER", "LABEL"):
            continue
        if c.semantic in NUMERIC_SEMS or (c.semantic == "quality_control" and c.allowed is None):
            out.append(c.name)
    return out


def categorical_columns(contract: WideContract, frame: pd.DataFrame) -> list[str]:
    """Binary / ordinal / categorical columns (incl. coded QA fields and categorical labels); identifiers and free dates excluded."""
    out = []
    for c in contract.columns:
        if c.name not in frame.columns or c.is_date or is_identifier(contract, c.name):
            continue
        if c.semantic in ("binary", "ordinal", "categorical") or (c.semantic in ("quality_control", "label") and (c.allowed is not None or c.is_text)):
            out.append(c.name)
    return out


def transformation_hint(x: pd.Series, semantic: str) -> str:
    """Transformations TO INVESTIGATE (never applied here; any choice is made on TRAIN only)."""
    x = x.dropna()
    if x.empty:
        return "no data"
    hints = []
    if (x < 0).any() and semantic == "count":
        return "negative counts: data error to resolve first; do not transform"
    if x.nunique() <= 2:
        return "binary-valued: no transformation"
    if semantic == "ordinal":
        return "ordinal score: keep as ordered levels (missing category separate); linear-trend vs categorical coding to compare on TRAIN"
    zero = float((x == 0).mean())
    skew = float(x.skew()) if len(x) > 2 else 0.0
    if zero >= 0.5:
        hints.append(f"zero-inflated ({100 * zero:.0f}% zeros): investigate an any/none indicator plus the count among non-zero (hurdle)")
    if skew > 1 and x.min() >= 0:
        hints.append("right-skewed: investigate log1p or fractional polynomials (FP, as in the eFalls process)")
    q99, mx = float(x.quantile(0.99)), float(x.max())
    if mx > 0 and q99 > 0 and mx > 3 * q99:
        hints.append("heavy upper tail (max > 3 x p99): investigate winsorising at the TRAIN p99")
    return "; ".join(hints) or "no transformation suggested by the distribution"


def numeric_profile(frame: pd.DataFrame, contract: WideContract, dictionary: DataDictionary, cols: list[str], bases: dict[str, pd.Series],
                    min_cell: int) -> pd.DataFrame:
    """Distribution summary per numeric column and basis (unsupervised)."""
    rows = []
    for name in cols:
        c = contract.get(name)
        for bname, mask in bases.items():
            v = numeric(frame.loc[mask, name])
            x = v.dropna()
            n = int(mask.sum())
            q = quantiles(v)
            r: dict[str, Any] = {"column": name, "basis": bname, "domain": dictionary.domain_label(name), "declared_type": c.semantic,
                                 "contract_role": c.role, "n_rows": n, "n_observed": cell(len(x), min_cell),
                                 "missing_pct": fnum(100.0 * (n - len(x)) / n, 2) if n else None}
            if len(x):
                iqr = (q["p75"] or 0.0) - (q["p25"] or 0.0)
                hi, lo = (q["p75"] or 0.0) + 3 * iqr, (q["p25"] or 0.0) - 3 * iqr
                r.update({"min": fnum(x.min(), 4), "max": fnum(x.max(), 4), "mean": fnum(x.mean(), 4), "sd": fnum(x.std(ddof=1), 4) if len(x) > 1 else None,
                          "median": q["p50"], "iqr": fnum(iqr, 4), **q,
                          "zero_pct": fnum(100.0 * float((x == 0).mean()), 2), "skewness": fnum(x.skew(), 3) if len(x) > 2 else None,
                          "excess_kurtosis": fnum(x.kurt(), 3) if len(x) > 3 else None,
                          "n_above_q3_plus_3iqr": cell(int((x > hi).sum()), min_cell) if iqr > 0 else 0,
                          "n_below_q1_minus_3iqr": cell(int((x < lo).sum()), min_cell) if iqr > 0 else 0,
                          "n_above_p99": cell(int((x > (q["p99"] or np.inf)).sum()), min_cell),
                          "n_negative": cell(int((x < 0).sum()), min_cell), "distinct_values": int(x.nunique()),
                          "transformations_to_investigate": transformation_hint(x, c.semantic)})
            rows.append(r)
    return pd.DataFrame(rows)


def categorical_profile(frame: pd.DataFrame, contract: WideContract, dictionary: DataDictionary, cols: list[str], bases: dict[str, pd.Series],
                        min_cell: int, max_levels: int = 30) -> pd.DataFrame:
    """Level counts (NULL as its own level) per column and basis; sparse = fewer than min_cell rows or < 0.5% of the basis."""
    rows = []
    for name in cols:
        c = contract.get(name)
        allowed = None if c.allowed is None else {str(int(a)) if c.is_integer else str(a) for a in c.allowed}
        for bname, mask in bases.items():
            s = frame.loc[mask, name].astype("string").fillna("NULL")
            n = int(mask.sum())
            vc = s.value_counts()
            for i, (level, k) in enumerate(vc.items()):
                if i >= max_levels:
                    rows.append({"column": name, "basis": bname, "domain": dictionary.domain_label(name), "level": f"(+{len(vc) - max_levels} more levels)",
                                 "n": None, "pct": None, "sparse": None, "out_of_contract": None})
                    break
                kk = cell(k, min_cell)
                rows.append({"column": name, "basis": bname, "domain": dictionary.domain_label(name), "declared_type": c.semantic, "level": str(level),
                             "n": kk, "pct": fnum(100.0 * k / n, 2) if kk != SUPPRESSED and n else (SUPPRESSED if kk == SUPPRESSED else None),
                             "sparse": bool(k < min_cell or (n and k / n < 0.005)),
                             "out_of_contract": bool(allowed is not None and level != "NULL" and level not in allowed)})
    return pd.DataFrame(rows)
