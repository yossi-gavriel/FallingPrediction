"""Pure, vectorised feature transforms (spec §5.3–5.9, §6.1, D-02, D-04, D-13, RT-12, RT-13, RT-17).

Every function is stateless and NaN-aware: missing values are mapped by the documented rule or
rejected loudly. Invalid data values raise ``DatasetValidationError``; invalid arguments raise
``ConfigError``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

from falls_ml.errors import ConfigError, DatasetValidationError

BMI_LEVELS: tuple[str, ...] = ("underweight", "normal", "overweight", "obese", "missing")
SMOKING_LEVELS: tuple[str, ...] = ("never", "ex", "current")
MISSING_LEVEL = "missing"

#: WHO cut-points (D-02): underweight < 18.5 <= normal < 25 <= overweight < obese cut-point <= obese.
BMI_UNDERWEIGHT_BELOW = 18.5
BMI_OVERWEIGHT_FROM = 25.0


# ---------------------------------------------------------------------- helpers
def _series(values: pd.Series | Sequence | np.ndarray, name: str) -> pd.Series:
    return values if isinstance(values, pd.Series) else pd.Series(values, name=name)


def _float_array(values: pd.Series, name: str) -> np.ndarray:
    try:
        arr = values.to_numpy(dtype="float64", na_value=np.nan)
    except (TypeError, ValueError) as exc:
        raise DatasetValidationError(f"{name}: expected numeric values, got dtype {values.dtype}") from exc
    if np.isinf(arr).any():
        raise DatasetValidationError(f"{name}: {int(np.isinf(arr).sum())} infinite values")
    return arr


def _checked_levels(values: pd.Series, allowed: Sequence[str], name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (object values, missing mask); undeclared non-missing levels raise."""
    missing = values.isna().to_numpy()
    obj = values.astype(object).to_numpy()
    observed = set(obj[~missing].tolist())
    bad = sorted(map(str, observed - set(allowed)))
    if bad:
        raise DatasetValidationError(f"{name}: undeclared levels {bad[:10]} (allowed {list(allowed)})")
    return obj, missing


def _str_series(values: np.ndarray, like: pd.Series) -> pd.Series:
    return pd.Series(values, index=like.index, name=like.name, dtype="str")


# ---------------------------------------------------------------------- categorical derivations
def bmi_category(bmi: pd.Series, obese_cutpoint: float = 30.0) -> pd.Series:
    """BMI value → WHO category (D-02; RT-17 boundaries); NaN → ``missing``.

    underweight < 18.5 <= normal < 25 <= overweight < ``obese_cutpoint`` <= obese.
    """
    if not obese_cutpoint > BMI_OVERWEIGHT_FROM:
        raise ConfigError(f"bmi_obese_cutpoint must be > {BMI_OVERWEIGHT_FROM}, got {obese_cutpoint}")
    s = _series(bmi, "bmi_value")
    b = _float_array(s, "bmi_value")
    out = np.select(
        [np.isnan(b), b < BMI_UNDERWEIGHT_BELOW, b < BMI_OVERWEIGHT_FROM, b < obese_cutpoint],
        [MISSING_LEVEL, "underweight", "normal", "overweight"],
        default="obese",
    ).astype(object)
    return _str_series(out, s)


def smoking_published(status: pd.Series) -> pd.Series:
    """Published smoking term (§5.7, D-04): ``current`` → ``current``; never/ex/missing → ``ex_never``."""
    s = _series(status, "smoking_status")
    obj, missing = _checked_levels(s, SMOKING_LEVELS, "smoking_status")
    filled = np.where(missing, "never", obj)  # compare only non-missing values (pd.NA == x is ambiguous)
    return _str_series(np.where(filled == "current", "current", "ex_never").astype(object), s)


def smoking_all_levels(status: pd.Series) -> pd.Series:
    """Retrained smoking candidate (D-04, D-10): never/ex/current with missing merged into ``never``."""
    s = _series(status, "smoking_status")
    obj, missing = _checked_levels(s, SMOKING_LEVELS, "smoking_status")
    return _str_series(np.where(missing, "never", obj).astype(object), s)


def alcohol_level(category: pd.Series, levels: Sequence[str]) -> pd.Series:
    """Alcohol category (§5.8, D-05): declared level, or ``missing`` when null. Undeclared levels raise."""
    if MISSING_LEVEL in levels:
        raise ConfigError(f"alcohol levels must not declare {MISSING_LEVEL!r} (it is derived from nulls)")
    s = _series(category, "alcohol_category")
    obj, missing = _checked_levels(s, levels, "alcohol_category")
    return _str_series(np.where(missing, MISSING_LEVEL, obj).astype(object), s)


# ---------------------------------------------------------------------- polypharmacy
def log_polypharmacy(count: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    """Published polypharmacy term ln((P + 1) / 10) (§5.5, RT-12). Nulls and negative counts raise.

    Returns a float64 Series (same index) for Series input, otherwise a float64 ndarray.
    """
    s = _series(count, "polypharmacy_count_120d")
    p = _float_array(s, "polypharmacy_count_120d")
    if np.isnan(p).any():
        raise DatasetValidationError(f"polypharmacy_count_120d: {int(np.isnan(p).sum())} null counts (absent must be 0)")
    if (p < 0).any():
        raise DatasetValidationError(f"polypharmacy_count_120d: {int((p < 0).sum())} negative counts")
    out = np.log((p + 1.0) / 10.0)
    if isinstance(count, pd.Series):
        return pd.Series(out, index=count.index, name=count.name, dtype="float64")
    return out


# ---------------------------------------------------------------------- fractional polynomials
def fp_auto_scaling(x: pd.Series | np.ndarray) -> tuple[float, float]:
    """Stata ``fp, scale`` automatic pre-scaling (§6.1, D-13, RT-13): returns ``(shift, scale)``.

    If min(x) <= 0, shift = -min(x) + counting interval (smallest gap between sorted distinct values),
    else 0. With r = range of x + shift and p = log10(r), scale = 10 ** (sign(p) * floor(|p|)).
    The transformed variable is (x + shift) / scale.
    """
    arr = np.asarray(x, dtype="float64")
    if arr.ndim != 1 or arr.size == 0:
        raise DatasetValidationError("fp_auto_scaling: expected a non-empty 1-D vector")
    if not np.isfinite(arr).all():
        raise DatasetValidationError("fp_auto_scaling: values must be finite and non-null")
    distinct = np.unique(arr)
    if distinct.size < 2:
        raise DatasetValidationError("fp_auto_scaling: constant variable cannot be scaled")
    lo, hi = float(distinct[0]), float(distinct[-1])
    shift = -lo + float(np.diff(distinct).min()) if lo <= 0 else 0.0
    p = math.log10((hi + shift) - (lo + shift))
    scale = 10.0 ** (math.copysign(1.0, p) * math.floor(abs(p)))
    return float(shift), float(scale)


def normalize_fp_powers(powers: Sequence[float]) -> tuple[float, ...]:
    """Validate FP powers (non-empty, non-decreasing) and return them as a float tuple."""
    pw = tuple(float(p) + 0.0 for p in powers)  # + 0.0 normalises -0.0
    if not pw:
        raise ConfigError("FP powers must be non-empty")
    if any(b < a for a, b in zip(pw, pw[1:])):
        raise ConfigError(f"FP powers must be non-decreasing, got {pw}")
    return pw


def fp_basis(z: pd.Series | np.ndarray, powers: Sequence[float]) -> np.ndarray:
    """FP basis matrix (n, m) of the (pre-scaled) variable ``z`` (§6.1).

    Power 0 is ln(z). Repeated powers: H1 = z^p1; Hj = H_{j-1} * ln(z) if pj == p_{j-1} else z^pj.
    ``z`` must be strictly positive unless ``powers == (1,)`` (a plain linear term).
    """
    pw = normalize_fp_powers(powers)
    zz = np.asarray(z, dtype="float64")
    if zz.ndim != 1:
        raise DatasetValidationError("fp_basis: expected a 1-D vector")
    if not np.isfinite(zz).all():
        raise DatasetValidationError("fp_basis: values must be finite and non-null")
    if pw == (1.0,):
        return zz.reshape(-1, 1).copy()
    if (zz <= 0).any():
        raise DatasetValidationError(
            f"fp_basis: {int((zz <= 0).sum())} values <= 0 after FP pre-scaling (outside the domain fixed at fit)")
    log_z = np.log(zz)
    cols: list[np.ndarray] = []
    for j, p in enumerate(pw):
        if j > 0 and p == pw[j - 1]:
            cols.append(cols[-1] * log_z)
        else:
            cols.append(log_z if p == 0 else np.power(zz, p))
    return np.column_stack(cols)


def fp_column_names(base: str, powers: Sequence[float]) -> list[str]:
    """Design-column names for FP terms, e.g. ``x__fp_p0``, ``x__fp_p0_ln``, ``x__fp_p-2`` (compact powers)."""
    pw = normalize_fp_powers(powers)
    names: list[str] = []
    repeats = 0
    for j, p in enumerate(pw):
        repeats = repeats + 1 if j > 0 and p == pw[j - 1] else 0
        suffix = "" if repeats == 0 else ("_ln" if repeats == 1 else f"_ln{repeats}")
        names.append(f"{base}__fp_p{p:g}{suffix}")
    return names
