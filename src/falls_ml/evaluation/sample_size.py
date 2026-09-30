"""Minimum sample sizes for developing and validating binary-outcome prediction models (spec §7.6; RT-09, RT-10).

Development: Riley et al. Stat Med 2019;38:1276 (criteria i–iii), emulating Stata ``pmsampsize`` 1.3.2.
External validation: Riley et al. Stat Med 2021;40:4230, emulating Stata ``pmvalsampsize`` 1.0.1.
Formulas and verification: docs/research_notes/validation_methods.md §8–9.
"""

from __future__ import annotations

import math
import numbers
from typing import Any

Z_975 = 1.96  # the packages and papers use 1.96, not the exact normal quantile
_CEIL_EPS = 1e-9  # guards ceil() against floating-point noise just above an integer


def _ceil(x: float) -> int:
    return math.ceil(x - _CEIL_EPS)


def _check_prevalence(prevalence: float) -> float:
    if not 0.0 < prevalence < 1.0:
        raise ValueError("prevalence must lie in (0, 1)")
    return float(prevalence)


def max_r2_cox_snell(prevalence: float) -> float:
    """Maximum Cox–Snell R^2 = 1 - (phi^phi (1 - phi)^(1 - phi))^2 (Riley 2019 eq 23)."""
    phi = _check_prevalence(prevalence)
    return 1.0 - math.exp(2.0 * (phi * math.log(phi) + (1.0 - phi) * math.log(1.0 - phi)))


def _round_half_up(x: float, digits: int) -> float:
    factor = 10**digits
    return math.floor(x * factor + 0.5) / factor


def _n_for_shrinkage(n_parameters: int, r2_cs: float, shrinkage: float) -> float:
    """Riley 2019 eq 11: n = p / ((S - 1) ln(1 - R2_CS / S))."""
    return n_parameters / ((shrinkage - 1.0) * math.log(1.0 - r2_cs / shrinkage))


def riley2019_binary_development(prevalence: float, n_parameters: int, *, r2_nagelkerke: float | None = None,
                                 r2_cox_snell: float | None = None, shrinkage: float = 0.9, margin: float = 0.05,
                                 max_r2_difference: float = 0.05, pmsampsize_rounding: bool = True) -> dict[str, Any]:
    """Riley 2019 development sample size (binary outcome).

    Criterion 1: expected uniform shrinkage >= ``shrinkage``. Criterion 2: apparent vs adjusted Nagelkerke R^2
    differ by <= ``max_r2_difference``. Criterion 3: overall risk within ± ``margin``. Each criterion n is
    rounded up; ``events_required = ceil(n · prevalence)``; ``epp = n · prevalence / n_parameters``.

    ``pmsampsize_rounding`` (RT-09) rounds the Nagelkerke → Cox–Snell conversion to 3 decimals as ``pmsampsize``
    does (R2_CS = round(R2_Nag · max R2_CS, 3)); it has no effect when ``r2_cox_snell`` is given.
    """
    phi = _check_prevalence(prevalence)
    if isinstance(n_parameters, bool) or not isinstance(n_parameters, numbers.Integral) or n_parameters < 1:
        raise ValueError("n_parameters must be a positive integer")
    if (r2_nagelkerke is None) == (r2_cox_snell is None):
        raise ValueError("give exactly one of r2_nagelkerke or r2_cox_snell")
    if not 0.0 < shrinkage < 1.0 or margin <= 0.0 or max_r2_difference <= 0.0:
        raise ValueError("shrinkage must lie in (0, 1); margin and max_r2_difference must be positive")
    max_r2 = max_r2_cox_snell(phi)
    if r2_nagelkerke is not None:
        if not 0.0 < r2_nagelkerke < 1.0:
            raise ValueError("r2_nagelkerke must lie in (0, 1)")
        r2_cs = r2_nagelkerke * max_r2
        if pmsampsize_rounding:
            r2_cs = _round_half_up(r2_cs, 3)
    else:
        r2_cs = float(r2_cox_snell)  # type: ignore[arg-type]
    if not 0.0 < r2_cs < max_r2:
        raise ValueError(f"Cox–Snell R^2 {r2_cs} must lie in (0, max R^2_CS = {max_r2:.5f})")
    n1 = _ceil(_n_for_shrinkage(n_parameters, r2_cs, shrinkage))
    shrinkage_2 = r2_cs / (r2_cs + max_r2_difference * max_r2)  # Riley 2019 eq 26
    n2 = _ceil(_n_for_shrinkage(n_parameters, r2_cs, shrinkage_2))
    n3 = _ceil((Z_975 / margin) ** 2 * phi * (1.0 - phi))  # eq 27
    n = max(n1, n2, n3)
    return {"max_r2_cs": max_r2, "r2_cs": r2_cs, "n_criterion_1": n1, "n_criterion_2": n2, "n_criterion_3": n3,
            "shrinkage_criterion_2": shrinkage_2, "n_required": n, "events_required": _ceil(n * phi),
            "epp": n * phi / n_parameters}


def newcombe_cstatistic_se(n: int, c_statistic: float, prevalence: float) -> float:
    """Approximate SE of the C-statistic (Newcombe; Riley 2021 eq 11)."""
    c, phi = c_statistic, prevalence
    a = n / 2.0 - 1.0
    return math.sqrt(c * (1 - c) * (1 + a * (1 - c) / (2 - c) + a * c / (1 + c)) / (n * n * phi * (1 - phi)))


def riley2021_validation_cstatistic(prevalence: float, c_statistic: float, ci_width: float = 0.1) -> dict[str, int]:
    """Smallest n whose Wald CI for C (Newcombe SE) is no wider than ``ci_width`` (RT-10).

    The SE is strictly decreasing in n, so the smallest n is found by bisection.
    """
    phi = _check_prevalence(prevalence)
    if not 0.5 <= c_statistic < 1.0 or ci_width <= 0.0:
        raise ValueError("c_statistic must lie in [0.5, 1) and ci_width must be positive")

    def wide(n: int) -> bool:
        return 2 * Z_975 * newcombe_cstatistic_se(n, c_statistic, phi) > ci_width

    if not wide(1):
        return {"n": 1, "events": _ceil(phi)}
    lo, hi = 1, 2
    while wide(hi):
        lo, hi = hi, 2 * hi
    while hi - lo > 1:  # invariant: wide(lo) and not wide(hi)
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if wide(mid) else (lo, mid)
    return {"n": hi, "events": _ceil(hi * phi)}


def oe_se_for_ci_width(oe: float = 1.0, ci_width: float = 0.2, increment: float = 1e-4) -> float:
    """``pmvalsampsize`` search: SE(ln O/E) grown in ``increment`` steps until the back-transformed CI width
    reaches ``ci_width`` (O/E 1, width 0.2 -> 0.051)."""
    if oe <= 0 or ci_width <= 0 or increment <= 0:
        raise ValueError("oe, ci_width and increment must be positive")
    k = 0
    while True:
        k += 1
        se = k * increment
        if math.exp(math.log(oe) + Z_975 * se) - math.exp(math.log(oe) - Z_975 * se) >= ci_width:
            return round(se, 10)


def riley2021_validation_oe(prevalence: float, se_ln_oe: float = 0.051) -> dict[str, int]:
    """n = ceil((1 - phi) / (phi · SE(ln O/E)^2)) (Riley 2021 eq 5), events = ceil(n · phi).

    For eFalls (phi 0.048, SE 0.051) this gives 7,626 (367); the published Table S2.2 prints 7,625 (366),
    i.e. the raw value 7,625.27 rounded down — the published figure is one below the ``pmvalsampsize`` ceil.
    """
    phi = _check_prevalence(prevalence)
    if se_ln_oe <= 0:
        raise ValueError("se_ln_oe must be positive")
    n = _ceil((1.0 - phi) / (phi * se_ln_oe**2))
    return {"n": n, "events": _ceil(n * phi)}
