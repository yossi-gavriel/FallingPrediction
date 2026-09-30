"""Sample-size regression tests RT-09, RT-10 and the O/E validation criterion (spec §7.6, §12.2).

Reference values: eFalls supplement Tables S2.1/S2.2, HTA Table 3, and the worked examples of Riley et al. 2019/2021,
reproduced in docs/research_notes/validation_methods.md §8–9 (pmsampsize 1.3.2 / pmvalsampsize 1.0.1 conventions).
"""

from __future__ import annotations

import math

import pytest

from falls_ml.evaluation.sample_size import (
    max_r2_cox_snell,
    newcombe_cstatistic_se,
    oe_se_for_ci_width,
    riley2019_binary_development,
    riley2021_validation_cstatistic,
    riley2021_validation_oe,
)

EFALLS_PREVALENCE = 0.048


def test_rt09_efalls_development_pmsampsize_convention() -> None:
    out = riley2019_binary_development(EFALLS_PREVALENCE, 90, r2_nagelkerke=0.05)
    assert out["max_r2_cs"] == pytest.approx(0.31966, abs=5e-6)
    assert out["r2_cs"] == 0.016  # round(0.05 * 0.31966, 3)
    assert (out["n_required"], out["events_required"]) == (50_174, 2_409)
    assert out["n_criterion_1"] == 50_174 and out["n_criterion_2"] < out["n_criterion_1"] and out["n_criterion_3"] == 71
    assert out["epp"] == pytest.approx(50_174 * 0.048 / 90)


def test_rt09_companion_unrounded_cox_snell() -> None:
    out = riley2019_binary_development(EFALLS_PREVALENCE, 90, r2_nagelkerke=0.05, pmsampsize_rounding=False)
    assert out["r2_cs"] == pytest.approx(0.05 * max_r2_cox_snell(EFALLS_PREVALENCE), abs=1e-15)
    assert out["n_required"] in (50_227, 50_228)


@pytest.mark.parametrize(("n_parameters", "nagelkerke", "expected"), [
    (76, 0.15, (13_867, 666)),     # eFalls Table S2.1 row 1 is reproduced only with 76 parameters (E-12)
    (108, 0.15, (19_706, 946)),    # HTA Table 3, falls
    (108, 0.05, (60_209, 2_891)),
])
def test_other_published_development_rows(n_parameters: int, nagelkerke: float, expected: tuple[int, int]) -> None:
    out = riley2019_binary_development(EFALLS_PREVALENCE, n_parameters, r2_nagelkerke=nagelkerke)
    assert (out["n_required"], out["events_required"]) == expected


def test_riley2019_paper_examples() -> None:
    assert max_r2_cox_snell(0.5) == pytest.approx(0.75)
    # criterion (i): p = 20, R2_CS = 0.1, S = 0.9 -> paper "1698" (raw 1698.04, rounded up 1699);
    # criterion (iii) at phi = 0.5 -> 384.2 -> 385
    out = riley2019_binary_development(0.5, 20, r2_cox_snell=0.1)
    assert out["n_criterion_1"] == 1699 and out["n_criterion_3"] == 385
    # criterion (ii) shrinkage with R2_CS 0.1 and max R2 0.33 (phi ~ 0.05) is 0.858
    assert 0.1 / (0.1 + 0.05 * 0.33) == pytest.approx(0.858, abs=5e-4)


def test_riley2019_input_validation() -> None:
    with pytest.raises(ValueError):
        riley2019_binary_development(0.048, 90)
    with pytest.raises(ValueError):
        riley2019_binary_development(0.048, 90, r2_nagelkerke=0.05, r2_cox_snell=0.016)
    with pytest.raises(ValueError):
        riley2019_binary_development(0.048, 90, r2_cox_snell=0.5)  # above max R2_CS
    with pytest.raises(ValueError):
        riley2019_binary_development(1.2, 90, r2_nagelkerke=0.05)


def test_rt10_efalls_validation_cstatistic() -> None:
    out = riley2021_validation_cstatistic(EFALLS_PREVALENCE, 0.743, ci_width=0.1)
    assert out == {"n": 2_027, "events": 98}
    n = out["n"]
    assert 2 * 1.96 * newcombe_cstatistic_se(n, 0.743, EFALLS_PREVALENCE) <= 0.1
    assert 2 * 1.96 * newcombe_cstatistic_se(n - 1, 0.743, EFALLS_PREVALENCE) > 0.1


@pytest.mark.parametrize(("prevalence", "c", "expected_n"), [(0.1, 0.7, 1154), (0.5, 0.8, 302), (0.018, 0.80, 4252),
                                                              (0.018, 0.75, 5125)])
def test_riley2021_cstatistic_paper_examples(prevalence: float, c: float, expected_n: int) -> None:
    assert riley2021_validation_cstatistic(prevalence, c)["n"] == expected_n


def test_efalls_validation_oe_criterion_off_by_one_is_documented() -> None:
    se = oe_se_for_ci_width(oe=1.0, ci_width=0.2)
    assert se == 0.051
    out = riley2021_validation_oe(EFALLS_PREVALENCE, se_ln_oe=se)
    raw = (1 - 0.048) / (0.048 * 0.051**2)
    assert raw == pytest.approx(7625.27, abs=0.01)
    # pmvalsampsize ceil gives 7,626 (367); eFalls Table S2.2 prints 7,625 (366) = floor of the raw value.
    assert out == {"n": 7_626, "events": 367}
    assert math.floor(raw) == 7_625
    # Riley 2021 eq 5 worked examples: 384.5 and 293.9; phi 0.1 with CI width 0.2 -> 3461
    assert riley2021_validation_oe(0.5, 0.051)["n"] == 385
    assert riley2021_validation_oe(0.1, 0.175)["n"] == 294
    assert riley2021_validation_oe(0.1, oe_se_for_ci_width())["n"] == 3461
