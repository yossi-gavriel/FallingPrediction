"""Unit tests for pure feature transforms (spec §5.5–5.8, D-02, D-04, RT-12, RT-13, RT-17)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from falls_ml.errors import ConfigError, DatasetValidationError
from falls_ml.features.transforms import (
    alcohol_level,
    bmi_category,
    fp_auto_scaling,
    fp_basis,
    fp_column_names,
    log_polypharmacy,
    smoking_all_levels,
    smoking_published,
)

ALCOHOL_LEVELS = ("harmful", "higher_risk", "lower_risk", "previous_higher_risk_or_harmful", "zero")


# ---------------------------------------------------------------------- BMI (D-02, RT-17)
def test_bmi_category_who_boundaries_rt17() -> None:
    bmi = pd.Series([18.49, 18.5, 24.99, 25.0, 29.99, 30.0, np.nan], index=list("abcdefg"))
    out = bmi_category(bmi)
    assert out.tolist() == ["underweight", "normal", "normal", "overweight", "overweight", "obese", "missing"]
    assert list(out.index) == list("abcdefg")
    assert pd.api.types.is_string_dtype(out)


def test_bmi_category_nullable_float_and_custom_cutpoint() -> None:
    bmi = pd.Series([35.0, 39.99, 40.0, pd.NA], dtype="Float64")
    assert bmi_category(bmi, obese_cutpoint=40.0).tolist() == ["overweight", "overweight", "obese", "missing"]


@pytest.mark.parametrize("cutpoint", [25.0, 20.0])
def test_bmi_category_rejects_cutpoint_not_above_25(cutpoint: float) -> None:
    with pytest.raises(ConfigError):
        bmi_category(pd.Series([22.0]), obese_cutpoint=cutpoint)


def test_bmi_category_rejects_infinite_and_non_numeric() -> None:
    with pytest.raises(DatasetValidationError):
        bmi_category(pd.Series([np.inf]))
    with pytest.raises(DatasetValidationError):
        bmi_category(pd.Series(["obese"], dtype=object))


# ---------------------------------------------------------------------- smoking (D-04)
def test_smoking_published_merges_never_ex_and_missing() -> None:
    status = pd.Series(["current", "never", "ex", None], dtype="str")
    assert smoking_published(status).tolist() == ["current", "ex_never", "ex_never", "ex_never"]


def test_smoking_all_levels_maps_missing_to_never() -> None:
    status = pd.Series(["current", "never", "ex", None], dtype=object)
    assert smoking_all_levels(status).tolist() == ["current", "never", "ex", "never"]


def test_categorical_transforms_accept_pandas_na() -> None:
    """Nullable ``string`` dtype carries pd.NA (not NaN); comparisons with it must not leak into the mapping."""
    status = pd.Series(["current", pd.NA, "ex"], dtype="string")
    assert smoking_published(status).tolist() == ["current", "ex_never", "ex_never"]
    assert smoking_all_levels(status).tolist() == ["current", "never", "ex"]
    cat = pd.Series([pd.NA, "zero"], dtype="string")
    assert alcohol_level(cat, ALCOHOL_LEVELS).tolist() == ["missing", "zero"]


@pytest.mark.parametrize("fn", [smoking_published, smoking_all_levels])
def test_smoking_unknown_level_raises(fn) -> None:
    with pytest.raises(DatasetValidationError, match="undeclared levels"):
        fn(pd.Series(["current", "sometimes"]))


# ---------------------------------------------------------------------- alcohol (D-05)
def test_alcohol_level_missing_and_declared() -> None:
    cat = pd.Series(["harmful", None, "zero", "lower_risk"], dtype="str")
    assert alcohol_level(cat, ALCOHOL_LEVELS).tolist() == ["harmful", "missing", "zero", "lower_risk"]


def test_alcohol_level_undeclared_raises() -> None:
    with pytest.raises(DatasetValidationError):
        alcohol_level(pd.Series(["moderate"]), ALCOHOL_LEVELS)
    with pytest.raises(DatasetValidationError):  # the literal 'missing' is not a data level
        alcohol_level(pd.Series(["missing"]), ALCOHOL_LEVELS)
    with pytest.raises(ConfigError):
        alcohol_level(pd.Series(["zero"]), (*ALCOHOL_LEVELS, "missing"))


# ---------------------------------------------------------------------- polypharmacy (RT-12)
def test_log_polypharmacy_rt12() -> None:
    np.testing.assert_allclose(log_polypharmacy(np.array([0, 9, 61])), [-2.302585, 0.0, 1.824549], atol=1e-6)


def test_log_polypharmacy_series_keeps_index() -> None:
    out = log_polypharmacy(pd.Series([0, 9], index=[10, 20], dtype="int8"))
    assert isinstance(out, pd.Series) and list(out.index) == [10, 20] and out.dtype == "float64"


@pytest.mark.parametrize("bad", [[1, -1], [1.0, np.nan]])
def test_log_polypharmacy_rejects_negative_and_null(bad: list[float]) -> None:
    with pytest.raises(DatasetValidationError):
        log_polypharmacy(np.array(bad, dtype=float))


# ---------------------------------------------------------------------- FP pre-scaling (RT-13)
@pytest.mark.parametrize(
    ("x", "expected"),
    [
        (np.arange(0, 62), (1.0, 10.0)),
        (np.arange(21, 81), (0.0, 10.0)),
        (np.arange(0, 2381), (1.0, 1000.0)),
        (np.arange(0, 61, 5), (5.0, 10.0)),
    ],
)
def test_fp_auto_scaling_rt13(x: np.ndarray, expected: tuple[float, float]) -> None:
    assert fp_auto_scaling(x) == expected


def test_fp_auto_scaling_small_ranges_and_negative_values() -> None:
    assert fp_auto_scaling(np.array([0.01, 0.06, 0.2])) == (0.0, 1.0)  # p = log10(0.19) -> 10**(-0) = 1
    assert fp_auto_scaling(np.array([1.0, 1.02, 1.05])) == (0.0, 0.1)  # p = log10(0.05) -> 10**(-1)
    shift, scale = fp_auto_scaling(np.array([-3.0, -1.0, 0.5, 400.0]))
    assert shift == pytest.approx(3.0 + 1.5) and scale == 100.0


def test_fp_auto_scaling_rejects_constant_and_null() -> None:
    with pytest.raises(DatasetValidationError):
        fp_auto_scaling(np.array([4.0, 4.0]))
    with pytest.raises(DatasetValidationError):
        fp_auto_scaling(np.array([1.0, np.nan]))


# ---------------------------------------------------------------------- FP basis
def test_fp_basis_powers_and_repeated_power_recursion() -> None:
    z = np.array([0.5, 1.0, 2.0, 6.2])
    ln = np.log(z)
    np.testing.assert_allclose(fp_basis(z, (1,)), z[:, None])
    np.testing.assert_allclose(fp_basis(z, (0,)), ln[:, None])
    np.testing.assert_allclose(fp_basis(z, (0, 0)), np.column_stack([ln, ln * ln]))
    np.testing.assert_allclose(fp_basis(z, (-2, 1)), np.column_stack([z**-2, z]))
    np.testing.assert_allclose(fp_basis(z, (0.5, 0.5)), np.column_stack([np.sqrt(z), np.sqrt(z) * ln]))
    np.testing.assert_allclose(fp_basis(z, (3, 3, 3)), np.column_stack([z**3, z**3 * ln, z**3 * ln**2]))
    assert fp_basis(z, (-1, 2)).shape == (4, 2)


def test_fp_basis_domain_and_power_order() -> None:
    np.testing.assert_allclose(fp_basis(np.array([-1.0, 0.0, 2.0]), (1,)), [[-1.0], [0.0], [2.0]])
    with pytest.raises(DatasetValidationError):
        fp_basis(np.array([0.0, 1.0]), (0,))
    with pytest.raises(DatasetValidationError):
        fp_basis(np.array([1.0, np.nan]), (2,))
    with pytest.raises(ConfigError):
        fp_basis(np.array([1.0, 2.0]), (2, 1))


def test_fp_column_names() -> None:
    assert fp_column_names("x", (1,)) == ["x__fp_p1"]
    assert fp_column_names("x", (0, 0)) == ["x__fp_p0", "x__fp_p0_ln"]
    assert fp_column_names("x", (-2, 1)) == ["x__fp_p-2", "x__fp_p1"]
    assert fp_column_names("x", (-0.5, 3.0)) == ["x__fp_p-0.5", "x__fp_p3"]
    assert fp_column_names("x", (-0.0,)) == ["x__fp_p0"]
