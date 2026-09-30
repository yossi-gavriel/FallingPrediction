"""Unit tests for recalibration (spec §7.4, D-07; RT-06, RT-07)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from falls_ml.errors import ConfigError, NotFittedError
from falls_ml.evaluation import metrics as M
from falls_ml.evaluation.calibration import (
    InterceptOnlyRecalibrator,
    LogisticRecalibrator,
    NoRecalibration,
    RecalibrationError,
    Recalibrator,
    make_recalibrator,
    recalibrator_from_dict,
)

# RT-06: printed Box S3.2 value 0.106; spec 0.1059302 (exact value of the formula 0.1059305), tolerance 5e-4.
RT06_SPEC_P, RT06_TOL = 0.1059302, 5e-4
# RT-07: the spec prints 0.1012433, but expit(-0.423 + 1.25 * -1.408453) = 0.10123598 (40-digit arithmetic);
# the same SPEC ERRATUM is documented in tests/regression/test_published_equation.py. We assert the formula value.
RT07_EXPECTED_P, RT07_SPEC_PRINTED_P, RT07_TOL = 0.1012360, 0.1012433, 1e-6


def _miscalibrated(n: int = 5000, seed: int = 21) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    lp = rng.normal(-2.5, 0.9, n)
    y = (rng.random(n) < M.expit(-0.6 + 1.3 * lp)).astype(np.int64)
    return y, np.asarray(M.expit(lp)), lp


def test_logistic_recalibration_same_data_is_perfectly_calibrated() -> None:
    y, p, lp = _miscalibrated()
    rec = LogisticRecalibrator().fit(p, y, lp=lp)
    a, b, _, _ = M.calibration_slope_intercept(y, lp)
    assert rec.params() == pytest.approx({"alpha": a, "beta": b})
    p_new = rec.transform(p, lp=lp)
    lp_new = M.logit(p_new)
    assert M.calibration_slope_intercept(y, lp_new)[1] == pytest.approx(1.0, abs=1e-6)
    assert M.citl(y, lp_new)[0] == pytest.approx(0.0, abs=1e-6)
    assert M.oe_ratio(y, p_new) == pytest.approx(1.0, abs=1e-6)


def test_intercept_only_recalibration_zeroes_citl_and_keeps_slope() -> None:
    y, p, lp = _miscalibrated()
    rec = InterceptOnlyRecalibrator().fit(p, y, lp=lp)
    assert rec.params()["delta"] == pytest.approx(M.citl(y, lp)[0])
    lp_new = M.logit(rec.transform(p, lp=lp))
    assert M.citl(y, lp_new)[0] == pytest.approx(0.0, abs=1e-6)
    assert M.calibration_slope_intercept(y, lp_new)[1] == pytest.approx(M.calibration_slope_intercept(y, lp)[1], abs=1e-6)
    np.testing.assert_allclose(lp_new, lp + rec.params()["delta"], atol=1e-9)


def test_lp_defaults_to_logit_p() -> None:
    y, p, lp = _miscalibrated(n=800)
    for cls in (LogisticRecalibrator, InterceptOnlyRecalibrator):
        with_lp = cls().fit(p, y, lp=lp)
        without = cls().fit(p, y)
        assert with_lp.params() == pytest.approx(without.params(), abs=1e-8)
        np.testing.assert_allclose(with_lp.transform(p, lp=lp), without.transform(p), atol=1e-10)


def test_rt06_rt07_published_bradford_recalibration() -> None:
    rec = LogisticRecalibrator(alpha=-0.423, beta=1.25)
    assert rec.is_fitted and rec.fitted_on_n is None
    lps = np.array([-1.368, -1.408453])
    out = rec.transform(np.asarray(M.expit(lps)), lp=lps)
    assert abs(out[0] - RT06_SPEC_P) <= RT06_TOL
    assert abs(out[0] - 0.106) <= RT06_TOL
    assert abs(out[1] - RT07_EXPECTED_P) <= RT07_TOL
    assert abs(out[1] - RT07_SPEC_PRINTED_P) > RT07_TOL  # documented spec erratum


def test_round_trip_to_dict_json() -> None:
    y, p, lp = _miscalibrated(n=1000)
    for rec in (LogisticRecalibrator().fit(p, y, lp=lp), InterceptOnlyRecalibrator().fit(p, y, lp=lp), NoRecalibration().fit(p, y)):
        state = json.loads(json.dumps(rec.to_dict()))
        assert state["method"] == rec.method and state["fitted_on_n"] == 1000
        clone = recalibrator_from_dict(state)
        assert type(clone) is type(rec) and clone.params() == rec.params() and clone.fitted_on_n == 1000
        np.testing.assert_array_equal(clone.transform(p, lp=lp), rec.transform(p, lp=lp))


def test_transform_before_fit_raises_not_fitted() -> None:
    for rec in (LogisticRecalibrator(), InterceptOnlyRecalibrator()):
        assert not rec.is_fitted
        with pytest.raises(NotFittedError):
            rec.transform(np.array([0.1, 0.2]))
        with pytest.raises(NotFittedError):
            rec.to_dict()


def test_no_recalibration_is_identity_and_does_not_alias_input() -> None:
    p = np.array([0.0, 0.2, 1.0])
    out = NoRecalibration().transform(p)
    np.testing.assert_array_equal(out, p)
    out[0] = 0.5
    assert p[0] == 0.0
    assert NoRecalibration().to_dict() == {"method": "none", "params": {}, "fitted_on_n": None}


def test_factory_and_invalid_states() -> None:
    assert {m: type(make_recalibrator(m)) for m in ("none", "intercept_only", "logistic_intercept_slope")} == {
        "none": NoRecalibration, "intercept_only": InterceptOnlyRecalibrator, "logistic_intercept_slope": LogisticRecalibrator}
    assert all(isinstance(make_recalibrator(m), Recalibrator) for m in ("none", "intercept_only"))
    with pytest.raises(ConfigError):
        make_recalibrator("isotonic")
    with pytest.raises(ConfigError):
        recalibrator_from_dict({"method": "logistic_intercept_slope", "params": {"alpha": 0.1}})
    with pytest.raises(ConfigError):
        recalibrator_from_dict({"method": "intercept_only", "params": {"delta": math.nan}})
    with pytest.raises(ValueError):
        LogisticRecalibrator(alpha=0.1)
    with pytest.raises(ValueError):
        LogisticRecalibrator().fit(np.array([0.1, 0.2]), np.array([0, 1, 1]))


def test_unestimable_recalibration_raises() -> None:
    p = np.array([0.1, 0.2, 0.3, 0.4])
    with pytest.warns(M.UndefinedMetricWarning), pytest.raises(RecalibrationError):
        InterceptOnlyRecalibrator().fit(p, np.zeros(4))
    with pytest.warns(M.UndefinedMetricWarning), pytest.raises(RecalibrationError):
        LogisticRecalibrator().fit(np.full(4, 0.3), np.array([0, 1, 0, 1]))  # constant LP
