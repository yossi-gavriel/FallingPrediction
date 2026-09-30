"""RT-11: net benefit arithmetic on the published external-validation test-accuracy row (spec §7.1, §12.2).

eFalls supplement Table S3.4 (second table), threshold 0.10, per 1,000 patients: TP 19.2, FP 174.3, TN 796.5,
FN 10.1. Connected Bradford prevalence 2,389 / 81,685.
"""

from __future__ import annotations

import numpy as np
import pytest

from falls_ml.evaluation.metrics import decision_curve, net_benefit, net_benefit_from_counts, net_benefit_treat_all

RT11_TOL = 1e-5
THRESHOLD = 0.10
TP, FP, TN, FN = 19.2, 174.3, 796.5, 10.1
PREVALENCE_CB = 2389 / 81685


def test_rt11_model_net_benefit_from_published_counts() -> None:
    assert net_benefit_from_counts(TP, FP, 1000, THRESHOLD) == pytest.approx(-0.00017, abs=RT11_TOL)
    # the four cells sum to 1000.1 because of rounding; the result is unchanged at the stated tolerance
    assert net_benefit_from_counts(TP, FP, TP + FP + TN + FN, THRESHOLD) == pytest.approx(-0.00017, abs=RT11_TOL)


def test_rt11_treat_all_is_negative_so_model_is_close_to_treat_none() -> None:
    nb_all = net_benefit_treat_all(PREVALENCE_CB, THRESHOLD)
    assert nb_all < 0
    assert nb_all == pytest.approx(PREVALENCE_CB - (1 - PREVALENCE_CB) / 9, abs=1e-15)
    assert abs(net_benefit_from_counts(TP, FP, 1000, THRESHOLD)) < 1e-3  # ~ treat-none (NB 0)


def test_rt11_patient_level_arrays_reproduce_the_count_formula() -> None:
    # 10x the per-1,000 row as integer patients (10,001 rows); positives have risk 0.5, negatives 0.05
    tp, fp, tn, fn = 192, 1743, 7965, 101
    y = np.r_[np.ones(tp), np.zeros(fp), np.zeros(tn), np.ones(fn)]
    p = np.r_[np.full(tp + fp, 0.5), np.full(tn + fn, 0.05)]
    n = y.size
    assert net_benefit(y, p, THRESHOLD) == pytest.approx(tp / n - fp / n * THRESHOLD / (1 - THRESHOLD), abs=1e-15)
    assert net_benefit(y, p, THRESHOLD) == pytest.approx(-0.00017, abs=RT11_TOL)
    row = decision_curve(y, p, [THRESHOLD]).iloc[0]
    assert row["net_benefit_model"] == pytest.approx(net_benefit(y, p, THRESHOLD))
    assert row["net_benefit_treat_all"] == pytest.approx(net_benefit_treat_all(y.mean(), THRESHOLD))
    assert row["standardized_net_benefit"] == pytest.approx(row["net_benefit_model"] / y.mean())
