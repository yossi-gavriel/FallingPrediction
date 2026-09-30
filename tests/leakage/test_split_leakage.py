"""Leakage tests for split plans and tuning (spec D-19 §2 embargo, §4 locked test; D-15 patients; architecture §1.6).

- no row, and (when required) no patient, belongs to two partitions;
- every training outcome window ends before the first validation index date, and every validation window
  before the first test index date;
- hyperparameter search fits only on training rows and scores only on validation rows (sentinel spies).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from falls_ml.config import GroupHoldoutSplit, RandomSplit, TemporalSplit, ValidationSection
from falls_ml.errors import LeakageError
from falls_ml.features.spec import FeatureSpec, load_feature_spec
from falls_ml.seeding import rng_for
from falls_ml.splitting import SPLITS, TestSetGuard, make_split
from falls_ml.tuning import run_search

REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "configs" / "features" / "efalls_v1.yaml"


@pytest.fixture(scope="module")
def spec() -> FeatureSpec:
    return load_feature_spec(SPEC_PATH)


@pytest.fixture(scope="module")
def cohort() -> pd.DataFrame:
    """Patients with 1-3 annual index dates starting on arbitrary days in 2016-2020 (many windows cross boundaries)."""
    rng = rng_for(20260915, "split_leakage_cohort")
    rows = []
    for i in range(800):
        start = pd.Timestamp("2016-01-01") + pd.Timedelta(days=int(rng.integers(0, 4 * 365 + 180)))
        for k in range(int(rng.integers(1, 4))):
            rows.append({"research_id": f"p{i:04d}", "index_date": start + pd.DateOffset(years=k),
                         "outcome_12m": int(rng.random() < 0.15), "practice_id": f"GP{i % 12:02d}"})
    df = pd.DataFrame(rows)
    return df.astype({"research_id": "str", "practice_id": "str", "outcome_12m": "int8"})


VALIDATIONS = {
    "temporal_disjoint": ValidationSection(strategy="temporal", seed=1, patients_disjoint=True,
                                           temporal=TemporalSplit("2018-12-31", "2020-12-31", True)),
    "temporal_overlapping_patients": ValidationSection(strategy="temporal", seed=1, patients_disjoint=False,
                                                       temporal=TemporalSplit("2018-12-31", "2020-12-31", True)),
    "group_holdout": ValidationSection(strategy="group_holdout", seed=1, patients_disjoint=True,
                                       group_holdout=GroupHoldoutSplit("practice_id", ("GP03", "GP04"), ("GP10", "GP11"))),
    "patient_grouped_random": ValidationSection(strategy="patient_grouped_random", seed=1, limitation_note="test fixture",
                                                patient_grouped_random=RandomSplit(0.2, 0.2, True)),
}


@pytest.mark.parametrize("name", sorted(VALIDATIONS))
def test_no_row_or_patient_in_two_partitions(cohort, spec, name):
    validation = VALIDATIONS[name]
    plan = make_split(cohort, spec, validation)
    positions = np.concatenate([plan.indices()[s] for s in SPLITS])
    assert len(np.unique(positions)) == len(positions)
    row_keys = {s: set(zip(cohort["research_id"].iloc[idx], cohort["index_date"].iloc[idx])) for s, idx in plan.indices().items()}
    patients = {s: set(cohort["research_id"].iloc[idx]) for s, idx in plan.indices().items()}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        assert not row_keys[a] & row_keys[b]
        if validation.patients_disjoint:
            assert not patients[a] & patients[b], f"{name}: patients shared by {a} and {b}"


@pytest.mark.parametrize("name", ["temporal_disjoint", "temporal_overlapping_patients"])
def test_outcome_windows_end_before_next_partition_index(cohort, spec, name):
    plan = make_split(cohort, spec, VALIDATIONS[name])
    assert plan.metadata["embargo_removed"]["train"] > 0 and plan.metadata["embargo_removed"]["validation"] > 0
    dates = {s: cohort["index_date"].iloc[idx] for s, idx in plan.indices().items()}
    assert spec.outcome.window_end(dates["train"]).max() < dates["validation"].min()
    assert spec.outcome.window_end(dates["validation"]).max() < dates["test"].min()
    assert dates["train"].max() < dates["validation"].min() and dates["validation"].max() < dates["test"].min()


def test_every_excluded_row_is_counted(cohort, spec):
    plan = make_split(cohort, spec, VALIDATIONS["temporal_disjoint"])
    removed = sum(plan.metadata["embargo_removed"].values()) + sum(plan.metadata["patient_overlap_removed"].values())
    assert sum(len(idx) for idx in plan.indices().values()) + removed == len(cohort)


def test_patient_spanning_holdout_groups_is_refused(cohort, spec):
    n_rows = cohort["research_id"].value_counts()
    patient = next(pid for pid in sorted(cohort.loc[cohort["practice_id"] == "GP00", "research_id"].unique()) if n_rows[pid] > 1)
    moved = cohort.copy()
    moved.loc[moved.index[moved["research_id"] == patient][0], "practice_id"] = "GP10"  # one row moves into a test group
    with pytest.raises(LeakageError):
        make_split(moved, spec, VALIDATIONS["group_holdout"])


def test_test_rows_leave_the_guard_only_once(cohort, spec):
    plan = make_split(cohort, spec, VALIDATIONS["temporal_disjoint"])
    guard = TestSetGuard(cohort.iloc[plan.test_idx].reset_index(drop=True))
    assert guard.n_rows == len(plan.test_idx)
    guard.release("final evaluation")
    with pytest.raises(LeakageError):
        guard.release("peek again")


# ---------------------------------------------------------------------- tuning spies
class SpyPipeline:
    def __init__(self, calls: dict[str, list[set[str]]]):
        self.calls = calls

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        self.calls["predict"].append(set(df["research_id"]))
        return expit(-1.5 + 0.02 * (df["age_years"].to_numpy() - 80.0))

    def linear_predictor(self, df: pd.DataFrame) -> np.ndarray:
        self.calls["linear_predictor"].append(set(df["research_id"]))
        return -1.5 + 0.02 * (df["age_years"].to_numpy() - 80.0)


def tuning_partitions(cohort: pd.DataFrame, spec: FeatureSpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    plan = make_split(cohort, spec, VALIDATIONS["temporal_disjoint"])
    rng = rng_for(3, "tuning_spy_age")
    frame = cohort.assign(age_years=rng.uniform(65, 95, len(cohort)))
    return frame.iloc[plan.train_idx].reset_index(drop=True), frame.iloc[plan.validation_idx].reset_index(drop=True)


def test_tuning_fits_on_train_rows_only(cohort, spec):
    train_df, val_df = tuning_partitions(cohort, spec)
    val_sentinel = set(val_df["research_id"])
    calls: dict[str, list[set[str]]] = {"fit": [], "predict": [], "linear_predictor": []}

    def fit_fn(df: pd.DataFrame, params: dict[str, Any]) -> SpyPipeline:
        calls["fit"].append(set(df["research_id"]))
        assert len(df) == len(train_df)
        pd.testing.assert_frame_equal(df, train_df)
        return SpyPipeline(calls)

    run_search(train_df, val_df, fit_fn=fit_fn, space={"C": [0.1, 1.0, 10.0]}, method="grid", n_iter=None, seed=0,
               weights={"auroc": 1.0, "brier": -1.0}, spec=spec)
    assert len(calls["fit"]) == 3
    assert all(ids == set(train_df["research_id"]) and not ids & val_sentinel for ids in calls["fit"])
    assert calls["predict"] and all(ids == val_sentinel for ids in calls["predict"] + calls["linear_predictor"])


class MutatingSpyPipeline(SpyPipeline):
    """A careless pipeline that shifts a predictor in the frame it is given before predicting."""

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        df["age_years"] = df["age_years"] + 10.0
        return super().predict_proba(df)


def test_tuning_trials_cannot_carry_state_through_the_frames(cohort, spec):
    train_df, val_df = tuning_partitions(cohort, spec)
    train_before, val_before = train_df.copy(deep=True), val_df.copy(deep=True)
    columns_seen: list[list[str]] = []

    def fit_fn(df: pd.DataFrame, params: dict[str, Any]) -> SpyPipeline:
        columns_seen.append(list(df.columns))
        df["leak"] = 1.0  # writes into its input
        return MutatingSpyPipeline({"predict": [], "linear_predictor": []})

    _, results = run_search(train_df, val_df, fit_fn=fit_fn, space={"C": [0.1, 1.0, 10.0]}, method="grid", n_iter=None,
                            seed=0, weights={"auroc": 1.0, "brier": -1.0}, spec=spec)
    assert all(columns == list(train_before.columns) for columns in columns_seen)
    pd.testing.assert_frame_equal(train_df, train_before)
    pd.testing.assert_frame_equal(val_df, val_before)
    assert results["objective"].nunique() == 1 and results["brier"].nunique() == 1  # every trial scored the same rows


def test_tuning_refuses_shared_rows(cohort, spec):
    train_df, val_df = tuning_partitions(cohort, spec)
    leaky_val = pd.concat([val_df, train_df.iloc[:1]], ignore_index=True)
    with pytest.raises(LeakageError):
        run_search(train_df, leaky_val, fit_fn=lambda df, params: SpyPipeline({"predict": [], "linear_predictor": []}),
                   space={"C": [1.0]}, method="grid", n_iter=None, seed=0, weights={"auroc": 1.0}, spec=spec)
