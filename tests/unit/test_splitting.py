"""Unit tests for split plans, the locked-test guard and grouped K-fold (spec D-19, D-15, D-00, D-11)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from falls_ml.config import GroupHoldoutSplit, RandomSplit, TemporalSplit, ValidationSection
from falls_ml.errors import ConfigError, DatasetValidationError, LeakageError
from falls_ml.features.spec import FeatureSpec, load_feature_spec
from falls_ml.seeding import rng_for
from falls_ml.splitting import SplitPlan, TestSetGuard, grouped_kfold_indices, make_split

REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "configs" / "features" / "efalls_v1.yaml"


@pytest.fixture(scope="module")
def spec() -> FeatureSpec:
    return load_feature_spec(SPEC_PATH)


def cohort(rows: list[tuple[str, str, int]], **extra: list[object]) -> pd.DataFrame:
    """(research_id, index_date, outcome) rows plus optional extra columns."""
    ids, dates, outcomes = zip(*rows)
    df = pd.DataFrame({"research_id": pd.Series(ids, dtype="str"), "index_date": pd.to_datetime(list(dates)),
                       "outcome_12m": np.asarray(outcomes, dtype="int8")})
    for name, values in extra.items():
        df[name] = pd.Series(values, dtype="str")
    return df


def temporal(train_end: str = "2018-12-31", validation_end: str = "2019-12-31", *, embargo: bool = True,
             disjoint: bool = False) -> ValidationSection:
    return ValidationSection(strategy="temporal", seed=42, patients_disjoint=disjoint,
                             temporal=TemporalSplit(train_end=train_end, validation_end=validation_end, embargo_outcome_windows=embargo))


def keys(frame: pd.DataFrame, idx: np.ndarray) -> set[tuple[str, str]]:
    part = frame.iloc[idx]
    return set(zip(part["research_id"], part["index_date"].dt.strftime("%Y-%m-%d")))


def annual_cohorts(n_per_year: int = 30) -> pd.DataFrame:
    rows = [(f"p{year}_{i}", f"{year}-04-01", int(i % 5 == 0)) for year in (2018, 2019, 2020) for i in range(n_per_year)]
    return cohort(rows)


# ---------------------------------------------------------------------- temporal
class TestTemporal:
    def test_date_boundaries(self, spec):
        frame = cohort([("a", "2018-12-31", 0), ("b", "2019-01-01", 1), ("c", "2019-12-31", 0), ("d", "2020-01-01", 1)])
        plan = make_split(frame, spec, temporal(embargo=False))
        assert keys(frame, plan.train_idx) == {("a", "2018-12-31")}
        assert keys(frame, plan.validation_idx) == {("b", "2019-01-01"), ("c", "2019-12-31")}
        assert keys(frame, plan.test_idx) == {("d", "2020-01-01")}
        assert plan.metadata == {"embargo_removed": {}, "patient_overlap_removed": {}}

    def test_annual_index_dates_remove_nothing(self, spec):
        # window_end(2018-04-01) = 2019-03-31 < 2019-04-01, so annual cohorts satisfy the embargo exactly (D-00, D-19 §2)
        assert spec.outcome.window_end(pd.Timestamp("2018-04-01")) == pd.Timestamp("2019-03-31")
        frame = annual_cohorts()
        plan = make_split(frame, spec, temporal())
        assert plan.metadata["embargo_removed"] == {"train": 0, "validation": 0}
        assert [len(plan.train_idx), len(plan.validation_idx), len(plan.test_idx)] == [30, 30, 30]

    def test_overlapping_windows_are_removed_and_counted(self, spec):
        frame = pd.concat([annual_cohorts(), cohort([("late_train", "2018-06-01", 1), ("late_val", "2019-06-01", 0)])],
                          ignore_index=True)
        plan = make_split(frame, spec, temporal())
        assert plan.metadata["embargo_removed"] == {"train": 1, "validation": 1}
        assigned = set(plan.assignments(frame, spec)["research_id"])
        assert {"late_train", "late_val"}.isdisjoint(assigned)
        assert len(assigned) == 90

    def test_window_ending_on_next_index_date_is_removed(self, spec):
        # 2018-04-02 -> window end 2019-04-01 == first validation index date: removed (strict <); 2018-04-01 rows kept
        frame = pd.concat([annual_cohorts(3), cohort([("edge", "2018-04-02", 0)])], ignore_index=True)
        plan = make_split(frame, spec, temporal())
        assert plan.metadata["embargo_removed"]["train"] == 1
        assert ("edge", "2018-04-02") not in keys(frame, plan.train_idx)
        assert len(plan.train_idx) == 3

    def test_embargo_off_keeps_overlapping_rows(self, spec):
        frame = pd.concat([annual_cohorts(), cohort([("late_train", "2018-06-01", 1)])], ignore_index=True)
        plan = make_split(frame, spec, temporal(embargo=False))
        assert ("late_train", "2018-06-01") in keys(frame, plan.train_idx)

    def test_patients_disjoint_removes_rows_from_earlier_partitions(self, spec):
        extra = cohort([("a", "2018-04-01", 0), ("a", "2020-04-01", 0),      # train + test
                        ("b", "2018-04-01", 1), ("b", "2019-04-01", 0),      # train + validation
                        ("c", "2019-04-01", 0), ("c", "2020-04-01", 1)])     # validation + test
        frame = pd.concat([annual_cohorts(), extra], ignore_index=True)
        plan = make_split(frame, spec, temporal(disjoint=True))
        assert plan.metadata["patient_overlap_removed"] == {"train": 2, "validation": 1}
        assert ("a", "2020-04-01") in keys(frame, plan.test_idx) and ("c", "2020-04-01") in keys(frame, plan.test_idx)
        assert ("b", "2019-04-01") in keys(frame, plan.validation_idx)
        assert not {("a", "2018-04-01"), ("b", "2018-04-01")} & keys(frame, plan.train_idx)
        assert ("c", "2019-04-01") not in keys(frame, plan.validation_idx)
        patients = {name: set(frame["research_id"].iloc[idx]) for name, idx in plan.indices().items()}
        assert not (patients["train"] & patients["validation"] or patients["train"] & patients["test"]
                    or patients["validation"] & patients["test"])

    def test_patient_overlap_allowed_when_not_disjoint(self, spec):
        frame = pd.concat([annual_cohorts(), cohort([("a", "2018-04-01", 0), ("a", "2020-04-01", 0)])], ignore_index=True)
        plan = make_split(frame, spec, temporal(disjoint=False))
        assert plan.metadata["patient_overlap_removed"] == {}
        assert ("a", "2018-04-01") in keys(frame, plan.train_idx)

    def test_empty_partition_raises(self, spec):
        with pytest.raises(ConfigError, match="empty"):
            make_split(annual_cohorts(), spec, temporal(validation_end="2021-12-31"))

    def test_partition_emptied_by_embargo_raises(self, spec):
        frame = cohort([("a", "2018-06-01", 0), ("b", "2019-04-01", 1), ("c", "2020-04-01", 0)])
        with pytest.raises(ConfigError, match="embargo"):
            make_split(frame, spec, temporal())

    def test_train_end_must_precede_validation_end(self, spec):
        with pytest.raises(ConfigError, match="precede"):
            make_split(annual_cohorts(), spec, temporal(train_end="2019-12-31", validation_end="2018-12-31"))

    def test_missing_temporal_section_raises(self, spec):
        with pytest.raises(ConfigError):
            make_split(annual_cohorts(), spec, ValidationSection(strategy="temporal", seed=1))

    def test_index_date_must_be_datetime(self, spec):
        frame = annual_cohorts().assign(index_date=lambda d: d["index_date"].dt.strftime("%Y-%m-%d"))
        with pytest.raises(DatasetValidationError, match="datetime64"):
            make_split(frame, spec, temporal())

    def test_frame_is_not_mutated(self, spec):
        frame = pd.concat([annual_cohorts(), cohort([("late", "2018-06-01", 1)])], ignore_index=True)
        before = frame.copy(deep=True)
        make_split(frame, spec, temporal(disjoint=True))
        pd.testing.assert_frame_equal(frame, before)


# ---------------------------------------------------------------------- group hold-out
def practice_frame() -> pd.DataFrame:
    rows, practices = [], []
    for k, practice in enumerate("ABCDE"):
        for i in range(10):
            rows.append((f"{practice}{i}", "2018-04-01", int((i + k) % 4 == 0)))
            practices.append(practice)
    return cohort(rows, practice_id=practices)


def holdout(validation_groups=("C",), test_groups=("D", "E"), *, disjoint: bool = True) -> ValidationSection:
    return ValidationSection(strategy="group_holdout", seed=0, patients_disjoint=disjoint,
                             group_holdout=GroupHoldoutSplit("practice_id", tuple(validation_groups), tuple(test_groups)))


class TestGroupHoldout:
    def test_groups_define_partitions(self, spec):
        frame = practice_frame()
        plan = make_split(frame, spec, holdout())
        part = {name: set(frame["practice_id"].iloc[idx]) for name, idx in plan.indices().items()}
        assert part == {"train": {"A", "B"}, "validation": {"C"}, "test": {"D", "E"}}
        assert len(plan.train_idx) + len(plan.validation_idx) + len(plan.test_idx) == len(frame)

    def test_unknown_group_raises(self, spec):
        with pytest.raises(ConfigError, match="not present"):
            make_split(practice_frame(), spec, holdout(test_groups=("Z",)))

    def test_group_in_both_partitions_raises(self, spec):
        with pytest.raises(ConfigError, match="both"):
            make_split(practice_frame(), spec, holdout(validation_groups=("C",), test_groups=("C",)))

    def test_patient_spanning_groups_raises_when_disjoint(self, spec):
        frame = pd.concat([practice_frame(), cohort([("A0", "2019-04-01", 0)], practice_id=["D"])], ignore_index=True)
        with pytest.raises(LeakageError, match="patients"):
            make_split(frame, spec, holdout())
        plan = make_split(frame, spec, holdout(disjoint=False))
        assert "A0" in set(frame["research_id"].iloc[plan.test_idx])

    def test_null_group_raises(self, spec):
        frame = practice_frame()
        frame["practice_id"] = frame["practice_id"].where(frame.index != 3, None)
        with pytest.raises(DatasetValidationError, match="null"):
            make_split(frame, spec, holdout())

    def test_missing_group_column_raises(self, spec):
        with pytest.raises(ConfigError, match="group_column"):
            make_split(practice_frame().drop(columns="practice_id"), spec, holdout())


# ---------------------------------------------------------------------- patient-grouped random
def repeated_patients(n_patients: int = 400) -> pd.DataFrame:
    rng = rng_for(7, "test_splitting_random")
    rows = []
    for i in range(n_patients):
        for year in range(2018, 2018 + int(rng.integers(1, 4))):
            rows.append((f"p{i:04d}", f"{year}-04-01", int(rng.random() < 0.2)))
    return cohort(rows)


def random_split(seed: int = 42, note: str = "single index date available (D-19 §10)", **kwargs: float) -> ValidationSection:
    return ValidationSection(strategy="patient_grouped_random", seed=seed, limitation_note=note,
                             patient_grouped_random=RandomSplit(**kwargs))


class TestPatientGroupedRandom:
    def test_deterministic_for_seed(self, spec):
        frame = repeated_patients()
        a, b = make_split(frame, spec, random_split(seed=3)), make_split(frame, spec, random_split(seed=3))
        c = make_split(frame, spec, random_split(seed=4))
        for name in ("train", "validation", "test"):
            np.testing.assert_array_equal(a.indices()[name], b.indices()[name])
        assert not np.array_equal(a.test_idx, c.test_idx)

    def test_independent_of_row_order(self, spec):
        frame = repeated_patients()
        shuffled = frame.iloc[rng_for(1, "shuffle").permutation(len(frame))].reset_index(drop=True)
        a, b = make_split(frame, spec, random_split()), make_split(shuffled, spec, random_split())
        for name in ("train", "validation", "test"):
            assert keys(frame, a.indices()[name]) == keys(shuffled, b.indices()[name])

    def test_patients_disjoint_fractions_and_stratification(self, spec):
        frame = repeated_patients(1000)
        plan = make_split(frame, spec, random_split(validation_fraction=0.2, test_fraction=0.3))
        patient_event = frame.groupby("research_id")["outcome_12m"].max()
        patients = {name: set(frame["research_id"].iloc[idx]) for name, idx in plan.indices().items()}
        assert not (patients["train"] & patients["test"] or patients["validation"] & patients["test"]
                    or patients["train"] & patients["validation"])
        assert len(patients["test"]) == pytest.approx(300, abs=2)
        assert len(patients["validation"]) == pytest.approx(200, abs=2)
        rates = {name: patient_event.loc[sorted(p)].mean() for name, p in patients.items()}
        assert max(rates.values()) - min(rates.values()) < 0.01
        assert plan.limitation_note and plan.metadata == {"embargo_removed": {}, "patient_overlap_removed": {}}

    def test_limitation_note_required(self, spec):
        with pytest.raises(ConfigError, match="limitation_note"):
            make_split(repeated_patients(), spec, random_split(note="  "))

    @pytest.mark.parametrize("fractions", [{"validation_fraction": 0.0}, {"test_fraction": 1.0},
                                           {"validation_fraction": 0.6, "test_fraction": 0.4}])
    def test_invalid_fractions_raise(self, spec, fractions):
        with pytest.raises(ConfigError, match="fractions"):
            make_split(repeated_patients(), spec, random_split(**fractions))

    def test_too_few_patients_gives_empty_partition(self, spec):
        with pytest.raises(ConfigError, match="empty"):
            make_split(cohort([("a", "2018-04-01", 0), ("b", "2018-04-01", 1)]), spec, random_split(stratify_outcome=False))


# ---------------------------------------------------------------------- plan outputs
class TestPlanOutputs:
    def test_assignments_summary_and_test_hash(self, spec):
        frame = annual_cohorts(4)
        plan = make_split(frame, spec, temporal())
        assignments = plan.assignments(frame, spec)
        assert list(assignments.columns) == ["research_id", "index_date", "split"]
        assert assignments["split"].tolist() == ["train"] * 4 + ["validation"] * 4 + ["test"] * 4
        summary = plan.summary(frame, spec)
        assert set(summary) == {"train", "validation", "test"}
        assert summary["test"] == {"n_rows": 4, "n_patients": 4, "n_events": 1, "prevalence": 0.25,
                                   "index_date_min": "2020-04-01", "index_date_max": "2020-04-01"}
        expected = hashlib.sha256("\n".join(sorted(f"p2020_{i}|2020-04-01" for i in range(4))).encode()).hexdigest()
        assert plan.test_rows_sha256(frame, spec) == expected
        reversed_frame = frame.iloc[::-1].reset_index(drop=True)
        assert make_split(reversed_frame, spec, temporal()).test_rows_sha256(reversed_frame, spec) == expected

    def test_indices_are_read_only_and_checked_against_frame(self, spec):
        frame = annual_cohorts(4)
        plan = make_split(frame, spec, temporal())
        with pytest.raises(ValueError):
            plan.train_idx[0] = 5
        with pytest.raises(ConfigError, match="indices"):
            plan.summary(frame.iloc[:5], spec)

    def test_manual_plan_with_empty_partition_summarises_nulls(self, spec):
        frame = annual_cohorts(2)
        plan = SplitPlan("manual", [0, 1], [2, 3], [], "manual", "", {})
        assert plan.summary(frame, spec)["test"]["prevalence"] is None

    @pytest.mark.parametrize("bad", [np.array([True, False, True]), np.array([0.0, 2.0]), np.array([[0, 1]]), [1, 1]],
                             ids=["boolean_mask", "float", "2d", "duplicates"])
    def test_plan_refuses_masks_and_duplicate_positions(self, bad):
        # np.asarray(mask, dtype=int64) would silently turn a boolean mask into positions [0, 1, 1]
        with pytest.raises(ConfigError, match="train_idx"):
            SplitPlan("manual", bad, [5], [6], "manual")


@pytest.mark.parametrize("strategy", ["temporal", "group_holdout", "patient_grouped_random"])
def test_make_split_never_mutates_frame(spec, strategy):
    frame, validation = {
        "temporal": (pd.concat([annual_cohorts(), cohort([("late", "2018-06-01", 1)])], ignore_index=True), temporal(disjoint=True)),
        "group_holdout": (practice_frame(), holdout()),
        "patient_grouped_random": (repeated_patients(), random_split()),
    }[strategy]
    frame.index = pd.RangeIndex(100, 100 + len(frame))  # non-default labels: indices must stay positional
    before = frame.copy(deep=True)
    plan = make_split(frame, spec, validation)
    pd.testing.assert_frame_equal(frame, before)
    assert sum(len(idx) for idx in plan.indices().values()) <= len(frame)
    assert max(int(idx.max()) for idx in plan.indices().values()) < len(frame)


# ---------------------------------------------------------------------- guard
class TestTestSetGuard:
    def test_release_once(self):
        frame = annual_cohorts(3)
        guard = TestSetGuard(frame)
        assert guard.n_rows == 9 and not guard.released
        released = guard.release("model selection frozen")
        pd.testing.assert_frame_equal(released, frame)
        assert guard.released and guard.n_rows == 9
        with pytest.raises(LeakageError, match="already released"):
            guard.release("again")

    def test_reason_required(self):
        guard = TestSetGuard(annual_cohorts(1))
        with pytest.raises(ConfigError):
            guard.release("")
        assert not guard.released

    def test_repr_does_not_expose_rows(self):
        assert repr(TestSetGuard(annual_cohorts(1))) == "TestSetGuard(n_rows=3, released=False)"


# ---------------------------------------------------------------------- grouped K-fold
class TestGroupedKFold:
    def groups(self) -> np.ndarray:
        rng = rng_for(3, "kfold_groups")
        return np.repeat([f"g{i}" for i in range(200)], rng.integers(1, 6, 200))

    def test_partition_properties(self):
        groups = self.groups()
        folds = grouped_kfold_indices(groups, 5, seed=11)
        assert len(folds) == 5
        test_all = np.concatenate([test for _, test in folds])
        np.testing.assert_array_equal(np.sort(test_all), np.arange(len(groups)))
        for train, test in folds:
            assert set(groups[train]).isdisjoint(groups[test])
            assert len(train) + len(test) == len(groups)
        sizes = [len(test) for _, test in folds]
        assert max(sizes) - min(sizes) <= 5  # at most the largest group size

    def test_deterministic_and_row_order_invariant(self):
        groups = self.groups()
        a = grouped_kfold_indices(groups, 4, seed=1)
        b = grouped_kfold_indices(groups, 4, seed=1)
        assert all(np.array_equal(x[1], y[1]) for x, y in zip(a, b))
        perm = rng_for(2, "perm").permutation(len(groups))
        c = grouped_kfold_indices(groups[perm], 4, seed=1)
        assert [set(groups[t]) for _, t in a] == [set(groups[perm][t]) for _, t in c]
        d = grouped_kfold_indices(groups, 4, seed=2)
        assert [set(groups[t]) for _, t in a] != [set(groups[t]) for _, t in d]

    def test_invalid_inputs(self):
        with pytest.raises(ConfigError):
            grouped_kfold_indices(["a", "b"], 3, seed=0)
        with pytest.raises(ConfigError):
            grouped_kfold_indices(["a", "b"], 1, seed=0)
        with pytest.raises(DatasetValidationError):
            grouped_kfold_indices(["a", None, "b"], 2, seed=0)
