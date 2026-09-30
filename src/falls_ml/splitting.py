"""Validation split plans and the locked-test guard (spec D-19 validation design, D-15 observation units, D-00).

Strategies (``validation.strategy``):

- ``temporal`` (preferred, D-19 §1-2): train ``index_date <= train_end``; validation
  ``train_end < index_date <= validation_end``; test ``index_date > validation_end``. With
  ``embargo_outcome_windows`` a training row is kept only if its outcome window ends before the earliest
  validation index date, and a validation row only if its window ends before the earliest test index date
  (``index + 1 year - 1 day < index(X)``, D-19 §2). Removed rows are counted in ``metadata['embargo_removed']``.
- ``group_holdout`` (D-19 §9): rows whose ``group_column`` value is listed in ``test_groups`` / ``validation_groups``
  form the test / validation partitions; all other rows are training rows.
- ``patient_grouped_random`` (D-19 §10 single-index-date fallback): patient-grouped random partition, stratified by
  each patient's maximum outcome; a limitation note is mandatory.

With ``patients_disjoint`` no patient may have rows in two partitions (D-15). The temporal split enforces it by
removing the patient's rows from the earlier partitions (counted in ``metadata['patient_overlap_removed']``); the
group hold-out raises ``LeakageError``; the random split is disjoint by construction.

Split functions never mutate the input frame, never drop rows silently and never return an empty partition.
Indices are positional (``frame.iloc``).
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from falls_ml.config import RandomSplit, ValidationSection
from falls_ml.errors import ConfigError, DatasetValidationError, LeakageError
from falls_ml.features.spec import FeatureSpec
from falls_ml.logging_utils import get_logger
from falls_ml.seeding import rng_for

log = get_logger(__name__)

SPLITS = ("train", "validation", "test")
SPLIT_COMPONENT = "split"
KFOLD_COMPONENT = "grouped_kfold"


@dataclass(frozen=True, eq=False)  # eq=False: a generated __eq__ over numpy arrays would be ambiguous
class SplitPlan:
    """Positional row indices of the train / validation / test partitions (read-only, sorted ascending)."""

    strategy: str
    train_idx: np.ndarray
    validation_idx: np.ndarray
    test_idx: np.ndarray
    description: str
    limitation_note: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("train_idx", "validation_idx", "test_idx"):
            raw = np.asarray(getattr(self, name))
            if raw.size == 0:
                raw = raw.reshape(0).astype(np.int64)
            # a boolean mask or float array would otherwise be coerced into wrong positions silently
            if raw.ndim != 1 or raw.dtype.kind not in "iu":
                raise ConfigError(f"SplitPlan.{name} must be a 1-D array of integer row positions, got dtype {raw.dtype}, "
                                  f"shape {raw.shape}")
            idx = np.sort(raw.astype(np.int64))
            if np.any(idx[1:] == idx[:-1]):
                raise ConfigError(f"SplitPlan.{name} contains duplicated row positions")
            idx.setflags(write=False)
            object.__setattr__(self, name, idx)

    def indices(self) -> dict[str, np.ndarray]:
        return {"train": self.train_idx, "validation": self.validation_idx, "test": self.test_idx}

    def _check_frame(self, frame: pd.DataFrame) -> None:
        n = len(frame)
        for name, idx in self.indices().items():
            if idx.size and (idx[0] < 0 or idx[-1] >= n):
                raise ConfigError(f"split plan: {name} indices do not fit a frame of {n} rows (plan built for another frame?)")

    def assignments(self, frame: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
        """``splits.csv`` rows (research_id, index_date, split) in frame order; excluded rows are absent."""
        self._check_frame(frame)
        id_col, idx_col = spec.identifier_columns[0], spec.index_column
        positions = np.concatenate([self.train_idx, self.validation_idx, self.test_idx])
        labels = np.repeat(np.array(SPLITS, dtype=object), [len(i) for i in self.indices().values()])
        order = np.argsort(positions, kind="stable")
        out = frame[[id_col, idx_col]].iloc[positions[order]].reset_index(drop=True)
        return out.assign(split=pd.Series(labels[order], dtype="str"))

    def summary(self, frame: pd.DataFrame, spec: FeatureSpec) -> dict[str, dict[str, Any]]:
        """``metrics.json`` ``cohort.by_split`` block (ARTIFACT_SCHEMAS)."""
        self._check_frame(frame)
        id_col, idx_col, y_col = spec.identifier_columns[0], spec.index_column, spec.outcome.name
        out: dict[str, dict[str, Any]] = {}
        for name, idx in self.indices().items():
            part = frame.iloc[idx]
            n = len(part)
            y = part[y_col].to_numpy(dtype=np.int64)
            dates = pd.to_datetime(part[idx_col])
            out[name] = {
                "n_rows": n, "n_patients": int(part[id_col].nunique()), "n_events": int(y.sum()),
                "prevalence": float(y.mean()) if n else None,
                "index_date_min": dates.min().strftime("%Y-%m-%d") if n else None,
                "index_date_max": dates.max().strftime("%Y-%m-%d") if n else None,
            }
        return out

    def test_rows_sha256(self, frame: pd.DataFrame, spec: FeatureSpec) -> str:
        """SHA-256 of the sorted ``research_id|YYYY-MM-DD`` keys of the test partition, newline-joined (D-19 §5)."""
        self._check_frame(frame)
        part = frame.iloc[self.test_idx]
        keys = part[spec.identifier_columns[0]].astype("str") + "|" + pd.to_datetime(part[spec.index_column]).dt.strftime("%Y-%m-%d")
        return hashlib.sha256("\n".join(sorted(keys.tolist())).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------- split construction
def make_split(frame: pd.DataFrame, spec: FeatureSpec, validation: ValidationSection) -> SplitPlan:
    """Build the split plan configured in ``validation`` (D-19). Raises ConfigError / LeakageError, never repairs."""
    ids, dates = _keys(frame, spec)
    if validation.strategy == "temporal":
        masks, description, metadata = _temporal(spec, ids, dates, validation)
    elif validation.strategy == "group_holdout":
        masks, description, metadata = _group_holdout(frame, validation)
    elif validation.strategy == "patient_grouped_random":
        masks, description, metadata = _patient_grouped_random(frame, spec, ids, validation)
    else:
        raise ConfigError(f"Unknown validation.strategy {validation.strategy!r}")
    plan = SplitPlan(strategy=validation.strategy, train_idx=np.flatnonzero(masks["train"]),
                     validation_idx=np.flatnonzero(masks["validation"]), test_idx=np.flatnonzero(masks["test"]),
                     description=description, limitation_note=validation.limitation_note, metadata=metadata)
    _assert_valid(plan, ids, patients_disjoint=validation.patients_disjoint)
    log.info("split_created", extra_fields={"strategy": plan.strategy, "seed": validation.seed,
                                            **{f"n_{k}": len(v) for k, v in plan.indices().items()}, **metadata})
    return plan


def _keys(frame: pd.DataFrame, spec: FeatureSpec) -> tuple[pd.Series, pd.Series]:
    """Patient ids (as strings) and index dates with a fresh RangeIndex; validates presence and nulls."""
    id_col, idx_col = spec.identifier_columns[0], spec.index_column
    missing = [c for c in (id_col, idx_col) if c not in frame.columns]
    if missing:
        raise DatasetValidationError("Split input is missing key columns", [str(missing)])
    if len(frame) == 0:
        raise DatasetValidationError("Split input has zero rows")
    if frame[id_col].isna().any():
        raise DatasetValidationError(f"{id_col}: {int(frame[id_col].isna().sum())} null identifiers")
    dates = frame[idx_col]
    if not pd.api.types.is_datetime64_any_dtype(dates):
        raise DatasetValidationError(f"{idx_col}: must be datetime64 dtype, got {dates.dtype}")
    if dates.isna().any():
        raise DatasetValidationError(f"{idx_col}: {int(dates.isna().sum())} null index dates")
    return frame[id_col].astype("str").reset_index(drop=True), dates.reset_index(drop=True)


def _parse_date(value: str, where: str) -> pd.Timestamp:
    try:
        return pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{where}: {value!r} is not a date") from exc


def _temporal(spec: FeatureSpec, ids: pd.Series, dates: pd.Series,
              validation: ValidationSection) -> tuple[dict[str, np.ndarray], str, dict[str, Any]]:
    cfg = validation.temporal
    if cfg is None:
        raise ConfigError("validation.strategy=temporal requires validation.temporal")
    train_end = _parse_date(cfg.train_end, "validation.temporal.train_end")
    validation_end = _parse_date(cfg.validation_end, "validation.temporal.validation_end")
    if not train_end < validation_end:
        raise ConfigError(f"validation.temporal: train_end {train_end.date()} must precede validation_end {validation_end.date()}")
    masks = {"train": (dates <= train_end).to_numpy(),
             "validation": ((dates > train_end) & (dates <= validation_end)).to_numpy(),
             "test": (dates > validation_end).to_numpy()}
    _require_non_empty(masks, "by the temporal date boundaries")

    embargo_removed: dict[str, int] = {}
    if cfg.embargo_outcome_windows:
        # D-19 §2: outcome windows of an earlier partition must end before the first index date of the next one.
        # Minimum dates are taken before any removal, which can only make the rule stricter.
        window_end = spec.outcome.window_end(dates)
        first_index = {name: dates[masks[name]].min() for name in ("validation", "test")}
        for earlier, later in (("train", "validation"), ("validation", "test")):
            violating = masks[earlier] & (window_end >= first_index[later]).to_numpy()
            embargo_removed[earlier] = int(violating.sum())
            masks[earlier] = masks[earlier] & ~violating
        if any(embargo_removed.values()):
            log.warning("split_embargo_rows_removed", extra_fields={"embargo_removed": embargo_removed})

    overlap_removed: dict[str, int] = {}
    if validation.patients_disjoint:
        overlap_removed = {"train": 0, "validation": 0}
        for later, earlier_parts in (("test", ("train", "validation")), ("validation", ("train",))):
            later_patients = ids[masks[later]].unique()
            for earlier in earlier_parts:
                clash = masks[earlier] & ids.isin(later_patients).to_numpy()
                overlap_removed[earlier] += int(clash.sum())
                masks[earlier] = masks[earlier] & ~clash
        if any(overlap_removed.values()):
            log.warning("split_patient_overlap_rows_removed", extra_fields={"patient_overlap_removed": overlap_removed})
    _require_non_empty(masks, f"after embargo (removed {embargo_removed}) and patient-overlap removal (removed {overlap_removed})")

    description = (f"temporal: train index_date <= {train_end.date()}; validation {train_end.date()} < index_date <= "
                   f"{validation_end.date()}; test index_date > {validation_end.date()}; "
                   f"outcome-window embargo {'on' if cfg.embargo_outcome_windows else 'off'}; "
                   f"patients disjoint {'yes' if validation.patients_disjoint else 'no'}")
    return masks, description, {"embargo_removed": embargo_removed, "patient_overlap_removed": overlap_removed}


def _group_holdout(frame: pd.DataFrame, validation: ValidationSection) -> tuple[dict[str, np.ndarray], str, dict[str, Any]]:
    cfg = validation.group_holdout
    if cfg is None:
        raise ConfigError("validation.strategy=group_holdout requires validation.group_holdout")
    col = cfg.group_column
    if col not in frame.columns:
        raise ConfigError(f"validation.group_holdout.group_column {col!r} is not a dataset column")
    if frame[col].isna().any():
        raise DatasetValidationError(f"{col}: {int(frame[col].isna().sum())} null group values; group hold-out needs a group per row")
    groups = frame[col].astype("str").reset_index(drop=True)
    val_groups, test_groups = set(cfg.validation_groups), set(cfg.test_groups)
    both = sorted(val_groups & test_groups)
    if both:
        raise ConfigError(f"validation.group_holdout: groups listed for both validation and test: {both}")
    unknown = sorted((val_groups | test_groups) - set(groups.unique()))
    if unknown:
        raise ConfigError(f"validation.group_holdout: groups not present in column {col!r}: {unknown}")
    masks = {"validation": groups.isin(val_groups).to_numpy(), "test": groups.isin(test_groups).to_numpy()}
    masks["train"] = ~(masks["validation"] | masks["test"])
    _require_non_empty(masks, f"by the {col!r} hold-out groups")
    description = (f"group_holdout on {col}: validation groups {sorted(val_groups)}; test groups {sorted(test_groups)}; "
                   "all other groups train")
    return masks, description, {"embargo_removed": {}, "patient_overlap_removed": {}}


def _patient_grouped_random(frame: pd.DataFrame, spec: FeatureSpec, ids: pd.Series,
                            validation: ValidationSection) -> tuple[dict[str, np.ndarray], str, dict[str, Any]]:
    cfg = validation.patient_grouped_random or RandomSplit()
    if not validation.limitation_note.strip():
        raise ConfigError("validation.strategy=patient_grouped_random requires validation.limitation_note (D-19 §10)")
    vf, tf = cfg.validation_fraction, cfg.test_fraction
    if not (0.0 < vf < 1.0 and 0.0 < tf < 1.0 and vf + tf < 1.0):
        raise ConfigError(f"patient_grouped_random: fractions must lie in (0, 1) and sum below 1, got validation {vf}, test {tf}")
    if cfg.stratify_outcome:
        y = _binary_outcome(frame, spec)
        patient_label = pd.Series(y).groupby(ids.to_numpy()).max()  # index sorted by patient id
        strata = [np.asarray(patient_label.index[patient_label.to_numpy() == level], dtype=object) for level in (0, 1)]
    else:
        strata = [np.array(sorted(ids.unique()), dtype=object)]
    rng = rng_for(validation.seed, SPLIT_COMPONENT)
    test_patients: list[Any] = []
    val_patients: list[Any] = []
    for members in strata:
        shuffled = members[rng.permutation(len(members))]
        n_test = int(np.floor(len(members) * tf + 0.5))
        n_val = int(np.floor(len(members) * vf + 0.5))
        test_patients.extend(shuffled[:n_test])
        val_patients.extend(shuffled[n_test:n_test + n_val])
    masks = {"test": ids.isin(test_patients).to_numpy(), "validation": ids.isin(val_patients).to_numpy()}
    masks["train"] = ~(masks["test"] | masks["validation"])
    _require_non_empty(masks, "by the patient-grouped random partition")
    description = (f"patient_grouped_random: patients shuffled with seed {validation.seed}"
                   f"{', stratified by patient maximum outcome' if cfg.stratify_outcome else ''}; "
                   f"validation fraction {vf}, test fraction {tf} of patients")
    return masks, description, {"embargo_removed": {}, "patient_overlap_removed": {}}


def _binary_outcome(frame: pd.DataFrame, spec: FeatureSpec) -> np.ndarray:
    y_col = spec.outcome.name
    if y_col not in frame.columns:
        raise DatasetValidationError(f"outcome column {y_col!r} is required for an outcome-stratified split")
    y = frame[y_col]
    if y.isna().any() or not set(pd.unique(y)) <= {0, 1}:
        raise DatasetValidationError(f"{y_col}: outcome must be non-null 0/1 for an outcome-stratified split")
    return y.to_numpy(dtype=np.int64)


def _require_non_empty(masks: dict[str, np.ndarray], context: str) -> None:
    empty = [name for name in SPLITS if not masks[name].any()]
    if empty:
        counts = {name: int(masks[name].sum()) for name in SPLITS}
        raise ConfigError(f"split partitions {empty} are empty {context} (row counts {counts})")


def _assert_valid(plan: SplitPlan, ids: pd.Series, *, patients_disjoint: bool) -> None:
    parts = plan.indices()
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if np.intersect1d(parts[a], parts[b]).size:
            raise LeakageError(f"split plan assigns rows to both {a} and {b}")
    for name, idx in parts.items():
        if idx.size == 0:
            raise ConfigError(f"split partition {name!r} is empty")
    if patients_disjoint:
        patients = {name: set(ids.iloc[idx]) for name, idx in parts.items()}
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
            shared = len(patients[a] & patients[b])
            if shared:
                raise LeakageError(f"{shared} patients have rows in both {a} and {b} but validation.patients_disjoint is true (D-15)")


# ---------------------------------------------------------------------- locked test set
class TestSetGuard:
    """Holds the locked test partition; ``release`` hands it out exactly once, after model selection is frozen (D-19 §4)."""

    __test__ = False  # not a pytest test class

    def __init__(self, frame: pd.DataFrame):
        if not isinstance(frame, pd.DataFrame):
            raise ConfigError(f"TestSetGuard expects a DataFrame, got {type(frame).__name__}")
        self._frame: pd.DataFrame | None = frame.copy(deep=False)
        self._n_rows = len(frame)
        self._release_reason: str | None = None

    @property
    def released(self) -> bool:
        return self._frame is None

    @property
    def n_rows(self) -> int:
        return self._n_rows

    def release(self, reason: str) -> pd.DataFrame:
        """Return the test rows; a second call raises ``LeakageError``."""
        if self._frame is None:
            raise LeakageError(f"test set was already released ({self._release_reason!r}); it may be released only once (D-19 §4)")
        if not isinstance(reason, str) or not reason.strip():
            raise ConfigError("TestSetGuard.release requires a non-empty reason")
        frame, self._frame = self._frame, None
        self._release_reason = reason
        log.info("test_set_released", extra_fields={"n_rows": self._n_rows, "reason": reason})
        return frame

    def __repr__(self) -> str:
        return f"TestSetGuard(n_rows={self._n_rows}, released={self.released})"


# ---------------------------------------------------------------------- grouped cross-validation
def grouped_kfold_indices(groups: Sequence[Any] | np.ndarray | pd.Series, n_folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Deterministic grouped K-fold ``[(train_idx, test_idx), ...]`` balanced by rows (D-11, D-15).

    Groups (patients) are ordered largest first, ties in a seeded random order, and each is placed in the fold
    with the fewest rows so far (lowest fold number on ties). The result does not depend on row order.
    """
    if isinstance(n_folds, bool) or not isinstance(n_folds, (int, np.integer)) or n_folds < 2:
        raise ConfigError(f"n_folds must be an integer >= 2, got {n_folds!r}")
    g = pd.Series(np.asarray(groups, dtype=object))
    if g.empty:
        raise DatasetValidationError("grouped_kfold_indices received zero rows")
    if g.isna().any():
        raise DatasetValidationError(f"grouped_kfold_indices: {int(g.isna().sum())} null group values")
    codes, uniques = pd.factorize(g.astype("str"), sort=True)
    n_groups = len(uniques)
    if n_groups < n_folds:
        raise ConfigError(f"grouped_kfold_indices: {n_groups} groups cannot fill {n_folds} folds")
    sizes = np.bincount(codes, minlength=n_groups)
    order = rng_for(seed, KFOLD_COMPONENT).permutation(n_groups)
    order = order[np.argsort(-sizes[order], kind="stable")]
    fold_of_group = np.empty(n_groups, dtype=np.int64)
    load = np.zeros(int(n_folds), dtype=np.int64)
    for group in order:
        fold = int(np.argmin(load))
        fold_of_group[group] = fold
        load[fold] += sizes[group]
    fold_of_row = fold_of_group[codes]
    return [(np.flatnonzero(fold_of_row != k), np.flatnonzero(fold_of_row == k)) for k in range(int(n_folds))]
