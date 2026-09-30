"""Immutable, versioned modelling dataset: Parquet data file + ``manifest.json``.

The data-engineering layer writes datasets with :func:`write_modeling_dataset`; the ML layer
reads them with :meth:`ModelingDataset.load`, which verifies the file hash and the schema.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import MISSING, asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from falls_ml.data.schema import ValidationReport, validate_modeling_dataset
from falls_ml.errors import DatasetValidationError
from falls_ml.features.spec import FeatureSpec

DATA_FILE = "modeling_dataset.parquet"
MANIFEST_FILE = "manifest.json"


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class DatasetManifest:
    dataset_version: str
    mapping_version: str
    source: str
    feature_spec_name: str
    feature_spec_version: str
    feature_spec_sha256: str
    data_file: str
    data_sha256: str
    n_rows: int
    n_patients: int
    index_date_min: str
    index_date_max: str
    outcome_prevalence: float | None
    created_utc: str
    generator: str
    scientific_use_allowed: bool
    notes: str = ""
    audit: dict[str, Any] | None = None   # pre-modelling audits, e.g. D-09 outcome audit and eligibility exclusions
    data_freeze_date: str | None = None   # last date of source data extraction (D-19 label maturity); required for scientific use

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DatasetManifest:
        fields = cls.__dataclass_fields__
        unknown = sorted(set(d) - set(fields))
        if unknown:
            raise DatasetValidationError("Manifest has unknown keys", unknown)
        missing = sorted(k for k, f in fields.items() if k not in d and f.default is MISSING and f.default_factory is MISSING)
        if missing:
            raise DatasetValidationError("Manifest missing keys", missing)
        return cls(**d)


class ModelingDataset:
    """Read-only view over a validated modelling dataset."""

    def __init__(self, frame: pd.DataFrame, manifest: DatasetManifest, spec: FeatureSpec, report: ValidationReport,
                 directory: Path | None = None):
        self._frame = frame
        self.manifest = manifest
        self.spec = spec
        self.validation_report = report
        self.directory = directory   # resolved absolute directory when loaded from disk (recorded in run artifacts)

    @property
    def frame(self) -> pd.DataFrame:
        # pandas Copy-on-Write: a shallow copy isolates callers from in-place column assignment.
        return self._frame.copy(deep=False)

    def __len__(self) -> int:
        return len(self._frame)

    @classmethod
    def from_frame(cls, df: pd.DataFrame, manifest: DatasetManifest, spec: FeatureSpec) -> ModelingDataset:
        _check_manifest_matches_spec(manifest, spec)
        report = validate_modeling_dataset(df, spec, mode="training", expected_versions=_versions(manifest))
        return cls(df.reset_index(drop=True), manifest, spec, report)

    @classmethod
    def load(cls, directory: str | Path, spec: FeatureSpec, *, features: Sequence[str] | None = None,
             infer_subset: bool = False) -> ModelingDataset:
        """Load and validate a dataset directory.

        ``spec`` is the full feature spec. ``features`` (an experiment's ``preprocessing.features``) also accepts a dataset
        built with exactly that subset spec; ``infer_subset`` accepts a dataset built with any subset of ``spec``, inferred
        from its predictor columns. Either way the manifest ``feature_spec_sha256`` decides; a mismatch fails loudly.
        """
        directory = Path(directory)
        manifest = read_manifest(directory)
        data_path = directory / manifest.data_file
        if not data_path.exists():
            raise DatasetValidationError(f"Dataset file not found: {data_path}")
        chosen = select_dataset_spec(manifest, spec, features=features,
                                     columns=_parquet_columns(data_path) if infer_subset and features is None else None)
        actual = sha256_file(data_path)
        if actual != manifest.data_sha256:
            raise DatasetValidationError("Dataset file hash does not match manifest (file modified?)",
                                         [f"manifest {manifest.data_sha256}", f"actual   {actual}"])
        df = pd.read_parquet(data_path)
        dataset = cls.from_frame(df, manifest, chosen)
        dataset.directory = directory.resolve()
        if len(df) != manifest.n_rows:
            raise DatasetValidationError(f"Row count {len(df)} differs from manifest {manifest.n_rows}")
        return dataset


def read_manifest(directory: str | Path) -> DatasetManifest:
    manifest_path = Path(directory) / MANIFEST_FILE
    if not manifest_path.is_file():
        raise DatasetValidationError(f"Dataset manifest not found: {manifest_path}")
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetValidationError(f"Dataset manifest is not valid JSON: {manifest_path} ({exc})") from exc
    if not isinstance(raw, dict):
        raise DatasetValidationError(f"Dataset manifest must be a JSON object: {manifest_path}")
    return DatasetManifest.from_dict(raw)


def _parquet_columns(path: Path) -> list[str]:
    import pyarrow.parquet as pq

    return list(pq.read_schema(path).names)


def select_dataset_spec(manifest: DatasetManifest, spec: FeatureSpec, *, features: Sequence[str] | None = None,
                        columns: Iterable[str] | None = None) -> FeatureSpec:
    """The feature spec a dataset was built with: the full ``spec`` or a subset of it (manifest ``feature_spec_sha256`` decides).

    Candidates: ``spec``; ``spec.subset(features)`` when ``features`` is declared; otherwise, when ``columns`` (the dataset's
    column names) are given, the subset of ``spec`` predictors present in them. Name/version are checked for the chosen spec;
    no match raises ``DatasetValidationError`` listing every expected SHA-256.
    """
    candidates: dict[str, tuple[str, FeatureSpec]] = {spec.content_sha256: ("full feature spec", spec)}
    if features is not None:
        sub = spec.subset(features)
        candidates.setdefault(sub.content_sha256, (f"declared feature subset ({len(sub.features)} of {len(spec.features)} predictors)", sub))
    elif columns is not None:
        present = [n for n in spec.predictor_names() if n in set(columns)]
        if present:
            sub = spec.subset(present)
            candidates.setdefault(sub.content_sha256, (f"subset inferred from the dataset columns ({len(sub.features)} of "
                                                       f"{len(spec.features)} predictors)", sub))
    if manifest.feature_spec_sha256 in candidates:
        chosen = candidates[manifest.feature_spec_sha256][1]
        _check_manifest_matches_spec(manifest, chosen)
        return chosen
    problems = []
    if manifest.feature_spec_name != spec.name:
        problems.append(f"feature spec name: manifest {manifest.feature_spec_name!r} vs loaded {spec.name!r}")
    if manifest.feature_spec_version != spec.version:
        problems.append(f"feature spec version: manifest {manifest.feature_spec_version!r} vs loaded {spec.version!r}")
    problems.append(f"feature spec content: manifest sha256 {manifest.feature_spec_sha256} matches none of the expected specs "
                    "(definitions changed without a version bump, or a different feature subset?)")
    problems += [f"expected {label}: sha256 {sha}" for sha, (label, _) in candidates.items()]
    if features is None and columns is None:
        problems.append("a dataset built for a declared feature subset loads only with that subset (experiment preprocessing.features, "
                        "kind efalls_retrained_reduced) or with subset inference (validate-dataset)")
    raise DatasetValidationError("Dataset was built for a different feature spec", problems)


def feature_subset_record(full: FeatureSpec, subset: FeatureSpec) -> dict[str, Any]:
    """Manifest audit entry for a dataset written with a subset feature spec (informational; the SHA-256 is authoritative)."""
    kept = set(subset.predictor_names())
    return {"root_feature_spec_sha256": full.content_sha256, "subset_feature_spec_sha256": subset.content_sha256,
            "n_predictors": len(subset.features), "n_root_predictors": len(full.features),
            "predictors": subset.predictor_names(), "absent_predictors": [n for n in full.predictor_names() if n not in kept]}


def _versions(m: DatasetManifest) -> dict[str, str]:
    return {"dataset_version": m.dataset_version, "mapping_version": m.mapping_version, "source": m.source}


def _check_manifest_matches_spec(manifest: DatasetManifest, spec: FeatureSpec) -> None:
    problems = []
    if manifest.feature_spec_name != spec.name:
        problems.append(f"feature spec name: manifest {manifest.feature_spec_name!r} vs loaded {spec.name!r}")
    if manifest.feature_spec_version != spec.version:
        problems.append(f"feature spec version: manifest {manifest.feature_spec_version!r} vs loaded {spec.version!r}")
    if manifest.feature_spec_sha256 != spec.content_sha256:
        problems.append(f"feature spec content: manifest sha256 {manifest.feature_spec_sha256[:12]}… vs loaded {spec.content_sha256[:12]}… "
                        "(definitions changed without a version bump?)")
    if problems:
        raise DatasetValidationError("Dataset was built for a different feature spec", problems)


def write_modeling_dataset(
    df: pd.DataFrame,
    directory: str | Path,
    spec: FeatureSpec,
    *,
    dataset_version: str,
    mapping_version: str,
    source: str,
    generator: str,
    scientific_use_allowed: bool,
    notes: str = "",
    created_utc: str | None = None,
    audit: dict[str, Any] | None = None,
    data_freeze_date: str | None = None,
    overwrite: bool = False,
) -> DatasetManifest:
    """Validate and write a dataset + manifest. Used by data engineering and synthetic fixtures.

    Datasets are immutable: writing into a non-empty directory requires ``overwrite=True``.
    """
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()) and not overwrite:
        raise DatasetValidationError(f"Dataset directory {directory} is not empty; datasets are immutable (pass overwrite=True to replace)")
    directory.mkdir(parents=True, exist_ok=True)
    for col, value in (("dataset_version", dataset_version), ("mapping_version", mapping_version), ("source", source)):
        if col in df.columns and not (df[col].astype(str) == str(value)).all():
            raise DatasetValidationError(f"input column {col!r} disagrees with the requested value {value!r}")
    df = df.copy()
    df["dataset_version"] = dataset_version
    df["mapping_version"] = mapping_version
    df["source"] = source
    validate_modeling_dataset(df, spec, mode="training",
                              expected_versions={"dataset_version": dataset_version, "mapping_version": mapping_version, "source": source})
    data_path = directory / DATA_FILE
    df.to_parquet(data_path, index=False)
    idx = df[spec.index_column]
    manifest = DatasetManifest(
        dataset_version=dataset_version, mapping_version=mapping_version, source=source,
        feature_spec_name=spec.name, feature_spec_version=spec.version, feature_spec_sha256=spec.content_sha256,
        data_file=DATA_FILE, data_sha256=sha256_file(data_path), n_rows=len(df),
        n_patients=int(df[spec.identifier_columns[0]].nunique()),
        index_date_min=str(idx.min().date()), index_date_max=str(idx.max().date()),
        outcome_prevalence=float(df[spec.outcome.name].mean()),
        created_utc=created_utc or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        generator=generator, scientific_use_allowed=scientific_use_allowed, notes=notes, audit=audit,
        data_freeze_date=data_freeze_date,
    )
    (directory / MANIFEST_FILE).write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8", newline="\n")
    return manifest
