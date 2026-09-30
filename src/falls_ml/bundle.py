"""Model bundle: the single production artifact, saved and loaded with SHA-256 integrity (architecture §1.4, §3.4).

A bundle directory holds everything production inference needs, so training and production run exactly the same
preprocessing:

- ``preprocessor.pkl``: the fitted :class:`~falls_ml.features.preprocessing.Preprocessor` (pickle protocol 5);
- ``adapter/``: model files written by :meth:`ModelAdapter.save`;
- ``calibrator.json``: recalibration method and parameters (method ``none`` when absent; D-07, D-19);
- ``reference_profile.json``: training reference distributions for monitoring;
- ``feature_spec.json``: feature spec identity (name, version, SHA-256, source paths, predictors, levels) and the
  embedded spec payloads (base + extension YAML mappings and the subset list), so loading never reads YAML from disk;
- ``bundle.json``: provenance metadata (including ``served_variant``) and the SHA-256 of every other file.

Security: ``load_bundle`` unpickles files. Pickles execute code when loaded, so only load bundles from trusted
locations. Every listed file's SHA-256 is verified (and unlisted files are refused) *before* anything is
unpickled, but ``bundle.json`` itself is not signed: integrity checking detects accidental corruption and casual
tampering, not a malicious author who rewrites the manifest.
"""

from __future__ import annotations

import json
import pickle
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from falls_ml.artifacts import environment_info, write_json
from falls_ml.config import ExperimentConfig, experiment_config_from_dict
from falls_ml.data.dataset import DatasetManifest, sha256_file
from falls_ml.errors import BundleIntegrityError, ConfigError, PreprocessingMismatchError
from falls_ml.evaluation.calibration import NoRecalibration, recalibrator_from_dict
from falls_ml.features.preprocessing import Preprocessor
from falls_ml.features.spec import FeatureSpec, feature_spec_payloads, load_feature_spec_from_payloads
from falls_ml.logging_utils import get_logger
from falls_ml.models.registry import get_adapter_class
from falls_ml.pipeline import FittedPipeline

log = get_logger(__name__)

SCHEMA_VERSION = "1.0"
PICKLE_PROTOCOL = 5
BUNDLE_FILE = "bundle.json"
PREPROCESSOR_FILE = "preprocessor.pkl"
ADAPTER_DIR = "adapter"
CALIBRATOR_FILE = "calibrator.json"
REFERENCE_PROFILE_FILE = "reference_profile.json"
FEATURE_SPEC_FILE = "feature_spec.json"
#: file-manager metadata that Windows Explorer (desktop.ini, Thumbs.db) and macOS Finder (.DS_Store) drop into folders; never
#: read by load_bundle, so an UNLISTED copy is ignored with a warning (case-insensitive, any depth, regular files only)
SHELL_METADATA_FILES = frozenset({"desktop.ini", "thumbs.db", ".ds_store"})


@dataclass(frozen=True)
class LoadedBundle:
    """A verified bundle: the fitted pipeline plus its provenance and training reference profile."""

    pipeline: FittedPipeline
    metadata: dict[str, Any]
    feature_spec: FeatureSpec
    reference_profile: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    directory: Path | None = None

    @property
    def model_version(self) -> str:
        return str(self.metadata["model"]["version"])

    @property
    def served_variant(self) -> str:
        """The prediction variant this bundle serves by default (``recalibrated`` iff a recalibrator is attached)."""
        return str(self.metadata["served_variant"])


# ============================================================================ save
def feature_spec_identity(spec: FeatureSpec) -> dict[str, Any]:
    """Content of ``feature_spec.json``: the spec identity plus the payloads that rebuild it without any file access."""
    if not spec.raw_payloads:
        raise PreprocessingMismatchError(f"feature spec {spec.name} {spec.version} carries no raw payloads and cannot be "
                                         "embedded in a bundle; load it with load_feature_spec")
    return {
        "name": spec.name,
        "version": spec.version,
        "sha256": spec.content_sha256,
        "source_paths": list(spec.source_paths),
        "predictor_names": spec.predictor_names(),
        "levels": {f.name: list(f.levels) for f in spec.features if f.levels},
        "payloads": feature_spec_payloads(spec),
    }


def rebuild_feature_spec(identity: Mapping[str, Any]) -> FeatureSpec:
    """Rebuild the training feature spec from the payloads embedded in ``feature_spec.json`` (never reads YAML files).

    The subset is re-applied when recorded; the caller verifies the resulting SHA-256.
    """
    try:
        payloads = identity["payloads"]
        spec = load_feature_spec_from_payloads(payloads["base"], payloads["extensions"], source_paths=payloads["source_paths"])
        if payloads["subset"] is not None:
            spec = spec.subset(payloads["subset"])
    except (ConfigError, KeyError, TypeError, AttributeError, ValueError) as exc:
        raise BundleIntegrityError(f"{FEATURE_SPEC_FILE} does not embed a rebuildable feature spec ({type(exc).__name__}: {exc}); "
                                   "re-save the bundle") from exc
    return spec


def _is_calibrated_variant(variant: str) -> bool:
    return variant == "recalibrated" or variant.endswith("+recalibrated")


def _has_calibrator(calibrator: Any) -> bool:
    return calibrator is not None and calibrator.method != "none"


def _served_variant(pipeline: FittedPipeline, extra_metadata: Mapping[str, Any]) -> str:
    """``extra_metadata['served_variant']`` when given (must agree with the attached calibrator), else derived from it."""
    calibrated = _has_calibrator(pipeline.calibrator)
    variant = extra_metadata.get("served_variant")
    if variant is None:
        return "recalibrated" if calibrated else "uncalibrated"
    if not isinstance(variant, str) or not variant.strip():
        raise ConfigError(f"extra_metadata['served_variant'] must be a non-empty string, got {variant!r}")
    if _is_calibrated_variant(variant) != calibrated:
        raise ConfigError(f"served_variant {variant!r} contradicts the pipeline: a recalibrator is "
                          f"{'attached' if calibrated else 'not attached'} (attach one iff the served variant is recalibrated)")
    return variant


def _selected_design_columns(pipeline: FittedPipeline, design_columns: list[str]) -> list[str]:
    """Design columns with a non-zero coefficient (linear models); all design columns when selection is undefined."""
    table = pipeline.model.get_feature_importance()
    selected = table["selected"]
    if selected.isna().any():
        return list(design_columns)
    chosen = set(table.loc[selected.astype(bool), "feature"])
    return [c for c in design_columns if c in chosen]


def _manifest_dict(manifest: DatasetManifest | Mapping[str, Any]) -> dict[str, Any]:
    return manifest.to_dict() if isinstance(manifest, DatasetManifest) else DatasetManifest.from_dict(dict(manifest)).to_dict()


def _risk_categories_record(config: ExperimentConfig) -> dict[str, Any]:
    """``bundle.json`` risk-category block; always derived from the (hashed) experiment config (M-12, B-03)."""
    rc = config.reporting.risk_categories
    return {"approved": rc.approved, "cutpoints": list(rc.cutpoints), "labels": list(rc.labels),
            "approval_reference": rc.approval_reference}


def _check_reference_profile(profile: Mapping[str, Any], spec: FeatureSpec, design_columns: list[str]) -> None:
    """The profile must summarise this pipeline's training rows: it supplies the contribution reference and drift baseline."""
    profile_sha = (profile.get("feature_spec") or {}).get("sha256")
    means = profile.get("design_column_means")
    if profile_sha != spec.content_sha256 or not isinstance(means, Mapping) or set(means) != set(design_columns):
        raise PreprocessingMismatchError("reference_profile was not built with this pipeline (feature spec SHA-256 or "
                                         "design_column_means columns differ); use monitoring.build_reference_profile")


def _hash_files(directory: Path) -> dict[str, str]:
    """{posix relative path: sha256} for every file in ``directory`` except ``bundle.json``."""
    # ordered by case-sensitive path components: the POSIX sorted(Path) order, also on Windows (where Path ordering ignores case)
    return {p.relative_to(directory).as_posix(): sha256_file(p)
            for p in sorted(directory.rglob("*"), key=lambda q: q.relative_to(directory).parts)
            if p.is_file() and p.relative_to(directory).as_posix() != BUNDLE_FILE}


def save_bundle(directory: str | Path, pipeline: FittedPipeline, *, experiment_config: ExperimentConfig,
                dataset_manifest: DatasetManifest | Mapping[str, Any], reference_profile: Mapping[str, Any],
                model_version: str, created_utc: str, extra_metadata: Mapping[str, Any] | None = None) -> Path:
    """Write a model bundle into ``directory`` (which must be absent or empty) and return the directory.

    Risk-category thresholds are stored with their ``approved`` flag; inference exposes categories only when
    approved (M-12, B-03). ``reference_profile`` must be built by ``monitoring.build_reference_profile`` with this
    pipeline (it supplies the contribution reference and the drift baseline).
    """
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise BundleIntegrityError(f"Refusing to write a bundle into non-empty directory {directory}")
    if not str(model_version).strip() or not str(created_utc).strip():
        raise ConfigError("save_bundle requires a non-empty model_version and created_utc")
    pre, model, spec = pipeline.preprocessor, pipeline.model, pipeline.spec
    design_columns = pre.design_columns()
    if pre.spec.content_sha256 != spec.content_sha256:
        raise PreprocessingMismatchError("pipeline.preprocessor was fitted with a different feature spec than pipeline.spec")
    if model.feature_names_ is not None and list(model.feature_names_) != design_columns:
        raise PreprocessingMismatchError("model feature names differ from the preprocessor design columns")
    _check_reference_profile(reference_profile, spec, design_columns)
    manifest = _manifest_dict(dataset_manifest)
    extra = dict(extra_metadata or {})
    served_variant = _served_variant(pipeline, extra)
    # What is written is exactly what load_bundle reads back: prove now that it rebuilds the training spec.
    try:
        spec_identity = json.loads(json.dumps(feature_spec_identity(spec), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise PreprocessingMismatchError(f"feature spec payloads are not JSON-serialisable: {exc}") from exc
    if rebuild_feature_spec(spec_identity).content_sha256 != spec.content_sha256:
        raise PreprocessingMismatchError("the embedded feature spec payloads do not reproduce the training feature spec SHA-256")
    directory.mkdir(parents=True, exist_ok=True)

    with (directory / PREPROCESSOR_FILE).open("wb") as handle:
        pickle.dump(pre, handle, protocol=PICKLE_PROTOCOL)
    (directory / ADAPTER_DIR).mkdir()
    model.save(directory / ADAPTER_DIR)
    calibrator = pipeline.calibrator if pipeline.calibrator is not None else NoRecalibration()
    calibration = calibrator.to_dict()
    write_json(directory / CALIBRATOR_FILE, calibration)
    write_json(directory / REFERENCE_PROFILE_FILE, dict(reference_profile))
    write_json(directory / FEATURE_SPEC_FILE, spec_identity)

    cfg = experiment_config
    environment = environment_info()
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "model": {"name": model.name, "version": str(model_version), "params": model.get_params(),
                  "is_linear": bool(model.is_linear), "representation": model.representation},
        "experiment": {"name": cfg.experiment.name, "kind": cfg.experiment.kind, "layers": list(cfg.experiment.layers),
                       "description": cfg.experiment.description},
        "config_sha256": cfg.sha256(),
        "config": cfg.to_dict(),
        "training_dataset": manifest,
        "dataset_version": manifest["dataset_version"],
        "mapping_version": manifest["mapping_version"],
        "feature_spec": {k: spec_identity[k] for k in ("name", "version", "sha256")},
        "preprocessing": {"representation": pre.representation, "fingerprint": pre.fingerprint(),
                          "design_columns": design_columns,
                          "selected_design_columns": _selected_design_columns(pipeline, design_columns),
                          "category_levels": spec_identity["levels"], "reference_levels": pre.reference_levels()},
        "risk_categories": _risk_categories_record(cfg),
        "candidate_thresholds": list(cfg.evaluation.thresholds),
        "calibration": calibration,
        "served_variant": served_variant,
        "code_version": environment["code_version"],
        "environment": environment,
        "created_utc": str(created_utc),
        "extra_metadata": extra,
        "files": _hash_files(directory),
    }
    write_json(directory / BUNDLE_FILE, metadata)
    log.info("bundle_saved", extra_fields={"directory": str(directory), "model": model.name, "model_version": str(model_version),
                                           "served_variant": served_variant,
                                           "preprocessor_fingerprint": metadata["preprocessing"]["fingerprint"],
                                           "n_files": len(metadata["files"])})
    return directory


# ============================================================================ load
def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BundleIntegrityError(f"Cannot read {path}: {exc}") from exc


def verify_bundle_files(directory: str | Path) -> dict[str, Any]:
    """Return ``bundle.json`` after checking every listed file's SHA-256 and that no file is missing or unlisted."""
    directory = Path(directory)
    if not (directory / BUNDLE_FILE).is_file():
        raise BundleIntegrityError(f"{directory} is not a model bundle ({BUNDLE_FILE} missing)")
    metadata = _read_json(directory / BUNDLE_FILE)
    if not isinstance(metadata, dict):
        raise BundleIntegrityError(f"{BUNDLE_FILE} is not a JSON object")
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise BundleIntegrityError(f"Unsupported bundle schema version {metadata.get('schema_version')!r} (expected {SCHEMA_VERSION})")
    listed = metadata.get("files")
    if not isinstance(listed, dict) or not listed:
        raise BundleIntegrityError(f"{BUNDLE_FILE} has no file manifest")
    required = {PREPROCESSOR_FILE, CALIBRATOR_FILE, REFERENCE_PROFILE_FILE, FEATURE_SPEC_FILE}
    problems = [f"required file not listed: {name}" for name in sorted(required - set(listed))]
    actual = {p.relative_to(directory).as_posix(): p for p in directory.rglob("*") if p.is_file() or p.is_symlink()}
    actual.pop(BUNDLE_FILE, None)
    problems += [f"missing file: {name}" for name in sorted(set(listed) - set(actual))]
    unlisted = sorted(set(actual) - set(listed))
    ignored = [n for n in unlisted if n.rsplit("/", 1)[-1].lower() in SHELL_METADATA_FILES and not actual[n].is_symlink() and actual[n].is_file()]
    if ignored:
        log.warning("bundle_shell_metadata_ignored", extra_fields={"bundle": str(directory), "files": ignored})
    unexpected = [n for n in unlisted if n not in ignored]
    problems += [f"unexpected file: {name}" for name in unexpected]
    for name in sorted(set(listed) & set(actual)):
        path = actual[name]
        if path.is_symlink() or not path.is_file():
            problems.append(f"not a regular file: {name}")
        elif sha256_file(path) != listed[name]:
            problems.append(f"SHA-256 mismatch: {name}")
    if problems:
        hint = (" (a bundle must contain exactly the files listed in bundle.json: remove files added after saving, e.g. sync-tool "
                "conflict copies such as OneDrive '<name>-<COMPUTER>.json')") if unexpected else ""
        raise BundleIntegrityError(f"Model bundle {directory} failed integrity verification: " + "; ".join(problems) + hint)
    return metadata


def _environment_warnings(recorded: Mapping[str, Any]) -> list[str]:
    current = environment_info()
    out = []
    for pkg, version in sorted((recorded.get("packages") or {}).items()):
        now = current["packages"].get(pkg)
        if now != version:
            out.append(f"library version differs: {pkg} bundle={version} runtime={now}")
    bundle_python, runtime_python = (str(v).split(" ", 1)[0] for v in (recorded.get("python", ""), current["python"]))
    if bundle_python != runtime_python:
        out.append(f"python version differs: bundle={bundle_python} runtime={runtime_python}")
    bundle_code = (recorded.get("code_version") or {}).get("source_tree_sha256")
    if bundle_code != current["code_version"]["source_tree_sha256"]:
        out.append("code version differs: falls_ml source tree SHA-256 at training differs from the runtime source tree")
    return out


def load_bundle(directory: str | Path, *, expected_feature_spec: FeatureSpec | None = None) -> LoadedBundle:
    """Load and verify a bundle written by :func:`save_bundle`.

    Order: file SHA-256 manifest (``BundleIntegrityError``) → adapter via the registry → preprocessor unpickling →
    fingerprint and design columns (``PreprocessingMismatchError``) → feature spec rebuilt from the payloads embedded
    in ``feature_spec.json`` (never from YAML on disk) and its SHA-256 verified; ``expected_feature_spec``, when given,
    must have the same SHA-256 (``PreprocessingMismatchError``) → calibrator, which must agree with ``served_variant``.
    The config in ``bundle.json`` must reproduce its recorded SHA-256 and the risk-category block must match it
    (``BundleIntegrityError``). Library, Python and code version differences are returned as warnings. Only load
    bundles from trusted locations.
    """
    directory = Path(directory)
    metadata = verify_bundle_files(directory)
    try:
        model_meta, pre_meta = metadata["model"], metadata["preprocessing"]
        model_version = str(model_meta["version"])
        recorded_fingerprint, recorded_columns = pre_meta["fingerprint"], list(pre_meta["design_columns"])
        identity = _read_json(directory / FEATURE_SPEC_FILE)
        recorded_sha = metadata["feature_spec"]["sha256"]
        served_variant = metadata["served_variant"]
        if identity["sha256"] != recorded_sha:
            raise BundleIntegrityError("feature_spec.json and bundle.json record different feature spec SHA-256 values")
        # bundle.json is not in its own hash manifest: its risk-category block (which gates category exposure, M-12)
        # must match the experiment config, whose SHA-256 was recorded at training.
        config = experiment_config_from_dict(metadata["config"], name=BUNDLE_FILE)
        if config.sha256() != metadata["config_sha256"]:
            raise BundleIntegrityError(f"{BUNDLE_FILE} config does not reproduce its recorded config_sha256 (edited?)")
        if metadata["risk_categories"] != _risk_categories_record(config):
            raise BundleIntegrityError(f"{BUNDLE_FILE} risk_categories differ from the hashed experiment config; approval "
                                       "of risk categories requires a new bundle (M-12, B-03)")

        adapter_cls = get_adapter_class(model_meta["name"])
        if adapter_cls.representation != pre_meta["representation"]:
            raise BundleIntegrityError(f"adapter {adapter_cls.name!r} needs representation {adapter_cls.representation!r}, "
                                       f"bundle records {pre_meta['representation']!r}")
        model = adapter_cls.load(directory / ADAPTER_DIR)
        calibrator_state = _read_json(directory / CALIBRATOR_FILE)
        reference_profile = _read_json(directory / REFERENCE_PROFILE_FILE)
    except (KeyError, TypeError) as exc:
        raise BundleIntegrityError(f"{BUNDLE_FILE} or {FEATURE_SPEC_FILE} is malformed: missing {exc}") from exc
    except ConfigError as exc:
        raise BundleIntegrityError(f"Bundle config or model cannot be rebuilt: {exc}") from exc

    with (directory / PREPROCESSOR_FILE).open("rb") as handle:  # SHA-256 verified above
        try:
            pre = pickle.load(handle)
        except (pickle.UnpicklingError, EOFError, AttributeError, ImportError, TypeError, ValueError) as exc:
            raise BundleIntegrityError(f"Cannot unpickle {PREPROCESSOR_FILE}: {exc}") from exc
    if not isinstance(pre, Preprocessor):
        raise BundleIntegrityError(f"{PREPROCESSOR_FILE} does not contain a Preprocessor (got {type(pre).__name__})")
    fingerprint = pre.fingerprint()
    if fingerprint != recorded_fingerprint:
        raise PreprocessingMismatchError(f"Preprocessor fingerprint {fingerprint[:12]}… differs from the bundle record "
                                         f"{str(recorded_fingerprint)[:12]}… (preprocessing changed since training)")
    if pre.design_columns() != recorded_columns or pre.representation != pre_meta["representation"]:
        raise PreprocessingMismatchError("Preprocessor design columns or representation differ from the bundle record")
    if model.feature_names_ is not None and list(model.feature_names_) != pre.design_columns():
        raise PreprocessingMismatchError("Model feature names differ from the preprocessor design columns")

    spec = rebuild_feature_spec(identity)
    if spec.content_sha256 != recorded_sha:
        raise PreprocessingMismatchError(f"The feature spec embedded in {FEATURE_SPEC_FILE} does not reproduce the training "
                                         f"SHA-256 ({spec.content_sha256[:12]}… vs {recorded_sha[:12]}…)")
    if expected_feature_spec is not None and expected_feature_spec.content_sha256 != recorded_sha:
        raise PreprocessingMismatchError(f"expected_feature_spec SHA-256 {expected_feature_spec.content_sha256[:12]}… differs "
                                         f"from the training feature spec {recorded_sha[:12]}…")
    if pre.spec.content_sha256 != recorded_sha:
        raise PreprocessingMismatchError("The pickled preprocessor was fitted with a different feature spec")

    try:
        calibrator = recalibrator_from_dict(calibrator_state)
    except ConfigError as exc:
        raise BundleIntegrityError(f"Invalid {CALIBRATOR_FILE}: {exc}") from exc
    if not isinstance(served_variant, str) or _is_calibrated_variant(served_variant) != _has_calibrator(calibrator):
        raise BundleIntegrityError(f"{BUNDLE_FILE} served_variant {served_variant!r} contradicts {CALIBRATOR_FILE} "
                                   f"(method {calibrator.method!r})")
    pipeline = FittedPipeline(spec=spec, preprocessor=pre, model=model,
                              calibrator=None if calibrator.method == "none" else calibrator)
    warnings = tuple(_environment_warnings(metadata.get("environment") or {}))
    for message in warnings:
        log.warning("bundle_environment_difference", extra_fields={"directory": str(directory), "detail": message})
    log.info("bundle_loaded", extra_fields={"directory": str(directory), "model": model.name,
                                            "model_version": model_version, "preprocessor_fingerprint": fingerprint,
                                            "calibration": calibrator.method, "served_variant": served_variant,
                                            "n_warnings": len(warnings)})
    return LoadedBundle(pipeline=pipeline, metadata=metadata, feature_spec=spec, reference_profile=reference_profile,
                        warnings=warnings, directory=directory)
