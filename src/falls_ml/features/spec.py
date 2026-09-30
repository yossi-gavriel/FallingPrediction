"""Machine-readable feature specification (loaded from ``configs/features/*.yaml``).

The feature spec is the single source of truth for predictor names, types, allowed levels,
missing-data rules, reference levels and layer tags. Model code never hard-codes clinical
definitions; it asks the spec.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from falls_ml.errors import ConfigError

DTYPES = {"binary", "float", "float_nullable", "count", "categorical", "categorical_nullable"}
MISSING_RULES = {"forbid", "absent_is_zero", "missing_category", "reference_level"}
LAYERS = {"L1_published", "L2_assumption", "L3a_meuhedet_mapping", "L3b_meuhedet_predictor", "L4_alternative"}
ROLES = {"predictor"}


@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    concept: str
    dtype: str
    layer: str
    missing_rule: str
    exact_efalls_baseline: bool
    clinically_validated: bool
    levels: tuple[str, ...] = ()
    reference_level: Any = None
    valid_range: Mapping[str, Any] | None = None
    efalls_term: str | None = None
    retained_in_published_model: bool | None = None
    group: str = "efalls_baseline"
    raw: Mapping[str, Any] = field(default_factory=dict, compare=False, hash=False, repr=False)

    @property
    def nullable(self) -> bool:
        return self.dtype.endswith("_nullable")

    @property
    def is_binary(self) -> bool:
        return self.dtype == "binary"

    @property
    def is_categorical(self) -> bool:
        return self.dtype.startswith("categorical")


@dataclass(frozen=True)
class OutcomeDefinition:
    """Outcome window (D-00/D-09): index_date <= event_date <= index_date + horizon_years calendar years - 1 day.

    A day-based window (``window.horizon_days`` instead of ``horizon_years``; exploratory Meuhedet outcomes only) ends at
    index_date + horizon_days - 1, i.e. the window covers ``horizon_days`` calendar days including the index day.
    """

    name: str
    horizon_years: int
    codes: tuple[Mapping[str, str], ...]
    raw: Mapping[str, Any] = field(default_factory=dict, compare=False, hash=False, repr=False)
    horizon_days: int | None = None

    @property
    def horizon_label(self) -> str:
        return f"{self.horizon_days}d" if self.horizon_days is not None else f"{self.horizon_years}y"

    @property
    def is_published_efalls(self) -> bool:
        """True for the published 12-month eFalls outcome (layer L1_published); False for exploratory outcomes."""
        return self.raw.get("layer") == "L1_published" and self.horizon_days is None

    def window_end(self, index_dates: Any) -> Any:
        """Last day (inclusive) of the outcome window for scalar or Series ``index_dates``."""
        import pandas as pd

        if self.horizon_days is not None:
            return pd.to_datetime(index_dates) + pd.Timedelta(days=self.horizon_days - 1)
        return pd.to_datetime(index_dates) + pd.DateOffset(years=self.horizon_years) - pd.Timedelta(days=1)


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    version: str
    features: tuple[FeatureDefinition, ...]
    outcome: OutcomeDefinition
    identifier_columns: tuple[str, ...]
    index_column: str
    provenance_columns: Mapping[str, Mapping[str, Any]]
    metadata_columns: Mapping[str, Mapping[str, Any]]
    age_min: float
    source_paths: tuple[str, ...]
    content_sha256: str
    cohort: Mapping[str, Any] = field(default_factory=dict, compare=False, hash=False)
    raw_payloads: tuple[Mapping[str, Any], ...] = field(default_factory=tuple, compare=False, hash=False, repr=False)
    subset_of: str | None = None   # content_sha256 of the full spec when this is a subset

    # ------------------------------------------------------------------ queries
    def predictor_names(self) -> list[str]:
        return [f.name for f in self.features]

    def get(self, name: str) -> FeatureDefinition:
        for f in self.features:
            if f.name == name:
                return f
        raise ConfigError(f"Feature {name!r} is not declared in feature spec {self.name} {self.version}")

    def binary_names(self) -> list[str]:
        return [f.name for f in self.features if f.is_binary]

    def groups(self) -> list[str]:
        seen: list[str] = []
        for f in self.features:
            if f.group not in seen:
                seen.append(f.group)
        return seen

    def subset(self, names: Iterable[str]) -> FeatureSpec:
        """Return a spec restricted to ``names`` (order preserved from this spec). Unknown or duplicated names fail.

        Root-based and idempotent: the digest depends only on the root (full) spec SHA-256 and the kept names, so a
        subset of a subset equals the direct subset, and a subset keeping every predictor is this spec itself.
        """
        wanted = list(names)
        if isinstance(names, str) or not wanted:
            raise ConfigError(f"A feature subset must be a non-empty list of predictor names, got {names!r}")
        unknown = sorted(set(wanted) - set(self.predictor_names()))
        if unknown:
            raise ConfigError(f"Unknown features requested (not in {'this subset of ' if self.subset_of else ''}"
                              f"feature spec {self.name} {self.version}): {unknown}")
        dupes = sorted({n for n in wanted if wanted.count(n) > 1})
        if dupes:
            raise ConfigError(f"Duplicated features requested: {dupes}")
        kept = tuple(f for f in self.features if f.name in set(wanted))
        if len(kept) == len(self.features):
            return self
        root = self.subset_of or self.content_sha256
        digest = _sha256_json({"base": root, "subset": [f.name for f in kept]})
        return FeatureSpec(
            name=self.name, version=self.version, features=kept, outcome=self.outcome,
            identifier_columns=self.identifier_columns, index_column=self.index_column,
            provenance_columns=self.provenance_columns, metadata_columns=self.metadata_columns,
            age_min=self.age_min, source_paths=self.source_paths, content_sha256=digest, cohort=self.cohort,
            raw_payloads=self.raw_payloads, subset_of=root,
        )

    @property
    def is_subset(self) -> bool:
        return self.subset_of is not None

    @property
    def is_pure_efalls(self) -> bool:
        return all(f.exact_efalls_baseline and f.layer == "L1_published" for f in self.features)


# ---------------------------------------------------------------------- loading
def _sha256_json(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _parse_feature(d: Mapping[str, Any], default_group: str) -> FeatureDefinition:
    unknown = sorted(set(d) - ALLOWED_FEATURE_KEYS)
    if unknown:
        raise ConfigError(f"Feature {d.get('name', '<unnamed>')!r} has unknown keys {unknown} (typo?)")
    required = ["name", "concept", "dtype", "layer", "missing_rule", "exact_efalls_baseline", "clinically_validated"]
    missing = [k for k in required if k not in d]
    if missing:
        raise ConfigError(f"Feature {d.get('name', '<unnamed>')!r} missing keys: {missing}")
    if d["dtype"] not in DTYPES:
        raise ConfigError(f"Feature {d['name']!r}: dtype {d['dtype']!r} not in {sorted(DTYPES)}")
    if d["missing_rule"] not in MISSING_RULES:
        raise ConfigError(f"Feature {d['name']!r}: missing_rule {d['missing_rule']!r} not in {sorted(MISSING_RULES)}")
    if d["layer"] not in LAYERS:
        raise ConfigError(f"Feature {d['name']!r}: layer {d['layer']!r} not in {sorted(LAYERS)}")
    role = d.get("role", "predictor")
    if role not in ROLES:
        raise ConfigError(f"Feature {d['name']!r}: role {role!r} not allowed (identifiers are never predictors)")
    levels = tuple(d.get("levels") or ())
    if d["dtype"].startswith("categorical") and not levels:
        raise ConfigError(f"Categorical feature {d['name']!r} must declare levels")
    if d["dtype"] == "binary" and d["missing_rule"] != "absent_is_zero":
        raise ConfigError(f"Binary feature {d['name']!r} must use missing_rule absent_is_zero (spec §5.9)")
    if not d["dtype"].endswith("_nullable") and d["missing_rule"] in {"missing_category", "reference_level"}:
        raise ConfigError(f"Feature {d['name']!r}: missing_rule {d['missing_rule']!r} requires a nullable dtype")
    return FeatureDefinition(
        name=str(d["name"]), concept=str(d["concept"]), dtype=d["dtype"], layer=d["layer"],
        missing_rule=d["missing_rule"], exact_efalls_baseline=bool(d["exact_efalls_baseline"]),
        clinically_validated=bool(d["clinically_validated"]), levels=levels,
        reference_level=d.get("reference_level"), valid_range=d.get("valid_range"),
        efalls_term=d.get("efalls_term"), retained_in_published_model=d.get("retained_in_published_model"),
        group=str(d.get("group", default_group)), raw=dict(d),
    )


ALLOWED_FEATURE_KEYS = frozenset({
    "name", "concept", "role", "dtype", "layer", "levels", "reference_level", "missing_rule", "valid_range", "efalls_term",
    "efalls_source", "efalls_origin", "exact_efalls_baseline", "retained_in_published_model", "time_window", "transformation",
    "decisions", "published_prevalence", "development_support", "meuhedet_source", "clinically_validated", "sensitivity_variants",
    "group",
})
ALLOWED_TOP_KEYS = frozenset({"feature_set", "identifiers", "provenance_columns", "metadata_columns", "cohort", "outcome", "features"})


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Feature spec not found: {path}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"Feature spec {path} must be a YAML mapping")
    return raw


def load_feature_spec(path: str | Path, extensions: Iterable[str | Path] = (), *, anchor: str | Path | None = None) -> FeatureSpec:
    """Load a base feature spec and optional extension specs (e.g. Meuhedet-enhanced feature groups).

    Relative paths are resolved with :func:`falls_ml.paths.resolve_path` (``anchor`` = where they were declared).
    Extension files contain only ``feature_set`` (name/version/layer) and ``features``; every extension
    feature must be tagged ``exact_efalls_baseline: false`` and a non-L1 layer, and carry a ``group``.
    """
    from falls_ml.paths import portable_path, resolve_path

    base_path = resolve_path(path, anchor=anchor)
    ext_paths = [resolve_path(e, anchor=anchor) for e in extensions]
    return load_feature_spec_from_payloads(_read_yaml(base_path), [_read_yaml(e) for e in ext_paths],
                                           source_paths=[portable_path(path), *map(portable_path, extensions)])


def feature_spec_payloads(spec: FeatureSpec) -> dict[str, Any]:
    """Raw YAML payloads needed to rebuild ``spec`` exactly (embedded in model bundles)."""
    return {"base": spec.raw_payloads[0], "extensions": list(spec.raw_payloads[1:]), "source_paths": list(spec.source_paths),
            "subset": [f.name for f in spec.features] if spec.subset_of is not None else None}


def load_feature_spec_from_payloads(base_raw: dict[str, Any], extension_raws: Iterable[dict[str, Any]] = (), *,
                                    source_paths: Iterable[str] = ()) -> FeatureSpec:
    """Build a FeatureSpec from already-parsed YAML mappings (no file access). Unknown keys are errors."""
    raw = base_raw
    unknown_top = sorted(set(raw) - ALLOWED_TOP_KEYS)
    if unknown_top:
        raise ConfigError(f"Feature spec has unknown top-level keys {unknown_top}")
    for key in ("feature_set", "identifiers", "outcome", "features", "cohort"):
        if key not in raw:
            raise ConfigError(f"Feature spec missing top-level key {key!r}")
    feats = [_parse_feature(f, "efalls_baseline") for f in raw["features"]]
    hash_payload: list[Any] = [raw]
    ext_list = list(extension_raws)
    for i, ext_raw in enumerate(ext_list):
        if not isinstance(ext_raw, dict) or set(ext_raw) - {"feature_set", "features"} or not ext_raw.get("features") \
                or not isinstance(ext_raw.get("feature_set"), dict):
            raise ConfigError(f"Extension spec #{i + 1} must be a mapping with only 'feature_set' (mapping) and a non-empty 'features' list")
        for fd in ext_raw["features"]:
            f = _parse_feature(fd, fd.get("group", ""))
            if f.exact_efalls_baseline or f.layer == "L1_published":
                raise ConfigError(f"Extension feature {f.name!r} cannot be tagged as exact eFalls baseline / L1")
            if not f.group:
                raise ConfigError(f"Extension feature {f.name!r} must declare a group (for ablation)")
            feats.append(f)
        hash_payload.append(ext_raw)
    names = [f.name for f in feats]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ConfigError(f"Duplicate feature names: {dupes}")

    identifiers = raw["identifiers"]
    id_cols = tuple(k for k, v in identifiers.items() if v.get("role") == "identifier")
    index_cols = [k for k, v in identifiers.items() if v.get("role") == "index"]
    if len(index_cols) != 1:
        raise ConfigError("Feature spec must declare exactly one index column")
    forbidden = set(id_cols) | set(index_cols) | set(raw.get("metadata_columns", {})) | set(raw.get("provenance_columns", {}))
    clash = sorted(forbidden & set(names))
    if clash:
        raise ConfigError(f"Identifier/metadata/provenance columns declared as predictors: {clash}")

    out = raw["outcome"]
    window = out["window"]
    if ("horizon_years" in window) == ("horizon_days" in window):
        raise ConfigError("outcome.window must declare exactly one of horizon_years (published) or horizon_days (exploratory)")
    horizon_days = int(window["horizon_days"]) if "horizon_days" in window else None
    if horizon_days is not None and horizon_days < 1:
        raise ConfigError("outcome.window.horizon_days must be >= 1")
    if horizon_days is not None and out.get("layer") == "L1_published":
        raise ConfigError("a day-based outcome window cannot be tagged L1_published (the published eFalls outcome is 12 months)")
    outcome = OutcomeDefinition(name=out["name"], horizon_years=0 if horizon_days is not None else int(window["horizon_years"]),
                                codes=tuple(out.get("codes", ())), raw=out, horizon_days=horizon_days)
    if outcome.name in names:
        raise ConfigError("Outcome column declared as a predictor")
    fs = raw["feature_set"]
    ext_names = [str(e["feature_set"].get("name")) for e in ext_list]
    return FeatureSpec(
        name="+".join([str(fs["name"]), *ext_names]), version=str(fs["version"]), features=tuple(feats), outcome=outcome,
        identifier_columns=id_cols, index_column=index_cols[0],
        provenance_columns=dict(raw.get("provenance_columns", {})), metadata_columns=dict(raw.get("metadata_columns", {})),
        age_min=float(raw["cohort"]["age_min"]), source_paths=tuple(source_paths), content_sha256=_sha256_json(hash_payload),
        cohort=dict(raw["cohort"]), raw_payloads=(raw, *ext_list),
    )
