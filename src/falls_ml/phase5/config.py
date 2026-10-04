"""Phase 5 settings (``configs/meuhedet/phase5.yaml``) and the authoritative V21 schema (``configs/meuhedet/phase5_v21_schema.yaml`` + the embedded
VIEW definition): validated, hashed (line-ending insensitive) and recorded in the frozen plan."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from falls_ml.errors import ConfigError
from falls_ml.paths import resolve_path
from falls_ml.phase5.schema import V21Schema, load_v21_schema

DEFAULT_CONFIG = "configs/meuhedet/phase5.yaml"
FAMILIES = ("LASSO", "ENET", "XGB")
PRIMARY_FAMILY = "ENET"
MODES = ("quick", "overnight")
SET_OLD, SET_ALL, SET_SAFE = "OLD", "OLD_PLUS_ALL_NEW_ELIGIBLE", "OLD_PLUS_NEW_SAFE"
SETS = (SET_OLD, SET_ALL, SET_SAFE)
SAFE_CLASSES = ("SAFE_VERIFIED", "SAFE_BOUNDED", "SAFE_ATTESTED")
CLASSES = (*SAFE_CLASSES, "UNCERTAIN_TIMING", "INELIGIBLE_TIMING", "INELIGIBLE_SEMANTICS", "INELIGIBLE_DATA", "INELIGIBLE_LEAKAGE")
REQUIRED = ("index_date", "seed", "phase3_config", "v21_schema", "x_sealing", "cohort", "outcome_contract", "eligibility", "cv", "families",
            "primary_family", "sets", "primary_comparison", "secondary_comparison", "domains", "domain_families", "ablations", "derived_tuning", "operating",
            "modes", "lasso", "enet", "xgb", "decision", "subgroups", "privacy", "resources")
BRIEF_SEALED = ("Fall_Next_30D_Ind", "Next_Fall_Date_30D", "Days_To_Next_Fall_30D", "Next_Fall_Event_ID_30D", "Next_Fall_Confidence_30D", "Label_Reason_30D",
                "Fall_Next_180D_Ind", "Next_Fall_Date_180D", "Days_To_Next_Fall_180D", "Next_Fall_Event_ID_180D", "Next_Fall_Confidence_180D",
                "Label_Reason_180D", "Fall_Next_180D_HighConf_Ind")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


@dataclass(frozen=True)
class Phase5Config:
    path: str
    sha256: str
    raw: dict[str, Any]
    schema: V21Schema
    mode: str = "overnight"
    overrides: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    @property
    def budget(self) -> dict[str, Any]:
        return self.raw["modes"][self.mode]

    @property
    def outer_folds(self) -> int:
        return int(self.raw["cv"]["outer_folds"])

    @property
    def inner_folds(self) -> int:
        return int(self.raw["cv"]["inner_folds"])

    @property
    def primary_sensitivity(self) -> float:
        return float(self.raw["operating"]["primary_sensitivity"])

    @property
    def targets(self) -> list[float]:
        return [float(t) for t in self.raw["operating"]["targets"]]

    def with_mode(self, mode: str) -> Phase5Config:
        if mode not in MODES:
            raise ConfigError(f"--mode must be one of {MODES}")
        return Phase5Config(path=self.path, sha256=self.sha256, raw=self.raw, schema=self.schema, mode=mode, overrides=self.overrides)


def load_phase5_config(path: str | Path = DEFAULT_CONFIG, *, mode: str = "overnight", v21_schema: str | Path | None = None,
                       overrides: dict[str, Any] | None = None) -> Phase5Config:
    """``overrides`` (tests only) is merged into the raw settings and recorded; the real run never passes it."""
    p = resolve_path(path)
    raw = (yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("phase5") or {}
    if overrides:
        raw = _merge(raw, overrides)
    problems = [f"missing {k}" for k in REQUIRED if k not in raw]
    if problems:
        raise ConfigError(f"{p}: invalid Phase 5 settings: " + "; ".join(problems))
    missing = [c for c in BRIEF_SEALED if c not in raw["x_sealing"]["columns"]]
    if missing:
        problems.append(f"x_sealing.columns must contain every outcome column of the brief: missing {missing}")
    if not {"LABEL", "FORBIDDEN_LEAKAGE"} <= set(raw["x_sealing"]["contract_roles"]):
        problems.append("x_sealing.contract_roles must contain LABEL and FORBIDDEN_LEAKAGE")
    if not {"OUTCOME_OR_FUTURE_FORBIDDEN", "IDENTIFIER"} <= set(raw["x_sealing"].get("schema_classes") or []):
        problems.append("x_sealing.schema_classes must contain OUTCOME_OR_FUTURE_FORBIDDEN and IDENTIFIER")
    for pat in raw["x_sealing"]["name_patterns"]:
        try:
            re.compile(pat)
        except re.error as exc:
            problems.append(f"x_sealing.name_patterns: {pat!r} ({exc})")
    if tuple(raw["families"]) != FAMILIES:
        problems.append(f"families must be exactly {list(FAMILIES)} (pre-declared)")
    if raw["primary_family"] != PRIMARY_FAMILY:
        problems.append(f"primary_family must be {PRIMARY_FAMILY} (pre-declared)")
    if tuple(raw["sets"]) != SETS:
        problems.append(f"sets must be exactly {list(SETS)}")
    if list(raw["primary_comparison"]) != [SET_OLD, SET_ALL]:
        problems.append(f"primary_comparison must be [{SET_OLD}, {SET_ALL}]")
    if list(raw["secondary_comparison"]) != [SET_OLD, SET_SAFE]:
        problems.append(f"secondary_comparison must be [{SET_OLD}, {SET_SAFE}]")
    el = raw["eligibility"]
    if list(el.get("safe_classes") or []) != list(SAFE_CLASSES):
        problems.append(f"eligibility.safe_classes must be {list(SAFE_CLASSES)}")
    if not set(SAFE_CLASSES) <= set(el.get("all_new_classes") or []) or not set(el.get("all_new_classes") or []) <= set(CLASSES[:4]):
        problems.append("eligibility.all_new_classes must hold the SAFE classes (+ optionally UNCERTAIN_TIMING) only")
    if el.get("old_changed_definition_policy") not in ("keep_v21_definition", "exclude"):
        problems.append("eligibility.old_changed_definition_policy must be keep_v21_definition or exclude")
    if float(raw["operating"]["primary_sensitivity"]) != 0.70:
        problems.append("operating.primary_sensitivity must be 0.70 (the management requirement)")
    if 0.70 not in [float(t) for t in raw["operating"]["targets"]]:
        problems.append("operating.targets must contain 0.70")
    for m in MODES:
        if m not in raw["modes"]:
            problems.append(f"modes.{m} missing")
    if int(raw["outcome_contract"]["window_days"]) != 180:
        problems.append("outcome_contract.window_days must be 180")
    for f in ("outer_folds", "inner_folds"):
        if int(raw["cv"][f]) < 2:
            problems.append(f"cv.{f} must be >= 2")
    for fam in FAMILIES:
        if raw["derived_tuning"].get(fam) not in ("full", "reuse_reference"):
            problems.append(f"derived_tuning.{fam} must be full or reuse_reference")
    if not set(raw["domain_families"]) <= set(FAMILIES) or not set(raw["ablations"].get("families") or []) <= set(FAMILIES):
        problems.append("domain_families / ablations.families must be model families")
    if PRIMARY_FAMILY not in raw["ablations"].get("families", []):
        problems.append("ablations.families must contain the primary family")
    if problems:
        raise ConfigError(f"{p}: invalid Phase 5 settings: " + "; ".join(problems))
    schema = load_v21_schema(v21_schema or raw["v21_schema"])
    unknown = [d for d in raw["domains"] if d not in schema.domains]
    if unknown:
        raise ConfigError(f"{p}: domains {unknown} are not V21 schema domains")
    sha = _sha(p)
    if overrides:
        sha = hashlib.sha256((sha + "|overrides|" + repr(sorted(_flat(overrides)))).encode()).hexdigest()
    return Phase5Config(path=str(p), sha256=sha, raw=raw, schema=schema, mode=mode if mode in MODES else "overnight", overrides=dict(overrides or {}))


def _merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _flat(d: dict[str, Any], prefix: str = "") -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for k, v in d.items():
        if isinstance(v, dict):
            out += _flat(v, f"{prefix}{k}.")
        else:
            out.append((f"{prefix}{k}", repr(v)))
    return out
