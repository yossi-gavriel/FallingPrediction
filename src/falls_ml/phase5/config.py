"""Phase 5 settings (``configs/meuhedet/phase5.yaml``) and the pre-declared V21 catalogue (``configs/meuhedet/phase5_v21_features.yaml``):
validated, hashed (line-ending insensitive) and recorded in the frozen plan."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from falls_ml.errors import ConfigError
from falls_ml.paths import resolve_path

DEFAULT_CONFIG = "configs/meuhedet/phase5.yaml"
FAMILIES = ("LASSO", "ENET", "XGB")
MODES = ("quick", "overnight")
SET_OLD, SET_NEW = "OLD", "OLD_PLUS_NEW_SAFE"
SET_VERIFIED, SET_LOWRISK = "OLD_PLUS_NEW_VERIFIED_ONLY", "OLD_PLUS_NEW_LOW_AVAILABILITY_RISK"
CLASSES = ("SAFE_VERIFIED", "SAFE_BOUNDED", "SAFE_ATTESTED", "INELIGIBLE_TIMING", "INELIGIBLE_SEMANTICS", "INELIGIBLE_DATA", "INELIGIBLE_LEAKAGE")
KINDS = ("binary", "count", "days", "ordinal", "categorical", "continuous")
TIMINGS = ("record_date", "days_value", "attested", "forbidden")
REQUIRED = ("index_date", "expected_definition_version", "seed", "phase3_config", "v21_catalogue", "x_sealing", "cohort", "outcome_contract",
            "eligibility", "cv", "families", "primary_sets", "sensitivity_sets", "domains", "ablations", "derived_tuning", "operating", "modes", "lasso",
            "enet", "xgb", "decision", "subgroups", "privacy", "resources")
BRIEF_SEALED = ("Fall_Next_30D_Ind", "Next_Fall_Date_30D", "Days_To_Next_Fall_30D", "Next_Fall_Event_ID_30D", "Next_Fall_Confidence_30D", "Label_Reason_30D",
                "Fall_Next_180D_Ind", "Next_Fall_Date_180D", "Days_To_Next_Fall_180D", "Next_Fall_Event_ID_180D", "Next_Fall_Confidence_180D",
                "Label_Reason_180D", "Fall_Next_180D_HighConf_Ind")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


@dataclass(frozen=True)
class V21Feature:
    column: str
    aliases: tuple[str, ...]
    domain: str
    kind: str
    timing: str
    missing: str
    text: str
    text_he: str
    levels: tuple[float, ...] | None = None
    record_date: tuple[str, ...] = ()
    availability: str | None = None
    process: bool = False


@dataclass(frozen=True)
class V21Catalogue:
    path: str
    sha256: str
    version: str
    domains: dict[str, dict[str, str]]
    availability: dict[str, dict[str, str]]
    features: tuple[V21Feature, ...]

    def risk_of(self, f: V21Feature) -> tuple[str, str]:
        if f.availability:
            return f.availability, "declared for the column"
        src = self.domains[f.domain].get("source", "")
        a = self.availability.get(src) or {}
        return str(a.get("risk", "UNKNOWN")), str(a.get("assumption", "no availability declaration"))


@dataclass(frozen=True)
class Phase5Config:
    path: str
    sha256: str
    raw: dict[str, Any]
    v21: V21Catalogue
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
        return Phase5Config(path=self.path, sha256=self.sha256, raw=self.raw, v21=self.v21, mode=mode, overrides=self.overrides)


def load_v21_catalogue(path: str | Path) -> V21Catalogue:
    p = resolve_path(path)
    raw = (yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("phase5_v21_features") or {}
    problems: list[str] = []
    domains = raw.get("domains") or {}
    feats: list[V21Feature] = []
    seen: set[str] = set()
    for i, f in enumerate(raw.get("features") or []):
        col = str(f.get("column", ""))
        if not col:
            problems.append(f"feature {i}: no column")
            continue
        names = [col, *[str(a) for a in f.get("aliases") or []]]
        for n in names:
            if n.lower() in seen:
                problems.append(f"{n}: declared twice")
            seen.add(n.lower())
        if f.get("domain") not in domains:
            problems.append(f"{col}: unknown domain {f.get('domain')!r}")
        if f.get("kind") not in KINDS:
            problems.append(f"{col}: kind must be one of {KINDS}")
        if f.get("timing") not in TIMINGS:
            problems.append(f"{col}: timing must be one of {TIMINGS}")
        if f.get("kind") in ("ordinal", "categorical") and not f.get("levels"):
            problems.append(f"{col}: {f.get('kind')} needs levels")
        feats.append(V21Feature(column=col, aliases=tuple(str(a) for a in f.get("aliases") or []), domain=str(f.get("domain")), kind=str(f.get("kind")),
                                timing=str(f.get("timing")), missing=str(f.get("missing", "no_event")), text=str(f.get("text", col)),
                                text_he=str(f.get("text_he", f.get("text", col))), levels=tuple(float(x) for x in f["levels"]) if f.get("levels") else None,
                                record_date=tuple(str(x) for x in (f.get("record_date") or [])), availability=f.get("availability"),
                                process=bool(f.get("process", False))))
    if problems:
        raise ConfigError(f"{p}: invalid V21 catalogue: " + "; ".join(problems))
    return V21Catalogue(path=str(p), sha256=_sha(p), version=str(raw.get("version", "")), domains=dict(domains),
                        availability=dict(raw.get("availability") or {}), features=tuple(feats))


def load_phase5_config(path: str | Path = DEFAULT_CONFIG, *, mode: str = "overnight", v21_catalogue: str | Path | None = None,
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
    for pat in raw["x_sealing"]["name_patterns"]:
        try:
            re.compile(pat)
        except re.error as exc:
            problems.append(f"x_sealing.name_patterns: {pat!r} ({exc})")
    if tuple(raw["families"]) != FAMILIES:
        problems.append(f"families must be exactly {list(FAMILIES)} (pre-declared)")
    if list(raw["primary_sets"]) != [SET_OLD, SET_NEW]:
        problems.append(f"primary_sets must be [{SET_OLD}, {SET_NEW}]")
    if list(raw["sensitivity_sets"]) != [SET_VERIFIED, SET_LOWRISK]:
        problems.append(f"sensitivity_sets must be [{SET_VERIFIED}, {SET_LOWRISK}]")
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
    if problems:
        raise ConfigError(f"{p}: invalid Phase 5 settings: " + "; ".join(problems))
    v21 = load_v21_catalogue(v21_catalogue or raw["v21_catalogue"])
    unknown = [d for d in raw["domains"] if d not in v21.domains]
    if unknown:
        raise ConfigError(f"{p}: domains {unknown} are not V21 catalogue domains")
    sha = _sha(p)
    if overrides:
        sha = hashlib.sha256((sha + "|overrides|" + repr(sorted(_flat(overrides)))).encode()).hexdigest()
    return Phase5Config(path=str(p), sha256=sha, raw=raw, v21=v21, mode=mode if mode in MODES else "overnight", overrides=dict(overrides or {}))


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
