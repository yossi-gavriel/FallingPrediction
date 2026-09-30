"""Phase 2 configuration: analysis settings (``configs/meuhedet/phase2.yaml``) and the engineered-feature catalogue
(``configs/meuhedet/phase2_features.yaml``), validated against the contract, the data dictionary and the eFalls mapping."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from falls_ml.data.meuhedet_wide import PREDICTOR_ROLES, WideContract, WideMapping
from falls_ml.errors import ConfigError
from falls_ml.paths import resolve_path

DEFAULT_CONFIG = "configs/meuhedet/phase2.yaml"
KINDS = ("binary", "count", "continuous", "days", "ordinal")
OPS = ("copy", "flag_lt", "flag_ge", "any_positive", "days_since", "present")
LINEAR = ("log1p", "none", "thermometer", "fall_recency_bands")
MISSING = ("not_assessed", "no_event", "source_absent", "unexpected")
CLASSES = ("EFALLS_HIGH_CONFIDENCE", "EFALLS_APPROXIMATE", "MEUHEDET_NATIVE")
OTHER_NATIVE = "OTHER_NATIVE"
FALL_RECENCY_BANDS = (90, 180, 365)   # days since the last fall: <=90, 91-180, 181-365, >365 (reference: no prior fall)


def _sha(path: Path) -> str:
    """Line-ending-insensitive sha256 (a CRLF copy of an unchanged file keeps its hash, so the frozen configuration still matches)."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


@dataclass(frozen=True)
class FeatureDef:
    name: str
    domain: str
    op: str
    inputs: tuple[str, ...]
    kind: str
    linear: str
    missing: str
    mapping_class: str
    text: str
    order: int
    efalls_concept: str | None = None
    threshold: float | None = None
    levels: tuple[float, ...] | None = None
    form_indicator: bool = False
    form: str | None = None            # assessment-conditional source (data-dictionary source) this feature belongs to

    @property
    def is_new(self) -> bool:
        return True


@dataclass(frozen=True)
class Catalogue:
    name: str
    version: str
    path: str
    sha256: str
    domains: dict[str, dict[str, str]]
    waterfall: tuple[str, ...]
    forms: dict[str, dict[str, str]]
    quarantine: dict[str, str]
    represented_by: dict[str, str]
    features: tuple[FeatureDef, ...]
    raw: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    def get(self, name: str) -> FeatureDef:
        for f in self.features:
            if f.name == name:
                return f
        raise KeyError(name)

    @property
    def names(self) -> list[str]:
        return [f.name for f in self.features]

    def by_domain(self, domain: str) -> list[FeatureDef]:
        return [f for f in self.features if f.domain == domain]

    def indicator_of(self, form: str) -> str:
        return self.forms[form]["indicator"]

    def reading(self, column: str) -> list[str]:
        return [f.name for f in self.features if column in f.inputs]


def load_catalogue(path: str | Path, contract: WideContract, dictionary: Any, mapping: WideMapping) -> Catalogue:
    p = resolve_path(path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    meta = raw.get("phase2_features") or {}
    problems: list[str] = []
    names = set(contract.names)
    domains = dict(raw.get("domains") or {})
    if not domains:
        problems.append("no domains declared")
    waterfall = tuple(raw.get("waterfall") or ())
    for w in waterfall:
        if w not in domains and w != OTHER_NATIVE:
            problems.append(f"waterfall step {w!r} is not a domain")
    if waterfall and waterfall[-1] != OTHER_NATIVE:
        problems.append("the last waterfall step must be OTHER_NATIVE (every remaining domain)")
    forms = {str(k): dict(v) for k, v in (raw.get("forms") or {}).items()}
    for src, fm in forms.items():
        if src not in dictionary.sources:
            problems.append(f"form {src}: not a data-dictionary source")
            continue
        if dictionary.sources[src].get("record_date") != fm.get("date"):
            problems.append(f"form {src}: date {fm.get('date')!r} is not the source's record date {dictionary.sources[src].get('record_date')!r}")
    quarantine = {str(k): str(v) for k, v in (raw.get("quarantine") or {}).items()}
    represented = {str(k): str(v) for k, v in (raw.get("represented_by") or {}).items()}
    for col in [*quarantine, *represented]:
        if col not in names:
            problems.append(f"{col}: not a contract column")
    feats: list[FeatureDef] = []
    seen: set[str] = set()
    for i, e in enumerate(raw.get("features") or []):
        n = str(e.get("name", ""))
        where = f"feature {n or i}"
        if not n or n in seen:
            problems.append(f"{where}: missing or duplicated name")
        seen.add(n)
        inputs = tuple(str(c) for c in e.get("inputs") or ())
        for key, allowed in (("domain", tuple(domains)), ("op", OPS), ("kind", KINDS), ("linear", LINEAR), ("missing", MISSING), ("mapping_class", CLASSES)):
            if e.get(key) not in allowed:
                problems.append(f"{where}: {key} {e.get(key)!r} not in {list(allowed)}")
        for c in inputs:
            if c not in names:
                problems.append(f"{where}: input {c} is not a contract column")
                continue
            cc = contract.get(c)
            if cc.role not in PREDICTOR_ROLES:
                problems.append(f"{where}: input {c} has contract role {cc.role} (only predictor-role columns may feed a feature)")
            if c in quarantine:
                problems.append(f"{where}: input {c} is quarantined ({quarantine[c]})")
        op = e.get("op")
        if op in ("copy", "flag_lt", "flag_ge", "days_since", "present") and len(inputs) != 1:
            problems.append(f"{where}: op {op} needs exactly one input")
        if op == "any_positive" and len(inputs) < 2:
            problems.append(f"{where}: any_positive needs >= 2 inputs")
        if op in ("days_since", "present") and inputs and inputs[0] in names and not contract.get(inputs[0]).is_date:
            problems.append(f"{where}: {op} needs a date input")
        if op in ("flag_lt", "flag_ge") and e.get("threshold") is None:
            problems.append(f"{where}: {op} needs a threshold")
        levels = tuple(float(x) for x in e["levels"]) if e.get("levels") is not None else None
        if e.get("linear") == "thermometer" and (e.get("kind") != "ordinal" or not levels or len(levels) < 2):
            problems.append(f"{where}: thermometer coding needs kind ordinal and >= 2 declared levels")
        srcs = {dictionary.source(c) for c in inputs if c in names}
        form_srcs = sorted(s for s in srcs if s in forms)
        if len(form_srcs) > 1:
            problems.append(f"{where}: reads several assessment forms {form_srcs}")
        form = form_srcs[0] if form_srcs else None
        if e.get("missing") == "not_assessed" and form is None:
            problems.append(f"{where}: missing 'not_assessed' but no input comes from an assessment form")
        is_ind = bool(e.get("form_indicator", False))
        if is_ind:
            if op != "present" or form is None or forms[form].get("indicator") != n:
                problems.append(f"{where}: a form indicator must be the 'present' feature named in forms.{form}")
        feats.append(FeatureDef(name=n, domain=str(e.get("domain")), op=str(op), inputs=inputs, kind=str(e.get("kind")), linear=str(e.get("linear")),
                                missing=str(e.get("missing")), mapping_class=str(e.get("mapping_class")), text=str(e.get("text", "")), order=i,
                                efalls_concept=e.get("efalls_concept"), threshold=None if e.get("threshold") is None else float(e["threshold"]),
                                levels=levels, form_indicator=is_ind, form=form))
    for src, fm in forms.items():
        if fm.get("indicator") not in seen:
            problems.append(f"form {src}: indicator {fm.get('indicator')!r} is not a catalogue feature")
    # every predictor-allowed value column needs an explicit disposition (feature input, BASELINE_15 source, quarantine or represented_by)
    baseline_sources = {c for f in mapping.features if f.include_in_baseline for c in f.source_columns}
    covered = {c for f in feats for c in f.inputs} | baseline_sources | set(quarantine) | set(represented)
    uncovered = [c.name for c in contract.columns if c.predictor_allowed and not c.is_date and c.name not in covered]
    if uncovered:
        problems.append(f"predictor-allowed columns without a Phase 2 disposition: {uncovered}")
    if problems:
        raise ConfigError(f"{p}: invalid Phase 2 feature catalogue ({len(problems)} problems): " + "; ".join(problems))
    return Catalogue(name=str(meta.get("name")), version=str(meta.get("version")), path=str(p), sha256=_sha(p), domains=domains, waterfall=waterfall,
                     forms=forms, quarantine=quarantine, represented_by=represented, features=tuple(feats), raw=raw)


@dataclass(frozen=True)
class Phase2Config:
    path: str
    sha256: str
    raw: dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)


REQUIRED = ("features", "d00_config", "seed", "threads", "eligibility", "screening", "cv", "lasso", "enet", "xgb", "selection", "validation_confirmation",
            "stability", "ablation", "explain", "consensus", "metrics", "gates")


def load_phase2_config(path: str | Path = DEFAULT_CONFIG) -> Phase2Config:
    p = resolve_path(path)
    raw = (yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("phase2") or {}
    missing = [k for k in REQUIRED if k not in raw]
    if missing:
        raise ConfigError(f"{p}: phase2 settings missing {missing}")
    s = raw["selection"]
    if not 0 < float(s["principal_capacity"]) < 1:
        raise ConfigError("selection.principal_capacity must be in (0, 1)")
    if float(s["principal_capacity"]) not in [float(x) for x in raw["metrics"]["capacities"]]:
        raise ConfigError("selection.principal_capacity must be one of metrics.capacities")
    if int(raw["cv"]["outer_folds"]) < 3 or int(raw["cv"]["inner_folds_linear"]) < 3 or int(raw["cv"]["inner_folds_xgb"]) < 3:
        raise ConfigError("cv folds must be >= 3")
    x = raw["xgb"]
    if int(x["stage1"]["n_trials"]) + int(x["stage2"]["n_trials"]) > int(x["max_trials_per_study"]):
        raise ConfigError("xgb stage1 + stage2 trials exceed max_trials_per_study")
    problems = []
    cats = (raw["eligibility"].get("categories") or {})
    if set(cats) != {"HISTORICAL_BASELINE_15", "SAFE_DISCOVERY", "EXPLORATORY_UNRESOLVED_SENSITIVITY"}:
        problems.append("eligibility.categories must be exactly HISTORICAL_BASELINE_15, SAFE_DISCOVERY, EXPLORATORY_UNRESOLVED_SENSITIVITY")
    if list(cats.get("SAFE_DISCOVERY") or []) != ["SAFE"]:
        problems.append("eligibility.categories.SAFE_DISCOVERY must be [SAFE] (only proven-SAFE provenance may enter the SAFE discovery)")
    if not set(cats.get("EXPLORATORY_UNRESOLVED_SENSITIVITY") or []) <= {"SAFE", "UNRESOLVED"}:
        problems.append("eligibility.categories.EXPLORATORY_UNRESOLVED_SENSITIVITY may hold SAFE and UNRESOLVED only (never UNSAFE)")
    if s.get("candidates") != "SAFE_DISCOVERY":
        problems.append("selection.candidates must be SAFE_DISCOVERY")
    if s.get("discovery_reference") != "LASSO:SAFE_BASE" or s.get("benchmark") != "LASSO:BASELINE_15":
        problems.append("selection.discovery_reference must be LASSO:SAFE_BASE and selection.benchmark LASSO:BASELINE_15")
    o = x.get("optuna") or {}
    supported = {"sampler": "TPE", "multivariate": True, "direction": "minimize", "objective": "inner_cv_mean_logloss", "pruner": "none"}
    problems += [f"xgb.optuna.{k} must be {v!r} (the implemented study)" for k, v in supported.items() if o.get(k) != v]
    if float(x["fixed"].get("scale_pos_weight", 1.0)) != 1.0:
        problems.append("xgb.fixed.scale_pos_weight must stay 1.0 (no class weighting; probabilities are evaluated for calibration)")
    if problems:
        raise ConfigError(f"{p}: invalid Phase 2 settings: " + "; ".join(problems))
    return Phase2Config(path=str(p), sha256=_sha(p), raw=raw)


def resolve_threads(cfg: Phase2Config) -> dict[str, int]:
    t = cfg["threads"]
    cap = int(t.get("max_auto", 4))
    n = max(1, min(cap, os.cpu_count() or 1))
    return {"blas": n if t.get("blas") == "auto" else int(t["blas"]), "xgboost": n if t.get("xgboost") == "auto" else int(t["xgboost"])}
