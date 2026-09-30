"""Phase 3 configuration: analysis settings (``configs/meuhedet/phase3.yaml``) and the historical-recovery rules
(``configs/meuhedet/phase3_recovery.yaml``), validated against the contract and the data dictionary. The feature catalogue is the unchanged
Phase 2 catalogue (``falls_ml.phase2.config.load_catalogue``)."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from falls_ml.errors import ConfigError
from falls_ml.paths import resolve_path

DEFAULT_CONFIG = "configs/meuhedet/phase3.yaml"
DEFAULT_RECOVERY = "configs/meuhedet/phase3_recovery.yaml"
P3_BASE = "P3_BASE"
P3_ALL = "P3_ALL_RECOVERED"
REQUIRED = ("features", "recovery", "time_contract", "standards", "d00_config", "seed", "threads", "population", "eligibility", "feasibility", "screening", "cv", "lasso", "enet",
            "xgb", "selection", "validation_confirmation", "stability", "ablation", "explain", "metrics", "gates")
ROW_RECOVERABLE_KIND = "COMPLETE_RECORD_DATE"


def _sha(path: Path) -> str:
    """Line-ending-insensitive sha256 (a CRLF copy of an unchanged file keeps its hash)."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


@dataclass(frozen=True)
class Phase3Config:
    path: str
    sha256: str
    raw: dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)


def load_phase3_config(path: str | Path = DEFAULT_CONFIG) -> Phase3Config:
    p = resolve_path(path)
    raw = (yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("phase3") or {}
    problems = [f"missing {k}" for k in REQUIRED if k not in raw]
    if problems:
        raise ConfigError(f"{p}: invalid Phase 3 settings: " + "; ".join(problems))
    pop = raw["population"]
    if pop.get("primary") != "FULL_LABELED" or pop.get("bridge") != "D00_CLEAN":
        problems.append("population.primary must be FULL_LABELED (no future-dependent row removal) and population.bridge D00_CLEAN")
    sp = pop.get("extra_rows_split") or {}
    tr, va = float(sp.get("train", -1)), float(sp.get("validation", -1))
    if not (0 < tr < 1 and 0 <= va < 1 and tr + va < 1 and sp.get("salt")):
        problems.append("population.extra_rows_split needs a salt and train/validation shares with train + validation < 1")
    st = raw["standards"]
    if set(st) != {"PRIMARY_FULL", "SENSITIVITY_VERIFIED_ONLY", "SENSITIVITY_LOW_AVAILABILITY_RISK", "EXPLORATORY_UNRESOLVED"} or any(
            c in (st.get(k) or []) for k in st for c in ("NOT_RECOVERABLE_FUTURE_RECORDS", "NOT_RECOVERABLE_FORBIDDEN", "INELIGIBLE_DATA")) or \
            "UNRESOLVED" in (st.get("PRIMARY_FULL") or []) or "SAFE_ATTESTED" in (st.get("SENSITIVITY_VERIFIED_ONLY") or []):
        problems.append("standards: PRIMARY_FULL / SENSITIVITY_VERIFIED_ONLY / EXPLORATORY_UNRESOLVED; never future / forbidden / data-ineligible; UNRESOLVED "
                        "never primary; ATTESTED never in the verified-only sensitivity")
    s = raw["selection"]
    if s.get("discovery_reference") != "LASSO:P3_BASE":
        problems.append("selection.discovery_reference must be LASSO:P3_BASE")
    if float(s["principal_capacity"]) not in [float(x) for x in raw["metrics"]["capacities"]]:
        problems.append("selection.principal_capacity must be one of metrics.capacities")
    if min(int(raw["cv"][k]) for k in ("outer_folds", "inner_folds_linear", "inner_folds_xgb")) < 3:
        problems.append("cv folds must be >= 3")
    x = raw["xgb"]
    if float(x["fixed"].get("scale_pos_weight", 1.0)) != 1.0:
        problems.append("xgb.fixed.scale_pos_weight must stay 1.0 (no class weighting)")
    if set(x["stage0"]["sets"]) - {P3_BASE, P3_ALL, "P3_ALL_NO_FALL_RECENCY"} or x["stage1"]["set"] != P3_ALL:
        problems.append("xgb.stage0.sets must be within [P3_BASE, P3_ALL_RECOVERED, P3_ALL_NO_FALL_RECENCY] and xgb.stage1.set P3_ALL_RECOVERED")
    if bool((x.get("stage2") or {}).get("enabled")):
        problems.append("xgb.stage2 is not implemented in Phase 3 (Astra F-06 decision kept)")
    if int(x["stage1"]["n_trials"]) > int(x["max_trials_per_study"]):
        problems.append("xgb.stage1.n_trials exceeds max_trials_per_study")
    supported = {"sampler": "TPE", "multivariate": True, "direction": "minimize", "objective": "inner_cv_mean_logloss", "pruner": "none"}
    problems += [f"xgb.optuna.{k} must be {v!r}" for k, v in supported.items() if (x.get("optuna") or {}).get(k) != v]
    f = raw["feasibility"]
    if not f.get("priority_domains") or int(f["min_domain_known_informative_train_rows"]) < 1 or int(f["min_domain_known_informative_train_events"]) < 1:
        problems.append("feasibility needs priority_domains and positive adequacy minima")
    if problems:
        raise ConfigError(f"{p}: invalid Phase 3 settings: " + "; ".join(problems))
    return Phase3Config(path=str(p), sha256=_sha(p), raw=raw)


def resolve_threads(cfg: Phase3Config) -> dict[str, int]:
    t = cfg["threads"]
    n = max(1, min(int(t.get("max_auto", 4)), os.cpu_count() or 1))
    return {"blas": n if t.get("blas") == "auto" else int(t["blas"]), "xgboost": n if t.get("xgboost") == "auto" else int(t["xgboost"])}


@dataclass(frozen=True)
class RecoveryRules:
    name: str
    version: str
    path: str
    sha256: str
    source_semantics: dict[str, dict[str, str]]
    not_bounded: dict[str, str]
    absence_indicators: dict[str, float]
    upper_bound_columns: dict[str, str]
    gates: dict[str, float]
    remediation: dict[str, str]
    raw: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)


def load_recovery_rules(path: str | Path, *, contract: Any, dictionary: Any, d00: Any) -> RecoveryRules:
    p = resolve_path(path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    head = raw.get("phase3_recovery") or {}
    problems: list[str] = []
    if head.get("prediction_time") != "END_OF_INDEX_DAY" or head.get("predictor_record_rule") != "source_event_date <= Index_Date":
        problems.append("the recovery rules must use the Phase 3 time contract: END_OF_INDEX_DAY with source_event_date <= Index_Date")
    names = set(contract.names)
    sem = {str(k): dict(v) for k, v in (raw.get("source_semantics") or {}).items()}
    bad = [s for s in sem if s not in dictionary.sources]
    if bad:
        problems.append(f"source_semantics: unknown sources {bad}")
    nb = {str(k): str(v) for k, v in (raw.get("not_bounded_by_record_date") or {}).items()}
    ab = {str(k): float(v) for k, v in (raw.get("absence_indicators") or {}).items()}
    ub = {str(k): str(v) for k, v in (raw.get("pre_index_upper_bound_columns") or {}).items()}
    for group, cols in (("not_bounded_by_record_date", nb), ("absence_indicators", ab), ("pre_index_upper_bound_columns", ub)):
        missing = [c for c in cols if c not in names]
        if missing:
            problems.append(f"{group}: not contract columns {missing}")
    for c in ub:
        if c in names and ("since 2022" not in dictionary.columns[c]["meaning"].lower() or dictionary.columns[c]["status"] != "DOCUMENTED"):
            problems.append(f"pre_index_upper_bound_columns: {c} is not a DOCUMENTED 'since 2022' cumulative column")
        if c in names and d00.source_evidence.get(dictionary.source(c)) != ROW_RECOVERABLE_KIND:
            problems.append(f"pre_index_upper_bound_columns: {c} is not in a row-recoverable source")
    g = {str(k): float(v) for k, v in (raw.get("gates") or {}).items()}
    for k in ("max_unknown_share_population", "max_unknown_share_informative", "min_known_informative_train"):
        if k not in g:
            problems.append(f"gates.{k} missing")
    if g and not (0 <= g.get("max_unknown_share_population", -1) <= 0.05 and 0 <= g.get("max_unknown_share_informative", -1) <= 0.25):
        problems.append("recoverability gates out of the pre-declared plausible range (population share <= 5%, informative share <= 25%)")
    if problems:
        raise ConfigError(f"{p}: invalid Phase 3 recovery rules: " + "; ".join(problems))
    return RecoveryRules(name=str(head.get("name")), version=str(head.get("version")), path=str(p), sha256=_sha(p), source_semantics=sem, not_bounded=nb,
                         absence_indicators=ab, upper_bound_columns=ub, gates=g, remediation={str(k): str(v) for k, v in (raw.get("remediation") or {}).items()},
                         raw=raw)
