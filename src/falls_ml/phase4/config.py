"""Phase 4 settings (``configs/meuhedet/phase4.yaml``): validated, hashed (line-ending insensitive) and recorded in every frozen manifest."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from falls_ml.errors import ConfigError
from falls_ml.paths import resolve_path

DEFAULT_CONFIG = "configs/meuhedet/phase4.yaml"
REQUIRED = ("index_date", "development_index_date", "seed", "models", "phase3_stage_lasso", "sealed_columns", "sealed_contract_roles", "sealed_name_patterns",
            "preflight", "outcome_contract", "metrics", "comparison")
# the user's explicit list (Phase 4 brief §3): never a predictor, never loaded before the frozen hashes are verified
BRIEF_SEALED = ("Fall_Next_30D_Ind", "Next_Fall_Date_30D", "Days_To_Next_Fall_30D", "Next_Fall_Event_ID_30D", "Next_Fall_Confidence_30D", "Label_Reason_30D",
                "Fall_Next_180D_Ind", "Next_Fall_Date_180D", "Days_To_Next_Fall_180D", "Next_Fall_Event_ID_180D", "Next_Fall_Confidence_180D",
                "Label_Reason_180D", "Fall_Next_180D_HighConf_Ind")


@dataclass(frozen=True)
class Phase4Config:
    path: str
    sha256: str
    raw: dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    @property
    def models(self) -> list[str]:
        return [self.raw["models"]["primary"], *self.raw["models"]["secondary"]]

    @property
    def primary(self) -> str:
        return str(self.raw["models"]["primary"])


def load_phase4_config(path: str | Path = DEFAULT_CONFIG) -> Phase4Config:
    p = resolve_path(path)
    raw = (yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("phase4") or {}
    problems = [f"missing {k}" for k in REQUIRED if k not in raw]
    if problems:
        raise ConfigError(f"{p}: invalid Phase 4 settings: " + "; ".join(problems))
    if raw["models"].get("primary") != "LASSO:P3_BASE":
        problems.append("models.primary must be LASSO:P3_BASE (declared before the 2026 data were examined)")
    for m in [raw["models"]["primary"], *raw["models"].get("secondary", [])]:
        if not str(m).startswith("LASSO:"):
            problems.append(f"{m}: only frozen Phase 3 LASSO configurations are validated")
    missing = [c for c in BRIEF_SEALED if c not in raw["sealed_columns"]]
    if missing:
        problems.append(f"sealed_columns must contain every outcome column of the brief: missing {missing}")
    if not {"LABEL", "FORBIDDEN_LEAKAGE"} <= set(raw["sealed_contract_roles"]):
        problems.append("sealed_contract_roles must contain LABEL and FORBIDDEN_LEAKAGE")
    for pat in raw["sealed_name_patterns"]:
        try:
            re.compile(pat)
        except re.error as exc:
            problems.append(f"sealed_name_patterns: {pat!r} ({exc})")
    m = raw["metrics"]
    if float(m["principal_capacity"]) not in [float(x) for x in m["capacities"]]:
        problems.append("metrics.principal_capacity must be one of metrics.capacities")
    if int(m["bootstrap_n"]) < 200:
        problems.append("metrics.bootstrap_n must be >= 200")
    if int(raw["outcome_contract"]["window_days"]) != 180:
        problems.append("outcome_contract.window_days must be 180 (the frozen Phase 3 outcome)")
    if problems:
        raise ConfigError(f"{p}: invalid Phase 4 settings: " + "; ".join(problems))
    sha = hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    return Phase4Config(path=str(p), sha256=sha, raw=raw)
