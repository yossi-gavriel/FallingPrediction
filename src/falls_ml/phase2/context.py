"""The run context shared by the stages: loaded configurations, and lazy access to committed stage products. VALIDATION rows are only
reachable through :meth:`Ctx.validation_frame`, which needs the frozen selection (the validation seal, REVIEW_DECISIONS A-01)."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from falls_ml.errors import LeakageError
from falls_ml.phase2 import durable as D
from falls_ml.phase2.config import Catalogue, Phase2Config
from falls_ml.phase2.featuresets import FeatureSets
from falls_ml.phase2.fitting import stratified_folds
from falls_ml.phase2.state import Phase2Stop, Run


def import_xgboost() -> Any:
    """Import XGBoost; on Windows retry once with scikit-learn's bundled OpenMP runtime (vcomp140.dll) on the DLL search path."""
    try:
        import xgboost

        return xgboost
    except Exception as first:   # noqa: BLE001 - an import failure of a compiled library can be any exception type
        if sys.platform != "win32":
            raise Phase2Stop("DEPENDENCY_MISSING", f"XGBoost cannot be imported: {first}",
                             ["install the Phase 2 dependencies (requirements-phase2.lock); see docs/meuhedet/WORK_PC_RUNBOOK.md"]) from None
        import os

        import sklearn

        libs = Path(sklearn.__file__).parent / ".libs"
        if libs.is_dir():
            os.add_dll_directory(str(libs))
        for m in [m for m in sys.modules if m == "xgboost" or m.startswith("xgboost.")]:
            del sys.modules[m]
        try:
            import xgboost

            return xgboost
        except Exception as second:   # noqa: BLE001
            raise Phase2Stop("DEPENDENCY_MISSING", f"XGBoost cannot be imported (also with scikit-learn's OpenMP runtime): {second}",
                             ["install the Microsoft Visual C++ Redistributable (x64) or ask IT", "then rerun with --resume"]) from None


@dataclass
class Ctx:
    run: Run
    src: Path
    ref_dir: Path
    pepper_file: Path | None
    cfg: Phase2Config
    cat: Catalogue
    contract: Any
    mapping: Any
    dictionary: Any
    d00: Any
    espec: Any
    threads: dict[str, int]
    protected: list[Path]
    input_sha: str
    cache: dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------------- committed products (read back from items)
    def item_file(self, sid: str, name: str, item: str, rel: str) -> Path:
        return self.run.stage(sid, name).path(item) / rel

    def work(self) -> pd.DataFrame:
        if "work" not in self.cache:
            self.cache["work"] = pd.read_parquet(self.item_file("01", "cohort", "cohort", "trainval.parquet"))
        return self.cache["work"]

    def values(self) -> pd.DataFrame:
        if "values" not in self.cache:
            self.cache["values"] = pd.read_parquet(self.item_file("03", "engineer", "engineer", "values.parquet"))
        return self.cache["values"]

    def fs(self) -> FeatureSets:
        if "fs" not in self.cache:
            self.cache["fs"] = FeatureSets.from_dict(json.loads(self.item_file("05", "feature_sets", "feature_sets", "FEATURE_SETS.json").read_text(encoding="utf-8")))
        return self.cache["fs"]

    def facts(self) -> dict[str, Any]:
        return self.run.stage("01", "cohort").result("cohort")

    @property
    def baseline(self) -> list[str]:
        return list(self.facts()["baseline_features"])

    def train_frame(self) -> pd.DataFrame:
        if "train" not in self.cache:
            w, v = self.work(), self.values()
            if len(w) != len(v):
                raise LeakageError("engineered values are not aligned with the cohort rows")
            both = pd.concat([w, v], axis=1)
            tr = both.loc[both["partition"] == "train"].reset_index(drop=True)
            if (tr["partition"] != "train").any():
                raise LeakageError("non-TRAIN rows in the TRAIN frame")
            self.cache["train"] = tr
        return self.cache["train"]

    def y_train(self) -> np.ndarray:
        return self.train_frame()["y"].to_numpy(dtype=int)

    def outer_folds(self) -> np.ndarray:
        if "outer" not in self.cache:
            self.cache["outer"] = stratified_folds(self.y_train(), int(self.cfg["cv"]["outer_folds"]), self.run.seed, "phase2_outer_folds")
        return self.cache["outer"]

    # ---------------------------------------------------------------- the validation seal
    def validation_frame(self, selection_record: dict[str, Any]) -> pd.DataFrame:
        sel_path = self.run.out / "SELECTION_FROZEN.json"
        if not sel_path.is_file() or D.sha256_file(sel_path) != selection_record.get("sha256"):
            raise LeakageError("VALIDATION is sealed: SELECTION_FROZEN.json is missing or differs from the frozen selection")
        registry = self.run.out / "validation_evaluation_registry.jsonl"
        records, _ = D.read_jsonl(registry)
        other = [r for r in records if r.get("selection_sha256") != selection_record.get("sha256")]
        if other:   # VALIDATION may be re-read after a crash only for the SAME frozen selection; never for a changed one
            raise LeakageError("VALIDATION was already opened for a different frozen selection (validation registry)")
        w, v = self.work(), self.values()
        both = pd.concat([w, v], axis=1)
        return both.loc[both["partition"] == "validation"].reset_index(drop=True)
