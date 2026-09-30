"""The Phase 3 run context. It offers the interface the unchanged Phase 2 fitting helpers read (``run``, ``cfg``, ``espec``, ``fs()``,
``train_frame()``, ``y_train()``, ``outer_folds()``, ``threads``, ``protected``), plus the recovery products: the UNKNOWN masks, the cumulative
upper bounds, the feature -> source map and the fitting mask. VALIDATION rows are reachable only through :meth:`validation_frame`, which needs
the frozen selection."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from falls_ml.errors import LeakageError
from falls_ml.phase2 import durable as D
from falls_ml.phase2.featuresets import FeatureSets
from falls_ml.phase2.fitting import stratified_folds
from falls_ml.phase2.state import Run


@dataclass
class P3Ctx:
    run: Run
    src: Path
    ref_dir: Path
    pepper_file: Path | None
    cfg: Any
    rules: Any
    tc: Any
    cat: Any
    contract: Any
    mapping: Any
    dictionary: Any
    d00: Any
    espec: Any
    threads: dict[str, int]
    protected: list[Path]
    input_sha: str
    phase2_out: Path | None = None
    cache: dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------------- committed products
    def item_file(self, sid: str, name: str, item: str, rel: str) -> Path:
        return self.run.stage(sid, name).path(item) / rel

    def _pq(self, key: str, sid: str, name: str, item: str, rel: str) -> pd.DataFrame:
        if key not in self.cache:
            self.cache[key] = pd.read_parquet(self.item_file(sid, name, item, rel))
        return self.cache[key]

    def work(self) -> pd.DataFrame:
        return self._pq("work", "01", "cohort", "cohort", "trainval.parquet")

    def facts(self) -> dict[str, Any]:
        return self.run.stage("01", "cohort").result("cohort")

    @property
    def baseline(self) -> list[str]:
        return list(self.facts()["baseline_features"])

    def registry(self) -> pd.DataFrame:
        if "reg" not in self.cache:
            self.cache["reg"] = pd.read_csv(self.item_file("03", "recovery", "recovery", "FEATURE_RECOVERY.csv"))
        return self.cache["reg"]

    def feature_source(self) -> dict[str, str]:
        if "fsrc" not in self.cache:
            self.cache["fsrc"] = json.loads(self.item_file("03", "recovery", "recovery", "feature_source.json").read_text(encoding="utf-8"))
        return self.cache["fsrc"]

    def values(self) -> pd.DataFrame:
        return self._pq("values", "04", "reconstruct", "reconstruct", "values.parquet")

    def unknown(self) -> pd.DataFrame:
        return self._pq("unknown", "04", "reconstruct", "reconstruct", "unknown.parquet")

    def upper_bound(self) -> pd.DataFrame:
        return self._pq("ub", "04", "reconstruct", "reconstruct", "upper_bound.parquet")

    def feasibility(self) -> dict[str, Any]:
        return self.run.stage("05", "feasibility").result("feasibility")

    def go(self) -> bool:
        return bool(self.feasibility().get("decision") == "GO")

    def fs(self) -> FeatureSets:
        if "fs" not in self.cache:
            self.cache["fs"] = FeatureSets.from_dict(json.loads(self.item_file("06", "feature_sets", "feature_sets", "FEATURE_SETS.json").read_text(encoding="utf-8")))
        return self.cache["fs"]

    # ---------------------------------------------------------------- TRAIN
    def _both(self) -> pd.DataFrame:
        w, v = self.work(), self.values()
        if len(w) != len(v):
            raise LeakageError("reconstructed values are not aligned with the cohort rows")
        dup = [c for c in v.columns if c in w.columns]
        return pd.concat([w.drop(columns=dup), v], axis=1)

    def _train_mask(self) -> np.ndarray:
        return (self.work()["partition"] == "train").to_numpy()

    def train_frame(self) -> pd.DataFrame:
        if "train" not in self.cache:
            both = self._both()
            tr = both.loc[self._train_mask()].reset_index(drop=True)
            if (tr["partition"] != "train").any():
                raise LeakageError("non-TRAIN rows in the TRAIN frame")
            self.cache["train"] = tr
        return self.cache["train"]

    def y_train(self) -> np.ndarray:
        return self.train_frame()["y"].to_numpy(dtype=int)

    def unknown_train(self) -> pd.DataFrame:
        return self.unknown().loc[self._train_mask()].reset_index(drop=True)

    def ub_train(self) -> pd.DataFrame:
        return self.upper_bound().loc[self._train_mask()].reset_index(drop=True)

    def outer_folds(self) -> np.ndarray:
        if "outer" not in self.cache:
            self.cache["outer"] = stratified_folds(self.y_train(), int(self.cfg["cv"]["outer_folds"]), self.run.seed, "phase3_outer_folds")
        return self.cache["outer"]

    def scope_features(self) -> list[str]:
        """Every feature any fitted set reads (new + baseline): the fitting mask excludes rows with an UNKNOWN cell in any of them."""
        fs = self.fs()
        out: set[str] = set()
        for s in fs.sets.values():
            out |= set(s["features"]) | set(s["baseline"])
        return sorted(out)

    def fit_mask_train(self) -> np.ndarray:
        """TRAIN rows whose every scope feature is KNOWN: the only rows any model is FITTED on (evaluation always uses every row)."""
        if "fitmask" not in self.cache:
            u = self.unknown_train()
            cols = [c for c in self.scope_features() if c in u.columns]
            self.cache["fitmask"] = ~u[cols].to_numpy(dtype=bool).any(axis=1) if cols else np.ones(len(u), dtype=bool)
        return self.cache["fitmask"]

    def bridge_train(self) -> np.ndarray:
        """TRAIN rows of the Phase 1-2 reference cohort D00_CLEAN (bookkeeping only; never a feature)."""
        return ~self.train_frame()["d00_row"].to_numpy(dtype=bool)

    # ---------------------------------------------------------------- the validation seal
    def validation_frame(self, selection_record: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        sel_path = self.run.out / "SELECTION_FROZEN.json"
        if not sel_path.is_file() or D.sha256_file(sel_path) != selection_record.get("sha256"):
            raise LeakageError("VALIDATION is sealed: SELECTION_FROZEN.json is missing or differs from the frozen selection")
        records, _ = D.read_jsonl(self.run.out / "validation_evaluation_registry.jsonl")
        if [r for r in records if r.get("selection_sha256") != selection_record.get("sha256")]:
            raise LeakageError("VALIDATION was already opened for a different frozen selection (validation registry)")
        m = (self.work()["partition"] == "validation").to_numpy()
        return (self._both().loc[m].reset_index(drop=True), self.unknown().loc[m].reset_index(drop=True),
                self.upper_bound().loc[m].reset_index(drop=True))
