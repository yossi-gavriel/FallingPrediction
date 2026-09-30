"""Discovery and loading of completed runs (the artifacts every experiment already writes). Read-only: nothing here writes into a run."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from falls_ml.errors import DatasetValidationError

COMPLETE = "RUN_COMPLETE.json"
DEFAULT_LABELS = "configs/meuhedet/feature_labels_he.yaml"


def analysis_of(name: str) -> tuple[str, str]:
    """(analysis key, display name) from an experiment name written by meuhedet-explore / meuhedet-sensitivity."""
    n = name.upper()
    d00 = re.fullmatch(r"MEUHEDET_EFALLS_D00_(.+)_180D_SENSITIVITY", n)
    if d00:   # meuhedet-d00 cells (falls_ml.d00.plan.run_label)
        return d00.group(1), f"D-00 {d00.group(1)}"
    if "PATIENT_DISJOINT" in n:
        base = "STRICT" if "STRICT" in n else "EXTENDED"
        return f"{base}_PATIENT_DISJOINT", f"{base} (patient-disjoint sensitivity)"
    if "SAFE_ALL_ROWS" in n:
        return "SAFE_ALL_ROWS", "SAFE-ALL-ROWS"
    m = re.search(r"ABLATION_(NO_.+?)(?:_180D|$)", n)
    if m:
        parts = [p for p in m.group(1).split("_NO_") if p]
        what = ", ".join(p.replace("NO_", "").lower() for p in parts)
        return f"ABLATION_{m.group(1)}", f"EXTENDED without {what}"
    if "STRICT" in n:
        return "STRICT", "STRICT"
    if "EXTENDED" in n:
        return "EXTENDED", "EXTENDED"
    return name, name


@dataclass
class Run:
    run_dir: Path
    metrics: dict[str, Any]
    results_dir: Path
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return str((self.metrics.get("experiment") or {}).get("name") or self.run_dir.name)

    @property
    def key(self) -> str:
        return analysis_of(self.name)[0]

    @property
    def display(self) -> str:
        return analysis_of(self.name)[1]

    @property
    def test_sha(self) -> str:
        return str((self.metrics.get("split") or {}).get("test_rows_sha256") or "")

    @property
    def served(self) -> str:
        return str(self.metrics.get("served_variant") or "uncalibrated")

    @property
    def synthetic(self) -> bool:
        return bool(self.metrics.get("synthetic_fixture")) or bool((self.dataset_manifest.get("source") or "") == "synthetic_fixture")

    def csv(self, name: str) -> pd.DataFrame:
        p = self.run_dir / name
        return pd.read_csv(p) if p.exists() else pd.DataFrame()

    @cached_property
    def config(self) -> dict[str, Any]:
        p = self.run_dir / "config.yaml"
        return yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else {}

    @cached_property
    def features(self) -> list[str]:
        return list((self.config.get("preprocessing") or {}).get("features") or [])

    @cached_property
    def dataset_manifest(self) -> dict[str, Any]:
        p = self.run_dir / "dataset_manifest.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    @cached_property
    def build(self) -> dict[str, Any]:
        return (self.dataset_manifest.get("audit") or {}).get("meuhedet_build") or {}

    def predictions(self, split: str) -> pd.DataFrame:
        p = self.run_dir / f"predictions_{split}.parquet"
        return pd.read_parquet(p) if p.exists() else pd.DataFrame()

    def perf(self, split: str = "test", variant: str | None = None) -> dict[str, Any]:
        return ((self.metrics.get("performance") or {}).get(split) or {}).get(variant or self.served) or {}

    def dataset_dir(self, search: list[Path]) -> Path | None:
        """The canonical dataset the run trained on: the recorded path, else a dataset folder with the same data sha256."""
        used = self.dataset_manifest.get("dataset_path_used")
        want = self.dataset_manifest.get("data_sha256")
        if used and (Path(used) / "manifest.json").exists():
            return Path(used)
        for root in search:
            for m in root.glob("**/datasets/*/manifest.json"):
                try:
                    if json.loads(m.read_text(encoding="utf-8")).get("data_sha256") == want:
                        return m.parent
                except (OSError, json.JSONDecodeError):
                    continue
        return None


def discover_runs(results_dirs: list[Path]) -> list[Run]:
    """Every completed run (RUN_COMPLETE.json + metrics.json) under the given results folders, de-duplicated by run id."""
    runs: dict[str, Run] = {}
    for root in results_dirs:
        if not root.is_dir():
            raise DatasetValidationError(f"results folder not found: {root}")
        for marker in sorted(root.glob("**/" + COMPLETE)):
            d = marker.parent
            if not (d / "metrics.json").exists():
                continue
            m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
            rid = str(m.get("run_id") or d.name)
            if rid not in runs:
                runs[rid] = Run(run_dir=d, metrics=m, results_dir=root)
    if not runs:
        raise DatasetValidationError("no completed run (RUN_COMPLETE.json + metrics.json) found", [str(p) for p in results_dirs])
    order = {"STRICT": 0, "EXTENDED": 1, "SAFE_ALL_ROWS": 3}
    return sorted(runs.values(), key=lambda r: (order.get(r.key, 2 if r.key.startswith("ABLATION") else 4), r.key))


def load_labels(path: str | Path = DEFAULT_LABELS) -> dict[str, Any]:
    from falls_ml.paths import resolve_path

    return yaml.safe_load(resolve_path(path).read_text(encoding="utf-8"))


def feature_label(labels: dict[str, Any], name: str, lang: str) -> str:
    """Label of a canonical feature or a design column (``age_years__fp_p1`` -> age; ``sex__female`` -> sex: female)."""
    feats = labels.get("features") or {}
    if "=" in name and "__" not in name:
        raw, _, level = name.partition("=")
        base = feats.get(raw, {}).get(lang) or raw
        lv = (labels.get("levels") or {}).get(level, {}).get(lang, level)
        return f"{base}: {lv}"
    raw, _, level = name.partition("__")
    base = feats.get(raw, {}).get(lang) or raw
    if level and not level.startswith("fp_"):
        lv = (labels.get("levels") or {}).get(level, {}).get(lang, level)
        return f"{base}: {lv}"
    if level.startswith("fp_") and level != "fp_p1":
        return f"{base} ({level.replace('fp_', 'FP ')})"
    return base
