"""The pre-specified D-00 analysis plan: the sensitivity-matrix cells, the ablations and the future cell, resolved against the feature sets
the dependency graph derived. Identical specifications (same cohort, same predictors) share ONE run; cells identical to a completed reference
run reuse it (never retrained). The plan is frozen - written with its sha256 - before any model is fitted or any prediction is read, and
nothing in it depends on a performance result."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

from falls_ml.d00.dependency import D00Config, Graph
from falls_ml.data.meuhedet_wide import WideMapping
from falls_ml.errors import ConfigError

COHORT_SUFFIX = {"FULL_LABELED": "FULL", "D00_CLEAN": "D00_CLEAN"}
REF_EXTENDED, REF_STRICT = "FULL_EXTENDED_D00_CLEAN", "REFERENCE_STRICT_D00_CLEAN"
FINAL = "FINAL_DWH_FIXED"


def run_label(cell: str) -> str:
    """Experiment name of a new D-00 run (parsed back by falls_ml.modelreport.artifacts.analysis_of)."""
    return f"MEUHEDET_EFALLS_D00_{cell}_180D_SENSITIVITY"


@dataclass
class Cell:
    cell: str                     # final name (a STRICT set the standard does not fully admit is renamed ..._SUBSET_...)
    planned: str                  # the name in the pre-specified plan
    kind: str                     # reference | matrix | ablation | future
    cohort: str                   # FULL_LABELED | D00_CLEAN
    feature_set: str              # derived set name, FULL_EXTENDED, or "EXTENDED minus [...]"
    features: list[str]
    standard: str | None = None   # evidence standard of a derived set (PROVEN_SAFE primary, SAFE_OR_UNRESOLVED secondary)
    question: str | None = None
    text: str = ""
    source: str = "run"           # run | reference:extended | reference:strict | alias | not_run
    alias_of: str | None = None
    label: str | None = None      # experiment name of the run that serves this cell
    not_run_reason: str = ""
    removed: list[str] = field(default_factory=list)

    @property
    def signature(self) -> tuple[str, tuple[str, ...]]:
        return self.cohort, tuple(sorted(self.features))


def build_plan(graph: Graph, config: D00Config, mapping: WideMapping, reference_features: dict[str, list[str]], *, secondary: bool = True) -> list[Cell]:
    sets = mapping.feature_sets()
    if sorted(reference_features["extended"]) != sorted(sets["extended"]) or sorted(reference_features["strict"]) != sorted(sets["strict"]):
        raise ConfigError("the reference run's STRICT / EXTENDED predictors differ from the current mapping's sets (the matrix would not be comparable)")
    order = [f for f in sets["extended"]]
    cells: list[Cell] = [
        Cell(cell=REF_EXTENDED, planned=REF_EXTENDED, kind="reference", cohort="D00_CLEAN", feature_set="FULL_EXTENDED", features=list(sets["extended"]),
             text="current reference: the completed EXTENDED run on the D-00-clean cohort (reused, never retrained)", source="reference:extended"),
        Cell(cell=REF_STRICT, planned=REF_STRICT, kind="reference", cohort="D00_CLEAN", feature_set="STRICT", features=list(sets["strict"]),
             text="the completed STRICT run on the D-00-clean cohort (reused, never retrained)", source="reference:strict")]
    stds = [config.primary_standard] + ([k for k in config.standards if k != config.primary_standard] if secondary else [])
    matrix = list(config.plan.get("matrix") or [])
    for std in stds:
        fs = graph.feature_sets[std]
        for m in matrix:
            if m["feature_set"] == "FULL_EXTENDED":
                continue   # the reference cell above
            s = fs[m["feature_set"]]
            name = f"{s['name']}_{COHORT_SUFFIX[m['cohort']]}"
            cells.append(Cell(cell=name, planned=m["cell"] if std == config.primary_standard else name, kind="matrix", cohort=m["cohort"],
                              feature_set=s["name"], features=[f for f in order if f in set(s["features"])], standard=std,
                              text=f"{s['rule']} on {m['cohort']} ({'primary' if std == config.primary_standard else 'secondary, pre-declared'} evidence standard {std})"))
        if std == config.primary_standard:
            for ab in config.plan.get("ablations") or []:
                removed = list(ab["remove"])
                missing = [f for f in removed if f not in set(sets["extended"])]
                if missing:
                    raise ConfigError(f"ablation {ab['cell']} removes predictors that are not in EXTENDED: {missing}")
                cells.append(Cell(cell=ab["cell"], planned=ab["cell"], kind="ablation", cohort="D00_CLEAN", feature_set=f"EXTENDED minus {removed}",
                                  features=[f for f in sets["extended"] if f not in set(removed)], question=ab.get("question"), text=ab.get("text", ""),
                                  removed=removed))
    for fut in config.plan.get("future") or []:
        cells.append(Cell(cell=fut["cell"], planned=fut["cell"], kind="future", cohort="FULL_LABELED", feature_set="FULL_EXTENDED", features=list(sets["extended"]),
                          text=" ".join(str(fut.get("text", "")).split()), source="not_run", not_run_reason="requires the DWH source correction; no result is fabricated"))
    # dedupe: the first cell with a (cohort, predictors) signature serves every later identical cell
    first: dict[tuple[str, tuple[str, ...]], Cell] = {}
    for c in cells:
        if c.kind == "future":
            continue
        if not c.features:
            c.source, c.not_run_reason = "not_run", "empty predictor set: no predictor of this set is admitted by its evidence standard"
            continue
        if c.signature in first:
            c.source, c.alias_of = "alias", first[c.signature].cell
            continue
        first[c.signature] = c
        if c.source == "run":
            c.label = run_label(c.cell)
    for c in cells:
        if c.source == "alias":
            c.label = first[c.signature].label
    return cells


def plan_record(cells: list[Cell], *, graph: Graph, config: D00Config, facts: dict[str, Any]) -> dict[str, Any]:
    """The frozen plan (no result inside) and its sha256."""
    body = {"cells": [asdict(c) for c in cells], "graph_sha256": graph.sha256, "config_sha256": config.sha256, **facts,
            "rules": ["every cell, feature set and ablation above was fixed before any model of this analysis was fitted or any test prediction read",
                      "feature sets come from the dependency graph (record dates and provenance only; no outcome values, no test partition)",
                      "identical specifications share one run; the completed reference runs are reused, never retrained",
                      "no cell is added, removed or re-specified after a result is seen; no model is called 'best'"]}
    text = json.dumps(body, sort_keys=True, default=str)
    return {**body, "plan_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
