"""S05 feature sets and the label-blind design specification, frozen (sha256) before any model is fitted (planning/EXPERIMENT_PLAN.md §5).

Three categories that are never mixed (FINAL readiness brief §3; reviews/ASTRA_FINAL_RESOLUTION.md):

- HISTORICAL_BASELINE_15: the previous benchmark, the 15 EXTENDED predictors with unchanged definitions, reproduced for comparison
  (it contains predictors whose timing is UNRESOLVED in this extract; that is reported, never hidden).
- SAFE_DISCOVERY (primary): only features - new ones AND baseline ones - whose prediction-time provenance is proven SAFE by the D-00 rules.
  Every set of this category starts from SAFE_BASE (the SAFE subset of BASELINE_15). An UNRESOLVED feature can never enter it (hard stop).
- EXPLORATORY_UNRESOLVED_SENSITIVITY: a separately labelled LASSO sensitivity that also offers UNRESOLVED sources (never UNSAFE ones).

Sets are defined by pre-declared rules applied to the registry (eligibility), the screening (redundancy representatives, computed separately
for the SAFE universe and the exploratory universe) and the catalogue (domains, forms, mapping classes). A set that contains an item of an
assessment form automatically contains that form's "assessed" indicator (not measured != negative), except the declared
WITHOUT_EXPLICIT_ASSESSMENT_FLAGS sensitivity.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from falls_ml.phase2.config import FALL_RECENCY_BANDS, OTHER_NATIVE, Catalogue

HISTORICAL = "HISTORICAL_BASELINE_15"
SAFE_DISCOVERY = "SAFE_DISCOVERY"
EXPLORATORY = "EXPLORATORY_UNRESOLVED_SENSITIVITY"
CATEGORIES = (HISTORICAL, SAFE_DISCOVERY, EXPLORATORY)
#: engineered-feature eligibility allowed per category (the SAFE invariant: SAFE_DISCOVERY = proven SAFE only)
CATEGORY_ELIGIBILITY = {SAFE_DISCOVERY: ("ELIGIBLE",), EXPLORATORY: ("ELIGIBLE", "ELIGIBLE_EXPLORATORY"), HISTORICAL: ()}
BASELINE = "BASELINE_15"
SAFE_BASE = "SAFE_BASE"
ALL_SAFE = "ALL_REVIEWED_SAFE"
EXPL_B15_SAFE = "EXPLORATORY_B15_PLUS_SAFE"
EXPL_ALL = "EXPLORATORY_ALL_REVIEWED"


@dataclass
class FeatureSets:
    sets: dict[str, dict[str, Any]]
    design: dict[str, Any]
    aliases: dict[str, str]                 # set -> the set with an identical specification that is fitted instead
    sha256: str = ""

    def fitted(self) -> list[str]:
        return [s for s in self.sets if s not in self.aliases]

    def fitted_name(self, name: str) -> str:
        return self.aliases.get(name, name)

    def category(self, name: str) -> str:
        return str(self.sets[name]["category"])

    def to_dict(self) -> dict[str, Any]:
        return {"sets": self.sets, "design": self.design, "aliases": self.aliases, "sha256": self.sha256}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FeatureSets:
        return cls(sets=d["sets"], design=d["design"], aliases=d["aliases"], sha256=d.get("sha256", ""))


def design_spec(values: pd.DataFrame, train_mask: np.ndarray, catalogue: Catalogue, eligible: list[str], *, unanswered_min_share: float) -> dict[str, Any]:
    """Label-blind encoding decisions made once on all TRAIN rows (the fitted parts - medians, FP forms - happen inside every fold)."""
    tr = values.loc[train_mask]
    feats: dict[str, Any] = {}
    na_groups: dict[str, dict[str, Any]] = {}
    mask_group: dict[str, str] = {}
    for name in eligible:
        f = catalogue.get(name)
        s = tr[name]
        na = s.isna().to_numpy()
        entry: dict[str, Any] = {"kind": f.kind, "linear": f.linear, "missing": f.missing, "form": f.form, "na_group": None, "levels_used": None,
                                 "fill": "zero" if f.kind in ("binary", "ordinal") or f.linear == "fall_recency_bands" else "median"}
        if f.linear == "thermometer":
            obs = s.dropna().to_numpy()
            entry["levels_used"] = [lv for lv in f.levels[1:] if 0 < int((obs >= lv).sum()) < len(obs)]
        if f.linear == "fall_recency_bands":
            entry["bands"] = list(FALL_RECENCY_BANDS)
        if na.any() and f.linear != "fall_recency_bands":
            covered = False
            if f.missing == "not_assessed" and f.form is not None and catalogue.indicator_of(f.form) in eligible:
                ind = tr[catalogue.indicator_of(f.form)].to_numpy() == 1.0
                mismatch = float(np.mean(na != ~ind))
                entry["mismatch_vs_form"] = round(mismatch, 5)
                covered = mismatch <= unanswered_min_share
            if not covered:
                key = hashlib.sha256(np.packbits(na).tobytes()).hexdigest()
                gid = mask_group.setdefault(key, f"na__{name}")
                na_groups.setdefault(gid, {"members": [], "share_train": round(float(na.mean()), 5)})["members"].append(name)
                entry["na_group"] = gid
            else:
                entry["na_covered_by"] = catalogue.indicator_of(f.form)
        feats[name] = entry
    return {"features": feats, "na_groups": na_groups, "rules": {
        "binary": "1 if positive else 0; NULL covered by the form indicator (or an na__ indicator when the NULL pattern differs)",
        "ordinal": "thermometer dummies >= level for the declared levels present on TRAIN (never a constant column)",
        "count/days/amount": "log1p (declared); NULL -> fold-median + indicator", "fall_recency_bands": f"<= {FALL_RECENCY_BANDS} days bands, reference = no prior fall",
        "unanswered_min_share": unanswered_min_share, "trees": "raw engineered values with NULL kept (native missing handling) + form indicators"}}


def _with_forms(feats: list[str], catalogue: Catalogue, eligible: set[str]) -> list[str]:
    """Add the "assessed" indicator of every form an item comes from. An indicator that is not eligible in the set's category (e.g. constant
    because every row was assessed, or not SAFE) is not added; its items then have no "not assessed" state to encode."""
    need = {catalogue.indicator_of(catalogue.get(f).form) for f in feats if catalogue.get(f).missing == "not_assessed" and catalogue.get(f).form}
    order = {f.name: f.order for f in catalogue.features}
    return sorted(set(feats) | (need & eligible), key=lambda x: order[x])


def build_feature_sets(catalogue: Catalogue, eng_registry: pd.DataFrame, representatives: dict[str, dict[str, str]], *, baseline_features: list[str],
                       baseline_status: dict[str, str], design: dict[str, Any]) -> FeatureSets:
    reg = eng_registry.set_index("feature")
    order = {f.name: f.order for f in catalogue.features}
    safe = [f for f in catalogue.names if reg.at[f, "eligibility"] in CATEGORY_ELIGIBILITY[SAFE_DISCOVERY]]
    expl = [f for f in catalogue.names if reg.at[f, "eligibility"] in CATEGORY_ELIGIBILITY[EXPLORATORY]]
    universe = {SAFE_DISCOVERY: set(safe), EXPLORATORY: set(expl), HISTORICAL: set()}
    rep_s, rep_x = representatives[SAFE_DISCOVERY], representatives[EXPLORATORY]
    # redundancy pruning inside each universe (a feature redundant with a baseline feature of that universe leaves the pooled sets too)
    safe_pruned = [f for f in safe if rep_s.get(f, f) == f]
    expl_pruned = [f for f in expl if rep_x.get(f, f) == f]
    safe_base = [f for f in baseline_features if baseline_status.get(f) == "SAFE"]
    sets: dict[str, dict[str, Any]] = {}

    def add(name: str, feats: list[str], *, category: str, kind: str, base: list[str], text: str, domain: str | None = None, forms: bool = True) -> None:
        feats = sorted(set(feats), key=lambda x: order[x])
        if forms:
            feats = _with_forms(feats, catalogue, universe[category])
        sets[name] = {"features": feats, "baseline": list(base), "category": category, "kind": kind, "domain": domain, "description": text,
                      "n_new_features": len(feats), "n_baseline_features": len(base)}

    add(BASELINE, [], category=HISTORICAL, kind="historical", base=baseline_features,
        text="HISTORICAL benchmark: the 15 EXTENDED predictors (unchanged definitions, provenance as found), refitted in the Phase 2 protocol")
    add(SAFE_BASE, [], category=SAFE_DISCOVERY, kind="main", base=safe_base,
        text=f"SAFE_BASE: the BASELINE_15 predictors whose provenance is SAFE ({', '.join(safe_base) or 'none'})")
    add(ALL_SAFE, safe_pruned, category=SAFE_DISCOVERY, kind="main", base=safe_base, text="SAFE_BASE + every SAFE non-redundant feature")
    add("EFALLS_MAPPABLE_SAFE", [f for f in safe_pruned if catalogue.get(f).mapping_class.startswith("EFALLS")], category=SAFE_DISCOVERY, kind="main",
        base=safe_base, text="SAFE_BASE + SAFE features approximating eFalls concepts (non-redundant)")
    add("MEUHEDET_NATIVE_SAFE", [f for f in safe_pruned if catalogue.get(f).mapping_class == "MEUHEDET_NATIVE"], category=SAFE_DISCOVERY, kind="main",
        base=safe_base, text="SAFE_BASE + SAFE Meuhedet-native features (non-redundant)")
    add(f"{ALL_SAFE}_WITHOUT_EXPLICIT_ASSESSMENT_FLAGS", [f for f in safe_pruned if not catalogue.get(f).form_indicator], category=SAFE_DISCOVERY,
        kind="sensitivity", base=safe_base, forms=False,
        text="ALL_REVIEWED_SAFE without the explicit form 'assessed' indicators (automatic insertion overridden; the other NULL indicators stay, so "
             "missingness can still carry assessment-process information - Astra F-10)")
    domains = list(catalogue.domains)
    for d in domains:
        fs = [f for f in safe if catalogue.get(f).domain == d]
        if fs:
            add(f"SAFE_BASE_PLUS_{d}", fs, category=SAFE_DISCOVERY, kind="domain_add", domain=d, base=safe_base,
                text=f"SAFE_BASE + all SAFE {catalogue.domains[d]['label']} features")
    cum: list[str] = []
    steps = []
    named = {s for s in catalogue.waterfall if s != OTHER_NATIVE}
    for k, step in enumerate(catalogue.waterfall, start=1):
        ds = [x for x in domains if x not in named] if step == OTHER_NATIVE else [step]
        added = [f for f in safe_pruned if catalogue.get(f).domain in ds]
        cum += added
        name = ALL_SAFE if step == OTHER_NATIVE else f"SAFE_WATERFALL_{k}_{step}"
        if step != OTHER_NATIVE:
            add(name, list(cum), category=SAFE_DISCOVERY, kind="waterfall", base=safe_base,
                text=f"cumulative: SAFE_BASE + {' + '.join(catalogue.waterfall[:k])} (SAFE, pruned)")
        steps.append({"step": k, "added": step, "domains": ds, "set": name, "n_safe_features_added": len(added)})
    for d in domains:
        keep = [f for f in safe_pruned if catalogue.get(f).domain != d and not catalogue.get(f).form_indicator]
        if len(keep) != len([f for f in safe_pruned if not catalogue.get(f).form_indicator]):
            add(f"{ALL_SAFE}_MINUS_{d}", keep, category=SAFE_DISCOVERY, kind="loo", domain=d, base=safe_base,
                text=f"ALL_REVIEWED_SAFE without the {catalogue.domains[d]['label']} features")
    # the separately labelled exploratory sensitivity (LASSO only; never a selection candidate)
    add(EXPL_B15_SAFE, [f for f in safe_pruned if rep_x.get(f, f) == f or rep_x.get(f) not in baseline_features], category=EXPLORATORY,
        kind="exploratory", base=baseline_features,
        text="EXPLORATORY: the historical BASELINE_15 (incl. its UNRESOLVED predictors) + every SAFE non-redundant new feature")
    add(EXPL_ALL, expl_pruned, category=EXPLORATORY, kind="exploratory", base=baseline_features,
        text="EXPLORATORY: BASELINE_15 + every SAFE or UNRESOLVED non-redundant new feature (never UNSAFE)")
    # identical specifications share one fit, within a category only (a SAFE set is never an alias of a historical / exploratory one);
    # SAFE_BASE and ALL_REVIEWED_SAFE are declared first, so they are always fitted under their own names unless identical to each other
    aliases: dict[str, str] = {}
    seen: dict[str, str] = {}
    for name, s in sets.items():
        key = json.dumps([s["category"], s["features"], s["baseline"]])
        if key in seen:
            aliases[name] = seen[key]
        else:
            seen[key] = name
    fs = FeatureSets(sets=sets, design={**design, "waterfall": steps, "safe_base": safe_base,
                                        "pruned_out": {SAFE_DISCOVERY: sorted(set(safe) - set(safe_pruned)), EXPLORATORY: sorted(set(expl) - set(expl_pruned))}},
                     aliases=aliases)
    fs.sha256 = hashlib.sha256(json.dumps({"sets": fs.sets, "design": fs.design, "aliases": fs.aliases}, sort_keys=True, default=str).encode()).hexdigest()
    return fs


def check_category_invariants(fs: FeatureSets, eng_registry: pd.DataFrame, baseline_status: dict[str, str], baseline_features: list[str]) -> list[str]:
    """Violations of the category rules (empty = clean). SAFE_DISCOVERY: every new feature ELIGIBLE (SAFE) and every baseline feature SAFE;
    EXPLORATORY: never UNSAFE / ineligible; HISTORICAL: exactly BASELINE_15, no new feature."""
    reg = eng_registry.set_index("feature")
    bad: list[str] = []
    for name, s in fs.sets.items():
        cat = s.get("category")
        if cat not in CATEGORIES:
            bad.append(f"{name}: unknown category {cat!r}")
            continue
        if cat == HISTORICAL:
            if s["features"] or list(s["baseline"]) != list(baseline_features):
                bad.append(f"{name}: the historical benchmark must be exactly BASELINE_15 with no new feature")
            continue
        allowed = CATEGORY_ELIGIBILITY[cat]
        bad += [f"{name}: {f} ({reg.at[f, 'eligibility'] if f in reg.index else 'not in registry'})" for f in s["features"]
                if f not in reg.index or reg.at[f, "eligibility"] not in allowed]
        if cat == SAFE_DISCOVERY:
            bad += [f"{name}: baseline {f} ({baseline_status.get(f)})" for f in s["baseline"] if baseline_status.get(f) != "SAFE"]
    return bad
