"""Phase 3 stages S00-S06: preflight, cohort (FULL_LABELED) + schema verification + time-contract verification + evaluation history, source timing,
feature eligibility (verified / attested / bounded), reconstruction + coverage, the scientific feasibility decision (GO / NO_GO) and - only after
GO - the frozen feature sets."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from falls_ml.phase2 import durable as D
from falls_ml.phase2.featuresets import FeatureSets, _with_forms, design_spec
from falls_ml.phase2.screen import rank_spearman, redundancy_clusters, univariate_screen
from falls_ml.phase2.stages_data import _ram_gb, _selftest, _versions, digest_protected, verify_protected
from falls_ml.phase2.state import Phase2Stop
from falls_ml.phase3.config import P3_ALL, P3_BASE
from falls_ml.phase3.context import P3Ctx
from falls_ml.phase3.recovery import (ELIGIBLE, NR_DATA, RECOVERED, RECOVERED_BOUNDED, SAFE_ATTESTED, UNRESOLVED, VERIFIED, build_recovery,
                                      remediation_rows, source_timing)

PRIMARY = "PRIMARY_FULL"                    # SAFE_VERIFIED + SAFE_VERIFIED_BOUNDED + SAFE_ATTESTED: the full, defensible feature space (primary)
VERIFIED_ONLY = "SENSITIVITY_VERIFIED_ONLY" # row-verified timing only (attested sources removed); LASSO; never a selection candidate
HISTORICAL = "HISTORICAL_BASELINE_15"       # the 15 historical predictors when some are not eligible (context only)
EXPLORATORY = "EXPLORATORY_UNRESOLVED"      # + UNRESOLVED features (never future / forbidden): potential value only

HIST_SET = "HISTORICAL_B15"
EXPL_SET = "EXPLORATORY_FULL_PLUS_UNRESOLVED"
V_BASE, V_ALL = "P3_VERIFIED_BASE", "P3_VERIFIED_ALL"
LOW_RISK = "SENSITIVITY_LOW_AVAILABILITY_RISK"
L_BASE, L_ALL = "P3_LOWRISK_BASE", "P3_LOWRISK_ALL"
NO_RECENCY = "P3_ALL_NO_FALL_RECENCY"
CATEGORIES = (PRIMARY, VERIFIED_ONLY, LOW_RISK, HISTORICAL, EXPLORATORY)


def read_columns(ctx: P3Ctx) -> list[str]:
    """Every extract column Phase 3 reads: catalogue inputs, baseline sources and every source record date (the strict schema set)."""
    from falls_ml.data.meuhedet_wide import feature_input_columns

    cols = {c for f in ctx.cat.features for c in f.inputs}
    for f in ctx.mapping.features:
        if f.include_in_baseline:
            i = feature_input_columns(f)
            cols |= set(i["value"]) | set(i["validation"])
    cols |= {m.get("record_date") for m in ctx.dictionary.sources.values() if m.get("record_date")}
    cols |= {"Index_Date", "Is_Eligible_Cohort", ctx.mapping.outcome["label_column"]}
    return sorted(c for c in cols if c in set(ctx.contract.names))


# ============================================================================ S00
def s00_preflight(ctx: P3Ctx) -> None:
    st = ctx.run.stage("00", "preflight")
    if st.complete_record():
        return

    def fn(tmp: Path, seed: int) -> dict[str, Any]:
        env = {"python": sys.version.split()[0], "implementation": platform.python_implementation(), "platform": platform.platform(),
               "machine": platform.machine(), "cpu_count": os.cpu_count(), "ram_gb": _ram_gb(), "packages": _versions(), "threads": ctx.threads,
               "disk_free_gb": round(shutil.disk_usage(ctx.run.out).free / 2**30, 1)}
        D.write_json(tmp / "environment.json", env)
        D.write_json(tmp / "protected_digest.json", digest_protected(ctx.protected))
        D.write_json(tmp / "selftest.json", _selftest(ctx))
        return {"python": env["python"], "cpu_count": env["cpu_count"], "selftest": "PASSED", "protected_folders": len(ctx.protected)}

    st.item("preflight", fn)
    env = json.loads((st.path("preflight") / "environment.json").read_text(encoding="utf-8"))
    out = D.write_json(ctx.run.artifacts / "ENVIRONMENT.json", {k: v for k, v in env.items() if k != "disk_free_gb"})
    st.finalize([out], {"selftest": "PASSED"})


# ============================================================================ S01
def s01_cohort(ctx: P3Ctx) -> None:
    from falls_ml.phase3.cohort import build_phase3_cohort, evaluation_history

    st = ctx.run.stage("01", "cohort", ("00",))
    if st.complete_record():
        return
    scratch = ctx.run.out / "work"
    scratch.mkdir(parents=True, exist_ok=True)

    def fn(tmp: Path, seed: int) -> dict[str, Any]:
        work, facts = build_phase3_cohort(ctx.src, ctx.ref_dir, contract=ctx.contract, mapping=ctx.mapping, spec=ctx.espec, pepper_file=ctx.pepper_file,
                                          scratch=scratch, input_sha=ctx.input_sha, split_cfg=ctx.cfg["population"]["extra_rows_split"],
                                          schema_columns=read_columns(ctx), time_contract=ctx.tc, dictionary=ctx.dictionary, d00=ctx.d00)
        D.write_parquet(tmp / "trainval.parquet", work)
        D.write_json(tmp / "EVALUATION_HISTORY.json", evaluation_history(ctx.ref_dir, [p for p in ctx.protected if p != ctx.ref_dir], ctx.phase2_out, facts))
        return facts

    facts = st.item("cohort", fn)
    audit = facts.get("label_death_audit") or {}
    ctx.run.gate("LABEL_DEATH_SEMANTICS", bool(audit.get("overwrite_suspected")),
                 "no member who died within the 180-day window carries a recorded fall/fracture (the label may overwrite events with death)",
                 [f"deaths within the window: {audit.get('deaths_within_window')}", "confirm the label rule with the warehouse team"])
    tcv = facts.get("time_contract") or {}
    outs = [D.write_json(ctx.run.artifacts / "COHORT_FACTS.json", {k: v for k, v in facts.items() if k not in ("pepper_source", "schema", "time_contract")}),
            D.write_json(ctx.run.artifacts / "SCHEMA_VERIFICATION.json", facts["schema"]),
            D.write_json(ctx.run.artifacts / "TIME_CONTRACT_VERIFICATION.json", {"contract": ctx.tc.header, **tcv}),
            D.write_bytes(ctx.run.artifacts / "EVALUATION_HISTORY.json", (st.path("cohort") / "EVALUATION_HISTORY.json").read_bytes())]
    if not tcv.get("hard_passed"):
        v1, v2 = tcv.get("V1_label_excludes_index_day", {}), tcv.get("V2_window_end", {})
        raise Phase2Stop("TIME_CONTRACT_CONTRADICTED", "the extract contradicts the corrected END-OF-INDEX-DAY time contract (the outcome would include "
                         "index-day events that the predictors can see)", [f"V1: {v1.get('positives_event_on_or_before_index')} positive labels with an event on/before "
                         f"Index_Date (min days to event {v1.get('min_days_to_event')})", f"V2: Label_End_180D - Index_Date = {v2.get('label_end_minus_index_days')}",
                         "confirm the label definition with the DWH developer; Phase 3 does not fit any model on a contradicted contract"])
    v7 = tcv.get("V7_proxy_episode_audit") or {}
    ctx.run.gate("EPISODE_CONTINUATION", bool(v7.get("investigation")),
                 "the fall-recency signal may partly be one injury episode counted as history AND as outcome (Phase 2 PROXY_DOMINANCE finding): review "
                 "TIME_CONTRACT_VERIFICATION.json V7 before training", [f"{k}: {v}" for k, v in v7.items() if k != "by_recency_band"])
    v6 = tcv.get("V6_boundary_episode") or {}
    ctx.run.gate("BOUNDARY_EPISODE", bool(v6.get("investigation")),
                 "recorded events on the day after the index day are concentrated among patients with an index-day record: one episode crossing midnight "
                 "may be counted as history AND as the outcome", [f"{k}: {v}" for k, v in v6.items()])
    st.finalize(outs, {"train_rows": facts["phase3_partitions"]["train"]["n_rows"], "validation_rows": facts["phase3_partitions"]["validation"]["n_rows"],
                       "time_contract_hard_passed": bool(tcv.get("hard_passed")), "attestation_holds": bool(tcv.get("attestation_holds"))})
    verify_protected(ctx)


# ============================================================================ S02 source timing (historical semantics, root cause)
def s02_timing(ctx: P3Ctx) -> None:
    st = ctx.run.stage("02", "timing", ("01",))
    if st.complete_record():
        return
    w = ctx.work()

    def fn(tmp: Path, seed: int) -> dict[str, Any]:
        t = source_timing(w, dictionary=ctx.dictionary, d00=ctx.d00, rules=ctx.rules, partition=w["partition"], bridge=~w["d00_row"].astype(bool))
        D.write_csv(tmp / "SOURCE_TIMING.csv", t)
        return {"n_sources": int(len(t)), "root_causes": t["root_cause"].value_counts().to_dict()}

    res = st.item("timing", fn)
    out = D.write_bytes(ctx.run.artifacts / "SOURCE_TIMING.csv", (st.path("timing") / "SOURCE_TIMING.csv").read_bytes())
    st.finalize([out], res)


# ============================================================================ S03 row-level recovery
def phase2_eligibility(ctx: P3Ctx, values: pd.DataFrame) -> tuple[dict[str, str], dict[str, Any]]:
    """The Phase 2 eligibility of every feature recomputed with the UNCHANGED Phase 2 code on the Phase 2 population (the D00_CLEAN TRAIN +
    VALIDATION rows): the comparison column 'what Phase 2 could use'. Printed so the user can check it against the Phase 2 preflight."""
    from falls_ml.phase2.engineer import engineered_registry
    from falls_ml.phase2.registry import Provenance

    w = ctx.work()
    ref = ~w["d00_row"].to_numpy(dtype=bool)
    wr = w.loc[ref].reset_index(drop=True)
    vr = values.loc[ref].reset_index(drop=True)
    prov = Provenance(wr, ctx.facts()["index_date"], contract=ctx.contract, dictionary=ctx.dictionary, d00_config=ctx.d00)
    er = engineered_registry(vr, (wr["partition"] == "train").to_numpy(), prov, catalogue=ctx.cat, contract=ctx.contract, dictionary=ctx.dictionary,
                             min_observed_train=int(ctx.cfg["eligibility"]["min_observed_train_rows"]))
    elig = dict(zip(er["feature"], er["eligibility"]))
    bstat = {f.canonical: prov.feature(tuple(f.source_columns))["status"] for f in ctx.mapping.features if f.canonical in ctx.baseline}
    elig.update({k: {"SAFE": "ELIGIBLE", "UNRESOLVED": "ELIGIBLE_EXPLORATORY", "UNSAFE": "INELIGIBLE_UNSAFE"}[v] for k, v in bstat.items()})
    counts = er["eligibility"].value_counts().to_dict()
    return elig, {"population": "D00_CLEAN TRAIN + VALIDATION (the Phase 2 population)", "catalogue_eligibility_counts": counts,
                  "baseline_status": bstat, "n_baseline_safe": sum(1 for v in bstat.values() if v == "SAFE")}


def s03_recovery(ctx: P3Ctx) -> None:
    from falls_ml.phase2.engineer import engineer

    st = ctx.run.stage("03", "recovery", ("02",))
    if st.complete_record():
        return
    w = ctx.work()

    def fn(tmp: Path, seed: int) -> dict[str, Any]:
        values = engineer(w, ctx.cat)
        p2, p2facts = phase2_eligibility(ctx, values)
        res = build_recovery(w, values, catalogue=ctx.cat, mapping=ctx.mapping, baseline=ctx.baseline, contract=ctx.contract, dictionary=ctx.dictionary,
                             d00=ctx.d00, rules=ctx.rules, train_mask=(w["partition"] == "train").to_numpy(),
                             min_observed_train=int(ctx.cfg["eligibility"]["min_observed_train_rows"]), phase2_eligibility=p2, tc=ctx.tc,
                             attestation_holds=bool((ctx.facts().get("time_contract") or {}).get("attestation_holds")))
        src = {}
        for _, r in res.registry.iterrows():
            srcs = [s for s in str(r["sources"]).split("; ") if s]
            dated = [s for s in srcs if ctx.d00.source_evidence.get(s) == "COMPLETE_RECORD_DATE"]
            src[r["feature"]] = (dated or srcs or ["STATIC"])[0]
        D.write_csv(tmp / "FEATURE_RECOVERY.csv", res.registry)
        D.write_json(tmp / "feature_source.json", src)
        D.write_parquet(tmp / "unknown_raw.parquet", res.unknown.reset_index(drop=True))
        D.write_parquet(tmp / "upper_bound.parquet", res.upper_bound.reset_index(drop=True))
        timing = pd.read_csv(ctx.run.artifacts / "SOURCE_TIMING.csv")
        rem = remediation_rows(timing, res.registry, rules=ctx.rules, priority_domains=list(ctx.cfg["feasibility"]["priority_domains"]),
                               catalogue_domains=ctx.cat.domains)
        D.write_csv(tmp / "DWH_REMEDIATION_REQUIREMENTS.csv", rem)
        D.write_json(tmp / "PHASE2_RULE_ON_PHASE2_POPULATION.json", p2facts)
        return {**res.facts, "phase2_rule": p2facts}

    res = st.item("recovery", fn)
    outs = [D.write_bytes(ctx.run.artifacts / n, (st.path("recovery") / n).read_bytes())
            for n in ("FEATURE_RECOVERY.csv", "DWH_REMEDIATION_REQUIREMENTS.csv", "PHASE2_RULE_ON_PHASE2_POPULATION.json")]
    st.finalize(outs, {"classes": res["classes"], "n_newly_recovered": res["n_newly_recovered"]})


# ============================================================================ S04 reconstruction + coverage
def s04_reconstruct(ctx: P3Ctx) -> None:
    from falls_ml.phase2.engineer import engineer

    st = ctx.run.stage("04", "reconstruct", ("03",))
    if st.complete_record():
        return
    w = ctx.work()
    reg = ctx.registry()

    def fn(tmp: Path, seed: int) -> dict[str, Any]:
        values = engineer(w, ctx.cat)
        unk = pd.read_parquet(ctx.item_file("03", "recovery", "recovery", "unknown_raw.parquet"))
        ub = pd.read_parquet(ctx.item_file("03", "recovery", "recovery", "upper_bound.parquet"))
        # the reconstructed table holds ONLY values provably available at the start of the index day: an UNKNOWN cell becomes NaN here and is
        # never read as a value again (fitting excludes its row; evaluation replaces it by the set of possible pre-index states)
        for f in [c for c in unk.columns if c in values.columns]:
            values.loc[unk[f].to_numpy(dtype=bool), f] = np.nan
        D.write_parquet(tmp / "values.parquet", values.reset_index(drop=True))
        D.write_parquet(tmp / "unknown.parquet", unk)
        D.write_parquet(tmp / "upper_bound.parquet", ub)
        tr = (w["partition"] == "train").to_numpy()
        rows = []
        for _, r in reg.iterrows():
            f = r["feature"]
            x = pd.to_numeric(values[f], errors="coerce") if f in values.columns else pd.to_numeric(
                (w[f].astype(str) == "female").astype(float) if f == "sex" else w[f], errors="coerce")
            u = unk[f].to_numpy(dtype=bool) if f in unk.columns else np.zeros(len(w), dtype=bool)
            known_obs = ~u & x.notna().to_numpy()
            rows.append({"feature": f, "domain": r["domain"], "phase3_class": r["phase3_class"], "n_rows": len(w), "n_known_observed": int(known_obs.sum()),
                         "n_known_observed_train": int((known_obs & tr).sum()), "n_unknown": int(u.sum()), "n_unknown_train": int((u & tr).sum()),
                         "known_observed_pct": round(100.0 * known_obs.mean(), 3), "unknown_pct": round(100.0 * u.mean(), 4)})
        cov = pd.DataFrame(rows)
        D.write_csv(tmp / "FEATURE_COVERAGE.csv", cov)
        return {"n_unknown_cells": int(unk.to_numpy().sum()) if unk.size else 0, "n_row_level_features": int(unk.shape[1])}

    res = st.item("reconstruct", fn)
    out = D.write_bytes(ctx.run.artifacts / "FEATURE_COVERAGE.csv", (st.path("reconstruct") / "FEATURE_COVERAGE.csv").read_bytes())
    st.finalize([out], res)


# ============================================================================ S05 scientific feasibility (GO / NO_GO)
def _selection_bias_table(ctx: P3Ctx) -> pd.DataFrame:
    """TRAIN only, descriptive: the recorded-outcome rate of rows whose source record is dated on/after the index day vs the others. It shows
    what deleting those patients would have done; it is never used by any gate or model."""
    w = ctx.work()
    tr = w.loc[w["partition"] == "train"]
    idx = tr["Index_Date"].dt.normalize()
    y = tr["y"].to_numpy(dtype=int)
    rows = []
    for src, meta in ctx.dictionary.sources.items():
        dc = meta.get("record_date")
        if not dc or dc not in tr.columns or not pd.api.types.is_datetime64_any_dtype(tr[dc]):
            continue
        hit = ((tr[dc].dt.normalize() - idx).dt.days >= 0).to_numpy()
        n1, n0 = int(hit.sum()), int((~hit).sum())
        e1, e0 = int(y[hit].sum()), int(y[~hit].sum())
        rows.append({"source": src, "record_date_column": dc, "n_train_rows_on_or_after": n1, "events_on_or_after": e1,
                     "outcome_rate_on_or_after_pct": round(100.0 * e1 / n1, 3) if n1 else None, "n_train_rows_other": n0, "events_other": e0,
                     "outcome_rate_other_pct": round(100.0 * e0 / n0, 3) if n0 else None,
                     "rate_ratio": round((e1 / n1) / (e0 / n0), 3) if n1 and n0 and e0 else None,
                     "note": "descriptive only (TRAIN); never used by a gate or a model"})
    return pd.DataFrame(rows)


def feasibility_decision(reg: pd.DataFrame, values: pd.DataFrame, unknown: pd.DataFrame, y: np.ndarray, train: np.ndarray, cfg: Any,
                         time_contract_ok: bool = True) -> dict[str, Any]:
    f = cfg["feasibility"]
    prim = set(cfg["standards"]["PRIMARY_FULL"])
    rows = []
    for dom in sorted(set(reg.loc[reg["kind"] == "catalogue", "domain"])):
        d = reg[(reg["kind"] == "catalogue") & (reg["domain"] == dom)]
        rec = d[d["phase3_class"].isin(prim)]
        rec_items = rec[~rec["form_indicator"].astype(bool)]
        new_items = rec_items[rec_items["newly_recovered"].astype(bool)]
        inf = np.zeros(len(values), dtype=bool)
        for feat in rec_items["feature"]:
            x = values[feat].to_numpy(dtype=float)
            u = unknown[feat].to_numpy(dtype=bool) if feat in unknown.columns else np.zeros(len(values), dtype=bool)
            inf |= np.isfinite(x) & ~u
        n_inf, e_inf = int((inf & train).sum()), int(y[inf & train].sum())
        prio = dom in set(f["priority_domains"])
        adequate = n_inf >= int(f["min_domain_known_informative_train_rows"]) and e_inf >= int(f["min_domain_known_informative_train_events"])
        rows.append({"domain": dom, "priority_domain": prio, "n_features": int(len(d)), "n_eligible": int(len(rec)),
                     "n_verified": int((rec["phase3_class"] == "SAFE_VERIFIED").sum()), "n_attested": int((rec["phase3_class"] == SAFE_ATTESTED).sum()),
                     "n_bounded": int((rec["phase3_class"] == RECOVERED_BOUNDED).sum()), "n_eligible_items": int(len(rec_items)),
                     "n_newly_eligible_items": int(len(new_items)), "newly_eligible_items": "; ".join(new_items["feature"]),
                     "n_known_informative_train": n_inf, "events_known_informative_train": e_inf, "adequate": adequate,
                     "domain_go": bool(time_contract_ok and prio and len(new_items) and adequate),
                     "n_not_eligible": int((~d["phase3_class"].isin([*prim, NR_DATA])).sum())})
    t = pd.DataFrame(rows)
    go = bool(t["domain_go"].any()) if len(t) else False
    return {"decision": "GO" if go else "NO_GO", "outcome": "A" if go else "B", "go_domains": t.loc[t["domain_go"], "domain"].tolist() if len(t) else [],
            "time_contract_hard_passed": bool(time_contract_ok), "rule": f["rule"], "table": t}


def s05_feasibility(ctx: P3Ctx) -> None:
    st = ctx.run.stage("05", "feasibility", ("04",))
    if st.complete_record():
        return
    reg = ctx.registry()
    w = ctx.work()

    def fn(tmp: Path, seed: int) -> dict[str, Any]:
        tc_ok = bool((ctx.facts().get("time_contract") or {}).get("hard_passed"))
        dec = feasibility_decision(reg, ctx.values(), ctx.unknown(), w["y"].to_numpy(dtype=int), (w["partition"] == "train").to_numpy(), ctx.cfg, tc_ok)
        D.write_csv(tmp / "DOMAIN_FEASIBILITY.csv", dec.pop("table"))
        D.write_csv(tmp / "SELECTION_BIAS_DIAGNOSTIC.csv", _selection_bias_table(ctx))
        return dec

    res = st.item("feasibility", fn)
    outs = [D.write_bytes(ctx.run.artifacts / n, (st.path("feasibility") / n).read_bytes()) for n in ("DOMAIN_FEASIBILITY.csv", "SELECTION_BIAS_DIAGNOSTIC.csv")]
    outs.append(D.write_json(ctx.run.artifacts / "FEASIBILITY_DECISION.json", res))
    ctx.run.events("S05_feasibility", "decision", res["decision"], go_domains=res["go_domains"])
    st.finalize(outs, {"decision": res["decision"], "go_domains": res["go_domains"]})


# ============================================================================ S06 feature sets (GO only), frozen before any fit
def _in(reg: pd.DataFrame, kind: str, classes: Any) -> list[str]:
    return reg.loc[(reg["kind"] == kind) & reg["phase3_class"].isin(list(classes)), "feature"].tolist()


def exploratory_candidates(ctx: P3Ctx, reg: pd.DataFrame) -> list[str]:
    """UNRESOLVED catalogue features (never future / forbidden) with enough data: the separately labelled exploratory track only."""
    rc = reg[reg["kind"] == "catalogue"].set_index("feature")
    return [f for f in ctx.cat.names if rc.at[f, "phase3_class"] == UNRESOLVED
            and int(rc.at[f, "n_known_observed_train"]) >= int(ctx.cfg["eligibility"]["min_observed_train_rows"]) and int(rc.at[f, "n_unique_known_train"]) > 1]


def build_p3_sets(ctx: P3Ctx, reg: pd.DataFrame, reps: dict[str, str], design: dict[str, Any]) -> FeatureSets:
    cat = ctx.cat
    order = {f.name: f.order for f in cat.features}
    std = ctx.cfg["standards"]
    full = [f for f in cat.names if f in set(_in(reg, "catalogue", std["PRIMARY_FULL"]))]
    pruned = [f for f in full if reps.get(f, f) == f]
    ver = [f for f in pruned if f in set(_in(reg, "catalogue", std["SENSITIVITY_VERIFIED_ONLY"]))]
    base_full = [b for b in ctx.baseline if b in set(_in(reg, "baseline", std["PRIMARY_FULL"]))]
    base_ver = [b for b in ctx.baseline if b in set(_in(reg, "baseline", std["SENSITIVITY_VERIFIED_ONLY"]))]
    expl_new = exploratory_candidates(ctx, reg)
    sets: dict[str, dict[str, Any]] = {}

    def add(name: str, feats: list[str], *, category: str, kind: str, base: list[str], text: str, universe: set[str], domain: str | None = None,
            forms: bool = True) -> None:
        feats = sorted(set(feats), key=lambda x: order[x])
        if forms:
            feats = _with_forms(feats, cat, universe)
        sets[name] = {"features": feats, "baseline": list(base), "category": category, "kind": kind, "domain": domain, "description": text,
                      "n_new_features": len(feats), "n_baseline_features": len(base)}

    U = set(full)
    add(P3_BASE, [], category=PRIMARY, kind="main", base=base_full, universe=U,
        text=f"P3_BASE: the historical predictors eligible under the corrected contract ({len(base_full)} of {len(ctx.baseline)}: {', '.join(base_full)})")
    add(P3_ALL, pruned, category=PRIMARY, kind="main", base=base_full, universe=U, text="P3_BASE + every eligible non-redundant clinical feature (the full experiment)")
    for d in cat.domains:
        fs = [f for f in full if cat.get(f).domain == d]
        if fs:
            add(f"P3_BASE_PLUS_{d}", fs, category=PRIMARY, kind="domain_add", domain=d, base=base_full, universe=U, text=f"P3_BASE + eligible {cat.domains[d]['label']}")
    cum: list[str] = []
    named = {s for s in cat.waterfall if s != "OTHER_NATIVE"}
    steps = []
    for k, step in enumerate(cat.waterfall, start=1):
        ds = [x for x in cat.domains if x not in named] if step == "OTHER_NATIVE" else [step]
        added = [f for f in pruned if cat.get(f).domain in ds]
        cum += added
        name = P3_ALL if step == "OTHER_NATIVE" else f"P3_WATERFALL_{k}_{step}"
        if step != "OTHER_NATIVE":
            add(name, list(cum), category=PRIMARY, kind="waterfall", base=base_full, universe=U, text=f"cumulative: P3_BASE + {' + '.join(cat.waterfall[:k])}")
        steps.append({"step": k, "added": step, "domains": ds, "set": name, "n_features_added": len(added)})
    add(f"{P3_ALL}_WITHOUT_ASSESSMENT_FLAGS", [f for f in pruned if not cat.get(f).form_indicator], category=PRIMARY, kind="sensitivity", base=base_full,
        universe=U, forms=False, text="P3_ALL_RECOVERED without the explicit form 'assessed' indicators")
    for d in cat.domains:
        keep = [f for f in pruned if cat.get(f).domain != d and not cat.get(f).form_indicator]
        if len(keep) != len([f for f in pruned if not cat.get(f).form_indicator]):
            add(f"{P3_ALL}_MINUS_{d}", keep, category=PRIMARY, kind="loo", domain=d, base=base_full, universe=U, text=f"P3_ALL_RECOVERED without {cat.domains[d]['label']}")
    low_levels = set((ctx.tc.header.get("availability") or {}).get("low_risk_levels") or ["LOW", "MEDIUM"])
    risk = dict(zip(reg["feature"], reg["availability_risk"]))
    low = [f for f in pruned if risk.get(f) in low_levels]
    base_low = [b for b in base_full if risk.get(b) in low_levels]
    LU = set(low)
    add(L_BASE, [], category=LOW_RISK, kind="sensitivity", base=base_low, universe=LU,
        text=f"low-availability-risk sensitivity: historical predictors whose declared availability risk is {sorted(low_levels)} ({', '.join(base_low)})")
    add(L_ALL, low, category=LOW_RISK, kind="sensitivity", base=base_low, universe=LU,
        text="low-availability-risk sensitivity: + every eligible feature with LOW / MEDIUM availability risk (registries / undocumented scores removed)")
    rec_feats = list((ctx.tc.header.get("proxy_audit") or {}).get("recency_features") or [])
    add(NO_RECENCY, [f for f in pruned if f not in rec_feats], category=PRIMARY, kind="proxy_ablation", base=base_full, universe=U,
        text=f"pre-declared proxy ablation: P3_ALL_RECOVERED without the fall-recency features {rec_feats} (Phase 2 PROXY_DOMINANCE finding)")
    VU = set(ver)
    add(V_BASE, [], category=VERIFIED_ONLY, kind="sensitivity", base=base_ver, universe=VU,
        text=f"verified-only sensitivity: historical predictors with row-verified timing ({', '.join(base_ver)})")
    add(V_ALL, ver, category=VERIFIED_ONLY, kind="sensitivity", base=base_ver, universe=VU,
        text="verified-only sensitivity: + every row-verified non-redundant feature (attested sources removed)")
    hist_ok = set(ctx.baseline) <= set(_in(reg, "baseline", [*std["PRIMARY_FULL"], UNRESOLVED]))   # never with future / forbidden predictors
    if base_full != list(ctx.baseline) and hist_ok:
        add(HIST_SET, [], category=HISTORICAL, kind="historical", base=list(ctx.baseline), universe=set(),
            text="the 15 historical predictors (some not eligible under the corrected contract): context only")
    if expl_new:
        add(EXPL_SET, [*pruned, *expl_new], category=EXPLORATORY, kind="exploratory",
            base=[b for b in ctx.baseline if b in set(_in(reg, "baseline", [*std["PRIMARY_FULL"], UNRESOLVED]))], universe=U | set(expl_new),
            text="EXPLORATORY: the full set + UNRESOLVED features (never future / forbidden): potential value only")
    aliases, seen = {}, {}
    for name, s in sets.items():
        key = json.dumps([s["category"], s["features"], s["baseline"]])
        if key in seen:
            aliases[name] = seen[key]
        else:
            seen[key] = name
    fs = FeatureSets(sets=sets, design={**design, "waterfall": steps, "p3_base": base_full, "eligible_full": full, "verified_only": ver,
                                        "exploratory_unresolved": expl_new, "pruned_out": sorted(set(full) - set(pruned))}, aliases=aliases)
    fs.sha256 = hashlib.sha256(json.dumps({"sets": fs.sets, "design": fs.design, "aliases": fs.aliases}, sort_keys=True, default=str).encode()).hexdigest()
    return fs


def check_p3_invariants(fs: FeatureSets, reg: pd.DataFrame, standards: dict[str, list[str]]) -> list[str]:
    """Violations (empty = clean): every set holds only the classes its category admits; nothing future / forbidden anywhere."""
    r = reg.set_index("feature")
    admit = {PRIMARY: set(standards["PRIMARY_FULL"]), VERIFIED_ONLY: set(standards["SENSITIVITY_VERIFIED_ONLY"]),
             LOW_RISK: set(standards["SENSITIVITY_LOW_AVAILABILITY_RISK"]),
             EXPLORATORY: set(standards["EXPLORATORY_UNRESOLVED"])}
    bad = []
    for name, s in fs.sets.items():
        for f in [*s["features"], *s["baseline"]]:
            cl = r.at[f, "phase3_class"] if f in r.index else "not in registry"
            if s["category"] in admit and cl not in admit[s["category"]] and not (s["category"] == EXPLORATORY and f in s["baseline"]):
                bad.append(f"{name}: {f} ({cl})")
            if cl in ("NOT_RECOVERABLE_FORBIDDEN", "NOT_RECOVERABLE_FUTURE_RECORDS") and s["category"] != HISTORICAL:
                bad.append(f"{name}: {f} ({cl})")
    return bad


def s06_feature_sets(ctx: P3Ctx) -> None:
    st = ctx.run.stage("06", "feature_sets", ("05",))
    if st.complete_record():
        return
    reg = ctx.registry()
    w, v, unk = ctx.work(), ctx.values(), ctx.unknown()
    tr = (w["partition"] == "train").to_numpy()
    rec = [f for f in reg.loc[(reg["kind"] == "catalogue") & reg["phase3_class"].isin(ctx.cfg["standards"]["PRIMARY_FULL"]), "feature"]]
    # label-blind decisions on TRAIN rows whose recovered features are all KNOWN
    cols = [c for c in [*rec, *reg.loc[(reg["kind"] == "baseline") & reg["phase3_class"].isin(ctx.cfg["standards"]["PRIMARY_FULL"]), "feature"]] if c in unk.columns]
    known = ~unk[cols].to_numpy(dtype=bool).any(axis=1) if cols else np.ones(len(w), dtype=bool)
    m = tr & known
    sc = ctx.cfg["screening"]

    def fn(tmp: Path, seed: int) -> dict[str, Any]:
        base = [b for b in ctx.baseline if b in set(reg.loc[(reg["kind"] == "baseline") & reg["phase3_class"].isin(ctx.cfg["standards"]["PRIMARY_FULL"]), "feature"])]
        bx = w.loc[m, base].copy()
        if "sex" in bx:
            bx["sex"] = (bx["sex"].astype(str) == "female").astype(float)
        X = pd.concat([v.loc[m, rec].reset_index(drop=True), bx.astype(float).reset_index(drop=True)], axis=1)
        corr, _ = rank_spearman(X)
        miss = {f: float(v.loc[m, f].isna().mean() * 100) for f in rec}
        order = {f.name: f.order for f in ctx.cat.features}
        clusters, reps = redundancy_clusters(corr, threshold=float(sc["redundancy_abs_spearman"]), missing_pct=miss, order=order, baseline=set(base))
        ds = design_spec(v, m, ctx.cat, [*rec, *exploratory_candidates(ctx, reg)], unanswered_min_share=float(sc["item_unanswered_min_share"]))
        fs = build_p3_sets(ctx, reg, reps, ds)
        bad = check_p3_invariants(fs, reg, ctx.cfg["standards"])
        if bad:
            raise Phase2Stop("EXCLUDED_FEATURE_IN_SET", "a feature set contains a feature its category does not admit (recovery rule)", bad[:20])
        er = reg.rename(columns={"phase3_class": "eligibility"})[["feature", "eligibility", "domain"]].copy()
        er["eligibility"] = er["eligibility"].map(lambda c: "ELIGIBLE" if c in set(ctx.cfg["standards"]["PRIMARY_FULL"]) else "NOT_ELIGIBLE")
        uni = univariate_screen(v.loc[m, rec].reset_index(drop=True), w.loc[m, "y"].to_numpy(dtype=int), er[er["feature"].isin(rec)],
                                min_cell=int(sc["min_cell"]), sparse_min_events=int(sc["sparse_min_events"]), baseline=w.loc[m, base].reset_index(drop=True))
        D.write_json(tmp / "FEATURE_SETS.json", fs.to_dict())
        D.write_csv(tmp / "UNIVARIATE_SCREEN.csv", uni)
        D.write_csv(tmp / "REDUNDANCY_CLUSTERS.csv", clusters)
        g = float(ctx.cfg["gates"]["single_feature_univariate_auroc"]) - 0.5
        dom = uni.loc[(uni["role"] == "candidate") & (uni["abs_auroc_deviation"].fillna(0) >= g), "feature"].tolist()
        return {"n_sets": len(fs.sets), "n_fitted": len(fs.fitted()), "sha256": fs.sha256, "n_screen_rows_train": int(m.sum()), "dominant_features": dom}

    res = st.item("feature_sets", fn)
    ctx.run.gate("SINGLE_FEATURE_DOMINANCE", bool(res["dominant_features"]),
                 f"recovered feature(s) with TRAIN univariate AUROC >= {ctx.cfg['gates']['single_feature_univariate_auroc']} (or <= its mirror): investigate "
                 "as a possible proxy of the outcome before use", [f"feature: {f}" for f in res["dominant_features"]])
    outs = [D.write_bytes(ctx.run.artifacts / n, (st.path("feature_sets") / n).read_bytes())
            for n in ("FEATURE_SETS.json", "UNIVARIATE_SCREEN.csv", "REDUNDANCY_CLUSTERS.csv")]
    ctx.run.checkpoint(feature_registry_sha256=res["sha256"])
    st.finalize(outs, {"n_sets": res["n_sets"], "sha256": res["sha256"]})
