"""``falls_ml meuhedet-phase5``: one resumable overnight command.

    PREFLIGHT   schema, X sealing, cohort, V21 definition, the 2026 outcome contract, feature eligibility, feature sets, the fixed fold
                assignment -> preflight/ (aggregate) and work/ (local, row-level) + the frozen PLAN.json; ends with SAFE TO MODEL or STOP
    PRIMARY     nested CV: 3 families x {OLD, OLD_PLUS_NEW_SAFE, the two sensitivity sets} x outer folds   (then an INTERIM report)
    FINAL       each family tuned on every patient for OLD and OLD_PLUS_NEW_SAFE (development models for coefficients / stability)
    DOMAIN      OLD + each new domain;   ABLATION   OLD_PLUS_NEW_SAFE minus each pre-declared block
    EXPLAIN     permutation importance (+ XGBoost SHAP) of the outer-fold models;   STABILITY   bootstrap refits of the final process
    REPORT      aggregate-only share/ (management + scientific summaries, tables, figures, privacy scan)

Every unit / task is committed by a COMPLETE.json written last; ``--resume`` verifies that the input, the settings, the mode and the code are
those of the plan and continues with the first unfinished unit (nothing finished is recomputed). RUN_STATUS.json (heartbeat, ETA),
RUN_TIMINGS.csv and OVERNIGHT_PROGRESS.log are written as the run goes; Ctrl+C / a power cut leaves a resumable folder.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from falls_ml.artifacts import utc_now
from falls_ml.phase2 import durable as D
from falls_ml.phase2.state import Phase2Stop
from falls_ml.phase5 import DESIGN_LABEL, PHASE5_VERSION
from falls_ml.phase5.config import FAMILIES, SET_LOWRISK, SET_NEW, SET_OLD, SET_VERIFIED, load_phase5_config
from falls_ml.phase5.engine import Ctx, UnitSpec, is_complete, outer_folds, run_unit
from falls_ml.phase5.models import DeviceState

PLAN = "PLAN.json"
SAFE_LINE = "SAFE TO MODEL"
STOP_LINE = "STOP - 2026 REDEVELOPMENT NOT DEFENSIBLE"
STAGE_ORDER = ("PREFLIGHT", "PRIMARY", "INTERIM_REPORT", "FINAL", "DOMAIN", "ABLATION", "EXPLAIN", "STABILITY", "REPORT")
EARLIER_PHASE_MARKERS = ("PHASE2_PLAN.json", "PHASE3_PLAN.json", "PHASE4_PLAN.json")


def dirs(out: Path) -> dict[str, Path]:
    return {"preflight": out / "preflight", "work": out / "work", "share": out / "share", "logs": out / "logs"}


# ============================================================================ status, heartbeat, progress log, timings
class Monitor:
    def __init__(self, out: Path, heartbeat: float):
        self.out = out
        self.lock = threading.Lock()
        self.state: dict[str, Any] = {}
        self.t0 = time.time()
        self.hb = float(heartbeat)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        prev = out / "RUN_STATUS.json"
        if prev.is_file():
            try:
                old = json.loads(prev.read_text(encoding="utf-8"))
                self.state.update({k: v for k, v in old.items() if k not in ("heartbeat", "current", "elapsed_seconds_this_session", "memory")})
                self.state["previous_elapsed_seconds"] = float(old.get("elapsed_seconds_total") or 0.0)
                self.state["retries"] = int(old.get("retries", 0))
                self.state["sessions"] = int(old.get("sessions", 0))
            except Exception:  # noqa: BLE001
                pass
        self.state["sessions"] = int(self.state.get("sessions", 0)) + 1
        self.state.setdefault("failures", [])
        self.state.setdefault("retries", 0)

    def log(self, msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with self.lock:
            self.out.mkdir(parents=True, exist_ok=True)
            with open(self.out / "OVERNIGHT_PROGRESS.log", "a", encoding="utf-8", newline="\n") as fh:
                fh.write(line + "\n")

    def update(self, **kw: Any) -> None:
        with self.lock:
            self.state.update(kw)
            self._write()

    def _write(self) -> None:
        from falls_ml.phase5.resources import memory_bytes, process_rss

        now = time.time()
        el = now - self.t0
        self.state["heartbeat"] = utc_now()
        self.state["elapsed_seconds_this_session"] = round(el, 1)
        self.state["elapsed_seconds_total"] = round(el + float(self.state.get("previous_elapsed_seconds", 0.0)), 1)
        rss = process_rss()
        mem = memory_bytes()
        self.state["memory"] = {"process_gb": round(rss / 2**30, 2) if rss else None,
                                "available_gb": round(mem["available"] / 2**30, 1) if mem.get("available") else None}
        try:
            D.write_json(self.out / "RUN_STATUS.json", self.state)
        except OSError:
            pass

    def start(self) -> None:
        def beat() -> None:
            while not self._stop.wait(self.hb):
                with self.lock:
                    self._write()

        self._thread = threading.Thread(target=beat, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def timing(self, row: dict[str, Any]) -> None:
        p = self.out / "RUN_TIMINGS.csv"
        cols = ["finished_at", "stage", "item", "family", "feature_set", "outer", "seconds", "n_configs", "device", "status"]
        line = ",".join(str(row.get(c, "")).replace(",", ";") for c in cols)
        with self.lock:
            new = not p.exists()
            with open(p, "a", encoding="utf-8", newline="\n") as fh:
                if new:
                    fh.write(",".join(cols) + "\n")
                fh.write(line + "\n")


def _timings(out: Path) -> pd.DataFrame:
    p = out / "RUN_TIMINGS.csv"
    if not p.is_file():
        return pd.DataFrame(columns=["stage", "family", "seconds", "status"])
    try:
        return pd.read_csv(p)
    except Exception:  # noqa: BLE001
        return pd.DataFrame(columns=["stage", "family", "seconds", "status"])


def eta_seconds(out: Path, remaining: list[tuple[str, str]], cfg: Any) -> float | None:
    """Mean seconds of the finished items of the same stage and family; otherwise scaled from PRIMARY of the family; otherwise ESTIMATE.json."""
    t = _timings(out)
    t = t[t.get("status", pd.Series(dtype=str)) == "done"] if len(t) else t
    est = {}
    ep = out / "ESTIMATE.json"
    if ep.is_file():
        try:
            est = json.loads(ep.read_text(encoding="utf-8")).get("per_item_seconds", {})
        except Exception:  # noqa: BLE001
            est = {}
    n_trials = max(1, int(cfg.budget["xgb"]["n_trials"]))
    total, known = 0.0, True
    for stage, fam in remaining:
        m = t[(t["stage"] == stage) & (t["family"] == fam)]["seconds"] if len(t) else pd.Series(dtype=float)
        if len(m):
            total += float(m.mean())
            continue
        pm = t[(t["stage"] == "PRIMARY") & (t["family"] == fam)]["seconds"] if len(t) else pd.Series(dtype=float)
        if len(pm):
            base = float(pm.mean())
            scale = {"FINAL": 1.25, "DOMAIN": (1.0 / n_trials + 0.1) if fam == "XGB" else 1.0, "ABLATION": (1.0 / n_trials + 0.1) if fam == "XGB" else 1.0,
                     "EXPLAIN": 0.5, "STABILITY": 1.0}.get(stage, 1.0)
            total += base * scale
        elif f"{stage}|{fam}" in est:
            total += float(est[f"{stage}|{fam}"])
        else:
            known = False
    return total if known else None


# ============================================================================ guards and the plan
def guard_out(out: Path, src: Path | None, allow_synced_folder: bool) -> None:
    for m in EARLIER_PHASE_MARKERS:
        if (out / m).exists():
            raise Phase2Stop("OUT_IS_EARLIER_PHASE", f"--out holds an earlier phase's run ({m}): Phase 5 never writes into a Phase 2 / 3 / 4 folder",
                             ["choose a NEW, empty folder for Phase 5"])
    if src is not None and src.resolve().parent == out.resolve():
        raise Phase2Stop("OUT_HOLDS_INPUT", "--out must not be the folder that holds the input file")
    synced = D.synced_folder(out.parent if not out.exists() else out)
    if synced and not allow_synced_folder:
        raise Phase2Stop("SYNCED_FOLDER", f"--out is inside a synchronised folder ({synced})", ["choose a local folder, or pass --allow-synced-folder (recorded)"])


def code_sha() -> str:
    from falls_ml.artifacts import source_tree_sha256

    return source_tree_sha256()


def build_sets(reg: pd.DataFrame, meta: dict[str, Any], cfg: Any) -> dict[str, Any]:
    el = cfg["eligibility"]
    prim, ver, low = set(el["primary_classes"]), set(el["verified_only_classes"]), set(el["low_availability_risk_levels"])
    r = reg[reg["feature"].isin(meta)]
    old = [f for f in r.loc[(r["origin"] == "OLD_PHASE3_UNIVERSE") & r["class"].isin(prim), "feature"]]
    newr = r[(r["origin"] == "NEW_V21") & r["class"].isin(prim)]
    new_safe = list(newr["feature"])
    new_ver = list(newr.loc[newr["class"].isin(ver), "feature"])
    new_low = list(newr.loc[newr["availability_risk"].isin(low), "feature"])
    sets: dict[str, list[str]] = {SET_OLD: old, SET_NEW: old + new_safe, SET_VERIFIED: old + new_ver, SET_LOWRISK: old + new_low}
    kinds = {SET_OLD: "PRIMARY", SET_NEW: "PRIMARY", SET_VERIFIED: "SENSITIVITY", SET_LOWRISK: "SENSITIVITY"}
    domains, ablations = {}, {}
    for dom in cfg["domains"]:
        add = list(newr.loc[newr["domain"] == dom, "feature"])
        domains[dom] = {"name": f"OLD_PLUS_{dom}", "added": add}
        if add:
            sets[f"OLD_PLUS_{dom}"] = old + add
            kinds[f"OLD_PLUS_{dom}"] = "DOMAIN"
    for ab, spec in cfg["ablations"].items():
        drop = set(spec.get("features") or [])
        drop |= {f for f in sets[SET_NEW] if meta[f]["domain"] in set(spec.get("domains") or [])}
        if spec.get("process_flag"):
            drop |= {f for f in sets[SET_NEW] if meta[f].get("process")}
        removed = [f for f in sets[SET_NEW] if f in drop]
        ablations[ab] = {"name": f"{SET_NEW}__{ab}", "removed": removed}
        if removed:
            sets[f"{SET_NEW}__{ab}"] = [f for f in sets[SET_NEW] if f not in drop]
            kinds[f"{SET_NEW}__{ab}"] = "ABLATION"
    canon: dict[tuple[str, ...], str] = {}
    alias: dict[str, str] = {}
    for name, feats in sets.items():
        key = tuple(feats)
        if key in canon:
            alias[name] = canon[key]
        else:
            canon[key] = name
            alias[name] = name
    return {"sets": sets, "kinds": kinds, "alias": alias, "domains": domains, "ablations": ablations,
            "n_new_safe": len(new_safe), "n_new_verified": len(new_ver), "n_new_lowrisk": len(new_low), "n_old": len(old)}


def x_guard(sets: dict[str, list[str]], reg: pd.DataFrame, kinds: dict[str, str], sealed: dict[str, str], frame_cols: list[str], cfg: Any) -> None:
    """HARD STOP if anything outcome / future / ineligible could reach X (checked at the preflight and again before every resumed run)."""
    el = cfg["eligibility"]
    prim, ver, low = set(el["primary_classes"]), set(el["verified_only_classes"]), set(el["low_availability_risk_levels"])
    info = reg.set_index("feature")
    bad = [c for c in frame_cols if c in sealed]
    for name, feats in sets.items():
        for f in feats:
            if f not in info.index:
                bad.append(f"{name}: {f} is not a catalogued feature")
                continue
            r = info.loc[f]
            raw = [c.strip() for c in str(r["raw_columns"]).split(";") if c.strip()]
            if any(c in sealed for c in raw):
                bad.append(f"{name}: {f} reads a sealed outcome / future column")
            if r["class"] not in prim:
                bad.append(f"{name}: {f} has class {r['class']} (only {sorted(prim)} may enter X)")
            if r["origin"] == "NEW_V21" and name == SET_VERIFIED and r["class"] not in ver:
                bad.append(f"{name}: {f} is not {sorted(ver)}")
            if r["origin"] == "NEW_V21" and name == SET_LOWRISK and r["availability_risk"] not in low:
                bad.append(f"{name}: {f} has availability risk {r['availability_risk']}")
            if r["origin"] == "NEW_V21" and name == SET_OLD:
                bad.append(f"{name}: the OLD set holds the new feature {f}")
    if bad:
        raise Phase2Stop("X_LEAKAGE", "an outcome / future / ineligible field would enter X: modelling refused", bad[:30])


# ============================================================================ preflight
def _sup(v: Any, k: int = 10) -> Any:
    return "<10" if isinstance(v, (int, np.integer)) and 0 < int(v) < k else v


def run_preflight(src: Path, out: Path, cfg: Any, L: dict[str, Any], mon: Monitor, *, synthetic: bool = False) -> dict[str, Any]:
    from falls_ml.phase4.common import input_identity
    from falls_ml.phase5.data import prepare

    d = dirs(out)
    for k in ("preflight", "work"):
        d[k].mkdir(parents=True, exist_ok=True)
    mon.update(status="RUNNING", stage="PREFLIGHT")
    mon.log(f"PREFLIGHT: reading {src.name} (X and outcome separately; outcome / future columns never enter X)")
    ident = input_identity(src)
    P = prepare(src, cfg, L, input_info=ident)
    res: dict[str, Any] = {"safe": P.safe, "checks": P.checks, "facts": P.facts, "input": {"name": ident["name"], "sha256": ident["sha256"], "bytes": ident["bytes"]}}
    sets_info: dict[str, Any] = {}
    if P.safe:
        sets_info = build_sets(P.registry, P.meta, cfg)
        try:
            x_guard(sets_info["sets"], P.registry, sets_info["kinds"], P.sealed, list(P.frame.columns), cfg)
            P.add("P8", "X guard: no outcome / future / ineligible field in any feature set (HARD)", "OK",
                  [f"{len(sets_info['sets'])} feature sets checked; OLD {sets_info['n_old']} features; new eligible: {sets_info['n_new_safe']} "
                   f"(verified-only {sets_info['n_new_verified']}, low availability risk {sets_info['n_new_lowrisk']})"])
        except Phase2Stop as exc:
            P.add("P8", "X guard: no outcome / future / ineligible field in any feature set (HARD)", "STOP", exc.details)
        if P.safe and sets_info["n_new_safe"] == 0:
            P.add("P9", "eligible new V21 predictors", "WARN", ["no new V21 predictor is eligible: OLD_PLUS_NEW_SAFE = OLD; the run reports "
                                                                "NO ELIGIBLE NEW FEATURES (see NEW_FEATURE_CATALOGUE.csv / V21_UNDECLARED_COLUMNS.csv)"])
    res["safe"], res["checks"] = P.safe, P.checks
    _write_preflight(out, P, res, sets_info, cfg)
    if not P.safe:
        return res
    order = np.argsort(P.frame["row_key"].to_numpy(), kind="mergesort")
    frame = P.frame.iloc[order].reset_index(drop=True)
    y = P.y[order].astype(int)
    folds = outer_folds(y, cfg.outer_folds, int(cfg["seed"]))
    D.write_parquet(d["work"] / "ANALYSIS_FRAME.parquet", frame)
    D.write_npz(d["work"] / "Y_FOLDS.npz", y=y, outer=folds)
    D.write_parquet(d["work"] / "FOLDS.parquet", pd.DataFrame({"row_key": frame["row_key"], "outer_fold": folds}))
    D.write_json(d["work"] / "META.json", P.meta)
    D.write_csv(d["work"] / "REGISTRY.csv", P.registry)
    D.write_json(d["work"] / "SEALED.json", P.sealed)
    plan = {"phase5_version": PHASE5_VERSION, "synthetic": bool(synthetic), "falls_ml_version": __import__("falls_ml").__version__, "design": DESIGN_LABEL, "mode": cfg.mode,
            "config_sha256": cfg.sha256, "config_path": Path(cfg.path).name, "v21_catalogue_sha256": cfg.v21.sha256, "overrides": cfg.overrides,
            "phase3_definitions": {"catalogue_sha256": L["cat"].sha256, "contract_sha256": L["contract"].content_sha256, "mapping_sha256": L["mapping"].content_sha256},
            "input": res["input"], "code_sha256": code_sha(), "seed": int(cfg["seed"]), "created_at": utc_now(), "n": int(len(y)), "events": int(y.sum()),
            "cv": {"outer_folds": cfg.outer_folds, "inner_folds": cfg.inner_folds}, "families": list(FAMILIES),
            "frame_sha256": D.sha256_file(d["work"] / "ANALYSIS_FRAME.parquet"), "folds_sha256": D.sha256_file(d["work"] / "FOLDS.parquet"),
            "fold_sizes": {str(k): int((folds == k).sum()) for k in range(cfg.outer_folds)},
            "fold_events": {str(k): int(y[folds == k].sum()) for k in range(cfg.outer_folds)}, **{k: v for k, v in sets_info.items()}}
    D.write_json(d["work"] / PLAN, plan)
    return res


def _write_preflight(out: Path, P: Any, res: dict[str, Any], sets_info: dict[str, Any], cfg: Any) -> None:
    from falls_ml.phase4.report import suppress_obj

    d = dirs(out)["preflight"]
    verdict = SAFE_LINE if P.safe else STOP_LINE
    D.write_json(d / "PREFLIGHT_RESULT.json", suppress_obj({**res, "verdict": verdict}))
    if P.outcome:
        D.write_json(d / "OUTCOME_CONTRACT_2026.json", suppress_obj(P.outcome))
    cf = {"index_date": cfg["index_date"], "design": DESIGN_LABEL, **P.facts.get("cohort", {}), "usable": P.facts.get("usable", {}),
          "definition_version": P.facts.get("definition_version", {}), "sealed_columns_count": P.facts.get("sealed", {}).get("n_sealed"),
          "sealed_by_reason": P.facts.get("sealed", {}).get("by_reason", {}), "v3_attestation": P.facts.get("v3_attestation", {}),
          "outcome_extra_audit": (P.outcome or {}).get("extra_audit", {})}
    D.write_json(d / "COHORT_FACTS_2026.json", suppress_obj(cf))
    if len(P.registry):
        reg = P.registry.copy()
        mem = {f: [] for f in reg["feature"]}
        for name, feats in sets_info.get("sets", {}).items():
            for f in feats:
                mem.setdefault(f, []).append(name)
        reg["feature_sets"] = reg["feature"].map(lambda f: "; ".join(mem.get(f, [])))
        reg["n_unknown_cells"] = reg["n_unknown_cells"].map(_sup)
        reg["n_known_observed"] = reg["n_known_observed"].map(_sup)
        D.write_csv(d / "FEATURE_ELIGIBILITY.csv", reg)
    if len(P.catalogue):
        D.write_csv(d / "NEW_FEATURE_CATALOGUE.csv", P.catalogue)
    D.write_csv(d / "V21_UNDECLARED_COLUMNS.csv", P.undeclared if len(P.undeclared) else
                pd.DataFrame(columns=["column", "suggested_domain", "missing_pct", "numeric_share", "n_distinct", "reason"]))
    if sets_info:
        D.write_json(d / "FEATURE_SETS.json", {"sets": {k: {"n_features": len(v), "kind": sets_info["kinds"].get(k), "alias_of": sets_info["alias"].get(k),
                                                            "features": v} for k, v in sets_info["sets"].items()},
                                               "domains": sets_info["domains"], "ablations": sets_info["ablations"]})
    lines = [f"# Phase 5 preflight - {DESIGN_LABEL}", "", f"Input: `{res['input']['name']}` (sha256 {res['input']['sha256'][:16]}…)", "",
             "| check | status | details |", "|---|---|---|"]
    for c in P.checks:
        lines.append(f"| {c['id']} {c['title']} | **{c['status']}** | " + "<br>".join(str(x).replace("|", "/") for x in c["details"]) + " |")
    lines += ["", "Outcome / future / follow-up columns are read only as the label and for the outcome-contract audit; they never enter X.",
              "Feature classes: SAFE_VERIFIED, SAFE_BOUNDED, SAFE_ATTESTED (eligible under the Phase 3 rule) / INELIGIBLE_TIMING, _SEMANTICS, _DATA, "
              "_LEAKAGE. See FEATURE_ELIGIBILITY.csv, NEW_FEATURE_CATALOGUE.csv, V21_UNDECLARED_COLUMNS.csv.", "", verdict, ""]
    D.write_str(d / "PHASE5_PREFLIGHT.md", "\n".join(lines))


# ============================================================================ the run
def _load_ctx(out: Path, cfg: Any, plan: dict[str, Any], jobs: int, state: DeviceState, mon: Monitor) -> Ctx:
    w = dirs(out)["work"]
    frame = pd.read_parquet(w / "ANALYSIS_FRAME.parquet")
    z = np.load(w / "Y_FOLDS.npz")
    meta = json.loads((w / "META.json").read_text(encoding="utf-8"))
    canon = {plan["alias"][k]: v for k, v in plan["sets"].items() if plan["alias"][k] == k}
    sets = {k: canon[plan["alias"][k]] for k in plan["sets"]}
    return Ctx(out=out, frame=frame, y=z["y"].astype(int), meta=meta, sets=sets, outer=z["outer"].astype(int), cfg=cfg, seed=int(plan["seed"]), jobs=jobs,
               state=state, log=mon.log)


def plan_units(plan: dict[str, Any], cfg: Any) -> dict[str, list[UnitSpec]]:
    K = int(plan["cv"]["outer_folds"])
    alias, kinds = plan["alias"], plan["kinds"]
    canon = [s for s in plan["sets"] if alias[s] == s]
    tun = cfg["derived_tuning"]
    out: dict[str, list[UnitSpec]] = {"PRIMARY": [], "FINAL": [], "DOMAIN": [], "ABLATION": []}
    prim = [s for s in canon if kinds.get(s) in ("PRIMARY", "SENSITIVITY")]
    for s in prim:
        for fam in FAMILIES:
            for k in range(K):
                out["PRIMARY"].append(UnitSpec(uid=f"PRIMARY|{fam}|{s}|outer{k}", stage="PRIMARY", family=fam, setname=s, outer=k))
    for fam in FAMILIES:
        for s in dict.fromkeys([alias[SET_OLD], alias[SET_NEW]]):
            out["FINAL"].append(UnitSpec(uid=f"FINAL|{fam}|{s}|all", stage="FINAL", family=fam, setname=s, outer=-1))
    for stage, ref_set in (("DOMAIN", alias[SET_OLD]), ("ABLATION", alias[SET_NEW])):
        for s in [s for s in canon if kinds.get(s) == stage]:
            for fam in FAMILIES:
                for k in range(K):
                    t = tun[fam]
                    out[stage].append(UnitSpec(uid=f"{stage}|{fam}|{s}|outer{k}", stage=stage, family=fam, setname=s, outer=k, tuning=t,
                                               reference=f"PRIMARY|{fam}|{ref_set}|outer{k}" if t == "reuse_reference" else None))
    return out


def run_phase5(input_path: str | Path | None, out_dir: str | Path, *, mode: str = "overnight", device: str = "auto", jobs: int | None = None,
               resume: bool = False, preflight_only: bool = False, report_only: bool = False, accept_code_change: str | None = None,
               allow_synced_folder: bool = False, config_path: str | Path | None = None, overrides: dict[str, Any] | None = None,
               v21_catalogue: str | Path | None = None, max_items: int | None = None, synthetic: bool = False) -> dict[str, Any]:
    """Returns {"status": ..., "exit_code": ...}. ``max_items`` (tests only) stops after that many newly computed units / tasks."""
    from falls_ml.phase3.runner import load_all
    from falls_ml.phase5.resources import default_jobs, environment, limit_threads, resolve_device

    out = Path(out_dir)
    src = Path(input_path) if input_path else None
    cfg = load_phase5_config(config_path or "configs/meuhedet/phase5.yaml", mode=mode, v21_catalogue=v21_catalogue, overrides=overrides)
    guard_out(out, src, allow_synced_folder)
    out.mkdir(parents=True, exist_ok=True)
    d = dirs(out)
    mon = Monitor(out, float(cfg["resources"]["heartbeat_seconds"]))
    limit_threads()
    jobs = int(jobs) if jobs else default_jobs(cfg)
    plan_path = d["work"] / PLAN
    L = load_all(cfg["phase3_config"])
    try:
        mon.start()
        if not plan_path.is_file():
            if src is None:
                raise Phase2Stop("NO_INPUT", "this folder has no Phase 5 plan yet: --input <2026 extract> is required")
            res = run_preflight(src, out, cfg, L, mon, synthetic=synthetic)
            line = SAFE_LINE if res["safe"] else STOP_LINE
            mon.log(f"PREFLIGHT finished: {line}")
            if not res["safe"]:
                mon.update(status="STOPPED_PREFLIGHT", stage="PREFLIGHT", verdict=STOP_LINE,
                           stop_reasons=[f"{c['id']} {c['title']}" for c in res["checks"] if c["status"] == "STOP"])
                print(STOP_LINE)
                return {"status": "STOPPED_PREFLIGHT", "exit_code": 2}
            mon.update(verdict=SAFE_LINE)
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        _verify_plan(plan, cfg, src, accept_code_change, mon, report_only=report_only)
        if preflight_only:
            mon.update(status="PREFLIGHT_COMPLETE", stage="PREFLIGHT", verdict=SAFE_LINE)
            print(SAFE_LINE)
            return {"status": "PREFLIGHT_COMPLETE", "exit_code": 0}
        units = plan_units(plan, cfg)
        started = any(is_complete(_ctx_stub(out), u) for us in units.values() for u in us) if not report_only else True
        if started and not resume and not report_only:
            raise Phase2Stop("RUN_EXISTS", "this folder already holds finished Phase 5 units: pass --resume to continue it (nothing finished is recomputed)")
        dev = resolve_device(device) if not report_only else {"requested": device, "device": "cpu", "reason": "report only"}
        env = environment(cfg, jobs, dev)
        D.write_json(d["work"] / f"ENVIRONMENT_session{mon.state['sessions']}.json", env)
        D.write_json(out / "ENVIRONMENT.json", env)
        state = DeviceState(dev["device"], log=mon.log)
        ctx = _load_ctx(out, cfg, plan, jobs, state, mon)
        x_guard(ctx.sets, pd.read_csv(d["work"] / "REGISTRY.csv"), plan["kinds"], json.loads((d["work"] / "SEALED.json").read_text(encoding="utf-8")),
                list(ctx.frame.columns), cfg)
        mon.update(status="RUNNING", mode=cfg.mode, jobs=jobs, device=dev, design=DESIGN_LABEL, run_started_at=mon.state.get("run_started_at") or utc_now())
        mon.log(f"Phase 5 {cfg.mode} run: {plan['n']} patients, {plan['events']} events; jobs {jobs}; XGBoost device {dev['device']} ({dev['reason']})")
        if report_only:
            from falls_ml.phase5.report import build_reports

            r = build_reports(ctx, plan, cfg, src=src, interim=False, mon=mon)
            mon.update(status="REPORT_COMPLETE", stage="REPORT", report=r)
            return {"status": "REPORT_COMPLETE", "exit_code": 0, "report": r}
        status = _execute(ctx, plan, cfg, units, mon, src, max_items=max_items)
        return {"status": status, "exit_code": 0 if status.startswith("COMPLETE") else 3}
    except KeyboardInterrupt:
        mon.log("INTERRUPTED (Ctrl+C): every committed unit is kept; continue with the same command and --resume")
        mon.update(status="INTERRUPTED", interrupted_at=utc_now())
        return {"status": "INTERRUPTED", "exit_code": 130}
    except Phase2Stop as exc:
        mon.log(str(exc))
        mon.update(status="STOPPED", stop_gate=exc.gate, stop_message=exc.message, stop_details=exc.details[:20])
        raise
    finally:
        mon.stop()


class _Stub:
    def __init__(self, out: Path):
        self.units_dir = out / "work" / "units"


def _ctx_stub(out: Path) -> Any:
    return _Stub(out)


def _verify_plan(plan: dict[str, Any], cfg: Any, src: Path | None, accept_code_change: str | None, mon: Monitor, *, report_only: bool) -> None:
    probs = []
    if plan["config_sha256"] != cfg.sha256:
        probs.append("configs/meuhedet/phase5.yaml (or the test overrides) differs from the settings frozen in the plan")
    if plan.get("v21_catalogue_sha256") != cfg.v21.sha256:
        probs.append("the V21 catalogue differs from the one frozen in the plan")
    if plan["mode"] != cfg.mode:
        probs.append(f"the plan was made for --mode {plan['mode']}; this call asks for --mode {cfg.mode} (use a separate --out folder per mode)")
    if src is not None:
        from falls_ml.data.dataset import sha256_file

        if sha256_file(src) != plan["input"]["sha256"]:
            probs.append("the input file differs from the file of the plan (sha256)")
    if probs:
        raise Phase2Stop("PLAN_MISMATCH", "this folder belongs to a different Phase 5 plan", probs)
    cs = code_sha()
    if cs != plan["code_sha256"]:
        if not accept_code_change:
            raise Phase2Stop("CODE_CHANGED", "the falls_ml code differs from the code that made the plan",
                             ["finished units stay as they are; pass --accept-code-change \"<reason>\" to continue with this code (recorded)"])
        D.append_jsonl(dirs(Path(mon.out))["logs"] / "code_changes.jsonl", {"ts": utc_now(), "reason": accept_code_change, "plan_code": plan["code_sha256"], "code": cs})
        mon.log(f"code change accepted: {accept_code_change}")


def _execute(ctx: Ctx, plan: dict[str, Any], cfg: Any, units: dict[str, list[UnitSpec]], mon: Monitor, src: Path | None, *,
             max_items: int | None) -> str:
    from falls_ml.phase5.explain import run_fold_explain, run_stability, task_done
    from falls_ml.phase5.report import build_reports

    budget = cfg.budget
    K = int(plan["cv"]["outer_folds"])
    computed = [0]
    alias = plan["alias"]
    explain_tasks = [(fam, s) for fam in FAMILIES for s in dict.fromkeys([alias[SET_OLD], alias[SET_NEW]])]
    stab_tasks = [(fam, alias[SET_NEW]) for fam in FAMILIES]

    def remaining() -> list[tuple[str, str]]:
        rem = [(u.stage, u.family) for st in ("PRIMARY", "FINAL", "DOMAIN", "ABLATION") for u in units[st] if not is_complete(ctx, u)]
        rem += [("EXPLAIN", f) for f, s in explain_tasks if not task_done(ctx, f"FOLDS|{f}|{s}")]
        rem += [("STABILITY", f) for f, s in stab_tasks if not task_done(ctx, f"STABILITY|{f}|{s}")]
        return rem

    def counts() -> dict[str, Any]:
        c = {st: {"done": sum(is_complete(ctx, u) for u in units[st]), "total": len(units[st])} for st in units}
        c["EXPLAIN"] = {"done": sum(task_done(ctx, f"FOLDS|{f}|{s}") for f, s in explain_tasks), "total": len(explain_tasks)}
        c["STABILITY"] = {"done": sum(task_done(ctx, f"STABILITY|{f}|{s}") for f, s in stab_tasks), "total": len(stab_tasks)}
        pf = {k: sum(1 for u in units["PRIMARY"] if u.outer == k and is_complete(ctx, u)) for k in range(K)}
        c["outer_folds_complete"] = sum(1 for k in range(K) if pf[k] == sum(1 for u in units["PRIMARY"] if u.outer == k))
        c["outer_folds_total"] = K
        return c

    def refresh(**kw: Any) -> None:
        e = eta_seconds(ctx.out, remaining(), cfg)
        mon.update(progress=counts(), eta_seconds=round(e) if e is not None else None,
                   eta_at=time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() + e)) if e is not None else None, **kw)

    def limit() -> bool:
        return max_items is not None and computed[0] >= max_items

    def do(stage: str, item: str, fam: str, setname: str, outer: Any, fn: Any, done: Any) -> bool:
        """Run one item with one retry; a failure is recorded and the night continues."""
        if done():
            return True
        if limit():
            return False
        current = {"stage": stage, "item": item}
        ctx.progress = (lambda **kw: mon.update(current={**current, **kw}))
        refresh(stage=stage, current=current)
        for attempt in (1, 2):
            t0 = time.time()
            try:
                info = fn()
                sec = round(time.time() - t0, 1)
                computed[0] += 1
                mon.timing({"finished_at": utc_now(), "stage": stage, "item": item, "family": fam, "feature_set": setname, "outer": outer, "seconds": sec,
                            "n_configs": (info or {}).get("n_configs", ""), "device": ctx.state.device if fam == "XGB" else "cpu", "status": "done"})
                mon.update(last_checkpoint={"item": item, "at": utc_now()},
                           failures=[f for f in mon.state.get("failures", []) if f.get("item") != item])     # a later success clears it
                e = eta_seconds(ctx.out, remaining(), cfg)
                mon.log(f"{stage} {item} done in {sec:.1f}s" + (f" | ETA {e / 3600:.1f} h" if e is not None else ""))
                return True
            except (KeyboardInterrupt, Phase2Stop):
                raise
            except Exception as exc:  # noqa: BLE001 - recorded; the overnight run continues with the next item
                tb = traceback.format_exc(limit=3)
                mon.log(f"{stage} {item} FAILED (attempt {attempt}): {type(exc).__name__}: {str(exc)[:300]}")
                mon.timing({"finished_at": utc_now(), "stage": stage, "item": item, "family": fam, "feature_set": setname, "outer": outer,
                            "seconds": round(time.time() - t0, 1), "status": f"failed_attempt{attempt}"})
                if attempt == 1:
                    mon.update(retries=int(mon.state.get("retries", 0)) + 1)
                    continue
                fl = [f for f in mon.state.get("failures", []) if f.get("item") != item]
                fl.append({"item": item, "error": f"{type(exc).__name__}: {str(exc)[:300]}", "trace": tb[-800:], "at": utc_now()})
                mon.update(failures=fl)
        return False

    for st in ("PRIMARY",):
        for u in units[st]:
            do(st, u.uid, u.family, u.setname, u.outer, lambda u=u: run_unit(ctx, u), lambda u=u: is_complete(ctx, u))
    if limit():
        refresh(status="PAUSED_TEST_LIMIT")
        return "PAUSED_TEST_LIMIT"
    if all(is_complete(ctx, u) for u in units["PRIMARY"]) and not (dirs(ctx.out)["work"] / "INTERIM_DONE").exists():
        mon.log("INTERIM REPORT: the primary OLD vs OLD_PLUS_NEW_SAFE comparison is complete - writing share/ now (updated at the end)")
        refresh(stage="INTERIM_REPORT")
        build_reports(ctx, plan, cfg, src=src, interim=True, mon=mon)
        D.write_str(dirs(ctx.out)["work"] / "INTERIM_DONE", utc_now())
    for st in ("FINAL", "DOMAIN", "ABLATION"):
        for u in units[st]:
            if u.reference and not is_complete(ctx, UnitSpec(uid=u.reference, stage="PRIMARY", family=u.family, setname="", outer=u.outer)):
                mon.log(f"{st} {u.uid} skipped: its reference unit {u.reference} is not complete")
                continue
            do(st, u.uid, u.family, u.setname, u.outer, lambda u=u: run_unit(ctx, u), lambda u=u: is_complete(ctx, u))
        if limit():
            refresh(status="PAUSED_TEST_LIMIT")
            return "PAUSED_TEST_LIMIT"
    for fam, s in explain_tasks:
        ok = all(is_complete(ctx, UnitSpec(uid=f"PRIMARY|{fam}|{s}|outer{k}", stage="PRIMARY", family=fam, setname=s, outer=k)) for k in range(K))
        if ok:
            do("EXPLAIN", f"FOLDS|{fam}|{s}", fam, s, "all",
               lambda fam=fam, s=s: run_fold_explain(ctx, fam, s, K, repeats=int(budget["permutation_repeats"]), shap_rows=int(budget["shap_rows"])),
               lambda fam=fam, s=s: task_done(ctx, f"FOLDS|{fam}|{s}"))
    for fam, s in stab_tasks:
        if is_complete(ctx, UnitSpec(uid=f"FINAL|{fam}|{s}|all", stage="FINAL", family=fam, setname=s, outer=-1)):
            reps = int(budget["stability_linear"] if fam != "XGB" else budget["stability_xgb"])
            do("STABILITY", f"STABILITY|{fam}|{s}", fam, s, "all", lambda fam=fam, s=s, reps=reps: run_stability(ctx, fam, s, replicates=reps),
               lambda fam=fam, s=s: task_done(ctx, f"STABILITY|{fam}|{s}"))
    if limit():
        refresh(status="PAUSED_TEST_LIMIT")
        return "PAUSED_TEST_LIMIT"
    refresh(stage="REPORT", current={"stage": "REPORT"})
    mon.log("REPORT: aggregate tables, management / scientific summaries, figures, privacy scan")
    r = build_reports(ctx, plan, cfg, src=src, interim=False, mon=mon)
    rem = remaining()
    failures = mon.state.get("failures", []) if rem else []          # every item finished: earlier failures were all recovered
    mon.update(failures=failures)
    status = "COMPLETE" if not failures and not rem else "COMPLETE_WITH_FAILURES"
    refresh(status=status, stage="DONE", current=None, finished_at=utc_now(), report=r)
    mon.log(f"Phase 5 {status}: share/ written ({r.get('files', '?')} files, privacy scan {'PASSED' if r.get('privacy_passed') else 'FAILED'})")
    return status
