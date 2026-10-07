"""The completed Phase 5 2.x run (PRE) that the Phase 5.1 repair (POST) is compared with - verified BEFORE anything is fitted, read only.

Phase 5.1 R-10 (docs/phase6/EXPERIMENT1_IMPLEMENTATION_CONTRACT.md): a real-data POST run refuses to fit unless every check below passes:

    (a) a completed PRE run exists: work/PLAN.json of Phase 5 2.x, RUN_STATUS.json COMPLETE*, share/RUN_MANIFEST.json of a FINAL report
    (b) the PRE input sha256 equals the POST input sha256 (the same 2026 extract, byte for byte)
    (c) the POST usable cohort equals the PRE cohort: the same pseudonymous row keys and the same labels (counts AND identity)
    (d) sha256(PRE work/FOLDS.parquet) equals folds_sha256 of the PRE plan; the POST run ADOPTS these outer folds - it never regenerates them
    (e) every PRE ENET PRIMARY unit of OLD / OLD_PLUS_ALL_NEW_ELIGIBLE / OLD_PLUS_NEW_SAFE verifies against its COMPLETE.json
    (f) share/TOP3_CAPACITY_PRIMARY.csv of the PRE run is REPRODUCED EXACTLY (selected, captured, false interventions per ENET arm at 3%) from
        the saved PRE unit arrays by the POST capacity code

Any failure stops with PRE_VERIFICATION_FAILED. The PRE folder is never written to: its digest is recorded at the preflight and re-checked at
report time (PRE_RUN_MODIFIED). Only the folder NAME and hashes are recorded (never a local path).
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from falls_ml.phase2 import durable as D
from falls_ml.phase2.state import Phase2Stop
from falls_ml.phase5 import PRE_MAJOR
from falls_ml.phase5.config import PRIMARY_FAMILY, SET_ALL, SET_OLD, SET_SAFE

GATE = "PRE_VERIFICATION_FAILED"
PRE_SETS = (SET_OLD, SET_ALL, SET_SAFE)
TOP3_TABLE = "TOP3_CAPACITY_PRIMARY.csv"
TARGET_PERMILLE = 30


class _Units:
    """The minimal context ``engine.is_complete`` / ``load_result`` need: a units directory."""

    def __init__(self, units_dir: Path):
        self.units_dir = units_dir


def _read_json(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def _fail(msg: str, details: list[str] | None = None) -> None:
    raise Phase2Stop(GATE, msg, details or [])


def pre_summary(pre: Path) -> dict[str, Any]:
    """(a) the PRE run is a completed Phase 5 2.x run with a FINAL report."""
    plan_p, status_p, man_p = pre / "work" / "PLAN.json", pre / "RUN_STATUS.json", pre / "share" / "RUN_MANIFEST.json"
    if not plan_p.is_file():
        _fail("the PRE folder has no work/PLAN.json: it is not a Phase 5 run folder")
    plan = _read_json(plan_p)
    ver = str(plan.get("phase5_version", ""))
    if ver.split(".")[0] != PRE_MAJOR:
        _fail(f"the PRE plan is Phase 5 {ver or '?'}; a completed Phase 5 {PRE_MAJOR}.x run is required as PRE")
    if not status_p.is_file():
        _fail("the PRE folder has no RUN_STATUS.json")
    status = _read_json(status_p)
    if not str(status.get("status", "")).startswith("COMPLETE"):
        _fail(f"the PRE run is not complete (RUN_STATUS.json status = {status.get('status')!r})")
    if not man_p.is_file():
        _fail("the PRE folder has no share/RUN_MANIFEST.json: its report was never published")
    man = _read_json(man_p)
    if str(man.get("report_kind", "")) != "FINAL":
        _fail(f"the PRE share holds a {man.get('report_kind')!r} report, not the FINAL one")
    for name in ("FOLDS.parquet", "Y_FOLDS.npz"):
        if not (pre / "work" / name).is_file():
            _fail(f"the PRE folder has no work/{name}")
    return {"plan": plan, "status": status, "manifest": man, "phase5_version": ver}


def pre_folds(pre: Path, plan: dict[str, Any]) -> pd.DataFrame:
    """(d) the PRE fold file, hash-verified against the PRE plan: row_key -> outer_fold (+ the PRE label, aligned from Y_FOLDS.npz)."""
    fp = pre / "work" / "FOLDS.parquet"
    sha = D.sha256_file(fp)
    if sha != str(plan.get("folds_sha256", "")):
        _fail("the PRE fold file does not match the fold hash frozen in the PRE plan (the PRE folds are not what the PRE run used)",
              [f"work/FOLDS.parquet sha256 {sha[:16]}… vs plan folds_sha256 {str(plan.get('folds_sha256', ''))[:16]}…"])
    folds = pd.read_parquet(fp)
    if not {"row_key", "outer_fold"} <= set(folds.columns):
        _fail("the PRE fold file lacks row_key / outer_fold columns")
    z = np.load(pre / "work" / "Y_FOLDS.npz")
    y, outer = z["y"].astype(int), z["outer"].astype(int)
    if len(y) != len(folds) or not np.array_equal(outer, folds["outer_fold"].to_numpy(dtype=int)):
        _fail("the PRE Y_FOLDS.npz does not align with work/FOLDS.parquet")
    out = folds[["row_key", "outer_fold"]].copy()
    out["y_pre"] = y
    return out


def verify_pre_run(pre: Path, *, post_input_sha256: str, post_row_keys: np.ndarray, post_y: np.ndarray, log: Any = print) -> dict[str, Any]:
    """Checks (a)-(f). Returns the PRE block for the POST plan and the outer folds to ADOPT, aligned to ``post_row_keys``."""
    pre = Path(pre)
    if not pre.is_dir():
        _fail(f"the PRE folder does not exist: {pre.name}")
    S = pre_summary(pre)
    plan = S["plan"]
    # (b) the same extract
    pre_sha = str((plan.get("input") or {}).get("sha256", ""))
    if pre_sha != str(post_input_sha256):
        _fail("the PRE run used a different input file (sha256 differs): PRE / POST must read the identical 2026 extract",
              [f"PRE input sha256 {pre_sha[:16]}…; POST input sha256 {str(post_input_sha256)[:16]}…"])
    # (c) + (d) cohort / labels / folds
    folds = pre_folds(pre, plan)
    keys_post = pd.Series(np.asarray(post_row_keys).astype(str))
    keys_pre = folds["row_key"].astype(str)
    if len(keys_pre) != len(keys_post) or set(keys_pre) != set(keys_post):
        only_pre, only_post = len(set(keys_pre) - set(keys_post)), len(set(keys_post) - set(keys_pre))
        _fail("COHORT_MISMATCH: the POST usable cohort differs from the PRE cohort (row keys)",
              [f"PRE {len(keys_pre)} patients, POST {len(keys_post)} patients; only in PRE {only_pre}, only in POST {only_post}"])
    aligned = folds.set_index("row_key").loc[keys_post.to_numpy()]
    y_pre = aligned["y_pre"].to_numpy(dtype=int)
    if not np.array_equal(y_pre, np.asarray(post_y).astype(int)):
        _fail("COHORT_MISMATCH: the labels of the POST cohort differ from the PRE labels for the same patients",
              [f"{int((y_pre != np.asarray(post_y).astype(int)).sum())} patients with a different label"])
    outer = aligned["outer_fold"].to_numpy(dtype=int)
    K = int(plan["cv"]["outer_folds"])
    if outer.min() < 0 or outer.max() != K - 1 or len(np.unique(outer)) != K:
        _fail("the PRE fold assignment is not a complete assignment into the PRE plan's outer folds")
    # (e) ENET PRIMARY units
    from falls_ml.phase5.engine import UnitSpec, is_complete

    U = _Units(pre / "work" / "units")
    alias = plan["alias"]
    missing = []
    for s in PRE_SETS:
        c = alias.get(s, s)
        for k in range(K):
            u = UnitSpec(uid=f"PRIMARY|{PRIMARY_FAMILY}|{c}|outer{k}", stage="PRIMARY", family=PRIMARY_FAMILY, setname=c, outer=k)
            if not is_complete(U, u):
                missing.append(u.uid)
    if missing:
        _fail(f"{len(missing)} required PRE ENET PRIMARY unit(s) are missing or fail their COMPLETE.json hashes", missing[:10])
    # (f) the PRE Top-3% table reproduced from the saved PRE arrays
    repro = reproduce_top3(pre, plan, keys_post=keys_post.to_numpy(), outer_post=outer, y_post=y_pre)
    log(f"PRE verified: Phase 5 {S['phase5_version']}, {len(keys_post)} patients, {int(y_pre.sum())} events, {K} folds adopted; "
        f"{TOP3_TABLE} reproduced for {len(repro['rows'])} ENET arm(s)")
    digest = pre_digest(pre)
    block = {"folder_name": pre.name, "phase5_version": S["phase5_version"], "falls_ml_version": plan.get("falls_ml_version"),
             "plan_sha256": D.sha256_file(pre / "work" / "PLAN.json"), "folds_sha256": str(plan.get("folds_sha256")), "frame_sha256": plan.get("frame_sha256"),
             "input_sha256": pre_sha, "code_sha256": plan.get("code_sha256"), "config_sha256": plan.get("config_sha256"), "seed": plan.get("seed"),
             "cv": plan.get("cv"), "n": int(plan.get("n", len(keys_post))), "events": int(plan.get("events", int(y_pre.sum()))),
             "alias": alias, "sets": {s: plan["sets"].get(s, []) for s in PRE_SETS if s in plan.get("sets", {})},
             "units_verified": [f"PRIMARY|{PRIMARY_FAMILY}|{alias.get(s, s)}|outer0..{K - 1}" for s in PRE_SETS],
             "top3_reproduced": repro["rows"], "digest_sha256": digest, "verified": True,
             "checks": ["a_completed_pre_run", "b_input_sha256", "c_cohort_and_labels", "d_fold_hash_adopted", "e_enet_units_complete",
                        "f_top3_capacity_table_reproduced"]}
    return {"pre": block, "outer": outer, "y": y_pre}


def pre_oof(pre: Path, plan: dict[str, Any], *, keys_post: np.ndarray, outer_post: np.ndarray, families: tuple[str, ...] = (PRIMARY_FAMILY,),
            targets: tuple[float, ...] = (0.70,)) -> dict[tuple[str, str], dict[str, Any]]:
    """The committed PRE outer-OOF predictions (and the PRE inner-selected thresholds / fold test rows) aligned to the POST row order."""
    from falls_ml.phase5.engine import UnitSpec, is_complete, load_result

    folds = pre_folds(pre, plan)
    pos = pd.Series(np.arange(len(folds)), index=folds["row_key"].astype(str))
    order = pos.loc[np.asarray(keys_post).astype(str)].to_numpy()        # PRE row -> POST row mapping
    inv = np.empty(len(order), dtype=int)
    inv[order] = np.arange(len(order))                                     # inv: PRE index -> POST index
    K = int(plan["cv"]["outer_folds"])
    U = _Units(pre / "work" / "units")
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for fam in families:
        for s in PRE_SETS:
            c = plan["alias"].get(s, s)
            specs = [UnitSpec(uid=f"PRIMARY|{fam}|{c}|outer{k}", stage="PRIMARY", family=fam, setname=c, outer=k) for k in range(K)]
            if not all(is_complete(U, u) for u in specs):
                continue
            p = np.full(len(order), np.nan)
            flags = {f"{t:.2f}": np.zeros(len(order), dtype=bool) for t in targets}
            fold_rows = []
            for u in specs:
                r = load_result(U, u)
                te_pre, pt = np.asarray(r["arrays"]["test_idx"]), np.asarray(r["arrays"]["p_test"], dtype=float)
                te = inv[te_pre]
                if not np.array_equal(np.sort(te), np.flatnonzero(outer_post == u.outer)):
                    _fail(f"PRE unit {u.uid}: its holdout rows differ from the adopted outer fold {u.outer}")
                p[te] = pt
                for t, thr in (r.get("thresholds") or {}).items():
                    if t in flags:
                        flags[t][te] = pt >= float(thr) if thr is not None and math.isfinite(float(thr)) else False
                fold_rows.append({"outer": u.outer, "test_idx": te, "thresholds": r.get("thresholds") or {}, "config": r.get("config") or {}})
            if not np.isfinite(p).all():
                _fail(f"PRE {fam} {s}: some patients have no finite outer-OOF prediction")
            out[(fam, s)] = {"p": p, "flags": flags, "folds": fold_rows, "canonical": c}
    return out


def reproduce_top3(pre: Path, plan: dict[str, Any], *, keys_post: np.ndarray, outer_post: np.ndarray, y_post: np.ndarray) -> dict[str, Any]:
    """(f) recompute the ENET rows of the PRE share/TOP3_CAPACITY_PRIMARY.csv from the saved PRE arrays with the POST capacity code."""
    from falls_ml.phase5.capacity import fold_capacity, n_selected, rank_model

    tp_ = pre / "share" / TOP3_TABLE
    if not tp_.is_file():
        _fail(f"the PRE share has no {TOP3_TABLE}: the PRE Top-3% primary result cannot be verified (run the 0.12.3 dashboard on the PRE folder first)")
    t = pd.read_csv(tp_)
    need = {"family", "feature_set", "selected_total", "tp", "fp"}
    if not need <= set(t.columns):
        _fail(f"{TOP3_TABLE} lacks the columns {sorted(need - set(t.columns))}")
    oof = pre_oof(pre, plan, keys_post=keys_post, outer_post=outer_post)
    y = np.asarray(y_post).astype(np.int64)
    N = len(y)
    T = n_selected(TARGET_PERMILLE, N)
    rows, problems = [], []
    for s in PRE_SETS:
        if (PRIMARY_FAMILY, s) not in oof:
            continue
        got = fold_capacity(y, rank_model(PRIMARY_FAMILY, s, y, outer_post, oof[(PRIMARY_FAMILY, s)]["p"]), T)
        sub = t[(t["family"].astype(str) == PRIMARY_FAMILY) & (t["feature_set"].astype(str) == s)]
        if "capacity_permille" in t.columns:
            sub = sub[pd.to_numeric(sub["capacity_permille"], errors="coerce") == TARGET_PERMILLE]
        if len(sub) != 1:
            problems.append(f"{s}: {len(sub)} rows in {TOP3_TABLE} for ENET at 3% (expected 1)")
            continue
        r = sub.iloc[0]
        row = {"feature_set": s, "selected_total_table": r["selected_total"], "tp_table": r["tp"], "fp_table": r["fp"],
               "selected_total_recomputed": int(got["flagged"]), "tp_recomputed": int(got["tp"]), "fp_recomputed": int(got["fp"])}
        for col, val in (("selected_total", got["flagged"]), ("tp", got["tp"]), ("fp", got["fp"])):
            v = pd.to_numeric(pd.Series([r[col]]), errors="coerce").iloc[0]
            if pd.isna(v):
                if col == "selected_total":
                    problems.append(f"{s}: selected_total is not numeric in {TOP3_TABLE}")
                continue                                   # a suppressed small cell ('<10') cannot be compared; selected_total always can
            if int(v) != int(val):
                problems.append(f"{s}: {col} in {TOP3_TABLE} = {int(v)} but recomputed from the PRE arrays = {int(val)}")
        row["reproduced"] = not any(p.startswith(f"{s}:") for p in problems)
        rows.append(row)
    if not rows:
        problems.append("no ENET arm could be reproduced")
    if problems:
        _fail(f"the PRE {TOP3_TABLE} is NOT reproduced exactly from the PRE unit arrays", problems[:10])
    return {"rows": rows, "selected_total": T, "n": N}


def pre_digest(pre: Path) -> str:
    """A digest of the PRE artefacts the POST run depends on (plan, folds, labels, unit commits / arrays, the whole share/): recorded at the
    preflight and re-checked at report time so that a changed PRE folder stops the run (PRE_RUN_MODIFIED)."""
    pre = Path(pre)
    files: list[Path] = [pre / "work" / "PLAN.json", pre / "work" / "FOLDS.parquet", pre / "work" / "Y_FOLDS.npz", pre / "RUN_STATUS.json"]
    units = pre / "work" / "units"
    if units.is_dir():
        for d in sorted(units.iterdir()):
            for name in ("COMPLETE.json", "arrays.npz", "result.json"):
                if (d / name).is_file():
                    files.append(d / name)
    share = pre / "share"
    if share.is_dir():
        files += sorted(p for p in share.rglob("*") if p.is_file())
    h = hashlib.sha256()
    for p in files:
        if p.is_file():
            h.update(p.relative_to(pre).as_posix().encode("utf-8"))
            h.update(b"\0")
            h.update(bytes.fromhex(D.sha256_file(p)))
            h.update(b"\n")
    return h.hexdigest()


def check_unmodified(pre: Path, recorded_digest: str) -> None:
    now = pre_digest(pre)
    if now != recorded_digest:
        raise Phase2Stop("PRE_RUN_MODIFIED", "the PRE folder changed since the POST preflight verified it (its digest differs): the PRE / POST comparison "
                         "cannot be trusted - restore the PRE folder and re-run the preflight in a new folder",
                         [f"recorded {recorded_digest[:16]}…, now {now[:16]}…"])
