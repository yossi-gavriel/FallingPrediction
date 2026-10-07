"""NEGATIVE CONTROLS on the real data (Phase 5.1 R-12, docs/phase6/EXPERIMENT1_IMPLEMENTATION_CONTRACT.md N-2): frozen-fold label permutations.

For each of the registered seeds the labels are permuted WITHIN each adopted outer fold (the fold assignment is frozen), and the COMPLETE unit
pipeline runs on the permuted labels: the per-fold coverage gate, the inner designs and lambda grids, the 2.2.0 selection objective, the outer
refit and the holdout prediction (ENET, the ADMISSIBLE set, the quick budget). Under this null the outer-OOF AUROC must sit near 0.50 and the
within-fold Recall@Top3 near 3% (the capacity itself). The registered sanity-stop thresholds (mean AUROC > 0.55 or mean Recall@Top3 > 5% over
the seeds -> STOP NEGATIVE_CONTROL_FAILED) are fixed in the settings and verified by the loader; a failure is a hard stop and no scientific
report of the run is published.

Everything stays local (work/negative_controls/seed<k>/); the share receives only the aggregate NEGATIVE_CONTROLS.csv.
"""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from falls_ml.artifacts import utc_now
from falls_ml.phase2 import durable as D
from falls_ml.phase2.state import Phase2Stop
from falls_ml.phase5.capacity import fold_capacity, n_selected, rank_model
from falls_ml.phase5.engine import Ctx, UnitSpec, is_complete, load_result, run_unit

RESULT = "NEGATIVE_CONTROLS_RESULT.json"
GATE = "NEGATIVE_CONTROL_FAILED"


def permute_within_folds(y: np.ndarray, outer: np.ndarray, seed: int) -> np.ndarray:
    """A permutation of the labels inside every outer fold (fold sizes and fold prevalences unchanged; the fold assignment is frozen)."""
    rng = np.random.default_rng(int(seed))
    yp = np.asarray(y).astype(int).copy()
    for f in np.unique(outer):
        m = np.flatnonzero(outer == f)
        yp[m] = yp[rng.permutation(m)]
    return yp


def result_path(out: Path) -> Path:
    return out / "work" / RESULT


def load_result_file(out: Path) -> dict[str, Any] | None:
    p = result_path(out)
    return __import__("json").loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def run_negative_controls(ctx: Ctx, plan: dict[str, Any], cfg: Any, *, mon: Any, seeds: int | None = None) -> dict[str, Any]:
    """Runs (or resumes) the permutation controls; writes work/NEGATIVE_CONTROLS_RESULT.json; STOPS on failure."""
    nc = cfg["negative_controls"]
    n_seeds = int(seeds if seeds is not None else nc["seeds"])
    fam, setname = str(nc.get("family", "ENET")), str(nc.get("set", "OLD_PLUS_NEW_SAFE"))
    canon = plan["alias"].get(setname, setname)
    permille = int(nc["capacity_permille"])
    lim_auc, lim_rec = float(nc["max_mean_auroc"]), float(nc["max_mean_recall_top3"])
    cfg_q = cfg.with_mode(str(nc.get("mode", "quick")))
    K = int(plan["cv"]["outer_folds"])
    N = len(ctx.y)
    T = n_selected(permille, N)
    from falls_ml.phase5.metrics import point_metrics

    rows = []
    t0 = time.time()
    for s in range(n_seeds):
        seed_s = int(plan["seed"]) + 100_000 + s
        yp = permute_within_folds(ctx.y, ctx.outer, seed_s)
        ctx_s = dataclasses.replace(ctx, y=yp, cfg=cfg_q, units_subdir=f"negative_controls/seed{s:02d}/units")
        p = np.full(N, np.nan)
        for k in range(K):
            spec = UnitSpec(uid=f"NEGCTRL{s:02d}|{fam}|{canon}|outer{k}", stage="PRIMARY", family=fam, setname=canon, outer=k)
            if not is_complete(ctx_s, spec):
                mon.update(current={"stage": "NEGATIVE_CONTROLS", "item": spec.uid, "seed": s, "n_seeds": n_seeds})
                run_unit(ctx_s, spec)
            r = load_result(ctx_s, spec)
            p[r["arrays"]["test_idx"]] = r["arrays"]["p_test"]
        if not np.isfinite(p).all():
            raise RuntimeError(f"negative control seed {s}: incomplete OOF predictions")
        pm = point_metrics(yp.astype(float), p)
        cap = fold_capacity(yp.astype(np.int64), rank_model(fam, canon, yp.astype(np.int64), ctx.outer.astype(int), p), T)
        rows.append({"seed_index": s, "seed": seed_s, "family": fam, "feature_set": setname, "n": N, "events": int(yp.sum()), "selected_total": T,
                     "auroc": round(float(pm["auroc"]), 4), "ap": round(float(pm["ap"]), 4), "recall_top3": round(float(cap["sensitivity"]), 4),
                     "ppv_top3": round(float(cap["ppv"]), 4), "chance_recall_top3": round(T / N, 4)})
        mon.log(f"NEGATIVE CONTROL seed {s + 1}/{n_seeds}: permuted-label AUROC {pm['auroc']:.3f}, Recall@Top3 {cap['sensitivity']:.3f} (chance {T / N:.3f})")
    t = pd.DataFrame(rows)
    mean_auc, mean_rec = float(t["auroc"].mean()), float(t["recall_top3"].mean())
    passed = bool(mean_auc <= lim_auc and mean_rec <= lim_rec)
    res = {"passed": passed, "n_seeds": n_seeds, "family": fam, "feature_set": setname, "capacity_permille": permille, "selected_total": T,
           "mean_auroc": round(mean_auc, 4), "mean_recall_top3": round(mean_rec, 4), "max_mean_auroc": lim_auc, "max_mean_recall_top3": lim_rec,
           "mode": cfg_q.mode, "seconds": round(time.time() - t0, 1), "finished_at": utc_now(),
           "rule": "labels permuted within each frozen outer fold; the complete unit pipeline (coverage gate, inner designs and lambda grids, the "
                   "2.2.0 selection objective, outer refit, holdout prediction) re-run per seed; mean AUROC > max_mean_auroc OR mean Recall@Top3 > "
                   "max_mean_recall_top3 -> STOP (no scientific report)",
           "seeds": rows}
    w = ctx.out / "work"
    w.mkdir(parents=True, exist_ok=True)
    D.write_csv(w / "NEGATIVE_CONTROLS.csv", t)
    D.write_json(result_path(ctx.out), res)
    if not passed:
        raise Phase2Stop(GATE, f"the frozen-fold permutation controls FAILED: mean AUROC {mean_auc:.3f} (limit {lim_auc:g}), mean Recall@Top3 "
                         f"{mean_rec:.3f} (limit {lim_rec:g}) - the pipeline carries information under the null; nothing is fitted or reported until "
                         "the cause is found and a dated amendment records it", [f"seed {r['seed_index']}: AUROC {r['auroc']}, Recall@Top3 {r['recall_top3']}" for r in rows])
    return res


def require_passed(out: Path, *, synthetic: bool, n_required: int) -> dict[str, Any]:
    """Before PRIMARY (and before any report) on real data: the controls must exist, cover the registered number of seeds and have passed."""
    r = load_result_file(out)
    if r is None:
        if synthetic:
            return {"passed": None, "skipped": "synthetic run without --negative-controls"}
        raise Phase2Stop("NEGATIVE_CONTROLS_REQUIRED", "the frozen-fold permutation controls have not run on this folder: run the same command with "
                         "--negative-controls (quick budget) before the overnight run; a real run never fits or reports without them")
    if not r.get("passed"):
        raise Phase2Stop(GATE, "the negative controls of this folder FAILED: no model is fitted and no scientific report is published",
                         [f"mean AUROC {r.get('mean_auroc')}, mean Recall@Top3 {r.get('mean_recall_top3')}"])
    if not synthetic and int(r.get("n_seeds", 0)) < int(n_required):
        raise Phase2Stop("NEGATIVE_CONTROLS_REQUIRED", f"the controls ran with {r.get('n_seeds')} seeds; the registration requires {n_required}")
    return r
