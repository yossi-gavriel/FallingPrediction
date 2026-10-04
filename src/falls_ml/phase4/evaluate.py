"""``falls_ml meuhedet-phase4-evaluate``: opens the 2026 outcomes ONLY after every frozen hash was verified, checks the 2026 outcome contract, and
only then computes the temporal-validation metrics of the frozen predictions (nothing is re-fitted, re-calibrated or re-thresholded).

1. verification (HARD stops): BLIND_SCORE_COMPLETE.txt exists; the prediction manifest, the predictions, the frozen model manifest, the feature
   list, the Phase 3 internal estimate and the overlap file have their recorded sha256; the 2026 input is the scored file; settings and code are
   those of the scoring (a code change needs --accept-code-change "<reason>", recorded);
2. the opening is recorded (evaluation/OUTCOMES_OPENED.jsonl) BEFORE any outcome cell is read;
3. the outcome contract (window, index-day exclusion, follow-up, censoring, label / date consistency; episode audit) - a material failure stops
   BEFORE any model performance is computed;
4. metrics on identical usable 2026 patients for every frozen model, the pre-declared overlap subgroups, the 2025-vs-2026 comparison.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from falls_ml.artifacts import utc_now
from falls_ml.phase2 import durable as D
from falls_ml.phase2.state import Phase2Stop
from falls_ml.phase4 import PHASE4_VERSION
from falls_ml.phase4.common import MODEL_MANIFEST, PRED_FILE, PRED_MANIFEST, SCORE_DONE, code_sha, dirs, event, read_json
from falls_ml.phase4.metrics import calibration_rows, capacity_rows, interpret, metrics_with_ci

SENTINEL_YEAR = 2900
OUTCOME_EXTRA = ("Fall_On_Index_Date_Ind", "Last_Fall_Date", "Is_Censored_180D", "Has_Full_Followup_180D", "Followup_End_Reason")


# ============================================================================ 1. hash verification
def verify_frozen(out: Path, input_2026: Path, *, cfg_sha: str, code: str, accept_code_change: str | None) -> dict[str, Any]:
    from falls_ml.data.dataset import sha256_file

    d = dirs(out)
    done = d["sealed"] / SCORE_DONE
    if not done.is_file():
        raise Phase2Stop("NOT_SCORED", "no frozen blind predictions in this folder (sealed/BLIND_SCORE_COMPLETE.txt): run meuhedet-phase4-score first; "
                                       "the outcomes stay sealed")
    rec = dict(line.split(" ", 1) for line in done.read_text(encoding="utf-8").splitlines()[1:] if " " in line)
    problems = []

    def check(path: Path, want: str | None, what: str) -> None:
        if not path.is_file():
            problems.append(f"{what}: {path.name} missing")
        elif want is None or D.sha256_file(path) != want:
            problems.append(f"{what}: {path.name} differs from its frozen sha256")

    check(d["sealed"] / PRED_MANIFEST, rec.get("manifest_sha256"), "prediction manifest")
    if problems:
        raise Phase2Stop("FROZEN_HASH_MISMATCH", "the frozen blind predictions were changed after scoring: evaluation refused", problems)
    man = read_json(d["sealed"] / PRED_MANIFEST)
    check(d["sealed"] / PRED_FILE, man["predictions_sha256"], "predictions")
    if rec.get("predictions_sha256") != man["predictions_sha256"]:
        problems.append("predictions sha256 in BLIND_SCORE_COMPLETE.txt and in the manifest differ")
    check(d["frozen"] / MODEL_MANIFEST, man["frozen_model_manifest_sha256"], "frozen model manifest")
    check(d["frozen"] / "PHASE3_INTERNAL_ESTIMATE.json", man["phase3_internal_estimate_sha256"], "Phase 3 internal estimate")
    check(d["sealed"] / "COHORT_OVERLAP.json", man["cohort_overlap_sha256"], "cohort overlap")
    if not problems:
        from falls_ml.phase4.common import sha_json

        fl = read_json(d["frozen"] / "FEATURE_LIST.json")
        if sha_json(fl) != man["feature_list_sha256"]:
            problems.append("feature list differs from the frozen feature list")
        mm = read_json(d["frozen"] / MODEL_MANIFEST)
        for c, m in mm["models"].items():
            check(out / m["artifact"], m["artifact_sha256"], f"model object {c}")
    if problems:
        raise Phase2Stop("FROZEN_HASH_MISMATCH", "the frozen blind predictions were changed after scoring: evaluation refused", problems)
    if sha256_file(input_2026) != man["input_2026"]["sha256"]:
        raise Phase2Stop("INPUT_CHANGED", "the 2026 input differs from the file that was scored (sha256)")
    if cfg_sha != man["config_sha256"]:
        raise Phase2Stop("CONFIG_CHANGED", "configs/meuhedet/phase4.yaml differs from the settings frozen at scoring")
    if code != man["code_sha256"] and not accept_code_change:
        raise Phase2Stop("CODE_CHANGED", "the falls_ml code differs from the code that scored the predictions",
                         ["the predictions stay frozen; pass --accept-code-change \"<reason>\" to evaluate them with this code (recorded)"])
    return man


# ============================================================================ 3. the 2026 outcome contract
def read_outcomes(input_2026: Path, contract: Any, cfg: Any) -> pd.DataFrame:
    from falls_ml.data.meuhedet_wide import NA_VALUES, coerce_wide_types_report
    from falls_ml.phase4.sealed import read_header

    oc = cfg["outcome_contract"]
    hdr = read_header(input_2026)
    want = ["Customer_Full_ID", "Index_Date", "Is_Eligible_Cohort", oc["label_column"], oc["event_date_column"], oc["days_to_event_column"],
            oc["window_end_column"], oc["reason_column"], oc["followup_end_column"], *OUTCOME_EXTRA]
    cols = [c for c in dict.fromkeys(want) if c in hdr]
    raw = pd.read_csv(input_2026, usecols=cols, dtype="string", na_values=list(NA_VALUES), keep_default_na=False, encoding="utf-8")[cols] \
        if input_2026.suffix.lower() != ".parquet" else pd.read_parquet(input_2026, columns=cols).astype("string")
    frame, _ = coerce_wide_types_report(raw, contract)
    return frame


def outcome_contract(o: pd.DataFrame, cfg: Any) -> dict[str, Any]:
    oc = cfg["outcome_contract"]
    idx = pd.Timestamp(cfg["index_date"])
    W = int(oc["window_days"])
    n = len(o)
    y = pd.to_numeric(o[oc["label_column"]], errors="coerce") if oc["label_column"] in o else pd.Series(np.nan, index=o.index)
    pos = (y == 1).fillna(False).to_numpy()
    neg = (y == 0).fillna(False).to_numpy()
    ev = o[oc["event_date_column"]].dt.normalize() if oc["event_date_column"] in o else pd.Series(pd.NaT, index=o.index)
    dte = (ev - idx).dt.days
    checks: dict[str, Any] = {}
    hard: list[str] = []
    # O1 the outcome starts strictly AFTER the index day (no index-day event counted as a future outcome)
    o1 = int((pos & (dte <= 0).fillna(False).to_numpy()).sum())
    dd = pd.to_numeric(o[oc["days_to_event_column"]], errors="coerce") if oc["days_to_event_column"] in o else None
    o1b = int((pos & (dd <= 0).fillna(False).to_numpy()).sum()) if dd is not None else 0
    checks["O1_outcome_after_index_day"] = {"passed": o1 == 0 and o1b == 0, "positives_event_on_or_before_index": o1, "positives_days_to_event_le_0": o1b,
                                            "min_days_to_event_positives": int(dte[pos].min()) if pos.any() and dte[pos].notna().any() else None}
    if not checks["O1_outcome_after_index_day"]["passed"]:
        hard.append("O1: positive labels with an event on / before the index day")
    # O2 nominal window end = Index_Date + 180
    if oc["window_end_column"] in o:
        we = (o[oc["window_end_column"]].dt.normalize() - idx).dt.days
        bad = int(((we != W) & we.notna()).sum())
        checks["O2_window_end"] = {"passed": bad == 0, "rows_window_end_not_index_plus_180": bad,
                                   "values": {str(k): int(v) for k, v in we.value_counts(dropna=False).head(5).items()}}
        if bad:
            hard.append(f"O2: window end differs from Index_Date + {W} on {bad} rows")
    else:
        checks["O2_window_end"] = {"passed": None, "note": f"{oc['window_end_column']} not in the extract: the window end is checked through O3 only"}
    # O3 no positive after the nominal window
    o3 = int((pos & (dte > W).fillna(False).to_numpy()).sum())
    checks["O3_event_within_window"] = {"passed": o3 == 0, "positives_event_after_window": o3, "max_days_to_event_positives": int(dte[pos].max()) if pos.any() and dte[pos].notna().any() else None}
    if o3:
        hard.append(f"O3: {o3} positives with an event after Index_Date + {W}")
    # O4 positives after the end of follow-up
    if oc["followup_end_column"] in o:
        fe = o[oc["followup_end_column"]]
        real = fe.notna() & (fe.dt.year < SENTINEL_YEAR)
        after = pos & (real & (ev > fe.dt.normalize())).fillna(False).to_numpy()
        share = after.sum() / max(1, pos.sum())
        lim = float(oc["max_positive_after_followup_share"])
        checks["O4_positive_after_followup_end"] = {"passed": share <= lim, "positives_event_after_followup_end": int(after.sum()), "share_of_positives": float(share),
                                                    "limit": lim, "rows_followup_end_within_window": int((real & ((fe.dt.normalize() - idx).dt.days <= W)).sum())}
        if share > lim:
            hard.append(f"O4: {100 * share:.2f}% of positives have an event after Followup_End_Date (> {100 * lim:g}%)")
    else:
        checks["O4_positive_after_followup_end"] = {"passed": None, "note": "Followup_End_Date not in the extract: cannot be assessed"}
    # O5 label / event-date consistency
    p_nodate = int((pos & ev.isna().to_numpy()).sum())
    n_date = int((neg & ((dte > 0) & (dte <= W)).fillna(False).to_numpy()).sum())
    lim5 = float(oc["max_label_date_inconsistency_share"])
    share5 = (p_nodate + n_date) / max(1, pos.sum() + neg.sum())
    mism = int((pos & dd.notna().to_numpy() & dte.notna().to_numpy() & (dd != dte).fillna(False).to_numpy()).sum()) if dd is not None else 0
    checks["O5_label_date_consistency"] = {"passed": share5 <= lim5, "positives_without_event_date": p_nodate, "negatives_with_event_in_window": n_date,
                                           "positives_days_to_event_not_matching_dates": mism, "share": float(share5), "limit": lim5}
    if share5 > lim5:
        hard.append(f"O5: label / event-date inconsistencies on {100 * share5:.3f}% of labelled rows")
    # O6 censoring and the usable population
    usable = pos | neg
    reasons = o.loc[~usable, oc["reason_column"]].astype("string").fillna("NULL").value_counts().to_dict() if oc["reason_column"] in o else {}
    ev_u = int(pos.sum())
    checks["O6_censoring_usable_population"] = {"passed": ev_u >= int(oc["min_usable_events"]), "scored_rows": n, "usable_rows": int(usable.sum()),
                                                "usable_share": float(usable.mean()) if n else 0.0, "censored_or_unlabelled_rows": int((~usable).sum()),
                                                "censored_by_reason": {str(k): int(v) for k, v in reasons.items()}, "usable_events": ev_u,
                                                "min_usable_events": int(oc["min_usable_events"]),
                                                "warning": (float(usable.mean()) < float(oc["warn_usable_share_below"])) if n else True,
                                                "rule": "the usable population = scored eligible patients with a 0/1 label; censored patients are reported, "
                                                        "never imputed, and are excluded from every metric (as in Phase 3)"}
    if ev_u < int(oc["min_usable_events"]):
        hard.append(f"O6: only {ev_u} usable 2026 events (< {oc['min_usable_events']})")
    # O7 episode continuation audit (descriptive; never a stop)
    ea = oc["episode_audit"]
    early = (pos & (dte <= int(ea["early_days"])).fillna(False).to_numpy())
    recent = np.zeros(n, dtype=bool)
    if "Last_Fall_Date" in o:
        ds = (idx - o["Last_Fall_Date"].dt.normalize()).dt.days
        recent = ((ds >= 0) & (ds <= int(ea["recent_fall_days"]))).fillna(False).to_numpy()
    iday = (pd.to_numeric(o["Fall_On_Index_Date_Ind"], errors="coerce") == 1).fillna(False).to_numpy() if "Fall_On_Index_Date_Ind" in o else np.zeros(n, dtype=bool)

    def rate(m: np.ndarray) -> float | str | None:
        mm = m & usable
        k, nn = int(early[mm].sum()), int(mm.sum())
        if not nn:
            return None
        return "suppressed" if 0 < k < 10 or 0 < nn - k < 10 else float(k / nn)     # a rate of 1-9 events gives a small cell back

    r_rec, r_oth = rate(recent), rate(~recent)
    n_early = int(early.sum())
    checks["O7_episode_audit"] = {"passed": True, "early_days": int(ea["early_days"]), "recent_fall_days": int(ea["recent_fall_days"]),
                                  "events_within_early_days": n_early,
                                  "share_of_events_within_early_days": "suppressed" if 0 < n_early < 10 else float(n_early / max(1, ev_u)),
                                  "early_event_rate_recent_fall": r_rec, "early_event_rate_other": r_oth,
                                  "early_rate_ratio_recent_vs_other": (r_rec / r_oth) if isinstance(r_rec, float) and isinstance(r_oth, float) and r_oth else None,
                                  "n_recent_fall": int((recent & usable).sum()), "n_index_day_fall": int((iday & usable).sum()),
                                  "early_events_after_index_day_fall": int((early & iday).sum()),
                                  "note": "descriptive: a high early-event rate after a recent / index-day fall may be one injury episode counted as history AND "
                                          "outcome (repeated diagnosis records); reported, never used to change a model"}
    return _py({"passed": not hard, "hard_failures": hard, "checks": checks, "index_date": cfg["index_date"], "window_days": W,
                "contract": "prediction at the END of Index_Date; outcome = first recorded fall / fracture from Index_Date + 1 day to Index_Date + 180 days"})


def _py(obj: Any) -> Any:
    """numpy scalars -> python (JSON)."""
    if isinstance(obj, dict):
        return {str(k): _py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_py(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


# ============================================================================ 4. metrics
def _intervals(pred: pd.DataFrame, setname: str) -> Any:
    from falls_ml.phase3.bounds import Intervals

    return Intervals(lo=pred[f"lo__{setname}"].to_numpy(dtype=float), hi=pred[f"hi__{setname}"].to_numpy(dtype=float))


def evaluate_models(pred: pd.DataFrame, y: np.ndarray, mm: dict[str, Any], est: dict[str, Any], cfg: Any) -> dict[str, Any]:
    from falls_ml.phase2.evaluate import top_mask
    from falls_ml.phase3.bounds import assign, bounded_delta

    met = cfg["metrics"]
    caps = tuple(float(q) for q in met["capacities"])
    nb, seed = int(met["bootstrap_n"]), int(cfg["seed"])
    seen = pred["seen_in_2025_cohort"].to_numpy(dtype=bool)
    comp, capr, calr, vs3 = [], [], [], []
    ivs = {}
    primary = mm["primary"]
    for c, m in mm["models"].items():
        iv = _intervals(pred, m["set"])
        ivs[c] = iv
        dev = float(m["development_prevalence_fit_rows"])
        exact = bool(iv.exact)
        for assignment in (("adverse",) if exact else ("adverse", "favourable")):
            p = assign(y, iv, assignment)
            label = "exact" if exact else assignment
            r = metrics_with_ci(y, p, caps=caps, dev_prev=dev, n_boot=nb if assignment == "adverse" else 0, seed=seed)
            comp.append({"model": c, "is_primary": c == primary, "population": "ALL_2026_USABLE", "assignment": label, "artifact_mode": m["mode"], **r})
            flags = {q: top_mask(p, q) for q in caps}
            capr += capacity_rows(y, p, caps, model=c, population="ALL_2026_USABLE", assignment=label)
            capr += capacity_rows(y, p, tuple(float(q) for q in met["frozen_cutoff_capacities"]), model=c, population="ALL_2026_USABLE", assignment=label,
                                  cutoffs={float(k): v for k, v in m["frozen_2025_cutoffs"].items()})
            if assignment == "adverse":
                calr += calibration_rows(y, p, int(met["calibration_groups"]), model=c, population="ALL_2026_USABLE", assignment=label)
            for sub, msk in (("A_SEEN_IN_2025", seen), ("B_NEW_IN_2026", ~seen)):
                ys, ps = y[msk], p[msk]
                ok = ys.sum() >= int(met["subgroup_min_events"]) and (len(ys) - ys.sum()) >= int(met["subgroup_min_nonevents"])
                if ok:
                    rs = metrics_with_ci(ys, ps, caps=caps, dev_prev=dev, n_boot=nb if assignment == "adverse" else 0, seed=seed + 7)
                    comp.append({"model": c, "is_primary": c == primary, "population": sub, "assignment": label, "artifact_mode": m["mode"], **rs})
                    capr += capacity_rows(ys, ps, caps, model=c, population=sub, assignment=label, mask_from={q: f[msk] for q, f in flags.items()})
                else:
                    comp.append({"model": c, "is_primary": c == primary, "population": sub, "assignment": label, "artifact_mode": m["mode"], "n": int(len(ys)),
                                 "events": int(ys.sum()), "note": "NOT ADEQUATELY SIZED (pre-declared minimum events / non-events): no metric reported"})
        e = (est.get("models") or {}).get(c) or {}
        if e.get("available"):
            r26 = next(x for x in comp if x["model"] == c and x["population"] == "ALL_2026_USABLE" and x["assignment"] in ("exact", "adverse"))
            a3 = e["adverse"]
            for k, lab in (("auroc", "AUROC"), ("ap", "Average precision (PR-AUC)"), ("brier", "Brier score"), ("capture_top10", "Capture@10%"),
                           ("ppv_top10", "PPV@10%"), ("lift_top10", "Lift@10%")):
                t3 = (a3.get(k, math.nan), a3.get(f"{k}_ci_low", math.nan), a3.get(f"{k}_ci_high", math.nan))
                t6 = (r26.get(k, math.nan), r26.get(f"{k}_ci_low", math.nan), r26.get(f"{k}_ci_high", math.nan))
                vs3.append({"model": c, "metric": k, "metric_label": lab, "phase3_internal_estimate": t3[0], "phase3_ci_low": t3[1], "phase3_ci_high": t3[2],
                            "temporal_2026": t6[0], "temporal_2026_ci_low": t6[1], "temporal_2026_ci_high": t6[2], "absolute_change": t6[0] - t3[0],
                            "interpretation": interpret(t3, t6)})
            for k, lab in (("prevalence", "Event prevalence"), ("n", "Cohort N"), ("events", "Event N")):
                vs3.append({"model": c, "metric": k, "metric_label": lab, "phase3_internal_estimate": a3.get(k), "temporal_2026": r26.get(k),
                            "absolute_change": (r26.get(k) or 0) - (a3.get(k) or 0), "interpretation": "descriptive"})
    deltas = []
    for c in mm["models"]:
        if c == primary or primary not in ivs:
            continue
        dlt = bounded_delta(y, ivs[primary], ivs[c], n_boot=nb, seed=seed + 11, principal=float(met["principal_capacity"]), capacities=caps)
        deltas.append({"model": c, "reference": primary, **dlt})
    return {"comparison": pd.DataFrame(comp), "capacity": pd.DataFrame(capr), "calibration": pd.DataFrame(calr), "vs_phase3": pd.DataFrame(vs3),
            "paired_vs_primary": pd.DataFrame(deltas)}


def run_evaluate(input_2026: str | Path, *, out_dir: str | Path, config_path: str | Path, phase3_config: str | Path,
                 accept_code_change: str | None = None) -> dict[str, Any]:
    from falls_ml.phase4.common import load_definitions
    from falls_ml.phase4.report import write_reports, write_share
    from falls_ml.phase4.score import row_key

    src, out = Path(input_2026), Path(out_dir)
    d = dirs(out)
    L = load_definitions(phase3_config, config_path)
    cfg = L["cfg4"]
    code = code_sha()
    man = verify_frozen(out, src, cfg_sha=cfg.sha256, code=code, accept_code_change=accept_code_change)
    d["evaluation"].mkdir(parents=True, exist_ok=True)
    if (d["evaluation"] / "EVALUATION_COMPLETE.json").is_file() and (d["share"] / "PRIVACY_SCAN.json").is_file():
        res = read_json(d["evaluation"] / "EVALUATION_COMPLETE.json")
        return {**res, "note": "already evaluated (deterministic); nothing recomputed"}
    D.append_jsonl(d["evaluation"] / "OUTCOMES_OPENED.jsonl", {"ts": utc_now(), "predictions_sha256": man["predictions_sha256"],
                                                               "frozen_model_manifest_sha256": man["frozen_model_manifest_sha256"], "code_sha256": code,
                                                               "accept_code_change": accept_code_change, "phase4_version": PHASE4_VERSION})
    event(out, "evaluate", "outcomes_opened", "RUNNING", predictions_sha256=man["predictions_sha256"])
    pred = pd.read_parquet(d["sealed"] / PRED_FILE)
    mm = read_json(d["frozen"] / MODEL_MANIFEST)
    est = read_json(d["frozen"] / "PHASE3_INTERNAL_ESTIMATE.json")
    overlap = read_json(d["sealed"] / "COHORT_OVERLAP.json")
    o = read_outcomes(src, L["contract"], cfg)
    from falls_ml.phase4.features import eligible_mask

    o = o.loc[eligible_mask(o, cfg["index_date"])].reset_index(drop=True)
    o["key"] = o["Customer_Full_ID"].astype("string").map(lambda v: row_key(man["input_2026"]["sha256"], str(v)))
    o = o.drop(columns=["Customer_Full_ID"]).set_index("key")
    if set(o.index) != set(pred["key"]) or len(o) != len(pred):
        raise Phase2Stop("ROW_LINKAGE", "the outcome rows do not correspond one-to-one to the frozen scored rows")
    o = o.loc[pred["key"]].reset_index(drop=True)
    oc = outcome_contract(o, cfg)
    D.write_json(d["evaluation"] / "OUTCOME_CONTRACT_2026.json", oc)
    if not oc["passed"]:
        D.write_str(d["evaluation"] / "STOPPED.md", "\n".join(["# STOPPED - the 2026 outcome contract failed", "",
                                                               "No model performance was computed or reported.", "", *[f"- {h}" for h in oc["hard_failures"]], ""]))
        write_share(out, L=L, cfg=cfg, man=man, stopped=oc, src=src)
        event(out, "evaluate", "outcome_contract", "STOPPED", failures=oc["hard_failures"])
        raise Phase2Stop("OUTCOME_CONTRACT_FAILED", "the 2026 outcome contract failed: STOP BEFORE REPORTING MODEL PERFORMANCE", oc["hard_failures"])
    usable = pd.to_numeric(o[cfg["outcome_contract"]["label_column"]], errors="coerce").isin([0, 1]).to_numpy()
    y = pd.to_numeric(o.loc[usable, cfg["outcome_contract"]["label_column"]]).to_numpy(dtype=int)
    res = evaluate_models(pred.loc[usable].reset_index(drop=True), y, mm, est, cfg)
    res["seen_usable"] = {"A_SEEN_IN_2025": int(pred.loc[usable, "seen_in_2025_cohort"].sum()), "B_NEW_IN_2026": int((~pred.loc[usable, "seen_in_2025_cohort"]).sum())}
    write_reports(out, res=res, oc=oc, mm=mm, est=est, overlap=overlap, man=man, cfg=cfg)
    scan = write_share(out, L=L, cfg=cfg, man=man, stopped=None, src=src)
    done = {"status": "EVALUATION_COMPLETE", "outcome_contract": "PASSED", "models": list(mm["models"]), "primary": mm["primary"], "privacy_scan": scan["passed"],
            "finished_at": utc_now()}
    D.write_json(d["evaluation"] / "EVALUATION_COMPLETE.json", done)
    event(out, "evaluate", "complete", "DONE")
    return done
