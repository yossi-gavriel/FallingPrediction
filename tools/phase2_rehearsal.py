"""Phase 2 crash / resume rehearsal on SYNTHETIC data (planning/EXPERIMENT_PLAN.md §10; delivery step 7).

1. Builds a small synthetic extract (same 221-column contract) and a completed reference ``meuhedet-explore`` run.
2. Run A: ``meuhedet-phase2`` uninterrupted.
3. Run B: the same command, killed abruptly (os._exit, no cleanup - like a power loss) at pre-declared points, then resumed with ``--resume``
   after every kill: mid-item in the cohort stage, before a directory rename (LASSO fold), between the rename and the commit record
   (LASSO final), in the middle of an Optuna trial, in a stability replicate, inside the one-time VALIDATION stage and in the share build.
4. Checks, written to REHEARSAL_REPORT.json / .md:
   - every attempt after a kill resumed (RESUME_AUDIT.json), nothing committed was recomputed or changed (the item commit records and
     file hashes of attempt k are a subset of attempt k+1, byte-identical);
   - the final result tables of B are byte-identical to A;
   - the reference folder is byte-identical before and after;
   - VALIDATION was opened for one frozen selection only.

SYNTHETIC DATA - SOFTWARE TEST ONLY, NOT SCIENTIFIC RESULTS.

Usage: python tools/phase2_rehearsal.py --work <empty folder> [--rows 6000]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CRASHES = [   # (stage label, item id, point) - each kills one attempt; the next attempt resumes
    ("S01_cohort", "cohort", "item_start"),
    ("S06_lasso", "LASSO__BASELINE_15__outer1", "before_rename"),
    ("S06_lasso", "LASSO__ALL_REVIEWED_SAFE__final", "before_record"),
    ("S08_xgb", "XGB1_outer1__t02", "item_start"),
    ("S10_stability", "STAB__ALL_REVIEWED_SAFE__r01", "before_record"),
    ("S13_validation", "validation", "item_start"),
    ("S16_share", "share", "before_rename"),
]
COMPARE = ["MODEL_COMPARISON.csv", "DOMAIN_INCREMENTAL_GAIN.csv", "FEATURE_CONSENSUS.csv", "ABLATION_RESULTS.csv", "OPERATIONAL_CAPACITY.csv",
           "THRESHOLD_RESULTS.csv", "CALIBRATION_SUMMARY.csv", "SUBGROUP_SUMMARY.csv", "COLUMN_FUNNEL.csv", "FEATURE_REGISTRY.csv", "FEATURE_QUALITY.csv",
           "SCIENTIFIC_SUMMARY.md", "MANAGEMENT_SUMMARY_HE.md", "tables/01_COLUMN_REGISTRY.csv", "tables/OOF_MODEL_COMPARISON.csv",
           "tables/VALIDATION_MODEL_COMPARISON.csv", "tables/STABILITY.csv", "tables/SHAP_SUMMARY.csv", "tables/PERMUTATION_IMPORTANCE.csv",
           "tables/FEATURE_SETS.json"]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def committed(out: Path) -> dict[str, str]:
    """Every committed item file (per its commit record) -> sha256 now."""
    res = {}
    for rec_path in sorted((out / "stages").glob("*/items/*.COMPLETE.json")):
        rec = json.loads(rec_path.read_text(encoding="utf-8"))
        d = rec_path.parent / rec_path.name[: -len(".COMPLETE.json")]
        res[str(rec_path.relative_to(out))] = sha(rec_path)
        for rel in rec["files"]:
            res[str((d / rel).relative_to(out))] = sha(d / rel)
    return res


def tree_digest(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(root)).encode())
            h.update(sha(p).encode())
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--rows", type=int, default=6000)
    a = ap.parse_args()
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONUTF8": "1"}
    py = sys.executable
    t0 = time.time()
    # ---- synthetic extract (+ planted index-day home-safety records -> that source must come out UNSAFE) and the reference run
    sys.path.insert(0, str(ROOT / "src"))
    from falls_ml.data.meuhedet_synthetic import generate_synthetic_wide_extract

    csv = work / "synthetic_extract.csv"
    if not csv.exists():
        df = generate_synthetic_wide_extract(a.rows, seed=20260929, index_date="2025-01-01", n_index_day_falls=10)
        idx = pd.Timestamp("2025-01-01")
        m = (df["Index_Date"] == idx) & (df["Is_Eligible_Cohort"] == 1) & df["Home_Safety_Assessment_Date"].notna() & df["Fall_Next_180D_Ind"].notna()
        pick = np.flatnonzero(m.to_numpy())[:12]
        df.loc[df.index[pick], "Home_Safety_Assessment_Date"] = np.datetime64(idx, "us")
        out = df.copy()
        for c in out.columns:
            if pd.api.types.is_datetime64_any_dtype(out[c]):
                out[c] = out[c].dt.strftime("%d/%m/%Y")
        out.to_csv(csv, index=False, na_rep="NULL", encoding="utf-8", lineterminator="\n")
    pepper = csv.with_name(csv.name + ".id_pepper.txt")
    if not pepper.exists():   # pinned: a random pepper can give a degenerate reference split on small synthetic extracts
        pepper.write_text("# falls_ml pseudonym\nsynthetic-phase2-rehearsal-pepper-01\n", encoding="utf-8", newline="\n")
    ref = work / "explore"
    if not (ref / "readiness.json").exists():
        subprocess.run([py, "-m", "falls_ml", "meuhedet-explore", "--input", str(csv), "--out", str(ref), "--index-date", "2025-01-01",
                        "--index-day-records", "drop_rows", "--model-report", "off"], check=True, env=env, cwd=ROOT, capture_output=True)
    ref_before = tree_digest(ref)
    # ---- small rehearsal config (same code paths, fewer trials / folds)
    cfg = yaml.safe_load((ROOT / "configs/meuhedet/phase2.yaml").read_text(encoding="utf-8"))
    p = cfg["phase2"]
    p["cv"] = {"outer_folds": 3, "inner_folds_linear": 4, "inner_folds_xgb": 3}
    p["xgb"]["stage1"].update({"n_trials": 4, "n_startup_trials": 2})
    p["xgb"]["stage2"].update({"n_trials": 2, "condition_min_ap_gain_vs_best_linear": -1.0})
    p["xgb"].update({"n_estimators_max": 300, "early_stopping_rounds": 30})
    p["lasso"]["n_lambda"] = 40
    p["enet"].update({"n_lambda": 30, "l1_ratios": [0.5, 0.9]})
    p["stability"]["n_replicates"] = 3
    p["ablation"]["individual_top_n"] = 3
    p["metrics"]["bootstrap_n"] = 100
    p["explain"].update({"shap_sample_rows": 2000, "permutation_repeats": 2})
    p["eligibility"]["min_observed_train_rows"] = 30
    p["features"] = str(ROOT / "configs/meuhedet/phase2_features.yaml")
    p["d00_config"] = str(ROOT / "configs/meuhedet/d00_sensitivity.yaml")
    cfg_path = work / "phase2_rehearsal.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8", newline="\n")

    def accept(gs: list[str]) -> list[str]:
        out = [x for g in gs for x in ("--accept-gate", g)]
        return [*out, "--reason", "synthetic rehearsal data: planted signal (software test)"] if out else []

    def phase2(out: Path, *extra: str, crash: tuple[str, str, str] | None = None) -> subprocess.CompletedProcess[str]:
        e = dict(env)
        if crash:
            e["FALLS_ML_PHASE2_CRASH_AT"] = "|".join(crash)
        cmd = [py, "-m", "falls_ml", "meuhedet-phase2", "--input", str(csv), "--reference", str(ref), "--out", str(out), "--config", str(cfg_path),
               "--allow-unfrozen-config", *extra]   # the reduced rehearsal configuration is deliberately not the frozen production one
        return subprocess.run(cmd, env=e, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")

    # ---- run A (no kill). Planted synthetic signals can trip plausibility gates (investigation stops): accept them with a recorded
    # reason, exactly as a user would after reviewing the evidence, and resume.
    A = work / "run_A"
    ta = time.time()
    gates: list[str] = []
    ra = phase2(A)
    while ra.returncode == 2 and "INVESTIGATION STOP [" in ra.stderr:
        gates.append(ra.stderr.split("INVESTIGATION STOP [", 1)[1].split("]", 1)[0])
        ra = phase2(A, "--resume", *accept(gates))
    if ra.returncode != 0:
        print(ra.stderr[-4000:])
        return 1
    ta = time.time() - ta
    # ---- run B (killed at every crash point, resumed each time)
    B = work / "run_B"
    attempts = []
    prev: dict[str, str] = {}
    for i, crash in enumerate([*CRASHES, None]):
        rb = phase2(B, *(["--resume", *accept(gates)] if i else []), crash=crash)   # --accept-gate is only valid with --resume
        snap = committed(B)
        changed = sorted(k for k, v in prev.items() if snap.get(k) != v)
        attempts.append({"attempt": i + 1, "crash_point": "|".join(crash) if crash else "none (final attempt)", "exit_code": rb.returncode,
                         "killed": rb.returncode == 137, "committed_files": len(snap), "earlier_committed_files_changed_or_missing": changed})
        if crash is None and rb.returncode != 0:
            print(rb.stderr[-4000:])
        prev = snap
    ok_resume = all(x["killed"] for x in attempts[:-1]) and attempts[-1]["exit_code"] == 0
    ok_immutable = all(not x["earlier_committed_files_changed_or_missing"] for x in attempts)
    same = {f: (A / "share" / f).exists() and (B / "share" / f).exists() and sha(A / "share" / f) == sha(B / "share" / f) for f in COMPARE}
    audit = json.loads((B / "RESUME_AUDIT.json").read_text(encoding="utf-8"))
    val = [json.loads(line) for line in (B / "validation_evaluation_registry.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    one_selection = len({r["selection_sha256"] for r in val}) == 1
    ref_after = tree_digest(ref)
    report = {"synthetic": True, "rows": a.rows, "run_A_seconds": round(ta, 1), "total_seconds": round(time.time() - t0, 1), "attempts": attempts,
              "investigation_gates_accepted_with_reason": gates, "resumed_after_every_kill": ok_resume, "completed_artifacts_never_changed": ok_immutable,
              "final_tables_identical_to_uninterrupted_run": all(same.values()), "tables_compared": same,
              "reference_folder_unchanged": ref_before == ref_after, "validation_opened_for_one_frozen_selection": one_selection,
              "validation_registry_records": len(val), "resume_audit_attempts": len(audit["attempts"]),
              "share_privacy_scan": json.loads((B / "share" / "PRIVACY_SCAN.json").read_text(encoding="utf-8"))["passed"]}
    report["PASSED"] = all([ok_resume, ok_immutable, report["final_tables_identical_to_uninterrupted_run"], report["reference_folder_unchanged"],
                            one_selection, report["share_privacy_scan"]])
    (work / "REHEARSAL_REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8", newline="\n")
    lines = ["# Phase 2 crash / resume rehearsal (SYNTHETIC DATA – software test only)", "", f"**Overall: {'PASSED' if report['PASSED'] else 'FAILED'}**", "",
             "| # | Kill point (stage, item, point) | Exit | Committed files after | Earlier committed files changed |", "|---|---|---|---|---|"]
    lines += [f"| {x['attempt']} | {x['crash_point']} | {x['exit_code']} | {x['committed_files']} | {len(x['earlier_committed_files_changed_or_missing'])} |"
              for x in attempts]
    lines += ["", f"- Resumed after every kill: **{ok_resume}**", f"- Completed artifacts never changed: **{ok_immutable}**",
              f"- Final tables byte-identical to the uninterrupted run ({len(same)} files): **{all(same.values())}**",
              f"- Reference folder unchanged: **{report['reference_folder_unchanged']}**",
              f"- VALIDATION opened for one frozen selection only: **{one_selection}** ({len(val)} registry record(s))",
              f"- Share privacy scan passed: **{report['share_privacy_scan']}**",
              f"- Investigation stops met and accepted with a recorded reason (planted synthetic signal): {', '.join(gates) or 'none'}", f"- Uninterrupted run: {report['run_A_seconds']} s; whole rehearsal: {report['total_seconds']} s", ""]
    (work / "REHEARSAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print("\n".join(lines))
    return 0 if report["PASSED"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
