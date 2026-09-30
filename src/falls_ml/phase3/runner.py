"""``falls_ml meuhedet-phase3``: the orchestrator. Preflight -> cohort + schema -> source timing -> row-level recovery -> reconstruction ->
scientific feasibility (GO / NO_GO) -> [GO only: feature sets -> LASSO -> elastic net -> XGBoost -> frozen selection -> stability -> ablation ->
explainability -> reused-VALIDATION check] -> reports -> privacy-safe share.

The commit / resume protocol is the unchanged Phase 2 one (``falls_ml.phase2.state``): every item is committed with a record written last, a
resumed run continues from the first incomplete item and never rewrites a completed artifact. The plan (PHASE3_PLAN.json) must re-derive to
the same sha256 on resume; input, code and protected folders are checked as in Phase 2."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from falls_ml.artifacts import utc_now
from falls_ml.phase2 import durable as D
from falls_ml.phase2.state import STOP_HARD, EventLog, Phase2Stop, Run, write_timings
from falls_ml.phase3 import PHASE3_VERSION, WATERMARK
from falls_ml.phase3.config import DEFAULT_CONFIG, load_phase3_config, load_recovery_rules, resolve_threads
from falls_ml.phase3.context import P3Ctx

PLAN_VERSION = 1
PLAN_FILE = "PHASE3_PLAN.json"


def load_all(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    from falls_ml.d00.dependency import load_d00_config
    from falls_ml.data.meuhedet_wide import load_wide_contract, load_wide_mapping
    from falls_ml.eda.dictionary import load_data_dictionary
    from falls_ml.features.spec import load_feature_spec
    from falls_ml.phase2.config import load_catalogue

    cfg = load_phase3_config(config_path)
    mapping = load_wide_mapping()
    contract = load_wide_contract(mapping.contract_path)
    dictionary = load_data_dictionary("configs/meuhedet/wide_v1_data_dictionary.yaml", contract)
    d00 = load_d00_config(cfg["d00_config"], contract=contract, dictionary=dictionary)
    rules = load_recovery_rules(cfg["recovery"], contract=contract, dictionary=dictionary, d00=d00)
    cat = load_catalogue(cfg["features"], contract, dictionary, mapping)
    espec = load_feature_spec(mapping.exploratory_spec_path)
    from falls_ml.phase3.timecontract import load_time_contract

    tc = load_time_contract(cfg["time_contract"], contract=contract, dictionary=dictionary)
    return {"cfg": cfg, "mapping": mapping, "contract": contract, "dictionary": dictionary, "d00": d00, "rules": rules, "cat": cat, "espec": espec, "tc": tc}


def effective_final_config(L: dict[str, Any], *, allow_unfrozen: bool) -> tuple[bytes, str, bool]:
    from falls_ml.phase3.final_config import build_final_config, load_frozen, to_bytes

    eff = to_bytes(build_final_config(L["cfg"], L["rules"], L["cat"], contract=L["contract"], dictionary=L["dictionary"], mapping=L["mapping"], d00=L["d00"],
                                      tc=L["tc"]))
    sha = D.sha256_bytes(eff)
    _, frozen_sha = load_frozen()
    if frozen_sha == sha:
        return eff, sha, True
    if not allow_unfrozen:
        raise Phase2Stop("PHASE3_FROZEN_CONFIG_MISMATCH", "the effective configuration is not the frozen PHASE3_FINAL_EXPERIMENT_CONFIG.json",
                         [f"frozen {str(frozen_sha)[:16]}…, effective {sha[:16]}…", "restore the package files; never edit phase3*.yaml for the real run"])
    return eff, sha, False


def run_phase3(input_path: str | Path, reference_dir: str | Path, *, out_dir: str | Path, protect: list[str | Path] | None = None,
               id_pepper_file: str | Path | None = None, phase2_out: str | Path | None = None, resume: bool = False,
               accept_gates: dict[str, str] | None = None, accept_code_change: str | None = None, allow_synced_folder: bool = False,
               config_path: str | Path = DEFAULT_CONFIG, allow_unfrozen_config: bool = False, stop_after: str | None = None) -> dict[str, Any]:
    from threadpoolctl import threadpool_limits

    from falls_ml.artifacts import source_tree_sha256
    from falls_ml.data.dataset import sha256_file

    src, ref, out = Path(input_path), Path(reference_dir), Path(out_dir)
    protected = [ref, *[Path(p) for p in (protect or [])]]
    p2 = Path(phase2_out) if phase2_out else None
    if not src.is_file():
        raise Phase2Stop("INPUT_MISSING", f"input file not found: {src.name}")
    for p in protected:
        if not p.is_dir():
            raise Phase2Stop("PROTECTED_MISSING", f"protected / reference folder not found: {p.name}")
        if out.resolve() == p.resolve() or p.resolve() in out.resolve().parents:
            raise Phase2Stop("OUT_INSIDE_PROTECTED", f"--out must not be inside the protected folder {p.name}")
    if p2 is not None and (out.resolve() == p2.resolve() or p2.resolve() in out.resolve().parents or out.resolve() in p2.resolve().parents):
        raise Phase2Stop("OUT_OVERLAPS_PHASE2", "--out must be a separate folder, not inside (or around) the Phase 2 output folder")
    synced = D.synced_folder(out.parent if not out.exists() else out)
    if synced and not allow_synced_folder:
        raise Phase2Stop("SYNCED_FOLDER", f"--out is inside a synchronised folder ({synced})", ["choose a local folder, or pass --allow-synced-folder (recorded)"])
    plan_file = out / PLAN_FILE
    if resume:
        if not plan_file.is_file():
            raise Phase2Stop("NOTHING_TO_RESUME", f"--resume: {out.name} holds no {PLAN_FILE} (start without --resume)")
    elif out.exists() and any(out.iterdir()):
        raise Phase2Stop("OUT_NOT_EMPTY", f"--out {out.name} is not empty; results are immutable - choose a new folder or pass --resume")
    if (out / "PHASE2_PLAN.json").exists():
        raise Phase2Stop("OUT_IS_PHASE2", "--out holds a Phase 2 run: Phase 3 never writes into a Phase 2 folder")
    out.mkdir(parents=True, exist_ok=True)
    L = load_all(config_path)
    cfg = L["cfg"]
    final_bytes, final_sha, frozen = effective_final_config(L, allow_unfrozen=allow_unfrozen_config)
    input_sha = sha256_file(src)
    code_sha = source_tree_sha256()
    stored = json.loads(plan_file.read_text(encoding="utf-8")) if resume else None
    threads = stored["plan"]["threads"] if stored else resolve_threads(cfg)
    plan = {"plan_version": PLAN_VERSION, "phase3_version": PHASE3_VERSION, "watermark": WATERMARK, "config": cfg.raw, "config_sha256": cfg.sha256,
            "recovery_sha256": L["rules"].sha256, "time_contract_sha256": L["tc"].sha256, "catalogue_sha256": L["cat"].sha256, "final_experiment_config_sha256": final_sha,
            "frozen_production_config": frozen, "contract_sha256": L["contract"].content_sha256, "dictionary_sha256": L["dictionary"].sha256,
            "mapping_sha256": L["mapping"].content_sha256, "d00_config_sha256": L["d00"].sha256, "input_sha256": input_sha, "input_name": src.name,
            "reference_folder": ref.name, "protected_folders": [p.name for p in protected], "threads": threads, "seed": int(cfg["seed"]),
            "python_series": ".".join(sys.version.split()[0].split(".")[:2])}
    plan_sha = D.sha256_text(json.dumps(plan, sort_keys=True, default=str))
    audit_path = out / "RESUME_AUDIT.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.is_file() else {"attempts": []}
    attempt = len(audit["attempts"]) + 1
    code_change = None
    if stored:
        if stored["plan"]["input_sha256"] != input_sha:
            raise Phase2Stop("INPUT_CHANGED", "--resume: the input file differs from the one this run started with (sha256)")
        if stored["plan_sha256"] != plan_sha:
            diff = sorted(k for k in plan if json.dumps(plan[k], sort_keys=True, default=str) != json.dumps(stored["plan"].get(k), sort_keys=True, default=str))
            raise Phase2Stop("PLAN_CHANGED", "--resume: the analysis plan differs from the frozen one", [f"changed: {k}" for k in diff])
        if stored["code_sha256_at_start"] != code_sha and code_sha not in [a.get("code_sha256") for a in audit["attempts"]]:
            if not accept_code_change:
                raise Phase2Stop("CODE_CHANGED", "--resume: the falls_ml code differs from the code of the earlier attempt(s)",
                                 ["completed items are never recomputed; pass --accept-code-change \"<reason>\" to continue (recorded)"])
            code_change = accept_code_change
        cp = out / "PHASE3_FINAL_EXPERIMENT_CONFIG.json"
        if not cp.is_file() or D.sha256_file(cp) != final_sha:
            raise Phase2Stop("PLAN_CHANGED", "--resume: PHASE3_FINAL_EXPERIMENT_CONFIG.json of this run is missing or differs")
    else:
        from falls_ml.phase3.final_config import sha_line

        D.write_json(plan_file, {"plan": plan, "plan_sha256": plan_sha, "code_sha256_at_start": code_sha, "created_at": utc_now()})
        for name, payload in (("PHASE3_FINAL_EXPERIMENT_CONFIG.json", final_bytes), ("PHASE3_FINAL_EXPERIMENT_CONFIG.sha256", sha_line(final_sha).encode())):
            D.write_bytes(out / name, payload)
            D.mark_readonly(out / name)
    events = EventLog(out / "logs" / "events.jsonl", attempt)
    prev = json.loads((out / "RUN_STATE.json").read_text(encoding="utf-8")) if (out / "RUN_STATE.json").is_file() else {}
    accepted = {**{g: r for a in audit["attempts"] for g, r in (a.get("accepted_gates") or {}).items()}, **(accept_gates or {})}
    run = Run(out=out, plan=plan, plan_sha=plan_sha, code_sha=code_sha, attempt=attempt, seed=int(cfg["seed"]), events=events, accepted_gates=accepted)
    run.state = {"phase": 3, "stage": prev.get("stage"), "completed_stages": prev.get("completed_stages", []), "current_item": None, "dataset_sha256": input_sha,
                 "plan_sha256": plan_sha, "code_sha256": code_sha, "seed": int(cfg["seed"]), "started_at": prev.get("started_at") or utc_now(),
                 "attempt_started_at": utc_now(), "status": "RUNNING", "resume_count": attempt - 1, "attempt": attempt, "threads": threads,
                 "final_experiment_config_sha256": final_sha, "frozen_production_config": frozen, "pid": os.getpid(), "decision": prev.get("decision")}
    entry = {"attempt": attempt, "started_at": utc_now(), "resumed": bool(resume), "code_sha256": code_sha, "accept_code_change": code_change,
             "accepted_gates": accept_gates or {}, "status": "RUNNING", "python": sys.version.split()[0]}
    audit["attempts"].append(entry)
    D.write_json(audit_path, audit)
    run.checkpoint()
    events("run", "attempt_start", "RUNNING", resumed=bool(resume), code_change=code_change)
    if code_change:
        D.append_jsonl(out / "logs" / "gates.jsonl", {"ts": utc_now(), "attempt": attempt, "gate": "CODE_CHANGED", "decision": "ACCEPTED", "reason": code_change})
    ctx = P3Ctx(run=run, src=src, ref_dir=ref, pepper_file=Path(id_pepper_file) if id_pepper_file else None, cfg=cfg, rules=L["rules"], tc=L["tc"], cat=L["cat"],
                contract=L["contract"], mapping=L["mapping"], dictionary=L["dictionary"], d00=L["d00"], espec=L["espec"], threads=threads, protected=protected,
                input_sha=input_sha, phase2_out=p2)
    status, stop = "COMPLETE", None
    try:
        with threadpool_limits(limits=int(threads["blas"])):
            from falls_ml.phase3 import stages_data as SD
            from falls_ml.phase3 import stages_models as SM
            from falls_ml.phase3 import stages_report as SR

            seq = [SD.s00_preflight, SD.s01_cohort, SD.s02_timing, SD.s03_recovery, SD.s04_reconstruct, SD.s05_feasibility]
            for fn in seq:
                fn(ctx)
                if stop_after and fn.__name__.startswith(stop_after):
                    raise _StopAfter()
            decision = ctx.feasibility()["decision"]
            run.checkpoint(decision=decision)
            if decision == "GO":
                for fn in (SD.s06_feature_sets, *SM.MODEL_STAGES):
                    fn(ctx)
                    if stop_after and fn.__name__.startswith(stop_after):
                        raise _StopAfter()
            for fn in (SR.s15_report, SR.s16_share):
                fn(ctx)
            status = "COMPLETE_OUTCOME_A" if decision == "GO" else "COMPLETE_OUTCOME_B_NO_GO"
    except _StopAfter:
        status = f"STOPPED_AFTER_{stop_after}"
    except Phase2Stop as exc:
        status, stop = ("STOPPED" if exc.kind == STOP_HARD else "INVESTIGATION"), exc
        _write_stop(out, exc)
        events("run", "stop", status, gate=exc.gate, error=exc.message, details=exc.details)
        raise
    except Exception as exc:
        status = "FAILED"
        events("run", "error", "FAILED", error=f"{type(exc).__name__}: {exc}"[:2000])
        raise
    finally:
        entry.update({"ended_at": utc_now(), "status": status, "stop_gate": stop.gate if stop else None})
        D.write_json(audit_path, audit)
        run.checkpoint(status=status, stop_gate=stop.gate if stop else None, current_item=None)
        write_timings(out)
    if status.startswith("COMPLETE"):
        D.write_json(out / "RUN_STATE_FINAL.json", run.state)
    return {"status": status, "out": str(out), "attempt": attempt, "plan_sha256": plan_sha, "decision": run.state.get("decision")}


class _StopAfter(Exception):
    pass


def _write_stop(out: Path, exc: Phase2Stop) -> None:
    title = "INVESTIGATION" if exc.kind != STOP_HARD else "STOPPED"
    lines = [f"# {title} - {exc.gate}", "", WATERMARK, "", exc.message, "", *[f"- {d}" for d in exc.details], ""]
    if exc.kind != STOP_HARD:
        lines += ["Investigation stop: nothing was rejected automatically. Review the evidence (artifacts/), then fix the cause and resume, or continue",
                  "with a recorded justification:", "", f"    --resume --accept-gate {exc.gate} --reason \"<your reason>\"", ""]
    else:
        lines += ["Hard stop: fix the cause, then run the same command again with --resume.", ""]
    D.write_str(out / (f"INVESTIGATION_{exc.gate}.md" if exc.kind != STOP_HARD else "STOPPED.md"), "\n".join(lines))
