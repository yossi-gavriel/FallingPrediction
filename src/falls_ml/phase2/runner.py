"""``falls_ml meuhedet-phase2``: the orchestrator. It freezes the plan, checks resume invariants (plan, input, code, protected folders),
pins the thread counts, runs the stages in order, records every attempt (RESUME_AUDIT.json) and every stop (RUN_STATE.json +
STOPPED.md / INVESTIGATION_<gate>.md)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from falls_ml.artifacts import utc_now
from falls_ml.phase2 import WATERMARK
from falls_ml.phase2 import durable as D
from falls_ml.phase2.config import DEFAULT_CONFIG, load_catalogue, load_phase2_config, resolve_threads
from falls_ml.phase2.context import Ctx
from falls_ml.phase2.state import STOP_HARD, EventLog, Phase2Stop, Run, write_timings

PLAN_VERSION = 1


def _stages() -> list[Any]:
    from falls_ml.phase2 import stages_data as SD
    from falls_ml.phase2 import stages_models as SM
    from falls_ml.phase2 import stages_report as SR

    return [SD.s00_preflight, SD.s01_cohort, SD.s02_registry, SD.s03_engineer, SD.s04_screen, SD.s05_feature_sets, SM.s06_linear, SM.s07_enet,
            SM.s08_xgb, SM.s09_select, SM.s10_stability, SM.s11_ablation, SM.s12_explain, SM.s13_validation, SR.s14_consensus, SR.s15_report,
            SR.s16_share]


def _plan(cfg: Any, cat: Any, *, contract: Any, dictionary: Any, mapping: Any, d00: Any, input_sha: str, input_name: str, ref_dir: Path,
          threads: dict[str, int], protected: list[Path], final_sha: str, frozen: bool) -> dict[str, Any]:
    return {"plan_version": PLAN_VERSION, "watermark": WATERMARK, "config": cfg.raw, "config_sha256": cfg.sha256, "catalogue_sha256": cat.sha256,
            "final_experiment_config_sha256": final_sha, "frozen_production_config": frozen,
            "contract_sha256": contract.content_sha256, "dictionary_sha256": dictionary.sha256, "mapping_sha256": mapping.content_sha256,
            "d00_config_sha256": d00.sha256, "input_sha256": input_sha, "input_name": input_name, "reference_folder": ref_dir.name,
            "protected_folders": [p.name for p in protected], "threads": threads, "seed": int(cfg["seed"]),
            "python_series": ".".join(sys.version.split()[0].split(".")[:2])}


def _plan_sha(plan: dict[str, Any]) -> str:
    return D.sha256_text(json.dumps(plan, sort_keys=True, default=str))


def run_phase2(input_path: str | Path, reference_dir: str | Path, *, out_dir: str | Path, protect: list[str | Path] | None = None,
               id_pepper_file: str | Path | None = None, resume: bool = False, accept_gates: dict[str, str] | None = None,
               accept_code_change: str | None = None, allow_synced_folder: bool = False, config_path: str | Path = DEFAULT_CONFIG,
               readonly: bool = True, stop_after: str | None = None, allow_unfrozen_config: bool = False) -> dict[str, Any]:
    from threadpoolctl import threadpool_limits

    from falls_ml.artifacts import source_tree_sha256
    from falls_ml.d00.dependency import load_d00_config
    from falls_ml.data.dataset import sha256_file
    from falls_ml.data.meuhedet_wide import load_wide_contract, load_wide_mapping
    from falls_ml.eda.dictionary import load_data_dictionary
    from falls_ml.features.spec import load_feature_spec

    src, ref, out = Path(input_path), Path(reference_dir), Path(out_dir)
    protected = [ref, *[Path(p) for p in (protect or [])]]
    if not src.is_file():
        raise Phase2Stop("INPUT_MISSING", f"input file not found: {src.name}")
    for p in protected:
        if not p.is_dir():
            raise Phase2Stop("PROTECTED_MISSING", f"protected / reference folder not found: {p.name}")
        if out.resolve() == p.resolve() or p.resolve() in out.resolve().parents:
            raise Phase2Stop("OUT_INSIDE_PROTECTED", f"--out must not be inside the protected folder {p.name}")
    synced = D.synced_folder(out.parent if not out.exists() else out)
    if synced and not allow_synced_folder:
        raise Phase2Stop("SYNCED_FOLDER", f"--out is inside a synchronised folder ({synced}); sync clients lock and rewrite files during the run",
                         ["choose a local folder (e.g. under Downloads or C:\\falls_ml_runs)", "or pass --allow-synced-folder (recorded)"])
    plan_file = out / "PHASE2_PLAN.json"
    if resume:
        if not plan_file.is_file():
            raise Phase2Stop("NOTHING_TO_RESUME", f"--resume: {out.name} holds no PHASE2_PLAN.json (start without --resume)")
    elif out.exists() and any(out.iterdir()):
        raise Phase2Stop("OUT_NOT_EMPTY", f"--out {out.name} is not empty; results are immutable - choose a new folder or pass --resume")
    out.mkdir(parents=True, exist_ok=True)

    cfg = load_phase2_config(config_path)
    mapping = load_wide_mapping()
    contract = load_wide_contract(mapping.contract_path)
    dictionary = load_data_dictionary("configs/meuhedet/wide_v1_data_dictionary.yaml", contract)
    d00 = load_d00_config(cfg["d00_config"], contract=contract, dictionary=dictionary)
    cat = load_catalogue(cfg["features"], contract, dictionary, mapping)
    espec = load_feature_spec(mapping.exploratory_spec_path)
    final_bytes, final_sha, frozen = effective_final_config(cfg, cat, contract=contract, dictionary=dictionary, mapping=mapping, d00=d00,
                                                            allow_unfrozen=allow_unfrozen_config)
    input_sha = sha256_file(src)
    code_sha = source_tree_sha256()
    stored = json.loads(plan_file.read_text(encoding="utf-8")) if resume else None
    threads = stored["plan"]["threads"] if stored else resolve_threads(cfg)
    plan = _plan(cfg, cat, contract=contract, dictionary=dictionary, mapping=mapping, d00=d00, input_sha=input_sha, input_name=src.name, ref_dir=ref,
                 threads=threads, protected=protected, final_sha=final_sha, frozen=frozen)
    plan_sha = _plan_sha(plan)
    audit_path = out / "RESUME_AUDIT.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.is_file() else {"attempts": []}
    attempt = len(audit["attempts"]) + 1
    code_change = None
    if stored:
        if stored["plan"]["input_sha256"] != input_sha:
            raise Phase2Stop("INPUT_CHANGED", "--resume: the input file differs from the one this run started with (sha256)")
        if stored["plan_sha256"] != plan_sha:
            diff = sorted(k for k in plan if json.dumps(plan[k], sort_keys=True, default=str) != json.dumps(stored["plan"].get(k), sort_keys=True, default=str))
            raise Phase2Stop("PLAN_CHANGED", "--resume: the analysis plan differs from the frozen one (configuration, catalogue or inputs changed)",
                             [f"changed: {k}" for k in diff])
        if stored["code_sha256_at_start"] != code_sha and code_sha not in [a.get("code_sha256") for a in audit["attempts"]]:
            if not accept_code_change:
                raise Phase2Stop("CODE_CHANGED", "--resume: the falls_ml code differs from the code of the earlier attempt(s)",
                                 ["completed items are never recomputed; pass --accept-code-change \"<reason>\" to continue (recorded)"])
            code_change = accept_code_change
        _verify_final_copy(out, final_sha)
    else:
        D.write_json(plan_file, {"plan": plan, "plan_sha256": plan_sha, "code_sha256_at_start": code_sha, "created_at": utc_now()})
        _write_final_copy(out, final_bytes, final_sha, readonly=readonly)

    events = EventLog(out / "logs" / "events.jsonl", attempt)
    prev_state = json.loads((out / "RUN_STATE.json").read_text(encoding="utf-8")) if (out / "RUN_STATE.json").is_file() else {}
    accepted = {**{g: r for a in audit["attempts"] for g, r in (a.get("accepted_gates") or {}).items()}, **(accept_gates or {})}
    run = Run(out=out, plan=plan, plan_sha=plan_sha, code_sha=code_sha, attempt=attempt, seed=int(cfg["seed"]), events=events, accepted_gates=accepted,
              readonly=readonly)
    run.state = {"stage": prev_state.get("stage"), "substage": None, "completed_stages": prev_state.get("completed_stages", []), "current_item": None,
                 "dataset_sha256": input_sha, "feature_registry_sha256": prev_state.get("feature_registry_sha256"), "config_sha256": cfg.sha256,
                 "catalogue_sha256": cat.sha256, "plan_sha256": plan_sha, "code_sha256": code_sha, "seed": int(cfg["seed"]),
                 "started_at": prev_state.get("started_at") or utc_now(), "attempt_started_at": utc_now(), "status": "RUNNING",
                 "resume_count": attempt - 1, "attempt": attempt, "threads": threads, "final_experiment_config_sha256": final_sha,
                 "frozen_production_config": frozen, "pid": os.getpid()}
    entry = {"attempt": attempt, "started_at": utc_now(), "resumed": bool(resume), "code_sha256": code_sha, "accept_code_change": code_change,
             "accepted_gates": accept_gates or {}, "status": "RUNNING", "python": sys.version.split()[0]}
    audit["attempts"].append(entry)
    D.write_json(audit_path, audit)
    run.checkpoint()
    events("run", "attempt_start", "RUNNING", resumed=bool(resume), code_change=code_change)
    if code_change:
        prev_codes = sorted({str(a.get("code_sha256")) for a in audit["attempts"][:-1]})
        events("run", "code_change_accepted", "CONTINUE", reason=code_change, code_sha256=code_sha, earlier_code_sha256=prev_codes)
        D.append_jsonl(out / "logs" / "gates.jsonl", {"ts": utc_now(), "attempt": attempt, "gate": "CODE_CHANGED", "decision": "ACCEPTED",
                                                     "reason": code_change, "code_sha256": code_sha, "earlier_code_sha256": prev_codes})
    ctx = Ctx(run=run, src=src, ref_dir=ref, pepper_file=Path(id_pepper_file) if id_pepper_file else None, cfg=cfg, cat=cat, contract=contract,
              mapping=mapping, dictionary=dictionary, d00=d00, espec=espec, threads=threads, protected=protected, input_sha=input_sha)
    status, stop = "COMPLETE", None
    try:
        with threadpool_limits(limits=int(threads["blas"])):
            for fn in _stages():
                fn(ctx)
                if stop_after and fn.__name__.startswith(stop_after):
                    status = f"STOPPED_AFTER_{stop_after}"
                    break
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
    if status == "COMPLETE":
        D.write_json(out / "RUN_STATE_FINAL.json", run.state)
    return {"status": status, "out": str(out), "attempt": attempt, "plan_sha256": plan_sha}


def effective_final_config(cfg: Any, cat: Any, *, contract: Any, dictionary: Any, mapping: Any, d00: Any,
                           allow_unfrozen: bool) -> tuple[bytes, str, bool]:
    """(bytes, sha256, frozen?) of the effective FINAL_EXPERIMENT_CONFIG. The production run must equal the repository's frozen file."""
    from falls_ml.phase2.final_config import build_final_config, load_frozen, to_bytes

    eff = to_bytes(build_final_config(cfg, cat, contract=contract, dictionary=dictionary, mapping=mapping, d00=d00))
    sha = D.sha256_bytes(eff)
    frozen_bytes, frozen_sha = load_frozen()
    if frozen_sha == sha:
        return eff, sha, True
    if not allow_unfrozen:
        detail = ["configs/meuhedet/FINAL_EXPERIMENT_CONFIG.json is missing" if frozen_bytes is None else
                  f"frozen sha256 {frozen_sha[:16]}…, effective {sha[:16]}…",
                  "the analysis settings, the feature catalogue or a result-relevant code constant differs from the reviewed freeze;",
                  "restore the patch files (never edit phase2.yaml / phase2_features.yaml for the real run)"]
        raise Phase2Stop("FROZEN_CONFIG_MISMATCH", "the effective configuration is not the frozen FINAL_EXPERIMENT_CONFIG.json", detail)
    return eff, sha, False


def _write_final_copy(out: Path, data: bytes, sha: str, *, readonly: bool) -> None:
    from falls_ml.phase2.final_config import sha_line

    for name, payload in (("FINAL_EXPERIMENT_CONFIG.json", data), ("FINAL_EXPERIMENT_CONFIG.sha256", sha_line(sha).encode("utf-8"))):
        D.write_bytes(out / name, payload)
        if readonly:
            D.mark_readonly(out / name)


def _verify_final_copy(out: Path, sha: str) -> None:
    p = out / "FINAL_EXPERIMENT_CONFIG.json"
    if not p.is_file() or D.sha256_file(p) != sha:
        raise Phase2Stop("PLAN_CHANGED", "--resume: FINAL_EXPERIMENT_CONFIG.json of this run is missing or differs from the effective configuration",
                         ["the configuration frozen at the first start is the only one this run may use"])


def _write_stop(out: Path, exc: Phase2Stop) -> None:
    title = "INVESTIGATION" if exc.kind != STOP_HARD else "STOPPED"
    lines = [f"# {title} - {exc.gate}", "", WATERMARK, "", exc.message, "", *[f"- {d}" for d in exc.details], ""]
    if exc.kind != STOP_HARD:
        lines += ["This is an investigation stop: nothing was rejected automatically. Review the evidence (aggregate tables in artifacts/),",
                  "then either fix the cause and resume, or continue with a recorded justification:", "",
                  f"    --resume --accept-gate {exc.gate} --reason \"<your reason>\"", ""]
    else:
        lines += ["This is a hard stop: fix the cause, then run the same command again with --resume.", ""]
    D.write_str(out / (f"INVESTIGATION_{exc.gate}.md" if exc.kind != STOP_HARD else "STOPPED.md"), "\n".join(lines))
