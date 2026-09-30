"""``falls_ml meuhedet-phase3 ... --preflight``: every check that can fail before the real run, with NO model fitted and NOTHING written to
``--out`` (not even created). Row-level work (reading the extract, rebuilding the reference cohort, reproducing the split, the FULL_LABELED
extension, the row-level recovery) happens in memory and in a temporary folder that is deleted.

It prints two separate verdicts:
  - SOFTWARE: the last line is exactly ``SAFE TO START FULL RUN`` or ``NOT SAFE TO START FULL RUN - <n> problem(s)`` (exit code 0 / 2);
  - SCIENCE (a forecast, not a permission): the pre-declared feasibility decision the run will take at S05 (GO -> Outcome A modelling;
    NO_GO -> Outcome B, DWH remediation, no model). Software readiness never implies scientific permission to train.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from falls_ml.phase2 import durable as D
from falls_ml.phase2.preflight import MIN_FREE_GB, Report, _environment
from falls_ml.phase2.state import Phase2Stop
from falls_ml.phase3.config import DEFAULT_CONFIG

SAFE_LINE = "SAFE TO START FULL RUN"


class _PreCtx:
    """The attributes of the run context that the recovery functions read (no run folder)."""

    def __init__(self, L: dict[str, Any], work: Any, facts: dict[str, Any]):
        self.__dict__.update({k: L[k] for k in ("cfg", "mapping", "contract", "dictionary", "d00", "rules", "cat", "espec", "tc")})
        self._work, self._facts = work, facts

    def work(self) -> Any:
        return self._work

    def facts(self) -> dict[str, Any]:
        return self._facts

    @property
    def baseline(self) -> list[str]:
        return list(self._facts["baseline_features"])


def _check(r: Report, ok: Any, what: str, detail: str, fix: str) -> None:
    if ok:
        r.ok(what, detail)
    else:
        r.fail(what, detail, fix)


def _paths(r: Report, src: Path, protected: list[Path], out: Path, phase2_out: Path | None, allow_synced: bool) -> bool:
    r.section("2. Paths (input, reference / protected folders, Phase 2 folder, output)")
    good = True
    if src.is_file():
        r.ok("input readable", f"{src.name} ({src.stat().st_size / 2**20:.1f} MB)")
    else:
        good = False
        r.fail("input file not found", str(src.name), "check the --input path")
    for p in protected:
        if not p.is_dir():
            good = False
            r.fail("reference / protected folder not found", p.name, "check --reference / --protect")
        else:
            r.ok("read-only folder", p.name)
        if p.is_dir() and (out.resolve() == p.resolve() or p.resolve() in out.resolve().parents):
            good = False
            r.fail("--out is inside a protected folder", p.name, "choose a new folder outside every earlier result folder")
    if phase2_out is not None:
        if out.resolve() == phase2_out.resolve() or phase2_out.resolve() in out.resolve().parents:
            good = False
            r.fail("--out is inside the Phase 2 folder", phase2_out.name, "Phase 3 must write to its own, separate folder")
        else:
            r.ok("Phase 2 folder is only read (two small status files), never written or hashed", phase2_out.name)
    if out.exists() and any(out.iterdir()):
        good = False
        r.fail("--out is not empty", out.name, "choose a NEW folder (to continue an existing Phase 3 run use --resume, not --preflight)")
    else:
        r.ok("--out", f"{out.name} {'exists and is empty' if out.exists() else 'does not exist yet (the run creates it; the preflight does not)'}")
    synced = D.synced_folder(out.parent if not out.exists() else out)
    if synced and not allow_synced:
        good = False
        r.fail("--out is inside a synchronised folder", synced, "use a local folder outside OneDrive")
    base = out.resolve()
    while not base.exists() and base != base.parent:
        base = base.parent
    free = shutil.disk_usage(base).free / 2**30
    (r.ok if free >= MIN_FREE_GB else r.fail)("free disk space", f"{free:.1f} GB")
    if free < MIN_FREE_GB:
        good = False
    return good


def run_preflight(input_path: str | Path, reference_dir: str | Path, *, out_dir: str | Path, protect: list[str | Path] | None = None,
                  id_pepper_file: str | Path | None = None, phase2_out: str | Path | None = None, config_path: str | Path = DEFAULT_CONFIG,
                  allow_synced_folder: bool = False, allow_unfrozen_config: bool = False, printer: Callable[[str], None] = print) -> int:
    from falls_ml.artifacts import source_tree_sha256
    from falls_ml.data.dataset import sha256_file
    from falls_ml.phase2.engineer import engineer
    from falls_ml.phase3.cohort import build_phase3_cohort
    from falls_ml.phase3.recovery import build_recovery
    from falls_ml.phase3.runner import effective_final_config, load_all
    from falls_ml.phase3.stages_data import feasibility_decision, phase2_eligibility, read_columns

    t0 = time.perf_counter()
    r = Report(printer)
    src, ref, out = Path(input_path), Path(reference_dir), Path(out_dir)
    protected = [ref, *[Path(p) for p in (protect or [])]]
    p2 = Path(phase2_out) if phase2_out else None
    printer("falls_ml meuhedet-phase3 PREFLIGHT - no model is fitted; nothing is written to --out; the Phase 2 installation and folders are not touched")
    _environment(r)
    paths_ok = _paths(r, src, protected, out, p2, allow_synced_folder)
    r.section("3. Configuration (frozen PHASE3_FINAL_EXPERIMENT_CONFIG)")
    L = None
    try:
        L = load_all(config_path)
        _, sha, frozen = effective_final_config(L, allow_unfrozen=allow_unfrozen_config)
        (r.ok if frozen else r.warn)("effective configuration" + (" = frozen" if frozen else " is NOT the frozen production configuration (tests only)"), f"sha256 {sha}")
        r.ok("recovery rules", f"{Path(L['rules'].path).name} v{L['rules'].version} sha256 {L['rules'].sha256}")
        r.ok("feature catalogue (unchanged Phase 2)", f"{len(L['cat'].features)} engineered features, sha256 {L['cat'].sha256}")
    except Phase2Stop as exc:
        r.fail(exc.gate, exc.message, "; ".join(exc.details))
    except Exception as exc:  # noqa: BLE001
        r.fail("configuration invalid", f"{type(exc).__name__}: {exc}", "restore the package files")
    r.ok("code", f"falls_ml source sha256 {source_tree_sha256()}")
    forecast = None
    if paths_ok and L is not None:
        r.section("4. Data: schema, reference cohort + split (in memory), FULL_LABELED, row-level recovery (record dates only)")
        tmp = Path(tempfile.mkdtemp(prefix="falls_ml_p3_preflight_"))
        try:
            input_sha = sha256_file(src)
            r.ok("input sha256", input_sha)
            pre = _PreCtx(L, None, {"baseline_features": []})
            work, facts = build_phase3_cohort(src, ref, contract=L["contract"], mapping=L["mapping"], spec=L["espec"], pepper_file=id_pepper_file,
                                              scratch=tmp, input_sha=input_sha, split_cfg=L["cfg"]["population"]["extra_rows_split"],
                                              schema_columns=read_columns(pre), time_contract=L["tc"], dictionary=L["dictionary"], d00=L["d00"])
            s = facts["schema"]
            r.ok("schema", f"{s['n_columns']} columns (contract {s['n_contract_columns']}), {s['n_rows']:,} rows; date layouts {s['declared_date_formats']}; "
                           f"no structural problem; {s['n_cell_problems_in_read_columns']} cell-level problem(s) in the {s['strict_columns']} columns "
                           f"Phase 3 reads (-> UNKNOWN cells {s.get('unreadable_cells_in_population', {})}); {s['n_warnings_other_columns']} counted warnings elsewhere")
            r.ok("reference cohort D00_CLEAN rebuilt and identical", f"{facts['cohort_rows']:,} rows; split reproduced (TEST sha256 equal); TEST dropped in memory")
            P = facts["phase3_partitions"]
            r.ok("FULL_LABELED (no row removed for a record date)",
                 f"{facts['full_labeled_rows']:,} rows = D00_CLEAN + {facts['d00_rows_in_full_labeled']:,} D-00 rows; TRAIN {P['train']['n_rows']:,} "
                 f"({P['train']['n_events']:,} events), VALIDATION {P['validation']['n_rows']:,}; TEST-assigned D-00 rows dropped {facts['extra_test_rows_dropped']:,}")
            tcv = facts.get("time_contract") or {}
            r.section("4b. TIME CONTRACT (corrected: prediction at the END of Index_Date; outcome from Index_Date + 1) - read-only checks, TRAIN + VALIDATION")
            v1, v2, v3 = tcv.get("V1_label_excludes_index_day", {}), tcv.get("V2_window_end", {}), tcv.get("V3_no_post_index_records", {})
            _check(r, v1.get("passed"), "V1 outcome excludes Index_Date", f"{v1.get('positives_event_on_or_before_index')} of {v1.get('positives')} positive labels "
                                                   f"have an event on/before Index_Date (on the index day: {v1.get('positives_event_on_index_day')}); min days to event "
                                                   f"{v1.get('min_days_to_event')}", "confirm the label definition with the DWH developer (the contract is contradicted)")
            _check(r, v2.get("passed"), "V2 window end = Index_Date + 180", str(v2.get("label_end_minus_index_days")), "confirm the label window with the DWH")
            rc = facts.get("cohort_reconciliation") or {}
            _check(r, rc.get("passed"), "V9 cohort reconciliation", f"extract {rc.get('extract_rows')}, on index date {rc.get('rows_on_index_date')}, eligible "
                   f"{rc.get('eligible_on_index_date')}, label NULL {rc.get('label_null_eligible')} {rc.get('label_null_by_reason')}, FULL_LABELED "
                   f"{rc.get('full_labeled')} = D00_CLEAN {rc.get('d00_clean')} + D-00 {rc.get('d00_rows')} (reference removed {rc.get('reference_rows_removed_for_d00')})",
                   "the counts must reconcile with the reference build")
            v7 = tcv.get("V7_proxy_episode_audit") or {}
            (r.warn if v7.get("investigation") else r.ok)("V7 fall-recency proxy / episode audit (TRAIN)",
                                                          f"{v7.get('interpretation')}; next-minus-last-fall gap {v7.get('gap_next_minus_last_fall')}; early share recent "
                                                          f"{v7.get('recent_fallers_early_share')} vs others {v7.get('others_early_share')}; Days_Since consistent "
                                                          f"{v7.get('days_since_consistent')} / inconsistent {v7.get('days_since_inconsistent')}")
            r.info("V8 availability indicators", str(tcv.get("V8_availability_indicators")))
            if v3.get("passed"):
                r.ok("V3 no predictor record dated after Index_Date", f"{len(v3.get('columns_checked', []))} record-date columns checked; attestation HOLDS")
            else:
                r.warn("V3 records dated AFTER Index_Date exist", f"{v3.get('rows_with_any_post_index_record')} rows; those values are UNKNOWN (bounded) and the "
                       "attestation for sources without a row-level date is WITHDRAWN")
                for c, v in (v3.get("by_column") or {}).items():
                    if v.get("n_after_index"):
                        r.info(f"   {c}", f"{v['n_after_index']} rows after Index_Date (max {v['max_days_after']} days)")
            for c, v in (v3.get("by_column") or {}).items():
                if v.get("n_on_index"):
                    r.info(f"   {c}", f"{v['n_on_index']} rows ON Index_Date (history under the corrected contract; were D-00 flags before)")
            v4 = tcv.get("V4_days_since_consistency", {})
            (r.ok if v4.get("passed") else r.warn)("V4 Days_Since_* consistent with their dates", str({k: v for k, v in (v4.get("by_column") or {}).items() if v.get("n_mismatch") or v.get("n_negative")}))
            r.info("V5 index-day fall x label", str(tcv.get("V5_index_day_fall_vs_label")))
            v6 = tcv.get("V6_boundary_episode", {})
            (r.warn if v6.get("investigation") else r.ok)("V6 boundary episodes (TRAIN)", str(v6))
            pre = _PreCtx(L, work, facts)
            values = engineer(work, L["cat"])
            p2elig, p2f = phase2_eligibility(pre, values)
            r.info("Phase 2 rule on the Phase 2 population (compare with the Phase 2 preflight)",
                   f"{p2f['catalogue_eligibility_counts']}; historical predictors SAFE: {p2f['n_baseline_safe']} of {len(p2f['baseline_status'])}")
            tr = (work["partition"] == "train").to_numpy()
            res = build_recovery(work, values, catalogue=L["cat"], mapping=L["mapping"], baseline=list(facts["baseline_features"]), contract=L["contract"],
                                 dictionary=L["dictionary"], d00=L["d00"], rules=L["rules"], train_mask=tr,
                                 min_observed_train=int(L["cfg"]["eligibility"]["min_observed_train_rows"]), phase2_eligibility=p2elig, tc=L["tc"],
                                 attestation_holds=bool(tcv.get("attestation_holds")))
            reg = res.registry
            cats = reg[reg["kind"] == "catalogue"]["phase3_class"].value_counts().to_dict()
            r.ok("feature eligibility (93 engineered features, corrected contract)", str(cats))
            base = reg[reg["kind"] == "baseline"]
            r.info("historical predictors by class", str(base.groupby("phase3_class")["feature"].apply(list).to_dict()))
            r.info("newly eligible (not usable in Phase 2)", f"{int(reg['newly_recovered'].astype(bool).sum())} features")
            v2 = values.copy()
            for f in [c for c in res.unknown.columns if c in v2.columns]:
                v2.loc[res.unknown[f].to_numpy(dtype=bool), f] = float("nan")
            forecast = feasibility_decision(reg, v2, res.unknown, work["y"].to_numpy(dtype=int), tr, L["cfg"], bool(tcv.get("hard_passed")))
        except Phase2Stop as exc:
            r.fail(exc.gate, exc.message, "; ".join(exc.details[:5]))
        except Exception as exc:  # noqa: BLE001
            r.fail("data check failed", f"{type(exc).__name__}: {str(exc)[:500]}", "see the message; nothing was written")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    r.section("5. Scientific feasibility FORECAST (the run decides at S05; software readiness is not scientific permission)")
    if forecast is not None:
        t = forecast["table"]
        for _, row in t.iterrows():
            r.info(f"{row['domain']}", f"eligible {int(row['n_eligible'])} (verified {int(row['n_verified'])}, attested {int(row['n_attested'])}), newly eligible "
                                       f"items {int(row['n_newly_eligible_items'])}, KNOWN informative TRAIN rows {int(row['n_known_informative_train'])}, "
                                       f"{'GO' if row['domain_go'] else '-'}")
        printer(f"  SCIENTIFIC FORECAST: {forecast['decision']} -> Outcome {forecast['outcome']} "
                f"({'extended modelling will run' if forecast['decision'] == 'GO' else 'no model will be fitted; DWH remediation report'})")
    else:
        printer("  SCIENTIFIC FORECAST: not available (see the problems above)")
    printer("")
    printer(f"preflight finished in {time.perf_counter() - t0:.0f} s; {len(r.warnings)} warning(s), {len(r.problems)} problem(s)")
    if r.problems:
        for p in r.problems:
            printer(f"  PROBLEM: {p}")
        printer(f"NOT SAFE TO START FULL RUN - {len(r.problems)} problem(s)")
        return 2
    printer(SAFE_LINE)
    return 0
