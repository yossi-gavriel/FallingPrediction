"""SAFE-ALL-ROWS sensitivity, ablations, provenance and convergence audit (v0.5.1). Synthetic fixtures only - no real data."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest
import yaml

from falls_ml.cli import main as cli_main
from falls_ml.data.meuhedet_synthetic import write_synthetic_wide_extract
from falls_ml.data.meuhedet_timing import timing_diagnostic
from falls_ml.data.meuhedet_wide import (DEFAULT_CONTRACT, DEFAULT_MAPPING, MeuhedetWideDatasetAdapter, load_wide_contract, load_wide_mapping,
                                         read_wide_extract_report)
from falls_ml.errors import ConfigError, DatasetValidationError
from falls_ml.features.spec import load_feature_spec
from falls_ml.meuhedet_explore import explore
from falls_ml.meuhedet_sensitivity import (FINAL_KEY, NOT_RUN, SAFE_KEY, SAFE_LABEL, convergence_audit, directory_digest, feature_provenance,
                                           render_convergence_markdown, sensitivity)

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "configs" / "experiments" / "meuhedet" / "explore_180d_template.yaml"
ID_RE = re.compile(r"\bS\d{10}\b")


@pytest.fixture(scope="module")
def contract():
    return load_wide_contract(DEFAULT_CONTRACT)


@pytest.fixture(scope="module")
def mapping(contract):
    return load_wide_mapping(DEFAULT_MAPPING, contract=contract)


@pytest.fixture(scope="module")
def spec(mapping):
    return load_feature_spec(mapping.exploratory_spec_path)


def plant(csv: Path, *, n_dx: int = 60, n_registry: int = 12) -> tuple[int, int]:
    """Same-day diagnosis records on n_dx eligible rows and index-day first registry entries on n_registry other eligible rows."""
    raw = pd.read_csv(csv, dtype=str, keep_default_na=False)
    # eligible AND labelled rows only, so the planted counts are exactly what the build's D-00 stage sees (label-null rows leave earlier)
    elig = (raw["Is_Eligible_Cohort"] == "1") & (raw["Fall_Next_180D_Ind"].isin(["0", "1"]))
    idx = raw.index[elig & (raw["Last_Dx_Date"] != "NULL")][:n_dx]
    raw.loc[idx, "Last_Dx_Date"] = raw.loc[idx, "Index_Date"]
    raw.loc[idx, "Days_Since_Last_Diagnosis"] = "0"
    no_fall = raw["Last_Fall_Date"] != raw["Index_Date"]   # registry-evidence rows must not also be D-00 rows of the reference
    reg = [i for i in raw.index[elig & no_fall & (raw["Registry_Missing_Ind"] == "0")] if i not in set(idx)][:n_registry]
    raw.loc[reg, "First_Registry_Date"] = raw.loc[reg, "Index_Date"]
    raw.loc[reg, "Days_Since_First_Registry"] = "0"
    raw.to_csv(csv, index=False, lineterminator="\n")
    return len(idx), len(reg)


def small_template(tmp_path: Path) -> Path:
    tpl = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
    tpl["evaluation"]["bootstrap"]["n"] = 20
    tpl["evaluation"]["permutation_importance_repeats"] = 1
    tpl["analysis"]["stability"]["n_bootstrap"] = 4
    tpl["analysis"]["optimism"]["n_bootstrap"] = 2
    p = tmp_path / "template.yaml"
    p.write_text(yaml.safe_dump(tpl, sort_keys=False), encoding="utf-8")
    return p


def shared_text(out: Path) -> str:
    return "".join(p.read_text(encoding="utf-8", errors="ignore") for p in out.rglob("*")
                   if p.suffix in {".md", ".json", ".csv", ".yaml", ".html", ".jsonl", ".txt"} and p.name != "id_pepper.txt" and "datasets" not in p.parts)


@pytest.fixture(scope="module")
def planted(tmp_path_factory, contract):
    tmp = tmp_path_factory.mktemp("planted")
    csv = write_synthetic_wide_extract(tmp / "wide.csv", n_rows=3000, seed=11, other_index_date_share=0.0, csv_date_format="%d/%m/%Y", n_index_day_falls=5)
    n_dx, n_reg = plant(csv)
    read = read_wide_extract_report(csv, contract)
    return {"csv": csv, "frame": read.frame, "n_dx": n_dx, "n_reg": n_reg, "dir": tmp}


# ------------------------------------------------------------------ mapping + adapter: the timing scopes
def test_mapping_declares_registry_evidence_and_the_cohort_guard_is_unchanged(mapping, contract) -> None:
    ev = {f.canonical: f.same_day_evidence_column for f in mapping.features if f.same_day_evidence_column}
    assert set(ev.values()) == {"First_Registry_Date"} and len(ev) == 10
    assert all(mapping.get(n).source_columns[0].startswith("Registry_") for n in ev)
    assert list(mapping.cohort["predictor_max_record_date"]["columns"]) == ["Last_Fall_Date", "Last_Dx_Date"]
    assert contract.get("First_Registry_Date").is_date and contract.get("First_Registry_Date").timing == "at_index"
    assert "Q-M-11" in mapping.open_questions


def test_adapter_scope_built_predictors_keeps_d00_rows_of_unread_sources_and_excludes_registry_evidence_rows(planted, mapping, contract, spec) -> None:
    sets = mapping.feature_sets()
    safe = [f for f in sets["extended"] if f not in ("falls", "mobility_problems")]
    cohort = MeuhedetWideDatasetAdapter(mapping, contract, spec, features=sets["extended"], timing_scope="cohort")
    _, rep_c = cohort.apply(planted["frame"], index_date="2025-01-01", index_day_records="drop_rows")
    built = MeuhedetWideDatasetAdapter(mapping, contract, spec, features=safe, timing_scope="built_predictors")
    assert built.timing_guard_columns == ["First_Registry_Date"]
    out, rep_b = built.apply(planted["frame"], index_date="2025-01-01", index_day_records="drop_rows")
    assert rep_b.timing_scope == "built_predictors" and rep_b.timing_guard_columns == ["First_Registry_Date"]
    assert rep_b.timing_violations_by_column == {"First_Registry_Date": planted["n_reg"]} and rep_b.n_rows_timing_violation == planted["n_reg"]
    assert rep_b.cohort_guard_rows_retained_by_column["Last_Dx_Date"] == planted["n_dx"] and rep_b.cohort_guard_rows_retained >= planted["n_dx"]
    # rows: the cohort scope dropped the D-00 rows; the built-predictors scope keeps them and drops only the evidence rows
    assert rep_c.n_rows_timing_violation >= planted["n_dx"] and rep_c.timing_scope == "cohort" and rep_c.cohort_guard_rows_retained == 0
    assert rep_b.n_rows_final == rep_c.n_rows_final + rep_c.n_rows_timing_violation - planted["n_reg"]
    assert (out["predictor_max_record_date"] < out["index_date"]).all()
    # a build that includes an implicated predictor guards on its own record-date column even in the built_predictors scope
    with_falls = MeuhedetWideDatasetAdapter(mapping, contract, spec, features=[*safe, "mobility_problems"], timing_scope="built_predictors")
    assert with_falls.timing_guard_columns == ["First_Registry_Date", "Last_Dx_Date"]
    with pytest.raises(DatasetValidationError, match="D-00 timing"):
        with_falls.apply(planted["frame"], index_date="2025-01-01", index_day_records="fail")
    with pytest.raises(ConfigError, match="timing_scope"):
        MeuhedetWideDatasetAdapter(mapping, contract, spec, features=safe, timing_scope="anything")


# ------------------------------------------------------------------ provenance
def test_provenance_excludes_exactly_the_proven_same_day_sources_and_traces_the_requested_columns(planted, mapping, contract) -> None:
    timing = timing_diagnostic(planted["frame"], contract, mapping)
    assert timing["columns"]["First_Registry_Date"]["n_on_index"] == planted["n_reg"] and timing["same_day_evidence_columns"]["First_Registry_Date"]["features"]
    prov = feature_provenance(mapping, contract, timing)
    assert prov["excluded_from_safe_all_rows"] == ["falls", "mobility_problems"]
    assert set(prov["safe_all_rows_feature_set"]) == set(mapping.feature_sets()["extended"]) - {"falls", "mobility_problems"}
    by = {e["feature"]: e for e in prov["features"]}
    assert by["falls"]["provenance_class"] == "EXCLUDED_PROVEN_SAME_DAY_SOURCE" and by["falls"]["d00_evidence"]["column"] == "Last_Fall_Date"
    assert by["mobility_problems"]["provenance_class"] == "EXCLUDED_PROVEN_SAME_DAY_SOURCE" and by["mobility_problems"]["d00_evidence"]["n_on_index"] >= planted["n_dx"]
    assert by["age_years"]["provenance_class"] == by["sex"]["provenance_class"] == "DERIVED_FROM_IMMUTABLE_ATTRIBUTES"
    assert by["dementia"]["provenance_class"] == "STATE_AT_INDEX_WITH_PARTIAL_EVIDENCE" and by["dementia"]["d00_evidence"]["kind"] == "same_day_evidence_sufficient_only"
    assert by["polypharmacy_count_120d"]["provenance_class"] == "STATE_AT_INDEX_UNVERIFIABLE"
    req = {r["name"]: r for r in prov["requested_trace"]}
    assert set(req) >= {"Diagnosis_Count_180D", "Prior_Fall_Count_365D", "Days_Since_Last_Fall", "Gait_Disorder_Since_Study_Start_Ind", "falls"}
    assert req["Gait_Disorder_Since_Study_Start_Ind"]["kind"] == "source column" and "mobility_problems" in req["Gait_Disorder_Since_Study_Start_Ind"]["resolution"]
    assert req["Diagnosis_Count_180D"]["kind"] == "Phase-2 column" and "not a modelling predictor" in req["Diagnosis_Count_180D"]["resolution"]
    p2 = {p["column"] for p in prov["phase2_columns_of_implicated_sources"]}
    assert {"Diagnosis_Count_365D", "Distinct_Diagnosis_Codes_365D", "Chronic_Diagnosis_Count_365D", "Prior_Fall_Count_30D", "Days_Since_Last_Fall"} <= p2
    assert "First_Registry_Date on/after Index_Date" in prov["safe_row_rule"]


def test_provenance_keeps_event_sources_that_are_clean_in_the_extract(mapping, contract, tmp_path) -> None:
    csv = write_synthetic_wide_extract(tmp_path / "clean.csv", n_rows=800, seed=3, other_index_date_share=0.0)
    timing = timing_diagnostic(read_wide_extract_report(csv, contract).frame, contract, mapping)
    prov = feature_provenance(mapping, contract, timing)
    assert prov["excluded_from_safe_all_rows"] == [] and len(prov["safe_all_rows_feature_set"]) == len(mapping.feature_sets()["extended"])
    assert {e["provenance_class"] for e in prov["features"] if e["record_date_column"]} == {"EVENT_SOURCE_CLEAN_IN_THIS_EXTRACT"}


# ------------------------------------------------------------------ convergence audit
def _log(events: list[dict]) -> str:
    return "\n".join(json.dumps({"ts": "t", "logger": "x", **e}) for e in events) + "\n"


def test_convergence_audit_separates_final_fit_from_replicates_and_flags_the_grid_boundary(tmp_path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.yaml").write_text(yaml.safe_dump({"analysis": {"stability": {"enabled": True, "n_bootstrap": 3}, "optimism": {"enabled": True, "n_bootstrap": 2}},
                                                    "evaluation": {"bootstrap": {"n": 10}}}), encoding="utf-8")
    events = [{"level": "INFO", "event": "run_started"},
              {"level": "INFO", "event": "fp_selection_done", "forms": {"age_years": {"powers": [1.0]}}},
              {"level": "WARNING", "event": "lasso_path_not_converged", "n_not_converged": 3},
              {"level": "WARNING", "event": "lasso_cv_fold_paths_not_converged"},
              {"level": "WARNING", "event": "lasso_cv_minimum_not_identified"},
              {"level": "INFO", "event": "lasso_cv_fitted", "lambda_star_index": 60, "lambda_stop_index": 70, "path_converged": True, "cv_paths_converged": False,
               "cv_minimum_identified": False},
              {"level": "INFO", "event": "fp_selection_done", "forms": {"age_years": {"powers": [1.0]}}},
              {"level": "WARNING", "event": "fp_selection_stata_omissions", "perfect_predictors": [{"column": "chronic_kidney_disease"}]},
              {"level": "WARNING", "event": "lasso_unpenalized_refit_warnings", "warnings": ["x"], "converged": False},
              {"level": "INFO", "event": "stability_replicate", "replicate": 0},
              {"level": "INFO", "event": "fp_selection_done", "forms": {"age_years": {"powers": [0.5]}}},
              {"level": "WARNING", "event": "fp_selection_stata_omissions", "perfect_predictors": [{"column": "chronic_kidney_disease"}]},
              {"level": "INFO", "event": "stability_replicate", "replicate": 1},
              {"level": "INFO", "event": "bootstrap_stability_done"},
              {"level": "WARNING", "event": "bootstrap_replicate_failed", "reason": "degenerate fit: single class"},
              {"level": "INFO", "event": "harrell_optimism_done"}]
    (run / "run_log.jsonl").write_text(_log(events), encoding="utf-8")
    metrics = {"experiment": {"name": "X"}, "lasso": {"lambda_star": 0.01, "lambda_max": 1.0, "lambda_ratio": 1e-4, "n_lambda": 100, "cv_minimum_identified": False,
                                                     "cv_paths_converged": False, "n_selected": 5}, "warnings": []}
    a = convergence_audit(run, metrics)
    assert a["final_fit"]["warnings"] == {"lasso_path_not_converged": 1, "lasso_cv_fold_paths_not_converged": 1, "lasso_cv_minimum_not_identified": 1}
    assert a["replicates"]["stability_warnings"] == {"fp_selection_stata_omissions": 2, "lasso_unpenalized_refit_warnings": 1}
    assert a["replicates"]["optimism_warnings"] == {"bootstrap_replicate_failed": 1}
    assert a["replicates"]["perfect_predictor_omissions_by_column"] == {"chronic_kidney_disease": 2}
    assert a["replicates"]["fp_form_instability"]["age_years"]["share_with_final_form"] == 0.5 and a["replicates"]["resampling"]["stability_n"] == 3
    g = a["final_fit"]["grid"]
    assert g["lambda_star_index"] == 60 and g["grid_points_to_end"] == 39 and not g["at_grid_end"] and not g["near_grid_end"]
    assert a["verdict"] == "VALID WITH QUALIFICATIONS" and a["solver_or_grid_change_warranted"] == [] and a["final_fit"]["unpenalised_refit"] is None
    assert any("flat" in q for q in a["qualifications"]) and any("unaffected" in q for q in a["qualifications"])
    # grid boundary: lambda* on the smallest grid value -> questionable, grid extension warranted; index reconstructed from metrics when the log lacks it
    (run / "run_log.jsonl").write_text(_log([e for e in events if e["event"] != "lasso_cv_fitted"]), encoding="utf-8")
    m2 = {**metrics, "lasso": {**metrics["lasso"], "lambda_star": 1.0 * (1e-4 ** 1.0), "cv_minimum_identified": True, "cv_paths_converged": True}}
    b = convergence_audit(run, m2)
    assert b["final_fit"]["grid"]["lambda_star_index"] == 99 and b["final_fit"]["grid"]["at_grid_end"] and b["verdict"].startswith("QUESTIONABLE")
    assert b["solver_or_grid_change_warranted"] and "lambda_min_ratio" in b["solver_or_grid_change_warranted"][0]
    text = render_convergence_markdown({"x": a, "y": b}, "WM")
    assert "VALID WITH QUALIFICATIONS" in text and "grid boundary" in text and "chronic_kidney_disease" in text and "significan" not in text.lower()


# ------------------------------------------------------------------ end to end (synthetic reference -> sensitivity package)
@pytest.mark.slow
def test_sensitivity_package_end_to_end_preserves_the_reference_and_separates_feature_and_population_effects(planted, tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(ROOT)
    csv = planted["csv"]
    tpl = small_template(tmp_path)
    ref = explore(csv, out_dir=tmp_path / "ref", index_day_records="drop_rows", template_path=tpl, fast=False)
    assert ref.ready and ref.build_reports["extended"].n_rows_timing_violation >= planted["n_dx"]
    ref_digest = directory_digest(tmp_path / "ref")
    # the CLI refuses the reference directory as output and a foreign pepper; then runs the package
    assert cli_main(["meuhedet-sensitivity", "--input", str(csv), "--reference", str(tmp_path / "ref"), "--out", str(tmp_path / "ref")]) == 2
    assert cli_main(["meuhedet-sensitivity", "--input", str(csv), "--reference", str(tmp_path / "ref"), "--out", str(tmp_path / "bad"), "--id-pepper", "other"]) == 2
    err = capsys.readouterr().err
    assert "pepper differs" in err and not (tmp_path / "bad" / "runs").exists()
    res = sensitivity(csv, tmp_path / "ref", out_dir=tmp_path / "sens")
    out = res.out_dir
    # 0. v0.6.0: both commands finished with their model reports; the combined comparison pairs only identical test rows
    assert ref.model_report["status"] == "built" and res.model_report["status"] == "built", (ref.model_report, res.model_report)
    notes = pd.read_csv(out / "reports" / "tables" / "model_comparison.csv").set_index("analysis")["comparison_note"]
    assert notes["EXTENDED"] == "reference" and notes["SAFE_ALL_ROWS"].startswith("NOT A DIRECT PAIRED MODEL COMPARISON")
    assert all(notes[k].startswith("paired") for k in notes.index if k.startswith("ABLATION_") or k == "STRICT")
    # 1. the reference is untouched (every file hashed before and after)
    assert res.integrity["unchanged"] and directory_digest(tmp_path / "ref") == ref_digest
    # 2. analyses present: reference STRICT/EXTENDED (read), three ablations on identical rows, SAFE-ALL-ROWS on all rows; FINAL_DWH_FIXED documented only
    keys = list(res.records)
    assert keys[:2] == ["reference_strict", "reference_extended"]
    assert {"ablation_no_falls", "ablation_no_mobility_problems", "ablation_no_falls_no_mobility_problems", SAFE_KEY} <= set(keys) and not res.skipped
    ref_sha = ref.split_audit["test_rows_sha256"]
    for k in ("ablation_no_falls", "ablation_no_mobility_problems", "ablation_no_falls_no_mobility_problems"):
        r = res.records[k]
        assert r["test_rows_sha256"] == ref_sha and r["n_rows"] == ref.build_reports["extended"].n_rows_final and r["timing_related_exclusion"]["n_rows_excluded"] == ref.build_reports["extended"].n_rows_timing_violation
    assert res.records["ablation_no_falls"]["n_predictors"] == 14 and res.records["ablation_no_falls_no_mobility_problems"]["n_predictors"] == 13
    assert sorted(res.records["ablation_no_falls_no_mobility_problems"]["predictors"]) == sorted(res.records[SAFE_KEY]["predictors"])
    safe = res.records[SAFE_KEY]
    br = res.build_reports[SAFE_KEY]
    assert safe["label"] == SAFE_LABEL and safe["test_rows_sha256"] != ref_sha and br.timing_scope == "built_predictors"
    assert safe["n_rows"] == ref.build_reports["extended"].n_rows_final + ref.build_reports["extended"].n_rows_timing_violation - planted["n_reg"]
    assert safe["timing_related_exclusion"]["n_rows_excluded"] == planted["n_reg"] and safe["timing_related_exclusion"]["cohort_guard_rows_retained"] >= planted["n_dx"]
    assert set(safe["predictors_excluded"]) == {"falls", "mobility_problems"}
    pop = res.population
    assert not pop["paired"] and pop["rows_only_in_safe"]["n_rows"] == ref.build_reports["extended"].n_rows_timing_violation - 0 or pop["rows_only_in_safe"]["n_rows"] > 0
    assert pop["rows_only_in_reference"]["n_rows"] == planted["n_reg"] and pop["rows_in_both"]["n_rows"] + pop["rows_only_in_safe"]["n_rows"] == safe["n_rows"]
    comp = json.loads((out / "sensitivity_comparison.json").read_text(encoding="utf-8"))
    assert comp[FINAL_KEY]["status"] == NOT_RUN and {fe["b"] for fe in comp["feature_set_effect"]} == {"reference_strict", "ablation_no_falls", "ablation_no_mobility_problems", "ablation_no_falls_no_mobility_problems"}
    assert comp["population_effect"]["same_predictors"] and not comp["population_effect"]["paired"] and "NOT a paired" in comp["population_effect"]["text"]
    # 3. every run used the reference's own config as template (same seed / proportions / resampling)
    for k, run_dir in res.runs.items():
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
        assert cfg["validation"]["seed"] == 42 and cfg["validation"]["strategy"] == "patient_grouped_random" and cfg["analysis"]["stability"]["n_bootstrap"] == 4
        assert (run_dir / "RUN_COMPLETE.json").exists() and (out / f"feature_report_{k}.csv").exists() and (out / f"adequacy_{k}.md").exists() and (out / f"split_audit_{k}.md").exists()
    # 4. reports: provenance, convergence audit (reference + new), comparison, plain-language report with the 14 sections and the watermark; no ids / pepper / 'significant'
    for name in ("feature_provenance.md", "convergence_audit.md", "SENSITIVITY_COMPARISON.md", "REAL_DATA_EXPLORATORY_REPORT.md", "SENSITIVITY_README.md", "reference_integrity.json"):
        assert (out / name).exists()
    assert set(res.audits) == set(res.records) and all(a["verdict"] for a in res.audits.values())
    report = (out / "REAL_DATA_EXPLORATORY_REPORT.md").read_text(encoding="utf-8")
    for n in range(1, 15):
        assert f"## {n}. " in report
    assert "NOT EFALLS REPRODUCTION" in report and "SYNTHETIC" in report and "FINAL_DWH_FIXED" in report and "removed at the dataset-build stage" in report
    text = shared_text(out)
    pepper = (out / "id_pepper.txt").read_text(encoding="utf-8").strip()
    assert not ID_RE.search(text) and pepper not in text and "significan" not in text.lower() and res.pepper_sha256 in text
    # 5. immutability: a second run into the same directory is refused; the reference's own directory too
    with pytest.raises(DatasetValidationError, match="not empty"):
        sensitivity(csv, tmp_path / "ref", out_dir=tmp_path / "sens")
