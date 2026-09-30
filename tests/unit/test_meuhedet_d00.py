"""D-00 sensitivity framework (falls_ml meuhedet-d00, v0.7.0; resume v0.7.1). Synthetic fixtures only - no real data.

The synthetic extract carries: clean pre-index events, same-day diagnosis events (enriched among fallers), same-day fall events, safe
predictors (age, sex), directly unsafe predictors (falls, mobility_problems), a transitively unsafe predictor (declared derivation in a test
configuration), missing values (nurse fields, prior-fall counts), sparse factors (registries), a strong prior-fall signal, a weak mobility
signal and redundant columns (prior-fall counts over nested windows)."""

from __future__ import annotations

import ast
import hashlib
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from falls_ml.cli import main as cli_main
from falls_ml.d00 import analysis as A
from falls_ml.d00.dependency import DEFAULT_D00_CONFIG, build_graph, cohort_masks, load_d00_config
from falls_ml.d00.plan import REF_EXTENDED, REF_STRICT, build_plan
from falls_ml.data.meuhedet_synthetic import write_synthetic_wide_extract
from falls_ml.data.meuhedet_wide import (DEFAULT_MAPPING, MeuhedetWideDatasetAdapter, feature_input_columns, load_wide_contract, load_wide_mapping,
                                         read_wide_extract_report)
from falls_ml.eda.dictionary import DEFAULT_DICTIONARY, load_data_dictionary
from falls_ml.errors import ConfigError, DatasetValidationError, LeakageError
from falls_ml.features.spec import load_feature_spec
from falls_ml.meuhedet_sensitivity import directory_digest

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "configs" / "experiments" / "meuhedet" / "explore_180d_template.yaml"
ID_RE = re.compile(r"\bS\d{10}\b")
UNSAFE_SOURCES = ("FALL_EVENTS", "DIAGNOSES")


# ------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def defs():
    mapping = load_wide_mapping(DEFAULT_MAPPING)
    contract = load_wide_contract(mapping.contract_path)
    dictionary = load_data_dictionary(DEFAULT_DICTIONARY, contract)
    config = load_d00_config(DEFAULT_D00_CONFIG, contract=contract, dictionary=dictionary)
    spec = load_feature_spec(mapping.exploratory_spec_path)
    return {"mapping": mapping, "contract": contract, "dictionary": dictionary, "config": config, "spec": spec}


def plant(csv: Path, *, n_rows: int, seed: int = 7) -> dict[str, int]:
    """Same-day diagnoses on ~6% of eligible labelled rows (12% of fallers, 5.5% of non-fallers: the D-00 rows differ in prevalence)."""
    raw = pd.read_csv(csv, dtype=str, keep_default_na=False)
    rng = np.random.default_rng(seed)
    elig = (raw["Is_Eligible_Cohort"] == "1") & raw["Fall_Next_180D_Ind"].isin(["0", "1"]) & (raw["Last_Dx_Date"] != "NULL")
    pos, neg = raw.index[elig & (raw["Fall_Next_180D_Ind"] == "1")], raw.index[elig & (raw["Fall_Next_180D_Ind"] == "0")]
    idx = list(rng.choice(pos, max(3, int(0.12 * len(pos))), replace=False)) + list(rng.choice(neg, int(0.055 * len(neg)), replace=False))
    raw.loc[idx, "Last_Dx_Date"] = raw.loc[idx, "Index_Date"]
    raw.loc[idx, "Days_Since_Last_Diagnosis"] = "0"
    raw.to_csv(csv, index=False, lineterminator="\n")
    return {"dx": len(idx)}


@pytest.fixture(scope="module")
def extract(tmp_path_factory, defs):
    tmp = tmp_path_factory.mktemp("d00x")
    csv = write_synthetic_wide_extract(tmp / "wide.csv", n_rows=3000, seed=11, other_index_date_share=0.0, csv_date_format="%d/%m/%Y", n_index_day_falls=8)
    planted = plant(csv, n_rows=3000)
    read = read_wide_extract_report(csv, defs["contract"])
    return {"csv": csv, "read": read, "planted": planted, "tmp": tmp}


def _graph(frame: pd.DataFrame, defs, config=None, mapping=None):
    m = mapping or defs["mapping"]
    co = cohort_masks(frame, m, "2025-01-01")
    return co, build_graph(frame, co, mapping=m, contract=defs["contract"], dictionary=defs["dictionary"], config=config or defs["config"])


def _config_variant(tmp: Path, defs, edit) -> object:
    raw = yaml.safe_load((ROOT / DEFAULT_D00_CONFIG).read_text(encoding="utf-8"))
    edit(raw)
    p = tmp / f"d00_{hashlib.sha256(json.dumps(raw, sort_keys=True, default=str).encode()).hexdigest()[:8]}.yaml"
    p.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return load_d00_config(p, contract=defs["contract"], dictionary=defs["dictionary"])


# ------------------------------------------------------------------ 1-2: dependency tracing
def test_direct_d00_dependencies_are_caught(extract, defs) -> None:
    co, g = _graph(extract["read"].frame, defs)
    assert co.counts["d00_rows_in_full_labeled"] >= extract["planted"]["dx"] and co.counts["d00_rows_by_Last_Fall_Date"] >= 1
    assert g.features["falls"]["FULL_LABELED"]["status"] == "UNSAFE" and g.features["mobility_problems"]["FULL_LABELED"]["status"] == "UNSAFE"
    assert g.features["falls"]["D00_CLEAN"]["status"] == "SAFE" and g.features["mobility_problems"]["D00_CLEAN"]["status"] == "SAFE"
    assert "Last_Dx_Date" in g.features["mobility_problems"]["FULL_LABELED"]["explanation"]
    for f in ("age_years", "sex"):
        assert g.features[f]["FULL_LABELED"]["status"] == "SAFE"
    for f in ("dementia", "copd", "polypharmacy_count_120d"):
        assert g.features[f]["FULL_LABELED"]["status"] == "UNRESOLVED"   # first-entry evidence only / no record date: never "proven safe"
    # every requested Phase-2 fall / diagnosis column is traced to its source
    for col in ("Prior_Fall_Count_30D", "Prior_Fall_Count_365D", "Days_Since_Last_Fall", "Diagnosis_Count_180D", "Chronic_Diagnosis_Count_365D",
                "Distinct_Diagnosis_Codes_365D", "Gait_Disorder_Since_Study_Start_Ind", "Prior_Fall_Since_Study_Start_Ind"):
        assert g.columns[col]["FULL_LABELED"]["status"] == "UNSAFE", col
    assert g.features["falls"]["mapping_vs_dictionary"] == "agree"


def test_transitive_dependencies_are_caught(extract, defs) -> None:
    tmp = extract["tmp"]

    def derive(raw):   # hypothetical: the COPD registry is populated from diagnosis records; liver registry via a two-step chain
        raw["derived_from"]["Registry_COPD_Ind"] = {"from": ["Chronic_Diagnosis_Count_365D"], "evidence": "test"}
        raw["derived_from"]["Registry_Liver_Ind"] = {"from": ["Days_Since_Last_Fall"], "evidence": "test"}

    cfg = _config_variant(tmp, defs, derive)
    _co, g = _graph(extract["read"].frame, defs, config=cfg)
    assert g.features["copd"]["FULL_LABELED"]["status"] == "UNSAFE" and "DIAGNOSES" in g.features["copd"]["sources"]
    liver = g.features["liver_problems"]
    assert liver["FULL_LABELED"]["status"] == "UNSAFE" and "FALL_EVENTS" in liver["sources"]   # Liver <- Days_Since_Last_Fall <- Last_Fall_Date
    assert g.columns["Registry_Liver_Ind"]["derived_from"] == ["Days_Since_Last_Fall"]
    for std in g.feature_sets.values():
        assert "copd" not in std["STRICT_SAFE"]["features"] and "liver_problems" not in std["SAFE_EXTENDED"]["features"]
    # without the mapping's own record_date_column the dictionary still finds the diagnosis source (safety is never read from the mapping alone)
    raw = yaml.safe_load((ROOT / DEFAULT_MAPPING).read_text(encoding="utf-8"))
    del raw["features"]["mobility_problems"]["record_date_column"]
    raw["cohort"]["predictor_max_record_date"]["columns"] = ["Last_Fall_Date"]
    p = tmp / "mapping_no_rdc.yaml"
    p.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8")
    m2 = load_wide_mapping(p)
    _co2, g2 = _graph(extract["read"].frame, defs, mapping=m2)
    assert g2.features["mobility_problems"]["FULL_LABELED"]["status"] == "UNSAFE"
    assert g2.features["mobility_problems"]["mapping_vs_dictionary"].startswith("DIFFER")


def test_registry_first_entry_evidence_and_unbounded_values(extract, defs) -> None:
    f = extract["read"].frame.copy()
    co = cohort_masks(f, defs["mapping"], "2025-01-01")
    rows = f.index[co["FULL_LABELED"] & f["First_Registry_Date"].notna()][:5]
    f.loc[rows, "First_Registry_Date"] = f.loc[rows, "Index_Date"]
    clean = f.index[co["D00_CLEAN"] & (f["Prior_Fall_Since_Study_Start_Ind"] == 0) & f["Last_Fall_Date"].isna()][:3]
    f.loc[clean, "Prior_Fall_Since_Study_Start_Ind"] = 1          # a recorded fall flag without any fall record date
    _co, g = _graph(f, defs)
    assert g.features["dementia"]["FULL_LABELED"]["status"] == "UNSAFE"      # first-entry date proves an index-day registry entry
    for std in g.feature_sets.values():
        assert "dementia" not in std["STRICT_SAFE"]["features"]
    assert g.features["falls"]["D00_CLEAN"]["status"] == "UNRESOLVED"
    assert "without" in g.features["falls"]["D00_CLEAN"]["explanation"] or "no Last_Fall_Date" in g.features["falls"]["D00_CLEAN"]["explanation"]


def test_statuses_never_use_outcome_values(extract, defs) -> None:
    f = extract["read"].frame
    _c1, g1 = _graph(f, defs)
    g = f.copy()
    lab = g["Fall_Next_180D_Ind"]
    keep = lab.notna()
    g.loc[keep, "Fall_Next_180D_Ind"] = np.random.default_rng(0).permutation(lab[keep].to_numpy())   # outcome values scrambled
    _c2, g2 = _graph(g, defs)
    assert g1.sha256 == g2.sha256 and g1.feature_sets == g2.feature_sets and g1.forbidden_columns == g2.forbidden_columns


# ------------------------------------------------------------------ feature sets and plan
def test_feature_sets_come_from_the_graph_and_deviations_are_renamed(extract, defs) -> None:
    _co, g = _graph(extract["read"].frame, defs)
    prim, sec = g.feature_sets["PROVEN_SAFE"], g.feature_sets["SAFE_OR_UNRESOLVED"]
    assert prim["STRICT_SAFE"]["name"] == "STRICT_SAFE_SUBSET" and not prim["STRICT_SAFE"]["identical_to_strict"]
    assert set(prim["STRICT_SAFE"]["deviation"]) == set(defs["mapping"].feature_sets()["strict"]) - {"age_years", "sex"}
    assert prim["SAFE_EXTENDED"]["features"] == ["age_years", "sex"]
    ext = defs["mapping"].feature_sets()["extended"]
    assert sorted(sec["SAFE_EXTENDED"]["features"]) == sorted(set(ext) - {"falls", "mobility_problems"})
    assert sec["STRICT_SAFE"]["identical_to_strict"] and sec["STRICT_SAFE"]["name"] == "STRICT_SAFE_OR_UNRESOLVED"

    def everything_static(raw):   # hypothetical evidence: registries and medications fully documented as static -> STRICT keeps its name
        raw["source_evidence"]["REGISTRIES"] = "STATIC_ATTRIBUTE"

    cfg = _config_variant(extract["tmp"], defs, everything_static)
    _co2, g2 = _graph(extract["read"].frame, defs, config=cfg)
    assert g2.feature_sets["PROVEN_SAFE"]["STRICT_SAFE"]["name"] == "STRICT_SAFE"
    cells = build_plan(g2, cfg, defs["mapping"], defs["mapping"].feature_sets())
    by = {c.cell: c for c in cells}
    assert by["STRICT_SAFE_D00_CLEAN"].source == "alias" and by["STRICT_SAFE_D00_CLEAN"].alias_of == REF_STRICT   # reused, never retrained


def test_plan_dedupes_reuses_references_and_marks_the_future_cell(extract, defs) -> None:
    _co, g = _graph(extract["read"].frame, defs)
    cells = build_plan(g, defs["config"], defs["mapping"], defs["mapping"].feature_sets())
    by = {c.cell: c for c in cells}
    assert by[REF_EXTENDED].source == "reference:extended" and by[REF_STRICT].source == "reference:strict"
    assert by["SAFE_EXTENDED_FULL"].alias_of == "STRICT_SAFE_SUBSET_FULL"
    assert by["EXTENDED_SAFE_OR_UNRESOLVED_D00_CLEAN"].alias_of == "EXTENDED_NO_FALLS_NO_MOBILITY"
    assert by["STRICT_SAFE_OR_UNRESOLVED_D00_CLEAN"].alias_of == REF_STRICT
    assert [c.removed for c in cells if c.kind == "ablation"] == [["falls"], ["mobility_problems"], ["falls", "mobility_problems"]]
    assert by["FINAL_DWH_FIXED"].source == "not_run" and by["FINAL_DWH_FIXED"].label is None
    assert all(c.cohort == "D00_CLEAN" for c in cells if c.feature_set == "FULL_EXTENDED" and c.kind != "future")
    runs = [c for c in cells if c.source == "run"]
    assert len({c.signature for c in runs}) == len(runs)   # one run per distinct specification


def test_config_is_validated(extract, defs) -> None:
    with pytest.raises(ConfigError):
        _config_variant(extract["tmp"], defs, lambda raw: raw["source_evidence"].pop("DIAGNOSES"))
    with pytest.raises(ConfigError):
        _config_variant(extract["tmp"], defs, lambda raw: raw["analysis_plan"]["matrix"].append(
            {"cell": "X", "cohort": "FULL_LABELED", "feature_set": "FULL_EXTENDED"}))
    with pytest.raises(ConfigError):
        _config_variant(extract["tmp"], defs, lambda raw: raw["analysis_plan"]["evidence_standards"]["SAFE_OR_UNRESOLVED"].update(
            {"admit": ["SAFE", "UNSAFE"]}))


# ------------------------------------------------------------------ 3-6: builds (no training)
def _build(frame: pd.DataFrame, defs, features: list[str], *, forbidden=None, scope="built_predictors", policy="fail"):
    ad = MeuhedetWideDatasetAdapter(defs["mapping"], defs["contract"], defs["spec"], features=features, id_pepper="pepper-for-tests",
                                    timing_scope=scope, forbidden_columns=forbidden)
    return ad.apply(frame, index_date="2025-01-01", index_day_records=policy)


def test_safe_models_cannot_read_unsafe_fields(extract, defs) -> None:
    _co, g = _graph(extract["read"].frame, defs)
    for std, sets in g.feature_sets.items():
        forbidden = set(g.forbidden_columns[std]["FULL_LABELED"])
        assert {"Prior_Fall_Since_Study_Start_Ind", "Gait_Disorder_Since_Study_Start_Ind", "Last_Dx_Date", "Last_Fall_Date"} <= forbidden
        for f in sets["SAFE_EXTENDED"]["features"]:
            inputs = feature_input_columns(defs["mapping"].get(f))
            assert not (set(inputs["value"]) | set(inputs["validation"])) & forbidden, (std, f)
    with pytest.raises(LeakageError):
        _build(extract["read"].frame, defs, defs["mapping"].feature_sets()["extended"], forbidden=g.forbidden_columns["SAFE_OR_UNRESOLVED"]["FULL_LABELED"])


def test_changing_unsafe_source_values_cannot_change_the_safe_matrix(extract, defs) -> None:
    frame = extract["read"].frame
    _co, g = _graph(frame, defs)
    std = "SAFE_OR_UNRESOLVED"
    feats = g.feature_sets[std]["SAFE_EXTENDED"]["features"]
    forbidden = g.forbidden_columns[std]["FULL_LABELED"]
    base, _rep = _build(frame, defs, feats, forbidden=forbidden)
    mutated = frame.copy()
    rng = np.random.default_rng(3)
    touched = [c for c, e in defs["dictionary"].columns.items() if e["source"] in UNSAFE_SOURCES and c in mutated.columns]
    assert len(touched) >= 15
    for c in touched:   # every value of every column of the two unsafe sources changes (within the contract's allowed values)
        s = mutated[c]
        allowed = defs["contract"].get(c).allowed
        if pd.api.types.is_datetime64_any_dtype(s):
            mutated[c] = s + pd.to_timedelta(rng.integers(-400, 30, len(s)), unit="D")   # includes index-day and future records
        elif allowed:
            mutated[c] = pd.array(rng.choice([int(v) for v in allowed], len(s)), dtype=s.dtype)
        elif pd.api.types.is_numeric_dtype(s):
            mutated[c] = pd.array(rng.integers(0, 5, len(s)), dtype=s.dtype) if str(s.dtype) in ("Int64", "int64") else rng.random(len(s))
        else:
            mutated[c] = s.sample(frac=1.0, random_state=1).to_numpy()
    after, rep2 = _build(mutated, defs, feats, forbidden=forbidden)
    pd.testing.assert_frame_equal(base.drop(columns=["predictor_max_record_date"]), after.drop(columns=["predictor_max_record_date"]))
    assert rep2.n_rows_timing_violation == 0


def test_full_labeled_keeps_d00_rows_and_excludes_unsafe_features_instead(extract, defs) -> None:
    frame = extract["read"].frame
    co, g = _graph(frame, defs)
    feats = g.feature_sets["SAFE_OR_UNRESOLVED"]["SAFE_EXTENDED"]["features"]
    out, rep = _build(frame, defs, feats, forbidden=g.forbidden_columns["SAFE_OR_UNRESOLVED"]["FULL_LABELED"])
    assert len(out) == co.counts["FULL_LABELED"] and rep.n_rows_timing_violation == 0
    assert rep.cohort_guard_rows_retained == co.counts["d00_rows_in_full_labeled"] > 0
    assert "falls" not in out.columns and "mobility_problems" not in out.columns
    clean, _r = _build(frame, defs, feats, scope="cohort", policy="drop_rows")   # the reference logic drops exactly the D-00 rows
    assert len(clean) == co.counts["D00_CLEAN"] == len(out) - co.counts["d00_rows_in_full_labeled"]


# ------------------------------------------------------------------ analysis helpers
def test_risk_concentration_paired_and_plateau_helpers() -> None:
    rng = np.random.default_rng(5)
    p = rng.uniform(0, 1, 4000)
    y = (rng.uniform(0, 1, 4000) < 0.25 * p).astype(int)
    rc = A.risk_concentration(y, p, fractions=(0.01, 0.02, 0.05, 0.1, 0.2), min_cell=10, n_boot=100)
    t10 = rc[rc["top_pct"] == 10.0].iloc[0]
    m = A.top_mask(p, 0.1)
    assert t10["n_selected"] == 400 and t10["falls_in_group"] == int(y[m].sum()) and t10["pct_of_population"] == 10.0
    assert t10["capture_ci_low"] <= t10["pct_of_all_falls_captured"] <= t10["capture_ci_high"] and t10["lift"] > 1
    assert rc[rc["top_pct"] == 1.0]["pct_of_all_falls_captured"].isna().iloc[0] or int(y[A.top_mask(p, 0.01)].sum()) >= 10
    same = A.paired_comparison(y, p, p, n_boot=50)
    assert same["auroc"]["estimate"] == 0 and same["capture_top10"]["estimate"] == 0
    noise = rng.uniform(0, 1, 4000)
    d = A.paired_comparison(y, p, noise, n_boot=100)
    assert d["auroc"]["ci_high"] < 0
    sg = A.share_of_gain(y, noise, p, (p + noise) / 2, n_boot=100)
    assert sg["defined"] and 0 < sg["share"] < 1
    flat = A.share_of_gain(y, p, p, noise, n_boot=50)
    assert not flat["defined"] and flat["share"] is None
    inc = [{"from_fraction": 0.6, "to_fraction": 0.8, "delta_validation_auroc": 0.004, "ci_low": -0.01, "ci_high": 0.02},
           {"from_fraction": 0.8, "to_fraction": 1.0, "delta_validation_auroc": 0.002, "ci_low": -0.008, "ci_high": 0.012}]
    st = A.plateau_statement(inc, max_gain=0.01, name="X")
    assert st["plateau_suggested"] and "approaching a plateau" in st["statement"] and "cannot establish" in st["statement"]
    assert "will not help" not in st["statement"].replace("would not help", "")
    rising = A.plateau_statement([{**inc[0]}, {**inc[1], "delta_validation_auroc": 0.03, "ci_low": 0.01, "ci_high": 0.05}], max_gain=0.01, name="X")
    assert rising["plateau_suggested"] is False and "no plateau is claimed" in rising["statement"]
    two = A.two_proportions(50, 500, 100, 5000)
    assert two["differs"] and two["difference"] == pytest.approx(0.08)


def test_python_311_syntax_of_every_module() -> None:
    bad = []
    for p in sorted((ROOT / "src" / "falls_ml").rglob("*.py")):
        try:
            ast.parse(p.read_text(encoding="utf-8"), filename=str(p), feature_version=(3, 11))
        except SyntaxError as exc:
            bad.append(f"{p.name}: {exc}")
    assert not bad, bad


# ------------------------------------------------------------------ end to end (slow)
def _small_template(tmp: Path) -> Path:
    tpl = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
    tpl["evaluation"]["bootstrap"]["n"] = 30
    tpl["evaluation"]["permutation_importance_repeats"] = 2
    tpl["analysis"]["stability"]["n_bootstrap"] = 5
    tpl["analysis"]["optimism"]["n_bootstrap"] = 2
    p = tmp / "template.yaml"
    p.write_text(yaml.safe_dump(tpl, sort_keys=False), encoding="utf-8")
    return p


@pytest.fixture(scope="module")
def d00_run(tmp_path_factory):
    from falls_ml.d00.runner import run_d00
    from falls_ml.meuhedet_explore import explore

    tmp = tmp_path_factory.mktemp("d00e2e")
    csv = write_synthetic_wide_extract(tmp / "wide.csv", n_rows=3000, seed=11, other_index_date_share=0.0, csv_date_format="%d/%m/%Y", n_index_day_falls=8)
    plant(csv, n_rows=3000)
    ref = tmp / "explore"
    explore(csv, out_dir=ref, index_day_records="drop_rows", template_path=_small_template(tmp), model_report="off", fast=False)
    reports = tmp / "existing_reports"       # stands in for the completed 0.6.0 report folder (protected, only read)
    (reports / "tables").mkdir(parents=True)
    (reports / "MANAGEMENT_MODEL_REPORT_HE.html").write_text("<html>existing</html>", encoding="utf-8")
    before = {"ref": directory_digest(ref), "reports": directory_digest(reports)}
    res = run_d00(csv, ref, out_dir=tmp / "d00", reports_dir=reports, learning_curve="reduced", n_boot=100)
    return {"res": res, "ref": ref, "reports": reports, "before": before, "tmp": tmp, "csv": csv}


@pytest.mark.slow
def test_d00_clean_reproduces_the_reference_and_ablations_are_paired(d00_run) -> None:
    res = d00_run["res"]
    ref_sha = res.records[REF_EXTENDED]["test_rows_sha256"]
    for cell in ("EXTENDED_NO_FALLS", "EXTENDED_NO_MOBILITY", "EXTENDED_NO_FALLS_NO_MOBILITY", "STRICT_SAFE_SUBSET_D00_CLEAN"):
        r = res.records[cell]
        assert r["test_rows_sha256"] == ref_sha, cell
        assert r["checks"].get("rows_equal_reference") and r["checks"].get("predictor_values_equal_reference"), cell
    ab = pd.read_csv(res.share_dir / "tables" / "ablation_paired.csv", skiprows=1)
    assert list(ab["compared"]) == ["EXTENDED_NO_FALLS", "EXTENDED_NO_MOBILITY", "EXTENDED_NO_FALLS_NO_MOBILITY"] and ab["paired"].all()
    for col in ("delta_auroc", "delta_auroc_ci_low", "delta_pr_auc", "delta_brier", "reference_calibration_slope", "compared_citl", "compared_oe_ratio",
                "reference_top5_capture_pct", "compared_top10_capture_pct", "reference_top20_capture_pct", "compared_top10_lift"):
        assert col in ab.columns, col
    full = [res.records[c] for c in ("STRICT_SAFE_SUBSET_FULL", "STRICT_SAFE_OR_UNRESOLVED_FULL", "EXTENDED_SAFE_OR_UNRESOLVED_FULL")]
    assert len({r["test_rows_sha256"] for r in full}) == 1 and full[0]["test_rows_sha256"] != ref_sha
    assert all(r["checks"]["rows_dropped_for_d00"] == 0 and r["checks"]["d00_rows_kept"] > 0 for r in full)
    assert res.records["SAFE_EXTENDED_FULL"]["status"].startswith("SAME RUN AS")
    assert res.records["FULL_EXTENDED_D00_CLEAN"]["status"].startswith("REUSED")
    assert res.records["FINAL_DWH_FIXED"]["status"].startswith("NOT RUN") and res.records["FINAL_DWH_FIXED"].get("auroc") is None


@pytest.mark.slow
def test_paired_and_unpaired_comparisons_are_labelled(d00_run) -> None:
    res = d00_run["res"]
    comp = res.summary["comparisons"]
    assert comp["population_effects"] and all(r["comparison_type"].startswith("POPULATION EFFECT - NOT A DIRECT PAIRED") for r in comp["population_effects"])
    for r in comp["feature_set_effects"] + comp["ablations"]:
        a, b = res.records[r["reference"]], res.records[r["compared"]]
        assert r["paired"] == (a["test_rows_sha256"] == b["test_rows_sha256"]), r
    html = (res.share_dir / "D00_SENSITIVITY_REPORT.html").read_text(encoding="utf-8")
    assert "NOT A DIRECT PAIRED MODEL COMPARISON" in html and "PAIRED" in html
    comp_mgmt = pd.read_csv(res.share_dir / "management" / "tables" / "model_comparison.csv").set_index("analysis")
    assert comp_mgmt.at["EXTENDED_SAFE_OR_UNRESOLVED_FULL", "comparison_note"].startswith("NOT A DIRECT PAIRED")
    assert comp_mgmt.at["EXTENDED_NO_FALLS", "comparison_note"].startswith("paired")


@pytest.mark.slow
def test_outputs_are_aggregate_labelled_and_the_protected_folders_unchanged(d00_run) -> None:
    res, share = d00_run["res"], d00_run["res"].share_dir
    assert res.privacy["passed"] and res.privacy["identifier_values_checked"] > 1000
    ids = set(pd.read_csv(d00_run["ref"] / "runs" / res.records[REF_EXTENDED]["run_id"] / "splits.csv")["research_id"].astype(str))
    pepper = (d00_run["tmp"] / "d00" / "id_pepper.txt").read_text(encoding="utf-8").strip()
    for p in share.rglob("*"):
        if p.suffix in {".md", ".csv", ".json", ".html", ".txt"}:
            text = re.sub(r"data:image/png;base64,[A-Za-z0-9+/=]+", "", p.read_text(encoding="utf-8", errors="ignore"))
            assert not ID_RE.search(text) and pepper not in text, p
            assert not (set(re.findall(r"[0-9a-f]{20}", text)) & ids), p
    for name in ("D00_SENSITIVITY_REPORT.html", "D00_SENSITIVITY_SUMMARY_HE.md", "D00_FEATURE_DEPENDENCY.md", "LASSO_WARNINGS_AUDIT.md", "README_D00.md"):
        text = (share / name).read_text(encoding="utf-8")
        assert "EXPLORATORY 180-DAY OUTCOME – NOT EFALLS REPRODUCTION" in text, name
    assert "not ready for clinical deployment" in (share / "D00_SENSITIVITY_REPORT.html").read_text(encoding="utf-8")
    assert (share / "tables" / "risk_concentration.csv").read_text(encoding="utf-8").startswith("EXPLORATORY 180-DAY OUTCOME")
    assert not (share / "datasets").exists() and not list(share.rglob("predictions_*"))
    assert directory_digest(d00_run["ref"]) == d00_run["before"]["ref"] and directory_digest(d00_run["reports"]) == d00_run["before"]["reports"]
    assert res.integrity["all_unchanged"]
    plan = json.loads((share / "ANALYSIS_PLAN.json").read_text(encoding="utf-8"))
    assert plan["plan_sha256"] == res.plan["plan_sha256"] and "auroc" not in json.dumps(plan["cells"])
    json.loads((share / "d00_sensitivity.json").read_text(encoding="utf-8"), parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))


@pytest.mark.slow
def test_reports_answer_the_questions_and_management_shows_importance_and_d00(d00_run) -> None:
    from falls_ml.modelreport.management import BANNED

    share = d00_run["res"].share_dir
    he = (share / "D00_SENSITIVITY_SUMMARY_HE.md").read_text(encoding="utf-8")
    for k in range(1, 13):
        assert f"## {k}." in he, k
    risk = pd.read_csv(share / "tables" / "risk_concentration.csv", skiprows=1)
    assert set(risk["top_pct"]) == {1.0, 2.0, 5.0, 10.0, 20.0}
    lc = pd.read_csv(share / "tables" / "learning_curve_EXTENDED.csv", skiprows=1)
    assert {"cv_minimum_identified", "warnings", "train_auroc_apparent", "validation_brier"} <= set(lc.columns)
    inc = pd.read_csv(share / "tables" / "learning_curve_increments_EXTENDED.csv", skiprows=1)
    assert [(a, b) for a, b in zip(inc["from_fraction"], inc["to_fraction"])] == [(0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    audit = json.loads((share / "LASSO_WARNINGS_AUDIT.json").read_text(encoding="utf-8"))
    assert set(audit["final_fits"]) == {"EXTENDED", "STRICT"} and "validation_auroc_at_lambda_star" in audit["lambda_sensitivity"]["EXTENDED"]
    assert audit["learning_curve"]["EXTENDED"]["available"] and audit["materiality"]["EXTENDED"]
    mg = share / "management"
    pngs = {p.stem for p in (mg / "figures").glob("*.png")}
    assert {"he_features", "he_permutation", "he_d00_matrix"} <= pngs
    page = (mg / "MANAGEMENT_MODEL_REPORT_HE.html").read_text(encoding="utf-8")
    assert "תרומה לחיזוי המודל" in page and "יציבות הבחירה" in page and "טרם הורץ" in (mg / "MANAGEMENT_MODEL_SUMMARY_HE.md").read_text(encoding="utf-8") + page
    assert "he_permutation" in (mg / "FIGURE_INDEX.md").read_text(encoding="utf-8")
    one = (mg / "EXECUTIVE_ONE_PAGER_HE.html").read_text(encoding="utf-8")
    assert "תרומה הגדולה ביותר לחיזוי המודל" in one
    for p in (mg / "MANAGEMENT_MODEL_REPORT_HE.html", mg / "EXECUTIVE_ONE_PAGER_HE.html", share / "D00_SENSITIVITY_SUMMARY_HE.md"):
        low = re.sub(r"data:image/png;base64,[A-Za-z0-9+/=]+", "", p.read_text(encoding="utf-8")).lower()
        assert not [w for w in BANNED if w in low], p


@pytest.mark.slow
def test_cli_refuses_protected_output_folders_and_supports_dependency_only(d00_run, tmp_path) -> None:
    ref, csv = d00_run["ref"], d00_run["csv"]
    assert cli_main(["meuhedet-d00", "--input", str(csv), "--reference", str(ref), "--out", str(ref / "inside")]) == 2
    assert cli_main(["meuhedet-d00", "--input", str(csv), "--reference", str(ref), "--out", str(d00_run["tmp"] / "d00")]) == 2   # not empty
    assert cli_main(["meuhedet-d00", "--input", str(csv), "--reference", str(ref), "--out", str(tmp_path / "dep"), "--dependency-only",
                     "--id-pepper-file", str(ref / "id_pepper.txt")]) == 0
    assert (tmp_path / "dep" / "share" / "D00_FEATURE_DEPENDENCY.csv").exists() and not (tmp_path / "dep" / "runs").exists()
    other = tmp_path / "other.csv"
    shutil.copy(csv, other)
    with other.open("a", encoding="utf-8") as fh:
        fh.write("\n")
    assert cli_main(["meuhedet-d00", "--input", str(other), "--reference", str(ref), "--out", str(tmp_path / "x"), "--dependency-only",
                     "--id-pepper-file", str(ref / "id_pepper.txt")]) == 2   # not the reference's file (sha256)


# ------------------------------------------------------------------ resume after an interrupted run (v0.7.1, slow)
TARGET = "EXTENDED_SAFE_OR_UNRESOLVED_FULL"


def _inject(mp: pytest.MonkeyPatch, *, policy: bool, calls: list[str]) -> None:
    """Stability replicate 0 of the TARGET cell does not converge at the configured iteration limit (the real fit runs, then a
    LassoConvergenceError carrying its real lambda* is raised) but converges when the full-data limit is raised - the situation of the
    interrupted real run, reproduced on synthetic data. policy=False reproduces the 0.7.0 behaviour (no retry: the replicate fails)."""
    import falls_ml.experiment as E
    from falls_ml.d00.plan import run_label
    from falls_ml.evaluation import convergence_retry as CR
    from falls_ml.models.lasso_cv import LassoConvergenceError, LassoLogisticCV
    from falls_ml.seeding import int_seed_for

    seed = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))["validation"]["seed"]
    hard = {int_seed_for(int_seed_for(seed, "stability"), "stability:fit:0")}
    state = {"active": False}
    real_run, real_fit = E.run_experiment, LassoLogisticCV.fit

    def run(model, ds, cfg_path, **kw):
        calls.append(Path(cfg_path).stem)
        state["active"] = Path(cfg_path).stem == run_label(TARGET)
        try:
            return real_run(model, ds, cfg_path, **kw)
        finally:
            state["active"] = False

    def fit(self, X, y, *, groups=None):
        out = real_fit(self, X, y, groups=groups)
        if state["active"] and self.params.get("full_path_max_iter") is None and self.random_state in hard:
            d = self.diagnostics_
            raise LassoConvergenceError("injected (synthetic test): full-data solve at the selected lambda did not converge",
                                        lambda_star=d["lambda_star"], lambda_star_index=d["lambda_star_index"], tol=d["tol"], max_iter=d["full_path_max_iter"])
        return out

    mp.setattr(E, "run_experiment", run)
    mp.setattr(LassoLogisticCV, "fit", fit)
    if not policy:
        mp.setattr(CR, "fit_replicate", lambda fit_fn, sample, fit_seed, **kw: fit_fn(sample, fit_seed))


def _files(root: Path) -> dict[str, tuple[str, int]]:
    return {p.relative_to(root).as_posix(): (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
            for part in ("runs", "datasets", "configs") for p in sorted((root / part).rglob("*")) if p.is_file()}


@pytest.fixture(scope="module")
def d00_resume(d00_run):
    from falls_ml.d00.runner import run_d00
    from falls_ml.evaluation.stability import ResamplingError

    tmp, csv, ref, reports = d00_run["tmp"], d00_run["csv"], d00_run["ref"], d00_run["reports"]
    before_ref = {"ref": directory_digest(ref), "reports": directory_digest(reports)}
    calls: dict[str, list[str]] = {k: [] for k in ("straight", "first", "resume", "again")}
    with pytest.MonkeyPatch.context() as mp:        # the same analysis in one uninterrupted attempt (retry policy on)
        _inject(mp, policy=True, calls=calls["straight"])
        straight = run_d00(csv, ref, out_dir=tmp / "d00_straight", reports_dir=reports, learning_curve="off", n_boot=100)
    out = tmp / "d00_resume"
    with pytest.MonkeyPatch.context() as mp:        # attempt 1 without the policy: the TARGET cell's stability aborts (1 of 5 > 10%)
        _inject(mp, policy=False, calls=calls["first"])
        with pytest.raises(ResamplingError, match="bootstrap replicates failed"):
            run_d00(csv, ref, out_dir=out, reports_dir=reports, learning_curve="off", n_boot=100)
    registry_before = (out / "runs" / "test_evaluation_registry.jsonl").read_text(encoding="utf-8").splitlines()
    files_before = _files(out)
    with pytest.MonkeyPatch.context() as mp:
        _inject(mp, policy=True, calls=calls["resume"])
        resumed = run_d00(csv, ref, out_dir=out, reports_dir=reports, n_boot=100, resume=True)
    files_after_resume = _files(out)
    with pytest.MonkeyPatch.context() as mp:        # resuming a finished analysis refits nothing and reproduces every number
        _inject(mp, policy=True, calls=calls["again"])
        again = run_d00(csv, ref, out_dir=out, reports_dir=reports, n_boot=100, resume=True)
    return {"straight": straight, "resumed": resumed, "again": again, "out": out, "calls": calls, "registry_before": registry_before,
            "files_before": files_before, "files_after_resume": files_after_resume, "before_ref": before_ref, **d00_run}


@pytest.mark.slow
def test_resume_reuses_completed_cells_byte_for_byte_and_refits_only_the_failed_cell(d00_resume) -> None:
    from falls_ml.d00.plan import run_label
    from falls_ml.d00.runner import STATUS_NEW, STATUS_REUSED_EARLIER

    r, out, calls = d00_resume["resumed"], d00_resume["out"], d00_resume["calls"]
    assert calls["first"][-1] == run_label(TARGET) and calls["resume"] == [run_label(TARGET)] and calls["again"] == []
    reused = [c for c in r.records if r.records[c]["status"] == STATUS_REUSED_EARLIER]
    assert sorted(run_label(c) for c in reused) == sorted(calls["first"][:-1])
    assert {"EXTENDED_NO_FALLS", "EXTENDED_NO_MOBILITY", "EXTENDED_NO_FALLS_NO_MOBILITY"} <= set(reused)
    assert r.records[TARGET]["status"] == STATUS_NEW
    # every file of the earlier attempt (completed runs incl. their test predictions, the incomplete run, datasets, configs) untouched
    before, after = d00_resume["files_before"], d00_resume["files_after_resume"]
    assert all(after.get(k) == v for k, v in before.items() if not k.endswith("test_evaluation_registry.jsonl"))
    for c in reused:
        run_dir = out / "runs" / r.records[c]["run_id"]
        assert (run_dir / "RUN_COMPLETE.json").exists() and before[f"runs/{run_dir.name}/predictions_test.parquet"] == after[f"runs/{run_dir.name}/predictions_test.parquet"]
    # the test set of a reused cell was released once, in the earlier attempt; the resume released only the refitted cell's
    reg_after = (out / "runs" / "test_evaluation_registry.jsonl").read_text(encoding="utf-8").splitlines()
    assert reg_after[:len(d00_resume["registry_before"])] == d00_resume["registry_before"] and len(reg_after) == len(d00_resume["registry_before"]) + 1
    assert json.loads(reg_after[-1])["run_id"] == r.records[TARGET]["run_id"]
    assert {json.loads(x)["run_id"] for x in d00_resume["registry_before"]} == {r.records[c]["run_id"] for c in reused}
    assert r.integrity["all_unchanged"] and r.integrity["earlier_attempt"]["unchanged"] and r.integrity["earlier_attempt"]["registry_entries_added"] == 1
    assert directory_digest(d00_resume["ref"]) == d00_resume["before_ref"]["ref"] and directory_digest(d00_resume["reports"]) == d00_resume["before_ref"]["reports"]


@pytest.mark.slow
def test_the_incomplete_run_is_never_read_as_a_result_and_the_retry_is_recorded(d00_resume) -> None:
    r, out = d00_resume["resumed"], d00_resume["out"]
    full_log = json.loads((r.share_dir / "RESUME_LOG.json").read_text(encoding="utf-8"))
    assert full_log["n_resume_attempts"] == 2 and full_log["latest"] == full_log["attempts"][1]     # the second resume refitted nothing
    assert all(c["action"] == "REUSED_COMPLETED_RUN" for c in full_log["attempts"][1]["cells"]) and full_log["attempts"][1]["completed"]
    log = full_log["attempts"][0]
    assert log["completed"] and log["earlier_attempt_files_unchanged"] and log["earlier_attempt_integrity"]["registry_entries_added"] == 1
    (inc,) = log["incomplete_attempts"]
    inc_dir = out / "runs" / inc["run_id"]
    assert inc["cell"] == TARGET and inc["test_set_released"] is False and "bootstrap replicates failed" in inc["failure"]["error"]
    assert inc_dir.is_dir() and not (inc_dir / "RUN_COMPLETE.json").exists() and inc["run_id"] != r.records[TARGET]["run_id"]
    assert inc["run_id"] not in json.dumps(r.summary, default=str) and inc["run_id"] not in (r.share_dir / "management" / "tables" / "model_comparison.csv").read_text(encoding="utf-8")
    actions = {c["cell"]: c["action"] for c in log["cells"]}
    assert actions[TARGET].startswith("REFITTED") and sum(a == "REUSED_COMPLETED_RUN" for a in actions.values()) == len(actions) - 1
    run_dir = out / "runs" / r.records[TARGET]["run_id"]
    retries = pd.read_csv(run_dir / "bootstrap_convergence_retries.csv")
    assert len(retries) == 1 and retries.at[0, "component"] == "stability" and retries.at[0, "replicate"] == 0
    assert retries.at[0, "outcome"] == "CONVERGED_ON_RETRY" and bool(retries.at[0, "retry_converged_at_lambda_star"])
    assert retries.at[0, "retry_max_iter"] == 10 * retries.at[0, "initial_max_iter"] and retries.at[0, "retry_lambda_star_index"] == retries.at[0, "lambda_star_index"]
    m = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert m["bootstrap_convergence_retry"]["n_converged_on_retry"] == 1 and m["bootstrap_convergence_retry"]["n_still_failed"] == 0
    audit = json.loads((r.share_dir / "LASSO_WARNINGS_AUDIT.json").read_text(encoding="utf-8"))
    assert audit["new_run_replicates"][TARGET]["convergence_retries"] == {"stability": {"CONVERGED_ON_RETRY": 1}}
    assert set(audit["new_runs"]) == {c for c in r.records if r.records[c]["status"].startswith(("RUN (new)", "REUSED (completed in an earlier"))}
    for name in ("README_D00.md", "D00_SENSITIVITY_REPORT.html", "D00_SENSITIVITY_SUMMARY_HE.md"):
        text = (r.share_dir / name).read_text(encoding="utf-8")
        assert ("RESUMED" in text or "Resumed" in text or "הושלם בהמשך להרצה שנקטעה" in text), name
    assert r.privacy["passed"] and "RESUME_LOG.json" in (r.share_dir / "README_D00.md").read_text(encoding="utf-8")
    assert "cells refitted in them: " + TARGET in (r.share_dir / "README_D00.md").read_text(encoding="utf-8")   # after the second resume


@pytest.mark.slow
def test_resumed_results_equal_an_uninterrupted_run_and_a_second_resume_changes_nothing(d00_resume) -> None:
    s, r, a = d00_resume["straight"], d00_resume["resumed"], d00_resume["again"]
    keys = ("auroc", "auroc_ci", "pr_auc", "brier", "calibration_slope", "citl", "oe_ratio", "n_rows", "n_events", "test_rows_sha256", "lambda_star",
            "selected_raw_features", "served_variant")
    for c in s.records:
        assert {k: s.records[c].get(k) for k in keys} == {k: r.records[c].get(k) for k in keys}, c
        assert {k: v for k, v in r.records[c].items() if k != "status"} == {k: v for k, v in a.records[c].items() if k != "status"}, c
    from falls_ml.d00.runner import STATUS_REUSED_EARLIER
    assert {c: x["status"] for c, x in a.records.items() if x["status"].startswith(("RUN", "REUSED (completed in an earlier"))} == {
        c: STATUS_REUSED_EARLIER for c, x in r.records.items() if x["status"].startswith(("RUN", "REUSED (completed in an earlier"))}
    for name in ("ablation_paired", "feature_set_effects_paired", "population_effects_unpaired", "risk_concentration"):
        t1 = (s.share_dir / "tables" / f"{name}.csv").read_text(encoding="utf-8")
        t2 = (r.share_dir / "tables" / f"{name}.csv").read_text(encoding="utf-8")
        assert t1 == t2, name
    assert r.plan["plan_sha256"] == s.plan["plan_sha256"] == a.plan["plan_sha256"]
    assert a.integrity["earlier_attempt"]["unchanged"] and a.integrity["earlier_attempt"]["registry_entries_added"] == 0


@pytest.mark.slow
def test_resume_refusals(d00_resume, tmp_path) -> None:
    from falls_ml.d00.runner import run_d00

    csv, ref, reports, out = d00_resume["csv"], d00_resume["ref"], d00_resume["reports"], d00_resume["out"]
    with pytest.raises(DatasetValidationError, match="nothing to resume"):
        run_d00(csv, ref, out_dir=tmp_path / "empty", resume=True)
    with pytest.raises(DatasetValidationError, match="differs from the frozen plan"):
        run_d00(csv, ref, out_dir=out, reports_dir=reports, resampling="reduced", resume=True)
    with pytest.raises(DatasetValidationError, match="--resume"):
        run_d00(csv, ref, out_dir=out, reports_dir=reports)                      # not empty: the message points to --resume
    assert cli_main(["meuhedet-d00", "--input", str(csv), "--reference", str(ref), "--out", str(out), "--resume", "--dependency-only"]) == 2
    # a completed run's test set was released; if it is no longer complete, refitting it would re-evaluate the test rows: refused
    released = tmp_path / "released"
    shutil.copytree(out, released)
    target_run = released / "runs" / d00_resume["resumed"].records[TARGET]["run_id"]
    (target_run / "RUN_COMPLETE.json").unlink()
    before = _files(released)
    with pytest.raises(LeakageError, match="already released the test set"):
        run_d00(csv, ref, out_dir=released, reports_dir=reports, resume=True)
    assert _files(released) == before
    # a regenerated config that differs from the earlier attempt's is never silently replaced
    tampered = tmp_path / "tampered"
    shutil.copytree(out, tampered)
    cfg = sorted((tampered / "configs").glob("*.yaml"))[0]
    cfg.write_text(cfg.read_text(encoding="utf-8") + "# edited\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="regenerated experiment config differs"):
        run_d00(csv, ref, out_dir=tampered, reports_dir=reports, resume=True)
