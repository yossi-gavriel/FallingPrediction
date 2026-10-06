"""Phase 5 contract tests (fast, synthetic data only).

Brief section 22 acceptance proofs covered here (the rest in tests/unit/test_phase5_e2e.py):
 1  the exact V1 -> V21 schema diff works                         test_schema_diff_is_exact_and_programmatic / ..._recomputed_from_the_real_header
 2  a known new feature is classified NEW                         test_known_new_features_are_new_and_mefi_stays_old
 3  the old MEFI feature remains OLD                              test_known_new_features_are_new_and_mefi_stays_old
 4  an unknown-but-valid new clinical feature is not dropped      test_unknown_clinical_column_stops_the_preflight
 5  an identifier is excluded                                     test_identifiers_and_outcome_fields_never_enter_x
 6  an outcome / future field is excluded                         test_identifiers_and_outcome_fields_never_enter_x
 7  post-index leakage hard-stops                                 test_post_index_or_ineligible_feature_in_x_hard_stops (+ post-index records / leaky)
 9-10 inner selection / threshold never see the outer labels      test_unit_choices_do_not_depend_on_outer_labels (LASSO, ENET, XGB)
 13 USEFUL needs an improvement in every outer fold               test_decision_rule_enet_primary_and_every_fold
 15 the privacy scan blocks patient-level outputs                 test_share_publication_fails_closed_on_an_identifier
 16 Earlier source frozen except reviewed SQLite lifecycle fix   test_phase2_3_4_source_is_frozen_except_reviewed_lifecycle_patch
 +  the first real-data action is the preflight alone             test_real_data_modelling_requires_a_preflight_first
 +  a rename is never inferred from position / name               test_registry_lineage_is_proven_never_inferred
 +  any positive after Followup_End_Date stops (zero tolerance)   test_positive_after_followup_end_stops_with_zero_tolerance
"""

from __future__ import annotations

import hashlib
import inspect
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]


# ============================================================================ fixtures
def _prepare(d: Path, traps: tuple[str, ...] = (), n: int = 2500, scenario: str = "planted") -> dict[str, Any]:
    from falls_ml.phase3.runner import load_all
    from falls_ml.phase4 import sealed
    from falls_ml.phase4.common import input_identity
    from falls_ml.phase5.config import load_phase5_config
    from falls_ml.phase5.data import prepare
    from falls_ml.phase5.synthetic import make_v21, write_v21_csv

    df, facts = make_v21(n, scenario=scenario, seed=31, traps=traps)
    csv = write_v21_csv(df, d / "v21.csv")
    cfg = load_phase5_config(mode="quick", overrides={"eligibility": {"min_known_observed_rows": 20}})
    L = load_all(cfg["phase3_config"])
    sealed.READ_LOG.clear()
    P = prepare(csv, cfg, L, input_info=input_identity(csv))
    return {"P": P, "cfg": cfg, "L": L, "csv": csv, "df": df, "facts": facts, "reads": list(sealed.READ_LOG), "dir": d}


@pytest.fixture(scope="module")
def prepared(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    from falls_ml.phase5.runner import build_sets

    r = _prepare(tmp_path_factory.mktemp("p5c"), traps=("future_column", "leaky_new"))
    r["S"] = build_sets(r["P"].registry, r["P"].meta, r["cfg"])
    return r


def _small_ctx(prepared: dict[str, Any], out: Path, y: np.ndarray | None = None, overrides: dict[str, Any] | None = None) -> Any:
    from falls_ml.phase5.config import load_phase5_config
    from falls_ml.phase5.engine import Ctx, outer_folds
    from falls_ml.phase5.models import DeviceState

    P, S = prepared["P"], prepared["S"]
    ov = {"eligibility": {"min_known_observed_rows": 20}, "cv": {"outer_folds": 3, "inner_folds": 3},
          "modes": {"quick": {"lasso": {"n_lambda": 6}, "enet": {"n_lambda": 4, "l1_ratios": [0.5]}, "xgb": {"n_trials": 3, "n_startup_trials": 2}}}}
    cfg = load_phase5_config(mode="quick", overrides={**ov, **(overrides or {})})
    order = np.argsort(P.frame["row_key"].to_numpy(), kind="mergesort")
    frame = P.frame.iloc[order].reset_index(drop=True)
    yy = P.y[order].astype(int) if y is None else y
    sets = {k: v for k, v in S["sets"].items() if k in ("OLD", "OLD_PLUS_ALL_NEW_ELIGIBLE")}
    folds = outer_folds(P.y[order].astype(int), cfg.outer_folds, 7)
    return Ctx(out=out, frame=frame, y=yy, meta=P.meta, sets=sets, outer=folds, cfg=cfg, seed=99, jobs=2, state=DeviceState("cpu"))


def _patterns() -> list[str]:
    import yaml

    return yaml.safe_load((ROOT / "configs/meuhedet/phase5.yaml").read_text(encoding="utf-8"))["phase5"]["x_sealing"]["name_patterns"]


# ============================================================================ 1-3 the exact V1 -> V21 schema diff
def test_schema_diff_is_exact_and_programmatic() -> None:
    from falls_ml.phase3.runner import load_all
    from falls_ml.phase5.data import universe_inputs
    from falls_ml.phase5.schema import CLASSES, load_v21_schema, parse_definition, schema_diff

    s = load_v21_schema("configs/meuhedet/phase5_v21_schema.yaml")
    hdr, meaning = parse_definition((ROOT / "configs/meuhedet/phase5_v21_view_definition.txt").read_text(encoding="utf-8"))
    assert list(s.header) == hdr == list(meaning) and len(hdr) == 224 and len(set(hdr)) == 224
    L = load_all("configs/meuhedet/phase3.yaml")
    v1 = list(L["contract"].names)
    d = schema_diff(hdr, s, L["contract"], L["dictionary"], sealed_patterns=_patterns(), feature_inputs=universe_inputs(L))
    # the name-level diff is plain set arithmetic on the V1 contract and the V21 header
    assert d.counts["v1_columns"] == len(v1) == 221
    assert sorted(d.removed["column"]) == sorted(c for c in v1 if c not in hdr) and len(d.removed) == 20
    assert d.counts["new_columns"] == len([c for c in hdr if c not in v1]) == 23
    assert d.counts["shared_names"] == len([c for c in hdr if c in v1]) == 201
    assert set(d.classes) == set(hdr) and set(d.classes.values()) <= set(CLASSES) and not d.unresolved and d.header_matches
    assert len(d.table) == len(set(v1) | set(hdr)) and len(d.classification) == 224
    rc = d.renamed_changed.set_index("v21_column")
    for v21c, v1c in (("Registry_Corona_Ind", "Registry_Blood_Pressure_Ind"), ("Registry_Dialysis_Ind", "Registry_Chronic_Renal_Failure_Ind"),
                      ("Registry_Immunosuppressant_Ind", "Registry_Transplant_Ind")):
        assert rc.loc[v21c, "v1_column"] == v1c and rc.loc[v21c, "lineage_class"] == "OLD_REMOVED_NEW_ADDED" and not rc.loc[v21c, "lineage_proven"]
        assert rc.loc[v21c, "change"] == "NOT_A_RENAME (OLD_REMOVED_NEW_ADDED)" and d.classes[v21c] == "NEW_CANDIDATE_PREDICTOR"
    assert d.counts["renamed_or_replaced"] == 0 and "RENAMED_OR_REPLACED" not in set(d.classes.values())
    assert d.counts["lineage_by_class"] == {"TRUE_RENAME_SAME_SEMANTICS": 0, "CORRECTED_LABEL_SAME_SOURCE": 0, "OLD_REMOVED_NEW_ADDED": 3,
                                            "MATERIAL_DEFINITION_CHANGE": 0}
    assert set(rc.index[rc["change"] == "OLD_CHANGED_DEFINITION"]) == {"Last_Hosp_Length", "Fall_Self_Report_Value"}
    rm = d.removed.set_index("column")
    assert "hypertension" in rm.loc["Registry_Blood_Pressure_Ind", "phase3_features_affected"]
    assert bool(rm.loc["Prior_Fall_Missing_Ind", "bridged"])


def test_schema_diff_is_recomputed_from_the_real_header(tmp_path: Path) -> None:
    from falls_ml.errors import ConfigError
    from falls_ml.phase3.runner import load_all
    from falls_ml.phase5.data import universe_inputs
    from falls_ml.phase5.schema import REVIEW, load_v21_schema, schema_diff

    s = load_v21_schema("configs/meuhedet/phase5_v21_schema.yaml")
    L = load_all("configs/meuhedet/phase3.yaml")
    hdr = [c for c in s.header if c != "Tremor_Ind"] + ["Balance_Clinic_Referral_Ind", "Fall_Next_365D_Ind", "Registry_Blood_Pressure_Ind"]
    d = schema_diff(hdr, s, L["contract"], L["dictionary"], sealed_patterns=_patterns(), feature_inputs=universe_inputs(L))
    assert d.missing_from_extract == ["Tremor_Ind"] and not d.header_matches
    assert d.classes["Balance_Clinic_Referral_Ind"] == REVIEW                    # undefined clinical-looking column: never silently dropped
    assert d.classes["Fall_Next_365D_Ind"] == "OUTCOME_OR_FUTURE_FORBIDDEN"      # future-looking name: sealed
    assert d.classes["Registry_Blood_Pressure_Ind"] == REVIEW                    # a V1 predictor V21 does not define: review, not OLD
    assert d.classes["Registry_Corona_Ind"] == "NEW_CANDIDATE_PREDICTOR"         # never a rename (lineage not proven)
    assert set(d.unresolved) == {"Balance_Clinic_Referral_Ind", "Registry_Blood_Pressure_Ind"}
    # the reviewed schema must cover exactly the embedded definition file, whose content is pinned by sha256
    bad = tmp_path / "schema.yaml"
    bad.write_text((ROOT / "configs/meuhedet/phase5_v21_schema.yaml").read_text(encoding="utf-8").replace("    Tremor_Ind:", "    Tremor_Typo_Ind:"),
                   encoding="utf-8")
    with pytest.raises(ConfigError):
        load_v21_schema(bad)
    defn = tmp_path / "def.txt"
    shutil.copyfile(ROOT / "configs/meuhedet/phase5_v21_view_definition.txt", defn)
    defn.write_text(defn.read_text(encoding="utf-8") + "\n-- edited\n", encoding="utf-8")
    bad2 = tmp_path / "schema2.yaml"
    bad2.write_text((ROOT / "configs/meuhedet/phase5_v21_schema.yaml").read_text(encoding="utf-8").replace(
        "definition_file: configs/meuhedet/phase5_v21_view_definition.txt", f"definition_file: {defn.as_posix()}"), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_v21_schema(bad2)


def test_registry_lineage_is_proven_never_inferred(tmp_path: Path) -> None:
    from falls_ml.errors import ConfigError
    from falls_ml.phase3.runner import load_all
    from falls_ml.phase5.data import universe_inputs
    from falls_ml.phase5.schema import load_v21_schema, schema_diff

    s = load_v21_schema("configs/meuhedet/phase5_v21_schema.yaml")
    assert set(s.lineage) == {"Registry_Blood_Pressure_Ind", "Registry_Chronic_Renal_Failure_Ind", "Registry_Transplant_Ind"}
    for g in s.lineage.values():
        assert g.klass == "OLD_REMOVED_NEW_ADDED" and g.proven is False and not s.predictors[g.v21_column].replaces
        assert g.evidence["v1_sql_expression"].startswith("NOT AVAILABLE") and g.evidence["v1_registry_ids"] == "NOT DOCUMENTED"
        assert g.evidence["v21_registry_ids"] and g.evidence["comparison"] and g.evidence["conclusion"]
    L = load_all("configs/meuhedet/phase3.yaml")
    d = schema_diff(list(s.header), s, L["contract"], L["dictionary"], sealed_patterns=_patterns(), feature_inputs=universe_inputs(L))
    t = d.table.set_index("column")
    for v1c, g in s.lineage.items():                                            # documented in SCHEMA_DIFF_V1_V21.csv on BOTH columns of the pair
        for c in (v1c, g.v21_column):
            assert t.loc[c, "lineage_class"] == "OLD_REMOVED_NEW_ADDED" and t.loc[c, "lineage_proven"] is False
            assert "registry_ids: NOT DOCUMENTED" in t.loc[c, "lineage_v1_evidence"] and g.evidence["v21_registry_ids"] in t.loc[c, "lineage_v21_evidence"]
            assert t.loc[c, "lineage_comparison"] and t.loc[c, "lineage_conclusion"]
        assert t.loc[v1c, "successor_in_v21"] == "" and t.loc[g.v21_column, "predecessor_in_v1"] == ""
    rm = d.removed.set_index("column")
    assert rm.loc["Registry_Blood_Pressure_Ind", "lineage_class"] == "OLD_REMOVED_NEW_ADDED"
    assert "genuinely new V21 predictor" in rm.loc["Registry_Blood_Pressure_Ind", "consequence"]
    # a same-lineage class is refused without proof / the V1 SQL evidence, and so is a 'replaces' without a proven lineage
    src = (ROOT / "configs/meuhedet/phase5_v21_schema.yaml").read_text(encoding="utf-8")
    defn = (ROOT / "configs/meuhedet/phase5_v21_view_definition.txt").as_posix()
    src = src.replace("definition_file: configs/meuhedet/phase5_v21_view_definition.txt", f"definition_file: {defn}")
    one = "      v21_column: Registry_Corona_Ind\n      class: OLD_REMOVED_NEW_ADDED\n      lineage_proven: false\n"
    assert src.count(one) == 1
    for bad in (src.replace(one, one.replace("OLD_REMOVED_NEW_ADDED", "CORRECTED_LABEL_SAME_SOURCE")),
                src.replace(one, one.replace("OLD_REMOVED_NEW_ADDED", "TRUE_RENAME_SAME_SEMANTICS").replace("false", "true")),
                src.replace(one, one.replace("OLD_REMOVED_NEW_ADDED", "MATERIAL_DEFINITION_CHANGE").replace("false", "true")),
                src.replace("provenance: DEFENSIBLE,\n        definition: '1 = valid membership at the index day in registries 116 / 118",
                            "provenance: DEFENSIBLE, replaces: Registry_Blood_Pressure_Ind,\n        definition: '1 = valid membership at the index day in registries 116 / 118")):
        assert bad != src
        p = tmp_path / "schema.yaml"
        p.write_text(bad, encoding="utf-8")
        with pytest.raises(ConfigError):
            load_v21_schema(p)


def test_positive_after_followup_end_stops_with_zero_tolerance(tmp_path: Path) -> None:
    import json

    from falls_ml.errors import ConfigError
    from falls_ml.phase5.config import load_phase5_config
    from falls_ml.phase5.runner import STOP_LINE, run_phase5
    from falls_ml.phase5.synthetic import make_v21, write_v21_csv

    df, facts = make_v21(1500, scenario="null", seed=5, traps=("positive_after_followup",))
    assert 0 < facts["positives_after_followup"] <= 7                         # a handful (< 0.5% of the positives at real scale): still a STOP
    el = (pd.to_numeric(df["Is_Eligible_Cohort"], errors="coerce") == 1) & (pd.to_datetime(df["Index_Date"], errors="coerce").dt.normalize()
                                                                            == pd.Timestamp("2026-01-01"))
    pos_before = int((pd.to_numeric(df.loc[el, "Fall_Next_180D_Ind"], errors="coerce") == 1).sum())
    (tmp_path / "in").mkdir()
    src = write_v21_csv(df, tmp_path / "in" / "v21.csv")
    out = tmp_path / "out"
    r = run_phase5(src, out, mode="quick", preflight_only=True, overrides={"eligibility": {"min_known_observed_rows": 20}})
    assert r["status"] == "STOPPED_PREFLIGHT" and r["exit_code"] == 2
    pf = out / "preflight"
    md = (pf / "PHASE5_PREFLIGHT.md").read_text(encoding="utf-8")
    assert md.rstrip().endswith(STOP_LINE) and "zero tolerance" in md
    oc = json.loads((pf / "OUTCOME_CONTRACT_2026.json").read_text(encoding="utf-8"))
    o4 = oc["checks"]["O4_positive_after_followup_end"]
    assert o4["passed"] is False and o4["limit"] == 0 and o4["positives_event_after_followup_end"] == "<10"     # small cell, aggregate only
    assert str(o4["pct_of_positives"]).startswith("<") and "<10 positive(s)" in md
    assert o4["positives"] == pos_before                                       # labels untouched, nobody excluded (aggregate only)
    assert any(str(h).startswith("O4:") for h in oc["hard_failures"]) and not oc["passed"]
    assert not (out / "work" / "PLAN.json").exists()
    # any tolerance is refused by the configuration
    for v in (0.005, 0.0001):
        with pytest.raises(ConfigError):
            load_phase5_config(overrides={"outcome_contract": {"max_positive_after_followup_share": v}})


def test_known_new_features_are_new_and_mefi_stays_old(prepared: dict[str, Any]) -> None:
    P, S = prepared["P"], prepared["S"]
    cls = P.diff.classes
    for c in ("Dizziness_Ind", "Gait_Abnormality_Ind", "Syncope_Ind", "Tremor_Ind", "Cataract_Ind", "Hearing_Loss_Dx_Ind", "Vision_Impairment_Dx_Ind",
              "Osteoporosis_Ind", "Parkinsonism_Ind", "Stroke_Dx_Ind", "Registry_Smoking_Ind", "Registry_Obesity_Ind", "Registry_Oncology_Ind",
              "Registry_IBD_Ind", "Registry_Opiate_Ind", "Registry_Severe_Function_Ind", "Registry_Smoking_SubCode", "Registry_Obesity_SubCode",
              "Deficit_Count_Proxy", "Registry_Dialysis_Ind", "Registry_Corona_Ind", "Registry_Immunosuppressant_Ind"):
        assert cls[c] == "NEW_CANDIDATE_PREDICTOR", c                         # lineage of the last three to V1 registries NOT proven: genuinely new
    for c in ("MEFI_Group_At_Index", "MEFI_Assessed_Ind", "Days_In_Current_MEFI_Group", "MEFI_Worsened_Ind", "Frailty_Not_Assessed_Ind"):
        assert cls[c] == "OLD_UNCHANGED", c                                     # MEFI is NOT new
    assert "frail_mefi_group" in S["sets"]["OLD"] and not any(f.startswith("new_mefi") for f in P.meta)
    reg = P.registry.set_index("feature")
    assert "new_dizziness_ind" in S["sets"]["OLD_PLUS_ALL_NEW_ELIGIBLE"] and "new_dizziness_ind" in S["sets"]["OLD_PLUS_NEW_SAFE"]
    assert "new_dizziness_ind" not in S["sets"]["OLD"] and reg.loc["new_dizziness_ind", "class"] == "SAFE_VERIFIED"
    # OLD under V21: 'falls' kept (removed validation input bridged); the three removed V1 registry features are not reproducible
    assert "falls" in S["sets"]["OLD"]
    for f in ("hypertension", "chronic_kidney_disease", "com_registry_transplant"):
        assert reg.loc[f, "class"] == "INELIGIBLE_DATA" and f not in S["sets"]["OLD"]
    # every genuine new predictor is in a set or excluded, always with a reason
    nw = P.registry[P.registry["origin"] == "NEW_V21"]
    assert len(nw) == 22 and nw["set_reason"].str.len().gt(10).all()
    f = "new_registry_smoking_subcode"                                        # unvalidated raw code: ALL_NEW only
    assert f in S["sets"]["OLD_PLUS_ALL_NEW_ELIGIBLE"] and f not in S["sets"]["OLD_PLUS_NEW_SAFE"]
    for c in ("Registry_Corona_Ind", "Registry_Dialysis_Ind", "Registry_Immunosuppressant_Ind"):   # genuinely new, documented V21 registries
        f = "new_" + c.lower()
        assert reg.loc[f, "domain"] == "NEW_REGISTRY" and reg.loc[f, "new_provenance"] == "DEFENSIBLE", c
        assert f in S["sets"]["OLD_PLUS_ALL_NEW_ELIGIBLE"] and (f in S["sets"]["OLD_PLUS_NEW_SAFE"]) == (reg.loc[f, "class"] == "SAFE_ATTESTED"), c
    assert "Diagnosis_Source_Absent_Ind" not in " ".join(P.registry["raw_columns"]) and cls["Diagnosis_Source_Absent_Ind"] == "METADATA_OR_ADMIN"
    cat = P.catalogue.set_index("raw_column")
    assert set(cat.index) >= {c for c in P.diff.classes if c not in set(prepared["L"]["contract"].names)}
    assert (cat["inclusion_or_exclusion_reason"].astype(str).str.len() > 5).all()


# ============================================================================ 4-7 sealing, unknown columns, leakage
def test_identifiers_and_outcome_fields_never_enter_x(prepared: dict[str, Any]) -> None:
    P, S = prepared["P"], prepared["S"]
    sealed = P.sealed
    for c in ("Fall_Next_180D_Ind", "Next_Fall_Date_180D", "Days_To_Next_Fall_180D", "Next_Fall_Event_ID_180D", "Label_Reason_180D", "Followup_End_Date",
              "Is_Censored_180D", "Fall_Next_30D_Ind", "Fall_Next_180D_HighConf_Ind", "Fall_Next_365D_Ind", "Snapshot_Key", "External_Care_Count_365D",
              "Max_Invoice_Lag_365D", "Abroad_Ind"):
        assert c in sealed, c
    assert P.diff.classes["Customer_Full_ID"] == "IDENTIFIER" and P.diff.classes["Snapshot_Key"] == "IDENTIFIER"
    x_reads = [cols for name, cols in prepared["reads"] if "Customer_Full_ID" in cols and "Is_Eligible_Cohort" in cols and len(cols) > 10]
    assert x_reads, "the X reader was not used"
    for cols in x_reads:
        assert not set(cols) & set(sealed), "a sealed outcome / future / identifier column was requested by the X reader"
    assert not set(P.frame.columns) & (set(sealed) | {"Customer_Full_ID"})
    reg = P.registry.set_index("feature")
    for name, feats in S["sets"].items():
        for f in feats:
            raw = [c.strip() for c in str(reg.loc[f, "raw_columns"]).split(";")]
            assert not set(raw) & set(sealed), (name, f)
            assert not set(raw) & {"Customer_Full_ID", "Snapshot_Key"}, (name, f)


def test_sealed_reader_refuses_an_outcome_column(prepared: dict[str, Any]) -> None:
    from falls_ml.phase2.state import Phase2Stop
    from falls_ml.phase4.sealed import read_columns

    with pytest.raises(Phase2Stop) as e:
        read_columns(prepared["csv"], ["Customer_Full_ID", "Fall_Next_180D_Ind"], prepared["P"].sealed, prepared["L"]["contract"])
    assert e.value.gate == "SEALED_COLUMN_REQUESTED"


def test_unknown_clinical_column_stops_the_preflight(tmp_path: Path) -> None:
    from falls_ml.phase5.runner import STOP_LINE, run_phase5
    from falls_ml.phase5.synthetic import make_v21, write_v21_csv

    df, facts = make_v21(1500, scenario="null", seed=5, traps=("unknown_column",))
    (tmp_path / "in").mkdir()
    src = write_v21_csv(df, tmp_path / "in" / "v21.csv")
    out = tmp_path / "out"
    r = run_phase5(src, out, mode="quick", preflight_only=True, overrides={"eligibility": {"min_known_observed_rows": 20}})
    assert r["status"] == "STOPPED_PREFLIGHT" and r["exit_code"] == 2
    pf = out / "preflight"
    assert (pf / "PHASE5_PREFLIGHT.md").read_text(encoding="utf-8").rstrip().endswith(STOP_LINE) and STOP_LINE == "STOP - REVIEW REQUIRED"
    und = pd.read_csv(pf / "V21_UNDECLARED_COLUMNS.csv")
    assert list(und["column"]) == ["Balance_Clinic_Referral_Ind"] and (und["class"] == "REQUIRES_SEMANTIC_REVIEW").all()
    cl = pd.read_csv(pf / "ALL_V21_COLUMN_CLASSIFICATION.csv").set_index("column")
    assert cl.loc["Balance_Clinic_Referral_Ind", "class"] == "REQUIRES_SEMANTIC_REVIEW"
    for name in ("SCHEMA_DIFF_V1_V21.csv", "REMOVED_V1_COLUMNS.csv", "RENAMED_OR_CHANGED_COLUMNS.csv", "PREFLIGHT_RESULT.json"):
        assert (pf / name).is_file(), name
    assert not (out / "work" / "PLAN.json").exists() and not (out / "work" / "units").exists()       # nothing fitted


def test_post_index_records_and_a_leaky_new_feature_are_excluded(prepared: dict[str, Any], tmp_path: Path) -> None:
    P, S = prepared["P"], prepared["S"]
    reg = P.registry.set_index("feature")
    assert reg.loc["new_deficit_count_proxy", "class"] == "INELIGIBLE_LEAKAGE"          # single-feature AUROC >= 0.80 (trap)
    assert "new_deficit_count_proxy" not in {f for v in S["sets"].values() for f in v}
    r = _prepare(tmp_path, traps=("post_index_dx",), n=2000)
    reg2 = r["P"].registry.set_index("feature")
    dx = [f for f in reg2.index if f.startswith("new_") and reg2.loc[f, "domain"] in ("NEW_DIAGNOSIS", "NEW_VISION_HEARING")]
    assert dx and all(reg2.loc[f, "class"] in ("INELIGIBLE_TIMING", "SAFE_BOUNDED") for f in dx)
    assert any(reg2.loc[f, "class"] == "INELIGIBLE_TIMING" for f in dx)
    assert not r["P"].facts["v3_attestation"]["passed"]                                  # Last_Dx_Date after the index day withdraws V3
    assert "SAFE_ATTESTED" not in set(reg2["class"])


def test_post_index_or_ineligible_feature_in_x_hard_stops(prepared: dict[str, Any]) -> None:
    from falls_ml.phase2.state import Phase2Stop
    from falls_ml.phase5.runner import x_guard

    P, S, cfg = prepared["P"], prepared["S"], prepared["cfg"]
    x_guard(S["sets"], P.registry, S["kinds"], P.sealed, list(P.frame.columns), cfg)          # the real sets pass
    for bad in ({**S["sets"], "OLD_PLUS_ALL_NEW_ELIGIBLE": [*S["sets"]["OLD_PLUS_ALL_NEW_ELIGIBLE"], "new_deficit_count_proxy"]},     # leaky
                {**S["sets"], "OLD_PLUS_NEW_SAFE": [*S["sets"]["OLD_PLUS_NEW_SAFE"], "new_registry_smoking_subcode"]},                # not SAFE
                {**S["sets"], "OLD": [*S["sets"]["OLD"], "new_dizziness_ind"]},                                                      # NEW in OLD
                {**S["sets"], "OLD": [*S["sets"]["OLD"], "hypertension"]}):                                                          # removed in V21
        with pytest.raises(Phase2Stop) as e:
            x_guard(bad, P.registry, S["kinds"], P.sealed, list(P.frame.columns), cfg)
        assert e.value.gate == "X_LEAKAGE"
    with pytest.raises(Phase2Stop):
        x_guard(S["sets"], P.registry, S["kinds"], P.sealed, [*P.frame.columns, "Fall_Next_180D_Ind"], cfg)


def test_real_data_modelling_requires_a_preflight_first(tmp_path: Path) -> None:
    from falls_ml.phase2.state import Phase2Stop
    from falls_ml.phase4 import sealed
    from falls_ml.phase5.runner import run_phase5

    src = tmp_path / "in" / "extract.csv"
    src.parent.mkdir()
    src.write_text("Customer_Full_ID\n", encoding="utf-8")
    sealed.READ_LOG.clear()
    with pytest.raises(Phase2Stop) as e:
        run_phase5(src, tmp_path / "out", mode="overnight")
    assert e.value.gate == "PREFLIGHT_REQUIRED" and not sealed.READ_LOG


def test_unknown_cells_take_the_no_record_state() -> None:
    from falls_ml.phase5.data import _no_record_state

    assert _no_record_state("binary", "unexpected") == 0.0
    assert _no_record_state("count", "source_absent") == 0.0
    assert math.isnan(_no_record_state("days", "not_assessed"))
    assert math.isnan(_no_record_state("binary", "no_event"))
    assert _no_record_state("binary", "no_event", op="present") == 0.0


def test_raw_codes_learn_their_levels_on_training_rows_only() -> None:
    from falls_ml.phase5.design import LinearDesign

    meta = {"c": {"kind": "categorical", "linear": "onehot", "levels": None}}
    tr = pd.DataFrame({"c": [1.0] * 30 + [2.0] * 30 + [7.0] * 3 + [np.nan] * 37})
    d = LinearDesign(["c"], meta).fit(tr)
    assert d.levels_["c"] == [1.0, 2.0] and any(c.startswith("c__eq2") for c in d.columns_)
    assert d.transform(pd.DataFrame({"c": [9.0, 2.0, np.nan]})).shape == (3, len(d.columns_))


# ============================================================================ thresholds, objective, decision rule
def test_threshold_is_the_highest_reaching_the_target() -> None:
    from falls_ml.phase5.thresholds import operating_point

    y = np.array([1, 1, 1, 0, 1, 0, 0, 1, 0, 0])
    p = np.array([.9, .8, .7, .65, .6, .5, .4, .3, .2, .1])
    op = operating_point(y, p, 0.70)            # 5 events: >= 3.5 -> 4 captured at threshold 0.6
    assert op["threshold"] == pytest.approx(0.6) and op["tp"] == 4 and op["fp"] == 1 and op["flagged"] == 5
    assert op["sensitivity"] == pytest.approx(0.8) and op["false_alert_share"] == pytest.approx(0.2)
    ties = np.array([.9, .5, .5, .5, .5, .5, .1, .1, .1, .1])
    opt = operating_point(y, ties, 0.70)
    assert opt["flagged"] == 6                  # ties at the threshold are flagged together
    assert not operating_point(np.zeros(5), np.ones(5), 0.7)["feasible"]


def test_objective_prefers_fewer_flagged_then_fewer_false_alerts() -> None:
    from falls_ml.phase5.thresholds import objective

    rng = np.random.default_rng(1)
    y = (rng.random(4000) < 0.05).astype(float)
    good = y * 1.5 + rng.normal(0, 1, 4000)
    bad = y * 0.3 + rng.normal(0, 1, 4000)
    og = objective(y, 1 / (1 + np.exp(-good)), target=0.7, complexity=5, tie=1e-4)
    ob = objective(y, 1 / (1 + np.exp(-bad)), target=0.7, complexity=1, tie=1e-4)
    assert og["key"] < ob["key"] and og["flagged_share"] < ob["flagged_share"]
    assert all(math.isfinite(v) for v in og["key"])


def test_decision_rule_enet_primary_and_every_fold() -> None:
    from falls_ml.phase5.analysis import decide, overall
    from falls_ml.phase5.config import PRIMARY_FAMILY, load_phase5_config

    cfg = load_phase5_config()
    assert PRIMARY_FAMILY == "ENET" and cfg["primary_family"] == "ENET"

    def cmp(d: float, hi: float, slope: float = 1.0, b_lo: float = -0.001, folds: list[float] | None = None) -> dict[str, Any]:
        m = {"calibration_slope": slope, "calibration_intercept": 0.05}
        return {"delta_op_false_alert_share": d, "delta_op_false_alert_share_ci_high": hi, "delta_desc_fas_0.70": d, "delta_desc_fas_0.70_ci_high": hi,
                "delta_brier_ci_low": b_lo, "a": {"calibration_slope": 1.0, "calibration_intercept": 0.05}, "b": m,
                "fold_delta_false_alert_share": folds or [d] * 5}

    U, P_, N = "NEW_FEATURES_OPERATIONALLY_USEFUL", "PROMISING_BUT_NOT_ROBUST", "NO_ROBUST_OPERATIONAL_GAIN"
    # a nested interval below 0 that the equal-sensitivity interval does not confirm (threshold placement) is not enough
    c = cmp(-0.05, -0.01)
    c["delta_desc_fas_0.70_ci_high"] = 0.004
    r = decide(c, None, no_new=False, safe_same=True, cfg=cfg)
    assert r["verdict"] == P_ and not r["criteria"]["2_paired_ci_below_zero_nested_and_at_equal_sensitivity"]
    assert decide(cmp(-0.05, -0.01), cmp(-0.04, -0.01), no_new=False, safe_same=False, cfg=cfg)["verdict"] == U
    assert decide(cmp(-0.05, -0.01), None, no_new=False, safe_same=True, cfg=cfg)["verdict"] == U
    # 13: a "significant" bootstrap interval that ONE outer fold contradicts is not USEFUL
    r = decide(cmp(-0.05, -0.01, folds=[-0.06, -0.07, 0.01, -0.05, -0.04]), None, no_new=False, safe_same=True, cfg=cfg)
    assert r["verdict"] == P_ and not r["criteria"]["3_every_outer_fold_improves"]
    # interval crossing 0 but every fold improves -> promising; interval crossing 0 and folds mixed -> unstable = no robust gain
    assert decide(cmp(-0.05, 0.01), None, no_new=False, safe_same=True, cfg=cfg)["verdict"] == P_
    assert decide(cmp(-0.05, 0.01, folds=[-0.06, 0.02, -0.05, 0.01, -0.04]), None, no_new=False, safe_same=True, cfg=cfg)["verdict"] == N
    # the gain disappears without the timing / provenance-questionable predictors (NEW_SAFE) -> not robust
    assert decide(cmp(-0.05, -0.01), cmp(0.002, 0.02), no_new=False, safe_same=False, cfg=cfg)["verdict"] == P_
    assert decide(cmp(-0.05, -0.01), None, no_new=False, safe_same=False, cfg=cfg)["verdict"] == P_       # NEW_SAFE = OLD: no safe gain
    assert decide(cmp(-0.05, -0.01, slope=0.5), None, no_new=False, safe_same=True, cfg=cfg)["verdict"] == P_
    assert decide(cmp(0.01, 0.03), None, no_new=False, safe_same=True, cfg=cfg)["verdict"] == N
    assert decide(None, None, no_new=True, safe_same=True, cfg=cfg)["verdict"] == "NO_ELIGIBLE_NEW_FEATURES"
    # the answer is the PRIMARY (ENET) verdict only - never the best-looking family
    assert overall({"LASSO": U, "ENET": U, "XGB": N}) == "YES"
    assert overall({"LASSO": U, "ENET": N, "XGB": U}) == "NO"
    assert overall({"LASSO": N, "ENET": P_, "XGB": N}) == "UNCERTAIN"


# ============================================================================ 11 paired bootstrap
def test_fast_weighted_metrics_equal_the_reference_implementation() -> None:
    from falls_ml.evaluation.metrics import _auroc, _pr_auc
    from falls_ml.phase5.metrics import _Sorted

    rng = np.random.default_rng(3)
    y = (rng.random(3000) < 0.08).astype(float)
    p = np.round(rng.random(3000) * 0.5 + 0.3 * y, 3)          # ties on purpose
    s = _Sorted(y, p).stats(np.ones(len(y)), (0.7,))
    assert s["auroc"] == pytest.approx(_auroc(y, p), abs=1e-12)
    assert s["ap"] == pytest.approx(_pr_auc(y, p), abs=1e-12)
    w = np.bincount(rng.integers(0, 3000, 3000), minlength=3000).astype(float)
    rep = np.repeat(np.arange(3000), w.astype(int))
    s2 = _Sorted(y, p).stats(w, (0.7,))
    assert s2["auroc"] == pytest.approx(_auroc(y[rep], p[rep]), abs=1e-10)
    assert s2["ap"] == pytest.approx(_pr_auc(y[rep], p[rep]), abs=1e-10)


def test_paired_bootstrap_identical_models_give_zero_and_a_better_model_a_negative_interval() -> None:
    from falls_ml.phase5.metrics import paired_bootstrap
    from falls_ml.phase5.thresholds import operating_point

    rng = np.random.default_rng(5)
    y = (rng.random(5000) < 0.05).astype(float)
    pa = 1 / (1 + np.exp(-(y * 0.8 + rng.normal(0, 1, 5000) - 3)))
    fa = pa >= operating_point(y, pa, 0.7)["threshold"]
    r = paired_bootstrap(y, pa, pa.copy(), fa, fa.copy(), n_boot=200, seed=1)
    assert r["delta_op_false_alert_share"] == 0 and r["delta_op_false_alert_share_ci_low"] == 0 and r["delta_op_false_alert_share_ci_high"] == 0
    assert r["delta_auroc_ci_low"] == 0 and r["delta_auroc_ci_high"] == 0
    pb = 1 / (1 + np.exp(-(y * 2.5 + rng.normal(0, 1, 5000) - 3)))
    fb = pb >= operating_point(y, pb, 0.7)["threshold"]
    r2 = paired_bootstrap(y, pa, pb, fa, fb, n_boot=300, seed=2)
    assert r2["delta_op_false_alert_share_ci_high"] < 0 and r2["delta_ap_ci_low"] > 0 and r2["delta_op_false_alerts_per_10000"] < 0
    r3 = paired_bootstrap(y, pa, pb, fa, fb, n_boot=300, seed=2)
    assert r3["delta_op_false_alert_share_ci_high"] == r2["delta_op_false_alert_share_ci_high"]     # seeded, reproducible


# ============================================================================ 5-8 the outer labels never influence the inner choices
@pytest.mark.parametrize("family", ["LASSO", "ENET", "XGB"])
def test_unit_choices_do_not_depend_on_outer_labels(prepared: dict[str, Any], tmp_path: Path, family: str) -> None:
    from falls_ml.phase5.engine import UnitSpec, load_result, run_unit, tune_and_fit

    assert "y_te" not in inspect.signature(tune_and_fit).parameters        # the unit never receives the holdout outcome
    c1 = _small_ctx(prepared, tmp_path / "a")
    spec = UnitSpec(uid=f"PRIMARY|{family}|OLD_PLUS_ALL_NEW_ELIGIBLE|outer0", stage="PRIMARY", family=family, setname="OLD_PLUS_ALL_NEW_ELIGIBLE", outer=0)
    r1 = run_unit(c1, spec)
    y2 = c1.y.copy()
    te = c1.outer == 0
    y2[te] = 1 - y2[te]                                                    # every holdout label flipped
    c2 = _small_ctx(prepared, tmp_path / "b", y=y2)
    r2 = run_unit(c2, spec)
    js = __import__("json").dumps
    assert r1["config"] == r2["config"] and r1["thresholds"] == r2["thresholds"]
    assert js(r1["inner_objective"], sort_keys=True, default=str) == js(r2["inner_objective"], sort_keys=True, default=str)
    a1, a2 = load_result(c1, spec)["arrays"], load_result(c2, spec)["arrays"]
    assert np.array_equal(a1["inner_oof"], a2["inner_oof"]) and np.array_equal(a1["p_test"], a2["p_test"]) and np.array_equal(a1["test_idx"], a2["test_idx"])
    t = r1["thresholds"]["0.70"]
    assert t == pytest.approx(__import__("falls_ml.phase5.thresholds", fromlist=["x"]).operating_point(
        c1.y[a1["train_idx"]].astype(float), a1["inner_oof"], 0.70)["threshold"])                       # 70% threshold = inner OOF only


def test_committed_unit_is_never_recomputed(prepared: dict[str, Any], tmp_path: Path) -> None:
    from falls_ml.phase5.engine import UnitSpec, is_complete, run_unit, unit_dir

    c = _small_ctx(prepared, tmp_path)
    spec = UnitSpec(uid="PRIMARY|LASSO|OLD|outer1", stage="PRIMARY", family="LASSO", setname="OLD", outer=1)
    run_unit(c, spec)
    d = unit_dir(c, spec)
    before = {p.name: (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest()) for p in d.iterdir()}
    assert is_complete(c, spec)
    run_unit(c, spec)
    assert before == {p.name: (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest()) for p in d.iterdir()}
    (d / "result.json").write_text("{}", encoding="utf-8")                  # a damaged commit is detected
    assert not is_complete(c, spec)


# ============================================================================ 13 GPU -> CPU
def test_gpu_failure_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    import xgboost

    from falls_ml.phase5.config import load_phase5_config
    from falls_ml.phase5.models import DeviceState, xgb_inner_fold
    from falls_ml.phase5.resources import resolve_device

    real = xgboost.train
    calls: list[str] = []

    def fake(params: dict[str, Any], *a: Any, **k: Any) -> Any:
        calls.append(params["device"])
        if params["device"] != "cpu":
            raise xgboost.core.XGBoostError("CUDA error: no CUDA-capable device is detected (simulated)")
        return real(params, *a, **k)

    monkeypatch.setattr(xgboost, "train", fake)
    cfg = load_phase5_config()
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 5)).astype(np.float32)
    y = (X[:, 0] + rng.normal(size=600) > 1).astype(float)
    logs: list[str] = []
    st = DeviceState("cuda", log=logs.append)
    params = {"max_depth": 3, "learning_rate": 0.1, "min_child_weight": 1.0, "subsample": 1.0, "colsample_bytree": 1.0, "reg_alpha": 0.0, "reg_lambda": 1.0, "gamma": 0.0}
    r = xgb_inner_fold(X, y, X[:50], params, cfg, seed=1, nthread=1, state=st)
    assert len(r["pred"]) == 50 and st.device == "cpu" and st.fallbacks and calls[:2] == ["cuda", "cpu"]
    assert any("GPU FAILURE" in m for m in logs)
    xgb_inner_fold(X, y, X[:50], params, cfg, seed=2, nthread=1, state=st)
    assert calls[-1] == "cpu"                                               # the rest of the run stays on CPU
    monkeypatch.setattr(xgboost, "train", real)
    dev = resolve_device("gpu")
    assert dev["device"] in ("cpu", "cuda") and dev["reason"]


# ============================================================================ resources, config, status
def test_default_jobs_never_take_every_core(monkeypatch: pytest.MonkeyPatch) -> None:
    from falls_ml.phase5 import resources
    from falls_ml.phase5.config import load_phase5_config

    cfg = load_phase5_config()
    for n, want in ((1, 1), (2, 1), (8, 4), (16, 9), (64, 12)):
        monkeypatch.setattr(resources, "logical_cpus", lambda n=n: n)
        assert resources.default_jobs(cfg) == want


def test_config_is_validated_and_predeclared() -> None:
    from falls_ml.errors import ConfigError
    from falls_ml.phase5.config import load_phase5_config

    cfg = load_phase5_config()
    assert cfg["families"] == ["LASSO", "ENET", "XGB"] and cfg.primary_sensitivity == 0.70 and cfg.outer_folds == 5 and cfg.inner_folds == 5
    assert cfg["xgb"]["fixed"]["scale_pos_weight"] == 1.0
    assert cfg["sets"] == ["OLD", "OLD_PLUS_ALL_NEW_ELIGIBLE", "OLD_PLUS_NEW_SAFE"] and cfg["primary_comparison"] == ["OLD", "OLD_PLUS_ALL_NEW_ELIGIBLE"]
    b = cfg.budget
    assert (b["xgb"]["n_trials"], b["lasso"]["n_lambda"], b["enet"]["n_lambda"], b["bootstrap_n"], b["stability_linear"], b["stability_xgb"]) == \
        (100, 60, 40, 2000, 100, 30)
    for bad in ({"families": ["LASSO", "ENET", "XGB", "RF"]}, {"primary_family": "XGB"}, {"sets": ["OLD", "OLD_PLUS_NEW_SAFE"]}):
        with pytest.raises(ConfigError):
            load_phase5_config(overrides=bad)
    with pytest.raises(ConfigError):
        load_phase5_config(overrides={"operating": {"primary_sensitivity": 0.6}})
    assert load_phase5_config(overrides={"cv": {"outer_folds": 3}}).sha256 != cfg.sha256


def test_status_is_read_only_and_reports_a_missing_run(tmp_path: Path) -> None:
    from falls_ml.phase5.status import phase5_status

    r = phase5_status(tmp_path)
    assert r["status"] == "NOT_STARTED" and not any(tmp_path.iterdir())


# ============================================================================ privacy
def test_share_publication_fails_closed_on_an_identifier(prepared: dict[str, Any], tmp_path: Path) -> None:
    from falls_ml.phase2.state import Phase2Stop
    from falls_ml.phase5.report import publish

    tmp = tmp_path / ".tmp-share"
    tmp.mkdir()
    some_id = str(prepared["df"]["Customer_Full_ID"].iloc[5])
    (tmp / "x.md").write_text(f"patient {some_id}\n", encoding="utf-8")
    with pytest.raises(Phase2Stop) as e:
        publish(tmp_path, tmp, prepared["csv"], prepared["P"].frame)
    assert e.value.gate == "PRIVACY_SCAN" and not (tmp_path / "share").exists()
    tmp.mkdir()
    pd.DataFrame({"row_key": ["a"], "p": [0.1]}).to_csv(tmp / "t.csv", index=False)
    with pytest.raises(Phase2Stop):
        publish(tmp_path, tmp, prepared["csv"], prepared["P"].frame)
    tmp.mkdir()
    (tmp / "m.pkl").write_bytes(b"x")
    with pytest.raises(Phase2Stop):
        publish(tmp_path, tmp, prepared["csv"], prepared["P"].frame)


def test_small_cells_and_their_rates_are_suppressed() -> None:
    from falls_ml.phase5.report import suppress, suppress_grid

    t = suppress(pd.DataFrame({"n": [500, 500], "op_tp": [5, 50], "op_ppv": [0.1, 0.2], "n_features": [3, 3], "captured_falls": [5, 40]}))
    assert t.loc[0, "op_tp"] == "<10" and t.loc[0, "op_ppv"] == "suppressed" and t.loc[0, "n_features"] == 3 and t.loc[1, "op_ppv"] == 0.2
    g = suppress_grid(pd.DataFrame({"threshold": [.5, .1], "tp": [3, 40], "fp": [20, 300], "fn": [37, 0], "tn": [400, 120], "sensitivity": [.07, 1.0]}))
    assert g.loc[0, "tp"] == "<10" and g.loc[0, "sensitivity"] == "suppressed" and g.loc[1, "sensitivity"] == 1.0


# ============================================================================ 15 Windows paths / CLI
def test_windows_paths_are_detected_in_share_text() -> None:
    from falls_ml.phase2.stages_report import _path_hits

    win = "C:" + "\\" + "\\".join(["Users", "someone", "Downloads", "out5", "share"])       # built at run time: no literal path in the repository
    mac = "/" + "/".join(["Users", "someone", "data"])
    assert _path_hits(f"see {win}", [])
    assert _path_hits(f"see {mac}", [])
    assert not _path_hits("aggregate table, no path", [])


def test_cli_help_lists_the_phase5_commands_utf8() -> None:
    env = {**__import__("os").environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONIOENCODING": "utf-8"}
    r = subprocess.run([sys.executable, "-m", "falls_ml", "meuhedet-phase5", "--help"], capture_output=True, text=True, encoding="utf-8", env=env, cwd=ROOT)
    assert r.returncode == 0
    for flag in ("--mode", "--device", "--jobs", "--resume", "--preflight-only", "--estimate", "--status", "--report-only"):
        assert flag in r.stdout
    r2 = subprocess.run([sys.executable, "-m", "falls_ml", "meuhedet-phase5-synthetic", "--help"], capture_output=True, text=True, encoding="utf-8", env=env,
                        cwd=ROOT)
    assert r2.returncode == 0 and "--scenario" in r2.stdout


def test_out_folder_guards(tmp_path: Path) -> None:
    from falls_ml.phase2.state import Phase2Stop
    from falls_ml.phase5.runner import guard_out

    for m in ("PHASE2_PLAN.json", "PHASE3_PLAN.json", "PHASE4_PLAN.json"):
        d = tmp_path / m.split("_")[0]
        d.mkdir()
        (d / m).write_text("{}", encoding="utf-8")
        with pytest.raises(Phase2Stop) as e:
            guard_out(d, None, False)
        assert e.value.gate == "OUT_IS_EARLIER_PHASE"
    src = tmp_path / "x.csv"
    src.write_text("a\n", encoding="utf-8")
    with pytest.raises(Phase2Stop):
        guard_out(tmp_path, src, False)


# ============================================================================ 18 Phase 2 / 3 / 4 unchanged
@pytest.mark.parametrize("manifest", ["PHASE2_0.8.1_PROTECTED.sha256", "PHASE3_0.9.0_PROTECTED.sha256", "PHASE4_0.10.0_PROTECTED.sha256"])
def test_phase2_3_4_source_is_frozen_except_reviewed_lifecycle_patch(manifest: str, protected_source_matches) -> None:
    lines = [ln for ln in (ROOT / "configs/meuhedet" / manifest).read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]
    assert len(lines) > 10
    bad = []
    for ln in lines:
        digest, rel = ln.split("  ", 1)
        p = ROOT / rel
        if not p.is_file() or not protected_source_matches(rel, digest, p.read_bytes()):
            bad.append(rel)
    assert not bad, f"protected earlier-phase files changed: {bad[:10]}"
