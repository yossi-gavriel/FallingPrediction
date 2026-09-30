"""Unit tests for the reference derivation (spec D-02..D-09, D-06, RT-17). Synthetic records built inline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from falls_ml.data.schema import validate_modeling_dataset
from falls_ml.dataeng.derive import (
    TABLE_COLUMNS,
    DerivationRules,
    derive_modeling_frame,
    empty_event_tables,
    load_icd9cm_rules,
)
from falls_ml.errors import ConfigError, DatasetValidationError
from falls_ml.features.spec import load_feature_spec

ROOT = Path(__file__).resolve().parents[2]
SPEC = load_feature_spec(ROOT / "configs" / "features" / "efalls_v1.yaml")
INDEX = pd.Timestamp("2018-04-01")


def d(days: int) -> pd.Timestamp:
    return INDEX + pd.Timedelta(days=days)


def person(rid: str = "p1", dob: str = "1940-06-15", sex: str = "female", start: str = "2000-01-01", end=None,
           death=None, practice: str = "P01") -> dict:
    return {"research_id": rid, "date_of_birth": dob, "sex": sex, "practice_id": practice, "deprivation_group": "3",
            "site_id": "S1", "death_date": death, "membership_start": start, "membership_end": end}


def cond(feature: str, date, available=None, rid: str = "p1", source_class: str = "community_diagnosis") -> dict:
    return {"research_id": rid, "feature": feature, "record_date": date,
            "available_date": date if available is None else available, "source_class": source_class}


def rx(paragraph: str, date, is_drug: bool = True, rid: str = "p1") -> dict:
    return {"research_id": rid, "bnf_paragraph": paragraph, "bnf_chapter": int(paragraph[:2]), "is_drug": is_drug,
            "record_date": date, "available_date": date}


def meas(kind: str, value: float, date, rid: str = "p1") -> dict:
    return {"research_id": rid, "kind": kind, "value": value, "record_date": date, "available_date": date}


def life(kind: str, value, date, rid: str = "p1") -> dict:
    return {"research_id": rid, "kind": kind, "value": value, "record_date": date, "available_date": date}


def enc(code: str, date, *, etype: str = "ed_attendance", method=None, spell=None, position: int = 1,
        system: str = "ICD-10", rid: str = "p1", eid: str = "e1") -> dict:
    return {"research_id": rid, "encounter_id": eid, "encounter_type": etype, "admission_method": method,
            "spell_start_date": spell, "event_date": date, "diagnosis_code": code, "diagnosis_position": position,
            "code_system": system}


def tables(persons=None, **records) -> dict[str, pd.DataFrame]:
    out = empty_event_tables()
    out["persons"] = pd.DataFrame(persons or [person()], columns=TABLE_COLUMNS["persons"])
    for name, rows in records.items():
        out[name] = pd.DataFrame(rows, columns=TABLE_COLUMNS[name])
    return out


def derive(tbls, rids=("p1",), index=INDEX, rules: DerivationRules | None = None):
    idx = pd.DataFrame({"research_id": list(rids), "index_date": [index] * len(rids)})
    return derive_modeling_frame(tbls, idx, SPEC, rules)


def one(tbls, rules: DerivationRules | None = None) -> pd.Series:
    frame, _ = derive(tbls, rules=rules)
    assert len(frame) == 1
    return frame.iloc[0]


# ---------------------------------------------------------------------- output contract
def test_frame_has_exact_modelling_columns_and_validates() -> None:
    tbls = tables(persons=[person("p1"), person("p2", sex="male")], encounters=[enc("S72.0", d(10), rid="p2")])
    frame, dlog = derive(tbls, rids=("p1", "p2"))
    expected = ["research_id", "index_date", "predictor_max_record_date", "outcome_12m", "outcome_first_event_date",
                *SPEC.predictor_names(), *SPEC.metadata_columns]
    assert list(frame.columns) == expected
    assert frame["outcome_12m"].dtype == "int8" and frame["polypharmacy_count_120d"].dtype == "int64"
    assert all(frame[b].dtype == "int8" for b in SPEC.binary_names()) and len(SPEC.binary_names()) == 72
    validate_modeling_dataset(frame.assign(dataset_version="t", mapping_version="t", source="unit"), SPEC)
    assert dlog.n_input == 2 and dlog.n_output == 2


# ---------------------------------------------------------------------- polypharmacy (D-03)
def test_polypharmacy_counts_distinct_drug_paragraphs_and_excludes_chapters() -> None:
    row = one(tables(prescriptions=[rx("040201", d(-10)), rx("040201", d(-40)), rx("020101", d(-5)),
                                    rx("200101", d(-5)), rx("230201", d(-5)), rx("060106", d(-5), is_drug=False)]))
    assert row["polypharmacy_count_120d"] == 2


def test_polypharmacy_sensitivity_window_and_chapters() -> None:
    tbls = tables(prescriptions=[rx("040201", d(-100)), rx("200101", d(-5))])
    row = one(tbls, DerivationRules(polypharmacy_window_days=90, excluded_bnf_chapters=()))
    assert row["polypharmacy_count_120d"] == 1  # d-100 outside 90 days; chapter 20 no longer excluded


# ---------------------------------------------------------------------- BMI (D-02)
def test_bmi_most_recent_valid_recorded_value_same_day_mean_and_invalid_counted() -> None:
    tbls = tables(measurements=[meas("bmi", 22.0, d(-900)), meas("bmi", 26.0, d(-50)), meas("bmi", 28.0, d(-50)),
                                meas("bmi", 250.0, d(-10)), meas("bmi", 5.0, d(-5)),
                                meas("weight_kg", 90.0, d(-3)), meas("height_m", 1.5, d(-3))])
    frame, dlog = derive(tbls)
    assert frame.loc[0, "bmi_value"] == pytest.approx(27.0)  # recorded beats a more recent computed value
    assert dlog.invalid_bmi_values == 2


def test_bmi_computed_from_nearest_weight_height_pair_when_no_recorded_value() -> None:
    older = [meas("weight_kg", 80.0, d(-400)), meas("height_m", 2.0, d(-420)),
             meas("weight_kg", 72.9, d(-100)), meas("height_m", 1.8, d(-130))]  # 30-day gap: paired (boundary)
    # the most recent weight has a VALID would-be BMI (99 / 1.9^2 = 27.4) but its only height is 31 days away
    tbls = tables(measurements=[*older, meas("weight_kg", 99.0, d(-20)), meas("height_m", 1.9, d(-51))])
    assert one(tbls)["bmi_value"] == pytest.approx(72.9 / 1.8 ** 2)
    tbls = tables(measurements=[*older, meas("weight_kg", 99.0, d(-20)), meas("height_m", 1.9, d(-50))])
    assert one(tbls)["bmi_value"] == pytest.approx(99.0 / 1.9 ** 2)  # 30 days: most recent pair wins
    wide = DerivationRules(weight_height_max_gap_days=31)
    tbls = tables(measurements=[*older, meas("weight_kg", 99.0, d(-20)), meas("height_m", 1.9, d(-51))])
    assert one(tbls, wide)["bmi_value"] == pytest.approx(99.0 / 1.9 ** 2)


def test_bmi_invalid_computed_value_ignored_and_window_boundary() -> None:
    five_years = INDEX - pd.DateOffset(years=5)
    tbls = tables(measurements=[meas("weight_kg", 70.0, d(-10)), meas("height_m", 170.0, d(-10)),  # cm: BMI invalid
                                meas("bmi", 31.0, five_years)])
    frame, dlog = derive(tbls)
    assert frame.loc[0, "bmi_value"] == 31.0 and dlog.invalid_bmi_values == 0  # recorded value wins; pair not needed
    tbls = tables(measurements=[meas("weight_kg", 70.0, d(-10)), meas("height_m", 170.0, d(-10)),
                                meas("bmi", 31.0, five_years - pd.Timedelta(days=1))])
    frame, dlog = derive(tbls)
    assert np.isnan(frame.loc[0, "bmi_value"]) and dlog.invalid_bmi_values == 1


# ---------------------------------------------------------------------- smoking (D-04)
@pytest.mark.parametrize(("records", "expected"), [
    ([("ex", -900), ("never", -100)], "ex"),
    ([("current", -3000), ("never", -10)], "ex"),
    ([("ex", -900), ("current", -100)], "current"),
    ([("current", -900), ("ex", -100)], "ex"),
    ([("never", -900), ("never", -100)], "never"),
    ([("never", -10), ("current", -10)], "current"),  # same-day conflict: most severe
    ([("current", 0)], None),  # index-day record unusable
    ([], None),
])
def test_smoking_status(records, expected) -> None:
    row = one(tables(lifestyle=[life("smoking", v, d(o)) for v, o in records]))
    assert (pd.isna(row["smoking_status"]) if expected is None else row["smoking_status"] == expected)


# ---------------------------------------------------------------------- alcohol (D-05)
@pytest.mark.parametrize(("units", "expected"), [(0, "zero"), (1, "lower_risk"), (20, "lower_risk"),
                                                 (21, "higher_risk"), (48, "higher_risk"), (49, "harmful"),
                                                 ("70", "harmful")])
def test_alcohol_units_mapping(units, expected) -> None:
    assert one(tables(lifestyle=[life("alcohol_units_week", units, d(-30))]))["alcohol_category"] == expected


def test_alcohol_precedence_window_and_same_day_coded_beats_units() -> None:
    row = one(tables(lifestyle=[life("alcohol", "harmful", d(-1400)), life("alcohol", "lower_risk", d(-10))]))
    assert row["alcohol_category"] == "harmful"
    row = one(tables(lifestyle=[life("alcohol", "previous_higher_risk_or_harmful", d(-10)),
                                life("alcohol", "lower_risk", d(-5)), life("alcohol", "zero", d(-1))]))
    assert row["alcohol_category"] == "previous_higher_risk_or_harmful"
    row = one(tables(lifestyle=[life("alcohol", "lower_risk", d(-10)), life("alcohol_units_week", 60, d(-10))]))
    assert row["alcohol_category"] == "lower_risk"
    row = one(tables(lifestyle=[life("alcohol", "lower_risk", d(-10)), life("alcohol_units_week", 60, d(-11))]))
    assert row["alcohol_category"] == "harmful"
    start = INDEX - pd.DateOffset(years=5)
    assert one(tables(lifestyle=[life("alcohol", "harmful", start)]))["alcohol_category"] == "harmful"
    outside = start - pd.Timedelta(days=1)
    assert pd.isna(one(tables(lifestyle=[life("alcohol", "harmful", outside)]))["alcohol_category"])
    assert pd.isna(one(tables(lifestyle=[life("alcohol", "harmful", INDEX)]))["alcohol_category"])  # index day unusable


# ---------------------------------------------------------------------- binaries: windows, age, resolution (D-08)
def test_age_at_record_rule_and_global_sensitivity() -> None:
    dob = "1945-01-01"  # fracture requires age >= 55 at the record
    young = pd.Timestamp("1945-01-01") + pd.Timedelta(days=int(54.99 * 365.25))
    old = pd.Timestamp("1945-01-01") + pd.Timedelta(days=int(55.01 * 365.25))
    assert one(tables(persons=[person(dob=dob)], conditions=[cond("fracture", young)]))["fracture"] == 0
    assert one(tables(persons=[person(dob=dob)], conditions=[cond("fracture", old)]))["fracture"] == 1
    loose = DerivationRules(apply_time_window_rules=False)
    assert one(tables(persons=[person(dob=dob)], conditions=[cond("fracture", young)]), loose)["fracture"] == 1


def test_lookback_years_boundary() -> None:
    start = INDEX - pd.DateOffset(years=5)
    assert one(tables(conditions=[cond("depression", start)]))["depression"] == 1
    assert one(tables(conditions=[cond("depression", start - pd.Timedelta(days=1))]))["depression"] == 0
    assert one(tables(conditions=[cond("dementia", "1990-01-01")]))["dementia"] == 1  # complete history


@pytest.mark.parametrize(("dementia_date", "expected"), [
    (None, 1), ("2016-01-01", 1), ("2017-01-01", 0), ("2017-06-01", 0), ("2018-04-01", 1)])
def test_resolution_hierarchy_timing(dementia_date, expected) -> None:
    records = [cond("memory_concerns", "2017-01-01")]
    if dementia_date:
        records.append(cond("dementia", dementia_date))
    row = one(tables(conditions=records))
    assert row["memory_concerns"] == expected


def test_resolution_uses_latest_resolved_record_and_measurement_evidence() -> None:
    row = one(tables(conditions=[cond("memory_concerns", "2015-01-01"), cond("cognitive_impairment", "2016-01-01"),
                                 cond("memory_concerns", "2017-01-01")]))
    assert row["memory_concerns"] == 1 and row["cognitive_impairment"] == 1
    row = one(tables(conditions=[cond("dementia", "2017-02-01")], measurements=[meas("sixcit", 10, "2017-01-01")]))
    assert row["cognitive_impairment"] == 0 and row["dementia"] == 1
    assert one(tables(measurements=[meas("sixcit", 8, "2017-01-01")]))["cognitive_impairment"] == 1


# ---------------------------------------------------------------------- measurement rules (D-08, M-14)
@pytest.mark.parametrize(("feature", "kind", "value", "expected"), [
    ("activity_limitation", "barthel", 18, 1), ("activity_limitation", "barthel", 19, 0),
    ("cognitive_impairment", "sixcit", 7, 0), ("osteoporosis", "dexa_tscore", -2.6, 1),
    ("osteoporosis", "dexa_tscore", -2.5, 0), ("peripheral_vascular_disease", "abpi", 0.94, 1),
    ("peripheral_vascular_disease", "abpi", 0.95, 0), ("chronic_kidney_disease", "egfr", 59.9, 1),
    ("chronic_kidney_disease", "egfr", 60.0, 0), ("chronic_kidney_disease", "acr_mg_mmol", 3.1, 1),
    ("chronic_kidney_disease", "acr_mg_mmol", 3.0, 0), ("thyroid_problems", "tsh_mu_l", 0.35, 1),
    ("thyroid_problems", "tsh_mu_l", 5.6, 1), ("thyroid_problems", "tsh_mu_l", 2.0, 0),
])
def test_single_reading_measurement_rules(feature, kind, value, expected) -> None:
    tbls = tables(measurements=[meas(kind, value, d(-100))])
    assert one(tbls)[feature] == expected
    assert one(tbls, DerivationRules(apply_measurement_rules=False))[feature] == 0


def test_blood_pressure_rules_need_three_readings_not_mixed() -> None:
    three = [meas("sbp", 150, d(-300)), meas("sbp", 141, d(-200)), meas("sbp", 140, d(-100))]
    assert one(tables(measurements=three))["hypertension"] == 1
    mixed = [meas("sbp", 150, d(-300)), meas("sbp", 141, d(-200)), meas("dbp", 95, d(-100))]
    assert one(tables(measurements=mixed))["hypertension"] == 0
    low = [meas("dbp", 55, d(-300)), meas("dbp", 59, d(-200)), meas("dbp", 50, d(-100)), meas("sbp", 120, d(-100))]
    row = one(tables(measurements=low))
    assert row["hypotension_or_syncope"] == 1 and row["hypertension"] == 0
    assert one(tables(measurements=three[:2] + [meas("sbp", 150, d(0))]))["hypertension"] == 0  # index day unusable


@pytest.mark.parametrize(("sex", "readings", "expected"), [
    ("female", [(11.4, -100)], 1), ("female", [(11.5, -100)], 0), ("male", [(12.9, -100)], 1),
    ("male", [(13.0, -100)], 0), ("female", [(10.0, -300), (12.5, -100)], 0), ("female", [(12.5, -300), (10.0, -100)], 1),
])
def test_anaemia_sex_specific_hb_with_later_normal_resolution(sex, readings, expected) -> None:
    tbls = tables(persons=[person(sex=sex)], measurements=[meas("hb_g_dl", v, d(o)) for v, o in readings])
    assert one(tbls)["anaemia_and_haematinic_deficiency"] == expected


def test_later_normal_hb_resolves_only_the_hb_criterion_not_codes() -> None:
    # D-08: "code OR low Hb, resolved by a later normal Hb" -- a haematinic-deficiency code is not removed by an Hb
    # value, so adding the measurement rule can never remove a code-based case.
    feature = "anaemia_and_haematinic_deficiency"
    coded = tables(conditions=[cond(feature, d(-300))], measurements=[meas("hb_g_dl", 13.5, d(-100))])
    assert one(coded)[feature] == 1
    assert one(coded, DerivationRules(apply_measurement_rules=False))[feature] == 1
    coded_and_low = tables(conditions=[cond(feature, d(-400))],
                           measurements=[meas("hb_g_dl", 10.0, d(-300)), meas("hb_g_dl", 13.5, d(-100))])
    frame, _ = derive(coded_and_low)
    assert frame.loc[0, feature] == 1
    only_low = tables(measurements=[meas("hb_g_dl", 10.0, d(-300)), meas("hb_g_dl", 13.5, d(-100))])
    assert one(only_low)[feature] == 0
    assert one(only_low, DerivationRules(apply_time_window_rules=False))[feature] == 1  # no resolution rules
    same_day = tables(measurements=[meas("hb_g_dl", 10.0, d(-100)), meas("hb_g_dl", 13.5, d(-100))])
    assert one(same_day)[feature] == 1  # resolution needs a strictly later normal value


# ---------------------------------------------------------------------- outcome (D-09, M-02, RT-17)
@pytest.mark.parametrize(("code", "system", "expected"), [
    ("S72.0", "ICD-10", 1), ("S720", "ICD-10", 1), ("s72.01", "ICD-10", 1), ("S62.1", "ICD-10", 0),
    ("T14.2", "ICD-10", 1), ("T14.1", "ICD-10", 0), ("W19", "ICD-10", 1), ("W1", "ICD-10", 0), ("W20", "ICD-10", 0),
    ("M80.0", "ICD-10", 1), ("E88.0", "ICD-10", 0), ("S02.0", "ICD-10", 0),
    ("S72001A", "ICD-10-CM", 1), ("S72001D", "ICD-10-CM", 0), ("S72001S", "ICD-10-CM", 0), ("W19XXXA", "ICD-10-CM", 1),
    ("S72.0", "ICD-10-CM", 1),
    ("E887", "ICD-9-CM", 0), ("820.8", "ICD-9-CM", 1), ("E880.9", "ICD-9-CM", 1), ("E886.9", "ICD-9-CM", 0),
    ("E886.0", "ICD-9-CM", 1), ("880.9", "ICD-9-CM", 0), ("805.2", "ICD-9-CM", 1), ("805.1", "ICD-9-CM", 0),
    ("733.10", "ICD-9-CM", 0), ("E88.0", "ICD-9-CM", 0), ("W1.9", "ICD-10", 0),
])
def test_outcome_code_matching(code, system, expected) -> None:
    assert one(tables(encounters=[enc(code, d(30), system=system)]))["outcome_12m"] == expected


def test_malformed_dotted_codes_are_counted_not_matched() -> None:
    _, dlog = derive(tables(encounters=[enc("E88.0", d(30), system="ICD-9-CM"), enc("W1.9", d(30), eid="e2")]))
    assert dlog.record_exclusions["encounters.malformed_ICD-9-CM_code_in_window"] == 1
    assert dlog.record_exclusions["encounters.malformed_ICD-10_code_in_window"] == 1


def test_admission_rules_elective_unknown_and_spell_start() -> None:
    elective = tables(encounters=[enc("S72.0", d(30), etype="hospital_admission", method="elective")])
    assert one(elective)["outcome_12m"] == 0
    assert one(elective, DerivationRules(outcome_admission_rule="any"))["outcome_12m"] == 1
    for method in ("emergency", "urgent", "via_ed"):
        assert one(tables(encounters=[enc("S72.0", d(30), etype="hospital_admission", method=method)]))["outcome_12m"] == 1
    other = tables(encounters=[enc("W19", d(30), etype="other")])
    assert one(other)["outcome_12m"] == 0

    unknown = tables(encounters=[enc("S72.0", d(30), etype="hospital_admission", method=None)])
    frame, dlog = derive(unknown)
    assert frame.loc[0, "outcome_12m"] == 0 and dlog.unknown_admission_method_count == 1
    assert one(unknown, DerivationRules(unknown_admission_method="include"))["outcome_12m"] == 1
    with pytest.raises(DatasetValidationError, match="unknown admission method"):
        derive(unknown, rules=DerivationRules(unknown_admission_method="error"))

    continuation = tables(encounters=[enc("S72.0", d(5), etype="hospital_admission", method="emergency", spell=d(-3))])
    assert one(continuation)["outcome_12m"] == 0
    assert one(continuation, DerivationRules(exclude_spells_started_before_index=False))["outcome_12m"] == 1


def test_position_rule_and_first_event_date_and_audit() -> None:
    tbls = tables(encounters=[enc("R55", d(40), eid="a", position=1), enc("S72.0", d(40), eid="a", position=2),
                              enc("S00.9", d(200), eid="b", position=1), enc("W19", d(200), eid="b", position=3),
                              enc("S82.1", d(20), eid="c", etype="hospital_admission", method="elective")])
    frame, dlog = derive(tbls)
    assert frame.loc[0, "outcome_12m"] == 1 and frame.loc[0, "outcome_first_event_date"] == d(40)
    frame, dlog2 = derive(tbls, rules=DerivationRules(outcome_position_rule="principal_fracture_or_any_external"))
    assert frame.loc[0, "outcome_first_event_date"] == d(200)
    audit = dlog.outcome_audit.set_index(["code_block", "status"])["n_records"]
    assert audit[("S72", "qualifying")] == 1 and audit[("W19", "qualifying")] == 1
    assert audit[("S82", "admission_not_non_elective")] == 1
    assert ("S72", "position_rule") in dlog2.outcome_audit.set_index(["code_block", "status"]).index
    assert list(dlog.outcome_audit.columns) == ["code_system", "efalls_code", "code_block", "kind", "diagnosis_position",
                                                "encounter_type", "admission_method", "status", "n_records"]


def test_icd9cm_candidate_list_is_unvalidated_and_longest_prefix_wins() -> None:
    rules = load_icd9cm_rules(ROOT / "configs" / "mappings" / "outcome_code_candidates.yaml")
    by_prefix = {r.prefix: r.status for r in rules}
    assert by_prefix["E886"] == "CANDIDATE" and by_prefix["E8869"] == "REVIEW" and by_prefix["E887"] == "EXCLUDE"
    assert {"8052", "8053", "8070", "8074", "810", "811", "812", "E880", "E888", "829"} <= set(by_prefix)
    import yaml

    raw = yaml.safe_load((ROOT / "configs" / "mappings" / "outcome_code_candidates.yaml").read_text(encoding="utf-8"))
    assert raw["code_list"]["clinically_validated"] is False and "UNVALIDATED" in raw["code_list"]["status"]


# ---------------------------------------------------------------------- eligibility and age (spec §4, D-06, M-01)
def test_eligibility_exclusions_are_counted_by_first_reason() -> None:
    persons = [
        person("ok"), person("young", dob="1960-01-01"), person("dead", death="2018-03-31"),
        person("dies_on_index", death="2018-04-01"), person("joined_late", start="2018-04-02"),
        person("left", end="2018-03-31"), person("sexless", sex="unknown"), person("no_dob", dob=None),
        person("gap", start="2000-01-01", end="2017-12-31", practice="P01"),
        person("gap", start="2018-03-01", end=None, practice="P02"),
    ]
    rids = ("ok", "young", "dead", "dies_on_index", "joined_late", "left", "sexless", "no_dob", "gap", "missing")
    frame, dlog = derive(tables(persons=persons), rids=rids)
    assert sorted(frame["research_id"]) == ["dies_on_index", "gap", "ok"]
    assert dlog.exclusions == {"missing_person_record": 1, "missing_date_of_birth": 1, "invalid_sex": 1,
                               "died_before_index": 1, "not_member_at_index": 2, "age_below_minimum": 1}
    assert dlog.n_input == 10 and dlog.n_output == 3
    gap = frame.set_index("research_id").loc["gap"]
    assert gap["practice_id"] == "P02"  # cluster valid at index (M-01)


def test_decimal_age_and_followup_end() -> None:
    tbls = tables(persons=[person(dob="1950-10-01", death="2018-12-01"), person("p2", dob="1950-04-01", end="2019-01-15"),
                           person("p3", dob="1940-01-01")])
    frame, _ = derive(tbls, rids=("p1", "p2", "p3"))
    f = frame.set_index("research_id")
    assert f.loc["p1", "age_years"] == pytest.approx((INDEX - pd.Timestamp("1950-10-01")).days / 365.25)
    assert f.loc["p1", "age_years"] % 1 != 0  # not floored to completed years
    assert f.loc["p1", "followup_end_date"] == pd.Timestamp("2018-12-01")
    assert f.loc["p2", "followup_end_date"] == pd.Timestamp("2019-01-15")
    assert f.loc["p3", "followup_end_date"] == pd.Timestamp("2019-03-31")


# ---------------------------------------------------------------------- contract violations fail loudly
def test_contract_violations_raise() -> None:
    with pytest.raises(DatasetValidationError, match="available_date"):
        derive(tables(conditions=[{**cond("falls", d(-10)), "available_date": None}]))
    with pytest.raises(DatasetValidationError, match="unknown kinds"):
        derive(tables(measurements=[meas("glucose", 5.0, d(-10))]))
    with pytest.raises(DatasetValidationError, match="overlapping"):
        derive(tables(persons=[person(start="2000-01-01", end="2019-01-01"), person(start="2018-01-01")]))
    with pytest.raises(DatasetValidationError, match="duplicated"):
        derive(tables(), rids=("p1", "p1"))
    with pytest.raises(DatasetValidationError, match="bnf_chapter"):
        derive(tables(prescriptions=[{**rx("040201", d(-10)), "bnf_chapter": 5}]))
    with pytest.raises(DatasetValidationError, match="smoking"):
        derive(tables(lifestyle=[life("smoking", "sometimes", d(-10))]))
    with pytest.raises(ConfigError):
        DerivationRules(outcome_admission_rule="elective_only")
    with pytest.raises(ConfigError):
        DerivationRules(source_availability={"encounters": "unknown"})


def test_condition_features_outside_spec_are_counted_not_used() -> None:
    frame, dlog = derive(tables(conditions=[cond("not_a_feature", d(-10)), cond("falls", d(-20))]))
    assert dlog.record_exclusions["conditions.feature_not_in_spec"] == 1
    assert frame.loc[0, "predictor_max_record_date"] == d(-20)
