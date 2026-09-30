"""Temporal leakage (spec D-00, D-09, RT-17): nothing dated or available on/after the index day reaches predictors."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from falls_ml.data.synthetic import generate_synthetic_event_tables
from falls_ml.dataeng.derive import (
    MEASUREMENT_KINDS,
    PREDICTOR_TABLES,
    TABLE_COLUMNS,
    DerivationRules,
    derive_modeling_frame,
    empty_event_tables,
)
from falls_ml.features.spec import load_feature_spec
from falls_ml.seeding import rng_for

ROOT = Path(__file__).resolve().parents[2]
SPEC = load_feature_spec(ROOT / "configs" / "features" / "efalls_v1.yaml")
INDEX = pd.Timestamp("2018-04-01")
PERSON = {"research_id": "p1", "date_of_birth": "1940-01-01", "sex": "male", "practice_id": "P1",
          "deprivation_group": "1", "site_id": "S1", "death_date": None, "membership_start": "1990-01-01",
          "membership_end": None}


def d(days: int) -> pd.Timestamp:
    return INDEX + pd.Timedelta(days=days)


def derive(rules: DerivationRules | None = None, **records) -> pd.Series:
    tables = empty_event_tables()
    tables["persons"] = pd.DataFrame([PERSON])
    for name, rows in records.items():
        tables[name] = pd.DataFrame(rows, columns=TABLE_COLUMNS[name])
    frame, _ = derive_modeling_frame(tables, pd.DataFrame({"research_id": ["p1"], "index_date": [INDEX]}), SPEC, rules)
    return frame.iloc[0]


def cond(feature: str, record, available=None) -> dict:
    return {"research_id": "p1", "feature": feature, "record_date": record,
            "available_date": record if available is None else available, "source_class": "community_diagnosis"}


def ed(code: str, event, position: int = 1) -> dict:
    return {"research_id": "p1", "encounter_id": f"e{event}", "encounter_type": "ed_attendance", "admission_method": None,
            "spell_start_date": None, "event_date": event, "diagnosis_code": code, "diagnosis_position": position,
            "code_system": "ICD-10"}


# ---------------------------------------------------------------------- predictors (D-00)
def test_condition_on_index_day_does_not_set_feature() -> None:
    row = derive(conditions=[cond("falls", d(0))])
    assert row["falls"] == 0 and pd.isna(row["predictor_max_record_date"])
    assert derive(conditions=[cond("falls", d(-1))])["falls"] == 1


def test_record_before_index_but_available_on_or_after_index_is_unusable() -> None:
    assert derive(conditions=[cond("dementia", d(-400), available=d(0))])["dementia"] == 0
    assert derive(conditions=[cond("dementia", d(-400), available=d(30))])["dementia"] == 0
    assert derive(conditions=[cond("dementia", d(-400), available=d(-1))])["dementia"] == 1


def test_unknown_availability_table_respects_lag_buffer() -> None:
    rules = DerivationRules(source_availability={"conditions": "unknown"})
    assert derive(rules, conditions=[cond("dementia", d(-30), available=pd.NaT)])["dementia"] == 0
    assert derive(rules, conditions=[cond("dementia", d(-31), available=pd.NaT)])["dementia"] == 1
    assert derive(rules, conditions=[cond("dementia", d(-100), available=d(0))])["dementia"] == 0
    wider = DerivationRules(source_availability={"conditions": "unknown"}, unknown_availability_lag_days=90)
    assert derive(wider, conditions=[cond("dementia", d(-60), available=pd.NaT)])["dementia"] == 0


def test_polypharmacy_window_boundaries_rt17() -> None:
    rx = [{"research_id": "p1", "bnf_paragraph": para, "bnf_chapter": int(para[:2]), "is_drug": True,
           "record_date": d(offset), "available_date": d(offset)}
          for para, offset in (("010101", -121), ("020101", -120), ("030101", -1), ("040101", 0))]
    row = derive(prescriptions=rx)
    assert row["polypharmacy_count_120d"] == 2
    assert row["predictor_max_record_date"] == d(-1)


# ---------------------------------------------------------------------- outcome window (D-00, D-09)
@pytest.mark.parametrize(("event", "expected"), [
    (d(-1), 0), (INDEX, 1), (pd.Timestamp("2019-03-31"), 1), (pd.Timestamp("2019-04-01"), 0)])
def test_outcome_window_boundaries_rt17(event, expected) -> None:
    row = derive(encounters=[ed("W19", event)])
    assert row["outcome_12m"] == expected
    assert (row["outcome_first_event_date"] == event) if expected else pd.isna(row["outcome_first_event_date"])


def test_outcome_window_end_handles_leap_years() -> None:
    assert SPEC.outcome.window_end(pd.Timestamp("2020-03-01")) == pd.Timestamp("2021-02-28")
    assert SPEC.outcome.window_end(pd.Timestamp("2019-03-01")) == pd.Timestamp("2020-02-29")


def test_outcome_encounters_never_feed_predictors() -> None:
    row = derive(encounters=[ed("W19", d(-30)), ed("S72.0", d(-30), position=2), ed("M80.0", d(-900))])
    assert row["falls"] == 0 and row["fracture"] == 0 and row["fragility_fracture"] == 0
    assert pd.isna(row["predictor_max_record_date"]) and row["outcome_12m"] == 0


# ---------------------------------------------------------------------- metamorphic and random-table checks
def _random_tables(seed: int, n: int = 60) -> dict[str, pd.DataFrame]:
    """Random canonical tables with records scattered around the index date (software test data only)."""
    rng = rng_for(seed, "tests.leakage.random_tables")
    ids = [f"r{i}" for i in range(n)]
    persons = pd.DataFrame({"research_id": ids, "date_of_birth": INDEX - pd.to_timedelta(rng.integers(66 * 365, 95 * 365, n), unit="D"),
                            "sex": rng.choice(["female", "male"], n), "practice_id": "P1", "deprivation_group": "1",
                            "site_id": "S1", "death_date": pd.NaT, "membership_start": pd.Timestamp("1990-01-01"),
                            "membership_end": pd.NaT})

    def dated(k: int) -> dict:
        record = INDEX + pd.to_timedelta(rng.integers(-3000, 400, k), unit="D")
        lag = np.where(rng.random(k) < 0.2, rng.integers(0, 900, k), rng.integers(0, 5, k))
        return {"research_id": rng.choice(ids, k), "record_date": record, "available_date": record + pd.to_timedelta(lag, unit="D")}

    k = 3000
    chapter = rng.choice([1, 2, 4, 20], k)
    kinds = rng.choice(MEASUREMENT_KINDS, k)
    tables = {
        "persons": persons,
        "conditions": pd.DataFrame({**dated(k), "feature": rng.choice(SPEC.binary_names(), k), "source_class": "community_diagnosis"}),
        "prescriptions": pd.DataFrame({**dated(k), "bnf_paragraph": [f"{c:02d}0{rng.integers(1, 4)}01" for c in chapter],
                                       "bnf_chapter": chapter, "is_drug": True}),
        "measurements": pd.DataFrame({**dated(k), "kind": kinds,
                                      "value": np.where(kinds == "height_m", rng.uniform(1.4, 1.9, k), rng.uniform(-3, 200, k))}),
        "lifestyle": pd.DataFrame({**dated(600), "kind": "smoking", "value": rng.choice(["never", "ex", "current"], 600)}),
        "encounters": pd.DataFrame({"research_id": rng.choice(ids, 400), "encounter_id": [f"e{i}" for i in range(400)],
                                    "encounter_type": "ed_attendance", "admission_method": None, "spell_start_date": pd.NaT,
                                    "event_date": INDEX + pd.to_timedelta(rng.integers(-800, 800, 400), unit="D"),
                                    "diagnosis_code": rng.choice(["W19", "S72.0", "R55"], 400), "diagnosis_position": 1,
                                    "code_system": "ICD-10"}),
    }
    return tables


PREDICTORS = ["predictor_max_record_date", *SPEC.predictor_names()]


@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("rules", [DerivationRules(), DerivationRules(source_availability={t: "unknown" for t in PREDICTOR_TABLES}),
                                   DerivationRules(apply_time_window_rules=False, apply_measurement_rules=False)],
                         ids=["primary", "unknown_availability", "ever_recorded"])
def test_future_records_cannot_change_predictors(seed: int, rules: DerivationRules) -> None:
    tables = _random_tables(seed)
    index = pd.DataFrame({"research_id": tables["persons"]["research_id"], "index_date": INDEX})
    full, _ = derive_modeling_frame(tables, index, SPEC, rules)
    assert (full["predictor_max_record_date"].isna() | (full["predictor_max_record_date"] < full["index_date"])).all()

    past_only = dict(tables)
    for name in PREDICTOR_TABLES:
        t = tables[name]
        past_only[name] = t.loc[(t["record_date"] < INDEX) & (t["available_date"] < INDEX)]
    past_only["encounters"] = tables["encounters"].iloc[0:0]
    reduced, _ = derive_modeling_frame(past_only, index, SPEC, rules)
    pd.testing.assert_frame_equal(full[PREDICTORS], reduced[PREDICTORS])
    assert full["outcome_12m"].sum() > 0 and reduced["outcome_12m"].sum() == 0


@pytest.mark.parametrize("rules", [DerivationRules(), DerivationRules(source_availability={t: "unknown" for t in PREDICTOR_TABLES})],
                         ids=["primary", "unknown_availability"])
def test_rows_for_several_index_dates_do_not_share_records(rules: DerivationRules) -> None:
    """A person's records between two index dates must not reach the earlier row (per-row pairing, D-00)."""
    tables = _random_tables(4)
    ids = tables["persons"]["research_id"]
    dates = [INDEX - pd.Timedelta(days=200), INDEX, INDEX + pd.DateOffset(years=1)]
    joint_index = pd.DataFrame({"research_id": [r for _ in dates for r in ids], "index_date": [x for x in dates for _ in ids]})
    joint, _ = derive_modeling_frame(tables, joint_index, SPEC, rules)
    separate = pd.concat([derive_modeling_frame(tables, pd.DataFrame({"research_id": ids, "index_date": x}), SPEC, rules)[0]
                          for x in dates], ignore_index=True)
    pd.testing.assert_frame_equal(joint, separate)
    later = joint.loc[joint["index_date"] == dates[-1], PREDICTORS].reset_index(drop=True)
    earlier = joint.loc[joint["index_date"] == dates[0], PREDICTORS].reset_index(drop=True)
    assert not later.equals(earlier)  # the dates really see different records


@pytest.mark.parametrize("seed", [5, 6])
def test_predictor_max_record_date_before_index_on_synthetic_tables(seed: int) -> None:
    tables = generate_synthetic_event_tables(250, seed=seed, spec=SPEC)
    for rules in (DerivationRules(), DerivationRules(source_availability={t: "unknown" for t in PREDICTOR_TABLES})):
        frame, _ = derive_modeling_frame(tables, tables["index_table"], SPEC, rules)
        pm = frame["predictor_max_record_date"]
        assert pm.notna().mean() > 0.9
        assert (pm.isna() | (pm < frame["index_date"])).all()
