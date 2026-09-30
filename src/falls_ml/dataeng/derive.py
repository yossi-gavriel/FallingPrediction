"""Reference derivation of the modelling dataset from canonical event tables.

Implements spec D-00 (timing), D-02 (BMI), D-03 (polypharmacy), D-04 (smoking), D-05 (alcohol),
D-06 (decimal age), D-08 (binary look-back, age-at-record, resolution and measurement rules),
D-09 (outcome and pre-modelling audit), D-22 (source classes), M-01, M-02, M-03 and M-14.
This is the data-engineering layer: the ML layer never imports it (architecture §1).

Canonical event tables (dates are datetime64 and day-precision; time components are dropped)
--------------------------------------------------------------------------------------------
``persons``        research_id, date_of_birth, sex, practice_id, deprivation_group, site_id, death_date,
                   membership_start, membership_end.
                   One row per membership interval; intervals must not overlap. date_of_birth, sex and
                   death_date are constant per person. Cluster columns are taken from the interval
                   covering the index date (M-01).
``conditions``     research_id, feature, record_date, available_date, source_class.
                   ``feature`` is a binary feature name: code-to-concept mapping happens upstream (M-05).
                   ``source_class`` is a D-22 class; ``registry`` records always need an available_date (M-03).
``prescriptions``  research_id, bnf_paragraph (6 digits), bnf_chapter (int, = first two digits), is_drug (bool),
                   record_date, available_date.
``measurements``   research_id, kind, value (finite float), record_date, available_date.
                   kind in {bmi, weight_kg, height_m, sbp, dbp, hb_g_dl, egfr, acr_mg_mmol, tsh_mu_l,
                   dexa_tscore, abpi, barthel, sixcit}.
``lifestyle``      research_id, kind, value, record_date, available_date.
                   kind smoking: value in {never, ex, current}; kind alcohol: value is an alcohol level;
                   kind alcohol_units_week: value is a number >= 0.
``encounters``     research_id, encounter_id, encounter_type {ed_attendance, hospital_admission, other},
                   admission_method {emergency, urgent, via_ed, elective, other, null = unknown},
                   spell_start_date (null = event_date), event_date, diagnosis_code,
                   diagnosis_position (1 = principal), code_system {ICD-10, ICD-10-CM, ICD-9-CM}.
                   One row per diagnosis. Encounters feed the outcome only, never predictors (D-00).

Timing (D-00): prediction at the start of the index day. A predictor record is usable only if
record_date < index_date and available_date < index_date; for a source whose availability is
``unknown`` it must instead satisfy record_date < index_date - lag (and available_date < index_date
when recorded). Outcome events satisfy index_date <= event_date <= spec.outcome.window_end(index_date).

``predictor_max_record_date`` is the latest record_date among records that entered a derivation, i.e.
passed the usability filter and the component's own selection/window filter; NaT if there are none.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from falls_ml.errors import ConfigError, DatasetValidationError, LeakageError
from falls_ml.features.spec import FeatureSpec
from falls_ml.logging_utils import get_logger

log = get_logger(__name__)

DATE_DTYPE = "datetime64[ns]"
REPO_ROOT = Path(__file__).resolve().parents[3]

PREDICTOR_TABLES = ("conditions", "prescriptions", "measurements", "lifestyle")
TABLE_COLUMNS: dict[str, tuple[str, ...]] = {
    "persons": ("research_id", "date_of_birth", "sex", "practice_id", "deprivation_group", "site_id", "death_date",
                "membership_start", "membership_end"),
    "conditions": ("research_id", "feature", "record_date", "available_date", "source_class"),
    "prescriptions": ("research_id", "bnf_paragraph", "bnf_chapter", "is_drug", "record_date", "available_date"),
    "measurements": ("research_id", "kind", "value", "record_date", "available_date"),
    "lifestyle": ("research_id", "kind", "value", "record_date", "available_date"),
    "encounters": ("research_id", "encounter_id", "encounter_type", "admission_method", "spell_start_date",
                   "event_date", "diagnosis_code", "diagnosis_position", "code_system"),
}
DATE_COLUMNS: dict[str, tuple[str, ...]] = {
    "persons": ("date_of_birth", "death_date", "membership_start", "membership_end"),
    **{t: ("record_date", "available_date") for t in PREDICTOR_TABLES},
    "encounters": ("spell_start_date", "event_date"),
}
SOURCE_CLASSES = ("community_diagnosis", "hospital_diagnosis", "prescription", "dispensing", "measurement", "lab",
                  "assessment", "registry")  # D-22
MEASUREMENT_KINDS = ("bmi", "weight_kg", "height_m", "sbp", "dbp", "hb_g_dl", "egfr", "acr_mg_mmol", "tsh_mu_l",
                     "dexa_tscore", "abpi", "barthel", "sixcit")
LIFESTYLE_KINDS = ("smoking", "alcohol", "alcohol_units_week")
ENCOUNTER_TYPES = ("ed_attendance", "hospital_admission", "other")
ADMISSION_METHODS = ("emergency", "urgent", "via_ed", "elective", "other")
NON_ELECTIVE_METHODS = ("emergency", "urgent", "via_ed")
CODE_SYSTEMS = ("ICD-10", "ICD-10-CM", "ICD-9-CM")
SEXES = ("female", "male")
SMOKING_LEVELS = ("never", "ex", "current")  # ascending same-day precedence
ALCOHOL_PRECEDENCE = ("zero", "lower_risk", "previous_higher_risk_or_harmful", "higher_risk", "harmful")  # ascending (D-05)

AGE, SEX, POLYPHARMACY = "age_years", "sex", "polypharmacy_count_120d"
BMI, SMOKING, ALCOHOL = "bmi_value", "smoking_status", "alcohol_category"
METADATA_COLUMNS = ("practice_id", "deprivation_group", "site_id", "death_date", "followup_end_date")
ANAEMIA = "anaemia_and_haematinic_deficiency"
HB_LOW_BELOW = {"male": 13.0, "female": 11.5}  # g/dL (D-08)

# D-08 non-code rules: feature -> (minimum qualifying readings per criterion, criteria). Systolic and diastolic
# criteria are counted separately ("not mixed"). Anaemia is sex-specific and handled explicitly.
MEASUREMENT_RULES: dict[str, tuple[int, tuple[tuple[str, str, float], ...]]] = {
    "hypertension": (3, (("sbp", ">=", 140.0), ("dbp", ">=", 90.0))),
    "hypotension_or_syncope": (3, (("sbp", "<", 90.0), ("dbp", "<", 60.0))),
    "activity_limitation": (1, (("barthel", "<=", 18.0),)),
    "cognitive_impairment": (1, (("sixcit", ">=", 8.0),)),
    "osteoporosis": (1, (("dexa_tscore", "<", -2.5),)),
    "peripheral_vascular_disease": (1, (("abpi", "<", 0.95),)),
    "chronic_kidney_disease": (1, (("egfr", "<", 60.0), ("acr_mg_mmol", ">", 3.0))),
    "thyroid_problems": (1, (("tsh_mu_l", "<", 0.36), ("tsh_mu_l", ">", 5.5))),
}
MEASUREMENT_RULE_FEATURES = (*MEASUREMENT_RULES, ANAEMIA)
_OPS = {"<": np.less, "<=": np.less_equal, ">": np.greater, ">=": np.greater_equal}

AUDIT_COLUMNS = ("code_system", "efalls_code", "code_block", "kind", "diagnosis_position", "encounter_type",
                 "admission_method", "status", "n_records")
ELIGIBILITY_REASONS = ("missing_person_record", "missing_date_of_birth", "invalid_sex", "died_before_index",
                       "not_member_at_index", "age_below_minimum")


# ---------------------------------------------------------------------------------------------- rules / log
@dataclass(frozen=True)
class DerivationRules:
    """Primary derivation rules with the documented sensitivity switches (D-00, D-02, D-03, D-05, D-08, D-09)."""

    polypharmacy_window_days: int = 120
    excluded_bnf_chapters: tuple[int, ...] = (20, 21, 22, 23)
    bmi_window_years: int = 5
    bmi_valid_range: tuple[float, float] = (10.0, 80.0)
    weight_height_max_gap_days: int = 30
    alcohol_window_years: int = 5
    outcome_admission_rule: str = "non_elective"  # 'non_elective' | 'any'
    unknown_admission_method: str = "exclude"  # 'exclude' | 'include' | 'error'
    outcome_position_rule: str = "any"  # 'any' | 'principal_fracture_or_any_external'
    exclude_spells_started_before_index: bool = True
    apply_time_window_rules: bool = True
    apply_measurement_rules: bool = True
    source_availability: Mapping[str, str] = field(default_factory=lambda: {t: "declared" for t in PREDICTOR_TABLES})
    unknown_availability_lag_days: int = 30
    icd9cm_candidates_path: str = "configs/mappings/outcome_code_candidates.yaml"

    def __post_init__(self) -> None:
        problems = []
        choices = {"outcome_admission_rule": ("non_elective", "any"),
                   "unknown_admission_method": ("exclude", "include", "error"),
                   "outcome_position_rule": ("any", "principal_fracture_or_any_external")}
        for name, allowed in choices.items():
            if getattr(self, name) not in allowed:
                problems.append(f"{name}={getattr(self, name)!r} not in {allowed}")
        for name in ("polypharmacy_window_days", "bmi_window_years", "alcohol_window_years"):
            if int(getattr(self, name)) < 1:
                problems.append(f"{name} must be >= 1")
        if self.weight_height_max_gap_days < 0 or self.unknown_availability_lag_days < 0:
            problems.append("day gaps/lags must be >= 0")
        lo, hi = self.bmi_valid_range
        if not lo < hi:
            problems.append(f"bmi_valid_range {self.bmi_valid_range} is empty")
        unknown_tables = sorted(set(self.source_availability) - set(PREDICTOR_TABLES))
        if unknown_tables:
            problems.append(f"source_availability has unknown tables {unknown_tables}")
        bad = {k: v for k, v in self.source_availability.items() if v not in ("declared", "unknown")}
        if bad:
            problems.append(f"source_availability values must be 'declared' or 'unknown': {bad}")
        if problems:
            raise ConfigError("Invalid DerivationRules: " + "; ".join(problems))

    def availability(self, table: str) -> str:
        """Availability semantics of a predictor table; undeclared tables default to 'declared' (D-00, M-03)."""
        return self.source_availability.get(table, "declared")


@dataclass(frozen=True)
class DerivationLog:
    """Counts of everything excluded or flagged during derivation (never silent) plus the D-09 outcome audit.

    ``exclusions``: index rows excluded by first failing eligibility reason. ``record_exclusions``: (index row,
    record) pairs not used, keyed ``<table>.<reason>`` (D-00 timing/availability, lag buffer, codes outside the
    spec, malformed codes). ``invalid_bmi_values``: (index row, value) pairs in the BMI window outside
    ``bmi_valid_range``, recorded or computed (D-02). ``outcome_audit``: in-window outcome-coded diagnoses by code
    block, position, encounter type, admission method, code system and qualifying status (pre-modelling audit, D-09).
    """

    exclusions: dict[str, int]
    n_input: int
    n_output: int
    invalid_bmi_values: int
    unknown_admission_method_count: int
    outcome_audit: pd.DataFrame
    record_exclusions: dict[str, int] = field(default_factory=dict)


def empty_event_tables() -> dict[str, pd.DataFrame]:
    """Empty canonical event tables with the declared columns (convenience for callers and tests)."""
    return {name: pd.DataFrame({c: pd.Series(dtype=object) for c in cols}) for name, cols in TABLE_COLUMNS.items()}


# ---------------------------------------------------------------------------------------------- public API
def derive_modeling_frame(tables: Mapping[str, pd.DataFrame], index_table: pd.DataFrame, spec: FeatureSpec,
                          rules: DerivationRules | None = None) -> tuple[pd.DataFrame, DerivationLog]:
    """Derive one modelling row per eligible (research_id, index_date) from canonical event tables.

    Returns every modelling-dataset column except the version columns, plus a :class:`DerivationLog`.
    Keys of ``tables`` other than the six canonical tables are ignored.
    """
    rules = rules or DerivationRules()
    binary_rules = _binary_rules(spec, rules)
    _check_supported(spec)
    t = _prepare_tables(tables, rules)
    idx = _prepare_index(index_table)
    rows, exclusions = _eligible_rows(idx, t["persons"], spec)
    n = len(rows)
    counts: Counter[str] = Counter()
    used: list[pd.DataFrame] = []

    usable = {name: _usable_records(rows, t[name], name, rules, counts) for name in PREDICTOR_TABLES}
    binaries = _derive_binaries(rows, usable["conditions"], usable["measurements"], binary_rules, rules, counts, used)
    poly = _derive_polypharmacy(n, usable["prescriptions"], rules, counts, used)
    bmi, invalid_bmi = _derive_bmi(n, usable["measurements"], rules, used)
    smoking = _derive_smoking(n, usable["lifestyle"], used)
    alcohol = _derive_alcohol(n, usable["lifestyle"], rules, used)
    outcome, first_date, audit, n_unknown = _derive_outcome(rows, t["encounters"], spec, rules, counts)

    pmax = _max_dates(n, used)
    if (pmax >= rows["index_date"]).any():  # structural guarantee; checked defensively (D-00)
        raise LeakageError("predictor_max_record_date on or after index_date: derivation used a post-index record")

    window_end = spec.outcome.window_end(rows["index_date"]).astype(DATE_DTYPE)
    followup_end = pd.concat([rows["death_date"], rows["membership_end"], window_end], axis=1).min(axis=1)
    derived = {
        AGE: rows[AGE].astype("float64"), SEX: rows["sex"].astype("str"), POLYPHARMACY: poly, BMI: bmi,
        SMOKING: smoking, ALCOHOL: alcohol, **{name: binaries[name] for name in binaries.columns},
    }
    metadata = {
        "practice_id": rows["practice_id"].astype("str"), "deprivation_group": rows["deprivation_group"].astype("str"),
        "site_id": rows["site_id"].astype("str"), "death_date": rows["death_date"],
        "followup_end_date": followup_end.astype(DATE_DTYPE),
    }
    columns: dict[str, pd.Series] = {
        "research_id": rows["research_id"].astype("str"), spec.index_column: rows["index_date"],
        "predictor_max_record_date": pmax, spec.outcome.name: outcome, "outcome_first_event_date": first_date,
    }
    columns.update({f.name: derived[f.name] for f in spec.features})
    columns.update({c: metadata[c] for c in spec.metadata_columns})
    frame = pd.DataFrame({k: pd.Series(v).reset_index(drop=True) for k, v in columns.items()})

    dlog = DerivationLog(exclusions=exclusions, n_input=len(idx), n_output=n, invalid_bmi_values=invalid_bmi,
                         unknown_admission_method_count=n_unknown, outcome_audit=audit,
                         record_exclusions=dict(sorted(counts.items())))
    log.info("modeling_frame_derived", extra_fields={
        "n_input": dlog.n_input, "n_output": dlog.n_output, "exclusions": exclusions,
        "invalid_bmi_values": invalid_bmi, "unknown_admission_method_count": n_unknown,
        "record_exclusions": dlog.record_exclusions, "n_outcome_events": int(outcome.sum())})
    return frame, dlog


# ---------------------------------------------------------------------------------------------- validation
def _check_supported(spec: FeatureSpec) -> None:
    known = {AGE, SEX, POLYPHARMACY, BMI, SMOKING, ALCOHOL}
    unsupported = [f.name for f in spec.features if f.name not in known and not f.is_binary]
    if unsupported:
        raise ConfigError(f"derive_modeling_frame cannot derive non-binary features {unsupported}; derive them upstream")
    missing = sorted(known - set(spec.predictor_names()))
    if missing:
        raise ConfigError(f"Feature spec lacks the eFalls core predictors {missing}")
    if set(spec.get(SEX).levels) != set(SEXES):
        raise ConfigError(f"sex levels {spec.get(SEX).levels} differ from {SEXES}")
    if not set(spec.get(SMOKING).levels) <= set(SMOKING_LEVELS):
        raise ConfigError(f"smoking levels {spec.get(SMOKING).levels} not derivable (D-04)")
    if set(spec.get(ALCOHOL).levels) != set(ALCOHOL_PRECEDENCE):
        raise ConfigError(f"alcohol levels {spec.get(ALCOHOL).levels} differ from {ALCOHOL_PRECEDENCE} (D-05)")
    extra_meta = sorted(set(spec.metadata_columns) - set(METADATA_COLUMNS))
    if extra_meta:
        raise ConfigError(f"derive_modeling_frame cannot populate metadata columns {extra_meta}")


def _to_dates(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s).dt.normalize().astype(DATE_DTYPE)


def _bad_values(s: pd.Series, allowed: tuple[str, ...], *, allow_null: bool = False) -> list[str]:
    bad = sorted({str(v) for v in s.dropna()} - set(allowed))
    return bad + (["<null>"] if not allow_null and s.isna().any() else [])


def _prepare_tables(tables: Mapping[str, pd.DataFrame], rules: DerivationRules) -> dict[str, pd.DataFrame]:
    problems: list[str] = []
    out: dict[str, pd.DataFrame] = {}
    for name, cols in TABLE_COLUMNS.items():
        if name not in tables:
            problems.append(f"missing table {name!r}")
            continue
        missing = [c for c in cols if c not in tables[name].columns]
        if missing:
            problems.append(f"{name}: missing columns {missing}")
            continue
        df = tables[name].loc[:, list(cols)].reset_index(drop=True)
        if df["research_id"].isna().any():
            problems.append(f"{name}: {int(df['research_id'].isna().sum())} null research_id")
        df["research_id"] = df["research_id"].astype("str")
        for c in DATE_COLUMNS[name]:
            try:
                df[c] = _to_dates(df[c])
            except (ValueError, TypeError) as exc:
                problems.append(f"{name}.{c}: not parseable as dates ({exc})")
        out[name] = df
    if problems:
        raise DatasetValidationError("Event tables are not in canonical form", problems)

    for name in PREDICTOR_TABLES:
        df = out[name]
        if df["record_date"].isna().any():
            problems.append(f"{name}: {int(df['record_date'].isna().sum())} null record_date")
        null_avail = df["available_date"].isna()
        if rules.availability(name) == "declared" and null_avail.any():
            problems.append(f"{name}: {int(null_avail.sum())} null available_date in a table declared with availability "
                            "semantics (declare it 'unknown' to apply the lag buffer, D-00)")
        if (df["available_date"] < df["record_date"]).any():
            problems.append(f"{name}: available_date before record_date in {int((df['available_date'] < df['record_date']).sum())} rows")

    cond = out["conditions"]
    if cond["feature"].isna().any():
        problems.append("conditions: null feature")
    cond["feature"] = cond["feature"].astype("str")
    bad = _bad_values(cond["source_class"], SOURCE_CLASSES)
    if bad:
        problems.append(f"conditions: source_class values {bad} not in D-22 classes {SOURCE_CLASSES}")
    if ((cond["source_class"] == "registry") & cond["available_date"].isna()).any():
        problems.append("conditions: registry records need a registry-entry available_date (M-03)")

    rx = out["prescriptions"]
    para = rx["bnf_paragraph"].astype("str")
    bad_para = ~para.str.fullmatch(r"\d{6}")
    chapter = pd.to_numeric(rx["bnf_chapter"], errors="coerce")
    if bad_para.any():
        problems.append(f"prescriptions: {int(bad_para.sum())} bnf_paragraph values are not 6-digit codes")
    if chapter.isna().any() or (chapter % 1 != 0).any():
        problems.append("prescriptions: bnf_chapter must be integer and non-null")
    elif (~bad_para & (para.str[:2].where(~bad_para, "0").astype("int64") != chapter)).any():
        problems.append("prescriptions: bnf_chapter disagrees with the first two digits of bnf_paragraph")
    if rx["is_drug"].isna().any() or not set(rx["is_drug"].dropna().unique()) <= {True, False}:
        problems.append("prescriptions: is_drug must be boolean and non-null")
    rx["bnf_paragraph"] = para
    rx["bnf_chapter"] = chapter.fillna(-1).astype("int64")
    rx["is_drug"] = rx["is_drug"].fillna(False).astype(bool)

    meas = out["measurements"]
    bad_kind = _bad_values(meas["kind"], MEASUREMENT_KINDS)
    if bad_kind:
        problems.append(f"measurements: unknown kinds {bad_kind}")
    value = pd.to_numeric(meas["value"], errors="coerce")
    if value.isna().any() or not np.isfinite(value.to_numpy(dtype="float64")).all():
        problems.append("measurements: value must be a finite number in every row")
    meas["kind"] = meas["kind"].astype("str")
    meas["value"] = value.astype("float64")

    life = out["lifestyle"]
    bad_kind = _bad_values(life["kind"], LIFESTYLE_KINDS)
    if bad_kind:
        problems.append(f"lifestyle: unknown kinds {bad_kind}")
    life["kind"] = life["kind"].astype("str")
    text = life["value"].astype("str")
    units = pd.to_numeric(life["value"].where(life["kind"] == "alcohol_units_week"), errors="coerce")
    for kind, allowed in (("smoking", SMOKING_LEVELS), ("alcohol", ALCOHOL_PRECEDENCE)):
        bad_values = _bad_values(text[life["kind"] == kind], allowed)
        if bad_values:
            problems.append(f"lifestyle: {kind} values {bad_values[:10]} not in {allowed}")
    is_units = life["kind"] == "alcohol_units_week"
    if (is_units & ~(units >= 0)).any() or not np.isfinite(units[is_units].to_numpy(dtype="float64")).all():
        problems.append("lifestyle: alcohol_units_week values must be finite numbers >= 0")
    life["value"] = text
    life["units"] = units.astype("float64")

    enc = out["encounters"]
    for col, allowed in (("encounter_type", ENCOUNTER_TYPES), ("code_system", CODE_SYSTEMS)):
        bad_values = _bad_values(enc[col], allowed)
        if bad_values:
            problems.append(f"encounters: {col} values {bad_values} not in {allowed}")
    bad_values = _bad_values(enc["admission_method"], ADMISSION_METHODS, allow_null=True)
    if bad_values:
        problems.append(f"encounters: admission_method values {bad_values} not in {ADMISSION_METHODS} or null")
    position = pd.to_numeric(enc["diagnosis_position"], errors="coerce")
    if position.isna().any() or (position < 1).any() or (position % 1 != 0).any():
        problems.append("encounters: diagnosis_position must be an integer >= 1")
    if enc["diagnosis_code"].isna().any() or enc["event_date"].isna().any() or enc["encounter_id"].isna().any():
        problems.append("encounters: diagnosis_code, event_date and encounter_id must be non-null")
    if (enc["spell_start_date"] > enc["event_date"]).any():
        problems.append("encounters: spell_start_date after event_date")
    for col in ("encounter_type", "code_system", "diagnosis_code", "encounter_id"):
        enc[col] = enc[col].astype("str")
    enc["admission_method"] = enc["admission_method"].astype("str")
    enc["diagnosis_position"] = position.fillna(0).astype("int64")

    persons = out["persons"]
    for col in ("date_of_birth", "sex", "death_date"):
        varying = persons.groupby("research_id")[col].nunique(dropna=False)
        if (varying > 1).any():
            problems.append(f"persons: {col} varies across membership rows for {int((varying > 1).sum())} persons")
    if (persons["membership_end"] < persons["membership_start"]).any():
        problems.append("persons: membership_end before membership_start")
    ordered = persons.sort_values(["research_id", "membership_start"], kind="stable")
    prev_end = ordered.groupby("research_id")["membership_end"].shift()
    not_first = ordered.groupby("research_id").cumcount() > 0
    if (not_first & (prev_end.isna() | (ordered["membership_start"] <= prev_end))).any():
        problems.append("persons: overlapping membership intervals")
    persons["sex"] = persons["sex"].astype("str")
    if problems:
        raise DatasetValidationError("Event tables violate the canonical event-table contract", problems)
    return out


def _prepare_index(index_table: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in ("research_id", "index_date") if c not in index_table.columns]
    if missing:
        raise DatasetValidationError(f"index table missing columns {missing}")
    idx = index_table.loc[:, ["research_id", "index_date"]].reset_index(drop=True)
    if idx.isna().any().any():
        raise DatasetValidationError("index table contains null research_id or index_date")
    idx["research_id"] = idx["research_id"].astype("str")
    idx["index_date"] = _to_dates(idx["index_date"])
    n_dup = int(idx.duplicated().sum())
    if n_dup:
        raise DatasetValidationError(f"index table has {n_dup} duplicated (research_id, index_date) pairs")
    return idx


# ---------------------------------------------------------------------------------------------- eligibility
def _eligible_rows(idx: pd.DataFrame, persons: pd.DataFrame, spec: FeatureSpec) -> tuple[pd.DataFrame, dict[str, int]]:
    """Eligibility at the start of the index day (spec §4, D-06, M-01); first failing reason is counted."""
    m = idx.assign(_input_row=np.arange(len(idx))).merge(persons, on="research_id", how="left", indicator=True)
    m["_covers"] = (m["membership_start"] <= m["index_date"]) & (
        m["membership_end"].isna() | (m["membership_end"] >= m["index_date"]))
    m = m.sort_values(["_input_row", "_covers"], ascending=[True, False], kind="stable")
    m = m.drop_duplicates("_input_row", keep="first").sort_values("_input_row").reset_index(drop=True)
    age = (m["index_date"] - m["date_of_birth"]).dt.days / 365.25
    reasons = {
        "missing_person_record": (m["_merge"] == "left_only").to_numpy(),
        "missing_date_of_birth": m["date_of_birth"].isna().to_numpy(),
        "invalid_sex": (~m["sex"].isin(SEXES)).to_numpy(),
        "died_before_index": (m["death_date"].notna() & (m["death_date"] < m["index_date"])).to_numpy(),
        "not_member_at_index": (~m["_covers"].astype(bool)).to_numpy(),
        "age_below_minimum": (~(age >= spec.age_min)).to_numpy(),
    }
    excluded = np.zeros(len(m), dtype=bool)
    exclusions: dict[str, int] = {}
    for reason in ELIGIBILITY_REASONS:
        new = reasons[reason] & ~excluded
        exclusions[reason] = int(new.sum())
        excluded |= new
    rows = m.loc[~excluded, ["research_id", "index_date", "date_of_birth", "sex", "practice_id", "deprivation_group",
                             "site_id", "death_date", "membership_end"]].reset_index(drop=True)
    rows[AGE] = age[~excluded].to_numpy(dtype="float64")
    rows["_row"] = np.arange(len(rows))
    return rows, exclusions


def _usable_records(rows: pd.DataFrame, table: pd.DataFrame, name: str, rules: DerivationRules,
                    counts: Counter[str]) -> pd.DataFrame:
    """Pair records with index rows and keep those usable at the start of the index day (D-00, M-03)."""
    pairs = rows[["_row", "research_id", "index_date", "date_of_birth", "sex"]].merge(table, on="research_id")
    index_date = pairs["index_date"]
    on_after = (pairs["record_date"] >= index_date).to_numpy()
    late = (pairs["available_date"].notna() & (pairs["available_date"] >= index_date)).to_numpy() & ~on_after
    if rules.availability(name) == "unknown":
        cutoff = index_date - pd.Timedelta(days=rules.unknown_availability_lag_days)
        lag = (pairs["record_date"] >= cutoff).to_numpy() & ~on_after
        late &= ~lag
        counts[f"{name}.within_unknown_availability_lag"] += int(lag.sum())
    else:
        lag = np.zeros(len(pairs), dtype=bool)
    counts[f"{name}.record_on_or_after_index"] += int(on_after.sum())
    counts[f"{name}.available_on_or_after_index"] += int(late.sum())
    return pairs.loc[~(on_after | late | lag)].reset_index(drop=True)


def _window_start(index_dates: pd.Series, *, years: int | None = None, days: int | None = None) -> pd.Series:
    if years is not None:
        return index_dates - pd.DateOffset(years=int(years))
    return index_dates - pd.Timedelta(days=int(days or 0))


def _max_dates(n: int, used: list[pd.DataFrame]) -> pd.Series:
    frames = [u[["_row", "record_date"]] for u in used if len(u)]
    if not frames:
        return pd.Series(pd.NaT, index=range(n), dtype=DATE_DTYPE)
    allu = pd.concat(frames, ignore_index=True)
    return allu.groupby("_row")["record_date"].max().reindex(range(n)).astype(DATE_DTYPE)


# ---------------------------------------------------------------------------------------------- binaries (D-08)
@dataclass(frozen=True)
class _BinaryRule:
    name: str
    lookback_years: int | None
    lookback_days: int | None
    age_at_record_min: float | None
    resolved_by: tuple[str, ...]
    measurement_rule: bool

    def window_mask(self, df: pd.DataFrame) -> np.ndarray:
        mask = np.ones(len(df), dtype=bool)
        if self.lookback_years is not None or self.lookback_days is not None:
            start = _window_start(df["index_date"], years=self.lookback_years, days=self.lookback_days)
            mask &= (df["record_date"] >= start).to_numpy()
        if self.age_at_record_min is not None:
            age_at_record = (df["record_date"] - df["date_of_birth"]).dt.days / 365.25
            mask &= (age_at_record >= self.age_at_record_min).to_numpy()
        return mask


def _binary_rules(spec: FeatureSpec, rules: DerivationRules) -> dict[str, _BinaryRule]:
    names = spec.binary_names()
    out: dict[str, _BinaryRule] = {}
    for name in names:
        tw = dict(spec.get(name).raw.get("time_window") or {})
        window = tw.get("window") or (tw if "type" in tw else {"type": "complete_history"})
        years = days = None
        if window["type"] == "lookback_years":
            years = int(window["years"])
        elif window["type"] == "lookback_days":
            days = int(window["days"])
        elif window["type"] != "complete_history":
            raise ConfigError(f"{name}: unsupported time window {window}")
        resolved_by = tuple(tw.get("resolved_by") or ())
        unknown = sorted(set(resolved_by) - set(names))
        if unknown:
            raise ConfigError(f"{name}: resolving features {unknown} are not binary features of the spec")
        age_min = tw.get("age_at_record_min")
        has_rule = tw.get("non_code_rule") is not None and rules.apply_measurement_rules
        if has_rule and name not in MEASUREMENT_RULE_FEATURES:
            raise ConfigError(f"{name}: non_code_rule declared but no measurement rule is implemented (D-08)")
        if not rules.apply_time_window_rules:
            years = days = age_min = None
            resolved_by = ()
        out[name] = _BinaryRule(name, years, days, None if age_min is None else float(age_min), resolved_by, has_rule)
    return out


def _derive_binaries(rows: pd.DataFrame, conditions: pd.DataFrame, measurements: pd.DataFrame,
                     brules: dict[str, _BinaryRule], rules: DerivationRules, counts: Counter[str],
                     used: list[pd.DataFrame]) -> pd.DataFrame:
    n = len(rows)
    names = list(brules)
    in_spec = conditions["feature"].isin(names).to_numpy()
    counts["conditions.feature_not_in_spec"] += int((~in_spec).sum())
    evidence: list[pd.DataFrame] = []
    for name, group in conditions.loc[in_spec].groupby("feature", sort=True):
        evidence.append(group.loc[brules[str(name)].window_mask(group), ["_row", "feature", "record_date"]])

    for name, brule in brules.items():
        if not brule.measurement_rule:
            continue
        kinds = {"hb_g_dl"} if name == ANAEMIA else {c[0] for c in MEASUREMENT_RULES[name][1]}
        readings = measurements.loc[measurements["kind"].isin(kinds).to_numpy()]
        readings = readings.loc[brule.window_mask(readings)]
        used.append(readings)
        if name == ANAEMIA:
            # Sex-specific low Hb; a strictly later normal Hb resolves the Hb criterion only (D-08: "code OR low Hb,
            # resolved by a later normal Hb"). Anaemia/haematinic-deficiency codes are never removed by an Hb value.
            low = (readings["value"] < readings["sex"].map(HB_LOW_BELOW)).to_numpy()
            low_hb = readings.loc[low, ["_row", "record_date"]]
            if rules.apply_time_window_rules:
                last_normal = readings.loc[~low].groupby("_row")["record_date"].max()
                last_low = low_hb.groupby("_row")["record_date"].max()
                resolved = last_low.index[(last_normal.reindex(last_low.index) > last_low).to_numpy()]
                low_hb = low_hb.loc[~low_hb["_row"].isin(resolved).to_numpy()]
            evidence.append(low_hb.assign(feature=name))
            continue
        min_readings, criteria = MEASUREMENT_RULES[name]
        for kind, op, threshold in criteria:  # criteria are counted separately (systolic/diastolic not mixed)
            sel = readings.loc[(readings["kind"] == kind).to_numpy() & _OPS[op](readings["value"].to_numpy(), threshold)]
            if min_readings > 1:
                sel = sel.loc[(sel.groupby("_row")["record_date"].transform("size") >= min_readings).to_numpy()]
            evidence.append(sel[["_row", "record_date"]].assign(feature=name))

    ev = pd.concat(evidence, ignore_index=True) if evidence else pd.DataFrame(
        {"_row": pd.Series(dtype="int64"), "feature": pd.Series(dtype="str"), "record_date": pd.Series(dtype=DATE_DTYPE)})
    used.append(ev[["_row", "record_date"]])
    latest = (ev.groupby(["_row", "feature"])["record_date"].max().unstack("feature")
              .reindex(index=range(n), columns=names).astype(DATE_DTYPE))
    values = latest.notna()
    for name, brule in brules.items():
        if brule.resolved_by:  # resolving record dated on or after the latest resolved record (D-08)
            resolver = latest[list(brule.resolved_by)].max(axis=1)
            values[name] = values[name] & ~(resolver >= latest[name])
    return values.astype("int8").reset_index(drop=True)


# ---------------------------------------------------------------------------------------------- other predictors
def _derive_polypharmacy(n: int, rx: pd.DataFrame, rules: DerivationRules, counts: Counter[str],
                         used: list[pd.DataFrame]) -> pd.Series:
    """Distinct BNF paragraphs, index - W days <= record_date <= index - 1 day, drugs only (D-03)."""
    in_window = (rx["record_date"] >= _window_start(rx["index_date"], days=rules.polypharmacy_window_days)).to_numpy()
    drug = (rx["is_drug"] & ~rx["bnf_chapter"].isin(rules.excluded_bnf_chapters)).to_numpy()
    counts["prescriptions.non_drug_or_excluded_chapter_in_window"] += int((in_window & ~drug).sum())
    sel = rx.loc[in_window & drug]
    used.append(sel)
    return sel.groupby("_row")["bnf_paragraph"].nunique().reindex(range(n), fill_value=0).astype("int64")


def _latest_day_mean(df: pd.DataFrame, value: str, n: int) -> pd.Series:
    latest = df["record_date"] == df.groupby("_row")["record_date"].transform("max")
    return df.loc[latest].groupby("_row")[value].mean().reindex(range(n))


def _derive_bmi(n: int, meas: pd.DataFrame, rules: DerivationRules, used: list[pd.DataFrame]) -> tuple[pd.Series, int]:
    """Most recent valid recorded BMI in window; else most recent weight/height pair within the gap (D-02)."""
    lo, hi = rules.bmi_valid_range
    win = meas.loc[(meas["record_date"] >= _window_start(meas["index_date"], years=rules.bmi_window_years)).to_numpy()
                   & meas["kind"].isin(("bmi", "weight_kg", "height_m")).to_numpy()]
    used.append(win)
    recorded = win.loc[win["kind"] == "bmi"]
    valid = recorded["value"].between(lo, hi)
    invalid = int((~valid).sum())
    bmi = _latest_day_mean(recorded.loc[valid], "value", n)

    need = set(np.flatnonzero(bmi.isna().to_numpy()))
    weights = win.loc[(win["kind"] == "weight_kg") & win["_row"].isin(need)].reset_index(drop=True)
    heights = win.loc[(win["kind"] == "height_m") & win["_row"].isin(need)]
    pairs = weights.assign(_w=np.arange(len(weights)))[["_row", "_w", "record_date", "value"]].merge(
        heights[["_row", "record_date", "value"]], on="_row", suffixes=("", "_h"))
    pairs["_gap"] = (pairs["record_date"] - pairs["record_date_h"]).abs().dt.days
    pairs = pairs.loc[pairs["_gap"] <= rules.weight_height_max_gap_days]
    pairs = pairs.sort_values(["_w", "_gap", "record_date_h"], ascending=[True, True, False], kind="stable")
    pairs = pairs.drop_duplicates("_w", keep="first")  # nearest height for each weight record
    pairs["bmi"] = pairs["value"] / pairs["value_h"] ** 2
    ok = pairs["bmi"].between(lo, hi)
    invalid += int((~ok).sum())
    computed = _latest_day_mean(pairs.loc[ok], "bmi", n)
    return bmi.fillna(computed).astype("float64").reset_index(drop=True), invalid


def _derive_smoking(n: int, life: pd.DataFrame, used: list[pd.DataFrame]) -> pd.Series:
    """Most recent status before index; 'never' after an earlier ex/current record is 'ex' (D-04)."""
    sm = life.loc[life["kind"] == "smoking"].assign(_rank=lambda d: d["value"].map(SMOKING_LEVELS.index))
    used.append(sm)
    last = sm.sort_values(["_row", "record_date", "_rank"], kind="stable").groupby("_row")["value"].last()
    ever_smoker = sm.loc[sm["value"] != "never"].groupby("_row").size() > 0
    status = last.where(~((last == "never") & ever_smoker.reindex(last.index, fill_value=False)), "ex")
    return status.reindex(range(n)).astype("str")


def _derive_alcohol(n: int, life: pd.DataFrame, rules: DerivationRules, used: list[pd.DataFrame]) -> pd.Series:
    """Highest level in the look-back by precedence; coded category beats same-day units (D-05)."""
    al = life.loc[life["kind"].isin(("alcohol", "alcohol_units_week")).to_numpy()
                  & (life["record_date"] >= _window_start(life["index_date"], years=rules.alcohol_window_years)).to_numpy()]
    used.append(al)
    coded = al.loc[al["kind"] == "alcohol"]
    units = al.loc[al["kind"] == "alcohol_units_week"]
    key = ["_row", "record_date"]
    units = units.loc[~pd.MultiIndex.from_frame(units[key]).isin(pd.MultiIndex.from_frame(coded[key]))]
    # units/week: 0 zero; (0, 21) lower risk; [21, 49) higher risk; >= 49 harmful
    unit_levels = pd.cut(units["units"], bins=[0.0, 21.0, 49.0, np.inf], right=False,
                         labels=["lower_risk", "higher_risk", "harmful"]).astype("str")
    unit_levels = unit_levels.where(units["units"] != 0.0, "zero")
    rank = pd.concat([coded["value"], unit_levels], ignore_index=True).map(ALCOHOL_PRECEDENCE.index)
    best = rank.groupby(np.concatenate([coded["_row"].to_numpy(), units["_row"].to_numpy()])).max()
    return best.map(lambda r: ALCOHOL_PRECEDENCE[int(r)]).reindex(range(n)).astype("str")


# ---------------------------------------------------------------------------------------------- outcome (D-09)
@dataclass(frozen=True)
class _CodeRule:
    prefix: str
    code_block: str
    efalls_code: str
    kind: str
    status: str


# Category part before '.' per code system; a dotted code with another shape cannot be matched safely (M-02:
# e.g. ICD-10 'E88.0' mislabelled as ICD-9-CM would otherwise normalise to the fall code E880).
_DOTTED_HEAD = {"ICD-10": r"[A-Z]\d{2}", "ICD-10-CM": r"[A-Z]\d[A-Z0-9]", "ICD-9-CM": r"E\d{3}|V\d{2}|\d{3}"}


def normalise_code(code: str) -> str:
    """Remove '.' and surrounding spaces and upper-case (D-09)."""
    return str(code).replace(".", "").strip().upper()


def _well_formed(codes: pd.Series, system: str) -> np.ndarray:
    text = codes.astype("str").str.strip().str.upper()
    head = text.str.split(".", n=1).str[0]
    return (~text.str.contains(".", regex=False) | head.str.fullmatch(_DOTTED_HEAD[system])).to_numpy(dtype=bool)


def _icd10_rules(spec: FeatureSpec) -> list[_CodeRule]:
    out = []
    for c in spec.outcome.codes:
        if c.get("match", "category_prefix") not in ("category_prefix", "exact_or_subcode"):
            raise ConfigError(f"outcome code {c.get('code')}: unsupported match {c.get('match')!r}")
        prefix = normalise_code(c["code"])
        out.append(_CodeRule(prefix, str(c["code"]), str(c["code"]), "fall" if prefix.startswith("W") else "fracture",
                             "CANDIDATE"))
    if not out:
        raise ConfigError("feature spec declares no outcome codes")
    return out


def _expand_code_range(text: str) -> list[str]:
    parts = [normalise_code(p) for p in str(text).split("-")]
    if len(parts) == 1:
        return parts
    matches = [re.fullmatch(r"([A-Z]?)(\d+)", p) for p in parts]
    if len(parts) != 2 or not all(matches) or matches[0][1] != matches[1][1] or len(matches[0][2]) != len(matches[1][2]) \
            or int(matches[0][2]) > int(matches[1][2]):
        raise ConfigError(f"invalid code range {text!r}")
    letter, width = matches[0][1], len(matches[0][2])
    return [f"{letter}{i:0{width}d}" for i in range(int(matches[0][2]), int(matches[1][2]) + 1)]


def load_icd9cm_rules(path: str | Path) -> list[_CodeRule]:
    """Load the M-02 ICD-9-CM candidate list; only CANDIDATE entries count as outcome codes."""
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = REPO_ROOT / p
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"ICD-9-CM candidate list not found: {path}") from exc
    out: list[_CodeRule] = []
    seen: dict[str, str] = {}
    for e in raw.get("icd9cm") or []:
        if e.get("status") not in ("CANDIDATE", "REVIEW", "EXCLUDE") or e.get("kind") not in ("fracture", "fall", "other"):
            raise ConfigError(f"ICD-9-CM entry {e} needs status CANDIDATE|REVIEW|EXCLUDE and kind fracture|fall|other")
        for prefix in _expand_code_range(e["codes"]):
            if prefix in seen:
                raise ConfigError(f"ICD-9-CM prefix {prefix} listed twice ({seen[prefix]}, {e['codes']})")
            seen[prefix] = str(e["codes"])
            out.append(_CodeRule(prefix, str(e["codes"]), str(e["efalls_code"]), e["kind"], e["status"]))
    if not out:
        raise ConfigError(f"ICD-9-CM candidate list {p} has no entries")
    return out


def _match_rules(codes: pd.Series, rules: list[_CodeRule]) -> np.ndarray:
    """Index of the longest matching prefix rule per code (-1 = no match)."""
    best = np.full(len(codes), -1)
    best_len = np.zeros(len(codes), dtype=int)
    for i, r in enumerate(rules):
        hit = codes.str.startswith(r.prefix).to_numpy(dtype=bool) & (len(r.prefix) > best_len)
        best[hit] = i
        best_len[hit] = len(r.prefix)
    return best


def _derive_outcome(rows: pd.DataFrame, encounters: pd.DataFrame, spec: FeatureSpec, rules: DerivationRules,
                    counts: Counter[str]) -> tuple[pd.Series, pd.Series, pd.DataFrame, int]:
    """Qualifying fall/fracture encounters in [index, window_end] per the D-09 rules, plus the audit table."""
    n = len(rows)
    enc = rows[["_row", "research_id", "index_date"]].merge(encounters, on="research_id")
    end = spec.outcome.window_end(enc["index_date"])
    enc = enc.loc[((enc["event_date"] >= enc["index_date"]) & (enc["event_date"] <= end)).to_numpy()].reset_index(drop=True)
    enc["code"] = enc["diagnosis_code"].map(normalise_code)

    matched = []
    systems = {"ICD-10": _icd10_rules, "ICD-10-CM": _icd10_rules}
    for system, sub in enc.groupby("code_system", sort=True):
        code_rules = systems[system](spec) if system in systems else load_icd9cm_rules(rules.icd9cm_candidates_path)
        which = _match_rules(sub["code"], code_rules)
        malformed = ~_well_formed(sub["diagnosis_code"], str(system))
        counts[f"encounters.malformed_{system}_code_in_window"] += int(malformed.sum())
        which[malformed] = -1
        sub = sub.loc[which >= 0]
        attrs = pd.DataFrame([vars(code_rules[i]) for i in which[which >= 0]], index=sub.index,
                             columns=["prefix", "code_block", "efalls_code", "kind", "status"])
        matched.append(pd.concat([sub, attrs.drop(columns="prefix").rename(columns={"status": "code_status"})], axis=1))
    empty_cols = [*enc.columns, "code_block", "efalls_code", "kind", "code_status"]
    enc = pd.concat(matched, ignore_index=True) if matched else pd.DataFrame(columns=empty_cols)

    method = enc["admission_method"]
    admission = (enc["encounter_type"] == "hospital_admission").to_numpy()
    unknown = admission & method.isna().to_numpy() & (enc["code_status"] == "CANDIDATE").to_numpy()
    n_unknown = int(enc.loc[unknown, ["research_id", "encounter_id"]].drop_duplicates().shape[0])
    non_elective = rules.outcome_admission_rule == "non_elective"
    if non_elective and rules.unknown_admission_method == "error" and n_unknown:
        raise DatasetValidationError(f"{n_unknown} outcome-coded hospital admissions have an unknown admission method "
                                     "(unknown_admission_method='error', D-09)")
    code = enc["code"].astype("str")
    spell = enc["spell_start_date"].astype(DATE_DTYPE)
    conditions = [
        (enc["code_status"] == "REVIEW").to_numpy(), (enc["code_status"] == "EXCLUDE").to_numpy(),
        ((enc["code_system"] == "ICD-10-CM") & (code.str.len() == 7) & ~code.str[6:].isin(("A", "B", "C"))).to_numpy(),
        (enc["encounter_type"] == "other").to_numpy(),
        rules.exclude_spells_started_before_index & (spell.notna() & (spell < enc["index_date"])).to_numpy(),
        non_elective & unknown & (rules.unknown_admission_method == "exclude"),
        non_elective & admission & method.notna().to_numpy() & ~method.isin(NON_ELECTIVE_METHODS).to_numpy(),
        (rules.outcome_position_rule == "principal_fracture_or_any_external")
        & ((enc["kind"] == "fracture") & (enc["diagnosis_position"] != 1)).to_numpy(),
    ]
    labels = ["icd9cm_review_not_counted", "icd9cm_excluded_code", "icd10cm_excluded_7th_character",
              "encounter_type_not_qualifying", "spell_started_before_index", "unknown_admission_method",
              "admission_not_non_elective", "position_rule"]
    enc["status"] = np.select(conditions, labels, default="qualifying") if len(enc) else pd.Series(dtype="str")

    qualifying = enc.loc[enc["status"] == "qualifying"]
    first = qualifying.groupby("_row")["event_date"].min().reindex(range(n)).astype(DATE_DTYPE)
    outcome = first.notna().astype("int8")
    audit = (enc.assign(diagnosis_position=np.where(enc["diagnosis_position"] == 1, "principal", "secondary"),
                        admission_method=method.astype("str").fillna("unknown"))
             .groupby(list(AUDIT_COLUMNS[:-1]), sort=True).size().rename("n_records").reset_index())
    audit = audit.reindex(columns=list(AUDIT_COLUMNS)) if len(audit) else pd.DataFrame(columns=list(AUDIT_COLUMNS))
    return outcome.reset_index(drop=True), first.reset_index(drop=True), audit, n_unknown
