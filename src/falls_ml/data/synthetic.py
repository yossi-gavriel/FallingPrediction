"""Deterministic SYNTHETIC fixtures for software tests only (spec §13.2). SYNTHETIC — NOT SCIENTIFIC EVIDENCE.

Nothing here resembles SAIL, Connected Bradford or Meuhedet:
- condition onset probabilities are drawn uniformly from 1%-25% (seeded) and are NOT tuned to published marginals;
- the outcome follows a KNOWN, ARBITRARY, NON-eFalls logistic model (a few strong effects, all other binary
  predictors without effect; coefficients deliberately unrelated to Table S3.2) with the intercept shifted to
  target ~8% prevalence. The model is written to ``synthetic_generating_model.json`` next to the dataset.

Event tables follow the canonical layout of :mod:`falls_ml.dataeng.derive` and deliberately contain post-index
and back-filled records, invalid BMI values, excluded BNF chapters, elective admissions, continuation spells,
non-listed injury codes, deaths and membership gaps. Qualifying fall/fracture encounters are drawn only inside
outcome windows of generated positives, so the derived outcome equals the generated one (checked on write).
This module is the only ML-package module allowed to import ``falls_ml.dataeng`` (architecture §1).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit

from falls_ml.artifacts import write_json
from falls_ml.data.dataset import DatasetManifest, feature_subset_record, write_modeling_dataset
from falls_ml.dataeng.derive import TABLE_COLUMNS, DerivationRules, derive_modeling_frame, empty_event_tables
from falls_ml.errors import ConfigError, FallsMLError
from falls_ml.features.spec import FeatureDefinition, FeatureSpec, load_feature_spec
from falls_ml.logging_utils import get_logger
from falls_ml.seeding import rng_for

log = get_logger(__name__)

SYNTHETIC_LABEL = "SYNTHETIC — NOT SCIENTIFIC EVIDENCE"
GENERATING_MODEL_FILE = "synthetic_generating_model.json"
DEFAULT_INDEX_DATES = ("2018-04-01", "2019-04-01", "2020-04-01", "2021-04-01")
DEFAULT_START_DATE, DEFAULT_END_DATE = "2012-01-01", "2023-12-31"
TARGET_PREVALENCE = 0.08
REPO_ROOT = Path(__file__).resolve().parents[3]
NAT = np.datetime64("NaT", "ns")

N_PRACTICES, N_SITES = 20, 4
DEPRIVATION_GROUPS = ("1", "2", "3", "4", "5", "missing")
BNF_DRUG_PARAGRAPHS = tuple(f"{ch:02d}{sec:02d}{par:02d}" for ch in range(1, 16) for sec in (1, 2, 3) for par in (1, 2))
BNF_OTHER_ITEMS = ("200101", "210101", "220201", "230301", "060106")  # pseudo-chapters 20-23; 060106 is non-drug
FRACTURE_CODES = ("S72.0", "S72.1", "S52.5", "S42.2", "S32.5", "S22.3", "S82.8", "T14.2", "M80.0", "T08", "T10", "T12")
FALL_CODES = ("W01", "W06", "W10", "W18", "W19")
NON_OUTCOME_CODES = ("S00.9", "S01.0", "S30.0", "S62.1", "S92.3", "T14.1", "R55", "J18.9", "R07.4", "X59.9")
ALCOHOL_LEVELS = ("harmful", "higher_risk", "lower_risk", "previous_higher_risk_or_harmful", "zero")
ALCOHOL_UNITS = (0, 4, 10, 18, 20, 21, 30, 48, 49, 70)

# Uniform ranges of the ARBITRARY generating coefficients (spec §13.2). Every range with a published eFalls counterpart
# (Table S3.2) excludes that value by more than 0.15, so no seed can reproduce a published coefficient. Polypharmacy enters
# as a raw count (published: ln((P+1)/10)); only 6 binaries have an effect, all above the largest published binary.
COEFFICIENT_RANGES: dict[str, tuple[float, float]] = {
    "age_per_decade_from_75": (0.6, 0.9),  # published 0.0415506/year = 0.416/decade
    "male": (0.5, 0.8),  # published sex term +-0.303708 (D-01)
    "polypharmacy_count_per_paragraph": (0.04, 0.09),  # different functional form from the published log term
    "bmi_missing": (0.3, 0.6),  # published missing -0.145 (vs overweight)
    "bmi_at_least_30": (0.3, 0.6),  # published obese -0.041 (vs overweight)
    "smoking_current": (0.4, 0.8),  # published current 0.068
    "alcohol_higher_risk_or_harmful": (0.7, 1.1),  # published harmful 0.416, higher risk 0.155
    "binary": (1.0, 2.0),  # published binary coefficients -0.235..0.803
}
N_STRONG_BINARIES = 6


class SyntheticFixtureError(FallsMLError):
    """The generated fixture is internally inconsistent (e.g. derived outcome != generated outcome)."""


@dataclass(frozen=True)
class _SyntheticWorld:
    tables: dict[str, pd.DataFrame]
    generating_model: dict[str, Any]


# ---------------------------------------------------------------------------------------------- public API
def generate_synthetic_event_tables(n_patients: int, *, seed: int, spec: FeatureSpec, start_date: str = DEFAULT_START_DATE,
                                    end_date: str = DEFAULT_END_DATE,
                                    index_dates: Sequence[str] = DEFAULT_INDEX_DATES) -> dict[str, pd.DataFrame]:
    """Canonical SYNTHETIC event tables plus ``index_table`` and ``synthetic_truth`` (spec §13.2).

    ``synthetic_truth`` holds, per eligible (research_id, index_date), the generated linear predictor, outcome,
    first event date and any extension-feature values (extension features are not derivable from event tables).
    """
    return _generate(n_patients, seed=seed, spec=spec, start_date=start_date, end_date=end_date,
                     index_dates=index_dates).tables


def generate_synthetic_modeling_dataset(directory: str | Path, *, n_patients: int = 1500,
                                        index_dates: Sequence[str] = DEFAULT_INDEX_DATES, seed: int = 20260914,
                                        feature_spec_path: str | Path = "configs/features/efalls_v1.yaml",
                                        extension_spec_paths: Sequence[str | Path] = (),
                                        dataset_version: str = "synthetic-1.0.0",
                                        created_utc: str = "2026-09-14T00:00:00+00:00",
                                        features: Sequence[str] | None = None) -> DatasetManifest:
    """Generate, derive (D-00..D-09 primary rules) and write a deterministic SYNTHETIC modelling dataset.

    ``features`` restricts the written dataset to those predictors and records it with the subset feature spec
    (``FeatureSpec.subset``). Generation always uses the full spec, so rows and outcomes equal the unrestricted fixture
    with the same seed, patients and index dates; only unlisted predictor columns are absent.
    """
    spec = load_feature_spec(_resolve(feature_spec_path), [_resolve(p) for p in extension_spec_paths])
    out_spec = spec if features is None else spec.subset(features)
    world = _generate(n_patients, seed=seed, spec=spec, start_date=DEFAULT_START_DATE, end_date=DEFAULT_END_DATE,
                      index_dates=index_dates)
    tables, truth = world.tables, world.tables["synthetic_truth"]
    frame, dlog = derive_modeling_frame(tables, tables["index_table"], _derivable(spec), DerivationRules())
    key = ["research_id", spec.index_column]
    check = frame[[*key, spec.outcome.name, "outcome_first_event_date"]].merge(
        truth[[*key, spec.outcome.name, "outcome_first_event_date"]], on=key, how="outer", suffixes=("", "_generated"),
        indicator=True)
    mismatch = (check["_merge"] != "both") | (check[spec.outcome.name] != check[f"{spec.outcome.name}_generated"]) | (
        check["outcome_first_event_date"].fillna(pd.Timestamp(0)) != check["outcome_first_event_date_generated"].fillna(pd.Timestamp(0)))
    if mismatch.any():
        raise SyntheticFixtureError(f"derived outcome differs from generated outcome in {int(mismatch.sum())} rows")
    extension = [f.name for f in spec.features if not f.exact_efalls_baseline]
    frame = frame.merge(truth[[*key, *extension]], on=key, how="left", validate="one_to_one")
    ordered = [c for c in frame.columns if c not in spec.predictor_names() and c not in spec.metadata_columns]
    frame = frame[[*ordered, *out_spec.predictor_names(), *spec.metadata_columns]]
    audit: dict[str, Any] = {"derivation_exclusions": dlog.exclusions,
                             "outcome_audit_d09": dlog.outcome_audit.to_dict(orient="records")
                             if hasattr(dlog.outcome_audit, "to_dict") else dlog.outcome_audit}
    if out_spec.is_subset:
        audit["feature_spec_subset"] = feature_subset_record(spec, out_spec)
    manifest = write_modeling_dataset(frame, directory, out_spec, dataset_version=dataset_version,
                                      mapping_version="synthetic-identity", source="synthetic_fixture",
                                      generator="falls_ml.data.synthetic", scientific_use_allowed=False,
                                      notes=SYNTHETIC_LABEL, created_utc=created_utc, data_freeze_date=DEFAULT_END_DATE, overwrite=True,
                                      audit=audit)
    payload = {**world.generating_model, "realised": {"n_rows": manifest.n_rows, "n_patients": manifest.n_patients,
                                                      "outcome_prevalence": manifest.outcome_prevalence},
               "derivation_exclusions": dlog.exclusions}
    if out_spec.is_subset:
        payload["written_feature_subset"] = {**feature_subset_record(spec, out_spec),
                                             "note": "the outcome was generated with the full predictor set; unlisted predictors are "
                                                     "absent from the written dataset (simulates an extraction lacking them)"}
    write_json(Path(directory) / GENERATING_MODEL_FILE, payload)
    log.info("synthetic_fixture_written", extra_fields={"directory": str(directory), "label": SYNTHETIC_LABEL,
                                                        "n_rows": manifest.n_rows, "data_sha256": manifest.data_sha256})
    return manifest


# ---------------------------------------------------------------------------------------------- orchestration
def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() or p.exists() else REPO_ROOT / p


def _derivable(spec: FeatureSpec) -> FeatureSpec:
    base = [f.name for f in spec.features if f.exact_efalls_baseline]
    return spec if len(base) == len(spec.features) else spec.subset(base)


def _generate(n_patients: int, *, seed: int, spec: FeatureSpec, start_date: str, end_date: str,
              index_dates: Sequence[str]) -> _SyntheticWorld:
    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    idx_dates = sorted(pd.Timestamp(x) for x in index_dates)
    if n_patients < 1 or not idx_dates:
        raise ConfigError("synthetic fixture needs n_patients >= 1 and at least one index date")
    if not start < idx_dates[0] <= idx_dates[-1] <= end:
        raise ConfigError(f"index dates must lie within ({start.date()}, {end.date()}]")
    ends = [spec.outcome.window_end(d) for d in idx_dates]
    if any(nxt <= e for e, nxt in zip(ends, idx_dates[1:])):
        raise ConfigError("synthetic index dates must have non-overlapping outcome windows")

    base = _derivable(spec)
    extension = [f for f in spec.features if not f.exact_efalls_baseline]
    persons = _persons(n_patients, seed, start, end)
    ids = persons["research_id"].drop_duplicates().to_numpy()
    onset_probability = dict(zip(base.binary_names(),
                                 np.round(rng_for(seed, "synthetic.prevalence").uniform(0.01, 0.25, len(base.binary_names())), 4)))
    tables = empty_event_tables()
    tables.update(persons=persons, conditions=_conditions(ids, onset_probability, seed, start, end),
                  prescriptions=_prescriptions(ids, seed, start, end),
                  measurements=_measurements(persons.drop_duplicates("research_id"), seed, start, end),
                  lifestyle=_lifestyle(ids, seed, start, end))
    index_table = pd.DataFrame({"research_id": np.repeat(ids, len(idx_dates)),
                                "index_date": np.tile(np.array(idx_dates, dtype="datetime64[ns]"), len(ids))})
    predictors, _ = derive_modeling_frame(tables, index_table, base, DerivationRules())
    if len(predictors) == 0:
        raise ConfigError("synthetic fixture has no eligible rows; increase n_patients")

    model = _generating_model(seed, base.binary_names(), extension)
    values = _extension_values(predictors, extension, seed)
    lp0 = _linear_predictor(pd.concat([predictors, values.drop(columns=["research_id", "index_date"])], axis=1), model)
    intercept = round(float(brentq(lambda b: float(expit(lp0 + b).mean()) - TARGET_PREVALENCE, -40.0, 40.0)), 6)
    y = rng_for(seed, "synthetic.outcome").random(len(lp0)) < expit(lp0 + intercept)
    encounters, first_event = _encounters(persons, predictors, y, idx_dates, spec, seed, start, end)

    truth = values.assign(linear_predictor=lp0 + intercept, outcome_12m=y.astype("int8"),
                          outcome_first_event_date=first_event)
    tables.update(encounters=encounters, index_table=index_table, synthetic_truth=truth)
    model = {"label": SYNTHETIC_LABEL, **model, "intercept": intercept, "target_prevalence": TARGET_PREVALENCE,
             "seed": int(seed), "n_patients": int(n_patients), "index_dates": [str(d.date()) for d in idx_dates],
             "condition_onset_probability": {k: float(v) for k, v in onset_probability.items()}}
    return _SyntheticWorld(tables=tables, generating_model=model)


# ---------------------------------------------------------------------------------------------- generating model
def _generating_model(seed: int, binary_names: list[str], extension: list[FeatureDefinition]) -> dict[str, Any]:
    rng = rng_for(seed, "synthetic.generating_model")
    strong = sorted(rng.choice(len(binary_names), size=min(N_STRONG_BINARIES, len(binary_names)), replace=False))
    u = lambda lo, hi: round(float(rng.uniform(lo, hi)), 4)  # noqa: E731
    terms = {name: u(*bounds) for name, bounds in COEFFICIENT_RANGES.items() if name != "binary"}
    binary = {binary_names[i]: u(*COEFFICIENT_RANGES["binary"]) for i in strong}
    ext: dict[str, dict[str, float]] = {}
    for f in extension:
        if f.dtype == "binary":
            ext[f.name] = {"coefficient": u(0.5, 1.2)}
        elif f.dtype in ("float", "float_nullable", "count"):
            ext[f.name] = {"per_unit": u(-0.4, -0.15), "missing": u(0.1, 0.4)}
        else:
            raise ConfigError(f"synthetic extension feature {f.name!r}: dtype {f.dtype} not supported")
    return {
        "purpose": "software tests only (spec §13.2): ARBITRARY coefficients deliberately unrelated to eFalls Table S3.2; "
                   "condition prevalences are arbitrary and NOT tuned to SAIL/CB marginals",
        "model_form": "logit p = intercept + age_per_decade_from_75*(age_years-75)/10 + male*[sex=male] "
                      "+ polypharmacy_count_per_paragraph*polypharmacy_count_120d + bmi_missing*[bmi_value missing] "
                      "+ bmi_at_least_30*[bmi_value>=30] + smoking_current*[smoking_status=current] "
                      "+ alcohol_higher_risk_or_harmful*[alcohol_category in {higher_risk, harmful}] "
                      "+ sum(binary_coefficients[j]*x_j) + extension terms (binary: coefficient*x; numeric: "
                      "per_unit*x with missing as 0 + missing*[x missing]); all other binaries have coefficient 0",
        "terms": terms, "binary_coefficients": binary, "extension_coefficients": ext,
        "coefficient_ranges": {name: list(bounds) for name, bounds in COEFFICIENT_RANGES.items()},
    }


def _linear_predictor(df: pd.DataFrame, model: dict[str, Any]) -> np.ndarray:
    t = model["terms"]
    lp = (t["age_per_decade_from_75"] * (df["age_years"].to_numpy() - 75.0) / 10.0
          + t["male"] * (df["sex"] == "male").to_numpy()
          + t["polypharmacy_count_per_paragraph"] * df["polypharmacy_count_120d"].to_numpy()
          + t["bmi_missing"] * df["bmi_value"].isna().to_numpy()
          + t["bmi_at_least_30"] * (df["bmi_value"] >= 30).to_numpy()
          + t["smoking_current"] * (df["smoking_status"] == "current").to_numpy()
          + t["alcohol_higher_risk_or_harmful"] * df["alcohol_category"].isin(("higher_risk", "harmful")).to_numpy())
    for name, b in model["binary_coefficients"].items():
        lp = lp + b * df[name].to_numpy()
    for name, c in model["extension_coefficients"].items():
        x = df[name].astype("float64")
        lp = lp + (c["coefficient"] * x.to_numpy() if "coefficient" in c
                   else c["per_unit"] * x.fillna(0.0).to_numpy() + c["missing"] * x.isna().to_numpy())
    return np.asarray(lp, dtype="float64")


def _extension_values(predictors: pd.DataFrame, extension: list[FeatureDefinition], seed: int) -> pd.DataFrame:
    out = predictors[["research_id", "index_date"]].copy()
    n = len(out)
    for f in extension:
        rng = rng_for(seed, f"synthetic.extension.{f.name}")
        if f.dtype == "binary":
            out[f.name] = (rng.random(n) < rng.uniform(0.08, 0.3)).astype("int8")
            continue
        vr = f.valid_range or {}
        lo = float(vr.get("min") if vr.get("min") is not None else 0.0)
        hi = float(vr.get("max") if vr.get("max") is not None else lo + 10.0)
        integral = f.dtype == "count" or (lo.is_integer() and hi.is_integer())
        vals = rng.integers(int(lo), int(hi) + 1, n).astype("float64") if integral else rng.uniform(lo, hi, n)
        if f.dtype == "count":
            out[f.name] = vals.astype("int64")
        else:
            out[f.name] = np.where(rng.random(n) < 0.3, np.nan, vals) if f.nullable else vals
    return out


# ---------------------------------------------------------------------------------------------- event tables
def _dates(rng: np.random.Generator, lo: pd.Timestamp, hi: pd.Timestamp, n: int) -> np.ndarray:
    days = rng.integers(0, (hi - lo).days + 1, n)
    return (np.datetime64(lo.date(), "D") + days).astype("datetime64[ns]")


def _availability(rng: np.random.Generator, record: np.ndarray, max_lag: int, backfill_rate: float,
                  backfill_days: tuple[int, int]) -> np.ndarray:
    lag = rng.integers(0, max_lag + 1, len(record))
    backfilled = rng.random(len(record)) < backfill_rate
    lag = np.where(backfilled, rng.integers(*backfill_days, len(record)), lag)
    return record + lag.astype("timedelta64[D]")


def _repeat_owner(counts: np.ndarray) -> np.ndarray:
    return np.repeat(np.arange(len(counts)), counts)


def _within_group_index(owner: np.ndarray) -> np.ndarray:
    return pd.Series(owner).groupby(owner).cumcount().to_numpy()


def _persons(n: int, seed: int, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    rng = rng_for(seed, "synthetic.persons")
    ids = np.array([f"SYN{i:06d}" for i in range(n)])
    age_2018 = rng.uniform(58.0, 97.0, n)
    dob = (np.datetime64("2018-04-01", "D") - np.round(age_2018 * 365.25).astype("timedelta64[D]")).astype("datetime64[ns]")
    sex = rng.choice(np.array(["female", "male"]), n, p=[0.55, 0.45]).astype(object)
    sex[rng.random(n) < 0.004] = "unknown"
    practice = rng.integers(1, N_PRACTICES + 1, n)
    deprivation = rng.choice(np.array(DEPRIVATION_GROUPS), n, p=[0.18] * 5 + [0.10])
    joined_late = rng.random(n) < 0.12
    m_start = np.where(joined_late, _dates(rng, start, end - pd.DateOffset(years=2), n),
                       _dates(rng, pd.Timestamp("1990-01-01"), start, n))
    death = np.where(rng.random(n) < 0.15, _dates(rng, start + pd.DateOffset(years=1), end, n), NAT)
    death = np.where(death < m_start, NAT, death)
    leave = np.where(rng.random(n) < 0.08, _dates(rng, start + pd.DateOffset(years=2), end, n), NAT)
    leave = np.where(leave <= m_start, NAT, leave)
    m_end = pd.Series(np.fmin(death, leave)).to_numpy(dtype="datetime64[ns]")  # fmin ignores NaT
    gap_start = _dates(rng, pd.Timestamp("2015-01-01"), pd.Timestamp("2021-06-30"), n)
    gap_end = gap_start + rng.integers(60, 700, n).astype("timedelta64[D]")
    has_gap = (rng.random(n) < 0.06) & (m_start < gap_start) & (np.isnat(m_end) | (gap_end < m_end))
    moved = rng.random(n) < 0.5

    def frame(mask: np.ndarray, first: np.ndarray, last: np.ndarray, prac: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame({"research_id": ids[mask], "date_of_birth": dob[mask], "sex": sex[mask],
                             "practice_id": [f"P{p:02d}" for p in prac[mask]],
                             "deprivation_group": deprivation[mask], "site_id": [f"S{(p - 1) % N_SITES + 1}" for p in prac[mask]],
                             "death_date": death[mask], "membership_start": first[mask], "membership_end": last[mask]})

    later_practice = np.where(moved, practice % N_PRACTICES + 1, practice)
    out = pd.concat([frame(np.ones(n, dtype=bool), m_start, np.where(has_gap, gap_start, m_end), practice),
                     frame(has_gap, gap_end, m_end, later_practice)], ignore_index=True)
    return out.sort_values(["research_id", "membership_start"], kind="stable").reset_index(drop=True)[list(TABLE_COLUMNS["persons"])]


def _conditions(ids: np.ndarray, onset_probability: dict[str, float], seed: int, start: pd.Timestamp,
                end: pd.Timestamp) -> pd.DataFrame:
    rng = rng_for(seed, "synthetic.conditions")
    features = np.array(list(onset_probability))
    has = rng.random((len(ids), len(features))) < np.array(list(onset_probability.values()))
    person_idx, feature_idx = np.nonzero(has)
    onset = _dates(rng, start, end, len(person_idx))
    owner = _repeat_owner(1 + rng.binomial(3, 0.3, len(person_idx)))
    gaps = np.where(_within_group_index(owner) == 0, 0, rng.integers(30, 900, len(owner)))
    offset = pd.Series(gaps).groupby(owner).cumsum().to_numpy()
    record = onset[owner] + offset.astype("timedelta64[D]")
    keep = record <= np.datetime64(end.date())
    owner, record = owner[keep], record[keep]
    df = pd.DataFrame({"research_id": ids[person_idx[owner]], "feature": features[feature_idx[owner]],
                       "record_date": record,
                       "available_date": _availability(rng, record, 14, 0.06, (180, 1500)),
                       "source_class": rng.choice(np.array(["community_diagnosis", "hospital_diagnosis"]), len(owner), p=[0.8, 0.2])})
    return df.sort_values(["research_id", "record_date", "feature"], kind="stable").reset_index(drop=True)


def _prescriptions(ids: np.ndarray, seed: int, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    rng = rng_for(seed, "synthetic.prescriptions")
    med_owner = _repeat_owner(rng.poisson(rng.gamma(2.0, 2.5, len(ids))))
    m = len(med_owner)
    paragraph = np.where(rng.random(m) < 0.1, rng.choice(np.array(BNF_OTHER_ITEMS), m),
                         rng.choice(np.array(BNF_DRUG_PARAGRAPHS), m))
    med_start = _dates(rng, start - pd.DateOffset(years=1), end, m)
    interval = rng.integers(28, 91, m)
    fills = rng.integers(30, 1500, m) // interval + 1
    fill_med = _repeat_owner(fills)
    record = med_start[fill_med] + (_within_group_index(fill_med) * interval[fill_med]).astype("timedelta64[D]")
    keep = (record >= np.datetime64(start.date())) & (record <= np.datetime64(end.date()))
    fill_med, record = fill_med[keep], record[keep]
    para = paragraph[fill_med]
    return pd.DataFrame({"research_id": ids[med_owner[fill_med]], "bnf_paragraph": para,
                         "bnf_chapter": pd.Series(para).str.slice(0, 2).astype("int64").to_numpy(), "is_drug": para != "060106",
                         "record_date": record, "available_date": _availability(rng, record, 3, 0.02, (100, 800))})


def _measurements(persons: pd.DataFrame, seed: int, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    rng = rng_for(seed, "synthetic.measurements")
    ids = persons["research_id"].to_numpy()
    n = len(ids)
    years = (end - start).days / 365.25
    parts: list[pd.DataFrame] = []

    def events(rate: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
        owner = _repeat_owner(rng.poisson(np.broadcast_to(rate * years, (n,))))
        return owner, _dates(rng, start, end, len(owner))

    def add(kind: str, owner: np.ndarray, record: np.ndarray, values: np.ndarray, decimals: int) -> None:
        parts.append(pd.DataFrame({"research_id": ids[owner], "kind": kind, "value": np.round(values, decimals),
                                   "record_date": record}))

    bmi_mu = rng.normal(27.5, 4.5, n)
    owner, record = events(0.25 * (rng.random(n) < 0.7))
    bmi = rng.normal(bmi_mu[owner], 1.2)
    invalid = rng.random(len(owner)) < 0.02
    add("bmi", owner, record, np.where(invalid, rng.choice(np.array([4.0, 8.5, 95.0, 275.0]), len(owner)), bmi), 1)
    height_mu = np.clip(rng.normal(1.66, 0.09, n), 1.35, 2.05)
    owner, record = events(0.12)
    add("weight_kg", owner, record, np.clip(rng.normal(bmi_mu[owner] * height_mu[owner] ** 2, 3.0), 35, 180), 1)
    add("height_m", owner, record + rng.integers(-40, 41, len(owner)).astype("timedelta64[D]"), height_mu[owner], 2)
    sbp_mu = rng.normal(135, 14, n)
    owner, record = events(0.5)
    add("sbp", owner, record, rng.normal(sbp_mu[owner], 12), 0)
    add("dbp", owner, record, rng.normal(sbp_mu[owner] * 0.58, 7), 0)
    hb_mu = np.where(persons["sex"].to_numpy() == "male", 14.0, 13.0) + rng.normal(0, 1.2, n)
    for kind, rate, sampler, decimals in (
        ("hb_g_dl", 0.3, lambda o: rng.normal(hb_mu[o], 0.8), 1),
        ("egfr", 0.3, lambda o: np.clip(rng.normal(np.clip(rng.normal(70, 18, n), 5, 120)[o], 6), 3, 130), 0),
        ("acr_mg_mmol", 0.1, lambda o: rng.lognormal(0.3, 1.0, len(o)), 1),
        ("tsh_mu_l", 0.15, lambda o: rng.lognormal(np.log(2.0), 0.6, len(o)), 2),
        ("dexa_tscore", 0.03, lambda o: rng.normal(-1.8, 1.0, len(o)), 1),
        ("abpi", 0.02, lambda o: rng.normal(1.05, 0.15, len(o)), 2),
        ("barthel", 0.03, lambda o: np.clip(np.round(rng.normal(17.5, 3.0, len(o))), 0, 20), 0),
        ("sixcit", 0.04, lambda o: np.clip(np.round(np.exp(rng.normal(1.2, 0.9, len(o)))), 0, 28), 0),
    ):
        owner, record = events(rate)
        add(kind, owner, record, sampler(owner), decimals)
    df = pd.concat(parts, ignore_index=True)
    df["record_date"] = df["record_date"].astype("datetime64[ns]")
    df["available_date"] = _availability(rng, df["record_date"].to_numpy(), 7, 0.03, (200, 1200))
    return df.sort_values(["research_id", "record_date", "kind"], kind="stable").reset_index(drop=True)


def _lifestyle(ids: np.ndarray, seed: int, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    rng = rng_for(seed, "synthetic.lifestyle")
    n, years = len(ids), (end - start).days / 365.25
    smoker_type = rng.choice(3, n, p=[0.5, 0.35, 0.15])  # never / ex / current tendency
    probs = np.array([[0.9, 0.1, 0.0], [0.15, 0.6, 0.25], [0.05, 0.15, 0.8]]).cumsum(axis=1)
    owner = _repeat_owner(rng.poisson(0.2 * years, n))
    status = (rng.random(len(owner))[:, None] > probs[smoker_type[owner]]).sum(axis=1)
    smoking = pd.DataFrame({"research_id": ids[owner], "kind": "smoking",
                            "value": np.array(["never", "ex", "current"])[np.minimum(status, 2)],
                            "record_date": _dates(rng, start, end, len(owner))})
    owner = _repeat_owner(rng.poisson(0.15 * years * (rng.random(n) < 0.3)))
    coded = rng.random(len(owner)) < 0.6
    value = np.where(coded, rng.choice(np.array(ALCOHOL_LEVELS), len(owner), p=[0.05, 0.15, 0.45, 0.05, 0.30]),
                     rng.choice(np.array([str(u) for u in ALCOHOL_UNITS]), len(owner)))
    alcohol = pd.DataFrame({"research_id": ids[owner], "kind": np.where(coded, "alcohol", "alcohol_units_week"),
                            "value": value, "record_date": _dates(rng, start, end, len(owner))})
    df = pd.concat([smoking, alcohol], ignore_index=True)
    df["available_date"] = _availability(rng, df["record_date"].to_numpy(), 7, 0.04, (200, 1200))
    df["value"] = df["value"].astype("str")
    return df.sort_values(["research_id", "record_date", "kind"], kind="stable").reset_index(drop=True)


def _encounters(persons: pd.DataFrame, predictors: pd.DataFrame, y: np.ndarray, idx_dates: list[pd.Timestamp],
                spec: FeatureSpec, seed: int, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, pd.Series]:
    """Qualifying encounters inside windows of generated positives only, plus never-qualifying noise (D-09)."""
    rng = rng_for(seed, "synthetic.encounters")
    enc: list[pd.DataFrame] = []

    def add(rid, etype, method, spell, event, codes) -> None:
        k = len(rid)
        enc.append(pd.DataFrame({"research_id": np.asarray(rid), "encounter_type": np.broadcast_to(etype, (k,)),
                                 "admission_method": np.broadcast_to(np.asarray(method, dtype=object), (k,)),
                                 "spell_start_date": np.asarray(spell, dtype="datetime64[ns]"),
                                 "event_date": np.asarray(event, dtype="datetime64[ns]"), "codes": list(codes)}))

    # qualifying episodes for generated positives: event in [index, min(window_end, death)]
    pos = predictors.loc[y].reset_index(drop=True)
    index = pos["index_date"].to_numpy(dtype="datetime64[ns]")
    last = np.fmin(spec.outcome.window_end(pos["index_date"]).to_numpy(dtype="datetime64[ns]"),
                   pos["death_date"].fillna(pd.Timestamp.max.normalize()).to_numpy(dtype="datetime64[ns]"))
    span = ((last - index) / np.timedelta64(1, "D")).astype("int64")
    first = index + np.where(rng.random(len(pos)) < 0.05, 0, rng.integers(0, span + 1)).astype("timedelta64[D]")
    second = first + rng.integers(0, ((last - first) / np.timedelta64(1, "D")).astype("int64") + 1).astype("timedelta64[D]")
    for event, mask in ((first, np.ones(len(pos), dtype=bool)), (second, rng.random(len(pos)) < 0.3)):
        k = int(mask.sum())
        pattern = rng.integers(0, 3, k)  # fracture+fall, fall only, fracture only
        fracture = rng.choice(np.array(FRACTURE_CODES), k)
        fall = rng.choice(np.array(FALL_CODES), k)
        injury = rng.choice(np.array(NON_OUTCOME_CODES[:5]), k)
        codes = [(fr, fa) if p == 0 else (inj, fa) if p == 1 else (fr,) for p, fr, fa, inj in zip(pattern, fracture, fall, injury)]
        ed = rng.random(k) < 0.55
        method = np.where(ed, None, rng.choice(np.array(["emergency", "urgent", "via_ed"]), k))
        add(pos.loc[mask, "research_id"], np.where(ed, "ed_attendance", "hospital_admission"), method,
            np.where(ed, NAT, event[mask]), event[mask], codes)

    # continuation spells started before index (excluded, D-09)
    cont = predictors.loc[rng.random(len(predictors)) < 0.03].reset_index(drop=True)
    c_index = cont["index_date"].to_numpy(dtype="datetime64[ns]")
    c_event = np.fmin(c_index + rng.integers(0, 21, len(cont)).astype("timedelta64[D]"),
                      cont["death_date"].fillna(pd.Timestamp.max.normalize()).to_numpy(dtype="datetime64[ns]"))
    add(cont["research_id"], "hospital_admission", "emergency", c_index - rng.integers(1, 31, len(cont)).astype("timedelta64[D]"),
        c_event, [("S72.0", "W19")] * len(cont))

    # never-qualifying noise at arbitrary dates (before death)
    people = persons.drop_duplicates("research_id").reset_index(drop=True)
    years = (end - start).days / 365.25
    windows = [(np.datetime64(d, "ns"), np.datetime64(spec.outcome.window_end(d), "ns")) for d in idx_dates]
    for etype, method, rate, code_pool, outside_windows in (
        ("ed_attendance", None, 0.10, NON_OUTCOME_CODES, False),
        ("hospital_admission", "elective", 0.02, ("S72.0", "S82.8"), False),
        ("other", None, 0.02, FALL_CODES, False),
        ("hospital_admission", None, 0.01, FRACTURE_CODES, False),
        ("ed_attendance", None, 0.05, FALL_CODES, True),
    ):
        owner = _repeat_owner(rng.poisson(rate * years, len(people)))
        event = _dates(rng, start, end, len(owner))
        death = people["death_date"].to_numpy(dtype="datetime64[ns]")[owner]
        keep = np.isnat(death) | (event <= death)
        if outside_windows:
            keep &= ~np.any([(event >= lo) & (event <= hi) for lo, hi in windows], axis=0)
        owner, event = owner[keep], event[keep]
        codes = [(c,) for c in rng.choice(np.array(code_pool), len(owner))]
        add(people["research_id"].to_numpy()[owner], etype, method,
            event if etype == "hospital_admission" else np.full(len(owner), NAT), event, codes)

    episodes = pd.concat(enc, ignore_index=True)
    episodes = episodes.sort_values(["research_id", "event_date"], kind="stable").reset_index(drop=True)
    episodes["encounter_id"] = [f"ENC{i:07d}" for i in range(len(episodes))]
    rows = episodes.explode("codes", ignore_index=True).rename(columns={"codes": "diagnosis_code"})
    rows["diagnosis_position"] = rows.groupby("encounter_id").cumcount().astype("int64") + 1
    rows["code_system"] = "ICD-10"
    rows["admission_method"] = rows["admission_method"].astype("str")

    positives = pos[["research_id", "index_date"]].assign(first=first)  # second episodes are never earlier
    merged = predictors[["research_id", "index_date"]].merge(positives, on=["research_id", "index_date"], how="left")
    first_event = pd.Series(merged["first"].to_numpy(dtype="datetime64[ns]"), index=predictors.index)
    return rows[list(TABLE_COLUMNS["encounters"])], first_event
