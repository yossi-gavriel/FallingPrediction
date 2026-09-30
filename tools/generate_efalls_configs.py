"""Generate the machine-readable eFalls configuration files from the parsed publication tables.

Inputs (verbatim extractions of Archer et al., Age Ageing 2024;53(3):afae057, CC BY 4.0):
  --coefficients  CSV parsed from Supplementary Table S3.2 (columns: group, term, coef, or_unpen, or_lo, or_hi)
  --predictors    JSON parsed from Supplementary Table S3.1 joined with eFI2 Appendix 1 rules
Outputs:
  configs/features/efalls_v1.yaml, configs/models/efalls_published.yaml, configs/mappings/meuhedet_v0.yaml

This script is a one-off provenance tool. The generated YAML files are the reviewed source of truth;
re-running it must reproduce them byte-for-byte (checked by tests/regression/test_config_provenance.py).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import yaml

CITATION = ("Archer L, Relton SD, Akbari A, et al. Development and external validation of the eFalls tool. "
            "Age Ageing 2024;53(3):afae057. doi:10.1093/ageing/afae057")

OUTCOME_CODES = [
    ("W00", "Fall on same level involving ice and snow"), ("W01", "Fall on same level from slipping, tripping and stumbling"),
    ("W02", "Fall involving ice-skates, skis, roller-skates or skateboards"),
    ("W03", "Other fall on same level due to collision with, or pushing by, another person"),
    ("W04", "Fall while being carried or supported by other persons"), ("W05", "Fall involving wheelchair"),
    ("W06", "Fall involving bed"), ("W07", "Fall involving chair"), ("W08", "Fall involving other furniture"),
    ("W09", "Fall involving playground equipment"), ("W10", "Fall on and from stairs and steps"),
    ("W11", "Fall on and from ladder"), ("W12", "Fall on and from scaffolding"),
    ("W13", "Fall from, out of or through building or structure"), ("W14", "Fall from tree"), ("W15", "Fall from cliff"),
    ("W16", "Diving or jumping into water causing injury other than drowning or submersion"),
    ("W17", "Other fall from one level to another"), ("W18", "Other fall on same level"), ("W19", "Unspecified fall"),
    ("M80", "Osteoporosis with pathological fracture"), ("S22", "Fracture of rib(s), sternum and thoracic spine"),
    ("S32", "Fracture of lumbar spine and pelvis"), ("S42", "Fracture of shoulder and upper arm"),
    ("S52", "Fracture of forearm"), ("S72", "Fracture of femur"), ("S82", "Fracture of lower leg, including ankle"),
    ("T08", "Fracture of spine, level unspecified"), ("T10", "Fracture of upper limb, level unspecified"),
    ("T12", "Fracture of lower limb, level unspecified"), ("T14.2", "Fracture of unspecified body region"),
]

NON_BINARY_TERMS = {
    ("", "Age (years)"): ("age_years", None),
    ("Polypharmacy", "ln((Polypharmacy+1)/10)"): ("polypharmacy_count_120d", None),
    ("Gender", "Female"): ("sex", "female"),
    ("BMI category", "Underweight"): ("bmi_category", "underweight"),
    ("BMI category", "Normal"): ("bmi_category", "normal"),
    ("BMI category", "Obese"): ("bmi_category", "obese"),
    ("BMI category", "Missing"): ("bmi_category", "missing"),
    ("Smoking", "Current"): ("smoking", "current"),
    ("Alcohol consumption", "Harmful drinking"): ("alcohol_category", "harmful"),
    ("Alcohol consumption", "Higher risk drinking"): ("alcohol_category", "higher_risk"),
    ("Alcohol consumption", "Previous higher risk/harmful drinking"): ("alcohol_category", "previous_higher_risk_or_harmful"),
    ("Alcohol consumption", "Zero alcohol"): ("alcohol_category", "zero"),
    ("Alcohol consumption", "Missing"): ("alcohol_category", "missing"),
}


def slug(name: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", name.lower())).strip("_")


# Predictors whose SAIL counts in eFI2 Table 2 equal eFalls Table S3.1 exactly (spec D-08 (a)).
COUNT_MATCHED = {
    "activity_limitation", "atrial_fibrillation", "cancer", "cognitive_impairment", "copd", "dementia",
    "dressing_and_grooming_problems", "environment_problems", "falls", "fracture", "fragility_fracture", "heart_failure",
    "housebound", "liver_problems", "medication_management", "memory_concerns", "mobility_problems",
    "motor_neurone_disease", "palliative_care", "parkinsonism_and_tremor", "peptic_ulcer_disease",
    "peripheral_vascular_disease", "requirement_for_care", "respiratory_disease", "seizures", "self_harm", "skin_ulcer",
    "stroke", "transient_ischaemic_attack", "weight_loss",
}
COUNT_MISMATCH = {"hypotension_or_syncope"}
NO_RULE_TEXT = {"hypertension", "shopping_problems"}
# Measurement (non-code) rules from the development team's implementation documents (S5/S10); applicability to eFalls UNRESOLVED (Q-02).
NON_CODE_RULES = {
    "hypertension": "code OR >= 3 readings ever with SBP >= 140 or DBP >= 90 (systolic/diastolic not mixed)",
    "hypotension_or_syncope": "code OR >= 3 readings ever with SBP < 90 or DBP < 60",
    "anaemia_and_haematinic_deficiency": "code OR Hb below sex-specific threshold (M < 13.0, F < 11.5 g/dL); resolved by a later normal Hb",
    "activity_limitation": "code OR Barthel index <= 18",
    "cognitive_impairment": "code OR 6CIT >= 8",
    "osteoporosis": "code OR DEXA T-score < -2.5",
    "peripheral_vascular_disease": "code OR ABPI < 0.95",
    "chronic_kidney_disease": "code OR eGFR < 60 or abnormal ACR/protein result",
    "thyroid_problems": "code OR TSH < 0.36 or > 5.5 mU/L",
}
SAIL_N = 660417
LOW_SUPPORT_THRESHOLD = 500


def parse_count(text: str) -> tuple[int | None, bool]:
    """Parse 'n (pct)' prevalence strings; '<10' is suppressed (returns None, True)."""
    head = text.split("(")[0].strip().replace(",", "")
    if head.startswith("<"):
        return None, True
    return int(head), False


def parse_rule(rule: str) -> dict:
    """Translate an eFI2 Appendix 1 rule symbol string into a structured, explicitly-labelled proxy rule (D-08)."""
    out: dict = {"proxy_source": "eFI2 Appendix 1 (Best et al. Age Ageing 2025;54:afaf077) — INFERRED proxy for eFalls (spec D-08)",
                 "proxy_rule_text": rule, "window": {"type": "complete_history"}, "age_at_record_min": None,
                 "resolved_by": [], "rule_fidelity": "full"}
    if rule.startswith("NOT LISTED"):
        out["rule_fidelity"] = "none"
        return out
    all_5y = "all 5y" in rule and not rule.startswith("‡")
    some_5y = rule.startswith("‡") or rule.startswith("5y except")
    if all_5y:
        out["window"] = {"type": "lookback_years", "years": 5}
    elif some_5y:
        # code-level exceptions require the official code list (Q-02); primary uses complete history (D-08)
        out["window"] = {"type": "complete_history"}
        out["rule_fidelity"] = "partial"
    if "age>=55" in rule or rule.startswith("α") or "‡α" in rule:
        out["age_at_record_min"] = 55 if "home visit" not in rule else None
        if "home visit" in rule:
            out["rule_fidelity"] = "partial"
    if rule.startswith("* age>=18"):
        out["age_at_record_min"] = 18
        if "lone AF" in rule:
            out["rule_fidelity"] = "partial"
    if "resolves if dementia" in rule:
        out["resolved_by"] = ["dementia"]
    if "resolves if cognitive impairment" in rule:
        out["resolved_by"] = ["cognitive_impairment", "dementia"]
    return out


def _time_window_for(name: str, rule: str) -> dict:
    tw = parse_rule(rule)
    if name in COUNT_MATCHED:
        tw["rule_evidence"] = "count_matched"
    elif name in COUNT_MISMATCH:
        tw.update({"rule_evidence": "count_mismatch", "window": {"type": "complete_history"}, "age_at_record_min": None,
                   "resolved_by": [], "rule_fidelity": "none"})
    else:
        tw["rule_evidence"] = "not_checkable"
    if name in NO_RULE_TEXT:
        tw.update({"window": {"type": "complete_history"}, "rule_fidelity": "none"})
    tw["non_code_rule"] = NON_CODE_RULES.get(name)
    if tw["non_code_rule"]:
        tw["non_code_rule_status"] = "IMPL S5/S10; primary codes+measurements; applicability to eFalls UNRESOLVED (Q-02, D-08)"
    return tw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coefficients", required=True)
    ap.add_argument("--predictors", required=True)
    ap.add_argument("--project-root", required=True)
    a = ap.parse_args()
    root = Path(a.project_root)
    with open(a.coefficients, encoding="utf-8", newline="") as fh:
        coef_rows = list(csv.DictReader(fh))
    coef = {(r["group"], r["term"]): r for r in coef_rows}
    with open(a.predictors, encoding="utf-8") as fh:
        binaries = json.load(fh)

    features: list[dict] = []
    common_meu = {"mapping_file": "configs/mappings/meuhedet_v0.yaml", "status": "TO_BE_MAPPED", "clinically_validated": False}
    features.append({
        "name": "age_years", "concept": "Decimal age at index date: (index_date - date_of_birth) / 365.25 (D-06)", "role": "predictor", "dtype": "float",
        "layer": "L1_published", "efalls_term": "Age (years)", "exact_efalls_baseline": True,
        "efalls_source": "Date of birth (SAIL: week of birth, approximate; continuous vs completed years not stated, Q-06)",
        "time_window": {"type": "at_index_date"},
        "valid_range": {"min": 65.0, "max": 120.0},
        "transformation": {"published": "linear (decimal years; no cap, D-06)", "retrained": "fractional_polynomial_selection (FP2 max; D-13)"},
        "missing_rule": "forbid", "reference_level": None, "decisions": ["D-06", "D-13"],
        "meuhedet_source": dict(common_meu), "clinically_validated": False,
    })
    features.append({
        "name": "sex", "concept": "Recorded sex", "role": "predictor", "dtype": "categorical",
        "levels": ["female", "male"], "layer": "L1_published", "efalls_term": "Gender", "exact_efalls_baseline": True,
        "efalls_source": "Registered sex (valid sex required for inclusion)", "time_window": {"type": "at_index_date"},
        "transformation": {"published": "sex term per sex_parameterisation (D-01)", "retrained": "all-levels indicators (D-10)"},
        "missing_rule": "forbid", "reference_level": "PUB-CONFLICT: see configs/models/efalls_published.yaml sex_parameterisations (D-01)",
        "decisions": ["D-01", "D-10"], "meuhedet_source": dict(common_meu), "clinically_validated": False,
    })
    features.append({
        "name": "polypharmacy_count_120d",
        "concept": "Number of unique BNF paragraphs ('sub-sub-chapters') with a drug prescribed in the 120 days before index, excluding non-drug chapters",
        "role": "predictor", "dtype": "count", "layer": "L1_published", "efalls_term": "ln((Polypharmacy+1)/10)",
        "exact_efalls_baseline": True,
        "efalls_source": "UK primary-care prescriptions coded to BNF; unit = unique BNF sub-sub-chapter (Box S3.1 footnote)",
        "time_window": {"type": "lookback_days", "days": 120, "boundary": "index_date - 120 days <= record_date <= index_date - 1 day (D-00)", "fidelity": "published"},
        "valid_range": {"min": 0, "max": None},
        "transformation": {"published": "log((P + 1) / 10)", "retrained": "fractional_polynomial_selection (FP2 max; Stata auto-scaling; D-13)"},
        "missing_rule": "absent_is_zero", "reference_level": None,
        "sensitivity_variants": ["window_90_days", "unit_bnf_subparagraph_7digit", "unit_chemical_substance"],
        "decisions": ["D-03"], "meuhedet_source": {**common_meu, "candidate": "ATC5+route -> BNF paragraph crosswalk from dispensing records (M-04)"},
        "clinically_validated": False,
    })
    features.append({
        "name": "bmi_value", "concept": "Most recent valid body-mass index (kg/m2) within look-back, categorised",
        "role": "predictor", "dtype": "float_nullable", "layer": "L1_published", "efalls_term": "BMI category",
        "exact_efalls_baseline": True, "efalls_source": "Recorded BMI (window not published)",
        "time_window": {"type": "lookback_years", "years": 5, "record_rule": "most recent valid value; precedence recorded value > computed weight/height (within 30 days) > unambiguous category code; same-day mean", "fidelity": "assumed (third-party S10 window; D-02)"},
        "valid_range": {"min": 10.0, "max": 80.0, "out_of_range": "treat_as_not_recorded", "status": "requires clinical approval (M-06)"},
        "transformation": {"published": {"categorise": {"underweight": "< 18.5", "normal": "18.5 <= bmi < 25", "overweight": "25 <= bmi < 30", "obese": ">= 30"},
                                           "note": "WHO cut-offs (D-02); printed 'obese >= 40' is erratum E-04"},
                           "retrained": "same categories, all-levels indicators (D-10)"},
        "levels": ["underweight", "normal", "overweight", "obese", "missing"],
        "missing_rule": "missing_category", "reference_level": "overweight", "decisions": ["D-02"],
        "meuhedet_source": dict(common_meu), "clinically_validated": False,
    })
    features.append({
        "name": "smoking_status", "concept": "Most recent recorded smoking status", "role": "predictor", "dtype": "categorical_nullable",
        "levels": ["never", "ex", "current"], "layer": "L1_published", "efalls_term": "Smoking", "exact_efalls_baseline": True,
        "efalls_source": "Smoking status codes (derivation not published)",
        "time_window": {"type": "complete_history", "record_rule": "most recent status before index; 'never' after an earlier smoker/ex code counts as ex; ex followed by current is current", "fidelity": "assumed (D-04)"},
        "transformation": {"published": "current -> 'current'; never/ex/missing -> 'ex_never' (reference)", "retrained": "all-levels indicators never/ex/current; missing merged into never (D-04, D-10)"},
        "missing_rule": "reference_level", "reference_level": "ex_never", "decisions": ["D-04"],
        "meuhedet_source": dict(common_meu), "clinically_validated": False,
    })
    features.append({
        "name": "alcohol_category", "concept": "Alcohol consumption category (highest-risk record within look-back)", "role": "predictor",
        "dtype": "categorical_nullable",
        "levels": ["harmful", "higher_risk", "lower_risk", "previous_higher_risk_or_harmful", "zero"],
        "layer": "L1_published", "efalls_term": "Alcohol consumption", "exact_efalls_baseline": True,
        "efalls_source": "Alcohol consumption codes/values (definitions not published)",
        "time_window": {"type": "lookback_years", "years": 5, "record_rule": "precedence harmful > higher_risk > previous_higher_risk_or_harmful > lower_risk > zero; units/week 0 zero, 1-20 lower, 21-48 higher, >=49 harmful; coded category wins same-day conflicts", "fidelity": "assumed (IMPL S5 / third-party S10; D-05)"},
        "transformation": {"published": "indicators vs reference lower_risk; null -> 'missing' level", "retrained": "all-levels indicators incl. missing (D-10)"},
        "missing_rule": "missing_category", "reference_level": "lower_risk", "decisions": ["D-05"],
        "meuhedet_source": dict(common_meu), "clinically_validated": False,
    })
    for r in binaries:
        name = slug(r["name"])
        features.append({
            "name": name, "concept": r["name"], "role": "predictor", "dtype": "binary", "layer": "L1_published",
            "efalls_term": r["name"], "exact_efalls_baseline": True,
            "retained_in_published_model": r["retained"] == "yes",
            "efalls_source": "UK primary-care SNOMED CT/CTV3/Read v2 code group; official code list on request (Q-02); unofficial proxy: DynAIRx Baseline2 group (IP unclear)",
            "efalls_origin": r["origin"],
            "time_window": _time_window_for(slug(r["name"]), r["rule"]),
            "transformation": {"published": "indicator (0/1)", "retrained": "indicator (0/1)"},
            "missing_rule": "absent_is_zero", "reference_level": 0, "decisions": ["D-08"],
            "published_prevalence": {"sail_development": r["sail"], "connected_bradford": r["cb"],
                                     "sail_count": parse_count(r["sail"])[0], "sail_count_suppressed_lt10": parse_count(r["sail"])[1],
                                     "sail_proportion": (parse_count(r["sail"])[0] / SAIL_N) if parse_count(r["sail"])[0] is not None else 5.0 / SAIL_N},
            "development_support": ("low" if r["retained"] == "yes" and (parse_count(r["sail"])[0] or 0) < LOW_SUPPORT_THRESHOLD else "adequate"),
            "meuhedet_source": dict(common_meu), "clinically_validated": False,
        })

    spec = {
        "feature_set": {"name": "efalls_v1", "version": "1.0.0", "layer": "L1_published",
                        "specification": "docs/EFALLS_REPRODUCTION_SPEC.md", "citation": CITATION,
                        "n_candidate_features": len(features)},
        "identifiers": {"research_id": {"dtype": "string", "role": "identifier", "never_predictor": True},
                        "index_date": {"dtype": "date", "role": "index"}},
        "provenance_columns": {"predictor_max_record_date": {"dtype": "date", "rule": "< index_date (D-00: predictors use records strictly before the index date)"},
                               "outcome_first_event_date": {"dtype": "date_nullable", "rule": "non-null iff outcome_12m == 1; index_date <= date <= index_date + 1 year - 1 day (D-00)"},
                               "dataset_version": {"dtype": "string"}, "mapping_version": {"dtype": "string"}, "source": {"dtype": "string"}},
        "metadata_columns": {"practice_id": {"dtype": "string_nullable", "role": "cluster"},
                             "deprivation_group": {"dtype": "string_nullable", "role": "cluster"},
                             "site_id": {"dtype": "string_nullable", "role": "cluster"},
                             "death_date": {"dtype": "date_nullable", "role": "metadata"},
                             "followup_end_date": {"dtype": "date_nullable", "role": "metadata"}},
        "cohort": {"age_min": 65.0, "one_row_per": ["research_id", "index_date"],
                   "mandatory_for_efalls_label": ["age_years", "sex", "polypharmacy_count_120d", "falls", "fracture", "fragility_fracture", "dementia"],
                   "coverage_threshold": 0.90},
        "outcome": {
            "name": "outcome_12m", "dtype": "binary", "layer": "L1_published",
            "concept": "One or more emergency department attendances or hospital admissions with a fall or fracture within 12 months after index",
            "window": {"start": "index_date (inclusive)", "end": "index_date + 1 calendar year - 1 day (inclusive)", "horizon_years": 1, "decision": "D-00, D-09"},
            "encounter_types": ["emergency_department_attendance", "non_elective_hospital_admission"],
            "encounter_rule": "ED attendance or non-elective admission (D-09); exclude spells started before index; sensitivity: any admission type",
            "diagnosis_position": "any (fracture codes and W00-W19); sensitivity: fracture in principal position or W00-W19 any position (D-09)",
            "icd_variant_rule": "declared per source; ICD-10-CM keeps 7th characters A/B/C only (D-09, M-02)",
            "deaths": "retained as non-events (published)",
            "code_system": "ICD-10 (WHO)",
            "codes": [{"code": c, "description": d, "match": "exact_or_subcode" if "." in c else "category_prefix"} for c, d in OUTCOME_CODES],
            "meuhedet_source": {**common_meu, "note": "ICD-9-CM + ICD-10 mapping required; E-code prefix collision risk (M-02)"},
            "clinically_validated": False,
        },
        "features": features,
    }

    # ---- published coefficients -------------------------------------------------------------
    c = lambda g, t: float(coef[(g, t)]["coef"])  # noqa: E731
    const = c("", "Constant")
    female = c("Gender", "Female")
    binary_coefs = {}
    for r in binaries:
        key = slug(r["name"])
        binary_coefs[key] = float(coef[("", r["name"])]["coef"]) if ("", r["name"]) in coef else 0.0
    unpen = {}
    for (g, t), row in coef.items():
        if row["or_unpen"]:
            unpen[f"{g or 'term'}::{t}"] = {"odds_ratio": float(row["or_unpen"]), "ci_low": float(row["or_lo"]), "ci_high": float(row["or_hi"])}
    published = {
        "model": {"name": "efalls_published", "layer": "L1_published", "citation": CITATION,
                  "coefficient_source": "Supplementary Table S3.2 'Final penalised model' (identical to HTA GJAC1008 Table 15)",
                  "licence": "Publication CC BY 4.0; equation available for research use; commercial/NHS/supplier use requires licence (spec §3.2)",
                  "specification": "docs/EFALLS_REPRODUCTION_SPEC.md §5.10, §9.1"},
        "link": "logit",
        "terms": {
            "age_years": {"transform": "identity", "coefficient": c("", "Age (years)")},
            "polypharmacy_count_120d": {"transform": "log((P + 1) / 10)", "coefficient": c("Polypharmacy", "ln((Polypharmacy+1)/10)")},
            "bmi_category": {"reference": "overweight", "levels": {k2: c("BMI category", t) for (g, t), (f, k2) in NON_BINARY_TERMS.items() if f == "bmi_category"}},
            "smoking": {"reference": "ex_never", "levels": {"current": c("Smoking", "Current")}},
            "alcohol_category": {"reference": "lower_risk", "levels": {k2: c("Alcohol consumption", t) for (g, t), (f, k2) in NON_BINARY_TERMS.items() if f == "alcohol_category"}},
            "binary": binary_coefs,
        },
        "published_constant_as_printed": const,
        "published_female_coefficient_as_printed": female,
        "sex_parameterisations": {
            "lp_c_box_s3_1": {"intercept_female": round(const + female, 7), "intercept_male": round(const + 2 * female, 7),
                              "role": "primary: published as validated (D-01; TRIPOD 8d/12b)", "source": "Supplementary Box S3.1 read literally: intercept -6.258 (female), '-0.304 (if male)'"},
            "lp_a_table_s3_2": {"intercept_female": round(const + female, 7), "intercept_male": const,
                                "role": "mandatory co-reported sensitivity: published as tabulated (D-01)", "source": "Supplementary Table S3.2 as printed: Male reference, Female -0.303708, Constant -5.954459"},
            "lp_b_label_swap": {"intercept_female": const, "intercept_male": round(const + female, 7),
                                "role": "CITL-shift variant of lp_c (+0.303708 for everyone) (D-01)", "source": "Table S3.2 with sex labels swapped"},
            "lp_d2_numeric_swap": {"intercept_female": round(const + 2 * female, 7), "intercept_male": round(const + female, 7),
                                   "role": "CITL-shift variant of lp_a (-0.303708 for everyone) (D-01)", "source": "sex entered as numeric 1/2 with the other coding"},
        },
        "unpenalised_refit_odds_ratios_for_reference_only": unpen,
        "development_distribution_sail": {
            "use": "M-11 coverage metric approximation only (sum of beta^2 * variance per term, covariances ignored)",
            "source": "S1 Table 1 (n = 660,417) and S3 l.556-560 (mean age 74.9, SD 7.5)",
            "n": SAIL_N,
            "age_years_sd": 7.5,
            "polypharmacy_log_term_sd_approx": round((math.log(10 / 10) - math.log(1 / 10)) / 1.349, 6),
            "polypharmacy_log_term_sd_note": "normal approximation from polypharmacy IQR 0-9: (ln(10/10) - ln(1/10)) / 1.349",
            "proportions": {
                "sex=female": round(348675 / SAIL_N, 6),
                "bmi_category=underweight": round(12642 / SAIL_N, 6), "bmi_category=normal": round(121946 / SAIL_N, 6),
                "bmi_category=obese": round(136646 / SAIL_N, 6), "bmi_category=missing": round(230507 / SAIL_N, 6),
                "smoking=current": round(86806 / SAIL_N, 6),
                "alcohol_category=harmful": round(4714 / SAIL_N, 6), "alcohol_category=higher_risk": round(686 / SAIL_N, 6),
                "alcohol_category=previous_higher_risk_or_harmful": round(90 / SAIL_N, 6), "alcohol_category=zero": round(1247 / SAIL_N, 6),
                "alcohol_category=missing": round(642449 / SAIL_N, 6),
            },
        },
        "recalibration_reference_connected_bradford": {
            "alpha": -0.423, "beta": 1.25, "applies_to": "lp_c_box_s3_1", "use": "regression tests only; never applied to Meuhedet (D-07)",
            "source": "Supplementary Box S3.2 (OMML) = HTA Table 34; HTA Table 35 rejected (E-09)"},
        "box_s3_1_rounded_example": {
            "use": "regression test RT-01 only", "intercept": -6.258, "age": 0.042, "log_polypharmacy": 0.330,
            "underweight": 0.490, "previous_higher_risk_or_harmful": 0.085, "dementia": 0.104, "liver_problems": 0.380, "osteoporosis": 0.128,
            "patient": {"sex": "female", "age_years": 89, "polypharmacy_count_120d": 8},
            "expected_lp_printed": -1.368, "expected_probability_printed": 0.203, "expected_recalibrated_probability_printed": 0.106},
    }

    # ---- Meuhedet mapping placeholder (L3) ---------------------------------------------------
    mapping = {
        "mapping": {"name": "meuhedet", "version": "0.0.0-unmapped", "layer": "L3a_meuhedet_mapping",
                    "status": "TO_BE_MAPPED — no feature may be used for a reportable scientific run until clinically_validated is true (spec §11, B-01, B-02)"},
        "outcome": {"source_tables": "UNKNOWN — TO DISCOVER", "code_systems": ["ICD-9-CM", "ICD-10"], "status": "TO_BE_MAPPED", "clinically_validated": False},
        "features": {f["name"]: {"source_table": "UNKNOWN — TO DISCOVER", "source_field": "UNKNOWN — TO DISCOVER", "code_system": "UNKNOWN — TO DISCOVER",
                                 "code_list": None, "source_class": None, "mapping_status": "TO_BE_MAPPED", "clinically_validated": False, "validated_by": None,
                                 "validation_date": None, "notes": ""} for f in features},
    }

    header = "# GENERATED by tools/generate_efalls_configs.py from verbatim publication tables — edit only via reviewed change (spec §16).\n"
    for rel, obj in (("configs/features/efalls_v1.yaml", spec), ("configs/models/efalls_published.yaml", published), ("configs/mappings/meuhedet_v0.yaml", mapping)):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        # newline="\n": the committed configs are LF and compared byte-for-byte (tests/regression/test_config_provenance.py);
        # without it Windows text mode would write CRLF.
        p.write_text(header + yaml.safe_dump(obj, sort_keys=False, allow_unicode=True, width=160), encoding="utf-8", newline="\n")
        print("wrote", p)


if __name__ == "__main__":
    main()
