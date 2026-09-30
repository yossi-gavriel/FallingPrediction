"""Generate the modelling-dataset column template ``data/example_schema.csv`` FROM THE CODE.

Sources of truth (nothing is typed by hand here):
  * predictor names, dtypes, levels, ranges, layers, eFalls terms, time windows, decisions: the feature spec
    ``configs/features/efalls_v1.yaml`` loaded with :func:`falls_ml.features.spec.load_feature_spec`;
  * which columns are required in training and scoring: :func:`falls_ml.data.schema.validate_modeling_dataset`
    itself (it is asked which columns are missing from an empty frame);
  * how nulls are handled per representation: the constants of :mod:`falls_ml.features.preprocessing`.

Outputs (UTF-8, LF newlines, deterministic):
  data/example_schema.csv        one row per column of the modelling dataset
  data/example_header_only.csv   header row of the minimum columns of an extract for ``build-dataset`` (no data)

Usage:
  python tools/generate_data_schema.py            # (re)write both files and print the feature-spec identity
  python tools/generate_data_schema.py --check    # exit 1 if a checked-in file differs or data/README.md is stale
"""
from __future__ import annotations

import argparse
import ast
import csv
import io
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SPEC_REL = "configs/features/efalls_v1.yaml"
SCHEMA_REL = "data/example_schema.csv"
HEADER_REL = "data/example_header_only.csv"
README_REL = "data/README.md"
COLUMNS = ("column", "role", "required_for_training", "required_for_scoring", "dtype", "allowed_values",
           "missing_value_behaviour", "efalls_term", "layer", "exact_efalls_baseline", "retained_in_published_model",
           "time_window", "notes")
#: spec dtype -> storage type accepted by the validator (parquet dtype; CSV spelling for build-dataset in brackets)
DTYPE_TEXT = {
    "string": "string", "string_nullable": "string, nullable",
    "date": "date (datetime64; CSV YYYY-MM-DD)", "date_nullable": "date, nullable (datetime64; CSV YYYY-MM-DD or empty)",
    "binary": "integer 0/1 (int8/int64 or bool)", "count": "integer (int64)", "float": "float64",
    "float_nullable": "float64, nullable", "categorical": "string", "categorical_nullable": "string, nullable",
}


def _bool(v: Any) -> str:
    return "" if v is None else ("true" if v else "false")


def _flatten(value: Any, prefix: str = "") -> list[str]:
    """Deterministic ``key=value`` rendering of a nested YAML mapping (YAML order; None/empty skipped; lists joined by '|')."""
    if isinstance(value, Mapping):
        out: list[str] = []
        for k, v in value.items():
            out += _flatten(v, f"{prefix}{k}.")
        return out
    if value is None or value == [] or value == "":
        return []
    text = "|".join(map(str, value)) if isinstance(value, (list, tuple)) else str(value)
    return [f"{prefix[:-1]}={text}" if prefix else text]


def required_columns(spec: Any, mode: str) -> list[str]:
    """Columns the validator requires in ``mode``, read from the validator's own error on an empty frame."""
    import pandas as pd

    from falls_ml.data.schema import validate_modeling_dataset
    from falls_ml.errors import DatasetValidationError

    prefix = "missing required columns: "
    try:
        validate_modeling_dataset(pd.DataFrame(), spec, mode=mode)
    except DatasetValidationError as exc:
        hits = [p for p in exc.problems if p.startswith(prefix)]
        if len(hits) != 1:
            raise RuntimeError(f"validator message format changed; cannot read required columns from {exc.problems!r}") from exc
        return list(ast.literal_eval(hits[0][len(prefix):]))
    raise RuntimeError("validate_modeling_dataset accepted an empty frame; the generator's introspection is broken")


def _range_text(f: Any, spec: Any) -> str:
    vr = f.valid_range or {}
    lo, hi = vr.get("min"), vr.get("max")
    if f.name == "age_years":  # same rule as falls_ml.data.schema._check_feature
        lo = max(spec.age_min, lo if lo is not None else spec.age_min)
    if lo is None and hi is None:
        return ""
    return f"{'' if lo is None else lo}..{'unbounded' if hi is None else hi}"


def _missing_text(f: Any) -> str:
    """Validation rule plus preprocessing per representation (mirrors falls_ml.features.preprocessing)."""
    from falls_ml.features import preprocessing as pp
    from falls_ml.features.transforms import MISSING_LEVEL

    if f.dtype == "binary":
        return ("validation: null rejected - absence of a qualifying record must be written as 0 (absent_is_zero); "
                "preprocessing: value used as-is in every representation")
    if f.missing_rule == "forbid":
        return "validation: null rejected; preprocessing: never imputed (every representation)"
    if f.name == pp.POLYPHARMACY:
        return ("validation: null rejected - no prescription in the window must be written as 0 (absent_is_zero); "
                "negative counts rejected; preprocessing: never imputed (every representation)")
    if f.name == pp.BMI:
        return ("validation: null allowed = no valid BMI in the window (values outside the range must be set to null "
                f"before building, they are rejected); preprocessing: published: null -> '{MISSING_LEVEL}' BMI category "
                f"(published coefficient vs reference {pp.BMI_REFERENCE}); retrained: '{MISSING_LEVEL}' is one of the "
                f"all-levels indicators; alternative: '{MISSING_LEVEL}' indicator (reference-coded vs {pp.BMI_REFERENCE}, "
                "or all-levels for tree models)")
    if f.name == pp.SMOKING:
        return ("validation: null allowed = no smoking record; preprocessing: published: null -> ex_never reference "
                "(only 'current' carries a coefficient); retrained: null merged into 'never' (all-levels never|ex|current); "
                "alternative: null merged into 'never' (reference-coded vs never, or all-levels for tree models)")
    if f.name == pp.ALCOHOL:
        return (f"validation: null allowed = no alcohol record; preprocessing: published: null -> '{MISSING_LEVEL}' level "
                f"(published coefficient vs reference {pp.ALCOHOL_REFERENCE}); retrained: '{MISSING_LEVEL}' all-levels "
                f"indicator; alternative: '{MISSING_LEVEL}' indicator (reference-coded vs {pp.ALCOHOL_REFERENCE}, or "
                "all-levels for tree models)")
    if f.dtype == "float_nullable":
        return (f"validation: null allowed; preprocessing: training-median imputation plus {f.name}__missing indicator "
                "(not accepted by the published representation)")
    if f.dtype == "categorical_nullable" and f.missing_rule == "reference_level":
        return f"validation: null allowed; preprocessing: null -> reference level {f.reference_level!r}"
    if f.dtype == "categorical_nullable":
        return f"validation: null allowed; preprocessing: null -> '{MISSING_LEVEL}' indicator"
    raise RuntimeError(f"{f.name}: no documented missing-value rule for dtype {f.dtype!r} / {f.missing_rule!r}")


def _notes(f: Any) -> str:
    raw = f.raw
    parts = [str(raw.get("concept", "")).rstrip(".")]
    if raw.get("decisions"):
        parts.append("decisions: " + ", ".join(map(str, raw["decisions"])))
    tr = raw.get("transformation")
    if isinstance(tr, Mapping):
        parts += [f"transformation {k}: " + "; ".join(_flatten(v)) for k, v in tr.items()]
    if isinstance(f.reference_level, str) and f.reference_level:
        parts.append(f"reference level: {f.reference_level}")
    if f.name == "bmi_value" and f.levels:
        parts.append("derived categories: " + "|".join(f.levels))
    ms = raw.get("meuhedet_source") or {}
    if ms.get("status"):
        parts.append(f"Meuhedet mapping status: {ms['status']}")
    return ". ".join(p for p in parts if p)


def build_rows(spec: Any) -> list[dict[str, str]]:
    """One row per modelling-dataset column (identifiers, index, provenance, outcome, versions, predictors, metadata)."""
    from falls_ml.data.schema import OUTCOME_EVENT_DATE, PREDICTOR_MAX_DATE, VERSION_COLUMNS

    training, scoring = required_columns(spec, "training"), required_columns(spec, "inference")
    yaml_ids = spec.raw_payloads[0]["identifiers"]
    prov, meta = spec.provenance_columns, spec.metadata_columns
    known_prov = {PREDICTOR_MAX_DATE, OUTCOME_EVENT_DATE, *VERSION_COLUMNS}
    unknown = sorted(set(prov) - known_prov)
    if unknown:
        raise RuntimeError(f"provenance columns without a validation rule in falls_ml.data.schema: {unknown}")
    horizon = spec.outcome.horizon_years
    window = f"index_date..index_date + {horizon} year{'s' if horizon != 1 else ''} - 1 day"
    rows: list[dict[str, str]] = []

    def add(column: str, role: str, dtype: str, allowed: str, missing: str, notes: str, **extra: str) -> None:
        if dtype not in DTYPE_TEXT:
            raise RuntimeError(f"{column}: unknown dtype {dtype!r}")
        rows.append({"column": column, "role": role, "required_for_training": "yes" if column in training else "no",
                     "required_for_scoring": "yes" if column in scoring else "no", "dtype": DTYPE_TEXT[dtype],
                     "allowed_values": allowed, "missing_value_behaviour": missing, "efalls_term": extra.get("efalls_term", ""),
                     "layer": extra.get("layer", ""), "exact_efalls_baseline": extra.get("exact_efalls_baseline", ""),
                     "retained_in_published_model": extra.get("retained_in_published_model", ""),
                     "time_window": extra.get("time_window", ""), "notes": notes})

    for c in spec.identifier_columns:
        add(c, "identifier", yaml_ids[c]["dtype"], f"non-null; unique together with {spec.index_column}",
            "validation: null rejected in training; optional when scoring (echoed in predictions when present)",
            "pseudonymised research identifier; never a predictor; one row per research_id x index_date (D-15)")
    add(spec.index_column, "index_date", yaml_ids[spec.index_column]["dtype"], "valid calendar date",
        "validation: null rejected",
        "prediction is made at the start of the index day (D-00); predictors use records strictly before this date")
    add(PREDICTOR_MAX_DATE, "predictor_provenance", prov[PREDICTOR_MAX_DATE]["dtype"], f"< {spec.index_column}",
        "validation: must be a date strictly before index_date; any row on or after index_date is rejected as LEAKAGE",
        f"latest record date used by any predictor of the row. rule: {prov[PREDICTOR_MAX_DATE].get('rule', '')}")
    out = spec.outcome.raw
    add(spec.outcome.name, "outcome", out.get("dtype", "binary"), "0|1",
        "validation: null rejected in training; not used when scoring",
        f"{out.get('concept', '')}. decisions: {out.get('window', {}).get('decision', '')}",
        layer=str(out.get("layer", "")), time_window="; ".join(_flatten(out.get("window", {}))))
    add(OUTCOME_EVENT_DATE, "outcome_provenance", prov[OUTCOME_EVENT_DATE]["dtype"], window,
        f"validation: required (non-null) when {spec.outcome.name} = 1 and must be null when {spec.outcome.name} = 0; "
        "dates outside the window are rejected",
        f"date of the first qualifying outcome event. rule: {prov[OUTCOME_EVENT_DATE].get('rule', '')}",
        time_window=window)
    for c in VERSION_COLUMNS:
        add(c, "version", prov[c]["dtype"], "exactly one constant value per dataset",
            "validation: null rejected; must equal the manifest",
            f"written by build-dataset from --{c.replace('_', '-')}; optional in an extract (if present it must equal that argument)")
    for f in spec.features:
        allowed = "0|1" if f.is_binary else ("|".join(f.levels) if f.is_categorical else _range_text(f, spec))
        add(f.name, "predictor", f.dtype, allowed, _missing_text(f), _notes(f), efalls_term=f.efalls_term or "",
            layer=f.layer, exact_efalls_baseline=_bool(f.exact_efalls_baseline),
            retained_in_published_model=_bool(f.retained_in_published_model),
            time_window="; ".join(_flatten(f.raw.get("time_window", {}))))
    for c, d in meta.items():
        add(c, "optional_metadata", d["dtype"], "", "validation: null allowed; not a predictor",
            f"role {d.get('role', 'metadata')}; only needed when an experiment config names it (e.g. a cluster column "
            "for heterogeneity or IECV); never used as a predictor")
    missing = [c for c in training + scoring if c not in {r["column"] for r in rows}]
    if missing:
        raise RuntimeError(f"validator requires columns the template does not describe: {missing}")
    return rows


def _csv_text(rows: list[list[str]]) -> str:
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerows(rows)
    return buf.getvalue()


def render_schema_csv(spec: Any) -> str:
    rows = build_rows(spec)
    return _csv_text([list(COLUMNS), *[[r[c] for c in COLUMNS] for r in rows]])


def render_header_csv(spec: Any) -> str:
    """Minimum extract columns for build-dataset: training-required columns except the version columns it writes."""
    from falls_ml.data.schema import VERSION_COLUMNS

    return _csv_text([[c for c in required_columns(spec, "training") if c not in VERSION_COLUMNS]])


def load_spec(root: Path = ROOT) -> Any:
    from falls_ml.features.spec import load_feature_spec

    return load_feature_spec(root / SPEC_REL)


def outputs(root: Path = ROOT, spec: Any = None) -> dict[str, bytes]:
    spec = spec if spec is not None else load_spec(root)
    return {SCHEMA_REL: render_schema_csv(spec).encode("utf-8"), HEADER_REL: render_header_csv(spec).encode("utf-8")}


def check(root: Path = ROOT) -> list[str]:
    """Problems with the checked-in files (empty list = up to date)."""
    spec = load_spec(root)
    problems = []
    for rel, data in outputs(root, spec).items():
        path = root / rel
        if not path.is_file():
            problems.append(f"{rel}: missing")
        elif path.read_bytes() != data:
            problems.append(f"{rel}: differs from the generator output (run python tools/generate_data_schema.py)")
    readme = root / README_REL
    if not readme.is_file() or spec.content_sha256 not in readme.read_text(encoding="utf-8"):
        problems.append(f"{README_REL}: does not quote the current feature-spec sha256 {spec.content_sha256}")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="verify the checked-in files instead of writing them")
    ap.add_argument("--root", type=Path, default=ROOT, help="project root (default: this repository)")
    a = ap.parse_args(argv)
    spec = load_spec(a.root)
    print(f"feature spec: name={spec.name} version={spec.version} content_sha256={spec.content_sha256}")
    if a.check:
        problems = check(a.root)
        for p in problems:
            print(f"STALE: {p}", file=sys.stderr)
        print("data schema template: " + ("STALE" if problems else "up to date"))
        return 1 if problems else 0
    for rel, data in outputs(a.root, spec).items():
        (a.root / rel).parent.mkdir(parents=True, exist_ok=True)
        (a.root / rel).write_bytes(data)
        print(f"wrote {rel} ({len(data.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
