"""The frozen Phase 3 predictors rebuilt on the 2026 snapshot with the UNCHANGED Phase 1-3 code - nothing re-chosen, nothing re-coded.

    canonical BASELINE_15 values   ``MeuhedetWideDatasetAdapter._feature`` (the code Phase 3 verified against the reference build)
    Phase 2 catalogue values       ``phase2.engineer.compute`` (frozen catalogue)
    KNOWN / UNKNOWN cells          ``phase3.recovery.build_recovery`` under the corrected END-OF-INDEX-DAY contract on the 2026 index date, with the
                                   attestation re-checked on 2026 (check V3: no predictor record dated after 2026-01-01)
    predictions                    ``phase3.bounds.linear_intervals``: KNOWN rows get the ordinary prediction; a row with an UNKNOWN used cell
                                   gets the exact interval over the pre-index states observed among the 2025 fitting rows (never imputed)

Only the features of the frozen models are built: a 2026-only column can never reach a frozen model (it is not an input of any frozen feature).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from falls_ml.phase2.state import Phase2Stop
from falls_ml.phase4.sealed import UNREADABLE_PREFIX, SealedRead

ID_COLS = ("Customer_Full_ID", "Snapshot_Key", "Index_Date", "Is_Eligible_Cohort")
QA_COLS = ("Leakage_Check_Ind", "Definition_Version")


@dataclass
class ModelSpec:
    config: str            # e.g. LASSO:P3_BASE
    setname: str
    new: list[str]
    baseline: list[str]

    @property
    def features(self) -> list[str]:
        return [*self.new, *self.baseline]


def model_specs(fs: Any, configs: list[str]) -> dict[str, ModelSpec]:
    out = {}
    for c in configs:
        s = c.split(":", 1)[1]
        if s not in fs.sets:
            raise Phase2Stop("FROZEN_SET_MISSING", f"{c}: the frozen Phase 3 feature sets have no set {s}")
        d = fs.sets[s]
        out[c] = ModelSpec(config=c, setname=s, new=list(d["features"]), baseline=list(d["baseline"]))
    return out


def union(specs: dict[str, ModelSpec]) -> tuple[list[str], list[str]]:
    new: list[str] = []
    base: list[str] = []
    for s in specs.values():
        new += [f for f in s.new if f not in new]
        base += [b for b in s.baseline if b not in base]
    return new, base


def feature_inputs(name: str, cat: Any, mapping: Any) -> list[str]:
    """Raw extract columns one frozen feature reads (value + validation inputs)."""
    from falls_ml.data.meuhedet_wide import feature_input_columns

    if name in cat.names:
        return list(cat.get(name).inputs)
    for f in mapping.features:
        if f.canonical == name:
            i = feature_input_columns(f)
            return [*i["value"], *i["validation"]]
    raise Phase2Stop("FROZEN_FEATURE_UNKNOWN", f"{name}: not a catalogue feature nor a baseline predictor of the mapping")


def needed_columns(specs: dict[str, ModelSpec], *, cat: Any, mapping: Any, dictionary: Any, d00: Any, tc: Any, header: list[str],
                   sealed: dict[str, str]) -> dict[str, Any]:
    """Every column the blind scoring reads, and every required input that is absent / sealed (both STOP)."""
    from falls_ml.phase3.timecontract import predictor_date_columns

    new, base = union(specs)
    inputs: dict[str, list[str]] = {f: feature_inputs(f, cat, mapping) for f in [*new, *base]}
    cols: list[str] = [c for c in ID_COLS if c in header]
    absent, sealed_inputs, rec_dates, absent_dates = [], [], {}, []
    for f, ins in inputs.items():
        for c in ins:
            if c not in header:
                absent.append((f, c))
            elif c in sealed:
                sealed_inputs.append((f, c, sealed[c]))
            elif c not in cols:
                cols.append(c)
            rd = dictionary.record_date(c) if c in dictionary.columns else None
            if rd:
                rec_dates.setdefault(f, set()).add(rd)
                if rd not in header:
                    absent_dates.append((f, rd))
                elif rd not in sealed and rd not in cols:
                    cols.append(rd)
    v3 = [c for c in predictor_date_columns(dictionary, d00, tc, header)]
    for c in v3:
        if c not in sealed and c not in cols:
            cols.append(c)
    for c in QA_COLS:
        if c in header and c not in sealed and c not in cols:
            cols.append(c)
    missing_ids = [c for c in ID_COLS if c not in header]
    return {"columns": cols, "inputs": inputs, "absent_inputs": absent, "sealed_inputs": sealed_inputs, "missing_id_columns": missing_ids,
            "record_dates": {k: sorted(v) for k, v in rec_dates.items()}, "absent_record_dates": sorted(set(absent_dates)),
            "v3_date_columns": [c for c in v3 if c not in sealed], "v3_sealed_date_columns": [c for c in v3 if c in sealed]}


@dataclass
class Built2026:
    frame: pd.DataFrame                 # eligible index-date rows (raw contract-typed inputs + canonical baseline + unreadable masks)
    values: pd.DataFrame                # catalogue values of the frozen features (UNKNOWN cells NaN, as Phase 3 S04)
    unknown: pd.DataFrame               # per feature UNKNOWN mask
    upper_bound: pd.DataFrame
    registry: pd.DataFrame              # 2026 eligibility class of every frozen feature (corrected contract)
    v3: dict[str, Any]
    adapter_problems: list[str] = field(default_factory=list)
    eligible_mask: np.ndarray | None = None


def eligible_mask(frame: pd.DataFrame, index_date: str) -> np.ndarray:
    idx = pd.Timestamp(index_date)
    return ((frame["Index_Date"].dt.normalize() == idx).fillna(False) & (pd.to_numeric(frame["Is_Eligible_Cohort"], errors="coerce") == 1).fillna(False)).to_numpy()


def v3_check(frame: pd.DataFrame, date_cols: list[str], index_date: str) -> dict[str, Any]:
    """Phase 3 check V3 on the 2026 eligible rows (record dates only - no outcome): attestation holds when no predictor record is dated after
    the index date."""
    idx = pd.Timestamp(index_date)
    per, total = {}, np.zeros(len(frame), dtype=bool)
    for c in date_cols:
        if c not in frame.columns or not pd.api.types.is_datetime64_any_dtype(frame[c]):
            continue
        off = (frame[c].dt.normalize() - idx).dt.days
        after = (off > 0).fillna(False).to_numpy()
        per[c] = {"n_after_index": int(after.sum()), "n_on_index": int((off == 0).fillna(False).sum()),
                  "max_days_after": int(off[after].max()) if after.any() else 0}
        total |= after
    return {"passed": int(total.sum()) == 0, "rows_with_any_post_index_record": int(total.sum()), "by_column": per, "columns_checked": sorted(per)}


def build_2026(read: SealedRead, specs: dict[str, ModelSpec], *, cat: Any, mapping: Any, contract: Any, espec: Any, dictionary: Any, d00: Any, rules: Any,
               tc: Any, index_date: str, work_dtypes: dict[str, Any], v3_columns: list[str], inputs: dict[str, list[str]]) -> Built2026:
    from falls_ml.data.meuhedet_wide import MeuhedetWideDatasetAdapter
    from falls_ml.phase2.engineer import compute
    from falls_ml.phase3.cohort import canonical_baseline
    from falls_ml.phase3.recovery import build_recovery

    new, base = union(specs)
    m = eligible_mask(read.frame, index_date)
    frame = read.frame.loc[m].reset_index(drop=True)
    unr = read.unreadable.loc[m].reset_index(drop=True) if read.unreadable is not None else pd.DataFrame(index=frame.index)
    for c in unr.columns:
        frame[UNREADABLE_PREFIX + c] = unr[c].to_numpy(dtype=bool)
    adapter = MeuhedetWideDatasetAdapter(mapping, contract, espec, features=base)
    canon, problems = canonical_baseline(adapter, mapping, frame, base)
    for b in base:
        frame[b] = canon[b].to_numpy()
        if b in work_dtypes:
            try:
                frame[b] = frame[b].astype(work_dtypes[b])
            except (TypeError, ValueError):
                problems.append(f"{b}: 2026 values cannot take the 2025 dtype {work_dtypes[b]}")
    values = pd.DataFrame({f: compute(cat.get(f), frame).to_numpy() for f in cat.names if f in set(new)}, index=frame.index)
    v3 = v3_check(frame, v3_columns, index_date)
    cat2 = dataclasses.replace(cat, features=tuple(f for f in cat.features if f.name in set(new)))
    rec = build_recovery(frame, values, catalogue=cat2, mapping=mapping, baseline=base, contract=contract, dictionary=dictionary, d00=d00, rules=rules,
                         train_mask=np.ones(len(frame), dtype=bool), min_observed_train=1, tc=tc, attestation_holds=bool(v3["passed"]))
    unknown = rec.unknown.copy()
    # an unreadable / not-allowed cell of ANY input is never read as a value (Phase 3 does this for row-level sources; Phase 4 extends it to every
    # input of a frozen feature): the row is UNKNOWN for that feature and is evaluated with bounds
    for f in [*new, *base]:
        extra = np.zeros(len(frame), dtype=bool)
        for c in inputs.get(f, []):
            if UNREADABLE_PREFIX + c in frame.columns:
                extra |= frame[UNREADABLE_PREFIX + c].to_numpy(dtype=bool)
        if extra.any():
            unknown[f] = (unknown[f].to_numpy(dtype=bool) | extra) if f in unknown.columns else extra
    for f in [c for c in unknown.columns if c in values.columns]:
        values.loc[unknown[f].to_numpy(dtype=bool), f] = np.nan
    return Built2026(frame=frame, values=values, unknown=unknown, upper_bound=rec.upper_bound, registry=rec.registry, v3=v3, adapter_problems=problems,
                     eligible_mask=m)


def scoring_frame(b: Built2026) -> pd.DataFrame:
    dup = [c for c in b.values.columns if c in b.frame.columns]
    return pd.concat([b.frame.drop(columns=dup), b.values], axis=1)


def predict_intervals(fit: Any, spec: ModelSpec, b: Built2026, fit_frame: pd.DataFrame, feature_source: dict[str, str]) -> Any:
    from falls_ml.phase3.bounds import linear_intervals, plan_unknown

    ev = scoring_frame(b)
    unk = b.unknown.reindex(columns=[c for c in b.unknown.columns], fill_value=False)
    plan = plan_unknown(spec.features, unk, b.upper_bound, feature_source)
    return linear_intervals(fit, ev, fit_frame, plan)
