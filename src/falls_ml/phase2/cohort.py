"""S01: rebuild the reference D00_CLEAN cohort from the same extract, prove it identical (rows and predictor values), reproduce the reference
split (test-row sha256 must equal the reference), map every extract row to its partition, and DROP THE TEST ROWS immediately (only their
count and hash are kept). Nothing here reads a test outcome; no file is written outside the run folder; the pepper is read, never created."""

from __future__ import annotations

import hashlib
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from falls_ml.phase2.state import Phase2Stop

IDENTIFIER_COLUMNS = ("Customer_Full_ID", "Snapshot_Key")
PARTS = ("train", "validation")


@dataclass
class CohortResult:
    frame: pd.DataFrame                     # TRAIN + VALIDATION rows only: research_id, index_date, partition, y, canonical baseline, raw columns
    facts: dict[str, Any] = field(default_factory=dict)
    profile: pd.DataFrame | None = None     # aggregate per-column profile of all contract columns (TRAIN+VALIDATION and TRAIN)
    ref: dict[str, Any] = field(default_factory=dict)


def read_reference_pepper(src: Path, ref_dir: Path, pepper_file: str | Path | None, ref_pepper_sha: str | None) -> tuple[str, dict[str, Any]]:
    """The reference run's pseudonymisation pepper: --id-pepper-file > <reference>/id_pepper.txt > <input>.id_pepper.txt. Never created."""
    candidates = [Path(pepper_file)] if pepper_file else [ref_dir / "id_pepper.txt", src.with_name(src.name + ".id_pepper.txt")]
    for path in candidates:
        if path.is_file():
            lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]
            if lines:
                pepper = lines[0]
                sha = hashlib.sha256(pepper.encode("utf-8")).hexdigest()
                if ref_pepper_sha and sha != ref_pepper_sha:
                    raise Phase2Stop("PEPPER_MISMATCH", "the pseudonymisation pepper differs from the reference run's (research ids and the split would differ)",
                                     [f"pepper file: {path.name}", "pass --id-pepper-file <reference explore folder>\\id_pepper.txt"])
                return pepper, {"sha256": sha, "source": path.name}
    raise Phase2Stop("PEPPER_MISSING", "the reference pseudonymisation pepper was not found (Phase 2 never creates a new one)",
                     ["looked for: " + ", ".join(p.name for p in candidates), "pass --id-pepper-file <reference explore folder>\\id_pepper.txt"])


OUTCOME_WINDOW_DAYS = 180
LABEL_REASON_COLUMN = "Label_Reason_180D"


def death_label_audit(tv: pd.DataFrame, label_column: str, index_column: str, *, min_cell: int = 10) -> dict[str, Any]:
    """Aggregate check of the outcome's death semantics on TRAIN + VALIDATION rows only (Astra F-05): a recorded fall/fracture inside the
    window must stay an event when death follows; death before any recorded event is a non-event. If members who died inside the window
    never carry an event, the warehouse label may overwrite prior events with death (a different, composite endpoint)."""
    y = pd.to_numeric(tv[label_column], errors="coerce")
    out: dict[str, Any] = {"rows": int(len(tv)), "window_days": OUTCOME_WINDOW_DAYS, "available": False}
    if "Is_Deceased_Ind" not in tv or "Death_Censor_Date" not in tv:
        return out
    dec = pd.to_numeric(tv["Is_Deceased_Ind"], errors="coerce") == 1
    days = (pd.to_datetime(tv["Death_Censor_Date"], errors="coerce").dt.normalize() - pd.to_datetime(tv[index_column]).dt.normalize()).dt.days
    inside = dec & days.between(0, OUTCOME_WINDOW_DAYS)
    ev = int((inside & (y == 1)).sum())
    ne = int((inside & (y == 0)).sum())
    sup = lambda n: n if n == 0 or n >= min_cell else f"<{min_cell}"  # noqa: E731
    reasons = {}
    if LABEL_REASON_COLUMN in tv:
        for k, n in tv[LABEL_REASON_COLUMN].astype("string").fillna("NULL").value_counts().items():
            reasons[str(k)] = sup(int(n))
    out.update({"available": True, "deaths_after_index": int(dec.sum()), "deaths_within_window": int(inside.sum()),
                "events_among_deaths_within_window": sup(ev), "non_events_among_deaths_within_window": sup(ne),
                "label_missing_among_deaths_within_window": sup(int((inside & y.isna()).sum())),
                "label_reason_180d_counts": reasons,
                "events_kept_when_death_follows": bool(ev > 0) if int(inside.sum()) else None,
                "overwrite_suspected": bool(int(inside.sum()) >= 50 and ev == 0)})
    return out


def build_cohort(src: Path, ref_dir: Path, *, contract: Any, mapping: Any, spec: Any, pepper_file: str | Path | None, scratch: Path,
                 read: Any | None = None, input_sha: str | None = None) -> CohortResult:
    from falls_ml.config import load_experiment_config
    from falls_ml.data.dataset import ModelingDataset, sha256_file
    from falls_ml.data.meuhedet_wide import build_meuhedet_dataset_from_frame, read_wide_extract_report
    from falls_ml.meuhedet_sensitivity import _differing_columns, _keys, _Stop, load_reference
    from falls_ml.splitting import make_split

    try:
        ref = load_reference(ref_dir)
    except _Stop as stop:
        raise Phase2Stop("REFERENCE_INVALID", stop.reason, stop.details) from None
    input_sha = input_sha or sha256_file(src)
    ref_input = ref["build"].get("input_sha256")
    if ref_input and ref_input != input_sha:
        raise Phase2Stop("INPUT_NOT_REFERENCE", "the input file is not the file the reference run used (sha256 differs)",
                         [f"reference {ref_input[:16]}…", f"this file {input_sha[:16]}…"])
    pepper, pinfo = read_reference_pepper(src, ref_dir, pepper_file, ref.get("pepper_sha256"))
    if read is None:
        read = read_wide_extract_report(src, contract)
    frame = read.frame
    index_date = str(ref["index_date"])
    features = list(ref["features"]["extended"])
    policy = "drop_rows" if int(ref["build"].get("n_rows_timing_violation") or 0) else None

    # ---- rebuild the reference cohort (EXTENDED definitions) and prove it identical
    with tempfile.TemporaryDirectory(prefix=".cohort_rebuild_", dir=scratch) as tmp:
        ds_dir = Path(tmp) / "extended"
        manifest, build = build_meuhedet_dataset_from_frame(
            frame, ds_dir, index_date=index_date, dataset_version=f"phase2-{index_date}", data_freeze_date=None, mapping=mapping, contract=contract,
            spec=spec, input_name=src.name, input_sha256=input_sha, wrong_type=read.wrong_type, wrong_type_examples=read.wrong_type_examples,
            date_formats=read.date_formats, features=features, feature_set_label="extended", id_pepper=pepper, index_day_records=policy,
            read_report=read.to_dict(), notes="Phase 2 cohort reconstruction (temporary)")
        ds = ModelingDataset.load(ds_dir, spec, features=features)
        canon = ds.frame.copy()
        dspec = ds.spec
    ref_frame = pd.read_parquet(ref["dir"] / "datasets" / "extended" / "modeling_dataset.parquet")
    keys, ref_keys = set(_keys(canon, dspec)), set(_keys(ref_frame, dspec))
    if keys != ref_keys:
        raise Phase2Stop("COHORT_MISMATCH", "the rebuilt D00_CLEAN cohort differs from the reference cohort",
                         [f"reference {len(ref_keys)} rows", f"rebuilt {len(keys)} rows", f"only in reference {len(ref_keys - keys)}", f"only rebuilt {len(keys - ref_keys)}"])
    differing = _differing_columns(ref_frame, canon, dspec, features)
    if differing:
        raise Phase2Stop("COHORT_VALUES_DIFFER", "rebuilt predictor values differ from the reference dataset", differing)

    # ---- reproduce the reference split
    cfg = load_experiment_config(ref["configs"]["extended"])
    plan = make_split(canon, dspec, cfg.validation)
    test_sha = plan.test_rows_sha256(canon, dspec)
    ref_test = ref["split_audit"].get("test_rows_sha256")
    if test_sha != ref_test:
        raise Phase2Stop("SPLIT_MISMATCH", "the reproduced split does not give the reference test partition (sha256 differs)",
                         [f"reference {str(ref_test)[:16]}…", f"reproduced {test_sha[:16]}…"])
    assign = plan.assignments(canon, dspec)
    id_col, idx_col, y_col = dspec.identifier_columns[0], dspec.index_column, dspec.outcome.name
    assign_keys = assign[id_col].astype(str) + "|" + pd.to_datetime(assign[idx_col]).dt.strftime("%Y-%m-%d")
    splits_csv = ref["runs"]["extended"] / "splits.csv"
    splits_checked = False
    if splits_csv.exists():
        stored = pd.read_csv(splits_csv, dtype={id_col: str})
        a = dict(zip(assign_keys, assign["split"].astype(str)))
        b = dict(zip(stored[id_col].astype(str) + "|" + pd.to_datetime(stored[idx_col]).dt.strftime("%Y-%m-%d"), stored["split"].astype(str)))
        if a != b:
            raise Phase2Stop("SPLIT_MISMATCH", "the reproduced assignment differs from the reference run's splits.csv")
        splits_checked = True
    canon = canon.assign(_key=_keys(canon, dspec).to_numpy())
    part_of = dict(zip(assign_keys, assign["split"].astype(str)))
    canon["partition"] = canon["_key"].map(part_of)
    if canon["partition"].isna().any():
        raise Phase2Stop("SPLIT_MISMATCH", "some cohort rows have no partition in the reproduced split")
    # patients never cross partitions
    by_patient = canon.groupby(id_col)["partition"].nunique()
    if int((by_patient > 1).sum()):
        raise Phase2Stop("PATIENT_OVERLAP", f"{int((by_patient > 1).sum())} patients appear in more than one partition")
    n_test_rows = int((canon["partition"] == "test").sum())
    n_test_patients = int(canon.loc[canon["partition"] == "test", id_col].nunique())
    # ---- DROP TEST (in memory, before anything else touches the rows)
    canon = canon.loc[canon["partition"] != "test"].copy()

    # ---- map extract rows -> canonical rows (TRAIN + VALIDATION)
    ids = frame[mapping.identity["research_id"]].astype("string")
    pseudo = ids.map(lambda v: hashlib.sha256(f"{pepper}|{v}".encode("utf-8")).hexdigest()[:20] if pd.notna(v) else None)
    idx = frame[mapping.identity["index_date"]].dt.normalize()
    on = (idx == pd.Timestamp(index_date)).fillna(False) & (frame["Is_Eligible_Cohort"] == 1).fillna(False)
    raw_key = (pseudo.astype("string") + "|" + idx.dt.strftime("%Y-%m-%d")).where(on)
    keep_keys = set(canon["_key"])
    rows = raw_key.isin(keep_keys).fillna(False).to_numpy()
    raw = frame.loc[rows].copy()
    raw["_key"] = raw_key[rows].to_numpy()
    if raw["_key"].duplicated().any() or len(raw) != len(canon):
        raise Phase2Stop("ROW_MAPPING", "could not map every TRAIN/VALIDATION row to exactly one extract row",
                         [f"mapped {len(raw)}", f"expected {len(canon)}", f"duplicated keys {int(raw['_key'].duplicated().sum())}"])
    label_audit = death_label_audit(frame.loc[rows], mapping.outcome["label_column"], mapping.identity["index_date"])
    label_cols = [c.name for c in contract.columns if c.role == "LABEL"]
    drop = [c for c in (*IDENTIFIER_COLUMNS, *label_cols) if c in raw.columns]
    raw = raw.drop(columns=drop)
    canon_cols = [id_col, idx_col, "partition", y_col, *features]
    work = canon[[*canon_cols, "_key"]].merge(raw, on="_key", how="left", validate="one_to_one")
    y_raw = frame.loc[rows].set_index(raw_key[rows].to_numpy())[mapping.outcome["label_column"]].reindex(work["_key"]).to_numpy()
    if not np.array_equal(pd.to_numeric(pd.Series(y_raw)).to_numpy(dtype=float), work[y_col].to_numpy(dtype=float)):
        raise Phase2Stop("LABEL_MISMATCH", "the canonical outcome differs from the extract's label on some TRAIN/VALIDATION rows")
    work = work.rename(columns={id_col: "research_id", idx_col: "index_date", y_col: "y"})
    work["y"] = work["y"].astype(int)
    order = np.lexsort((work["research_id"].astype(str).to_numpy(), (work["partition"] != "train").to_numpy()))
    work = work.iloc[order].drop(columns=["_key"]).reset_index(drop=True)
    work.insert(0, "row_id", np.arange(len(work), dtype=np.int64))

    # ---- aggregate profile of every contract column on TRAIN+VALIDATION rows (never a value)
    prof_rows = []
    tv = frame.loc[rows]
    tr_mask = (tv.assign(_k=raw_key[rows].to_numpy())["_k"].map(part_of) == "train").to_numpy()
    for c in contract.columns:
        s = tv[c.name]
        s_tr = s[tr_mask]
        nn = s.dropna()
        n_unique = int(nn.nunique())
        top_share = float(nn.value_counts(normalize=True).iloc[0]) if len(nn) else None
        prof_rows.append({"column": c.name, "n_rows": int(len(s)), "n_observed": int(s.notna().sum()), "missing_pct": round(100.0 * float(s.isna().mean()), 3),
                          "n_observed_train": int(s_tr.notna().sum()), "unique_count": n_unique,
                          "constant_status": "EMPTY" if len(nn) == 0 else "CONSTANT" if n_unique == 1 else
                          "NEAR_CONSTANT" if top_share is not None and top_share >= 0.995 else "VARIABLE",
                          "top_value_share": None if top_share is None else round(top_share, 4)})
    profile = pd.DataFrame(prof_rows)
    def keys_sha(part: str) -> str:
        m = work["partition"] == part
        keys = work.loc[m, "research_id"].astype(str) + "|" + pd.to_datetime(work.loc[m, "index_date"]).dt.strftime("%Y-%m-%d")
        return hashlib.sha256("\n".join(sorted(keys.tolist())).encode("utf-8")).hexdigest()

    counts = {p: {"n_rows": int((work["partition"] == p).sum()), "n_patients": int(work.loc[work["partition"] == p, "research_id"].nunique()),
                  "n_events": int(work.loc[work["partition"] == p, "y"].sum())} for p in PARTS}
    facts = {"index_date": index_date, "input_sha256": input_sha, "pepper_sha256": pinfo["sha256"], "pepper_source": pinfo["source"],
             "reference_dataset_sha256": ref["manifests"]["extended"].get("data_sha256"), "rebuilt_dataset_sha256": manifest.data_sha256,
             "d00_policy": policy or "none", "cohort_rows": int(len(keys)), "cohort_equal_reference": True, "predictor_values_equal_reference": True,
             "split_strategy": plan.strategy, "split_seed": int(cfg.validation.seed), "test_rows_sha256": test_sha, "test_equal_reference": True,
             "train_rows_sha256": keys_sha("train"), "validation_rows_sha256": keys_sha("validation"),
             "splits_csv_checked": splits_checked, "test_rows_dropped": n_test_rows, "test_patients_dropped": n_test_patients,
             "test_outcomes_read": False, "partitions": counts, "baseline_features": features, "synthetic": bool(ref["build"].get("synthetic")),
             "label_death_audit": label_audit,
             "reference_features_strict": list(ref["features"]["strict"])}
    return CohortResult(frame=work, facts=facts, profile=profile, ref=ref)
