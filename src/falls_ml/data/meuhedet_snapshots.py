"""Snapshot audit and temporal design for a multi-snapshot Meuhedet extract (aggregate only).

One VIEW extract can hold many monthly ``Index_Date`` snapshots of the same members. This module

1. audits every snapshot (rows, patients, eligible, censored, invalid labels, events, prevalence, D-00 violations, 180-day
   outcome observability) and marks each one ``USABLE`` or ``EXCLUDED`` with the reason - observability is read from the VIEW's
   own label fields (``Has_Full_180D_Label``, ``Is_Censored_180D``, ``Fall_Next_180D_Ind``), never from calendar arithmetic alone;
   a data-freeze date, when given, is a cross-check that can only exclude more;
2. chooses the temporal train / validation / test boundaries from the usable snapshots and the outcome window: test = the latest
   snapshots, validation before it, train before that, with the D-19 outcome-window embargo simulated exactly as
   ``falls_ml.splitting`` applies it (a training row survives only if its outcome window ends before the first validation index
   date, a validation row only if its window ends before the first test index date). Boundaries are chosen from the data; nothing
   is hard-coded.

Every number is a count over the extract; no identifiers or row-level values appear.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from falls_ml.data.meuhedet_wide import WideContract, WideMapping
from falls_ml.errors import DatasetValidationError

USABLE, EXCLUDED = "USABLE", "EXCLUDED"
PARTITIONS = ("train", "validation", "test")


@dataclass
class SnapshotRow:
    index_date: str
    n_rows: int
    n_patients: int
    n_eligible: int
    n_eligible_patients: int
    n_labelled: int                 # eligible rows with a non-null label
    n_censored: int                 # eligible rows without a usable label (label NULL / Is_Censored = 1)
    n_label_invalid: int            # eligible labelled rows whose label is not 0/1 or contradicts the event date
    n_events: int
    prevalence: float | None        # among eligible labelled rows
    n_d00_violations: int           # eligible rows whose guard record dates fall on/after the index day
    n_full_window: int              # eligible rows the VIEW flags as having the full 180-day window
    n_labelled_without_full_window: int
    window_end: str
    status: str = USABLE
    reason: str = ""
    partition: str = ""             # train / validation / test / train (embargoed) / validation (embargoed) / "" when excluded
    n_modelling_rows: int | None = None   # eligible labelled rows the build would keep (after D-00 drop_rows)
    n_modelling_events: int | None = None


@dataclass
class TemporalDesign:
    train_end: str
    validation_end: str
    horizon_days: int
    embargo_days: int
    partition_by_snapshot: dict[str, str]
    expected: dict[str, dict[str, int]]       # per partition: snapshots, rows, events before and after the embargo
    embargo_removed: dict[str, dict[str, int]]
    rule: str = ""
    limitation_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def snapshot_audit(frame: pd.DataFrame, contract: WideContract, mapping: WideMapping, *, data_freeze_date: str | None = None,
                   max_censored_share: float = 0.5, index_day_records: str = "fail") -> tuple[list[SnapshotRow], dict[str, Any]]:
    """Audit every Index_Date of the extract and decide which snapshots carry a fully observable 180-day outcome."""
    idx_col, label_col = mapping.identity["index_date"], mapping.outcome["label_column"]
    id_col, ev_col, end_col = mapping.identity["research_id"], mapping.outcome["event_date_column"], mapping.outcome["window_end_column"]
    horizon = int(mapping.outcome["window_days_including_index"]) - 1
    full_col = f"Has_Full_{horizon}D_Label"
    cens_col = f"Is_Censored_{horizon}D"
    guard = [c for c in mapping.cohort["predictor_max_record_date"]["columns"] if c in frame.columns]
    idx = frame[idx_col].dt.normalize()
    eligible = frame["Is_Eligible_Cohort"].fillna(0) == 1
    y = frame[label_col]
    labelled = eligible & y.notna()
    censored = eligible & (y.isna() | ((frame[cens_col].fillna(0) == 1) if cens_col in frame.columns else False))
    y_num = pd.to_numeric(y, errors="coerce")
    ev = frame[ev_col].dt.normalize()
    invalid = labelled & (~y_num.isin([0, 1]) | ((y_num == 1) & ev.isna()) | ((y_num == 0) & ev.notna()))
    full = eligible & ((frame[full_col].fillna(0) == 1) if full_col in frame.columns else True)
    viol = pd.Series(False, index=frame.index)
    for col in guard:
        viol |= frame[col].dt.normalize() >= idx
    viol &= eligible
    freeze = pd.Timestamp(data_freeze_date) if data_freeze_date else None
    rows: list[SnapshotRow] = []
    for date in sorted(idx.dropna().unique()):
        m = idx == date
        n_lab, n_full = int((m & labelled).sum()), int((m & full).sum())
        n_lab_no_full = int((m & labelled & ~full).sum())
        n_elig = int((m & eligible).sum())
        n_cens = int((m & censored).sum())
        events = int((y_num[m & labelled] == 1).sum())
        window_end = pd.Timestamp(date) + pd.Timedelta(days=horizon)
        row = SnapshotRow(index_date=str(pd.Timestamp(date).date()), n_rows=int(m.sum()), n_patients=int(frame.loc[m, id_col].nunique()),
                          n_eligible=n_elig, n_eligible_patients=int(frame.loc[m & eligible, id_col].nunique()), n_labelled=n_lab, n_censored=n_cens,
                          n_label_invalid=int((m & invalid).sum()), n_events=events, prevalence=(events / n_lab if n_lab else None),
                          n_d00_violations=int((m & viol).sum()), n_full_window=n_full, n_labelled_without_full_window=n_lab_no_full,
                          window_end=str(window_end.date()))
        reasons = []
        if n_elig == 0:
            reasons.append("no eligible rows")
        elif n_lab == 0:
            reasons.append(f"no usable {horizon}-day label (every eligible row is censored / label NULL)")
        else:
            if full_col in frame.columns and n_lab_no_full:
                reasons.append(f"{n_lab_no_full} labelled rows carry {full_col} = 0: the VIEW marks the {horizon}-day window incomplete "
                               "(partially observed window - early events labelled, the rest censored)")
            share = n_cens / n_elig
            if share > max_censored_share:
                reasons.append(f"censored share {share:.1%} of eligible rows exceeds max_censored_share {max_censored_share:.0%} "
                               "(partially observable outcome window)")
            if freeze is not None and window_end > freeze:
                reasons.append(f"the {horizon}-day window ends {window_end.date()}, after the data freeze date {freeze.date()} "
                               "(calendar cross-check; verify --data-freeze-date if the VIEW flags the window complete)")
        if row.n_label_invalid:
            reasons.append(f"{row.n_label_invalid} labelled rows contradict the outcome definition (label not 0/1 or event date inconsistent)")
        row.status = EXCLUDED if reasons else USABLE
        row.reason = "; ".join(reasons)
        kept = n_lab - row.n_label_invalid
        dropped = m & viol & labelled if index_day_records == "drop_rows" else pd.Series(False, index=frame.index)
        row.n_modelling_rows = kept - int(dropped.sum())
        row.n_modelling_events = events - int((y_num[dropped] == 1).sum())
        rows.append(row)
    usable = [r for r in rows if r.status == USABLE]
    elig_all = frame.loc[eligible, id_col]
    totals = {"n_snapshots": len(rows), "n_usable_snapshots": len(usable), "excluded_snapshots": {r.index_date: r.reason for r in rows if r.status == EXCLUDED},
              "total_snapshot_rows": int(len(frame)), "unique_patients": int(frame[id_col].nunique()), "eligible_snapshot_rows": int(eligible.sum()),
              "unique_eligible_patients": int(elig_all.nunique()), "eligible_labelled_rows": int(labelled.sum()), "total_events": int((y_num[labelled] == 1).sum()),
              "event_prevalence_labelled": (float((y_num[labelled] == 1).mean()) if labelled.any() else None),
              "usable_rows": int(sum(r.n_modelling_rows or 0 for r in usable)), "usable_events": int(sum(r.n_modelling_events or 0 for r in usable)),
              "mean_snapshots_per_eligible_patient": (float(elig_all.value_counts().mean()) if len(elig_all) else None),
              "horizon_days": horizon, "observability_fields": [c for c in (label_col, f"Label_Reason_{horizon}D", cens_col, full_col) if c in frame.columns],
              "max_censored_share": max_censored_share, "data_freeze_date": data_freeze_date, "d00_policy": index_day_records}
    return rows, totals


def choose_temporal_design(rows: list[SnapshotRow], *, horizon_days: int, min_test_events: int = 10, min_validation_events: int = 10,
                           min_train_events: int = 20, test_share: float = 0.2) -> TemporalDesign:
    """Temporal boundaries over the USABLE snapshots (ascending): the test block is the latest snapshots holding at least
    ``test_share`` of the usable rows and ``min_test_events`` events; validation ends just before it; the train/validation boundary
    is the latest one whose validation block still holds ``min_validation_events`` after the outcome-window embargo (so training is
    as large as possible). Raises ``DatasetValidationError`` when no boundary satisfies the constraints, naming the arithmetic."""
    usable = sorted((r for r in rows if r.status == USABLE), key=lambda r: r.index_date)
    if len(usable) < 3:
        raise DatasetValidationError("The temporal design needs at least three usable snapshots (train, validation, test)",
                                     [f"usable snapshots: {[r.index_date for r in usable]}"])
    dates = [pd.Timestamp(r.index_date) for r in usable]
    n_rows = np.array([r.n_modelling_rows or 0 for r in usable])
    events = np.array([r.n_modelling_events if r.n_modelling_events is not None else r.n_events for r in usable])
    total = int(n_rows.sum())
    # test block: latest snapshots until the share and event floor are met (at least one snapshot, never everything)
    n_test = 1
    while n_test < len(usable) - 2 and (n_rows[-n_test:].sum() < test_share * total or events[-n_test:].sum() < min_test_events):
        n_test += 1
    if events[-n_test:].sum() < min_test_events:
        raise DatasetValidationError(f"INSUFFICIENT EVENTS for a temporal test block: the latest {n_test} usable snapshots hold "
                                     f"{int(events[-n_test:].sum())} events (< {min_test_events})", [f"events per snapshot: {dict(zip([r.index_date for r in usable], events.tolist()))}"])
    j = len(usable) - n_test          # index of the first test snapshot
    first_test = dates[j]
    window_end = [d + pd.Timedelta(days=horizon_days) for d in dates]
    chosen = None
    tried = []
    for k in range(j - 1, 0, -1):     # train = usable[:k], validation = usable[k:j]
        first_val = dates[k]
        val_keep = [i for i in range(k, j) if window_end[i] < first_test]
        train_keep = [i for i in range(0, k) if window_end[i] < first_val]
        val_ev, train_ev = int(events[val_keep].sum()), int(events[train_keep].sum())
        tried.append((usable[k - 1].index_date, val_ev, train_ev))
        if val_ev >= min_validation_events and train_ev >= min_train_events and val_keep and train_keep:
            chosen = (k, val_keep, train_keep)
            break
    if chosen is None:
        raise DatasetValidationError(
            f"No temporal boundary satisfies the design with a {horizon_days}-day outcome window: the D-19 embargo removes every training "
            f"snapshot whose window ends on/after the first validation index date and every validation snapshot whose window ends on/after "
            f"the first test index date ({first_test.date()}); required after embargo: validation >= {min_validation_events} events, "
            f"train >= {min_train_events} events",
            [f"train_end {te}: validation events after embargo {ve}, train events after embargo {tr}" for te, ve, tr in tried]
            + [f"usable snapshots {len(usable)} ({usable[0].index_date}..{usable[-1].index_date}); more snapshots or a shorter window are needed"])
    k, val_keep, train_keep = chosen
    train_end, validation_end = usable[k - 1].index_date, usable[j - 1].index_date
    partition: dict[str, str] = {}
    for i, r in enumerate(usable):
        if i >= j:
            partition[r.index_date] = "test"
        elif i >= k:
            partition[r.index_date] = "validation" if i in val_keep else "validation (embargoed)"
        else:
            partition[r.index_date] = "train" if i in train_keep else "train (embargoed)"
    for r in rows:
        r.partition = partition.get(r.index_date, "")
    expected = {"train": {"snapshots": k, "rows": int(n_rows[:k].sum()), "events": int(events[:k].sum()), "snapshots_after_embargo": len(train_keep),
                          "rows_after_embargo": int(n_rows[train_keep].sum()), "events_after_embargo": int(events[train_keep].sum())},
                "validation": {"snapshots": j - k, "rows": int(n_rows[k:j].sum()), "events": int(events[k:j].sum()), "snapshots_after_embargo": len(val_keep),
                               "rows_after_embargo": int(n_rows[val_keep].sum()), "events_after_embargo": int(events[val_keep].sum())},
                "test": {"snapshots": n_test, "rows": int(n_rows[j:].sum()), "events": int(events[j:].sum()), "snapshots_after_embargo": n_test,
                         "rows_after_embargo": int(n_rows[j:].sum()), "events_after_embargo": int(events[j:].sum())}}
    removed = {"train": {"snapshots": k - len(train_keep), "rows": int(n_rows[:k].sum() - n_rows[train_keep].sum())},
               "validation": {"snapshots": (j - k) - len(val_keep), "rows": int(n_rows[k:j].sum() - n_rows[val_keep].sum())}}
    rule = (f"test = the latest {n_test} usable snapshot(s) (>= {test_share:.0%} of usable rows and >= {min_test_events} events); validation = the "
            f"snapshots before it down to the latest train_end at which validation keeps >= {min_validation_events} events after the embargo; "
            f"train = everything earlier. Embargo: a train row survives only if index + {horizon_days}d < first validation index date "
            f"({dates[k].date()}); a validation row only if index + {horizon_days}d < first test index date ({first_test.date()}).")
    note = (f"Temporal repeated-risk design over {len(usable)} usable monthly snapshots: the same patient may appear in several partitions "
            f"(repeated risk prediction over time, not independent patients); the outcome-window embargo removed {removed['train']['snapshots']} "
            f"train and {removed['validation']['snapshots']} validation snapshot(s). A patient-disjoint hold-out is reported separately as a sensitivity analysis.")
    return TemporalDesign(train_end=train_end, validation_end=validation_end, horizon_days=horizon_days, embargo_days=horizon_days,
                          partition_by_snapshot=partition, expected=expected, embargo_removed=removed, rule=rule, limitation_note=note)


def snapshot_table(rows: list[SnapshotRow]) -> pd.DataFrame:
    return pd.DataFrame([asdict(r) for r in rows])


def render_snapshot_markdown(rows: list[SnapshotRow], totals: dict[str, Any], design: TemporalDesign | None) -> str:
    cols = ["index_date", "n_rows", "n_patients", "n_eligible", "n_labelled", "n_censored", "n_label_invalid", "n_events", "prevalence",
            "n_d00_violations", "n_full_window", "status", "partition", "reason"]
    lines = ["# Snapshot audit (aggregate, non-identifying)", "",
             f"Snapshots: {totals['n_snapshots']} ({totals['n_usable_snapshots']} usable). Total snapshot rows {totals['total_snapshot_rows']}; unique patients "
             f"{totals['unique_patients']}; eligible snapshot rows {totals['eligible_snapshot_rows']}; unique eligible patients {totals['unique_eligible_patients']}; "
             f"eligible labelled rows {totals['eligible_labelled_rows']}; total events {totals['total_events']}; event prevalence (labelled) "
             f"{(totals['event_prevalence_labelled'] or 0):.4f}; mean snapshots per eligible patient {(totals['mean_snapshots_per_eligible_patient'] or 0):.2f}.",
             f"Observability fields read from the VIEW: {totals['observability_fields']}; max censored share {totals['max_censored_share']}; "
             f"data freeze date {totals['data_freeze_date']}; D-00 policy {totals['d00_policy']}.", "",
             "| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        d = asdict(r)
        d["prevalence"] = "" if d["prevalence"] is None else f"{d['prevalence']:.4f}"
        lines.append("| " + " | ".join(str(d[c]).replace("|", "/") for c in cols) + " |")
    if design is not None:
        lines += ["", "## Temporal design", "", f"- train_end {design.train_end}; validation_end {design.validation_end}; test = index dates after validation_end",
                  f"- rule: {design.rule}", f"- expected rows/events per partition (before -> after embargo): {design.expected}",
                  f"- embargo removed: {design.embargo_removed}", f"- {design.limitation_note}"]
    return "\n".join(lines) + "\n"


def write_snapshot_audit(rows: list[SnapshotRow], totals: dict[str, Any], design: TemporalDesign | None, out_dir: str | Path) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"json": out / "snapshot_audit.json", "md": out / "snapshot_audit.md", "csv": out / "snapshot_audit.csv"}
    payload = {"totals": totals, "snapshots": [asdict(r) for r in rows], "temporal_design": design.to_dict() if design else None}
    paths["json"].write_text(json.dumps(payload, indent=2, default=str, ensure_ascii=False), encoding="utf-8", newline="\n")
    paths["md"].write_text(render_snapshot_markdown(rows, totals, design), encoding="utf-8", newline="\n")
    snapshot_table(rows).to_csv(paths["csv"], index=False, lineterminator="\n")
    return paths
