"""The Phase 3 time contract (configs/meuhedet/phase3_time_contract.yaml) and its READ-ONLY verification on the extract.

Prediction at the END of Index_Date (predictors: records dated <= Index_Date); outcome from Index_Date + 1 day. The DWH developer's explanation is
accepted only as far as the data confirm it:
  V1 (HARD)          no positive label with an event on/before Index_Date;
  V2 (HARD)          Label_End_180D - Index_Date = 180 days;
  V3 (ATTESTATION)   no predictor record date after Index_Date - otherwise those rows are UNKNOWN and sources without a row-level date lose
                     the attestation (UNRESOLVED);
  V4 (WARN)          Days_Since_* consistent with their dates;
  V5 (INFO)          index-day fall x label;
  V6 (INVESTIGATION) boundary episodes (TRAIN only).
Only TRAIN + VALIDATION rows are ever passed here: TEST labels are never read. Aggregate counts only."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from falls_ml.errors import ConfigError
from falls_ml.paths import resolve_path

DEFAULT_TIME_CONTRACT = "configs/meuhedet/phase3_time_contract.yaml"
END_OF_DAY = "END_OF_INDEX_DAY"
RULE = "source_event_date <= Index_Date"


@dataclass(frozen=True)
class TimeContract:
    path: str
    sha256: str
    raw: dict[str, Any]
    excluded_date_sources: dict[str, str]
    documented_as_of: dict[str, str]
    attested_kinds: tuple[str, ...]
    checks: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def header(self) -> dict[str, Any]:
        return self.raw["phase3_time_contract"]


def load_time_contract(path: str | Path = DEFAULT_TIME_CONTRACT, *, contract: Any = None, dictionary: Any = None) -> TimeContract:
    p = resolve_path(path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    h = raw.get("phase3_time_contract") or {}
    problems = []
    if h.get("prediction_time") != END_OF_DAY or h.get("predictor_record_rule") != RULE:
        problems.append(f"prediction_time must be {END_OF_DAY} with '{RULE}' (the corrected contract)")
    for k in ("V1_label_excludes_index_day", "V2_window_end", "V3_no_post_index_records", "V6_boundary_episode"):
        if k not in (h.get("checks") or {}):
            problems.append(f"check {k} missing")
    ex = {str(k): str(v) for k, v in (h.get("excluded_date_sources") or {}).items()}
    doc = {str(k): str(v) for k, v in (h.get("documented_as_of_columns") or {}).items()}
    if dictionary is not None:
        problems += [f"excluded_date_sources: unknown source {s}" for s in ex if s not in dictionary.sources]
    if contract is not None:
        problems += [f"documented_as_of_columns: unknown column {c}" for c in doc if c not in set(contract.names)]
    if problems:
        raise ConfigError(f"{p}: invalid Phase 3 time contract: " + "; ".join(problems))
    return TimeContract(path=str(p), sha256=hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest(), raw=raw, excluded_date_sources=ex,
                        documented_as_of=doc, attested_kinds=tuple(h.get("attested_evidence_kinds") or ()), checks=dict(h.get("checks") or {}))


def predictor_date_columns(dictionary: Any, d00: Any, tc: TimeContract, columns: list[str]) -> list[str]:
    """Record-date columns of the predictor sources (dictionary sources and per-column record dates), minus the documented excluded sources."""
    out = set()
    for src, meta in dictionary.sources.items():
        if d00.source_evidence.get(src) in ("COMPLETE_RECORD_DATE", "SUFFICIENT_ONLY_RECORD_DATE") and src not in tc.excluded_date_sources and meta.get("record_date"):
            out.add(meta["record_date"])
    for c, d in dictionary.columns.items():
        rd = d.get("record_date")
        if rd and d["source"] not in tc.excluded_date_sources and d00.source_evidence.get(d["source"]) in ("COMPLETE_RECORD_DATE", "SUFFICIENT_ONLY_RECORD_DATE"):
            out.add(rd)
    return sorted(c for c in out if c in columns)


def _sup(n: int, min_cell: int = 10) -> int | str:
    return n if n == 0 or n >= min_cell else f"<{min_cell}"


def availability_of(tc: TimeContract, column: str, source: str) -> dict[str, str]:
    """The declared information-availability risk and assumption of one column (column override > source > UNKNOWN)."""
    av = tc.header.get("availability") or {}
    col = (av.get("columns") or {}).get(column)
    if col:
        return {"risk": str(col["risk"]), "assumption": str(col["assumption"]), "level": "column"}
    src = (av.get("sources") or {}).get(source)
    if src:
        return {"risk": str(src["risk"]), "assumption": str(src["assumption"]), "level": "source"}
    return {"risk": "UNKNOWN", "assumption": "no availability declaration for this source", "level": "none"}


RISK_ORDER = ("LOW", "MEDIUM", "HIGH", "UNKNOWN")


def worst_risk(risks: list[str]) -> str:
    return max(risks, key=lambda r: RISK_ORDER.index(r) if r in RISK_ORDER else 3) if risks else "LOW"


def proxy_episode_audit(frame: pd.DataFrame, partition: pd.Series, ev: pd.Series, dt: pd.Series, pos: np.ndarray, idx: pd.Series, tc: TimeContract,
                        min_cell: int = 10) -> dict[str, Any]:
    """TRAIN only, aggregate. Is the fall-recency signal genuine history or one episode counted twice (last fall = next fall)?"""
    pa = tc.header.get("proxy_audit") or {}
    tr = (partition == "train").to_numpy()
    lf = frame["Last_Fall_Date"].dt.normalize() if "Last_Fall_Date" in frame else pd.Series(pd.NaT, index=frame.index)
    ds = pd.to_numeric(frame.get("Days_Since_Last_Fall"), errors="coerce") if "Days_Since_Last_Fall" in frame else pd.Series(np.nan, index=frame.index)
    exp = (idx - lf).dt.days
    both = lf.notna() & ds.notna()
    res: dict[str, Any] = {"feature": pa.get("feature"), "rows_train": int(tr.sum()),
                           "days_since_consistent": int((both & (ds == exp)).sum()), "days_since_inconsistent": int((both & (ds != exp)).sum()),
                           "days_since_negative": int((ds < 0).sum()), "last_fall_after_index": int((lf > idx).sum())}
    bands = [(-1, 0, "0 (index day)"), (1, 7, "1-7"), (8, 30, "8-30"), (31, 90, "31-90"), (91, 180, "91-180"), (181, 365, "181-365"), (366, 10**6, ">365")]
    y = pos
    rows = []
    early = int(pa.get("early_event_days", 14))
    for lo, hi, lab in [*bands, (None, None, "no prior fall")]:
        m = (exp.isna() if lo is None else exp.between(lo if lo >= 0 else 0, hi)).fillna(False).to_numpy() & tr
        n, e = int(m.sum()), int((m & y).sum())
        e_early = int((m & y & (dt <= early).fillna(False).to_numpy()).sum())
        rows.append({"days_since_last_fall": lab, "n": _sup(n, min_cell), "events": _sup(e, min_cell),
                     "event_rate_pct": round(100 * e / n, 2) if n >= min_cell and e >= min_cell else None,
                     f"events_within_{early}d": _sup(e_early, min_cell), f"share_within_{early}d": round(e_early / e, 3) if e >= min_cell else None})
    res["by_recency_band"] = rows
    gap = (ev - lf).dt.days
    g_ok = (y & tr & gap.notna().to_numpy())
    n_g = int(g_ok.sum())
    res["gap_next_minus_last_fall"] = {"positives_with_prior_fall": n_g,
                                       **{f"le_{k}d": _sup(int((g_ok & (gap <= k).fillna(False).to_numpy()).sum()), min_cell) for k in (1, 7, 14, 30)},
                                       "share_le_7d": round(float((g_ok & (gap <= 7).fillna(False).to_numpy()).sum()) / n_g, 3) if n_g >= min_cell else None}
    recent = (exp <= int(pa.get("recent_fall_days", 30))).fillna(False).to_numpy() & tr
    other = ~recent & tr
    er = int((recent & y).sum())
    er_early = int((recent & y & (dt <= early).fillna(False).to_numpy()).sum())
    eo = int((other & y).sum())
    eo_early = int((other & y & (dt <= early).fillna(False).to_numpy()).sum())
    s_r = er_early / er if er else None
    s_o = eo_early / eo if eo else None
    ratio = (s_r / s_o) if s_r is not None and s_o else None
    flag_early = bool(er >= int(pa.get("min_events", 10)) and s_r is not None and s_r > float(pa.get("max_early_share_recent", 0.5))
                      and (ratio is None or ratio > float(pa.get("max_early_ratio", 3.0))))
    share7 = res["gap_next_minus_last_fall"]["share_le_7d"]
    flag_gap = bool(share7 is not None and share7 > float(pa.get("max_gap_7d_share", 0.25)))
    res.update({"recent_fallers_events": _sup(er, min_cell), "recent_fallers_early_share": None if s_r is None or er < min_cell else round(s_r, 3),
                "others_early_share": None if s_o is None or eo < min_cell else round(s_o, 3), "early_share_ratio": None if ratio is None else round(ratio, 2),
                "flag_early_concentration": flag_early, "flag_short_gap": flag_gap, "investigation": bool(flag_early or flag_gap),
                "interpretation": ("the fall-recency signal may partly be ONE episode counted as history and as outcome: review before trusting it"
                                   if (flag_early or flag_gap) else "no sign that the last and the next fall are the same episode (recency looks like history)")})
    return res


def verify(frame: pd.DataFrame, partition: pd.Series, *, dictionary: Any, d00: Any, tc: TimeContract, min_cell: int = 10) -> dict[str, Any]:
    """Run V1-V6 on TRAIN + VALIDATION rows of the raw extract (label columns present). Returns aggregate results; never raises."""
    idx = frame["Index_Date"].dt.normalize()
    y = pd.to_numeric(frame["Fall_Next_180D_Ind"], errors="coerce")
    lab = y.notna()
    pos = (y == 1).to_numpy()
    out: dict[str, Any] = {"rows_checked": int(len(frame)), "partitions": sorted(set(partition.astype(str))), "test_rows_read": False}
    # V1
    ev = frame["Next_Fall_Date_180D"].dt.normalize() if "Next_Fall_Date_180D" in frame else pd.Series(pd.NaT, index=frame.index)
    dt = pd.to_numeric(frame.get("Days_To_Next_Fall_180D"), errors="coerce") if "Days_To_Next_Fall_180D" in frame else pd.Series(np.nan, index=frame.index)
    on_or_before = pos & ((ev <= idx).fillna(False).to_numpy() | (dt < 1).fillna(False).to_numpy())
    on_day = pos & (ev == idx).fillna(False).to_numpy()
    min_days = float(dt[pos].min()) if pos.any() and dt[pos].notna().any() else None
    out["V1_label_excludes_index_day"] = {"passed": int(on_or_before.sum()) == 0, "positives": int(pos.sum()), "positives_event_on_or_before_index": int(on_or_before.sum()),
                                          "positives_event_on_index_day": int(on_day.sum()), "min_days_to_event": min_days,
                                          "positives_without_event_date": int((pos & ev.isna().to_numpy()).sum())}
    # V2
    end = frame["Label_End_180D"].dt.normalize() if "Label_End_180D" in frame else pd.Series(pd.NaT, index=frame.index)
    w = (end - idx).dt.days
    vals = {str(int(k)): int(v) for k, v in w[lab].value_counts().items() if pd.notna(k)}
    out["V2_window_end"] = {"passed": set(vals) <= {"180"} and bool(vals), "label_end_minus_index_days": vals,
                            "events_after_window_end": int((pos & (ev > end).fillna(False).to_numpy()).sum())}
    # V3
    dcols = predictor_date_columns(dictionary, d00, tc, list(frame.columns))
    per, total_rows = {}, np.zeros(len(frame), dtype=bool)
    for c in dcols:
        after = ((frame[c].dt.normalize() - idx).dt.days > 0).fillna(False).to_numpy()
        on = ((frame[c].dt.normalize() - idx).dt.days == 0).fillna(False).to_numpy()
        per[c] = {"n_after_index": int(after.sum()), "n_on_index": int(on.sum()),
                  "max_days_after": int(((frame[c].dt.normalize() - idx).dt.days[after]).max()) if after.any() else 0}
        total_rows |= after
    out["V3_no_post_index_records"] = {"passed": int(total_rows.sum()) == 0, "rows_with_any_post_index_record": int(total_rows.sum()), "by_column": per,
                                       "columns_checked": dcols, "excluded_sources": tc.excluded_date_sources}
    # V4
    from falls_ml.data.meuhedet_timing import DAYS_SINCE_COMPANION

    v4 = {}
    for dc, ds in DAYS_SINCE_COMPANION.items():
        if dc in frame and ds in frame and dictionary.source(dc) not in tc.excluded_date_sources:
            both = frame[dc].notna() & frame[ds].notna()
            exp = (idx - frame[dc].dt.normalize()).dt.days
            mism = both & (pd.to_numeric(frame[ds], errors="coerce") != exp)
            v4[ds] = {"n_both": int(both.sum()), "n_mismatch": int(mism.sum()), "n_negative": int((pd.to_numeric(frame[ds], errors="coerce") < 0).sum())}
    out["V4_days_since_consistency"] = {"passed": all(v["n_mismatch"] == 0 and v["n_negative"] == 0 for v in v4.values()), "by_column": v4}
    # V5
    if "Fall_On_Index_Date_Ind" in frame:
        f = pd.to_numeric(frame["Fall_On_Index_Date_Ind"], errors="coerce") == 1
        out["V5_index_day_fall_vs_label"] = {"index_day_falls": int(f.sum()), "with_label_1": _sup(int((f & (y == 1)).sum()), min_cell),
                                             "with_label_0": _sup(int((f & (y == 0)).sum()), min_cell),
                                             "note": "under the corrected contract an index-day fall is history, not an outcome event"}
    # V6 (TRAIN only)
    tr = (partition == "train").to_numpy()
    idx_day = np.zeros(len(frame), dtype=bool)
    for c in ("Last_Fall_Date", "Last_Dx_Date", "Last_Visit_Date"):
        if c in frame:
            idx_day |= ((frame[c].dt.normalize() - idx).dt.days == 0).fillna(False).to_numpy()
    day1 = pos & (dt == 1).fillna(False).to_numpy()
    a, n_a = int((day1 & idx_day & tr).sum()), int((idx_day & tr).sum())
    b, n_b = int((day1 & ~idx_day & tr).sum()), int((~idx_day & tr).sum())
    ra = (a / n_a) / (b / n_b) if n_a and n_b and b else None
    cfg6 = tc.checks.get("V6_boundary_episode", {})
    flag = bool(a >= int(cfg6.get("min_events", 10)) and (ra is None or ra > float(cfg6.get("max_rate_ratio", 5.0))))
    out["V6_boundary_episode"] = {"train_rows_with_index_day_record": n_a, "day1_events_among_them": _sup(a, min_cell), "train_rows_other": n_b,
                                  "day1_events_other": _sup(b, min_cell), "rate_ratio": None if ra is None else round(ra, 2), "investigation": flag}
    # V7 proxy / episode audit (TRAIN only): Last_Fall_Date, Days_Since_Last_Fall, Index_Date, Next_Fall_Date_180D
    out["V7_proxy_episode_audit"] = proxy_episode_audit(frame, partition, ev, dt, pos, idx, tc, min_cell)
    # V8 availability indicators (aggregate data-quality signals of delayed availability)
    v8 = {}
    for c in ("Unsettled_Visit_Counter_Ind", "Future_Dated_365D", "Fallback_Exposure_365D", "External_Care_Hidden_By_Billing_Lag_365D"):
        if c in frame:
            x = pd.to_numeric(frame[c], errors="coerce")
            v8[c] = {"rows_positive": int((x > 0).sum()), "share_positive": round(float((x > 0).mean()), 5), "sum": float(x.fillna(0).sum())}
    out["V8_availability_indicators"] = v8
    out["hard_passed"] = bool(out["V1_label_excludes_index_day"]["passed"] and out["V2_window_end"]["passed"])
    out["attestation_holds"] = bool(out["V3_no_post_index_records"]["passed"])
    return out
