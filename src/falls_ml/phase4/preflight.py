"""``falls_ml meuhedet-phase4-preflight``: the read-only predictor / schema / timing audit of the 2026 extract.

Outcome columns (the brief's list, every LABEL / FORBIDDEN_LEAKAGE / post-index contract column, and new columns whose name looks like outcome or
future information) are never loaded. The audit compares the 2026 predictors with the frozen 2025 Phase 3 models and stops - never "fixes" -
when a frozen predictor is absent, semantically changed, unreadable, or no longer available at prediction time. Writes only <out>/preflight and
<out>/PHASE4_PLAN.json (aggregate counts; small cells suppressed). Last line: SAFE TO SCORE BLIND or STOP - TEMPORAL VALIDATION NOT DEFENSIBLE.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from falls_ml.artifacts import utc_now
from falls_ml.phase2 import durable as D
from falls_ml.phase2.state import Phase2Stop
from falls_ml.phase4 import PHASE4_VERSION, WATERMARK
from falls_ml.phase4.common import PLAN, SCORE_DONE, code_sha, dirs, input_identity, sha_json
from falls_ml.phase4.features import Built2026, ModelSpec, build_2026, eligible_mask, model_specs, needed_columns, scoring_frame, union
from falls_ml.phase4.phase3_source import S01, S03, S04, S06, Phase3Source
from falls_ml.phase4.sealed import read_columns, read_header, sealed_map

SAFE_LINE = "SAFE TO SCORE BLIND"
STOP_LINE = "STOP - TEMPORAL VALIDATION NOT DEFENSIBLE"
OK, WARN, STOP = "OK", "WARN", "STOP"
ELIGIBLE_CLASSES = ("SAFE_VERIFIED", "SAFE_VERIFIED_BOUNDED", "SAFE_ATTESTED")
GROUPS = [("DIZZINESS_VERTIGO", r"dizz|vertig"), ("GAIT_ABNORMALITY", r"gait"), ("SYNCOPE", r"syncop|faint"), ("TREMOR", r"tremor"),
          ("CATARACT", r"catar"), ("HEARING", r"hear|deaf|audio"), ("VISION", r"vision|visual|eye|blind|ophth"), ("OSTEOPOROSIS", r"osteopor"),
          ("PARKINSONISM", r"parkins"), ("STROKE", r"stroke|cva|cerebrovasc"), ("SMOKING", r"smok"), ("OBESITY", r"obes|bmi"),
          ("ONCOLOGY", r"onco|cancer|tumou?r|malig"), ("IBD", r"ibd|crohn|colitis"), ("OPIATE_REGISTRY", r"opi(at|oid)"),
          ("SEVERE_FUNCTION_REGISTRY", r"sever|function"), ("MEFI_FRAILTY", r"mefi|frail"), ("MEDICATION", r"med|drug|atc|purchas|prescri"),
          ("EXTERNAL_CARE_BILLING", r"external|invoice|bill"), ("REGISTRY_OTHER", r"regist")]


@dataclass
class Audit:
    checks: list[dict[str, Any]] = field(default_factory=list)
    excluded_models: dict[str, str] = field(default_factory=dict)        # secondary models excluded by a pre-declared rule (with the reason)
    facts: dict[str, Any] = field(default_factory=dict)
    shift: pd.DataFrame | None = None
    schema: pd.DataFrame | None = None
    catalogue: pd.DataFrame | None = None
    specs: dict[str, ModelSpec] = field(default_factory=dict)
    built: Built2026 | None = None
    p3: Phase3Source | None = None
    header: list[str] = field(default_factory=list)
    sealed: dict[str, str] = field(default_factory=dict)
    need: dict[str, Any] = field(default_factory=dict)
    read: Any = None
    input: dict[str, Any] = field(default_factory=dict)
    model_sources: dict[str, dict[str, Any]] = field(default_factory=dict)

    def add(self, cid: str, title: str, status: str, details: list[str] | None = None) -> None:
        self.checks.append({"id": cid, "title": title, "status": status, "details": list(details or [])})

    @property
    def safe(self) -> bool:
        return not any(c["status"] == STOP for c in self.checks)

    def fingerprint(self, cfg_sha: str, code: str) -> str:
        return sha_json({"input": self.input.get("sha256"), "phase3_digest": self.p3.digest() if self.p3 else None, "config": cfg_sha, "code": code,
                         "safe": self.safe, "excluded": self.excluded_models, "checks": [(c["id"], c["status"]) for c in self.checks],
                         "models": {k: v.features for k, v in self.specs.items()}})


def _sup(n: Any, min_cell: int = 10) -> Any:
    try:
        v = int(n)
    except (TypeError, ValueError):
        return n
    return v if v == 0 or v >= min_cell else f"<{min_cell}"


def _pct_sup(k: int, n: int, min_cell: int = 10) -> Any:
    """A percentage of n rows whose count (or complement) is 1..min_cell-1 would give a small cell back: suppressed."""
    if not n:
        return None
    return "suppressed" if 0 < k < min_cell or 0 < n - k < min_cell else round(100.0 * k / n, 3)


def _num(s: pd.Series) -> np.ndarray:
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        return (s.astype(str) == "female").to_numpy(dtype=float)
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)


def _psi(a: np.ndarray, b: np.ndarray, bins: int) -> float:
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if not len(a) or not len(b):
        return math.nan
    ua = np.unique(a)
    if len(ua) <= bins:
        cats = np.union1d(ua, np.unique(b))
        pa = np.array([(a == c).mean() for c in cats])
        pb = np.array([(b == c).mean() for c in cats])
    else:
        edges = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        pa = np.histogram(a, edges)[0] / len(a)
        pb = np.histogram(b, edges)[0] / len(b)
    pa, pb = np.clip(pa, 1e-4, None), np.clip(pb, 1e-4, None)
    return float(np.sum((pb - pa) * np.log(pb / pa)))


def _stats(x: np.ndarray) -> dict[str, float]:
    o = x[np.isfinite(x)]
    if not len(o):
        return {"mean": math.nan, "sd": math.nan, "p05": math.nan, "p25": math.nan, "p50": math.nan, "p75": math.nan, "p95": math.nan, "share_nonzero": math.nan,
                "n_unique": 0}
    q = np.quantile(o, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {"mean": float(o.mean()), "sd": float(o.std(ddof=1)) if len(o) > 1 else 0.0, "p05": float(q[0]), "p25": float(q[1]), "p50": float(q[2]),
            "p75": float(q[3]), "p95": float(q[4]), "share_nonzero": float((o != 0).mean()), "n_unique": int(len(np.unique(o)))}


def _kind(f: str, cat: Any, mapping: Any) -> str:
    if f in cat.names:
        return str(cat.get(f).kind)
    for m in mapping.features:
        if m.canonical == f:
            return {"recode": "categorical", "any_positive": "binary", "count": "count"}.get(m.op, "continuous")
    return "unknown"


def predictor_shift(a: Audit, L: dict[str, Any], cfg: Any) -> pd.DataFrame:
    p3, b = a.p3, a.built
    pf = cfg["preflight"]
    tr = p3.train_frame()
    ukn25 = p3.unknown_train()
    ev26 = scoring_frame(b)
    reg25 = p3.registry().set_index("feature")
    reg26 = b.registry.set_index("feature")
    new, base = union(a.specs)
    rows = []
    for f in [*base, *new]:
        k25 = ~ukn25[f].to_numpy(dtype=bool) if f in ukn25.columns else np.ones(len(tr), dtype=bool)
        k26 = ~b.unknown[f].to_numpy(dtype=bool) if f in b.unknown.columns else np.ones(len(ev26), dtype=bool)
        x25, x26 = _num(tr[f])[k25], _num(ev26[f])[k26]
        s25, s26 = _stats(x25), _stats(x26)
        kind = _kind(f, L["cat"], L["mapping"])
        miss25 = float(np.mean(~np.isfinite(x25))) if len(x25) else math.nan
        miss26 = float(np.mean(~np.isfinite(x26))) if len(x26) else math.nan
        pooled = math.sqrt(((s25["sd"] or 0) ** 2 + (s26["sd"] or 0) ** 2) / 2) if np.isfinite(s25["sd"]) and np.isfinite(s26["sd"]) else math.nan
        smd = (s26["mean"] - s25["mean"]) / pooled if pooled and np.isfinite(pooled) and pooled > 0 else math.nan
        psi = _psi(x25, x26, int(pf["psi_bins"]))
        stop, warn = [], []
        obs25 = float(np.isfinite(x25).mean()) if len(x25) else 0.0
        if obs25 >= float(pf["min_observed_share_2025_for_absence_stop"]) and not np.isfinite(x26).any():
            stop.append("observed in 2025 but no observed 2026 value (semantically absent)")
        if kind == "binary":
            if np.isfinite(x26).any() and not set(np.unique(x26[np.isfinite(x26)])) <= {0.0, 1.0}:
                stop.append("binary predictor with values outside {0, 1} in 2026")
            if (s25["share_nonzero"] or 0) >= float(pf["min_observed_share_2025_for_absence_stop"]) and s26["share_nonzero"] == 0:
                stop.append("indicator recorded in 2025 but never set in 2026 (semantically absent)")
        if s25["n_unique"] > 1 and s26["n_unique"] == 1 and len(x26):
            stop.append("constant in 2026 but variable in 2025")
        if kind in ("continuous", "count", "ordinal"):
            ref = "p50" if kind == "continuous" else "p95"
            v25, v26 = s25[ref], s26[ref]
            R = float(pf["unit_change_ratio"])
            if np.isfinite(v25) and np.isfinite(v26) and v25 > 0 and v26 > 0 and (v26 / v25 > R or v25 / v26 > R):
                stop.append(f"{ref} ratio 2026/2025 = {v26 / v25:.3g} (beyond x{R:g}: possible unit / definition change)")
        if f == "sex":
            lv25 = set(tr.loc[k25, f].dropna().astype(str))
            lv26 = set(ev26.loc[k26, f].dropna().astype(str))
            if lv26 - lv25:
                stop.append(f"new category levels in 2026: {sorted(lv26 - lv25)}")
        if np.isfinite(miss25) and np.isfinite(miss26) and abs(miss26 - miss25) * 100 > float(pf["warn_missingness_change_pp"]):
            warn.append(f"missingness {100 * miss25:.1f}% -> {100 * miss26:.1f}%")
        if np.isfinite(psi) and psi > float(pf["warn_psi"]):
            warn.append(f"PSI {psi:.3f} > {pf['warn_psi']}")
        c25 = str(reg25["phase3_class"].get(f, "")) if f in reg25.index else ""
        c26 = str(reg26["phase3_class"].get(f, "")) if f in reg26.index else ""
        mc = int(pf["min_cell"])
        if kind in ("binary", "categorical"):            # a share of a rare level is a small cell: suppress share / mean / quantiles
            for s_, x_ in ((s25, x25), (s26, x26)):
                o_ = x_[np.isfinite(x_)]
                k_ = int((o_ != 0).sum())
                if 0 < k_ < mc or 0 < len(o_) - k_ < mc:
                    for kk in ("mean", "sd", "share_nonzero", "p05", "p25", "p50", "p75", "p95"):
                        s_[kk] = "suppressed"
        rows.append({"feature": f, "kind": kind, "models": "; ".join(k for k, s in a.specs.items() if f in s.features),
                     "used_by_primary": f in a.specs[cfg.primary].features, "n_known_2025_train": int(len(x25)), "n_known_2026": int(len(x26)),
                     "n_unknown_2026": int((~k26).sum()), "missing_pct_2025": _pct_sup(int((~np.isfinite(x25)).sum()), len(x25), mc),
                     "missing_pct_2026": _pct_sup(int((~np.isfinite(x26)).sum()), len(x26), mc),
                     **{f"{k}_2025": v for k, v in s25.items() if k != "n_unique"}, **{f"{k}_2026": v for k, v in s26.items() if k != "n_unique"},
                     "smd": smd, "psi": psi, "class_2025": c25, "class_2026": c26,
                     "availability_risk": str(reg25["availability_risk"].get(f, "")) if f in reg25.index and "availability_risk" in reg25.columns else "",
                     "sources": str(reg25["sources"].get(f, "")) if f in reg25.index else "",
                     "verdict": STOP if stop else WARN if warn else OK, "reason": "; ".join(stop + warn)})
    t = pd.DataFrame(rows)
    for c in ("n_known_2025_train", "n_known_2026", "n_unknown_2026"):
        t[c] = t[c].map(lambda v: _sup(v, int(pf["min_cell"])))
    return t


def schema_table(a: Audit, L: dict[str, Any], cfg: Any) -> pd.DataFrame:
    contract = L["contract"]
    header = set(a.header)
    inputs_of: dict[str, list[str]] = {}
    for f, ins in a.need["inputs"].items():
        for c in ins:
            inputs_of.setdefault(c, []).append(f)
    elig = eligible_mask(a.read.frame, cfg["index_date"]) if a.read is not None and "Index_Date" in a.read.frame else None
    w25 = a.p3.work() if a.p3 is not None else None
    mc = int(cfg["preflight"]["min_cell"])
    rows = []
    for c in [*contract.names, *[h for h in a.header if h not in set(contract.names)]]:
        cc = contract.get(c) if c in set(contract.names) else None
        read = a.read is not None and c in a.read.frame.columns
        miss26 = _pct_sup(int(a.read.frame.loc[elig, c].isna().sum()), int(elig.sum()), mc) if read and elig is not None and elig.any() else None
        miss25 = _pct_sup(int(w25[c].isna().sum()), len(w25), mc) if w25 is not None and c in w25.columns else None
        status = ("SEALED_NOT_READ" if c in a.sealed else "NEW_IN_2026_CATALOGUED_ONLY" if cc is None else
                  ("ABSENT_FROZEN_INPUT" if c in inputs_of else "ABSENT_NOT_USED") if c not in header else
                  "FROZEN_INPUT" if c in inputs_of else "PRESENT_NOT_USED")
        rows.append({"column": c, "in_2025_contract": cc is not None, "in_2026_extract": c in header, "contract_role": cc.role if cc else "",
                     "contract_timing": cc.timing if cc else "", "contract_type": cc.sql if cc else "", "sealed_reason": a.sealed.get(c, ""),
                     "frozen_input_of": "; ".join(sorted(set(inputs_of.get(c, [])))), "wrong_type_cells_2026": _sup(a.read.wrong_type.get(c, 0), mc) if read else None,
                     "not_allowed_cells_2026": _sup(a.read.not_allowed.get(c, 0), mc) if read else None, "missing_pct_2025_trainval": miss25,
                     "missing_pct_2026_eligible": miss26, "status": status})
    return pd.DataFrame(rows)


def new_catalogue(a: Audit, L: dict[str, Any], cfg: Any) -> pd.DataFrame:
    from falls_ml.data.meuhedet_wide import _coerce_date_column

    names = set(L["contract"].names)
    elig = eligible_mask(a.read.frame, cfg["index_date"]) if a.read is not None else None
    idx = pd.Timestamp(cfg["index_date"])
    mc = int(cfg["preflight"]["min_cell"])
    rows = []
    for c in [h for h in a.header if h not in names]:
        grp = "SEALED_OUTCOME_OR_FUTURE" if c in a.sealed else next((g for g, pat in GROUPS if re.search(pat, c, re.IGNORECASE)), "OTHER")
        r: dict[str, Any] = {"column": c, "candidate_group": grp, "sealed": c in a.sealed, "sealed_reason": a.sealed.get(c, ""),
                             "looks_like_date": bool(re.search(r"date|_dt$", c, re.IGNORECASE)), "non_null_pct_2026_eligible": None,
                             "n_dated_after_index_2026": None,
                             "status": "SEALED - the name looks like outcome / future information: never read, never a predictor candidate" if c in a.sealed
                             else "CATALOGUED ONLY - not part of any frozen Phase 3 model; never used in the temporal validation",
                             "recommended_use": "confirm its definition with the DWH team" if c in a.sealed
                             else "backfill for Index_Date 2025-01-01, develop on 2025, validate once on a still-unseen later snapshot"}
        if c not in a.sealed and a.read is not None and c in a.read.raw.columns and elig is not None:
            s = a.read.raw.loc[elig, c]
            r["non_null_pct_2026_eligible"] = _pct_sup(int(s.notna().sum()), len(s), mc)
            if r["looks_like_date"]:
                parsed, _bad, _ex, _used = _coerce_date_column(s, L["contract"].date_formats)
                r["n_dated_after_index_2026"] = _sup(int(((parsed.dt.normalize() - idx).dt.days > 0).fillna(False).sum()), mc)
        rows.append(r)
    return pd.DataFrame(rows, columns=["column", "candidate_group", "sealed", "sealed_reason", "looks_like_date", "non_null_pct_2026_eligible",
                                       "n_dated_after_index_2026", "status", "recommended_use"])


def audit(input_2026: Path, phase3_out: Path, *, L: dict[str, Any], allow_unfrozen_phase3: bool, expected_definition_version: str | None) -> Audit:
    cfg = L["cfg4"]
    contract, mapping, dictionary = L["contract"], L["mapping"], L["dictionary"]
    a = Audit()
    a.input = input_identity(input_2026)
    # ---- P0 the frozen Phase 3 run (read-only, every file verified against its commit record)
    p3 = Phase3Source(phase3_out, allow_unfrozen=allow_unfrozen_phase3)
    a.p3 = p3
    info = p3.check_frozen()
    a.facts["phase3"] = info
    fs = p3.feature_sets()
    a.specs = model_specs(fs, cfg.models)
    for sid, name, item in (S01, S03, S04, S06):
        p3._item_dir(sid, name, item)
    for c, s in a.specs.items():
        m = p3.persisted_model(s.setname, cfg["phase3_stage_lasso"])
        a.model_sources[c] = {"mode": "A" if m else "B", "fitted_set": p3.model_item(s.setname)[0],
                              "artifact": "persisted Phase 3 final fit (model.pkl, commit record verified)" if m else
                              "not persisted: deterministic frozen re-fit from the committed 2025 products at scoring"}
    a.add("P0", "frozen Phase 3 models (2025)", OK if info["frozen_production_config"] else WARN,
          [f"Phase 3 run status {info.get('run_status')}, decision {info.get('decision')}, frozen production configuration {info['frozen_production_config']}",
           *[f"{c}: mode {v['mode']} - {v['artifact']} (fitted set {v['fitted_set']})" for c, v in a.model_sources.items()],
           f"{len(p3.verified)} Phase 3 files verified against their commit records (digest {p3.digest()[:16]}…)"])
    # ---- header, sealing, required inputs
    a.header = read_header(input_2026)
    a.sealed = sealed_map(a.header, contract, cfg)
    a.need = needed_columns(a.specs, cat=L["cat"], mapping=mapping, dictionary=dictionary, d00=L["d00"], tc=L["tc"], header=a.header, sealed=a.sealed)
    probs = [f"{f}: input column {c} is ABSENT from the 2026 extract" for f, c in a.need["absent_inputs"]]
    probs += [f"{f}: input column {c} is sealed ({why}) - a frozen predictor may never read outcome / future information" for f, c, why in a.need["sealed_inputs"]]
    probs += [f"identifier / eligibility column {c} is absent" for c in a.need["missing_id_columns"]]
    a.add("P1", "frozen predictor inputs present (no absent / renamed input)", STOP if probs else OK,
          probs or [f"all {sum(len(v) for v in a.need['inputs'].values())} input references of {len(a.need['inputs'])} frozen features are present"])
    leak = []
    for f, ins in a.need["inputs"].items():
        for c in ins:
            if c not in set(contract.names):
                leak.append(f"{f}: input {c} is not a 2025 contract column")
                continue
            cc = contract.get(c)
            if cc.role in ("LABEL", "FORBIDDEN_LEAKAGE", "IDENTIFIER") or cc.timing == "post_index":
                leak.append(f"{f}: input {c} has contract role {cc.role} / timing {cc.timing}")
    a.add("P2", "leakage / target / future fields excluded from X", STOP if leak else OK,
          leak or [f"{len(a.sealed)} columns sealed (never loaded): {sum(1 for v in a.sealed.values() if v == 'BRIEF_OUTCOME_COLUMN')} outcome columns of "
                   f"the brief, {sum(1 for v in a.sealed.values() if v.startswith('CONTRACT'))} contract LABEL / FORBIDDEN / post-index, "
                   f"{sum(1 for v in a.sealed.values() if v.startswith('NEW'))} new columns by name", "no frozen predictor reads a sealed, label, forbidden or post-index column"])
    a.facts["sealed"] = {"n_sealed": len(a.sealed), "by_reason": pd.Series(list(a.sealed.values())).value_counts().to_dict() if a.sealed else {}}
    if not a.safe:
        return a
    # ---- read every NON-sealed column (predictors, eligibility, QA, new columns for the catalogue)
    a.read = read_columns(input_2026, [c for c in a.header if c not in a.sealed], a.sealed, contract)
    fr = a.read.frame
    idx = pd.Timestamp(cfg["index_date"])
    idx_counts = fr["Index_Date"].dt.normalize().value_counts(dropna=False)
    n_on = int((fr["Index_Date"].dt.normalize() == idx).fillna(False).sum())
    elig = eligible_mask(fr, cfg["index_date"])
    n_el = int(elig.sum())
    a.facts["cohort"] = {"extract_rows": int(len(fr)), "rows_on_index_date": n_on, "eligible_on_index_date": n_el,
                         "distinct_index_dates": int(idx_counts.shape[0]), "index_dates": {str(k.date()) if pd.notna(k) else "NULL": int(v) for k, v in idx_counts.head(10).items()}}
    a.add("P3", f"Index_Date = {cfg['index_date']} and eligible cohort", STOP if n_el < int(cfg["preflight"]["min_eligible_rows"]) else OK,
          [f"{len(fr)} extract rows; {n_on} on {cfg['index_date']}; {n_el} eligible (Is_Eligible_Cohort = 1)",
           f"distinct index dates in the file: {idx_counts.shape[0]} (only {cfg['index_date']} is used)"])
    dups = []
    for c in ("Customer_Full_ID", "Snapshot_Key"):
        if c in fr.columns:
            s = fr.loc[elig, c].dropna()
            d = int(s.duplicated().sum())
            if d:
                dups.append(f"{c}: {d} duplicated values among the eligible index-date rows")
    a.add("P4", "one row per patient (Customer_Full_ID / Snapshot_Key unique)", STOP if dups and cfg["preflight"]["stop_if_duplicate_ids"] else OK,
          dups or ["no duplicated Customer_Full_ID or Snapshot_Key among the eligible index-date rows"])
    # ---- QA constants
    qa = []
    st = OK
    if "Leakage_Check_Ind" in fr.columns:
        nl = int((pd.to_numeric(fr.loc[elig, "Leakage_Check_Ind"], errors="coerce").fillna(0) != 0).sum())
        qa.append(f"Leakage_Check_Ind non-zero on {nl} eligible rows (the VIEW's own check; NOT a complete leakage guarantee - Phase 4 relies on its "
                  "own sealing and timing checks)")
        st = STOP if nl else st
    else:
        qa.append("Leakage_Check_Ind absent from the 2026 extract (the VIEW's own check cannot be confirmed)")
        st = WARN
    if "Definition_Version" in fr.columns:
        vals = fr.loc[elig, "Definition_Version"].astype("string").fillna("NULL").value_counts().to_dict()
        qa.append(f"Definition_Version values on eligible rows: {vals}")
        a.facts["definition_version"] = {str(k): int(v) for k, v in vals.items()}
        if expected_definition_version:
            if set(vals) != {expected_definition_version}:
                qa.append(f"expected only {expected_definition_version!r} (--expected-definition-version): the extracted object is NOT the intended definition")
                st = STOP
            else:
                qa.append(f"matches the expected definition {expected_definition_version!r}")
        else:
            qa.append("no --expected-definition-version given: confirm that this is the intended V21 definition")
            st = WARN if st == OK else st
    else:
        qa.append("Definition_Version absent: the extract definition cannot be confirmed")
        st = WARN if st == OK else st
    a.add("P5", "extract definition (V21) and the VIEW's leakage flag", st, qa)
    # ---- cell problems in frozen inputs
    lim = float(cfg["preflight"]["max_unreadable_share_frozen_input"])
    cells, cstop = [], False
    for f, ins in a.need["inputs"].items():
        for c in ins:
            if a.read.unreadable is not None and c in a.read.unreadable.columns:
                k = int(a.read.unreadable.loc[elig, c].sum())
                if k:
                    share = k / max(1, n_el)
                    cells.append(f"{f} <- {c}: {_sup(k)} unreadable / not-allowed cells ({100 * share:.3f}% of eligible rows)"
                                 + (" -> STOP (format / coding change)" if share > lim else " -> those rows UNKNOWN for the feature (bounded)"))
                    cstop |= share > lim
    a.add("P6", "data types and categorical / allowed codes of the frozen inputs", STOP if cstop else WARN if cells else OK,
          sorted(set(cells)) or ["every frozen input cell is readable with the 2025 contract type and allowed codes"])
    # ---- build the frozen predictors on 2026 (unchanged code), V3 timing
    b = build_2026(a.read, a.specs, cat=L["cat"], mapping=mapping, contract=contract, espec=L["espec"], dictionary=dictionary, d00=L["d00"], rules=L["rules"],
                   tc=L["tc"], index_date=cfg["index_date"], work_dtypes=dict(p3.work().dtypes), v3_columns=a.need["v3_date_columns"], inputs=a.need["inputs"])
    a.built = b
    a.add("P7", "canonical baseline rebuilt with the unchanged adapter (codes / NULL rules)", STOP if b.adapter_problems else OK,
          b.adapter_problems[:15] or ["no undeclared code, forbidden NULL or negative count in the frozen baseline inputs"])
    v3 = b.v3
    v3d = [f"{c}: {_sup(v['n_after_index'])} rows dated after {cfg['index_date']} (max +{v['max_days_after']} d); {_sup(v['n_on_index'])} on the index day"
           for c, v in v3["by_column"].items() if v["n_after_index"]]
    a.facts["V3_2026"] = {"attestation_holds": bool(v3["passed"]), "rows_with_any_post_index_record": _sup(v3["rows_with_any_post_index_record"]),
                          "columns_checked": v3["columns_checked"], "sealed_date_columns_not_checked": a.need["v3_sealed_date_columns"]}
    a.add("P8", "predictor record dates after Index_Date (check V3 on 2026)", OK if v3["passed"] else WARN,
          (v3d or [f"no predictor record dated after {cfg['index_date']} in {len(v3['columns_checked'])} record-date columns: the DWH attestation holds on 2026"])
          + ([f"sealed (not checked): {a.need['v3_sealed_date_columns']}"] if a.need["v3_sealed_date_columns"] else []))
    # ---- 2026 eligibility class of every frozen feature; pre-declared model rule
    reg26 = b.registry.set_index("feature")
    reg25 = p3.registry().set_index("feature")
    p9, p9stop = [], False
    for c, s in a.specs.items():
        bad = []
        for f in s.features:
            k26 = str(reg26["phase3_class"].get(f, "")) if f in reg26.index else "MISSING"
            if k26 not in ELIGIBLE_CLASSES and not (k26 == "INELIGIBLE_DATA" and f in reg26.index and int(reg26.at[f, "n_known_observed_train"]) > 0
                                                    and int(reg26.at[f, "n_unique_known_train"]) > 1):
                bad.append(f"{f}: 2026 class {k26} (2025: {reg25['phase3_class'].get(f, '?') if f in reg25.index else '?'}) - {reg26['phase3_reason'].get(f, '') if f in reg26.index else ''}")
        if bad:
            if c == cfg.primary:
                p9stop = True
                p9 += [f"PRIMARY {c}: " + x for x in bad]
            else:
                a.excluded_models[c] = "; ".join(bad)[:500]
                p9 += [f"secondary {c} EXCLUDED by the pre-declared rule: " + x for x in bad]
    a.add("P9", "frozen predictors still available at prediction time on 2026 (corrected contract)", STOP if p9stop else WARN if p9 else OK,
          p9 or ["every frozen predictor keeps an eligible timing class on 2026 (SAFE_VERIFIED / _BOUNDED / SAFE_ATTESTED)"])
    nd = [f"{f}: record date {rd} absent from the 2026 extract (timing no longer row-verifiable)" for f, rd in a.need["absent_record_dates"]]
    if nd:
        a.add("P9b", "record-date columns of the frozen predictors", WARN, nd)
    # ---- shift (missingness, ranges, categories)
    a.shift = predictor_shift(a, L, cfg)
    sh_stop = a.shift[a.shift["verdict"] == STOP]
    sh_warn = a.shift[a.shift["verdict"] == WARN]
    for _, r in sh_stop.iterrows():
        if r["used_by_primary"]:
            continue
        for c, s in a.specs.items():
            if c != cfg.primary and r["feature"] in s.features:
                a.excluded_models.setdefault(c, f"{r['feature']}: {r['reason']}")
    prim_stop = sh_stop[sh_stop["used_by_primary"].astype(bool)]
    a.add("P10", "missingness / value-range / category shift of the frozen predictors (semantic change)", STOP if len(prim_stop) else WARN if len(sh_stop) or len(sh_warn) else OK,
          [*[f"STOP {r['feature']}: {r['reason']}" for _, r in sh_stop.iterrows()], *[f"warn {r['feature']}: {r['reason']}" for _, r in sh_warn.iterrows()]]
          or ["no semantic change, absence, unit change or new category; no large missingness / distribution shift"])
    # ---- known V21 concerns (documented assumptions)
    conc = []
    srcs = a.shift.set_index("feature")["sources"].to_dict()
    risk = a.shift.set_index("feature")["availability_risk"].to_dict()
    med = sorted(f for f, s in srcs.items() if re.search(r"MEDIC|PRESCRI|PURCHAS", str(s), re.I))
    ext = sorted(f for f, s in srcs.items() if re.search(r"EXTERNAL|INVOICE", str(s), re.I))
    conc.append(f"medication timing (purchase timing / counters may reach after Index_Date): frozen features from medication sources: {med or 'none'}"
                + (f"; availability risk {sorted({risk.get(f, '') for f in med})}" if med else ""))
    conc.append(f"external-care / billing availability lag: frozen features from external-care sources: {ext or 'none'} "
                "(Phase 3 excluded the external-care invoices and hospitalisation dates as predictors)")
    newd = [c for c in a.header if c not in set(contract.names) and c not in a.sealed and re.search(r"date", c, re.I)]
    conc.append(f"new 2026 date columns checked for dates after Index_Date: {len(newd)} (see NEW_2026_FEATURES_CATALOGUE.csv)")
    conc.append("repeated fall / fracture diagnosis records may be one episode: audited after the outcomes are opened (episode audit), never a predictor change")
    conc.append("outcome fields vs Followup_End_Date: verified by meuhedet-phase4-evaluate before any metric (outcome contract)")
    a.add("P11", "known V21 documentation concerns", OK if not (med or ext) else WARN, conc)
    a.schema = schema_table(a, L, cfg)
    a.catalogue = new_catalogue(a, L, cfg)
    a.add("P12", "new 2026 columns", OK, [f"{len(a.catalogue)} columns not in the 2025 contract: catalogued only "
                                          f"({int(a.catalogue['sealed'].sum()) if len(a.catalogue) else 0} sealed by name); never used by a frozen model"])
    a.facts["models_scored"] = [c for c in a.specs if c not in a.excluded_models]
    a.facts["models_excluded"] = a.excluded_models
    return a


def report_md(a: Audit, cfg: Any, verdict_line: str) -> str:
    L = ["# Phase 4 - 2026 preflight (predictors only; outcomes sealed)", "", f"**{WATERMARK}**", "",
         f"Input: `{a.input.get('name')}` (sha256 `{a.input.get('sha256', '')[:16]}…`). Index date {cfg['index_date']}. Frozen models: "
         f"primary {cfg.primary}; secondary {', '.join(cfg['models']['secondary'])}.", "",
         "No outcome value was read: the outcome / follow-up / censoring columns and every LABEL, FORBIDDEN_LEAKAGE or post-index column are never "
         "loaded (SCHEMA_COMPARISON_2025_2026.csv, status SEALED_NOT_READ).", "", "| Check | Status | Evidence |", "|---|---|---|"]
    for c in a.checks:
        ev = "<br>".join(str(d).replace("|", "/") for d in c["details"][:12])
        L.append(f"| {c['id']} {c['title']} | **{c['status']}** | {ev} |")
    if a.excluded_models:
        L += ["", "## Secondary models excluded by the pre-declared rule", ""]
        L += [f"- {k}: {v}" for k, v in a.excluded_models.items()]
    L += ["", "Files: SCHEMA_COMPARISON_2025_2026.csv (every column), PREDICTOR_SHIFT.csv (every frozen predictor: 2025 TRAIN vs 2026), "
          "NEW_2026_FEATURES_CATALOGUE.csv (catalogued only).", "", verdict_line, ""]
    return "\n".join(L)


def run_preflight(input_2026: str | Path, phase3_out: str | Path, *, out_dir: str | Path, phase3_config: str | Path, config_path: str | Path,
                  allow_unfrozen_phase3: bool = False, expected_definition_version: str | None = None, allow_synced_folder: bool = False) -> int:
    from falls_ml.errors import FallsMLError
    from falls_ml.phase4.common import event, guard_out, load_definitions

    src, p3, out = Path(input_2026), Path(phase3_out), Path(out_dir)
    try:
        guard_out(out, phase3_out=p3, inputs=[src], allow_synced_folder=allow_synced_folder)
        if (out / "sealed" / SCORE_DONE).exists():
            raise Phase2Stop("ALREADY_SCORED", "the blind predictions of this folder are frozen: the preflight is closed (use meuhedet-phase4-status)")
        L = load_definitions(phase3_config, config_path)
        cfg = L["cfg4"]
        a = audit(src, p3, L=L, allow_unfrozen_phase3=allow_unfrozen_phase3, expected_definition_version=expected_definition_version)
    except FallsMLError as exc:
        print(f"[STOP] {getattr(exc, 'gate', type(exc).__name__)}: {getattr(exc, 'message', str(exc))}")
        for d in getattr(exc, "details", []) or []:
            print(f"   - {d}")
        print(STOP_LINE)
        return 2
    verdict = SAFE_LINE if a.safe else STOP_LINE
    d = dirs(out)
    d["preflight"].mkdir(parents=True, exist_ok=True)
    D.write_str(d["preflight"] / "PHASE4_PREFLIGHT.md", report_md(a, cfg, verdict))
    if a.schema is not None:
        D.write_csv(d["preflight"] / "SCHEMA_COMPARISON_2025_2026.csv", a.schema)
    if a.shift is not None:
        D.write_csv(d["preflight"] / "PREDICTOR_SHIFT.csv", a.shift)
    if a.catalogue is not None:
        D.write_csv(d["preflight"] / "NEW_2026_FEATURES_CATALOGUE.csv", a.catalogue)
    code = code_sha()
    res = {"verdict": verdict, "safe": a.safe, "checks": a.checks, "facts": a.facts, "excluded_models": a.excluded_models, "model_sources": a.model_sources,
           "input_2026": a.input, "phase3_digest": a.p3.digest() if a.p3 else None, "config_sha256": cfg.sha256, "code_sha256": code,
           "fingerprint": a.fingerprint(cfg.sha256, code), "phase4_version": PHASE4_VERSION, "falls_ml_version": __import__("falls_ml").__version__,
           "created_at": utc_now(), "outcomes_read": False}
    D.write_json(d["preflight"] / "PREFLIGHT_RESULT.json", res)
    D.write_json(out / PLAN, {"phase4_version": PHASE4_VERSION, "index_date": cfg["index_date"], "input_2026": a.input, "phase3_out": p3.name,
                              "phase3_digest": res["phase3_digest"], "config_sha256": cfg.sha256, "code_sha256_at_preflight": code,
                              "preflight_fingerprint": res["fingerprint"], "preflight_verdict": verdict, "created_at": res["created_at"]})
    event(out, "preflight", "verdict", "SAFE" if a.safe else "STOP", fingerprint=res["fingerprint"])
    print(report_md(a, cfg, verdict).replace("<br>", "\n      "))
    return 0 if a.safe else 2
