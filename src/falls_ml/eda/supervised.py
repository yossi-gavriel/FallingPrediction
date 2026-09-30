"""Target-aware EDA on the TRAIN partition only.

Every function here accepts a :class:`TrainPartition`, which can only be built from the split assignment and refuses any row that
is not in the training partition. Validation and test rows never reach these computations, so nothing learnt here can carry
held-out information into a modelling decision. Associations are univariate and descriptive (effect sizes with 95% intervals, no
p-values): not causal, not multivariate importance, never a selection rule on their own.

The one deliberate exception is :func:`table1` with ``basis=POSTHOC_DESCRIPTIVE``: the conventional descriptive Table 1 of the whole
modelling population, labelled as post-hoc and never used for model choice.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from falls_ml.data.meuhedet_wide import WideContract
from falls_ml.eda.common import SUPPRESSED, TRAIN_ONLY, auroc, cell, cramers_v, fnum, med_iqr, numeric, prevalence_ratio, smd, wilson
from falls_ml.eda.dictionary import DataDictionary
from falls_ml.errors import LeakageError


class TrainPartition:
    """Rows of the TRAIN partition with their 0/1 outcome. Construction fails if any other partition is present."""

    __test__ = False

    def __init__(self, frame: pd.DataFrame, y: np.ndarray, partition: pd.Series):
        labels = set(pd.Series(partition).astype(str).unique())
        if labels != {"train"}:
            raise LeakageError(f"TrainPartition received rows of partitions {sorted(labels)}; target-aware EDA may use TRAIN rows only")
        if len(frame) != len(y) or len(frame) != len(partition):
            raise ValueError("frame, outcome and partition labels must have the same length")
        yy = np.asarray(y, dtype=np.int64)
        if not np.isin(yy, (0, 1)).all():
            raise ValueError("the outcome must be 0/1 on every TRAIN row")
        self.frame = frame.reset_index(drop=True)
        self.y = yy
        self.n = len(frame)
        self.n_events = int(yy.sum())

    @classmethod
    def from_assignment(cls, raw: pd.DataFrame, partition: pd.Series, label_column: str) -> TrainPartition:
        """TRAIN rows of the raw frame (``partition`` aligned to it: train / validation / test / NaN for rows outside the cohort)."""
        keep = (partition == "train").fillna(False).to_numpy(dtype=bool)
        return cls(raw.loc[keep], raw.loc[keep, label_column].astype("int64").to_numpy(), partition[keep])


def _kind(contract: WideContract, col: str, s: pd.Series) -> str:
    c = contract.get(col)
    if c.semantic == "binary" or (c.allowed is not None and set(map(str, c.allowed)) <= {"0", "1"}):
        return "binary"
    if c.semantic in ("categorical",) or c.is_text:
        return "categorical"
    if c.semantic == "ordinal":
        return "ordinal"
    return "numeric"


def numeric_outcome(tp: TrainPartition, cols: list[str], contract: WideContract, dictionary: DataDictionary, min_cell: int) -> pd.DataFrame:
    """Per numeric column: distribution among fallers vs non-fallers (TRAIN), SMD and univariate AUROC (descriptive)."""
    rows = []
    for col in cols:
        x = numeric(tp.frame[col]).to_numpy()
        x1, x0 = x[tp.y == 1], x[tp.y == 0]
        o1, o0 = np.isfinite(x1), np.isfinite(x0)
        a, lo, hi = auroc(x, tp.y)
        rows.append({"column": col, "domain": dictionary.domain_label(col), "n_observed_fall": cell(int(o1.sum()), min_cell),
                     "n_observed_no_fall": cell(int(o0.sum()), min_cell),
                     "missing_pct_fall": fnum(100.0 * (1 - o1.mean()), 2) if len(x1) else None,
                     "missing_pct_no_fall": fnum(100.0 * (1 - o0.mean()), 2) if len(x0) else None,
                     "mean_fall": fnum(np.nanmean(x1), 4) if o1.any() else None, "mean_no_fall": fnum(np.nanmean(x0), 4) if o0.any() else None,
                     "median_iqr_fall": med_iqr(pd.Series(x1)), "median_iqr_no_fall": med_iqr(pd.Series(x0)),
                     "smd_fall_vs_no_fall": fnum(smd(x1, x0), 3), "univariate_auroc": fnum(a, 3), "auroc_ci_low": fnum(lo, 3), "auroc_ci_high": fnum(hi, 3),
                     "basis": TRAIN_ONLY})
    return pd.DataFrame(rows)


def categorical_outcome(tp: TrainPartition, cols: list[str], contract: WideContract, dictionary: DataDictionary, min_cell: int,
                        max_levels: int = 25) -> pd.DataFrame:
    """Per column and level (NULL as its own level): N, %, events, prevalence (Wilson 95% CI) and the prevalence ratio versus the most
    frequent level (Katz 95% CI); sparse levels and levels with no events / only events (separation) are flagged. TRAIN only."""
    rows = []
    for col in cols:
        s = tp.frame[col].astype("string").fillna("NULL")
        vc = s.value_counts()
        ref = str(vc.index[0])
        ref_mask = (s == ref).to_numpy()
        ref_n, ref_e = int(ref_mask.sum()), int(tp.y[ref_mask].sum())
        for i, (level, n) in enumerate(vc.items()):
            if i >= max_levels:
                break
            m = (s == level).to_numpy()
            e = int(tp.y[m].sum())
            sup = small(n, min_cell) or small(e, min_cell) or small(n - e, min_cell)
            p, plo, phi = wilson(e, int(n))
            pr = prevalence_ratio(e, int(n), ref_e, ref_n) if str(level) != ref else (1.0, None, None)
            sep = "no events in this level" if e == 0 and n >= min_cell else ("only events in this level" if e == n and n >= min_cell else "")
            rows.append({"column": col, "domain": dictionary.domain_label(col), "level": str(level), "is_reference_level": str(level) == ref,
                         "n": cell(n, min_cell), "pct": fnum(100.0 * n / tp.n, 2) if not small(n, min_cell) else SUPPRESSED,
                         "events": cell(e, min_cell), "prevalence": SUPPRESSED if sup else fnum(p, 4),
                         "prevalence_ci_low": None if sup else fnum(plo, 4), "prevalence_ci_high": None if sup else fnum(phi, 4),
                         "prevalence_ratio_vs_reference": None if sup else fnum(pr[0], 3), "pr_ci_low": None if sup else fnum(pr[1], 3),
                         "pr_ci_high": None if sup else fnum(pr[2], 3), "sparse": bool(n < min_cell or n / tp.n < 0.005),
                         "separation": sep, "missing_category": str(level) == "NULL", "basis": TRAIN_ONLY})
    return pd.DataFrame(rows)


def small(n: Any, min_cell: int) -> bool:
    n = int(n)
    return 0 < n < min_cell


def correlation_matrix(tp: TrainPartition, cols: list[str], method: str = "spearman") -> pd.DataFrame:
    """Pairwise-complete correlation on TRAIN over columns with at least two distinct observed values."""
    x = pd.DataFrame({c: numeric(tp.frame[c]) for c in cols})
    usable = [c for c in cols if x[c].nunique(dropna=True) >= 2]
    return x[usable].corr(method=method, min_periods=30)


def correlation_pairs(tp: TrainPartition, spear: pd.DataFrame, pear: pd.DataFrame, contract: WideContract, dictionary: DataDictionary,
                      extended_sources: set[str], *, threshold: float = 0.3, nominal: list[str] | None = None,
                      nominal_partners: list[str] | None = None) -> pd.DataFrame:
    """Pairs with |Spearman| >= threshold (TRAIN), with Pearson (phi for two 0/1 columns), redundancy flags and, for nominal
    categorical columns, Cramér's V."""
    rows = []
    cols = list(spear.columns)
    kinds = {c: _kind(contract, c, tp.frame[c]) for c in cols}
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = spear.at[a, b]
            if not np.isfinite(r) or abs(r) < threshold:
                continue
            pr = pear.at[a, b] if a in pear.index and b in pear.columns else np.nan
            both = int((tp.frame[a].notna() & tp.frame[b].notna()).sum())
            level = "CANDIDATE_DUPLICATE_DEFINITION" if abs(r) >= 0.95 else ("HIGHLY_REDUNDANT" if abs(r) >= 0.8 else ("MODERATE" if abs(r) >= 0.5 else "WEAK"))
            rows.append({"column_a": a, "column_b": b, "kind": f"{kinds[a]}-{kinds[b]}", "spearman": fnum(r, 4),
                         "pearson_or_phi": fnum(pr, 4), "cramers_v": None, "n_pairwise": both, "redundancy": level,
                         "same_source": dictionary.source(a) == dictionary.source(b), "domain_a": dictionary.domain_label(a), "domain_b": dictionary.domain_label(b),
                         "a_in_extended": a in extended_sources, "b_in_extended": b in extended_sources, "basis": TRAIN_ONLY})
    for a in nominal or []:
        for b in nominal_partners or []:
            if a == b or a not in tp.frame.columns or b not in tp.frame.columns:
                continue
            v = cramers_v(tp.frame[a], tp.frame[b])
            if v is None or v < threshold:
                continue
            rows.append({"column_a": a, "column_b": b, "kind": f"nominal-{_kind(contract, b, tp.frame[b])}", "spearman": None, "pearson_or_phi": None,
                         "cramers_v": fnum(v, 4), "n_pairwise": int((tp.frame[a].notna() & tp.frame[b].notna()).sum()),
                         "redundancy": "CANDIDATE_DUPLICATE_DEFINITION" if v >= 0.95 else ("HIGHLY_REDUNDANT" if v >= 0.8 else ("MODERATE" if v >= 0.5 else "WEAK")),
                         "same_source": dictionary.source(a) == dictionary.source(b), "domain_a": dictionary.domain_label(a), "domain_b": dictionary.domain_label(b),
                         "a_in_extended": a in extended_sources, "b_in_extended": b in extended_sources, "basis": TRAIN_ONLY})
    out = pd.DataFrame(rows)
    if len(out):
        key = out["spearman"].abs().fillna(out["cramers_v"])
        out = out.assign(_k=key).sort_values("_k", ascending=False).drop(columns="_k").reset_index(drop=True)
    return out


def redundancy_clusters(spear: pd.DataFrame, dictionary: DataDictionary, extended_sources: set[str], threshold: float = 0.8) -> pd.DataFrame:
    """Groups of columns linked by |Spearman| >= threshold (average-linkage hierarchical clustering on 1 - |rho|, TRAIN)."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    cols = list(spear.columns)
    if len(cols) < 2:
        return pd.DataFrame()
    a = np.nan_to_num(np.abs(spear.to_numpy()), nan=0.0)
    d = np.clip(1.0 - a, 0.0, 1.0)
    np.fill_diagonal(d, 0.0)
    labels = fcluster(linkage(squareform(d, checks=False), method="average"), t=1.0 - threshold, criterion="distance")
    rows = []
    for k in sorted(set(labels)):
        members = [c for c, lab in zip(cols, labels) if lab == k]
        if len(members) < 2:
            continue
        sub = a[np.ix_([cols.index(m) for m in members], [cols.index(m) for m in members])]
        off = sub[~np.eye(len(members), dtype=bool)]
        rows.append({"cluster": f"R{len(rows) + 1:02d}", "n_columns": len(members), "columns": "; ".join(members),
                     "min_abs_spearman": fnum(off.min(), 3), "max_abs_spearman": fnum(off.max(), 3),
                     "domains": "; ".join(sorted({dictionary.domain_label(m) for m in members})),
                     "extended_sources_in_cluster": "; ".join(m for m in members if m in extended_sources), "basis": TRAIN_ONLY})
    return pd.DataFrame(rows)


def univariate_association(tp: TrainPartition, cols: list[str], contract: WideContract, dictionary: DataDictionary, min_cell: int,
                           availability: dict[str, str]) -> pd.DataFrame:
    """One row per candidate column (TRAIN): SMD between fallers and non-fallers, univariate AUROC, and for 0/1 columns the prevalence
    among 1 vs 0 with the prevalence ratio. UNIVARIATE DESCRIPTIVE ASSOCIATION — NOT CAUSAL, NOT MULTIVARIATE IMPORTANCE."""
    rows = []
    for col in cols:
        kind = _kind(contract, col, tp.frame[col])
        x = numeric(tp.frame[col]).to_numpy() if kind != "categorical" else None
        r: dict[str, Any] = {"column": col, "domain": dictionary.domain_label(col), "kind": kind, "contract_role": contract.get(col).role,
                             "available_at_prediction_time": availability.get(col, ""),
                             "missing_pct": fnum(100.0 * tp.frame[col].isna().mean(), 2)}
        if x is not None:
            obs = np.isfinite(x)
            r["smd_fall_vs_no_fall"] = fnum(smd(x[tp.y == 1], x[tp.y == 0]), 3)
            a, lo, hi = auroc(x, tp.y)
            r.update({"univariate_auroc": fnum(a, 3), "auroc_ci_low": fnum(lo, 3), "auroc_ci_high": fnum(hi, 3), "n_observed": cell(int(obs.sum()), min_cell)})
            if kind == "binary":
                one, zero = obs & (x == 1), obs & (x == 0)
                a1, n1, a0, n0 = int(tp.y[one].sum()), int(one.sum()), int(tp.y[zero].sum()), int(zero.sum())
                sup = any(small(v, min_cell) for v in (a1, n1 - a1, a0, n0 - a0))
                pr, plo, phi = prevalence_ratio(a1, n1, a0, n0)
                r.update({"n_flag_1": cell(n1, min_cell), "prevalence_if_1": SUPPRESSED if sup else fnum(a1 / n1 if n1 else None, 4),
                          "prevalence_if_0": SUPPRESSED if sup else fnum(a0 / n0 if n0 else None, 4),
                          "prevalence_ratio_1_vs_0": None if sup else fnum(pr, 3), "pr_ci_low": None if sup else fnum(plo, 3),
                          "pr_ci_high": None if sup else fnum(phi, 3)})
        else:
            s = tp.frame[col].astype("string").fillna("NULL")
            r["cramers_v_with_outcome"] = fnum(cramers_v(s, pd.Series(tp.y, index=s.index)), 3)
        r["basis"] = TRAIN_ONLY
        rows.append(r)
    return pd.DataFrame(rows)


#: Table 1 variables: (label, column or derived key, kind); derived keys are computed by _table1_series
TABLE1 = (
    ("Age, years", "Age_At_Index", "continuous"), ("Age band", "_age_band", "categorical"), ("Sex", "_sex", "categorical"),
    ("Birth date suspect (default birth date)", "Birth_Date_Suspect_Ind", "binary"), ("HMO seniority, months", "HMO_Seniority_Months", "continuous"),
    ("Any fall/fracture since 2022 [falls]", "Prior_Fall_Since_Study_Start_Ind", "binary"), ("Fall/fracture events, 365 days", "Prior_Fall_Count_365D", "continuous"),
    ("Difficulty walking diagnosis 719.7 [mobility_problems]", "Gait_Disorder_Since_Study_Start_Ind", "binary"),
    ("Dementia registry [dementia]", "Registry_Dementia_Ind", "binary"), ("Home-confined registry [housebound]", "Registry_Home_Confined_Ind", "binary"),
    ("Diabetes registry T1 or T2 [diabetes_mellitus]", "_diabetes", "binary"), ("Chronic renal failure registry [chronic_kidney_disease]", "Registry_Chronic_Renal_Failure_Ind", "binary"),
    ("COPD registry [copd]", "Registry_COPD_Ind", "binary"), ("Asthma registry [asthma]", "Registry_Asthma_Ind", "binary"),
    ("Hypertension registry [hypertension]", "Registry_Blood_Pressure_Ind", "binary"), ("Liver registry [liver_problems]", "Registry_Liver_Ind", "binary"),
    ("SMI registry level >= 1 [severe_mental_illness]", "_smi", "binary"), ("Suicide-attempt registry [self_harm]", "Registry_Suicide_Attempt_Ind", "binary"),
    ("Heart-disease registry (Phase 2)", "Registry_Heart_Disease_Ind", "binary"),
    ("Distinct active substances [polypharmacy_count_120d]", "Distinct_Active_Substance_Count", "continuous"),
    ("Five or more active substances", "Polypharmacy_5Plus_Ind", "binary"), ("Visits, 365 days", "Visit_Count_365D", "continuous"),
    ("Diagnosis records, 365 days", "Diagnosis_Count_365D", "continuous"), ("Hospitalisations, 365 days (timing unconfirmed)", "Hospitalization_Count_365D", "continuous"),
    ("MEFI frailty group", "_mefi", "categorical"), ("Charlson group", "_cci", "categorical"),
    ("Nurse assessment performed", "_assessed", "binary"), ("Nurse mobility score", "_mobility", "categorical"),
    ("Uses walking aid (nurse)", "_walking_aid", "categorical"),
)


def _table1_series(frame: pd.DataFrame, key: str) -> pd.Series | None:
    if not key.startswith("_"):
        return frame[key] if key in frame.columns else None
    if key == "_age_band" and "Age_At_Index" in frame.columns:
        return pd.cut(numeric(frame["Age_At_Index"]), [-np.inf, 65, 70, 75, 80, 85, 90, np.inf], right=False,
                      labels=["<65", "65-69", "70-74", "75-79", "80-84", "85-89", "90+"]).astype("string")
    if key == "_sex" and "Gender_Code" in frame.columns:
        code = frame["Gender_Code"].astype("string")
        desc: dict[str, str] = {}
        if "Gender_Desc" in frame.columns:
            pairs = pd.DataFrame({"c": code, "d": frame["Gender_Desc"].astype("string")}).dropna()
            desc = {str(k): str(v.value_counts().index[0]) for k, v in pairs.groupby("c")["d"] if len(v)}
        return code.map(lambda v: (f"code {v} ({desc[str(v)]})" if str(v) in desc else f"code {v}") if pd.notna(v) else pd.NA).astype("string")
    if key == "_diabetes" and {"Registry_Diabetes_T1_Ind", "Registry_Diabetes_T2_Ind"} <= set(frame.columns):
        a, b = numeric(frame["Registry_Diabetes_T1_Ind"]), numeric(frame["Registry_Diabetes_T2_Ind"])
        return ((a.fillna(0) > 0) | (b.fillna(0) > 0)).astype("float64").where(a.notna() | b.notna())
    if key == "_smi" and "Registry_SMI_Level" in frame.columns:
        return (numeric(frame["Registry_SMI_Level"]).fillna(0) >= 1).astype("float64")
    if key == "_mefi" and "MEFI_Group_At_Index" in frame.columns:
        return frame["MEFI_Group_At_Index"].astype("string").fillna("not assessed")
    if key == "_cci" and "CCI_Group" in frame.columns:
        return frame["CCI_Group"].astype("string").fillna("not computed")
    if key == "_assessed" and "Assessment_Not_Performed_Ind" in frame.columns:
        return 1.0 - numeric(frame["Assessment_Not_Performed_Ind"])
    if key == "_mobility" and "Mobility_Score" in frame.columns:
        return frame["Mobility_Score"].astype("string").fillna("not assessed")
    if key == "_walking_aid" and "Uses_Walking_Aid_Ind" in frame.columns:
        return frame["Uses_Walking_Aid_Ind"].astype("string").fillna("not assessed")
    return None


def table1(frame: pd.DataFrame, y: np.ndarray, *, basis: str, min_cell: int) -> pd.DataFrame:
    """Descriptive Table 1: no fall / fall / total side by side (n (%) or median [IQR]), SMD fall vs no fall, missing n. No p-values."""
    if basis == TRAIN_ONLY and not isinstance(frame, pd.DataFrame):
        raise TypeError("table1 expects a DataFrame")
    y = np.asarray(y, dtype=np.int64)
    g1, g0 = y == 1, y == 0
    n1, n0, n = int(g1.sum()), int(g0.sum()), len(y)
    rows = [{"variable": "N (rows)", "level": "", "no_fall": str(n0), "fall": str(n1), "total": str(n), "smd_fall_vs_no_fall": None, "missing": ""}]
    for label, key, kind in TABLE1:
        s = _table1_series(frame.reset_index(drop=True), key)
        if s is None:
            continue
        s = s.reset_index(drop=True)
        miss = int(s.isna().sum())
        if kind == "continuous":
            x = numeric(s)
            rows.append({"variable": label, "level": "median [IQR]", "no_fall": med_iqr(x[g0]), "fall": med_iqr(x[g1]), "total": med_iqr(x),
                         "smd_fall_vs_no_fall": fnum(smd(x[g1].to_numpy(), x[g0].to_numpy()), 3), "missing": str(cell(miss, min_cell))})
        elif kind == "binary":
            x = numeric(s)
            k0, k1, k = int((x[g0] == 1).sum()), int((x[g1] == 1).sum()), int((x == 1).sum())
            rows.append({"variable": label, "level": "yes, n (%)", "no_fall": _np(k0, n0, min_cell), "fall": _np(k1, n1, min_cell), "total": _np(k, n, min_cell),
                         "smd_fall_vs_no_fall": fnum(smd(x[g1].to_numpy(), x[g0].to_numpy()), 3), "missing": str(cell(miss, min_cell))})
        else:
            st = s.astype("string").fillna("missing")
            levels = list(st.value_counts().index)[:12]
            for lv in sorted(levels, key=str):
                ind = (st == lv).astype("float64")
                k0, k1, k = int(ind[g0].sum()), int(ind[g1].sum()), int(ind.sum())
                rows.append({"variable": label, "level": str(lv), "no_fall": _np(k0, n0, min_cell), "fall": _np(k1, n1, min_cell), "total": _np(k, n, min_cell),
                             "smd_fall_vs_no_fall": fnum(smd(ind[g1].to_numpy(), ind[g0].to_numpy()), 3), "missing": ""})
    out = pd.DataFrame(rows)
    out["basis"] = basis
    return out


def _np(k: int, n: int, min_cell: int) -> str:
    c = cell(k, min_cell)
    return SUPPRESSED if c == SUPPRESSED else (f"{k} ({100.0 * k / n:.1f}%)" if n else str(k))
