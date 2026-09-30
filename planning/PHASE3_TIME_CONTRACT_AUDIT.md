# Phase 3 time-contract audit (2026-09-30)

Status: written before any Phase 3 result and before any Phase 2 VALIDATION result was returned. No row-level data was used. Every claim about the
real extract below is a **check the Windows preflight runs** (`falls_ml.phase3.timecontract`, V1-V9); nothing is assumed from this document.

## 1. What the DWH developer said (reported by the project lead)

1. Index_Date = 2025-01-01.
2. Clinical predictors include information available through the END of 2025-01-01.
3. The 180-day fall outcome EXCLUDES falls on 2025-01-01; the window starts on 2025-01-02.
4. The extraction does not include clinical events dated after Index_Date.

Corrected contract: `source_event_date <= Index_Date`, prediction at the start of the next day (END_OF_INDEX_DAY). Phases 1-2 used
`source_event_date < Index_Date` (START_OF_INDEX_DAY), pinned in `configs/meuhedet/wide_v1_efalls_mapping.yaml`; that file is **not changed** (the
historical runs and the running Phase 2 keep their contract). Phase 3 declares its own contract in `configs/meuhedet/phase3_time_contract.yaml`.

## 2. Consistency with the repository evidence

| Evidence | Old reading (start of day) | Developer's reading (end of day) | Verdict |
|---|---|---|---|
| Mapping: `Label_End_180D = Index_Date + 180` (checked on the real data in 0.4.x) | a 181-day window including the index day | a 180-day window Jan 2 ... Jun 30 | fits the developer better |
| Real D-00 findings (0.5.0): 4,168 `Last_Dx_Date = Index_Date`, 22 `Last_Fall_Date = Index_Date`; none reported AFTER the index day | "future" records | same-day history (A_SAME_DAY_INCLUSION) | consistent |
| Contract: `Fall_On_Index_Date_Ind` is FORBIDDEN "because an index-day event lies in the outcome window" | follows the old reading | an index-day fall is history | depends on V1 |
| EDA check F08 "fall on the index day but label 0" (WARNING under the old reading) | a label error | expected | the real count is not in the repository |
| Positive labels with an event ON Index_Date | allowed by the adapter check | must be 0 | **unverified -> V1 (HARD)** |

Conclusion: the explanation is consistent with every piece of evidence in the repository, but its decisive consequence (no outcome event on the index
day) has never been measured. Phase 3 therefore makes it a hard, read-only preflight check and never fits a model if it fails.

## 3. Checks (TRAIN + VALIDATION rows only; TEST labels never read)

| Check | Severity | What |
|---|---|---|
| V1 | HARD | no positive label with an event on/before Index_Date; minimum Days_To_Next_Fall_180D >= 1 |
| V2 | HARD | Label_End_180D - Index_Date = 180 on every labelled row |
| V3 | ATTESTATION | no predictor record date after Index_Date (13 record-date columns; billing-lag / unknown-timing sources excluded by documentation). If violated: those rows are UNKNOWN (bounded) and every source WITHOUT a row-level date loses the attestation (UNRESOLVED, exploratory only) |
| V4 | WARN | Days_Since_* equal Index_Date minus their date, never negative |
| V5 | INFO | index-day fall x label |
| V6 | INVESTIGATION | events on day +1 among patients with an index-day record vs others (one episode crossing midnight) |
| V7 | INVESTIGATION | the Phase 2 PROXY_DOMINANCE finding: Last_Fall_Date / Days_Since_Last_Fall / Index_Date / Next_Fall_Date_180D - consistency, next-minus-last-fall gaps, early-event concentration among recent fallers |
| V8 | INFO | availability-lag signals: Unsettled_Visit_Counter_Ind, Future_Dated_365D, Fallback_Exposure_365D, External_Care_Hidden_By_Billing_Lag_365D |
| V9 | HARD | exact reconciliation: FULL_LABELED = D00_CLEAN + D-00 rows, D-00 rows = the rows the reference build removed, label-NULL rows = the reference count |

## 4. What changes in feature eligibility (if V1-V3 pass)

| Before (Phase 2 rule on D00_CLEAN) | Phase 3 class | Why |
|---|---|---|
| UNSAFE: nurse forms (Mini-Cog, falls risk, get-up-and-go, function, home safety), visits, MEFI, nurse summary | SAFE_VERIFIED | their only defect was records ON the index day, which are history now; row-level dates <= Index_Date on every row |
| SAFE: fall counts, diagnosis counts | SAFE_VERIFIED, now on FULL_LABELED | the 4,168 D-00 patients are re-included |
| UNRESOLVED: medications, prescriptions, laboratories, registries, comorbidity / social status, 11 historical predictors | SAFE_ATTESTED | no row-level date: eligible on the DWH statement + V3, NOT row-verified; reported separately; removed by the VERIFIED_ONLY sensitivity |
| FORBIDDEN: billing-lag external-care totals, labels, follow-up, death, hospitalisations (timing unknown) | unchanged | post-index by construction or undocumented |
| Fall_On_Index_Date_Ind (FORBIDDEN under the old window) | not used in this freeze | now history, but its information is already in Days_Since_Last_Fall = 0; a dedicated feature needs a reviewed contract change |

## 5. Event timing is not information availability

The DWH statement and V3 are about EVENT dates. A record dated before Index_Date can still have been unavailable at the end of Index_Date (late nurse
documentation, backdated registry memberships, reimbursement claims, laboratory imports, MEFI batch computation, a score computed at extraction). The
extract has no entry timestamps, so availability is **an explicit assumption per source**, with a pre-declared risk (`availability` in the time
contract). HIGH / UNKNOWN risk: registries (backdating), the comorbidity / social status (construction undocumented), purchases outside Meuhedet
(claims lag). A pre-declared LOW_RISK sensitivity removes them; a gain that does not survive it is reported as **not established** until the DWH
answers Q-P3-02.

## 6. The 4,168 D-00 patients

Under the corrected contract their index-day records are history, so excluding them was a selection on (legitimate) same-day information that was also
associated with the outcome. They are re-included after V1, V2 and V9 pass, with a pre-declared, outcome-blind partition rule (keyed hash of the
pseudonymised id; 60 / 20 / 20; the TEST-assigned 20% dropped unread). Re-inclusion does not remove every selection: a usable 180-day label is still
required (censoring by death or leaving the HMO is a post-index criterion) - reported as a limitation. D00_CLEAN is preserved: the reference run is
untouched, every row carries its D00_CLEAN membership, and the bridge tables restrict the same predictions to those patients.

## 7. The Phase 2 PROXY_DOMINANCE finding

Phase 2 stopped because `falls_days_since_last` held 89% of the XGBoost permutation AP loss. A dominant recency effect can be genuine (recent fallers fall
again) or an artefact (one injury episode recorded on several days appears both as the last fall and as the next fall). Phase 3: V7 (before any model;
investigation stop if flagged), the pre-declared ablation `P3_ALL_NO_FALL_RECENCY` (LASSO + XGBoost, identical patients, bounded paired bootstrap), and
the PROXY_DOMINANCE gate kept for every other feature (for this one it stops only if V7 flagged it; the dominance is always reported).

## 8. Simplification

The row-level reconstruction machinery is no longer the core of Phase 3: if the developer is right, no row is UNKNOWN and every interval collapses to
the ordinary prediction. It stays as the safeguard for any post-index record, undated value or unreadable cell the preflight finds.
