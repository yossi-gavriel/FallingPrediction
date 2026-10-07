# FINAL CONSENSUS — Phase 5.1 / Phase 6 roadmap after the independent adversarial review

| Item | Value |
|---|---|
| Status | CONSENSUS ROADMAP. Nothing implemented, nothing trained, no real data read. Supersedes `docs/phase6/CLAUDE_STRATEGY.md` wherever the two differ. |
| Inputs | Lead proposal `docs/phase6/CLAUDE_STRATEGY.md` (commit `c375aba`); Astra/Codex adversarial review `docs/phase6/ASTRA_REVIEW.md` (commit `d3f20d2f9a39404ab02a31994bd2ed859d40c2a2`) |
| Base software | falls_ml 0.12.3, Phase 5 2.2.0 (unchanged) |
| Author | Lead Scientist (Claude), resolving every substantive review item as ACCEPT / PARTIALLY ACCEPT / REJECT |
| Date | 2026-10-07 |

**Operational figures used below** (supplied by the PI to the reviewer from the completed work-PC run; not reproduced here, to be confirmed against `COHORT_FACTS_2026.json` and `TOP3_CAPACITY_PRIMARY.csv`): 97,453 patients, 2,155 with a recorded fall in 180 days, 2,924 contact places at 3%, 1,086 captured (50.4% recall, 37.1% PPV). 55% recall on the same denominator = 1,186 captured, about 100 more patients.

---

## תקציר להנהלה

### איפה אנחנו היום

- יש מודל (Elastic Net) שפותח על צילום־מצב 01.01.2026 ונבדק בהצלבה פנימית בלבד. לפי הנתונים שנמסרו לסוקר, ברשימת 3% מהאוכלוסייה (2,924 אנשים) הוא תופס כ־1,086 מתוך 2,155 הנופלים ב־180 הימים הבאים, כלומר **כ־50% תפיסה ו־37% דיוק** (37 מכל 100 שנפנה אליהם ייפלו בפועל).
- הביקורת העצמאית (Astra) אישרה את הכיוון, אבל מצאה שבקוד שאימן את המודל יש **פגם מתודולוגי אחד מהותי**: סינון משתנים שהשתמש בתוצאות של כל המטופלים לפני חלוקת הקפלים. זה לא "דליפה" במובן של ידע מהעתיד, אבל זה אומר שהמספר 50% אינו הערכה נקייה לחלוטין עד שנתקן ונריץ שוב.
- עדיין **אין בדיקה על תקופה שהמודל לא ראה**. 50% הוא מספר פיתוח, לא מספר מאומת.

### מה מתקנים קודם

1. מסירים את סינון המשתנים האוטומטי לפי AUROC ומחליפים אותו בכללי קבילות שאינם משתמשים בתוצאה (תזמון, מקור, משמעות קלינית). כל שלב נלמד רק מתוך נתוני האימון של כל קפל.
2. מבררים את המשמעות של קבוצת צ'רלסון (CCI_Group) מול ה־DWH וקובעים קידוד אחד מראש.
3. מריצים מחדש את אותו ניסוי, על אותם מטופלים ואותם קפלים, ומדווחים **לפני / אחרי** — כולל בדיקות שליליות (ערבוב תוויות) שמוכיחות שהצינור חוזר לרמת מקריות כשאין מידע אמיתי.
4. רק המודל המתוקן הוא נקודת הייחוס. כל שיפור עתידי נמדד מולו, לא מול ה־50.4% ההיסטורי.

### איך ננסה לשפר מ־~50% תפיסה ב־3% התערבות

- **המדד הראשי מעכשיו:** כמה נופלים נתפסים ברשימת ה־3% (Recall@Top3), ולא "70% רגישות". ברשימה בגודל קבוע, כל נופל נוסף שנתפס הוא בדיוק פנייה אחת מיותרת פחות.
- **55% הוא שאיפה, לא סף הצלחה.** הצלחה תוגדר מראש כשיפור זוגי מובהק מעל "ההפרש המינימלי המשמעותי" (MOD) שההנהלה תקבע בכתב לפני שרואים תוצאות — למשל, 5 נופלים נוספים לכל 10,000 מבוטחים (כ־49 אנשים, כ־2.3 נקודות תפיסה). לא נחפש "עד שיופיע 55%".
- **הסיכוי הגבוה ביותר לשיפור אמיתי** הוא מידע חדש עם תאריכים ולא אלגוריתם חדש: שינויים אחרונים בתרופות מגבירות־נפילה, מרווחים בין אירועי נפילה קודמים, הידרדרות תפקודית/ניידות, ביקורי מיון ואשפוזים אחרונים, ותאריך האבחנה (סחרחורת, סינקופה, הפרעת הליכה) ולא רק "קיימת/לא קיימת".
- נבדוק בנוסף שתי משפחות מודל בלבד (Elastic Net ו־XGBoost מוגבל) ועוד גרסה אחת לא־לינארית מרוסנת. לא נריץ תחרות אלגוריתמים.
- **הבדיקה המכרעת** תהיה על צילום־מצב מאוחר יותר (01.07.2026) שתוצאותיו נשארות חתומות עד שיבשילו (אחרי 28.12.2026 + פיגור תיעוד נמדד), בתנאי שאיש מצוות הפיתוח לא נחשף לתוצאותיו.

### מה נעשה עם clustering

- **פרופיל הנופלים (מה משותף לנופלים?)** — נעשה מיד: טבלת שכיחות של כל מאפיין אצל נופלים מול לא־נופלים, הפרש, העשרה, ויחסי סיכון מותאמי גיל/מין. זה עונה על שאלת ההנהלה הראשונה גם בלי clustering.
- **פנוטיפים של נופלים (האם יש סוגים קליניים שונים של נופלים?)** — נעשה: אשכולות על 2,155 הנופלים בלבד, על פאנל קליני קטן שנקבע מראש, 2–5 אשכולות, עם מבחן יציבות (100 דגימות חוזרות). ייתכן שהתשובה תהיה "אין פנוטיפים יציבים" — וזו תוצאה לגיטימית.
- **פנוטיפים בכלל האוכלוסייה** — נדחה לשלב הבא, לאחר שנראה אם יש פנוטיפים יציבים בקרב הנופלים.
- **שימוש באשכולות לחיזוי** — לא בסבב הזה. האשכולות הם פונקציה של אותם משתנים שהמודל כבר רואה; הערך שלהם הוא **הבנה קלינית**, והוא ערך גם אם לא ישפרו את התחזית.

### חמשת הניסויים הראשונים (בסדר הזה)

1. **תיקון Phase 5.1** — אותו ניסוי, אותם קפלים, בלי סינון AUROC; קובע את נקודת הייחוס המתוקנת ומדווח לפני/אחרי.
2. **ייצוג לא־לינארי מרוסן** — XGBoost מוגבל ל־20 תצורות רשומות וגרסת Elastic Net עם בסיס לא־לינארי קטן, על אותם משתנים ואותם קפלים.
3. **חבילת מידע אורכי אחת** — תרופות עם תאריכים, אירועי נפילה קודמים, הידרדרות תפקודית — רק אם ה־DWH מספק את הנתונים עם תאריכי זמינות; אחרת הניסוי נדחה (לא מוחלף באלגוריתמים).
4. **פרופיל נופלים + פנוטיפים של נופלים** — במקביל, לא תלוי בשיפור חיזוי.
5. **אימות זמני חתום אחד** — המודל שנבחר מול ה־Elastic Net המתוקן, על צילום 01.07.2026, פתיחת תוצאות פעם אחת, כלל הצלחה שנקבע מראש. תוצאה לא חד־משמעית = נשארים עם המודל המתוקן ולא מחפשים הלאה על אותם נתונים.

---

## 1. What changed from the lead proposal

The review's goal was to reduce sprawl and to remove weak links. I withdraw the following elements of `CLAUDE_STRATEGY.md`; the reasons are in section 2.

| Withdrawn | Replaced by |
|---|---|
| C-01 fold-specific AUROC exclusion inside outer folds | Removal of the automatic outcome-AUROC membership gate; label-free admissibility gates; a training-only forensic diagnostic that reports and never excludes |
| T-01's 0.75 margin as a reason to let the 2.2.0 estimates "stand" | T-01 as a read-only inventory of what the screen did; no validity certificate; the corrected run is the reference regardless |
| The five-criterion ≥ 70%-sensitivity verdict as the ongoing primary rule | Kept only as a historical audit output of Phase 5.1; the primary endpoint is Recall@Top3 with a separate preregistered success rule |
| "CI > 0 and point estimate ≥ MOD → confirmed" | Lower paired 95% limit > MOD → confirmed; five result categories (section 4.6) |
| Capacity allocated among labelled patients as the operational estimand | Allocation among all baseline-eligible patients before label usability; the labelled-subcohort version kept for reconciliation |
| "Death-exclusion sensitivity" | Withdrawn (survivor selection); replaced by an outcome/death/censoring estimand audit and bounds for unascertainable selected patients |
| "Freeze Phase 6 before any 2026-07-01 outcome exists" (Nov–Dec 2026) | Impossible as of 2026-10-07; replaced by a custodial access inventory and "before any validation outcome is accessed by the development team" |
| Pooled-OOF equal-sensitivity comparison as part of a verdict | Descriptive reconciliation output only |
| Domain additions × ablations × three families with full re-tuning | OLD vs ADMISSIBLE (primary), legacy ALL (audit only), one source-risk contrast (NO_NEW_REGISTRY) |
| LASSO as a separately tuned family | The l1_ratio = 1 endpoint of the ENET path |
| Optional sensitivity S-02 "tune by log loss and see whether the verdict agrees" | One registered smoother-tuning development arm, shared by all serious candidates |
| Phase 7 as the first place for longitudinal information | Extraction and specification start now (Track C); one bounded bundle experiment before the model freeze if data arrive |
| No clustering protocol | A clinical-understanding track with three separated questions (section 4.10) |
| PPV = 0.10 precision illustration | Replaced by precision planning from development overlap and discordant selections, at the supplied PPV ≈ 0.37 |

What the review confirmed and I keep unchanged: the audit's diagnosis of the defect; nested medians/levels/indicators/scales; ENET as principal comparator; fixed-capacity anchoring; development versus temporal-validation distinction; protected historical artefacts and new output folders; the data-request direction; silent prospective validation and an intervention pathway before implementation; no neural, LLM or stacking work.

---

## 2. Classification of every substantive review item

Legend: **A** = ACCEPT, **PA** = PARTIALLY ACCEPT, **R** = REJECT. "Reason" is the scientific reason, kept short.

### 2.1 Phase 5.1 correction (review §3, §4)

| Review item | Decision | Reason |
|---|---|---|
| C-01 as proposed leaves inner-validation influence; remove the screen or nest it at both levels | **A** | A screen fixed per outer-training fold still lets inner-validation labels shape the candidate universe the inner loop chooses from. Removal is simpler and complete. |
| AUROC ≥ 0.80 is not a leakage detector; admissibility is timing / lineage / semantics | **A** | A legitimately strong predictor can exceed 0.80 and a weak or joint leak can stay below it. The gate measures strength, not provenance. The label-free gates (sealing, name patterns, timing classes, provenance classes, V3 attestation) already carry the admissibility logic. |
| T-01's 0.75 margin does not rehabilitate historical estimates | **A** | Fold-to-fold variation of a univariate AUROC is not uniformly ±0.02 across missingness patterns and rare binaries, and a non-binding screen says nothing about the other repairs. T-01 becomes an inventory. |
| In the special case of zero exclusions the realised membership effect of the screen is bounded | **A** | Correct and useful: with zero exclusions the PRE/POST difference measures the other repairs, not the screen. This is a bound on one component, not a validity statement. |
| Keep the five-criterion 70% verdict as a historical audit only; preregister a separate 3% rule | **A** | The 70% point and the 3% point sit far apart on the ranking; a rule at one cannot certify the other. |
| Retain the historical 70% inner objective for the repair arm; log-loss tuning is an improvement branch | **A** | Changing the objective inside the "repair" would confound the PRE/POST comparison. The inner ENET path is computed once per inner fold; both selection policies read the same inner OOF predictions, so the second arm costs only the extra outer refits. |
| Lambda grid: a dimensionless lambda/lambda_max ratio with lambda_max computed inside each inner training fold, as part of the repair (C-05 not optional) | **A** | Cheap, aligns candidates by recipe, and makes inner selection strictly inner. Agreed that this is not evidence the old grid exposed outer labels. |
| Label-blind global coverage/constant gates may remain as QA stops but must not learn membership from holdout covariates | **PA** | The minimum-known-rows (100), constant and unreadable-share gates are outcome-free and conservative; I keep them as cohort-level QA stops (they stop a corrupted extract) and additionally recompute coverage inside each outer-training fold so that a feature's presence in X never depends on holdout rows. |
| CCI: document order first; categorical if only nominal; quarantine if undocumented; one encoding, never chosen by outer result; explicit missing/rare/unseen policy; fix tree semantics too | **A** | Thermometer coding assumes a documented order; a bare code list is not one. Comparing encodings by outcome would be a selection step. The linear path's current pooling of rare and unseen codes into the reference is not a defensible semantic policy; an explicit "other" level is. |
| Negative controls: boundary traps through the real input boundary; ~10 reduced-budget full-pipeline permutations with frozen folds; holdout-label mutation with a frozen split; a legitimate synthetic AUROC > 0.80 predictor that must be retained; no deterministic chance-verdict requirement | **A** | My single quick permutation and a perfect sentinel were too weak, and the sentinel was circular once the gate existed. A retained strong legitimate predictor is the right demonstration that strength and leakage differ. |
| PRE/POST is a combined effect of all repairs, not attributable to the screen; verify fold hashes, not just the seed; reuse historical predictions, do not rerun the historical model | **A** | Several components change together; no single-component attribution is warranted. Fold identity must be proven by hash. |
| Freeze the repair recipe before reading the eligibility table; log every amendment | **A** | Otherwise T-01 becomes a knob. The recipe in section 4.2 is frozen by this document; T-01 is read afterwards. |
| Nested CV estimates a specified procedure; it cannot undo prior human adaptation to the January 2026 data | **A** | Agreed and stated in every summary: Phase 5.1 is a corrected development estimate; only Experiment 5 is confirmatory. |
| C-03 promote within-fold Top3 outputs; C-06 per-fold calibration; C-08 limitations in summaries; C-10 versioning | **A** | As reviewed. |
| C-04 within-fold equal-sensitivity: descriptive only; pooled legacy output for reconciliation; neither confirmatory | **A** | Both are reporting sensitivities computed from saved predictions. |
| C-09 immutable PRE run with paired saved folds | **A** | As reviewed. |
| C-07 scope note: no controls on real data are run by the review | **A** | They are part of the registered Phase 5.1 protocol, run by the PI on the work PC. |

### 2.2 Operational endpoint, tuning and success rule (review §5)

| Review item | Decision | Reason |
|---|---|---|
| Recall@Top3 is the primary comparative endpoint; k = round-half-up(0.03·N); label-, order- and model-independent tie rule | **A** | At fixed N, events and k, TP, recall and PPV order models identically. The existing implementation already uses exact integer rounding and deterministic pseudonymous row-key ties (`capacity.py:47–53, 76–83`). |
| Do not tune hundreds of configurations directly on Top3 | **A** | One event moves a fold's recall by about 0.23 points; a discontinuous statistic over many near-tied candidates selects noise. |
| Policy: unweighted inner log loss → one-SE shortlist → within-inner-fold Recall@Top3 → simpler on near ties; freeze the tolerance in operational units; all selection inside outer training | **A** | A proper scoring rule for the search, the operational statistic only among a few near-equivalent candidates. Tolerance frozen here: a near tie is a mean difference of fewer than one captured patient per inner validation fold (≈ 0.3 recall points at ≈ 345 events per inner fold). |
| Upper-tail ranking surrogates (pairwise losses, partial AUC) DEFERRED | **A** | No evidence yet that the shortlist policy misses tail structure; a new loss needs a new protocol. |
| Forbidden: seed search, capacity jitter (2.8% / 3.2%), class-weight search, threshold engineering, accumulating optional sensitivities until a gain appears | **A** | Each is a hidden search over the test statistic. Neighbouring capacities are plotted to show fragility only. |
| Paired patient bootstrap with identical weights, capacity re-allocated and re-selected within fold; report discordant selections and selected-set overlap; no independent bootstraps, no fold t-tests | **A** | Already how `capacity.py` works; overlap and discordance explain paired precision better than event totals. |
| The OOF bootstrap is conditional on fitted models; not a confirmatory interval after model shopping; prefer compute on an untouched temporal test | **A** | As reviewed. Full-pipeline resampling is not budgeted. |
| Success categories: lower 95% limit > MOD → CONFIRMED_MATERIAL_GAIN_AT_3_PERCENT; lower limit > 0 and point ≥ MOD → PROMISING_MATERIAL_GAIN_UNCERTAIN; lower limit > 0 and point < MOD → SMALL_STATISTICAL_GAIN; interval includes 0 → INCONCLUSIVE; upper limit < 0 → WORSE_AT_CAPACITY | **A** | A point estimate above MOD with an interval merely above zero does not establish that the benefit exceeds MOD. |
| MOD declared by management in captured people per 10,000 eligible at a fixed 300 contacts per 10,000; provisional illustration 5 per 10,000 (≈ 49 patients, ≈ 2.3 recall points) | **A** (as placeholder) | Units are right. The number is a planning placeholder until management declares it from programme cost, uptake and intervention effectiveness (decision D-4). |
| Precision planning from development score overlap and discordant selections by simulation of the exact capacity rule; my PPV = 0.10 illustration was mismatched | **A** | At the supplied PPV ≈ 0.37 the single-model half-width is ≈ 1.75 points; the paired contrast depends on discordance, which only the development predictions can show. Added to Experiment 1's outputs. |
| One primary comparison, one test opening; inconclusive → stop and retain the comparator; a new hypothesis needs a new registered plan and a new untouched test | **A** | This is the only rule that prevents searching until 55% appears. |
| 55% is a working aspiration only | **A** | About 100 extra captured patients on the supplied denominator; not a target, threshold or stop rule. |
| "If development uses outer results to choose a winner, label that CV performance exploratory" and include procedure selection inside the inner loop where the selected procedure's outer performance is reported | **A** | Inner OOF predictions of every arm exist; selecting the procedure within outer training costs nothing extra. Both the per-arm outer results and the selected-procedure outer result are reported; neither is the confirmatory estimate. |

### 2.3 Features, families and drops (review §6, §7, §11)

| Review item | Decision | Reason |
|---|---|---|
| Top-10 hypothesis ranking led by timed medication change, fall-episode intervals, functional transitions, diagnosis-specific timing and acute-care transitions; re-expressing existing counts is representation, not information | **A** | The OLD catalogue already holds windowed counts, recency bands, current/previous MEFI and latest function; only dated raw histories add ordering, episode identity and change. |
| Fixed representation bundle; no separate experiments for count ratios, accelerations and extra recency cutpoints | **A** | They are functions of the same information and would multiply comparisons. |
| ≤ 3 clinically declared interactions with shrinkage, not a pairwise search | **A** | XGB already learns interactions; a few named terms expose conditional effects to ENET without a search. |
| ENET KEEP (primary), XGB KEEP (one constrained challenger), one constrained additive challenger KEEP, LASSO DROP standalone, CatBoost/LightGBM/RF/ExtraTrees/survival/LTR DEFER, MoE/NN/LLM DROP | **A** | Algorithms cannot recover unextracted dates. Two learners plus one representation variant cover linear, additive-nonlinear and interaction structure. |
| The additive challenger as a penalised-spline logistic GAM | **PA** | Same scientific intent, different implementation: the locked environment has no GAM library and adding dependencies is a governance step. The additive arm is ENET on a declared basis (restricted cubic spline or the in-repo fractional-polynomial basis, `src/falls_ml/features/fractional_polynomial.py`) for ≤ 5 named continuous variables with knots fixed on training quantiles inside each fit, plus the ≤ 3 interactions. EBM/pyGAM is not added. |
| XGB capped at ≤ 20 registered configurations per outer-training selection | **A**, with a refinement | A fixed registered library replaces the 100-trial Optuna search. Because the library is fixed, both selection policies (historical 70% objective; log-loss shortlist) read the same inner OOF predictions, so XGB also shares fits across arms. Depth ≤ 4, min_child_weight ≥ 10, inner-only early stopping, log-loss tuning. |
| DROP the OLD/ALL/SAFE × domain × ablation × family grid; keep the primary safe universe plus one source-risk contrast; alias identical sets; timing-uncertain ALL cannot win | **A** | The 2.2.0 run already produced the descriptive domain tables once; the deployment decision needs the admissible universe, its OLD baseline and one audit contrast. |
| DROP repeated CCI encoding comparisons; DROP pooled-vs-within-fold thresholding as model-selection experiments; DROP prior-faller experts triggered by subgroup calibration; DROP raw K-means / outcome-coloured embeddings; DEFER backfill/redevelopment for "independent" confirmation | **A** | As reviewed. |

### 2.4 Clustering / phenotyping (review §8)

| Review item | Decision | Reason |
|---|---|---|
| Material omission: no profiling/phenotyping protocol | **A** | Management asked; my strategy answered a different question (subgroup calibration). |
| A. Faller profiling: prevalence among fallers and non-fallers, difference, enrichment, coverage; risk differences/ratios from P(fall \| feature); age/sex-adjusted and fuller adjusted associations with intervals; small declared co-occurrence list; FDR control; small-cell suppression; first deliverable | **A** | Answers "what is common among fallers" with the correct denominators and without clustering. Enrichment and risk ratio are different quantities and both are reported. |
| B. Faller-only phenotyping: concise prespecified baseline panel, duplicates removed, exclude future severity/treatment; Gower + PAM primary; 2–5 clusters; size floor max(100, 5% of fallers); ≥ 100 resamples; matched Jaccard ≥ 0.75 as screening; clinician naming blind to risk rankings; "no reproducible phenotypes" is an acceptable outcome | **A** | Conditioning on future falling induces associations, so phenotypes are descriptive of recorded fallers, not causal types. Stability criteria prevent a one-off partition. |
| C. Full-population phenotyping: label-blind; cluster count never chosen by risk separation; lock partition on development, assign later population, then estimate risks; ≥ 1% size floor; deferred under the strict budget | **PA** | Deferred, not dropped: it runs after B if B yields stable, interpretable phenotypes, as a CLARA-style medoid method with a frozen assignment rule. Management explicitly asked for population phenotypes; the deferral is sequencing, not dismissal. |
| Methods: Gower + PAM KEEP; hierarchical as one sensitivity; LCA possible alternative; FAMD/HDBSCAN/k-prototypes DEFER; raw K-means / outcome-selected UMAP DROP | **PA** | Gower + PAM primary and average-linkage hierarchical on the same Gower matrix as the one sensitivity, both implementable with numpy/scipy under the locked dependencies. LCA is not available in the locked environment and is not added now. |
| D. Predictive use: whole clustering pipeline inside inner and outer training; heldout assignment by frozen rule; never a full-cohort cluster ID in X; not in the first five experiments; expected gain small | **A** | Clusters are functions of existing predictors; their benefit would be representation only, and XGB/ENET interactions already use the same variables. |
| Expected value mainly clinical understanding; accept a useful phenotype analysis with zero Top3 gain | **A** | Stated plainly to management (Hebrew section). |

### 2.5 Temporal validation, estimand, death and censoring (review §9)

| Review item | Decision | Reason |
|---|---|---|
| July's outcome window has begun; "before any outcome exists" is impossible; untouched means access-controlled; written custodian access inventory; if contaminated, July is exploratory and a later sealed period is used | **A** | Correct on the calendar. Untouched is a property of access, not of time. |
| An October-frozen model is not a prospectively deployed July model; call the July test a retrospective temporal test of unseen outcomes; document training records that became available after July 1 | **A** | Honest labelling; the training labels through June 30 plus lag could not all have existed on July 1. |
| Predictor state reconstructed as-of July 1 using creation/load/version times, not event dates alone; V21's August 1 study end cannot certify December-mature July labels; new label cutoff and completeness evidence; freeze date never chosen after seeing performance | **A** | A 2027 query for "events before July 1" includes backfilled knowledge. This makes the availability metadata (R-03, expanded) a prerequisite of Experiment 5, not an optional improvement. |
| Score predictors and lock predictions first; a custodian verifies maturity without returning performance; mature labels linked in a separate stage | **A** | The Phase 4 write-once pattern already does this. |
| Changing or dropping an unavailable frozen predictor at test time changes the model: stop, or evaluate a separately prespecified fallback artefact; Mode B equivalence by scores/configuration within tolerance, not pickle bytes | **A** | As reviewed; a fallback artefact (corrected ENET on OLD) is frozen alongside the champion. |
| Capacity denominator: allocate 3% among all baseline-eligible patients before label usability; report the ascertainable share and the difference from the labelled-subcohort estimand | **A** (Phase 6 P0) / **PA** (Phase 5.1) | For deployment the quota is filled before follow-up is known. In Phase 5.1 training requires labels and the historical PRE/POST must stay on the labelled subcohort; the censored/unlabelled rows are counted (already reported in preflight check P5) and, if they can be scored outcome-blind by fold without delaying the repair, the operational-denominator version is added as a secondary descriptive output. |
| Death: "death as non-event" was my shorthand, not proven by code; audit the actual label semantics; death before any fall can be a known non-fall for cumulative incidence when ascertainment is complete; loss to follow-up is unknown, not negative; no survivor-conditioned primary analysis; IPCW/competing-risk evaluation only if censoring is material or selective, else evaluable subcohort with bounds | **A** | The code uses the supplied binary label and drops non-binary rows; the schema says a negative requires full personal follow-up. The estimand audit (which of death, disenrolment and administrative censoring yield NULL, 0 or 1) is a P0 question to the DWH. Pre-declared: if more than 5% of the selected 3% have an unascertainable outcome, an IPCW sensitivity is required; otherwise bounds (unascertained selected patients counted as non-captures for the lower bound and as captures for the upper bound) suffice. |
| Transport gates (AUROC −0.03, slope 0.80–1.25, CITL 0.20) are pragmatic triggers, not constants; comparing temporal recall with a CV lower bound is not a paired test; investigation cannot license new model searches on the test | **A** | The only inferential statement in Experiment 5 is the paired champion-versus-comparator contrast; the transport check of each frozen model against its own development estimate is descriptive with intervals. |
| Direction consistency is supportive replication evidence, not a veto | **A** | A sealed-test material gain stands on its own; a conflicting development sign is explained, not used to overrule it. |
| Subgroups: measure selection share and capture among the globally selected 3% within each subgroup; descriptive with intervals | **A** | Independent per-subgroup quotas answer a different allocation question. |
| 2025 backfill DEFER; never an independent confirmation of January 2026 | **A** | January 2026 has been studied twice. |
| Phase 7: forward time splits, group-aware dependence, embargo covering outcome windows and availability lag | **A** | Patient grouping and temporal separation solve different problems. |

### 2.6 V22 data request and label validation (review §10)

| Review item | Decision | Reason |
|---|---|---|
| MUST: source event time and available-at/creation/load/version time for every primary predictor source, registry membership history including updates/overwrites/batch times; view SQL hashes, version and extraction cutoff | **A** (expanded from R-03/R-10) | Deployability and as-of reconstruction depend on it; it is also the prerequisite for Experiment 5. |
| MUST: label/source/follow-up audit with on-site clinician-adjudicated stratified samples returning counts and intervals; aggregate source counts alone do not estimate specificity or missed falls | **A** (expanded from R-05) | Chart review of positives estimates PPV; sampled negatives or linked records are needed for missed falls. |
| MUST: medication prescribing/dispensing/start/change dates with ATC, supply/dose when reliable, stop/cancel history, coverage and outside-HMO capture | **A** (expanded from R-02) | Highest-value new time-bounded source. |
| MUST: a custodially sealed later evaluation snapshot with access log, cutoff and identical definitions | **A** (modified R-09) | July only if the access and availability conditions hold. |
| HIGH: repeated mobility/ADL/gait, device transitions, physiotherapy/rehab dates; historical fall/fracture records with encounter/episode IDs and availability times; dated diagnosis histories; ED/hospital encounter type with admission/discharge/arrival times; repeated MEFI components and GUG/TUG values with instrument/version/computation time; coarse clinic/district retained locally | **A** | Raised from my P2/P3; these carry the trajectory information the OLD summaries discard. |
| NICE: restored hypertension/CKD/transplant with true lineage; a small preregistered vitals/labs panel; 2025 backfill | **A** | Comparability and exploratory value; low expected Top3 gain. |
| NOT WORTH IT: untimed static registries, overlapping deficit sums, undocumented risk codes, post-index purchases, raw text/LLM extraction | **A** | Repackaging or unresolved timing. |
| L-01's overall PPV ≥ 0.80 gate is not universal; prespecify the intended outcome, acceptable accuracy and precision by source and code family; keep any outcome revision in a separate protocol | **A** | A single global PPV hides source-specific failure; the Phase 5.1 repair keeps its labels fixed. |

### 2.7 Roadmap-level dispositions (review §12, §13)

| Review item | Decision | Reason |
|---|---|---|
| Priority order: P0 assurance → P0 correction → P1 representation → P1 information → P2 clinical understanding in parallel → P0 confirmation | **A** | Adopted as the roadmap skeleton (section 5). |
| Five bounded protocols; champion frozen before the fifth; if the bundle experiment cannot finish before the freeze, freeze the best eligible procedure from 1–2 and defer the bundle to a future test | **A** | Adopted with the refinements in section 6. |
| Only Experiment 1 must precede new performance claims; extraction, specification and descriptive design proceed in parallel | **A** | Resolves item 1 of the mandate. |
| D-1 add a clinical-understanding track; D-2 reject fold-specific screen; D-3 July conditional; D-4 lower-CI-over-MOD; D-5 raise function/event histories; D-6 defer backfill; D-7 accept versioning; M-1…M-10 as reviewed | **A** | Each is resolved above. |

**Items I reject:** none of the review's substantive recommendations is rejected. Two are accepted only in part (the GAM implementation route and the Phase 5.1 operational denominator), and the full-population clustering is deferred rather than dropped, for the reasons given.

---

## 3. Answers to the reviewer's sixteen questions

| # | Answer |
|---|---|
| 1 | Yes, the automatic AUROC membership exclusion is removed. A training-only univariate AUROC is still computed inside each outer-training fold as a **forensic diagnostic**: it is reported, it never excludes, and an adjudicated availability violation (not a number) triggers a logged amendment and a new run. No fold-union veto exists. |
| 2 | PI to confirm from the completed run's `RUN_MANIFEST.json`, `work/PLAN.json` (code, config, schema, input and fold hashes) and whether 1,086/2,155 comes from `TOP3_CAPACITY_PRIMARY.csv` (within-fold quota) rather than the pooled descriptive table. The source computes the primary estimate within folds (`capacity.py` docstring). |
| 3 | From the source, 97,453 is almost certainly the **usable labelled** cohort: `prepare` drops rows whose label is not 0/1 before the frame is built (`data.py:307, 317`). The censored/unlabelled count is printed in preflight check P5 and must be read from `COHORT_FACTS_2026.json`. How many would have occupied the quota is unknown until the operational-denominator analysis runs. |
| 4 | Intended event: a medically recorded fall or fall-related fracture within 180 days (recorded-event proxy; diagnosis code 888 or fracture prefixes). Episode identity and PPV/missed-event ascertainment are unknown and are the subject of the MUST label audit (section 2.6). The clinical claim stays "recorded fall/fracture proxy" until that audit reports. |
| 5 | Nothing proves availability timing today beyond the DWH attestation (V3: no predictor record dated after the index day). Registries, MEFI computation, medication purchase status and hospital discharge remain attested or uncertain. The MUST availability-metadata request is the resolution. |
| 6 | Unknown: the V21 schema says "raw CCI group code of the population record, no Charlson computation". The DWH must supply the dictionary and order before Experiment 1 is registered; the fallback policy is in section 4.3. |
| 7 | Yes: a dimensionless lambda/lambda_max grid with lambda_max from each inner training fold. Remaining global decisions are label-free QA stops only (minimum known rows, constant, unreadable share, timing/provenance classes); coverage is additionally recomputed per outer-training fold. |
| 8 | Repair = screen removal + inner lambda recipe + per-fold coverage + CCI semantic encoding + negative controls, under the historical 70% objective. Improvement = the log-loss/Top3 shortlist policy, the XGB registered library, the additive basis, the longitudinal bundle. PRE/POST reports the combined repair effect without attribution. |
| 9 | Section 4.5 and section 6 fix the candidate library (ENET path on two l1 grids, ≤ 20 XGB configurations, one additive basis, one bundle), the tuning objective, the one-SE shortlist, the near-tie tolerance, a compute cap (one overnight per experiment on the work PC), the multiplicity policy (one primary contrast; everything else descriptive) and the amendment rule (written, dated, before the affected data are read). |
| 10 | The PI and management declare the contact programme, the eligible denominator and the MOD in writing before Experiment 5's extract is requested (D-4). Management is asked explicitly to accept that an imprecise positive result is INCONCLUSIVE and that the protocol then stops. |
| 11 | To be established by a written access inventory signed by the data custodian (who has seen any July-onward outcome, count or performance figure). The development team's position: no July-onward outcome has been accessed through this repository. The October-frozen model's July evaluation will be labelled a retrospective temporal test of unseen outcomes. |
| 12 | Reconstructed from availability metadata (creation/load/version times) requested under MUST; the label cutoff is December 28, 2026 plus the measured source-specific lag, with a completeness tolerance fixed before the extract; the August 1 research end is superseded by an explicit new observation cutoff. |
| 13 | Estimand audit first (which of death, disenrolment and administrative censoring produce NULL, 0 or 1). Allocation among all baseline-eligible patients precedes any label-based exclusion. If more than 5% of selected patients lack an ascertainable outcome, an IPCW sensitivity is pre-declared; otherwise bounds are reported and the primary claim is restricted to the evaluable selected patients. |
| 14 | Yes: three distinct deliverables (profiling table; faller-only phenotypes; later population phenotypes). A clinician at Meuhedet names phenotypes from baseline profiles without seeing fall-risk rankings; the risk by phenotype is revealed afterwards. |
| 15 | Unknown until the V22 discovery returns coverage and timestamps. Experiment 3 is conditional on that answer and is deferred, not substituted, if the histories do not exist. |
| 16 | Phenotype and model decisions use development data only (January 2026 fallers and cohort); cluster count, panel and representation are fixed before any risk is examined; "no stable phenotypes" and "no material gain" are pre-written report outcomes. |

---

## 4. Resolutions of the eleven mandated points

### 4.1 Must Phase 5.1 precede new modelling claims?

**Yes, for claims.** No performance claim about any new model is made until the corrected ENET comparator exists and the negative controls have run. Data extraction, feature specification, profiling design, phenotyping design and the registration of Experiments 2–5 proceed in parallel and do not wait.

### 4.2 The full-sample AUROC eligibility screen

**Removed entirely as a membership mechanism.** Replaced by a predeclared, label-free admissibility contract that already exists in the code and is now the only gate: sealed outcome/future/identifier columns and name patterns; V1/V21 roles and classes; timing classes (SAFE_VERIFIED / SAFE_BOUNDED / SAFE_ATTESTED admissible; UNCERTAIN_TIMING and INELIGIBLE_* not); provenance (DEFENSIBLE admissible; UNVALIDATED_CODES and EXPERIMENTAL_COMPOSITE not); the V3 attestation; label-free QA stops (minimum known rows, constant, unreadable share) at cohort level and recomputed per outer-training fold. The univariate AUROC survives only as a **training-fold forensic diagnostic** written to the report. Assurance comes from boundary traps (synthetic forbidden, future-loaded, renamed-lineage and weak-proxy fields injected through the real input boundary must be rejected before fitting; a legitimate strong synthetic predictor must be retained) and from ~10 reduced-budget full-pipeline label permutations with frozen folds.

### 4.3 Exact treatment of CCI_Group

1. Before registration, obtain the CCI_Group code dictionary and its order from the DWH (question 6).
2. Declare exactly one encoding, by what the dictionary supports:
   - **documented ordinal order** → linear path: thermometer (cumulative) dummies with a missing indicator; tree path: raw numeric code (thresholds use order, not spacing);
   - **documented nominal meanings only** → linear and tree paths: one-hot with reference coding, rare levels (< 10 training rows) and unseen test levels mapped to an explicit **other** level (not to the reference), missing to a missing indicator;
   - **undocumented** → CCI_Group is quarantined from the admissible universe; at most an exploratory arm, never compared against alternatives by outer outcome.
3. Implement as a Phase 5 feature override so the protected Phase 2/3 definition files stay byte-identical. The "other" level is a small design change applied to every nominal code (it also fixes the smoking/obesity sub-code semantics, which remain excluded from the admissible universe by provenance anyway).
4. No encoding is ever selected by its outer result.

### 4.4 Final primary endpoint

**Recall@Top3**: captured fallers among the exact 3% quota, k = round-half-up(0.03 · N), N = all baseline-eligible patients at scoring time, selection within outer folds in development (proportional largest-remainder allocation, deterministic row-key ties) and by direct ranking on the sealed temporal sample. Comparative statistic: paired Δ captured fallers per 10,000 eligible (equivalently ΔTP; ΔPPV = ΔTP / k; ΔFP = −ΔTP), with a 2,000-replicate paired patient bootstrap. The historical 70%-sensitivity outputs remain secondary audit tables.

### 4.5 Is 55% an aspiration or a success threshold?

**Aspiration only.** It is reported to management as "about 100 more captured patients on the development denominator". It is not a tuning target, not a success threshold, not a stop rule, and it never appears in a selection criterion. A model below 55% can deliver a confirmed material gain; a model above it can fail to beat the corrected comparator.

### 4.6 Predeclared success rule that does not reward repeated searching

- **One** registered primary contrast: frozen champion versus frozen corrected ENET comparator, Δ captured per 10,000 eligible at 3%, on **one** sealed temporal sample, opened **once**.
- **Categories** (paired 95% interval): lower limit > MOD → CONFIRMED_MATERIAL_GAIN_AT_3_PERCENT; lower limit > 0 and point ≥ MOD → PROMISING_MATERIAL_GAIN_UNCERTAIN; lower limit > 0 and point < MOD → SMALL_STATISTICAL_GAIN; interval includes 0 → INCONCLUSIVE; upper limit < 0 → WORSE_AT_CAPACITY.
- **MOD** declared in writing by management before the sealed extract is requested (placeholder 5 per 10,000 ≈ 49 patients ≈ 2.3 recall points).
- **Stop rule**: INCONCLUSIVE, SMALL or WORSE → the protocol ends and the corrected ENET is retained. No re-tuning, no second opening, no alternative champion is evaluated on this sample.
- **New hypotheses** (new data, new representation) require a new registered development plan and the **next** sealed sample (2027-01-01 at the earliest), never this one.
- **Forbidden searches**: random seeds, capacity jitter, class weights, threshold engineering, optional sensitivities, encoding alternatives, additional families, and any change after reading an outer or test result without a dated amendment that precedes the data read.
- **Candidate library and compute cap** fixed before Experiment 1's outer results are read (section 6).

### 4.7 Minimum number of model families worth testing

**Two learners and one representation variant**: penalised logistic regression (ENET, whose path includes the LASSO endpoint l1_ratio = 1), constrained XGBoost (fixed registered library ≤ 20 configurations), and ENET on a declared additive basis with ≤ 3 named interactions. Nothing else until a distinct information hypothesis demands it; survival modelling only if the censoring audit shows material or selective censoring.

### 4.8 Highest-value feature-engineering directions

In order: (1) time-bounded medication exposure and recent change (psychotropics, opioids, antihypertensives, hypoglycaemics; 30/90-day change, concurrent exposure); (2) prior-fall episode trajectory (last and penultimate episode dates, intervals, shortening intervals); (3) mobility/ADL/assistive-device transitions and decline between comparable assessments; (4) diagnosis-specific first/last dates and recent emergence; (5) acute-care transitions (recent ED visit, discharge, rising contacts). Then, as representation only: one fixed bundle of exact-window contrasts from existing counts and ≤ 3 declared interactions (prior falls × mobility impairment; recent sedating exposure × gait impairment; recent syncope × medication change — final list fixed by the clinical lead before registration). All five information directions need V22 histories with availability times; none is fabricated from existing summary counts.

### 4.9 V22 data genuinely worth requesting

- **MUST (4)**: availability/creation/load/version times for every primary predictor source including registry membership history and batch computation times, with view SQL hashes, version and extraction cutoff; label/source/follow-up audit with on-site adjudicated samples (counts and intervals returned); medication prescribing/dispensing/start/change/stop dates with ATC and supply where reliable; a custodially sealed later evaluation snapshot with an access log.
- **HIGH (6)**: repeated mobility/ADL/gait/device records and physiotherapy/rehab dates; historical fall/fracture records with encounter/episode IDs and availability times; dated diagnosis histories; ED/hospital encounters with admission/discharge/arrival times; repeated MEFI components and GUG/TUG values with instrument/version/computation time; coarse clinic/district retained locally.
- **NICE (3)**: hypertension/CKD/transplant restored with true V1 lineage; a small preregistered vitals/labs panel; 2025 backfill with creation times.
- **NOT REQUESTED**: more untimed registries, overlapping deficit sums, undocumented risk codes, post-index purchases, free-text/LLM extraction.

### 4.10 The role of clustering

| Question | Do it? | Role | Method | Output |
|---|---|---|---|---|
| 1. Faller profiling: what is common among fallers? | **Yes, first** | Clinical understanding; also a sanity check on the data | Supervised descriptive table on the development cohort: per feature prevalence among fallers and non-fallers, absolute difference, enrichment, coverage by group; risk difference and risk ratio from P(fall \| feature); age/sex-adjusted and fuller adjusted associations with intervals; a small declared co-occurrence list; FDR control; small cells suppressed | `FALLER_PROFILE.csv`, `FALLER_PROFILE_HE.md` |
| 2. Faller-only phenotyping: are there clinical types of recorded fallers? | **Yes** | Clinical understanding only; descriptive of recorded fallers, not causal types | Gower dissimilarity + PAM on the ≈ 2,155 development fallers over a prespecified domain-balanced baseline panel (prior-fall episodes, mobility/function, cognition, medication exposure as available, sensory/neurological conditions, frailty), 2–5 clusters, size floor max(100, 5%), 100 fixed-seed resamples of the whole pipeline, matched Jaccard ≥ 0.75 screening, one sensitivity (average-linkage hierarchical on the same matrix), clinician naming blind to risk | `FALLER_PHENOTYPES.md`, medoid profiles, stability report, or "no reproducible phenotypes" |
| 3. Full-population phenotyping: population groups with different fall risk? | **Deferred** (after 2) | Clinical and planning understanding | Label-blind CLARA-style medoids on a representative subsample with a frozen assignment rule; cluster count never chosen by risk separation; risk by phenotype estimated afterwards with intervals; reproduced on the later snapshot | Not in the first five experiments |
| 4. Predictive use of clusters | **Not now** | Representation only; expected gain small | Only under a future registered nested added-value protocol with the clustering fitted inside inner and outer training and heldout assignment by frozen rule; never a full-cohort cluster ID in X | — |

**Statement for management:** clustering in this project is for clinical understanding. A stable, interpretable phenotype analysis with zero prediction gain is a success of that track; the absence of stable phenotypes is also a legitimate and reportable result.

### 4.11 Temporal validation timing and what remains truly untouched

- **Snapshot**: Index_Date 2026-07-01 (outcome window July 2 – December 28, 2026; no overlap with the development window January 2 – June 30).
- **Truly untouched** means: no member of the development team has accessed any outcome, partial count, enrichment or performance figure dated after June 30, 2026; a written custodian access inventory establishes this before the extract is requested. If it cannot be established, July is used only as exploratory evaluation and the sealed confirmatory sample becomes the 2027-01-01 snapshot (labels mature 2027-06-30 plus lag), with a silent prospective snapshot after it.
- **Predictor state** is reconstructed as of the end of July 1 from availability metadata (creation/load/version times), not from event dates alone.
- **Label opening** no earlier than December 28, 2026 plus the measured source-specific documentation lag, with a completeness tolerance and a freeze rule fixed before the extract; mid-February 2027 is a scheduling assumption, not a validated date.
- **Description**: a retrospective temporal test of unseen outcomes by a model frozen in late 2026; same-HMO internal temporal validation; not external and not prospective.
- **Order**: freeze champion, comparator, fallback artefact, preprocessing, feature definitions, estimand, tie rule, MOD and verdict → score predictors and lock predictions → custodian verifies maturity → open labels once.

---

## 5. Reduced roadmap

### P0 — MUST DO

1. **Assurance registration**: freeze the repair recipe (section 4.2–4.3), the admissible universe, the label-free gates, the saved folds, the historical objective for the repair arm, the shortlist policy for the improvement arm, the candidate library, the compute cap, the amendment rule; then read T-01 as an inventory.
2. **Estimand and denominator**: outcome/death/censoring audit with the DWH; baseline-eligible operational denominator; exact 3% rule and ties; MOD and contact programme declared by management in writing.
3. **Phase 5.1 corrected run** (Experiment 1) with negative controls, PRE/POST and the corrected ENET comparator; package 0.13.0 / Phase 5 3.0.0.
4. **Sealed later sample**: custodian access inventory; availability metadata; label cutoff and completeness rule; one opening (Experiment 5).

### P1 — HIGHEST VALUE

5. **Representation comparison** (Experiment 2): corrected ENET vs constrained XGB (registered library ≤ 20) vs additive-basis ENET, same admissible inputs, same folds, shortlist policy.
6. **Longitudinal information bundle** (Experiment 3): one clinically reviewed bundle from verified timed histories (medication change, fall episodes, functional change), conditional on V22 delivery with availability times.
7. **MUST and HIGH data requests** issued now (section 4.9).

### P2 — EXPLORATORY

8. **Faller profiling** then **faller-only phenotyping** (Experiment 4); full-population phenotyping afterwards if phenotypes are stable.
9. Survival/competing-risk evaluation only if the censoring audit shows it is needed; site-level validation if clinic identifiers arrive; 2025 backfill only if cheap and as-of verified.

### DROP — DO NOT SPEND TIME

- Automatic AUROC vetoes in any form; triage margins as validity certificates.
- Seed, capacity, class-weight or threshold searches aimed at 55%; optional sensitivities accumulated until a gain appears.
- Standalone LASSO; CatBoost/LightGBM/RF/ExtraTrees tournaments; mixture-of-experts; tabular neural networks; LLM features; stacking; learning-to-rank.
- The domain × ablation × family grid; CCI encoding comparisons by outcome; pooled-vs-within-fold thresholding as experiments.
- Re-expressions of existing counts as separate experiments; untimed static registries; deficit sums; undocumented codes.
- Raw K-means, outcome-coloured embeddings, clustering-algorithm sweeps; cluster IDs in X in this round.
- Prior-faller expert models triggered by subgroup calibration.

---

## 6. The first five experiments, in order

Conventions: "runtime" is work-PC wall time, to be confirmed with `--estimate` before each run; the Phase 5 2.2.0 overnight run on the same machine is the reference. Every experiment writes to a new output folder, never into the 2.2.0 folder. All development experiments reuse the saved January 2026 folds (hash-verified), the parameter-free history calculations and the saved predictions.

### Experiment 1 — Corrected Phase 5.1 baseline (P0)

| Field | Content |
|---|---|
| Hypothesis | Removing the outcome-dependent screen, making the lambda recipe strictly inner, recomputing coverage per fold and encoding CCI by its documented semantics changes the development Recall@Top3 of ENET by a measurable but not necessarily material amount; the corrected pipeline returns to chance under label permutation and rejects injected forbidden fields. |
| Exact data required | The same January 2026 V21 extract (sha256 verified against the 2.2.0 plan); the saved folds and usable-patient keys of the 2.2.0 run (hash-verified); the CCI_Group dictionary from the DWH; the 2.2.0 `share\` tables read-only for PRE. |
| Model / analysis | ENET (path over l1 ratios including 1.0, dimensionless inner lambda grid) on three arms: OLD, ADMISSIBLE (OLD + NEW with SAFE timing and DEFENSIBLE provenance), legacy ALL (audit only, reproduced only if its membership differs from ADMISSIBLE); one source-risk contrast ADMISSIBLE minus NEW_REGISTRY. Repair arm tuned by the historical 70% objective; improvement arm selected by the log-loss one-SE shortlist → within-inner-fold Recall@Top3 (same inner path, extra outer refits only). Negative controls: boundary traps through the real input boundary; 10 reduced-budget full-pipeline permutations on frozen folds (ENET only); holdout-label mutation with frozen splits; retained strong legitimate synthetic predictor. Within-fold 3% capacity with paired bootstrap; per-fold and pooled calibration; discordance and overlap tables; precision planning simulation for the paired contrast. |
| Primary metric | Recall@Top3 (within-fold, labelled-subcohort denominator for PRE/POST comparability; operational denominator as a secondary descriptive if scoreable) with paired Δ captured per 10,000 between ADMISSIBLE and OLD. |
| Comparison baseline | PRE: the 2.2.0 predictions on identical folds (combined repair effect, no attribution). Within POST: ENET OLD. |
| Stopping rule | Any boundary trap not rejected, any permutation run with mean AUROC > 0.55 or Recall@Top3 > 5%, or a fold-hash mismatch → stop, investigate, amend, rerun. Otherwise the experiment completes once; its result is not re-tuned. |
| Expected runtime | Smaller than the 2.2.0 overnight run (no LASSO, no XGB, no domain grid): a few hours for the three arms plus the contrast; 1–3 hours for the ten reduced-budget permutations; minutes for traps and reporting. One night. |
| Decision that follows | The corrected ENET on ADMISSIBLE (or OLD if ADMISSIBLE is not better by the registered inner rule) becomes **the comparator** for all later claims and the **fallback artefact** for Experiment 5. Management is told the corrected Recall@Top3, with its interval, as the new reference, whether above or below 50.4%. The precision simulation tells management what MOD is detectable. |

### Experiment 2 — One representation comparison (P1)

| Field | Content |
|---|---|
| Hypothesis | Bounded nonlinearity (thresholds and interactions via constrained XGB; smooth shapes via an additive basis in ENET) adds a stable tail-ranking signal to the corrected ENET on the same admissible inputs. |
| Exact data required | Experiment 1's frame, folds, admissible universe and saved inner OOF predictions; a registered XGB library (≤ 20 configurations: depth 2–4, min_child_weight ≥ 10, learning rate ≤ 0.1, subsample/colsample ≥ 0.7, L2 ≥ 1, inner-only early stopping); the declared additive basis (≤ 5 named continuous variables: age, days since last fall, visit count, days in MEFI group, medication count or as finalised by the clinical lead; restricted cubic spline with knots at fixed training quantiles or the in-repo fractional-polynomial basis) and ≤ 3 interactions fixed before registration. |
| Model / analysis | Three arms on identical folds: corrected ENET (from Experiment 1), constrained XGB, additive-basis ENET. Selection inside outer training by the shortlist policy; procedure selection among the three also inside outer training (free, from inner OOF). Paired within-fold Recall@Top3 bootstrap of each challenger against the corrected ENET; all arms reported, the maximum never quoted as a validated estimate. |
| Primary metric | Paired Δ captured per 10,000 at 3% versus the corrected ENET, with 95% interval and fold-sign consistency (descriptive). |
| Comparison baseline | Corrected ENET on the same admissible universe. |
| Stopping rule | A challenger is carried forward only if its paired lower limit > 0 and its point estimate ≥ 0.5 × MOD in development and it is calibration-acceptable (slope and CITL within the investigation ranges); otherwise the representation track ends here with no further search. Near ties → the simpler model. |
| Expected runtime | XGB: ≤ 20 configurations × 25 inner fits × 3 arms-equivalent ≈ one fifth of the 2.2.0 XGB budget; additive ENET ≈ the ENET cost. One night. |
| Decision that follows | Fixes the backbone family for the champion (ENET, additive ENET or XGB) by the registered rule, before any outer result is used to pick a winner for claims. If no challenger passes, the champion backbone is the corrected ENET. |

### Experiment 3 — One new-information bundle (P1, conditional)

| Field | Content |
|---|---|
| Hypothesis | A compact, clinically reviewed bundle of verified timed histories (recent medication initiation/escalation and burden change; last/penultimate fall-episode dates and intervals; functional or device transitions) adds captured fallers at 3% beyond the admissible universe. |
| Exact data required | V22 deliveries with availability times for the January 2026 index: medication prescribing/dispensing dates with ATC; historical fall/fracture records with encounter/episode IDs; repeated mobility/ADL/device records; the availability-metadata contract. Coverage and timestamp audit returned first. **If these do not exist or lack availability times, the experiment is deferred, not replaced.** |
| Model / analysis | The Experiment 2 backbone with and without the bundle, the with/without choice made inside each inner loop (a two-element candidate set); the bundle's variables are declared, exact-window, parameter-free and computed as-of January 1; redundancy report against OLD summaries; paired within-fold Recall@Top3 bootstrap. |
| Primary metric | Paired Δ captured per 10,000 at 3%, bundle procedure versus the same backbone without the bundle. |
| Comparison baseline | The Experiment 2 backbone on the admissible universe (which is also the corrected ENET if no challenger passed). |
| Stopping rule | Bundle retained for the champion only if the paired lower limit > 0 and the point estimate ≥ 0.5 × MOD in development, with coverage ≥ 80% of the cohort for its inputs and no availability class worse than SAFE_ATTESTED. If the data arrive after the champion freeze date, the bundle waits for the next sealed sample. |
| Expected runtime | One night (backbone cost × 2 arms). |
| Decision that follows | Defines the frozen champion procedure (backbone ± bundle) to be refitted on all development data by the registered inner selection and frozen with the corrected ENET comparator and the fallback artefact. |

### Experiment 4 — Faller profiling plus one phenotype protocol (P2, parallel)

| Field | Content |
|---|---|
| Hypothesis | (A) Specific baseline features are enriched among recorded fallers beyond what age and sex explain. (B) Recorded fallers form 2–5 reproducible baseline clinical phenotypes; the alternative "no reproducible phenotypes" is an admissible outcome. |
| Exact data required | The Experiment 1 development frame (baseline predictors, labels, follow-up status) on the work PC; a prespecified domain-balanced panel of about 20–30 baseline variables with duplicates and summary scores removed and future severity/treatment fields excluded; Hebrew labels from `configs/meuhedet/feature_labels_he.yaml`. |
| Model / analysis | (A) Per feature: prevalence among fallers and non-fallers, difference, enrichment, coverage; risk difference and ratio with intervals; age/sex-adjusted and fuller adjusted logistic associations with shrinkage; ≤ 10 declared co-occurrence pairs/triples with support and lift; FDR control; small-cell suppression. (B) Gower + PAM on the development fallers for k = 2…5, 100 fixed-seed resamples of imputation/representation/clustering, matched Jaccard and co-assignment stability, size floor max(100, 5%), sensitivity to domain weights and one hierarchical sensitivity; medoid profiles; clinician naming blind to risk; risk by phenotype revealed afterwards with intervals. Gower and PAM implemented in numpy/scipy (no new dependency). |
| Primary metric | (A) Adjusted risk ratio and absolute difference per feature. (B) Mean matched Jaccard per cluster (≥ 0.75 screening) with its distribution, cluster sizes and clinical interpretability. |
| Comparison baseline | (A) Age/sex-adjusted null. (B) The one-cluster solution and a label-permuted stability reference. |
| Stopping rule | (B) If no k has all clusters ≥ 0.75 mean Jaccard and above the size floor, report "no reproducible faller phenotypes" and stop; no further algorithm or embedding search. |
| Expected runtime | (A) Minutes. (B) About 2,155 patients × 30 variables: the Gower matrix and PAM are seconds; 100 resamples × 4 values of k: under an hour. |
| Decision that follows | Management receives the profiling table and either named phenotypes or the "no stable phenotypes" result. If phenotypes are stable, full-population phenotyping is registered as the next P2 protocol. No cluster enters any predictive model in this round. |

### Experiment 5 — One sealed temporal confirmation (P0)

| Field | Content |
|---|---|
| Hypothesis | On a later, access-controlled sample, the frozen champion captures materially more fallers at 3% than the frozen corrected ENET comparator (Δ per 10,000 eligible with lower 95% limit > MOD). |
| Exact data required | The 2026-07-01 V21-equivalent snapshot with predictor state reconstructed as of the end of July 1 from availability metadata; the custodian access inventory; the outcome extract with a label cutoff ≥ December 28, 2026 plus the measured lag and a completeness certificate; the frozen champion, comparator and fallback artefacts with hashes; the written MOD and contact programme. If any of these fail, the sample is the 2027-01-01 snapshot. |
| Model / analysis | Phase 4 pattern: frozen hashes verified → predictors-only preflight → blind scoring of every baseline-eligible patient → predictions locked → custodian maturity check → outcome contract → one opening. Exact 3% quota among all eligible; paired bootstrap of Δ captured per 10,000; bounds for unascertainable selected patients, IPCW sensitivity if they exceed 5%; descriptive transport report (AUROC, AP, Brier, slope, CITL, decile calibration, drift) for each frozen model against its own development estimate; subgroup capture among the globally selected; same-patient and new-member strata. |
| Primary metric | Paired Δ captured fallers per 10,000 eligible at 3%, champion minus corrected ENET, 95% paired interval, classified by the five categories. |
| Comparison baseline | The frozen corrected ENET comparator scored on the same patients. |
| Stopping rule | One opening. CONFIRMED → proceed to silent prospective validation and intervention-pathway design. PROMISING / SMALL / INCONCLUSIVE / WORSE → the protocol ends, the corrected ENET is retained, and nothing is re-tuned or re-opened on this sample. |
| Expected runtime | Scoring and bootstrap: minutes to an hour; the calendar constraint is label maturity (earliest December 28, 2026 plus lag). |
| Decision that follows | Whether a model goes to silent prospective validation at all, and which one (champion, or comparator if the champion fails). Any further improvement hypothesis is registered against the next sealed sample. |

---

## 7. Governance for this roadmap

- **Registration before results**: a dated `docs/phase6/REGISTRATION_EXP1-5.md` fixes, before Experiment 1's outer results are read: the admissible universe, gates, saved-fold hashes, candidate library, objectives, shortlist tolerance, interaction list, basis variables, XGB library, bundle definition (or its deferral), MOD placeholder, success categories, compute cap and amendment rule.
- **Amendment log**: every change after registration is written, dated and justified before the affected data are read (`docs/phase6/AMENDMENTS.md`).
- **Access inventory**: signed by the data custodian before the Experiment 5 extract is requested.
- **Champion freeze**: before Experiment 5; champion, comparator and fallback hashed; Mode B equivalence by scores and configuration within a stated tolerance.
- **Claims discipline**: Experiments 1–3 are development estimates; Experiment 4 is descriptive; only Experiment 5 is confirmatory; "captured" means recorded events identified.
- **Protection and privacy**: unchanged from Phase 5 (new output folders; historical artefacts hashed; share only, aggregate only, small cells suppressed).
- **Versioning**: 0.13.0 / Phase 5 3.0.0 for the repair and development experiments; 0.14.0 / Phase 6 1.0.0 for the sealed confirmation package.

---

## 8. Decisions required from the PI and management

| # | Decision | Needed before |
|---|---|---|
| D-1 | Adopt this consensus roadmap and the five experiments in this order | implementation starts |
| D-2 | Confirm the provenance of the supplied 97,453 / 2,155 / 2,924 / 1,086 figures (table, denominator, hashes) | Experiment 1 registration |
| D-3 | Obtain the CCI_Group dictionary and the outcome/death/censoring label semantics from the DWH | Experiment 1 registration |
| D-4 | Management declares the contact programme, the eligible denominator and the MOD in writing, and accepts the stop rule | Experiment 5 extract request |
| D-5 | Issue the MUST and HIGH V22 requests now | this week |
| D-6 | Commission the custodian access inventory for July-onward outcomes | Experiment 5 extract request |
| D-7 | Fix the clinical panel for profiling/phenotyping, the ≤ 3 interactions and the ≤ 5 basis variables with the clinical lead | Experiments 2 and 4 registration |

---

## Appendix — Evidence pointers added in this document

| Claim | Where verified |
|---|---|
| Censored/unlabelled patients are dropped before folds; the 3% quota is allocated among labelled patients | `src/falls_ml/phase5/data.py:307, 317`; Phase 4 `evaluate.py:325–327` |
| Capacity uses exact half-up integer rounding, largest-remainder allocation and deterministic row-key ties | `src/falls_ml/phase5/capacity.py:6–11, 47–53, 76–83` |
| No faller-versus-non-faller profiling table exists in Phase 5 outputs | `src/falls_ml/phase5/report.py`, `analysis.py`, `explain.py` (search for prevalence/profile) |
| A fractional-polynomial basis exists in the repository | `src/falls_ml/features/fractional_polynomial.py` |
| Locked dependencies contain no clustering or GAM library (numpy, pandas, scipy, scikit-learn, statsmodels, xgboost, optuna) | `requirements.lock` |
| Inner ENET path is computed once per inner fold and objectives only select among its points | `src/falls_ml/phase5/engine.py:200–241` |
