# Phase 5 handoff implementation plan

> **For agentic workers:** Use systematic-debugging and test-driven-development for the narrow lifecycle fix; audits run independently.

**Goal:** Fix Windows SQLite ownership, prove the existing dashboard is safe, and audit exact Phase 5 preprocessing without retraining.

**Architecture:** One explicit Optuna RDBStorage is owned by each run_study invocation. All studies in that invocation share it; finally removes the scoped session then disposes its engine. TrialStore owns paths and the ledger, never connections. Phase 5 model code stays unchanged.

**Tech Stack:** Python 3.11, Optuna 5, SQLAlchemy, SQLite, pytest; existing mail-safe packaging.

**Spec:** User takeover brief for baseline d1e611c, falls_ml 0.12.2 / Phase 5 2.2.0.

## Global constraints

- Synthetic development and testing only; no patient data access.
- No retraining or modification of completed predictions and fitted models.
- No Windows skip, blind retries, ignored PermissionError, or weakened resume guarantees.
- Preserve within-fold proportional capacity selection and completed output compatibility.

## Review focus

- Storage lifetime on successful return and objective exceptions, with retained references and GC disabled.
- Immediate archive in both rollback-journal and WAL modes.
- Four successive restarts preserve suggestions, objective values, and item seeds.
- Dashboard fitting/storage blockers and unchanged input byte/mtime snapshots.
- Audit findings distinguish exact source evidence from uninspected Windows runtime artifacts.

## Task 1: Provenance and lifecycle fix

- [x] Verify clean baseline branch/HEAD/version and supplied ZIP SHA; compare all 162 restored source files byte for byte.
- [x] Add real SQLite lifecycle regressions in tests/unit/test_xgb_storage_lifecycle.py; observe six failures before changes on Python 3.11 and 3.13.
- [x] Own one explicit RDBStorage in src/falls_ml/phase2/xgb_tuning.py per run_study; close session then engine in finally.
- [x] Verify original resume test plus new journal/WAL tests, and full fast regression suite.

## Task 2: Independent audits

- [x] Trace dashboard CLI, block fitting/tuning/storage creation dynamically and snapshot completed synthetic artifacts.
- [x] Trace Phase 5 preprocessing; write 130-candidate source table and classify every prior concern with evidence.
- [x] Review findings; preserve scientific code and report retraining implications honestly.

## Task 3: Patch delivery

- [x] Bump falls_ml to 0.12.3; Phase 5 remains 2.2.0.
- [x] Build mail-safe package, restore and validate manifest with Python 3.11; repeat targeted tests from restored source.
- [x] Review diff and prepare release metadata and exact Windows setup/test/dashboard commands; final commit and ZIP SHA are recorded separately in dist/DELIVERY_0.12.3.md.

The restored-package fast suite passes; an optional full slow-suite attempt exposed a legacy EDA recalibration failure reproduced on the original release with the identical synthetic fixture. No full-suite pass is claimed. Targeted slow integration checks are tracked separately.

The rebuilt ZIP restored all 389 manifest payloads and passed 143 targeted/packaging checks. The existing Phase 5 dashboard integration passed against a complete synthetic trained run. These establish the requested storage and dashboard guarantees; supplemental full Phase 2/3 crash-integration outcomes are reported separately.
