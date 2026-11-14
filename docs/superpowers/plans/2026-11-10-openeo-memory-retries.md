# openEO memory retry implementation plan

**Goal:** Recover confirmed remote JVM memory failures with bounded per-batch resource escalation while preserving successful work.

**Design:** Keep backend resource options separate from scientific/cache identity. Persist effective JSON job options, prior attempt options and failure classification in each tile job row. Read the exact failed job's saved logs. Confirmed JVM OOM doubles executor memory from the configured starting value (2G default), capped at 8G by default. Unknown/transient failures retain bounded existing retries; explicit invalid graph/input codes are not retried. At the memory cap, stop and recommend smaller batches. Do not infer OOM from shuffle errors alone.

**Recovery:** Ordinary Resume retains retry limits. An explicit UI checkbox / API boolean / CLI flag permits exhausted JVM OOM batches to try strictly higher memory tiers up to the saved policy cap. Counters and history never reset. Pass this as a runtime option after verifying the original configuration/input hashes; do not edit saved configuration. Existing completed/running jobs retain their identities and owners.

**Scope:** Tile job manager (both schedulers), JSON configuration and UI resource options, resume propagation, tests and documentation. No live stop, deployment or remote submission during implementation.

- [x] Test and implement validated options and failure classification in `sources/openeo/job_options.py`.
- [x] Test and implement persistent retry policy in `parcel_stats/job_retry.py`; integrate the manager and both scheduler paths.
- [x] Add explicit exhausted-memory recovery to runner, CLI, owned resume API and UI; retain hash/account checks.
- [x] Add JSON object configuration UI and bounded memory settings; document policy and upgrade/resume steps.
- [x] Run parcel/web regressions, frontend checks, quality checks and independent review; inspect live state read-only.

**Review focus:** absent/corrupt logs; legacy rows; crash between retry preparation and submission; simultaneous batches sharing one Parquet; exhausted retries across repeated Resume calls; invalid/backend-specific resource values; no secret or scientific changes through resume options.

## Verification and review

Final parcel/web regression suite: 452 passed against a disposable PostgreSQL
instance, with remote calls mocked. Frontend: 11 tests passed and production
build passed. Changed-file Ruff/docstrings and new-module Bandit passed; the
repository quality notebook generated updated Radon reports. No new retry
function exceeds the notebook's complexity threshold of 15. The repository-wide
auto-fixing check script was not run against unrelated changes.

Independent review identified prepared replacements losing account pinning after
a restart. `test_prepared_replacement_stays_pinned_to_original_account` reproduced
this, then passed after pinning by persisted owner as well as remote ID. The final
suite includes the fix. No remaining material review findings.

Built `satellites-parcel-web:local` (shared by API/worker/inspector) and
`satellites-parcel-ui:local` without replacing live containers. The packaged image
classified the actual failed batch as JVM OOM using a read-only mount and disabled
network: ordinary retry remains exhausted after three retries; explicit recovery
selects 4G. No remote job was submitted and no run file was changed.

Activation remains an operator step: stop the existing worker immediately before
starting the rebuilt services, then Resume the original analysis with higher-memory
recovery selected. Existing remote jobs may continue and are reconciled on resume.
