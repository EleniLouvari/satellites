# Run and EDA deletion implementation plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Let an owner permanently delete an idle run and its related local/remote data, or a single EDA analysis.

**Architecture:** Owner/CSRF-protected DELETE routes serialize deletion per parent run, reject active jobs, and mark targets deleting before side effects. Persist a remote-job manifest before removing anything. Remove tracked remote jobs first, then exact owned storage directories, then dependent database rows. Partial failures remain retryable.

**Tech Stack:** FastAPI, PostgreSQL, openEO Python client, React.

**Spec:** The user's deletion request and explicit choice to also delete remote openEO jobs.

## Constraints and review focus

- Test only disposable data; mock all remote deletions. Do not delete live user runs.
- Delete current and failed/retried job IDs only with their recorded owner. Never bulk-delete an account's jobs.
- Preserve shared uploads until the last referencing run is removed; preserve EO accounts.
- Reject active run/EDA/ZIP/inspection work and unsafe/symlinked storage roots.
- Missing historical run storage must not silently discard remote-job tracking.
- A durable deleting state blocks resubmission; failed cleanup supports retry.
- Serialise deletion of a run and its EDAs; preserve account reservations until cleanup completes.

## Tasks

- [x] Add focused tests for ownership/CSRF, cascades, shared uploads, active jobs, paths, retry, and remote job scoping.
- [x] Implement remote manifest collection and idempotent cleanup, durable deletion routes and additive migration.
- [x] Add confirmation controls to run lists/details and EDA history, with deletion/retry state.
- [x] Run tests/build, review the changes, document behavior, migrate and deploy without interrupting active processing.

## Verification ledger

Real openEO DELETE calls and real user-data deletion are excluded from verification.
User added percentage progress on 2026-11-09: implemented measured remote/local
counts, EDA stage completion and checkpointed remote cleanup counts. No ETA is invented.

Ruling: continue in the existing shared working tree to preserve the user's
ongoing app and notebook changes; no unrelated changes are committed or reset.

Review found a shared-upload reference race and missing remote tracking guard.
Reproduced both with failing tests, then corrected final upload locking and
validated batch/partition tracking coverage. Initial EDA history fetch now works
independently of source/options availability.

Final verification on 2026-11-09: 81 backend/EDA tests passed against a disposable
PostgreSQL database with remote deletion mocked. Seven frontend helper tests,
TypeScript/Vite build, Ruff checks and whitespace checks passed. Follow-up review
found no remaining important issues in the deletion fixes.

Built both Docker images, applied the additive migration, confirmed the queue was
idle, and started API, run worker, inspector and web services. The API is healthy;
the web page and health endpoint return HTTP 200, and nginx configuration checks
pass. Read-only deployment checks confirmed both DELETE routes and progress
serialization for all four existing runs without exposing deletion manifests.
No live runs or remote jobs were deleted. The disposable test database was stopped.
Browser interaction verification was unavailable because the browser access host
was unavailable; API behavior, production builds and HTTP deployment were checked.
