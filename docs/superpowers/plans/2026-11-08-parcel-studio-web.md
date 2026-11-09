# Parcel Studio Implementation Plan

> Use subagent-driven-development for the independent frontend, identity, and
> worker tasks; the primary agent integrates upload/run APIs, packaging and tests.

**Goal:** Deliver the approved web application connected to the existing runner.
**Architecture:** React/TypeScript frontend, FastAPI session API, PostgreSQL durable
jobs, persistent storage and a separate worker process. Existing CLI stays usable.
**Spec:** `docs/web_app/DESIGN.md` (approved by user).
**Contracts:** `docs/web_app/IMPLEMENTATION_API.md`.

## Constraints and decisions

- Immediate registration; Argon2id app passwords; Fernet EO credentials.
- All data is owner-scoped. Browser paths and owner IDs are never trusted.
- Single heavy run, durable queue, immutable submitted config/account identities.
- Keep the existing user's database untouched during development; test in an
  isolated PostgreSQL container. Apply migrations explicitly at deployment.
- Work in this checkout because the approved CLI/Docker prerequisites are
  uncommitted here. Preserve unrelated notebooks and user edits; do not commit
  their changes as part of this task.
- Use the existing image as the worker base; no Docker socket access.

## Tasks

1. [x] Identity/database foundation: settings, safe DB helpers, additive explicit
   migration, sessions/CSRF, register/login/logout, EO account CRUD/test jobs,
   password-reset operator command. Tests: generic login failures, immediate
   registration, hashed passwords, session revocation, owner isolation, CSRF.
2. [x] Upload/run APIs: streamed bounded file ingestion, job-based inspection,
   server-owned config, selected accounts, revision-based validation, submission,
   resume, cancel queued, duplicate, artifacts/logs/downloads. Tests: foreign IDs,
   traversal, oversized uploads, stale revisions, repeated submission, no final
   downloads for incomplete runs.
3. [x] Worker: durable claim and heartbeat, exclusive processing, isolated child
   tasks, inspect/preflight/run/resume/account-test/bundle handlers, safe logs,
   exact account snapshot and artifact registration. Tests: claims, crash recovery,
   metadata inspection, sample-run completion, checkpoint resume, secret redaction.
4. [x] Frontend: implement approved responsive design with real fetch calls and
   empty/error/loading states. Auth, accounts, upload/map, four-step wizard,
   saved drafts, list/detail, logs/config/results. Tests/build: TypeScript build,
   UI/API integration flows and available browser checks.
5. [x] Integrate Compose/web Docker build and Windows launcher; setup guide and
   env template; explicit migrations and existing-password reset process.
6. [x] Verify against isolated PostgreSQL, run synthetic end-to-end pipeline
   with remote downloads stubbed at the acquisition boundary, compile/lint/test,
   review implementation, fix material issues. Do not claim live EO verification.

## Review focus

- Foreign-owned IDs must never leak accounts, geometries, logs or files (tasks 1/2).
- New EO accounts must not change old run account order (tasks 2/3).
- Worker death must not create duplicate active jobs (task 3).
- Credentials must not appear in errors, files or downloads (tasks 1/2/3).
- Authenticated form retries must not create duplicate submitted runs (task 2).

## Execution ledger

- Plan recorded; user explicitly authorized implementation of the reviewed design.

- Implemented identity, encrypted EO account management, file inspection, complete
  processing form, durable queue, immutable submissions, history, safe downloads,
  bundles and checkpoint resume. Added scheduling selection and per-user
  `analysis_runs` JSON configuration persistence as requested during execution.
- Verification: 25 backend tests passed against disposable PostgreSQL, including
  a real synthetic local statistics run/resume with remote acquisition stubbed;
  two frontend import tests passed; TypeScript/Vite production build and Ruff
  passed. Host test warnings: Starlette httpx test-client deprecation and a native
  NumPy ABI warning. Docker dependency check passed; npm audit reported zero issues.
- Built API and frontend Docker images. Applied the additive migration explicitly
  to the configured database and verified the existing one app user and eight EO
  accounts remain. No live Copernicus jobs were submitted by these checks.
- Started the local Compose stack using the Windows launcher. Frontend HTTP 200,
  API health `ok`, API healthy and both worker lanes running. Registration, upload
  inspection and configuration validation also passed over HTTP in the isolated
  test deployment before stopping that deployment.
- Browser visual verification was attempted but Computer Use stopped because it
  could not determine the current browser URL safely. Visual rendering remains
  unverified; API/build checks are not a substitute for that check.
- Existing plaintext app password requires the documented interactive reset to
  Argon2id. Local port binding is loopback only; HTTPS/team hosting is an operator
  deployment step. No active remote cancellation or automatic retention included.

## Configuration and run-screen follow-up

- Added seven collapsible configuration panels and dependency metadata shared
  by the UI and draft validation. IQR follows `remove_outliers`; filling follows
  `fill_nulls`; IDW and kriging expose their own distance/variogram inputs.
  New web form defaults: bidirectional temporal filling and no temporal reducer.
- Changed the analysis list action to Open. Derived elapsed duration from saved
  processing-start events, with terminal run timestamps; no schema migration.
- Added a prominent Download all files (.zip) action with preparation status,
  automatic download on readiness, failure/retry handling and reuse of ready ZIPs.
- Follow-up verification: six focused backend tests, TypeScript checking, Ruff
  and whitespace checking passed. Read-only checks against the disposable test DB
  confirmed historic run duration, the locking query and ZIP readiness.
- Full integration execution was blocked by sandbox temporary-directory access.
  Elevated frontend test/build approval review failed because its usage limit was
  reached, so frontend runtime tests and Docker rebuild/redeployment of this
  follow-up remain outstanding. Earlier deployment/test results above refer to
  the original app, not this follow-up. Update with build/start when available.

- Follow-up deployment resolved during EDA integration: configuration grouping,
  duration and ZIP changes are now built and deployed alongside EDA. Current
  verification: 34 backend tests, 6 frontend tests and production builds passed.
