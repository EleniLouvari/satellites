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

1. [ ] Identity/database foundation: settings, safe DB helpers, additive explicit
   migration, sessions/CSRF, register/login/logout, EO account CRUD/test jobs,
   password-reset operator command. Tests: generic login failures, immediate
   registration, hashed passwords, session revocation, owner isolation, CSRF.
2. [ ] Upload/run APIs: streamed bounded file ingestion, job-based inspection,
   server-owned config, selected accounts, revision-based validation, submission,
   resume, cancel queued, duplicate, artifacts/logs/downloads. Tests: foreign IDs,
   traversal, oversized uploads, stale revisions, repeated submission, no final
   downloads for incomplete runs.
3. [ ] Worker: durable claim and heartbeat, exclusive processing, isolated child
   tasks, inspect/preflight/run/resume/account-test/bundle handlers, safe logs,
   exact account snapshot and artifact registration. Tests: claims, crash recovery,
   metadata inspection, sample-run completion, checkpoint resume, secret redaction.
4. [ ] Frontend: implement approved responsive design with real fetch calls and
   empty/error/loading states. Auth, accounts, upload/map, four-step wizard,
   saved drafts, list/detail, logs/config/results. Tests/build: TypeScript build,
   UI/API integration flows and available browser checks.
5. [ ] Integrate Compose/web Docker build and Windows launcher; setup guide and
   env template; explicit migrations and existing-password reset process.
6. [ ] Verify against isolated PostgreSQL, run synthetic end-to-end pipeline
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
