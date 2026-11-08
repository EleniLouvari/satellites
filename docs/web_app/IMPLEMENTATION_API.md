# Implementation contracts

All API routes have `/api` prefix. JSON errors: `{detail: "Error: safe message"}`.
Use UUID strings for web IDs; integer IDs for existing app/EO users. Dates ISO UTC.
Lists return arrays. Runs/uploads include JSON-native metadata/config (not strings).

## Python module ownership

`src/parcel_app/`: new standalone package, add to setuptools include.
Foundation: `settings.py`, `db.py`, `schema.sql`, `security.py`, `auth.py`,
`accounts.py`, `app.py`, `manage.py`, `__init__.py`.
Primary: `uploads.py`, `runs.py`, `files.py`, `config.py`.
Worker: `worker.py`, `tasks.py`, `inspection.py`, `artifacts.py` (+ worker helpers).

`settings.get_settings()` returns cached Settings with `storage_root: Path`,
`credential_key: str`, `cookie_secure: bool`, `allowed_origin: str`,
`upload_limit_bytes: int` (250 MiB), `max_batch_workers: int` (4),
`storage_limit_bytes: int` (100 GiB), `session_hours: int` (12).
Environment: `PARCEL_STORAGE_ROOT`, `SATELLITES_CREDENTIAL_KEY`,
`PARCEL_COOKIE_SECURE` default true, `PARCEL_ALLOWED_ORIGIN` default
`https://localhost`, `PARCEL_UPLOAD_LIMIT_MB`, `PARCEL_MAX_BATCH_WORKERS`,
`PARCEL_STORAGE_LIMIT_GB`. PostgreSQL uses existing `POSTGRES_*`, `DB_NAME`.

`db.connection()` contextmanager yields psycopg2 connection with RealDictCursor
factory, commits on successful exit, rolls back on error, always closes.
`db.one(query, params=(), *, conn=None) -> dict|None`,
`db.all_rows(query, params=(), *, conn=None) -> list[dict]`,
`db.execute(query, params=(), *, conn=None) -> int` rowcount.
Always use bound params; use psycopg2.extras.Json for JSON writes.
`security.current_user(request) -> dict` FastAPI dependency with app_user_id and
username; checks session, active flag and CSRF on mutation using `X-CSRF-Token`.
`security.cipher() -> Fernet`. Login/register mutation Origin checked too.
Session cookie `parcel_session`; opaque token stored hashed. `/auth/me` response
`{app_user_id, username, csrf_token}`; login/register same response.
`app.create_app()` includes auth/accounts/uploads/runs/files routers. Imports of
primary routes may be deferred until written. `app.app = create_app()`.

## Database schema (public; migration explicit/idempotent)

Existing app_users: app_user_id bigint-compatible primary key, username text,
password text, is_active boolean. Existing eo_users: eo_user_id integer-compatible
primary key, app_user_id FK, username text, password text; add label text default
'', test_status text default 'untested', tested_at timestamptz nullable.
Ensure usernames uniqueness case-insensitive per owner; no plaintext login fallback.
Sessions: `app_sessions(token_hash text PK, app_user_id bigint FK, csrf_token text,
expires_at timestamptz, created_at timestamptz default now())`.
Auth rate limits persisted or rigorously bounded with explicit single API process.

`parcel_uploads(upload_id uuid PK, app_user_id bigint FK, filename text,
storage_path text, sha256 text, size_bytes bigint, status text, metadata jsonb
default '{}', error text nullable, created_at timestamptz default now())`.

`analysis_runs(run_id uuid PK, app_user_id bigint FK, name text, upload_id uuid FK,
layer text nullable, parcel_id_field text, partition_field text nullable,
processing jsonb, account_ids jsonb, revision int default 1,
validated_revision int nullable, status text default 'draft',
summary jsonb default '{}', config jsonb default '{}', output_dir text,
error text nullable, created_at timestamptz default now(),
updated_at timestamptz default now())`.
Statuses draft/validating/ready/queued/running/complete/failed/interrupted/cancelled.
`run_accounts(run_id uuid FK, eo_user_id bigint FK, position int, username text,
PRIMARY KEY(run_id, eo_user_id), UNIQUE(run_id,position))` frozen at submit.
`run_events(event_id bigint identity PK, run_id uuid FK, stage text,
level text default 'INFO', message text, created_at timestamptz default now())`.
`run_artifacts(artifact_id uuid PK, run_id uuid FK, kind text, name text,
storage_path text, size_bytes bigint, sha256 text, created_at timestamptz default now(),
UNIQUE(run_id,kind,name))` only safe published files, not raw checkpoint logs.

`web_jobs(job_id uuid PK, app_user_id bigint FK, kind text, target_id text,
payload jsonb default '{}', status text default 'pending', claim_token uuid nullable,
heartbeat_at timestamptz nullable, error text nullable,
created_at timestamptz default now(), updated_at timestamptz default now())`.
Kinds inspect/validate/run/resume/bundle/test_account. target_id UUID or EO integer as text.
Job status pending/running/done/failed. Jobs created in same transaction as state.
Partial unique index (kind,target_id) WHERE status IN ('pending','running').

No table/column name may be provided by the browser. Uses public existing tables;
custom legacy table names are not a web v1 feature.

## HTTP contract

- GET `/health`: `{status: "ok"}` (no secrets).
- POST `/auth/register`, `/auth/login`: `{username,password}` returns user + csrf.
  Registration password >=12 characters, max 256; username 3..80 ASCII letters,
  digits, underscores, dots, hyphens; no approval. POST `/auth/logout`: 204.
- GET `/auth/me`: user + csrf (401 if anonymous).
- GET `/accounts`: `[{eo_user_id,username,label,test_status,tested_at}]`.
- POST `/accounts`: `{username,password,label}` returns safe account.
- PATCH `/accounts/{id}`: `{label?,password?}`; username immutable in v1.
- DELETE `/accounts/{id}`: 204 if no run_accounts references.
- POST `/accounts/{id}/test`: `{job_id}` queue authentication only.
- GET `/config`: `{defaults: object, fields: [{name,type,default,...}], limits: {max_batch_workers,upload_limit_bytes}}`.
  Derive names/defaults from runner constructor; explicit choices metadata.
- POST `/uploads?filename=...`: **raw file body**, Content-Type application/octet-stream,
  streams size-bounded, returns owned upload row (202), queues inspection.
- GET `/uploads`: list own rows (latest 100).
- GET `/uploads/{id}`: safe row, metadata layers list. Each layer:
  `{name: string|null, columns: string[], parcel_count, crs, bounds, preview: GeoJSON|null,
  geometry_invalid_count, geometry_empty_count}`. Preview max 500 features / 1 MiB.
  Metadata GeoJSON does not contain original parcel attributes.
- POST `/runs`: `{name,upload_id,layer:null|string,parcel_id_field,partition_field:null|string,
  processing: object,account_ids:number[]}` -> owned run row.
- PATCH `/runs/{id}` same body, only draft/ready/failed-preflight; increment revision,
  clear validated_revision and summary, status=draft. No edits after submit.
- GET `/runs`: list own newest 100; GET `/runs/{id}` owned row.
- POST `/runs/{id}/validate`: queue preflight, status validating, return run (202).
  Job payload `{revision}`; normalized `config` and `summary` on successful ready.
- POST `/runs/{id}/submit`: `{revision}`; require ready validated_revision==revision;
  freeze selected accounts (ordered as request), enqueue run, status queued.
  Repeated request on already queued/running/complete same revision returns row,
  no duplicate job. Client cannot supply owner/path/image/env.
- POST `/runs/{id}/resume`: failed/interrupted submitted runs only; preserve config,
  account IDs/usernames/order; enqueue resume. Repeat while queued/running idempotent.
- POST `/runs/{id}/cancel`: cancel pending run only (not active), status cancelled;
  do not cancel resume with possibly existing remote jobs.
- POST `/runs/{id}/duplicate`: new draft copying input/account/settings.
- GET `/runs/{id}/events?after=0`: latest/next 200 safe events.
- GET `/runs/{id}/artifacts`: safe artifact rows (no storage_path).
- POST `/runs/{id}/bundle`: queue ZIP of registered artifacts; `{job_id}`.
- GET `/artifacts/{id}/download`: authorize owner then safe contained FileResponse.
  No arbitrary filename/path arguments accepted.

Run JSON excludes absolute `output_dir`, upload excludes storage_path. `config`
returned to browser strips absolute paths. Worker uses full config from DB.
Errors never include raw exceptions or query strings with credentials.

## Worker contract

Start `python -m parcel_app.worker` (long-lived poller); `--once` for one job/test.
`tasks.execute_task(job: dict) -> None` runs in a fresh spawned child. DB writes
use transactions, claim ownership is checked before final state writes. Parent
maintains heartbeat, singleton advisory lock plus storage-root process lock to
prevent an orphan child overlapping another worker after connection/process loss.
Metadata/preflight lane may use separate singleton key while run lane remains 1.
`worker.enqueue(kind,target_id,user_id,payload=None,conn=None)` helper returns job
dict; may instead primary inserts directly to avoid module coupling.

Preflight build config from run upload/server paths, call prepare_run, store
normalized full config/summary; only revision-matching results become ready.
Run/resume load exact frozen accounts, decrypt in child memory, call dictionary
runner with app_user_id. Preserve output paths and saved input bytes. Completion
requires runner manifest; artifacts registers contained files/checksums, cleaning
CSV, safe config/manifest exports and sanitized log copies. Redact secrets in
all log streams before persistence where possible; block non-sanitized logs from API.
Account test reuses fixed openEO endpoint's existing authentication helper.
No arbitrary endpoints or process commands supplied by browser.
Interrupted work is marked interrupted; recovery is explicit resume after parent
reconciliation. Do not blindly resubmit. Reserve account identities across failed/
interrupted runs until safely resumed/completed; cross-owner duplicate usernames
must be accounted for too. Raw application failures remain generic in UI.

## Verification

Integration tests use disposable PostgreSQL database/container, never live tables.
Dependency installs only in `.venv-checks` or frontend local node_modules.
Frontend must have `npm run build` and uses `/api` same-origin cookies/CSRF.
No fake success states or demo data in the live UI. Tests may stub acquisition.
