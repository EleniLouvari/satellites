# Running Parcel Studio

Parcel Studio connects the approved React interface to FastAPI, PostgreSQL, and
the existing parcel-statistics pipeline. Registration grants immediate access.
Each user owns their uploads, EO accounts, saved configurations, run history,
events and downloads. The CLI and its batch launcher remain usable independently.

## Windows / Docker Desktop

Use Linux containers. The existing `scripts/.env_docker` holds database connection
settings and the **existing** Fernet key:

```dotenv
POSTGRES_HOST=host.docker.internal
POSTGRES_PORT=5435
DB_NAME=satellites
POSTGRES_USER=your_database_role
POSTGRES_PASSWORD=your_database_password
SATELLITES_CREDENTIAL_KEY=your_existing_key
```

Do not generate a replacement key for previously encrypted accounts. The web
application uses `public.app_users` and `public.eo_users`; legacy custom table
overrides are not supported by this web release.

From the repository root:

```powershell
.\scripts\run_web_app.bat build
.\scripts\run_web_app.bat migrate
.\scripts\run_web_app.bat start
```

Open **http://localhost:8080**. To inspect or stop services:

```powershell
.\scripts\run_web_app.bat logs
.\scripts\run_web_app.bat stop
```

`build` builds the existing pipeline image, API/worker image, and static frontend.
`migrate` explicitly runs the packaged additive migration in one transaction.
It preserves user/account rows, makes password columns TEXT, adds missing active
flags and generated-ID defaults, and creates web tables. It will fail instead of
silently discarding conflicting duplicate usernames. API startup never migrates.
An operator/database-owner role is needed for migration; use a restricted runtime
role for regular operation.

Docker's named volume `parcel-studio_parcel_storage` retains uploaded files,
checkpoints, logs and results through restarts. Back up this volume, PostgreSQL,
and the Fernet key together. Do not run `docker compose down -v` against data you
want to keep. The application does not delete old results automatically.

### Store application files in a Windows folder

Set `PARCEL_HOST_STORAGE` to a Windows folder to use a bind mount instead of the
named volume. Keep `PARCEL_STORAGE_ROOT=/data/parcel-studio`: that is the Linux
path inside the container and is also used by existing saved configurations.

```powershell
$env:PARCEL_HOST_STORAGE = "C:/work_sat/parcel-studio"
.\scripts\run_web_app.bat start
```

For persistence across PowerShell sessions, set `PARCEL_HOST_STORAGE` in the
repository-root Compose `.env` file. `scripts/.env_docker` supplies container
environment variables; it does not supply Compose substitutions for this mount.

This maps `C:/work_sat/parcel-studio` on Windows to `/data/parcel-studio` in the
API, worker, inspector and management containers. Uploaded parcels, statistics,
EDA outputs and logs use this folder. PostgreSQL data stays in its own database
storage.

For an existing installation, finish active jobs, stop the app, and copy the
contents of `parcel-studio_parcel_storage` into the Windows folder **before**
switching. Merely changing the mount does not migrate existing files; otherwise
saved run records would point to files absent from the new folder. Preserve the
original volume until the copied results are verified. No image rebuild is needed.

## Existing application passwords

New registration saves an Argon2id password hash. Existing plaintext application
passwords are deliberately not accepted at login. Set an existing user's password
interactively (the value never appears in command arguments):

```powershell
docker compose -f compose.web.yml run --rm manage -m parcel_app.manage set-password YOUR_USERNAME
```

This changes only that application's password hash and revokes its web sessions;
it does not change Copernicus credentials. Alternatively, register a new user and
add their own Copernicus accounts. Existing EO accounts remain linked to their
original app_user_id and are available when that user signs in.

## Using the application

1. Register or log in. Registration is immediate; there is no approval step.
2. Add one or more Copernicus accounts. Passwords are encrypted at write time;
   the browser cannot retrieve stored passwords. Connection testing is explicit
   and authenticates without creating a processing job.
3. Create an analysis and upload GeoPackage, GeoParquet or GeoJSON. Wait for
   inspection, choose the layer and unique parcel identifier, and optionally a
   partition column. The map is a simplified preview, limited to 500 polygons.
4. Select accounts and configure the run. Every exposed Python constructor
   option is listed in collapsible Basics, Sentinel imagery, Outlier removal,
   Gap filling, Interpolation, Tiles and execution, and Storage panels.
   IQR controls require outlier removal; temporal/spatial filling and interpolation
   require gap filling. IDW enables search distance; kriging enables variogram
   lags and distance; nearest requires neither. Disabled inputs keep their values
   while editing but are omitted from the saved effective processing request.
   New forms default to `temporal_fill_mode=bidirectional` and
   `temporal_reducer=none`. These web defaults do not overwrite existing saved
   selections or change the CLI constructor defaults. Reporting intervals remain
   editable with reducer `none` because they still define the output periods.
   Bands, indices and statistics use supported choices from the Python classes.
   Scheduling is available separately. Import/export processing JSON is supported.
5. Save a draft and validate. Validation runs separately from HTTP requests.
   Editing invalidates the previous validation. Submit the validated revision.
6. Return to My runs for progress, elapsed duration, sanitized stage events,
   configuration, results, cleaning reports and logs. Use **Download all files
   (.zip)** in the completed run header or Files panel. The first click queues
   archive preparation and starts the download when ready; later downloads reuse
   that archive. It includes every published file (results, reports, configuration,
   manifest and sanitized logs), excluding raw caches and private internal files.

Duration comes from the first recorded processing-start event to completion or
failure, and updates while running. Initial queue time is excluded; pauses between
resume attempts are included. Runs without a recorded start show a dash. Existing
run events provide this timing, so this change requires no new database columns.

To change an existing analysis name, open its details and select **Rename**,
enter a name (up to 120 characters), and select **Save name**. Running and completed
analyses can also be renamed. This changes only the display name; validation,
processing settings, results and recorded duration are preserved. Renaming is
unavailable while an analysis is being deleted.

Log redaction recognizes credential fields (including quoted passwords, tokens,
usernames and authorization headers) and URLs containing authentication data.
Ordinary values such as `B11`, `1.5` and `1M` remain readable even when they contain
the same text as a credential. Public backend URLs remain readable; private server
paths are masked separately. Previously saved redacted log text cannot be restored
by this change.

Server paths, credential keys and owner IDs are intentionally not configurable
from the form. The server generates paths and uses the authenticated user ID.
The submitted run keeps its normalized configuration and account order. Adding
accounts later does not change a run that may need to resume. A referenced account
cannot be deleted. Account passwords can be replaced without changing identity.

## Progress and permanent deletion

Run lists and run details show saved remote-job completion, local batch processing,
and partition assembly separately. Local batches count planned cubes with both
committed statistics and report Parquets, even when completed NetCDFs have been
removed. Partial files, cleaning reports alone and duplicate signatures do not
increase the count. Partition assembly can remain at zero while batches finish.
Remote jobs at 100% does not mean the
download, local processing or final publication has finished. A run is marked
complete only after verified publication. These percentages count work units,
not estimated remaining time; unavailable totals display the current stage.
EDA history shows five processing stages (profiling, charts, tables, report,
ZIP/publication), with 100% only after its artifacts are published. Stage lengths
can differ substantially; disabled optional stages complete immediately.

**Delete** is available in run lists/details and EDA history. Each action opens
a confirmation describing its permanent effects:

- Deleting a run removes its tracked remote openEO jobs/results, including saved
  retry attempts, then its local outputs, logs, caches, EDA analyses and database
  history. The [openEO deletion API](https://api.openeo.org/) stops computation and
  removes remote job data. Cleanup never lists and bulk-deletes an account's jobs.
- The uploaded parcel copy is removed only when no other run references it.
  Copernicus accounts and the user's original source file remain intact.
- Deleting one EDA analysis removes that analysis's report, plots, tables, ZIP,
  logs and configuration; its parent run and sibling analyses remain intact.

Active processing, queued work, EDA and ZIP creation block deletion. Finish the
work first, or cancel an initial queued parcel run. An analysis being deleted
cannot be resumed. Remote progress is checkpointed in PostgreSQL; if credentials,
connectivity or storage prevent cleanup, **Retry deletion** continues safely.
Local files are retained until remote cleanup succeeds. Partial local cleanup
may already have removed some files; the run remains non-resumable until deletion
finishes. Restoring access and retrying is required after such a failure.

The original run folder and its `tile_jobs.parquet` files are needed to identify
remote jobs and owners. Missing or known-incomplete tracking blocks deletion;
restore old-volume files to their original mounted location first. Cleanup uses
the currently configured storage root and the original recorded accounts.

Apply the additive migration before updating the API/worker: it adds
`analysis_runs.deletion_manifest` and `eda_runs.progress`. Existing scientific
results are not changed. No existing run is deleted automatically by deployment.

## Exploratory data analysis (EDA)

The **EDA** action appears beside Open in the run list and in each run's header.
It becomes available only when parcel processing has completed. The server also
checks completion and ownership before exposing columns or accepting an EDA job.

The dialog has Setup and History views. Setup reads columns from the registered
final statistics Parquet. Search columns, select all or numeric columns, and
choose exactly which inputs to analyse. Set optional target and ID roles from the
selected columns; numeric class codes should use `target_task=classification`.
Include the geometry column when geospatial summaries are desired.

All analytical/reporting `EDAConfig` options appear in grouped panels. Output
directory, artifact filenames, schema version and `reset_output_dir=False` are
listed under Managed output settings. Each submission receives a separate server
directory; it never clears parcel outputs or an earlier EDA directory. Saved
configurations and selected columns can be reused for another analysis.

EDA uses the heavy worker queue and continues when the dialog/browser closes.
History shows queued/running/complete/failed/interrupted status, timestamps,
duration, the saved request and downloads. Results include the HTML report (when
enabled), diagnostic tables, plots, annotated data, configuration and a log.
When an analysis completes while its EDA window is open, the report appears
automatically in the **Report** tab. **Open report** in History opens it directly
in a new browser tab, including for previously completed analyses. These actions
appear only when HTML reporting was enabled. Browser reports embed their plots
and run in a sandbox isolated from application sessions. Ownership checks apply.
The ZIP still includes the original report and plots for offline use: extract it
and open `report.html`. Failed analyses offer a diagnostic log when available;
new failures retain the exception traceback and show its type in History.

The additive migration creates `eda_runs`, linked to `analysis_runs` and
`app_users`, with input checksum, selected columns, configuration, status,
timestamps and published-file metadata. Apply the migration before starting an
updated worker:

```powershell
.\scripts\run_web_app.bat build
.\scripts\run_web_app.bat migrate
.\scripts\run_web_app.bat start
```

Interrupted/failed EDA analyses are preserved. Use their saved settings to submit
a fresh analysis. EDA proposals are exploratory diagnostics; final feature
selection requires train-only validation, as explained in the
[EDA guide](../eda/README.md).

## Stored history

`analysis_runs` stores each user's draft/submitted request, processing JSON,
scheduling mode, layer/fields, selected account IDs, normalized configuration,
validation revision, summary, status, error and timestamps. `run_accounts` freezes
the submitted identities/order. Related tables are `parcel_uploads`, `web_jobs`,
`run_events`, `run_artifacts`, and `app_sessions`.

The migration SQL is [src/parcel_app/schema.sql](../../src/parcel_app/schema.sql).
Existing CLI run directories are not automatically imported into web history.
All runs created through this application are saved in PostgreSQL.

## Processing and recovery

The run worker handles one heavy analysis at a time. The inspector service handles
uploads, validation and account checks independently, so new drafts remain usable
while imagery processing runs. PostgreSQL advisory locks and shared-filesystem
locks prevent duplicate lane workers and protect orphan child processes.

Closing the browser has no effect on running jobs. Interrupted/failed submitted
runs retain account reservations because their remote jobs may still exist. Resume
the original run to reconcile its checkpoints. Do not edit its uploaded bytes,
paths or saved configuration. Active remote cancellation is not implemented;
only never-started queued runs can be cancelled in this release.

The UI shows actual stage events rather than an invented percentage or ETA. Some
events can be dropped under excessive log volume; complete sanitized pipeline logs
are published after completion/failure. Native subprocess console output is not
used as a source of downloadable logs. A finished openEO job is not final result
publication: downloads require the runner's verified completion manifest.

## Server configuration

Compose defaults are local-only: `127.0.0.1:8080`, HTTP, and non-Secure cookies for
localhost. Before team deployment, put the app behind HTTPS, restrict it to the
intended private network, set `PARCEL_COOKIE_SECURE=true`, and set
`PARCEL_ALLOWED_ORIGIN=https://your-host`. Change port bindings deliberately;
immediate registration permits anyone who can reach the app to create an account.
The proxy preserves Origin, and the API requires the matching origin plus CSRF
token for authenticated mutations. Do not configure permissive CORS.

Set Compose substitutions in the shell or a separate Compose `--env-file`:

| Variable | Default |
| --- | --- |
| `PARCEL_ENV_FILE` | `./scripts/.env_docker` |
| `PARCEL_HOST_STORAGE` | `parcel_storage` (named volume; set a host folder for a bind mount) |
| `PARCEL_WEB_PORT` | `8080` |
| `PARCEL_ALLOWED_ORIGIN` | `http://localhost:8080` |
| `PARCEL_COOKIE_SECURE` | `false` for localhost |
| `PARCEL_UPLOAD_LIMIT_MB` | `250` |
| `PARCEL_MAX_BATCH_WORKERS` | `4` |
| `PARCEL_WORKER_CPUS` | `4` |
| `PARCEL_WORKER_MEMORY` | `16g` |
| `PARCEL_STORAGE_LIMIT_GB` | `100` |

The nginx upload ceiling is also 250 MiB; update `web/nginx.conf` if increasing
the API ceiling. CPU/memory controls are per service. Storage limits gate new
uploads/submissions; allocate and monitor sufficient filesystem capacity because
processing can grow during an active run. Failed jobs leave checkpoints intact.
Multiple independently launched CLI jobs are outside the application's account
reservation system and should not run concurrently with the same accounts.

## Development and verification

Install the `web`, `database`, `credentials`, `acquisition`, and `interpolation`
extras in a project virtual environment. Use the same PostgreSQL environment
variables and a private `PARCEL_STORAGE_ROOT`. Start the API with
`python -m uvicorn parcel_app.app:app --port 8000`, workers with
`python -m parcel_app.worker --lane run` and `--lane small`, and frontend with
`npm ci` / `npm run dev` under `web/`. Set the allowed origin to the Vite URL and
disable Secure cookies only for localhost development.

Tests in `tests/test_parcel_app/` default to skipping database integration unless
`PARCEL_TEST_DB=1` is set with **disposable PostgreSQL credentials**. Run migrations
there first. They cover registration, isolation, CSRF, uploads, validation,
submission idempotency, worker fencing, output redaction, synthetic real local
statistics, authorized downloads/bundles and checkpoint resume. They do not submit
live openEO work. Frontend: `npm test` and `npm run build`.

See [the design](DESIGN.md) and [implementation contracts](IMPLEMENTATION_API.md).
