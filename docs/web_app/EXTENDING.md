# Extending Parcel Studio

Parcel Studio has two extension points:

- **Data sources** (`src/parcel_app/sources/<provider>/`): everything specific to one
  imagery provider, such as accounts, connection tests and remote job cleanup.
  Today: `openeo`. Planned: HUB.
- **Pipelines** (`src/parcel_app/pipelines/<pipeline>/`): one analysis the user can run.
  Today: `uploads`, `parcel_stats`, `eda`, `ml`.

Both are registered in one place, [`jobs/registry.py`](../../src/parcel_app/jobs/registry.py).
The registry drives the worker lanes, job dispatch, failure messages, recovery of
interrupted jobs and the mounted API routes. See the
[repository structure](../repository_structure.md#web-application-parcel_app) for the full layout.

## How a package is registered

Each source or pipeline package declares two constants in its `__init__.py`:

```python
from parcel_app.jobs.registry import JobSpec

JOBS = (
    JobSpec(
        kind="ml_train",                                   # stored in web_jobs.kind; must be unique
        lane="run",                                        # "run" = long analysis, "small" = quick metadata work
        handler="parcel_app.pipelines.ml.tasks:train",      # runs in the worker child process
        failure_message="Error: Training failed; ...",     # fixed, browser-safe text
        on_failure="parcel_app.pipelines.ml.tasks:mark_failed",  # f(job, message, conn): mark own record failed
        interrupt_sql=("UPDATE ml_runs SET status='interrupted' ... kind=ANY(%s)",),  # worker-crash recovery
    ),
)
ROUTERS = ("parcel_app.pipelines.ml.routes",)              # modules exposing `router`, mounted under /api
```

Then add the package name to `PACKAGES` in `jobs/registry.py`. Nothing else in the
worker, the dispatcher or `app.py` needs to change.

Rules that keep the worker safe:

- Write handler and hook references as strings. The API process must not import heavy
  dependencies (openEO, TensorFlow, ...); the worker child imports them when the job runs.
- Inside a handler, call `parcel_app.jobs.queue.owned_claim(job, conn)` in the same
  transaction before every database write that publishes results. This fences stale or
  stopped workers.
- Write files only beneath the directory allocated for the record, under
  `PARCEL_STORAGE_ROOT/users/<app_user_id>/<kind>/<id>/`, and check paths with
  `parcel_app.core.storage.contained`.
- Pass user-visible logs and errors through `parcel_app.core.storage.sanitize_text`.
- Set `stoppable=True` only if the handler runs in a stoppable process group and its
  record supports the stop/resume states.

## Checklist: adding a data provider (for example HUB)

1. **Library first.** Add or reuse the acquisition adapter in
   `src/data_preparation/sources/<provider>/`. `parcel_stats/core/` must stay independent
   of every provider.
2. **Decide the shared provider contract.** Adding the second provider is the time to
   decide this. Compare what `sources/openeo/` provides today: account routes
   (`accounts.py`), a connection test job (`tasks.py`), per-run credentials
   (`credentials.py`) and remote job cleanup on deletion (`remote_cleanup.py`, called from
   `pipelines/parcel_stats/deletion.py`). Also decide the database changes, for example a
   `provider` column on accounts and on `analysis_runs`, and where the provider is
   chosen in the wizard.
3. Create `src/parcel_app/sources/<provider>/` with `__init__.py` (`JOBS`, `ROUTERS`) and
   add it to `PACKAGES`.
4. Add a migration `src/parcel_app/migrations/00N_<provider>.sql`. Make it additive and
   idempotent (`IF NOT EXISTS`); it must run safely on every deployment.
5. Frontend: add `web/src/features/sources/<provider>/` and the matching calls and types
   in `web/src/api.ts`.
6. Tests: add `tests/test_parcel_app/` tests for the routes, the job handler with
   `PARCEL_TEST_DB=1` against a disposable database, and the failure/interrupt hooks.
7. Docs: update [the web app guide](README.md) and this file.

## Checklist: adding a pipeline (for example ML classification)

1. **Library first.** The pipeline must already run from Python, as
   `ml_classification.GeospatialClassificationPipeline` does from the notebook.
2. Decide the input: a completed parcel-statistics run, an EDA result, or a new upload.
   This decides the foreign key of the new table. EDA is the reference example:
   `eda_runs.run_id` points to `analysis_runs`, and the EDA job verifies the source
   checksum before running.
3. Create `src/parcel_app/pipelines/<pipeline>/` with:
   - `config.py`: validate and normalise the user's configuration against the library
     config class (see `pipelines/eda/config.py`).
   - `routes.py`: owner-scoped endpoints that create the record and call
     `jobs.queue.enqueue(...)` in the same transaction.
   - `tasks.py`: the handler, plus an `on_failure` hook.
   - `__init__.py`: `JOBS` and `ROUTERS`; then add the package to `PACKAGES`.
4. Add `src/parcel_app/migrations/00N_<pipeline>.sql` for the pipeline's table.
5. Deletion: if the record belongs to a parcel-statistics run, include it in
   `pipelines/parcel_stats/deletion.py` the way `eda_runs` is.
6. Resources: heavy pipelines run in the `run` lane. Check that the worker image in
   `Dockerfile.web` has the needed extras (for ML, the `ml` extra and optionally
   `neural`) and that the worker memory limit in `compose.web.yml` is enough.
7. Frontend: add `web/src/features/<pipeline>/`, reusing `components/ConfigForm.tsx` for
   schema-driven settings, as `features/eda/EdaPage.tsx` does.
8. Tests and docs, as for a data provider.
