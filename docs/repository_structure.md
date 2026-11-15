# Repository structure and migration

The installable distribution contains five top-level packages directly under
`src/`: `data_preparation`, `eda`, `ml_classification`, `shared`, and `parcel_app`. Notebooks,
tests, documentation, and generated results live outside these packages.

## Source boundaries

```text
src/
  parcel_app/                    Parcel Studio web API and worker (see below)
  data_preparation/
    cli.py                       satellites parcel-stats validate/run/resume
    sources/
      openeo/cubes.py             authentication, cube graphs, remote jobs
      hub/                       catalog queries, asset downloads and audits
    parcel_stats/
      run_config.py              offline dictionary/JSON preflight and partitioning
      runner.py                  run locking, state, resume and verified publication
      accounts.py                application-user-scoped PostgreSQL credentials
      core/base.py               shared construction, local execution, worker state
      core/configuration.py      validators and unchanged cache signatures
      core/parcel_batches.py     parcel geometry preparation and batching
      core/raster_cleaning.py    shared local raster cleaning
      core/parcel_statistics/
        __init__.py              stable public parcel statistics API re-export
        calculator.py            calculator class assembled from mixins
        checkpoints.py           checkpoint signatures and persistence helpers
        statistics.py            parcel masking and reducer operations
        streaming.py             streamed checkpoint execution/orchestration
        reshape.py               derived geometry features and ML reshaping
      core/crs.py                projected CRS selection
      openeo.py                  openEO acquisition plus local statistics
      job_manager.py             restartable openEO tile workflow
      multiuser.py               multi-account orchestration
    features/
      temporal.py                annual reduction of dated satellite features
      elevation.py               DEM features
      spatial_context.py         soil and nearest-lake features
  eda/
    pipeline.py
    core/ reporting/ visuals/
  ml_classification/
    pipeline.py
    step_01_check/check.py        each numbered step owns a libraries/ folder
    step_02_prepare/prepare.py
    step_03_train/train.py
    step_04_evaluate/evaluate.py
    step_05_predict/predict.py
    shared/                      classification helpers used across steps
      config/                    base.py, config.py and tune_params.py
      models/                    models.py, modeling_context.py and TensorFlow implementations
      reports/                   common report rendering, viewer, diagnostics and dashboard
    sensitivity/runner.py        sensitivity helpers live in libraries/
  shared/
    io.py constants.py assertions.py
    geometry.py                  geometry validation and distance units
    postgress.py                 database connections and table reads
    spatial_statistics/          buffer-based parcel clustering
    data_cleaning/               outlier analysis, interpolation, and variograms
    notebook.py                  explicit notebook setup
```

`OpenEOZonalStats` uses `ParcelStatsBase` for local processing. The base retains
sensor configuration names for constructor/cache compatibility, but imports no
acquisition service. Public parcel-statistics exports are loaded lazily. HUB
catalog utilities remain available; the former Planet delivery and statistics
adapters are not included in this checkout.

EDA and classification consume tables and shared helpers. They do not import
acquisition code or each other. Shared cleaning stays below the pipelines;
temporal reduction and environmental enrichment belong to data preparation.

## Web application (`parcel_app`)

`parcel_app` is the backend of Parcel Studio. It calls the library packages; the
library packages never import it. Code is grouped by responsibility:

```text
src/parcel_app/
  app.py                 FastAPI factory; mounts the routers listed in the registry
  manage.py              migrate / set-password operator commands
  settings.py            environment-based settings
  core/                  db.py, security.py, storage.py (path containment, checksums, log redaction)
  auth/routes.py         registration, login, sessions
  jobs/                  generic background-job engine
    registry.py          the one list of data sources and pipelines, their job kinds and routers
    queue.py             enqueue, claim, claim fencing, reconciliation, stop requests
    dispatch.py          runs one claimed job and records failures
    worker.py            worker service (`python -m parcel_app.jobs.worker --lane run|small`)
    process_control.py   process groups / Windows job objects for stoppable tasks
  sources/               data providers, one package each
    openeo/              Copernicus accounts, connection test, run credentials, remote job cleanup
  pipelines/             analyses, one package each (routes.py, tasks.py, ...)
    uploads/             parcel file upload and inspection
    parcel_stats/        configuration, validation, runs, results, bundles, deletion
    eda/                 EDA of a completed parcel-statistics result
    ml/                  ML classification scenarios of a completed EDA analysis, run step by step
  migrations/            001_core.sql, 002_eda.sql, 003_ml.sql; applied in name order by `manage migrate`
```

Each `sources/*` and `pipelines/*` package declares its background job kinds
(`JOBS`) and API routers (`ROUTERS`) in its `__init__.py`; `jobs/registry.py`
lists the enabled packages. The worker lanes, job dispatch, failure handling,
interrupted-job recovery and the API routes are all derived from that list.
To add a data provider (HUB) or a pipeline (ML), follow
[Extending Parcel Studio](web_app/EXTENDING.md).

The React frontend in `web/src/` mirrors this split: `api.ts` (API client and
types), `components/` (shared widgets and the schema-driven `ConfigForm`), and
`features/` (`auth/`, `sources/openeo/`, `parcel-stats/`, `eda/`, `ml/`).

### Shared spatial statistics

`shared.spatial_statistics` is a package. Buffer clustering remains available as
`from shared.spatial_statistics import spatial_clustering_using_buffer`.
The retained implementation is `clustering.py:spatial_clustering_using_buffer`,
used by the parcel acquisition notebook. Its geometry and distance-unit helpers
remain in `shared.geometry`.

### Shared utility cleanup

The shared-library usage audit follows calls and function references from
repository source code and notebook code cells, including callbacks and the
internal helpers they require. Tests, package exports, documentation examples,
and saved notebook outputs do not by themselves establish runtime usage.

Unused functions and their dedicated tests have been removed. Entirely unused
modules (`shared.raster`, `shared.logging`, `shared.formatting`, `shared.tabular`,
the other spatial-statistics modules, and `data_cleaning.tabular_imputation`)
have also been removed. Database utilities retain connection management and
table reads; outlier utilities provide IQR and z-score summaries plus combined
IQR, z-score, Isolation Forest, PCA/DBSCAN, and Mahalanobis row diagnostics.
See the [EDA outlier contract](eda/README.md#combined-parcel-outliers). These removals narrow the Python API; callers
outside this repository must update any imports of removed functions.

## Installation and imports

Install or reinstall the repository into the environment used by your Jupyter
kernel after changing the source layout:

```powershell
python -m pip install --no-deps --no-build-isolation -e .
```

Restart the Python process or notebook kernel afterward. Import directly from
`data_preparation`, `eda`, `ml_classification`, and `shared`; there is no enclosing
Python namespace or compatibility package. Use the public imports in the
[root README](../README.md#public-apis).
The distribution name in package metadata remains unchanged.

For source-tree development, `PYTHONPATH` needs only `src`. Pytest and VS Code
already use this directory. Package discovery includes the five top-level
packages and their subpackages; coverage measures the same five packages.

Constructor arguments, output filenames, log namespaces and cache signatures
remain unchanged by the directory move. Python pickle/joblib artifacts store
module paths, so previously serialized custom classes may need regeneration
under the new imports. Raster caches and tabular output files do not depend on
Python class import paths.

## Notebooks, reference data and artifacts

- Neuro notebooks remain together in `notebooks/projects/neuro/`, in workflow order.
- Experimental notebooks are in `notebooks/experiments/`; maintenance notebooks
  are in `notebooks/maintenance/`.
- Imports use canonical paths. Notebook code no longer patches `sys.path` or
  imports the all-libraries module. Display and seed setup is explicit.
- Existing external data/output paths and saved notebook outputs are retained.
  Historical outputs may still display the paths used when they were executed.
- The ML output template is documented under
  `docs/ml_classification/output_template/`; it is not a source-code package.
- New runs should use `outputs/<project>/<run>/` and dedicated pipeline
  subfolders. Configure external data roots in notebook configuration cells.
- Generated results and scratch files are ignored by Git.

The original Word manuals are preserved unchanged in `docs/manuals/`. Markdown
guides provide the updated paths. No remote data was downloaded, no production
pipeline was executed, and no external run directory was moved as part of this migration.

## Development and verification

Tests live under `tests/`, and VS Code is configured for pytest. Development
scripts use the active Python environment, resolve the repository relative to
their own location, and place reports in `outputs/quality/`. The code-check script
reports failures through its exit code and runs Ruff with `--fix`, which can
modify source. The separate quality-check script writes Radon metrics.

Layout tests in `tests/test_repository_layout.py` cover top-level package
locations, fitted-model/configuration serialization with current module paths,
Windows-style spawned-worker serialization, and import-side-effect isolation.
Parcel-statistics tests cover local cleaning, checkpoints, and resume behavior.
