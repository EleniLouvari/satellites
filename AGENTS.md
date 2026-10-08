# Repository guidance for Codex

This file applies to the entire repository. Reuse the existing guidance below as
the source of truth rather than duplicating its contents here. All paths are
relative to the repository root.

## Required project guidance

Before starting repository work, read and follow:

- [.github/copilot-instructions.md](.github/copilot-instructions.md)
- [.copilot/behavioral-guidelines/SKILL.md](.copilot/behavioral-guidelines/SKILL.md)

## Instructions for specific files

Before editing or reviewing files, inspect the `applyTo` frontmatter in
`.github/instructions/*.instructions.md`. Read and follow every instruction file
whose patterns match the files involved in the task. Treat comma-separated
patterns as alternatives and paths as relative to the repository root.

## Pipeline guide

Before changing a pipeline, read its linked guide and inspect the relevant source
entry point. Start with the [documentation index](docs/README.md) and
[repository structure](docs/repository_structure.md) for package boundaries.
Use current Markdown guides and source code for APIs and paths; the historical
Word manuals in `docs/manuals/` can contain obsolete imports and examples.

The usual flow is imagery acquisition → parcel statistics → feature enrichment
→ EDA → classification. EDA and classification also accept independently prepared
DataFrames or GeoDataFrames. Classification expects one row per entity.

### openEO / Sentinel parcel statistics

Acquires Sentinel-1/2 cubes, cleans and fills physical bands locally, calculates
indices and parcel statistics, and saves GeoParquet outputs and cleaning reports.
The public classes are `OpenEOZonalStats` and `OpenEOJobManagerZonalStats`.

- Start with the [public API](docs/data_preparation/README.md),
  [configuration guide](docs/data_preparation/openeo/README.md), and
  [detailed workflow and module map](docs/data_preparation/openeo/WORKFLOW.md).
- For tile planning, accounts, queues, retries, and resume behavior, read
  [parcel grouping](docs/data_preparation/openeo/PARCEL_GROUPING.md),
  [scheduling](docs/data_preparation/openeo/SCHEDULING.md), and
  [database account setup](docs/data_preparation/openeo/CREDENTIALS.md).
- For local cleaning, filling, temporary files, and completion checkpoints, read
  [NetCDF processing](docs/data_preparation/NETCDF_PROCESSING.md).
- Source: `src/data_preparation/sources/openeo/cubes.py` handles acquisition;
  `src/data_preparation/parcel_stats/openeo.py`, `job_manager.py`, and
  `multiuser.py` handle extraction and orchestration. Shared local processing
  lives in `src/data_preparation/parcel_stats/core/`.
- Project example:
  [1_parcel_stats_openEO_fill.ipynb](notebooks/projects/neuro/1_parcel_stats_openEO_fill.ipynb).

### Planet acquisition and parcel statistics

`PlanetBasemapZonalStats` processes monthly Planet basemaps from local imagery,
download manifests, or the HUB catalog, using the shared local statistics engine.

- Read the [Planet workflow](docs/data_preparation/PLANET.md) for delivery
  downloads, local manifests, catalog inputs, spectral assumptions, and outputs.
- Source: `src/data_preparation/sources/planet/delivery.py` handles delivery
  downloads; `src/data_preparation/sources/hub/` handles catalog access;
  `src/data_preparation/parcel_stats/planet.py` handles parcel statistics.
- Keep service-specific acquisition in its adapter. The shared
  `parcel_stats/core/` engine must remain independent of acquisition services.

### Parcel feature enrichment

Adds temporal summaries, elevation, soil, and nearest-lake features to parcel tables.

- Start with
  [2_parcel_features.ipynb](notebooks/projects/neuro/2_parcel_features.ipynb)
  for the project workflow and configured input/output paths.
- Inspect `src/data_preparation/features/temporal.py`, `elevation.py`, and
  `spatial_context.py` for the corresponding implementations and API docstrings.
  See [source boundaries](docs/repository_structure.md#source-boundaries)
  for their relationship to acquisition and statistics.

### Exploratory data analysis and feature screening

`EDAPipeline` with `EDAConfig` produces an HTML report, diagnostic tables,
annotated data, and an initial feature proposal.

- Read the [EDA guide](docs/eda/README.md) for configuration and artifacts, and
  [feature-selection criteria](docs/eda/FEATURE_SELECTION_GUIDE.md) for screening.
  Feature proposals are diagnostics; final selection requires train-only validation.
- Source entry point: `src/eda/pipeline.py`; configuration and profiling live in
  `src/eda/core/`, with reporting and plots in `reporting/` and `visuals/`.
- Project example: [3_eda_report.ipynb](notebooks/projects/neuro/3_eda_report.ipynb).

### ML classification and inspection priority

`GeospatialClassificationPipeline` with `ClassificationPipelineConfig` runs
Check → Prepare → Train → Evaluate → Predict. Each step persists artifacts under
`01_check/` through `05_predict/`; reports are linked from `report_index.html`.

- Start with the [overview](docs/ml_classification/OVERVIEW.md),
  [configuration and usage](docs/ml_classification/README.md), and
  [source guide](src/ml_classification/README.md).
- For implementation ownership and saved artifacts, use the
  [file usage index](src/ml_classification/FILE_USAGE.md) and
  [structure guide](docs/ml_classification/STRUCTURE.md).
- For resuming or adding models, read the
  [incremental workflow](docs/ml_classification/INCREMENTAL_WORKFLOW.md).
  For confidence and inspection scoring, read the
  [confidence methodology](docs/ml_classification/rank_based_ensemble_confidence_methodology.md)
  and [inspection decision table](docs/ml_classification/INSPECTION_PRIORITY_DECISION_TABLE.md).
- Source entry point: `src/ml_classification/pipeline.py`. Implementations live
  in `step_01_check/` through `step_05_predict/`, each with its own `libraries/`;
  classification-wide configuration, models, and reports live in `shared/`.
- Project example:
  [4_ml_classification.ipynb](notebooks/projects/neuro/4_ml_classification.ipynb).

### Classification seed sensitivity

`ClassificationSensitivityRunner` repeats the classification pipeline across
random seeds and writes a results CSV, accuracy plot, and HTML summary.

- Read [sensitivity analysis](docs/ml_classification/OVERVIEW.md#sensitivity-analysis).
- Source entry point: `src/ml_classification/sensitivity/runner.py`.
- Example: [test_sensitivity.ipynb](notebooks/experiments/test_sensitivity.ipynb).

## Working with notebooks and run outputs

- Read [notebook guidance](notebooks/README.md) and the notebook's path/configuration
  cells before execution. Existing projects may use external work directories.
- For status-only requests, inspect saved notebook outputs, logs, job records,
  and checkpoint files without executing cells, restarting kernels, or changing
  run files. Saved notebook output can lag behind the current logs.
- For openEO, distinguish remote job completion from download completion and
  local statistics completion. `tile_jobs.parquet` records remote job state;
  `.partial.nc` files are unfinished stages. A batch statistics cache requires
  both matching statistics and report Parquets. With
  `remove_nc_after_completion=True`, successful batches delete their NetCDFs.
  Consult the scheduling and NetCDF guides before interpreting missing files.
- Use the four `satellite_*.log` files documented in the
  [logging reference](docs/data_preparation/README.md#logs-and-persisted-filenames)
  to distinguish orchestration, acquisition, filling, and parcel statistics.
- EDA and ML can reset their configured output directories. Keep source inputs
  outside those directories and give each pipeline its own run subdirectory.

## Repository skills

Look for task-relevant `SKILL.md` files under both:

- `.github/skills/`
- `.copilot/`

Use each skill's name and description to decide whether it applies. Read the full
`SKILL.md` before using a selected skill, and load referenced resources only as
needed. Resolve relative resource paths against the skill's own directory.
If the user explicitly names a repository skill, locate and read it.

These directories are additional sources to consult under this instruction;
they do not replace skills supplied by Codex or installed plugins. Avoid loading
unrelated skills or running their scripts merely because they exist.

## Applying guidance

Explicit user instructions take precedence over repository guidance, subject to
system and developer instructions. Within repository guidance, apply more
specific file or task instructions over general defaults. If a material conflict
cannot be resolved by scope, explain it and request clarification.

If a required guidance file is missing or unreadable, report that limitation
rather than claiming to have followed it. Preserve unrelated local changes.
