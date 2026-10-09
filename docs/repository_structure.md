# Repository structure and migration

The installable distribution contains four top-level packages directly under
`src/`: `data_preparation`, `eda`, `ml_classification`, and `shared`. Notebooks,
tests, documentation, and generated results live outside these packages.

## Source boundaries

```text
src/
  data_preparation/
    sources/
      openeo/cubes.py             authentication, cube graphs, remote jobs
      planet/delivery.py          delivery discovery, downloads, manifests
      hub/                       catalog queries, asset downloads and audits
    parcel_stats/
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
      planet.py                  Planet input handling plus local statistics
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
    io.py logging.py constants.py formatting.py
    geometry.py raster.py s3.py
    tabular/
      __init__.py               stable public tabular API re-export
      core.py                   generic helpers (formatting, notebook/runtime helpers)
      transforms.py             dataframe normalization and type/categorical transforms
      analysis.py               plotting, outlier utilities, and confidence/stat helpers
      io.py                     model persistence and GIS/ZIP extraction helpers
    spatial_statistics/           clustering, weights, hotspots, and exploratory diagnostics
    data_cleaning/
    notebook.py                  explicit notebook setup
```

`PlanetBasemapZonalStats` and `OpenEOZonalStats` share `ParcelStatsBase`.
Planet no longer subclasses the openEO workflow. The base retains the historical
sensor configuration names for constructor/cache compatibility, but imports no
acquisition service. The public parcel-statistics package loads source classes
lazily, so importing Planet does not require openEO.

EDA and classification consume tables and shared helpers. They do not import
acquisition code or each other. Shared cleaning stays below the pipelines;
temporal reduction and environmental enrichment belong to data preparation.

### Shared spatial statistics

`shared.spatial_statistics` is a package. Existing function imports remain valid,
including `from shared.spatial_statistics import spatial_clustering_using_buffer`.
Implementations are grouped by responsibility:

| Module | Responsibility |
| --- | --- |
| `clustering.py` | K-means, DBSCAN, buffer/network clusters, and Voronoi polygons |
| `weights.py` | Distance-based and Queen-contiguity neighborhoods |
| `hotspots.py` | Attribute z-scores, local/global Moran statistics, and hotspot plots |
| `scale_selection.py` | Neighborhood-distance sweeps and their multiprocessing worker |
| `distributions.py` | Distribution comparisons, normality checks, and transformations |
| `trends.py` | Directional regressions against spatial coordinates |
| `similarity.py` | Buffered extent and rowwise overlap comparisons |
| `time_series.py` | CUSUM and PELT change-point diagnostics |

Submodules import their dependencies directly: hotspots uses weights, and distance
selection uses hotspots. The package initializer exports the existing functions.
This split preserves calculations and return values. Inline comments explain
assumptions and existing limitations, including legacy normality decisions,
row-count-based overlap, unit conversions, and sorting of paired samples.

## Installation and imports

Install or reinstall the repository into the environment used by your Jupyter
kernel after changing the source layout:

```powershell
python -m pip install --no-deps --no-build-isolation -e .
```

Restart the Python process or notebook kernel afterward. Import directly from
`data_preparation`, `eda`, `ml_classification`, and `shared`; there is no enclosing
Python namespace or compatibility package. See the [import guide](import_migration.md).
The distribution name in package metadata remains unchanged.

For source-tree development, `PYTHONPATH` needs only `src`. Pytest and VS Code
already use this directory. Package discovery includes the four top-level
packages and their subpackages; coverage measures the same four packages.

Constructor arguments, output filenames, log namespaces and cache signatures
remain unchanged by the directory move. Python pickle/joblib artifacts store
module paths, so previously serialized custom classes may need regeneration
under the new imports. Raster caches and tabular output files do not depend on
Python class import paths.

## Notebooks, reference data and artifacts

- Volvi notebooks remain together in `notebooks/projects/volvi/`, in workflow order.
- Experimental notebooks are in `notebooks/experiments/`; maintenance notebooks
  are in `notebooks/maintenance/`.
- Imports use canonical paths. Notebook code no longer patches `sys.path` or
  imports the all-libraries module. Display and seed setup is explicit.
- Existing external data/output paths and saved notebook outputs are retained.
  Historical outputs may still display the paths used when they were executed.
- The Planet grid is in `data/reference/planet_basemap_quads.gpkg`, unchanged.
- The ML output template is documented under
  `docs/ml_classification/output_template/`; it is not a source-code package.
- New runs should use `outputs/<project>/<run>/` and dedicated pipeline
  subfolders. The example in `configs/` documents external data roots.
- Old coverage and quality reports were retained in `outputs/quality/previous/`.
  Old source-folder remnants were retained in `tmp/reorganization-leftovers/`.
  Generated results and scratch files are ignored by Git.

The original Word manuals are preserved unchanged in `docs/manuals/`. Markdown
guides provide the updated paths. No remote data was downloaded, no production
pipeline was executed, and no external run directory was moved as part of this migration.

## Development and verification

Tests live under `tests/`, and VS Code is configured for pytest. Development
scripts use the active Python environment, resolve the repository relative to
their own location, and place reports in `outputs/quality/`. The code-check script
reports failures through its exit code and does not automatically modify source.

Layout tests in `tests/test_repository_layout.py` cover top-level package
locations, fitted-model/configuration serialization with current module paths,
Windows-style spawned-worker serialization, and import-side-effect isolation.
The local Planet suite covers data discovery, downloaded-manifest processing,
cleaning checkpoints and resume behavior.
