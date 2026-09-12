# Repository structure and migration

The repository has one installable package, `satellites`, with separate data
preparation, EDA, and classification areas. Notebooks, tests, documentation,
reference data, and generated results are outside the maintained source package.

## Source boundaries

```text
src/satellites/
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
      core/parcel_statistics.py  masking, aggregation and parcel features
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
    core/ steps/ reporting/ visuals/ sensitivity/
  shared/
    io.py logging.py constants.py formatting.py
    geometry.py raster.py tabular.py spatial_statistics.py s3.py
    data_cleaning/
    notebook.py                  explicit notebook setup
    legacy/                      old convenience imports and EDA class
```

`PlanetBasemapZonalStats` and `SatelliteZonalStats` now share `ParcelStatsBase`.
Planet no longer subclasses the openEO workflow. The base retains the historical
sensor configuration names for constructor/cache compatibility, but imports no
acquisition service. The public parcel-statistics package loads source classes
lazily, so importing Planet does not require openEO.

EDA and classification consume tables and shared helpers. They do not import
acquisition code or each other. Shared cleaning stays below the pipelines;
temporal reduction and environmental enrichment belong to data preparation.

## Installation and compatibility

Install the repository into the environment used by your Jupyter kernel:

```powershell
python -m pip install --no-deps --no-build-isolation -e .
```

Existing geospatial environments can be reused. `pyproject.toml` declares the
maintained pipeline dependencies and optional acquisition, ML, neural,
interpolation, notebook, GIS-utility, and development groups. The broad historical
`import_libraries` API still needs its historical environment; the installation
does not reconstruct every dependency of those legacy convenience imports.

See the complete [import map](import_migration.md). Old module paths are retained
in `src/_compat/`, with two standalone compatibility modules at the `src` root.
These aliases resolve to canonical module objects rather than duplicate class
implementations. This preserves old nested imports, private symbols, monkeypatch
targets, and import paths carried by existing Python pickles/joblib models.

Class names, public constructor arguments, output filenames, log namespaces and
cache signatures remain stable. `OpenEOZonalStats` continues to alias
`SatelliteZonalStats`. Code explicitly testing whether a Planet extractor is an
instance of the openEO class should now check `ParcelStatsBase` instead.

An editable installation is the normal entry point. If using raw `PYTHONPATH`,
include both `src` and `src/_compat` for old imports. The maintained package only
needs `src`. Pytest configures both paths for source-tree development.

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

Migration tests in `tests/test_repository_layout.py` cover historical module
identity, loading fitted-model/configuration pickles with old module paths,
Windows-style spawned-worker serialization, and import-side-effect isolation.
The local Planet suite covers data discovery, downloaded-manifest processing,
cleaning checkpoints and resume behavior.

The pre-migration test suite already imports a missing
`core.rank_confidence_calibration` module from `test_rank_confidence.py`.
Without that uncollectable file, the original suite also has five failures in
`test_train_evaluate_steps.py`: model-candidate expectations, two outdated method
signatures, and two fixtures missing `log_path`. These tests remain visible;
the reorganization does not suppress them or change classification behavior to
make obsolete tests pass. See `validation.md` for the completed comparison.

The older `shared/raster.py` utility also carries pre-existing unresolved helper
references/constants from its former `common_libraries` home. It is not used by
the maintained parcel-statistics raster-cleaning engine. Repairing those legacy
functions is a separate task; their contents are retained.
