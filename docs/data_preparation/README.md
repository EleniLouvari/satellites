# Satellite parcel statistics

Use the source-neutral public package for new workflows:

```python
from data_preparation.parcel_stats import (
    HubZonalStats,
    OpenEOZonalStats,
    OpenEOJobManagerZonalStats,
    reduce_annual_median_features,
)
```

| Name | Responsibility |
| --- | --- |
| `HubZonalStats` | Read HUB Sentinel-1/2 ARD windows and calculate local parcel statistics. |
| `OpenEOZonalStats` | Acquire Sentinel-1/2 cubes through openEO and calculate parcel statistics. |
| `OpenEOJobManagerZonalStats` | The optional tile/job-manager variant of the openEO workflow. |
| `ParcelBatchPlanner`, `RasterCleaner`, `ParcelStatisticsCalculator` | Shared geometry batching, raster cleaning and parcel aggregation components in `parcel_stats/core/`. |

`openEO` identifies the processing service; `Sentinel-1` and `Sentinel-2`
identify the imagery. Constructor options such as `sentinel2_bands` and
`sentinel1_orbit_direction` retain their sensor-specific names. Sentinel-2 red
is `B04`. [HUB Sentinel ARD](hub/README.md) acquisition lives under `sources/hub/`;
its parcel-statistics adapter is `parcel_stats/hub.py`. This checkout does not
export `PlanetBasemapZonalStats` or include a Planet delivery adapter.

Shared optical processing exposes `optical_bands`, `optical_indices` and
`_calculate_local_optical_index`. Each adapter defines `OPTICAL_BAND_ROLES`,
which translates native band IDs to roles such as `red`, `nir` and `swir_1`.
`_calculate_optical_index` implements the formulas once using those roles.
Future optical adapters should define their own roles and supported indices;
missing spectral coverage must not be replaced with an unrelated band.

## Implementation

For JSON configuration, command-line execution and restart, see the
[CLI and dictionary runner](openeo/CLI.md). The `prepare_run`, `run_parcel_stats`
and `resume_parcel_stats` public functions wrap the existing openEO engine.

`OpenEOZonalStats` and `OpenEOJobManagerZonalStats` are the canonical class names.
Only the current class names are exported. Implementations live in
`data_preparation.parcel_stats`, with service-independent processing
in `core/` and openEO acquisition in its adapter.

The openEO constructor uses sensor-specific `sentinel2_*` fields. The shared
engine reads `optical_*` properties and calculates indices through
`_calculate_local_optical_index`, using each adapter's spectral band roles.

See the [detailed openEO workflow](openeo/WORKFLOW.md) for acquisition through
parcel features. `OpenEOZonalStats.run_from_batches()` accepts already acquired
`(batch_number, parcels, netcdf_path)` tuples; see that guide for the input contract.

The multi-user openEO notebook loads accounts from PostgreSQL. See
[database account setup](openeo/CREDENTIALS.md) for dependencies, connection
settings, required table columns and restart ordering.

For a step-by-step explanation of local `.nc` processing, temporary `.partial.nc`
files, cleaning, parcel statistics, and restart behavior, see
[How NetCDF processing works](NETCDF_PROCESSING.md).

## Logs and persisted filenames

All pipeline-owned console and file loggers use `parcel_stats_pipeline` as their
namespace, including worker processes. The startup message identifies the actual
source; radar configuration is printed only when
radar inputs are selected.

- `satellite_zonal_stats.log`: orchestration and configuration.
- `satellite_source.log`: source acquisition activity.
- `satellite_raster_filling.log`: cleaning and filling.
- `satellite_parcel_statistics.log`: parcel aggregation and checkpoints.

When a dictionary-configured run (`run_parcel_stats` or `resume_parcel_stats`)
fails, the runner appends the exception type, message and causes, for example
the endpoint error behind a token-retry wrapper, to a run-level
`satellite_zonal_stats.log` in the output directory. `run.json` records only
`error_type`. The record uses the `parcel_stats_pipeline` namespace, so Parcel
Studio redacts it, shows it as a run event, and includes the file in failure
diagnostics.

Workers add `.worker-<process-id>` to their individual file names. With
`batch_workers > 1`, worker records are also forwarded to the parent process and
logged again under the same logger name, without writing a second file entry, so
caller logging hooks such as Parcel Studio's live events receive them.
Acquisition code uses `source_logger`.
Actual openEO backend connection and job messages identify that service.

Result tables, cleaning reports and raster checkpoints retain their existing
source-neutral `satellite_*`, batch and temporal filenames, preserving resume
behavior.
