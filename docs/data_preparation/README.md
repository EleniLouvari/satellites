# Satellite parcel statistics

Use the source-neutral public package for new workflows:

```python
from data_preparation.parcel_stats import (
    OpenEOZonalStats,
    OpenEOJobManagerZonalStats,
    PlanetBasemapZonalStats,
    reduce_annual_median_features,
)
```

| Name | Responsibility |
| --- | --- |
| `OpenEOZonalStats` | Acquire Sentinel-1/2 cubes through openEO and calculate parcel statistics. |
| `OpenEOJobManagerZonalStats` | The optional tile/job-manager variant of the openEO workflow. |
| `PlanetBasemapZonalStats` | Process Planet basemaps from local files or the HUB catalog. |
| `ParcelBatchPlanner`, `RasterCleaner`, `ParcelStatisticsCalculator` | Shared geometry batching, raster cleaning and parcel aggregation components in `parcel_stats/core/`. |

`openEO` identifies the processing service; `Sentinel-1`, `Sentinel-2` and
`Planet` identify the imagery. Accordingly, openEO constructor options such as
`sentinel2_bands` and `sentinel1_orbit_direction` retain their sensor-specific
names. Planet uses `indices`, `reflectance_scale`, `run_local` and `run_catalog`.
Band identifiers in output columns retain their source meaning: Planet `B6`
is red, whereas Sentinel-2 red is `B04`.

Shared optical processing exposes `optical_bands`, `optical_indices` and
`_calculate_local_optical_index`. Each adapter defines `OPTICAL_BAND_ROLES`,
which translates native band IDs to roles such as `red`, `nir` and `swir_1`.
`_calculate_optical_index` implements the formulas once using those roles.
Future optical adapters should define their own roles and supported indices;
missing spectral coverage must not be replaced with an unrelated band.

## Implementation

`OpenEOZonalStats` and `OpenEOJobManagerZonalStats` are the canonical class names.
Only the current class names are exported. Implementations live in
`data_preparation.parcel_stats`, with service-independent processing
in `core/` and separate openEO and Planet workflows.

The openEO constructor uses sensor-specific `sentinel2_*` fields. The shared
engine reads `optical_*` properties and calculates indices through
`_calculate_local_optical_index`, using each adapter's spectral band roles.

See the [detailed openEO workflow](openeo/WORKFLOW.md) for acquisition through
parcel features, or [Planet usage](PLANET.md) for local/downloaded inputs.

The multi-user openEO notebook loads accounts from PostgreSQL. See
[database account setup](openeo/CREDENTIALS.md) for dependencies, connection
settings, required table columns and restart ordering.

For a step-by-step explanation of local `.nc` processing, temporary `.partial.nc`
files, cleaning, parcel statistics, and restart behavior, see
[How NetCDF processing works](NETCDF_PROCESSING.md).

## Logs and persisted filenames

All pipeline-owned console and file loggers use `parcel_stats_pipeline` as their
namespace, including worker processes. The startup message identifies the actual
source (`openEO` or `Planet Basemaps`); radar configuration is printed only when
radar inputs are selected.

- `satellite_zonal_stats.log`: orchestration and configuration.
- `satellite_source.log`: source acquisition activity.
- `satellite_raster_filling.log`: cleaning and filling.
- `satellite_parcel_statistics.log`: parcel aggregation and checkpoints.

Workers add `.worker-<process-id>` to their individual file names.
Acquisition code uses `source_logger`.
Actual openEO backend connection and job messages identify that service.

Result tables, cleaning reports and raster checkpoints retain their existing
source-neutral `satellite_*`, batch and temporal filenames, preserving resume
behavior. Planet-specific manifests and delivery inventories retain `planet_*`
names because they describe Planet assets.
