# Satellite parcel statistics

Use the source-neutral public package for new workflows:

```python
from satellites.data_preparation.parcel_stats import (
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

## Compatibility

`OpenEOZonalStats` is an alias of the original `SatelliteZonalStats`;
`OpenEOJobManagerZonalStats` aliases `JobManagerSatelliteZonalStats`.
Existing imports, class identity, constructor arguments, saved output names and
cache signatures remain unchanged. Implementations now live in
`satellites.data_preparation.parcel_stats`, with service-independent processing
in `core/` and separate openEO and Planet workflows. Old module paths forward
to these implementations through compatibility shims.

The shared constructor still stores optical configuration in legacy
`sentinel2_*` fields. Read-only `optical_*` properties expose the same values
without duplicating state. Planet's legacy constant aliases bridge that
constructor; they do not imply Sentinel-2 imagery. The old private
`_calculate_local_sentinel2_index` method delegates to the optical implementation.

See [Planet usage](planet.md) for local/downloaded inputs.

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
New runs no longer create `satellite_openeo.log`. Existing log files are left in
place. `openeo_logger` remains a compatibility alias for `source_logger`.
Actual openEO backend connection and job messages still identify that service.

Result tables, cleaning reports and raster checkpoints retain their existing
source-neutral `satellite_*`, batch and temporal filenames, preserving resume
behavior. Planet-specific manifests and delivery inventories retain `planet_*`
names because they describe Planet assets. Python traceback paths and imports
can still contain the legacy implementation package name; these are not progress
messages and changing them would require relocating the compatibility package.
