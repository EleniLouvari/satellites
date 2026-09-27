# Satellite parcel zonal statistics

`satellites.data_preparation.parcel_stats` extracts Sentinel-1 and Sentinel-2 observations from the
Copernicus Data Space Ecosystem (CDSE) openEO backend and calculates temporal
statistics for parcel polygons.

The remote openEO jobs create temporally aggregated raster cubes, or retain all
acquisitions when `temporal_reducer="none"`. Pixel cleaning,
parcel masking, zonal statistics, derived statistics, and final file creation are
performed locally.

## Implementations

The package provides two classes with the same processing and output options:

| Class | Remote batching strategy | Recommended use |
|---|---|---|
| `SatelliteZonalStats` | Groups nearby parcels into 50 km grid cells and splits dense cells into batches of at most 5,000 parcels | General use and parcel-count-bounded jobs |
| `JobManagerSatelliteZonalStats` | Creates one job per non-empty, fixed-size geographic tile and manages the jobs with openEO `MultiBackendJobManager` | Restartable tile jobs, explicit tile extents, and controlled remote concurrency |

The job-manager implementation does **not** balance jobs by parcel count. A tile
containing 56,150 parcel representative points and neighboring tiles containing 2
and 137 points is valid. The tiles are geographic grid cells; parcel density
determines the number of parcels assigned to each job.

## Internal architecture

`SatelliteZonalStats` remains the backward-compatible public facade and
orchestrator. Its processing responsibilities are separated into focused modules:

| Module | Responsibility |
|---|---|
| `configuration.py` | Option validation, temporal intervals, sensor-variable selection, and cache signatures |
| `parcel_batches.py` | Parcel geometry repair, normalization, and spatial batching |
| `openeo_pipeline.py` | openEO authentication, cube graphs, remote jobs, downloads, and caching |
| `raster_cleaning.py` | NetCDF variable selection, raster-level IQR removal, staged null filling, and cleaning audits |
| `parcel_statistics.py` | Parcel masking, zonal reductions, and derived parcel statistics |
| `zonal_stats.py` | Public construction, component orchestration, parallel local execution, and persistence |

The component classes retain the established private method names through
inheritance so `JobManagerSatelliteZonalStats` and existing integrations remain
compatible while each implementation has a single scope.

## Processing flow

1. Validate dates, CRS, sensor selections, statistics, and cleaning options.
2. Repair parcel geometries, retain polygonal parts, validate unique parcel IDs,
   and transform parcels to EPSG:4326 for openEO.
3. Build spatial batches or tiles.
4. Connect and authenticate to `https://openeo.dataspace.copernicus.eu`.
5. Load Sentinel collections for the requested date range.
6. For Sentinel-2, resample reflectance to 10 m, apply the `0.0001` scale factor,
   mask invalid/cloud SCL classes, and retain all physical bands required by the
   requested outputs and indices.
7. For Sentinel-1, calculate linear-power `sigma0-ellipsoid` backscatter using the
   `COPERNICUS_30` elevation model and resample to 10 m. No logarithmic/dB
   conversion is applied.
8. Aggregate acquisitions into the configured temporal periods, or retain all
   acquisition timestamps when `temporal_reducer="none"`.
9. Download each raster cube as NetCDF and cache it locally.
10. For every physical band and time slice, remove IQR outliers once across the
    complete raster, then fill nulls with the temporal, 3x3, 5x5, and final
    interpolation passes.
11. Calculate every requested Sentinel-2 and Sentinel-1 index independently at
    each filled raster pixel. An observation-only copy is calculated separately
    so statistics and provenance distinguish indices that depend on an imputed
    source pixel.
12. Mask the filled raster by parcel and calculate parcel statistics. With
    `temporal_reducer="none"`, pool all valid pixel observations across acquisitions
    within each configured period, separately for each band/index. Spatial filling
    may use neighboring pixels that fall inside an adjacent parcel.
13. Add parcel metrics and derived statistics.
14. Pivot every temporal feature to dated columns, attach the original parcel
    attributes and geometry once, reproject geometry to `working_epsg`, and write
    one ML-ready GeoParquet row per parcel.

## Requirements and setup

Install the repository into your geospatial environment using `pyproject.toml`
(see the root README for installation). Important
dependencies include GeoPandas, openEO Python client 0.50.x, pandas, NumPy,
Xarray, rioxarray, Rasterio, PyProj, Shapely, NetCDF4/H5NetCDF, PyArrow, and
PyKrige.

The repository's `src` directory must be importable. When running outside the
provided notebooks or development environment, add it to `PYTHONPATH` or install
the repository as a package.

An active CDSE account and openEO OIDC authentication are required. On the first
run, the openEO client may request interactive authentication. Later runs can use
the refresh token stored by the client.

To select a particular CDSE account in a notebook, pass its username and password
from environment variables. Do not put credentials directly in notebook source:

```python
import os

extractor = SatelliteZonalStats(
    # ...the normal arguments...
    openeo_username=os.environ["OPENEO_USERNAME"],
    openeo_password=os.environ["OPENEO_PASSWORD"],
)
```

Both credential arguments must be supplied together. If neither is supplied,
the existing cached/interactive OIDC authentication behavior is retained. This
password flow also requires support from the configured CDSE OIDC provider.

## Parcel input

`parcels` must be a `geopandas.GeoDataFrame` with:

- a defined CRS;
- Polygon or MultiPolygon geometries;
- a unique parcel identifier column matching `parcel_id_field`.

If the requested identifier column does not exist, string values from the
GeoDataFrame index are used. Null, empty, or non-polygonal geometries are removed.
Invalid geometries are repaired where possible. Duplicate parcel IDs raise an
error.

Choose a projected `working_epsg` appropriate for the parcel location, normally
the local UTM zone. It is used for metre-based grids, parcel area, tile buffers,
local interpolation, the requested NetCDF raster grid, and output GeoParquet
geometry. A geographic CRS such as EPSG:4326 is not accepted as `working_epsg`.

Select the WGS84 UTM CRS containing the dissolved parcel centroid when the
working CRS should be chosen automatically:

```python
from satellites.data_preparation.parcel_stats import estimate_utm_epsg_from_parcels

working_epsg = estimate_utm_epsg_from_parcels(parcels)
parcels = parcels.to_crs(working_epsg)
```

Example input:

```python
import geopandas as gpd

parcels = gpd.read_file("parcels.gpkg", layer="parcels")
parcels = parcels[["parcel_id", "geometry"]]
```

## Common constructor parameters

These parameters apply to both classes.

### Required parameters

| Parameter | Type | Meaning |
|---|---|---|
| `parcels` | `GeoDataFrame` | Parcel polygons with a CRS |
| `start_date` | string or `pandas.Timestamp` | Inclusive first date |
| `end_date` | string or `pandas.Timestamp` | Inclusive last date; must not precede `start_date` |
| `output_dir` | string or `Path` | Cache, log, and result directory |
| `working_epsg` | integer | Projected CRS used for metric operations |

### Sensor parameters

| Parameter | Default | Accepted values and behavior |
|---|---:|---|
| `parcel_id_field` | `"parcel_id"` | Unique parcel ID column name |
| `sentinel2_bands` | `None` | `None` selects `B02`, `B03`, `B04`, `B05`, `B08`; `[]` disables standalone S2 bands; supported bands are `B01`, `B02`, `B03`, `B04`, `B05`, `B06`, `B07`, `B08`, `B8A`, `B09`, `B11`, `B12` |
| `sentinel2_indices` | `[]` | Any of `NDVI`, `NDWI`, `MNDWI`, `NDMI`, `NBR`, `GNDVI`, `EVI`, `SAVI`, `MSAVI`, `NDRE`, `PSRI`, `CI`; `None` or `[]` disables indices |
| `sentinel1_bands` | `None` | Any of `VV`, `VH`; `None` or `[]` disables standalone Sentinel-1 band outputs unless an index requires them as sources |
| `sentinel1_indices` | `[]` | `R` = `VV / VH`; `RVI` = `4 * VH / (VV + VH)`; `None` or `[]` disables indices and source bands VV/VH are loaded automatically otherwise |
| `sentinel1_orbit_direction` | `"BOTH"` | `ASCENDING`, `DESCENDING`, or `BOTH`; `BOTH` loads both directions without an orbit filter |

At least one Sentinel-2 band/index or Sentinel-1 band/index must be
selected. Source bands required to calculate an index are loaded automatically
and cleaned even when they are not emitted as standalone output variables.
Indices already present in a reused raw NetCDF are ignored and recalculated
locally from these cleaned source bands.

#### Why bands are cleaned before indices are calculated

The pipeline deliberately applies the following order:

```text
physical bands -> IQR outlier removal -> null filling -> pixel-level indices -> parcel statistics
```

Cleaning the physical source bands first is more defensible than filling
each derived index independently:

- one missing or anomalous source-band value is handled consistently for every
  index that depends on it;
- anomalous reflectance does not propagate into several indices;
- indices such as NDVI, SAVI, and EVI remain algebraically consistent because
  they use the same cleaned red and near-infrared values;
- invalid denominators and physically unsupported index results remain explicit;
- cleaned physical bands can support additional indices later without repeating
  the cleaning process.

Filled reflectance values are imputed rather than directly observed. The final
checkpoint therefore retains `cleaned` values, an `observed_mask`, and a
`temporal_filled_mask`. These distinguish original post-IQR observations from
temporal fills and later spatial fills. Index provenance is calculated from the
same intermediate source-band states, so an index is attributed to temporal
filling only when all source values needed for that index are available after
the temporal stage.

### Temporal and statistic parameters

| Parameter | Default | Accepted values and behavior |
|---|---:|---|
| `temporal_period` | `"1M"` | Positive day, month, or year interval such as `15D`, `1M`, `2M`, `3M`, `1Y` |
| `temporal_reducer` | `"median"` | `mean`, `median`, `min`, `max`, `sum`, or `"none"` / `None` to pool all acquisition pixels locally |
| `spatial_statistics` | `None` | `mean`, `median`, `sd`, `min`, `max`, `p10`, `p25`, `p75`, `p90`; `mean` is always added. Legacy `count` is accepted but emits only the static parcel pixel count described below. |
| `batch_workers` | `1` | Positive integer controlling base-class remote job waves and local statistics processes; for the manager it controls local statistics only |

Temporal intervals start at `start_date`, are half-open internally, and the final
interval is clipped after the inclusive `end_date`. For example, `1M` means
one-month steps from the configured start date, not necessarily calendar months.

### All valid pixels within each period

Set these constructor options (or the same keys in a multi-user configuration):

```python
temporal_period="1M",
temporal_reducer="none",
remove_outliers=False,
fill_nulls=False,
```

This downloads every acquisition after the existing sensor preprocessing and
cloud masking, without a temporal image reduction. Locally, each parcel's valid
pixels from every acquisition in the period form one sample pool per band/index.
For example, images contributing `[1, 2, 3, 4]` and `[10]` produce statistics over
`[1, 2, 3, 4, 10]`: mean `4` and median `3`. Every observation has equal weight;
a spatial pixel observed on several dates contributes once per acquisition.
NaN and infinite values do not participate. An empty period has null statistics.

Indices are calculated from bands at the same pixel and acquisition before
pooling. Sentinel-1 and Sentinel-2 may have different acquisition timestamps;
each output uses its own available observations. Acquisitions on the same day
are retained separately. The configured final calendar day is included.

Other intervals, such as `15D`, `3M`, or `1Y`, work the same way. Output naming
is unchanged: `NDVI_median__20240101` is the median of all valid NDVI pixel
observations in that period. The separate annual feature file still summarizes
the per-period parcel medians.

The cleaning options remain active: `remove_outliers=True` filters each
acquisition/band before pooling, and `fill_nulls=True` allows imputed pixels to
participate. Keep both `False` to use all observed valid pixels. Cleaning audit
rows and raster checkpoints retain acquisition timestamps in this mode.
`intersected_pixel_count` still counts spatial cells, not repeated observations;
fill provenance is calculated across acquisition/band/pixel slots.

Raw downloads use a separate cache signature from composite runs. Keeping every
acquisition increases download size and checkpoint disk use. The existing reducers
and default `"median"` behavior remain available.

### Memory use and large acquisition cubes

The openEO local processing path streams data through NetCDF checkpoints:

- Raw data are read one acquisition/band layer at a time. IQR bounds still use
  the complete raster layer.
- Temporal filling processes small spatial strips containing every acquisition.
  Coverage thresholds are calculated for the full layer, and temporal neighbors
  can cross reporting-period boundaries as before.
- Spatial filling and interpolation process complete individual layers. Indices
  are calculated one acquisition and output variable at a time.
- Parcel statistics read one parcel window, band/index, and reporting period at
  a time. A resumed final checkpoint is read in windows without loading either
  the complete raw or final cube.

Files are closed on completion or failure. Temporary arrays are released between
stages and cubes; garbage collection runs after each cube. Failed processing
frames are cleared so notebook error tracebacks do not retain their raster arrays.

Within a process, multi-user queues serialize local raster work while remote
openEO jobs can continue concurrently. `batch_workers > 1` explicitly creates
separate worker processes, each with its own memory needs. Keep
`batch_workers=1` when memory is limited. Peak memory still depends on a full
spatial layer and the largest parcel/band/period sample pool, particularly for
exact medians and percentiles; it no longer requires all dates and bands in RAM.

This trades additional disk I/O for lower RAM use. Allow space for physical-band
and final index checkpoints alongside the raw downloads. The intermediate
physical checkpoint is removed after the final checkpoint commits unless
`keep_cleaned_checkpoint=True`. Existing complete final checkpoints remain
usable. Older intermediate checkpoints without temporal masks are rebuilt from
the cached raw cube, without downloading again.

After updating code following a notebook `MemoryError`, restart the kernel before
rerunning to load the new implementation and release arrays held by the old run.

### Outlier removal and null filling

| Parameter | Default | Meaning |
|---|---:|---|
| `remove_outliers` | `False` | Replace pixel values outside parcel/variable/period IQR bounds with nulls |
| `iqr_quantiles` | `(0.25, 0.75)` | Lower and upper quantiles; must satisfy `0 <= lower < upper <= 1` |
| `iqr_multiplier` | `1.5` | Non-negative multiplier used to extend the IQR bounds |
| `iqr_min_valid_pixels` | `20` | Minimum valid raster pixels required before applying IQR filtering |
| `fill_nulls` | `False` | Enable temporal, sequential spatial-neighbor, and final interpolation filling |
| `spatial_fill_window_sizes` | `(3, 5)` | Ordered spatial-mean passes: `(3,)`, `(3, 5)`, `(3, 5, 7)`, or `(3, 5, 7, 9)` |
| `keep_cleaned_checkpoint` | `False` | Retain the intermediate cleaned physical-band NetCDF after the final NetCDF commits |
| `temporal_fill_mode` | `"past_only"` | `past_only` carries the nearest earlier value; `bidirectional` uses closest earlier/later values |
| `minimum_parcel_pixels` | `3` | Threshold for the output quality flag; small parcels are retained |
| `minimum_observed_fraction_for_fill` | `0.20` | Minimum observed fraction for a raster-variable-period to be filled or used as a temporal source |
| `interpolation_method` | `"nearest"` | Final interpolation method: raster-native `nearest`, `idw`, or `kriging` |
| `interpolation_max_distance_in_meters` | `None` | Required positive search distance in metres for `idw` |
| `interpolation_variogram_lags` | `15` | Positive kriging variogram lag count |
| `interpolation_variogram_max_distance_in_meters` | `None` | Required positive distance in metres for `kriging` |

Cleaning is performed independently for every physical band and temporal period
(or acquisition when `temporal_reducer="none"`)
across the complete batch raster before parcel masking. `fill_nulls=True` does
not guarantee every null can be filled;
the observed-fraction threshold, available neighbors, and distance settings still
apply.

The default stops after 3x3 and 5x5 means before using raster-native nearest
interpolation. Larger 7x7 and 9x9 passes are optional because they progressively
smooth reflectance and can mix a wider neighborhood. In a 256x256 benchmark with
15% missing pixels, average stage times were approximately 1.0 ms (3x3), 1.7 ms
(5x5), 2.7 ms (7x7), and 4.5 ms (9x9). Raster-native nearest took about 2.5 ms,
versus 39.8 ms for the former geometry-based nearest path (approximately 16x
faster). Results vary with raster size and missingness, but this supports keeping
7x7/9x9 opt-in rather than paying their smoothing and runtime costs by default.

## Basic usage

### Sentinel-1-only extraction

```python
from satellites.data_preparation.parcel_stats import SatelliteZonalStats

extractor = SatelliteZonalStats(
    parcels=parcels,
    start_date="2023-01-01",
    end_date="2024-12-31",
    output_dir=r"C:\work_dir\kozani_s1",
    working_epsg=32634,
    parcel_id_field="parcel_id",
    sentinel2_bands=[],
    sentinel2_indices=[],
    sentinel1_bands=["VV", "VH"],
    sentinel1_indices=["R", "RVI"],
    sentinel1_orbit_direction="ASCENDING",
    spatial_statistics=[
        "mean", "median", "sd", "min", "max",
        "p10", "p25", "p75", "p90",
    ],
    remove_outliers=True,
    iqr_quantiles=(0.10, 0.90),
    iqr_multiplier=1.5,
    fill_nulls=True,
    temporal_period="1M",
    temporal_reducer="median",
    batch_workers=2,
)

result = extractor.run()
cleaning_report = extractor.cleaning_report
```

### Sentinel-2 bands and indices

```python
from satellites.data_preparation.parcel_stats import SatelliteZonalStats

extractor = SatelliteZonalStats(
    parcels=parcels,
    start_date="2024-03-01",
    end_date="2024-10-31",
    output_dir=r"C:\work_dir\kozani_s2",
    working_epsg=32634,
    sentinel2_bands=["B02", "B03", "B04", "B08"],
    sentinel2_indices=["NDVI", "NDMI"],
    sentinel1_bands=[],
    spatial_statistics=["mean", "median", "sd", "count"],
    temporal_period="15D",
    temporal_reducer="median",
)

result = extractor.run()
```

## Job-manager tile workflow

`JobManagerSatelliteZonalStats` accepts every common parameter plus:

| Parameter | Default | Meaning |
|---|---:|---|
| `tile_size_metres` | `50000` | Core tile width/height, or maximum buffered width/height when fitting to parcels, in metres |
| `tile_buffer_metres` | `500` | Non-negative buffer around every remote raster extent |
| `fit_tiles_to_parcels` | `False` | Fit cubes to parcel bounds plus buffer; split only when either dimension exceeds the tile limit |
| `openeo_parallel_jobs` | `2` | Maximum active CDSE jobs managed remotely |
| `job_poll_seconds` | `30` | Positive number of seconds between status polls |
| `max_job_retries` | `3` | Bounded retries after terminal `error` or a failed start request; persisted across restarts |
| `job_retry_delay_seconds` | `60` | Cooldown before retry waves for transient backend/API failures |
| `run_identifier` | generated timestamp | Readable label, at most 80 characters, included in openEO job titles |

### Fitting cubes to cluster bounds

Create one `spatial_part` per cluster and enable fitted extents in the multi-user
extraction configuration:

```python
if gdf_grouped["cluster"].isna().any():
    raise ValueError("Every parcel must have a cluster label.")
spatial_parts = [part.copy() for _, part in gdf_grouped.groupby("cluster", sort=True, observed=True)]
extraction_config.update(
    fit_tiles_to_parcels=True,
    tile_size_metres=20_000,
    tile_buffer_metres=1_400,
)
```

Each partition first gets one rectangular extent computed from all its parcel
geometries in `working_epsg`, padded by the buffer on every side. The requested
width is `east - west + 2 * buffer`, and height is `north - south + 2 * buffer`.
If both are at most `tile_size_metres`, the entire partition uses one cube.
Otherwise, the planner divides the parcels spatially along the longer axis and
repeats until every subgroup fits. Each subgroup uses its own bounds plus buffer.
Parcel geometries are never cut and each parcel belongs to exactly one job.
An individual parcel exceeding the limit with its buffer raises an error before
submission. The limit controls requested spatial extent, not backend memory or
pixel-count quotas, which also depend on bands, resolution and time range.

Existing per-user queues distribute the cluster partitions. Fitted cubes have a
separate cache namespace from regular grid cubes. The Neuro fill notebook uses
this mode; other callers retain regular grid tiles unless they enable it.

### How regular-grid parcel ownership works

1. A regular core grid covers the total parcel extent in `working_epsg`.
2. Each parcel is owned by the one core tile intersecting its representative
   interior point.
3. Shared-edge matches are de-duplicated deterministically.
4. Empty tiles do not create jobs.
5. Every non-empty tile creates one job, regardless of its parcel count.
6. The remote raster extent is the core tile plus `tile_buffer_metres` on every
   side, ensuring that boundary-crossing parcels remain complete.
7. Ownership by the unbuffered core tile prevents duplicate parcel results where
   buffered raster extents overlap.

Core tile size is configured, not calculated from the number of intersecting
parcels. Edge cells can be clipped by the overall AOI. Parcel counts can therefore
be extremely unequal. `openeo_parallel_jobs=2` allows only two active remote jobs;
later jobs wait for a slot, which can create a long delay between job-creation log
messages.

### Selecting tile size and buffer

Two helper functions are available:

```python
from satellites.data_preparation.parcel_stats.job_manager import (
    compute_tile_buffer_metres,
    compute_tile_width,
)

tile_size = compute_tile_width(
    parcels,
    parcel_id_col="parcel_id",
    working_epsg=32634,
    spatial_parts=spatial_parts,
    user_count=len(users_list),
    jobs_per_user=2,
)
tile_buffer = compute_tile_buffer_metres(
    parcels,
    working_epsg=32634,
)
```

`compute_tile_width` tests 5, 10, 20, 30, 40, 50, and 60 km candidates inside
each actual spatial partition. It returns the **largest** candidate that still
creates at least `jobs_per_user` non-empty tile jobs in every partition. With
one initial partition per user and `jobs_per_user=2`, all accounts can start two
remote jobs while avoiding an unnecessarily large number of tiny jobs. Parcels
may be supplied in any defined CRS; the helper reprojects internally.

Core-tile crossings do not control this recommendation. Ownership uses each
parcel's representative point, while `compute_tile_buffer_metres` ensures the
full parcel is present in the downloaded buffered extent.

`compute_tile_buffer_metres` measures the largest parcel radius from its
representative point to its bounding-box corners, multiplies it by a safety
factor of 2.5, and rounds upward to 50 m by default. It accepts parcels in any
defined CRS because it reprojects internally.

If a parcel is not completely covered by its assigned buffered tile, planning
stops and writes `uncovered_buffered_tile_parcels.csv` inside the cube cache.
Increase `tile_buffer_metres` and retry.

### Complete manager example

```python
from satellites.data_preparation.parcel_stats import JobManagerSatelliteZonalStats
from satellites.data_preparation.parcel_stats.job_manager import (
    compute_tile_buffer_metres,
    compute_tile_width,
)

working_epsg = 32634
tile_size = compute_tile_width(
    parcels,
    "parcel_id",
    working_epsg,
    spatial_parts=spatial_parts,
    user_count=len(users_list),
    jobs_per_user=2,
)
tile_buffer = compute_tile_buffer_metres(parcels, working_epsg)

extractor = JobManagerSatelliteZonalStats(
    parcels=parcels,
    start_date="2023-01-01",
    end_date="2024-12-31",
    output_dir=r"C:\work_dir\kozani_s1_manager",
    working_epsg=working_epsg,
    parcel_id_field="parcel_id",
    sentinel2_bands=[],
    sentinel1_bands=["VV", "VH"],
    spatial_statistics=[
        "mean", "median", "sd", "min", "max",
        "p10", "p25", "p75", "p90",
    ],
    remove_outliers=True,
    iqr_quantiles=(0.10, 0.90),
    fill_nulls=True,
    temporal_period="1M",
    temporal_reducer="median",
    batch_workers=2,
    tile_size_metres=tile_size,
    tile_buffer_metres=tile_buffer,
    openeo_parallel_jobs=2,
    job_poll_seconds=30,
    run_identifier="kozani-2023-2024-S1",
)

result = extractor.run()
```

## Outputs and cache

The output directory contains:

| Path | Contents |
|---|---|
| `satellite_parcel_ml_features.geoparquet` | One ML-ready row per parcel, with parcel attributes, geometry/CRS, static metrics, and dated temporal features |
| `satellite_pixel_cleaning_report.csv` | Per-batch raster, period, and variable null counts after every cleaning/filling step |
| `satellite_zonal_stats.log` | Overall orchestration and configuration |
| `satellite_openeo.log` | openEO authentication, job creation, execution, and downloads |
| `satellite_raster_filling.log` | Per-variable, per-period raster filling progress and counts |
| `satellite_parcel_statistics.log` | Parcel-mask and aggregation progress |
| `monthly_cubes/<signature>/batch_XXXXX_monthly.nc` | Cached temporal raster cubes |
| `monthly_cubes/<signature>/batch_XXXXX_monthly_cleaned.nc` | Temporary restart-safe cleaned pixels plus a compact observation mask |
| `monthly_cubes/<signature>/batch_XXXXX_monthly_cleaning_report.parquet` | Per-batch cleaning audit paired with the cleaned raster checkpoint |
| `monthly_cubes/<signature>/batch_XXXXX_monthly_final.nc` | Final requested bands and locally calculated indices plus a compact observation mask |
| `monthly_cubes/<signature>/tile_jobs.parquet` | Manager job database, manager implementation only |
| `monthly_cubes/<signature>/job_manager/` | Manager status/error artifacts, manager implementation only |

The cleaning report records `null_count_before_iqr`, `null_count_after_iqr`,
`null_count_after_temporal_fill`, `null_count_after_3x3_fill`,
`null_count_after_5x5_fill`, `null_count_after_7x7_fill`,
`null_count_after_9x9_fill`, and `null_count_after_interpolation`. Counts for
optional windows that were not enabled are carried forward unchanged. The
corresponding `filled_count_by_*` fields record how many pixels each stage
filled. The
`final_null_count` and `has_remaining_nulls` fields make incomplete filling
explicit for every batch, period, and variable.

With `batch_workers > 1`, local filling and parcel aggregation run in separate
processes. Their files use the same scope names with a `.worker-<pid>.log`
suffix so concurrent writers do not corrupt one shared log. Console messages
remain visible while the workers run.

The final GeoDataFrame has exactly one row per parcel. Core columns include:

- the parcel ID and every non-geometry attribute supplied with the input parcels,
  including a class/label column when present;
- static pixel fields such as `intersected_pixel_count`,
  `meets_minimum_pixel_count`, and `batch_number`; per-band/per-period
  `*_count` columns are no longer emitted;
- `expected_pixel_count`, calculated as intersected pixels x requested output
  bands/indices x temporal periods, plus `temporal_filled_pixel_count` and
  `spatial_filled_pixel_count` summed over those same slots. Spatial filling
  includes every configured neighborhood pass and final interpolation;
- `temporal_filled_ratio` and `spatial_filled_ratio`, each using
  `expected_pixel_count` as its denominator;
- `data_reliability_score`, the directly observed pixel-value count divided by
  `expected_pixel_count`. It ranges from 0 to 1; higher values indicate less
  dependence on temporal/spatial imputation and fewer unresolved gaps;
- area fields `pixel_area` (intersected pixels x raster cell area), `geom_area`
  (projected polygon area), and `geom_interior_area_ratio` (`pixel_area /
  geom_area`). A ratio of 1 means the rasterized and vector areas agree;
- shape fields `geom_compactness` (`4*pi*A/P^2`),
  `geom_perimeter_area_ratio` (`P/A`), `geom_shape_index`
  (`P/(2*sqrt(pi*A))`), and rotation-independent `geom_elongation`. Elongation
  is the long/short side ratio of the minimum rotated rectangle, so 1 is roughly
  square and larger values are longer and narrower;
- `geom_shape_complexity_score`, a non-negative composite of raster/vector area
  disagreement, boundary irregularity, and elongation. Zero is the theoretical
  simplest case; larger values indicate more complex parcel shapes. The score
  uses absolute/log penalties, normalizes perimeter-area ratio against an
  equal-area circle, and averages compactness, normalized perimeter-area ratio,
  and shape index into one boundary term so equivalent circularity signals are
  not triple-counted;
- dated temporal features such as `VV_mean__20240101`,
  `NDVI_median__20240201`, or `B04_p90__20240301`;
- derived fields when their source statistics exist: range, variance,
  coefficient of variation, IQR, Bowley skewness, p90-p10 spread, and valid-pixel
  fractions, also suffixed with their period;
- parcel geometry in the configured projected `working_epsg`.

The `feature__YYYYMMDD` naming matches the former ML-pipeline longitudinal
reshape. The ML pipeline can therefore select dated feature columns directly;
it no longer needs `period_start`, `reshape_time_series`, or `time_column`.

The cache signature includes dates, processing configuration, parcel IDs, and
parcel geometries. The manager also includes tile size and buffer. An identical
run reuses completed raw and cleaned NetCDF files. After a batch finishes IQR
outlier removal and null filling, its cleaned raster and audit are written
atomically. Indices already present in an older raw NetCDF are ignored; configured
indices are recalculated from cleaned physical bands and atomically stored in a
separate `*_final.nc`. A restarted notebook resumes from the raw, cleaned, or final
stage available for each batch. Reuse requires the downloaded cube to contain all
physical bands needed by the configured indices. Changing relevant inputs creates a different cache directory.

Local cleaning checkpoints contain a separate cleaning signature. Changing the
window sequence or interpolation settings reuses the original downloaded cube
but rejects stale `*_cleaned.nc` and `*_final.nc` results and rebuilds them.

Checkpoint NetCDF files store one `float32` variable named `cleaned` and one
`uint8` variable named `observed_mask`; they do not duplicate the raster as a
second observed-value array. In memory, observed values are reconstructed as
`where(observed_mask, cleaned, NaN)`. By default, `keep_cleaned_checkpoint=False`
removes the intermediate `*_cleaned.nc` only after `*_final.nc` is committed.
Set it to `True` to retain the intermediate physical-band raster. The raw cube
and Parquet cleaning audit are never removed by this cleanup. NetCDF compression
is not enabled.
The final Parquet and cleaning-report files are rewritten by `run()`.

Partial downloads use `.partial.nc` files and are renamed only after a successful
download. Do not treat a partial file as a completed cache item.

## Reading log messages

Example:

```text
Tile plan: 56289 parcels in 3 remote jobs; parcels/job min=2,
median=137.0, max=56150; singleton jobs=0.

Starting 3x3 spatial fill for batch 1, variable B04, period 2024-01-01: 428 null pixels.
Finished 3x3 spatial fill for batch 1, variable B04, period 2024-01-01:
filled 391 pixels; 37 null pixels remain.
```

This means three non-empty geographic tiles were created. It does not mean that
the first tile is physically larger. Its representative-point ownership area
simply contains many more parcels. The median of `[56150, 2, 137]` is 137.

If the manager creates at least ten jobs and every job contains exactly one
parcel, it rejects the plan as pathological. Other imbalanced distributions are
logged but accepted.

## Operational guidance

- Start with a small parcel subset and short date range to verify authentication,
  CRS choice, output variables, and backend behavior.
- Keep `openeo_parallel_jobs` within the CDSE account quota. Two is the intended
  general-user setting in this implementation.
- A larger tile reduces the number of remote jobs but increases raster area,
  memory, runtime, and the chance of a dense tile containing many parcels.
- A smaller tile creates more jobs and may improve spatial workload granularity,
  but increases job-management overhead.
- A tile buffer must cover the full geometry of every parcel from its ownership
  point. It increases loaded raster area and does not change ownership or parcel
  counts.
- `batch_workers` and `openeo_parallel_jobs` are different in the manager:
  `batch_workers` controls local processes; `openeo_parallel_jobs` controls remote
  CDSE concurrency.
- The base class sends parcel geometries to openEO for spatial filtering. The
  manager loads rectangular buffered tile extents and applies parcel masks only
  during local statistics.
- Preserve the `monthly_cubes` directory when resuming a run. Removing it forces
  remote processing to be repeated.
- Inspect `tile_jobs.parquet` and `job_manager/` when the manager ends without an
  expected NetCDF file.

## Common validation errors

- **No valid Polygon or MultiPolygon parcels:** check geometry types and empty
  geometries.
- **Duplicate parcel IDs:** make `parcel_id_field` unique before construction.
- **Projected CRS required:** choose a local projected EPSG code for
  `working_epsg`.
- **No sensor output selected:** select at least one S1 band, S2 band, or S2
  index.
- **Indices supplied while calculation is disabled:** set
  a non-empty `sentinel2_indices` list.
- **IDW or kriging distance missing:** provide the corresponding required
  distance parameter.
- **Parcel exceeds buffered tile:** increase `tile_buffer_metres` using the CSV
  diagnostic and `compute_tile_buffer_metres` as guidance.
- **Missing CRS metadata in NetCDF:** the cube cannot safely be aligned to parcel
  geometries; inspect the backend output instead of forcing an assumed CRS.
