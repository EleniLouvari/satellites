# Satellite parcel zonal statistics

`satellite_extract_data` extracts Sentinel-1 and Sentinel-2 observations from the
Copernicus Data Space Ecosystem (CDSE) openEO backend and calculates temporal
statistics for parcel polygons.

The remote openEO jobs create temporally aggregated raster cubes. Pixel cleaning,
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

## Processing flow

1. Validate dates, CRS, sensor selections, statistics, and cleaning options.
2. Repair parcel geometries, retain polygonal parts, validate unique parcel IDs,
   and transform parcels to EPSG:4326 for openEO.
3. Build spatial batches or tiles.
4. Connect and authenticate to `https://openeo.dataspace.copernicus.eu`.
5. Load Sentinel collections for the requested date range.
6. For Sentinel-2, resample reflectance to 10 m, apply the `0.0001` scale factor,
   mask invalid/cloud SCL classes, and calculate requested indices.
7. For Sentinel-1, calculate linear-power `sigma0-ellipsoid` backscatter using the
   `COPERNICUS_30` elevation model and resample to 10 m.
8. Aggregate acquisitions into the configured temporal periods.
9. Download each raster cube as NetCDF and cache it locally.
10. Mask the raster pixels by parcel locally, optionally remove outliers and fill
    missing pixels, then calculate parcel statistics.
11. Add parcel metrics and derived statistics.
12. Write CSV, Parquet, and a pixel-cleaning report.

## Requirements and setup

Use the repository environment in `geospatial_enviroment_python.yaml`. Important
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
and local interpolation. A geographic CRS such as EPSG:4326 is not accepted as
`working_epsg`.

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
| `calculate_sentinel2_indices` | `False` | Must be `True` when indices are requested |
| `sentinel2_indices` | `None` | Any of `NDVI`, `NDWI`, `MNDWI`, `NDMI`, `NBR`, `GNDVI`, `EVI`, `SAVI`, `MSAVI`, `NDRE`, `PSRI`, `CI` |
| `sentinel1_bands` | `None` | Any of `VV`, `VH`; `None` or `[]` disables Sentinel-1 |

At least one Sentinel-2 band, Sentinel-2 index, or Sentinel-1 band must be
selected. Source bands required to calculate an index are loaded automatically
but are not emitted as standalone variables unless selected explicitly.

### Temporal and statistic parameters

| Parameter | Default | Accepted values and behavior |
|---|---:|---|
| `temporal_period` | `"1M"` | Positive day, month, or year interval such as `15D`, `1M`, `2M`, `3M`, `1Y` |
| `temporal_reducer` | `"median"` | `mean`, `median`, `min`, `max`, or `sum` |
| `spatial_statistics` | `None` | `mean`, `median`, `sd`, `min`, `max`, `count`, `p10`, `p25`, `p75`, `p90`; `mean` is always added |
| `batch_workers` | `1` | Positive integer controlling base-class remote job waves and local statistics processes; for the manager it controls local statistics only |

Temporal intervals start at `start_date`, are half-open internally, and the final
interval is clipped after the inclusive `end_date`. For example, `1M` means
one-month steps from the configured start date, not necessarily calendar months.

### Outlier removal and null filling

| Parameter | Default | Meaning |
|---|---:|---|
| `remove_outliers` | `False` | Replace pixel values outside parcel/variable/period IQR bounds with nulls |
| `iqr_quantiles` | `(0.25, 0.75)` | Lower and upper quantiles; must satisfy `0 <= lower < upper <= 1` |
| `iqr_multiplier` | `1.5` | Non-negative multiplier used to extend the IQR bounds |
| `iqr_min_valid_pixels` | `20` | Minimum valid pixels required before applying IQR filtering |
| `fill_nulls` | `False` | Enable temporal, 3x3 spatial-neighbor, and final interpolation filling |
| `temporal_fill_mode` | `"past_only"` | `past_only` carries the nearest earlier value; `bidirectional` uses closest earlier/later values |
| `minimum_parcel_pixels` | `3` | Threshold for the output quality flag; small parcels are retained |
| `minimum_observed_fraction_for_fill` | `0.20` | Minimum observed fraction for a parcel-variable-period to be filled or used as a temporal source |
| `interpolation_method` | `"nearest"` | Final interpolation method: `nearest`, `idw`, or `kriging` |
| `interpolation_max_distance_in_meters` | `None` | Required positive search distance in metres for `idw` |
| `interpolation_variogram_lags` | `15` | Positive kriging variogram lag count |
| `interpolation_variogram_max_distance_in_meters` | `None` | Required positive distance in metres for `kriging` |

Cleaning is performed independently for every parcel, output variable, and
temporal period. `fill_nulls=True` does not guarantee every null can be filled;
the observed-fraction threshold, available neighbors, and distance settings still
apply.

## Basic usage

### Sentinel-1-only extraction

```python
from satellite_extract_data import SatelliteZonalStats

extractor = SatelliteZonalStats(
    parcels=parcels,
    start_date="2023-01-01",
    end_date="2024-12-31",
    output_dir=r"C:\work_dir\kozani_s1",
    working_epsg=32634,
    parcel_id_field="parcel_id",
    sentinel2_bands=[],
    calculate_sentinel2_indices=False,
    sentinel2_indices=[],
    sentinel1_bands=["VV", "VH"],
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
from satellite_extract_data import SatelliteZonalStats

extractor = SatelliteZonalStats(
    parcels=parcels,
    start_date="2024-03-01",
    end_date="2024-10-31",
    output_dir=r"C:\work_dir\kozani_s2",
    working_epsg=32634,
    sentinel2_bands=["B02", "B03", "B04", "B08"],
    calculate_sentinel2_indices=True,
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
| `tile_size_metres` | `50000` | Positive core tile width/height in the projected working CRS |
| `tile_buffer_metres` | `500` | Non-negative buffer around every remote raster extent |
| `openeo_parallel_jobs` | `2` | Maximum active CDSE jobs managed remotely |
| `job_poll_seconds` | `30` | Positive number of seconds between status polls |
| `run_identifier` | generated timestamp | Readable label, at most 80 characters, included in openEO job titles |

### How parcel ownership works

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
from satellite_extract_data.zonal_stats_job_manager import (
    compute_tile_buffer_metres,
    compute_tile_width,
)

# compute_tile_width currently expects coordinates already expressed in the
# supplied projected CRS.
metric_parcels = parcels.to_crs(epsg=32634)

tile_size = compute_tile_width(
    metric_parcels,
    parcel_id_col="parcel_id",
    working_epsg=32634,
)
tile_buffer = compute_tile_buffer_metres(
    parcels,
    working_epsg=32634,
)
```

`compute_tile_width` tests 5, 10, 20, 30, 40, 50, and 60 km candidates and
returns the smallest size for which every parcel lies completely within a core
tile. This helper optimizes containment, **not** equal job workload. Always pass a
GeoDataFrame reprojected to `working_epsg`, as shown above.

`compute_tile_buffer_metres` measures the largest parcel radius from its
representative point to its bounding-box corners, multiplies it by a safety
factor of 2.5, and rounds upward to 50 m by default. It accepts parcels in any
defined CRS because it reprojects internally.

If a parcel is not completely covered by its assigned buffered tile, planning
stops and writes `uncovered_buffered_tile_parcels.csv` inside the cube cache.
Increase `tile_buffer_metres` and retry.

### Complete manager example

```python
from satellite_extract_data import JobManagerSatelliteZonalStats
from satellite_extract_data.zonal_stats_job_manager import (
    compute_tile_buffer_metres,
    compute_tile_width,
)

working_epsg = 32634
metric_parcels = parcels.to_crs(epsg=working_epsg)
tile_size = compute_tile_width(metric_parcels, "parcel_id", working_epsg)
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
| `satellite_temporal_parcel_statistics.csv` | Final tabular results; geometry is serialized by pandas |
| `satellite_temporal_parcel_statistics.parquet` | Final GeoDataFrame with geometry and CRS metadata |
| `satellite_pixel_cleaning_report.csv` | Per-parcel, period, and variable cleaning counts |
| `satellite_zonal_stats.log` | Progress and configuration log |
| `monthly_cubes/<signature>/batch_XXXXX_monthly.nc` | Cached temporal raster cubes |
| `monthly_cubes/<signature>/tile_jobs.parquet` | Manager job database, manager implementation only |
| `monthly_cubes/<signature>/job_manager/` | Manager status/error artifacts, manager implementation only |

The final GeoDataFrame has one row per parcel and temporal period. Core columns
include:

- parcel ID, `period_start`, and exclusive `period_end`;
- `eligible_pixel_count` and `meets_minimum_pixel_count`;
- requested columns such as `VV_mean`, `NDVI_median`, or `B04_p90`;
- `parcel_area_m2` and `approx_pixel_count`;
- derived fields when their source statistics exist: range, variance,
  coefficient of variation, IQR, Bowley skewness, p90-p10 spread, and valid-pixel
  fractions;
- parcel geometry in EPSG:4326;
- `batch_number`.

The cache signature includes dates, processing configuration, parcel IDs, and
parcel geometries. The manager also includes tile size and buffer. An identical
run reuses completed NetCDF files. Changing relevant inputs creates a different
cache directory. Final CSV/Parquet/report files are rewritten by `run()`.

Partial downloads use `.partial.nc` files and are renamed only after a successful
download. Do not treat a partial file as a completed cache item.

## Reading log messages

Example:

```text
Tile plan: 56289 parcels in 3 remote jobs; parcels/job min=2,
median=137.0, max=56150; singleton jobs=0.
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
  `calculate_sentinel2_indices=True`.
- **IDW or kriging distance missing:** provide the corresponding required
  distance parameter.
- **Parcel exceeds buffered tile:** increase `tile_buffer_metres` using the CSV
  diagnostic and `compute_tile_buffer_metres` as guidance.
- **Missing CRS metadata in NetCDF:** the cube cannot safely be aligned to parcel
  geometries; inspect the backend output instead of forcing an assumed CRS.
