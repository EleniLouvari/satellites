# HUB Sentinel ARD parcel statistics

The HUB adapter reads physical bands from the `ARD_S2` and `ARD_S1` STAC
collections, constructs local raster cubes, and uses the existing parcel
cleaning, index, aggregation, and GeoParquet output pipeline. ARD production
remains in the separate ingestion repository. Use it from Python, the example
notebook, or the [Parcel Studio source selector](../../web_app/README.md#hub-sentinel-ard).

## Start with the example notebook

Use [1_parcel_stats_hub_ard.ipynb](../../../notebooks/projects/neuro/1_parcel_stats_hub_ard.ipynb).
The notebook starts with five parcels, a short date interval, Sentinel-2 B04/B08,
and NDVI. Change `SENSOR_MODE` to select Sentinel-1 VV/VH with R/RVI or both
sensors. Each sensor configuration has a separate output directory.

Choose a kernel with this repository installed in editable mode and the
`acquisition` and `notebooks` extras available. Import from the installed
`data_preparation` package; do not modify `sys.path`.

Set these variables in the environment or the repository `.env`:

| Variable | Purpose |
| --- | --- |
| `CATALOG_URL` | HUB STAC API base URL. |
| `KEYCLOAK_URL` | Keycloak realm URL used to acquire a client-credentials token. |
| `CATALOG_CLIENT_ID` | Configured catalog client role. |
| `CATALOG_CLIENT_SECRET` | Secret for the configured client. |

Load `.env` **before importing catalog modules**, because the existing catalog
constants read credentials at import time. Do not print tokens or signed asset
URLs. The notebook preserves existing environment values when loading `.env`.
Its initial parcel path follows the existing Neuro `WORK_PATH` convention; edit
the path and date interval to match your data and the catalog's coverage.

## Python usage

Given a parcel GeoDataFrame with a CRS:

```python
from dotenv import load_dotenv

load_dotenv(".env", override=False)

from data_preparation.parcel_stats import HubZonalStats

pipeline = HubZonalStats(
    parcels=parcels,
    start_date="2025-07-01",
    end_date="2025-07-31",
    output_dir="outputs/my_project/hub_s2_test",
    working_epsg=2100,
    sentinel2_bands=["B04", "B08"],
    sentinel2_indices=["NDVI"],
    sentinel1_bands=[],
    sentinel1_indices=[],
    spatial_statistics=["mean", "median"],
    temporal_reducer="none",
    fill_nulls=False,
    batch_workers=1,
    hub_read_workers=8,
    resume_completed_partitions=True,
)
results = pipeline.run()
```

Pass `hub_catalog_url` and `hub_client_id` to override the corresponding
environment settings. `working_epsg` defaults to **2100 (Greek Grid)** when
omitted; an explicit projected CRS overrides parcel batching/output geometry,
while HUB raster acquisition remains in EPSG:2100.
`hub_read_workers` controls parallel asset-window reads;
`batch_workers` controls concurrent local statistics processes. Each complete
batch cube enters local processing while acquisition continues for later
batches, including with `batch_workers=1`. Separate processes isolate NetCDF
access. At most `batch_workers` local jobs are submitted, with one additional
acquired cube allowed to wait for a worker slot; acquisition pauses when that
limit is reached. `hub_block_size`
defaults to 512 pixels and must be at least 64. NetCDF writes remain on the
acquisition thread. Construction validates configuration without contacting the
catalog; `run()` performs discovery and extraction.

## Source contract and access

The November 2026 read-only probe verified the following for one item from each
collection. These samples do not establish date or spatial coverage for every
project.

- Both collections advertise HTTPS COG assets. The asset gateway accepts a
  bearer token and redirects to storage. Storage supports HTTP range requests,
  permitting window reads without downloading full scene files. Bearer tokens
  belong on gateway requests; they are not forwarded to storage.
- The inspected physical-band rasters are float32, with identity scale/offset
  and nodata equal to float32 maximum. Reflectance is already physical: applying
  the openEO reflectance scale factor again would give incorrect values.
- Inspected rasters have EPSG:2100 coordinates and 10 m pixels. Origins are not
  snapped to common 10 m multiples; matching CRS/resolution alone is insufficient
  for alignment between items. The adapter must normalize the spatial grid.
- The inspected Sentinel-2 SCL is uint8 with nodata 255. MASK is uint8 with nodata
  0; ingestion source code uses 1 for valid and 2 for invalid pixels. Sentinel-2
  cloud/shadow masking and Sentinel-1 layover/shadow masking require different
  source interpretation before shared local processing.
- HUB Sentinel-1 assets named `VV` and `VH` originate from gamma0 exports.
  openEO uses linear sigma0 over the ellipsoid. Keep calibration and source
  provenance with outputs; equivalent names do not imply equivalent features.

Requested indices use the necessary physical source bands, even when those
bands are not requested as standalone outputs. The shared engine calculates
indices after local cleaning, preserving observation/fill diagnostics.

## Pixel and mosaic rules

- Sentinel-2 excludes SCL classes 0, 1, 3, 8, 9, 10, 11 and SCL nodata 255.
  Classes 3, 8, 9, 10 and 11 receive a 3 × 3 cloud/shadow buffer. ARD `MASK==0`
  is an additional nodata guard.
- Sentinel-1 keeps only `MASK==1` and positive backscatter values. Zero and
  negative values, including noise-removal artifacts, are invalid.
- Every physical band required for a sensor must be valid at a pixel before it
  contributes to a mosaic, so an index cannot combine bands from different
  scenes at that pixel.
- S2 items are grouped by datatake; S1 items by platform and absolute orbit.
  Without that acquisition identity, items stay separate. Within a group, the
  first valid whole pixel wins, ordered by lowest nodata fraction, lowest cloud
  cover, newest processing time, and item ID.
- The 10 m grid follows the dominant S2 asset lattice, then S1 when no S2 lattice
  is available, with a snapped EPSG:2100 grid as the fallback. Aligned pixels are
  copied; other assets are masked before NaN-aware bilinear reprojection.

## Outputs and resume

The shared result assembler produces:

- `satellite_parcel_time_stats.geoparquet`: one row per parcel with dated
  features and retained parcel attributes/geometry.
- `satellite_parcel_annual_stats.geoparquet`: summaries of temporal medians.
- `satellite_pixel_cleaning_report.csv`: local processing diagnostics.
- `satellite_partition_completion.json`: completion state and processing
  identity, including HUB inventory and calibration provenance.
- `hub_item_manifest.parquet`: the refreshed item/asset inventory, including
  stable query-free asset locations, without signed queries or access tokens.

Source-specific cubes, another inventory copy, and the acquisition coverage CSV
are under `monthly_cubes/<run_signature>/`. The coverage CSV describes cubes
built during extraction; the shared cleaning report describes local processing.
They serve different purposes.

If a requested sensor has no usable observations for a batch, discovery raises
a `ValueError` identifying the batch and collection. Check catalog coverage,
date range, required assets, and cloud/orbit filters; selecting both sensors
requires eligible imagery from both. A monthly interval can still have fewer
observations or missing pixels than another interval; inspect the coverage CSV
and cleaning report together.

Sentinel-2 cloud-cover filtering is applied in the catalog database using the
configured `sentinel2_max_scene_cloud_cover`; items without `eo:cloud_cover` are
excluded by that comparison. Orbit direction is filtered after discovery,
keeping the database query restricted to supported queryable properties.

With unlimited nearest interpolation, acquisition reads every block in the
batch extent. A finite interpolation-distance limit permits block selection
around parcel context. Start with filling disabled, as in the example notebook,
and inspect the spatial scope before enabling it for a larger batch.

A completed partition is reusable only with compatible processing settings,
matching parcel identity, and complete required outputs. HUB inventory must be
refreshed before checking whether outputs can be reused, so newly discovered or
reprocessed items cannot be silently skipped. Expiring storage URLs are not
durable source identities.

HUB requires a valid completion/provenance marker. Changing the spatial-fill
windows or kriging search distance invalidates the raw cube when more context
may be needed. Changing parcel attributes such as labels invalidates final
tables while retaining compatible batch statistics, so corrected metadata can
be republished without downloading imagery again. A replacement run is marked
as processing before its new inventory is published or acquisition starts;
acquisition failure therefore cannot leave it marked complete.

Keep source inputs outside the configured output directory. Keep NetCDFs on the
first small test to inspect intermediate rasters; once results are validated,
the existing `remove_nc_after_completion` option can reduce storage use. Batch
statistics and cleaning-report checkpoints must both exist before a cleaned-up
batch is reusable. A `.partial.nc` is never a completed cube.

Local workers save each batch's statistics and cleaning report before deleting
its NetCDFs, so cleanup can happen while later cubes are still being acquired.
Final partition tables are assembled after all acquisition and local work
succeeds. On failure, completed batch checkpoints remain available for resume;
unstarted local jobs are cancelled and running jobs finish their checkpoints
before the error is returned. Worker filling and statistics messages are
forwarded to the web run events and written to
`satellite_raster_filling.worker-<pid>.log` and
`satellite_parcel_statistics.worker-<pid>.log`, even with one local worker.

The openEO multi-account scheduler `run_parallel_extractions` is not supported
for HUB in this version. Use `HubZonalStats.run()`; its `hub_read_workers` and
`batch_workers` settings control acquisition reads and local statistics.

See [NetCDF processing](../NETCDF_PROCESSING.md) for local cleaning order,
checkpoint files, and the distinction between acquired rasters and completed
parcel statistics. The four source-neutral log filenames are documented in the
[parcel-statistics API guide](../README.md#logs-and-persisted-filenames).

## Validation

Start with a small nearby parcel set and confirm the inventory covers the
chosen dates. Inspect missingness, pixel counts, source provenance, and cleaning
diagnostics before expanding the run. Compare HUB and openEO outputs with their
different acquisition, geometry, cloud-mask, and radar-calibration choices in
mind. Saved notebook outputs describe only their configured parcel set and dates;
they make no general accuracy or catalog-coverage claim about other parcels.
