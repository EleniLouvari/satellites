# Planet Basemaps parcel statistics

`PlanetBasemapZonalStats` reads monthly eight-band Planet analysis GeoTIFFs from
the HUB `PL_BSM` catalog or a local manifest. It reuses the shared **local** raster cleaning, fill provenance, index formulas, parcel
statistics, feature reduction and GeoParquet schema. It does not submit openEO
jobs. Both imagery workflows share `ParcelStatsBase`.

The historical real-data validation note is not present in this checkout.
The preserved notebook outputs describe earlier runs; this migration uses local
synthetic regression tests and does not rerun remote downloads.

## Temporary delivery-bucket and already-downloaded inputs

For imagery that has not yet been registered in HUB, use
[`0_download_planet_tiles.ipynb`](../../notebooks/projects/volvi/0_download_planet_tiles.ipynb).
It is configured for the nine requested quads and January–September 2025, saves
downloads under `C:/work_dir/planet_local_2025_01_09`, and includes a reproducible
integration test on real Volvi parcels. Set `DOWNLOAD_FILES=True` to download or
resume; the default processes the existing local download.

The read-only delivery adapter in `planet_delivery.py` follows the layout used
by `esa_govermental_hub/src/pipeline/download/download_planet.py`, which is called
by that repository's running `upload_batch_process_planet.ipynb`:

```text
s3://planet-data-axis3/PL_BSM/PL_BSM_<MONTH>_<YEAR>/<order_id>/
  manifest.json
  <mosaic_name>/<quad_id>_metadata.json
  <mosaic_name>/<quad_id>_quad_bandmath.tif
  <mosaic_name>/<quad_id>_ortho_udm2.tif
  <mosaic_name>/<quad_id>_provenance_raster.tif
  <mosaic_name>/<quad_id>_provenance_vector.zip
```

It reads completed order manifests, verifies each downloaded object's size and
SHA256, and retains imagery in month/quad folders. Existing files are reused using
completion metadata; unmarked files are adopted only after checksum verification.
No ingestion classes are instantiated, no assets are uploaded, and no catalog
records are changed. Credentials use the existing `PLANET_AWS_ACCESS_KEY_ID`,
`PLANET_AWS_SECRET_ACCESS_KEY`, `PLANET_S3_REGION` and optional `PLANET_S3_URL`
settings. An optional `.env` file is read without altering it or the process's
environment. The running ingestion notebook is not executed or modified.

Use retained files entirely offline:

```python
# A directory with original Planet metadata sidecars and neighboring TIFFs:
result = extractor.run_local("C:/work_dir/planet_local_2025_01_09")

# Or a CSV with period_start, quad_id and path (relative paths are CSV-relative):
result = extractor.run_local("C:/work_dir/planet_local_2025_01_09/planet_local_manifest.csv")
```

Directory discovery accepts `<quad_id>_quad_bandmath.tif` and
`<quad_id>_quad.tif`. It obtains dates and quad IDs from `mosaic` and `quad` in
the adjacent `<quad_id>_metadata.json`; no catalog-style filenames are required.
For differently named files, supply a manifest CSV or call `run(dataframe)`.
The delivered metadata's `last_acquired` is an **exclusive** bound (for example,
February 1 for the January mosaic). The pipeline's `end_date` remains inclusive.

The sampled 2025 delivery uses ten float32 bands: eight physical bands stored
as scaled DN, NDVI in band 9 and tagged alpha in band 10. Float32 storage does
not mean that reflectance scaling has already been applied. The integration
notebook checks the supplied NDVI against B8/B6 and uses `reflectance_scale=0.0001`.
UDM2/provenance are downloaded and preserved but are not additional inputs to
the current cleaning engine.

## Inputs and spectral assumptions

- Parcels: a GeoDataFrame with a CRS, polygon geometries and unique, non-null,
  nonblank IDs. Original attributes and geometry are retained in the final output.
  If the ID column is absent, the shared planner uses the dataframe index.
- Tile grid for catalog runs: a GeoDataFrame with a CRS and unique `quad_id`
  values. Supply all quads covering the parcels **and their processing buffers**;
  do not filter solely by parcel centroids.
- Dates: complete calendar months, from the first day of the starting month to
  the last day of the ending month, inclusive. A monthly mosaic cannot be clipped
  to acquisitions before a mid-month cutoff, so partial months are rejected.
- Working CRS: a projected EPSG code with metre units, normally a local UTM CRS.
- TIFFs: at least eight bands in the order below, with valid georeferencing.
  The code validates band count and CRS, but cannot establish spectral meaning
  from band count alone. Verify that HUB assets follow this layout.

| Output band | Spectral role |
| --- | --- |
| B1 | Coastal blue |
| B2 | Blue |
| B3 | Green I |
| B4 | Green II |
| B5 | Yellow |
| B6 | Red |
| B7 | Red edge |
| B8 | Near infrared |

This ordering follows [Planet's Basemaps specification](https://assets.planet.com/products/basemap/planet-basemaps-product-specifications.pdf).
Extra TIFF bands are excluded from spectral calculations. A tagged alpha band
is handled as a validity mask by Rasterio; an untagged HUB NDVI band is ignored.
Nodata and masks are respected before mosaicking. There is no additional cloud
classification or UDM2 ingestion in this adapter.

TIFF scale and offset metadata are applied before cleaning. When **all** first
eight scales are 1 and offsets are 0, `reflectance_scale=0.0001` is used instead.
That fallback is an input contract for the HUB analysis assets, not something
the adapter can infer from their pixel values. Set `reflectance_scale=1.0` for
already-scaled reflectance TIFFs without scale metadata. Incorrect scaling
changes EVI and SAVI because their formulas contain additive constants.

Supported indices are EVI, NDRE, NDVI, NDWI and SAVI. They are calculated after
cleaning each physical band, using Planet's spectral roles and the shared index
algebra. NDMI is rejected because these eight bands contain no SWIR channel.
The sensor-specific bands are not spectrally interchangeable with Sentinel-2.

## Run directly from HUB with bounded temporary storage

Install the repository into your existing geospatial environment in editable mode; see the root README.
HUB authentication and asset transport remain in `CatalogSearchUtils` and the
shared catalog/S3 helpers. Configure their existing environment variables; never
put passwords or access tokens in notebook source.

```python
import os
import geopandas as gpd

from satellites.data_preparation.sources.hub.catalog_library import CatalogSearchUtils
from satellites.data_preparation.parcel_stats.planet import PlanetBasemapZonalStats
from satellites.data_preparation.parcel_stats import estimate_utm_epsg_from_parcels

parcels = gpd.read_file("parcels.gpkg")
tiles = gpd.read_file("planet_quad_grid.gpkg")
catalog = CatalogSearchUtils(
    catalog_endpoint=os.environ["CATALOG_URL"], client_id="internal", verbose=False
)
extractor = PlanetBasemapZonalStats(
    parcels=parcels,
    start_date="2024-01-01",
    end_date="2024-12-31",
    output_dir="outputs/planet_2024",
    working_epsg=estimate_utm_epsg_from_parcels(parcels),
    parcel_id_field="parcel_id",
    resolution_metres=5,
    batch_size_metres=2000,
    buffer_metres=100,
    indices=["EVI", "NDRE", "NDVI", "NDWI", "SAVI"],
    remove_outliers=True,
    fill_nulls=True,
    temporal_fill_mode="past_only",
)
result = extractor.run_catalog(catalog, tiles, scratch_dir="scratch/planet")
```

Processing steps:

1. Repair and normalize parcels using the shared planner. Validate the quad grid
   and fail on uncovered parcels before downloading.
2. Assign each parcel once to a containing quad or a set of seam quads. Split
   groups into spatial batches and select all source quads intersecting each
   batch's buffered extent.
3. Resolve one HUB item per quad/month using the identifier substring
   `<english_month>_<year>_<quad_id>_`. Missing or ambiguous results fail explicitly.
4. Download only the single asset matching `*Analysis*.tif` (case insensitive).
   Reuse the resolved item rather than querying it again by ID.
5. Check raster footprints, warp their first eight bands onto the batch grid
   with nearest-neighbor resampling, and convert them to reflectance. First valid
   pixels win in sorted quad/path order, preventing overlap double counting.
6. Discard that month's downloaded files after copying its cropped pixels into
   the in-memory time cube. Repeat for subsequent months.
7. Run shared IQR cleaning, configured filling, index calculation and zonal
   statistics. Footprint coverage does not imply that every pixel is observed;
   nodata inside footprints is handled by this stage and its quality metrics.
8. Commit the batch result and cleaning report, then their JSON completion marker.
   Remove the invocation's temporary TIFFs and NetCDFs. Existing scratch files
   outside its uniquely created temporary directory are preserved.
9. Combine results, attach parcel attributes and geometry, and write the outputs.

## Retain downloads for repeated experiments

```python
from satellites.data_preparation.parcel_stats.planet import download_monthly_tiles

manifest = download_monthly_tiles(
    catalog, tiles, "2024-01-01", "2024-12-31", "downloads/planet",
    analysis_only=True,
)
result = extractor.run(manifest)
```

An existing manifest must contain `period_start` (the first day of the month),
`quad_id` and `path`. Paths may be strings or Path objects. There must be one
entry per quad/month and every retained quad must have all requested months.
Rows outside the requested period are ignored. Optional `item_id` is retained
in the audit CSV.

The downloader defaults to `analysis_only=False` for compatibility with callers
that want every item asset. Prefer `True` for extraction. Completion markers
validate selected asset metadata, download mode, filenames, file sizes and
modification times. A changed or missing file, changed asset metadata, or a
different download mode triggers a refresh. This is not a full content checksum.

## Performance and restart behavior

| Mode | Storage and reuse |
| --- | --- |
| `run_catalog(...)` | Retains only compact batch checkpoints. One batch/month's source TIFFs coexist in scratch; repeated batches or seam groups can download the same quad again. |
| `run(manifest)` | Retains source TIFFs and raw/final NetCDF checkpoints, avoiding repeated network transfers while experimenting with cleaning settings. |

Both modes run batches sequentially (`batch_workers=1`). Batching limits are
based on parcel centroids, so a large parcel can exceed the nominal batch width.
The raw memory guard checks `months × 8 × height × width × 4` bytes before
allocating or downloading a batch. `max_cube_bytes` defaults to 512,000,000;
actual peak RAM is several times larger during cleaning, provenance and index
calculation. Reduce batch size, buffer size or increase pixel resolution when
necessary. Buffer and batch choices can affect IQR and spatial filling results.

Streaming `resume=True` reuses completed batch results **without contacting the
catalog**. Its key includes parcel IDs/geometries, the ID column name, processing
settings, normalized tile geometries and CRS, and catalog endpoint. Use
`resume=False` when source imagery changes under the same catalog identity.
Changing configuration creates a separate checkpoint namespace. Older streaming
checkpoints from before the buffer/CRS fixes are intentionally not reused.

Local raw-cube reuse is based on source paths (including GDAL mask/metadata
sidecars), file sizes/mtimes, periods, grid
and reflectance scale. Cleaning settings have their own shared checkpoint key.
`keep_cleaned_checkpoint=True` optionally retains the physical-band checkpoint
in addition to the final bands-and-indices checkpoint; it defaults to `False`.
Use a dedicated output directory per run and avoid concurrent writers to it.
A failed refresh raises an exception; previously published output tables may
still exist and should not be interpreted as a successful new run.

## Outputs and verification

- `satellite_parcel_time_stats.geoparquet`: the shared wide parcel table, with dated
  features such as `B6_mean__20240101` and `NDVI_median__20240101`.
- `satellite_parcel_annual_stats.geoparquet`: mean/min/max/population standard
  deviation of monthly parcel medians **over the entire requested interval**.
  Despite the legacy filename, multi-year intervals are not split by year.
- `satellite_pixel_cleaning_report.csv`: shared cleaning diagnostics.
- `planet_tile_manifest.csv`: source provenance. Streaming rows indicate
  `local_files_deleted=True`; their historical scratch paths no longer exist.
- `streaming_results/`: restart checkpoints for catalog mode.

Synthetic tests in `tests/data_preparation/test_planet_parcel_stats.py` exercise seams,
buffer neighbors, scaling, masks, missing months, download reuse, configuration,
memory limits and interrupted-run recovery without contacting HUB. Run them in
the geospatial environment with `python -m pytest tests/data_preparation/test_planet_parcel_stats.py`.
Live HUB asset contents and throughput require a separate representative data run.

## Source-neutral naming

New workflows can import `PlanetBasemapZonalStats`, `OpenEOZonalStats` and
`OpenEOJobManagerZonalStats` from `satellites.data_preparation.parcel_stats`. Existing imports
remain valid. Shared optical processing uses native band IDs mapped to spectral
roles, so Planet bands no longer need Sentinel-2 aliases for index algebra.
See [naming and compatibility](README.md).

### Notebook input and scratch paths

In `1_parcel_stats_PL_Hub.ipynb`, `INPUT_MODE="local"` reads the existing
`LOCAL_TILES_PATH` (by default `C:/work_dir/planet_local_2025_01_09`). Source
files are always preserved. Parcels outside the selected local quads are reported
and excluded; actual raster coverage is checked for each month.

`SCRATCH_PATH` applies only to catalog mode. The notebook defaults
`DELETE_TEMPORARY_FILES=False`, retaining newly downloaded files and intermediate
cubes. Set it to `True` to clean only owned scratch subdirectories, including on
exceptions. The library keeps its previous `run_catalog` default of cleanup for
compatibility. Retention does not turn scratch into an automatic download cache.
