# Calling the multi-user extraction workflow

This snippet assumes the notebook variables such as `gdf_parcels`,
`parcel_id_column`, dates, bands, indices, statistics, and output paths have
already been defined.

```python
from satellites.data_preparation.parcel_stats.job_manager import (
    compute_tile_buffer_metres,
    compute_tile_width,
)
from satellites.data_preparation.parcel_stats.multiuser import (
    load_openeo_users_from_env,
    merge_and_save_results,
    run_parallel_extractions,
    split_geodataframe_by_grid,
)

OPENEO_JOBS_PER_USER = 2

# Load users first so the grid has one initial partition per account.
users_list = load_openeo_users_from_env()
spatial_parts = split_geodataframe_by_grid(
    gdf_parcels,
    partition_count=len(users_list),
)

# Select the largest tested tile width that still creates at least two
# non-empty tile jobs in every spatial partition.
TILE_SIZE_METRES = compute_tile_width(
    gdf_parcels,
    parcel_id_col=parcel_id_column,
    working_epsg=WORKING_EPSG,
    spatial_parts=spatial_parts,
    user_count=len(users_list),
    jobs_per_user=OPENEO_JOBS_PER_USER,
)
TILE_BUFFER_METRES = compute_tile_buffer_metres(
    gdf_parcels,
    working_epsg=WORKING_EPSG,
)

extraction_config = {
    "output_dir": OUTPUT_DIR,
    "run_identifier": RUN_IDENTIFIER,
    "parcel_id_field": parcel_id_column,
    "start_date": START_DATE,
    "end_date": END_DATE,
    "working_epsg": WORKING_EPSG,
    "sentinel2_bands": SENTINEL2_BANDS,
    "sentinel2_indices": SENTINEL2_INDICES,
    "sentinel1_bands": SENTINEL1_BANDS,
    "sentinel1_indices": SENTINEL1_INDICES,
    "sentinel1_orbit_direction": SENTINEL1_ORBIT_DIRECTION,
    "spatial_statistics": SPATIAL_STATISTICS,
    "tile_size_metres": TILE_SIZE_METRES,
    "tile_buffer_metres": TILE_BUFFER_METRES,
    "openeo_parallel_jobs": OPENEO_JOBS_PER_USER,
    "job_poll_seconds": 30,
    "max_job_retries": 3,
    "job_retry_delay_seconds": 60,
    "batch_workers": LOCAL_BATCH_WORKERS,
    "remove_outliers": REMOVE_OUTLIERS,
    "iqr_quantiles": IQR_QUANTILES,
    "iqr_multiplier": IQR_MULTIPLIER,
    "iqr_min_valid_pixels": IQR_MIN_VALID_PIXELS,
    "fill_nulls": FILL_NULLS,
    "temporal_fill_mode": TEMPORAL_FILL_MODE,
    "minimum_parcel_pixels": MINIMUM_PARCEL_PIXELS,
    "minimum_observed_fraction_for_fill": MINIMUM_OBSERVED_FRACTION_FOR_FILL,
    "interpolation_method": INTERPOLATION_METHOD,
    "interpolation_max_distance_in_meters": INTERPOLATION_MAX_DISTANCE_IN_METERS,
    "spatial_fill_window_sizes": (3, 5),
}

# User queues run concurrently. A user processes its partitions sequentially,
# while its job manager keeps up to two openEO jobs active.
run_parallel_extractions(
    spatial_parts=spatial_parts,
    users_list=users_list,
    config=extraction_config,
)

# The extraction step persists each partition. Merge discovers those files, so
# it can also be run later or in a fresh Python session.
merged_gdf, geoparquet_path = merge_and_save_results(
    gdf_parcels=gdf_parcels,
    output_dir=OUTPUT_DIR,
    parcel_id_column=parcel_id_column,
    working_epsg=WORKING_EPSG,
)
```

With `N` unique users, the intended remote capacity is `2 × N`. Duplicate
usernames are rejected and `openeo_parallel_jobs` cannot exceed 2; add accounts
instead of increasing the per-user setting.
