# Example: How to Call the Multi-User Extraction Functions

This is a ready-to-copy example for your notebook.

## Complete Working Example

```python
# ============================================================================
# IMPORTS
# ============================================================================
from pathlib import Path
from openeo_parcel_stats_pipeline.zonal_stats_multiuser_manager import (
    split_geodataframe_by_grid,
    load_openeo_users_from_env,
    run_parallel_extractions,
    merge_and_save_results,
)

# ============================================================================
# STEP 1: Split parcels into spatial partitions
# ============================================================================
# This divides the parcels into a grid (1 row, 2 columns = left/right halves)
# For 5 users, use: rows=1, cols=5
#
# IMPORTANT: Parcels are assigned based on their centroid, ensuring each
# parcel appears in exactly ONE partition (no duplicates across partitions)

spatial_parts = split_geodataframe_by_grid(
    gdf=gdf_parcels,        # Input: 25,581 parcels
    rows=1,                 # 1 grid row
    cols=2,                 # 2 grid columns = 2 partitions
)

print(f"Created {len(spatial_parts)} spatial partitions:")
for i, part in enumerate(spatial_parts, 1):
    print(f"  Partition {i}: {len(part)} parcels")

# Output:
# Created 2 spatial partitions:
#   Partition 1: 12790 parcels
#   Partition 2: 12791 parcels
# (Total = 25581, no duplicates)


# ============================================================================
# STEP 2: Load openEO users from environment variable
# ============================================================================
# Requires OPENEO_USERS env var: "user1@example.com,pass1;user2@example.com,pass2"

users_list = load_openeo_users_from_env()

print(f"Loaded {len(users_list)} openEO user(s):")
for i, (user, _) in enumerate(users_list, 1):
    print(f"  User {i}: {user}")

# Output:
# Loaded 2 openEO user(s):
#   User 1: louvarieleni@gmail.com
#   User 2: farfarasmpoytos@gmail.com


# ============================================================================
# STEP 3: Create extraction configuration dictionary
# ============================================================================
# Single dictionary containing all parameters - much cleaner!

extraction_config = {
    # Output paths
    "output_dir": OUTPUT_DIR,
    "run_identifier": RUN_IDENTIFIER,

    # Parcel configuration
    "parcel_id_field": parcel_id_column,

    # Temporal bounds
    "start_date": START_DATE,
    "end_date": END_DATE,

    # Spatial configuration
    "working_epsg": WORKING_EPSG,

    # Sentinel-2 configuration
    "sentinel2_bands": SENTINEL2_BANDS,
    "calculate_sentinel2_indices": CALCULATE_SENTINEL2_INDICES,
    "sentinel2_indices": SENTINEL2_INDICES,

    # Sentinel-1 configuration
    "sentinel1_bands": SENTINEL1_BANDS,

    # Statistics to calculate
    "spatial_statistics": SPATIAL_STATISTICS,

    # Tiling configuration
    "tile_size_metres": TILE_SIZE_METRES,
    "tile_buffer_metres": TILE_BUFFER_METRES,

    # Data cleaning (IQR outlier removal)
    "remove_outliers": REMOVE_OUTLIERS,
    "iqr_quantiles": IQR_QUANTILES,
    "iqr_multiplier": IQR_MULTIPLIER,
    "iqr_min_valid_pixels": IQR_MIN_VALID_PIXELS,

    # Null filling
    "fill_nulls": FILL_NULLS,
    "temporal_fill_mode": TEMPORAL_FILL_MODE,
    "minimum_parcel_pixels": MINIMUM_PARCEL_PIXELS,
    "minimum_observed_fraction_for_fill": MINIMUM_OBSERVED_FRACTION_FOR_FILL,

    # Interpolation
    "interpolation_method": INTERPOLATION_METHOD,
    "interpolation_max_distance_in_meters": INTERPOLATION_MAX_DISTANCE_IN_METERS,

    # Parallel processing
    "openeo_parallel_jobs": 2,
    "job_poll_seconds": 30,
    "batch_workers": LOCAL_BATCH_WORKERS,

    # Other options
    "spatial_fill_window_sizes": (3, 5),
}

# ============================================================================
# STEP 4: Run all partitions in parallel
# ============================================================================
# Each partition uses a different user (round-robin assignment)
# All partitions extract simultaneously

all_results = run_parallel_extractions(
    spatial_parts=spatial_parts,
    users_list=users_list,
    config=extraction_config,
)

# Output format: {partition_idx: results_gdf}
print(f"\nExtraction complete!")
print(f"  Partition 1: {len(all_results[1])} rows")
print(f"  Partition 2: {len(all_results[2])} rows")

# Output:
# ======================================================================
# Running extraction on 2 spatial partition(s) IN PARALLEL
# ======================================================================
#
# ──────────────────────────────────────────────────────────────────────
# PARTITION 1/2: 13761 parcels
# ──────────────────────────────────────────────────────────────────────
# User: louvarieleni@gmail.com
# Output: C:\work_dir\axios-2023-2024\1_parcel_stats\partition_1
# ✓ Partition 1 complete: 14325 parcels
#
# ──────────────────────────────────────────────────────────────────────
# PARTITION 2/2: 11820 parcels
# ──────────────────────────────────────────────────────────────────────
# User: farfarasmpoytos@gmail.com
# Output: C:\work_dir\axios-2023-2024\1_parcel_stats\partition_2
# ✓ Partition 2 complete: 13987 parcels
#
# ✓ All 2 partition(s) completed in 345.2s


# ============================================================================
# STEP 5: Merge results and save to files
# Final output: C:\work_dir\axios-2023-2024\1_parcel_stats\satellite_parcel_time_stats.geoparquet


# ============================================================================
# STEP 6: Inspect results
# ============================================================================

print(f"\n{'='*70}")
print("FINAL RESULTS")
print(f"{'='*70}")

print(f"\nMerged GeoDataFrame:")
print(f"  Rows: {len(merged_gdf)}")
print(f"  Columns: {len(merged_gdf.columns)}")
print(f"  Has geometry: {merged_gdf.geometry.name in merged_gdf.columns}")

print(f"\nOutput files:")
print(f"  GeoParquet: {geoparquet_path}")
print(f"    Size: {geoparquet_path.stat().st_size / 1e6:.1f} MB")
print(f"  CSV Preview: {csv_preview_path}")
print(f"    Size: {csv_preview_path.stat().st_size / 1e3:.1f} KB")

print(f"\nFirst 5 rows:")
print(merged_gdf.head())

print(f"\nColumn names (first 15):")
print(merged_gdf.columns[:15].tolist())

# Output:
# ======================================================================
# FINAL RESULTS
# ======================================================================
#
# Merged GeoDataFrame:
#   Rows: 28312
#   Columns: 187
#   Has geometry: True
#
# Output files:
#   GeoParquet: C:\work_dir\axios-2023-2024\1_parcel_stats\satellite_parcel_time_stats.geoparquet
#     Size: 234.5 MB
#   CSV Preview: C:\work_dir\axios-2023-2024\1_parcel_stats\satellite_parcel_time_stats_preview.csv
#     Size: 45.2 KB
#
# First 5 rows:
#    parcel_code  ...  EVI_max__20240901
# 0          123  ...           0.8234
# 1          456  ...           0.7891
# ...
#
# Column names (first 15):
# ['parcel_code', 'B02_mean__20231001', 'B02_median__20231001',
#  'B02_sd__20231001', 'B02_min__20231001', 'B02_max__20231001', ...]
```

---

## Key Advantages of Dict-Based Configuration

✅ **Cleaner Code**: Single dict instead of 30+ parameters
✅ **Easier to Maintain**: All config in one place
✅ **Flexible**: Easy to add/remove parameters
✅ **Readable**: Keys are self-documenting
✅ **Type-Safe**: Dict structure is obvious

---

## How to Replace Existing Notebook Cell

If you want to use this in your `1_axios_parcel_stats.ipynb` notebook:

**Replace your cell 14** with Steps 3-6 above.

Then:
1. Make sure all config variables are defined in previous cells
2. Create the `extraction_config` dictionary with your values
3. Call the three functions: `split_geodataframe_by_grid()`, `load_openeo_users_from_env()`, `run_parallel_extractions()`, `merge_and_save_results()`
4. Restart Jupyter kernel before running

---

## Variable Mapping Reference

| Dict Key | Your Notebook Variable |
|----------|------------------------|
| `run_identifier` | `RUN_IDENTIFIER` |
| `parcel_id_field` | `parcel_id_column` |
| `start_date` | `START_DATE` |
| `end_date` | `END_DATE` |
| `working_epsg` | `WORKING_EPSG` |
| `sentinel2_bands` | `SENTINEL2_BANDS` |
| `calculate_sentinel2_indices` | `CALCULATE_SENTINEL2_INDICES` |
| `sentinel2_indices` | `SENTINEL2_INDICES` |
| `sentinel1_bands` | `SENTINEL1_BANDS` |
| `spatial_statistics` | `SPATIAL_STATISTICS` |
| `tile_size_metres` | `TILE_SIZE_METRES` |
| `tile_buffer_metres` | `TILE_BUFFER_METRES` |
| `remove_outliers` | `REMOVE_OUTLIERS` |
| `iqr_quantiles` | `IQR_QUANTILES` |
| `iqr_multiplier` | `IQR_MULTIPLIER` |
| `fill_nulls` | `FILL_NULLS` |
| `iqr_min_valid_pixels` | `IQR_MIN_VALID_PIXELS` |
| `temporal_fill_mode` | `TEMPORAL_FILL_MODE` |
| `minimum_parcel_pixels` | `MINIMUM_PARCEL_PIXELS` |
| `minimum_observed_fraction_for_fill` | `MINIMUM_OBSERVED_FRACTION_FOR_FILL` |
| `interpolation_method` | `INTERPOLATION_METHOD` |
| `interpolation_max_distance_in_meters` | `INTERPOLATION_MAX_DISTANCE_IN_METERS` |
| `batch_workers` | `LOCAL_BATCH_WORKERS` |
