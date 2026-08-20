# Multi-User Spatial Partition Extraction - Usage Example

This module enables parallel openEO extractions by splitting large parcel datasets spatially and distributing them across multiple user accounts to bypass per-user concurrent job limits (2 jobs max on CDSE).

## Environment Setup

Set the `OPENEO_USERS` environment variable with credentials in format:
```
username1,password1;username2,password2;username3,password3
```

Example:
```bash
export OPENEO_USERS="louvarieleni@gmail.com,mypass1;farfarasmpoytos@gmail.com,mypass2;esagovhub1@gmail.com,mypass3"
```

## Complete Notebook Example

```python
from pathlib import Path
from openeo_parcel_stats_pipeline.zonal_stats_multiuser_manager import (
    split_geodataframe_by_grid,
    load_openeo_users_from_env,
    run_parallel_extractions,
    merge_and_save_results,
)
from common_libraries.io_library import read_data
import common_libraries.geom_library as geom_l

# ============================================================================
# STEP 1: Load and prepare input parcels
# ============================================================================
WORK_PATH = Path("C:/work_dir/axios-2023-2024")
PARCELS_PATH = WORK_PATH / "parcels_axios.gpkg"
WORKING_EPSG = 32634

gdf_parcels = read_data(str(PARCELS_PATH))
gdf_parcels = geom_l.convert_multipolygons_to_polygons(gdf_parcels)
gdf_parcels = gdf_parcels.to_crs(WORKING_EPSG)

print(f"Loaded {len(gdf_parcels)} parcels")

# ============================================================================
# STEP 2: Split parcels spatially
# ============================================================================
# Split into 2 parts (1 row, 2 columns)
# Adjust rows/cols based on number of users: if 5 users, use rows=1, cols=5
spatial_parts = split_geodataframe_by_grid(gdf_parcels, rows=1, cols=2)
print(f"Split into {len(spatial_parts)} partitions")

# ============================================================================
# STEP 3: Load openEO users from environment
# ============================================================================
users_list = load_openeo_users_from_env()
print(f"Loaded {len(users_list)} openEO user(s)")

# ============================================================================
# STEP 4: Configure extraction parameters
# ============================================================================
RUN_IDENTIFIER = "axios-2023-2024"
OUTPUT_DIR = WORK_PATH / "1_parcel_stats"
START_DATE = "2023-10-01"
END_DATE = "2024-09-30"

extraction_config = dict(
    # Spatial and temporal
    working_epsg=WORKING_EPSG,
    start_date=START_DATE,
    end_date=END_DATE,

    # Sentinel-2
    sentinel2_bands=["B02", "B03", "B04", "B05", "B08", "B11", "B12"],
    calculate_sentinel2_indices=True,
    sentinel2_indices=["EVI", "NDMI", "NDRE", "NDVI", "NDWI", "SAVI"],

    # Sentinel-1
    sentinel1_bands=["VV", "VH"],

    # Statistics
    spatial_statistics=["mean", "median", "sd", "min", "max", "p10", "p25", "p75", "p90"],

    # Tiling
    tile_size_metres=20_000,
    tile_buffer_metres=2_100,

    # Data cleaning
    remove_outliers=True,
    iqr_quantiles=(0.1, 0.9),
    iqr_multiplier=1.5,
    fill_nulls=True,
    iqr_min_valid_pixels=20,
    temporal_fill_mode="bidirectional",
    minimum_parcel_pixels=3,
    minimum_observed_fraction_for_fill=0.20,

    # Interpolation
    interpolation_method="nearest",
    interpolation_max_distance_in_meters=500,

    # Processing
    openeo_parallel_jobs=2,
    job_poll_seconds=30,
    batch_workers=1,
)

# ============================================================================
# STEP 5: Run parallel extractions (MAIN COMPUTATION)
# ============================================================================
print(f"\n{'='*70}")
print("Starting parallel extraction across all partitions...")
print(f"{'='*70}\n")

all_results = run_parallel_extractions(
    spatial_parts=spatial_parts,
    users_list=users_list,
    output_dir=OUTPUT_DIR,
    run_identifier=RUN_IDENTIFIER,
    parcel_id_field="parcel_code",
    **extraction_config,
)

# ============================================================================
# STEP 6: Merge results and save
# ============================================================================
merged_gdf, geoparquet_path, csv_preview_path = merge_and_save_results(
    gdf_parcels=gdf_parcels,
    all_results=all_results,
    output_dir=OUTPUT_DIR,
    parcel_id_column="parcel_code",
)

print(f"\n{'='*70}")
print("SUCCESS: Extraction complete!")
print(f"{'='*70}")
print(f"GeoParquet: {geoparquet_path}")
print(f"CSV Preview: {csv_preview_path}")

# Display sample results
print("\nSample results (first 5 rows):")
print(merged_gdf.head())
```

## How It Works

1. **Spatial Splitting**: Divides parcels into N equal-area grid cells (e.g., left/right halves)
2. **User Assignment**: Each partition gets a different user (round-robin)
3. **Parallel Execution**: All partitions extract simultaneously using ThreadPoolExecutor
4. **Job Tracking**: Each connection only tracks its own jobs (2 max per user on CDSE)
5. **Result Merging**: Combines all partition outputs into a single GeoParquet
6. **CSV Export**: Saves a preview with parcel IDs + key temporal features

## Performance Tips

- **Partitions ≈ Users**: Best performance when `num_partitions ≈ num_users`
  - 2 users → 1×2 grid (2 partitions)
  - 5 users → 1×5 grid (5 partitions)

- **Per-User Parallelism**: Each user runs 2 concurrent openEO jobs (CDSE limit)
  - Total parallelism = `num_users × 2`
  - 5 users = 10 concurrent jobs maximum

- **Output Structure**: Each partition creates its own subdirectory:
  ```
  OUTPUT_DIR/
    partition_1/
      monthly_cubes/
      satellite_openeo.log
      satellite_parcel_time_stats.geoparquet
    partition_2/
      ...
    satellite_parcel_time_stats.geoparquet  (merged)
    satellite_parcel_time_stats_preview.csv (merged)
  ```

## Troubleshooting

**Error: OPENEO_USERS environment variable not set**
```bash
export OPENEO_USERS="user1@example.com,pass1;user2@example.com,pass2"
# Then restart notebook kernel
```

**Error: Authentication failed for user X**
- Verify credentials are correct
- Check CDSE API status: https://openeo.dataspace.copernicus.eu

**Uneven parcel distribution**
- Parcels are split by geographic bounds, not parcel count
- Adjust grid dimensions if needed (rows/cols parameters)

**Job limits exceeded**
- Verify `openeo_parallel_jobs=2` (CDSE per-user limit)
- Don't increase this value; use more users instead

## Function Reference

### `load_openeo_users_from_env() → list[tuple[str, str]]`
Parse and validate openEO credentials from environment variable.

### `split_geodataframe_by_grid(gdf, rows=1, cols=2) → list[GeoDataFrame]`
Split parcels into spatial grid cells.

### `run_parallel_extractions(...) → dict[int, GeoDataFrame]`
Execute all partitions in parallel, each with assigned user.

### `merge_and_save_results(...) → tuple[GeoDataFrame, Path, Path]`
Combine partition results and save as GeoParquet + CSV preview.
