# Multi-User Parallel Extraction Module - Implementation Summary

## File: `zonal_stats_multiuser_manager.py`

This module provides a complete toolkit for spatial partitioning and parallel openEO extractions using multiple user accounts.

### Functions Completed

#### 1. `load_openeo_users_from_env() → list[tuple[str, str]]`
**Purpose**: Parse openEO credentials from `OPENEO_USERS` environment variable

**Input**:
- Environment variable `OPENEO_USERS` in format: `user1,pass1;user2,pass2;user3,pass3`

**Output**:
- List of (username, password) tuples

**Example**:
```python
users_list = load_openeo_users_from_env()
# Returns: [('user1@example.com', 'pass1'), ('user2@example.com', 'pass2')]
```

---

#### 2. `split_geodataframe_by_grid(gdf, rows=1, cols=2) → list[GeoDataFrame]`
**Purpose**: Split parcels spatially into N equal-area grid cells

**Parameters**:
- `gdf`: Input GeoDataFrame with parcel geometries
- `rows`: Number of grid rows (default: 1)
- `cols`: Number of grid columns (default: 2)

**Output**:
- List of GeoDataFrames, one per grid cell

**Example**:
```python
# Split 25,581 parcels into 2 equal parts (left/right)
spatial_parts = split_geodataframe_by_grid(gdf_parcels, rows=1, cols=2)
# Returns: [gdf_part1 (~13,761 parcels), gdf_part2 (~11,820 parcels)]

# For 5 users, use 1 row × 5 columns
spatial_parts = split_geodataframe_by_grid(gdf_parcels, rows=1, cols=5)
```

---

#### 3. `run_partition_extractor(...) → tuple[int, GeoDataFrame]`
**Purpose**: Extract one spatial partition using assigned user credentials

**Parameters**:
- `part_idx`: Partition number (1-based)
- `gdf_part`: Parcels for this partition
- `users_list`: List of (username, password) tuples
- `total_partitions`: Total partition count (for display)
- `output_dir`: Base output directory
- `run_identifier`: Run name (e.g., "axios-2023-2024")
- `parcel_id_field`: Column name for parcel IDs
- `start_date`, `end_date`: ISO date strings
- `working_epsg`: Projected CRS code
- `sentinel2_bands`, `sentinel2_indices`: S2 configuration
- `calculate_sentinel2_indices`: Boolean
- `sentinel1_bands`: S1 polarizations
- `spatial_statistics`: List of reducers
- `tile_size_metres`, `tile_buffer_metres`: Tiling config
- `**kwargs`: Additional options (IQR, interpolation, fill settings, etc.)

**Output**:
- Tuple: (partition_index, results_gdf)

**Internal Logic**:
- Selects user via round-robin: `user_idx = (part_idx - 1) % len(users_list)`
- Creates `partition_X/` subdirectory
- Instantiates `JobManagerSatelliteZonalStats` with single user credentials
- Calls `.run()` to extract and process
- Returns results DataFrame

**Note**: This function is called internally by `run_parallel_extractions()`. It's not typically called directly from notebooks.

---

#### 4. `run_parallel_extractions(...) → dict[int, GeoDataFrame]`
**Purpose**: Execute all partitions simultaneously using ThreadPoolExecutor

**Parameters**: Same as `run_partition_extractor()`, but for all partitions

**Output**:
- Dictionary: `{1: gdf_results_part1, 2: gdf_results_part2, ...}`

**Internal Logic**:
- Creates ThreadPoolExecutor with `max_workers=len(spatial_parts)`
- Submits all partitions to run concurrently
- Collects results as they complete (not in order)
- Returns sorted results dict
- Displays total elapsed time

**Example**:
```python
all_results = run_parallel_extractions(
    spatial_parts=spatial_parts,
    users_list=users_list,
    output_dir=Path("C:/work_dir/output"),
    run_identifier="axios-2023-2024",
    parcel_id_field="parcel_code",
    start_date="2023-10-01",
    end_date="2024-09-30",
    working_epsg=32634,
    sentinel2_bands=["B02", "B03", "B04", "B05", "B08", "B11", "B12"],
    calculate_sentinel2_indices=True,
    sentinel2_indices=["EVI", "NDMI", "NDRE", "NDVI", "NDWI", "SAVI"],
    sentinel1_bands=["VV", "VH"],
    spatial_statistics=["mean", "median", "sd", "min", "max", "p10", "p25", "p75", "p90"],
    tile_size_metres=20_000,
    tile_buffer_metres=2_100,
    openeo_parallel_jobs=2,
    remove_outliers=True,
    iqr_quantiles=(0.1, 0.9),
    iqr_multiplier=1.5,
    fill_nulls=True,
    iqr_min_valid_pixels=20,
    temporal_fill_mode="bidirectional",
    minimum_parcel_pixels=3,
    minimum_observed_fraction_for_fill=0.20,
    interpolation_method="nearest",
    interpolation_max_distance_in_meters=500,
    batch_workers=1,
)
# Returns: {1: gdf_results_1, 2: gdf_results_2}
```

---

#### 5. `merge_and_save_results(...) → tuple[GeoDataFrame, Path, Path]`
**Purpose**: Combine partition results and save to disk

**Parameters**:
- `gdf_parcels`: Original input GeoDataFrame (for reference counts)
- `all_results`: Results dictionary from `run_parallel_extractions()`
- `output_dir`: Output directory
- `parcel_id_column`: Parcel ID column name

**Output**:
- Tuple: (merged_gdf, geoparquet_path, csv_preview_path)

**Files Created**:
1. `satellite_parcel_time_stats.geoparquet` - Full merged results with geometry and all features
2. `satellite_parcel_time_stats_preview.csv` - First 100 rows with parcel ID + 10 median feature columns

**Example**:
```python
merged_gdf, geoparquet_path, csv_preview_path = merge_and_save_results(
    gdf_parcels=gdf_parcels,
    all_results=all_results,
    output_dir=Path("C:/work_dir/output"),
    parcel_id_column="parcel_code",
)

print(f"Saved: {geoparquet_path}")
print(f"Preview: {csv_preview_path}")
print(merged_gdf.head())
```

---

## Complete Notebook Workflow

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

# 1. Load parcels
gdf_parcels = read_data("C:/work_dir/axios-2023-2024/parcels_axios.gpkg")
gdf_parcels = geom_l.convert_multipolygons_to_polygons(gdf_parcels)
gdf_parcels = gdf_parcels.to_crs(32634)

# 2. Split spatially
spatial_parts = split_geodataframe_by_grid(gdf_parcels, rows=1, cols=2)

# 3. Load users from environment
users_list = load_openeo_users_from_env()

# 4. Run parallel extractions (MAIN COMPUTATION)
all_results = run_parallel_extractions(
    spatial_parts=spatial_parts,
    users_list=users_list,
    output_dir=Path("C:/work_dir/axios-2023-2024/1_parcel_stats"),
    run_identifier="axios-2023-2024",
    parcel_id_field="parcel_code",
    start_date="2023-10-01",
    end_date="2024-09-30",
    working_epsg=32634,
    sentinel2_bands=["B02", "B03", "B04", "B05", "B08", "B11", "B12"],
    calculate_sentinel2_indices=True,
    sentinel2_indices=["EVI", "NDMI", "NDRE", "NDVI", "NDWI", "SAVI"],
    sentinel1_bands=["VV", "VH"],
    spatial_statistics=["mean", "median", "sd", "min", "max", "p10", "p25", "p75", "p90"],
    tile_size_metres=20_000,
    tile_buffer_metres=2_100,
    openeo_parallel_jobs=2,
    remove_outliers=True,
    iqr_quantiles=(0.1, 0.9),
    iqr_multiplier=1.5,
    fill_nulls=True,
    iqr_min_valid_pixels=20,
    temporal_fill_mode="bidirectional",
    minimum_parcel_pixels=3,
    minimum_observed_fraction_for_fill=0.20,
    interpolation_method="nearest",
    interpolation_max_distance_in_meters=500,
    batch_workers=1,
)

# 5. Merge and save
merged_gdf, geoparquet_path, csv_preview_path = merge_and_save_results(
    gdf_parcels=gdf_parcels,
    all_results=all_results,
    output_dir=Path("C:/work_dir/axios-2023-2024/1_parcel_stats"),
    parcel_id_column="parcel_code",
)

print(f"SUCCESS! Results saved to:\n  {geoparquet_path}\n  {csv_preview_path}")
```

---

## Key Design Decisions

1. **Function Isolation**: Each function has clear input/output; no global variables
2. **Type Hints**: All functions have complete type annotations for IDE support
3. **Flexible Parameters**: Uses `**kwargs` to pass through extraction settings without maintaining huge parameter lists
4. **Parallel Execution**: ThreadPoolExecutor manages concurrent job submission per partition
5. **User Round-Robin**: Simple modulo arithmetic assigns users: `(part_idx - 1) % num_users`
6. **Result Ordering**: Results dict maintains partition order during merge via `sorted(all_results.keys())`
7. **CSV Preview**: Shows parcel IDs + median features (most commonly used statistics)

---

## Environment Variable Setup

Before running, set the `OPENEO_USERS` environment variable:

**Linux/macOS**:
```bash
export OPENEO_USERS="user1@example.com,pass1;user2@example.com,pass2;user3@example.com,pass3"
```

**Windows (PowerShell)**:
```powershell
$env:OPENEO_USERS="user1@example.com,pass1;user2@example.com,pass2;user3@example.com,pass3"
```

**Windows (cmd.exe)**:
```cmd
set OPENEO_USERS=user1@example.com,pass1;user2@example.com,pass2;user3@example.com,pass3
```

---

## Performance Expectations

With 2 users and 2 partitions (rows=1, cols=2):
- User 1 runs Partition 1 + 2 concurrent openEO jobs = 2 jobs
- User 2 runs Partition 2 + 2 concurrent openEO jobs = 2 jobs
- **Total parallelism: 4 concurrent jobs (2 users × 2 jobs per user)**
- **Execution time: ~50% of sequential extraction** (if balanced)

With 5 users and 5 partitions (rows=1, cols=5):
- **Total parallelism: 10 concurrent jobs (5 users × 2 jobs per user)**
- **Execution time: ~10-20% of sequential extraction** (if balanced)

---

## Output Structure

```
OUTPUT_DIR/
├── partition_1/
│   ├── monthly_cubes/
│   │   ├── f26268027b137f65-tiles-fe792142/
│   │   │   ├── 2023-10-01--2023-11-01.nc
│   │   │   └── ...
│   ├── satellite_openeo.log
│   ├── satellite_parcel_statistics.log
│   ├── satellite_raster_filling.log
│   └── satellite_parcel_time_stats.geoparquet  (partition results)
├── partition_2/
│   └── (same structure)
├── satellite_parcel_time_stats.geoparquet      (MERGED - final output)
└── satellite_parcel_time_stats_preview.csv     (MERGED - CSV preview)
```

---

## Error Handling

The module will raise clear exceptions for:
- Missing `OPENEO_USERS` environment variable
- Invalid credential format (missing semicolons, commas)
- Invalid grid parameters (rows/cols must be positive)
- Authentication failures (checked during load)
- GeoDataFrame type validation

All exceptions include descriptive messages to guide users toward solutions.
