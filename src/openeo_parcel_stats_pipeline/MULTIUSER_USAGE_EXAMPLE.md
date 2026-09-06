# Multi-user spatial extraction example

The workflow uses one openEO account queue per user and up to two remote jobs
inside each queue. For `N` unique accounts, the maximum intended remote
concurrency is therefore `2 × N`.

Set the credentials before starting Python or the notebook kernel:

```powershell
$env:OPENEO_USERS="user1@example.com,password1;user2@example.com,password2"
```

Then run:

```python
from pathlib import Path

from openeo_parcel_stats_pipeline.zonal_stats_job_manager import (
    compute_tile_buffer_metres,
    compute_tile_width,
)
from openeo_parcel_stats_pipeline.zonal_stats_multiuser_manager import (
    load_openeo_users_from_env,
    merge_and_save_results,
    run_parallel_extractions,
    split_geodataframe_by_grid,
)

WORKING_EPSG = 32634
OUTPUT_DIR = Path("C:/work_dir/example/1_parcel_stats")
OPENEO_JOBS_PER_USER = 2

# gdf_parcels must already be loaded and contain a unique parcel_code column.
users_list = load_openeo_users_from_env()
spatial_parts = split_geodataframe_by_grid(
    gdf_parcels,
    partition_count=len(users_list),
)

tile_size_metres = compute_tile_width(
    gdf_parcels,
    parcel_id_col="parcel_code",
    working_epsg=WORKING_EPSG,
    spatial_parts=spatial_parts,
    user_count=len(users_list),
    jobs_per_user=OPENEO_JOBS_PER_USER,
)
tile_buffer_metres = compute_tile_buffer_metres(
    gdf_parcels,
    working_epsg=WORKING_EPSG,
)

config = {
    "output_dir": OUTPUT_DIR,
    "run_identifier": "example-2023-2024",
    "parcel_id_field": "parcel_code",
    "start_date": "2023-10-01",
    "end_date": "2024-09-30",
    "working_epsg": WORKING_EPSG,
    "sentinel2_bands": ["B02", "B03", "B04", "B05", "B08", "B11", "B12"],
    "sentinel2_indices": ["EVI", "NDMI", "NDRE", "NDVI", "NDWI", "SAVI"],
    "sentinel1_bands": ["VV", "VH"],
    "sentinel1_indices": ["R", "RVI"],
    "sentinel1_orbit_direction": "BOTH",
    "spatial_statistics": ["mean", "median", "sd", "min", "max"],
    "tile_size_metres": tile_size_metres,
    "tile_buffer_metres": tile_buffer_metres,
    "openeo_parallel_jobs": OPENEO_JOBS_PER_USER,
    "job_poll_seconds": 30,
    "batch_workers": 1,
}

run_parallel_extractions(spatial_parts, users_list, config)

merged_gdf, output_path = merge_and_save_results(
    gdf_parcels=gdf_parcels,
    output_dir=OUTPUT_DIR,
    parcel_id_column="parcel_code",
    working_epsg=WORKING_EPSG,
)
print(output_path)
```

## Scheduling behavior

- `partition_count=len(users_list)` creates one initial grid cell per account.
- Each parcel belongs to exactly one partition, based on its centroid.
- Partition queues run concurrently across users.
- A user queue is sequential, preventing two job managers from competing for
  the same account's two remote slots.
- The active manager runs at most two remote tile jobs for its account.
- If a grid cell is empty, it is omitted and the corresponding user may be idle.
- If there are more partitions than users, partitions are assigned round-robin
  and processed sequentially within their assigned user queue.

`compute_tile_width` evaluates each candidate separately inside every actual
partition and selects the largest width that still supplies two non-empty jobs
per partition. If no candidate can do that, it selects the smallest candidate
to expose the maximum concurrency available in the data and prints a warning.
