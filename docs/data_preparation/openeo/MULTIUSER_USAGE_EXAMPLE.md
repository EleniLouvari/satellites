# Multi-user spatial extraction example

The workflow plans all partition tiles first, then shares unsubmitted batches
across all users, with up to two remote jobs per account. For `N` unique accounts, the maximum intended remote
concurrency is therefore `2 × N`.

Configure the PostgreSQL account table and kernel environment using
[database account setup](CREDENTIALS.md): install the `database` extra, set
`DB_NAME`, `TBL_USERS`, and the `POSTGRES_*` connection settings, and provide one
unique CDSE account per row with `username` and `password` columns. The loader
does not read `.env` automatically.

Then run:

```python
from pathlib import Path

from data_preparation.parcel_stats.job_manager import (
    compute_tile_buffer_metres,
    compute_tile_width,
)
from data_preparation.parcel_stats.multiuser import (
    load_openeo_users_from_db,
    merge_and_save_results,
    run_parallel_extractions,
    split_geodataframe_by_grid,
)

WORKING_EPSG = 32634
OUTPUT_DIR = Path("C:/work_dir/example/1_parcel_stats")
OPENEO_JOBS_PER_USER = 2

# gdf_parcels must already be loaded and contain a unique parcel_code column.
users_list = load_openeo_users_from_db()
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

`multiuser.run_parallel_extractions` validates the configuration and dispatches
to `batch_scheduler.run_shared_batches` or
`partition_scheduler.run_shared_partitions`. Each scheduler owns its execution
strategy; the notebook keeps the same entry point.

- `partition_count=len(users_list)` creates that many grid cells; partitions
  can also come from spatial clusters and need not match the account count.
- Each parcel belongs to exactly one partition, based on its centroid.
- `run_parallel_extractions(..., scheduling="batches")` is the default. All
  accounts pull unsubmitted batches from one shared queue. A partition with
  20 batches can use every account even when the other partitions are finished.
- Each account has at most `config["openeo_parallel_jobs"]` active slots (1 or 2).
  There is no partition-wide or fixed-wave barrier between remote batches.
- Cached NetCDFs are reused. Job IDs, retries and the account username persist
  in each partition's existing `monthly_cubes/<signature>/tile_jobs.parquet`.
  Passwords are not saved in these local job files; they come from PostgreSQL.
- Existing remote jobs stay with their owning account. When first resuming an
  older run without saved usernames, keep the original account list and order:
  ownership is inferred from the previous round-robin partition assignment.
  The database loader sorts by username; explicitly restore the original order
  when migrating such runs, following the [account ordering example](CREDENTIALS.md#account-ordering-and-restarting-jobs).
- Local statistics start after acquisition completes. At most one partition
  per account is processed locally at a time, each using `batch_workers`.
  Partition paths, batch numbering and the merge call are unchanged.
- `scheduling="partitions"` retains the legacy scheduling for legacy runs.
  Keep using batch scheduling to resume a run whose batches used mixed accounts.

Let a currently running notebook finish normally. Start a fresh kernel before
the next run to load the updated scheduler; do not run a second scheduler against
the same output directory concurrently. The notebook need not repartition the
parcels just to use all accounts.

`compute_tile_width` evaluates each candidate separately inside every actual
partition and selects the largest width that still supplies two non-empty jobs
per partition. If no candidate can do that, it selects the smallest candidate
to expose the maximum concurrency available in the data and prints a warning.
