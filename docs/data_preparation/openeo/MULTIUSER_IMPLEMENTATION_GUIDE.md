# Multi-user extraction implementation guide

The multi-user layer plans spatial partitions and shares their unsubmitted tile
batches across accounts. The implementation allows one or two remote slots per
account, exposing at most `2 × number_of_unique_users` simultaneous remote jobs.

See [scheduling workflows and flowcharts](SCHEDULING.md) for the differences
between `scheduling="batches"` and `scheduling="partitions"`.
See [parcel grouping with buffers or tiles](PARCEL_GROUPING.md) for choosing and
creating the spatial partitions supplied to either scheduler.

## Planning flow

```text
PostgreSQL account table (DB_NAME + TBL_USERS)
    -> load_openeo_users_from_db(): validate and sort unique accounts
    -> grid or cluster partitions
    -> plan all tile batches using every actual partition
    -> reuse cached cubes and pin existing jobs to their account
    -> share unsubmitted batches across accounts (up to two slots each)
    -> complete remote downloads
    -> local processing and persistence per partition
    -> explicit merge of saved time-statistics files
```

## Public functions

### `load_openeo_users_from_db()`

Reads `username` and `password` columns from the PostgreSQL table configured by
`DB_NAME` and `TBL_USERS`, using `POSTGRES_*` connection settings. Missing settings,
empty tables, missing columns, blank/non-text credentials and duplicate usernames
are rejected before scheduling. Usernames are trimmed and compared ignoring case;
passwords are preserved exactly. The connection closes after the read and accounts
are sorted by case-insensitive username.

See [database account setup](CREDENTIALS.md) for dependencies, table requirements,
environment loading and restoring the original account order for legacy runs.

### `split_geodataframe_by_grid(...)`

Two calling modes are supported:

```python
# Preferred for multi-user extraction: exactly one grid cell per user.
parts = split_geodataframe_by_grid(
    parcels,
    partition_count=len(users_list),
)

# Explicit grid shape remains supported.
parts = split_geodataframe_by_grid(parcels, rows=2, cols=3)
```

For `partition_count`, an exact factor pair is chosen to give grid cells whose
shape best fits the AOI aspect ratio. Parcels are assigned by centroid and occur
in exactly one output partition. Empty grid cells are omitted.

### `compute_tile_width(...)`

Pass the complete parcel set and the already-created partitions:

```python
tile_width = compute_tile_width(
    parcels,
    parcel_id_col="parcel_id",
    working_epsg=32634,
    spatial_parts=parts,
    user_count=len(users_list),
    jobs_per_user=2,
)
```

For each candidate width, the function reproduces the real tile planner's
representative-point ownership independently within every partition. It reports
the total jobs and the minimum, median, and maximum jobs per partition. The
selected width is the largest candidate for which every partition creates at
least `jobs_per_user` non-empty tile jobs. Geometry crossing a core tile is
handled by `compute_tile_buffer_metres`, not by the width selection.

### `run_parallel_extractions(spatial_parts, users_list, config)`

The default `scheduling="batches"` creates an `OpenEOJobManagerZonalStats` per
partition, plans all jobs, then lets account slots pull unsubmitted batches
from one shared queue. Existing job IDs remain pinned to their saved username;
each slot resumes its owned jobs before taking new work. Job database writes
are locked per partition. After downloads complete, local partition processing
runs with at most one partition per account and each partition's `batch_workers`.

With `scheduling="partitions"`, partitions instead enter round-robin account
queues and each account processes its partitions sequentially. Local raster work
is serialized within each process; explicit batch workers use separate processes.

The function returns `None`; each partition persists its result under
`output_dir/partition_<n>/satellite_parcel_time_stats.geoparquet`. Values above
two for `openeo_parallel_jobs` are rejected—add accounts to increase capacity.

### `merge_and_save_results(...)`

The merge can run immediately or in a later Python session. When `all_results`
is omitted, it discovers the saved partition result files under `output_dir`:

```python
merged, output_path = merge_and_save_results(
    gdf_parcels=parcels,
    output_dir=output_dir,
    parcel_id_column="parcel_id",
    working_epsg=32634,
)
```

The final output is
`output_dir/satellite_parcel_time_stats.geoparquet`.

## Capacity examples

| Unique users | Initial partitions | Remote jobs per user | Maximum remote jobs |
|---:|---:|---:|---:|
| 1 | 1 | 2 | 2 |
| 2 | 2 | 2 | 4 |
| 5 | 5 | 2 | 10 |

These are capacity ceilings. Sparse or empty partitions, backend scheduling,
and partitions that produce fewer than two non-empty tile jobs can reduce the
observed concurrency.
