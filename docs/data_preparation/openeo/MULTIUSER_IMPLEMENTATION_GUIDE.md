# Multi-user extraction implementation guide

The multi-user layer combines spatial grid partitioning with the two-job openEO
account limit. It exposes `2 × number_of_unique_users` remote slots without
allowing multiple local job managers to compete for the same account.

## Planning flow

```text
OPENEO_USERS
    -> one grid partition per user
    -> choose tile width using every actual partition
    -> one sequential partition queue per user
    -> one active job manager per user
    -> up to two remote tile jobs per active manager
```

## Public functions

### `load_openeo_users_from_env()`

Parses `OPENEO_USERS` entries in `username,password;username,password` format.
Empty credentials and duplicate usernames are rejected so the number of users
is a truthful concurrency count.

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

Partitions are assigned round-robin to user queues. User queues execute in
parallel, but each individual queue processes its partitions sequentially. Each
active partition creates one `JobManagerSatelliteZonalStats`, whose
`openeo_parallel_jobs=2` setting fills that account's two remote slots.

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
