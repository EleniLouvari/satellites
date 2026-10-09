# Grouping parcels with spatial buffers or tiles

Parcels can be grouped by proximity using `spatial_clustering_using_buffer`, or
by their position in a regular grid. Both approaches keep the original parcel
geometries whole. They differ in what determines membership and how the raster
extent for each openEO job is chosen.

The Neuro `1_parcel_stats_openEO_fill.ipynb` currently uses buffer clusters as
partitions, followed by fitted tile extents. The examples below show that approach
and an alternative using grid partitions followed by regular tiles.

## Partitions and remote jobs

There are two spatial grouping levels in the multi-user workflow:

1. **Partitions:** `spatial_parts`, a list of GeoDataFrames passed to
   `run_parallel_extractions`. Create these from cluster labels or with
   `split_geodataframe_by_grid`. Each partition has its own output directory.
2. **Remote batches:** the job manager plans one or more raster extents within
   each partition. `fit_tiles_to_parcels=True` fits extents to parcel bounds;
   `False` uses regular core tiles.

One partition is therefore not necessarily one openEO job. Clustering and tiling
can be combined: cluster membership defines the partition, and the tile planner
subdivides it when necessary. With the default `scheduling="batches"`, accounts
share the remote batches from all partitions. Partition count does not have to
match account count; see [scheduling](SCHEDULING.md).

## Method 1: Group nearby parcels with spatial buffers

`shared.spatial_statistics.spatial_clustering_using_buffer` builds clusters from
the parcel polygons themselves, rather than from centroid distances.

### How the function works

1. Copy the input and create the cluster-label column.
2. Buffer each parcel by **half** of `min_cluster_distance_in_m`, converted to
   the input CRS units.
3. Union the buffered geometries and explode the result into polygon components.
   Parcels whose buffers merge become part of the same connected group.
4. Optionally simplify the cluster boundaries using `simplify_in_m`.
5. Sort cluster polygons by descending buffered area. The largest receives
   `"Cluster Main"`; the others receive `"Cluster 1"`, `"Cluster 2"`, and so on.
6. Assign labels back to the original parcels using an `intersects` spatial join.
   If a parcel intersects multiple cluster polygons, keep the largest-area match.

The result is a GeoDataFrame with the original parcel geometries and an added
label column, not a list of groups or buffered replacement parcels. Use `groupby`
to turn the labels into `spatial_parts`.

### What the distance means

For a distance setting of 20,000 m, every parcel is buffered by 10,000 m.
This makes nearby polygons join across gaps of approximately 20 km, subject to
the actual buffer geometry and boundary contacts. The setting is a connectivity
threshold, not a maximum cluster width, height, radius, or parcel count.

For example, if A is 15 km from B and B is 15 km from C, their overlapping buffers
can connect all three into one cluster even when A and C are more than 20 km apart.
A long chain can therefore produce a very large cluster.

`simplify_in_m` defaults to `24.384` m. It simplifies the merged cluster boundary
before label assignment; it does not simplify the original parcel polygons.
Use `simplify_in_m=0` to disable this step. Always check for missing labels after
the function returns, because invalid/empty geometry handling or the spatial
join can leave parcels unassigned.

### Preparing cluster partitions

These examples assume `gdf_parcels`, `parcel_id_column`, and a suitable local
projected `WORKING_EPSG` with metre units are already defined. Use valid, non-empty
polygon geometries and unique, non-null parcel IDs. Reset to a unique unnamed
index because the clustering and grid helpers use the input index for assignment.

```python
from shared.spatial_statistics import spatial_clustering_using_buffer

parcels = gdf_parcels.to_crs(epsg=WORKING_EPSG).reset_index(drop=True)
CLUSTER_FIELD = "cluster"

gdf_grouped = spatial_clustering_using_buffer(
    parcels,
    field_id=parcel_id_column,
    min_cluster_distance_in_m=20_000,
    simplify_in_m=24.384,
    field_cluster=CLUSTER_FIELD,
    plot_results=False,
)
if gdf_grouped[CLUSTER_FIELD].isna().any():
    raise ValueError("Error: Every parcel must have a cluster label before partitioning.")

spatial_parts = [
    part.copy()
    for _, part in gdf_grouped.groupby(CLUSTER_FIELD, sort=True, observed=True)
]
```

The notebook reuses existing `cluster` labels if that column is already present.
Changing the distance parameter alone will therefore not regroup saved parcels;
explicitly rerun clustering if you intend to replace those labels. The function
prints/displays a cluster summary even with `plot_results=False`.

### From a cluster to fitted raster extents

Set `fit_tiles_to_parcels=True` in the extraction configuration. For each group,
the planner starts with the bounding box of all its complete parcel geometries
and adds `tile_buffer_metres` on every side:

```text
requested width  = maximum parcel x - minimum parcel x + 2 * tile_buffer_metres
requested height = maximum parcel y - minimum parcel y + 2 * tile_buffer_metres
```

If both dimensions are at most `tile_size_metres`, that group becomes one remote
batch. Otherwise, the planner splits along the longer spatial dimension, assigning
whole parcels using their bounding-box centers, and repeats. It splits at the
spatial midpoint, with a median fallback if all centers fall on one side.

The original cluster remains one output partition even if it needs several remote
batches. Parcel polygons are never cut. If one parcel alone exceeds the size
limit including its buffer, planning raises an error before submission.

```python
from data_preparation.parcel_stats.job_manager import compute_tile_buffer_metres

grouping_config = {
    "fit_tiles_to_parcels": True,
    "tile_size_metres": 20_000,
    "tile_buffer_metres": compute_tile_buffer_metres(gdf_grouped, WORKING_EPSG),
}
```

Here, 20,000 m is the maximum **buffered** width and height. For example, a group
spanning 18 km with a 1.4 km buffer needs 20.8 km and must be subdivided. The
clustering distance and this size limit are independent settings.

## Method 2: Group parcels with a regular grid and tiles

This route first divides the study area into outer grid partitions, then creates
regular remote tiles within each partition. The two grids have different controls
and ownership rules.

### Outer grid partitions

`split_geodataframe_by_grid` divides the input's total bounding box into rows and
columns. It assigns every parcel to one cell using its **centroid** and returns
the non-empty groups in row/column order. A parcel crossing a cell boundary stays
whole in its assigned group. Cell boundaries do not clip its geometry.

Choose either `rows` and `cols`, or `partition_count`. With `partition_count`, the
helper selects a factor pair based on the study area's aspect ratio. The requested
count is the number of grid cells; empty cells are omitted, so the number of
returned partitions may be smaller. This is spatial subdivision, not balancing
the number of parcels per group. Reproject before calling this helper: it does
not reproject its input itself.

```python
from data_preparation.parcel_stats.multiuser import (
    load_openeo_users_from_db,
    split_geodataframe_by_grid,
)

users_list = load_openeo_users_from_db()
parcels = gdf_parcels.to_crs(epsg=WORKING_EPSG).reset_index(drop=True)
spatial_parts = split_geodataframe_by_grid(
    parcels,
    partition_count=len(users_list),
)
# Alternatively: split_geodataframe_by_grid(parcels, rows=2, cols=3)
```

### Regular remote tiles

With `fit_tiles_to_parcels=False`, the job manager covers each partition's parcel
bounding box with regular core tiles controlled by `tile_size_metres`. It then:

1. Assigns each parcel to the core tile containing its **representative interior
   point**. This differs from the centroid rule used for outer partitions.
2. Resolves points on shared tile edges deterministically to one owner.
3. Skips tiles with no assigned parcels.
4. Buffers occupied tiles by `tile_buffer_metres` and verifies they cover their
   owned parcels completely.
5. Creates one remote job per occupied tile, requesting the buffered rectangular
   extent. Parcel masks are applied locally when calculating statistics.

Neighboring raster extents can overlap, but each parcel belongs to only one job.
If the buffer does not cover a parcel, planning stops and writes
`uncovered_buffered_tile_parcels.csv`; increase the buffer before retrying.

Here, `tile_size_metres` controls the **unbuffered core** dimensions, with smaller
edge cells possible. A full 20 km core tile with a 1.4 km buffer requests a
22.8 km-wide raster. It does not impose the 20 km buffered limit used by fitted
extents.

```python
from data_preparation.parcel_stats.job_manager import (
    compute_tile_buffer_metres,
    compute_tile_width,
)

grouping_config = {
    "fit_tiles_to_parcels": False,
    "tile_size_metres": compute_tile_width(
        parcels,
        parcel_id_col=parcel_id_column,
        working_epsg=WORKING_EPSG,
        spatial_parts=spatial_parts,
        user_count=len(users_list),
        jobs_per_user=2,
    ),
    "tile_buffer_metres": compute_tile_buffer_metres(parcels, WORKING_EPSG),
}
```

`compute_tile_width` tests 5, 10, 20, 30, 40, 50, and 60 km core widths by default.
It chooses the largest candidate producing at least `jobs_per_user` occupied
tiles in **every** supplied partition, or the smallest candidate if none qualify.
This is a regular-grid planning heuristic, not a memory guarantee or a fitted
extent size selector. Its partition-based concurrency messages should be read
alongside the scheduler behavior: shared batch scheduling can use several
accounts within one partition.

If outer subdivisions are unnecessary, use `spatial_parts = [parcels.copy()]`
and let regular tiles create all remote groups within that single partition.

## Worked examples with the same six parcels

These examples use synthetic 200 m by 200 m squares in a projected CRS. Run the
Python blocks in this section in order in the repository's geospatial environment
(including the `gis-utilities` dependencies used by the clustering module).
They only create in-memory geometries and groups; no database or openEO account
is needed. The remote-plan tables below describe the jobs those groups would
produce, without submitting them.

### Example 1: Create a small parcel layout

All six parcels share the same northing range. Their horizontal positions,
relative to the western edge of parcel A, are:

| Parcel | West edge (m) | East edge (m) | Gap from previous parcel (m) |
| --- | ---: | ---: | ---: |
| A | 0 | 200 | — |
| B | 600 | 800 | 400 |
| C | 1,200 | 1,400 | 400 |
| D | 5,000 | 5,200 | 3,600 |
| E | 5,600 | 5,800 | 400 |
| F | 12,000 | 12,200 | 6,200 |

```python
import geopandas as gpd
from shapely.geometry import box

demo_parcels = gpd.GeoDataFrame(
    {"parcel_id": ["A", "B", "C", "D", "E", "F"]},
    geometry=[
        box(500_000 + offset, 4_500_000, 500_200 + offset, 4_500_200)
        for offset in [0, 600, 1_200, 5_000, 5_600, 12_000]
    ],
    crs="EPSG:32634",
)
```

### Example 2: Cluster with a 500 m distance threshold

Each polygon gets a 250 m clustering buffer. The 400 m gaps between A/B, B/C,
and D/E are bridged, while the larger gaps remain separate.

```python
from shared.spatial_statistics import spatial_clustering_using_buffer

demo_grouped = spatial_clustering_using_buffer(
    demo_parcels,
    field_id="parcel_id",
    min_cluster_distance_in_m=500,
    simplify_in_m=0,
    plot_results=False,
)
demo_membership = demo_grouped.groupby("cluster", sort=True)["parcel_id"].agg(list)
print(demo_membership.to_dict())
```

Expected membership, in addition to the function's own summary:

```text
{'Cluster 1': ['D', 'E'], 'Cluster 2': ['F'], 'Cluster Main': ['A', 'B', 'C']}
```

A and C are separated by a 1,000 m edge-to-edge gap, but B connects them into
the same cluster. This demonstrates why the 500 m threshold is not a maximum
distance between every pair of parcels in a cluster.

Changing only the threshold produces these groups for this layout, with
`simplify_in_m=0` throughout:

| Distance threshold | Groups, ignoring their labels | Number of groups |
| ---: | --- | ---: |
| 300 m | A; B; C; D; E; F | 6 |
| 500 m | A, B, C; D, E; F | 3 |
| 4,000 m | A, B, C, D, E; F | 2 |
| 7,000 m | A, B, C, D, E, F | 1 |

### Example 3: Split the same parcels into grid partitions

A one-row, two-column grid splits the 12,200 m total width at 6,100 m. The first
cell therefore contains A through E, even though C and D are far apart.

```python
from data_preparation.parcel_stats.multiuser import split_geodataframe_by_grid

demo_grid_parts = split_geodataframe_by_grid(demo_parcels, rows=1, cols=2)
print([part["parcel_id"].tolist() for part in demo_grid_parts])
```

Expected membership:

```text
[['A', 'B', 'C', 'D', 'E'], ['F']]
```

Using four columns instead places boundaries at 3,050 m, 6,100 m, and 9,150 m:

```python
demo_four_cells = split_geodataframe_by_grid(demo_parcels, rows=1, cols=4)
print([part["parcel_id"].tolist() for part in demo_four_cells])
```

Expected membership:

```text
[['A', 'B', 'C'], ['D', 'E'], ['F']]
```

The third cell is empty, so four requested cells produce three partitions. In
this particular layout, those memberships happen to match the 500 m clusters;
the grouping rules are still different.

### Example 4: Turn the clusters into fitted remote jobs

Use the three clusters from Example 2 as partitions. For these 200 m squares,
`compute_tile_buffer_metres` returns 400 m: the radius is approximately 141.4 m,
multiplied by 2.5 and rounded upward to a multiple of 50 m.

```python
from data_preparation.parcel_stats.job_manager import compute_tile_buffer_metres

demo_cluster_parts = [
    part.copy()
    for _, part in demo_grouped.groupby("cluster", sort=True, observed=True)
]
demo_fitted_config = {
    "fit_tiles_to_parcels": True,
    "tile_size_metres": 2_000,
    "tile_buffer_metres": compute_tile_buffer_metres(demo_parcels, 32634),
}
```

The 2,000 m limit applies to the requested width and height, including the buffer:

| Cluster | Initial buffered extent | Planned parcel groups | Final raster extents |
| --- | --- | --- | --- |
| A, B, C | 2,200 × 1,000 m | A; B, C | 1,000 × 1,000 m; 1,600 × 1,000 m |
| D, E | 1,600 × 1,000 m | D, E | 1,600 × 1,000 m |
| F | 1,000 × 1,000 m | F | 1,000 × 1,000 m |

There are **three partitions and four remote jobs**. A/B/C must split because its
buffered width exceeds the limit. Its midpoint lies at 700 m, exactly at B's
bounding-box center; the current split rule places B on the right with C.
Increasing the fitted size limit to 2,500 m lets A/B/C stay together, producing
three jobs overall with the same cluster memberships and buffer.

### Example 5: Use regular remote tiles on the same parcel layout

To isolate the remote tiling rule, put all six parcels in **one** outer partition
and use 1,000 m core tiles with the same 400 m tile buffer:

```python
demo_regular_parts = [demo_parcels.copy()]
demo_regular_config = {
    "fit_tiles_to_parcels": False,
    "tile_size_metres": 1_000,
    "tile_buffer_metres": 400,
}
```

For this layout, regular tile boundaries align with the kilometre offsets shown
below. A parcel's representative point determines its owner tile:

| Occupied core x range relative to A | Owned parcels | Requested raster width × height |
| --- | --- | --- |
| 0–1,000 m | A, B | 1,800 × 1,000 m |
| 1,000–2,000 m | C | 1,800 × 1,000 m |
| 5,000–6,000 m | D, E | 1,800 × 1,000 m |
| 12,000–12,200 m | F | 1,000 × 1,000 m |

There is **one partition and four remote jobs**. Empty tiles create no jobs.
The study area is only 200 m high, and the last tile is clipped to 200 m wide;
adding two 400 m margins explains the 1,000 m buffered dimensions at those edges.

The regular and fitted examples both produce four jobs, but assign A/B/C
differently and request different raster widths. Equal job counts do not mean
equal grouping or download sizes. These deliberately small settings illustrate
the rules; they are not production size recommendations.

## Choosing between the approaches

| Question | Buffer clusters with fitted extents | Grid partitions with regular tiles |
| --- | --- | --- |
| What defines the initial groups? | Connected buffered parcel polygons | Centroid location in an outer grid cell |
| What controls group count? | Parcel layout and distance threshold | Grid dimensions, excluding empty cells |
| What defines the remote extent? | Parcel bounds plus buffer, split to fit | Occupied core tile plus buffer |
| What does `tile_size_metres` mean? | Maximum requested buffered width and height | Unbuffered core tile size |
| Are parcel counts balanced? | No; a chain can connect many parcels | No; dense tiles can contain many more parcels |
| Which layouts suit the rule? | Separated concentrations of nearby parcels | Broad coverage in regular spatial cells |
| What can increase work? | Long chains and elongated cluster bounds | Many small tiles and overlapping rasters |

Both routes download rectangular rasters, including space between parcels.
Neither guarantees the fewest jobs, smallest download, or shortest runtime.
Inspect the planned job counts and buffered extents for your actual parcel layout.

There are also two different uses of buffers: the **clustering buffer** determines
which parcels join a cluster, while the **tile buffer** adds raster coverage around
a remote extent. `compute_tile_buffer_metres` derives the latter from the largest
parcel's representative-point-to-bounding-box-corner distance, multiplies it by
2.5, and rounds upward to 50 m by default. It does not choose cluster membership.

## Running and resuming either approach

After choosing one of the grouping examples, apply its options to your existing
extraction configuration and run the same scheduler:

```python
from data_preparation.parcel_stats.multiuser import run_parallel_extractions

# extraction_config already contains dates, output paths, sensors and statistics.
# users_list must be loaded using the database account loader for either method.
extraction_config.update(grouping_config)
run_parallel_extractions(
    spatial_parts=spatial_parts,
    users_list=users_list,
    config=extraction_config,
    scheduling="batches",
)
```

Before submission, confirm every original parcel ID appears exactly once across
`spatial_parts`. Check ID membership as well as the total row count; equal row
counts alone cannot detect one omitted parcel plus one duplicate.

Keep saved cluster labels, partition membership/order, account ownership and
planning settings stable when resuming. Cluster labels are based on area ranking,
and `groupby(sort=True)` orders their strings rather than their numeric suffixes;
recomputing clusters can change both labels and partition numbering. Fitted and
regular tiles use different cache signatures. Use a new run/output directory
when deliberately changing the grouping strategy instead of reusing old
partition outputs as though their membership were unchanged.

See [account setup](CREDENTIALS.md), [the calling example](CALLING_EXAMPLE.md),
and [the detailed workflow](WORKFLOW.md) for the remaining configuration and merge
steps.

## Implementation references

- [Buffer clustering](../../../src/shared/spatial_statistics/clustering.py): `spatial_clustering_using_buffer`.
- [Outer grid partitioning](../../../src/data_preparation/parcel_stats/multiuser.py):
  `split_geodataframe_by_grid`.
- [Remote tile planning](../../../src/data_preparation/parcel_stats/job_manager.py):
  `_build_tile_plan`, `_build_fitted_tile_plan`, `compute_tile_width`, and `compute_tile_buffer_metres`.
