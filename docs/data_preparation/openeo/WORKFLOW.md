# Detailed openEO data preparation workflow

This guide follows the implementation in `data_preparation`: parcel
polygons and Sentinel selections enter the pipeline; dated parcel features,
run-wide summaries, cleaning audits, and restart checkpoints leave it. The remote
openEO backend prepares physical-band rasters. Local code cleans those bands,
calculates indices, masks parcels, and writes the statistics.

The main deliverable is `satellite_parcel_time_stats.geoparquet`: one row per
retained parcel, with its attributes, geometry, quality measures, and dated
satellite features. A finished openEO job, a downloaded `.nc`, and a completed
`_final.nc` are earlier milestones; none alone means that this table is ready.

Read section 2 for the complete route, section 6 for the cleaning decisions,
sections 7–8 for the conversion from pixels to parcel features, and sections
9–11 for persistence, merging, and recovery. The
[NetCDF processing guide](../NETCDF_PROCESSING.md) explains disk usage and
temporary files in more detail.

For limited disk space, enable `remove_nc_after_completion=True`: each batch's
NetCDF files are removed only after durable statistics and audit checkpoints
are saved. `resume_completed_partitions=True` also reuses completed partition
outputs. See [cleanup and resume](SCHEDULING.md#disk-cleanup-and-resume) for the
streaming behavior and requirements for resuming older runs.

## 1. Entry points and responsibilities

| Entry point | Work performed |
| --- | --- |
| `OpenEOZonalStats.run()` | Acquire parcel-batch cubes, then process and save them locally. |
| `OpenEOJobManagerZonalStats.run()` | Use the same local engine with tile planning, persistent remote job state, and retries. |
| `OpenEOZonalStats.run_from_batches(batches)` | Process already acquired `(batch_number, parcels, netcdf_path)` tuples and save results. |
| `run_parallel_extractions(parts, users, config)` | Schedule partition tile jobs across accounts and persist each partition's results. Returns `None`. |
| `merge_and_save_results(...)` | Load saved partition time-statistics files, merge them, and return `(GeoDataFrame, output_path)`. |

Use the current `OpenEOZonalStats` and `OpenEOJobManagerZonalStats` class names.
There are no historical class aliases in the public package.

All paths below are relative to `src/data_preparation/`:

| Module | Responsibility and principal methods |
| --- | --- |
| [parcel_stats/openeo.py](../../../src/data_preparation/parcel_stats/openeo.py) | `run`, `run_from_batches`: acquire, aggregate, reshape, persist. |
| [parcel_stats/core/base.py](../../../src/data_preparation/parcel_stats/core/base.py) | Constructor, loggers, `_run_batch_statistics`, process workers. |
| [parcel_stats/core/configuration.py](../../../src/data_preparation/parcel_stats/core/configuration.py) | Validation, required/output variables, intervals, raw-cache signature. |
| [parcel_stats/core/parcel_batches.py](../../../src/data_preparation/parcel_stats/core/parcel_batches.py) | `_prepare_parcels`, `_iter_spatial_batches`. |
| [sources/openeo/cubes.py](../../../src/data_preparation/sources/openeo/cubes.py) | Authentication, process graphs, temporal aggregation, base job waves and downloads. |
| [parcel_stats/job_manager.py](../../../src/data_preparation/parcel_stats/job_manager.py) | Tile extents, job database, concurrency, retries, download recovery. |
| [parcel_stats/core/streamed_raster.py](../../../src/data_preparation/parcel_stats/core/streamed_raster.py) | Disk-backed physical and final checkpoints, layer/strip/window processing. |
| [parcel_stats/core/raster_cleaning.py](../../../src/data_preparation/parcel_stats/core/raster_cleaning.py) | IQR filtering, temporal filling, spatial means, interpolation, audit counts. |
| [parcel_stats/core/parcel_statistics/](../../../src/data_preparation/parcel_stats/core/parcel_statistics) | Cube validation, index formulas, parcel masks, reducers, geometry metrics, streamed checkpointing, and wide output. |
| [parcel_stats/multiuser.py](../../../src/data_preparation/parcel_stats/multiuser.py) | Accounts, partitions, scheduler selection, saved-result merging. |
| [parcel_stats/batch_scheduler.py](../../../src/data_preparation/parcel_stats/batch_scheduler.py) | Default shared batch queue and persisted account ownership. |
| [parcel_stats/partition_scheduler.py](../../../src/data_preparation/parcel_stats/partition_scheduler.py) | Optional sequential partition queue per account. |
| [features/temporal.py](../../../src/data_preparation/features/temporal.py) | Summarize each source's dated parcel medians. |

## 2. End-to-end flow

The standalone flowchart includes acquisition, local cleaning decisions, parcel
aggregation, output writes, and restart shortcuts. Read its three columns from
left to right using the matching A/B continuation connectors.

[Open PNG](OPENEO_CLEANING_FLOWCHART.png) ·
[Open SVG](OPENEO_CLEANING_FLOWCHART.svg) ·
[Editable draw.io diagram](OPENEO_CLEANING_FLOWCHART.drawio)

![openEO acquisition, cleaning, and final GeoParquet flowchart](OPENEO_CLEANING_FLOWCHART.svg)

```mermaid
flowchart TD
    A[Parcel polygons, dates, sensor selections and settings] --> B[Validate configuration and repair geometries]
    B --> P0{Completed partition reusable and resume enabled?}
    P0 -->|Yes| DONE[Return saved partition GeoDataFrame]
    P0 -->|No| C[Plan parcel batches or buffered tiles]
    C --> BC{Matching batch statistics and report saved?}
    BC -->|Yes| LOAD[Read saved batch tables]
    BC -->|No| D{Complete raw cube cached?}
    D -->|No| E[Authenticate and build physical-band process graph]
    E --> F[Execute remote jobs]
    F --> DL[Download partial NetCDF and commit raw NetCDF]
    D -->|Yes| G[Validate raw cube metadata and time coordinates]
    DL --> G
    G --> H{Final checkpoint and audit reusable?}
    H -->|Yes| N[Read parcel windows from final checkpoint]
    H -->|No| I{Physical checkpoint and audit reusable?}
    I -->|No| J[Read each physical band and time slice; optional IQR filtering]
    J --> K[Optional temporal fill, spatial mean passes and interpolation]
    K --> L[Commit physical checkpoint and cleaning audit]
    I -->|Yes| M[Calculate pixel indices and their provenance]
    L --> M
    M --> O[Commit final checkpoint; optionally remove physical checkpoint]
    O --> N
    N --> P[Reduce valid parcel pixels for each reporting period]
    P --> SAVE[Commit batch report, then batch statistics Parquet]
    SAVE --> CLEAN[Optionally delete completed batch NetCDFs]
    CLEAN --> ALL[Collect tables from all batches]
    LOAD --> ALL
    ALL --> Q[Add geometry and derived statistics]
    Q --> R[Pivot dated features; attach parcel attributes and geometry]
    R --> S[Write time statistics, annual summary and cleaning report]
    S --> MARK[Mark partition complete and return GeoDataFrame]
    MARK --> T[For multi-user runs: explicitly merge partition time statistics]
    DONE --> T
```

The diagram shows dependencies and cache shortcuts for one extraction partition;
it does not prescribe concurrency. With default multi-user scheduling, all
partitions finish remote acquisition before local batch processing begins.

### What changes at each boundary?

| Stage | Data representation | Where it runs | Evidence of completion |
| --- | --- | --- | --- |
| Parcel preparation | Polygons with unique IDs and configuration | Local Python | Validated parcels and batch/tile plan |
| Image preparation | Physical-band raster cube over time | Remote openEO backend | Remote job has finished |
| Download | Raw physical-band NetCDF | Backend to local disk | Committed `batch_XXXXX_monthly.nc` |
| Cleaning | Physical bands plus observation/fill masks | Local Python | Compatible `_cleaned.nc` and cleaning report, or a later checkpoint |
| Index creation | Requested bands and pixel indices plus masks | Local Python | Compatible `_final.nc` and cleaning report |
| Parcel aggregation | One intermediate row per parcel/reporting period | Local Python | Matching batch statistics and report Parquets |
| Feature assembly | One wide row per parcel, including geometry | Local Python | Both GeoParquets, cleaning CSV, and complete partition manifest |
| Multi-partition merge | One combined parcel table | Local Python | Explicitly saved root time-statistics GeoParquet and coverage checks |

Here, **raw** means before local cleaning. Remote scaling, cloud masking, and
optional temporal aggregation have already occurred. **Final raster** means
ready for parcel aggregation; **final GeoParquet** means the parcel features
have been assembled and saved.

## 3. Prepare inputs and resolve configuration

Supply a `GeoDataFrame`, `start_date`, `end_date`, `output_dir`, and projected
`working_epsg`. Choose an identifier with `parcel_id_field` (default `parcel_id`).
The constructor performs the following work before remote submission:

1. Validate dates, projected CRS, selected bands/indices, temporal settings,
   reducers, cleaning settings, interpolation distances, and worker counts.
2. Copy parcels and preserve their source attributes. If the identifier column
   is absent, create it from stringified index values; identifiers are converted
   to strings. Supply meaningful non-null IDs upstream.
3. Repair geometries using `make_valid`, extract polygonal components, and
   remove null, empty, and non-polygonal results. Reject an empty result or
   duplicated IDs after conversion and filtering.
4. Store normalized parcels in EPSG:4326. Use `working_epsg` for metric planning
   and final geometry; align parcel masks to the actual raster CRS locally.
5. Resolve physical source bands independently from requested output variables.
   For example, NDVI requires B08 and B04 even if neither is an output band.
6. Construct reporting intervals anchored at `start_date`, with an exclusive
   upper boundary and labels equal to interval starts. `1M` means monthly steps
   from that date; `15D`, `2M`, `3M`, and `1Y` are also supported.

Default S2 output bands are B02, B03, B04, B05, and B08; indices and S1 outputs
are disabled unless selected. At least one band or index must be requested.
`mean` is always included in spatial statistics. Request `median` explicitly
for useful annual-summary features. IQR removal and filling default to disabled.

## 4. Plan spatial work

Multi-user planning has two levels: create the outer `spatial_parts`, then let
each partition's job manager plan its remote raster batches. A partition can
contain several jobs. See [parcel grouping](PARCEL_GROUPING.md) for both methods
and [worked examples](PARCEL_GROUPING.md#worked-examples-with-the-same-six-parcels)
showing actual memberships and raster dimensions.

### 4.1 Create the outer partitions

Choose how to build the GeoDataFrames passed to `run_parallel_extractions`:

- **Buffer clusters:** `spatial_clustering_using_buffer` buffers each parcel by
  half of `min_cluster_distance_in_m`, unions the buffers into polygon components,
  and assigns cluster labels back to the original parcels. Create `spatial_parts`
  by grouping on those labels. Connectivity can chain across many parcels, so the
  distance threshold does not limit a cluster's width or height. Reject missing
  labels before `groupby`, which would otherwise omit those parcels.
- **Grid partitions:** `split_geodataframe_by_grid` assigns whole parcels to
  outer cells by their **centroids**. Supply `rows` and `cols`, or
  `partition_count`. Empty cells are omitted; parcel counts need not be balanced.
  Use a projected input CRS and a unique index for reliable assignment.

Alternatively, supply one partition containing all parcels and let the manager
create the remote batches. Each parcel must appear in exactly one partition.
These helpers organize whole geometries; they do not cut parcels at boundaries.

The Neuro fill notebook currently reuses existing `cluster` labels or generates
them with a 20,000 m clustering threshold. It creates one partition per cluster
and selects fitted extents with a 20,000 m maximum buffered dimension. These two
20 km settings have independent meanings; a connected cluster can exceed the
remote size limit and require several jobs.

### 4.2 Plan remote batches within each partition

The manager supports two extent strategies, independently of how outer partitions
were created:

| Manager setting | Parcel assignment | Requested raster extent |
| --- | --- | --- |
| `fit_tiles_to_parcels=False` (default) | Representative point selects one core tile | Core tile bounds plus buffer |
| `fit_tiles_to_parcels=True` | Whole-parcel groups split only above the size limit | Group bounds plus buffer |

Regular-grid planning resolves points on shared edges deterministically, skips
empty tiles, and checks that each buffered tile covers its assigned whole parcels.
A coverage failure writes `uncovered_buffered_tile_parcels.csv` and stops before
submission. Parcel counts do not determine tile sizes; dense tiles can contain
many more parcels than their neighbors.

Fitted planning begins with all parcels in the partition. It splits oversized
groups along the longer spatial dimension using parcel bounding-box centers,
with a median fallback when all centers fall on one side of the midpoint. It
repeats until every requested extent fits. An individual parcel whose bounds plus
buffer exceed the limit raises an error, because the planner cannot cut it.

In both manager modes, one non-empty planned batch becomes one remote job.
Each parcel has one owner even when buffered rectangular download extents overlap.

The base `OpenEOZonalStats` path uses a separate batching strategy: group projected
parcel centroids into 50 km cells, sort spatially, then split each cell into
batches of at most 5,000 parcels. It sends batch geometries through
`filter_spatial`; it does not use the manager's fitted/regular tile setting.

### 4.3 Interpret tile sizes and buffers

For regular tiles, `tile_size_metres` is the **unbuffered core** size. For fitted
extents, it is the **maximum requested width and height including the buffer**.
The manager defaults are 50,000 m and a 500 m buffer; the Neuro notebook overrides
the size to 20,000 m and calculates its buffer from the parcel geometries.

For example, with `tile_size_metres=20_000` and `tile_buffer_metres=1_400`:

- A full regular core tile requests a 22,800 m-wide raster after buffering.
- A fitted group spanning 18,000 m needs 20,800 m after buffering, so it must split.

`compute_tile_buffer_metres` chooses a conservative coverage margin from parcel
geometry. This tile buffer is separate from the buffer used to connect clusters.
`compute_tile_width` compares regular-grid candidates using the actual partitions,
selecting the largest candidate with enough occupied tiles per partition, or the
smallest candidate if none meet the target. It does not select a fitted-extent
limit or guarantee backend memory usage.

### 4.4 Schedule and preserve the plan

The base extractor uses remote waves of at most `batch_workers` and the same
setting for local processes. The manager uses `openeo_parallel_jobs` for remote
slots and `batch_workers` for local processing. In default multi-user batch
scheduling, remote slots belong to accounts and draw from all partitions;
partition count does not have to equal account count. See [scheduling](SCHEDULING.md).

Keep partition membership/order and planning settings stable when resuming.
Fitted and regular tiles use separate cache signatures. When deliberately
regrouping parcels, use a new run/output directory so completed partition outputs
are not mistaken for results of the new grouping.

## 5. Acquire Sentinel rasters remotely

Reuse matching batch statistics and report tables when available; these can
make acquisition unnecessary even after NetCDF cleanup. Otherwise reuse an
existing complete `batch_XXXXX_monthly.nc`. For missing cubes and tables,
connect to the configured CDSE endpoint. With both `openeo_username` and
`openeo_password`, use the configured OIDC password flow; otherwise use the
client's cached/interactive OIDC flow. Credentials are not written to job tables.

### Sentinel-2 branch

1. Load `SENTINEL2_L2A` with required physical bands plus SCL, the temporal extent,
   and configured maximum scene cloud cover (70).
2. Restrict spatial extent using batch polygons or the manager's rectangle.
3. Resample reflectance bilinearly to 10 m in `working_epsg`, then scale by 0.0001.
4. Align SCL to reflectance with nearest-neighbor resampling.
5. Mask SCL classes 0, 1, 3, 8, 9, 10, and 11. Expand cloud-related classes
   3, 8, 9, 10, and 11 with a 3x3 kernel before masking cloud edges.
6. Retain physical source bands and apply the selected temporal behavior.

### Sentinel-1 branch

1. Load `SENTINEL1_GRD` with required VV/VH polarizations. Apply the requested
   ascending/descending filter, or no direction filter for `BOTH`.
2. Calculate `sigma0-ellipsoid` backscatter with `COPERNICUS_30` and
   `local_incidence_angle=False`. Values remain linear power.
3. Apply temporal behavior, then align radar to the S2 grid when S2 is present.
   For S1-only runs, resample directly to 10 m in `working_epsg`.
4. Merge the two sensor cubes when both are enabled. Indices are computed locally.

### Temporal behavior

| Setting | Remote raster | Local statistic sample |
| --- | --- | --- |
| `mean`, `median`, `min`, `max`, `sum` | One composite per reporting period. For `1M` starting on a month boundary, use the native monthly operator; otherwise explicit intervals. | Valid parcel pixels in each composite. |
| `"none"` or `None` | Retain acquisition timestamps. | Pool valid pixel observations from all acquisitions within the reporting interval, separately per output variable. |

For retained acquisitions, `[1, 2, 3, 4]` from one image and `[10]` from another
give a pooled mean of 4 and median of 3. Every valid observation has equal weight.
Indices are calculated at each pixel/acquisition before pooling. With composites,
indices use composite physical bands; an index of band medians is generally
different from a median of acquisition-level indices.

Local intervals include the configured final calendar day. The current remote
extent helper adds one day to `end_date` only for retained acquisitions; composite
runs pass `end_date` unchanged. Thus the code does not apply the same explicit
final-day extension in both modes. This is a current implementation limitation.

### Job completion and downloads

The base path creates bounded waves, waits for jobs, downloads to `.partial.nc`,
and renames on success. The manager persists `tile_jobs.parquet`, monitors slots,
and uses the same partial-file completion rule. Default polling is 30 seconds;
the manager allows three retries with a 60-second cooldown.

Retryable statuses are `error`, `start_failed`, and `queued_for_start_failed`.
Start failures with a saved ID retry that existing job; other eligible failures
create replacements. Retry counts and failed job IDs persist across restarts.
Missing downloads for finished jobs are restored from their saved remote IDs.
If expected cubes remain missing, extraction raises with job/database details.

## 6. Validate and clean each cube locally

`_run_batch_statistics` first reuses matching saved batch statistics/report pairs.
For batches that still require raster processing, `_calculate_streamed_statistics`
opens the raw NetCDF and validates physical bands, temporal/x/y dimensions, CRS,
and timestamps before reusing or creating checkpoints. Acquisition timestamps
are sorted stably and must lie inside the configured intervals. Composite period
labels must be unique and recognized. Missing raster CRS is an error.

The cleaning order is:

```text
physical bands -> IQR filtering -> temporal filling -> spatial means
               -> final interpolation -> pixel indices -> parcel statistics
```

Both `remove_outliers` and `fill_nulls` default to **False**. Enable each
explicitly to run its stages. Cloud masking on the remote S2 cube is separate
from these switches and still takes place when local cleaning is disabled.

| Stage | Behavior |
| --- | --- |
| IQR, when enabled | Compute quantiles across the complete raster for one physical band/time slice. Remove values outside `Qlow - multiplier * IQR` and `Qhigh + multiplier * IQR`. Skip filtering below `iqr_min_valid_pixels` (default 20). |
| Fill eligibility | Use the full layer's post-IQR observed fraction. Default threshold is 0.20; a layer below it cannot be filled or supply temporal observations. |
| Temporal fill | Work at the same spatial pixel. Default `past_only` uses the nearest earlier observation. `bidirectional` averages closest earlier/later observations, or uses the available side. Optional odd windows count time steps, not days. Filled values do not extend the temporal search. |
| Spatial means | Apply configured neighborhoods sequentially: default 3x3 then 5x5; optional 7x7 and 9x9. Earlier passes can supply later passes. |
| Final interpolation | Fill eligible remaining gaps using `nearest`, `idw`, or `kriging`. IDW and kriging require their corresponding distance settings. Remaining nulls are retained. |

### 6.1 Cleaning decision flow

```mermaid
flowchart TD
    RAW[Validated raw physical bands] --> READ[Read one band and time layer as float32]
    READ --> IQR{Outlier removal enabled and enough finite pixels?}
    IQR -->|Yes| FILTER[Compute raster IQR bounds and replace outliers with NaN]
    IQR -->|No| OBS[Store observations and observed mask]
    FILTER --> OBS
    OBS --> NEXT{All physical layers stored?}
    NEXT -->|No| READ
    NEXT -->|Yes| COVER[Calculate post-IQR observed fraction for every layer]
    COVER --> FILL{Null filling enabled?}
    FILL -->|No| COMMIT[Commit physical cleaning report and cleaned NetCDF]
    FILL -->|Yes| ALLOW{Layer meets minimum observed fraction?}
    ALLOW -->|No| KEEP[Keep existing observations and leave gaps missing]
    ALLOW -->|Yes| TEMP[Fill at same pixel using eligible temporal observations]
    TEMP --> SPACE[Fill remaining gaps with configured spatial mean passes]
    SPACE --> GAP{Gaps remain?}
    GAP -->|Yes| INTERP[Apply configured final interpolation]
    GAP -->|No| AUDIT[Record remaining nulls and fill-stage counts]
    INTERP --> AUDIT
    KEEP --> AUDIT
    AUDIT --> COMMIT
    COMMIT --> INDEX[Calculate requested pixel indices and their provenance]
    INDEX --> FINAL[Commit final NetCDF for parcel statistics]
```

Eligibility is decided independently for each physical band/time layer. The
diagram's eligible/ineligible branches apply across those layers, and the
checkpoint commits after all of them have been processed. Implementation:
[`_stream_physical_bands` and `_stream_temporal_fill`](../../../src/data_preparation/parcel_stats/core/streamed_raster.py).

### 6.2 IQR removal and the coverage gate

With the default quantiles and multiplier:

```text
IQR = Q75 - Q25
lower bound = Q25 - 1.5 * IQR
upper bound = Q75 + 1.5 * IQR
observed fraction = finite pixels after IQR / all spatial cells in the layer
```

For example, if a layer has Q25 = 0.20 and Q75 = 0.40, its accepted interval
is -0.10 through 0.70. A value of 0.90 becomes missing when filtering is enabled
and there are at least 20 finite reference pixels. These are statistical bounds,
not physical reflectance limits. Each band/date gets its own bounds.

For a 10,000-cell layer, 2,000 observed cells meet the default 20% fill threshold;
1,999 do not. An ineligible layer neither receives fills nor supplies temporal
donor values. Its existing observations remain usable for parcel statistics.
The denominator includes the rectangular raster outside parcels and cloud-masked
cells. It is not the number of cells in a particular parcel.

### 6.3 Temporal filling before spatial filling

Suppose one pixel has eligible observations `[0.20, NaN, 0.60]` at three stored
times, and the middle layer also meets the coverage threshold:

| Temporal mode | Result | Meaning |
| --- | --- | --- |
| `past_only` (default) | `[0.20, 0.20, 0.60]` | Carry the nearest preceding observation forward. |
| `bidirectional` | `[0.20, 0.40, 0.60]` | Average the nearest preceding and following observations. |

The bidirectional average is not weighted by elapsed days. If only one side is
available, that side supplies the value. `temporal_fill_window_sizes=None`
allows an unlimited search across stored time steps; a window of 5 searches at
most two steps away in each permitted direction. Wider configured windows retry
remaining gaps using original eligible observations.

After temporal filling, spatial passes run separately on each band/time layer:

1. A missing cell with finite neighbors in the default 3x3 window receives
   their mean. Existing finite cells are preserved.
2. Remaining gaps are retried with the default 5x5 window. Values filled in
   the earlier pass can contribute to this pass.
3. Remaining gaps reach the interpolation method, default `nearest`.
   Nearest filling has no maximum-distance cutoff in this implementation.
   IDW/kriging use their configured distance options.

The spatial passes operate before parcel masking, so a donor may be outside the
target parcel. Raster buffer size and grouping can therefore affect both the
available donors and the IQR reference distribution.

Temporal filling can also cross reporting-period boundaries.
With `temporal_reducer="none"`, disable both IQR removal and filling to measure
all observed valid pixels without outlier exclusion or imputation.

The engine reads one physical band/time layer at a time, fills temporal strips
containing all dates, then processes complete spatial layers. Parcel reduction
reads one parcel window, output variable, and reporting period at a time. Exact
medians/percentiles still need that sample pool in memory. Within a process,
`LOCAL_RASTER_LOCK` serializes raster access; extra batch processes have separate
memory requirements. The streamed design reduces RAM use by writing intermediate
NetCDF data to disk.

## 7. Calculate indices and preserve provenance

`_stream_output_indices` creates requested output bands and indices from cleaned
physical bands, one time slice/output at a time. Extra dependency bands do not
become standalone outputs unless requested. Existing indices in raw cubes are
ignored and recalculated.

Examples: `NDVI = (B08 - B04) / (B08 + B04)`, `NDWI = (B03 - B08) / (B03 + B08)`,
`R = VV / VH`, and `RVI = 4 * VH / (VV + VH)`. Safe division leaves invalid
denominators as nulls. See `_calculate_optical_index` and
`_calculate_local_sentinel1_index` for all formulas.

Both physical and final checkpoints have dimensions `(period, variable, y, x)`:

| Array | Meaning |
| --- | --- |
| `cleaned` (`float32`) | Final values for that checkpoint's bands or indices. |
| `observed_mask` (`uint8`) | A finite value available from post-IQR observations alone. |
| `temporal_filled_mask` (`uint8`) | A value made available at the temporal stage. |

Index masks are derived by evaluating the index from observation-only and
post-temporal source states. A finite final value that is neither observed nor
temporal is attributed to spatial filling, including final interpolation.

For NDVI, the practical provenance cases are:

| B08 and B04 inputs | Classification of a finite NDVI result |
| --- | --- |
| Both observed after IQR | Observed |
| Available after temporal filling, but not both observed | Temporally filled |
| At least one required input needs spatial filling/interpolation | Spatially filled |
| Formula remains invalid or a required input stays missing | Missing |

This is why filling an index directly would lose the connection to its source
bands. The pipeline first fills the physical bands and then evaluates the formula.

## 8. Aggregate parcel pixels and reshape features

1. Reproject each parcel to the raster CRS and crop to its pixel bounding window.
2. Rasterize with `all_touched=True`: boundary-intersecting cells participate;
   there is no fractional-area weighting.
3. Reduce finite values for each variable/reporting period. Supported statistics
   include mean, median, sample SD (`ddof=1`), min, max, range, and p10/p25/p75/p90.
   Empty samples produce null statistics; SD requires at least two observations.
4. Record static counts, quality flags, filling provenance, and rasterized area.
   Parcels below `minimum_parcel_pixels` (default 3) remain in the results.
5. Add projected geometry area, compactness, perimeter/area ratio, shape index,
   elongation, and shape complexity. Add derived range, variance, coefficient of
   variation, IQR, Bowley skewness, and p90-p10 spread where source statistics exist.
6. Verify one row per parcel/period and that static metrics are consistent across
   periods. Pivot temporal columns to `<variable>_<statistic>__YYYYMMDD`.
7. Pad missing reporting-period feature columns with nulls. Attach parcel
   attributes and geometry once, rejecting collisions with generated columns,
   and write geometry in `working_epsg`.

The quality denominator is based on the stored cube:

```text
expected_pixel_count = intersected spatial cells × output variables × stored time steps
data_reliability_score = observed_count / expected_pixel_count
temporal_filled_ratio = temporal_filled_pixel_count / expected_pixel_count
spatial_filled_ratio = spatial_filled_pixel_count / expected_pixel_count
```

Stored time steps mean composites in reduced mode or acquisition timestamps in
retained-acquisition mode. A merged multisensor time axis can include timestamps
without observations for one sensor. Reliability is observed coverage of those
slots, not a classification confidence or an accuracy estimate. A zero denominator
produces null ratios. `intersected_pixel_count` counts spatial cells only.

### Worked example: cleaned pixels become a GeoParquet row

The following illustrative values are already cleaned and belong to three
intersected cells for one parcel in one composite period:

| Cell | B08 | B04 | Pixel NDVI | Provenance |
| --- | --- | --- | --- | --- |
| 1 | 0.60 | 0.20 | 0.50 | Both bands observed |
| 2 | 0.40 | 0.10 | 0.60 | B08 temporally filled; B04 observed |
| 3 | 0.30 | 0.10 | 0.50 | B04 spatially filled; B08 observed |

For NDVI, local reduction gives `mean = 0.533333`, `median = 0.50`,
`min = 0.50`, and `max = 0.60`. This is a reduction of **pixel NDVI**.
Calculating NDVI from the mean bands would instead give approximately 0.529412;
the operations are not interchangeable.

If the period starts on 2024-01-01, the intermediate row contains
`parcel_id`, `period_start`, `NDVI_mean`, `NDVI_median`, and other requested
statistics. Pivoting produces columns such as:

```text
parcel_id | NDVI_mean__20240101 | NDVI_median__20240101 | ... | geometry
P001      | 0.533333            | 0.50                  | ... | POLYGON (...)
```

If NDVI is the only requested output and this is the only stored time, the
parcel has three expected value slots: one observed, one temporally filled,
and one spatially filled. Reliability and both fill ratios are each 1/3.
The B08/B04 dependency bands do not add output slots unless also requested.

With multiple periods, the row gains additional dated columns; it does not gain
additional parcel rows. Geometry and original parcel attributes are attached
once. The annual sibling then summarizes the dated **parcel medians**, rather
than returning to raster pixels.

## 9. Persist results

Paths are relative to the extractor's `output_dir`:

| Artifact | Contents |
| --- | --- |
| `satellite_parcel_time_stats.geoparquet` | One row per retained parcel; original attributes, static metrics, dated features, projected geometry. Also returned by `run()`. |
| `satellite_parcel_annual_stats.geoparquet` | Static fields plus mean/min/max/population SD of each selected source's dated parcel medians. |
| `satellite_pixel_cleaning_report.csv` | Physical-band raster audit by batch/time/variable: IQR and per-fill-stage null/fill counts. |
| `satellite_zonal_stats.log` | Configuration and orchestration. |
| `satellite_source.log` | Acquisition, remote jobs and downloads. |
| `satellite_raster_filling.log` | Filling stage progress. |
| `satellite_parcel_statistics.log` | Cube validation, checkpoints and parcel reduction. |
| `monthly_cubes/<signature>/batch_XXXXX_monthly.nc` | Complete raw physical-band cube; filename also used for non-monthly/acquisition runs. |
| `.../batch_XXXXX_monthly_cleaned.nc` | Intermediate cleaned physical bands and provenance; normally removed after final checkpoint commit. |
| `.../batch_XXXXX_monthly_cleaning_report.parquet` | Persisted per-batch raster cleaning audit. |
| `.../batch_XXXXX_monthly_final.nc` | Requested bands/indices and provenance used for parcel statistics. |
| `.../batch_XXXXX_monthly_stats_<statistics_signature>.parquet` | Durable per-parcel/per-period batch statistics before final feature assembly. |
| `.../batch_XXXXX_monthly_stats_<statistics_signature>_report.parquet` | Companion batch audit; both tables must exist for batch-cache reuse. |
| `satellite_partition_completion.json` | Processing signature and `processing`/`complete` state for partition resume checks. |
| `.../tile_jobs.parquet`, `.../job_manager/` | Manager job state and diagnostic artifacts. |

Workers use `.worker-<pid>.log` suffixes. The annual file is an aggregation of
per-period parcel medians over the **entire run**, even for multi-year date ranges;
it does not split by calendar year. Its names are, for example,
`NDVI_median_annual_mean` and `NDVI_median_annual_std` (`ddof=0`). All dated columns
are removed from that file. Include spatial `median`: with no median columns,
the current reducer can emit null annual features.

Writes occur in this order: time statistics, annual statistics, cleaning CSV.
The annual file and raster checkpoints use temporary files plus replacement;
the three final outputs are not committed as one transaction. Confirm all three
outputs belong to the completed run before downstream analysis.

### Commit order and the meaning of completion

For each batch that needs local calculation:

1. Write physical bands to `_cleaned.partial.nc` and the audit to a temporary
   Parquet. Commit the audit first, then the closed `_cleaned.nc`.
2. Write requested bands/indices to `_final.partial.nc`, close it, and rename
   it to `_final.nc`. By default, remove `_cleaned.nc` at this point; retain its audit.
3. Calculate parcel statistics. Commit the signature-specific batch report
   Parquet first, then the batch statistics Parquet.
4. If `remove_nc_after_completion=True`, remove the batch's raw, cleaned, final,
   and corresponding partial NetCDFs only after that table pair is saved.

For partition assembly, mark the partition `processing`, collect batch tables,
derive and pivot features, write the time GeoParquet, write the annual GeoParquet,
and write the cleaning CSV. Only then mark the partition `complete` and return
the time-statistics GeoDataFrame. In the shared scheduler, batch checkpoints can
already exist before partition assembly starts.

`.partial.nc` means an unfinished write, not a partially cloudy image or a
percentage of parcels completed. Such a file is rebuilt from earlier committed
inputs after interruption; it is never promoted by manually renaming it.

## 10. Multi-user orchestration

See [batches versus partitions: scheduling workflows and flowcharts](SCHEDULING.md)
for account assignment, slot reuse, phase boundaries, and a worked example.

1. Load unique account credentials using `load_openeo_users_from_db` and prepare
   disjoint grid or cluster partitions. Partition count need not equal user count.
   See [database account setup](CREDENTIALS.md) for the required environment,
   table columns, validation, and deterministic username ordering.
2. Call `run_parallel_extractions(..., scheduling="batches")`, the default.
   The scheduler constructs one extractor and tile plan per partition under
   `output_dir/partition_<n>/`.
3. Skip cached cubes. Pin submitted jobs to their saved `openeo_user`; put new
   jobs into the shared queue. Accounts take owned jobs before unsubmitted work.
   Older job tables without usernames infer ownership from original account order.
   Restore that original order explicitly if migrating from the environment loader.
4. Run up to `openeo_parallel_jobs` slots per account (the multi-user API accepts
   1 or 2). Database row updates are locked so concurrent jobs preserve each
   other's progress. Passwords are not persisted in local job files; they are
   retrieved from the separate PostgreSQL account table.
5. After remote downloads complete, run local partition processing with at most
   `min(account_count, partition_count)` active partition threads. Each partition
   uses its configured local `batch_workers`; same-process raster work is locked.
6. Call `merge_and_save_results` explicitly. It aligns partition columns, merges
   the saved time-statistics files, optionally reprojects, and writes the root
   `satellite_parcel_time_stats.geoparquet`.

Supplying `parcel_id_column` to the merge drops duplicate IDs, keeping the first.
Validate partitions are disjoint and all expected IDs are present; the merge
prints counts but does not enforce full input coverage. It does not merge annual
files or cleaning audits. Use a run-specific output directory so discovery does
not include results from unrelated partitions.

`scheduling="partitions"` instead runs sequential partition queues assigned
round-robin across accounts. See the [calling example](CALLING_EXAMPLE.md) and
[multi-user guide](MULTIUSER_IMPLEMENTATION_GUIDE.md).

## 11. Resume and failure handling

| Available state or failure | Next action / behavior |
| --- | --- |
| Complete partition outputs, matching manifest, and resume enabled | Reuse the saved partition; restore its cleaning report. |
| Matching batch statistics and report Parquets | Reuse tables and skip acquisition/local raster computation, even if NetCDFs were removed. |
| Complete raw cube | Skip remote computation and validate locally. Raw file existence is the acquisition completion marker. |
| Compatible final checkpoint plus audit | Skip cleaning and index calculation; recompute parcel statistics from windows. Raw metadata is still required. |
| Compatible physical checkpoint plus audit | Skip physical cleaning; rebuild indices/final checkpoint. |
| Raw only | Rebuild physical and final checkpoints locally. |
| `.partial.nc` only | Incomplete output; it is not a reusable completed cube. |
| Changed local signature | Rebuild incompatible local checkpoints. Keep raw cubes when their namespace still matches. |
| Finished manager job, missing raw cube | Restore the download by saved job ID and owning account. If unavailable, fail with diagnostic details. |
| Job retries exhausted | Inspect `tile_jobs.parquet` and `job_manager/`; resolve the backend/configuration cause before retrying. |
| Missing bands, invalid times, or absent CRS | Inspect the raw cube/configuration; do not force assumed alignment. |
| Incomplete filling | Read the audit's `final_null_count` and `has_remaining_nulls`; null statistics can be valid outputs. |
| Memory pressure | Reduce local workers and spatial extent. Keep cache files; restart the notebook kernel to release retained arrays. |

The raw-cache hash includes dates, sensor/temporal options, CRS, parcel IDs and
geometries, and several cleaning options. Manager namespaces also include tile
strategy, size and buffer. Changing some cleaning options therefore changes the
raw namespace too. Temporal/spatial window sequences and kriging variogram
settings are checked in the separate local cleaning signature. Do not assume
every cleaning change reuses the same download.

Set `keep_cleaned_checkpoint=True` to retain physical checkpoints. Final
checkpoints and their audit are normally sufficient for local restart; preserve
raw cubes as well because metadata validation precedes checkpoint reuse.

`remove_nc_after_completion=True` overrides `keep_cleaned_checkpoint=True`
after batch-table persistence. This cleanup does not require keeping raster
checkpoints for later runs with matching tabular caches. Changing processing
settings can invalidate those caches and require downloading imagery again.

### Locate a stalled or incomplete stage

| Symptom | Inspect | Interpretation |
| --- | --- | --- |
| Backend job finished but no committed raw cube | `tile_jobs.parquet` and `satellite_source.log` | Remote completion and download completion are separate. |
| `_cleaned.partial.nc` persists during a run | `satellite_parcel_statistics.log` and `satellite_raster_filling.log` | Physical-band observations/filling may still be processing. File size is not a progress percentage. |
| `_final.nc` exists but no batch statistics | `satellite_parcel_statistics.log` | Index preparation is complete; parcel reduction or table persistence remains. |
| NetCDFs disappeared but batch table pairs remain | Cleanup settings and orchestration log | Expected after successful completion cleanup. |
| Time GeoParquet exists but manifest is `processing` | Annual output, cleaning CSV, and `satellite_zonal_stats.log` | The partition has not completed all final writes. |
| Partitions completed but root GeoParquet is absent | Explicit `merge_and_save_results` call | Extraction does not perform the final multi-partition merge automatically. |
| GeoParquet contains nulls | Cleaning audit, output bands, and dated columns | Remaining gaps or empty reporting periods are valid possible outcomes. |

Read worker-suffixed logs when local process workers are enabled. For a status
check, inspect existing logs, job records, and files without rerunning notebook
cells. An old notebook output cell may lag behind current on-disk progress.

## 12. Minimal single-account example and completion checks

This example uses the base batch API and requires the repository's acquisition
dependencies plus an authenticated CDSE account. It starts remote jobs when run.

```python
from pathlib import Path
import geopandas as gpd
from data_preparation.parcel_stats import (
    OpenEOZonalStats,
    estimate_utm_epsg_from_parcels,
)

def main():
    parcels = gpd.read_file("parcels.gpkg")
    extractor = OpenEOZonalStats(
        parcels=parcels,
        parcel_id_field="parcel_id",
        start_date="2024-01-01",
        end_date="2024-12-31",
        output_dir=Path("outputs/openeo_parcel_stats"),
        working_epsg=estimate_utm_epsg_from_parcels(parcels),
        sentinel2_bands=["B04", "B08"],
        sentinel2_indices=["NDVI"],
        temporal_period="1M",
        temporal_reducer="none",
        spatial_statistics=["mean", "median", "sd", "min", "max"],
        remove_outliers=False,
        fill_nulls=False,
        batch_workers=1,
    )
    result = extractor.run()
    assert result["parcel_id"].is_unique
    assert set(result["parcel_id"]) == set(extractor.parcels["parcel_id"])
    assert result.crs.to_epsg() == extractor.working_epsg
    return result

if __name__ == "__main__":
    main()
```

Before passing outputs to feature engineering or classification, verify parcel
coverage and CRS, expected dated columns, useful median-summary columns, all
three persisted outputs, and cleaning/reliability metrics. For multi-user runs,
perform these checks after the explicit merge and review partition audits.
