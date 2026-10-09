# How NetCDF processing works

This guide explains the `.nc` files processed by `data_preparation`, especially
the openEO path used by `OpenEOZonalStats` and `OpenEOJobManagerZonalStats`.
Both classes use the same local processing engine. The descriptions below follow
the current implementation, including its defaults and restart behavior.

The purpose is to turn satellite raster values into a table of features for each
parcel. A NetCDF file is a container for arrays: a value is identified by its time,
band, and position on a grid. It is not yet a table with one row per parcel.

## 1. What enters the local processing stage?

A batch supplies three things: a batch number, its parcel polygons, and a raw
NetCDF path. A batch is a group of parcels processed together; a partition can
contain several batches.

For openEO, a typical raw path is:

```text
<output_dir>/monthly_cubes/<run_signature>/batch_00001_monthly.nc
```

`run_signature` is a hash of configuration and parcel identity/geometry used to
separate caches. `monthly` is a historical filename: the file can also contain
other reporting intervals or individual acquisitions.

The remote job has already selected and prepared the imagery. Sentinel-2 data
has been scaled to reflectance, resampled, and cloud/invalid-pixel masked;
Sentinel-1 data is linear-power backscatter. Depending on `temporal_reducer`,
the file contains either one composite per reporting interval (default:
`"median"`) or individual acquisition timestamps (`"none"` or `None`).
See the [full acquisition workflow](openeo/WORKFLOW.md#5-acquire-sentinel-rasters-remotely).

The raw cube contains physical bands needed for the requested outputs. For
example, requesting only NDVI still requires B08 and B04. Local processing
calculates the index from these bands; any indices in an older raw cube are
ignored. Here, **raw means before local cleaning**, not unprocessed satellite
measurements.

## 2. What does “partial” mean?

**`.partial.nc` means a write has not been committed as complete.** It is a
temporary working filename, not a percentage of parcels, a partially cloudy
image, or a particular date range.

There are three separate uses:

| Temporary filename | What is being written | Name after success |
| --- | --- | --- |
| `batch_00001_monthly.partial.nc` | Download of the raw cube | `batch_00001_monthly.nc` |
| `batch_00001_monthly_cleaned.partial.nc` | Local cleaning of physical bands | `batch_00001_monthly_cleaned.nc` |
| `batch_00001_monthly_final.partial.nc` | Requested output bands and calculated indices | `batch_00001_monthly_final.nc` |

The pattern is: write to the temporary path, finish and close the write, then
replace the destination with `Path.replace()`. Subsequent processing uses the
committed filename. For the cleaned checkpoint, its cleaning report is committed
first and the NetCDF is committed last.

A partial file during an active run is normal. It can be large and remain under
that name while many layers are processed. File size alone does not tell you how
close the stage is to completion: NetCDF allocates storage in chunks and some
stages update values already written.

If execution stops, a partial file can remain. The pipeline does **not** treat
it as a reusable completed checkpoint or resume from its last written layer.
It retries the unfinished stage using earlier completed inputs. Download paths
remove an existing partial before retrying; local checkpoint creation opens the
partial path in write mode and rebuilds it. Do not manually rename a partial
file to make it look complete.

The same convention appears in `.partial.parquet` for temporary tables and
`.partial.json` for temporary completion manifests. The suffix describes write
state, not data quality. A successfully committed file may legitimately contain
missing values (`NaN`).

## 3. The local processing sequence

```text
Complete raw .nc
    |
    v
Validate bands, dimensions, CRS, and timestamps
    |
    v
Physical bands: optional IQR filtering, then optional gap filling
    |
    v
Commit _cleaned.nc and _cleaning_report.parquet
    |
    v
Select requested bands and calculate pixel indices
    |
    v
Commit _final.nc
    |
    v
Read each parcel's pixels and calculate statistics
    |
    v
Commit batch statistics and report as Parquet
    |
    v
Assemble parcel features and final GeoParquet outputs
```

Completed checkpoints can skip stages, as explained in section 7.

### A. Open and validate the cube

The engine opens the raw file with Xarray, decoding coordinate metadata and
applying encoded masks/scales. It identifies temporal, x, and y dimensions,
checks the required physical bands, reads the raster coordinate reference
system (CRS), and aligns parcel geometries to that CRS.

Missing CRS, missing required bands, or unparseable dates stop processing.
Acquisition timestamps are sorted and must fall within the configured reporting
intervals. Composite timestamps must have unique, recognized period-start labels.
The raw metadata is still read when a cleaned or final raster checkpoint is reused.

### B. Clean physical bands

Cleaning operates on the raster before individual parcel masks are applied.
In this streamed path, the eligible area is the full rectangular raster grid.

1. **Read observations.** Read one time slice and one physical band, convert to
   `float32`, and record missing/valid values.
2. **Optionally remove outliers.** With `remove_outliers=True`, calculate the
   interquartile range (IQR) independently for each band/time layer across its
   finite raster values. Defaults use Q25 and Q75 and remove values outside
   `Q25 - 1.5 * IQR` through `Q75 + 1.5 * IQR`, replacing them with `NaN`.
   Filtering is skipped below `iqr_min_valid_pixels` (default 20).
3. **Check fill eligibility.** After IQR filtering, the layer must have at least
   `minimum_observed_fraction_for_fill` observed coverage (default 0.20) to
   receive fills or supply temporal observations. The denominator is all cells
   in the layer, not only cells within parcels. Existing observations in an
   ineligible layer remain available for statistics.
4. **Optionally fill through time.** With `fill_nulls=True`, fill gaps at the
   same spatial pixel using eligible observations at other times. Default
   `past_only` uses the nearest previous observation. `bidirectional` uses the
   average of the nearest previous and next observations when both exist, or
   the available side otherwise; this is not a time-distance-weighted average.
   Optional `temporal_fill_window_sizes` limit the search in time steps, not
   calendar days. The default `None` imposes no window limit.
5. **Fill from spatial neighbors.** Apply mean neighborhoods sequentially,
   defaulting to 3x3 then 5x5. These fill missing cells and retain existing
   finite values. Earlier fills can supply later spatial passes.
6. **Interpolate remaining gaps.** Use the configured `nearest`, `idw`, or
   `kriging` method. Default nearest filling uses the nearest finite raster
   cell and has no maximum-distance cutoff in this implementation. Unfilled
   values remain missing.

Both `remove_outliers` and `fill_nulls` default to **False**. When disabled,
their stages do not alter observations, but the pipeline still writes
checkpoints and provenance for subsequent processing.

Spatial filling can use values across parcel boundaries. Temporal filling can
cross reporting-period boundaries. These operations estimate missing values;
the provenance records distinguish them from observed values.

### C. Save cleaned bands and calculate indices

The intermediate `_cleaned.nc` contains physical bands after the enabled
cleaning stages. Its companion `_cleaning_report.parquet` records raster-level
counts and IQR information for each batch/time/band, including null counts after
the filling stages.

The next stage produces `_final.nc` containing only requested output bands and
indices. For example:

```text
NDVI at each pixel = (cleaned B08 - cleaned B04) / (cleaned B08 + cleaned B04)
```

Invalid divisions remain missing. Extra bands downloaded only to calculate an
index do not become standalone output features unless requested.

Both checkpoint types contain arrays with dimensions `(period, variable, y, x)`:

| Array | Meaning |
| --- | --- |
| `cleaned` (`float32`) | Values at this checkpoint stage, including any fills. In `_final.nc`, this includes indices. |
| `observed_mask` (`uint8`) | Value available from post-IQR observations without filling. |
| `temporal_filled_mask` (`uint8`) | Value made available by temporal filling. |

A finite final value with neither mask set is attributed to spatial filling,
including interpolation. For an index, provenance is determined by evaluating
its formula with observed-only and observed-plus-temporally-filled inputs.
The `period` axis contains acquisition timestamps when acquisitions are retained.

The final NetCDF is a processing checkpoint; **final here means ready for parcel
aggregation**, not that the final parcel table has already been written.

### D. Calculate each parcel's statistics

For each parcel, the engine reads its bounding window from `_final.nc` and
rasterizes the polygon with `all_touched=True`. Boundary-intersecting cells
participate, with no fractional-area weighting.

For each output variable and reporting period, it reduces the finite values
using the configured statistics: mean, median, sample standard deviation, min,
max, range, and/or percentiles. Mean is always included. Empty samples give
missing statistics; sample SD requires two values. Although `"count"` is accepted
in configuration, it is removed from the per-variable reducer list; the public
pipeline emits the static parcel counts described in section 4 instead.

With a temporal composite, the sample consists of that composite's parcel
pixels. With `temporal_reducer="none"`, it pools pixel observations from all
acquisitions in the reporting interval, separately for each variable. For
example, samples `[1, 2, 3, 4]` and `[10]` yield a pooled mean of 4 and median of
3. Acquisitions with more valid pixels contribute more observations.

Indices are calculated before parcel reduction. Thus a parcel's mean NDVI is
the mean of pixel NDVI values, not NDVI calculated from parcel-mean bands.

### E. Persist batch tables and assemble outputs

Each completed batch saves its statistics and cleaning audit as a pair of
signature-specific Parquet files. The report commits first and the statistics
file commits last. These tables allow later runs to skip raster processing.

The partition then combines batch tables, adds geometry and derived statistics,
and pivots dated features into one row per parcel, such as
`NDVI_mean__20240101`. Parcel attributes and projected geometry are attached.
Missing reporting-period feature columns are padded with nulls.

Final outputs are:

- `satellite_parcel_time_stats.geoparquet`: dated features and static parcel fields.
- `satellite_parcel_annual_stats.geoparquet`: summaries of dated parcel medians
  over the whole run; request spatial `median` for these summaries. Despite the
  name, a multi-year run is not split into separate calendar years.
- `satellite_pixel_cleaning_report.csv`: the combined physical-band cleaning audit.
- `satellite_partition_completion.json`: marked `complete` after those outputs
  are saved, for partition resume checks.

## 4. What do the quality counts mean?

`intersected_pixel_count` counts spatial cells in the parcel mask, independent
of whether their values are valid. The other counts cover all stored time steps
and requested output variables for the parcel:

```text
expected_pixel_count = intersected_pixel_count * stored_times * output_variables
data_reliability_score = observed_count / expected_pixel_count
temporal_filled_ratio = temporal_filled_pixel_count / expected_pixel_count
spatial_filled_ratio = spatial_filled_pixel_count / expected_pixel_count
```

For example, 10 cells, 3 stored times, and 2 output variables give 60 expected
value slots. If 36 are observed, 12 temporally filled, and 6 spatially filled,
the respective ratios are 0.60, 0.20, and 0.10; 6 slots remain missing.

These are static whole-cube metrics retained once per parcel, not separate
metrics for each dated column. Reliability measures observed coverage, not
classification accuracy. Missing acquisitions absent from the cube do not add
time slots to the denominator. A shared multisensor time axis can include slots
where one sensor has no observation. Zero expected slots produce null ratios.

`minimum_parcel_pixels` defaults to 3. It sets the
`meets_minimum_pixel_count` flag; small parcels are retained in the results.

## 5. Why does processing take time or use several large files?

The code keeps raster checkpoints on disk to avoid loading the entire
time-by-band cube at once. It reads individual band/time layers for cleaning,
spatial strips across all dates for temporal filling, and all physical bands
for one time slice during index creation. Parcel statistics read one parcel,
variable, and reporting period at a time.

This reduces memory demand but still requires complete spatial layers and
temporary arrays. The temporal strip's nominal 8 MiB target is not a total RAM
limit. Exact medians/percentiles also require a parcel's sample pool in memory.
Raster checkpoints are chunked and are not compressed by this writer.

Raw, cleaned, and final files can coexist while processing. Within one process,
`LOCAL_RASTER_LOCK` serializes local raster work across threads. Separate worker
processes have separate memory requirements. With
`remove_nc_after_completion=True`, the direct batch-processing path finishes,
saves, and releases batches sequentially before loading tables for assembly.

## 6. How to read the progress messages

These messages appear in `satellite_parcel_statistics.log`, or its worker log:

| Message | Current stage |
| --- | --- |
| `Reading temporal pixel cube ...` | Opening and validating the raw cube. |
| `Stored observations X/Y ...` | Physical band/time layers have been written after optional IQR filtering; filling can still remain. |
| `Temporal filling band X/Y in strips ...` | Filling one band's time series across spatial strips. |
| `Cleaned acquisition X/Y ...` | Spatial filling/interpolation and audit assembly for that time slice. “Acquisition” can also refer to a composite slice here. |
| `Saved output indices for acquisition X/Y` | Requested output layers are being written into the final partial checkpoint. |
| `Using completed post-index checkpoint ...` | Reusing `_final.nc` and proceeding to parcel statistics. |
| `Calculated ... locally for ... parcels ...` | Raster aggregation has finished; durable batch-table writes follow. |

These counters describe the current stage, not an overall completion percentage.

## 7. Restart and cleanup behavior

| Available files | What the pipeline can reuse |
| --- | --- |
| Matching batch statistics **and** report Parquets | Skip local raster computation, even if the NetCDFs were deleted. |
| Raw cube plus compatible `_final.nc` and cleaning report | Skip cleaning and index calculation; calculate parcel statistics. |
| Raw cube plus compatible `_cleaned.nc` and cleaning report | Skip physical-band cleaning; rebuild the final checkpoint and statistics. |
| Raw cube only | Redo local cleaning, indices, and statistics. |
| Only `.partial.nc` for a stage | Retry that unfinished stage; partial contents are not reused. |
| Complete partition outputs and a matching completion manifest | With `resume_completed_partitions=True`, reuse the completed partition. |

Raster checkpoint reuse checks the cleaning signature, time labels, variables,
and array shapes. Physical checkpoints also require their completion attribute.
A changed cleaning signature causes rebuilding; inconsistent labels/shapes can
raise an error. A committed raw filename is initially checked by existence,
not by a checksum; local opening and validation can still fail on a bad file.

Batch tables are named like
`batch_00001_monthly_stats_<statistics_signature>.parquet` and
`batch_00001_monthly_stats_<statistics_signature>_report.parquet`. Both must exist;
partial tables do not qualify. Reading them performs Parquet validation.

Cleanup options have different scopes:

- `keep_cleaned_checkpoint=False` (default): remove the intermediate
  `_cleaned.nc` after `_final.nc` commits. Keep its cleaning report.
- `remove_nc_after_completion=False` (default): retain raw and final NetCDFs.
- `remove_nc_after_completion=True`: after both batch tables are persisted,
  remove that batch's raw, cleaned, final, and corresponding partial NetCDFs.
  This overrides `keep_cleaned_checkpoint=True`. Tables and reports remain.

Calculation or table-write failures prevent this completion cleanup, preserving
available raster inputs. An absent `.nc` after successful cleanup is expected.
If processing options change after cleanup, the saved tables may no longer
match and raw data may need downloading again. Some cleaning options also form
part of the raw-cache signature, so not every settings change reuses downloads.

## 8. Where this is implemented

Paths are relative to the repository:

- [core/base.py](../../src/data_preparation/parcel_stats/core/base.py): batch processing, checkpoint persistence order, and worker execution.
- [core/parcel_statistics/](../../src/data_preparation/parcel_stats/core/parcel_statistics): raw validation, index formulas, parcel masks, reducers, checkpoint orchestration, and output reshaping.
- [core/streamed_raster.py](../../src/data_preparation/parcel_stats/core/streamed_raster.py): active streamed processing, temporary NetCDF writes, provenance, and checkpoint reuse.
- [core/raster_cleaning.py](../../src/data_preparation/parcel_stats/core/raster_cleaning.py): IQR filtering and temporal/spatial filling algorithms.
- [core/batch_cache.py](../../src/data_preparation/parcel_stats/core/batch_cache.py): durable batch tables and optional NetCDF deletion.
- [sources/openeo/cubes.py](../../src/data_preparation/sources/openeo/cubes.py) and [parcel_stats/job_manager.py](../../src/data_preparation/parcel_stats/job_manager.py): remote acquisition and raw partial downloads.
- [parcel_stats/openeo.py](../../src/data_preparation/parcel_stats/openeo.py): final output assembly and partition completion.

For account scheduling and cleanup timing, see [Scheduling](openeo/SCHEDULING.md).
Planet also creates temporary NetCDFs and uses shared processing components,
but its acquisition paths and scratch-file lifecycle differ; see
[Planet processing](PLANET.md).
