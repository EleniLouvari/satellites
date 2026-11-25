"""Bounded-memory NetCDF cleaning and index checkpoints for openEO cubes."""

from contextlib import ExitStack, contextmanager
from math import ceil, prod
from threading import RLock

import numpy as np
import pandas as pd
import xarray as xr

from shared.io import write_data

# Multi-user queues run in threads. Serialize their local raster stages while
# remote jobs continue independently; netCDF-C also requires serialized access.
LOCAL_RASTER_LOCK = RLock()
# This targets one temporal strip payload, not total RAM including working copies and masks.
TEMPORAL_BLOCK_BYTES = 64 * 1024**2
TEMPORAL_CACHE_BYTES = 64 * 1024**2
# Spatial edge of checkpoint chunks; parcels are grouped for block reads by these chunks.
CHECKPOINT_CHUNK_PIXELS = 256
# Payload target for one parcel-group read of an output and its two provenance masks. It bounds the
# read block only; parcel pixel copies, reducer temporaries and backend caches are additional memory.
PARCEL_BLOCK_BYTES = 64 * 1024**2
# Separate budget for independent boolean window masks, including overlapping parcel memberships.
PARCEL_MASK_BYTES = 16 * 1024**2
# Bytes per raster cell across the three arrays read together: float32 values plus two uint8 masks.
PARCEL_BLOCK_CELL_BYTES = 6


class StreamedRasterProcessing:
    """Keep raster payloads on disk and materialize only bounded working slices."""

    def _parcel_read_groups(self, geometries, transform, raster_shape, period_groups):
        """Group parcel positions so each group's outputs can be read as one bounded block.

        Parcels are keyed by the checkpoint chunk that holds their window's upper-left pixel, so a
        group's union window spans little more than the chunks its parcels touch anyway. Parcels
        outside the raster share one 1x1 group. A group whose union window would exceed the payload
        target even for its longest reporting period is split into single parcels. Independent masks
        have a separate summed-window budget; an oversized individual parcel is processed alone.
        """
        height, width = raster_shape
        windows, by_chunk = [], {}
        for position, geometry in enumerate(geometries):
            bounds = self._parcel_window_bounds(height, width, geometry, transform)
            # Outside parcels read the same 1x1 placeholder window that _parcel_pixel_window returns.
            windows.append(bounds or (0, 1, 0, 1))
            key = (-1, -1) if bounds is None else (bounds[0] // CHECKPOINT_CHUNK_PIXELS, bounds[2] // CHECKPOINT_CHUNK_PIXELS)
            by_chunk.setdefault(key, []).append(position)
        longest_period = max([len(indices) for indices in period_groups] + [1])
        groups = []
        # Chunk-row-major order keeps consecutive groups on neighbouring checkpoint chunks.
        for key in sorted(by_chunk):
            positions = by_chunk[key]
            cells = self._union_window_cells([windows[position] for position in positions])
            if len(positions) > 1 and longest_period * cells * PARCEL_BLOCK_CELL_BYTES > PARCEL_BLOCK_BYTES:
                groups.extend([position] for position in positions)
                continue
            # Overlaps share raster bytes but still allocate one independent mask per parcel.
            current, mask_bytes = [], 0
            for position in positions:
                r0, r1, c0, c1 = windows[position]
                required = (r1 - r0) * (c1 - c0)
                if current and mask_bytes + required > PARCEL_MASK_BYTES:
                    groups.append(current)
                    current, mask_bytes = [], 0
                current.append(position)
                mask_bytes += required
            groups.append(current)
        return groups

    @staticmethod
    def _union_window_cells(windows):
        """Return the cell count of the bounding window covering every ``(r0, r1, c0, c1)`` window."""
        rows = max(window[1] for window in windows) - min(window[0] for window in windows)
        columns = max(window[3] for window in windows) - min(window[2] for window in windows)
        return rows * columns

    @staticmethod
    def _block_time_reads(intervals, time_count, cells):
        """Return ``(time slice, period positions)`` reads for one group block.

        All stored times are read at once when the block fits the payload target; otherwise each
        reporting period is read separately, because pooled statistics need a period's times together.
        """
        if time_count * cells * PARCEL_BLOCK_CELL_BYTES <= PARCEL_BLOCK_BYTES:
            return [(slice(0, time_count), list(range(len(intervals))))]
        return [(interval, [position]) for position, interval in enumerate(intervals)]

    @staticmethod
    def _read_parcel_block(checkpoint, name, variable_index, times, rows, columns):
        """Read one output's ``(time, y, x)`` block for a parcel group in a single request."""
        return np.asarray(checkpoint[name].isel(period=times, variable=variable_index, y=rows, x=columns).values)

    def _stream_group_statistics(self, checkpoint, geometries, transform, periods, groups, variables):
        """Read each output once per parcel group and reduce every parcel's pixels from memory.

        Returns one ``(rows, metrics)`` pair per geometry, in the given order. Per-parcel arithmetic is
        identical to reading each parcel's window separately; only the number of disk requests changes.
        """
        time_count = checkpoint.sizes["period"]
        # Rasterize each parcel once within its own window; overlapping parcels keep independent masks.
        windows = [
            self._parcel_pixel_window(checkpoint.sizes["y"], checkpoint.sizes["x"], geometry, transform)
            for geometry in geometries
        ]
        block_rows, block_columns, local_windows = self._group_block_window(windows)
        intervals = self._period_intervals(groups)
        cells = (block_rows.stop - block_rows.start) * (block_columns.stop - block_columns.start)
        parcel_rows = [[{"period_start": period} for period in periods] for _ in geometries]
        counts = [{"observed": 0, "temporal": 0, "spatial": 0} for _ in geometries]
        for variable_index, variable in enumerate(variables):
            for times, positions in self._block_time_reads(intervals, time_count, cells):
                blocks = [
                    self._read_parcel_block(checkpoint, name, variable_index, times, block_rows, block_columns)
                    for name in ("cleaned", "observed_mask", "temporal_filled_mask")
                ]
                periods_read = [(position, intervals[position]) for position in positions]
                for parcel, local_window in enumerate(local_windows):
                    self._reduce_parcel_block(
                        blocks, local_window, times, periods_read, variable, parcel_rows[parcel], counts[parcel]
                    )
                del blocks
        # Coverage counts value slots across all stored times and outputs, not just unique spatial cells.
        return [
            (rows, self._parcel_fill_metrics(int(window[2].sum()), time_count * len(variables), parcel_counts))
            for rows, window, parcel_counts in zip(parcel_rows, windows, counts)
        ]

    @staticmethod
    def _period_intervals(groups):
        """Return each reporting period's stored-time slice; empty periods give an empty slice."""
        # Configured periods are contiguous on the sorted acquisition axis, so basic slices select each group.
        return [slice(int(indices[0]), int(indices[-1]) + 1) if len(indices) else slice(0, 0) for indices in groups]

    @staticmethod
    def _group_block_window(windows):
        """Return the group's union row/column slices and each parcel's window relative to that block."""
        block_rows = slice(min(window[0].start for window in windows), max(window[0].stop for window in windows))
        block_columns = slice(min(window[1].start for window in windows), max(window[1].stop for window in windows))
        local_windows = [
            (
                slice(rows.start - block_rows.start, rows.stop - block_rows.start),
                slice(columns.start - block_columns.start, columns.stop - block_columns.start),
                mask,
            )
            for rows, columns, mask, _ in windows
        ]
        return block_rows, block_columns, local_windows

    def _reduce_parcel_block(self, blocks, local_window, times, periods_read, variable, rows, counts):
        """Reduce one parcel's pixels from a group block for every reporting period covered by the read."""
        window_rows, window_columns, mask = local_window
        # The same (time, parcel_pixel) arrays a separate read of this parcel's window would give.
        pixels, observed, temporal = (block[:, window_rows, window_columns][:, mask] for block in blocks)
        observed, temporal = observed.astype(bool), temporal.astype(bool)
        for position, interval in periods_read:
            # Period intervals are absolute stored-time slices; shift them into the block's time slice.
            relative = slice(interval.start - times.start, interval.stop - times.start)
            self._reduce_parcel_period(
                rows[position], variable, (pixels[relative], observed[relative], temporal[relative]), counts
            )

    def _reduce_parcel_period(self, row, variable, arrays, counts):
        """Add one output's statistics for one reporting period to a parcel row and update provenance counts."""
        pixels, observed, temporal = arrays
        pixels = pixels.copy()
        counts["observed"] += int(observed.sum())
        counts["temporal"] += int(temporal.sum())
        # Finite values with neither observation nor temporal provenance are attributed to spatial filling.
        counts["spatial"] += int((~observed & ~temporal & np.isfinite(pixels)).sum())
        if self.require_parcel_observation:
            # A date with no observed parcel pixel would be described only by filled values
            # borrowed from other dates or neighbouring pixels, so it is excluded from the pool.
            pixels[~observed.any(axis=1)] = np.nan
        # Pool pixels across acquisitions in the reporting interval before computing parcel statistics.
        stats = self._reduce_local_pixels(pixels.reshape(1, -1))
        for statistic, values in stats.items():
            row[f"{variable}_{statistic}"] = float(values[0])

    @staticmethod
    def _parcel_fill_metrics(pixel_count, slots_per_pixel, counts):
        """Return a parcel's static coverage and fill-provenance metrics."""
        expected_count = pixel_count * slots_per_pixel
        return {
            "intersected_pixel_count": pixel_count,
            "expected_pixel_count": expected_count,
            "observed_count": counts["observed"],
            "temporal_filled_pixel_count": counts["temporal"],
            "spatial_filled_pixel_count": counts["spatial"],
            "temporal_filled_ratio": counts["temporal"] / expected_count if expected_count else np.nan,
            "spatial_filled_ratio": counts["spatial"] / expected_count if expected_count else np.nan,
            "data_reliability_score": counts["observed"] / expected_count if expected_count else np.nan,
        }

    @contextmanager
    def _new_streamed_checkpoint(self, path, periods, variables, spatial_shape):
        """Create a new streamed checkpoint for storing parcel statistics."""
        # Keep acquisition-only dependencies out of the shared source-neutral import path.
        from netCDF4 import Dataset

        with Dataset(path, "w", format="NETCDF4") as checkpoint:
            dimensions = ("period", "variable", "y", "x")
            for name, size in zip(dimensions, (len(periods), len(variables), *spatial_shape)):
                checkpoint.createDimension(name, size)
            for name, labels in (("period", periods), ("variable", variables)):
                coordinate = checkpoint.createVariable(name, str, (name,))
                coordinate[:] = np.asarray(labels, dtype=object)
            # Single-time, single-variable chunks support layer writes and small parcel-window reads.
            chunks = (1, 1, min(CHECKPOINT_CHUNK_PIXELS, spatial_shape[0]), min(CHECKPOINT_CHUNK_PIXELS, spatial_shape[1]))
            for name, dtype in (("cleaned", "f4"), ("observed_mask", "u1"), ("temporal_filled_mask", "u1")):
                variable = checkpoint.createVariable(name, dtype, dimensions, chunksizes=chunks)
                variable.set_var_chunk_cache(size=1024**2, nelems=1009, preemption=0.75)
            checkpoint.set_auto_mask(False)
            checkpoint.cleaning_signature = self._cleaning_checkpoint_signature()
            yield checkpoint

    def _streamed_checkpoint_matches(self, path, periods, variables, spatial_shape, *, physical=False):
        """Validate coordinates and shape without reading the raster payload."""
        if not path.exists():
            return False
        with xr.open_dataset(path, cache=False) as checkpoint:
            if checkpoint.attrs.get("cleaning_signature") != self._cleaning_checkpoint_signature():
                return False
            if physical and checkpoint.attrs.get("streamed_physical_complete") != 1:
                return False
            if checkpoint["period"].astype(str).values.tolist() != periods:
                raise ValueError(f"Error: Checkpoint periods do not match the raw cube: {path}")
            if checkpoint["variable"].astype(str).values.tolist() != variables:
                raise ValueError(f"Error: Checkpoint variables do not match the configured outputs: {path}")
            expected = (len(periods), len(variables), *spatial_shape)
            for name in ("cleaned", "observed_mask", "temporal_filled_mask"):
                if name not in checkpoint or checkpoint[name].shape != expected:
                    raise ValueError(f"Error: Checkpoint shape does not match the raw cube: {path}")
        return True

    def _stream_physical_bands(
        self, target, dataset, dimensions, order, periods, bands, spatial_shape, transform, cube_crs, batch_number
    ):
        """Clean one full raster layer at a time, retaining global IQR/fill semantics."""
        mask = np.ones(spatial_shape, dtype=bool)
        shape = (len(periods), len(bands))
        audit = {}
        observed_counts = np.zeros(shape, dtype="int64")
        with self._new_streamed_checkpoint(target, periods, bands, spatial_shape) as checkpoint:
            stored = checkpoint.variables["cleaned"]
            original_mask = checkpoint.variables["observed_mask"]
            temporal_mask = checkpoint.variables["temporal_filled_mask"]
            self._store_observed_physical_bands(
                dataset=dataset,
                dimensions=dimensions,
                order=order,
                periods=periods,
                bands=bands,
                batch_number=batch_number,
                mask=mask,
                stored=stored,
                original_mask=original_mask,
                temporal_mask=temporal_mask,
                observed_counts=observed_counts,
                audit=audit,
            )

            counts = mask.size - observed_counts
            stage_counts = {"after_temporal_fill": counts.copy()}
            # Determine fill eligibility from whole-layer post-IQR coverage, not from individual strip coverage.
            allowed = observed_counts / mask.size >= self.minimum_observed_fraction_for_fill
            if self.fill_nulls:
                self._stream_temporal_fill(checkpoint, allowed, stage_counts["after_temporal_fill"], observed_counts > 0)
            rows = self._clean_streamed_physical_layers(
                stored=stored,
                original_mask=original_mask,
                periods=periods,
                bands=bands,
                mask=mask,
                stage_counts=stage_counts,
                allowed=allowed,
                transform=transform,
                cube_crs=cube_crs,
                audit=audit,
                batch_number=batch_number,
            )
            # Mark complete only after every physical layer and its cleaning report have been assembled.
            checkpoint.streamed_physical_complete = 1
        return pd.concat(rows, ignore_index=True)

    def _store_observed_physical_bands(
        self,
        dataset,
        dimensions,
        order,
        periods,
        bands,
        batch_number,
        mask,
        stored,
        original_mask,
        temporal_mask,
        observed_counts,
        audit,
    ) -> None:
        """Store outlier-filtered observed values for every period/band pair."""
        time_dim, x_dim, y_dim = dimensions
        for time_index, source_index in enumerate(order):
            for band_index, band in enumerate(bands):
                selected = self._select_monthly_variables(
                    dataset.isel({time_dim: slice(source_index, source_index + 1)}),
                    time_dim,
                    x_dim,
                    y_dim,
                    expected_variables=[band],
                ).transpose(time_dim, "variable", y_dim, x_dim)
                values = np.asarray(selected.values, dtype="float32")
                bounds = self._calculate_iqr_bounds(values)
                observed, state = self._remove_iqr_outliers(
                    values, mask, batch_number, [periods[time_index]], iqr_bounds=bounds, variable_names=[band]
                )
                finite = np.isfinite(observed[0, 0])
                stored[time_index, band_index] = observed[0, 0]
                original_mask[time_index, band_index] = finite
                temporal_mask[time_index, band_index] = 0
                observed_counts[time_index, band_index] = finite.sum()
                audit[(time_index, band_index)] = state[(0, 0)]
                del selected, values, observed, finite
            if (time_index + 1) % 10 == 0 or time_index + 1 == len(periods):
                self.parcel_logger.info("Stored observations %s/%s for batch %s.", time_index + 1, len(periods), batch_number)

    def _clean_streamed_physical_layers(
        self,
        stored,
        original_mask,
        periods,
        bands,
        mask,
        stage_counts,
        allowed,
        transform,
        cube_crs,
        audit,
        batch_number,
    ) -> list[pd.DataFrame]:
        """Apply spatial/interpolation filling and build per-layer cleaning reports."""
        rows = []
        for time_index, period in enumerate(periods):
            for band_index, band in enumerate(bands):
                layer = stored[time_index, band_index]
                observed = np.where(original_mask[time_index, band_index], layer, np.nan)
                current = int(stage_counts["after_temporal_fill"][time_index, band_index])
                local_counts = {"after_temporal_fill": np.array([[current]])}
                if self.fill_nulls and allowed[time_index, band_index]:
                    for size in self.spatial_fill_window_sizes:
                        if current:
                            layer = self._fill_spatial_neighbors(layer[np.newaxis], mask, window_size=size)[0]
                            current = int((~np.isfinite(layer)).sum())
                        local_counts[f"after_{size}x{size}_fill"] = np.array([[current]])
                for size in (3, 5, 7, 9):
                    local_counts.setdefault(f"after_{size}x{size}_fill", np.array([[current]]))
                if self.fill_nulls and allowed[time_index, band_index] and current:
                    layer = self._interpolate_interval_layer(layer, mask, transform, cube_crs)
                    current = int((~np.isfinite(layer)).sum())
                local_counts["after_interpolation"] = np.array([[current]])
                stored[time_index, band_index] = layer
                report = self._build_cleaning_report(
                    layer[np.newaxis, np.newaxis],
                    observed[np.newaxis, np.newaxis],
                    mask,
                    [period],
                    {(0, 0): audit[(time_index, band_index)]},
                    stage_null_counts=local_counts,
                    variable_names=[band],
                )
                rows.append(report)
                del layer, observed
            self.parcel_logger.info("Cleaned acquisition %s/%s for batch %s.", time_index + 1, len(periods), batch_number)
        return rows

    def _temporal_block_shape(self, stored):
        """Choose chunk-aligned spatial blocks with a bounded automatic payload."""
        time_count, _, height, width = stored.shape
        chunk_rows, chunk_columns = stored.chunking()[-2:]
        block_rows = self.temporal_fill_block_rows
        if block_rows is None:
            block_rows = TEMPORAL_BLOCK_BYTES // max(1, time_count * min(width, chunk_columns) * 4)
        block_rows = max(1, min(height, block_rows))
        if block_rows >= chunk_rows:
            block_rows = (block_rows // chunk_rows) * chunk_rows
        # Even tiny row strips touch complete disk chunks; limit column groups
        # using that footprint so repeated strips can reuse the same cache.
        columns = TEMPORAL_BLOCK_BYTES // max(1, time_count * max(block_rows, chunk_rows) * 4)
        block_columns = min(width, max(chunk_columns, (columns // chunk_columns) * chunk_columns))
        return block_rows, block_columns

    @staticmethod
    def _temporal_spatial_windows(stored, block_rows, block_columns):
        """Finish sub-chunk strips before advancing to the next spatial chunk group."""
        height, width = stored.shape[-2:]
        row_span = max(block_rows, stored.chunking()[-2])
        for row_start in range(0, height, row_span):
            row_end = min(height, row_start + row_span)
            for column_start in range(0, width, block_columns):
                columns = slice(column_start, min(width, column_start + block_columns))
                for start in range(row_start, row_end, block_rows):
                    yield slice(start, min(row_end, start + block_rows)), columns

    @contextmanager
    def _temporal_chunk_caches(self, variables, block_rows, block_columns):
        """Temporarily retain each block's chunks, with a per-variable cache ceiling."""
        with ExitStack() as restore:
            for variable in variables:
                chunks = variable.chunking()
                chunk_count = ceil(variable.shape[0] / chunks[0]) * ceil(block_rows / chunks[2]) * ceil(block_columns / chunks[3])
                footprint = chunk_count * prod(chunks) * variable.dtype.itemsize
                previous = variable.get_var_chunk_cache()
                restore.callback(variable.set_var_chunk_cache, *previous)
                # Prime hash-table size avoids collisions among resident chunks.
                # LRU keeps chunks reusable between the read and subsequent write.
                variable.set_var_chunk_cache(size=min(TEMPORAL_CACHE_BYTES, max(1024**2, footprint)), nelems=10007, preemption=0)
            yield

    def _stream_temporal_fill(self, checkpoint, allowed, null_counts, active=None):
        """Fill chunk-aligned blocks across all dates, retaining global coverage guards.

        ``active`` flags, per date and band, whether the whole layer holds any
        observation. Dates without one (another sensor's acquisitions on a
        merged time axis) are skipped as temporal neighbours.
        """
        stored = checkpoint.variables["cleaned"]
        temporal_mask = checkpoint.variables["temporal_filled_mask"]
        block_rows, block_columns = self._temporal_block_shape(stored)
        band_count = stored.shape[1]
        with self._temporal_chunk_caches((stored, temporal_mask), block_rows, block_columns):
            for band_index in range(band_count):
                self.parcel_logger.info(
                    "Temporal filling band %s/%s in blocks of up to %s rows x %s columns.",
                    band_index + 1, band_count, block_rows, block_columns,
                )
                for rows, columns in self._temporal_spatial_windows(stored, block_rows, block_columns):
                    observed = stored[:, band_index, rows, columns]
                    source = observed.copy()
                    # Low-coverage dates cannot donate values; retain their original observations.
                    source[~allowed[:, band_index]] = np.nan
                    candidate = self._fill_temporal_active_steps(
                        source, None if active is None else active[:, band_index]
                    )
                    filled = self._merge_filled_values(
                        observed, candidate, np.ones(observed.shape[-2:], dtype=bool), allowed[:, band_index]
                    )
                    added = ~np.isfinite(observed) & np.isfinite(filled)
                    null_counts[:, band_index] -= added.sum(axis=(1, 2))
                    stored[:, band_index, rows, columns] = filled
                    temporal_mask[:, band_index, rows, columns] = added
                    del observed, source, candidate, filled, added

    def _stream_output_indices(self, source, target, periods, bands, variables, spatial_shape):
        """Write one acquisition/output at a time, including index fill provenance."""
        with (
            xr.open_dataset(source, cache=False) as physical,
            self._new_streamed_checkpoint(target, periods, variables, spatial_shape) as final,
        ):
            for time_index in range(len(periods)):
                cleaned = np.asarray(physical["cleaned"].isel(period=time_index).values, dtype="float32")
                states = [cleaned]
                if self.fill_nulls:
                    observed_mask = physical["observed_mask"].isel(period=time_index).values.astype(bool)
                    temporal_mask = physical["temporal_filled_mask"].isel(period=time_index).values.astype(bool)
                    # Recompute indices with observed-only and temporal-inclusive bands to establish fill provenance.
                    states.extend(
                        [np.where(observed_mask, cleaned, np.nan), np.where(observed_mask | temporal_mask, cleaned, np.nan)]
                    )
                    del observed_mask, temporal_mask
                for variable_index, variable in enumerate(variables):
                    outputs = []
                    for state in states:
                        by_band = dict(zip(bands, state))
                        if variable in bands:
                            output = by_band[variable]
                        elif variable in self.optical_indices:
                            output = self._calculate_local_optical_index(by_band, variable)
                        else:
                            output = self._calculate_local_sentinel1_index(by_band, variable)
                        outputs.append(output)
                    valid_observed = np.isfinite(outputs[1] if self.fill_nulls else outputs[0])
                    final.variables["cleaned"][time_index, variable_index] = outputs[0]
                    final.variables["observed_mask"][time_index, variable_index] = valid_observed
                    final.variables["temporal_filled_mask"][time_index, variable_index] = (
                        ~valid_observed & np.isfinite(outputs[2]) if self.fill_nulls else 0
                    )
                    del outputs, output, by_band, state
                del states, cleaned
                if (time_index + 1) % 10 == 0 or time_index + 1 == len(periods):
                    self.parcel_logger.info("Saved output indices for acquisition %s/%s.", time_index + 1, len(periods))

    def _ensure_streamed_checkpoint(
        self, netcdf_path, dataset, dimensions, order, periods, bands, variables, spatial_shape, transform, cube_crs, batch_number
    ):
        """Reuse complete checkpoints, committing newly streamed files atomically."""
        final_path = self._final_checkpoint_path(netcdf_path)
        physical_path, report_path = self._cleaning_checkpoint_paths(netcdf_path)
        if report_path.exists() and self._streamed_checkpoint_matches(final_path, periods, variables, spatial_shape):
            self.parcel_logger.info("Using completed post-index checkpoint %s without loading the full cube.", final_path)
            return final_path
        if not (
            report_path.exists()
            and self._streamed_checkpoint_matches(physical_path, periods, bands, spatial_shape, physical=True)
        ):
            temporary = physical_path.with_suffix(".partial.nc")
            report = self._stream_physical_bands(
                temporary, dataset, dimensions, order, periods, bands, spatial_shape, transform, cube_crs, batch_number
            )
            temporary_report = report_path.with_suffix(".partial.parquet")
            write_data(report, str(temporary_report))
            # Commit the report first and the closed NetCDF last; partial filenames never qualify for reuse.
            temporary_report.replace(report_path)
            temporary.replace(physical_path)
        temporary = final_path.with_suffix(".partial.nc")
        self._stream_output_indices(physical_path, temporary, periods, bands, variables, spatial_shape)
        temporary.replace(final_path)
        # Remove the intermediate physical checkpoint only after the final indexed checkpoint is committed.
        if not getattr(self, "keep_cleaned_checkpoint", False):
            self._remove_cleaned_raster_checkpoint(netcdf_path)
        return final_path
