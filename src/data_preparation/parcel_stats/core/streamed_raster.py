"""Bounded-memory NetCDF cleaning and index checkpoints for openEO cubes."""

from contextlib import contextmanager
from threading import RLock

import numpy as np
import pandas as pd
import xarray as xr

from shared.io import write_data

# Multi-user queues run in threads. Serialize their local raster stages while
# remote jobs continue independently; netCDF-C also requires serialized access.
LOCAL_RASTER_LOCK = RLock()
# This targets one temporal strip payload, not total RAM including working copies and masks.
TEMPORAL_BLOCK_BYTES = 8 * 1024**2


class StreamedRasterProcessing:
    """Keep raster payloads on disk and materialize only bounded working slices."""

    def _read_parcel_pixels(self, checkpoint, name, indices, variable_index, geometry, transform):
        """Read the pixels for a single parcel within the specified indices and geometry."""
        # Configured periods are contiguous on the sorted acquisition axis. Basic
        # slices avoid backend fancy-indexing copies and empty-index shape quirks.
        interval = slice(int(indices[0]), int(indices[-1]) + 1) if len(indices) else slice(0, 0)
        selected = checkpoint[name].isel(period=interval, variable=slice(variable_index, variable_index + 1))
        window, mask, _ = self._mask_parcel_window(selected, geometry, transform)
        if not len(indices):
            return np.empty((0, 1, int(mask.sum())), dtype=selected.dtype)
        return np.asarray(window)[:, :, mask]

    def _stream_parcel_statistics(self, checkpoint, geometry, transform, periods, groups, variables):
        """Read a single parcel, output band and period at a time, including provenance."""
        # Determine the mask for the parcel using an empty slice to avoid loading data.
        _, mask, _ = self._mask_parcel_window(
            checkpoint["cleaned"].isel(period=slice(0, 0), variable=slice(0, 1)), geometry, transform
        )
        pixel_count = int(mask.sum())
        # Coverage counts value slots across all stored times and outputs, not just unique spatial cells.
        expected_count = pixel_count * checkpoint.sizes["period"] * len(variables)
        observed_count = temporal_count = spatial_count = 0
        rows = []
        for period, indices in zip(periods, groups):
            row = {"period_start": period}
            for variable_index, variable in enumerate(variables):
                pixels = self._read_parcel_pixels(checkpoint, "cleaned", indices, variable_index, geometry, transform)
                observed = self._read_parcel_pixels(
                    checkpoint, "observed_mask", indices, variable_index, geometry, transform
                ).astype(bool)
                temporal = self._read_parcel_pixels(
                    checkpoint, "temporal_filled_mask", indices, variable_index, geometry, transform
                ).astype(bool)
                observed_count += int(observed.sum())
                temporal_count += int(temporal.sum())
                # Finite values with neither observation nor temporal provenance are attributed to spatial filling.
                spatial_count += int((~observed & ~temporal & np.isfinite(pixels)).sum())
                # Pool pixels across acquisitions in the reporting interval before computing parcel statistics.
                stats = self._reduce_local_pixels(pixels.reshape(1, -1))
                for statistic, values in stats.items():
                    row[f"{variable}_{statistic}"] = float(values[0])
                del pixels, observed, temporal
            rows.append(row)
        metrics = {
            "intersected_pixel_count": pixel_count,
            "expected_pixel_count": expected_count,
            "observed_count": observed_count,
            "temporal_filled_pixel_count": temporal_count,
            "spatial_filled_pixel_count": spatial_count,
            "temporal_filled_ratio": temporal_count / expected_count if expected_count else np.nan,
            "spatial_filled_ratio": spatial_count / expected_count if expected_count else np.nan,
            "data_reliability_score": observed_count / expected_count if expected_count else np.nan,
        }
        return rows, metrics

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
            chunks = (1, 1, min(256, spatial_shape[0]), min(256, spatial_shape[1]))
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
                self._stream_temporal_fill(checkpoint, allowed, stage_counts["after_temporal_fill"])
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

    def _stream_temporal_fill(self, checkpoint, allowed, null_counts):
        """Fill spatial strips using every date, with coverage guards computed globally."""
        stored = checkpoint.variables["cleaned"]
        temporal_mask = checkpoint.variables["temporal_filled_mask"]
        time_count, band_count, height, width = stored.shape
        block_rows = max(1, min(height, TEMPORAL_BLOCK_BYTES // max(1, time_count * width * 4)))
        for band_index in range(band_count):
            self.parcel_logger.info("Temporal filling band %s/%s in strips of %s rows.", band_index + 1, band_count, block_rows)
            for start in range(0, height, block_rows):
                window = slice(start, min(height, start + block_rows))
                observed = stored[:, band_index, window, :]
                source = observed.copy()
                # Low-coverage dates cannot donate temporal values; their original observations remain in observed.
                source[~allowed[:, band_index]] = np.nan
                candidate = self._fill_temporal_neighbors(source)
                filled = self._merge_filled_values(
                    observed, candidate, np.ones(observed.shape[-2:], dtype=bool), allowed[:, band_index]
                )
                added = ~np.isfinite(observed) & np.isfinite(filled)
                null_counts[:, band_index] -= added.sum(axis=(1, 2))
                stored[:, band_index, window, :] = filled
                temporal_mask[:, band_index, window, :] = added
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
