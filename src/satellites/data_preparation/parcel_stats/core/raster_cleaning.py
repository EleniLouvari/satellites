"""Raster loading, outlier removal, null filling, and cleaning audits.

These helpers provide a staged cleaning pipeline applied to downloaded
NetCDF cubes: IQR-based outlier masking, temporal neighbor fills, spatial
mean fills (3x3..9x9), and optional geospatial interpolation (IDW/kriging).
Functions return lightweight audit reports to trace every fill step.
"""

from __future__ import annotations

from collections.abc import Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from satellites.shared.data_cleaning import fill_null_values_using_interpolation
from rasterio.transform import xy
from scipy.ndimage import distance_transform_edt
from shapely.geometry import Point


class RasterCleaner:
    """Provide raster-level cleaning operations for the zonal-statistics pipeline.

    The concrete orchestrator supplies validated configuration attributes such
    as ``remove_outliers``, ``temporal_fill_mode`` and interpolation options.
    """

    def _find_cube_dimension(self, dataset: xr.Dataset, candidates: Sequence[str], role: str) -> str:
        """Find a cube dimension using common openEO/NetCDF dimension names."""
        dimensions = {name.lower(): name for name in dataset.dims}
        # Preserve source spelling because later xarray operations require the exact dimension label.
        for candidate in candidates:
            if candidate.lower() in dimensions:
                return dimensions[candidate.lower()]
        raise ValueError(
            f"Could not identify the {role} dimension in the NetCDF cube. Tried candidates: {candidates}. Available dimensions: {list(dataset.dims)}"
        )

    def _select_monthly_variables(
        self,
        dataset: xr.Dataset,
        time_dimension: str,
        x_dimension: str,
        y_dimension: str,
        expected_variables: Sequence[str] | None = None,
    ) -> xr.DataArray:
        """Return requested bands and indices as one labelled xarray DataArray."""
        # Resolve the target variable ordering deterministically either from
        # separate data variables or from a labelled variable dimension.
        expected = list(expected_variables or self._output_sensor_variables())
        variable_lookup = {name.upper(): name for name in dataset.data_vars}
        # Prefer the common layout where each requested band or index is a separate data variable.
        if all(name.upper() in variable_lookup for name in expected):
            arrays = [dataset[variable_lookup[name.upper()]] for name in expected]
            return xr.concat(arrays, dim=xr.IndexVariable("variable", expected), join="exact")

        # Some backends instead place all bands under one additional labelled dimension.
        core_dimensions = {time_dimension, x_dimension, y_dimension}
        for data_variable in dataset.data_vars.values():
            for dimension in set(data_variable.dims).difference(core_dimensions):
                if dimension not in data_variable.coords:
                    continue
                labels = [str(value) for value in data_variable[dimension].values]
                label_lookup = {label.upper(): label for label in labels}
                if all(name.upper() in label_lookup for name in expected):
                    selected = data_variable.sel({dimension: [label_lookup[name.upper()] for name in expected]})
                    if dimension != "variable":
                        selected = selected.rename({dimension: "variable"})
                    return selected.assign_coords(variable=expected)

        raise ValueError(
            "The monthly NetCDF does not contain the expected output variables "
            f"{expected}. Available data variables: {list(dataset.data_vars)}"
        )

    def _calculate_iqr_bounds(self, values: np.ndarray) -> dict[tuple[int, int], dict]:
        """Calculate one IQR range per interval and variable across the raster."""
        # Compute per-interval/variable IQR bounds so outlier removal does
        # not leak statistics across time or spectral bands.
        bounds = {}
        # Keep bounds independent by time and variable so seasonal distributions cannot leak across layers.
        for time_index in range(values.shape[0]):
            for variable_index in range(values.shape[1]):
                layer = values[time_index, variable_index]
                finite_values = layer[np.isfinite(layer)]
                applied = self.remove_outliers and finite_values.size >= self.iqr_min_valid_pixels
                lower_bound = upper_bound = np.nan
                if applied:
                    q1, q3 = np.quantile(finite_values, self.iqr_quantiles)
                    iqr = q3 - q1
                    lower_bound = float(q1 - self.iqr_multiplier * iqr)
                    upper_bound = float(q3 + self.iqr_multiplier * iqr)
                bounds[(time_index, variable_index)] = {
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                    "iqr_applied": bool(applied),
                    "valid_count": int(finite_values.size),
                }
        return bounds

    def _remove_iqr_outliers(
        self,
        values: np.ndarray,
        eligible_mask: np.ndarray,
        batch_number: int,
        periods: Sequence[str],
        iqr_bounds: dict[tuple[int, int], dict] | None = None,
        variable_names: Sequence[str] | None = None,
    ) -> tuple[np.ndarray, dict[tuple[int, int], dict]]:
        """Apply shared interval-variable IQR bounds to eligible raster pixels."""
        cleaned = values.copy()
        # Pixels outside the extraction footprint never participate in cleaning or filling.
        cleaned[:, :, ~eligible_mask] = np.nan
        variables = list(variable_names or self._output_sensor_variables())
        eligible_count = int(eligible_mask.sum())
        audit_state = {}

        for time_index, period_start in enumerate(periods):
            for variable_index, variable in enumerate(variables):
                layer = cleaned[time_index, variable_index]
                finite_values = layer[eligible_mask]
                finite_values = finite_values[np.isfinite(finite_values)]
                original_null_count = eligible_count - finite_values.size
                bounds = (iqr_bounds or {}).get((time_index, variable_index))
                lower_bound = bounds["lower_bound"] if bounds else np.nan
                upper_bound = bounds["upper_bound"] if bounds else np.nan
                outlier_count = 0
                iqr_applied = bool(bounds and bounds["iqr_applied"])
                if iqr_applied:
                    outliers = np.isfinite(layer) & eligible_mask & ((layer < lower_bound) | (layer > upper_bound))
                    outlier_count = int(outliers.sum())
                    layer[outliers] = np.nan
                    cleaned[time_index, variable_index] = layer

                audit_state[(time_index, variable_index)] = {
                    "batch_number": batch_number,
                    "period_start": period_start,
                    "variable": variable,
                    "eligible_values": eligible_count,
                    "original_null_count": original_null_count,
                    "iqr_outlier_count": outlier_count,
                    "iqr_lower_bound": lower_bound,
                    "iqr_upper_bound": upper_bound,
                    "iqr_applied": iqr_applied,
                    "iqr_scope": "batch_interval_variable",
                    "iqr_reference_value_count": bounds["valid_count"] if bounds else 0,
                }
        return cleaned, audit_state

    def _fill_temporal_neighbors(self, variable_values: np.ndarray) -> np.ndarray:
        """Fill temporal gaps independently at each spatial pixel."""
        window_sizes = getattr(self, "temporal_fill_window_sizes", None)
        if window_sizes is not None:
            return self._fill_bounded_temporal_neighbors(variable_values, window_sizes)
        filled = variable_values.copy()
        forward = filled.copy()
        # Carry observations forward first; this is the complete behavior in past-only mode.
        for time_index in range(1, forward.shape[0]):
            missing = ~np.isfinite(forward[time_index])
            forward[time_index][missing] = forward[time_index - 1][missing]

        backward = filled.copy()
        if self.temporal_fill_mode == "bidirectional":
            # Future observations are eligible only when bidirectional filling is explicitly enabled.
            for time_index in range(backward.shape[0] - 2, -1, -1):
                missing = ~np.isfinite(backward[time_index])
                backward[time_index][missing] = backward[time_index + 1][missing]

        missing = ~np.isfinite(filled)
        forward_valid = np.isfinite(forward)
        backward_valid = np.isfinite(backward)
        both = missing & forward_valid & backward_valid
        # When both directions exist, use their midpoint instead of favoring either temporal direction.
        filled[both] = (forward[both] + backward[both]) / 2.0
        filled[missing & forward_valid & ~backward_valid] = forward[missing & forward_valid & ~backward_valid]
        filled[missing & ~forward_valid & backward_valid] = backward[missing & ~forward_valid & backward_valid]
        return filled

    def _fill_bounded_temporal_neighbors(self, variable_values: np.ndarray, window_sizes: tuple[int, ...]) -> np.ndarray:
        """Retry gaps with wider windows without propagating imputed values."""
        filled = variable_values.copy()
        for size in window_sizes:
            forward = variable_values.copy()
            backward = variable_values.copy()
            radius = min(size // 2, variable_values.shape[0] - 1)
            # Search nearest first, always reading original observations.
            for offset in range(1, radius + 1):
                source = variable_values[:-offset]
                target = forward[offset:]
                np.copyto(target, source, where=~np.isfinite(target) & np.isfinite(source))
                if self.temporal_fill_mode == "bidirectional":
                    source = variable_values[offset:]
                    target = backward[:-offset]
                    np.copyto(target, source, where=~np.isfinite(target) & np.isfinite(source))

            missing = ~np.isfinite(filled)
            forward_valid = np.isfinite(forward)
            backward_valid = np.isfinite(backward)
            both = missing & forward_valid & backward_valid
            filled[both] = (forward[both] + backward[both]) / 2.0
            use_forward = missing & forward_valid & ~backward_valid
            use_backward = missing & ~forward_valid & backward_valid
            filled[use_forward] = forward[use_forward]
            filled[use_backward] = backward[use_backward]
        return filled

    def _fill_spatial_neighbors(self, variable_values: np.ndarray, eligible_mask: np.ndarray, window_size: int = 3) -> np.ndarray:
        """Fill each interval once with the mean of valid local neighbors."""
        if window_size < 1 or window_size % 2 == 0:
            raise ValueError("window_size must be a positive odd integer.")

        filled = variable_values.copy()
        height, width = eligible_mask.shape
        padding = window_size // 2
        # Compute neighborhood sums/counts via padded shifts to avoid
        # per-pixel Python loops for improved performance on large rasters.
        for time_index in range(filled.shape[0]):
            layer = filled[time_index]
            valid = np.isfinite(layer) & eligible_mask
            padded_values = np.pad(np.where(valid, layer, 0.0), padding, mode="constant", constant_values=0.0)
            padded_valid = np.pad(valid.astype("int16"), padding, mode="constant", constant_values=0)
            # Shift padded arrays over the extent to compute neighborhoods without a per-pixel loop.
            neighbor_sum = np.zeros((height, width), dtype="float64")
            neighbor_count = np.zeros((height, width), dtype="int16")
            for y_offset in range(window_size):
                for x_offset in range(window_size):
                    neighbor_sum += padded_values[y_offset : y_offset + height, x_offset : x_offset + width]
                    neighbor_count += padded_valid[y_offset : y_offset + height, x_offset : x_offset + width]

            neighbor_mean = np.full((height, width), np.nan, dtype="float64")
            np.divide(neighbor_sum, neighbor_count, out=neighbor_mean, where=neighbor_count > 0)
            fillable = ~np.isfinite(layer) & eligible_mask & (neighbor_count > 0)
            layer[fillable] = neighbor_mean[fillable]
            filled[time_index] = layer
        return filled

    def _eligible_pixel_geometry(self, eligible_mask, transform) -> tuple:
        """Return eligible raster indexes and their pixel-center points."""
        rows, columns = np.nonzero(eligible_mask)
        # Geospatial interpolators consume projected point geometry at raster-cell centers.
        x_coordinates, y_coordinates = xy(transform, rows, columns, offset="center")
        points = [Point(x_coordinate, y_coordinate) for x_coordinate, y_coordinate in zip(x_coordinates, y_coordinates)]
        return rows, columns, points

    def _interpolate_interval_layer(self, layer: np.ndarray, eligible_mask: np.ndarray, transform, cube_crs) -> np.ndarray:
        """Apply the configured geospatial interpolator to one raster layer."""
        if self.interpolation_method == "nearest":
            return self._fill_nearest_raster(layer, eligible_mask, transform)

        rows, columns, points = self._eligible_pixel_geometry(eligible_mask, transform)
        if rows.size == 0:
            return layer
        pixel_values = layer[rows, columns]
        missing = ~np.isfinite(pixel_values)
        # Interpolation requires both a gap to fill and at least one observed source pixel.
        if not missing.any() or not np.isfinite(pixel_values).any():
            return layer

        pixels = gpd.GeoDataFrame({"pixel_value": pixel_values}, geometry=points, crs=cube_crs)
        interpolated = fill_null_values_using_interpolation(
            pixels,
            method=self.interpolation_method,
            fill_col_name="pixel_value",
            null_value=np.nan,
            max_distance_in_meters=self.interpolation_max_distance_in_meters,
            variogram_lags=self.interpolation_variogram_lags,
            variogram_lags_max_dist_in_meters=self.interpolation_variogram_max_distance_in_meters,
            plot=False,
        )
        result = layer.copy()
        result[rows, columns] = interpolated["pixel_value"].to_numpy()
        return result

    def _fill_nearest_raster(self, layer: np.ndarray, eligible_mask: np.ndarray, transform) -> np.ndarray:
        """Fill raster gaps from the nearest observed pixel without constructing geometries."""
        valid = np.isfinite(layer) & eligible_mask
        targets = ~np.isfinite(layer) & eligible_mask
        if not targets.any() or not valid.any():
            return layer

        # Sampling preserves projected nearest-distance behavior for non-square raster pixels.
        sampling = (abs(float(transform.e)), abs(float(transform.a))) if transform is not None else None
        _, nearest_indexes = distance_transform_edt(
            ~valid, sampling=sampling, return_distances=True, return_indices=True
        )
        result = layer.copy()
        result[targets] = layer[nearest_indexes[0][targets], nearest_indexes[1][targets]]
        return result

    def _log_fill_stage_start(
        self,
        stage: str,
        variable_name: str | None,
        periods: Sequence[str] | None,
        batch_number: int | None,
        null_counts: np.ndarray,
        fill_allowed: np.ndarray,
    ) -> None:
        """Log the null count before one raster-filling stage."""

        logger = getattr(self, "filling_logger", None) or getattr(self, "logger", None)
        if logger is None or variable_name is None or periods is None:
            return
        for time_index, period_start in enumerate(periods):
            if fill_allowed[time_index]:
                logger.info(
                    "Starting %s for batch %s, variable %s, period %s: %s null pixels.",
                    stage,
                    batch_number,
                    variable_name,
                    period_start,
                    int(null_counts[time_index]),
                )

    def _log_fill_stage_finished(
        self,
        stage: str,
        variable_name: str | None,
        periods: Sequence[str] | None,
        batch_number: int | None,
        before_counts: np.ndarray,
        after_counts: np.ndarray,
        fill_allowed: np.ndarray,
    ) -> None:
        """Log pixels filled and nulls remaining after one raster stage."""

        logger = getattr(self, "filling_logger", None) or getattr(self, "logger", None)
        if logger is None or variable_name is None or periods is None:
            return
        for time_index, period_start in enumerate(periods):
            if fill_allowed[time_index]:
                filled_count = int(before_counts[time_index] - after_counts[time_index])
                logger.info(
                    "Finished %s for batch %s, variable %s, period %s: filled %s pixels; %s null pixels remain.",
                    stage,
                    batch_number,
                    variable_name,
                    period_start,
                    filled_count,
                    int(after_counts[time_index]),
                )

    def _merge_filled_values(
        self, original: np.ndarray, candidate: np.ndarray, eligible_mask: np.ndarray, fill_allowed: np.ndarray
    ) -> np.ndarray:
        """Merge permitted fill values into the original cube and mask ineligible pixels."""
        result = original.copy()
        # Never overwrite observations or fill periods rejected by the minimum-coverage guard.
        fillable = (
            ~np.isfinite(original)
            & eligible_mask[np.newaxis, :, :]
            & fill_allowed[:, np.newaxis, np.newaxis]
        )
        result[fillable] = candidate[fillable]
        result[:, ~eligible_mask] = np.nan
        return result

    def _count_visible_nulls(
        self, original: np.ndarray, candidate: np.ndarray, eligible_mask: np.ndarray, fill_allowed: np.ndarray
    ) -> np.ndarray:
        """Count nulls after applying only fills permitted for each period."""
        visible = self._merge_filled_values(original, candidate, eligible_mask, fill_allowed)
        # Spatial boolean indexing leaves one flattened pixel vector per period.
        return (~np.isfinite(visible[:, eligible_mask])).sum(axis=1).astype("int64")

    def _run_fill_stage(
        self,
        stage: str,
        filled: np.ndarray,
        fill_operation,
        original: np.ndarray,
        eligible_mask: np.ndarray,
        fill_allowed: np.ndarray,
        before_counts: np.ndarray,
        variable_name: str | None,
        periods: Sequence[str] | None,
        batch_number: int | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run and log one cube-wide filling stage."""
        self._log_fill_stage_start(stage, variable_name, periods, batch_number, before_counts, fill_allowed)
        filled = fill_operation(filled)
        after_counts = self._count_visible_nulls(original, filled, eligible_mask, fill_allowed)
        self._log_fill_stage_finished(
            stage, variable_name, periods, batch_number, before_counts, after_counts, fill_allowed
        )
        return filled, after_counts

    def _run_interpolation_stage(
        self,
        filled: np.ndarray,
        original: np.ndarray,
        eligible_mask: np.ndarray,
        fill_allowed: np.ndarray,
        before_counts: np.ndarray,
        transform,
        cube_crs,
        variable_name: str | None,
        periods: Sequence[str] | None,
        batch_number: int | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Interpolate only periods that are permitted and still contain nulls."""
        stage = f"{self.interpolation_method} interpolation"
        after_counts = before_counts.copy()
        # Completed and low-coverage periods bypass the expensive geospatial interpolator.
        for time_index in np.flatnonzero(fill_allowed & (before_counts > 0)):
            single_period = periods[time_index : time_index + 1] if periods is not None else None
            single_allowed = np.array([True])
            before_count = before_counts[time_index : time_index + 1]
            self._log_fill_stage_start(stage, variable_name, single_period, batch_number, before_count, single_allowed)
            filled[time_index] = self._interpolate_interval_layer(
                filled[time_index], eligible_mask, transform, cube_crs
            )
            after_count = self._count_visible_nulls(
                original[time_index : time_index + 1],
                filled[time_index : time_index + 1],
                eligible_mask,
                single_allowed,
            )
            after_counts[time_index] = after_count[0]
            self._log_fill_stage_finished(
                stage, variable_name, single_period, batch_number, before_count, after_count, single_allowed
            )
        return filled, after_counts

    def _fill_variable_cube(
        self,
        variable_values: np.ndarray,
        eligible_mask: np.ndarray,
        transform,
        cube_crs,
        return_null_counts: bool = False,
        variable_name: str | None = None,
        periods: Sequence[str] | None = None,
        batch_number: int | None = None,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
        """Run temporal, 3x3, 5x5, and configured interpolation filling."""
        if periods is not None and len(periods) != variable_values.shape[0]:
            raise ValueError(f"Expected {variable_values.shape[0]} period labels for filling, received {len(periods)}.")
        eligible_count = int(eligible_mask.sum())
        if eligible_count == 0:
            logger = getattr(self, "filling_logger", None) or getattr(self, "logger", None)
            if logger is not None and variable_name is not None:
                logger.info(
                    "Skipping raster filling for batch %s, variable %s: the raster has no eligible pixels.",
                    batch_number,
                    variable_name,
                )
            result = variable_values.copy()
            empty_counts = np.zeros(variable_values.shape[0], dtype="int64")
            counts = {
                "after_temporal_fill": empty_counts.copy(),
                "after_3x3_fill": empty_counts.copy(),
                "after_5x5_fill": empty_counts.copy(),
                "after_7x7_fill": empty_counts.copy(),
                "after_9x9_fill": empty_counts.copy(),
                "after_interpolation": empty_counts.copy(),
            }
            return (result, counts) if return_null_counts else result

        observed_counts = np.isfinite(variable_values[:, eligible_mask]).sum(axis=1)
        # Rejected slices are also hidden as temporal sources to prevent weak periods contaminating others.
        fill_allowed = observed_counts / eligible_count >= self.minimum_observed_fraction_for_fill
        initial_null_counts = (eligible_count - observed_counts).astype("int64")
        fill_source = variable_values.copy()
        fill_source[~fill_allowed] = np.nan

        logger = getattr(self, "filling_logger", None) or getattr(self, "logger", None)
        if logger is not None and variable_name is not None and periods is not None:
            for time_index, period_start in enumerate(periods):
                if not fill_allowed[time_index]:
                    logger.info(
                        "Skipping raster filling for batch %s, variable %s, period %s: "
                        "observed fraction %.4f is below the minimum %.4f; %s null pixels remain.",
                        batch_number,
                        variable_name,
                        period_start,
                        observed_counts[time_index] / eligible_count,
                        self.minimum_observed_fraction_for_fill,
                        int(initial_null_counts[time_index]),
                    )

        temporal_stage = f"temporal fill ({self.temporal_fill_mode})"
        filled, current_counts = self._run_fill_stage(
            temporal_stage,
            fill_source,
            self._fill_temporal_neighbors,
            variable_values,
            eligible_mask,
            fill_allowed,
            initial_null_counts,
            variable_name,
            periods,
            batch_number,
        )
        null_counts = {"after_temporal_fill": current_counts}

        window_sizes = getattr(self, "spatial_fill_window_sizes", (3, 5))
        spatial_stages = tuple(
            (f"{size}x{size} spatial fill", f"after_{size}x{size}_fill", size) for size in window_sizes
        )
        # Counts are carried forward when early completion skips a later stage, preserving the audit schema.
        for stage, audit_key, window_size in spatial_stages:
            if current_counts.any():
                def operation(values, size=window_size):
                    return self._fill_spatial_neighbors(values, eligible_mask, window_size=size)

                filled, current_counts = self._run_fill_stage(
                    stage, filled, operation, variable_values, eligible_mask, fill_allowed,
                    current_counts, variable_name, periods, batch_number
                )
            null_counts[audit_key] = current_counts.copy()

        # Optional larger windows retain carried-forward counts when they are not configured.
        for window_size in (3, 5, 7, 9):
            null_counts.setdefault(f"after_{window_size}x{window_size}_fill", current_counts.copy())

        if current_counts.any():
            filled, current_counts = self._run_interpolation_stage(
                filled, variable_values, eligible_mask, fill_allowed, current_counts, transform, cube_crs,
                variable_name, periods, batch_number
            )
        null_counts["after_interpolation"] = current_counts.copy()

        result = self._merge_filled_values(variable_values, filled, eligible_mask, fill_allowed)
        return (result, null_counts) if return_null_counts else result

    def _fill_temporal_cube_only(self, observed: np.ndarray, eligible_mask: np.ndarray) -> np.ndarray:
        """Return the cube state immediately after temporal filling.

        This reproduces the temporal stage of :meth:`_fill_variable_cube`
        without running any spatial operation.  Keeping this intermediate
        state lets parcel aggregation attribute every imputed output pixel to
        either temporal or spatial filling.
        """
        if not getattr(self, "fill_nulls", False):
            return observed.copy()

        eligible_count = int(eligible_mask.sum())
        if eligible_count == 0:
            return observed.copy()

        temporally_filled = observed.copy()
        for variable_index in range(observed.shape[1]):
            variable_values = observed[:, variable_index]
            observed_counts = np.isfinite(variable_values[:, eligible_mask]).sum(axis=1)
            fill_allowed = observed_counts / eligible_count >= self.minimum_observed_fraction_for_fill
            fill_source = variable_values.copy()
            # Match the main filling path: weak periods are neither targets nor
            # temporal sources for other periods.
            fill_source[~fill_allowed] = np.nan
            candidate = self._fill_temporal_neighbors(fill_source)
            temporally_filled[:, variable_index] = self._merge_filled_values(
                variable_values, candidate, eligible_mask, fill_allowed
            )
        return temporally_filled

    def _build_cleaning_report(
        self,
        cleaned: np.ndarray,
        observed: np.ndarray,
        eligible_mask: np.ndarray,
        periods: Sequence[str],
        audit_state: dict[tuple[int, int], dict],
        fill_allowed: np.ndarray | None = None,
        stage_null_counts: dict[str, np.ndarray] | None = None,
        variable_names: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Summarize IQR removal and every pre-statistics filling stage."""
        rows = []
        # Produce a stable row per period-variable pair so reports concatenate cleanly across batches.
        eligible_count = int(eligible_mask.sum())
        for time_index, _period_start in enumerate(periods):
            variables = variable_names or self._output_sensor_variables()
            for variable_index, _variable in enumerate(variables):
                observed_count = int(np.isfinite(observed[time_index, variable_index][eligible_mask]).sum())
                final_count = int(np.isfinite(cleaned[time_index, variable_index][eligible_mask]).sum())
                pre_fill_nulls = eligible_count - observed_count
                final_nulls = eligible_count - final_count
                state = audit_state[(time_index, variable_index)]
                step_counts = {
                    step: int(counts[time_index, variable_index]) for step, counts in (stage_null_counts or {}).items()
                }
                after_temporal = step_counts.get("after_temporal_fill", final_nulls)
                after_3x3 = step_counts.get("after_3x3_fill", final_nulls)
                after_5x5 = step_counts.get("after_5x5_fill", final_nulls)
                after_7x7 = step_counts.get("after_7x7_fill", final_nulls)
                after_9x9 = step_counts.get("after_9x9_fill", final_nulls)
                after_interpolation = step_counts.get("after_interpolation", final_nulls)
                state.update(
                    {
                        "null_count_before_iqr": state["original_null_count"],
                        "null_count_after_iqr": pre_fill_nulls,
                        "null_count_after_temporal_fill": after_temporal,
                        "filled_count_by_temporal_fill": pre_fill_nulls - after_temporal,
                        "null_count_after_3x3_fill": after_3x3,
                        "filled_count_by_3x3_fill": after_temporal - after_3x3,
                        "null_count_after_5x5_fill": after_5x5,
                        "filled_count_by_5x5_fill": after_3x3 - after_5x5,
                        "null_count_after_7x7_fill": after_7x7,
                        "filled_count_by_7x7_fill": after_5x5 - after_7x7,
                        "null_count_after_9x9_fill": after_9x9,
                        "filled_count_by_9x9_fill": after_7x7 - after_9x9,
                        "null_count_after_interpolation": after_interpolation,
                        "filled_count_by_interpolation": after_9x9 - after_interpolation,
                        "pre_fill_null_count": pre_fill_nulls,
                        "filled_null_count": pre_fill_nulls - final_nulls,
                        "final_null_count": final_nulls,
                        "has_remaining_nulls": bool(final_nulls),
                        "final_null_rate": final_nulls / eligible_count if eligible_count else np.nan,
                        "observed_fraction": observed_count / eligible_count if eligible_count else np.nan,
                        "fill_allowed": bool(
                            fill_allowed[time_index, variable_index]
                            if fill_allowed is not None
                            else eligible_count and observed_count / eligible_count >= self.minimum_observed_fraction_for_fill
                        ),
                        "interpolation_method": self.interpolation_method,
                    }
                )
                rows.append(state)
        return pd.DataFrame(rows)

    def _clean_monthly_pixels(
        self,
        values: np.ndarray,
        eligible_mask: np.ndarray,
        batch_number: int,
        periods: Sequence[str],
        transform,
        cube_crs,
        iqr_bounds: dict[tuple[int, int], dict] | None = None,
        variable_names: Sequence[str] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Run the complete raster-cleaning pipeline before parcel statistics."""
        if values.shape[0] != len(periods):
            raise ValueError(f"Expected {values.shape[0]} period labels, received {len(periods)}.")
        if iqr_bounds is None:
            iqr_bounds = self._calculate_iqr_bounds(values)
        variables = tuple(variable_names or self._output_sensor_variables())
        cleaned, audit_state = self._remove_iqr_outliers(
            values, eligible_mask, batch_number, periods, iqr_bounds=iqr_bounds, variable_names=variables
        )
        # Keep post-IQR observations separate because count statistics must exclude values imputed later.
        observed = cleaned.copy()

        after_iqr_counts = (~np.isfinite(observed[:, :, eligible_mask])).sum(axis=-1).astype("int64")
        stage_null_counts = {
            "after_temporal_fill": after_iqr_counts.copy(),
            "after_3x3_fill": after_iqr_counts.copy(),
            "after_5x5_fill": after_iqr_counts.copy(),
            "after_7x7_fill": after_iqr_counts.copy(),
            "after_9x9_fill": after_iqr_counts.copy(),
            "after_interpolation": after_iqr_counts.copy(),
        }
        if self.fill_nulls:
            for variable_index, variable in enumerate(variables):
                filled_variable, variable_counts = self._fill_variable_cube(
                    cleaned[:, variable_index],
                    eligible_mask,
                    transform,
                    cube_crs,
                    return_null_counts=True,
                    variable_name=variable,
                    periods=periods,
                    batch_number=batch_number,
                )
                cleaned[:, variable_index] = filled_variable
                for step, counts in variable_counts.items():
                    stage_null_counts[step][:, variable_index] = counts

        report = self._build_cleaning_report(
            cleaned,
            observed,
            eligible_mask,
            periods,
            audit_state,
            stage_null_counts=stage_null_counts,
            variable_names=variables,
        )
        return cleaned, observed, report
