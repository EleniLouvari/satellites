"""Parcel masking, zonal reduction, and derived-statistic calculation.

This module implements the core raster->parcel aggregation logic: it
persists cleaning checkpoints, computes per-parcel statistics for each
temporal period, and reshapes the results into ML-ready, dated feature
columns. The implementation emphasizes deterministic auditability and
atomic file operations to make interrupted runs resumable.
"""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray  # noqa: F401 - registers the xarray ``rio`` accessor.
import xarray as xr
from rasterio.features import geometry_mask
from rasterio.windows import Window, from_bounds, transform as window_transform
from shapely.geometry import mapping

from common_libraries.io_library import read_data, write_data


class ParcelStatisticsCalculator:
    """Aggregate an already-cleaned raster into parcel-level statistics."""

    # These values describe the parcel or extraction batch, so they remain unsuffixed after the temporal pivot.
    STATIC_RESULT_COLUMNS = (
        "batch_number",
        "eligible_pixel_count",
        "meets_minimum_pixel_count",
        "parcel_area_m2",
        "approx_pixel_count",
    )

    def _cleaning_checkpoint_signature(self) -> str:
        """Fingerprint local operations without invalidating reusable raw downloads."""
        configuration = {
            "checkpoint_format": "cleaned_plus_observed_mask_v2",
            "remove_outliers": self.remove_outliers,
            "iqr_quantiles": self.iqr_quantiles,
            "iqr_multiplier": self.iqr_multiplier,
            "iqr_min_valid_pixels": self.iqr_min_valid_pixels,
            "fill_nulls": getattr(self, "fill_nulls", False),
            "temporal_fill_mode": self.temporal_fill_mode,
            "minimum_observed_fraction_for_fill": self.minimum_observed_fraction_for_fill,
            "spatial_fill_window_sizes": getattr(self, "spatial_fill_window_sizes", (3, 5)),
            "interpolation_method": self.interpolation_method,
            "interpolation_max_distance_in_meters": self.interpolation_max_distance_in_meters,
            "interpolation_variogram_lags": self.interpolation_variogram_lags,
            "interpolation_variogram_max_distance_in_meters": self.interpolation_variogram_max_distance_in_meters,
        }
        payload = json.dumps(configuration, sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]

    def _safe_ratio(self, numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        """Divide arrays while representing zero or invalid denominators as nulls."""
        # Return NaN where the denominator is zero or invalid to avoid
        # division-by-zero artifacts in index calculations.
        result = np.full(numerator.shape, np.nan, dtype="float32")
        valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator != 0)
        np.divide(numerator, denominator, out=result, where=valid)
        return result

    def _calculate_local_index(
        self, values_by_band: dict[str, np.ndarray], index_name: str
    ) -> np.ndarray:
        """Calculate one Sentinel-2 index from already-cleaned physical bands."""
        # Use a simple mapping of index names to numpy operations that
        # operate on pre-cleaned band arrays. Results use NaN for invalid
        # pixels to keep downstream reducers consistent.
        band = values_by_band.__getitem__
        if index_name == "NDVI":
            return self._safe_ratio(band("B08") - band("B04"), band("B08") + band("B04"))
        if index_name == "NDWI":
            return self._safe_ratio(band("B03") - band("B08"), band("B03") + band("B08"))
        if index_name == "MNDWI":
            return self._safe_ratio(band("B03") - band("B11"), band("B03") + band("B11"))
        if index_name == "NDMI":
            return self._safe_ratio(band("B08") - band("B11"), band("B08") + band("B11"))
        if index_name == "NBR":
            return self._safe_ratio(band("B08") - band("B12"), band("B08") + band("B12"))
        if index_name == "GNDVI":
            return self._safe_ratio(band("B08") - band("B03"), band("B08") + band("B03"))
        if index_name == "EVI":
            denominator = band("B08") + 6.0 * band("B04") - 7.5 * band("B02") + 1.0
            return self._safe_ratio(2.5 * (band("B08") - band("B04")), denominator)
        if index_name == "SAVI":
            return self._safe_ratio(1.5 * (band("B08") - band("B04")), band("B08") + band("B04") + 0.5)
        if index_name == "MSAVI":
            doubled_nir_plus_one = 2.0 * band("B08") + 1.0
            discriminant = doubled_nir_plus_one**2 - 8.0 * (band("B08") - band("B04"))
            result = np.full(discriminant.shape, np.nan, dtype="float32")
            valid = np.isfinite(discriminant) & (discriminant >= 0)
            result[valid] = (doubled_nir_plus_one[valid] - np.sqrt(discriminant[valid])) / 2.0
            return result
        if index_name == "NDRE":
            return self._safe_ratio(band("B08") - band("B05"), band("B08") + band("B05"))
        if index_name == "PSRI":
            return self._safe_ratio(band("B04") - band("B02"), band("B06"))
        if index_name == "CI":
            ratio = self._safe_ratio(band("B08"), band("B05"))
            return np.where(np.isfinite(ratio), ratio - 1.0, np.nan).astype("float32")
        raise ValueError(f"Unsupported index: {index_name}")

    def _build_local_output_cube(self, values: np.ndarray, cube_variables: list[str]) -> np.ndarray:
        """Select requested bands and append indices derived from cleaned source bands."""
        values_by_band = {name: values[:, index] for index, name in enumerate(cube_variables)}
        output_layers = [values_by_band[name] for name in self.sentinel2_bands]
        output_layers.extend(self._calculate_local_index(values_by_band, name) for name in self.sentinel2_indices)
        output_layers.extend(values_by_band[name] for name in self.sentinel1_bands)
        return np.stack(output_layers, axis=1).astype("float32", copy=False)

    def _cleaning_checkpoint_paths(self, netcdf_path) -> tuple[Path, Path]:
        """Return deterministic cleaned-raster and audit paths for one raw cube."""
        source = Path(netcdf_path)
        # Derive sibling paths from the raw stem so batch and tile-manager naming both work.
        base_stem = source.stem
        return (
            source.with_name(f"{base_stem}_cleaned.nc"),
            source.with_name(f"{base_stem}_cleaning_report.parquet"),
        )

    def _final_checkpoint_path(self, netcdf_path) -> Path:
        """Return the final post-index NetCDF path for one raw cube."""
        source = Path(netcdf_path)
        base_stem = source.stem
        return source.with_name(f"{base_stem}_final.nc")

    def _load_final_checkpoint(
        self,
        netcdf_path,
        periods: list[str],
        variables: list[str],
        expected_shape: tuple[int, ...],
    ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame] | None:
        """Load final bands and indices when the post-index checkpoint is complete."""
        final_path = self._final_checkpoint_path(netcdf_path)
        _, report_path = self._cleaning_checkpoint_paths(netcdf_path)
        if not final_path.exists() or not report_path.exists():
            return None
        with xr.open_dataset(final_path, mask_and_scale=True) as checkpoint:
            if checkpoint.attrs.get("cleaning_signature") != self._cleaning_checkpoint_signature():
                self.parcel_logger.info(f"Ignoring stale post-index checkpoint {final_path}.")
                return None
            saved_periods = checkpoint["period"].astype(str).values.tolist()
            saved_variables = checkpoint["variable"].astype(str).values.tolist()
            cleaned = np.asarray(checkpoint["cleaned"].values, dtype="float32")
            observed_mask = np.asarray(checkpoint["observed_mask"].values, dtype=bool)
            observed = np.where(observed_mask, cleaned, np.nan).astype("float32", copy=False)
        if saved_periods != periods or saved_variables != variables:
            raise ValueError(f"Final checkpoint metadata does not match the configured outputs: {final_path}")
        if cleaned.shape != expected_shape or observed.shape != expected_shape:
            raise ValueError(f"Final checkpoint shape does not match the raw cube: {final_path}")
        self.parcel_logger.info(f"Using completed post-index checkpoint {final_path}.")
        return cleaned, observed, read_data(str(report_path))

    def _save_final_checkpoint(
        self,
        netcdf_path,
        cleaned: np.ndarray,
        observed: np.ndarray,
        periods: list[str],
        variables: list[str],
    ) -> None:
        """Atomically save final requested bands and locally calculated indices."""
        final_path = self._final_checkpoint_path(netcdf_path)
        temporary_path = final_path.with_suffix(".partial.nc")
        dimensions = ("period", "variable", "y", "x")
        xr.Dataset(
            {
                "cleaned": (dimensions, cleaned.astype("float32", copy=False)),
                "observed_mask": (dimensions, np.isfinite(observed).astype("uint8")),
            },
            coords={"period": periods, "variable": variables},
            attrs={"cleaning_signature": self._cleaning_checkpoint_signature()},
        ).to_netcdf(temporary_path)
        temporary_path.replace(final_path)
        self.parcel_logger.info(f"Saved completed post-index checkpoint {final_path}.")

    def _load_cleaning_checkpoint(
        self,
        netcdf_path,
        periods: list[str],
        variables: list[str],
        expected_shape: tuple[int, ...],
    ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame] | None:
        """Load a complete and compatible cleaning checkpoint when available."""
        cleaned_path, report_path = self._cleaning_checkpoint_paths(netcdf_path)
        # The pair is reusable only when both raster payload and audit were committed.
        if not cleaned_path.exists() or not report_path.exists():
            return None

        with xr.open_dataset(cleaned_path, mask_and_scale=True) as checkpoint:
            if checkpoint.attrs.get("cleaning_signature") != self._cleaning_checkpoint_signature():
                self.parcel_logger.info(f"Ignoring stale cleaning checkpoint {cleaned_path}.")
                return None
            # Materialize arrays before closing the lazily opened xarray dataset.
            saved_periods = checkpoint["period"].astype(str).values.tolist()
            saved_variables = checkpoint["variable"].astype(str).values.tolist()
            cleaned = np.asarray(checkpoint["cleaned"].values, dtype="float32")
            observed_mask = np.asarray(checkpoint["observed_mask"].values, dtype=bool)
            observed = np.where(observed_mask, cleaned, np.nan).astype("float32", copy=False)

        if saved_periods != periods or saved_variables != variables:
            # Never reuse a stale checkpoint whose temporal or variable axes differ from the raw cube.
            raise ValueError(f"Cleaning checkpoint metadata does not match the raw cube: {cleaned_path}")
        if cleaned.shape != expected_shape or observed.shape != expected_shape:
            raise ValueError(f"Cleaning checkpoint shape does not match the raw cube: {cleaned_path}")

        report = read_data(str(report_path))
        self.parcel_logger.info(f"Using completed cleaning checkpoint {cleaned_path}.")
        return cleaned, observed, report

    def _save_cleaning_checkpoint(
        self,
        netcdf_path,
        cleaned: np.ndarray,
        observed: np.ndarray,
        report: pd.DataFrame,
        periods: list[str],
        variables: list[str],
    ) -> None:
        """Atomically persist cleaned pixels, observation state, and their audit."""
        cleaned_path, report_path = self._cleaning_checkpoint_paths(netcdf_path)
        temporary_cleaned = cleaned_path.with_suffix(".partial.nc")
        temporary_report = report_path.with_suffix(".partial.parquet")
        # Temporary siblings prevent interrupted writes from appearing complete.
        dimensions = ("period", "variable", "y", "x")
        checkpoint = xr.Dataset(
            {
                "cleaned": (dimensions, cleaned.astype("float32", copy=False)),
                "observed_mask": (dimensions, np.isfinite(observed).astype("uint8")),
            },
            coords={"period": periods, "variable": variables},
            attrs={"cleaning_signature": self._cleaning_checkpoint_signature()},
        )
        checkpoint.to_netcdf(temporary_cleaned)
        write_data(report, str(temporary_report))
        # Commit NetCDF last because its final path is the checkpoint completion marker.
        temporary_report.replace(report_path)
        temporary_cleaned.replace(cleaned_path)
        self.parcel_logger.info(f"Saved completed cleaning checkpoint {cleaned_path}.")

    def _remove_cleaned_raster_checkpoint(self, netcdf_path) -> None:
        """Remove the redundant intermediate raster after final output commits."""
        cleaned_path, _ = self._cleaning_checkpoint_paths(netcdf_path)
        if cleaned_path.exists():
            cleaned_path.unlink()
            self.parcel_logger.info(
                f"Removed intermediate cleaning raster {cleaned_path}; raw data and audit remain available."
            )

    def _reduce_local_pixels(self, pixel_values: np.ndarray) -> dict[str, np.ndarray]:
        """Calculate requested statistics across the final pixel axis."""
        valid_count = np.isfinite(pixel_values).sum(axis=-1)
        # Empty parcel masks need explicit outputs because most reducers reject zero-length axes.
        if pixel_values.shape[-1] == 0:
            empty = np.full(valid_count.shape, np.nan, dtype="float64")
            return {s: valid_count.copy() if s == "count" else empty.copy() for s in self.spatial_statistics}
        totals = np.nansum(pixel_values, axis=-1)
        mean = np.full(valid_count.shape, np.nan, dtype="float64")
        np.divide(totals, valid_count, out=mean, where=valid_count > 0)
        reduced = {}
        with warnings.catch_warnings():
            # All-null slices are valid here; suppress only their expected reducer warnings.
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for statistic in self.spatial_statistics:
                if statistic == "mean":
                    values = mean
                elif statistic == "median":
                    values = np.nanmedian(pixel_values, axis=-1)
                elif statistic == "sd":
                    squares = np.nansum((pixel_values - mean[..., np.newaxis]) ** 2, axis=-1)
                    values = np.full(valid_count.shape, np.nan, dtype="float64")
                    np.divide(squares, valid_count - 1, out=values, where=valid_count > 1)
                    values = np.sqrt(values)
                elif statistic == "min":
                    values = np.nanmin(pixel_values, axis=-1)
                elif statistic == "max":
                    values = np.nanmax(pixel_values, axis=-1)
                elif statistic == "count":
                    values = valid_count
                else:
                    values = np.nanquantile(pixel_values, self.QUANTILE_PROBABILITIES[statistic], axis=-1, method="linear")
                reduced[statistic] = values
        return reduced

    def _crop_values_to_parcel_mask(
        self, values: np.ndarray, parcel_mask: np.ndarray, transform
    ) -> tuple[np.ndarray, np.ndarray, object]:
        """Crop a cube to the bounding window of an eligible-pixel mask."""
        rows, columns = np.nonzero(parcel_mask)
        if rows.size == 0:
            empty_window = Window(col_off=0, row_off=0, width=1, height=1)
            return (values[:, :, :1, :1].copy(), np.zeros((1, 1), dtype=bool), window_transform(empty_window, transform))

        row_start, row_stop = int(rows.min()), int(rows.max()) + 1
        column_start, column_stop = int(columns.min()), int(columns.max()) + 1
        window = Window(col_off=column_start, row_off=row_start, width=column_stop - column_start, height=row_stop - row_start)
        return (
            values[:, :, row_start:row_stop, column_start:column_stop].copy(),
            parcel_mask[row_start:row_stop, column_start:column_stop].copy(),
            window_transform(window, transform),
        )

    def _mask_parcel_window(self, values: np.ndarray, geometry, transform):
        """Rasterize one parcel only within its clipped pixel bounding window."""
        raster_height, raster_width = values.shape[-2:]
        raw_window = from_bounds(*geometry.bounds, transform=transform)
        # Clamp geometry bounds to the raster before allocating a parcel-local mask.
        column_start = max(0, math.floor(raw_window.col_off))
        row_start = max(0, math.floor(raw_window.row_off))
        column_stop = min(raster_width, math.ceil(raw_window.col_off + raw_window.width))
        row_stop = min(raster_height, math.ceil(raw_window.row_off + raw_window.height))

        if column_start >= column_stop or row_start >= row_stop:
            empty_window = Window(col_off=0, row_off=0, width=1, height=1)
            return (values[:, :, :1, :1].copy(), np.zeros((1, 1), dtype=bool), window_transform(empty_window, transform))

        window = Window(col_off=column_start, row_off=row_start, width=column_stop - column_start, height=row_stop - row_start)
        parcel_transform = window_transform(window, transform)
        parcel_mask = geometry_mask(
            # Pixel centers define membership consistently across all parcel reductions.
            [mapping(geometry)],
            out_shape=(int(window.height), int(window.width)),
            transform=parcel_transform,
            invert=True,
            all_touched=False,
        )
        return (values[:, :, row_start:row_stop, column_start:column_stop].copy(), parcel_mask, parcel_transform)

    def _calculate_local_statistics(self, netcdf_path, parcels, batch_number):
        """Clean the complete raster first, then calculate parcel statistics."""
        self.parcel_logger.info(f"Reading monthly pixel cube {netcdf_path}.")
        with xr.open_dataset(netcdf_path, decode_coords="all", mask_and_scale=True) as dataset:
            time_dimension = self._find_cube_dimension(dataset, ("t", "time", "temporal"), "temporal")
            x_dimension = self._find_cube_dimension(dataset, ("x", "longitude", "lon"), "x")
            y_dimension = self._find_cube_dimension(dataset, ("y", "latitude", "lat"), "y")
            cube_variables = list(self._cube_sensor_variables())
            monthly_values = self._select_monthly_variables(
                dataset, time_dimension, x_dimension, y_dimension, expected_variables=cube_variables
            )
            monthly_values = monthly_values.rio.set_spatial_dims(x_dim=x_dimension, y_dim=y_dimension, inplace=False)
            cube_crs = monthly_values.rio.crs or dataset.rio.crs
            if cube_crs is None:
                raise ValueError(
                    "The native-CRS NetCDF does not contain usable CRS metadata. "
                    "Parcel geometries cannot be aligned safely with the raster."
                )
            monthly_values = monthly_values.rio.write_crs(cube_crs, inplace=False)
            transform = monthly_values.rio.transform(recalc=False)
            ordered = monthly_values.transpose(time_dimension, "variable", y_dimension, x_dimension)
            # Materialize one compact layout shared by cleaning and parcel aggregation.
            values = np.asarray(ordered.values, dtype="float32")
            time_values = pd.to_datetime(ordered[time_dimension].values, utc=True, errors="coerce")
            if pd.isna(time_values).any():
                raise ValueError(
                    f"The NetCDF temporal coordinate {time_dimension!r} contains values that cannot be parsed as dates."
                )
            periods = time_values.strftime("%Y-%m-%d").tolist()
            if len(periods) != len(set(periods)):
                raise ValueError("The temporal NetCDF contains duplicate period-start labels.")
            intervals, labels = self._temporal_intervals()
            period_end_lookup = {label: interval[1] for label, interval in zip(labels, intervals)}
            unknown = sorted(set(periods).difference(period_end_lookup))
            if unknown:
                raise ValueError(f"The temporal NetCDF contains unexpected period labels: {unknown}")

            variables = list(self._output_sensor_variables())
            projected = parcels.to_crs(cube_crs)
            final_shape = (values.shape[0], len(variables), *values.shape[-2:])
            final_checkpoint = self._load_final_checkpoint(netcdf_path, periods, variables, final_shape)
            if final_checkpoint is None:
                checkpoint = self._load_cleaning_checkpoint(netcdf_path, periods, cube_variables, values.shape)
                if checkpoint is None:
                    # Clean the complete batch so adjacent parcels can contribute valid fill sources.
                    iqr_bounds = self._calculate_iqr_bounds(values)
                    raster_mask = np.ones(values.shape[-2:], dtype=bool)
                    cleaned_values, raster_observed, raster_report = self._clean_monthly_pixels(
                        values,
                        raster_mask,
                        batch_number,
                        periods,
                        transform,
                        cube_crs,
                        iqr_bounds=iqr_bounds,
                        variable_names=cube_variables,
                    )
                    self._save_cleaning_checkpoint(netcdf_path, cleaned_values, raster_observed, raster_report, periods, cube_variables)
                else:
                    # Resume index calculation from a completed physical-band cleaning checkpoint.
                    cleaned_values, raster_observed, raster_report = checkpoint

                # Ignore indices present in the raw cube and recalculate them from the cleaned physical bands.
                cleaned_values = self._build_local_output_cube(cleaned_values, cube_variables)
                raster_observed = self._build_local_output_cube(raster_observed, cube_variables)
                self._save_final_checkpoint(netcdf_path, cleaned_values, raster_observed, periods, variables)
                if not getattr(self, "keep_cleaned_checkpoint", False):
                    self._remove_cleaned_raster_checkpoint(netcdf_path)
            else:
                # Resume directly at parcel aggregation when the post-index NetCDF is already complete.
                cleaned_values, raster_observed, raster_report = final_checkpoint

            rows = []
            for (_, parcel), (_, projected_parcel) in zip(parcels.iterrows(), projected.iterrows()):
                # Keep source attributes and projected geometry aligned by their preserved row order.
                parcel_id = str(parcel[self.PARCEL_ID_FIELD])
                cleaned, cleaned_mask, _ = self._mask_parcel_window(cleaned_values, projected_parcel.geometry, transform)
                observed, observed_mask, _ = self._mask_parcel_window(raster_observed, projected_parcel.geometry, transform)
                if not np.array_equal(observed_mask, cleaned_mask):
                    raise RuntimeError("Parcel masks changed between raster filling and aggregation.")

                eligible_pixel_count = int(cleaned_mask.sum())
                stats = self._reduce_local_pixels(cleaned[:, :, cleaned_mask])
                if "count" in stats:
                    stats["count"] = np.isfinite(observed[:, :, cleaned_mask]).sum(axis=-1)
                for time_index, period_start in enumerate(periods):
                    row = {
                        self.PARCEL_ID_FIELD: parcel_id,
                        "period_start": period_start,
                        "period_end": period_end_lookup[period_start],
                        "eligible_pixel_count": eligible_pixel_count,
                        "meets_minimum_pixel_count": eligible_pixel_count >= self.minimum_parcel_pixels,
                    }
                    for variable_index, variable in enumerate(variables):
                        for statistic in self.spatial_statistics:
                            value = stats[statistic][time_index, variable_index]
                            row[f"{variable}_{statistic}"] = (
                                int(value) if statistic == "count" and np.isfinite(value) else float(value)
                            )
                    rows.append(row)

        result = pd.DataFrame(rows)
        self.parcel_logger.info(
            f"Calculated {list(self.spatial_statistics)} locally for {len(parcels)} parcels from {netcdf_path.name}."
        )
        return result, raster_report

    def _add_derived_stats(self, data: pd.DataFrame) -> pd.DataFrame:
        """Append deterministic parcel features in one non-fragmenting operation."""
        metric_parcels = self.parcels.to_crs(epsg=self.working_epsg)
        # Calculate area in a projected metric CRS rather than geographic degrees.
        parcel_metrics = pd.DataFrame(
            {
                self.PARCEL_ID_FIELD: self.parcels[self.PARCEL_ID_FIELD].to_numpy(),
                "parcel_area_m2": metric_parcels.geometry.area.to_numpy(),
            }
        )
        parcel_metrics["approx_pixel_count"] = parcel_metrics["parcel_area_m2"] / self.TARGET_RESOLUTION_METRES**2
        enriched = data.merge(parcel_metrics, on=self.PARCEL_ID_FIELD, how="left", validate="many_to_one")

        derived_columns: dict[str, pd.Series] = {}
        # Accumulate derived series and concatenate once to avoid DataFrame fragmentation.
        for variable in self._output_sensor_variables():
            mean = f"{variable}_mean"
            median = f"{variable}_median"
            standard_deviation = f"{variable}_sd"
            minimum = f"{variable}_min"
            maximum = f"{variable}_max"
            count = f"{variable}_count"
            p10 = f"{variable}_p10"
            p25 = f"{variable}_p25"
            p75 = f"{variable}_p75"
            p90 = f"{variable}_p90"

            if minimum in enriched and maximum in enriched:
                derived_columns[f"{variable}_range"] = enriched[maximum] - enriched[minimum]
            if standard_deviation in enriched:
                derived_columns[f"{variable}_variance"] = enriched[standard_deviation] ** 2
                if mean in enriched:
                    absolute_mean = enriched[mean].abs()
                    derived_columns[f"{variable}_coefficient_of_variation"] = enriched[standard_deviation] / absolute_mean.where(
                        absolute_mean > 1e-12
                    )
            if p25 in enriched and p75 in enriched:
                iqr = enriched[p75] - enriched[p25]
                derived_columns[f"{variable}_iqr"] = iqr
                if median in enriched:
                    derived_columns[f"{variable}_bowley_skewness"] = (
                        enriched[p75] + enriched[p25] - 2 * enriched[median]
                    ) / iqr.where(iqr.abs() > 1e-12)
            if p10 in enriched and p90 in enriched:
                derived_columns[f"{variable}_p90_p10_spread"] = enriched[p90] - enriched[p10]
            if count in enriched:
                approximate_fraction = enriched[count] / enriched["approx_pixel_count"]
                derived_columns[f"{variable}_valid_pixel_fraction_approx"] = approximate_fraction.clip(0.0, 1.0)
                if "eligible_pixel_count" in enriched:
                    exact_denominator = enriched["eligible_pixel_count"].where(enriched["eligible_pixel_count"] > 0)
                    exact_fraction = enriched[count] / exact_denominator
                    derived_columns[f"{variable}_valid_pixel_fraction"] = exact_fraction.clip(0.0, 1.0)

        if not derived_columns:
            return enriched
        derived = pd.DataFrame(derived_columns, index=enriched.index)
        return pd.concat([enriched, derived], axis=1)

    def _reshape_time_series_for_ml(self, data: pd.DataFrame) -> gpd.GeoDataFrame:
        """Pivot parcel-period statistics into one ML-ready row per parcel."""

        required = {self.PARCEL_ID_FIELD, "period_start"}
        missing = sorted(required.difference(data.columns))
        if missing:
            raise ValueError(f"Missing columns required for time-series reshaping: {missing}")

        # Normalize IDs and dates before checking the parcel-period table grain.
        working = data.copy()
        working[self.PARCEL_ID_FIELD] = working[self.PARCEL_ID_FIELD].astype(str)
        parsed_periods = pd.to_datetime(working["period_start"], errors="coerce", utc=True)
        invalid_periods = parsed_periods.isna()
        if invalid_periods.any():
            examples = working.loc[invalid_periods, self.PARCEL_ID_FIELD].head(10).tolist()
            raise ValueError(
                f"'period_start' contains {int(invalid_periods.sum())} invalid timestamps. "
                f"Example {self.PARCEL_ID_FIELD} values: {examples}"
            )
        working["_period_label"] = parsed_periods.dt.strftime("%Y%m%d")

        # Each parcel can contribute only one observation to a dated feature column.
        duplicate_grain = working.duplicated([self.PARCEL_ID_FIELD, "_period_label"], keep=False)
        if duplicate_grain.any():
            examples = working.loc[duplicate_grain, [self.PARCEL_ID_FIELD, "period_start"]].head(10).to_dict("records")
            raise ValueError(f"Zonal statistics must have at most one row per parcel and period. Duplicate examples: {examples}")

        # Verify that fields retained once per parcel do not silently vary between periods.
        static_columns = [column for column in self.STATIC_RESULT_COLUMNS if column in working.columns]
        for column in static_columns:
            distinct_counts = working.groupby(self.PARCEL_ID_FIELD, sort=False)[column].nunique(dropna=False)
            inconsistent = distinct_counts[distinct_counts > 1]
            if not inconsistent.empty:
                raise ValueError(
                    f"Static column {column!r} changes across periods for "
                    f"{len(inconsistent)} parcels; examples: {inconsistent.index[:10].tolist()}"
                )

        # Every remaining statistic is temporal and receives a YYYYMMDD suffix.
        excluded = {self.PARCEL_ID_FIELD, "period_start", "period_end", "_period_label", *static_columns}
        temporal_features = [column for column in working.columns if column not in excluded]
        if not temporal_features:
            raise ValueError("No temporal feature columns are available for ML reshaping.")

        # Pivot all variables and statistics together to guarantee one row per parcel.
        pivoted = working.pivot(index=self.PARCEL_ID_FIELD, columns="_period_label", values=temporal_features)
        pivoted.columns = [f"{feature}__{period_label}" for feature, period_label in pivoted.columns]
        pivoted = pivoted.sort_index(axis=1).reset_index()

        if static_columns:
            static = working[[self.PARCEL_ID_FIELD, *static_columns]].drop_duplicates(self.PARCEL_ID_FIELD)
            pivoted = static.merge(pivoted, on=self.PARCEL_ID_FIELD, how="inner", validate="one_to_one")

        # Reattach source labels/metadata and geometry once, after all dated features have been generated.
        # Parcels are kept internally in WGS84 for openEO feature collections, but persisted results use the
        # configured projected CRS so partition and merged GeoParquet outputs have a stable metric projection.
        output_parcels = self.parcels.to_crs(epsg=self.working_epsg)
        parcel_attribute_columns = [
            column for column in output_parcels.columns if column not in {self.PARCEL_ID_FIELD, "geometry"}
        ]
        collisions = sorted(set(parcel_attribute_columns).intersection(pivoted.columns))
        if collisions:
            raise ValueError(
                "Parcel attributes collide with generated zonal-statistics columns. "
                f"Rename these parcel columns before extraction: {collisions}"
            )
        parcel_attributes = output_parcels[[self.PARCEL_ID_FIELD, *parcel_attribute_columns, "geometry"]]
        result = parcel_attributes.merge(pivoted, on=self.PARCEL_ID_FIELD, how="inner", validate="one_to_one")
        result = gpd.GeoDataFrame(result, geometry="geometry", crs=output_parcels.crs)
        self.parcel_logger.info(
            "Reshaped %s parcel-period rows into %s ML-ready parcel rows with %s dated features in EPSG:%s.",
            len(data),
            len(result),
            len(pivoted.columns) - len(static_columns) - 1,
            self.working_epsg,
        )
        return result
