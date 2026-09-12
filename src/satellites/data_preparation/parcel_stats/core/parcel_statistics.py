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

from satellites.shared.io import read_data, write_data


class ParcelStatisticsCalculator:
    """Aggregate an already-cleaned raster into parcel-level statistics."""

    # These values describe the parcel or extraction batch, so they remain unsuffixed after the temporal pivot.
    STATIC_RESULT_COLUMNS = (
        "batch_number",
        "intersected_pixel_count",
        "meets_minimum_pixel_count",
        "expected_pixel_count",
        "temporal_filled_pixel_count",
        "spatial_filled_pixel_count",
        "temporal_filled_ratio",
        "spatial_filled_ratio",
        "data_reliability_score",
        "pixel_area",
        "geom_area",
        "geom_interior_area_ratio",
        "geom_compactness",
        "geom_perimeter_area_ratio",
        "geom_shape_index",
        "geom_elongation",
        "geom_shape_complexity_score",
    )

    def _cleaning_checkpoint_signature(self) -> str:
        """Fingerprint local operations without invalidating reusable raw downloads."""
        configuration = {
            "checkpoint_format": "cleaned_plus_fill_provenance_v3",
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

    def _calculate_local_optical_index(self, values_by_band, index_name):
        """Translate source band identifiers to spectral roles before shared algebra."""
        values_by_role = {role: values_by_band[name] for role, name in self.OPTICAL_BAND_ROLES.items() if name in values_by_band}
        return self._calculate_optical_index(values_by_role, index_name)

    def _calculate_local_sentinel2_index(self, values_by_band, index_name):
        """Compatibility wrapper for callers using the original method name."""
        return self._calculate_local_optical_index(values_by_band, index_name)

    def _calculate_optical_index(self, values_by_role: dict[str, np.ndarray], index_name: str) -> np.ndarray:
        """Calculate an optical index from cleaned reflectance arrays keyed by spectral role."""
        # Use a simple mapping of index names to numpy operations that
        # operate on pre-cleaned band arrays. Results use NaN for invalid
        # pixels to keep downstream reducers consistent.
        band = values_by_role.__getitem__
        if index_name == "NDVI":
            return self._safe_ratio(band("nir") - band("red"), band("nir") + band("red"))
        if index_name == "NDWI":
            return self._safe_ratio(band("green") - band("nir"), band("green") + band("nir"))
        if index_name == "MNDWI":
            return self._safe_ratio(band("green") - band("swir_1"), band("green") + band("swir_1"))
        if index_name == "NDMI":
            return self._safe_ratio(band("nir") - band("swir_1"), band("nir") + band("swir_1"))
        if index_name == "NBR":
            return self._safe_ratio(band("nir") - band("swir_2"), band("nir") + band("swir_2"))
        if index_name == "GNDVI":
            return self._safe_ratio(band("nir") - band("green"), band("nir") + band("green"))
        if index_name == "EVI":
            denominator = band("nir") + 6.0 * band("red") - 7.5 * band("blue") + 1.0
            return self._safe_ratio(2.5 * (band("nir") - band("red")), denominator)
        if index_name == "SAVI":
            return self._safe_ratio(1.5 * (band("nir") - band("red")), band("nir") + band("red") + 0.5)
        if index_name == "MSAVI":
            doubled_nir_plus_one = 2.0 * band("nir") + 1.0
            discriminant = doubled_nir_plus_one**2 - 8.0 * (band("nir") - band("red"))
            result = np.full(discriminant.shape, np.nan, dtype="float32")
            valid = np.isfinite(discriminant) & (discriminant >= 0)
            result[valid] = (doubled_nir_plus_one[valid] - np.sqrt(discriminant[valid])) / 2.0
            return result
        if index_name == "NDRE":
            return self._safe_ratio(band("nir") - band("red_edge"), band("nir") + band("red_edge"))
        if index_name == "PSRI":
            return self._safe_ratio(band("red") - band("blue"), band("red_edge_2"))
        if index_name == "CI":
            ratio = self._safe_ratio(band("nir"), band("red_edge"))
            return np.where(np.isfinite(ratio), ratio - 1.0, np.nan).astype("float32")
        raise ValueError(f"Unsupported index: {index_name}")

    def _calculate_local_sentinel1_index(self, values_by_band: dict[str, np.ndarray], index_name: str) -> np.ndarray:
        """Calculate one Sentinel-1 index per pixel from linear-power sigma0."""
        vv = values_by_band["VV"]
        vh = values_by_band["VH"]
        finite_negative = (np.isfinite(vv) & (vv < 0)) | (np.isfinite(vh) & (vh < 0))
        if np.any(finite_negative):
            raise ValueError(
                "Sentinel-1 R and RVI require non-negative linear-power sigma0 values; "
                "negative values suggest dB-scaled or physically invalid input."
            )
        if index_name == "R":
            return self._safe_ratio(vv, vh)
        if index_name == "RVI":
            return self._safe_ratio(4.0 * vh, vv + vh)
        raise ValueError(f"Unsupported Sentinel-1 index: {index_name}")

    def _build_local_output_cube(self, values: np.ndarray, cube_variables: list[str]) -> np.ndarray:
        """Select requested bands and append indices derived from cleaned source bands."""
        values_by_band = {name: values[:, index] for index, name in enumerate(cube_variables)}
        output_layers = [values_by_band[name] for name in self.optical_bands]
        output_layers.extend(self._calculate_local_optical_index(values_by_band, name) for name in self.optical_indices)
        output_layers.extend(values_by_band[name] for name in self.sentinel1_bands)
        output_layers.extend(self._calculate_local_sentinel1_index(values_by_band, name) for name in self.sentinel1_indices)
        return np.stack(output_layers, axis=1).astype("float32", copy=False)

    def _cleaning_checkpoint_paths(self, netcdf_path) -> tuple[Path, Path]:
        """Return deterministic cleaned-raster and audit paths for one raw cube."""
        source = Path(netcdf_path)
        # Derive sibling paths from the raw stem so batch and tile-manager naming both work.
        base_stem = source.stem
        return (source.with_name(f"{base_stem}_cleaned.nc"), source.with_name(f"{base_stem}_cleaning_report.parquet"))

    def _final_checkpoint_path(self, netcdf_path) -> Path:
        """Return the final post-index NetCDF path for one raw cube."""
        source = Path(netcdf_path)
        base_stem = source.stem
        return source.with_name(f"{base_stem}_final.nc")

    def _load_final_checkpoint(
        self, netcdf_path, periods: list[str], variables: list[str], expected_shape: tuple[int, ...]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame] | None:
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
            temporal_filled_mask = np.asarray(checkpoint["temporal_filled_mask"].values, dtype=bool)
            observed = np.where(observed_mask, cleaned, np.nan).astype("float32", copy=False)
        if saved_periods != periods or saved_variables != variables:
            raise ValueError(f"Final checkpoint metadata does not match the configured outputs: {final_path}")
        if cleaned.shape != expected_shape or observed.shape != expected_shape or temporal_filled_mask.shape != expected_shape:
            raise ValueError(f"Final checkpoint shape does not match the raw cube: {final_path}")
        self.parcel_logger.info(f"Using completed post-index checkpoint {final_path}.")
        return cleaned, observed, temporal_filled_mask, read_data(str(report_path))

    def _save_final_checkpoint(
        self,
        netcdf_path,
        cleaned: np.ndarray,
        observed: np.ndarray,
        periods: list[str],
        variables: list[str],
        temporally_filled: np.ndarray | None = None,
    ) -> None:
        """Atomically save final requested bands and locally calculated indices."""
        final_path = self._final_checkpoint_path(netcdf_path)
        temporary_path = final_path.with_suffix(".partial.nc")
        dimensions = ("period", "variable", "y", "x")
        if temporally_filled is None:
            temporally_filled = observed
        temporal_filled_mask = ~np.isfinite(observed) & np.isfinite(temporally_filled)
        xr.Dataset(
            {
                "cleaned": (dimensions, cleaned.astype("float32", copy=False)),
                "observed_mask": (dimensions, np.isfinite(observed).astype("uint8")),
                "temporal_filled_mask": (dimensions, temporal_filled_mask.astype("uint8")),
            },
            coords={"period": periods, "variable": variables},
            attrs={"cleaning_signature": self._cleaning_checkpoint_signature()},
        ).to_netcdf(temporary_path)
        temporary_path.replace(final_path)
        self.parcel_logger.info(f"Saved completed post-index checkpoint {final_path}.")

    def _load_cleaning_checkpoint(
        self, netcdf_path, periods: list[str], variables: list[str], expected_shape: tuple[int, ...]
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
            self.parcel_logger.info(f"Removed intermediate cleaning raster {cleaned_path}; raw data and audit remain available.")

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
                elif statistic == "range":
                    values = np.nanmax(pixel_values, axis=-1) - np.nanmin(pixel_values, axis=-1)
                else:
                    values = np.nanquantile(pixel_values, self.QUANTILE_PROBABILITIES[statistic], axis=-1, method="linear")
                reduced[statistic] = values
        return reduced

    @staticmethod
    def _summarize_parcel_filling(
        observed: np.ndarray, temporal_filled_mask: np.ndarray, final: np.ndarray, intersected_pixel_count: int
    ) -> dict[str, int | float]:
        """Aggregate fill provenance over every requested period and band."""
        expected_pixel_count = intersected_pixel_count * observed.shape[0] * observed.shape[1]
        observed_mask = np.isfinite(observed)
        temporal_mask = temporal_filled_mask.astype(bool, copy=False)
        final_mask = np.isfinite(final)
        observed_count = int(observed_mask.sum())
        temporal_count = int(temporal_mask.sum())
        # Any newly available value not attributed to the temporal stage was
        # produced by a spatial-neighbor or interpolation stage.
        spatial_count = int((~observed_mask & ~temporal_mask & final_mask).sum())
        return {
            "expected_pixel_count": expected_pixel_count,
            "temporal_filled_pixel_count": temporal_count,
            "spatial_filled_pixel_count": spatial_count,
            "temporal_filled_ratio": temporal_count / expected_pixel_count if expected_pixel_count else np.nan,
            "spatial_filled_ratio": spatial_count / expected_pixel_count if expected_pixel_count else np.nan,
            "data_reliability_score": observed_count / expected_pixel_count if expected_pixel_count else np.nan,
        }

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
            # Include every raster cell intersected by the parcel, including
            # cells whose center lies just outside a parcel boundary.
            [mapping(geometry)],
            out_shape=(int(window.height), int(window.width)),
            transform=parcel_transform,
            invert=True,
            all_touched=True,
        )
        return (values[:, :, row_start:row_stop, column_start:column_stop].copy(), parcel_mask, parcel_transform)

    def _calculate_local_statistics(self, netcdf_path, parcels, batch_number):
        """Clean the complete raster first, then calculate parcel statistics."""
        self.parcel_logger.info(f"Reading temporal pixel cube {netcdf_path}.")
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
            raster_mask = np.ones(values.shape[-2:], dtype=bool)
            final_checkpoint = self._load_final_checkpoint(netcdf_path, periods, variables, final_shape)
            if final_checkpoint is None:
                checkpoint = self._load_cleaning_checkpoint(netcdf_path, periods, cube_variables, values.shape)
                if checkpoint is None:
                    # Clean the complete batch so adjacent parcels can contribute valid fill sources.
                    iqr_bounds = self._calculate_iqr_bounds(values)
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
                    self._save_cleaning_checkpoint(
                        netcdf_path, cleaned_values, raster_observed, raster_report, periods, cube_variables
                    )
                else:
                    # Resume index calculation from a completed physical-band cleaning checkpoint.
                    cleaned_values, raster_observed, raster_report = checkpoint

                # Ignore indices present in the raw cube and recalculate them from the cleaned physical bands.
                temporally_filled_values = self._fill_temporal_cube_only(raster_observed, raster_mask)
                cleaned_values = self._build_local_output_cube(cleaned_values, cube_variables)
                raster_observed = self._build_local_output_cube(raster_observed, cube_variables)
                temporally_filled_values = self._build_local_output_cube(temporally_filled_values, cube_variables)
                temporal_filled_mask = ~np.isfinite(raster_observed) & np.isfinite(temporally_filled_values)
                self._save_final_checkpoint(
                    netcdf_path, cleaned_values, raster_observed, periods, variables, temporally_filled=temporally_filled_values
                )
                if not getattr(self, "keep_cleaned_checkpoint", False):
                    self._remove_cleaned_raster_checkpoint(netcdf_path)
            else:
                # Resume directly at parcel aggregation when the post-index NetCDF is already complete.
                cleaned_values, raster_observed, temporal_filled_mask, raster_report = final_checkpoint

            rows = []
            for (_, parcel), (_, projected_parcel) in zip(parcels.iterrows(), projected.iterrows()):
                # Keep source attributes and projected geometry aligned by their preserved row order.
                parcel_id = str(parcel[self.PARCEL_ID_FIELD])
                cleaned, cleaned_mask, _ = self._mask_parcel_window(cleaned_values, projected_parcel.geometry, transform)
                observed, observed_mask, _ = self._mask_parcel_window(raster_observed, projected_parcel.geometry, transform)
                local_temporal_filled, temporal_mask, _ = self._mask_parcel_window(
                    temporal_filled_mask, projected_parcel.geometry, transform
                )
                if not np.array_equal(observed_mask, cleaned_mask) or not np.array_equal(temporal_mask, cleaned_mask):
                    raise RuntimeError("Parcel masks changed between raster filling and aggregation.")

                intersected_pixel_count = int(cleaned_mask.sum())
                local_temporal_filled = local_temporal_filled[:, :, cleaned_mask].astype(bool, copy=False)
                fill_metrics = self._summarize_parcel_filling(
                    observed[:, :, cleaned_mask], local_temporal_filled, cleaned[:, :, cleaned_mask], intersected_pixel_count
                )
                pixel_cell_area = abs(float(transform.a * transform.e - transform.b * transform.d))
                stats = self._reduce_local_pixels(cleaned[:, :, cleaned_mask])
                for time_index, period_start in enumerate(periods):
                    row = {
                        self.PARCEL_ID_FIELD: parcel_id,
                        "period_start": period_start,
                        "period_end": period_end_lookup[period_start],
                        "intersected_pixel_count": intersected_pixel_count,
                        "meets_minimum_pixel_count": intersected_pixel_count >= self.minimum_parcel_pixels,
                        **fill_metrics,
                        "pixel_area": intersected_pixel_count * pixel_cell_area,
                    }
                    for variable_index, variable in enumerate(variables):
                        for statistic in self.spatial_statistics:
                            value = stats[statistic][time_index, variable_index]
                            row[f"{variable}_{statistic}"] = float(value)
                    rows.append(row)

        result = pd.DataFrame(rows)
        self.parcel_logger.info(
            f"Calculated {list(self.spatial_statistics)} locally for {len(parcels)} parcels from {netcdf_path.name}."
        )
        return result, raster_report

    def _add_derived_stats(self, data: pd.DataFrame) -> pd.DataFrame:
        """Append deterministic parcel features in one non-fragmenting operation."""
        metric_parcels = self.parcels.to_crs(epsg=self.working_epsg)
        # Calculate all geometry measures in a projected metric CRS rather than
        # geographic degrees.
        geom_area = metric_parcels.geometry.area.astype("float64")
        geom_perimeter = metric_parcels.geometry.length.astype("float64")
        positive_area = geom_area.where(geom_area > 0)
        positive_perimeter = geom_perimeter.where(geom_perimeter > 0)
        parcel_metrics = pd.DataFrame(
            {
                self.PARCEL_ID_FIELD: self.parcels[self.PARCEL_ID_FIELD].to_numpy(),
                "geom_area": geom_area.to_numpy(),
                "geom_compactness": (4.0 * np.pi * geom_area / positive_perimeter.pow(2)).to_numpy(),
                "geom_perimeter_area_ratio": (geom_perimeter / positive_area).to_numpy(),
                "geom_shape_index": (geom_perimeter / (2.0 * np.sqrt(np.pi * positive_area))).to_numpy(),
                "geom_elongation": metric_parcels.geometry.map(self._minimum_rotated_rectangle_elongation).to_numpy(),
            }
        )
        enriched = data.merge(parcel_metrics, on=self.PARCEL_ID_FIELD, how="left", validate="many_to_one")
        enriched["geom_interior_area_ratio"] = enriched["pixel_area"] / enriched["geom_area"].where(enriched["geom_area"] > 0)
        enriched["geom_shape_complexity_score"] = self._calculate_geometry_complexity_score(enriched)

        derived_columns: dict[str, pd.Series] = {}

        def add_derived_column(column_name: str, values: pd.Series) -> None:
            """Keep configured reducers authoritative when a derived name already exists."""
            if column_name in enriched.columns or column_name in derived_columns:
                return
            derived_columns[column_name] = values

        # Accumulate derived series and concatenate once to avoid DataFrame fragmentation.
        for variable in self._output_sensor_variables():
            mean = f"{variable}_mean"
            median = f"{variable}_median"
            standard_deviation = f"{variable}_sd"
            minimum = f"{variable}_min"
            maximum = f"{variable}_max"
            p10 = f"{variable}_p10"
            p25 = f"{variable}_p25"
            p75 = f"{variable}_p75"
            p90 = f"{variable}_p90"

            if minimum in enriched and maximum in enriched:
                add_derived_column(f"{variable}_range", enriched[maximum] - enriched[minimum])
            if standard_deviation in enriched:
                add_derived_column(f"{variable}_variance", enriched[standard_deviation] ** 2)
                if mean in enriched:
                    absolute_mean = enriched[mean].abs()
                    add_derived_column(
                        f"{variable}_coefficient_of_variation",
                        enriched[standard_deviation] / absolute_mean.where(absolute_mean > 1e-12),
                    )
            if p25 in enriched and p75 in enriched:
                iqr = enriched[p75] - enriched[p25]
                add_derived_column(f"{variable}_iqr", iqr)
                if median in enriched:
                    add_derived_column(
                        f"{variable}_bowley_skewness",
                        (enriched[p75] + enriched[p25] - 2 * enriched[median]) / iqr.where(iqr.abs() > 1e-12),
                    )
            if p10 in enriched and p90 in enriched:
                add_derived_column(f"{variable}_p90_p10_spread", enriched[p90] - enriched[p10])
        if not derived_columns:
            return enriched
        derived = pd.DataFrame(derived_columns, index=enriched.index)
        return pd.concat([enriched, derived], axis=1)

    @staticmethod
    def _minimum_rotated_rectangle_elongation(geometry) -> float:
        """Return the long/short side ratio of a geometry's rotated rectangle."""
        if geometry is None or geometry.is_empty:
            return np.nan
        rectangle = geometry.minimum_rotated_rectangle
        if rectangle.is_empty or rectangle.geom_type != "Polygon":
            return np.nan
        coordinates = list(rectangle.exterior.coords)
        side_lengths = [math.hypot(x2 - x1, y2 - y1) for (x1, y1), (x2, y2) in zip(coordinates, coordinates[1:])]
        positive_lengths = [length for length in side_lengths if length > 0]
        if len(positive_lengths) < 2:
            return np.nan
        return float(max(positive_lengths) / min(positive_lengths))

    @staticmethod
    def _calculate_geometry_complexity_score(data: pd.DataFrame) -> pd.Series:
        """Combine raster agreement, boundary irregularity, and elongation.

        Log penalties make the score dimensionless and keep multiplicative
        departures comparable. Compactness, normalized perimeter/area, and
        shape index encode the same circularity relationship, so they are
        averaged into one boundary term before the final three-part mean.
        """
        positive_area = data["geom_area"].where(data["geom_area"] > 0)
        interior_ratio = data["geom_interior_area_ratio"].where(data["geom_interior_area_ratio"] > 0)
        compactness = data["geom_compactness"].where(data["geom_compactness"] > 0)
        shape_index = data["geom_shape_index"].where(data["geom_shape_index"] > 0)
        elongation = data["geom_elongation"].where(data["geom_elongation"] > 0)

        raster_agreement_penalty = np.log(interior_ratio).abs()
        compactness_penalty = (-0.5 * np.log(compactness)).clip(lower=0.0)
        equal_area_circle_perimeter_area_ratio = 2.0 * np.sqrt(np.pi / positive_area)
        normalized_perimeter_area_ratio = (data["geom_perimeter_area_ratio"] / equal_area_circle_perimeter_area_ratio).where(
            lambda values: values > 0
        )
        perimeter_penalty = np.log(normalized_perimeter_area_ratio).clip(lower=0.0)
        shape_index_penalty = np.log(shape_index).clip(lower=0.0)
        elongation_penalty = np.log(elongation).clip(lower=0.0)

        boundary_penalty = pd.concat([compactness_penalty, perimeter_penalty, shape_index_penalty], axis=1).mean(
            axis=1, skipna=False
        )
        return pd.concat([raster_agreement_penalty, boundary_penalty, elongation_penalty], axis=1).mean(axis=1, skipna=False)

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

        # Ensure every configured calendar period produces a column, even when a period is
        # 100% cloud-covered and has no observations in this partition's area.  Without this
        # padding, different partitions produce different column counts (the missing-period
        # features are simply absent rather than NaN), which causes a BlockManager shape
        # mismatch when the in-memory partition results are later merged.
        _, expected_labels = self._temporal_intervals()
        expected_columns = sorted(
            f"{feature}__{period_label}"
            for feature in temporal_features
            for period_label in [label.replace("-", "") for label in expected_labels]
        )
        missing_columns = [col for col in expected_columns if col not in pivoted.columns]
        if missing_columns:
            self.parcel_logger.warning(
                "Adding %s all-NaN column(s) for calendar period(s) with no valid observations in this partition "
                "(likely 100%% cloud cover). First missing: %s.",
                len(missing_columns),
                missing_columns[0],
            )
            for col in missing_columns:
                pivoted[col] = np.nan

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
