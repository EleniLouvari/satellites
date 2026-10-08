"""Validation and cache-signature helpers for satellite zonal statistics.

This module centralizes configuration validation and deterministic cache
signature construction used by the zonal-statistics pipeline. Validators
normalize inputs (dates, EPSG codes, temporal intervals) and raise clear
errors for invalid or inconsistent options so callers can fail fast.
"""

from __future__ import annotations

import hashlib
import json
import re

import numpy as np
import pandas as pd
from pyproj import CRS


class ZonalStatsConfiguration:
    """Validate configuration consumed by the zonal-statistics components."""

    # The original constructor and cache schema use Sentinel-specific fields.
    # Properties keep one source of truth while shared code uses optical names.
    @property
    def optical_bands(self) -> tuple[str, ...]:
        """Selected optical reflectance bands in their native source naming."""
        return self.sentinel2_bands

    @property
    def optical_indices(self) -> tuple[str, ...]:
        """Selected indices derived from cleaned optical reflectance."""
        return self.sentinel2_indices

    def _validate_openeo_credentials(self, username: str | None, password: str | None) -> tuple[str | None, str | None]:
        """Validate openEO credentials: username and password pair."""
        if (username is None) != (password is None):
            raise ValueError("Error: openeo_username and openeo_password must be supplied together.")
        if username is None:
            return None, None
        if not isinstance(username, str) or not username.strip():
            raise ValueError("Error: openeo_username must be a non-empty string.")
        if not isinstance(password, str) or not password:
            raise ValueError("Error: openeo_password must be a non-empty string.")
        return username.strip(), password

    def _validate_batch_workers(self, batch_workers: int) -> int:
        """Return a positive process count for local batch calculations."""
        if isinstance(batch_workers, bool):
            raise TypeError("Error: batch_workers must be a positive integer.")
        try:
            workers = int(batch_workers)
        except (TypeError, ValueError) as exc:
            raise TypeError("Error: batch_workers must be a positive integer.") from exc
        if workers < 1:
            raise ValueError("Error: batch_workers must be at least 1.")
        return workers

    def _validate_spatial_fill_windows(self, window_sizes) -> tuple[int, ...]:
        """Validate an ordered prefix of the supported spatial mean windows."""
        if not isinstance(window_sizes, (tuple, list)):
            raise TypeError("Error: spatial_fill_window_sizes must be a tuple or list.")
        try:
            normalized = tuple(int(size) for size in window_sizes)
        except (TypeError, ValueError) as exc:
            raise TypeError("Error: spatial_fill_window_sizes must contain integers.") from exc
        supported = (3, 5, 7, 9)
        if not normalized or normalized != supported[: len(normalized)]:
            raise ValueError("Error: spatial_fill_window_sizes must be one of (3,), (3, 5), (3, 5, 7), or (3, 5, 7, 9).")
        return normalized

    def _validate_temporal_fill_windows(self, window_sizes) -> tuple[int, ...] | None:
        """Validate increasing odd widths measured in raster time steps."""
        if window_sizes is None:
            return None
        if not isinstance(window_sizes, (tuple, list)):
            raise TypeError("Error: temporal_fill_window_sizes must be a tuple or list, or None.")
        if any(isinstance(size, (bool, np.bool_)) or not isinstance(size, (int, np.integer)) for size in window_sizes):
            raise TypeError("Error: temporal_fill_window_sizes must contain integers.")
        normalized = tuple(int(size) for size in window_sizes)
        if (
            not normalized
            or any(size < 3 or size % 2 == 0 for size in normalized)
            or any(left >= right for left, right in zip(normalized, normalized[1:]))
        ):
            raise ValueError("Error: temporal_fill_window_sizes must contain strictly increasing odd integers of at least 3.")
        return normalized

    def _validate_dates(self, start_date: str | pd.Timestamp, end_date: str | pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Parse the temporal bounds and verify that they are ordered."""
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        if pd.isna(start) or pd.isna(end):
            raise ValueError("Error: start_date and end_date must be valid dates.")
        if start > end:
            raise ValueError("Error: start_date must be on or before end_date.")
        return start, end

    def _validate_working_epsg(self, working_epsg: int) -> int:
        """Return a valid projected EPSG code suitable for metric operations."""
        try:
            epsg = int(working_epsg)
            crs = CRS.from_epsg(epsg)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Error: Invalid EPSG code: {working_epsg!r}.") from exc
        if not crs.is_projected:
            raise ValueError("Error: working_epsg must identify a projected CRS.")
        return epsg

    def _validate_temporal_aggregation(self, temporal_period: str, temporal_reducer: str | None) -> tuple[str, str]:
        """Normalize a day/month bin width and acquisition reducer."""
        if not isinstance(temporal_period, str):
            raise TypeError("Error: temporal_period must be a string such as 15D or 2M.")
        aliases = {"DAY": "1D", "DAILY": "1D", "MONTH": "1M", "MONTHLY": "1M"}
        # Normalize friendly aliases before applying the strict cache-stable interval syntax.
        raw_period = temporal_period.strip().upper()
        period = aliases.get(raw_period, raw_period)
        if re.fullmatch(r"([1-9][0-9]*)(D|M|Y)", period) is None:
            raise ValueError("Error: temporal_period must be a positive day/month/year interval such as 15D, 1M, 2M, 3M or 1Y.")
        reducer = str(temporal_reducer).strip().lower()
        reducer = {"average": "mean", "avg": "mean"}.get(reducer, reducer)
        supported_reducers = {"mean", "median", "min", "max", "sum", "none"}
        if reducer not in supported_reducers:
            raise ValueError(
                f"Error: Unsupported temporal_reducer: {temporal_reducer!r}. Supported reducers: {sorted(supported_reducers)}"
            )
        return period, reducer

    def _temporal_intervals(self) -> tuple[list[list[str]], list[str]]:
        """Return explicit half-open openEO intervals and start-date labels."""
        # Build explicit half-open intervals matching the user's bin width so
        # downstream openEO aggregation calls receive stable, cacheable labels.
        match = re.fullmatch(r"([1-9][0-9]*)(D|M|Y)", self.temporal_period)
        if match is None:
            raise RuntimeError(f"Error: Invalid temporal period: {self.temporal_period}")
        amount = int(match.group(1))
        unit = match.group(2)
        if unit == "D":
            step = pd.Timedelta(days=amount)
        elif unit == "M":
            step = pd.DateOffset(months=amount)
        else:
            step = pd.DateOffset(years=amount)

        cursor = self.start_date.normalize()
        # openEO intervals are half-open, so include the user's final calendar day via an exclusive bound.
        end_exclusive = self.end_date.normalize() + pd.Timedelta(days=1)
        intervals: list[list[str]] = []
        labels: list[str] = []
        while cursor < end_exclusive:
            interval_end = min(cursor + step, end_exclusive)
            intervals.append([cursor.strftime("%Y-%m-%d"), interval_end.strftime("%Y-%m-%d")])
            labels.append(cursor.strftime("%Y-%m-%d"))
            cursor = interval_end
        return intervals, labels

    def _validate_interpolation_options(
        self,
        method: str,
        max_distance_in_meters: float | None,
        variogram_lags: int,
        variogram_max_distance_in_meters: float | None,
    ) -> tuple[str, float | None, int, float | None]:
        """Validate raster-level spatial interpolation configuration."""
        normalized_method = str(method).strip().lower()
        # Retain common legacy names while storing one canonical method in the configuration.
        normalized_method = {"nearest_neighbor": "nearest", "inverse_distance_weighting": "idw"}.get(
            normalized_method, normalized_method
        )
        if normalized_method not in {"nearest", "idw", "kriging"}:
            raise ValueError("Error: interpolation_method must be nearest, idw, or kriging.")

        def optional_positive(value, name):
            if value is None:
                return None
            numeric = float(value)
            if not np.isfinite(numeric) or numeric <= 0:
                raise ValueError(f"Error: {name} must be a positive finite number.")
            return numeric

        maximum_distance = optional_positive(max_distance_in_meters, "interpolation_max_distance_in_meters")
        variogram_distance = optional_positive(variogram_max_distance_in_meters, "interpolation_variogram_max_distance_in_meters")
        try:
            lags = int(variogram_lags)
        except (TypeError, ValueError) as exc:
            raise TypeError("Error: interpolation_variogram_lags must be an integer.") from exc
        if lags < 1:
            raise ValueError("Error: interpolation_variogram_lags must be positive.")
        if normalized_method == "idw" and maximum_distance is None:
            raise ValueError("Error: interpolation_max_distance_in_meters is required for idw.")
        if normalized_method == "kriging" and variogram_distance is None:
            raise ValueError("Error: interpolation_variogram_max_distance_in_meters is required for kriging.")
        return normalized_method, maximum_distance, lags, variogram_distance

    def _validate_sentinel2_indices(self, indices: list[str] | None) -> tuple[str, ...]:
        """Validate and normalize the optional Sentinel-2 index list."""
        if indices is None:
            indices = []
        if not isinstance(indices, list):
            raise TypeError("Error: sentinel2_indices must be a list of index names.")
        normalized = tuple(dict.fromkeys(str(name).strip().upper() for name in indices))
        # Ordered de-duplication keeps output-band ordering deterministic.
        unsupported = sorted(set(normalized).difference(self.SUPPORTED_SENTINEL2_INDICES))
        if unsupported:
            raise ValueError(
                f"Error: Unsupported Sentinel-2 indices: {unsupported}. Supported indices: {sorted(self.SUPPORTED_SENTINEL2_INDICES)}"
            )
        return normalized

    def _validate_sentinel1_indices(self, indices: list[str] | None) -> tuple[str, ...]:
        """Validate and normalize optional indices derived from Sentinel-1 linear power."""
        if indices is None:
            indices = []
        if not isinstance(indices, list):
            raise TypeError("Error: sentinel1_indices must be a list of index names.")
        normalized = tuple(dict.fromkeys(str(name).strip().upper() for name in indices))
        unsupported = sorted(set(normalized).difference(self.SUPPORTED_SENTINEL1_INDICES))
        if unsupported:
            raise ValueError(
                f"Error: Unsupported Sentinel-1 indices: {unsupported}. Supported indices: {sorted(self.SUPPORTED_SENTINEL1_INDICES)}"
            )
        return normalized

    def _validate_spatial_statistics(self, statistics: list[str] | None) -> tuple[str, ...]:
        """Normalize per-layer reducers and guarantee that mean is calculated.

        ``count`` remains accepted for configuration compatibility, but pixel
        counts are now emitted once per parcel instead of once per layer.
        """
        if statistics is None:
            statistics = ["mean"]
        if not isinstance(statistics, list):
            raise TypeError("Error: spatial_statistics must be a list of reducer names.")
        normalized = []
        # Resolve aliases before de-duplication so synonymous reducers cannot create duplicate columns.
        for statistic in statistics:
            name = str(statistic).strip().lower()
            name = self.SPATIAL_STATISTIC_ALIASES.get(name, name)
            if name not in normalized:
                normalized.append(name)
        unsupported = sorted(set(normalized).difference(self.SUPPORTED_SPATIAL_STATISTICS))
        if unsupported:
            raise ValueError(
                f"Error: Unsupported spatial statistics: {unsupported}. Supported statistics: {list(self.SUPPORTED_SPATIAL_STATISTICS)}"
            )
        return tuple(["mean", *[name for name in normalized if name not in {"mean", "count"}]])

    def _validate_sentinel1_bands(self, bands: list[str] | None) -> tuple[str, ...]:
        """Normalize the optional Sentinel-1 polarization list."""
        if bands is None:
            bands = []
        if not isinstance(bands, list):
            raise TypeError("Error: sentinel1_bands must be a list of band names.")
        normalized = tuple(dict.fromkeys(str(name).strip().upper() for name in bands))
        unsupported = sorted(set(normalized).difference(self.SUPPORTED_SENTINEL1_BANDS))
        if unsupported:
            raise ValueError(
                f"Error: Unsupported Sentinel-1 bands: {unsupported}. Supported bands: {list(self.SUPPORTED_SENTINEL1_BANDS)}"
            )
        return normalized

    def _validate_sentinel1_orbit_direction(self, direction: str) -> str:
        """Normalize the Sentinel-1 orbit-direction selection."""
        if not isinstance(direction, str):
            raise TypeError("Error: sentinel1_orbit_direction must be a string.")
        normalized = direction.strip().upper()
        if normalized not in self.SUPPORTED_SENTINEL1_ORBIT_DIRECTIONS:
            raise ValueError("Error: sentinel1_orbit_direction must be ASCENDING, DESCENDING, or BOTH.")
        return normalized

    def _sentinel1_load_options(self) -> dict:
        """Return the optional openEO collection-property orbit filter."""
        if self.sentinel1_orbit_direction == "BOTH":
            return {}
        orbit_direction = self.sentinel1_orbit_direction
        return {"properties": {"sat:orbit_state": lambda value: value == orbit_direction}}

    def _validate_sentinel2_bands(self, bands: list[str] | None) -> tuple[str, ...]:
        """Normalize Sentinel-2 outputs, allowing an explicit empty selection."""
        if bands is None:
            return self.DEFAULT_SENTINEL2_BANDS
        if not isinstance(bands, list):
            raise TypeError("Error: sentinel2_bands must be a list of band names.")
        normalized = tuple(dict.fromkeys(str(name).strip().upper() for name in bands))
        unsupported = sorted(set(normalized).difference(self.SUPPORTED_SENTINEL2_BANDS))
        if unsupported:
            raise ValueError(
                f"Error: Unsupported Sentinel-2 bands: {unsupported}. Supported bands: {list(self.SUPPORTED_SENTINEL2_BANDS)}"
            )
        return normalized

    def _validate_sensor_selection(self) -> None:
        """Require at least one output from Sentinel-1 or Sentinel-2."""
        if not self._output_sensor_variables():
            raise ValueError("Error: Select at least one Sentinel-2 band/index or Sentinel-1 band/index.")

    def _validate_cleaning_options(
        self,
        remove_outliers: bool,
        iqr_quantiles: tuple[float, float],
        iqr_multiplier: float,
        fill_nulls: bool,
        iqr_min_valid_pixels: int,
        temporal_fill_mode: str,
        minimum_parcel_pixels: int,
        minimum_observed_fraction_for_fill: float,
    ) -> tuple[bool, tuple[float, float], float, bool, int, str, int, float]:
        """Validate and normalize raster-cleaning configuration."""
        if not isinstance(remove_outliers, bool):
            raise TypeError("Error: remove_outliers must be a bool.")
        if not isinstance(fill_nulls, bool):
            raise TypeError("Error: fill_nulls must be a bool.")
        if not isinstance(iqr_quantiles, (tuple, list)) or len(iqr_quantiles) != 2:
            raise TypeError("Error: iqr_quantiles must contain exactly two numbers.")
        try:
            lower_quantile, upper_quantile = map(float, iqr_quantiles)
            multiplier = float(iqr_multiplier)
        except (TypeError, ValueError) as exc:
            raise TypeError("Error: iqr_quantiles and iqr_multiplier must contain numeric values.") from exc
        if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
            raise ValueError("Error: iqr_quantiles must satisfy 0 <= lower < upper <= 1.")
        if multiplier < 0.0 or not np.isfinite(multiplier):
            raise ValueError("Error: iqr_multiplier must be a finite non-negative number.")
        if isinstance(iqr_min_valid_pixels, bool) or isinstance(minimum_parcel_pixels, bool):
            raise TypeError("Error: Pixel-count thresholds must be positive integers.")
        try:
            minimum_iqr_sample = int(iqr_min_valid_pixels)
            minimum_pixel_count = int(minimum_parcel_pixels)
        except (TypeError, ValueError) as exc:
            raise TypeError("Error: Pixel-count thresholds must be positive integers.") from exc
        if minimum_iqr_sample < 1 or minimum_pixel_count < 1:
            raise ValueError("Error: Pixel-count thresholds must be at least 1.")
        normalized_fill_mode = str(temporal_fill_mode).strip().lower()
        if normalized_fill_mode not in ["past_only", "bidirectional"]:
            raise ValueError("Error: temporal_fill_mode must be 'past_only' or 'bidirectional'.")
        try:
            minimum_fill_fraction = float(minimum_observed_fraction_for_fill)
        except (TypeError, ValueError) as exc:
            raise TypeError("Error: minimum_observed_fraction_for_fill must be a number.") from exc
        if not np.isfinite(minimum_fill_fraction) or not 0.0 <= minimum_fill_fraction <= 1.0:
            raise ValueError("Error: minimum_observed_fraction_for_fill must be between 0 and 1.")
        return (
            remove_outliers,
            (lower_quantile, upper_quantile),
            multiplier,
            fill_nulls,
            minimum_iqr_sample,
            normalized_fill_mode,
            minimum_pixel_count,
            minimum_fill_fraction,
        )

    def _required_sentinel2_bands(self) -> tuple[str, ...]:
        """Return selected and index-source Sentinel-2 bands without duplicates."""
        bands = list(self.sentinel2_bands)
        # Index source bands are downloaded even when they are not requested as final outputs.
        for index_name in self.sentinel2_indices:
            bands.extend(self.SUPPORTED_SENTINEL2_INDICES[index_name])
        return tuple(dict.fromkeys(bands))

    def _required_sentinel1_bands(self) -> tuple[str, ...]:
        """Return selected and index-source Sentinel-1 bands without duplicates."""
        bands = list(self.sentinel1_bands)
        for index_name in self.sentinel1_indices:
            bands.extend(self.SUPPORTED_SENTINEL1_INDICES[index_name])
        return tuple(dict.fromkeys(bands))

    def _output_sensor_variables(self) -> tuple[str, ...]:
        """Return ordered physical-band and pixel-index output labels."""
        return (*self.optical_bands, *self.optical_indices, *self.sentinel1_bands, *self.sentinel1_indices)

    def _cube_sensor_variables(self) -> tuple[str, ...]:
        """Return physical bands that must be cleaned before local index calculation."""
        return (*self._required_sentinel2_bands(), *self._required_sentinel1_bands())

    def _run_signature(self) -> str:
        """Create a cache key from processing options, parcel IDs and geometry."""
        configuration = {
            # Any option capable of changing downloaded or cleaned pixels must invalidate this cache namespace.
            # Retain the established raw-cube namespace so compatible downloaded NetCDF files remain reusable.
            "processing_strategy": "sentinel1_sentinel2_temporal_local_v3",
            "start_date": self.start_date.strftime("%Y-%m-%d"),
            "end_date": self.end_date.strftime("%Y-%m-%d"),
            "temporal_period": self.temporal_period,
            "temporal_reducer": self.temporal_reducer,
            "working_epsg": self.working_epsg,
            "sentinel2_collection": self.SENTINEL2_COLLECTION,
            "sentinel2_bands": self.sentinel2_bands,
            "sentinel2_indices": self.sentinel2_indices,
            "sentinel1_collection": self.SENTINEL1_COLLECTION,
            "sentinel1_bands": self.sentinel1_bands,
            "sentinel1_indices": self.sentinel1_indices,
            "sentinel1_orbit_direction": self.sentinel1_orbit_direction,
            "sentinel1_backscatter_coefficient": self.SENTINEL1_BACKSCATTER_COEFFICIENT,
            "sentinel1_backscatter_scale": self.SENTINEL1_BACKSCATTER_SCALE,
            "sentinel1_elevation_model": self.SENTINEL1_ELEVATION_MODEL,
            "target_resolution_metres": self.TARGET_RESOLUTION_METRES,
            "sentinel2_scene_classification_band": self.SENTINEL2_SCENE_CLASSIFICATION_BAND,
            "sentinel2_reflectance_scale_factor": self.SENTINEL2_REFLECTANCE_SCALE_FACTOR,
            "sentinel2_max_scene_cloud_cover": self.SENTINEL2_MAX_SCENE_CLOUD_COVER,
            "sentinel2_invalid_scl_classes": self.SENTINEL2_INVALID_SCL_CLASSES,
            "sentinel2_buffered_scl_classes": self.SENTINEL2_BUFFERED_SCL_CLASSES,
            "remove_outliers": self.remove_outliers,
            "iqr_quantiles": self.iqr_quantiles,
            "iqr_multiplier": self.iqr_multiplier,
            "iqr_min_valid_pixels": self.iqr_min_valid_pixels,
            "fill_nulls": self.fill_nulls,
            "temporal_fill_mode": self.temporal_fill_mode,
            "minimum_parcel_pixels": self.minimum_parcel_pixels,
            "minimum_observed_fraction_for_fill": self.minimum_observed_fraction_for_fill,
            "interpolation_method": self.interpolation_method,
            "interpolation_max_distance_in_meters": self.interpolation_max_distance_in_meters,
        }
        digest = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode("utf-8"))
        # Parcel identity and exact geometry complete the content-addressed cache key.
        for parcel_id, geometry in zip(self.parcels[self.PARCEL_ID_FIELD], self.parcels.geometry):
            digest.update(str(parcel_id).encode("utf-8"))
            digest.update(geometry.wkb)
        return digest.hexdigest()[:16]
