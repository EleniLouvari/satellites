"""Sentinel-1 and Sentinel-2 temporal zonal statistics for parcels."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterator, Sequence

import geopandas as gpd
import numpy as np
import openeo
import pandas as pd
import rioxarray  # noqa: F401 - registers the xarray ``rio`` accessor.
import xarray as xr
from data_analysis.fill_nulls.fill_nulls_library import (
    fill_null_values_using_interpolation,
)
from pyproj import CRS
from rasterio.features import geometry_mask
from rasterio.transform import xy
from rasterio.windows import Window, from_bounds, transform as window_transform
from shapely import make_valid
from shapely.geometry import GeometryCollection, MultiPolygon, Point, Polygon, mapping
from shapely.ops import unary_union


class SatelliteZonalStats:
    """Calculate temporal Sentinel-1 and Sentinel-2 parcel features.

    Parameters
    ----------
    parcels:
        Parcel polygons. A unique ``parcel_id`` column is retained when present;
        otherwise the GeoDataFrame index is converted into parcel IDs.
    start_date, end_date:
        Inclusive ISO-style dates accepted by :func:`pandas.Timestamp`.
    output_dir:
        Directory for cached monthly NetCDF cubes and combined CSV/Parquet results.
    working_epsg:
        Projected CRS used for spatial batching, parcel metrics and final
        interpolation. The openEO raster remains in its native projected CRS.
    parcel_id_field:
        Name of the column containing unique parcel identifiers. If the column
        is absent, stable string identifiers are created from the input index.
    sentinel2_bands:
        Sentinel-2 L2A reflectance bands to include in the output. When omitted,
        :attr:`DEFAULT_SENTINEL2_BANDS` is used. Extra bands required by indices
        are loaded automatically without being added as standalone outputs.
        Pass an empty list to disable standalone Sentinel-2 bands.
    calculate_sentinel2_indices:
        If ``True``, calculate the requested spectral indices on openEO.
    sentinel2_indices:
        List of index names. Supported values are NDVI, NDWI, MNDWI, NDMI,
        NBR, GNDVI, EVI, SAVI, MSAVI, NDRE, PSRI and CI. Required when
        ``calculate_sentinel2_indices`` is ``True``.
    sentinel1_bands:
        Optional Sentinel-1 GRD polarizations to add to the monthly cube.
        Supported values are ``VV`` and ``VH``. Backscatter is calculated as
        linear-power sigma0 over the ellipsoid. At least one Sentinel-1 band,
        Sentinel-2 band, or Sentinel-2 index must be selected overall.
    spatial_statistics:
        Optional parcel statistics calculated locally from the monthly cube.
        ``mean`` is always included; supported additions are median, sd, min,
        max, count and selected percentiles.
    remove_outliers:
        If ``True``, calculate IQR bounds independently for each variable and
        monthly cube slice, then replace outlying pixels with ``NaN`` before
        parcel aggregation.
    iqr_quantiles:
        Lower and upper quantiles used to calculate the IQR. The standard
        values are ``(0.25, 0.75)``.
    iqr_multiplier:
        Non-negative multiplier applied below Q1 and above Q3. The standard
        Tukey value is ``1.5``.
    fill_nulls:
        If ``True``, fill post-IQR pixel nulls using temporal neighbors,
        a 3x3 spatial mean, and finally the configured spatial interpolator
        independently inside each parcel.
    iqr_min_valid_pixels:
        Minimum valid parcel pixels required before applying the IQR filter.
    temporal_fill_mode:
        Temporal source direction used when filling a missing pixel. Use
        ``'past_only'`` to carry the latest earlier value forward. Use
        ``'bidirectional'`` to average the closest earlier and later values;
        if only one direction is available, that value is used.
    minimum_parcel_pixels:
        Minimum eligible pixels for the emitted parcel-quality flag. Smaller
        parcels are retained rather than discarded.
    minimum_observed_fraction_for_fill:
        Minimum observed share required before a parcel-variable-period may be
        imputed or used as a temporal filling source.
    temporal_period:
        Temporal bin width. Use ``'15D'`` for 15 days or ``'1M'``, ``'2M'``
        and ``'3M'`` for one-, two- and three-month bins.
    temporal_reducer:
        Reducer applied to acquisitions in each temporal bin. Supported values
        are ``'mean'``, ``'median'``, ``'min'``, ``'max'`` and ``'sum'``.
    interpolation_method:
        Final pixel-level interpolation applied per variable and temporal period.
        Choose ``'nearest'``, ``'idw'`` or ``'kriging'``.
    interpolation_max_distance_in_meters:
        Required maximum search distance in metres when using IDW.
    interpolation_variogram_lags, interpolation_variogram_max_distance_in_meters:
        Kriging variogram configuration; maximum distance is expressed in metres.
    batch_workers:
        Maximum number of openEO jobs run concurrently and worker processes used
        for local batch statistics. A value of 1 keeps sequential behaviour.
    openeo_username, openeo_password:
        Optional CDSE credentials used with the OIDC resource-owner-password
        flow. Both values must be supplied together. When omitted, the normal
        cached/interactive OIDC authentication flow is used.
    """

    SENTINEL2_COLLECTION = "SENTINEL2_L2A"
    SUPPORTED_SENTINEL2_BANDS = (
        "B01", "B02", "B03", "B04", "B05", "B06", "B07",
        "B08", "B8A", "B09", "B11", "B12",
    )
    DEFAULT_SENTINEL2_BANDS = ("B02", "B03", "B04", "B05", "B08")
    SENTINEL2_SCENE_CLASSIFICATION_BAND = "SCL"
    SENTINEL2_REFLECTANCE_SCALE_FACTOR = 0.0001
    SENTINEL2_MAX_SCENE_CLOUD_COVER = 70
    SENTINEL2_INVALID_SCL_CLASSES = (0, 1, 3, 8, 9, 10, 11)
    SENTINEL2_BUFFERED_SCL_CLASSES = (3, 8, 9, 10, 11)

    SENTINEL1_COLLECTION = "SENTINEL1_GRD"
    SUPPORTED_SENTINEL1_BANDS = ("VV", "VH")
    SENTINEL1_BACKSCATTER_COEFFICIENT = "sigma0-ellipsoid"
    SENTINEL1_ELEVATION_MODEL = "COPERNICUS_30"

    TARGET_RESOLUTION_METRES = 10
    MAX_FEATURES_PER_JOB = 5_000
    GRID_SIZE_METRES = 50_000
    PARCEL_ID_FIELD = "parcel_id"
    OPENEO_URL = "https://openeo.dataspace.copernicus.eu"
    LOG_FILE_NAME = "satellite_zonal_stats.log"

    # Band definitions use Sentinel-2 collection band names. NDWI follows the
    # McFeeters green/NIR convention; NDMI represents the NIR/SWIR moisture index.
    SUPPORTED_SENTINEL2_INDICES = {
        "NDVI": ("B08", "B04"),
        "NDWI": ("B03", "B08"),
        "MNDWI": ("B03", "B11"),
        "NDMI": ("B08", "B11"),
        "NBR": ("B08", "B12"),
        "GNDVI": ("B08", "B03"),
        "EVI": ("B08", "B04", "B02"),
        "SAVI": ("B08", "B04"),
        "MSAVI": ("B08", "B04"),
        "NDRE": ("B08", "B05"),
        "PSRI": ("B04", "B02", "B06"),
        "CI": ("B08", "B05"),
    }
    SUPPORTED_SPATIAL_STATISTICS = (
        "mean",
        "median",
        "sd",
        "min",
        "max",
        "count",
        "p10",
        "p25",
        "p75",
        "p90",
    )
    QUANTILE_PROBABILITIES = {
        "p10": 0.10,
        "p25": 0.25,
        "p75": 0.75,
        "p90": 0.90,
    }
    SPATIAL_STATISTIC_ALIASES = {
        "average": "mean",
        "std": "sd",
        "stdev": "sd",
        "standard_deviation": "sd",
    }

    def __init__(
        self,
        parcels: gpd.GeoDataFrame,
        start_date: str | pd.Timestamp,
        end_date: str | pd.Timestamp,
        output_dir: str | Path,
        working_epsg: int,
        parcel_id_field: str = "parcel_id",
        sentinel2_bands: list[str] | None = None,
        calculate_sentinel2_indices: bool = False,
        sentinel2_indices: list[str] | None = None,
        sentinel1_bands: list[str] | None = None,
        spatial_statistics: list[str] | None = None,
        remove_outliers: bool = False,
        iqr_quantiles: tuple[float, float] = (0.25, 0.75),
        iqr_multiplier: float = 1.5,
        fill_nulls: bool = False,
        temporal_period: str = "1M",  # Valid values: '1M', '3M', '6M', '1Y', '15D', etc.
        temporal_reducer: str = "median",
        interpolation_method: str = "nearest",
        interpolation_max_distance_in_meters: float | None = None,
        interpolation_variogram_lags: int = 15,
        interpolation_variogram_max_distance_in_meters: float | None = None,
        batch_workers: int = 1,
        iqr_min_valid_pixels: int = 20,
        temporal_fill_mode: str = "past_only",
        minimum_parcel_pixels: int = 3,
        minimum_observed_fraction_for_fill: float = 0.20,
        openeo_username: str | None = None,
        openeo_password: str | None = None,
    ) -> None:
        """Validate inputs and prepare a reusable extraction instance."""

        self.start_date, self.end_date = self._validate_dates(start_date, end_date)
        self.temporal_period, self.temporal_reducer = self._validate_temporal_aggregation(
            temporal_period,
            temporal_reducer,
        )
        (
            self.interpolation_method,
            self.interpolation_max_distance_in_meters,
            self.interpolation_variogram_lags,
            self.interpolation_variogram_max_distance_in_meters,
        ) = self._validate_interpolation_options(
            interpolation_method,
            interpolation_max_distance_in_meters,
            interpolation_variogram_lags,
            interpolation_variogram_max_distance_in_meters,
        )
        self.output_dir = Path(output_dir)
        self.working_epsg = self._validate_working_epsg(working_epsg)
        self.openeo_username, self.openeo_password = self._validate_openeo_credentials(
            openeo_username,
            openeo_password,
        )
        if not isinstance(parcel_id_field, str):
            raise TypeError("parcel_id_field must be a string.")
        parcel_id_field = parcel_id_field.strip()
        if not parcel_id_field or parcel_id_field == "geometry":
            raise ValueError("parcel_id_field must be a non-empty, non-geometry column name.")
        self.PARCEL_ID_FIELD = parcel_id_field
        self.batch_workers = self._validate_batch_workers(batch_workers)
        self.sentinel2_bands = self._validate_sentinel2_bands(sentinel2_bands)
        if not isinstance(calculate_sentinel2_indices, bool):
            raise TypeError("calculate_sentinel2_indices must be a bool.")
        self.calculate_sentinel2_indices = calculate_sentinel2_indices
        self.sentinel2_indices = self._validate_sentinel2_indices(sentinel2_indices)
        self.sentinel1_bands = self._validate_sentinel1_bands(sentinel1_bands)
        self._validate_sensor_selection()
        self.spatial_statistics = self._validate_spatial_statistics(spatial_statistics)
        (
            self.remove_outliers,
            self.iqr_quantiles,
            self.iqr_multiplier,
            self.fill_nulls,
            self.iqr_min_valid_pixels,
            self.temporal_fill_mode,
            self.minimum_parcel_pixels,
            self.minimum_observed_fraction_for_fill,
        ) = self._validate_cleaning_options(
            remove_outliers=remove_outliers,
            iqr_quantiles=iqr_quantiles,
            iqr_multiplier=iqr_multiplier,
            fill_nulls=fill_nulls,
            iqr_min_valid_pixels=iqr_min_valid_pixels,
            temporal_fill_mode=temporal_fill_mode,
            minimum_parcel_pixels=minimum_parcel_pixels,
            minimum_observed_fraction_for_fill=minimum_observed_fraction_for_fill,
        )
        # Start file logging only after non-spatial configuration is valid.
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = self._create_logger()
        self.parcels = self._prepare_parcels(parcels)
        self.logger.info(
            f"Configured Sentinel-2 bands={list(self.sentinel2_bands)}, "
            f"Sentinel-2 indices={list(self.sentinel2_indices)}, "
            f"Sentinel-1 bands={list(self.sentinel1_bands)}, "
            f"parcel statistics={list(self.spatial_statistics)}, "
            f"remove_outliers={self.remove_outliers}, "
            f"iqr_quantiles={self.iqr_quantiles}, "
            f"iqr_multiplier={self.iqr_multiplier}, and "
            f"fill_nulls={self.fill_nulls}, "
            f"temporal_period={self.temporal_period}, and "
            f"temporal_reducer={self.temporal_reducer}, and "
            f"batch_workers={self.batch_workers}."
        )

    @staticmethod
    def _validate_openeo_credentials(
        username: str | None,
        password: str | None,
    ) -> tuple[str | None, str | None]:
        """Validate an optional openEO username/password pair."""

        if (username is None) != (password is None):
            raise ValueError("openeo_username and openeo_password must be supplied together.")
        if username is None:
            return None, None
        if not isinstance(username, str) or not username.strip():
            raise ValueError("openeo_username must be a non-empty string.")
        if not isinstance(password, str) or not password:
            raise ValueError("openeo_password must be a non-empty string.")
        return username.strip(), password

    def _authenticate_openeo_connection(self, connection: openeo.Connection) -> openeo.Connection:
        """Authenticate with explicit credentials or the default OIDC flow."""

        if self.openeo_username is not None:
            connection.authenticate_oidc_resource_owner_password_credentials(
                username=self.openeo_username,
                password=self.openeo_password,
            )
        else:
            connection.authenticate_oidc()
        return connection

    def __getstate__(self) -> dict:
        """Exclude non-picklable logging handlers from worker payloads."""

        state = self.__dict__.copy()
        state.pop("logger", None)
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore lightweight console logging inside a worker process."""

        self.__dict__.update(state)
        logger = logging.getLogger(f"{__name__}.worker.{os.getpid()}")
        if not logger.handlers:
            logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.INFO)
        logger.propagate = False
        self.logger = logger

    def _validate_batch_workers(self, batch_workers: int) -> int:
        """Return a positive process count for local batch calculations."""

        if isinstance(batch_workers, bool):
            raise TypeError("batch_workers must be a positive integer.")
        try:
            workers = int(batch_workers)
        except (TypeError, ValueError) as exc:
            raise TypeError("batch_workers must be a positive integer.") from exc
        if workers < 1:
            raise ValueError("batch_workers must be at least 1.")
        return workers

    def _create_logger(self) -> logging.Logger:
        """Log progress to the console and to the extraction output directory."""

        # A per-instance name prevents handlers from being shared by two runs.
        logger = logging.getLogger(f"{__name__}.{id(self)}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        for handler in (
            logging.StreamHandler(),
            logging.FileHandler(self.output_dir / self.LOG_FILE_NAME, encoding="utf-8"),
        ):
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def _validate_dates(
        self,
        start_date: str | pd.Timestamp,
        end_date: str | pd.Timestamp,
    ) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Parse the temporal bounds and verify that they are ordered."""

        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        if pd.isna(start) or pd.isna(end):
            raise ValueError("start_date and end_date must be valid dates.")
        if start > end:
            raise ValueError("start_date must be on or before end_date.")
        return start, end

    def _validate_working_epsg(self, working_epsg: int) -> int:
        """Return a valid projected EPSG code suitable for metric operations."""

        try:
            epsg = int(working_epsg)
            crs = CRS.from_epsg(epsg)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid EPSG code: {working_epsg!r}.") from exc
        if not crs.is_projected:
            raise ValueError("working_epsg must identify a projected CRS.")
        return epsg

    def _validate_temporal_aggregation(
        self,
        temporal_period: str,
        temporal_reducer: str,
    ) -> tuple[str, str]:
        """Normalize a day/month bin width and its acquisition reducer."""

        if not isinstance(temporal_period, str):
            raise TypeError("temporal_period must be a string such as 15D or 2M.")
        aliases = {
            "DAY": "1D",
            "DAILY": "1D",
            "MONTH": "1M",
            "MONTHLY": "1M",
        }
        raw_period = temporal_period.strip().upper()
        period = aliases.get(raw_period, raw_period)
        match = re.fullmatch(r"([1-9][0-9]*)(D|M|Y)", period)
        if match is None:
            raise ValueError("temporal_period must be a positive day/month/year interval such as 15D, 1M, 2M, 3M or 1Y.")

        reducer = str(temporal_reducer).strip().lower()
        reducer = {"average": "mean", "avg": "mean"}.get(reducer, reducer)
        supported_reducers = {"mean", "median", "min", "max", "sum"}
        if reducer not in supported_reducers:
            raise ValueError(
                f"Unsupported temporal_reducer: {temporal_reducer!r}. " f"Supported reducers: {sorted(supported_reducers)}"
            )
        return period, reducer

    def _temporal_intervals(self) -> tuple[list[list[str]], list[str]]:
        """Return explicit half-open openEO intervals and start-date labels."""

        match = re.fullmatch(r"([1-9][0-9]*)(D|M|Y)", self.temporal_period)
        if match is None:  # Guarded during initialization.
            raise RuntimeError(f"Invalid temporal period: {self.temporal_period}")
        amount = int(match.group(1))
        unit = match.group(2)
        if unit == "D":
            step = pd.Timedelta(days=amount)
        elif unit == "M":
            step = pd.DateOffset(months=amount)
        else:
            step = pd.DateOffset(years=amount)

        cursor = self.start_date.normalize()
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
        """Validate parcel-level spatial interpolation configuration."""

        normalized_method = str(method).strip().lower()
        aliases = {
            "nearest_neighbor": "nearest",
            "inverse_distance_weighting": "idw",
        }
        normalized_method = aliases.get(normalized_method, normalized_method)
        if normalized_method not in {"nearest", "idw", "kriging"}:
            raise ValueError("interpolation_method must be nearest, idw, or kriging.")

        def optional_positive(value, name):
            """Normalize an optional distance and reject non-positive values."""

            if value is None:
                return None
            numeric = float(value)
            if not np.isfinite(numeric) or numeric <= 0:
                raise ValueError(f"{name} must be a positive finite number.")
            return numeric

        maximum_distance = optional_positive(
            max_distance_in_meters,
            "interpolation_max_distance_in_meters",
        )
        variogram_distance = optional_positive(
            variogram_max_distance_in_meters,
            "interpolation_variogram_max_distance_in_meters",
        )
        try:
            lags = int(variogram_lags)
        except (TypeError, ValueError) as exc:
            raise TypeError("interpolation_variogram_lags must be an integer.") from exc
        if lags < 1:
            raise ValueError("interpolation_variogram_lags must be positive.")
        if normalized_method == "idw" and maximum_distance is None:
            raise ValueError("interpolation_max_distance_in_meters is required for idw.")
        if normalized_method == "kriging" and variogram_distance is None:
            raise ValueError("interpolation_variogram_max_distance_in_meters is required for kriging.")
        return normalized_method, maximum_distance, lags, variogram_distance

    def _validate_sentinel2_indices(self, indices: list[str] | None) -> tuple[str, ...]:
        """Validate and normalize the optional Sentinel-2 index list."""

        if indices is None:
            indices = []
        if not isinstance(indices, list):
            raise TypeError("sentinel2_indices must be a list of index names.")

        normalized = tuple(dict.fromkeys(str(name).strip().upper() for name in indices))
        unsupported = sorted(set(normalized).difference(self.SUPPORTED_SENTINEL2_INDICES))
        if unsupported:
            raise ValueError(
                f"Unsupported Sentinel-2 indices: {unsupported}. "
                f"Supported indices: {sorted(self.SUPPORTED_SENTINEL2_INDICES)}"
            )
        if self.calculate_sentinel2_indices and not normalized:
            raise ValueError(
                "sentinel2_indices must contain at least one name when "
                "calculate_sentinel2_indices=True."
            )
        if not self.calculate_sentinel2_indices and normalized:
            raise ValueError(
                "Set calculate_sentinel2_indices=True when providing sentinel2_indices."
            )
        return normalized

    def _validate_spatial_statistics(self, statistics: list[str] | None) -> tuple[str, ...]:
        """Normalize parcel reducers and guarantee that mean is calculated."""

        if statistics is None:
            statistics = ["mean"]
        if not isinstance(statistics, list):
            raise TypeError("spatial_statistics must be a list of reducer names.")

        normalized = []
        for statistic in statistics:
            name = str(statistic).strip().lower()
            name = self.SPATIAL_STATISTIC_ALIASES.get(name, name)
            if name not in normalized:
                normalized.append(name)
        unsupported = sorted(set(normalized).difference(self.SUPPORTED_SPATIAL_STATISTICS))
        if unsupported:
            raise ValueError(
                f"Unsupported spatial statistics: {unsupported}. Supported statistics: {list(self.SUPPORTED_SPATIAL_STATISTICS)}"
            )

        # Parcel mean is the mandatory final step in the requested workflow.
        normalized = ["mean", *[name for name in normalized if name != "mean"]]
        return tuple(normalized)

    def _validate_sentinel1_bands(self, bands: list[str] | None) -> tuple[str, ...]:
        """Normalize the optional Sentinel-1 polarization list."""

        if bands is None:
            bands = []
        if not isinstance(bands, list):
            raise TypeError("sentinel1_bands must be a list of band names.")
        normalized = tuple(dict.fromkeys(str(name).strip().upper() for name in bands))
        unsupported = sorted(set(normalized).difference(self.SUPPORTED_SENTINEL1_BANDS))
        if unsupported:
            raise ValueError(
                f"Unsupported Sentinel-1 bands: {unsupported}. "
                f"Supported bands: {list(self.SUPPORTED_SENTINEL1_BANDS)}"
            )
        return normalized

    def _validate_sentinel2_bands(self, bands: list[str] | None) -> tuple[str, ...]:
        """Normalize Sentinel-2 outputs, allowing an explicit empty selection."""

        if bands is None:
            return self.DEFAULT_SENTINEL2_BANDS
        if not isinstance(bands, list):
            raise TypeError("sentinel2_bands must be a list of band names.")
        normalized = tuple(dict.fromkeys(str(name).strip().upper() for name in bands))
        unsupported = sorted(set(normalized).difference(self.SUPPORTED_SENTINEL2_BANDS))
        if unsupported:
            raise ValueError(
                f"Unsupported Sentinel-2 bands: {unsupported}. "
                f"Supported bands: {list(self.SUPPORTED_SENTINEL2_BANDS)}"
            )
        return normalized

    def _validate_sensor_selection(self) -> None:
        """Require at least one output from Sentinel-1 or Sentinel-2."""

        if not self._output_sensor_variables():
            raise ValueError(
                "Select at least one Sentinel-2 band, Sentinel-2 index, "
                "or Sentinel-1 band."
            )

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
        """Validate and normalize pixel-cleaning configuration."""

        if not isinstance(remove_outliers, bool):
            raise TypeError("remove_outliers must be a bool.")
        if not isinstance(fill_nulls, bool):
            raise TypeError("fill_nulls must be a bool.")
        if not isinstance(iqr_quantiles, (tuple, list)) or len(iqr_quantiles) != 2:
            raise TypeError("iqr_quantiles must contain exactly two numbers.")
        try:
            lower_quantile, upper_quantile = map(float, iqr_quantiles)
            multiplier = float(iqr_multiplier)
        except (TypeError, ValueError) as exc:
            raise TypeError("iqr_quantiles and iqr_multiplier must contain numeric values.") from exc
        if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
            raise ValueError("iqr_quantiles must satisfy 0 <= lower < upper <= 1.")
        if multiplier < 0.0 or not np.isfinite(multiplier):
            raise ValueError("iqr_multiplier must be a finite non-negative number.")
        if isinstance(iqr_min_valid_pixels, bool) or isinstance(minimum_parcel_pixels, bool):
            raise TypeError("Pixel-count thresholds must be positive integers.")
        try:
            minimum_iqr_sample = int(iqr_min_valid_pixels)
            minimum_pixel_count = int(minimum_parcel_pixels)
        except (TypeError, ValueError) as exc:
            raise TypeError("Pixel-count thresholds must be positive integers.") from exc
        if minimum_iqr_sample < 1 or minimum_pixel_count < 1:
            raise ValueError("Pixel-count thresholds must be at least 1.")

        normalized_fill_mode = str(temporal_fill_mode).strip().lower()
        if normalized_fill_mode not in ["past_only", "bidirectional"]:
            raise ValueError("temporal_fill_mode must be 'past_only' or 'bidirectional'.")
        try:
            minimum_fill_fraction = float(minimum_observed_fraction_for_fill)
        except (TypeError, ValueError) as exc:
            raise TypeError("minimum_observed_fraction_for_fill must be a number.") from exc
        if not np.isfinite(minimum_fill_fraction) or not 0.0 <= minimum_fill_fraction <= 1.0:
            raise ValueError("minimum_observed_fraction_for_fill must be between 0 and 1.")
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
        for index_name in self.sentinel2_indices:
            bands.extend(self.SUPPORTED_SENTINEL2_INDICES[index_name])
        return tuple(dict.fromkeys(bands))

    def _output_sensor_variables(self) -> tuple[str, ...]:
        """Return ordered Sentinel-2, index, and Sentinel-1 output labels."""
        return (*self.sentinel2_bands, *self.sentinel2_indices, *self.sentinel1_bands)

    def _run_signature(self) -> str:
        """Create a cache key from processing options, parcel IDs and geometry."""

        configuration = {
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
            "sentinel1_backscatter_coefficient": self.SENTINEL1_BACKSCATTER_COEFFICIENT,
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
        for parcel_id, geometry in zip(self.parcels[self.PARCEL_ID_FIELD], self.parcels.geometry):
            digest.update(str(parcel_id).encode("utf-8"))
            digest.update(geometry.wkb)
        return digest.hexdigest()[:16]

    def _extract_polygonal_geometry(self, geometry):
        """Repair a geometry and retain only polygonal components."""

        if geometry is None or geometry.is_empty:
            return None
        # Repair self-intersections and other common topology problems.
        geometry = make_valid(geometry)
        if isinstance(geometry, (Polygon, MultiPolygon)):
            return geometry
        if isinstance(geometry, GeometryCollection):
            # Discard non-area components introduced by geometry repair.
            polygon_parts = []
            for part in geometry.geoms:
                if isinstance(part, Polygon):
                    polygon_parts.append(part)
                elif isinstance(part, MultiPolygon):
                    polygon_parts.extend(part.geoms)
            if polygon_parts:
                return unary_union(polygon_parts)
        return None

    def _prepare_parcels(self, parcels: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Validate IDs/geometries and return parcels in WGS84 for openEO."""

        if not isinstance(parcels, gpd.GeoDataFrame):
            raise TypeError("parcels must be a geopandas.GeoDataFrame.")
        if parcels.crs is None:
            raise ValueError("The parcels GeoDataFrame must have a CRS.")

        prepared = parcels.copy()
        # Preserve caller IDs when available; otherwise expose stable index values.
        if self.PARCEL_ID_FIELD in prepared.columns:
            prepared = prepared[[self.PARCEL_ID_FIELD, "geometry"]].copy()
        else:
            prepared = prepared[["geometry"]].copy()
            prepared.insert(0, self.PARCEL_ID_FIELD, parcels.index.map(str))

        prepared[self.PARCEL_ID_FIELD] = prepared[self.PARCEL_ID_FIELD].astype(str)
        prepared["geometry"] = prepared.geometry.map(self._extract_polygonal_geometry)
        prepared = prepared.loc[prepared.geometry.notna() & ~prepared.geometry.is_empty].copy()
        if prepared.empty:
            raise ValueError("No valid Polygon or MultiPolygon parcels were found.")

        duplicated = prepared[self.PARCEL_ID_FIELD].duplicated(keep=False)
        if duplicated.any():
            examples = prepared.loc[duplicated, self.PARCEL_ID_FIELD].unique()[:10]
            raise ValueError(
                f"{self.PARCEL_ID_FIELD} values must be unique. "
                f"Example duplicates: {examples.tolist()}"
            )

        # GeoJSON geometries submitted to openEO must use longitude/latitude.
        prepared = prepared.to_crs("EPSG:4326").reset_index(drop=True)
        self.logger.info(f"Prepared {len(prepared)} valid parcels.")
        return prepared

    def _iter_spatial_batches(self) -> Iterator[gpd.GeoDataFrame]:
        """Yield geographically compact parcel batches within the job-size limit."""

        # Use the projected working CRS so grid cells have metre-based dimensions.
        metric = self.parcels.to_crs(epsg=self.working_epsg)
        centroids = metric.geometry.centroid
        # Assign centroid coordinates to a fixed spatial grid.
        grouping = pd.DataFrame(
            {
                "grid_x": (centroids.x // self.GRID_SIZE_METRES).astype("int64"),
                "grid_y": (centroids.y // self.GRID_SIZE_METRES).astype("int64"),
                "centroid_x": centroids.x,
                "centroid_y": centroids.y,
            },
            index=self.parcels.index,
        )
        for indices in grouping.groupby(["grid_x", "grid_y"]).groups.values():
            order = grouping.loc[list(indices)].sort_values(["centroid_y", "centroid_x"]).index
            cell = self.parcels.loc[order].copy()
            # Split dense cells without mixing them with geographically distant cells.
            for start in range(0, len(cell), self.MAX_FEATURES_PER_JOB):
                yield cell.iloc[start : start + self.MAX_FEATURES_PER_JOB].reset_index(drop=True)

    def _to_feature_collection(self, parcels: gpd.GeoDataFrame) -> dict:
        """Convert one WGS84 parcel batch to an openEO GeoJSON collection."""

        # Feature order is retained because openEO returns zero-based feature indexes.
        return {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "id": str(row[self.PARCEL_ID_FIELD]),
                    "properties": {self.PARCEL_ID_FIELD: str(row[self.PARCEL_ID_FIELD])},
                    "geometry": mapping(row.geometry),
                }
                for _, row in parcels.iterrows()
            ],
        }

    def _combine_class_conditions(self, scl_cube, classes: Sequence[int]):
        """Build an openEO Boolean cube matching any requested SCL class."""

        condition = scl_cube == classes[0]
        for class_id in classes[1:]:
            condition = condition | (scl_cube == class_id)
        return condition

    def _normalized_difference(self, left, right):
        """Calculate a masked normalized difference between two pixel cubes."""

        denominator = left + right
        index_cube = (left - right) / denominator
        # Avoid infinite index values where the two reflectances sum to zero.
        return index_cube.mask(denominator == 0)

    def _calculate_sentinel2_index(self, sentinel2_reflectance, index_name: str):
        """Calculate one Sentinel-2 index for every valid acquisition pixel."""

        blue = sentinel2_reflectance.band("B02") if index_name in {"EVI", "PSRI"} else None
        green = sentinel2_reflectance.band("B03") if index_name in {"NDWI", "MNDWI", "GNDVI"} else None
        red = sentinel2_reflectance.band("B04") if index_name in {"NDVI", "EVI", "SAVI", "MSAVI", "PSRI"} else None
        nir = sentinel2_reflectance.band("B08") if index_name not in {"MNDWI", "PSRI"} else None
        red_edge1 = sentinel2_reflectance.band("B05") if index_name in {"NDRE", "CI"} else None
        red_edge2 = sentinel2_reflectance.band("B06") if index_name == "PSRI" else None

        if index_name == "NDVI":
            index_cube = self._normalized_difference(nir, red)
        elif index_name == "NDWI":
            index_cube = self._normalized_difference(green, nir)
        elif index_name == "MNDWI":
            index_cube = self._normalized_difference(green, sentinel2_reflectance.band("B11"))
        elif index_name == "NDMI":
            index_cube = self._normalized_difference(nir, sentinel2_reflectance.band("B11"))
        elif index_name == "NBR":
            index_cube = self._normalized_difference(nir, sentinel2_reflectance.band("B12"))
        elif index_name == "GNDVI":
            index_cube = self._normalized_difference(nir, green)
        elif index_name == "EVI":
            denominator = nir + (6.0 * red) - (7.5 * blue) + 1.0
            index_cube = (2.5 * (nir - red) / denominator).mask(denominator == 0)
        elif index_name == "SAVI":
            denominator = nir + red + 0.5
            index_cube = (1.5 * (nir - red) / denominator).mask(denominator == 0)
        elif index_name == "MSAVI":
            doubled_nir_plus_one = (2.0 * nir) + 1.0
            discriminant = (doubled_nir_plus_one**2) - (8.0 * (nir - red))
            # Keep the square root inside the band-math graph. Calling ``mask``
            # first exits openEO's band-math mode, so a subsequent ``** 0.5``
            # raises ``BandMathException: Must be in band math mode already``.
            index_cube = (doubled_nir_plus_one - (discriminant**0.5)) / 2.0
            index_cube = index_cube.mask(discriminant < 0)
        elif index_name == "NDRE":
            index_cube = self._normalized_difference(nir, red_edge1)
        elif index_name == "PSRI":
            index_cube = ((red - blue) / red_edge2).mask(red_edge2 == 0)
        elif index_name == "CI":
            index_cube = (nir / red_edge1) - 1.0
        else:  # Guarded by _validate_sentinel2_indices; retained defensively.
            raise ValueError(f"Unsupported index: {index_name}")

        # Restore a one-label bands dimension so the index can join the band cube.
        return index_cube.add_dimension(name="bands", label=index_name, type="bands")

    def _build_sentinel2_output_cube(self, masked_sentinel2_reflectance):
        """Select Sentinel-2 output bands and append requested indices."""

        # Start with standalone bands when requested. Index-only runs avoid an
        # invalid empty filter_bands process by starting from the first index.
        output_cube = None
        if self.sentinel2_bands:
            output_cube = masked_sentinel2_reflectance.filter_bands(
                list(self.sentinel2_bands)
            )
        for index_name in self.sentinel2_indices:
            index_cube = self._calculate_sentinel2_index(masked_sentinel2_reflectance, index_name)
            output_cube = index_cube if output_cube is None else output_cube.merge_cubes(index_cube)
        if output_cube is None:  # Guarded by _validate_sensor_selection.
            raise RuntimeError("No Sentinel-2 output variables were selected.")
        return output_cube

    def _aggregate_temporal_cube(self, cube):
        """Aggregate a sensor cube using the configured shared intervals."""

        if self.temporal_period == "1M" and self.start_date.is_month_start:
            return cube.aggregate_temporal_period(period="month", reducer=self.temporal_reducer)

        intervals, labels = self._temporal_intervals()
        return cube.aggregate_temporal(intervals=intervals, labels=labels, reducer=self.temporal_reducer)

    def _build_multisensor_temporal_cube(self, connection: openeo.Connection, feature_collection: dict):
        """Build the Sentinel-1/Sentinel-2 temporal openEO cube.

        Sentinel-2 and Sentinel-1 are processed independently to the configured
        temporal periods. Sentinel-1 is then aligned to the Sentinel-2 spatial
        grid and merged by bands.
        """

        sentinel2_temporal = None
        if self.sentinel2_bands or self.sentinel2_indices:
            # Load every source band required by an output or index formula.
            required_sentinel2_bands = self._required_sentinel2_bands()
            sentinel2 = connection.load_collection(
                self.SENTINEL2_COLLECTION,
                temporal_extent=[self.start_date.strftime("%Y-%m-%d"), self.end_date.strftime("%Y-%m-%d")],
                bands=[*required_sentinel2_bands, self.SENTINEL2_SCENE_CLASSIFICATION_BAND],
                max_cloud_cover=self.SENTINEL2_MAX_SCENE_CLOUD_COVER,
            ).filter_spatial(feature_collection)

            # Align continuous bands before formulas combine 10 m and 20 m inputs.
            sentinel2_reflectance = sentinel2.filter_bands(list(required_sentinel2_bands)).resample_spatial(
                resolution=self.TARGET_RESOLUTION_METRES,
                projection=f"EPSG:{self.working_epsg}",
                method="bilinear",
            )
            # Scale the reflectance values to the openEO range of 0.0–1.0.
            sentinel2_reflectance = sentinel2_reflectance * self.SENTINEL2_REFLECTANCE_SCALE_FACTOR
            # Mask out invalid surface pixels and any cloud-adjacent pixels.
            sentinel2_scl = sentinel2.band(self.SENTINEL2_SCENE_CLASSIFICATION_BAND).resample_cube_spatial(sentinel2_reflectance, method="near")
            invalid_surface = self._combine_class_conditions(sentinel2_scl, self.SENTINEL2_INVALID_SCL_CLASSES)
            cloud_related = self._combine_class_conditions(sentinel2_scl, self.SENTINEL2_BUFFERED_SCL_CLASSES)
            # Apply a 3x3 kernel to expand the cloud mask to adjacent pixels.
            cloud_buffer = cloud_related.apply_kernel([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) > 0
            masked_reflectance = sentinel2_reflectance.mask(invalid_surface | cloud_buffer)
            # Calculate the requested output bands and indices, then aggregate to the configured temporal periods.
            sentinel2_temporal = self._aggregate_temporal_cube(self._build_sentinel2_output_cube(masked_reflectance))

        if not self.sentinel1_bands:
            return sentinel2_temporal

        # Sentinel-1 is processed independently of Sentinel-2
        sentinel1 = connection.load_collection(
            self.SENTINEL1_COLLECTION,
            temporal_extent=[self.start_date.strftime("%Y-%m-%d"), self.end_date.strftime("%Y-%m-%d")],
            bands=list(self.sentinel1_bands),
        ).filter_spatial(feature_collection)
        # Convert the raw Sentinel-1 backscatter to decibels using the requested coefficient and a DEM for local incidence angle correction.
        sentinel1 = sentinel1.sar_backscatter(
            coefficient=self.SENTINEL1_BACKSCATTER_COEFFICIENT,
            local_incidence_angle=False,
            elevation_model=self.SENTINEL1_ELEVATION_MODEL,
        )
        sentinel1_temporal = self._aggregate_temporal_cube(sentinel1)
        if sentinel2_temporal is None:
            # Sentinel-1-only runs establish their own projected output grid.
            return sentinel1_temporal.resample_spatial(
                resolution=self.TARGET_RESOLUTION_METRES,
                projection=f"EPSG:{self.working_epsg}",
                method="bilinear",
            )
        # Resample Sentinel-1 to match the spatial resolution of Sentinel-2.
        sentinel1_temporal = sentinel1_temporal.resample_cube_spatial(sentinel2_temporal, method="bilinear")
        return sentinel2_temporal.merge_cubes(sentinel1_temporal)

    def _find_cube_dimension(
        self,
        dataset: xr.Dataset,
        candidates: Sequence[str],
        role: str,
    ) -> str:
        """Find a cube dimension using common openEO/NetCDF dimension names."""

        dimensions = {name.lower(): name for name in dataset.dims}
        for candidate in candidates:
            if candidate.lower() in dimensions:
                return dimensions[candidate.lower()]
        raise ValueError(
            f"Could not identify the {role} dimension in the NetCDF cube. Available dimensions: {list(dataset.dims)}"
        )

    def _select_monthly_variables(
        self,
        dataset: xr.Dataset,
        time_dimension: str,
        x_dimension: str,
        y_dimension: str,
    ) -> xr.DataArray:
        """Return requested bands and indices as one labelled xarray DataArray."""

        expected = list(self._output_sensor_variables())
        variable_lookup = {name.upper(): name for name in dataset.data_vars}

        # CDSE NetCDF normally stores every openEO band as a separate data variable.
        if all(name.upper() in variable_lookup for name in expected):
            arrays = [dataset[variable_lookup[name.upper()]] for name in expected]
            return xr.concat(
                arrays,
                dim=xr.IndexVariable("variable", expected),
                join="exact",
            )

        # Also support NetCDF files that retain a single explicit bands dimension.
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

    def _fill_temporal_neighbors(self, variable_values: np.ndarray) -> np.ndarray:
        """Fill temporal gaps independently at each spatial pixel.

        ``past_only`` carries the closest earlier finite value forward.
        ``bidirectional`` averages the closest finite values on both sides, or
        uses the available side when the series has a leading or trailing gap.
        Existing finite observations are never replaced.
        """
        filled = variable_values.copy()

        # Propagate the closest earlier finite observation through every gap.
        forward = filled.copy()
        for time_index in range(1, forward.shape[0]):
            missing = ~np.isfinite(forward[time_index])
            forward[time_index][missing] = forward[time_index - 1][missing]

        # Retrospective runs may also propagate the closest later observation.
        backward = filled.copy()
        if self.temporal_fill_mode == "bidirectional":
            for time_index in range(backward.shape[0] - 2, -1, -1):
                missing = ~np.isfinite(backward[time_index])
                backward[time_index][missing] = backward[time_index + 1][missing]

        # Modify only gaps from the original cube, preserving observations.
        missing = ~np.isfinite(filled)
        forward_valid = np.isfinite(forward)
        backward_valid = np.isfinite(backward)

        # Average bracketing observations; this is not linear interpolation.
        both = missing & forward_valid & backward_valid
        filled[both] = (forward[both] + backward[both]) / 2.0

        # Leading/trailing gaps use whichever temporal direction is available.
        filled[missing & forward_valid & ~backward_valid] = forward[missing & forward_valid & ~backward_valid]
        filled[missing & ~forward_valid & backward_valid] = backward[missing & ~forward_valid & backward_valid]
        return filled

    def _fill_spatial_neighbors(
        self,
        variable_values: np.ndarray,
        eligible_mask: np.ndarray,
    ) -> np.ndarray:
        """Fill each interval once with the mean of its valid 3x3 neighbors."""
        filled = variable_values.copy()
        height, width = eligible_mask.shape
        for time_index in range(filled.shape[0]):
            layer = filled[time_index]
            valid = np.isfinite(layer) & eligible_mask
            padded_values = np.pad(np.where(valid, layer, 0.0), 1, mode="constant", constant_values=0.0)
            padded_valid = np.pad(valid.astype("int16"), 1, mode="constant", constant_values=0)
            neighbor_sum = np.zeros((height, width), dtype="float64")
            neighbor_count = np.zeros((height, width), dtype="int16")
            for y_offset in range(3):
                for x_offset in range(3):
                    neighbor_sum += padded_values[
                        y_offset : y_offset + height,
                        x_offset : x_offset + width,
                    ]
                    neighbor_count += padded_valid[
                        y_offset : y_offset + height,
                        x_offset : x_offset + width,
                    ]

            neighbor_mean = np.full((height, width), np.nan, dtype="float64")
            np.divide(neighbor_sum, neighbor_count, out=neighbor_mean, where=neighbor_count > 0)
            fillable = ~np.isfinite(layer) & eligible_mask & (neighbor_count > 0)
            layer[fillable] = neighbor_mean[fillable]
            filled[time_index] = layer
        return filled

    def _remove_iqr_outliers(
        self,
        values: np.ndarray,
        eligible_mask: np.ndarray,
        batch_number: int,
        periods: Sequence[str],
        iqr_bounds: dict[tuple[int, int], dict] | None = None,
    ) -> tuple[np.ndarray, dict[tuple[int, int], dict]]:
        """Apply shared interval-variable IQR bounds to eligible pixels."""
        cleaned = values.copy()
        cleaned[:, :, ~eligible_mask] = np.nan
        variables = list(self._output_sensor_variables())
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
                    "iqr_applied": bool(iqr_applied),
                    "iqr_scope": "batch_interval_variable",
                    "iqr_reference_value_count": bounds["valid_count"] if bounds else 0,
                }
        return cleaned, audit_state

    def _calculate_iqr_bounds(self, values: np.ndarray) -> dict[tuple[int, int], dict]:
        """Calculate one IQR range per interval and variable across the batch cube."""
        bounds = {}
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

    def _eligible_pixel_geometry(self, eligible_mask, transform) -> tuple:
        """Return eligible raster indexes and their pixel-center points."""
        rows, columns = np.nonzero(eligible_mask)
        x_coordinates, y_coordinates = xy(transform, rows, columns, offset="center")
        points = [Point(x_coordinate, y_coordinate) for x_coordinate, y_coordinate in zip(x_coordinates, y_coordinates)]
        return rows, columns, points

    def _interpolate_interval_layer(
        self,
        layer: np.ndarray,
        eligible_mask: np.ndarray,
        transform,
        cube_crs,
    ) -> np.ndarray:
        """Apply the configured geospatial interpolator to one raster layer."""
        rows, columns, points = self._eligible_pixel_geometry(eligible_mask, transform)
        if rows.size == 0:
            return layer

        pixel_values = layer[rows, columns]
        missing = ~np.isfinite(pixel_values)
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

    def _fill_variable_cube(
        self,
        variable_values: np.ndarray,
        eligible_mask: np.ndarray,
        transform,
        cube_crs,
    ) -> np.ndarray:
        """Run temporal, 3x3 spatial, then configured interpolation."""
        eligible_count = int(eligible_mask.sum())
        if eligible_count == 0:
            return variable_values.copy()

        observed_counts = np.isfinite(variable_values[:, eligible_mask]).sum(axis=1)
        observed_fractions = observed_counts / eligible_count
        fill_allowed = observed_fractions >= self.minimum_observed_fraction_for_fill

        # Sparse slices retain their observations, but cannot be imputed or used
        # as temporal sources for other periods.
        fill_source = variable_values.copy()
        fill_source[~fill_allowed] = np.nan
        # Use the configured past-only or bidirectional temporal sources.
        filled = self._fill_temporal_neighbors(fill_source)
        # Fill remaining gaps from valid 3x3 neighbors in the same period.
        filled = self._fill_spatial_neighbors(filled, eligible_mask)
        # Apply the configured geospatial interpolation to fill remaining gaps.
        for time_index in range(filled.shape[0]):
            filled[time_index] = self._interpolate_interval_layer(filled[time_index], eligible_mask, transform, cube_crs)
        result = variable_values.copy()
        for time_index, allowed in enumerate(fill_allowed):
            if allowed:
                missing = ~np.isfinite(result[time_index]) & eligible_mask
                result[time_index][missing] = filled[time_index][missing]
        result[:, ~eligible_mask] = np.nan
        return result

    def _build_cleaning_report(
        self,
        cleaned: np.ndarray,
        observed: np.ndarray,
        eligible_mask: np.ndarray,
        periods: Sequence[str],
        audit_state: dict[tuple[int, int], dict],
    ) -> pd.DataFrame:
        """Summarize IQR removal and all pre-statistics pixel filling."""
        rows = []
        eligible_count = int(eligible_mask.sum())
        for time_index, _period_start in enumerate(periods):
            for variable_index, _variable in enumerate(self._output_sensor_variables()):
                # Define counts based on the eligible pixels, not the entire raster layer.
                observed_count = int(np.isfinite(observed[time_index, variable_index][eligible_mask]).sum())
                final_count = int(np.isfinite(cleaned[time_index, variable_index][eligible_mask]).sum())
                pre_fill_nulls = eligible_count - observed_count
                final_nulls = eligible_count - final_count
                state = audit_state[(time_index, variable_index)]
                state.update(
                    {
                        "pre_fill_null_count": pre_fill_nulls,
                        "filled_null_count": pre_fill_nulls - final_nulls,
                        "final_null_count": final_nulls,
                        "final_null_rate": (final_nulls / eligible_count if eligible_count else np.nan),
                        "observed_fraction": (observed_count / eligible_count if eligible_count else np.nan),
                        "fill_allowed": bool(
                            eligible_count
                            and observed_count / eligible_count
                            >= self.minimum_observed_fraction_for_fill
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
    ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Run the complete pixel-cleaning pipeline before parcel statistics."""
        if values.shape[0] != len(periods):
            raise ValueError(f"Expected {values.shape[0]} period labels, received {len(periods)}.")

        # Remove IQR outliers before any filling to avoid contaminating the temporal/spatial sources.
        if iqr_bounds is None:
            iqr_bounds = self._calculate_iqr_bounds(values)
        cleaned, audit_state = self._remove_iqr_outliers(values, eligible_mask, batch_number, periods, iqr_bounds=iqr_bounds)
        observed = cleaned.copy()

        if self.fill_nulls:
            # Fill each variable cube per variable to avoid cross-variable contamination during interpolation.
            for variable_index, _variable in enumerate(self._output_sensor_variables()):
                cleaned[:, variable_index] = self._fill_variable_cube(
                    cleaned[:, variable_index], eligible_mask, transform, cube_crs
                )

        report = self._build_cleaning_report(cleaned, observed, eligible_mask, periods, audit_state)
        return cleaned, observed, report

    def _reduce_local_pixels(self, pixel_values: np.ndarray) -> dict[str, np.ndarray]:
        """Calculate requested statistics across the final pixel axis."""
        valid_count = np.isfinite(pixel_values).sum(axis=-1)
        if pixel_values.shape[-1] == 0:
            empty = np.full(valid_count.shape, np.nan, dtype="float64")
            return {s: valid_count.copy() if s == "count" else empty.copy() for s in self.spatial_statistics}
        totals = np.nansum(pixel_values, axis=-1)
        mean = np.full(valid_count.shape, np.nan, dtype="float64")
        np.divide(totals, valid_count, out=mean, where=valid_count > 0)
        reduced = {}
        with warnings.catch_warnings():
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
        self,
        values: np.ndarray,
        parcel_mask: np.ndarray,
        transform,
    ) -> tuple[np.ndarray, np.ndarray, object]:
        """Crop a cube to one parcel so cleaning cannot cross parcel boundaries."""
        rows, columns = np.nonzero(parcel_mask)
        if rows.size == 0:
            empty_window = Window(col_off=0, row_off=0, width=1, height=1)
            return (
                values[:, :, :1, :1].copy(),
                np.zeros((1, 1), dtype=bool),
                window_transform(empty_window, transform),
            )

        row_start, row_stop = int(rows.min()), int(rows.max()) + 1
        column_start, column_stop = int(columns.min()), int(columns.max()) + 1
        window = Window(
            col_off=column_start,
            row_off=row_start,
            width=column_stop - column_start,
            height=row_stop - row_start,
        )
        return (
            values[:, :, row_start:row_stop, column_start:column_stop].copy(),
            parcel_mask[row_start:row_stop, column_start:column_stop].copy(),
            window_transform(window, transform),
        )

    def _mask_parcel_window(self, values: np.ndarray, geometry, transform):
        """Rasterize one parcel only within its clipped pixel bounding window."""
        raster_height, raster_width = values.shape[-2:]
        raw_window = from_bounds(*geometry.bounds, transform=transform)
        column_start = max(0, math.floor(raw_window.col_off))
        row_start = max(0, math.floor(raw_window.row_off))
        column_stop = min(raster_width, math.ceil(raw_window.col_off + raw_window.width))
        row_stop = min(raster_height, math.ceil(raw_window.row_off + raw_window.height))

        if column_start >= column_stop or row_start >= row_stop:
            empty_window = Window(col_off=0, row_off=0, width=1, height=1)
            return (
                values[:, :, :1, :1].copy(),
                np.zeros((1, 1), dtype=bool),
                window_transform(empty_window, transform),
            )

        window = Window(
            col_off=column_start,
            row_off=row_start,
            width=column_stop - column_start,
            height=row_stop - row_start,
        )
        parcel_transform = window_transform(window, transform)
        parcel_mask = geometry_mask(
            [mapping(geometry)],
            out_shape=(int(window.height), int(window.width)),
            transform=parcel_transform,
            invert=True,
            all_touched=False,
        )
        return (
            values[:, :, row_start:row_stop, column_start:column_stop].copy(),
            parcel_mask,
            parcel_transform,
        )

    def _calculate_local_statistics(self, netcdf_path, parcels, batch_number):
        """Clean pixels independently per parcel and calculate local statistics."""
        self.logger.info(f"Reading monthly pixel cube {netcdf_path}.")
        with xr.open_dataset(netcdf_path, decode_coords="all", mask_and_scale=True) as dataset:
            time_dimension = self._find_cube_dimension(dataset, ("t", "time", "temporal"), "temporal")
            x_dimension = self._find_cube_dimension(dataset, ("x", "longitude", "lon"), "x")
            y_dimension = self._find_cube_dimension(dataset, ("y", "latitude", "lat"), "y")
            monthly_values = self._select_monthly_variables(dataset, time_dimension, x_dimension, y_dimension)
            monthly_values = monthly_values.rio.set_spatial_dims(x_dim=x_dimension, y_dim=y_dimension, inplace=False)
            cube_crs = monthly_values.rio.crs or dataset.rio.crs
            if cube_crs is None:
                raise ValueError(
                    "The native-CRS NetCDF does not contain usable CRS metadata. "
                    "Parcel geometries cannot be aligned safely with the raster."
                )
            # Add cube crs to the xarray DataArray so that geometry_mask can use it for reprojection.
            monthly_values = monthly_values.rio.write_crs(cube_crs, inplace=False)
            # The transform is used to convert between pixel coordinates and spatial coordinates.
            transform = monthly_values.rio.transform(recalc=False)
            # Reorder the xarray to a consistent dimension order for downstream processing.
            ordered = monthly_values.transpose(time_dimension, "variable", y_dimension, x_dimension)
            # Convert the xarray values to a NumPy array for efficient processing.
            # Remote sensing values do not need float64 precision here. Keeping
            # the cube in float32 halves the dominant per-worker allocation.
            values = np.asarray(ordered.values, dtype="float32")
            # Convert the temporal coordinate values to pandas Timestamps for period labeling.
            time_values = pd.to_datetime(ordered[time_dimension].values, utc=True, errors="coerce")
            if pd.isna(time_values).any():
                raise ValueError(
                    f"The NetCDF temporal coordinate {time_dimension!r} contains values that cannot be parsed as dates."
                )
            # Convert the Timestamps to ISO 8601 date strings for period labeling.
            periods = time_values.strftime("%Y-%m-%d").tolist()
            if len(periods) != len(set(periods)):
                raise ValueError("The temporal NetCDF contains duplicate period-start labels.")
            # Validate that the NetCDF periods match the expected temporal intervals.
            intervals, labels = self._temporal_intervals()
            # Create a lookup dictionary for period end dates based on the expected intervals.
            period_end_lookup = {label: interval[1] for label, interval in zip(labels, intervals)}
            # Check for any unexpected period labels in the NetCDF.
            unknown = sorted(set(periods).difference(period_end_lookup))
            if unknown:
                raise ValueError(f"The temporal NetCDF contains unexpected period labels: {unknown}")
            # Reproject the parcels to the cube CRS for accurate spatial alignment.
            projected = parcels.to_crs(cube_crs)
            # Learn robust thresholds from all valid batch pixels for each
            # interval-variable layer, rather than from each small parcel.
            iqr_bounds = self._calculate_iqr_bounds(values)
            rows = []
            parcel_reports = []
            variables = list(self._output_sensor_variables())
            for ((_, parcel), (_, projected_parcel)) in zip(parcels.iterrows(), projected.iterrows()):
                parcel_id = str(parcel[self.PARCEL_ID_FIELD])
                parcel_values, cropped_mask, parcel_transform = self._mask_parcel_window(values, projected_parcel.geometry, transform)
                eligible_pixel_count = int(cropped_mask.sum())
                cleaned, observed, parcel_report = self._clean_monthly_pixels(
                    parcel_values,
                    cropped_mask,
                    batch_number,
                    periods,
                    parcel_transform,
                    cube_crs,
                    iqr_bounds=iqr_bounds,
                )
                parcel_report.insert(1, self.PARCEL_ID_FIELD, parcel_id)
                parcel_report.insert(5, "eligible_pixel_count", eligible_pixel_count)
                parcel_reports.append(parcel_report)

                stats = self._reduce_local_pixels(cleaned[:, :, cropped_mask])
                if "count" in stats:
                    stats["count"] = np.isfinite(observed[:, :, cropped_mask]).sum(axis=-1)
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
        report = pd.concat(parcel_reports, ignore_index=True) if parcel_reports else pd.DataFrame()
        self.logger.info(
            f"Calculated {list(self.spatial_statistics)} locally for {len(parcels)} parcels from {netcdf_path.name}."
        )
        return result, report

    def _add_derived_stats(self, data: pd.DataFrame) -> pd.DataFrame:
        """Append deterministic parcel features in one non-fragmenting operation."""
        metric_parcels = self.parcels.to_crs(epsg=self.working_epsg)
        parcel_metrics = pd.DataFrame(
            {
                self.PARCEL_ID_FIELD: self.parcels[self.PARCEL_ID_FIELD].to_numpy(),
                "parcel_area_m2": metric_parcels.geometry.area.to_numpy(),
            }
        )
        parcel_metrics["approx_pixel_count"] = (
            parcel_metrics["parcel_area_m2"] / self.TARGET_RESOLUTION_METRES**2
        )
        enriched = data.merge(parcel_metrics, on=self.PARCEL_ID_FIELD, how="left", validate="many_to_one")

        # Build every derived series first. Appending them together avoids the
        # fragmented block layout caused by repeated DataFrame column insertion.
        derived_columns: dict[str, pd.Series] = {}
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
                    exact_denominator = enriched["eligible_pixel_count"].where(
                        enriched["eligible_pixel_count"] > 0
                    )
                    exact_fraction = enriched[count] / exact_denominator
                    derived_columns[f"{variable}_valid_pixel_fraction"] = exact_fraction.clip(0.0, 1.0)

        if not derived_columns:
            return enriched
        derived = pd.DataFrame(derived_columns, index=enriched.index)
        return pd.concat([enriched, derived], axis=1)

    def _run_openeo_batch_job(self, job, temporary_path, netcdf_path, batch_number):
        """Run and download one already-created openEO batch job."""
        job.start_and_wait(print=lambda _message: None)
        job.get_results().download_file(target=temporary_path)
        temporary_path.replace(netcdf_path)
        return batch_number, netcdf_path

    def _process_zonal_stats_batch(self, netcdf_path, batch, batch_number):
        """Calculate one cached batch in a worker process."""
        result, report = self._calculate_local_statistics(netcdf_path, batch, batch_number)
        result["batch_number"] = batch_number
        return batch_number, result, report

    def _run_openeo_job_wave(self, remote_jobs):
        """Run and download one bounded wave of already-created openEO jobs."""

        if not remote_jobs:
            return
        batch_numbers = [number for number, *_rest in remote_jobs]
        self.logger.info(f"Starting openEO job wave for batches {batch_numbers}.")
        workers = min(self.batch_workers, len(remote_jobs))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self._run_openeo_batch_job, job, temporary, path, number): number
                for number, job, temporary, path in remote_jobs
            }
            for future in as_completed(futures):
                expected = futures[future]
                number, path = future.result()
                if number != expected:
                    raise RuntimeError(f"openEO job returned batch {number}; expected {expected}.")
                self.logger.info(f"Cached completed batch {number} temporal cube at {path}.")

    def _run_openeo_jobs(self, cube_dir):
        """Create, run and download uncached batches in bounded job waves."""
        connection = None
        pending_batches, remote_jobs = [], []
        for batch_number, batch in enumerate(self._iter_spatial_batches(), start=1):
            netcdf_path = cube_dir / f"batch_{batch_number:05d}_monthly.nc"
            if netcdf_path.exists():
                self.logger.info(f"Using existing batch {batch_number} monthly cube {netcdf_path}.")
            else:
                if connection is None:
                    self.logger.info(f"Connecting to openEO backend {self.OPENEO_URL}.")
                    connection = openeo.connect(self.OPENEO_URL, auto_validate=False)
                    self._authenticate_openeo_connection(connection)

                print(f"Creating openEO job for batch {batch_number} with {len(batch)} parcels.")
                feature_collection = self._to_feature_collection(batch)
                cube = self._build_multisensor_temporal_cube(connection, feature_collection)
                temporary_path = netcdf_path.with_suffix(".partial.nc")
                if temporary_path.exists():
                    temporary_path.unlink()
                job = cube.create_job(
                    out_format="netCDF",
                    title=f"Sentinel-1/Sentinel-2 temporal cube batch {batch_number}",
                )
                remote_jobs.append((batch_number, job, temporary_path, netcdf_path))
                # Do not create every remote job up front. Keeping at most one
                # concurrency-sized wave limits preflight requests and respects
                # the backend's active batch-job quota.
                if len(remote_jobs) >= self.batch_workers:
                    self._run_openeo_job_wave(remote_jobs)
                    remote_jobs.clear()
            pending_batches.append((batch_number, batch, netcdf_path))
        if not pending_batches:
            raise RuntimeError("No processing batches were produced.")
        self._run_openeo_job_wave(remote_jobs)
        return pending_batches

    def _run_batch_statistics(self, batches):
        """Calculate local statistics for downloaded batches in parallel."""
        results, reports = {}, {}
        if self.batch_workers == 1 or len(batches) == 1:
            for number, batch, path in batches:
                returned, result, report = self._process_zonal_stats_batch(path, batch, number)
                results[returned], reports[returned] = result, report
        else:
            workers = min(self.batch_workers, len(batches))
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(self._process_zonal_stats_batch, path, batch, number): number
                    for number, batch, path in batches
                }
                for future in as_completed(futures):
                    expected = futures[future]
                    number, result, report = future.result()
                    if number != expected:
                        raise RuntimeError(f"Worker returned batch {number}; expected {expected}.")
                    results[number], reports[number] = result, report
        numbers = sorted(results)
        return [results[n] for n in numbers], [reports[n] for n in numbers]

    def run(self) -> gpd.GeoDataFrame:
        """Run extraction, local cleaning, zonal statistics and persistence."""
        cube_dir = self.output_dir / "monthly_cubes" / self._run_signature()
        cube_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Monthly cube cache: {cube_dir}.")
        # Run openEO jobs for each uncached parcel batch, downloading the monthly NetCDF cubes.
        batches = self._run_openeo_jobs(cube_dir)
        # Run local statistics for each batch in parallel, returning a list of DataFrames and cleaning reports.
        completed_results, cleaning_reports = self._run_batch_statistics(batches)
        # Concatenate the batch results into a single DataFrame, sort by parcel and period, and add derived statistics.
        final = pd.concat(completed_results, ignore_index=True)
        final = final.sort_values([self.PARCEL_ID_FIELD, "period_start"]).reset_index(drop=True)
        final = self._add_derived_stats(final)
        # Merge the final statistics DataFrame with the original parcel geometries for spatial context.
        final = final.merge(
            self.parcels[[self.PARCEL_ID_FIELD, "geometry"]],
            on=self.PARCEL_ID_FIELD,
            how="left",
            validate="many_to_one",
        )
        # Convert the final DataFrame to a GeoDataFrame with the appropriate geometry and CRS.
        final = gpd.GeoDataFrame(final, geometry="geometry", crs=self.parcels.crs)
        # Persist the final statistics and cleaning report to CSV and Parquet files in the output directory.
        csv_output = self.output_dir / "satellite_temporal_parcel_statistics.csv"
        parquet_output = self.output_dir / "satellite_temporal_parcel_statistics.parquet"
        final.to_csv(csv_output, index=False)
        final.to_parquet(parquet_output, index=False)
        self.cleaning_report = pd.concat(cleaning_reports, ignore_index=True)
        cleaning_output = self.output_dir / "satellite_pixel_cleaning_report.csv"
        self.cleaning_report.to_csv(cleaning_output, index=False)
        self.logger.info(f"Created {csv_output}, {parquet_output}, and {cleaning_output} " f"({len(final)} rows).")
        return final
