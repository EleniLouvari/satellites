"""Shared local parcel processing, configuration and worker state.

The historical sensor option names and cache fields remain stable. Acquisition
methods live in source adapters; this module does not import openEO or Planet.
"""

from __future__ import annotations

import gc
import hashlib
import logging
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import pandas as pd

from data_preparation.parcel_stats.core.batch_cache import BatchResultCache
from data_preparation.parcel_stats.core.configuration import ZonalStatsConfiguration
from data_preparation.parcel_stats.core.parcel_batches import ParcelBatchPlanner
from data_preparation.parcel_stats.core.parcel_statistics import ParcelStatisticsCalculator
from data_preparation.parcel_stats.core.raster_cleaning import RasterCleaner
from data_preparation.parcel_stats.core.streamed_raster import LOCAL_RASTER_LOCK


class ParcelStatsBase(
    BatchResultCache, ZonalStatsConfiguration, ParcelBatchPlanner, RasterCleaner, ParcelStatisticsCalculator
):
    """Calculate temporal Sentinel-1 and Sentinel-2 parcel features.

    Parameters
    ----------
    parcels:
        Parcel polygons. A unique ``parcel_id`` column is retained when present;
        otherwise the GeoDataFrame index is converted into parcel IDs.
    start_date, end_date:
        Inclusive ISO-style dates accepted by :func:`pandas.Timestamp`.
    output_dir:
        Directory for cached monthly NetCDF cubes and the final ML-ready GeoParquet.
    working_epsg:
        Projected CRS used for spatial batching, parcel metrics and final
        interpolation. The openEO graph requests its raster grid in this CRS;
        local processing verifies the actual CRS in the downloaded cube.
    parcel_id_field:
        Name of the column containing unique parcel identifiers. If the column
        is absent, stable string identifiers are created from the input index.
    sentinel2_bands:
        Sentinel-2 L2A reflectance bands to include in the output. When omitted,
        :attr:`DEFAULT_SENTINEL2_BANDS` is used. Extra bands required by indices
        are loaded automatically without being added as standalone outputs.
        Pass an empty list to disable standalone Sentinel-2 bands.
    sentinel2_indices:
        Optional list of index names. Supported values are NDVI, NDWI, MNDWI,
        NDMI, NBR, GNDVI, EVI, SAVI, MSAVI, NDRE, PSRI and CI. ``None`` or an
        empty list disables Sentinel-2 indices; a non-empty list enables them.
    sentinel1_bands:
        Optional Sentinel-1 GRD polarizations to add to the monthly cube.
        Supported values are ``VV`` and ``VH``. Backscatter is calculated as
        linear-power sigma0 over the ellipsoid; no dB conversion is applied.
        At least one Sentinel-1 or Sentinel-2 band/index must be selected.
    sentinel1_indices:
        Optional Sentinel-1 indices. ``R`` is ``VV / VH`` and ``RVI`` is
        ``4 * VH / (VV + VH)``. Both automatically load VV and VH. ``None``
        or an empty list disables Sentinel-1 indices.
    sentinel1_orbit_direction:
        Sentinel-1 orbit direction to load. Use ``'ASCENDING'``,
        ``'DESCENDING'``, or ``'BOTH'``. The default ``'BOTH'`` applies no
        orbit filter and preserves the existing behavior.
    spatial_statistics:
        Optional parcel statistics calculated locally from the monthly cube.
        ``mean`` is always included; supported additions are median, sd, min,
        max and selected percentiles. ``count`` is accepted for compatibility,
        but now produces the single parcel-level ``intersected_pixel_count``
        instead of dated per-layer count features.
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
        3x3 and 5x5 spatial means, and finally the configured spatial interpolator
        across each variable/time raster before applying parcel masks.
    iqr_min_valid_pixels:
        Minimum valid raster pixels required before applying the IQR filter.
    temporal_fill_mode:
        Temporal source direction used when filling a missing pixel. Use
        ``'past_only'`` to carry the latest earlier value forward. Use
        ``'bidirectional'`` to average the closest earlier and later values;
        if only one direction is available, that value is used.
    minimum_parcel_pixels:
        Minimum eligible pixels for the emitted parcel-quality flag. Smaller
        parcels are retained rather than discarded.
    minimum_observed_fraction_for_fill:
        Minimum observed share required before a raster-variable-period may be
        imputed or used as a temporal filling source.
    spatial_fill_window_sizes:
        Ordered spatial-mean windows applied after temporal filling. Defaults to
        ``(3, 5)``; append 7 and 9 only when wider smoothing is acceptable.
    temporal_fill_window_sizes:
        Increasing odd temporal window widths, including the target step.
        ``(3,)`` uses only the immediately previous/next step at the same pixel,
        subject to ``temporal_fill_mode``. Wider windows retry remaining gaps
        using original observations. ``None`` preserves unlimited search.
    keep_cleaned_checkpoint:
        Retain the intermediate physical-band checkpoint after the final
        bands-and-indices checkpoint commits. Defaults to ``False``.
    remove_nc_after_completion:
        Remove a batch's raw, cleaned, final and partial NetCDF files only after
        its statistics and cleaning report commit to Parquet. Defaults to
        ``False``. Overrides ``keep_cleaned_checkpoint`` after batch completion.
    resume_completed_partitions:
        Reuse complete openEO partition outputs. Defaults to ``False``. For
        legacy outputs without a completion manifest, enable only when resuming
        the same parcels and processing configuration.
    temporal_period:
        Temporal bin width. Use ``'15D'`` for 15 days or ``'1M'``, ``'2M'``
        and ``'3M'`` for one-, two- and three-month bins.
    temporal_reducer:
        Reducer applied to acquisitions in each temporal bin. Supported values
        are ``'mean'``, ``'median'``, ``'min'``, ``'max'`` and ``'sum'``.
        For openEO, ``'none'`` (or ``None``) retains all acquisitions and pools
        their valid parcel pixels within each bin for statistics. Indices are
        calculated per acquisition before pooling. Cleaning options still apply;
        use ``fill_nulls=False`` to exclude imputed observations.
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

    # =========================================================================
    # Source band IDs stay unchanged in outputs; shared algebra uses these roles.
    OPTICAL_BAND_ROLES = {
        "blue": "B02",
        "green": "B03",
        "red": "B04",
        "red_edge": "B05",
        "red_edge_2": "B06",
        "nir": "B08",
        "swir_1": "B11",
        "swir_2": "B12",
    }

    # Sentinel-2 configuration constants
    # =========================================================================
    SENTINEL2_COLLECTION = "SENTINEL2_L2A"
    SUPPORTED_SENTINEL2_BANDS = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12")
    DEFAULT_SENTINEL2_BANDS = ("B02", "B03", "B04", "B05", "B08")
    SENTINEL2_SCENE_CLASSIFICATION_BAND = "SCL"
    SENTINEL2_REFLECTANCE_SCALE_FACTOR = 0.0001
    SENTINEL2_MAX_SCENE_CLOUD_COVER = 70
    SENTINEL2_INVALID_SCL_CLASSES = (0, 1, 3, 8, 9, 10, 11)
    SENTINEL2_BUFFERED_SCL_CLASSES = (3, 8, 9, 10, 11)
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

    # =========================================================================
    # Sentinel-1 configuration constants
    # =========================================================================
    SENTINEL1_COLLECTION = "SENTINEL1_GRD"
    SUPPORTED_SENTINEL1_BANDS = ("VV", "VH")
    SUPPORTED_SENTINEL1_INDICES = {"R": ("VV", "VH"), "RVI": ("VV", "VH")}
    SUPPORTED_SENTINEL1_ORBIT_DIRECTIONS = ("ASCENDING", "DESCENDING", "BOTH")
    SENTINEL1_BACKSCATTER_COEFFICIENT = "sigma0-ellipsoid"
    SENTINEL1_BACKSCATTER_SCALE = "linear_power"
    SENTINEL1_ELEVATION_MODEL = "COPERNICUS_30"

    # =========================================================================
    # General configuration constants
    # =========================================================================
    TARGET_RESOLUTION_METRES = 10
    MAX_FEATURES_PER_JOB = 5_000
    GRID_SIZE_METRES = 50_000
    PARCEL_ID_FIELD = "parcel_id"
    OPENEO_URL = "https://openeo.dataspace.copernicus.eu"
    OPENEO_OIDC_PASSWORD_CLIENT_ID = "cdse-public"
    # Logger namespaces describe the public pipeline, independent of module layout.
    LOGGER_NAMESPACE = "parcel_stats_pipeline"
    SOURCE_NAME = "openEO"
    SOURCE_LOG_FILE_NAME = "satellite_source.log"
    LOG_FILE_NAME = "satellite_zonal_stats.log"
    FILLING_LOG_FILE_NAME = "satellite_raster_filling.log"
    PARCEL_LOG_FILE_NAME = "satellite_parcel_statistics.log"
    PARCEL_OUTPUT_FILE_NAME = "satellite_parcel_time_stats.geoparquet"
    REDUCED_PARCEL_OUTPUT_FILE_NAME = "satellite_parcel_annual_stats.geoparquet"

    # =========================================================================
    # Spatial statistics configuration constants
    # =========================================================================
    SUPPORTED_SPATIAL_STATISTICS = ("mean", "median", "sd", "min", "max", "range", "count", "p10", "p25", "p75", "p90")
    QUANTILE_PROBABILITIES = {"p10": 0.10, "p25": 0.25, "p75": 0.75, "p90": 0.90}
    SPATIAL_STATISTIC_ALIASES = {"average": "mean", "std": "sd", "stdev": "sd", "standard_deviation": "sd"}

    # =========================================================================
    # Initialization, validation, and runtime configuration
    # =========================================================================

    def __init__(
        self,
        parcels: gpd.GeoDataFrame,
        start_date: str | pd.Timestamp,
        end_date: str | pd.Timestamp,
        output_dir: str | Path,
        working_epsg: int,
        parcel_id_field: str = "parcel_id",
        sentinel2_bands: list[str] | None = None,
        sentinel2_indices: list[str] | None = [],
        sentinel1_bands: list[str] | None = None,
        sentinel1_indices: list[str] | None = [],
        sentinel1_orbit_direction: str = "BOTH",
        spatial_statistics: list[str] | None = None,
        remove_outliers: bool = False,
        iqr_quantiles: tuple[float, float] = (0.25, 0.75),
        iqr_multiplier: float = 1.5,
        fill_nulls: bool = False,
        temporal_period: str = "1M",  # Valid values: '1M', '3M', '6M', '1Y', '15D', etc.
        temporal_reducer: str | None = "median",
        interpolation_method: str = "nearest",
        interpolation_max_distance_in_meters: float | None = None,
        interpolation_variogram_lags: int = 15,
        interpolation_variogram_max_distance_in_meters: float | None = None,
        batch_workers: int = 1,
        iqr_min_valid_pixels: int = 20,
        temporal_fill_mode: str = "past_only",
        minimum_parcel_pixels: int = 3,
        minimum_observed_fraction_for_fill: float = 0.20,
        spatial_fill_window_sizes: tuple[int, ...] | list[int] = (3, 5),
        keep_cleaned_checkpoint: bool = False,
        openeo_username: str | None = None,
        openeo_password: str | None = None,
        temporal_fill_window_sizes: tuple[int, ...] | list[int] | None = None,
        remove_nc_after_completion: bool = False,
        resume_completed_partitions: bool = False,
    ) -> None:
        """Validate inputs and prepare a reusable extraction instance."""

        self.start_date, self.end_date = self._validate_dates(start_date, end_date)
        self.temporal_period, self.temporal_reducer = self._validate_temporal_aggregation(temporal_period, temporal_reducer)
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
        self.openeo_username, self.openeo_password = self._validate_openeo_credentials(openeo_username, openeo_password)
        if not isinstance(parcel_id_field, str):
            raise TypeError("Error: parcel_id_field must be a string.")
        parcel_id_field = parcel_id_field.strip()
        if not parcel_id_field or parcel_id_field == "geometry":
            raise ValueError("Error: parcel_id_field must be a non-empty, non-geometry column name.")
        self.PARCEL_ID_FIELD = parcel_id_field
        self.batch_workers = self._validate_batch_workers(batch_workers)
        self.spatial_fill_window_sizes = self._validate_spatial_fill_windows(spatial_fill_window_sizes)
        self.temporal_fill_window_sizes = self._validate_temporal_fill_windows(temporal_fill_window_sizes)
        if not isinstance(keep_cleaned_checkpoint, bool):
            raise TypeError("Error: keep_cleaned_checkpoint must be a bool.")
        self.keep_cleaned_checkpoint = keep_cleaned_checkpoint
        for name, value in (
            ("remove_nc_after_completion", remove_nc_after_completion),
            ("resume_completed_partitions", resume_completed_partitions),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"Error: {name} must be a bool.")
            setattr(self, name, value)
        self.sentinel2_bands = self._validate_sentinel2_bands(sentinel2_bands)
        self.sentinel2_indices = self._validate_sentinel2_indices(sentinel2_indices)
        self.sentinel1_bands = self._validate_sentinel1_bands(sentinel1_bands)
        self.sentinel1_indices = self._validate_sentinel1_indices(sentinel1_indices)
        self.sentinel1_orbit_direction = self._validate_sentinel1_orbit_direction(sentinel1_orbit_direction)
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
        self.logger = self._create_logger("orchestrator", self.LOG_FILE_NAME)
        self.source_logger = self._create_logger("source", self.SOURCE_LOG_FILE_NAME)
        self.filling_logger = self._create_logger("filling", self.FILLING_LOG_FILE_NAME)
        self.parcel_logger = self._create_logger("parcels", self.PARCEL_LOG_FILE_NAME)
        self.parcels = self._prepare_parcels(parcels)
        self.logger.info(
            "Log files: orchestration=%s, source=%s, raster filling=%s, parcel statistics=%s.",
            self.output_dir / self.LOG_FILE_NAME,
            self.output_dir / self.SOURCE_LOG_FILE_NAME,
            self.output_dir / self.FILLING_LOG_FILE_NAME,
            self.output_dir / self.PARCEL_LOG_FILE_NAME,
        )
        self.logger.info(
            f"Source={self.SOURCE_NAME}, optical bands={list(self.optical_bands)}, "
            f"optical indices={list(self.optical_indices)}, "
            f"parcel statistics={list(self.spatial_statistics)}, "
            f"remove_outliers={self.remove_outliers}, "
            f"iqr_quantiles={self.iqr_quantiles}, "
            f"iqr_multiplier={self.iqr_multiplier}, and "
            f"fill_nulls={self.fill_nulls}, "
            f"temporal_period={self.temporal_period}, and "
            f"temporal_reducer={self.temporal_reducer}, and "
            f"batch_workers={self.batch_workers}."
        )

        if self.sentinel1_bands or self.sentinel1_indices:
            self.logger.info(
                "Radar source=Sentinel-1, bands=%s, indices=%s, orbit direction=%s.",
                list(self.sentinel1_bands),
                list(self.sentinel1_indices),
                self.sentinel1_orbit_direction,
            )

    def __getstate__(self) -> dict:
        """Exclude non-picklable logging handlers from worker payloads."""

        state = self.__dict__.copy()
        for attribute in ("logger", "source_logger", "filling_logger", "parcel_logger"):
            state.pop(attribute, None)
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore scope-specific console and file logging inside a worker."""

        self.__dict__.update(state)
        self.logger = self._create_worker_console_logger("orchestrator")
        self.source_logger = self._create_worker_console_logger("source")
        self.filling_logger = self._create_worker_logger("filling", self.FILLING_LOG_FILE_NAME)
        self.parcel_logger = self._create_worker_logger("parcels", self.PARCEL_LOG_FILE_NAME)

    def _configure_logger_handlers(self, logger: logging.Logger, log_path: Path) -> logging.Logger:
        """Attach consistently formatted console and UTF-8 file handlers."""

        if not logger.handlers:
            formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
            for handler in (logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")):
                handler.setFormatter(formatter)
                logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        return logger

    def _create_logger(self, scope: str, file_name: str) -> logging.Logger:
        """Create one main-process logger for a pipeline scope."""

        # Object IDs can be reused after an extractor is collected while logging
        # retains its handlers. Include the output path to avoid cross-run writes.
        output_key = hashlib.sha1(str(self.output_dir.resolve()).encode("utf-8")).hexdigest()[:10]
        logger = logging.getLogger(f"{self.LOGGER_NAMESPACE}.{output_key}.{id(self)}.{scope}")
        return self._configure_logger_handlers(logger, self.output_dir / file_name)

    def _create_worker_logger(self, scope: str, file_name: str) -> logging.Logger:
        """Create a per-process scope logger safe for parallel batch execution."""

        output_key = hashlib.sha1(str(self.output_dir.resolve()).encode("utf-8")).hexdigest()[:10]
        process_id = os.getpid()
        logger = logging.getLogger(f"{self.LOGGER_NAMESPACE}.worker.{process_id}.{output_key}.{scope}")
        base_path = Path(file_name)
        worker_name = f"{base_path.stem}.worker-{process_id}{base_path.suffix}"
        return self._configure_logger_handlers(logger, self.output_dir / worker_name)

    def _create_worker_console_logger(self, scope: str) -> logging.Logger:
        """Create a console-only worker logger for scopes unused by local jobs."""

        output_key = hashlib.sha1(str(self.output_dir.resolve()).encode("utf-8")).hexdigest()[:10]
        logger = logging.getLogger(f"{self.LOGGER_NAMESPACE}.worker.{os.getpid()}.{output_key}.{scope}")
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        return logger

    # =========================================================================
    # Local batch execution and result orchestration
    # =========================================================================

    def _process_zonal_stats_batch(self, netcdf_path, batch, batch_number):
        """Return a batch number, statistics and report, reusing saved tables.

        Persist newly calculated tables before optional NetCDF deletion. The
        caller owns the returned DataFrames; use the checkpoint wrapper when
        they must be released before another raster starts.
        """
        if self._has_batch_statistics(netcdf_path):
            result, report = self._cached_batch_statistics(netcdf_path)
        else:
            result, report = self._calculate_local_statistics(netcdf_path, batch, batch_number)
            result["batch_number"] = batch_number
            self._save_batch_statistics(netcdf_path, result, report)
        self._remove_completed_batch_netcdfs(netcdf_path)
        return batch_number, result, report

    def _checkpoint_zonal_stats_batch(self, netcdf_path, batch, batch_number):
        """Save one batch and release its tables before another raster starts.

        Hold the shared raster lock through calculation, persistence, optional
        file deletion and garbage collection. Return only the batch number.
        On failure, clear completed traceback frames to release retained tables
        and propagate the exception while preserving recoverable raster files.
        """
        # Keep the lock through persistence and cleanup, not just raster reads.
        # The inner raster lock is reentrant. Return only the batch number so
        # callers and Futures cannot retain the statistics between raster jobs.
        with LOCAL_RASTER_LOCK:
            completed = None
            try:
                completed = self._process_zonal_stats_batch(netcdf_path, batch, batch_number)
                if completed[0] != batch_number:
                    raise RuntimeError(f"Error: Worker returned batch {completed[0]}; expected {batch_number}.")
                return batch_number
            except Exception as error:
                # Failed Parquet writes can otherwise pin the full result in
                # notebook/Future tracebacks, even after local variables clear.
                traceback.clear_frames(error.__traceback__)
                raise
            finally:
                completed = None
                gc.collect()

    def _run_batch_statistics(self, batches):
        """Return ordered lists of batch statistics and cleaning reports.

        Accept ``(number, parcels, path)`` tuples. With cleanup enabled, process
        and release each batch sequentially before loading saved tables for
        final assembly. Otherwise use the configured local worker count.
        """
        if self.remove_nc_after_completion:
            # No batch tables accumulate while any NetCDF still needs work.
            # The shared scheduler normally completed these checkpoints already;
            # direct/partition callers use the same bounded-memory lifecycle.
            for number, batch, path in batches:
                self._checkpoint_zonal_stats_batch(path, batch, number)
            completed = [self._cached_batch_statistics(path) for _, _, path in sorted(batches, key=lambda b: b[0])]
            return [result for result, _ in completed], [report for _, report in completed]
        results, reports = {}, {}
        if self.batch_workers == 1 or len(batches) == 1:
            for number, batch, path in batches:
                returned, result, report = self._process_zonal_stats_batch(path, batch, number)
                results[returned], reports[returned] = result, report
        else:
            self.filling_logger.info(
                "Parallel raster filling uses %s.worker-<pid>%s files in %s.",
                Path(self.FILLING_LOG_FILE_NAME).stem,
                Path(self.FILLING_LOG_FILE_NAME).suffix,
                self.output_dir,
            )
            self.parcel_logger.info(
                "Parallel parcel statistics use %s.worker-<pid>%s files in %s.",
                Path(self.PARCEL_LOG_FILE_NAME).stem,
                Path(self.PARCEL_LOG_FILE_NAME).suffix,
                self.output_dir,
            )
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
                        raise RuntimeError(f"Error: Worker returned batch {number}; expected {expected}.")
                    results[number], reports[number] = result, report
        numbers = sorted(results)
        return [results[n] for n in numbers], [reports[n] for n in numbers]
