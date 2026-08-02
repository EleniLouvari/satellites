"""Sentinel-2 temporal zonal statistics for parcel geometries."""

from __future__ import annotations

import hashlib
import json
import logging
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
from shapely import make_valid
from shapely.geometry import GeometryCollection, MultiPolygon, Point, Polygon, mapping
from shapely.ops import unary_union


class SatelliteZonalStats:
    """Calculate monthly Sentinel-2 parcel features for machine learning.

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
    calculate_indices:
        If ``True``, calculate the requested spectral indices on openEO.
    indices:
        List of index names. Supported values are NDVI, NDWI, NDMI, NBR,
        GNDVI, EVI and SAVI. Required when ``calculate_indices`` is ``True``.
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
        If ``True``, fill post-IQR pixel nulls using a 3x3 spatial mean,
        temporal neighbors, and finally the configured spatial interpolator
        independently within each temporal period.
    temporal_period:
        Temporal bin width. Use ``'15D'`` for 15 days or ``'1M'``, ``'2M'``
        and ``'3M'`` for one-, two- and three-month bins.
    temporal_reducer:
        Reducer applied to acquisitions in each temporal bin. Supported values
        are ``'mean'``, ``'median'``, ``'min'``, ``'max'`` and ``'sum'``.
    interpolation_method:
        Final pixel-level interpolation applied per variable and temporal period.
        Choose ``'nearest'``, ``'idw'`` or ``'kriging'``.
    interpolation_max_distance_in_feet:
        Required maximum search distance when using IDW.
    interpolation_variogram_lags, interpolation_variogram_max_distance:
        Kriging variogram configuration; maximum distance is expressed in feet.
    batch_workers:
        Maximum number of openEO jobs run concurrently and worker processes used
        for local batch statistics. A value of 1 keeps sequential behaviour.
    """

    BANDS = ("B02", "B03", "B04", "B05", "B08")
    TARGET_RESOLUTION = 10
    MAX_FEATURES_PER_JOB = 10_000
    GRID_SIZE_METRES = 70_000
    MAX_SCENE_CLOUD_COVER = 70
    # Reject no-data, defective pixels, cloud shadow, clouds/cirrus and snow.
    # Water (class 6) remains valid so NDWI can retain its intended signal.
    INVALID_SCL_CLASSES = (0, 1, 3, 8, 9, 10, 11)
    BUFFERED_SCL_CLASSES = (3, 8, 9, 10, 11)
    PARCEL_ID_FIELD = "parcel_id"
    OPENEO_URL = "https://openeo.dataspace.copernicus.eu"
    LOG_FILE_NAME = "satellite_zonal_stats.log"

    # Band definitions use Sentinel-2 collection band names. NDWI follows the
    # McFeeters green/NIR convention; NDMI represents the NIR/SWIR moisture index.
    SUPPORTED_INDICES = {
        "NDVI": ("B08", "B04"),
        "NDWI": ("B03", "B08"),
        "NDMI": ("B08", "B11"),
        "NBR": ("B08", "B12"),
        "GNDVI": ("B08", "B03"),
        "EVI": ("B08", "B04", "B02"),
        "SAVI": ("B08", "B04"),
        "NDRE": ("B08", "B05"),
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
        calculate_indices: bool = False,
        indices: list[str] | None = None,
        spatial_statistics: list[str] | None = None,
        remove_outliers: bool = False,
        iqr_quantiles: tuple[float, float] = (0.25, 0.75),
        iqr_multiplier: float = 1.5,
        fill_nulls: bool = False,
        temporal_period: str = "1M",  # Valid values: '1M', '3M', '6M', '1Y', '15D', etc.
        temporal_reducer: str = "median",
        interpolation_method: str = "nearest",
        interpolation_max_distance_in_feet: float | None = None,
        interpolation_variogram_lags: int = 15,
        interpolation_variogram_max_distance: float | None = None,
        batch_workers: int = 1,
    ) -> None:
        """Validate inputs and prepare a reusable extraction instance."""

        self.start_date, self.end_date = self._validate_dates(start_date, end_date)
        self.temporal_period, self.temporal_reducer = self._validate_temporal_aggregation(
            temporal_period,
            temporal_reducer,
        )
        (
            self.interpolation_method,
            self.interpolation_max_distance_in_feet,
            self.interpolation_variogram_lags,
            self.interpolation_variogram_max_distance,
        ) = self._validate_interpolation_options(
            interpolation_method,
            interpolation_max_distance_in_feet,
            interpolation_variogram_lags,
            interpolation_variogram_max_distance,
        )
        self.output_dir = Path(output_dir)
        self.working_epsg = self._validate_working_epsg(working_epsg)
        self.batch_workers = self._validate_batch_workers(batch_workers)
        if not isinstance(calculate_indices, bool):
            raise TypeError("calculate_indices must be a bool.")
        self.calculate_indices = calculate_indices
        self.indices = self._validate_indices(indices)
        self.spatial_statistics = self._validate_spatial_statistics(spatial_statistics)
        (
            self.remove_outliers,
            self.iqr_quantiles,
            self.iqr_multiplier,
            self.fill_nulls,
        ) = self._validate_cleaning_options(
            remove_outliers=remove_outliers,
            iqr_quantiles=iqr_quantiles,
            iqr_multiplier=iqr_multiplier,
            fill_nulls=fill_nulls,
        )
        # Start file logging only after non-spatial configuration is valid.
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = self._create_logger()
        self.parcels = self._prepare_parcels(parcels)
        self.logger.info(
            f"Configured indices={list(self.indices)}, "
            f"parcel statistics={list(self.spatial_statistics)}, "
            f"remove_outliers={self.remove_outliers}, "
            f"iqr_quantiles={self.iqr_quantiles}, "
            f"iqr_multiplier={self.iqr_multiplier}, and "
            f"fill_nulls={self.fill_nulls}, "
            f"temporal_period={self.temporal_period}, and "
            f"temporal_reducer={self.temporal_reducer}, and "
            f"batch_workers={self.batch_workers}."
        )

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
        max_distance_in_feet: float | None,
        variogram_lags: int,
        variogram_max_distance: float | None,
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
            if value is None:
                return None
            numeric = float(value)
            if not np.isfinite(numeric) or numeric <= 0:
                raise ValueError(f"{name} must be a positive finite number.")
            return numeric

        maximum_distance = optional_positive(max_distance_in_feet, "interpolation_max_distance_in_feet")
        variogram_distance = optional_positive(variogram_max_distance, "interpolation_variogram_max_distance")
        try:
            lags = int(variogram_lags)
        except (TypeError, ValueError) as exc:
            raise TypeError("interpolation_variogram_lags must be an integer.") from exc
        if lags < 1:
            raise ValueError("interpolation_variogram_lags must be positive.")
        if normalized_method == "idw" and maximum_distance is None:
            raise ValueError("interpolation_max_distance_in_feet is required for idw.")
        if normalized_method == "kriging" and variogram_distance is None:
            raise ValueError("interpolation_variogram_max_distance is required for kriging.")
        return normalized_method, maximum_distance, lags, variogram_distance

    def _validate_indices(self, indices: list[str] | None) -> tuple[str, ...]:
        """Validate and normalize the optional spectral-index list."""

        if indices is None:
            indices = []
        if not isinstance(indices, list):
            raise TypeError("indices must be a list of index names.")

        normalized = tuple(dict.fromkeys(str(name).strip().upper() for name in indices))
        unsupported = sorted(set(normalized).difference(self.SUPPORTED_INDICES))
        if unsupported:
            raise ValueError(f"Unsupported indices: {unsupported}. Supported indices: {sorted(self.SUPPORTED_INDICES)}")
        if self.calculate_indices and not normalized:
            raise ValueError("indices must contain at least one name when calculate_indices=True.")
        if not self.calculate_indices and normalized:
            raise ValueError("Set calculate_indices=True when providing an indices list.")
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

    def _validate_cleaning_options(
        self,
        remove_outliers: bool,
        iqr_quantiles: tuple[float, float],
        iqr_multiplier: float,
        fill_nulls: bool,
    ) -> tuple[bool, tuple[float, float], float, bool]:
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
        return (
            remove_outliers,
            (lower_quantile, upper_quantile),
            multiplier,
            fill_nulls,
        )

    def _required_bands(self) -> tuple[str, ...]:
        """Return reflectance and index-source bands without duplicates."""

        bands = list(self.BANDS)
        for index_name in self.indices:
            bands.extend(self.SUPPORTED_INDICES[index_name])
        return tuple(dict.fromkeys(bands))

    def _output_variables(self) -> tuple[str, ...]:
        """Return ordered band and index labels expected in openEO results."""
        return (*self.BANDS, *self.indices)

    def _run_signature(self) -> str:
        """Create a cache key from processing options, parcel IDs and geometry."""

        configuration = {
            "processing_strategy": "temporal_native_crs_local_v2",
            "start_date": self.start_date.strftime("%Y-%m-%d"),
            "end_date": self.end_date.strftime("%Y-%m-%d"),
            "temporal_period": self.temporal_period,
            "temporal_reducer": self.temporal_reducer,
            "working_epsg": self.working_epsg,
            "bands": self.BANDS,
            "indices": self.indices,
            "target_resolution": self.TARGET_RESOLUTION,
            "max_scene_cloud_cover": self.MAX_SCENE_CLOUD_COVER,
            "invalid_scl_classes": self.INVALID_SCL_CLASSES,
            "buffered_scl_classes": self.BUFFERED_SCL_CLASSES,
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
            raise ValueError(f"parcel_id values must be unique. Example duplicates: {examples.tolist()}")

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

    def _calculate_index(self, reflectance, index_name: str):
        """Calculate one spectral index for every valid acquisition pixel."""

        blue = reflectance.band("B02") if index_name == "EVI" else None
        green = reflectance.band("B03") if index_name in {"NDWI", "GNDVI"} else None
        red = reflectance.band("B04") if index_name in {"NDVI", "EVI", "SAVI"} else None
        nir = reflectance.band("B08")
        red_edge1 = reflectance.band("B05") if index_name in {"NDRE", "CI"} else None

        if index_name == "NDVI":
            index_cube = self._normalized_difference(nir, red)
        elif index_name == "NDWI":
            index_cube = self._normalized_difference(green, nir)
        elif index_name == "NDMI":
            index_cube = self._normalized_difference(nir, reflectance.band("B11"))
        elif index_name == "NBR":
            index_cube = self._normalized_difference(nir, reflectance.band("B12"))
        elif index_name == "GNDVI":
            index_cube = self._normalized_difference(nir, green)
        elif index_name == "EVI":
            denominator = nir + (6.0 * red) - (7.5 * blue) + 1.0
            index_cube = (2.5 * (nir - red) / denominator).mask(denominator == 0)
        elif index_name == "SAVI":
            denominator = nir + red + 0.5
            index_cube = (1.5 * (nir - red) / denominator).mask(denominator == 0)
        elif index_name == "NDRE":
            index_cube = self._normalized_difference(nir, red_edge1)
        elif index_name == "CI":
            index_cube = (nir / red_edge1) - 1.0
        else:  # Guarded by _validate_indices; retained for defensive use.
            raise ValueError(f"Unsupported index: {index_name}")

        # Restore a one-label bands dimension so the index can join the band cube.
        return index_cube.add_dimension(name="bands", label=index_name, type="bands")

    def _add_indices(self, masked_reflectance):
        """Append requested index bands to the masked acquisition-level cube."""

        # Keep the configured reflectance outputs while allowing extra source bands
        # (such as B11/B12) to be used only inside index formulas.
        output_cube = masked_reflectance.filter_bands(list(self.BANDS))
        for index_name in self.indices:
            index_cube = self._calculate_index(masked_reflectance, index_name)
            output_cube = output_cube.merge_cubes(index_cube)
        return output_cube

    def _build_monthly_cube(self, connection: openeo.Connection, feature_collection: dict):
        """Build the acquisition-to-monthly-pixel portion of the openEO graph.

        The order is intentionally fixed: load Sentinel-2 L2A, mask invalid
        acquisition pixels, calculate indices for each acquisition, and finally
        calculate the monthly median independently at every pixel.
        """

        # Load every source band required by either the output or an index formula.
        collection_bands = [*self._required_bands(), "SCL"]
        sentinel2 = connection.load_collection(
            "SENTINEL2_L2A",
            temporal_extent=[self.start_date.strftime("%Y-%m-%d"), self.end_date.strftime("%Y-%m-%d")],
            bands=collection_bands,
            max_cloud_cover=self.MAX_SCENE_CLOUD_COVER,
        ).filter_spatial(feature_collection)

        # Align all continuous bands before formulas combine 10 m and 20 m inputs.
        reflectance = sentinel2.filter_bands(list(self._required_bands())).resample_spatial(
            resolution=self.TARGET_RESOLUTION,
            projection=None,
            method="bilinear",
        )
        # Sentinel-2 L2A digital numbers use a reflectance scale factor of 10,000.
        reflectance = reflectance * 0.0001

        # Nearest-neighbor resampling preserves categorical SCL values.
        scl = sentinel2.band("SCL").resample_cube_spatial(reflectance, method="near")
        invalid_surface = self._combine_class_conditions(scl, self.INVALID_SCL_CLASSES)
        cloud_related = self._combine_class_conditions(scl, self.BUFFERED_SCL_CLASSES)
        # Expand cloud-related pixels by one output pixel to mask cloud edges.
        cloud_buffer = cloud_related.apply_kernel([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) > 0
        masked_reflectance = reflectance.mask(invalid_surface | cloud_buffer)

        # Indices are calculated after masking and before temporal aggregation.
        acquisition_cube = self._add_indices(masked_reflectance)
        if self.temporal_period == "1M" and self.start_date.is_month_start:
            return acquisition_cube.aggregate_temporal_period(period="month", reducer=self.temporal_reducer)

        intervals, labels = self._temporal_intervals()
        return acquisition_cube.aggregate_temporal(intervals=intervals, labels=labels, reducer=self.temporal_reducer)

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

        expected = list(self._output_variables())
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
        """Fill gaps using the nearest values before and after each interval."""
        filled = variable_values.copy()
        forward = filled.copy()
        for time_index in range(1, forward.shape[0]):
            missing = ~np.isfinite(forward[time_index])
            forward[time_index][missing] = forward[time_index - 1][missing]

        backward = filled.copy()
        for time_index in range(backward.shape[0] - 2, -1, -1):
            missing = ~np.isfinite(backward[time_index])
            backward[time_index][missing] = backward[time_index + 1][missing]

        missing = ~np.isfinite(filled)
        forward_valid = np.isfinite(forward)
        backward_valid = np.isfinite(backward)
        both = missing & forward_valid & backward_valid
        filled[both] = (forward[both] + backward[both]) / 2.0
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
            padded_values = np.pad(
                np.where(valid, layer, 0.0),
                1,
                mode="constant",
                constant_values=0.0,
            )
            padded_valid = np.pad(
                valid.astype("int16"),
                1,
                mode="constant",
                constant_values=0,
            )
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
            np.divide(
                neighbor_sum,
                neighbor_count,
                out=neighbor_mean,
                where=neighbor_count > 0,
            )
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
    ) -> tuple[np.ndarray, dict[tuple[int, int], dict]]:
        """Replace IQR outliers with NaN for each interval and variable."""
        cleaned = values.copy()
        cleaned[:, :, ~eligible_mask] = np.nan
        variables = list(self._output_variables())
        eligible_count = int(eligible_mask.sum())
        audit_state = {}

        for time_index, period_start in enumerate(periods):
            for variable_index, variable in enumerate(variables):
                layer = cleaned[time_index, variable_index]
                finite_values = layer[eligible_mask]
                finite_values = finite_values[np.isfinite(finite_values)]
                original_null_count = eligible_count - finite_values.size
                lower_bound = upper_bound = np.nan
                outlier_count = 0

                if self.remove_outliers and finite_values.size:
                    q1, q3 = np.quantile(finite_values, self.iqr_quantiles)
                    iqr = q3 - q1
                    lower_bound = float(q1 - self.iqr_multiplier * iqr)
                    upper_bound = float(q3 + self.iqr_multiplier * iqr)
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
                }
        return cleaned, audit_state

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
            max_distance_in_feet=self.interpolation_max_distance_in_feet,
            variogram_lags=self.interpolation_variogram_lags,
            variogram_lags_max_dist=self.interpolation_variogram_max_distance,
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
        """Run 3x3, temporal, then configured interpolation for one variable."""
        # Apply a 3x3 spatial mean to fill gaps before temporal interpolation.
        filled = self._fill_spatial_neighbors(variable_values, eligible_mask)
        # Apply a forward/backward temporal fill to fill gaps before geospatial interpolation.
        filled = self._fill_temporal_neighbors(filled)
        # Apply the configured geospatial interpolation to fill remaining gaps.
        for time_index in range(filled.shape[0]):
            filled[time_index] = self._interpolate_interval_layer(filled[time_index], eligible_mask, transform, cube_crs)
        filled[:, ~eligible_mask] = np.nan
        return filled

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
            for variable_index, _variable in enumerate(self._output_variables()):
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
    ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Run the complete pixel-cleaning pipeline before parcel statistics."""
        if values.shape[0] != len(periods):
            raise ValueError(f"Expected {values.shape[0]} period labels, received {len(periods)}.")

        # Counts must describe observed post-IQR pixels, not imputed pixels.
        cleaned, audit_state = self._remove_iqr_outliers(values, eligible_mask, batch_number, periods)
        observed = cleaned.copy()

        if self.fill_nulls:
            # Fill each variable cube per variable to avoid cross-variable contamination during interpolation.
            for variable_index, _variable in enumerate(self._output_variables()):
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

    def _calculate_local_statistics(self, netcdf_path, parcels, batch_number):
        """Clean monthly pixels and calculate local parcel statistics."""
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
            values = np.asarray(ordered.values, dtype="float64")
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
            raster_shape = (ordered.sizes[y_dimension], ordered.sizes[x_dimension])
            # Create a boolean mask for each parcel geometry, indicating which pixels fall within the parcel.
            parcel_masks = [
                geometry_mask(
                    [mapping(row.geometry)], out_shape=raster_shape, transform=transform, invert=True, all_touched=False
                )
                for _, row in projected.iterrows()
            ]
            # Combine all parcel masks to create a single eligible mask for pixel cleaning.
            eligible_mask = np.logical_or.reduce(parcel_masks) if parcel_masks else np.zeros(raster_shape, dtype=bool)
            # Clean pixels before calculating local statistics: IQR, 3x3 spatial
            # fill, temporal fill, then per-interval/variable interpolation.
            cleaned, observed, report = self._clean_monthly_pixels(
                values, eligible_mask, batch_number, periods, transform, cube_crs
            )
            rows = []
            variables = list(self._output_variables())
            for (_, parcel), parcel_mask in zip(parcels.iterrows(), parcel_masks):
                # Calculate requested statistics for the eligible pixels within the parcel mask.
                stats = self._reduce_local_pixels(cleaned[:, :, parcel_mask])
                if "count" in stats:
                    stats["count"] = np.isfinite(observed[:, :, parcel_mask]).sum(axis=-1)
                for time_index, period_start in enumerate(periods):
                    row = {
                        self.PARCEL_ID_FIELD: parcel[self.PARCEL_ID_FIELD],
                        "period_start": period_start,
                        "period_end": period_end_lookup[period_start],
                    }
                    for variable_index, variable in enumerate(variables):
                        for statistic in self.spatial_statistics:
                            value = stats[statistic][time_index, variable_index]
                            row[f"{variable}_{statistic}"] = (
                                int(value) if statistic == "count" and np.isfinite(value) else float(value)
                            )

                    rows.append(row)
        result = pd.DataFrame(rows)
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
        parcel_metrics["approx_pixel_count"] = parcel_metrics["parcel_area_m2"] / self.TARGET_RESOLUTION**2
        enriched = data.merge(parcel_metrics, on=self.PARCEL_ID_FIELD, how="left", validate="many_to_one")

        # Build every derived series first. Appending them together avoids the
        # fragmented block layout caused by repeated DataFrame column insertion.
        derived_columns: dict[str, pd.Series] = {}
        for variable in self._output_variables():
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
                valid_fraction = enriched[count] / enriched["approx_pixel_count"]
                derived_columns[f"{variable}_valid_pixel_fraction_approx"] = valid_fraction.clip(0.0, 1.0)

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

    def _run_openeo_jobs(self, cube_dir):
        """Create, run and download one openEO job per uncached parcel batch."""
        connection = None
        pending_batches, remote_jobs = [], []
        for batch_number, batch in enumerate(self._iter_spatial_batches(), start=1):
            netcdf_path = cube_dir / f"batch_{batch_number:05d}_monthly.nc"
            if netcdf_path.exists():
                self.logger.info(f"Using existing batch {batch_number} monthly cube {netcdf_path}.")
            else:
                if connection is None:
                    self.logger.info(f"Connecting to openEO backend {self.OPENEO_URL}.")
                    connection = openeo.connect(self.OPENEO_URL)
                    connection.authenticate_oidc()

                print(f"Creating openEO job for batch {batch_number} with {len(batch)} parcels.")
                feature_collection = self._to_feature_collection(batch)
                cube = self._build_monthly_cube(connection, feature_collection)
                temporary_path = netcdf_path.with_suffix(".partial.nc")
                if temporary_path.exists():
                    temporary_path.unlink()
                job = cube.create_job(out_format="netCDF", title=f"Sentinel-2 monthly pixel cube batch {batch_number}")
                remote_jobs.append((batch_number, job, temporary_path, netcdf_path))
            pending_batches.append((batch_number, batch, netcdf_path))
        if not pending_batches:
            raise RuntimeError("No processing batches were produced.")
        if remote_jobs:
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
                    self.logger.info(f"Cached completed batch {number} monthly cube at {path}.")
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
        csv_output = self.output_dir / "sentinel2_temporal_parcel_statistics.csv"
        parquet_output = self.output_dir / "sentinel2_temporal_parcel_statistics.parquet"
        final.to_csv(csv_output, index=False)
        final.to_parquet(parquet_output, index=False)
        self.cleaning_report = pd.concat(cleaning_reports, ignore_index=True)
        cleaning_output = self.output_dir / "sentinel2_pixel_cleaning_report.csv"
        self.cleaning_report.to_csv(cleaning_output, index=False)
        self.logger.info(f"Created {csv_output}, {parquet_output}, and {cleaning_output} " f"({len(final)} rows).")
        return final
