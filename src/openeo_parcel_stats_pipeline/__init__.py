"""openEO extraction and parcel-level satellite statistics pipeline."""

from .zonal_stats import SatelliteZonalStats
from .zonal_stats_job_manager import JobManagerSatelliteZonalStats
from .crs import estimate_utm_epsg_from_parcels
from .feature_reduction import (
    DEFAULT_REDUCED_FILENAME,
    DEFAULT_TEMPORAL_SOURCES,
    reduce_annual_median_features,
    save_reduced_annual_parcel_features,
)

__all__ = [
    "DEFAULT_REDUCED_FILENAME",
    "DEFAULT_TEMPORAL_SOURCES",
    "JobManagerSatelliteZonalStats",
    "SatelliteZonalStats",
    "estimate_utm_epsg_from_parcels",
    "reduce_annual_median_features",
    "save_reduced_annual_parcel_features",
]
