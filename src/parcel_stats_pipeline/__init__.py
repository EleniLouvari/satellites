"""Source-neutral public entry points for satellite parcel statistics.

Adapters own acquisition and band interpretation. Local cleaning, aggregation,
and output schemas are shared. Legacy imports remain supported.
"""

from openeo_parcel_stats_pipeline.zonal_stats import OpenEOZonalStats
from openeo_parcel_stats_pipeline.zonal_stats_job_manager import OpenEOJobManagerZonalStats
from hub_catalog.planet_parcel_stats import PlanetBasemapZonalStats
from openeo_parcel_stats_pipeline.crs import estimate_utm_epsg_from_parcels
from openeo_parcel_stats_pipeline.feature_reduction import reduce_annual_median_features, save_reduced_annual_parcel_features

__all__ = [
    "OpenEOZonalStats",
    "OpenEOJobManagerZonalStats",
    "PlanetBasemapZonalStats",
    "estimate_utm_epsg_from_parcels",
    "reduce_annual_median_features",
    "save_reduced_annual_parcel_features",
]
