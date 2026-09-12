"""Source-neutral public entry points for satellite parcel statistics.

Adapters own acquisition and band interpretation. Local cleaning, aggregation,
and output schemas are shared. Legacy imports remain supported.
"""

from importlib import import_module

_EXPORTS = {
    "OpenEOZonalStats": (".openeo", "OpenEOZonalStats"),
    "SatelliteZonalStats": (".openeo", "SatelliteZonalStats"),
    "OpenEOJobManagerZonalStats": (".job_manager", "OpenEOJobManagerZonalStats"),
    "JobManagerSatelliteZonalStats": (".job_manager", "JobManagerSatelliteZonalStats"),
    "PlanetBasemapZonalStats": (".planet", "PlanetBasemapZonalStats"),
    "ParcelStatsBase": (".core.base", "ParcelStatsBase"),
    "ParcelBatchPlanner": (".core.parcel_batches", "ParcelBatchPlanner"),
    "RasterCleaner": (".core.raster_cleaning", "RasterCleaner"),
    "ParcelStatisticsCalculator": (".core.parcel_statistics", "ParcelStatisticsCalculator"),
    "estimate_utm_epsg_from_parcels": (".core.crs", "estimate_utm_epsg_from_parcels"),
    "DEFAULT_REDUCED_FILENAME": ("..features.temporal", "DEFAULT_REDUCED_FILENAME"),
    "DEFAULT_TEMPORAL_SOURCES": ("..features.temporal", "DEFAULT_TEMPORAL_SOURCES"),
    "reduce_annual_median_features": ("..features.temporal", "reduce_annual_median_features"),
    "save_reduced_annual_parcel_features": ("..features.temporal", "save_reduced_annual_parcel_features"),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    """Resolve a public entry point without importing unrelated source clients."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
