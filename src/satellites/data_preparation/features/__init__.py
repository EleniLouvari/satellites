"""Public API for parcel-level environmental feature engineering."""

from .elevation import append_parcel_elevation, create_dem_mosaic, find_dem_tiles
from .spatial_context import (
    append_nearest_lake_features,
    append_soil_feature,
    calculate_centroid_soil_feature,
    calculate_nearest_lake_features,
    load_lakes,
    load_soil_polygons,
    validate_projected_metric_crs,
)

__all__ = [
    "append_nearest_lake_features",
    "append_parcel_elevation",
    "append_soil_feature",
    "calculate_centroid_soil_feature",
    "calculate_nearest_lake_features",
    "create_dem_mosaic",
    "find_dem_tiles",
    "load_lakes",
    "load_soil_polygons",
    "validate_projected_metric_crs",
]
