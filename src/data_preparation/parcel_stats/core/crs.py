"""Coordinate-reference-system helpers for parcel extraction."""

from __future__ import annotations

import math

import geopandas as gpd
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info


def estimate_utm_epsg_from_parcels(parcels: gpd.GeoDataFrame) -> int:
    """Return the WGS84 UTM EPSG code containing the dissolved parcel centroid.

    The dissolved centroid is calculated in a provisional projected CRS to
    avoid taking a geometric centroid directly in longitude/latitude degrees.
    The final UTM CRS is selected from the PROJ database at that centroid, so
    northern and southern hemisphere EPSG codes are handled automatically.
    """

    if not isinstance(parcels, gpd.GeoDataFrame):
        raise TypeError("Error: parcels must be a GeoDataFrame.")
    if parcels.empty:
        raise ValueError("Error: parcels must not be empty.")
    if parcels.crs is None:
        raise ValueError("Error: parcels must have a CRS before estimating a UTM zone.")

    usable = parcels.loc[parcels.geometry.notna() & ~parcels.geometry.is_empty, ["geometry"]].copy()
    if usable.empty:
        raise ValueError("Error: parcels must contain at least one non-empty geometry.")

    provisional_crs = usable.estimate_utm_crs(datum_name="WGS 84")
    if provisional_crs is None:
        raise ValueError("Error: Could not estimate a provisional UTM CRS for the parcel extent.")
    projected = usable.to_crs(provisional_crs)
    dissolved_centroid = projected.geometry.union_all().centroid
    centroid_wgs84 = gpd.GeoSeries([dissolved_centroid], crs=projected.crs).to_crs(4326).iloc[0]
    longitude, latitude = float(centroid_wgs84.x), float(centroid_wgs84.y)
    if not math.isfinite(longitude) or not math.isfinite(latitude):
        raise ValueError("Error: The dissolved parcel centroid is not finite.")

    candidates = query_utm_crs_info(
        datum_name="WGS 84",
        area_of_interest=AreaOfInterest(
            west_lon_degree=longitude,
            south_lat_degree=latitude,
            east_lon_degree=longitude,
            north_lat_degree=latitude,
        ),
    )
    if not candidates:
        raise ValueError(
            f"Error: No WGS84 UTM CRS covers the parcel centroid at latitude {latitude:.6f}, longitude {longitude:.6f}."
        )
    return int(candidates[0].code)
