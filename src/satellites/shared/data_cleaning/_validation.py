"""Small validation helpers shared by cleaning modules."""

from __future__ import annotations

from collections.abc import Iterable

import geopandas as gpd


def require_columns(dataframe, columns: Iterable[str]) -> None:
    """Raise a clear error when required columns are absent."""
    # Compute which of the requested column names are not present in the
    # provided dataframe and raise a helpful error listing them. This helps
    # callers fail fast with an actionable message when upstream data is
    # malformed or a column was renamed.
    missing = sorted(set(columns).difference(dataframe.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def representative_points(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return a copy whose active geometry contains representative points."""
    # Validate input type early to avoid surprising attribute errors below.
    if not isinstance(gdf, gpd.GeoDataFrame):
        raise TypeError("Expected a geopandas.GeoDataFrame.")
    # Work on a shallow copy so the caller's GeoDataFrame is not mutated.
    result = gdf.copy()
    # Use representative points to ensure one point per geometry (e.g. for
    # polygons) which is required by interpolation routines that expect point
    # coordinates rather than complex shapes.
    result.geometry = result.geometry.representative_point()
    return result


def projected_distance_units(gdf: gpd.GeoDataFrame, metres: float) -> float:
    """Convert metres to the linear units of a projected GeoDataFrame CRS."""
    # Ensure the GeoDataFrame has a CRS and that it is projected (not
    # geographic) because distance calculations assume linear units.
    if gdf.crs is None:
        raise ValueError("A CRS is required for distance-based interpolation.")
    if gdf.crs.is_geographic:
        raise ValueError("Distance-based interpolation requires a projected CRS.")
    # Normalize the input to a float and validate it is positive.
    value = float(metres)
    if value <= 0:
        raise ValueError("Distance must be positive.")
    # The CRS object exposes a conversion factor from axis units to metres.
    # Dividing the requested metres by that factor yields the equivalent
    # distance in the CRS' linear units (e.g. feet, metres).
    unit_to_metres = gdf.crs.axis_info[0].unit_conversion_factor
    return value / unit_to_metres
