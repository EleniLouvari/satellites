"""Shared geometry helper utilities."""


import warnings

import pyproj
from IPython.display import display
from shapely.geometry import MultiPolygon

from shared.assertions import expect_true


def explode_multigeometries(df, geom_column="geometry"):
    """Convert multi-geometries (like MultLineString, MultiPolygon, MultiPoint) to single geometries.

    Keyword Arguments:
    df -- the dataframe to update
    geom_column -- the geometry column
    Returns:
    the updated dataframe having only single geometries (Like Linestring, Polygon, Point)

    """
    if geom_column not in df.columns:
        raise ValueError(f"Error: {geom_column} column is not included in the dataframe!")

    df = df.copy()
    df = df.reset_index(drop=True)
    df = df.sort_index()
    df = df.explode(column=geom_column, index_parts=True).reset_index(drop=True)
    df.drop_duplicates(inplace=True)
    df.reset_index(inplace=True, drop=True)
    return df

def convert_multipolygons_to_polygons(df, geom_column="geometry"):
    """Keep one row per feature by retaining each MultiPolygon's largest part.

    Polygon geometries are returned unchanged. Smaller disconnected parts of a
    MultiPolygon are discarded, which is appropriate when each row represents
    one simple feature, such as an agricultural parcel. Unlike
    :func:`explode_multigeometries`, this function does not duplicate rows or
    feature identifiers.
    """
    if geom_column not in df.columns:
        raise ValueError(f"Error: {geom_column} column is not included in the dataframe!")

    converted = df.copy()

    def largest_polygon(geometry):
        if isinstance(geometry, MultiPolygon) and not geometry.is_empty:
            return max(geometry.geoms, key=lambda part: part.area)
        return geometry

    converted[geom_column] = converted[geom_column].map(largest_polygon)
    return converted

def geometry_to_point(geometry_value):
    """Geometry to point."""
    if geometry_value.geom_type.lower() == "point":
        return geometry_value

    if geometry_value.geom_type.lower() == "linestring":
        centroid = geometry_value.interpolate(0.5, True)

    elif geometry_value.geom_type.lower() == "multilinestring":
        sel_line = None
        max_len = 0
        for line in geometry_value.geoms:
            if line.length > max_len:
                max_len = line.length
                sel_line = line

        centroid = sel_line.interpolate(0.5, True)
    else:
        centroid = geometry_value.centroid

    if geometry_value.intersects(centroid.buffer(geometry_value.length / 100)):
        return centroid
    else:
        return geometry_value.representative_point()


def convert_geometries_to_points(df):
    """Convert the geometry of every row into a single point.

    Keyword Arguments:
    df -- the dataframe
    Returns:
    the updated dataframe

    """
    # this function is used by data checks - auto leak
    expect_true("geometry" in df.columns, "there is no geometry field")
    df = df.copy()
    df['geometry'] = df['geometry'].apply(lambda x: geometry_to_point(x))
    return df


def get_unit_of_length(df):
    """Get unit of length."""
    if "geometry" not in list(df.columns):
        raise ValueError("Error: There is no 'geometry' field in the dataframe")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if df.crs.to_epsg():
            unit_name = df.crs.coordinate_system.axis_list[0].unit_name.lower()
        else:
            pr = pyproj.crs.CRS.to_proj4(df.crs)
            ind = int(pr.find("+units="))
            if ind > 0:
                unit_name = pr[ind: ind + 40]
            else:
                unit_name = "other"

        # Normalize common unit names and known proj4 suffixes
        uname = unit_name.lower()
        if any(k in uname for k in ("us survey foot", "foot", "ft", "+units=ft", "+units=foot", "+units=us-ft")):
            unit_name = "ft"
        elif any(k in uname for k in ("metre", "meter", "+units=m")):
            unit_name = "m"
        else:
            raise ValueError("Error: Unrecognized projection unit!")

    return unit_name


def create_unit_conversion_factor(source_unit, target_unit):
    """Create unit conversion factor."""
    source_unit = source_unit.lower()
    target_unit = target_unit.lower()
    conversion_factors = {
        ("m", "ft"): 3.28084,
        ("ft", "m"): 0.3048,
        # Approximations based on 1 degree ≈ 111 km at the equator.
        ("degree", "m"): 111000.0,
        ("m", "degree"): 0.000009,
        ("degree", "ft"): 111000.0 * 3.28084,
        ("ft", "degree"): 0.000003,
    }
    if source_unit == target_unit:
        return 1.0
    try:
        return conversion_factors[(source_unit, target_unit)]
    except KeyError as exc:
        raise ValueError(f"Error: Cannot handle units: {source_unit}, {target_unit}") from exc

def convert_value_in_ft_to_df_units(df, value_in_ft):
    """Convert value in ft to df units."""
    local_df_unit = get_unit_of_length(df)
    conversion_factor = create_unit_conversion_factor(source_unit="ft", target_unit=local_df_unit)
    return value_in_ft * conversion_factor


def convert_value_in_m_to_df_units(df, value_in_m):
    """Convert value in m to df units."""
    local_df_unit = get_unit_of_length(df)
    conversion_factor = create_unit_conversion_factor(source_unit="m", target_unit=local_df_unit)
    return value_in_m * conversion_factor


def convert_value_in_df_units_from_ft(df, value_in_df_units):
    """Convert value in df units from ft."""
    local_df_unit = get_unit_of_length(df)
    conversion_factor = create_unit_conversion_factor(source_unit=local_df_unit, target_unit="ft")
    return value_in_df_units * conversion_factor

def convert_value_in_df_units_from_m(df, value_in_df_units):
    """Convert value in df units from m."""
    local_df_unit = get_unit_of_length(df)
    conversion_factor = create_unit_conversion_factor(source_unit=local_df_unit, target_unit="m")
    return value_in_df_units * conversion_factor

def get_ids_of_invalid_geoms(df, colid, geom_column="geometry"):
    """Get ids of invalid geoms."""
    df_buffer = df.copy()
    df_buffer[geom_column] = df_buffer[geom_column].buffer(0.001)
    df_invalid = df_buffer[(df_buffer[geom_column].is_empty) |
                           (df_buffer[geom_column].isna()) |
                           (df_buffer[geom_column].isnull()) |
                           (~df_buffer[geom_column].is_valid)].copy()
    invalid_ids = list(df_invalid[colid])
    return invalid_ids


def return_valid_geometries(df, geom_column="geometry"):
    """Find only valid geometries.

    Keyword Arguments:
    df -- the dataframe to check
    geom_column -- the geometry column

    Returns:
    a dataframe with the valid geometries only

    """
    expect_true(geom_column in df.columns, "there is no geometry field")
    df = df.copy()
    df.reset_index(inplace=True, drop=True)
    df['tmp_id'] = df.index
    invalid_ids = get_ids_of_invalid_geoms(df, colid="tmp_id", geom_column=geom_column)

    df_valid = df[~df['tmp_id'].isin(invalid_ids)].copy()
    df_valid.drop(columns=['tmp_id'], inplace=True)
    df_valid.reset_index(inplace=True, drop=True)
    return df_valid

def return_invalid_geometries(df, geom_column="geometry"):
    """Return invalid geometries."""
    expect_true(geom_column in df.columns, "there is no geometry field")
    df = df.copy()
    df.reset_index(inplace=True, drop=True)
    df['tmp_id'] = df.index
    invalid_ids = get_ids_of_invalid_geoms(df, colid="tmp_id", geom_column=geom_column)

    df_invalid = df[df['tmp_id'].isin(invalid_ids)].copy()
    df_invalid.drop(columns=['tmp_id'], inplace=True)
    df_invalid.reset_index(inplace=True, drop=True)
    return df_invalid

def get_duplicate_geometries(df, col_id, geometry_field="geometry", verbose=True):
    """Get duplicate geometries."""
    df = df.copy()
    df_issues = df[(df.duplicated(subset=geometry_field, keep=False))].sort_values(by=col_id).copy()
    df_issues = convert_geometries_to_points(df_issues)
    df_issues['x'] = df_issues[geometry_field].x
    df_issues['y'] = df_issues[geometry_field].y
    df_issues.sort_values(by=['x', 'y'], inplace=True)

    if verbose:
        if not df_issues.empty:
            print(f"Duplicate geometries: {len(df_issues)}")
            display(df_issues.head())
        else:
            print("There aren't any duplicate geometries")
    return df_issues
