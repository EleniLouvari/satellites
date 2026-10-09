"""Shared geometry helper utilities."""


import logging
import math
import warnings

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pyproj
from IPython.display import display
from shapely.geometry import MultiPolygon, Point, Polygon
from tqdm.auto import tqdm as tq_auto

import shared.tabular as cm_l
from shared.assertions import expect_true


def calc_vector_azimuth_dist(p1, p2):
    """Calculate vector azimuth dist."""
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    dist = math.hypot(dx, dy)
    if dist == 0:
        return 0, dist
    degrees = math.degrees(math.atan2(dx, dy)) % 360
    return degrees, dist

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


def get_line_coordinates(geometry, reverse=False):
    """Create a list of the coordinates of a line.

    Keyword Arguments:
    geometry -- the geometry value, like LineString, Point, etc...
    reverse -- if True the reversed order of the coordinates will be returned
    Returns:
    the list of the coordinates

    """
    geom_coords = []
    try:
        geom_coords = list(geometry.coords)
    except Exception:
        logging.getLogger(__name__).debug("Error: get_line_coordinates failed; using its fallback.", exc_info=True)
        for line in geometry.geoms:
            geom_coords.extend(list(line.coords))

    if reverse:
        geom_coords.reverse()
    return geom_coords


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


def create_nearby_network(df, field_id, buffer_in_m=0.3048):
    """Create nearby network."""
    if "geometry" not in df.columns:
        raise ValueError("Error: geometry column is not included in the <fg> dataframe!")
    if field_id not in df.columns:
        raise ValueError(f"Error: {field_id} column is not included in the <fg> dataframe!")

    print("Start searching for nearby geometries")
    df = df.copy()
    buffer_radius = convert_value_in_m_to_df_units(df, buffer_in_m)

    # create the graph
    G = nx.Graph()
    col_left = field_id + "_left"
    col_right = field_id + "_right"

    bdf = df.copy()
    bdf['geometry'] = bdf['geometry'].apply(lambda x: x.buffer(buffer_radius))
    df_merge = gpd.sjoin(bdf, df, how="inner", predicate="intersects")
    df_merge = df_merge[df_merge[col_left] != df_merge[col_right]]

    if len(df_merge) == 0:
        print("There are no geometries connected")
        G.add_nodes_from(list(set(df[field_id])))
        return G, df

    df_connections = df_merge.groupby(by=col_left)[col_right].apply(list)
    df_connections = df_connections.reset_index()

    def find_connections(pipe):
        pipe_id = pipe[col_left]
        G.add_node(pipe_id)
        connected = list(set(pipe[col_right]))
        G.add_edges_from(list(zip([pipe_id] * len(connected), connected)))

    tq_auto.pandas(desc="Find nearby connections")
    df_connections.progress_apply(lambda row: find_connections(row), axis=1)
    tq_auto.pandas(desc="")

    G.add_nodes_from(list(df[~df[field_id].isin(list(df_connections[col_left]))][field_id]))
    return G


def order_components_bylen(G):
    """Order components bylen."""
    components = [c for c in nx.connected_components(G)]
    components = [[len(c), c] for c in components]
    components.sort(key=lambda x: x[0], reverse=True)
    return components


def create_outer_polygons(df, max_distance_in_m=91.44, simplify_in_m=24.384):
    """Create outer polygons."""
    if "geometry" not in df.columns:
        raise ValueError("Error: geometry column is not included in the dataframe")

    df = df.copy()
    buffer_dist = convert_value_in_m_to_df_units(df, max_distance_in_m)
    simplify_tolerance = convert_value_in_m_to_df_units(df, simplify_in_m)

    df['geometry'] = df['geometry'].buffer(buffer_dist)
    geoms = [df['geometry'].unary_union]

    gdf_areas = gpd.GeoDataFrame(geoms, geometry=geoms, crs=df.crs)
    gdf_areas = gdf_areas[['geometry']].copy()
    gdf_areas = explode_multigeometries(gdf_areas)
    gdf_areas = return_valid_geometries(gdf_areas)
    gdf_areas['geometry'] = gdf_areas['geometry'].apply(lambda x: Polygon(x.exterior).buffer(-int(9 * buffer_dist / 10)))
    gdf_areas['geometry'] = gdf_areas['geometry'].simplify(simplify_tolerance, preserve_topology=True)
    return gdf_areas


def create_polygon_from_bounds(bounds, buffer_in_m=0):
    """Create polygon from bounds."""
    buffer_radius = buffer_in_m
    if buffer_radius is None or np.isnan(buffer_radius):
        buffer_radius = 0

    if isinstance(bounds, gpd.geodataframe.GeoDataFrame):
        west, south, east, north = bounds.total_bounds
        if buffer_radius > 0:
            buffer_radius = convert_value_in_m_to_df_units(bounds, buffer_in_m)
    else:
        west, south, east, north = bounds

    p1, p2, p3, p4 = Point(west, south), Point(west, north), Point(east, north), Point(east, south)
    pointList = [p1, p2, p3, p4, p1]
    polygon_geom = Polygon([[p.x, p.y] for p in pointList])
    if buffer_radius > 0:
        polygon_geom = polygon_geom.buffer(buffer_radius)

    return polygon_geom


def get_coordinate_outliers(df, col_id, plot_graphs, figsize=(18,3), verbose=True, save_folder=None):
    """Get coordinate outliers."""
    df_check = df.copy()
    geom_errors = []
    min_coordinate_x = min_coordinate_y = np.finfo("float32").min
    if df.crs == "epsg:4326" or df.crs == 4326 or df.crs == {'init': 'epsg:4326'}:
        min_coordinate_x, min_coordinate_y = -180, -90
    df_check['geometry'] = df_check['geometry'].apply(lambda geom: geometry_to_point(geom))
    df_check['x'] = df_check['geometry'].x
    df_check['y'] = df_check['geometry'].y
    cond = (df_check['x'].isin([np.inf, -np.inf, np.nan])) | (df_check['y'].isin([np.inf, -np.inf, np.nan])) | \
           (df_check['x'] < min_coordinate_x) | (df_check['y'] < min_coordinate_y)
    geom_inf = df_check[cond].copy()
    geom_errors.append(geom_inf)
    df_check = df_check[~cond].copy()

    df_check['coords'] = df_check.apply(lambda row: math.sqrt(row['x'] ** 2 + row['y'] ** 2), axis=1)
    _min_Q1, max_Q3 = cm_l.get_IQR_outlier_limits(df_check, "coords", [0., 0.8], 2, verbose=verbose, plot=plot_graphs)
    geom_outliers = df_check[(df_check['coords'] > max_Q3)].copy()
    if not geom_outliers.empty:
        geom_errors.append(geom_outliers)
    geom_errors = pd.concat(geom_errors, ignore_index=True)
    if plot_graphs or save_folder:
        cm_l.plot_stat_numeric(df_check, column_name="coords", figsize=figsize, save_folder=save_folder)

    if not geom_errors.empty:
        if verbose:
            print(f"Invalid coordinates: {len(geom_errors)}")
            display(geom_errors.head())
        df = df[~df[col_id].isin(list(geom_errors[col_id]))].copy()
    return df, geom_errors


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
