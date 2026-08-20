from import_libraries import *
import global_variables as gb_l
import common_libraries.generic_library as cm_l
import common_libraries.io_library as io_l


def calc_vector_azimuth_dist(p1, p2):
    """Calculates the azimuth and the length of a vector.

    Keyword arguments:
    p1 -- start_point
    p2 -- end_point
    :return:
    the azimuth of the vector ranging from 0 to 360 degrees and the length of the vector
    """
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    degrees = 0
    if dy != 0:
        degrees = math.degrees(math.atan(abs(dx) / abs(dy)))

    dist = math.sqrt(dx * dx + dy * dy)

    if abs(dx) == 0 and abs(dy) == 0:
        degrees = 0
    elif abs(dy) == 0:
        if dx > 0:
            degrees = 90
        else:
            degrees = 270
    elif abs(dx) == 0:
        if dy > 0:
            degrees = 0
        else:
            degrees = 180
    elif dx > 0 and dy < 0:
        degrees = 180 - degrees
    elif dx < 0 and dy < 0:
        degrees = 180 + degrees
    elif dx < 0 and dy > 0:
        degrees = 360 - degrees

    return degrees, dist

def get_line_coordinates_base(geometry):
    """Base function to extract coordinates from LineString or MultiLineString geometries."""
    if geometry.geom_type == "LineString":
        return list(geometry.coords)
    elif geometry.geom_type == "MultiLineString":
        return [coord for line in geometry.geoms for coord in line.coords]
    else:
        raise ValueError(f"Unsupported geometry type: {geometry.geom_type}")


def get_line_nodes(geometry):
    """Get line nodes as Point objects."""
    return [Point(coord) for coord in get_line_coordinates_base(geometry)]


def get_line_coordinates(geometry, reverse=False):
    """Get line coordinates as a list of tuples."""
    coords = get_line_coordinates_base(geometry)
    return coords[::-1] if reverse else coords


def extract_line_nodes(gdf):
    """Extracts all nodes from line geometries."""
    cm_l.check_needed_df_columns(gdf, ['geometry'])
    gdf = gdf.copy()

    # Extract nodes from each geometry
    nodes_gdf = gdf.copy()
    nodes_gdf = nodes_gdf.reset_index().rename(columns={'index': 'original_index'})
    nodes_gdf['geometry'] = nodes_gdf['geometry'].apply(get_line_nodes)
    nodes_gdf = nodes_gdf.explode("geometry")

    # Create a new GeoDataFrame with the nodes
    nodes_gdf = gpd.GeoDataFrame(nodes_gdf, geometry=nodes_gdf['geometry'], crs=gdf.crs)

    # Add node index within each original feature
    nodes_gdf['node_index'] = nodes_gdf.groupby('original_index').cumcount()
    nodes_gdf.reset_index(inplace=True, drop=True)

    return nodes_gdf


def get_line_segments_azimuths(x, norm=True):
    """
    1. Reads the geometry of lines.
    2. Find the coordinates of the vertices.
    3. Calculates the azimuth of each segment of the line.
    4. Calculates the length of each segment of the line.
    5. Return lists with the azimuths and the lengths of all the segments.

    Keyword arguments:
    x -- the line geometry
    norm -- if True the values will be normalized between 0-180 degrees
    :return: two lists:
    segments_az: the azimuth of each segment
    segments_length: the length of each segment
    """
    coords = get_line_coordinates(x)
    n = len(coords)

    segments_az = []
    segments_length = []
    for i in range(1, n):
        point_start = Point(coords[i - 1])
        point_end = Point(coords[i])
        az, dist = calc_vector_azimuth_dist(point_start, point_end)
        if norm:
            az = az - 180 if az >= 180 else az
        segments_az.append(az)
        segments_length.append(dist)

    return segments_az, segments_length


def calculate_line_azimuth(lines, column_name="line_azimuth", def_class=False, norm=True, step=30, geom_column="geometry"):
    lines = lines.copy()
    if def_class:
        lines[column_name] = None
    else:
        lines[column_name] = 0.0

    tq_auto.pandas(desc="Computing line azimuths")
    lines[column_name] = lines[geom_column].progress_apply(lambda x: calculate_representative_line_azimuth(x, norm, def_class, step))
    tq_auto.pandas(desc="")
    return lines


def get_begin(x):
    """Get the point of the first vertex of a line geometry.

    Keyword arguments:
    x -- the line geometry
    :return:
    the point of the first vertex
    """
    try:
        return Point(x.coords[0])
    except Exception:
        return Point(x.geoms[0].coords[0])


def get_end(x):
    """Get the point of the last vertex of a line geometry.

    Keyword arguments:
    x -- the line geometry
    :return:
    the point of the last vertex
    """
    try:
        return Point(x.coords[-1])
    except Exception:
        return Point(x.geoms[-1].coords[-1])


def get_line_startend(geometry):
    """Get the points of the first and last vertices of a line geometry.

    Keyword arguments:
    x -- the line geometry
    :return:
    the point of the first vertex
    the point of the last vertex
    """
    try:
        start_point = get_begin(geometry)
        end_point = get_end(geometry)
    except Exception:
        raise Exception(f"Unknown geometry type: {geometry.geom_type}")

    return start_point, end_point


def convert_line_start_end_to_points(df):
    """Returns the start & end points of the lines included in the <df> dataframe."""
    begin_points = df.copy()
    end_points = df.copy()
    begin_points['geometry'] = begin_points['geometry'].apply(lambda x: get_begin(x))
    end_points['geometry'] = end_points['geometry'].apply(lambda x: get_end(x))
    begin_points['type'] = "begin"
    end_points['type'] = "end"
    gdf_nodes = pd.concat([begin_points, end_points], ignore_index=True)
    return gdf_nodes


# def calculate_representative_line_azimuth(x, norm=True, defclass=False):
#     """Calculates the azimuths and the lengths of all the line segments and returns the azimuth of the segments
#     with the maximum length.

#     Keyword arguments:
#     x -- the line geometry
#     defclass -- if defclass = True it returns the classified value of the azimuth
#                 otherwise it returns the actual value of the azimuth
#     :return:
#     the line azimuth in degrees
#     """
#     azs, lengths = get_line_segments_azimuths(x, norm)

#     # get the azimuth of the longest segment
#     if len(lengths) > 0:
#         max_value = max(lengths)
#         max_index = lengths.index(max_value)
#         line_az = azs[max_index]

#         if defclass:
#             if (0 <= line_az < 15) or (165 <= line_az <= 180):
#                 return "N-S"
#             if 15 <= line_az < 45:
#                 return "NNE-SSW"
#             if 45 <= line_az < 75:
#                 return "NE-SW"
#             if 75 <= line_az < 105:
#                 return "E-W"
#             if 105 <= line_az < 135:
#                 return "ESE-WNW"
#             if 135 <= line_az < 165:
#                 return "SE-NW"
#         else:
#             return line_az
#     else:
#         return np.nan
def calculate_representative_line_azimuth(x, norm=True, defclass=False, step=30):
    """
    Calculates the azimuths and the lengths of all the line segments and returns the azimuth of the segments
    with the maximum length. If defclass is True, returns the azimuth range based on the step size.

    Keyword arguments:
    x -- the line geometry
    defclass -- if defclass = True it returns the classified value of the azimuth
                otherwise it returns the actual value of the azimuth
    step -- Step size for azimuth classification (default is 30 degrees)
    :return:
    the line azimuth in degrees or a classification string if defclass is True
    """
    azs, lengths = get_line_segments_azimuths(x, norm)

    # get the azimuth of the longest segment
    if len(lengths) > 0:
        max_value = max(lengths)
        max_index = lengths.index(max_value)
        line_az = azs[max_index]

        if defclass:
            # Normalize azimuth to be between 0 and 360
            line_az = line_az % 360

            # Calculate azimuth group based on step
            lower_bound = (line_az // step) * step
            upper_bound = lower_bound + step

            # Handle case where upper_bound exceeds 360
            if upper_bound > 360:
                upper_bound = 360

            # Return formatted string of the azimuth range
            return f"{int(lower_bound)}-{int(upper_bound)}"

        else:
            return line_az
    else:
        return np.nan


def add_shape_area_length(df, geom_column="geometry"):
    df = df.copy()
    for col in list(df.columns):
        if col.lower() == "shape_length" or col.lower() == "shape_area":
            df.drop(columns=[col], inplace=True)

    if "polygon" in df.iloc[0][geom_column].geom_type.lower():
        df['shape_area'] = df[geom_column].area
        df['shape_length'] = df[geom_column].length
    elif "line" in df.iloc[0][geom_column].geom_type.lower():
        df['shape_length'] = df[geom_column].length
    return df


def explode_multigeometries(df, geom_column="geometry"):
    """Convert multi-geometries (like MultLineString, MultiPolygon, MultiPoint) to single geometries.

    Keyword arguments:
    df -- the dataframe to update
    geom_column -- the geometry column
    Returns:
    the updated dataframe having only single geometries (Like Linestring, Polygon, Point)
    """
    if geom_column not in df.columns:
        raise Exception(f"{geom_column} column is not included in the dataframe!")

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
        raise ValueError(f"{geom_column} column is not included in the dataframe!")

    converted = df.copy()

    def largest_polygon(geometry):
        if isinstance(geometry, MultiPolygon) and not geometry.is_empty:
            return max(geometry.geoms, key=lambda part: part.area)
        return geometry

    converted[geom_column] = converted[geom_column].map(largest_polygon)
    return converted


def get_line_coordinates(geometry, reverse=False):
    """Create a list of the coordinates of a line.

    Keyword arguments:
    geometry -- the geometry value, like LineString, Point, etc...
    reverse -- if True the reversed order of the coordinates will be returned
    Returns:
    the list of the coordinates
    """
    geom_coords = []
    try:
        geom_coords = list(geometry.coords)
    except Exception:
        for line in geometry.geoms:
            geom_coords.extend(list(line.coords))

    if reverse:
        geom_coords.reverse()
    return geom_coords


def convert_geometry_three_to_two_d(df):
    """Takes a GeoSeries of 3D Multi/Polygons/Lines (has_z) and returns a list of 2D Multi/Polygons/Lines.

    Keyword arguments:
    df -- the dataframe to update
    Returns:
    the updated dataframe
    """
    df = df.copy()
    if "geometry" in list(df.columns):
        df['geometry'] = df['geometry'].force_2d()
        return df
    else:
        print("Drop Z: Dataframe does not contain geometry column")
        return df


def geometry_to_point(geometry_value):
    """Calculate the centroid of the geometry and if touches the line return the centroid.
    If not, then return the representative point.

    Keyword arguments:
    geometry_value -- the given geometry
    Returns:
    the converted geometry
    """
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

    Keyword arguments:
    df -- the dataframe
    Returns:
    the updated dataframe
    """
    # this function is used by data checks - auto leak
    assert "geometry" in df.columns, "there is no geometry field"
    df = df.copy()
    df['geometry'] = df['geometry'].apply(lambda x: geometry_to_point(x))
    return df


def get_unit_of_length(df):
    """
    Returns the unit of the length of the given geo dataframe <df>
    """
    if "geometry" not in list(df.columns):
        raise Exception("There is no 'geometry' field in the dataframe")

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
            raise Exception("Unrecognized projection unit!")

    return unit_name


def create_unit_conversion_factor(source_unit, target_unit):
    source_unit = source_unit.lower()
    target_unit = target_unit.lower()

    if source_unit == target_unit:
        conversion_factor = 1.0
    elif source_unit == "m" and target_unit == "ft":
        conversion_factor = 3.28084
    elif source_unit == "ft" and target_unit == "m":
        conversion_factor = 0.3048

    # the rest are approximations based on the
    # fact that at Equator 1 degree equals approximately 111 Km.
    # Going to the poles this changes and 1 degree equals less km
    elif source_unit == "degree" and target_unit == "m":
        conversion_factor = 111000
    elif source_unit == "m" and target_unit == "degree":
        conversion_factor = 0.000009
    elif source_unit == "degree" and target_unit == "ft":
        conversion_factor = 111000 * 3.28084
    elif source_unit == "ft" and target_unit == "degree":
        conversion_factor = 0.000003
    else:
        raise Exception(f"Cannot handle units: {source_unit}, {target_unit}")

    return conversion_factor


def convert_value_in_ft_to_df_units(df, value_in_ft):
    """
    Get the projection units of the df geo dataframe
    and converts a value from ft to projection units
    """
    local_df_unit = get_unit_of_length(df)
    conversion_factor = create_unit_conversion_factor(source_unit="ft", target_unit=local_df_unit)
    return value_in_ft * conversion_factor


def convert_value_in_df_units_from_ft(df, value_in_df_units):
    """
    Get the projection units of the df geo dataframe
    and converts a value from projection units to ft
    """
    local_df_unit = get_unit_of_length(df)
    conversion_factor = create_unit_conversion_factor(source_unit=local_df_unit, target_unit="ft")
    return value_in_df_units * conversion_factor


def get_ids_of_invalid_geoms(df, colid, geom_column="geometry"):
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

    Keyword arguments:
    df -- the dataframe to check
    geom_column -- the geometry column

    Returns:
    a dataframe with the valid geometries only
    """
    assert geom_column in df.columns, "there is no geometry field"
    df = df.copy()
    df.reset_index(inplace=True, drop=True)
    df['tmp_id'] = df.index
    invalid_ids = get_ids_of_invalid_geoms(df, colid="tmp_id", geom_column=geom_column)

    df_valid = df[~df['tmp_id'].isin(invalid_ids)].copy()
    df_valid.drop(columns=['tmp_id'], inplace=True)
    df_valid.reset_index(inplace=True, drop=True)
    return df_valid


def return_invalid_geometries(df, geom_column="geometry"):
    """The function detects invalid and null geometries by initially generating a temporary geometry column that
    contains buffered geometries. A buffer value of 1 foot is employed to identify instances where the geometry
    possesses invalid coordinates. Through this method, geometries with invalid coordinates produce an empty polygon,
    making it straightforward to subsequently select them.

    Keyword arguments:
    df -- the dataframe to check
    geom_column -- the geometry column

    Returns:
    a dataframe with the invalid geometries only
    """
    assert geom_column in df.columns, "there is no geometry field"
    df = df.copy()
    df.reset_index(inplace=True, drop=True)
    df['tmp_id'] = df.index
    invalid_ids = get_ids_of_invalid_geoms(df, colid="tmp_id", geom_column=geom_column)

    df_invalid = df[df['tmp_id'].isin(invalid_ids)].copy()
    df_invalid.drop(columns=['tmp_id'], inplace=True)
    df_invalid.reset_index(inplace=True, drop=True)
    return df_invalid


def preprocess_and_reproject_geometries(df, crs):
    df = convert_geometry_three_to_two_d(df)
    df = explode_multigeometries(df)
    df = return_valid_geometries(df)
    if df.crs != crs:
        df = df.to_crs(crs)
    return df


def create_nearby_network(df, field_id, buffer_in_feet=1):
    if "geometry" not in df.columns:
        raise Exception("geometry column is not included in the <fg> dataframe!")
    if field_id not in df.columns:
        raise Exception(f"{field_id} column is not included in the <fg> dataframe!")

    print("Start searching for nearby geometries")
    df = df.copy()
    buffer_radius = convert_value_in_ft_to_df_units(df, buffer_in_feet)

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
        return None

    tq_auto.pandas(desc="Find nearby connections")
    df_connections.progress_apply(lambda row: find_connections(row), axis=1)
    tq_auto.pandas(desc="")

    G.add_nodes_from(list(df[~df[field_id].isin(list(df_connections[col_left]))][field_id]))
    return G


def order_components_bylen(G):
    components = [c for c in nx.connected_components(G)]
    components = [[len(c), c] for c in components]
    components.sort(key=lambda x: x[0], reverse=True)
    return components


def nearest_point_distance_ckdtree(gdf, col_dist="near_dist", col_nearest_count="near_count", n_nearest=1, max_distance_in_ft=None, plot=False):
    """Calculates the distance of each geometry in the <gdf> geodataframe from the nearest of the rest geometries.
    The distance is saved in the column <col_dist>."""
    gdf = gdf.copy()

    # Ensure the GeoDataFrame is using a projected CRS
    if not gdf.crs or gdf.crs.is_geographic:
        print("Warning: GeoDataFrame should use a projected CRS for accurate distance calculations.")

    points = gdf.copy()
    points = convert_geometries_to_points(points)
    # Extract coordinates
    coords = np.array(list(points.geometry.apply(lambda x: (x.x, x.y))))

    max_dist = None
    if max_distance_in_ft is not None:
        max_dist = convert_value_in_ft_to_df_units(gdf, max_distance_in_ft)

    # Build the cKDTree
    tree = cKDTree(coords)

    # Query the tree for neighbors within max_dist
    # Use a large k value to get all possible neighbors within max_dist
    k = len(coords) - 1  # maximum possible number of neighbors
    if max_dist is None:
        distances, indices = tree.query(coords, k=k)
    else:
        distances, indices = tree.query(coords, k=k, distance_upper_bound=max_dist)

    # Calculate the average distance to neighbors within max_dist (excluding the point itself)
    avg_distances = []
    nearest_counts = []
    for i, (dist, idx) in enumerate(zip(distances, indices)):
        valid_distances = dist[1:][idx[1:] < len(coords)]  # Exclude self and invalid indices
        valid_count = np.sum(valid_distances <= max_dist) if max_dist is not None else len(valid_distances)
        nearest_counts.append(valid_count)
        if len(valid_distances) > 0:
            avg_distances.append(np.mean(valid_distances[:min(n_nearest, len(valid_distances))]))
        else:
            avg_distances.append(np.nan)  # No neighbors within max_dist

    # Add average distances and nearest counts to the GeoDataFrame
    gdf[col_dist] = avg_distances
    gdf[col_nearest_count] = nearest_counts

    if plot:
        for title, col in {'Average Distance': col_dist, 'Number of points': col_nearest_count}.items():
            fig, axes = plt.subplots(1, 2, figsize=(18, 5))
            x_label = f"{title}"
            if col == col_dist:
                x_label = x_label + f" ({gdf.crs.axis_info[0].unit_name})"
            # Histogram plot
            sns.histplot(gdf[col].dropna(), kde=True, color="skyblue", bins=100, ax=axes[0])
            axes[0].set_title(f"{title} to Up to {n_nearest} Nearest Points Within {max_distance_in_ft} ft")
            axes[0].set_xlabel(x_label)
            # Cumulative distribution plot
            sns.histplot(gdf[col].dropna(), cumulative=True, kde=True, color="skyblue", bins=100, ax=axes[1])
            axes[1].set_title(f"Cumulative Distribution of {title} to Up to {n_nearest} Nearest Points Within {max_distance_in_ft} ft")
            axes[1].set_xlabel(x_label)
            axes[1].set_ylabel("Cumulative Frequency")

        sns.jointplot(data=gdf, x=col_dist, y=col_nearest_count, kind="hist", height=7)
    return gdf


def get_unique_from_list(my_list):
    """Creates a list of the uniques values of the input list.

    Keyword arguments:
    my_list -- the list for we which we want to get the unique values
    :return:
    a list with the unique values
    """
    unique_list = []
    # traverse for all elements
    for x in my_list:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)

    return unique_list


def create_gdf_polygons_from_extend(gdf, num_polygons=10):
    """Creates a grid with polygons based on the <gdf> extend."""

    # get the bounding box of the geodataframe
    bbox = gdf.total_bounds

    # define the size of the polygons
    dx = (bbox[2] - bbox[0]) / num_polygons
    dy = (bbox[3] - bbox[1]) / num_polygons

    # create a grid of polygons covering the extent of the geodataframe
    polygons = []
    for i in range(num_polygons):
        for j in range(num_polygons):
            x_min = bbox[0] + i * dx
            y_min = bbox[1] + j * dy
            x_max = bbox[0] + (i + 1) * dx
            y_max = bbox[1] + (j + 1) * dy
            polygons.append(Polygon([(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max), (x_min, y_min)]))
    poly_gdf = gpd.GeoDataFrame(geometry=polygons, crs=gdf.crs)
    return poly_gdf


def get_grid_sub_sample(points_in_poly, poly_id, total_points, total_samples, seed):
    """Helper function for <create_uniform_spatial_random_sample>.
    Its purpose is to take a proportional number of samples from a geo-dataframe <points_in_poly>, containing geometries
    within a specific polygon. The function ensures that at least one sample is taken from each polygon.
    The function calculates the sample size by taking the maximum value between 1 and the proportion of data points in
    the target polygon to the total number of samples <n>. It then takes the calculated sample size from the
    <points_in_poly> geo-dataframe using random sampling with the provided seed number.
    Then assigns the `poly_id`, `sample_size`, and `count` (the number of geometries in the target polygon)
    to the sampled data points and returns the resulting sub-sample.

    Keyword arguments:
    points_in_poly -- a geo-dataframe containing points within the polygon
    poly_id -- a unique identifier for the target polygon within a geo-dataframe of polygons.
    total_points -- the total number of data points to sample
    n -- the total number of samples to select
    seed -- the seed number for sampling
    :return: the samples selected in the polygon
    """
    points_in_polygon = len(points_in_poly)
    sample_size = max(1, round(total_samples * points_in_polygon / total_points))
    sub_sample = points_in_poly.sample(n=sample_size, random_state=seed)
    sub_sample['poly_id'] = poly_id
    sub_sample['sample_size'] = sample_size
    sub_sample['count'] = points_in_polygon
    return sub_sample


def create_uniform_spatial_random_sample(df_to_sample, total_samples, seed, col_id, grid):
    """Creates a spatial uniform random sample, ensuring that the whole spatial extent of the geodataframe
    <df_to_sample> will be sampled.

    Keyword arguments:
    df_to_sample -- the dataframe to sample
    total_samples -- the total number of samples
    seed -- the seed number for sampling
    col_id -- the column name in the <df_to_sample> dataframe, holding the id
    grid -- the number of polygons in the x and y directions.
            The total number of polygons that will be created is grid x grid
    :return: the dataframe with the samples
    """
    df_points = df_to_sample[[col_id, 'geometry']].copy()
    poly_gdf = create_gdf_polygons_from_extend(df_points, grid)
    poly_gdf['poly_id'] = poly_gdf.index.to_series().apply(lambda x: f"pol{str(x)}")

    df_points = convert_geometries_to_points(df_points)
    poly_gdf = poly_gdf[poly_gdf['geometry'].intersects(df_points['geometry'].unary_union)]

    # create a proportional to n, random sample of points from each polygon
    total_points = len(df_points.index)
    samples = []
    for index, polygon in poly_gdf.iterrows():
        # selected the geometries from the df_points that fall within each polygon
        points_in_poly = df_points[df_points['geometry'].within(polygon['geometry'])]
        if len(points_in_poly):
            samples.append(get_grid_sub_sample(points_in_poly, polygon['poly_id'], total_points, total_samples, seed))

    sample_gdf = gpd.GeoDataFrame(pd.concat(samples), geometry="geometry", crs=df_points.crs)
    sample_gdf.drop_duplicates(subset=col_id, inplace=True)

    if total_samples > len(sample_gdf):
        # if the total number of samples is less than n, add additional random samples
        more_n = total_samples - len(sample_gdf)
        df_subset = df_points[~df_points[col_id].isin(list(sample_gdf[col_id]))].copy()
        new_sample = df_subset.sample(n=more_n, random_state=seed)
        sample_gdf = pd.concat([sample_gdf, new_sample], ignore_index=True)

    elif total_samples < len(sample_gdf):
        # if the total number of samples is more than n then remove samples,
        # first from the polygons having the most samples
        null_dist = 9999
        # we sort the polygons based on the number of samples they contain, and the number of the data fall in
        sample_gdf.sort_values(by=['sample_size', 'count'], ascending=[False, True], inplace=True)
        # we calculate the distance of each sample with respect to the others
        sample_gdf = nearest_point_distance_ckdtree(sample_gdf, "near_dist")
        sample_gdf['near_dist'] = sample_gdf['near_dist'].fillna(null_dist).astype(int)

        # we sample only the polygons having more than 1 sample
        to_sample_gdf = sample_gdf[sample_gdf['sample_size'] > 1].copy()
        if not to_sample_gdf.empty:
            # give priority to remove one sample from the ones that are close each-other
            unique_poly_ids = get_unique_from_list(list(to_sample_gdf['poly_id']))
            id_to_check = 0
            single_sample_subset = 0
            while len(sample_gdf) > total_samples and single_sample_subset <= len(unique_poly_ids):
                # loop through the polygons and remove one sample each time
                poly_to_check = unique_poly_ids[id_to_check]
                df_subset = sample_gdf[sample_gdf['poly_id'] == poly_to_check].sort_values(by="near_dist", ascending=True)

                if len(df_subset) > 1:
                    single_sample_subset = 0
                    # from the closest samples we remove the one and the others are marked in order not to be selected
                    # in the next iteration of the loop
                    cond = (sample_gdf['poly_id'] == poly_to_check) & \
                           (sample_gdf['near_dist'] == df_subset.iloc[0]['near_dist'])
                    sample_gdf.loc[cond, 'near_dist'] = null_dist
                    sample_gdf = sample_gdf[sample_gdf[col_id] != df_subset.iloc[0][col_id]]
                else:
                    single_sample_subset += 1

                id_to_check = (id_to_check + 1) % len(unique_poly_ids)
            if len(sample_gdf) > total_samples:
                sample_gdf = sample_gdf.sample(n=total_samples, random_state=seed)
        else:
            # if all the polygons have one sample each, then just remove the one, from every pair of samples that are
            # closest
            sample_gdf = sample_gdf.sort_values(by="near_dist", ascending=True)
            sample_gdf['mark'] = 0
            sample_gdf.loc[sample_gdf.index[::2], 'mark'] = 1
            sample_gdf = sample_gdf.sort_values(by=['mark', 'near_dist'], ascending=False).head(total_samples)

    random_sample = df_to_sample[df_to_sample[col_id].isin(list(sample_gdf[col_id]))]
    return random_sample


def create_voronoi_polygons(df_points, col_id, col_voronoi, df_extend=None, max_distance_in_ft=10000, simplify_in_ft=500):
    if len(df_points) == 0:
        print("Error: Create voronoi on empty geodataframe")
        return None

    cm_l.check_needed_df_columns(df_points, [col_id, 'geometry'])
    df_points = df_points[[col_id, 'geometry']].copy()

    if df_extend is not None:
        cm_l.check_needed_df_columns(df_extend, ['geometry'])
        df_extend = df_extend[['geometry']].copy()
    else:
        df_extend = df_points[['geometry']].copy()

    df_points = convert_geometries_to_points(df_points)
    voronoi_pols = gpd.GeoDataFrame(geometry=df_points.voronoi_polygons(), crs=df_points.crs)

    if isinstance(df_extend, gpd.geodataframe.GeoDataFrame):
        clip_polygon = create_outer_polygons(df_extend, max_distance_in_ft, simplify_in_ft)
        clip_polygon = clip_polygon.unary_union
        voronoi_pols = gpd.clip(voronoi_pols, clip_polygon)

    voronoi_pols = explode_multigeometries(voronoi_pols)
    voronoi_pols = return_valid_geometries(voronoi_pols)
    voronoi_pols[col_voronoi] = voronoi_pols['geometry'].area
    return voronoi_pols


def update_df_with_intersecting_polygons(df, items, col_id, item_cols):
    assert {col_id, 'geometry'}.issubset(df.columns), "df: one of the needed field: [id, geometry] is missing"
    assert set(item_cols + ['geometry']).issubset(items.columns), f"items: one of the needed field is missing: {item_cols} or geometry"

    len_doubles = len(df[df.duplicated(subset=[col_id], keep=False)])
    assert len_doubles == 0, f"{col_id} is not unique. \nFound {str(len_doubles)} non unique values in the df."

    df = df.copy()
    items = items[item_cols + ['geometry']].copy()
    items = items.to_crs(df.crs)

    items = return_valid_geometries(items)
    items = clip_dataframe(items, df, expand=100)
    items.reset_index(inplace=True, drop=True)

    object_cols, numerical_cols = cm_l.get_column_types(df)

    # update df with items, getting the attributes of the maximum intersection area
    buffer_radius = convert_value_in_ft_to_df_units(df, 1.64042)
    b_df = df.copy()
    b_df['geometry'] = b_df['geometry'].apply(lambda x: x.buffer(buffer_radius))
    b_df = return_valid_geometries(b_df)

    res_intersection = gpd.overlay(b_df, items, how="intersection")
    res_intersection = cm_l.fill_nulls(res_intersection, object_cols, numerical_cols, gb_l.string_null_values_list, gb_l.numeric_null_values_list, "Other", 0)
    res_intersection['inters_area'] = res_intersection['geometry'].apply(lambda x: x.area)
    res_intersection = res_intersection[[col_id] + item_cols + ['inters_area']].groupby([col_id] + item_cols).agg(sumarea=pd.NamedAgg('inters_area', 'sum')).reset_index()
    res_intersection = res_intersection.sort_values(by=[col_id, 'sumarea'], ascending=[True, True]).drop_duplicates(subset=col_id, keep="last")
    df = pd.merge(df[[col_id, 'geometry']], res_intersection[[col_id] + item_cols], how="left", on=col_id)
    return df


def create_outer_polygons(df, max_distance_in_ft=300, simplify_in_ft=80):
    if "geometry" not in df.columns:
        raise Exception("geometry column is not included in the dataframe")

    df = df.copy()
    buffer_dist = convert_value_in_ft_to_df_units(df, max_distance_in_ft)
    simplify_tolerance = convert_value_in_ft_to_df_units(df, simplify_in_ft)

    df['geometry'] = df['geometry'].buffer(buffer_dist)
    geoms = [df['geometry'].unary_union]

    gdf_areas = gpd.GeoDataFrame(geoms, geometry=geoms, crs=df.crs)
    gdf_areas = gdf_areas[['geometry']].copy()
    gdf_areas = explode_multigeometries(gdf_areas)
    gdf_areas = return_valid_geometries(gdf_areas)
    gdf_areas['geometry'] = gdf_areas['geometry'].apply(lambda x: Polygon(x.exterior).buffer(-int(9 * buffer_dist / 10)))
    gdf_areas['geometry'] = gdf_areas['geometry'].simplify(simplify_tolerance, preserve_topology=True)
    return gdf_areas


def create_polygon_from_bounds(bounds, buffer_in_ft=0):
    """
    Purpose: create a polygon geometry from the bound of a dataframe
    """
    buffer_radius = buffer_in_ft
    if buffer_radius is None or buffer_radius == np.nan:
        buffer_radius = 0

    if isinstance(bounds, gpd.geodataframe.GeoDataFrame):
        west, south, east, north = bounds.total_bounds
        if buffer_radius > 0:
            buffer_radius = convert_value_in_ft_to_df_units(bounds, buffer_in_ft)
    else:
        west, south, east, north = bounds

    p1, p2, p3, p4 = Point(west, south), Point(west, north), Point(east, north), Point(east, south)
    pointList = [p1, p2, p3, p4, p1]
    polygon_geom = Polygon([[p.x, p.y] for p in pointList])
    if buffer_radius > 0:
        polygon_geom = polygon_geom.buffer(buffer_radius)

    return polygon_geom


def clip_dataframe(df_toclip, bounds, expand=0):
    df_toclip = df_toclip.copy()

    if isinstance(bounds, gpd.geodataframe.GeoDataFrame):
        if df_toclip.crs != bounds.crs:
            bounds = bounds.to_crs(df_toclip.crs)

    polygon_geom = create_polygon_from_bounds(bounds, expand)
    result = gpd.clip(df_toclip, polygon_geom)

    return result


def get_lat_lon_bounds(df, expand=0):
    df = df.copy()
    if not isinstance(df, gpd.geodataframe.GeoDataFrame):
        raise Exception("Input data is not geodataframe!")

    polygon_geom = create_polygon_from_bounds(df, expand)
    df_fl = gpd.GeoDataFrame(geometry=[polygon_geom], crs=df.crs)
    df_fl = df_fl.to_crs("epsg:4326")
    return df_fl.total_bounds


def reproject_coordinates(source_crs_EPSG, dest_crs_EPSG, x, y):
    crs_src = CRS.from_epsg(source_crs_EPSG)
    crs_dst = CRS.from_epsg(dest_crs_EPSG)

    # Transform coordinates
    transformer = Transformer.from_crs(crs_src, crs_dst, always_xy=True)
    new_x, new_y = transformer.transform(x, y)
    return new_x, new_y


def find_nearest_item(df, nearest_item_id_field, items, item_id_field, max_distance_in_feet=100, distance_field=None,
                      item_to_point=False, drop_duplicate=True):
    """Finds the nearest item from <items> for each geometry in <df>.

    Returns a copy of the given <df> having a new field <nearest_item_id_field> where it stores the nearest
    id from <item_id_field> of the dataframe <items> that is within the given <max_distance_in_feet>.

    If <distance_field> is given, then it also creates that field and stores there the distance found.
    If <max_distance_in_feet> is None, then it doesn't limit the search area.
    if <item_to_point> = True, it converts the geometries of the items to points
    """
    start = time.time()

    assert {'geometry'}.issubset(df.columns), "df: geometry field is missing"
    assert {item_id_field, 'geometry'}.issubset(items.columns), "items: one of the needed field is missing"

    df = df.copy()
    items = items[[item_id_field, 'geometry']].copy()

    df = return_valid_geometries(df)
    items = return_valid_geometries(items)
    items = explode_multigeometries(items)
    items = items.drop_duplicates(subset="geometry")
    if item_to_point:
        items = convert_geometries_to_points(items)

    if items.crs != df.crs:
        items = items.to_crs(df.crs)

    save_dist = True
    if distance_field is None:
        save_dist = False
        distance_field = "tmp_col_distance"

    df.reset_index(inplace=True, drop=True)
    items.reset_index(inplace=True, drop=True)
    for col in [nearest_item_id_field, distance_field]:
        if col in list(df.columns):
            del df[col]

    df['tmpid'] = "p" + df.index.astype(str)
    items = items.rename(columns={item_id_field: nearest_item_id_field})
    if max_distance_in_feet:
        dist_search_max = convert_value_in_ft_to_df_units(df, max_distance_in_feet)
        df = gpd.sjoin_nearest(df, items, how="left", distance_col=distance_field,
                               max_distance=dist_search_max)
    else:
        df = gpd.sjoin_nearest(df, items, how="left", distance_col=distance_field)
    if drop_duplicate:
        # if multiple items have exactly the same distance from a geometry in df
        df = df.drop_duplicates(subset=['tmpid'], keep="first")

    df = df.drop(columns=['index_right', 'tmpid'])
    if not save_dist:
        del df[distance_field]

    end = time.time()
    print(f"\nFinding nearest item: Total elapsed time: {round((end - start) / 60, 2)} min")
    return df


def calculate_polygons_intersection_area(id1, id2, polygon1, polygon2):
    """Helper function for <create_line_network>. Finds the intersection area between two polygons.

    Keyword arguments:
    id1 -- the id of the polygon1
    id2 -- the id of the polygon2
    polygon1 -- the geometry of the first polygon
    polygon2 -- the geometry of the second polygon
    :return:
    the intersection area between the two polygons.
    """
    try:
        if type(polygon1) in [MultiPolygon, GeometryCollection]:
            polygon1 = max(polygon1.geoms, key=lambda geom: geom.area)
        if type(polygon2) in [MultiPolygon, GeometryCollection]:
            polygon2 = max(polygon2.geoms, key=lambda geom: geom.area)
        return polygon1.intersection(polygon2).area
    except Exception as err:
        print(f"Error:{err} while intersecting {id1} with {id2}.")
        return 0


def get_list_of_connected(g, x):
    """Helper function for lines_get_connected.

    Keyword arguments:
    g -- the network
    x -- the the line_id
    :return:
    a list of the connected lines
    """
    try:
        connections = list(g.neighbors(x))
    except Exception:
        connections = []
    return connections


def create_line_network(lines_to_update, lines_to_search,
                        lines_to_update_colid="line_id", lines_to_search_colid="line_id",
                        network_tolerance=1, check_line_continuous=False, max_az_dif=None, directed=False):
    """Creates a network with nodes and edges.

    Keyword arguments:
    lines_to_update -- the dataframe to update, with ID in the column <lines_to_update_colid>
    lines_to_search -- the dataframe to search, with ID in the column <lines_to_search_colid>
    tolerance -- the buffer size (in feet) around the start & end points
                 of each line, used to find the connected lines if line_continuous=True,
                 else is the buffer intersection of the lines
    line_continuous -- True, if we want to include in the network only the lines that intersect
                        in the start or end point of each line
    max_az_dif -- the maximum difference (in degrees) in the azimuths between two lines in order to be considered
                  as parallel. If equals None then parallels will not be checked and the function will run the
                  simple approach <create_simple_line_network>
    :return:
    the G <network graph>
    the dataframe <lines_to_update> with the column <connected> which is a list of the
    <lines_to_search_colid> of the connected
    """
    lines_to_update = lines_to_update.copy()
    lines_to_search = lines_to_search.copy()
    cm_l.check_needed_df_columns(lines_to_update, [lines_to_update_colid, 'geometry'])
    cm_l.check_needed_df_columns(lines_to_search, [lines_to_search_colid, 'geometry'])

    len_doubles = len(lines_to_update[lines_to_update.duplicated(subset=[lines_to_update_colid], keep=False)])
    assert len_doubles == 0, f"{lines_to_update_colid} is not unique.\nFound {len_doubles} non unique values " \
                             f"in the dataframe"
    len_doubles = len(lines_to_search[lines_to_search.duplicated(subset=[lines_to_search_colid], keep=False)])
    assert len_doubles == 0, f"{lines_to_search_colid} is not unique.\nFound {len_doubles} non unique values " \
                             f"in the dataframe"

    len_null = len(lines_to_update[(lines_to_update[lines_to_update_colid].isna())])
    assert len_null == 0, f"Found {len_null} null values of the {lines_to_update_colid} in the dataframe"
    len_null = len(lines_to_search[(lines_to_search[lines_to_search_colid].isna())])
    assert len_null == 0, f"Found {len_null} null values of the {lines_to_search_colid} in the dataframe"

    if lines_to_search.crs != lines_to_update.crs:
        lines_to_search = lines_to_search.to_crs(lines_to_update.crs)

    print("Start searching for connected lines")
    # create the network graph
    G = nx.DiGraph() if directed else nx.Graph()
    buffer_radius = convert_value_in_ft_to_df_units(lines_to_update, network_tolerance)
    buffer_intersect = convert_value_in_ft_to_df_units(lines_to_update, 0.001)

    if max_az_dif is not None:
        tq_auto.pandas(desc="Computing segment azimuths")
        lines_to_search['list_azim'] = lines_to_search['geometry'].progress_apply(lambda x:
                                                                                  get_line_segments_azimuths(x)[0])
        tq_auto.pandas(desc="")

    df_merge, gdf_nodes, merge_cols = find_connected_lines(lines_to_update, lines_to_search, lines_to_update_colid,
                                                           lines_to_search_colid, buffer_radius)
    if df_merge is None:
        print("There are no lines connected")
        G.add_nodes_from(list(set(lines_to_update[lines_to_update_colid])))
        return G, lines_to_update

    if max_az_dif is None:
        G = create_simple_line_network(G, lines_to_update, df_merge, merge_cols, lines_to_update_colid)
    else:
        G = create_detailed_line_network(G, lines_to_update, df_merge, gdf_nodes, lines_to_update_colid, merge_cols,
                                         buffer_radius, buffer_intersect, max_az_dif)

    if check_line_continuous:
        lines_to_update = check_line_continuous_in_network(G, lines_to_update, lines_to_search, lines_to_update_colid,
                                                           lines_to_search_colid, buffer_radius)
    return G, lines_to_update


def check_line_continuous_in_network(G, lines_to_update, lines_to_search,
                                     lines_to_update_colid, lines_to_search_colid, buffer_radius):
    """Helper function in <create_line_network>. This function checks if two connected lines are continuous in
    a network.

    Keyword arguments:
    G -- the network with the edges and nodes.
    lines_to_update -- the dataframe to update, with ID in the column <lines_to_update_colid>
    lines_to_search -- the dataframe to search, with ID in the column <lines_to_search_colid>
    buffer_radius -- the maximum distance (in data projection's metric) of the start/end points between two lines
                     should have in order to be defined as connected
    :return:
    the dataframe  <lines_to_update> with two new columns:
                   <from_edge> the id of the line that is connected with the start point of the checking line
                   <to_edge> the id of the line that is connected with the end point of the checking line
    """

    lines_to_update = lines_to_update[[lines_to_update_colid, 'geometry']].copy()
    line_ids = list(lines_to_update[lines_to_update_colid])
    lines_to_update = lines_to_update.set_index(lines_to_update_colid, drop=True)
    lines_to_search = lines_to_search.set_index(lines_to_search_colid, drop=True)
    from_edges, to_edges = [], []
    for line_id in tqdm(line_ids, total=len(line_ids), desc="Finding From/To edges"):
        line = lines_to_update.loc[line_id]
        b, e = get_begin(line['geometry']).buffer(buffer_radius), get_end(line['geometry']).buffer(buffer_radius)
        left, right = [], []
        for neighbor in G.neighbors(line_id):
            other = lines_to_search.loc[neighbor]
            other_b, other_e = get_begin(other['geometry']), get_end(other['geometry'])
            if b.contains(other_b) or b.contains(other_e):
                left.append(neighbor)
            elif e.contains(other_b) or e.contains(other_e):
                right.append(neighbor)
        from_edges.append(left)
        to_edges.append(right)

    lines_to_update['from_edge'] = from_edges
    lines_to_update['to_edge'] = to_edges
    return lines_to_update.reset_index()


def create_simple_line_network(G, lines_to_update, df_merge, merge_cols, lines_to_update_colid):
    """Helper function in <create_line_network>. This is a simple approach of the line network. We check only the
    intersection area of the lines, so we exclude from the connections the lines that have
    intersection area > 0.5 * min line area

    Keyword arguments:
    G -- the empty network object
    lines_to_update -- the dataframe to update, with ID in the column <lines_to_update_colid>
    df_merge -- the dataframe which is the result of the spatial join of the <lines_to_update> and the start/end points
                of the <lines_to_search> dataframe
    merge_cols -- the <lines_to_update_colid> and the <lines_to_search_colid> columns which, as a result of the merge
                  process, have the suffix <_left> <_right> if they are identical,
                  and are referenced as [col_left, col_right]
    :return:
    the updated network G with the nodes and edges
    """

    df_merge = df_merge[df_merge['inters_area'] < 0.5 * df_merge['min_area']]
    df_connections = df_merge.groupby(by=merge_cols[0])[merge_cols[1]].apply(list)

    def find_connections_simple(line):
        pid = line[lines_to_update_colid]
        G.add_node(pid)
        if pid in df_connections:
            connected = list(set(df_connections[pid]))
            G.add_edges_from(list(zip([pid] * len(connected), connected)))
        return None

    tq_auto.pandas(desc="Creating Network")
    lines_to_update.progress_apply(lambda row: find_connections_simple(row), axis=1)
    tq_auto.pandas(desc="")
    return G


def create_detailed_line_network(G, lines_to_update, df_merge, gdf_nodes, lines_to_update_colid, merge_cols,
                                 buffer_radius, buffer_intersect, max_az_dif):
    """Helper function in <create_line_network>. It performs a more thorough search in order to find the connected
    lines.

    Keyword arguments:
    G -- the empty network object
    lines_to_update -- the dataframe to update, with ID in the column <lines_to_update_colid>
    df_merge -- the dataframe which is the result of the spatial join of the <lines_to_update> and the start/end points
                of the <lines_to_search> dataframe.
    gdf_nodes -- the dataframe with the start/end points of the <lines_to_search> dataframe
    merge_cols -- the <lines_to_update_colid> and the <lines_to_search_colid> columns which, as a result of the merge
                  process, have the suffix <_left> <_right> if they are identical,
                  and are referenced as [col_left, col_right]
    buffer_radius -- the maximum distance (in data projection's metric) that the start/end points of two lines
                     should have in order to be defined as connected
    max_az_dif -- the maximum difference (in degrees) in the azimuths between two lines in order to be considered
                  as parallel
    :return:
    the updated network G with the nodes and edges
    """

    # 1. exclude the lines that are almost parallel and identical
    df_merge = df_merge[df_merge['inters_area'] < 0.9 * df_merge['min_area']]
    # 2. create a dataframe with all the lines and their possible connected lines
    gdf_nodes.rename(columns={'geometry': 'point'}, inplace=True)
    df_merge = pd.merge(df_merge, gdf_nodes[['point']], how="left", left_on="index_right", right_index=True)
    df_connections = df_merge.groupby(merge_cols[0]).agg(
        {merge_cols[1]: list, 'min_area': list, 'inters_area': list, 'point': list, 'type': list, 'list_azim': list})
    df_connections_list = list(df_connections.index)

    def find_connections_detail(line):
        drop_ids = []
        pid, geom = line[lines_to_update_colid], line.geometry.interpolate(0.5, True)
        G.add_node(pid, pos=(geom.x, geom.y))

        if pid in df_connections_list:
            line_rect, line_buffer = line['line_rect'], line['line_buffer']
            # 3. create a geo dataframe with the possible connected lines for the selected line
            df_check_row = df_connections.loc[pid]
            df_check = gpd.GeoDataFrame({merge_cols[1]: df_check_row[merge_cols[1]],
                                         'min_area': df_check_row['min_area'],
                                         'point': df_check_row['point'],
                                         'type': df_check_row['type'],
                                         'list_azim': df_check_row['list_azim'],
                                         'inters_area': df_check_row['inters_area']})
            df_check.set_geometry("point", inplace=True)
            df_check.crs = lines_to_update.crs

            # 4. exclude from the checks the lines that lie at the end or start of the selected line
            ids_to_check = list(set(df_check[merge_cols[1]]))
            line_rect_in = line_rect.buffer(-2 * buffer_intersect)
            line_buffer_in = line_buffer.buffer(-2 * buffer_intersect)
            temp_df = df_check[
                (~df_check['point'].within(line_rect_in)) & (df_check['point'].within(line_buffer_in))]
            if not temp_df.empty:
                ids_to_check = list(set(ids_to_check) - set(temp_df[merge_cols[1]]))

            # 5. check which lines are parallels to the nearest line segment
            if len(ids_to_check) > 0:
                line_splitted = split(line['geometry'], MultiPoint(get_line_coordinates(line['geometry'])))
                parallel_ids = []
                for check_id in ids_to_check:
                    check = df_check[df_check[merge_cols[1]] == check_id].iloc[0]
                    # get the azimuth of the specific segment of the checking line
                    line_azimuth = list(check['list_azim'])[0] if check['type'] == "begin" else \
                        list(check['list_azim'])[-1]

                    # get the segment of the line which is nearest to the checking line
                    line_segment, min_dist = None, 2 * buffer_radius
                    for seg_geom in line_splitted.geoms:
                        dist = seg_geom.distance(check['point'])
                        if min_dist > dist:
                            min_dist, linee_segment = dist, seg_geom

                    # compute the angle between the line segment and the nearest line segment
                    line_azimuth = calculate_representative_line_azimuth(line_segment)
                    az_dif = abs(line_azimuth - line_azimuth)
                    az_dif = 180 - az_dif if az_dif > 90 else az_dif
                    if az_dif <= max_az_dif:
                        line_intersect = line['geometry'].buffer(buffer_intersect)
                        check_intersect = df_check[(df_check['point'].within(line_intersect))]
                        doubles = list(set(
                            check_intersect[check_intersect.duplicated(subset=[merge_cols[1]], keep=False)][
                                merge_cols[1]]))

                        # find the parallel lines in order to check them if they are truly connected with the line or \
                        # they are just parallel
                        if check_id in list(set(check_intersect[merge_cols[1]])):
                            if check_id in doubles and az_dif < 2:
                                drop_ids.append(check_id)
                            elif check_id in doubles and az_dif >= 2:
                                parallel_ids.append(check_id)
                            elif check_id not in doubles and az_dif < 2:
                                parallel_ids.append(check_id)
                        else:
                            parallel_ids.append(check_id)
                ids_to_check = parallel_ids

            # 6. final exclusion of the parallel lines that have intersection area > 0.5 of min area
            if len(ids_to_check) > 0:
                temp_df = df_check[df_check[merge_cols[1]].isin(ids_to_check)]
                temp_df = temp_df[(temp_df['inters_area'] >= (0.5 * temp_df['min_area']))]
                if not temp_df.empty:
                    drop_ids.extend(list(temp_df[merge_cols[1]]))

            # 7. final selection of the connected lines
            if len(drop_ids) > 0:
                df_check = df_check[~df_check[merge_cols[1]].isin(drop_ids)]

            if not df_check.empty:
                df_unique = list(set(df_check[merge_cols[1]]))
                G.add_edges_from(list(zip([pid] * len(df_unique), df_unique)))
        return None

    tq_auto.pandas(desc="Creating detailed Network")
    lines_to_update.progress_apply(lambda row: find_connections_detail(row), axis=1)
    tq_auto.pandas(desc="")
    return G


def find_connected_lines(lines_to_update, lines_to_search, lines_to_update_colid, lines_to_search_colid, buffer_radius):
    """Helper function for <create_pipe_network>. For each line in the <lines_to_update> dataframe finds the connected
    lines from the <lines_to_search> dataframe.

    Keyword arguments:
    lines_to_update -- the dataframe to update, with ID in the column <lines_to_update_colid>
    lines_to_search -- the dataframe to search, with ID in the column <lines_to_search_colid>
    buffer_radius -- the maximum distance (in data projection's metric) of the start/end points between two lines
                     should have in order to be defined as connected
    :return:
    <df_merge> the dataframe which is the result of spatial join between the <lines_to_update> and <lines_to_search>
    <gdf_nodes> the dataframe with the start/end points of the <lines_to_search>
    [col_left, col_right] a list with the ids <lines_to_update_colid> and <lines_to_search_colid> having the suffix:
    '_left' and '_right' appended, respectively, if they are identical
    """

    # compute line buffer rectangle and area
    lines_to_update['pipe_buffer'] = lines_to_update['geometry'].buffer(buffer_radius)
    lines_to_update['pipe_rect'] = lines_to_update['geometry'].buffer(buffer_radius, join_style=2, cap_style=2)
    lines_to_update['pipe_area'] = lines_to_update['pipe_rect'].area
    lines_to_update = lines_to_update[[lines_to_update_colid, 'pipe_buffer', 'pipe_rect', 'pipe_area', 'geometry']]

    # get only the needed columns of the dataframe to search
    lines_to_search['line_rect'] = lines_to_search['geometry'].buffer(buffer_radius, join_style=2, cap_style=2)
    lines_to_search['line_area'] = lines_to_search['line_rect'].area
    cols_needed = [lines_to_search_colid, 'line_rect', 'line_area']
    if "list_azim" in list(lines_to_search.columns):
        cols_needed = cols_needed + ['list_azim']
    lines_to_search = lines_to_search[cols_needed + ['geometry']]

    # create a geodataframe with the start-end points of the lines
    gdf_nodes = convert_line_start_end_to_points(lines_to_search)

    cm_l.reset_index(lines_to_update, gdf_nodes)
    if lines_to_update_colid == lines_to_search_colid:
        col_left, col_right = lines_to_update_colid + "_left", lines_to_search_colid + "_right"
    else:
        col_left, col_right = lines_to_update_colid, lines_to_search_colid

    df_merge = lines_to_update.copy()
    df_merge.set_geometry("pipe_buffer", inplace=True)
    df_merge = gpd.sjoin(df_merge, gdf_nodes, how="inner", predicate="contains")
    if lines_to_update_colid == lines_to_search_colid:
        df_merge = df_merge[df_merge[col_left] != df_merge[col_right]]

    if len(df_merge) == 0:
        return None, None, []

    df_merge['min_area'] = df_merge.apply(lambda row: min(row['pipe_area'], row['line_area']), axis=1)
    df_merge['inters_area'] = df_merge.apply(lambda row: calculate_polygons_intersection_area(row[col_left],
                                                                                              row[col_right],
                                                                                              row["pipe_rect"],
                                                                                              row["line_rect"]),
                                             axis=1)
    return df_merge, gdf_nodes, [col_left, col_right]


def lines_get_connected(lines, field_id="line_id", tolerance=1, col_null=None, allow_zeros=True,
                        line_continuous=False, max_az_dif=None):
    """Find the connected lines for each line in the dataframe <df>.

    Keyword arguments:
    field_id  -- the column id of the line's
    tolerance -- the maximum distance (in ft) between the lines nodes in order to handle them as connected
    col_Null  -- if this column not equals None, then the script finds the connected only for the lines
                 with values in this column==None
    allow_zeros -- if col_Null<>None and allow_zeros=False then the script finds the connected only for the lines
                   with values in this column==None and <> zero
    line_continuous -- if True, get the connected only at start or end point of each line
    max_az_dif -- the maximum difference in the azimuths between two lines in order to be considered as parallel
                  if max_az_dif==None then parallels will not be checked
    :return:
    the lines dataframe with added column 'connected'
    * if line_continuous=True, it will add two more columns:
    from_edge: lines are connected at the start point of the lines
    to_edge  : lines are connected at the end point of the lines
    """
    start = time.time()

    lines = lines.copy()

    len_doubles = len(lines[lines.duplicated(subset=[field_id], keep=False)])
    assert len_doubles == 0, f"{field_id} is not unique.\nFound {len_doubles} non unique values in the lines"

    if "geometry" not in lines.columns:
        raise Exception("geometry column is not included in the lines dataframe!")

    if "connected" in lines.columns:
        del lines['connected']

    if col_null is not None:
        condition = (lines[col_null].isna())
        if not allow_zeros:
            condition = condition | (lines[col_null] == 0)
        df_to_update = lines[condition].copy()
    else:
        df_to_update = lines.copy()

    g, df_to_update = create_line_network(df_to_update, lines, field_id, field_id, tolerance, line_continuous,
                                          max_az_dif)
    tq_auto.pandas(desc="Adding connected")
    df_to_update['connected'] = df_to_update[field_id].progress_apply(lambda x: get_list_of_connected(g, x))
    tq_auto.pandas(desc="")

    if line_continuous:
        lines = pd.merge(lines, df_to_update[[field_id, 'connected', 'from_edge', 'to_edge']], on=field_id, how="left")
    else:
        lines = pd.merge(lines, df_to_update[[field_id, 'connected']], on=field_id, how="left")

    print("Finish searching for connected lines")
    end = time.time()
    print(f"Total elapsed time: {round((end - start) / 60, 2)} min")
    return lines


def remove_multiple_points_within_radius(df_reference, gdf_poi, col_category="category", search_radius_in_ft=350):
    gdf_poi = gdf_poi.copy()
    gdf_poi.reset_index(inplace=True, drop=True)
    gdf_poi['checked'] = False
    gdf_poi['delete'] = False

    clean_radius = convert_value_in_ft_to_df_units(df_reference, search_radius_in_ft)

    spatial_index = gdf_poi.sindex
    for index, row in tqdm(gdf_poi.iterrows(), total=len(gdf_poi.index)):
        geom = row['geometry']
        cat = row[col_category]

        if not gdf_poi.loc[index]['checked']:
            polygon = geom.buffer(clean_radius)
            possible_items_index = list(spatial_index.intersection(polygon.bounds))
            if len(possible_items_index) > 0:
                possible_items = gdf_poi.iloc[possible_items_index]
                precise_items = possible_items[possible_items.intersects(polygon)]
                if len(precise_items.index) > 0:
                    precise_items = precise_items.loc[
                        (precise_items[col_category] == str(cat)) & (precise_items['checked'] == False) & (precise_items.index != index)]
                    precise_items_index = list(precise_items.index)
                    if len(precise_items_index) > 0:
                        gdf_poi.loc[precise_items_index, 'checked'] = True
                        gdf_poi.loc[precise_items_index, 'delete'] = True
        gdf_poi.at[index, 'checked'] = True

    gdf_poi = gdf_poi[~gdf_poi['delete']].copy()
    gdf_poi.drop(columns = ['delete', 'checked'], inplace=True)
    gdf_poi.reset_index(drop=True, inplace=True)
    return gdf_poi


def orient_lines_to_flow(gdf, col_id, target_col_id):
    # Create a copy of the original GeoDataFrame
    oriented_gdf = gdf.copy()

    # Create a network graph from the lines
    G = nx.Graph()
    for idx, row in oriented_gdf.iterrows():
        line = row.geometry
        start, end = line.coords[0], line.coords[-1]
        G.add_edge(start, end, id=row[col_id], geometry=line, index=idx)

    # Find the target edge
    target_edge = next(edge for edge in G.edges(data=True) if edge[2]['id'] == target_col_id)

    # Determine which node of the target edge is the downstream node
    node1, node2 = target_edge[0], target_edge[1]
    connected_to_node1 = sum(1 for _ in G.neighbors(node1) if _ != node2)
    connected_to_node2 = sum(1 for _ in G.neighbors(node2) if _ != node1)

    target_node = node2 if connected_to_node1 > connected_to_node2 else node1

    # Create a directed graph for the flow
    DG = nx.DiGraph()

    # Use BFS to orient the lines
    queue = deque([(target_node, None)])
    visited = set([target_node])

    while queue:
        current_node, parent = queue.popleft()
        for neighbor in G.neighbors(current_node):
            if neighbor != parent:
                edge_data = G.get_edge_data(current_node, neighbor)
                line = edge_data['geometry']
                if line.coords[-1] != current_node:
                    line = LineString(list(line.coords)[::-1])
                DG.add_edge(neighbor, current_node, id=edge_data['id'], geometry=line, index=edge_data['index'])
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, current_node))

    # Update the geometries in the GeoDataFrame
    for _, _, data in DG.edges(data=True):
        oriented_gdf.loc[data['index'], 'geometry'] = data['geometry']

    # Update the geometry of the target line to end at the target node
    target_line = oriented_gdf[oriented_gdf[col_id] == target_col_id].iloc[0]['geometry']
    if get_begin(target_line).buffer(target_line.length/2).contains(Point(target_node)):
        target_line = LineString(get_line_coordinates(target_line, reverse=True))
        oriented_gdf.loc[oriented_gdf[col_id] == target_col_id, 'geometry'] = target_line
        # Update the corresponding edge in the DG graph
        target_edge_in_DG = next((u, v, data) for u, v, data in DG.edges(data=True) if data['id'] == target_col_id)
        DG[target_edge_in_DG[0]][target_edge_in_DG[1]]['geometry'] = target_line

    return DG, oriented_gdf


def calculate_shreve_strahler(DG, oriented_gdf, col_strahler="strahler", col_shreve="shreve"):
    oriented_gdf = oriented_gdf.copy()
    # Initialize all edges with Strahler order 1
    nx.set_edge_attributes(DG, 1, col_strahler)
    nx.set_edge_attributes(DG, 1, col_shreve)

    # Calculate Shreve order and update Strahler order
    for node in nx.topological_sort(DG):  # Process nodes from upstream to downstream
        in_edges = list(DG.in_edges(node))
        if not in_edges:  # This is a source (leaf)
            for out_edge in DG.out_edges(node):
                DG.edges[out_edge][col_shreve] = 1
        else:
            shreve_sum = 0
            strahler_values = []
            for in_edge in in_edges:
                shreve_sum += DG.edges[in_edge][col_shreve]
                strahler_values.append(DG.edges[in_edge][col_strahler])

            # Update Shreve order for outgoing edges
            for out_edge in DG.out_edges(node):
                DG.edges[out_edge][col_shreve] = shreve_sum

            # Update Strahler order for outgoing edges
            max_strahler = max(strahler_values)
            if strahler_values.count(max_strahler) > 1:
                new_strahler = max_strahler + 1
            else:
                new_strahler = max_strahler

            for out_edge in DG.out_edges(node):
                DG.edges[out_edge][col_strahler] = new_strahler

    # Update the GeoDataFrame with calculated orders
    for _, _, data in DG.edges(data=True):
        oriented_gdf.loc[data['index'], col_shreve] = data[col_shreve]
        oriented_gdf.loc[data['index'], col_strahler] = data[col_strahler]
    return oriented_gdf


def split_lines_at_intersections(gdf, col_id=None):
    """
    Splits lines at their intersections in a GeoDataFrame and retains original attributes.

    Parameters:
    gdf (GeoDataFrame): A GeoDataFrame containing LineString geometries.

    Returns:
    GeoDataFrame: A new GeoDataFrame with lines split at intersections and original attributes retained.
    """
    # Ensure the input GeoDataFrame is valid
    if gdf.empty or not all(gdf.geometry.type == 'LineString'):
        raise ValueError("Input GeoDataFrame must contain LineString geometries")

    # Collect all intersection points
    intersection_points = []
    for i, line1 in enumerate(gdf.geometry):
        for j, line2 in enumerate(gdf.geometry):
            if i >= j:  # Avoid duplicate checks and self-intersection
                continue
            intersection = line1.intersection(line2)
            if not intersection.is_empty:
                if intersection.geom_type == 'Point':
                    intersection_points.append(intersection)
                elif intersection.geom_type == 'MultiPoint':
                    intersection_points.extend(intersection.geoms)  # Corrected here

    # Remove duplicate points
    unique_intersection_points = MultiPoint(list(set(intersection_points)))

    # Prepare a list to hold the split lines and their corresponding attributes
    split_lines_with_attributes = []

    # Split lines at intersection points and retain attributes
    for idx, row in gdf.iterrows():
        line = row.geometry
        if line.intersects(unique_intersection_points):
            result = split(line, unique_intersection_points)
            for segment in result.geoms:
                new_row = row.copy()
                new_row.geometry = segment
                split_lines_with_attributes.append(new_row)
        else:
            split_lines_with_attributes.append(row)

    # Create a new GeoDataFrame with split lines and original attributes
    split_gdf = gpd.GeoDataFrame(split_lines_with_attributes, columns=gdf.columns, crs=gdf.crs)

    if col_id:
        split_gdf[f"old_{col_id}"] = split_gdf[col_id]
        split_gdf = cm_l.update_double_ids(split_gdf, col_id)

    return split_gdf

def get_geodataframe_center(df):
    x_min, y_min, x_max, y_max = df.total_bounds
    y_center = (y_min + y_max) / 2
    x_center = (x_min + x_max) / 2
    df_center = Point(x_center, y_center)
    return df_center


def get_geodataframe_US_state_and_center(df):
    """
    Purpose: Find the State if the customer is in the USA based on the data
    df: the dataframe

    For Non USA customers state='NA' and state_name='Other'
    """
    if not df.crs:
        raise Exception("Dataframe have no projection information")
    df = df.copy()

    USA_states_shapefile = os.path.join(gb_l.gis_library_dir, "States", "USA_States.shp")
    USA_states = io_l.read_data(USA_states_shapefile)
    if not USA_states.crs:
        raise Exception("States have no projection information")

    # Reproject pipes to USA projection
    if USA_states.crs != df.crs:
        df = df.to_crs(USA_states.crs)
    assert df is not None, "Cannot reproject pipes to USA projection"
    # Get df center
    df_center = get_geodataframe_center(df)

    # Intersect pipes center with states
    USA_states['intersect'] = USA_states.geometry.apply(lambda g: g.intersects(df_center))
    USA_states = USA_states[USA_states['intersect']]
    if len(USA_states) > 0:
        state = USA_states.iloc[0]['STUSPS']
        state_name = USA_states.iloc[0]['NAME']
    else:
        state = "NA"
        state_name = "Other"

    # Reproject pipes to latitude-Longitude
    df = df.to_crs(4326)
    x_min, y_min, x_max, y_max = df.total_bounds
    lat_center = (y_min + y_max) / 2
    lon_center = (x_min + x_max) / 2
    study_center = [lon_center, lat_center]
    print(f"State and center retrieved for pipes: {state_name} {state} {study_center}")
    return state, study_center


def group_azimuths(df, azimuth_column, step=25):
    """
    Group azimuth values into bins of specified step size.

    :param df: DataFrame containing the azimuth values
    :param azimuth_column: Name of the column containing azimuth values
    :param step: Step size for grouping (in degrees)
    :return: DataFrame with grouped azimuths and their percentages
    """
    # Ensure azimuths are within 0-360 range
    df[azimuth_column] = df[azimuth_column] % 360

    # Create bins
    bins = list(range(0, 361, step))
    labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)]

    # Group azimuths
    df['azimuth_group'] = pd.cut(df[azimuth_column], bins=bins, labels=labels, include_lowest=True)

    # Calculate percentages
    group_counts = df['azimuth_group'].value_counts()
    total_count = group_counts.sum()
    percentages = (group_counts / total_count * 100).sort_index()

    # Create result DataFrame
    result_df = pd.DataFrame({
        'Azimuth_Group': percentages.index,
        'Percentage': percentages.values
    })

    return result_df


def plot_azimuth_rose(df, azimuth_column, step=15, show_yticks=False, color_map='viridis',
                      title="Azimuth Distribution Rose Diagram", fig_size=(10, 10), save_folder=""):
    """
    Plot a rose diagram from grouped azimuth data.

    :param df: DataFrame with 'Azimuth_Group' and 'Percentage' columns
    :param fig_size: Tuple specifying figure size
    :param title: Title for the plot
    :param show_yticks: Boolean to show y-tick labels (percentages)
    :param color_map: Colormap to use for the bars (e.g., 'viridis', 'plasma', 'coolwarm')
    """
    df = df.copy()
    if azimuth_column not in df.columns:
        df = calculate_line_azimuth(df, column_name=azimuth_column, norm=False, def_class=False)

    if pd.api.types.is_numeric_dtype(df[azimuth_column]) and (0 <= df[azimuth_column].max() <= 360):
        df = group_azimuths(df, azimuth_column, step)
    else:
        raise Exception(f"Column {azimuth_column} must have values between 0 and 360 degrees!")

    plt.figure(figsize=fig_size)
    ax = plt.subplot(111, projection="polar")

    # Extract start angles by converting 'Azimuth_Group' from label to numeric (e.g., '0-25' -> 0)
    step = int(df.iloc[0]['Azimuth_Group'].split("-")[1])
    angles = df['Azimuth_Group'].apply(lambda x: np.radians(int(x.split("-")[0]) + step/2))
    percentages = df['Percentage'].values

    # Width of each bar
    width = np.radians(360 / len(df))

    # Apply colormap
    cmap = plt.get_cmap(color_map)
    colors = cmap(percentages / max(percentages))  # Normalize the percentages to get colors

    # Plot bars with the chosen colormap
    bars = ax.bar(angles, percentages, width=width, bottom=0.0, alpha=0.8, color=colors)

    # Customize the plot
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.arange(0, 360, 45), labels=['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])

    # Set y-ticks (percentage)
    ax.set_rticks(np.arange(0, max(percentages), 5))

    # Optionally add y-tick labels
    if show_yticks:
        ax.set_yticklabels([f"{x}%" for x in ax.get_yticks()])
    else:
        ax.set_yticklabels('')

    # Add percentage labels to the bars
    for angle, percentage, bar in zip(angles, percentages, bars):
        label_angle = angle
        alignment = 'center'
        ax.text(label_angle, percentage, f"{percentage:.1f}%",
                ha=alignment, va="center", rotation=np.degrees(label_angle - np.pi/2),
                rotation_mode="anchor")

    plt.title(title, y=1.1)
    plt.tight_layout()

    if save_folder == "" and save_folder is not None:
        plt.show()
    else:
        plt.savefig(os.path.join(save_folder, "rose_diagram.png"))
        plt.close()

    return df


def get_coordinate_outliers(df, col_id, plot_graphs, figsize=(18,3), verbose=True, save_folder=None):
    """Finds the coordinates that are outliers, ie they have extreme values.

    Keyword arguments:
    df -- the dataframe to be checked
    col_id -- the column with the id value
    plot_graphs -- a boolean variable for plotting or not the distribution graph of the coordinates
    :return:
    the <df> dataframe with the valid coordinates
    the <geom_errors> dataframe with the error coordinates
    """

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
    min_Q1, max_Q3 = cm_l.get_IQR_outlier_limits(df_check, "coords", [0., 0.8], 2, verbose=verbose, plot=plot_graphs)
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
    """Finds the duplicate geometries."""
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
