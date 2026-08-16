from import_libraries import *
import global_variables as gb_l
import libraries.common_libraries.generic_library as cm_l
import libraries.common_libraries.geom_library as geom_l


def string_similarity(str1, str2):
    """Calculate the similarity between two strings; for example two addresses."""
    str1 = str(str1).strip().upper()
    str2 = str(str2).strip().upper()
    if str1 == "" or str2 == "":
        return 0
    elif str1 == str2:
        return 1
    elif str1 in str2 or str2 in str1:
        return 0.9
    words1 = set(re.split("[, ]", str1))
    words2 = set(re.split("[, ]", str2))
    if not words1 or not words2:
        return 0
    match_count = len(words1.intersection(words2))
    similarity = match_count / min(len(words1), len(words2))
    return similarity


def text_contains_numeric(text):
    return any(char.isnumeric() for char in text)


def get_street_name_and_num_from_address(txt_address):
    """Splits the input address into the street name and street number."""
    txt = str(txt_address).strip().upper()
    if not txt or txt == "0" or " BOX" in txt:
        return pd.Series(["0", ""])

    values = txt.split(" ")
    values = [str(x).strip() for x in values]
    street_num = values[0] if text_contains_numeric(values[0]) else "0"
    street_name = " ".join(values[1:]) if street_num != "0" else txt
    return pd.Series([street_name, street_num])


def standardise_street_name(df, street_name_field):
    """Standardise the street name based on the road dictionary."""
    df = df.copy()
    df[street_name_field] = df[street_name_field].fillna("").astype(str).str.strip().str.upper()
    # split the street name in words
    df['list_address'] = df[street_name_field].str.split(",| ", expand=False)
    df['list_address'] = df['list_address'].apply(lambda x: [y.strip() for y in x])
    # standardize the street name based on the road dictionary
    for key, value in geom_l.road_dict.items():
        df['list_address'] = df['list_address'].apply(lambda x: [value if y == key else y for y in x])
    df[street_name_field] = df['list_address'].apply(lambda x: " ".join(x))
    df.drop(columns=['list_address'], inplace=True)
    return df


def preprocess_address_and_check_invalid(df, address_field, street_name_field, street_num_field,
                                         string_null_values_list):
    """Preprocess the address field, splitting it into street name and street number and finds the ones
    with missing street name or street number."""
    df = df.copy()

    df[street_num_field], df[street_name_field] = "0", ""
    df[address_field] = df[address_field].fillna("").astype(str).str.strip().str.upper()
    df[[street_name_field, street_num_field]] = df[address_field].apply(
        lambda x: get_street_name_and_num_from_address(x))
    df = standardise_street_name(df, street_name_field)
    df.loc[df[street_name_field].isin(string_null_values_list), street_name_field] = ""
    df.loc[df[street_num_field].isin(string_null_values_list), street_name_field] = "0"
    df_issues = df[(df[street_name_field] == "") | (df[street_num_field] == "0")].copy()
    if not df_issues.empty:
        print(f"Invalid address: {len(df_issues)} ({round(100 * len(df_issues) / len(df), 2)}%)")
        display(df_issues.head())
    else:
        print("There aren't any invalid addresses")
    return df, df_issues


# TODO: check if this function is needed for leak analysis
def get_conflicts_address_geometries(df, public_data_dir, col_id, street_name_field, street_num_field, city_field,
                                     zip_field, max_distance_in_ft):
    """Finds the conflicts between the geometry derived from geocoding the address and the actual geometry.

    Keyword arguments:
    df -- the input dataFrame
    public_data_dir -- the public data directory, where the files used by geocoding are stored
    col_id -- the column id
    city_field -- the column with the city
    zip_field -- the column with the zip
    max_distance_in_ft -- the maximum distance in feet between the geocoded geometry and the actual geometry

    :return:
    df_issues -- the dataFrame containing the pipes with distance > max_distance_in_ft between the geocoded geometry
    and the actual geometry
    """
    cm_l.check_needed_df_columns(df, [street_name_field, street_num_field, city_field, zip_field, 'geometry'])
    df = df.copy()

    # create a geometry column by geocoding the addresses and calculate the distance between the geocoded geometry and
    # the actual geometry
    max_distance = geom_l.convert_value_in_ft_to_df_units(df, max_distance_in_ft)
    cond_valid_address = (df[street_name_field] != "") & (df[street_num_field] != "0")
    df_geocode = df[cond_valid_address].copy()
    df_geocode.drop(columns=['geometry'], inplace=True)
    if zip_field is not None and zip_field not in df_geocode.columns:
        df_geocode[zip_field] = None
    df_geocode, df_errors = geom_l.geocode_failures_and_project_on_road(df, df_geocode, public_data_dir,
                                                                        col_address=street_name_field,
                                                                        col_number=street_num_field,
                                                                        col_zip=zip_field, col_city=city_field,
                                                                        project_on_road=False)
    df_geocode = pd.merge(df, df_geocode[[col_id, 'geocode_score', 'geometry']], on=col_id, how="left")
    df_geocode['geom_distance'] = df_geocode.apply(lambda x: x['geometry_x'].distance(x['geometry_y']), axis=1)
    df_issues = df_geocode[(df_geocode['geom_distance'] > max_distance)].copy()
    if not df_issues.empty:
        print(f"Pipes with distance between the geocoded geometry from addresses and the actual geometry > "
              f"{max_distance_in_ft}: {len(df_issues)} ({round(100 * len(df_issues) / len(df), 2)}%)")
        display(df_issues.head())
    else:
        print(f"All geometries geocoded from addresses have distance < {max_distance_in_ft}"
              f" from the actual geometries")
    return df_issues


def get_mismatched_addresses(df, col_id, street_name_field, street_num_field, city_field, zip_field, state_field,
                             country_field, string_null_values_list):
    """Finds the addresses based on the geometry and checks if these addresses match the actual addresses.

    Keyword arguments:
    df -- the input dataFrame
    col_id -- the column id
    city_field -- the column with the city
    zip_field -- the column with the zip
    state_field -- the column with the state
    country_field -- the column with the country

    :return:
    df_issues -- the dataFrame containing the mismatched addresses
    """
    bs_l.check_needed_df_columns(df, [street_name_field, street_num_field, 'geometry'])

    # select only the records with valid street numbers and street names
    df_address = df[(df[street_num_field] != "0") & (df[street_name_field] != "")].copy()

    # create a geocoded address column based on the geometry
    df_address = geom_l.reverse_geocode_address_arcgis(df_address, "geocode_address")
    # create and format the geo_street_num and geo_street_name columns based on the geocoded address
    df_address['geo_street_num'], df_address['geo_street_name'] = "0", ""
    df_address['geocode_address'] = df_address['geocode_address'].fillna("").str.strip().str.upper()
    df_address[['geo_street_name', 'geo_street_num']] = df_address['geocode_address'].apply(
        get_street_name_and_num_from_address)
    df_address.loc[df_address['geo_street_name'].isin(string_null_values_list), 'geo_street_name'] = ""
    df_address.loc[df_address['geo_street_num'].isin(string_null_values_list), 'geo_street_num'] = "0"
    df_address = standardise_street_name(df_address, 'geo_street_name')
    # format the geo_street_name column to match the actual street name
    for col in [city_field, zip_field, state_field, country_field]:
        if col in df_address.columns:
            cond_no_nulls = (df_address[col].notna()) & (df_address[col] != "")
            df_address.loc[cond_no_nulls, 'geo_street_name'] = df_address[cond_no_nulls].apply(
                lambda x: x['geo_street_name'].upper().replace(f" {x[col].upper()}", ""), axis=1)
    # get the geocoded street_name and street_num
    df_address = pd.merge(df, df_address[[col_id, 'geo_street_name', 'geo_street_num']], on=col_id, how="left")
    # from the street numbers get only the integer part
    for column in [street_num_field, 'geo_street_num']:
        df_address[column] = df_address[column].fillna("0").str.split("-").str[0].str.extract(r"(\d+)", expand=False)
        df_address[column] = pd.to_numeric(df_address[column], errors="coerce").fillna(0).astype(int)

    # 1. first match the addresses based on a similarity index
    df_address['similarity'] = df_address.apply(
        lambda row: string_similarity(row[street_name_field], row['geo_street_name']), axis=1)
    df_address['valid_street_name'] = True
    df_address.loc[(df_address['geo_street_name'] == ""), 'valid_street_name'] = False
    # 2. then check if the actual street number is close to the geocoded street number (+-10 numbers)
    df_address['valid_street_num'] = False
    cond_valid_street_num = (df_address[street_num_field] >= (df_address['geo_street_num'] - 10)) & \
                            (df_address[street_num_field] <= (df_address['geo_street_num'] + 10))
    df_address.loc[cond_valid_street_num, 'valid_street_num'] = True
    # 3. select the pipes with no matching address or no matching street number
    df_issues = df_address[((~df_address['valid_street_name']) | (df_address['similarity'] < 0.5) |
                            (~df_address['valid_street_num'])) & (df_address['geo_street_num'] > 0)].copy()
    df_issues.drop(columns=['valid_street_num', 'valid_street_num'])
    if not df_issues.empty:
        print(f"Pipes with geocoded address not matching the actual address: "
              f"{len(df_issues)} ({round(100 * len(df_issues) / len(df), 2)}%)")
        display(df_issues.head())
    else:
        print("All geocoded addresses match the actual addresses")
    return df_issues


def get_active_state_conflicts(public, private, joined_df, pipe_id_field, active_field, join_public_field, join_private_field):
    """Finds the conflicts between the active/inactive state of the public and private pipes."""
    public = public.copy()
    private = private.copy()
    public = public.rename(columns={pipe_id_field: 'pb_id', active_field: 'pb_active'})
    private = private.rename(columns={pipe_id_field: 'pr_id', active_field: 'pr_active'})
    joined_df = joined_df.rename(columns={join_public_field: 'pb_id', join_private_field: 'pr_id'})
    df_merge = pd.merge(private[['pr_id', 'pr_active']], joined_df, on="pr_id")
    df_merge = pd.merge(df_merge, public[['pb_id', 'pb_active']], on="pb_id")
    df_merge[['pb_active', 'pr_active']] = df_merge[['pb_active', 'pr_active']].fillna("UNKNOWN").astype(bool)
    df_issues = df_merge[((df_merge['pb_active']) & (~df_merge['pr_active'])) |
                         ((~df_merge['pb_active']) & (df_merge['pr_active']))].copy()
    if not df_issues.empty:
        print(f"There are {len(df_issues)} public-private connections with incompatible active/inactive state")
        display(df_issues.head())
    else:
        print("There aren't any public-private connections with incompatible active/inactive state")
    return df_issues


def get_install_yr_conflicts(active_df, abandoned_df, pipe_id_field, install_yr_field, abandon_yr_field,
                             material_field):
    """Finds the conflicts between the installation year of the in-place pipes and the installation/abandon year of the
    abandoned pipes."""
    cm_l.check_needed_df_columns(active_df, [install_yr_field, material_field])
    cm_l.check_needed_df_columns(abandoned_df, [install_yr_field, material_field, abandon_yr_field])
    active_df = active_df.copy()
    abandoned_df = abandoned_df.copy()
    cols = [pipe_id_field, install_yr_field, material_field]
    df_merge = pd.merge(active_df[cols], abandoned_df[cols + [abandon_yr_field]], on=pipe_id_field,
                        suffixes=("_act", "_abd"))
    df_issues1 = df_merge[(df_merge[f"{install_yr_field}_act"] <= df_merge[f"{install_yr_field}_abd"]) &
                          (df_merge[f"{install_yr_field}_act"] > 0)].copy()
    df_issues1['issue'] = "0 < install_yr_act <= install_yr_abd"
    df_issues2 = df_merge[(df_merge[f"{install_yr_field}_act"] < df_merge[abandon_yr_field]) &
                          (df_merge[f"{install_yr_field}_act"] > 0)].copy()
    df_issues2['issue'] = "0 < install_yr_act < abandon_yr"
    df_issues = pd.concat([df_issues1, df_issues2], ignore_index=True)

    if not df_issues.empty:
        print(
            f"Pipes with install year conflicts: {len(df_issues)} ({round(100 * len(df_issues) / len(active_df), 2)}%)"
            f" between the in-place and the abandoned pipes")
        display(df_issues.head())
    else:
        print("There aren't any install year conflicts between the in-place and the abandoned pipes")
    return df_issues


def get_nearby_geometries(df, pipe_id_field, address_field, geometry_field, max_distance_in_ft):
    """Finds the nearby geometries, within the distance of max_distance_in_ft, excluding the duplicates, with
    different addresses."""
    df = df.copy()
    max_distance = geom_l.convert_value_in_ft_to_df_units(df, max_distance_in_ft)
    df_double = df[(df.duplicated(subset=geometry_field, keep=False))].copy()
    df_to_check = df[~df[pipe_id_field].isin(list(df_double[pipe_id_field]))].copy()
    df_to_check[geometry_field] = df_to_check[geometry_field].buffer(max_distance)
    df_issues = gpd.sjoin(df_to_check, df, predicate="contains")
    df_issues = df_issues[(df_issues[f"{pipe_id_field}_left"] != df_issues[f"{pipe_id_field}_right"])].copy()
    df_issues = df_issues[(df_issues[f"{address_field}_left"] != df_issues[f"{address_field}_right"])].copy()
    if not df_issues.empty:
        print(f"Nearby geometries: {len(df_issues)} with distances less than {max_distance_in_ft} ft and "
              f"different {address_field}")
        display(df_issues[[f"{pipe_id_field}_left", f"{address_field}_left",
                           f"{pipe_id_field}_right", f"{address_field}_right"]].head())
    else:
        print("There aren't any nearby geometries wih different addresses")
    return df_issues


def geocode_pipes_with_invalid_geometry(df, df_invalid_geometries, public_dir, geometry_field):
    """Geocodes the pipes with invalid geometry based on the address fields."""
    df_invalid_geometries = df_invalid_geometries.copy()
    if {'street_name', 'street_num', 'city', 'zip'}.issubset(df_invalid_geometries.columns):
        df_invalid_geometries.drop(columns=geometry_field, inplace=True)
        df_invalid_geometries, errors = geom_l.geocode_failures_and_project_on_road(df, df_invalid_geometries,
                                                                                    public_dir, "street_name",
                                                                                    "street_num", "zip",
                                                                                    "city", project_on_road=False)
        df_invalid_geometries.drop(columns=['latitude', 'longitude'], inplace=True)
        df_invalid_geometries = df_invalid_geometries.to_crs(df.crs)
        df_invalid_geometries['x'] = df_invalid_geometries[geometry_field].x
        df_invalid_geometries['y'] = df_invalid_geometries[geometry_field].y
    else:
        print(f"The records with invalid geometries do not have the needed address fields:'street_name',"
              f"'street_num', 'city', 'zip' in order to geocode them!")
    return df_invalid_geometries


def preprocess_geometries(df, df_initial_doubles, field_double, label, crs, null_values, field_mapping, public_dir):
    """Converts multi-geometries to single and handles the projections.

    Keyword arguments:
    df -- the dataframe to check the geometries
    df_initial_doubles -- a dataframe with the initial duplicates based on the column <field_double>. This is needed
                          in order to calculate the new duplicates created after the exploding of the geometries.
    field_double -- the column name with the id, used to find the duplicates (i.e. pipe_id)
    label -- a printing text
    crs -- the projection of the default crs to be used in reprojecting the <df>
    null_values -- the list with possible null values
    field_mapping -- the dictionary with the field mapping
    public_dir -- the public directory
    :return:
    the df with the exploded multi-geometries to single-geometries
    the df_invalid_geometries with the invalid geometries
    the new df_doubles created after the conversion of multi-geometries to single-geometries
    """
    df = df.copy()
    try:
        geom_l.get_unit_of_length(df)
    except:
        cm_l.print_formatted_txt("The projection unit is not valid; should be 'ft' or 'm'", "ERROR")

    len_before = len(df)
    df_invalid_geometries = geom_l.return_invalid_geometries(df, geom_column=field_mapping['geometry_field'])
    # if there are invalid geometries, try to geocode them based on the address fields
    if not df_invalid_geometries.empty:
        print(f"There are {len(df_invalid_geometries)} {label} with invalid geometries")
        display(df_invalid_geometries.head())
        df_invalid_geometries = geocode_pipes_with_invalid_geometry(df, df_invalid_geometries, public_dir,
                                                                    field_mapping['geometry_field'])

    df = geom_l.preprocess_and_reproject_geometries(df, crs)
    # in case of pipes, merge the continuous pipes with same pipe_id, material and diameter
    # then recalculate the duplicate pipe_ids
    if "line" in df.iloc[0].geometry.geom_type.lower() and \
            {field_mapping['pipe_id_field'], field_mapping['install_yr_field'], field_mapping['material_field'],
             field_mapping['diameter_field']}.issubset(df.columns):
        cond_invalid_id = (df[field_double].str.upper().isin(null_values)) | \
                          (df[field_double].str.contains("UNKNOWN", na=False))
        df_with_no_pipe_id, df_with_pipe_id = df[cond_invalid_id].copy(), df[~cond_invalid_id].copy()

        df_with_pipe_id, merged = geom_l.merge_duplicate_pipeid(df_with_pipe_id, field_mapping['pipe_id_field'],
                                                                field_mapping['diameter_field'],
                                                                field_mapping['install_yr_field'],
                                                                field_mapping['material_field'], max_dif_az=30)
        df_with_pipe_id, df_issue = drop_duplicate_pipes_on_multiple_columns(df_with_pipe_id,
                                                                             f"{label} after exploding",
                                                                             field_mapping)
        df_doubles = df_with_pipe_id[(df_with_pipe_id.duplicated(subset=field_double, keep=False))]
        df = pd.concat([df_with_pipe_id, df_with_no_pipe_id], ignore_index=True)
    else:
        df_doubles = df[(df.duplicated(subset=field_double, keep=False)) &
                        (~df[field_double].str.upper().isin(null_values))]

    # these are the new duplicate pipe_ids after exploding
    if not df_initial_doubles.empty:
        df_doubles = df_doubles[~df_doubles['auto_id'].isin(list(df_initial_doubles['auto_id']))]

    if not df_doubles.empty:
        print(f"\nThe number of {label} changed after exploding are: {(len_before - len(df))} records")
        display(df_doubles)
    return df, df_invalid_geometries, df_doubles


def drop_duplicate_pipes_on_multiple_columns(df, label, field_mapping):
    """Checks and drops the pipes with duplicate records based on multiple columns."""
    df = df.copy()
    dupl_cols = [field_mapping['pipe_id_field'], field_mapping['install_yr_field'], field_mapping['material_field'],
                 field_mapping['diameter_field'], field_mapping['geometry_field']]

    all_doubles = []
    len_before = len(df)
    df_doubles = df[(df.duplicated(subset=dupl_cols, keep=False))]
    to_drop = df_doubles.drop_duplicates(subset=dupl_cols, keep="first")
    df = df[~df['auto_id'].isin(list(to_drop['auto_id']))]
    print(f"{label}: drop duplicate records on {dupl_cols}: {(len_before - len(df))}")
    if not df_doubles.empty:
        df_doubles['description'] = f"doubles based on the columns: {','.join(map(str, df_doubles))}"
        df_doubles = df_doubles.sort_values(by="pipe_id")
        display(df_doubles.head())
        all_doubles.append(df_doubles)

    if len(all_doubles) > 0:
        all_doubles = pd.concat(all_doubles, ignore_index=True)
    else:
        all_doubles = pd.DataFrame()

    return df, all_doubles


def get_duplicate_pipe_ids(df, df_nulls, df_doubles, col_id):
    """Finds the duplicate pipe_ids."""
    df = df.copy()
    cond = (df.duplicated(subset=col_id, keep=False))
    if not df_nulls.empty:
        cond = cond & (~df['auto_id'].isin(list(df_nulls['auto_id'])))
    if not df_doubles.empty:
        cond = cond & (~df['auto_id'].isin(list(df_doubles['auto_id'])))
    df_issues = df[cond].sort_values(by=col_id)
    if not df_issues.empty:
        print(f"Duplicate pipe_id: {len(df_issues)}")
        display(df_issues.head())
    else:
        print("There aren't any duplicate pipe_ids")
    return df_issues


def get_public_private_connections(df_public, df_private, col_id, search_distance_in_ft):
    """Finds the public and private connected pipes based on the distance and :
    1. finds the pipes that are not connected with any other pipe, and
    2. finds the pipes connected with more than one pipe
    """
    df_public = df_public.copy()
    df_private = df_private.copy()
    col_id_public = f"{col_id}_public"
    col_id_private = f"{col_id}_private"
    df_public = df_public.rename(columns={col_id: col_id_public})
    df_private = df_private.rename(columns={col_id: col_id_private})

    dict_issues = {}
    for system in ['public', 'private']:
        print("\n" + "-" * 100)
        if system == "public":
            other_system = "private"
            df1, df2 = df_public, df_private
            col_id1, col_id2 = col_id_public, col_id_private
        else:
            other_system = "public"
            df1, df2 = df_private, df_public
            col_id1, col_id2 = col_id_private, col_id_public

        df_join = gpd.sjoin_nearest(df1[[col_id1, 'geometry']], df2[[col_id2, 'geometry']], how="left",
                                    max_distance=search_distance_in_ft)
        df_join[col_id2].fillna("UNKNOWN", inplace=True)
        df_unknown_connections = df_join[df_join[col_id2] == "UNKNOWN"].copy()
        df_join_grouped = df_join[df_join[col_id2] != "UNKNOWN"].copy()
        df_join_grouped = df_join_grouped.groupby(col_id1)[col_id2].apply(list).reset_index(name="connected_ids")
        df_join_grouped['total_connected_ids'] = df_join_grouped['connected_ids'].apply(lambda x: len(x))
        df_multiple_connections = df_join_grouped[df_join_grouped['total_connected_ids'] > 1].copy()
        df_multiple_connections.sort_values(by="total_connected_ids", ascending=False, inplace=True)
        dict_issues[system] = {'multiple_connections': df_multiple_connections,
                               'unknown_connections': df_unknown_connections}
        if not df_multiple_connections.empty:
            print(
                f"There are {system} pipes connected with more than one {other_system} pipes: {len(df_multiple_connections)}")
            display(df_multiple_connections.head())
        else:
            print(f"All {system} pipes are connected with one pipe")

        if not df_unknown_connections.empty:
            print(f"There are {system} pipes not connected with any {other_system} pipe")
            display(df_unknown_connections.head())
        else:
            print(f"All {system} pipes are connected with a {other_system} pipe")
    return dict_issues
