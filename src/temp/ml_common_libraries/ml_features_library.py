from import_libraries import *
import global_variables as gb_l
import libraries.common_libraries.geom_library as geom_l
import libraries.common_libraries.generic_library as cm_l
import libraries.common_libraries.fill_nulls_library as fill_l


def calculate_disadvantages_communities(df, state, col_id, gis_library_dir, col_name="dis_com", plot=True, verbose=True):
    """Generates a layer representing disadvantaged communities for a specific customer."""

    dis_com_file = os.path.join(gis_library_dir, "Disadvantaged_Communities", f"{state}_dis_com.gpkg")
    if not os.path.exists(dis_com_file):
        raise Exception(f"The Disadvantaged Community file for the State {state} does not exist!")

    print(f"State: {state} - found disadvantaged communities.")
    dis_com_df = cm_l.read_data(dis_com_file)

    # 1. calculate COF - rename and keep the appropriate columns
    dis_com_df = dis_com_df.rename(columns={'M_CLT_EOMI': 'climate',  # climate Change
                                            'M_ENY_EOMI': 'energy',  # clean Energy and energy efficiency
                                            'M_HLTH_102': 'health',  # health burdens
                                            'M_HSG_EOMI': 'housing',  # affordable and sustainable housing
                                            'M_PLN_EOMI': 'pollution',
                                            # reduction and remediation of legacy pollution
                                            'M_TRN_EOMI': 'transit',  # clean transit
                                            'M_WKFC': 'training',  # training and workforce development
                                            'M_WTR_EOMI': 'water',
                                            # critical clean water and waste infrastructure
                                            'CC': col_name})  # total factors, if >0 disadvantaged

    dis_com_df.fillna(0, inplace=True)
    need_cols = [col_name, 'climate', 'energy', 'health', 'housing', 'pollution', 'transit', 'training', 'water']
    for col in need_cols:
        dis_com_df[col] = dis_com_df[col].astype(int)

    # 2. create a layer to publish
    dis_com_df = dis_com_df[need_cols + ['geometry']].copy()
    dis_com_df = geom_l.preprocess_and_reproject_geometries(dis_com_df, df.crs)
    dis_com_df = geom_l.clip_dataframe(dis_com_df, df.total_bounds, expand=2000)
    dis_com_df[col_name] = dis_com_df[col_name].fillna(0).astype(int)

    item_cols = [col_name]
    for col in item_cols:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    df_updated = geom_l.update_df_with_intersecting_polygons(df[[col_id, 'geometry']], dis_com_df, col_id, item_cols)
    df = pd.merge(df, df_updated[[col_id] + item_cols], how="left", on=col_id)
    for col in item_cols:
        df[col] = df[col].fillna(0)

    if verbose:
        display(df.head())

    if plot:
        # plot the income distribution
        plt.figure(figsize=(15, 5))
        df[col_name].hist()
        plt.show()

    return df


def calculate_income(df, state, col_id, gis_library_dir, col_name="income", plot=True, verbose=True):
    """Calculates the feature <income> based on ArcGIS Online data."""
    df = df.copy()

    source_data_dir = os.path.join(gis_library_dir, "Income")
    source_file = os.path.join(source_data_dir, f"{state}_income.gpkg")
    # check that the file exists
    if not os.path.exists(source_file):
        raise Exception(f"The Income file for the State {state} does not exist!")

    print (f"State: {state}: Found Income")
    income_df = cm_l.read_data(source_file)
    income_df = geom_l.preprocess_and_reproject_geometries(income_df, df.crs)
    income_df = geom_l.clip_dataframe(income_df, df.total_bounds, expand=2000)
    income_df.rename(columns = {'B19049_001E': col_name}, inplace=True)
    income_df[col_name] = income_df[col_name].fillna(0).astype(int)

    item_cols = [col_name]
    for col in item_cols:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    df_updated = geom_l.update_df_with_intersecting_polygons(df[[col_id, 'geometry']], income_df, col_id, item_cols)
    df = pd.merge(df, df_updated[[col_id] + item_cols], how="left", on=col_id)
    for col in item_cols:
        df[col] = df[col].fillna(0)
        df = fill_l.fill_null_values_using_interpolation(df, method="nearest", fill_col_name=col, null_value=0, plot=False)

    if verbose:
        display(df.head())

    if plot:
        # plot the income distribution
        plt.figure(figsize=(15, 5))
        df[col_name].hist()
        plt.show()

    return df


def calculate_house_year_risk(df, state, col_id, gis_library_dir, col_yr_median, col_yr_score, plot=True, verbose=True):
    """ Calculates the <house_year> and <house_yr_median> features based on the house year built values
    per tract from ArcGIS Online data."""
    df = df.copy()

    dict_housing_years = {
        'B25034_002E': 0,  # housing units built 2020 or later
        'B25034_003E': 0,  # housing units built 2010 to 2019
        'B25034_004E': 0,  # housing units built 2000 to 2009
        'B25034_005E': 0,  # housing units built 1990 to 1999
        'B25034_006E': 0.001,  # housing units built 1980 to 1989
        'B25034_007E': 0.01,  # housing units built 1970 to 1979
        'B25034_008E': 0.1,  # housing units built 1960 to 1969
        'B25034_009E': 0.4,  # housing units built 1950 to 1959
        'B25034_010E': 0.7,  # housing units built 1940 to 1949
        'B25034_011E': 1.0}  # housing units built 1939 or earlier

    source_data_dir = os.path.join(gis_library_dir, "ACS_House_Units")
    source_file = os.path.join(source_data_dir, f"{state}_USA_House_Year_Built.gpkg")
    # check that the file exists
    if not os.path.exists(source_file):
        raise Exception(f"The House Year Risk file for the State {state} does not exist!")

    print (f"State: {state}: Found House Year Risk")
    acs_tracts = cm_l.read_data(source_file)
    acs_tracts = geom_l.preprocess_and_reproject_geometries(acs_tracts, df.crs)
    acs_tracts = geom_l.clip_dataframe(acs_tracts, df.total_bounds, expand=2000)

    acs_tracts[col_yr_median] = acs_tracts['B25035_calc_txt001E']
    acs_tracts[col_yr_median] = acs_tracts[col_yr_median].astype(str)
    acs_tracts.loc[acs_tracts[col_yr_median]=="1939_or_Earlier", col_yr_median] = "1939"
    acs_tracts.loc[acs_tracts[col_yr_median].str.upper()=="OTHER", col_yr_median] = 0
    acs_tracts.loc[acs_tracts[col_yr_median].str.upper().isin(gb_l.string_null_values_list), col_yr_median] = 0
    acs_tracts[col_yr_median] = acs_tracts[col_yr_median].astype(int)

    acs_tracts[col_yr_score] = 0
    def find_risk(row):
        house_score = 0.0
        total_houses = 0
        for col in dict_housing_years.keys():
            house_score += row[col] * dict_housing_years[col]
            if dict_housing_years[col] > 0:
                total_houses += row[col]
        return house_score / total_houses if total_houses !=0 else 0
    acs_tracts[col_yr_score] = acs_tracts.apply(lambda row: find_risk(row), axis=1)

    # check if these features already exist in the dataframe and remove them
    item_cols = [col_yr_score, col_yr_median]
    for col in item_cols:
        if col in list(df.columns):
            df.drop(columns=[col], inplace=True)

    df_updated = geom_l.update_df_with_intersecting_polygons(df[[col_id, 'geometry']], acs_tracts, col_id, item_cols)
    df = pd.merge(df, df_updated[[col_id] + item_cols], how="left", on=col_id)
    for col in item_cols:
        df[col] = df[col].fillna(0)
        df = fill_l.fill_null_values_using_interpolation(df, method="nearest", fill_col_name=col, null_value=0, plot=False)

    if verbose:
        display(df.head())

    if plot:
        # plot the acs_house_year distribution
        plt.figure(figsize=(15, 5))
        df[col_yr_score].hist()
        plt.show()

    return df


def convert_df_column_to_percentage(df, column_name, max_value=None):
    """Converts the values of the <column_name> of the dataframe <df> into percentages.
    If the max_value is None then it takes the maximum values of the <column_name>.

    Keyword arguments:
    df -- the dataframe to update
    column_name -- the column name to update
    max_value -- if it equals None then it is replaced my the maximum value of the <column_name>
    :return:
    the updated <df> dataframe
    """
    df = df.copy()
    max_value = np.max(df[column_name]) if max_value is None else max_value
    if max_value != 0:
        df[column_name] = (100. * df[column_name]) / max_value
    else:
        df[column_name] = 0
    df[column_name] = df[column_name].fillna(0)
    return df


def calc_lat_long_coordinates(df, name, geom_col):
    """Calculate Longitude and Latitude for a given <df> and geometry values.

    Keyword arguments:
    df -- the dataframe we want to calculate coordinates
    name -- the name of <df> in order to append this name to new field
    geom_col -- the geometry column name
    :return:
    <df> with 2 extra columns: latitude and longitude
    """
    df[f"{name}_Longitude"] = df[geom_col].map(lambda p: p.centroid.x)
    df[f"{name}_Latitude"] = df[geom_col].map(lambda p: p.centroid.y)
    return df


def combined_plot(gdf, column):
    """
    This function creates a combined plot with the GeoDataFrame plot on the left, the legend plot on the right,
    and the box plot on top and histogram on the bottom.
    """
    fig = plt.figure(figsize=(18, 10))

    # Create a GridSpec with 2 rows and 3 columns
    gs = fig.add_gridspec(2, 3, width_ratios=[2, 0.1, 3])

    # Create the GeoDataFrame plot on the left
    cmap = "coolwarm_r"
    ax1 = fig.add_subplot(gs[:, 0])
    gdf.plot(column=column, cmap=cmap, legend=False, scheme="fisher_jenks",
             edgecolor="#B3B3B3", k=10, linewidth=0.1, ax=ax1)
    ax1.set_title(f"Spatial distribution of {column}")
    ax1.set_aspect("equal")  # Ensure aspect ratio is equal

    # Create the colorbar plot on the right
    ax2 = fig.add_subplot(gs[:, 1])
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array(gdf[column])
    plt.colorbar(sm, cax=ax2)

    # Create the box plot on top
    ax3 = fig.add_subplot(gs[0, 2])
    cm_l.plot_box_plot(gdf, column, ax3)

    # Create the histogram on the bottom
    ax4 = fig.add_subplot(gs[1, 2])
    cm_l.plot_histogram_with_mean_std_lines(gdf, column, ax4)

    plt.tight_layout()
    plt.show()


def create_feature_voronoi(df, col_id, col_target, col_voronoi, max_distance_in_ft=1500, simplify_in_ft=80, plot=True):
    print(f"Calculating '{col_voronoi}' feature...")
    cm_l.check_needed_df_columns(df, [col_id, col_target])
    df = df.copy()

    df_target = df[df[col_target] == 1].copy()
    if not df_target.empty:
        if col_voronoi in list(df.columns):
            df.drop(columns=[col_voronoi], inplace=True)

        df_vor_pols = geom_l.create_voronoi_polygons(df_target, col_id, col_voronoi, df, max_distance_in_ft, simplify_in_ft)
        df_with_vor = geom_l.update_df_with_intersecting_polygons(df, df_vor_pols, col_id, [col_voronoi])
        df_with_vor[col_voronoi] = df_with_vor[col_voronoi].astype(float)
        max_value = df_with_vor[col_voronoi].max()

        df = pd.merge(df, df_with_vor[[col_id, col_voronoi]], how="left", on=col_id)
        df[col_voronoi] = df[col_voronoi].fillna(max_value)
        total_area = int(df_vor_pols.geometry.unary_union.area + 0.5)
        df[col_voronoi] = (100 / total_area) * df[col_voronoi]
        if plot:
            combined_plot(df_vor_pols, col_voronoi)

    else:
        cm_l.print_formatted_txt(f"Cannot calculate {col_voronoi} feature: no {col_target} found.", "ERROR")
    return df


def spatial_join_and_calc_stats(df, df_other, buffer_size, col_id, col_num, col_dist):
    buffer_radius = geom_l.convert_value_in_ft_to_df_units(df, buffer_size)
    df_buffer = df.copy()
    df_buffer['geometry'] = df_buffer['geometry'].buffer(buffer_radius)

    col_left = col_id + "_left"
    col_right = col_id + "_right"
    df_merge = gpd.sjoin(df_buffer[[col_id, 'geometry']], df_other[[col_id, 'geometry']])
    df_merge = df_merge[df_merge[col_left] != df_merge[col_right]][[col_left, col_right]]
    df_merge = pd.merge(df_merge[[col_left, col_right]], df[[col_id, 'geometry']], left_on=col_left, right_on=col_id)
    df_merge = pd.merge(df_merge[[col_left, col_right, 'geometry']], df[[col_id, 'geometry']], left_on=col_right, right_on=col_id)
    if len(df_merge) > 0:
        df_merge['dist'] = df_merge.apply(lambda row: row['geometry_x'].distance(row['geometry_y']), axis=1)
        grouped_df = df_merge.groupby(col_left, as_index=False).agg({col_right: 'size', 'dist': 'mean'})
        grouped_df.rename(columns={col_left: col_id, col_right: col_num, 'dist': col_dist}, inplace=True)
        return grouped_df
    else:
        return None


def create_features_nearby(df, col_id, col_target, col_nearby, col_ratio, col_avrg_distance, search_distance_in_ft=300, features=[], plot=False):
    df = df.copy()

    buffer_radius = round(geom_l.convert_value_in_ft_to_df_units(df, search_distance_in_ft))
    df_target = df[df[col_target] == 1].copy()
    df_no_target = df[df[col_target] == 0].copy()

    grouped_df_target = spatial_join_and_calc_stats(df, df_target, search_distance_in_ft, col_id, col_num="num_target", col_dist="dist_target")
    grouped_df_no_target = spatial_join_and_calc_stats(df, df_no_target, search_distance_in_ft, col_id, col_num="num_no_target", col_dist="dist_no_target")

    if grouped_df_target is not None:
        df = pd.merge(df, grouped_df_target[[col_id, 'num_target', 'dist_target']], how="left", on=col_id)
        df['num_target'] = df['num_target'].fillna(0).astype(int)
        df['dist_target'] = df['dist_target'].fillna(buffer_radius + 1)
    else:
        df['num_target'] = 0
        df['dist_target'] = buffer_radius + 1

    if grouped_df_no_target is not None:
        df = pd.merge(df, grouped_df_no_target[[col_id, 'num_no_target', 'dist_no_target']], how="left", on=col_id)
        df['num_no_target'] = df['num_no_target'].fillna(0).astype(int)
        df['dist_no_target'] = df['dist_no_target'].fillna(buffer_radius + 1)
    else:
        df['num_no_target'] = 0
        df['dist_no_target'] = buffer_radius + 1

    df['total'] = df['num_target'] + df['num_no_target']
    cond = (df['total'] > 0)

    if col_nearby in features:
        print(f"Calculating '{col_nearby}' feature...")
        df[col_nearby] = 0.0
        df.loc[cond, col_nearby] = 100 * (df[cond]['num_target'] - df[cond]['num_no_target']) / df[cond]['total']
    else:
        cm_l.print_formatted_txt(f"Feature {col_nearby} is not created.", "WARNING")

    if col_ratio in features:
        print(f"Calculating '{col_ratio}' feature...")
        df[col_ratio] = 0.0
        df.loc[cond, col_ratio] = (100 * df[cond]['num_target']) / df[cond]['total']
    else:
        cm_l.print_formatted_txt(f"Feature {col_ratio} is not created.", "WARNING")

    if col_avrg_distance in features:
        print(f"Calculating '{col_avrg_distance}' feature...")
        df[col_avrg_distance] = buffer_radius + 1
        df[col_avrg_distance] = df['dist_no_target'] - df['dist_target']
        df[col_avrg_distance] = df[col_avrg_distance].round(1)
    else:
        cm_l.print_formatted_txt(f"Feature {col_avrg_distance} is not created.", "WARNING")

    df.drop(columns=['total', 'num_target', 'num_no_target', 'dist_target', 'dist_no_target'], inplace=True)

    if plot:
        for col in set([col_nearby, col_ratio, col_avrg_distance]).intersection(set(list(df.columns))):
            combined_plot(df, col)
    return df


def create_feature_kernel(df, col_target, col_kernel, cell_size=500, plot=False):
    cm_l.check_needed_df_columns(df, [col_target])
    df_target = df[df[col_target] == 1].copy()

    if len(df_target) <= 2:
        cm_l.print_formatted_txt(f"Feature target_kernel is skipped: not enough data : {len(df_target)}", "ERROR")
    else:
        print(f"Calculating '{col_kernel}' feature...")
        df = df.copy()
        df_target = df_target.copy()
        cm_l.check_needed_df_columns(df, ['geometry'])
        cm_l.check_needed_df_columns(df_target, ['geometry'])

        df[col_kernel] = np.nan

        df_target = calc_lat_long_coordinates(df_target, "target", "geometry")
        df = calc_lat_long_coordinates(df, "r", "geometry")
        df[col_kernel] = kernel_density_combine(df_target, df, cell_size, plot)

        # normalize kernel density to values 0-100
        df = convert_df_column_to_percentage(df, column_name=col_kernel)
        df.drop(columns=['r_Longitude', 'r_Latitude'], inplace=True)
        if plot:
            combined_plot(df, col_kernel)
    return df


def kernel_density_combine(df_target, df, cell_size, plot=False):
    # Create the kernel density plot based on the f_pipe extend
    m1 = df_target['target_Longitude']
    m2 = df_target['target_Latitude']
    m1g = df['r_Longitude']
    m2g = df['r_Latitude']
    grid = df[['r_Longitude', 'r_Latitude']].values.T

    x_min = m1g.min()
    x_max = m1g.max()
    y_min = m2g.min()
    y_max = m2g.max()

    cols_x = round((x_max - x_min) / cell_size)
    cols_y = round((y_max - y_min) / cell_size)
    if cols_x < 200:
        cols_x = 200
    if cols_y < 200:
        cols_y = 200

    X, Y = np.mgrid[x_min:x_max:cols_x, y_min:y_max:cols_y]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([m1, m2])
    kernel = gaussian_kde(values, bw_method="silverman")
    Z = np.reshape(kernel(positions).T, X.shape)
    Z_p = kernel.evaluate(points=grid)

    # if plot:
    #     fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    #     sc = ax.scatter(m1g, m2g, c=Z_p, marker="o", s=2, cmap=plt.cm.gist_earth_r)
    #     ax.set_xlim([x_min, x_max])
    #     ax.set_ylim([y_min, y_max])
    #     # Add a colorbar to the plot
    #     cbar = plt.colorbar(sc, ax=ax)
    #     cbar.set_label("Kernel Density")
    #     plt.show()

    return Z_p


def calculate_target_percentage_per_decade(df, col_year, col_target, col_year_score, ban_year, plot=False):
    print(f"Calculating '{col_year_score}' feature...")
    cm_l.check_needed_df_columns(df, [col_year, col_target])
    df = df.copy()

    df['decade'] = None
    df[col_year_score] = 0.0
    year_decades = ['1500-1900', '1900-1920', '1920-1940', '1940-1960', '1960-ban_year', 'ban_year-today']
    for decade in year_decades:
        decade = decade.replace("ban_year", str(ban_year))
        decade = decade.replace("today", str(datetime.now().year))
        min_year, max_year = decade.split("-")
        min_year, max_year = int(min_year), int(max_year)
        cond_year = (df[col_year]>=min_year) & (df[col_year]<max_year)
        cond_target = (df[col_target]==1)
        target_percentage = 100 * len(df[cond_target & cond_year]) / len(df[cond_year])
        df.loc[cond_year, col_year_score] = target_percentage
        df.loc[cond_year, 'decade'] = decade

    if plot:
        grouped_df = df.groupby("decade").agg({col_year_score: 'mean'}).reset_index()
        display(grouped_df)
        # sns.catplot(data=grouped_df, kind="bar", x="decade", y=col_year_score, palette="pastel", legend=False, height=5)
        sns.catplot(data=grouped_df, kind="bar", hue="decade", y=col_year_score, palette="pastel", legend=False, height=5, aspect=2)
        plt.show()
    df.drop(columns=['decade'], inplace=True)
    return df
