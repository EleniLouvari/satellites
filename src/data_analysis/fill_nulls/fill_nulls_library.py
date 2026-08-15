'''Spatial interpolation helpers for filling GeoDataFrame null values.'''

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from pykrige.ok import OrdinaryKriging
from scipy.interpolate import NearestNDInterpolator
from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist
from sklearn.impute import KNNImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm


class _CommonHelpers:
    @staticmethod
    def check_needed_df_columns(df, columns):
        missing = sorted(set(columns).difference(df.columns))
        if missing:
            raise ValueError(f'Missing required columns: {missing}')


class _GeometryHelpers:
    @staticmethod
    def convert_geometries_to_points(gdf):
        result = gdf.copy()
        result.geometry = result.geometry.representative_point()
        return result

    @staticmethod
    def return_valid_geometries(gdf):
        result = gdf.loc[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
        result.geometry = result.geometry.make_valid()
        return result

    @staticmethod
    def convert_value_in_ft_to_df_units(gdf, value_in_feet):
        if gdf.crs is None:
            raise ValueError('A CRS is required for distance-based interpolation.')
        if gdf.crs.is_geographic:
            raise ValueError('Distance-based interpolation requires a projected CRS.')
        unit_to_metres = gdf.crs.axis_info[0].unit_conversion_factor
        return float(value_in_feet) * 0.3048 / unit_to_metres

    @staticmethod
    def convert_value_in_m_to_df_units(gdf, value_in_metres):
        """Convert metres to the linear units of a projected GeoDataFrame CRS."""
        if gdf.crs is None:
            raise ValueError('A CRS is required for distance-based interpolation.')
        if gdf.crs.is_geographic:
            raise ValueError('Distance-based interpolation requires a projected CRS.')
        unit_to_metres = gdf.crs.axis_info[0].unit_conversion_factor
        return float(value_in_metres) / unit_to_metres


cm_l = _CommonHelpers()
geom_l = _GeometryHelpers()


def fill_nulls_nearest_neighbor(nulls, non_nulls, fill_col_name):
    """
    Fill nulls in the fill_col_name column using Nearest Neighbor Interpolation.

    Parameters:
        nulls (geopandas.GeoDataFrame): DataFrame containing only rows with null values in <fill_col_name>.
        non_nulls (geopandas.GeoDataFrame): DataFrame containing no-nulls values.
        fill_col_name (str): Name of the column we want to fill.

    Returns:
        numpy.ndarray: Filled nulls values.
    """
    # Extract x and y coordinates from non-nulls geometry
    non_nulls_coordinates = non_nulls.geometry.apply(lambda point: (point.x, point.y)).tolist()

    # Extract fill column values from non-nulls
    non_nulls_values = non_nulls[fill_col_name].tolist()

    # Create interpolator
    interpolator = NearestNDInterpolator(non_nulls_coordinates, non_nulls_values)

    # Extract x and y coordinates from nulls geometry
    nulls_coordinates = nulls.geometry.apply(lambda point: (point.x, point.y)).tolist()

    # Interpolate nulls
    filled_nulls = interpolator(nulls_coordinates)

    return filled_nulls


def fill_nulls_inverse_distance_weighting(nulls, non_nulls, fill_col_name, max_distance=None):
    """
    Fill nulls in the fill_col_name column using Inverse Distance Weighting.

    Parameters:
        nulls (geopandas.GeoDataFrame): DataFrame containing only rows with null values in <fill_col_name>.
        non_nulls (geopandas.GeoDataFrame): DataFrame containing no-nulls values.
        fill_col_name (str): Name of the column we want to fill.
        max_distance (float): Maximum search distance for filling nulls. If None, all non-nulls will be considered.

    Returns:
        numpy.ndarray: Filled nulls values.
    """
    # Calculate distances between null values and non-null values
    distances = np.sqrt((nulls.geometry.x.values[:, np.newaxis] - non_nulls.geometry.x.values) ** 2 +
                        (nulls.geometry.y.values[:, np.newaxis] - non_nulls.geometry.y.values) ** 2)

    within_distance = np.ones(distances.shape, dtype=bool)
    if max_distance is not None:
        within_distance = distances <= max_distance

    # Excluded observations receive zero weight. Coincident points are protected
    # against division by zero without turning out-of-range points into donors.
    inverse_distances = np.where(
        within_distance,
        1.0 / np.maximum(distances, 1e-6),
        0.0,
    )
    numerator = np.sum(
        non_nulls[fill_col_name].values[np.newaxis, :] * inverse_distances,
        axis=1,
    )
    denominator = np.sum(inverse_distances, axis=1)
    weighted_values = np.full(len(nulls), np.nan, dtype=float)
    np.divide(
        numerator,
        denominator,
        out=weighted_values,
        where=denominator > 0,
    )

    return weighted_values


def fill_nulls_using_neighborhood_values(df, field_name, allow_zeros=False, method="median", distance_in_feet=500, min_neighbors=5,
                                         expand=True, select_closest=False):
    """Fills in missing data (0 or NaN values) in the given <field_name> with the <method> from the neighborhood.
    if col_material!=None fills in missing data of the same col_material.
    """
    df = df.copy()

    distance = geom_l.convert_value_in_ft_to_df_units(df, distance_in_feet)
    df = geom_l.return_valid_geometries(df)
    df.reset_index(inplace=True, drop=True)
    df['pindex'] = df.index + 1

    condition = df[field_name].isnull()
    if not allow_zeros:
        condition = condition | (df[field_name] == 0)

    non_nulls = df[~condition].copy()
    non_nulls.reset_index(inplace=True, drop=True)
    spatial_index = non_nulls.sindex

    nulls = df[condition].copy()
    len_problem_df = len(nulls.index)
    len_non_nulls = len(non_nulls.index)

    if nulls.empty:
        print(f"No nulls found in the field: {field_name}")
        return df.drop(columns=['pindex'])

    print(f"We have {len(nulls)} missing data in the field: {field_name}")

    for index, row in tqdm(nulls.iterrows(), total=len_problem_df):
        geom = row['geometry']
        pindex = row['pindex']
        try:
            buffer_dist = distance
            local_expand = expand
            new_val = None
            while new_val is None:
                polygon = geom.buffer(buffer_dist)
                possible_matches_index = list(spatial_index.intersection(polygon.bounds))
                possible_matches = non_nulls.iloc[possible_matches_index]
                precise_matches = possible_matches[possible_matches.intersects(polygon)]

                if len(precise_matches.index) == len_non_nulls:
                    local_expand = False

                if len(precise_matches.index) >= min_neighbors:
                    if select_closest:
                        precise_matches = precise_matches.copy()
                        precise_matches['dist'] = precise_matches['geometry'].distance(geom)
                        precise_matches.sort_values(by="dist", inplace=True)
                        precise_matches = precise_matches.head(min_neighbors)

                    if method == "min":
                        new_val = precise_matches[field_name].min()
                    elif method == "max":
                        new_val = precise_matches[field_name].max()
                    elif method == "median":
                        new_val = precise_matches[field_name].median()
                    elif method == "mean":
                        new_val = precise_matches[field_name].mean()
                    elif method == "mode":
                        new_val = precise_matches[field_name].mode()[0]
                    elif method == "sum":
                        new_val = precise_matches[field_name].sum()
                    else:
                        raise Exception(f"Cannot find neighborhood value, unknown method: {method}")

                if pd.isnull(new_val) or (isinstance(new_val, str) and new_val == ""):
                    new_val = None

                if new_val is None:
                    if expand:
                        if not local_expand:
                            new_val = 0
                        else:
                            buffer_dist = 2 * buffer_dist
                    else:
                        new_val = 0
                else:
                    df.loc[(df['pindex'] == pindex), field_name] = new_val.astype(df[field_name].dtype)
        except:
            raise Exception("Error in fill_missing_data_using_neighborhood_values")

    df.drop(columns=['pindex'], inplace=True)
    return df


def fill_missing_values_in_text_fields(df, string_null_values_list, cols_to_skip=None, filling_method=None):
    """Search into <df> columns for missing values and fill them with the given <filling_method>.
    df -- the df to search and fill missing values
    string_null_values_list -- the list containing all possible strings for nan values
    cols_to_skip -- the columns to skip from the text fields, do not fill them
    filling_method -- the method to fill missing values, if none missing values are replaced by "UNKNOWN"
    """
    if cols_to_skip is None:
        cols_to_skip = []
    cols_to_check = [col for col in df.select_dtypes([np.object_]).columns if col not in cols_to_skip]
    print(f"Fields to check: {cols_to_check}")
    for col in cols_to_check:
        # sometimes we have mixed types into one field, so we need to be sure that no conflicts will come up
        df[col] = df[col].astype(str).str.strip().str.upper()
        print(f"We are checking {col} field with type: {df[col].dtype}")
        invalid_values = df[col].isna().sum() + len(df[df[col].isin(string_null_values_list)])
        if invalid_values > 0:
            print(f"Found: {invalid_values}, None, nan or empty strings in column: {col}")
            df[col] = df[col].replace(string_null_values_list, None)
            if filling_method is None:
                df[col] = df[col].fillna("UNKNOWN")
            elif filling_method == "mode":
                most_common_value = df[df[col].notna()][col].mode().iloc[0]
                df[col] = df[col].fillna(most_common_value)
                df[col] = df[col].replace(string_null_values_list, most_common_value)
            elif filling_method=="neighborhood":
                df = fill_nulls_using_neighborhood_values(df, col, inplace=False, method="mode")
            else:
                raise Exception(f"Filling method: {filling_method} is invalid; valid values: None, 'mode', 'neighborhood'.")
    return df


def fill_null_values_using_interpolation(
    gdf,
    method,
    fill_col_name,
    null_value,
    max_distance_in_meters=None,
    variogram_lags=15,
    variogram_lags_max_dist_in_meters=None,
    plot=True,
):
    """
    Fill nulls in the <fill_col_name>. column of a GeoDataFrame with non-nulls values using different methods.

    Parameters:
        gdf (geopandas.GeoDataFrame): Input GeoDataFrame.
        method (str): Interpolation method: 'nearest', 'idw', or 'kriging'.
        fill_col_name (str): Numeric column whose null values will be filled.
        max_distance_in_meters (float): Required IDW search distance in metres.
        variogram_lags_max_dist_in_meters (float): Required kriging variogram
            maximum distance in metres.

    Returns:
        geopandas.GeoDataFrame: GeoDataFrame with nulls filled in the <fill_col_name> column.
    """
    valid_methods = ['nearest', 'idw', 'kriging']
    if not method in valid_methods:
        raise ValueError(
            f"Invalid interpolation method. Only {valid_methods} are acceptable"
        )

    cm_l.check_needed_df_columns(gdf, [fill_col_name, "geometry"])
    if not is_numeric_dtype(gdf[fill_col_name]):
        raise ValueError(f"The {fill_col_name} column is not numeric.")

    # Copy the GeoDataFrame to avoid modifying the original
    filled_gdf = gdf.copy()
    filled_gdf = geom_l.convert_geometries_to_points(filled_gdf)

    # Extract dtype of the fill_col_name column
    original_dtype = filled_gdf[fill_col_name].dtype
    # and convert to float, needed for interpolation
    filled_gdf[fill_col_name] = filled_gdf[fill_col_name].astype(float)

    filter_nulls = (filled_gdf[fill_col_name].isna())
    if not np.isnan(null_value):
        filter_nulls = filter_nulls | (filled_gdf[fill_col_name] == null_value)

    gdf_no_nulls = filled_gdf[~filter_nulls].copy()
    gdf_nulls = filled_gdf[filter_nulls].copy()
    if gdf_nulls.empty:
        print(f"No nulls found in the column: {fill_col_name}!")
        return gdf
    else:
        print(f"We will fill {len(gdf_nulls)} null values in the column: {fill_col_name}!")

    # Fill zeros using different methods
    if method == "nearest":
        filled_nulls = fill_nulls_nearest_neighbor(gdf_nulls, gdf_no_nulls, fill_col_name)
    elif method == "idw":
        if max_distance_in_meters is None:
            raise ValueError("max_distance_in_meters is required for idw interpolation.")
        if not np.isfinite(float(max_distance_in_meters)) or float(max_distance_in_meters) <= 0:
            raise ValueError("max_distance_in_meters must be a positive finite number.")
        max_distance = geom_l.convert_value_in_m_to_df_units(gdf, max_distance_in_meters)
        filled_nulls = fill_nulls_inverse_distance_weighting(gdf_nulls, gdf_no_nulls, fill_col_name, max_distance)
    elif method == "kriging":
        if variogram_lags_max_dist_in_meters is None:
            raise ValueError(
                "variogram_lags_max_dist_in_meters is required for kriging interpolation."
            )
        if (
            not np.isfinite(float(variogram_lags_max_dist_in_meters))
            or float(variogram_lags_max_dist_in_meters) <= 0
        ):
            raise ValueError(
                "variogram_lags_max_dist_in_meters must be a positive finite number."
            )
        max_distance = geom_l.convert_value_in_m_to_df_units(
            gdf,
            variogram_lags_max_dist_in_meters,
        )
        best_model, best_params = calculate_variogram(gdf_no_nulls, attribute_column=fill_col_name, variogram_lags=variogram_lags,
                                                      max_distance=max_distance, plot=plot)
        filled_nulls = fill_nulls_kriging(gdf_nulls, gdf_no_nulls, fill_col_name, variogram_model_name=best_model,
                                          variogram_lags=variogram_lags, plot=plot)

    # Update filled values in the GeoDataFrame
    filled_nulls = filled_nulls.astype(original_dtype)
    filled_gdf.loc[gdf_nulls.index, fill_col_name] = filled_nulls

    if plot:
        #cmap = mcolors.LinearSegmentedColormap.from_list("", ["red", "green"])
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        filled_gdf.plot(column=fill_col_name, ax=ax, cmap="coolwarm_r", legend=True,
                 legend_kwds={'label': fill_col_name, 'orientation': 'horizontal'},
                 markersize=5)
        plt.show()

    return filled_gdf


#------------------------------Kriging
# Custom variogram model functions
def linear_model(lags, slope, nugget):
    return slope * lags + nugget


def power_model(lags, scale, exponent, nugget):
    return scale * np.power(lags, exponent) + nugget


def gaussian_model(lags, sill, range_, nugget):
    return sill * (1.0 - np.exp(-lags**2 / (range_**2))) + nugget


def exponential_model(lags, sill, range_, nugget):
    return sill * (1.0 - np.exp(-lags / range_)) + nugget


def spherical_model(lags, sill, range_, nugget):
    return np.where(lags <= range_,
                    sill * (1.5 * (lags / range_) - 0.5 * (lags / range_)**3) + nugget,
                    sill + nugget)

def compute_variogram(x, y, values, n_lags=15, max_distance=None):
    """
    Compute the experimental variogram.

    Parameters:
    x (np.array): Array of x coordinates.
    y (np.array): Array of y coordinates.
    values (np.array): Array of values at the coordinates.
    n_lags (int): Number of lag bins.
    max_distance (float): Maximum distance to consider for the variogram.

    Returns:
    tuple: Lags and semivariance arrays.
    """
    coordinates = np.column_stack([x, y])
    distances = pdist(coordinates)
    semivariances = pdist(values.reshape(-1, 1), metric="sqeuclidean") / 2.0

    # Filter distances and semivariances based on max_distance
    if max_distance is not None:
        mask = distances <= max_distance
        distances = distances[mask]
        semivariances = semivariances[mask]

    bins = np.linspace(0, np.max(distances), n_lags + 1)
    indices = np.digitize(distances, bins) - 1

    lags = []
    semi = []
    for i in range(n_lags):
        if len(semivariances[indices == i]) > 0:
            lags.append(np.mean(distances[indices == i]))
            semi.append(np.mean(semivariances[indices == i]))

    return np.array(lags), np.array(semi)


def fit_variogram_model(lags, semivariance, model, p0):
    """
    Fit a variogram model to the experimental variogram.

    Parameters:
    lags (np.array): Array of lag distances.
    semivariance (np.array): Array of semivariances.
    model (callable): Variogram model function.
    p0 (list): Initial guess for model parameters.

    Returns:
    tuple: Fitted model parameters and the model score.
    """
    popt, pcov = curve_fit(model, lags, semivariance, p0=p0)
    fitted_model = model(lags, *popt)
    score = np.mean((semivariance - fitted_model) ** 2)
    return popt, score


def calculate_variogram(gdf, attribute_column, variogram_lags, max_distance, plot=True):
    """
    Creates the experimental variogram and fit different models to select the best fitting one.

    Parameters:
    gdf (GeoDataFrame): Input GeoDataFrame.
    attribute_column (str): Column name for the attribute to create the variogram.

    Returns:
    str: Best fitting variogram model name.
    """
    cm_l.check_needed_df_columns(gdf, [attribute_column])
    if not is_numeric_dtype(gdf[attribute_column]):
        raise ValueError(f"The {attribute_column} column is not numeric.")

    gdf = gdf.copy()
    gdf = geom_l.convert_geometries_to_points(gdf)

    gdf = gdf[gdf[attribute_column].notna()].copy()
    x = gdf.geometry.x.values
    y = gdf.geometry.y.values
    values = gdf[attribute_column].values

    lags, semivariance = compute_variogram(x, y, values, variogram_lags, max_distance)

    models = {
        'linear': (linear_model, [1.0, 0.1]),
        'power': (power_model, [1.0, 1.0, 0.1]),
        'gaussian': (gaussian_model, [1.0, 1.0, 0.1]),
        'exponential': (exponential_model, [1.0, 1.0, 0.1]),
        'spherical': (spherical_model, [1.0, 1.0, 0.1])
    }

    best_model = None
    best_score = np.inf
    best_params = None
    plt.figure(figsize=(10, 6))
    plt.plot(lags, semivariance, "o", label="Experimental Variogram")

    for name, (model, p0) in models.items():
        try:
            params, score = fit_variogram_model(lags, semivariance, model, p0)
            fitted_model = model(lags, *params)
            if plot:
                plt.plot(lags, fitted_model, label=f"{name} model (score: {score:.4f})")

            if score < best_score:
                best_score = score
                best_model = name
                best_params = params
        except RuntimeError as e:
            print(f"Model {name} failed to converge: {e}")

    if plot:
        plt.xlabel("Lag")
        plt.ylabel("Semivariance")
        plt.legend()
        plt.title("Variogram Model Fitting")
        plt.show()

    print(f"Best fitting model: {best_model} with parameters {best_params}")
    return best_model, best_params


def fill_nulls_kriging(gdf_nulls, gdf_no_nulls, fill_col_name, variogram_model_name, variogram_lags=15, plot=True):
    """
    Perform Kriging interpolation on a specified column of a GeoDataFrame using the given variogram model.

    Parameters:
    gdf (GeoDataFrame): Input GeoDataFrame.
    attribute_column (str): Column name to be used for interpolation.
    variogram_model_name (str): Name of the variogram model to be used.

    Returns:
    GeoDataFrame: GeoDataFrame with null values filled based on the Kriging interpolation model.
    """

    # Extract the non-null and null values
    non_nulls = gdf_no_nulls.copy()
    nulls = gdf_nulls.copy()

    # Extract coordinates and values
    x = non_nulls.geometry.x
    y = non_nulls.geometry.y
    values = non_nulls[fill_col_name]

    # Variogram models mapping
    variogram_models = {
        'linear': linear_variogram_model,
        'power': power_variogram_model,
        'gaussian': gaussian_variogram_model,
        'exponential': exponential_variogram_model,
        'spherical': spherical_variogram_model
    }

    # Check if the variogram model name is valid
    if variogram_model_name not in variogram_models:
        raise ValueError(f"Invalid variogram model name: {variogram_model_name}. Must be one of {list(variogram_models.keys())}.")

    # Create the Kriging model
    OK = OrdinaryKriging(
        x, y, values,
        variogram_model=variogram_model_name,
        verbose=False,
        enable_plotting=False
    )

    # Perform the interpolation for the null values
    z_pred, ss = OK.execute("points", nulls.geometry.x, nulls.geometry.y)

    return z_pred


def fill_numerical_with_KNN(df, fill_col_name, features_for_knn, n_neighbors=3):
    # Ensure specified columns are in the DataFrame
    cm_l.check_needed_df_columns(df, [fill_col_name] + features_for_knn)

    # Make a copy of the DataFrame to avoid modifying the original
    df_filled = df.copy()

    # Select only the required columns for imputation
    columns_for_imputation = [fill_col_name] + features_for_knn
    imputation_df = df_filled[columns_for_imputation]

    # Initialize KNNImputer
    imputer = KNNImputer(n_neighbors=n_neighbors, weights="distance")

    # Apply KNN imputer
    imputed_array = imputer.fit_transform(imputation_df)

    # Update the DataFrame with imputed values for the target column
    df_filled[fill_col_name] = imputed_array[:, 0]  # The first column corresponds to fill_col_name

    return df_filled


def fill_categorical_with_KNN(df, fill_col_name, features_for_knn, n_neighbors=3):
    # Ensure specified columns are in the DataFrame
    cm_l.check_needed_df_columns(df, [fill_col_name] + features_for_knn)

    # Make a copy of the DataFrame to avoid modifying the original
    df_filled = df.copy()

    # Get rows with and without missing values for the current column
    missing_rows = df_filled[df_filled[fill_col_name].isnull()][features_for_knn].copy()
    non_missing_rows = df_filled[features_for_knn].dropna(subset=[fill_col_name]).copy()

    if not missing_rows.empty:
        # Encode categorical values as numbers
        le = LabelEncoder()
        non_missing_rows[fill_col_name] = le.fit_transform(non_missing_rows[fill_col_name])

        # Select features (all other columns specified in features_for_knn)
        X_train = non_missing_rows[features_for_knn]
        y_train = non_missing_rows[fill_col_name]

        # Train KNN classifier
        knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance")
        knn.fit(X_train, y_train)

        # Prepare the rows with missing values for prediction
        X_test = missing_rows[features_for_knn]

        # Predict and fill the missing values
        y_pred = knn.predict(X_test)
        df_filled.loc[missing_rows.index, fill_col_name] = le.inverse_transform(y_pred)

    return df_filled
