"""Dataframe transformation, normalization, and categorical helpers."""

import re
from copy import deepcopy

import numpy as np
import pandas as pd
from IPython.display import display
from pandas.api.types import is_bool_dtype, is_numeric_dtype, is_object_dtype, is_string_dtype
from pandas.api.types import is_datetime64_any_dtype as is_datetime_dtype
from tqdm.notebook import tqdm

import shared.geometry as geom_l
from shared.assertions import expect_true

from .core import print_formatted_txt

try:
    from skimpy import skim
except ModuleNotFoundError:  # pragma: no cover - optional dependency for dataframe summaries
    skim = None


def fix_json_chars_in_column_names(df):
    """Fix json chars in column names."""
    col_names = set(df.columns)
    df = df.rename(columns=lambda x: re.sub("[^A-Za-z0-9_]+", "_", x))
    col_names_fixed = set(df.columns)
    if col_names != col_names_fixed:
        print(f"We removed JSON characters:\nBefore: {col_names}\nAfter: {col_names_fixed}")
    return df


def fix_json_chars_in_column_values(df, col_to_exclude):
    """Fix json chars in column values."""
    cols = [col for col, dt in df.dtypes.items() if dt == object]
    cols = [col for col in cols if col not in col_to_exclude]

    for col in cols:
        if isinstance(df[df[col].notna()].iloc[0][col], str):
            print(f"Fixing json character values in column: {col}")
            df[col] = df[col].apply(lambda x: re.sub("[^A-Za-z0-9_]+", "_", x))
    return df


def convert_datetime_to_string(df):
    """Coverts datetime column to string of a given dataframe.

    Keyword Arguments:
    df -- dataframe to convert its datetime columns
    :return: the converted dataframe

    """
    df_copy = df.copy()
    for col in df_copy.columns:
        if pd.api.types.is_datetime64_any_dtype(df_copy[col]):
            df_copy[col] = df_copy[col].astype(str)
    return df_copy


def convert_datetime_to_object(df):
    """Convert datetime to object."""
    for col in df.columns:
        if "datetime64" in str(df[col].dtype):
            df[col] = df[col].astype(object)
    return df


def standardize_fields(df, field_mapping, reverse=False):
    """Standardize fields."""
    df = df.copy()
    field_mapping = deepcopy(field_mapping)
    if reverse:
        field_mapping = {value: key for key, value in field_mapping.items()}
        print(field_mapping)
    df.rename(index=str, columns=field_mapping, inplace=True)
    # drop other fields
    fields_to_keep = field_mapping.values()
    df = df[fields_to_keep]
    df.reset_index()
    expect_true(len(set(df.columns)) == len(list(df.columns)), "Duplicate column names")

    return df


def calculate_length_and_convert_to_ft(df, col_length="pipe_len"):
    """Calculate length and convert to ft."""
    df = df.copy()
    df[col_length] = df["geometry"].length
    df_current_unit = geom_l.get_unit_of_length(df)
    if df_current_unit != "ft":
        conversion_factor = geom_l.create_unit_conversion_factor(source_unit=df_current_unit, target_unit="ft")
        if conversion_factor != 1:
            df[col_length] = df[col_length] * conversion_factor
    return df


def convert_col_id_to_str(df, field_name="pipe_id"):
    """Convert pipe_id type to string.

    Keyword Arguments:
    df -- the dataframe we want to convert pipe_id to str type
    field_name -- the field name of id

    """
    # this function is used by data checks - auto leak
    expect_true(field_name in df.columns, "We cannot find the field")

    # handle int and float cases
    if is_numeric_dtype(df[field_name]):
        df[field_name] = df[field_name].fillna(-1).astype(int).astype(str)
        df.loc[df[field_name] == "-1", field_name] = None


def convert_col_id_to_str_helper(*dfs):
    """Convert col id to str helper."""
    for df in dfs:
        convert_col_id_to_str(df)


def clean_col_id_helper(*dfs):
    """Clean col id helper."""
    for df in dfs:
        clean_col_id(df)


def clean_col_id(df, col_id):
    """Clean the col_id field from whitespaces (beginning and end of string)."""
    expect_true(col_id in df.columns, f"Field {col_id} is missing")
    df[col_id] = df[col_id].apply(lambda x: str(x).strip())


def get_column_types(df, exclude_geometry_column=True, raise_error=False):
    """Get column types."""
    df = df.copy()
    if exclude_geometry_column and "geometry" in list(df.columns):
        df.drop(columns=["geometry"], inplace=True)

    object_cols = list(df.select_dtypes(include=["object"]).columns)
    numerical_cols = list(df.select_dtypes(include=["number"], exclude=["object", "bool"]).columns)
    # Handle potential mixed data types within categorical columns
    for col in object_cols:
        if pd.api.types.is_string_dtype(df[col]):
            # Safe conversion to string for consistent encoding
            df[col] = df[col].astype(str)
        else:
            txt_error = f"Column '{col}' contains mixed data types."
            if raise_error:
                # Raise an informative error if the column contains mixed data types
                raise ValueError(txt_error)
            print(txt_error + " Converted to string!")
            df[col] = df[col].astype(str)
    return df, object_cols, numerical_cols


def fill_column_null(df, column, null_values_list, null_value):
    """Fill column null."""
    df = df.copy()
    expect_true(column != "geometry", "Geometry columns cannot be filled")
    if df[column].dtype == "object":
        df.loc[df[column].str.upper().isin(null_values_list), column] = "NONE"
        df[column] = df[column].astype(str).str.strip().fillna("NONE")
    elif df[column].dtype == "bool":
        pass
    else:
        df[column] = df[column].fillna(np.inf)

    df.loc[df[column].isin(null_values_list), column] = null_value
    return df


def fill_nulls(df, object_cols, numerical_cols, string_null_values_list, numeric_null_values_list, string_null_value, numeric_null_value):
    """Fill nulls."""
    df = df.copy()
    for col in object_cols:
        df = fill_column_null(df, col, string_null_values_list, string_null_value)
    for col in numerical_cols:
        df = fill_column_null(df, col, numeric_null_values_list, numeric_null_value)
    return df


def convert_boolean_to_integer(df, field_name, fillna_val=0):
    """Convert boolean to integer."""
    df = df.copy()
    df[field_name] = df[field_name].fillna(fillna_val)

    if is_bool_dtype(df[field_name]):
        df[field_name] = df[field_name].map({True: 1, False: 0}).astype(int)

    elif is_numeric_dtype(df[field_name]):
        if "int" in str(df[field_name].dtype) and not df[~df[field_name].isin([0, 1])].empty:
            raise ValueError(f"Error: Column {field_name} contains not valid numbers!")

    elif is_string_dtype(df[field_name]):
        valid_values = ["0", "1", "FALSE", "TRUE"]
        expect_true(
            len(df[~(df[field_name].str.upper().isin(valid_values))]) == 0,
            f"Invalid values in the column {field_name}!",
        )
        print_formatted_txt(f"Field type string {field_name} is converted to integer!", "WARNING")
        df[field_name] = df[field_name].str.lower().map({"true": 1, "false": 0}).astype(int)

    df[field_name] = df[field_name].astype(int)
    return df


def standardize_dataframe(gdf, string_null_values_list, numeric_null_values_list, string_null_value=None, numeric_null_value=np.nan, raise_error=True):
    """Standardize dataframe."""
    gdf = gdf.copy()

    # Separate object and numerical columns
    gdf, object_cols, numerical_cols = get_column_types(gdf, raise_error=raise_error)
    print(f"Total records: {len(gdf)}")
    gdf = fill_nulls(gdf, object_cols, numerical_cols, string_null_values_list, numeric_null_values_list, string_null_value, numeric_null_value)

    return gdf


def convert_columns_to_string(df):
    """Convert columns to string."""
    df = df.copy()
    columns = list(df.columns)
    if "geometry" in columns:
        columns.remove("geometry")
    for column in columns:
        if is_numeric_dtype(df[column]):
            pass
        elif is_object_dtype(df[column]):
            df[column] = df[column].astype(str)
        elif is_datetime_dtype(df[column]):
            print(f"Converting Datetype column {column} to String")
            df[column] = pd.to_datetime(df[column], format="%Y-%m-%d")
            df[column] = df[column].astype(str)
        else:
            try:
                print(f"Try to convert to Datetype {column}")
                df[column] = pd.to_datetime(df[column], format="%Y-%m-%d")
                df[column] = df[column].astype(str)
            except (TypeError, ValueError):
                df[column] = df[column].astype(str)
    return df


def update_double_ids(df, col_id):
    """Update double ids."""
    df = df.copy()
    df[col_id] = df[col_id].astype(str).fillna("NULL")
    df_double = df[df.duplicated(subset=[col_id], keep=False)]
    double_ids = list(set(df_double[col_id]))
    if len(double_ids) > 0:
        print(f"Found {len(double_ids)} duplicates for {col_id}. They will be updated: {double_ids}")
        for pid in tqdm(double_ids):
            cond = df[col_id] == pid
            len_data = len(df[cond])
            list_a = list(len_data * [pid])
            list_b = [*range(1, len_data + 1, 1)]
            new_codes = [str(i) + "_" + str(j) for i, j in zip(list_a, list_b)]
            df.loc[cond, col_id] = new_codes
    else:
        print(f"No duplicates found for {col_id}.")
    return df


def update_list_columns(df):
    """Update list columns."""
    for col in df.columns:
        df[col] = df[col].apply(lambda x: x[0] if isinstance(x, list) else x)
    return df


def group_all_categorical_in_df(df, col_id):
    """Group all categorical in df."""
    df = df.copy()
    df, object_cols, _numerical_cols = get_column_types(df)
    for col in object_cols:
        if col != col_id:
            df[col] = df[col].replace(" ", "_", regex=True)
            # For each categorical keep the top most frequent and group the rest into one category with a value="Other"
            df = group_categorical(df, col)
    return df


def group_categorical(df, cat_column, max_categories=10, min_perc=1):
    """Group categorical."""
    df = df.copy()

    # Get counts of each category
    category_counts = df[cat_column].value_counts()

    # Calculate the threshold count based on the minimum percentage
    threshold_count = int(len(df) * min_perc / 100)

    # Get the top categories either based on max_categories or the threshold
    if len(category_counts) > max_categories:
        top_categories = category_counts.nlargest(max_categories)
    else:
        top_categories = category_counts

    # Ensure categories are above the threshold count
    top_categories = top_categories[top_categories >= threshold_count].index.tolist()

    # Update categories not in the top categories or above the threshold with 'Other'
    df.loc[~df[cat_column].isin(top_categories), cat_column] = "Other"

    return df


def list_to_str(x):
    """List to str."""
    return ", ".join(map(str, list(x)))


def convert_list_columns(df):
    """Convert list columns."""
    for col in df.columns:
        df[col] = df[col].apply(lambda x: list_to_str(x) if isinstance(x, list) else x)
    return df


def sort_df_columns(df, col_sort):
    """Sort df columns."""
    df = df.copy()
    # Extract column names and remove the sorting column
    features = [col for col in list(df.columns) if col != col_sort]

    # Create the final column order
    features = [col_sort] + sorted(features)

    # Remove 'geometry' if present
    if "geometry" in features:
        features.remove("geometry")
        features.append("geometry")

    # Select and sort the DataFrame
    df = df[features].sort_values(by=col_sort).reset_index(drop=True)

    return df


def define_field_data_types(df, dict_data_types):
    """Apply the data type for each column based on the dictionary <dict_data_types>.

    Keyword Arguments:
    df -- the dataframe to apply specific dtype for each column
    dict_data_types -- a dictionary where the key is the dtype and the value is a list with the dataframe column names
                       to apply the new dtype
    :return:
    the updated dataframe

    """
    df = df.copy()
    for field_type, field_list in dict_data_types.items():
        cols_to_convert = [col for col in df.columns if col in field_list]
        if len(cols_to_convert) > 0:
            if "int" in field_type:
                df[cols_to_convert] = df[cols_to_convert].round(0)
            print(f"Apply dtype '{field_type}' to columns: {cols_to_convert}")
            df[cols_to_convert] = df[cols_to_convert].astype(field_type)
    return df


def format_df_dtypes(df, col_id, field_dtypes, verbose):
    """Format df dtypes."""
    # we need to re-define the dtypes, since new features added (i.e. kernels, voronoi, etc.)
    print("Format the field types:")
    df = define_field_data_types(df, field_dtypes)
    # we sort all the columns and the rows to ensure stability
    print("Sort columns and records:")
    df = sort_df_columns(df, col_sort=col_id)

    if verbose:
        display(df.head(5))
        if skim is not None:
            display(skim(df))
    return df
