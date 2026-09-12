
from IPython import get_ipython
from IPython.display import display
from copy import deepcopy
from datetime import datetime
from matplotlib.ticker import MaxNLocator
from numpy import arange
from numpy import array
from pandas.api.types import is_bool_dtype
from pandas.api.types import is_datetime64_any_dtype as is_datetime_dtype
from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_object_dtype
from pandas.api.types import is_string_dtype
from scipy import stats
from scipy.stats import norm
from skimpy import skim
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from tqdm.notebook import tqdm
import dill
import fiona
import math
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
import pandas as pd
import random
import re
import seaborn as sns
import statistics
import time
import zipfile
from functools import wraps
import satellites.shared.constants as gb_l
import satellites.shared.geometry as geom_l
import satellites.shared.tabular as cm_l
import satellites.shared.io as io_l
from satellites.shared.formatting import human_bytes  # noqa: F401 - historical public export

def nearest_odd(number):
    # Ensure the returned value is odd and at least 3
    odd = number if number % 2 == 1 else number + 1
    return max(odd, 3)


def calculate_time_duration(begin_time, end_time):
    tot_secs = end_time - begin_time
    hours, remainder = divmod(tot_secs, 3600)
    minutes, seconds = divmod(remainder, 60)
    duration = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
    return duration


def time_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        begin_time = time.time()
        begin_datetime = datetime.fromtimestamp(begin_time)
        text2 = f"Function <{func.__name__}> started on {begin_datetime.strftime('%Y-%m-%d %H:%M:%S')}"
        text1 = "=" * (len(text2) + 6)
        text = text1 + "\n" + text2
        print_formatted_txt(text, "RUN")

        output = func(*args, **kwargs)

        end_time = time.time()
        end_datetime = datetime.fromtimestamp(end_time)
        duration = calculate_time_duration(begin_time, end_time)
        text1 = f"Execution time of function <{func.__name__}> ended on {end_datetime.strftime('%Y-%m-%d %H:%M:%S')} - duration: {duration}"
        text2 = "=" * (len(text1) + 6)
        text = text1 + "\n" + text2
        print_formatted_txt(text, "RUN")

        return output
    return wrapper


def reset_index(*dfs):
    for df in dfs:
        df.reset_index(inplace=True, drop=True)


def save_ml_model(model, file_path):
    """Saves the ML model."""
    file_path = file_path.replace("\\", "/")
    model_name = file_path.split("/")[-1]

    try:
        # remove the file if it already exists
        if os.path.exists(file_path):
            print(f"Removing existing file: {file_path}")
            os.remove(file_path)

        print(f"\nSaving model: {model_name}")
        with open(file_path, "wb") as file_obj:
            dill.dump(model, file_obj)

    except Exception:
        print(f"Error saving model: {model_name}")


def read_ml_model(file_path):
    """Reads the ML model."""
    file_path = file_path.replace("\\", "/")
    model_name = file_path.split("/")[-1]

    try:
        print(f"\nOpening model: {model_name}")
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception:
        print(f"Error opening model: {model_name}")


def fix_json_chars_in_column_names(df):
    """Remove JSON characters from dataframe's column names.
    param df -- the dataframe we want to apply edits on column names
    :return: the dataframe without JSON characters on column names
    """
    col_names = set(df.columns)
    df = df.rename(columns=lambda x: re.sub("[^A-Za-z0-9_]+", "_", x))
    col_names_fixed = set(df.columns)
    if col_names != col_names_fixed:
        print(f"We removed JSON characters:\nBefore: {col_names}\nAfter: {col_names_fixed}")
    return df


def fix_json_chars_in_column_values(df, col_to_exclude):
    """Remove JSON characters from dataframe's column values.
    param df -- the dataframe we want to apply edits on column values
    param col_to_exclude -- the column names to exclude from this check
    :return: the dataframe without JSON characters on column values
    """
    cols = [col for col, dt in df.dtypes.items() if dt == object]
    cols = [col for col in cols if col not in col_to_exclude]

    for col in cols:
        if isinstance(df[df[col].notna()].iloc[0][col], str):
            print(f"Fixing json character values in column: {col}")
            df[col] = df[col].apply(lambda x: re.sub("[^A-Za-z0-9_]+", "_", x))
    return df


def convert_datetime_to_string(df):
    """
    Coverts datetime column to string of a given dataframe.

    Keyword arguments:
    df -- dataframe to convert its datetime columns
    :return: the converted dataframe
    """
    df_copy = df.copy()
    for col in df_copy.columns:
        if pd.api.types.is_datetime64_any_dtype(df_copy[col]):
            df_copy[col] = df_copy[col].astype(str)
    return df_copy


def handle_xlsx_file(file_path, watch_curly_brackets, encoding):
    """
    Handles the xlsx files we want to read.

    Keyword arguments:
    file_path -- path to file to read
    watch_curly_brackets -- whether to watch curly brackets
    encoding -- csv encoding
    :return: Dataframe df read from file
    """
    out_dir = os.sep + os.path.join(*[p for p in file_path.split(os.sep)[:-1]])
    # command to turn the xlsx file into a csv file
    libre_office_command = f"libreoffice --convert-to csv:\"Text - txt - csv " \
                           f"(StarCalc)\":59,34,UTF8 \"{file_path}\" --outdir {out_dir}"
    os.system(libre_office_command)
    file_path = f"{os.path.splitext(file_path)[0]}.csv"
    df = io_l.read_csv(file_path, watch_curly_brackets, encoding, delimiter=";")
    os.remove(file_path)
    return df


def convert_datetime_to_object(df):
    for col in df.columns:
        if "datetime64" in str(df[col].dtype):
            df[col] = df[col].astype(object)
    return df


def standardize_fields(df, field_mapping, reverse=False):
    """Rename fields according to given field_mapping. Drop the rest.
    If reverse=True, reverses the <field_mapping> keys to values.

    Keyword arguments:
    df -- the dataframe we want to standardize
    field_mapping -- a dictionary with old and new field name
    :return:
    renamed dataframe with only fields that exist on <field_mapping>
    """
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
    assert len(set(list(df.columns))) == len(list(df.columns)), "Duplicate column names"

    return df


def calculate_length_and_convert_to_ft(df, col_length="pipe_len"):
    df = df.copy()
    df[col_length] = df['geometry'].length
    df_current_unit = geom_l.get_unit_of_length(df)
    if df_current_unit != "ft":
        conversion_factor = geom_l.create_unit_conversion_factor(source_unit=df_current_unit, target_unit="ft")
        if conversion_factor != 1:
            df[col_length] = df[col_length] * conversion_factor
    return df


def print_formatted_txt(msg, txt_format="SECTION"):
    """Print <msg> as formatted text based on <txt_format> value.
    If <verbose> is 0 do not print the message.

    keyword arguments:
    msg -- the message we want to print
    txt_format -- format of the text
    """
    print("")
    if msg.lower().startswith("start"):
        print(100*"-")
    format_dict = {
        'RUN': '\033[94m',          # blue text for runs
        'SECTION': '\033[1;30;46m',  # text for sections
        'SUBSECTION': '\033[1;30m',    # subsections
        'RESULTS': '\033[92m',      # green text for results
        'WARNING': '\033[0;31m',    # critical text as warning
        'ERROR': '\033[0;31;40m',   # critical text as error - the code stops
        'END': '\033[0m'
    }
    print(f"{format_dict[txt_format]}{msg}{format_dict['END']}")
    if msg.lower().startswith("end"):
        print(100*"*")
        print("")


def check_needed_df_columns(df, needed_columns):
    """Checks if <needed_columns> exist if in df columns."""
    not_found_columns = []
    for column in needed_columns:
        if column not in df.columns:
            not_found_columns.append(column)
    if len(not_found_columns) > 0:
        raise KeyError(f"{not_found_columns} not in dataframe.")


def convert_col_id_to_str(df, field_name="pipe_id"):
    """Convert pipe_id type to string.

    Keyword arguments:
    df -- the dataframe we want to convert pipe_id to str type
    field_name -- the field name of id
    """
    # this function is used by data checks - auto leak
    assert field_name in df.columns, "We cannot find the field"

    # handle int and float cases
    if is_numeric_dtype(df[field_name]):
        df[field_name] = df[field_name].fillna(-1).astype(int).astype(str)
        df.loc[df[field_name] == "-1", field_name] = None


def convert_col_id_to_str_helper(*dfs):
    for df in dfs:
        convert_col_id_to_str(df)


def clean_col_id_helper(*dfs):
    for df in dfs:
        clean_col_id(df)


def clean_col_id(df, col_id):
    """Clean the col_id field from whitespaces (beginning and end of string)."""
    assert col_id in df.columns, f"Field {col_id} is missing"
    df[col_id] = df[col_id].apply(lambda x: str(x).strip())


def move_col_to_last(df, col_name, inplace=False):
    """Moves the given <col_name> to be last in the given dataframe <df>"""
    if not inplace:
        df = df.copy()
    assert col_name in df.columns, f"Column: {col_name} does not exist in dataframe with columns: {str(df.columns)}"
    if df.columns[-1] == col_name:
        pass
    else:
        col = df.pop(col_name)
        df.insert(len(df.columns), col_name, col)
    if not inplace:
        return df


def get_column_types(df, exclude_geometry_column=True, raise_error=False):
    df = df.copy()
    if exclude_geometry_column and "geometry" in list(df.columns):
        df.drop(columns=['geometry'], inplace=True)

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
            else:
                print(txt_error + " Converted to string!")
                df[col] = df[col].astype(str)
    return df, object_cols, numerical_cols


def fill_column_null(df, column, null_values_list, null_value):
    df = df.copy()
    assert column != "geometry", "Geometry columns cannot be filled"
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
    df = df.copy()
    for col in object_cols:
        df = fill_column_null(df, col, string_null_values_list, string_null_value)
    for col in numerical_cols:
        df = fill_column_null(df, col, numeric_null_values_list, numeric_null_value)
    return df


def plot_histogram_with_mean_std_lines(df, column_name, ax, x_label=""):
    series = df[column_name]
    mean_val = series.mean()
    std_val = series.std()

    sns.histplot(series, kde=True, color="skyblue", bins=100, ax=ax)

    for value, color, label in [
        (mean_val, "red", f"Mean: {mean_val:.2f}"),
        (mean_val - std_val, "green", f"Mean - Std: {mean_val - std_val:.2f}"),
        (mean_val + std_val, "green", f"Mean + Std: {mean_val + std_val:.2f}")
    ]:
        ax.axvline(value, color=color, linestyle="--", label=label)

    min_val, max_val = series.min(), series.max()
    range_val = max_val - min_val

    if range_val == 0:
        # Handle the case where all values are the same
        ax.set_xticks([min_val])
        ax.set_xticklabels([f"{min_val:.2f}"])
    else:
        step = range_val / 10

        if range_val > 10:
            ticks = np.arange(int(min_val), int(max_val) + 2, max(1, int(step)))
        elif range_val > 0.1:
            ticks = np.arange(round(min_val, 2), round(max_val, 2) + step, round(step, 2))
        else:
            ticks = np.linspace(min_val, max_val, num=11)

        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{tick:.2f}" for tick in ticks])

    ax.set_xlabel(x_label)
    ax.legend()


def plot_box_plot(df, column_name, ax):
    sns.boxplot(x=df[column_name], ax=ax, color="skyblue", vert=False, flierprops={'markerfacecolor': 'r', 'marker': 'D'})
    ax.set_title(f"Distribution of {column_name}")


def plot_stat_numeric(df, column_name, exclude_zeros=True, figsize=(18, 3), save_folder=""):
    df = df.copy()
    if exclude_zeros:
        df = df[df[column_name] > 0].copy()

    fig, axs = plt.subplots(ncols=1, nrows=2, sharex=True, figsize=figsize)
    plot_box_plot(df, column_name, axs[0])
    plot_histogram_with_mean_std_lines(df, column_name, axs[1])

    plt.xlabel(column_name)
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    # Show when save_folder is falsy (None or empty string)
    if not save_folder:
        plt.show()
    else:
        plt.savefig(os.path.join(save_folder, f"{column_name}_barplot.png"))
        plt.close()


def plot_stat_object(df, column_name, show_common=10, palette="pastel", figsize=(30, 6), save_folder=""):
    df = df.copy()
    top_common_values = df[column_name].value_counts().head(show_common).reset_index()
    filtered_df = df[df[column_name].isin(top_common_values[column_name])]
    plt.figure(figsize=figsize)
    ax = sns.countplot(data=filtered_df, y=column_name, order=top_common_values[column_name], hue=column_name, palette=palette, legend=False)
    plt.title(f"Distribution of {column_name}")
    annotate_bars(ax)
    plt.tight_layout()
    # Show when save_folder is falsy (None or empty string)
    if not save_folder:
        plt.show()
    else:
        plt.savefig(os.path.join(save_folder, f"{column_name}_barplot.png"))
        plt.close()


def annotate_bars(ax, precision=0):
    for p in ax.patches:
        ax.annotate(f'{p.get_width():.{precision}f}',
                    (p.get_width(), p.get_y() + p.get_height() / 2),
                    xytext=(5, 0),
                    textcoords="offset points",
                    ha="left",
                    va="center")


def preprocess_number_column(df, field_to_check, null_values, min_value, max_value, plot_graphs, exclude_zeros_in_plot=True):
    """Checks and converts to float the column <field_to_check>."""
    df = df.copy()
    if is_numeric_dtype(df[field_to_check]):
        print(f"{field_to_check} is numeric")
        df[field_to_check] = df[field_to_check].fillna(0).astype(float)
        df.loc[df[field_to_check].isin(null_values), field_to_check] = 0
    else:
        print(f"{field_to_check} is object")
        df.loc[df[field_to_check].isin(null_values), field_to_check] = "0"
        df[field_to_check] = df[field_to_check].fillna("0").astype(str)

        if any(~df[field_to_check].astype(str).str.match(r"^-?\d+(.\d+)?$")):
            display(df[field_to_check].value_counts(dropna=False))
            raise Exception(f"Check column {field_to_check} values")
        df[field_to_check] = df[field_to_check].astype(float)

    df_nulls = df[df[field_to_check] == 0]
    print(f"\nNull {field_to_check}: {len(df_nulls)} ({round(100 * len(df_nulls) / len(df), 2)}%)")

    cond_invalid = (df[field_to_check] < min_value) | (df[field_to_check] > max_value)
    df_issues = df[(df[field_to_check] != 0) & cond_invalid].copy()
    if len(df_issues) > 0:
        print(f"Invalid {field_to_check} values: {len(df_issues)} (diameter<{min_value} or diameter>{max_value})")
        display(df_issues.head())
        df.loc[cond_invalid, field_to_check] = 0

    if plot_graphs:
        plot_stat_numeric(df, field_to_check, exclude_zeros=exclude_zeros_in_plot)
    return df, df_issues


def process_date(df, null_values, col_date, min_year, max_year, col_year="year", col_month="month", col_day="day", plot_graphs=False):
    """Process date field and separate year-month-day."""
    df = df.copy()
    if is_numeric_dtype(df[col_date]):
        print(f"{col_date} is numeric")
        df[col_year] = df[col_date].fillna(0).astype(int)
        df[col_year] = df[col_year].apply(lambda x: 0 if x in null_values else x)
    else:
        print(f"{col_date} is not numeric")
        df[col_date] = df[col_date].astype(str)

        def parse_date(date_string):
            date_string = date_string.split(" |T")[0].replace("/", "-")
            vals = date_string.split("-")
            year, month, day = 0, 0, 0

            if len(vals) == 3:
                if len(vals[0]) == 4:
                    year, month, day = vals
                elif len(vals[2]) == 4:
                    year, day, month = vals[2], vals[1], vals[0]
            elif len(vals) == 2:
                if len(vals[0]) == 4:
                    year, month = vals
                elif len(vals[1]) == 4:
                    year, month = vals[1], vals[0]
            elif len(vals) == 1 and len(vals[0]) == 4:
                year = vals[0]

            return pd.Series({col_year: year, col_month: month, col_day: day})

        df[[col_year, col_month, col_day]] = df[col_date].apply(parse_date)
        df[[col_year, col_month, col_day]] = df[[col_year, col_month, col_day]].apply(pd.to_numeric, errors="coerce")

        # Swap month and day if month > 12 and day <= 12
        cond_invalid_values = (df[col_month] > 12) & (df[col_day] <= 12)
        if not df[cond_invalid_values].empty:
            df[[col_month, col_day]] = df[[col_day, col_month]]

        # Convert to Int64, NaN will become null
        for col in [col_year, col_month, col_day]:
            df[col] = df[col].astype('Int64')

    df_nulls = df[df[col_year] == 0]
    print(f"\nNull {col_year}: {len(df_nulls)} ({round(100 * len(df_nulls) / len(df), 2)}%)")

    cond_invalid = ((df[col_year] < min_year) | (df[col_year] > max_year))
    df_issues = df[(df[col_year] != 0) & cond_invalid].copy()
    if len(df_issues) > 0:
        print(f"Invalid {col_year} values: {len(df_issues)} ({col_year}<{min_year} or {col_year}>{max_year})")
        display(df_issues.head(10))
        df.loc[cond_invalid, col_year] = 0

    if plot_graphs:
        plot_stat_numeric(df, col_year)

    return df, df_issues


def get_geometries_info(gdf, figsize=(18,3), plot_graphs=True, verbose=True, save_folder=None):
    gdf = gdf.copy()
    gdf.reset_index(inplace=True)
    unit = geom_l.get_unit_of_length(gdf)
    len_invalid_geoms = len(geom_l.return_invalid_geometries(gdf))
    gdf = geom_l.return_valid_geometries(gdf)

    col_id = "gid"
    gdf[col_id] = gdf.index+1
    _, geom_outliers = geom_l.get_coordinate_outliers(gdf, col_id=col_id, plot_graphs=plot_graphs, figsize=figsize, verbose=verbose, save_folder=save_folder)
    geom_doubles = geom_l.get_duplicate_geometries(gdf, col_id=col_id, verbose=verbose)

    gdf['geom_type'] = gdf['geometry'].geom_type.astype(str)
    geom_types = gdf['geom_type'].value_counts(dropna=False).reset_index()

    # Create the formatted string
    geom_types_str = ", ".join([f"{row['geom_type']}: {row['count']}" for _, row in geom_types.iterrows()])

    result = [{'Column': 'geometry', 'Invalid Geometries': len_invalid_geoms, 'Outliers Geometries': len(geom_outliers), 'Double Geometries': len(geom_doubles), 'Coordinates projected': gdf.crs.is_projected, 'Coordinates Unit': unit, 'Geometry Types': geom_types_str}]
    return pd.DataFrame(result)


def df_info_per_column(df, object_cols, numerical_cols, verbose=True):
    """Get information per column of a dataframe.

    Args:
        df ([dataframe]): Input dataframe
        object_cols ([list]): Columns of object type
        numerical_cols ([list]): Columns of numerical type

    Returns:
        [dataframes]: two dataframes with info got object and numerical columns,
        respectively
    """
    df = df.copy()
    object_cols = set(object_cols)
    numerical_cols = set(numerical_cols)
    all_cols = list(object_cols.union(numerical_cols))

    total_rows = len(df)

    def format_count(count):
        return f"{count} ({100*count/total_rows:.2f}%)"

    def column_info(col):
        series = df[col]
        null_count = series.isna().sum() + series.isin(['None', 'Null', 'nan', 'NaN']).sum()
        non_null_count = total_rows - null_count

        if col == "geometry":
            geom_nulls = geom_l.return_invalid_geometries(df)
            null_count = len(geom_nulls)
            non_null_count = total_rows - null_count
            zero_count = 0
            no_zero_count = non_null_count
            object_cols.add(col)
        else:
            zero_count = (series == 0).sum()
            no_zero_count = non_null_count - zero_count

        return {
            'Column': col,
            'Data type': str(series.dtype),
            'Nulls': format_count(null_count),
            'Non-nulls': format_count(non_null_count),
            'Zeros': format_count(zero_count),
            'Non-Zeros': format_count(no_zero_count),
            'Duplicates': df.duplicated(subset=[col]).sum(),
            'Unique': series.nunique()
        }

    info = [column_info(col) for col in df.columns if col in all_cols]
    info_df = pd.DataFrame(info).sort_values('Column')

    info_df_obj = info_df[info_df['Column'].isin(object_cols)][['Column', 'Data type', 'Nulls', 'Non-nulls', 'Duplicates', 'Unique']]
    info_df_num = info_df[info_df['Column'].isin(numerical_cols)][['Column', 'Data type', 'Nulls', 'Non-nulls', 'Zeros', 'Non-Zeros', 'Duplicates', 'Unique']]

    if verbose:
        display(info_df_obj)
        display(info_df_num)
    return info_df_obj, info_df_num


def convert_boolean_to_integer(df, field_name, fillna_val=0):
    df = df.copy()
    df[field_name] = df[field_name].fillna(fillna_val)

    if is_bool_dtype(df[field_name]):
        df[field_name] = df[field_name].map({True: 1, False: 0}).astype(int)

    elif is_numeric_dtype(df[field_name]):
        if "int" in str(df[field_name].dtype):
            if not df[~df[field_name].isin([0, 1])].empty:
                raise Exception(f"Column {field_name} contains not valid numbers!")

    elif is_string_dtype(df[field_name]):
        valid_values = ['0', '1', 'FALSE', 'TRUE']
        assert len(df[~(df[field_name].str.upper().isin(valid_values))]) == 0, f"Invalid values in the column {field_name}!"
        cm_l.print_formatted_txt(f"Field type string {field_name} is converted to integer!", "WARNING")
        df[field_name] = df[field_name].str.lower().map({'true': 1, 'false': 0}).astype(int)

    df[field_name] = df[field_name].astype(int)
    return df


def standardize_dataframe(gdf, string_null_values_list, numeric_null_values_list, string_null_value=None, numeric_null_value=np.nan, raise_error=True):
    gdf = gdf.copy()

    # Separate object and numerical columns
    gdf, object_cols, numerical_cols = get_column_types(gdf, raise_error=raise_error)
    print(f"Total records: {len(gdf)}")
    gdf = fill_nulls(gdf, object_cols, numerical_cols, string_null_values_list, numeric_null_values_list, string_null_value, numeric_null_value)

    return gdf


def analyze_dataframe(gdf, hue_column=None, exclude_zeros_in_plot=True, plot=True, figsize=(15, 3)):

    gdf, object_cols, numerical_cols = get_column_types(gdf)

    # Print info per column
    print("Info per column:")
    _, _ = df_info_per_column(gdf, object_cols, numerical_cols)

    if plot:
        # Plot bar plot for object columns
        for col in object_cols:
            plot_stat_object(gdf, col, show_common=10, figsize=figsize)

        # Plot box plot for numerical columns
        for col in numerical_cols:
            plot_stat_numeric(gdf, col, exclude_zeros=exclude_zeros_in_plot, figsize=figsize)

        display(gdf.describe())

        if exclude_zeros_in_plot:
            for col in numerical_cols:
                df = gdf[gdf[col]!=0].copy()

        if len(list(df.columns)) > 0:
            sns.pairplot(data=df, corner=True) if hue_column is None else sns.pairplot(data=df, hue=hue_column, corner=True)

    return gdf


def plot_correlation_matrix(df, figsize=(10, 5), fontsize=6):
    # Calculate the correlation matrix
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, mask=mask, cmap="coolwarm", vmax=.3, center=0, square=True, linewidths=.5, annot=True,
                cbar_kws={"shrink": .5}, annot_kws={"fontsize": fontsize})
    plt.title("Correlation Matrix with Heatmap")
    plt.show()


def get_IQR_outlier_limits(df, field_name, quantiles, max_range=1.5, verbose=True, plot=True):
    """Finds the outliers in the column <field_name> with Q1-Q3 method. The IQR (Interquartile Range) is a statistical
    methodology used to measure the spread or dispersion of a dataset. It finds the main distribution and returns the
    min and max values of the main distribution. Any data points falling outside of these bounds are considered
    outliers.

    Keyword arguments:
    df -- the dataframe to be checked
    filed_name -- the column of the <df> to be checked
    quantiles -- the percentages defining the interquartile range of the distribution.
    max_range -- the maximum value from the first quartile Q1 and the last quartile Q3 defining the outliers
    :return:
    the values of min_Q1, max_Q3 defining the first and last values of the distribution
    """
    df = df.copy()
    # calculate Q1 and Q3: determine the 1st quartile(quantiles[0]) and 3rd quartile(quantiles[1]) of the data.
    Q1 = df[field_name].quantile(quantiles[0])
    Q3 = df[field_name].quantile(quantiles[1])
    # calculate the Interquartile Range (IQR) by subtracting Q1 from Q3.
    IQR = Q3 - Q1

    # find the lower and upper bounds for detecting outliers; these are typically calculated as
    # Q1-max_range*IQR (for the lower bound) and
    # Q3+max_range* IQR (for the upper bound)
    # any data points falling outside of these bounds are considered outliers
    min_Q1 = Q1 - max_range * IQR
    max_Q3 = Q3 + max_range * IQR
    if verbose:
        print("----------------------------------")
        print(f"Q1-Q3 analysis for {field_name}")
        print(f"Q1: {round(Q1, 5)} Q3: {round(Q3, 5)}")
        print(f"IQR: {round(IQR, 5)}")
        print(f"Q1-{max_range}*IQR: {round(min_Q1, 5)}")
        print(f"Q3+{max_range}*IQR: {round(max_Q3, 5)}")
        print("----------------------------------")

    if plot:
        plot_stat_numeric(df, field_name, exclude_zeros=False)
    return min_Q1, max_Q3


def get_dataframe_outliers_with_IQR(df, field_name, quantiles, max_range=1.5, verbose=True, plot=True):
    min_Q1, max_Q3 = get_IQR_outlier_limits(df, field_name, quantiles, max_range=max_range, verbose=verbose, plot=plot)

    filter_outlier = (df[field_name] < min_Q1) | (df[field_name] > max_Q3)
    if verbose:
        print(f"There are {len(df[filter_outlier])} outliers!")

    df['is_outlier'] = False
    df.loc[filter_outlier, 'is_outlier'] = True
    return df


def get_dataframe_outliers_with_zscore(df, field_name, zscore=3, verbose=True, plot=True):
    """Finds the outliers in the column <field_name> with zscore method."""
    df = df.copy()
    # Calculate z-scores
    df['z_scores'] = np.abs(stats.zscore(df[field_name]))

    # Identify anomalies (z-score threshold > zscore)
    filter_outlier = df['z_scores'] > zscore
    if verbose:
        print(f"There are {len(df[filter_outlier])}!")
    df.loc[filter_outlier, 'is_outlier'] = True
    df.drop(columns=['z_scores'], inplace=True)
    if plot:
        plot_stat_numeric(df, field_name, exclude_zeros=False)
    return df


def get_dataframe_outliers_with_isolation_forest(df, field_name, contamination=0.01, verbose=True, plot=True):
    """Finds the outliers in the column <field_name> with Isolation Forest method."""
    df = df.copy()
    # Fit the Isolation Forest model
    iso_forest = IsolationForest(contamination=contamination, random_state=gb_l.SEED_NUMBER)

    iso_forest.fit(df[[field_name]])
    df['anomalies'] = iso_forest.predict(df[[field_name]])

    filter_outlier = df['anomalies'] == 0
    if verbose:
        print(f"There are {len(df[filter_outlier])}!")

    df.loc[filter_outlier, 'is_outlier'] = True
    df.drop(columns=['anomalies'], inplace=True)
    if plot:
        plot_stat_numeric(df, field_name, exclude_zeros=False)
    return df


def get_dataframe_outliers_with_DBSCAN(df, field_name, eps=0.01, min_samples=10, verbose=True, plot=True):
    """Finds the outliers in the column <field_name> with Isolation Forest method."""
    df = df.copy()
    # Initialize the DBSCAN model
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)

    # Fit the DBSCAN model and predict anomalies
    df['anomalies'] = dbscan.fit_predict(df[[field_name]])

    filter_outlier = df['anomalies'] == -1
    if verbose:
        print(f"There are {len(df[filter_outlier])}!")

    df.loc[filter_outlier, 'is_outlier'] = True
    df.drop(columns=['anomalies'], inplace=True)
    if plot:
        plot_stat_numeric(df, field_name, exclude_zeros=False)
    return df


def convert_columns_to_string(df):
    """Checks if a column has datetime format and converts it to string."""
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


def get_statistics_per_group(df, field_group, field_to_check, exclude_zeros=True, digits=2):
    """This function groups the <df> dataframe based on the column <field_group> and prints a dataframe with
    the statistics of the <field_to_check>.

    Keyword arguments:
    df -- the df dataframe to process
    field_group -- the name of the field to group the df
    field_to_check -- the name of the field for which we want to calculate the statistics
    exclude_zeros -- a boolean variable to exclude or not the zero values from the statistics
    digits -- the number of digits to show in the stastics dataframe
    :return:
    the dataframe with the statistics
    """
    check_needed_df_columns(df, [field_group, field_to_check])
    if field_to_check == "geometry" or str(df[field_to_check].dtype) == "object":
        raise Exception("Geometry and Object columns cannot be checked!")

    df = df.copy()
    if exclude_zeros:
        df = df[df[field_to_check]!=0].copy()

    results = []
    for group_name, grouped_df in df.groupby(field_group):
        cond = (grouped_df[field_to_check].isna())
        df_no_nulls = grouped_df[~cond].copy()
        total_df = len(grouped_df)
        perc_null = round(100 * cond.sum() / total_df, 2)

        if len(df_no_nulls):
            stats = {f"{field_to_check}_{k}": round(v(df_no_nulls[field_to_check]), digits) for k, v in
                     zip(['mode', 'median', 'mean', 'min', 'max', 'std'], [lambda x: x.mode()[0], lambda x: x.median(), lambda x: x.mean(),
                                                                     lambda x: x.min(), lambda x: x.max(), lambda x: x.std()])}
        else:
            stats = {f"{field_to_check}_{k}": None for k in ['mode', 'median', 'mean', 'min', 'max', 'std']}

        results.append({field_group: group_name, "total_df": total_df, f"{field_to_check}_perc_null": perc_null, **stats})

    df_stats = pd.DataFrame(results).sort_values(by="total_df", ascending=False)
    style_format = "{:." + str(digits) + "f}"
    for col in df_stats.columns.drop([field_group, 'total_df']):
        df_stats[col] = df_stats[col].astype(float).round(digits).map(style_format.format)

    display(df_stats)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 10))
    sns.histplot(data=df, x=field_to_check, hue=field_group,  multiple="stack", ax=ax)


def xicor(X, Y, ties=True):
    """New correlation coefficient between the Y and Y."""
    random.seed(42)
    n = len(X)
    order = array([i[0] for i in sorted(enumerate(X), key=lambda x: x[1])])
    if ties:
        l = array([sum(y >= Y[order]) for y in Y[order]])
        r = l.copy()
        for j in range(n):
            if sum([r[j] == r[i] for i in range(n)]) > 1:
                tie_index = array([r[j] == r[i] for i in range(n)])
                r[tie_index] = random.choice(r[tie_index] - arange(0, sum([r[j] == r[i] for i in range(n)])), sum(tie_index), replace=False)
        return 1 - n*sum( abs(r[1:] - r[:n-1]) ) / (2*sum(l*(n - l)))
    else:
        r = array([sum(y >= Y[order]) for y in Y[order]])
        return 1 - 3 * sum( abs(r[1:] - r[:n-1]) ) / (n**2 - 1)


def convert_gdb_to_gpkg(gdb_file, output_file):
    for layer in tqdm(fiona.listlayers(gdb_file), desc="Exporting feature classes"):
        print("----------------------------------------------------------------------")
        print(f"Layer: {layer}")
        gdf = io_l.read_data(file_path=gdb_file, layer=layer)
        if not gdf.empty:
            gdf = convert_columns_to_string(gdf)
            io_l.write_data(gdf, output_file, layer=layer)
    print("Process completed successfully.")


def get_gdb_layers(gdb_file):
    layers = list(fiona.listlayers(gdb_file))
    print(f"Layers in the {gdb_file}:\n{layers}")
    return layers

def update_double_ids(df, col_id):
    """Updates the duplicates of <col_id> column, appending an auto-increment number.

    Keyword arguments:
    df -- the dataframe we want to apply edits on column values
    col_id -- the column name we want to update
    :return:
    the dataframe with the updated <col_id>
    """

    df = df.copy()
    df[col_id] = df[col_id].astype(str).fillna("NULL")
    df_double = df[df.duplicated(subset=[col_id], keep=False)]
    double_ids = list(set(df_double[col_id]))
    if len(double_ids) > 0:
        print(f"Found {len(double_ids)} duplicates for {col_id}. They will be updated: {double_ids}")
        for pid in tqdm(double_ids):
            cond = (df[col_id] == pid)
            len_data = len(df[cond])
            list_a = list(len_data * [pid])
            list_b = [*range(1, len_data + 1, 1)]
            new_codes = [str(i) + "_" + str(j) for i, j in zip(list_a, list_b)]
            df.loc[cond, col_id] = new_codes
    else:
        print(f"No duplicates found for {col_id}.")
    return df


def fill_empty_string(val, num_spaces=10):
    return str(val).ljust(num_spaces)


def update_list_columns(df):
    for col in df.columns:
        df[col] = df[col].apply(lambda x: x[0] if isinstance(x, list) else x)
    return df


def group_all_categorical_in_df(df, col_id):
    df = df.copy()
    df, object_cols, numerical_cols = get_column_types(df)
    for col in object_cols:
        if col != col_id:
            df[col] = df[col].replace(" ", "_", regex=True)
            # For each categorical keep the top most frequent and group the rest into one category with a value="Other"
            df = group_categorical(df, col)
    return df


def group_categorical(df, cat_column, max_categories=10, min_perc=1):
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
    return ", ".join(map(str, list(x)))


def convert_list_columns(df):
    for col in df.columns:
        df[col] = df[col].apply(lambda x: list_to_str(x) if isinstance(x, list) else x)
    return df


def highlight_df_rows(index_names, color="yellow"):
    def highlight(row):
        if row.name in index_names:
            return [f'font-weight: bold; background-color: {color}'] * len(row)
        else:
            return [''] * len(row)
    return highlight


def get_notebook_name():
    if multiprocessing.current_process().name != "MainProcess":
        return None  # Return a default value or handle it differently in the multiprocessing context

    ip = get_ipython()
    if ip is None:
        return None  # Return a default value or handle it differently when not in IPython

    if "__vsc_ipynb_file__" in ip.user_ns:
        return ip.user_ns["__vsc_ipynb_file__"]
    return None


def sort_df_columns(df, col_sort):
    df = df.copy()
    # Extract column names and remove the sorting column
    features = [col for col in list(df.columns) if col != col_sort]

    # Create the final column order
    features = [col_sort] + sorted(features)

    # Remove 'geometry' if present
    if "geometry" in features:
        features.remove('geometry')
        features.append('geometry')

    # Select and sort the DataFrame
    df = df[features].sort_values(by=col_sort).reset_index(drop=True)

    return df


def define_field_data_types(df, dict_data_types):
    """Apply the data type for each column based on the dictionary <dict_data_types>.

    Keyword arguments:
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
    """Formats the dtypes and saves the final data.

    Keyword arguments:
    df -- the dataframe to format
    col_id -- the column with the ids
    field_dtypes -- the datatypes of each field
    verbose -- verbose the results
    :return: the final dataframe with the formated dtypes
    """
    # we need to re-define the dtypes, since new features added (i.e. kernels, voronoi, etc.)
    print("Format the field types:")
    df = define_field_data_types(df, field_dtypes)
    # we sort all the columns and the rows to ensure stability
    print("Sort columns and records:")
    df = sort_df_columns(df, col_sort=col_id)

    if verbose:
        display(df.head(5))
        display(skim(df))
    return df


def calc_confidence_level(zeta):
    return norm.cdf(zeta, loc=0, scale=1) - norm.cdf(-zeta, loc=0, scale=1)


def calc_confidence_level_for_sample(n_population, n_sample):
    if n_population > n_sample:
        e = 0.05
        p = 0.5
        a = n_sample * (n_population - 1) / (n_population - n_sample)
        z = math.sqrt((a * e * e) / (p * (1 - p)))
        return round(calc_confidence_level(z) * 100, 1)
    elif n_population == n_sample:
        return 100
    else:
        raise Exception("The sample size exceeds the population.")


def zeta_score(conf_level):
    z = abs(norm.interval(conf_level, loc=0, scale=1)[1])
    return z


def get_population_mean_and_margins(n_population, n_sample, lead_in_sample, conf_level=95):
    p = lead_in_sample / n_sample
    z = zeta_score(conf_level / 100)
    total_mean = round(n_population * p)
    total_variance = n_population * p * (1 - p)
    total_sdev = math.sqrt(total_variance)
    margin = round(z * total_sdev)
    p = round(100 * p, 2)
    return p, total_mean, margin


def calculate_confidence_interval_from_list(data, conf_level, test, verbose=True, plot=True):
    conf_level_perc = conf_level/100
    mean = np.mean(data)
    std_dev = np.std(data)
    n = len(data)
    z = zeta_score(conf_level_perc)
    conf_interval = (mean - z * (std_dev / np.sqrt(n)), mean + z * (std_dev / np.sqrt(n)))
    output_text = f"{conf_level}%: {int(conf_interval[0])} - {int(conf_interval[1]+0.5)}"

    if verbose:
        print(f"\nConfidence interval for {output_text}")

    if plot:
        fig, axs = plt.subplots(1, 3, figsize=(20,5))
        axs[0].plot(data, "o")
        axs[0].axhline(y=mean, color="r", linestyle="-")  # average line
        axs[0].axhline(y=conf_interval[0], color="g", linestyle="--")  # lower bound of confidence interval
        axs[0].axhline(y=conf_interval[1], color="b", linestyle="--")  # upper bound of confidence interval
        axs[0].fill_between(range(n), conf_interval[0], conf_interval[1], color="b", alpha=0.1)  # fill between lines
        axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))
        axs[0].yaxis.set_major_locator(MaxNLocator(integer=True))
        axs[0].set_title(f"Test {test}: Mean and Conf. Intervals for {conf_level}% conf. level", fontsize = 10)

        # plot distribution
        axs[1].hist(data, bins=10, density=True, alpha=0.6, color="g")
        xmin, xmax = axs[1].get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mean, std_dev)
        axs[1].plot(x, p, "k", linewidth=2)
        axs[1].set_title(f"Test {test}: Distribution of Data", fontsize = 10)

        # Q-Q plot
        stats.probplot(data, dist="norm", plot=axs[2])
        axs[2].set_title(f"Test {test}: Q-Q Plot", fontsize = 10)
        plt.show()
    return conf_interval, output_text


def calculate_confidence_level_from_list(data, verbose):
    """
    Calculates the confidence level and margin of error from a list of data.

    Parameters:
    data (list): A list of numerical data.
    verbose (bool): A flag indicating whether to print detailed information.

    Returns:
    confidence_level (float): The confidence level as a percentage.
    output_text (str): A string representation of the confidence level and margin of error.
    """
    n = len(data)
    mean_value, min_value, max_value = np.mean(data), min(data), max(data)
    std_dev = np.std(data)
    var_value = statistics.variance(data)

    if verbose:
        print(f"\nMean: {round(mean_value, 2)}, Min: {round(min_value)}, Max: {round(max_value)}, SDev: {round(std_dev, 2)}, Variance: {round(var_value, 2)}")

    # use variance as margin of error
    margin_of_error = var_value
    z_score = margin_of_error * np.sqrt(n) / std_dev
    confidence_level = 100.0 * norm.cdf(z_score)
    min_value = int(mean_value - margin_of_error)
    max_value = min(int(mean_value + margin_of_error + 0.5), 100)
    if confidence_level <= 99:
        output_text = f"{round(confidence_level, 1)}%: {min_value} - {max_value}"
    else:
        output_text = f">99%: {min_value} - {max_value}"

    if verbose:
        print(f"Confidence Level: {output_text}")

    return confidence_level, output_text


def create_bar_graph(df, col_name, labels, graph_type, color_dict):
    category_counts = df[col_name].value_counts()

    plt.figure(figsize=(15, 8))
    colors = [color_dict.get(cat, "#000000") for cat in category_counts.index]
    if graph_type == "bar":
        ax = sns.barplot(x=category_counts.index, y=category_counts.values, palette=colors)
        for p in ax.patches:
            ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height()), ha="center", va="center", xytext=(0, 5), textcoords="offset points", weight="bold")

    elif graph_type == "pie":
        wedges, texts, autotexts = plt.pie(category_counts.values, labels=category_counts.index, autopct="%1.1f%%", startangle=90, colors=colors)
        plt.setp(autotexts, size=10, weight="bold", color="black")
        center_circle = plt.Circle((0,0),0.40, fc="white")
        plt.gca().add_artist(center_circle)

    else:
        print("Not valid graph type")
        return

    plt.title(labels['title'])
    plt.xlabel(labels['x'])
    plt.ylabel(labels['y'])

    plt.xticks(rotation=45)
    plt.show()


def extract_zip_to_same_folder(zip_path):
    # Get the folder containing the zip file
    extract_folder = os.path.dirname(zip_path)

    # Open the zip file and extract all files to the specified folder
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_folder)

    print(f"Files extracted to: {extract_folder}")


def unzip_gis_file(zip_file, valid_extensions, extract_to_folder=None, raise_error=True):
    """
    Extracts a ZIP file to the directory where the ZIP file is located, replacing existing files if they exist.
    Checks for specific file extensions (.shp, .gpkg, .gdb) in the extracted files.
    For .gdb, it treats it as a folder rather than a single file.
    """
    extracted_files = []
    found_files = []

    # Check if the provided path is a valid zip file
    if not zipfile.is_zipfile(zip_file):
        txt_error = f"{zip_file} is not a valid ZIP file."
        if raise_error:
            raise ValueError(txt_error)
        else:
            print(txt_error)
            return found_files

    if extract_to_folder is None:
        # Get the directory where the ZIP file is located
        extract_to_folder = os.path.dirname(zip_file)

    # Open the zip file in read mode
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        # Iterate through each file in the zip archive
        for file_name in zip_ref.namelist():
            # Create the full path for the extracted file or folder
            file_path = os.path.join(extract_to_folder, file_name)

            # Extract the file or folder, replacing if it already exists
            zip_ref.extract(file_name, extract_to_folder)

            # Append the extracted path to the list
            extracted_files.append(file_path)

            # Check if the item is a valid match based on extensions
            if len(valid_extensions) > 0:
                # Handle .gdb as a folder, not a single file
                if file_name.lower().endswith('.gdb/') and '.gdb' in valid_extensions:
                    # Ensure only the folder path (not its internal files) is added
                    gdb_folder = os.path.join(extract_to_folder, file_name.rstrip('/'))
                    found_files.append(gdb_folder)
                elif any(file_name.lower().endswith(ext) for ext in valid_extensions):
                    found_files.append(file_path)
            else:
                found_files.append(file_path)

    # Handle cases based on the number of found files with the desired extensions
    if not found_files:
        txt_error = f"No files with extensions {valid_extensions} were found in the ZIP file."
        if raise_error:
            raise ValueError(txt_error)
        else:
            print(txt_error)
            return found_files

    if len(valid_extensions) > 0:
        if len(found_files) > 1:
            print(f"Multiple files found with valid extensions: {found_files}. Returning the first one.")
        return found_files[0]
    else:
        return found_files


def remove_special_characters_from_text(text, keep_chars=".,"):
    # Create a translation table
    chars_to_remove = "".join(c for c in set(text) if not c.isalnum() and c not in keep_chars)
    trans_table = str.maketrans("", "", chars_to_remove)

    # Apply the translation
    return text.translate(trans_table)




# TODO
# 11. Reducing Multicollinearity

    # There can be multicollinearity among the selected independent variables. It can cause the deep learning model to become unstable and prone to overfitting.
    # To remove it, we calculated each predictor’s variance inflation factor (VIF), where a value above ten means that the column is causing multicollinearity.

# # importing library for calculating vif
# from statsmodels.stats.outliers_influence import variance_inflation_factor

# # calculating vif ofthe selected features
# vif_df = pd.DataFrame()
# vif_df["Variable"] = X.columns
# vif_df["VIF"] = [variance_inflation_factor(X.values, i) forWein range(X.shape[1])]
# print("Variance Inflation Factors:")
# print(vif_df.sort_values(by = "VIF", ascending=False).reset_index(drop=True))
