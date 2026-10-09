"""Library containing functions for input/output files."""

import csv
import logging
import os
import pathlib
import shutil
import stat
import time
import zipfile
from collections.abc import Callable
from glob import glob
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import psutil

import shared.logging as log_l

_GPKG_EXTENSION = ".gpkg"
_XLSX_EXTENSION = ".xlsx"
_XLS_EXTENSION = ".xls"
_PKL_XZ_EXTENSION = ".pkl.xz"
_SHP_EXTENSION = ".shp"
_CSV_EXTENSION = ".csv"
_PKL_EXTENSION = ".pkl"
_XZ_EXTENSION = ".xz"
_GDB_EXTENSION = ".gdb"
_PARQUET_EXTENSION = ".parquet"
_GEOPARQUET_EXTENSION = ".geoparquet"


def is_valid_path_filename(input_str: str) -> bool:
    """Check whether a given string is a valid file path pointing to an existing file with an extension.

    Parameters
    ----------
    input_str :str
        The file path string to validate.

    Returns
    -------
    bool
        True if the input is a valid file path pointing to an existing file with a proper extension; otherwise, False.

    """
    # 1. Check if the input is a string
    if not isinstance(input_str, str):
        return False

    # 2. Use pathlib to handle the path structure
    path_obj = pathlib.Path(input_str)

    # 3. Check if it has a valid file extension (e.g., .txt, .jpg)
    has_extension = path_obj.suffix != ""

    # 4. Ensure the filename part is valid (not a directory)
    if not path_obj.name or not has_extension:
        return False

    # 5. Check if the path exists and is a file
    return bool(os.path.exists(input_str) and path_obj.is_file())


def separate_path_filename_extension(filepath: str) -> tuple[str, str, str]:
    """Split a file path into directory path, filename (without extension), and file extension.

    Parameters
    ----------
    filepath : str
        The full path to the file.

    Returns
    -------
    Tuple[str, str, str]
        A tuple containing the directory path, filename without extension, and the file extension.

    """
    # Split the file path into directory, filename, and extension
    filepath = filepath.replace("\\", "/")
    path, filename_with_extension = os.path.split(filepath)
    # If the file path is a directory, return empty strings for filename and extension
    filename, extension = os.path.splitext(filename_with_extension)
    return path, filename, extension


def has_subfolders(path: str) -> bool:
    """Check whether the given directory contains any sub-folders.

    Parameters
    ----------
    path : str
        The directory path to check.

    Returns
    -------
    bool
        True if the directory contains at least one sub-folder; otherwise, False.

    """
    return any(os.path.isdir(os.path.join(path, entry)) for entry in os.listdir(path))


def get_all_folders(path: str) -> list[str]:
    """Return a list of all sub-folders in the given directory.

    Parameters
    ----------
    path : str
        The path where to search for folders.

    """
    return [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]


def is_folder_empty(path: str) -> bool:
    """Check if the input folder is empty.

    Parameters
    ----------
    path : str
        The path where to search.

    """
    return len(os.listdir(path)) == 0


def move_folder(src_folder: str, dst_root: str) -> None:
    """Move a folder to a new destination directory.

    Parameters
    ----------
    src_folder : str
        Path to the source folder to move.
    dst_root : str
        Path to the destination directory. The folder will be moved inside this directory.

    Raises
    ------
    FileNotFoundError
        If the source folder does not exist.
    FileExistsError
        If a folder with the same name already exists at the destination.

    """
    src = Path(src_folder).resolve()
    dst = Path(dst_root).resolve() / src.name

    if not src.exists() or not src.is_dir():
        raise FileNotFoundError(f"Error: Source folder does not exist: {src}")

    if dst.exists():
        raise FileExistsError(f"Error: Destination folder already exists: {dst}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


def copy_folder(src_folder: str, dst_root: str) -> None:
    """Copy a folder to a new destination directory.

    Parameters
    ----------
    src_folder : str
        Path to the source folder to move.
    dst_root : str
        Path to the destination directory. The folder will be moved inside this directory.

    Raises
    ------
    FileNotFoundError
        If the source folder does not exist.
    FileExistsError
        If a folder with the same name already exists at the destination.

    """
    src = Path(src_folder).resolve()
    dst = Path(dst_root).resolve() / src.name

    if not src.exists() or not src.is_dir():
        raise FileNotFoundError(f"Error: Source folder does not exist: {src}")

    if dst.exists():
        raise FileExistsError(f"Error: Destination folder already exists: {dst}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(str(src), str(dst))


def close_open_files(file_path: str):
    """Close all processes that have the specified file open.

    Parameters
    ----------
    file_path : str
        The path of the file to check for open handles.

    Returns
    -------
    None

    Raises
    ------
    psutil.NoSuchProcess
        If a process no longer exists during iteration.
    psutil.AccessDenied
        If access to a process's information is denied.

    """
    # Iterate through all processes and check if the file is open
    # Note: This may require administrative privileges to close some processes
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            # Check if the process has the file open
            open_files = proc.open_files()
            # Check if the file path matches any open file paths
            if any(f.path == file_path for f in open_files):
                # Attempt to close the process
                proc.kill()
                print(f"Closed process {proc.name()} (PID: {proc.pid})")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue


def file_exists(file_path: str, verbose: bool = True) -> bool:
    """Check whether a file exists at the specified path.

    Parameters
    ----------
    file_path : str
        The path to the file to check.
    verbose : bool, optional
        If True, prints a message if the file does not exist. Defaults to True.

    Returns
    -------
    bool
        True if the file exists; otherwise, False.

    """
    if os.path.exists(file_path):
        return True
    else:
        if verbose:
            if os.path.isdir(file_path):
                print(f"Path '{file_path}' is a directory, not a file.")
            else:
                print(f"File '{file_path}' does not exist.")
        return False


def rename_file(src_path: str, new_name: str) -> None:
    """Rename a file to a new name in the same directory.

    Parameters
    ----------
    src_path : str
        Full path to the existing file.
    new_name : str
        New name for the file (just the filename, not full path).

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If the source file does not exist.
    FileExistsError
        If a file with the new name already exists.
    ValueError
        If new_name contains path separators.

    """
    if not os.path.isfile(src_path):
        raise FileNotFoundError(f"Error: File not found: {src_path}")

    if os.path.sep in new_name or (os.path.altsep and os.path.altsep in new_name):
        raise ValueError(f"Error: New name should not contain path separators: {new_name}")

    # Get the directory path and construct the new full path
    dir_path = os.path.dirname(src_path)
    dst_path = os.path.join(dir_path, new_name)

    if os.path.exists(dst_path):
        raise FileExistsError(f"Error: Destination file already exists: {dst_path}")

    os.rename(src_path, dst_path)


def delete_file(file_path: str):
    """Delete a file after attempting to close any processes that may be using it.

    Parameters
    ----------
    file_path : str
        The path to the file that should be deleted.

    Returns
    -------
    None

    Raises
    ------
    Exception
        If the source file cannot be deleted.

    """
    if file_exists(file_path):
        try:
            # Attempt to delete the file
            os.remove(file_path)
            print(f"{file_path} has been deleted.")
        except Exception as e:
            logging.getLogger(__name__).debug("Error: delete_file failed; using its fallback.", exc_info=True)
            try:
                print(f"First attempt to delete file {file_path} failed {e}; trying to close open files first!")
                # Attempt to close any open files before deleting
                close_open_files(file_path)
                os.remove(file_path)
                print(f"{file_path} has been deleted.")
            except Exception as e:
                raise RuntimeError(f"Error: Could not delete file {file_path}: {e}") from e
    else:
        print(f"{file_path} does not exist.")


def move_file(src_file: str, dst_file: str):
    """Move a file from a source path to a destination path. Deletes the destination file first if it exists.

    Parameters
    ----------
    src_file : str
        The path to the source file.
    dst_file : str
        The path to the destination file.

    Returns
    -------
    None

    Raises
    ------
    Exception
        If the source file does not exist.

    """
    if not file_exists(src_file):
        raise ValueError(f"Error: Source file '{src_file}' does not exist.")
    if file_exists(dst_file, verbose=False):
        delete_file(dst_file)
    shutil.move(src_file, dst_file)
    print(f"{src_file} has been moved to {dst_file}.")


def copy_file(src_file: str, dst_path: str, verbose: bool = True):
    """Copy a file from a source path to a destination path.

    Parameters
    ----------
    src_file : str
        The path to the source file.
    dst_path : str
        The path to the destination file or directory.

    Returns
    -------
    None

    Raises
    ------
    Exception
        If the source file does not exist.

    """
    if not file_exists(src_file):
        raise ValueError(f"Error: Source file '{src_file}' does not exist.")
    shutil.copy(src_file, dst_path)
    if verbose:
        print(f"{src_file} has been copied to {dst_path}.")


def _validate_folder_path(folder_path: Path) -> bool:
    """Validate that the folder path exists and is a directory.

    Parameters
    ----------
    folder_path : path
        The path to validate.

    Returns
    -------
    bool
        True if path is valid for deletion, False otherwise.
    """
    if not folder_path.exists():
        print(f"Folder '{folder_path}' does not exist.")
        return False

    if not folder_path.is_dir():
        print(f"Path '{folder_path}' is not a directory.")
        return False

    return True


def _log_deletion_error(error: Exception, attempt: int) -> None:
    """Log deletion error with appropriate message based on error type."""
    error_messages = {
        PermissionError: "Permission error",
        FileNotFoundError: "File not found",
        OSError: "OS error",
    }

    error_type = type(error)
    message = error_messages.get(error_type, "Unexpected error")
    print(f"{message} on attempt {attempt}: {error}")


def _attempt_deletion(folder_path: Path, attempt: int, force_permissions: bool) -> bool:
    """Attempt to delete the folder once.

    Parameters
    ----------
    folder_path : Path
        The path to delete.
    attempt : int
        The current attempt number.
    force_permissions : bool
        Whether to force permissions.

    Returns
    -------
    bool
        True if deletion was successful, False otherwise.
    """
    try:
        if force_permissions:
            shutil.rmtree(folder_path, onexc=_handle_readonly_error)
        else:
            shutil.rmtree(folder_path)

        print(f"Successfully deleted folder '{folder_path}' on attempt {attempt}")
        return True

    except Exception as e:
        logging.getLogger(__name__).debug("Error: _attempt_deletion failed; using its fallback.", exc_info=True)
        _log_deletion_error(e, attempt)
        return False


def _handle_readonly_error(func, path, exc_info):
    """Handle read-only file errors during deletion."""
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception as inner_e:
        print(f"Failed to change permissions for '{path}': {inner_e}")
        raise


def delete_folder(folder_path: str, max_retries: int = 3, delay: float = 1.0, force_permissions: bool = False):
    """Delete a folder with enhanced error handling and optional permission forcing.

    Parameters
    ----------
    folder_path : str
        The path to the folder to be deleted.
    max_retries : int, optional
        The number of retry attempts if deletion fails. Defaults to 3.
    delay : float, optional
        Delay in seconds between retry attempts. Defaults to 1.0.
    force_permissions : bool, optional
        Whether to attempt changing permissions on deletion failure. Defaults to True.

    Returns
    -------
    bool
        True if deletion was successful, False otherwise.
    """
    folder_path = Path(folder_path)

    if not _validate_folder_path(folder_path):
        return folder_path.exists() is False  # True if doesn't exist, False if not a dir

    print(f"Deleting folder '{folder_path}'...")

    for attempt in range(1, max_retries + 1):
        if _attempt_deletion(folder_path, attempt, force_permissions):
            return True

        if attempt < max_retries:
            print(f"Waiting {delay} seconds before retry...")
            time.sleep(delay)

    print(f"Failed to delete folder '{folder_path}' after {max_retries} attempts.")
    return False


def create_folder(folder_path: str, verbose=True):
    """Create a local folder at the specified path if it does not already exist.

    Parameters
    ----------
    folder_path : str
        The path where the folder should be created.

    Returns
    -------
    None

    """
    folder_path = folder_path.replace("\\", "/")
    if not os.path.exists(folder_path):
        try:
            # Create the folder
            os.makedirs(folder_path)
            if verbose:
                print(f"Folder '{folder_path}' created successfully.")
        except Exception as e:
            logging.getLogger(__name__).debug("Error: create_folder failed; using its fallback.", exc_info=True)
            print(f"Failed to create folder '{folder_path}': {e}")
    else:
        print(f"Folder '{folder_path}' already exists.")


def read_types_from_csvt(file_path: str):
    """Read a CSVT file and extracts the data types for each field into a dictionary.

    If any field has a 'datetime' type, its name is returned separately and removed from the dictionary.

    Parameters
    ----------
    file_path : str
        The path to the .csvt file.

    Returns
    -------
    Tuple[List[str], Dict[str, str]]
        A list containing the name of the datetime field (if any),and a dictionary mapping field names to their types
        (excluding datetime).

    """
    # Read the CSVT file
    with open(file_path, mode="r") as infile:
        reader = csv.reader(infile)
        mydict = {rows[0]: rows[1] for rows in reader}
        dates = []
        # Check for datetime fields and remove them from the dictionary
        for k, v in mydict.items():
            if "datetime" in v:
                dates = [k]
                mydict = {key: val for key, val in mydict.items() if key != k}
    return dates, mydict


def _typed_csv_dtype_label(dtype: Any) -> str:
    """Return a stable dtype label for typed CSV headers.

    Parameters
    ----------
    dtype : Any
        The pandas or NumPy dtype to serialize.

    Returns
    -------
    str
        A normalized dtype label suitable for typed CSV headers.

    """
    dtype_label = str(dtype)
    text_dtype_labels = {"str", "string", "string[python]", "string[pyarrow]"}
    if dtype_label in text_dtype_labels:
        return "object"
    return dtype_label


def write_to_typed_csv(df: pd.DataFrame | gpd.GeoDataFrame, file_path: str):
    """Write a DataFrame to a CSV file with typed headers, where each column name includes its data type.

    Parameters
    ----------
    df : Union[pd.DataFrame, gpd.GeoDataFrame]
        The DataFrame to write.
    file_path : str
        The path where the typed CSV file should be saved.

    Returns
    -------
    None

    """
    new_columns = {}
    # Create a new dictionary with column names and their types
    for i, col_type in enumerate(df.dtypes):
        col_name = df.columns[i]
        # Create a new column name with the type in curly brackets
        new_columns[col_name] = col_name + " {" + _typed_csv_dtype_label(col_type) + "}"
    # Rename the columns in the DataFrame
    df2 = df.rename(columns=new_columns)
    # Write the DataFrame to a CSV file with the new column names
    df2.to_csv(file_path, index=False)


def convert_df_col_datetime_to_string(df: pd.DataFrame | gpd.GeoDataFrame):
    """Convert all datetime columns in a DataFrame to string format.

    Parameters
    ----------
    df : Union[pd.DataFrame, gpd.GeoDataFrame]
        The input DataFrame whose datetime columns will be converted.

    Returns
    -------
    Union[pd.DataFrame, gpd.GeoDataFrame]
        A copy of the input DataFrame with datetime columns converted to strings.

    """
    df_copy = df.copy()
    for col in df_copy.columns:
        # Check if the column is of datetime type
        if pd.api.types.is_datetime64_any_dtype(df_copy[col]):
            # Convert the datetime column to string format
            df_copy[col] = df_copy[col].astype(str)
    return df_copy


def read_csv(file_path: str, watch_curly_brackets: bool, encoding: str | None, delimiter: str) -> pd.DataFrame:
    """Read a CSV file and returns a DataFrame. Supports three formats.

      * Typed CSV: field types are included in the header using curly brackets.
      * CSVT-based CSV: types are stored in a sidecar .csvt file.
      * Plain CSV: no type metadata.

    Parameters
    ----------
    file_path : str
        The path to the CSV file.
    watch_curly_brackets : bool
        Whether to interpret field types from curly brackets in the header.
    encoding : Optional[str]
        The encoding used to read the file.
    delimiter : str
        The delimiter used in the CSV file.

    Returns
    -------
    pd.DataFrame
        The DataFrame parsed from the CSV file.

    """
    # Read the first line to understand if it's a typed csv or not
    header = pd.read_csv(file_path, nrows=0, encoding=encoding, delimiter=delimiter).columns.tolist()

    # It's a typed csv, that has the col types in the header
    if all("{" in col and "}" in col for col in header) and watch_curly_brackets:
        type_dict = {}
        rename_dict = {}
        date_cols = []
        # Read the first line of the file to get the column names and types
        for col in header:
            if not col.endswith("}"):
                raise ValueError(f"Error: Header format issue: {col}")

            base_name, col_type = col.rsplit(" {", 1)
            col_type = col_type.rstrip("}")
            col_name = base_name.strip().strip('"')

            if "datetime" in col_type:
                date_cols.append(col)
            else:
                type_dict[col] = col_type

            rename_dict[col] = col_name

        # Read the CSV file with the specified types
        df = pd.read_csv(file_path, parse_dates=date_cols, dtype=type_dict, encoding=encoding, delimiter=delimiter)
        df.rename(columns=rename_dict, inplace=True)

    elif os.path.isfile(file_path + "t"):  # it's a csv with an accompanying .csvt file containing the col types
        dates, type_dict = read_types_from_csvt(file_path + "t")
        df = pd.read_csv(file_path, parse_dates=dates, dtype=type_dict, encoding=encoding, delimiter=delimiter)

    else:
        # it's a vanilla csv without col type info or we don't want to rename the column names based on curly
        # brackets flag
        df = pd.read_csv(file_path, low_memory=False, encoding=encoding, delimiter=delimiter)

    return df


def shapefile_helper(path_with_filename: str, needed_shapefile_ext: list):
    """Check whether all required Shapefile-related file extensions are present for proper reading.

    Parameters
    ----------
    path_with_filename : str
        The full path to the .shp file including its filename and extension.
    needed_shapefile_ext : list
        A list of required file extensions (e.g., ['.shp', '.shx', '.dbf']).

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If any of the required Shapefile extensions are missing.

    """
    file_name_without_ext, ext = os.path.splitext(path_with_filename)
    if ext != _SHP_EXTENSION:
        return
    # Check if all required Shapefile extensions are present
    all_files = glob(f"{file_name_without_ext}.*")
    all_shapefile_ext_present = all(any(file.endswith(ext) for file in all_files) for ext in needed_shapefile_ext)
    if not all_shapefile_ext_present:
        raise ValueError("Error: Some required Shapefile extensions are missing.")


def read_data(
    file_path: str,
    file_name: str | dict[str, str] | None = None,
    layer: str | None = None,
    sheet_name: str | int = 0,
    watch_curly_brackets: bool = True,
    encoding: str | None = None,
    delimiter: str | None = None,
    converters: dict[str, Callable[[Any], Any]] | None = None,
    fid_as_index: bool = False,
) -> pd.DataFrame | gpd.GeoDataFrame:
    """General-purpose function to read a file in various supported formats and return a DataFrame or GeoDataFrame.

    Supported formats include: pickle (.pkl, .xz), Parquet and GeoParquet (.parquet, .geoparquet),
    shapefiles (.shp), file geodatabases (.gdb), geopackages (.gpkg), CSV (.csv), and Excel (.xlsx, .xls).

    Parameters
    ----------
    file_path : str
        The directory path or full path to the file.
    file_name : Optional[Union[str, Dict[str, str]]]
        Optional filename or dictionary with keys: 'file', 'layer', and 'sheet'.
    layer : Optional[str]
        The layer to read (for geopackages or geodatabases).
    sheet_name : Union[str, int], optional
        The Excel sheet to read. Defaults to 0.
    watch_curly_brackets : bool
        Whether to interpret field types from curly-bracketed headers in typed CSVs.
    encoding : Optional[str]
        Encoding to use for CSV or text-based files.
    delimiter : Optional[str]
        Delimiter for CSV files.
    converters : Optional[Dict[str, Callable[[Any], Any]]]
        Custom converters for reading Excel files.
    fid_as_index : bool, optional
        Preserve a vector datasource's feature ID as the dataframe index.

    Returns
    -------
    Union[pd.DataFrame, gpd.GeoDataFrame]
        A DataFrame or GeoDataFrame containing the file contents.

    Raises
    ------
    ValueError
        If the file extension is unsupported or the path contains spaces.
    FileNotFoundError
        If no matching files with allowed extensions are found.

    """
    allowed_ext = [
        _PKL_EXTENSION,
        _PKL_XZ_EXTENSION,
        ".xz",
        _SHP_EXTENSION,
        ".gdb",
        _GPKG_EXTENSION,
        _CSV_EXTENSION,
        _XLSX_EXTENSION,
        _XLS_EXTENSION,
        _PARQUET_EXTENSION,
        _GEOPARQUET_EXTENSION,
    ]
    needed_shapefile_ext = [_SHP_EXTENSION, ".shx", ".prj", ".dbf"]

    if " " in file_path:
        raise ValueError("Error: The file path contains space characters, which may lead to issues.")

    # Check if the file path is a dictionary
    if isinstance(file_name, dict):
        layer = file_name.get("layer", layer)
        sheet_name = file_name.get("sheet", sheet_name)
        file_path = os.path.join(file_path, file_name["file"])
        file_name = None

    file_path = file_path.replace("\\", "/")

    # If file_name is None, split the file_path to get the filename
    if file_name is None:
        directory_path, file_name = os.path.split(file_path)
        file_path = directory_path + "/"

    # Get the file extension
    file_name_without_ext, ext = os.path.splitext(file_name)
    if not ext:
        # If no extension is provided, check for matching files with allowed extensions
        matching_files = glob(os.path.join(file_path, f"{file_name_without_ext}.*"))
        allowed_files = [f for f in matching_files if os.path.splitext(f)[1] in allowed_ext]
        if not allowed_files:
            raise FileNotFoundError(f"Error: No matching files found for: {file_name_without_ext} with accepted extension.")
        full_path = max(allowed_files, key=os.path.getmtime)
    else:
        full_path = os.path.join(file_path, file_name)

    ext = os.path.splitext(full_path)[1]
    msg = f"Reading file: {full_path}"
    if layer:
        msg += f" Layer: {layer}"
    print(msg)

    # Check gis-related files
    if ext in (_SHP_EXTENSION, _GPKG_EXTENSION, _GDB_EXTENSION):
        shapefile_helper(full_path, needed_shapefile_ext)

    readers = {
        _PKL_EXTENSION: lambda: pd.read_pickle(full_path, compression=None),  # nosec
        _XZ_EXTENSION: lambda: pd.read_pickle(full_path, compression="xz"),  # nosec
        _SHP_EXTENSION: lambda: gpd.read_file(full_path, fid_as_index=fid_as_index),
        _GPKG_EXTENSION: lambda: gpd.read_file(full_path, layer=layer, fid_as_index=fid_as_index),
        _GDB_EXTENSION: lambda: gpd.read_file(full_path, layer=layer, fid_as_index=fid_as_index),
        _CSV_EXTENSION: lambda: read_csv(full_path, watch_curly_brackets, encoding, delimiter),
        _XLSX_EXTENSION: lambda: pd.read_excel(full_path, parse_dates=True, sheet_name=sheet_name, converters=converters),
        _XLS_EXTENSION: lambda: pd.read_excel(full_path, parse_dates=True, sheet_name=sheet_name, converters=converters),
        _PARQUET_EXTENSION: lambda: read_parquet(full_path),
        _GEOPARQUET_EXTENSION: lambda: gpd.read_parquet(full_path),
    }

    reader = readers.get(ext)
    if reader:
        return reader()
    else:
        raise ValueError(f"Error: Unsupported file extension: {ext}")


def write_data(
    df: pd.DataFrame | gpd.GeoDataFrame,
    file_path: str,
    file_name: str | None = None,
    plain_csv: bool = False,
    layer: str | None = None,
) -> None:
    """General function to export a DataFrame or GeoDataFrame to a file in various formats.

    Supported output formats include: pickle (.pkl, .pkl.xz), Parquet and GeoParquet (.parquet, .geoparquet),
    shapefile (.shp), geopackage (.gpkg), CSV (.csv), and Excel (.xlsx). A GeoDataFrame written to .parquet
    is automatically stored as GeoParquet. If no extension is provided, the function defaults to .gpkg for
    GeoDataFrames and .pkl.xz for regular DataFrames.

    Parameters
    ----------
    df : Union[pd.DataFrame, gpd.GeoDataFrame]
        The DataFrame to be exported.
    file_path : str
        The directory path or full path to save the file.
    file_name : Optional[str]
        Optional file name to append to file_path. If None, file_path must include the name.
    plain_csv : bool, optional
        Indicates whether to export CSV without type info. Defaults to False.
    layer : Optional[str]
        GeoPackage layer name. Supplying a layer preserves other layers in an
        existing GeoPackage.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the file extension is not supported.

    """
    df = df.copy()
    # Check if the file path is a directory or a full path
    if file_name:
        file_path = os.path.join(file_path, file_name)
    file_path = file_path.replace("\\", "/")

    _, ext = os.path.splitext(file_path)

    # If no extension is provided, determine the default based on the DataFrame type
    if not ext:
        ext = _GPKG_EXTENSION if "geometry" in df.columns else _PKL_XZ_EXTENSION
        file_path += ext

    # If file_path exists, remove it
    if os.path.exists(file_path) and not (ext == _GPKG_EXTENSION and layer is not None):
        print(f"Removing existing file: {file_path}")
        os.remove(file_path)

    print(f"Writing file: {file_path}")

    writers = {
        _PKL_EXTENSION: lambda: df.to_pickle(file_path, compression=None),
        _PKL_XZ_EXTENSION: lambda: df.to_pickle(file_path, compression="xz"),
        _SHP_EXTENSION: lambda: write_shapefile(df, file_path),
        _GPKG_EXTENSION: lambda: write_geopackage(df, file_path, layer=layer),
        _CSV_EXTENSION: lambda: write_csv(df, file_path, plain_csv),
        _XLSX_EXTENSION: lambda: write_excel(df, file_path),
        _XLS_EXTENSION: lambda: write_excel(df, file_path),
        _PARQUET_EXTENSION: lambda: write_parquet(df, file_path),
        _GEOPARQUET_EXTENSION: lambda: write_geoparquet(df, file_path),
    }

    writer = writers.get(ext)
    if writer:
        writer()
    else:
        raise ValueError(f"Error: Unsupported file extension: {ext}")


def read_parquet(file_path: str) -> pd.DataFrame | gpd.GeoDataFrame:
    """Read Parquet data, returning a GeoDataFrame when GeoParquet metadata is present."""
    try:
        return gpd.read_parquet(file_path)
    except ValueError as error:
        if "geo metadata" not in str(error).lower():
            raise
        return pd.read_parquet(file_path)


def write_parquet(df: pd.DataFrame | gpd.GeoDataFrame, file_path: str) -> None:
    """Write a DataFrame as Parquet or a GeoDataFrame as GeoParquet."""
    df.to_parquet(file_path, index=False)


def write_geoparquet(df: gpd.GeoDataFrame, file_path: str) -> None:
    """Write a GeoDataFrame as GeoParquet."""
    if not isinstance(df, gpd.GeoDataFrame):
        raise TypeError("Error: GeoParquet output requires a GeoDataFrame.")
    df.to_parquet(file_path, index=False)


def write_shapefile(df: gpd.GeoDataFrame, file_path: str) -> None:
    """Write a GeoDataFrame to a shapefile. If the input is not a GeoDataFrame, it is saved as a CSV instead.

    Parameters
    ----------
    df : gpd.GeoDataFrame
        The GeoDataFrame to be written.
    file_path : str
        The target file path for the shapefile.

    Returns
    -------
    None

    """
    if isinstance(df, gpd.GeoDataFrame):
        #  Convert datetime columns to string format
        df_no_datetime = convert_df_col_datetime_to_string(df)
        df_no_datetime.to_file(file_path, driver="ESRI Shapefile", index=False)
    else:
        print("Cannot save a non-geodataframe as a shapefile - converting to pandas...")
        pd.DataFrame(df).to_csv(os.path.splitext(file_path)[0] + _CSV_EXTENSION, index=False)


def write_geopackage(df: gpd.GeoDataFrame, file_path: str, layer: str | None = None) -> None:
    """Write a GeoDataFrame to a GeoPackage (.gpkg) file.

    Parameters
    ----------
    df : gpd.GeoDataFrame
        The GeoDataFrame to be written.
    file_path : str
        The target file path for the GeoPackage.
    layer : Optional[str]
        Layer name to create or replace.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the input is not a GeoDataFrame.

    """
    if isinstance(df, gpd.GeoDataFrame):
        df.to_file(file_path, layer=layer, driver="GPKG", index=False)
    else:
        raise TypeError("Error: Cannot save a non-geodataframe as a geopackage.")


def write_csv(df: pd.DataFrame | gpd.GeoDataFrame, file_path: str, plain_csv: bool = False) -> None:
    """Write a DataFrame to a CSV file, optionally using a typed header format.

    Parameters
    ----------
    df : Union[pd.DataFrame, gpd.GeoDataFrame]
        The DataFrame to be written.
    file_path : str
        The target file path for the CSV.
    plain_csv : bool, optional
        If True, writes a plain CSV without type annotations in the header. Defaults to False.

    Returns
    -------
    None

    """
    df = df.copy()
    if isinstance(df, gpd.GeoDataFrame):
        # Drop the geometry column if it exists
        df = df.drop(columns=["geometry"], inplace=True, errors="ignore")
        print("Geometry column found. It will be dropped.")
    if plain_csv:
        df.to_csv(file_path, index=False)
    else:
        write_to_typed_csv(df, file_path)


def write_excel(df: pd.DataFrame | gpd.GeoDataFrame, file_path: str) -> None:
    """Write a DataFrame to an Excel (.xlsx) file. Drops the 'geometry' column if present.

    Parameters
    ----------
    df : Union[pd.DataFrame, gpd.GeoDataFrame]
        The DataFrame to be written.
    file_path : str
        The target file path for the Excel file.

    Returns
    -------
    None

    """
    if "geometry" in df.columns:
        print("Geometry column found. It will be dropped.")
        df.drop(columns="geometry", inplace=True)
    df.to_excel(file_path, header=True, index=False)


def unzip_file(zip_file_path: str, extract_to: str | None = None, logger: logging.Logger | None = None) -> str:
    """Unzip a ZIP file and extracts its contents into a structured directory.

    - If the ZIP file contains a folder matching its own name, it is extracted directly to 'extract_to'.
    - Otherwise, a new folder named after the ZIP file is created and the contents are extracted there.

    Parameters
    ----------
    zip_file_path : str
        The path to the ZIP file.
    extract_to : str, optional
        The directory where files will be extracted. If None, a directory will be created based on the ZIP file name
        in the same location.
    logger : logging.Logger
        A configured logger instance used to write the log.

    Returns
    -------
    str
        The path to the directory where the contents were extracted.

    Raises
    ------
    FileNotFoundError
        If the ZIP file does not exist.

    """
    # Ensure the provided zip file exists
    if not os.path.isfile(zip_file_path):
        raise FileNotFoundError(f"Error: The file {zip_file_path} does not exist.")

    # Get the base name of the ZIP file (without extension)
    zip_file_path = zip_file_path.replace("\\", "/")
    zip_base_name = os.path.splitext(os.path.basename(zip_file_path))[0]

    # Determine the extraction directory
    if extract_to is None:
        extract_to = os.path.join(os.path.dirname(zip_file_path), zip_base_name)
    extract_to = extract_to.replace("\\", "/")

    # Create the extraction directory if it doesn't exist
    os.makedirs(extract_to, exist_ok=True)

    # Extract the contents of the zip file
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        file_list = zip_ref.namelist()  # Get list of files/folders inside the ZIP
        # Check if the ZIP contains a folder with the same name as the ZIP file
        zip_contains_matching_folder = any(name.startswith(zip_base_name + "/") for name in file_list)

        try:
            if zip_contains_matching_folder:
                # Extract directly into 'extract_to' since it contains a matching folder
                zip_ref.extractall(extract_to)
                return extract_to
            else:
                # Create a new folder with the ZIP name inside 'extract_to'
                new_extract_folder = os.path.join(extract_to, zip_base_name)
                os.makedirs(new_extract_folder, exist_ok=True)
                zip_ref.extractall(new_extract_folder)
                return new_extract_folder
        except Exception as e:
            msg = f"Error: Failed to unzip the file: {zip_file_path}"
            log_l.log_message(logger, msg, "error")
            raise RuntimeError(msg) from e


def zip_file(folder_to_zip: str, zip_filename: str, to_delete: bool = True) -> None:
    """Compress the contents of a folder into a single ZIP file.

    Parameters
    ----------
    folder_to_zip : str
        The path to the folder whose contents will be zipped.
    zip_filename : str
        The name (including path) of the output ZIP file.
    to_delete : bool, optional
        If True, deletes the original folder after zipping. Defaults to True.

    Returns
    -------
    None

    """
    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        # Walk through the folder and add files to the ZIP archive
        for root, _, files in os.walk(folder_to_zip):
            for file in files:
                file_path = os.path.join(root, file)
                # Use a relative path to keep folder structure inside the zip archive clean
                arcname = os.path.relpath(file_path, folder_to_zip)
                zipf.write(file_path, arcname=arcname)
    print(f"Folder '{folder_to_zip}' has been zipped into '{zip_filename}'.")

    # Optionally delete the original folder after zipping
    if to_delete:
        delete_folder(folder_to_zip)
