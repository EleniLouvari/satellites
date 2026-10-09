"""Tabular-adjacent I/O helpers (model persistence, conversions, and ZIP extraction)."""

import logging
import os
import subprocess  # nosec
import zipfile

import dill  # nosec
from tqdm.notebook import tqdm

import shared.io as io_l

from .transforms import convert_columns_to_string

try:
    import fiona
except ModuleNotFoundError:  # pragma: no cover - optional dependency for GDB conversion
    fiona = None


def save_ml_model(model, file_path):
    """Save ml model."""
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
        logging.getLogger(__name__).debug("Error: save_ml_model failed; using its fallback.", exc_info=True)
        print(f"Error saving model: {model_name}")


def read_ml_model(file_path):
    """Read ml model."""
    file_path = file_path.replace("\\", "/")
    model_name = file_path.split("/")[-1]

    try:
        print(f"\nOpening model: {model_name}")
        with open(file_path, "rb") as file_obj:
            # Model artifacts are loaded from trusted project outputs.
            return dill.load(file_obj)  # nosec

    except Exception:
        logging.getLogger(__name__).debug("Error: read_ml_model failed; using its fallback.", exc_info=True)
        print(f"Error opening model: {model_name}")
    return None


def handle_xlsx_file(file_path, watch_curly_brackets, encoding):
    """Handle xlsx file."""
    out_dir = os.sep + os.path.join(*[p for p in file_path.split(os.sep)[:-1]])
    # command to turn the xlsx file into a csv file
    command = [
        "libreoffice",
        "--convert-to",
        "csv:Text - txt - csv (StarCalc):59,34,UTF8",
        file_path,
        "--outdir",
        out_dir,
    ]
    subprocess.run(command, check=True, shell=False, capture_output=True, text=True)  # nosec
    file_path = f"{os.path.splitext(file_path)[0]}.csv"
    df = io_l.read_csv(file_path, watch_curly_brackets, encoding, delimiter=";")
    os.remove(file_path)
    return df


def convert_gdb_to_gpkg(gdb_file, output_file):
    """Convert gdb to gpkg."""
    _ensure_fiona_available()
    for layer in tqdm(fiona.listlayers(gdb_file), desc="Exporting feature classes"):
        print("----------------------------------------------------------------------")
        print(f"Layer: {layer}")
        gdf = io_l.read_data(file_path=gdb_file, layer=layer)
        if not gdf.empty:
            gdf = convert_columns_to_string(gdf)
            io_l.write_data(gdf, output_file, layer=layer)
    print("Process completed successfully.")


def get_gdb_layers(gdb_file):
    """Get gdb layers."""
    _ensure_fiona_available()
    layers = list(fiona.listlayers(gdb_file))
    print(f"Layers in the {gdb_file}:\n{layers}")
    return layers


def _ensure_fiona_available():
    """Ensure Fiona is available for GDB-specific operations."""
    if fiona is None:
        raise ModuleNotFoundError("Error: Optional dependency 'fiona' is required for GDB operations.")


def extract_zip_to_same_folder(zip_path):
    # Get the folder containing the zip file
    """Extract zip to same folder."""
    extract_folder = os.path.dirname(zip_path)

    # Open the zip file and extract all files to the specified folder
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_folder)

    print(f"Files extracted to: {extract_folder}")


def unzip_gis_file(zip_file, valid_extensions, extract_to_folder=None, raise_error=True):
    """Unzip gis file."""
    found_files = []
    if not zipfile.is_zipfile(zip_file):
        return _handle_unzip_error(f"{zip_file} is not a valid ZIP file.", raise_error)
    if extract_to_folder is None:
        extract_to_folder = os.path.dirname(zip_file)

    normalized_extensions = tuple(ext.lower() for ext in valid_extensions)
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        for file_name in zip_ref.namelist():
            extracted_path = _extract_zip_member(zip_ref, extract_to_folder, file_name)
            _collect_matching_extracted_file(
                found_files=found_files,
                normalized_extensions=normalized_extensions,
                file_name=file_name,
                extracted_path=extracted_path,
                extract_to_folder=extract_to_folder,
            )

    if not found_files:
        return _handle_unzip_error(
            f"No files with extensions {valid_extensions} were found in the ZIP file.",
            raise_error,
        )
    if normalized_extensions:
        if len(found_files) > 1:
            print(f"Multiple files found with valid extensions: {found_files}. Returning the first one.")
        return found_files[0]
    return found_files


def _handle_unzip_error(message, raise_error):
    """Raise or print an unzip error depending on caller preference."""
    if raise_error:
        raise ValueError(message)
    print(message)
    return []


def _extract_zip_member(zip_ref, extract_to_folder, file_name):
    """Extract one zip member and return its full target path."""
    zip_ref.extract(file_name, extract_to_folder)
    return os.path.join(extract_to_folder, file_name)


def _collect_matching_extracted_file(
    found_files,
    normalized_extensions,
    file_name,
    extracted_path,
    extract_to_folder,
):
    """Append extracted path when it matches requested extensions."""
    if not normalized_extensions:
        found_files.append(extracted_path)
        return
    lower_name = file_name.lower()
    if lower_name.endswith(".gdb/") and ".gdb" in normalized_extensions:
        found_files.append(os.path.join(extract_to_folder, file_name.rstrip("/")))
        return
    if any(lower_name.endswith(extension) for extension in normalized_extensions):
        found_files.append(extracted_path)
