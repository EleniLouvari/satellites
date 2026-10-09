"""Library containing functions for handling raster files."""

import logging
import os
import re

# subprocess is required for controlled shell=False command execution.
import subprocess  # nosec
import time
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import botocore.exceptions
import geopandas as gpd
import numpy as np
import rasterio
import xarray as xr
from affine import Affine
from osgeo import gdal
from pyproj import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_gcps
from rasterio.warp import calculate_default_transform, reproject
from shapely.geometry import box

import shared.io as io_l
import shared.logging as log_l

# Raster-specific defaults live with their only consumer.
_BIG_TIF_YES = "BIGTIFF=YES"
_EPSG_PREFIX = "EPSG:"
BLOCKSIZE = 512
COORD_ROUNDING = 8
EPSG_4326 = "EPSG:4326"


def _read_raster_georeferencing(dataset) -> tuple:
    """Read native georeferencing, falling back to GCPs and GDAL metadata."""
    crs, transform = dataset.crs, dataset.transform
    if crs is not None and transform is not None and transform != Affine.identity():
        return crs, transform
    gcps, gcp_crs = dataset.gcps
    if gcps:
        return gcp_crs or crs, from_gcps(gcps)
    metadata = gdal.Info(dataset.name, format="json") or {}
    if crs is None:
        wkt = metadata.get("coordinateSystem", {}).get("wkt")
        if wkt:
            crs = rasterio.crs.CRS.from_wkt(wkt)
    if transform is None or transform == Affine.identity():
        geotransform = metadata.get("geoTransform")
        if geotransform is not None:
            transform = Affine.from_gdal(*geotransform)
    return crs, transform


def _get_gdal_capacity() -> tuple[int, int]:
    """Return GDAL's configured cache in MiB and the available CPU count."""
    return max(1, gdal.GetCacheMax() // (1024 * 1024)), os.cpu_count() or 1


def _extract_source_georeferencing(
    src,
    logger: logging.Logger | None = None,
) -> tuple:
    """Extract CRS and transform from source raster, handling JP2 files with missing geotransforms.

    Parameters
    ----------
    src : rasterio.DatasetReader
        Open rasterio dataset for the source raster.
    logger : logging.Logger, optional
        Logger for messages.

    Returns
    -------
    tuple
        (src_crs, src_transform) extracted from raster or metadata.
    """
    src_crs = src.crs
    src_transform = src.transform

    # Recover missing native georeferencing from GCPs or GDAL metadata.
    if src_crs is None or src_transform == Affine.identity() or src_transform is None:
        src_crs, src_transform = _read_raster_georeferencing(src)
        if src_transform is not None and src_transform != Affine.identity():
            log_l.log_message(
                logger,
                f"Extracted georeferencing from {os.path.basename(src.name)} metadata (JP2 or similar format)",
            )

    # If still no valid transform, use identity
    if src_transform is None or src_transform == Affine.identity():
        src_transform = Affine.identity()
        log_l.log_message(logger, f"Using identity transform for {os.path.basename(src.name)}")

    return src_crs, src_transform


def _reproject_block_windowed(
    src,
    dst,
    dst_transform,
    dst_crs,
    src_crs,
    src_transform,
    resampling,
    fill_value,
) -> None:
    """Reproject source raster to destination using block-based windowing.

    Parameters
    ----------
    src : rasterio.DatasetReader
        Source raster dataset.
    dst : rasterio.DatasetWriter
        Destination raster dataset.
    dst_transform : Affine
        Destination georeferencing transform.
    dst_crs : CRS
        Destination coordinate reference system.
    src_crs : CRS
        Source coordinate reference system.
    src_transform : Affine
        Source georeferencing transform.
    resampling : rasterio.enums.Resampling
        Resampling method.
    fill_value : float
        Fill value for output.
    """
    for i in range(1, src.count + 1):
        # Process in blocks to reduce memory usage
        for _, window in dst.block_windows(1):
            # Calculate corresponding window in destination
            dst_window_transform = rasterio.windows.transform(window, dst_transform)

            # Create destination block array
            data = np.full((window.height, window.width), fill_value, dtype=src.dtypes[0])

            # Reproject only this block with explicit CRS handling
            reproject(
                source=rasterio.band(src, i),
                destination=data,
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=dst_window_transform,
                dst_crs=dst_crs,
                resampling=resampling,
                src_nodata=fill_value,
            )

            # Write block
            dst.write(data, i, window=window)

            # Clean up block array
            del data


def _normalize_bbox(bbox: Sequence[float]) -> tuple[float, float, float, float]:
    """Return bbox as (x_min, y_min, x_max, y_max) with valid ordering."""
    b = tuple(bbox)
    if len(b) != 4:
        raise ValueError("Error: bbox must have 4 elements: (x_min, y_min, x_max, y_max)")

    x_min, y_min, x_max, y_max = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))

    if x_min > x_max:
        x_min, x_max = x_max, x_min
    if y_min > y_max:
        y_min, y_max = y_max, y_min

    if x_min == x_max or y_min == y_max:
        raise ValueError("Error: Invalid bbox: zero width or height.")

    return x_min, y_min, x_max, y_max


def _crop_with_gdal_translate(
    x_min: float,
    y_max: float,
    x_max: float,
    y_min: float,
    creation_options: Sequence[str],
    input_file: str,
    output_file: str,
    logger: logging.Logger | None = None,
) -> None:
    """Crop a raster to bbox using gdal.Translate (fast crop without padding/resampling).

    Parameters
    ----------
    x_min, y_max, x_max, y_min : float
        Crop window in source CRS. Note GDAL Translate expects projWin order:
        (ulx, uly, lrx, lry) which maps to (x_min, y_max, x_max, y_min).
    creation_options : Sequence[str]
        GTiff creation options (e.g., tiling, compression).
    input_file : str
        Path to input raster.
    output_file : str
        Path to output cropped raster.
    logger : logging.Logger | None, optional
        Logger for messages.

    Raises
    ------
    RuntimeError
        If gdal.Translate fails.
    """
    log_l.log_message(logger, f"Cropping file: {os.path.basename(input_file)}...")
    translate_options = gdal.TranslateOptions(
        projWin=[x_min, y_max, x_max, y_min],
        format="GTiff",
        creationOptions=list(creation_options),
    )
    out_ds = gdal.Translate(output_file, input_file, options=translate_options)
    if out_ds is None:
        raise RuntimeError("Error: gdal.Translate failed.")

    out_ds.FlushCache()
    out_ds = None
    log_l.log_message(logger, f"Success: Raster cropped and saved to: {os.path.basename(output_file)}")


def _crop_with_gdal_warp(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    creation_options: Sequence[str],
    input_file: str,
    output_file: str,
    fill_value: float,
    use_fill_as_nodata: bool,
    logger: logging.Logger | None = None,
) -> None:
    """Crop and pad a raster to bbox using gdal.Warp (forces exact output bounds).

    This path is used when you want the output raster to have the *exact* bbox extent,
    even if parts of that bbox fall outside the source raster. Pixels not covered by
    the source are initialized to `fill_value` via INIT_DEST.

    Parameters
    ----------
    x_min, y_min, x_max, y_max : float
        Target bbox in source CRS as (x_min, y_min, x_max, y_max).
    creation_options : Sequence[str]
        GTiff creation options (e.g., tiling, compression).
    input_file : str
        Path to input raster.
    output_file : str
        Path to output cropped/padded raster.
    fill_value : float
        Value used to fill pixels not covered by the source raster.
    use_fill_as_nodata : bool
        If True, set dstNodata to fill_value. Otherwise keep source nodata if present.
    logger : logging.Logger | None, optional
        Logger for messages.

    Raises
    ------
    RuntimeError
        If the input cannot be opened, lacks geotransform, or gdal.Warp fails.
    """
    log_l.log_message(logger, f"Cropping file: {os.path.basename(input_file)}...")
    src_ds = gdal.Open(input_file, gdal.GA_ReadOnly)
    if src_ds is None:
        raise RuntimeError(f"Error: Could not open input raster {os.path.basename(input_file)}.")

    try:
        gt = src_ds.GetGeoTransform(can_return_null=True)
        if gt is None:
            raise RuntimeError(f"Error: Input raster {os.path.basename(input_file)} has no geotransform.")

        x_res = float(gt[1])
        y_res = abs(float(gt[5]))

        src_band = src_ds.GetRasterBand(1)
        src_nodata = src_band.GetNoDataValue() if src_band else None

        warp_kwargs = {
            "format": "GTiff",
            "outputBounds": (x_min, y_min, x_max, y_max),
            "xRes": x_res,
            "yRes": y_res,
            "resampleAlg": gdal.GRA_NearestNeighbour,
            "multithread": True,
            "warpOptions": [f"INIT_DEST={float(fill_value)}", "NUM_THREADS=ALL_CPUS"],
            "creationOptions": list(creation_options),
        }

        if use_fill_as_nodata:
            warp_kwargs["dstNodata"] = float(fill_value)
        elif src_nodata is not None:
            warp_kwargs["dstNodata"] = src_nodata

        out_ds = gdal.Warp(output_file, src_ds, **warp_kwargs)
        if out_ds is None:
            raise RuntimeError("Error: gdal.Warp failed.")

        out_ds.FlushCache()
        out_ds = None

    finally:
        src_ds = None

    log_l.log_message(logger, f"Success: Raster cropped/padded and saved to: {os.path.basename(output_file)}")


def crop_raster(
    input_file: str,
    output_file: str,
    bbox: Iterable[float],
    fill_value: float | None = None,
    use_fill_as_nodata: bool = False,
    logger: logging.Logger | None = None,
    tile_output: bool = True,
) -> None:
    """Crop (and, if needed, pad) a raster to a bbox. Areas outside the source are filled."""
    try:
        x_min, y_min, x_max, y_max = _normalize_bbox(bbox)

        # Detect input dtype to select appropriate PREDICTOR
        with rasterio.open(input_file) as src:
            is_float = np.issubdtype(src.dtypes[0], np.floating)

        tiling_option = "TILED=YES" if tile_output else "TILED=NO"
        predictor = "3" if is_float else "2"
        creation_options = [tiling_option, "COMPRESS=DEFLATE", f"PREDICTOR={predictor}", "BIGTIFF=IF_SAFER"]

        if fill_value is None:
            _crop_with_gdal_translate(x_min, y_max, x_max, y_min, creation_options, input_file, output_file, logger)
        else:
            _crop_with_gdal_warp(
                x_min,
                y_min,
                x_max,
                y_max,
                creation_options,
                input_file,
                output_file,
                float(fill_value),
                use_fill_as_nodata,
                logger,
            )

    except Exception as e:
        raise RuntimeError(f"Error: Raster {os.path.basename(input_file)} could not be cropped/padded: {e}") from e


def projection_is_epsg_or_wkt(projection_string: str) -> str:
    """Check whether the given projection string is an EPSG code or WKT format.

    Parameters
    ----------
    projection_string : str
        The projection string to check.

    Returns
    -------
    str
        "EPSG" if it is an EPSG code, "WKT" if it is WKT format, "UNKNOWN" otherwise.

    """
    # Check if the projection string starts with "EPSG:"
    if projection_string.upper().startswith(_EPSG_PREFIX):
        return "EPSG"
    elif any(keyword in projection_string for keyword in ["GEOGCS", "PROJCS", "DATUM", "SPHEROID", "AUTHORITY"]):
        return "WKT"
    else:
        return "UNKNOWN"


def reproject_bbox(bbox: Iterable[float], source_crs: str, target_crs: str) -> list:
    """Reproject a bounding box to a target CRS.

    Parameters
    ----------
    bbox : tuple
        The original bounding box in the format (min_lon, min_lat, max_lon, max_lat).
    source_crs : str or int
        The CRS of the input bounding box (e.g., EPSG:4326).
    target_crs : str or int
        The CRS to reproject to (e.g., EPSG:2100).

    Returns
    -------
    tuple
        The reprojected bounding box in the format (min_lon, min_lat, max_lon, max_lat).

    """
    # Create a Shapely box geometry
    bbox_geom = box(*bbox)
    # Create a GeoDataFrame from the initial bbox
    gdf = gpd.GeoDataFrame({"geometry": [bbox_geom]}, crs=source_crs)

    # Reproject the GeoDataFrame to the target CRS
    gdf_reprojected = gdf.to_crs(target_crs)

    # Get the reprojected bounding box
    reprojected_bbox = list(gdf_reprojected.total_bounds)  # Returns minx, miny, maxx, maxy
    return reprojected_bbox


def define_resampling_algorithm(resample_alg: str = "nearest", for_library: str = "rasterio"):
    """Return the corresponding Rasterio resampling method based on the input string.

    Parameters
    ----------
    resample_alg : str
        The name of the resampling algorithm (e.g., 'nearest', 'bilinear', 'cubic'). Defaults to 'nearest'. Case-insensitive.

    Returns
    -------
    rasterio.enums.Resampling
        The corresponding Resampling method. Defaults to Resampling.nearest if not recognized.

    """
    if for_library == "rasterio":
        resampling_methods = {
            "nearest": Resampling.nearest,
            "bilinear": Resampling.bilinear,
            "cubic": Resampling.cubic,
            "cubic_spline": Resampling.cubic_spline,
            "lanczos": Resampling.lanczos,
            "average": Resampling.average,
            "mode": Resampling.mode,
            "max": Resampling.max,
            "min": Resampling.min,
            "med": Resampling.med,
            "q1": Resampling.q1,
            "q3": Resampling.q3,
        }
        # Default to nearest if an invalid method is provided (fail-safe)
        if resample_alg.lower() not in resampling_methods:
            log_l.log_message(None, f"Invalid resampling method '{resample_alg}'. Falling back to 'nearest'.")
            return Resampling.nearest
        return resampling_methods[resample_alg.lower()]
    elif for_library == "gdal":
        resample_alg = "near" if resample_alg == "nearest" else resample_alg
        if resample_alg not in [
            "near",
            "bilinear",
            "cubic",
            "cubicspline",
            "lanczos",
            "average",
            "rms",
            "mode",
            "max",
            "min",
            "med",
            "q1",
            "q3",
            "sum",
        ]:
            raise ValueError(f"Error: Not valid resample_alg: {resample_alg}")
        return resample_alg
    else:
        raise ValueError(f"Error: Not valid input 'for_library' variable to define the {resample_alg}; use 'rasterio' or 'gdal'!")


def _extract_reference_georeferencing(ref) -> tuple[int, int, Any, Any]:
    """Extract destination width/height/CRS/transform from a reference raster."""
    dst_width, dst_height = ref.width, ref.height
    dst_crs, dst_transform = ref.crs, ref.transform

    if dst_crs is None or dst_transform is None or dst_transform == Affine.identity():
        dst_crs, dst_transform = _read_raster_georeferencing(ref)
        if dst_transform is None or dst_transform == Affine.identity():
            raise ValueError("Error: Could not determine valid georeferencing from raster tags.")

    if dst_crs is None or dst_transform is None:
        raise ValueError("Error: Could not determine valid georeferencing (CRS or transform) from GeoTIFF.")

    return dst_width, dst_height, dst_crs, dst_transform


def _resolve_nodata_and_fill(src, nodata_value: int | None, apply_mask: bool, mask_value: float | None) -> tuple:
    """Resolve output nodata and fill value used during reprojection."""
    src_nodata = src.nodata
    final_nodata = nodata_value if nodata_value is not None and not np.isnan(nodata_value) else src_nodata
    fill_value = mask_value if apply_mask else final_nodata
    return final_nodata, fill_value


def _build_reference_resample_profile(
    ref,
    src,
    dst_width: int,
    dst_height: int,
    dst_transform,
    dst_crs,
    final_nodata,
) -> dict:
    """Build output profile for resampling to a reference grid."""
    profile = ref.profile.copy()
    profile.update(
        {
            "dtype": src.dtypes[0],
            "count": src.count,
            "driver": "GTiff",
            "width": dst_width,
            "height": dst_height,
            "transform": dst_transform,
            "crs": dst_crs,
            "nodata": final_nodata,
            "tiled": True,
            "blockxsize": BLOCKSIZE,
            "blockysize": BLOCKSIZE,
            "BIGTIFF": "YES",
        }
    )
    return profile


def resample_to_reference(
    input_file: str,
    reference_file: str,
    output_file: str,
    resample_alg: str = "nearest",
    nodata_value: int | None = 0,
    apply_mask: bool | None = False,
    mask_value: float | None = 0,
    logger: logging.Logger | None = None,
) -> None:
    """Resample a raster to match the resolution, extent, CRS, and shape of a reference raster.

    This function uses rasterio's 'reproject' to resample an input raster so that it aligns exactly
    with a reference raster. It ensures that the output has the same pixel resolution, coordinate reference
    system, bounding box, and dimensions (rows x columns) as the reference.

    Uses block-based processing to minimize memory usage for large rasters.

    Parameters
    ----------
    input_file : str
        Path to the source raster to be resampled.
    reference_file : str
        Path to the reference raster whose grid geometry (transform, resolution, CRS, and size) will be matched.
    output_file : str
        Path where the resampled and aligned raster will be saved.
    resample_alg : str, optional
        Resampling method to use. Supported values include:
        "nearest", "bilinear", "cubic", "cubic_spline", "lanczos", "average", "mode", "max", "min", "med", "q1", "q3".
        Default is "nearest".
    nodata_value : int, optional
        Value to use as NoData in the output raster. Defaults to 0.
    apply_mask : bool, optional
        Whether to fill the new dataset with the mask_value. Defaults to False.
    mask_value : float, optional
        If apply_mask is True then use mask_value to fill the new dataset.
    logger : Optional[logging.Logger]
        A logger instance to log the message. If None, prints to the console.

    Returns
    -------
    None

    Notes
    -----
    - The output raster will use the data type of the input raster and the spatial profile of the reference raster.
    - This is especially useful for aligning multi-resolution Sentinel-2 bands (10m, 20m, 60m) to a common 10m grid.

    Raises
    ------
    AttributeError
        If the specified resampling algorithm name is not valid.
    """
    try:
        log_l.log_message(logger, f"Resampling {os.path.basename(input_file)} to match {os.path.basename(reference_file)})..")
        resampling = define_resampling_algorithm(resample_alg=resample_alg, for_library="rasterio")

        with rasterio.open(reference_file) as ref:
            dst_width, dst_height, dst_crs, dst_transform = _extract_reference_georeferencing(ref)

            with rasterio.open(input_file) as src:
                final_nodata, fill_value = _resolve_nodata_and_fill(src, nodata_value, apply_mask, mask_value)

                # Extract source georeferencing (handles JP2 files)
                src_crs, src_transform = _extract_source_georeferencing(src, logger)

                profile = _build_reference_resample_profile(
                    ref=ref,
                    src=src,
                    dst_width=dst_width,
                    dst_height=dst_height,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    final_nodata=final_nodata,
                )

                # Perform block-based reprojection
                with rasterio.open(output_file, "w", **profile) as dst:
                    _reproject_block_windowed(
                        src=src,
                        dst=dst,
                        dst_transform=dst_transform,
                        dst_crs=dst_crs,
                        src_crs=src_crs,
                        src_transform=src_transform,
                        resampling=resampling,
                        fill_value=fill_value,
                    )

        log_l.log_message(logger, f"Resampled to match {reference_file} and saved: {output_file}")

    except Exception as e:
        raise RuntimeError(f"Error: Could not resample raster {os.path.basename(input_file)}: {e}") from e


def resample_raster(
    input_file: str,
    output_file: str,
    target_resolution: Sequence[float] | Sequence[int],
    resample_alg: str = "nearest",
    logger: logging.Logger | None = None,
) -> None:
    """Resample a raster to a specified spatial resolution using block-based processing.

    Uses windowed reading and writing to minimize memory usage for large rasters.

    Parameters
    ----------
    input_file : str
        Path to the input raster file.
    output_file : str
        Path where the resampled raster will be saved.
    target_resolution : Union[Sequence[float], Sequence[int]]
        Target resolution as (x_res, y_res) in CRS units.
    resample_alg : str
        Resampling algorithm to use (e.g., "nearest", "bilinear", "cubic"). Defaults to "nearest".
    logger : Optional[logging.Logger]
        A logger instance to log the message. If None, prints to the console.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If target resolution is invalid.
    FileNotFoundError
        If input file does not exist.
    RuntimeError
        If resampling fails.
    """
    # Validate target resolution
    if not isinstance(target_resolution, (tuple, list)) or len(target_resolution) != 2:
        raise ValueError("Error: Invalid target resolution. Provide (x_res, y_res) as a tuple or list.")
    if any(res <= 0 for res in target_resolution):
        raise ValueError("Error: Target resolution values must be positive.")

    # Validate input file
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Error: Input file {input_file} does not exist.")

    # Create output directory
    output_dir = os.path.dirname(os.path.abspath(output_file))
    io_l.create_folder(output_dir)

    try:
        resampling = define_resampling_algorithm(resample_alg=resample_alg, for_library="rasterio")

        with rasterio.open(input_file) as src:
            src_crs, original_transform = _extract_source_georeferencing(src, logger)

            # Calculate scaling factors
            original_x_res = original_transform.a
            original_y_res = abs(original_transform.e)

            scale_x = original_x_res / target_resolution[0]
            scale_y = original_y_res / target_resolution[1]

            # Calculate new dimensions
            new_width = round(src.width * scale_x)
            new_height = round(src.height * scale_y)

            # Calculate new transform
            new_transform = original_transform * original_transform.scale((src.width / new_width), (src.height / new_height))

            # Update profile for output
            profile = src.profile.copy()
            profile.update(
                {
                    "crs": src_crs,
                    "transform": new_transform,
                    "width": new_width,
                    "height": new_height,
                    "driver": "GTiff",
                    "tiled": True,
                    "blockxsize": BLOCKSIZE,
                    "blockysize": BLOCKSIZE,
                    "BIGTIFF": "YES",
                }
            )

            # Create output file and process band by band with windowing
            with rasterio.open(output_file, "w", **profile) as dst:
                # Define block size for reading (use smaller blocks for large scale changes)
                read_block_size = 2048

                for band_idx in range(1, src.count + 1):
                    # Process in blocks across the destination raster
                    for col_off in range(0, new_width, read_block_size):
                        for row_off in range(0, new_height, read_block_size):
                            # Calculate block dimensions
                            block_width = min(read_block_size, new_width - col_off)
                            block_height = min(read_block_size, new_height - row_off)

                            # Create destination window
                            dst_window = rasterio.windows.Window(col_off, row_off, block_width, block_height)

                            # Calculate corresponding source window
                            src_window = rasterio.windows.from_bounds(
                                *rasterio.windows.bounds(dst_window, new_transform), transform=src.transform
                            )

                            # Read source data for this window with resampling
                            data = src.read(
                                band_idx, window=src_window, out_shape=(block_height, block_width), resampling=resampling
                            )

                            # Write resampled block
                            dst.write(data, band_idx, window=dst_window)

                            # Clean up
                            del data

        log_l.log_message(
            logger, f"Success: Raster resampled at {target_resolution} and saved to: {os.path.basename(output_file)}"
        )

    except Exception as e:
        raise RuntimeError(f"Error: Could not resample raster {os.path.basename(input_file)}: {e}") from e


def reproject_raster_rasterio(
    input_file: str,
    output_file: str,
    target_crs: str | dict,
    target_resolution: int | tuple[float, float] | list[float] | None = None,
    resample_alg: str = "nearest",
    logger: logging.Logger | None = None,
) -> None:
    """Reproject a multi-band raster to a target CRS and optionally resamples it to a specified resolution.

    Parameters
    ----------
    input_file : str
        Path to the input raster file.
    output_file : str
        Path where the reprojected raster will be saved.
    target_crs : Union[str, dict])
        The target coordinate reference system (e.g., 'EPSG:4326' or a CRS dict).
    target_resolution : Optional[Union[int, Tuple[float, float], List[float]]])
        Target resolution as a scalar or tuple (x_res, y_res).
    resample_alg : str, optional
        Resampling algorithm to use (e.g., "nearest", "bilinear", "cubic"). Defaults to "nearest".
    logger : Optional[logging.Logger]
        A logger instance to log the message. If None, prints to the console.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If the reprojection fails due to an error in input parameters or raster processing.

    """
    try:
        # Default to bilinear if an invalid method is provided
        resampling = define_resampling_algorithm(resample_alg=resample_alg, for_library="rasterio")

        with rasterio.open(input_file) as src:
            profile = src.profile.copy()
            # Check if the input file is a ENVI and set the driver accordingly
            if "ENVI" in profile.get("driver", "").upper():
                profile.pop("BLOCKXSIZE", None)
                profile.pop("BLOCKYSIZE", None)

            # Ensure JP2 files are explicitly saved as GTiff before reprojecting
            if str(input_file).lower().endswith(".jp2"):
                profile["driver"] = "GTiff"  # Force GeoTIFF output
                profile.pop("INTERLEAVE", None)  # Remove unsupported INTERLEAVE option

            # Determine resolution
            if target_resolution and isinstance(target_resolution, (tuple, list)) and len(target_resolution) == 2:
                x_res, y_res = target_resolution
            elif target_resolution and isinstance(target_resolution, int):
                x_res, y_res = target_resolution, target_resolution
            else:
                x_res, y_res = src.res  # Default to source resolution

            # Calculate the new transform, width, and height
            transform, width, height = calculate_default_transform(
                src.crs, target_crs, src.width, src.height, *src.bounds, resolution=x_res
            )

            # Update profile with new CRS, transform, and resolution
            profile.update(
                {
                    "crs": target_crs,
                    "transform": transform,
                    "width": width,
                    "height": height,
                    "count": src.count,  # Ensure multi-band support
                }
            )

            # Create output raster and reproject each band
            with rasterio.open(output_file, "w", **profile) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=target_crs,
                        resampling=resampling,
                    )

        msg = f"Success: {os.path.basename(input_file)} reprojected to {os.path.basename(output_file)}"
        msg = f"{msg} at {x_res}x{y_res} resolution."
        log_l.log_message(logger, msg)
    except Exception as e:
        raise RuntimeError(f"Error: Could not reproject raster {os.path.basename(input_file)}: {e}") from e


def run_subprocess(cmd: list[str], wait_file: str | None = None, logger: logging.Logger | None = None):
    """Execute a subprocess command and print its output.

    Optionally waits for a specific file to become accessible after execution.

    Parameters
    ----------
    cmd : list[str]
        The command to execute as a list of strings (e.g., ['gdalwarp', '-t_srs', ...]).
    wait_file : str, optional
        If provided, waits for the file at this path to become accessible after the subprocess completes.
    logger : logging.Logger
        A logger instance to log the message. If None, prints to the console.

    Raises
    ------
    Exception
        If the subprocess returns a non-zero exit code or if the specified file remains inaccessible after execution.

    """
    log_l.log_message(logger, f"Executing command: {' '.join(cmd)}")
    cmd_command = cmd[0]
    try:
        result = subprocess.run(  # nosec
            cmd,
            check=True,
            shell=False,
            capture_output=True,
            text=True,
            timeout=3600,
        )
        log_l.log_message(logger, f"{cmd_command} STDOUT: {result.stdout}")

        # Ensure the output file is not locked
        if wait_file:
            for _ in range(10):  # Try up to 10 times
                try:
                    with open(wait_file, "rb"):
                        break
                except Exception:
                    logging.getLogger(__name__).debug("Error: run_subprocess failed; using its fallback.", exc_info=True)
                    time.sleep(0.2)
            else:
                raise RuntimeError(f"Error: File still locked after subprocess: {wait_file}")

        log_l.log_message(logger, f"{cmd_command} Command executed successfully")
    except subprocess.CalledProcessError as e:
        log_l.log_message(logger, f"{cmd_command} failed with return code {e.returncode}", type="error")
        log_l.log_message(logger, f"{cmd_command} STDOUT:{e.stdout}")
        log_l.log_message(logger, f"{cmd_command} STDERR:{e.stderr}")
        raise RuntimeError(f"Error: {cmd_command} failed: {e}") from e


def compute_slope_from_dem(dem_path: str, slope_path: str) -> None:
    """Compute slope (in degrees) from a DEM using gdaldem.

    Parameters
    ----------
    dem_path : str
        Path to the input DEM file.
    slope_path : str
        Path to the output slope file.

    """
    cmd = ["gdaldem", "slope", dem_path, slope_path, "-alg", "Horn", "-compute_edges", "-of", "GTiff"]
    run_subprocess(cmd)


def compute_aspect_from_dem(dem_path: str, aspect_path: str) -> None:
    """Compute aspect (in degrees from north, clockwise) from a DEM using gdaldem.

    Parameters
    ----------
    dem_path : str
        Path to the input DEM file.
    aspect_path : str
        Path to the output aspect file.

    """
    cmd = ["gdaldem", "aspect", dem_path, aspect_path, "-alg", "Horn", "-compute_edges", "-of", "GTiff"]
    run_subprocess(cmd)


def reproject_raster_with_gdalwarp(
    input_file: str,
    output_file: str,
    target_crs: str,
    target_resolution: float | tuple[float, float] | None = None,
    source_crs: str | None = None,
    resample_alg: str = "near",
    nodata_value: int | None = None,
    logger: logging.Logger | None = None,
) -> None:
    """Reproject a raster horizontally using gdalwarp.

    Parameters
    ----------
    input_file : str
        Path to the input raster file.
    output_file : str
        Path to the reprojected output file.
    target_crs : str
        Target CRS in Proj4 or EPSG format.
    target_resolution : int | float | tuple[float, float], optional
        Resolution in target CRS units (e.g., meters for projected CRS).
    source_crs : str
        Source CRS in Proj4 or EPSG format.
    resample_alg : str, default="near"
        Resampling algorithm (e.g., "near", "bilinear").
    nodata_value : float, optional
        Value to use as NoData in the output raster. Defaults to None.
    logger : Optional[logging.Logger]
        A logger instance to log the message. If None, prints to the console.

    Raises
    ------
    ValueError
        If target_resolution is invalid.
    Exception
        If gdalwarp fails or required environment variables or files are missing.

    """
    try:
        # Check the resampling algorithm
        resampling = define_resampling_algorithm(resample_alg=resample_alg, for_library="gdal")

        # Delete output file if existing, otherwise gdal raises error
        if io_l.file_exists(output_file, verbose=False):
            io_l.delete_file(output_file)

        # Format output crs in the form gdal accepts
        to_crs = f"EPSG:{parse_epsg_code(target_crs)}"

        # Get computer's capabilities
        gdal_cachemax_mb, num_threads = _get_gdal_capacity()

        cmd = [
            "gdalwarp",
            "-t_srs",
            to_crs,
            "-r",
            resampling,
            "-of",
            "GTiff",
            "-co",
            _BIG_TIF_YES,
            "-multi",
            "-wm",
            "4096",
            "--config",
            "GDAL_NUM_THREADS",
            str(num_threads),
            "--config",
            "GDAL_CACHEMAX",
            str(gdal_cachemax_mb),
        ]
        if nodata_value is not None:
            cmd += ["-dstnodata", str(nodata_value)]

        if source_crs:
            from_crs = f"EPSG:{parse_epsg_code(source_crs)}"
            cmd += ["-s_srs", from_crs]

        if target_resolution is not None:
            if isinstance(target_resolution, (int, float)):
                x_res = y_res = float(target_resolution)
            elif isinstance(target_resolution, (tuple, list)) and len(target_resolution) == 2:
                x_res, y_res = map(float, target_resolution)
            else:
                raise ValueError(f"Error: Invalid target_resolution: {target_resolution}")
            cmd += ["-tr", str(x_res), str(y_res)]
            log_l.log_message(
                logger,
                f"Reprojecting {os.path.basename(input_file)} horizontally to {target_crs} at {x_res}x{y_res} resolution...",
            )
        else:
            log_l.log_message(
                logger, f"Reprojecting {os.path.basename(input_file)} horizontally to {target_crs} with native resolution..."
            )

        cmd += [input_file.replace("\\", "/"), output_file.replace("\\", "/")]
        run_subprocess(cmd, wait_file=output_file.replace("\\", "/"), logger=logger)
        log_l.log_message(logger, f"Success: {os.path.basename(input_file)} reprojected to {os.path.basename(output_file)}\n")
    except Exception as e:
        raise RuntimeError(f"Error: Could not reproject raster {os.path.basename(input_file)}: {e}") from e


def vertical_transformation_with_geoid_file(
    dem_file: str,
    output_file: str,
    source_crs: str = EPSG_4326,
    resample_alg: str = "nearest",
    nodata_value: int = -9999,
    apply_mask: bool = False,
    mask_value: int = 0,
    logger: logging.Logger | None = None,
) -> None:
    """Convert geoid-based (orthometric) DEM heights to ellipsoidal heights (h = H + N).

    This function uses the environmental variable DEM_VERTICAL_EGM, which should point to
    an EGM2008 geoid raster (undulation N, in meters). Both the DEM and EGM rasters must
    be in EPSG:4326.

    The transformation assumes:
        DEM values = H (orthometric heights, EGM2008)
        EGM raster values = N (geoid undulation, EGM2008)
        Output values = h = H + N (ellipsoidal heights).

    Parameters
    ----------
    dem_file : str
        Path to the input DEM raster (orthometric heights, WGS84 + EGM2008).
    output_file : str
        Path where the ellipsoidal-height DEM will be saved.
    source_crs : str, optional
        Horizontal CRS of the input raster. Must be EPSG:4326.
    resample_alg : str, optional
        Resampling algorithm used to resample the EGM raster to the DEM grid.
    nodata_value : int, optional
        Value to use as NoData in the output raster. Defaults to -9999.
    apply_mask : bool, optional
        Whether to mask specific values equal to mask_value in the EGM raster
        (treated as nodata). Defaults to False. Normally leave this False for EGM.
    mask_value : int, optional
        Value in the EGM raster to treat as nodata when apply_mask is True.
    logger : Optional[logging.Logger]
        Logger instance. If None, messages are printed.

    Raises
    ------
    RuntimeError
        If the function fails or required environment variables/files are missing.

    """
    log_l.log_message(logger, "Applying vertical transformation (EGM → ellipsoid)...")

    horizontal_epsg = parse_epsg_code(source_crs)
    if str(horizontal_epsg) != "4326":
        raise ValueError(f"Error: Invalid source projection: {source_crs}; both DEM and EGM rasters must be in EPSG:4326.")

    # Validate geoid file
    source_geoid_file = os.getenv("DEM_VERTICAL_EGM")
    if not source_geoid_file:
        raise ValueError("Error: Source geoid file is missing; set the 'DEM_VERTICAL_EGM' environment variable.")
    if not io_l.file_exists(source_geoid_file):
        raise FileNotFoundError(f"Error: Source geoid file not found: {source_geoid_file}")

    try:
        # Resample/crop the EGM to the DEM grid
        dem_path, _, _ = io_l.separate_path_filename_extension(dem_file)
        crop_geoid_file = os.path.join(dem_path, os.path.basename(source_geoid_file))

        resample_to_reference(
            input_file=source_geoid_file,
            reference_file=dem_file,
            output_file=crop_geoid_file,
            resample_alg=resample_alg,
            nodata_value=nodata_value,
            apply_mask=apply_mask,
            mask_value=mask_value,
            logger=logger,
        )

        # Ensure output directory exists
        output_dir = os.path.dirname(os.path.abspath(output_file))
        io_l.create_folder(output_dir)

        # Add EGM raster to the DEM to produce ellipsoidal heights
        math_rasters(
            raster1_path=dem_file,
            raster2_path=crop_geoid_file,
            output_raster_path=output_file,
            operation="add",
            logger=logger,
        )

    except Exception as e:
        geoid_name = os.path.basename(source_geoid_file) if source_geoid_file else "UNKNOWN"
        raise RuntimeError(
            f"Error: Could not apply vertical transformation to raster {os.path.basename(dem_file)} "
            f"using EGM raster: {geoid_name} ({e})"
        ) from e


def vertical_transformation_with_gdalwarp(
    input_file: str,
    output_file: str,
    source_crs: str,
    nodata_value: int = -9999,
    geoid: str = "ESA",
    logger: logging.Logger | None = None,
) -> None:
    """Apply vertical datum transformation to convert geoid-based (orthometric) heights to ellipsoidal heights.

    This function supports multiple geoid models (e.g., EGM2008, HEPOS) and performs the transformation using GDAL.
    - If the geoid model is recognized and supported as a compound EPSG code, it is used directly.
    - Otherwise, the geoid file is copied (if needed) into the GDAL/PROJ data path, and a +geoidgrids transformation is applied.

    Parameters
    ----------
    input_file : str
        Path to the input raster file in WGS84 with geoid-based heights.
    output_file : str
        Path where the vertically transformed raster (ellipsoidal heights) will be saved.
    source_crs : str
        The CRS of the input raster (e.g., EPSG:4326).
    nodata_value : int, optional
        Value to use as NoData in the output raster. Defaults to -9999.
    geoid : str, optional
        Select the appropriate geoid based in the input dem. Defaults to ESA.
    logger : Optional[logging.Logger]
        A logger instance to log the message. If None, prints to the console.

    Raises
    ------
    Exception
        If gdalwarp fails or required environment variables or files are missing.

    """
    try:
        log_l.log_message(logger, "Applying vertical transformation...")
        horizontal_epsg = parse_epsg_code(source_crs)

        # Define mapping of horizontal CRS + geoid to compound EPSG codes
        # compound_crs_mapping = {
        #     ("ESA", 4326): "EPSG:4326+5773",  # WGS84 + EGM2008 geoid height (orthometric)
        #     ("ESA", 3857): "EPSG:3857+5773",  # Web Mercator + EGM2008 geoid height
        #     ("ESA", 3395): "EPSG:3395+5773",  # World Mercator + EGM2008 geoid height
        # }
        compound_crs_mapping = {
            ("ESA", 4326): "EPSG:9518",  # WGS84 + EGM2008
            ("ESA", 3857): "EPSG:6871",  # Web Mercator + EGM2008
            ("ESA", 3395): "EPSG:6893",  # World Mercator + EGM2008
        }

        # Try to find a compound EPSG code
        compound_key = (geoid, horizontal_epsg)
        if compound_key in compound_crs_mapping:
            source_srs_str = compound_crs_mapping[compound_key]
        else:
            source_geoid_file = os.getenv("DEM_VERTICAL_DATUM")

            # Validate and copy geoid file
            if not io_l.file_exists(source_geoid_file):
                raise FileNotFoundError(f"Error: Source geoid file not found: {source_geoid_file}")

            # Build source SRS string (assuming WGS84 with geoid correction)
            source_srs_str = f"EPSG:{horizontal_epsg}+geoidgrids={source_geoid_file}"

        # Get computer's capabilities
        gdal_cachemax_mb, num_threads = _get_gdal_capacity()

        # Target is standard WGS84 ellipsoidal heights
        target_srs_str = EPSG_4326
        cmd = [
            "gdalwarp",
            "-s_srs",
            source_srs_str,
            "-t_srs",
            target_srs_str,
            "-r",
            "near",
            "-multi",
            "-wm",
            "4096",
            "--config",
            "GDAL_NUM_THREADS",
            str(num_threads),
            "--config",
            "GDAL_CACHEMAX",
            str(gdal_cachemax_mb),
            "-of",
            "GTiff",
            "-co",
            _BIG_TIF_YES,
        ]
        if nodata_value is not None:
            cmd += ["-dstnodata", str(nodata_value)]
        cmd += [input_file.replace("\\", "/"), output_file.replace("\\", "/")]

        log_l.log_message(logger, f"Source SRS: {source_srs_str}")
        log_l.log_message(logger, f"Target SRS: {target_srs_str}")
        run_subprocess(cmd, wait_file=output_file.replace("\\", "/"), logger=logger)
    except Exception as e:
        raise RuntimeError(f"Error: Could not apply vertical transformation to raster {os.path.basename(input_file)}: {e}") from e


def ortho_rectify_rpc(
    input_file: str,
    output_file: str,
    input_dem: str,
    target_crs: str,
    target_resolution: float | None = None,
    resample_alg: str = "near",
    nodata_value: int = 0,
    logger: logging.Logger | None = None,
) -> None:
    """Orthorectify a georeferenced image using RPCs and an external DEM via gdalwarp.

    Parameters
    ----------
    input_file : str
        Path to the input raster file.
    output_file : str
        Path to the reprojected output file.
    input_dem : str
        Path to the DEM file (in WGS84) used for RPC correction.
    target_crs : str
        Target CRS in Proj4 or EPSG format.
    target_resolution : int or float, optional; default is None
        Target resolution of the output file
    resample_alg: str
        Resample algorith; default is 'near'
    nodata_value : float, optional
        Value to use as NoData in the output raster. Defaults to 0.
    logger : Optional[logging.Logger]
        A logger instance to log the message. If None, prints to the console.

    Raises
    ------
    Exception
        If gdalwarp fails or required files are missing.

    Returns
    -------
    None

    """
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Error: Input file not found: {input_file}")
    if not os.path.isfile(input_dem):
        raise FileNotFoundError(f"Error: DEM file not found: {input_dem}")

    if os.path.exists(output_file):
        io_l.delete_file(output_file)

    with rasterio.open(input_dem) as dem:
        min_x, min_y, max_x, max_y = dem.bounds

    # Check the resampling algorithm
    resampling = define_resampling_algorithm(resample_alg=resample_alg, for_library="gdal")

    # Format output crs in the form gdal accepts
    to_crs = f"EPSG:{parse_epsg_code(target_crs)}"

    # Get computer's capabilities
    gdal_cachemax_mb, num_threads = _get_gdal_capacity()

    cmd = [
        "gdalwarp",
        "-rpc",
        "-r",
        resampling,
        "-t_srs",
        to_crs,
        "-te",
        str(round(min_x, COORD_ROUNDING)),
        str(round(min_y, COORD_ROUNDING)),
        str(round(max_x, COORD_ROUNDING)),
        str(round(max_y, COORD_ROUNDING)),
        "-dstnodata",
        str(nodata_value),
        "-multi",
        "-wm",
        "4096",
        "--config",
        "GDAL_NUM_THREADS",
        str(num_threads),
        "--config",
        "GDAL_CACHEMAX",
        str(gdal_cachemax_mb),
        "-to",
        f"RPC_DEM={input_dem}",
        "-co",
        _BIG_TIF_YES,
    ]
    if target_resolution is not None:
        cmd += ["-tr", str(target_resolution), str(target_resolution)]

    cmd += [input_file, output_file]
    run_subprocess(cmd, logger=logger)


def create_empty_dem_raster(
    output_path: str,
    bbox: list[float],
    resolution: float,
    crs: int | str = 2100,  # Default to GREEK_GRID (EPSG:2100)
    nodata_value: int = 0,
    mask_value: int = 0,
) -> None:
    """Create an empty raster filled with 0 using the provided bounding box.

    Parameters
    ----------
    output_path : str
        Path to save the output raster.
    bbox : list[float]
        Bounding box in projected coordinates [minx, miny, maxx, maxy]. It's coordinated must be in the 'crs' projection.
    resolution : float
        Pixel size in projection units (e.g., meters).
    crs : int or str
        EPSG code or PROJ string of the target CRS.
    nodata_value : int
        Value to use for all pixels and NoData.
    mask_value : int
        Value to fill the new raster.

    Raises
    ------
    ValueError
        If the bounding box is invalid.

    """
    if len(bbox) != 4:
        raise ValueError(f"Error: Invalid bbox: {bbox}")

    minx, miny, maxx, maxy = bbox
    width = int(np.ceil((maxx - minx) / resolution))
    height = int(np.ceil((maxy - miny) / resolution))

    transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy, width, height)

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=rasterio.float32,
        crs=CRS.from_user_input(crs),
        transform=transform,
        nodata=nodata_value,
        compress="deflate",
    ) as dst:
        dst.write(np.full((height, width), mask_value, dtype=np.float32), 1)


MOSAIC_S3_GET_OBJECT_MAX_ATTEMPTS = 3
MOSAIC_S3_GET_OBJECT_BASE_SLEEP_SECONDS = 1.0
MOSAIC_S3_RETRYABLE_ERROR_CODES = {"500", "502", "503", "504", "RequestTimeout", "SlowDown", "Throttling"}
MOSAIC_S3_RETRYABLE_EXCEPTIONS = (
    botocore.exceptions.EndpointConnectionError,
    botocore.exceptions.ConnectionClosedError,
    botocore.exceptions.ReadTimeoutError,
)


def _is_retryable_mosaic_s3_error(exc: Exception) -> bool:
    """Return True when a mosaic S3 download failure looks transient."""
    if isinstance(exc, MOSAIC_S3_RETRYABLE_EXCEPTIONS):
        return True

    if isinstance(exc, botocore.exceptions.ClientError):
        error = exc.response.get("Error", {})
        error_code = str(error.get("Code", ""))
        http_status = str(error.get("HTTPStatusCode", ""))
        error_message = str(error.get("Message", exc)).lower()

        if error_code in MOSAIC_S3_RETRYABLE_ERROR_CODES or http_status in MOSAIC_S3_RETRYABLE_ERROR_CODES:
            return True

        return any(hint in error_message for hint in ("temporarily unavailable", "service unavailable", "slow down"))

    message = str(exc).lower()
    return any(hint in message for hint in ("temporarily unavailable", "service unavailable", "slow down", "timeout"))


def _get_s3_object_with_retry(
    s3_client,
    bucket_name: str,
    s3_key: str,
    logger: logging.Logger | None = None,
) -> dict:
    """Fetch an S3 object with bounded retries for transient errors."""
    last_exception: Exception | None = None

    for attempt in range(1, MOSAIC_S3_GET_OBJECT_MAX_ATTEMPTS + 1):
        try:
            return s3_client.get_object(Bucket=bucket_name, Key=s3_key)
        except Exception as exc:
            last_exception = exc
            if attempt >= MOSAIC_S3_GET_OBJECT_MAX_ATTEMPTS or not _is_retryable_mosaic_s3_error(exc):
                raise

            sleep_s = MOSAIC_S3_GET_OBJECT_BASE_SLEEP_SECONDS * attempt
            log_l.log_message(
                logger,
                f"Transient S3 error while reading s3://{bucket_name}/{s3_key} "
                f"(attempt {attempt}/{MOSAIC_S3_GET_OBJECT_MAX_ATTEMPTS}): {exc}. Retrying in {sleep_s:.1f}s...",
            )
            time.sleep(sleep_s)

    raise RuntimeError(f"Error: Unable to retrieve s3://{bucket_name}/{s3_key}: {last_exception}") from last_exception




def _compute_output_grid(
    src_datasets: list[rasterio.io.DatasetReader],
    nodata_value: float,
    apply_mask: bool,
    mask_value: float,
) -> tuple[float, int, int, rasterio.Affine, dict]:
    """Compute fill value, output grid dimensions, transform, and metadata."""
    fill_value = mask_value if apply_mask else nodata_value

    xs: list[float] = []
    ys: list[float] = []
    for src in src_datasets:
        left, bottom, right, top = src.bounds
        xs.extend([left, right])
        ys.extend([bottom, top])

    out_bounds = (min(xs), min(ys), max(xs), max(ys))
    ref_transform = src_datasets[0].transform

    out_width = int(np.ceil((out_bounds[2] - out_bounds[0]) / abs(ref_transform.a)))
    out_height = int(np.ceil((out_bounds[3] - out_bounds[1]) / abs(ref_transform.e)))

    transform = rasterio.transform.from_bounds(
        out_bounds[0],
        out_bounds[1],
        out_bounds[2],
        out_bounds[3],
        out_width,
        out_height,
    )

    out_meta = src_datasets[0].meta.copy()
    out_meta.update(
        {
            "height": out_height,
            "width": out_width,
            "transform": transform,
            "nodata": nodata_value,
            "count": 1,
            "dtype": np.float32,
            "tiled": True,
            "blockxsize": BLOCKSIZE,
            "blockysize": BLOCKSIZE,
            "compress": "deflate",
            "BIGTIFF": "YES",
        }
    )

    return fill_value, out_height, out_width, transform, out_meta


def _merge_block_from_sources(
    src_datasets: list[rasterio.io.DatasetReader],
    window_bounds: tuple[float, float, float, float],
    block_height: int,
    block_width: int,
    fill_value: float,
    nodata_value: float,
    mask_value: float,
    apply_mask: bool,
) -> np.ndarray:
    """Merge a single block (window) from all sources into an output block."""
    out_block = np.full((block_height, block_width), fill_value, dtype=np.float32)

    for src in src_datasets:
        src_bounds = src.bounds
        intersects = (
            window_bounds[0] < src_bounds[2]
            and window_bounds[2] > src_bounds[0]
            and window_bounds[1] < src_bounds[3]
            and window_bounds[3] > src_bounds[1]
        )
        if not intersects:
            continue

        try:
            src_window = rasterio.windows.from_bounds(
                *window_bounds,
                transform=src.transform,
            )
            data = src.read(
                1,
                window=src_window,
                out_shape=(block_height, block_width),
                boundless=True,
                fill_value=fill_value,
            )

            if apply_mask:
                data = np.where(data <= 0, mask_value, data)

            if src.nodata is not None and src.nodata != mask_value:
                data = np.where(data == src.nodata, nodata_value, data)

            data = np.where(np.isnan(data), nodata_value, data)

            mask = out_block == fill_value
            out_block[mask] = data[mask]
        except Exception as e:
            logging.getLogger(__name__).debug("Error: _merge_block_from_sources failed; using its fallback.", exc_info=True)
            print(f"Warning: Skipping source {getattr(src, 'name', '?')}: {e}")
            continue

    return out_block


def _build_cog_creation_options(input_file: str, is_float: bool) -> list[str]:
    """Build GDAL COG creation options for an input raster."""
    cog_options = [
        "COMPRESS=DEFLATE",
        "PREDICTOR=" + ("3" if is_float else "2"),
        "TILED=YES",
        f"BLOCKXSIZE={BLOCKSIZE}",
        f"BLOCKYSIZE={BLOCKSIZE}",
        "BIGTIFF=IF_SAFER",
        "USE_RRD=NO",
        "ADD_OVERVIEWS=YES",
    ]

    # Remove BLOCKXSIZE and BLOCKYSIZE for unsupported drivers
    unsupported_drivers = ["ENVI", "JP2OpenJPEG"]
    if any(driver in input_file.upper() for driver in unsupported_drivers):
        cog_options = [opt for opt in cog_options if "BLOCKXSIZE" not in opt and "BLOCKYSIZE" not in opt]

    return cog_options


def _build_cog_translate_command(
    gdal_cachemax_mb: int,
    num_threads: int,
    cog_options: list[str],
    input_file: str,
    output_file: str,
    bands: list[int] | None = None,
) -> list[str]:
    """Build a gdal_translate command for COG conversion."""
    cmd = [
        "gdal_translate",
        "-of",
        "COG",
        "--config",
        "GDAL_NUM_THREADS",
        str(num_threads),
        "--config",
        "GDAL_CACHEMAX",
        str(gdal_cachemax_mb),
    ]

    for option in cog_options:
        cmd += ["-co", option]

    if bands:
        for band in bands:
            cmd += ["-b", str(band)]

    cmd += [input_file, output_file]
    return cmd


def is_cloud_optimized(profile: dict) -> bool:
    """Determine whether a raster profile corresponds to a Cloud-Optimized GeoTIFF (COG).

    A COG is typically:
    - Internally tiled ('tiled=True' or 'TILED=True')
    - Contains defined internal block sizes ('blockxsize'/'blockysize' or 'BLOCKXSIZE'/'BLOCKYSIZE')

    Parameters
    ----------
    profile : dict
        The raster profile dictionary obtained from 'dataset.profile' in rasterio.
        This includes metadata such as tiling, block sizes, compression, dtype, etc.

    Returns
    -------
    bool
        True if the file has internal tiling and defined block sizes (i.e., likely a COG),
        False otherwise.

    """
    is_tiled = profile.get("tiled", False) or profile.get("TILED", False)
    has_blocks = (
        (profile.get("blockxsize") is not None and profile.get("blockysize") is not None)
        or (profile.get("BLOCKXSIZE") is not None and profile.get("BLOCKYSIZE") is not None)
        or (profile.get("blocksize") is not None)
        or (profile.get("BLOCKSIZE") is not None)
    )
    return is_tiled and has_blocks


def is_geotiff(dataset: rasterio.io.DatasetReader) -> bool:
    """
    Determine whether a raster dataset qualifies as a valid GeoTIFF.

    A file is considered a GeoTIFF if:
    - It contains a valid Coordinate Reference System (CRS), and
    - Its affine transform is not the identity matrix (i.e., spatial positioning exists)

    Parameters
    ----------
    dataset : rasterio.io.DatasetReader
        An open rasterio dataset object (e.g., as returned by 'rasterio.open()').

    Returns
    -------
    bool
        True if the dataset has both a valid CRS and a meaningful transform,
        indicating it is georeferenced (GeoTIFF). False otherwise.

    """
    return dataset.crs is not None and dataset.crs.to_string() != "" and not dataset.transform.is_identity


def get_tif_type(file_path: str) -> str:
    """Determine the type of a TIFF file: COG, GeoTIFF, or regular TIFF.

    Parameters
    ----------
    file_path : str
        Path to the TIFF file.

    Returns
    -------
    str
        "COG" if the file is a Cloud-Optimized GeoTIFF,
        "GEOTIFF" if it's georeferenced,
        or "TIFF" if it is a regular TIFF file.

    Raises
    ------
    Exception
        If the TIFF file cannot be read or analyzed.
    """
    if not file_path.lower().endswith((".tif", ".tiff")):
        return "Not a TIFF file"

    try:
        with rasterio.open(file_path) as dataset:
            profile = dataset.profile

            if is_cloud_optimized(profile):
                return "COG"
            if is_geotiff(dataset):
                return "GEOTIFF"
            return "TIFF"

    except Exception as e:
        raise RuntimeError(f"Error: Could not retrieve TIFF type for {file_path}: {e}") from e


def get_epsg_from_crs(dataset_crs: str | CRS) -> int | str | CRS:
    """Extract the EPSG code from a CRS (Coordinate Reference System) object or string.

    Parameters
    ----------
    dataset_crs : Union[str, CRS])
        The CRS input, which can be a rasterio CRS object, WKT string, PROJ string, or EPSG string.

    Returns
    -------
    Union[int, str, CRS]
        The extracted EPSG code if found; otherwise, returns the original CRS input.

    Raises
    ------
    None
        explicitly; if an error occurs, a message is printed and the original CRS is returned.

    """
    try:
        # Fast-path for plain EPSG string inputs to avoid unnecessary CRS parsing.
        if isinstance(dataset_crs, str):
            match = re.match(r"^\s*EPSG\s*:\s*(\d+)\s*$", dataset_crs, flags=re.IGNORECASE)
            if match:
                return int(match.group(1))

        if isinstance(dataset_crs, CRS):
            # Check if the CRS directly provides an EPSG code
            epsg_code = dataset_crs.to_epsg()
            if epsg_code is not None:
                return epsg_code

            # Attempt to re-parse from WKT to handle non-EPSG CRS definitions
            crs_from_wkt = CRS.from_wkt(dataset_crs.to_wkt())
            epsg_code = crs_from_wkt.to_epsg()
            if epsg_code is not None:
                return epsg_code

        # Handle string inputs (e.g., WKT, PROJ string)
        if isinstance(dataset_crs, str):
            crs = CRS.from_string(dataset_crs)
            epsg_code = crs.to_epsg()
            if epsg_code is not None:
                return epsg_code

        # Fallback: return original CRS if EPSG code can't be determined
        return dataset_crs

    except Exception as e:
        logging.getLogger(__name__).debug("Error: get_epsg_from_crs failed; using its fallback.", exc_info=True)
        print(f"\nError: Could not determine EPSG code: {e}")
        return dataset_crs


def parse_epsg_code(epsg_code: str | int) -> int:
    """Normalize an EPSG code input to an integer value.

    Parameters
    ----------
    epsg_code : Union[str, int]
        The EPSG code as a string (e.g., EPSG_4326) or integer.

    Returns
    -------
    int
        The EPSG code as an integer.

    """
    if isinstance(epsg_code, str) and epsg_code.upper().startswith(_EPSG_PREFIX):
        # Remove "EPSG:" prefix and convert to uppercase
        epsg_code = epsg_code.upper().replace(_EPSG_PREFIX, "").strip()
    return int(epsg_code)


def normalize_unit_name(unit_name: str) -> str:
    """Normalize a CRS unit name to one of the standard forms: 'm', 'ft', or 'degrees'.

    Parameters
    ----------
    unit_name : str
        The original unit name string (e.g., from a CRS definition).

    Returns
    -------
    str
        Normalized unit name ('m', 'ft', 'degrees', or 'unknown' if not recognized).

    """
    # Normalize the unit name to lowercase
    unit_name = unit_name.lower()
    # Check for common keywords in the unit name
    if any(kw in unit_name for kw in ["foot", "ft", "us survey foot"]):
        return "ft"
    elif "metre" in unit_name or "meter" in unit_name or "+units=m" in unit_name:
        return "m"
    elif "degree" in unit_name:
        return "degrees"
    return "unknown"


def get_units_from_epsg(epsg_code: str | int) -> str:
    """Determine the coordinate units (e.g., 'm', 'ft', 'degrees') based on the EPSG code.

    Parameters
    ----------
    epsg_code : Union[str, int])
        The EPSG code as a string or integer (e.g., 'EPSG:32633' or 4326).

    Returns
    -------
    str
        One of 'm', 'ft', 'degrees', or 'unknown' if the unit cannot be determined.

    """
    try:
        # Parse the EPSG code to an integer if it's a string
        epsg = parse_epsg_code(epsg_code)

        # Handle special case for EPSG:4326 (WGS 84)
        if epsg == 4326:
            return "degrees"

        # Get the CRS object from the EPSG code
        crs = CRS.from_epsg(epsg)
        unit_name = crs.axis_info[0].unit_name.lower()

        # Normalize the unit name
        return normalize_unit_name(unit_name)

    except Exception:
        logging.getLogger(__name__).debug("Error: get_units_from_epsg failed; using its fallback.", exc_info=True)
        return "unknown"


def _resolve_math_operation(operation: str) -> tuple[str, Callable]:
    """Normalize operation token and return numpy operator function."""
    ops = {
        "add": np.add,
        "+": np.add,
        "subtract": np.subtract,
        "-": np.subtract,
        "multiply": np.multiply,
        "*": np.multiply,
        "divide": np.divide,
        "/": np.divide,
    }
    op_key = (operation or "").strip().lower()
    if op_key not in ops:
        raise ValueError(f"Error: Unsupported operation: {operation!r}")
    return op_key, ops[op_key]


def _validate_compatible_rasters(src1, src2) -> None:
    """Validate basic compatibility between two raster datasets."""
    if (src1.width != src2.width) or (src1.height != src2.height):
        raise ValueError("Error: Input rasters must have the same dimensions.")
    if (src1.crs != src2.crs) or (src1.transform != src2.transform):
        raise ValueError("Error: Input rasters must have the same CRS and geotransform.")


def _build_math_output_metadata(src1, src2) -> tuple[float, dict]:
    """Build output nodata and metadata for raster math outputs."""
    out_nodata = next(
        (float(v) for v in (src1.nodata, src2.nodata) if v is not None and np.isfinite(v)),
        np.nan,
    )

    meta = src1.meta.copy()
    meta.update(
        {
            "dtype": rasterio.float32,
            "count": 1,
            "driver": "GTiff",
            "nodata": out_nodata,
            "tiled": True,
            "compress": "deflate",
            "predictor": 3,
            "blockxsize": BLOCKSIZE,
            "blockysize": BLOCKSIZE,
            "BIGTIFF": "YES",
        }
    )
    return out_nodata, meta


def _apply_math_to_window(
    src1,
    src2,
    window,
    op_key: str,
    op_func: Callable,
    out_nodata: float,
) -> np.ndarray:
    """Compute math operation result for one output window."""
    r1 = src1.read(1, window=window).astype(np.float32, copy=False)
    r2 = src2.read(1, window=window).astype(np.float32, copy=False)

    invalid = create_nodata_mask(r1, src1.nodata) | create_nodata_mask(r2, src2.nodata)
    valid = (src1.read_masks(1, window=window) != 0) & (src2.read_masks(1, window=window) != 0)
    valid &= ~invalid

    if op_key in ("divide", "/"):
        valid &= r2 != 0

    with np.errstate(divide="ignore", invalid="ignore"):
        out_vals = op_func(r1, r2)

    result = np.full(r1.shape, out_nodata, dtype=np.float32)
    result[valid] = out_vals[valid]

    invalid = ~np.isfinite(result)
    if invalid.any():
        result[invalid] = out_nodata

    return result


def math_rasters(
    raster1_path: str,
    raster2_path: str,
    output_raster_path: str,
    operation: str = "subtract",
    logger: logging.Logger | None = None,
) -> None:
    """Perform pixel-wise math operation on two rasters using block-based processing.

    Uses windowed processing to minimize memory usage for large rasters.

    Nodata handling
    ---------------
    - Reads the per-band validity masks from BOTH rasters.
      Pixels are considered valid only where *both* masks are valid.
    - If a nodata value is defined on either input, the union of invalid pixels is
      set to nodata in the output.
    - For division, pixels where the second raster is 0 are also set to nodata.
    - Output is written as float32.

    Parameters
    ----------
    raster1_path : str
        File path to the first raster (band 1 is used).
    raster2_path : str
        File path to the second raster (band 1 is used).
    output_raster_path : str
        File path to save the output raster.
    operation : str
        One of {'add','subtract','multiply','divide','+','-','*','/'}; default 'subtract'.
    logger : Optional[logging.Logger]
        A logger instance to log the message. If None, prints to the console.

    Raises
    ------
    ValueError
        If rasters differ in dimensions, CRS, or transform, or operation is unsupported.
    RuntimeError
        If the operation fails due to a processing error.
    """
    try:
        op_key, op_func = _resolve_math_operation(operation)

        # Open inputs and validate
        with rasterio.open(raster1_path) as src1, rasterio.open(raster2_path) as src2:
            _validate_compatible_rasters(src1, src2)
            out_nodata, meta = _build_math_output_metadata(src1, src2)

            # Process in blocks
            with rasterio.open(output_raster_path, "w", **meta) as dst:
                for _, window in dst.block_windows(1):
                    result = _apply_math_to_window(
                        src1=src1,
                        src2=src2,
                        window=window,
                        op_key=op_key,
                        op_func=op_func,
                        out_nodata=out_nodata,
                    )

                    # Write block
                    dst.write(result, 1, window=window)

                    # Clean up
                    del result

        msg = f"Raster math operation completed successfully: {os.path.basename(raster1_path)} {operation} "
        msg += f"{os.path.basename(raster2_path)} -> {os.path.basename(output_raster_path)}"
        log_l.log_message(logger, msg)

    except Exception as e:
        raise RuntimeError(f"Error: Error processing raster operation {operation}: {e}") from e


def create_no_data_mask(input_tif: str, output_file: str, logger: logging.Logger | None = None) -> None:
    """Create a 0/1 mask raster using block-based processing: 0 for nodata, 1 for valid.

    Uses windowed processing to minimize memory usage for large rasters.

    Parameters
    ----------
    input_tif : str
        Path to the source GeoTIFF.
    output_file : str
        Destination path for the mask GeoTIFF (uint8).
    logger : logging.Logger
        Logger instance for logging messages.

    Raises
    ------
    Exception
        If any I/O or write error occurs.
    """
    try:
        with rasterio.open(input_tif) as src:
            nodata = src.nodata if src.nodata is not None else 0

            profile = src.profile.copy()
            profile.update(
                {
                    "dtype": "uint8",
                    "count": 1,
                    "driver": "GTiff",
                    "nodata": 0,
                    "tiled": True,
                    "compress": "deflate",
                    "predictor": 2,
                    "blockxsize": BLOCKSIZE,
                    "blockysize": BLOCKSIZE,
                    "BIGTIFF": "YES",
                }
            )

            # Remove floating metadata irrelevant to uint8
            profile.pop("scales", None)
            profile.pop("offsets", None)

            # Process in blocks
            with rasterio.open(output_file, "w", **profile) as dst:
                for _, window in dst.block_windows(1):
                    # Read data and mask for this block
                    data = src.read(1, window=window, masked=False)
                    band_mask = src.read_masks(1, window=window) != 0

                    # Determine validity
                    # 1. Start with band mask
                    valid = band_mask.copy()

                    # 2. Exclude nodata values
                    invalid = create_nodata_mask(data, nodata)
                    valid &= ~invalid

                    # 3. Exclude NaN/Inf (relevant for float products)
                    valid &= np.isfinite(data)

                    # Create output: 1 for valid, 0 for nodata
                    out_block = np.where(valid, 1, 0).astype(np.uint8, copy=False)

                    # Write block
                    dst.write(out_block, 1, window=window)

                    # Clean up
                    del data, band_mask, valid, out_block

        log_l.log_message(logger, f"Success: MASK raster written at {os.path.basename(output_file)}")

    except Exception as e:
        msg = f"Error AXIS12PA-017: Failed to create MASK raster from {os.path.basename(input_tif)}: {e}"
        log_l.log_message(logger, msg, type="error")
        raise RuntimeError(msg) from e


def create_nodata_mask(data: np.ndarray, nodata_val: float, exclude_negative: bool = False) -> np.ndarray:
    """Create a boolean mask for nodata values in the input array.

    Parameters
    ----------
    data : np.ndarray
        Input raster data array.
    nodata_val : float
        Nodata value to check against.
    exclude_negative : bool
        If True, also consider negative values as nodata.

    Returns
    -------
    np.ndarray
        Boolean mask where True indicates nodata pixels.

    """
    if nodata_val is None:
        return np.isnan(data)

    if np.issubdtype(data.dtype, np.floating):
        nodata_mask = np.isnan(data) | np.isclose(data, float(nodata_val), rtol=0.0, atol=1e-12) | np.isinf(data)
    else:
        nodata_mask = data == nodata_val

    if exclude_negative:
        nodata_mask |= data < 0

    return nodata_mask


def _pick_coord_name(coords, coord_names: list[str]) -> str | None:
    """Return primary coord name if present, else fallback, else None."""
    for name in coord_names:
        if name in coords:
            return name
    return None


def _validate_geotiff_output_path(out_tif: str) -> None:
    """Validate output path extension for GeoTIFF export."""
    if not out_tif.lower().endswith((".tif", ".tiff")):
        raise ValueError(f"Error: out_tif must be a .tif/.tiff path, got: {out_tif}")


def _extract_layer_and_spatial_coords(ds, nc_layer: str, nc_path: str):
    """Extract target layer and detect lon/lat coordinate names."""
    if nc_layer not in ds.data_vars:
        return None, None, None

    da = ds[nc_layer]
    lon_name = _pick_coord_name(da.coords, ["longitude", "lon"])
    lat_name = _pick_coord_name(da.coords, ["latitude", "lat"])
    if lon_name is None or lat_name is None:
        raise ValueError(f"Error: Cannot find lon/lat coords in {nc_path}. Coords: {list(da.coords)}")

    return da, lon_name, lat_name


def _prepare_dataarray_for_geotiff(da, lon_name: str, lat_name: str):
    """Slice/sort/annotate DataArray for GeoTIFF export in EPSG:4326."""
    # Slice any non-spatial dims (valid_time/time/step/etc.)
    extra_dims = [d for d in da.dims if d not in (lat_name, lon_name)]
    if extra_dims:
        da = da.isel(dict.fromkeys(extra_dims, 0))

    da = da.transpose(lat_name, lon_name)

    # Ensure lon ascending
    if da[lon_name][0] > da[lon_name][-1]:
        da = da.sortby(lon_name)

    da = da.rio.set_spatial_dims(x_dim=lon_name, y_dim=lat_name, inplace=False)
    da = da.rio.write_crs(EPSG_4326, inplace=False)

    if not np.issubdtype(da.dtype, np.floating):
        da = da.astype("float32")

    return da


def export_nc_layer_to_geotiff(nc_path: str, nc_layer: str, out_tif: str) -> bool:
    """Export a layer from a NetCDF to a single-band GeoTIFF (EPSG:4326)."""
    try:
        _validate_geotiff_output_path(out_tif)

        with xr.open_dataset(nc_path, mask_and_scale=True) as ds:
            da, lon_name, lat_name = _extract_layer_and_spatial_coords(ds, nc_layer, nc_path)
            if da is None:
                return False

            da = _prepare_dataarray_for_geotiff(da, lon_name, lat_name)
            os.makedirs(os.path.dirname(out_tif), exist_ok=True)
            da.rio.to_raster(out_tif)
            return True

    except Exception as e:
        logging.getLogger(__name__).debug("Error: export_nc_layer_to_geotiff failed; using its fallback.", exc_info=True)
        print(f"Error: Failed exporting {nc_layer} to GeoTIFF from: {nc_path} -> {out_tif}. Error: {e}")
        return False
