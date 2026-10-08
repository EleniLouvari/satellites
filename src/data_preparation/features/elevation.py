"""Utilities for mosaicking DEM tiles and adding parcel elevation features."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.errors import WindowError
from rasterio.features import geometry_mask, geometry_window
from rasterio.merge import merge

from shared.io import read_data, write_data

from .spatial_context import validate_projected_metric_crs


WORK_PATH = Path("C:/work_dir")
DEM_DIR = WORK_PATH / "dem"
PARCELS_PATH = WORK_PATH / "1_parcel_stats" / "satellite_parcel_time_stats.geoparquet"
MOSAIC_PATH = DEM_DIR / "dem_mosaic.tif"
ELEVATION_COLUMN = "elevation_mean_m"
PIXEL_COUNT_COLUMN = "elevation_valid_pixel_count"
MOSAIC_NODATA = -9999.0


def find_dem_tiles(dem_dir: Path, mosaic_path: Path) -> list[Path]:
    """Return deterministic DEM source tiles, excluding generated mosaics."""
    if not dem_dir.is_dir():
        raise FileNotFoundError(f"Error: DEM directory does not exist: {dem_dir}")

    mosaic_resolved = mosaic_path.resolve()
    tiles = sorted(
        path
        for path in dem_dir.glob("DEM*.tif")
        if path.is_file() and path.resolve() != mosaic_resolved
    )
    return tiles


def _validate_dem_tiles(tile_paths: Iterable[Path]) -> None:
    """Fail early when source rasters cannot form a meaningful elevation mosaic."""
    paths = list(tile_paths)
    with rasterio.open(paths[0]) as reference:
        expected_crs = reference.crs
        expected_count = reference.count
        expected_dtype = reference.dtypes[0]

    if expected_crs is None:
        raise ValueError(f"Error: DEM has no CRS: {paths[0]}")
    if expected_count != 1:
        raise ValueError(f"Error: Elevation DEM must contain one band; {paths[0]} has {expected_count}.")

    for path in paths[1:]:
        with rasterio.open(path) as source:
            if source.crs != expected_crs:
                raise ValueError(f"Error: DEM CRS mismatch: {path} has {source.crs}, expected {expected_crs}.")
            if source.count != 1:
                raise ValueError(f"Error: Elevation DEM must contain one band; {path} has {source.count}.")
            if source.dtypes[0] != expected_dtype:
                raise ValueError(
                    f"Error: DEM dtype mismatch: {path} has {source.dtypes[0]}, expected {expected_dtype}."
                )


def create_dem_mosaic(tile_paths: list[Path], mosaic_path: Path, overwrite: bool = False) -> Path:
    """Create a tiled, compressed DEM mosaic without loading it fully into memory."""
    if len(tile_paths) == 0:
        raise ValueError(f"Error: No DEM tiles found to create a mosaic: {mosaic_path}")
    if len(tile_paths) == 1:
        print(f"Only one DEM tile found; using it as the mosaic: {tile_paths[0]}")
        return tile_paths[0]

    if mosaic_path.exists() and not overwrite:
        print(f"Using existing DEM mosaic: {mosaic_path}")
        return mosaic_path

    _validate_dem_tiles(tile_paths)
    mosaic_path.parent.mkdir(parents=True, exist_ok=True)
    sources = [rasterio.open(path) for path in tile_paths]
    try:
        merge(
            sources,
            nodata=MOSAIC_NODATA,
            dtype="float32",
            method="first",
            target_aligned_pixels=True,
            mem_limit=256,
            dst_path=mosaic_path,
            dst_kwds={
                "driver": "GTiff",
                "compress": "DEFLATE",
                "predictor": 3,
                "tiled": True,
                "blockxsize": 512,
                "blockysize": 512,
                "BIGTIFF": "IF_SAFER",
            },
        )
    finally:
        for source in sources:
            source.close()

    print(f"Created DEM mosaic from {len(tile_paths)} tiles: {mosaic_path}")
    return mosaic_path


def _mean_elevation_for_geometry(
    dem: rasterio.io.DatasetReader,
    geometry,
    *,
    all_touched: bool,
) -> tuple[float, int]:
    """Calculate the unweighted mean of valid DEM cell values inside one geometry."""
    if geometry is None or geometry.is_empty:
        return np.nan, 0

    try:
        window = geometry_window(dem, [geometry], pad_x=0, pad_y=0)
    except (WindowError, ValueError):
        return np.nan, 0

    elevation = dem.read(1, window=window, masked=True)
    if elevation.size == 0:
        return np.nan, 0

    inside = geometry_mask(
        [geometry],
        out_shape=elevation.shape,
        transform=dem.window_transform(window),
        invert=True,
        all_touched=all_touched,
    )
    valid = inside & ~np.ma.getmaskarray(elevation)
    count = int(valid.sum())
    if count == 0:
        return np.nan, 0
    return float(np.asarray(elevation.data)[valid].mean(dtype=np.float64)), count


def append_parcel_elevation(
    parcels_path: Path,
    mosaic_path: Path,
    output_path: Path,
    *,
    all_touched: bool = False,
    overwrite: bool = False,
) -> gpd.GeoDataFrame:
    """Calculate parcel elevation means and atomically save a GeoParquet."""
    if not parcels_path.is_file():
        raise FileNotFoundError(f"Error: Parcel GeoParquet does not exist: {parcels_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Error: Output already exists: {output_path}. Use --overwrite to replace it.")

    parcels = read_data(str(parcels_path))
    original_crs = parcels.crs
    print(f"Parcel dataset CRS: {original_crs}")
    print(f"Parcel CRS type: {type(original_crs)}")
    if original_crs is not None:
        print(f"Parcel CRS to_epsg(): {original_crs.to_epsg()}")
        print(f"Parcel CRS is_projected: {original_crs.is_projected}")
    validate_projected_metric_crs(original_crs, dataset_name="Parcel dataset")
    if parcels.geometry.isna().all():
        raise ValueError("Error: Parcel dataset contains no usable geometries.")

    with rasterio.open(mosaic_path) as dem:
        if dem.crs is None:
            raise ValueError(f"Error: DEM mosaic has no CRS: {mosaic_path}")
        working_geometry = parcels.geometry
        if parcels.crs != dem.crs:
            print(f"Reprojecting parcel geometries from {parcels.crs} to {dem.crs} for zonal statistics.")
            working_geometry = parcels.to_crs(dem.crs).geometry

        means = np.full(len(parcels), np.nan, dtype=np.float64)
        counts = np.zeros(len(parcels), dtype=np.int32)
        for position, geometry in enumerate(working_geometry):
            means[position], counts[position] = _mean_elevation_for_geometry(
                dem, geometry, all_touched=all_touched
            )
            if (position + 1) % 5000 == 0 or position + 1 == len(parcels):
                print(f"Processed {position + 1:,}/{len(parcels):,} parcels")

    result = parcels.copy()
    result[ELEVATION_COLUMN] = pd.Series(means, index=result.index, dtype="float64")
    result[PIXEL_COUNT_COLUMN] = pd.Series(counts, index=result.index, dtype="int32")
    if result.crs != original_crs or not result.geometry.equals(parcels.geometry):
        raise RuntimeError("Error: Elevation feature creation unexpectedly changed the original parcel geometry or CRS.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.stem}.tmp{output_path.suffix}")
    try:
        write_data(result, str(temporary_path))
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    missing = int(result[ELEVATION_COLUMN].isna().sum())
    print(f"Saved parcels with mean elevation: {output_path}")
    print(f"Parcels without a valid DEM pixel: {missing:,}/{len(result):,}")
    return result
