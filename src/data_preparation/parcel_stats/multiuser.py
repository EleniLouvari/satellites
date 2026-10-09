"""Multi-user spatial partitioning for parallel openEO extraction.

Partitions retain their spatial identity and output paths. By default, tile
batches are shared across all accounts, with at most two CDSE jobs per account.
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

from data_preparation.parcel_stats.batch_scheduler import run_shared_batches
from data_preparation.parcel_stats.partition_scheduler import (
    run_partition_extractor as run_partition_extractor,  # noqa: PLC0414 - explicit public re-export
)
from data_preparation.parcel_stats.partition_scheduler import (
    run_shared_partitions,
)
from shared.io import write_data

PARTITION_RESULT_FILE_NAME = "satellite_parcel_time_stats.geoparquet"
MAX_OPENEO_JOBS_PER_USER = 2


def _grid_shape_for_partition_count(gdf: gpd.GeoDataFrame, partition_count: int) -> tuple[int, int]:
    """Choose an exact factor grid whose cells best match the AOI aspect ratio."""
    if isinstance(partition_count, bool) or int(partition_count) < 1:
        raise ValueError("Error: partition_count must be a positive integer.")
    count = int(partition_count)
    minx, miny, maxx, maxy = gdf.total_bounds
    width = float(maxx - minx)
    height = float(maxy - miny)
    if not all(math.isfinite(value) for value in (width, height)) or width <= 0 or height <= 0:
        raise ValueError("Error: Parcel bounds must have positive finite width and height.")
    factor_pairs = [(rows, count // rows) for rows in range(1, math.isqrt(count) + 1) if count % rows == 0]
    # Prefer cells that are as close to square as the exact factorization allows.
    return min(factor_pairs, key=lambda shape: abs(math.log((width / shape[1]) / (height / shape[0]))))


def _resolve_grid_shape(
    gdf: gpd.GeoDataFrame,
    rows: int | None,
    cols: int | None,
    partition_count: int | None,
) -> tuple[int, int]:
    """Resolve and validate the effective grid shape."""
    if partition_count is not None:
        if rows is not None or cols is not None:
            raise ValueError("Error: Use either partition_count or rows/cols, not both.")
        return _grid_shape_for_partition_count(gdf, partition_count)
    if rows is None and cols is None:
        return 1, 2
    if rows is None or cols is None:
        raise ValueError("Error: rows and cols must be supplied together.")
    if isinstance(rows, bool) or isinstance(cols, bool) or int(rows) < 1 or int(cols) < 1:
        raise ValueError("Error: rows and cols must be positive integers.")
    return int(rows), int(cols)


def _grid_bounds_and_cell_size(
    gdf: gpd.GeoDataFrame, rows: int, cols: int
) -> tuple[float, float, float, float, float, float]:
    """Return bounds and cell dimensions for the resolved grid."""
    minx, miny, maxx, maxy = gdf.total_bounds
    cell_width = (maxx - minx) / cols
    cell_height = (maxy - miny) / rows
    return float(minx), float(miny), float(maxx), float(maxy), float(cell_width), float(cell_height)


def _assign_parcels_to_grid_cells(
    gdf: gpd.GeoDataFrame,
    rows: int,
    cols: int,
    minx: float,
    miny: float,
    cell_width: float,
    cell_height: float,
) -> dict[tuple[int, int], list[Any]]:
    """Assign each parcel index to exactly one grid cell using centroid coordinates."""
    cell_parcels: dict[tuple[int, int], list[Any]] = {(row, col): [] for row in range(rows) for col in range(cols)}
    centroids = gdf.geometry.centroid
    for parcel_idx, centroid in zip(gdf.index, centroids):
        col_idx = min(int((centroid.x - minx) / cell_width), cols - 1)
        row_idx = min(int((centroid.y - miny) / cell_height), rows - 1)
        col_idx = max(0, col_idx)
        row_idx = max(0, row_idx)
        cell_parcels[(row_idx, col_idx)].append(parcel_idx)
    return cell_parcels


def _build_non_empty_grid_parts(
    gdf: gpd.GeoDataFrame,
    rows: int,
    cols: int,
    cell_parcels: dict[tuple[int, int], list[Any]],
) -> list[gpd.GeoDataFrame]:
    """Build non-empty GeoDataFrame parts while preserving legacy progress output."""
    grid_parts: list[gpd.GeoDataFrame] = []
    total_parcels_assigned = 0
    for row in range(rows):
        for col in range(cols):
            parcel_indices = cell_parcels[(row, col)]
            if len(parcel_indices) > 0:
                parcels_in_cell = gdf.loc[parcel_indices].copy()
                grid_parts.append(parcels_in_cell)
                total_parcels_assigned += len(parcel_indices)
                print(f"  Cell ({row},{col}): {len(parcel_indices)} parcels")
            else:
                print(f"  Cell ({row},{col}): (empty)")
    if total_parcels_assigned != len(gdf):
        raise ValueError(
            f"Error: Parcel assignment mismatch: {total_parcels_assigned} assigned vs {len(gdf)} input parcels"
        )
    return grid_parts


def split_geodataframe_by_grid(
    gdf: gpd.GeoDataFrame,
    rows: int | None = None,
    cols: int | None = None,
    partition_count: int | None = None,
) -> list[gpd.GeoDataFrame]:
    """
    Split a GeoDataFrame spatially into a grid of N parts (no duplicate parcels).

    Parcels are assigned to grid cells based on their centroid, ensuring that each
    parcel appears in exactly one partition, even if its geometry intersects multiple
    grid cell boundaries.

    Parameters
    ----------
    gdf : GeoDataFrame
        Input parcels GeoDataFrame
    rows, cols : int, optional
        Explicit grid shape. When all grid arguments are omitted, the legacy
        1x2 default is used.
    partition_count : int, optional
        Exact number of grid cells. The rows and columns are selected from its
        factor pairs to suit the AOI aspect ratio. Use this with the number of
        openEO users so each account receives one initial spatial partition.

    Returns
    -------
    list[GeoDataFrame]
        List of GeoDataFrames, one per grid cell, maintaining original columns and CRS.
        Each parcel appears in exactly one partition (no duplicates).
    """
    if not isinstance(gdf, gpd.GeoDataFrame) or gdf.empty:
        raise ValueError("Error: gdf must be a non-empty GeoDataFrame.")
    if gdf.crs is None:
        raise ValueError("Error: gdf must have a CRS.")
    rows, cols = _resolve_grid_shape(gdf, rows, cols, partition_count)
    minx, miny, maxx, maxy, cell_width, cell_height = _grid_bounds_and_cell_size(gdf, rows, cols)

    print(f"Spatial grid: {rows}x{cols} cells")
    print(f"  Bounds: X=[{minx:.0f}, {maxx:.0f}], Y=[{miny:.0f}, {maxy:.0f}]")
    print(f"  Cell size: {cell_width:.0f}m x {cell_height:.0f}m")

    cell_parcels = _assign_parcels_to_grid_cells(gdf, rows, cols, minx, miny, cell_width, cell_height)
    return _build_non_empty_grid_parts(gdf, rows, cols, cell_parcels)


def load_openeo_users_from_db() -> list[tuple[str, str]]:
    """Load openEO accounts from the PostgreSQL table configured by ``TBL_USERS``.

    ``DB_NAME`` and ``TBL_USERS`` must be set in the process environment.
    Connection settings are read from ``POSTGRES_*`` by ``create_db_handler``.
    The table must contain non-empty text ``username`` and ``password`` columns.
    Usernames are stripped and must be unique ignoring case. Passwords are
    preserved exactly. Accounts are sorted by case-insensitive username; callers
    resuming legacy jobs without saved owners must restore the original order.
    This function does not load a ``.env`` file.

    Returns
    -------
    list[tuple[str, str]]
        List of (username, password) tuples, sorted by username.

    Raises
    ------
    ValueError
        If configuration is missing, database access fails, or accounts are invalid.
    """
    db_name = os.getenv("DB_NAME", "").strip()
    table_name = os.getenv("TBL_USERS", "").strip()
    if not db_name or not table_name:
        raise ValueError("Error: Set both DB_NAME and TBL_USERS before loading openEO accounts.")

    from shared.postgress import create_db_handler
    try:
        db_handler = create_db_handler(db_name=db_name)
    except Exception as exc:
        raise ValueError("Error: Failed to connect to the openEO accounts database; check POSTGRES_* settings.") from exc

    try:
        df_users = db_handler.get_df_from_db(table_name=table_name)
    except Exception as exc:
        raise ValueError("Error: Failed to read the openEO accounts table; check TBL_USERS and SELECT permissions.") from exc
    finally:
        db_handler.close_connection()

    credentials = _validate_openeo_accounts(df_users)
    print(f"Loaded {len(credentials)} openEO user(s) from the database.")
    return credentials


def _validate_openeo_accounts(df_users: pd.DataFrame) -> list[tuple[str, str]]:
    """Validate account rows and return normalized, deterministically sorted tuples."""
    if df_users.empty:
        raise ValueError("Error: The openEO accounts table is empty.")
    if not {"username", "password"}.issubset(df_users.columns):
        raise ValueError("Error: The openEO accounts table must contain username and password columns.")

    credentials = []
    for username, password in df_users[["username", "password"]].itertuples(index=False, name=None):
        if any(not isinstance(value, str) or not value.strip() for value in (username, password)):
            raise ValueError("Error: Each openEO username and password must be non-empty text.")
        credentials.append((username.strip(), password))
    user_names = [user for user, _ in credentials]
    if len({user.casefold() for user in user_names}) != len(user_names):
        raise ValueError("Error: The openEO accounts table contains duplicate usernames; list each account only once.")
    credentials.sort(key=lambda credential: credential[0].casefold())
    return credentials


def load_saved_partition_results(output_dir: Path) -> dict[int, gpd.GeoDataFrame]:
    """Load completed partition results from an extraction output directory.

    The expected layout is ``output_dir/partition_<n>/`` with one
    :data:`PARTITION_RESULT_FILE_NAME` file in each completed partition.

    Parameters
    ----------
    output_dir : Path
        Base output directory containing the partition directories.

    Returns
    -------
    dict[int, GeoDataFrame]
        Saved results keyed by their numeric partition index.

    Raises
    ------
    FileNotFoundError
        If no saved partition result files are found.
    ValueError
        If a matching partition directory does not have a numeric suffix.
    """
    output_dir = Path(output_dir)
    result_paths = list(output_dir.glob(f"partition_*/{PARTITION_RESULT_FILE_NAME}"))
    if not result_paths:
        raise FileNotFoundError(
            f"Error: No partition results found under {output_dir}. Expected files matching partition_*/{PARTITION_RESULT_FILE_NAME}."
        )

    indexed_paths = []
    for result_path in result_paths:
        partition_suffix = result_path.parent.name.removeprefix("partition_")
        try:
            partition_idx = int(partition_suffix)
        except ValueError as exc:
            raise ValueError(
                f"Error: Invalid partition directory {result_path.parent.name!r}; expected partition_<number>."
            ) from exc
        indexed_paths.append((partition_idx, result_path))

    saved_results = {}
    for partition_idx, result_path in sorted(indexed_paths):
        print(f"Loading partition {partition_idx}: {result_path}")
        saved_results[partition_idx] = gpd.read_parquet(result_path)
    return saved_results


def _prepare_partition_results(
    all_results: dict[int, gpd.GeoDataFrame] | None,
    output_dir: Path,
) -> dict[int, gpd.GeoDataFrame]:
    """Load or validate the saved partition outputs that should be merged."""
    if all_results is None:
        all_results = load_saved_partition_results(output_dir)
    if not all_results:
        raise ValueError("Error: No partition results were provided to merge.")
    return all_results


def _align_partition_columns(
    ordered_parts: list[gpd.GeoDataFrame],
) -> tuple[list[gpd.GeoDataFrame], str, Any]:
    """Align parcel columns across partitions and recover the effective geometry metadata."""
    all_columns = list(dict.fromkeys(col for part in ordered_parts for col in part.columns))
    geometry_col = next(
        (part.geometry.name for part in ordered_parts if hasattr(part, "geometry")),
        "geometry",
    )
    crs = next((part.crs for part in ordered_parts if hasattr(part, "crs") and part.crs is not None), None)
    aligned_parts = [
        gpd.GeoDataFrame(part.reindex(columns=all_columns), geometry=geometry_col, crs=part.crs)
        for part in ordered_parts
    ]
    return aligned_parts, geometry_col, crs


def _deduplicate_partition_results(
    merged_results: gpd.GeoDataFrame,
    parcel_id_column: str | None,
) -> gpd.GeoDataFrame:
    """Drop duplicate parcels when a parcel identifier is supplied."""
    if parcel_id_column is None:
        return merged_results
    if parcel_id_column not in merged_results.columns:
        raise KeyError(f"Error: Parcel ID column {parcel_id_column!r} is missing from partition results.")
    return merged_results.drop_duplicates(parcel_id_column, keep="first").reset_index(drop=True)


def _reproject_merged_results(
    merged_results: gpd.GeoDataFrame,
    working_epsg: int | None,
) -> gpd.GeoDataFrame:
    """Project the merged output to the configured working CRS if requested."""
    if working_epsg is None:
        return merged_results
    try:
        output_epsg = int(working_epsg)
        output_crs = gpd.GeoSeries([], crs=f"EPSG:{output_epsg}").crs
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Error: Invalid working EPSG code: {working_epsg!r}.") from exc
    if not output_crs.is_projected:
        raise ValueError("Error: working_epsg must identify a projected CRS.")
    return merged_results.to_crs(output_crs)


def _save_merged_results(merged_results: gpd.GeoDataFrame, output_dir: Path) -> Path:
    """Persist the merged result to the standard GeoParquet path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_geoparquet = output_dir / PARTITION_RESULT_FILE_NAME
    write_data(merged_results, str(output_geoparquet))
    return output_geoparquet


def merge_and_save_results(
    gdf_parcels: gpd.GeoDataFrame | None = None,
    all_results: dict[int, gpd.GeoDataFrame] | None = None,
    output_dir: Path | None = None,
    parcel_id_column: str | None = None,
    working_epsg: int | None = None,
) -> tuple[gpd.GeoDataFrame, Path]:
    """Merge partition results and save as GeoParquet.

    When ``all_results`` is omitted, results are loaded from the saved
    ``partition_*/satellite_parcel_time_stats.geoparquet`` files. This allows
    merging to run independently of :func:`run_parallel_extractions` and in a
    different Python session.

    Parameters
    ----------
    gdf_parcels : GeoDataFrame, optional
        Original input parcels, used only to report the expected row count.
    all_results : dict[int, GeoDataFrame], optional
        In-memory results keyed by partition index. If omitted, saved partition
        results are discovered below ``output_dir``.
    output_dir : Path, optional
        Output directory for merged files
    parcel_id_column : str, optional
        Column used to remove duplicate parcels. When omitted, all partition
        rows are retained because spatial partitions are disjoint by design.
    working_epsg : int, optional
        Projected output CRS. New partition outputs already use their configured
        working EPSG; supplying this also reprojects previously saved partition
        results before writing the merged GeoParquet.

    Returns
    -------
    tuple[GeoDataFrame, Path]
        (merged_gdf, output_geoparquet_path)
    """
    if output_dir is None:
        raise ValueError("Error: output_dir is required.")
    output_dir = Path(output_dir)
    all_results = _prepare_partition_results(all_results, output_dir)

    print(f"\n{'='*70}")
    print(f"Merging {len(all_results)} partition result(s)")
    print(f"{'='*70}")

    ordered_parts = [all_results[i] for i in sorted(all_results)]
    aligned_parts, geometry_col, crs = _align_partition_columns(ordered_parts)
    merged_results = gpd.GeoDataFrame(
        pd.concat(aligned_parts, ignore_index=True),
        geometry=geometry_col,
        crs=crs,
    )
    merged_results = _deduplicate_partition_results(merged_results, parcel_id_column)
    merged_results = _reproject_merged_results(merged_results, working_epsg)

    input_count = len(gdf_parcels) if gdf_parcels is not None else sum(map(len, all_results.values()))
    print(f"\nMerged results: {len(merged_results)} parcels (input: {input_count})")
    print(f"Output columns: {merged_results.shape[1]}")
    print(f"Output CRS: {merged_results.crs}")

    print(f"\n{'='*70}")
    print("Saving final merged results")
    print(f"{'='*70}")

    output_geoparquet = _save_merged_results(merged_results, output_dir)
    print(f"\nSaved final output: {output_geoparquet}")
    return merged_results, output_geoparquet


def run_parallel_extractions(
    spatial_parts: list[gpd.GeoDataFrame],
    users_list: list[tuple[str, str]],
    config: dict,
    *,
    scheduling: str = "batches",
) -> None:
    """Run parallel openEO extractions for multiple spatial partitions.

    Plan all partitions first, then share their unsubmitted tile batches across
    accounts. Each account has ``openeo_parallel_jobs`` slots (normally two).
    Existing remote jobs remain on their owning account when resuming. Use
    ``scheduling="partitions"`` for the legacy round-robin partition queues.

    Parameters
    ----------
    spatial_parts : list[GeoDataFrame]
        List of partitioned parcel GeoDataFrames
    users_list : list[tuple[str, str]]
        Available (username, password) tuples
    config : dict
        Configuration dictionary containing all extraction parameters:
        - output_dir: base output directory
        - run_identifier: run identifier string
        - parcel_id_field: name of parcel ID column
        - start_date, end_date: date range
        - working_epsg: projected EPSG code
        - sentinel2_bands, sentinel2_indices: Sentinel-2 config
        - sentinel1_bands, sentinel1_indices, sentinel1_orbit_direction: Sentinel-1 config
        - spatial_statistics: statistics to calculate
        - tile_size_metres, tile_buffer_metres: tiling parameters
        - Plus all additional options: IQR parameters, fill nulls,
          interpolation method, job poll seconds, batch workers, etc.

    Returns
    -------
    None
        Each partition writes its results to its own output directory.
    """
    print(f"\n{'='*70}")
    if not spatial_parts:
        raise ValueError("Error: spatial_parts must contain at least one partition.")
    if not users_list:
        raise ValueError("Error: users_list must contain at least one openEO account.")
    usernames = [user.strip().casefold() for user, _ in users_list]
    if len(set(usernames)) != len(usernames):
        raise ValueError("Error: Duplicate usernames; each account may be listed only once.")
    if scheduling not in {"batches", "partitions"}:
        raise ValueError("Error: scheduling must be 'batches' or 'partitions'.")
    jobs_per_user = int(config.get("openeo_parallel_jobs", 2))
    if jobs_per_user < 1:
        raise ValueError("Error: openeo_parallel_jobs must be positive.")
    if jobs_per_user > MAX_OPENEO_JOBS_PER_USER:
        raise ValueError(
            f"Error: openeo_parallel_jobs cannot exceed {MAX_OPENEO_JOBS_PER_USER}; add openEO users to increase total concurrency."
        )

    print(f"Running extraction on {len(spatial_parts)} spatial partition(s) across {len(users_list)} user queue(s)")
    print(f"Remote concurrency capacity: {len(users_list)} users x {jobs_per_user} jobs = {len(users_list) * jobs_per_user}")
    print(f"{'='*70}\n")

    start_time = time.time()

    if scheduling == "batches":
        run_shared_batches(spatial_parts, users_list, config, jobs_per_user)
    else:
        run_shared_partitions(spatial_parts, users_list, config)

    elapsed = time.time() - start_time
    print(f"\nAll {len(spatial_parts)} partition(s) completed in {elapsed:.1f}s")
