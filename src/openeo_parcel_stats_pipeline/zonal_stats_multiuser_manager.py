"""Multi-user spatial partitioning for parallel openEO extraction.

This module orchestrates splitting large parcel GeoDataFrames into spatial
partitions and running independent extractors in parallel, each using a different
openEO user account to bypass per-user concurrent job limits (2 jobs max on CDSE).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import geopandas as gpd
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from shapely.geometry import box
import time

from common_libraries.io_library import write_data
from .zonal_stats_job_manager import JobManagerSatelliteZonalStats

if TYPE_CHECKING:
    from _typeshed import SupportsRead


def split_geodataframe_by_grid(gdf: gpd.GeoDataFrame, rows: int = 1, cols: int = 2) -> list[gpd.GeoDataFrame]:
    """
    Split a GeoDataFrame spatially into a grid of N parts (no duplicate parcels).

    Parcels are assigned to grid cells based on their centroid, ensuring that each
    parcel appears in exactly one partition, even if its geometry intersects multiple
    grid cell boundaries.

    Parameters
    ----------
    gdf : GeoDataFrame
        Input parcels GeoDataFrame
    rows : int
        Number of grid rows (default 1)
    cols : int
        Number of grid columns (default 2)

    Returns
    -------
    list[GeoDataFrame]
        List of GeoDataFrames, one per grid cell, maintaining original columns and CRS.
        Each parcel appears in exactly one partition (no duplicates).
    """
    # Get bounds of all parcels
    minx, miny, maxx, maxy = gdf.total_bounds

    # Calculate cell dimensions
    cell_width = (maxx - minx) / cols
    cell_height = (maxy - miny) / rows

    print(f"Spatial grid: {rows}x{cols} cells")
    print(f"  Bounds: X=[{minx:.0f}, {maxx:.0f}], Y=[{miny:.0f}, {maxy:.0f}]")
    print(f"  Cell size: {cell_width:.0f}m x {cell_height:.0f}m")

    # Compute centroid coordinates for each parcel
    centroids = gdf.geometry.centroid

    # Initialize empty dictionaries for each cell
    cell_parcels = {(row, col): [] for row in range(rows) for col in range(cols)}

    # Assign each parcel to exactly one cell based on its centroid
    for idx, (parcel_idx, centroid) in enumerate(zip(gdf.index, centroids)):
        # Determine which cell this centroid falls into
        col_idx = min(int((centroid.x - minx) / cell_width), cols - 1)
        row_idx = min(int((centroid.y - miny) / cell_height), rows - 1)

        # Ensure indices are in valid range (handles edge cases)
        col_idx = max(0, col_idx)
        row_idx = max(0, row_idx)

        cell_parcels[(row_idx, col_idx)].append(parcel_idx)

    # Create grid parts from assigned parcels
    grid_parts = []
    total_parcels_assigned = 0

    for row in range(rows):
        for col in range(cols):
            parcel_indices = cell_parcels[(row, col)]
            if len(parcel_indices) > 0:
                parcels_in_cell = gdf.loc[parcel_indices].copy()
                grid_parts.append((row, col, parcels_in_cell))
                total_parcels_assigned += len(parcel_indices)
                print(f"  Cell ({row},{col}): {len(parcel_indices)} parcels")
            else:
                print(f"  Cell ({row},{col}): (empty)")

    # Verify no duplicates
    if total_parcels_assigned != len(gdf):
        raise ValueError(
            f"Parcel assignment mismatch: {total_parcels_assigned} assigned "
            f"vs {len(gdf)} input parcels"
        )

    return [gdf for _, _, gdf in grid_parts]

def load_openeo_users_from_env() -> list[tuple[str, str]]:
    """Parse openEO user credentials from OPENEO_USERS environment variable.

    The environment variable format is: username,password;another_username,another_password

    Returns
    -------
    list[tuple[str, str]]
        List of (username, password) tuples

    Raises
    ------
    ValueError
        If OPENEO_USERS is not set or has invalid format
    """
    openeo_users_env = os.environ.get("OPENEO_USERS", "").strip()
    if not openeo_users_env:
        raise ValueError("OPENEO_USERS environment variable not set.")

    credentials = [
        tuple(part.strip() for part in entry.strip().split(",", maxsplit=1))
        for entry in openeo_users_env.split(";")
        if entry.strip()
    ]
    if not credentials or any(len(pair) != 2 or not all(pair) for pair in credentials):
        raise ValueError("OPENEO_USERS must contain one or more username,password pairs separated by semicolons.")

    user_names = [user for user, _ in credentials]
    print(f"Loaded {len(credentials)} openEO user(s): {user_names}")
    print(f"Each spatial partition will use one user (round-robin assignment)")
    return credentials

def run_partition_extractor(
    part_idx: int,
    gdf_part: gpd.GeoDataFrame,
    users_list: list[tuple[str, str]],
    total_partitions: int,
    config: dict,
) -> tuple[int, gpd.GeoDataFrame]:
    """Extract one spatial partition with assigned user.

    Parameters
    ----------
    part_idx : int
        Partition index (1-based)
    gdf_part : GeoDataFrame
        Parcels for this partition
    users_list : list[tuple[str, str]]
        Available (username, password) tuples
    total_partitions : int
        Total number of partitions (for display)
    config : dict
        Configuration dictionary containing:
        - output_dir: base output directory
        - run_identifier: run identifier string
        - parcel_id_field: name of parcel ID column
        - start_date, end_date: date range
        - working_epsg: projected EPSG code
        - sentinel2_bands, sentinel2_indices: Sentinel-2 config
        - calculate_sentinel2_indices: whether to calculate indices
        - sentinel1_bands: Sentinel-1 config
        - spatial_statistics: statistics to calculate
        - tile_size_metres, tile_buffer_metres: tiling parameters
        - Plus any additional kwargs (IQR, fill, interpolation, etc.)

    Returns
    -------
    tuple[int, GeoDataFrame]
        (partition_index, results_gdf)
        Run identifier string
    parcel_id_field : str
        Name of parcel ID column
    start_date, end_date : str
        Date range
    working_epsg : int
        Projected EPSG code
    sentinel2_bands, sentinel2_indices : list[str]
        Sentinel-2 bands and indices
    calculate_sentinel2_indices : bool
        Whether to calculate indices
    sentinel1_bands : list[str]
        Sentinel-1 bands
    spatial_statistics : list[str]
        Statistics to calculate
    tile_size_metres, tile_buffer_metres : int
        Tiling parameters
    **kwargs
        Additional parameters (IQR, interpolation, fill options, etc.)

    Returns
    -------
    tuple[int, GeoDataFrame]
        (partition_index, results_dataframe)
    """
    print(f"\n{'─'*70}")
    print(f"PARTITION {part_idx}/{total_partitions}: {len(gdf_part)} parcels")
    print(f"{'─'*70}")

    # Select the user for this partition (round-robin)
    selected_user_idx = (part_idx - 1) % len(users_list)
    selected_user = users_list[selected_user_idx]

    # Create partition-specific output directory
    partition_output_dir = output_dir / f"partition_{part_idx}"

    # Build extractor configuration
    extractor_config = dict(
        parcels=gdf_part,
        parcel_id_field=parcel_id_field,
        start_date=start_date,
        end_date=end_date,
        output_dir=partition_output_dir,
        working_epsg=working_epsg,
        openeo_username=selected_user[0],
        openeo_password=selected_user[1],
        sentinel2_bands=sentinel2_bands,
        calculate_sentinel2_indices=calculate_sentinel2_indices,
        sentinel2_indices=sentinel2_indices,
        sentinel1_bands=sentinel1_bands,
        spatial_statistics=spatial_statistics,
        tile_size_metres=tile_size_metres,
        tile_buffer_metres=tile_buffer_metres,
        run_identifier=f"{run_identifier}_part{part_idx}",
        **kwargs,
    )

    print(f"User: {selected_user[0]}")
    print(f"Output: {partition_output_dir}")

    # Create and run extractor
    extractor = JobManagerSatelliteZonalStats(**extractor_config)
    results = extractor.run()

    print(f"✓ Partition {part_idx} complete: {len(results)} parcels")
    return part_idx, results


def merge_and_save_results(
    gdf_parcels: gpd.GeoDataFrame,
    all_results: dict[int, gpd.GeoDataFrame],
    output_dir: Path,
    parcel_id_column: str,
) -> tuple[gpd.GeoDataFrame, Path, Path]:
    """Merge partition results and save as GeoParquet.

    Parameters
    ----------
    gdf_parcels : GeoDataFrame
        Original input parcels (for reference counts)
    all_results : dict[int, GeoDataFrame]
        Results keyed by partition index
    output_dir : Path
        Output directory for merged files
    parcel_id_column : str
        Name of parcel ID column

    Returns
    -------
    tuple[GeoDataFrame, Path, Path]
        (merged_gdf, output_geoparquet_path)
    """
    print(f"\n{'='*70}")
    print(f"Merging {len(all_results)} partition result(s)")
    print(f"{'='*70}")

    # Concatenate in partition order
    merged_results = pd.concat([all_results[i] for i in sorted(all_results.keys())], ignore_index=False)
    merged_results = merged_results[~merged_results.index.duplicated(keep="first")]

    print(f"\nMerged results: {len(merged_results)} parcels (input: {len(gdf_parcels)})")
    print(f"Output columns: {merged_results.shape[1]}")

    # Create output paths
    output_dir.mkdir(parents=True, exist_ok=True)
    output_geoparquet = output_dir / "satellite_parcel_time_stats.geoparquet"

    print(f"\n{'='*70}")
    print(f"Saving final merged results")
    print(f"{'='*70}")

    # Save as GeoParquet
    write_data(merged_results, str(output_geoparquet))
    print(f"\nSaved final output: {output_geoparquet}")
    return merged_results, output_geoparquet


def run_parallel_extractions(
    spatial_parts: list[gpd.GeoDataFrame],
    users_list: list[tuple[str, str]],
    config: dict,
) -> dict[int, gpd.GeoDataFrame]:
    """Run parallel openEO extractions for multiple spatial partitions.

    Each partition is extracted independently using a different user account
    (round-robin assignment) to bypass CDSE per-user job limits.

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
        - calculate_sentinel2_indices: whether to calculate indices
        - sentinel1_bands: Sentinel-1 bands
        - spatial_statistics: statistics to calculate
        - tile_size_metres, tile_buffer_metres: tiling parameters
        - Plus all additional options: IQR parameters, fill nulls,
          interpolation method, job poll seconds, batch workers, etc.

    Returns
    -------
    dict[int, GeoDataFrame]
        Results indexed by partition number
    """
    print(f"\n{'='*70}")
    print(f"Running extraction on {len(spatial_parts)} spatial partition(s) IN PARALLEL")
    print(f"{'='*70}\n")

    all_results = {}
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=len(spatial_parts)) as executor:
        # Submit all partition jobs
        futures = {
            executor.submit(
                run_partition_extractor,
                part_idx,
                gdf_part,
                users_list,
                len(spatial_parts),
                config,
            ): part_idx
            for part_idx, gdf_part in enumerate(spatial_parts, start=1)
        }

        # Collect results as they complete
        for future in as_completed(futures):
            part_idx, results = future.result()
            all_results[part_idx] = results
            print(f"[Completed] Partition {part_idx} result stored")

    elapsed = time.time() - start_time
    print(f"\n✓ All {len(spatial_parts)} partition(s) completed in {elapsed:.1f}s")

    return all_results
