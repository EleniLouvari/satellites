"""Multi-user spatial partitioning for parallel openEO extraction.

This module orchestrates splitting large parcel GeoDataFrames into spatial
partitions and running independent extractors in parallel, each using a different
openEO user account to bypass per-user concurrent job limits (2 jobs max on CDSE).
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import pandas as pd

from common_libraries.io_library import write_data
from .zonal_stats_job_manager import JobManagerSatelliteZonalStats

PARTITION_RESULT_FILE_NAME = "satellite_parcel_time_stats.geoparquet"


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
    print("Each spatial partition will use one user (round-robin assignment)")
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

    # Unpack known keys from config; remaining keys are passed as kwargs
    output_dir = Path(config["output_dir"])
    run_identifier = config["run_identifier"]
    parcel_id_field = config["parcel_id_field"]
    start_date = config["start_date"]
    end_date = config["end_date"]
    working_epsg = config["working_epsg"]
    sentinel2_bands = config["sentinel2_bands"]
    calculate_sentinel2_indices = config["calculate_sentinel2_indices"]
    sentinel2_indices = config["sentinel2_indices"]
    sentinel1_bands = config["sentinel1_bands"]
    spatial_statistics = config["spatial_statistics"]
    tile_size_metres = config["tile_size_metres"]
    tile_buffer_metres = config["tile_buffer_metres"]
    known_keys = {
        "output_dir", "run_identifier", "parcel_id_field", "start_date", "end_date",
        "working_epsg", "sentinel2_bands", "calculate_sentinel2_indices", "sentinel2_indices",
        "sentinel1_bands", "spatial_statistics", "tile_size_metres", "tile_buffer_metres",
    }
    kwargs = {k: v for k, v in config.items() if k not in known_keys}

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
            f"No partition results found under {output_dir}. Expected files matching "
            f"partition_*/{PARTITION_RESULT_FILE_NAME}."
        )

    indexed_paths = []
    for result_path in result_paths:
        partition_suffix = result_path.parent.name.removeprefix("partition_")
        try:
            partition_idx = int(partition_suffix)
        except ValueError as exc:
            raise ValueError(
                f"Invalid partition directory {result_path.parent.name!r}; "
                "expected partition_<number>."
            ) from exc
        indexed_paths.append((partition_idx, result_path))

    saved_results = {}
    for partition_idx, result_path in sorted(indexed_paths):
        print(f"Loading partition {partition_idx}: {result_path}")
        saved_results[partition_idx] = gpd.read_parquet(result_path)
    return saved_results


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
        raise ValueError("output_dir is required.")
    output_dir = Path(output_dir)

    if all_results is None:
        all_results = load_saved_partition_results(output_dir)
    if not all_results:
        raise ValueError("No partition results were provided to merge.")

    print(f"\n{'='*70}")
    print(f"Merging {len(all_results)} partition result(s)")
    print(f"{'='*70}")

    # Partition outputs each have their own RangeIndex, so their indexes must not
    # be used for de-duplication after concatenation.
    merged_results = pd.concat(
        [all_results[i] for i in sorted(all_results)],
        ignore_index=True,
    )
    if parcel_id_column is not None:
        if parcel_id_column not in merged_results.columns:
            raise KeyError(f"Parcel ID column {parcel_id_column!r} is missing from partition results.")
        merged_results = merged_results.drop_duplicates(parcel_id_column, keep="first").reset_index(drop=True)

    if working_epsg is not None:
        try:
            output_epsg = int(working_epsg)
            output_crs = gpd.GeoSeries([], crs=f"EPSG:{output_epsg}").crs
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid working EPSG code: {working_epsg!r}.") from exc
        if not output_crs.is_projected:
            raise ValueError("working_epsg must identify a projected CRS.")
        merged_results = merged_results.to_crs(output_crs)

    input_count = len(gdf_parcels) if gdf_parcels is not None else sum(map(len, all_results.values()))
    print(f"\nMerged results: {len(merged_results)} parcels (input: {input_count})")
    print(f"Output columns: {merged_results.shape[1]}")
    print(f"Output CRS: {merged_results.crs}")

    # Create output paths
    output_dir.mkdir(parents=True, exist_ok=True)
    output_geoparquet = output_dir / PARTITION_RESULT_FILE_NAME

    print(f"\n{'='*70}")
    print("Saving final merged results")
    print(f"{'='*70}")

    # Save as GeoParquet
    write_data(merged_results, str(output_geoparquet))
    print(f"\nSaved final output: {output_geoparquet}")
    return merged_results, output_geoparquet


def run_parallel_extractions(
    spatial_parts: list[gpd.GeoDataFrame],
    users_list: list[tuple[str, str]],
    config: dict,
) -> None:
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
    None
        Each partition writes its results to its own output directory.
    """
    print(f"\n{'='*70}")
    print(f"Running extraction on {len(spatial_parts)} spatial partition(s) IN PARALLEL")
    print(f"{'='*70}\n")

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
            part_idx, _ = future.result()
            print(f"[Completed] Partition {part_idx} result stored")

    elapsed = time.time() - start_time
    print(f"\n✓ All {len(spatial_parts)} partition(s) completed in {elapsed:.1f}s")
