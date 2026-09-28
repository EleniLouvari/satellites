"""Schedule whole partitions in one sequential queue per openEO account."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd

from satellites.data_preparation.parcel_stats.job_manager import JobManagerSatelliteZonalStats


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
        - sentinel1_bands, sentinel1_indices, sentinel1_orbit_direction: Sentinel-1 config
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
    sentinel1_bands : list[str]
        Sentinel-1 bands
    sentinel1_indices : list[str]
        Sentinel-1 indices (R and/or RVI)
    sentinel1_orbit_direction : str
        Sentinel-1 orbit selection: ASCENDING, DESCENDING, or BOTH
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
    sentinel2_indices = config.get("sentinel2_indices", [])
    sentinel1_bands = config["sentinel1_bands"]
    sentinel1_indices = config.get("sentinel1_indices", [])
    spatial_statistics = config["spatial_statistics"]
    tile_size_metres = config["tile_size_metres"]
    tile_buffer_metres = config["tile_buffer_metres"]
    known_keys = {
        "output_dir", "run_identifier", "parcel_id_field", "start_date", "end_date",
        "working_epsg", "sentinel2_bands", "sentinel2_indices",
        "sentinel1_bands", "sentinel1_indices", "spatial_statistics", "tile_size_metres", "tile_buffer_metres",
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
        sentinel2_indices=sentinel2_indices,
        sentinel1_bands=sentinel1_bands,
        sentinel1_indices=sentinel1_indices,
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


def run_shared_partitions(
    spatial_parts: list[gpd.GeoDataFrame],
    users_list: list[tuple[str, str]],
    config: dict,
) -> None:
    """Run validated partitions in one sequential round-robin queue per account.

    Accounts run concurrently, while each partition's extractor controls its
    account's remote job slots. Called after validation in run_parallel_extractions.
    """
    user_queues: list[list[tuple[int, gpd.GeoDataFrame]]] = [[] for _ in users_list]
    for part_idx, gdf_part in enumerate(spatial_parts, start=1):
        user_queues[(part_idx - 1) % len(users_list)].append((part_idx, gdf_part))

    def run_user_queue(queue: list[tuple[int, gpd.GeoDataFrame]]) -> list[int]:
        completed = []
        for part_idx, gdf_part in queue:
            returned_idx, _ = run_partition_extractor(part_idx, gdf_part, users_list, len(spatial_parts), config)
            completed.append(returned_idx)
            print(f"[Completed] Partition {returned_idx} result stored")
        return completed

    active_queues = [queue for queue in user_queues if queue]
    with ThreadPoolExecutor(max_workers=len(active_queues)) as executor:
        futures = [executor.submit(run_user_queue, queue) for queue in active_queues]
        for future in as_completed(futures):
            future.result()
