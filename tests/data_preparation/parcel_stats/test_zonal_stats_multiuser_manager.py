from pathlib import Path
import threading
import time

import geopandas as gpd
import pytest
from shapely.geometry import Point

from satellites.data_preparation.parcel_stats import multiuser as manager
from satellites.data_preparation.parcel_stats import partition_scheduler


def _result(parcel_id: int, index: int) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"parcel_id": [parcel_id]},
        geometry=[Point(parcel_id, parcel_id)],
        index=[index],
        crs="EPSG:4326",
    )


def test_load_saved_partition_results_uses_numeric_partition_order(monkeypatch):
    output_dir = Path("saved-results")
    result_paths = [
        output_dir / "partition_10" / manager.PARTITION_RESULT_FILE_NAME,
        output_dir / "partition_2" / manager.PARTITION_RESULT_FILE_NAME,
    ]
    monkeypatch.setattr(Path, "glob", lambda self, pattern: result_paths)

    loaded_paths = []

    def fake_read_parquet(path: Path) -> gpd.GeoDataFrame:
        loaded_paths.append(path)
        partition_idx = int(path.parent.name.removeprefix("partition_"))
        return _result(partition_idx, partition_idx)

    monkeypatch.setattr(manager.gpd, "read_parquet", fake_read_parquet)

    results = manager.load_saved_partition_results(output_dir)

    assert list(results) == [2, 10]
    assert loaded_paths == [result_paths[1], result_paths[0]]


def test_load_saved_partition_results_requires_at_least_one_result(monkeypatch):
    monkeypatch.setattr(Path, "glob", lambda self, pattern: [])

    with pytest.raises(FileNotFoundError, match="No partition results found"):
        manager.load_saved_partition_results(Path("saved-results"))


def test_merge_and_save_results_can_load_saved_results(monkeypatch):
    output_dir = Path(".")
    # Each independently saved partition starts its own RangeIndex at zero.
    saved_results = {2: _result(202, 0), 1: _result(101, 0)}
    written = {}

    monkeypatch.setattr(
        manager,
        "load_saved_partition_results",
        lambda output_dir: saved_results,
    )

    def fake_write_data(data: gpd.GeoDataFrame, path: str) -> None:
        written["data"] = data.copy()
        written["path"] = path

    monkeypatch.setattr(manager, "write_data", fake_write_data)

    merged, output_path = manager.merge_and_save_results(output_dir=output_dir)

    expected_path = output_dir / manager.PARTITION_RESULT_FILE_NAME
    assert merged["parcel_id"].tolist() == [101, 202]
    assert written["data"].equals(merged)
    assert written["path"] == str(expected_path)
    assert output_path == expected_path


def test_merge_and_save_results_still_accepts_in_memory_results(monkeypatch):
    monkeypatch.setattr(manager, "write_data", lambda data, path: None)

    merged, _ = manager.merge_and_save_results(
        gdf_parcels=_result(101, 1),
        all_results={1: _result(101, 1)},
        output_dir=Path("."),
        parcel_id_column="parcel_id",
    )

    assert merged["parcel_id"].tolist() == [101]


def test_merge_and_save_results_deduplicates_by_parcel_id(monkeypatch):
    monkeypatch.setattr(manager, "write_data", lambda data, path: None)

    merged, _ = manager.merge_and_save_results(
        all_results={1: _result(101, 0), 2: _result(101, 0)},
        output_dir=Path("."),
        parcel_id_column="parcel_id",
    )

    assert merged["parcel_id"].tolist() == [101]


def test_merge_and_save_results_reprojects_to_working_epsg(monkeypatch):
    monkeypatch.setattr(manager, "write_data", lambda data, path: None)

    merged, _ = manager.merge_and_save_results(
        all_results={1: _result(1, 0)},
        output_dir=Path("."),
        working_epsg=3857,
    )

    assert merged.crs.to_epsg() == 3857
    assert merged.geometry.iloc[0].x == pytest.approx(111_319.49, rel=1e-5)


def test_run_parallel_extractions_returns_none_after_all_partitions_complete(monkeypatch):
    completed_partitions = []

    def fake_run_partition_extractor(
        part_idx,
        gdf_part,
        users_list,
        total_partitions,
        config,
    ):
        completed_partitions.append(part_idx)
        return part_idx, _result(part_idx, 0)

    monkeypatch.setattr(partition_scheduler, "run_partition_extractor", fake_run_partition_extractor)

    result = manager.run_parallel_extractions(
        spatial_parts=[_result(1, 0), _result(2, 0)],
        users_list=[("user", "password")],
        config={},
        scheduling="partitions",
    )

    assert result is None
    assert sorted(completed_partitions) == [1, 2]


def test_partition_count_builds_one_grid_cell_per_user() -> None:
    parcels = gpd.GeoDataFrame(
        {"parcel_id": [1, 2, 3, 4]},
        geometry=[Point(0, 0), Point(10, 0), Point(0, 10), Point(10, 10)],
        crs="EPSG:3857",
    )

    parts = manager.split_geodataframe_by_grid(parcels, partition_count=4)

    assert len(parts) == 4
    assert sorted(parcel_id for part in parts for parcel_id in part["parcel_id"]) == [1, 2, 3, 4]


def test_load_users_rejects_duplicate_accounts(monkeypatch) -> None:
    monkeypatch.setenv("OPENEO_USERS", "same,password1;same,password2")

    with pytest.raises(ValueError, match="duplicate usernames"):
        manager.load_openeo_users_from_env()


def test_parallel_extractions_rejects_more_than_two_jobs_per_user() -> None:
    with pytest.raises(ValueError, match="cannot exceed 2"):
        manager.run_parallel_extractions(
            spatial_parts=[_result(1, 0)],
            users_list=[("user", "password")],
            config={"openeo_parallel_jobs": 3},
        )


def test_parallel_extractions_never_reuses_one_user_concurrently(monkeypatch):
    users = [("first", "password"), ("second", "password")]
    lock = threading.Lock()
    active_by_user = {user: 0 for user, _ in users}
    maximum_by_user = {user: 0 for user, _ in users}
    active_total = 0
    maximum_total = 0

    def fake_run_partition_extractor(part_idx, gdf_part, users_list, total_partitions, config):
        nonlocal active_total, maximum_total
        user = users_list[(part_idx - 1) % len(users_list)][0]
        with lock:
            active_by_user[user] += 1
            active_total += 1
            maximum_by_user[user] = max(maximum_by_user[user], active_by_user[user])
            maximum_total = max(maximum_total, active_total)
        time.sleep(0.03)
        with lock:
            active_by_user[user] -= 1
            active_total -= 1
        return part_idx, _result(part_idx, 0)

    monkeypatch.setattr(partition_scheduler, "run_partition_extractor", fake_run_partition_extractor)

    manager.run_parallel_extractions(
        spatial_parts=[_result(index, index) for index in range(1, 5)],
        users_list=users,
        config={"openeo_parallel_jobs": 2},
        scheduling="partitions",
    )

    assert maximum_by_user == {"first": 1, "second": 1}
    assert maximum_total == 2
