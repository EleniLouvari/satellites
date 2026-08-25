from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Point

from openeo_parcel_stats_pipeline import zonal_stats_multiuser_manager as manager


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

    monkeypatch.setattr(manager, "run_partition_extractor", fake_run_partition_extractor)

    result = manager.run_parallel_extractions(
        spatial_parts=[_result(1, 0), _result(2, 0)],
        users_list=[("user", "password")],
        config={},
    )

    assert result is None
    assert sorted(completed_partitions) == [1, 2]
