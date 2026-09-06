"""Local-only tests for the tile job-manager implementation."""

from __future__ import annotations

import logging

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from openeo_parcel_stats_pipeline.zonal_stats_job_manager import (
    JobManagerSatelliteZonalStats,
    compute_tile_buffer_metres,
    compute_tile_width,
)


class _MemoryJobDatabase:
    """Small job-database double covering retry state persistence."""

    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe.copy()

    def get_by_status(self, statuses, max=None):
        selected = self.df.loc[self.df["status"].isin(statuses)]
        return selected.head(max) if max is not None else selected

    def persist(self, dataframe):
        for column in dataframe.columns:
            self.df.loc[dataframe.index, column] = dataframe[column]


def test_tile_buffer_uses_parcel_dimensions_not_square_root_area() -> None:
    parcels = gpd.GeoDataFrame({"parcel_id": ["long"]}, geometry=[box(0, 0, 1_000, 10)], crs="EPSG:3857")

    buffer_metres = compute_tile_buffer_metres(parcels, working_epsg=3857, safety_factor=1.0, round_up_to_metres=1)

    assert buffer_metres == 501


def test_tile_width_fills_two_remote_slots_in_every_user_partition() -> None:
    first = gpd.GeoDataFrame(
        {"parcel_id": ["a", "b", "c"]},
        geometry=[box(0, 0, 100, 100), box(9_000, 0, 9_100, 100), box(18_000, 0, 18_100, 100)],
        crs="EPSG:3857",
    )
    second = gpd.GeoDataFrame(
        {"parcel_id": ["d", "e", "f"]},
        geometry=[box(0, 1_000, 100, 1_100), box(9_000, 1_000, 9_100, 1_100), box(18_000, 1_000, 18_100, 1_100)],
        crs="EPSG:3857",
    )
    parcels = gpd.GeoDataFrame(pd.concat([first, second], ignore_index=True), crs="EPSG:3857")

    width = compute_tile_width(
        parcels,
        "parcel_id",
        3857,
        spatial_parts=[first, second],
        user_count=2,
        jobs_per_user=2,
        candidate_widths=(5_000, 10_000, 20_000),
    )

    assert width == 10_000


def _planner(parcels: gpd.GeoDataFrame) -> JobManagerSatelliteZonalStats:
    planner = object.__new__(JobManagerSatelliteZonalStats)
    planner.parcels = parcels
    planner.working_epsg = 3857
    planner.tile_size_metres = 10
    planner.tile_buffer_metres = 0
    planner.PARCEL_ID_FIELD = "parcel_id"
    planner.logger = logging.getLogger("test_job_manager_planner")
    planner.openeo_logger = planner.logger
    return planner


def test_job_title_contains_the_run_identifier() -> None:
    planner = object.__new__(JobManagerSatelliteZonalStats)
    planner.run_identifier = "kozani autumn 2026"

    title = planner._job_title({"tile_id": "tile_00003", "batch_number": 4})

    assert title == "Zonal statistics [kozani autumn 2026] tile_00003 batch 4"


def test_default_job_retry_limit_is_three() -> None:
    assert JobManagerSatelliteZonalStats.DEFAULT_MAX_JOB_RETRIES == 3


def test_run_identifier_rejects_empty_labels() -> None:
    planner = object.__new__(JobManagerSatelliteZonalStats)
    with pytest.raises(ValueError, match="must not be empty"):
        planner._validate_run_identifier("   ")


def test_error_job_is_reset_and_retry_is_persisted() -> None:
    planner = object.__new__(JobManagerSatelliteZonalStats)
    planner.max_job_retries = 2
    planner.openeo_logger = logging.getLogger("test_job_manager_retry")
    job_db = _MemoryJobDatabase(
        pd.DataFrame(
            {
                "batch_number": [5, 6],
                "id": ["failed-5", "finished-6"],
                "status": ["error", "finished"],
            }
        )
    )

    reset_count = planner._reset_retryable_failed_jobs(job_db)

    assert reset_count == 1
    assert job_db.df.loc[0, "status"] == "not_started"
    assert job_db.df.loc[0, "retry_count"] == 1
    assert job_db.df.loc[0, "failed_job_ids"] == "failed-5"
    assert job_db.df.loc[0, "retry_mode"] == "replace"
    assert job_db.df.loc[1, "status"] == "finished"
    assert job_db.df.loc[1, "retry_count"] == 0


def test_start_failure_retries_the_existing_job() -> None:
    planner = object.__new__(JobManagerSatelliteZonalStats)
    planner.max_job_retries = 2
    planner.openeo_logger = logging.getLogger("test_job_manager_start_retry")
    job_db = _MemoryJobDatabase(
        pd.DataFrame(
            {
                "batch_number": [6],
                "id": ["created-but-not-started"],
                "status": ["start_failed"],
            }
        )
    )

    reset_count = planner._reset_retryable_failed_jobs(job_db)

    assert reset_count == 1
    assert job_db.df.loc[0, "status"] == "not_started"
    assert job_db.df.loc[0, "retry_mode"] == "restart"
    assert job_db.df.loc[0, "id"] == "created-but-not-started"
    assert job_db.df.loc[0, "failed_job_ids"] == ""


def test_restart_retry_reuses_remote_job_without_building_a_replacement() -> None:
    planner = object.__new__(JobManagerSatelliteZonalStats)
    expected_job = object()

    class RecordingConnection:
        def job(self, job_id):
            assert job_id == "existing-job"
            return expected_job

    planner._build_tile_cube = lambda *_args, **_kwargs: pytest.fail("A replacement cube must not be built.")

    job, reused = planner._build_or_reuse_tile_job(
        pd.Series({"id": "existing-job", "retry_mode": "restart"}),
        RecordingConnection(),
        {"spatial_extent": {}},
    )

    assert job is expected_job
    assert reused is True


def test_failed_job_is_not_reset_after_retry_limit() -> None:
    planner = object.__new__(JobManagerSatelliteZonalStats)
    planner.max_job_retries = 2
    planner.openeo_logger = logging.getLogger("test_job_manager_retry_limit")
    job_db = _MemoryJobDatabase(
        pd.DataFrame(
            {
                "batch_number": [5],
                "id": ["third-failure"],
                "status": ["error"],
                "retry_count": [2],
                "failed_job_ids": ["first-failure;second-failure"],
            }
        )
    )

    reset_count = planner._reset_retryable_failed_jobs(job_db)

    assert reset_count == 0
    assert job_db.df.loc[0, "status"] == "error"
    assert job_db.df.loc[0, "retry_count"] == 2


def test_sentinel1_only_tile_graph_does_not_load_sentinel2() -> None:
    calls = []

    class RecordingCube:
        def sar_backscatter(self, **kwargs):
            calls.append(("sar_backscatter", kwargs))
            return self

        def aggregate_temporal_period(self, **kwargs):
            calls.append(("aggregate_temporal_period", kwargs))
            return self

        def resample_spatial(self, **kwargs):
            calls.append(("resample_spatial", kwargs))
            return self

    class RecordingConnection:
        def load_collection(self, collection, **kwargs):
            calls.append(("load_collection", collection, kwargs))
            return RecordingCube()

    extractor = object.__new__(JobManagerSatelliteZonalStats)
    extractor.sentinel2_bands = ()
    extractor.sentinel2_indices = ()
    extractor.sentinel1_bands = ("VV", "VH")
    extractor.sentinel1_indices = ()
    extractor.sentinel1_orbit_direction = "ASCENDING"
    extractor.start_date = pd.Timestamp("2024-01-01")
    extractor.end_date = pd.Timestamp("2024-03-31")
    extractor.temporal_period = "1M"
    extractor.temporal_reducer = "median"
    extractor.working_epsg = 32634

    extractor._build_tile_cube(RecordingConnection(), {"west": 0, "south": 0, "east": 1, "north": 1, "crs": "EPSG:32634"})

    loaded_collections = [call[1] for call in calls if call[0] == "load_collection"]
    assert loaded_collections == [extractor.SENTINEL1_COLLECTION]
    sentinel1_options = next(call[2] for call in calls if call[:2] == ("load_collection", extractor.SENTINEL1_COLLECTION))
    orbit_filter = sentinel1_options["properties"]["sat:orbit_state"]
    assert orbit_filter("ASCENDING") is True
    assert orbit_filter("DESCENDING") is False
    assert any(call[0] == "resample_spatial" and call[1]["projection"] == "EPSG:32634" for call in calls)


def test_tile_plan_assigns_contained_parcels_to_tile_jobs(tmp_path) -> None:
    parcels = gpd.GeoDataFrame({"parcel_id": ["west", "east"]}, geometry=[box(1, 1, 2, 2), box(12, 1, 13, 2)], crs="EPSG:3857")

    plans = _planner(parcels)._build_tile_plan(tmp_path)

    assert [plan["tile_id"] for plan in plans] == ["tile_00000", "tile_00001"]
    assert [plan["parcels"]["parcel_id"].tolist() for plan in plans] == [["west"], ["east"]]
    assert all("crs" in plan["spatial_extent"] for plan in plans)


def test_tile_plan_groups_nearby_parcels_in_the_same_grid_tile(tmp_path) -> None:
    parcels = gpd.GeoDataFrame(
        {"parcel_id": ["one", "two", "distant"]}, geometry=[box(1, 1, 2, 2), box(3, 1, 4, 2), box(12, 1, 13, 2)], crs="EPSG:3857"
    )

    plans = _planner(parcels)._build_tile_plan(tmp_path)

    assert [plan["parcels"]["parcel_id"].tolist() for plan in plans] == [["one", "two"], ["distant"]]
    assert plans[0]["spatial_extent"]["west"] == 1.0
    assert plans[0]["spatial_extent"]["east"] == 11.0


def test_tile_plan_stops_when_buffer_does_not_cover_a_parcel(tmp_path) -> None:
    parcels = gpd.GeoDataFrame(
        {"parcel_id": ["anchor", "crossing", "east"]},
        geometry=[box(1, 1, 2, 2), box(10, 1, 12, 2), box(19, 1, 20, 2)],
        crs="EPSG:3857",
    )

    with pytest.raises(ValueError, match="exceed the 0 m tile buffer"):
        _planner(parcels)._build_tile_plan(tmp_path)

    report = tmp_path / "uncovered_buffered_tile_parcels.csv"
    assert report.exists()
    assert "crossing" in report.read_text(encoding="utf-8")


def test_buffer_keeps_crossing_parcel_complete_in_one_owner_tile(tmp_path) -> None:
    parcels = gpd.GeoDataFrame(
        {"parcel_id": ["anchor", "crossing", "east"]},
        geometry=[box(1, 1, 2, 2), box(10, 1, 12, 2), box(19, 1, 20, 2)],
        crs="EPSG:3857",
    )
    planner = _planner(parcels)
    planner.tile_buffer_metres = 2

    plans = planner._build_tile_plan(tmp_path)

    parcel_ids = [parcel_id for plan in plans for parcel_id in plan["parcels"]["parcel_id"]]
    assert sorted(parcel_ids) == ["anchor", "crossing", "east"]
    assert len(parcel_ids) == len(set(parcel_ids))
