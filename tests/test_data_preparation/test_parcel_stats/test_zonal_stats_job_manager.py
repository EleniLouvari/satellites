"""Local-only tests for the tile job-manager implementation."""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

import data_preparation.parcel_stats.job_manager as job_manager_module
from data_preparation.parcel_stats.job_manager import (
    OpenEOJobManagerZonalStats,
    compute_tile_buffer_metres,
    compute_tile_width,
)
from tests.utils import expect_equal, expect_in, expect_true


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

    expect_equal(buffer_metres, 501)


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

    expect_equal(width, 10_000)


def test_tile_width_uses_smallest_candidate_when_no_width_fills_requested_slots() -> None:
    first = gpd.GeoDataFrame(
        {"parcel_id": ["a", "b"]},
        geometry=[box(0, 0, 100, 100), box(9_000, 0, 9_100, 100)],
        crs="EPSG:3857",
    )
    second = gpd.GeoDataFrame(
        {"parcel_id": ["c", "d"]},
        geometry=[box(0, 1_000, 100, 1_100), box(9_000, 1_000, 9_100, 1_100)],
        crs="EPSG:3857",
    )
    parcels = gpd.GeoDataFrame(pd.concat([first, second], ignore_index=True), crs="EPSG:3857")

    width = compute_tile_width(
        parcels,
        "parcel_id",
        3857,
        spatial_parts=[first, second],
        user_count=2,
        jobs_per_user=3,
        candidate_widths=(5_000, 10_000),
    )

    expect_equal(width, 5_000)


def test_tile_width_rejects_duplicate_candidate_widths() -> None:
    parcels = gpd.GeoDataFrame(
        {"parcel_id": ["a", "b"]},
        geometry=[box(0, 0, 100, 100), box(9_000, 0, 9_100, 100)],
        crs="EPSG:3857",
    )
    with pytest.raises(ValueError, match="candidate_widths must contain unique positive integers"):
        compute_tile_width(
            parcels,
            "parcel_id",
            3857,
            candidate_widths=(5_000, 5_000, 10_000),
        )


def test_tile_width_rejects_spatial_parts_missing_any_input_parcel() -> None:
    first = gpd.GeoDataFrame(
        {"parcel_id": ["a"]},
        geometry=[box(0, 0, 100, 100)],
        crs="EPSG:3857",
    )
    second = gpd.GeoDataFrame(
        {"parcel_id": ["b"]},
        geometry=[box(9_000, 0, 9_100, 100)],
        crs="EPSG:3857",
    )
    parcels = gpd.GeoDataFrame(pd.concat([first, second], ignore_index=True), crs="EPSG:3857")

    with pytest.raises(ValueError, match="spatial_parts must contain every input parcel exactly once"):
        compute_tile_width(
            parcels,
            "parcel_id",
            3857,
            spatial_parts=[first],
        )


def _planner(parcels: gpd.GeoDataFrame) -> OpenEOJobManagerZonalStats:
    planner = object.__new__(OpenEOJobManagerZonalStats)
    planner.parcels = parcels
    planner.working_epsg = 3857
    planner.tile_size_metres = 10
    planner.tile_buffer_metres = 0
    planner.fit_tiles_to_parcels = False
    planner.PARCEL_ID_FIELD = "parcel_id"
    planner.logger = logging.getLogger("test_job_manager_planner")
    planner.source_logger = planner.logger
    # Geometry-only fixture: give tabular cache lookups a stable namespace.
    planner._statistics_signature = lambda: "test-planner"
    return planner


def test_job_title_contains_the_run_identifier() -> None:
    planner = object.__new__(OpenEOJobManagerZonalStats)
    planner.run_identifier = "kozani autumn 2026"

    title = planner._job_title({"tile_id": "tile_00003", "batch_number": 4})

    expect_equal(title, "Zonal statistics [kozani autumn 2026] tile_00003 batch 4")


def test_default_job_retry_limit_is_three() -> None:
    expect_equal(OpenEOJobManagerZonalStats.DEFAULT_MAX_JOB_RETRIES, 3)


def test_run_identifier_rejects_empty_labels() -> None:
    planner = object.__new__(OpenEOJobManagerZonalStats)
    with pytest.raises(ValueError, match="must not be empty"):
        planner._validate_run_identifier("   ")


def test_error_job_is_reset_and_retry_is_persisted() -> None:
    planner = object.__new__(OpenEOJobManagerZonalStats)
    planner.max_job_retries = 2
    planner.source_logger = logging.getLogger("test_job_manager_retry")
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

    expect_equal(reset_count, 1)
    expect_equal(job_db.df.loc[0, "status"], "not_started")
    expect_equal(job_db.df.loc[0, "retry_count"], 1)
    expect_equal(job_db.df.loc[0, "failed_job_ids"], "failed-5")
    expect_equal(job_db.df.loc[0, "retry_mode"], "replace")
    expect_equal(job_db.df.loc[1, "status"], "finished")
    expect_equal(job_db.df.loc[1, "retry_count"], 0)


def test_start_failure_retries_the_existing_job() -> None:
    planner = object.__new__(OpenEOJobManagerZonalStats)
    planner.max_job_retries = 2
    planner.source_logger = logging.getLogger("test_job_manager_start_retry")
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

    expect_equal(reset_count, 1)
    expect_equal(job_db.df.loc[0, "status"], "not_started")
    expect_equal(job_db.df.loc[0, "retry_mode"], "restart")
    expect_equal(job_db.df.loc[0, "id"], "created-but-not-started")
    expect_equal(job_db.df.loc[0, "failed_job_ids"], "")


def test_restart_retry_reuses_remote_job_without_building_a_replacement() -> None:
    planner = object.__new__(OpenEOJobManagerZonalStats)
    expected_job = object()

    class RecordingConnection:
        def job(self, job_id):
            expect_equal(job_id, "existing-job")
            return expected_job

    planner._build_tile_cube = lambda *_args, **_kwargs: pytest.fail("A replacement cube must not be built.")

    job, reused = planner._build_or_reuse_tile_job(
        pd.Series({"id": "existing-job", "retry_mode": "restart"}),
        RecordingConnection(),
        {"spatial_extent": {}},
    )

    expect_true(job is expected_job)
    expect_true(reused is True)


def test_failed_job_is_not_reset_after_retry_limit() -> None:
    planner = object.__new__(OpenEOJobManagerZonalStats)
    planner.max_job_retries = 2
    planner.source_logger = logging.getLogger("test_job_manager_retry_limit")
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

    expect_equal(reset_count, 0)
    expect_equal(job_db.df.loc[0, "status"], "error")
    expect_equal(job_db.df.loc[0, "retry_count"], 2)


@pytest.mark.parametrize("temporal_reducer", ["median", "none"])
def test_sentinel1_only_tile_graph_does_not_load_sentinel2(temporal_reducer) -> None:
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

    extractor = object.__new__(OpenEOJobManagerZonalStats)
    extractor.sentinel2_bands = ()
    extractor.sentinel2_indices = ()
    extractor.sentinel1_bands = ("VV", "VH")
    extractor.sentinel1_indices = ()
    extractor.sentinel1_orbit_direction = "ASCENDING"
    extractor.start_date = pd.Timestamp("2024-01-01")
    extractor.end_date = pd.Timestamp("2024-03-31")
    extractor.temporal_period = "1M"
    extractor.temporal_reducer = temporal_reducer
    extractor.working_epsg = 32634

    extractor._build_tile_cube(RecordingConnection(), {"west": 0, "south": 0, "east": 1, "north": 1, "crs": "EPSG:32634"})

    loaded_collections = [call[1] for call in calls if call[0] == "load_collection"]
    expect_equal(loaded_collections, [extractor.SENTINEL1_COLLECTION])
    sentinel1_options = next(call[2] for call in calls if call[:2] == ("load_collection", extractor.SENTINEL1_COLLECTION))
    orbit_filter = sentinel1_options["properties"]["sat:orbit_state"]
    expect_true(orbit_filter("ASCENDING") is True)
    expect_true(orbit_filter("DESCENDING") is False)
    expect_true(any(call[0] == "resample_spatial" and call[1]["projection"] == "EPSG:32634" for call in calls))
    expect_equal(any(call[0] == "aggregate_temporal_period" for call in calls), (temporal_reducer == "median"))
    if temporal_reducer == "none":
        expect_equal(sentinel1_options["temporal_extent"], ["2024-01-01", "2024-04-01"])


def test_tile_plan_assigns_contained_parcels_to_tile_jobs(tmp_path) -> None:
    parcels = gpd.GeoDataFrame({"parcel_id": ["west", "east"]}, geometry=[box(1, 1, 2, 2), box(12, 1, 13, 2)], crs="EPSG:3857")

    plans = _planner(parcels)._build_tile_plan(tmp_path)

    expect_equal([plan["tile_id"] for plan in plans], ["tile_00000", "tile_00001"])
    expect_equal([plan["parcels"]["parcel_id"].tolist() for plan in plans], [["west"], ["east"]])
    expect_true(all("crs" in plan["spatial_extent"] for plan in plans))


def test_tile_plan_groups_nearby_parcels_in_the_same_grid_tile(tmp_path) -> None:
    parcels = gpd.GeoDataFrame(
        {"parcel_id": ["one", "two", "distant"]}, geometry=[box(1, 1, 2, 2), box(3, 1, 4, 2), box(12, 1, 13, 2)], crs="EPSG:3857"
    )

    plans = _planner(parcels)._build_tile_plan(tmp_path)

    expect_equal([plan["parcels"]["parcel_id"].tolist() for plan in plans], [["one", "two"], ["distant"]])
    expect_equal(plans[0]["spatial_extent"]["west"], 1.0)
    expect_equal(plans[0]["spatial_extent"]["east"], 11.0)


def test_tile_plan_stops_when_buffer_does_not_cover_a_parcel(tmp_path) -> None:
    parcels = gpd.GeoDataFrame(
        {"parcel_id": ["anchor", "crossing", "east"]},
        geometry=[box(1, 1, 2, 2), box(10, 1, 12, 2), box(19, 1, 20, 2)],
        crs="EPSG:3857",
    )

    with pytest.raises(ValueError, match="exceed the 0 m tile buffer"):
        _planner(parcels)._build_tile_plan(tmp_path)

    report = tmp_path / "uncovered_buffered_tile_parcels.csv"
    expect_true(report.exists())
    expect_in("crossing", report.read_text(encoding="utf-8"))


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
    expect_equal(sorted(parcel_ids), ["anchor", "crossing", "east"])
    expect_equal(len(parcel_ids), len(set(parcel_ids)))


def test_fitted_cube_uses_group_bounds_plus_buffer(tmp_path) -> None:
    parcels = gpd.GeoDataFrame(
        {"parcel_id": ["a", "b"]}, geometry=[box(1, 2, 3, 3), box(4, 4, 6, 5)], crs="EPSG:3857"
    )
    planner = _planner(parcels)
    planner.fit_tiles_to_parcels = True
    planner.tile_buffer_metres = 1

    plans = planner._build_tile_plan(tmp_path)

    expect_equal(len(plans), 1)
    expect_equal(plans[0]["spatial_extent"], {"west": 0.0, "south": 1.0, "east": 7.0, "north": 6.0, "crs": "EPSG:3857"})
    expect_equal(plans[0]["parcels"]["parcel_id"].tolist(), ["a", "b"])


@pytest.mark.parametrize("vertical", [False, True])
@pytest.mark.parametrize("end,expected_jobs", [(8, 1), (9, 2)])
def test_fitted_cube_splits_only_when_buffered_dimension_exceeds_limit(tmp_path, vertical, end, expected_jobs) -> None:
    second = box(0, end - 1, 1, end) if vertical else box(end - 1, 0, end, 1)
    parcels = gpd.GeoDataFrame({"parcel_id": ["a", "b"]}, geometry=[box(0, 0, 1, 1), second], crs="EPSG:3857")
    planner = _planner(parcels)
    planner.fit_tiles_to_parcels = True
    planner.tile_buffer_metres = 1

    plans = planner._build_tile_plan(tmp_path)

    expect_equal(len(plans), expected_jobs)
    expect_equal(sorted(parcel_id for plan in plans for parcel_id in plan["parcels"]["parcel_id"]), ["a", "b"])
    for plan in plans:
        extent = plan["spatial_extent"]
        expect_true(extent["east"] - extent["west"] <= planner.tile_size_metres)
        expect_true(extent["north"] - extent["south"] <= planner.tile_size_metres)


def test_fitted_split_keeps_full_geometries_and_is_deterministic(tmp_path) -> None:
    parcels = gpd.GeoDataFrame(
        {"parcel_id": ["a", "b", "c", "d"]},
        geometry=[box(0, 0, 6, 2), box(4, 0, 10, 2), box(20, 0, 23, 2), box(26, 0, 29, 2)],
        crs="EPSG:3857",
        index=[0, 0, 1, 1],
    )
    planner = _planner(parcels)
    planner.fit_tiles_to_parcels = True
    planner.tile_buffer_metres = 1
    plans = planner._build_tile_plan(tmp_path)
    planner.parcels = parcels.iloc[::-1]
    reversed_plans = planner._build_tile_plan(tmp_path)

    expect_equal([plan["spatial_extent"] for plan in plans], [plan["spatial_extent"] for plan in reversed_plans])
    expect_equal([plan["parcels"]["parcel_id"].tolist() for plan in plans], [
        plan["parcels"]["parcel_id"].tolist() for plan in reversed_plans
    ])
    originals = parcels.set_index("parcel_id").geometry
    assigned = []
    for plan in plans:
        group = plan["parcels"]
        bounds = group.total_bounds
        extent = plan["spatial_extent"]
        expect_equal([extent[key] for key in ("west", "south", "east", "north")], pytest.approx(
            [bounds[0] - 1, bounds[1] - 1, bounds[2] + 1, bounds[3] + 1]
        ))
        for _, row in group.iterrows():
            expect_true(row.geometry.equals(originals.loc[row.parcel_id]))
            assigned.append(row.parcel_id)
    expect_equal(sorted(assigned), ["a", "b", "c", "d"])


def test_fitted_plan_rejects_single_parcel_larger_than_limit(tmp_path) -> None:
    parcels = gpd.GeoDataFrame({"parcel_id": ["too-wide"]}, geometry=[box(0, 0, 9, 1)], crs="EPSG:3857")
    planner = _planner(parcels)
    planner.fit_tiles_to_parcels = True
    planner.tile_buffer_metres = 1

    with pytest.raises(ValueError, match="Cannot split whole parcels.*too-wide"):
        planner._build_tile_plan(tmp_path)


def test_fitted_plan_uses_separate_cache_namespace(monkeypatch) -> None:
    from data_preparation.parcel_stats.openeo import OpenEOZonalStats

    monkeypatch.setattr(OpenEOZonalStats, "_run_signature", lambda self: "base")
    planner = _planner(gpd.GeoDataFrame())
    grid_signature = planner._run_signature()
    planner.fit_tiles_to_parcels = True
    fitted_signature = planner._run_signature()
    expect_true(fitted_signature != grid_signature)
    planner.tile_buffer_metres = 1
    expect_true(planner._run_signature() != fitted_signature)


def test_fitted_constructor_rejects_buffer_that_leaves_no_room_for_parcels() -> None:
    with pytest.raises(ValueError, match="must exceed twice"):
        OpenEOJobManagerZonalStats(tile_size_metres=10, tile_buffer_metres=5, fit_tiles_to_parcels=True)


def test_fitted_plan_accepts_necessary_singleton_jobs(tmp_path) -> None:
    parcels = gpd.GeoDataFrame(
        {"parcel_id": [str(i) for i in range(10)]},
        geometry=[box(i * 20, 0, i * 20 + 1, 1) for i in range(10)],
        crs="EPSG:3857",
    )
    planner = _planner(parcels)
    planner.fit_tiles_to_parcels = True
    plans = planner._build_tile_plan(tmp_path)
    expect_equal(len(plans), 10)
    # Cached files keep this check local; the grid-only singleton guard must not fire.
    for plan in plans:
        Path(plan["target_path"]).touch()
    expect_equal(len(planner._run_openeo_jobs(tmp_path)), 10)


def test_resume_restores_only_missing_cubes_from_finished_jobs(tmp_path) -> None:
    planner = _planner(gpd.GeoDataFrame())
    plans = {number: {"batch_number": number, "target_path": str(tmp_path / f"batch_{number}.nc")} for number in (1, 2, 3)}
    Path(plans[2]["target_path"]).touch()
    database = _MemoryJobDatabase(
        pd.DataFrame({"batch_number": [1, 2, 3], "id": ["one", "two", "three"], "status": ["finished", "finished", "error"]})
    )
    downloads = []

    class Connection:
        def job(self, job_id):
            return job_id

    class Manager:
        def on_job_done(self, job, row):
            downloads.append((job, row.target_path))
            Path(row.target_path).write_bytes(b"restored")

    planner._restore_finished_tile_downloads(Manager(), Connection(), database, plans)

    expect_equal(downloads, [("one", plans[1]["target_path"])])
    expect_equal(Path(plans[1]["target_path"]).read_bytes(), b"restored")
    expect_equal(database.df.status.tolist(), ["finished", "finished", "error"])


def test_resume_reports_unavailable_remote_results_without_replacing_job(tmp_path) -> None:
    planner = _planner(gpd.GeoDataFrame())
    plans = {1: {"batch_number": 1, "target_path": str(tmp_path / "missing.nc")}}
    database = _MemoryJobDatabase(pd.DataFrame({"batch_number": [1], "id": ["expired"], "status": ["finished"]}))

    class Connection:
        def job(self, job_id):
            raise RuntimeError("job results expired")

    with pytest.raises(RuntimeError, match="Could not restore raw cube for batch 1"):
        planner._restore_finished_tile_downloads(None, Connection(), database, plans)
    expect_equal(database.df.status.tolist(), ["finished"])


def test_run_openeo_jobs_rejects_pathological_non_fitted_singleton_plan(tmp_path) -> None:
    planner = object.__new__(OpenEOJobManagerZonalStats)
    planner.fit_tiles_to_parcels = False
    planner.source_logger = logging.getLogger("test_job_manager_pathological_guard")
    planner._build_tile_plan = lambda _cube_dir: [
        {"batch_number": idx, "parcels": gpd.GeoDataFrame({"parcel_id": [str(idx)]}), "target_path": str(tmp_path / f"{idx}.nc")}
        for idx in range(10)
    ]

    with pytest.raises(RuntimeError, match="pathological tile plan"):
        planner._run_openeo_jobs(tmp_path)


def test_run_openeo_jobs_returns_pending_batches_when_every_tile_is_cached(tmp_path, monkeypatch) -> None:
    planner = object.__new__(OpenEOJobManagerZonalStats)
    planner.fit_tiles_to_parcels = False
    planner.source_logger = logging.getLogger("test_job_manager_cached_batches")
    parcels = gpd.GeoDataFrame({"parcel_id": ["a"]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:3857")
    target_path = tmp_path / "batch_00001_monthly.nc"
    planner._build_tile_plan = lambda _cube_dir: [
        {
            "batch_number": 1,
            "parcels": parcels,
            "target_path": str(target_path),
            "tile_id": "tile_00000",
            "spatial_extent": {"west": 0.0, "south": 0.0, "east": 1.0, "north": 1.0, "crs": "EPSG:3857"},
        }
    ]
    planner._batch_available = lambda _path: True
    monkeypatch.setattr(job_manager_module.openeo, "connect", lambda *_args, **_kwargs: pytest.fail("openEO must not connect"))

    pending = planner._run_openeo_jobs(tmp_path)

    expect_equal(len(pending), 1)
    expect_equal(pending[0][0], 1)
    expect_true(pending[0][1] is parcels)
    expect_equal(pending[0][2], target_path)


def test_run_openeo_jobs_reports_failed_jobs_when_outputs_are_missing(tmp_path, monkeypatch) -> None:
    planner = object.__new__(OpenEOJobManagerZonalStats)
    planner.fit_tiles_to_parcels = False
    planner.source_logger = logging.getLogger("test_job_manager_missing_outputs")
    planner.OPENEO_URL = "https://backend.example"
    planner.run_identifier = "run-42"
    planner.job_poll_seconds = 0
    planner.openeo_parallel_jobs = 1
    planner.max_job_retries = 1
    planner.job_retry_delay_seconds = 0
    planner.RETRYABLE_JOB_FAILURE_STATUSES = ["error", "start_failed", "queued_for_start_failed", "skipped"]
    planner._authenticate_openeo_connection = lambda _connection: None
    planner._restore_finished_tile_downloads = lambda *_args, **_kwargs: None
    planner._build_or_reuse_tile_job = lambda *_args, **_kwargs: pytest.fail("No jobs should be started in this scenario")
    planner._reset_retryable_failed_jobs = lambda _job_db: 0
    planner._wait_before_job_retry = lambda _retry_count: None
    planner._batch_available = lambda _path: False
    planner._build_tile_plan = lambda _cube_dir: [
        {
            "batch_number": 1,
            "parcels": gpd.GeoDataFrame({"parcel_id": ["one"]}),
            "target_path": str(tmp_path / "missing.nc"),
            "tile_id": "tile_00000",
            "spatial_extent": {"west": 0.0, "south": 0.0, "east": 1.0, "north": 1.0, "crs": "EPSG:3857"},
        }
    ]

    class _Connection:
        pass

    class _Manager:
        def __init__(self, **_kwargs):
            self.ran = False

        def add_backend(self, *_args, **_kwargs):
            return None

        def run_jobs(self, **_kwargs):
            self.ran = True

    class _JobDB:
        def __init__(self):
            self.df = pd.DataFrame(
                {
                    "batch_number": [1],
                    "id": ["failed-123"],
                    "status": ["error"],
                    "retry_count": [2],
                    "failed_job_ids": [""],
                    "retry_mode": [""],
                }
            )

        def get_by_status(self, statuses, max=None):
            selected = self.df.loc[self.df["status"].isin(statuses)]
            return selected.head(max) if max is not None else selected

        def persist(self, dataframe):
            for column in dataframe.columns:
                self.df.loc[dataframe.index, column] = dataframe[column]

    monkeypatch.setattr(job_manager_module.openeo, "connect", lambda *_args, **_kwargs: _Connection())
    monkeypatch.setattr(job_manager_module, "_NetCDFTileJobManager", _Manager)
    monkeypatch.setattr(job_manager_module, "create_job_db", lambda _path, df: _JobDB())

    with pytest.raises(RuntimeError, match="Failed jobs: batch 1 job failed-123 with status error after 2 retries"):
        planner._run_openeo_jobs(tmp_path)
