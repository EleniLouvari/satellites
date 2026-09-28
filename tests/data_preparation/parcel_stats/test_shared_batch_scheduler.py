"""Offline checks for shared scheduling, per-account limits and durable resume."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue
from threading import Barrier, Lock, RLock
from types import SimpleNamespace
import time

import geopandas as gpd
import pandas as pd
import pytest
from openeo.extra.job_management import create_job_db, get_job_db
from shapely.geometry import box

from satellites.data_preparation.parcel_stats import batch_scheduler as scheduler
from satellites.data_preparation.parcel_stats.multiuser import run_parallel_extractions


def test_batch_views_preserve_concurrent_updates_and_isolate_rows(tmp_path):
    database = create_job_db(tmp_path / "jobs.parquet", df=pd.DataFrame({"batch_number": [1, 2]}))
    lock = RLock()
    views = [scheduler._BatchJobDatabase(database, index, lock) for index in range(2)]
    barrier = Barrier(2)

    def update(index):
        row = views[index].df
        barrier.wait(timeout=5)
        row["id"] = f"job-{index}"
        row["status"] = "finished"
        row["openeo_user"] = f"user-{index}"
        views[index].persist(row)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(update, range(2)))
    saved = get_job_db(tmp_path / "jobs.parquet").df
    assert saved["id"].tolist() == ["job-0", "job-1"]
    assert saved["openeo_user"].tolist() == ["user-0", "user-1"]
    assert views[0].count_by_status() == {"finished": 1}
    assert views[0].get_by_indices([1]).empty
    with pytest.raises(ValueError, match="own job row"):
        views[0].persist(views[1].df)


@pytest.mark.parametrize("slots", [1, 2])
def test_one_large_partition_uses_every_account_with_bounded_slots(monkeypatch, slots):
    shared, pinned = Queue(), {"first": Queue(), "second": Queue()}
    users = [("first", "password"), ("second", "password")]
    extractor = SimpleNamespace(OPENEO_URL="offline", OPENEO_OIDC_PASSWORD_CLIENT_ID="test")
    for batch in range(20):
        shared.put(SimpleNamespace(partition=14, plan={"batch_number": batch}, extractor=extractor))
    monkeypatch.setattr(scheduler.openeo, "connect", lambda *a, **k: SimpleNamespace(
        authenticate_oidc_resource_owner_password_credentials=lambda **kw: None,
    ))
    active = dict.fromkeys(pinned, 0)
    maxima = active.copy()
    completed = []
    lock = Lock()
    barrier = Barrier(2 * slots)

    def run(task, username, connection):
        with lock:
            active[username] += 1
            maxima[username] = max(maxima[username], active[username])
        if task.plan["batch_number"] < 2 * slots:
            barrier.wait(timeout=5)
        time.sleep(0.005)
        with lock:
            completed.append((task.plan["batch_number"], username))
            active[username] -= 1

    monkeypatch.setattr(scheduler, "_run_batch", run)
    scheduler._drain_batches(shared, pinned, users, slots)
    assert maxima == {"first": slots, "second": slots}
    assert sorted(batch for batch, _ in completed) == list(range(20))
    assert {user for _, user in completed} == {"first", "second"}


def _config(tmp_path):
    return dict(
        output_dir=tmp_path, run_identifier="test", parcel_id_field="parcel_id",
        start_date="2024-01-01", end_date="2024-01-31", working_epsg=3857,
        sentinel2_bands=["B02"], sentinel2_indices=[], sentinel1_bands=[],
        spatial_statistics=["mean"], tile_size_metres=1000, tile_buffer_metres=0,
        fit_tiles_to_parcels=True, batch_workers=1, job_retry_delay_seconds=0,
    )


def _parcels():
    return gpd.GeoDataFrame(
        {"parcel_id": ["a", "b", "c"]},
        geometry=[box(x, 0, x + 100, 100) for x in (0, 2000, 4000)], crs="EPSG:3857",
    )


def test_plans_before_acquisition_preserves_caches_and_job_owners(tmp_path, monkeypatch):
    config = _config(tmp_path)
    options = dict(config, parcels=_parcels(), output_dir=tmp_path / "partition_1", run_identifier="test_part1")
    extractor = scheduler.JobManagerSatelliteZonalStats(**options)
    cube_dir = extractor.output_dir / "monthly_cubes" / extractor._run_signature()
    cube_dir.mkdir(parents=True)
    plans = extractor._build_tile_plan(cube_dir)
    Path(plans[0]["target_path"]).touch()
    # Existing job belongs to second, even after the account list is reordered.
    row = scheduler._job_row(extractor, plans[1])
    row.update(id="remote-job", status="running", backend_name="cdse", openeo_user="second")
    create_job_db(cube_dir / "tile_jobs.parquet", df=pd.DataFrame([row]))
    acquired, finalized = [], []

    def drain(shared, pinned, users, jobs):
        assert shared.qsize() == 1
        assert pinned["second"].qsize() == 1
        assert pinned["first"].empty()
        tasks = [shared.get_nowait(), pinned["second"].get_nowait()]
        for task in tasks:
            acquired.append(task.plan["batch_number"])
            Path(task.plan["target_path"]).touch()

    def finalize(self, batches):
        assert all(path.exists() for _, _, path in batches)
        finalized.extend(number for number, _, _ in batches)

    monkeypatch.setattr(scheduler, "_drain_batches", drain)
    monkeypatch.setattr(scheduler.JobManagerSatelliteZonalStats, "run_from_batches", finalize)
    assert run_parallel_extractions([_parcels()], [("second", "pw"), ("first", "pw")], config) is None
    assert sorted(acquired) == [2, 3]
    assert finalized == [1, 2, 3]
    saved = get_job_db(cube_dir / "tile_jobs.parquet").df
    assert saved.loc[saved.batch_number == 2, "id"].iloc[0] == "remote-job"


def test_legacy_jobs_are_pinned_and_missing_owner_is_rejected(tmp_path, monkeypatch):
    config = _config(tmp_path)
    extractor = scheduler.JobManagerSatelliteZonalStats(
        **dict(config, parcels=_parcels(), output_dir=tmp_path / "partition_1", run_identifier="test_part1"),
    )
    cube_dir = extractor.output_dir / "monthly_cubes" / extractor._run_signature()
    cube_dir.mkdir(parents=True)
    plan = extractor._build_tile_plan(cube_dir)[0]
    row = scheduler._job_row(extractor, plan)
    row.pop("openeo_user")
    row.update(id="legacy-job", status="finished", backend_name="cdse")
    db = create_job_db(cube_dir / "tile_jobs.parquet", df=pd.DataFrame([row]))

    def drain(shared, pinned, users, jobs):
        assert pinned["first"].get_nowait().database.df.iloc[0]["id"] == "legacy-job"
        assert shared.qsize() == 2

    monkeypatch.setattr(scheduler, "_drain_batches", drain)
    monkeypatch.setattr(scheduler.JobManagerSatelliteZonalStats, "run_from_batches", lambda *a: None)
    run_parallel_extractions([_parcels()], [("first", "pw")], config)
    db = get_job_db(db.path)
    db.df["openeo_user"] = "missing-account"
    db.persist(db.df.copy())
    with pytest.raises(ValueError, match="required to resume"):
        run_parallel_extractions([_parcels()], [("first", "pw")], config)


def test_duplicate_accounts_rejected_for_direct_callers():
    with pytest.raises(ValueError, match="Duplicate usernames"):
        run_parallel_extractions([_parcels()], [("Same", "pw"), (" same ", "pw")], {})


@pytest.mark.parametrize("status", ["not_started", "start_failed", "finished", "error"])
def test_batch_runner_restores_or_retries_without_losing_ownership(tmp_path, monkeypatch, status):
    extractor = scheduler.JobManagerSatelliteZonalStats(**dict(_config(tmp_path), parcels=_parcels()))
    plan = extractor._build_tile_plan(tmp_path)[0]
    row = scheduler._job_row(extractor, plan)
    row.update(status=status, id="original" if status != "not_started" else "", backend_name="cdse")
    database = create_job_db(tmp_path / "tile_jobs.parquet", df=pd.DataFrame([row]))
    view = scheduler._BatchJobDatabase(database, 0, RLock())
    task = scheduler._BatchTask(14, extractor, plan, view, tmp_path)
    created, retrieved, limits = [], [], []

    def job(job_id):
        return SimpleNamespace(job_id=job_id, get_results=lambda: SimpleNamespace(
            download_file=lambda target: Path(target).write_bytes(b"offline-cube"),
        ))

    def create_job(**kwargs):
        assert view.df.iloc[0]["openeo_user"] == "owner"
        created.append(kwargs)
        return job("replacement")

    def get_job(job_id):
        retrieved.append(job_id)
        return job(job_id)

    class Manager(scheduler._NetCDFTileJobManager):
        def add_backend(self, name, connection, parallel_jobs):
            limits.append(parallel_jobs)

        def run_jobs(self, job_db, start_job):
            pending = job_db.get_by_status(["not_started"])
            assert len(pending) == 1
            remote = start_job(pending.iloc[0], connection)
            pending["id"], pending["status"] = remote.job_id, "finished"
            job_db.persist(pending)
            self.on_job_done(remote, pending.iloc[0])

    connection = SimpleNamespace(job=get_job)
    monkeypatch.setattr(scheduler, "_NetCDFTileJobManager", Manager)
    monkeypatch.setattr(extractor, "_build_tile_cube", lambda *a: SimpleNamespace(create_job=create_job))
    scheduler._run_batch(task, "owner", connection)
    assert Path(plan["target_path"]).read_bytes() == b"offline-cube"
    assert limits == [1]
    saved = get_job_db(tmp_path / "tile_jobs.parquet").df.iloc[0]
    assert saved["openeo_user"] == "owner"
    assert saved["status"] == "finished"
    assert saved["retry_count"] == (1 if status in {"error", "start_failed"} else 0)
    assert len(created) == (1 if status in {"not_started", "error"} else 0)
    assert retrieved == (["original"] if status in {"start_failed", "finished"} else [])


def test_cached_only_run_never_authenticates(tmp_path, monkeypatch):
    config = _config(tmp_path)
    extractor = scheduler.JobManagerSatelliteZonalStats(
        **dict(config, parcels=_parcels(), output_dir=tmp_path / "partition_1", run_identifier="test_part1"),
    )
    cube_dir = extractor.output_dir / "monthly_cubes" / extractor._run_signature()
    cube_dir.mkdir(parents=True)
    for plan in extractor._build_tile_plan(cube_dir):
        Path(plan["target_path"]).touch()

    def unexpected_connection(*args, **kwargs):
        pytest.fail("Cached-only runs must not contact openEO")

    monkeypatch.setattr(scheduler.openeo, "connect", unexpected_connection)
    monkeypatch.setattr(scheduler.JobManagerSatelliteZonalStats, "run_from_batches", lambda *a: None)
    run_parallel_extractions([_parcels()], [("first", "pw"), ("second", "pw")], config)


def test_failed_acquisition_prevents_incomplete_partition_results(tmp_path, monkeypatch):
    def fail(*args):
        raise RuntimeError("download failed")

    def unexpected_finalize(*args):
        pytest.fail("Do not write incomplete partition results")

    monkeypatch.setattr(scheduler, "_drain_batches", fail)
    monkeypatch.setattr(scheduler.JobManagerSatelliteZonalStats, "run_from_batches", unexpected_finalize)
    with pytest.raises(RuntimeError, match="download failed"):
        run_parallel_extractions([_parcels()], [("first", "pw")], _config(tmp_path))
