"""Offline checks for shared scheduling, per-account limits and durable resume."""

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue
from threading import Barrier, Lock, RLock
from types import SimpleNamespace

import geopandas as gpd
import pandas as pd
import pytest
from openeo.extra.job_management import create_job_db, get_job_db
from shapely.geometry import box

from data_preparation.parcel_stats import batch_scheduler as scheduler
from data_preparation.parcel_stats.multiuser import run_parallel_extractions
from tests.utils import expect_equal, expect_is_none, expect_true


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
    expect_equal(saved["id"].tolist(), ["job-0", "job-1"])
    expect_equal(saved["openeo_user"].tolist(), ["user-0", "user-1"])
    expect_equal(views[0].count_by_status(), {"finished": 1})
    expect_true(views[0].get_by_indices([1]).empty)
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
    expect_equal(maxima, {"first": slots, "second": slots})
    expect_equal(sorted(batch for batch, _ in completed), list(range(20)))
    expect_equal({user for _, user in completed}, {"first", "second"})


def _config(tmp_path):
    return {
        "output_dir": tmp_path, "run_identifier": "test", "parcel_id_field": "parcel_id",
        "start_date": "2024-01-01", "end_date": "2024-01-31", "working_epsg": 3857,
        "sentinel2_bands": ["B02"], "sentinel2_indices": [], "sentinel1_bands": [],
        "spatial_statistics": ["mean"], "tile_size_metres": 1000, "tile_buffer_metres": 0,
        "fit_tiles_to_parcels": True, "batch_workers": 1, "job_retry_delay_seconds": 0,
    }


def _parcels():
    return gpd.GeoDataFrame(
        {"parcel_id": ["a", "b", "c"]},
        geometry=[box(x, 0, x + 100, 100) for x in (0, 2000, 4000)], crs="EPSG:3857",
    )


def test_plans_before_acquisition_preserves_caches_and_job_owners(tmp_path, monkeypatch):
    config = _config(tmp_path)
    options = dict(config, parcels=_parcels(), output_dir=tmp_path / "partition_1", run_identifier="test_part1")
    extractor = scheduler.OpenEOJobManagerZonalStats(**options)
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
        expect_equal(shared.qsize(), 1)
        expect_equal(pinned["second"].qsize(), 1)
        expect_true(pinned["first"].empty())
        tasks = [shared.get_nowait(), pinned["second"].get_nowait()]
        for task in tasks:
            acquired.append(task.plan["batch_number"])
            Path(task.plan["target_path"]).touch()

    def finalize(self, batches):
        expect_true(all(path.exists() for _, _, path in batches))
        finalized.extend(number for number, _, _ in batches)

    monkeypatch.setattr(scheduler, "_drain_batches", drain)
    monkeypatch.setattr(scheduler.OpenEOJobManagerZonalStats, "run_from_batches", finalize)
    expect_is_none(run_parallel_extractions([_parcels()], [("second", "pw"), ("first", "pw")], config))
    expect_equal(sorted(acquired), [2, 3])
    expect_equal(finalized, [1, 2, 3])
    saved = get_job_db(cube_dir / "tile_jobs.parquet").df
    expect_equal(saved.loc[saved.batch_number == 2, "id"].iloc[0], "remote-job")


def test_legacy_jobs_are_pinned_and_missing_owner_is_rejected(tmp_path, monkeypatch):
    config = _config(tmp_path)
    extractor = scheduler.OpenEOJobManagerZonalStats(
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
        expect_equal(pinned["first"].get_nowait().database.df.iloc[0]["id"], "legacy-job")
        expect_equal(shared.qsize(), 2)

    monkeypatch.setattr(scheduler, "_drain_batches", drain)
    monkeypatch.setattr(scheduler.OpenEOJobManagerZonalStats, "run_from_batches", lambda *a: None)
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
    extractor = scheduler.OpenEOJobManagerZonalStats(**dict(_config(tmp_path), parcels=_parcels()))
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
        expect_equal(view.df.iloc[0]["openeo_user"], "owner")
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
            expect_equal(len(pending), 1)
            remote = start_job(pending.iloc[0], connection)
            pending["id"], pending["status"] = remote.job_id, "finished"
            job_db.persist(pending)
            self.on_job_done(remote, pending.iloc[0])

    connection = SimpleNamespace(job=get_job)
    monkeypatch.setattr(scheduler, "_NetCDFTileJobManager", Manager)
    monkeypatch.setattr(extractor, "_build_tile_cube", lambda *a: SimpleNamespace(create_job=create_job))
    scheduler._run_batch(task, "owner", connection)
    expect_equal(Path(plan["target_path"]).read_bytes(), b"offline-cube")
    expect_equal(limits, [1])
    saved = get_job_db(tmp_path / "tile_jobs.parquet").df.iloc[0]
    expect_equal(saved["openeo_user"], "owner")
    expect_equal(saved["status"], "finished")
    expect_equal(saved["retry_count"], (1 if status in {"error", "start_failed"} else 0))
    expect_equal(len(created), (1 if status in {"not_started", "error"} else 0))
    expect_equal(retrieved, (["original"] if status in {"start_failed", "finished"} else []))


def test_cached_only_run_never_authenticates(tmp_path, monkeypatch):
    config = _config(tmp_path)
    extractor = scheduler.OpenEOJobManagerZonalStats(
        **dict(config, parcels=_parcels(), output_dir=tmp_path / "partition_1", run_identifier="test_part1"),
    )
    cube_dir = extractor.output_dir / "monthly_cubes" / extractor._run_signature()
    cube_dir.mkdir(parents=True)
    for plan in extractor._build_tile_plan(cube_dir):
        Path(plan["target_path"]).touch()

    def unexpected_connection(*args, **kwargs):
        pytest.fail("Cached-only runs must not contact openEO")

    monkeypatch.setattr(scheduler.openeo, "connect", unexpected_connection)
    monkeypatch.setattr(scheduler.OpenEOJobManagerZonalStats, "run_from_batches", lambda *a: None)
    run_parallel_extractions([_parcels()], [("first", "pw"), ("second", "pw")], config)


def test_failed_acquisition_prevents_incomplete_partition_results(tmp_path, monkeypatch):
    def fail(*args):
        raise RuntimeError("download failed")

    def unexpected_finalize(*args):
        pytest.fail("Do not write incomplete partition results")

    monkeypatch.setattr(scheduler, "_drain_batches", fail)
    monkeypatch.setattr(scheduler.OpenEOJobManagerZonalStats, "run_from_batches", unexpected_finalize)
    with pytest.raises(RuntimeError, match="download failed"):
        run_parallel_extractions([_parcels()], [("first", "pw")], _config(tmp_path))


def test_collect_queued_plans_filters_cached_and_present_targets(tmp_path):
    existing = tmp_path / "existing.nc"
    existing.touch()
    plans = [
        {"batch_number": 1, "target_path": str(tmp_path / "cached.nc")},
        {"batch_number": 2, "target_path": str(existing)},
        {"batch_number": 3, "target_path": str(tmp_path / "missing.nc")},
    ]
    extractor = SimpleNamespace(
        remove_nc_after_completion=False,
        _has_batch_statistics=lambda target: Path(target).name == "cached.nc",
    )

    queued = scheduler._collect_queued_plans(extractor, plans)
    expect_equal([plan["batch_number"] for plan in queued], [3])

    extractor.remove_nc_after_completion = True
    queued_with_cleanup = scheduler._collect_queued_plans(extractor, plans)
    expect_equal([plan["batch_number"] for plan in queued_with_cleanup], [2, 3])


def test_validate_partition_tile_plan_rejects_pathological_single_parcel_batches():
    extractor = SimpleNamespace(fit_tiles_to_parcels=False)
    plans = [{"parcels": [index], "batch_number": index} for index in range(10)]

    with pytest.raises(RuntimeError, match="pathological tile plan"):
        scheduler._validate_partition_tile_plan(extractor, plans)


def test_resolve_task_owner_falls_back_and_rejects_unknown_users():
    extractor = SimpleNamespace(_clean_job_id=lambda value: str(value).strip() if value else None)
    users_list = [("first", "pw"), ("second", "pw")]
    pinned = {"first": Queue(), "second": Queue()}
    plan = {"batch_number": 4}

    owner = scheduler._resolve_task_owner(
        part_idx=2,
        extractor=extractor,
        users_list=users_list,
        pinned=pinned,
        plan=plan,
        row={"openeo_user": ""},
    )
    expect_equal(owner, "second")

    with pytest.raises(ValueError, match="required to resume"):
        scheduler._resolve_task_owner(
            part_idx=1,
            extractor=extractor,
            users_list=users_list,
            pinned={"first": Queue()},
            plan=plan,
            row={"openeo_user": "missing-account"},
        )
