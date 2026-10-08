"""Share planned tile jobs across accounts while retaining partition checkpoints."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from threading import RLock

import openeo
import pandas as pd
from openeo.extra.job_management import FullDataFrameJobDatabase, create_job_db, get_job_db

from data_preparation.parcel_stats.job_manager import (
    OpenEOJobManagerZonalStats,
    _NetCDFTileJobManager,
)


class _BatchJobDatabase(FullDataFrameJobDatabase):
    """A single-row view with serialized writes to the partition's job database.

    Managers may run concurrently for different batches of the same partition.
    Keep the original row index and merge under one shared lock so updates cannot
    overwrite another manager's progress. No credentials are stored here.
    """

    def __init__(self, database, index, lock):
        super().__init__()
        self.database, self.index, self.lock = database, index, lock

    def exists(self):
        return self.database.exists()

    def read(self):
        with self.lock:
            return self.database.df.loc[[self.index]].copy()

    @property
    def df(self):
        return self.read()

    def get_by_indices(self, indices):
        return self.df.loc[[self.index] if self.index in indices else []]

    def persist(self, df):
        if not df.index.isin([self.index]).all():
            raise ValueError("Error: A batch manager may only update its own job row.")
        with self.lock:
            # FullDataFrameJobDatabase.update does not add new metadata columns.
            for column in df.columns.difference(self.database.df.columns):
                self.database.df[column] = None
            self.database.persist(df)


@dataclass
class _BatchTask:
    partition: int
    extractor: OpenEOJobManagerZonalStats
    plan: dict
    database: _BatchJobDatabase
    cube_dir: Path


def _job_row(extractor, plan):
    return {
        "batch_number": plan["batch_number"],
        "run_identifier": extractor.run_identifier,
        "tile_id": plan["tile_id"],
        **plan["spatial_extent"],
        "target_path": plan["target_path"],
        "retry_count": 0,
        "failed_job_ids": "",
        "retry_mode": "",
        "openeo_user": "",
        "status": "not_started",
    }


def _prepare_database(extractor, plans, cube_dir):
    """Reuse the existing partition database, including legacy job IDs."""
    path = cube_dir / "tile_jobs.parquet"
    if not path.exists():
        return create_job_db(path, df=pd.DataFrame([_job_row(extractor, plan) for plan in plans]))
    database = get_job_db(path)
    extractor._ensure_job_retry_tracking(database)
    if database.df["batch_number"].duplicated().any():
        raise ValueError(f"Error: Duplicate batch numbers in {path}.")
    if "openeo_user" not in database.df:
        database.df["openeo_user"] = ""
    known = set(database.df["batch_number"].astype(int))
    for plan in plans:
        if plan["batch_number"] not in known:
            # Legacy databases omit batches that were already cached at creation.
            row = _job_row(extractor, plan)
            index = int(database.df.index.max()) + 1 if len(database.df) else 0
            database.df.loc[index] = pd.Series(row).reindex(database.df.columns)
    database.persist(database.df.copy())
    return database


def _run_batch(task, username, connection):
    """Run one batch to completion using its persisted retry/download behavior."""
    extractor, plan, database = task.extractor, task.plan, task.database
    row = database.df.copy()
    row["openeo_user"] = username
    database.persist(row)  # Persist ownership before a remote job can be created.
    manager = _NetCDFTileJobManager(
        poll_sleep=extractor.job_poll_seconds,
        root_dir=task.cube_dir / "job_manager",
        download_results=False,
    )
    manager.add_backend("cdse", connection=connection, parallel_jobs=1)
    extractor._restore_finished_tile_downloads(manager, connection, database, {plan["batch_number"]: plan})
    if Path(plan["target_path"]).exists():
        return

    def start_job(row, connection, **_kwargs):
        job, reused = extractor._build_or_reuse_tile_job(row, connection, plan)
        print(
            f"{'Restarting' if reused else 'Created'} job {job.job_id}: "
            f"partition {task.partition}, batch {plan['batch_number']}, user {username}.",
            flush=True,
        )
        return job

    retry_count = extractor._reset_retryable_failed_jobs(database)
    extractor._wait_before_job_retry(retry_count)
    while True:
        manager.run_jobs(job_db=database, start_job=start_job)
        retry_count = extractor._reset_retryable_failed_jobs(database)
        if not retry_count:
            break
        extractor._wait_before_job_retry(retry_count)
    if not Path(plan["target_path"]).exists():
        row = database.df.iloc[0]
        raise RuntimeError(
            f"Error: Partition {task.partition}, batch {plan['batch_number']} has no NetCDF: job {row.get('id')} status {row['status']} after {row['retry_count']} retries. Inspect {task.cube_dir / 'tile_jobs.parquet'}."
        )


def _drain_batches(shared, pinned, users_list, jobs_per_user):
    """Drain cached, account-owned and shared batches through account slots.

    Prioritize existing cubes when cleanup is enabled. Each slot authenticates
    only when a download is needed and checkpoints its batch before taking more
    work. Local processing and memory cleanup share one raster lock.
    """
    cached = Queue()
    # Free already-downloaded rasters before recovering missing remote assets.
    # These tasks need no remote account and were placed in the shared queue.
    for _ in range(shared.qsize()):
        task = shared.get_nowait()
        if getattr(task.extractor, "remove_nc_after_completion", False) and Path(task.plan["target_path"]).is_file():
            cached.put(task)
        else:
            shared.put(task)

    def worker(username, password):
        """Consume available batches for one account slot until its queues empty."""
        connection = None
        while True:
            try:
                task = cached.get_nowait()
            except Empty:
                try:
                    task = pinned[username.casefold()].get_nowait()
                except Empty:
                    try:
                        task = shared.get_nowait()
                    except Empty:
                        return
            process_immediately = getattr(task.extractor, "remove_nc_after_completion", False)
            if not (process_immediately and Path(task.plan["target_path"]).is_file()):
                if connection is None:
                    connection = openeo.connect(task.extractor.OPENEO_URL, auto_validate=False)
                    connection.authenticate_oidc_resource_owner_password_credentials(
                        username=username,
                        password=password,
                        client_id=task.extractor.OPENEO_OIDC_PASSWORD_CLIENT_ID,
                    )
                _run_batch(task, username, connection)
            if process_immediately:
                # Hold the local lock through saving, deletion and collection.
                # Only a batch number returns; no tables survive in the worker.
                task.extractor._checkpoint_zonal_stats_batch(
                    Path(task.plan["target_path"]), task.plan["parcels"], task.plan["batch_number"],
                )

    with ThreadPoolExecutor(max_workers=len(users_list) * jobs_per_user) as executor:
        futures = [executor.submit(worker, user, password) for user, password in users_list for _ in range(jobs_per_user)]
        for future in as_completed(futures):
            future.result()


def run_shared_batches(spatial_parts, users_list, config, jobs_per_user):
    """Schedule partition batches across accounts and persist partition outputs.

    Reuse completed partitions when requested and preserve ownership of remote
    jobs. Cleanup mode checkpoints each batch during queue processing; final
    partition assembly then reads its saved tables without requiring NetCDFs.
    """
    users_list = [(user.strip(), password) for user, password in users_list]
    shared = Queue()
    pinned = {user.casefold(): Queue() for user, _ in users_list}
    partitions = []
    for part_idx, parcels in enumerate(spatial_parts, start=1):
        options = dict(config)
        options.update(
            parcels=parcels,
            output_dir=Path(config["output_dir"]) / f"partition_{part_idx}",
            run_identifier=f"{config['run_identifier']}_part{part_idx}",
        )
        extractor = OpenEOJobManagerZonalStats(**options)
        if extractor._load_completed_partition() is not None:
            print(f"Partition {part_idx}: reusing completed outputs.")
            continue
        cube_dir = extractor.output_dir / "monthly_cubes" / extractor._run_signature()
        cube_dir.mkdir(parents=True, exist_ok=True)
        plans = extractor._build_tile_plan(cube_dir)
        if not extractor.fit_tiles_to_parcels and len(plans) >= 10 and all(len(p["parcels"]) == 1 for p in plans):
            raise RuntimeError("Error: Refusing to submit a pathological tile plan: every remote job contains one parcel.")
        batches = [(p["batch_number"], p["parcels"], Path(p["target_path"])) for p in plans]
        partitions.append((part_idx, extractor, batches))
        uncached = [
            p for p in plans
            if not extractor._has_batch_statistics(p["target_path"])
            and (extractor.remove_nc_after_completion or not Path(p["target_path"]).exists())
        ]
        print(f"Partition {part_idx}: {len(parcels)} parcels, {len(plans)} batches, {len(uncached)} queued.")
        if not uncached:
            continue
        database = _prepare_database(extractor, plans, cube_dir)
        lock = RLock()
        indices = {int(row.batch_number): index for index, row in database.df.iterrows()}
        for plan in uncached:
            view = _BatchJobDatabase(database, indices[plan["batch_number"]], lock)
            task = _BatchTask(part_idx, extractor, plan, view, cube_dir)
            row = view.df.iloc[0]
            if extractor._clean_job_id(row.get("id")) and not Path(plan["target_path"]).is_file():
                # Older partition queues did not persist usernames. Their jobs
                # belong to the original round-robin account; keep account order
                # unchanged for the first resume of a legacy run.
                owner = extractor._clean_job_id(row.get("openeo_user"))
                owner = owner or users_list[(part_idx - 1) % len(users_list)][0]
                if owner.casefold() not in pinned:
                    raise ValueError(f"Error: Account {owner!r} is required to resume partition {part_idx} batch {plan['batch_number']}.")
                pinned[owner.casefold()].put(task)
            else:
                shared.put(task)

    print(f"Shared queue: {shared.qsize()} unsubmitted batches; {sum(q.qsize() for q in pinned.values())} owned jobs to resume.")
    if not partitions:
        return
    _drain_batches(shared, pinned, users_list, jobs_per_user)
    # Match the legacy upper bound on local workers: one partition per account,
    # each with its own configured batch_workers. Remote slots do not multiply it.
    with ThreadPoolExecutor(max_workers=min(len(users_list), len(partitions))) as executor:
        futures = {executor.submit(extractor.run_from_batches, batches): idx for idx, extractor, batches in partitions}
        for future in as_completed(futures):
            future.result()
            print(f"[Completed] Partition {futures[future]} result stored")
