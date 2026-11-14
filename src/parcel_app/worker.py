"""Durable single-worker queue with child-held orphan fencing."""

import argparse
import multiprocessing
import os
import time
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

LOCK_KEY = 724931802


@contextmanager
def process_lock(root, lane="all"):
    """Fence the complete task lifetime, including orphan children, with an OS lock."""
    if lane == "all":
        with process_lock(root, "run"), process_lock(root, "small"):
            yield
        return
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / f".parcel-web-worker-{lane}.lock").open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def enqueue(kind, target_id, user_id, payload=None, conn=None):
    """Insert an idempotent pending task in the caller's transaction."""
    from psycopg2.extras import Json
    from . import db

    return db.one(
        """INSERT INTO web_jobs(job_id,app_user_id,kind,target_id,payload)
        VALUES(%s,%s,%s,%s,%s) ON CONFLICT(kind,target_id) WHERE status IN ('pending','running')
        DO UPDATE SET target_id=EXCLUDED.target_id RETURNING *""",
        (str(uuid4()), user_id, kind, str(target_id), Json(payload or {})),
        conn=conn,
    )


def claim(lane="all"):
    """Atomically claim one pending task with a fresh fencing token."""
    from . import db

    kinds = ["run", "resume", "bundle", "eda"] if lane == "run" else ["inspect", "validate", "test_account"]
    if lane == "all":
        kinds += ["run", "resume", "bundle", "eda"]
    return db.one(
        """UPDATE web_jobs SET status='running',claim_token=%s,heartbeat_at=now(),updated_at=now()
        WHERE job_id=(SELECT job_id FROM web_jobs WHERE status='pending' AND kind=ANY(%s)
        ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT 1) RETURNING *""",
        (str(uuid4()), kinds),
    )


def reconcile(lane="all"):
    """Mark abandoned work interrupted only while the caller holds both worker locks."""
    from . import db

    kinds = ["run", "resume", "bundle", "eda"] if lane == "run" else ["inspect", "validate", "test_account"]
    if lane == "all":
        kinds += ["run", "resume", "bundle", "eda"]
    with db.connection() as conn:
        # Stop requests and task publication lock jobs before analysis rows.
        db.all_rows(
            "SELECT job_id FROM web_jobs WHERE status='running' AND kind=ANY(%s) ORDER BY job_id FOR UPDATE",
            (kinds,),
            conn=conn,
        )
        db.execute(
            """UPDATE eda_runs SET status='interrupted',finished_at=now(),
            error='Error: EDA worker interrupted; create a new analysis with these settings.'
            WHERE eda_id::text IN (SELECT target_id FROM web_jobs WHERE status='running' AND kind='eda' AND kind=ANY(%s))
            AND status IN ('queued','running')""",
            (kinds,),
            conn=conn,
        )
        db.execute(
            """UPDATE analysis_runs r SET status='interrupted',error=CASE
            WHEN j.payload->>'stop_requested'='true' THEN 'Stopped by request. Resume analysis to reuse completed checkpoints.'
            ELSE 'Error: Worker interrupted; resume explicitly.' END,updated_at=now()
            FROM web_jobs j WHERE r.run_id::text=j.target_id AND j.status='running'
            AND j.kind IN ('run','resume') AND j.kind=ANY(%s) AND r.status IN ('queued','running')""",
            (kinds,),
            conn=conn,
        )
        db.execute(
            """UPDATE analysis_runs SET status='draft',error='Error: Validation interrupted; validate again.'
            WHERE status='validating' AND run_id::text IN
            (SELECT target_id FROM web_jobs WHERE status='running' AND kind='validate' AND kind=ANY(%s))""",
            (kinds,),
            conn=conn,
        )
        db.execute(
            """UPDATE parcel_uploads SET status='failed',error='Error: Inspection interrupted; upload again.'
            WHERE upload_id::text IN (SELECT target_id FROM web_jobs WHERE status='running' AND kind='inspect' AND kind=ANY(%s))""",
            (kinds,),
            conn=conn,
        )
        db.execute(
            """UPDATE web_jobs SET status='failed',error='Error: Worker interrupted.',updated_at=now()
            WHERE status='running' AND kind=ANY(%s)""",
            (kinds,),
            conn=conn,
        )


def stop_requested(job):
    """Read a stop request only for the exact active parcel-analysis claim."""
    from . import db

    if job["kind"] not in {"run", "resume"}:
        return False
    row = db.one(
        "SELECT payload FROM web_jobs WHERE job_id=%s AND claim_token=%s AND status='running'",
        (str(job["job_id"]), str(job["claim_token"])),
    )
    return bool(row and row["payload"].get("stop_requested"))


def child_main(job, root, lane, ready=None, group_name=None):
    """Execute in fresh memory while retaining the filesystem lock until process exit."""
    from .tasks import execute_task
    from .process_control import isolate_task

    isolate_task(group_name)
    if ready is not None:
        ready.set()
    with process_lock(root, lane):
        execute_task(job)


def run_worker(once=False, lane="all"):
    """Hold a PostgreSQL singleton and maintain heartbeats while a spawned child works."""
    from . import db
    from .settings import get_settings
    from .process_control import task_group

    root = get_settings().storage_root
    # This connection must remain open and must never be shared with a child.
    with db.connection() as singleton:
        row = db.one("SELECT pg_try_advisory_lock(%s) AS acquired", (LOCK_KEY + (1 if lane == "small" else 0),), conn=singleton)
        if not row["acquired"]:
            raise RuntimeError("Error: Another worker is active.")
        if lane == "all":
            extra = db.one("SELECT pg_try_advisory_lock(%s) AS acquired", (LOCK_KEY + 1,), conn=singleton)
            if not extra["acquired"]:
                raise RuntimeError("Error: Another metadata worker is active.")
        singleton.commit()
        with process_lock(root, lane):
            reconcile(lane)
        while True:
            # Verify the singleton session before claiming any further work.
            db.one("SELECT 1 AS alive", conn=singleton)
            singleton.commit()
            job = claim(lane)
            if job is None:
                if once:
                    return
                time.sleep(2)
                continue
            context = multiprocessing.get_context("spawn")
            ready = context.Event()
            with task_group() as group:
                child = context.Process(
                    target=child_main, args=(job, str(root), lane, ready, group.name if group else None),
                )
                child.start()
                supervise_task(child, ready, group, job, singleton)
                # Native crashes cannot run the child's exception handler.
                with process_lock(root, lane):
                    reconcile(lane)
            if once:
                return


def supervise_task(child, ready, group, job, singleton):
    """Finish the entire task group before allowing reconciliation or Resume."""
    from . import db
    from .process_control import stop_task

    while child.is_alive():
        child.join(timeout=5)
        db.one("SELECT 1 AS alive", conn=singleton)
        singleton.commit()
        changed = db.execute(
            """UPDATE web_jobs SET heartbeat_at=now(),updated_at=now()
            WHERE job_id=%s AND claim_token=%s AND status='running'""",
            (str(job["job_id"]), str(job["claim_token"])),
        )
        if not changed and child.is_alive():
            child.join(timeout=1)
        if ready.is_set() and stop_requested(job):
            break
    # Also clean up after early leader exit, even if Stop arrives after the last
    # poll. Descendants can otherwise keep writing after reconciliation/Resume.
    if ready.is_set():
        stop_task(child, group)


def main():
    """Run the queue service or process at most one queued task."""
    parser = argparse.ArgumentParser(description="Parcel Studio durable worker")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--lane", choices=("all", "run", "small"), default="all")
    arguments = parser.parse_args()
    run_worker(arguments.once, arguments.lane)


if __name__ == "__main__":
    main()
