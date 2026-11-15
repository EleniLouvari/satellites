"""Worker service: one spawned child per claimed job, with child-held orphan fencing."""

import argparse
import multiprocessing
import os
import time
from contextlib import contextmanager
from pathlib import Path

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


def child_main(job, root, lane, ready=None, group_name=None):
    """Execute in fresh memory while retaining the filesystem lock until process exit."""
    from parcel_app.jobs.dispatch import execute_task
    from parcel_app.jobs.process_control import isolate_task

    isolate_task(group_name)
    if ready is not None:
        ready.set()
    with process_lock(root, lane):
        execute_task(job)


def run_worker(once=False, lane="all"):
    """Hold a PostgreSQL singleton and maintain heartbeats while a spawned child works."""
    from parcel_app.core import db
    from parcel_app.jobs.process_control import task_group
    from parcel_app.jobs.queue import claim, reconcile
    from parcel_app.settings import get_settings

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
    from parcel_app.core import db
    from parcel_app.jobs.process_control import stop_task
    from parcel_app.jobs.queue import stop_requested

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
