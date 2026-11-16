"""Owner-scoped, retryable deletion of remote jobs, local files and database history."""

import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from psycopg2.extras import Json

from parcel_app.core import db
from parcel_app.core.security import current_user
from parcel_app.core.storage import eda_directory, ml_directory, run_directory
from parcel_app.settings import get_settings
from parcel_app.sources.openeo.credentials import run_credentials
from parcel_app.sources.openeo.remote_cleanup import collect_jobs, delete_jobs

from .routes import owned_run

router = APIRouter()
# "paused" is an ML run waiting for its next step.
IDLE = {"draft", "ready", "complete", "failed", "interrupted", "cancelled", "deleting", "paused"}


def owned_directory(user_id, kind, identifier, stored_path, *allocated):
    """Require an allocated directory of the record, rejecting redirects through links/junctions.

    ``allocated`` lists the record's locations in the nested ``runs/<id>/...`` layout;
    records saved before that layout keep ``users/<user>/<kind>/<id>``.
    """
    root = get_settings().storage_root.resolve()
    legacy = root / "users" / str(user_id) / kind / str(identifier)
    allowed = {legacy, *(Path(os.path.abspath(path)) for path in allocated)}
    stored = Path(os.path.abspath(stored_path))
    if stored not in allowed or stored.resolve() != stored or not stored.is_relative_to(root):
        raise HTTPException(409, "Error: The saved storage location is unsafe for deletion; contact the administrator.")
    if stored.exists() and not stored.is_dir():
        raise HTTPException(409, "Error: The saved output directory is unavailable for deletion.")
    return stored


def eda_locations(user_id, eda):
    """Return an EDA analysis's ``output/`` folder and the analysis folder holding it."""
    home = eda_directory(user_id, eda["run_id"], eda["eda_id"])
    return home / "output", home


def ml_location(user_id, ml):
    """Return an ML run's folder inside its EDA analysis."""
    return ml_directory(user_id, ml["run_id"], ml["eda_id"], ml["ml_id"])


def remove_directory(path):
    """Remove an already verified owned directory; absent files make retries safe."""
    if path.resolve() != path:
        raise ValueError("Error: Storage changed during cleanup; refusing a redirected directory.")
    if path.exists():
        # Python's rmtree unlinks internal symlinks and does not traverse Windows junctions.
        shutil.rmtree(path)


@contextmanager
def deletion_connection(run_id, user_id):
    """Hold one session lock across commits so parent/child deletion cannot overlap."""
    with db.connection() as conn:
        owned_run(run_id, user_id, conn=conn)
        lock = db.one("SELECT pg_try_advisory_lock(hashtextextended(%s,0)) AS acquired", (f"delete:{run_id}",), conn=conn)
        if not lock["acquired"]:
            raise HTTPException(409, "Error: Cleanup is already in progress for this run. Please wait.")
        # Closing this dedicated connection releases the session-level lock, including on failure.
        yield conn


def ensure_idle(rows, targets, user_id, conn):
    """Lock related queue records and reject any pending or running writer."""
    if any(row["status"] not in IDLE for row in rows):
        raise HTTPException(409, "Error: Wait for processing to finish, or cancel queued work, before deleting.")
    jobs = db.all_rows(
        "SELECT status FROM web_jobs WHERE app_user_id=%s AND target_id=ANY(%s) FOR UPDATE", (user_id, targets), conn=conn
    )
    if any(job["status"] in {"pending", "running"} for job in jobs):
        raise HTTPException(409, "Error: Related processing, EDA, ML or ZIP work is still active. Try deletion when it finishes.")


def prepare_run(run_id, user_id, conn):
    """Validate every target and persist the remote manifest before any irreversible action."""
    run = owned_run(run_id, user_id, conn=conn, lock=True)
    edas = db.all_rows("SELECT * FROM eda_runs WHERE run_id=%s ORDER BY eda_id FOR UPDATE", (str(run_id),), conn=conn)
    mls = db.all_rows("SELECT * FROM ml_runs WHERE run_id=%s ORDER BY ml_id FOR UPDATE", (str(run_id),), conn=conn)
    targets = [str(run_id), *[str(row["eda_id"]) for row in edas], *[str(row["ml_id"]) for row in mls]]
    ensure_idle([run, *edas, *mls], targets, user_id, conn)
    # Old runs stored their outputs directly in the run folder; new ones use output/.
    home = owned_directory(user_id, "runs", run_id, run_directory(user_id, run_id))
    output = owned_directory(user_id, "runs", run_id, run["output_dir"], home / "output")
    directories = [
        *[owned_directory(user_id, "ml", row["ml_id"], row["output_dir"], ml_location(user_id, row)) for row in mls],
        *[owned_directory(user_id, "eda", row["eda_id"], row["output_dir"], *eda_locations(user_id, row)) for row in edas],
        output,
        home,
    ]
    upload = db.one(
        "SELECT * FROM parcel_uploads WHERE upload_id=%s AND app_user_id=%s FOR UPDATE",
        (str(run["upload_id"]), user_id),
        conn=conn,
    )
    if not upload:
        raise HTTPException(409, "Error: The run's uploaded input is unavailable.")
    owned_directory(user_id, "uploads", upload["upload_id"], Path(upload["storage_path"]).parent)
    shared = db.one(
        "SELECT 1 FROM analysis_runs WHERE upload_id=%s AND run_id<>%s LIMIT 1",
        (str(upload["upload_id"]), str(run_id)),
        conn=conn,
    )
    if not shared:
        ensure_idle([upload], [str(upload["upload_id"])], user_id, conn)
    manifest = run.get("deletion_manifest")
    if manifest is None:
        names = [
            row["username"]
            for row in db.all_rows(
                "SELECT username FROM run_accounts WHERE run_id=%s ORDER BY position", (str(run_id),), conn=conn
            )
        ]
        if names and run["status"] not in {"draft", "ready", "cancelled"} and not output.exists():
            raise HTTPException(
                409, "Error: Restore this run's storage folder before deletion so its remote jobs can be identified."
            )
        try:
            summary = run.get("summary") or {}
            expected = summary if run["status"] == "complete" and names else {}
            jobs = collect_jobs(
                output,
                names,
                expected_batches=expected.get("batch_count", 0),
                expected_partitions=expected.get("partition_count", 0),
            )
            manifest = {"jobs": jobs, "remote_deleted": 0, "remote_complete": False}
        except Exception:  # noqa: BLE001 - any unreadable job record blocks deletion with a fixed message
            raise HTTPException(
                409, "Error: Cannot identify the run's remote jobs and owners. Restore its job records before deletion."
            ) from None
        if names and run["status"] == "complete" and not manifest["jobs"]:
            raise HTTPException(
                409, "Error: This completed run has no saved remote job IDs. Restore its tile job records before deletion."
            )
    db.execute(
        "UPDATE analysis_runs SET status='deleting',error=NULL,deletion_manifest=%s WHERE run_id=%s",
        (Json(manifest), str(run_id)),
        conn=conn,
    )
    db.execute("UPDATE eda_runs SET status='deleting',error=NULL WHERE run_id=%s", (str(run_id),), conn=conn)
    db.execute("UPDATE ml_runs SET status='deleting',error=NULL WHERE run_id=%s", (str(run_id),), conn=conn)
    if not shared:
        db.execute("UPDATE parcel_uploads SET status='deleting' WHERE upload_id=%s", (str(upload["upload_id"]),), conn=conn)
    conn.commit()
    return run, manifest, directories, targets


def purge_run(run, user_id, targets, conn):
    """Recheck the last input reference under lock immediately before final cleanup."""
    upload_id, run_id = str(run["upload_id"]), str(run["run_id"])
    upload = db.one("SELECT * FROM parcel_uploads WHERE upload_id=%s FOR UPDATE", (upload_id,), conn=conn)
    shared = db.one("SELECT 1 FROM analysis_runs WHERE upload_id=%s AND run_id<>%s LIMIT 1", (upload_id, run_id), conn=conn)
    if not shared:
        ensure_idle([upload], [upload_id], user_id, conn)
        directory = owned_directory(user_id, "uploads", upload_id, Path(upload["storage_path"]).parent)
        db.execute("UPDATE parcel_uploads SET status='deleting' WHERE upload_id=%s", (upload_id,), conn=conn)
        # New drafts cannot take a share lock and validate a ready input after this commit.
        conn.commit()
        remove_directory(directory)
        targets = [*targets, upload_id]
    # For a shared input, keep its row lock until the parent deletion commits. A
    # simultaneous deletion then sees the updated reference count, avoiding orphan uploads.
    db.execute("DELETE FROM web_jobs WHERE app_user_id=%s AND target_id=ANY(%s)", (user_id, targets), conn=conn)
    # ml_runs references eda_runs, so it is removed first.
    for table in ("ml_runs", "eda_runs", "run_artifacts", "run_events", "run_accounts"):
        db.execute(f"DELETE FROM {table} WHERE run_id=%s", (run_id,), conn=conn)  # nosec B608 - fixed table tuple
    db.execute("DELETE FROM analysis_runs WHERE run_id=%s", (run_id,), conn=conn)
    if not shared:
        db.execute("DELETE FROM parcel_uploads WHERE upload_id=%s", (upload_id,), conn=conn)
    conn.commit()


def remote_cleanup(run, manifest, conn):
    """Checkpoint remote progress independently of later filesystem cleanup."""
    if manifest["remote_complete"]:
        return
    offset = manifest["remote_deleted"]
    remaining = manifest["jobs"][offset:]

    def save_progress(count):
        manifest["remote_deleted"] = offset + count
        db.execute(
            "UPDATE analysis_runs SET deletion_manifest=%s WHERE run_id=%s", (Json(manifest), str(run["run_id"])), conn=conn
        )
        conn.commit()

    if remaining:
        delete_jobs(remaining, run_credentials(run), on_deleted=save_progress)
    manifest["remote_complete"] = True
    save_progress(len(remaining))


def deletion_failed(table, id_column, identifier, conn, remote=False):
    """Keep a retryable tombstone with a fixed message, never raw service errors."""
    message = (
        "Error: Remote cleanup failed. Check the original Copernicus accounts and retry deletion; local files are retained."
        if remote
        else "Error: Cleanup could not finish. Check storage access and retry deletion."
    )
    conn.rollback()
    # table and id_column are literals passed by this module's callers, never request input.
    db.execute(f"UPDATE {table} SET error=%s WHERE {id_column}=%s", (message, str(identifier)), conn=conn)  # nosec B608
    conn.commit()
    raise HTTPException(503, message) from None


@router.delete("/runs/{run_id}")
def delete_run(run_id: UUID, user: dict = Depends(current_user)):
    """Delete an idle run's tracked remote jobs, private outputs and unshared input."""
    user_id = user["app_user_id"]
    with deletion_connection(run_id, user_id) as conn:
        run, manifest, directories, targets = prepare_run(run_id, user_id, conn)
        try:
            remote_cleanup(run, manifest, conn)
            for directory in directories:
                remove_directory(directory)
            purge_run(run, user_id, targets, conn)
        except Exception:  # noqa: BLE001 - keep a retryable tombstone, never raw errors
            deletion_failed("analysis_runs", "run_id", run_id, conn, remote=not manifest["remote_complete"])
    return {"deleted": True}


@router.delete("/eda/{eda_id}")
def delete_eda(eda_id: UUID, user: dict = Depends(current_user)):
    """Remove only one idle EDA analysis, preserving its source and siblings."""
    user_id = user["app_user_id"]
    row = db.one("SELECT run_id FROM eda_runs WHERE eda_id=%s AND app_user_id=%s", (str(eda_id), user_id))
    if not row:
        raise HTTPException(404, "Error: EDA analysis not found.")
    with deletion_connection(row["run_id"], user_id) as conn:
        run = owned_run(row["run_id"], user_id, conn=conn, lock=True)
        if run["status"] == "deleting":
            raise HTTPException(409, "Error: The parent run is being deleted. Retry its deletion instead.")
        eda = db.one("SELECT * FROM eda_runs WHERE eda_id=%s AND app_user_id=%s FOR UPDATE", (str(eda_id), user_id), conn=conn)
        if not eda:
            raise HTTPException(404, "Error: EDA analysis not found.")
        ensure_idle([eda], [str(eda_id)], user_id, conn)
        if db.one("SELECT 1 FROM ml_runs WHERE eda_id=%s LIMIT 1", (str(eda_id),), conn=conn):
            raise HTTPException(409, "Error: ML runs use this EDA analysis. Delete those ML runs first.")
        output_dir, home = eda_locations(user_id, eda)
        output = owned_directory(user_id, "eda", eda_id, eda["output_dir"], output_dir, home)
        home = owned_directory(user_id, "eda", eda_id, home, home)
        db.execute("UPDATE eda_runs SET status='deleting',error=NULL WHERE eda_id=%s", (str(eda_id),), conn=conn)
        conn.commit()
        try:
            remove_directory(output)
            remove_directory(home)
            db.execute(
                "DELETE FROM web_jobs WHERE target_id=%s AND kind='eda' AND app_user_id=%s", (str(eda_id), user_id), conn=conn
            )
            db.execute("DELETE FROM eda_runs WHERE eda_id=%s", (str(eda_id),), conn=conn)
            conn.commit()
        except Exception:  # noqa: BLE001 - keep a retryable tombstone, never raw errors
            deletion_failed("eda_runs", "eda_id", eda_id, conn)
    return {"deleted": True}


@router.delete("/ml/{ml_id}")
def delete_ml(ml_id: UUID, user: dict = Depends(current_user)):
    """Remove only one idle ML run, preserving its parcel run, EDA analysis and siblings."""
    user_id = user["app_user_id"]
    row = db.one("SELECT run_id FROM ml_runs WHERE ml_id=%s AND app_user_id=%s", (str(ml_id), user_id))
    if not row:
        raise HTTPException(404, "Error: ML run not found.")
    with deletion_connection(row["run_id"], user_id) as conn:
        run = owned_run(row["run_id"], user_id, conn=conn, lock=True)
        if run["status"] == "deleting":
            raise HTTPException(409, "Error: The parent run is being deleted. Retry its deletion instead.")
        ml = db.one("SELECT * FROM ml_runs WHERE ml_id=%s AND app_user_id=%s FOR UPDATE", (str(ml_id), user_id), conn=conn)
        if not ml:
            raise HTTPException(404, "Error: ML run not found.")
        ensure_idle([ml], [str(ml_id)], user_id, conn)
        output = owned_directory(user_id, "ml", ml_id, ml["output_dir"], ml_location(user_id, ml))
        db.execute("UPDATE ml_runs SET status='deleting',error=NULL WHERE ml_id=%s", (str(ml_id),), conn=conn)
        conn.commit()
        try:
            remove_directory(output)
            db.execute(
                "DELETE FROM web_jobs WHERE target_id=%s AND kind='ml' AND app_user_id=%s", (str(ml_id), user_id), conn=conn
            )
            db.execute("DELETE FROM ml_runs WHERE ml_id=%s", (str(ml_id),), conn=conn)
            conn.commit()
        except Exception:  # noqa: BLE001 - keep a retryable tombstone, never raw errors
            deletion_failed("ml_runs", "ml_id", ml_id, conn)
    return {"deleted": True}
