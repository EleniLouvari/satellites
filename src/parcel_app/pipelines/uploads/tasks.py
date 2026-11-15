"""Background inspection of uploaded parcel files."""

from psycopg2.extras import Json

from parcel_app.core import db
from parcel_app.core.storage import contained
from parcel_app.jobs.queue import owned_claim
from parcel_app.settings import get_settings


def inspect(job):
    """Inspect local vector bytes in the isolated worker child."""
    from .inspection import inspect_upload

    upload = db.one("SELECT * FROM parcel_uploads WHERE upload_id=%s AND app_user_id=%s", (job["target_id"], job["app_user_id"]))
    if upload is None:
        raise ValueError("Error: Upload is unavailable.")
    path = contained(get_settings().storage_root, upload["storage_path"])
    metadata = inspect_upload(path)
    with db.connection() as conn:
        owned_claim(job, conn)
        db.execute(
            "UPDATE parcel_uploads SET status='ready',metadata=%s,error=NULL WHERE upload_id=%s",
            (Json(metadata), job["target_id"]),
            conn=conn,
        )


def mark_upload_failed(job, message, conn):
    """Record a failed inspection on the upload."""
    db.execute("UPDATE parcel_uploads SET status='failed',error=%s WHERE upload_id=%s", (message, job["target_id"]), conn=conn)
