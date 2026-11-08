"""Owner-scoped, bounded raw uploads followed by asynchronous parcel inspection."""

import hashlib
import shutil
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from psycopg2.extras import Json

from . import db
from .security import current_user
from .settings import get_settings

router = APIRouter()


def public_upload(row: dict) -> dict:
    """Hide filesystem locations and ownership implementation details."""
    return {k: v for k, v in row.items() if k not in {"storage_path", "app_user_id"}}


def owned_upload(upload_id, user_id, *, conn=None) -> dict:
    """Resolve only an authenticated user's upload."""
    row = db.one("SELECT * FROM parcel_uploads WHERE upload_id=%s AND app_user_id=%s", (str(upload_id), user_id), conn=conn)
    if row is None:
        raise HTTPException(404, "Error: Upload not found.")
    return row


def enqueue(kind, target_id, user_id, payload=None, *, conn=None):
    """Create a durable work item transactionally with its parent resource."""
    return db.one(
        "INSERT INTO web_jobs(job_id,app_user_id,kind,target_id,payload) VALUES(%s,%s,%s,%s,%s) "
        "ON CONFLICT (kind,target_id) WHERE status IN ('pending','running') DO UPDATE SET target_id=EXCLUDED.target_id RETURNING *",
        (str(uuid4()), user_id, kind, str(target_id), Json(payload or {})), conn=conn)


def storage_available(reserve: int = 0) -> bool:
    """Check configured used-storage and physical free-space ceilings."""
    root = get_settings().storage_root
    root.mkdir(parents=True, exist_ok=True)
    used = sum(p.stat().st_size for p in root.rglob("*") if p.is_file() and not p.is_symlink())
    return used + reserve <= get_settings().storage_limit_bytes and shutil.disk_usage(root).free > reserve + 64 * 1024**2


@router.post("/uploads", status_code=202)
async def upload_file(request: Request, filename: str, user: dict = Depends(current_user)):
    """Stream into an opaque file; never execute filenames or trust declared size."""
    if not filename or len(filename) > 200 or any(c in filename for c in "/\\:\x00") or filename in {".", ".."}:
        raise HTTPException(422, "Error: Supply a simple filename without a directory.")
    suffix = Path(filename).suffix.lower()
    if suffix not in {".gpkg", ".parquet", ".geoparquet", ".geojson"}:
        raise HTTPException(422, "Error: Upload GeoPackage, GeoParquet or GeoJSON.")
    limit = get_settings().upload_limit_bytes
    length = request.headers.get("content-length")
    if length and (not length.isdigit() or int(length) > limit):
        raise HTTPException(413, "Error: Upload exceeds the size limit.")
    reserve = int(length) if length else limit
    upload_id = str(uuid4())
    destination = get_settings().storage_root / "users" / str(user["app_user_id"]) / "uploads" / upload_id / f"source{suffix}"
    with db.connection() as conn:
        db.one("SELECT pg_advisory_xact_lock(84123301)", conn=conn)
        outstanding = db.one("SELECT COALESCE(sum(size_bytes),0) AS n FROM parcel_uploads WHERE status='uploading'", conn=conn)["n"]
        if not storage_available(reserve + outstanding):
            raise HTTPException(507, "Error: Server storage limit reached. Contact the operator.")
        db.execute("INSERT INTO parcel_uploads(upload_id,app_user_id,filename,storage_path,sha256,size_bytes,status) "
                   "VALUES(%s,%s,%s,%s,'',%s,'uploading')", (upload_id, user["app_user_id"], filename, str(destination), reserve), conn=conn)
    total, digest = 0, hashlib.sha256()
    try:
        destination.parent.mkdir(parents=True)
        with destination.open("xb") as stream:
            async for chunk in request.stream():
                total += len(chunk)
                if total > limit:
                    raise HTTPException(413, "Error: Upload exceeds the size limit.")
                digest.update(chunk)
                stream.write(chunk)
        if total == 0:
            raise HTTPException(422, "Error: Uploaded file is empty.")
        with db.connection() as conn:
            db.execute("UPDATE parcel_uploads SET sha256=%s,size_bytes=%s,status='pending' WHERE upload_id=%s",
                       (digest.hexdigest(), total, upload_id), conn=conn)
            enqueue("inspect", upload_id, user["app_user_id"], conn=conn)
            return public_upload(owned_upload(upload_id, user["app_user_id"], conn=conn))
    except BaseException:
        # Only this request's newly created temporary upload is removed.
        destination.unlink(missing_ok=True)
        db.execute("DELETE FROM parcel_uploads WHERE upload_id=%s AND status='uploading'", (upload_id,))
        raise


@router.get("/uploads")
def list_uploads(user: dict = Depends(current_user)):
    """List the most recent one hundred owned uploads."""
    return [public_upload(row) for row in db.all_rows("SELECT * FROM parcel_uploads WHERE app_user_id=%s ORDER BY created_at DESC LIMIT 100", (user["app_user_id"],))]


@router.get("/uploads/{upload_id}")
def get_upload(upload_id: UUID, user: dict = Depends(current_user)):
    """Return asynchronous inspection metadata for an owned file."""
    return public_upload(owned_upload(upload_id, user["app_user_id"]))
