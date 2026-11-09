"""Authorize and stream only registered, contained analysis deliverables."""

from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from . import db
from .runs import owned_run
from .security import current_user
from .uploads import enqueue

router = APIRouter()


def contained_artifact(run, artifact) -> Path:
    """Resolve artifact paths and reject escaped files or symbolic-link traversal."""
    root = Path(run["output_dir"]).resolve()
    path = Path(artifact["storage_path"]).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Error: Artifact is unavailable.")
    return path


@router.get("/runs/{run_id}/artifacts")
def artifacts(run_id: UUID, user: dict = Depends(current_user)):
    """List registered outputs without exposing filesystem locations."""
    owned_run(run_id, user["app_user_id"])
    return db.all_rows(
        "SELECT artifact_id,kind,name,size_bytes,sha256,created_at FROM run_artifacts WHERE run_id=%s ORDER BY kind,name",
        (str(run_id),),
    )


@router.get("/artifacts/{artifact_id}/download")
def download(artifact_id: UUID, user: dict = Depends(current_user)):
    """Authorize by the artifact's owning run before accessing the filesystem."""
    row = db.one(
        "SELECT a.*,r.output_dir,r.status FROM run_artifacts a JOIN analysis_runs r ON r.run_id=a.run_id "
        "WHERE a.artifact_id=%s AND r.app_user_id=%s",
        (str(artifact_id), user["app_user_id"]),
    )
    if row is None:
        raise HTTPException(404, "Error: File not found.")
    if row["kind"] in {"time_statistics", "cleaning_report", "bundle"} and row["status"] != "complete":
        raise HTTPException(409, "Error: Final results are not complete.")
    try:
        path = contained_artifact(row, row)
    except ValueError:
        raise HTTPException(404, "Error: File is unavailable.") from None
    return FileResponse(
        path,
        filename=Path(row["name"]).name,
        media_type="application/octet-stream",
        headers={"X-Content-Type-Options": "nosniff", "Cache-Control": "private, no-store"},
    )


@router.post("/runs/{run_id}/bundle", status_code=202)
def bundle(run_id: UUID, user: dict = Depends(current_user)):
    """Queue ZIP generation from verified deliverables, never the whole run tree."""
    with db.connection() as conn:
        run = owned_run(run_id, user["app_user_id"], conn=conn, lock=True)
        if run["status"] != "complete":
            raise HTTPException(409, "Error: Complete the analysis before preparing a result bundle.")
        ready = db.one("SELECT artifact_id FROM run_artifacts WHERE run_id=%s AND kind='bundle'", (str(run_id),), conn=conn)
        if ready:
            return {"status": "ready", "artifact_id": ready["artifact_id"]}
        job = enqueue("bundle", run_id, user["app_user_id"], conn=conn)
        return {"job_id": job["job_id"], "status": job["status"]}


@router.get("/runs/{run_id}/bundle")
def bundle_status(run_id: UUID, user: dict = Depends(current_user)):
    """Expose owned archive readiness and failures so the browser can finish a download."""
    run = owned_run(run_id, user["app_user_id"])
    if run["status"] != "complete":
        return {"status": "unavailable"}
    artifact = db.one("SELECT artifact_id FROM run_artifacts WHERE run_id=%s AND kind='bundle'", (str(run_id),))
    if artifact:
        return {"status": "ready", "artifact_id": artifact["artifact_id"]}
    job = db.one(
        "SELECT status,error FROM web_jobs WHERE target_id=%s AND kind='bundle' ORDER BY created_at DESC LIMIT 1", (str(run_id),)
    )
    return job or {"status": "unrequested"}
