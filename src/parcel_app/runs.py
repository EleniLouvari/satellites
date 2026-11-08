"""Private drafts, validated submission and stable account snapshots for analyses."""

import json
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException
from psycopg2.extras import Json
from pydantic import BaseModel, ConfigDict, Field

from . import db
from .config import check_processing
from .security import current_user
from .settings import get_settings
from .uploads import enqueue, owned_upload, storage_available

router = APIRouter()


class RunDraft(BaseModel):
    """Allow only editable scientific and owned resource selections."""

    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=120)
    upload_id: UUID
    layer: str | None = Field(default=None, max_length=200)
    parcel_id_field: str = Field(min_length=1, max_length=200)
    partition_field: str | None = Field(default=None, max_length=200)
    processing: dict
    account_ids: list[int] = Field(min_length=1, max_length=50)


class Submission(BaseModel):
    """Bind submission to the exact validated draft revision."""

    model_config = ConfigDict(extra="forbid")
    revision: int = Field(ge=1)


def portable_config(config: dict) -> dict:
    """Remove private storage paths from any configuration representation."""
    clean = dict(config or {})
    clean.pop("output_dir", None)
    if "input" in clean:
        clean["input"] = {k: v for k, v in clean["input"].items() if k != "path"}
    return clean


def public_run(row: dict) -> dict:
    """Expose scientific settings and progress without filesystem locations."""
    result = {k: v for k, v in row.items() if k not in {"output_dir", "app_user_id"}}
    result["config"] = portable_config(result.get("config", {}))
    result["summary"] = dict(result.get("summary", {}))
    if "config" in result["summary"]:
        result["summary"]["config"] = portable_config(result["summary"]["config"])
    return result


def owned_run(run_id, user_id, *, conn=None, lock=False) -> dict:
    """Resolve one owned run, optionally locking its lifecycle transition."""
    query = "SELECT * FROM analysis_runs WHERE run_id=%s AND app_user_id=%s" + (" FOR UPDATE" if lock else "")
    row = db.one(query, (str(run_id), user_id), conn=conn)
    if row is None:
        raise HTTPException(404, "Error: Analysis not found.")
    return row


def selected_accounts(ids, user_id, *, conn=None):
    """Validate ownership and preserve the user's explicit selection order."""
    if len(ids) != len(set(ids)) or any(type(i) is not int or i < 1 for i in ids):
        raise HTTPException(422, "Error: Select unique Copernicus accounts.")
    rows = db.all_rows("SELECT eo_user_id,username FROM eo_users WHERE app_user_id=%s AND eo_user_id=ANY(%s) ORDER BY eo_user_id FOR SHARE",
                       (user_id, ids), conn=conn)
    mapping = {row["eo_user_id"]: row for row in rows}
    if len(mapping) != len(ids):
        raise HTTPException(422, "Error: One or more accounts are unavailable.")
    return [mapping[i] for i in ids]


def validate_draft(draft: RunDraft, user_id, *, conn=None):
    """Check file metadata and resource selection before scheduling preflight."""
    upload = owned_upload(draft.upload_id, user_id, conn=conn)
    if upload["status"] != "ready":
        raise HTTPException(409, "Error: Wait for successful parcel inspection.")
    layers = upload["metadata"].get("layers", [])
    selected = next((item for item in layers if item["name"] == draft.layer), None)
    if selected is None:
        raise HTTPException(422, "Error: Choose a layer from the uploaded dataset.")
    if draft.parcel_id_field not in selected["columns"] or (draft.partition_field and draft.partition_field not in selected["columns"]):
        raise HTTPException(422, "Error: Choose existing parcel and partition fields.")
    selected_accounts(draft.account_ids, user_id, conn=conn)
    try:
        check_processing(draft.processing)
    except (ValueError, TypeError):
        raise HTTPException(422, "Error: Invalid processing settings or server worker limit exceeded.") from None
    if not draft.name.strip():
        raise HTTPException(422, "Error: Give the analysis a name.")


def insert_draft(draft, user_id, *, conn):
    """Allocate an immutable server-owned output location for a fresh draft."""
    validate_draft(draft, user_id, conn=conn)
    run_id = str(uuid4())
    output = get_settings().storage_root / "users" / str(user_id) / "runs" / run_id
    return db.one(
        "INSERT INTO analysis_runs(run_id,app_user_id,name,upload_id,layer,parcel_id_field,partition_field,processing,account_ids,output_dir) "
        "VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING *",
        (run_id, user_id, draft.name.strip(), str(draft.upload_id), draft.layer, draft.parcel_id_field,
         draft.partition_field, Json(draft.processing), Json(draft.account_ids), str(output)), conn=conn)


@router.post("/runs", status_code=201)
def create_run(draft: RunDraft, user: dict = Depends(current_user)):
    """Persist an owned editable draft."""
    with db.connection() as conn:
        return public_run(insert_draft(draft, user["app_user_id"], conn=conn))


@router.patch("/runs/{run_id}")
def edit_run(run_id: UUID, draft: RunDraft, user: dict = Depends(current_user)):
    """Invalidate preflight whenever editable draft inputs change."""
    with db.connection() as conn:
        run = owned_run(run_id, user["app_user_id"], conn=conn, lock=True)
        submitted = db.one("SELECT 1 FROM run_accounts WHERE run_id=%s LIMIT 1", (str(run_id),), conn=conn)
        if run["status"] not in {"draft", "ready", "failed"} or submitted:
            raise HTTPException(409, "Error: Submitted or validating analyses cannot be edited; duplicate instead.")
        validate_draft(draft, user["app_user_id"], conn=conn)
        row = db.one("UPDATE analysis_runs SET name=%s,upload_id=%s,layer=%s,parcel_id_field=%s,partition_field=%s,"
                     "processing=%s,account_ids=%s,revision=revision+1,validated_revision=NULL,status='draft',"
                     "summary='{}',config='{}',error=NULL,updated_at=now() WHERE run_id=%s RETURNING *",
                     (draft.name.strip(), str(draft.upload_id), draft.layer, draft.parcel_id_field, draft.partition_field,
                      Json(draft.processing), Json(draft.account_ids), str(run_id)), conn=conn)
        return public_run(row)


@router.get("/runs")
def list_runs(user: dict = Depends(current_user)):
    """Return the most recent one hundred analyses belonging to this user."""
    return [public_run(row) for row in db.all_rows("SELECT * FROM analysis_runs WHERE app_user_id=%s ORDER BY created_at DESC LIMIT 100", (user["app_user_id"],))]


@router.get("/runs/{run_id}")
def get_run(run_id: UUID, user: dict = Depends(current_user)):
    """Fetch status and normalized configuration without leaking local paths."""
    return public_run(owned_run(run_id, user["app_user_id"]))


@router.post("/runs/{run_id}/validate", status_code=202)
def validate_run(run_id: UUID, user: dict = Depends(current_user)):
    """Queue offline preflight for a precise draft revision."""
    with db.connection() as conn:
        run = owned_run(run_id, user["app_user_id"], conn=conn, lock=True)
        if run["status"] == "validating":
            return public_run(run)
        if run["status"] not in {"draft", "ready", "failed"} or db.one("SELECT 1 FROM run_accounts WHERE run_id=%s LIMIT 1", (str(run_id),), conn=conn):
            raise HTTPException(409, "Error: Only unsubmitted drafts can be validated.")
        enqueue("validate", run_id, user["app_user_id"], {"revision": run["revision"]}, conn=conn)
        row = db.one("UPDATE analysis_runs SET status='validating',validated_revision=NULL,error=NULL,updated_at=now() WHERE run_id=%s RETURNING *", (str(run_id),), conn=conn)
        return public_run(row)


def reserve_accounts(run, accounts, *, conn):
    """Serialize account reservation and block identities with unresolved remote work."""
    db.one("SELECT pg_advisory_xact_lock(84123302)", conn=conn)
    names = [a["username"].strip().casefold() for a in accounts]
    conflict = db.one("SELECT 1 FROM run_accounts a JOIN analysis_runs r ON r.run_id=a.run_id "
                      "WHERE lower(a.username)=ANY(%s) AND r.run_id<>%s "
                      "AND r.status IN ('queued','running','failed','interrupted') LIMIT 1", (names, str(run["run_id"])), conn=conn)
    if conflict:
        raise HTTPException(409, "Error: An account is reserved by an active or interrupted analysis. Complete or resume it first.")


@router.post("/runs/{run_id}/submit", status_code=202)
def submit_run(run_id: UUID, body: Submission, user: dict = Depends(current_user)):
    """Submit exactly once and freeze account identities/order in one transaction."""
    with db.connection() as conn:
        run = owned_run(run_id, user["app_user_id"], conn=conn, lock=True)
        if body.revision != run["revision"]:
            raise HTTPException(409, "Error: The draft changed. Review its current validation.")
        if run["status"] in {"queued", "running", "complete"}:
            return public_run(run)
        if run["status"] != "ready" or run["validated_revision"] != run["revision"]:
            raise HTTPException(409, "Error: Validate this draft before submitting it.")
        if not storage_available(64 * 1024**2):
            raise HTTPException(507, "Error: Server storage limit reached.")
        accounts = selected_accounts(run["account_ids"], user["app_user_id"], conn=conn)
        reserve_accounts(run, accounts, conn=conn)
        for position, account in enumerate(accounts):
            db.execute("INSERT INTO run_accounts(run_id,eo_user_id,position,username) VALUES(%s,%s,%s,%s)",
                       (str(run_id), account["eo_user_id"], position, account["username"]), conn=conn)
        enqueue("run", run_id, user["app_user_id"], conn=conn)
        return public_run(db.one("UPDATE analysis_runs SET status='queued',updated_at=now() WHERE run_id=%s RETURNING *", (str(run_id),), conn=conn))


@router.post("/runs/{run_id}/resume", status_code=202)
def resume_run(run_id: UUID, user: dict = Depends(current_user)):
    """Resume the original file/configuration/account snapshot without editing it."""
    with db.connection() as conn:
        run = owned_run(run_id, user["app_user_id"], conn=conn, lock=True)
        if run["status"] in {"queued", "running"}:
            return public_run(run)
        accounts = db.all_rows("SELECT eo_user_id,username FROM run_accounts WHERE run_id=%s ORDER BY position", (str(run_id),), conn=conn)
        if run["status"] not in {"failed", "interrupted"} or not accounts:
            raise HTTPException(409, "Error: This analysis has no submitted work to resume.")
        selected_accounts([a["eo_user_id"] for a in accounts], user["app_user_id"], conn=conn)
        reserve_accounts(run, accounts, conn=conn)
        enqueue("resume", run_id, user["app_user_id"], conn=conn)
        return public_run(db.one("UPDATE analysis_runs SET status='queued',error=NULL,updated_at=now() WHERE run_id=%s RETURNING *", (str(run_id),), conn=conn))


@router.post("/runs/{run_id}/cancel")
def cancel_queued(run_id: UUID, user: dict = Depends(current_user)):
    """Cancel only a never-started queued run, not possibly active remote jobs."""
    with db.connection() as conn:
        run = owned_run(run_id, user["app_user_id"], conn=conn, lock=True)
        job = db.one("SELECT * FROM web_jobs WHERE target_id=%s AND kind IN ('run','resume') "
                     "AND status IN ('pending','running') FOR UPDATE", (str(run_id),), conn=conn)
        if run["status"] != "queued" or not job or job["kind"] != "run" or job["status"] != "pending":
            raise HTTPException(409, "Error: Only an initial queued analysis can be cancelled.")
        db.execute("UPDATE web_jobs SET status='failed',error='Cancelled before execution',updated_at=now() WHERE job_id=%s", (str(job["job_id"]),), conn=conn)
        return public_run(db.one("UPDATE analysis_runs SET status='cancelled',updated_at=now() WHERE run_id=%s RETURNING *", (str(run_id),), conn=conn))


@router.post("/runs/{run_id}/duplicate", status_code=201)
def duplicate_run(run_id: UUID, user: dict = Depends(current_user)):
    """Copy editable settings to a fresh draft with its own output directory."""
    with db.connection() as conn:
        run = owned_run(run_id, user["app_user_id"], conn=conn)
        draft = RunDraft(**{k: run[k] for k in RunDraft.model_fields})
        draft.name = (draft.name[:110] + " (copy)")
        return public_run(insert_draft(draft, user["app_user_id"], conn=conn))


@router.get("/runs/{run_id}/events")
def get_events(run_id: UUID, after: int = 0, user: dict = Depends(current_user)):
    """Return bounded safe event pages scoped to the authenticated owner."""
    owned_run(run_id, user["app_user_id"])
    return db.all_rows("SELECT event_id,stage,level,message,created_at FROM run_events "
                       "WHERE run_id=%s AND event_id>%s ORDER BY event_id LIMIT 200", (str(run_id), max(after, 0)))
