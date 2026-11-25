"""Private drafts, validated submission and stable account snapshots for analyses."""

from datetime import UTC, datetime
from typing import Literal
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException
from psycopg2.extras import Json
from pydantic import BaseModel, ConfigDict, Field

from parcel_app.core import db
from parcel_app.core.security import current_user
from parcel_app.core.storage import run_directory, storage_available
from parcel_app.jobs.queue import enqueue
from parcel_app.pipelines.uploads.routes import owned_upload

from .config import check_processing

router = APIRouter()

# Existing runs already record their true first start, so duration needs no
# fabricated creation-time estimate or migration of historic analyses.
RUN_TIMING = """(SELECT min(e.created_at) FROM run_events e WHERE e.run_id=analysis_runs.run_id
    AND e.stage='processing' AND e.message='Analysis started.') AS started_at,
    EXISTS(SELECT 1 FROM web_jobs j WHERE j.target_id=analysis_runs.run_id::text
    AND j.kind IN ('run','resume') AND j.status='running'
    AND j.payload->>'stop_requested'='true') AS stop_requested"""


class RunDraft(BaseModel):
    """Allow only editable scientific and owned resource selections."""

    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=120)
    source: Literal["openeo", "hub"] = "openeo"
    upload_id: UUID
    layer: str | None = Field(default=None, max_length=200)
    parcel_id_field: str = Field(min_length=1, max_length=200)
    partition_field: str | None = Field(default=None, max_length=200)
    processing: dict
    account_ids: list[int] = Field(default_factory=list, max_length=50)
    scheduling: Literal["batches", "partitions"] = "batches"


class Submission(BaseModel):
    """Bind submission to the exact validated draft revision."""

    model_config = ConfigDict(extra="forbid")
    revision: int = Field(ge=1)


class ResumeOptions(BaseModel):
    """Permit bounded resource recovery without editing the submitted science configuration."""

    model_config = ConfigDict(extra="forbid", strict=True)
    retry_exhausted_memory: bool = False


class RunName(BaseModel):
    """Accept only the display name for an existing analysis."""

    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=120)


def portable_config(config: dict) -> dict:
    """Remove private storage paths from any configuration representation."""
    clean = dict(config or {})
    clean.pop("output_dir", None)
    if "input" in clean:
        clean["input"] = {k: v for k, v in clean["input"].items() if k != "path"}
    return clean


def public_run(row: dict, *, scan: bool = True) -> dict:
    """Expose scientific settings and progress without filesystem locations.

    ``scan=False`` skips reading saved checkpoints for stopped runs, which list views poll often.
    """
    result = {k: v for k, v in row.items() if k not in {"output_dir", "app_user_id", "deletion_manifest"}}
    result["source"] = row.get("source", "openeo")
    result["config"] = portable_config(result.get("config", {}))
    result["summary"] = dict(result.get("summary", {}))
    if "config" in result["summary"]:
        result["summary"]["config"] = portable_config(result["summary"]["config"])
    started = result.get("started_at")
    terminal = result.get("status") in {"complete", "failed", "interrupted", "cancelled"}
    result["finished_at"] = result.get("updated_at") if started and terminal else None
    ended = result["finished_at"] or datetime.now(UTC)
    result["duration_seconds"] = max(0, (ended - started).total_seconds()) if started else None
    from .completion import run_progress

    result["progress"] = run_progress(row, scan=scan)
    return result


def owned_run(run_id, user_id, *, conn=None, lock=False) -> dict:
    """Resolve one owned run, optionally locking its lifecycle transition."""
    # RUN_TIMING and the lock clause are module constants; identifiers are bound parameters.
    query = f"SELECT *, {RUN_TIMING} FROM analysis_runs WHERE run_id=%s AND app_user_id=%s" + (" FOR UPDATE" if lock else "")  # nosec B608
    row = db.one(query, (str(run_id), user_id), conn=conn)
    if row is None:
        raise HTTPException(404, "Error: Analysis not found.")
    return row


def selected_accounts(ids, user_id, *, conn=None):
    """Validate ownership and preserve the user's explicit selection order."""
    if len(ids) != len(set(ids)) or any(type(i) is not int or i < 1 for i in ids):
        raise HTTPException(422, "Error: Select unique Copernicus accounts.")
    rows = db.all_rows(
        "SELECT eo_user_id,username FROM eo_users WHERE app_user_id=%s AND eo_user_id=ANY(%s) ORDER BY eo_user_id FOR SHARE",
        (user_id, ids),
        conn=conn,
    )
    mapping = {row["eo_user_id"]: row for row in rows}
    if len(mapping) != len(ids):
        raise HTTPException(422, "Error: One or more accounts are unavailable.")
    return [mapping[i] for i in ids]


def draft_accounts(draft, user_id, conn):
    """Separate operator HUB access from user-owned Copernicus accounts."""
    if draft.source == "hub":
        from parcel_app.sources.hub.credentials import configuration_error

        if error := configuration_error():
            raise HTTPException(422, error)
        if draft.account_ids:
            raise HTTPException(422, "Error: HUB requires configured server access and empty account selection.")
        draft.scheduling = "batches"
    else:
        if not draft.account_ids:
            raise HTTPException(422, "Error: Select at least one Copernicus account.")
        selected_accounts(draft.account_ids, user_id, conn=conn)


def submitted_work(run, conn):
    """Detect immutable submitted HUB work even though it has no account snapshots."""
    if run.get("source", "openeo") == "hub":
        return db.one(
            "SELECT 1 FROM web_jobs WHERE target_id=%s AND kind IN ('run','resume') LIMIT 1",
            (str(run["run_id"]),), conn=conn,
        )
    return db.one("SELECT 1 FROM run_accounts WHERE run_id=%s LIMIT 1", (str(run["run_id"]),), conn=conn)


def validate_draft(draft: RunDraft, user_id, *, conn=None):
    """Check file metadata and resource selection before scheduling preflight."""
    upload = owned_upload(draft.upload_id, user_id, conn=conn, lock=True)
    if upload["status"] != "ready":
        raise HTTPException(409, "Error: Wait for successful parcel inspection.")
    layers = upload["metadata"].get("layers", [])
    selected = next((item for item in layers if item["name"] == draft.layer), None)
    if selected is None:
        raise HTTPException(422, "Error: Choose a layer from the uploaded dataset.")
    if draft.parcel_id_field not in selected["columns"] or (
        draft.partition_field and draft.partition_field not in selected["columns"]
    ):
        raise HTTPException(422, "Error: Choose existing parcel and partition fields.")
    draft_accounts(draft, user_id, conn)
    try:
        draft.processing = check_processing(draft.processing, draft.source)
    except (ValueError, TypeError):
        raise HTTPException(422, "Error: Invalid processing settings or server worker limit exceeded.") from None
    if not draft.name.strip():
        raise HTTPException(422, "Error: Give the analysis a name.")


def insert_draft(draft, user_id, *, conn):
    """Allocate an immutable server-owned output location for a fresh draft."""
    validate_draft(draft, user_id, conn=conn)
    run_id = str(uuid4())
    output = run_directory(user_id, run_id) / "output"
    return db.one(
        "INSERT INTO analysis_runs(run_id,app_user_id,name,upload_id,layer,parcel_id_field,partition_field,processing,account_ids,output_dir,scheduling,source) "
        "VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING *",
        (
            run_id,
            user_id,
            draft.name.strip(),
            str(draft.upload_id),
            draft.layer,
            draft.parcel_id_field,
            draft.partition_field,
            Json(draft.processing),
            Json(draft.account_ids),
            str(output),
            draft.scheduling,
            draft.source,
        ),
        conn=conn,
    )


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
        submitted = submitted_work(run, conn)
        if run["status"] not in {"draft", "ready", "failed"} or submitted:
            raise HTTPException(409, "Error: Submitted or validating analyses cannot be edited; duplicate instead.")
        validate_draft(draft, user["app_user_id"], conn=conn)
        row = db.one(
            "UPDATE analysis_runs SET name=%s,upload_id=%s,layer=%s,parcel_id_field=%s,partition_field=%s,"
            "processing=%s,account_ids=%s,revision=revision+1,validated_revision=NULL,status='draft',"
            "summary='{}',config='{}',error=NULL,scheduling=%s,source=%s,updated_at=now() WHERE run_id=%s RETURNING *",
            (
                draft.name.strip(),
                str(draft.upload_id),
                draft.layer,
                draft.parcel_id_field,
                draft.partition_field,
                Json(draft.processing),
                Json(draft.account_ids),
                draft.scheduling,
                draft.source,
                str(run_id),
            ),
            conn=conn,
        )
        return public_run(row)


@router.patch("/runs/{run_id}/name")
def rename_run(run_id: UUID, body: RunName, user: dict = Depends(current_user)):
    """Rename an owned run without invalidating validation or changing its timing."""
    name = body.name.strip()
    if not name:
        raise HTTPException(422, "Error: Give the analysis a name.")
    with db.connection() as conn:
        run = owned_run(run_id, user["app_user_id"], conn=conn, lock=True)
        if run["status"] == "deleting":
            raise HTTPException(409, "Error: This analysis is being deleted.")
        # updated_at also records terminal timing; a label edit must not change it.
        db.execute("UPDATE analysis_runs SET name=%s WHERE run_id=%s", (name, str(run_id)), conn=conn)
        return public_run(dict(run, name=name))


@router.get("/runs")
def list_runs(user: dict = Depends(current_user)):
    """Return the most recent one hundred analyses belonging to this user.

    Each run carries ``eda_count`` and ``ml_count`` so the run list can show what lies beneath it
    before the user expands it.
    """
    # Only running analyses change between polls; stopped ones show counts on their own page.
    return [
        public_run(row, scan=row["status"] == "running")
        for row in db.all_rows(
            f"SELECT *, {RUN_TIMING},"  # nosec B608
            " (SELECT COUNT(*) FROM eda_runs e WHERE e.run_id=analysis_runs.run_id AND e.app_user_id=analysis_runs.app_user_id) AS eda_count,"
            " (SELECT COUNT(*) FROM ml_runs m WHERE m.run_id=analysis_runs.run_id AND m.app_user_id=analysis_runs.app_user_id) AS ml_count"
            " FROM analysis_runs WHERE app_user_id=%s ORDER BY created_at DESC LIMIT 100",
            (user["app_user_id"],),
        )
    ]


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
        if run["status"] not in {"draft", "ready", "failed"} or submitted_work(run, conn):
            raise HTTPException(409, "Error: Only unsubmitted drafts can be validated.")
        enqueue("validate", run_id, user["app_user_id"], {"revision": run["revision"]}, conn=conn)
        row = db.one(
            "UPDATE analysis_runs SET status='validating',validated_revision=NULL,error=NULL,updated_at=now() WHERE run_id=%s RETURNING *",
            (str(run_id),),
            conn=conn,
        )
        return public_run(row)


def reserve_accounts(run, accounts, *, conn):
    """Serialize account reservation and block identities with unresolved remote work."""
    db.one("SELECT pg_advisory_xact_lock(84123302)", conn=conn)
    names = [a["username"].strip().casefold() for a in accounts]
    conflict = db.one(
        "SELECT 1 FROM run_accounts a JOIN analysis_runs r ON r.run_id=a.run_id "
        "WHERE lower(btrim(a.username))=ANY(%s) AND r.run_id<>%s "
        "AND r.status IN ('queued','running','failed','interrupted','deleting') LIMIT 1",
        (names, str(run["run_id"])),
        conn=conn,
    )
    if conflict:
        raise HTTPException(
            409, "Error: An account is reserved by an active or interrupted analysis. Complete or resume it first."
        )


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
        accounts = []
        if run.get("source", "openeo") == "openeo":
            accounts = selected_accounts(run["account_ids"], user["app_user_id"], conn=conn)
            reserve_accounts(run, accounts, conn=conn)
        for position, account in enumerate(accounts):
            db.execute(
                "INSERT INTO run_accounts(run_id,eo_user_id,position,username) VALUES(%s,%s,%s,%s)",
                (str(run_id), account["eo_user_id"], position, account["username"]),
                conn=conn,
            )
        enqueue("run", run_id, user["app_user_id"], conn=conn)
        return public_run(
            db.one(
                "UPDATE analysis_runs SET status='queued',updated_at=now() WHERE run_id=%s RETURNING *", (str(run_id),), conn=conn
            )
        )


@router.post("/runs/{run_id}/resume", status_code=202)
def resume_run(run_id: UUID, user: dict = Depends(current_user), options: ResumeOptions | None = None):
    """Resume the original file/configuration/account snapshot without editing it."""
    with db.connection() as conn:
        run = owned_run(run_id, user["app_user_id"], conn=conn, lock=True)
        if run["status"] in {"queued", "running"}:
            return public_run(run)
        accounts = db.all_rows(
            "SELECT eo_user_id,username FROM run_accounts WHERE run_id=%s ORDER BY position", (str(run_id),), conn=conn
        )
        hub = run.get("source", "openeo") == "hub"
        if run["status"] not in {"failed", "interrupted"} or (not accounts and not hub) or (hub and not submitted_work(run, conn)):
            raise HTTPException(409, "Error: This analysis has no submitted work to resume.")
        if not hub:
            selected_accounts([a["eo_user_id"] for a in accounts], user["app_user_id"], conn=conn)
            reserve_accounts(run, accounts, conn=conn)
        payload = {"retry_exhausted_memory": True} if options and options.retry_exhausted_memory else {}
        enqueue("resume", run_id, user["app_user_id"], payload=payload, conn=conn)
        return public_run(
            db.one(
                "UPDATE analysis_runs SET status='queued',error=NULL,updated_at=now() WHERE run_id=%s RETURNING *",
                (str(run_id),),
                conn=conn,
            )
        )


@router.post("/runs/{run_id}/stop", status_code=202)
def stop_run(run_id: UUID, user: dict = Depends(current_user)):
    """Request local interruption without releasing the claim or deleting checkpoints."""
    with db.connection() as conn:
        owned_run(run_id, user["app_user_id"], conn=conn)
        # Match task publication's lock order: queue claim, then analysis row.
        job = db.one(
            "SELECT * FROM web_jobs WHERE target_id=%s AND app_user_id=%s "
            "AND kind IN ('run','resume') AND status='running' FOR UPDATE",
            (str(run_id), user["app_user_id"]), conn=conn,
        )
        run = owned_run(run_id, user["app_user_id"], conn=conn, lock=True)
        if run["status"] == "interrupted":
            return public_run(run)
        if run["status"] != "running" or not job:
            raise HTTPException(409, "Error: Only a running analysis can be stopped.")
        if not job["payload"].get("stop_requested"):
            db.execute(
                "UPDATE web_jobs SET payload=payload || '{\"stop_requested\":true}'::jsonb,updated_at=now() WHERE job_id=%s",
                (str(job["job_id"]),), conn=conn,
            )
            db.execute(
                "INSERT INTO run_events(run_id,stage,message) VALUES(%s,'processing',%s)",
                (str(run_id), "Stop requested. Waiting for local processing to exit; completed checkpoints will be kept."), conn=conn,
            )
        return public_run({**run, "stop_requested": True})


@router.post("/runs/{run_id}/cancel")
def cancel_queued(run_id: UUID, user: dict = Depends(current_user)):
    """Cancel only a never-started queued run, not possibly active remote jobs."""
    with db.connection() as conn:
        run = owned_run(run_id, user["app_user_id"], conn=conn, lock=True)
        job = db.one(
            "SELECT * FROM web_jobs WHERE target_id=%s AND kind IN ('run','resume') "
            "AND status IN ('pending','running') FOR UPDATE",
            (str(run_id),),
            conn=conn,
        )
        if run["status"] != "queued" or not job or job["kind"] != "run" or job["status"] != "pending":
            raise HTTPException(409, "Error: Only an initial queued analysis can be cancelled.")
        db.execute(
            "UPDATE web_jobs SET status='failed',error='Cancelled before execution',updated_at=now() WHERE job_id=%s",
            (str(job["job_id"]),),
            conn=conn,
        )
        return public_run(
            db.one(
                "UPDATE analysis_runs SET status='cancelled',updated_at=now() WHERE run_id=%s RETURNING *",
                (str(run_id),),
                conn=conn,
            )
        )


@router.post("/runs/{run_id}/duplicate", status_code=201)
def duplicate_run(run_id: UUID, user: dict = Depends(current_user)):
    """Copy editable settings to a fresh draft with its own output directory."""
    with db.connection() as conn:
        run = owned_run(run_id, user["app_user_id"], conn=conn)
        if run["status"] == "deleting":
            raise HTTPException(409, "Error: This analysis is being deleted.")
        draft = RunDraft(**{k: run[k] for k in RunDraft.model_fields if k in run})
        draft.name = draft.name[:110] + " (copy)"
        return public_run(insert_draft(draft, user["app_user_id"], conn=conn))


@router.get("/runs/{run_id}/events")
def get_events(run_id: UUID, after: int = 0, tail: bool = False, user: dict = Depends(current_user)):
    """Return bounded safe event pages scoped to the authenticated owner.

    ``tail`` returns the newest page in ascending order, so a viewer starts at the present.
    """
    owned_run(run_id, user["app_user_id"])
    if tail:
        return db.all_rows(
            "SELECT * FROM (SELECT event_id,stage,level,message,created_at FROM run_events "
            "WHERE run_id=%s ORDER BY event_id DESC LIMIT 200) newest ORDER BY event_id",
            (str(run_id),),
        )
    return db.all_rows(
        "SELECT event_id,stage,level,message,created_at FROM run_events "
        "WHERE run_id=%s AND event_id>%s ORDER BY event_id LIMIT 200",
        (str(run_id), max(after, 0)),
    )
