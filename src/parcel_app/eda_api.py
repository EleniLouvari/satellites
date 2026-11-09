"""Completed-run EDA configuration, submissions and private downloads."""

import base64
import json
from html import escape
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from psycopg2.extras import Json
from pydantic import BaseModel, ConfigDict, Field

from . import db
from .artifacts import contained, sanitize_text
from .eda_config import eda_schema, validate_eda
from .runs import owned_run
from .security import current_user
from .settings import get_settings
from .uploads import enqueue, storage_available

router = APIRouter()


class EDASubmission(BaseModel):
    """Bind a unique browser request to an immutable EDA column/config snapshot."""

    model_config = ConfigDict(extra="forbid")
    request_id: UUID
    columns: list[str] = Field(min_length=1, max_length=10000)
    config: dict


def completed_run(run_id, user_id, *, conn=None, lock=False):
    """Enforce completion and ownership before reading any source metadata."""
    run = owned_run(run_id, user_id, conn=conn, lock=lock)
    if run["status"] != "complete":
        raise HTTPException(409, "Error: Complete the parcel analysis before starting EDA.")
    return run


def source_artifact(run, *, conn=None):
    """Resolve only the registered final statistics file, never a client path."""
    artifact = db.one("SELECT * FROM run_artifacts WHERE run_id=%s AND kind='time_statistics'", (str(run["run_id"]),), conn=conn)
    if not artifact:
        raise HTTPException(409, "Error: Verified parcel results are unavailable.")
    path = contained(run["output_dir"], artifact["storage_path"])
    if not path.is_file():
        raise HTTPException(409, "Error: The parcel result file is unavailable.")
    return artifact, path


def source_columns(path):
    """Inspect Parquet schema without loading all parcel values into the API."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pq.read_schema(path)
    geo = json.loads((schema.metadata or {}).get(b"geo", b"{}"))
    geometries = set(geo.get("columns", {}))
    return [
        {
            "name": field.name,
            "dtype": str(field.type),
            "numeric": pa.types.is_integer(field.type) or pa.types.is_floating(field.type),
            "geometry": field.name in geometries,
        }
        for field in schema
        if not field.name.startswith("__index_level_")
    ]


def public_eda(row):
    """Expose saved configuration and timing without storage paths or owner IDs."""
    result = {key: value for key, value in row.items() if key not in {"output_dir", "app_user_id", "input_sha256"}}
    if row["status"] == "complete":
        result["progress"] = {"stage": "Results ready", "percent": 100, "completed": 5, "total": 5}
    elif row["status"] == "deleting":
        result["progress"] = {"stage": "Removing EDA files and history", "percent": None}
    return result


@router.get("/runs/{run_id}/eda/options")
def options(run_id: UUID, user: dict = Depends(current_user)):
    """Supply actual completed-result columns and every EDA configuration option."""
    run = completed_run(run_id, user["app_user_id"])
    _, path = source_artifact(run)
    columns = source_columns(path)
    schema = eda_schema(columns)
    schema["defaults"]["report_title"] = f"EDA: {run['name']}"
    identifier = run["parcel_id_field"] if run["parcel_id_field"] in {c["name"] for c in columns} else None
    schema["defaults"]["id_column"] = identifier
    return {
        "columns": columns,
        "schema": schema,
        "default_columns": [c["name"] for c in columns if c["numeric"] or c["geometry"] or c["name"] == identifier],
    }


@router.get("/runs/{run_id}/eda")
def list_eda(run_id: UUID, user: dict = Depends(current_user)):
    """Return this user's EDA history attached to the source run."""
    owned_run(run_id, user["app_user_id"])
    return [
        public_eda(row)
        for row in db.all_rows(
            "SELECT * FROM eda_runs WHERE run_id=%s AND app_user_id=%s ORDER BY created_at DESC",
            (str(run_id), user["app_user_id"]),
        )
    ]


@router.post("/runs/{run_id}/eda", status_code=202)
def submit_eda(run_id: UUID, body: EDASubmission, user: dict = Depends(current_user)):
    """Persist and enqueue one immutable EDA analysis, idempotent for request retries."""
    with db.connection() as conn:
        run = completed_run(run_id, user["app_user_id"], conn=conn, lock=True)
        existing = db.one("SELECT * FROM eda_runs WHERE eda_id=%s", (str(body.request_id),), conn=conn)
        if existing:
            if existing["app_user_id"] != user["app_user_id"] or str(existing["run_id"]) != str(run_id):
                raise HTTPException(409, "Error: Use a new EDA request identifier.")
            try:
                same_config = validate_eda(body.config, body.columns, existing["selected_columns"]) == existing["config"]
            except (ValueError, TypeError):
                same_config = False
            if body.columns != existing["selected_columns"] or not same_config:
                raise HTTPException(409, "Error: This request already has different settings; create a new EDA analysis.")
            return public_eda(existing)
        artifact, path = source_artifact(run, conn=conn)
        columns = source_columns(path)
        try:
            config = validate_eda(body.config, body.columns, [column["name"] for column in columns])
            for role in ("target_column", "id_column"):
                if any(c["name"] == config[role] and c["geometry"] for c in columns):
                    raise ValueError("Error: Geometry cannot be used as a target or ID.")
        except (ValueError, TypeError) as error:
            raise HTTPException(422, str(error) if str(error).startswith("Error:") else "Error: Invalid EDA settings.") from None
        if not storage_available(64 * 1024**2):
            raise HTTPException(507, "Error: Server storage limit reached.")
        output = get_settings().storage_root / "users" / str(user["app_user_id"]) / "eda" / str(body.request_id)
        row = db.one(
            "INSERT INTO eda_runs(eda_id,run_id,app_user_id,selected_columns,config,input_sha256,output_dir) "
            "VALUES(%s,%s,%s,%s,%s,%s,%s) RETURNING *",
            (
                str(body.request_id),
                str(run_id),
                user["app_user_id"],
                Json(body.columns),
                Json(config),
                artifact["sha256"],
                str(output),
            ),
            conn=conn,
        )
        enqueue("eda", body.request_id, user["app_user_id"], conn=conn)
        return public_eda(row)


def owned_eda(eda_id, user_id):
    """Hide foreign analyses and their artifacts behind the same not-found response."""
    row = db.one("SELECT * FROM eda_runs WHERE eda_id=%s AND app_user_id=%s", (str(eda_id), user_id))
    if not row:
        raise HTTPException(404, "Error: EDA analysis not found.")
    return row


@router.get("/eda/{eda_id}/report")
def open_report(eda_id: UUID, user: dict = Depends(current_user)):
    """Serve an owned report with embedded plots in an isolated browser origin."""
    row = owned_eda(eda_id, user["app_user_id"])
    if row["status"] != "complete":
        raise HTTPException(409, "Error: EDA results are not complete.")
    names = {item["name"] for item in row["artifacts"]}
    root = Path(row["output_dir"])
    report = contained(root, root / "report.html")
    if "report.html" not in names or not report.is_file():
        raise HTTPException(404, "Error: This analysis has no HTML report.")
    html = report.read_text(encoding="utf-8")
    # Embed only worker-registered PNGs. Sandboxed documents need no authenticated subrequests.
    for name in sorted(names):
        if not name.startswith("plots/") or not name.endswith(".png"):
            continue
        source = f"src='{escape(name)}'"
        if source in html:
            path = contained(root, root / name)
            if not path.is_file():
                raise HTTPException(404, "Error: A report plot is unavailable.")
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            html = html.replace(source, f"src='data:image/png;base64,{encoded}'")
    return HTMLResponse(html, headers={
        "Cache-Control": "private, no-store",
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "SAMEORIGIN",
        "Content-Security-Policy": (
            "sandbox allow-scripts; default-src 'none'; img-src data:; style-src 'unsafe-inline'; "
            "script-src 'unsafe-inline'; frame-ancestors 'self'; base-uri 'none'; form-action 'none'"
        ),
    })


@router.get("/eda/{eda_id}/log")
def download_log(eda_id: UUID, user: dict = Depends(current_user)):
    """Expose terminal-run diagnostics only to their owner, with sensitive text redacted."""
    row = owned_eda(eda_id, user["app_user_id"])
    if row["status"] not in {"complete", "failed", "interrupted"}:
        raise HTTPException(409, "Error: EDA is still processing.")
    root = Path(row["output_dir"])
    path = contained(root, root / "eda.log")
    if not path.is_file():
        raise HTTPException(404, "Error: No EDA log is available.")
    return PlainTextResponse(
        sanitize_text(path.read_text(encoding="utf-8", errors="replace"), [str(root)]),
        headers={"Content-Disposition": 'attachment; filename="eda.log"', "Cache-Control": "private, no-store",
                 "X-Content-Type-Options": "nosniff", "Content-Security-Policy": "sandbox"},
    )


@router.get("/eda/{eda_id}/files/{artifact_id}")
def download_eda(eda_id: UUID, artifact_id: UUID, user: dict = Depends(current_user)):
    """Download a published EDA attachment without accepting filesystem paths."""
    row = owned_eda(eda_id, user["app_user_id"])
    if row["status"] != "complete":
        raise HTTPException(409, "Error: EDA results are not complete.")
    artifact = next((item for item in row["artifacts"] if item["artifact_id"] == str(artifact_id)), None)
    if not artifact:
        raise HTTPException(404, "Error: EDA file not found.")
    path = contained(row["output_dir"], Path(row["output_dir"]) / artifact["name"])
    if not path.is_file():
        raise HTTPException(404, "Error: EDA file is unavailable.")
    return FileResponse(
        path,
        filename=path.name,
        media_type="application/octet-stream",
        headers={"X-Content-Type-Options": "nosniff", "Cache-Control": "private, no-store", "Content-Security-Policy": "sandbox"},
    )
