"""ML classification of a completed parcel result: options, submissions, reports and downloads."""

import base64
import contextlib
import csv
import json
import posixpath
import re
from pathlib import Path
from urllib.parse import unquote
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from psycopg2.extras import Json
from pydantic import BaseModel, ConfigDict, Field

from parcel_app.core import db
from parcel_app.core.security import current_user
from parcel_app.core.storage import contained, ml_directory, sanitize_text, storage_available
from parcel_app.jobs.queue import enqueue
from parcel_app.pipelines.eda.routes import completed_run, source_artifact, source_columns

from .config import OUTLIER_COLUMN, ml_schema, validate_ml

router = APIRouter()
ANNOTATED = "eda_annotated_data.parquet"
PROPOSAL = "tables/feature_selection/feature_selection_proposal.csv"
# The five pipeline steps in order: (name, method, progress text, report page file).
STEPS = (
    ("check", "run_check", "Checking data", "01_check/report.html"),
    ("prepare", "run_prepare", "Preparing splits", "02_prepare/report.html"),
    ("train", "run_train", "Training models", "03_train/report.html"),
    ("evaluate", "run_evaluate", "Evaluating models", "04_evaluate/report.html"),
    ("predict", "run_predict", "Predicting parcels", "05_predict/report.html"),
)
# Fixed report pages; the browser names a page, never a file path.
REPORTS = {name: report for name, _, _, report in STEPS} | {"dashboard": "final_dashboard/report.html"}
# A run in one of these states has steps left and no job working on it.
CONTINUABLE = {"paused", "failed", "interrupted"}


class MLSubmission(BaseModel):
    """Bind a unique browser request to an immutable feature list and configuration."""

    model_config = ConfigDict(extra="forbid")
    request_id: UUID
    name: str = Field(min_length=1, max_length=200)
    features: list[str] = Field(min_length=1, max_length=10000)
    config: dict
    through: int = Field(default=len(STEPS), ge=1, le=len(STEPS))


class MLContinuation(BaseModel):
    """Ask for the remaining steps up to and including step number ``through``."""

    model_config = ConfigDict(extra="forbid")
    through: int = Field(ge=1, le=len(STEPS))


def owned_eda_source(eda_id, user_id, *, conn=None, lock=False):
    """Resolve an owned, completed EDA analysis with its completed parcel run."""
    query = "SELECT * FROM eda_runs WHERE eda_id=%s AND app_user_id=%s" + (" FOR SHARE" if lock else "")  # nosec B608
    eda = db.one(query, (str(eda_id), user_id), conn=conn)
    if not eda:
        raise HTTPException(404, "Error: EDA analysis not found.")
    run = completed_run(eda["run_id"], user_id, conn=conn)
    if eda["status"] != "complete":
        raise HTTPException(409, "Error: Complete the EDA analysis before starting ML classification.")
    return eda, run


def annotated_source(eda):
    """Resolve the registered EDA table that carries the calculated columns, never a client path."""
    artifact = next((item for item in eda["artifacts"] if item["name"] == ANNOTATED), None)
    if eda["status"] != "complete" or artifact is None:
        raise HTTPException(409, "Error: The EDA analysis has no annotated result.")
    path = contained(eda["output_dir"], Path(eda["output_dir"]) / ANNOTATED)
    if not path.is_file():
        raise HTTPException(409, "Error: The EDA result file is unavailable.")
    return artifact, path


def proposed_features(eda):
    """Read the features the EDA analysis proposed to use; none when it made no proposal."""
    if not any(item["name"] == PROPOSAL for item in eda["artifacts"]):
        return []
    path = contained(eda["output_dir"], Path(eda["output_dir"]) / PROPOSAL)
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        return [row["feature"] for row in csv.DictReader(stream) if row.get("proposed_action") == "use" and row.get("feature")]


def feature_columns(eda, run, *, conn=None):
    """List selectable columns: those analysed by EDA first, then EDA-calculated, then the rest.

    The parcel result is the model input. Columns that exist only in the EDA table
    (such as ``is_outlier``) were calculated by EDA and are offered but never pre-selected.
    """
    artifact, path = source_artifact(run, conn=conn)
    if artifact["sha256"] != eda["input_sha256"]:
        raise HTTPException(409, "Error: The parcel result changed after this EDA analysis; run EDA again.")
    _, eda_path = annotated_source(eda)
    result = source_columns(path)
    names = {column["name"] for column in result}
    analysed = set(eda["selected_columns"])
    proposed = set(proposed_features(eda))
    calculated = [column for column in source_columns(eda_path) if column["name"] not in names]
    if OUTLIER_COLUMN not in {column["name"] for column in calculated}:
        raise HTTPException(409, "Error: The EDA analysis has no outlier flag; run EDA again.")
    ordered = (
        [column | {"source": "eda"} for column in result if column["name"] in analysed]
        + [column | {"source": "calculated"} for column in calculated]
        + [column | {"source": "result"} for column in result if column["name"] not in analysed]
    )
    return [column | {"proposed": column["name"] in proposed} for column in ordered]


def public_ml(row):
    """Expose saved configuration, step state and timing without storage paths, checksums or owner IDs."""
    hidden = {"output_dir", "app_user_id", "input_sha256", "eda_sha256"}
    result = {key: value for key, value in row.items() if key not in hidden}
    if row["status"] == "deleting":
        result["progress"] = {"stage": "Removing ML files and history", "percent": None}
    return result


@router.get("/eda/{eda_id}/ml/options")
def options(eda_id: UUID, user: dict = Depends(current_user)):
    """Supply the selectable columns, the EDA proposal and every grouped ML setting with its default."""
    eda, run = owned_eda_source(eda_id, user["app_user_id"])
    columns = feature_columns(eda, run)
    # The target and ID are read from the parcel result, so EDA-calculated columns cannot take those roles.
    schema = ml_schema([column for column in columns if column["source"] != "calculated"])
    names = {column["name"] for column in columns if not column["geometry"] and column["source"] != "calculated"}
    target = eda["config"].get("target_column")
    identifier = next((name for name in (eda["config"].get("id_column"), run["parcel_id_field"]) if name in names), None)
    schema["defaults"] |= {"target_column": target if target in names else None, "id_column": identifier}
    roles = {target, identifier}
    proposed = [c["name"] for c in columns if c["proposed"] and c["name"] not in roles]
    analysed = [c["name"] for c in columns if c["source"] == "eda" and c["numeric"] and c["name"] not in roles]
    return {
        "columns": columns,
        "schema": schema,
        # Start from the EDA proposal; an EDA without a target makes none, so fall back to its numeric columns.
        "default_features": proposed or analysed,
        "default_name": f"ML: {eda['config'].get('report_title') or run['name']}",
        "eda_title": eda["config"].get("report_title"),
        "run_name": run["name"],
    }


@router.get("/eda/{eda_id}/ml")
def list_ml(eda_id: UUID, user: dict = Depends(current_user)):
    """Return this user's ML runs of one EDA analysis, newest first."""
    row = db.one("SELECT 1 FROM eda_runs WHERE eda_id=%s AND app_user_id=%s", (str(eda_id), user["app_user_id"]))
    if not row:
        raise HTTPException(404, "Error: EDA analysis not found.")
    return [
        public_ml(row)
        for row in db.all_rows(
            "SELECT * FROM ml_runs WHERE eda_id=%s AND app_user_id=%s ORDER BY created_at DESC",
            (str(eda_id), user["app_user_id"]),
        )
    ]


@router.post("/eda/{eda_id}/ml", status_code=202)
def submit_ml(eda_id: UUID, body: MLSubmission, user: dict = Depends(current_user)):
    """Persist one immutable ML scenario and enqueue its first steps, idempotent for request retries."""
    user_id = user["app_user_id"]
    with db.connection() as conn:
        eda, run = owned_eda_source(eda_id, user_id, conn=conn, lock=True)
        columns = feature_columns(eda, run, conn=conn)
        available = [column["name"] for column in columns if not column["geometry"]]
        try:
            config = validate_ml(body.config, body.features, available, geometry=any(c["geometry"] for c in columns))
            calculated = {column["name"] for column in columns if column["source"] == "calculated"}
            if {config["target_column"], config["id_column"]} & calculated:
                raise ValueError("Error: Choose the target and ID from the parcel result columns.")
        except (ValueError, TypeError) as error:
            raise HTTPException(422, str(error) if str(error).startswith("Error:") else "Error: Invalid ML settings.") from None
        existing = db.one("SELECT * FROM ml_runs WHERE ml_id=%s", (str(body.request_id),), conn=conn)
        if existing:
            if existing["app_user_id"] != user_id or str(existing["eda_id"]) != str(eda_id):
                raise HTTPException(409, "Error: Use a new ML request identifier.")
            if (body.name, body.features, config) != (existing["name"], existing["feature_columns"], existing["config"]):
                raise HTTPException(409, "Error: This request already has different settings; create a new ML run.")
            return public_ml(existing)
        if not storage_available(1024**3):
            raise HTTPException(507, "Error: Server storage limit reached.")
        output = ml_directory(user_id, run["run_id"], eda_id, body.request_id)
        row = db.one(
            "INSERT INTO ml_runs(ml_id,run_id,eda_id,app_user_id,name,feature_columns,config,input_sha256,eda_sha256,output_dir) "
            "VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING *",
            (
                str(body.request_id),
                str(run["run_id"]),
                str(eda_id),
                user_id,
                body.name,
                Json(body.features),
                Json(config),
                eda["input_sha256"],
                annotated_source(eda)[0]["sha256"],
                str(output),
            ),
            conn=conn,
        )
        enqueue("ml", body.request_id, user_id, {"through": body.through}, conn=conn)
        return public_ml(row)


@router.post("/ml/{ml_id}/continue", status_code=202)
def continue_ml(ml_id: UUID, body: MLContinuation, user: dict = Depends(current_user)):
    """Queue the next steps of a paused, failed or interrupted run; finished steps are reused."""
    user_id = user["app_user_id"]
    with db.connection() as conn:
        row = db.one("SELECT * FROM ml_runs WHERE ml_id=%s AND app_user_id=%s FOR UPDATE", (str(ml_id), user_id), conn=conn)
        if not row:
            raise HTTPException(404, "Error: ML run not found.")
        if row["status"] not in CONTINUABLE or body.through <= row["steps_done"]:
            raise HTTPException(409, "Error: This ML run has no such step to run now.")
        completed_run(row["run_id"], user_id, conn=conn)
        if not storage_available(1024**3):
            raise HTTPException(507, "Error: Server storage limit reached.")
        row = db.one(
            "UPDATE ml_runs SET status='queued',error=NULL,finished_at=NULL WHERE ml_id=%s RETURNING *", (str(ml_id),), conn=conn
        )
        enqueue("ml", ml_id, user_id, {"through": body.through}, conn=conn)
        return public_ml(row)


def owned_ml(ml_id, user_id):
    """Hide foreign runs and their artifacts behind the same not-found response."""
    row = db.one("SELECT * FROM ml_runs WHERE ml_id=%s AND app_user_id=%s", (str(ml_id), user_id))
    if not row:
        raise HTTPException(404, "Error: ML run not found.")
    return row


@router.get("/ml/{ml_id}/report/{page}")
def open_report(ml_id: UUID, page: str, user: dict = Depends(current_user)):
    """Serve one finished step's report with its plots embedded, in an isolated browser origin."""
    row = owned_ml(ml_id, user["app_user_id"])
    names = {item["name"] for item in row["artifacts"]}
    root = Path(row["output_dir"])
    name = REPORTS.get(page)
    if name is None or name not in names:
        raise HTTPException(404, "Error: This run has no such report yet.")
    report = contained(root, root / name)
    if not report.is_file():
        raise HTTPException(404, "Error: The report is unavailable.")
    folder = name.rsplit("/", 1)[0]

    def embed(match):
        """Inline a worker-registered PNG; sandboxed documents cannot make authenticated subrequests."""
        relative = posixpath.normpath(f"{folder}/{unquote(match.group(1))}")
        if relative not in names or not relative.endswith(".png"):
            return match.group(0)
        encoded = base64.b64encode(contained(root, root / relative).read_bytes()).decode("ascii")
        return f"<img src='data:image/png;base64,{encoded}'"

    html = re.sub(r"<img src='([^']+)'", embed, report.read_text(encoding="utf-8")).replace(str(root), "")
    return HTMLResponse(html, headers={
        "Cache-Control": "private, no-store",
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "SAMEORIGIN",
        "Content-Security-Policy": (
            "sandbox allow-scripts; default-src 'none'; img-src data:; style-src 'unsafe-inline'; "
            "script-src 'unsafe-inline'; frame-ancestors 'self'; base-uri 'none'; form-action 'none'"
        ),
    })


@router.get("/ml/{ml_id}/log")
def download_log(ml_id: UUID, user: dict = Depends(current_user)):
    """Expose run diagnostics to their owner, also while training, with sensitive text redacted."""
    row = owned_ml(ml_id, user["app_user_id"])
    root = Path(row["output_dir"])
    path = contained(root, root / "ml.log")
    if row["status"] in {"queued", "deleting"} or not path.is_file():
        raise HTTPException(404, "Error: No ML log is available.")
    text = re.sub(r"\x1b\[[0-9;]*m", "", path.read_text(encoding="utf-8", errors="replace"))
    return PlainTextResponse(
        sanitize_text(text, private_paths=[str(root)]),
        headers={"Content-Disposition": 'attachment; filename="ml.log"', "Cache-Control": "private, no-store",
                 "X-Content-Type-Options": "nosniff", "Content-Security-Policy": "sandbox"},
    )


# Live view limits: the end of the log is enough to follow a run; the full log is downloadable.
LIVE_LOG_BYTES = 64 * 1024
LIVE_LOG_LINES = 200
TRAINING_PROGRESS = "03_train/training_progress.json"


def log_tail(path, root):
    """Return the last lines of a log without colour codes or private paths."""
    with path.open("rb") as handle:
        size = handle.seek(0, 2)
        handle.seek(max(0, size - LIVE_LOG_BYTES))
        text = handle.read().decode("utf-8", errors="replace")
    lines = text.splitlines()
    if size > LIVE_LOG_BYTES:
        lines = lines[1:]  # The first line was cut by the byte limit.
    text = re.sub(r"\x1b\[[0-9;]*m", "", "\n".join(lines[-LIVE_LOG_LINES:]))
    return sanitize_text(text, private_paths=[str(root)])


@router.get("/ml/{ml_id}/live")
def live_ml(ml_id: UUID, user: dict = Depends(current_user)):
    """Per-model training progress and the end of the log, read while the run works."""
    row = owned_ml(ml_id, user["app_user_id"])
    root = Path(row["output_dir"])
    result = {"training": None, "log": ""}
    if row["status"] in {"queued", "deleting"} or not root.is_dir():
        return result
    progress = contained(root, root / TRAINING_PROGRESS)
    if progress.is_file():
        with contextlib.suppress(OSError, ValueError):
            data = json.loads(progress.read_text(encoding="utf-8"))
            models = [
                {key: item[key] for key in ("name", "status", "started_at", "finished_at", "duration_seconds") if key in item}
                | ({"error": sanitize_text(str(item["error"]), private_paths=[str(root)])} if item.get("error") else {})
                for item in data.get("models", [])
                if isinstance(item, dict) and isinstance(item.get("name"), str)
            ]
            result["training"] = {"started_at": data.get("started_at"), "finished_at": data.get("finished_at"), "models": models}
    path = contained(root, root / "ml.log")
    if path.is_file():
        result["log"] = log_tail(path, root)
    return result


@router.get("/ml/{ml_id}/files/{artifact_id}")
def download_ml(ml_id: UUID, artifact_id: UUID, user: dict = Depends(current_user)):
    """Download a published ML file without accepting filesystem paths."""
    row = owned_ml(ml_id, user["app_user_id"])
    if row["status"] in {"queued", "running", "deleting"}:
        raise HTTPException(409, "Error: Wait for the running step to finish.")
    artifact = next((item for item in row["artifacts"] if item["artifact_id"] == str(artifact_id)), None)
    if not artifact:
        raise HTTPException(404, "Error: ML file not found.")
    path = contained(row["output_dir"], Path(row["output_dir"]) / artifact["name"])
    if not path.is_file():
        raise HTTPException(404, "Error: ML file is unavailable.")
    return FileResponse(
        path,
        filename=path.name,
        media_type="application/octet-stream",
        headers={"X-Content-Type-Options": "nosniff", "Cache-Control": "private, no-store", "Content-Security-Policy": "sandbox"},
    )
