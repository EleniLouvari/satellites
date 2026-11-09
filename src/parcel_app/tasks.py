"""Owned task handlers; all final writes are fenced by the queue claim token."""

import logging
import os
from pathlib import Path
from psycopg2.extras import Json
from . import db
from .artifacts import build_bundle, contained, register_artifacts
from .settings import get_settings


def _owned(job, conn):
    """Lock and verify the claim before changing task-owned database state."""
    row = db.one("SELECT * FROM web_jobs WHERE job_id=%s FOR UPDATE", (str(job["job_id"]),), conn=conn)
    if not row or row["status"] != "running" or str(row["claim_token"]) != str(job["claim_token"]):
        raise RuntimeError("Error: Worker claim is no longer active.")


def _run(job):
    """Fetch an owned run without trusting identifiers from a browser payload."""
    row = db.one("SELECT * FROM analysis_runs WHERE run_id=%s AND app_user_id=%s", (job["target_id"], job["app_user_id"]))
    if row is None:
        raise ValueError("Error: Run is unavailable.")
    contained(get_settings().storage_root, row["output_dir"])
    return row


def _inspect(job):
    """Inspect local vector bytes in the isolated worker child."""
    from .inspection import inspect_upload

    upload = db.one("SELECT * FROM parcel_uploads WHERE upload_id=%s AND app_user_id=%s", (job["target_id"], job["app_user_id"]))
    if upload is None:
        raise ValueError("Error: Upload is unavailable.")
    path = contained(get_settings().storage_root, upload["storage_path"])
    metadata = inspect_upload(path)
    with db.connection() as conn:
        _owned(job, conn)
        db.execute(
            "UPDATE parcel_uploads SET status='ready',metadata=%s,error=NULL WHERE upload_id=%s",
            (Json(metadata), job["target_id"]),
            conn=conn,
        )


def _validate(job):
    """Plan locally and commit results only against the requested exact revision."""
    from data_preparation.parcel_stats.run_config import prepare_run
    from .artifacts import checksum

    run = _run(job)
    revision = job["payload"]["revision"]
    if run["revision"] != revision:
        return
    upload = db.one(
        "SELECT * FROM parcel_uploads WHERE upload_id=%s AND app_user_id=%s", (str(run["upload_id"]), run["app_user_id"])
    )
    if not upload or upload["status"] != "ready":
        raise ValueError("Error: Upload inspection must finish before validation.")
    path = contained(get_settings().storage_root, upload["storage_path"])
    if checksum(path) != upload["sha256"]:
        raise ValueError("Error: Uploaded file changed; upload the original file again.")
    config = {
        "schema_version": 1,
        "input": {
            "path": str(path),
            "layer": run["layer"],
            "parcel_id_field": run["parcel_id_field"],
            "partition_field": run["partition_field"],
        },
        "processing": run["processing"],
        "output_dir": run["output_dir"],
        "run_identifier": f"web-{run['run_id']}",
        "scheduling": run.get("scheduling", "batches"),
    }
    prepared = prepare_run(config)
    summary = prepared.summary()
    summary.pop("config", None)
    summary["input_sha256"] = prepared.input_sha256
    with db.connection() as conn:
        _owned(job, conn)
        db.execute(
            """UPDATE analysis_runs SET status='ready',validated_revision=%s,config=%s,summary=%s,
            error=NULL,updated_at=now() WHERE run_id=%s AND revision=%s AND status='validating'""",
            (revision, Json(prepared.config), Json(summary), job["target_id"], revision),
            conn=conn,
        )


def _credentials(run):
    """Decrypt exactly the ordered frozen account identities, never all owner accounts."""
    from .security import cipher

    rows = db.all_rows(
        """SELECT ra.username AS frozen_username,e.username,e.password
        FROM run_accounts ra JOIN eo_users e ON e.eo_user_id=ra.eo_user_id
        JOIN app_users u ON u.app_user_id=e.app_user_id
        WHERE ra.run_id=%s AND e.app_user_id=%s AND u.is_active ORDER BY ra.position""",
        (str(run["run_id"]), run["app_user_id"]),
    )
    if len(rows) != len(run["account_ids"]) or not rows:
        raise ValueError("Error: Selected Copernicus accounts are unavailable.")
    if any(row["username"] != row["frozen_username"] for row in rows):
        raise ValueError("Error: Copernicus account identity changed; restore the original account.")
    return [(row["frozen_username"], cipher().decrypt(row["password"].encode()).decode()) for row in rows]


class _SecretFilter(logging.Filter):
    """Redact every standard logging handler's message before it is formatted."""

    def __init__(self, secrets):
        super().__init__()
        self.secrets = secrets

    def filter(self, record):
        from .artifacts import sanitize_text

        record.msg = sanitize_text(record.getMessage(), self.secrets)
        record.args = ()
        # Third-party exceptions may embed passwords or tokens in tracebacks.
        record.exc_info = None
        record.exc_text = None
        return True


def _execute_run(job):
    """Execute the dictionary runner with immutable config and selected credentials."""
    from data_preparation.parcel_stats.runner import run_parcel_stats, resume_parcel_stats
    from .artifacts import checksum

    run = _run(job)
    accounts = _credentials(run)
    if checksum(Path(run["config"]["input"]["path"])) != run["summary"]["input_sha256"]:
        raise ValueError("Error: Uploaded file changed; restore the original file before resuming.")
    secrets = [item for pair in accounts for item in pair] + [
        str(get_settings().storage_root),
        run["output_dir"],
        run["config"]["input"]["path"],
    ]
    from .progress import event_logging

    with db.connection() as conn:
        _owned(job, conn)
        db.execute(
            "UPDATE analysis_runs SET status='running',error=NULL,updated_at=now() WHERE run_id=%s",
            (job["target_id"],),
            conn=conn,
        )
        db.execute(
            "INSERT INTO run_events(run_id,stage,message) VALUES(%s,'processing','Analysis started.')",
            (job["target_id"],),
            conn=conn,
        )
    # Suppress raw native/third-party stdout and stderr; only sanitized pipeline
    # logs and explicit lifecycle events may reach the web application.
    with open(os.devnull, "w") as sink:
        import contextlib

        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink), event_logging(job, secrets):
            if job["kind"] == "resume" and (Path(run["output_dir"]) / "run.json").is_file():
                resume_parcel_stats(run["output_dir"], accounts=accounts, app_user_id=run["app_user_id"])
            else:
                run_parcel_stats(run["config"], accounts=accounts, app_user_id=run["app_user_id"])
    with db.connection() as conn:
        _owned(job, conn)
        register_artifacts(run, secrets, conn=conn)
        db.execute(
            "UPDATE analysis_runs SET status='complete',error=NULL,updated_at=now() WHERE run_id=%s",
            (job["target_id"],),
            conn=conn,
        )
        db.execute(
            "INSERT INTO run_events(run_id,stage,message) VALUES(%s,'publication','Verified results are ready.')",
            (job["target_id"],),
            conn=conn,
        )


def _test_account(job):
    """Authenticate explicitly against the fixed backend without creating a remote job."""
    import openeo
    from types import SimpleNamespace
    from data_preparation.sources.openeo.cubes import OpenEOCubePipeline
    from data_preparation.parcel_stats.core.base import ParcelStatsBase
    from .security import cipher

    account = db.one("SELECT * FROM eo_users WHERE eo_user_id=%s AND app_user_id=%s", (int(job["target_id"]), job["app_user_id"]))
    if not account:
        raise ValueError("Error: Copernicus account is unavailable.")
    credentials = SimpleNamespace(
        openeo_username=account["username"],
        openeo_password=cipher().decrypt(account["password"].encode()).decode(),
        OPENEO_OIDC_PASSWORD_CLIENT_ID=ParcelStatsBase.OPENEO_OIDC_PASSWORD_CLIENT_ID,
    )
    import contextlib

    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        OpenEOCubePipeline._authenticate_openeo_connection(credentials, openeo.connect(ParcelStatsBase.OPENEO_URL))
    with db.connection() as conn:
        _owned(job, conn)
        db.execute(
            "UPDATE eo_users SET test_status='connected',tested_at=now() WHERE eo_user_id=%s AND app_user_id=%s",
            (int(job["target_id"]), job["app_user_id"]),
            conn=conn,
        )


def _failure(job, error=None):
    """Expose actionable fixed messages without embedding raw third-party errors."""
    messages = {
        "inspect": "Error: File inspection failed; verify supported format, geometry and CRS.",
        "validate": "Error: Preflight failed; check parcel IDs, geometry, date range and processing settings.",
        "test_account": "Error: Connection test failed; check credentials and backend availability.",
        "bundle": "Error: Bundle creation failed; verified artifacts may be unavailable.",
        "eda": "Error: EDA failed; check the selected columns and configuration, then create a new EDA analysis.",
    }
    message = messages.get(job["kind"], "Error: Analysis failed; inspect sanitized logs and resume when ready.")
    if job["kind"] == "eda" and error is not None:
        message = (
            f"Error: EDA failed ({type(error).__name__}). Check the EDA log if available, "
            "then create a new EDA analysis."
        )
    with db.connection() as conn:
        _owned(job, conn)
        if job["kind"] == "inspect":
            db.execute(
                "UPDATE parcel_uploads SET status='failed',error=%s WHERE upload_id=%s", (message, job["target_id"]), conn=conn
            )
        elif job["kind"] == "eda":
            db.execute(
                "UPDATE eda_runs SET status='failed',error=%s,finished_at=now() WHERE eda_id=%s",
                (message, job["target_id"]),
                conn=conn,
            )
        elif job["kind"] in {"validate", "run", "resume"}:
            db.execute(
                "UPDATE analysis_runs SET status='failed',error=%s,updated_at=now() WHERE run_id=%s",
                (message, job["target_id"]),
                conn=conn,
            )
            db.execute(
                "INSERT INTO run_events(run_id,stage,level,message) VALUES(%s,%s,'ERROR',%s)",
                (job["target_id"], job["kind"], message),
                conn=conn,
            )
        elif job["kind"] == "test_account":
            db.execute(
                "UPDATE eo_users SET test_status='failed',tested_at=now() WHERE eo_user_id=%s AND app_user_id=%s",
                (int(job["target_id"]), job["app_user_id"]),
                conn=conn,
            )
        db.execute(
            "UPDATE web_jobs SET status='failed',error=%s,updated_at=now() WHERE job_id=%s",
            (message, str(job["job_id"])),
            conn=conn,
        )


def execute_task(job):
    """Dispatch a claimed task and safely persist its terminal status."""
    from .eda_task import execute_eda

    try:
        with db.connection() as conn:
            _owned(job, conn)
        handlers = {
            "inspect": _inspect,
            "validate": _validate,
            "run": _execute_run,
            "resume": _execute_run,
            "test_account": _test_account,
            "bundle": lambda task: _bundle(task),
            "eda": execute_eda,
        }
        handlers[job["kind"]](job)
        with db.connection() as conn:
            _owned(job, conn)
            db.execute("UPDATE web_jobs SET status='done',updated_at=now() WHERE job_id=%s", (str(job["job_id"]),), conn=conn)
    except Exception as error:
        try:
            if job["kind"] in {"run", "resume"}:
                from .artifacts import register_diagnostics

                run = _run(job)
                if not (Path(run["output_dir"]) / "run.json").is_file():
                    raise ValueError("No initialized run; diagnostics must not occupy the fresh run directory.")
                credentials = _credentials(run)
                secrets = [item for pair in credentials for item in pair]
                with db.connection() as conn:
                    _owned(job, conn)
                    register_diagnostics(run, secrets, conn=conn)
        except Exception:
            pass
        try:
            _failure(job, error)
        except Exception:
            # Connection loss or fencing: parent reconciliation is authoritative.
            pass


def _bundle(job):
    """Build an archive under claim ownership."""
    run = _run(job)
    with db.connection() as conn:
        _owned(job, conn)
        build_bundle(run, conn=conn)
