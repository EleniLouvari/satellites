"""Parcel-statistics job handlers; all final writes are fenced by the queue claim token."""

import contextlib
import logging
import os
from pathlib import Path

from psycopg2.extras import Json

from parcel_app.core import db
from parcel_app.core.storage import checksum, contained, link_or_copy, run_directory, sanitize_text
from parcel_app.jobs.queue import owned_claim
from parcel_app.settings import get_settings
from parcel_app.sources.openeo.credentials import run_credentials

from .artifacts import build_bundle, register_artifacts


def _run(job):
    """Fetch an owned run without trusting identifiers from a browser payload."""
    row = db.one("SELECT * FROM analysis_runs WHERE run_id=%s AND app_user_id=%s", (job["target_id"], job["app_user_id"]))
    if row is None:
        raise ValueError("Error: Run is unavailable.")
    contained(get_settings().storage_root, row["output_dir"])
    return row


def validate(job):
    """Plan locally and commit results only against the requested exact revision."""
    from data_preparation.parcel_stats.run_config import prepare_run

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
        "source": run.get("source", "openeo"),
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
        owned_claim(job, conn)
        db.execute(
            """UPDATE analysis_runs SET status='ready',validated_revision=%s,config=%s,summary=%s,
            error=NULL,updated_at=now() WHERE run_id=%s AND revision=%s AND status='validating'""",
            (revision, Json(prepared.config), Json(summary), job["target_id"], revision),
            conn=conn,
        )


class _SecretFilter(logging.Filter):
    """Redact every standard logging handler's message before it is formatted."""

    def __init__(self, secrets):
        super().__init__()
        self.secrets = secrets

    def filter(self, record):
        record.msg = sanitize_text(record.getMessage(), self.secrets)
        record.args = ()
        # Third-party exceptions may embed passwords or tokens in tracebacks.
        record.exc_info = None
        record.exc_text = None
        return True


def _snapshot_input(job, run):
    """Before the first start, give the run its own copy of the validated upload in ``input/``.

    Runs allocated before the ``input/``/``output/`` split keep reading their upload in place.
    """
    root = Path(os.path.abspath(run_directory(run["app_user_id"], run["run_id"])))
    source = Path(os.path.abspath(run["config"]["input"]["path"]))
    output = Path(os.path.abspath(run["output_dir"]))
    if output != root / "output" or (output / "run.json").is_file() or source.parent == root / "input":
        return run
    target = link_or_copy(source, root / "input" / source.name)
    config = {**run["config"], "input": {**run["config"]["input"], "path": str(target)}}
    with db.connection() as conn:
        owned_claim(job, conn)
        db.execute("UPDATE analysis_runs SET config=%s WHERE run_id=%s", (Json(config), job["target_id"]), conn=conn)
    return {**run, "config": config}


def _provider_credentials(run):
    """Keep provider secrets in worker memory and out of persisted configuration."""
    if run.get("source", "openeo") == "hub":
        from parcel_app.sources.hub.credentials import require_configuration, run_secrets

        require_configuration()
        return [], run_secrets()
    accounts = run_credentials(run)
    return accounts, [item for pair in accounts for item in pair]


def _execute_run(job):
    """Execute the dictionary runner with immutable config and selected credentials."""
    from data_preparation.parcel_stats.runner import resume_parcel_stats, run_parcel_stats

    run = _snapshot_input(job, _run(job))
    accounts, secrets = _provider_credentials(run)
    if checksum(Path(run["config"]["input"]["path"])) != run["summary"]["input_sha256"]:
        raise ValueError("Error: Uploaded file changed; restore the original file before resuming.")
    private_paths = [
        str(get_settings().storage_root),
        run["output_dir"],
        run["config"]["input"]["path"],
    ]
    from .progress import event_logging

    with db.connection() as conn:
        owned_claim(job, conn)
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

        with (
            contextlib.redirect_stdout(sink),
            contextlib.redirect_stderr(sink),
            event_logging(job, secrets, private_paths=private_paths),
        ):
            if job["kind"] == "resume" and (Path(run["output_dir"]) / "run.json").is_file():
                resume_parcel_stats(
                    run["output_dir"], accounts=accounts, app_user_id=run["app_user_id"],
                    retry_exhausted_memory=job.get("payload", {}).get("retry_exhausted_memory", False),
                )
            else:
                run_parcel_stats(run["config"], accounts=accounts, app_user_id=run["app_user_id"])
    with db.connection() as conn:
        owned_claim(job, conn)
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


def bundle(job):
    """Build an archive under claim ownership."""
    run = _run(job)
    with db.connection() as conn:
        owned_claim(job, conn)
        build_bundle(run, conn=conn)


def execute_run(job):
    """Run or resume, publishing sanitized diagnostics when an initialized run fails."""
    try:
        _execute_run(job)
    except Exception:
        # Best effort only: a diagnostics problem must never replace the original error.
        with contextlib.suppress(Exception):
            _publish_diagnostics(job)
        raise


def _publish_diagnostics(job):
    """Export configuration and redacted logs of an initialized run that failed."""
    from .artifacts import register_diagnostics

    run = _run(job)
    if not (Path(run["output_dir"]) / "run.json").is_file():
        # No initialized run; diagnostics must not occupy the fresh run directory.
        return
    if run.get("source", "openeo") == "hub":
        from parcel_app.sources.hub.credentials import run_secrets

        secrets = run_secrets()
    else:
        secrets = [item for pair in run_credentials(run) for item in pair]
    with db.connection() as conn:
        owned_claim(job, conn)
        register_diagnostics(run, secrets, conn=conn)


def mark_run_failed(job, message, conn):
    """Record a failed validation, run or resume on the analysis and its event log."""
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
