"""Run one claimed job in the worker child and persist its terminal status."""

import contextlib

from parcel_app.core import db
from parcel_app.jobs.queue import owned_claim
from parcel_app.jobs.registry import job_specs, resolve


def execute_task(job):
    """Dispatch a claimed task and safely persist its terminal status."""
    try:
        with db.connection() as conn:
            owned_claim(job, conn)
        resolve(job_specs()[job["kind"]].handler)(job)
        with db.connection() as conn:
            owned_claim(job, conn)
            db.execute("UPDATE web_jobs SET status='done',updated_at=now() WHERE job_id=%s", (str(job["job_id"]),), conn=conn)
    except Exception as error:  # noqa: BLE001 - worker boundary: every failure is recorded with a fixed message
        # Connection loss or fencing: parent reconciliation is authoritative.
        with contextlib.suppress(Exception):
            fail(job, error)


def fail(job, error=None):
    """Expose actionable fixed messages without embedding raw third-party errors."""
    spec = job_specs()[job["kind"]]
    message = spec.failure_message
    if spec.error_message and error is not None:
        message = spec.error_message.format(error_type=type(error).__name__)
    with db.connection() as conn:
        owned_claim(job, conn)
        if spec.on_failure:
            resolve(spec.on_failure)(job, message, conn)
        db.execute(
            "UPDATE web_jobs SET status='failed',error=%s,updated_at=now() WHERE job_id=%s",
            (message, str(job["job_id"])),
            conn=conn,
        )
