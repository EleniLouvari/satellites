"""Durable PostgreSQL job queue: enqueue, claim, fencing and reconciliation."""

from uuid import uuid4

from psycopg2.extras import Json

from parcel_app.core import db
from parcel_app.jobs.registry import job_specs, lane_kinds


def enqueue(kind, target_id, user_id, payload=None, conn=None):
    """Insert an idempotent pending task in the caller's transaction."""
    return db.one(
        """INSERT INTO web_jobs(job_id,app_user_id,kind,target_id,payload)
        VALUES(%s,%s,%s,%s,%s) ON CONFLICT(kind,target_id) WHERE status IN ('pending','running')
        DO UPDATE SET target_id=EXCLUDED.target_id RETURNING *""",
        (str(uuid4()), user_id, kind, str(target_id), Json(payload or {})),
        conn=conn,
    )


def owned_claim(job, conn):
    """Lock and verify the claim before changing task-owned database state."""
    row = db.one("SELECT * FROM web_jobs WHERE job_id=%s FOR UPDATE", (str(job["job_id"]),), conn=conn)
    if not row or row["status"] != "running" or str(row["claim_token"]) != str(job["claim_token"]):
        raise RuntimeError("Error: Worker claim is no longer active.")
    if row.get("payload", {}).get("stop_requested"):
        raise RuntimeError("Error: Analysis stop requested; task publication is fenced.")


def claim(lane="all"):
    """Atomically claim one pending task with a fresh fencing token."""
    return db.one(
        """UPDATE web_jobs SET status='running',claim_token=%s,heartbeat_at=now(),updated_at=now()
        WHERE job_id=(SELECT job_id FROM web_jobs WHERE status='pending' AND kind=ANY(%s)
        ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT 1) RETURNING *""",
        (str(uuid4()), lane_kinds(lane)),
    )


def reconcile(lane="all"):
    """Mark abandoned work interrupted only while the caller holds both worker locks."""
    kinds = lane_kinds(lane)
    # Kinds that share one owning record (run/resume) also share a statement; run it once.
    statements = dict.fromkeys(sql for kind in kinds for sql in job_specs()[kind].interrupt_sql)
    with db.connection() as conn:
        # Stop requests and task publication lock jobs before analysis rows.
        db.all_rows(
            "SELECT job_id FROM web_jobs WHERE status='running' AND kind=ANY(%s) ORDER BY job_id FOR UPDATE",
            (kinds,),
            conn=conn,
        )
        for sql in statements:
            db.execute(sql, (kinds,), conn=conn)
        db.execute(
            """UPDATE web_jobs SET status='failed',error='Error: Worker interrupted.',updated_at=now()
            WHERE status='running' AND kind=ANY(%s)""",
            (kinds,),
            conn=conn,
        )


def stop_requested(job):
    """Read a stop request only for the exact active claim of a stoppable job."""
    spec = job_specs().get(job["kind"])
    if spec is None or not spec.stoppable:
        return False
    row = db.one(
        "SELECT payload FROM web_jobs WHERE job_id=%s AND claim_token=%s AND status='running'",
        (str(job["job_id"]), str(job["claim_token"])),
    )
    return bool(row and row["payload"].get("stop_requested"))
