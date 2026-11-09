"""Bounded asynchronous publication of sanitized pipeline log events."""

import logging
import os
import queue
import threading
from contextlib import contextmanager

from . import db
from .artifacts import sanitize_text


@contextmanager
def event_logging(job, secrets):
    """Redact records before handlers and publish bounded events without blocking jobs."""
    pending = queue.Queue(maxsize=1000)
    stop = threading.Event()
    previous = logging.getLogRecordFactory()
    protected = list(secrets) + [os.getenv("POSTGRES_PASSWORD", ""), os.getenv("SATELLITES_CREDENTIAL_KEY", "")]

    def factory(*args, **kwargs):
        record = previous(*args, **kwargs)
        record.msg = sanitize_text(record.getMessage(), protected)
        record.args = ()
        record.exc_info = record.exc_text = None
        if record.name.startswith("parcel_stats_pipeline") and record.levelno >= logging.INFO:
            source = (
                "acquisition"
                if ".source" in record.name
                else "local processing"
                if any(s in record.name for s in ("raster", "statistic"))
                else "orchestration"
            )
            try:
                pending.put_nowait((source, record.levelname, record.msg[:4000]))
            except queue.Full:
                pass  # Full sanitized log files remain downloadable after execution.
        return record

    def publish():
        while not stop.is_set() or not pending.empty():
            batch = []
            try:
                batch.append(pending.get(timeout=0.5))
            except queue.Empty:
                continue
            while len(batch) < 100:
                try:
                    batch.append(pending.get_nowait())
                except queue.Empty:
                    break
            try:
                with db.connection() as conn:
                    claim = db.one("SELECT status,claim_token FROM web_jobs WHERE job_id=%s", (str(job["job_id"]),), conn=conn)
                    if not claim or claim["status"] != "running" or str(claim["claim_token"]) != str(job["claim_token"]):
                        return
                    for stage, level, message in batch:
                        db.execute(
                            "INSERT INTO run_events(run_id,stage,level,message) VALUES(%s,%s,%s,%s)",
                            (job["target_id"], stage, level, message),
                            conn=conn,
                        )
            except Exception:
                # Operational telemetry cannot break scientific processing.
                pass

    logging.setLogRecordFactory(factory)
    thread = threading.Thread(target=publish, daemon=True, name="parcel-safe-events")
    thread.start()
    try:
        yield
    finally:
        logging.setLogRecordFactory(previous)
        stop.set()
        thread.join(timeout=15)
