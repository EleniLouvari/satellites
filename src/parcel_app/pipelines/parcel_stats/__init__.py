"""Parcel statistics analyses: configuration, validation, runs, results and deletion."""

from parcel_app.jobs.registry import JobSpec

_TASKS = "parcel_app.pipelines.parcel_stats.tasks"
_RUN_FAILED = "Error: Analysis failed; inspect sanitized logs and resume when ready."
_RUN_INTERRUPTED = """UPDATE analysis_runs r SET status='interrupted',error=CASE
            WHEN j.payload->>'stop_requested'='true' THEN 'Stopped by request. Resume analysis to reuse completed checkpoints.'
            ELSE 'Error: Worker interrupted; resume explicitly.' END,updated_at=now()
            FROM web_jobs j WHERE r.run_id::text=j.target_id AND j.status='running'
            AND j.kind IN ('run','resume') AND j.kind=ANY(%s) AND r.status IN ('queued','running')"""

JOBS = (
    JobSpec(
        kind="validate",
        lane="small",
        handler=f"{_TASKS}:validate",
        failure_message="Error: Preflight failed; check parcel IDs, geometry, date range and processing settings.",
        on_failure=f"{_TASKS}:mark_run_failed",
        interrupt_sql=(
            """UPDATE analysis_runs SET status='draft',error='Error: Validation interrupted; validate again.'
            WHERE status='validating' AND run_id::text IN
            (SELECT target_id FROM web_jobs WHERE status='running' AND kind='validate' AND kind=ANY(%s))""",
        ),
    ),
    JobSpec(
        kind="run",
        lane="run",
        handler=f"{_TASKS}:execute_run",
        failure_message=_RUN_FAILED,
        on_failure=f"{_TASKS}:mark_run_failed",
        interrupt_sql=(_RUN_INTERRUPTED,),
        stoppable=True,
    ),
    JobSpec(
        kind="resume",
        lane="run",
        handler=f"{_TASKS}:execute_run",
        failure_message=_RUN_FAILED,
        on_failure=f"{_TASKS}:mark_run_failed",
        interrupt_sql=(_RUN_INTERRUPTED,),
        stoppable=True,
    ),
    JobSpec(
        kind="bundle",
        lane="run",
        handler=f"{_TASKS}:bundle",
        failure_message="Error: Bundle creation failed; verified artifacts may be unavailable.",
    ),
)
ROUTERS = (
    "parcel_app.pipelines.parcel_stats.routes",
    "parcel_app.pipelines.parcel_stats.files",
    "parcel_app.pipelines.parcel_stats.config",
    "parcel_app.pipelines.parcel_stats.deletion",
)
