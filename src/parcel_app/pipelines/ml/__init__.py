"""ML classification scenarios of a completed EDA analysis, run step by step on the parcel result."""

from parcel_app.jobs.registry import JobSpec

JOBS = (
    JobSpec(
        kind="ml",
        lane="run",
        handler="parcel_app.pipelines.ml.tasks:execute_ml",
        failure_message="Error: ML classification failed; check the ML log, then retry the step or create a new ML run.",
        error_message="Error: ML classification failed ({error_type}). Check the ML log, then retry the step or create a new ML run.",
        on_failure="parcel_app.pipelines.ml.tasks:mark_ml_failed",
        interrupt_sql=(
            """UPDATE ml_runs SET status='interrupted',finished_at=now(),
            error='Error: ML worker interrupted; continue the run to reuse its finished steps and models.'
            WHERE ml_id::text IN (SELECT target_id FROM web_jobs WHERE status='running' AND kind='ml' AND kind=ANY(%s))
            AND status IN ('queued','running')""",
        ),
    ),
)
ROUTERS = ("parcel_app.pipelines.ml.routes",)
