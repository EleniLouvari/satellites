"""Exploratory data analysis of a completed parcel-statistics result."""

from parcel_app.jobs.registry import JobSpec

JOBS = (
    JobSpec(
        kind="eda",
        lane="run",
        handler="parcel_app.pipelines.eda.tasks:execute_eda",
        failure_message="Error: EDA failed; check the selected columns and configuration, then create a new EDA analysis.",
        error_message="Error: EDA failed ({error_type}). Check the EDA log if available, then create a new EDA analysis.",
        on_failure="parcel_app.pipelines.eda.tasks:mark_eda_failed",
        interrupt_sql=(
            """UPDATE eda_runs SET status='interrupted',finished_at=now(),
            error='Error: EDA worker interrupted; create a new analysis with these settings.'
            WHERE eda_id::text IN (SELECT target_id FROM web_jobs WHERE status='running' AND kind='eda' AND kind=ANY(%s))
            AND status IN ('queued','running')""",
        ),
    ),
)
ROUTERS = ("parcel_app.pipelines.eda.routes",)
