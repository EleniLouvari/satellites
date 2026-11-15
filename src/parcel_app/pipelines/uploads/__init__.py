"""Parcel file uploads and their background inspection."""

from parcel_app.jobs.registry import JobSpec

JOBS = (
    JobSpec(
        kind="inspect",
        lane="small",
        handler="parcel_app.pipelines.uploads.tasks:inspect",
        failure_message="Error: File inspection failed; verify supported format, geometry and CRS.",
        on_failure="parcel_app.pipelines.uploads.tasks:mark_upload_failed",
        interrupt_sql=(
            """UPDATE parcel_uploads SET status='failed',error='Error: Inspection interrupted; upload again.'
            WHERE upload_id::text IN (SELECT target_id FROM web_jobs WHERE status='running' AND kind='inspect' AND kind=ANY(%s))""",
        ),
    ),
)
ROUTERS = ("parcel_app.pipelines.uploads.routes",)
