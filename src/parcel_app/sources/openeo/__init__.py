"""Copernicus Data Space (openEO) accounts, connection tests and remote job cleanup."""

from parcel_app.jobs.registry import JobSpec

JOBS = (
    JobSpec(
        kind="test_account",
        lane="small",
        handler="parcel_app.sources.openeo.tasks:test_account",
        failure_message="Error: Connection test failed; check credentials and backend availability.",
        on_failure="parcel_app.sources.openeo.tasks:mark_test_failed",
    ),
)
ROUTERS = ("parcel_app.sources.openeo.accounts",)
