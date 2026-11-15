"""Background jobs for Copernicus (openEO) accounts."""

import contextlib
import os

from parcel_app.core import db
from parcel_app.jobs.queue import owned_claim


def test_account(job):
    """Authenticate explicitly against the fixed backend without creating a remote job."""
    from types import SimpleNamespace

    import openeo

    from data_preparation.parcel_stats.core.base import ParcelStatsBase
    from data_preparation.sources.openeo.cubes import OpenEOCubePipeline
    from parcel_app.core.security import cipher

    account = db.one("SELECT * FROM eo_users WHERE eo_user_id=%s AND app_user_id=%s", (int(job["target_id"]), job["app_user_id"]))
    if not account:
        raise ValueError("Error: Copernicus account is unavailable.")
    credentials = SimpleNamespace(
        openeo_username=account["username"],
        openeo_password=cipher().decrypt(account["password"].encode()).decode(),
        OPENEO_OIDC_PASSWORD_CLIENT_ID=ParcelStatsBase.OPENEO_OIDC_PASSWORD_CLIENT_ID,
    )
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        OpenEOCubePipeline._authenticate_openeo_connection(credentials, openeo.connect(ParcelStatsBase.OPENEO_URL))
    with db.connection() as conn:
        owned_claim(job, conn)
        db.execute(
            "UPDATE eo_users SET test_status='connected',tested_at=now() WHERE eo_user_id=%s AND app_user_id=%s",
            (int(job["target_id"]), job["app_user_id"]),
            conn=conn,
        )


def mark_test_failed(job, message, conn):
    """Record a failed connection test on the owned account."""
    db.execute(
        "UPDATE eo_users SET test_status='failed',tested_at=now() WHERE eo_user_id=%s AND app_user_id=%s",
        (int(job["target_id"]), job["app_user_id"]),
        conn=conn,
    )
