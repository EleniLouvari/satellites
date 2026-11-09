"""Delete only remote jobs recorded beneath a run's trusted storage directory."""

import re
from pathlib import Path

import pandas as pd

from .artifacts import contained


def _text(value):
    """Normalize nullable job database cells without turning nulls into job IDs."""
    return "" if value is None or pd.isna(value) else str(value).strip()


def collect_jobs(root, usernames, *, expected_batches=0, expected_partitions=0):
    """Include current and retry IDs, with frozen partition order for legacy owners."""
    root = Path(root).resolve()
    owners = {name.strip().casefold(): name for name in usernames}
    jobs = set()
    tracked_batches, tracked_partitions = 0, set()
    for source in sorted(root.rglob("tile_jobs.parquet")):
        source = contained(root, source)
        partition = re.fullmatch(r"partition_([1-9][0-9]*)", source.relative_to(root).parts[0])
        for row in pd.read_parquet(source).to_dict("records"):
            ids = {_text(row.get("id")), *filter(None, _text(row.get("failed_job_ids")).split(";"))} - {""}
            if not ids:
                continue
            tracked_batches += 1
            if partition:
                tracked_partitions.add(int(partition[1]))
            owner = _text(row.get("openeo_user"))
            if not owner and partition and usernames:
                owner = usernames[(int(partition[1]) - 1) % len(usernames)]
            if owner.casefold() not in owners:
                raise ValueError("Error: Restore the saved owner of the run's remote jobs before deletion.")
            for job_id in ids:
                if not re.fullmatch(r"[\w.~\-]+", job_id, re.ASCII):
                    raise ValueError("Error: The run contains an invalid remote job identifier.")
                jobs.add((owners[owner.casefold()], job_id))
    if tracked_batches < expected_batches or not set(range(1, expected_partitions + 1)).issubset(tracked_partitions):
        raise ValueError("Error: Remote tracking is incomplete; restore the run's saved job databases before deletion.")
    return [{"username": owner, "job_id": job_id} for owner, job_id in sorted(jobs)]


def connect_account(username, password):
    """Authenticate only to the configured fixed Copernicus backend."""
    import openeo
    from data_preparation.parcel_stats.core.base import ParcelStatsBase

    connection = openeo.connect(ParcelStatsBase.OPENEO_URL, auto_validate=False, default_timeout=30)
    connection.authenticate_oidc_resource_owner_password_credentials(
        username=username, password=password, client_id=ParcelStatsBase.OPENEO_OIDC_PASSWORD_CLIENT_ID
    )
    return connection


def delete_jobs(jobs, credentials, on_deleted=None):
    """Delete tracked IDs only; already absent jobs make retry safe after partial failure."""
    from openeo.rest import OpenEoApiError

    passwords = dict(credentials)
    if any(item["username"] not in passwords for item in jobs):
        raise ValueError("Error: The run's original Copernicus accounts are required for remote cleanup.")
    connections = {}
    for index, item in enumerate(jobs, 1):
        username = item["username"]
        if username not in connections:
            connections[username] = connect_account(username, passwords[username])
        try:
            # DELETE /jobs/{id} also stops computation and removes its remote results.
            connections[username].job(item["job_id"]).delete()
        except OpenEoApiError as error:
            if error.http_status_code != 404 or error.code != "JobNotFound":
                raise
        if on_deleted:
            on_deleted(index)
