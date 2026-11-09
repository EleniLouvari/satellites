"""Remote deletion tests use saved job records and mocks, never a live backend."""

from unittest.mock import Mock

import pandas as pd
import pytest


def test_remote_manifest_includes_retries_and_resolves_saved_owners(tmp_path):
    from parcel_app.remote_cleanup import collect_jobs

    folder = tmp_path / "partition_2" / "cubes"
    folder.mkdir(parents=True)
    pd.DataFrame(
        [
            {"id": "job-current", "failed_job_ids": "job-old;job-current", "openeo_user": "first"},
            {"id": "job-legacy", "failed_job_ids": "", "openeo_user": None},
        ]
    ).to_parquet(folder / "tile_jobs.parquet")
    assert collect_jobs(tmp_path, ["first", "second"]) == [
        {"username": "first", "job_id": "job-current"},
        {"username": "first", "job_id": "job-old"},
        {"username": "second", "job_id": "job-legacy"},
    ]


@pytest.mark.parametrize("owner,job_id", [("foreign", "job"), ("first", "../other")])
def test_remote_manifest_rejects_unknown_owner_or_unsafe_job_id(tmp_path, owner, job_id):
    from parcel_app.remote_cleanup import collect_jobs

    pd.DataFrame([{"id": job_id, "openeo_user": owner}]).to_parquet(tmp_path / "tile_jobs.parquet")
    with pytest.raises(ValueError):
        collect_jobs(tmp_path, ["first"])


def test_remote_cleanup_deletes_only_manifest_ids_and_accepts_already_deleted(monkeypatch):
    from openeo.rest import OpenEoApiError
    from parcel_app import remote_cleanup

    connection = Mock()
    connection.job.return_value.delete.side_effect = [
        None,
        OpenEoApiError(http_status_code=404, code="JobNotFound", message="gone"),
    ]
    connect = Mock(return_value=connection)
    monkeypatch.setattr(remote_cleanup, "connect_account", connect)
    jobs = [{"username": "first", "job_id": "one"}, {"username": "first", "job_id": "gone"}]
    remote_cleanup.delete_jobs(jobs, [("first", "secret"), ("unused", "secret")])
    connect.assert_called_once_with("first", "secret")
    assert [call.args[0] for call in connection.job.call_args_list] == ["one", "gone"]
    connection.list_jobs.assert_not_called()
    connection.job.return_value.delete.side_effect = OpenEoApiError(http_status_code=403, code="Forbidden", message="denied")
    with pytest.raises(OpenEoApiError):
        remote_cleanup.delete_jobs(jobs, [("first", "secret")])


def test_remote_manifest_refuses_partial_tracking_coverage(tmp_path):
    from parcel_app.remote_cleanup import collect_jobs

    folder = tmp_path / "partition_1"
    folder.mkdir()
    pd.DataFrame([{"id": "one", "openeo_user": "first", "failed_job_ids": "retry-one;retry-two"}]).to_parquet(
        folder / "tile_jobs.parquet"
    )
    with pytest.raises(ValueError, match="tracking"):
        collect_jobs(tmp_path, ["first"], expected_batches=2, expected_partitions=1)
    with pytest.raises(ValueError, match="tracking"):
        collect_jobs(tmp_path, ["first"], expected_batches=1, expected_partitions=2)
