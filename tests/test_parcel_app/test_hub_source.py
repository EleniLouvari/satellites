"""HUB provider boundaries without remote calls or application database access."""

import importlib
import json
from uuid import uuid4

import pytest
from fastapi import HTTPException

from parcel_app.pipelines.parcel_stats import config, routes
from parcel_app.settings import get_settings


def test_hub_schema_exposes_only_safe_source_options(monkeypatch):
    """Catch accidental exposure of credentials or irrelevant openEO job controls."""
    schema = config.configuration(user={}, source="hub")
    assert schema["source"] == "hub"
    fields = {field["name"] for field in schema["fields"]}
    assert {"hub_read_workers", "hub_block_size"} <= fields
    assert not {"hub_catalog_url", "hub_client_id", "tile_size_metres", "openeo_parallel_jobs"} & fields
    assert schema["defaults"]["working_epsg"] == 2100
    assert config.configuration(user={})["source"] == "openeo"


def test_hub_concurrency_limit_is_enforced(monkeypatch):
    """Catch bypassing the operator's remote-read concurrency ceiling."""
    monkeypatch.setenv("PARCEL_MAX_HUB_READ_WORKERS", "2")
    get_settings.cache_clear()
    try:
        with pytest.raises(ValueError, match="worker"):
            config.check_processing(
                {"start_date": "2024-01-01", "end_date": "2024-01-31", "hub_read_workers": 3}, source="hub",
            )
    finally:
        get_settings.cache_clear()


def _configure_hub(monkeypatch, value):
    """Set well-formed service URLs and the given value for every other HUB setting."""
    monkeypatch.setenv("CATALOG_URL", "https://catalog.example.test")
    monkeypatch.setenv("KEYCLOAK_URL", "https://auth.example.test/realms/hub/")
    monkeypatch.setenv("CATALOG_CLIENT_ID", value)
    monkeypatch.setenv("CATALOG_CLIENT_SECRET", value)


def test_hub_status_never_exposes_environment_values(monkeypatch):
    """Catch leaking service credentials through the authenticated status response."""
    _configure_hub(monkeypatch, "private-value")
    module = importlib.import_module("parcel_app.sources.hub.routes")
    assert module.status(user={"app_user_id": 7}) == {"configured": True}
    monkeypatch.delenv("CATALOG_CLIENT_SECRET")
    assert module.status(user={"app_user_id": 7}) == {"configured": False}


def _draft(**overrides):
    """Build the minimum provider-selectable draft."""
    return routes.RunDraft(**{
        "name": "HUB test", "upload_id": uuid4(), "layer": "parcels", "parcel_id_field": "parcel_id",
        "processing": {"start_date": "2024-01-01", "end_date": "2024-01-31"},
        "source": "hub", "account_ids": [], **overrides,
    })


def test_hub_draft_skips_copernicus_and_normalizes_scheduling(monkeypatch):
    """Catch HUB drafts being forced through Copernicus account ownership checks."""
    monkeypatch.setattr(routes, "owned_upload", lambda *a, **k: {
        "status": "ready", "metadata": {"layers": [{"name": "parcels", "columns": ["parcel_id"]}]},
    })
    monkeypatch.setattr(routes, "selected_accounts", lambda *a, **k: pytest.fail("Copernicus lookup for HUB"))
    _configure_hub(monkeypatch, "configured")
    draft = _draft(scheduling="partitions")
    routes.validate_draft(draft, 7)
    assert draft.scheduling == "batches"
    with pytest.raises(HTTPException) as error:
        routes.validate_draft(_draft(account_ids=[1]), 7)
    assert error.value.status_code == 422


def test_openeo_still_requires_accounts(monkeypatch):
    """Catch weakening the existing openEO account requirement while adding HUB."""
    monkeypatch.setattr(routes, "owned_upload", lambda *a, **k: {
        "status": "ready", "metadata": {"layers": [{"name": "parcels", "columns": ["parcel_id"]}]},
    })
    with pytest.raises(HTTPException) as error:
        routes.validate_draft(_draft(source="openeo"), 7)
    assert error.value.status_code == 422


def test_hub_progress_reports_partitions_without_remote_jobs(tmp_path):
    """Catch HUB progress staying at zero remote openEO jobs throughout execution."""
    import pandas as pd

    from parcel_app.pipelines.parcel_stats.completion import run_progress

    for index in (1, 2):
        partition = tmp_path / f"partition_{index}"
        partition.mkdir()
        pd.DataFrame({"batch_number": [1, 1, 2]}).to_parquet(partition / "hub_item_manifest.parquet")
    (tmp_path / "partition_1" / "satellite_partition_completion.json").write_text(json.dumps({"status": "complete"}))
    progress = run_progress({
        "source": "hub", "status": "running", "output_dir": str(tmp_path),
        "summary": {"batch_count": 99, "partition_count": 2},
    })
    assert progress["stage"] == "HUB batches"
    assert progress["local"]["completed"] == 2 and progress["local"]["total"] == 4
    assert progress["percent"] == 50
    assert progress["partitions"]["completed"] == 1
    assert "remote" not in progress


def test_hub_progress_counts_batches_within_one_partition(tmp_path):
    """Catch one large HUB partition reading 0 of 1 while its batches are already processed."""
    import pandas as pd

    from parcel_app.pipelines.parcel_stats.completion import run_progress

    partition = tmp_path / "partition_1"
    cubes = partition / "monthly_cubes" / "cubesignature0000"
    cubes.mkdir(parents=True)
    pd.DataFrame({"batch_number": [n for n in range(1, 5) for _ in range(3)]}).to_parquet(
        partition / "hub_item_manifest.parquet"
    )
    current, stale = "aaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbb"
    marker = partition / "satellite_partition_completion.json"
    marker.write_text(json.dumps({"status": "processing", "signature": current}))
    for name in (
        f"batch_00001_monthly_stats_{current}.parquet",
        f"batch_00001_monthly_stats_{current}_report.parquet",
        f"batch_00002_monthly_stats_{current}.parquet",
        f"batch_00002_monthly_stats_{current}_report.parquet",
        f"batch_00003_monthly_stats_{current}.parquet",  # report not yet committed
        f"batch_00004_monthly_stats_{stale}.parquet",  # earlier processing settings
        f"batch_00004_monthly_stats_{stale}_report.parquet",
        "batch_00003_monthly_cleaning_report.parquet",
    ):
        pd.DataFrame({"value": [1]}).to_parquet(cubes / name)
    run = {"source": "hub", "status": "running", "output_dir": str(tmp_path),
           "summary": {"batch_count": 1, "partition_count": 1}}
    progress = run_progress(run)
    assert progress["local"] == {"stage": "HUB batches", "completed": 2, "total": 4, "percent": 50}
    assert progress["stage"] == "HUB batches" and progress["percent"] == 50
    assert progress["partitions"]["completed"] == 0

    for n in (3, 4):
        for suffix in ("", "_report"):
            pd.DataFrame({"value": [1]}).to_parquet(cubes / f"batch_{n:05d}_monthly_stats_{current}{suffix}.parquet")
    assert run_progress(run)["stage"] == "Partition assembly"
    marker.write_text(json.dumps({"status": "complete", "signature": current}))
    assert run_progress(run)["stage"] == "Verifying and publishing results"


def test_hub_progress_before_catalog_resolution_is_not_invented(tmp_path):
    from parcel_app.pipelines.parcel_stats.completion import run_progress

    progress = run_progress({"source": "hub", "status": "running", "output_dir": str(tmp_path),
                             "summary": {"batch_count": 3, "partition_count": 1}})
    assert progress["stage"] == "Resolving HUB catalog" and progress["percent"] is None


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("KEYCLOAK_URL", "<https://auth.example.test/realms/hub/>"),
        ("CATALOG_URL", "<https://catalog.example.test>"),
        ("CATALOG_URL", "catalog.example.test"),
        ("CATALOG_URL", "ftp://catalog.example.test"),
        ("KEYCLOAK_URL", " https://auth.example.test/realms/hub/"),
    ],
)
def test_hub_rejects_malformed_service_urls(monkeypatch, name, value):
    """Catch pasted link markup reaching HUB login, which otherwise fails only after ten token retries."""
    from parcel_app.sources.hub.credentials import configured, require_configuration

    _configure_hub(monkeypatch, "internal")
    monkeypatch.setenv(name, value)
    assert not configured()
    with pytest.raises(ValueError, match=name):
        require_configuration()
    with pytest.raises(HTTPException) as error:
        routes.draft_accounts(_draft(), 7, None)
    assert error.value.status_code == 422
    assert name in error.value.detail
    # Operator environment values stay out of browser-visible errors.
    assert value.strip() not in error.value.detail
