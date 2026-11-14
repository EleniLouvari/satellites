"""Exercise web boundaries against a deliberately opted-in disposable database."""

import os
from pathlib import Path
from uuid import uuid4

import pytest


@pytest.fixture
def clients(monkeypatch, tmp_path):
    """Create independent authenticated users without touching a production DB."""
    if os.environ.get("PARCEL_TEST_DB") != "1":
        pytest.skip("Set PARCEL_TEST_DB=1 with disposable PostgreSQL credentials")
    from cryptography.fernet import Fernet
    from fastapi.testclient import TestClient
    from parcel_app.app import create_app
    from parcel_app.settings import get_settings
    from parcel_app.security import cipher

    monkeypatch.setenv("PARCEL_STORAGE_ROOT", str(tmp_path))
    monkeypatch.setenv("PARCEL_COOKIE_SECURE", "false")
    monkeypatch.setenv("PARCEL_ALLOWED_ORIGIN", "http://testserver")
    monkeypatch.setenv("SATELLITES_CREDENTIAL_KEY", Fernet.generate_key().decode())
    get_settings.cache_clear()
    cipher.cache_clear()
    result = []
    for _ in range(2):
        client = TestClient(create_app())
        client.headers["Origin"] = "http://testserver"
        response = client.post(
            "/api/auth/register", json={"username": f"test_{uuid4().hex[:16]}", "password": "Test-password-2345"}
        )
        assert response.status_code in {200, 201}, response.text
        client.headers["X-CSRF-Token"] = response.json()["csrf_token"]
        result.append(client)
    yield result
    for client in result:
        client.close()
    get_settings.cache_clear()
    cipher.cache_clear()


def test_unknown_processing_and_excess_workers_rejected():
    """Browser configuration cannot smuggle server paths or exceed worker limits."""
    from parcel_app.config import check_processing

    with pytest.raises(ValueError):
        check_processing({"start_date": "2024-01-01", "end_date": "2024-01-31", "output_dir": "/etc"})
    with pytest.raises(ValueError):
        check_processing({"start_date": "2024-01-01", "end_date": "2024-01-31", "batch_workers": 99999})


def test_temporal_fill_block_rows_web_configuration():
    """Expose optional numeric tuning and preserve it only when filling is enabled."""
    from parcel_app.config import check_processing, configuration

    fields = {field["name"]: field for field in configuration(user={})["fields"]}
    field = fields["temporal_fill_block_rows"]
    assert field["type"] == "number"
    assert field["nullable"] is True
    assert field["group"] == "filling"
    options = {"start_date": "2024-01-01", "end_date": "2024-01-31", "fill_nulls": True, "temporal_fill_block_rows": 256}
    assert check_processing(options)["temporal_fill_block_rows"] == 256
    assert check_processing({**options, "temporal_fill_block_rows": None})["temporal_fill_block_rows"] is None
    assert "temporal_fill_block_rows" not in check_processing({**options, "fill_nulls": False})
    for value in (0, -1, True, 1.5, "256"):
        with pytest.raises((TypeError, ValueError), match="temporal_fill_block_rows"):
            check_processing({**options, "temporal_fill_block_rows": value})


def test_public_config_hides_paths():
    """Only portable, non-secret configuration is returned to browsers."""
    from parcel_app.runs import public_run

    row = {
        "run_id": "x",
        "output_dir": "/private/run",
        "config": {"input": {"path": "/private/data", "parcel_id_field": "id"}, "output_dir": "/private/run", "processing": {}},
        "summary": {"config": {"output_dir": "/private/run"}},
    }
    rendered = public_run(row)
    assert "/private" not in str(rendered)
    assert "output_dir" not in rendered


def test_duration_uses_processing_start_and_freezes_at_completion():
    """Queue time is excluded and records without a start do not invent durations."""
    from datetime import datetime, timedelta, timezone
    from parcel_app.runs import public_run

    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    result = public_run({"started_at": start, "updated_at": start + timedelta(seconds=125), "status": "complete"})
    assert result["duration_seconds"] == 125
    assert result["finished_at"] == start + timedelta(seconds=125)
    assert public_run({"status": "draft"})["duration_seconds"] is None


def test_form_covers_runner_options_and_nullable_values():
    """A new engine option must remain editable without manually updating UI fields."""
    from data_preparation.parcel_stats.run_config import _processing_parameters
    from parcel_app.config import configuration, check_processing

    schema = configuration(user={})
    fields = {field["name"]: field for field in schema["fields"]}
    assert set(fields) == set(_processing_parameters()) == set(schema["defaults"])
    assert fields["working_epsg"]["type"] == "number"
    assert fields["working_epsg"]["nullable"]
    options = {**schema["defaults"], "start_date": "2024-01-01", "end_date": "2024-02-01"}
    assert "working_epsg" not in check_processing(options)


def test_grouped_configuration_dependencies_and_defaults():
    """Every current option has a panel and scientifically accurate dependencies."""
    from parcel_app.config import configuration

    schema = configuration(user={})
    fields = {field["name"]: field for field in schema["fields"]}
    assert schema["defaults"]["temporal_fill_mode"] == "bidirectional"
    assert schema["defaults"]["temporal_reducer"] == "none"
    assert len(schema["groups"]) == 7
    assert all(field["group"] in {group["id"] for group in schema["groups"]} for field in fields.values())
    assert fields["iqr_quantiles"]["enabled_when"] == {"remove_outliers": [True]}
    assert fields["temporal_fill_mode"]["enabled_when"] == {"fill_nulls": [True]}
    assert fields["interpolation_max_distance_in_meters"]["enabled_when"] == {
        "fill_nulls": [True],
        "interpolation_method": ["idw"],
    }
    assert fields["interpolation_variogram_lags"]["enabled_when"] == {"fill_nulls": [True], "interpolation_method": ["kriging"]}
    assert not fields["temporal_period"]["enabled_when"]


def test_inactive_settings_do_not_block_web_validation():
    """Inactive IDW and unfinished IQR inputs must not invalidate an unrelated run."""
    from parcel_app.config import check_processing

    options = check_processing(
        {
            "start_date": "2024-01-01",
            "end_date": "2024-02-01",
            "fill_nulls": False,
            "interpolation_method": "idw",
            "interpolation_max_distance_in_meters": None,
            "remove_outliers": False,
            "iqr_quantiles": [0.9, 0.1],
            "iqr_min_valid_pixels": None,
        }
    )
    assert "interpolation_method" not in options
    assert "iqr_quantiles" not in options
    assert "iqr_min_valid_pixels" not in options


def test_upload_rejects_traversal_and_isolates_owner(clients):
    """A foreign user cannot discover uploaded metadata or write outside storage."""
    owner, stranger = clients
    bad = owner.post("/api/uploads?filename=../parcels.geojson", content=b"{}")
    assert bad.status_code == 422
    response = owner.post("/api/uploads?filename=parcels.geojson", content=b'{"type":"FeatureCollection","features":[]}')
    assert response.status_code == 202, response.text
    upload = response.json()
    assert "storage_path" not in upload
    assert stranger.get(f"/api/uploads/{upload['upload_id']}").status_code == 404


def test_oversized_stream_does_not_leave_upload(clients, monkeypatch):
    """Reject actual streamed size, not just a declared Content-Length."""
    from parcel_app.settings import get_settings

    owner, _ = clients
    monkeypatch.setenv("PARCEL_UPLOAD_LIMIT_MB", "1")
    get_settings.cache_clear()
    response = owner.post("/api/uploads?filename=huge.geojson", content=b"x" * (1024 * 1024 + 1))
    assert response.status_code == 413
    assert not any(Path(get_settings().storage_root).rglob("source.geojson"))


def test_run_owner_and_submit_revision(clients):
    """A run is private; stale or unvalidated submissions do not start jobs."""
    from parcel_app import db
    from psycopg2.extras import Json

    owner, stranger = clients
    upload = owner.post("/api/uploads?filename=parcels.geojson", content=b"{}").json()
    db.execute(
        "UPDATE parcel_uploads SET status='ready', metadata=%s WHERE upload_id=%s",
        (Json({"layers": [{"name": None, "columns": ["parcel_id"]}]}), upload["upload_id"]),
    )
    account = owner.post(
        "/api/accounts", json={"username": f"{uuid4().hex}@example.org", "password": "eo-example-password", "label": "Test"}
    ).json()
    response = owner.post(
        "/api/runs",
        json={
            "name": "Boundary test",
            "upload_id": upload["upload_id"],
            "layer": None,
            "parcel_id_field": "parcel_id",
            "processing": {
                "start_date": "2024-01-01",
                "end_date": "2024-01-31",
                "sentinel2_bands": ["B04"],
                "sentinel2_indices": [],
            },
            "account_ids": [account["eo_user_id"]],
        },
    )
    assert response.status_code == 201, response.text
    run = response.json()
    run_id = run["run_id"]
    assert stranger.get(f"/api/runs/{run_id}").status_code == 404
    assert stranger.post(f"/api/runs/{run_id}/validate").status_code == 404
    assert owner.post(f"/api/runs/{run_id}/submit", json={"revision": 1}).status_code == 409
    db.execute("UPDATE analysis_runs SET status='ready', validated_revision=revision WHERE run_id=%s", (run_id,))
    assert owner.post(f"/api/runs/{run_id}/submit", json={"revision": 99}).status_code == 409
    assert owner.post(f"/api/runs/{run_id}/submit", json={"revision": 1}).status_code == 202
    assert owner.post(f"/api/runs/{run_id}/submit", json={"revision": 1}).status_code == 202
    row = db.one("SELECT COUNT(*) AS n FROM web_jobs WHERE kind='run' AND target_id=%s", (run_id,))
    assert row["n"] == 1
    assert owner.patch(f"/api/runs/{run_id}", json={}).status_code in {409, 422}
    assert stranger.get(f"/api/runs/{run_id}/artifacts").status_code == 404
    assert owner.post(f"/api/runs/{run_id}/cancel").status_code == 200


def test_artifact_path_escape_blocked(clients):
    """Even a corrupted database artifact cannot expose a path outside its run."""
    from parcel_app.files import contained_artifact

    with pytest.raises(ValueError):
        contained_artifact({"output_dir": "/safe/run"}, {"storage_path": "/etc/passwd"})


def test_full_pipeline_queue_download_and_resume(clients, tmp_path, monkeypatch):
    """Use real local statistics and checkpoints with only remote downloads stubbed."""
    import io
    import zipfile
    import geopandas as gpd
    import numpy as np
    import pandas as pd
    import xarray as xr
    from shapely.geometry import box
    from parcel_app import db
    from parcel_app.tasks import execute_task

    owner, stranger = clients
    path = tmp_path / "original.gpkg"
    gpd.GeoDataFrame({"parcel_id": ["a"]}, geometry=[box(0, 0, 100, 100)], crs=3857).to_file(path, layer="parcels", driver="GPKG")

    def execute(kind, target):
        job = db.one(
            "UPDATE web_jobs SET status='running',claim_token=%s WHERE kind=%s AND target_id=%s AND status='pending' RETURNING *",
            (str(uuid4()), kind, str(target)),
        )
        assert job is not None
        execute_task(job)
        assert db.one("SELECT status,error FROM web_jobs WHERE job_id=%s", (str(job["job_id"]),))["status"] == "done"

    upload = owner.post("/api/uploads?filename=parcels.gpkg", content=path.read_bytes()).json()
    execute("inspect", upload["upload_id"])
    meta = owner.get(f"/api/uploads/{upload['upload_id']}").json()
    assert meta["metadata"]["layers"][0]["name"] == "parcels"
    account = owner.post(
        "/api/accounts", json={"username": f"{uuid4().hex}@example.org", "password": "private-eo-secret", "label": "Offline"}
    ).json()
    response = owner.post(
        "/api/runs",
        json={
            "name": "Synthetic pipeline",
            "upload_id": upload["upload_id"],
            "layer": "parcels",
            "parcel_id_field": "parcel_id",
            "processing": {
                "start_date": "2024-01-01",
                "end_date": "2024-01-31",
                "working_epsg": 3857,
                "sentinel2_bands": ["B02"],
                "sentinel2_indices": [],
                "spatial_statistics": ["mean", "median"],
                "fit_tiles_to_parcels": True,
                "tile_size_metres": 1000,
                "tile_buffer_metres": 0,
                "batch_workers": 1,
                "remove_nc_after_completion": True,
            },
            "account_ids": [account["eo_user_id"]],
            "scheduling": "batches",
        },
    )
    assert response.status_code == 201, response.text
    run_id = response.json()["run_id"]
    assert owner.post(f"/api/runs/{run_id}/validate").status_code == 202
    execute("validate", run_id)
    assert owner.get(f"/api/runs/{run_id}").json()["status"] == "ready"
    assert owner.post(f"/api/runs/{run_id}/submit", json={"revision": 1}).status_code == 202

    def synthetic_download(shared, pinned, users, jobs_per_user, *, batch_workers):
        assert batch_workers == 1
        while not shared.empty():
            task = shared.get_nowait()
            cube = xr.Dataset(
                {"B02": (("t", "y", "x"), np.full((1, 10, 10), 0.25, dtype="float32"))},
                coords={"t": pd.to_datetime(["2024-01-01"]), "x": np.arange(10) * 10 + 5, "y": 95 - np.arange(10) * 10},
            ).rio.write_crs(3857)
            cube.to_netcdf(task.plan["target_path"])
            cube.close()

    monkeypatch.setattr("openeo.connect", lambda *a, **k: pytest.fail("No remote connection allowed"))
    monkeypatch.setattr("data_preparation.parcel_stats.batch_scheduler._drain_batches", synthetic_download)
    execute("run", run_id)
    result = owner.get(f"/api/runs/{run_id}").json()
    assert result["status"] == "complete"
    assert result["started_at"]
    assert result["finished_at"]
    assert result["duration_seconds"] >= 0
    assert (
        next(r for r in owner.get("/api/runs").json() if r["run_id"] == run_id)["duration_seconds"] == result["duration_seconds"]
    )
    assert "output_dir" not in result
    files = owner.get(f"/api/runs/{run_id}/artifacts").json()
    stats = next(f for f in files if f["kind"] == "time_statistics")
    url = f"/api/artifacts/{stats['artifact_id']}/download"
    assert stranger.get(url).status_code == 404
    frame = gpd.read_parquet(io.BytesIO(owner.get(url).content))
    assert frame["B02_mean__20240101"].iloc[0] == 0.25
    assert "private-eo-secret" not in owner.get(f"/api/runs/{run_id}/events").text
    assert stranger.get(f"/api/runs/{run_id}/bundle").status_code == 404
    assert owner.get(f"/api/runs/{run_id}/bundle").json()["status"] == "unrequested"
    assert owner.post(f"/api/runs/{run_id}/bundle").status_code == 202
    assert owner.get(f"/api/runs/{run_id}/bundle").json()["status"] == "pending"
    execute("bundle", run_id)
    zipped = next(f for f in owner.get(f"/api/runs/{run_id}/artifacts").json() if f["kind"] == "bundle")
    archive = zipfile.ZipFile(io.BytesIO(owner.get(f"/api/artifacts/{zipped['artifact_id']}/download").content))
    assert owner.get(f"/api/runs/{run_id}/bundle").json()["status"] == "ready"
    assert owner.post(f"/api/runs/{run_id}/bundle").json()["artifact_id"] == zipped["artifact_id"]
    assert set(archive.namelist()) == {f"{f['kind']}/{f['name']}" for f in files}
    assert all(not name.endswith(".nc") for name in archive.namelist())
    # A new account must not change the saved selection or resume ordering.
    owner.post("/api/accounts", json={"username": f"{uuid4().hex}@example.org", "password": "another-secret", "label": "Later"})
    db.execute("UPDATE analysis_runs SET status='interrupted' WHERE run_id=%s", (run_id,))
    assert owner.post(f"/api/runs/{run_id}/resume").status_code == 202
    monkeypatch.setattr(
        "data_preparation.parcel_stats.batch_scheduler._drain_batches",
        lambda *a, **k: pytest.fail("Resume must reuse completion"),
    )
    execute("resume", run_id)
    assert owner.get(f"/api/runs/{run_id}").json()["status"] == "complete"


def test_openeo_parallel_jobs_form_and_server_limit():
    """The form offers only supported per-account slots and the server enforces the cap."""
    from parcel_app.config import check_processing, configuration

    schema = configuration(user={})
    field = next(field for field in schema["fields"] if field["name"] == "openeo_parallel_jobs")
    assert field["choices"] == [1, 2]
    assert field["default"] == 2
    options = {"start_date": "2024-01-01", "end_date": "2024-01-31"}
    for slots in (1, 2):
        assert check_processing({**options, "openeo_parallel_jobs": slots})["openeo_parallel_jobs"] == slots
    for slots in (0, 3, 999, True, 2.5):
        with pytest.raises(ValueError, match="openeo_parallel_jobs"):
            check_processing({**options, "openeo_parallel_jobs": slots})


def test_eda_submission_queue_isolation_and_download(clients, tmp_path):
    """Run real EDA via the queue against an owned completed result and verify access."""
    import io
    from zipfile import ZipFile
    import pandas as pd
    from parcel_app import db
    from parcel_app.artifacts import checksum, register_file
    from parcel_app.tasks import execute_task

    owner, stranger = clients
    user_id = owner.get("/api/auth/me").json()["app_user_id"]
    upload = owner.post("/api/uploads?filename=original.geojson", content=b"{}").json()
    output = tmp_path / "completed"
    output.mkdir()
    source = output / "result.parquet"
    pd.DataFrame({"parcel_id": range(20), "value": [i * 0.1 for i in range(20)], "excluded": ["private"] * 20}).to_parquet(source)
    source_hash = checksum(source)
    run_id = str(uuid4())
    run = db.one(
        "INSERT INTO analysis_runs(run_id,app_user_id,name,upload_id,parcel_id_field,processing,account_ids,output_dir,status) "
        "VALUES(%s,%s,'EDA source',%s,'parcel_id','{}','[]',%s,'running') RETURNING *",
        (run_id, user_id, upload["upload_id"], str(output)),
    )
    register_file(run, "time_statistics", source)
    assert owner.get(f"/api/runs/{run_id}/eda/options").status_code == 409
    db.execute("UPDATE analysis_runs SET status='complete' WHERE run_id=%s", (run_id,))
    assert stranger.get(f"/api/runs/{run_id}/eda/options").status_code == 404
    options = owner.get(f"/api/runs/{run_id}/eda/options").json()
    assert {c["name"] for c in options["columns"]} == {"parcel_id", "value", "excluded"}
    body = {"request_id": str(uuid4()), "columns": ["parcel_id", "value"], "config": {"id_column": "parcel_id"}}
    assert stranger.post(f"/api/runs/{run_id}/eda", json=body).status_code == 404
    response = owner.post(f"/api/runs/{run_id}/eda", json=body)
    assert response.status_code == 202, response.text
    eda_id = response.json()["eda_id"]
    report_link = f"/api/eda/{eda_id}/report"
    assert owner.get(report_link).status_code == 409
    assert stranger.get(report_link).status_code == 404
    assert stranger.get(f"/api/eda/{eda_id}/log").status_code == 404
    assert owner.post(f"/api/runs/{run_id}/eda", json=body).json()["eda_id"] == eda_id
    assert owner.post(f"/api/runs/{run_id}/eda", json={**body, "columns": ["value"]}).status_code == 409
    assert db.one("SELECT count(*) AS n FROM web_jobs WHERE target_id=%s AND kind='eda'", (eda_id,))["n"] == 1
    job = db.one(
        "UPDATE web_jobs SET status='running',claim_token=%s WHERE target_id=%s AND kind='eda' RETURNING *",
        (str(uuid4()), eda_id),
    )
    execute_task(job)
    row = owner.get(f"/api/runs/{run_id}/eda").json()[0]
    assert row["status"] == "complete", row
    assert row["progress"]["percent"] == 100
    report = owner.get(report_link)
    assert report.status_code == 200
    assert "text/html" in report.headers["content-type"]
    assert "sandbox allow-scripts;" in report.headers["content-security-policy"]
    assert "allow-same-origin" not in report.headers["content-security-policy"]
    assert "data:image/png;base64," in report.text
    assert "src='plots/" not in report.text
    assert owner.get(f"/api/eda/{eda_id}/log").status_code == 200
    assert "output_dir" not in row and "app_user_id" not in row
    assert checksum(source) == source_hash
    assert owner.get(f"/api/runs/{run_id}").json()["status"] == "complete"
    zip_artifact = next(file for file in row["artifacts"] if file["name"] == "eda-results.zip")
    link = f"/api/eda/{eda_id}/files/{zip_artifact['artifact_id']}"
    assert stranger.get(link).status_code == 404
    downloaded = owner.get(link)
    assert downloaded.status_code == 200
    assert "attachment" in downloaded.headers["content-disposition"]
    with ZipFile(io.BytesIO(downloaded.content)) as archive:
        assert "report.html" in archive.namelist()
        assert any(name.startswith("plots/") for name in archive.namelist())
        annotated = pd.read_parquet(io.BytesIO(archive.read("eda_annotated_data.parquet")))
        assert "excluded" not in annotated.columns

    body["request_id"] = str(uuid4())
    second = owner.post(f"/api/runs/{run_id}/eda", json=body).json()
    source.write_bytes(b"changed-after-submission")
    second_job = db.one(
        "UPDATE web_jobs SET status='running',claim_token=%s WHERE target_id=%s AND kind='eda' RETURNING *",
        (str(uuid4()), second["eda_id"]),
    )
    execute_task(second_job)
    assert db.one("SELECT status FROM eda_runs WHERE eda_id=%s", (second["eda_id"],))["status"] == "failed"
    assert owner.get(link).status_code == 200  # Earlier EDA outputs survive a later failed analysis.
