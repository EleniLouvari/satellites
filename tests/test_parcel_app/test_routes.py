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

    monkeypatch.setenv("PARCEL_STORAGE_ROOT", str(tmp_path))
    monkeypatch.setenv("PARCEL_COOKIE_SECURE", "false")
    monkeypatch.setenv("PARCEL_ALLOWED_ORIGIN", "http://testserver")
    monkeypatch.setenv("SATELLITES_CREDENTIAL_KEY", Fernet.generate_key().decode())
    get_settings.cache_clear()
    result = []
    for _ in range(2):
        client = TestClient(create_app())
        client.headers["Origin"] = "http://testserver"
        response = client.post("/api/auth/register", json={"username": f"test_{uuid4().hex[:16]}", "password": "Test-password-2345"})
        assert response.status_code in {200, 201}, response.text
        client.headers["X-CSRF-Token"] = response.json()["csrf_token"]
        result.append(client)
    yield result
    for client in result:
        client.close()
    get_settings.cache_clear()


def test_unknown_processing_and_excess_workers_rejected():
    """Browser configuration cannot smuggle server paths or exceed worker limits."""
    from parcel_app.config import check_processing

    with pytest.raises(ValueError):
        check_processing({"start_date": "2024-01-01", "end_date": "2024-01-31", "output_dir": "/etc"})
    with pytest.raises(ValueError):
        check_processing({"start_date": "2024-01-01", "end_date": "2024-01-31", "batch_workers": 99999})


def test_public_config_hides_paths():
    """Only portable, non-secret configuration is returned to browsers."""
    from parcel_app.runs import public_run

    row = {"run_id": "x", "output_dir": "/private/run", "config": {"input": {"path": "/private/data", "parcel_id_field": "id"}, "output_dir": "/private/run", "processing": {}}, "summary": {"config": {"output_dir": "/private/run"}}}
    rendered = public_run(row)
    assert "/private" not in str(rendered)
    assert "output_dir" not in rendered


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
    db.execute("UPDATE parcel_uploads SET status='ready', metadata=%s WHERE upload_id=%s", (Json({"layers": [{"name": None, "columns": ["parcel_id"]}]}), upload["upload_id"]))
    account = owner.post("/api/accounts", json={"username": f"{uuid4().hex}@example.org", "password": "eo-example-password", "label": "Test"}).json()
    response = owner.post("/api/runs", json={"name": "Boundary test", "upload_id": upload["upload_id"], "layer": None, "parcel_id_field": "parcel_id", "processing": {"start_date": "2024-01-01", "end_date": "2024-01-31", "sentinel2_bands": ["B04"], "sentinel2_indices": []}, "account_ids": [account["eo_user_id"]]})
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
