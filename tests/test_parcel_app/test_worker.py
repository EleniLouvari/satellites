"""Worker safety and artifact publication tests."""

import json
import pytest


def test_containment_rejects_escape(tmp_path):
    from parcel_app.artifacts import contained

    with pytest.raises(ValueError):
        contained(tmp_path, tmp_path.parent / "outside")


def test_redaction_removes_credentials_and_urls():
    from parcel_app.artifacts import sanitize_text

    value = sanitize_text("password=hunter2 https://user:secret@host?token=abcd hunter2", ["hunter2"])
    assert "hunter2" not in value
    assert "secret" not in value
    assert "abcd" not in value


def test_preview_has_no_attributes_and_is_bounded():
    import geopandas as gpd
    from shapely.geometry import Point
    from parcel_app.inspection import layer_metadata

    frame = gpd.GeoDataFrame({"private": ["secret"] * 600}, geometry=[Point(1, 2)] * 600, crs=4326)
    metadata = layer_metadata(frame, None)
    assert len(metadata["preview"]["features"]) <= 500
    assert all(feature["properties"] == {} for feature in metadata["preview"]["features"])
    assert len(json.dumps(metadata["preview"]).encode()) <= 1024 * 1024


def test_process_lock_excludes_second_holder(tmp_path):
    from parcel_app.worker import process_lock

    with process_lock(tmp_path):
        with pytest.raises(OSError):
            with process_lock(tmp_path):
                pass


@pytest.fixture
def queue_owner(monkeypatch):
    """Create an isolated queue owner only on an explicitly configured disposable DB."""
    import os
    from uuid import uuid4

    if os.getenv("PARCEL_TEST_DB") != "1":
        pytest.skip("Disposable PostgreSQL opt-in required")
    from parcel_app import db

    owner = db.one(
        "INSERT INTO app_users(username,password,is_active) VALUES(%s,'test-only',true) RETURNING *", (f"worker_{uuid4().hex}",)
    )
    yield owner
    db.execute("DELETE FROM web_jobs WHERE app_user_id=%s", (owner["app_user_id"],))
    db.execute("DELETE FROM app_users WHERE app_user_id=%s", (owner["app_user_id"],))


def test_queue_idempotence_claim_and_fencing(queue_owner):
    """A stale child cannot finish a replaced claim or requeue duplicate work."""
    from parcel_app import db
    from parcel_app.worker import enqueue
    from parcel_app.tasks import _owned
    from uuid import uuid4

    target = str(uuid4())
    job = enqueue("inspect", target, queue_owner["app_user_id"])
    assert enqueue("inspect", target, queue_owner["app_user_id"])["job_id"] == job["job_id"]
    # Claim directly by own ID so concurrent test users do not affect ordering.
    token = str(uuid4())
    claimed = db.one(
        "UPDATE web_jobs SET status='running',claim_token=%s WHERE job_id=%s RETURNING *", (token, str(job["job_id"]))
    )
    with db.connection() as conn:
        _owned(claimed, conn)
    stale = dict(claimed, claim_token=str(uuid4()))
    with pytest.raises(RuntimeError, match="claim"):
        with db.connection() as conn:
            _owned(stale, conn)


def test_unverified_manifest_cannot_publish(tmp_path):
    """A present output path is insufficient without verified final runner state."""
    from parcel_app.artifacts import register_artifacts

    (tmp_path / "artifacts.json").write_text(json.dumps({"status": "running"}))
    (tmp_path / "run.json").write_text(json.dumps({"status": "running"}))
    with pytest.raises(ValueError, match="completion"):
        register_artifacts({"output_dir": str(tmp_path)})


def test_inspection_task_persists_owned_metadata(queue_owner, tmp_path, monkeypatch):
    """A real queued inspection completes with no original properties in preview."""
    from uuid import uuid4
    import geopandas as gpd
    from shapely.geometry import Polygon
    from parcel_app import db
    from parcel_app.settings import get_settings
    from parcel_app.artifacts import checksum
    from parcel_app.tasks import execute_task
    from parcel_app.worker import enqueue

    monkeypatch.setenv("PARCEL_STORAGE_ROOT", str(tmp_path))
    get_settings.cache_clear()
    path = tmp_path / "parcels.geojson"
    gpd.GeoDataFrame({"id": [1], "secret": ["private"]}, geometry=[Polygon([(1, 1), (2, 1), (2, 2), (1, 1)])], crs=4326).to_file(
        path, driver="GeoJSON"
    )
    upload = str(uuid4())
    db.execute(
        """INSERT INTO parcel_uploads(upload_id,app_user_id,filename,storage_path,sha256,size_bytes,status)
        VALUES(%s,%s,'parcels.geojson',%s,%s,%s,'pending')""",
        (upload, queue_owner["app_user_id"], str(path), checksum(path), path.stat().st_size),
    )
    try:
        job = enqueue("inspect", upload, queue_owner["app_user_id"])
        job = db.one(
            "UPDATE web_jobs SET status='running',claim_token=%s WHERE job_id=%s RETURNING *", (str(uuid4()), str(job["job_id"]))
        )
        execute_task(job)
        result = db.one("SELECT * FROM parcel_uploads WHERE upload_id=%s", (upload,))
        assert result["status"] == "ready"
        assert result["metadata"]["layers"][0]["parcel_count"] == 1
        assert "private" not in json.dumps(result["metadata"])
        assert db.one("SELECT status FROM web_jobs WHERE job_id=%s", (str(job["job_id"]),))["status"] == "done"
    finally:
        db.execute("DELETE FROM parcel_uploads WHERE upload_id=%s", (upload,))
        get_settings.cache_clear()


def test_failed_diagnostics_redact_password_and_private_paths(tmp_path, monkeypatch):
    """Failed runs may publish safe diagnostics without claiming final results."""
    from parcel_app import artifacts

    (tmp_path / "satellite_acquisition.log").write_text(f"password=actual-secret {tmp_path} https://host?token=hidden\n")
    registered = []
    monkeypatch.setattr(artifacts, "register_file", lambda run, kind, path, **kwargs: registered.append((kind, path)))
    artifacts.register_diagnostics(
        {"output_dir": str(tmp_path), "config": {"input": {"path": "private-input"}}}, ["actual-secret"]
    )
    text = "".join(path.read_text() for _, path in registered)
    assert "actual-secret" not in text
    assert str(tmp_path) not in text
    assert "private-input" not in text
    assert "hidden" not in text
    assert all(kind in {"log", "configuration"} for kind, _ in registered)
