"""Destructive operations are exercised only on disposable DB rows and temp files."""

from pathlib import Path
from unittest.mock import Mock
from uuid import uuid4

import pytest
from psycopg2.extras import Json

from tests.test_parcel_app.test_routes import clients as _clients_fixture

clients = _clients_fixture


def seed_run(client, root, upload=None):
    """Create an owned completed run with real temporary files and dependent records."""
    from parcel_app import db

    user_id = client.get("/api/auth/me").json()["app_user_id"]
    if upload is None:
        upload_id = str(uuid4())
        source = root / "users" / str(user_id) / "uploads" / upload_id / "source.geojson"
        source.parent.mkdir(parents=True)
        source.write_text("{}")
        upload = db.one(
            "INSERT INTO parcel_uploads(upload_id,app_user_id,filename,storage_path,sha256,size_bytes,status) "
            "VALUES(%s,%s,'source.geojson',%s,'test',2,'ready') RETURNING *",
            (upload_id, user_id, str(source)),
        )
    run_id = str(uuid4())
    output = root / "users" / str(user_id) / "runs" / run_id
    output.mkdir(parents=True)
    (output / "result.txt").write_text("result")
    run = db.one(
        "INSERT INTO analysis_runs(run_id,app_user_id,name,upload_id,parcel_id_field,processing,account_ids,output_dir,status) "
        "VALUES(%s,%s,'Delete test',%s,'id','{}','[]',%s,'complete') RETURNING *",
        (run_id, user_id, str(upload["upload_id"]), str(output)),
    )
    db.execute("INSERT INTO run_events(run_id,stage,message) VALUES(%s,'test','test')", (run_id,))
    db.execute(
        "INSERT INTO run_artifacts(artifact_id,run_id,kind,name,storage_path,size_bytes,sha256) "
        "VALUES(%s,%s,'test','result.txt',%s,6,'test')",
        (str(uuid4()), run_id, str(output / "result.txt")),
    )
    eda_id = str(uuid4())
    eda_output = root / "users" / str(user_id) / "eda" / eda_id
    eda_output.mkdir(parents=True)
    (eda_output / "report.html").write_text("report")
    db.execute(
        "INSERT INTO eda_runs(eda_id,run_id,app_user_id,selected_columns,config,input_sha256,output_dir,status) "
        "VALUES(%s,%s,%s,'[]','{}','test',%s,'complete')",
        (eda_id, run_id, user_id, str(eda_output)),
    )
    for kind, target in [("run", run_id), ("bundle", run_id), ("eda", eda_id), ("inspect", str(upload["upload_id"]))]:
        db.execute(
            "INSERT INTO web_jobs(job_id,app_user_id,kind,target_id,status) VALUES(%s,%s,%s,%s,'done')",
            (str(uuid4()), user_id, kind, target),
        )
    return run, upload, eda_id, eda_output


def test_delete_run_owner_csrf_cascade_and_shared_input(clients, tmp_path):
    from parcel_app import db

    owner, stranger = clients
    run, upload, eda_id, eda_output = seed_run(owner, tmp_path)
    other, _, _, other_eda = seed_run(owner, tmp_path, upload)
    link = f"/api/runs/{run['run_id']}"
    assert stranger.delete(link).status_code == 404
    assert owner.delete(link, headers={"X-CSRF-Token": "wrong"}).status_code == 403
    assert owner.delete(link).status_code == 200
    assert not Path(run["output_dir"]).exists() and not eda_output.exists()
    assert Path(upload["storage_path"]).exists() and other_eda.exists()
    for table in ("analysis_runs", "run_accounts", "run_events", "run_artifacts", "eda_runs"):
        assert not db.one(f"SELECT 1 FROM {table} WHERE run_id=%s", (str(run["run_id"]),))
    assert not db.one("SELECT 1 FROM web_jobs WHERE target_id=ANY(%s)", ([str(run["run_id"]), eda_id],))
    assert owner.delete(f"/api/runs/{other['run_id']}").status_code == 200
    assert not Path(upload["storage_path"]).parent.exists()
    assert not db.one("SELECT 1 FROM parcel_uploads WHERE upload_id=%s", (str(upload["upload_id"]),))


def test_delete_eda_keeps_parent_and_input(clients, tmp_path):
    owner, stranger = clients
    run, upload, eda_id, output = seed_run(owner, tmp_path)
    link = f"/api/eda/{eda_id}"
    assert stranger.delete(link).status_code == 404
    assert owner.delete(link, headers={"X-CSRF-Token": "wrong"}).status_code == 403
    assert owner.delete(link).status_code == 200
    assert not output.exists()
    assert Path(run["output_dir"]).exists() and Path(upload["storage_path"]).exists()
    assert owner.get(f"/api/runs/{run['run_id']}/eda").json() == []


@pytest.mark.parametrize("kind,status", [("run", "running"), ("bundle", "pending"), ("eda", "running")])
def test_delete_blocks_active_related_work(clients, tmp_path, kind, status):
    from parcel_app import db

    owner, _ = clients
    run, _, eda_id, output = seed_run(owner, tmp_path)
    target = eda_id if kind == "eda" else str(run["run_id"])
    db.execute("UPDATE web_jobs SET status=%s WHERE target_id=%s AND kind=%s", (status, target, kind))
    assert owner.delete(f"/api/runs/{run['run_id']}").status_code == 409
    assert output.exists() and Path(run["output_dir"]).exists()
    if kind == "eda":
        assert owner.delete(f"/api/eda/{eda_id}").status_code == 409


def test_delete_rejects_escaped_path_before_touching_files(clients, tmp_path):
    from parcel_app import db

    owner, _ = clients
    run, _, _, _ = seed_run(owner, tmp_path)
    protected = tmp_path / "protected"
    protected.mkdir()
    (protected / "keep").write_text("keep")
    db.execute("UPDATE analysis_runs SET output_dir=%s WHERE run_id=%s", (str(protected), str(run["run_id"])))
    assert owner.delete(f"/api/runs/{run['run_id']}").status_code == 409
    assert (protected / "keep").exists()


def test_remote_failure_preserves_files_and_retry_is_durable(clients, tmp_path, monkeypatch):
    from parcel_app import db, deletion

    owner, _ = clients
    run, _, _, output = seed_run(owner, tmp_path)
    manifest = {"jobs": [{"username": "test", "job_id": "one"}], "remote_deleted": 0, "remote_complete": False}
    db.execute("UPDATE analysis_runs SET deletion_manifest=%s WHERE run_id=%s", (Json(manifest), str(run["run_id"])))
    monkeypatch.setattr(deletion, "_credentials", lambda r: [("test", "password")])
    remote = Mock(side_effect=RuntimeError("private password=secret"))
    monkeypatch.setattr(deletion, "delete_jobs", remote)
    link = f"/api/runs/{run['run_id']}"
    response = owner.delete(link)
    assert response.status_code == 503 and "secret" not in response.text
    assert output.exists() and Path(run["output_dir"]).exists()
    row = owner.get(link).json()
    assert row["status"] == "deleting" and "deletion_manifest" not in row
    assert owner.post(f"{link}/resume").status_code == 409
    remote.side_effect = None
    assert owner.delete(link).status_code == 200


def test_filesystem_failure_is_retryable(clients, tmp_path, monkeypatch):
    from parcel_app import deletion

    owner, _ = clients
    run, _, _, _ = seed_run(owner, tmp_path)
    original = deletion.remove_directory
    monkeypatch.setattr(deletion, "remove_directory", Mock(side_effect=PermissionError("locked")))
    link = f"/api/runs/{run['run_id']}"
    assert owner.delete(link).status_code == 503
    assert owner.get(link).json()["status"] == "deleting"
    monkeypatch.setattr(deletion, "remove_directory", original)
    assert owner.delete(link).status_code == 200


def test_shared_upload_removed_when_other_run_is_deleted_during_remote_cleanup(clients, tmp_path, monkeypatch):
    """Recheck references at finalization, after a different deletion has finished."""
    from parcel_app import db, deletion

    owner, _ = clients
    first, upload, _, _ = seed_run(owner, tmp_path)
    second, _, _, _ = seed_run(owner, tmp_path, upload)
    manifest = {"jobs": [{"username": "test", "job_id": "one"}], "remote_deleted": 0, "remote_complete": False}
    db.execute("UPDATE analysis_runs SET deletion_manifest=%s WHERE run_id=%s", (Json(manifest), str(first["run_id"])))
    monkeypatch.setattr(deletion, "_credentials", lambda r: [("test", "password")])

    def interleaved(*args, **kwargs):
        assert owner.delete(f"/api/runs/{second['run_id']}").status_code == 200

    monkeypatch.setattr(deletion, "delete_jobs", interleaved)
    assert owner.delete(f"/api/runs/{first['run_id']}").status_code == 200
    assert not Path(upload["storage_path"]).exists()
    assert not db.one("SELECT 1 FROM parcel_uploads WHERE upload_id=%s", (str(upload["upload_id"]),))


def test_completed_run_with_missing_remote_records_is_not_deleted(clients, tmp_path):
    """An empty directory scan cannot prove that a submitted run has no remote jobs."""
    from parcel_app import db

    owner, _ = clients
    run, _, _, _ = seed_run(owner, tmp_path)
    account = owner.post("/api/accounts", json={"username": "cleanup-test", "password": "test-secret"}).json()
    db.execute(
        "INSERT INTO run_accounts(run_id,eo_user_id,position,username) VALUES(%s,%s,0,'cleanup-test')",
        (str(run["run_id"]), account["eo_user_id"]),
    )
    assert owner.delete(f"/api/runs/{run['run_id']}").status_code == 409
    assert Path(run["output_dir"]).exists()
