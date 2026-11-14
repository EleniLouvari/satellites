"""Explicit remote-memory recovery remains inside the existing owned resume boundary."""

from uuid import uuid4

import pytest

from tests.test_parcel_app.test_routes import clients  # noqa: F401


@pytest.mark.parametrize("enabled", [False, True])
def test_owned_resume_persists_only_recovery_flag(clients, tmp_path, enabled):  # noqa: F811
    """Resume accepts only an explicit boolean and never overwrites the science snapshot."""
    from parcel_app import db
    from tests.test_parcel_app.test_deletion import seed_run

    owner, stranger = clients
    run, _, _, _ = seed_run(owner, tmp_path)
    identifier = str(run["run_id"])
    account = db.one(
        "INSERT INTO eo_users(app_user_id,username,password) VALUES(%s,%s,'test-only') RETURNING *",
        (run["app_user_id"], f"recovery_{uuid4().hex}"),
    )
    db.execute("INSERT INTO run_accounts(run_id,eo_user_id,position,username) VALUES(%s,%s,0,%s)",
               (identifier, account["eo_user_id"], account["username"]))
    db.execute("UPDATE analysis_runs SET status='failed',validated_revision=revision WHERE run_id=%s", (identifier,))
    original = db.one("SELECT config,processing FROM analysis_runs WHERE run_id=%s", (identifier,))
    link = f"/api/runs/{identifier}/resume"
    assert stranger.post(link, json={"retry_exhausted_memory": True}).status_code == 404
    assert owner.post(link, json={"retry_exhausted_memory": "true"}).status_code == 422
    assert owner.post(link, json={"retry_exhausted_memory": True, "processing": {}}).status_code == 422
    assert owner.post(link, json={"retry_exhausted_memory": enabled}, headers={"X-CSRF-Token": "bad"}).status_code == 403
    result = owner.post(link, json={"retry_exhausted_memory": enabled})
    assert result.status_code == 202, result.text
    job = db.one("SELECT * FROM web_jobs WHERE target_id=%s AND kind='resume'", (identifier,))
    assert job["payload"] == ({"retry_exhausted_memory": True} if enabled else {})
    # An idempotent second request cannot expand the queued claim's authorization.
    assert owner.post(link, json={"retry_exhausted_memory": not enabled}).status_code == 202
    assert db.one("SELECT payload FROM web_jobs WHERE job_id=%s", (str(job["job_id"]),))["payload"] == job["payload"]
    assert db.one("SELECT config,processing FROM analysis_runs WHERE run_id=%s", (identifier,)) == original
