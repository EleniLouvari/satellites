"""Stop requests preserve ownership, worker isolation and completed checkpoints."""

from contextlib import nullcontext
from uuid import uuid4

import pytest
from fastapi import HTTPException
from tests.test_parcel_app.test_routes import clients  # noqa: F401


def test_stop_request_is_owned_idempotent_and_keeps_claim(monkeypatch):
    """Persist a request without releasing the active claim or account reservations."""
    from parcel_app import runs

    run_id = uuid4()
    run = {"run_id": run_id, "status": "running"}
    job = {"job_id": uuid4(), "status": "running", "payload": {}}
    writes = []
    monkeypatch.setattr(runs.db, "connection", lambda: nullcontext(object()))
    monkeypatch.setattr(runs.db, "one", lambda *a, **k: job)
    monkeypatch.setattr(runs, "public_run", lambda row: row)

    def owned(identifier, owner, **kwargs):
        """Require owner resolution before any stop write."""
        assert identifier == run_id and owner == 42
        return run.copy()

    monkeypatch.setattr(runs, "owned_run", owned)
    monkeypatch.setattr(runs.db, "execute", lambda sql, params, **kw: writes.append(sql))
    result = runs.stop_run(run_id, {"app_user_id": 42})
    assert result["stop_requested"] is True
    assert result["status"] == "running"
    assert any("payload" in sql for sql in writes)
    assert all("status=" not in sql and "run_accounts" not in sql for sql in writes)
    job["payload"] = {"stop_requested": True}
    writes.clear()
    assert runs.stop_run(run_id, {"app_user_id": 42})["stop_requested"] is True
    assert not writes


@pytest.mark.parametrize("status", ["draft", "ready", "queued", "complete", "failed", "cancelled", "deleting"])
def test_stop_rejects_nonrunning_runs(monkeypatch, status):
    """Stop cannot mutate queued or terminal analyses."""
    from parcel_app import runs

    monkeypatch.setattr(runs.db, "connection", lambda: nullcontext(object()))
    monkeypatch.setattr(runs, "owned_run", lambda *a, **k: {"status": status})
    monkeypatch.setattr(runs.db, "one", lambda *a, **k: None)
    monkeypatch.setattr(runs.db, "execute", lambda *a, **k: pytest.fail("Unexpected write"))
    with pytest.raises(HTTPException) as error:
        runs.stop_run(uuid4(), {"app_user_id": 42})
    assert error.value.status_code == 409


def test_stop_cannot_access_other_owner(monkeypatch):
    """Unknown and foreign analyses fail before queue access."""
    from parcel_app import runs

    monkeypatch.setattr(runs.db, "connection", lambda: nullcontext(object()))

    def unavailable(*args, **kwargs):
        """Simulate the existing ownership boundary."""
        raise HTTPException(404, "Error: Analysis not found.")

    monkeypatch.setattr(runs, "owned_run", unavailable)
    monkeypatch.setattr(runs.db, "one", lambda *a, **k: pytest.fail("Foreign queue access"))
    with pytest.raises(HTTPException) as error:
        runs.stop_run(uuid4(), {"app_user_id": 42})
    assert error.value.status_code == 404


def test_stop_request_fences_task_publication(monkeypatch):
    """A stopped task cannot publish success or overwrite the requested interruption."""
    from parcel_app import tasks

    job = {"job_id": uuid4(), "claim_token": uuid4(), "kind": "run"}
    row = {**job, "status": "running", "payload": {"stop_requested": True}}
    monkeypatch.setattr(tasks.db, "one", lambda *a, **k: row)
    with pytest.raises(RuntimeError, match="stop"):
        tasks._owned(job, object())


def test_stop_poll_is_scoped_to_exact_run_claim(monkeypatch):
    """Only the currently supervised run/resume claim can request termination."""
    from parcel_app import db, worker

    job = {"job_id": uuid4(), "claim_token": uuid4(), "kind": "resume"}

    def query(sql, params):
        """Verify both identity and claim fencing in the polling query."""
        assert params == (str(job["job_id"]), str(job["claim_token"]))
        assert "status='running'" in sql
        return {"payload": {"stop_requested": True}}

    monkeypatch.setattr(db, "one", query)
    assert worker.stop_requested(job)
    assert not worker.stop_requested({**job, "kind": "eda"})


def _synthetic_analysis(root, ready, group_name=None, exit_early=False):
    """Simulate a task with one raster subprocess and a committed checkpoint."""
    import subprocess
    import sys
    import time
    from pathlib import Path
    from parcel_app.process_control import isolate_task

    isolate_task(group_name)
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    Path(root, "child.pid").write_text(str(child.pid))
    Path(root, "completed.parquet").write_bytes(b"committed checkpoint")
    ready.set()
    if exit_early:
        return
    time.sleep(120)


@pytest.mark.parametrize("exit_early", [False, True])
def test_stop_terminates_descendants_but_preserves_other_processes_and_files(tmp_path, exit_early):
    """Exercise real process termination without touching any production worker."""
    import multiprocessing
    import subprocess
    import sys
    import psutil
    from parcel_app.process_control import stop_task, task_group

    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    sibling = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    try:
        with task_group() as group:
            child = context.Process(
                target=_synthetic_analysis,
                args=(str(tmp_path), ready, group.name if group else None, exit_early),
            )
            child.start()
            try:
                assert ready.wait(20)
                descendant = psutil.Process(int((tmp_path / "child.pid").read_text()))
                if exit_early:
                    child.join(timeout=20)
                    assert not child.is_alive()
                    assert descendant.is_running()
                stop_task(child, group)
                assert not child.is_alive()
                try:
                    assert not descendant.is_running() or descendant.status() == psutil.STATUS_ZOMBIE
                except psutil.NoSuchProcess:
                    pass
                assert sibling.poll() is None
                assert (tmp_path / "completed.parquet").read_bytes() == b"committed checkpoint"
            finally:
                if ready.is_set():
                    stop_task(child, group)
                elif child.is_alive():
                    child.kill()
                    child.join(timeout=10)
    finally:
        sibling.kill()
        sibling.wait(timeout=10)


def test_supervisor_cleans_descendants_when_leader_already_exited(monkeypatch):
    """An exit before the first polling iteration must still stop orphaned work."""
    from types import SimpleNamespace
    from parcel_app import process_control, worker

    child = SimpleNamespace(is_alive=lambda: False)
    ready = SimpleNamespace(is_set=lambda: True)
    group = object()
    stopped = []
    monkeypatch.setattr(process_control, "stop_task", lambda *args: stopped.append(args))
    worker.supervise_task(child, ready, group, {}, None)
    assert stopped == [(child, group)]


def test_reconciliation_locks_queue_before_analysis_rows(monkeypatch):
    """Recovery uses the same lock order as stop requests and task publication."""
    from parcel_app import db, worker

    statements = []
    monkeypatch.setattr(db, "connection", lambda: nullcontext(object()))
    monkeypatch.setattr(db, "all_rows", lambda sql, *a, **kw: statements.append(sql))
    monkeypatch.setattr(db, "execute", lambda sql, *a, **kw: statements.append(sql))
    worker.reconcile("run")
    assert "web_jobs" in statements[0] and "FOR UPDATE" in statements[0]
    assert any("UPDATE analysis_runs" in sql for sql in statements[1:])


def test_stop_api_ownership_csrf_reconciliation_and_resume(clients, tmp_path):  # noqa: F811
    """Exercise the actual PostgreSQL stop/resume lifecycle without remote jobs."""
    from parcel_app import db
    from parcel_app.tasks import _owned
    from parcel_app.worker import enqueue, reconcile, stop_requested
    from tests.test_parcel_app.test_deletion import seed_run

    owner, stranger = clients
    run, _, _, _ = seed_run(owner, tmp_path)
    run_id = str(run["run_id"])
    account = db.one(
        "INSERT INTO eo_users(app_user_id,username,password) VALUES(%s,%s,'test-only') RETURNING *",
        (run["app_user_id"], f"stop_{uuid4().hex}"),
    )
    db.execute(
        "INSERT INTO run_accounts(run_id,eo_user_id,position,username) VALUES(%s,%s,0,%s)",
        (run_id, account["eo_user_id"], account["username"]),
    )
    db.execute("UPDATE analysis_runs SET status='running',validated_revision=revision WHERE run_id=%s", (run_id,))
    job = enqueue("run", run_id, run["app_user_id"])
    job = db.one(
        "UPDATE web_jobs SET status='running',claim_token=%s WHERE job_id=%s RETURNING *",
        (str(uuid4()), str(job["job_id"])),
    )
    link = f"/api/runs/{run_id}"
    assert stranger.post(f"{link}/stop").status_code == 404
    assert owner.post(f"{link}/stop", headers={"X-CSRF-Token": "wrong"}).status_code == 403
    for _ in range(2):
        response = owner.post(f"{link}/stop")
        assert response.status_code == 202, response.text
        assert response.json()["stop_requested"] is True
        assert response.json()["status"] == "running"
    assert owner.get(link).json()["stop_requested"] is True
    assert stop_requested(job)
    with db.connection() as conn:
        with pytest.raises(RuntimeError, match="stop"):
            _owned(job, conn)
    # No processes were started in this fixture, so reconciliation is safe now.
    reconcile("run")
    stopped = owner.get(link).json()
    assert stopped["status"] == "interrupted"
    assert stopped["stop_requested"] is False
    assert "Stopped by request" in stopped["error"]
    assert (tmp_path / "users" / str(run["app_user_id"]) / "runs" / run_id / "result.txt").read_text() == "result"
    assert owner.post(f"{link}/stop").status_code == 202
    resumed = owner.post(f"{link}/resume")
    assert resumed.status_code == 202, resumed.text
    assert resumed.json()["status"] == "queued"
    next_job = db.one("SELECT * FROM web_jobs WHERE target_id=%s AND kind='resume'", (run_id,))
    assert next_job["payload"] == {}
    assert next_job["status"] == "pending"
