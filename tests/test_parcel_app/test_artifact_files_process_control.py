"""Database-free unit tests for artifact file routes and analysis process-tree control."""

import os
from types import SimpleNamespace
from uuid import uuid4

import psutil
import pytest
from fastapi import HTTPException

from parcel_app.jobs import process_control
from parcel_app.pipelines.parcel_stats import files

USER = {"app_user_id": 7}


@pytest.fixture
def run_dir(tmp_path):
    """Return a run output folder containing one deliverable and one outside file."""
    output = tmp_path / "run"
    output.mkdir()
    (output / "stats.parquet").write_bytes(b"data")
    (tmp_path / "outside.txt").write_text("secret", encoding="utf-8")
    return output


def _row(run_dir, name="stats.parquet", kind="time_statistics", status="complete"):
    """Build the joined artifact/run row returned by the download query."""
    return {"output_dir": str(run_dir), "storage_path": str(run_dir / name), "name": name, "kind": kind, "status": status}


# --------------------------------------------------------------------------------
# contained_artifact / download
# --------------------------------------------------------------------------------


def test_contained_artifact_accepts_files_inside_run(run_dir):
    """A registered file under the run folder resolves to its real path."""
    row = _row(run_dir)
    assert files.contained_artifact(row, row) == (run_dir / "stats.parquet").resolve()


@pytest.mark.parametrize("storage", ["../outside.txt", "missing.parquet", "."])
def test_contained_artifact_rejects_escapes_missing_files_and_directories(run_dir, storage):
    """Paths outside the run, missing files and directories are unavailable."""
    run = {"output_dir": str(run_dir)}
    with pytest.raises(ValueError, match="Error: Artifact is unavailable"):
        files.contained_artifact(run, {"storage_path": str(run_dir / storage)})


def test_download_streams_contained_artifact_with_safe_headers(run_dir, monkeypatch):
    """Owned, complete artifacts stream as attachments that browsers must not sniff or cache."""
    monkeypatch.setattr(files.db, "one", lambda *_args, **_kwargs: _row(run_dir))

    response = files.download(uuid4(), user=USER)

    assert response.path == (run_dir / "stats.parquet").resolve()
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["cache-control"] == "private, no-store"
    assert "stats.parquet" in response.headers["content-disposition"]


@pytest.mark.parametrize(
    ("row_factory", "status_code"),
    [
        (lambda _run_dir: None, 404),
        (lambda run_dir: _row(run_dir, kind="bundle", status="running"), 409),
        (lambda run_dir: _row(run_dir, name="../outside.txt"), 404),
    ],
)
def test_download_rejects_unowned_incomplete_and_escaped_artifacts(run_dir, monkeypatch, row_factory, status_code):
    """Missing ownership, incomplete final results and escaped paths map to HTTP errors."""
    monkeypatch.setattr(files.db, "one", lambda *_args, **_kwargs: row_factory(run_dir))

    with pytest.raises(HTTPException) as error:
        files.download(uuid4(), user=USER)

    assert error.value.status_code == status_code
    assert error.value.detail.startswith("Error: ")


def test_download_allows_intermediate_artifacts_before_completion(run_dir, monkeypatch):
    """Non-final artifact kinds remain downloadable while a run is still active."""
    monkeypatch.setattr(files.db, "one", lambda *_args, **_kwargs: _row(run_dir, kind="log", status="running"))
    assert files.download(uuid4(), user=USER).path == (run_dir / "stats.parquet").resolve()


def test_artifacts_lists_rows_for_owned_run(monkeypatch):
    """Listing checks ownership first and passes the run id as a bound parameter."""
    run_id, owned, queries = uuid4(), [], []
    monkeypatch.setattr(files, "owned_run", lambda rid, uid: owned.append((rid, uid)))
    monkeypatch.setattr(files.db, "all_rows", lambda sql, params: queries.append(params) or [{"name": "a"}])

    assert files.artifacts(run_id, user=USER) == [{"name": "a"}]
    assert owned == [(run_id, 7)]
    assert queries == [(str(run_id),)]


# --------------------------------------------------------------------------------
# bundle / bundle_status
# --------------------------------------------------------------------------------


class _Connection:
    """Context manager standing in for a pooled database connection."""

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


def _patch_bundle(monkeypatch, status="complete", ready=None):
    """Patch ownership, the bundle lookup and job queue; return the enqueue call log."""
    enqueued = []
    monkeypatch.setattr(files.db, "connection", _Connection)
    monkeypatch.setattr(files, "owned_run", lambda *_args, **_kwargs: {"status": status})
    monkeypatch.setattr(files.db, "one", lambda *_args, **_kwargs: ready)
    monkeypatch.setattr(
        files, "enqueue", lambda kind, run_id, user_id, conn: enqueued.append(kind) or {"job_id": 9, "status": "queued"}
    )
    return enqueued


def test_bundle_requires_complete_run(monkeypatch):
    """Archives are only prepared for complete runs."""
    _patch_bundle(monkeypatch, status="running")
    with pytest.raises(HTTPException) as error:
        files.bundle(uuid4(), user=USER)
    assert error.value.status_code == 409


def test_bundle_reuses_ready_archive_or_queues_job(monkeypatch):
    """A ready archive is returned directly; otherwise one bundle job is queued."""
    enqueued = _patch_bundle(monkeypatch, ready={"artifact_id": 3})
    assert files.bundle(uuid4(), user=USER) == {"status": "ready", "artifact_id": 3}
    assert enqueued == []

    enqueued = _patch_bundle(monkeypatch, ready=None)
    assert files.bundle(uuid4(), user=USER) == {"job_id": 9, "status": "queued"}
    assert enqueued == ["bundle"]


def test_bundle_status_reports_each_lifecycle_state(monkeypatch):
    """Status reflects run completion, archive readiness, the latest job and the unrequested default."""
    lookups = []

    def fake_one(sql, _params):
        return lookups.pop(0)

    monkeypatch.setattr(files.db, "one", fake_one)
    monkeypatch.setattr(files, "owned_run", lambda *_args: {"status": "running"})
    assert files.bundle_status(uuid4(), user=USER) == {"status": "unavailable"}

    monkeypatch.setattr(files, "owned_run", lambda *_args: {"status": "complete"})
    lookups[:] = [{"artifact_id": 5}]
    assert files.bundle_status(uuid4(), user=USER) == {"status": "ready", "artifact_id": 5}
    lookups[:] = [None, {"status": "failed", "error": "Error: disk full"}]
    assert files.bundle_status(uuid4(), user=USER) == {"status": "failed", "error": "Error: disk full"}
    lookups[:] = [None, None]
    assert files.bundle_status(uuid4(), user=USER) == {"status": "unrequested"}


# --------------------------------------------------------------------------------
# process_control
# --------------------------------------------------------------------------------


class _Process:
    """psutil.Process stand-in with configurable liveness."""

    def __init__(self, running=True, status="running", missing=False):
        self._running, self._status, self._missing = running, status, missing

    def is_running(self):
        if self._missing:
            raise psutil.NoSuchProcess(pid=1)
        return self._running

    def status(self):
        return self._status


def test_can_write_excludes_exited_zombie_and_vanished_processes():
    """Only running, non-zombie processes count as active."""
    assert process_control._can_write(_Process()) is True
    assert process_control._can_write(_Process(running=False)) is False
    assert process_control._can_write(_Process(status=psutil.STATUS_ZOMBIE)) is False
    assert process_control._can_write(_Process(missing=True)) is False


def test_active_members_prefers_job_accounting_over_process_handles():
    """A job object answers for its members; otherwise process handles are checked."""
    assert process_control._active_members([_Process()], SimpleNamespace(active=lambda: 0)) is False
    assert process_control._active_members([], SimpleNamespace(active=lambda: 2)) is True
    assert process_control._active_members([_Process(running=False), _Process()], None) is True
    assert process_control._active_members([_Process(running=False)], None) is False


class _Child:
    """multiprocessing.Process stand-in recording join calls."""

    def __init__(self, alive=False):
        self.pid, self.alive, self.joined = 4321, alive, []

    def join(self, timeout=None):
        self.joined.append(timeout)

    def is_alive(self):
        return self.alive


class _Group:
    """Windows job stand-in whose members exit on termination."""

    def __init__(self, stays_active=False):
        self.terminated, self.stays_active = False, stays_active

    def terminate(self):
        self.terminated = True

    def active(self):
        return self.stays_active or not self.terminated


@pytest.mark.skipif(os.name != "nt", reason="Windows job-object branch")
def test_stop_task_terminates_windows_job_and_waits_for_child():
    """The job is terminated and the child joined before the run is released."""
    child, group = _Child(), _Group()

    process_control.stop_task(child, group)

    assert group.terminated is True
    assert child.joined == [10]


@pytest.mark.skipif(os.name != "nt", reason="Windows job-object branch")
def test_stop_task_refuses_release_while_processes_remain(monkeypatch):
    """A child or job that stays alive past the deadline blocks release; a missing job is rejected."""
    clock = iter([0.0, 0.0, 100.0])
    monkeypatch.setattr(process_control.time, "monotonic", lambda: next(clock, 100.0))
    monkeypatch.setattr(process_control.time, "sleep", lambda _seconds: None)

    with pytest.raises(RuntimeError, match="Error: Analysis processes have not exited"):
        process_control.stop_task(_Child(), _Group(stays_active=True))
    with pytest.raises(RuntimeError, match="Error: Analysis processes have not exited"):
        process_control.stop_task(_Child(alive=True), _Group())
    with pytest.raises(RuntimeError, match="Error: Cannot stop Windows analysis without its owned job"):
        process_control.stop_task(_Child(), None)


@pytest.mark.skipif(os.name != "nt", reason="Windows job-object branch")
def test_isolate_task_requires_supervisor_job_on_windows():
    """Windows tasks cannot isolate themselves without a supervisor-owned job name."""
    with pytest.raises(RuntimeError, match="Error: Windows analysis requires a supervisor-owned job"):
        process_control.isolate_task(None)


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group branch")
def test_stop_posix_group_returns_no_members_for_missing_group(monkeypatch):
    """An already-exited process group is treated as stopped."""

    def missing_group(_pid, _signal):
        raise ProcessLookupError

    monkeypatch.setattr(process_control.os, "killpg", missing_group)
    assert process_control._stop_posix_group(999_999) == []
