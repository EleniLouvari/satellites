"""Check name-only updates without connecting to the application database."""

from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from parcel_app import runs


@pytest.fixture
def rename_client(monkeypatch):
    """Serve real routes with authentication and database I/O isolated."""
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    row = {
        "run_id": str(uuid4()),
        "app_user_id": 7,
        "name": "Original",
        "status": "complete",
        "revision": 3,
        "validated_revision": 3,
        "started_at": start,
        "updated_at": start + timedelta(seconds=125),
        "config": {},
        "summary": {},
        "output_dir": "/private/run",
    }
    reads = Mock(return_value=row)
    writes = Mock()
    monkeypatch.setattr(runs.db, "connection", lambda: nullcontext(None))
    monkeypatch.setattr(runs.db, "one", reads)
    monkeypatch.setattr(runs.db, "execute", writes)
    app = FastAPI()
    app.include_router(runs.router)
    app.dependency_overrides[runs.current_user] = lambda: {"app_user_id": 7}
    with TestClient(app) as client:
        yield client, row, reads, writes


@pytest.mark.parametrize(
    "status", ["draft", "ready", "validating", "queued", "running", "complete", "failed", "interrupted", "cancelled"]
)
def test_rename_preserves_processing_and_timing(rename_client, status):
    """A label change preserves validation, lifecycle state and completion timing."""
    client, row, reads, writes = rename_client
    row["status"] = status
    response = client.patch(f"/runs/{row['run_id']}/name", json={"name": "  Field survey  "})
    assert response.status_code == 200
    result = response.json()
    assert result["name"] == "Field survey"
    assert result["status"] == status
    assert result["revision"] == result["validated_revision"] == 3
    assert datetime.fromisoformat(result["updated_at"]) == row["updated_at"]
    if status == "complete":
        assert result["duration_seconds"] == 125
    assert "output_dir" not in result
    query, params = reads.call_args.args
    assert "app_user_id=%s" in query and "FOR UPDATE" in query
    assert params == (row["run_id"], 7)
    query, params = writes.call_args.args
    assert query == "UPDATE analysis_runs SET name=%s WHERE run_id=%s"
    assert params == ("Field survey", row["run_id"])


@pytest.mark.parametrize("payload", [{"name": ""}, {"name": "   "}, {"name": "x" * 121}, {"name": "New", "status": "complete"}])
def test_rename_rejects_invalid_input(rename_client, payload):
    """Reject blank, oversized and non-name fields without writing anything."""
    client, row, _, writes = rename_client
    response = client.patch(f"/runs/{row['run_id']}/name", json=payload)
    assert response.status_code == 422
    writes.assert_not_called()


def test_rename_rejects_foreign_or_missing_run(rename_client):
    """Ownership lookup prevents renaming a run unavailable to this user."""
    client, row, reads, writes = rename_client
    reads.return_value = None
    response = client.patch(f"/runs/{row['run_id']}/name", json={"name": "New"})
    assert response.status_code == 404
    writes.assert_not_called()


def test_rename_rejects_deleting_run(rename_client):
    """Do not mutate a run once deletion has claimed it."""
    client, row, _, writes = rename_client
    row["status"] = "deleting"
    response = client.patch(f"/runs/{row['run_id']}/name", json={"name": "New"})
    assert response.status_code == 409
    writes.assert_not_called()
