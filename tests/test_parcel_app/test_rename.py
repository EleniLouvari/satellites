"""Check name-only updates of runs, EDA analyses and ML scenarios without connecting to the database."""

from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from parcel_app.pipelines.eda import routes as eda_routes
from parcel_app.pipelines.ml import routes as ml_routes
from parcel_app.pipelines.parcel_stats import routes as runs


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


@pytest.fixture
def eda_rename_client(monkeypatch):
    """Serve the EDA routes with authentication and database I/O isolated."""
    row = {
        "eda_id": str(uuid4()),
        "run_id": str(uuid4()),
        "app_user_id": 7,
        "name": "EDA: Original",
        "status": "complete",
        "config": {"report_title": "EDA: Original"},
        "selected_columns": ["a"],
        "artifacts": [],
        "input_sha256": "hash",
        "output_dir": "/private/eda",
    }
    reads = Mock(return_value=row)
    writes = Mock()
    monkeypatch.setattr(eda_routes.db, "connection", lambda: nullcontext(None))
    monkeypatch.setattr(eda_routes.db, "one", reads)
    monkeypatch.setattr(eda_routes.db, "execute", writes)
    app = FastAPI()
    app.include_router(eda_routes.router)
    app.dependency_overrides[eda_routes.current_user] = lambda: {"app_user_id": 7}
    with TestClient(app) as client:
        yield client, row, reads, writes


@pytest.mark.parametrize("status", ["queued", "running", "complete", "failed", "interrupted"])
def test_eda_rename_changes_only_the_display_name(eda_rename_client, status):
    """An EDA label change keeps the saved report title, settings and private paths."""
    client, row, reads, writes = eda_rename_client
    row["status"] = status
    response = client.patch(f"/eda/{row['eda_id']}/name", json={"name": "  Wheat only  "})
    assert response.status_code == 200
    result = response.json()
    assert result["name"] == "Wheat only"
    assert result["status"] == status
    assert result["config"]["report_title"] == "EDA: Original"
    assert "output_dir" not in result and "app_user_id" not in result
    query, params = reads.call_args.args
    assert "app_user_id=%s" in query and "FOR UPDATE" in query
    assert params == (row["eda_id"], 7)
    query, params = writes.call_args.args
    assert query == "UPDATE eda_runs SET name=%s WHERE eda_id=%s"
    assert params == ("Wheat only", row["eda_id"])


@pytest.mark.parametrize("payload", [{"name": ""}, {"name": "   "}, {"name": "x" * 121}, {"name": "New", "config": {}}])
def test_eda_rename_rejects_invalid_input(eda_rename_client, payload):
    """Reject blank, oversized and non-name fields without writing anything."""
    client, row, _, writes = eda_rename_client
    assert client.patch(f"/eda/{row['eda_id']}/name", json=payload).status_code == 422
    writes.assert_not_called()


def test_eda_rename_rejects_foreign_missing_or_deleting(eda_rename_client):
    """Hide other users' analyses and leave analyses being deleted untouched."""
    client, row, reads, writes = eda_rename_client
    reads.return_value = None
    assert client.patch(f"/eda/{row['eda_id']}/name", json={"name": "New"}).status_code == 404
    reads.return_value = row | {"status": "deleting"}
    assert client.patch(f"/eda/{row['eda_id']}/name", json={"name": "New"}).status_code == 409
    writes.assert_not_called()


@pytest.fixture
def ml_rename_client(monkeypatch):
    """Serve the ML routes with authentication and database I/O isolated."""
    row = {
        "ml_id": str(uuid4()),
        "eda_id": str(uuid4()),
        "run_id": str(uuid4()),
        "app_user_id": 7,
        "name": "ML: Original",
        "status": "paused",
        "steps_done": 2,
        "config": {"selected_models": ["knn"]},
        "feature_columns": ["a"],
        "artifacts": [],
        "input_sha256": "hash",
        "eda_sha256": "hash",
        "output_dir": "/private/ml",
    }
    reads = Mock(return_value=row)
    writes = Mock()
    monkeypatch.setattr(ml_routes.db, "connection", lambda: nullcontext(None))
    monkeypatch.setattr(ml_routes.db, "one", reads)
    monkeypatch.setattr(ml_routes.db, "execute", writes)
    app = FastAPI()
    app.include_router(ml_routes.router)
    app.dependency_overrides[ml_routes.current_user] = lambda: {"app_user_id": 7}
    with TestClient(app) as client:
        yield client, row, reads, writes


@pytest.mark.parametrize("status", ["queued", "running", "paused", "complete", "failed", "interrupted"])
def test_ml_rename_changes_only_the_display_name(ml_rename_client, status):
    """A scenario label change keeps its settings, completed steps and private paths."""
    client, row, reads, writes = ml_rename_client
    row["status"] = status
    response = client.patch(f"/ml/{row['ml_id']}/name", json={"name": "  Trees, all indices  "})
    assert response.status_code == 200
    result = response.json()
    assert result["name"] == "Trees, all indices"
    assert result["status"] == status and result["steps_done"] == 2
    assert result["config"] == row["config"]
    assert not {"output_dir", "app_user_id", "input_sha256", "eda_sha256"} & set(result)
    query, params = reads.call_args.args
    assert "app_user_id=%s" in query and "FOR UPDATE" in query
    assert params == (row["ml_id"], 7)
    query, params = writes.call_args.args
    assert query == "UPDATE ml_runs SET name=%s WHERE ml_id=%s"
    assert params == ("Trees, all indices", row["ml_id"])


@pytest.mark.parametrize("payload", [{"name": ""}, {"name": "   "}, {"name": "x" * 201}, {"name": "New", "config": {}}])
def test_ml_rename_rejects_invalid_input(ml_rename_client, payload):
    """Reject blank, oversized and non-name fields without writing anything."""
    client, row, _, writes = ml_rename_client
    assert client.patch(f"/ml/{row['ml_id']}/name", json=payload).status_code == 422
    writes.assert_not_called()


def test_ml_rename_rejects_foreign_missing_or_deleting(ml_rename_client):
    """Hide other users' scenarios and leave scenarios being deleted untouched."""
    client, row, reads, writes = ml_rename_client
    reads.return_value = None
    assert client.patch(f"/ml/{row['ml_id']}/name", json={"name": "New"}).status_code == 404
    reads.return_value = row | {"status": "deleting"}
    assert client.patch(f"/ml/{row['ml_id']}/name", json={"name": "New"}).status_code == 409
    writes.assert_not_called()
