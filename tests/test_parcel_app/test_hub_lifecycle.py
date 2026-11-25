"""Real owner-scoped HUB queue lifecycle against disposable PostgreSQL only."""

from uuid import uuid4

import geopandas as gpd
from shapely.geometry import box

from tests.test_data_preparation.test_parcel_stats.test_hub import hub_source as hub_source  # noqa: PLC0414 - pytest fixture export
from tests.test_parcel_app.test_routes import clients as clients  # noqa: PLC0414 - pytest fixture export


def test_hub_lifecycle_without_copernicus_accounts(clients, hub_source, tmp_path, monkeypatch):
    """Catch source loss, incorrect reservations, unsafe ownership, or incompatible worker publication."""
    from parcel_app.core import db
    from parcel_app.jobs.dispatch import execute_task

    for name in ("CATALOG_URL", "KEYCLOAK_URL", "CATALOG_CLIENT_ID", "CATALOG_CLIENT_SECRET"):
        monkeypatch.setenv(name, "https://catalog.example.test" if name.endswith("URL") else "internal")
    def no_copernicus(*args, **kwargs):
        """Fail if HUB execution or deletion reaches a Copernicus boundary."""
        raise AssertionError("HUB accessed a Copernicus service")

    monkeypatch.setattr("parcel_app.pipelines.parcel_stats.tasks.run_credentials", no_copernicus)
    owner, stranger = clients
    path = tmp_path / "input.gpkg"
    gpd.GeoDataFrame(
        {"parcel_id": ["a"]}, geometry=[box(500010, 4200010, 500030, 4200030)], crs=2100,
    ).to_file(path, layer="parcels", driver="GPKG")

    def execute(kind, target):
        """Claim one real queued task and run the normal fenced dispatcher."""
        job = db.one(
            "UPDATE web_jobs SET status='running',claim_token=%s "
            "WHERE kind=%s AND target_id=%s AND status='pending' RETURNING *",
            (str(uuid4()), kind, str(target)),
        )
        assert job is not None
        execute_task(job)
        result = db.one("SELECT status,error FROM web_jobs WHERE job_id=%s", (str(job["job_id"]),))
        assert result["status"] == "done", result

    upload = owner.post("/api/uploads?filename=parcels.gpkg", content=path.read_bytes()).json()
    execute("inspect", upload["upload_id"])
    body = {
        "name": "HUB lifecycle", "source": "hub", "account_ids": [], "upload_id": upload["upload_id"],
        "layer": "parcels", "parcel_id_field": "parcel_id", "scheduling": "partitions",
        "processing": {
            "start_date": "2024-01-01", "end_date": "2024-01-31", "working_epsg": 2100,
            "sentinel2_bands": ["B04", "B08"], "sentinel2_indices": ["NDVI"],
            "spatial_statistics": ["mean", "median"], "hub_read_workers": 1, "hub_block_size": 64,
        },
    }
    response = owner.post("/api/runs", json=body)
    assert response.status_code == 201, response.text
    run = response.json()
    run_id = run["run_id"]
    assert run["source"] == "hub" and run["scheduling"] == "batches"
    assert stranger.get(f"/api/runs/{run_id}").status_code == 404
    assert owner.post(f"/api/runs/{run_id}/validate").status_code == 202
    execute("validate", run_id)
    assert owner.post(f"/api/runs/{run_id}/submit", json={"revision": 1}).status_code == 202
    execute("run", run_id)
    assert owner.get(f"/api/runs/{run_id}").json()["status"] == "complete"
    assert db.one("SELECT count(*) AS n FROM run_accounts WHERE run_id=%s", (run_id,))["n"] == 0
    artifacts = owner.get(f"/api/runs/{run_id}/artifacts").json()
    assert any(row["kind"] == "time_statistics" for row in artifacts)
    copy = owner.post(f"/api/runs/{run_id}/duplicate")
    assert copy.status_code == 201, copy.text
    assert copy.json()["source"] == "hub" and copy.json()["account_ids"] == []
    reads = len(hub_source.opened)
    db.execute("UPDATE analysis_runs SET status='interrupted' WHERE run_id=%s", (run_id,))
    assert owner.post(f"/api/runs/{run_id}/resume").status_code == 202
    execute("resume", run_id)
    assert owner.get(f"/api/runs/{run_id}").json()["status"] == "complete"
    assert len(hub_source.opened) == reads
    assert owner.patch(f"/api/runs/{run_id}", json=body).status_code == 409
    monkeypatch.setattr("parcel_app.pipelines.parcel_stats.deletion.collect_jobs", no_copernicus)
    monkeypatch.setattr("parcel_app.pipelines.parcel_stats.deletion.delete_jobs", no_copernicus)
    assert stranger.delete(f"/api/runs/{run_id}").status_code == 404
    deleted = owner.delete(f"/api/runs/{run_id}")
    assert deleted.status_code == 200 and deleted.json()["deleted"], deleted.text
    assert owner.get(f"/api/runs/{copy.json()['run_id']}").status_code == 200
