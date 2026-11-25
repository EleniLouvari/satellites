"""Source-aware request validation and publication without network access."""

import json
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

from data_preparation.parcel_stats.hub import HubZonalStats
from data_preparation.parcel_stats.run_config import _processing_parameters, prepare_run
from data_preparation.parcel_stats.runner import resume_parcel_stats, run_parcel_stats
from tests.test_data_preparation.test_parcel_stats.test_hub import hub_source as hub_source  # noqa: PLC0414 - pytest fixture export
from tests.utils import expect_equal, expect_true


@pytest.fixture
def hub_request(tmp_path, hub_source):
    """Persist one owned input separate from the runner's output directory."""
    parcels = gpd.GeoDataFrame(
        {"parcel_id": ["a"], "label": [3]}, geometry=[box(500010, 4200010, 500030, 4200030)], crs=2100,
    )
    path = tmp_path / "parcels.geoparquet"
    parcels.to_parquet(path)
    return {
        "schema_version": 1, "source": "hub", "run_identifier": "hub-test",
        "input": {"path": str(path), "parcel_id_field": "parcel_id"},
        "output_dir": str(tmp_path / "run"),
        "processing": {
            "start_date": "2024-01-01", "end_date": "2024-01-31",
            "sentinel2_bands": ["B04", "B08"], "sentinel2_indices": ["NDVI"],
            "sentinel1_bands": [], "sentinel1_indices": [],
            "spatial_statistics": ["mean", "median"], "temporal_reducer": "none",
            "fill_nulls": False, "batch_workers": 1, "hub_read_workers": 1, "hub_block_size": 64,
        },
    }


@pytest.fixture(autouse=True)
def hub_environment(monkeypatch):
    """Supply non-secret fixture endpoint settings before offline construction."""
    monkeypatch.setenv("CATALOG_URL", "https://catalog.example.test")
    monkeypatch.setenv("CATALOG_CLIENT_ID", "internal")


def test_hub_preflight_is_offline_and_uses_source_parameters(hub_request, hub_source):
    """Catch openEO options or discovery leaking into HUB's offline preflight."""
    parameters = _processing_parameters("hub")
    expect_true("hub_read_workers" in parameters)
    expect_true("openeo_parallel_jobs" not in parameters)
    expect_true("tile_size_metres" not in parameters)
    prepared = prepare_run(hub_request)
    expect_equal(prepared.config["source"], "hub")
    expect_equal(prepared.config["processing"]["working_epsg"], 2100)
    expect_equal(prepared.batch_count, 1)
    expect_equal(hub_source.searches, [])
    expect_true(not Path(hub_request["output_dir"]).exists())


def test_hub_runner_publishes_and_resumes_without_copernicus_accounts(hub_request, hub_source):
    """Catch missing lifecycle artifacts, duplicate acquisition, or account requirements."""
    manifest = run_parcel_stats(hub_request, app_user_id=17)
    expect_equal(manifest["status"], "complete")
    final = gpd.read_parquet(manifest["artifacts"]["time_statistics"])
    np.testing.assert_allclose(final["NDVI_median__20240101"], [0.5], rtol=1e-6)
    reads = len(hub_source.opened)
    resumed = resume_parcel_stats(hub_request["output_dir"], app_user_id=17)
    expect_equal(resumed["status"], "complete")
    expect_equal(len(hub_source.opened), reads)
    state = json.loads((Path(hub_request["output_dir"]) / "run.json").read_text())
    expect_equal(state["status"], "complete")
    expect_equal(state["accounts"], [])
    with pytest.raises(ValueError, match="own"):
        resume_parcel_stats(hub_request["output_dir"], app_user_id=18)


def test_hub_rejects_copernicus_accounts_and_private_processing(hub_request):
    """Catch inappropriate account use and browser-controlled catalog endpoints."""
    with pytest.raises(ValueError, match="HUB"):
        run_parcel_stats(hub_request, accounts=[("someone", "password")])
    hub_request["processing"]["hub_catalog_url"] = "https://untrusted.example.test"
    with pytest.raises(ValueError, match="unsupported"):
        prepare_run(hub_request)


def test_unknown_source_is_rejected(hub_request):
    """Catch silently falling back to openEO for an unknown provider."""
    hub_request["source"] = "other"
    with pytest.raises(ValueError, match="source"):
        prepare_run(hub_request)


def test_hub_preflight_counts_the_finer_acquisition_batches(hub_request):
    """Catch progress denominators based on obsolete wide parent batches."""
    parcels = gpd.GeoDataFrame(
        {"parcel_id": ["a", "b"]},
        geometry=[box(500010, 4200010, 500030, 4200030), box(520010, 4200010, 520030, 4200030)], crs=2100,
    )
    parcels.to_parquet(hub_request["input"]["path"])
    expect_equal(prepare_run(hub_request).batch_count, 2)


def test_hub_publication_uses_the_acquired_inventory_snapshot(hub_request, hub_source, monkeypatch):
    """Catch a successful run failing when the catalog changes just before publication."""
    from data_preparation.sources.hub import cubes

    original = cubes.CatalogSearchUtils.fetch_features

    def evolving_catalog(instance, search_body, **kwargs):
        """Expose a new item version on every independent discovery call."""
        version = 20 + len(hub_source.searches)
        hub_source.features["ARD_S2"]["properties"]["processing:datetime"] = f"2026-11-{version:02d}T00:00:00Z"
        return original(instance, search_body, **kwargs)

    monkeypatch.setattr(cubes.CatalogSearchUtils, "fetch_features", evolving_catalog)
    expect_equal(run_parcel_stats(hub_request)["status"], "complete")
    expect_equal(len(hub_source.searches), 1)


def test_failed_run_logs_the_reason_and_its_cause(hub_request, monkeypatch, caplog):
    """Catch failures that persist only an exception type, hiding the endpoint error behind a retry wrapper."""
    def fail(self):
        try:
            raise ValueError("No connection adapters were found for '<https://auth.example.test>/token'")
        except ValueError as exc:
            raise RuntimeError("Failed to acquire OAuth2 token") from exc

    monkeypatch.setattr(HubZonalStats, "run", fail)
    with caplog.at_level(logging.ERROR, logger="parcel_stats_pipeline"), pytest.raises(RuntimeError):
        run_parcel_stats(hub_request, app_user_id=17)
    output = Path(hub_request["output_dir"])
    expect_equal(json.loads((output / "run.json").read_text())["status"], "failed")
    expected = (
        "RuntimeError: Failed to acquire OAuth2 token <- caused by ValueError: "
        "No connection adapters were found for '<https://auth.example.test>/token'"
    )
    # Run-level orchestration log: exported by Parcel Studio diagnostics.
    expect_true(expected in (output / "satellite_zonal_stats.log").read_text(encoding="utf-8"))
    # Pipeline-namespace record: forwarded by Parcel Studio as a live run event.
    expect_true(any(
        record.name.startswith("parcel_stats_pipeline") and expected in record.getMessage() for record in caplog.records
    ))
