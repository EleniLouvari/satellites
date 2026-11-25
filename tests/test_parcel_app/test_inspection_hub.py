"""HUB inspection routes enforce ownership and selection before rendering."""

import sys
from dataclasses import dataclass
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from parcel_app.app import create_app
from parcel_app.core.security import current_user
from parcel_app.pipelines.ml import inspection


@dataclass
class Layer:
    """Provider layer contract, without acquisition dependencies."""

    name: str = "NDVI"
    sensor: str = "S2"
    kind: str = "index"
    legend: dict | None = None


@dataclass
class Acquisition:
    """Provider acquisition contract with a public cache revision."""

    id: str = "s2|pass-a"
    datetime: str = "2025-05-01T10:00:00Z"
    sensor: str = "S2"
    cloud_cover: float = 10
    revision: str = "0123456789abcdef"


@pytest.fixture
def hub_client(tmp_path, monkeypatch):
    """Provide owned predictions and a fake external renderer, without a database."""
    prediction = tmp_path / inspection.PREDICTIONS
    prediction.parent.mkdir()
    prediction.write_text("parcel_id,inspection_need\np1,HIGH\n", encoding="utf-8")
    row = {"ml_id": uuid4(), "run_id": uuid4(), "app_user_id": 1, "is_best": True,
           "status": "complete", "output_dir": str(tmp_path), "config": {"id_column": "parcel_id"},
           "artifacts": [{"name": inspection.PREDICTIONS, "sha256": "test"}]}
    run = {"source": "hub", "output_dir": str(tmp_path), "processing": {"start_date": "2025-01-01"},
           "config": {"processing": {"end_date": "2025-12-31", "sentinel2_indices": ["NDVI"]}}}

    def owned(ml_id, user_id):
        """Model the real owner query's not-found boundary."""
        if str(ml_id) != str(row["ml_id"]) or user_id != 1:
            raise HTTPException(404, "Error: ML run not found.")
        return row

    calls = []

    def render(*args):
        """Record any access across the external rendering boundary."""
        calls.append(args)
        return b"png"

    provider = SimpleNamespace(
        preview_layers=lambda processing: [Layer()],
        parcel_acquisitions=lambda *args: [Acquisition()],
        parcel_preview_bounds=lambda parcel: [[38.0, 23.0], [38.01, 23.01]],
        render_preview=render,
    )
    monkeypatch.setitem(sys.modules, "parcel_app.sources.hub.previews", provider)
    monkeypatch.setattr(inspection, "owned_ml", owned)
    monkeypatch.setattr(inspection, "owned_run", lambda *args: run)
    monkeypatch.setattr(inspection, "completed_run", lambda *args: run)
    monkeypatch.setattr(inspection, "source_artifact", lambda *args: ({}, tmp_path / "result.parquet"))
    monkeypatch.setattr(inspection, "parcel_geometry", lambda *args: {"type": "Feature", "geometry": {"type": "Polygon"}})
    from parcel_app.sources.hub import credentials

    monkeypatch.setattr(credentials, "configured", lambda: True)
    app = create_app()
    app.dependency_overrides[current_user] = lambda: {"app_user_id": 1}
    with TestClient(app, raise_server_exceptions=False) as client:
        yield SimpleNamespace(client=client, app=app, row=row, run=run, provider=provider, calls=calls,
                              tmp_path=tmp_path, prediction=prediction, url=f"/api/ml/{row['ml_id']}/inspection")


def test_hub_metadata_and_source_dispatch(hub_client, monkeypatch):
    """HUB runs advertise HUB layers without touching Copernicus credentials."""
    ctx = hub_client
    monkeypatch.setattr(inspection, "sentinelhub_instance", lambda *args: pytest.fail("Unexpected Copernicus lookup"))
    source = ctx.client.get(f"{ctx.url}/sentinel2")
    assert source.status_code == 200 and source.json()["source"] == "hub"
    result = ctx.client.get(f"{ctx.url}/hub")
    assert result.status_code == 200
    assert result.json()["layers"][0]["name"] == "NDVI"
    assert result.json()["available"] is True


def test_acquisitions_include_exact_bounds_and_revision(hub_client):
    """A valid prediction gets the provider's bounds and acquisition cache identity."""
    ctx = hub_client
    result = ctx.client.get(f"{ctx.url}/hub/acquisitions", params={"parcel_id": "p1"})
    assert result.status_code == 200
    assert result.json()["bounds"] == [[38.0, 23.0], [38.01, 23.01]]
    assert result.json()["acquisitions"][0]["revision"] == Acquisition().revision


def test_image_selection_and_private_cache(hub_client):
    """Only a configured layer/acquisition can render, with private browser caching."""
    ctx = hub_client
    params = {"parcel_id": "p1", "acquisition": Acquisition().id, "layer": "NDVI", "rev": Acquisition().revision}
    result = ctx.client.get(f"{ctx.url}/hub/image.png", params=params)
    assert result.status_code == 200 and result.content == b"png"
    assert result.headers["content-type"] == "image/png"
    assert result.headers["cache-control"] == "private, max-age=86400"
    assert len(ctx.calls) == 1
    for changes in ({"layer": "EVI"}, {"acquisition": "unknown"}, {"rev": "stale"}, {"parcel_id": "foreign"}):
        assert ctx.client.get(f"{ctx.url}/hub/image.png", params={**params, **changes}).status_code in (404, 409)
    assert len(ctx.calls) == 1


@pytest.mark.parametrize("suffix", ["/hub", "/hub/acquisitions?parcel_id=p1",
                                        "/hub/image.png?parcel_id=p1&acquisition=s2%7Cpass-a&layer=NDVI"])
def test_foreign_owner_cannot_reach_provider(hub_client, suffix):
    """Ownership applies to metadata and to image requests even on a cache hit."""
    ctx = hub_client
    ctx.app.dependency_overrides[current_user] = lambda: {"app_user_id": 2}
    result = ctx.client.get(f"{ctx.url}{suffix}")
    assert result.status_code == 404 and result.json()["detail"] == "Error: ML run not found."
    assert not ctx.calls


def test_hub_source_and_best_scenario_guards(hub_client):
    """The provider cannot be reached for another source or an unfinished scenario."""
    ctx = hub_client
    ctx.run["source"] = "openeo"
    assert ctx.client.get(f"{ctx.url}/hub").status_code == 404
    ctx.run["source"] = "hub"
    ctx.row["is_best"] = False
    assert ctx.client.get(f"{ctx.url}/hub").status_code == 409


def test_safe_provider_errors_and_missing_configuration(hub_client, monkeypatch):
    """Failures expose safe messages rather than upstream credentials or URLs."""
    ctx = hub_client

    def unavailable(*args):
        """Represent a provider validation failure."""
        raise ValueError("Error: Original imagery is unavailable.")

    def failed(*args):
        """Represent a remote failure containing private details."""
        raise RuntimeError("https://private.invalid?token=secret")

    def invalid_upstream(*args):
        """An upstream parser's ValueError is not the provider's safe validation."""
        raise ValueError("https://private.invalid?token=secret")

    params = {"parcel_id": "p1"}
    monkeypatch.setattr(ctx.provider, "parcel_acquisitions", unavailable)
    result = ctx.client.get(f"{ctx.url}/hub/acquisitions", params=params)
    assert result.status_code == 409 and "Original imagery" in result.text
    monkeypatch.setattr(ctx.provider, "parcel_acquisitions", failed)
    result = ctx.client.get(f"{ctx.url}/hub/acquisitions", params=params)
    assert result.status_code == 502 and "secret" not in result.text and "private.invalid" not in result.text
    monkeypatch.setattr(ctx.provider, "parcel_acquisitions", invalid_upstream)
    result = ctx.client.get(f"{ctx.url}/hub/acquisitions", params=params)
    assert result.status_code == 502 and "secret" not in result.text
    from parcel_app.sources.hub import credentials

    monkeypatch.setattr(credentials, "configured", lambda: False)
    assert ctx.client.get(f"{ctx.url}/hub").json()["available"] is False
    assert ctx.client.get(f"{ctx.url}/hub/acquisitions", params=params).status_code == 409


def test_statistics_return_only_calculated_values_per_period(hub_client):
    """Saved mean, median, sd and range of the selected index are returned per period; NaN becomes null."""
    import pandas as pd

    ctx = hub_client
    pd.DataFrame({
        "parcel_id": [1, 2],
        "NDVI_mean__20250501": [0.61, 0.1],
        "NDVI_median__20250501": [0.63, 0.1],
        "NDVI_sd__20250501": [float("nan"), 0.1],
        "NDVI_median__20250401": [0.2, 0.1],
        "NDVI_p10__20250501": [0.4, 0.1],
        "NDVI_mean": [0.5, 0.1],
        "NDWI_mean__20250501": [-0.3, 0.1],
    }).to_parquet(ctx.tmp_path / "result.parquet")
    ctx.run["processing"]["temporal_period"] = "1M"
    result = ctx.client.get(f"{ctx.url}/hub/statistics", params={"parcel_id": "p1", "layer": "NDVI"})
    assert result.status_code == 404  # p1 is in the predictions but has no row in the result file.
    ctx.prediction.write_text("parcel_id,inspection_need\n1,HIGH\n", encoding="utf-8")
    inspection.ordered_predictions.cache_clear()
    result = ctx.client.get(f"{ctx.url}/hub/statistics", params={"parcel_id": "1", "layer": "NDVI"})
    assert result.status_code == 200
    assert result.json() == {"variable": "NDVI", "temporal_period": "1M", "periods": [
        {"start": "2025-04-01", "values": {"median": 0.2}},
        {"start": "2025-05-01", "values": {"mean": 0.61, "median": 0.63, "sd": None}},
    ]}


def test_statistics_guard_layers_and_parcels(hub_client):
    """Unknown layers and foreign parcels are refused; composites and Planet layers have no statistics."""
    ctx = hub_client
    params = {"parcel_id": "p1", "layer": "NDVI"}
    assert ctx.client.get(f"{ctx.url}/hub/statistics", params={**params, "layer": "EVI"}).status_code == 404
    assert ctx.client.get(f"{ctx.url}/hub/statistics", params={**params, "parcel_id": "foreign"}).status_code == 404
    for layer in (Layer(name="RGB", kind="rgb"), Layer(name="PL_NDVI", sensor="PL")):
        ctx.provider.preview_layers = lambda processing, layer=layer: [layer]
        result = ctx.client.get(f"{ctx.url}/hub/statistics", params={**params, "layer": layer.name})
        assert result.status_code == 200 and result.json()["periods"] == []
