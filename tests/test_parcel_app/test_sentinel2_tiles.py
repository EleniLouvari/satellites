"""Sentinel-2 map tiles fetched from Sentinel Hub for the inspection map."""

import pytest
from fastapi import HTTPException

from parcel_app.sources.openeo import sentinel2_tiles


def test_tile_bounds_cover_the_web_mercator_world():
    half = sentinel2_tiles.HALF_WORLD
    assert sentinel2_tiles.tile_bounds(0, 0, 0) == pytest.approx((-half, -half, half, half))
    # Tile (1, 1) at zoom 1 is the south-east quarter.
    assert sentinel2_tiles.tile_bounds(1, 1, 1) == pytest.approx((0, -half, half, 0))


class Reply:
    def __init__(self, status, content_type, content=b"png"):
        self.status_code, self.headers, self.content = status, {"content-type": content_type}, content


def test_tile_request_uses_the_run_criteria_and_rejects_non_images(monkeypatch):
    import requests

    calls = []
    monkeypatch.setattr(requests, "get", lambda url, params, timeout: calls.append((url, params)) or Reply(200, "image/png"))
    assert sentinel2_tiles.fetch_tile("instance", 14, 9200, 6250, "2024-03-01", "2024-09-30", 40) == b"png"
    url, params = calls[0]
    assert url.endswith("/ogc/wms/instance")
    assert (params["TIME"], params["MAXCC"], params["LAYERS"], params["SHOWLOGO"]) == (
        "2024-03-01T00:00:00Z/2024-09-30T23:59:59Z", 40, "TRUE_COLOR", "false")
    # Sentinel Hub reports errors as XML, sometimes with status 200.
    monkeypatch.setattr(requests, "get", lambda url, params, timeout: Reply(200, "application/vnd.ogc.se_xml"))
    with pytest.raises(HTTPException) as error:
        sentinel2_tiles.fetch_tile("instance", 14, 0, 0, "2024-03-01", "2024-09-30", 40)
    assert error.value.status_code == 502 and "instance" not in error.value.detail


class Listing:
    status_code = 200

    def __init__(self, features):
        self.features = features

    def json(self):
        return {"features": self.features}


def test_dates_are_one_per_day_oldest_first_across_pages(monkeypatch):
    import requests

    def acquisition(date, cloud):
        return {"properties": {"date": date, "cloudCoverPercentage": cloud}}

    # A full first page (newest first, as Sentinel Hub returns them), then a short last page.
    first = [acquisition(f"2024-06-{day:02d}", 50.0) for day in range(30, 0, -1)] * 3 + [acquisition("2024-06-15", 5.0)] * 10
    pages = [first, [acquisition("2024-05-20", 12.5), acquisition("2024-05-20", 3.0)]]
    calls = []

    def fake_get(url, params, timeout):
        calls.append(params)
        return Listing(pages[len(calls) - 1])

    monkeypatch.setattr(requests, "get", fake_get)
    dates = sentinel2_tiles.fetch_dates("instance", (22.3, 39.5, 22.4, 39.6), "2024-05-01", "2024-06-30", 30)
    assert [params["FEATURE_OFFSET"] for params in calls] == [0, 100]
    assert calls[0]["BBOX"] == "39.5,22.3,39.6,22.4" and calls[0]["MAXCC"] == 30
    assert [item["date"] for item in dates] == ["2024-05-20", *[f"2024-06-{day:02d}" for day in range(1, 31)]]
    # Granules of the same day become one date with the lowest cloud cover.
    assert dates[0]["cloud_cover"] == 3.0 and next(item for item in dates if item["date"] == "2024-06-15")["cloud_cover"] == 5.0
