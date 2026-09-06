"""Local-delivery discovery and checksum/resume tests without network access."""

from io import BytesIO
import hashlib
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import geopandas as gpd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from hub_catalog.planet_delivery import (
    build_local_planet_manifest,
    discover_planet_delivery_tiles,
    download_planet_delivery_tiles,
)
from hub_catalog.planet_parcel_stats import PlanetBasemapZonalStats


@pytest.fixture
def tmp_path():
    path = Path("tmp") / f"planet_delivery_test_{uuid4().hex}"
    path.mkdir(parents=True)
    return path


def make_delivery(path):
    path.mkdir(parents=True, exist_ok=True)
    raster_path = path / "1154-1277_quad_bandmath.tif"
    data = np.ones((10, 4, 4), dtype="float32")
    data[5], data[7] = 2000, 6000
    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        width=4,
        height=4,
        count=10,
        dtype="float32",
        crs=2100,
        transform=from_origin(500000, 4500020, 5, 5),
    ) as dst:
        dst.write(data)
    metadata = {
        "mosaic": {"first_acquired": "2025-01-01T00:00:00Z", "last_acquired": "2025-02-01T00:00:00Z"},
        "quad": {"id": "1154-1277"},
    }
    (path / "1154-1277_metadata.json").write_text(json.dumps(metadata))
    return raster_path


def test_raw_directory_and_relative_csv_run_without_catalog(tmp_path):
    raster_path = make_delivery(tmp_path / "raw")
    manifest = build_local_planet_manifest(tmp_path / "raw")
    assert manifest.period_start.tolist() == ["2025-01-01"]
    parcels = gpd.GeoDataFrame({"parcel_id": ["p1"]}, geometry=[box(500005.1, 4500005.1, 500014.9, 4500014.9)], crs=2100)
    pipeline = PlanetBasemapZonalStats(parcels, "2025-01-01", "2025-01-31", tmp_path / "out", 2100, buffer_metres=0)
    result = pipeline.run_local(tmp_path / "raw")
    assert result.iloc[0]["NDVI_mean__20250101"] == pytest.approx(0.5)
    manifest["path"] = raster_path.name
    manifest.to_csv(tmp_path / "raw" / "manifest.csv", index=False)
    pd.testing.assert_frame_equal(result, pipeline.run_local(tmp_path / "raw" / "manifest.csv"))


def test_missing_or_duplicate_local_analysis_rejected(tmp_path):
    with pytest.raises(ValueError, match="No local Planet"):
        build_local_planet_manifest(tmp_path)
    raster_path = make_delivery(tmp_path / "raw")
    raster_path.with_name("1154-1277_quad.tif").write_bytes(raster_path.read_bytes())
    with pytest.raises(ValueError, match="Expected one local analysis"):
        build_local_planet_manifest(tmp_path)


class FakeS3:
    def __init__(self, source):
        self.files = {}
        self.downloads = []
        root = "PL_BSM/PL_BSM_JANUARY_2025/order-1/"
        entries = []
        for path in source.iterdir():
            data = path.read_bytes()
            relative = f"mosaic/{path.name}"
            self.files[root + relative] = data
            entries.append(
                {
                    "path": relative,
                    "size": len(data),
                    "digests": {"sha256": hashlib.sha256(data).hexdigest()},
                    "annotations": {"planet/quad_id": "1154-1277"},
                }
            )
        self.files[root + "manifest.json"] = json.dumps({"files": entries}).encode()

    def get_paginator(self, name):
        return self

    def paginate(self, Bucket, Prefix):
        yield {
            "Contents": [
                {"Key": key, "Size": len(value), "ETag": '"fake-etag"'}
                for key, value in self.files.items()
                if key.startswith(Prefix)
            ]
        }

    def get_object(self, Bucket, Key):
        return {"Body": BytesIO(self.files[Key])}

    def download_file(self, bucket, key, path, Config):
        self.downloads.append(key)
        Path(path).write_bytes(self.files[key])


def test_manifest_download_checksum_and_resume(tmp_path):
    make_delivery(tmp_path / "source")
    client = FakeS3(tmp_path / "source")
    inventory = discover_planet_delivery_tiles(client, "bucket", ["1154-1277"], "2025-01-01", "2025-01-31")
    manifest = download_planet_delivery_tiles(client, "bucket", inventory, tmp_path / "download")
    assert len(manifest) == 1 and len(client.downloads) == 2
    download_planet_delivery_tiles(client, "bucket", inventory, tmp_path / "download")
    assert len(client.downloads) == 2
    Path(manifest.iloc[0].path).write_bytes(b"corrupt")
    download_planet_delivery_tiles(client, "bucket", inventory, tmp_path / "download")
    assert len(client.downloads) == 3
    # Refreshing January must not inspect unfinished February imagery nearby.
    partial = tmp_path / "download" / "partial_february"
    partial.mkdir()
    (partial / "1154-1277_metadata.json").write_text(
        json.dumps(
            {
                "quad": {"id": "1154-1277"},
                "mosaic": {"first_acquired": "2025-02-01T00:00:00Z", "last_acquired": "2025-03-01T00:00:00Z"},
            }
        )
    )
    assert len(download_planet_delivery_tiles(client, "bucket", inventory, tmp_path / "download")) == 1


def test_bad_checksum_never_publishes_tiff(tmp_path):
    make_delivery(tmp_path / "source")
    client = FakeS3(tmp_path / "source")
    inventory = discover_planet_delivery_tiles(client, "bucket", ["1154-1277"], "2025-01-01", "2025-01-31")
    inventory.loc[inventory.filename.str.endswith(".tif"), "sha256"] = "bad-checksum"
    with pytest.raises(ValueError, match="does not match"):
        download_planet_delivery_tiles(client, "bucket", inventory, tmp_path / "download")
    assert not list((tmp_path / "download").rglob("*_quad_bandmath.tif"))


def test_incomplete_order_rejected(tmp_path):
    make_delivery(tmp_path / "source")
    client = FakeS3(tmp_path / "source")
    client.files = {key: value for key, value in client.files.items() if not key.endswith("manifest.json")}
    with pytest.raises(ValueError, match="Expected one analysis"):
        discover_planet_delivery_tiles(client, "bucket", ["1154-1277"], "2025-01-01", "2025-01-31")


def test_submicrometre_seam_is_tolerated_but_real_gap_rejected():
    from hub_catalog.planet_parcel_stats import _require_coverage

    parcels = gpd.GeoDataFrame(geometry=[box(0.5, 0.2, 1.5, 0.8)], crs=2100)
    tiles = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1), box(1 + 1e-8, 0, 2, 1)], crs=2100)
    _require_coverage(tiles, parcels, "2025-01-01")
    tiles.geometry = [box(0, 0, 1, 1), box(1.01, 0, 2, 1)]
    with pytest.raises(ValueError, match="footprints do not cover"):
        _require_coverage(tiles, parcels, "2025-01-01")


def test_local_discovery_ignores_unrequested_partial_month(tmp_path):
    make_delivery(tmp_path / "complete")
    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "1154-1277_metadata.json").write_text(
        json.dumps(
            {
                "quad": {"id": "1154-1277"},
                "mosaic": {"first_acquired": "2025-02-01T00:00:00Z", "last_acquired": "2025-03-01T00:00:00Z"},
            }
        )
    )
    manifest = build_local_planet_manifest(tmp_path, start_date="2025-01-01", end_date="2025-01-31")
    assert len(manifest) == 1
