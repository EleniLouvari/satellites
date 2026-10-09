"""Regression coverage for restored shared raster helpers."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.control import GroundControlPoint
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from tests.utils import expect_equal, expect_in, expect_true

pytest.importorskip("osgeo.gdal")
pytest.importorskip("botocore.exceptions")

from botocore.exceptions import ClientError, EndpointConnectionError

from shared import raster


def test_native_georeferencing_preserves_the_grid():
    """Reading an already georeferenced raster must preserve its grid exactly."""
    transform = Affine.translation(100, 200) * Affine.scale(10, -10)
    with MemoryFile() as memory:
        with memory.open(driver="GTiff", width=2, height=3, count=1, dtype="uint8", crs="EPSG:3857", transform=transform) as dst:
            dst.write(np.ones((1, 3, 2), dtype="uint8"))
        with memory.open() as source:
            expect_equal(raster._extract_source_georeferencing(source), (source.crs, transform))
            expect_equal(raster._extract_reference_georeferencing(source), (2, 3, source.crs, transform))


def test_gcps_recover_a_missing_raster_transform():
    """Ground control points supply both the CRS and affine grid."""
    crs = rasterio.crs.CRS.from_epsg(3857)
    source = SimpleNamespace(
        name="gcps.tif",
        crs=None,
        transform=Affine.identity(),
        width=2,
        height=3,
        gcps=(
            [
                GroundControlPoint(row=0, col=0, x=100, y=200),
                GroundControlPoint(row=0, col=2, x=120, y=200),
                GroundControlPoint(row=3, col=0, x=100, y=170),
            ],
            crs,
        ),
    )
    actual_crs, transform = raster._extract_source_georeferencing(source)
    expect_equal(actual_crs, crs)
    expect_true(transform.almost_equals(Affine(10, 0, 100, 0, -10, 200)))


def test_gdal_metadata_recovers_missing_georeferencing(monkeypatch):
    """GDAL metadata can supply a grid missing from the Rasterio reader."""
    crs = rasterio.crs.CRS.from_epsg(3857)
    source = SimpleNamespace(name="metadata.jp2", crs=None, transform=None, gcps=([], None), width=2, height=3)
    monkeypatch.setattr(
        raster.gdal,
        "Info",
        lambda *args, **kwargs: {"coordinateSystem": {"wkt": crs.to_wkt()}, "geoTransform": [100, 10, 0, 200, 0, -10]},
    )
    expect_equal(raster._extract_reference_georeferencing(source), (2, 3, crs, Affine(10, 0, 100, 0, -10, 200)))


def test_missing_reference_georeferencing_is_rejected(monkeypatch):
    """A reference raster without a grid must not silently place pixels at the origin."""
    source = SimpleNamespace(name="empty.tif", crs=None, transform=Affine.identity(), gcps=([], None), width=2, height=3)
    monkeypatch.setattr(raster.gdal, "Info", lambda *args, **kwargs: {})
    with pytest.raises(ValueError, match="Could not determine valid georeferencing"):
        raster._extract_reference_georeferencing(source)


def test_invalid_resampling_method_falls_back_without_a_logger(capsys):
    """The documented fallback works even when no logger was supplied."""
    expect_equal(raster.define_resampling_algorithm("invalid"), Resampling.nearest)
    expect_in("Falling back to 'nearest'", capsys.readouterr().out)


def test_gdal_capacity_handles_an_unknown_cpu_count(monkeypatch):
    """Subprocess options remain valid on systems without a reported CPU count."""
    monkeypatch.setattr(raster.os, "cpu_count", lambda: None)
    monkeypatch.setattr(raster.gdal, "GetCacheMax", lambda: 64 * 1024 * 1024)
    expect_equal(raster._get_gdal_capacity(), (64, 1))


def test_run_subprocess_captures_output(capsys):
    """The subprocess wrapper executes commands and exposes captured output."""
    raster.run_subprocess([sys.executable, "-c", "print('raster helper check')"])
    expect_in("STDOUT: raster helper check", capsys.readouterr().out)


def test_run_subprocess_preserves_failure_details():
    """A failed command becomes a chained RuntimeError."""
    with pytest.raises(RuntimeError, match="failed") as error:
        raster.run_subprocess([sys.executable, "-c", "raise SystemExit(3)"])
    expect_equal(error.value.__cause__.returncode, 3)


@pytest.mark.parametrize(
    "error,expected",
    [
        (EndpointConnectionError(endpoint_url="https://example.invalid"), True),
        (ClientError({"Error": {"Code": "SlowDown"}}, "GetObject"), True),
        (ClientError({"Error": {"Code": "AccessDenied"}}, "GetObject"), False),
        (ValueError("Error: invalid raster"), False),
    ],
)
def test_mosaic_retry_classification(error, expected):
    """Connection/transient errors are retried, while invalid access and data are not."""
    expect_true(raster._is_retryable_mosaic_s3_error(error) is expected)
