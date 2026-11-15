"""Exercise DEM mosaics and parcel elevation using tiny real raster fixtures."""

from pathlib import Path
from unittest.mock import Mock

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from geopandas.testing import assert_geodataframe_equal
from rasterio.transform import from_origin
from shapely.geometry import Polygon, box

from data_preparation.features import elevation as el
from shared.io import read_data, write_data


@pytest.fixture
def raster_factory(tmp_path):
    """Write small rasters with explicit CRS, nodata, and optional adjacent origins."""

    def create(name="DEM_a.tif", *, values=None, crs="EPSG:3857", count=1, dtype="float32", x=0):
        """Create one test raster and return its path."""
        values = np.asarray([[10, 20], [30, -9999]] if values is None else values, dtype=dtype)
        path = tmp_path / name
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=2,
            height=2,
            count=count,
            dtype=dtype,
            crs=crs,
            transform=from_origin(x, 2, 1, 1),
            nodata=-9999,
        ) as dst:
            for band in range(1, count + 1):
                dst.write(values, band)
        return path

    return create


@pytest.fixture
def parcel_inputs(tmp_path, raster_factory):
    """Provide intersecting, outside, and missing parcels with unsorted identifiers."""
    dem = raster_factory()
    parcels = gpd.GeoDataFrame({"parcel_id": [30, 10, 20]}, geometry=[box(0, 0, 2, 2), box(5, 5, 6, 6), None], crs=3857)
    path = tmp_path / "parcels.geoparquet"
    write_data(parcels, str(path))
    return parcels, path, dem


def test_tile_discovery_is_sorted_and_excludes_mosaic_and_directories(tmp_path):
    """Only matching DEM files are inputs; an existing output is never merged into itself."""
    for name in ("DEM_b.tif", "DEM_a.tif", "DEM_mosaic.tif", "other.tif"):
        (tmp_path / name).touch()
    (tmp_path / "DEM_directory.tif").mkdir()
    assert el.find_dem_tiles(tmp_path, tmp_path / "DEM_mosaic.tif") == [tmp_path / "DEM_a.tif", tmp_path / "DEM_b.tif"]
    with pytest.raises(FileNotFoundError, match="DEM directory"):
        el.find_dem_tiles(tmp_path / "missing", tmp_path / "out.tif")


def test_mosaic_empty_single_and_reuse_cases(tmp_path, raster_factory):
    """Empty input fails, one tile is reused, and existing mosaics are preserved by default."""
    first = raster_factory()
    output = tmp_path / "mosaic.tif"
    with pytest.raises(ValueError, match="No DEM tiles"):
        el.create_dem_mosaic([], output)
    assert el.create_dem_mosaic([first], output) == first
    assert not output.exists()
    output.write_bytes(b"existing mosaic")
    assert el.create_dem_mosaic([first, tmp_path / "unused.tif"], output) == output
    assert output.read_bytes() == b"existing mosaic"


@pytest.mark.parametrize(
    "bad_options,first,message",
    [
        ({"crs": None}, True, "no CRS"),
        ({"count": 2}, True, "one band"),
        ({"count": 2}, False, "one band"),
        ({"crs": "EPSG:4326"}, False, "CRS mismatch"),
        ({"dtype": "int16"}, False, "dtype mismatch"),
    ],
)
def test_mosaic_rejects_incompatible_tiles(tmp_path, raster_factory, bad_options, first, message):
    """Merging incompatible CRS, band counts, or data types fails before publishing output."""
    good, bad = raster_factory(), raster_factory("DEM_bad.tif", **bad_options)
    output = tmp_path / "mosaic.tif"
    with pytest.raises(ValueError, match=message):
        el.create_dem_mosaic([bad, good] if first else [good, bad], output)
    assert not output.exists()


def test_mosaic_merges_adjacent_tiles_and_overwrites_when_requested(tmp_path, raster_factory):
    """Actual raster merging preserves location, values, nodata, and the declared float dtype."""
    first = raster_factory()
    second = raster_factory("DEM_b.tif", values=[[40, 50], [60, 70]], x=2)
    output = tmp_path / "nested" / "mosaic.tif"
    output.parent.mkdir()
    output.write_bytes(b"old mosaic")
    assert el.create_dem_mosaic([first, second], output, overwrite=True) == output
    with rasterio.open(output) as result:
        np.testing.assert_array_equal(result.read(1), [[10, 20, 40, 50], [30, -9999, 60, 70]])
        assert result.crs.to_epsg() == 3857
        assert result.nodata == -9999
        assert result.dtypes == ("float32",)


def test_mosaic_closes_all_sources_if_merge_fails(tmp_path, raster_factory, monkeypatch):
    """A failing merge releases file handles so a subsequent attempt can reopen the tiles."""
    paths = [raster_factory(), raster_factory("DEM_b.tif", x=2)]
    merger = Mock(side_effect=OSError("disk full"))
    monkeypatch.setattr(el, "merge", merger)
    with pytest.raises(OSError, match="disk full"):
        el.create_dem_mosaic(paths, tmp_path / "mosaic.tif")
    assert all(source.closed for source in merger.call_args.args[0])


@pytest.mark.parametrize("geometry", [None, Polygon(), box(9, 9, 10, 10), box(1, 0, 2, 1)])
def test_zonal_mean_without_valid_pixels_is_missing(raster_factory, geometry):
    """Absent geometries, outside parcels, and nodata-only parcels yield NaN and zero pixels."""
    with rasterio.open(raster_factory()) as dem:
        mean, count = el._mean_elevation_for_geometry(dem, geometry, all_touched=False)
    assert np.isnan(mean)
    assert count == 0


def test_all_touched_includes_pixels_without_contained_centers(raster_factory):
    """A small parcel touching one pixel contributes only when all_touched is enabled."""
    with rasterio.open(raster_factory()) as dem:
        mean, count = el._mean_elevation_for_geometry(dem, box(0.01, 1.01, 0.1, 1.1), all_touched=False)
        assert np.isnan(mean) and count == 0
        assert el._mean_elevation_for_geometry(dem, box(0.01, 1.01, 0.1, 1.1), all_touched=True) == (10, 1)


@pytest.mark.parametrize("reproject", [False, True])
def test_append_preserves_parcel_order_geometry_and_crs(tmp_path, parcel_inputs, reproject):
    """Published features ignore nodata and keep original parcels even when sampling another CRS."""
    parcels, path, dem = parcel_inputs
    if reproject:
        parcels = parcels.to_crs(3395)
        write_data(parcels, str(path))
    output = tmp_path / "result" / "features.geoparquet"
    result = el.append_parcel_elevation(path, dem, output)
    assert_geodataframe_equal(result.drop(columns=[el.ELEVATION_COLUMN, el.PIXEL_COUNT_COLUMN]), parcels)
    assert_geodataframe_equal(read_data(str(output)), result)
    np.testing.assert_allclose(result[el.ELEVATION_COLUMN], [20, np.nan, np.nan], equal_nan=True)
    assert result[el.PIXEL_COUNT_COLUMN].tolist() == [3, 0, 0]
    assert str(result[el.PIXEL_COUNT_COLUMN].dtype) == "int32"
    assert_geodataframe_equal(read_data(str(path)), parcels)
    assert not list(output.parent.glob(".*.tmp.*"))


def test_failed_atomic_write_preserves_existing_output(parcel_inputs, monkeypatch):
    """An interrupted in-place write retains the original bytes and removes the partial file."""
    _, path, dem = parcel_inputs
    before = path.read_bytes()

    def fail_write(frame, temporary):
        """Simulate an interrupted write after producing a partial file."""
        Path(temporary).write_bytes(b"partial")
        raise OSError("interrupted")

    monkeypatch.setattr(el, "write_data", fail_write)
    with pytest.raises(OSError, match="interrupted"):
        el.append_parcel_elevation(path, dem, path, overwrite=True)
    assert path.read_bytes() == before
    assert not list(path.parent.glob(".*.tmp.*"))


def test_append_rejects_missing_input_and_existing_output(tmp_path, parcel_inputs):
    """Path preflight prevents reading absent inputs or replacing output without overwrite."""
    _, path, dem = parcel_inputs
    with pytest.raises(FileNotFoundError, match="Parcel GeoParquet"):
        el.append_parcel_elevation(tmp_path / "missing", dem, path)
    with pytest.raises(FileExistsError, match="Output already exists"):
        el.append_parcel_elevation(path, dem, path)


@pytest.mark.parametrize(
    "case,message", [("missing_geometry", "no usable geometries"), ("missing_dem_crs", "DEM mosaic has no CRS")]
)
def test_append_rejects_unusable_spatial_metadata(tmp_path, parcel_inputs, raster_factory, case, message):
    """Spatial metadata failures leave no published output."""
    parcels, path, dem = parcel_inputs
    if case == "missing_geometry":
        parcels.geometry = [None] * len(parcels)
        write_data(parcels, str(path))
    else:
        dem = raster_factory("unreferenced.tif", crs=None)
    output = tmp_path / "out.geoparquet"
    with pytest.raises(ValueError, match=message):
        el.append_parcel_elevation(path, dem, output)
    assert not output.exists()
