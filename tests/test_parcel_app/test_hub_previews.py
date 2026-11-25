"""HUB inspection previews: display grid, alignment, masks, mosaics and acquisitions, without remote calls."""

import io
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from affine import Affine
from PIL import Image
from pyproj import Transformer
from rasterio.io import MemoryFile
from shapely.geometry import box, mapping

from data_preparation.sources.hub.ard_items import ArdAsset, ArdItem, Observation
from parcel_app.sources.hub import preview_raster, previews

# Source lattice deliberately not on 10 m multiples, as observed in HUB ARD rasters.
ORIGIN = (478_703.0, 4_200_307.0)
SIZE = 160  # covers the parcel plus the 500 m preview buffer
PARCEL_2100 = box(479_403.0, 4_199_407.0, 479_603.0, 4_199_607.0)
CLOUD_ROWS = slice(70, 90)


@pytest.fixture
def parcel():
    """Return the WGS84 GeoJSON feature of a 200 m square parcel."""
    geometry = gpd.GeoSeries([PARCEL_2100], crs=2100).to_crs(4326).iloc[0]
    return {"type": "Feature", "properties": {}, "geometry": mapping(geometry)}


@pytest.fixture(autouse=True)
def _fresh_caches():
    """Isolate cached acquisitions and windows between tests."""
    previews._observations.cache_clear()
    previews._WINDOWS._entries.clear()
    previews._WINDOWS._bytes = 0
    yield
    previews._observations.cache_clear()


def _tif(array, nodata, origin=ORIGIN, crs="EPSG:2100", resolution=10.0):
    """Return an in-memory GeoTIFF of one band, EPSG:2100 at 10 m unless overridden."""
    memory = MemoryFile()
    transform = Affine(resolution, 0.0, origin[0], 0.0, -resolution, origin[1])
    profile = {"driver": "GTiff", "width": SIZE, "height": SIZE, "count": 1, "dtype": array.dtype.name,
               "crs": crs, "transform": transform, "nodata": nodata}  # fmt: skip
    with memory.open(**profile) as dataset:
        dataset.write(array, 1)
    return memory


class FakeResolver:
    """Open in-memory assets by href, as ``HubAssetResolver.open`` does for COGs."""

    def __init__(self):
        """Start without assets."""
        self.files, self.opened = {}, []

    @contextmanager
    def open(self, href):
        """Yield the dataset of ``href``."""
        self.opened.append(href)
        with self.files[href].open() as dataset:
            yield dataset


def _sentinel2_item(resolver, item_id, values, *, origin=ORIGIN, scl=None, mask=None, checksum=None, grid=None):
    """Register one Sentinel-2 item whose reflectance bands are given per band name.

    ``checksum`` is given to every asset; ``grid`` maps a band to ``_tif`` overrides (CRS, resolution).
    """
    nodata = float(np.finfo("float32").max)
    scl = np.full((SIZE, SIZE), 4, dtype="uint8") if scl is None else scl
    mask = np.ones((SIZE, SIZE), dtype="uint8") if mask is None else mask
    arrays = {**{name: np.asarray(value, dtype="float32") for name, value in values.items()}, "SCL": scl, "MASK": mask}
    assets = {}
    for key, array in arrays.items():
        href = f"https://hub.test/{item_id}/{key}.tif"
        asset_nodata = {"SCL": 255, "MASK": 0}.get(key, nodata)
        resolver.files[href] = _tif(array, asset_nodata, origin, **(grid or {}).get(key, {}))
        assets[key] = ArdAsset(key, href, checksum, asset_nodata, (10.0, 0.0, origin[0], 0.0, -10.0, origin[1]), (SIZE, SIZE))
    return ArdItem("ARD_S2", item_id, pd.Timestamp("2025-07-01T09:00Z"), "sentinel-2a", None, None, None, "dt1",
                   12.0, 0.0, "2025-07-02T00:00:00Z", "v1", 2100, assets, None)  # fmt: skip


def _bands(value=0.1, nir=0.5):
    """Return constant reflectance bands for RGB and NDVI."""
    full = lambda number: np.full((SIZE, SIZE), number, dtype="float32")
    return {"B04": full(value), "B03": full(value), "B02": full(value), "B08": full(nir)}


def test_grid_has_fixed_ground_resolution_and_matches_the_bounds(parcel):
    """Catch per-parcel resampling, a grid missing the buffer, or bounds differing from the image."""
    grid = preview_raster.display_grid(previews._parcel_geometry(parcel))
    to_wgs84 = Transformer.from_crs(3857, 4326, always_xy=True)
    latitude = np.radians(to_wgs84.transform(0, grid.transform.f)[1])
    assert grid.transform.a * np.cos(latitude) == pytest.approx(2.5, rel=1e-3)
    assert 470 <= grid.width <= 490 and 470 <= grid.height <= 490  # 1200 m at 2.5 m, plus snapping
    south_west, north_east = previews.parcel_preview_bounds(parcel)
    left, bottom, right, top = grid.mercator_bounds()
    assert to_wgs84.transform(left, bottom) == pytest.approx((south_west[1], south_west[0]))
    assert to_wgs84.transform(right, top) == pytest.approx((north_east[1], north_east[0]))
    buffer = PARCEL_2100.buffer(preview_raster.BUFFER_METRES, join_style="mitre")
    buffered = gpd.GeoSeries([buffer], crs=2100).to_crs(3857).iloc[0]
    assert box(left, bottom, right, top).contains(buffered)


def test_oversized_parcel_is_refused():
    """Catch silently degrading resolution for a parcel wider than the source window limit."""
    large = gpd.GeoSeries([box(400_000, 4_100_000, 406_000, 4_100_100)], crs=2100).to_crs(4326).iloc[0]
    with pytest.raises(ValueError, match="Error: The parcel is too large"):
        preview_raster.display_grid(large)


def test_source_pixels_keep_their_true_position(parcel):
    """Catch shifted or stretched imagery: each display pixel must show the source pixel under it."""
    resolver = FakeResolver()
    columns = np.tile(np.arange(SIZE, dtype="float32"), (SIZE, 1)) / 1000
    item = _sentinel2_item(resolver, "a", {"B04": columns, "B03": columns, "B02": columns})
    geometry = previews._parcel_geometry(parcel)
    bounds = preview_raster.parcel_source_bounds(geometry)
    with ThreadPoolExecutor(2) as pool:
        window = preview_raster.read_item_window(resolver, item, ("B04", "B03", "B02"), bounds, pool)
    grid = preview_raster.display_grid(geometry)
    red = preview_raster.mosaic([window], ("B04",), grid, masked=False)["B04"]
    to_greek = Transformer.from_crs(3857, 2100, always_xy=True)
    for row, column in ((3, 3), (50, 50), (grid.height - 4, grid.width - 4)):
        x, _y = to_greek.transform(*(grid.transform @ (column + 0.5, row + 0.5)))
        assert red[row, column] * 1000 == int((x - ORIGIN[0]) // 10)


def test_rgb_keeps_clouds_while_indices_mask_them(parcel):
    """Catch hiding clouds in RGB, or index values computed under clouds, shadows or their buffer."""
    resolver = FakeResolver()
    scl = np.full((SIZE, SIZE), 4, dtype="uint8")
    scl[CLOUD_ROWS] = 9
    item = _sentinel2_item(resolver, "a", _bands(), scl=scl)
    geometry = previews._parcel_geometry(parcel)
    bands = ("B04", "B03", "B02", "B08")
    with ThreadPoolExecutor(2) as pool:
        window = preview_raster.read_item_window(resolver, item, bands, preview_raster.parcel_source_bounds(geometry), pool)
    grid = preview_raster.display_grid(geometry)
    rgb = previews._image(previews.PreviewLayer("RGB", "RGB", "S2", "rgb", None), [window], grid)
    ndvi = previews._image(previews.PreviewLayer("NDVI", "NDVI", "S2", "index", None), [window], grid)
    assert (rgb[..., 3] == 255).all()
    assert 0 < (ndvi[..., 3] == 0).sum() < ndvi[..., 3].size
    assert (rgb[..., 3] == 255).sum() > (ndvi[..., 3] == 255).sum()


def test_nodata_is_transparent_in_every_layer(parcel):
    """Catch showing pixels outside the ARD data area."""
    resolver = FakeResolver()
    mask = np.ones((SIZE, SIZE), dtype="uint8")
    mask[:, :80] = 0
    item = _sentinel2_item(resolver, "a", _bands(), mask=mask)
    geometry = previews._parcel_geometry(parcel)
    with ThreadPoolExecutor(2) as pool:
        window = preview_raster.read_item_window(
            resolver, item, ("B04", "B03", "B02"), preview_raster.parcel_source_bounds(geometry), pool
        )
    assert np.isnan(window.bands["B04"]).any() and np.isfinite(window.bands["B04"]).any()
    rgb = previews._image(previews.PreviewLayer("RGB", "RGB", "S2", "rgb", None), [window],
                          preview_raster.display_grid(geometry))  # fmt: skip
    assert (rgb[..., 3] == 0).any() and (rgb[..., 3] == 255).any()


def test_mosaic_fills_from_the_next_item_in_priority_order(parcel):
    """Catch the lower-priority item overwriting valid pixels, or gaps it could have filled."""
    resolver = FakeResolver()
    first_values = _bands(0.1)
    first_values["B04"][:, 90:] = np.finfo("float32").max  # nodata on the eastern half
    first = _sentinel2_item(resolver, "first", first_values)
    second = _sentinel2_item(resolver, "second", _bands(0.2), origin=(ORIGIN[0] + 4, ORIGIN[1] - 3))
    geometry = previews._parcel_geometry(parcel)
    grid = preview_raster.display_grid(geometry)
    bounds = preview_raster.parcel_source_bounds(geometry)
    with ThreadPoolExecutor(2) as pool:
        windows = [preview_raster.read_item_window(resolver, item, ("B04",), bounds, pool) for item in (first, second)]
    red = preview_raster.mosaic(windows, ("B04",), grid, masked=False)["B04"]
    assert np.allclose(np.unique(red), [0.1, 0.2])
    assert np.isfinite(red).all()


def test_layers_follow_the_run_configuration():
    """Catch offering layers the run did not calculate, or missing the requested composites."""
    names = lambda processing: [layer.name for layer in previews.preview_layers(processing)]
    planet = ["PL_RGB", "PL_NDVI"]  # reference basemaps are offered whatever the run configured
    assert names({"sentinel2_indices": ["ndvi", "NDWI"]}) == ["RGB", "NDVI", "NDWI", *planet]
    assert names({"sentinel2_bands": [], "sentinel1_indices": ["RVI"]}) == ["S1_FALSE_COLOUR", "RVI", *planet]
    assert names({"sentinel2_bands": ["B04"], "sentinel1_bands": ["VV"]}) == ["RGB", "S1_FALSE_COLOUR", *planet]
    assert previews.preview_layers({})[-1].legend == previews.palettes.legend("NDVI")
    assert previews.preview_layers({"sentinel2_indices": ["NDVI"]})[1].legend["min"] == -1.0
    with pytest.raises(ValueError, match="Error: NDMI is not a preview layer"):
        previews._layer({"sentinel2_indices": ["NDVI"]}, "NDMI")


def _manifest(tmp_path, partition, rows):
    """Write one partition's HUB manifest: rows of item, processing datetime/version, asset key and checksum."""
    directory = tmp_path / f"partition_{partition}"
    directory.mkdir()
    columns = ["item_id", "processing_datetime", "processing_version", "asset_key", "asset_checksum"]
    frame = pd.DataFrame(rows, columns=columns)
    frame.insert(0, "collection", "ARD_S2")
    frame.to_parquet(directory / previews.MANIFEST_NAME, index=False)


def test_acquisitions_are_limited_to_unchanged_run_items(tmp_path, parcel, monkeypatch):
    """Catch showing newer, reprocessed or replaced-asset items, or duplicates across partitions."""
    resolver = FakeResolver()
    used = _sentinel2_item(resolver, "used", _bands())
    newer = _sentinel2_item(resolver, "newer", _bands())
    reprocessed = _sentinel2_item(resolver, "reprocessed", _bands())
    replaced = _sentinel2_item(resolver, "replaced", _bands(), checksum="new")
    _manifest(tmp_path, 1, [
        ["used", "2025-07-02T00:00:00Z", "v1", "B04", None],
        ["reprocessed", "2025-07-02T00:00:00Z", "v0", "B04", None],
        ["replaced", "2025-07-02T00:00:00Z", "v1", "B04", "old"],
    ])  # fmt: skip
    _manifest(tmp_path, 2, [["used", "2025-07-02T00:00:00Z", "v1", "B04", None]])
    monkeypatch.setattr(previews, "_catalog", lambda: object())
    monkeypatch.setattr(
        "data_preparation.sources.hub.ard_items.search_ard_items", lambda *args, **kwargs: [used, newer, reprocessed, replaced]
    )
    processing = {"start_date": "2025-07-01", "end_date": "2025-07-31", "sentinel2_indices": ["NDVI"]}
    found = previews.parcel_acquisitions(tmp_path, processing, parcel)
    assert [(item.id, item.datetime, item.sensor, item.cloud_cover) for item in found] == [
        ("s2|dt1", "2025-07-01T09:00:00Z", "S2", 12.0)
    ]
    assert len(found[0].revision) == 16
    observations = previews._parcel_observations(tmp_path, processing, previews._parcel_geometry(parcel))
    assert [item.item_id for item in observations[0].items] == ["used"]
    monkeypatch.setattr(previews, "_resolver", lambda: resolver)
    image = Image.open(io.BytesIO(previews.render_preview(tmp_path, processing, parcel, "s2|dt1", "NDVI")))
    grid = preview_raster.display_grid(previews._parcel_geometry(parcel))
    assert image.mode == "RGBA" and image.size == (grid.width, grid.height)
    opened = len(resolver.opened)
    previews.render_preview(tmp_path, processing, parcel, "s2|dt1", "RGB")
    assert len(resolver.opened) == opened  # layer switch served from the window cache


def test_render_refuses_unknown_or_mismatched_selections(tmp_path, parcel, monkeypatch):
    """Catch rendering an acquisition outside the run, or a layer of the other sensor."""
    _manifest(tmp_path, 1, [["used", "2025-07-02T00:00:00Z", "v1", "B04", None]])
    observation = Observation("ARD_S2", "s2|dt1", pd.Timestamp("2025-07-01T09:00Z"), ())
    monkeypatch.setattr(previews, "_parcel_observations", lambda *args: (observation,))
    processing = {"start_date": "2025-07-01", "end_date": "2025-07-31", "sentinel1_indices": ["RVI"]}
    with pytest.raises(ValueError, match="Error: This acquisition does not cover"):
        previews.render_preview(tmp_path, processing, parcel, "s2|other", "RGB")
    with pytest.raises(ValueError, match="not available for a S2 acquisition"):
        previews.render_preview(tmp_path, processing, parcel, "s2|dt1", "RVI")


def test_run_without_inventory_is_refused(tmp_path, parcel):
    """Catch a confusing failure for runs without HUB manifests."""
    with pytest.raises(ValueError, match="Error: This parcel statistics run has no HUB item inventory"):
        previews.parcel_acquisitions(tmp_path, {"start_date": "2025-07-01", "end_date": "2025-07-31"}, parcel)


def test_window_cache_is_bounded_by_bytes_and_entries():
    """Catch unbounded memory growth of cached band windows."""
    cache = previews.WindowCache(max_bytes=100, max_entries=2)
    cache.put("a", 1, 60)
    cache.put("b", 2, 30)
    assert cache.get("a") == 1  # "a" is now the most recent
    cache.put("c", 3, 30)  # over 100 bytes: evicts the least recent, "b"
    assert cache.get("b") is None and cache.get("a") == 1 and cache.get("c") == 3
    cache.put("d", 4, 10)  # over two entries: evicts "a"
    assert cache.get("a") is None
    cache.put("huge", 5, 1000)
    assert cache.get("huge") is None


@pytest.mark.parametrize(
    ("grid", "message"),
    [
        ({"B03": {"crs": "EPSG:32634"}}, "do not share one pixel grid"),
        ({key: {"resolution": 20.0} for key in ("B04", "B03", "B02", "SCL", "MASK")}, "north-up 10 m grid"),
    ],
)
def test_unsupported_asset_grids_are_refused(parcel, grid, message):
    """Catch applying one window and mask across assets in different CRSs, or to a non-native resolution."""
    resolver = FakeResolver()
    item = _sentinel2_item(resolver, "a", _bands(), grid=grid)
    geometry = previews._parcel_geometry(parcel)
    with ThreadPoolExecutor(2) as pool, pytest.raises(ValueError, match=message):
        preview_raster.read_item_window(
            resolver, item, ("B04", "B03", "B02"), preview_raster.parcel_source_bounds(geometry), pool
        )


# ---------------------------------------------------------------------------
# Planet monthly basemaps (PL_BSM)
# ---------------------------------------------------------------------------
PLANET_RESOLUTION = 4.77731426716  # deployed quad pixel size, EPSG:3857
PLANET_PAD = 20  # quad pixels beyond each side of the display grid


def _basemap_item(resolver, item_id, when, grid, *, gap_columns=0):
    """Register one 10-band quad covering ``grid``: bands 1-8 = band number x 100, NDVI 0.5, alpha 255.

    The first ``PLANET_PAD + gap_columns`` quad columns get alpha 0, as outside a mosaic's data area.
    """
    left, _bottom, right, top = grid.mercator_bounds()
    size = int(np.ceil((right - left) / PLANET_RESOLUTION)) + 2 * PLANET_PAD
    values = np.stack([np.full((size, size), band * 100.0, dtype="float32") for band in range(1, 9)])
    alpha = np.full((size, size), 255.0, dtype="float32")
    alpha[:, : PLANET_PAD + gap_columns] = 0.0
    values = np.concatenate([values, np.full((1, size, size), 0.5, dtype="float32"), alpha[None]])
    origin = (left - PLANET_PAD * PLANET_RESOLUTION, top + PLANET_PAD * PLANET_RESOLUTION)
    transform = Affine(PLANET_RESOLUTION, 0.0, origin[0], 0.0, -PLANET_RESOLUTION, origin[1])
    memory = MemoryFile()
    with memory.open(driver="GTiff", width=size, height=size, count=10, dtype="float32", crs="EPSG:3857",
                     transform=transform) as dataset:  # fmt: skip
        dataset.write(values)
    href = f"https://hub.test/{item_id}/quad_bandmath.tif"
    resolver.files[href] = memory
    asset = ArdAsset("quad_bandmath", href, f"sum-{item_id}", None, tuple(transform)[:6], (size, size))
    return ArdItem("PL_BSM", item_id, pd.Timestamp(when), None, None, None, None, None, None, None, None, None, 4326,
                   {"quad_bandmath": asset}, None)  # fmt: skip


def test_basemap_layers_use_bands_6_4_2_and_band_9_and_hide_alpha_gaps(parcel):
    """Catch wrong band order, an unscaled reflectance stretch, or showing pixels outside the mosaic data."""
    resolver = FakeResolver()
    grid = preview_raster.display_grid(previews._parcel_geometry(parcel))
    item = _basemap_item(resolver, "quad", "2025-07-01T00:00:00Z", grid, gap_columns=60)
    window = preview_raster.read_basemap_window(resolver, item, grid.mercator_bounds())
    rgb = previews._image(previews._layer({}, "PL_RGB"), [window], grid)
    ndvi = previews._image(previews._layer({}, "PL_NDVI"), [window], grid)
    shown = rgb[..., 3] == 255
    assert 0 < shown.sum() < shown.size and not shown[:, 0].any() and shown[:, -1].all()
    assert np.array_equal(ndvi[..., 3] == 255, shown)
    # Reflectance x 10000 on the 0-0.3 stretch: band 6 = 0.06 (red), band 4 = 0.04 (green), band 2 = 0.02 (blue).
    assert {tuple(pixel) for pixel in rgb[shown][:, :3]} == {(51, 34, 17)}
    assert {tuple(pixel) for pixel in ndvi[shown][:, :3]} == {previews.palettes._hex_channels("#78c679")}
    far = tuple(value + 50_000 for value in grid.mercator_bounds())
    assert preview_raster.read_basemap_window(resolver, item, far) is None


def test_basemap_months_are_listed_without_manifest_entries_and_rendered(tmp_path, parcel, monkeypatch):
    """Catch dropping basemaps absent from the run's manifest, missing the start month, or mixing sensors."""
    resolver = FakeResolver()
    grid = preview_raster.display_grid(previews._parcel_geometry(parcel))
    found_items = {
        "ARD_S2": [_sentinel2_item(resolver, "used", _bands())],
        "PL_BSM": [_basemap_item(resolver, "jul_a", "2025-07-01T00:00:00Z", grid),
                   _basemap_item(resolver, "jul_b", "2025-07-01T00:00:00Z", grid),
                   _basemap_item(resolver, "aug", "2025-08-01T00:00:00Z", grid)],
    }  # fmt: skip
    _manifest(tmp_path, 1, [["used", "2025-07-02T00:00:00Z", "v1", "B04", None]])
    searched = {}

    def search(_catalog, collection, _geometry, start, end, **_kwargs):
        """Record each collection's date range and return its prepared items."""
        searched[collection] = (pd.Timestamp(start), pd.Timestamp(end))
        return found_items.get(collection, [])

    monkeypatch.setattr(previews, "_catalog", lambda: object())
    monkeypatch.setattr("data_preparation.sources.hub.ard_items.search_ard_items", search)
    processing = {"start_date": "2025-07-15", "end_date": "2025-08-31"}
    found = previews.parcel_acquisitions(tmp_path, processing, parcel)
    assert [(item.id, item.sensor) for item in found] == [("pl|2025-07", "PL"), ("s2|dt1", "S2"), ("pl|2025-08", "PL")]
    # The July basemap is dated 1 July, before the run's first day.
    assert searched["PL_BSM"] == (pd.Timestamp("2025-07-01"), pd.Timestamp("2025-08-31"))
    observations = previews._parcel_observations(tmp_path, processing, previews._parcel_geometry(parcel))
    assert [item.item_id for item in observations[0].items] == ["jul_a", "jul_b"]
    monkeypatch.setattr(previews, "_resolver", lambda: resolver)
    image = Image.open(io.BytesIO(previews.render_preview(tmp_path, processing, parcel, "pl|2025-07", "PL_NDVI")))
    assert image.mode == "RGBA" and image.size == (grid.width, grid.height)
    with pytest.raises(ValueError, match="not available for a S2 acquisition"):
        previews.render_preview(tmp_path, processing, parcel, "s2|dt1", "PL_RGB")
