"""Offline tests for HUB ARD cube construction with synthetic local GeoTIFF assets."""

import dataclasses
import threading
import time
from contextlib import contextmanager
from unittest.mock import Mock

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
import xarray as xr
from affine import Affine
from shapely.geometry import box

from data_preparation.parcel_stats.core.base import ParcelStatsBase
from data_preparation.sources.hub import ard_items
from data_preparation.sources.hub import cubes as hub_cubes
from data_preparation.sources.hub.cubes import HubCubePipeline

NODATA_FLOAT = 3.4028234663852886e38
# Deployed S2 tile T34TFK lattice offsets (x mod 10, y mod 10) and a different S1 lattice.
S2_ORIGIN = (399502.7322, 4400509.0556)
S1_ORIGIN = (399505.0, 4400505.0)
SIZE = 100


class HubStats(ParcelStatsBase, HubCubePipeline):
    """Shared engine composed with the HUB cube mixin, as HubZonalStats will be."""


class LocalResolver:
    """Resolver stand-in that opens hrefs as local files and counts opens."""

    def __init__(self):
        self.opened = []

    @contextmanager
    def open(self, href):
        self.opened.append(href)
        with rasterio.open(href) as dataset:
            yield dataset


def write_raster(path, array, origin, nodata):
    """Write a single-band EPSG:2100 GeoTIFF on a 10 m grid."""
    transform = Affine(10, 0, origin[0], 0, -10, origin[1])
    profile = {
        "driver": "GTiff",
        "height": array.shape[0],
        "width": array.shape[1],
        "count": 1,
        "dtype": array.dtype,
        "crs": "EPSG:2100",
        "transform": transform,
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dataset:
        dataset.write(array, 1)
    return list(transform)[:6]


def make_item(tmp_path, collection, item_id, when, layers, origin, properties, footprint=None):
    """Write item layers to disk and return the normalised ArdItem pointing at them."""
    assets = {}
    for key, array in layers.items():
        nodata = NODATA_FLOAT if array.dtype == np.float32 else (255 if key == "SCL" else 0)
        path = tmp_path / f"{item_id}_{key}.tif"
        transform = write_raster(path, array, origin, nodata)
        assets[key] = {
            "href": str(path),
            "hub:checksum": f"{item_id}-{key}",
            "raster:bands": [{"nodata": nodata}],
            "proj:transform": transform,
            "proj:shape": list(array.shape),
        }
    feature = {
        "id": item_id,
        "properties": {"datetime": when, "processing:datetime": "2026-11-10T00:00:00Z", **properties},
        "assets": assets,
    }
    if footprint is not None:
        # Footprints are given in EPSG:2100 for readability and stored as WGS84, as in STAC.
        feature["geometry"] = gpd.GeoSeries([footprint], crs=2100).to_crs(4326).iloc[0].__geo_interface__
    return ard_items.normalise_item(feature, collection)


def s2_layers(b04, b08, scl=4, mask=1):
    """Return constant Sentinel-2 layers that tests then edit."""
    return {
        "B04": np.full((SIZE, SIZE), b04, dtype=np.float32),
        "B08": np.full((SIZE, SIZE), b08, dtype=np.float32),
        "SCL": np.full((SIZE, SIZE), scl, dtype=np.uint8),
        "MASK": np.full((SIZE, SIZE), mask, dtype=np.uint8),
    }


def s1_layers(vv=0.05, vh=0.01):
    """Return constant Sentinel-1 layers that tests then edit."""
    return {
        "VV": np.full((SIZE, SIZE), vv, dtype=np.float32),
        "VH": np.full((SIZE, SIZE), vh, dtype=np.float32),
        "MASK": np.ones((SIZE, SIZE), dtype=np.uint8),
    }


@pytest.fixture
def stats_factory(tmp_path, monkeypatch):
    """Build a HubStats instance over one 100 m square parcel in EPSG:2100."""
    monkeypatch.setattr(hub_cubes, "USERS", {"internal": "secret"})

    def build(**overrides):
        parcels = gpd.GeoDataFrame({"parcel_id": ["p1"]}, geometry=[box(400000, 4400000, 400100, 4400100)], crs=2100)
        options = {
            "parcels": parcels,
            "start_date": "2024-06-01",
            "end_date": "2024-06-30",
            "output_dir": tmp_path / "out",
            "working_epsg": 2100,
            "sentinel2_bands": ["B04", "B08"],
            "sentinel2_indices": ["NDVI"],
            "sentinel1_bands": ["VV", "VH"],
            "sentinel1_indices": [],
            "temporal_reducer": "none",
            "spatial_statistics": ["mean", "median"],
        }
        stats = HubStats(**(options | overrides))
        stats._configure_hub_source(catalog_url="https://catalog.test", client_id="internal", read_workers=2, block_size=64)
        return stats

    return build


def test_snap_grid_uses_the_given_lattice_and_covers_bounds():
    """Grid corners lie on the lattice and the grid covers the requested bounds."""
    transform, width, height = hub_cubes.snap_grid((400000, 4400000, 400100, 4400100), S2_ORIGIN, 10)
    assert (transform.c - S2_ORIGIN[0]) % 10 == pytest.approx(0, abs=1e-6)
    assert (transform.f - S2_ORIGIN[1]) % 10 == pytest.approx(0, abs=1e-6)
    assert transform.c <= 400000 and transform.c + 10 * width >= 400100
    assert transform.f >= 4400100 and transform.f - 10 * height <= 4400000
    with pytest.raises(ValueError, match="positive"):
        hub_cubes.snap_grid((1, 1, 1, 2), (0, 0), 10)


def test_pixel_rules_mask_dilate_and_reject_nonpositive_radar():
    """SCL invalid classes, dilated clouds, ARD nodata and non-positive gamma0 become NaN."""
    scl = np.full((5, 5), 4, dtype=np.uint8)
    scl[2, 2], scl[0, 4] = 9, 1
    mask = np.ones((5, 5), dtype=np.uint8)
    mask[4, 0] = 0
    invalid = hub_cubes.sentinel2_invalid(
        scl, mask, ParcelStatsBase.SENTINEL2_INVALID_SCL_CLASSES, ParcelStatsBase.SENTINEL2_BUFFERED_SCL_CLASSES
    )
    assert invalid[1:4, 1:4].all() and invalid[0, 4] and invalid[4, 0]
    assert invalid.sum() == 11
    values = np.array([[0.1, NODATA_FLOAT, -0.001, 0.0]], dtype=np.float32)
    radar = hub_cubes.clean_values(values, NODATA_FLOAT, np.zeros(values.shape, bool), positive_only=True)
    assert np.isfinite(radar).tolist() == [[True, False, False, False]]
    optical = hub_cubes.clean_values(values, NODATA_FLOAT, np.zeros(values.shape, bool), positive_only=False)
    assert np.isfinite(optical).tolist() == [[True, False, True, True]]
    assert hub_cubes.sentinel1_invalid(np.array([0, 1, 2])).tolist() == [True, False, True]


def test_mosaic_takes_whole_pixels_and_reducers_keep_all_nan_pixels():
    """A pixel never combines bands from different scenes; empty pixels stay NaN after reduction."""
    first = {"B04": np.array([[0.1, 0.1]], np.float32), "B08": np.array([[np.nan, 0.5]], np.float32)}
    second = {"B04": np.array([[0.2, 0.2]], np.float32), "B08": np.array([[0.6, 0.6]], np.float32)}
    mosaic = hub_cubes.mosaic_first_valid([first, second], ("B04", "B08"), (1, 2))
    assert mosaic["B04"][0].tolist() == pytest.approx([0.2, 0.1])
    assert mosaic["B08"][0].tolist() == pytest.approx([0.6, 0.5])
    stack = [np.array([[0.1, np.nan]], np.float32), np.array([[0.3, np.nan]], np.float32)]
    assert hub_cubes.reduce_observations(stack, "median", (1, 2))[0, 0] == pytest.approx(0.2)
    assert np.isnan(hub_cubes.reduce_observations(stack, "sum", (1, 2))[0, 1])
    assert np.isnan(hub_cubes.reduce_observations([], "mean", (1, 2))).all()


def test_cube_mosaics_masks_aligns_and_feeds_the_shared_engine(tmp_path, stats_factory):
    """End to end: HUB-style assets become a cube that the existing local engine accepts."""
    stats = stats_factory()
    # Datatake D1: complete-but-partly-empty product A plus lower-priority product B.
    a = s2_layers(0.1, 0.5)
    a["B04"][:, :50] = NODATA_FLOAT  # left half missing in A
    a["MASK"][:, :50] = 0
    a["B08"][:, 60] = NODATA_FLOAT  # one column with B04 but no B08 in A
    b = s2_layers(0.2, 0.6)
    item_a = make_item(
        tmp_path, "ARD_S2", "A", "2024-06-05T09:15:59Z", a, S2_ORIGIN, {"s2:datatake_id": "D1", "statistics:nodata_pixel": 1.0}
    )
    item_b = make_item(
        tmp_path, "ARD_S2", "B", "2024-06-05T09:15:59Z", b, S2_ORIGIN, {"s2:datatake_id": "D1", "statistics:nodata_pixel": 80.0}
    )
    # Datatake D2: one cloud pixel to check 3x3 dilation.
    c = s2_layers(0.3, 0.7)
    c["SCL"][45, 54] = 9  # inside the parcel; the no-fill grid is only parcel + read halo
    item_c = make_item(tmp_path, "ARD_S2", "C", "2024-06-15T09:15:59Z", c, S2_ORIGIN, {"s2:datatake_id": "D2"})
    radar = s1_layers()
    radar["MASK"][40:50, 40:50] = 2  # layover/shadow block
    radar["VH"][20, 20] = -0.0007  # thermal-noise artefact
    item_s1 = make_item(
        tmp_path, "ARD_S1", "R", "2024-06-09T16:24:34Z", radar, S1_ORIGIN, {"platform": "sentinel-1a", "sat:absolute_orbit": 1}
    )
    observations = ard_items.group_observations([item_b, item_s1, item_c, item_a])
    path = tmp_path / "batch_00001_monthly.nc"

    coverage = stats._write_hub_cube(LocalResolver(), 1, stats.parcels, observations, path)

    assert path.exists() and not path.with_suffix(".partial.nc").exists()
    assert {row["collection"]: row["observations"] for row in coverage if row["time_label"].startswith("2024-06-05")} == {
        "ARD_S2": 1,
        "ARD_S1": 0,
    }
    with xr.open_dataset(path, decode_coords="all") as cube:
        assert cube.rio.crs.to_epsg() == 2100
        assert list(cube.data_vars) == ["B04", "B08", "VV", "VH"]
        assert (float(cube.x[0]) - 5 - S2_ORIGIN[0]) % 10 == pytest.approx(0, abs=1e-4)
        assert pd.to_datetime(cube.t.values).strftime("%Y-%m-%d").tolist() == ["2024-06-05", "2024-06-09", "2024-06-15"]
        first = cube.isel(t=0)

        def index(coordinates, value):
            """Return the exact grid index of a source pixel centre; never clamp to the edge."""
            position = int(np.argmin(np.abs(coordinates - value)))
            assert abs(coordinates[position] - value) < 1e-3, "sample pixel lies outside the cube"
            return position

        column = lambda x: index(cube.x.values, S2_ORIGIN[0] + 10 * x + 5)
        row = index(cube.y.values, S2_ORIGIN[1] - 10 * 45 - 5)
        assert float(first.B04[row, column(48)]) == pytest.approx(0.2)  # A empty → B
        assert float(first.B04[row, column(58)]) == pytest.approx(0.1)  # A valid
        assert float(first.B08[row, column(60)]) == pytest.approx(0.6)  # whole pixel from B, not A's B04
        assert float(first.B04[row, column(60)]) == pytest.approx(0.2)
        assert np.isnan(first.VV).all()  # no radar at the optical timestamp
        cloudy = cube.isel(t=2).B04.values
        assert np.isnan(cloudy[row - 1 : row + 2, column(53) : column(55) + 1]).all()
        assert np.isfinite(cloudy[row, column(57)])
        vv, vh = cube.isel(t=1).VV.values, cube.isel(t=1).VH.values
        assert np.nanmax(np.abs(vv - 0.05)) < 1e-6 and np.nanmax(np.abs(vh - 0.01)) < 1e-6
        assert np.isnan(vv).any() and (np.isfinite(vv) == np.isfinite(vh)).all()

    result, _report = stats._calculate_local_statistics(path, stats.parcels, 1)
    # With temporal_reducer="none" the engine pools the three acquisitions into the June label.
    assert result["period_start"].tolist() == ["2024-06-01"]
    assert result["NDVI_mean"].notna().all() and result["VV_mean"].iloc[0] == pytest.approx(0.05, abs=1e-6)


def test_monthly_cube_reduces_observations_to_interval_labels(tmp_path, stats_factory):
    """With a temporal reducer, observations within one interval are reduced and labelled by its start."""
    stats = stats_factory(temporal_reducer="median", sentinel1_bands=[])
    items = [
        make_item(tmp_path, "ARD_S2", name, when, s2_layers(value, 0.5), S2_ORIGIN, {"s2:datatake_id": name})
        for name, when, value in (("X", "2024-06-03T09:00:00Z", 0.1), ("Y", "2024-06-13T09:00:00Z", 0.3))
    ]
    path = tmp_path / "batch_00001_monthly.nc"
    stats._write_hub_cube(LocalResolver(), 1, stats.parcels, ard_items.group_observations(items), path)
    with xr.open_dataset(path) as cube:
        assert pd.to_datetime(cube.t.values).strftime("%Y-%m-%d").tolist() == ["2024-06-01"]
        assert float(cube.B04.isel(t=0).median()) == pytest.approx(0.2)
    result, _report = stats._calculate_local_statistics(path, stats.parcels, 1)
    assert result["B04_median"].iloc[0] == pytest.approx(0.2)


def test_resolve_manifest_fails_loudly_when_a_requested_sensor_is_missing(tmp_path, stats_factory, monkeypatch):
    """A batch with S2 items but no S1 items must not silently produce all-NaN radar columns."""
    stats = stats_factory()
    s2 = make_item(tmp_path, "ARD_S2", "A", "2024-06-05T09:15:59Z", s2_layers(0.1, 0.5), S2_ORIGIN, {"s2:datatake_id": "D1"})
    monkeypatch.setattr(hub_cubes, "CatalogSearchUtils", Mock())
    monkeypatch.setattr(
        hub_cubes.ard_items, "search_ard_items", lambda _c, collection, *_a, **_k: [s2] if collection == "ARD_S2" else []
    )
    with pytest.raises(ValueError, match="batch 1: ARD_S1"):
        stats._resolve_hub_manifest()


def test_resolve_manifest_records_digest_and_signature(tmp_path, stats_factory, monkeypatch):
    """Discovery stores batches, observations and a digest used by the HUB signature."""
    stats = stats_factory(sentinel1_bands=[])
    s2 = make_item(tmp_path, "ARD_S2", "A", "2024-06-05T09:15:59Z", s2_layers(0.1, 0.5), S2_ORIGIN, {"s2:datatake_id": "D1"})
    monkeypatch.setattr(hub_cubes, "CatalogSearchUtils", Mock())
    search = Mock(return_value=[s2])
    monkeypatch.setattr(hub_cubes.ard_items, "search_ard_items", search)
    with pytest.raises(RuntimeError, match="Resolve the HUB manifest"):
        stats._hub_source_signature()
    manifest = stats._resolve_hub_manifest()
    assert search.call_count == 1 and search.call_args.args[1] == "ARD_S2"
    assert set(manifest["asset_key"]) == {"B04", "B08", "SCL", "MASK"}
    signature = stats._hub_source_signature()
    assert signature["hub_manifest_digest"] == stats._hub_manifest_digest
    assert signature["hub_collections"] == ["ARD_S2"]
    assert "href" not in manifest.columns


def test_run_hub_batches_writes_cubes_once_and_reuses_them(tmp_path, stats_factory, monkeypatch):
    """Batch tuples follow the openEO contract and existing cubes are reused."""
    stats = stats_factory(sentinel1_bands=[])
    s2 = make_item(tmp_path, "ARD_S2", "A", "2024-06-05T09:15:59Z", s2_layers(0.1, 0.5), S2_ORIGIN, {"s2:datatake_id": "D1"})
    monkeypatch.setattr(hub_cubes, "CatalogSearchUtils", Mock())
    monkeypatch.setattr(hub_cubes.ard_items, "search_ard_items", Mock(return_value=[s2]))
    monkeypatch.setattr(hub_cubes, "HubAssetResolver", lambda _client: LocalResolver())
    with pytest.raises(RuntimeError, match="Resolve the HUB manifest"):
        list(stats._run_hub_batches(tmp_path / "cubes"))
    stats._resolve_hub_manifest()
    batches = list(stats._run_hub_batches(tmp_path / "cubes"))
    assert [(number, path.name) for number, _parcels, path in batches] == [(1, "batch_00001_monthly.nc")]
    assert (tmp_path / "cubes" / "hub_item_manifest.parquet").exists()
    assert (tmp_path / "cubes" / "hub_coverage_report.csv").exists()
    modified = batches[0][2].stat().st_mtime_ns
    writer = Mock(side_effect=AssertionError("cube rebuilt"))
    monkeypatch.setattr(stats, "_write_hub_cube", writer)
    assert next(stats._run_hub_batches(tmp_path / "cubes"))[2].stat().st_mtime_ns == modified


def test_hub_yields_completed_cube_and_coverage_before_later_acquisition_fails(tmp_path, stats_factory, monkeypatch):
    """An eager batch list would lose the first handoff when the second cube fails."""
    stats = stats_factory(sentinel1_bands=[])
    stats._hub_manifest = pd.DataFrame({"batch_number": [1, 2]})
    stats._hub_batches = [(1, stats.parcels), (2, stats.parcels)]
    stats._hub_observations = {1: [], 2: []}
    monkeypatch.setattr(hub_cubes, "HubAssetResolver", lambda _client: LocalResolver())
    monkeypatch.setattr(stats, "_batch_available", lambda _path: False)

    def acquire(_resolver, number, _parcels, _observations, path):
        """Commit the first cube and simulate a storage outage for the second."""
        if number == 2:
            raise OSError("Error: Storage unavailable.")
        path.write_bytes(b"complete cube")
        return [{"batch_number": 1, "observations": 19}]

    monkeypatch.setattr(stats, "_write_hub_cube", acquire)
    batches = stats._run_hub_batches(tmp_path / "cubes")
    number, _, path = next(batches)
    assert number == 1 and path.is_file()
    coverage = pd.read_csv(path.parent / "hub_coverage_report.csv")
    assert coverage.to_dict("records") == [{"batch_number": 1, "observations": 19}]
    with pytest.raises(OSError, match="Storage unavailable"):
        next(batches)
    assert path.is_file()
    pd.testing.assert_frame_equal(pd.read_csv(path.parent / "hub_coverage_report.csv"), coverage)


def test_configure_hub_source_validates_settings(stats_factory, monkeypatch):
    """Missing URL, unknown client and invalid worker counts are rejected before network access."""
    stats = stats_factory()
    monkeypatch.delenv("CATALOG_URL", raising=False)
    with pytest.raises(ValueError, match="catalog URL"):
        stats._configure_hub_source()
    with pytest.raises(ValueError, match="no configured secret"):
        stats._configure_hub_source(catalog_url="https://c.test", client_id="admin")
    with pytest.raises(ValueError, match="read_workers"):
        stats._configure_hub_source(catalog_url="https://c.test", client_id="internal", read_workers=0)


def test_split_by_cell_groups_parcels_deterministically_by_centroid_cell():
    """Parcels far apart never share a HUB batch; order is stable by cell row then column."""
    parcels = gpd.GeoDataFrame(
        {"parcel_id": ["east", "west", "near_west"]},
        geometry=[
            box(425000, 4400000, 425100, 4400100),
            box(401000, 4400000, 401100, 4400100),
            box(402000, 4401000, 402100, 4401100),
        ],
        crs=2100,
    )
    groups = hub_cubes.split_by_cell(parcels, 10_000)
    assert [group["parcel_id"].tolist() for group in groups] == [["west", "near_west"], ["east"]]
    assert all(group.index.tolist() == list(range(len(group))) for group in groups)


def test_context_is_only_the_read_halo_without_filling(stats_factory):
    """With fill_nulls=False no filling or interpolation runs, so no wide context is read."""
    assert stats_factory()._hub_context_metres() == hub_cubes.READ_HALO_PIXELS * 10
    filling = stats_factory(fill_nulls=True, interpolation_max_distance_in_meters=100)
    assert filling._hub_context_metres() == (7 // 2 + hub_cubes.READ_HALO_PIXELS) * 10 + 100
    assert stats_factory(fill_nulls=True)._hub_context_metres() is None


def test_footprint_pruning_drops_observations_that_miss_every_parcel(tmp_path):
    """Neighbouring tiles whose raster extent overlaps but whose data misses the parcels cost no reads."""
    parcel = box(400000, 4400000, 400100, 4400100)
    covering = make_item(
        tmp_path,
        "ARD_S2",
        "cover",
        "2024-06-05T09:00:00Z",
        s2_layers(0.1, 0.5),
        S2_ORIGIN,
        {"s2:datatake_id": "D1"},
        footprint=parcel.buffer(500),
    )
    missing = make_item(
        tmp_path,
        "ARD_S2",
        "miss",
        "2024-06-06T09:00:00Z",
        s2_layers(0.1, 0.5),
        S2_ORIGIN,
        {"s2:datatake_id": "D2"},
        footprint=box(410000, 4410000, 411000, 4411000),
    )
    unknown = make_item(
        tmp_path, "ARD_S2", "unknown", "2024-06-07T09:00:00Z", s2_layers(0.1, 0.5), S2_ORIGIN, {"s2:datatake_id": "D3"}
    )
    region = gpd.GeoSeries([parcel], crs=2100).to_crs(4326).iloc[0]
    kept = ard_items.intersecting_observations(ard_items.group_observations([covering, missing, unknown]), region)
    assert [o.key for o in kept] == ["s2|D1", "s2|D3"]


def test_block_items_skip_items_whose_footprint_misses_the_block(tmp_path):
    """Within one observation, an item is read only for blocks its footprint touches."""
    west = make_item(
        tmp_path,
        "ARD_S2",
        "west",
        "2024-06-05T09:00:00Z",
        s2_layers(0.1, 0.5),
        S2_ORIGIN,
        {"s2:datatake_id": "D1"},
        footprint=box(399000, 4399000, 400050, 4401000),
    )
    east = make_item(
        tmp_path,
        "ARD_S2",
        "east",
        "2024-06-05T09:00:00Z",
        s2_layers(0.2, 0.6),
        S2_ORIGIN,
        {"s2:datatake_id": "D1"},
        footprint=box(400060, 4399000, 401000, 4401000),
    )
    observation = ard_items.group_observations([west, east])[0]
    footprints = hub_cubes.project_footprints([observation])
    assert set(footprints) == {"west", "east"}
    west_block = box(399990, 4400000, 400040, 4400050)
    assert [item.item_id for item in hub_cubes.block_items(observation, west_block, footprints)] == ["west"]
    assert hub_cubes.project_footprints([]) == {}


def test_item_reads_use_metadata_windows_and_one_open_per_asset(tmp_path, stats_factory):
    """STAC proj metadata locates the window, so each asset is opened exactly once."""
    stats = stats_factory(sentinel1_bands=[])
    item = make_item(tmp_path, "ARD_S2", "A", "2024-06-05T09:15:59Z", s2_layers(0.1, 0.5), S2_ORIGIN, {"s2:datatake_id": "D1"})
    resolver = LocalResolver()
    path = tmp_path / "batch_00001_monthly.nc"
    stats._write_hub_cube(resolver, 1, stats.parcels, ard_items.group_observations([item]), path)
    assert sorted(resolver.opened) == sorted(asset.href for asset in item.assets.values())


def test_work_chunks_bound_concurrent_observation_reads(stats_factory):
    """Chunks hold about two reads per worker, but never split one (time, block) unit."""
    stats = stats_factory()
    stats.hub_read_workers = 2
    blocks = list(hub_cubes.iter_blocks(4, 4, 2))
    groups = [["o1", "o2"], [], ["o3"]]
    chunks = list(stats._hub_work_chunks(groups, blocks))
    assert sum(len(chunk) for chunk in chunks) == 2 * len(blocks)
    assert all(sum(len(groups[t]) for t, _ in chunk) <= 4 for chunk in chunks)
    assert all(t != 1 for chunk in chunks for t, _ in chunk)


class ConcurrencyResolver(LocalResolver):
    """Local resolver that records the peak number of simultaneously open assets."""

    def __init__(self):
        super().__init__()
        self.active = self.peak = 0
        self.lock = threading.Lock()

    @contextmanager
    def open(self, href):
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            time.sleep(0.02)  # widen the overlap window so excess concurrency would be observed
            with super().open(href) as dataset:
                yield dataset
        finally:
            with self.lock:
                self.active -= 1


def test_metadata_fallback_opens_respect_the_remote_concurrency_cap(tmp_path, stats_factory):
    """Items lacking proj metadata open on the shared asset pool, so opens never exceed hub_read_workers."""
    stats = stats_factory(sentinel1_bands=[])
    stats.hub_read_workers = 2
    items = []
    for index in range(6):
        item = make_item(
            tmp_path,
            "ARD_S2",
            f"I{index}",
            f"2024-06-{index + 3:02d}T09:00:00Z",
            s2_layers(0.1, 0.5),
            S2_ORIGIN,
            {"s2:datatake_id": f"D{index}"},
        )
        if index % 2:
            for key, asset in list(item.assets.items()):
                item.assets[key] = dataclasses.replace(asset, shape=None)  # force the open-based window fallback
        items.append(item)
    resolver = ConcurrencyResolver()
    stats._write_hub_cube(resolver, 1, stats.parcels, ard_items.group_observations(items), tmp_path / "batch_00001_monthly.nc")
    assert resolver.peak <= 2
    # Metadata items open each asset once; fallback items add one window open each.
    assert len(resolver.opened) == 6 * 4 + 3
