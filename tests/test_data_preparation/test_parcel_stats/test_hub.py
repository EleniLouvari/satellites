"""HUB adapter integration using real local rasters instead of network assets."""

import json
import pickle
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from affine import Affine
from shapely.geometry import box

from data_preparation.parcel_stats import HubZonalStats
from data_preparation.sources.hub import cubes
from tests.utils import expect_equal, expect_true


@pytest.fixture
def hub_source(tmp_path, monkeypatch):
    """Replace only catalog and remote access with in-process/local equivalents."""
    directory = tmp_path / "assets"
    directory.mkdir()
    transform = Affine(10, 0, 500000, 0, -10, 4200040)
    features = {}
    for collection, values in {
        "ARD_S2": {"B04": 0.2, "B08": 0.6, "SCL": 4, "MASK": 1},
        "ARD_S1": {"VV": 4.0, "VH": 1.0, "MASK": 1},
    }.items():
        assets = {}
        for name, value in values.items():
            path = directory / f"{collection}_{name}.tif"
            categorical = name in {"MASK", "SCL"}
            dtype = "uint8" if categorical else "float32"
            nodata = 255 if name == "SCL" else (0 if name == "MASK" else float(np.finfo("float32").max))
            with rasterio.open(
                path, "w", driver="GTiff", width=4, height=4, count=1,
                dtype=dtype, crs=2100, transform=transform, nodata=nodata,
            ) as dataset:
                dataset.write(np.full((4, 4), value, dtype=dtype), 1)
            assets[name] = {
                "href": str(path), "proj:transform": list(transform)[:6],
                "raster:bands": [{"nodata": nodata, "scale": 1, "offset": 0}],
            }
        features[collection] = {
            "id": f"{collection}-fixture", "collection": collection, "assets": assets,
            "properties": {
                "datetime": "2024-01-15T10:00:00Z" if collection == "ARD_S2" else "2024-01-16T10:00:00Z",
                "platform": "sentinel-2a" if collection == "ARD_S2" else "sentinel-1a",
                "s2:datatake_id": "fixture", "sat:absolute_orbit": 123,
                "sat:orbit_state": "ascending", "proj:code": "EPSG:2100",
                "processing:datetime": "2026-11-20T00:00:00Z", "eo:cloud_cover": 0,
            },
        }
    state = SimpleNamespace(features=features, searches=[], opened=[])

    class LocalCatalog:
        """Return deterministic fixture STAC items for the requested collection."""

        def __init__(self, **kwargs):
            """Accept the production catalog constructor without a network session."""

        def fetch_features(self, search_body, **kwargs):
            """Record discovery and return the selected collection's current item."""
            collection = search_body["collections"][0]
            state.searches.append(collection)
            return [state.features[collection]]

    class LocalResolver:
        """Use real GeoTIFF reads without remote credentials or HTTP requests."""

        def __init__(self, client_id):
            """Accept the configured client, which is unused for local fixtures."""

        def open(self, href):
            """Record physical reads and open an actual local raster."""
            state.opened.append(href)
            return rasterio.open(href)

    monkeypatch.setattr(cubes, "USERS", {"internal": "fixture-secret"})
    monkeypatch.setattr(cubes, "CatalogSearchUtils", LocalCatalog)
    monkeypatch.setattr(cubes, "HubAssetResolver", LocalResolver)
    return state


def _extractor(output_dir, mode):
    """Return a HUB processor for one four-pixel parcel."""
    optical = mode != "s1"
    radar = mode != "s2"
    return HubZonalStats(
        parcels=gpd.GeoDataFrame(
            {"parcel_id": ["a"], "label": [3]}, geometry=[box(500010, 4200010, 500030, 4200030)], crs=2100,
        ),
        start_date="2024-01-01", end_date="2024-01-31", output_dir=output_dir, working_epsg=2100,
        hub_catalog_url="https://catalog.example.test", hub_client_id="internal", hub_read_workers=1,
        sentinel2_bands=["B04", "B08"] if optical else [], sentinel2_indices=["NDVI"] if optical else [],
        sentinel1_bands=["VV", "VH"] if radar else [], sentinel1_indices=["R", "RVI"] if radar else [],
        spatial_statistics=["mean", "median"], temporal_reducer="none",
        remove_outliers=False, fill_nulls=False, batch_workers=1, resume_completed_partitions=True,
    )


@pytest.mark.parametrize("mode", ["s2", "s1", "combined"])
def test_hub_run_writes_physical_features_and_calibration_provenance(tmp_path, hub_source, mode):
    """Catch double scaling, missing sensor rows, or loss of HUB calibration identity."""
    extractor = _extractor(tmp_path / mode, mode)
    final = extractor.run()
    expect_equal(final["parcel_id"].tolist(), ["a"])
    expect_equal(final["label"].tolist(), [3])
    if mode != "s1":
        np.testing.assert_allclose(final["B04_mean__20240101"], [0.2], rtol=1e-6)
        np.testing.assert_allclose(final["NDVI_median__20240101"], [0.5], rtol=1e-6)
    if mode != "s2":
        np.testing.assert_allclose(final["VV_mean__20240101"], [4.0])
        np.testing.assert_allclose(final["R_median__20240101"], [4.0])
        np.testing.assert_allclose(final["RVI_median__20240101"], [0.8], rtol=1e-6)
    marker = json.loads(extractor._partition_completion_path().read_text(encoding="utf-8"))
    expect_equal(marker["status"], "complete")
    expect_equal(marker["provenance"]["source"], "HUB")
    expect_equal(marker["provenance"]["hub_manifest_digest"], extractor._hub_manifest_digest)
    expect_equal(marker["provenance"]["hub_sentinel1_backscatter"], "gamma0-terrain-flattened-linear")
    expect_true("fixture-secret" not in json.dumps(marker))
    manifest = pd.read_parquet(extractor.output_dir / "hub_item_manifest.parquet")
    expect_true("href" not in manifest.columns)
    expect_true((extractor.output_dir / extractor.REDUCED_PARCEL_OUTPUT_FILE_NAME).is_file())


def test_hub_resume_discovers_inventory_before_reusing_completed_outputs(tmp_path, hub_source):
    """Catch stale partition reuse when the HUB reprocesses an existing item."""
    extractor = _extractor(tmp_path / "resume", "s2")
    with pytest.raises(RuntimeError, match="manifest"):
        extractor._run_signature()
    first = extractor.run()
    first_signature = extractor._run_signature()
    physical_reads = len(hub_source.opened)
    search_count = len(hub_source.searches)
    pd.testing.assert_frame_equal(extractor.run(), first)
    expect_equal(len(hub_source.opened), physical_reads)
    expect_true(len(hub_source.searches) > search_count)
    hub_source.features["ARD_S2"]["properties"]["processing:datetime"] = "2026-11-21T00:00:00Z"
    extractor.run()
    expect_true(extractor._run_signature() != first_signature)
    expect_true(len(hub_source.opened) > physical_reads)


def test_hub_invalid_source_settings_fail_before_output_creation(tmp_path, hub_source):
    """Catch configuration failures that create misleading run directories."""
    with pytest.raises(ValueError, match="hub read_workers"):
        HubZonalStats(
            output_dir=tmp_path / "invalid", hub_catalog_url="https://catalog.example.test",
            hub_client_id="internal", hub_read_workers=0,
        )
    expect_true(not (tmp_path / "invalid").exists())


@pytest.mark.parametrize("epsg", [None, 3857])
def test_hub_defaults_to_greek_grid_and_accepts_explicit_crs(tmp_path, hub_source, epsg):
    """Catch a missing Greek Grid default or an ignored explicit output CRS."""
    options = {} if epsg is None else {"working_epsg": epsg}
    extractor = HubZonalStats(
        parcels=gpd.GeoDataFrame({"parcel_id": ["a"]}, geometry=[box(500010, 4200010, 500030, 4200030)], crs=2100),
        start_date="2024-01-01", end_date="2024-01-31", output_dir=tmp_path / "grid",
        hub_catalog_url="https://catalog.example.test", hub_client_id="internal", **options,
    )
    expect_equal(extractor.working_epsg, 2100 if epsg is None else epsg)


def test_hub_preserves_inherited_positional_crs_override(tmp_path, hub_source):
    """Catch supplying a duplicate default when callers use the inherited positional API."""
    parcels = gpd.GeoDataFrame({"parcel_id": ["a"]}, geometry=[box(500010, 4200010, 500030, 4200030)], crs=2100)
    extractor = HubZonalStats(
        parcels, "2024-01-01", "2024-01-31", tmp_path / "positional", 3857,
        hub_catalog_url="https://catalog.example.test", hub_client_id="internal",
    )
    expect_equal(extractor.working_epsg, 3857)


@pytest.mark.parametrize(
    ("option", "value", "method"),
    [
        ("spatial_fill_window_sizes", (3, 5, 11), "nearest"),
        ("interpolation_variogram_max_distance_in_meters", 200.0, "kriging"),
    ],
)
def test_hub_wider_context_reacquires_raw_cube(tmp_path, hub_source, option, value, method):
    """Catch reuse of a smaller raster after changing the required fill context."""
    extractor = _extractor(tmp_path / "context", "s2")
    extractor.interpolation_method = method
    extractor.run()
    first_signature = extractor._run_signature()
    physical_reads = len(hub_source.opened)
    setattr(extractor, option, value)
    extractor.run()
    expect_true(len(hub_source.opened) > physical_reads)
    expect_true(extractor._run_signature() != first_signature)


def test_hub_resume_refreshes_parcel_attributes_without_reading_rasters(tmp_path, hub_source):
    """Catch stale training labels without discarding reusable pixel statistics."""
    extractor = _extractor(tmp_path / "attributes", "s2")
    extractor.remove_nc_after_completion = True
    extractor.run()
    physical_reads = len(hub_source.opened)
    extractor.parcels["label"] = 9
    final = extractor.run()
    expect_equal(final["label"].tolist(), [9])
    reduced = gpd.read_parquet(extractor.output_dir / extractor.REDUCED_PARCEL_OUTPUT_FILE_NAME)
    expect_equal(reduced["label"].tolist(), [9])
    extractor.parcels = extractor.parcels.drop(columns="label")
    expect_true("label" not in extractor.run().columns)
    expect_equal(len(hub_source.opened), physical_reads)


def test_hub_requires_completion_marker_when_statistics_change(tmp_path, hub_source):
    """Catch accidental application of openEO's markerless legacy fallback to HUB."""
    extractor = _extractor(tmp_path / "marker", "s2")
    extractor.run()
    extractor._partition_completion_path().unlink()
    extractor.spatial_statistics = ("median",)
    final = extractor.run()
    expect_true("B04_mean__20240101" not in final.columns)
    expect_true("B04_median__20240101" in final.columns)
    expect_true(extractor._partition_completion_path().is_file())


def test_hub_changed_inventory_failure_never_retains_complete_state(tmp_path, hub_source, monkeypatch):
    """Catch a new published manifest beside an old complete marker after acquisition fails."""
    extractor = _extractor(tmp_path / "failure", "s2")
    extractor.run()
    hub_source.features["ARD_S2"]["properties"]["processing:datetime"] = "2026-11-21T00:00:00Z"

    def fail_acquisition(cube_dir):
        """Simulate a storage outage before replacement cubes can be acquired."""
        raise OSError("storage unavailable")

    monkeypatch.setattr(extractor, "_run_hub_batches", fail_acquisition)
    with pytest.raises(OSError, match="storage unavailable"):
        extractor.run()
    marker = json.loads(extractor._partition_completion_path().read_text(encoding="utf-8"))
    expect_equal(marker["status"], "processing")
    expect_equal(marker["provenance"]["hub_manifest_digest"], extractor._hub_manifest_digest)


def test_hub_worker_payload_omits_large_acquisition_inventory(tmp_path, hub_source):
    """Catch duplicating large STAC inventories into every local worker process."""
    extractor = _extractor(tmp_path / "payload", "s2")
    extractor._resolve_hub_manifest()
    signature = extractor._statistics_signature()
    extractor._hub_manifest = pd.DataFrame({"row": np.arange(150000, dtype="int64")})
    payload = pickle.dumps(extractor)
    expect_true(len(payload) < 100_000)
    restored = pickle.loads(payload)
    expect_equal(restored._statistics_signature(), signature)


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("cleanup", [False, True])
def test_hub_streams_multiple_batches_through_real_local_workers(tmp_path, hub_source, monkeypatch, workers, cleanup):
    """Preserve numerical features and checkpoint cleanup across streamed raster batches."""
    extractor = _extractor(tmp_path / "streamed", "s2")
    second = extractor.parcels.copy()
    second["parcel_id"] = "b"
    extractor.parcels = gpd.GeoDataFrame(pd.concat([extractor.parcels, second], ignore_index=True), crs=4326)
    extractor.batch_workers = workers
    extractor.remove_nc_after_completion = cleanup
    monkeypatch.setattr(HubZonalStats, "_hub_iter_batches", lambda self: (
        self.parcels.iloc[[index]].reset_index(drop=True) for index in range(2)
    ))
    final = extractor.run()
    expect_equal(final["parcel_id"].tolist(), ["a", "b"])
    np.testing.assert_allclose(final["NDVI_mean__20240101"], [0.5, 0.5], rtol=1e-6)
    cubes = list(extractor.output_dir.rglob("batch_*_monthly.nc"))
    expect_equal(len(cubes), 0 if cleanup else 2)
    marker = json.loads(extractor._partition_completion_path().read_text(encoding="utf-8"))
    expect_equal(marker["status"], "complete")
    expect_true(bool(list(extractor.output_dir.glob("satellite_parcel_statistics.worker-*.log"))))
    # Reassemble from per-batch tables, including when raw rasters were deleted.
    extractor._partition_completion_path().unlink()
    physical_reads = len(hub_source.opened)
    pd.testing.assert_frame_equal(extractor.run(), final)
    expect_equal(len(hub_source.opened), physical_reads)
