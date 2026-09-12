"""Synthetic seam and schema regression tests for Planet extraction."""

import numpy as np
import pandas as pd
import geopandas as gpd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box
from satellites.data_preparation.parcel_stats.planet import PlanetBasemapZonalStats


@pytest.fixture
def tmp_path():
    # Avoid pytest's restrictive Windows chmod on this managed workspace.
    from pathlib import Path
    from uuid import uuid4

    path = Path("tmp") / ("planet_" + uuid4().hex)
    path.mkdir(parents=True)
    return path


def inputs(tmp_path):
    records = []
    for month in (1, 2):
        for quad, left, red, nir in [("a", 500000, 2, 6), ("b", 500020, 4, 4)]:
            path = tmp_path / f"{month}_{quad}.tif"
            data = np.ones((9, 4, 4), dtype="float32")
            data[5] = red
            data[7] = nir
            data[8] = 999  # Must be replaced by the Planet NDVI formula.
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                width=4,
                height=4,
                count=9,
                dtype="float32",
                crs="EPSG:2100",
                transform=from_origin(left, 4500020, 5, 5),
                nodata=-9999,
            ) as dst:
                dst.write(data)
            records.append(dict(period_start=f"2024-{month:02d}-01", quad_id=quad, path=str(path)))
    parcels = gpd.GeoDataFrame(
        {"parcel_code": ["seam"], "label": ["wheat"]}, geometry=[box(500015.1, 4500005.1, 500024.9, 4500014.9)], crs=2100
    )
    return parcels, pd.DataFrame(records)


def test_seam_schema_and_resume(tmp_path):
    parcels, manifest = inputs(tmp_path)
    extractor = PlanetBasemapZonalStats(
        parcels, "2024-01-01", "2024-02-29", tmp_path / "output", 2100, parcel_id_field="parcel_code", buffer_metres=0
    )
    result = extractor.run(manifest)
    assert len(result) == 1
    assert result.iloc[0]["NDVI_mean__20240101"] == pytest.approx(0.25)
    assert result.iloc[0]["B6_mean__20240101"] == pytest.approx(0.0003)
    assert result.iloc[0]["intersected_pixel_count"] == 4
    assert result.iloc[0]["label"] == "wheat"
    for index in ["EVI", "NDRE", "NDVI", "NDWI", "SAVI"]:
        for statistic in ["mean", "median", "sd", "min", "max", "range"]:
            assert f"{index}_{statistic}__20240101" in result.columns
    assert result.crs.to_epsg() == 2100
    assert result.iloc[0]["data_reliability_score"] == 1
    assert (tmp_path / "output" / extractor.REDUCED_PARCEL_OUTPUT_FILE_NAME).exists()
    pd.testing.assert_frame_equal(result, extractor.run(manifest))


def test_missing_quad_month_rejected(tmp_path):
    parcels, manifest = inputs(tmp_path)
    extractor = PlanetBasemapZonalStats(
        parcels, "2024-01-01", "2024-02-29", tmp_path / "out", 2100, parcel_id_field="parcel_code"
    )
    with pytest.raises(ValueError, match="Missing months"):
        extractor.run(manifest.iloc[:-1])


def test_zero_denominator_is_null(tmp_path):
    parcels, _ = inputs(tmp_path)
    extractor = PlanetBasemapZonalStats(
        parcels, "2024-01-01", "2024-02-29", tmp_path / "out", 2100, parcel_id_field="parcel_code"
    )
    result = extractor._calculate_local_sentinel2_index({"B8": np.array([0.0, 6.0]), "B6": np.array([0.0, 2.0])}, "NDVI")
    assert np.isnan(result[0])
    assert result[1] == pytest.approx(0.5)


def test_catalog_cross_year_and_download_resume(tmp_path, monkeypatch):
    from satellites.data_preparation.parcel_stats.planet import download_monthly_tiles
    from satellites.data_preparation.sources.hub import catalog_s3_checks

    queries, downloads = [], []

    class Catalog:
        client_id = "internal"

        def fetch_features(self, search_body, max_items):
            identifier = search_body["filter"]["args"][1].strip("%")
            queries.append(identifier)
            assert max_items == 2
            return [{"id": identifier, "assets": {"analysis": {"href": "Analysis.tif"}}}]

    def download(collection, item_id, root, **kwargs):
        downloads.append(item_id)
        folder = root / collection / item_id
        folder.mkdir(parents=True, exist_ok=True)
        with rasterio.open(
            folder / "Analysis.tif",
            "w",
            driver="GTiff",
            width=2,
            height=2,
            count=8,
            dtype="float32",
            crs="EPSG:2100",
            transform=from_origin(500000, 4500020, 5, 5),
        ) as dst:
            dst.write(np.ones((8, 2, 2), dtype="float32"))
        return {"failed": []}

    monkeypatch.setattr(catalog_s3_checks, "download_all_assets_for_item", download)
    catalog = Catalog()
    tiles = pd.DataFrame({"quad_id": ["1234-5678"]})
    result = download_monthly_tiles(catalog, tiles, "2023-11-01", "2024-02-29", tmp_path)
    assert result.period_start.tolist() == ["2023-11-01", "2023-12-01", "2024-01-01", "2024-02-01"]
    assert len(downloads) == 4
    download_monthly_tiles(catalog, tiles, "2023-11-01", "2024-02-29", tmp_path)
    assert len(downloads) == 4


def test_nodata_and_temporal_fill(tmp_path):
    parcels, manifest = inputs(tmp_path)
    for path in manifest[manifest.period_start == "2024-02-01"].path:
        with rasterio.open(path, "r+") as dst:
            data = dst.read()
            data[:, 1, :] = -9999
            dst.write(data)
    extractor = PlanetBasemapZonalStats(
        parcels,
        "2024-01-01",
        "2024-02-29",
        tmp_path / "output",
        2100,
        parcel_id_field="parcel_code",
        buffer_metres=5,
        fill_nulls=True,
        temporal_fill_mode="past_only",
        remove_outliers=True,
    )
    result = extractor.run(manifest)
    assert result.iloc[0]["NDVI_mean__20240201"] == pytest.approx(0.25)
    assert result.iloc[0]["temporal_filled_pixel_count"] > 0
    assert result.iloc[0]["data_reliability_score"] < 1


@pytest.mark.parametrize(
    "index, expected",
    [
        ("EVI", 2.5 * 0.4 / (0.6 + 6 * 0.2 - 7.5 * 0.1 + 1)),
        ("NDRE", 0.2 / 1.0),
        ("NDVI", 0.4 / 0.8),
        ("NDWI", -0.3 / 0.9),
        ("SAVI", 1.5 * 0.4 / 1.3),
    ],
)
def test_planet_index_formulas(index, expected):
    extractor = object.__new__(PlanetBasemapZonalStats)
    bands = {
        name: np.array([value], dtype="float32")
        for name, value in {"B2": 0.1, "B4": 0.3, "B6": 0.2, "B7": 0.4, "B8": 0.6}.items()
    }
    assert extractor._calculate_local_sentinel2_index(bands, index)[0] == pytest.approx(expected)


def test_ndmi_rejected(tmp_path):
    parcels, _ = inputs(tmp_path)
    with pytest.raises(ValueError, match="no SWIR"):
        PlanetBasemapZonalStats(
            parcels, "2024-01-01", "2024-02-29", tmp_path / "out", 2100, parcel_id_field="parcel_code", indices=["NDMI"]
        )


class LocalTileCatalog:
    """Exercise the real download helper using local files and no credentials."""

    client_id = "internal"
    base_url = "local-test-catalog"

    def __init__(self, manifest, scratch):
        import calendar

        self.scratch = scratch
        self.items = {}
        self.searches = []
        self.fail_identifier = None
        for row in manifest.itertuples():
            month = pd.Timestamp(row.period_start)
            identifier = f"{calendar.month_name[month.month].lower()}_{month.year}_{row.quad_id}_"
            self.items[identifier] = {
                "id": identifier,
                "assets": {
                    "analysis": {"href": row.path, "hub:relativePath": "Analysis.tif"},
                    # Downloading unneeded assets would fail; streaming must filter them.
                    "preview": {"href": "missing_preview.png"},
                },
            }

    def fetch_features(self, search_body, max_items):
        if "ids" in search_body:
            return [self.items[search_body["ids"][0]]]
        identifier = search_body["filter"]["args"][1].strip("%")
        if identifier == self.fail_identifier:
            raise RuntimeError("simulated catalog failure")
        # Previous months and batches must already be gone when a new month starts.
        files = list(self.scratch.rglob("*.tif")) if self.scratch.exists() else []
        month_prefix = identifier.split("_", 2)[:2]
        assert all(path.parent.name.split("_", 2)[:2] == month_prefix for path in files)
        self.searches.append((identifier, len(files)))
        return [self.items[identifier]]


def streaming_inputs(tmp_path):
    from pathlib import Path

    parcels, manifest = inputs(tmp_path)
    manifest["path"] = manifest.path.map(lambda value: str(Path(value).resolve()))
    parcels = gpd.GeoDataFrame(
        {"parcel_code": ["left", "right", "seam"], "label": ["wheat"] * 3},
        geometry=[
            box(500005.1, 4500005.1, 500014.9, 4500014.9),
            box(500025.1, 4500005.1, 500034.9, 4500014.9),
            parcels.geometry.iloc[0],
        ],
        crs=2100,
    )
    tiles = gpd.GeoDataFrame(
        {"quad_id": ["a", "b"]}, geometry=[box(500000, 4500000, 500020, 4500020), box(500020, 4500000, 500040, 4500020)], crs=2100
    )
    return parcels, manifest, tiles


def test_streaming_seams_cleanup_and_resume(tmp_path):
    parcels, manifest, tiles = streaming_inputs(tmp_path)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    sentinel = scratch / "user_file.txt"
    sentinel.write_text("keep me")
    catalog = LocalTileCatalog(manifest, scratch)
    extractor = PlanetBasemapZonalStats(
        parcels, "2024-01-01", "2024-02-29", tmp_path / "output", 2100, parcel_id_field="parcel_code", buffer_metres=0
    )
    result = extractor.run_catalog(catalog, tiles, scratch_dir=scratch)
    assert len(result) == 3
    means = result.set_index("parcel_code")["NDVI_mean__20240101"]
    assert means["left"] == pytest.approx(0.5)
    assert means["right"] == pytest.approx(0)
    assert means["seam"] == pytest.approx(0.25)
    assert result.parcel_code.is_unique
    assert len(catalog.searches) == 8  # Two months for a, b, and the a+b seam.
    assert max(count for _, count in catalog.searches) == 1  # At most two tiles coexist.
    assert sorted(path.name for path in scratch.iterdir()) == ["user_file.txt"]
    assert sentinel.read_text() == "keep me"
    assert all(__import__("pathlib").Path(path).exists() for path in manifest.path)
    assert not list((tmp_path / "output").rglob("*.nc"))
    pd.testing.assert_frame_equal(result, extractor.run_catalog(catalog, tiles, scratch_dir=scratch))
    assert len(catalog.searches) == 8  # Completed results require no new downloads.
    audit = pd.read_csv(tmp_path / "output" / "planet_tile_manifest.csv")
    assert audit.local_files_deleted.all()


def test_streaming_failure_cleanup_and_restart(tmp_path):
    parcels, manifest, tiles = streaming_inputs(tmp_path)
    scratch = tmp_path / "scratch"
    catalog = LocalTileCatalog(manifest, scratch)
    # First quad commits; fail during the second month's download for quad b.
    catalog.fail_identifier = "february_2024_b_"
    extractor = PlanetBasemapZonalStats(
        parcels, "2024-01-01", "2024-02-29", tmp_path / "output", 2100, parcel_id_field="parcel_code", buffer_metres=0
    )
    with pytest.raises(RuntimeError, match="simulated"):
        extractor.run_catalog(catalog, tiles, scratch_dir=scratch)
    assert list(scratch.iterdir()) == []
    assert len(list((tmp_path / "output" / "streaming_results").rglob("batch_*.json"))) == 1
    assert not (tmp_path / "output" / extractor.PARCEL_OUTPUT_FILE_NAME).exists()
    first_quad_searches = catalog.searches.count(("january_2024_a_", 0))
    catalog.fail_identifier = None
    result = extractor.run_catalog(catalog, tiles, scratch_dir=scratch)
    assert len(result) == 3
    # Quad a's single-tile batch was skipped; it was fetched again only for the seam.
    assert catalog.searches.count(("january_2024_a_", 0)) == first_quad_searches + 1
    assert list(scratch.iterdir()) == []


def test_download_cache_validates_mode_and_local_file(tmp_path):
    from pathlib import Path
    from satellites.data_preparation.parcel_stats.planet import download_monthly_tiles

    _, manifest, tiles = streaming_inputs(tmp_path)
    catalog = LocalTileCatalog(manifest, tmp_path / "scratch")
    tiles = tiles.iloc[:1]
    kwargs = dict(catalog=catalog, tiles=tiles, start_date="2024-01-01", end_date="2024-01-31", download_dir=catalog.scratch)
    first = download_monthly_tiles(**kwargs, analysis_only=True)
    analysis = Path(first.iloc[0].path)
    # An existing marker must not hide a modified or truncated local download.
    analysis.write_bytes(b"broken raster")
    download_monthly_tiles(**kwargs, analysis_only=True)
    with rasterio.open(analysis) as source:
        assert source.count == 9
    # Analysis-only completion cannot certify a later request for all assets.
    with pytest.raises(RuntimeError, match="Asset download failed"):
        download_monthly_tiles(**kwargs, analysis_only=False)
    assert not (analysis.parent / "planet_download_complete.json").exists()


def test_streaming_includes_buffer_neighbors(tmp_path, monkeypatch):
    import xarray as xr

    parcels, manifest, tiles = streaming_inputs(tmp_path)
    extractor = PlanetBasemapZonalStats(
        parcels.iloc[:1], "2024-01-01", "2024-02-29", tmp_path / "output", 2100, parcel_id_field="parcel_code", buffer_metres=10
    )
    cubes = []
    original = PlanetBasemapZonalStats._build_tile_cube

    def capture_cube(self, *args, **kwargs):
        path = original(self, *args, **kwargs)
        with xr.open_dataset(path) as dataset:
            cubes.append(dataset.load())
        return path

    monkeypatch.setattr(PlanetBasemapZonalStats, "_build_tile_cube", capture_cube)
    local = extractor.run(manifest)
    catalog = LocalTileCatalog(manifest, tmp_path / "scratch")
    streamed = extractor.run_catalog(catalog, tiles, scratch_dir=catalog.scratch)
    # The neighboring quad supplies the right-hand buffer even though the parcel
    # itself belongs entirely to the left quad.
    assert float(cubes[1].B6.max()) == pytest.approx(0.0004)
    xr.testing.assert_equal(cubes[0], cubes[1])
    pd.testing.assert_frame_equal(local, streamed)


def test_checkpoint_respects_parcel_id_column(tmp_path):
    parcels, manifest, tiles = streaming_inputs(tmp_path)
    catalog = LocalTileCatalog(manifest, tmp_path / "scratch")
    first = PlanetBasemapZonalStats(
        parcels.iloc[:1], "2024-01-01", "2024-02-29", tmp_path / "output", 2100, parcel_id_field="parcel_code", buffer_metres=0
    )
    first.run_catalog(catalog, tiles, scratch_dir=catalog.scratch)
    renamed = parcels.iloc[:1].rename(columns={"parcel_code": "new_id"})
    second = PlanetBasemapZonalStats(
        renamed, "2024-01-01", "2024-02-29", tmp_path / "output", 2100, parcel_id_field="new_id", buffer_metres=0
    )
    result = second.run_catalog(catalog, tiles, scratch_dir=catalog.scratch)
    assert result.new_id.tolist() == ["left"]
    assert len(catalog.searches) == 4


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"end_date": "2024-02-15"}, "last day"),
        ({"working_epsg": 2263}, "metre units"),
        ({"batch_workers": 2}, "sequential"),
        ({"sentinel1_indices": ["RVI"]}, "Sentinel sensor options"),
    ],
)
def test_planet_configuration_rejects_unsupported_options(tmp_path, changes, message):
    parcels, _ = inputs(tmp_path)
    kwargs = dict(
        parcels=parcels,
        start_date="2024-01-01",
        end_date="2024-02-29",
        output_dir=tmp_path / "output",
        working_epsg=2100,
        parcel_id_field="parcel_code",
    )
    kwargs.update(changes)
    with pytest.raises(ValueError, match=message):
        PlanetBasemapZonalStats(**kwargs)


def test_tiff_scale_and_offset_override_fallback(tmp_path):
    parcels, manifest = inputs(tmp_path)
    for path in manifest.path:
        with rasterio.open(path, "r+") as raster:
            raster.scales = (0.1,) * 9
            raster.offsets = (0.1,) * 9
    extractor = PlanetBasemapZonalStats(
        parcels,
        "2024-01-01",
        "2024-02-29",
        tmp_path / "output",
        2100,
        parcel_id_field="parcel_code",
        buffer_metres=0,
        keep_cleaned_checkpoint=True,
    )
    result = extractor.run(manifest)
    assert result.iloc[0]["B6_mean__20240101"] == pytest.approx(0.4)
    assert result.iloc[0]["NDVI_mean__20240101"] == pytest.approx(0.2)


def test_rotated_raster_envelope_is_not_coverage(tmp_path):
    from rasterio import Affine

    parcels, manifest = inputs(tmp_path)
    for path in manifest.path:
        with rasterio.open(path, "r+") as raster:
            raster.transform = Affine.translation(500000, 4500020) * Affine.rotation(45) * Affine.scale(5, -5)
    # This lies in the axis-aligned envelope, but outside the actual diamond.
    parcels.geometry = [box(500001, 4500007, 500002, 4500008)]
    extractor = PlanetBasemapZonalStats(
        parcels, "2024-01-01", "2024-02-29", tmp_path / "output", 2100, parcel_id_field="parcel_code", buffer_metres=0
    )
    with pytest.raises(ValueError, match="footprints do not cover"):
        extractor.run(manifest)


def test_memory_limit_prevents_catalog_download(tmp_path):
    parcels, manifest, tiles = streaming_inputs(tmp_path)
    catalog = LocalTileCatalog(manifest, tmp_path / "scratch")
    extractor = PlanetBasemapZonalStats(
        parcels, "2024-01-01", "2024-02-29", tmp_path / "output", 2100, parcel_id_field="parcel_code", max_cube_bytes=1
    )
    with pytest.raises(MemoryError, match="raw cube needs"):
        extractor.run_catalog(catalog, tiles, scratch_dir=catalog.scratch)
    assert catalog.searches == []


def test_alpha_mask_is_not_treated_as_reflectance(tmp_path):
    from rasterio.enums import ColorInterp

    parcels, manifest = inputs(tmp_path)
    for row in manifest.itertuples():
        with rasterio.open(row.path, "r+") as raster:
            raster.nodata = None
            raster.colorinterp = (ColorInterp.gray,) + (ColorInterp.undefined,) * 7 + (ColorInterp.alpha,)
            raster.write(np.full((4, 4), 0 if row.quad_id == "b" else 255, dtype="float32"), 9)
    extractor = PlanetBasemapZonalStats(
        parcels, "2024-01-01", "2024-02-29", tmp_path / "output", 2100, parcel_id_field="parcel_code", buffer_metres=0
    )
    result = extractor.run(manifest)
    assert result.iloc[0]["NDVI_mean__20240101"] == pytest.approx(0.5)
    assert result.iloc[0]["data_reliability_score"] < 1


def test_forced_refresh_failure_invalidates_old_commit(tmp_path):
    parcels, manifest, tiles = streaming_inputs(tmp_path)
    catalog = LocalTileCatalog(manifest, tmp_path / "scratch")
    extractor = PlanetBasemapZonalStats(
        parcels.iloc[:1], "2024-01-01", "2024-02-29", tmp_path / "output", 2100, parcel_id_field="parcel_code", buffer_metres=0
    )
    extractor.run_catalog(catalog, tiles, scratch_dir=catalog.scratch)
    catalog.fail_identifier = "february_2024_a_"
    with pytest.raises(RuntimeError, match="simulated"):
        extractor.run_catalog(catalog, tiles, scratch_dir=catalog.scratch, resume=False)
    assert not list((tmp_path / "output" / "streaming_results").rglob("batch_*.json"))
    catalog.fail_identifier = None
    searches_before = len(catalog.searches)
    extractor.run_catalog(catalog, tiles, scratch_dir=catalog.scratch)
    assert len(catalog.searches) == searches_before + 2


def test_external_mask_change_invalidates_raw_cube(tmp_path):
    parcels, manifest = inputs(tmp_path)
    extractor = PlanetBasemapZonalStats(
        parcels, "2024-01-01", "2024-02-29", tmp_path / "output", 2100, parcel_id_field="parcel_code", buffer_metres=0
    )
    original = extractor.run(manifest)
    assert original.iloc[0]["NDVI_mean__20240101"] == pytest.approx(0.25)
    # An external mask changes observed coverage without changing TIFF pixels.
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=False):
        for row in manifest[manifest.quad_id == "b"].itertuples():
            with rasterio.open(row.path, "r+") as raster:
                raster.write_mask(np.zeros((4, 4), dtype="uint8"))
    updated = extractor.run(manifest)
    assert updated.iloc[0]["NDVI_mean__20240101"] == pytest.approx(0.5)


def test_source_neutral_api_and_optical_roles():
    from satellites.data_preparation.parcel_stats import OpenEOZonalStats, OpenEOJobManagerZonalStats
    from satellites.data_preparation.parcel_stats import SatelliteZonalStats, JobManagerSatelliteZonalStats

    assert OpenEOZonalStats is SatelliteZonalStats
    assert OpenEOJobManagerZonalStats is JobManagerSatelliteZonalStats
    red = np.array([0.2, 0.0], dtype="float32")
    nir = np.array([0.6, 0.0], dtype="float32")
    sentinel = OpenEOZonalStats.__new__(OpenEOZonalStats)
    planet = PlanetBasemapZonalStats.__new__(PlanetBasemapZonalStats)
    # Identical reflectance must yield identical algebra despite native band IDs.
    expected = sentinel._calculate_local_optical_index({"B04": red, "B08": nir}, "NDVI")
    actual = planet._calculate_local_optical_index({"B6": red, "B8": nir}, "NDVI")
    np.testing.assert_allclose(actual, expected, equal_nan=True)
    assert actual[0] == pytest.approx(0.5)
    assert np.isnan(actual[1])
    np.testing.assert_allclose(planet._calculate_local_sentinel2_index({"B6": red, "B8": nir}, "NDVI"), actual, equal_nan=True)
    planet.sentinel2_bands = ("B6", "B8")
    planet.sentinel2_indices = ("NDVI",)
    assert planet.optical_bands == ("B6", "B8")
    assert planet.optical_indices == ("NDVI",)


def test_catalog_retains_scratch_when_requested(tmp_path, monkeypatch):
    from pathlib import Path

    parcels, manifest, tiles = streaming_inputs(tmp_path)
    scratch = tmp_path / "retained"
    catalog = LocalTileCatalog(manifest, scratch)
    # This catalog fixture normally asserts deletion; retention intentionally
    # allows several months to coexist in the scratch directory.
    monkeypatch.setattr(
        catalog, "fetch_features", lambda search_body, max_items: [catalog.items[search_body["filter"]["args"][1].strip("%")]]
    )
    extractor = PlanetBasemapZonalStats(
        parcels, "2024-01-01", "2024-02-29", tmp_path / "output", 2100, parcel_id_field="parcel_code", buffer_metres=0
    )
    result = extractor.run_catalog(catalog, tiles, scratch_dir=scratch, delete_temporary_files=False)
    assert len(result) == len(parcels)
    audit = pd.read_csv(extractor.output_dir / "planet_tile_manifest.csv")
    assert not audit.local_files_deleted.any()
    assert all(Path(path).is_file() for path in audit.path)
    assert list(scratch.rglob("*.nc"))
    assert all(Path(path).is_file() for path in manifest.path)


def test_source_neutral_logs_and_worker_restore(tmp_path):
    import pickle

    parcels, _ = inputs(tmp_path)
    extractor = PlanetBasemapZonalStats(
        parcels, "2024-01-01", "2024-02-29", tmp_path / "output", 2100, parcel_id_field="parcel_code"
    )
    for attribute in ("logger", "source_logger", "filling_logger", "parcel_logger"):
        assert getattr(extractor, attribute).name.startswith("parcel_stats_pipeline.")
    assert extractor.openeo_logger is extractor.source_logger
    assert (extractor.output_dir / "satellite_source.log").is_file()
    assert not (extractor.output_dir / "satellite_openeo.log").exists()
    log = (extractor.output_dir / extractor.LOG_FILE_NAME).read_text(encoding="utf-8")
    assert "Source=Planet Basemaps" in log
    assert "satellites.data_preparation.parcel_stats" not in log
    assert "Sentinel-1" not in log
    restored = pickle.loads(pickle.dumps(extractor))
    for attribute in ("logger", "source_logger", "filling_logger", "parcel_logger"):
        assert getattr(restored, attribute).name.startswith("parcel_stats_pipeline.worker.")
    assert restored.openeo_logger is restored.source_logger
