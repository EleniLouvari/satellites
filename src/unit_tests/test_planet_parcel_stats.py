"""Synthetic seam and schema regression tests for Planet extraction."""
import numpy as np
import pandas as pd
import geopandas as gpd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box
from hub_catalog.planet_parcel_stats import PlanetBasemapZonalStats


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
        for quad, left, red, nir in [('a', 500000, 2, 6), ('b', 500020, 4, 4)]:
            path = tmp_path / f'{month}_{quad}.tif'
            data = np.ones((9, 4, 4), dtype='float32')
            data[5] = red
            data[7] = nir
            data[8] = 999  # Must be replaced by the Planet NDVI formula.
            with rasterio.open(path, 'w', driver='GTiff', width=4, height=4,
                               count=9, dtype='float32', crs='EPSG:2100',
                               transform=from_origin(left, 4500020, 5, 5), nodata=-9999) as dst:
                dst.write(data)
            records.append(dict(period_start=f'2024-{month:02d}-01', quad_id=quad, path=str(path)))
    parcels = gpd.GeoDataFrame({'parcel_code': ['seam'], 'label': ['wheat']},
                              geometry=[box(500015.1, 4500005.1, 500024.9, 4500014.9)], crs=2100)
    return parcels, pd.DataFrame(records)


def test_seam_schema_and_resume(tmp_path):
    parcels, manifest = inputs(tmp_path)
    extractor = PlanetBasemapZonalStats(parcels, '2024-01-01', '2024-02-29',
                                       tmp_path / 'output', 2100, parcel_id_field='parcel_code', buffer_metres=0)
    result = extractor.run(manifest)
    assert len(result) == 1
    assert result.iloc[0]['NDVI_mean__20240101'] == pytest.approx(0.25)
    assert result.iloc[0]['B6_mean__20240101'] == pytest.approx(0.0003)
    assert result.iloc[0]['intersected_pixel_count'] == 4
    assert result.iloc[0]['label'] == 'wheat'
    for index in ["EVI", "NDRE", "NDVI", "NDWI", "SAVI"]:
        for statistic in ["mean", "median", "sd", "min", "max", "range"]:
            assert f"{index}_{statistic}__20240101" in result.columns
    assert result.crs.to_epsg() == 2100
    assert result.iloc[0]['data_reliability_score'] == 1
    assert (tmp_path / 'output' / extractor.REDUCED_PARCEL_OUTPUT_FILE_NAME).exists()
    pd.testing.assert_frame_equal(result, extractor.run(manifest))


def test_missing_quad_month_rejected(tmp_path):
    parcels, manifest = inputs(tmp_path)
    extractor = PlanetBasemapZonalStats(parcels, '2024-01-01', '2024-02-29',
                                       tmp_path / 'out', 2100, parcel_id_field='parcel_code')
    with pytest.raises(ValueError, match='Missing months'):
        extractor.run(manifest.iloc[:-1])


def test_zero_denominator_is_null(tmp_path):
    parcels, _ = inputs(tmp_path)
    extractor = PlanetBasemapZonalStats(parcels, '2024-01-01', '2024-02-29',
                                       tmp_path / 'out', 2100, parcel_id_field='parcel_code')
    result = extractor._calculate_local_sentinel2_index({'B8': np.array([0., 6.]), 'B6': np.array([0., 2.])}, 'NDVI')
    assert np.isnan(result[0])
    assert result[1] == pytest.approx(0.5)

def test_catalog_cross_year_and_download_resume(tmp_path, monkeypatch):
    from hub_catalog.planet_parcel_stats import download_monthly_tiles
    from common_libraries import catalog_s3_checks

    queries, downloads = [], []
    class Catalog:
        client_id = 'internal'
        def fetch_features(self, search_body, max_items):
            identifier = search_body['filter']['args'][1].strip('%')
            queries.append(identifier)
            assert max_items == 2
            return [{'id': identifier}]

    def download(collection, item_id, root, **kwargs):
        downloads.append(item_id)
        folder = root / collection / item_id
        folder.mkdir(parents=True)
        with rasterio.open(folder / 'Analysis.tif', 'w', driver='GTiff', width=2, height=2,
                           count=8, dtype='float32', crs='EPSG:2100',
                           transform=from_origin(500000, 4500020, 5, 5)) as dst:
            dst.write(np.ones((8, 2, 2), dtype='float32'))
        return {'failed': []}

    monkeypatch.setattr(catalog_s3_checks, 'download_all_assets_for_item', download)
    catalog = Catalog()
    tiles = pd.DataFrame({'quad_id': ['1234-5678']})
    result = download_monthly_tiles(catalog, tiles, '2023-11-01', '2024-02-29', tmp_path)
    assert result.period_start.tolist() == ['2023-11-01', '2023-12-01', '2024-01-01', '2024-02-01']
    assert len(downloads) == 4
    download_monthly_tiles(catalog, tiles, '2023-11-01', '2024-02-29', tmp_path)
    assert len(downloads) == 4


def test_nodata_and_temporal_fill(tmp_path):
    parcels, manifest = inputs(tmp_path)
    for path in manifest[manifest.period_start == '2024-02-01'].path:
        with rasterio.open(path, 'r+') as dst:
            data = dst.read()
            data[:, 1, :] = -9999
            dst.write(data)
    extractor = PlanetBasemapZonalStats(
        parcels, '2024-01-01', '2024-02-29', tmp_path / 'output', 2100,
        parcel_id_field='parcel_code', buffer_metres=5,
        fill_nulls=True, temporal_fill_mode='past_only', remove_outliers=True,
    )
    result = extractor.run(manifest)
    assert result.iloc[0]['NDVI_mean__20240201'] == pytest.approx(0.25)
    assert result.iloc[0]['temporal_filled_pixel_count'] > 0
    assert result.iloc[0]['data_reliability_score'] < 1


@pytest.mark.parametrize("index, expected", [
    ("EVI", 2.5 * 0.4 / (0.6 + 6 * 0.2 - 7.5 * 0.1 + 1)),
    ("NDRE", 0.2 / 1.0), ("NDVI", 0.4 / 0.8),
    ("NDWI", -0.3 / 0.9), ("SAVI", 1.5 * 0.4 / 1.3),
])
def test_planet_index_formulas(index, expected):
    extractor = object.__new__(PlanetBasemapZonalStats)
    bands = {name: np.array([value], dtype="float32") for name, value in
             {"B2": 0.1, "B4": 0.3, "B6": 0.2, "B7": 0.4, "B8": 0.6}.items()}
    assert extractor._calculate_local_sentinel2_index(bands, index)[0] == pytest.approx(expected)


def test_ndmi_rejected(tmp_path):
    parcels, _ = inputs(tmp_path)
    with pytest.raises(ValueError, match="no SWIR"):
        PlanetBasemapZonalStats(parcels, "2024-01-01", "2024-02-29", tmp_path / "out", 2100,
                               parcel_id_field="parcel_code", indices=["NDMI"])

class LocalTileCatalog:
    """Exercise the real download helper using local files and no credentials."""
    client_id = 'internal'
    base_url = 'local-test-catalog'

    def __init__(self, manifest, scratch):
        import calendar
        self.scratch = scratch
        self.items = {}
        self.searches = []
        self.fail_identifier = None
        for row in manifest.itertuples():
            month = pd.Timestamp(row.period_start)
            identifier = f'{calendar.month_name[month.month].lower()}_{month.year}_{row.quad_id}_'
            self.items[identifier] = {
                'id': identifier,
                'assets': {
                    'analysis': {'href': row.path, 'hub:relativePath': 'Analysis.tif'},
                    # Downloading unneeded assets would fail; streaming must filter them.
                    'preview': {'href': 'missing_preview.png'},
                },
            }

    def fetch_features(self, search_body, max_items):
        if 'ids' in search_body:
            return [self.items[search_body['ids'][0]]]
        identifier = search_body['filter']['args'][1].strip('%')
        if identifier == self.fail_identifier:
            raise RuntimeError('simulated catalog failure')
        # Previous months and batches must already be gone when a new month starts.
        files = list(self.scratch.rglob('*.tif')) if self.scratch.exists() else []
        month_prefix = identifier.split('_', 2)[:2]
        assert all(path.parent.name.split('_', 2)[:2] == month_prefix for path in files)
        self.searches.append((identifier, len(files)))
        return [self.items[identifier]]


def streaming_inputs(tmp_path):
    from pathlib import Path
    parcels, manifest = inputs(tmp_path)
    manifest['path'] = manifest.path.map(lambda value: str(Path(value).resolve()))
    parcels = gpd.GeoDataFrame(
        {'parcel_code': ['left', 'right', 'seam'], 'label': ['wheat'] * 3},
        geometry=[box(500005.1, 4500005.1, 500014.9, 4500014.9),
                  box(500025.1, 4500005.1, 500034.9, 4500014.9), parcels.geometry.iloc[0]], crs=2100)
    tiles = gpd.GeoDataFrame({'quad_id': ['a', 'b']},
                            geometry=[box(500000, 4500000, 500020, 4500020),
                                      box(500020, 4500000, 500040, 4500020)], crs=2100)
    return parcels, manifest, tiles


def test_streaming_seams_cleanup_and_resume(tmp_path):
    parcels, manifest, tiles = streaming_inputs(tmp_path)
    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    sentinel = scratch / 'user_file.txt'
    sentinel.write_text('keep me')
    catalog = LocalTileCatalog(manifest, scratch)
    extractor = PlanetBasemapZonalStats(parcels, '2024-01-01', '2024-02-29',
                                       tmp_path / 'output', 2100, parcel_id_field='parcel_code', buffer_metres=0)
    result = extractor.run_catalog(catalog, tiles, scratch_dir=scratch)
    assert len(result) == 3
    means = result.set_index('parcel_code')['NDVI_mean__20240101']
    assert means['left'] == pytest.approx(0.5)
    assert means['right'] == pytest.approx(0)
    assert means['seam'] == pytest.approx(0.25)
    assert result.parcel_code.is_unique
    assert len(catalog.searches) == 8  # Two months for a, b, and the a+b seam.
    assert max(count for _, count in catalog.searches) == 1  # At most two tiles coexist.
    assert sorted(path.name for path in scratch.iterdir()) == ['user_file.txt']
    assert sentinel.read_text() == 'keep me'
    assert all(__import__('pathlib').Path(path).exists() for path in manifest.path)
    assert not list((tmp_path / 'output').rglob('*.nc'))
    pd.testing.assert_frame_equal(result, extractor.run_catalog(catalog, tiles, scratch_dir=scratch))
    assert len(catalog.searches) == 8  # Completed results require no new downloads.
    audit = pd.read_csv(tmp_path / 'output' / 'planet_tile_manifest.csv')
    assert audit.local_files_deleted.all()


def test_streaming_failure_cleanup_and_restart(tmp_path):
    parcels, manifest, tiles = streaming_inputs(tmp_path)
    scratch = tmp_path / 'scratch'
    catalog = LocalTileCatalog(manifest, scratch)
    # First quad commits; fail during the second month's download for quad b.
    catalog.fail_identifier = 'february_2024_b_'
    extractor = PlanetBasemapZonalStats(parcels, '2024-01-01', '2024-02-29',
                                       tmp_path / 'output', 2100, parcel_id_field='parcel_code', buffer_metres=0)
    with pytest.raises(RuntimeError, match='simulated'):
        extractor.run_catalog(catalog, tiles, scratch_dir=scratch)
    assert list(scratch.iterdir()) == []
    assert len(list((tmp_path / 'output' / 'streaming_results').rglob('batch_*.json'))) == 1
    assert not (tmp_path / 'output' / extractor.PARCEL_OUTPUT_FILE_NAME).exists()
    first_quad_searches = catalog.searches.count(('january_2024_a_', 0))
    catalog.fail_identifier = None
    result = extractor.run_catalog(catalog, tiles, scratch_dir=scratch)
    assert len(result) == 3
    # Quad a's single-tile batch was skipped; it was fetched again only for the seam.
    assert catalog.searches.count(('january_2024_a_', 0)) == first_quad_searches + 1
    assert list(scratch.iterdir()) == []
