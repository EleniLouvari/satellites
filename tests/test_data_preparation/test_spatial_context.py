"""Regression tests for saving soil-enriched parcel datasets."""

import geopandas as gpd
import pytest
from geopandas.testing import assert_geodataframe_equal
from shapely.geometry import box

from data_preparation.features import spatial_context
from shared.io import read_data, write_data


@pytest.fixture
def soil_inputs(tmp_path):
    """Create projected parcels and one soil polygon with a known match."""
    parcels = gpd.GeoDataFrame(
        {"parcel_id": [10, 20]},
        geometry=[box(500000, 4200000, 500010, 4200010), box(500100, 4200100, 500110, 4200110)],
        crs="EPSG:2100",
    )
    soil = gpd.GeoDataFrame(
        {"GAIOIKANOT": ["A"]},
        geometry=[box(499990, 4199990, 500020, 4200020)],
        crs=parcels.crs,
    )
    soil_path = tmp_path / "soil.gpkg"
    soil.to_file(soil_path, layer="agrosoil", driver="GPKG", index=False)
    return parcels, soil_path


@pytest.mark.parametrize("suffix", [".gpkg", ".geoparquet"])
@pytest.mark.parametrize("in_place", [True, False])
def test_append_soil_feature_saves_parcels(tmp_path, soil_inputs, suffix, in_place):
    """Save matched and unmatched parcels without changing geometry or CRS."""
    parcels, soil_path = soil_inputs
    parcels_path = tmp_path / f"cleaned_grouped_parcels{suffix}"
    output_path = parcels_path if in_place else tmp_path / f"featured_parcels{suffix}"
    write_data(parcels, str(parcels_path))

    result = spatial_context.append_soil_feature(
        parcels_path, soil_path, output_path, soil_layer="agrosoil", overwrite=in_place,
    )

    saved = read_data(str(output_path))
    assert_geodataframe_equal(saved.drop(columns="GAIOIKANOT"), parcels)
    assert_geodataframe_equal(result.drop(columns="GAIOIKANOT"), parcels)
    assert result.loc[0, "GAIOIKANOT"] == "A"
    assert result["GAIOIKANOT"].isna().tolist() == [False, True]
    assert saved.loc[0, "GAIOIKANOT"] == "A"
    assert saved["GAIOIKANOT"].isna().tolist() == [False, True]
    if suffix == ".gpkg":
        assert gpd.list_layers(output_path)["name"].tolist() == [output_path.stem]
    if not in_place:
        assert_geodataframe_equal(read_data(str(parcels_path)), parcels)
    assert not list(tmp_path.glob(".*.tmp.*"))


def test_append_soil_feature_failed_write_preserves_input(tmp_path, soil_inputs, monkeypatch):
    """An interrupted temporary write must not replace the original GeoPackage."""
    parcels, soil_path = soil_inputs
    parcels_path = tmp_path / "parcels.gpkg"
    write_data(parcels, str(parcels_path))
    original_bytes = parcels_path.read_bytes()

    def fail_after_write(*args, **kwargs):
        """Simulate a failure after a real temporary dataset has been written."""
        write_data(*args, **kwargs)
        raise OSError("Error: Simulated write failure")

    monkeypatch.setattr(spatial_context, "write_data", fail_after_write)
    with pytest.raises(OSError, match="Simulated write failure"):
        spatial_context.append_soil_feature(
            parcels_path, soil_path, parcels_path, soil_layer="agrosoil", overwrite=True,
        )

    assert parcels_path.read_bytes() == original_bytes
    assert not list(tmp_path.glob(".*.tmp.*"))
