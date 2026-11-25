"""New web extractions use conservative resources without rewriting saved choices."""

import geopandas as gpd
import pytest
from fastapi.testclient import TestClient
from shapely.geometry import box

from data_preparation.parcel_stats.run_config import prepare_run
from parcel_app.app import create_app
from parcel_app.core.security import current_user
from parcel_app.pipelines.parcel_stats.config import check_processing, configuration


def _processing(**overrides):
    """Prepare valid dated form values from the actual server defaults."""
    return configuration(user={})["defaults"] | {
        "start_date": "2024-01-01", "end_date": "2024-01-31", "working_epsg": 3857,
    } | overrides


def test_configuration_endpoint_supplies_slow_disk_form_defaults():
    """The browser receives usable numeric/null metadata and a low-concurrency profile."""
    app = create_app()
    app.dependency_overrides[current_user] = lambda: {"app_user_id": 1}
    with TestClient(app) as client:
        response = client.get("/api/config")
    assert response.status_code == 200
    schema = response.json()
    defaults = schema["defaults"]
    expected = {
        "batch_workers": 1, "openeo_parallel_jobs": 1, "tile_size_metres": 5000,
        "temporal_fill_block_rows": None, "fit_tiles_to_parcels": True, "remove_nc_after_completion": True,
    }
    for name, value in expected.items():
        assert defaults[name] == value
    fields = {field["name"]: field for field in schema["fields"]}
    for name, value in expected.items():
        assert fields[name]["default"] == value
    assert fields["temporal_fill_block_rows"]["type"] == "number"
    assert fields["temporal_fill_block_rows"]["nullable"] is True
    assert defaults["openeo_parallel_jobs"] in fields["openeo_parallel_jobs"]["choices"]
    assert defaults["batch_workers"] <= schema["limits"]["max_batch_workers"]
    assert "resume_completed_partitions" not in defaults  # Execution-owned, already enabled by the runner.
    assert defaults["sentinel2_bands"] == ["B04", "B08"]
    assert defaults["sentinel2_indices"] == ["NDVI"]
    assert defaults["spatial_statistics"] == ["mean", "median"]
    assert defaults["temporal_reducer"] == "none"
    assert defaults["fill_nulls"] is False


def test_default_auto_blocks_survive_validation_when_filling_is_enabled():
    """Blank automatic sizing must reach the engine instead of restoring the old row count."""
    values = _processing(fill_nulls=True)
    checked = check_processing(values)
    assert checked["temporal_fill_block_rows"] is None
    assert checked["batch_workers"] == 1
    assert checked["openeo_parallel_jobs"] == 1
    assert checked["tile_size_metres"] == 5000
    assert values["temporal_fill_block_rows"] is None


def test_saved_resource_choices_are_not_replaced_by_new_defaults():
    """Explicit earlier settings remain valid and take priority when a run is edited or resumed."""
    saved = _processing(
        fill_nulls=True, batch_workers=2, openeo_parallel_jobs=2, tile_size_metres=20000, temporal_fill_block_rows=2048
    )
    checked = check_processing(saved)
    for name in ("batch_workers", "openeo_parallel_jobs", "tile_size_metres", "temporal_fill_block_rows"):
        assert checked[name] == saved[name]


def test_new_web_profile_passes_offline_tile_preflight(tmp_path):
    """The complete form settings produce small tile plans without connecting or creating run outputs."""
    source = tmp_path / "parcels.parquet"
    gpd.GeoDataFrame(
        {"parcel_id": ["west", "east"]}, geometry=[box(0, 0, 100, 100), box(9000, 0, 9100, 100)], crs=3857
    ).to_parquet(source)
    output = tmp_path / "run"
    prepared = prepare_run({
        "schema_version": 1,
        "input": {"path": str(source), "parcel_id_field": "parcel_id"},
        "output_dir": str(output),
        "processing": check_processing(_processing(fill_nulls=True)),
    })
    assert prepared.batch_count == 2
    assert prepared.config["scheduling"] == "batches"
    assert prepared.config["processing"]["temporal_fill_block_rows"] is None
    assert prepared.config["processing"]["tile_size_metres"] == 5000
    assert not output.exists()


def test_new_profile_reports_parcels_that_need_larger_tiles(tmp_path):
    """The smaller default fails clearly when one whole parcel plus its buffer cannot fit."""
    source = tmp_path / "large-parcel.parquet"
    gpd.GeoDataFrame({"parcel_id": ["large"]}, geometry=[box(0, 0, 4500, 100)], crs=3857).to_parquet(source)
    with pytest.raises(ValueError, match="Increase tile_size_metres"):
        prepare_run({
            "schema_version": 1,
            "input": {"path": str(source), "parcel_id_field": "parcel_id"},
            "output_dir": str(tmp_path / "run"),
            "processing": check_processing(_processing()),
        })
