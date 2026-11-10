"""Regression coverage for native raster grids and output-only reprojection."""

import json
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from openeo.metadata import Band, BandDimension, CubeMetadata, SpatialDimension, TemporalDimension
from openeo.rest.datacube import DataCube
from shapely.geometry import box

from data_preparation.parcel_stats.core import configuration
from data_preparation.parcel_stats.job_manager import OpenEOJobManagerZonalStats
from data_preparation.parcel_stats.openeo import OpenEOZonalStats
from tests.utils import expect_equal, expect_is_none, expect_true


class _OfflineConnection:
    """Build real openEO graphs without fetching remote collection metadata."""

    @staticmethod
    def load_collection(collection, **kwargs):
        """Return an offline cube with the requested collection and arguments."""
        cube = DataCube.load_collection(collection, **kwargs)
        metadata = CubeMetadata(
            dimensions=[
                SpatialDimension("x", [500000, 500020], crs=32634, step=10),
                SpatialDimension("y", [4200000, 4200020], crs=32634, step=10),
                TemporalDimension("t", kwargs["temporal_extent"]),
                BandDimension("bands", [Band(name) for name in kwargs["bands"]]),
            ]
        )
        cube.metadata = metadata
        return cube


def _extractor(tmp_path, extractor_class=OpenEOZonalStats, **overrides):
    """Use distinct input, raster and output CRSes around a two-by-two UTM grid."""
    parcels = gpd.GeoDataFrame(
        {"parcel_id": ["sample"]}, geometry=[box(500001, 4200001, 500019, 4200019)], crs=32634
    ).to_crs(32635)
    options = {
        "parcels": parcels,
        "start_date": "2025-06-01",
        "end_date": "2025-06-30",
        "output_dir": tmp_path,
        "working_epsg": 2100,
        "sentinel2_bands": ["B04", "B08"],
        "sentinel2_indices": ["NDVI"],
        "sentinel1_bands": [],
        "sentinel1_indices": [],
        "temporal_reducer": "none",
        "spatial_statistics": ["mean", "median", "range"],
        "remove_outliers": False,
        "fill_nulls": False,
        "resume_completed_partitions": True,
    }
    return extractor_class(**(options | overrides))


@pytest.mark.parametrize("extractor_class", [OpenEOZonalStats, OpenEOJobManagerZonalStats])
@pytest.mark.parametrize("sensor", ["optical", "radar", "both"])
@pytest.mark.parametrize("temporal_reducer", ["none", "median"])
def test_acquisition_preserves_native_projection(tmp_path, extractor_class, sensor, temporal_reducer):
    """Neither acquisition entry point may force the output CRS onto raster pixels."""
    extractor = _extractor(
        tmp_path,
        extractor_class,
        sentinel2_bands=[] if sensor == "radar" else ["B04", "B08"],
        sentinel2_indices=[],
        sentinel1_bands=[] if sensor == "optical" else ["VV", "VH"],
        temporal_reducer=temporal_reducer,
    )
    connection = _OfflineConnection()
    if extractor_class is OpenEOJobManagerZonalStats:
        cube = extractor._build_tile_cube(
            connection, {"west": 20.9, "south": 37.9, "east": 21.1, "north": 38.1, "crs": "EPSG:4326"}
        )
    else:
        cube = extractor._build_multisensor_temporal_cube(connection, extractor._to_feature_collection(extractor.parcels))
    graph = cube.flat_graph()
    resampling = [node["arguments"] for node in graph.values() if node["process_id"] == "resample_spatial"]
    expect_true(bool(resampling))
    for arguments in resampling:
        expect_is_none(arguments.get("projection"))
        expect_equal(arguments["resolution"], 10)
    if sensor == "both":
        # Radar must still align to the optical raster before merging sensors.
        merge = next(node["arguments"] for node in graph.values() if node["process_id"] == "merge_cubes" and node.get("result"))
        radar = graph[merge["cube2"]["from_node"]]
        expect_equal(radar["process_id"], "resample_cube_spatial")
        expect_equal(radar["arguments"]["target"], merge["cube1"])


def test_parcels_use_netcdf_crs_and_final_geometry_uses_requested_epsg(tmp_path):
    """Known raster samples must survive different input and output parcel CRSes."""
    extractor = _extractor(tmp_path)
    path = tmp_path / "native.nc"
    dataset = xr.Dataset(
        {
            "B04": (("t", "y", "x"), np.array([[[0.1, 0.2], [0.3, 0.4]]], dtype="float32")),
            "B08": (("t", "y", "x"), np.array([[[0.3, 0.6], [0.9, 1.2]]], dtype="float32")),
        },
        coords={"t": pd.to_datetime(["2025-06-15"]), "x": [500005.0, 500015.0], "y": [4200015.0, 4200005.0]},
    ).rio.write_crs(32634)
    dataset.to_netcdf(path)

    stats, _ = extractor._calculate_local_statistics(path, extractor.parcels, 1)
    expect_equal(stats.iloc[0].intersected_pixel_count, 4)
    expect_equal(stats.iloc[0].B04_median, pytest.approx(0.25))
    expect_equal(stats.iloc[0].B04_range, pytest.approx(0.3))
    expect_equal(stats.iloc[0].NDVI_median, pytest.approx(0.5))
    result = extractor._reshape_time_series_for_ml(extractor._add_derived_stats(stats))
    expect_equal(result.crs.to_epsg(), 2100)
    expect_equal(result.iloc[0].B04_median__20250601, pytest.approx(0.25))
    expect_true(result.geometry.iloc[0].equals_exact(extractor.parcels.to_crs(2100).geometry.iloc[0], 1e-6))
    with xr.open_dataset(path, decode_coords="all") as saved:
        expect_equal(saved.rio.crs.to_epsg(), 32634)


@pytest.mark.parametrize("extractor_class", [OpenEOZonalStats, OpenEOJobManagerZonalStats])
def test_previous_projected_raster_caches_are_not_reused(tmp_path, monkeypatch, extractor_class):
    """Reject completed partitions and raster namespaces from the old acquisition strategy."""
    extractor = _extractor(tmp_path, extractor_class)

    def legacy_dumps(value, **kwargs):
        """Reproduce the previous acquisition signature for the same configuration."""
        return json.dumps(value | {"processing_strategy": "sentinel1_sentinel2_temporal_local_v3"}, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(configuration, "json", SimpleNamespace(dumps=legacy_dumps))
        old_signature = extractor._run_signature()
        extractor._write_partition_completion("complete")
    for name in (extractor.PARCEL_OUTPUT_FILE_NAME, extractor.REDUCED_PARCEL_OUTPUT_FILE_NAME):
        extractor.parcels.to_crs(2100).to_parquet(tmp_path / name)
    pd.DataFrame({"batch_number": [1]}).to_csv(tmp_path / "satellite_pixel_cleaning_report.csv", index=False)

    expect_true(extractor._run_signature() != old_signature)
    expect_is_none(extractor._load_completed_partition())
