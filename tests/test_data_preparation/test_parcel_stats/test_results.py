"""Acquired cubes can be finalized and resumed without an acquisition adapter."""

import json

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from affine import Affine
from shapely.geometry import box

from data_preparation.parcel_stats.core.base import ParcelStatsBase
from tests.utils import expect_equal, expect_is_none, expect_true


def _extractor(tmp_path, processor_class=ParcelStatsBase):
    """Configure a source-independent processor for one two-pixel parcel."""
    return processor_class(
        parcels=gpd.GeoDataFrame({"parcel_id": ["a"], "label": [7]}, geometry=[box(0, 0, 20, 10)], crs=3857),
        output_dir=tmp_path,
        start_date="2024-01-01",
        end_date="2024-01-31",
        working_epsg=3857,
        sentinel2_bands=["B04"],
        sentinel2_indices=[],
        sentinel1_bands=[],
        spatial_statistics=["mean", "median"],
        temporal_reducer="none",
        remove_outliers=False,
        fill_nulls=False,
        batch_workers=1,
        resume_completed_partitions=True,
    )


def _batches(extractor):
    """Write a tiny acquired physical-band cube with hand-computable values."""
    path = extractor.output_dir / "acquired.nc"
    cube = xr.Dataset(
        {"B04": (("t", "y", "x"), np.array([[[2.0, 6.0]]], dtype="float32"))},
        coords={"t": pd.to_datetime(["2024-01-15"]), "x": [5.0, 15.0], "y": [5.0]},
    ).rio.write_crs(3857).rio.write_transform(Affine(10, 0, 0, 0, -10, 10))
    cube.to_netcdf(path)
    return [(1, extractor.parcels, path)]


def test_source_independent_results_preserve_parcels_and_resume(tmp_path):
    """Catch lost outputs or stale resume acceptance when moving result assembly."""
    extractor = _extractor(tmp_path)
    final = extractor.run_from_batches(_batches(extractor))

    expect_equal(final["parcel_id"].tolist(), ["a"])
    expect_equal(final["label"].tolist(), [7])
    np.testing.assert_allclose(final["B04_mean__20240101"], [4.0])
    np.testing.assert_allclose(final["B04_median__20240101"], [4.0])
    saved = gpd.read_parquet(tmp_path / extractor.PARCEL_OUTPUT_FILE_NAME)
    pd.testing.assert_frame_equal(saved, final)
    reduced = gpd.read_parquet(tmp_path / extractor.REDUCED_PARCEL_OUTPUT_FILE_NAME)
    np.testing.assert_allclose(reduced["B04_median_annual_mean"], [4.0])
    expect_true((tmp_path / "satellite_pixel_cleaning_report.csv").is_file())
    marker = json.loads(extractor._partition_completion_path().read_text(encoding="utf-8"))
    expect_equal(marker, {"status": "complete", "signature": extractor._statistics_signature()})
    pd.testing.assert_frame_equal(extractor._load_completed_partition(), final)

    extractor.spatial_statistics = ("median",)
    expect_is_none(extractor._load_completed_partition())


def test_completion_manifest_validates_source_provenance(tmp_path):
    """Catch metadata loss and reuse of outputs from a changed source inventory."""
    class InventoryProcessor(ParcelStatsBase):
        """Attach an acquisition inventory to generic completion records."""

        def _partition_completion_record(self, status):
            """Extend generic completion data with the current source inventory."""
            return super()._partition_completion_record(status) | {"provenance": self.inventory}

    extractor = _extractor(tmp_path, InventoryProcessor)
    extractor.inventory = {"source": "fixture", "manifest_digest": "first"}
    final = extractor.run_from_batches(_batches(extractor))
    marker = json.loads(extractor._partition_completion_path().read_text(encoding="utf-8"))
    expect_equal(marker.get("provenance"), extractor.inventory)
    pd.testing.assert_frame_equal(extractor._load_completed_partition(), final)
    extractor.inventory = {"source": "fixture", "manifest_digest": "second"}
    expect_is_none(extractor._load_completed_partition())


def test_failed_result_publication_leaves_partition_incomplete(tmp_path, monkeypatch):
    """Catch a completion marker committed before all output writes succeed."""
    extractor = _extractor(tmp_path)
    batches = _batches(extractor)
    original = gpd.GeoDataFrame.to_parquet

    def fail_reduced_write(frame, path, **kwargs):
        """Inject a disk failure for the reduced output only."""
        if ".tmp" in str(path):
            raise OSError("disk full")
        return original(frame, path, **kwargs)

    monkeypatch.setattr(gpd.GeoDataFrame, "to_parquet", fail_reduced_write)
    with pytest.raises(OSError, match="disk full"):
        extractor.run_from_batches(batches)
    marker = json.loads(extractor._partition_completion_path().read_text(encoding="utf-8"))
    expect_equal(marker["status"], "processing")
    expect_is_none(extractor._load_completed_partition())
