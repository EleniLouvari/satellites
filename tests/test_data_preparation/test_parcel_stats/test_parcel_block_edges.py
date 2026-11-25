"""Block I/O preserves temporal edge cases and limits overlapping parcel masks."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from rasterio.transform import from_origin
from shapely.geometry import box

from data_preparation.parcel_stats.core import streamed_raster
from data_preparation.parcel_stats.openeo import OpenEOZonalStats
from tests.test_data_preparation.test_parcel_stats.test_temporal_pixel_pooling import _extractor, _reference_parcel_statistics


@pytest.mark.parametrize("require", [False, True])
@pytest.mark.parametrize("payload_budget", [1, 64 * 1024**2])
def test_composite_and_missing_period_match_original_reducers(tmp_path, monkeypatch, require, payload_budget):
    """Missing composites stay empty under both all-date and period-sliced reads."""
    monkeypatch.setattr(streamed_raster, "PARCEL_BLOCK_BYTES", payload_budget)
    extractor = _extractor(
        tmp_path, temporal_reducer="median", sentinel2_bands=["B04"], sentinel2_indices=[], require_parcel_observation=require
    )
    final = xr.Dataset({
        "cleaned": (("period", "variable", "y", "x"), np.array([
            [[[1, np.nan], [3, 4]]], [[[9, 10], [11, 12]]],
        ], dtype="float32")),
        "observed_mask": (("period", "variable", "y", "x"), np.array([
            [[[1, 0], [1, 1]]], [[[0, 0], [0, 0]]],
        ], dtype="uint8")),
        "temporal_filled_mask": (("period", "variable", "y", "x"), np.array([
            [[[0, 0], [0, 0]]], [[[1, 0], [0, 0]]],
        ], dtype="uint8")),
    })
    periods = ["2024-01-01", "2024-02-01", "2024-03-01"]
    context = {
        "transform": from_origin(0, 20, 10, 10),
        "projected": extractor.parcels,
        "result_periods": periods,
        "period_groups": [np.array([0]), np.array([], dtype=int), np.array([1])],
        "variables": ["B04"],
        "period_end_lookup": dict(zip(periods, ["2024-02-01", "2024-03-01", "2024-04-01"])),
    }
    expected = _reference_parcel_statistics(extractor, final, extractor.parcels, context)
    result = pd.DataFrame(extractor._build_streamed_stat_rows(final, extractor.parcels, context))
    assert result.parcel_id.drop_duplicates().tolist() == extractor.parcels.parcel_id.tolist()
    for row in result.to_dict("records"):
        for statistic in extractor.spatial_statistics:
            name = f"B04_{statistic}"
            np.testing.assert_array_equal(row[name], expected[(row["parcel_id"], row["period_start"], name)])
        counts = expected[(row["parcel_id"], "counts")]
        assert row["observed_count"] == counts["observed"]
        assert row["temporal_filled_pixel_count"] == counts["temporal"]
        assert row["spatial_filled_pixel_count"] == counts["spatial"]
    assert result.loc[result.period_start == "2024-02-01", "B04_mean"].isna().all()


def test_many_overlapping_masks_are_split_before_rasterization(monkeypatch):
    """A tiny shared raster union must not permit arbitrarily many retained parcel masks."""
    monkeypatch.setattr(streamed_raster, "PARCEL_MASK_BYTES", 128, raising=False)
    extractor = OpenEOZonalStats.__new__(OpenEOZonalStats)
    # Each independent 8x8 boolean mask occupies 64 bytes, despite sharing the exact same raster window.
    geometries = [box(0.1, 0.1, 7.9, 7.9)] * 9
    groups = extractor._parcel_read_groups(geometries, from_origin(0, 8, 1, 1), (8, 8), [np.array([0])])
    assert sorted(position for group in groups for position in group) == list(range(9))
    assert all(len(group) * 64 <= 128 for group in groups)
