"""Focused regression tests for raster cleaning and zonal statistics."""

from __future__ import annotations

from io import BytesIO

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import Point, Polygon
from rasterio.transform import from_origin

from openeo_parcel_stats_pipeline.zonal_stats import SatelliteZonalStats


def test_explicit_openeo_credentials_use_password_oidc_flow() -> None:
    calls = []

    class Connection:
        def authenticate_oidc_resource_owner_password_credentials(self, **kwargs):
            calls.append(("password", kwargs))

        def authenticate_oidc(self):
            calls.append(("default", {}))

    extractor = object.__new__(SatelliteZonalStats)
    extractor.openeo_username = "user@example.com"
    extractor.openeo_password = "secret"

    returned = extractor._authenticate_openeo_connection(Connection())

    assert isinstance(returned, Connection)
    assert calls == [("password", {"username": "user@example.com", "password": "secret", "client_id": "cdse-public"})]


def test_missing_openeo_credentials_use_default_oidc_flow() -> None:
    calls = []

    class Connection:
        def authenticate_oidc(self):
            calls.append("default")

    extractor = object.__new__(SatelliteZonalStats)
    extractor.openeo_username = None
    extractor.openeo_password = None

    extractor._authenticate_openeo_connection(Connection())

    assert calls == ["default"]


def test_openeo_credentials_must_be_supplied_together() -> None:
    extractor = object.__new__(SatelliteZonalStats)
    with pytest.raises(ValueError, match="must be supplied together"):
        extractor._validate_openeo_credentials("user@example.com", None)


def _extractor(**overrides) -> SatelliteZonalStats:
    """Build a lightweight instance for testing pure array operations."""
    extractor = object.__new__(SatelliteZonalStats)
    defaults = {
        "sentinel2_bands": ("B02",),
        "sentinel2_indices": (),
        "sentinel1_bands": (),
        "temporal_fill_mode": "past_only",
        "remove_outliers": True,
        "iqr_quantiles": (0.25, 0.75),
        "iqr_multiplier": 1.5,
        "iqr_min_valid_pixels": 5,
        "minimum_observed_fraction_for_fill": 0.20,
        "interpolation_method": "nearest",
        "interpolation_max_distance_in_meters": None,
        "interpolation_variogram_lags": 15,
        "interpolation_variogram_max_distance_in_meters": None,
        "spatial_fill_window_sizes": (3, 5),
        "working_epsg": 3857,
    }
    for name, value in {**defaults, **overrides}.items():
        setattr(extractor, name, value)
    return extractor


class _SilentLogger:
    def info(self, *_args, **_kwargs) -> None:
        pass


def test_parcel_preparation_preserves_ml_attributes() -> None:
    extractor = _extractor()
    extractor.PARCEL_ID_FIELD = "parcel_code"
    extractor.logger = _SilentLogger()
    parcels = gpd.GeoDataFrame(
        {"parcel_code": [101], "label": [7], "farm_name": ["sample"], "geometry": [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]},
        crs="EPSG:4326",
    )

    prepared = extractor._prepare_parcels(parcels)

    assert prepared["parcel_code"].tolist() == ["101"]
    assert prepared["label"].tolist() == [7]
    assert prepared["farm_name"].tolist() == ["sample"]


def test_time_series_reshape_creates_one_ml_ready_geoparquet_row_per_parcel() -> None:
    extractor = _extractor()
    extractor.PARCEL_ID_FIELD = "parcel_code"
    extractor.parcel_logger = _SilentLogger()
    extractor.parcels = gpd.GeoDataFrame(
        {"parcel_code": ["A", "B"], "label": [1, 2], "geometry": [Point(0, 0), Point(1, 1)]}, crs="EPSG:4326"
    )
    long_data = pd.DataFrame(
        {
            "parcel_code": ["A", "A", "B", "B"],
            "period_start": ["2024-01-01", "2024-02-01"] * 2,
            "period_end": ["2024-02-01", "2024-03-01"] * 2,
            "batch_number": [1, 1, 2, 2],
            "intersected_pixel_count": [12, 12, 8, 8],
            "meets_minimum_pixel_count": [True, True, True, True],
            "geom_area": [1200.0, 1200.0, 800.0, 800.0],
            "NDVI_median": [0.2, 0.4, 0.3, 0.5],
            "VV_mean": [-10.0, -9.0, -8.0, -7.0],
        }
    )

    result = extractor._reshape_time_series_for_ml(long_data)

    assert isinstance(result, gpd.GeoDataFrame)
    assert result.crs.to_epsg() == extractor.working_epsg
    assert result.loc[1, "geometry"].x == pytest.approx(111_319.49, rel=1e-5)
    assert result["parcel_code"].tolist() == ["A", "B"]
    assert result["label"].tolist() == [1, 2]
    assert "period_start" not in result.columns
    assert result.loc[0, "NDVI_median__20240101"] == pytest.approx(0.2)
    assert result.loc[0, "NDVI_median__20240201"] == pytest.approx(0.4)
    assert result.loc[1, "VV_mean__20240201"] == pytest.approx(-7.0)
    assert result.loc[0, "intersected_pixel_count"] == 12
    assert "eligible_pixel_count" not in result
    assert "parcel_area_m2" not in result
    assert "approx_pixel_count" not in result

    geoparquet = BytesIO()
    result.to_parquet(geoparquet, index=False)
    geoparquet.seek(0)
    restored = gpd.read_parquet(geoparquet)
    assert restored.crs == result.crs
    assert restored["label"].tolist() == [1, 2]


def test_count_configuration_emits_no_per_layer_count_reducer() -> None:
    extractor = _extractor()

    statistics = extractor._validate_spatial_statistics(["mean", "count", "median"])

    assert statistics == ("mean", "median")


def test_parcel_mask_includes_every_intersected_raster_pixel() -> None:
    extractor = _extractor()
    values = np.ones((1, 1, 2, 2), dtype="float32")
    # This small polygon crosses the junction of four 10x10 cells without
    # containing any of their centers.
    geometry = Polygon([(9, 9), (11, 9), (11, 11), (9, 11)])

    _window, mask, _transform = extractor._mask_parcel_window(
        values, geometry, from_origin(0, 20, 10, 10)
    )

    assert int(mask.sum()) == 4


def test_past_only_fill_does_not_use_future_observation() -> None:
    extractor = _extractor(temporal_fill_mode="past_only")
    values = np.array([np.nan, 1.0, np.nan], dtype=float).reshape(3, 1, 1)

    filled = extractor._fill_temporal_neighbors(values)

    assert np.isnan(filled[0, 0, 0])
    np.testing.assert_allclose(filled[1:, 0, 0], [1.0, 1.0])


def test_bidirectional_fill_is_explicitly_available_for_retrospective_runs() -> None:
    extractor = _extractor(temporal_fill_mode="bidirectional")
    values = np.array([np.nan, 1.0, np.nan], dtype=float).reshape(3, 1, 1)

    filled = extractor._fill_temporal_neighbors(values)

    np.testing.assert_allclose(filled[:, 0, 0], [1.0, 1.0, 1.0])


def test_bidirectional_fill_averages_bracketing_values_across_consecutive_gaps() -> None:
    extractor = _extractor(temporal_fill_mode="bidirectional")
    values = np.array([0.0, np.nan, np.nan, 9.0], dtype=float).reshape(4, 1, 1)

    filled = extractor._fill_temporal_neighbors(values)

    np.testing.assert_allclose(filled[:, 0, 0], [0.0, 4.5, 4.5, 9.0])


def test_spatial_fill_ignores_pixels_outside_eligible_mask() -> None:
    extractor = _extractor()
    values = np.array([[[np.nan, 100.0], [2.0, 100.0]]])
    parcel_mask = np.array([[True, False], [True, False]])

    filled = extractor._fill_spatial_neighbors(values, parcel_mask)

    assert filled[0, 0, 0] == 2.0
    assert np.isnan(filled[0, 0, 1]) or filled[0, 0, 1] == 100.0


def test_raster_spatial_fill_can_borrow_from_an_adjacent_parcel() -> None:
    extractor = _extractor()
    values = np.array([[[np.nan, 7.0]]])
    complete_raster_mask = np.ones((1, 2), dtype=bool)

    filled = extractor._fill_spatial_neighbors(values, complete_raster_mask, window_size=3)

    assert filled[0, 0, 0] == 7.0


def test_variable_fill_applies_3x3_then_5x5_spatial_passes() -> None:
    extractor = _extractor(minimum_observed_fraction_for_fill=0.0)
    window_sizes = []

    def record_spatial_pass(values, _eligible_mask, window_size=3):
        window_sizes.append(window_size)
        return values

    extractor._fill_spatial_neighbors = record_spatial_pass
    extractor._interpolate_interval_layer = lambda layer, *_args: layer
    values = np.array([[[1.0, np.nan], [np.nan, np.nan]]])
    parcel_mask = np.ones((2, 2), dtype=bool)

    extractor._fill_variable_cube(values, parcel_mask, None, None)

    assert window_sizes == [3, 5]


def test_optional_spatial_fill_windows_add_7x7_and_9x9_passes() -> None:
    extractor = _extractor(spatial_fill_window_sizes=(3, 5, 7, 9), minimum_observed_fraction_for_fill=0.0)
    window_sizes = []

    def record_spatial_pass(values, _eligible_mask, window_size=3):
        window_sizes.append(window_size)
        return values

    extractor._fill_spatial_neighbors = record_spatial_pass
    extractor._interpolate_interval_layer = lambda layer, *_args: layer
    extractor._fill_variable_cube(
        np.array([[[1.0, np.nan]]]), np.ones((1, 2), dtype=bool), None, None
    )

    assert window_sizes == [3, 5, 7, 9]


def test_variable_fill_stops_after_temporal_fill_when_no_nulls_remain() -> None:
    extractor = _extractor(minimum_observed_fraction_for_fill=0.0)
    later_methods = []
    extractor._fill_temporal_neighbors = lambda values: np.nan_to_num(values, nan=7.0)
    extractor._fill_spatial_neighbors = lambda *args, **kwargs: later_methods.append("spatial")
    extractor._interpolate_interval_layer = lambda *args, **kwargs: later_methods.append("interpolation")
    values = np.array([[[1.0, np.nan]]])

    filled, counts = extractor._fill_variable_cube(
        values, np.ones((1, 2), dtype=bool), None, None, return_null_counts=True
    )

    assert later_methods == []
    assert filled[0, 0, 1] == 7.0
    assert all(counts[stage][0] == 0 for stage in counts)


def test_variable_fill_stops_after_3x3_fill_when_no_nulls_remain() -> None:
    extractor = _extractor(minimum_observed_fraction_for_fill=0.0)
    window_sizes = []
    interpolation_calls = []

    def fill_spatial(values, _eligible_mask, window_size=3):
        window_sizes.append(window_size)
        return np.nan_to_num(values, nan=5.0)

    extractor._fill_temporal_neighbors = lambda values: values
    extractor._fill_spatial_neighbors = fill_spatial
    extractor._interpolate_interval_layer = lambda *args: interpolation_calls.append(True)

    extractor._fill_variable_cube(np.array([[[1.0, np.nan]]]), np.ones((1, 2), dtype=bool), None, None)

    assert window_sizes == [3]
    assert interpolation_calls == []


def test_cleaning_checkpoint_round_trip_reuses_cleaned_pixels(tmp_path) -> None:
    extractor = _extractor()
    extractor.parcel_logger = _SilentLogger()
    raw_path = tmp_path / "batch_00001_monthly.nc"
    cleaned = np.array([[[[1.0, 2.0]]]], dtype="float32")
    observed = np.array([[[[1.0, np.nan]]]], dtype="float32")
    report = pd.DataFrame({"batch_number": [1], "final_null_count": [0]})
    periods = ["2024-01-01"]
    variables = ["B02"]

    extractor._save_cleaning_checkpoint(raw_path, cleaned, observed, report, periods, variables)
    restored = extractor._load_cleaning_checkpoint(raw_path, periods, variables, cleaned.shape)

    assert restored is not None
    restored_cleaned, restored_observed, restored_report = restored
    np.testing.assert_array_equal(restored_cleaned, cleaned)
    np.testing.assert_array_equal(restored_observed, observed)
    pd.testing.assert_frame_equal(restored_report, report)
    checkpoint_path, report_path = extractor._cleaning_checkpoint_paths(raw_path)
    assert checkpoint_path.exists()
    assert report_path.exists()
    assert not checkpoint_path.with_suffix(".partial.nc").exists()
    with xr.open_dataset(checkpoint_path) as checkpoint:
        assert set(checkpoint.data_vars) == {"cleaned", "observed_mask"}
        assert checkpoint["observed_mask"].dtype == np.dtype("uint8")


def test_final_checkpoint_round_trip_reuses_calculated_indices(tmp_path) -> None:
    extractor = _extractor()
    extractor.parcel_logger = _SilentLogger()
    raw_path = tmp_path / "batch_00001_monthly.nc"
    final = np.array([[[[0.6, 0.5]]]], dtype="float32")
    observed = np.array([[[[0.6, np.nan]]]], dtype="float32")
    report = pd.DataFrame({"batch_number": [1], "final_null_count": [0]})
    _, report_path = extractor._cleaning_checkpoint_paths(raw_path)
    report.to_parquet(report_path, index=False)

    extractor._save_final_checkpoint(raw_path, final, observed, ["2024-01-01"], ["NDVI"])
    restored = extractor._load_final_checkpoint(
        raw_path, ["2024-01-01"], ["NDVI"], final.shape
    )

    assert restored is not None
    np.testing.assert_array_equal(restored[0], final)
    np.testing.assert_array_equal(restored[1], observed)
    np.testing.assert_array_equal(restored[2], np.zeros_like(final, dtype=bool))
    pd.testing.assert_frame_equal(restored[3], report)
    with xr.open_dataset(extractor._final_checkpoint_path(raw_path)) as checkpoint:
        assert set(checkpoint.data_vars) == {"cleaned", "observed_mask", "temporal_filled_mask"}


def test_temporal_cube_checkpoint_mask_distinguishes_temporal_fills(tmp_path) -> None:
    extractor = _extractor()
    extractor.parcel_logger = _SilentLogger()
    raw_path = tmp_path / "batch_00001_monthly.nc"
    final = np.array([[[[1.0, 2.0, 3.0]]]], dtype="float32")
    observed = np.array([[[[1.0, np.nan, np.nan]]]], dtype="float32")
    temporal = np.array([[[[1.0, 2.0, np.nan]]]], dtype="float32")
    _, report_path = extractor._cleaning_checkpoint_paths(raw_path)
    pd.DataFrame({"batch_number": [1]}).to_parquet(report_path, index=False)

    extractor._save_final_checkpoint(
        raw_path,
        final,
        observed,
        ["2024-01-01"],
        ["B02"],
        temporally_filled=temporal,
    )
    restored = extractor._load_final_checkpoint(raw_path, ["2024-01-01"], ["B02"], final.shape)

    assert restored is not None
    np.testing.assert_array_equal(restored[2], [[[[False, True, False]]]])


def test_parcel_fill_summary_sums_all_period_band_pixel_slots() -> None:
    observed = np.array(
        [
            [[1.0, np.nan], [np.nan, 4.0]],
            [[np.nan, 2.0], [3.0, np.nan]],
        ]
    )
    temporal_mask = np.array(
        [
            [[False, True], [False, False]],
            [[False, False], [False, True]],
        ]
    )
    final = np.nan_to_num(observed, nan=9.0)

    metrics = SatelliteZonalStats._summarize_parcel_filling(
        observed, temporal_mask, final, intersected_pixel_count=2
    )

    assert metrics["expected_pixel_count"] == 8
    assert metrics["temporal_filled_pixel_count"] == 2
    assert metrics["spatial_filled_pixel_count"] == 2
    assert metrics["temporal_filled_ratio"] == pytest.approx(0.25)
    assert metrics["spatial_filled_ratio"] == pytest.approx(0.25)
    assert metrics["data_reliability_score"] == pytest.approx(0.5)


def test_geometry_metrics_use_projected_area_perimeter_and_rotated_rectangle() -> None:
    extractor = _extractor()
    extractor.PARCEL_ID_FIELD = "parcel_code"
    extractor.parcels = gpd.GeoDataFrame(
        {
            "parcel_code": ["A"],
            "geometry": [Polygon([(0, 0), (4, 0), (4, 2), (0, 2)])],
        },
        crs="EPSG:3857",
    )
    data = pd.DataFrame(
        {
            "parcel_code": ["A"],
            "period_start": ["2024-01-01"],
            "pixel_area": [10.0],
            "B02_mean": [1.0],
        }
    )

    result = extractor._add_derived_stats(data)

    assert result.loc[0, "geom_area"] == pytest.approx(8.0)
    assert result.loc[0, "geom_interior_area_ratio"] == pytest.approx(1.25)
    assert result.loc[0, "geom_compactness"] == pytest.approx(2 * np.pi / 9)
    assert result.loc[0, "geom_perimeter_area_ratio"] == pytest.approx(1.5)
    assert result.loc[0, "geom_shape_index"] == pytest.approx(12 / (2 * np.sqrt(8 * np.pi)))
    assert result.loc[0, "geom_elongation"] == pytest.approx(2.0)
    area_penalty = abs(np.log(10 / 8))
    boundary_penalty = np.log(result.loc[0, "geom_shape_index"])
    elongation_penalty = np.log(2.0)
    assert result.loc[0, "geom_shape_complexity_score"] == pytest.approx(
        np.mean([area_penalty, boundary_penalty, elongation_penalty])
    )


def test_geometry_complexity_is_zero_for_ideal_simple_metrics() -> None:
    metrics = pd.DataFrame(
        {
            "geom_area": [100.0],
            "geom_interior_area_ratio": [1.0],
            "geom_compactness": [1.0],
            "geom_perimeter_area_ratio": [2.0 * np.sqrt(np.pi / 100.0)],
            "geom_shape_index": [1.0],
            "geom_elongation": [1.0],
        }
    )

    score = SatelliteZonalStats._calculate_geometry_complexity_score(metrics)

    assert score.iloc[0] == pytest.approx(0.0)


def test_variable_fill_logs_each_stage_and_number_of_filled_pixels() -> None:
    extractor = _extractor(minimum_observed_fraction_for_fill=0.0)
    messages = []

    class RecordingLogger:
        def info(self, message, *args):
            messages.append(message % args)

    extractor.filling_logger = RecordingLogger()
    extractor._interpolate_interval_layer = lambda layer, *_args: layer
    values = np.full((1, 3, 3), np.nan)
    values[0, 1, 1] = 4.0
    raster_mask = np.ones((3, 3), dtype=bool)

    extractor._fill_variable_cube(
        values, raster_mask, transform=None, cube_crs=None, variable_name="B02", periods=["2024-01-01"], batch_number=7
    )

    assert any("Starting temporal fill (past_only)" in message for message in messages)
    assert any("Starting 3x3 spatial fill" in message for message in messages)
    assert any("Finished 3x3 spatial fill" in message and "filled 8 pixels" in message for message in messages)
    assert not any("Starting 5x5 spatial fill" in message for message in messages)
    assert not any("Starting nearest interpolation" in message for message in messages)


def test_5x5_spatial_pass_reaches_beyond_3x3_neighborhood() -> None:
    extractor = _extractor()
    values = np.full((1, 5, 5), np.nan)
    values[0, 0, 0] = 4.0
    parcel_mask = np.ones((5, 5), dtype=bool)

    after_3x3 = extractor._fill_spatial_neighbors(values, parcel_mask, window_size=3)
    after_5x5 = extractor._fill_spatial_neighbors(after_3x3, parcel_mask, window_size=5)

    assert np.isnan(after_3x3[0, 2, 2])
    assert after_5x5[0, 2, 2] == 4.0


def test_raster_native_nearest_fills_from_closest_observed_pixel() -> None:
    extractor = _extractor()
    layer = np.array([[1.0, np.nan, np.nan, 9.0]])
    eligible_mask = np.ones(layer.shape, dtype=bool)

    filled = extractor._fill_nearest_raster(
        layer, eligible_mask, from_origin(0, 10, 10, 10)
    )

    np.testing.assert_array_equal(filled, [[1.0, 1.0, 9.0, 9.0]])


def test_nulls_are_filled_per_variable_and_time_before_parcel_reduction() -> None:
    extractor = _extractor(fill_nulls=True, remove_outliers=False, minimum_observed_fraction_for_fill=0.0)
    extractor.sentinel2_bands = ("B02", "B03")
    extractor.spatial_statistics = ("mean", "count")
    extractor._interpolate_interval_layer = lambda layer, *_args: layer
    values = np.full((2, 2, 5, 5), np.nan)
    values[0, 0, 0, 0] = 2.0
    values[0, 1, 0, 0] = 20.0
    values[1, 0, 0, 0] = 4.0
    values[1, 1, 0, 0] = 40.0
    parcel_mask = np.ones((5, 5), dtype=bool)

    cleaned, _observed, _report = extractor._clean_monthly_pixels(
        values, parcel_mask, batch_number=1, periods=["2024-01-01", "2024-02-01"], transform=None, cube_crs=None
    )
    stats = extractor._reduce_local_pixels(cleaned[:, :, parcel_mask])

    np.testing.assert_allclose(stats["mean"], [[2.0, 20.0], [4.0, 40.0]])
    assert np.all(stats["count"] > 1)


def test_iqr_filter_is_skipped_for_unreliable_small_samples() -> None:
    extractor = _extractor(iqr_min_valid_pixels=5)
    values = np.array([[[[1.0, 1.0], [1.0, 100.0]]]])
    parcel_mask = np.ones((2, 2), dtype=bool)

    cleaned, audit = extractor._remove_iqr_outliers(values, parcel_mask, 1, ["2024-01-01"])

    assert cleaned[0, 0, 1, 1] == 100.0
    assert audit[(0, 0)]["iqr_applied"] is False


def test_raster_iqr_removal_runs_once_before_variable_filling() -> None:
    extractor = _extractor(fill_nulls=True)
    events = []
    remove_iqr_outliers = extractor._remove_iqr_outliers
    fill_variable_cube = extractor._fill_variable_cube

    def record_iqr_removal(*args, **kwargs):
        events.append("iqr")
        return remove_iqr_outliers(*args, **kwargs)

    def record_variable_fill(*args, **kwargs):
        events.append("fill")
        return fill_variable_cube(*args, **kwargs)

    extractor._remove_iqr_outliers = record_iqr_removal
    extractor._fill_variable_cube = record_variable_fill
    extractor._interpolate_interval_layer = lambda layer, *_args: layer
    values = np.ones((2, 1, 3, 3), dtype=float)
    raster_mask = np.ones((3, 3), dtype=bool)

    extractor._clean_monthly_pixels(
        values, raster_mask, batch_number=1, periods=["2024-01-01", "2024-02-01"], transform=None, cube_crs=None
    )

    assert events == ["iqr", "fill"]


def test_cleaning_report_records_null_count_after_every_raster_step() -> None:
    extractor = _extractor(fill_nulls=True, remove_outliers=False, minimum_observed_fraction_for_fill=0.0)
    extractor._interpolate_interval_layer = lambda layer, *_args: layer
    values = np.full((1, 1, 5, 5), np.nan)
    values[0, 0, 0, 0] = 4.0
    raster_mask = np.ones((5, 5), dtype=bool)

    _cleaned, _observed, report = extractor._clean_monthly_pixels(
        values, raster_mask, batch_number=1, periods=["2024-01-01"], transform=None, cube_crs=None
    )

    row = report.iloc[0]
    assert row["null_count_before_iqr"] == 24
    assert row["null_count_after_iqr"] == 24
    assert row["null_count_after_temporal_fill"] == 24
    assert row["null_count_after_3x3_fill"] == 21
    assert row["null_count_after_5x5_fill"] == 9
    assert row["null_count_after_7x7_fill"] == 9
    assert row["null_count_after_9x9_fill"] == 9
    assert row["null_count_after_interpolation"] == 9
    assert row["filled_count_by_temporal_fill"] == 0
    assert row["filled_count_by_3x3_fill"] == 3
    assert row["filled_count_by_5x5_fill"] == 12
    assert row["filled_count_by_7x7_fill"] == 0
    assert row["filled_count_by_9x9_fill"] == 0
    assert row["filled_count_by_interpolation"] == 0
    assert row["final_null_count"] == 9
    assert bool(row["has_remaining_nulls"]) is True


def test_low_coverage_period_is_not_imputed_or_used_as_fill_source() -> None:
    extractor = _extractor(minimum_observed_fraction_for_fill=0.50)
    extractor._interpolate_interval_layer = lambda layer, *_args: layer
    values = np.array([[[1.0, np.nan], [np.nan, np.nan]], [[2.0, 2.0], [np.nan, np.nan]]])
    parcel_mask = np.ones((2, 2), dtype=bool)

    filled = extractor._fill_variable_cube(values, parcel_mask, None, None)

    assert np.isfinite(filled[0]).sum() == 1
    assert np.isfinite(filled[1]).sum() == 4


def test_added_crop_indices_request_their_source_bands() -> None:
    extractor = _extractor()
    extractor.sentinel2_bands = ("B02", "B03", "B04", "B08")
    extractor.sentinel2_indices = ("MSAVI", "MNDWI", "PSRI")

    required = extractor._required_sentinel2_bands()

    assert {"B02", "B03", "B04", "B06", "B08", "B11"}.issubset(required)


def test_cube_contains_source_bands_while_output_contains_requested_index() -> None:
    extractor = _extractor()
    extractor.sentinel2_bands = ()
    extractor.sentinel2_indices = ("NDVI",)

    assert extractor._cube_sensor_variables() == ("B08", "B04")
    assert extractor._output_sensor_variables() == ("NDVI",)


def test_existing_raw_index_is_ignored_when_selecting_physical_source_bands() -> None:
    extractor = _extractor()
    dataset = xr.Dataset(
        {
            "B08": (("t", "y", "x"), np.ones((1, 1, 1))),
            "B04": (("t", "y", "x"), np.full((1, 1, 1), 0.5)),
            "NDVI": (("t", "y", "x"), np.full((1, 1, 1), 999.0)),
        }
    )

    selected = extractor._select_monthly_variables(
        dataset, "t", "x", "y", expected_variables=("B08", "B04")
    )

    assert selected["variable"].values.tolist() == ["B08", "B04"]
    assert 999.0 not in selected.values


def test_ndvi_is_calculated_from_filled_source_band_values() -> None:
    extractor = _extractor()
    extractor.sentinel2_bands = ()
    extractor.sentinel2_indices = ("NDVI",)
    extractor.sentinel1_bands = ()
    cube_variables = ["B08", "B04"]
    filled_source_bands = np.array([[[[0.8, 0.6]], [[0.2, 0.2]]]], dtype="float32")

    output = extractor._build_local_output_cube(filled_source_bands, cube_variables)

    np.testing.assert_allclose(output[0, 0, 0], [0.6, 0.5], atol=1e-7)


def test_explicit_empty_sentinel2_bands_supports_sentinel1_only_runs() -> None:
    extractor = _extractor()

    assert extractor._validate_sentinel2_bands([]) == ()

    extractor.sentinel2_bands = ()
    extractor.sentinel2_indices = ()
    extractor.sentinel1_bands = ("VV", "VH")
    extractor._validate_sensor_selection()
    assert extractor._output_sensor_variables() == ("VV", "VH")


def test_sensor_selection_rejects_a_run_without_any_output_variables() -> None:
    extractor = _extractor()
    extractor.sentinel2_bands = ()
    extractor.sentinel2_indices = ()
    extractor.sentinel1_bands = ()

    with pytest.raises(ValueError, match="Select at least one"):
        extractor._validate_sensor_selection()
