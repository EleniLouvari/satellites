"""Focused regression tests for raster cleaning and zonal statistics."""

from __future__ import annotations

import warnings
from io import BytesIO

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from rasterio.transform import from_origin
from shapely.geometry import Point, Polygon

from data_preparation.parcel_stats.openeo import OpenEOZonalStats
from tests.utils import (
    expect_equal,
    expect_false,
    expect_is_not_none,
    expect_isinstance,
    expect_not_in,
    expect_subset,
    expect_true,
)


def test_explicit_openeo_credentials_use_password_oidc_flow() -> None:
    calls = []

    class Connection:
        def authenticate_oidc_resource_owner_password_credentials(self, **kwargs):
            calls.append(("password", kwargs))

        def authenticate_oidc(self):
            calls.append(("default", {}))

    extractor = object.__new__(OpenEOZonalStats)
    extractor.openeo_username = "user@example.com"
    extractor.openeo_password = "secret"

    returned = extractor._authenticate_openeo_connection(Connection())

    expect_isinstance(returned, Connection)
    expect_equal(calls, [("password", {"username": "user@example.com", "password": "secret", "client_id": "cdse-public"})])


def test_missing_openeo_credentials_use_default_oidc_flow() -> None:
    calls = []

    class Connection:
        def authenticate_oidc(self):
            calls.append("default")

    extractor = object.__new__(OpenEOZonalStats)
    extractor.openeo_username = None
    extractor.openeo_password = None

    extractor._authenticate_openeo_connection(Connection())

    expect_equal(calls, ["default"])


def test_openeo_credentials_must_be_supplied_together() -> None:
    extractor = object.__new__(OpenEOZonalStats)
    with pytest.raises(ValueError, match="must be supplied together"):
        extractor._validate_openeo_credentials("user@example.com", None)


def test_sentinel1_orbit_direction_is_normalized_and_validated() -> None:
    extractor = object.__new__(OpenEOZonalStats)

    expect_equal(extractor._validate_sentinel1_orbit_direction(" ascending "), "ASCENDING")
    expect_equal(extractor._validate_sentinel1_orbit_direction("descending"), "DESCENDING")
    expect_equal(extractor._validate_sentinel1_orbit_direction("both"), "BOTH")

    with pytest.raises(ValueError, match="ASCENDING, DESCENDING, or BOTH"):
        extractor._validate_sentinel1_orbit_direction("eastbound")


def test_sentinel1_both_orbits_does_not_add_a_collection_filter() -> None:
    extractor = object.__new__(OpenEOZonalStats)
    extractor.sentinel1_orbit_direction = "BOTH"

    expect_equal(extractor._sentinel1_load_options(), {})


def test_validate_cleaning_options_normalizes_temporal_mode_and_fraction() -> None:
    extractor = object.__new__(OpenEOZonalStats)
    validated = extractor._validate_cleaning_options(
        remove_outliers=True,
        iqr_quantiles=(0.2, 0.8),
        iqr_multiplier=1.5,
        fill_nulls=False,
        iqr_min_valid_pixels=2,
        temporal_fill_mode=" Bidirectional ",
        minimum_parcel_pixels=3,
        minimum_observed_fraction_for_fill=0.4,
    )
    expect_equal(validated[5], "bidirectional")
    expect_equal(validated[7], 0.4)


def test_validate_cleaning_options_rejects_non_boolean_flags() -> None:
    extractor = object.__new__(OpenEOZonalStats)
    with pytest.raises(TypeError, match="remove_outliers must be a bool"):
        extractor._validate_cleaning_options(
            remove_outliers=1,
            iqr_quantiles=(0.25, 0.75),
            iqr_multiplier=1.5,
            fill_nulls=True,
            iqr_min_valid_pixels=2,
            temporal_fill_mode="past_only",
            minimum_parcel_pixels=1,
            minimum_observed_fraction_for_fill=0.5,
        )


def _extractor(**overrides) -> OpenEOZonalStats:
    """Build a lightweight instance for testing pure array operations."""
    extractor = object.__new__(OpenEOZonalStats)
    defaults = {
        "sentinel2_bands": ("B02",),
        "sentinel2_indices": (),
        "sentinel1_bands": (),
        "sentinel1_indices": (),
        "start_date": pd.Timestamp("2024-01-01"),
        "end_date": pd.Timestamp("2024-02-29"),
        "temporal_period": "1M",
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

    def warning(self, *_args, **_kwargs) -> None:
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

    expect_equal(prepared["parcel_code"].tolist(), ["101"])
    expect_equal(prepared["label"].tolist(), [7])
    expect_equal(prepared["farm_name"].tolist(), ["sample"])


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

    expect_isinstance(result, gpd.GeoDataFrame)
    expect_equal(result.crs.to_epsg(), extractor.working_epsg)
    expect_equal(result.loc[1, "geometry"].x, pytest.approx(111_319.49, rel=1e-5))
    expect_equal(result["parcel_code"].tolist(), ["A", "B"])
    expect_equal(result["label"].tolist(), [1, 2])
    expect_not_in("period_start", result.columns)
    expect_equal(result.loc[0, "NDVI_median__20240101"], pytest.approx(0.2))
    expect_equal(result.loc[0, "NDVI_median__20240201"], pytest.approx(0.4))
    expect_equal(result.loc[1, "VV_mean__20240201"], pytest.approx(-7.0))
    expect_equal(result.loc[0, "intersected_pixel_count"], 12)
    expect_not_in("eligible_pixel_count", result)
    expect_not_in("parcel_area_m2", result)
    expect_not_in("approx_pixel_count", result)

    geoparquet = BytesIO()
    result.to_parquet(geoparquet, index=False)
    geoparquet.seek(0)
    restored = gpd.read_parquet(geoparquet)
    expect_equal(restored.crs, result.crs)
    expect_equal(restored["label"].tolist(), [1, 2])


def test_time_series_reshape_rejects_missing_required_columns() -> None:
    extractor = _extractor()
    extractor.PARCEL_ID_FIELD = "parcel_code"
    extractor.parcel_logger = _SilentLogger()
    extractor.parcels = gpd.GeoDataFrame(
        {"parcel_code": ["A"], "geometry": [Point(0, 0)]},
        crs="EPSG:4326",
    )
    data = pd.DataFrame({"parcel_code": ["A"], "NDVI_median": [0.2]})

    with pytest.raises(ValueError, match="Missing columns required for time-series reshaping"):
        extractor._reshape_time_series_for_ml(data)


def test_time_series_reshape_rejects_invalid_period_start_values() -> None:
    extractor = _extractor()
    extractor.PARCEL_ID_FIELD = "parcel_code"
    extractor.parcel_logger = _SilentLogger()
    extractor.parcels = gpd.GeoDataFrame(
        {"parcel_code": ["A"], "geometry": [Point(0, 0)]},
        crs="EPSG:4326",
    )
    data = pd.DataFrame(
        {
            "parcel_code": ["A"],
            "period_start": ["not-a-date"],
            "NDVI_median": [0.2],
        }
    )

    with pytest.raises(ValueError, match="contains 1 invalid timestamps"):
        extractor._reshape_time_series_for_ml(data)


def test_time_series_reshape_rejects_duplicate_parcel_period_rows() -> None:
    extractor = _extractor()
    extractor.PARCEL_ID_FIELD = "parcel_code"
    extractor.parcel_logger = _SilentLogger()
    extractor.parcels = gpd.GeoDataFrame(
        {"parcel_code": ["A"], "geometry": [Point(0, 0)]},
        crs="EPSG:4326",
    )
    data = pd.DataFrame(
        {
            "parcel_code": ["A", "A"],
            "period_start": ["2024-01-01", "2024-01-01"],
            "NDVI_median": [0.2, 0.3],
        }
    )

    with pytest.raises(ValueError, match="at most one row per parcel and period"):
        extractor._reshape_time_series_for_ml(data)


def test_time_series_reshape_rejects_inconsistent_static_columns() -> None:
    extractor = _extractor()
    extractor.PARCEL_ID_FIELD = "parcel_code"
    extractor.parcel_logger = _SilentLogger()
    extractor.parcels = gpd.GeoDataFrame(
        {"parcel_code": ["A"], "geometry": [Point(0, 0)]},
        crs="EPSG:4326",
    )
    data = pd.DataFrame(
        {
            "parcel_code": ["A", "A"],
            "period_start": ["2024-01-01", "2024-02-01"],
            "batch_number": [1, 2],
            "NDVI_median": [0.2, 0.3],
        }
    )

    with pytest.raises(ValueError, match="changes across periods"):
        extractor._reshape_time_series_for_ml(data)


def test_time_series_reshape_rejects_parcel_attribute_collisions() -> None:
    extractor = _extractor()
    extractor.PARCEL_ID_FIELD = "parcel_code"
    extractor.parcel_logger = _SilentLogger()
    extractor.parcels = gpd.GeoDataFrame(
        {
            "parcel_code": ["A"],
            "NDVI_median__20240101": [99.0],
            "geometry": [Point(0, 0)],
        },
        crs="EPSG:4326",
    )
    data = pd.DataFrame(
        {
            "parcel_code": ["A"],
            "period_start": ["2024-01-01"],
            "NDVI_median": [0.2],
        }
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        with pytest.raises(ValueError, match="Parcel attributes collide with generated zonal-statistics columns"):
            extractor._reshape_time_series_for_ml(data)


def test_count_configuration_emits_no_per_layer_count_reducer() -> None:
    extractor = _extractor()

    statistics = extractor._validate_spatial_statistics(["mean", "count", "median"])

    expect_equal(statistics, ("mean", "median"))


def test_parcel_mask_includes_every_intersected_raster_pixel() -> None:
    extractor = _extractor()
    values = np.ones((1, 1, 2, 2), dtype="float32")
    # This small polygon crosses the junction of four 10x10 cells without
    # containing any of their centers.
    geometry = Polygon([(9, 9), (11, 9), (11, 11), (9, 11)])

    _window, mask, _transform = extractor._mask_parcel_window(
        values, geometry, from_origin(0, 20, 10, 10)
    )

    expect_equal(int(mask.sum()), 4)


def test_past_only_fill_does_not_use_future_observation() -> None:
    extractor = _extractor(temporal_fill_mode="past_only")
    values = np.array([np.nan, 1.0, np.nan], dtype=float).reshape(3, 1, 1)

    filled = extractor._fill_temporal_neighbors(values)

    expect_true(np.isnan(filled[0, 0, 0]))
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

    expect_equal(filled[0, 0, 0], 2.0)
    expect_true(np.isnan(filled[0, 0, 1]) or filled[0, 0, 1] == 100.0)


@pytest.mark.parametrize(
    "mode, expected",
    [
        ("past_only", [0, 0, 8, 8, np.nan, np.nan, 20, 20]),
        ("bidirectional", [0, 4, 8, 8, np.nan, 20, 20, 20]),
    ],
)
def test_three_step_temporal_window_uses_only_adjacent_original_pixels(mode, expected) -> None:
    extractor = _extractor(
        temporal_fill_mode=mode, temporal_fill_window_sizes=(3,), spatial_fill_window_sizes=(3, 5, 7)
    )
    values = np.array([0, np.nan, 8, np.nan, np.nan, np.nan, 20, np.nan]).reshape(-1, 1, 1)
    original = values.copy()

    filled = extractor._fill_temporal_neighbors(values)

    np.testing.assert_allclose(filled[:, 0, 0], expected, equal_nan=True)
    np.testing.assert_array_equal(values, original)
    extractor.spatial_fill_window_sizes = (3,)
    np.testing.assert_array_equal(extractor._fill_temporal_neighbors(values), filled)


def test_wider_temporal_windows_retry_only_remaining_gaps() -> None:
    extractor = _extractor(temporal_fill_mode="bidirectional", temporal_fill_window_sizes=(3, 5))
    values = np.array([0, np.nan, np.nan, np.nan, 20], dtype=float).reshape(-1, 1, 1)

    filled = extractor._fill_temporal_neighbors(values)

    np.testing.assert_allclose(filled[:, 0, 0], [0, 0, 10, 20, 20])


@pytest.mark.parametrize("windows", [None, (3,), [3, 5, 7]])
def test_temporal_window_validation_accepts_supported_inputs(windows) -> None:
    expect_equal(_extractor()._validate_temporal_fill_windows(windows), (None if windows is None else tuple(windows)))


@pytest.mark.parametrize(
    "windows, error",
    [(3, TypeError), ([3.0], TypeError), ([True], TypeError), ([], ValueError),
     ([1], ValueError), ([4], ValueError), ([5, 3], ValueError), ([3, 3], ValueError)],
)
def test_temporal_window_validation_rejects_invalid_inputs(windows, error) -> None:
    with pytest.raises(error, match="temporal_fill_window_sizes"):
        _extractor()._validate_temporal_fill_windows(windows)


def test_temporal_window_changes_invalidate_cleaning_checkpoints() -> None:
    extractor = _extractor()
    unlimited_signature = extractor._cleaning_checkpoint_signature()
    extractor.temporal_fill_window_sizes = (3,)
    bounded_signature = extractor._cleaning_checkpoint_signature()
    extractor.temporal_fill_window_sizes = (3, 5)

    expect_equal(len({unlimited_signature, bounded_signature, extractor._cleaning_checkpoint_signature()}), 3)


def test_raster_spatial_fill_can_borrow_from_an_adjacent_parcel() -> None:
    extractor = _extractor()
    values = np.array([[[np.nan, 7.0]]])
    complete_raster_mask = np.ones((1, 2), dtype=bool)

    filled = extractor._fill_spatial_neighbors(values, complete_raster_mask, window_size=3)

    expect_equal(filled[0, 0, 0], 7.0)


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

    expect_equal(window_sizes, [3, 5])


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

    expect_equal(window_sizes, [3, 5, 7, 9])


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

    expect_equal(later_methods, [])
    expect_equal(filled[0, 0, 1], 7.0)
    expect_true(all(counts[stage][0] == 0 for stage in counts))


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

    expect_equal(window_sizes, [3])
    expect_equal(interpolation_calls, [])


def test_variable_fill_rejects_period_labels_with_wrong_length() -> None:
    extractor = _extractor()
    values = np.array([[[1.0]], [[2.0]]])
    parcel_mask = np.ones((1, 1), dtype=bool)

    with pytest.raises(ValueError, match="Expected 2 period labels"):
        extractor._fill_variable_cube(values, parcel_mask, None, None, periods=["2024-01-01"])


def test_variable_fill_returns_zero_stage_counts_when_no_pixels_are_eligible() -> None:
    extractor = _extractor()
    messages = []

    class RecordingLogger:
        def info(self, message, *args):
            messages.append(message % args)

    extractor.logger = RecordingLogger()
    values = np.array([[[1.0, np.nan]], [[np.nan, 3.0]]])
    eligible_mask = np.zeros((1, 2), dtype=bool)

    filled, counts = extractor._fill_variable_cube(
        values, eligible_mask, None, None, return_null_counts=True, variable_name="B02", batch_number=9
    )

    np.testing.assert_array_equal(filled, values)
    expected_keys = {
        "after_temporal_fill",
        "after_3x3_fill",
        "after_5x5_fill",
        "after_7x7_fill",
        "after_9x9_fill",
        "after_interpolation",
    }
    expect_equal(set(counts), expected_keys)
    expect_true(all(np.array_equal(stage_counts, np.array([0, 0], dtype="int64")) for stage_counts in counts.values()))
    expect_true(any("no eligible pixels" in message for message in messages))


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

    expect_is_not_none(restored)
    restored_cleaned, restored_observed, restored_report = restored
    np.testing.assert_array_equal(restored_cleaned, cleaned)
    np.testing.assert_array_equal(restored_observed, observed)
    pd.testing.assert_frame_equal(restored_report, report)
    checkpoint_path, report_path = extractor._cleaning_checkpoint_paths(raw_path)
    expect_true(checkpoint_path.exists())
    expect_true(report_path.exists())
    expect_false(checkpoint_path.with_suffix(".partial.nc").exists())
    with xr.open_dataset(checkpoint_path) as checkpoint:
        expect_equal(set(checkpoint.data_vars), {"cleaned", "observed_mask"})
        expect_equal(checkpoint["observed_mask"].dtype, np.dtype("uint8"))


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

    expect_is_not_none(restored)
    np.testing.assert_array_equal(restored[0], final)
    np.testing.assert_array_equal(restored[1], observed)
    np.testing.assert_array_equal(restored[2], np.zeros_like(final, dtype=bool))
    pd.testing.assert_frame_equal(restored[3], report)
    with xr.open_dataset(extractor._final_checkpoint_path(raw_path)) as checkpoint:
        expect_equal(set(checkpoint.data_vars), {"cleaned", "observed_mask", "temporal_filled_mask"})


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

    expect_is_not_none(restored)
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

    metrics = OpenEOZonalStats._summarize_parcel_filling(
        observed, temporal_mask, final, intersected_pixel_count=2
    )

    expect_equal(metrics["expected_pixel_count"], 8)
    expect_equal(metrics["temporal_filled_pixel_count"], 2)
    expect_equal(metrics["spatial_filled_pixel_count"], 2)
    expect_equal(metrics["temporal_filled_ratio"], pytest.approx(0.25))
    expect_equal(metrics["spatial_filled_ratio"], pytest.approx(0.25))
    expect_equal(metrics["data_reliability_score"], pytest.approx(0.5))


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

    expect_equal(result.loc[0, "geom_area"], pytest.approx(8.0))
    expect_equal(result.loc[0, "geom_interior_area_ratio"], pytest.approx(1.25))
    expect_equal(result.loc[0, "geom_compactness"], pytest.approx(2 * np.pi / 9))
    expect_equal(result.loc[0, "geom_perimeter_area_ratio"], pytest.approx(1.5))
    expect_equal(result.loc[0, "geom_shape_index"], pytest.approx(12 / (2 * np.sqrt(8 * np.pi))))
    expect_equal(result.loc[0, "geom_elongation"], pytest.approx(2.0))
    area_penalty = abs(np.log(10 / 8))
    boundary_penalty = np.log(result.loc[0, "geom_shape_index"])
    elongation_penalty = np.log(2.0)
    expect_equal(result.loc[0, "geom_shape_complexity_score"], pytest.approx(
        np.mean([area_penalty, boundary_penalty, elongation_penalty])
    ))


def test_requested_range_statistic_does_not_create_duplicate_columns_before_pivot() -> None:
    extractor = _extractor(sentinel2_bands=("B02",))
    extractor.PARCEL_ID_FIELD = "parcel_code"
    extractor.parcel_logger = _SilentLogger()
    extractor.parcels = gpd.GeoDataFrame(
        {
            "parcel_code": ["A", "B"],
            "geometry": [Point(0, 0), Point(1, 1)],
        },
        crs="EPSG:4326",
    )
    long_data = pd.DataFrame(
        {
            "parcel_code": ["A", "A", "B", "B"],
            "period_start": ["2024-01-01", "2024-02-01"] * 2,
            "period_end": ["2024-02-01", "2024-03-01"] * 2,
            "pixel_area": [10.0, 10.0, 10.0, 10.0],
            "B02_mean": [1.0, 2.0, 3.0, 4.0],
            "B02_min": [0.5, 1.5, 2.5, 3.5],
            "B02_max": [1.5, 2.5, 3.5, 4.5],
            "B02_range": [1.0, 1.0, 1.0, 1.0],
        }
    )

    enriched = extractor._add_derived_stats(long_data)

    expect_equal(list(enriched.columns).count("B02_range"), 1)

    reshaped = extractor._reshape_time_series_for_ml(enriched)

    expect_equal(reshaped.loc[0, "B02_range__20240101"], pytest.approx(1.0))
    expect_equal(reshaped.loc[1, "B02_range__20240201"], pytest.approx(1.0))


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

    score = OpenEOZonalStats._calculate_geometry_complexity_score(metrics)

    expect_equal(score.iloc[0], pytest.approx(0.0))


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

    expect_true(any("Starting temporal fill (past_only)" in message for message in messages))
    expect_true(any("Starting 3x3 spatial fill" in message for message in messages))
    expect_true(any("Finished 3x3 spatial fill" in message and "filled 8 pixels" in message for message in messages))
    expect_false(any("Starting 5x5 spatial fill" in message for message in messages))
    expect_false(any("Starting nearest interpolation" in message for message in messages))


def test_5x5_spatial_pass_reaches_beyond_3x3_neighborhood() -> None:
    extractor = _extractor()
    values = np.full((1, 5, 5), np.nan)
    values[0, 0, 0] = 4.0
    parcel_mask = np.ones((5, 5), dtype=bool)

    after_3x3 = extractor._fill_spatial_neighbors(values, parcel_mask, window_size=3)
    after_5x5 = extractor._fill_spatial_neighbors(after_3x3, parcel_mask, window_size=5)

    expect_true(np.isnan(after_3x3[0, 2, 2]))
    expect_equal(after_5x5[0, 2, 2], 4.0)


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
    expect_true(np.all(stats["count"] > 1))


def test_iqr_filter_is_skipped_for_unreliable_small_samples() -> None:
    extractor = _extractor(iqr_min_valid_pixels=5)
    values = np.array([[[[1.0, 1.0], [1.0, 100.0]]]])
    parcel_mask = np.ones((2, 2), dtype=bool)

    cleaned, audit = extractor._remove_iqr_outliers(values, parcel_mask, 1, ["2024-01-01"])

    expect_equal(cleaned[0, 0, 1, 1], 100.0)
    expect_true(audit[(0, 0)]["iqr_applied"] is False)


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

    expect_equal(events, ["iqr", "fill"])


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
    expect_equal(row["null_count_before_iqr"], 24)
    expect_equal(row["null_count_after_iqr"], 24)
    expect_equal(row["null_count_after_temporal_fill"], 24)
    expect_equal(row["null_count_after_3x3_fill"], 21)
    expect_equal(row["null_count_after_5x5_fill"], 9)
    expect_equal(row["null_count_after_7x7_fill"], 9)
    expect_equal(row["null_count_after_9x9_fill"], 9)
    expect_equal(row["null_count_after_interpolation"], 9)
    expect_equal(row["filled_count_by_temporal_fill"], 0)
    expect_equal(row["filled_count_by_3x3_fill"], 3)
    expect_equal(row["filled_count_by_5x5_fill"], 12)
    expect_equal(row["filled_count_by_7x7_fill"], 0)
    expect_equal(row["filled_count_by_9x9_fill"], 0)
    expect_equal(row["filled_count_by_interpolation"], 0)
    expect_equal(row["final_null_count"], 9)
    expect_true(bool(row["has_remaining_nulls"]) is True)


def test_low_coverage_period_is_not_imputed_or_used_as_fill_source() -> None:
    extractor = _extractor(minimum_observed_fraction_for_fill=0.50)
    extractor._interpolate_interval_layer = lambda layer, *_args: layer
    values = np.array([[[1.0, np.nan], [np.nan, np.nan]], [[2.0, 2.0], [np.nan, np.nan]]])
    parcel_mask = np.ones((2, 2), dtype=bool)

    filled = extractor._fill_variable_cube(values, parcel_mask, None, None)

    expect_equal(np.isfinite(filled[0]).sum(), 1)
    expect_equal(np.isfinite(filled[1]).sum(), 4)


def test_added_crop_indices_request_their_source_bands() -> None:
    extractor = _extractor()
    extractor.sentinel2_bands = ("B02", "B03", "B04", "B08")
    extractor.sentinel2_indices = ("MSAVI", "MNDWI", "PSRI")

    required = extractor._required_sentinel2_bands()

    expect_subset({"B02", "B03", "B04", "B06", "B08", "B11"}, required)


def test_cube_contains_source_bands_while_output_contains_requested_index() -> None:
    extractor = _extractor()
    extractor.sentinel2_bands = ()
    extractor.sentinel2_indices = ("NDVI",)

    expect_equal(extractor._cube_sensor_variables(), ("B08", "B04"))
    expect_equal(extractor._output_sensor_variables(), ("NDVI",))


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

    expect_equal(selected["variable"].values.tolist(), ["B08", "B04"])
    expect_not_in(999.0, selected.values)


def test_ndvi_is_calculated_from_filled_source_band_values() -> None:
    extractor = _extractor()
    extractor.sentinel2_bands = ()
    extractor.sentinel2_indices = ("NDVI",)
    extractor.sentinel1_bands = ()
    cube_variables = ["B08", "B04"]
    filled_source_bands = np.array([[[[0.8, 0.6]], [[0.2, 0.2]]]], dtype="float32")

    output = extractor._build_local_output_cube(filled_source_bands, cube_variables)

    np.testing.assert_allclose(output[0, 0, 0], [0.6, 0.5], atol=1e-7)


def test_sentinel1_indices_use_linear_power_pixels_before_spatial_reduction() -> None:
    extractor = _extractor()
    extractor.sentinel2_bands = ()
    extractor.sentinel2_indices = ()
    extractor.sentinel1_bands = ()
    extractor.sentinel1_indices = ("R", "RVI")
    extractor.spatial_statistics = ("mean",)
    cube_variables = ["VV", "VH"]
    linear_power = np.array([[[[4.0, 1.0]], [[1.0, 2.0]]]], dtype="float32")

    pixel_indices = extractor._build_local_output_cube(linear_power, cube_variables)
    reduced = extractor._reduce_local_pixels(pixel_indices.reshape(1, 2, -1))

    np.testing.assert_allclose(pixel_indices[0, 0, 0], [4.0, 0.5])
    np.testing.assert_allclose(pixel_indices[0, 1, 0], [0.8, 8.0 / 3.0])
    np.testing.assert_allclose(reduced["mean"][0], [2.25, 26.0 / 15.0])
    expect_true(reduced["mean"][0, 0] != pytest.approx(linear_power[:, 0].mean() / linear_power[:, 1].mean()))


def test_sentinel1_indices_reject_db_or_other_negative_inputs() -> None:
    extractor = _extractor(sentinel2_bands=(), sentinel1_indices=("R",))
    db_values = np.array([[[[-8.0]], [[-14.0]]]], dtype="float32")

    with pytest.raises(ValueError, match="linear-power sigma0"):
        extractor._build_local_output_cube(db_values, ["VV", "VH"])


def test_sentinel1_indices_require_and_load_both_polarizations() -> None:
    extractor = _extractor()

    expect_equal(extractor._validate_sentinel1_indices(None), ())
    expect_equal(extractor._validate_sentinel1_indices([]), ())
    expect_equal(extractor._validate_sentinel1_indices(["r", "RVI", "r"]), ("R", "RVI"))

    extractor.sentinel2_bands = ()
    extractor.sentinel1_bands = ()
    extractor.sentinel1_indices = ("R", "RVI")
    expect_equal(extractor._required_sentinel1_bands(), ("VV", "VH"))
    expect_equal(extractor._cube_sensor_variables(), ("VV", "VH"))
    expect_equal(extractor._output_sensor_variables(), ("R", "RVI"))


def test_sentinel2_indices_are_enabled_only_by_a_non_empty_list() -> None:
    extractor = _extractor()

    expect_equal(extractor._validate_sentinel2_indices(None), ())
    expect_equal(extractor._validate_sentinel2_indices([]), ())
    expect_equal(extractor._validate_sentinel2_indices(["ndvi", "NDMI"]), ("NDVI", "NDMI"))


def test_explicit_empty_sentinel2_bands_supports_sentinel1_only_runs() -> None:
    extractor = _extractor()

    expect_equal(extractor._validate_sentinel2_bands([]), ())

    extractor.sentinel2_bands = ()
    extractor.sentinel2_indices = ()
    extractor.sentinel1_bands = ("VV", "VH")
    extractor.sentinel1_indices = ()
    extractor._validate_sensor_selection()
    expect_equal(extractor._output_sensor_variables(), ("VV", "VH"))


def test_sensor_selection_rejects_a_run_without_any_output_variables() -> None:
    extractor = _extractor()
    extractor.sentinel2_bands = ()
    extractor.sentinel2_indices = ()
    extractor.sentinel1_bands = ()

    with pytest.raises(ValueError, match="Select at least one"):
        extractor._validate_sensor_selection()
