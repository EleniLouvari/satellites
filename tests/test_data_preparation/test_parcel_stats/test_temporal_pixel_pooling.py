"""Regression coverage for parcel statistics over unreduced acquisitions."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import box

from data_preparation.parcel_stats.openeo import OpenEOZonalStats
from tests.utils import expect_equal, expect_false, expect_is_none, expect_is_not_none, expect_true


def _extractor(tmp_path, **overrides):
    options = {
        "parcels": gpd.GeoDataFrame(
            {"parcel_id": ["whole", "right", "outside"]},
            geometry=[box(0.1, 0.1, 19.9, 19.9), box(10.1, 0.1, 19.9, 19.9), box(30, 30, 40, 40)],
            crs=3857,
        ),
        "start_date": "2024-01-01",
        "end_date": "2024-03-31",
        "output_dir": tmp_path,
        "working_epsg": 3857,
        "sentinel2_bands": ["B04", "B08"],
        "sentinel2_indices": ["NDVI"],
        "sentinel1_bands": [],
        "sentinel1_indices": [],
        "temporal_reducer": "none",
        "spatial_statistics": ["mean", "median", "sd", "min", "max", "range", "p10", "p25", "p75", "p90"],
        "remove_outliers": False,
        "fill_nulls": False,
        "keep_cleaned_checkpoint": True,
    }
    return OpenEOZonalStats(**(options | overrides))


def _write_cube(path, times, **bands):
    dataset = xr.Dataset(
        {name: (("t", "y", "x"), np.asarray(values, dtype="float32")) for name, values in bands.items()},
        coords={"t": pd.to_datetime(times, format="mixed"), "x": [5.0, 15.0], "y": [15.0, 5.0]},
    ).rio.write_crs(3857)
    dataset.to_netcdf(path)
    return path


@pytest.mark.parametrize("reducer", [None, "none", " NONE "])
def test_no_reducer_option_is_normalized(tmp_path, reducer):
    extractor = _extractor(tmp_path, temporal_reducer=reducer)
    expect_equal(extractor.temporal_reducer, "none")
    cube = object()
    expect_true(extractor._aggregate_temporal_cube(cube) is cube)
    expect_equal(extractor._openeo_temporal_extent(), ["2024-01-01", "2024-04-01"])


def test_pooling_has_a_separate_download_cache(tmp_path):
    extractor = _extractor(tmp_path)
    pooled_signature = extractor._run_signature()
    extractor.temporal_reducer = "median"
    expect_true(extractor._run_signature() != pooled_signature)


def test_monthly_statistics_pool_observations_and_calculate_indices_per_acquisition(tmp_path, monkeypatch):
    extractor = _extractor(tmp_path)
    # Deliberately unordered, including two separate acquisitions on the same day.
    path = _write_cube(
        tmp_path / "acquisitions.nc",
        ["2024-02-29T23:59:59", "2024-01-02T12:00:00", "2024-02-01T00:00:00", "2024-01-02T08:00:00"],
        B04=[[[3, 3], [3, 3]], [[10, np.nan], [np.inf, np.nan]], [[2, 2], [2, 2]], [[1, 2], [3, 4]]],
        B08=[[[6, 6], [6, 6]], [[20, np.nan], [np.nan, np.nan]], [[4, 4], [4, 4]], [[3, 6], [9, 12]]],
    )
    result, report = extractor._calculate_local_statistics(path, extractor.parcels, 1)
    january = result.loc[(result.parcel_id == "whole") & (result.period_start == "2024-01-01")].iloc[0]
    samples = np.array([1, 2, 3, 4, 10], dtype="float64")
    expected = {
        "mean": 4.0,
        "median": 3.0,
        "sd": np.std(samples, ddof=1),
        "min": 1.0,
        "max": 10.0,
        "range": 9.0,
        "p10": 1.4,
        "p25": 2.0,
        "p75": 4.0,
        "p90": 7.6,
    }
    for statistic, value in expected.items():
        expect_equal(january[f"B04_{statistic}"], pytest.approx(value))
    # Four first-acquisition indices of 0.5 plus one second-acquisition index of 1/3.
    expect_equal(january.NDVI_mean, pytest.approx((4 * 0.5 + 1 / 3) / 5))
    expect_equal(january.NDVI_median, pytest.approx(0.5))
    expect_equal(january.intersected_pixel_count, 4)
    expect_equal(january.temporal_filled_pixel_count, 0)
    expect_equal(january.spatial_filled_pixel_count, 0)
    right = result.loc[(result.parcel_id == "right") & (result.period_start == "2024-01-01")].iloc[0]
    expect_equal(right.B04_mean, pytest.approx(3.0))
    february = result.loc[(result.parcel_id == "whole") & (result.period_start == "2024-02-01")].iloc[0]
    expect_equal(february.B04_mean, pytest.approx(2.5))
    expect_equal(february.period_end, "2024-03-01")
    expect_true(result.loc[result.period_start == "2024-03-01", "B04_mean"].isna().all())
    expect_true(result.loc[result.parcel_id == "outside", "NDVI_median"].isna().all())
    expect_equal(report.period_start.nunique(), 4)

    # Resume from the final checkpoint, then exercise the public run and dated output.
    monkeypatch.setattr(extractor, "_calculate_iqr_bounds", lambda *_: pytest.fail("Checkpoint should be reused"))
    resumed, _ = extractor._calculate_local_statistics(path, extractor.parcels, 1)
    pd.testing.assert_frame_equal(result, resumed)
    monkeypatch.setattr(extractor, "_run_openeo_jobs", lambda _: [(1, extractor.parcels, path)])
    final = extractor.run().set_index("parcel_id")
    expect_equal(final.loc["whole", "B04_mean__20240101"], pytest.approx(4.0))
    expect_equal(final.loc["whole", "NDVI_median__20240101"], pytest.approx(0.5))
    expect_true(pd.isna(final.loc["whole", "B04_mean__20240301"]))
    saved = gpd.read_parquet(tmp_path / extractor.PARCEL_OUTPUT_FILE_NAME).set_index("parcel_id")
    expect_equal(saved.loc["whole", "B04_mean__20240101"], pytest.approx(4.0))


def test_custom_periods_use_half_open_intervals_and_keep_all_null_periods(tmp_path):
    extractor = _extractor(
        tmp_path,
        start_date="2024-01-05",
        end_date="2024-02-06",
        temporal_period="15D",
        sentinel2_bands=["B04"],
        sentinel2_indices=[],
    )
    path = _write_cube(
        tmp_path / "custom.nc",
        ["2024-01-05", "2024-01-19T23:59:59", "2024-01-20", "2024-02-06T12:00:00"],
        B04=[np.full((2, 2), value) for value in [1, 3, 10, np.nan]],
    )
    result, _ = extractor._calculate_local_statistics(path, extractor.parcels, 1)
    whole = result.loc[result.parcel_id == "whole"]
    expect_equal(whole.period_start.tolist(), ["2024-01-05", "2024-01-20", "2024-02-04"])
    expect_equal(whole.period_end.tolist(), ["2024-01-20", "2024-02-04", "2024-02-07"])
    np.testing.assert_allclose(whole.B04_mean, [2, 10, np.nan], equal_nan=True)


def test_disjoint_sensor_timestamps_do_not_create_cross_date_indices(tmp_path):
    extractor = _extractor(tmp_path, sentinel1_bands=["VV", "VH"], sentinel1_indices=["R", "RVI"])
    missing = np.full((2, 2), np.nan)
    path = _write_cube(
        tmp_path / "sensors.nc",
        ["2024-01-02", "2024-01-03", "2024-01-04"],
        B04=[np.ones((2, 2)), missing, missing],
        B08=[np.full((2, 2), 3), missing, missing],
        VV=[missing, [[2, 8], [np.nan, np.nan]], [[12, np.nan], [np.nan, np.nan]]],
        VH=[missing, [[1, 2], [np.nan, np.nan]], [[3, np.nan], [np.nan, np.nan]]],
    )
    result, _ = extractor._calculate_local_statistics(path, extractor.parcels, 1)
    january = result.loc[(result.parcel_id == "whole") & (result.period_start == "2024-01-01")].iloc[0]
    expect_equal(january.NDVI_mean, pytest.approx(0.5))
    expect_equal(january.VV_mean, pytest.approx(22 / 3))
    expect_equal(january.R_mean, pytest.approx(10 / 3))
    expect_equal(january.RVI_mean, pytest.approx((4 / 3 + 0.8 + 0.8) / 3))


def test_out_of_range_acquisition_is_rejected(tmp_path):
    extractor = _extractor(tmp_path, sentinel2_bands=["B04"], sentinel2_indices=[])
    path = _write_cube(tmp_path / "outside.nc", ["2024-04-01"], B04=[np.ones((2, 2))])
    with pytest.raises(ValueError, match="outside the configured temporal intervals"):
        extractor._calculate_local_statistics(path, extractor.parcels, 1)


def test_existing_composites_still_reduce_only_spatial_pixels(tmp_path):
    extractor = _extractor(tmp_path, temporal_reducer="median", sentinel2_bands=["B04"], sentinel2_indices=[])
    path = _write_cube(
        tmp_path / "composites.nc",
        ["2024-01-01", "2024-02-01", "2024-03-01"],
        B04=[[[1, 2], [3, 4]], [[10, 20], [30, 40]], [[5, 5], [5, 5]]],
    )
    result, _ = extractor._calculate_local_statistics(path, extractor.parcels, 1)
    np.testing.assert_allclose(result.loc[result.parcel_id == "whole", "B04_mean"], [2.5, 25, 5])


@pytest.mark.parametrize("fill_mode", ["past_only", "bidirectional"])
@pytest.mark.parametrize("remove_outliers", [False, True])
@pytest.mark.parametrize("temporal_windows", [None, (3,), (3, 5)])
def test_streamed_cleaning_matches_full_cube_with_filling(tmp_path, monkeypatch, fill_mode, remove_outliers, temporal_windows):
    from data_preparation.parcel_stats.core import streamed_raster

    extractor = _extractor(
        tmp_path,
        fill_nulls=True,
        temporal_fill_mode=fill_mode,
        remove_outliers=remove_outliers,
        iqr_min_valid_pixels=3,
        minimum_observed_fraction_for_fill=0.5,
        spatial_fill_window_sizes=(3, 5, 7),
        temporal_fill_window_sizes=temporal_windows,
    )
    times = ["2024-01-01", "2024-01-10", "2024-02-01", "2024-02-15"]
    red = np.array(
        [[[1, np.nan], [2, 1000]], [[np.nan, 3], [2, np.nan]], [[8, np.nan], [np.nan, np.nan]], [[2, 4], [np.nan, 8]]],
        dtype="float32",
    )
    nir = np.array(
        [[[3, 6], [np.nan, 12]], [[np.nan, 6], [4, np.nan]], [[np.nan, np.nan], [np.nan, np.nan]], [[4, np.nan], [8, 16]]],
        dtype="float32",
    )
    path = _write_cube(tmp_path / "filled.nc", times, B04=red, B08=nir)
    values = np.stack([red, nir], axis=1)
    mask = np.ones((2, 2), dtype=bool)
    periods = [pd.Timestamp(time, tz="UTC").isoformat() for time in times]
    from rasterio.transform import from_origin

    transform = from_origin(0, 20, 10, 10)
    cleaned, observed, expected_report = extractor._clean_monthly_pixels(
        values, mask, 1, periods, transform, "EPSG:3857", variable_names=["B04", "B08"]
    )
    temporal = extractor._fill_temporal_cube_only(observed, mask)
    expected_cleaned = extractor._build_local_output_cube(cleaned, ["B04", "B08"])
    expected_observed = np.isfinite(extractor._build_local_output_cube(observed, ["B04", "B08"]))
    expected_temporal = ~expected_observed & np.isfinite(extractor._build_local_output_cube(temporal, ["B04", "B08"]))
    # Force temporal strips to one row, exercising cross-strip and cross-period fills.
    monkeypatch.setattr(streamed_raster, "TEMPORAL_BLOCK_BYTES", 1)
    result, report = extractor._calculate_local_statistics(path, extractor.parcels, 1)
    with xr.open_dataset(extractor._final_checkpoint_path(path)) as final:
        np.testing.assert_allclose(final.cleaned.values, expected_cleaned, equal_nan=True)
        np.testing.assert_array_equal(final.observed_mask.values, expected_observed)
        np.testing.assert_array_equal(final.temporal_filled_mask.values, expected_temporal)
    pd.testing.assert_frame_equal(report, expected_report)
    whole = result.loc[result.parcel_id == "whole"].iloc[0]
    metrics = extractor._summarize_parcel_filling(
        np.where(expected_observed, expected_cleaned, np.nan), expected_temporal, expected_cleaned, 4
    )
    for name, value in metrics.items():
        expect_equal(whole[name], pytest.approx(value))


def test_streamed_path_never_selects_the_complete_raw_cube(tmp_path, monkeypatch):
    extractor = _extractor(tmp_path)
    path = _write_cube(
        tmp_path / "bounded.nc", ["2024-01-01", "2024-01-10", "2024-02-01"], B04=np.ones((3, 2, 2)), B08=np.full((3, 2, 2), 3)
    )
    original_select = extractor._select_monthly_variables
    selected_shapes = []

    def bounded_select(dataset, time_dimension, *args, **kwargs):
        expect_true(dataset.sizes[time_dimension] <= 1)
        if dataset.sizes[time_dimension]:
            expect_equal(len(kwargs["expected_variables"]), 1)
        selected_shapes.append(dataset.sizes[time_dimension])
        return original_select(dataset, time_dimension, *args, **kwargs)

    monkeypatch.setattr(extractor, "_select_monthly_variables", bounded_select)
    monkeypatch.setattr(extractor, "_build_local_output_cube", lambda *_: pytest.fail("Full output cube allocated"))
    monkeypatch.setattr(extractor, "_load_final_checkpoint", lambda *_: pytest.fail("Full checkpoint loaded"))
    extractor._calculate_local_statistics(path, extractor.parcels, 1)
    expect_equal(selected_shapes, [0, 1, 1, 1, 1, 1, 1])
    selected_shapes.clear()
    extractor._calculate_local_statistics(path, extractor.parcels, 1)
    expect_equal(selected_shapes, [0])  # A resumed run only needs raw metadata.


def test_streamed_physical_checkpoint_survives_index_failure(tmp_path, monkeypatch):
    import weakref

    extractor = _extractor(tmp_path)
    path = _write_cube(tmp_path / "resume.nc", ["2024-01-01"], B04=np.ones((1, 2, 2)), B08=np.full((1, 2, 2), 3))
    build_indices = extractor._stream_output_indices
    references = []

    def fail_indices(*_args):
        retained = np.ones((2, 2))
        references.append(weakref.ref(retained))
        raise RuntimeError("interrupted index stage")

    monkeypatch.setattr(extractor, "_stream_output_indices", fail_indices)
    with pytest.raises(RuntimeError, match="interrupted index stage") as error:
        extractor._calculate_local_statistics(path, extractor.parcels, 1)
    expect_is_not_none(error.value)  # Holding the traceback must not hold raster arrays.
    expect_is_none(references[0]())
    expect_false(extractor._final_checkpoint_path(path).exists())
    monkeypatch.setattr(extractor, "_stream_output_indices", build_indices)
    monkeypatch.setattr(extractor, "_stream_physical_bands", lambda *_: pytest.fail("Completed cleaning must be reused"))
    result, _ = extractor._calculate_local_statistics(path, extractor.parcels, 1)
    expect_equal(result.loc[result.parcel_id == "whole", "NDVI_mean"].iloc[0], pytest.approx(0.5))


def test_local_user_threads_serialize_raster_work_and_release_arrays(tmp_path, monkeypatch):
    import weakref
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    extractor = _extractor(tmp_path)
    first_entered, second_started, release_first = Event(), Event(), Event()
    references = []
    entered = []

    def work(_path, _parcels, number):
        raster = np.ones((4, 4))
        references.append(weakref.ref(raster))
        entered.append(number)
        if number == 1:
            first_entered.set()
            expect_true(release_first.wait(5))
        return pd.DataFrame({"value": [number]}), pd.DataFrame()

    def second():
        second_started.set()
        return extractor._calculate_local_statistics(None, None, 2)

    monkeypatch.setattr(extractor, "_calculate_streamed_statistics", work)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first_result = executor.submit(extractor._calculate_local_statistics, None, None, 1)
        try:
            expect_true(first_entered.wait(5))
            second_result = executor.submit(second)
            expect_true(second_started.wait(5))
            expect_equal(entered, [1])
        finally:
            release_first.set()
        first_result.result()
        second_result.result()
    expect_equal(entered, [1, 2])
    expect_true(all(reference() is None for reference in references))
