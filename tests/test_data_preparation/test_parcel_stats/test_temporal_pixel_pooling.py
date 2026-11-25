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
@pytest.mark.parametrize("block_rows", [1, 256])
def test_streamed_cleaning_matches_full_cube_with_filling(tmp_path, fill_mode, remove_outliers, temporal_windows, block_rows):
    """Small and oversized strips preserve values, provenance, audits and parcel statistics."""
    extractor = _extractor(
        tmp_path,
        fill_nulls=True,
        temporal_fill_mode=fill_mode,
        remove_outliers=remove_outliers,
        iqr_min_valid_pixels=3,
        minimum_observed_fraction_for_fill=0.5,
        spatial_fill_window_sizes=(3, 5, 7),
        temporal_fill_window_sizes=temporal_windows,
        temporal_fill_block_rows=block_rows,
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


@pytest.mark.parametrize("block_rows", [0, -1, True, 1.5, "256"])
def test_temporal_fill_block_rows_rejects_invalid_values(tmp_path, block_rows):
    """Reject invalid strip sizes before creating output directories."""
    output = tmp_path / "invalid"
    with pytest.raises((TypeError, ValueError), match="temporal_fill_block_rows"):
        _extractor(output, temporal_fill_block_rows=block_rows)
    expect_false(output.exists())


@pytest.mark.parametrize("cloud_cover", [-1, 101, True, 50.5, "70"])
def test_sentinel2_max_scene_cloud_cover_rejects_invalid_values(tmp_path, cloud_cover):
    """Reject invalid scene cloud-cover limits before creating output directories."""
    output = tmp_path / "invalid"
    with pytest.raises((TypeError, ValueError), match="sentinel2_max_scene_cloud_cover"):
        _extractor(output, sentinel2_max_scene_cloud_cover=cloud_cover)
    expect_false(output.exists())


def test_sentinel2_max_scene_cloud_cover_changes_run_signature(tmp_path):
    """Scene filtering changes downloaded imagery, so it must invalidate cached results."""
    default = _extractor(tmp_path / "default")
    expect_equal(default.sentinel2_max_scene_cloud_cover, OpenEOZonalStats.SENTINEL2_MAX_SCENE_CLOUD_COVER)
    strict = _extractor(tmp_path / "strict", sentinel2_max_scene_cloud_cover=30)
    expect_equal(strict.sentinel2_max_scene_cloud_cover, 30)
    expect_false(default._run_signature() == strict._run_signature())


def test_temporal_fill_block_rows_preserves_cache_signatures(tmp_path):
    """Execution-only tuning can reuse previously calculated raster and batch caches."""
    extractor = _extractor(tmp_path, temporal_fill_block_rows=None)
    expect_is_none(extractor.temporal_fill_block_rows)
    signatures = (extractor._run_signature(), extractor._cleaning_checkpoint_signature())
    extractor.temporal_fill_block_rows = 256
    expect_equal((extractor._run_signature(), extractor._cleaning_checkpoint_signature()), signatures)


@pytest.mark.parametrize("block_rows,expected_heights", [(3, [3, 2]), (256, [5]), (None, [1, 1, 1, 1, 1])])
def test_temporal_fill_reads_configured_strips(tmp_path, monkeypatch, block_rows, expected_heights):
    """Honor explicit rows, clamp the last strip and retain automatic sizing."""
    from data_preparation.parcel_stats.core import streamed_raster

    extractor = _extractor(tmp_path, fill_nulls=True, temporal_fill_mode="bidirectional", temporal_fill_block_rows=block_rows)
    values = np.arange(30, dtype="float32").reshape(3, 5, 2)
    values[1, :, 0] = np.nan
    expected = extractor._fill_temporal_neighbors(values)
    heights = []
    fill = extractor._fill_temporal_neighbors

    def record_strip(strip):
        """Capture real input sizes while retaining the numerical fill operation."""
        heights.append(strip.shape[1])
        return fill(strip)

    monkeypatch.setattr(extractor, "_fill_temporal_neighbors", record_strip)
    monkeypatch.setattr(streamed_raster, "TEMPORAL_BLOCK_BYTES", 1)
    with extractor._new_streamed_checkpoint(tmp_path / "strips.nc", ["a", "b", "c"], ["B04"], (5, 2)) as checkpoint:
        checkpoint.variables["cleaned"][:, 0] = values
        counts = (~np.isfinite(values)).sum(axis=(1, 2))[:, None]
        extractor._stream_temporal_fill(checkpoint, np.ones((3, 1), dtype=bool), counts)
        np.testing.assert_allclose(checkpoint.variables["cleaned"][:, 0], expected)
        np.testing.assert_array_equal(checkpoint.variables["temporal_filled_mask"][:, 0], ~np.isfinite(values))
        np.testing.assert_array_equal(counts, 0)
    expect_equal(heights, expected_heights)


def test_streamed_temporal_fill_skips_dates_without_any_observation(tmp_path):
    """A band's empty dates on a merged sensor axis neither block nor receive temporal fills."""
    extractor = _extractor(tmp_path, fill_nulls=True, temporal_fill_mode="bidirectional", temporal_fill_window_sizes=(3,))
    values = np.full((5, 1, 2), np.nan, dtype="float32")
    values[0] = [[0.2, 0.2]]
    values[2] = [[np.nan, 0.9]]
    values[4] = [[0.6, 0.6]]
    observed_counts = np.isfinite(values).sum(axis=(1, 2))[:, None]
    allowed = observed_counts / 2 >= 0.5
    with extractor._new_streamed_checkpoint(tmp_path / "merged.nc", list("abcde"), ["B04"], (1, 2)) as checkpoint:
        checkpoint.variables["cleaned"][:, 0] = values
        counts = (~np.isfinite(values)).sum(axis=(1, 2))[:, None]
        extractor._stream_temporal_fill(checkpoint, allowed, counts, observed_counts > 0)
        stored = np.asarray(checkpoint.variables["cleaned"][:, 0])
        np.testing.assert_allclose(stored[2, 0], [0.4, 0.9], rtol=1e-6)
        expect_true(np.isnan(stored[[1, 3]]).all())
        np.testing.assert_array_equal(checkpoint.variables["temporal_filled_mask"][:, 0, 0, 0], [0, 0, 1, 0, 0])
        np.testing.assert_array_equal(counts[:, 0], [0, 2, 0, 2, 0])


def test_streamed_temporal_fill_uses_only_adjacent_dates_of_the_same_band(tmp_path):
    """Window 3: mean of previous/next, the single available side, or null; never another band."""
    extractor = _extractor(tmp_path, fill_nulls=True, temporal_fill_mode="bidirectional", temporal_fill_window_sizes=(3,))
    nan = np.nan
    # Date 2 pixel columns: both neighbours, previous only, next only, neither, observed.
    b04 = np.array(
        [[0.1, 0.1, nan, nan, 0.1], [nan, nan, nan, nan, 0.2], [0.3, nan, 0.3, nan, 0.3], [0.4] * 5, [0.5] * 5], dtype="float32"
    )
    b08 = np.full((5, 5), 0.9, dtype="float32")
    values = np.stack([b04, b08], axis=1).reshape(5, 2, 1, 5)
    observed_counts = np.isfinite(values).sum(axis=(2, 3))
    with extractor._new_streamed_checkpoint(tmp_path / "bands.nc", list("abcde"), ["B04", "B08"], (1, 5)) as checkpoint:
        checkpoint.variables["cleaned"][:] = values
        counts = (~np.isfinite(values)).sum(axis=(2, 3))
        extractor._stream_temporal_fill(checkpoint, np.ones((5, 2), dtype=bool), counts, observed_counts > 0)
        stored = np.asarray(checkpoint.variables["cleaned"][:])
        np.testing.assert_allclose(stored[1, 0, 0], [0.2, 0.1, 0.3, nan, 0.2], rtol=1e-6, equal_nan=True)
        # Fills read original observations only: date 3 takes date 4, never a value just filled into date 2.
        np.testing.assert_allclose(stored[2, 0, 0], [0.3, 0.4, 0.3, 0.4, 0.3], rtol=1e-6)
        np.testing.assert_array_equal(stored[:, 1, 0], b08)
        np.testing.assert_array_equal(counts[:, 0], [2, 1, 0, 0, 0])


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


def _partially_observed_cube(tmp_path):
    """Write acquisitions where the right-hand column is unobserved on 5 January and 10 February."""
    nan = np.nan
    return _write_cube(
        tmp_path / "partial.nc",
        ["2024-01-02", "2024-01-05", "2024-01-20", "2024-02-10"],
        B04=[[[0, 10], [0, 10]], [[1, nan], [1, nan]], [[2, 4], [2, 4]], [[3, nan], [3, nan]]],
    )


@pytest.mark.parametrize("require", [False, True])
def test_pooled_statistics_can_exclude_acquisitions_without_parcel_observations(tmp_path, require):
    """Filled-only acquisitions leave the pool, while partially observed ones keep their fills."""
    extractor = _extractor(
        tmp_path,
        sentinel2_bands=["B04"],
        sentinel2_indices=[],
        fill_nulls=True,
        temporal_fill_mode="past_only",
        require_parcel_observation=require,
    )
    result, _ = extractor._calculate_local_statistics(_partially_observed_cube(tmp_path), extractor.parcels, 1)
    stats = result.set_index(["parcel_id", "period_start"])["B04_mean"]
    # Without the check, 5 January's right column carries 10 forward from 2 January and 10 February carries 4.
    expect_equal(stats["right", "2024-01-01"], pytest.approx(7.0 if require else 8.0))
    if require:
        expect_true(np.isnan(stats["right", "2024-02-01"]))
    else:
        expect_equal(stats["right", "2024-02-01"], pytest.approx(4.0))
    # The whole parcel was observed on 10 February, so its temporally filled right column still counts.
    expect_equal(stats["whole", "2024-02-01"], pytest.approx(3.5))
    # Fill provenance continues to describe the cube, independent of the check.
    right = result.loc[result.parcel_id == "right"].iloc[0]
    expect_equal(right.temporal_filled_pixel_count, 4)


@pytest.mark.parametrize("require", [False, True])
def test_composite_statistics_can_exclude_periods_without_parcel_observations(tmp_path, require):
    """A composite whose parcel pixels were all spatially filled becomes null when required."""
    extractor = _extractor(
        tmp_path,
        temporal_reducer="median",
        sentinel2_bands=["B04"],
        sentinel2_indices=[],
        fill_nulls=True,
        temporal_fill_mode="past_only",
        require_parcel_observation=require,
    )
    path = _write_cube(
        tmp_path / "composites.nc",
        ["2024-01-01", "2024-02-01", "2024-03-01"],
        B04=[[[1, np.nan], [1, np.nan]], [[2, 6], [2, 6]], [[5, 5], [5, 5]]],
    )
    result, _ = extractor._calculate_local_statistics(path, extractor.parcels, 1)
    right = result.loc[result.parcel_id == "right"].set_index("period_start")["B04_mean"]
    # January's right column has no earlier date, so only spatial filling from the left column reaches it.
    np.testing.assert_allclose(right, [np.nan if require else 1.0, 6.0, 5.0], equal_nan=True)


def test_parcel_observation_check_changes_only_the_statistics_signature(tmp_path):
    """Enabling the check recalculates statistics but reuses downloads and cleaned rasters."""
    extractor = _extractor(tmp_path)
    expect_false(extractor.require_parcel_observation)
    signatures = (extractor._run_signature(), extractor._cleaning_checkpoint_signature())
    statistics_signature = extractor._statistics_signature()
    extractor.require_parcel_observation = True
    expect_equal((extractor._run_signature(), extractor._cleaning_checkpoint_signature()), signatures)
    expect_true(extractor._statistics_signature() != statistics_signature)


def test_parcel_observation_check_rejects_non_boolean_values(tmp_path):
    """Reject truthy strings before creating output directories."""
    output = tmp_path / "invalid"
    with pytest.raises(TypeError, match="require_parcel_observation"):
        _extractor(output, require_parcel_observation="yes")
    expect_false(output.exists())


def _gridded_cube(tmp_path, size=12):
    """Write a reproducible gap-filled cube on a 10 m grid with several dates per month."""
    generator = np.random.default_rng(7)
    times = ["2024-01-03", "2024-01-15", "2024-01-28", "2024-02-09", "2024-03-02", "2024-03-20"]
    red = generator.uniform(1, 50, (len(times), size, size)).astype("float32")
    nir = generator.uniform(60, 120, (len(times), size, size)).astype("float32")
    # Remove a third of the observations so temporal and spatial fills both contribute provenance.
    red[generator.random(red.shape) < 0.33] = np.nan
    nir[generator.random(nir.shape) < 0.33] = np.nan
    centres = np.arange(size) * 10.0 + 5.0
    dataset = xr.Dataset(
        {"B04": (("t", "y", "x"), red), "B08": (("t", "y", "x"), nir)},
        coords={"t": pd.to_datetime(times), "x": centres, "y": centres[::-1]},
    ).rio.write_crs(3857)
    path = tmp_path / "gridded.nc"
    dataset.to_netcdf(path)
    return path


def _gridded_parcels():
    """Parcels deliberately listed out of spatial order, including tiny, edge and outside shapes."""
    return gpd.GeoDataFrame(
        {"parcel_id": ["south_east", "north_west", "centre", "sliver", "edge", "outside"]},
        geometry=[
            box(70.2, 0.5, 119.0, 40.0),
            box(0.5, 80.0, 45.0, 119.5),
            box(30.0, 30.0, 90.0, 90.0),
            box(55.1, 55.1, 55.9, 64.9),
            box(-20.0, 50.0, 12.0, 70.0),
            box(500.0, 500.0, 520.0, 520.0),
        ],
        crs=3857,
    )


def _reference_parcel_statistics(extractor, final, parcels, context):
    """Recompute statistics with one masked read per parcel, period and output (the previous algorithm)."""
    expected = {}
    for parcel_id, geometry in zip(parcels.parcel_id, context["projected"].geometry):
        counts = {"observed": 0, "temporal": 0, "spatial": 0}
        for period, indices in zip(context["result_periods"], context["period_groups"]):
            interval = slice(int(indices[0]), int(indices[-1]) + 1) if len(indices) else slice(0, 0)
            for variable_index, variable in enumerate(context["variables"]):
                arrays = {}
                for name in ("cleaned", "observed_mask", "temporal_filled_mask"):
                    values = final[name].values[interval, variable_index : variable_index + 1]
                    window, mask, _ = extractor._mask_parcel_window(values, geometry, context["transform"])
                    arrays[name] = window[:, :, mask]
                pixels = arrays["cleaned"].astype("float32")
                observed = arrays["observed_mask"].astype(bool)
                temporal = arrays["temporal_filled_mask"].astype(bool)
                counts["observed"] += int(observed.sum())
                counts["temporal"] += int(temporal.sum())
                counts["spatial"] += int((~observed & ~temporal & np.isfinite(pixels)).sum())
                if extractor.require_parcel_observation:
                    pixels[~observed.any(axis=(1, 2))] = np.nan
                for statistic, value in extractor._reduce_local_pixels(pixels.reshape(1, -1)).items():
                    expected[(parcel_id, period, f"{variable}_{statistic}")] = float(value[0])
        expected[(parcel_id, "counts")] = counts
    return expected


def _block_read_statistics(tmp_path, monkeypatch, require):
    """Run streamed statistics on the gridded cube, recording each block read and the reference statistics."""
    parcels = _gridded_parcels()
    extractor = _extractor(
        tmp_path,
        parcels=parcels,
        fill_nulls=True,
        temporal_fill_window_sizes=(3,),
        spatial_fill_window_sizes=(3, 5),
        minimum_observed_fraction_for_fill=0.2,
        require_parcel_observation=require,
    )
    path = _gridded_cube(tmp_path)
    reads = []
    original_read = extractor._read_parcel_block
    monkeypatch.setattr(extractor, "_read_parcel_block", lambda *args: (reads.append(args[1:]), original_read(*args))[1])
    result, _ = extractor._calculate_local_statistics(path, extractor.parcels, 1)

    with xr.open_dataset(path, decode_coords="all") as dataset:
        context = extractor._build_stream_statistics_context(dataset, extractor.parcels)
    with xr.open_dataset(extractor._final_checkpoint_path(path)) as final:
        expected = _reference_parcel_statistics(extractor, final, extractor.parcels, context)
    return parcels, extractor, context, result, expected, reads


def _assert_matches_reference(parcels, extractor, context, result, expected):
    """Group block reads must reproduce the per-parcel algorithm exactly, in input parcel order."""
    expect_equal(result.parcel_id.drop_duplicates().to_list(), parcels.parcel_id.to_list())
    statistic_columns = [column for column in result.columns if column.split("_")[0] in context["variables"]]
    expect_equal(len(statistic_columns), len(context["variables"]) * len(extractor.spatial_statistics))
    for row in result.itertuples(index=False):
        for column in statistic_columns:
            # Per-parcel arithmetic is unchanged, so values must be bit-identical, not merely close.
            np.testing.assert_array_equal(getattr(row, column), expected[(row.parcel_id, row.period_start, column)])
        counts = expected[(row.parcel_id, "counts")]
        expect_equal(row.observed_count, counts["observed"])
        expect_equal(row.temporal_filled_pixel_count, counts["temporal"])
        expect_equal(row.spatial_filled_pixel_count, counts["spatial"])
    whole_counts = expected[("centre", "counts")]
    expect_true(whole_counts["temporal"] > 0 and whole_counts["spatial"] > 0)


@pytest.mark.parametrize("require", [False, True])
def test_group_block_reads_match_per_parcel_statistics(tmp_path, monkeypatch, require):
    """All inside parcels share one chunk, so each output is read once for them and once for the outside parcel."""
    parcels, extractor, context, result, expected, reads = _block_read_statistics(tmp_path, monkeypatch, require)
    _assert_matches_reference(parcels, extractor, context, result, expected)
    # Two groups (the shared chunk and the outside placeholder), three arrays, every output: independent of parcel count.
    expect_equal(len(reads), 3 * len(context["variables"]) * 2)
    # Every read covers all stored dates when the block fits the payload target.
    expect_true(all(read[2] == slice(0, 6) for read in reads))


@pytest.mark.parametrize("require", [False, True])
def test_group_block_reads_cross_checkpoint_chunks(tmp_path, monkeypatch, require):
    """Small checkpoint chunks give several groups whose union windows cross chunk boundaries."""
    from data_preparation.parcel_stats.core import streamed_raster

    monkeypatch.setattr(streamed_raster, "CHECKPOINT_CHUNK_PIXELS", 4)
    parcels, extractor, context, result, expected, reads = _block_read_statistics(tmp_path, monkeypatch, require)
    _assert_matches_reference(parcels, extractor, context, result, expected)
    # Upper-left chunks: north_west and centre (0, 0); edge (1, 0); sliver (1, 1); south_east (2, 1); outside.
    expect_equal(len(reads), 3 * len(context["variables"]) * 5)


def test_oversized_groups_split_and_read_one_period_at_a_time(tmp_path, monkeypatch):
    """Above the payload target, groups split into single parcels and each reporting period is read separately."""
    from data_preparation.parcel_stats.core import streamed_raster

    monkeypatch.setattr(streamed_raster, "PARCEL_BLOCK_BYTES", 1)
    parcels, extractor, context, result, expected, reads = _block_read_statistics(tmp_path, monkeypatch, True)
    _assert_matches_reference(parcels, extractor, context, result, expected)
    # One read per parcel, reporting period, array and output.
    expect_equal(len(reads), 3 * len(context["variables"]) * len(parcels) * len(context["result_periods"]))
    expect_true(all(read[2] != slice(0, 6) for read in reads))


def test_parcel_read_groups_key_by_upper_left_chunk_and_keep_outside_parcels_together():
    """Parcels are grouped by the chunk of their window's upper-left pixel, in chunk-row-major order."""
    from rasterio.transform import from_origin

    transform = from_origin(0, 10_000, 10, 10)
    # Upper-left pixels: (row 300, col 10), (row 5, col 300), (row 5, col 5), (row 200, col 100), outside, (row 6, col 6).
    geometries = [
        box(100, 10_000 - 3_010, 120, 10_000 - 3_000),
        box(3_000, 10_000 - 60, 3_020, 10_000 - 50),
        box(50, 10_000 - 60, 70, 10_000 - 50),
        box(1_000, 10_000 - 2_010, 1_020, 10_000 - 2_000),
        box(-500, 20_000, -400, 20_100),
        box(60, 10_000 - 70, 80, 10_000 - 60),
    ]
    extractor = OpenEOZonalStats.__new__(OpenEOZonalStats)
    groups = extractor._parcel_read_groups(geometries, transform, (1_000, 1_000), [np.arange(3)])
    expect_equal(groups, [[4], [2, 3, 5], [1], [0]])
