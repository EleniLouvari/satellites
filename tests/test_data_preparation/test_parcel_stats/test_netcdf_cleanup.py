"""Cleanup must preserve durable results and resume without remote downloads."""

import warnings
import weakref
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import box

from data_preparation.parcel_stats import batch_scheduler as scheduler
from data_preparation.parcel_stats.core import base
from data_preparation.parcel_stats.job_manager import OpenEOJobManagerZonalStats
from tests.utils import expect_equal, expect_false, expect_is_none, expect_is_not_none, expect_true


def _config(tmp_path, **overrides):
    """Build a minimal offline extraction configuration with optional overrides."""
    return dict(
        output_dir=tmp_path, run_identifier="cleanup", parcel_id_field="parcel_id",
        start_date="2024-01-01", end_date="2024-01-31", working_epsg=3857,
        sentinel2_bands=["B02"], sentinel2_indices=[], sentinel1_bands=[],
        spatial_statistics=["mean"], tile_size_metres=1000, tile_buffer_metres=0,
        fit_tiles_to_parcels=True, batch_workers=1, **overrides,
    )


def _parcels():
    """Return one projected parcel for cleanup and resume tests."""
    return gpd.GeoDataFrame({"parcel_id": ["a"]}, geometry=[box(0, 0, 100, 100)], crs=3857)


def _extractor(tmp_path, **overrides):
    """Create a configured extractor rooted in the temporary test directory."""
    return OpenEOJobManagerZonalStats(parcels=_parcels(), **_config(tmp_path, **overrides))


def _batch(extractor):
    """Create dummy raster checkpoints and return their raw path and file list."""
    directory = extractor.output_dir / "monthly_cubes" / extractor._run_signature()
    directory.mkdir(parents=True, exist_ok=True)
    path = Path(extractor._build_tile_plan(directory)[0]["target_path"])
    cleaned, _ = extractor._cleaning_checkpoint_paths(path)
    final = extractor._final_checkpoint_path(path)
    paths = [p for raster in (path, cleaned, final) for p in (raster, raster.with_suffix(".partial.nc"))]
    for raster in paths:
        raster.write_bytes(b"test raster")
    return path, paths


def _statistics(*args):
    """Return small deterministic parcel statistics and cleaning-report tables."""
    return pd.DataFrame({"parcel_id": ["a"], "period_start": ["2024-01-01"], "B02_mean": [1.0]}), pd.DataFrame({"batch_number": [1]})


@pytest.mark.parametrize("cleanup", [False, True])
def test_cleanup_and_resume_preserve_results_and_other_batches(tmp_path, monkeypatch, cleanup):
    """Verify optional cleanup preserves saved results and unrelated raster files."""
    extractor = _extractor(tmp_path, remove_nc_after_completion=cleanup, keep_cleaned_checkpoint=True)
    raw, paths = _batch(extractor)
    unrelated = raw.with_name("batch_00002_monthly.nc")
    unrelated.touch()
    monkeypatch.setattr(extractor, "_calculate_local_statistics", _statistics)
    _, result, report = extractor._process_zonal_stats_batch(raw, extractor.parcels, 1)
    expect_true(all(path.exists() != cleanup for path in paths))
    expect_true(unrelated.exists())
    expect_true(extractor._has_batch_statistics(raw))

    def no_recalculation(*args):
        """Fail if resume attempts to recalculate a completed batch."""
        pytest.fail("A completed batch must resume from its saved statistics")

    monkeypatch.setattr(extractor, "_calculate_local_statistics", no_recalculation)
    _, resumed, resumed_report = extractor._process_zonal_stats_batch(raw, extractor.parcels, 1)
    pd.testing.assert_frame_equal(result, resumed)
    pd.testing.assert_frame_equal(report, resumed_report)
    expect_equal(extractor._run_openeo_jobs(raw.parent)[0][2], raw)


@pytest.mark.parametrize("failure", ["calculation", "report", "result", "commit"])
def test_failure_keeps_rasters_and_never_marks_batch_complete(tmp_path, monkeypatch, failure):
    """Verify calculation and persistence failures leave raster recovery inputs intact."""
    extractor = _extractor(tmp_path, remove_nc_after_completion=True)
    raw, paths = _batch(extractor)
    monkeypatch.setattr(extractor, "_calculate_local_statistics", _statistics)

    def fail(*args, **kwargs):
        """Simulate a disk-full error during calculation or checkpoint persistence."""
        raise OSError("disk full")

    if failure == "calculation":
        monkeypatch.setattr(extractor, "_calculate_local_statistics", fail)
    elif failure == "commit":
        monkeypatch.setattr(Path, "replace", fail)
    else:
        original = pd.DataFrame.to_parquet

        def write(frame, path, **kwargs):
            """Inject a write failure for the selected table and persist the other normally."""
            if ("_report" in path.name) == (failure == "report"):
                fail()
            original(frame, path, **kwargs)

        monkeypatch.setattr(pd.DataFrame, "to_parquet", write)
    with pytest.raises(OSError, match="disk full"):
        extractor._process_zonal_stats_batch(raw, extractor.parcels, 1)
    expect_true(all(path.exists() for path in paths))
    expect_false(extractor._has_batch_statistics(raw))


def test_cleanup_flags_do_not_invalidate_cache_but_processing_changes_do(tmp_path):
    """Verify cache identity depends on processing options rather than storage flags."""
    extractor = _extractor(tmp_path)
    signature = extractor._statistics_signature()
    extractor.remove_nc_after_completion = True
    extractor.resume_completed_partitions = True
    expect_equal(extractor._statistics_signature(), signature)
    extractor.temporal_fill_window_sizes = (3, 5)
    expect_true(extractor._statistics_signature() != signature)
    extractor.temporal_fill_window_sizes = None
    extractor.spatial_statistics = ("mean", "median")
    expect_true(extractor._statistics_signature() != signature)


def _save_partition(extractor):
    """Write legacy partition outputs and their matching raw-cache directory."""
    for filename in (extractor.PARCEL_OUTPUT_FILE_NAME, extractor.REDUCED_PARCEL_OUTPUT_FILE_NAME):
        extractor.parcels.to_parquet(extractor.output_dir / filename)
    pd.DataFrame({"batch_number": [1]}).to_csv(extractor.output_dir / "satellite_pixel_cleaning_report.csv", index=False)
    (extractor.output_dir / "monthly_cubes" / extractor._run_signature()).mkdir(parents=True, exist_ok=True)


def test_legacy_partition_resume_requires_complete_outputs_and_matching_parcels(tmp_path):
    """Verify complete legacy outputs resume and incomplete or stale manifests do not."""
    extractor = _extractor(tmp_path, resume_completed_partitions=True)
    _save_partition(extractor)
    expect_is_not_none(extractor._load_completed_partition())
    extractor._write_partition_completion("processing")
    expect_is_none(extractor._load_completed_partition())
    extractor._write_partition_completion("complete")
    expect_is_not_none(extractor._load_completed_partition())
    extractor.spatial_statistics = ("mean", "median")
    expect_is_none(extractor._load_completed_partition())
    extractor._partition_completion_path().unlink()
    (extractor.output_dir / extractor.REDUCED_PARCEL_OUTPUT_FILE_NAME).unlink()
    expect_is_none(extractor._load_completed_partition())


def test_shared_scheduler_skips_completed_legacy_partition(tmp_path, monkeypatch):
    """Verify a completed legacy partition requires no scheduling or recalculation."""
    config = _config(tmp_path, resume_completed_partitions=True, remove_nc_after_completion=True)
    extractor = _extractor(tmp_path / "partition_1", resume_completed_partitions=True)
    _save_partition(extractor)

    def no_work(*args, **kwargs):
        """Fail if completed partition outputs trigger further batch work."""
        pytest.fail("A completed partition must not download or recalculate")

    monkeypatch.setattr(scheduler, "_drain_batches", no_work)
    monkeypatch.setattr(OpenEOJobManagerZonalStats, "run_from_batches", no_work)
    scheduler.run_shared_batches([_parcels()], [("user", "pw")], config, 1)


def test_shared_scheduler_processes_each_download_before_next(tmp_path, monkeypatch):
    # Two partitions exercise repeated work in a single account slot.
    """Verify each download is checkpointed before the slot acquires another batch."""
    config = _config(tmp_path, remove_nc_after_completion=True)
    events = []
    monkeypatch.setattr(scheduler.openeo, "connect", lambda *a, **kw: type("Connection", (), {
        "authenticate_oidc_resource_owner_password_credentials": lambda *a, **kw: None,
    })())

    def acquire(task, *args):
        """Simulate a download and record the partition acquisition order."""
        events.append(("download", task.partition))
        Path(task.plan["target_path"]).touch()

    def statistics(self, raw, parcels, number):
        """Record local processing and return deterministic batch tables."""
        events.append(("process", int(self.output_dir.name.split("_")[-1])))
        return _statistics()

    def finalize(self, batches):
        """Check that partition assembly reuses saved tables after raster deletion."""
        for number, batch, raw in batches:
            expect_false(raw.exists())
            expect_true(self._has_batch_statistics(raw))
            self._process_zonal_stats_batch(raw, batch, number)

    monkeypatch.setattr(scheduler, "_run_batch", acquire)
    monkeypatch.setattr(OpenEOJobManagerZonalStats, "_calculate_local_statistics", statistics)
    monkeypatch.setattr(OpenEOJobManagerZonalStats, "run_from_batches", finalize)
    scheduler.run_shared_batches([_parcels(), _parcels()], [("user", "pw")], config, 1)
    expect_equal(events, [("download", 1), ("process", 1), ("download", 2), ("process", 2)])
    events.clear()
    scheduler.run_shared_batches([_parcels(), _parcels()], [("user", "pw")], config, 1)
    expect_equal(events, [])


def test_real_raster_cleanup_and_partition_assembly_resume_offline(tmp_path, monkeypatch):
    """Verify real raster cleanup and interrupted partition assembly recover offline."""
    config = _config(tmp_path, remove_nc_after_completion=True, resume_completed_partitions=True)
    config["spatial_statistics"] = ["mean", "median"]
    extractor = OpenEOJobManagerZonalStats(**dict(config, parcels=_parcels(), output_dir=tmp_path / "partition_1"))
    raw, _ = _batch(extractor)
    with xr.Dataset(
        {"B02": (("t", "y", "x"), [[[1.0, 2.0], [3.0, 4.0]]])},
        coords={"t": pd.to_datetime(["2024-01-01"]), "x": [25.0, 75.0], "y": [75.0, 25.0]},
    ).rio.write_crs(3857) as dataset, warnings.catch_warnings():
        # netCDF4's old Cython ndarray declaration emits this known import warning.
        # Remove this scoped workaround once supported builds include the upstream fix:
        # https://github.com/Unidata/netcdf4-python/pull/1471
        warnings.filterwarnings(
            "ignore",
            message=r"^numpy\.ndarray size changed, may indicate binary incompatibility\. "
            r"Expected 16 from C header, got 96 from PyObject$",
            category=RuntimeWarning,
        )
        dataset.to_netcdf(raw, engine="netcdf4")
    # Remove dummy checkpoints so the real raster stages run.
    extractor._final_checkpoint_path(raw).unlink()
    extractor._cleaning_checkpoint_paths(raw)[0].unlink()

    def no_network(*args, **kwargs):
        """Fail if offline cleanup or resume attempts to contact openEO."""
        pytest.fail("Existing raw files and completed statistics must work offline")

    monkeypatch.setattr(scheduler.openeo, "connect", no_network)
    scheduler.run_shared_batches([_parcels()], [("user", "pw")], config, 1)
    expect_false(list(raw.parent.glob("*.nc")))
    saved = gpd.read_parquet(extractor.output_dir / extractor.PARCEL_OUTPUT_FILE_NAME)
    expect_equal(saved.B02_mean__20240101.iloc[0], pytest.approx(2.5))
    expect_is_not_none(extractor._load_completed_partition())

    # Simulate interruption during partition output writes, after batch cleanup.
    extractor._write_partition_completion("processing")
    (extractor.output_dir / extractor.PARCEL_OUTPUT_FILE_NAME).unlink()
    scheduler.run_shared_batches([_parcels()], [("user", "pw")], config, 1)
    restored = gpd.read_parquet(extractor.output_dir / extractor.PARCEL_OUTPUT_FILE_NAME)
    pd.testing.assert_frame_equal(saved, restored)
    expect_is_not_none(extractor._load_completed_partition())


def test_checkpoint_releases_tables_and_holds_lock_through_cleanup(tmp_path, monkeypatch):
    """Verify batch objects are collected and the raster lock covers every cleanup stage."""
    extractor = _extractor(tmp_path, remove_nc_after_completion=True)
    raw, _ = _batch(extractor)
    references = []
    events = []

    def calculate(*args):
        """Create batch tables and track their lifetimes through weak references."""
        expect_true(all(reference() is None for reference in references))
        result, report = _statistics()
        # A cycle tests garbage collection as well as ordinary reference drops.
        object.__setattr__(result, "_test_cycle", result)
        references.extend([weakref.ref(result), weakref.ref(report)])
        events.append("calculate")
        return result, report

    def assert_lock_held():
        """Verify another thread cannot acquire the raster lifecycle lock."""
        def try_lock():
            """Attempt a nonblocking lock acquisition from a separate worker thread."""
            acquired = base.LOCAL_RASTER_LOCK.acquire(blocking=False)
            if acquired:
                base.LOCAL_RASTER_LOCK.release()
            return acquired
        with ThreadPoolExecutor(max_workers=1) as executor:
            expect_true(executor.submit(try_lock).result() is False)

    save = extractor._save_batch_statistics
    remove = extractor._remove_completed_batch_netcdfs
    collect = base.gc.collect

    def save_batch(*args):
        """Verify lock ownership while committing tables and recording completion."""
        assert_lock_held()
        save(*args)
        events.append("saved")

    def remove_batch(path):
        """Verify tables are committed before deleting rasters under the lock."""
        assert_lock_held()
        expect_true(extractor._has_batch_statistics(path))
        remove(path)
        events.append("deleted")

    def collect_batch():
        """Collect batch objects under the lock and verify their references expire."""
        assert_lock_held()
        count = collect()
        expect_true(all(reference() is None for reference in references))
        events.append("released")
        return count

    monkeypatch.setattr(extractor, "_calculate_local_statistics", calculate)
    monkeypatch.setattr(extractor, "_save_batch_statistics", save_batch)
    monkeypatch.setattr(extractor, "_remove_completed_batch_netcdfs", remove_batch)
    monkeypatch.setattr(base.gc, "collect", collect_batch)
    for number in (1, 2):
        path = raw if number == 1 else raw.with_name("batch_00002_monthly.nc")
        path.touch()
        expect_equal(extractor._checkpoint_zonal_stats_batch(path, extractor.parcels, number), number)
        expect_false(path.exists())
    expect_equal(events, ["calculate", "saved", "deleted", "released"] * 2)


def test_failed_save_does_not_retain_statistics_in_traceback(tmp_path, monkeypatch):
    """Verify failed writes preserve rasters without pinning tables in traceback frames."""
    extractor = _extractor(tmp_path, remove_nc_after_completion=True)
    raw, _ = _batch(extractor)
    references = []

    def calculate(*args):
        """Create batch tables and track their lifetimes through weak references."""
        result, report = _statistics()
        references.extend([weakref.ref(result), weakref.ref(report)])
        return result, report

    def fail(path, result, report):
        """Simulate a disk-full error during calculation or checkpoint persistence."""
        raise OSError("disk full")

    monkeypatch.setattr(extractor, "_calculate_local_statistics", calculate)
    monkeypatch.setattr(extractor, "_save_batch_statistics", fail)
    with pytest.raises(OSError, match="disk full") as caught:
        extractor._checkpoint_zonal_stats_batch(raw, extractor.parcels, 1)
    expect_is_not_none(caught.value.__traceback__)
    expect_true(all(reference() is None for reference in references))
    expect_true(raw.exists())
