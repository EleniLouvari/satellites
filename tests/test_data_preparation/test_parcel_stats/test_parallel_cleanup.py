"""Exercise cleanup with real spawned processes, raster files and durable caches."""

import multiprocessing
import os
from pathlib import Path

import pandas as pd
import pytest
import xarray as xr

from data_preparation.parcel_stats import batch_scheduler as scheduler
from data_preparation.parcel_stats.core import base
from data_preparation.parcel_stats.job_manager import OpenEOJobManagerZonalStats
from tests.test_data_preparation.test_parcel_stats.test_netcdf_cleanup import _config, _parcels


class ConcurrentCleanupExtractor(OpenEOJobManagerZonalStats):
    """Synchronize test raster work while retaining real processing and persistence."""

    def _calculate_local_statistics(self, path, parcels, number):
        """Require four simultaneous processes before reading the tiny test raster."""
        Path(path).with_suffix(".pid").write_text(str(os.getpid()), encoding="utf-8")
        self.test_barrier.wait(timeout=30)
        return super()._calculate_local_statistics(path, parcels, number)

    def _save_batch_statistics(self, path, result, report):
        """Optionally fail persistence, after successful raster calculation."""
        if result["batch_number"].iloc[0] == self.test_fail_batch:
            raise OSError("Error: simulated checkpoint write failure")
        return super()._save_batch_statistics(path, result, report)


def _write_raster(path):
    """Write four known pixel values in the parcel's CRS without acquisition."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with xr.Dataset(
        {"B02": (("t", "y", "x"), [[[1.0, 2.0], [3.0, 4.0]]])},
        coords={"t": pd.to_datetime(["2024-01-01"]), "x": [25.0, 75.0], "y": [75.0, 25.0]},
    ).rio.write_crs(3857) as dataset:
        dataset.to_netcdf(path, engine="netcdf4")


@pytest.mark.parametrize("shared_queue", [False, True])
@pytest.mark.parametrize("fail_save", [False, True])
def test_four_cleanup_workers_preserve_checkpoints_and_failed_rasters(tmp_path, monkeypatch, shared_queue, fail_save):
    """Catch serialized cleanup, account-slot limits and deletion before a successful save."""
    config = _config(tmp_path, remove_nc_after_completion=True)
    # Direct Python callers have historically accepted integer-like strings.
    config["batch_workers"] = "4" if shared_queue else 4
    config["spatial_statistics"] = ["mean", "median"]
    batch_count = 4 if fail_save else 8
    entries = []
    with multiprocessing.get_context("spawn").Manager() as manager:
        barrier = manager.Barrier(4)

        def build(options, part_idx, parcels):
            """Create picklable extractors whose local stages rendezvous across partitions."""
            extractor = ConcurrentCleanupExtractor(**dict(
                options, parcels=parcels, output_dir=tmp_path / f"partition_{part_idx}",
                run_identifier=f"parallel_part{part_idx}",
            ))
            extractor.test_barrier = barrier
            extractor.test_fail_batch = (1 if shared_queue else 2) if fail_save and part_idx == 2 else None
            return extractor

        if shared_queue:
            monkeypatch.setattr(scheduler, "_build_partition_extractor", build)
            monkeypatch.setattr(scheduler.openeo, "connect", lambda *args, **kwargs: type("Connection", (), {
                "authenticate_oidc_resource_owner_password_credentials": lambda *args, **kwargs: None,
            })())

            def download(task, *args):
                """Replace only the remote download; local processing remains real."""
                path = Path(task.plan["target_path"])
                _write_raster(path)
                entries.append((task.extractor, path, task.plan["batch_number"]))

            monkeypatch.setattr(scheduler, "_run_batch", download)

            def run():
                """Feed four local workers from just one remote account slot."""
                scheduler.run_shared_batches([_parcels()] * batch_count, [("offline", "password")], config, 1)
        else:
            extractor = build(config, 2, _parcels())
            for number in range(1, batch_count + 1):
                path = extractor.output_dir / f"batch_{number:05d}_monthly.nc"
                _write_raster(path)
                entries.append((extractor, path, number))

            def run():
                """Exercise the same direct path used by partition scheduling."""
                results, _ = extractor._run_batch_statistics([
                    (number, extractor.parcels, path) for _, path, number in reversed(entries)
                ])
                assert [int(frame.batch_number.iloc[0]) for frame in results] == list(range(1, batch_count + 1))

        if fail_save:
            with pytest.raises(OSError, match="simulated checkpoint write failure"):
                run()
        else:
            run()

    assert len(entries) == batch_count
    pids = {int(path.with_suffix(".pid").read_text(encoding="utf-8")) for _, path, _ in entries}
    assert len(pids) == 4
    assert os.getpid() not in pids
    for extractor, path, number in entries:
        failed = number == extractor.test_fail_batch
        assert path.exists() == failed
        assert extractor._has_batch_statistics(path) != failed
        if not failed:
            result, report = extractor._cached_batch_statistics(path)
            assert result.B02_mean.iloc[0] == pytest.approx(2.5)
            assert not report.empty
            # A completed batch must resume without entering the broken/closed barrier.
            assert extractor._checkpoint_zonal_stats_batch(path, extractor.parcels, number) == number


@pytest.mark.parametrize("batch_count", [0, 2])
def test_cleanup_assembly_does_not_spawn_workers_for_cached_or_empty_batches(tmp_path, monkeypatch, batch_count):
    """Avoid starting redundant pools during shared-queue partition assembly."""
    config = _config(tmp_path, remove_nc_after_completion=True)
    config["batch_workers"] = 4
    extractor = OpenEOJobManagerZonalStats(parcels=_parcels(), **config)
    batches = []
    for number in range(batch_count):
        path = tmp_path / f"batch_{number}.nc"
        path.touch()
        extractor._save_batch_statistics(path, pd.DataFrame({"batch_number": [number]}), pd.DataFrame({"audit": [1]}))
        batches.append((number, extractor.parcels, path))

    def unexpected_pool(*args, **kwargs):
        """Cached assembly must not multiply the shared queue's process limit."""
        pytest.fail("Cached or empty cleanup batches must not spawn workers")

    monkeypatch.setattr(base, "ProcessPoolExecutor", unexpected_pool)
    results, reports = extractor._run_batch_statistics(batches)
    assert len(results) == len(reports) == batch_count
    assert all(not path.exists() for _, _, path in batches)
