"""Exercise incremental acquisition with real spawned local checkpoint workers."""

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Manager

import pandas as pd
import pytest

from data_preparation.parcel_stats.core.base import ParcelStatsBase
from data_preparation.parcel_stats.core.batch_execution import run_streamed_batch_statistics


class _CheckpointProcessor(ParcelStatsBase):
    """Replace scientific calculation with deterministic, synchronized batch tables."""

    def __init__(self, directory, started, release, workers=1, cleanup=True, failure=None):
        """Keep real checkpoint writes, cleanup, serialization and worker logging."""
        self.output_dir = directory
        self.batch_workers = workers
        self.remove_nc_after_completion = cleanup
        self.started = started
        self.release = release
        self.failure = failure
        self.logger = logging.getLogger(__name__)
        self.filling_logger = self.logger
        self.parcel_logger = self.logger

    def _statistics_signature(self):
        """Use a fixed fixture identity independent of scientific configuration."""
        return "fixture"

    def _cleaning_checkpoint_paths(self, path):
        """Name the fixture's unused cleaning files for real cleanup."""
        return path.with_suffix(".cleaned.nc"), path.with_suffix(".cleaning.parquet")

    def _final_checkpoint_path(self, path):
        """Name the fixture's unused final raster for real cleanup."""
        return path.with_suffix(".final.nc")

    def _calculate_local_statistics(self, path, batch, number):
        """Signal actual execution and wait for explicit test-controlled completion."""
        self.started.put((number, os.getpid()))
        if not self.release[number].wait(timeout=30):
            raise TimeoutError("Error: Test did not release local processing.")
        if number == self.failure:
            raise ValueError("Error: Local processing failed.")
        self.parcel_logger.info("Processed fixture batch %s", number)
        return pd.DataFrame({"value": [number]}), pd.DataFrame({"report": [number]})


def _cube(directory, number):
    """Create a committed raw cube stand-in for scheduler/checkpoint tests."""
    path = directory / f"batch_{number:05d}_monthly.nc"
    path.write_bytes(b"complete fixture")
    return number, None, path


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("cleanup", [False, True])
def test_processing_starts_before_acquisition_finishes(tmp_path, workers, cleanup):
    """An eager acquisition list or an in-process worker breaks this handshake."""
    with Manager() as manager:
        started = manager.Queue()
        release = {number: manager.Event() for number in range(1, workers + 2)}
        processor = _CheckpointProcessor(tmp_path, started, release, workers, cleanup)

        def acquire():
            """Wait for earlier local work while acquiring the final cube."""
            for number in range(1, workers + 1):
                yield _cube(tmp_path, number)
            observed = [started.get(timeout=30) for _ in range(workers)]
            assert sorted(number for number, _ in observed) == list(range(1, workers + 1))
            assert all(pid != os.getpid() for _, pid in observed)
            for event in release.values():
                event.set()
            yield _cube(tmp_path, workers + 1)

        results, reports = run_streamed_batch_statistics(processor, acquire())
        assert [int(frame.value.iloc[0]) for frame in results] == list(range(1, workers + 2))
        assert [int(frame.report.iloc[0]) for frame in reports] == list(range(1, workers + 2))
        assert len(list(tmp_path.glob("*.nc"))) == (0 if cleanup else workers + 1)


def test_acquisition_waits_when_local_workers_and_one_ready_cube_are_full(tmp_path):
    """Acquisition cannot build an unbounded raw-cube backlog behind slow workers."""
    with Manager() as manager:
        started = manager.Queue()
        release = {number: manager.Event() for number in range(1, 5)}
        third_acquired, fourth_acquired = manager.Event(), manager.Event()
        processor = _CheckpointProcessor(tmp_path, started, release, workers=2)

        def acquire():
            """Expose when the producer is allowed to advance beyond a full queue."""
            yield _cube(tmp_path, 1)
            yield _cube(tmp_path, 2)
            third_acquired.set()
            yield _cube(tmp_path, 3)
            fourth_acquired.set()
            yield _cube(tmp_path, 4)

        with ThreadPoolExecutor(max_workers=1) as caller:
            future = caller.submit(run_streamed_batch_statistics, processor, acquire())
            try:
                assert third_acquired.wait(timeout=30)
                assert sorted(started.get(timeout=30)[0] for _ in range(2)) == [1, 2]
                assert not fourth_acquired.is_set()
            finally:
                for event in release.values():
                    event.set()
            results, _ = future.result(timeout=30)
        assert [int(frame.value.iloc[0]) for frame in results] == [1, 2, 3, 4]


def test_acquisition_failure_keeps_completed_checkpoints(tmp_path):
    """An upstream outage propagates without discarding work already in flight."""
    with Manager() as manager:
        started = manager.Queue()
        release = {1: manager.Event()}
        processor = _CheckpointProcessor(tmp_path, started, release)

        def acquire():
            """Fail after a real local worker starts processing the first cube."""
            yield _cube(tmp_path, 1)
            assert started.get(timeout=30)[0] == 1
            release[1].set()
            raise OSError("Error: Acquisition failed.")

        with pytest.raises(OSError, match="Acquisition failed"):
            run_streamed_batch_statistics(processor, acquire())
        path = tmp_path / "batch_00001_monthly.nc"
        assert processor._has_batch_statistics(path)
        assert not path.exists()


def test_local_failure_stops_acquisition_and_retains_failed_cube(tmp_path):
    """Worker errors stop further submissions rather than silently losing a batch."""
    with Manager() as manager:
        started = manager.Queue()
        release = {number: manager.Event() for number in range(1, 4)}
        for event in release.values():
            event.set()
        processor = _CheckpointProcessor(tmp_path, started, release, failure=1)
        acquired = []

        def acquire():
            """Produce one extra ready cube while the first worker is failing."""
            for number in range(1, 4):
                acquired.append(number)
                yield _cube(tmp_path, number)

        with pytest.raises(ValueError, match="Local processing failed"):
            run_streamed_batch_statistics(processor, acquire())
        assert acquired == [1, 2]
        assert (tmp_path / "batch_00001_monthly.nc").exists()
        assert not processor._has_batch_statistics(tmp_path / "batch_00001_monthly.nc")


def test_completed_caches_resume_without_starting_worker_pool(tmp_path, monkeypatch):
    """Reusable statistics need neither raw cubes nor new worker processes."""
    processor = _CheckpointProcessor(tmp_path, None, {})
    batches = [_cube(tmp_path, number) for number in (2, 1)]
    for number, _, path in batches:
        processor._save_batch_statistics(path, pd.DataFrame({"value": [number]}), pd.DataFrame({"report": [number]}))

    def unexpected_pool(*args, **kwargs):
        """Fail if a cache-only resume starts a new processing pool."""
        pytest.fail("Cache-only resume started a worker pool")

    monkeypatch.setattr("data_preparation.parcel_stats.core.batch_execution.ProcessPoolExecutor", unexpected_pool)
    results, _ = run_streamed_batch_statistics(processor, iter(batches))
    assert [int(frame.value.iloc[0]) for frame in results] == [1, 2]
    assert not list(tmp_path.glob("*.nc"))
