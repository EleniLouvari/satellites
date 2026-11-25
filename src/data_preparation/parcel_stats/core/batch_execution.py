"""Bounded local processing of incrementally acquired parcel cubes."""

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import ExitStack
from pathlib import Path

from .worker_logging import spawn_pool_logging


def _collect_completed(pending, *, block=False):
    """Release completed futures and propagate worker failures or batch mismatches."""
    if not pending:
        return
    completed, _ = wait(pending, timeout=None if block else 0, return_when=FIRST_COMPLETED)
    for future in completed:
        expected = pending.pop(future)
        number = future.result()
        if number != expected:
            raise RuntimeError(f"Error: Worker returned batch {number}; expected {expected}.")


def run_streamed_batch_statistics(processor, batches):
    """Process ready cubes in isolated workers while the producer acquires more.

    Keep at most ``batch_workers`` submitted jobs and one acquired cube waiting
    for a slot. Workers persist tables and return only batch IDs, irrespective
    of raster-retention mode. Load ordered tables only after all work succeeds.
    Cache-only resumes validate tables and clean up without starting a pool.
    """
    records, pending = [], {}
    executor = None
    iterator = iter(batches)
    with ExitStack() as resources:
        try:
            for number, parcels, path in iterator:
                _collect_completed(pending)
                records.append((number, path))
                if processor._has_batch_statistics(path):
                    processor._checkpoint_zonal_stats_batch(path, parcels, number)
                    continue
                if executor is None:
                    for logger, name in (
                        (processor.filling_logger, processor.FILLING_LOG_FILE_NAME),
                        (processor.parcel_logger, processor.PARCEL_LOG_FILE_NAME),
                    ):
                        logger.info(
                            "Streaming local processing uses up to %s workers; logs: %s.worker-<pid>%s in %s.",
                            processor.batch_workers, Path(name).stem, Path(name).suffix, processor.output_dir,
                        )
                    options = resources.enter_context(spawn_pool_logging())
                    executor = resources.enter_context(ProcessPoolExecutor(max_workers=processor.batch_workers, **options))
                # Acquire the next cube before waiting so even a single local
                # worker can overlap with acquisition in the parent process.
                while len(pending) >= processor.batch_workers:
                    _collect_completed(pending, block=True)
                future = executor.submit(processor._checkpoint_zonal_stats_batch, path, parcels, number)
                pending[future] = number
            while pending:
                _collect_completed(pending, block=True)
        except BaseException:
            # Running workers finish their checkpoint before pool shutdown.
            # Unstarted jobs are cancelled and their raw cubes remain reusable.
            for future in pending:
                future.cancel()
            raise
        finally:
            close = getattr(iterator, "close", None)
            if close is not None:
                close()
    completed = [processor._cached_batch_statistics(path) for _, path in sorted(records)]
    return [result for result, _ in completed], [report for _, report in completed]
