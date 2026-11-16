"""Forwarding of spawned batch-worker log records to the parent process."""

import logging
from concurrent.futures import ProcessPoolExecutor

from data_preparation.parcel_stats.core.worker_logging import forward_worker_logger, spawn_pool_logging
from tests.utils import expect_equal, expect_true


def _log_in_worker(name):
    """Log like a batch worker: own console handler plus forwarding, then return the handler count."""
    logger = logging.getLogger(name)
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.INFO)
    logger.propagate = False
    forward_worker_logger(logger)
    # A second call (another unpickled task in the same worker) must not duplicate forwarding.
    forward_worker_logger(logger)
    logger.info("Cleaned acquisition %s/%s for batch %s.", 3, 77, 6)
    logger.debug("Below the worker level; never forwarded.")
    return len(logger.handlers)


class _ListHandler(logging.Handler):
    """Collect formatted messages emitted in the parent."""

    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def test_spawned_worker_records_reach_parent_logger_and_record_factory():
    """Worker records are re-created in the parent, so caller record factories observe them."""
    name = "parcel_stats_pipeline.worker.test.forwarding.parcels"
    parent_logger = logging.getLogger(name)
    collected = _ListHandler()
    parent_logger.addHandler(collected)
    parent_logger.propagate = False
    previous_factory = logging.getLogRecordFactory()
    seen_by_factory = []

    def factory(*args, **kwargs):
        record = previous_factory(*args, **kwargs)
        if record.name == name:
            seen_by_factory.append(record.getMessage())
        return record

    logging.setLogRecordFactory(factory)
    try:
        with spawn_pool_logging() as options, ProcessPoolExecutor(max_workers=1, **options) as executor:
            handler_count = executor.submit(_log_in_worker, name).result()
    finally:
        logging.setLogRecordFactory(previous_factory)
        parent_logger.removeHandler(collected)

    # One local handler plus exactly one queue handler.
    expect_equal(handler_count, 2)
    expect_equal(collected.messages, ["Cleaned acquisition 3/77 for batch 6."])
    expect_equal(seen_by_factory, ["Cleaned acquisition 3/77 for batch 6."])


def test_forwarding_is_inactive_outside_pool_workers():
    """Loggers created in the parent or in sequential runs keep only their own handlers."""
    logger = logging.getLogger("parcel_stats_pipeline.worker.test.parent-only")
    expect_true(forward_worker_logger(logger) is logger)
    expect_equal(logger.handlers, [])


def test_forwarded_parent_logger_writes_nothing_itself():
    """A forwarded logger without handlers gets a null handler and does not reach the root logger."""
    name = "parcel_stats_pipeline.worker.test.silent.filling"
    with spawn_pool_logging() as options, ProcessPoolExecutor(max_workers=1, **options) as executor:
        executor.submit(_log_in_worker, name).result()
    logger = logging.getLogger(name)
    expect_true(all(isinstance(handler, logging.NullHandler) for handler in logger.handlers))
    expect_equal(logger.propagate, False)
