"""Forward log records from spawned batch workers to the parent process.

Spawned workers start from a fresh interpreter, so logging hooks installed by the
caller (for example a web application's event publisher) never see their records.
Workers still write their own ``*.worker-<pid>.log`` files; in addition, each worker
logger sends its records through a process queue, and the parent logs the message
again under the same logger name, where the caller's hooks apply.
"""

import logging
from contextlib import contextmanager
from logging.handlers import QueueHandler, QueueListener
from multiprocessing import get_context

# Set only inside pool workers by ``_install_worker_log_queue``; ``None`` in the parent.
_WORKER_LOG_QUEUE = None


def _install_worker_log_queue(log_queue) -> None:
    """Pool initializer: remember the parent's queue for worker loggers created later."""
    global _WORKER_LOG_QUEUE
    _WORKER_LOG_QUEUE = log_queue


def forward_worker_logger(logger: logging.Logger) -> logging.Logger:
    """Attach one queue handler to a worker logger when the parent is listening."""
    if _WORKER_LOG_QUEUE is not None and not any(isinstance(handler, QueueHandler) for handler in logger.handlers):
        logger.addHandler(QueueHandler(_WORKER_LOG_QUEUE))
    return logger


class _ParentReemitHandler(logging.Handler):
    """Log forwarded worker messages again in the parent without writing them a second time."""

    def emit(self, record: logging.LogRecord) -> None:
        """Create a new parent record so record factories and filters run in this process."""
        logger = logging.getLogger(record.name)
        if not logger.handlers:
            # Workers already wrote console and file output; the parent only exposes the record to hooks.
            logger.addHandler(logging.NullHandler())
            logger.propagate = False
        # makeRecord applies the active record factory; QueueHandler already merged the message arguments.
        # The worker applied its level, so handle() bypasses the parent logger's level check.
        logger.handle(
            logger.makeRecord(
                record.name, record.levelno, record.pathname, record.lineno, record.getMessage(), (), None, record.funcName
            )
        )


@contextmanager
def spawn_pool_logging():
    """Yield ``ProcessPoolExecutor`` keyword arguments that forward worker logs to this process.

    Enter this context before the pool so that the pool shuts down, and its workers flush
    their queued records, before the listener stops.
    """
    context = get_context("spawn")
    log_queue = context.Queue()
    listener = QueueListener(log_queue, _ParentReemitHandler())
    listener.start()
    try:
        yield {"mp_context": context, "initializer": _install_worker_log_queue, "initargs": (log_queue,)}
    finally:
        listener.stop()
        log_queue.close()
        log_queue.join_thread()
