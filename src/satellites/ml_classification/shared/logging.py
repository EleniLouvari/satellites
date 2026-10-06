"""Console logging, output capture, and timing shared by pipeline steps."""

from __future__ import annotations

import re
import sys
import time
import traceback
from contextlib import contextmanager, nullcontext, redirect_stderr, redirect_stdout
from contextvars import ContextVar
from datetime import datetime
from functools import wraps
from pathlib import Path

ACTIVE_LOG_PATH: ContextVar[Path | None] = ContextVar("ACTIVE_LOG_PATH", default=None)


class TeeStream:
    """Mirror stream writes to multiple destinations such as console and log files."""

    def __init__(self, *streams):
        """Initialize the tee stream with output destinations."""
        # Preserve destination order so console output stays first.
        self.streams = streams

    def write(self, data):
        """Write incoming text to every configured destination stream."""
        # Strip ANSI color codes for non-primary streams like log files so
        # log files remain readable while console retains styling.
        for idx, stream in enumerate(self.streams):
            text = _strip_ansi(data) if idx > 0 else data
            stream.write(text)
        return len(data)

    def flush(self):
        """Flush all destination streams."""
        # Forward flush calls so buffered log/file streams are persisted.
        for stream in self.streams:
            stream.flush()

    def close(self):
        """Allow logging-handler shutdown without closing borrowed streams.

        The console belongs to the caller and the log file is managed by
        tee_output. A handler can retain this wrapper after that context exits.
        """
        return None


def _strip_ansi(text: str) -> str:
    """Remove ANSI color escape sequences from text."""
    # Keep log output readable by removing terminal styling codes.
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def append_log(message: str, level: str = "INFO", log_path: str | Path | None = None) -> None:
    """Append a timestamped log entry to the active or provided log file."""
    # Resolve the target log path from explicit input or context-local state.
    resolved_log_path = Path(log_path) if log_path is not None else ACTIVE_LOG_PATH.get()
    if resolved_log_path is None:
        return
    resolved_log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    clean_message = _strip_ansi(str(message))
    with resolved_log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"[{timestamp}] [{level}] {clean_message}\n")


@contextmanager
def tee_output(log_path: str | Path):
    """Mirror stdout and stderr to both console and a log file."""
    # Open the log file once and duplicate writes through tee streams.
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        stdout_tee = TeeStream(sys.stdout, log_file)
        stderr_tee = TeeStream(sys.stderr, log_file)
        with redirect_stdout(stdout_tee), redirect_stderr(stderr_tee):
            yield


def print_formatted_txt(msg: str, txt_format: str = "SECTION") -> None:
    """Print color-formatted text and persist the same message to logs."""
    # Keep terminal messaging and log records synchronized.
    print("")
    if msg.lower().startswith("start"):
        print(100 * "-")
    format_dict = {
        "RUN": "\033[1;36m",
        "SECTION": "\033[1;37;44m",
        "SUBSECTION": "\033[1;30m",
        "INFO": "\033[0;90m",
        "RESULTS": "\033[1;32m",
        "WARNING": "\033[1;33m",
        "ERROR": "\033[1;37;41m",
        "END": "\033[0m",
    }
    style = format_dict.get(txt_format, format_dict["INFO"])
    print(f"{style}{msg}{format_dict['END']}")
    append_log(msg, level=txt_format)
    if msg.lower().startswith("end"):
        print(100 * "*")
        print("")


def calculate_time_duration(begin_time: float, end_time: float) -> str:
    """Format elapsed seconds as HH:MM:SS."""
    # Convert raw elapsed seconds into a fixed-width human-readable duration.
    total_secs = end_time - begin_time
    hours, remainder = divmod(total_secs, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"


def time_decorator(func):
    """Wrap a function with runtime logging, timing, and error capture."""

    # Preserve function metadata while adding execution instrumentation.
    @wraps(func)
    def wrapper(*args, **kwargs):
        """Execute the wrapped function inside logging and timing contexts."""
        # Attach step-level logs when a config object is available on self.
        config = getattr(args[0], "config", None) if args else None
        log_context = tee_output(config.log_path) if config is not None else nullcontext()
        with log_context:
            log_token = ACTIVE_LOG_PATH.set(config.log_path) if config is not None else None
            begin_time = time.time()
            begin_datetime = datetime.fromtimestamp(begin_time)
            text2 = f"Function <{func.__name__}> started on {begin_datetime.strftime('%Y-%m-%d %H:%M:%S')}"
            text1 = "=" * (len(text2) + 6)
            print_formatted_txt(text1 + "\n" + text2, "RUN")
            try:
                output = func(*args, **kwargs)
                append_log(f"{func.__name__} returned: {repr(output)}", level="INFO")
                end_time = time.time()
                end_datetime = datetime.fromtimestamp(end_time)
                duration = calculate_time_duration(begin_time, end_time)
                text1 = (
                    f"Execution time of function <{func.__name__}> ended on "
                    f"{end_datetime.strftime('%Y-%m-%d %H:%M:%S')} - duration: {duration}"
                )
                text2 = "=" * (len(text1) + 6)
                print_formatted_txt(text1 + "\n" + text2, "RUN")
                return output
            except Exception as exc:
                append_log(f"{func.__name__} failed: {exc}", level="ERROR")
                append_log(traceback.format_exc(), level="ERROR")
                raise
            finally:
                if log_token is not None:
                    ACTIVE_LOG_PATH.reset(log_token)

    return wrapper
