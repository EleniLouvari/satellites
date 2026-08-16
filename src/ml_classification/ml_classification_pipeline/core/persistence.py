"""Persistence, logging, timing, and filesystem utilities used across the pipeline."""

from __future__ import annotations

import importlib
import json
import os
import re
import shutil
import stat
import sys
import time

# Filesystem helpers use consistent encodings and artifact paths across all pipeline stages.
import traceback
from contextlib import contextmanager, nullcontext, redirect_stderr, redirect_stdout
from contextvars import ContextVar
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from common_libraries.io_library import write_data


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
    ensure_dir(resolved_log_path.parent)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    clean_message = _strip_ansi(str(message))
    with resolved_log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"[{timestamp}] [{level}] {clean_message}\n")


@contextmanager
def tee_output(log_path: str | Path):
    """Mirror stdout and stderr to both console and a log file."""
    # Open the log file once and duplicate writes through tee streams.
    log_path = Path(log_path)
    ensure_dir(log_path.parent)
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


def ensure_dir(path: str | Path) -> Path:
    """Create a directory path if needed and return it as Path."""
    # Normalize all directory creation through a single helper.
    folder = Path(path)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def delete_dir(path: str | Path) -> None:
    """Delete a directory tree, tolerating known permission edge cases."""
    # Try project-specific deletion helper before falling back to shutil.
    folder = Path(path)
    if folder.exists():
        if _delete_dir_with_repo_helper(folder):
            return
        try:
            shutil.rmtree(folder, onexc=_handle_remove_readonly)
        except PermissionError:
            return


def clear_directory(path: str | Path) -> None:
    """Remove all files and folders within a directory."""
    # Iterate child paths and delete each while handling permission issues.
    folder = ensure_dir(path)
    for item in folder.iterdir():
        if item.is_dir():
            delete_dir(item)
        else:
            try:
                item.unlink()
            except PermissionError:
                continue


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    """Serialize a dictionary to JSON on disk with standard formatting."""
    # Ensure parent directories exist before writing serialized content.
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    append_log(f"Saved JSON: {path}", level="INFO")


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON document from disk into a dictionary."""
    # Record read access for traceability in pipeline logs.
    path = Path(path)
    append_log(f"Loaded JSON: {path}", level="INFO")
    return json.loads(path.read_text(encoding="utf-8"))


def save_joblib(payload: Any, path: str | Path) -> None:
    """Persist a Python object to disk using joblib compression."""
    # Use compressed joblib artifacts to reduce output size.
    path = Path(path)
    ensure_dir(path.parent)
    joblib.dump(payload, path, compress=3)
    append_log(f"Saved joblib: {path}", level="INFO")


def load_joblib(path: str | Path) -> Any:
    """Load a joblib artifact from disk."""
    # Record artifact loading so report logs capture data dependencies.
    path = Path(path)
    append_log(f"Loaded joblib: {path}", level="INFO")
    return joblib.load(path)


def save_frame_csv(df: pd.DataFrame, path: str | Path) -> None:
    """Save a dataframe to CSV without index columns."""
    # Persist tabular artifacts in a portable plain-text format.
    path = Path(path)
    ensure_dir(path.parent)
    write_data(df, str(path), plain_csv=True)
    append_log(f"Saved CSV: {path} with shape={df.shape}", level="INFO")


def reset_project_outputs(config) -> None:
    """Clear and reinitialize project output folders for a new run."""
    # Reset previous artifacts while honoring cleanup strictness settings.
    existed_before = config.project_dir.exists()
    append_log(f"Resetting project outputs under: {config.project_dir}", level="INFO", log_path=config.log_path)
    clear_directory(config.project_dir)
    for step_dir in (config.check_dir, config.prepare_dir, config.train_dir, config.evaluate_dir, config.predict_dir):
        delete_dir(step_dir)
    if config.fail_on_cleanup_error and existed_before:
        leftovers = [path for path in config.project_dir.glob("*")]
        if leftovers:
            raise PermissionError(f"Could not fully clear existing project directory: {config.project_dir}")
    append_log(f"Project outputs ready under: {config.project_dir}", level="INFO", log_path=config.log_path)


def _handle_remove_readonly(func, path, exc_info) -> None:
    """Handle read-only filesystem entries during recursive deletion."""
    # Relax file permissions, then retry the original removal function.
    os.chmod(path, stat.S_IWRITE)
    func(path)


def _delete_dir_with_repo_helper(folder: Path) -> bool:
    """Attempt directory deletion via optional repository helper library."""
    # Gracefully fall back when helper modules are unavailable.
    try:
        io_library = importlib.import_module("common_libraries.io_library")
        io_library.delete_folder(str(folder))
        return not folder.exists()
    except Exception:
        return False


def _json_default(value: Any) -> Any:
    """Provide JSON serialization fallbacks for known custom value types."""
    # Convert common non-JSON-native values before final serialization.
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")
