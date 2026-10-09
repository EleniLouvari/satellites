"""JSON, joblib, CSV, and output-directory operations shared by pipeline steps."""

from __future__ import annotations

import importlib
import json
import logging
import os
import shutil
import stat
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from ml_classification.shared.logging import append_log
from shared.io import write_data


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
            raise PermissionError(f"Error: Could not fully clear existing project directory: {config.project_dir}")
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
        io_library = importlib.import_module("shared.io")
        io_library.delete_folder(str(folder))
        return not folder.exists()
    except Exception:
        logging.getLogger(__name__).debug("Error: _delete_dir_with_repo_helper failed; using its fallback.", exc_info=True)
        return False


def _json_default(value: Any) -> Any:
    """Provide JSON serialization fallbacks for known custom value types."""
    # Convert common non-JSON-native values before final serialization.
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Error: Object of type {type(value)!r} is not JSON serializable")
