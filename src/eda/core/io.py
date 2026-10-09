"""Filesystem and serialization helpers for EDA artifacts."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from shared.io import write_data

# Keep this helper focused on a single transformation so the reporting pipeline stays easy to follow.



def ensure_dir(path: str | Path) -> Path:
    """Create a directory and return it as a Path."""
    folder = Path(path)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def reset_dir(path: str | Path) -> Path:
    """Delete and recreate a directory."""
    folder = Path(path)
    if folder.exists():
        # Reset removes all prior artifacts; callers must supply a dedicated EDA output directory.
        shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def save_json(payload: dict[str, Any], path: str | Path) -> Path:
    """Write a dictionary to JSON."""
    output_path = Path(path)
    ensure_dir(output_path.parent)
    output_path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    return output_path


def save_frame(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a dataframe to CSV."""
    output_path = Path(path)
    ensure_dir(output_path.parent)
    # Use plain CSV for report tables rather than geometry-aware serialization.
    write_data(df, str(output_path), plain_csv=True)
    return output_path


def save_parquet_frame(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a DataFrame or GeoDataFrame to Parquet while preserving geometry metadata."""
    output_path = Path(path)
    ensure_dir(output_path.parent)
    # Let the shared writer preserve GeoParquet metadata when the input carries geometry.
    write_data(df, str(output_path))
    return output_path


def save_text(value: str, path: str | Path) -> Path:
    """Write a UTF-8 text artifact."""
    output_path = Path(path)
    ensure_dir(output_path.parent)
    output_path.write_text(value, encoding="utf-8")
    return output_path


def _json_default(value: Any) -> Any:
    """Serialize common non-JSON-native values."""
    # Convert NumPy scalars before falling back to array conversion or display text.
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    # ISO timestamps retain their date/time meaning in the JSON summary.
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)
