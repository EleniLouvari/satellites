"""Shared package; import helpers from their defining modules."""

from pathlib import Path as _Path

# Keep historical module paths from the flat shared layout available during testing.
_compatibility_directory = _Path(__file__).resolve().parents[1] / "to_delete" / "shared"
if _compatibility_directory.is_dir():
    __path__.append(str(_compatibility_directory))
