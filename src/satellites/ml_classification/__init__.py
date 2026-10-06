"""Public exports for the geospatial machine-learning classification pipeline."""

from pathlib import Path as _Path

from satellites.ml_classification.pipeline import GeospatialClassificationPipeline
from satellites.ml_classification.shared.config.config import ClassificationPipelineConfig

# Temporary support for historical imports while the archived wrappers are tested.
# Removing to_delete later leaves the new pipeline and numbered steps independent.
_compatibility_directory = _Path(__file__).parent / "to_delete"
if _compatibility_directory.is_dir():
    __path__.append(str(_compatibility_directory))

# Keep the package-level API deliberately small and stable for downstream imports.
__all__ = ["GeospatialClassificationPipeline", "ClassificationPipelineConfig"]
