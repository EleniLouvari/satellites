"""Public exports for the geospatial machine-learning classification pipeline."""

from .core.config import ClassificationPipelineConfig
from .pipeline import GeospatialClassificationPipeline

# Keep the package-level API deliberately small and stable for downstream imports.
__all__ = ["GeospatialClassificationPipeline", "ClassificationPipelineConfig"]
