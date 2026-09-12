"""Core package exports for configuration, base classes, and shared pipeline utilities.

This module re-exports the most commonly used core primitives so callers can
import them from a single location (for example: ``from satellites.ml_classification.core import ClassificationPipelineConfig``).
"""

from .base import PipelineStepBase
from .config import ClassificationPipelineConfig
from .spatial_split import SpatialTrainTestSplitter
from .metrics import *  # noqa: F401,F403
from .persistence import *  # noqa: F401,F403
from .selection import *  # noqa: F401,F403

# Re-export the most commonly used core primitives from a single import location.
__all__ = ["PipelineStepBase", "ClassificationPipelineConfig", "SpatialTrainTestSplitter"]
