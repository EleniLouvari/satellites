"""Core package exports for configuration, base classes, and shared pipeline utilities."""

from .base import PipelineStepBase
from .config import ClassificationPipelineConfig
from .spatial_split import SpatialTrainTestSplitter
from .metrics import *  # noqa: F401,F403
from .persistence import *  # noqa: F401,F403
from .selection import *  # noqa: F401,F403

__all__ = [
    "PipelineStepBase",
    "ClassificationPipelineConfig",
    "SpatialTrainTestSplitter",
]
