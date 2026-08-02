"""Core EDA analysis primitives."""

from .config import EDAConfig
from .profiling import build_eda_artifacts

__all__ = ["EDAConfig", "build_eda_artifacts"]
