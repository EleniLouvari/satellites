"""Top-level orchestration entrypoint for the classification pipeline steps."""

from __future__ import annotations

from typing import Any
import warnings

import pandas as pd

from .core.persistence import print_formatted_txt, time_decorator
from .core.config import ClassificationPipelineConfig
from .steps import CheckStep, EvaluateStep, PredictStep, PrepareStep, TrainStep


class GeospatialClassificationPipeline(CheckStep, PrepareStep, TrainStep, EvaluateStep, PredictStep):
    """Compose all step mixins into a geospatial-ready classification workflow."""

    @time_decorator
    def run_all(self, df: pd.DataFrame) -> dict[str, Any]:
        """Run all pipeline steps in order and return step summaries."""
        # Execute the end-to-end pipeline workflow.
        print_formatted_txt("ML Classification Pipeline...", "SECTION")
        return {
            "check": self.run_check(df),
            "prepare": self.run_prepare(),
            "train": self.run_train(),
            "evaluate": self.run_evaluate(),
            "predict": self.run_predict(),
        }


class AutonomousClassificationPipeline(GeospatialClassificationPipeline):
    """Backward-compatible alias for `GeospatialClassificationPipeline`.

    Notes:
        Deprecated. Prefer `GeospatialClassificationPipeline` in new code.
    """

    def __init__(self, config: ClassificationPipelineConfig):
        """Initialize deprecated alias and emit a deprecation warning."""
        warnings.warn(
            "AutonomousClassificationPipeline is deprecated; use GeospatialClassificationPipeline instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(config)
