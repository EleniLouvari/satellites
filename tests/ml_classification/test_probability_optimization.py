from pathlib import Path

import numpy as np
import pytest

from satellites.ml_classification.core.config import ClassificationPipelineConfig
from satellites.ml_classification.core.selection import (
    apply_class_probability_multipliers,
    optimize_probability_multipliers,
)


def _config(**overrides):
    values = {
        "project_dir": Path("unused"),
        "target_column": "label",
        "feature_columns": ["feature"],
    }
    values.update(overrides)
    return ClassificationPipelineConfig(**values)


def test_probability_adjustment_normalizes_rows_and_changes_decision():
    probabilities = np.array([[0.55, 0.45], [0.8, 0.2]])

    adjusted = apply_class_probability_multipliers(
        probabilities, ["majority", "minority"], {"minority": 1.5}
    )

    assert adjusted.sum(axis=1) == pytest.approx([1.0, 1.0])
    assert np.argmax(adjusted[0]) == 1
    assert np.argmax(adjusted[1]) == 0


def test_optimizer_improves_oof_macro_f1_without_exceeding_accuracy_guardrail():
    probabilities = np.array(
        [
            [0.90, 0.10],
            [0.80, 0.20],
            [0.70, 0.30],
            [0.55, 0.45],
            [0.55, 0.45],
        ]
    )
    y_true = np.array(["majority", "majority", "majority", "minority", "minority"])
    config = _config(
        probability_multiplier_grid=(1.0, 1.5),
        probability_optimization_iterations=1,
        probability_optimization_max_accuracy_drop=0.0,
    )

    multipliers, diagnostics = optimize_probability_multipliers(
        probabilities, y_true, ["majority", "minority"], config
    )

    assert multipliers["minority"] == 1.5
    assert diagnostics["optimized_oof_macro_f1"] > diagnostics["baseline_oof_macro_f1"]
    assert diagnostics["optimized_oof_accuracy"] >= diagnostics["baseline_oof_accuracy"]


def test_probability_optimization_configuration_rejects_invalid_values():
    with pytest.raises(ValueError, match="must all be > 0"):
        _config(probability_multiplier_grid=(1.0, 0.0))

    with pytest.raises(TypeError, match="must be a bool"):
        _config(optimize_class_probabilities="yes")
