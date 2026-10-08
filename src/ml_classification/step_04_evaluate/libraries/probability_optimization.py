"""Learn class probability adjustments from training out-of-fold predictions."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from ml_classification.shared.config.config import ClassificationPipelineConfig
from ml_classification.shared.probabilities import apply_class_probability_multipliers


def optimize_probability_multipliers(
    probabilities: np.ndarray, y_true: np.ndarray, labels: list[str], config: ClassificationPipelineConfig
) -> tuple[dict[str, float], dict[str, float]]:
    """Learn class multipliers from out-of-fold predictions using coordinate search."""
    # Normalize inputs and compute baseline metrics used as optimization anchors.
    y_true = np.asarray(y_true).astype(str)
    label_array = np.asarray(labels).astype(str)
    baseline_pred = label_array[np.argmax(probabilities, axis=1)]
    baseline_f1 = float(f1_score(y_true, baseline_pred, average="macro", zero_division=0))
    baseline_accuracy = float(accuracy_score(y_true, baseline_pred))
    minimum_accuracy = baseline_accuracy - float(config.probability_optimization_max_accuracy_drop)
    multipliers = {str(label): 1.0 for label in labels}
    best_f1 = baseline_f1
    best_accuracy = baseline_accuracy
    for _ in range(config.probability_optimization_iterations):
        changed = False
        for label in labels:
            current = multipliers[str(label)]
            local_best = (best_f1, best_accuracy, current)
            for candidate in config.probability_multiplier_grid:
                trial = dict(multipliers)
                trial[str(label)] = float(candidate)
                adjusted = apply_class_probability_multipliers(probabilities, labels, trial)
                predicted = label_array[np.argmax(adjusted, axis=1)]
                accuracy = float(accuracy_score(y_true, predicted))
                score = float(f1_score(y_true, predicted, average="macro", zero_division=0))
                if accuracy >= minimum_accuracy and (score, accuracy) > local_best[:2]:
                    local_best = (score, accuracy, float(candidate))
            if local_best[2] != current:
                multipliers[str(label)] = local_best[2]
                best_f1, best_accuracy = local_best[:2]
                changed = True
        if not changed:
            break
    return multipliers, {
        "baseline_oof_macro_f1": baseline_f1,
        "optimized_oof_macro_f1": best_f1,
        "baseline_oof_accuracy": baseline_accuracy,
        "optimized_oof_accuracy": best_accuracy,
    }
