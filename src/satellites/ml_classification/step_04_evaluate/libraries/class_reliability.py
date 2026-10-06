"""Learn frozen per-class reliability from training out-of-fold predictions."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from satellites.ml_classification.shared.class_reliability import CLASS_RELIABILITY_METHOD


def calculate_oof_class_reliability(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    class_names: Sequence[str],
    minimum_support: int = 100,
    high_min_precision: float = 0.80,
    medium_min_precision: float = 0.60,
) -> dict[str, Any]:
    """Calculate one frozen OOF precision/reliability record per predicted class."""
    true = np.asarray(true_labels).astype(str)
    predicted = np.asarray(predicted_labels).astype(str)
    classes = [str(value) for value in class_names]
    if true.ndim != 1 or predicted.ndim != 1 or true.shape != predicted.shape:
        raise ValueError("true_labels and predicted_labels must be aligned one-dimensional arrays.")
    if true.size == 0:
        raise ValueError("OOF class reliability requires at least one prediction.")
    if len(classes) < 2 or len(set(classes)) != len(classes):
        raise ValueError("class_names must contain at least two unique canonical classes.")
    unknown_predictions = sorted(set(predicted).difference(classes))
    unknown_true = sorted(set(true).difference(classes))
    if unknown_predictions or unknown_true:
        raise ValueError(f"OOF labels are outside the canonical class set; predicted={unknown_predictions}, true={unknown_true}.")
    minimum_support = int(minimum_support)
    high_min_precision = float(high_min_precision)
    medium_min_precision = float(medium_min_precision)
    if minimum_support < 1:
        raise ValueError("minimum_support must be at least 1.")
    if not 0.0 <= medium_min_precision <= high_min_precision <= 1.0:
        raise ValueError("Precision thresholds must satisfy 0 <= MEDIUM <= HIGH <= 1.")

    records: dict[str, dict[str, Any]] = {}
    for class_name in classes:
        selected = predicted == class_name
        support = int(np.sum(selected))
        correct = int(np.sum(true[selected] == class_name))
        precision = float(correct / support) if support else None
        if support < minimum_support:
            level, valid, reason = "LOW", False, "insufficient_oof_class_support"
        elif precision is not None and precision >= high_min_precision:
            level, valid, reason = "HIGH", True, "sufficient_oof_support"
        elif precision is not None and precision >= medium_min_precision:
            level, valid, reason = "MEDIUM", True, "precision_below_high_threshold"
        else:
            level, valid, reason = "LOW", True, "precision_below_medium_threshold"
        records[class_name] = {
            "support": support,
            "correct": correct,
            "precision": precision,
            "level": level,
            "valid": valid,
            "reason": reason,
        }
    return {
        "method": CLASS_RELIABILITY_METHOD,
        "source": "oof_soft_voting_precision",
        "version": "2.0",
        "minimum_oof_support": minimum_support,
        "high_min_precision": high_min_precision,
        "medium_min_precision": medium_min_precision,
        "oof_rows_total": int(true.size),
        "classes": records,
    }
