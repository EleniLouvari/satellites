"""Learn frozen per-class reliability from training out-of-fold predictions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ml_classification.shared.class_reliability import CLASS_RELIABILITY_METHOD

# Keep this helper focused on a single transformation so the reporting pipeline stays easy to follow.



def _validate_oof_inputs(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    class_names: Sequence[str],
    minimum_support: int,
    high_min_precision: float,
    medium_min_precision: float,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Validate labels, class names, and threshold arguments without mutating them."""
    true = np.asarray(true_labels).astype(str)
    predicted = np.asarray(predicted_labels).astype(str)
    classes = [str(value) for value in class_names]

    if true.ndim != 1 or predicted.ndim != 1 or true.shape != predicted.shape:
        raise ValueError("Error: true_labels and predicted_labels must be aligned one-dimensional arrays.")
    if true.size == 0:
        raise ValueError("Error: OOF class reliability requires at least one prediction.")
    if len(classes) < 2 or len(set(classes)) != len(classes):
        raise ValueError("Error: class_names must contain at least two unique canonical classes.")

    unknown_predictions = sorted(set(predicted).difference(classes))
    unknown_true = sorted(set(true).difference(classes))
    if unknown_predictions or unknown_true:
        raise ValueError(
            f"Error: OOF labels are outside the canonical class set; predicted={unknown_predictions}, true={unknown_true}."
        )

    minimum_support = int(minimum_support)
    high_min_precision = float(high_min_precision)
    medium_min_precision = float(medium_min_precision)
    if minimum_support < 1:
        raise ValueError("Error: minimum_support must be at least 1.")
    if not 0.0 <= medium_min_precision <= high_min_precision <= 1.0:
        raise ValueError("Error: Precision thresholds must satisfy 0 <= MEDIUM <= HIGH <= 1.")
    return true, predicted, classes


def _class_reliability_record(
    class_name: str,
    true: np.ndarray,
    predicted: np.ndarray,
    minimum_support: int,
    high_min_precision: float,
    medium_min_precision: float,
) -> dict[str, Any]:
    """Build the reliability record for one canonical class."""
    # Precision conditions on the predicted class, so support counts predictions rather than true labels.
    selected = predicted == class_name
    support = int(np.sum(selected))
    correct = int(np.sum(true[selected] == class_name))
    precision = float(correct / support) if support else None

    # Insufficient support invalidates reliability even if the observed precision happens to be high.
    if support < minimum_support:
        return {
            "support": support,
            "correct": correct,
            "precision": precision,
            "level": "LOW",
            "valid": False,
            "reason": "insufficient_oof_class_support",
        }
    if precision is not None and precision >= high_min_precision:
        return {
            "support": support,
            "correct": correct,
            "precision": precision,
            "level": "HIGH",
            "valid": True,
            "reason": "sufficient_oof_support",
        }
    if precision is not None and precision >= medium_min_precision:
        return {
            "support": support,
            "correct": correct,
            "precision": precision,
            "level": "MEDIUM",
            "valid": True,
            "reason": "precision_below_high_threshold",
        }
    return {
        "support": support,
        "correct": correct,
        "precision": precision,
        "level": "LOW",
        "valid": True,
        "reason": "precision_below_medium_threshold",
    }


def calculate_oof_class_reliability(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    class_names: Sequence[str],
    minimum_support: int = 100,
    high_min_precision: float = 0.80,
    medium_min_precision: float = 0.60,
) -> dict[str, Any]:
    """Calculate one frozen OOF precision/reliability record per predicted class."""
    true, predicted, classes = _validate_oof_inputs(
        true_labels,
        predicted_labels,
        class_names,
        minimum_support,
        high_min_precision,
        medium_min_precision,
    )
    # Include every canonical class, including those with no OOF predictions.
    records: dict[str, dict[str, Any]] = {}
    for class_name in classes:
        records[class_name] = _class_reliability_record(
            class_name,
            true,
            predicted,
            int(minimum_support),
            float(high_min_precision),
            float(medium_min_precision),
        )
    return {
        "method": CLASS_RELIABILITY_METHOD,
        # Persist the evidence source alongside thresholds for later application without refitting.
        "source": "oof_soft_voting_precision",
        "version": "2.0",
        "minimum_oof_support": int(minimum_support),
        "high_min_precision": float(high_min_precision),
        "medium_min_precision": float(medium_min_precision),
        "oof_rows_total": int(true.size),
        "classes": records,
    }
