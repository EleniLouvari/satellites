"""Apply frozen class reliability and combine it with rank-based confidence."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

CLASS_RELIABILITY_METHOD = "oof_precision"
CONFIDENCE_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}


def get_class_reliability(predicted_classes: np.ndarray, reliability_contract: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Apply frozen per-class reliability to arbitrary parcels without refitting."""
    if reliability_contract.get("method") != CLASS_RELIABILITY_METHOD:
        raise ValueError(f"Error: Unsupported class reliability method: {reliability_contract.get('method')!r}")
    predicted = np.asarray(predicted_classes).astype(str)
    if predicted.ndim != 1:
        raise ValueError("Error: predicted_classes must be one-dimensional.")
    records = reliability_contract.get("classes")
    if not isinstance(records, Mapping):
        raise ValueError("Error: Class reliability contract must contain a classes mapping.")  # noqa: TRY004 - contract validation

    # Unseen classes retain missing precision, zero support, and invalid LOW reliability.
    precision = np.full(predicted.shape, np.nan, dtype=np.float64)
    support = np.zeros(predicted.shape, dtype=int)
    correct = np.zeros(predicted.shape, dtype=int)
    levels = np.full(predicted.shape, "LOW", dtype=object)
    valid = np.zeros(predicted.shape, dtype=bool)
    reasons = np.full(predicted.shape, "missing_class_reliability", dtype=object)
    for index, class_name in enumerate(predicted):
        record = records.get(str(class_name))
        if not isinstance(record, Mapping):
            continue
        if record.get("precision") is not None:
            precision[index] = float(record["precision"])
        support[index] = int(record.get("support", 0))
        correct[index] = int(record.get("correct", 0))
        levels[index] = str(record.get("level", "LOW"))
        valid[index] = bool(record.get("valid", False))
        reasons[index] = str(record.get("reason", "missing_class_reliability"))
    if not set(levels).issubset(CONFIDENCE_ORDER):
        raise ValueError("Error: Class reliability contract contains an unknown confidence level.")
    return {
        "class_oof_precision": precision,
        "class_oof_support": support,
        "class_oof_correct": correct,
        "class_reliability_level": levels.astype(str),
        "class_reliability_valid": valid,
        "class_reliability_reason": reasons.astype(str),
    }


def combine_confidence_components(
    rank_confidence_level: np.ndarray,
    rank_confidence_valid: np.ndarray,
    rank_confidence_reason: np.ndarray,
    class_reliability_level: np.ndarray,
    class_reliability_valid: np.ndarray,
    class_reliability_reason: np.ndarray,
) -> dict[str, np.ndarray]:
    """Take the lower valid component and retain a concise explanation."""
    rank_level = np.asarray(rank_confidence_level).astype(str)
    rank_valid = np.asarray(rank_confidence_valid, dtype=bool)
    rank_reason = np.asarray(rank_confidence_reason).astype(str)
    class_level = np.asarray(class_reliability_level).astype(str)
    class_valid = np.asarray(class_reliability_valid, dtype=bool)
    class_reason = np.asarray(class_reliability_reason).astype(str)
    arrays = (rank_valid, rank_reason, class_level, class_valid, class_reason)
    if rank_level.ndim != 1 or any(array.shape != rank_level.shape for array in arrays):
        raise ValueError("Error: Rank and class reliability components must be aligned one-dimensional arrays.")
    if not set(rank_level).issubset(CONFIDENCE_ORDER) or not set(class_level).issubset(CONFIDENCE_ORDER):
        raise ValueError("Error: Confidence components contain an unknown level.")

    final_level = np.full(rank_level.shape, "LOW", dtype=object)
    # Both sources of evidence must be valid before a combined confidence level is trusted.
    final_valid = rank_valid & class_valid
    reasons = np.empty(rank_level.shape, dtype=object)
    for index in range(rank_level.size):
        # Invalid evidence keeps the default LOW level and preserves the reason for the failed component.
        if not rank_valid[index]:
            reasons[index] = rank_reason[index] or "invalid_rank_evidence"
            continue
        if not class_valid[index]:
            reasons[index] = class_reason[index] or "invalid_class_reliability"
            continue
        rank_value = CONFIDENCE_ORDER[rank_level[index]]
        class_value = CONFIDENCE_ORDER[class_level[index]]
        # The weaker component caps confidence even when the other component is HIGH.
        final_level[index] = min((rank_level[index], class_level[index]), key=CONFIDENCE_ORDER.get)
        if rank_value == 2 and class_value == 2:
            reasons[index] = "high_rank_consensus_and_high_class_reliability"
        elif rank_value <= class_value:
            reasons[index] = rank_reason[index] or f"rank_confidence_{rank_level[index].lower()}"
        else:
            reasons[index] = f"class_reliability_{class_level[index].lower()}"
    return {
        "prediction_confidence_level": final_level.astype(str),
        "prediction_confidence_valid": final_valid,
        "prediction_confidence_reason": reasons.astype(str),
    }
