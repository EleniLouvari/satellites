"""OOF predicted-class precision guard for rank-based confidence."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


CLASS_RELIABILITY_METHOD = "oof_precision"
CONFIDENCE_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}


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
        raise ValueError(
            f"OOF labels are outside the canonical class set; predicted={unknown_predictions}, true={unknown_true}."
        )
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


def get_class_reliability(
    predicted_classes: np.ndarray,
    reliability_contract: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Apply frozen per-class reliability to arbitrary parcels without refitting."""
    if reliability_contract.get("method") != CLASS_RELIABILITY_METHOD:
        raise ValueError(f"Unsupported class reliability method: {reliability_contract.get('method')!r}")
    predicted = np.asarray(predicted_classes).astype(str)
    if predicted.ndim != 1:
        raise ValueError("predicted_classes must be one-dimensional.")
    records = reliability_contract.get("classes")
    if not isinstance(records, Mapping):
        raise ValueError("Class reliability contract must contain a classes mapping.")

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
        raise ValueError("Class reliability contract contains an unknown confidence level.")
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
        raise ValueError("Rank and class reliability components must be aligned one-dimensional arrays.")
    if not set(rank_level).issubset(CONFIDENCE_ORDER) or not set(class_level).issubset(CONFIDENCE_ORDER):
        raise ValueError("Confidence components contain an unknown level.")

    final_level = np.full(rank_level.shape, "LOW", dtype=object)
    final_valid = rank_valid & class_valid
    reasons = np.empty(rank_level.shape, dtype=object)
    for index in range(rank_level.size):
        if not rank_valid[index]:
            reasons[index] = rank_reason[index] or "invalid_rank_evidence"
            continue
        if not class_valid[index]:
            reasons[index] = class_reason[index] or "invalid_class_reliability"
            continue
        rank_value = CONFIDENCE_ORDER[rank_level[index]]
        class_value = CONFIDENCE_ORDER[class_level[index]]
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
