"""Apply frozen predicted-class reliability independently of rank confidence."""

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
