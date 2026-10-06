"""Apply persisted class multipliers to model probabilities across steps."""

from __future__ import annotations

import numpy as np


def apply_class_probability_multipliers(
    probabilities: np.ndarray, labels: list[str], multipliers: dict[str, float] | None
) -> np.ndarray:
    """Apply persisted class multipliers and return normalized probabilities."""
    # Copy into a floating array and apply per-class multiplicative weights
    # followed by row-wise normalization so probabilities remain valid.
    adjusted = np.asarray(probabilities, dtype=np.float64).copy()
    if not multipliers:
        return adjusted
    weights = np.asarray([float(multipliers.get(str(label), 1.0)) for label in labels])
    adjusted *= weights
    row_sums = adjusted.sum(axis=1, keepdims=True)
    # Use numpy.divide with `where` to avoid division-by-zero for degenerate rows.
    return np.divide(adjusted, row_sums, out=np.zeros_like(adjusted), where=row_sums != 0)
