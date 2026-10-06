"""Rank-only support diagnostics for a soft-voting ensemble prediction.

Soft voting remains responsible for choosing the predicted class. This module
uses raw selected-member probability ordering only to measure how strongly
those members support that already-selected class.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import rankdata

DEFAULT_RANK_CONFIDENCE_THRESHOLDS: dict[str, float] = {
    "high_min_borda": 90.0,
    "high_max_range": 2.0,
    "medium_min_borda": 75.0,
    "medium_max_range": 4.0,
}


def rank_confidence_thresholds_from_config(config: Any) -> dict[str, float]:
    """Return the simplified rank thresholds in frozen-contract form."""
    return {
        "high_min_borda": float(getattr(config, "rank_confidence_high_min_borda", 90.0)),
        "high_max_range": float(getattr(config, "rank_confidence_high_max_range", 2.0)),
        "medium_min_borda": float(getattr(config, "rank_confidence_medium_min_borda", 75.0)),
        "medium_max_range": float(getattr(config, "rank_confidence_medium_max_range", 4.0)),
    }


def _normalize_thresholds(thresholds: Mapping[str, Any] | None) -> dict[str, float]:
    configured = dict(DEFAULT_RANK_CONFIDENCE_THRESHOLDS if thresholds is None else thresholds)
    required = set(DEFAULT_RANK_CONFIDENCE_THRESHOLDS)
    missing = sorted(required.difference(configured))
    if missing:
        raise ValueError(f"Rank-confidence thresholds are missing fields: {missing}")
    values = {name: float(configured[name]) for name in required}
    if not np.isfinite(list(values.values())).all():
        raise ValueError("Rank-confidence thresholds must be finite.")
    if not 0.0 <= values["medium_min_borda"] <= values["high_min_borda"] <= 100.0:
        raise ValueError("Borda thresholds must satisfy 0 <= MEDIUM <= HIGH <= 100.")
    if values["high_max_range"] < 0 or values["medium_max_range"] < values["high_max_range"]:
        raise ValueError("Rank-range thresholds must satisfy 0 <= HIGH <= MEDIUM.")
    return values


def validate_model_probabilities(
    model_probabilities: np.ndarray, class_names: Sequence[str], model_names: Sequence[str] | None = None
) -> np.ndarray:
    """Validate and return a float tensor shaped ``[models, rows, classes]``."""
    values = np.asarray(model_probabilities, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("model_probabilities must have shape (n_models, n_rows, n_classes).")
    n_models, n_rows, n_classes = values.shape
    if n_models < 1 or n_rows < 1:
        raise ValueError("model_probabilities must contain at least one model and one row.")
    if n_classes < 2:
        raise ValueError("Rank confidence requires at least two canonical classes.")
    if n_classes != len(class_names):
        raise ValueError(
            f"Probability class dimension does not match the canonical class list: {n_classes} != {len(class_names)}."
        )
    if len(set(map(str, class_names))) != len(class_names):
        raise ValueError("Canonical class names must be unique.")
    if model_names is not None:
        if len(model_names) != n_models:
            raise ValueError("model_names length must equal the model probability dimension.")
        if len(set(map(str, model_names))) != len(model_names):
            raise ValueError("model_names must be unique.")
    invalid = ~np.isfinite(values)
    if invalid.any():
        invalid_models = np.unique(np.where(invalid)[0]).tolist()
        names = [str(model_names[index]) for index in invalid_models] if model_names is not None else invalid_models
        raise ValueError(f"Non-finite model probabilities found for models: {names}.")
    if np.any(values < 0):
        raise ValueError("Model probabilities must be non-negative.")
    if np.any(values.sum(axis=2) <= 0):
        raise ValueError("Each model probability row must contain positive total support.")
    return values


def probabilities_to_ranks(probabilities: np.ndarray) -> np.ndarray:
    """Convert class scores to descending average ranks along the final axis."""
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] < 2:
        raise ValueError("probabilities must include at least two classes.")
    if not np.isfinite(values).all():
        raise ValueError("probabilities contain non-finite values.")
    return np.asarray(rankdata(-values, method="average", axis=-1), dtype=np.float64)


def ranks_to_scores(ranks: np.ndarray, n_classes: int | None = None) -> np.ndarray:
    """Convert ranks to normalized Borda scores in the interval [0, 100]."""
    values = np.asarray(ranks, dtype=np.float64)
    if values.ndim < 1:
        raise ValueError("ranks must have at least one dimension.")
    class_count = int(n_classes if n_classes is not None else values.shape[-1])
    if class_count < 2 or values.shape[-1] != class_count:
        raise ValueError("n_classes must match the final rank dimension and be at least two.")
    if not np.isfinite(values).all() or np.any((values < 1.0) | (values > class_count)):
        raise ValueError("ranks must be finite and between 1 and n_classes.")
    return np.clip(100.0 * (class_count - values) / (class_count - 1), 0.0, 100.0)


def aggregate_rank_scores(model_ranks: np.ndarray, model_weights: np.ndarray | None = None) -> np.ndarray:
    """Return mean Borda scores across models; production uses equal weights."""
    ranks = np.asarray(model_ranks, dtype=np.float64)
    if ranks.ndim != 3:
        raise ValueError("model_ranks must have shape (n_models, n_rows, n_classes).")
    scores = ranks_to_scores(ranks, ranks.shape[-1])
    if model_weights is None:
        return np.mean(scores, axis=0, dtype=np.float64)
    weights = np.asarray(model_weights, dtype=np.float64)
    if weights.ndim != 1 or len(weights) != ranks.shape[0]:
        raise ValueError("model_weights must contain one value per model.")
    if not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("model_weights must be finite, non-negative, and sum to more than zero.")
    return np.average(scores, axis=0, weights=weights)


def calculate_rank_confidence_features(
    model_ranks: np.ndarray, aggregate_scores: np.ndarray, predicted_class_indices: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    """Calculate rank diagnostics for the actual final soft-voting class."""
    ranks = np.asarray(model_ranks, dtype=np.float64)
    scores = np.asarray(aggregate_scores, dtype=np.float64)
    if ranks.ndim != 3 or scores.shape != ranks.shape[1:]:
        raise ValueError("Ranks and aggregate scores have incompatible shapes.")
    n_models, n_rows, n_classes = ranks.shape
    borda_winner_indices = np.argmax(scores, axis=1)
    maximum_scores = np.max(scores, axis=1, keepdims=True)
    borda_winner_tied = np.sum(np.isclose(scores, maximum_scores, rtol=1e-12, atol=1e-10), axis=1) > 1
    if predicted_class_indices is None:
        predicted_indices = borda_winner_indices
    else:
        predicted_indices = np.asarray(predicted_class_indices, dtype=int)
        if predicted_indices.shape != (n_rows,):
            raise ValueError("predicted_class_indices must contain one index per row.")
        if np.any((predicted_indices < 0) | (predicted_indices >= n_classes)):
            raise ValueError("predicted_class_indices contains an invalid class index.")

    row_indices = np.arange(n_rows)
    predicted_ranks = ranks[:, row_indices, predicted_indices]
    predicted_borda = scores[row_indices, predicted_indices]
    competing_scores = scores.copy()
    competing_scores[row_indices, predicted_indices] = -np.inf
    runner_up_indices = np.argmax(competing_scores, axis=1)
    runner_up_scores = scores[row_indices, runner_up_indices]
    rank_range = np.max(predicted_ranks, axis=0) - np.min(predicted_ranks, axis=0)
    agrees = borda_winner_indices == predicted_indices
    mean_rank = np.mean(predicted_ranks, axis=0)
    median_rank = np.median(predicted_ranks, axis=0)
    rank_std = np.std(predicted_ranks, axis=0, ddof=0)
    rank_iqr = np.percentile(predicted_ranks, 75, axis=0) - np.percentile(predicted_ranks, 25, axis=0)

    return {
        "predicted_class_indices": predicted_indices,
        "borda_winner_indices": borda_winner_indices,
        "runner_up_indices": runner_up_indices,  # This is the second-place class by Borda score.
        "prediction_mean_borda": predicted_borda,  # This is the Borda scores for the predicted class.
        "prediction_mean_rank": mean_rank,  # This is the mean rank for the predicted class.
        "prediction_median_rank": median_rank,  # This is the median rank for the predicted class.
        "prediction_rank_range": rank_range,  # This is the range of ranks for the predicted class.
        "prediction_rank_std": rank_std,  # This is the standard deviation of ranks for the predicted class.
        "prediction_rank_iqr": rank_iqr,  # This is the interquartile range of ranks for the predicted class.
        "n_models_used": np.full(n_rows, n_models, dtype=int),
        # Additional diagnostics. These do not determine H/M/L.
        "runner_up_borda": runner_up_scores,
        "borda_margin": predicted_borda - runner_up_scores,
        "top1_agreement": np.mean(predicted_ranks == 1.0, axis=0),
        "top2_agreement": np.mean(predicted_ranks <= 2.0, axis=0),
        "top3_agreement": np.mean(predicted_ranks <= 3.0, axis=0),
        "rank_winner_tied": borda_winner_tied,  # This is True if there are multiple classes tied for the highest Borda score.
        "rank_agrees_with_prediction": agrees,
    }


def assign_confidence_levels(
    features: Mapping[str, np.ndarray], thresholds: Mapping[str, Any] | None = None, minimum_models: int = 3
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assign simplified parcel rank confidence from Borda, range, and winner."""
    configured = _normalize_thresholds(thresholds)
    if int(minimum_models) < 1:
        raise ValueError("minimum_models must be at least 1.")
    score_value = features.get("prediction_mean_borda")
    range_value = features.get("prediction_rank_range")
    agrees_value = features.get("rank_agrees_with_prediction")
    tied_value = features.get("rank_winner_tied")
    if score_value is None or range_value is None or agrees_value is None or tied_value is None:
        raise ValueError("Rank-confidence features are incomplete.")
    score = np.asarray(score_value, dtype=np.float64)
    rank_range = np.asarray(range_value, dtype=np.float64)
    models = np.asarray(features["n_models_used"], dtype=int)
    agrees = np.asarray(agrees_value, dtype=bool)
    tied = np.asarray(tied_value, dtype=bool)
    arrays = (score, rank_range, models, agrees, tied)
    if any(array.shape != score.shape for array in arrays) or score.ndim != 1:
        raise ValueError("Rank-confidence features must be aligned one-dimensional arrays.")
    finite = np.isfinite(score) & np.isfinite(rank_range)
    valid = finite & (models >= int(minimum_models))
    eligible = valid & agrees & ~tied
    high = eligible & (score >= configured["high_min_borda"]) & (rank_range <= configured["high_max_range"])
    medium = eligible & ~high & (score >= configured["medium_min_borda"]) & (rank_range <= configured["medium_max_range"])
    levels = np.full(score.shape, "LOW", dtype=object)
    levels[medium] = "MEDIUM"
    levels[high] = "HIGH"
    reasons = np.full(score.shape, "rank_confidence_low", dtype=object)
    reasons[medium] = "rank_confidence_medium"
    reasons[high] = "rank_confidence_high"
    reasons[valid & tied] = "borda_winner_tie"
    reasons[valid & ~tied & ~agrees] = "soft_vote_borda_disagreement"
    reasons[~finite] = "invalid_rank_evidence"
    reasons[models < int(minimum_models)] = "insufficient_models"
    return levels.astype(str), valid, reasons.astype(str)


def _calculate_validated_rank_confidence(
    probabilities: np.ndarray,
    predicted_class_indices: np.ndarray | None,
    class_names: Sequence[str],
    thresholds: Mapping[str, Any] | None,
    minimum_models: int,
) -> dict[str, Any]:
    """Calculate rank confidence from an already validated probability tensor."""
    ranks = probabilities_to_ranks(probabilities)
    aggregate_scores = aggregate_rank_scores(ranks)
    if predicted_class_indices is None:
        predicted_class_indices = np.argmax(aggregate_scores, axis=1)
    features = calculate_rank_confidence_features(ranks, aggregate_scores, predicted_class_indices)
    levels, valid, reasons = assign_confidence_levels(features, thresholds, minimum_models)
    labels = np.asarray(class_names, dtype=str)
    return {
        **features,
        "borda_prediction": labels[features["borda_winner_indices"]],
        "rank_confidence_level": levels,
        "rank_confidence_valid": valid,
        "rank_confidence_reason": reasons,
        "runner_up_class": labels[features["runner_up_indices"]],
    }


def calculate_rank_confidence(
    model_probabilities: np.ndarray,
    predicted_class_indices: np.ndarray,
    class_names: Sequence[str],
    thresholds: Mapping[str, Any] | None = None,
    minimum_models: int = 3,
    model_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Calculate simplified rank confidence for final soft-voting predictions."""
    probabilities = validate_model_probabilities(model_probabilities, class_names, model_names)
    return _calculate_validated_rank_confidence(probabilities, predicted_class_indices, class_names, thresholds, minimum_models)


def classify_ensemble_rank_based(
    model_probabilities: np.ndarray,
    class_names: Sequence[str],
    model_names: Sequence[str] | None = None,
    predicted_class_indices: np.ndarray | None = None,
    model_weights: np.ndarray | None = None,
    confidence_thresholds: Mapping[str, Any] | None = None,
    minimum_models: int = 3,
) -> dict[str, Any]:
    """Compatibility entry point for rank aggregation and simplified confidence."""
    if model_weights is not None:
        weights = np.asarray(model_weights, dtype=np.float64)
        if not np.allclose(weights, np.full(weights.shape, weights.mean())):
            raise ValueError("Simplified rank confidence requires equal model weights.")
    probabilities = validate_model_probabilities(model_probabilities, class_names, model_names)
    return _calculate_validated_rank_confidence(
        probabilities,
        None if predicted_class_indices is None else np.asarray(predicted_class_indices, dtype=int),
        class_names,
        confidence_thresholds,
        minimum_models,
    )
