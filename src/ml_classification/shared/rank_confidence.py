"""Rank-only support diagnostics for a soft-voting ensemble prediction.

Soft voting remains responsible for choosing the predicted class. This module
uses raw selected-member probability ordering only to measure how strongly
those members support that already-selected class.

Rank agreement needs several voting models. With fewer than the configured
minimum, the level is graded from the mean member probability of the predicted
class instead, so a one- or two-model strategy still gets a valid confidence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.stats import rankdata

DEFAULT_RANK_CONFIDENCE_THRESHOLDS: dict[str, float] = {
    "high_min_borda": 90.0,
    "high_max_range": 2.0,
    "medium_min_borda": 75.0,
    "medium_max_range": 4.0,
}
# Probability limits used only when there are too few voting models for rank agreement.
DEFAULT_PROBABILITY_FALLBACK_THRESHOLDS: dict[str, float] = {
    "fallback_high_min_probability": 0.80,
    "fallback_medium_min_probability": 0.60,
}
PROBABILITY_FALLBACK_SOURCE = "probability_fallback"
RANK_ONLY_SOURCE = "rank_only"


def rank_confidence_thresholds_from_config(config: Any) -> dict[str, float]:
    """Return the simplified rank thresholds in frozen-contract form."""
    return {
        "high_min_borda": float(getattr(config, "rank_confidence_high_min_borda", 90.0)),
        "high_max_range": float(getattr(config, "rank_confidence_high_max_range", 2.0)),
        "medium_min_borda": float(getattr(config, "rank_confidence_medium_min_borda", 75.0)),
        "medium_max_range": float(getattr(config, "rank_confidence_medium_max_range", 4.0)),
        # The fallback reuses the probability review cut-off as its MEDIUM limit.
        "fallback_high_min_probability": float(getattr(config, "prediction_high_confidence_threshold", 0.80)),
        "fallback_medium_min_probability": float(getattr(config, "prediction_confidence_threshold", 0.60)),
    }


def _normalize_thresholds(thresholds: Mapping[str, Any] | None) -> dict[str, float]:
    configured = dict(DEFAULT_RANK_CONFIDENCE_THRESHOLDS if thresholds is None else thresholds)
    required = set(DEFAULT_RANK_CONFIDENCE_THRESHOLDS)
    missing = sorted(required.difference(configured))
    if missing:
        raise ValueError(f"Error: Rank-confidence thresholds are missing fields: {missing}")
    values = {name: float(configured[name]) for name in required}
    if not np.isfinite(list(values.values())).all():
        raise ValueError("Error: Rank-confidence thresholds must be finite.")
    if not 0.0 <= values["medium_min_borda"] <= values["high_min_borda"] <= 100.0:
        raise ValueError("Error: Borda thresholds must satisfy 0 <= MEDIUM <= HIGH <= 100.")
    if values["high_max_range"] < 0 or values["medium_max_range"] < values["high_max_range"]:
        raise ValueError("Error: Rank-range thresholds must satisfy 0 <= HIGH <= MEDIUM.")
    return values


def _normalize_fallback_thresholds(thresholds: Mapping[str, Any] | None) -> dict[str, float]:
    """Return the probability-fallback limits; contracts saved before they existed use the defaults."""
    configured = {} if thresholds is None else thresholds
    values = {
        name: float(configured.get(name, default)) for name, default in DEFAULT_PROBABILITY_FALLBACK_THRESHOLDS.items()
    }
    medium, high = values["fallback_medium_min_probability"], values["fallback_high_min_probability"]
    if not (np.isfinite(medium) and np.isfinite(high) and 0.0 <= medium <= high <= 1.0):
        raise ValueError("Error: Probability-fallback thresholds must satisfy 0 <= MEDIUM <= HIGH <= 1.")
    return values


def predicted_class_probability(probabilities: np.ndarray, predicted_class_indices: np.ndarray) -> np.ndarray:
    """Return the mean member probability of each row's predicted class.

    Every model row is normalized to sum to one first, so members that return unnormalized scores
    carry the same weight as the others.
    """
    values = np.asarray(probabilities, dtype=np.float64)
    normalized = values / values.sum(axis=2, keepdims=True)
    rows = np.arange(values.shape[1])
    return np.mean(normalized[:, rows, np.asarray(predicted_class_indices, dtype=int)], axis=0)


def assign_probability_fallback_levels(
    support: np.ndarray, thresholds: Mapping[str, Any] | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Grade confidence from predicted-class probability when rank agreement cannot be measured."""
    configured = _normalize_fallback_thresholds(thresholds)
    values = np.asarray(support, dtype=np.float64)
    valid = np.isfinite(values)
    high = valid & (values >= configured["fallback_high_min_probability"])
    medium = valid & ~high & (values >= configured["fallback_medium_min_probability"])
    levels = np.full(values.shape, "LOW", dtype=object)
    levels[medium] = "MEDIUM"
    levels[high] = "HIGH"
    reasons = np.full(values.shape, "probability_fallback_low", dtype=object)
    reasons[medium] = "probability_fallback_medium"
    reasons[high] = "probability_fallback_high"
    reasons[~valid] = "invalid_probability_evidence"
    return levels.astype(str), valid, reasons.astype(str)


def validate_model_probabilities(
    model_probabilities: np.ndarray, class_names: Sequence[str], model_names: Sequence[str] | None = None
) -> np.ndarray:
    """Validate and return a float tensor shaped ``[models, rows, classes]``."""
    values = np.asarray(model_probabilities, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("Error: model_probabilities must have shape (n_models, n_rows, n_classes).")
    n_models, n_rows, n_classes = values.shape
    if n_models < 1 or n_rows < 1:
        raise ValueError("Error: model_probabilities must contain at least one model and one row.")
    if n_classes < 2:
        raise ValueError("Error: Rank confidence requires at least two canonical classes.")
    if n_classes != len(class_names):
        raise ValueError(
            f"Error: Probability class dimension does not match the canonical class list: {n_classes} != {len(class_names)}."
        )
    if len(set(map(str, class_names))) != len(class_names):
        raise ValueError("Error: Canonical class names must be unique.")
    if model_names is not None:
        if len(model_names) != n_models:
            raise ValueError("Error: model_names length must equal the model probability dimension.")
        if len(set(map(str, model_names))) != len(model_names):
            raise ValueError("Error: model_names must be unique.")
    invalid = ~np.isfinite(values)
    if invalid.any():
        invalid_models = np.unique(np.where(invalid)[0]).tolist()
        names = [str(model_names[index]) for index in invalid_models] if model_names is not None else invalid_models
        raise ValueError(f"Error: Non-finite model probabilities found for models: {names}.")
    if np.any(values < 0):
        raise ValueError("Error: Model probabilities must be non-negative.")
    if np.any(values.sum(axis=2) <= 0):
        raise ValueError("Error: Each model probability row must contain positive total support.")
    return values


def probabilities_to_ranks(probabilities: np.ndarray) -> np.ndarray:
    """Convert class scores to descending average ranks along the final axis."""
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] < 2:
        raise ValueError("Error: probabilities must include at least two classes.")
    if not np.isfinite(values).all():
        raise ValueError("Error: probabilities contain non-finite values.")
    return np.asarray(rankdata(-values, method="average", axis=-1), dtype=np.float64)


def ranks_to_scores(ranks: np.ndarray, n_classes: int | None = None) -> np.ndarray:
    """Convert ranks to normalized Borda scores in the interval [0, 100]."""
    values = np.asarray(ranks, dtype=np.float64)
    if values.ndim < 1:
        raise ValueError("Error: ranks must have at least one dimension.")
    class_count = int(n_classes if n_classes is not None else values.shape[-1])
    if class_count < 2 or values.shape[-1] != class_count:
        raise ValueError("Error: n_classes must match the final rank dimension and be at least two.")
    if not np.isfinite(values).all() or np.any((values < 1.0) | (values > class_count)):
        raise ValueError("Error: ranks must be finite and between 1 and n_classes.")
    return np.clip(100.0 * (class_count - values) / (class_count - 1), 0.0, 100.0)


def aggregate_rank_scores(model_ranks: np.ndarray, model_weights: np.ndarray | None = None) -> np.ndarray:
    """Return mean Borda scores across models; production uses equal weights."""
    ranks = np.asarray(model_ranks, dtype=np.float64)
    if ranks.ndim != 3:
        raise ValueError("Error: model_ranks must have shape (n_models, n_rows, n_classes).")
    scores = ranks_to_scores(ranks, ranks.shape[-1])
    if model_weights is None:
        return np.mean(scores, axis=0, dtype=np.float64)
    weights = np.asarray(model_weights, dtype=np.float64)
    if weights.ndim != 1 or len(weights) != ranks.shape[0]:
        raise ValueError("Error: model_weights must contain one value per model.")
    if not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Error: model_weights must be finite, non-negative, and sum to more than zero.")
    return np.average(scores, axis=0, weights=weights)


def calculate_rank_confidence_features(
    model_ranks: np.ndarray, aggregate_scores: np.ndarray, predicted_class_indices: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    """Calculate rank diagnostics for the actual final soft-voting class."""
    ranks = np.asarray(model_ranks, dtype=np.float64)
    scores = np.asarray(aggregate_scores, dtype=np.float64)
    if ranks.ndim != 3 or scores.shape != ranks.shape[1:]:
        raise ValueError("Error: Ranks and aggregate scores have incompatible shapes.")
    n_models, n_rows, n_classes = ranks.shape
    borda_winner_indices = np.argmax(scores, axis=1)
    maximum_scores = np.max(scores, axis=1, keepdims=True)
    borda_winner_tied = np.sum(np.isclose(scores, maximum_scores, rtol=1e-12, atol=1e-10), axis=1) > 1
    if predicted_class_indices is None:
        predicted_indices = borda_winner_indices
    else:
        predicted_indices = np.asarray(predicted_class_indices, dtype=int)
        if predicted_indices.shape != (n_rows,):
            raise ValueError("Error: predicted_class_indices must contain one index per row.")
        if np.any((predicted_indices < 0) | (predicted_indices >= n_classes)):
            raise ValueError("Error: predicted_class_indices contains an invalid class index.")

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
        raise ValueError("Error: minimum_models must be at least 1.")
    score_value = features.get("prediction_mean_borda")
    range_value = features.get("prediction_rank_range")
    agrees_value = features.get("rank_agrees_with_prediction")
    tied_value = features.get("rank_winner_tied")
    if score_value is None or range_value is None or agrees_value is None or tied_value is None:
        raise ValueError("Error: Rank-confidence features are incomplete.")
    score = np.asarray(score_value, dtype=np.float64)
    rank_range = np.asarray(range_value, dtype=np.float64)
    models = np.asarray(features["n_models_used"], dtype=int)
    agrees = np.asarray(agrees_value, dtype=bool)
    tied = np.asarray(tied_value, dtype=bool)
    arrays = (score, rank_range, models, agrees, tied)
    if any(array.shape != score.shape for array in arrays) or score.ndim != 1:
        raise ValueError("Error: Rank-confidence features must be aligned one-dimensional arrays.")
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
    support = predicted_class_probability(probabilities, features["predicted_class_indices"])
    # Rank agreement between one or two models says little; grade those strategies from probability instead.
    use_fallback = probabilities.shape[0] < int(minimum_models)
    if use_fallback:
        levels, valid, reasons = assign_probability_fallback_levels(support, thresholds)
    else:
        levels, valid, reasons = assign_confidence_levels(features, thresholds, minimum_models)
    labels = np.asarray(class_names, dtype=str)
    return {
        **features,
        "borda_prediction": labels[features["borda_winner_indices"]],
        "rank_confidence_level": levels,
        "rank_confidence_valid": valid,
        "rank_confidence_reason": reasons,
        "rank_confidence_source": PROBABILITY_FALLBACK_SOURCE if use_fallback else RANK_ONLY_SOURCE,
        "prediction_mean_probability": support,
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
            raise ValueError("Error: Simplified rank confidence requires equal model weights.")
    probabilities = validate_model_probabilities(model_probabilities, class_names, model_names)
    return _calculate_validated_rank_confidence(
        probabilities,
        None if predicted_class_indices is None else np.asarray(predicted_class_indices, dtype=int),
        class_names,
        confidence_thresholds,
        minimum_models,
    )
