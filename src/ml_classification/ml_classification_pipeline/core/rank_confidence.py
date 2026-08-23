"""Rank-based ensemble prediction confidence calculations.

The functions in this module compare class orderings within each model rather
than comparing probability magnitudes across heterogeneous model families.
All array-oriented functions operate on the final class axis.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import rankdata


# Keep provisional global rules as a documented fallback when OOF calibration is unavailable.
DEFAULT_RANK_CONFIDENCE_THRESHOLDS: dict[str, dict[str, float]] = {
    # HIGH requires strong agreement, aggregate rank support, and runner-up separation.
    "high": {"min_top1_agreement": 0.75, "min_top2_agreement": 0.80, "min_rank_score": 0.80, "min_rank_margin": 0.20},
    # MEDIUM uses weaker versions of the same four conjunctive requirements.
    "medium": {"min_top1_agreement": 0.50, "min_top2_agreement": 0.60, "min_rank_score": 0.60, "min_rank_margin": 0.10},
}


def rank_confidence_thresholds_from_config(config: Any) -> dict[str, dict[str, float]]:
    """Return the configured HIGH and MEDIUM rank-confidence thresholds."""
    # Convert every configured value to float so the frozen JSON contract is type-stable.
    return {
        # Read HIGH settings, retaining documented defaults for legacy configuration objects.
        "high": {
            "min_top1_agreement": float(getattr(config, "rank_confidence_high_min_top1", 0.75)),
            "min_top2_agreement": float(getattr(config, "rank_confidence_high_min_top2", 0.80)),
            "min_rank_score": float(getattr(config, "rank_confidence_high_min_score", 0.80)),
            "min_rank_margin": float(getattr(config, "rank_confidence_high_min_margin", 0.20)),
        },
        # Read the corresponding MEDIUM settings with the same compatibility behavior.
        "medium": {
            "min_top1_agreement": float(getattr(config, "rank_confidence_medium_min_top1", 0.50)),
            "min_top2_agreement": float(getattr(config, "rank_confidence_medium_min_top2", 0.60)),
            "min_rank_score": float(getattr(config, "rank_confidence_medium_min_score", 0.60)),
            "min_rank_margin": float(getattr(config, "rank_confidence_medium_min_margin", 0.10)),
        },
    }


def validate_model_probabilities(
    model_probabilities: np.ndarray, class_names: Sequence[str], model_names: Sequence[str] | None = None
) -> np.ndarray:
    """Validate and return a float tensor shaped ``[models, rows, classes]``."""
    # Normalize all model outputs to a common floating-point tensor without changing order.
    values = np.asarray(model_probabilities, dtype=np.float64)
    # Require the explicit [model, parcel, canonical class] axis contract.
    if values.ndim != 3:
        raise ValueError("model_probabilities must have shape (n_models, n_rows, n_classes).")
    # Name each axis once so all subsequent validation remains readable.
    n_models, n_rows, n_classes = values.shape
    # Ranking is undefined when there is no model or no parcel row.
    if n_models < 1 or n_rows < 1:
        raise ValueError("model_probabilities must contain at least one model and one row.")
    # At least two classes are required to define a winner and runner-up.
    if n_classes < 2:
        raise ValueError("Rank confidence requires at least two canonical classes.")
    # Protect against silently assigning a probability column to the wrong class label.
    if n_classes != len(class_names):
        raise ValueError(
            f"Probability class dimension does not match the canonical class list: {n_classes} != {len(class_names)}."
        )
    # Duplicate canonical labels would make class-index lookup ambiguous.
    if len(set(map(str, class_names))) != len(class_names):
        raise ValueError("Canonical class names must be unique.")
    # Optional model names must identify every tensor slice exactly once.
    if model_names is not None and len(model_names) != n_models:
        raise ValueError("model_names length must equal the model probability dimension.")
    # Locate NaN and infinite values before ranking can propagate invalid results.
    invalid = ~np.isfinite(values)
    if invalid.any():
        # Identify the affected model indices to make the exception operationally useful.
        invalid_models = np.unique(np.where(invalid)[0]).tolist()
        # Prefer caller-provided model names; otherwise report numeric tensor indices.
        names = [str(model_names[index]) for index in invalid_models] if model_names is not None else invalid_models
        raise ValueError(f"Non-finite model probabilities found for models: {names}.")
    # Negative probabilities/scores violate the expected classifier-output contract.
    if np.any(values < 0):
        raise ValueError("Model probabilities must be non-negative.")
    # Reject all-zero class rows because they contain no usable ranking support.
    if np.any(values.sum(axis=2) <= 0):
        raise ValueError("Each model probability row must contain positive total support.")
    # Return the normalized tensor after every alignment and numeric check succeeds.
    return values


def probabilities_to_ranks(probabilities: np.ndarray) -> np.ndarray:
    """Convert class scores to descending average ranks along the final axis."""
    # Convert lists or lower-precision arrays to a consistent numeric representation.
    values = np.asarray(probabilities, dtype=np.float64)
    # The final axis must contain at least two candidate classes to rank.
    if values.ndim < 1 or values.shape[-1] < 2:
        raise ValueError("probabilities must include at least two classes.")
    # Reject NaN/infinite inputs so SciPy never returns misleading partial ranks.
    if not np.isfinite(values).all():
        raise ValueError("probabilities contain non-finite values.")
    # Negate scores so the largest probability gets rank 1; average preserves ties fairly.
    return np.asarray(rankdata(-values, method="average", axis=-1), dtype=np.float64)


def ranks_to_scores(ranks: np.ndarray, n_classes: int | None = None) -> np.ndarray:
    """Convert ranks to normalized Borda-style scores in the interval [0, 1]."""
    # Normalize ranks before applying the Borda transformation on the final class axis.
    values = np.asarray(ranks, dtype=np.float64)
    # Infer class count from the array unless the caller explicitly supplies it.
    class_count = int(n_classes if n_classes is not None else values.shape[-1])
    # Require the denominator and final axis to describe the same canonical class set.
    if class_count < 2 or values.shape[-1] != class_count:
        raise ValueError("n_classes must match the final rank dimension and be at least two.")
    # Map rank 1 to 1.0 and the last rank to 0.0 using normalized Borda scoring.
    scores = (class_count - values) / (class_count - 1)
    # Clip tiny floating-point excursions while preserving all valid intermediate scores.
    return np.clip(scores, 0.0, 1.0)


def aggregate_rank_scores(model_ranks: np.ndarray, model_weights: np.ndarray | None = None) -> np.ndarray:
    """Aggregate model rank scores, using equal weights by default."""
    # Normalize input to the required [models, rows, classes] floating-point tensor.
    ranks = np.asarray(model_ranks, dtype=np.float64)
    # Refuse ambiguous arrays whose model and parcel axes cannot be identified.
    if ranks.ndim != 3:
        raise ValueError("model_ranks must have shape (n_models, n_rows, n_classes).")
    # Convert each within-model rank into a comparable normalized Borda score.
    scores = ranks_to_scores(ranks, ranks.shape[-1])
    # Equal weighting is the default methodology and simply averages across models.
    if model_weights is None:
        return np.mean(scores, axis=0, dtype=np.float64)
    # Normalize optional validation-derived weights to one scalar per model.
    weights = np.asarray(model_weights, dtype=np.float64)
    # Prevent accidental row/class weights from broadcasting into the model axis.
    if weights.ndim != 1 or len(weights) != ranks.shape[0]:
        raise ValueError("model_weights must contain one value per model.")
    # Require usable non-negative finite weights with a positive total.
    if not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("model_weights must be finite, non-negative, and sum to more than zero.")
    # Compute the weighted model-axis average while retaining [rows, classes].
    return np.average(scores, axis=0, weights=weights)


def calculate_rank_confidence_features(
    model_ranks: np.ndarray, aggregate_scores: np.ndarray, predicted_class_indices: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    """Calculate parcel-level rank confidence features for selected class indices."""
    # Normalize model ranks to [models, rows, classes].
    ranks = np.asarray(model_ranks, dtype=np.float64)
    # Normalize aggregate scores to [rows, classes].
    scores = np.asarray(aggregate_scores, dtype=np.float64)
    # Ensure scores describe exactly the parcel/class portion of the rank tensor.
    if ranks.ndim != 3 or scores.shape != ranks.shape[1:]:
        raise ValueError("Ranks and aggregate scores have incompatible shapes.")

    # Name tensor dimensions for indexing and per-row diagnostics.
    n_models, n_rows, n_classes = ranks.shape
    # Choose the canonical first maximum as the deterministic rank-aggregation winner.
    rank_winner_indices = np.argmax(scores, axis=1)
    # Retain each row's maximum so all classes tied at that value can be counted.
    maximum_scores = np.max(scores, axis=1, keepdims=True)
    # Mark exact/numerically equivalent aggregate ties instead of hiding argmax tie-breaking.
    rank_winner_tied = np.sum(np.isclose(scores, maximum_scores, rtol=1e-12, atol=1e-12), axis=1) > 1
    # Without an external strategy, calculate confidence for the rank winner itself.
    if predicted_class_indices is None:
        predicted_indices = rank_winner_indices
    else:
        # Otherwise evaluate rank support for the actual final soft-voting prediction.
        predicted_indices = np.asarray(predicted_class_indices, dtype=int)
        # Require exactly one predicted class index for every parcel row.
        if predicted_indices.shape != (n_rows,):
            raise ValueError("predicted_class_indices must contain one index per row.")
        # Prevent invalid indices from wrapping or indexing outside the canonical classes.
        if np.any((predicted_indices < 0) | (predicted_indices >= n_classes)):
            raise ValueError("predicted_class_indices contains an invalid class index.")

    # Build row indices for vectorized selection of each row's actual predicted class.
    row_indices = np.arange(n_rows)
    # Extract every model's rank for the actual predicted class, shaped [models, rows].
    winning_ranks = ranks[:, row_indices, predicted_indices]
    # Extract the aggregate Borda score of the actual predicted class.
    predicted_scores = scores[row_indices, predicted_indices]
    # Copy scores before masking so the original aggregate matrix stays unchanged.
    competing_scores = scores.copy()
    # Exclude the actual predicted class when searching for its strongest competitor.
    competing_scores[row_indices, predicted_indices] = -np.inf
    # Locate the highest-scoring competing class for every parcel.
    runner_up_indices = np.argmax(competing_scores, axis=1)
    # Retrieve each runner-up's original aggregate Borda score.
    runner_up_scores = scores[row_indices, runner_up_indices]

    # Return every primitive needed for calibration, reporting, and explainability.
    return {
        "predicted_class_indices": predicted_indices,  # Actual strategy class used for confidence.
        "rank_winner_indices": rank_winner_indices,  # Independent aggregate-rank winner.
        "runner_up_indices": runner_up_indices,  # Best competitor to the actual prediction.
        "aggregate_rank_score": predicted_scores,  # Borda strength of the actual prediction.
        "runner_up_rank_score": runner_up_scores,  # Borda strength of its best competitor.
        "rank_margin": predicted_scores - runner_up_scores,  # Signed prediction separation.
        "top1_agreement": np.mean(winning_ranks == 1.0, axis=0),  # Fraction ranking it first.
        "top2_agreement": np.mean(winning_ranks <= 2.0, axis=0),  # Fraction ranking it Top-2.
        "top3_agreement": np.mean(winning_ranks <= 3.0, axis=0),  # Fraction ranking it Top-3.
        "mean_rank": np.mean(winning_ranks, axis=0),  # Average predicted-class rank.
        "median_rank": np.median(winning_ranks, axis=0),  # Robust central predicted-class rank.
        "rank_std": np.std(winning_ranks, axis=0, ddof=0),  # Cross-model rank dispersion.
        "rank_iqr": np.percentile(winning_ranks, 75, axis=0) - np.percentile(winning_ranks, 25, axis=0),  # Robust dispersion.
        "rank_winner_tied": rank_winner_tied,  # Exact aggregate-winner ambiguity flag.
        "rank_agrees_with_prediction": rank_winner_indices == predicted_indices,  # Strategy agreement flag.
        "n_models_used": np.full(n_rows, n_models, dtype=int),  # Ensemble size for validity/calibration.
    }


def assign_confidence_levels(
    features: Mapping[str, np.ndarray], thresholds: Mapping[str, Mapping[str, float]] | None = None, minimum_models: int = 3
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assign HIGH/MEDIUM/LOW levels and return validity metadata."""
    # Use caller-frozen thresholds, or provisional defaults for legacy/fallback runs.
    configured = thresholds or DEFAULT_RANK_CONFIDENCE_THRESHOLDS
    # Normalize each feature to a one-dimensional parcel-aligned numeric array.
    top1 = np.asarray(features["top1_agreement"], dtype=np.float64)
    top2 = np.asarray(features["top2_agreement"], dtype=np.float64)
    score = np.asarray(features["aggregate_rank_score"], dtype=np.float64)
    margin = np.asarray(features["rank_margin"], dtype=np.float64)
    # Retain the ensemble size used for the minimum-model validity safeguard.
    models = np.asarray(features["n_models_used"], dtype=int)
    # Identify whether rank aggregation independently supports the final strategy class.
    agrees = np.asarray(features["rank_agrees_with_prediction"], dtype=bool)
    # Default old feature payloads to no tie while supporting the explicit modern flag.
    winner_tied = np.asarray(features.get("rank_winner_tied", np.zeros(top1.shape, dtype=bool)), dtype=bool)

    # A row is numerically valid only when enough member models contributed.
    valid = models >= int(minimum_models)
    # Only valid, non-tied, rank-agreeing rows may satisfy provisional H/M thresholds.
    eligible = valid & agrees & ~winner_tied
    # Read HIGH and MEDIUM rules once for concise vectorized comparisons.
    high_cfg = configured["high"]
    medium_cfg = configured["medium"]
    # HIGH requires every configured agreement, strength, and separation condition.
    high = (
        eligible
        & (top1 >= float(high_cfg["min_top1_agreement"]))
        & (top2 >= float(high_cfg["min_top2_agreement"]))
        & (score >= float(high_cfg["min_rank_score"]))
        & (margin >= float(high_cfg["min_rank_margin"]))
    )
    # MEDIUM applies only to non-HIGH rows satisfying all weaker conditions.
    medium = (
        eligible
        & ~high
        & (top1 >= float(medium_cfg["min_top1_agreement"]))
        & (top2 >= float(medium_cfg["min_top2_agreement"]))
        & (score >= float(medium_cfg["min_rank_score"]))
        & (margin >= float(medium_cfg["min_rank_margin"]))
    )
    # Initialize conservatively so any unhandled or threshold-failing row remains LOW.
    levels = np.full(top1.shape, "LOW", dtype=object)
    # Promote rows that satisfy the complete MEDIUM rule.
    levels[medium] = "MEDIUM"
    # Promote rows satisfying HIGH after MEDIUM assignment so HIGH takes precedence.
    levels[high] = "HIGH"
    # Explain ordinary LOW rows as failing one or more provisional thresholds.
    reasons = np.full(top1.shape, "thresholds_not_met", dtype=object)
    # HIGH and MEDIUM need no downgrade explanation.
    reasons[medium | high] = ""
    # Give exact aggregate ties a distinct operational explanation.
    reasons[valid & winner_tied] = "rank_winner_tie"
    # Minimum-model failure takes precedence because it makes confidence invalid.
    reasons[models < int(minimum_models)] = "insufficient_models"
    # Explain valid non-tied rows where soft voting and rank aggregation disagree.
    reasons[valid & ~winner_tied & ~agrees] = "rank_prediction_disagreement"
    # Convert object strings to stable arrays and return levels, validity, and reasons together.
    return levels.astype(str), valid, reasons.astype(str)


def classify_ensemble_rank_based(
    model_probabilities: np.ndarray,
    class_names: Sequence[str],
    model_names: Sequence[str] | None = None,
    predicted_class_indices: np.ndarray | None = None,
    model_weights: np.ndarray | None = None,
    confidence_thresholds: Mapping[str, Mapping[str, float]] | None = None,
    minimum_models: int = 3,
) -> dict[str, Any]:
    """Return rank aggregation and confidence results for a batch of parcels."""
    # Validate the full [models, rows, classes] tensor before any ranking calculation.
    probabilities = validate_model_probabilities(model_probabilities, class_names, model_names)
    # Convert each model's probability ordering to descending average ranks.
    ranks = probabilities_to_ranks(probabilities)
    # Convert ranks to Borda scores and aggregate them across ensemble members.
    aggregate_scores = aggregate_rank_scores(ranks, model_weights=model_weights)
    # Calculate consensus diagnostics for the actual final strategy prediction.
    features = calculate_rank_confidence_features(ranks, aggregate_scores, predicted_class_indices=predicted_class_indices)
    # Apply provisional/frozen rank rules; OOF calibration may replace these labels later.
    levels, valid, reasons = assign_confidence_levels(features, thresholds=confidence_thresholds, minimum_models=minimum_models)
    # Normalize canonical labels once before converting stored class indices back to names.
    labels = np.asarray(class_names, dtype=str)
    # Return rank features, fallback levels, safeguards, and human-readable competing classes.
    return {
        **features,  # Preserve every numeric rank diagnostic for calibration and reporting.
        "confidence_level": levels,  # Provisional/fallback category before optional calibration.
        "confidence_valid": valid,  # Minimum-model validity from the rank stage.
        "confidence_reason": reasons,  # Structural or threshold-based explanation.
        "rank_prediction": labels[features["rank_winner_indices"]],  # Rank winner label.
        "runner_up_class": labels[features["runner_up_indices"]],  # Best competitor label.
    }
