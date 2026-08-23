"""OOF-derived class-aware calibration for rank-based confidence features."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


# Persist a stable method identifier so Step 5 can reject unknown calibration schemas.
CALIBRATION_METHOD = "oof_predicted_class_top1_empirical_accuracy"


def _confidence_level(accuracy: float, high_min_accuracy: float, medium_min_accuracy: float) -> str:
    # Test HIGH first because its threshold is stricter than the MEDIUM threshold.
    if accuracy >= high_min_accuracy:
        return "HIGH"
    # Values below HIGH but at or above the medium boundary receive MEDIUM.
    if accuracy >= medium_min_accuracy:
        return "MEDIUM"
    # All empirically weaker regions are conservatively assigned LOW.
    return "LOW"


def _validate_calibration_inputs(parcel_df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    # List every OOF field required to fit a leakage-safe calibration table.
    required = {
        "predicted_class",
        "correct",
        "top1_agreement",
        "n_models_used",
        "confidence_valid",
        "rank_agrees_with_prediction",
        "rank_winner_tied",
    }
    # Report all absent columns together so malformed artifacts are easy to diagnose.
    missing = sorted(required.difference(parcel_df.columns))
    if missing:
        raise ValueError(f"OOF rank-confidence data is missing required columns: {missing}")
    # An empty OOF frame cannot provide an empirical correctness estimate.
    if parcel_df.empty:
        raise ValueError("OOF rank-confidence data cannot be empty.")

    # Work on a copy so validation never mutates the caller's diagnostic frame.
    frame = parcel_df.copy()
    # Normalize ensemble size to integers and reject non-numeric values immediately.
    model_counts = pd.to_numeric(frame["n_models_used"], errors="raise").astype(int)
    # Calibration assumes the same frozen ensemble generated every OOF row.
    unique_model_counts = np.unique(model_counts)
    if len(unique_model_counts) != 1 or unique_model_counts[0] < 1:
        raise ValueError("OOF calibration requires one positive, constant n_models_used value.")
    # Store the validated ensemble size for table keys and deployment checks.
    n_models = int(unique_model_counts[0])
    # Convert agreement fractions to numeric values before recovering vote counts.
    top1 = pd.to_numeric(frame["top1_agreement"], errors="raise").to_numpy(dtype=np.float64)
    # Recover how many models ranked the final predicted class first.
    top1_counts = np.rint(top1 * n_models).astype(int)
    # Ensure each fraction is realizable for this fixed number of models.
    if np.any((top1_counts < 0) | (top1_counts > n_models)) or not np.allclose(
        top1, top1_counts / n_models, rtol=1e-9, atol=1e-9
    ):
        raise ValueError("top1_agreement is incompatible with n_models_used.")
    # Canonicalize class keys as strings because the table is persisted through JSON.
    frame["predicted_class"] = frame["predicted_class"].astype(str)
    # Normalize correctness so summing the column counts correct OOF predictions.
    frame["correct"] = frame["correct"].astype(bool)
    # Attach the discrete Top-1 vote count used as the second calibration key.
    frame["top1_count"] = top1_counts
    # Return both normalized evidence and its constant ensemble size.
    return frame, n_models


def fit_class_aware_confidence_calibration(
    oof_parcel_df: pd.DataFrame,
    *,
    high_min_accuracy: float = 0.85,
    medium_min_accuracy: float = 0.65,
    minimum_oof_support: int = 100,
    max_top1_pool_distance: int = 1,
) -> dict[str, Any]:
    """Fit class/Top-1 empirical correctness tables from OOF predictions only."""
    # Normalize configuration values to stable scalar types for JSON serialization.
    high_min_accuracy = float(high_min_accuracy)
    medium_min_accuracy = float(medium_min_accuracy)
    # Normalize the maximum distance to an integer number of model votes.
    max_top1_pool_distance = int(max_top1_pool_distance)
    minimum_oof_support = int(minimum_oof_support)
    # Enforce an ordered empirical interpretation from LOW through HIGH.
    if not 0.0 <= medium_min_accuracy <= high_min_accuracy <= 1.0:
        raise ValueError("Empirical confidence thresholds must satisfy 0 <= MEDIUM <= HIGH <= 1.")
    # At least one OOF row must support any deployable estimate.
    if minimum_oof_support < 1:
        raise ValueError("minimum_oof_support must be at least 1.")
    # A negative distance has no meaningful neighboring-bin interpretation.
    if max_top1_pool_distance < 0:
        raise ValueError("max_top1_pool_distance must be at least 0.")


    # Validate the OOF evidence before fitting any class-specific groups.
    frame, n_models = _validate_calibration_inputs(oof_parcel_df)
    # Exclude structural ambiguity because deployment downgrades these rows directly.
    eligible = frame.loc[
        frame["confidence_valid"].astype(bool)
        & frame["rank_agrees_with_prediction"].astype(bool)
        & ~frame["rank_winner_tied"].astype(bool)
    ].copy()

    # Accumulate JSON-safe rows for all predicted-class/vote-count combinations.
    table: list[dict[str, Any]] = []
    # Fit each predicted class independently to capture systematic class bias.
    for predicted_class in sorted(frame["predicted_class"].unique()):
        # Use only eligible OOF predictions assigned to this class.
        class_rows = eligible.loc[eligible["predicted_class"] == predicted_class]
        # Summarize sample size and correct predictions for each observed vote count.
        grouped = {
            int(top1_count): {
                "support": int(len(group)),
                "correct": int(group["correct"].sum()),
            }
            for top1_count, group in class_rows.groupby("top1_count", observed=True)
        }
        # Keep observed counts sorted so pooling and saved output are deterministic.
        observed_counts = sorted(grouped)
        # Create a deployable lookup row for every possible Top-1 vote count.
        for target_count in range(n_models + 1):
            # Read the target bin's own evidence before considering any neighbors.
            exact_group = grouped.get(target_count)
            # Record zero when this class/Top-1 combination was never observed in OOF.
            exact_support = int(exact_group["support"]) if exact_group is not None else 0
            # Record the exact-bin numerator separately for transparent audit output.
            exact_correct = int(exact_group["correct"]) if exact_group is not None else 0

            # Never infer an unseen Top-1 pattern from another, potentially distant pattern.
            if exact_support == 0:
                # No bins contribute because the exact target pattern has no OOF evidence.
                included_counts: list[int] = []
                # Keep support at zero so the artifact clearly distinguishes unseen patterns.
                support = 0
                # Keep the numerator at zero because no empirical estimate is made.
                correct = 0
                # Null accuracy communicates that no correctness probability was estimated.
                empirical_accuracy = None
                # An unseen class/Top-1 key is not valid for operational calibration.
                calibration_valid = False
                # Expose the exact reason through the frozen table and Step 5 output.
                source = "unseen_class_top1_bin"
                # Conservative review behavior is always LOW for unseen patterns.
                level = "LOW"
                # No pooling occurred, so there is no observed pooling distance.
                max_pool_distance = 0
            # Use a sufficiently large exact bin without diluting it with neighbors.
            elif exact_support >= minimum_oof_support:
                # The exact vote-count bin is the only contributor.
                included_counts = [target_count]
                # The effective support equals the target bin's own OOF support.
                support = exact_support
                # The effective numerator equals the target bin's correct predictions.
                correct = exact_correct
                # Calculate empirical correctness directly from the exact OOF bin.
                empirical_accuracy = float(correct / support)
                # Sufficient exact support makes this lookup operationally valid.
                calibration_valid = True
                # Identify this row as an exact rather than pooled estimate.
                source = "exact_class_top1_bin"
                # Map the exact empirical correctness to the configured category.
                level = _confidence_level(empirical_accuracy, high_min_accuracy, medium_min_accuracy)
                # Exact-only calibration has a pooling distance of zero.
                max_pool_distance = 0
            # Sparse but observed targets may borrow evidence only from bounded adjacent bins.
            else:
                # Select all observed bins within the configured vote-count distance.
                included_counts = [
                    count
                    for count in observed_counts
                    if abs(count - target_count) <= max_top1_pool_distance
                ]
                # Sum the bounded region's OOF observations for the effective support.
                support = sum(grouped[count]["support"] for count in included_counts)
                # Sum correct predictions across the same bounded region.
                correct = sum(grouped[count]["correct"] for count in included_counts)
                # Some exact evidence guarantees positive pooled support here.
                empirical_accuracy = float(correct / support)
                # The pooled region is usable only if it reaches the configured evidence floor.
                calibration_valid = support >= minimum_oof_support
                # Record the farthest contributor actually included in this row.
                max_pool_distance = max(abs(count - target_count) for count in included_counts)
                # Identify successful bounded pooling separately from invalid sparse evidence.
                source = (
                    "adjacent_class_top1_bins"
                    if calibration_valid
                    else "insufficient_adjacent_class_top1_support"
                )
                # Invalid sparse estimates stay LOW even if their point accuracy is high.
                level = (
                    _confidence_level(empirical_accuracy, high_min_accuracy, medium_min_accuracy)
                    if calibration_valid
                    else "LOW"
                )

            # Freeze the estimate, assigned label, support, and provenance as one row.
            table.append(
                {
                    "predicted_class": predicted_class,  # First lookup key and systematic-bias control.
                    "top1_count": int(target_count),  # Second lookup key as an exact vote count.
                    "top1_agreement": float(target_count / n_models),  # Human-readable vote fraction.
                    "empirical_accuracy": empirical_accuracy,  # Supported exact/pooled P(correct).
                    "support": int(support),  # Backward-compatible effective OOF support.
                    "exact_support": exact_support,  # OOF rows in this exact class/Top-1 bin.
                    "pooled_support": int(support),  # OOF rows after bounded adjacent pooling.
                    "max_pool_distance": int(max_pool_distance),  # Farthest included vote-count bin.
                    "confidence_level": level,  # Frozen user-facing empirical category.
                    "calibration_valid": bool(calibration_valid),  # Whether support met the floor.
                    "calibration_source": source,  # Exact, pooled, or insufficient provenance.
                    "included_top1_counts": included_counts,  # Vote bins included in this estimate.
                }
            )

    # Return a self-describing contract that Step 5 can apply without live labels.
    return {
        "schema_version": "1.1.0",  # Version the bounded-pooling calibration contract.
        "method": CALIBRATION_METHOD,  # Tell inference which application algorithm to use.
        "feature": "top1_agreement",  # Document the rank feature used for table bins.
        "class_aware": True,  # Confirm predicted class is part of every lookup key.
        "n_models": n_models,  # Freeze the ensemble size used to derive vote counts.
        "high_min_empirical_accuracy": high_min_accuracy,  # Record HIGH semantics.
        "medium_min_empirical_accuracy": medium_min_accuracy,  # Record MEDIUM semantics.
        "minimum_oof_support": minimum_oof_support,  # Record the evidence floor.
        "max_top1_pool_distance": max_top1_pool_distance,  # Freeze the permitted vote distance.
        "pooling": "bounded_adjacent_top1_bins_within_predicted_class",  # Record sparse handling.
        "unsupported_fallback": "LOW_invalid",  # Record conservative unseen/rare behavior.
        "oof_rows_total": int(len(frame)),  # Count all OOF rows presented to calibration.
        "oof_rows_eligible": int(len(eligible)),  # Count rows left after structural exclusions.
        "table": table,  # Store the complete frozen class/vote lookup.
    }


def apply_class_aware_confidence_calibration(
    *,
    predicted_classes: Sequence[str],
    top1_agreement: np.ndarray,
    n_models_used: np.ndarray,
    confidence_valid: np.ndarray,
    confidence_reason: np.ndarray,
    rank_agrees_with_prediction: np.ndarray,
    rank_winner_tied: np.ndarray,
    calibration: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Apply a frozen OOF calibration table without using current-row labels."""
    # Reject contracts produced by an unknown algorithm instead of guessing semantics.
    if calibration.get("method") != CALIBRATION_METHOD:
        raise ValueError(f"Unsupported rank-confidence calibration method: {calibration.get('method')!r}")
    # Read the ensemble size frozen when the OOF calibration was fitted.
    expected_models = int(calibration["n_models"])
    # Normalize predicted class labels to the same string keys stored in JSON.
    predicted = np.asarray(predicted_classes, dtype=str)
    # Normalize Top-1 agreement to floating-point values for vote-count recovery.
    top1 = np.asarray(top1_agreement, dtype=np.float64)
    # Normalize per-row ensemble size to integers for the frozen-size check.
    models = np.asarray(n_models_used, dtype=int)
    # Preserve the base rank calculation's minimum-model validity result.
    base_valid = np.asarray(confidence_valid, dtype=bool)
    # Use object strings while editing so long reason values are never truncated by NumPy.
    reasons = np.asarray(confidence_reason, dtype=object).copy()
    # Track whether the soft-voting prediction matches the independent rank winner.
    agrees = np.asarray(rank_agrees_with_prediction, dtype=bool)
    # Track exact aggregate-score ties, which always remain structurally LOW.
    tied = np.asarray(rank_winner_tied, dtype=bool)
    # Collect all aligned arrays so their shapes can be validated uniformly.
    arrays = (top1, models, base_valid, reasons, agrees, tied)
    # Every input must provide exactly one value for every prediction row.
    if any(values.shape != predicted.shape for values in arrays):
        raise ValueError("Calibration inputs must contain one aligned value per prediction.")

    # Build an O(1) lookup keyed by predicted class and integer Top-1 vote count.
    lookup = {
        (str(row["predicted_class"]), int(row["top1_count"])): row
        for row in calibration.get("table", [])
    }
    # Default every row to LOW so all failure and fallback paths are conservative.
    levels = np.full(predicted.shape, "LOW", dtype=object)
    # Use NaN when no supported empirical accuracy can be assigned to a row.
    empirical_accuracy = np.full(predicted.shape, np.nan, dtype=np.float64)
    # Store zero support until a matching calibration row is found.
    support = np.zeros(predicted.shape, dtype=int)
    # Retain whether the result came from an exact bin, pooled bins, or a safeguard.
    sources = np.full(predicted.shape, "", dtype=object)
    # Mark empirical calibration valid only after a supported table lookup succeeds.
    calibration_valid = np.zeros(predicted.shape, dtype=bool)

    # Apply the frozen lookup and safeguards independently to every prediction row.
    for index, predicted_class in enumerate(predicted):
        # Preserve an invalid base result, such as an ensemble with too few models.
        if not base_valid[index]:
            sources[index] = "base_confidence_invalid"
            continue
        # Exact aggregate ties are valid observations but are always operationally LOW.
        if tied[index]:
            reasons[index] = "rank_winner_tie"
            sources[index] = "structural_rank_safeguard"
            continue
        # Soft-vote/rank disagreement is also retained as a valid hard-LOW safeguard.
        if not agrees[index]:
            reasons[index] = "rank_prediction_disagreement"
            sources[index] = "structural_rank_safeguard"
            continue
        # Refuse calibration when the deployed ensemble differs from the fitted ensemble.
        if models[index] != expected_models:
            reasons[index] = "calibration_model_count_mismatch"
            sources[index] = "model_count_mismatch"
            continue

        # Convert the agreement fraction back to its stable integer vote-count key.
        top1_count = int(np.rint(top1[index] * models[index]))
        # Retrieve the frozen estimate for this predicted class and vote count.
        row = lookup.get((str(predicted_class), top1_count))
        # A class never predicted in OOF has no defensible empirical calibration.
        if row is None:
            reasons[index] = "unseen_predicted_class"
            sources[index] = "unsupported_fallback"
            continue
        # Expose the number of OOF parcels underlying this row's estimate.
        support[index] = int(row["support"])
        # Expose whether the estimate used an exact bin or neighboring-bin pooling.
        sources[index] = str(row["calibration_source"])
        # Read the empirical accuracy, which is null for classes with no eligible evidence.
        accuracy = row.get("empirical_accuracy")
        if accuracy is not None:
            empirical_accuracy[index] = float(accuracy)
        # Insufficient support invalidates calibration even when a small-sample accuracy exists.
        if not bool(row["calibration_valid"]):
            # Preserve the stronger distinction between unseen and merely sparse patterns.
            reasons[index] = (
                "unseen_class_top1_bin"
                if row.get("calibration_source") == "unseen_class_top1_bin"
                else "insufficient_oof_calibration_support"
            )
            continue

        # Mark the frozen lookup as supported before assigning its empirical category.
        calibration_valid[index] = True
        # Apply the precomputed H/M/L label; no threshold is recalculated at inference.
        levels[index] = str(row["confidence_level"])
        # Explain supported LOW rows while leaving HIGH and MEDIUM reason-free.
        reasons[index] = "" if levels[index] != "LOW" else "empirical_accuracy_below_medium"

    # Only ordinary rank-agreeing, non-tied rows require a successful empirical lookup.
    requires_empirical_lookup = base_valid & agrees & ~tied
    # Start from base validity so structural LOW safeguards remain valid observations.
    final_valid = base_valid.copy()
    # For empirically calibrated rows, validity additionally requires sufficient OOF support.
    final_valid[requires_empirical_lookup] = calibration_valid[requires_empirical_lookup]
    # Return aligned arrays that replace the provisional confidence fields and add diagnostics.
    return {
        "confidence_level": levels.astype(str),  # Final frozen HIGH/MEDIUM/LOW category.
        "confidence_valid": final_valid,  # Overall structural and empirical validity.
        "confidence_reason": reasons.astype(str),  # Explanation for LOW or invalid rows.
        "confidence_empirical_accuracy": empirical_accuracy,  # Frozen OOF correctness estimate.
        "confidence_calibration_support": support,  # OOF evidence behind the estimate.
        "confidence_calibration_valid": calibration_valid,  # Empirical lookup support status.
        "confidence_calibration_source": sources.astype(str),  # Exact/pooling/fallback source.
    }


def apply_calibration_to_parcel_frame(parcel_df: pd.DataFrame, calibration: Mapping[str, Any]) -> pd.DataFrame:
    """Return a parcel frame whose confidence fields use the frozen calibration."""
    # Preserve schema and avoid calling the array API when there are no parcels.
    if parcel_df.empty:
        return parcel_df.copy()
    # Extract aligned columns and apply the exact same frozen logic used by Step 5.
    calibrated = apply_class_aware_confidence_calibration(
        predicted_classes=parcel_df["predicted_class"].astype(str).to_numpy(),
        top1_agreement=parcel_df["top1_agreement"].to_numpy(),
        n_models_used=parcel_df["n_models_used"].to_numpy(),
        confidence_valid=parcel_df["confidence_valid"].to_numpy(),
        confidence_reason=parcel_df["confidence_reason"].to_numpy(),
        rank_agrees_with_prediction=parcel_df["rank_agrees_with_prediction"].to_numpy(),
        rank_winner_tied=parcel_df["rank_winner_tied"].to_numpy(),
        calibration=calibration,
    )
    # Copy the frame so evaluation callers keep ownership of their original data.
    result = parcel_df.copy()
    # Replace provisional fields and attach calibration diagnostics column by column.
    for column, values in calibrated.items():
        result[column] = values
    # Return the calibrated evaluation frame for OOF, train, or untouched holdout use.
    return result
