"""Apply frozen confidence evidence and inspection policy to prediction outputs."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from ml_classification.shared.class_reliability import combine_confidence_components, get_class_reliability
from ml_classification.shared.rank_confidence import (
    classify_ensemble_rank_based,
    rank_confidence_thresholds_from_config,
)
from ml_classification.step_05_predict.libraries.inspection_priority.scoring import calculate_inspection_metrics


def apply_rank_confidence_columns(config, final_df: pd.DataFrame, rank_confidence: dict[str, np.ndarray] | None) -> pd.DataFrame:
    """Attach rank-confidence outputs to an existing prediction frame."""
    if rank_confidence is None:
        return final_df

    rank_columns = {
        config.prediction_confidence_level_column: "prediction_confidence_level",
        "prediction_confidence_valid": "prediction_confidence_valid",
        "prediction_confidence_reason": "prediction_confidence_reason",
        "prediction_confidence_source": "prediction_confidence_source",
        "prediction_class_reliability_applied": "prediction_class_reliability_applied",
        "prediction_mean_borda": "prediction_mean_borda",
        "prediction_rank_range": "prediction_rank_range",
        "prediction_mean_rank": "prediction_mean_rank",
        "prediction_median_rank": "prediction_median_rank",
        "prediction_rank_std": "prediction_rank_std",
        "prediction_rank_iqr": "prediction_rank_iqr",
        "prediction_borda_winner": "borda_prediction",
        "prediction_borda_winner_tied": "rank_winner_tied",
        "prediction_rank_agrees_with_final": "rank_agrees_with_prediction",
        "prediction_rank_confidence_level": "rank_confidence_level",
        "prediction_class_oof_precision": "class_oof_precision",
        "prediction_class_oof_support": "class_oof_support",
        "prediction_class_reliability_level": "class_reliability_level",
        "prediction_class_reliability_valid": "class_reliability_valid",
        "prediction_class_reliability_reason": "class_reliability_reason",
        "prediction_borda_margin": "borda_margin",
        "prediction_top1_agreement": "top1_agreement",
        "prediction_top2_agreement": "top2_agreement",
        "prediction_top3_agreement": "top3_agreement",
        "prediction_runner_up_class": "runner_up_class",
        "prediction_runner_up_borda": "runner_up_borda",
        "prediction_rank_models_used": "n_models_used",
        "prediction_rank_confidence_valid": "rank_confidence_valid",
        "prediction_rank_confidence_reason": "rank_confidence_reason",
    }
    for output_column, result_key in rank_columns.items():
        if result_key in rank_confidence:
            final_df[output_column] = rank_confidence[result_key]
    final_df[config.prediction_review_column] = (final_df[config.prediction_confidence_level_column] == "LOW") | ~final_df[
        "prediction_confidence_valid"
    ].astype(bool)
    return final_df


def resolve_rank_confidence_contract(config, selection: dict[str, Any]) -> dict[str, Any]:
    """Resolve the frozen Step-4 confidence contract with legacy fallback."""
    frozen = selection.get("confidence")
    legacy_rank = selection.get("rank_confidence")
    supported_methods = {"rank_consensus_with_class_reliability_guard", "rank_consensus_only"}
    if frozen is None and isinstance(legacy_rank, dict) and legacy_rank.get("method") in supported_methods:
        frozen = legacy_rank
    if frozen is not None:
        # Reject corrupted or manually edited summaries with an invalid object type.
        if not isinstance(frozen, dict):
            raise RuntimeError("Error: selection_summary rank_confidence must be an object.")
        # Step 4's enabled flag is authoritative even if the live config later changes.
        enabled = bool(frozen.get("enabled", False))
        if not enabled:
            # Return an explicit disabled contract so downstream summary fields stay stable.
            return {
                "enabled": False,
                "thresholds": None,
                "minimum_models": None,
                "class_reliability": None,
                "class_reliability_enabled": False,
                "source": "selection_summary",
            }
        if frozen.get("method") not in supported_methods:
            raise RuntimeError(
                "Error: The frozen confidence contract uses an obsolete method. Rerun Step 4 to create the rank-consensus-with-class-reliability contract."
            )
        rank = frozen.get("rank")
        reliability = frozen.get("class_reliability")
        requires_reliability = frozen["method"] == "rank_consensus_with_class_reliability_guard"
        if not isinstance(rank, dict) or (requires_reliability and not isinstance(reliability, dict)):
            raise RuntimeError("Error: The frozen confidence contract is incomplete; rerun Step 4.")
        return {
            "enabled": True,
            "method": frozen["method"],
            "class_reliability_enabled": bool(frozen.get("class_reliability_enabled", True)),
            "thresholds": {
                "high_min_borda": rank["high_min_borda"],
                "high_max_range": rank["high_max_range"],
                "medium_min_borda": rank["medium_min_borda"],
                "medium_max_range": rank["medium_max_range"],
            },
            "minimum_models": int(rank["minimum_models"]),
            "class_reliability": reliability,
            "source": "selection_summary",
        }

    if selection.get("rank_confidence") is not None and config.rank_confidence_enabled:
        raise RuntimeError(
            "Error: selection_summary.json contains the obsolete class x Top-1 confidence method. Rerun Step 4 before production prediction."
        )
    if config.rank_confidence_enabled and getattr(config, "class_reliability_enabled", True):
        raise RuntimeError("Error: selection_summary.json has no frozen class-reliability contract. Rerun Step 4 before prediction.")
    return {
        "enabled": bool(config.rank_confidence_enabled),
        "method": "rank_consensus_only" if config.rank_confidence_enabled else None,
        "class_reliability_enabled": bool(getattr(config, "class_reliability_enabled", True)),
        "thresholds": rank_confidence_thresholds_from_config(config),
        "minimum_models": int(config.rank_confidence_minimum_models),
        "class_reliability": None,
        "source": "live_config_legacy_fallback",
    }


def calculate_rank_confidence(config, selection, predictions, member_probabilities):
    """Calculate confidence using the definition frozen during evaluation."""
    # Resolve Step 4's contract before examining any production probabilities.
    contract = resolve_rank_confidence_contract(config, selection)
    # Skip all rank-confidence work when the frozen strategy disabled it.
    if not contract["enabled"]:
        return None, contract

    # Preserve the selected model order used by voting and the frozen calibration.
    selected_models = list(selection["selected_models"])
    # Stack raw member probabilities as [models, rows, classes] for within-model ranking.
    stacked_probabilities = np.stack([member_probabilities[model_name] for model_name in selected_models], axis=0)
    # Map canonical string labels to encoded probability-column indices.
    label_to_index = {str(label): index for index, label in enumerate(selection["labels"])}
    # Identify the actual final soft-voting class whose confidence must be explained.
    predicted_indices = np.asarray([label_to_index[str(label)] for label in predictions], dtype=int)
    # Calculate scale-invariant rank features and structural safeguard outcomes first.
    result = classify_ensemble_rank_based(
        stacked_probabilities,
        class_names=list(selection["labels"]),
        model_names=selected_models,
        predicted_class_indices=predicted_indices,
        confidence_thresholds=contract["thresholds"],
        minimum_models=contract["minimum_models"],
    )
    if contract["class_reliability"] is not None:
        reliability = get_class_reliability(predictions, contract["class_reliability"])
        result.update(reliability)
        class_reliability_applied = bool(contract.get("class_reliability_enabled", True))
        result["prediction_class_reliability_applied"] = class_reliability_applied
        result["prediction_confidence_source"] = "rank_plus_class_reliability" if class_reliability_applied else "rank_only"
        if class_reliability_applied:
            final = combine_confidence_components(
                result["rank_confidence_level"],
                result["rank_confidence_valid"],
                result["rank_confidence_reason"],
                reliability["class_reliability_level"],
                reliability["class_reliability_valid"],
                reliability["class_reliability_reason"],
            )
            result.update(final)
        else:
            result["prediction_confidence_level"] = result["rank_confidence_level"]
            result["prediction_confidence_valid"] = result["rank_confidence_valid"]
            result["prediction_confidence_reason"] = result["rank_confidence_reason"]
    else:
        result["prediction_class_reliability_applied"] = False
        result["prediction_confidence_source"] = "rank_only"
        result["prediction_confidence_level"] = result["rank_confidence_level"]
        result["prediction_confidence_valid"] = result["rank_confidence_valid"]
        result["prediction_confidence_reason"] = result["rank_confidence_reason"]
    # Return both parcel results and contract provenance for output summaries.
    return result, contract


def add_inspection_metrics(config, final_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Attach inspection risks when all three independent inputs exist."""
    # Allow deployments to disable the complete inspection layer without changing prediction.
    if not config.inspection_scoring_enabled:
        return final_df, {"inspection_scoring_enabled": False, "inspection_scoring_available": False}

    # Declaration columns are optional; only the three label-independent risk inputs are required.
    required = {
        config.prediction_confidence_level_column,
        config.inspection_data_reliability_column,
        config.inspection_geometry_complexity_column,
    }
    missing = sorted(required.difference(final_df.columns))
    if missing:
        warnings.warn(f"Inspection scoring skipped because required columns are missing: {missing}", RuntimeWarning, stacklevel=2)
        return final_df, {
            "inspection_scoring_enabled": True,
            "inspection_scoring_available": False,
            "inspection_missing_columns": missing,
        }

    # The helper adds score, label agreement, final need, check type, and reasons per parcel.
    scored = calculate_inspection_metrics(final_df, config)
    mean_score = scored["inspection_score"].mean()
    # Persist aggregate counts in predict_summary.json for fast operational monitoring.
    return scored, {
        "inspection_scoring_enabled": True,
        "inspection_scoring_available": True,
        "inspection_weights": {
            "model": config.inspection_model_weight,
            "data": config.inspection_data_weight,
            "geometry": config.inspection_geometry_weight,
        },
        "inspection_need_counts": scored["inspection_need"].value_counts(dropna=False).to_dict(),
        "inspection_check_type_counts": scored["inspection_check_type"].value_counts(dropna=False).to_dict(),
        "label_prediction_status_counts": scored["label_prediction_status"].value_counts(dropna=False).to_dict(),
        "inspection_score_mean": None if pd.isna(mean_score) else float(mean_score),
    }
