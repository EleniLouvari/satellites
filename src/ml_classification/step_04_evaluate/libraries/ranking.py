"""Build parcel-level ranking outputs and summarize confidence correctness."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score

from ml_classification.shared.class_reliability import combine_confidence_components, get_class_reliability
from ml_classification.shared.probabilities import apply_class_probability_multipliers
from ml_classification.shared.rank_confidence import (
    classify_ensemble_rank_based,
    probabilities_to_ranks,
    rank_confidence_thresholds_from_config,
)


def rank_positions_desc(matrix: np.ndarray) -> np.ndarray:
    """Return descending within-row average ranks, including deterministic ties."""
    return probabilities_to_ranks(matrix)


def build_parcel_ranking_outputs(
    config,
    probability_cache: dict[str, np.ndarray],
    selection: dict[str, Any],
    labels: list[str],
    y_true: pd.Series,
    id_col: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create per-parcel best-class outputs and ranking-method summary metrics."""
    selected_models = _resolve_ranking_models(selection, probability_cache)

    if not selected_models:
        return pd.DataFrame(), pd.DataFrame()

    probability_matrices = _validated_probability_matrices(probability_cache, selected_models, len(labels))
    stacked = np.stack(probability_matrices, axis=0)  # [n_models, n_rows, n_classes]
    avg_prob, median_prob, avg_prob_norm, median_prob_norm = _average_and_median_probabilities(stacked)
    model_ranks = probabilities_to_ranks(stacked)
    avg_rank = np.mean(model_ranks, axis=0)
    median_rank = np.median(model_ranks, axis=0)

    rank_confidence, strategy_indices = _rank_confidence_and_indices(config, selection, labels, stacked, selected_models, avg_prob_norm)
    parcel_df = _build_parcel_ranking_frame(
        config,
        labels,
        y_true,
        id_col,
        avg_prob,
        median_prob,
        avg_prob_norm,
        median_prob_norm,
        avg_rank,
        median_rank,
        strategy_indices,
        rank_confidence,
    )
    summary_df = _build_ranking_summary(
        config, labels, y_true, parcel_df, avg_prob_norm, median_prob_norm, len(selected_models)
    )
    return parcel_df, summary_df


def _resolve_ranking_models(selection: dict[str, Any], probability_cache: dict[str, np.ndarray]) -> list[str]:
    """Resolve selected model names with validation and fallback behavior."""
    requested_models = list(selection.get("selected_models", []))
    missing_models = [model_name for model_name in requested_models if model_name not in probability_cache]
    if missing_models:
        raise RuntimeError(f"Error: Selected models are missing probability outputs: {missing_models}")
    return requested_models or sorted(probability_cache.keys())


def _validated_probability_matrices(
    probability_cache: dict[str, np.ndarray], selected_models: list[str], n_labels: int
) -> list[np.ndarray]:
    """Return validated probability matrices for selected models."""
    matrices: list[np.ndarray] = []
    for model_name in selected_models:
        matrix = np.asarray(probability_cache[model_name], dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != n_labels:
            raise ValueError(f"Error: Model {model_name!r} returned {matrix.shape} probabilities; expected (n_rows, {n_labels}).")
        if not np.isfinite(matrix).all() or np.any(matrix < 0) or np.any(matrix.sum(axis=1) <= 0):
            raise ValueError(f"Error: Model {model_name!r} returned malformed or non-finite probabilities.")
        matrices.append(matrix)
    return matrices


def _average_and_median_probabilities(stacked: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute raw and row-normalized average/median probability matrices."""
    avg_prob = np.mean(stacked, axis=0)
    median_prob = np.median(stacked, axis=0)
    avg_sum = avg_prob.sum(axis=1, keepdims=True)
    median_sum = median_prob.sum(axis=1, keepdims=True)
    # Normalize aggregate rows; coordinate-wise medians need not sum to one.
    avg_prob_norm = avg_prob / np.where(avg_sum > 0, avg_sum, 1)
    median_prob_norm = median_prob / np.where(median_sum > 0, median_sum, 1)
    return avg_prob, median_prob, avg_prob_norm, median_prob_norm


def _rank_confidence_and_indices(
    config,
    selection: dict[str, Any],
    labels: list[str],
    stacked: np.ndarray,
    selected_models: list[str],
    avg_prob_norm: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Compute rank-confidence outputs and strategy prediction indices."""
    # Apply frozen probability multipliers before assessing rank support for the final strategy class.
    strategy_probabilities = apply_class_probability_multipliers(
        avg_prob_norm, labels, selection.get("class_probability_multipliers")
    )
    strategy_indices = np.argmax(strategy_probabilities, axis=1)
    rank_confidence = classify_ensemble_rank_based(
        stacked,
        class_names=labels,
        model_names=selected_models,
        predicted_class_indices=strategy_indices,
        confidence_thresholds=rank_confidence_thresholds_from_config(config),
        minimum_models=int(getattr(config, "rank_confidence_minimum_models", 3)),
    )
    return rank_confidence, strategy_indices


def _build_parcel_ranking_frame(
    config,
    labels: list[str],
    y_true: pd.Series,
    id_col: pd.Series,
    avg_prob: np.ndarray,
    median_prob: np.ndarray,
    avg_prob_norm: np.ndarray,
    median_prob_norm: np.ndarray,
    avg_rank: np.ndarray,
    median_rank: np.ndarray,
    strategy_indices: np.ndarray,
    rank_confidence: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Build row-level parcel ranking outputs."""
    labels_arr = np.asarray(labels)
    best_idx_prob_avg = np.nanargmax(avg_prob_norm, axis=1)
    best_idx_prob_median = np.nanargmax(median_prob_norm, axis=1)
    # Smaller ranks are better, whereas probability-based winners maximize their scores.
    best_idx_rank_avg = np.nanargmin(avg_rank, axis=1)
    best_idx_rank_median = np.nanargmin(median_rank, axis=1)
    true_values = y_true.astype(str).values
    predicted_values = labels_arr[strategy_indices]
    return pd.DataFrame(
        {
            config.id_column: id_col.astype(str).values,
            "true_label": true_values,
            "best_class_by_rank_avg": labels_arr[best_idx_rank_avg],
            "best_class_by_rank_median": labels_arr[best_idx_rank_median],
            "best_class_by_prob_avg": labels_arr[best_idx_prob_avg],
            "best_class_by_prob_median": labels_arr[best_idx_prob_median],
            "best_rank_avg_value": np.nanmin(avg_rank, axis=1),
            "best_rank_median_value": np.nanmin(median_rank, axis=1),
            "best_prob_avg_value": np.nanmax(avg_prob, axis=1),
            "best_prob_median_value": np.nanmax(median_prob, axis=1),
            "predicted_class": predicted_values,
            "correct": predicted_values == true_values,
            "rank_confidence_level": rank_confidence["rank_confidence_level"],
            "rank_confidence_valid": rank_confidence["rank_confidence_valid"],
            "rank_confidence_reason": rank_confidence["rank_confidence_reason"],
            "prediction_mean_borda": rank_confidence["prediction_mean_borda"],
            "prediction_rank_range": rank_confidence["prediction_rank_range"],
            "borda_margin": rank_confidence["borda_margin"],
            "top1_agreement": rank_confidence["top1_agreement"],
            "top2_agreement": rank_confidence["top2_agreement"],
            "top3_agreement": rank_confidence["top3_agreement"],
            "prediction_mean_rank": rank_confidence["prediction_mean_rank"],
            "prediction_median_rank": rank_confidence["prediction_median_rank"],
            "prediction_rank_std": rank_confidence["prediction_rank_std"],
            "prediction_rank_iqr": rank_confidence["prediction_rank_iqr"],
            "runner_up_class": rank_confidence["runner_up_class"],
            "runner_up_borda": rank_confidence["runner_up_borda"],
            "borda_prediction": rank_confidence["borda_prediction"],
            "rank_winner_tied": rank_confidence["rank_winner_tied"],
            "rank_agrees_with_prediction": rank_confidence["rank_agrees_with_prediction"],
            "n_models_used": rank_confidence["n_models_used"],
        }
    )


def _build_ranking_summary(
    config,
    labels: list[str],
    y_true: pd.Series,
    parcel_df: pd.DataFrame,
    avg_prob_norm: np.ndarray,
    median_prob_norm: np.ndarray,
    n_models_used: int,
) -> pd.DataFrame:
    """Build method-level ranking summary metrics."""
    method_predictions = {
        "probability_average": parcel_df["best_class_by_prob_avg"].values,
        "probability_median": parcel_df["best_class_by_prob_median"].values,
        "rank_average": parcel_df["best_class_by_rank_avg"].values,
        "rank_median": parcel_df["best_class_by_rank_median"].values,
    }
    # Rank-only methods have no probability matrix and therefore do not receive ROC AUC values.
    method_probabilities = {
        "probability_average": avg_prob_norm,
        "probability_median": median_prob_norm,
        "rank_average": None,
        "rank_median": None,
    }
    y_true_arr = y_true.astype(str).values
    summary_rows = [
        _ranking_method_metrics(labels, y_true_arr, method_name, y_pred, method_probabilities.get(method_name), n_models_used)
        for method_name, y_pred in method_predictions.items()
    ]
    return pd.DataFrame(summary_rows).sort_values(by=config.scoring_primary, ascending=False)


def _ranking_method_metrics(
    labels: list[str],
    y_true_arr: np.ndarray,
    method_name: str,
    y_pred: np.ndarray,
    probabilities: np.ndarray | None,
    n_models_used: int,
) -> dict[str, Any]:
    """Calculate summary metrics for one parcel-ranking method."""
    precision_arr, recall_arr, _, support = precision_recall_fscore_support(
        y_true_arr, y_pred, average=None, zero_division=0, labels=labels
    )
    precision_w = float(np.average(precision_arr, weights=support))
    recall_w = float(np.average(recall_arr, weights=support))
    return {
        "method": method_name,
        "accuracy": float(accuracy_score(y_true_arr, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred)),
        "f1_macro": float(f1_score(y_true_arr, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true_arr, y_pred, average="weighted", zero_division=0)),
        "precision_weighted": precision_w,
        "recall_weighted": recall_w,
        "roc_auc_weighted": _weighted_roc_auc(labels, y_true_arr, probabilities),
        "n_models_used": n_models_used,
    }


def _weighted_roc_auc(labels: list[str], y_true_arr: np.ndarray, probabilities: np.ndarray | None) -> float:
    """Calculate weighted ROC AUC for binary or multiclass probability matrices."""
    if probabilities is None:
        return np.nan
    try:
        if len(labels) == 2:
            return float(roc_auc_score(y_true_arr, probabilities[:, 1]))
        return float(roc_auc_score(y_true_arr, probabilities, labels=labels, multi_class="ovr", average="weighted"))
    except ValueError:
        return np.nan


def summarize_rank_confidence(parcel_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize empirical correctness overall and by predicted class."""
    summary_columns = [
        "confidence_level",
        "parcel_count",
        "coverage",
        "correct",
        "incorrect",
        "accuracy",
        "error_rate",
        "macro_f1",
        "parcels",
        "percentage",
    ]
    class_columns = ["predicted_class", "confidence_level", "count", "accuracy", "parcels"]
    if parcel_df.empty:
        return pd.DataFrame(columns=summary_columns), pd.DataFrame(columns=class_columns)

    # Prefer guarded final confidence, with older column names retained as report fallbacks.
    level_column = next(
        (
            column
            for column in ("prediction_confidence_level", "rank_confidence_level", "confidence_level")
            if column in parcel_df
        ),
        None,
    )
    if level_column is None:
        return pd.DataFrame(columns=summary_columns), pd.DataFrame(columns=class_columns)

    total = len(parcel_df)
    summary_rows = []
    for level in ("HIGH", "MEDIUM", "LOW"):
        subset = parcel_df.loc[parcel_df[level_column] == level]
        if subset.empty:
            summary_rows.append(
                {
                    "confidence_level": level,
                    "parcel_count": 0,
                    "coverage": 0.0,
                    "correct": 0,
                    "incorrect": 0,
                    "accuracy": np.nan,
                    "error_rate": np.nan,
                    "macro_f1": np.nan,
                    "parcels": 0,
                    "percentage": 0.0,
                }
            )
            continue
        correct = int(subset["correct"].sum())
        parcel_count = len(subset)
        summary_rows.append(
            {
                "confidence_level": level,
                "parcel_count": parcel_count,
                "coverage": float(parcel_count / total),
                "correct": correct,
                "incorrect": parcel_count - correct,
                "accuracy": float(subset["correct"].mean()),
                "error_rate": float(1.0 - subset["correct"].mean()),
                "macro_f1": float(f1_score(subset["true_label"], subset["predicted_class"], average="macro", zero_division=0)),
                "parcels": parcel_count,
                "percentage": float(parcel_count / total),
            }
        )

    class_summary = (
        parcel_df.groupby(["predicted_class", level_column], observed=True)
        .agg(count=("correct", "size"), accuracy=("correct", "mean"))
        .reset_index()
        .rename(columns={level_column: "confidence_level"})
    )
    class_summary["parcels"] = class_summary["count"]
    return pd.DataFrame(summary_rows, columns=summary_columns), class_summary.reindex(columns=class_columns)


def apply_class_reliability_guard(
    parcel_df: pd.DataFrame, reliability_contract: dict[str, Any], *, combine_with_rank: bool = True
) -> pd.DataFrame:
    """Attach frozen OOF class reliability and optionally combine it with rank confidence."""
    if parcel_df.empty:
        return parcel_df.copy()
    result = parcel_df.copy()
    # Lookup uses the frozen OOF contract; these evaluation rows do not refit class reliability.
    reliability = get_class_reliability(result["predicted_class"].to_numpy(), reliability_contract)
    for name, values in reliability.items():
        result[name] = values
    if combine_with_rank:
        final = combine_confidence_components(
            result["rank_confidence_level"].to_numpy(),
            result["rank_confidence_valid"].to_numpy(),
            result["rank_confidence_reason"].to_numpy(),
            reliability["class_reliability_level"],
            reliability["class_reliability_valid"],
            reliability["class_reliability_reason"],
        )
        for name, values in final.items():
            result[name] = values
    return result
