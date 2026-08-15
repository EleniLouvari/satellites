"""Selection logic for single-model versus soft-voting strategies and ranking helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ClassificationPipelineConfig
from .metrics import score_predictions
from .persistence import print_formatted_txt, save_frame_csv


# Selection scores use the same metric definitions as evaluation to avoid ranking drift.
def rank_probability_models_from_cv(
    config: ClassificationPipelineConfig,
    cv_metrics_df: pd.DataFrame,
    available_probability_models: set[str],
) -> list[str]:
    """Return probability-capable models in frozen cross-validation rank order."""
    required_columns = {"model", "cv_ranking_metric"}
    missing = required_columns.difference(cv_metrics_df.columns)
    if missing:
        raise ValueError(f"CV metrics are missing required columns: {sorted(missing)}")
    ranked = cv_metrics_df.sort_values(
        ["cv_ranking_metric", "best_cv_score"],
        ascending=[False, False],
    )
    models = [model for model in ranked["model"].tolist() if model in available_probability_models]
    if config.top_voting_models is not None:
        models = models[: config.top_voting_models]
    return models


def evaluate_voting_candidate(
    config: ClassificationPipelineConfig,
    cv_metrics_df: pd.DataFrame,
    train_metrics_df: pd.DataFrame,
    test_metrics_df: pd.DataFrame,
    probability_cache_train: dict[str, np.ndarray],
    probability_cache_test: dict[str, np.ndarray],
    y_train: pd.Series,
    y_test: pd.Series,
    labels: list[str],
) -> dict[str, object]:
    """Evaluate optional soft-voting ensemble and return final selection metadata."""
    # Import plotting helpers lazily to avoid unnecessary import overhead.
    from ..visuals.plots import save_confusion_matrix_plot, save_model_comparison_plot

    # Candidate order and final strategy are determined only by training CV.
    # Holdout probabilities below are used solely to report the frozen strategy.
    prob_models = rank_probability_models_from_cv(config, cv_metrics_df, set(probability_cache_test))
    if not prob_models:
        raise RuntimeError("No cross-validated model with predict_proba is available for final classification.")
    best_single_model = prob_models[0]
    best_single_score = float(cv_metrics_df.loc[cv_metrics_df["model"] == best_single_model, "cv_ranking_metric"].iloc[0])
    selection = {
        "selection_type": "single_model",
        "selected_models": [best_single_model],
        "selected_metric": f"cv_{config.cv_ranking_method}",
        "selected_score": best_single_score,
        "selection_source": "cross_validation",
        "labels": labels,
    }
    if len(prob_models) < 2:
        if config.selection_type == "soft_voting":
            print_formatted_txt(
                (
                    "Soft voting requested, but fewer than 2 probability-capable models are available. "
                    "Falling back to single_model."
                ),
                "WARNING",
            )
        return selection

    averaged_probs_test = np.zeros_like(probability_cache_test[prob_models[0]], dtype=np.float64)
    for model_name in prob_models:
        averaged_probs_test += probability_cache_test[model_name]
    averaged_probs_test /= len(prob_models)
    voting_predictions_test = np.asarray(labels)[np.argmax(averaged_probs_test, axis=1)]
    voting_metrics_test = score_predictions(y_test, voting_predictions_test, averaged_probs_test, labels, "soft_voting")

    averaged_probs_train = np.zeros_like(probability_cache_train[prob_models[0]], dtype=np.float64)
    for model_name in prob_models:
        averaged_probs_train += probability_cache_train[model_name]
    averaged_probs_train /= len(prob_models)
    voting_predictions_train = np.asarray(labels)[np.argmax(averaged_probs_train, axis=1)]
    voting_metrics_train = score_predictions(y_train, voting_predictions_train, averaged_probs_train, labels, "soft_voting")

    train_with_voting = pd.concat([train_metrics_df, pd.DataFrame([voting_metrics_train])], ignore_index=True)
    test_with_voting = pd.concat([test_metrics_df, pd.DataFrame([voting_metrics_test])], ignore_index=True)
    train_with_voting = sort_metrics_with_voting_last(train_with_voting, config)
    test_with_voting = sort_metrics_with_voting_last(test_with_voting, config)
    save_frame_csv(train_with_voting, config.evaluate_dir / "model_metrics_train_with_voting.csv")
    save_frame_csv(test_with_voting, config.evaluate_dir / "model_metrics_test_with_voting.csv")
    save_model_comparison_plot(
        test_with_voting,
        config.evaluate_dir / "plots" / "model_comparison_with_voting.png",
        config.scoring_primary,
    )
    save_confusion_matrix_plot(
        y_test,
        voting_predictions_test,
        labels,
        config.evaluate_dir / "confusion_matrices" / "soft_voting_confusion_matrix.png",
        "Confusion Matrix: soft_voting",
    )

    if config.selection_type == "soft_voting":
        selection = {
            "selection_type": "soft_voting",
            "selected_models": prob_models,
            "selected_metric": f"cv_{config.cv_ranking_method}",
            "selected_score": float(cv_metrics_df.loc[cv_metrics_df["model"].isin(prob_models), "cv_ranking_metric", ].mean()),
            "selection_source": "cross_validation",
            "labels": labels,
        }
    return selection


def fit_and_predict_selected_strategy(
    config: ClassificationPipelineConfig,
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_all: pd.DataFrame,
    selection: dict[str, object],
    model_specs: dict[str, dict[str, object]],
    numeric_features: list[str],
    categorical_features: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Fit selected model strategy on labeled rows and predict full-dataset probabilities."""
    # Build and fit each selected estimator before probability averaging.
    from .models import build_estimator_by_name

    fitted_probabilities = []
    for model_name in selection["selected_models"]:
        estimator = build_estimator_by_name(
            model_name,
            config,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            num_classes=len(selection["labels"]),
        )
        estimator.set_params(**model_specs[model_name]["best_params"])
        estimator.fit(X_fit, y_fit)
        fitted_probabilities.append(estimator.predict_proba(X_all))

    averaged_probs = np.mean(fitted_probabilities, axis=0)
    prediction_encoded = np.argmax(averaged_probs, axis=1)
    predictions = np.asarray(selection["labels"])[prediction_encoded]
    return predictions, averaged_probs


def sort_metrics_with_voting_last(
    metrics_df: pd.DataFrame,
    config: ClassificationPipelineConfig,
) -> pd.DataFrame:
    """Sort metric rows while forcing optional soft-voting row to the end."""
    # Delegate sorting behavior to shared helper with voting-placement option.
    return _sort_metrics(metrics_df, config, keep_voting_last=True)


def sort_metrics_without_voting(
    metrics_df: pd.DataFrame,
    config: ClassificationPipelineConfig,
) -> pd.DataFrame:
    """Sort metric rows without forcing soft-voting row placement."""
    # Reuse shared sorting logic with standard ordering behavior.
    return _sort_metrics(metrics_df, config, keep_voting_last=False)


def _sort_metrics(
    metrics_df: pd.DataFrame,
    config: ClassificationPipelineConfig,
    keep_voting_last: bool,
) -> pd.DataFrame:
    """Apply consistent model metric sorting with optional voting-row placement."""
    # Keep soft-voting summary at the bottom when requested for report readability.
    if not keep_voting_last:
        return metrics_df.sort_values(config.scoring_primary, ascending=True).reset_index(drop=True)
    voting_df = metrics_df[metrics_df["model"] == "soft_voting"].copy()
    base_df = metrics_df[metrics_df["model"] != "soft_voting"].copy()
    base_df = base_df.sort_values(config.scoring_primary, ascending=True)
    return pd.concat([base_df, voting_df], ignore_index=True)
