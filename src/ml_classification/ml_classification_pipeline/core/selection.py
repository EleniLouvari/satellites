"""Selection logic for single-model versus soft-voting strategies and ranking helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from .config import ClassificationPipelineConfig
from .metrics import score_predictions
from .persistence import print_formatted_txt, save_frame_csv


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


def optimize_probability_multipliers(
    probabilities: np.ndarray, y_true: np.ndarray, labels: list[str], config: ClassificationPipelineConfig
) -> tuple[dict[str, float], dict[str, float]]:
    """Learn class multipliers from out-of-fold predictions using coordinate search."""
    # Normalize inputs and compute baseline metrics used as optimization anchors.
    y_true = np.asarray(y_true).astype(str)
    label_array = np.asarray(labels).astype(str)
    baseline_pred = label_array[np.argmax(probabilities, axis=1)]
    baseline_f1 = float(f1_score(y_true, baseline_pred, average="macro", zero_division=0))
    baseline_accuracy = float(accuracy_score(y_true, baseline_pred))
    minimum_accuracy = baseline_accuracy - float(config.probability_optimization_max_accuracy_drop)
    multipliers = {str(label): 1.0 for label in labels}
    best_f1 = baseline_f1
    best_accuracy = baseline_accuracy
    for _ in range(config.probability_optimization_iterations):
        changed = False
        for label in labels:
            current = multipliers[str(label)]
            local_best = (best_f1, best_accuracy, current)
            for candidate in config.probability_multiplier_grid:
                trial = dict(multipliers)
                trial[str(label)] = float(candidate)
                adjusted = apply_class_probability_multipliers(probabilities, labels, trial)
                predicted = label_array[np.argmax(adjusted, axis=1)]
                accuracy = float(accuracy_score(y_true, predicted))
                score = float(f1_score(y_true, predicted, average="macro", zero_division=0))
                if accuracy >= minimum_accuracy and (score, accuracy) > local_best[:2]:
                    local_best = (score, accuracy, float(candidate))
            if local_best[2] != current:
                multipliers[str(label)] = local_best[2]
                best_f1, best_accuracy = local_best[:2]
                changed = True
        if not changed:
            break
    return multipliers, {
        "baseline_oof_macro_f1": baseline_f1,
        "optimized_oof_macro_f1": best_f1,
        "baseline_oof_accuracy": baseline_accuracy,
        "optimized_oof_accuracy": best_accuracy,
    }


# Selection scores use the same metric definitions as evaluation to avoid ranking drift.
def rank_probability_models_from_cv(
    config: ClassificationPipelineConfig, cv_metrics_df: pd.DataFrame, available_probability_models: set[str]
) -> list[str]:
    """Return probability-capable models in frozen cross-validation rank order."""
    # Ensure required CV summary columns are present and then select
    # probability-capable models in frozen CV rank order.
    required_columns = {"model", "cv_ranking_metric"}
    missing = required_columns.difference(cv_metrics_df.columns)
    if missing:
        raise ValueError(f"CV metrics are missing required columns: {sorted(missing)}")
    ranked = cv_metrics_df.sort_values(["cv_ranking_metric", "best_cv_score"], ascending=[False, False])
    models = [model for model in ranked["model"].tolist() if model in available_probability_models]
    if config.top_voting_models is not None:
        models = models[: config.top_voting_models]
    return models


def _average_cached_probabilities(probability_cache: dict[str, np.ndarray], model_names: list[str]) -> np.ndarray:
    """Average aligned class probabilities for a non-empty model list."""
    if not model_names:
        raise ValueError("At least one probability model is required.")
    return np.mean([probability_cache[name] for name in model_names], axis=0, dtype=np.float64)


def _build_selection(
    config: ClassificationPipelineConfig,
    cv_metrics_df: pd.DataFrame,
    selected_models: list[str],
    labels: list[str],
    selection_type: str,
) -> dict[str, object]:
    """Build consistent cross-validation selection metadata."""
    # Derive a compact selection metadata dictionary that records selected
    # models, the metric used for ranking, and the label ordering.
    selected_scores = cv_metrics_df.loc[cv_metrics_df["model"].isin(selected_models), "cv_ranking_metric"]
    return {
        "selection_type": selection_type,
        "selected_models": selected_models,
        "selected_metric": f"cv_{config.cv_ranking_method}",
        "selected_score": float(selected_scores.mean()),
        "selection_source": "cross_validation",
        "labels": labels,
    }


def _attach_probability_optimization(
    selection: dict[str, object],
    config: ClassificationPipelineConfig,
    selected_models: list[str],
    oof_probability_cache: dict[str, np.ndarray] | None,
    y_train: pd.Series,
    labels: list[str],
) -> None:
    """Learn and attach OOF class multipliers when the option is enabled."""
    # When enabled, learn per-class multipliers from out-of-fold predictions
    # and attach results and diagnostics to the selection metadata.
    if not config.optimize_class_probabilities:
        return
    missing_models = [name for name in selected_models if not oof_probability_cache or name not in oof_probability_cache]
    if missing_models:
        raise RuntimeError(
            "Probability optimization is enabled, but out-of-fold probabilities are missing for: "
            f"{missing_models}"
        )
    oof_probabilities = _average_cached_probabilities(oof_probability_cache, selected_models)
    multipliers, diagnostics = optimize_probability_multipliers(oof_probabilities, np.asarray(y_train), labels, config)
    selection["class_probability_multipliers"] = multipliers
    selection["probability_optimization"] = diagnostics


def _score_probability_strategy(
    probabilities: np.ndarray,
    y_true: pd.Series,
    labels: list[str],
    model_name: str,
) -> tuple[np.ndarray, dict[str, object]]:
    """Convert probabilities to labels and calculate the standard metrics row."""
    # Convert probabilities to hard labels using argmax and then compute all
    # standard metrics (including optional ROC AUC when probabilities are present).
    predictions = np.asarray(labels)[np.argmax(probabilities, axis=1)]
    return predictions, score_predictions(y_true, predictions, probabilities, labels, model_name)


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
    oof_probability_cache: dict[str, np.ndarray] | None = None,
) -> dict[str, object]:
    """Evaluate optional soft-voting ensemble and return final selection metadata."""
    # Import plotting helpers lazily to avoid unnecessary import overhead.
    from ..visuals.plots import save_confusion_matrix_plot, save_model_comparison_plot

    # Candidate order and final strategy are determined only by training CV.
    # Holdout probabilities below are used solely to report the frozen strategy.
    prob_models = rank_probability_models_from_cv(config, cv_metrics_df, set(probability_cache_test))
    if not prob_models:
        raise RuntimeError("No cross-validated model with predict_proba is available for final classification.")
    use_soft_voting = config.selection_type == "soft_voting" and len(prob_models) >= 2
    if config.selection_type == "soft_voting" and not use_soft_voting:
        print_formatted_txt(
            "Soft voting requested, but fewer than 2 probability-capable models are available. "
            "Falling back to single_model.",
            "WARNING",
        )
    selected_models = prob_models if use_soft_voting else [prob_models[0]]
    selection_type = "soft_voting" if use_soft_voting else "single_model"
    selection = _build_selection(config, cv_metrics_df, selected_models, labels, selection_type)
    _attach_probability_optimization(selection, config, selected_models, oof_probability_cache, y_train, labels)

    # There is no ensemble to report when only one probability model is available.
    if len(prob_models) < 2:
        return selection

    # Average per-model probabilities on train and test then optionally apply
    # class multipliers learned from OOF predictions.
    voting_probabilities = {
        "train": _average_cached_probabilities(probability_cache_train, prob_models),
        "test": _average_cached_probabilities(probability_cache_test, prob_models),
    }
    if use_soft_voting:
        for split_name, probabilities in voting_probabilities.items():
            voting_probabilities[split_name] = apply_class_probability_multipliers(
                probabilities, labels, selection.get("class_probability_multipliers")
            )
    voting_predictions_train, voting_metrics_train = _score_probability_strategy(voting_probabilities["train"], y_train, labels, "soft_voting")
    voting_predictions_test, voting_metrics_test = _score_probability_strategy(voting_probabilities["test"], y_test, labels, "soft_voting")

    # Append the soft-voting summary row to the existing metric tables and
    # persist artifacts used in the evaluation report.
    train_with_voting = pd.concat([train_metrics_df, pd.DataFrame([voting_metrics_train])], ignore_index=True)
    test_with_voting = pd.concat([test_metrics_df, pd.DataFrame([voting_metrics_test])], ignore_index=True)
    train_with_voting = sort_metrics_with_voting_last(train_with_voting, config)
    test_with_voting = sort_metrics_with_voting_last(test_with_voting, config)
    save_frame_csv(train_with_voting, config.evaluate_dir / "model_metrics_train_with_voting.csv")
    save_frame_csv(test_with_voting, config.evaluate_dir / "model_metrics_test_with_voting.csv")
    save_model_comparison_plot(test_with_voting, config.evaluate_dir / "plots" / "model_comparison_with_voting.png", config.scoring_primary)
    save_confusion_matrix_plot(
        y_test,
        voting_predictions_test,
        labels,
        config.evaluate_dir / "confusion_matrices" / "soft_voting_confusion_matrix.png",
        "Confusion Matrix: soft_voting",
    )

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
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Fit selected model strategy on labeled rows and predict full-dataset probabilities."""
    # Build, tune (via provided specs), and fit each selected estimator, then
    # average their probabilities to produce final frozen predictions.
    from .models import build_estimator_by_name

    fitted_probabilities: dict[str, np.ndarray] = {}
    for model_name in selection["selected_models"]:
        estimator = build_estimator_by_name(
            model_name,
            config,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            num_classes=len(selection["labels"]),
        )
        # Apply the chosen hyperparameters discovered during CV.
        estimator.set_params(**model_specs[model_name]["best_params"])
        estimator.fit(X_fit, y_fit)
        expected_classes = np.arange(len(selection["labels"]), dtype=int)
        estimator_classes = getattr(estimator, "classes_", expected_classes)
        if not np.array_equal(np.asarray(estimator_classes, dtype=int), expected_classes):
            raise RuntimeError(
                f"Model {model_name!r} probability columns are not aligned with the canonical class order."
            )
        probabilities = np.asarray(estimator.predict_proba(X_all), dtype=np.float64)
        if probabilities.ndim != 2 or probabilities.shape[1] != len(selection["labels"]):
            raise RuntimeError(
                f"Model {model_name!r} returned {probabilities.shape} probabilities; expected "
                f"(n_rows, {len(selection['labels'])})."
            )
        fitted_probabilities[model_name] = probabilities

    averaged_probs = np.mean(list(fitted_probabilities.values()), axis=0)
    averaged_probs = apply_class_probability_multipliers(averaged_probs, list(selection["labels"]), selection.get("class_probability_multipliers"))
    prediction_encoded = np.argmax(averaged_probs, axis=1)
    predictions = np.asarray(selection["labels"])[prediction_encoded]
    return predictions, averaged_probs, fitted_probabilities


def sort_metrics_with_voting_last(metrics_df: pd.DataFrame, config: ClassificationPipelineConfig) -> pd.DataFrame:
    """Sort metric rows while forcing optional soft-voting row to the end."""
    # Delegate sorting behavior to shared helper with voting-placement option.
    return _sort_metrics(metrics_df, config, keep_voting_last=True)


def sort_metrics_without_voting(metrics_df: pd.DataFrame, config: ClassificationPipelineConfig) -> pd.DataFrame:
    """Sort metric rows without forcing soft-voting row placement."""
    # Reuse shared sorting logic with standard ordering behavior.
    return _sort_metrics(metrics_df, config, keep_voting_last=False)


def _sort_metrics(metrics_df: pd.DataFrame, config: ClassificationPipelineConfig, keep_voting_last: bool) -> pd.DataFrame:
    """Apply consistent model metric sorting with optional voting-row placement."""
    # Keep soft-voting summary at the bottom when requested for report readability.
    if not keep_voting_last:
        return metrics_df.sort_values(config.scoring_primary, ascending=False).reset_index(drop=True)
    voting_df = metrics_df[metrics_df["model"] == "soft_voting"].copy()
    base_df = metrics_df[metrics_df["model"] != "soft_voting"].copy()
    base_df = base_df.sort_values(config.scoring_primary, ascending=False)
    return pd.concat([base_df, voting_df], ignore_index=True)
