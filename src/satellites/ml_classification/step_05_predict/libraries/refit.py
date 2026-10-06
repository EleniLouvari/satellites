"""Reconstruct the frozen selected strategy, refit it, and predict all rows."""

from __future__ import annotations

import numpy as np
import pandas as pd

from satellites.ml_classification.shared.config.config import ClassificationPipelineConfig
from satellites.ml_classification.shared.probabilities import apply_class_probability_multipliers


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
    from satellites.ml_classification.shared.models.models import build_estimator_by_name

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
            raise RuntimeError(f"Model {model_name!r} probability columns are not aligned with the canonical class order.")
        # Predict probabilities for all rows and validate the output shape and values.
        probabilities = np.asarray(estimator.predict_proba(X_all), dtype=np.float64)
        if probabilities.ndim != 2 or probabilities.shape[1] != len(selection["labels"]):
            raise RuntimeError(
                f"Model {model_name!r} returned {probabilities.shape} probabilities; expected "
                f"(n_rows, {len(selection['labels'])})."
            )
        if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities.sum(axis=1) <= 0):
            raise RuntimeError(f"Model {model_name!r} returned malformed or non-finite probabilities.")
        fitted_probabilities[model_name] = probabilities

    # Average the selected model probabilities and apply any learned class multipliers.
    averaged_probs = np.mean(list(fitted_probabilities.values()), axis=0)
    # Apply class multipliers learned from out-of-fold predictions if they exist.
    averaged_probs = apply_class_probability_multipliers(
        averaged_probs, list(selection["labels"]), selection.get("class_probability_multipliers")
    )
    # Convert averaged probabilities to hard labels using argmax and return all outputs.
    prediction_encoded = np.argmax(averaged_probs, axis=1)
    # Convert the integer-encoded predictions back to the original label strings.
    predictions = np.asarray(selection["labels"])[prediction_encoded]
    return predictions, averaged_probs, fitted_probabilities
