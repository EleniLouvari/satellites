"""Choose models for interpretation and extract their feature-importance tables."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def select_top_models_for_interpretability(metrics_df: pd.DataFrame, top_n: int) -> list[str]:
    """Return top-ranked base model names for interpretability plotting."""
    # Exclude soft-voting aggregate rows from base-model interpretability selection.
    if top_n <= 0 or metrics_df.empty:
        return []
    base_df = metrics_df.loc[metrics_df["model"] != "soft_voting"].copy()
    if base_df.empty:
        return []
    if "cv_ranking_metric" in base_df.columns:
        ranked_df = base_df.sort_values("cv_ranking_metric", ascending=False)
    else:
        ranked_df = base_df.sort_values("test_metric" if "test_metric" in base_df.columns else "f1_macro", ascending=False)
    if "cv_ranking_metric" not in base_df.columns and "test_metric" not in base_df.columns:
        metric_columns = [
            column for column in ("f1_macro", "balanced_accuracy", "f1_weighted", "accuracy") if column in base_df.columns
        ]
        if metric_columns:
            ranked_df = base_df.sort_values(metric_columns[0], ascending=False)
    return ranked_df["model"].head(top_n).tolist()


def extract_feature_importance_frame(estimator: Any) -> pd.DataFrame | None:
    """Extract model feature importances or coefficient magnitudes into a dataframe."""
    # Unwrap pipeline/meta-estimator structures to locate the fitted core estimator.
    fitted_preprocessor = None
    fitted_model = estimator
    if hasattr(estimator, "named_steps"):
        fitted_preprocessor = estimator.named_steps.get("preprocessor")
        fitted_model = estimator.named_steps.get("model", estimator)
    if fitted_model.__class__.__name__ == "ContiguousLabelClassifier" and hasattr(fitted_model, "estimator_"):
        fitted_model = fitted_model.estimator_
    elif hasattr(fitted_model, "base_estimator") and hasattr(fitted_model.base_estimator, "feature_importances_"):
        fitted_model = fitted_model.base_estimator

    importances = None
    if hasattr(fitted_model, "feature_importances_"):
        importances = np.asarray(fitted_model.feature_importances_, dtype=float)
    elif hasattr(fitted_model, "coef_"):
        coefficients = np.asarray(fitted_model.coef_, dtype=float)
        importances = np.abs(coefficients)
        if importances.ndim > 1:
            importances = importances.mean(axis=0)
    if importances is None:
        return None

    feature_names = None
    if fitted_preprocessor is not None and hasattr(fitted_preprocessor, "get_feature_names_out"):
        try:
            feature_names = list(fitted_preprocessor.get_feature_names_out())
        except Exception:
            feature_names = None
    if feature_names is None and hasattr(estimator, "feature_names_in_"):
        feature_names = list(estimator.feature_names_in_)
    if feature_names is None:
        feature_names = [f"feature_{idx}" for idx in range(len(importances))]

    cleaned_feature_names = [name.split("__", 1)[-1] for name in feature_names]
    usable_length = min(len(cleaned_feature_names), len(importances))
    if usable_length == 0:
        return None

    importance_df = pd.DataFrame({"feature": cleaned_feature_names[:usable_length], "importance": importances[:usable_length]})
    importance_df["importance_abs"] = importance_df["importance"].abs()
    return importance_df.sort_values("importance_abs", ascending=False).reset_index(drop=True)
