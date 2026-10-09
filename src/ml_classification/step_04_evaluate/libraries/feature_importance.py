"""Choose models for interpretation and extract their feature-importance tables."""

from __future__ import annotations

import logging
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
    fitted_preprocessor, fitted_model = _resolve_fitted_components(estimator)
    importances = _resolve_importance_values(fitted_model)
    if importances is None:
        return None

    feature_names = _resolve_feature_names(estimator, fitted_preprocessor, len(importances))
    # Strip the transformer prefix while retaining the feature or encoded-category suffix.
    cleaned_feature_names = [name.split("__", maxsplit=1)[-1] for name in feature_names]
    # Limit both arrays to their common length when an estimator exposes incomplete feature metadata.
    usable_length = min(len(cleaned_feature_names), len(importances))
    if usable_length == 0:
        return None

    importance_df = pd.DataFrame({"feature": cleaned_feature_names[:usable_length], "importance": importances[:usable_length]})
    importance_df["importance_abs"] = importance_df["importance"].abs()
    return importance_df.sort_values("importance_abs", ascending=False).reset_index(drop=True)


def _resolve_fitted_components(estimator: Any) -> tuple[Any, Any]:
    """Resolve the fitted preprocessor and model from pipeline/meta-estimator wrappers."""
    fitted_preprocessor = None
    fitted_model = estimator
    if hasattr(estimator, "named_steps"):
        fitted_preprocessor = estimator.named_steps.get("preprocessor")
        fitted_model = estimator.named_steps.get("model", estimator)
    return fitted_preprocessor, _unwrap_model_for_importance(fitted_model)


def _unwrap_model_for_importance(model: Any) -> Any:
    """Unwrap wrappers that proxy feature-importance attributes."""
    if model.__class__.__name__ == "ContiguousLabelClassifier" and hasattr(model, "estimator_"):
        return model.estimator_
    if hasattr(model, "base_estimator") and hasattr(model.base_estimator, "feature_importances_"):
        return model.base_estimator
    return model


def _resolve_importance_values(model: Any) -> np.ndarray | None:
    """Return feature importance magnitudes when the estimator exposes them."""
    if hasattr(model, "feature_importances_"):
        return np.asarray(model.feature_importances_, dtype=float)
    if hasattr(model, "coef_"):
        coefficients = np.asarray(model.coef_, dtype=float)
        importances = np.abs(coefficients)
        # Average coefficient magnitudes across classes so opposite signs do not cancel.
        return importances.mean(axis=0) if importances.ndim > 1 else importances
    return None


def _resolve_feature_names(estimator: Any, preprocessor: Any, n_importances: int) -> list[str]:
    """Resolve output feature names for interpretability tables."""
    # Prefer transformed names because one-hot encoding can expand the original input columns.
    names = _try_preprocessor_feature_names(preprocessor)
    if names is not None:
        return names
    if hasattr(estimator, "feature_names_in_"):
        return list(estimator.feature_names_in_)
    return [f"feature_{index}" for index in range(n_importances)]


def _try_preprocessor_feature_names(preprocessor: Any) -> list[str] | None:
    """Safely call ``get_feature_names_out`` when present."""
    if preprocessor is None or not hasattr(preprocessor, "get_feature_names_out"):
        return None
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        logging.getLogger(__name__).debug("Error: extract_feature_importance_frame failed; using its fallback.", exc_info=True)
        return None
