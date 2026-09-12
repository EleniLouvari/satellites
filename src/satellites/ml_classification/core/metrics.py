"""Metrics, validation, and shared modeling-context helpers for classification tasks."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from satellites.shared.io import read_data
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

# Metric helpers in this module return serializable values suitable for reports and artifacts.
from sklearn.preprocessing import LabelEncoder

from .config import ClassificationPipelineConfig
from .persistence import load_json


def validate_input(df: pd.DataFrame, config: ClassificationPipelineConfig) -> None:
    """Validate required columns, labels, and minimum class counts for training."""
    # Verify all configured feature columns are available in the input frame.
    missing_features = [column for column in config.feature_columns if column not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")
    if config.target_column not in df.columns:
        raise ValueError(f"Target column '{config.target_column}' was not found.")
    if df.empty:
        raise ValueError("Input dataframe is empty.")
    labeled = df[df[config.target_column].notna()]
    if labeled.empty:
        raise ValueError("No labeled rows found. The target column only contains missing values.")
    class_counts = labeled[config.target_column].astype(str).value_counts()
    if len(class_counts) < 2:
        raise ValueError("Classification requires at least two target classes.")
    if (class_counts < config.cv_folds).any():
        too_small = class_counts[class_counts < config.cv_folds].to_dict()
        raise ValueError(f"Each class needs at least {config.cv_folds} rows for stratified CV. Found: {too_small}")


def optimize_dataframe(
    df: pd.DataFrame, config: ClassificationPipelineConfig, feature_columns: list[str] | None = None
) -> pd.DataFrame:
    """Cast feature columns to optimized numeric/category dtypes for efficiency."""
    # Work on a copy to keep caller-provided data unchanged.
    optimized = df.copy()
    for column in feature_columns or config.feature_columns:
        if pd.api.types.is_numeric_dtype(optimized[column]):
            optimized[column] = pd.to_numeric(optimized[column], errors="coerce").astype(config.float_dtype)
        else:
            optimized[column] = optimized[column].astype("category")
    return optimized


def build_feature_profile(
    df: pd.DataFrame, active_features: list[str], config: ClassificationPipelineConfig, feature_columns: list[str] | None = None
) -> pd.DataFrame:
    """Build a dataframe that summarizes feature dtypes, null rates, and activity."""
    # Collect one diagnostics row per requested feature column.
    rows = []
    for column in feature_columns or config.feature_columns:
        rows.append(
            {
                "feature": column,
                "dtype": str(df[column].dtype),
                "is_active": column in active_features,
                "null_pct": round(float(df[column].isna().mean() * 100), 4),
                "n_unique": int(df[column].nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows).sort_values(["is_active", "null_pct", "feature"], ascending=[False, False, True])


def load_modeling_context(config: ClassificationPipelineConfig) -> dict[str, Any]:
    """Load persisted check/prepare artifacts and rebuild model context objects."""
    # Recreate feature groups and label encoder from saved preparation artifacts.
    check_summary = load_json(config.check_dir / "check_summary.json")
    prepare_summary = load_json(config.prepare_dir / "prepare_summary.json")
    label_mapping = read_data(str(config.prepare_dir / "label_mapping.csv"), watch_curly_brackets=False)
    active_features = prepare_summary["active_features"]
    numeric_features = [column for column in check_summary["numeric_features"] if column in active_features]
    categorical_features = [column for column in check_summary["categorical_features"] if column in active_features]
    label_encoder = LabelEncoder()
    label_encoder.classes_ = label_mapping["target_label"].astype(str).to_numpy()
    return {
        "check_summary": check_summary,
        "prepare_summary": prepare_summary,
        "label_mapping": label_mapping,
        "active_features": active_features,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "label_encoder": label_encoder,
        "labels": prepare_summary["target_labels"],
    }


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


def score_predictions(
    y_true: pd.Series, y_pred: np.ndarray, probabilities: np.ndarray | None, labels: list[str], model_name: str
) -> dict[str, Any]:
    """Compute common classification metrics from labels and optional probabilities."""
    # Aggregate core point-estimate metrics used by training and evaluation reports.
    metrics = {
        "model": model_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    if probabilities is not None:
        try:
            if len(labels) == 2:
                metrics["roc_auc_weighted"] = float(roc_auc_score(y_true, probabilities[:, 1]))
            else:
                metrics["roc_auc_weighted"] = float(
                    roc_auc_score(y_true, probabilities, labels=labels, multi_class="ovr", average="weighted")
                )
        except ValueError:
            metrics["roc_auc_weighted"] = np.nan
    else:
        metrics["roc_auc_weighted"] = np.nan
    return metrics
