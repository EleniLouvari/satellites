"""Interpretability plotting utilities including feature importance and SHAP summaries.

SHAP support is optional: functions attempt a lazy import and provide clear
errors when the environment does not include the `shap` package. The helpers
write image files and clean up matplotlib state to avoid interfering with
other plotting in the same process.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from satellites.ml_classification.shared.logging import print_formatted_txt
from satellites.ml_classification.shared.persistence import save_frame_csv
from satellites.ml_classification.shared.reports.plotting import _save_figure as _save_figure
from satellites.ml_classification.step_04_evaluate.libraries.feature_importance import (
    extract_feature_importance_frame,
    select_top_models_for_interpretability,
)

# Interpretability routines degrade gracefully when optional SHAP support is unavailable.


def save_feature_importance_plot(importance_df: pd.DataFrame, output_path: str | Path, model_name: str, top_n: int = 15) -> None:
    """Save a horizontal bar chart of top feature importances."""
    # Rank by absolute importance so positive/negative signs do not hide impact.
    if importance_df.empty:
        return
    plot_df = importance_df.head(top_n).sort_values("importance_abs", ascending=True)
    fig, ax = plt.subplots(figsize=(12, max(4, len(plot_df) * 0.45)))
    ax.barh(plot_df["feature"], plot_df["importance_abs"], color="#476C9B")
    ax.set_title(f"Top Feature Importance: {model_name}")
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    _save_figure(fig, output_path)


def save_shap_summary_plot(
    estimator, X_sample: pd.DataFrame, output_path: str | Path, model_name: str, max_display: int = 15
) -> None:
    """Compute and save a SHAP summary visualization for an estimator."""
    # Import SHAP lazily so the dependency remains optional. Raising a
    # RuntimeError here surfaces the missing dependency clearly to callers
    # that expect interpretability outputs.
    try:
        import shap
    except Exception as exc:
        raise RuntimeError("SHAP is not installed in the current environment.") from exc

    fitted_preprocessor = estimator.named_steps.get("preprocessor") if hasattr(estimator, "named_steps") else None
    fitted_model = estimator.named_steps.get("model", estimator) if hasattr(estimator, "named_steps") else estimator
    if fitted_model.__class__.__name__ == "ContiguousLabelClassifier" and hasattr(fitted_model, "estimator_"):
        fitted_model = fitted_model.estimator_

    transformed = fitted_preprocessor.transform(X_sample) if fitted_preprocessor is not None else X_sample
    feature_names = None
    if fitted_preprocessor is not None and hasattr(fitted_preprocessor, "get_feature_names_out"):
        try:
            feature_names = list(fitted_preprocessor.get_feature_names_out())
        except Exception:
            feature_names = None
    if feature_names is None and hasattr(X_sample, "columns"):
        feature_names = list(X_sample.columns)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    transformed = np.asarray(transformed)
    if transformed.ndim != 2 or transformed.shape[0] == 0:
        return
    if feature_names is None:
        feature_names = [f"feature_{idx}" for idx in range(transformed.shape[1])]
    feature_names = [name.split("__", 1)[-1] for name in feature_names]

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=r"The NumPy global RNG was seeded by calling `np\.random\.seed`.*", category=FutureWarning
        )
        background = transformed[: min(100, len(transformed))]
        eval_data = transformed[: min(200, len(transformed))]

        shap_values = None
        try:
            shap_values = shap.Explainer(fitted_model).shap_values(eval_data)
        except Exception:
            shap_values = None
        if shap_values is None and _supports_tree_shap(fitted_model):
            try:
                shap_values = shap.TreeExplainer(fitted_model, feature_names=feature_names).shap_values(eval_data)
            except Exception:
                shap_values = None
        if shap_values is None and hasattr(fitted_model, "predict_proba"):
            shap_values = shap.Explainer(fitted_model.predict_proba, background, feature_names=feature_names)(eval_data)
        elif shap_values is None and hasattr(fitted_model, "predict"):
            shap_values = shap.Explainer(fitted_model.predict, background, feature_names=feature_names)(eval_data)
        if shap_values is None:
            raise RuntimeError(
                f"SHAP is not supported for model type {fitted_model.__class__.__name__} because it is not callable."
            )
        _render_shap_summary_figure(shap, shap_values, eval_data, feature_names, output_path, model_name, max_display)


def _supports_tree_shap(model) -> bool:
    """Return whether the estimator type is supported by tree SHAP explainers."""
    # Match by class name to avoid hard dependencies on external estimator packages.
    return model.__class__.__name__ in {
        "DecisionTreeClassifier",
        "RandomForestClassifier",
        "ExtraTreesClassifier",
        "GradientBoostingClassifier",
        "HistGradientBoostingClassifier",
        "XGBClassifier",
        "LGBMClassifier",
    }


def _render_shap_summary_figure(shap, shap_values, eval_data, feature_names, output_path, model_name, max_display):
    """Render SHAP values into a single-class or multiclass summary figure."""
    # Handle explanation object variants across SHAP versions and model types.
    if hasattr(shap_values, "values"):
        values = np.asarray(shap_values.values)
        if values.ndim == 3:
            _save_multiclass_shap_grid(shap, values, eval_data, feature_names, output_path, model_name, max_display)
            return
        plt.figure(figsize=(12, 7))
        shap.summary_plot(shap_values, eval_data, feature_names=feature_names, max_display=max_display, show=False)
        plt.title(f"SHAP Summary Plot for {model_name}")
        _save_figure(plt.gcf(), output_path)
        return
    if isinstance(shap_values, list):
        if len(shap_values) == 2:
            plt.figure(figsize=(12, 7))
            shap.summary_plot(
                np.asarray(shap_values[1]), eval_data, feature_names=feature_names, max_display=max_display, show=False
            )
            plt.title(f"SHAP Summary Plot for {model_name}")
            _save_figure(plt.gcf(), output_path)
            return
        if len(shap_values) > 2:
            _save_multiclass_shap_grid(
                shap,
                np.stack([np.asarray(class_values) for class_values in shap_values], axis=2),
                eval_data,
                feature_names,
                output_path,
                model_name,
                max_display,
            )
            return
    shap_array = np.asarray(shap_values)
    if shap_array.ndim == 3:
        _save_multiclass_shap_grid(shap, shap_array, eval_data, feature_names, output_path, model_name, max_display)
        return
    plt.figure(figsize=(12, 7))
    shap.summary_plot(shap_array, eval_data, feature_names=feature_names, max_display=max_display, show=False)
    plt.title(f"SHAP Summary Plot for {model_name}")
    _save_figure(plt.gcf(), output_path)


def _save_multiclass_shap_grid(shap, shap_values_3d, eval_data, feature_names, output_path, model_name, max_display):
    """Save a grid of per-class SHAP summary plots for multiclass models."""
    # Render class-specific SHAP plots into image tiles and compose a single figure.
    n_classes = shap_values_3d.shape[2]
    n_cols = 3
    n_rows = int(np.ceil(n_classes / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = np.atleast_1d(axes).flatten()
    for class_idx in range(n_classes):
        ax = axes[class_idx]
        shap_fig = plt.figure(figsize=(7, 5))
        shap.summary_plot(
            shap_values_3d[:, :, class_idx], eval_data, feature_names=feature_names, max_display=max_display, show=False
        )
        shap_fig.canvas.draw()
        img = _canvas_to_rgb_array(shap_fig.canvas)
        plt.close(shap_fig)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"Class {class_idx}", fontsize=10)
    for class_idx in range(n_classes, len(axes)):
        fig.delaxes(axes[class_idx])
    fig.suptitle(f"SHAP Summary Plots for {model_name}", fontsize=14)
    _save_figure(fig, output_path)


def _canvas_to_rgb_array(canvas) -> np.ndarray:
    """Convert a Matplotlib canvas to an RGB NumPy array."""
    # Support both legacy and modern canvas buffer APIs.
    width, height = canvas.get_width_height()
    if hasattr(canvas, "tostring_rgb"):
        return np.frombuffer(canvas.tostring_rgb(), dtype="uint8").reshape((height, width, 3))
    if hasattr(canvas, "buffer_rgba"):
        rgba = np.asarray(canvas.buffer_rgba())
        if rgba.ndim == 3 and rgba.shape[2] == 4:
            return rgba[:, :, :3].copy()
    raise RuntimeError("Matplotlib canvas does not expose tostring_rgb or buffer_rgba.")


def save_interpretability(config, cv_metrics_df, test_metrics_df, fitted_estimators, X_train):
    # Select top models for interpretability analysis and optionally produce SHAP/importance outputs.
    available = cv_metrics_df["model"].tolist()
    top_n = min(config.interpretability_top_models, len(available))
    if config.interpretability_top_models > top_n:
        print_formatted_txt(
            f"interpretability_top_models is larger than the available selected models. "
            f"Using {top_n} instead of {config.interpretability_top_models}.",
            "WARNING",
        )
    # Define which model families support feature importance extraction.
    tree_models = {
        "random_forest",
        "extra_trees",
        "gradient_boosting",
        "decision_tree",
        "hist_gradient_boosting",
        "xgboost",
        "lightgbm",
    }
    rows = []
    for model_name in select_top_models_for_interpretability(cv_metrics_df, top_n):
        estimator = fitted_estimators.get(model_name)
        if estimator is None:
            # Skip interpretability if estimator not available (should be rare).
            continue
        row = {
            "model": model_name,
            "cv_ranking_metric": float(cv_metrics_df.loc[cv_metrics_df["model"] == model_name, "cv_ranking_metric"].iloc[0]),
            "test_metric": float(test_metrics_df.loc[test_metrics_df["model"] == model_name, config.scoring_primary].iloc[0]),
            "feature_importance_created": False,
            "shap_created": False,
            "notes": "",
        }
        # For tree-based models, extract and persist feature importance data + plot.
        if model_name in tree_models:
            importance = extract_feature_importance_frame(estimator)
            if importance is not None and not importance.empty:
                importance_top_n = min(config.feature_importance_top_n, len(importance))
                if config.feature_importance_top_n > importance_top_n:
                    print_formatted_txt(
                        f"feature_importance_top_n for {model_name} is larger than the available features. "
                        f"Using {importance_top_n} instead of {config.feature_importance_top_n}.",
                        "WARNING",
                    )
                base_path = config.evaluate_dir / "interpretability" / f"{model_name}_feature_importance"
                save_frame_csv(importance, base_path.with_suffix(".csv"))
                save_feature_importance_plot(importance, base_path.with_suffix(".png"), model_name, top_n=importance_top_n)
                row["feature_importance_created"] = True
            else:
                row["notes"] = "Feature importance was not available for the fitted estimator."
        else:
            row["notes"] = "Feature importance is only generated for tree-based top models."
        # Optionally attempt SHAP summaries; exceptions are caught and logged.
        if config.interpretability_include_shap:
            sample = X_train.sample(n=min(config.shap_sample_size, len(X_train)), random_state=config.random_state)
            try:
                save_shap_summary_plot(
                    estimator,
                    sample,
                    config.evaluate_dir / "interpretability" / f"{model_name}_shap_summary.png",
                    model_name,
                    max_display=min(config.feature_importance_top_n, max(len(sample.columns), 1)),
                )
                row["shap_created"] = True
            except Exception as exc:
                message = f"SHAP skipped for {model_name}: {exc}"
                print_formatted_txt(message, "WARNING")
                row["notes"] = f"{row['notes']} {message}".strip()
        rows.append(row)
    if rows:
        save_frame_csv(pd.DataFrame(rows), config.evaluate_dir / "interpretability" / "interpretability_summary.csv")
