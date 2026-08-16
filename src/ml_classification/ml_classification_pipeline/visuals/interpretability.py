"""Interpretability plotting utilities including feature importance and SHAP summaries.

SHAP support is optional: functions attempt a lazy import and provide clear
errors when the environment does not include the `shap` package. The helpers
write image files and clean up matplotlib state to avoid interfering with
other plotting in the same process.
"""
from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..core.persistence import ensure_dir

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


def _save_figure(fig: plt.Figure, output_path: str | Path) -> None:
    """Save and close a Matplotlib figure to disk."""
    # Ensure destination directories exist before writing figure files.
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
