"""Load fitted estimators, validate probability alignment, and score train/test predictions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, roc_auc_score

from satellites.ml_classification.shared.logging import print_formatted_txt
from satellites.ml_classification.step_04_evaluate.libraries.metrics import score_predictions
from satellites.ml_classification.shared.persistence import load_joblib, save_frame_csv
from satellites.ml_classification.step_04_evaluate.libraries.plots import (
    save_binary_curve_plots,
    save_binary_evaluation_panel,
    save_confusion_matrix_plot,
    save_multiclass_evaluation_panel,
)


def validate_probability_class_order(estimator: Any, labels: list[str], model_name: str) -> None:
    """Ensure predict_proba columns match the pipeline's encoded class order."""
    if not hasattr(estimator, "predict_proba"):
        return
    expected_classes = np.arange(len(labels), dtype=int)
    estimator_classes = getattr(estimator, "classes_", None)
    try:
        actual_classes = np.asarray(estimator_classes, dtype=int)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Model {model_name!r} exposes an invalid classes_ value; expected "
            f"{expected_classes.tolist()} for probability-column alignment."
        ) from exc
    if not np.array_equal(actual_classes, expected_classes):
        raise RuntimeError(
            f"Model {model_name!r} probability class order {actual_classes.tolist()} does not match "
            f"the expected encoded class order {expected_classes.tolist()}."
        )


def to_geo_classifier_result_row(
    y_true: pd.Series, y_pred: np.ndarray, probabilities: np.ndarray | None, labels: list[str], model_name: str
) -> dict[str, Any]:
    """Build legacy GeoDataFrameClassifier-style metric rows for reporting."""
    # Branch metric formatting for binary versus multiclass reporting outputs.
    binary_problem = len(labels) == 2
    if binary_problem:
        conf_matrix = confusion_matrix(y_true, y_pred, labels=labels)
        if conf_matrix.shape == (2, 2):
            tn, fp, fn, tp = conf_matrix.ravel()
        else:
            tn = fp = fn = tp = 0
        # Compute confusion-derived metrics safely (avoid division-by-zero).
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        # Overall accuracy across examples.
        accuracy = float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))
        # Compute binary AUC when probabilities available; handle invalid cases.
        auc = np.nan
        if probabilities is not None and probabilities.shape[1] >= 2:
            try:
                auc = float(roc_auc_score(y_true, probabilities[:, 1]))
            except ValueError:
                auc = np.nan
        return {
            "model": model_name,
            "roc_auc": (100 * auc) if not np.isnan(auc) else "Undefined",
            "accuracy": 100 * accuracy,
            "precision": 100 * precision,
            "recall": 100 * recall,
            "f1_score": 100 * f1,
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
        }

    precision_arr, recall_arr, f1_arr, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=labels, zero_division=0
    )
    # Per-class precision/recall/f1 and overall accuracy for multiclass problems.
    accuracy = float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))
    roc_auc_str = "Undefined"
    if probabilities is not None:
        try:
            # Compute per-class ROC AUC with one-vs-rest approach when possible.
            auc_per_class = roc_auc_score(y_true, probabilities, labels=labels, multi_class="ovr", average=None)
            roc_auc_str = ", ".join([f"class {idx} ({100 * score:.2f})" for idx, score in enumerate(auc_per_class)])
        except ValueError:
            roc_auc_str = "Undefined"
    precision_str = ", ".join([f"class {idx} ({100 * score:.2f})" for idx, score in enumerate(precision_arr)])
    recall_str = ", ".join([f"class {idx} ({100 * score:.2f})" for idx, score in enumerate(recall_arr)])
    f1_str = ", ".join([f"class {idx} ({100 * score:.2f})" for idx, score in enumerate(f1_arr)])
    return {
        "model": model_name,
        "accuracy": 100 * accuracy,
        "f1_score": 100 * float(np.mean(f1_arr)),
        "precision": 100 * float(np.mean(precision_arr)),
        "recall": 100 * float(np.mean(recall_arr)),
        "f1_score_class": f1_str,
        "precision_class": precision_str,
        "recall_class": recall_str,
        "roc_auc_class": roc_auc_str,
    }


def model_artifact_path(config, model_name):
    # Prefer model artefact inside model-specific folder, fallback to top-level train dir.
    path = config.train_model_dir(model_name) / "best_model.joblib"
    return path if path.exists() else config.train_dir / f"{model_name}_best_model.joblib"


def save_model_evaluation_plot(config, model_name, y_test, predictions, probabilities, labels):
    # If no probabilities, only save confusion matrix for the model.
    if probabilities is None:
        save_confusion_matrix_plot(
            y_test,
            predictions,
            labels,
            config.evaluate_dir / "confusion_matrices" / f"{model_name}_confusion_matrix.png",
            f"Confusion Matrix: {model_name}",
        )
        return
    # When probabilities exist, save combined ROC/confusion panels for binary or multiclass.
    panel_path = config.evaluate_dir / "panels" / f"{model_name}_roc_confusion.png"
    if len(labels) == 2:
        # Binary-specific evaluation visuals use the positive-class probability column.
        save_binary_evaluation_panel(y_test, predictions, probabilities[:, 1], labels, panel_path, model_name)
        save_binary_curve_plots(
            y_test, probabilities[:, 1], config.evaluate_dir / "curves", model_name, pos_label=labels[1], include_roc=False
        )
    else:
        # Multiclass panel visualizes per-class metrics and curves.
        save_multiclass_evaluation_panel(y_test, predictions, probabilities, labels, panel_path, model_name)


def evaluate_base_models(config, inputs):
    # Evaluate each trained model on train/test splits, collecting metrics and artifacts.
    context = inputs["context"]
    labels = context["labels"]
    encoder = context["label_encoder"]
    # Caches hold probability arrays for train/test/oof (if available) to support voting.
    caches = {"train": {}, "test": {}, "oof": {}}
    metric_rows = {"train": [], "test": []}
    geo_rows = {"train": [], "test": []}
    estimators = {}

    # Filter models to only those in current selected_models if configured.
    all_model_names = list(inputs["model_specs"].keys())
    models_to_evaluate = all_model_names
    skipped_models = []

    if config.selected_models is not None:
        # Only evaluate models that are both trained AND in current selected_models.
        models_to_evaluate = [model_name for model_name in all_model_names if model_name in config.selected_models]
        skipped_models = [model_name for model_name in all_model_names if model_name not in config.selected_models]
        if skipped_models:
            print_formatted_txt(
                f"Filtered evaluation: {len(models_to_evaluate)} models selected, "
                f"{len(skipped_models)} trained models skipped: {skipped_models}",
                "INFO",
            )

    for model_name in models_to_evaluate:
        # Load the persisted best estimator for this candidate model.
        estimator = load_joblib(model_artifact_path(config, model_name))
        estimators[model_name] = estimator
        validate_probability_class_order(estimator, labels, model_name)
        # Load out-of-fold probabilities if saved during training (used for class optimization).
        oof_path = config.train_model_dir(model_name) / "oof_probabilities.joblib"
        if oof_path.exists():
            caches["oof"][model_name] = load_joblib(oof_path)
        split_values = {}
        for split_name in ("train", "test"):
            X = inputs[f"X_{split_name}"]
            y = inputs[f"y_{split_name}"]
            # Predict encoded labels and convert back to original string labels.
            encoded = estimator.predict(X)
            predictions = encoder.inverse_transform(np.asarray(encoded, dtype=int))
            # Obtain class probabilities when supported by estimator.
            probabilities = estimator.predict_proba(X) if hasattr(estimator, "predict_proba") else None
            split_values[split_name] = (predictions, probabilities)
            # Score predictions and record legacy geo-classifier-style rows.
            metric_rows[split_name].append(score_predictions(y, predictions, probabilities, labels, model_name))
            geo_rows[split_name].append(to_geo_classifier_result_row(y, predictions, probabilities, labels, model_name))
            if probabilities is not None:
                caches[split_name][model_name] = probabilities
        # Save a per-model classification report and evaluation plots for the test split.
        test_predictions, test_probabilities = split_values["test"]
        report = pd.DataFrame(classification_report(inputs["y_test"], test_predictions, output_dict=True)).T.reset_index()
        report.rename(columns={"index": "label"}, inplace=True)
        save_frame_csv(report, config.evaluate_dir / "reports" / f"{model_name}_classification_report.csv")
        save_model_evaluation_plot(config, model_name, inputs["y_test"], test_predictions, test_probabilities, labels)
    return metric_rows, geo_rows, caches, estimators
