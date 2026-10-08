"""Classification scores with a consistent definition across evaluation strategies."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def score_predictions(
    y_true: pd.Series, y_pred: np.ndarray, probabilities: np.ndarray | None, labels: list[str], model_name: str
) -> dict[str, Any]:
    """Compute common classification metrics from labels and optional probabilities."""
    # Use the same metric definitions when evaluating individual and voting models.
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
