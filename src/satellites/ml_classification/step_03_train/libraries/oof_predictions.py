"""Generate and persist out-of-fold probabilities for downstream evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict

from satellites.ml_classification.shared.persistence import save_joblib


def save_oof_probabilities(
    config,
    candidate: Any,
    estimator: Any,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    cv: list[tuple[np.ndarray, np.ndarray]],
    model_output_dir: Any,
    *,
    n_jobs: int,
) -> None:
    """Persist OOF probabilities when downstream optimization or validation requires them."""
    # OOF probabilities support both class-multiplier learning and leakage-safe
    # rank-confidence validation.
    requires_oof = bool(
        getattr(config, "optimize_class_probabilities", False) or getattr(config, "rank_confidence_enabled", False)
    )
    if not (requires_oof and candidate.supports_predict_proba):
        return
    try:
        probabilities = cross_val_predict(estimator, X_train, y_train, cv=cv, method="predict_proba", n_jobs=n_jobs)
    except PermissionError:
        # Retry without parallelism if process spawning is restricted.
        probabilities = cross_val_predict(estimator, X_train, y_train, cv=cv, method="predict_proba", n_jobs=1)
    # Persist OOF probabilities for later class probability optimization.
    save_joblib(probabilities, model_output_dir / "oof_probabilities.joblib")
