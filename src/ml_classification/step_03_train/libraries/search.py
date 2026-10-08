"""Configure successive-halving searches and summarize winning CV scores."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingRandomSearchCV, ParameterGrid


def prepare_search_parameters(config, candidate: Any, cv: list[tuple[np.ndarray, np.ndarray]]) -> tuple[dict, int]:
    """Return safe parameter distributions and the effective search size."""
    # Copy the candidate's parameter distributions so we can safely modify them.
    param_distributions = dict(candidate.param_distributions)
    # Calculate how many unique parameter combinations are available.
    available_candidates = len(ParameterGrid(param_distributions))
    # Limit effective candidates to configured maximum to bound compute.
    effective_candidates = min(config.max_search_candidates, available_candidates)
    # Special-case KNN to ensure n_neighbors is not larger than any fold's train size.
    if candidate.name == "knn":
        min_fold_train_size = min(len(train_idx) for train_idx, _ in cv)
        max_neighbors = max(1, min_fold_train_size)
        safe_neighbors = [value for value in param_distributions["model__n_neighbors"] if value <= max_neighbors]
        param_distributions["model__n_neighbors"] = safe_neighbors or [1]
    return param_distributions, effective_candidates


def build_search(
    config,
    candidate: Any,
    param_distributions: dict,
    cv: list[tuple[np.ndarray, np.ndarray]],
    effective_candidates: int,
    n_jobs: int,
) -> HalvingRandomSearchCV:
    """Construct one consistently configured successive-halving search."""
    # Construct a HalvingRandomSearchCV with consistent configuration for each candidate.
    return HalvingRandomSearchCV(
        estimator=candidate.builder(),
        param_distributions=param_distributions,
        factor=2,
        cv=cv,
        scoring=config.scoring_primary,
        n_jobs=n_jobs,
        random_state=config.random_state,
        n_candidates=effective_candidates,
        refit=True,
        error_score="raise",
    )


def summarize_search(candidate: Any, search: HalvingRandomSearchCV, results_df: pd.DataFrame) -> tuple[dict, list[dict]]:
    """Extract the model summary and per-fold scores from the winning search row."""
    best_row = results_df.iloc[0]
    split_columns = sorted(
        [column for column in results_df if column.startswith("split") and column.endswith("_test_score")],
        key=lambda value: int(value.split("_")[0].replace("split", "")),
    )
    fold_scores = [float(best_row[column]) for column in split_columns]
    score_mean = float(np.mean(fold_scores)) if fold_scores else float(search.best_score_)
    score_std = float(np.std(fold_scores, ddof=0)) if fold_scores else 0.0
    summary = {
        "model": candidate.name,
        "best_cv_score": score_mean,
        "cv_score_std": score_std,
        "cv_score_minus_std": score_mean - score_std,
        "best_iteration": int(getattr(search, "n_iterations_", 0)),
        "supports_predict_proba": candidate.supports_predict_proba,
    }
    per_fold = [
        {"model": candidate.name, "fold": fold_idx, "score": float(best_row[column])}
        for fold_idx, column in enumerate(split_columns)
    ]
    # Return summary for ranking and a per-fold list for reporting.
    return summary, per_fold
