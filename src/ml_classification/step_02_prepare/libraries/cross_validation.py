"""Build stratified or spatial CV folds with class coverage and row balance."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, StratifiedKFold

from ml_classification.step_02_prepare.libraries.spatial_allocation import allocate_spatial_rows


def build_cv_folds(config, train_df, active_features, spatial_splitter):
    # Disjoint CV targets 1 / cv_folds of the training rows per validation fold.
    if spatial_splitter is None:
        splitter = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)
        split_iterator = splitter.split(train_df[active_features], train_df["_target_encoded"])
    else:
        # Build spatial cell labels for CV scoring in both spatial modes.
        groups = spatial_splitter.build_spatial_groups(train_df)
        if config.spatial_split_method == "by_group":
            # Keep cells intact while prioritizing row balance over class stratification.
            split_iterator = balanced_spatial_cv_splits(config, train_df, active_features, groups)
        else:
            # Prefer row-wise folds that keep broad tile coverage in train and validation.
            split_iterator = balanced_spatial_row_cv_splits(config, train_df, active_features, groups)
    folds = []
    all_labels = set(train_df["_target_encoded"])
    for fold_id, (train_idx, valid_idx) in enumerate(split_iterator):
        # For spatial folds, ensure every class appears in the training portion of each fold.
        if spatial_splitter is not None:
            missing = sorted(all_labels - set(train_df.iloc[train_idx]["_target_encoded"]))
            if missing:
                raise ValueError(
                    f"Error: Spatial CV fold {fold_id} has no training parcels for encoded classes {missing}. Merge unsupported rare classes or reduce cv_folds."
                )
        folds.append({"fold": fold_id, "train_index": train_idx.tolist(), "valid_index": valid_idx.tolist()})
    return folds


def balanced_spatial_cv_splits(config, train_df, active_features, groups):
    """Choose row-balanced whole-cell folds with every class retained in training."""
    if groups.nunique() < config.cv_folds:
        raise ValueError("Error: Spatial CV requires at least cv_folds occupied grid cells. Increase spatial_split_grid_size.")
    target = train_df["_target_encoded"]
    overall_distribution = target.value_counts(normalize=True)
    all_labels = set(overall_distribution.index)
    candidates = []
    # GroupKFold balances row counts; the stratified candidate can rescue class
    # coverage or improve label balance when its fold sizes are equally good.
    splitters = (
        GroupKFold(n_splits=config.cv_folds),
        StratifiedGroupKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state),
    )
    for splitter in splitters:
        splits = list(splitter.split(train_df[active_features], target, groups=groups))
        if any(set(target.iloc[train_idx]) != all_labels for train_idx, _ in splits):
            continue
        size_errors = [abs(len(valid_idx) / len(train_df) - 1 / config.cv_folds) for _, valid_idx in splits]
        class_error = sum(
            float(
                (
                    target.iloc[valid_idx].value_counts(normalize=True).reindex(overall_distribution.index, fill_value=0.0)
                    - overall_distribution
                )
                .abs()
                .mean()
            )
            for _, valid_idx in splits
        )
        score = (max(size_errors), sum(error**2 for error in size_errors), class_error)
        candidates.append((score, splits))
    if not candidates:
        raise ValueError(
            "Error: Spatial CV could not retain every class in every training fold. Increase spatial_split_grid_size, merge unsupported rare classes, or reduce cv_folds."
        )
    return min(candidates, key=lambda candidate: candidate[0])[1]


def balanced_spatial_row_cv_splits(config, train_df, active_features, groups):
    """Jointly spread validation and training coverage, retaining class support."""
    n_rows, n_folds = len(train_df), config.cv_folds
    sizes = np.full(n_folds, n_rows // n_folds, dtype=int)
    sizes[: n_rows % n_folds] += 1
    assignments = allocate_spatial_rows(train_df["_target_encoded"], groups, sizes, config.random_state)
    return [(np.flatnonzero(assignments != fold), np.flatnonzero(assignments == fold)) for fold in range(n_folds)]
