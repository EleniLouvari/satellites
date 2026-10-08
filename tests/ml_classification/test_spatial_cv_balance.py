"""Regression checks for grouped and row-wise spatial CV fold behavior."""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_predict

from ml_classification.shared.config.config import ClassificationPipelineConfig
from ml_classification.step_02_prepare.prepare import PrepareStep


def _folds(tmp_path, sizes, labels=None, method="by_group"):
    groups = np.repeat(np.arange(len(sizes)), sizes)
    frame = pd.DataFrame({
        "feature": np.arange(len(groups)),
        "_target_encoded": np.arange(len(groups)) % 2 if labels is None else labels,
    }, index=np.arange(len(groups)) + 1000)
    config = ClassificationPipelineConfig(
        project_dir=tmp_path, target_column="label", feature_columns=["feature"],
        spatial_split=True, spatial_split_method=method, cv_folds=5, test_size=0.2,
    )
    spatial_splitter = Mock()
    spatial_splitter.build_spatial_groups.return_value = pd.Series(groups, index=frame.index)
    folds = PrepareStep(config)._build_cv_folds(frame, ["feature"], spatial_splitter)
    return folds, frame, groups


def _assert_partition(folds, frame, groups):
    validation_rows = []
    for fold in folds:
        train, valid = fold["train_index"], fold["valid_index"]
        assert train and valid
        assert set(train).isdisjoint(valid)
        assert sorted(train + valid) == list(range(len(frame)))
        assert set(groups[train]).isdisjoint(groups[valid])
        assert set(frame.iloc[train]["_target_encoded"]) == set(frame["_target_encoded"])
        validation_rows.extend(valid)
    assert sorted(validation_rows) == list(range(len(frame)))


def test_spatial_cv_balances_rows_and_supports_oof_predictions(tmp_path):
    sizes = [18, 17, 16, 15, 14, 6, 5, 4, 3, 2]
    folds, frame, groups = _folds(tmp_path, sizes)
    assert [len(fold["valid_index"]) for fold in folds] == [20] * 5
    _assert_partition(folds, frame, groups)
    repeated, _, _ = _folds(tmp_path, sizes)
    assert folds == repeated
    probabilities = cross_val_predict(
        DummyClassifier(), frame[["feature"]], frame["_target_encoded"],
        cv=[(fold["train_index"], fold["valid_index"]) for fold in folds],
        method="predict_proba",
    )
    assert probabilities.shape == (100, 2)


def test_oversized_spatial_cell_remains_intact(tmp_path):
    folds, frame, groups = _folds(tmp_path, [40, 15, 15, 10, 10, 5, 5])
    assert max(len(fold["valid_index"]) for fold in folds) == 40
    _assert_partition(folds, frame, groups)


def test_spatial_cv_rejects_class_confined_to_one_cell(tmp_path):
    labels = np.zeros(100, dtype=int)
    labels[:18] = 1
    with pytest.raises(ValueError, match="retain every class"):
        _folds(tmp_path, [18, 17, 16, 15, 14, 6, 5, 4, 3, 2], labels=labels)


def test_spatial_cv_rejects_too_few_cells(tmp_path):
    with pytest.raises(ValueError, match="at least cv_folds occupied grid cells"):
        _folds(tmp_path, [30, 30, 20, 20])


def test_by_row_spatial_cv_prefers_tile_coverage_over_whole_cell_blocking(tmp_path):
    folds, frame, groups = _folds(tmp_path, [18, 17, 16, 15, 14, 6, 5, 4, 3, 2], method="by_row")
    all_group_ids = set(np.unique(groups))

    assert len(folds) == 5
    for fold in folds:
        train, valid = fold["train_index"], fold["valid_index"]
        assert train and valid
        assert set(train).isdisjoint(valid)
        assert sorted(train + valid) == list(range(len(frame)))

        train_groups = set(groups[train])
        valid_groups = set(groups[valid])
        # By-row CV should allow mixed tiles in both sides when possible.
        assert train_groups.intersection(valid_groups)
        # Validation should still cover most of the area, not just a few cells.
        assert len(valid_groups) >= 7
        assert len(train_groups) == len(all_group_ids)


def test_by_row_spatial_cv_is_deterministic_for_same_seed(tmp_path):
    folds_first, _, _ = _folds(tmp_path, [18, 17, 16, 15, 14, 6, 5, 4, 3, 2], method="by_row")
    folds_second, _, _ = _folds(tmp_path, [18, 17, 16, 15, 14, 6, 5, 4, 3, 2], method="by_row")
    assert folds_first == folds_second


def test_by_row_cv_covers_all_cells_and_classes_when_feasible(tmp_path):
    folds, frame, groups = _folds(tmp_path, [5] * 10, method="by_row")
    validation_rows = []
    for fold in folds:
        train, valid = fold["train_index"], fold["valid_index"]
        assert len(valid) == 10
        assert set(train).isdisjoint(valid)
        assert len(set(groups[train])) == len(set(groups[valid])) == 10
        assert set(frame.iloc[train]["_target_encoded"]) == set(frame.iloc[valid]["_target_encoded"]) == {0, 1}
        assert frame.iloc[valid]["_target_encoded"].value_counts().tolist() == [5, 5]
        validation_rows.extend(valid)
    assert sorted(validation_rows) == list(range(len(frame)))
    predictions = cross_val_predict(
        DummyClassifier(), frame[["feature"]], frame["_target_encoded"],
        cv=[(fold["train_index"], fold["valid_index"]) for fold in folds], method="predict_proba",
    )
    assert predictions.shape == (50, 2)


def test_by_row_cv_distributes_rare_class_across_distinct_folds(tmp_path):
    labels = np.zeros(50, dtype=int)
    labels[:2] = 1
    with pytest.warns(UserWarning, match="fewer samples than cv_folds"):
        folds, frame, _ = _folds(tmp_path, [5] * 10, labels=labels, method="by_row")
    assert all(set(frame.iloc[fold["train_index"]]["_target_encoded"]) == {0, 1} for fold in folds)
    assert sum(1 in set(frame.iloc[fold["valid_index"]]["_target_encoded"]) for fold in folds) == 2


def test_by_row_cv_rejects_singleton_class(tmp_path):
    labels = np.zeros(50, dtype=int)
    labels[0] = 1
    with pytest.raises(ValueError, match="at least 2 rows per class"):
        _folds(tmp_path, [5] * 10, labels=labels, method="by_row")


def test_by_row_cv_handles_uneven_fold_sizes(tmp_path):
    folds, frame, groups = _folds(tmp_path, [6, 6, 6] + [5] * 7, method="by_row")
    assert [len(fold["valid_index"]) for fold in folds] == [11, 11, 11, 10, 10]
    validation_rows = []
    for fold in folds:
        train, valid = fold["train_index"], fold["valid_index"]
        assert set(train).isdisjoint(valid)
        assert set(frame.iloc[valid]["_target_encoded"]) == {0, 1}
        assert len(set(groups[valid])) == 10
        validation_rows.extend(valid)
    assert sorted(validation_rows) == list(range(len(frame)))
