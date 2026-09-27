"""Regression checks for row balance without splitting spatial cells."""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_predict

from satellites.ml_classification.core.config import ClassificationPipelineConfig
from satellites.ml_classification.steps.prepare_step import PrepareStep


def _folds(tmp_path, sizes, labels=None):
    groups = np.repeat(np.arange(len(sizes)), sizes)
    frame = pd.DataFrame({
        "feature": np.arange(len(groups)),
        "_target_encoded": np.arange(len(groups)) % 2 if labels is None else labels,
    }, index=np.arange(len(groups)) + 1000)
    config = ClassificationPipelineConfig(
        project_dir=tmp_path, target_column="label", feature_columns=["feature"],
        spatial_split=True, cv_folds=5, test_size=0.2,
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
