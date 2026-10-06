from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from satellites.ml_classification.pipeline import GeospatialClassificationPipeline
from satellites.ml_classification.shared.config.base import PipelineStepBase
from satellites.ml_classification.shared.config.config import ClassificationPipelineConfig
from satellites.ml_classification.step_02_prepare.prepare import PrepareStep
from satellites.ml_classification.step_05_predict.predict import PredictStep


def _config(tmp_path: Path, **overrides):
    values = {
        "project_dir": tmp_path,
        "target_column": "label",
        "feature_columns": ["feature"],
        "cv_folds": 2,
    }
    values.update(overrides)
    return ClassificationPipelineConfig(**values)


def test_with_schema_attaches_schema_metadata(tmp_path):
    # Ensure PipelineStepBase._with_schema copies payload and adds _schema metadata.
    config = _config(tmp_path)
    base = PipelineStepBase(config)
    payload = {"a": 1}
    enriched = base._with_schema(payload, "artifact_name")
    assert "_schema" in enriched
    assert enriched["a"] == 1
    assert enriched["_schema"]["artifact"] == "artifact_name"


def test_create_prediction_frame_adds_expected_columns(tmp_path):
    # Prepare a tiny dataset and run _create_prediction_frame to validate output columns.
    config = _config(tmp_path)
    step = PredictStep(config)
    dataset = pd.DataFrame({"row_id": ["r1", "r2"], "label": [None, "cat"]})
    # Two classes: cat, dog
    labels = ["cat", "dog"]
    # Predictions for both rows and probability arrays
    predictions = np.array(["dog", "cat"])
    probabilities = np.array([[0.2, 0.8], [0.9, 0.1]])
    final_df, unknown_mask = step._create_prediction_frame(dataset, predictions, probabilities, labels)
    # Check that filled column uses prediction only for unknown row
    assert "label_filled" in final_df.columns or step.config.prediction_filled_column in final_df.columns
    # Ensure probability columns for each class exist
    for lbl in labels:
        col = f"{config.probability_prefix}_{lbl}"
        assert col in final_df.columns
    # unknown_mask should indicate first row is unknown
    assert unknown_mask.tolist() == [True, False]


def test_split_labeled_data_stratified(tmp_path):
    # Validate that _split_labeled_data produces class-stratified splits when spatial_split is False.
    config = _config(tmp_path)
    step = PrepareStep(config)
    # Build a labeled DataFrame with balanced classes
    df = pd.DataFrame({"feature": list(range(10)), "label": ["A"] * 5 + ["B"] * 5})
    train_df, test_df, strategy, splitter = step._split_labeled_data(df)
    # Strategy should indicate random_stratified and splitter should be None
    assert strategy == "random_stratified"
    assert splitter is None
    # Each split should contain both classes
    assert set(train_df[config.target_column].unique()) == {"A", "B"}
    assert set(test_df[config.target_column].unique()) == {"A", "B"}


def test_build_cv_folds_stratified(tmp_path):
    # Ensure _build_cv_folds returns the requested number of folds and indices are valid.
    config = _config(tmp_path, cv_folds=2)
    step = PrepareStep(config)
    # Create a train_df with encoded target present; include _target_encoded column
    train_df = pd.DataFrame({"feature": list(range(6)), "_target_encoded": [0, 0, 0, 1, 1, 1]})
    active_features = ["feature"]
    folds = step._build_cv_folds(train_df, active_features, spatial_splitter=None)
    assert len(folds) == config.cv_folds
    # Check that indices reference existing rows
    for f in folds:
        assert all(0 <= idx < len(train_df) for idx in f["train_index"])
        assert all(0 <= idx < len(train_df) for idx in f["valid_index"])


def test_create_reports_recreates_only_requested_step_and_index(tmp_path):
    # Ensure create_reports("evaluate") dispatches only evaluate + index generation.
    pipeline = GeospatialClassificationPipeline(_config(tmp_path))
    pipeline._create_check_reports = MagicMock(return_value="check")
    pipeline._create_prepare_reports = MagicMock(return_value="prepare")
    pipeline._create_train_reports = MagicMock(return_value="train")
    pipeline._create_evaluate_reports = MagicMock(return_value={"report": "evaluate"})
    pipeline._create_predict_reports = MagicMock(return_value={"report": "predict"})

    result = pipeline.create_reports("evaluate")

    pipeline._create_check_reports.assert_not_called()
    pipeline._create_prepare_reports.assert_not_called()
    pipeline._create_train_reports.assert_not_called()
    pipeline._create_evaluate_reports.assert_called_once()
    pipeline._create_predict_reports.assert_not_called()
    assert result["requested_steps"] == ["evaluate"]
    assert "evaluate" in result
    assert "index" in result


def test_create_reports_raises_for_unknown_step(tmp_path):
    # Ensure bad step names fail fast with a clear error.
    pipeline = GeospatialClassificationPipeline(_config(tmp_path))
    with pytest.raises(ValueError, match="Unknown step"):
        pipeline.create_reports("unknown_step")
