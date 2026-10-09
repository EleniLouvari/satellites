"""Regression checks for top-level packages and serialized worker/model objects."""
import importlib
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from tests.utils import expect_equal, expect_in, expect_true


@pytest.mark.parametrize("name", ["data_preparation", "eda", "ml_classification", "shared"])
def test_packages_are_direct_children_of_src(name):
    module = importlib.import_module(name)
    source_root = Path(__file__).resolve().parents[1] / "src"
    expect_equal(Path(module.__file__).resolve(), source_root / name / "__init__.py")


def test_model_pickle_loads_and_predicts():
    """A model saved with its current module path retains predictions."""
    from ml_classification.shared.models.models import ContiguousLabelClassifier

    data = pd.DataFrame({"value": [-2.0, -1.0, 1.0, 2.0]})
    model = ContiguousLabelClassifier(LogisticRegression(), num_classes=2).fit(data, np.array([0, 0, 1, 1]))
    expected = model.predict_proba(data)
    payload = pickle.dumps(model)
    expect_in(b"ml_classification.shared.models.models", payload)
    restored = pickle.loads(payload)
    expect_true(type(restored) is ContiguousLabelClassifier)
    np.testing.assert_allclose(restored.predict_proba(data), expected)


def test_config_pickle_preserves_run_paths(tmp_path):
    from ml_classification import ClassificationPipelineConfig

    config = ClassificationPipelineConfig(
        project_dir=tmp_path / "ml", target_column="label", feature_columns=["value"], feature_importance_top_n=1
    )
    restored = pickle.loads(pickle.dumps(config))
    expect_true(type(restored) is ClassificationPipelineConfig)
    expect_equal(restored.train_dir, config.train_dir)
    expect_equal(restored.predict_dir, config.predict_dir)
