"""Regression coverage for neural estimator tags and halving CV sampling."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone, is_classifier
from sklearn.pipeline import Pipeline

from ml_classification.shared.models.tensorflow_lstm_models import KerasLSTMClassifier
from ml_classification.shared.models.tensorflow_models import TensorFlowDenseClassifier
from ml_classification.step_03_train.train import TrainStep


@pytest.mark.parametrize("estimator_class", [TensorFlowDenseClassifier, KerasLSTMClassifier])
def test_neural_estimators_advertise_classifier_tags(estimator_class):
    estimator = estimator_class()
    assert is_classifier(estimator)
    assert is_classifier(clone(estimator))
    assert is_classifier(Pipeline([("model", estimator)]))


@pytest.mark.parametrize("estimator_class", [TensorFlowDenseClassifier, KerasLSTMClassifier])
def test_neural_halving_search_handles_uneven_folds(monkeypatch, estimator_class):
    # Exercise real search/sampling without needing a TensorFlow runtime.
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.full(len(X), self.classes_[0])

    monkeypatch.setattr(estimator_class, "fit", fit)
    monkeypatch.setattr(estimator_class, "predict", predict)
    X = pd.DataFrame({"feature": np.arange(100, dtype=float)})
    y = np.arange(100) % 3
    indices = np.arange(len(X))
    cv = [(np.setdiff1d(indices, test), test) for test in np.split(indices, [10, 55])]
    candidate = SimpleNamespace(builder=lambda: Pipeline([("model", estimator_class())]))
    step = object.__new__(TrainStep)
    step.config = SimpleNamespace(scoring_primary="accuracy", random_state=42)

    search = step._build_search(candidate, {}, cv, effective_candidates=1, n_jobs=1)
    search.fit(X, y)

    # Classifier resources include the class count; without the classifier tag,
    # six resources leave the smallest test fold with int(6 / 100 * 10) == 0.
    assert search.min_resources_ == 2 * len(cv) * len(np.unique(y))
    assert np.isfinite(search.best_score_)
    assert search.predict(X).shape == y.shape
