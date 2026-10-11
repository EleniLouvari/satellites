"""Exercise SVM search and probability outputs through the shared model catalog."""

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from ml_classification.shared.config.config import ClassificationPipelineConfig
from ml_classification.shared.models import models
from ml_classification.step_03_train.train import TrainStep
from tests.utils import expect_equal, expect_subset, expect_true


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")


@pytest.mark.parametrize("n_classes", [2, 3])
@pytest.mark.parametrize("kernel", ["linear", "rbf", "poly"])
def test_svm_search_probabilities_and_serialization(monkeypatch, tmp_path, n_classes, kernel):
    """Search, OOF predictions, and persistence retain warning-free probability outputs."""
    monkeypatch.setattr(models, "_safe_import", lambda name: None)
    config = ClassificationPipelineConfig(
        project_dir=tmp_path,
        target_column="label",
        feature_columns=["feature", "category"],
        selected_models=("support_vector_machine",),
        tune_params={"support_vector_machine": {"model__C": [0.1, 1.0], "model__kernel": [kernel]}},
        cv_folds=2,
        max_search_candidates=2,
        n_jobs=1,
        selection_type="single_model",
        interpretability_top_models=0,
        feature_importance_top_n=2,
    )
    candidates = models.build_model_candidates(config, ["feature"], ["category"])
    expect_equal([candidate.name for candidate in candidates], ["support_vector_machine"])
    candidate = candidates[0]
    expect_true(candidate.supports_predict_proba)

    y = np.repeat(np.arange(n_classes), 30)
    frame = pd.DataFrame({"feature": y * 100.0 + np.tile(np.arange(30), n_classes), "category": "known"})
    frame.loc[::17, "feature"] = np.nan
    frame.loc[::19, "category"] = np.nan
    cv = list(StratifiedKFold(2, shuffle=True, random_state=42).split(frame, y))
    step = TrainStep(config)
    params, count = step._prepare_search_parameters(candidate, cv)
    search = step._build_search(candidate, params, cv, count, n_jobs=1)
    # Tiny halving subsamples can contain only one class; use full folds here.
    search.set_params(min_resources=len(frame))
    search.fit(frame, y)
    fitted = search.best_estimator_
    expect_equal(search.best_params_["model__kernel"], kernel)
    scaler = fitted.named_steps["preprocessor"].named_transformers_["numeric"].named_steps["scaler"]
    expect_equal(scaler.n_samples_seen_, len(frame))

    oof = cross_val_predict(fitted, frame, y, cv=cv, method="predict_proba", n_jobs=1)
    expect_equal(oof.shape, (len(frame), n_classes))
    expect_true(np.isfinite(oof).all())
    np.testing.assert_allclose(oof.sum(axis=1), 1.0)

    unseen = pd.DataFrame({"feature": [np.nan, 150.0], "category": ["unseen", np.nan]})
    probabilities = fitted.predict_proba(unseen)
    expect_equal(probabilities.shape, (2, n_classes))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    expect_subset(fitted.predict(unseen), y)
    model_path = tmp_path / "svm.joblib"
    joblib.dump(fitted, model_path)
    np.testing.assert_allclose(joblib.load(model_path).predict_proba(unseen), probabilities)


@pytest.mark.parametrize("samples_per_class", [2, 4, 6])
def test_svm_calibration_handles_small_classes_reproducibly(samples_per_class):
    """Small folds calibrate without requiring five observations per class."""
    y = np.repeat(["crop_a", "crop_b"], samples_per_class)
    X = np.arange(len(y), dtype=float).reshape(-1, 1)
    model = models.CalibratedSVC(random_state=42, class_weight="balanced")
    model.set_params(C=0.5, kernel="linear", probability=True)
    fitted = clone(model).fit(X, y)
    probabilities = fitted.predict_proba(X)
    expect_equal(fitted.classes_.tolist(), ["crop_a", "crop_b"])
    expect_true(np.isfinite(probabilities).all())
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    np.testing.assert_allclose(clone(model).fit(X, y).predict_proba(X), probabilities)
    np.testing.assert_array_equal(fitted.predict(X), fitted.classes_[probabilities.argmax(axis=1)])


@pytest.mark.parametrize("labels", [[0, 1, 1], [0, 0, 0]])
def test_svm_calibration_rejects_insufficient_class_support(labels):
    """Unsupported class counts fail with an actionable error before calibration."""
    with pytest.raises(ValueError, match="Error: SVM probability calibration requires"):
        models.CalibratedSVC().fit(np.arange(len(labels)).reshape(-1, 1), labels)


def test_svm_calibration_requires_probabilities():
    """A legacy override cannot disable probabilities required by the pipeline."""
    with pytest.raises(ValueError, match="Error: The SVM candidate requires probability=True"):
        models.CalibratedSVC(probability=False).fit([[0], [1], [2], [3]], [0, 0, 1, 1])
