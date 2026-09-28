"""Exercise SVM search and probability outputs through the shared model catalog."""

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from satellites.ml_classification.core import models
from satellites.ml_classification.core.config import ClassificationPipelineConfig
from satellites.ml_classification.steps.train_step import TrainStep


@pytest.mark.parametrize("n_classes", [2, 3])
@pytest.mark.parametrize("kernel", ["linear", "rbf", "poly"])
def test_svm_search_probabilities_and_serialization(monkeypatch, tmp_path, n_classes, kernel):
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
    assert [candidate.name for candidate in candidates] == ["support_vector_machine"]
    candidate = candidates[0]
    assert candidate.supports_predict_proba

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
    assert search.best_params_["model__kernel"] == kernel
    scaler = fitted.named_steps["preprocessor"].named_transformers_["numeric"].named_steps["scaler"]
    assert scaler.n_samples_seen_ == len(frame)

    oof = cross_val_predict(fitted, frame, y, cv=cv, method="predict_proba", n_jobs=1)
    assert oof.shape == (len(frame), n_classes)
    assert np.isfinite(oof).all()
    np.testing.assert_allclose(oof.sum(axis=1), 1.0)

    unseen = pd.DataFrame({"feature": [np.nan, 150.0], "category": ["unseen", np.nan]})
    probabilities = fitted.predict_proba(unseen)
    assert probabilities.shape == (2, n_classes)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    assert set(fitted.predict(unseen)).issubset(set(y))
    model_path = tmp_path / "svm.joblib"
    joblib.dump(fitted, model_path)
    np.testing.assert_allclose(joblib.load(model_path).predict_proba(unseen), probabilities)
