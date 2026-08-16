import numpy as np
import pandas as pd
from types import SimpleNamespace

from ml_classification.ml_classification_pipeline.steps.train_step import TrainStep
from ml_classification.ml_classification_pipeline.steps.evaluate_step import EvaluateStep


def test_prepare_search_parameters_knn_limits_neighbors():
    step = object.__new__(TrainStep)
    step.config = SimpleNamespace(max_search_candidates=100)

    class Candidate:
        name = "knn"
        param_distributions = {"model__n_neighbors": [1, 2, 3, 4, 5], "model__p": [1]}

    # two folds with small train sizes -> min_fold_train_size == 2
    cv = [(np.array([0, 1]), np.array([2])), (np.array([0, 1, 2]), np.array([3]))]
    params, effective = step._prepare_search_parameters(Candidate(), cv)

    assert params["model__n_neighbors"] == [1, 2]
    assert effective <= 100


def test_summarize_search_returns_summary_and_per_fold():
    # Minimal fake search/results to exercise summarization logic
    candidate = SimpleNamespace(name="dummy", supports_predict_proba=False)
    class FakeSearch:
        n_iterations_ = 5
        best_score_ = 0.8

    search = FakeSearch()
    import pandas as pd

    results_df = pd.DataFrame([
        {"split0_test_score": 0.9, "split1_test_score": 0.8}
    ])

    summary, per_fold = TrainStep._summarize_search(candidate, search, results_df)

    assert summary["model"] == "dummy"
    assert abs(summary["best_cv_score"] - 0.85) < 1e-6
    assert len(per_fold) == 2


def test_to_geo_classifier_result_row_binary_and_multiclass():
    eval_step = object.__new__(EvaluateStep)

    # Binary case
    y_true = pd.Series([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    probs = np.array([[0.6, 0.4], [0.1, 0.9], [0.7, 0.3], [0.8, 0.2]])
    row = eval_step._to_geo_classifier_result_row(y_true, y_pred, probs, [0, 1], "m1")
    assert row["model"] == "m1"
    assert "accuracy" in row and "roc_auc" in row

    # Multiclass case
    y_true = pd.Series(["a", "b", "c", "a"])
    y_pred = np.array(["a", "b", "b", "c"])
    probs = np.array([
        [0.8, 0.1, 0.1],
        [0.0, 0.9, 0.1],
        [0.1, 0.7, 0.2],
        [0.2, 0.3, 0.5],
    ])
    row2 = eval_step._to_geo_classifier_result_row(y_true, y_pred, probs, ["a", "b", "c"], "m2")
    assert row2["model"] == "m2"
    assert "f1_score_class" in row2 and "roc_auc_class" in row2
