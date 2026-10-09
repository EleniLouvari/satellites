from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ml_classification.shared.config.config import ClassificationPipelineConfig
from ml_classification.shared.rank_confidence import (
    aggregate_rank_scores,
    assign_confidence_levels,
    classify_ensemble_rank_based,
    probabilities_to_ranks,
)
from ml_classification.step_03_train.train import TrainStep
from ml_classification.step_04_evaluate.evaluate import EvaluateStep
from ml_classification.step_04_evaluate.libraries import confidence as evaluate_confidence_module
from ml_classification.step_04_evaluate.libraries.class_reliability import calculate_oof_class_reliability
from ml_classification.step_05_predict.libraries.refit import fit_and_predict_selected_strategy
from ml_classification.step_05_predict.predict import PredictStep
from tests.utils import expect_equal, expect_false, expect_true

CLASSES = ["a", "b", "c", "d"]


def test_compute_holdout_monotonic_requires_finite_three_level_accuracy() -> None:
    monotonic = evaluate_confidence_module._compute_holdout_monotonic(
        pd.DataFrame(
            {
                "confidence_level": ["HIGH", "MEDIUM", "LOW"],
                "accuracy": [0.9, 0.7, 0.5],
            }
        )
    )
    non_monotonic = evaluate_confidence_module._compute_holdout_monotonic(
        pd.DataFrame(
            {
                "confidence_level": ["HIGH", "MEDIUM", "LOW"],
                "accuracy": [0.9, 0.7, 0.8],
            }
        )
    )
    missing_level = evaluate_confidence_module._compute_holdout_monotonic(
        pd.DataFrame(
            {
                "confidence_level": ["HIGH", "LOW"],
                "accuracy": [0.9, 0.5],
            }
        )
    )

    expect_true(monotonic is True)
    expect_true(non_monotonic is False)
    expect_true(missing_level is None)


def test_confidence_level_counts_falls_back_to_rank_levels_when_prediction_level_missing() -> None:
    fallback_counts = evaluate_confidence_module._confidence_level_counts(
        pd.DataFrame({"rank_confidence_level": ["HIGH", "LOW", "LOW"]}),
        preferred_column="prediction_confidence_level",
        enabled=True,
    )
    disabled_counts = evaluate_confidence_module._confidence_level_counts(
        pd.DataFrame({"rank_confidence_level": ["HIGH"]}),
        preferred_column="prediction_confidence_level",
        enabled=False,
    )

    expect_equal(fallback_counts, {"LOW": 2, "HIGH": 1})
    expect_equal(disabled_counts, {})


def test_unanimous_ranking_produces_high_confidence():
    probabilities = np.array(
        [[[0.70, 0.20, 0.07, 0.03]], [[0.45, 0.30, 0.20, 0.05]], [[0.95, 0.03, 0.01, 0.01]], [[0.55, 0.25, 0.15, 0.05]]]
    )

    result = classify_ensemble_rank_based(probabilities, CLASSES, minimum_models=3)

    expect_equal(result["borda_prediction"].tolist(), ["a"])
    expect_equal(result["rank_confidence_level"].tolist(), ["HIGH"])
    expect_equal(result["top1_agreement"], pytest.approx([1.0]))
    expect_equal(result["prediction_mean_borda"], pytest.approx([100.0]))
    expect_true(result["borda_margin"][0] >= 20.0)


def test_probability_scale_invariance_and_average_tie_ranks():
    differently_scaled = np.array([[0.90, 0.06, 0.03, 0.01], [0.55, 0.25, 0.15, 0.05]])
    ranks = probabilities_to_ranks(differently_scaled)
    expect_true(np.array_equal(ranks[0], ranks[1]))

    tied = probabilities_to_ranks(np.array([[0.40, 0.40, 0.20]]))
    expect_equal(tied[0], pytest.approx([1.5, 1.5, 3.0]))


def test_strong_disagreement_has_more_dispersion_than_mild_disagreement():
    mild = np.array(
        [
            [[0.8, 0.1, 0.06, 0.04]],
            [[0.7, 0.1, 0.15, 0.05]],
            [[0.6, 0.2, 0.1, 0.1]],
            [[0.3, 0.5, 0.1, 0.1]],
            [[0.3, 0.4, 0.2, 0.1]],
        ]
    )
    strong = mild.copy()
    strong[3, 0] = [0.05, 0.60, 0.25, 0.10]
    strong[4, 0] = [0.01, 0.45, 0.30, 0.24]

    mild_result = classify_ensemble_rank_based(mild, CLASSES)
    strong_result = classify_ensemble_rank_based(strong, CLASSES)

    expect_true(strong_result["prediction_rank_std"][0] > mild_result["prediction_rank_std"][0])
    expect_true(strong_result["prediction_mean_borda"][0] < mild_result["prediction_mean_borda"][0])


def test_near_tie_is_low_confidence():
    probabilities = np.array([[[0.6, 0.3, 0.1]], [[0.6, 0.3, 0.1]], [[0.3, 0.6, 0.1]], [[0.3, 0.6, 0.1]]])

    result = classify_ensemble_rank_based(probabilities, ["a", "b", "c"])

    expect_equal(result["borda_margin"], pytest.approx([0.0]))
    expect_equal(result["rank_winner_tied"].tolist(), [True])
    expect_equal(result["rank_confidence_level"].tolist(), ["LOW"])
    expect_equal(result["rank_confidence_reason"].tolist(), ["borda_winner_tie"])


def test_high_and_medium_threshold_boundaries_are_inclusive():
    features = {
        "prediction_mean_borda": np.array([90.0, 75.0]),
        "prediction_rank_range": np.array([2.0, 4.0]),
        "n_models_used": np.array([4, 4]),
        "rank_agrees_with_prediction": np.array([True, True]),
        "rank_winner_tied": np.array([False, False]),
    }

    levels, valid, reasons = assign_confidence_levels(features)

    expect_equal(levels.tolist(), ["HIGH", "MEDIUM"])
    expect_equal(valid.tolist(), [True, True])
    expect_equal(reasons.tolist(), ["rank_confidence_high", "rank_confidence_medium"])


def test_prediction_disagreement_is_explicitly_low_but_data_remain_valid():
    probabilities = np.repeat(np.array([[[0.8, 0.15, 0.05]]]), repeats=3, axis=0)

    result = classify_ensemble_rank_based(probabilities, ["a", "b", "c"], predicted_class_indices=np.array([1]))

    expect_equal(result["borda_prediction"].tolist(), ["a"])
    expect_equal(result["rank_confidence_level"].tolist(), ["LOW"])
    expect_equal(result["rank_confidence_valid"].tolist(), [True])
    expect_equal(result["rank_confidence_reason"].tolist(), ["soft_vote_borda_disagreement"])
    expect_true(result["borda_margin"][0] < 0)


def test_too_few_models_is_flagged_and_downgraded():
    probabilities = np.repeat(np.array([[[0.8, 0.15, 0.05]]]), repeats=2, axis=0)

    result = classify_ensemble_rank_based(probabilities, ["a", "b", "c"], minimum_models=3)

    expect_equal(result["rank_confidence_level"].tolist(), ["LOW"])
    expect_equal(result["rank_confidence_valid"].tolist(), [False])
    expect_equal(result["rank_confidence_reason"].tolist(), ["insufficient_models"])


def test_invalid_values_and_class_mismatch_are_rejected():
    invalid = np.array([[[0.8, np.nan, 0.2]]])
    with pytest.raises(ValueError, match="Non-finite"):
        classify_ensemble_rank_based(invalid, ["a", "b", "c"])

    valid = np.array([[[0.8, 0.2]]])
    with pytest.raises(ValueError, match="canonical class list"):
        classify_ensemble_rank_based(valid, ["a", "b", "c"])

    empty = np.array([[[0.0, 0.0, 0.0]]])
    with pytest.raises(ValueError, match="positive total support"):
        classify_ensemble_rank_based(empty, ["a", "b", "c"])

    negative = np.array([[[0.9, 0.2, -0.1]]])
    with pytest.raises(ValueError, match="non-negative"):
        classify_ensemble_rank_based(negative, ["a", "b", "c"])


def test_equal_weights_reproduce_unweighted_aggregation():
    probabilities = np.array([[[0.7, 0.2, 0.1]], [[0.2, 0.7, 0.1]], [[0.6, 0.1, 0.3]]])
    ranks = probabilities_to_ranks(probabilities)

    unweighted = aggregate_rank_scores(ranks)
    weighted = aggregate_rank_scores(ranks, model_weights=np.ones(3))

    expect_equal(weighted, pytest.approx(unweighted))


def test_rank_confidence_configuration_validates_threshold_order():
    with pytest.raises(ValueError, match="Borda thresholds"):
        ClassificationPipelineConfig(
            project_dir=Path("unused"),
            target_column="label",
            feature_columns=["feature"],
            rank_confidence_high_min_borda=70.0,
            rank_confidence_medium_min_borda=75.0,
        )

    with pytest.raises(ValueError, match="Class-reliability precision thresholds"):
        ClassificationPipelineConfig(
            project_dir=Path("unused"),
            target_column="label",
            feature_columns=["feature"],
            class_reliability_high_min_precision=0.60,
            class_reliability_medium_min_precision=0.70,
        )
    with pytest.raises(ValueError, match="range thresholds"):
        ClassificationPipelineConfig(
            project_dir=Path("unused"),
            target_column="label",
            feature_columns=["feature"],
            rank_confidence_high_max_range=5,
            rank_confidence_medium_max_range=4,
        )


def test_prediction_frame_includes_rank_confidence_and_uses_low_for_review():
    step = object.__new__(PredictStep)
    step.config = SimpleNamespace(
        target_column="label",
        prediction_column="label_prediction",
        prediction_confidence_column="prediction_max_probability",
        prediction_confidence_level_column="prediction_confidence_level",
        prediction_review_column="prediction_needs_review",
        prediction_confidence_threshold=0.60,
        prediction_filled_column="label_filled",
        probability_prefix="probability",
    )
    dataset = pd.DataFrame({"row_id": ["1"], "label": [None]})
    probabilities = np.array([[0.8, 0.2]])
    rank_confidence = {
        "prediction_confidence_level": np.array(["LOW"]),
        "prediction_confidence_valid": np.array([False]),
        "prediction_confidence_reason": np.array(["insufficient_models"]),
        "prediction_mean_borda": np.array([100.0]),
        "prediction_rank_range": np.array([0.0]),
        "top1_agreement": np.array([1.0]),
        "top2_agreement": np.array([1.0]),
        "top3_agreement": np.array([1.0]),
        "prediction_mean_rank": np.array([1.0]),
        "prediction_median_rank": np.array([1.0]),
        "prediction_rank_std": np.array([0.0]),
        "prediction_rank_iqr": np.array([0.0]),
        "runner_up_class": np.array(["b"]),
        "runner_up_borda": np.array([0.0]),
        "borda_prediction": np.array(["a"]),
        "rank_winner_tied": np.array([False]),
        "rank_agrees_with_prediction": np.array([True]),
        "n_models_used": np.array([2]),
        "rank_confidence_level": np.array(["LOW"]),
        "rank_confidence_valid": np.array([False]),
        "rank_confidence_reason": np.array(["insufficient_models"]),
    }

    frame, _ = step._create_prediction_frame(dataset, np.array(["a"]), probabilities, ["a", "b"], rank_confidence=rank_confidence)

    expect_equal(frame.loc[0, "prediction_confidence_level"], "LOW")
    expect_equal(frame.loc[0, "prediction_max_probability"], pytest.approx(0.8))
    expect_false(bool(frame.loc[0, "prediction_borda_winner_tied"]))
    expect_true(bool(frame.loc[0, "prediction_needs_review"]))
    expect_equal(frame.loc[0, "prediction_rank_confidence_reason"], "insufficient_models")


def test_rank_confidence_requires_oof_artifact_for_incremental_resume():
    class Artifact:
        def __init__(self, present):
            self.present = present

        def exists(self):
            return self.present

    best_model = Artifact(True)
    cv_results = Artifact(True)
    oof = Artifact(False)

    class ModelDirectory:
        def __truediv__(self, filename):
            expect_equal(filename, "oof_probabilities.joblib")
            return oof

    step = object.__new__(TrainStep)
    step.config = SimpleNamespace(
        optimize_class_probabilities=False, rank_confidence_enabled=True, train_model_dir=lambda _name: ModelDirectory()
    )
    step._model_artifact_paths = lambda _name: (best_model, cv_results)

    expect_false(step._is_model_fully_trained("model", supports_predict_proba=True))
    oof.present = True
    expect_true(step._is_model_fully_trained("model", supports_predict_proba=True))


def test_selected_strategy_returns_member_probabilities(monkeypatch):
    class Estimator:
        classes_ = np.array([0, 1])

        def set_params(self, **_params):
            return self

        def fit(self, _x, _y):
            return self

        def predict_proba(self, x):
            return np.repeat(np.array([[0.7, 0.3]]), len(x), axis=0)

    monkeypatch.setattr(
        "ml_classification.shared.models.models.build_estimator_by_name", lambda *_args, **_kwargs: Estimator()
    )
    frame = pd.DataFrame({"feature": [1.0, 2.0]})

    predictions, probabilities, members = fit_and_predict_selected_strategy(
        config=SimpleNamespace(),
        X_fit=frame,
        y_fit=pd.Series([0, 1]),
        X_all=frame,
        selection={"selected_models": ["m1", "m2"], "labels": ["a", "b"]},
        model_specs={"m1": {"best_params": {}}, "m2": {"best_params": {}}},
        numeric_features=["feature"],
        categorical_features=[],
    )

    expect_equal(predictions.tolist(), ["a", "a"])
    expect_true(np.allclose(probabilities, [[0.7, 0.3], [0.7, 0.3]]))
    expect_equal(list(members), ["m1", "m2"])


def test_selected_strategy_rejects_model_class_order_mismatch(monkeypatch):
    class Estimator:
        classes_ = np.array([1, 0])

        def set_params(self, **_params):
            return self

        def fit(self, _x, _y):
            return self

        def predict_proba(self, x):
            return np.repeat(np.array([[0.7, 0.3]]), len(x), axis=0)

    monkeypatch.setattr(
        "ml_classification.shared.models.models.build_estimator_by_name",
        lambda *_args, **_kwargs: Estimator(),
    )
    frame = pd.DataFrame({"feature": [1.0, 2.0]})

    with pytest.raises(RuntimeError, match="class order"):
        fit_and_predict_selected_strategy(
            config=SimpleNamespace(),
            X_fit=frame,
            y_fit=pd.Series([0, 1]),
            X_all=frame,
            selection={"selected_models": ["m1"], "labels": ["a", "b"]},
            model_specs={"m1": {"best_params": {}}},
            numeric_features=["feature"],
            categorical_features=[],
        )


def test_evaluation_rejects_probability_class_order_mismatch():
    estimator = SimpleNamespace(
        classes_=np.array([1, 0]),
        predict_proba=lambda _x: np.array([[0.7, 0.3]]),
    )

    with pytest.raises(RuntimeError, match="probability class order"):
        EvaluateStep._validate_probability_class_order(estimator, ["a", "b"], "misaligned_model")


def test_prediction_uses_frozen_step_four_confidence_contract():
    step = object.__new__(PredictStep)
    step.config = SimpleNamespace(
        rank_confidence_enabled=True,
        class_reliability_enabled=True,
        rank_confidence_minimum_models=9,
        rank_confidence_high_min_borda=100.0,
        rank_confidence_high_max_range=0.0,
        rank_confidence_medium_min_borda=100.0,
        rank_confidence_medium_max_range=0.0,
    )
    frozen_thresholds = {
        "high_min_borda": 90.0,
        "high_max_range": 2.0,
        "medium_min_borda": 75.0,
        "medium_max_range": 4.0,
    }
    reliability = calculate_oof_class_reliability(
        np.array(["a"] * 100), np.array(["a"] * 100), ["a", "b"]
    )
    selection = {
        "selected_models": ["m1", "m2", "m3"],
        "labels": ["a", "b"],
        "confidence": {
            "enabled": True,
            "method": "rank_consensus_with_class_reliability_guard",
            "rank": {"minimum_models": 3, **frozen_thresholds},
            "class_reliability": reliability,
        },
    }
    members = {
        name: np.array([[0.8, 0.2]])
        for name in selection["selected_models"]
    }

    result, contract = step._calculate_rank_confidence(selection, np.array(["a"]), members)

    expect_equal(contract["source"], "selection_summary")
    expect_equal(contract["thresholds"], frozen_thresholds)
    expect_equal(contract["minimum_models"], 3)
    expect_equal(result["prediction_confidence_level"].tolist(), ["HIGH"])


def test_prediction_applies_frozen_oof_class_reliability():
    reliability = calculate_oof_class_reliability(
        np.array(["a"] * 95 + ["b"] * 5 + ["b"] * 20 + ["a"] * 80),
        np.array(["a"] * 100 + ["b"] * 100),
        ["a", "b"],
    )
    thresholds = {
        "high_min_borda": 90.0,
        "high_max_range": 2.0,
        "medium_min_borda": 75.0,
        "medium_max_range": 4.0,
    }
    selection = {
        "selected_models": ["m1", "m2", "m3"],
        "labels": ["a", "b"],
        "confidence": {
            "enabled": True,
            "method": "rank_consensus_with_class_reliability_guard",
            "rank": {"minimum_models": 3, **thresholds},
            "class_reliability": reliability,
        },
    }
    member_matrix = np.array([[0.8, 0.2], [0.2, 0.8]])
    members = {name: member_matrix for name in selection["selected_models"]}
    step = object.__new__(PredictStep)
    step.config = SimpleNamespace()

    result, _ = step._calculate_rank_confidence(selection, np.array(["a", "b"]), members)

    expect_equal(result["prediction_confidence_level"].tolist(), ["HIGH", "LOW"])
    expect_equal(result["class_oof_precision"].tolist(), pytest.approx([0.95, 0.20]))


def test_prediction_can_compute_class_reliability_without_applying_it():
    reliability = calculate_oof_class_reliability(
        np.array(["a"] * 95 + ["b"] * 5 + ["b"] * 20 + ["a"] * 80),
        np.array(["a"] * 100 + ["b"] * 100),
        ["a", "b"],
    )
    thresholds = {
        "high_min_borda": 90.0,
        "high_max_range": 2.0,
        "medium_min_borda": 75.0,
        "medium_max_range": 4.0,
    }
    selection = {
        "selected_models": ["m1", "m2", "m3"],
        "labels": ["a", "b"],
        "confidence": {
            "enabled": True,
            "method": "rank_consensus_with_class_reliability_guard",
            "class_reliability_enabled": False,
            "rank": {"minimum_models": 3, **thresholds},
            "class_reliability": reliability,
        },
    }
    member_matrix = np.array([[0.8, 0.2], [0.2, 0.8]])
    members = {name: member_matrix for name in selection["selected_models"]}
    step = object.__new__(PredictStep)
    step.config = SimpleNamespace()

    result, _ = step._calculate_rank_confidence(selection, np.array(["a", "b"]), members)

    expect_equal(result["prediction_confidence_level"].tolist(), ["HIGH", "HIGH"])
    expect_equal(result["class_oof_precision"].tolist(), pytest.approx([0.95, 0.20]))
    expect_true(result["prediction_class_reliability_applied"] is False)
    expect_equal(result["prediction_confidence_source"], "rank_only")


def test_confidence_summary_reports_accuracy_by_level():
    parcels = pd.DataFrame(
        {
            "true_label": ["a", "a", "b"],
            "predicted_class": ["a", "b", "b"],
            "correct": [True, False, True],
            "confidence_level": ["HIGH", "LOW", "HIGH"],
        }
    )

    summary, by_class = EvaluateStep._summarize_rank_confidence(parcels)

    high = summary.loc[summary["confidence_level"] == "HIGH"].iloc[0]
    expect_equal(high["parcels"], 2)
    expect_equal(high["accuracy"], pytest.approx(1.0))
    expect_false(by_class.empty)
