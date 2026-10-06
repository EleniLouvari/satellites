from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from satellites.ml_classification.step_04_evaluate.libraries.class_reliability import calculate_oof_class_reliability
from satellites.ml_classification.shared.config.config import ClassificationPipelineConfig
from satellites.ml_classification.shared.rank_confidence import (
    aggregate_rank_scores,
    assign_confidence_levels,
    classify_ensemble_rank_based,
    probabilities_to_ranks,
)
from satellites.ml_classification.core.rank_confidence_calibration import (
    apply_calibration_to_parcel_frame,
    fit_class_aware_confidence_calibration,
)
from satellites.ml_classification.step_03_train.train import TrainStep
from satellites.ml_classification.step_04_evaluate.evaluate import EvaluateStep
from satellites.ml_classification.step_05_predict.libraries.refit import fit_and_predict_selected_strategy
from satellites.ml_classification.step_05_predict.predict import PredictStep

CLASSES = ["a", "b", "c", "d"]


def test_unanimous_ranking_produces_high_confidence():
    probabilities = np.array(
        [[[0.70, 0.20, 0.07, 0.03]], [[0.45, 0.30, 0.20, 0.05]], [[0.95, 0.03, 0.01, 0.01]], [[0.55, 0.25, 0.15, 0.05]]]
    )

    result = classify_ensemble_rank_based(probabilities, CLASSES, minimum_models=3)

    assert result["borda_prediction"].tolist() == ["a"]
    assert result["rank_confidence_level"].tolist() == ["HIGH"]
    assert result["top1_agreement"] == pytest.approx([1.0])
    assert result["prediction_mean_borda"] == pytest.approx([100.0])
    assert result["borda_margin"][0] >= 20.0


def test_probability_scale_invariance_and_average_tie_ranks():
    differently_scaled = np.array([[0.90, 0.06, 0.03, 0.01], [0.55, 0.25, 0.15, 0.05]])
    ranks = probabilities_to_ranks(differently_scaled)
    assert np.array_equal(ranks[0], ranks[1])

    tied = probabilities_to_ranks(np.array([[0.40, 0.40, 0.20]]))
    assert tied[0] == pytest.approx([1.5, 1.5, 3.0])


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

    assert strong_result["prediction_rank_std"][0] > mild_result["prediction_rank_std"][0]
    assert strong_result["prediction_mean_borda"][0] < mild_result["prediction_mean_borda"][0]


def test_near_tie_is_low_confidence():
    probabilities = np.array([[[0.6, 0.3, 0.1]], [[0.6, 0.3, 0.1]], [[0.3, 0.6, 0.1]], [[0.3, 0.6, 0.1]]])

    result = classify_ensemble_rank_based(probabilities, ["a", "b", "c"])

    assert result["borda_margin"] == pytest.approx([0.0])
    assert result["rank_winner_tied"].tolist() == [True]
    assert result["rank_confidence_level"].tolist() == ["LOW"]
    assert result["rank_confidence_reason"].tolist() == ["rank_winner_tie"]


def test_high_and_medium_threshold_boundaries_are_inclusive():
    features = {
        "prediction_mean_borda": np.array([90.0, 75.0]),
        "prediction_rank_range": np.array([2.0, 4.0]),
        "n_models_used": np.array([4, 4]),
        "rank_agrees_with_prediction": np.array([True, True]),
        "rank_winner_tied": np.array([False, False]),
    }

    levels, valid, reasons = assign_confidence_levels(features)

    assert levels.tolist() == ["HIGH", "MEDIUM"]
    assert valid.tolist() == [True, True]
    assert reasons.tolist() == ["rank_confidence_high", "rank_confidence_medium"]


def test_prediction_disagreement_is_explicitly_low_but_data_remain_valid():
    probabilities = np.repeat(np.array([[[0.8, 0.15, 0.05]]]), repeats=3, axis=0)

    result = classify_ensemble_rank_based(probabilities, ["a", "b", "c"], predicted_class_indices=np.array([1]))

    assert result["borda_prediction"].tolist() == ["a"]
    assert result["rank_confidence_level"].tolist() == ["LOW"]
    assert result["rank_confidence_valid"].tolist() == [True]
    assert result["rank_confidence_reason"].tolist() == ["soft_vote_borda_disagreement"]
    assert result["borda_margin"][0] < 0


def test_too_few_models_is_flagged_and_downgraded():
    probabilities = np.repeat(np.array([[[0.8, 0.15, 0.05]]]), repeats=2, axis=0)

    result = classify_ensemble_rank_based(probabilities, ["a", "b", "c"], minimum_models=3)

    assert result["rank_confidence_level"].tolist() == ["LOW"]
    assert result["rank_confidence_valid"].tolist() == [False]
    assert result["rank_confidence_reason"].tolist() == ["insufficient_models"]


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

    assert weighted == pytest.approx(unweighted)


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

    assert frame.loc[0, "prediction_confidence_level"] == "LOW"
    assert frame.loc[0, "prediction_max_probability"] == pytest.approx(0.8)
    assert not bool(frame.loc[0, "prediction_borda_winner_tied"])
    assert bool(frame.loc[0, "prediction_needs_review"])
    assert frame.loc[0, "prediction_rank_confidence_reason"] == "insufficient_models"


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
            assert filename == "oof_probabilities.joblib"
            return oof

    step = object.__new__(TrainStep)
    step.config = SimpleNamespace(
        optimize_class_probabilities=False, rank_confidence_enabled=True, train_model_dir=lambda _name: ModelDirectory()
    )
    step._model_artifact_paths = lambda _name: (best_model, cv_results)

    assert not step._is_model_fully_trained("model", supports_predict_proba=True)
    oof.present = True
    assert step._is_model_fully_trained("model", supports_predict_proba=True)


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
        "satellites.ml_classification.shared.models.models.build_estimator_by_name", lambda *_args, **_kwargs: Estimator()
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

    assert predictions.tolist() == ["a", "a"]
    assert np.allclose(probabilities, [[0.7, 0.3], [0.7, 0.3]])
    assert list(members) == ["m1", "m2"]


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
        "satellites.ml_classification.shared.models.models.build_estimator_by_name",
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

    assert contract["source"] == "selection_summary"
    assert contract["thresholds"] == frozen_thresholds
    assert contract["minimum_models"] == 3
    assert result["prediction_confidence_level"].tolist() == ["HIGH"]


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

    assert result["prediction_confidence_level"].tolist() == ["HIGH", "LOW"]
    assert result["class_oof_precision"].tolist() == pytest.approx([0.95, 0.20])


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

    assert result["prediction_confidence_level"].tolist() == ["HIGH", "HIGH"]
    assert result["class_oof_precision"].tolist() == pytest.approx([0.95, 0.20])
    assert result["prediction_class_reliability_applied"].tolist() == [False, False]
    assert result["prediction_confidence_source"].tolist() == ["rank_only", "rank_only"]


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
    assert high["parcels"] == 2
    assert high["accuracy"] == pytest.approx(1.0)
    assert not by_class.empty


def _calibration_rows(predicted_class, top1_count, correct_count, total, n_models=5):
    return pd.DataFrame(
        {
            "predicted_class": [predicted_class] * total,
            "correct": [True] * correct_count + [False] * (total - correct_count),
            "top1_agreement": [top1_count / n_models] * total,
            "n_models_used": [n_models] * total,
            "confidence_valid": [True] * total,
            "rank_agrees_with_prediction": [True] * total,
            "rank_winner_tied": [False] * total,
            "confidence_reason": [""] * total,
        }
    )


def test_oof_calibration_is_class_aware_for_identical_rank_agreement():
    oof = pd.concat(
        [
            _calibration_rows("reliable", 5, correct_count=95, total=100),
            _calibration_rows("biased", 5, correct_count=20, total=100),
        ],
        ignore_index=True,
    )
    calibration = fit_class_aware_confidence_calibration(oof, minimum_oof_support=100)
    scored = apply_calibration_to_parcel_frame(oof.iloc[[0, 100]].copy(), calibration)

    assert scored["confidence_level"].tolist() == ["HIGH", "LOW"]
    assert scored["confidence_empirical_accuracy"].tolist() == pytest.approx([0.95, 0.20])
    assert scored["confidence_calibration_support"].tolist() == [100, 100]


def test_oof_calibration_pools_only_bounded_adjacent_top1_bins():
    oof = pd.concat(
        [
            _calibration_rows("a", 5, correct_count=36, total=40),
            _calibration_rows("a", 4, correct_count=30, total=40),
            _calibration_rows("a", 3, correct_count=15, total=20),
        ],
        ignore_index=True,
    )
    calibration = fit_class_aware_confidence_calibration(oof, minimum_oof_support=100)
    top1_four = next(
        row for row in calibration["table"] if row["predicted_class"] == "a" and row["top1_count"] == 4
    )

    assert top1_four["calibration_valid"]
    assert top1_four["support"] == 100
    assert top1_four["exact_support"] == 40
    assert top1_four["pooled_support"] == 100
    assert top1_four["max_pool_distance"] == 1
    assert top1_four["included_top1_counts"] == [3, 4, 5]
    assert top1_four["calibration_source"] == "adjacent_class_top1_bins"
    assert top1_four["empirical_accuracy"] == pytest.approx(0.81)
    assert top1_four["confidence_level"] == "MEDIUM"


def test_calibration_does_not_extrapolate_from_distant_top1_bins():
    # Provide strong OOF evidence only for unanimous 5/5 Top-1 agreement.
    oof = _calibration_rows("a", top1_count=5, correct_count=95, total=100, n_models=5)
    # Fit with the production default of one adjacent vote-count bin.
    calibration = fit_class_aware_confidence_calibration(oof, minimum_oof_support=100)

    # Inspect the unseen zero-vote class/Top-1 combination.
    zero_votes = next(
        row
        for row in calibration["table"]
        if row["predicted_class"] == "a" and row["top1_count"] == 0
    )

    # Distant 5/5 evidence must never be extrapolated to the unseen 0/5 pattern.
    assert not zero_votes["calibration_valid"]
    assert zero_votes["confidence_level"] == "LOW"
    assert zero_votes["calibration_source"] == "unseen_class_top1_bin"
    assert zero_votes["exact_support"] == 0
    assert zero_votes["pooled_support"] == 0
    assert zero_votes["support"] == 0
    assert zero_votes["max_pool_distance"] == 0
    assert zero_votes["included_top1_counts"] == []

    # Applying the frozen row must expose the unseen-bin reason to downstream review.
    zero_vote_prediction = oof.iloc[[0]].copy()
    zero_vote_prediction["top1_agreement"] = 0.0
    scored = apply_calibration_to_parcel_frame(zero_vote_prediction, calibration)
    assert not bool(scored.iloc[0]["confidence_valid"])
    assert scored.iloc[0]["confidence_reason"] == "unseen_class_top1_bin"


def test_oof_calibration_downgrades_classes_without_minimum_support():
    oof = _calibration_rows("rare", 5, correct_count=49, total=50)
    calibration = fit_class_aware_confidence_calibration(oof, minimum_oof_support=100)
    scored = apply_calibration_to_parcel_frame(oof.iloc[[0]].copy(), calibration)

    assert scored.loc[0, "confidence_level"] == "LOW"
    assert not bool(scored.loc[0, "confidence_valid"])
    assert scored.loc[0, "confidence_reason"] == "insufficient_oof_calibration_support"


def test_oof_calibration_preserves_valid_structural_low_safeguard():
    oof = _calibration_rows("a", 5, correct_count=95, total=100)
    calibration = fit_class_aware_confidence_calibration(oof, minimum_oof_support=100)
    row = oof.iloc[[0]].copy()
    row["rank_agrees_with_prediction"] = False
    row["confidence_reason"] = "rank_prediction_disagreement"

    scored = apply_calibration_to_parcel_frame(row, calibration)

    assert scored.loc[0, "confidence_level"] == "LOW"
    assert bool(scored.loc[0, "confidence_valid"])
    assert scored.loc[0, "confidence_reason"] == "rank_prediction_disagreement"
