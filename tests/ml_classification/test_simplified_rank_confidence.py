from copy import deepcopy

import numpy as np
import pytest

from ml_classification.shared.class_reliability import combine_confidence_components, get_class_reliability
from ml_classification.step_04_evaluate.libraries.class_reliability import calculate_oof_class_reliability
from ml_classification.shared.rank_confidence import (
    assign_confidence_levels,
    calculate_rank_confidence,
    probabilities_to_ranks,
)


def test_rank_results_are_probability_scale_invariant_and_average_ties():
    first = np.array([[[0.4, 0.4, 0.2]], [[0.8, 0.8, 0.1]], [[0.2, 0.2, 0.01]]])
    second = np.array([[[0.9, 0.9, 0.01]], [[0.3, 0.3, 0.2]], [[10.0, 10.0, 1.0]]])
    assert probabilities_to_ranks(first[0, 0]).tolist() == [1.5, 1.5, 3.0]
    first_result = calculate_rank_confidence(first, np.array([0]), ["a", "b", "c"])
    second_result = calculate_rank_confidence(second, np.array([0]), ["a", "b", "c"])
    assert first_result["prediction_mean_borda"] == pytest.approx(second_result["prediction_mean_borda"])
    assert first_result["prediction_rank_range"] == pytest.approx(second_result["prediction_rank_range"])


def test_unanimous_borda_winner_is_high():
    members = np.array([[[0.9, 0.08, 0.02]]] * 5)
    result = calculate_rank_confidence(members, np.array([0]), ["a", "b", "c"])
    assert result["prediction_mean_borda"].tolist() == [100.0]
    assert result["prediction_rank_range"].tolist() == [0.0]
    assert result["rank_confidence_level"].tolist() == ["HIGH"]


def test_rank_boundaries_are_inclusive():
    features = {
        "prediction_mean_borda": np.array([90.0, 75.0]),
        "prediction_rank_range": np.array([2.0, 4.0]),
        "n_models_used": np.array([5, 5]),
        "rank_agrees_with_prediction": np.array([True, True]),
        "rank_winner_tied": np.array([False, False]),
    }
    levels, valid, _ = assign_confidence_levels(features)
    assert levels.tolist() == ["HIGH", "MEDIUM"]
    assert valid.tolist() == [True, True]


@pytest.mark.parametrize(
    ("agrees", "tied", "models", "reason"),
    [
        (False, False, 5, "soft_vote_borda_disagreement"),
        (True, True, 5, "borda_winner_tie"),
        (True, False, 2, "insufficient_models"),
    ],
)
def test_structural_rank_safeguards_are_low(agrees, tied, models, reason):
    features = {
        "prediction_mean_borda": np.array([100.0]),
        "prediction_rank_range": np.array([0.0]),
        "n_models_used": np.array([models]),
        "rank_agrees_with_prediction": np.array([agrees]),
        "rank_winner_tied": np.array([tied]),
    }
    levels, valid, reasons = assign_confidence_levels(features)
    assert levels.tolist() == ["LOW"]
    assert valid.tolist() == [models >= 3]
    assert reasons.tolist() == [reason]


@pytest.mark.parametrize(("correct", "expected_level"), [(80, "HIGH"), (60, "MEDIUM"), (59, "LOW")])
def test_oof_precision_thresholds(correct, expected_level):
    true = np.array(["a"] * correct + ["b"] * (100 - correct))
    predicted = np.array(["a"] * 100)
    contract = calculate_oof_class_reliability(true, predicted, ["a", "b"])
    assert contract["classes"]["a"]["precision"] == pytest.approx(correct / 100)
    assert contract["classes"]["a"]["level"] == expected_level
    assert contract["classes"]["a"]["valid"]


def test_insufficient_class_support_is_not_pooled():
    true = np.array(["a"] * 99 + ["b"] * 100)
    predicted = np.array(["a"] * 99 + ["b"] * 100)
    contract = calculate_oof_class_reliability(true, predicted, ["a", "b"])
    assert contract["classes"]["a"]["support"] == 99
    assert contract["classes"]["a"]["level"] == "LOW"
    assert not contract["classes"]["a"]["valid"]
    assert contract["classes"]["a"]["reason"] == "insufficient_oof_class_support"
    applied = get_class_reliability(np.array(["a"]), contract)
    assert applied["class_oof_support"].tolist() == [99]


def test_holdout_lookup_does_not_modify_frozen_oof_contract():
    contract = calculate_oof_class_reliability(np.array(["a"] * 80 + ["b"] * 20), np.array(["a"] * 100), ["a", "b"])
    frozen = deepcopy(contract)
    get_class_reliability(np.array(["a", "b", "a"]), contract)
    assert contract == frozen


def test_complete_final_confidence_matrix_and_invalid_evidence():
    rank = np.repeat(np.array(["HIGH", "MEDIUM", "LOW"]), 3)
    reliability = np.tile(np.array(["HIGH", "MEDIUM", "LOW"]), 3)
    result = combine_confidence_components(
        rank,
        np.ones(9, dtype=bool),
        np.array([f"rank_confidence_{value.lower()}" for value in rank]),
        reliability,
        np.ones(9, dtype=bool),
        np.array(["sufficient_oof_support"] * 9),
    )
    assert result["prediction_confidence_level"].tolist() == [
        "HIGH",
        "MEDIUM",
        "LOW",
        "MEDIUM",
        "MEDIUM",
        "LOW",
        "LOW",
        "LOW",
        "LOW",
    ]
    invalid = combine_confidence_components(
        np.array(["HIGH", "HIGH"]),
        np.array([False, True]),
        np.array(["insufficient_models", "rank_confidence_high"]),
        np.array(["HIGH", "HIGH"]),
        np.array([True, False]),
        np.array(["sufficient_oof_support", "insufficient_oof_class_support"]),
    )
    assert invalid["prediction_confidence_level"].tolist() == ["LOW", "LOW"]
    assert invalid["prediction_confidence_valid"].tolist() == [False, False]
