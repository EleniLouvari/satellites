from copy import deepcopy

import numpy as np
import pytest

from ml_classification.shared.class_reliability import combine_confidence_components, get_class_reliability
from ml_classification.shared.rank_confidence import (
    assign_confidence_levels,
    calculate_rank_confidence,
    probabilities_to_ranks,
)
from ml_classification.step_04_evaluate.libraries.class_reliability import calculate_oof_class_reliability
from tests.utils import expect_equal, expect_false, expect_true


def test_rank_results_are_probability_scale_invariant_and_average_ties():
    first = np.array([[[0.4, 0.4, 0.2]], [[0.8, 0.8, 0.1]], [[0.2, 0.2, 0.01]]])
    second = np.array([[[0.9, 0.9, 0.01]], [[0.3, 0.3, 0.2]], [[10.0, 10.0, 1.0]]])
    expect_equal(probabilities_to_ranks(first[0, 0]).tolist(), [1.5, 1.5, 3.0])
    first_result = calculate_rank_confidence(first, np.array([0]), ["a", "b", "c"])
    second_result = calculate_rank_confidence(second, np.array([0]), ["a", "b", "c"])
    expect_equal(first_result["prediction_mean_borda"], pytest.approx(second_result["prediction_mean_borda"]))
    expect_equal(first_result["prediction_rank_range"], pytest.approx(second_result["prediction_rank_range"]))


def test_unanimous_borda_winner_is_high():
    members = np.array([[[0.9, 0.08, 0.02]]] * 5)
    result = calculate_rank_confidence(members, np.array([0]), ["a", "b", "c"])
    expect_equal(result["prediction_mean_borda"].tolist(), [100.0])
    expect_equal(result["prediction_rank_range"].tolist(), [0.0])
    expect_equal(result["rank_confidence_level"].tolist(), ["HIGH"])


def test_rank_boundaries_are_inclusive():
    features = {
        "prediction_mean_borda": np.array([90.0, 75.0]),
        "prediction_rank_range": np.array([2.0, 4.0]),
        "n_models_used": np.array([5, 5]),
        "rank_agrees_with_prediction": np.array([True, True]),
        "rank_winner_tied": np.array([False, False]),
    }
    levels, valid, _ = assign_confidence_levels(features)
    expect_equal(levels.tolist(), ["HIGH", "MEDIUM"])
    expect_equal(valid.tolist(), [True, True])


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
    expect_equal(levels.tolist(), ["LOW"])
    expect_equal(valid.tolist(), [models >= 3])
    expect_equal(reasons.tolist(), [reason])


@pytest.mark.parametrize(("correct", "expected_level"), [(80, "HIGH"), (60, "MEDIUM"), (59, "LOW")])
def test_oof_precision_thresholds(correct, expected_level):
    true = np.array(["a"] * correct + ["b"] * (100 - correct))
    predicted = np.array(["a"] * 100)
    contract = calculate_oof_class_reliability(true, predicted, ["a", "b"])
    expect_equal(contract["classes"]["a"]["precision"], pytest.approx(correct / 100))
    expect_equal(contract["classes"]["a"]["level"], expected_level)
    expect_true(contract["classes"]["a"]["valid"])


def test_insufficient_class_support_is_not_pooled():
    true = np.array(["a"] * 99 + ["b"] * 100)
    predicted = np.array(["a"] * 99 + ["b"] * 100)
    contract = calculate_oof_class_reliability(true, predicted, ["a", "b"])
    expect_equal(contract["classes"]["a"]["support"], 99)
    expect_equal(contract["classes"]["a"]["level"], "LOW")
    expect_false(contract["classes"]["a"]["valid"])
    expect_equal(contract["classes"]["a"]["reason"], "insufficient_oof_class_support")
    applied = get_class_reliability(np.array(["a"]), contract)
    expect_equal(applied["class_oof_support"].tolist(), [99])


def test_oof_contract_validates_alignment_and_label_sets():
    with pytest.raises(ValueError, match="aligned one-dimensional arrays"):
        calculate_oof_class_reliability(np.array(["a", "b"]), np.array(["a"]), ["a", "b"])
    with pytest.raises(ValueError, match="requires at least one prediction"):
        calculate_oof_class_reliability(np.array([]), np.array([]), ["a", "b"])
    with pytest.raises(ValueError, match="at least two unique canonical classes"):
        calculate_oof_class_reliability(np.array(["a", "b"]), np.array(["a", "b"]), ["a", "a"])
    with pytest.raises(ValueError, match="outside the canonical class set"):
        calculate_oof_class_reliability(np.array(["a", "c"]), np.array(["a", "a"]), ["a", "b"])


@pytest.mark.parametrize(
    ("thresholds", "match"),
    [
        ({"minimum_support": 0}, "minimum_support must be at least 1"),
        ({"high_min_precision": 0.5, "medium_min_precision": 0.6}, "0 <= MEDIUM <= HIGH <= 1"),
    ],
)
def test_oof_contract_validates_threshold_ranges(thresholds, match):
    kwargs = {"minimum_support": 2, "high_min_precision": 0.8, "medium_min_precision": 0.6}
    kwargs.update(thresholds)
    with pytest.raises(ValueError, match=match):
        calculate_oof_class_reliability(np.array(["a", "b"]), np.array(["a", "b"]), ["a", "b"], **kwargs)


def test_oof_contract_marks_below_medium_precision_as_valid_low():
    true = np.array(["a"] * 60 + ["b"] * 40)
    predicted = np.array(["a"] * 60 + ["b"] * 40)
    contract = calculate_oof_class_reliability(true, predicted, ["a", "b"], minimum_support=10, high_min_precision=0.80, medium_min_precision=0.60)
    expect_equal(contract["classes"]["a"]["level"], "HIGH")
    expect_equal(contract["classes"]["b"]["level"], "HIGH")

    low_precision = calculate_oof_class_reliability(
        np.array(["a"] * 55 + ["b"] * 45),
        np.array(["a"] * 100),
        ["a", "b"],
        minimum_support=10,
        high_min_precision=0.80,
        medium_min_precision=0.60,
    )
    expect_equal(low_precision["classes"]["a"]["level"], "LOW")
    expect_equal(low_precision["classes"]["a"]["reason"], "precision_below_medium_threshold")
    expect_true(low_precision["classes"]["a"]["valid"])
    expect_equal(low_precision["classes"]["b"]["level"], "LOW")
    expect_false(low_precision["classes"]["b"]["valid"])
    expect_equal(low_precision["classes"]["b"]["reason"], "insufficient_oof_class_support")


def test_holdout_lookup_does_not_modify_frozen_oof_contract():
    contract = calculate_oof_class_reliability(np.array(["a"] * 80 + ["b"] * 20), np.array(["a"] * 100), ["a", "b"])
    frozen = deepcopy(contract)
    get_class_reliability(np.array(["a", "b", "a"]), contract)
    expect_equal(contract, frozen)


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
    expect_equal(result["prediction_confidence_level"].tolist(), [
        "HIGH",
        "MEDIUM",
        "LOW",
        "MEDIUM",
        "MEDIUM",
        "LOW",
        "LOW",
        "LOW",
        "LOW",
    ])
    invalid = combine_confidence_components(
        np.array(["HIGH", "HIGH"]),
        np.array([False, True]),
        np.array(["insufficient_models", "rank_confidence_high"]),
        np.array(["HIGH", "HIGH"]),
        np.array([True, False]),
        np.array(["sufficient_oof_support", "insufficient_oof_class_support"]),
    )
    expect_equal(invalid["prediction_confidence_level"].tolist(), ["LOW", "LOW"])
    expect_equal(invalid["prediction_confidence_valid"].tolist(), [False, False])
