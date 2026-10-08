"""Tests for explainable post-prediction inspection scoring."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ml_classification.step_05_predict.libraries.inspection_priority.scoring import calculate_inspection_metrics
from ml_classification.step_05_predict.predict import PredictStep


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        target_column="declared_crop",
        prediction_column="predicted_crop",
        prediction_confidence_column="prediction_max_probability",
        prediction_confidence_level_column="prediction_confidence_level",
        prediction_review_column="prediction_needs_review",
        prediction_filled_column="declared_crop_filled",
        probability_prefix="probability",
        prediction_confidence_threshold=0.60,
        inspection_data_reliability_column="data_reliability_score",
        inspection_geometry_complexity_column="geom_shape_complexity_score",
        inspection_model_weight=0.60,
        inspection_data_weight=0.25,
        inspection_geometry_weight=0.15,
        inspection_low_data_reliability_threshold=0.60,
        inspection_medium_data_reliability_threshold=0.80,
        inspection_medium_geometry_risk_threshold=0.50,
        inspection_high_geometry_risk_threshold=0.75,
        inspection_medium_score_threshold=25.0,
        inspection_high_score_threshold=50.0,
        inspection_scoring_enabled=True,
    )


def test_inspection_metrics_combine_independent_risks_and_explain_reasons() -> None:
    parcels = pd.DataFrame(
        {
            "prediction_confidence_level": ["HIGH", "MEDIUM", "LOW", "HIGH", None],
            "data_reliability_score": [1.0, 0.8, 0.4, 0.35, 0.9],
            "geom_shape_complexity_score": [1.0, 2.0, 3.0, 4.0, np.nan],
        }
    )

    result = calculate_inspection_metrics(parcels, _config())

    assert result["confidence_risk"].iloc[:4].tolist() == [0.0, 0.5, 1.0, 0.0]
    assert result["data_risk"].iloc[:4].tolist() == pytest.approx([0.0, 0.2, 0.6, 0.65])
    assert result["geometry_risk"].iloc[:4].tolist() == pytest.approx([0.25, 0.50, 0.75, 1.0])
    assert result["inspection_score"].iloc[:4].tolist() == pytest.approx([3.75, 42.5, 86.25, 31.25])
    assert result["inspection_need"].tolist() == ["LOW", "MEDIUM", "HIGH", "MEDIUM", "UNKNOWN"]
    assert result.loc[0, "inspection_reasons"] == "NONE"
    assert result.loc[1, "inspection_reasons"] == (
        "INSUFFICIENT_EO_EVIDENCE|MEDIUM_MODEL_CONFIDENCE|MEDIUM_GEOMETRY_COMPLEXITY"
    )
    assert result.loc[2, "inspection_reasons"] == (
        "INSUFFICIENT_EO_EVIDENCE|LOW_MODEL_CONFIDENCE|LOW_DATA_RELIABILITY|HIGH_GEOMETRY_COMPLEXITY"
    )
    assert result.loc[3, "inspection_reasons"] == (
        "INSUFFICIENT_EO_EVIDENCE|LOW_DATA_RELIABILITY|HIGH_GEOMETRY_COMPLEXITY"
    )
    assert result.loc[4, "inspection_reasons"] == (
        "INSUFFICIENT_EO_EVIDENCE|INVALID_MODEL_CONFIDENCE|MISSING_GEOMETRY_COMPLEXITY"
    )
    assert pd.isna(result.loc[4, "inspection_score"])


def test_label_aware_policy_reproduces_operational_examples() -> None:
    parcels = pd.DataFrame(
        {
            "declared_crop": ["Cotton", "Cotton", "Cotton", "Cotton", "Cotton", "Cotton"],
            "predicted_crop": ["Cotton", "Maize", "Maize", "Cotton", "Maize", "Maize"],
            "prediction_confidence_level": ["HIGH", "HIGH", "LOW", "LOW", "HIGH", "HIGH"],
            "data_reliability_score": [0.9, 0.9, 0.4, 0.4, 0.4, 0.9],
            "geom_shape_complexity_score": [1.0, 1.0, 4.0, 4.0, 4.0, 4.0],
        }
    )

    result = calculate_inspection_metrics(parcels, _config())

    assert result["label_prediction_status"].tolist() == [
        "SAME",
        "DIFFERENT",
        "DIFFERENT",
        "SAME",
        "DIFFERENT",
        "DIFFERENT",
    ]
    assert result["inspection_need"].tolist() == [
        "LOW",
        "VERY_HIGH",
        "MEDIUM_UNCERTAIN",
        "MEDIUM",
        "MEDIUM_HIGH",
        "HIGH",
    ]
    assert result["inspection_check_type"].tolist() == [
        "NONE",
        "DECLARATION_CONFLICT",
        "INSUFFICIENT_EO_EVIDENCE",
        "INSUFFICIENT_EO_EVIDENCE",
        "INSUFFICIENT_EO_EVIDENCE",
        "DECLARATION_CONFLICT",
    ]
    assert result.loc[1, "inspection_reasons"] == "STRONG_DECLARATION_CONFLICT"
    assert result.loc[2, "inspection_reasons"].startswith(
        "DECLARATION_PREDICTION_DIFFERENCE|INSUFFICIENT_EO_EVIDENCE"
    )
    assert result.loc[5, "inspection_reasons"].startswith("DECLARATION_CONFLICT|HIGH_GEOMETRY_COMPLEXITY")


def test_inspection_geometry_risk_uses_average_percentile_for_ties() -> None:
    parcels = pd.DataFrame(
        {
            "prediction_confidence_level": ["HIGH", "HIGH", "HIGH"],
            "data_reliability_score": [1.0, 1.0, 1.0],
            "geom_shape_complexity_score": [1.0, 1.0, 3.0],
        }
    )

    result = calculate_inspection_metrics(parcels, _config())

    assert result["geometry_risk"].tolist() == pytest.approx([0.5, 0.5, 1.0])


def test_inspection_scoring_rejects_missing_source_columns() -> None:
    with pytest.raises(ValueError, match="geom_shape_complexity_score"):
        calculate_inspection_metrics(
            pd.DataFrame(
                {
                    "prediction_confidence_level": ["HIGH"],
                    "data_reliability_score": [1.0],
                }
            ),
            _config(),
        )


def test_predict_step_attaches_inspection_outputs_and_summary() -> None:
    step = object.__new__(PredictStep)
    step.config = _config()
    parcels = pd.DataFrame(
        {
            "prediction_confidence_level": ["HIGH", "LOW"],
            "data_reliability_score": [0.9, 0.4],
            "geom_shape_complexity_score": [1.0, 3.0],
        }
    )

    result, summary = step._add_inspection_metrics(parcels)

    assert summary["inspection_scoring_available"]
    assert summary["inspection_weights"] == {"model": 0.60, "data": 0.25, "geometry": 0.15}
    assert {
        "confidence_risk",
        "data_risk",
        "geometry_risk",
        "inspection_score",
        "label_prediction_status",
        "inspection_check_type",
    }.issubset(result.columns)


def test_predict_step_calculates_prediction_quality_without_full_run() -> None:
    step = object.__new__(PredictStep)
    step.config = _config()
    parcels = pd.DataFrame(
        {
            "declared_crop": ["Cotton", "Cotton"],
            "data_reliability_score": [0.9, 0.7],
            "geom_shape_complexity_score": [1.0, 2.0],
        }
    )
    predictions = np.array(["Cotton", "Maize"])
    probabilities = np.array([[0.9, 0.1], [0.2, 0.8]])
    member_probabilities = {
        "m1": np.array([[0.9, 0.1], [0.2, 0.8]]),
        "m2": np.array([[0.8, 0.2], [0.3, 0.7]]),
    }
    selection = {
        "labels": ["Cotton", "Maize"],
        "selected_models": ["m1", "m2"],
        "confidence": {
            "enabled": True,
            "method": "rank_consensus_only",
            "rank": {"minimum_models": 2, "high_min_borda": 90.0, "high_max_range": 2.0, "medium_min_borda": 75.0, "medium_max_range": 4.0},
            "class_reliability": None,
        },
    }

    result, unknown_mask, rank_confidence, rank_contract, inspection_summary = step.calculate_prediction_quality(
        dataset=parcels,
        predictions=predictions,
        probabilities=probabilities,
        selection=selection,
        member_probabilities=member_probabilities,
    )

    assert unknown_mask.tolist() == [False, False]
    assert rank_contract["enabled"] is True
    assert rank_confidence is not None
    assert "prediction_confidence_level" in result.columns
    assert result["inspection_score"].notna().all()
    assert inspection_summary["inspection_scoring_available"]


def test_predict_step_skips_inspection_scoring_when_inputs_are_absent() -> None:
    step = object.__new__(PredictStep)
    step.config = _config()

    with pytest.warns(RuntimeWarning, match="required columns are missing"):
        result, summary = step._add_inspection_metrics(pd.DataFrame({"prediction_confidence_level": ["HIGH"]}))

    assert list(result.columns) == ["prediction_confidence_level"]
    assert not summary["inspection_scoring_available"]
    assert summary["inspection_missing_columns"] == ["data_reliability_score", "geom_shape_complexity_score"]
