"""Tests for explainable post-prediction inspection scoring."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ml_classification.step_05_predict.libraries.inspection_priority.scoring import (
    _apply_label_aware_policy,
    _inspection_reasons,
    calculate_inspection_metrics,
)
from ml_classification.step_05_predict.predict import PredictStep
from tests.utils import expect_equal, expect_false, expect_in, expect_is_not_none, expect_startswith, expect_subset, expect_true


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
        class_reliability_enabled=True,
    )


def test_inspection_metrics_combine_independent_risks_and_explain_reasons() -> None:
    parcels = pd.DataFrame(
        {
            "predicted_crop": ["Cotton"] * 5,
            "prediction_confidence_level": ["HIGH", "MEDIUM", "LOW", "HIGH", None],
            "data_reliability_score": [1.0, 0.8, 0.4, 0.35, 0.9],
            "geom_shape_complexity_score": [1.0, 2.0, 3.0, 4.0, np.nan],
        }
    )

    result = calculate_inspection_metrics(parcels, _config())

    expect_equal(result["confidence_risk"].iloc[:4].tolist(), [0.0, 0.5, 1.0, 0.0])
    expect_equal(result["data_risk"].iloc[:4].tolist(), pytest.approx([0.0, 0.2, 0.6, 0.65]))
    expect_equal(result["geometry_risk"].iloc[:4].tolist(), pytest.approx([0.25, 0.50, 0.75, 1.0]))
    expect_equal(result["inspection_score"].iloc[:4].tolist(), pytest.approx([3.75, 42.5, 86.25, 31.25]))
    expect_equal(result["inspection_need"].tolist(), ["MEDIUM", "MEDIUM", "MEDIUM", "MEDIUM", "UNKNOWN"])
    expect_equal(result.loc[0, "inspection_reasons"], "INSUFFICIENT_EO_EVIDENCE|NO_DECLARATION")
    expect_equal(
        result.loc[1, "inspection_reasons"],
        ("INSUFFICIENT_EO_EVIDENCE|MEDIUM_MODEL_CONFIDENCE|MEDIUM_DATA_RELIABILITY|MEDIUM_GEOMETRY_COMPLEXITY|NO_DECLARATION"),
    )
    expect_equal(
        result.loc[2, "inspection_reasons"],
        ("INSUFFICIENT_EO_EVIDENCE|LOW_MODEL_CONFIDENCE|LOW_DATA_RELIABILITY|HIGH_GEOMETRY_COMPLEXITY|NO_DECLARATION"),
    )
    expect_equal(
        result.loc[3, "inspection_reasons"],
        ("INSUFFICIENT_EO_EVIDENCE|LOW_DATA_RELIABILITY|HIGH_GEOMETRY_COMPLEXITY|NO_DECLARATION"),
    )
    expect_equal(
        result.loc[4, "inspection_reasons"], ("INSUFFICIENT_EO_EVIDENCE|INVALID_MODEL_CONFIDENCE|MISSING_GEOMETRY_COMPLEXITY")
    )
    expect_true(pd.isna(result.loc[4, "inspection_score"]))


def test_inspection_reasons_handles_invalid_confidence_and_reliability_edges() -> None:
    config = _config()
    row = pd.Series(
        {
            "label_prediction_status": "DIFFERENT",
            "inspection_check_type": "DECLARATION_CONFLICT",
            "inspection_need": "HIGH",
            "prediction_confidence_level": "unknown",
            "data_reliability_score": "not-a-number",
            "geometry_risk": np.nan,
        }
    )

    reasons = _inspection_reasons(row, config)

    expect_equal(
        reasons,
        "DECLARATION_CONFLICT|INVALID_MODEL_CONFIDENCE|MISSING_DATA_RELIABILITY|MISSING_GEOMETRY_COMPLEXITY",
    )


def test_inspection_reasons_marks_medium_bands_without_declaration_conflict() -> None:
    config = _config()
    row = pd.Series(
        {
            "label_prediction_status": "NO_DECLARATION",
            "inspection_check_type": "INSUFFICIENT_EO_EVIDENCE",
            "inspection_need": "MEDIUM",
            "prediction_confidence_level": "medium",
            "data_reliability_score": config.inspection_medium_data_reliability_threshold - 0.01,
            "geometry_risk": config.inspection_medium_geometry_risk_threshold,
        }
    )

    reasons = _inspection_reasons(row, config)

    expect_equal(
        reasons,
        "INSUFFICIENT_EO_EVIDENCE|MEDIUM_MODEL_CONFIDENCE|MEDIUM_DATA_RELIABILITY|MEDIUM_GEOMETRY_COMPLEXITY",
    )


def test_label_aware_policy_reproduces_operational_examples() -> None:
    parcels = pd.DataFrame(
        {
            "prediction_class_reliability_level": ["HIGH"] * 6,
            "prediction_class_reliability_valid": [True] * 6,
            "declared_crop": ["Cotton", "Cotton", "Cotton", "Cotton", "Cotton", "Cotton"],
            "predicted_crop": ["Cotton", "Maize", "Maize", "Cotton", "Maize", "Maize"],
            "prediction_confidence_level": ["HIGH", "HIGH", "LOW", "LOW", "HIGH", "HIGH"],
            "data_reliability_score": [0.9, 0.9, 0.4, 0.4, 0.4, 0.9],
            "geom_shape_complexity_score": [1.0, 1.0, 4.0, 4.0, 4.0, 4.0],
        }
    )

    result = calculate_inspection_metrics(parcels, _config())

    expect_equal(result["label_prediction_status"].tolist(), ["SAME", "DIFFERENT", "DIFFERENT", "SAME", "DIFFERENT", "DIFFERENT"])
    expect_equal(result["inspection_need"].tolist(), ["VERY_LOW", "VERY_HIGH", "MEDIUM", "MEDIUM", "MEDIUM", "HIGH"])
    expect_equal(
        result["inspection_check_type"].tolist(),
        [
            "NONE",
            "DECLARATION_CONFLICT",
            "INSUFFICIENT_EO_EVIDENCE",
            "INSUFFICIENT_EO_EVIDENCE",
            "INSUFFICIENT_EO_EVIDENCE",
            "DECLARATION_CONFLICT",
        ],
    )
    expect_equal(result.loc[1, "inspection_reasons"], "STRONG_DECLARATION_CONFLICT")
    expect_startswith(result.loc[2, "inspection_reasons"], "DECLARATION_PREDICTION_DIFFERENCE|INSUFFICIENT_EO_EVIDENCE")
    expect_startswith(result.loc[5, "inspection_reasons"], "DECLARATION_CONFLICT|HIGH_GEOMETRY_COMPLEXITY")


def test_inspection_geometry_risk_uses_average_percentile_for_ties() -> None:
    parcels = pd.DataFrame(
        {
            "prediction_confidence_level": ["HIGH", "HIGH", "HIGH"],
            "data_reliability_score": [1.0, 1.0, 1.0],
            "geom_shape_complexity_score": [1.0, 1.0, 3.0],
        }
    )

    result = calculate_inspection_metrics(parcels, _config())

    expect_equal(result["geometry_risk"].tolist(), pytest.approx([0.5, 0.5, 1.0]))


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

    expect_true(summary["inspection_scoring_available"])
    expect_equal(summary["inspection_weights"], {"model": 0.60, "data": 0.25, "geometry": 0.15})
    expect_subset({
        "confidence_risk",
        "data_risk",
        "geometry_risk",
        "inspection_score",
        "label_prediction_status",
        "inspection_check_type",
    }, result.columns)


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

    expect_equal(unknown_mask.tolist(), [False, False])
    expect_true(rank_contract["enabled"] is True)
    expect_is_not_none(rank_confidence)
    expect_in("prediction_confidence_level", result.columns)
    expect_true(result["inspection_score"].notna().all())
    expect_true(inspection_summary["inspection_scoring_available"])


def test_predict_step_skips_inspection_scoring_when_inputs_are_absent() -> None:
    step = object.__new__(PredictStep)
    step.config = _config()

    with pytest.warns(RuntimeWarning, match="required columns are missing"):
        result, summary = step._add_inspection_metrics(pd.DataFrame({"prediction_confidence_level": ["HIGH"]}))

    expect_equal(list(result.columns), ["prediction_confidence_level"])
    expect_false(summary["inspection_scoring_available"])
    expect_equal(summary["inspection_missing_columns"], ["data_reliability_score", "geom_shape_complexity_score"])


@pytest.mark.parametrize(
    ("class_level", "rank_level", "expected"),
    [
        ("HIGH", "HIGH", ["VERY_LOW", "LOW", "VERY_HIGH", "HIGH"]),
        ("MEDIUM", "HIGH", ["LOW", "LOW_MEDIUM", "HIGH", "MEDIUM_HIGH"]),
        ("LOW", "HIGH", ["MEDIUM"] * 4),
        ("HIGH", "MEDIUM", ["LOW_MEDIUM", "MEDIUM", "MEDIUM_HIGH", "MEDIUM"]),
        ("MEDIUM", "MEDIUM", ["MEDIUM"] * 4),
        ("LOW", "MEDIUM", ["MEDIUM"] * 4),
        ("HIGH", "LOW", ["MEDIUM"] * 4),
        ("MEDIUM", "LOW", ["MEDIUM"] * 4),
        ("LOW", "LOW", ["MEDIUM"] * 4),
    ],
)
@pytest.mark.parametrize("data_reliability", [0.0, 0.8, 0.800001, 1.0])
def test_decision_table_covers_every_valid_combination(class_level, rank_level, expected, data_reliability):
    """Cover all class/rank/status/geometry combinations and the strict data boundary."""
    frame = pd.DataFrame(
        {
            "label_prediction_status": ["SAME", "SAME", "DIFFERENT", "DIFFERENT"],
            "geometry_risk": [0.499999, 0.5, 0.499999, 0.5],
            "prediction_class_reliability_level": [class_level] * 4,
            "prediction_class_reliability_valid": [True] * 4,
        },
        index=[10, 20, 30, 40],
    )
    confidence = pd.Series(rank_level, index=frame.index)
    data = pd.Series(data_reliability, index=frame.index)

    need, check_type = _apply_label_aware_policy(frame, confidence, data, _config())

    expect_equal(need.tolist(), expected if data_reliability > 0.8 else ["MEDIUM"] * 4)
    expect_true(frame["inspection_class_reliability_applied"].all())
    expect_true(frame["inspection_rule"].str.match(r"R\d{2}").all())
    for priority, check in zip(need, check_type, strict=True):
        expected_check = "INSUFFICIENT_EO_EVIDENCE"
        if priority in {"VERY_LOW", "LOW"}:
            expected_check = "NONE"
        elif priority in {"VERY_HIGH", "HIGH", "MEDIUM_HIGH"}:
            expected_check = "DECLARATION_CONFLICT"
        expect_equal(check, expected_check)


def _reliable_parcels() -> pd.DataFrame:
    """Return comparable rows spanning both sides of the geometry boundary."""
    return pd.DataFrame(
        {
            "declared_crop": ["A", "A", "A", "A"],
            "predicted_crop": ["A", "B", "A", "B"],
            "prediction_confidence_level": ["HIGH"] * 4,
            "prediction_class_reliability_level": ["HIGH"] * 4,
            "prediction_class_reliability_valid": [True] * 4,
            "data_reliability_score": [0.9] * 4,
            "geom_shape_complexity_score": [1.0, 2.0, 3.0, 4.0],
        }
    )


def test_policy_is_independent_of_numeric_score_and_preserves_input():
    """Changing legacy diagnostic weights cannot change inspection priority."""
    parcels = _reliable_parcels()
    original = parcels.copy(deep=True)
    config = _config()
    first = calculate_inspection_metrics(parcels, config)
    config.inspection_model_weight = 0.0
    config.inspection_data_weight = 0.0
    config.inspection_geometry_weight = 1.0
    second = calculate_inspection_metrics(parcels, config)
    pd.testing.assert_frame_equal(parcels, original)
    expect_false(first["inspection_score"].equals(second["inspection_score"]))
    pd.testing.assert_series_equal(first["inspection_need"], second["inspection_need"])
    expect_equal(first["inspection_rule"].tolist(), ["R01", "R12", "R02", "R12"])
    expect_equal(first["inspection_priority_label"].tolist(), ["Very Low", "High", "Low", "High"])


@pytest.mark.parametrize("enabled", [True, False])
def test_frozen_policy_flag_overrides_live_configuration(enabled):
    """Disabling class reliability bypasses it without falsely reporting HIGH reliability."""
    parcels = _reliable_parcels()
    parcels["prediction_class_reliability_level"] = "LOW"
    parcels["inspection_class_reliability_enabled"] = enabled
    config = _config()
    config.class_reliability_enabled = not enabled
    result = calculate_inspection_metrics(parcels, config)
    expect_equal(result["inspection_need"].tolist(), ["MEDIUM"] * 4 if enabled else ["VERY_LOW", "HIGH", "LOW", "HIGH"])
    expect_equal(result["prediction_class_reliability_level"].tolist(), ["LOW"] * 4)
    expect_equal(result["inspection_class_reliability_applied"].tolist(), [enabled] * 4)


@pytest.mark.parametrize("missing", [True, False])
def test_unavailable_class_reliability_is_medium_without_invalidating_rank(missing):
    """Missing support is distinct from measured LOW reliability and from invalid rank evidence."""
    parcels = _reliable_parcels()
    if missing:
        parcels = parcels.drop(columns=["prediction_class_reliability_level", "prediction_class_reliability_valid"])
    else:
        parcels["prediction_class_reliability_valid"] = False
    result = calculate_inspection_metrics(parcels, _config())
    expect_equal(result["inspection_need"].tolist(), ["MEDIUM"] * 4)
    expect_equal(result["inspection_rule"].tolist(), ["CLASS_RELIABILITY_UNAVAILABLE"] * 4)
    expect_equal(result["prediction_confidence_level"].tolist(), ["HIGH"] * 4)
    expect_false(result["inspection_class_reliability_applied"].any())


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("prediction_confidence_level", None),
        ("prediction_confidence_level", "UNKNOWN"),
        ("prediction_rank_confidence_valid", False),
        ("data_reliability_score", np.nan),
        ("data_reliability_score", 1.1),
        ("geom_shape_complexity_score", np.inf),
        ("geom_shape_complexity_score", -1.0),
        ("predicted_crop", None),
    ],
)
def test_missing_or_invalid_required_evidence_is_unscored(column, value):
    """Invalid inputs must not accidentally receive a reassuring priority."""
    parcels = _reliable_parcels()
    parcels[column] = value
    result = calculate_inspection_metrics(parcels, _config())
    expect_equal(result["inspection_need"].tolist(), ["UNKNOWN"] * 4)
    expect_equal(result["inspection_rule"].tolist(), ["REQUIRED_EVIDENCE_UNAVAILABLE"] * 4)


def test_no_declaration_is_medium_and_missing_prediction_is_unknown():
    """A missing declaration cannot become agreement or a declaration conflict."""
    parcels = _reliable_parcels()
    parcels["declared_crop"] = None
    parcels.loc[0, "predicted_crop"] = None
    result = calculate_inspection_metrics(parcels, _config())
    expect_equal(result["inspection_need"].tolist(), ["UNKNOWN", "MEDIUM", "MEDIUM", "MEDIUM"])
    expect_equal(result["label_prediction_status"].tolist(), ["UNAVAILABLE"] + ["NO_DECLARATION"] * 3)


def test_seven_priority_queue_order_and_stable_ties():
    """Queue order follows categories, includes intermediate priorities, and ignores score ties."""
    from ml_classification.step_05_predict.libraries.inspection_priority.diagnostics import _top_priority_parcels

    config = _config()
    config.id_column = "id"
    frame = pd.DataFrame(
        {
            "id": ["low-medium", "medium", "high", "very-high", "medium-high", "medium-second"],
            "inspection_need": ["LOW_MEDIUM", "MEDIUM", "HIGH", "VERY_HIGH", "MEDIUM_HIGH", "MEDIUM"],
            "inspection_check_type": ["INSUFFICIENT_EO_EVIDENCE"] * 6,
            "inspection_score": [100, 0, 50, 0, 90, 100],
        }
    )
    queue = _top_priority_parcels(frame, config)
    expect_equal(queue["id"].tolist(), ["very-high", "high", "medium-high", "medium", "medium-second", "low-medium"])


@pytest.mark.parametrize("enabled", [True, False])
def test_prediction_quality_routes_frozen_class_evidence_only_to_policy(enabled):
    """The complete quality path keeps confidence/review unchanged while policy uses frozen reliability."""
    step = object.__new__(PredictStep)
    step.config = _config()
    step.config.class_reliability_enabled = not enabled
    parcels = pd.DataFrame({
        "declared_crop": ["Cotton", "Cotton"],
        "data_reliability_score": [0.9, 0.9],
        "geom_shape_complexity_score": [1.0, 2.0],
    })
    probabilities = np.array([[0.9, 0.1], [0.1, 0.9]])
    selection = {
        "labels": ["Cotton", "Maize"],
        "selected_models": ["a", "b", "c"],
        "confidence": {
            "enabled": True,
            "method": "rank_consensus_only",
            "version": "3.0",
            "class_reliability_role": "inspection_policy",
            "class_reliability_enabled": enabled,
            "rank": {
                "minimum_models": 3, "high_min_borda": 90.0, "high_max_range": 2.0,
                "medium_min_borda": 75.0, "medium_max_range": 4.0,
            },
            "class_reliability": {"method": "oof_precision", "classes": {
                "Cotton": {"level": "HIGH", "valid": True, "precision": 0.9, "support": 100},
                "Maize": {"level": "LOW", "valid": True, "precision": 0.4, "support": 100},
            }},
        },
    }
    result, _, _, contract, summary = step.calculate_prediction_quality(
        parcels, np.array(["Cotton", "Maize"]), probabilities, selection,
        {name: probabilities for name in selection["selected_models"]},
    )
    expect_equal(result["prediction_confidence_level"].tolist(), ["HIGH", "HIGH"])
    expect_equal(result["prediction_confidence_valid"].tolist(), [True, True])
    expect_equal(result["prediction_needs_review"].tolist(), [False, False])
    expect_equal(result["inspection_need"].tolist(), ["LOW", "MEDIUM" if enabled else "HIGH"])
    expect_equal(contract["class_reliability_enabled"], enabled)
    expect_equal(summary["class_reliability_applied_to_inspection_policy"], enabled)


def test_report_data_bands_match_policy_boundaries():
    """Report bands must keep exactly 0.8 in MEDIUM and exactly 0.6 out of LOW."""
    from ml_classification.step_05_predict.libraries.inspection_priority.diagnostics import _evidence_quality_table

    frame = pd.DataFrame({
        "data_reliability_score": [0.59, 0.6, 0.8, 0.800001, np.nan, 1.1],
        "geometry_risk": [0.25] * 6,
        "inspection_need": ["MEDIUM"] * 6,
        "inspection_check_type": ["INSUFFICIENT_EO_EVIDENCE"] * 6,
        "inspection_score": [50.0] * 6,
    })
    table = _evidence_quality_table(frame, _config()).set_index("data_reliability_level")
    expect_equal(table["parcels"].to_dict(), {"LOW": 1, "MEDIUM": 2, "HIGH": 1})
