"""Tests for prediction-confidence report diagnostics."""

from __future__ import annotations

from pathlib import Path
import shutil
from types import SimpleNamespace
from uuid import uuid4

import pandas as pd

from ml_classification.ml_classification_pipeline.reporting.confidence_diagnostics import (
    export_prediction_confidence_diagnostics,
    prepare_confidence_diagnostics,
)
from ml_classification.ml_classification_pipeline.reporting.reports import write_predict_report


def _prediction_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "parcel_code": [str(index) for index in range(1, 9)],
            "label": ["A", "A", "B", "B", "C", "C", None, 1.0],
            "label_prediction": ["A", "B", "B", "A", "C", "A", "B", "1"],
            "label_filled": ["A", "A", "B", "B", "C", "C", "B", "1"],
            "prediction_borda_winner": ["A", "B", "A", "B", "C", "A", "B", 1],
            "prediction_max_probability": [0.91, 0.76, 0.72, 0.61, 0.88, 0.67, 0.55, 0.93],
            "prediction_borda_margin": [18.0, 12.0, 4.0, 3.0, 16.0, 6.0, 2.0, 20.0],
            "prediction_mean_borda": [8.0, 7.0, 6.0, 5.0, 8.5, 6.5, 4.0, 9.0],
            "prediction_rank_range": [1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 8.0, 1.0],
            "prediction_top1_agreement": [1.0, 0.8, 0.6, 0.5, 1.0, 0.7, 0.4, 1.0],
            "prediction_needs_review": [False, False, False, True, False, False, True, False],
            "prediction_confidence_level": ["HIGH", "MEDIUM", "MEDIUM", "LOW", "HIGH", "MEDIUM", "LOW", "HIGH"],
            "prediction_confidence_valid": [True] * 8,
            "prediction_confidence_reason": ["rank_confidence_high"] * 8,
            "prediction_class_oof_precision": [0.93, 0.74, 0.82, 0.74, 0.89, 0.93, 0.82, 0.96],
            "prediction_class_reliability_level": ["HIGH", "MEDIUM", "HIGH", "MEDIUM", "HIGH", "HIGH", "HIGH", "HIGH"],
            "prediction_rank_confidence_level": ["HIGH", "HIGH", "MEDIUM", "LOW", "HIGH", "MEDIUM", "LOW", "HIGH"],
        }
    )


def test_prepare_confidence_diagnostics_uses_known_reference_rows_and_normalizes_codes():
    diagnostics = prepare_confidence_diagnostics(
        _prediction_rows(),
        target_column="label",
        prediction_column="label_prediction",
    )

    assert len(diagnostics) == 7
    assert diagnostics["diagnostic_group"].astype(str).value_counts().to_dict() == {
        "Borda agrees + correct": 3,
        "Borda agrees + incorrect": 2,
        "Borda disagrees + correct": 1,
        "Borda disagrees + incorrect": 1,
    }
    numeric_code = diagnostics.loc[7]
    assert numeric_code["prediction_correct"]
    assert numeric_code["borda_agrees_prediction"]


def test_prediction_report_exports_confidence_graphs_tables_and_sankey(request):
    output_dir = Path(__file__).parent / f"_prediction_report_{uuid4().hex}"
    request.addfinalizer(lambda: shutil.rmtree(output_dir, ignore_errors=True))
    output_dir.mkdir(parents=True)
    predictions = _prediction_rows()

    artifacts = export_prediction_confidence_diagnostics(
        predictions,
        target_column="label",
        prediction_column="label_prediction",
        output_dir=output_dir,
    )

    assert artifacts is not None
    assert artifacts.summary["validation_rows_with_reference"] == 7
    assert artifacts.summary["need_to_check_rows"] == 2
    assert set(artifacts.images) == {
        "overview",
        "class_rate",
        "reason_heatmap",
        "confusion",
        "distribution",
        "joint_behavior",
        "class_confidence_risk",
    }
    assert set(artifacts.embeds) == {"reason_sankey"}
    assert all(path.exists() for path in [*artifacts.images.values(), *artifacts.embeds.values()])
    assert (output_dir / "data" / "need_to_check_by_predicted_class.csv").exists()
    assert (output_dir / "data" / "confidence_metric_median_summary.csv").exists()
    assert artifacts.tables["confidence_metric_median_summary"].shape == (6, 4)

    config = SimpleNamespace(
        id_column="parcel_code",
        target_column="label",
        prediction_column="label_prediction",
        prediction_filled_column="label_filled",
        prediction_confidence_column="prediction_max_probability",
        prediction_confidence_level_column="prediction_confidence_level",
        prediction_review_column="prediction_needs_review",
        probability_prefix="probability",
        predict_dir=output_dir,
    )
    write_predict_report(
        config,
        predictions,
        {"selection_type": "soft_voting", "selected_models": ["model_a", "model_b"]},
    )

    report = (output_dir / "report.html").read_text(encoding="utf-8")
    assert "Borda Consensus and Prediction Correctness" in report
    assert "Confidence Metric Median Comparison" in report
    assert "Distribution Comparison" in report
    assert "vertical axis is the share of that group at or below the selected value" in report
    assert "These curves compare group distributions" in report
    assert "More favorable" in report
    assert "Joint Probability-Consensus Behavior" in report
    assert "Class-Level Confidence Risk" in report
    assert "Need-to-Check Rate by Predicted Class" in report
    assert "Confidence Components of Incorrect Consensus Predictions" in report
    assert "Need-to-Check Class Confusions" in report
    assert "need_to_check_confidence_flow.html" in report
    assert "Unlabeled fill rows are excluded" in report
