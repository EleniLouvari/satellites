"""Tests for prediction-confidence report diagnostics."""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import matplotlib.image as mpimg
import pandas as pd
import pytest

from ml_classification.shared.reports.confidence_diagnostics import (
    export_prediction_confidence_diagnostics,
    prepare_confidence_diagnostics,
)
from ml_classification.step_05_predict.libraries import report as report_module
from ml_classification.step_05_predict.libraries.inspection_priority.plots import save_inspection_relationship_plot
from ml_classification.step_05_predict.libraries.inspection_priority.scoring import calculate_inspection_metrics
from ml_classification.step_05_predict.libraries.prediction_quality import resolve_rank_confidence_contract
from ml_classification.step_05_predict.libraries.report import write_predict_report
from tests.utils import expect_equal, expect_in, expect_is_not_none, expect_true


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

    expect_equal(len(diagnostics), 7)
    expect_equal(diagnostics["diagnostic_group"].astype(str).value_counts().to_dict(), {
        "Borda agrees + correct": 3,
        "Borda agrees + incorrect": 2,
        "Borda disagrees + correct": 1,
        "Borda disagrees + incorrect": 1,
    })
    numeric_code = diagnostics.loc[7]
    expect_true(numeric_code["prediction_correct"])
    expect_true(numeric_code["borda_agrees_prediction"])


def test_build_prediction_summary_includes_contract_and_artifact_metadata() -> None:
    config = SimpleNamespace(
        target_column="label",
        prediction_confidence_level_column="prediction_confidence_level",
    )
    predictions = pd.DataFrame(
        {
            "label": ["A", None],
            "prediction_confidence_level": ["HIGH", "LOW"],
            "prediction_class_reliability_applied": [True, True],
            "inspection_need": ["HIGH", "LOW"],
            "inspection_check_type": ["DECLARATION_CONFLICT", "INSUFFICIENT_EO_EVIDENCE"],
            "label_prediction_status": ["agree", "missing_label"],
            "inspection_score": [80.0, 20.0],
        }
    )
    selection = {"selection_type": "soft_voting", "selected_models": ["m1"], "confidence": {"class_reliability_enabled": True}}
    confidence_artifacts = SimpleNamespace(summary={"validation_rows_with_reference": 1})

    summary, class_reliability_applied = report_module._build_prediction_summary(
        config=config,
        final_df=predictions,
        selection=selection,
        confidence_artifacts=confidence_artifacts,
    )

    expect_true(class_reliability_applied)
    expect_equal(summary["rows_total"], 2)
    expect_equal(summary["rows_unknown_original"], 1)
    expect_equal(summary["confidence_method"], "within-model ranks")
    expect_equal(summary["class_reliability_enabled"], True)
    expect_equal(summary["final_confidence_rule"], "rank_plus_class_reliability")
    expect_equal(summary["validation_rows_with_reference"], 1)


def test_resolve_rank_confidence_contract_returns_disabled_summary_contract() -> None:
    contract = resolve_rank_confidence_contract(
        config=SimpleNamespace(
            rank_confidence_enabled=False,
            class_reliability_enabled=False,
            rank_confidence_minimum_models=3,
        ),
        selection={"confidence": {"enabled": False}},
    )
    expect_equal(contract["enabled"], False)
    expect_equal(contract["source"], "selection_summary")
    expect_equal(contract["class_reliability_enabled"], False)


def test_resolve_rank_confidence_contract_rejects_invalid_frozen_type() -> None:
    with pytest.raises(TypeError, match="must be an object"):
        resolve_rank_confidence_contract(
            config=SimpleNamespace(
                rank_confidence_enabled=False,
                class_reliability_enabled=False,
                rank_confidence_minimum_models=2,
            ),
            selection={"confidence": "corrupt"},
        )


def test_build_output_links_appends_confidence_and_inspection_exports(tmp_path) -> None:
    config = SimpleNamespace(predict_dir=tmp_path)

    links = report_module._build_output_links(
        config=config,
        confidence_artifacts=SimpleNamespace(summary={}),
        inspection_artifacts=SimpleNamespace(tables={}, images={}),
    )

    labels = {item["label"] for item in links}
    expect_in("Final Predictions Joblib", labels)
    expect_in("Confidence Metric Median Summary", labels)
    expect_in("Inspection Need Summary", labels)


def test_build_prediction_diagnostics_section_returns_unavailable_tabs_when_artifacts_missing() -> None:
    section = report_module._build_prediction_diagnostics_section(
        config=SimpleNamespace(),
        confidence_artifacts=None,
        inspection_artifacts=None,
        data_reliability_column="data_reliability_score",
        geometry_complexity_column="geom_shape_complexity_score",
    )

    expect_equal(section["title"], "Prediction Diagnostics")
    tabs = section["tabs"]
    expect_equal(len(tabs), 2)
    expect_equal(tabs[0]["label"], "Confidence Levels")
    expect_equal(tabs[1]["label"], "Inspection Priority")
    expect_equal(tabs[0]["sections"][0]["title"], "Confidence Diagnostics Unavailable")
    expect_equal(tabs[1]["sections"][0]["title"], "Operational Inspection Priority Unavailable")


def test_build_confidence_sections_formats_median_summary_values() -> None:
    confidence_artifacts = SimpleNamespace(
        images={},
        embeds={},
        tables={
            "confidence_metric_median_summary": pd.DataFrame(
                [
                    {
                        "Metric": "rank_range",
                        "Correct": 1.23456,
                        "Needs check": 2.34567,
                        "Needs check minus Correct": -1.11111,
                    }
                ]
            )
        },
    )

    sections = report_module._build_confidence_sections(confidence_artifacts)

    median_section = next(section for section in sections if section["title"] == "Confidence Metric Median Comparison")
    table = median_section["table"]
    expect_equal(table.iloc[0]["Correct"], "1.235")
    expect_equal(table.iloc[0]["Needs check"], "2.346")
    expect_equal(table.iloc[0]["Needs check minus Correct"], "-1.111")


def test_prediction_report_exports_confidence_graphs_tables_and_sankey(request):
    output_dir = Path(__file__).parent / f"_prediction_report_{uuid4().hex}"
    request.addfinalizer(lambda: shutil.rmtree(output_dir, ignore_errors=True))
    output_dir.mkdir(parents=True)
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
        random_state=42,
    )
    predictions = _prediction_rows().assign(
        data_reliability_score=[0.95, 0.82, 0.74, 0.55, 0.91, 0.68, 0.42, 0.87],
        geom_shape_complexity_score=[1.0, 1.4, 1.8, 2.5, 1.2, 2.1, 3.2, 1.1],
    )
    # Include one reliable disagreement so the declaration-conflict matrix is exercised.
    predictions.loc[1, "prediction_confidence_level"] = "HIGH"
    predictions = calculate_inspection_metrics(predictions, config)

    artifacts = export_prediction_confidence_diagnostics(
        predictions,
        target_column="label",
        prediction_column="label_prediction",
        output_dir=output_dir,
    )

    expect_is_not_none(artifacts)
    expect_equal(artifacts.summary["validation_rows_with_reference"], 7)
    expect_equal(artifacts.summary["need_to_check_rows"], 2)
    expect_equal(set(artifacts.images), {
        "overview",
        "class_rate",
        "reason_heatmap",
        "confusion",
        "distribution",
        "joint_behavior",
        "class_confidence_risk",
    })
    expect_equal(set(artifacts.embeds), {"reason_sankey"})
    expect_true(all(path.exists() for path in [*artifacts.images.values(), *artifacts.embeds.values()]))
    expect_true((output_dir / "data" / "need_to_check_by_predicted_class.csv").exists())
    expect_true((output_dir / "data" / "confidence_metric_median_summary.csv").exists())
    expect_equal(artifacts.tables["confidence_metric_median_summary"].shape, (6, 4))

    expect_true(save_inspection_relationship_plot(
        predictions,
        output_dir / "plots" / "inspection_risk_relationships.png",
        config,
    ))
    write_predict_report(
        config,
        predictions,
        {"selection_type": "soft_voting", "selected_models": ["model_a", "model_b"]},
    )

    report = (output_dir / "report.html").read_text(encoding="utf-8")
    expect_in("Borda Consensus and Prediction Correctness", report)
    expect_in("Confidence Metric Median Comparison", report)
    expect_in("Distribution Comparison", report)
    expect_in("vertical axis is the share of that group at or below the selected value", report)
    expect_in("These curves compare group distributions", report)
    expect_in("More favorable", report)
    expect_in("Joint Probability-Consensus Behavior", report)
    expect_in("Class-Level Confidence Risk", report)
    expect_in("Need-to-Check Rate by Predicted Class", report)
    expect_in("Confidence Components of Incorrect Consensus Predictions", report)
    expect_in("Need-to-Check Class Confusions", report)
    expect_in("need_to_check_confidence_flow.html", report)
    expect_in("Unlabeled fill rows are excluded", report)
    expect_in("Inspection Priority: Confidence, Data Quality, and Geometry", report)
    expect_in("inspection_risk_relationships.png", report)
    expect_in("inspection_reasons", report)
    expect_in("DECLARATION_CONFLICT", report)
    expect_in("INSUFFICIENT_EO_EVIDENCE", report)
    expect_in("role='tablist'", report)
    expect_in(">Confidence Levels</button>", report)
    expect_in(">Inspection Priority</button>", report)
    expect_in("Operational Check Types", report)
    expect_in("Inspection Score by Confidence and Data Reliability", report)
    expect_in("Data Reliability and Geometry Evidence", report)
    expect_in("Operational Inspection Priority by Predicted Crop", report)
    expect_in("Confidence Review versus Operational Inspection Need", report)
    expect_in("Top-Priority Parcel Inspection Queue", report)

    expected_inspection_tables = {
        "inspection_need_summary.csv",
        "inspection_check_type_summary.csv",
        "inspection_priority_by_predicted_class.csv",
        "inspection_declaration_conflict_matrix.csv",
        "inspection_evidence_quality_matrix.csv",
        "inspection_confidence_review_comparison.csv",
        "inspection_need_by_declaration_status.csv",
        "inspection_score_by_confidence_data.csv",
        "inspection_top_priority_parcels.csv",
    }
    expect_true(all((output_dir / "data" / filename).exists() for filename in expected_inspection_tables))
    expected_inspection_plots = {
        "inspection_risk_relationships.png",
        "inspection_check_type_distribution.png",
        "inspection_priority_by_predicted_class.png",
        "inspection_need_by_declaration_status.png",
        "inspection_score_confidence_data_heatmap.png",
        "inspection_declaration_conflicts.png",
        "inspection_confidence_review_comparison.png",
        "inspection_evidence_quality_heatmap.png",
    }
    expect_true(all((output_dir / "plots" / filename).exists() for filename in expected_inspection_plots))
    relationship_image = mpimg.imread(output_dir / "plots" / "inspection_risk_relationships.png")
    expect_true(relationship_image.shape[1] >= 2_000)
    expect_true(relationship_image.shape[0] >= 1_000)
