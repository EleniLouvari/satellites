"""Assemble the predict step HTML report from its persisted and computed results."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ml_classification.shared.reports.confidence_diagnostics import (
    export_prediction_confidence_diagnostics,
    prepare_class_risk_display_table,
)
from ml_classification.shared.reports.report_html import write_html_report
from ml_classification.step_05_predict.libraries.inspection_priority.diagnostics import (
    export_operational_inspection_diagnostics,
)


def write_predict_report(config, final_df: pd.DataFrame, selection: dict[str, Any]) -> None:
    """Write the step-5 prediction report with final fill outputs."""
    data_reliability_column = getattr(config, "inspection_data_reliability_column", "data_reliability_score")
    geometry_complexity_column = getattr(config, "inspection_geometry_complexity_column", "geom_shape_complexity_score")
    # Prepare preview columns and optional probability columns for display.
    preview_columns = [config.id_column, config.target_column, config.prediction_column, config.prediction_filled_column]
    confidence_columns = [
        config.prediction_confidence_column,
        config.prediction_confidence_level_column,
        "prediction_confidence_valid",
        "prediction_confidence_reason",
        "prediction_confidence_source",
        "prediction_class_reliability_applied",
        "prediction_mean_borda",
        "prediction_rank_range",
        "prediction_mean_rank",
        "prediction_median_rank",
        "prediction_borda_winner",
        "prediction_borda_winner_tied",
        "prediction_rank_agrees_with_final",
        "prediction_rank_confidence_level",
        "prediction_class_oof_precision",
        "prediction_class_oof_support",
        "prediction_class_reliability_level",
        "prediction_class_reliability_valid",
        "prediction_class_reliability_reason",
        "prediction_borda_margin",
        "prediction_top1_agreement",
        "prediction_top2_agreement",
        "prediction_top3_agreement",
        "prediction_runner_up_class",
        "prediction_runner_up_borda",
        "prediction_rank_models_used",
        "prediction_rank_confidence_valid",
        "prediction_rank_confidence_reason",
        data_reliability_column,
        geometry_complexity_column,
        "confidence_risk",
        "data_risk",
        "geometry_risk",
        "inspection_score",
        "inspection_need_base",
        "inspection_need",
        "inspection_check_type",
        "label_prediction_status",
        "inspection_reasons",
        config.prediction_review_column,
    ]
    # Optional confidence and inspection fields vary with the artifacts produced by the run.
    preview_columns.extend(column for column in confidence_columns if column in final_df.columns)
    probability_columns = [column for column in final_df.columns if column.startswith(f"{config.probability_prefix}_")]
    confidence_artifacts = export_prediction_confidence_diagnostics(
        final_df,
        config.target_column,
        config.prediction_column,
        config.predict_dir,
        probability_threshold=getattr(config, "prediction_confidence_threshold", 0.60),
    )
    # Operational inspection uses all eligible parcels; correctness diagnostics need known reference labels.
    inspection_artifacts = export_operational_inspection_diagnostics(final_df, config, config.predict_dir)
    prediction_summary, class_reliability_applied = _build_prediction_summary(
        config=config,
        final_df=final_df,
        selection=selection,
        confidence_artifacts=confidence_artifacts,
    )

    sections: list[dict[str, Any]] = [
        {"title": "Prediction Summary", "kv": prediction_summary},
        {"title": "Prediction Preview", "table": final_df[preview_columns + probability_columns].head(50)},
    ]
    confidence_formula = _build_confidence_formula(class_reliability_applied)
    sections.append(
        {
            "title": "How Final Confidence Is Calculated",
            "text": (
                "Rank confidence is always calculated. Class reliability is also always calculated and reported, "
                "but it only influences the final prediction confidence when class_reliability_enabled is true."
            ),
            "table": confidence_formula,
        }
    )
    sections.append(
        _build_prediction_diagnostics_section(
            config=config,
            confidence_artifacts=confidence_artifacts,
            inspection_artifacts=inspection_artifacts,
            data_reliability_column=data_reliability_column,
            geometry_complexity_column=geometry_complexity_column,
        )
    )
    sections.append(
        {
            "title": "Saved Outputs",
            "links": _build_output_links(
                config=config, confidence_artifacts=confidence_artifacts, inspection_artifacts=inspection_artifacts
            ),
        }
    )

    write_html_report(
        config.predict_dir / "report.html",
        "Step 5 Report: Final Prediction Fill",
        "Predictions on the full dataset, including rows that originally had unknown labels.",
        sections=sections,
    )


def _build_prediction_diagnostics_section(
    config,
    confidence_artifacts,
    inspection_artifacts,
    data_reliability_column: str,
    geometry_complexity_column: str,
) -> dict[str, Any]:
    """Build the combined diagnostics section with confidence and inspection tabs."""
    # Keep model-confidence auditing and operational inspection policy in separate report tabs.
    confidence_sections = _build_confidence_sections(confidence_artifacts)
    inspection_sections = _build_inspection_sections(
        config=config,
        inspection_artifacts=inspection_artifacts,
        data_reliability_column=data_reliability_column,
        geometry_complexity_column=geometry_complexity_column,
    )
    return {
        "title": "Prediction Diagnostics",
        "text": (
            "Use Confidence Levels to audit model agreement and prediction errors. Use Inspection Priority to "
            "plan parcel checks using confidence, EO data reliability, geometry, and declaration agreement."
        ),
        "tabs": [
            {"label": "Confidence Levels", "sections": confidence_sections},
            {"label": "Inspection Priority", "sections": inspection_sections},
        ],
    }


def _build_inspection_sections(
    config,
    inspection_artifacts,
    data_reliability_column: str,
    geometry_complexity_column: str,
) -> list[dict[str, Any]]:
    """Build inspection-priority diagnostic sections."""
    if inspection_artifacts is None:
        return [
            {
                "title": "Operational Inspection Priority Unavailable",
                "text": "The required confidence, data-reliability, geometry-risk, or inspection fields were absent.",
            }
        ]

    inspection_formula = pd.DataFrame(
        [
            {
                "Component": "Model risk",
                "Definition": "HIGH=0, MEDIUM=0.5, LOW=1",
                "Weight": getattr(config, "inspection_model_weight", 0.60),
            },
            {
                "Component": "Data risk",
                "Definition": f"1 - {data_reliability_column}",
                "Weight": getattr(config, "inspection_data_weight", 0.25),
            },
            {
                "Component": "Geometry risk",
                "Definition": f"Percentile rank of {geometry_complexity_column}",
                "Weight": getattr(config, "inspection_geometry_weight", 0.15),
            },
        ]
    )
    sections: list[dict[str, Any]] = []
    relationship_section: dict[str, Any] = {
        "title": "Inspection Priority: Confidence, Data Quality, and Geometry",
        "text": (
            "Inspection score combines three independent uncertainty signals on a 0-100 scale. The scatter "
            "shows data reliability against the score, color identifies prediction confidence, point size "
            "represents relative geometry risk, and marker shape distinguishes declaration agreement. The "
            "numeric score remains label-independent; declaration agreement only refines inspection need and "
            "check type. Geometry interior-area ratio is not added separately because it is already incorporated "
            "in geom_shape_complexity_score."
        ),
        "table": inspection_formula,
    }
    relationship_path = inspection_artifacts.images.get("relationship")
    if relationship_path is not None:
        relationship_section["images"] = [{"title": "Inspection Score Relationships", "path": relationship_path}]
    sections.append(relationship_section)

    confidence_data_section: dict[str, Any] = {
        "title": "Inspection Score by Confidence and Data Reliability",
        "text": (
            "This enlarged matrix preserves the previous confidence-by-reliability view. Each cell is the median "
            "label-independent inspection score for parcels in that combination; geometry risk still contributes "
            "to every parcel score and therefore to the displayed median."
        ),
        "table": inspection_artifacts.tables["inspection_score_by_confidence_data"],
    }
    confidence_data_path = inspection_artifacts.images.get("confidence_data_score")
    if confidence_data_path is not None:
        confidence_data_section["images"] = [
            {"title": "Median Inspection Score by Confidence and Reliability", "path": confidence_data_path}
        ]
    sections.append(confidence_data_section)

    check_type_section: dict[str, Any] = {
        "title": "Operational Check Types",
        "text": (
            "DECLARATION_CONFLICT identifies reliable EO evidence supporting a crop different from the farmer "
            "declaration. INSUFFICIENT_EO_EVIDENCE identifies parcels where confidence, observation coverage, "
            "or geometry does not support a reliable verification. These types are mutually exclusive."
        ),
        "table": inspection_artifacts.tables["inspection_check_type_summary"],
    }
    check_type_path = inspection_artifacts.images.get("check_types")
    if check_type_path is not None:
        check_type_section["images"] = [{"title": "Operational Check Types", "path": check_type_path}]
    sections.append(check_type_section)

    declaration_need_section: dict[str, Any] = {
        "title": "Final Inspection Need by Declaration Agreement",
        "text": (
            "This matrix shows how agreement with the farmer declaration changes the operational priority. "
            "The numeric inspection score is unchanged; only the final need category receives this transparent "
            "policy overlay."
        ),
        "table": inspection_artifacts.tables["inspection_need_summary"],
    }
    declaration_need_path = inspection_artifacts.images.get("declaration_need")
    if declaration_need_path is not None:
        declaration_need_section["images"] = [{"title": "Inspection Need by Declaration Status", "path": declaration_need_path}]
    sections.append(declaration_need_section)

    evidence_section: dict[str, Any] = {
        "title": "Data Reliability and Geometry Evidence",
        "text": (
            "The heatmap reports the operational inspection rate for each data-reliability and geometry-quality "
            "combination. Cell support in the table should be considered before interpreting extreme rates."
        ),
        "table": inspection_artifacts.tables["inspection_evidence_quality_matrix"],
    }
    evidence_path = inspection_artifacts.images.get("evidence_quality")
    if evidence_path is not None:
        evidence_section["images"] = [{"title": "Inspection Rate by EO Evidence Quality", "path": evidence_path}]
    sections.append(evidence_section)

    class_section: dict[str, Any] = {
        "title": "Operational Inspection Priority by Predicted Crop",
        "text": (
            "Rates use every parcel predicted as each crop. Counts distinguish high-priority cases, reliable "
            "declaration conflicts, and parcels where EO evidence is insufficient. Small crop groups should be "
            "interpreted together with their parcel denominators."
        ),
        "table": inspection_artifacts.tables["inspection_priority_by_predicted_class"],
    }
    class_path = inspection_artifacts.images.get("predicted_class")
    if class_path is not None:
        class_section["images"] = [{"title": "Operational Inspection Rate by Predicted Crop", "path": class_path}]
    sections.append(class_section)

    conflict_section: dict[str, Any] = {
        "title": "Reliable Declaration Conflicts",
        "text": (
            "Only DECLARATION_CONFLICT parcels are included. Rows are farmer-declared crops and columns are "
            "model-predicted crops, allowing systematic, evidence-supported substitutions to be identified."
        ),
        "table": inspection_artifacts.tables["inspection_declaration_conflict_matrix"],
    }
    conflict_path = inspection_artifacts.images.get("conflict_matrix")
    if conflict_path is not None:
        conflict_section["images"] = [{"title": "Farmer Declaration versus Predicted Crop", "path": conflict_path}]
    sections.append(conflict_section)

    comparison_section: dict[str, Any] = {
        "title": "Confidence Review versus Operational Inspection Need",
        "text": (
            "This cross-check shows where the confidence-only review flag and the composite operational policy "
            "agree or differ. Differences are expected because operational priority also uses data reliability, "
            "geometry, and farmer-declaration agreement."
        ),
        "table": inspection_artifacts.tables["inspection_confidence_review_comparison"],
    }
    comparison_path = inspection_artifacts.images.get("review_comparison")
    if comparison_path is not None:
        comparison_section["images"] = [{"title": "Confidence Review versus Inspection Need", "path": comparison_path}]
    sections.append(comparison_section)

    sections.append(
        {
            "title": "Top-Priority Parcel Inspection Queue",
            "text": (
                "The queue is ordered first by final inspection need and then by inspection score. It preserves "
                "the declared and predicted crops, all evidence components, check type, and human-readable reason "
                "codes required for parcel-level follow-up."
            ),
            "table": inspection_artifacts.tables["inspection_top_priority_parcels"],
        }
    )
    return sections


def _build_confidence_sections(confidence_artifacts) -> list[dict[str, Any]]:
    """Build confidence-level diagnostic sections."""
    if confidence_artifacts is None:
        return [
            {
                "title": "Confidence Diagnostics Unavailable",
                "text": "Known declarations, predictions, or Borda diagnostics were unavailable for this run.",
            }
        ]

    confidence_sections: list[dict[str, Any]] = []
    overview_path = confidence_artifacts.images.get("overview")
    if overview_path is not None:
        confidence_sections.append(
            {
                "title": "Borda Consensus and Prediction Correctness",
                "text": (
                    "Correctness is evaluated only for parcels with a known original reference label. "
                    "The four groups separate whether the Borda winner supports the final prediction from "
                    "whether that prediction matches the reference label. Unlabeled fill rows are excluded."
                ),
                "images": [{"title": "Borda Consensus and Prediction Correctness", "path": overview_path}],
            }
        )

    # Format a display copy so numeric source tables remain usable for subsequent analysis.
    median_summary = confidence_artifacts.tables["confidence_metric_median_summary"].copy()
    if not median_summary.empty:
        numeric_columns = ["Correct", "Needs check", "Needs check minus Correct"]
        for column in numeric_columns:
            median_summary[column] = median_summary[column].map(lambda value: f"{value:.3f}" if pd.notna(value) else "n/a")
        confidence_sections.append(
            {
                "title": "Confidence Metric Median Comparison",
                "text": (
                    "Medians compare correct predictions with the needs-check cohort after restricting both "
                    "groups to rows where Borda agrees with the final prediction. The final column is "
                    "Needs check minus Correct; warm cells are positive differences and cool cells are "
                    "negative differences. For rank range, lower values indicate tighter model agreement."
                ),
                "table": median_summary,
                "numeric_cell_styles": {"Needs check minus Correct": {"positive": "danger", "negative": "success"}},
            }
        )

    distribution_path = confidence_artifacts.images.get("distribution")
    if distribution_path is not None:
        distribution_guide = pd.DataFrame(
            [
                {
                    "Metric": "Maximum predicted probability",
                    "More favorable": "Higher",
                    "Meaning": "Probability assigned to the final predicted class.",
                },
                {
                    "Metric": "Borda winner margin",
                    "More favorable": "Higher",
                    "Meaning": "Separation between the Borda winner and runner-up.",
                },
                {
                    "Metric": "Mean Borda score",
                    "More favorable": "Higher",
                    "Meaning": "Overall ensemble support for the predicted class.",
                },
                {
                    "Metric": "Rank range",
                    "More favorable": "Lower",
                    "Meaning": "Smaller values mean models ranked the class more consistently.",
                },
                {
                    "Metric": "Top-1 model agreement",
                    "More favorable": "Higher",
                    "Meaning": "Share of models selecting the same class as their first choice.",
                },
                {
                    "Metric": "OOF precision of predicted class",
                    "More favorable": "Higher",
                    "Meaning": "Historical precision of that class in out-of-fold validation.",
                },
            ]
        )
        confidence_sections.append(
            {
                "title": "Distribution Comparison",
                "text": (
                    "This comparison uses known-label parcels where the Borda winner agrees with the final "
                    "prediction. Blue represents correct predictions and orange represents incorrect "
                    "predictions marked Needs check. In every panel, the horizontal axis is the confidence "
                    "metric and the vertical axis is the share of that group at or below the selected value. "
                    "For example, reading upward from an x value shows what percentage of Correct and Needs "
                    "check parcels have a metric no greater than that value. For metrics where higher is more "
                    "favorable, an orange curve that rises earlier or lies above the blue curve indicates that "
                    "Needs check cases tend to have lower values. Rank range is reversed: lower is more "
                    "favorable, so an earlier blue curve indicates tighter model agreement among correct "
                    "predictions. A large gap between curves means the metric separates the groups well; "
                    "substantial overlap means the metric is weak on its own. These curves compare group "
                    "distributions and should not be read as the probability that an individual prediction is "
                    "wrong."
                ),
                "images": [{"title": "Confidence Distribution Comparison", "path": distribution_path}],
                "table": distribution_guide,
            }
        )

    joint_path = confidence_artifacts.images.get("joint_behavior")
    if joint_path is not None:
        confidence_sections.append(
            {
                "title": "Joint Probability-Consensus Behavior",
                "text": (
                    "The left panel estimates empirical needs-check risk within sufficiently populated "
                    "probability-Borda hexagons; the right panel shows the supporting parcel density. Read the "
                    "panels together so rare combinations are not mistaken for dominant failure modes."
                ),
                "images": [{"title": "Joint Probability-Consensus Behavior", "path": joint_path}],
            }
        )

    class_confidence_path = confidence_artifacts.images.get("class_confidence_risk")
    if class_confidence_path is not None:
        confidence_sections.append(
            {
                "title": "Class-Level Confidence Risk",
                "text": (
                    "Each point is a predicted class: horizontal position is median maximum probability, "
                    "vertical position is observed needs-check rate, point size is parcel support, and color "
                    "is median out-of-fold class precision. Labels prioritize classes contributing the "
                    "largest high-confidence error burden."
                ),
                "images": [{"title": "Class-Level Confidence Risk", "path": class_confidence_path}],
            }
        )

    class_rate_path = confidence_artifacts.images.get("class_rate")
    if class_rate_path is not None:
        confidence_sections.append(
            {
                "title": "Class Reliability by Predicted Class",
                "text": (
                    "This view shows the class-reliability signal used by the final confidence calculation. "
                    "Each row groups parcels by predicted class and reports the rate that needs checking, the "
                    "median out-of-fold precision for that class, and the support count. When "
                    "class_reliability_enabled is false, the same class-reliability table is still shown as a "
                    "diagnostic, but it does not lower the final confidence. This is the same view previously "
                    "described as Need-to-Check Rate by Predicted Class."
                ),
                "images": [{"title": "Class Reliability by Predicted Class", "path": class_rate_path}],
                "table": prepare_class_risk_display_table(confidence_artifacts.tables["need_to_check_by_predicted_class"]),
            }
        )

    reason_path = confidence_artifacts.images.get("reason_heatmap")
    reason_embed = confidence_artifacts.embeds.get("reason_sankey")
    if reason_path is not None or reason_embed is not None:
        reason_section: dict[str, Any] = {
            "title": "Confidence Components of Incorrect Consensus Predictions",
            "text": (
                "For need-to-check parcels, the heatmap crosses predicted-class reliability with rank/ensemble "
                "confidence. The interactive flow then follows predicted class through class reliability, rank "
                "confidence, and final confidence. Large flows reveal combinations that repeatedly produce "
                "confident errors rather than isolated cases."
            ),
        }
        if reason_path is not None:
            reason_section["images"] = [{"title": "Class Reliability versus Rank Confidence", "path": reason_path}]
        if reason_embed is not None:
            reason_section["embeds"] = [{"title": "Need-to-Check Confidence Flow", "path": reason_embed}]
        confidence_sections.append(reason_section)

    confusion_path = confidence_artifacts.images.get("confusion")
    if confusion_path is not None:
        confidence_sections.append(
            {
                "title": "Need-to-Check Class Confusions",
                "text": (
                    "The matrix focuses on the twelve labels most often involved in incorrect Borda-consensus "
                    "predictions. Color is normalized within each displayed true class while cell annotations "
                    "show exact parcel counts, making systematic substitutions visible without losing support."
                ),
                "images": [{"title": "True versus Predicted Class Confusions", "path": confusion_path}],
            }
        )
    return confidence_sections


def _build_prediction_summary(
    config, final_df: pd.DataFrame, selection: dict[str, Any], confidence_artifacts
) -> tuple[dict[str, Any], bool]:
    """Build top-level prediction summary fields for the report."""
    prediction_summary: dict[str, Any] = {
        "selection_type": selection["selection_type"],
        "selected_models": selection["selected_models"],
        "rows_total": len(final_df),
        "rows_unknown_original": int(final_df[config.target_column].isna().sum()),
        "confidence_method": (
            "within-model ranks" if config.prediction_confidence_level_column in final_df else "maximum probability"
        ),
        "confidence_level_counts": (
            final_df[config.prediction_confidence_level_column].value_counts().to_dict()
            if config.prediction_confidence_level_column in final_df
            else {}
        ),
        "inspection_need_counts": (
            final_df["inspection_need"].value_counts(dropna=False).to_dict() if "inspection_need" in final_df else {}
        ),
        "inspection_check_type_counts": (
            final_df["inspection_check_type"].value_counts(dropna=False).to_dict() if "inspection_check_type" in final_df else {}
        ),
        "label_prediction_status_counts": (
            final_df["label_prediction_status"].value_counts(dropna=False).to_dict()
            if "label_prediction_status" in final_df
            else {}
        ),
        "inspection_score_mean": (
            None
            if "inspection_score" not in final_df or pd.isna(final_df["inspection_score"].mean())
            else float(final_df["inspection_score"].mean())
        ),
    }
    if confidence_artifacts is not None:
        prediction_summary.update(confidence_artifacts.summary)
    # Accept both current and legacy contract keys when summarizing saved selections.
    confidence_contract = selection.get("confidence") or selection.get("rank_confidence") or {}
    class_reliability_enabled = bool(confidence_contract.get("class_reliability_enabled", False))
    # Report whether the guard was actually applied, separately from whether it was configured.
    class_reliability_applied = bool(final_df.get("prediction_class_reliability_applied", pd.Series([False])).iloc[0])
    prediction_summary.update(
        {
            "class_reliability_enabled": class_reliability_enabled,
            "class_reliability_applied_to_final_confidence": class_reliability_applied,
            "final_confidence_rule": "rank_plus_class_reliability" if class_reliability_applied else "rank_only",
        }
    )
    return prediction_summary, class_reliability_applied


def _build_confidence_formula(class_reliability_applied: bool) -> pd.DataFrame:
    """Build the explanatory table for final-confidence calculation."""
    return pd.DataFrame(
        [
            {
                "Component": "Rank confidence",
                "How it is computed": (
                    "Borda score and rank-range agreement across models; final class must agree with the Borda winner"
                ),
                "Used in final confidence": "Always",
            },
            {
                "Component": "Class reliability",
                "How it is computed": ("OOF precision of the predicted class from the frozen class-reliability contract"),
                "Used in final confidence": "Only when class_reliability_enabled is true",
            },
            {
                "Component": "Final prediction confidence",
                "How it is computed": (
                    "If class reliability is enabled, take the lower of rank confidence and class reliability; otherwise"
                    " use rank confidence only"
                ),
                "Used in final confidence": ("rank_plus_class_reliability" if class_reliability_applied else "rank_only"),
            },
        ]
    )


def _build_output_links(config, confidence_artifacts, inspection_artifacts) -> list[dict[str, Any]]:
    """Build saved-output links for available report artifacts."""
    output_links: list[dict[str, Any]] = [
        {"label": "Final Predictions Joblib", "path": config.predict_dir / "final_predictions.joblib"},
        {"label": "Final Predictions Preview CSV", "path": config.predict_dir / "final_predictions_preview.csv"},
    ]
    if confidence_artifacts is not None:
        output_links.extend(
            [
                {
                    "label": "Borda/Correctness Group Summary",
                    "path": config.predict_dir / "data" / "diagnostic_group_summary.csv",
                },
                {
                    "label": "Confidence Metric Median Summary",
                    "path": config.predict_dir / "data" / "confidence_metric_median_summary.csv",
                },
                {
                    "label": "Need-to-Check by Predicted Class",
                    "path": config.predict_dir / "data" / "need_to_check_by_predicted_class.csv",
                },
                {"label": "Confidence Component Matrix", "path": config.predict_dir / "data" / "need_to_check_reason_matrix.csv"},
                {"label": "Need-to-Check Confusion Matrix", "path": config.predict_dir / "data" / "need_to_check_confusion.csv"},
            ]
        )
    if inspection_artifacts is not None:
        inspection_output_labels = {
            "inspection_need_summary": "Inspection Need Summary",
            "inspection_check_type_summary": "Inspection Check Type Summary",
            "inspection_priority_by_predicted_class": "Inspection Priority by Predicted Crop",
            "inspection_declaration_conflict_matrix": "Declaration Conflict Matrix",
            "inspection_evidence_quality_matrix": "EO Evidence Quality Matrix",
            "inspection_confidence_review_comparison": "Confidence Review Comparison",
            "inspection_need_by_declaration_status": "Inspection Need by Declaration Status",
            "inspection_score_by_confidence_data": "Inspection Score by Confidence and Data Reliability",
            "inspection_top_priority_parcels": "Top-Priority Parcel Queue",
        }
        output_links.extend(
            {"label": label, "path": config.predict_dir / "data" / f"{artifact_name}.csv"}
            for artifact_name, label in inspection_output_labels.items()
        )
    return output_links
