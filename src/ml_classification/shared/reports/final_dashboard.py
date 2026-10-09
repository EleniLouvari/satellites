"""Public compatibility façade for the final dashboard modules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import seaborn as sns  # noqa: F401

from ml_classification.shared.persistence import ensure_dir, load_joblib
from ml_classification.shared.reports.confidence_diagnostics import (
    export_prediction_confidence_diagnostics,
    prepare_class_risk_display_table,
)
from ml_classification.shared.reports.final_dashboard_artifact import (
    _INDEX_COLUMN_PATTERN,
    _MODELLED_LABEL_COLUMN,
    _MODELLED_LABEL_NAME_COLUMN,
    DashboardPaths,
    _build_class_counts_rows,
    _build_class_lookup,
    _build_cv_rows,
    _build_dashboard_blocks,
    _build_dashboard_charts,
    _build_dashboard_sources,
    _build_index_distribution_rows,
    _build_map_html,
    _build_overview_rows,
    _build_performance_rows,
    _build_phenology_rows,
    _build_probability_rows,
    _build_seasonal_range_rows,
    _build_seasonal_summary,
    _clean_label_code,
    _dashboard_source_paths,
    _display_name,
    _filter_model_rows,
    _index_columns,
    _json_value,
    _load_dashboard_class_data,
    _load_dashboard_model_data,
    _metric_columns,
    _read_json,
    _read_parcels,
    _records,
    _require_complete_sources,
    _source,
    _sql_quoted_values,
    _strategy_model_name,
    _summarize_phenology,
    build_final_dashboard_artifact,
)
from ml_classification.shared.reports.final_dashboard_pipeline import (
    _pipeline_dashboard_data_links,
    _pipeline_dashboard_summary,
    _pipeline_performance_section,
    _save_monthly_index_boxplots,
    _save_monthly_index_range_lines,
    _save_selected_cv_fold_plot,
    _save_train_test_plot,
    _voting_member_row_styles_for_pipeline,
)
from ml_classification.shared.reports.report_html import write_html_report

# Preserve the established import surface while implementations live in the focused dashboard modules.
__all__ = [
    "_INDEX_COLUMN_PATTERN",
    "_MODELLED_LABEL_COLUMN",
    "_MODELLED_LABEL_NAME_COLUMN",
    "DashboardPaths",
    "_build_class_counts_rows",
    "_build_class_lookup",
    "_build_cv_rows",
    "_build_dashboard_blocks",
    "_build_dashboard_charts",
    "_build_dashboard_sources",
    "_build_index_distribution_rows",
    "_build_map_html",
    "_build_overview_rows",
    "_build_performance_rows",
    "_build_phenology_rows",
    "_build_probability_rows",
    "_build_seasonal_range_rows",
    "_build_seasonal_summary",
    "_clean_label_code",
    "_dashboard_source_paths",
    "_display_name",
    "_filter_model_rows",
    "_index_columns",
    "_json_value",
    "_load_dashboard_class_data",
    "_load_dashboard_model_data",
    "_metric_columns",
    "_pipeline_confidence_section",
    "_pipeline_dashboard_data_links",
    "_pipeline_dashboard_summary",
    "_pipeline_parcels",
    "_pipeline_performance_section",
    "_read_json",
    "_read_parcels",
    "_records",
    "_require_complete_sources",
    "_save_monthly_index_boxplots",
    "_save_monthly_index_range_lines",
    "_save_selected_cv_fold_plot",
    "_save_train_test_plot",
    "_source",
    "_sql_quoted_values",
    "_strategy_model_name",
    "_summarize_phenology",
    "_voting_member_row_styles_for_pipeline",
    "build_final_dashboard_artifact",
    "main",
    "write_final_dashboard_artifact",
    "write_pipeline_final_dashboard",
]


def _pipeline_parcels(config, prepare: dict[str, Any]) -> tuple[gpd.GeoDataFrame, dict[str, str]]:
    """Load and normalize the check-stage dataset for automatic reporting."""
    source = load_joblib(config.check_dir / "input_dataset.joblib")
    if config.target_column not in source.columns:
        raise ValueError(f"Error: Input dataset is missing target column {config.target_column!r}.")

    parcels = source.copy()
    parcels[_MODELLED_LABEL_COLUMN] = parcels[config.target_column].map(_clean_label_code)
    # Restrict seasonal summaries to the same canonical class set used during model preparation.
    modelled_labels = {_clean_label_code(value) for value in prepare.get("target_labels", [])}
    parcels = parcels.loc[parcels[_MODELLED_LABEL_COLUMN].isin(modelled_labels)].copy()

    label_name_candidates = [config.target_column.removesuffix("_code"), "initial_label", "label_name", "class_name"]
    label_name_column = next(
        (column for column in label_name_candidates if column != config.target_column and column in parcels.columns), None
    )
    if label_name_column:
        parcels[_MODELLED_LABEL_NAME_COLUMN] = parcels[label_name_column]
    class_lookup = _build_class_lookup(parcels)
    return parcels, class_lookup


def _export_pipeline_seasonality(parcels, class_lookup, data_dir, plots_dir):
    """Export monthly tables and plots, returning phenology, images, and quality notes."""
    index_columns = [column for column in parcels.columns if _INDEX_COLUMN_PATTERN.fullmatch(str(column))]
    phenology = pd.DataFrame()
    seasonal_range = pd.DataFrame()
    seasonal_images: list[dict[str, Any]] = []
    seasonal_quality_notes: list[str] = []
    if index_columns:
        distribution = _build_index_distribution_rows(parcels, class_lookup)
        phenology = _summarize_phenology(distribution)
        seasonal_range = _build_seasonal_range_rows(phenology)
        phenology.to_csv(data_dir / "monthly_index_by_class.csv", index=False)
        seasonal_range.to_csv(data_dir / "monthly_index_high_low_by_class.csv", index=False)
        _save_monthly_index_boxplots(distribution, plots_dir / "seasonal_ndvi_ndwi_boxplots_by_month_class.png")
        _save_monthly_index_range_lines(phenology, "NDVI", plots_dir / "seasonal_ndvi_high_low_by_class.png")
        _save_monthly_index_range_lines(phenology, "NDWI", plots_dir / "seasonal_ndwi_high_low_by_class.png")
        for index in ("NDVI", "NDWI"):
            values = distribution.loc[distribution["index"] == index, "parcel_value"]
            if not values.empty and values.nunique(dropna=True) <= 1:
                seasonal_quality_notes.append(
                    f"Data-quality warning: every {index} value in this evaluation input equals {float(values.iloc[0]):.2f}, "
                    "so its monthly boxes have no spread."
                )
        for stale_plot in (
            "seasonal_peak_by_class.png",
            "seasonal_ndvi_boxplots_by_class_month.png",
            "seasonal_ndwi_boxplots_by_class_month.png",
        ):
            (plots_dir / stale_plot).unlink(missing_ok=True)
        seasonal_images = [
            {
                "title": "Monthly NDVI and NDWI Distributions by Modeled Class",
                "path": plots_dir / "seasonal_ndvi_ndwi_boxplots_by_month_class.png",
            },
            {
                "title": "Monthly Highest and Lowest NDVI by Modeled Class",
                "path": plots_dir / "seasonal_ndvi_high_low_by_class.png",
            },
            {
                "title": "Monthly Highest and Lowest NDWI by Modeled Class",
                "path": plots_dir / "seasonal_ndwi_high_low_by_class.png",
            },
        ]

    return phenology, seasonal_images, seasonal_quality_notes


def _pipeline_confidence_section(confidence_artifacts):
    """Build a confidence section using the diagnostic images available for this run."""
    confidence_images = []
    for key, title in (
        ("overview", "Borda Consensus and Prediction Correctness"),
        ("class_rate", "Need-to-Check Rate by Predicted Class"),
        ("reason_heatmap", "Confidence Components of Incorrect Consensus Predictions"),
    ):
        path = confidence_artifacts.images.get(key)
        if path is not None:
            confidence_images.append({"title": title, "path": path})
    return {
        "title": "Prediction Confidence Diagnostics",
        "text": (
            "These diagnostics use only parcels with known original reference labels. The overview shows "
            "whether Borda consensus aligns with correctness; the class chart identifies where incorrect "
            "consensus predictions concentrate; and the component matrix shows how class reliability and "
            "rank confidence combine inside the need-to-check cohort. Unknown fill rows are excluded from "
            "all correctness rates."
        ),
        "images": confidence_images,
        "table": prepare_class_risk_display_table(
            confidence_artifacts.tables["need_to_check_by_predicted_class"], limit=15
        ),
    }


def write_pipeline_final_dashboard(
    config,
    train_metrics_df,
    test_metrics_df,
    selection,
    prediction_df=None,
) -> Path:
    """Create the final HTML dashboard, optionally including prediction diagnostics."""
    dashboard_dir = ensure_dir(config.final_dashboard_dir)
    plots_dir = ensure_dir(dashboard_dir / "plots")
    data_dir = ensure_dir(dashboard_dir / "data")
    prepare = _read_json(config.prepare_dir / "prepare_summary.json")
    parcels, class_lookup = _pipeline_parcels(config, prepare)

    phenology, seasonal_images, seasonal_quality_notes = _export_pipeline_seasonality(
        parcels, class_lookup, data_dir, plots_dir
    )

    training = pd.read_csv(config.train_dir / "training_summary.csv")
    fold_scores = pd.read_csv(config.train_dir / "best_cv_fold_scores.csv")
    selected_models = {str(value) for value in selection.get("selected_models", [])}
    strategy_model = _strategy_model_name(selection, test_metrics_df)
    training = _filter_model_rows(training, selected_models, "training_summary.csv")
    fold_scores = _filter_model_rows(fold_scores, selected_models, "best_cv_fold_scores.csv")
    # Include both selected members and the aggregate strategy in train/test comparisons.
    performance_models = selected_models | {strategy_model}
    selected_train_metrics = _filter_model_rows(train_metrics_df, performance_models, "train evaluation metrics")
    selected_test_metrics = _filter_model_rows(test_metrics_df, performance_models, "test evaluation metrics")
    cv_folds, cv_ranking = _build_cv_rows(training, fold_scores, selected_models)
    performance, performance_table = _build_performance_rows(selected_train_metrics, selected_test_metrics)
    primary_metric = config.scoring_primary if config.scoring_primary in performance["metric"].unique() else "f1_macro"
    if primary_metric not in performance["metric"].unique():
        primary_metric = str(performance["metric"].iloc[0])
    _save_selected_cv_fold_plot(cv_folds, plots_dir / "selected_model_cv_folds.png")
    _save_train_test_plot(performance, primary_metric, plots_dir / "train_test_model_comparison.png")
    cv_ranking.to_csv(data_dir / "cv_model_ranking.csv", index=False)
    performance_table.to_csv(data_dir / "train_test_model_metrics.csv", index=False)

    confidence_artifacts = None
    # Evaluation can build this dashboard before prediction diagnostics are available.
    if prediction_df is not None:
        prediction_column = getattr(config, "prediction_column", f"{config.target_column}_prediction")
        confidence_artifacts = export_prediction_confidence_diagnostics(
            prediction_df,
            config.target_column,
            prediction_column,
            dashboard_dir,
            include=("overview", "class_rate", "reason_heatmap"),
        )

    summary = _pipeline_dashboard_summary(
        prepare,
        selection,
        selected_train_metrics,
        selected_test_metrics,
        strategy_model,
        primary_metric,
        cv_ranking,
        confidence_artifacts,
    )
    test_score = summary[f"test_{primary_metric}"]

    probability_optimization = selection.get("probability_optimization") or {}
    probability_table = pd.DataFrame(
        [
            {
                "metric": "Macro F1",
                "baseline": probability_optimization.get("baseline_oof_macro_f1"),
                "optimized": probability_optimization.get("optimized_oof_macro_f1"),
            },
            {
                "metric": "Accuracy",
                "baseline": probability_optimization.get("baseline_oof_accuracy"),
                "optimized": probability_optimization.get("optimized_oof_accuracy"),
            },
        ]
    # Omit optimization rows with no saved OOF scores instead of presenting fabricated zeroes.
    ).dropna(how="all", subset=["baseline", "optimized"])

    map_embeds = []
    osm_map = config.check_dir / "plots" / "known_labels_map_osm.html"
    if osm_map.exists():
        map_embeds.append({"title": "Known Modeled Parcels on OpenStreetMap", "path": osm_map})
    data_links = _pipeline_dashboard_data_links(
        data_dir, has_phenology=not phenology.empty, has_confidence=confidence_artifacts is not None
    )
    sections: list[dict[str, Any]] = [
        {"title": "Evaluation Snapshot", "kv": summary},
        {
            "title": "Study Area and Parcel Coverage",
            "text": "The interactive map uses OpenStreetMap tiles when network access is available.",
            "embeds": map_embeds,
        },
        {
            "title": "Seasonal Crop Profiles",
            "text": (
                "The combined Seaborn figure has one row per month and two columns: NDVI on the left and NDWI "
                "on the right. Every panel places modeled class/label on the shared x-axis and the parcel-level "
                "index distribution on the y-axis. "
                "Boxes show the interquartile range and median; whiskers use 1.5×IQR and outlier markers are hidden for readability. "
                "The high/low line figures show each class's highest and lowest parcel median in every month. "
                + " ".join(seasonal_quality_notes)
            ),
            "images": seasonal_images,
        },
        {
            "title": "Cross-Validation Stability",
            "text": "Only selected strategy members are shown. Fold scores use each selected model's best hyperparameter setting; the table follows the configured CV ranking score.",
            "images": [{"title": "Selected-Model CV Score per Fold", "path": plots_dir / "selected_model_cv_folds.png"}],
            "table": cv_ranking,
            "row_styles_where": _voting_member_row_styles_for_pipeline(cv_ranking, selected_models),
        },
        _pipeline_performance_section(performance_table, strategy_model, plots_dir),
    ]
    if confidence_artifacts is not None:
        sections.append(_pipeline_confidence_section(confidence_artifacts))
    sections.extend(
        [
            {
                "title": "Best Model and Voting Result",
                "kv": {
                    "selection_type": selection.get("selection_type"),
                    "selected_models": list(selection.get("selected_models", [])),
                    "selected_metric": selection.get("selected_metric"),
                    "selected_cv_score": selection.get("selected_score"),
                    f"holdout_{primary_metric}": test_score,
                },
                "table": probability_table,
            },
            {"title": "Dashboard Data", "links": data_links},
        ]
    )
    output_path = dashboard_dir / "report.html"
    write_html_report(
        output_path,
        "Final Crop Classification Dashboard",
        (
            "Study-area coverage, crop seasonality, cross-validation stability, holdout performance, final model "
            "selection, and prediction confidence diagnostics."
            if confidence_artifacts is not None
            else "Study-area coverage, crop seasonality, cross-validation stability, holdout performance, and final model selection."
        ),
        sections,
    )
    return output_path


def write_final_dashboard_artifact(
    output_path: str | Path,
    parcels_path: str | Path,
    project_dir: str | Path,
    *,
    title: str = "Crop Classification — Final Dashboard",
) -> Path:
    """Build and write the canonical dashboard artifact JSON."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact = build_final_dashboard_artifact(parcels_path, project_dir, title=title)
    output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the final crop-classification dashboard artifact.")
    parser.add_argument("--parcels", required=True, type=Path, help="Source GeoParquet with monthly parcel statistics.")
    parser.add_argument("--project-dir", required=True, type=Path, help="Completed ML classification project directory.")
    parser.add_argument("--output", required=True, type=Path, help="Destination artifact.json path.")
    parser.add_argument("--title", default="Crop Classification — Final Dashboard")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = _parse_args()
    output = write_final_dashboard_artifact(args.output, args.parcels, args.project_dir, title=args.title)
    print(f"Wrote dashboard artifact: {output}")


if __name__ == "__main__":
    main()
