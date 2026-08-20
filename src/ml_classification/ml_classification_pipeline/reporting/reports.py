"""High-level report assembly functions for all pipeline stages."""

from __future__ import annotations

from typing import Any

import pandas as pd

from common_libraries.io_library import read_data

from .html import write_html_report


# Each writer adapts stage-specific results to the shared HTML report renderer.
def write_check_report(config, summary: dict[str, Any], feature_profile: pd.DataFrame) -> None:
    """Write the step-1 check report with summaries and validation visuals.

    Build small preview sections showing validation summaries, a feature
    profile table, and any generated diagnostic plots or interactive map
    embeds that exist in the check output folder.
    """
    # Assemble available plots and optional map embeds for the check report.
    images = [
        {"title": "Missing Values", "path": config.check_dir / "plots" / "missing_values.png"},
        {"title": "Known Target Distribution", "path": config.check_dir / "plots" / "target_distribution.png"},
    ]
    embeds = []
    label_map_path = config.check_dir / "plots" / "known_labels_map.png"
    if label_map_path.exists():
        images.append({"title": "Known Labels Map", "path": label_map_path})
    osm_map_path = config.check_dir / "plots" / "known_labels_map_osm.html"
    if osm_map_path.exists():
        embeds.append({"title": "Known Labels Map with OSM Basemap", "path": osm_map_path})

    write_html_report(
        config.check_dir / "report.html",
        "Step 1 Report: Data Check",
        "Input validation, feature screening, and basic class diagnostics.",
        sections=[
            {"title": "Summary", "kv": summary},
            {
                "title": "Feature Profile",
                "text": "Active features are the ones that move forward into model preparation.",
                "table": feature_profile,
            },
            {"title": "Plots", "images": images, "embeds": embeds},
        ],
    )


def write_prepare_report(config, prepare_summary: dict[str, Any], train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Write the step-2 preparation report with split and fold diagnostics."""
    # Build compact summary tables for split statistics and fold sizes.
    summary_kv = {key: value for key, value in prepare_summary.items() if key != "cv_folds"}
    folds_df = pd.DataFrame(
        [
            {"fold": fold["fold"], "train_rows": len(fold["train_index"]), "valid_rows": len(fold["valid_index"])}
            for fold in prepare_summary["cv_folds"]
        ]
    )
    split_preview = pd.DataFrame(
        {
            "split": ["train", "test"],
            "rows": [len(train_df), len(test_df)],
            "unique_labels": [train_df[config.target_column].nunique(), test_df[config.target_column].nunique()],
        }
    )
    write_html_report(
        config.prepare_dir / "report.html",
        "Step 2 Report: Data Preparation",
        "Train-test split, fold generation, and preparation outputs for the modeling stage.",
        sections=[
            {"title": "Summary", "kv": summary_kv | {"train_preview_rows": min(5, len(train_df))}},
            {"title": "Split Overview", "table": split_preview},
            {
                "title": "Cross-Validation Folds",
                "text": "Fold sizes only, without the raw train and validation index lists.",
                "table": folds_df,
            },
            {
                "title": "Plots",
                "images": [
                    {"title": "Train-Test Class Balance", "path": config.prepare_dir / "plots" / "train_test_distribution.png"},
                    {
                        "title": "Numeric Correlation Heatmap",
                        "path": config.prepare_dir / "plots" / "numeric_correlation_heatmap.png",
                    },
                    {
                        "title": "Feature Distributions Train vs Test",
                        "path": config.prepare_dir / "plots" / "feature_distributions_train_vs_test.png",
                    },
                    {"title": "Spatial Train Test Split", "path": config.prepare_dir / "plots" / "spatial_train_test_split.png"},
                ],
            },
        ],
    )


def write_train_report(
    config, training_summary_df: pd.DataFrame, model_specs: dict[str, dict[str, Any]], failed_models_df: pd.DataFrame
) -> None:
    """Write the step-3 training report with ranking and search artifacts."""
    # Collect per-model output links so users can inspect full search details.
    links = []
    for model_name in model_specs:
        model_dir = config.train_model_dir(model_name)
        links.extend(
            [
                {"label": f"{model_name} CV Results CSV", "path": model_dir / "cv_results.csv"},
                {"label": f"{model_name} Best Model (joblib)", "path": model_dir / "best_model.joblib"},
                {"label": f"{model_name} Search Results Plot", "path": model_dir / "search_results.png"},
            ]
        )
    params_df = pd.DataFrame(
        [{"model": model_name, "best_params": spec["best_params"]} for model_name, spec in model_specs.items()]
    )
    write_html_report(
        config.train_dir / "report.html",
        "Step 3 Report: Model Training",
        "Candidate models, hyperparameter search outcomes, and selected estimators.",
        sections=[
            {
                "title": "Training Summary",
                "text": (
                    "Models are ranked by the configured CV ranking method. "
                    "`score_minus_std` favors models with both strong average CV score and lower fold-to-fold variability."
                ),
                "table": training_summary_df,
            },
            {"title": "Best Parameters", "table": params_df, "compact_first_column": True},
            {
                "title": "Failed Models",
                "text": "Models listed here were skipped after an error so the rest of the training run could continue.",
                "table": failed_models_df,
            },
            {
                "title": "Cross-Validation Fold Scores",
                "text": (
                    "Each line shows the best hyperparameter setting for a model, with its score on each CV fold. "
                    "This is usually easier to read than raw search-ranking plots."
                ),
                "images": [
                    {"title": "Best CV Score per Fold by Model", "path": config.train_dir / "plots" / "best_cv_fold_scores.png"}
                ],
            },
            {"title": "Detailed Search Outputs", "links": links},
        ],
    )


def write_evaluate_report(
    config,
    train_metrics_df: pd.DataFrame,
    test_metrics_df: pd.DataFrame,
    selection: dict[str, Any],
    geo_train_metrics_df: pd.DataFrame | None = None,
    geo_test_metrics_df: pd.DataFrame | None = None,
    ranking_method_metrics_df: pd.DataFrame | None = None,
    ranking_train_metrics_df: pd.DataFrame | None = None,
) -> None:
    """Write the step-4 evaluation report including strategy selection details."""
    # Gather generated visuals and interpretability outputs for report sections.
    image_paths = list((config.evaluate_dir / "confusion_matrices").glob("*.png"))
    image_paths += list((config.evaluate_dir / "panels").glob("*.png"))
    image_paths += list((config.evaluate_dir / "curves").glob("*.png"))
    image_paths += list((config.evaluate_dir / "plots").glob("*.png"))
    images = [{"title": path.stem.replace("_", " ").title(), "path": path} for path in sorted(image_paths)]
    links = [{"label": path.name, "path": path} for path in sorted((config.evaluate_dir / "reports").glob("*.csv"))]
    interpretability_summary_path = config.evaluate_dir / "interpretability" / "interpretability_summary.csv"
    interpretability_df = (
        read_data(str(interpretability_summary_path), watch_curly_brackets=False)
        if interpretability_summary_path.exists()
        else None
    )
    interpretability_image_paths = list((config.evaluate_dir / "interpretability").glob("*.png"))
    interpretability_images = [
        {"title": path.stem.replace("_", " ").title(), "path": path} for path in sorted(interpretability_image_paths)
    ]
    interpretability_links = [
        {"label": path.name, "path": path}
        for path in sorted((config.evaluate_dir / "interpretability").glob("*.csv"))
        if path.name != "interpretability_summary.csv"
    ]
    highlighted_models = ["soft_voting"] if selection["selection_type"] == "soft_voting" else [selection["selected_models"][0]]
    sections = [
        {"title": "Selected Strategy", "kv": selection},
        {
            "title": "Metric Definitions",
            "table": pd.DataFrame(
                [
                    {"metric": "accuracy", "meaning": "Overall share of correct predictions across all rows."},
                    {"metric": "balanced_accuracy", "meaning": "Average recall across classes, giving each class equal weight."},
                    {
                        "metric": "f1_macro",
                        "meaning": "Average F1 across classes with equal weight, useful when minority classes matter.",
                    },
                    {
                        "metric": "f1_weighted",
                        "meaning": "Average F1 across classes weighted by class frequency, so common classes count more.",
                    },
                ]
            ),
            "compact_first_column": True,
        },
        {
            "title": "Model Results on Train Set (Analytical per Class)",
            "text": "GeoDataFrameClassifier-style table included for continuity with legacy outputs.",
            "table": geo_train_metrics_df if geo_train_metrics_df is not None else train_metrics_df,
            "highlight_rows_where": {"column": "model", "values": ["Voting"]},
        },
        {
            "title": "Model Results on Test Set (Analytical per Class)",
            "text": "GeoDataFrameClassifier-style table included for continuity with legacy outputs.",
            "table": geo_test_metrics_df if geo_test_metrics_df is not None else test_metrics_df,
            "highlight_rows_where": {"column": "model", "values": ["Voting"]},
        },
        {
            "title": "Model Results on Train Set (Average)",
            "table": train_metrics_df,
            "highlight_rows_where": {"column": "model", "values": highlighted_models},
        },
        {
            "title": "Model Results on Test Set (Average)",
            "table": test_metrics_df,
            "highlight_rows_where": {"column": "model", "values": highlighted_models},
        },
        {"title": "Classification Reports", "links": links},
        {"title": "Evaluation Visuals", "images": images},
    ]

    # Append rank_average and rank_median as extra rows in the test-set model results table.
    # Rank-based methods are agnostic to probability calibration differences between model families,
    # so they serve as a natural complement to soft_voting in the same comparison table.
    if ranking_method_metrics_df is not None and not ranking_method_metrics_df.empty:
        rank_rows = ranking_method_metrics_df[
            ranking_method_metrics_df["method"].isin({"rank_average", "rank_median"})
        ].copy()
        if not rank_rows.empty:
            rank_rows = rank_rows.rename(columns={"method": "model"})
            rank_rows = rank_rows.drop(columns=["n_models_used", "is_preferred"], errors="ignore")
            for col in test_metrics_df.columns:
                if col not in rank_rows.columns:
                    rank_rows[col] = float("nan")
            rank_rows = rank_rows[test_metrics_df.columns]
            augmented_test_df = pd.concat([test_metrics_df, rank_rows], ignore_index=True)
            for idx, section in enumerate(sections):
                if section.get("title") == "Model Results on Test Set (Average)":
                    sections[idx] = {
                        "title": "Model Results on Test Set (Average)",
                        "text": (
                            "rank_average and rank_median rows show the performance of rank-based ensemble aggregation. "
                            "These methods are robust to probability calibration differences across model families."
                        ),
                        "table": augmented_test_df,
                        "highlight_rows_where": {"column": "model", "values": highlighted_models + ["rank_average", "rank_median"]},
                    }
                    break

    # Append rank rows to the train-set average table for comparison with the test-set results.
    if ranking_train_metrics_df is not None and not ranking_train_metrics_df.empty:
        rank_rows_train = ranking_train_metrics_df[
            ranking_train_metrics_df["method"].isin({"rank_average", "rank_median"})
        ].copy()
        if not rank_rows_train.empty:
            rank_rows_train = rank_rows_train.rename(columns={"method": "model"})
            rank_rows_train = rank_rows_train.drop(columns=["n_models_used", "is_preferred"], errors="ignore")
            for col in train_metrics_df.columns:
                if col not in rank_rows_train.columns:
                    rank_rows_train[col] = float("nan")
            rank_rows_train = rank_rows_train[train_metrics_df.columns]
            augmented_train_df = pd.concat([train_metrics_df, rank_rows_train], ignore_index=True)
            for idx, section in enumerate(sections):
                if section.get("title") == "Model Results on Train Set (Average)":
                    sections[idx] = {
                        "title": "Model Results on Train Set (Average)",
                        "text": (
                            "rank_average and rank_median rows show rank-based ensemble aggregation on the training set. "
                            "Compare with the test-set table to assess overfitting."
                        ),
                        "table": augmented_train_df,
                        "highlight_rows_where": {"column": "model", "values": highlighted_models + ["rank_average", "rank_median"]},
                    }
                    break

    ranking_parcels_path = config.evaluate_dir / "ranking" / "parcel_best_class_by_ranking.csv"
    ranking_metrics_path = config.evaluate_dir / "ranking" / "ranking_method_metrics.csv"
    ranking_links = []
    if ranking_metrics_path.exists():
        ranking_links.append({"label": "Ranking Method Metrics CSV", "path": ranking_metrics_path})
    if ranking_parcels_path.exists():
        ranking_links.append({"label": "Parcel Best Class by Ranking CSV", "path": ranking_parcels_path})
    if ranking_links:
        sections.insert(len(sections) - 1, {"title": "Ranking Artifacts", "links": ranking_links})
    if interpretability_df is not None or interpretability_images or interpretability_links:
        sections.append(
            {
                "title": "Interpretability",
                "text": (
                    "This optional section focuses on the top-ranked base models. "
                    "Feature importance is lightweight and SHAP remains opt-in."
                ),
                "table": interpretability_df,
                "images": interpretability_images,
                "links": interpretability_links,
            }
        )
    write_html_report(
        config.evaluate_dir / "report.html",
        "Step 4 Report: Model Evaluation",
        "Holdout-set comparison, report tables, and final single-model versus voting selection.",
        sections=sections,
    )


def write_predict_report(config, final_df: pd.DataFrame, selection: dict[str, Any]) -> None:
    """Write the step-5 prediction report with final fill outputs."""
    # Prepare preview columns and optional probability columns for display.
    preview_columns = [config.id_column, config.target_column, config.prediction_column, config.prediction_filled_column]
    probability_columns = [column for column in final_df.columns if column.startswith(f"{config.probability_prefix}_")]
    images = [{"title": "Filled Target Distribution", "path": config.predict_dir / "plots" / "filled_target_distribution.png"}]
    predicted_map_path = config.predict_dir / "plots" / "predicted_labels_map.png"
    if predicted_map_path.exists():
        images.append({"title": "Classified Predicted Labels Map", "path": predicted_map_path})
    write_html_report(
        config.predict_dir / "report.html",
        "Step 5 Report: Final Prediction Fill",
        "Predictions on the full dataset, including rows that originally had unknown labels.",
        sections=[
            {
                "title": "Prediction Summary",
                "kv": {
                    "selection_type": selection["selection_type"],
                    "selected_models": selection["selected_models"],
                    "rows_total": len(final_df),
                    "rows_unknown_original": int(final_df[config.target_column].isna().sum()),
                },
            },
            {"title": "Prediction Preview", "table": final_df[preview_columns + probability_columns].head(50)},
            {"title": "Plots", "images": images},
            {
                "title": "Saved Outputs",
                "links": [
                    {"label": "Final Predictions Joblib", "path": config.predict_dir / "final_predictions.joblib"},
                    {"label": "Final Predictions Preview CSV", "path": config.predict_dir / "final_predictions_preview.csv"},
                ],
            },
        ],
    )


def write_index_report(config) -> None:
    """Write a top-level index page linking all step reports."""
    # Provide a single navigation entrypoint for generated pipeline reports.
    write_html_report(
        config.project_dir / "report_index.html",
        "ML Classification Pipeline Report Index",
        "Open any step report below to inspect saved outputs, tables, and plots.",
        sections=[
            {
                "title": "Step Reports",
                "links": [
                    {"label": "01 Check", "path": config.check_dir / "report.html"},
                    {"label": "02 Prepare", "path": config.prepare_dir / "report.html"},
                    {"label": "03 Train", "path": config.train_dir / "report.html"},
                    {"label": "04 Evaluate", "path": config.evaluate_dir / "report.html"},
                    {"label": "05 Predict", "path": config.predict_dir / "report.html"},
                ],
            }
        ],
    )
