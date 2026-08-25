"""High-level report assembly functions for all pipeline stages."""

from __future__ import annotations

from typing import Any

import pandas as pd

from common_libraries.io_library import read_data

from .html import write_html_report


_SPECIAL_MODEL_DISPLAY_NAMES = {
    "soft_voting": "Voting",
    "rank_average": "Rank Average",
    "rank_median": "Rank Median",
}
_AVERAGE_SPECIAL_MODELS = ("soft_voting", "rank_average", "rank_median")
_ANALYTICAL_SPECIAL_MODELS = ("Voting",)
_RANKING_METHODS = ("rank_average", "rank_median")
_ENSEMBLE_DISPLAY_NAMES = frozenset(_SPECIAL_MODEL_DISPLAY_NAMES.values())
_VOTING_STYLE_NOTE = (
    "Green rows are base models included in Voting; gray rows are the remaining base models; "
    "amber rows are aggregate Voting or ranking outputs."
)
_PERCENTAGE_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "f1_macro",
    "f1_weighted",
    "precision_weighted",
    "recall_weighted",
    "roc_auc_weighted",
)


def _sort_model_rows_only(
    metrics_df: pd.DataFrame,
    scoring_primary: str,
    special_models: tuple[str, ...],
) -> pd.DataFrame:
    """Sort base models by score and retain the requested special-model order."""
    if "model" not in metrics_df.columns:
        return metrics_df.copy()

    ordered_df = metrics_df.copy().reset_index(drop=True)
    ordered_df["_original_order"] = range(len(ordered_df))
    base_df = ordered_df[~ordered_df["model"].isin(special_models)].copy()
    special_df = ordered_df[ordered_df["model"].isin(special_models)].copy()

    if scoring_primary in base_df.columns:
        base_df = base_df.sort_values(by=scoring_primary, ascending=False)

    special_order = {model_name: position for position, model_name in enumerate(special_models)}
    special_df["_special_order"] = special_df["model"].map(special_order).fillna(len(special_models))
    special_df = special_df.sort_values(by=["_special_order", "_original_order"], ascending=[True, True])

    combined_df = pd.concat([base_df, special_df], ignore_index=True)
    return combined_df.drop(columns=["_original_order", "_special_order"], errors="ignore").reset_index(drop=True)


def _rename_special_models(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with display-friendly names for ensemble rows."""
    if "model" not in metrics_df.columns:
        return metrics_df.copy()

    renamed_df = metrics_df.copy()
    renamed_df["model"] = renamed_df["model"].replace(_SPECIAL_MODEL_DISPLAY_NAMES)
    return renamed_df


def _prepare_average_metrics(metrics_df: pd.DataFrame, scoring_primary: str) -> pd.DataFrame:
    """Sort, label, and scale average metrics for report display."""
    sorted_df = _sort_model_rows_only(metrics_df, scoring_primary, _AVERAGE_SPECIAL_MODELS)
    display_df = _rename_special_models(sorted_df)
    percentage_columns = [column for column in _PERCENTAGE_METRICS if column in display_df.columns]
    display_df[percentage_columns] = display_df[percentage_columns] * 100
    return display_df


def _build_analytical_model_order(metrics_df: pd.DataFrame) -> list[str]:
    """Return the average-table display order to reuse in analytical tables."""
    if "model" not in metrics_df.columns:
        return []

    model_order = []
    for model_name in metrics_df["model"].dropna().astype(str):
        display_name = _SPECIAL_MODEL_DISPLAY_NAMES.get(model_name, model_name)
        if display_name not in {"Rank Average", "Rank Median"} and display_name not in model_order:
            model_order.append(display_name)
    return model_order


def _sort_analytical_by_model_order(
    metrics_df: pd.DataFrame,
    model_order: list[str],
    scoring_primary: str,
) -> pd.DataFrame:
    """Order analytical rows like the average table, with Voting kept last."""
    if "model" not in metrics_df.columns:
        return metrics_df.copy()

    ordered_df = metrics_df.copy().reset_index(drop=True)
    ordered_df["_original_order"] = range(len(ordered_df))
    base_df = ordered_df[~ordered_df["model"].isin(_ANALYTICAL_SPECIAL_MODELS)].copy()
    special_df = ordered_df[ordered_df["model"].isin(_ANALYTICAL_SPECIAL_MODELS)].copy()

    order_lookup = {model_name: position for position, model_name in enumerate(model_order)}
    base_df["_model_order"] = base_df["model"].map(order_lookup).fillna(len(model_order))
    sort_columns = ["_model_order"]
    ascending = [True]
    if scoring_primary in base_df.columns:
        sort_columns.append(scoring_primary)
        ascending.append(False)
    base_df = base_df.sort_values(by=sort_columns, ascending=ascending)

    special_df["_special_order"] = special_df["model"].map({"Voting": 0}).fillna(len(_ANALYTICAL_SPECIAL_MODELS))
    special_df = special_df.sort_values(by=["_special_order", "_original_order"], ascending=[True, True])

    combined_df = pd.concat([base_df, special_df], ignore_index=True)
    helper_columns = ["_original_order", "_model_order", "_special_order"]
    return combined_df.drop(columns=helper_columns, errors="ignore").reset_index(drop=True)


def _prepare_analytical_metrics(
    analytical_df: pd.DataFrame | None,
    average_df: pd.DataFrame,
    scoring_primary: str,
) -> pd.DataFrame:
    """Prepare an analytical table, falling back to the average metrics table."""
    if analytical_df is None:
        return _prepare_average_metrics(average_df, scoring_primary)
    return _sort_analytical_by_model_order(
        analytical_df,
        _build_analytical_model_order(average_df),
        scoring_primary,
    )


def _ranking_rows(ranking_metrics_df: pd.DataFrame | None, columns: pd.Index) -> pd.DataFrame:
    """Convert supported ranking-method metrics to model rows with a target schema."""
    if ranking_metrics_df is None or ranking_metrics_df.empty or "method" not in ranking_metrics_df.columns:
        return pd.DataFrame(columns=columns)

    rank_rows = ranking_metrics_df[ranking_metrics_df["method"].isin(_RANKING_METHODS)].copy()
    rank_rows = rank_rows.rename(columns={"method": "model"})
    rank_rows = rank_rows.drop(columns=["n_models_used", "is_preferred"], errors="ignore")
    return rank_rows.reindex(columns=columns)


def _prepare_cv_results_table(
    training_summary_df: pd.DataFrame | None,
    selected_models: list[str] | None = None,
) -> pd.DataFrame:
    """Build a concise CV ranking table with the best-performing model first."""
    if training_summary_df is None or training_summary_df.empty:
        return pd.DataFrame()

    required_columns = {"model", "best_cv_score", "cv_score_std", "cv_ranking_metric"}
    if not required_columns.issubset(training_summary_df.columns):
        return pd.DataFrame()

    ranked_df = training_summary_df.copy()
    ranked_df = ranked_df.sort_values(
        ["cv_ranking_metric", "best_cv_score"],
        ascending=[False, False],
    ).reset_index(drop=True)
    ranked_df.insert(0, "cv_rank", range(1, len(ranked_df) + 1))
    ranked_df = ranked_df.rename(
        columns={
            "best_cv_score": "mean_cv_score",
            "cv_ranking_metric": "cv_selection_score",
            "cv_ranking_method": "ranking_method",
        }
    )
    # Evaluation metrics elsewhere in the report are percentages, so use the
    # same display convention for CV scores.
    score_columns = ["mean_cv_score", "cv_score_std", "cv_selection_score"]
    for column in score_columns:
        ranked_df[column] = pd.to_numeric(ranked_df[column], errors="coerce") * 100

    if selected_models is not None:
        selected_model_set = {str(model_name) for model_name in selected_models}
        ranked_df["selected_for_strategy"] = ranked_df["model"].astype(str).map(
            lambda model_name: "Yes" if model_name in selected_model_set else "No"
        )

    display_columns = [
        "cv_rank",
        "model",
        "mean_cv_score",
        "cv_score_std",
        "cv_selection_score",
        "ranking_method",
        "supports_predict_proba",
        "selected_for_strategy",
    ]
    return ranked_df[[column for column in display_columns if column in ranked_df.columns]]


def _supports_probability(value: Any) -> bool:
    """Normalize boolean-like values loaded directly or through CSV."""
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def _configured_strategy_models(config, training_summary_df: pd.DataFrame) -> list[str]:
    """Infer the training-stage strategy candidates using the production CV ordering."""
    if training_summary_df.empty or "model" not in training_summary_df.columns:
        return []

    ranked_df = training_summary_df.sort_values(
        ["cv_ranking_metric", "best_cv_score"],
        ascending=[False, False],
    )
    if "supports_predict_proba" in ranked_df.columns:
        ranked_df = ranked_df[ranked_df["supports_predict_proba"].map(_supports_probability)]
    ranked_models = ranked_df["model"].astype(str).tolist()
    top_voting_models = getattr(config, "top_voting_models", None)
    if top_voting_models is not None:
        ranked_models = ranked_models[: int(top_voting_models)]
    if getattr(config, "selection_type", "single_model") != "soft_voting":
        ranked_models = ranked_models[:1]
    return ranked_models


def _cv_ranking_formula(scoring_primary: str, ranking_method: str) -> str:
    """Return a human-readable definition of the configured selection score."""
    if ranking_method == "score_minus_std":
        return f"mean CV {scoring_primary} - CV standard deviation"
    return f"mean CV {scoring_primary}"


def _model_selection_strategy(config, selection: dict[str, Any], cv_table: pd.DataFrame) -> dict[str, Any]:
    """Combine configuration and frozen selection metadata for evaluation reporting."""
    scoring_primary = getattr(config, "scoring_primary", "unknown")
    ranking_method = getattr(config, "cv_ranking_method", None)
    if not ranking_method:
        ranking_method = str(selection.get("selected_metric", "cv_mean_score")).removeprefix("cv_")
    selected_models = list(selection.get("selected_models", []))
    selected_score = selection.get("selected_score")
    details = {
        "selection_type": selection.get("selection_type"),
        "selection_source": selection.get("selection_source", "cross_validation"),
        "scoring_primary": scoring_primary,
        "cv_ranking_method": ranking_method,
        "cv_ranking_formula": _cv_ranking_formula(scoring_primary, ranking_method),
        "top_voting_models": getattr(config, "top_voting_models", None),
        "best_cv_model": cv_table.iloc[0]["model"] if not cv_table.empty else None,
        "selected_model_count": len(selected_models),
        "selected_models": selected_models,
        "selected_metric": selection.get("selected_metric", f"cv_{ranking_method}"),
        "mean_selected_cv_score": float(selected_score) * 100 if selected_score is not None else None,
        "score_display_unit": "percent",
    }
    for key in ("class_probability_multipliers", "probability_optimization", "labels"):
        if key in selection:
            details[key] = selection[key]
    if "confidence" in selection:
        confidence = dict(selection["confidence"])
        reliability = confidence.get("class_reliability")
        if isinstance(reliability, dict):
            confidence["class_reliability"] = {
                key: value for key, value in reliability.items() if key != "classes"
            }
        details["confidence"] = confidence
    return details


def _voting_member_row_styles(table: pd.DataFrame, voting_models: list[str] | None) -> list[dict[str, Any]]:
    """Return green/gray row rules that distinguish voting members from other base models."""
    if not voting_models or "model" not in table.columns:
        return []

    model_names = table["model"].dropna().astype(str).drop_duplicates().tolist()
    voting_model_set = {str(model_name) for model_name in voting_models}
    selected_models = [model_name for model_name in model_names if model_name in voting_model_set]
    other_models = [
        model_name
        for model_name in model_names
        if model_name not in voting_model_set and model_name not in _ENSEMBLE_DISPLAY_NAMES
    ]
    return [
        {"column": "model", "values": selected_models, "style": "success"},
        {"column": "model", "values": other_models, "style": "muted"},
    ]


def _build_average_metrics_section(
    title: str,
    metrics_df: pd.DataFrame,
    ranking_metrics_df: pd.DataFrame | None,
    scoring_primary: str,
    highlighted_models: list[str],
    ranking_text: str,
    voting_models: list[str] | None = None,
) -> dict[str, Any]:
    """Build an average-results section, optionally augmented with rank ensembles."""
    rank_rows = _ranking_rows(ranking_metrics_df, metrics_df.columns)
    combined_df = pd.concat([metrics_df, rank_rows], ignore_index=True) if not rank_rows.empty else metrics_df
    display_table = _prepare_average_metrics(combined_df, scoring_primary)
    section = {
        "title": title,
        "table": display_table,
        "highlight_rows_where": {"column": "model", "values": highlighted_models.copy()},
        "row_styles_where": _voting_member_row_styles(display_table, voting_models),
    }
    section_text = _VOTING_STYLE_NOTE if voting_models else ""
    if not rank_rows.empty:
        section_text = " ".join(filter(None, (ranking_text, section_text)))
        section["highlight_rows_where"]["values"] += ["Rank Average", "Rank Median"]
    if section_text:
        section["text"] = section_text
    return section


def _build_ranking_links(evaluate_dir) -> list[dict[str, Any]]:
    """Return links for ranking artifacts that were produced."""
    candidates = (
        ("Ranking Method Metrics CSV", evaluate_dir / "ranking" / "ranking_method_metrics.csv"),
        ("Parcel Best Class by Ranking CSV", evaluate_dir / "ranking" / "parcel_best_class_by_ranking.csv"),
    )
    return [{"label": label, "path": path} for label, path in candidates if path.exists()]


def _report_images(*directories) -> list[dict[str, Any]]:
    """Return consistently titled image descriptors from report directories."""
    image_paths = [path for directory in directories for path in directory.glob("*.png")]
    return [{"title": path.stem.replace("_", " ").title(), "path": path} for path in sorted(image_paths)]


def _report_links(directory, excluded_names: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    """Return sorted CSV link descriptors, excluding any internal summary files."""
    return [
        {"label": path.name, "path": path}
        for path in sorted(directory.glob("*.csv"))
        if path.name not in excluded_names
    ]


def _read_optional_csv(path) -> pd.DataFrame | None:
    """Read a report CSV only when it exists."""
    if not path.exists():
        return None
    return read_data(str(path), watch_curly_brackets=False)


def _highlighted_average_models(selection: dict[str, Any]) -> list[str]:
    """Return display names of models highlighted in average result tables."""
    if selection["selection_type"] == "soft_voting":
        return ["Voting"]
    return [selection["selected_models"][0]]


def _metric_definitions() -> pd.DataFrame:
    """Return the metric glossary shown in every evaluation report."""
    return pd.DataFrame(
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
    )


def _build_evaluation_sections(
    selection: dict[str, Any],
    model_selection_strategy: dict[str, Any],
    cv_selection_table: pd.DataFrame,
    analytical_train_df: pd.DataFrame,
    analytical_test_df: pd.DataFrame,
    train_average_section: dict[str, Any],
    test_average_section: dict[str, Any],
    classification_links: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build the fixed, ordered core of the evaluation report."""
    analytical_text = "GeoDataFrameClassifier-style table included for continuity with legacy outputs."
    selected_strategy_models = list(selection["selected_models"])
    voting_models = selected_strategy_models if selection["selection_type"] == "soft_voting" else None
    if voting_models:
        analytical_text = f"{analytical_text} {_VOTING_STYLE_NOTE}"
    sections = [
        {"title": "Model Selection Strategy", "kv": model_selection_strategy},
    ]
    if not cv_selection_table.empty:
        sections.append(
            {
                "title": "Voting Model Selection from Cross-Validation",
                "text": (
                    "Models are ordered by the CV selection score used by the pipeline, highest first. "
                    "Green rows are included in the frozen strategy and gray rows are not selected. "
                    "Scores are displayed as percentages."
                ),
                "table": cv_selection_table,
                "row_styles_where": _voting_member_row_styles(cv_selection_table, selected_strategy_models),
            }
        )
    sections.extend(
        [
            {"title": "Metric Definitions", "table": _metric_definitions(), "compact_first_column": True},
            {
                "title": "Model Results on Train Set (Analytical per Class)",
                "text": analytical_text,
                "table": analytical_train_df,
                "highlight_rows_where": {"column": "model", "values": ["Voting"]},
                "row_styles_where": _voting_member_row_styles(analytical_train_df, voting_models),
            },
            {
                "title": "Model Results on Test Set (Analytical per Class)",
                "text": analytical_text,
                "table": analytical_test_df,
                "highlight_rows_where": {"column": "model", "values": ["Voting"]},
                "row_styles_where": _voting_member_row_styles(analytical_test_df, voting_models),
            },
            train_average_section,
            test_average_section,
            {"title": "Classification Reports", "links": classification_links},
        ]
    )
    return sections


def _build_supplementary_sections(
    ranking_links: list[dict[str, Any]],
    evaluation_images: list[dict[str, Any]],
    interpretability_df: pd.DataFrame | None,
    interpretability_images: list[dict[str, Any]],
    interpretability_links: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build optional artifact sections while preserving their report order."""
    sections = []
    if ranking_links:
        sections.append({"title": "Ranking Artifacts", "links": ranking_links})
    sections.append({"title": "Evaluation Visuals", "images": evaluation_images})
    if any((interpretability_df is not None, interpretability_images, interpretability_links)):
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
    return sections


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
    strategy_models = _configured_strategy_models(config, training_summary_df)
    cv_results_table = _prepare_cv_results_table(training_summary_df, strategy_models)
    scoring_primary = config.scoring_primary
    ranking_method = config.cv_ranking_method
    strategy_summary = {
        "scoring_primary": scoring_primary,
        "cv_ranking_method": ranking_method,
        "cv_ranking_formula": _cv_ranking_formula(scoring_primary, ranking_method),
        "selection_type": config.selection_type,
        "top_voting_models": config.top_voting_models,
        "best_cv_model": cv_results_table.iloc[0]["model"] if not cv_results_table.empty else None,
        "configured_strategy_models": strategy_models,
    }
    write_html_report(
        config.train_dir / "report.html",
        "Step 3 Report: Model Training",
        "Candidate models, hyperparameter search outcomes, and selected estimators.",
        sections=[
            {"title": "Model Selection Strategy", "kv": strategy_summary},
            {
                "title": "Cross-Validation Model Ranking",
                "text": (
                    "Models are ordered by the configured CV selection score, highest first. "
                    "Green rows are expected strategy members and gray rows are the remaining models. "
                    "Scores are displayed as percentages; score_minus_std rewards both performance and fold stability."
                ),
                "table": cv_results_table,
                "row_styles_where": _voting_member_row_styles(cv_results_table, strategy_models),
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
    confidence_metrics_df: pd.DataFrame | None = None,
    oof_confidence_metrics_df: pd.DataFrame | None = None,
    confidence_by_class_df: pd.DataFrame | None = None,
) -> None:
    """Write the step-4 evaluation report including strategy selection details."""

    images = _report_images(
        config.evaluate_dir / "confusion_matrices",
        config.evaluate_dir / "panels",
        config.evaluate_dir / "curves",
        config.evaluate_dir / "plots",
    )
    classification_links = _report_links(config.evaluate_dir / "reports")
    interpretability_summary_path = config.evaluate_dir / "interpretability" / "interpretability_summary.csv"
    interpretability_df = _read_optional_csv(interpretability_summary_path)
    interpretability_dir = config.evaluate_dir / "interpretability"
    interpretability_images = _report_images(interpretability_dir)
    interpretability_links = _report_links(interpretability_dir, excluded_names=(interpretability_summary_path.name,))
    highlighted_models = _highlighted_average_models(selection)
    voting_models = list(selection["selected_models"]) if selection["selection_type"] == "soft_voting" else None
    cv_summary_path = config.train_dir / "training_summary.csv"
    cv_summary_df = _read_optional_csv(cv_summary_path)
    cv_selection_table = _prepare_cv_results_table(cv_summary_df, list(selection.get("selected_models", [])))
    sorted_geo_train_metrics_df = _prepare_analytical_metrics(
        geo_train_metrics_df, train_metrics_df, config.scoring_primary
    )
    sorted_geo_test_metrics_df = _prepare_analytical_metrics(geo_test_metrics_df, test_metrics_df, config.scoring_primary)
    train_average_section = _build_average_metrics_section(
        "Model Results on Train Set (Average)",
        train_metrics_df,
        ranking_train_metrics_df,
        config.scoring_primary,
        highlighted_models,
        (
            "rank_average and rank_median rows show rank-based ensemble aggregation on the training set. "
            "Compare with the test-set table to assess overfitting."
        ),
        voting_models,
    )
    test_average_section = _build_average_metrics_section(
        "Model Results on Test Set (Average)",
        test_metrics_df,
        ranking_method_metrics_df,
        config.scoring_primary,
        highlighted_models,
        (
            "rank_average and rank_median rows show the performance of rank-based ensemble aggregation. "
            "These methods are robust to probability calibration differences across model families."
        ),
        voting_models,
    )
    sections = _build_evaluation_sections(
        selection,
        _model_selection_strategy(config, selection, cv_selection_table),
        cv_selection_table,
        sorted_geo_train_metrics_df,
        sorted_geo_test_metrics_df,
        train_average_section,
        test_average_section,
        classification_links,
    )
    if oof_confidence_metrics_df is not None and not oof_confidence_metrics_df.empty:
        sections.append(
            {
                "title": "OOF Confidence Performance (Descriptive)",
                "text": (
                    "These out-of-fold rows fitted the per-class precision guard, so this section "
                    "describes the frozen evidence but is not an independent validation result."
                ),
                "table": oof_confidence_metrics_df,
            }
        )
    if confidence_metrics_df is not None and not confidence_metrics_df.empty:
        confidence_text = (
            "Final confidence is the lower of parcel rank consensus and the frozen OOF precision level "
            "for the predicted class. Soft voting still selects the class; ranks only measure member support."
        )
        sections.append(
            {
                "title": "Holdout Validation: Rank Consensus with Class Reliability Guard",
                "text": confidence_text,
                "table": confidence_metrics_df,
                "links": _report_links(config.evaluate_dir / "confidence"),
            }
        )
    if confidence_by_class_df is not None and not confidence_by_class_df.empty:
        sections.append(
            {
                "title": "Holdout Confidence by Predicted Class",
                "table": confidence_by_class_df,
            }
        )
    sections.extend(
        _build_supplementary_sections(
            _build_ranking_links(config.evaluate_dir),
            images,
            interpretability_df,
            interpretability_images,
            interpretability_links,
        )
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
    confidence_columns = [
        config.prediction_confidence_column,
        config.prediction_confidence_level_column,
        "prediction_confidence_valid",
        "prediction_confidence_reason",
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
        "prediction_borda_margin",
        "prediction_top1_agreement",
        "prediction_top2_agreement",
        "prediction_top3_agreement",
        "prediction_runner_up_class",
        "prediction_runner_up_borda",
        "prediction_rank_models_used",
        "prediction_rank_confidence_valid",
        "prediction_rank_confidence_reason",
        config.prediction_review_column,
    ]
    preview_columns.extend(column for column in confidence_columns if column in final_df.columns)
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
                    "confidence_method": (
                        "within-model ranks" if config.prediction_confidence_level_column in final_df else "maximum probability"
                    ),
                    "confidence_level_counts": (
                        final_df[config.prediction_confidence_level_column].value_counts().to_dict()
                        if config.prediction_confidence_level_column in final_df
                        else {}
                    ),
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
    links = [
        {"label": "01 Check", "path": config.check_dir / "report.html"},
        {"label": "02 Prepare", "path": config.prepare_dir / "report.html"},
        {"label": "03 Train", "path": config.train_dir / "report.html"},
        {"label": "04 Evaluate", "path": config.evaluate_dir / "report.html"},
        {"label": "05 Predict", "path": config.predict_dir / "report.html"},
    ]
    final_dashboard_path = config.final_dashboard_dir / "report.html"
    if final_dashboard_path.exists():
        links.insert(0, {"label": "Final Dashboard", "path": final_dashboard_path})
    write_html_report(
        config.project_dir / "report_index.html",
        "ML Classification Pipeline Report Index",
        "Open any step report below to inspect saved outputs, tables, and plots.",
        sections=[
            {
                "title": "Step Reports",
                "links": links,
            }
        ],
    )
