"""Assemble the evaluate step HTML report from its persisted and computed results."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ml_classification.shared.reports.report_helpers import (
    _SPECIAL_MODEL_DISPLAY_NAMES,
    _cv_ranking_formula,
    _prepare_cv_results_table,
    _voting_member_row_styles,
)
from ml_classification.shared.reports.report_html import write_html_report
from shared.io import read_data

_AVERAGE_SPECIAL_MODELS = ("soft_voting", "rank_average", "rank_median")


_ANALYTICAL_SPECIAL_MODELS = ("Voting",)


_RANKING_METHODS = ("rank_average", "rank_median")


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

# Keep this helper focused on a single transformation so the reporting pipeline stays easy to follow.



def _sort_model_rows_only(metrics_df: pd.DataFrame, scoring_primary: str, special_models: tuple[str, ...]) -> pd.DataFrame:
    """Sort base models by score and retain the requested special-model order."""
    if "model" not in metrics_df.columns:
        return metrics_df.copy()

    ordered_df = metrics_df.copy().reset_index(drop=True)
    ordered_df["_original_order"] = range(len(ordered_df))
    base_df = ordered_df[~ordered_df["model"].isin(special_models)].copy()
    special_df = ordered_df[ordered_df["model"].isin(special_models)].copy()

    if scoring_primary in base_df.columns:
        base_df = base_df.sort_values(by=scoring_primary, ascending=False)

    # Keep aggregate rows in their explicit display order after score-sorted base models.
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
    # Scale only the display copy; source metrics remain proportions for computation.
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


def _sort_analytical_by_model_order(metrics_df: pd.DataFrame, model_order: list[str], scoring_primary: str) -> pd.DataFrame:
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
    analytical_df: pd.DataFrame | None, average_df: pd.DataFrame, scoring_primary: str
) -> pd.DataFrame:
    """Prepare an analytical table, falling back to the average metrics table."""
    if analytical_df is None:
        return _prepare_average_metrics(average_df, scoring_primary)
    return _sort_analytical_by_model_order(analytical_df, _build_analytical_model_order(average_df), scoring_primary)


def _ranking_rows(ranking_metrics_df: pd.DataFrame | None, columns: pd.Index) -> pd.DataFrame:
    """Convert supported ranking-method metrics to model rows with a target schema."""
    if ranking_metrics_df is None or ranking_metrics_df.empty or "method" not in ranking_metrics_df.columns:
        return pd.DataFrame(columns=columns)

    rank_rows = ranking_metrics_df[ranking_metrics_df["method"].isin(_RANKING_METHODS)].copy()
    rank_rows = rank_rows.rename(columns={"method": "model"})
    rank_rows = rank_rows.drop(columns=["n_models_used", "is_preferred"], errors="ignore")
    # Align rank-method rows to the model table schema, leaving unavailable metrics missing.
    return rank_rows.reindex(columns=columns)


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
            # Summarize the contract here without embedding the potentially large per-class lookup.
            confidence["class_reliability"] = {key: value for key, value in reliability.items() if key != "classes"}
        details["confidence"] = confidence
    return details


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
    return [{"label": path.name, "path": path} for path in sorted(directory.glob("*.csv")) if path.name not in excluded_names]


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
    sections = [{"title": "Model Selection Strategy", "kv": model_selection_strategy}]
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
    sorted_geo_train_metrics_df = _prepare_analytical_metrics(geo_train_metrics_df, train_metrics_df, config.scoring_primary)
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
    # OOF evidence fitted the guard, so display it separately from independent holdout diagnostics.
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
        sections.append({"title": "Holdout Confidence by Predicted Class", "table": confidence_by_class_df})
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
