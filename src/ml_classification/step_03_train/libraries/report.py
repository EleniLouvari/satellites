"""Assemble the train step HTML report from its persisted and computed results."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ml_classification.shared.reports.report_helpers import (
    _cv_ranking_formula,
    _prepare_cv_results_table,
    _voting_member_row_styles,
)
from ml_classification.shared.reports.report_html import write_html_report


def _supports_probability(value: Any) -> bool:
    """Normalize boolean-like values loaded directly or through CSV."""
    # CSV strings such as "False" must not inherit Python's truthiness for nonempty strings.
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def _configured_strategy_models(config, training_summary_df: pd.DataFrame) -> list[str]:
    """Infer the training-stage strategy candidates using the production CV ordering."""
    if training_summary_df.empty or "model" not in training_summary_df.columns:
        return []

    # Mirror the production CV ordering so highlighted candidates match the configured strategy.
    ranked_df = training_summary_df.sort_values(["cv_ranking_metric", "best_cv_score"], ascending=[False, False])
    # Only probability-capable candidates can participate in the inferred strategy.
    if "supports_predict_proba" in ranked_df.columns:
        ranked_df = ranked_df[ranked_df["supports_predict_proba"].map(_supports_probability)]
    ranked_models = ranked_df["model"].astype(str).tolist()
    top_voting_models = getattr(config, "top_voting_models", None)
    if top_voting_models is not None:
        ranked_models = ranked_models[: int(top_voting_models)]
    if getattr(config, "selection_type", "single_model") != "soft_voting":
        ranked_models = ranked_models[:1]
    return ranked_models


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
    # These are expected members from training results; evaluation later persists the frozen selection.
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
