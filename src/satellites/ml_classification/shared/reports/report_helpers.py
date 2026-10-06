"""CV table formatting and row styles reused by training and evaluation reports."""

from __future__ import annotations

from typing import Any

import pandas as pd

_SPECIAL_MODEL_DISPLAY_NAMES = {"soft_voting": "Voting", "rank_average": "Rank Average", "rank_median": "Rank Median"}


_ENSEMBLE_DISPLAY_NAMES = frozenset(_SPECIAL_MODEL_DISPLAY_NAMES.values())


def _prepare_cv_results_table(training_summary_df: pd.DataFrame | None, selected_models: list[str] | None = None) -> pd.DataFrame:
    """Build a concise CV ranking table with the best-performing model first."""
    if training_summary_df is None or training_summary_df.empty:
        return pd.DataFrame()

    required_columns = {"model", "best_cv_score", "cv_score_std", "cv_ranking_metric"}
    if not required_columns.issubset(training_summary_df.columns):
        return pd.DataFrame()

    ranked_df = training_summary_df.copy()
    ranked_df = ranked_df.sort_values(["cv_ranking_metric", "best_cv_score"], ascending=[False, False]).reset_index(drop=True)
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
        ranked_df["selected_for_strategy"] = (
            ranked_df["model"].astype(str).map(lambda model_name: "Yes" if model_name in selected_model_set else "No")
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


def _cv_ranking_formula(scoring_primary: str, ranking_method: str) -> str:
    """Return a human-readable definition of the configured selection score."""
    if ranking_method == "score_minus_std":
        return f"mean CV {scoring_primary} - CV standard deviation"
    return f"mean CV {scoring_primary}"


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
