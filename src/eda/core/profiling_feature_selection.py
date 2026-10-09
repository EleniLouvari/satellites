"""Feature selection proposal helpers for EDA profiling."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from .config import EDAConfig
from .profiling_constants import _EDA_DIAGNOSTIC_COLUMNS
from .profiling_helpers import _categorical_columns, _round, _safe_percent, _target_task
from .profiling_statistics import _association_strength

# Keep this helper focused on a single transformation so the reporting pipeline stays easy to follow.



def _feature_selection_note(strength: str, qvalue: float, alpha: float) -> str:
    """Summarize univariate screening evidence without presenting it as a final model decision."""
    # Require both a meaningful effect and FDR support before recommending a priority candidate.
    significant = np.isfinite(qvalue) and qvalue <= alpha
    if strength in {"strong", "large", "moderate"} and significant:
        return "priority candidate; validate with cross-validation and redundancy checks"
    if strength in {"strong", "large", "moderate"}:
        return "meaningful effect; statistical evidence is limited"
    if significant:
        return "statistically detectable but weak univariate effect"
    return "limited univariate evidence"

def _build_feature_target_associations(
    df: pd.DataFrame,
    config: EDAConfig,
    categorical_target_tests: pd.DataFrame,
    numeric_target_tests: pd.DataFrame,
    numeric_target_correlations: pd.DataFrame,
    categorical_numeric_target_tests: pd.DataFrame,
) -> pd.DataFrame:
    """Create one comparable target-association screening table for ML feature review."""
    columns = [
        "rank_within_method",
        "feature",
        "feature_type",
        "target_task",
        "method",
        "effect_size",
        "direction",
        "association_strength",
        "pvalue",
        "qvalue",
        "significant_fdr",
        "rows_used",
        "missing_percent",
        "selection_note",
    ]
    task = _target_task(df, config)
    if task is None:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    if task == "classification":
        for _, row in numeric_target_tests.iterrows():
            rows.append(
                {
                    "feature": row["column"],
                    "feature_type": "numeric",
                    "method": "kruskal_epsilon_squared",
                    "effect_size": row["epsilon_squared"],
                    "direction": np.nan,
                    "pvalue": row["kruskal_pvalue"],
                    "qvalue": row["qvalue"],
                    "rows_used": row["rows_used"],
                }
            )
        for _, row in categorical_target_tests.iterrows():
            rows.append(
                {
                    "feature": row["column"],
                    "feature_type": "categorical",
                    "method": "bias_corrected_cramers_v",
                    "effect_size": row["cramers_v_bias_corrected"],
                    "direction": np.nan,
                    "pvalue": row["pvalue"],
                    "qvalue": row["qvalue"],
                    "rows_used": row["rows_used"],
                }
            )
    else:
        for _, row in numeric_target_correlations.iterrows():
            rows.append(
                {
                    "feature": row["column"],
                    "feature_type": "numeric",
                    "method": "absolute_spearman_correlation",
                    "effect_size": row["abs_spearman_correlation"],
                    "direction": row["spearman_correlation"],
                    "pvalue": row["spearman_pvalue"],
                    "qvalue": row["qvalue"],
                    "rows_used": row["rows_used"],
                }
            )
        for _, row in categorical_numeric_target_tests.iterrows():
            rows.append(
                {
                    "feature": row["column"],
                    "feature_type": "categorical",
                    "method": "kruskal_epsilon_squared",
                    "effect_size": row["epsilon_squared"],
                    "direction": np.nan,
                    "pvalue": row["kruskal_pvalue"],
                    "qvalue": row["qvalue"],
                    "rows_used": row["rows_used"],
                }
            )
    if not rows:
        return pd.DataFrame(columns=columns)

    result = pd.DataFrame(rows)
    result["target_task"] = task
    result["effect_size"] = pd.to_numeric(result["effect_size"], errors="coerce")
    result["pvalue"] = pd.to_numeric(result["pvalue"], errors="coerce")
    result["qvalue"] = pd.to_numeric(result["qvalue"], errors="coerce")
    result["association_strength"] = [
        _association_strength(method, effect) for method, effect in zip(result["method"], result["effect_size"])
    ]
    result["significant_fdr"] = result["qvalue"].le(config.feature_selection_alpha).fillna(False)
    result["missing_percent"] = result["feature"].map(lambda feature: _safe_percent(df[feature].isna().sum(), len(df)))
    result["selection_note"] = [
        _feature_selection_note(strength, qvalue, config.feature_selection_alpha)
        for strength, qvalue in zip(result["association_strength"], result["qvalue"])
    ]
    # Rank within each method because its effect-size scale is not interchangeable with the others.
    result["rank_within_method"] = result.groupby("method")["effect_size"].rank(method="min", ascending=False).astype(int)
    result = result.sort_values(["method", "rank_within_method", "qvalue"], na_position="last").reset_index(drop=True)
    return result[columns]

def _build_feature_selection_criteria(config: EDAConfig) -> pd.DataFrame:
    """Describe the ordered, auditable screening criteria used for the proposal."""
    rows = [
        {
            "criterion_order": 1,
            "criterion": "hard exclusions and leakage review",
            "automated_rule": "Exclude target, ID, geometry, EDA diagnostics, and configured exclusions.",
            "decision_guidance": "Manually add post-outcome or otherwise leaky columns to feature_selection_exclude_columns.",
        },
        {
            "criterion_order": 2,
            "criterion": "data quality and availability",
            "automated_rule": (
                f"Exclude constant, quasi-constant, or >= {config.high_missing_percent_threshold:g}% missing columns."
            ),
            "decision_guidance": (
                "Review rare-category-heavy columns and whether missingness will also be available at prediction time."
            ),
        },
        {
            "criterion_order": 3,
            "criterion": "target association",
            "automated_rule": (
                f"Prioritize moderate/large method-specific effect sizes with Benjamini-Hochberg q <= "
                f"{config.feature_selection_alpha:g}."
            ),
            "decision_guidance": (
                "Treat weak or statistically uncertain univariate evidence as review items, not automatic proof of no value."
            ),
        },
        {
            "criterion_order": 4,
            "criterion": "redundancy",
            "automated_rule": (
                f"Keep one priority feature when absolute Spearman correlation or Cramer's V is >= "
                f"{config.feature_selection_redundancy_threshold:g}."
            ),
            "decision_guidance": "Prefer the candidate with the better within-method rank and lower missingness.",
        },
        {
            "criterion_order": 5,
            "criterion": "parsimonious initial shortlist",
            "automated_rule": (
                f"Retain at most {config.feature_selection_max_features} candidates."
                if config.feature_selection_max_features is not None
                else "No automatic feature-count limit is applied."
            ),
            "decision_guidance": (
                "Treat lower-ranked supported features as expansion candidates rather than permanently rejected variables."
            ),
        },
        {
            "criterion_order": 6,
            "criterion": "train-only model validation",
            "automated_rule": "Not automated by EDA.",
            "decision_guidance": (
                "After splitting, compare cross-validated performance and permutation importance using preprocessing fitted only "
                "on training data."
            ),
        },
    ]
    return pd.DataFrame(rows)

def _build_feature_selection_proposal(
    df: pd.DataFrame,
    config: EDAConfig,
    associations: pd.DataFrame,
    quality_flags: pd.DataFrame,
    spearman_matrix: pd.DataFrame,
    categorical_associations: pd.DataFrame,
) -> pd.DataFrame:
    """Turn target evidence, data quality, and redundancy diagnostics into an initial ML feature proposal."""
    columns = [
        "selection_order",
        "feature",
        "feature_type",
        "proposed_action",
        "primary_reason",
        "association_method",
        "effect_size",
        "association_strength",
        "qvalue",
        "missing_percent",
        "redundant_with",
        "redundancy_measure",
        "passes_hard_exclusions",
        "passes_data_quality",
        "passes_target_evidence",
        "passes_redundancy",
        "passes_shortlist_limit",
    ]
    if associations.empty:
        return pd.DataFrame(columns=columns)

    blocked_issues = {"constant", "quasi_constant", "high_missing"}
    flagged_issues = (
        quality_flags.groupby("column")["issue_type"].agg(set).to_dict() if not quality_flags.empty else {}
    )
    excluded = {"geometry", config.target_column, config.id_column, *config.feature_selection_exclude_columns}
    proposal_rows = []
    for _, association in associations.iterrows():
        feature = str(association["feature"])
        issues = flagged_issues.get(feature, set())
        # Leakage exclusions take precedence over apparently strong target associations.
        hard_exclusion = feature in excluded or feature in _EDA_DIAGNOSTIC_COLUMNS or feature.startswith("eda_")
        passes_quality = not bool(issues & blocked_issues)
        selection_note = str(association["selection_note"])
        passes_target = selection_note.startswith("priority candidate")

        if hard_exclusion:
            action = "exclude"
            reason = "target, identifier, geometry, diagnostic, or configured leakage exclusion"
        elif not passes_quality:
            action = "exclude"
            reason = "data-quality exclusion: " + ", ".join(sorted(issues & blocked_issues))
        elif passes_target:
            action = "use"
            reason = "priority target association after FDR correction"
        elif selection_note == "limited univariate evidence":
            action = "exclude"
            reason = "limited univariate target evidence; retain only if domain or model validation supports it"
        else:
            action = "review"
            reason = selection_note

        proposal_rows.append(
            {
                "feature": feature,
                "feature_type": association["feature_type"],
                "proposed_action": action,
                "primary_reason": reason,
                "association_method": association["method"],
                "effect_size": association["effect_size"],
                "association_strength": association["association_strength"],
                "qvalue": association["qvalue"],
                "missing_percent": association["missing_percent"],
                "rank_within_method": association["rank_within_method"],
                "redundant_with": "",
                "redundancy_measure": np.nan,
                "passes_hard_exclusions": not hard_exclusion,
                "passes_data_quality": passes_quality,
                "passes_target_evidence": passes_target,
                "passes_redundancy": True,
                "passes_shortlist_limit": True,
            }
        )

    proposal = pd.DataFrame(proposal_rows)
    proposal = proposal.sort_values(
        ["rank_within_method", "missing_percent", "feature"], na_position="last"
    ).reset_index(drop=True)
    # Greedily retain the best-ranked representative before checking later features for redundancy.
    kept: list[str] = []
    for row_index in proposal.index[proposal["proposed_action"] == "use"]:
        feature = str(proposal.at[row_index, "feature"])
        feature_type = str(proposal.at[row_index, "feature_type"])
        redundant_feature, redundancy = _find_redundant_feature(
            feature,
            feature_type,
            kept,
            spearman_matrix,
            categorical_associations,
            config.feature_selection_redundancy_threshold,
        )
        if redundant_feature:
            proposal.at[row_index, "proposed_action"] = "review"
            proposal.at[row_index, "primary_reason"] = f"redundant with retained feature {redundant_feature}"
            proposal.at[row_index, "redundant_with"] = redundant_feature
            proposal.at[row_index, "redundancy_measure"] = _round(redundancy)
            proposal.at[row_index, "passes_redundancy"] = False
        else:
            kept.append(feature)

    # Move supported overflow candidates to review so the shortlist cap does not permanently reject them.
    if config.feature_selection_max_features is not None:
        use_indexes = proposal.index[proposal["proposed_action"] == "use"].tolist()
        for row_index in use_indexes[config.feature_selection_max_features :]:
            proposal.at[row_index, "proposed_action"] = "review"
            proposal.at[row_index, "primary_reason"] = (
                f"outside the configured top-{config.feature_selection_max_features} initial shortlist"
            )
            proposal.at[row_index, "passes_shortlist_limit"] = False

    action_order = proposal["proposed_action"].map({"use": 0, "review": 1, "exclude": 2})
    proposal = proposal.assign(_action_order=action_order).sort_values(
        ["_action_order", "rank_within_method", "missing_percent", "feature"], na_position="last"
    )
    proposal = proposal.drop(columns=["_action_order", "rank_within_method"]).reset_index(drop=True)
    proposal.insert(0, "selection_order", np.arange(1, len(proposal) + 1))
    return proposal[columns]

def _find_redundant_feature(
    feature: str,
    feature_type: str,
    retained_features: list[str],
    spearman_matrix: pd.DataFrame,
    categorical_associations: pd.DataFrame,
    threshold: float,
) -> tuple[str | None, float]:
    """Find the first already-retained same-type feature exceeding the configured redundancy threshold."""
    for retained in retained_features:
        if feature_type == "numeric" and feature in spearman_matrix.index and retained in spearman_matrix.columns:
            association = abs(float(spearman_matrix.loc[feature, retained]))
        elif feature_type == "categorical":
            association = _categorical_pair_association(feature, retained, categorical_associations)
        else:
            continue
        if np.isfinite(association) and association >= threshold:
            return retained, association
    return None, np.nan

def _categorical_pair_association(left: str, right: str, associations: pd.DataFrame) -> float:
    """Look up Cramer's V for an unordered categorical feature pair."""
    if associations.empty:
        return np.nan
    # Pairwise association is symmetric even when the source table stores only one orientation.
    pair = associations[
        ((associations["feature_1"] == left) & (associations["feature_2"] == right))
        | ((associations["feature_1"] == right) & (associations["feature_2"] == left))
    ]
    return float(pair.iloc[0]["cramers_v"]) if not pair.empty else np.nan

def _build_feature_selection_message(proposal: pd.DataFrame, criteria: pd.DataFrame) -> str:
    """Create the terminal/report summary for the automated feature proposal."""
    criteria_order = " -> ".join(criteria.sort_values("criterion_order")["criterion"].tolist())
    if proposal.empty:
        return "No feature proposal was created. Configure a valid target column. Criteria: " + criteria_order + "."
    proposed = proposal.loc[proposal["proposed_action"] == "use", "feature"].tolist()
    review = proposal.loc[proposal["proposed_action"] == "review", "feature"].tolist()
    excluded_count = int((proposal["proposed_action"] == "exclude").sum())
    proposed_text = ", ".join(proposed) if proposed else "none"
    review_text = ", ".join(review[:10]) if review else "none"
    if len(review) > 10:
        review_text += f", ... {len(review) - 10} more (see feature_selection_proposal.csv)"
    return (
        f"Proposed initial ML features ({len(proposed)}): {proposed_text}. Review separately ({len(review)}): {review_text}. "
        f"Automatically excluded: {excluded_count}. Criteria order: {criteria_order}. "
        "This is an EDA shortlist; confirm it with train-only cross-validation and permutation importance."
    )

def _build_plot_feature_order(
    df: pd.DataFrame,
    config: EDAConfig,
    proposal: pd.DataFrame,
    column_profile: pd.DataFrame,
    quality_flags: pd.DataFrame,
) -> dict[str, list[str]]:
    """Order plot candidates by ML screening evidence, then by data quality and availability."""
    excluded = {"geometry", config.target_column, config.id_column, *_EDA_DIAGNOSTIC_COLUMNS}
    candidates = [column for column in df.columns if column not in excluded and not column.startswith("eda_")]
    proposal_order = _proposal_feature_order(proposal, candidates)

    profile = column_profile.set_index("column") if not column_profile.empty else pd.DataFrame()
    issues = quality_flags.groupby("column")["issue_type"].agg(set).to_dict() if not quality_flags.empty else {}
    remaining = _sorted_remaining_plot_candidates(candidates, proposal_order, issues, profile)
    # Use the proposal first, then append features lacking target evidence using quality-based ordering.
    ordered = proposal_order + remaining
    numeric = [column for column in ordered if ptypes.is_numeric_dtype(df[column])]
    categorical_candidates = set(_categorical_columns(df))
    categorical = [column for column in ordered if column in categorical_candidates]
    return {"all": ordered, "numeric": numeric, "categorical": categorical}

def _proposal_feature_order(proposal: pd.DataFrame, candidates: list[str]) -> list[str]:
    """Return proposal-ranked feature order restricted to available candidates."""
    if proposal.empty:
        return []
    ranked = proposal.sort_values("selection_order")
    return [feature for feature in ranked["feature"].tolist() if feature in candidates]

def _sorted_remaining_plot_candidates(
    candidates: list[str], proposal_order: list[str], issues: dict[str, set[str]], profile: pd.DataFrame
) -> list[str]:
    """Sort non-proposal candidates by issue severity and missingness."""
    blocking_issues = {"constant", "quasi_constant", "high_missing", "id_like_or_nearly_unique"}
    original_position = {column: index for index, column in enumerate(candidates)}

    def fallback_key(column: str) -> tuple[int, int, float, int]:
        column_issues = issues.get(column, set())
        blocked = int(bool(column_issues & blocking_issues))
        flagged = int(bool(column_issues))
        missing_percent = float(profile.at[column, "missing_percent"]) if column in profile.index else 100.0
        # Original column position breaks ties without introducing arbitrary alphabetical reordering.
        return blocked, flagged, missing_percent, original_position[column]

    return sorted((column for column in candidates if column not in proposal_order), key=fallback_key)
