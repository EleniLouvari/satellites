"""Data-quality checks for EDA profiling."""

from __future__ import annotations

import pandas as pd

from .config import EDAConfig

# Keep this helper focused on a single transformation so the reporting pipeline stays easy to follow.



def _build_data_quality_flags(df: pd.DataFrame, column_profile: pd.DataFrame, config: EDAConfig) -> pd.DataFrame:
    """Flag columns that may need special handling before analysis or modeling."""
    rows = []
    total_rows = len(df)
    for _, profile in column_profile.iterrows():
        context = _quality_flag_context(df, profile, total_rows)
        if context is None:
            continue
        # Flags are cumulative: a column can have multiple independent quality issues.
        rows.extend(_base_quality_flags(context, config))
        rows.extend(_distribution_quality_flags(context, config, total_rows))
    # Retain the schema even when no issues are found so downstream report code can still select columns.
    return pd.DataFrame(rows, columns=["column", "issue_type", "severity", "details"])

def _quality_flag(column: str, issue_type: str, severity: str, details: str) -> dict[str, str]:
    """Build one data-quality flag row."""
    return {"column": column, "issue_type": issue_type, "severity": severity, "details": details}

def _quality_flag_context(df: pd.DataFrame, profile: pd.Series, total_rows: int) -> dict[str, object] | None:
    """Collect the per-column context used by quality-flag checks."""
    column = profile["column"]
    if column not in df.columns:
        return None
    series = df[column]
    non_null = int(profile["non_null_count"])
    value_counts = (
        series.dropna().astype(str).value_counts() if column != "geometry" and non_null else pd.Series(dtype="int64")
    )
    return {
        "column": column,
        "series": series,
        "non_null": non_null,
        "missing_percent": float(profile["missing_percent"]),
        "unique_percent": float(profile["unique_percent"]),
        "logical_type": str(profile["logical_type"]),
        "is_constant": bool(profile["is_constant"]),
        "value_counts": value_counts,
        "total_rows": total_rows,
    }

def _base_quality_flags(context: dict[str, object], config: EDAConfig) -> list[dict[str, str]]:
    """Create flags based on missingness, uniqueness, and constant columns."""
    column = str(context["column"])
    non_null = int(context["non_null"])
    missing_percent = float(context["missing_percent"])
    unique_percent = float(context["unique_percent"])
    flags: list[dict[str, str]] = []
    if bool(context["is_constant"]):
        flags.append(_quality_flag(column, "constant", "high", "Column has one or zero distinct non-missing values."))
    if non_null and missing_percent >= config.high_missing_percent_threshold:
        flags.append(_quality_flag(column, "high_missing", "medium", f"Missing in {missing_percent:.2f}% of rows."))
    if non_null and unique_percent >= config.categorical_association_max_unique_ratio * 100:
        flags.append(_quality_flag(column, "id_like_or_nearly_unique", "medium", f"Unique in {unique_percent:.2f}% of non-missing rows."))
    return flags

def _distribution_quality_flags(context: dict[str, object], config: EDAConfig, total_rows: int) -> list[dict[str, str]]:
    """Create frequency-distribution flags such as quasi-constant and rare categories."""
    value_counts = context["value_counts"]
    if value_counts.empty:
        return []
    column = str(context["column"])
    non_null = int(context["non_null"])
    logical_type = str(context["logical_type"])
    flags: list[dict[str, str]] = []
    # Measure dominance among observed values; missingness has its own quality check.
    top_ratio = float(value_counts.iloc[0] / non_null) if non_null else 0.0
    # Avoid duplicating the constant-column flag with a quasi-constant warning.
    if not bool(context["is_constant"]) and top_ratio >= config.quasi_constant_threshold:
        flags.append(
            _quality_flag(
                column,
                "quasi_constant",
                "medium",
                f"Most common value covers {100 * top_ratio:.2f}% of non-missing rows.",
            )
        )
    if logical_type not in {"categorical", "boolean"} or len(value_counts) < 5:
        return flags
    rare_threshold = max(1, int(0.01 * max(non_null, total_rows)))
    # This is the share of distinct categories that are rare, not the share of rows in rare categories.
    rare_ratio = float((value_counts <= rare_threshold).sum() / len(value_counts))
    if rare_ratio >= 0.5:
        flags.append(
            _quality_flag(
                column,
                "rare_category_heavy",
                "low",
                f"{100 * rare_ratio:.2f}% of categories occur at most {rare_threshold} time(s).",
            )
        )
    return flags
