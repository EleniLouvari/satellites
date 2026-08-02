"""Statistical profiling functions for dataframe and geodataframe EDA."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as ptypes
from scipy import stats

from ...outliers import OutlierAnalysis

from .config import EDAConfig


def build_eda_artifacts(df: pd.DataFrame, config: EDAConfig) -> dict[str, Any]:
    """Build all tabular and JSON-ready EDA artifacts for a dataframe."""
    dataset = df.copy()
    if dataset.empty:
        raise ValueError("EDA requires a dataframe with at least one row.")

    geospatial = _build_geospatial_summary(dataset, config)
    numeric_columns = _numeric_columns(dataset)
    categorical_columns = _categorical_columns(dataset)
    datetime_columns = _datetime_columns(dataset)

    summary = _build_dataset_summary(
        dataset,
        config,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        datetime_columns=datetime_columns,
        geospatial=geospatial,
    )
    column_profile = _build_column_profile(dataset)
    data_quality_flags = _build_data_quality_flags(dataset, column_profile, config)
    numeric_summary = _build_numeric_summary(dataset, numeric_columns, config)
    robust_numeric_summary = _build_robust_numeric_summary(dataset, numeric_columns)
    categorical_summary = _build_categorical_summary(dataset, categorical_columns, config)
    datetime_summary = _build_datetime_summary(dataset, datetime_columns)
    normality_tests = _build_normality_tests(dataset, numeric_columns, config)
    outlier_summary = _build_outlier_summary(dataset, numeric_columns, config)
    correlation_matrices = _build_correlation_matrices(dataset, numeric_columns)
    strong_pairs = _build_strong_correlation_pairs(correlation_matrices["pearson"], config)
    multicollinearity = _build_multicollinearity_summary(dataset, numeric_columns, config)
    missingness_summary = _build_missingness_summary(dataset)
    categorical_associations = _build_categorical_associations(dataset, categorical_columns, config)
    categorical_target_tests = _build_categorical_target_tests(dataset, categorical_columns, config)
    numeric_target_tests = _build_numeric_target_tests(dataset, numeric_columns, config)
    numeric_target_correlations = _build_numeric_target_correlations(dataset, numeric_columns, config)
    missingness_target_tests = _build_missingness_target_tests(dataset, config)
    pca_summary = _build_pca_summary(dataset, numeric_columns, config)
    multivariate_outliers = _build_multivariate_outliers(dataset, numeric_columns, config)
    target_summary = _build_target_summary(dataset, config)

    return {
        "summary": summary,
        "column_profile": column_profile,
        "data_quality_flags": data_quality_flags,
        "numeric_summary": numeric_summary,
        "robust_numeric_summary": robust_numeric_summary,
        "categorical_summary": categorical_summary,
        "datetime_summary": datetime_summary,
        "normality_tests": normality_tests,
        "outlier_summary": outlier_summary,
        "correlation_matrices": correlation_matrices,
        "strong_correlation_pairs": strong_pairs,
        "multicollinearity": multicollinearity,
        "missingness_summary": missingness_summary,
        "categorical_associations": categorical_associations,
        "categorical_target_tests": categorical_target_tests,
        "numeric_target_tests": numeric_target_tests,
        "numeric_target_correlations": numeric_target_correlations,
        "missingness_target_tests": missingness_target_tests,
        "pca_summary": pca_summary,
        "multivariate_outliers": multivariate_outliers,
        "target_summary": target_summary,
        "geospatial_summary": geospatial,
    }


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric non-geometry columns."""
    return [
        column
        for column in df.columns
        if column != "geometry" and ptypes.is_numeric_dtype(df[column])
    ]


def _categorical_columns(df: pd.DataFrame) -> list[str]:
    """Return object-like categorical columns."""
    return [
        column
        for column in df.columns
        if column != "geometry"
        and not ptypes.is_numeric_dtype(df[column])
        and not ptypes.is_datetime64_any_dtype(df[column])
    ]


def _datetime_columns(df: pd.DataFrame) -> list[str]:
    """Return datetime columns."""
    return [
        column
        for column in df.columns
        if column != "geometry" and ptypes.is_datetime64_any_dtype(df[column])
    ]


def _build_dataset_summary(
    df: pd.DataFrame,
    config: EDAConfig,
    numeric_columns: list[str],
    categorical_columns: list[str],
    datetime_columns: list[str],
    geospatial: pd.DataFrame,
) -> dict[str, Any]:
    """Create a JSON-safe high-level dataset summary."""
    duplicate_rows = int(df.duplicated().sum()) if "geometry" not in df.columns else int(df.drop(columns=["geometry"]).duplicated().sum())
    target_status = "not configured"
    if config.target_column:
        target_status = "present" if config.target_column in df.columns else "missing"
    return {
        "_schema": {"artifact": "eda_summary", "schema_version": config.schema_version},
        "config": asdict(config),
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "datetime_columns": datetime_columns,
        "geometry_column_present": bool("geometry" in df.columns),
        "geospatial_summary_present": bool(not geospatial.empty),
        "target_column": config.target_column,
        "target_status": target_status,
        "id_column": config.id_column,
        "duplicate_rows_excluding_geometry": duplicate_rows,
        "total_missing_values": int(df.isna().sum().sum()),
        "total_missing_percent": _safe_percent(df.isna().sum().sum(), df.size),
        "memory_usage_mb": round(float(df.memory_usage(deep=True).sum()) / 1_000_000, 4),
    }


def _build_column_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Profile every column in the input dataframe."""
    rows = []
    total_rows = len(df)
    for column in df.columns:
        series = df[column]
        non_null = int(series.notna().sum())
        unique = int(series.nunique(dropna=True)) if column != "geometry" else int(series.geom_type.nunique(dropna=True)) if hasattr(series, "geom_type") else 0
        rows.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "logical_type": _logical_type(series, column),
                "non_null_count": non_null,
                "missing_count": int(series.isna().sum()),
                "missing_percent": _safe_percent(series.isna().sum(), total_rows),
                "unique_count": unique,
                "unique_percent": _safe_percent(unique, max(non_null, 1)),
                "is_constant": bool(unique <= 1),
                "memory_usage_mb": round(float(series.memory_usage(deep=True)) / 1_000_000, 4),
            }
        )
    return pd.DataFrame(rows)


def _build_data_quality_flags(df: pd.DataFrame, column_profile: pd.DataFrame, config: EDAConfig) -> pd.DataFrame:
    """Flag columns that may need special handling before analysis or modeling."""
    rows = []
    total_rows = len(df)
    for _, profile in column_profile.iterrows():
        column = profile["column"]
        if column not in df.columns:
            continue
        series = df[column]
        non_null = int(profile["non_null_count"])
        missing_percent = float(profile["missing_percent"])
        unique_percent = float(profile["unique_percent"])
        logical_type = str(profile["logical_type"])

        if bool(profile["is_constant"]):
            rows.append(_quality_flag(column, "constant", "high", "Column has one or zero distinct non-missing values."))
        if non_null and missing_percent >= config.high_missing_percent_threshold:
            rows.append(_quality_flag(column, "high_missing", "medium", f"Missing in {missing_percent:.2f}% of rows."))
        if non_null and unique_percent >= config.categorical_association_max_unique_ratio * 100:
            rows.append(_quality_flag(column, "id_like_or_nearly_unique", "medium", f"Unique in {unique_percent:.2f}% of non-missing rows."))

        if column != "geometry" and non_null:
            value_counts = series.dropna().astype(str).value_counts()
            top_ratio = float(value_counts.iloc[0] / non_null) if len(value_counts) else 0.0
            if not bool(profile["is_constant"]) and top_ratio >= config.quasi_constant_threshold:
                rows.append(_quality_flag(column, "quasi_constant", "medium", f"Most common value covers {100 * top_ratio:.2f}% of non-missing rows."))
            if logical_type in {"categorical", "boolean"} and len(value_counts) >= 5:
                rare_threshold = max(1, int(0.01 * max(non_null, total_rows)))
                rare_ratio = float((value_counts <= rare_threshold).sum() / len(value_counts))
                if rare_ratio >= 0.5:
                    rows.append(_quality_flag(column, "rare_category_heavy", "low", f"{100 * rare_ratio:.2f}% of categories occur at most {rare_threshold} time(s)."))
    return pd.DataFrame(rows, columns=["column", "issue_type", "severity", "details"])


def _quality_flag(column: str, issue_type: str, severity: str, details: str) -> dict[str, str]:
    """Build one data-quality flag row."""
    return {"column": column, "issue_type": issue_type, "severity": severity, "details": details}

def _build_numeric_summary(df: pd.DataFrame, numeric_columns: list[str], config: EDAConfig) -> pd.DataFrame:
    """Calculate descriptive statistics for numeric columns."""
    rows = []
    for column in numeric_columns:
        values = pd.to_numeric(df[column], errors="coerce")
        clean = values.dropna()
        q1 = clean.quantile(0.25) if len(clean) else np.nan
        q3 = clean.quantile(0.75) if len(clean) else np.nan
        iqr = q3 - q1 if len(clean) else np.nan
        lower = q1 - config.iqr_multiplier * iqr if len(clean) else np.nan
        upper = q3 + config.iqr_multiplier * iqr if len(clean) else np.nan
        rows.append(
            {
                "column": column,
                "count": int(clean.count()),
                "missing_count": int(values.isna().sum()),
                "missing_percent": _safe_percent(values.isna().sum(), len(values)),
                "mean": _round(clean.mean()),
                "std": _round(clean.std()),
                "min": _round(clean.min()),
                "q1": _round(q1),
                "median": _round(clean.median()),
                "q3": _round(q3),
                "max": _round(clean.max()),
                "iqr": _round(iqr),
                "skew": _round(clean.skew()),
                "kurtosis": _round(clean.kurtosis()),
                "coefficient_of_variation": _round(clean.std() / clean.mean()) if len(clean) and clean.mean() else np.nan,
                "zeros_count": int((clean == 0).sum()),
                "negative_count": int((clean < 0).sum()),
                "iqr_lower_bound": _round(lower),
                "iqr_upper_bound": _round(upper),
                "iqr_outlier_count": int(((clean < lower) | (clean > upper)).sum()) if len(clean) else 0,
            }
        )
    return pd.DataFrame(rows)



def _build_robust_numeric_summary(df: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
    """Calculate robust univariate statistics for numeric columns."""
    rows = []
    for column in numeric_columns:
        clean = pd.to_numeric(df[column], errors="coerce").dropna()
        clean = clean[np.isfinite(clean)]
        if clean.empty:
            continue
        median = clean.median()
        mad = stats.median_abs_deviation(clean, scale="normal", nan_policy="omit")
        trimmed = stats.trim_mean(clean, 0.1) if len(clean) >= 10 else clean.mean()
        sem = stats.sem(clean) if len(clean) > 1 else np.nan
        ci_low, ci_high = (np.nan, np.nan)
        if len(clean) > 1 and pd.notna(sem):
            ci_low, ci_high = stats.t.interval(0.95, len(clean) - 1, loc=clean.mean(), scale=sem)
        rows.append(
            {
                "column": column,
                "count": int(len(clean)),
                "trimmed_mean_10pct": _round(trimmed),
                "median": _round(median),
                "mad_normalized": _round(mad),
                "range": _round(clean.max() - clean.min()),
                "percentile_1": _round(clean.quantile(0.01)),
                "percentile_5": _round(clean.quantile(0.05)),
                "percentile_95": _round(clean.quantile(0.95)),
                "percentile_99": _round(clean.quantile(0.99)),
                "mean_ci95_low": _round(ci_low),
                "mean_ci95_high": _round(ci_high),
            }
        )
    return pd.DataFrame(rows)

def _build_categorical_summary(df: pd.DataFrame, categorical_columns: list[str], config: EDAConfig) -> pd.DataFrame:
    """Summarize categorical columns and their most frequent values."""
    rows = []
    for column in categorical_columns:
        series = df[column].dropna().astype(str)
        value_counts = series.value_counts()
        top_value = value_counts.index[0] if len(value_counts) else None
        top_count = int(value_counts.iloc[0]) if len(value_counts) else 0
        rare_threshold = max(1, int(0.01 * len(series)))
        probabilities = value_counts / value_counts.sum() if value_counts.sum() else pd.Series(dtype=float)
        entropy = float(stats.entropy(probabilities)) if len(probabilities) else np.nan
        rows.append(
            {
                "column": column,
                "count": int(series.count()),
                "missing_count": int(df[column].isna().sum()),
                "missing_percent": _safe_percent(df[column].isna().sum(), len(df)),
                "unique_count": int(series.nunique()),
                "top_value": top_value,
                "top_count": top_count,
                "top_percent": _safe_percent(top_count, len(series)),
                "rare_value_count": int((value_counts <= rare_threshold).sum()),
                "entropy": _round(entropy),
                "top_values": dict(value_counts.head(config.max_categories)),
            }
        )
    return pd.DataFrame(rows)


def _build_datetime_summary(df: pd.DataFrame, datetime_columns: list[str]) -> pd.DataFrame:
    """Summarize datetime columns."""
    rows = []
    for column in datetime_columns:
        series = pd.to_datetime(df[column], errors="coerce")
        clean = series.dropna()
        rows.append(
            {
                "column": column,
                "count": int(clean.count()),
                "missing_count": int(series.isna().sum()),
                "missing_percent": _safe_percent(series.isna().sum(), len(series)),
                "min": clean.min(),
                "max": clean.max(),
                "range_days": _round((clean.max() - clean.min()).days) if len(clean) else np.nan,
                "unique_count": int(clean.nunique()),
            }
        )
    return pd.DataFrame(rows)


def _build_normality_tests(df: pd.DataFrame, numeric_columns: list[str], config: EDAConfig) -> pd.DataFrame:
    """Run common normality checks for numeric columns."""
    rows = []
    rng = np.random.default_rng(config.random_state)
    for column in numeric_columns:
        clean = pd.to_numeric(df[column], errors="coerce").dropna()
        clean = clean[np.isfinite(clean)]
        if len(clean) > config.normality_sample_size:
            clean = pd.Series(rng.choice(clean.to_numpy(), config.normality_sample_size, replace=False))
        row = {
            "column": column,
            "sample_size": int(len(clean)),
            "shapiro_statistic": np.nan,
            "shapiro_pvalue": np.nan,
            "dagostino_statistic": np.nan,
            "dagostino_pvalue": np.nan,
            "jarque_bera_statistic": np.nan,
            "jarque_bera_pvalue": np.nan,
            "anderson_statistic": np.nan,
            "normality_hint": "insufficient_data",
        }
        if len(clean) >= 3:
            shapiro = stats.shapiro(clean)
            row["shapiro_statistic"] = _round(shapiro.statistic)
            row["shapiro_pvalue"] = _round(shapiro.pvalue)
        if len(clean) >= 8:
            dagostino = stats.normaltest(clean)
            jb = stats.jarque_bera(clean)
            anderson = stats.anderson(clean, dist="norm")
            row.update(
                {
                    "dagostino_statistic": _round(dagostino.statistic),
                    "dagostino_pvalue": _round(dagostino.pvalue),
                    "jarque_bera_statistic": _round(jb.statistic),
                    "jarque_bera_pvalue": _round(jb.pvalue),
                    "anderson_statistic": _round(anderson.statistic),
                    "normality_hint": "likely_normal" if dagostino.pvalue >= 0.05 and jb.pvalue >= 0.05 else "likely_not_normal",
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _build_outlier_summary(df: pd.DataFrame, numeric_columns: list[str], config: EDAConfig) -> pd.DataFrame:
    """Build the EDA univariate outlier artifact.

    The calculation is delegated to :class:`OutlierAnalysis` so interactive
    analysis and generated EDA reports use identical IQR and z-score rules.
    """
    return _outlier_analysis(df, config).summarize_univariate(numeric_columns)


def _build_correlation_matrices(df: pd.DataFrame, numeric_columns: list[str]) -> dict[str, pd.DataFrame]:
    """Calculate Pearson and Spearman correlation matrices."""
    if len(numeric_columns) < 2:
        return {"pearson": pd.DataFrame(), "spearman": pd.DataFrame()}
    numeric_df = df[numeric_columns].apply(pd.to_numeric, errors="coerce")
    return {
        "pearson": numeric_df.corr(method="pearson"),
        "spearman": numeric_df.corr(method="spearman"),
    }


def _build_strong_correlation_pairs(correlation_matrix: pd.DataFrame, config: EDAConfig) -> pd.DataFrame:
    """Return pairs whose absolute Pearson correlation exceeds the threshold."""
    if correlation_matrix.empty:
        return pd.DataFrame(columns=["feature_1", "feature_2", "pearson_correlation", "abs_correlation"])
    pairs = []
    columns = list(correlation_matrix.columns)
    for idx, left in enumerate(columns):
        for right in columns[idx + 1 :]:
            corr = correlation_matrix.loc[left, right]
            if pd.notna(corr) and abs(corr) >= config.correlation_threshold:
                pairs.append(
                    {
                        "feature_1": left,
                        "feature_2": right,
                        "pearson_correlation": _round(corr),
                        "abs_correlation": _round(abs(corr)),
                    }
                )
    return pd.DataFrame(pairs).sort_values("abs_correlation", ascending=False) if pairs else pd.DataFrame(columns=["feature_1", "feature_2", "pearson_correlation", "abs_correlation"])



def _build_multicollinearity_summary(df: pd.DataFrame, numeric_columns: list[str], config: EDAConfig) -> pd.DataFrame:
    """Calculate variance inflation factors for numeric columns."""
    numeric_df = _prepared_numeric_matrix(df, numeric_columns, config)
    if numeric_df.shape[1] < 2 or numeric_df.shape[0] < 3:
        return pd.DataFrame(columns=["column", "vif", "tolerance", "r_squared", "interpretation"])
    rows = []
    columns = list(numeric_df.columns)
    values = numeric_df.to_numpy(dtype=float)
    for index, column in enumerate(columns):
        y = values[:, index]
        x = np.delete(values, index, axis=1)
        x = np.column_stack([np.ones(len(x)), x])
        try:
            beta, *_ = np.linalg.lstsq(x, y, rcond=None)
            fitted = x @ beta
            ss_total = float(np.sum((y - y.mean()) ** 2))
            ss_resid = float(np.sum((y - fitted) ** 2))
            r_squared = 1 - ss_resid / ss_total if ss_total else 1.0
            r_squared = min(max(r_squared, 0.0), 0.999999)
            tolerance = 1 - r_squared
            vif = 1 / tolerance if tolerance > 0 else np.inf
        except Exception:
            r_squared = np.nan
            tolerance = np.nan
            vif = np.nan
        rows.append(
            {
                "column": column,
                "vif": _round(vif),
                "tolerance": _round(tolerance),
                "r_squared": _round(r_squared),
                "interpretation": _vif_interpretation(vif),
            }
        )
    return pd.DataFrame(rows).sort_values("vif", ascending=False).reset_index(drop=True)


def _vif_interpretation(vif: float) -> str:
    """Return a compact VIF severity label."""
    if pd.isna(vif):
        return "not_available"
    if vif >= 10:
        return "high_multicollinearity"
    if vif >= 5:
        return "moderate_multicollinearity"
    return "low_multicollinearity"

def _build_missingness_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize missingness and columns that tend to be missing together."""
    missing = df.isna()
    rows = []
    total_rows = len(df)
    for column in df.columns:
        missing_count = int(missing[column].sum())
        if missing_count == 0:
            continue
        other_missing = missing.drop(columns=[column], errors="ignore")
        paired = []
        if not other_missing.empty:
            rates = other_missing[missing[column]].mean().sort_values(ascending=False)
            paired = [f"{idx}: {_round(100 * value, 2)}%" for idx, value in rates.head(5).items() if value > 0]
        rows.append(
            {
                "column": column,
                "missing_count": missing_count,
                "missing_percent": _safe_percent(missing_count, total_rows),
                "non_missing_count": int(total_rows - missing_count),
                "top_co_missing_columns": "; ".join(paired),
            }
        )
    return pd.DataFrame(rows)


def _build_categorical_associations(df: pd.DataFrame, categorical_columns: list[str], config: EDAConfig) -> pd.DataFrame:
    """Run chi-square tests and Cramer's V for categorical column pairs."""
    rows = []
    columns = [
        column
        for column in categorical_columns
        if column != config.id_column and _is_categorical_association_candidate(df[column], config)
    ]
    for left_idx, left in enumerate(columns):
        for right in columns[left_idx + 1 :]:
            pair_df = df[[left, right]].dropna().astype(str)
            if len(pair_df) < 2:
                continue
            contingency = pd.crosstab(pair_df[left], pair_df[right])
            if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                continue
            chi2, p_value, dof, _ = stats.chi2_contingency(contingency)
            n = contingency.to_numpy().sum()
            min_dim = min(contingency.shape) - 1
            cramers_v = np.sqrt(chi2 / (n * min_dim)) if n and min_dim else np.nan
            rows.append(
                {
                    "feature_1": left,
                    "feature_2": right,
                    "rows_used": int(n),
                    "chi2_statistic": _round(chi2),
                    "pvalue": _round(p_value),
                    "degrees_of_freedom": int(dof),
                    "cramers_v": _round(cramers_v),
                }
            )
    return pd.DataFrame(rows).sort_values("cramers_v", ascending=False) if rows else pd.DataFrame(columns=["feature_1", "feature_2", "rows_used", "chi2_statistic", "pvalue", "degrees_of_freedom", "cramers_v"])


def _is_categorical_association_candidate(series: pd.Series, config: EDAConfig) -> bool:
    """Exclude constant and nearly row-unique categorical columns from association tests."""
    non_null = int(series.notna().sum())
    if non_null < 2:
        return False
    unique = int(series.nunique(dropna=True))
    if unique < 2:
        return False
    unique_ratio = unique / non_null
    return unique_ratio < config.categorical_association_max_unique_ratio


def _build_categorical_target_tests(df: pd.DataFrame, categorical_columns: list[str], config: EDAConfig) -> pd.DataFrame:
    """Compare categorical features against a categorical target."""
    target = config.target_column
    if not target or target not in df.columns or ptypes.is_numeric_dtype(df[target]):
        return pd.DataFrame(columns=["column", "target_column", "rows_used", "feature_levels", "target_levels", "chi2_statistic", "pvalue", "degrees_of_freedom", "cramers_v"])
    rows = []
    for column in categorical_columns:
        if column == target or column == config.id_column or not _is_categorical_association_candidate(df[column], config):
            continue
        pair_df = df[[column, target]].dropna().astype(str)
        if len(pair_df) < 2:
            continue
        contingency = pd.crosstab(pair_df[column], pair_df[target])
        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            continue
        chi2, p_value, dof, _ = stats.chi2_contingency(contingency)
        n = contingency.to_numpy().sum()
        min_dim = min(contingency.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim)) if n and min_dim else np.nan
        rows.append(
            {
                "column": column,
                "target_column": target,
                "rows_used": int(n),
                "feature_levels": int(contingency.shape[0]),
                "target_levels": int(contingency.shape[1]),
                "chi2_statistic": _round(chi2),
                "pvalue": _round(p_value),
                "degrees_of_freedom": int(dof),
                "cramers_v": _round(cramers_v),
            }
        )
    return pd.DataFrame(rows).sort_values("cramers_v", ascending=False) if rows else pd.DataFrame(columns=["column", "target_column", "rows_used", "feature_levels", "target_levels", "chi2_statistic", "pvalue", "degrees_of_freedom", "cramers_v"])


def _build_numeric_target_correlations(df: pd.DataFrame, numeric_columns: list[str], config: EDAConfig) -> pd.DataFrame:
    """Correlate numeric features with a numeric target."""
    target = config.target_column
    if not target or target not in df.columns or not ptypes.is_numeric_dtype(df[target]):
        return pd.DataFrame(columns=["column", "target_column", "rows_used", "pearson_correlation", "pearson_pvalue", "spearman_correlation", "spearman_pvalue", "abs_spearman_correlation"])
    rows = []
    for column in numeric_columns:
        if column == target:
            continue
        pair_df = df[[column, target]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(pair_df) < 3 or pair_df[column].nunique() < 2 or pair_df[target].nunique() < 2:
            continue
        pearson = stats.pearsonr(pair_df[column], pair_df[target])
        spearman = stats.spearmanr(pair_df[column], pair_df[target])
        rows.append(
            {
                "column": column,
                "target_column": target,
                "rows_used": int(len(pair_df)),
                "pearson_correlation": _round(pearson.statistic),
                "pearson_pvalue": _round(pearson.pvalue),
                "spearman_correlation": _round(spearman.statistic),
                "spearman_pvalue": _round(spearman.pvalue),
                "abs_spearman_correlation": _round(abs(spearman.statistic)),
            }
        )
    return pd.DataFrame(rows).sort_values("abs_spearman_correlation", ascending=False) if rows else pd.DataFrame(columns=["column", "target_column", "rows_used", "pearson_correlation", "pearson_pvalue", "spearman_correlation", "spearman_pvalue", "abs_spearman_correlation"])


def _build_missingness_target_tests(df: pd.DataFrame, config: EDAConfig) -> pd.DataFrame:
    """Test whether each column's missingness is associated with a categorical target."""
    target = config.target_column
    if not target or target not in df.columns or ptypes.is_numeric_dtype(df[target]):
        return pd.DataFrame(columns=["column", "target_column", "missing_count", "rows_used", "chi2_statistic", "pvalue", "degrees_of_freedom", "cramers_v"])
    rows = []
    target_series = df[target]
    for column in df.columns:
        if column == target:
            continue
        missing_flag = df[column].isna()
        if int(missing_flag.sum()) == 0 or missing_flag.nunique() < 2:
            continue
        pair_df = pd.DataFrame({"missing": missing_flag, target: target_series}).dropna()
        if len(pair_df) < 2:
            continue
        contingency = pd.crosstab(pair_df["missing"].map({False: "present", True: "missing"}), pair_df[target].astype(str))
        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            continue
        chi2, p_value, dof, _ = stats.chi2_contingency(contingency)
        n = contingency.to_numpy().sum()
        min_dim = min(contingency.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim)) if n and min_dim else np.nan
        rows.append(
            {
                "column": column,
                "target_column": target,
                "missing_count": int(missing_flag.sum()),
                "rows_used": int(n),
                "chi2_statistic": _round(chi2),
                "pvalue": _round(p_value),
                "degrees_of_freedom": int(dof),
                "cramers_v": _round(cramers_v),
            }
        )
    return pd.DataFrame(rows).sort_values("cramers_v", ascending=False) if rows else pd.DataFrame(columns=["column", "target_column", "missing_count", "rows_used", "chi2_statistic", "pvalue", "degrees_of_freedom", "cramers_v"])

def _build_numeric_target_tests(df: pd.DataFrame, numeric_columns: list[str], config: EDAConfig) -> pd.DataFrame:
    """Compare numeric columns across categorical target groups when possible."""
    target = config.target_column
    if not target or target not in df.columns or ptypes.is_numeric_dtype(df[target]):
        return pd.DataFrame()
    rows = []
    for column in numeric_columns:
        groups = [
            pd.to_numeric(group[column], errors="coerce").dropna()
            for _, group in df[[target, column]].dropna().groupby(target)
        ]
        groups = [group[np.isfinite(group)] for group in groups if len(group[np.isfinite(group)]) >= 2]
        if len(groups) < 2:
            continue
        kruskal_stat, kruskal_p = stats.kruskal(*groups)
        f_stat, anova_p = stats.f_oneway(*groups)
        group_sizes = [len(group) for group in groups]
        rows.append(
            {
                "column": column,
                "target_column": target,
                "groups": len(groups),
                "min_group_size": int(min(group_sizes)),
                "max_group_size": int(max(group_sizes)),
                "anova_f_statistic": _round(f_stat),
                "anova_pvalue": _round(anova_p),
                "kruskal_statistic": _round(kruskal_stat),
                "kruskal_pvalue": _round(kruskal_p),
            }
        )
    return pd.DataFrame(rows)


def _build_pca_summary(df: pd.DataFrame, numeric_columns: list[str], config: EDAConfig) -> pd.DataFrame:
    """Build a PCA explained-variance summary using numeric columns."""
    numeric_df = _prepared_numeric_matrix(df, numeric_columns, config)
    if numeric_df.shape[1] < 2 or numeric_df.shape[0] < 3:
        return pd.DataFrame()
    _, singular_values, _ = np.linalg.svd(numeric_df.to_numpy(), full_matrices=False)
    variances = (singular_values**2) / max(numeric_df.shape[0] - 1, 1)
    explained = variances / variances.sum() if variances.sum() else np.zeros_like(variances)
    cumulative = np.cumsum(explained)
    rows = []
    for idx, (ratio, cum_ratio) in enumerate(zip(explained, cumulative), start=1):
        rows.append(
            {
                "component": f"PC{idx}",
                "explained_variance_ratio": _round(ratio),
                "cumulative_explained_variance_ratio": _round(cum_ratio),
            }
        )
    return pd.DataFrame(rows)


def _build_multivariate_outliers(df: pd.DataFrame, numeric_columns: list[str], config: EDAConfig) -> pd.DataFrame:
    """Build the EDA Mahalanobis-distance outlier artifact.

    Returns the 100 rows with the largest multivariate distance, matching the
    default limit of the shared :class:`OutlierAnalysis` implementation.
    """
    return _outlier_analysis(df, config).detect_multivariate_mahalanobis(
        numeric_columns
    )


def _outlier_analysis(df: pd.DataFrame, config: EDAConfig) -> OutlierAnalysis:
    """Create an outlier calculator configured identically to the EDA run."""
    # Pass every EDA-controlled outlier option through to prevent calculation
    # drift between this pipeline and direct OutlierAnalysis usage.
    return OutlierAnalysis(
        df,
        iqr_multiplier=config.iqr_multiplier,
        z_score_threshold=config.outlier_zscore_threshold,
        random_seed=config.random_state,
        max_multivariate_features=config.max_multivariate_features,
    )

def _prepared_numeric_matrix(df: pd.DataFrame, numeric_columns: list[str], config: EDAConfig) -> pd.DataFrame:
    """Return a standardized numeric matrix for multivariate diagnostics."""
    columns = numeric_columns[: config.max_multivariate_features]
    if not columns:
        return pd.DataFrame()
    numeric_df = df[columns].apply(pd.to_numeric, errors="coerce")
    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)
    numeric_df = numeric_df.dropna(axis=1, how="all")
    if numeric_df.empty:
        return pd.DataFrame()
    numeric_df = numeric_df.fillna(numeric_df.median(numeric_only=True))
    numeric_df = numeric_df.loc[:, numeric_df.std(ddof=0) > 0]
    if numeric_df.empty:
        return pd.DataFrame()
    return (numeric_df - numeric_df.mean()) / numeric_df.std(ddof=0)

def _build_target_summary(df: pd.DataFrame, config: EDAConfig) -> pd.DataFrame:
    """Summarize target distribution and numeric features by target when configured."""
    if not config.target_column or config.target_column not in df.columns:
        return pd.DataFrame()
    target = df[config.target_column]
    numeric_columns = [column for column in _numeric_columns(df) if column != config.target_column]
    if ptypes.is_numeric_dtype(target):
        rows = [{"section": "target_numeric", **_series_numeric_summary(target, config)}]
    else:
        counts = target.dropna().astype(str).value_counts()
        rows = [
            {
                "section": "target_distribution",
                "target_value": value,
                "count": int(count),
                "percent": _safe_percent(count, target.notna().sum()),
            }
            for value, count in counts.items()
        ]
    for column in numeric_columns:
        grouped = df.groupby(config.target_column, dropna=True)[column].agg(["count", "mean", "median", "std", "min", "max"]).reset_index()
        for _, row in grouped.iterrows():
            rows.append(
                {
                    "section": "numeric_by_target",
                    "column": column,
                    "target_value": row[config.target_column],
                    "count": int(row["count"]),
                    "mean": _round(row["mean"]),
                    "median": _round(row["median"]),
                    "std": _round(row["std"]),
                    "min": _round(row["min"]),
                    "max": _round(row["max"]),
                }
            )
    return pd.DataFrame(rows)


def _build_geospatial_summary(df: pd.DataFrame, config: EDAConfig) -> pd.DataFrame:
    """Create geospatial diagnostics when geometry is available."""
    if not config.include_geospatial or "geometry" not in df.columns:
        return pd.DataFrame()
    geometry = df["geometry"]
    if not hasattr(geometry, "geom_type"):
        return pd.DataFrame()
    rows = []
    crs = getattr(df, "crs", None)
    rows.append({"metric": "crs", "value": str(crs)})
    rows.append({"metric": "total_geometries", "value": int(len(geometry))})
    rows.append({"metric": "missing_geometries", "value": int(geometry.isna().sum())})
    rows.append({"metric": "empty_geometries", "value": int(geometry.is_empty.fillna(False).sum())})
    rows.append({"metric": "invalid_geometries", "value": int((~geometry.is_valid.fillna(False)).sum())})
    for geom_type, count in geometry.geom_type.value_counts(dropna=True).items():
        rows.append({"metric": f"geometry_type_{geom_type}", "value": int(count)})
    try:
        bounds = df.total_bounds
        rows.extend(
            [
                {"metric": "min_x", "value": _round(bounds[0])},
                {"metric": "min_y", "value": _round(bounds[1])},
                {"metric": "max_x", "value": _round(bounds[2])},
                {"metric": "max_y", "value": _round(bounds[3])},
            ]
        )
    except Exception:
        pass
    try:
        is_projected = bool(getattr(crs, "is_projected", False))
        rows.append({"metric": "is_projected_crs", "value": is_projected})
        if is_projected:
            areas = geometry.area.replace([np.inf, -np.inf], np.nan).dropna()
            lengths = geometry.length.replace([np.inf, -np.inf], np.nan).dropna()
            rows.extend(
                [
                    {"metric": "area_mean", "value": _round(areas.mean())},
                    {"metric": "area_total", "value": _round(areas.sum())},
                    {"metric": "length_mean", "value": _round(lengths.mean())},
                    {"metric": "length_total", "value": _round(lengths.sum())},
                ]
            )
    except Exception:
        pass
    return pd.DataFrame(rows)


def _series_numeric_summary(series: pd.Series, config: EDAConfig) -> dict[str, Any]:
    """Return compact numeric statistics for a single series."""
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return {
        "column": series.name,
        "count": int(clean.count()),
        "missing_count": int(series.isna().sum()),
        "mean": _round(clean.mean()),
        "std": _round(clean.std()),
        "min": _round(clean.min()),
        "median": _round(clean.median()),
        "max": _round(clean.max()),
    }


def _logical_type(series: pd.Series, column: str) -> str:
    """Infer a friendly type label for report tables."""
    if column == "geometry":
        return "geometry"
    if ptypes.is_bool_dtype(series):
        return "boolean"
    if ptypes.is_numeric_dtype(series):
        return "numeric"
    if ptypes.is_datetime64_any_dtype(series):
        return "datetime"
    return "categorical"


def _safe_percent(numerator: float, denominator: float) -> float:
    """Calculate a percentage while avoiding division by zero."""
    return _round(100 * numerator / denominator) if denominator else 0.0


def _round(value: Any, digits: int = 4) -> Any:
    """Round finite numeric values and preserve missing values."""
    try:
        if pd.isna(value) or not np.isfinite(value):
            return np.nan
        return round(float(value), digits)
    except Exception:
        return value
