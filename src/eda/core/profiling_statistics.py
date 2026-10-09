"""Statistical diagnostics and association tests for EDA profiling."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

from .config import EDAConfig
from .profiling_helpers import _prepared_numeric_matrix, _round, _target_task
from .profiling_multivariate import _outlier_analysis


def _build_normality_tests(df: pd.DataFrame, numeric_columns: list[str], config: EDAConfig) -> pd.DataFrame:
    """Run common normality checks for numeric columns."""
    rows = []
    # Bound normality-test cost with reproducible sampling while leaving the input data untouched.
    rng = np.random.default_rng(config.random_state)
    for column in numeric_columns:
        clean = pd.to_numeric(df[column], errors="coerce").dropna()
        clean = clean[np.isfinite(clean)]
        if len(clean) > config.normality_sample_size:
            clean = pd.Series(rng.choice(clean.to_numpy(), config.normality_sample_size, replace=False))
        row = {
            "column": column,
            "sample_size": len(clean),
            "shapiro_statistic": np.nan,
            "shapiro_pvalue": np.nan,
            "dagostino_statistic": np.nan,
            "dagostino_pvalue": np.nan,
            "jarque_bera_statistic": np.nan,
            "jarque_bera_pvalue": np.nan,
            "anderson_statistic": np.nan,
            "anderson_pvalue": np.nan,
            "normality_hint": "insufficient_data",
        }
        if len(clean) >= 3:
            shapiro = stats.shapiro(clean)
            row["shapiro_statistic"] = _round(shapiro.statistic)
            row["shapiro_pvalue"] = _round(shapiro.pvalue)
        if len(clean) >= 8:
            dagostino = stats.normaltest(clean)
            jb = stats.jarque_bera(clean)
            try:
                # SciPy 1.17+ requires an explicit p-value method; older versions do not accept this keyword.
                anderson = stats.anderson(clean, dist="norm", method="interpolate")
            except TypeError:
                anderson = stats.anderson(clean, dist="norm")
            row.update(
                {
                    "dagostino_statistic": _round(dagostino.statistic),
                    "dagostino_pvalue": _round(dagostino.pvalue),
                    "jarque_bera_statistic": _round(jb.statistic),
                    "jarque_bera_pvalue": _round(jb.pvalue),
                    "anderson_statistic": _round(anderson.statistic),
                    "anderson_pvalue": _round(getattr(anderson, "pvalue", np.nan)),
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
    return {"pearson": numeric_df.corr(method="pearson"), "spearman": numeric_df.corr(method="spearman")}

def _build_strong_correlation_pairs(correlation_matrix: pd.DataFrame, config: EDAConfig) -> pd.DataFrame:
    """Return pairs whose absolute Pearson correlation exceeds the threshold."""
    if correlation_matrix.empty:
        return pd.DataFrame(columns=["feature_1", "feature_2", "pearson_correlation", "abs_correlation"])
    pairs = []
    columns = list(correlation_matrix.columns)
    for idx, left in enumerate(columns):
        # Visit only the upper triangle to omit self-correlations and duplicate symmetric pairs.
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
    return (
        pd.DataFrame(pairs).sort_values("abs_correlation", ascending=False)
        if pairs
        else pd.DataFrame(columns=["feature_1", "feature_2", "pearson_correlation", "abs_correlation"])
    )

def _build_multicollinearity_summary(df: pd.DataFrame, numeric_columns: list[str], config: EDAConfig) -> pd.DataFrame:
    """Calculate variance inflation factors for numeric columns."""
    numeric_df = _prepared_numeric_matrix(df, numeric_columns, config, max_features=config.max_features_per_plot)
    if numeric_df.shape[1] < 2 or numeric_df.shape[0] < 3:
        return pd.DataFrame(columns=["column", "vif", "tolerance", "r_squared", "interpretation"])
    rows = []
    columns = list(numeric_df.columns)
    values = numeric_df.to_numpy(dtype=float)
    for index, column in enumerate(columns):
        y = values[:, index]
        x = np.delete(values, index, axis=1)
        # Include an intercept when regressing each feature on the remaining features for VIF.
        x = np.column_stack([np.ones(len(x)), x])
        try:
            beta, *_ = np.linalg.lstsq(x, y, rcond=None)
            fitted = x @ beta
            ss_total = float(np.sum((y - y.mean()) ** 2))
            ss_resid = float(np.sum((y - fitted) ** 2))
            r_squared = 1 - ss_resid / ss_total if ss_total else 1.0
            # Cap near-perfect fits to keep the reported VIF finite and numerically stable.
            r_squared = min(max(r_squared, 0.0), 0.999999)
            tolerance = 1 - r_squared
            vif = 1 / tolerance if tolerance > 0 else np.inf
        except Exception:
            logging.getLogger(__name__).debug(
                "Error: _build_multicollinearity_summary failed; using its fallback.", exc_info=True
            )
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
    return (
        pd.DataFrame(rows).sort_values("cramers_v", ascending=False)
        if rows
        else pd.DataFrame(
            columns=["feature_1", "feature_2", "rows_used", "chi2_statistic", "pvalue", "degrees_of_freedom", "cramers_v"]
        )
    )

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

def _benjamini_hochberg(pvalues: pd.Series) -> pd.Series:
    """Control the false-discovery rate while preserving the source index."""
    adjusted = pd.Series(np.nan, index=pvalues.index, dtype=float)
    valid = pd.to_numeric(pvalues, errors="coerce").dropna().clip(0, 1)
    if valid.empty:
        return adjusted
    ordered = valid.sort_values()
    ranks = np.arange(1, len(ordered) + 1)
    raw_adjusted = ordered.to_numpy() * len(ordered) / ranks
    # Reverse cumulative minima enforce monotone Benjamini-Hochberg adjusted p-values.
    monotonic = np.minimum.accumulate(raw_adjusted[::-1])[::-1]
    # Restore the original feature alignment after sorting p-values for the correction.
    adjusted.loc[ordered.index] = np.clip(monotonic, 0, 1)
    return adjusted

def _cramers_v_effect(contingency: pd.DataFrame, chi2: float) -> tuple[float, float]:
    """Return conventional and finite-sample bias-corrected Cramer's V."""
    rows, columns = contingency.shape
    observations = float(contingency.to_numpy().sum())
    min_dimension = min(rows, columns) - 1
    conventional = np.sqrt(chi2 / (observations * min_dimension)) if observations and min_dimension else np.nan
    if observations <= 1:
        return conventional, np.nan
    phi_squared = chi2 / observations
    # Subtract finite-sample bias before normalizing the categorical association strength.
    corrected_phi = max(0.0, phi_squared - ((columns - 1) * (rows - 1)) / (observations - 1))
    corrected_rows = rows - (rows - 1) ** 2 / (observations - 1)
    corrected_columns = columns - (columns - 1) ** 2 / (observations - 1)
    denominator = min(corrected_rows - 1, corrected_columns - 1)
    corrected = np.sqrt(corrected_phi / denominator) if denominator > 0 else np.nan
    return conventional, corrected

def _group_effect_sizes(groups: list[pd.Series], kruskal_statistic: float | None = None) -> tuple[float, float]:
    """Calculate ANOVA eta-squared and Kruskal epsilon-squared group effects."""
    values = np.concatenate([group.to_numpy(dtype=float) for group in groups])
    grand_mean = values.mean()
    between_sum_squares = sum(len(group) * (float(group.mean()) - grand_mean) ** 2 for group in groups)
    total_sum_squares = float(np.sum((values - grand_mean) ** 2))
    eta_squared = between_sum_squares / total_sum_squares if total_sum_squares > 0 else np.nan
    kruskal_statistic = float(stats.kruskal(*groups).statistic) if kruskal_statistic is None else kruskal_statistic
    # Kruskal effect size uses residual degrees of freedom and is clipped to a nonnegative value.
    denominator = len(values) - len(groups)
    epsilon_squared = max(0.0, (kruskal_statistic - len(groups) + 1) / denominator) if denominator > 0 else np.nan
    return eta_squared, min(epsilon_squared, 1.0) if np.isfinite(epsilon_squared) else np.nan

def _build_categorical_target_tests(df: pd.DataFrame, categorical_columns: list[str], config: EDAConfig) -> pd.DataFrame:
    """Compare categorical features against a categorical target."""
    target = config.target_column
    if not target or target not in df.columns or _target_task(df, config) != "classification":
        return pd.DataFrame(
            columns=[
                "column",
                "target_column",
                "rows_used",
                "feature_levels",
                "target_levels",
                "chi2_statistic",
                "pvalue",
                "degrees_of_freedom",
                "cramers_v",
                "cramers_v_bias_corrected",
                "qvalue",
            ]
        )
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
        cramers_v, corrected_cramers_v = _cramers_v_effect(contingency, chi2)
        rows.append(
            {
                "column": column,
                "target_column": target,
                "rows_used": int(n),
                "feature_levels": int(contingency.shape[0]),
                "target_levels": int(contingency.shape[1]),
                "chi2_statistic": _round(chi2),
                "pvalue": float(p_value),
                "degrees_of_freedom": int(dof),
                "cramers_v": _round(cramers_v),
                "cramers_v_bias_corrected": _round(corrected_cramers_v),
            }
        )
    if rows:
        result = pd.DataFrame(rows)
        result["qvalue"] = _benjamini_hochberg(result["pvalue"]).round(4)
        result["pvalue"] = result["pvalue"].round(6)
        return result.sort_values("cramers_v_bias_corrected", ascending=False)
    return pd.DataFrame(
        columns=[
            "column",
            "target_column",
            "rows_used",
            "feature_levels",
            "target_levels",
            "chi2_statistic",
            "pvalue",
            "degrees_of_freedom",
            "cramers_v",
            "cramers_v_bias_corrected",
            "qvalue",
        ]
    )

def _build_numeric_target_correlations(df: pd.DataFrame, numeric_columns: list[str], config: EDAConfig) -> pd.DataFrame:
    """Correlate numeric features with a numeric target."""
    target = config.target_column
    if not target or target not in df.columns or _target_task(df, config) != "regression":
        return pd.DataFrame(
            columns=[
                "column",
                "target_column",
                "rows_used",
                "pearson_correlation",
                "pearson_pvalue",
                "spearman_correlation",
                "spearman_pvalue",
                "kendall_correlation",
                "kendall_pvalue",
                "abs_spearman_correlation",
                "qvalue",
            ]
        )
    rows = []
    for column in numeric_columns:
        if column == target or column == config.id_column:
            continue
        pair_df = df[[column, target]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(pair_df) < 3 or pair_df[column].nunique() < 2 or pair_df[target].nunique() < 2:
            continue
        pearson = stats.pearsonr(pair_df[column], pair_df[target])
        spearman = stats.spearmanr(pair_df[column], pair_df[target])
        kendall = stats.kendalltau(pair_df[column], pair_df[target])
        rows.append(
            {
                "column": column,
                "target_column": target,
                "rows_used": len(pair_df),
                "pearson_correlation": _round(pearson.statistic),
                "pearson_pvalue": float(pearson.pvalue),
                "spearman_correlation": _round(spearman.statistic),
                "spearman_pvalue": float(spearman.pvalue),
                "kendall_correlation": _round(kendall.statistic),
                "kendall_pvalue": float(kendall.pvalue),
                "abs_spearman_correlation": _round(abs(spearman.statistic)),
            }
        )
    if rows:
        result = pd.DataFrame(rows)
        result["qvalue"] = _benjamini_hochberg(result["spearman_pvalue"]).round(4)
        for column in ["pearson_pvalue", "spearman_pvalue", "kendall_pvalue"]:
            result[column] = result[column].round(6)
        return result.sort_values("abs_spearman_correlation", ascending=False)
    return pd.DataFrame(
        columns=[
            "column",
            "target_column",
            "rows_used",
            "pearson_correlation",
            "pearson_pvalue",
            "spearman_correlation",
            "spearman_pvalue",
            "kendall_correlation",
            "kendall_pvalue",
            "abs_spearman_correlation",
            "qvalue",
        ]
    )

def _build_missingness_target_tests(df: pd.DataFrame, config: EDAConfig) -> pd.DataFrame:
    """Test whether each column's missingness is associated with a categorical target."""
    target = config.target_column
    if not target or target not in df.columns or _target_task(df, config) != "classification":
        return pd.DataFrame(
            columns=[
                "column",
                "target_column",
                "missing_count",
                "rows_used",
                "chi2_statistic",
                "pvalue",
                "degrees_of_freedom",
                "cramers_v",
                "cramers_v_bias_corrected",
                "qvalue",
            ]
        )
    rows = []
    target_series = df[target]
    for column in df.columns:
        if column == target or column == config.id_column:
            continue
        # Test the missingness indicator itself, which can carry target-related information.
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
        cramers_v, corrected_cramers_v = _cramers_v_effect(contingency, chi2)
        rows.append(
            {
                "column": column,
                "target_column": target,
                "missing_count": int(missing_flag.sum()),
                "rows_used": int(n),
                "chi2_statistic": _round(chi2),
                "pvalue": float(p_value),
                "degrees_of_freedom": int(dof),
                "cramers_v": _round(cramers_v),
                "cramers_v_bias_corrected": _round(corrected_cramers_v),
            }
        )
    if rows:
        result = pd.DataFrame(rows)
        result["qvalue"] = _benjamini_hochberg(result["pvalue"]).round(4)
        result["pvalue"] = result["pvalue"].round(6)
        return result.sort_values("cramers_v_bias_corrected", ascending=False)
    return pd.DataFrame(
        columns=[
            "column",
            "target_column",
            "missing_count",
            "rows_used",
            "chi2_statistic",
            "pvalue",
            "degrees_of_freedom",
            "cramers_v",
            "cramers_v_bias_corrected",
            "qvalue",
        ]
    )

def _build_numeric_target_tests(df: pd.DataFrame, numeric_columns: list[str], config: EDAConfig) -> pd.DataFrame:
    """Compare numeric columns across categorical target groups when possible."""
    target = config.target_column
    result_columns = [
        "column",
        "target_column",
        "groups",
        "rows_used",
        "min_group_size",
        "max_group_size",
        "median_range",
        "anova_f_statistic",
        "anova_pvalue",
        "eta_squared",
        "kruskal_statistic",
        "kruskal_pvalue",
        "epsilon_squared",
        "qvalue",
    ]
    if not target or target not in df.columns or _target_task(df, config) != "classification":
        return pd.DataFrame(columns=result_columns)
    rows = []
    for column in numeric_columns:
        if column == target or column == config.id_column:
            continue
        groups = _classification_numeric_target_groups(df, target, column)
        if len(groups) < 2:
            continue
        rows.append(_target_group_test_row(column, target, groups))
    if not rows:
        return pd.DataFrame(columns=result_columns)
    result = pd.DataFrame(rows)
    result["qvalue"] = _benjamini_hochberg(result["kruskal_pvalue"]).round(4)
    result[["anova_pvalue", "kruskal_pvalue"]] = result[["anova_pvalue", "kruskal_pvalue"]].round(6)
    return result.sort_values("epsilon_squared", ascending=False)[result_columns]

def _build_categorical_numeric_target_tests(df: pd.DataFrame, categorical_columns: list[str], config: EDAConfig) -> pd.DataFrame:
    """Compare a numeric target distribution across categorical feature levels."""
    target = config.target_column
    result_columns = [
        "column",
        "target_column",
        "groups",
        "rows_used",
        "min_group_size",
        "max_group_size",
        "median_range",
        "anova_f_statistic",
        "anova_pvalue",
        "eta_squared",
        "kruskal_statistic",
        "kruskal_pvalue",
        "epsilon_squared",
        "qvalue",
    ]
    if not target or target not in df.columns or _target_task(df, config) != "regression":
        return pd.DataFrame(columns=result_columns)

    rows = []
    for column in categorical_columns:
        if column == target or column == config.id_column or not _is_categorical_association_candidate(df[column], config):
            continue
        groups = _regression_target_groups_by_categorical(df, column, target)
        if len(groups) < 2:
            continue
        rows.append(_target_group_test_row(column, target, groups))
    if not rows:
        return pd.DataFrame(columns=result_columns)
    result = pd.DataFrame(rows)
    result["qvalue"] = _benjamini_hochberg(result["kruskal_pvalue"]).round(4)
    result[["anova_pvalue", "kruskal_pvalue"]] = result[["anova_pvalue", "kruskal_pvalue"]].round(6)
    return result.sort_values("epsilon_squared", ascending=False)[result_columns]

def _classification_numeric_target_groups(df: pd.DataFrame, target: str, column: str) -> list[pd.Series]:
    """Build finite numeric groups for a classification target test."""
    groups = [pd.to_numeric(group[column], errors="coerce").dropna() for _, group in df[[target, column]].dropna().groupby(target)]
    # Require at least two finite observations per retained class group.
    return [group[np.isfinite(group)] for group in groups if len(group[np.isfinite(group)]) >= 2]

def _regression_target_groups_by_categorical(df: pd.DataFrame, column: str, target: str) -> list[pd.Series]:
    """Build numeric-target groups keyed by categorical feature levels."""
    working = df[[column, target]].copy()
    working[target] = pd.to_numeric(working[target], errors="coerce")
    working = working.replace([np.inf, -np.inf], np.nan).dropna()
    return [group[target] for _, group in working.groupby(column, observed=True) if len(group) >= 2]

def _target_group_test_row(column: str, target_column: str, groups: list[pd.Series]) -> dict[str, float | int | str]:
    """Build one ANOVA/Kruskal/effect-size row from grouped values."""
    kruskal_statistic, kruskal_pvalue = stats.kruskal(*groups)
    anova_statistic, anova_pvalue = stats.f_oneway(*groups)
    eta_squared, epsilon_squared = _group_effect_sizes(groups, float(kruskal_statistic))
    group_sizes = [len(group) for group in groups]
    return {
        "column": column,
        "target_column": target_column,
        "groups": len(groups),
        "rows_used": int(sum(group_sizes)),
        "min_group_size": int(min(group_sizes)),
        "max_group_size": int(max(group_sizes)),
        "median_range": _round(max(group.median() for group in groups) - min(group.median() for group in groups)),
        "anova_f_statistic": _round(anova_statistic),
        "anova_pvalue": float(anova_pvalue),
        "eta_squared": _round(eta_squared),
        "kruskal_statistic": _round(kruskal_statistic),
        "kruskal_pvalue": float(kruskal_pvalue),
        "epsilon_squared": _round(epsilon_squared),
    }

def _association_strength(method: str, effect_size: float) -> str:
    """Translate method-specific effect sizes into cautious screening labels."""
    if not np.isfinite(effect_size):
        return "unavailable"
    if method == "kruskal_epsilon_squared":
        if effect_size >= 0.14:
            return "large"
        if effect_size >= 0.06:
            return "moderate"
        if effect_size >= 0.01:
            return "small"
        return "negligible"
    if effect_size >= 0.5:
        return "strong"
    if effect_size >= 0.3:
        return "moderate"
    if effect_size >= 0.1:
        return "weak"
    return "negligible"
