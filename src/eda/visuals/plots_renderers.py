"""Rendering functions for EDA plots."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

# Select the headless backend before importing pyplot.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from pandas.plotting import scatter_matrix
from scipy import stats

from ..core.config import EDAConfig
from .plots_common import (
    PROFILE_LABEL_FONT_SIZE,
    _apply_target_color_mapping,
    _boxplot,
    _format_vertical_category_labels,
    _limited_categories,
    _prepared_numeric_matrix,
    _style_profile_labels,
    _target_color_mapping,
    _target_median_percentile_rows,
)


def _plot_data_quality_flags(flags: pd.DataFrame, output_path: Path) -> None:
    """Plot counts of data-quality flags by issue type."""
    counts = flags["issue_type"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(counts) + 2)))
    sns.barplot(x=counts.values, y=counts.index, ax=ax, color="#e45756")
    ax.set_title("Data Quality Flags")
    ax.set_xlabel("Columns flagged")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)

def _plot_missing_values(
    column_profile: pd.DataFrame, ordered_features: list[str], max_features: int, output_path: Path
) -> bool:
    """Plot missing percentage by column."""
    available = set(column_profile["column"].tolist())
    selected = [feature for feature in ordered_features if feature in available][:max_features]
    if not selected:
        return False
    data = column_profile.set_index("column").loc[selected].reset_index().sort_values("missing_percent", ascending=True)
    height = max(4, min(18, 0.35 * len(data) + 2))
    fig, ax = plt.subplots(figsize=(10, height))
    sns.barplot(data=data, x="missing_percent", y="column", color="#4c78a8", ax=ax)
    ax.set_title("Missing Values by Column")
    ax.set_xlabel("Missing (%)")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return True

def _plot_missingness_heatmap(
    df: pd.DataFrame, ordered_features: list[str], max_features: int, output_path: Path
) -> bool:
    """Plot a row-sampled missingness heatmap."""
    selected = [feature for feature in ordered_features if feature in df.columns][:max_features]
    if not selected:
        return False
    missing = df[selected].isna()
    if len(missing) > 500:
        # Keep the image bounded and reproducible, then restore row order for pattern inspection.
        missing = missing.sample(500, random_state=42).sort_index()
    fig, ax = plt.subplots(figsize=(max(8, 0.35 * len(missing.columns)), 6))
    sns.heatmap(missing, cbar=False, yticklabels=False, cmap=["#f8faf8", "#1f6f5b"], ax=ax)
    ax.set_title("Missingness Pattern")
    ax.set_xlabel("Columns")
    ax.set_ylabel("Rows")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return True

def _plot_numeric_distributions(df: pd.DataFrame, columns: list[str], output_path: Path) -> None:
    """Plot histogram and boxplot pairs for numeric columns."""
    rows = len(columns)
    fig, axes = plt.subplots(rows, 2, figsize=(12, max(3, 3 * rows)))
    # Ensure axes is two-dimensional for consistent iteration when rows == 1
    axes = np.atleast_2d(axes)
    for axes_row, column in zip(axes, columns):
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        sns.histplot(values, kde=True, ax=axes_row[0], color="#4c78a8")
        axes_row[0].set_title(f"{column} histogram")
        _boxplot(x=values, ax=axes_row[1], color="#f58518")
        axes_row[1].set_title(f"{column} boxplot")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)

def _plot_qq_plots(df: pd.DataFrame, columns: list[str], output_path: Path, config: EDAConfig) -> None:
    """Plot normal QQ plots for numeric columns."""
    cols = 2 if len(columns) > 1 else 1
    rows = int(np.ceil(len(columns) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)
    rng = np.random.default_rng(config.random_state)
    for ax, column in zip(axes, columns):
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        values = values[np.isfinite(values)]
        if len(values) > config.normality_sample_size:
            values = pd.Series(rng.choice(values.to_numpy(), config.normality_sample_size, replace=False))
        if len(values) >= 3:
            stats.probplot(values, dist="norm", plot=ax)
            ax.set_title(f"{column} QQ plot")
        else:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
            ax.set_title(f"{column} QQ plot")
    # Hide unused grid cells when the feature count does not fill the last subplot row.
    for ax in axes[len(columns) :]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)

def _plot_scatter_matrix(df: pd.DataFrame, columns: list[str], output_path: Path) -> None:
    """Plot a compact numeric scatter matrix."""
    numeric_df = df[columns].apply(pd.to_numeric, errors="coerce").dropna()
    if len(numeric_df) > 1000:
        numeric_df = numeric_df.sample(1000, random_state=42)
    axes = scatter_matrix(numeric_df, figsize=(3 * len(columns), 3 * len(columns)), diagonal="kde", alpha=0.45)
    for ax in np.ravel(axes):
        ax.tick_params(axis="x", labelrotation=90)
        ax.tick_params(axis="y", labelrotation=0)
    plt.suptitle("Numeric Scatter Matrix", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close()

def _plot_categorical_distributions(df: pd.DataFrame, columns: list[str], max_categories: int, output_path: Path) -> None:
    """Plot top categorical value counts."""
    rows = len(columns)
    fig, axes = plt.subplots(rows, 1, figsize=(12, max(3, 3.2 * rows)))
    if rows == 1:
        axes = [axes]
    for ax, column in zip(axes, columns):
        counts = df[column].dropna().astype(str).value_counts().head(max_categories).sort_values()
        sns.barplot(x=counts.values, y=counts.index, ax=ax, color="#54a24b")
        ax.set_title(f"{column} top values")
        ax.set_xlabel("Count")
        ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)

def _plot_categorical_association_heatmap(
    associations: pd.DataFrame, selected_features: list[str], output_path: Path
) -> bool:
    """Plot Cramer's V association matrix from pairwise results."""
    available = set(associations["feature_1"]).union(set(associations["feature_2"]))
    features = [feature for feature in selected_features if feature in available]
    if len(features) < 2:
        return False
    selected_set = set(features)
    associations = associations[
        associations["feature_1"].isin(selected_set) & associations["feature_2"].isin(selected_set)
    ]
    # Expand the stored pairs into a symmetric matrix with self-association on the diagonal.
    matrix = pd.DataFrame(np.eye(len(features)), index=features, columns=features)
    for _, row in associations.iterrows():
        matrix.loc[row["feature_1"], row["feature_2"]] = row["cramers_v"]
        matrix.loc[row["feature_2"], row["feature_1"]] = row["cramers_v"]
    fig, ax = plt.subplots(figsize=(max(6, 0.8 * len(features)), max(5, 0.7 * len(features))))
    sns.heatmap(matrix, vmin=0, vmax=1, cmap="crest", annot=len(features) <= 10, fmt=".2f", ax=ax)
    ax.set_title("Categorical Association Heatmap (Cramer's V)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return True

def _plot_correlation_heatmap(correlation_matrix: pd.DataFrame, output_path: Path, title: str) -> None:
    """Plot a correlation heatmap using the ML pipeline's visual style."""
    # Hide the diagonal and mirrored half so each feature pair appears once.
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    fontsize = 6 if correlation_matrix.shape[1] > 8 else 8
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        correlation_matrix,
        mask=mask,
        cmap="coolwarm",
        vmax=0.3,
        center=0,
        square=True,
        linewidths=0.5,
        annot=True,
        fmt=".2f",
        cbar_kws={"shrink": 0.7},
        annot_kws={"fontsize": fontsize},
        ax=ax,
    )
    ax.tick_params(axis="x", labelrotation=90)
    ax.tick_params(axis="y", labelrotation=0)
    ax.set_title(title)
    for axis in fig.axes:
        _style_profile_labels(axis)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

def _plot_vif_summary(vif_summary: pd.DataFrame, max_features: int, output_path: Path) -> None:
    """Plot VIF values for numeric features."""
    data = vif_summary.copy()
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=["vif"])
    if data.empty:
        return
    data = data.sort_values("vif", ascending=True).tail(max_features)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(data) + 2)))
    sns.barplot(data=data, x="vif", y="column", ax=ax, color="#f58518")
    ax.axvline(5, color="#9aa5b1", linestyle="--", linewidth=1, label="VIF 5")
    ax.axvline(10, color="#d62728", linestyle="--", linewidth=1, label="VIF 10")
    ax.set_title("Variance Inflation Factor")
    ax.set_xlabel("VIF")
    ax.set_ylabel("")
    _style_profile_labels(ax)
    ax.legend(fontsize=PROFILE_LABEL_FONT_SIZE)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)

def _plot_association_bars(
    table: pd.DataFrame, label_column: str, value_column: str, title: str, max_features: int, output_path: Path
) -> None:
    """Plot sorted association strengths from a table."""
    data = table[[label_column, value_column]].dropna().sort_values(value_column, ascending=True).tail(max_features)
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(data) + 2)))
    sns.barplot(data=data, x=value_column, y=label_column, ax=ax, color="#54a24b")
    ax.set_xlim(0, 1)
    ax.set_title(title)
    ax.set_xlabel(value_column.replace("_", " ").title())
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)

def _plot_numeric_target_correlations(correlations: pd.DataFrame, max_features: int, output_path: Path) -> None:
    """Plot strongest numeric-target Spearman correlations."""
    data = correlations[["column", "spearman_correlation"]].dropna().copy()
    if data.empty:
        return
    data["abs_correlation"] = data["spearman_correlation"].abs()
    data = data.sort_values("abs_correlation", ascending=True).tail(max_features)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(data) + 2)))
    sns.barplot(data=data, x="spearman_correlation", y="column", ax=ax, color="#b279a2")
    ax.axvline(0, color="#52606d", linewidth=0.8)
    ax.set_xlim(-1, 1)
    ax.set_title("Numeric Features vs Target (Spearman)")
    ax.set_xlabel("Spearman correlation")
    ax.set_ylabel("")
    _style_profile_labels(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)

def _plot_numeric_target_scatter(df: pd.DataFrame, columns: list[str], target_column: str, output_path: Path) -> bool:
    """Plot top numeric features against a numeric target."""
    columns = [column for column in columns if column in df.columns and column != target_column]
    if not columns:
        return False
    rows = len(columns)
    fig, axes = plt.subplots(rows, 1, figsize=(10, max(3.2, 3.2 * rows)))
    if rows == 1:
        axes = [axes]
    for ax, column in zip(axes, columns):
        plot_df = df[[column, target_column]].apply(pd.to_numeric, errors="coerce").dropna()
        if plot_df.empty:
            ax.text(0.5, 0.5, "No paired data", ha="center", va="center")
        else:
            sns.regplot(
                data=plot_df,
                x=column,
                y=target_column,
                ax=ax,
                scatter_kws={"alpha": 0.55, "s": 26},
                line_kws={"color": "#e45756"},
            )
        ax.set_title(f"{column} vs {target_column}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return True

def _plot_pca_explained_variance(pca_summary: pd.DataFrame, output_path: Path) -> None:
    """Plot PCA explained variance."""
    fig, ax = plt.subplots(figsize=(10, 5))
    components = pca_summary["component"]
    ax.bar(components, pca_summary["explained_variance_ratio"], color="#4c78a8", label="Component")
    ax.plot(components, pca_summary["cumulative_explained_variance_ratio"], color="#f58518", marker="o", label="Cumulative")
    ax.set_title("PCA Explained Variance")
    ax.set_ylabel("Explained variance ratio")
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)

def _plot_pca_scores(df: pd.DataFrame, numeric_columns: list[str], config: EDAConfig, output_path: Path) -> bool:
    """Plot the first two PCA scores."""
    numeric_df = _prepared_numeric_matrix(df, numeric_columns, config)
    if numeric_df.shape[1] < 2 or numeric_df.shape[0] < 3:
        return False
    u, singular_values, _ = np.linalg.svd(numeric_df.to_numpy(), full_matrices=False)
    # For a centered matrix, U times singular values gives the principal-component scores.
    scores = u[:, :2] * singular_values[:2]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(scores[:, 0], scores[:, 1], s=28, alpha=0.7, color="#4c78a8")
    ax.axhline(0, color="#9aa5b1", linewidth=0.8)
    ax.axvline(0, color="#9aa5b1", linewidth=0.8)
    ax.set_xlabel("PC1 score")
    ax.set_ylabel("PC2 score")
    ax.set_title("PCA Score Plot")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return True

def _plot_target_distribution(
    series: pd.Series, target_task: str | None, max_target_levels: int, output_path: Path
) -> None:
    """Plot target distribution."""
    fig, ax = plt.subplots(figsize=(10, 5))
    if target_task == "regression":
        sns.histplot(series.dropna(), kde=True, ax=ax, color="#b279a2")
        ax.set_xlabel(series.name)
    else:
        target_colors = _target_color_mapping(series, max_target_levels)
        values = _apply_target_color_mapping(series, target_colors)
        counts = values.value_counts().sort_values()
        ax.barh(
            counts.index.astype(str),
            counts.values,
            color=[target_colors[str(level)] for level in counts.index],
        )
        ax.set_xlabel("Count")
        ax.set_ylabel("")
    ax.set_title("Target Distribution")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)

def _plot_numeric_target_median_percentile_heatmap(
    df: pd.DataFrame, columns: list[str], target_column: str, max_target_levels: int, output_path: Path
) -> bool:
    """Compare target-class medians across numeric features on a common percentile scale."""
    if not columns:
        return False
    target_colors = _target_color_mapping(df[target_column], max_target_levels)
    target_levels = list(target_colors)
    percentile_rows = _target_median_percentile_rows(df, columns, target_column, target_colors, target_levels)
    if not percentile_rows:
        return False
    profile = pd.DataFrame.from_dict(percentile_rows, orient="index", columns=target_levels).dropna(axis=1, how="all")
    if profile.empty:
        return False
    cmap = LinearSegmentedColormap.from_list(
        "target_median_percentiles",
        ["#4C78A8", "#F4F5F7", "#F58518"],
    )
    fig, ax = plt.subplots(
        figsize=(max(8, 1.15 * len(profile.columns) + 4), max(4, 0.55 * len(profile.index) + 2.5))
    )
    sns.heatmap(
        profile,
        mask=profile.isna(),
        annot=True,
        fmt=".0f",
        cmap=cmap,
        vmin=0,
        vmax=100,
        center=50,
        linewidths=0.5,
        linecolor="#FFFFFF",
        cbar_kws={"label": "Class median percentile within feature"},
        annot_kws={"fontsize": PROFILE_LABEL_FONT_SIZE},
        ax=ax,
    )
    ax.set_title("Target-Class Median Percentiles Across Numeric Features")
    ax.set_xlabel(target_column)
    ax.set_ylabel("Numeric feature")
    ax.tick_params(axis="x", labelrotation=90)
    for axis in fig.axes:
        _style_profile_labels(axis)
    for tick_label in ax.get_xticklabels():
        if tick_label.get_text() in target_colors:
            tick_label.set_color(target_colors[tick_label.get_text()])
            tick_label.set_fontweight("bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return True

def _plot_numeric_target_distribution_comparison(
    df: pd.DataFrame, columns: list[str], target_column: str, max_target_levels: int, output_path: Path
) -> bool:
    """Compare numeric feature distributions across target classes with box plots and ECDFs."""
    if not columns:
        return False
    fig, axes = plt.subplots(len(columns), 2, figsize=(15, max(4, 3.8 * len(columns))), squeeze=False)
    target_colors = _target_color_mapping(df[target_column], max_target_levels)
    for row_index, column in enumerate(columns):
        plot_df = df[[target_column, column]].copy()
        plot_df[column] = pd.to_numeric(plot_df[column], errors="coerce")
        plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna()
        plot_df[target_column] = _apply_target_color_mapping(plot_df[target_column], target_colors)
        order = [level for level in target_colors if (plot_df[target_column] == level).any()]

        box_axis, ecdf_axis = axes[row_index]
        _boxplot(
            data=plot_df,
            x=target_column,
            y=column,
            order=order,
            hue=target_column,
            hue_order=order,
            palette=target_colors,
            dodge=False,
            saturation=1,
            legend=False,
            ax=box_axis,
        )
        box_axis.set_title(f"{column}: spread by {target_column}")
        _format_vertical_category_labels(box_axis)
        for level_index, level in enumerate(order):
            values = np.sort(plot_df.loc[plot_df[target_column] == level, column].to_numpy(dtype=float))
            if values.size == 0:
                continue
            # ECDF height is the fraction of this class at or below each sorted observation.
            cumulative = np.arange(1, values.size + 1) / values.size
            ecdf_axis.step(
                values,
                cumulative,
                where="post",
                color=target_colors[level],
                linestyle=("-", "--", "-.", ":")[level_index % 4],
                label=f"{level} (n={values.size})",
            )
        ecdf_axis.set_title(f"{column}: cumulative distributions")
        ecdf_axis.set_xlabel(column)
        ecdf_axis.set_ylabel("Cumulative proportion")
        ecdf_axis.set_ylim(0, 1.02)
    fig.suptitle(f"Numeric Feature Distributions by Target: {target_column}", y=1.002)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return True

def _plot_categorical_target_composition(
    df: pd.DataFrame, columns: list[str], target_column: str, max_feature_levels: int, max_target_levels: int, output_path: Path
) -> bool:
    """Plot target-class proportions within the strongest categorical features."""
    if not columns:
        return False
    fig, axes = plt.subplots(len(columns), 1, figsize=(13, max(4, 4.2 * len(columns))), squeeze=False)
    target_colors = _target_color_mapping(df[target_column], max_target_levels)
    for axis, column in zip(axes[:, 0], columns):
        plot_df = df[[column, target_column]].dropna().copy()
        plot_df[column] = _limited_categories(plot_df[column], max_feature_levels)
        plot_df[target_column] = _apply_target_color_mapping(plot_df[target_column], target_colors)
        proportions = pd.crosstab(plot_df[column], plot_df[target_column], normalize="index")
        proportions = proportions.loc[plot_df[column].value_counts().reindex(proportions.index).sort_values().index]
        proportions.plot(
            kind="barh",
            stacked=True,
            ax=axis,
            color=[target_colors[str(level)] for level in proportions.columns],
            edgecolor="#334E48",
            linewidth=0.4,
        )
        axis.set_title(f"Target composition within {column}")
        axis.set_xlabel("Proportion within feature level")
        axis.set_ylabel(column)
        axis.set_xlim(0, 1)
        axis.legend(title=target_column, bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return True

def _plot_numeric_target_by_categorical_features(
    df: pd.DataFrame, columns: list[str], target_column: str, max_categories: int, output_path: Path
) -> bool:
    """Compare a numeric target distribution across categorical feature levels."""
    if not columns:
        return False
    fig, axes = plt.subplots(len(columns), 1, figsize=(13, max(4, 4 * len(columns))), squeeze=False)
    for axis, column in zip(axes[:, 0], columns):
        plot_df = df[[column, target_column]].copy()
        plot_df[target_column] = pd.to_numeric(plot_df[target_column], errors="coerce")
        plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna()
        plot_df[column] = _limited_categories(plot_df[column], max_categories)
        order = plot_df.groupby(column, observed=True)[target_column].median().sort_values().index
        _boxplot(data=plot_df, x=target_column, y=column, order=order, ax=axis, color="#4C78A8")
        axis.set_title(f"{target_column} distribution by {column}")
        axis.set_xlabel(target_column)
        axis.set_ylabel(column)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return True

def _plot_feature_target_association_ranking(
    associations: pd.DataFrame, max_features: int, output_path: Path
) -> None:
    """Plot the strongest univariate target associations with method-aware labels."""
    methods = associations["method"].dropna().unique().tolist()
    fig, axes = plt.subplots(len(methods), 1, figsize=(11, max(4.5, 4.2 * len(methods))), squeeze=False)
    for axis, method in zip(axes[:, 0], methods):
        data = (
            associations.loc[associations["method"] == method]
            .head(max_features)
            .sort_values("effect_size", ascending=True)
            .copy()
        )
        colors = data["feature_type"].map({"numeric": "#4C78A8", "categorical": "#E3BA22"}).fillna("#9AA5B1")
        axis.barh(data["feature"], data["effect_size"], color=colors, edgecolor="#334E48", linewidth=0.5)
        axis.set_title(method.replace("_", " ").title())
        axis.set_xlabel("Effect size")
        axis.set_ylabel("")
        _style_profile_labels(axis)
        for position, value in enumerate(data["effect_size"]):
            axis.text(value, position, f" {value:.3f}", va="center", fontsize=PROFILE_LABEL_FONT_SIZE)
    handles = [
        Line2D([0], [0], color="#4C78A8", linewidth=8, label="Numeric feature"),
        Line2D([0], [0], color="#E3BA22", linewidth=8, label="Categorical feature"),
    ]
    axes[0, 0].legend(handles=handles, loc="lower right", fontsize=PROFILE_LABEL_FONT_SIZE)
    fig.suptitle("Feature-Target Association Screening (Rank Within Method)", y=1.002)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

def _plot_geometry_target_heatmap(
    df: pd.DataFrame,
    target_column: str,
    target_task: str | None,
    max_target_levels: int,
    output_path: Path,
) -> bool:
    """Plot representative points colored by target, with the legend outside the map."""
    try:
        import geopandas as gpd

        plot_df = df[[target_column, "geometry"]].dropna(subset=["geometry"]).copy()
        plot_df = gpd.GeoDataFrame(plot_df, geometry="geometry", crs=getattr(df, "crs", None))
        plot_df = plot_df.loc[~plot_df.geometry.is_empty].copy()
        if plot_df.empty or plot_df[target_column].notna().sum() == 0:
            return False
        plot_df.geometry = plot_df.geometry.representative_point()
        fig, ax = plt.subplots(figsize=(9, 8))
        if target_task == "regression":
            plot_df.plot(
                column=target_column,
                ax=ax,
                cmap="viridis",
                legend=True,
                legend_kwds={"shrink": 0.7, "label": target_column},
                markersize=18,
                edgecolor="#2f4b4f",
                linewidth=0.3,
                alpha=0.85,
                missing_kwds={"color": "#e5e7eb", "label": "Missing"},
            )
        else:
            target_colors = _target_color_mapping(df[target_column], max_target_levels)
            plot_df[target_column] = _apply_target_color_mapping(plot_df[target_column], target_colors)
            present_levels = [level for level in target_colors if (plot_df[target_column] == level).any()]
            has_missing = plot_df[target_column].isna().any()
            plot_df.plot(
                ax=ax,
                color=plot_df[target_column].map(target_colors).fillna("#e5e7eb"),
                markersize=18,
                edgecolor="#2f4b4f",
                linewidth=0.3,
                alpha=0.85,
            )
            handles = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="",
                    markerfacecolor=target_colors[level],
                    markeredgecolor="#2f4b4f",
                    label=level,
                )
                for level in present_levels
            ]
            if has_missing:
                handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        linestyle="",
                        markerfacecolor="#e5e7eb",
                        markeredgecolor="#2f4b4f",
                        label="Missing",
                    )
                )
            ax.legend(
                handles=handles,
                title=target_column,
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
                fontsize=8,
                title_fontsize=8,
                borderaxespad=0,
            )
        for axis in fig.axes:
            _style_profile_labels(axis)
        ax.set_title(f"Target Heatmap: {target_column}")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(output_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        return True
    except Exception:
        logging.getLogger(__name__).debug("Error: _plot_geometry_target_heatmap failed; using its fallback.", exc_info=True)
        return False

def _plot_geometry_overview(df: pd.DataFrame, output_path: Path) -> bool:
    """Plot a simple geometry overview for GeoDataFrames."""
    try:
        fig, ax = plt.subplots(figsize=(8, 8))
        df.plot(ax=ax, color="#72b7b2", edgecolor="#2f4b4f", alpha=0.75)
        ax.set_title("Geometry Overview")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(output_path, dpi=140)
        plt.close(fig)
        return True
    except Exception:
        logging.getLogger(__name__).debug("Error: _plot_geometry_overview failed; using its fallback.", exc_info=True)
        return False
