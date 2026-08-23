"""Matplotlib and seaborn visual outputs for EDA."""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from pandas.plotting import scatter_matrix
from scipy import stats

from ..core.config import EDAConfig
from ..core.io import ensure_dir

TARGET_PALETTE = ["#4C78A8", "#E3BA22", "#F58518", "#7A9A48", "#B279A2", "#72B7B2", "#9C755F", "#BAB0AC"]


def _boxplot(**kwargs) -> None:
    """Draw a seaborn box plot without its Matplotlib orientation transition warning."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="vert: bool will be deprecated.*", category=PendingDeprecationWarning)
        sns.boxplot(**kwargs)


def _ordered_available_features(ordered_features: list[str], table: pd.DataFrame, max_features: int) -> list[str]:
    """Keep the highest-priority features that are available in a plot-specific statistical table."""
    available = set(table["column"].tolist()) if "column" in table.columns else set()
    return [feature for feature in ordered_features if feature in available][:max_features]


def create_eda_plots(df: pd.DataFrame, artifacts: dict, config: EDAConfig) -> dict[str, Path]:
    """Create standard EDA plots and return a map of plot names to paths."""
    if not config.include_plots:
        return {}
    plots_dir = ensure_dir(config.plots_dir)
    plot_paths: dict[str, Path] = {}
    target_task = artifacts.get("summary", {}).get("target_task")
    plot_feature_order = artifacts.get("plot_feature_order", {})
    ordered_features = plot_feature_order.get("all", [])
    numeric_feature_columns = plot_feature_order.get("numeric", [])
    categorical_columns = plot_feature_order.get("categorical", [])
    selected_numeric = numeric_feature_columns[: config.max_features_per_plot]
    selected_categorical = categorical_columns[: config.max_features_per_plot]

    missing_path = plots_dir / "missing_values.png"
    if _plot_missing_values(artifacts["column_profile"], ordered_features, config.max_features_per_plot, missing_path):
        plot_paths["missing_values"] = missing_path

    if not artifacts.get("missingness_summary", pd.DataFrame()).empty:
        missing_heatmap_path = plots_dir / "missingness_heatmap.png"
        if _plot_missingness_heatmap(df, ordered_features, config.max_features_per_plot, missing_heatmap_path):
            plot_paths["missingness_heatmap"] = missing_heatmap_path

    numeric_columns = artifacts["numeric_summary"]["column"].tolist() if not artifacts["numeric_summary"].empty else []

    if not artifacts.get("data_quality_flags", pd.DataFrame()).empty:
        quality_path = plots_dir / "data_quality_flags.png"
        _plot_data_quality_flags(artifacts["data_quality_flags"], quality_path)
        plot_paths["data_quality_flags"] = quality_path

    if not artifacts.get("multicollinearity", pd.DataFrame()).empty:
        vif_path = plots_dir / "vif_multicollinearity.png"
        _plot_vif_summary(artifacts["multicollinearity"], config.max_features_per_plot, vif_path)
        plot_paths["vif_multicollinearity"] = vif_path

    if selected_numeric:
        numeric_path = plots_dir / "numeric_distributions.png"
        _plot_numeric_distributions(df, selected_numeric, numeric_path)
        plot_paths["numeric_distributions"] = numeric_path

    if selected_numeric:
        qq_dir = ensure_dir(plots_dir / "qq_plots")
        qq_path = qq_dir / "numeric_qq_plots.png"
        _plot_qq_plots(df, selected_numeric, qq_path, config)
        plot_paths["numeric_qq_plots"] = qq_path

    if len(selected_numeric) >= 2:
        scatter_path = plots_dir / "numeric_scatter_matrix.png"
        _plot_scatter_matrix(df, selected_numeric, scatter_path)
        plot_paths["numeric_scatter_matrix"] = scatter_path

    if selected_categorical:
        categorical_path = plots_dir / "categorical_distributions.png"
        _plot_categorical_distributions(df, selected_categorical, config.max_categories, categorical_path)
        plot_paths["categorical_distributions"] = categorical_path

    if not artifacts.get("categorical_associations", pd.DataFrame()).empty:
        cat_assoc_path = plots_dir / "categorical_association_heatmap.png"
        if _plot_categorical_association_heatmap(
            artifacts["categorical_associations"], selected_categorical, cat_assoc_path
        ):
            plot_paths["categorical_association_heatmap"] = cat_assoc_path

    if not artifacts.get("categorical_target_tests", pd.DataFrame()).empty:
        cat_target_path = plots_dir / "categorical_target_associations.png"
        _plot_association_bars(
            artifacts["categorical_target_tests"],
            "column",
            "cramers_v_bias_corrected",
            "Categorical Features vs Target (Bias-Corrected Cramer's V)",
            config.max_features_per_plot,
            cat_target_path,
        )
        plot_paths["categorical_target_associations"] = cat_target_path

    if not artifacts.get("missingness_target_tests", pd.DataFrame()).empty:
        missing_target_path = plots_dir / "missingness_target_associations.png"
        _plot_association_bars(
            artifacts["missingness_target_tests"],
            "column",
            "cramers_v_bias_corrected",
            "Missingness vs Target (Bias-Corrected Cramer's V)",
            config.max_features_per_plot,
            missing_target_path,
        )
        plot_paths["missingness_target_associations"] = missing_target_path

    pearson = artifacts["correlation_matrices"]["pearson"]
    if not pearson.empty and selected_numeric:
        corr_path = plots_dir / "pearson_correlation_heatmap.png"
        _plot_correlation_heatmap(pearson.loc[selected_numeric, selected_numeric], corr_path, "Pearson Correlation Heatmap")
        plot_paths["pearson_correlation_heatmap"] = corr_path

    spearman = artifacts["correlation_matrices"]["spearman"]
    if not spearman.empty and selected_numeric:
        spearman_path = plots_dir / "spearman_correlation_heatmap.png"
        _plot_correlation_heatmap(
            spearman.loc[selected_numeric, selected_numeric], spearman_path, "Spearman Correlation Heatmap"
        )
        plot_paths["spearman_correlation_heatmap"] = spearman_path

    if not artifacts.get("pca_summary", pd.DataFrame()).empty:
        pca_variance_path = plots_dir / "pca_explained_variance.png"
        _plot_pca_explained_variance(artifacts["pca_summary"], pca_variance_path)
        plot_paths["pca_explained_variance"] = pca_variance_path
        pca_scores_path = plots_dir / "pca_score_plot.png"
        if _plot_pca_scores(df, numeric_columns, config, pca_scores_path):
            plot_paths["pca_score_plot"] = pca_scores_path

    if config.target_column and config.target_column in df.columns:
        target_path = plots_dir / "target_distribution.png"
        _plot_target_distribution(
            df[config.target_column], target_task, config.max_target_levels_for_plots, target_path
        )
        plot_paths["target_distribution"] = target_path
        if target_task == "classification" and not artifacts.get("numeric_target_tests", pd.DataFrame()).empty:
            target_features = artifacts.get("numeric_target_tests", pd.DataFrame())
            selected_columns = _ordered_available_features(numeric_feature_columns, target_features, config.max_features_per_plot)
            target_profile_path = plots_dir / "numeric_target_median_percentile_heatmap.png"
            if _plot_numeric_target_median_percentile_heatmap(
                df,
                selected_columns,
                config.target_column,
                config.max_target_levels_for_plots,
                target_profile_path,
            ):
                plot_paths["numeric_target_median_percentile_heatmap"] = target_profile_path
            distribution_path = plots_dir / "numeric_target_distribution_comparison.png"
            if _plot_numeric_target_distribution_comparison(
                df, selected_columns, config.target_column, config.max_target_levels_for_plots, distribution_path
            ):
                plot_paths["numeric_target_distribution_comparison"] = distribution_path
        if target_task == "classification" and not artifacts.get("categorical_target_tests", pd.DataFrame()).empty:
            composition_path = plots_dir / "categorical_target_composition.png"
            selected_columns = _ordered_available_features(
                categorical_columns, artifacts["categorical_target_tests"], config.max_features_per_plot
            )
            if _plot_categorical_target_composition(
                df,
                selected_columns,
                config.target_column,
                config.max_categories,
                config.max_target_levels_for_plots,
                composition_path,
            ):
                plot_paths["categorical_target_composition"] = composition_path
        if target_task == "regression" and not artifacts.get("categorical_numeric_target_tests", pd.DataFrame()).empty:
            categorical_target_path = plots_dir / "numeric_target_by_categorical_features.png"
            selected_columns = _ordered_available_features(
                categorical_columns, artifacts["categorical_numeric_target_tests"], config.max_features_per_plot
            )
            if _plot_numeric_target_by_categorical_features(
                df, selected_columns, config.target_column, config.max_categories, categorical_target_path
            ):
                plot_paths["numeric_target_by_categorical_features"] = categorical_target_path
        if not artifacts.get("numeric_target_correlations", pd.DataFrame()).empty:
            numeric_target_corr_path = plots_dir / "numeric_target_correlations.png"
            _plot_numeric_target_correlations(
                artifacts["numeric_target_correlations"], config.max_features_per_plot, numeric_target_corr_path
            )
            plot_paths["numeric_target_correlations"] = numeric_target_corr_path
            target_scatter_path = plots_dir / "numeric_target_scatter.png"
            top_columns = _ordered_available_features(
                numeric_feature_columns, artifacts["numeric_target_correlations"], config.max_features_per_plot
            )
            if _plot_numeric_target_scatter(df, top_columns, config.target_column, target_scatter_path):
                plot_paths["numeric_target_scatter"] = target_scatter_path
        if not artifacts.get("feature_target_associations", pd.DataFrame()).empty:
            feature_ranking_path = plots_dir / "feature_target_association_ranking.png"
            _plot_feature_target_association_ranking(
                artifacts["feature_target_associations"], config.max_features_per_plot, feature_ranking_path
            )
            plot_paths["feature_target_association_ranking"] = feature_ranking_path

    if "geometry" in df.columns and hasattr(df, "plot") and config.include_geospatial:
        map_path = plots_dir / "geometry_overview.png"
        if _plot_geometry_overview(df, map_path):
            plot_paths["geometry_overview"] = map_path
        if config.target_column and config.target_column in df.columns:
            target_heatmap_path = plots_dir / "geometry_target_heatmap.png"
            if _plot_geometry_target_heatmap(
                df,
                config.target_column,
                target_task,
                config.max_target_levels_for_plots,
                target_heatmap_path,
            ):
                plot_paths["geometry_target_heatmap"] = target_heatmap_path

    return plot_paths


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
        ax.tick_params(axis="x", labelrotation=45)
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
    """Plot a correlation heatmap."""
    size = max(6, min(16, 0.55 * len(correlation_matrix.columns) + 4))
    fig, ax = plt.subplots(figsize=(size, size))
    sns.heatmap(correlation_matrix, cmap="vlag", center=0, annot=len(correlation_matrix.columns) <= 10, fmt=".2f", ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
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
    ax.legend()
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


def _limited_categories(series: pd.Series, max_categories: int) -> pd.Series:
    """Keep the most frequent categories and group the remaining levels as Other."""
    values = series.astype("string")
    counts = values.value_counts()
    if len(counts) <= max_categories:
        return values
    top_categories = counts.head(max_categories - 1).index
    return values.where(values.isna() | values.isin(top_categories), "Other")


def _target_color_mapping(series: pd.Series, max_target_levels: int) -> dict[str, str]:
    """Assign stable report-wide colors to categorical target levels."""
    levels = series.astype("string").value_counts().index.astype(str).tolist()
    if len(levels) > max_target_levels:
        levels = [*levels[: max_target_levels - 1], "Other"]
    return {str(level): TARGET_PALETTE[index % len(TARGET_PALETTE)] for index, level in enumerate(levels)}


def _apply_target_color_mapping(series: pd.Series, target_colors: dict[str, str]) -> pd.Series:
    """Apply the report-wide target grouping without re-ranking levels in a subset."""
    values = series.astype("string")
    if "Other" not in target_colors:
        return values
    retained_levels = [level for level in target_colors if level != "Other"]
    return values.where(values.isna() | values.isin(retained_levels), "Other")


def _plot_numeric_target_median_percentile_heatmap(
    df: pd.DataFrame, columns: list[str], target_column: str, max_target_levels: int, output_path: Path
) -> bool:
    """Compare target-class medians across numeric features on a common percentile scale."""
    if not columns:
        return False
    target_colors = _target_color_mapping(df[target_column], max_target_levels)
    target_levels = list(target_colors)
    percentile_rows: dict[str, dict[str, float]] = {}
    for column in columns:
        plot_df = df[[target_column, column]].copy()
        plot_df[column] = pd.to_numeric(plot_df[column], errors="coerce")
        plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna()
        if plot_df.empty:
            continue
        plot_df[target_column] = _apply_target_color_mapping(plot_df[target_column], target_colors)
        feature_values = plot_df[column].to_numpy(dtype=float)
        percentile_rows[column] = {
            level: (
                float(
                    stats.percentileofscore(
                        feature_values,
                        plot_df.loc[plot_df[target_column] == level, column].median(),
                        kind="mean",
                    )
                )
                if (plot_df[target_column] == level).any()
                else np.nan
            )
            for level in target_levels
        }
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
        ax=ax,
    )
    ax.set_title("Target-Class Median Percentiles Across Numeric Features")
    ax.set_xlabel(target_column)
    ax.set_ylabel("Numeric feature")
    ax.tick_params(axis="x", labelrotation=30)
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
        box_axis.tick_params(axis="x", labelrotation=30)
        for level_index, level in enumerate(order):
            values = np.sort(plot_df.loc[plot_df[target_column] == level, column].to_numpy(dtype=float))
            if values.size == 0:
                continue
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
        ecdf_axis.legend(loc="best", fontsize=8)
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
        for position, value in enumerate(data["effect_size"]):
            axis.text(value, position, f" {value:.3f}", va="center", fontsize=8)
    handles = [
        Line2D([0], [0], color="#4C78A8", linewidth=8, label="Numeric feature"),
        Line2D([0], [0], color="#E3BA22", linewidth=8, label="Categorical feature"),
    ]
    axes[0, 0].legend(handles=handles, loc="lower right")
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
    """Plot geometries colored by the configured target column."""
    try:
        plot_df = df[[target_column, "geometry"]].dropna(subset=["geometry"]).copy()
        if plot_df.empty or plot_df[target_column].notna().sum() == 0:
            return False
        fig, ax = plt.subplots(figsize=(9, 8))
        if target_task == "regression":
            plot_df.plot(
                column=target_column,
                ax=ax,
                cmap="viridis",
                legend=True,
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
                edgecolor="#2f4b4f",
                linewidth=0.3,
                alpha=0.85,
            )
            handles = [
                Line2D(
                    [0],
                    [0],
                    marker="s",
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
                        marker="s",
                        linestyle="",
                        markerfacecolor="#e5e7eb",
                        markeredgecolor="#2f4b4f",
                        label="Missing",
                    )
                )
            ax.legend(handles=handles, title=target_column)
        ax.set_title(f"Target Heatmap: {target_column}")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(output_path, dpi=140)
        plt.close(fig)
        return True
    except Exception:
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
        return False


def _prepared_numeric_matrix(df: pd.DataFrame, numeric_columns: list[str], config: EDAConfig) -> pd.DataFrame:
    """Return standardized numeric matrix for PCA plots."""
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
