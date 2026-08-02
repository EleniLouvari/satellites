"""Matplotlib and seaborn visual outputs for EDA."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas.plotting import scatter_matrix
from scipy import stats

from ..core.config import EDAConfig
from ..core.io import ensure_dir


def create_eda_plots(df: pd.DataFrame, artifacts: dict, config: EDAConfig) -> dict[str, Path]:
    """Create standard EDA plots and return a map of plot names to paths."""
    if not config.include_plots:
        return {}
    plots_dir = ensure_dir(config.plots_dir)
    plot_paths: dict[str, Path] = {}

    missing_path = plots_dir / "missing_values.png"
    _plot_missing_values(artifacts["column_profile"], missing_path)
    plot_paths["missing_values"] = missing_path

    if not artifacts.get("missingness_summary", pd.DataFrame()).empty:
        missing_heatmap_path = plots_dir / "missingness_heatmap.png"
        _plot_missingness_heatmap(df, missing_heatmap_path)
        plot_paths["missingness_heatmap"] = missing_heatmap_path

    numeric_columns = artifacts["numeric_summary"]["column"].tolist() if not artifacts["numeric_summary"].empty else []
    categorical_columns = artifacts["categorical_summary"]["column"].tolist() if not artifacts["categorical_summary"].empty else []

    if not artifacts.get("data_quality_flags", pd.DataFrame()).empty:
        quality_path = plots_dir / "data_quality_flags.png"
        _plot_data_quality_flags(artifacts["data_quality_flags"], quality_path)
        plot_paths["data_quality_flags"] = quality_path

    if not artifacts.get("multicollinearity", pd.DataFrame()).empty:
        vif_path = plots_dir / "vif_multicollinearity.png"
        _plot_vif_summary(artifacts["multicollinearity"], vif_path)
        plot_paths["vif_multicollinearity"] = vif_path

    if numeric_columns:
        numeric_path = plots_dir / "numeric_distributions.png"
        _plot_numeric_distributions(df, numeric_columns[: config.max_columns_for_plots], numeric_path)
        plot_paths["numeric_distributions"] = numeric_path

        qq_path = plots_dir / "numeric_qq_plots.png"
        _plot_qq_plots(df, numeric_columns[: config.max_columns_for_plots], qq_path, config)
        plot_paths["numeric_qq_plots"] = qq_path

    if len(numeric_columns) >= 2:
        scatter_path = plots_dir / "numeric_scatter_matrix.png"
        _plot_scatter_matrix(df, numeric_columns[: config.max_pairplot_features], scatter_path)
        plot_paths["numeric_scatter_matrix"] = scatter_path

    if categorical_columns:
        categorical_path = plots_dir / "categorical_distributions.png"
        _plot_categorical_distributions(
            df,
            categorical_columns[: config.max_columns_for_plots],
            config.max_categories,
            categorical_path,
        )
        plot_paths["categorical_distributions"] = categorical_path

    if not artifacts.get("categorical_associations", pd.DataFrame()).empty:
        cat_assoc_path = plots_dir / "categorical_association_heatmap.png"
        _plot_categorical_association_heatmap(artifacts["categorical_associations"], cat_assoc_path)
        plot_paths["categorical_association_heatmap"] = cat_assoc_path

    if not artifacts.get("categorical_target_tests", pd.DataFrame()).empty:
        cat_target_path = plots_dir / "categorical_target_associations.png"
        _plot_association_bars(artifacts["categorical_target_tests"], "column", "cramers_v", "Categorical Features vs Target (Cramer's V)", cat_target_path)
        plot_paths["categorical_target_associations"] = cat_target_path

    if not artifacts.get("missingness_target_tests", pd.DataFrame()).empty:
        missing_target_path = plots_dir / "missingness_target_associations.png"
        _plot_association_bars(artifacts["missingness_target_tests"], "column", "cramers_v", "Missingness vs Target (Cramer's V)", missing_target_path)
        plot_paths["missingness_target_associations"] = missing_target_path

    pearson = artifacts["correlation_matrices"]["pearson"]
    if not pearson.empty:
        corr_path = plots_dir / "pearson_correlation_heatmap.png"
        _plot_correlation_heatmap(pearson, corr_path, "Pearson Correlation Heatmap")
        plot_paths["pearson_correlation_heatmap"] = corr_path

    spearman = artifacts["correlation_matrices"]["spearman"]
    if not spearman.empty:
        spearman_path = plots_dir / "spearman_correlation_heatmap.png"
        _plot_correlation_heatmap(spearman, spearman_path, "Spearman Correlation Heatmap")
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
        _plot_target_distribution(df[config.target_column], target_path)
        plot_paths["target_distribution"] = target_path
        if numeric_columns and not pd.api.types.is_numeric_dtype(df[config.target_column]):
            target_boxplot_path = plots_dir / "numeric_by_target_boxplots.png"
            _plot_numeric_by_target_boxplots(df, numeric_columns[: config.max_columns_for_plots], config.target_column, target_boxplot_path)
            plot_paths["numeric_by_target_boxplots"] = target_boxplot_path
        if not artifacts.get("numeric_target_correlations", pd.DataFrame()).empty:
            numeric_target_corr_path = plots_dir / "numeric_target_correlations.png"
            _plot_numeric_target_correlations(artifacts["numeric_target_correlations"], numeric_target_corr_path)
            plot_paths["numeric_target_correlations"] = numeric_target_corr_path
            target_scatter_path = plots_dir / "numeric_target_scatter.png"
            top_columns = artifacts["numeric_target_correlations"]["column"].head(config.max_target_scatter_features).tolist()
            if _plot_numeric_target_scatter(df, top_columns, config.target_column, target_scatter_path):
                plot_paths["numeric_target_scatter"] = target_scatter_path

    if "geometry" in df.columns and hasattr(df, "plot") and config.include_geospatial:
        map_path = plots_dir / "geometry_overview.png"
        if _plot_geometry_overview(df, map_path):
            plot_paths["geometry_overview"] = map_path
        if config.target_column and config.target_column in df.columns:
            target_heatmap_path = plots_dir / "geometry_target_heatmap.png"
            if _plot_geometry_target_heatmap(df, config.target_column, target_heatmap_path):
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

def _plot_missing_values(column_profile: pd.DataFrame, output_path: Path) -> None:
    """Plot missing percentage by column."""
    data = column_profile.sort_values("missing_percent", ascending=True)
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


def _plot_missingness_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    """Plot a row-sampled missingness heatmap."""
    missing = df.isna()
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


def _plot_numeric_distributions(df: pd.DataFrame, columns: list[str], output_path: Path) -> None:
    """Plot histogram and boxplot pairs for numeric columns."""
    rows = len(columns)
    fig, axes = plt.subplots(rows, 2, figsize=(12, max(3, 3 * rows)))
    if rows == 1:
        axes = [axes]
    for axes_row, column in zip(axes, columns):
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        sns.histplot(values, kde=True, ax=axes_row[0], color="#4c78a8")
        axes_row[0].set_title(f"{column} histogram")
        sns.boxplot(x=values, ax=axes_row[1], color="#f58518")
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


def _plot_categorical_association_heatmap(associations: pd.DataFrame, output_path: Path) -> None:
    """Plot Cramer's V association matrix from pairwise results."""
    features = sorted(set(associations["feature_1"]).union(set(associations["feature_2"])))
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


def _plot_correlation_heatmap(correlation_matrix: pd.DataFrame, output_path: Path, title: str) -> None:
    """Plot a correlation heatmap."""
    size = max(6, min(16, 0.55 * len(correlation_matrix.columns) + 4))
    fig, ax = plt.subplots(figsize=(size, size))
    sns.heatmap(correlation_matrix, cmap="vlag", center=0, annot=len(correlation_matrix.columns) <= 10, fmt=".2f", ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def _plot_vif_summary(vif_summary: pd.DataFrame, output_path: Path) -> None:
    """Plot VIF values for numeric features."""
    data = vif_summary.copy()
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=["vif"])
    if data.empty:
        return
    data = data.sort_values("vif", ascending=True).tail(25)
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


def _plot_association_bars(table: pd.DataFrame, label_column: str, value_column: str, title: str, output_path: Path) -> None:
    """Plot sorted association strengths from a table."""
    data = table[[label_column, value_column]].dropna().sort_values(value_column, ascending=True).tail(25)
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


def _plot_numeric_target_correlations(correlations: pd.DataFrame, output_path: Path) -> None:
    """Plot strongest numeric-target Spearman correlations."""
    data = correlations[["column", "spearman_correlation"]].dropna().copy()
    if data.empty:
        return
    data["abs_correlation"] = data["spearman_correlation"].abs()
    data = data.sort_values("abs_correlation", ascending=True).tail(25)
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
            sns.regplot(data=plot_df, x=column, y=target_column, ax=ax, scatter_kws={"alpha": 0.55, "s": 26}, line_kws={"color": "#e45756"})
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


def _plot_target_distribution(series: pd.Series, output_path: Path) -> None:
    """Plot target distribution."""
    fig, ax = plt.subplots(figsize=(10, 5))
    if pd.api.types.is_numeric_dtype(series) and series.nunique(dropna=True) > 20:
        sns.histplot(series.dropna(), kde=True, ax=ax, color="#b279a2")
        ax.set_xlabel(series.name)
    else:
        counts = series.dropna().astype(str).value_counts().head(30).sort_values()
        sns.barplot(x=counts.values, y=counts.index, ax=ax, color="#b279a2")
        ax.set_xlabel("Count")
        ax.set_ylabel("")
    ax.set_title("Target Distribution")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def _plot_numeric_by_target_boxplots(df: pd.DataFrame, columns: list[str], target_column: str, output_path: Path) -> None:
    """Plot numeric feature distributions by categorical target."""
    rows = len(columns)
    fig, axes = plt.subplots(rows, 1, figsize=(12, max(3.2, 3.2 * rows)))
    if rows == 1:
        axes = [axes]
    for ax, column in zip(axes, columns):
        plot_df = df[[target_column, column]].dropna().copy()
        plot_df[column] = pd.to_numeric(plot_df[column], errors="coerce")
        plot_df = plot_df.dropna()
        sns.boxplot(data=plot_df, x=target_column, y=column, ax=ax, color="#b279a2")
        ax.set_title(f"{column} by {target_column}")
        ax.tick_params(axis="x", labelrotation=30)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def _plot_geometry_target_heatmap(df: pd.DataFrame, target_column: str, output_path: Path) -> bool:
    """Plot geometries colored by the configured target column."""
    try:
        plot_df = df[[target_column, "geometry"]].dropna(subset=["geometry"]).copy()
        if plot_df.empty or plot_df[target_column].notna().sum() == 0:
            return False
        fig, ax = plt.subplots(figsize=(9, 8))
        if pd.api.types.is_numeric_dtype(plot_df[target_column]):
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
            plot_df[target_column] = plot_df[target_column].astype(str)
            plot_df.plot(
                column=target_column,
                ax=ax,
                categorical=True,
                legend=True,
                cmap="tab20",
                edgecolor="#2f4b4f",
                linewidth=0.3,
                alpha=0.85,
                missing_kwds={"color": "#e5e7eb", "label": "Missing"},
            )
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
