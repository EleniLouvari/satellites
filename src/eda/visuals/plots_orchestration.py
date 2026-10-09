"""Orchestration helpers for assembling EDA plots."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..core.config import EDAConfig
from ..core.io import ensure_dir
from .plots_common import _artifact_has_rows, _ordered_available_features
from .plots_renderers import (
    _plot_association_bars,
    _plot_categorical_association_heatmap,
    _plot_categorical_distributions,
    _plot_categorical_target_composition,
    _plot_correlation_heatmap,
    _plot_data_quality_flags,
    _plot_feature_target_association_ranking,
    _plot_geometry_overview,
    _plot_geometry_target_heatmap,
    _plot_missing_values,
    _plot_missingness_heatmap,
    _plot_numeric_distributions,
    _plot_numeric_target_by_categorical_features,
    _plot_numeric_target_correlations,
    _plot_numeric_target_distribution_comparison,
    _plot_numeric_target_median_percentile_heatmap,
    _plot_numeric_target_scatter,
    _plot_pca_explained_variance,
    _plot_pca_scores,
    _plot_qq_plots,
    _plot_scatter_matrix,
    _plot_target_distribution,
    _plot_vif_summary,
)

# Keep this helper focused on a single transformation so the reporting pipeline stays easy to follow.



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
    # Apply display limits after screening order; the complete statistical tables remain available.
    selected_numeric = numeric_feature_columns[: config.max_features_per_plot]
    selected_categorical = categorical_columns[: config.max_features_per_plot]

    numeric_columns = artifacts["numeric_summary"]["column"].tolist() if not artifacts["numeric_summary"].empty else []

    _create_missingness_quality_plots(df, artifacts, ordered_features, config, plots_dir, plot_paths)
    _create_distribution_plots(df, selected_numeric, selected_categorical, config, plots_dir, plot_paths)
    _create_association_correlation_plots(
        artifacts, selected_numeric, selected_categorical, config, plots_dir, plot_paths
    )
    _create_pca_plots(df, artifacts, numeric_columns, config, plots_dir, plot_paths)

    _create_target_plots(
        df=df,
        artifacts=artifacts,
        config=config,
        plots_dir=plots_dir,
        plot_paths=plot_paths,
        target_task=target_task,
        numeric_feature_columns=numeric_feature_columns,
        categorical_columns=categorical_columns,
    )
    _create_geospatial_plots(
        df=df,
        config=config,
        plots_dir=plots_dir,
        plot_paths=plot_paths,
        target_task=target_task,
    )

    return plot_paths

def _create_missingness_quality_plots(
    df: pd.DataFrame,
    artifacts: dict,
    ordered_features: list[str],
    config: EDAConfig,
    plots_dir: Path,
    plot_paths: dict[str, Path],
) -> None:
    """Create missingness and quality-control plots."""
    missing_path = plots_dir / "missing_values.png"
    # Register conditional plots only when the renderer reports that it produced an image.
    if _plot_missing_values(artifacts["column_profile"], ordered_features, config.max_features_per_plot, missing_path):
        plot_paths["missing_values"] = missing_path

    if _artifact_has_rows(artifacts, "missingness_summary"):
        missing_heatmap_path = plots_dir / "missingness_heatmap.png"
        if _plot_missingness_heatmap(df, ordered_features, config.max_features_per_plot, missing_heatmap_path):
            plot_paths["missingness_heatmap"] = missing_heatmap_path

    if _artifact_has_rows(artifacts, "data_quality_flags"):
        quality_path = plots_dir / "data_quality_flags.png"
        _plot_data_quality_flags(artifacts["data_quality_flags"], quality_path)
        plot_paths["data_quality_flags"] = quality_path

    if _artifact_has_rows(artifacts, "multicollinearity"):
        vif_path = plots_dir / "vif_multicollinearity.png"
        _plot_vif_summary(artifacts["multicollinearity"], config.max_features_per_plot, vif_path)
        plot_paths["vif_multicollinearity"] = vif_path

def _create_distribution_plots(
    df: pd.DataFrame,
    selected_numeric: list[str],
    selected_categorical: list[str],
    config: EDAConfig,
    plots_dir: Path,
    plot_paths: dict[str, Path],
) -> None:
    """Create core numeric and categorical distribution plots."""
    if selected_numeric:
        numeric_path = plots_dir / "numeric_distributions.png"
        _plot_numeric_distributions(df, selected_numeric, numeric_path)
        plot_paths["numeric_distributions"] = numeric_path

        qq_dir = ensure_dir(plots_dir / "qq_plots")
        qq_path = qq_dir / "numeric_qq_plots.png"
        _plot_qq_plots(df, selected_numeric, qq_path, config)
        plot_paths["numeric_qq_plots"] = qq_path

    # A scatter matrix needs a pair of numeric features, unlike the univariate charts above.
    if len(selected_numeric) >= 2:
        scatter_path = plots_dir / "numeric_scatter_matrix.png"
        _plot_scatter_matrix(df, selected_numeric, scatter_path)
        plot_paths["numeric_scatter_matrix"] = scatter_path

    if selected_categorical:
        categorical_path = plots_dir / "categorical_distributions.png"
        _plot_categorical_distributions(df, selected_categorical, config.max_categories, categorical_path)
        plot_paths["categorical_distributions"] = categorical_path

def _create_association_correlation_plots(
    artifacts: dict,
    selected_numeric: list[str],
    selected_categorical: list[str],
    config: EDAConfig,
    plots_dir: Path,
    plot_paths: dict[str, Path],
) -> None:
    """Create association summaries and correlation heatmaps."""
    if _artifact_has_rows(artifacts, "categorical_associations"):
        cat_assoc_path = plots_dir / "categorical_association_heatmap.png"
        if _plot_categorical_association_heatmap(
            artifacts["categorical_associations"], selected_categorical, cat_assoc_path
        ):
            plot_paths["categorical_association_heatmap"] = cat_assoc_path

    if _artifact_has_rows(artifacts, "categorical_target_tests"):
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

    if _artifact_has_rows(artifacts, "missingness_target_tests"):
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

def _create_pca_plots(
    df: pd.DataFrame,
    artifacts: dict,
    numeric_columns: list[str],
    config: EDAConfig,
    plots_dir: Path,
    plot_paths: dict[str, Path],
) -> None:
    """Create PCA summary plots when PCA diagnostics are available."""
    # Skip PCA visuals when profiling could not produce a usable PCA summary.
    if not _artifact_has_rows(artifacts, "pca_summary"):
        return

    pca_variance_path = plots_dir / "pca_explained_variance.png"
    _plot_pca_explained_variance(artifacts["pca_summary"], pca_variance_path)
    plot_paths["pca_explained_variance"] = pca_variance_path
    pca_scores_path = plots_dir / "pca_score_plot.png"
    if _plot_pca_scores(df, numeric_columns, config, pca_scores_path):
        plot_paths["pca_score_plot"] = pca_scores_path

def _create_target_plots(
    df: pd.DataFrame,
    artifacts: dict,
    config: EDAConfig,
    plots_dir: Path,
    plot_paths: dict[str, Path],
    target_task: str | None,
    numeric_feature_columns: list[str],
    categorical_columns: list[str],
) -> None:
    """Create plots that depend on an available target column."""
    if not (config.target_column and config.target_column in df.columns):
        return

    target_path = plots_dir / "target_distribution.png"
    _plot_target_distribution(df[config.target_column], target_task, config.max_target_levels_for_plots, target_path)
    plot_paths["target_distribution"] = target_path

    _create_classification_target_plots(
        df, artifacts, config, plots_dir, plot_paths, target_task, numeric_feature_columns, categorical_columns
    )
    _create_regression_target_plots(df, artifacts, config, plots_dir, plot_paths, target_task, categorical_columns)
    _create_numeric_target_correlation_plots(df, artifacts, config, plots_dir, plot_paths, numeric_feature_columns)
    _create_feature_target_ranking_plot(artifacts, config, plots_dir, plot_paths)

def _create_classification_target_plots(
    df: pd.DataFrame,
    artifacts: dict,
    config: EDAConfig,
    plots_dir: Path,
    plot_paths: dict[str, Path],
    target_task: str | None,
    numeric_feature_columns: list[str],
    categorical_columns: list[str],
) -> None:
    """Create classification-target plots from numeric and categorical diagnostics."""
    if target_task != "classification":
        return
    _create_numeric_classification_target_plots(df, artifacts, config, plots_dir, plot_paths, numeric_feature_columns)
    _create_categorical_classification_target_plots(df, artifacts, config, plots_dir, plot_paths, categorical_columns)

def _create_numeric_classification_target_plots(
    df: pd.DataFrame,
    artifacts: dict,
    config: EDAConfig,
    plots_dir: Path,
    plot_paths: dict[str, Path],
    numeric_feature_columns: list[str],
) -> None:
    """Create numeric-feature plots for classification targets."""
    if not _artifact_has_rows(artifacts, "numeric_target_tests"):
        return
    target_features = artifacts.get("numeric_target_tests", pd.DataFrame())
    # Intersect screening priorities with features that actually have target-test evidence.
    selected_columns = _ordered_available_features(numeric_feature_columns, target_features, config.max_features_per_plot)
    target_profile_path = plots_dir / "numeric_target_median_percentile_heatmap.png"
    if _plot_numeric_target_median_percentile_heatmap(
        df, selected_columns, config.target_column, config.max_target_levels_for_plots, target_profile_path
    ):
        plot_paths["numeric_target_median_percentile_heatmap"] = target_profile_path
    distribution_path = plots_dir / "numeric_target_distribution_comparison.png"
    if _plot_numeric_target_distribution_comparison(
        df, selected_columns, config.target_column, config.max_target_levels_for_plots, distribution_path
    ):
        plot_paths["numeric_target_distribution_comparison"] = distribution_path

def _create_categorical_classification_target_plots(
    df: pd.DataFrame,
    artifacts: dict,
    config: EDAConfig,
    plots_dir: Path,
    plot_paths: dict[str, Path],
    categorical_columns: list[str],
) -> None:
    """Create categorical-composition plots for classification targets."""
    if not _artifact_has_rows(artifacts, "categorical_target_tests"):
        return
    composition_path = plots_dir / "categorical_target_composition.png"
    selected_columns = _ordered_available_features(
        categorical_columns, artifacts["categorical_target_tests"], config.max_features_per_plot
    )
    if _plot_categorical_target_composition(
        df, selected_columns, config.target_column, config.max_categories, config.max_target_levels_for_plots, composition_path
    ):
        plot_paths["categorical_target_composition"] = composition_path

def _create_regression_target_plots(
    df: pd.DataFrame,
    artifacts: dict,
    config: EDAConfig,
    plots_dir: Path,
    plot_paths: dict[str, Path],
    target_task: str | None,
    categorical_columns: list[str],
) -> None:
    """Create regression-target plots that group target values by categories."""
    if target_task != "regression" or not _artifact_has_rows(artifacts, "categorical_numeric_target_tests"):
        return
    categorical_target_path = plots_dir / "numeric_target_by_categorical_features.png"
    selected_columns = _ordered_available_features(
        categorical_columns, artifacts["categorical_numeric_target_tests"], config.max_features_per_plot
    )
    if _plot_numeric_target_by_categorical_features(
        df, selected_columns, config.target_column, config.max_categories, categorical_target_path
    ):
        plot_paths["numeric_target_by_categorical_features"] = categorical_target_path

def _create_numeric_target_correlation_plots(
    df: pd.DataFrame,
    artifacts: dict,
    config: EDAConfig,
    plots_dir: Path,
    plot_paths: dict[str, Path],
    numeric_feature_columns: list[str],
) -> None:
    """Create target-correlation ranking and scatter plots for numeric features."""
    if not _artifact_has_rows(artifacts, "numeric_target_correlations"):
        return
    numeric_target_corr_path = plots_dir / "numeric_target_correlations.png"
    _plot_numeric_target_correlations(artifacts["numeric_target_correlations"], config.max_features_per_plot, numeric_target_corr_path)
    plot_paths["numeric_target_correlations"] = numeric_target_corr_path
    target_scatter_path = plots_dir / "numeric_target_scatter.png"
    top_columns = _ordered_available_features(
        numeric_feature_columns, artifacts["numeric_target_correlations"], config.max_features_per_plot
    )
    if _plot_numeric_target_scatter(df, top_columns, config.target_column, target_scatter_path):
        plot_paths["numeric_target_scatter"] = target_scatter_path

def _create_feature_target_ranking_plot(
    artifacts: dict,
    config: EDAConfig,
    plots_dir: Path,
    plot_paths: dict[str, Path],
) -> None:
    """Create global feature-target association ranking chart when available."""
    if not _artifact_has_rows(artifacts, "feature_target_associations"):
        return
    feature_ranking_path = plots_dir / "feature_target_association_ranking.png"
    _plot_feature_target_association_ranking(
        artifacts["feature_target_associations"], config.max_features_per_plot, feature_ranking_path
    )
    plot_paths["feature_target_association_ranking"] = feature_ranking_path

def _create_geospatial_plots(
    df: pd.DataFrame,
    config: EDAConfig,
    plots_dir: Path,
    plot_paths: dict[str, Path],
    target_task: str | None,
) -> None:
    """Create geometry and geometry-target plots when geospatial plotting is enabled."""
    if not ("geometry" in df.columns and hasattr(df, "plot") and config.include_geospatial):
        return

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

