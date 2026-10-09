"""Pipeline-facing plot and HTML assembly for the final dashboard."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ml_classification.shared.persistence import ensure_dir, load_joblib
from ml_classification.shared.reports.confidence_diagnostics import (
    export_prediction_confidence_diagnostics,
    prepare_class_risk_display_table,
)
from ml_classification.shared.reports.final_dashboard_artifact import (
    _INDEX_COLUMN_PATTERN,
    _MODELLED_LABEL_COLUMN,
    _MODELLED_LABEL_NAME_COLUMN,
    _build_class_lookup,
    _build_cv_rows,
    _build_index_distribution_rows,
    _build_performance_rows,
    _build_seasonal_range_rows,
    _clean_label_code,
    _display_name,
    _filter_model_rows,
    _read_json,
    _strategy_model_name,
    _summarize_phenology,
)
from ml_classification.shared.reports.report_html import write_html_report


def _pipeline_parcels(config, prepare: dict[str, Any]) -> tuple[gpd.GeoDataFrame, dict[str, str]]:
    """Load and normalize the check-stage dataset for automatic reporting."""
    source = load_joblib(config.check_dir / "input_dataset.joblib")
    if config.target_column not in source.columns:
        raise ValueError(f"Error: Input dataset is missing target column {config.target_column!r}.")

    parcels = source.copy()
    parcels[_MODELLED_LABEL_COLUMN] = parcels[config.target_column].map(_clean_label_code)
    # Filter to modeled classes so seasonality and performance describe the same class population.
    modelled_labels = {_clean_label_code(value) for value in prepare.get("target_labels", [])}
    parcels = parcels.loc[parcels[_MODELLED_LABEL_COLUMN].isin(modelled_labels)].copy()

    label_name_candidates = [config.target_column.removesuffix("_code"), "initial_label", "label_name", "class_name"]
    label_name_column = next(
        (column for column in label_name_candidates if column != config.target_column and column in parcels.columns), None
    )
    if label_name_column:
        parcels[_MODELLED_LABEL_NAME_COLUMN] = parcels[label_name_column]
    class_lookup = _build_class_lookup(parcels)
    return parcels, class_lookup


def _save_monthly_index_boxplots(distribution: pd.DataFrame, output_path: Path) -> None:
    """Save one 12-row figure with NDVI left, NDWI right, and class boxes on x."""
    plot = distribution.loc[distribution["index"].isin(["NDVI", "NDWI"])].copy()
    if plot.empty:
        return
    ensure_dir(output_path.parent)
    plot["date"] = pd.to_datetime(plot["date"])
    month_order = sorted(plot["date"].drop_duplicates().tolist())
    # Fix class order across all monthly panels so each x-position retains its meaning.
    class_order = (
        plot.loc[:, ["class_code", "class_label"]]
        .drop_duplicates()
        .sort_values(["class_code", "class_label"], kind="stable")["class_label"]
        .tolist()
    )
    index_order = ("NDVI", "NDWI")
    colors = {"NDVI": "#2e6f95", "NDWI": "#d99a2b"}

    fig, axes = plt.subplots(
        len(month_order), 2, figsize=(18, 2.5 * len(month_order) + 2.0), sharex=True, sharey=True, squeeze=False
    )
    median_color = "#17212b"
    for row_index, month in enumerate(month_order):
        month_label = pd.Timestamp(month).strftime("%b %Y")
        for column_index, index in enumerate(index_order):
            ax = axes[row_index, column_index]
            panel_rows = plot.loc[(plot["date"] == month) & (plot["index"] == index)]
            if not panel_rows.empty:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", message="vert: bool will be deprecated.*", category=PendingDeprecationWarning
                    )
                    sns.boxplot(
                        data=panel_rows,
                        x="class_label",
                        y="parcel_value",
                        order=class_order,
                        ax=ax,
                        color=colors[index],
                        width=0.64,
                        linewidth=0.8,
                        # Hide markers for readability; the underlying observations still determine the box statistics.
                        showfliers=False,
                        medianprops={"color": median_color, "linewidth": 1.35},
                        whiskerprops={"color": "#52606d", "linewidth": 0.8},
                        capprops={"color": "#52606d", "linewidth": 0.8},
                        boxprops={"edgecolor": "#52606d"},
                    )
                ax.text(
                    0.01, 0.93, f"n={len(panel_rows):,}", transform=ax.transAxes, ha="left", va="top", fontsize=7, color="#52606d"
                )
                if panel_rows["parcel_value"].nunique(dropna=True) <= 1:
                    constant_value = float(panel_rows["parcel_value"].iloc[0])
                    ax.text(
                        0.99,
                        0.93,
                        f"No variation ({constant_value:.2f})",
                        transform=ax.transAxes,
                        ha="right",
                        va="top",
                        fontsize=7,
                        color="#8a3b2f",
                    )
            else:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", color="#64748b")

            if row_index == 0:
                ax.set_title(index, loc="left", fontsize=13, fontweight="bold", color=colors[index])
            else:
                ax.set_title("")
            ax.axhline(0, color="#7b8794", linewidth=0.7, linestyle="--", zorder=0)
            ax.set_xlabel("")
            ax.set_ylabel(f"{month_label}\nIndex value" if column_index == 0 else "")
            ax.set_ylim(-1, 1)
            ax.grid(axis="y", color="#d9e2e8", linewidth=0.6)
            ax.grid(axis="x", visible=False)
            # Shared x-axes need category labels only on the final row of the monthly grid.
            is_bottom_row = row_index == len(month_order) - 1
            ax.tick_params(axis="x", labelsize=7, labelbottom=is_bottom_row, rotation=58)
            if is_bottom_row:
                for label in ax.get_xticklabels():
                    label.set_horizontalalignment("right")
                    label.set_rotation_mode("anchor")
            ax.tick_params(axis="y", labelsize=8)

    first_month = pd.Timestamp(month_order[0]).strftime("%b %Y")
    last_month = pd.Timestamp(month_order[-1]).strftime("%b %Y")
    fig.suptitle(
        "Monthly NDVI and NDWI distributions by modeled class", x=0.075, y=0.995, ha="left", fontsize=17, fontweight="bold"
    )
    fig.text(
        0.075,
        0.978,
        f"Parcel-level values, {first_month}-{last_month}, fixed index scale -1 to 1. "
        "Each row is one month: NDVI is left and NDWI is right. Classes share the x-axis; "
        "boxes show IQR and median, whiskers use 1.5 x IQR, and outlier markers are hidden.",
        ha="left",
        va="top",
        fontsize=9,
        color="#52606d",
    )
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.105, top=0.945, hspace=0.18, wspace=0.08)
    fig.savefig(output_path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _save_monthly_index_range_lines(phenology: pd.DataFrame, index: str, output_path: Path) -> None:
    """Save monthly highest/lowest parcel lines faceted by modeled class."""
    plot = phenology.loc[phenology["index"] == index].copy()
    if plot.empty:
        return
    ensure_dir(output_path.parent)
    plot["date"] = pd.to_datetime(plot["date"])
    class_order = plot.groupby("class_label", observed=True)["parcels"].max().sort_values(ascending=False).index.tolist()
    ncols = min(4, max(1, len(class_order)))
    nrows = int(np.ceil(len(class_order) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.35 * nrows + 1.2), sharex=True, sharey=True, squeeze=False)
    for panel_index, (ax, class_label) in enumerate(zip(axes.flat, class_order)):
        rows = plot.loc[plot["class_label"] == class_label].sort_values("date")
        ax.plot(rows["date"], rows["highest"], color="#2e6f95", marker="o", markersize=3.5, linewidth=1.6, label="Highest")
        ax.plot(
            rows["date"],
            rows["lowest"],
            color="#d99a2b",
            marker="o",
            markerfacecolor="white",
            markersize=3.5,
            linewidth=1.5,
            linestyle="--",
            label="Lowest",
        )
        parcel_count = int(rows["parcels"].max())
        ax.set_title(f"{class_label} (n={parcel_count:,})", loc="left", fontsize=9, fontweight="bold")
        ax.axhline(0, color="#7b8794", linewidth=0.7, linestyle=":", zorder=0)
        ax.set_ylim(-1, 1)
        ax.set_xlabel("")
        ax.set_ylabel(index if panel_index % ncols == 0 else "")
        ax.grid(axis="y", color="#d9e2e8", linewidth=0.6)
        ax.grid(axis="x", visible=False)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%y"))
        ax.tick_params(axis="x", labelsize=7, rotation=0)
        ax.tick_params(axis="y", labelsize=8)
        if rows[["lowest", "highest"]].nunique(dropna=True).max() <= 1:
            constant_value = float(rows["highest"].iloc[0])
            ax.text(
                0.98,
                0.92,
                f"No variation ({constant_value:.2f})",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7,
                color="#8a3b2f",
            )
    for ax in axes.flat[len(class_order) :]:
        ax.set_visible(False)
    first_month = plot["date"].min().strftime("%b %Y")
    last_month = plot["date"].max().strftime("%b %Y")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.99, 0.975), frameon=False, ncol=2)
    fig.suptitle(
        f"Monthly highest and lowest {index} by modeled class", x=0.055, y=0.995, ha="left", fontsize=16, fontweight="bold"
    )
    fig.text(
        0.055,
        0.968,
        f"Highest and lowest parcel median in each class and month, {first_month}–{last_month}; fixed index scale −1 to 1.",
        ha="left",
        va="top",
        fontsize=9,
        color="#52606d",
    )
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.055, top=0.925, hspace=0.43, wspace=0.18)
    fig.savefig(output_path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _save_train_test_plot(performance: pd.DataFrame, metric: str, output_path: Path) -> None:
    """Save a horizontal train-versus-holdout comparison for one metric."""
    plot = performance.loc[performance["metric"] == metric].copy()
    if plot.empty:
        return
    ensure_dir(output_path.parent)
    wide = plot.pivot(index="model", columns="split", values="score")
    sort_column = "Test" if "Test" in wide.columns else wide.columns[0]
    wide = wide.sort_values(sort_column, ascending=True)
    fig, ax = plt.subplots(figsize=(12, max(6, len(wide) * 0.55)))
    wide.plot(kind="barh", ax=ax, color={"Train": "#2e6f95", "Test": "#d99a2b"}, width=0.72)
    ax.set_xlim(0, 1)
    ax.set_xlabel(_display_name(metric))
    ax.set_ylabel("")
    ax.set_title(f"Train versus test holdout — {_display_name(metric)}", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#d9e2e8", linewidth=0.7)
    ax.legend(title="Split", frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _save_selected_cv_fold_plot(cv_folds: pd.DataFrame, output_path: Path) -> None:
    """Save fold stability for selected strategy members only."""
    if cv_folds.empty:
        return
    ensure_dir(output_path.parent)
    model_order = cv_folds.loc[:, ["model", "rank"]].drop_duplicates("model").sort_values("rank")["model"].astype(str).tolist()
    palette = ["#2e6f95", "#d99a2b", "#7a9a48", "#b279a2", "#c66b3d"]
    fig, ax = plt.subplots(figsize=(10, 5.8))
    for model_index, model in enumerate(model_order):
        rows = cv_folds.loc[cv_folds["model"].astype(str) == model].sort_values("fold")
        ax.plot(
            rows["fold"],
            rows["score"],
            color=palette[model_index % len(palette)],
            marker="o",
            markersize=5,
            linewidth=1.8,
            label=model,
        )
    scores = pd.to_numeric(cv_folds["score"], errors="coerce").dropna()
    if not scores.empty and scores.between(0, 1).all():
        ax.set_ylim(0, 1)
    ax.set_xticks(sorted(pd.to_numeric(cv_folds["fold"], errors="coerce").dropna().unique()))
    ax.set_xlabel("CV fold")
    ax.set_ylabel("Validation score")
    ax.set_title("Cross-validation stability of selected models", loc="left", fontweight="bold")
    ax.grid(axis="y", color="#d9e2e8", linewidth=0.7)
    ax.legend(title="Selected model", frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _export_pipeline_seasonality(parcels, class_lookup, data_dir, plots_dir):
    """Export monthly tables and plots, returning phenology, images, and quality notes."""
    index_columns = [column for column in parcels.columns if _INDEX_COLUMN_PATTERN.fullmatch(str(column))]
    phenology = pd.DataFrame()
    seasonal_range = pd.DataFrame()
    seasonal_images: list[dict[str, Any]] = []
    seasonal_quality_notes: list[str] = []
    if index_columns:
        distribution = _build_index_distribution_rows(parcels, class_lookup)
        phenology = _summarize_phenology(distribution)
        seasonal_range = _build_seasonal_range_rows(phenology)
        phenology.to_csv(data_dir / "monthly_index_by_class.csv", index=False)
        seasonal_range.to_csv(data_dir / "monthly_index_high_low_by_class.csv", index=False)
        _save_monthly_index_boxplots(distribution, plots_dir / "seasonal_ndvi_ndwi_boxplots_by_month_class.png")
        _save_monthly_index_range_lines(phenology, "NDVI", plots_dir / "seasonal_ndvi_high_low_by_class.png")
        _save_monthly_index_range_lines(phenology, "NDWI", plots_dir / "seasonal_ndwi_high_low_by_class.png")
        for index in ("NDVI", "NDWI"):
            values = distribution.loc[distribution["index"] == index, "parcel_value"]
            if not values.empty and values.nunique(dropna=True) <= 1:
                seasonal_quality_notes.append(
                    f"Data-quality warning: every {index} value in this evaluation input equals {float(values.iloc[0]):.2f}, "
                    "so its monthly boxes have no spread."
                )
        for stale_plot in (
            "seasonal_peak_by_class.png",
            "seasonal_ndvi_boxplots_by_class_month.png",
            "seasonal_ndwi_boxplots_by_class_month.png",
        ):
            (plots_dir / stale_plot).unlink(missing_ok=True)
        seasonal_images = [
            {
                "title": "Monthly NDVI and NDWI Distributions by Modeled Class",
                "path": plots_dir / "seasonal_ndvi_ndwi_boxplots_by_month_class.png",
            },
            {
                "title": "Monthly Highest and Lowest NDVI by Modeled Class",
                "path": plots_dir / "seasonal_ndvi_high_low_by_class.png",
            },
            {
                "title": "Monthly Highest and Lowest NDWI by Modeled Class",
                "path": plots_dir / "seasonal_ndwi_high_low_by_class.png",
            },
        ]

    return phenology, seasonal_images, seasonal_quality_notes


def _pipeline_dashboard_summary(
    prepare, selection, selected_train_metrics, selected_test_metrics, strategy_model, primary_metric, cv_ranking,
    confidence_artifacts,
):
    """Summarize strategy scores, sample counts, and optional prediction diagnostics."""
    strategy_test = selected_test_metrics.loc[selected_test_metrics["model"].astype(str) == strategy_model].iloc[0]
    strategy_train = selected_train_metrics.loc[selected_train_metrics["model"].astype(str) == strategy_model]
    train_score = float(strategy_train.iloc[0][primary_metric]) if not strategy_train.empty else np.nan
    test_score = float(strategy_test[primary_metric])
    best_cv = cv_ranking.iloc[0]
    summary = {
        "modeled_parcels": int(prepare.get("train_rows", 0)) + int(prepare.get("test_rows", 0)),
        "train_parcels": int(prepare.get("train_rows", 0)),
        "spatial_holdout_parcels": int(prepare.get("test_rows", 0)),
        "modeled_classes": len(prepare.get("target_labels", [])),
        "best_cv_model": best_cv["model"],
        "best_mean_cv_score": float(best_cv["mean_cv_score"]),
        "final_strategy": selection.get("selection_type"),
        "selected_models": list(selection.get("selected_models", [])),
        f"test_{primary_metric}": test_score,
        f"train_test_{primary_metric}_gap": float(train_score - test_score) if not np.isnan(train_score) else np.nan,
    }
    if confidence_artifacts is not None:
        summary.update(
            {
                "prediction_validation_rows": confidence_artifacts.summary["validation_rows_with_reference"],
                "prediction_accuracy": confidence_artifacts.summary["prediction_accuracy"],
                "borda_agreement_rate": confidence_artifacts.summary["borda_agreement_rate"],
                "need_to_check_rows": confidence_artifacts.summary["need_to_check_rows"],
                "need_to_check_rate_among_borda_agreement": confidence_artifacts.summary[
                    "need_to_check_rate_among_borda_agreement"
                ],
            }
        )

    return summary


def _pipeline_dashboard_data_links(data_dir, *, has_phenology, has_confidence):
    """List exported tables in report order, including available optional sections."""
    data_links = [
        {"label": "CV Model Ranking", "path": data_dir / "cv_model_ranking.csv"},
        {"label": "Train-Test Model Metrics", "path": data_dir / "train_test_model_metrics.csv"},
    ]
    if has_phenology:
        data_links[0:0] = [
            {"label": "Monthly NDVI/NDWI by Class", "path": data_dir / "monthly_index_by_class.csv"},
            {"label": "Monthly High/Low Index Values by Class", "path": data_dir / "monthly_index_high_low_by_class.csv"},
        ]
    if has_confidence:
        data_links.extend(
            [
                {"label": "Borda/Correctness Group Summary", "path": data_dir / "diagnostic_group_summary.csv"},
                {"label": "Need-to-Check by Predicted Class", "path": data_dir / "need_to_check_by_predicted_class.csv"},
                {"label": "Confidence Component Matrix", "path": data_dir / "need_to_check_reason_matrix.csv"},
            ]
        )
    return data_links


def _pipeline_confidence_section(confidence_artifacts):
    """Build a confidence section using the diagnostic images available for this run."""
    confidence_images = []
    for key, title in (
        ("overview", "Borda Consensus and Prediction Correctness"),
        ("class_rate", "Need-to-Check Rate by Predicted Class"),
        ("reason_heatmap", "Confidence Components of Incorrect Consensus Predictions"),
    ):
        path = confidence_artifacts.images.get(key)
        if path is not None:
            confidence_images.append({"title": title, "path": path})
    return {
        "title": "Prediction Confidence Diagnostics",
        "text": (
            "These diagnostics use only parcels with known original reference labels. The overview shows "
            "whether Borda consensus aligns with correctness; the class chart identifies where incorrect "
            "consensus predictions concentrate; and the component matrix shows how class reliability and "
            "rank confidence combine inside the need-to-check cohort. Unknown fill rows are excluded from "
            "all correctness rates."
        ),
        "images": confidence_images,
        "table": prepare_class_risk_display_table(
            confidence_artifacts.tables["need_to_check_by_predicted_class"], limit=15
        ),
    }


def _pipeline_performance_section(performance_table, strategy_model, plots_dir):
    """Render holdout scores and train-test gaps with consistent cell highlighting."""
    test_metric_column_styles = {
        str(column): "success" for column in performance_table.columns if str(column).startswith("test_")
    }
    # Positive train-minus-test gaps receive warning styling to highlight potential overfitting.
    gap_metric_cell_styles = {
        str(column): {"positive": "danger", "negative": "success"}
        for column in performance_table.columns
        if str(column).endswith("_gap")
    }
    voting_test_cell_styles = [
        {"column": "model", "values": [strategy_model], "target_columns": list(test_metric_column_styles), "style": "warning"}
    ]

    return {
        "title": "Train versus Spatial Holdout",
        "text": (
            "The chart contains only selected models and the final strategy result, using the primary selection metric. "
            "Test/holdout values are green for selected models and yellow for the final voting result. "
            "Each gap is train minus test: negative gaps are green because test is higher, "
            "positive gaps are red because test is lower, and zero is neutral."
        ),
        "images": [{"title": "Train versus Spatial Holdout", "path": plots_dir / "train_test_model_comparison.png"}],
        "table": performance_table,
        "column_styles": test_metric_column_styles,
        "numeric_cell_styles": gap_metric_cell_styles,
        "cell_styles_where": voting_test_cell_styles,
        "highlight_rows_where": {"column": "model", "values": [strategy_model]},
    }


def write_pipeline_final_dashboard(
    config,
    train_metrics_df: pd.DataFrame,
    test_metrics_df: pd.DataFrame,
    selection: dict[str, Any],
    prediction_df: pd.DataFrame | None = None,
) -> Path:
    """Create the final HTML dashboard, optionally including prediction diagnostics."""
    dashboard_dir = ensure_dir(config.final_dashboard_dir)
    plots_dir = ensure_dir(dashboard_dir / "plots")
    data_dir = ensure_dir(dashboard_dir / "data")
    prepare = _read_json(config.prepare_dir / "prepare_summary.json")
    parcels, class_lookup = _pipeline_parcels(config, prepare)

    phenology, seasonal_images, seasonal_quality_notes = _export_pipeline_seasonality(
        parcels, class_lookup, data_dir, plots_dir
    )

    training = pd.read_csv(config.train_dir / "training_summary.csv")
    fold_scores = pd.read_csv(config.train_dir / "best_cv_fold_scores.csv")
    selected_models = {str(value) for value in selection.get("selected_models", [])}
    strategy_model = _strategy_model_name(selection, test_metrics_df)
    training = _filter_model_rows(training, selected_models, "training_summary.csv")
    fold_scores = _filter_model_rows(fold_scores, selected_models, "best_cv_fold_scores.csv")
    performance_models = selected_models | {strategy_model}
    selected_train_metrics = _filter_model_rows(train_metrics_df, performance_models, "train evaluation metrics")
    selected_test_metrics = _filter_model_rows(test_metrics_df, performance_models, "test evaluation metrics")
    cv_folds, cv_ranking = _build_cv_rows(training, fold_scores, selected_models)
    performance, performance_table = _build_performance_rows(selected_train_metrics, selected_test_metrics)
    primary_metric = config.scoring_primary if config.scoring_primary in performance["metric"].unique() else "f1_macro"
    if primary_metric not in performance["metric"].unique():
        primary_metric = str(performance["metric"].iloc[0])
    _save_selected_cv_fold_plot(cv_folds, plots_dir / "selected_model_cv_folds.png")
    _save_train_test_plot(performance, primary_metric, plots_dir / "train_test_model_comparison.png")
    cv_ranking.to_csv(data_dir / "cv_model_ranking.csv", index=False)
    performance_table.to_csv(data_dir / "train_test_model_metrics.csv", index=False)

    confidence_artifacts = None
    if prediction_df is not None:
        prediction_column = getattr(config, "prediction_column", f"{config.target_column}_prediction")
        confidence_artifacts = export_prediction_confidence_diagnostics(
            prediction_df,
            config.target_column,
            prediction_column,
            dashboard_dir,
            include=("overview", "class_rate", "reason_heatmap"),
        )

    summary = _pipeline_dashboard_summary(
        prepare, selection, selected_train_metrics, selected_test_metrics, strategy_model, primary_metric, cv_ranking,
        confidence_artifacts,
    )
    test_score = summary[f"test_{primary_metric}"]

    probability_optimization = selection.get("probability_optimization") or {}
    probability_table = pd.DataFrame(
        [
            {
                "metric": "Macro F1",
                "baseline": probability_optimization.get("baseline_oof_macro_f1"),
                "optimized": probability_optimization.get("optimized_oof_macro_f1"),
            },
            {
                "metric": "Accuracy",
                "baseline": probability_optimization.get("baseline_oof_accuracy"),
                "optimized": probability_optimization.get("optimized_oof_accuracy"),
            },
        ]
    ).dropna(how="all", subset=["baseline", "optimized"])

    map_embeds = []
    osm_map = config.check_dir / "plots" / "known_labels_map_osm.html"
    if osm_map.exists():
        map_embeds.append({"title": "Known Modeled Parcels on OpenStreetMap", "path": osm_map})
    data_links = _pipeline_dashboard_data_links(
        data_dir, has_phenology=not phenology.empty, has_confidence=confidence_artifacts is not None
    )
    sections: list[dict[str, Any]] = [
        {"title": "Evaluation Snapshot", "kv": summary},
        {
            "title": "Study Area and Parcel Coverage",
            "text": "The interactive map uses OpenStreetMap tiles when network access is available.",
            "embeds": map_embeds,
        },
        {
            "title": "Seasonal Crop Profiles",
            "text": (
                "The combined Seaborn figure has one row per month and two columns: NDVI on the left and NDWI "
                "on the right. Every panel places modeled class/label on the shared x-axis and the parcel-level "
                "index distribution on the y-axis. "
                "Boxes show the interquartile range and median; whiskers use 1.5×IQR and outlier markers are hidden for readability. "
                "The high/low line figures show each class's highest and lowest parcel median in every month. "
                + " ".join(seasonal_quality_notes)
            ),
            "images": seasonal_images,
        },
        {
            "title": "Cross-Validation Stability",
            "text": "Only selected strategy members are shown. Fold scores use each selected model's best hyperparameter setting; the table follows the configured CV ranking score.",
            "images": [{"title": "Selected-Model CV Score per Fold", "path": plots_dir / "selected_model_cv_folds.png"}],
            "table": cv_ranking,
            "row_styles_where": _voting_member_row_styles_for_pipeline(cv_ranking, selected_models),
        },
        _pipeline_performance_section(performance_table, strategy_model, plots_dir),
    ]
    if confidence_artifacts is not None:
        sections.append(_pipeline_confidence_section(confidence_artifacts))
    sections.extend(
        [
            {
                "title": "Best Model and Voting Result",
                "kv": {
                    "selection_type": selection.get("selection_type"),
                    "selected_models": list(selection.get("selected_models", [])),
                    "selected_metric": selection.get("selected_metric"),
                    "selected_cv_score": selection.get("selected_score"),
                    f"holdout_{primary_metric}": test_score,
                },
                "table": probability_table,
            },
            {"title": "Dashboard Data", "links": data_links},
        ]
    )
    output_path = dashboard_dir / "report.html"
    write_html_report(
        output_path,
        "Final Crop Classification Dashboard",
        (
            "Study-area coverage, crop seasonality, cross-validation stability, holdout performance, final model "
            "selection, and prediction confidence diagnostics."
            if confidence_artifacts is not None
            else "Study-area coverage, crop seasonality, cross-validation stability, holdout performance, and final model selection."
        ),
        sections,
    )
    return output_path


def _voting_member_row_styles_for_pipeline(table: pd.DataFrame, selected_models: set[str]) -> list[dict[str, Any]]:
    """Return report-renderer row styles for selected and non-selected CV models."""
    models = table["model"].astype(str).tolist() if "model" in table.columns else []
    return [
        {"column": "model", "values": [model for model in models if model in selected_models], "style": "success"},
        {"column": "model", "values": [model for model in models if model not in selected_models], "style": "muted"},
    ]
