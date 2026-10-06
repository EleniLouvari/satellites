"""Operational inspection-priority tables and readable report visuals."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns

from satellites.ml_classification.step_05_predict.libraries.inspection_priority.plots import save_inspection_relationship_plot

_NEED_ORDER = ("VERY_HIGH", "HIGH", "MEDIUM_HIGH", "MEDIUM_UNCERTAIN", "MEDIUM", "LOW", "UNKNOWN")
_CHECK_TYPE_ORDER = ("DECLARATION_CONFLICT", "INSUFFICIENT_EO_EVIDENCE", "NONE")
_DECLARATION_ORDER = ("SAME", "DIFFERENT", "NO_DECLARATION", "UNAVAILABLE")


@dataclass(frozen=True)
class InspectionDiagnosticsArtifacts:
    """Saved inspection-priority assets and their exact supporting tables."""

    summary: dict[str, Any]
    images: dict[str, Path]
    tables: dict[str, pd.DataFrame]


def _inspection_need_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Count parcels and summarize score within each final priority level."""
    rows = data["inspection_need"].value_counts().reindex(_NEED_ORDER, fill_value=0)
    medians = data.groupby("inspection_need", observed=True)["inspection_score"].median()
    result = rows.rename("parcels").rename_axis("inspection_need").reset_index()
    result["share_percent"] = 100.0 * result["parcels"] / max(len(data), 1)
    result["median_inspection_score"] = result["inspection_need"].map(medians)
    return result.loc[result["parcels"].gt(0)].reset_index(drop=True)


def _inspection_check_type_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Count mutually exclusive operational check outcomes."""
    rows = data["inspection_check_type"].value_counts().reindex(_CHECK_TYPE_ORDER, fill_value=0)
    medians = data.groupby("inspection_check_type", observed=True)["inspection_score"].median()
    result = rows.rename("parcels").rename_axis("inspection_check_type").reset_index()
    result["share_percent"] = 100.0 * result["parcels"] / max(len(data), 1)
    result["median_inspection_score"] = result["inspection_check_type"].map(medians)
    return result.loc[result["parcels"].gt(0)].reset_index(drop=True)


def _priority_by_predicted_class(data: pd.DataFrame, prediction_column: str) -> pd.DataFrame:
    """Summarize operational inspection burden by predicted crop."""
    working = data.copy()
    working["predicted_class"] = working[prediction_column].astype("string").fillna("<missing>")
    working["needs_operational_check"] = working["inspection_check_type"].ne("NONE")
    working["is_declaration_conflict"] = working["inspection_check_type"].eq("DECLARATION_CONFLICT")
    working["is_insufficient_evidence"] = working["inspection_check_type"].eq("INSUFFICIENT_EO_EVIDENCE")
    working["is_high_priority"] = working["inspection_need"].isin(("HIGH", "VERY_HIGH"))
    summary = (
        working.groupby("predicted_class", observed=True)
        .agg(
            parcels=("predicted_class", "size"),
            needs_check=("needs_operational_check", "sum"),
            high_or_very_high=("is_high_priority", "sum"),
            declaration_conflicts=("is_declaration_conflict", "sum"),
            insufficient_eo_evidence=("is_insufficient_evidence", "sum"),
            median_inspection_score=("inspection_score", "median"),
        )
        .reset_index()
    )
    summary["needs_check_rate_percent"] = 100.0 * summary["needs_check"] / summary["parcels"]
    return summary.sort_values(
        ["high_or_very_high", "declaration_conflicts", "needs_check_rate_percent", "parcels"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def _declaration_conflict_matrix(data: pd.DataFrame, target_column: str, prediction_column: str) -> pd.DataFrame:
    """Cross-tabulate declared and predicted crops for reliable conflicts only."""
    conflicts = data.loc[
        data["inspection_check_type"].eq("DECLARATION_CONFLICT") & data[target_column].notna() & data[prediction_column].notna()
    ]
    if conflicts.empty:
        return pd.DataFrame()
    return pd.crosstab(
        conflicts[target_column].astype(str),
        conflicts[prediction_column].astype(str),
        rownames=["declared_crop"],
        colnames=["predicted_crop"],
    )


def _evidence_quality_table(data: pd.DataFrame, config) -> pd.DataFrame:
    """Cross data reliability and geometry risk using the configured policy thresholds."""
    working = data.copy()
    working["data_reliability_level"] = pd.cut(
        pd.to_numeric(working[config.inspection_data_reliability_column], errors="coerce"),
        bins=[
            -np.inf,
            config.inspection_low_data_reliability_threshold,
            config.inspection_medium_data_reliability_threshold,
            np.inf,
        ],
        labels=["LOW", "MEDIUM", "HIGH"],
        right=False,
    )
    working["geometry_quality_level"] = pd.cut(
        pd.to_numeric(working["geometry_risk"], errors="coerce"),
        bins=[-np.inf, config.inspection_medium_geometry_risk_threshold, config.inspection_high_geometry_risk_threshold, np.inf],
        labels=["GOOD", "MODERATE", "COMPLEX"],
        right=False,
    )
    working["needs_operational_check"] = working["inspection_check_type"].ne("NONE")
    grouped = (
        working.dropna(subset=["data_reliability_level", "geometry_quality_level"])
        .groupby(["data_reliability_level", "geometry_quality_level"], observed=True)
        .agg(
            parcels=("inspection_need", "size"),
            needs_check=("needs_operational_check", "sum"),
            median_inspection_score=("inspection_score", "median"),
        )
        .reset_index()
    )
    grouped["needs_check_rate_percent"] = 100.0 * grouped["needs_check"] / grouped["parcels"]
    return grouped


def _review_comparison(data: pd.DataFrame, review_column: str) -> pd.DataFrame:
    """Compare confidence-only review flags with final operational inspection need."""
    if review_column not in data:
        return pd.DataFrame()
    review = data[review_column].fillna(False).astype(bool).map({False: "No confidence review", True: "Confidence review"})
    return pd.crosstab(review, data["inspection_need"].astype("string")).reindex(
        index=["No confidence review", "Confidence review"], columns=_NEED_ORDER, fill_value=0
    )


def _declaration_need_table(data: pd.DataFrame) -> pd.DataFrame:
    """Cross declaration agreement with the final label-aware need."""
    return pd.crosstab(data["label_prediction_status"], data["inspection_need"]).reindex(
        index=_DECLARATION_ORDER, columns=_NEED_ORDER, fill_value=0
    )


def _confidence_data_score_table(data: pd.DataFrame, config) -> pd.DataFrame:
    """Summarize median inspection score by confidence and reliability band."""
    reliability_band = pd.cut(
        pd.to_numeric(data[config.inspection_data_reliability_column], errors="coerce"),
        bins=[
            -np.inf,
            config.inspection_low_data_reliability_threshold,
            config.inspection_medium_data_reliability_threshold,
            np.inf,
        ],
        labels=["LOW", "MEDIUM", "HIGH"],
        right=False,
    )
    confidence = data[config.prediction_confidence_level_column].astype("string").str.upper()
    return (
        data.assign(_confidence=confidence, _reliability_band=reliability_band)
        .pivot_table(index="_confidence", columns="_reliability_band", values="inspection_score", aggfunc="median", observed=True)
        .reindex(index=["HIGH", "MEDIUM", "LOW"], columns=["LOW", "MEDIUM", "HIGH"])
        .rename_axis(index="prediction_confidence", columns="data_reliability")
    )


def _top_priority_parcels(data: pd.DataFrame, config, limit: int = 100) -> pd.DataFrame:
    """Return the highest-priority parcel-level audit queue with its explanations."""
    priority_rank = {need: rank for rank, need in enumerate(_NEED_ORDER)}
    working = data.loc[data["inspection_check_type"].ne("NONE")].copy()
    working["_priority_rank"] = working["inspection_need"].map(priority_rank).fillna(len(priority_rank))
    columns = [
        config.id_column,
        config.target_column,
        config.prediction_column,
        config.prediction_confidence_level_column,
        config.inspection_data_reliability_column,
        "geometry_risk",
        "inspection_score",
        "inspection_need",
        "inspection_check_type",
        "inspection_reasons",
    ]
    columns = [column for column in columns if column in working]
    return (
        working.sort_values(["_priority_rank", "inspection_score"], ascending=[True, False])
        .head(limit)[columns]
        .reset_index(drop=True)
    )


def _save_figure(fig: plt.Figure, output_path: Path) -> Path:
    """Save a report-size figure and close its resources."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _save_check_type_bar(table: pd.DataFrame, output_path: Path) -> Path | None:
    """Save parcel counts for declaration conflicts and insufficient evidence."""
    plot = table.loc[table["inspection_check_type"].ne("NONE")].copy()
    if plot.empty:
        return None
    colors = {"DECLARATION_CONFLICT": "#D97706", "INSUFFICIENT_EO_EVIDENCE": "#2E6F95"}
    fig, ax = plt.subplots(figsize=(13, 7))
    bars = ax.barh(
        plot["inspection_check_type"],
        plot["parcels"],
        color=[colors.get(value, "#8D99AE") for value in plot["inspection_check_type"]],
        edgecolor="#334E68",
    )
    ax.bar_label(bars, labels=[f"{int(value):,}" for value in plot["parcels"]], padding=5)
    ax.set_title("Operational inspection check types", loc="left", fontweight="bold")
    ax.set_xlabel("Parcels")
    ax.set_ylabel("")
    ax.grid(axis="x", color="#D9E2EC", linewidth=0.8)
    ax.grid(axis="y", visible=False)
    return _save_figure(fig, output_path)


def _save_predicted_class_priority(table: pd.DataFrame, output_path: Path, limit: int = 20) -> Path | None:
    """Save the operational needs-check rate for the highest-burden predicted crops."""
    if table.empty:
        return None
    plot = table.head(limit).sort_values("needs_check_rate_percent", ascending=True)
    fig_height = max(7.0, 0.48 * len(plot) + 2.5)
    fig, ax = plt.subplots(figsize=(15, fig_height))
    bars = ax.barh(
        plot["predicted_class"].astype(str), plot["needs_check_rate_percent"] / 100.0, color="#2E6F95", edgecolor="#173F5F"
    )
    labels = [f"{int(row.needs_check):,}/{int(row.parcels):,}" for row in plot.itertuples()]
    ax.bar_label(bars, labels=labels, padding=5, fontsize=9)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_xlim(0.0, max(1.0, float((plot["needs_check_rate_percent"] / 100.0).max()) * 1.18))
    ax.set_title("Operational inspection rate by predicted crop", loc="left", fontweight="bold")
    ax.set_xlabel("Share of predicted-crop parcels requiring operational inspection")
    ax.set_ylabel("Predicted crop")
    ax.grid(axis="x", color="#D9E2EC", linewidth=0.8)
    ax.grid(axis="y", visible=False)
    return _save_figure(fig, output_path)


def _save_matrix(
    table: pd.DataFrame,
    output_path: Path,
    title: str,
    x_label: str,
    y_label: str,
    *,
    fmt: str = "g",
    cmap: str = "Blues",
    cbar_label: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> Path | None:
    """Save a large annotated heatmap from a prepared matrix."""
    if table.empty:
        return None
    # Seaborn requires plain numeric NaN values rather than pandas nullable scalars.
    plot_table = table.apply(pd.to_numeric, errors="coerce").astype(float)
    fig_width = max(12.0, 1.25 * len(table.columns) + 5.0)
    fig_height = max(7.0, 0.65 * len(table.index) + 3.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    cbar_options = {"label": cbar_label} if cbar_label else None
    sns.heatmap(
        plot_table,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.6,
        cbar=cbar_label is not None,
        cbar_kws=cbar_options,
        ax=ax,
    )
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    return _save_figure(fig, output_path)


def export_operational_inspection_diagnostics(
    data: pd.DataFrame, config, output_dir: str | Path
) -> InspectionDiagnosticsArtifacts | None:
    """Build and persist the operational inspection tables and report figures."""
    data_reliability_column = getattr(config, "inspection_data_reliability_column", "data_reliability_score")
    required = {
        config.id_column,
        config.target_column,
        config.prediction_column,
        config.prediction_confidence_level_column,
        data_reliability_column,
        "geometry_risk",
        "inspection_score",
        "inspection_need",
        "inspection_check_type",
        "label_prediction_status",
        "inspection_reasons",
    }
    if not required.issubset(data.columns):
        return None

    root = Path(output_dir)
    plots_dir = root / "plots"
    data_dir = root / "data"
    plots_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    need_summary = _inspection_need_summary(data)
    check_summary = _inspection_check_type_summary(data)
    class_priority = _priority_by_predicted_class(data, config.prediction_column)
    conflict_matrix = _declaration_conflict_matrix(data, config.target_column, config.prediction_column)
    evidence_quality = _evidence_quality_table(data, config)
    review_comparison = _review_comparison(data, config.prediction_review_column)
    declaration_need = _declaration_need_table(data)
    confidence_data_score = _confidence_data_score_table(data, config)
    top_priority = _top_priority_parcels(data, config)
    tables = {
        "inspection_need_summary": need_summary,
        "inspection_check_type_summary": check_summary,
        "inspection_priority_by_predicted_class": class_priority,
        "inspection_declaration_conflict_matrix": conflict_matrix,
        "inspection_evidence_quality_matrix": evidence_quality,
        "inspection_confidence_review_comparison": review_comparison,
        "inspection_need_by_declaration_status": declaration_need,
        "inspection_score_by_confidence_data": confidence_data_score,
        "inspection_top_priority_parcels": top_priority,
    }
    matrix_tables = {
        "inspection_declaration_conflict_matrix",
        "inspection_confidence_review_comparison",
        "inspection_need_by_declaration_status",
        "inspection_score_by_confidence_data",
    }
    for name, table in tables.items():
        table.to_csv(data_dir / f"{name}.csv", index=name in matrix_tables)

    images: dict[str, Path] = {}
    relationship_path = plots_dir / "inspection_risk_relationships.png"
    if save_inspection_relationship_plot(data, relationship_path, config):
        images["relationship"] = relationship_path
    builders = {
        "check_types": _save_check_type_bar(check_summary, plots_dir / "inspection_check_type_distribution.png"),
        "predicted_class": _save_predicted_class_priority(
            class_priority, plots_dir / "inspection_priority_by_predicted_class.png"
        ),
        "declaration_need": _save_matrix(
            declaration_need,
            plots_dir / "inspection_need_by_declaration_status.png",
            "Final inspection need by declaration agreement",
            "Label-aware inspection need",
            "Declaration versus prediction",
        ),
        "confidence_data_score": _save_matrix(
            confidence_data_score,
            plots_dir / "inspection_score_confidence_data_heatmap.png",
            "Median inspection score by confidence and data reliability",
            "Data reliability level",
            "Prediction confidence level",
            fmt=".1f",
            cmap="YlOrRd",
            cbar_label="Median inspection score",
            vmin=0.0,
            vmax=100.0,
        ),
        "conflict_matrix": _save_matrix(
            conflict_matrix,
            plots_dir / "inspection_declaration_conflicts.png",
            "Reliable declaration conflicts",
            "Predicted crop",
            "Farmer-declared crop",
            cmap="YlOrBr",
        ),
        "review_comparison": _save_matrix(
            review_comparison,
            plots_dir / "inspection_confidence_review_comparison.png",
            "Confidence review versus operational inspection need",
            "Operational inspection need",
            "Confidence-only review flag",
        ),
    }
    evidence_matrix = evidence_quality.pivot(
        index="geometry_quality_level", columns="data_reliability_level", values="needs_check_rate_percent"
    ).reindex(index=["GOOD", "MODERATE", "COMPLEX"], columns=["LOW", "MEDIUM", "HIGH"])
    builders["evidence_quality"] = _save_matrix(
        evidence_matrix,
        plots_dir / "inspection_evidence_quality_heatmap.png",
        "Operational inspection rate by EO evidence quality",
        "Data reliability level",
        "Geometry quality level",
        fmt=".1f",
        cmap="YlOrRd",
        cbar_label="Needs-check rate (%)",
        vmin=0.0,
        vmax=100.0,
    )
    images.update({name: path for name, path in builders.items() if path is not None})

    summary = {
        "rows": int(len(data)),
        "rows_requiring_operational_check": int(data["inspection_check_type"].ne("NONE").sum()),
        "declaration_conflicts": int(data["inspection_check_type"].eq("DECLARATION_CONFLICT").sum()),
        "insufficient_eo_evidence": int(data["inspection_check_type"].eq("INSUFFICIENT_EO_EVIDENCE").sum()),
    }
    return InspectionDiagnosticsArtifacts(summary=summary, images=images, tables=tables)
