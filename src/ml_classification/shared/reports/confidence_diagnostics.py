"""Prediction-confidence diagnostics shared by step reports and dashboards."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns

_DIAGNOSTIC_GROUP_ORDER = (
    "Borda agrees + correct",
    "Borda agrees + incorrect",
    "Borda disagrees + correct",
    "Borda disagrees + incorrect",
)
_LEVEL_ORDER = ("HIGH", "MEDIUM", "LOW")
_CONFIDENCE_METRICS = {
    "prediction_max_probability": "Maximum predicted probability",
    "prediction_borda_margin": "Borda winner margin",
    "prediction_mean_borda": "Mean Borda score",
    "prediction_rank_range": "Rank range (lower is better)",
    "prediction_top1_agreement": "Top-1 model agreement",
    "prediction_class_oof_precision": "OOF precision of predicted class",
}
_AVAILABLE_VISUALS = frozenset(
    {
        "overview",
        "class_rate",
        "reason_heatmap",
        "confusion",
        "reason_sankey",
        "distribution",
        "joint_behavior",
        "class_confidence_risk",
    }
)


@dataclass(frozen=True)
class ConfidenceDiagnosticsArtifacts:
    """Saved confidence-diagnostic assets and their supporting tables."""

    summary: dict[str, Any]
    images: dict[str, Path]
    embeds: dict[str, Path]
    tables: dict[str, pd.DataFrame]


def prepare_class_risk_display_table(class_risk: pd.DataFrame, limit: int = 20) -> pd.DataFrame:
    """Return a reader-facing class-risk table with explicit percentage units."""
    display = class_risk.head(limit).copy()
    percentage_columns = {"need_to_check_rate": "need_to_check_rate_percent", "oof_precision": "oof_precision_percent"}
    for source_column, output_column in percentage_columns.items():
        if source_column in display.columns:
            display[output_column] = pd.to_numeric(display[source_column], errors="coerce") * 100
    return display.drop(columns=list(percentage_columns), errors="ignore")


def _normalize_label(value: Any) -> Any:
    """Normalize label values so numeric codes and strings compare reliably."""
    if pd.isna(value):
        return pd.NA
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def prepare_confidence_diagnostics(predictions: pd.DataFrame, target_column: str, prediction_column: str) -> pd.DataFrame:
    """Return correctness diagnostics for rows with a known reference label."""
    base_columns = [target_column, prediction_column, "prediction_borda_winner"]
    required = set(base_columns)
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise KeyError(f"Error: Missing columns required for confidence diagnostics: {missing}")

    optional_columns = {
        "prediction_class_oof_precision",
        "prediction_class_reliability_level",
        "prediction_rank_confidence_level",
        "prediction_confidence_level",
        "prediction_max_probability",
        "prediction_borda_margin",
        "prediction_mean_borda",
        "prediction_rank_range",
        "prediction_top1_agreement",
    }
    analysis_columns = [column for column in [*base_columns, *sorted(optional_columns)] if column in predictions.columns]
    diagnostics = predictions.loc[
        predictions[target_column].notna() & predictions[prediction_column].notna(), analysis_columns
    ].copy()
    diagnostics["reference_label"] = diagnostics[target_column].map(_normalize_label)
    diagnostics["predicted_label"] = diagnostics[prediction_column].map(_normalize_label)
    diagnostics["borda_label"] = diagnostics["prediction_borda_winner"].map(_normalize_label)
    diagnostics["prediction_correct"] = diagnostics["predicted_label"].eq(diagnostics["reference_label"])
    diagnostics["borda_agrees_prediction"] = diagnostics["borda_label"].notna() & diagnostics["borda_label"].eq(
        diagnostics["predicted_label"]
    )

    conditions = [
        diagnostics["borda_agrees_prediction"] & diagnostics["prediction_correct"],
        diagnostics["borda_agrees_prediction"] & ~diagnostics["prediction_correct"],
        ~diagnostics["borda_agrees_prediction"] & diagnostics["prediction_correct"],
        ~diagnostics["borda_agrees_prediction"] & ~diagnostics["prediction_correct"],
    ]
    diagnostics["diagnostic_group"] = pd.Categorical(
        np.select(conditions, _DIAGNOSTIC_GROUP_ORDER, default="Unknown"),
        categories=[*_DIAGNOSTIC_GROUP_ORDER, "Unknown"],
        ordered=True,
    )
    return diagnostics


def _confidence_audit(diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Return the Borda-agreement cohort used by the confidence audit."""
    audit = diagnostics.loc[diagnostics["borda_agrees_prediction"]].copy()
    audit["audit_error"] = ~audit["prediction_correct"]
    audit["audit_group"] = pd.Categorical(
        np.where(audit["audit_error"], "Needs check", "Correct"), categories=["Correct", "Needs check"], ordered=True
    )
    for column in _CONFIDENCE_METRICS:
        if column in audit.columns:
            audit[column] = pd.to_numeric(audit[column], errors="coerce")
    return audit


def _confidence_median_summary(confidence_audit: pd.DataFrame) -> pd.DataFrame:
    """Compare confidence-component medians for correct and audit-error rows."""
    available_metrics = [column for column in _CONFIDENCE_METRICS if column in confidence_audit.columns]
    if confidence_audit.empty or not available_metrics:
        return pd.DataFrame(columns=["confidence_metric", "Correct", "Needs check", "Needs check minus Correct"])
    summary = (
        confidence_audit.groupby("audit_group", observed=True)[available_metrics].median().reindex(["Correct", "Needs check"]).T
    )
    summary.index = summary.index.map(_CONFIDENCE_METRICS)
    summary["Needs check minus Correct"] = summary["Needs check"] - summary["Correct"]
    return summary.rename_axis("confidence_metric").reset_index()


def _diagnostic_summary(diagnostics: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    """Build the report summary and the overview chart's exact source table."""
    group_summary = (
        diagnostics["diagnostic_group"].value_counts(sort=False).rename_axis("diagnostic_group").reset_index(name="rows")
    )
    group_summary = group_summary.loc[group_summary["rows"].gt(0)].copy()
    group_summary["share"] = group_summary["rows"] / max(len(diagnostics), 1)

    borda_agree = diagnostics["borda_agrees_prediction"]
    need_to_check = borda_agree & ~diagnostics["prediction_correct"]
    borda_rows = int(borda_agree.sum())
    summary = {
        "validation_rows_with_reference": int(len(diagnostics)),
        "correct_predictions": int(diagnostics["prediction_correct"].sum()),
        "incorrect_predictions": int((~diagnostics["prediction_correct"]).sum()),
        "prediction_accuracy": float(diagnostics["prediction_correct"].mean()) if len(diagnostics) else np.nan,
        "borda_agreement_rows": borda_rows,
        "borda_agreement_rate": float(borda_agree.mean()) if len(diagnostics) else np.nan,
        "need_to_check_rows": int(need_to_check.sum()),
        "need_to_check_rate_among_borda_agreement": (float(need_to_check.sum() / borda_rows) if borda_rows else np.nan),
    }
    return summary, group_summary


def _class_risk_table(diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Summarize incorrect Borda-consensus predictions by predicted class."""
    borda_agree = diagnostics.loc[diagnostics["borda_agrees_prediction"]].copy()
    if borda_agree.empty:
        return pd.DataFrame(
            columns=[
                "predicted_class",
                "total",
                "need_to_check",
                "need_to_check_rate",
                "median_probability",
                "median_borda_margin",
                "oof_precision",
                "confident_error_burden",
            ]
        )
    for column in ("prediction_class_oof_precision", "prediction_max_probability", "prediction_borda_margin"):
        if column in borda_agree.columns:
            borda_agree[column] = pd.to_numeric(borda_agree[column], errors="coerce")
    named_aggregations: dict[str, tuple[str, Any]] = {
        "total": ("prediction_correct", "size"),
        "need_to_check": ("prediction_correct", lambda values: int((~values).sum())),
    }
    if "prediction_class_oof_precision" in borda_agree.columns:
        named_aggregations["oof_precision"] = ("prediction_class_oof_precision", "median")
    if "prediction_max_probability" in borda_agree.columns:
        named_aggregations["median_probability"] = ("prediction_max_probability", "median")
    if "prediction_borda_margin" in borda_agree.columns:
        named_aggregations["median_borda_margin"] = ("prediction_borda_margin", "median")
    stats = (
        borda_agree.groupby("predicted_label", observed=True)
        .agg(**named_aggregations)
        .reset_index()
        .rename(columns={"predicted_label": "predicted_class"})
    )
    if "oof_precision" not in stats.columns:
        stats["oof_precision"] = np.nan
    if "median_probability" not in stats.columns:
        stats["median_probability"] = np.nan
    if "median_borda_margin" not in stats.columns:
        stats["median_borda_margin"] = np.nan
    stats["need_to_check_rate"] = stats["need_to_check"] / stats["total"]
    stats["confident_error_burden"] = stats["total"] * stats["need_to_check_rate"] * stats["median_probability"]
    return stats.sort_values(
        ["confident_error_burden", "need_to_check_rate", "need_to_check", "total"],
        ascending=[False, False, False, False],
        na_position="last",
    )


def _ordered_levels(values: pd.Series) -> list[str]:
    """Keep the confidence scale stable while retaining unexpected values."""
    observed = [str(value) for value in values.dropna().unique()]
    return [*(_level for _level in _LEVEL_ORDER if _level in observed), *sorted(set(observed) - set(_LEVEL_ORDER))]


def _reason_matrix(need_to_check: pd.DataFrame) -> pd.DataFrame:
    """Cross-tabulate class reliability and rank confidence for audit errors."""
    required = {"prediction_class_reliability_level", "prediction_rank_confidence_level"}
    if need_to_check.empty or not required.issubset(need_to_check.columns):
        return pd.DataFrame()
    rows = _ordered_levels(need_to_check["prediction_class_reliability_level"])
    columns = _ordered_levels(need_to_check["prediction_rank_confidence_level"])
    return pd.crosstab(
        need_to_check["prediction_class_reliability_level"], need_to_check["prediction_rank_confidence_level"]
    ).reindex(index=rows, columns=columns, fill_value=0)


def _confusion_matrix(need_to_check: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
    """Build a readable confusion matrix for the most frequent error labels."""
    if need_to_check.empty:
        return pd.DataFrame()
    label_frequency = pd.concat(
        [need_to_check["reference_label"], need_to_check["predicted_label"]], ignore_index=True
    ).value_counts()
    top_labels = label_frequency.head(top_n).index
    shown = need_to_check.loc[
        need_to_check["reference_label"].isin(top_labels) & need_to_check["predicted_label"].isin(top_labels)
    ]
    matrix = pd.crosstab(shown["reference_label"], shown["predicted_label"])
    matrix = matrix.reindex(index=top_labels, columns=top_labels, fill_value=0)
    return matrix.loc[matrix.sum(axis=1).gt(0), matrix.sum(axis=0).gt(0)]


def _save_diagnostic_overview(group_summary: pd.DataFrame, output_path: Path) -> Path | None:
    """Save the four-way Borda-consensus and correctness overview."""
    if group_summary.empty:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    colors = {
        "Borda agrees + correct": "#2e6f95",
        "Borda agrees + incorrect": "#d97732",
        "Borda disagrees + correct": "#8fb7ce",
        "Borda disagrees + incorrect": "#edb183",
        "Unknown": "#9aa5b1",
    }
    plot = group_summary.iloc[::-1]
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    bars = ax.barh(
        plot["diagnostic_group"].astype(str),
        plot["rows"],
        color=[colors[str(group)] for group in plot["diagnostic_group"]],
        edgecolor="#334e68",
        linewidth=0.6,
    )
    max_rows = max(int(plot["rows"].max()), 1)
    for bar, (_, row) in zip(bars, plot.iterrows()):
        ax.text(
            bar.get_width() + max_rows * 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{int(row['rows']):,} ({row['share']:.1%})",
            va="center",
            fontsize=9,
        )
    ax.set_xlim(0, max_rows * 1.28)
    ax.set_xlabel("Parcels with a known reference label")
    ax.set_ylabel("")
    ax.set_title("Borda consensus and prediction correctness", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#d9e2e8", linewidth=0.7)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def _save_class_risk(class_risk: pd.DataFrame, output_path: Path) -> Path | None:
    """Save need-to-check rate and support by predicted class."""
    if class_risk.empty:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot = class_risk.sort_values("need_to_check_rate", ascending=True)
    fig_height = min(18.0, max(6.0, 0.48 * len(plot) + 2.0))
    fig, ax = plt.subplots(figsize=(13.5, fig_height))
    bars = ax.barh(
        plot["predicted_class"].astype(str), plot["need_to_check_rate"], color="#2e6f95", edgecolor="#173f5f", linewidth=0.6
    )
    max_rate = max(float(plot["need_to_check_rate"].max()), 0.01)
    axis_max = min(1.0, max_rate * 1.48 + 0.02)
    for bar, (_, row) in zip(bars, plot.iterrows()):
        precision = row["oof_precision"]
        precision_text = f"{precision:.1%}" if pd.notna(precision) else "n/a"
        ax.text(
            min(bar.get_width() + axis_max * 0.008, axis_max * 0.985),
            bar.get_y() + bar.get_height() / 2,
            f"{int(row['need_to_check']):,}/{int(row['total']):,} | OOF {precision_text}",
            va="center",
            fontsize=8,
        )
    ax.set_xlim(0, axis_max)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1))
    ax.set_xlabel("Incorrect predictions among Borda-agreement parcels")
    ax.set_ylabel("Predicted class")
    ax.set_title("Need-to-check rate by predicted class", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#d9e2e8", linewidth=0.7)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def _save_distribution_comparison(confidence_audit: pd.DataFrame, output_path: Path) -> Path | None:
    """Save ECDF comparisons for the confidence components."""
    available_metrics = [
        column for column in _CONFIDENCE_METRICS if column in confidence_audit.columns and confidence_audit[column].notna().any()
    ]
    if confidence_audit.empty or not available_metrics:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    palette = {"Correct": "#2e6f95", "Needs check": "#d97732"}
    for axis, column in zip(axes.flat, available_metrics):
        plot = confidence_audit[[column, "audit_group"]].dropna()
        sns.ecdfplot(
            data=plot, x=column, hue="audit_group", hue_order=["Correct", "Needs check"], palette=palette, linewidth=2.1, ax=axis
        )
        axis.set_title(_CONFIDENCE_METRICS[column], loc="left", fontweight="bold")
        axis.set_xlabel("")
        axis.set_ylabel("Cumulative share")
        axis.yaxis.set_major_formatter(mtick.PercentFormatter(1))
        axis.grid(color="#d9e2e8", linewidth=0.7)
    for axis in axes.flat[len(available_metrics) :]:
        axis.set_visible(False)
    fig.suptitle(
        "Confidence distributions: correct versus needs-check predictions", x=0.06, ha="left", fontsize=15, fontweight="bold"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def _save_joint_behavior(confidence_audit: pd.DataFrame, output_path: Path, probability_threshold: float) -> Path | None:
    """Save empirical error risk beside supporting probability-consensus density."""
    required = {"prediction_max_probability", "prediction_borda_margin", "audit_error"}
    if confidence_audit.empty or not required.issubset(confidence_audit.columns):
        return None
    joint_data = confidence_audit[list(required)].dropna().copy()
    if joint_data.empty:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    minimum_hex_rows = min(20, max(1, len(joint_data) // 10))
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), sharex=True, sharey=True)
    risk_hex = axes[0].hexbin(
        joint_data["prediction_max_probability"],
        joint_data["prediction_borda_margin"],
        C=joint_data["audit_error"].astype(float),
        reduce_C_function=np.mean,
        gridsize=11,
        mincnt=minimum_hex_rows,
        cmap=sns.light_palette("#d97732", as_cmap=True),
        vmin=0,
        vmax=1,
        linewidths=0.35,
    )
    risk_colorbar = fig.colorbar(risk_hex, ax=axes[0])
    risk_colorbar.set_label("Observed needs-check rate")
    risk_colorbar.ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
    count_hex = axes[1].hexbin(
        joint_data["prediction_max_probability"],
        joint_data["prediction_borda_margin"],
        gridsize=11,
        mincnt=1,
        bins="log",
        cmap=sns.light_palette("#2e6f95", as_cmap=True),
        linewidths=0.35,
    )
    count_colorbar = fig.colorbar(count_hex, ax=axes[1])
    count_colorbar.set_label("Parcels per hexagon (log scale)")
    axes[0].set_title(f"Empirical error rate (at least {minimum_hex_rows:,} parcels per hexagon)", loc="left", fontweight="bold")
    axes[1].set_title("Data support", loc="left", fontweight="bold")
    for axis in axes:
        axis.axvline(probability_threshold, color="#555555", linestyle="--", linewidth=1.4)
        axis.set_xlim(0, 1)
        axis.set_xlabel("Maximum predicted probability")
        axis.grid(False)
    axes[0].set_ylabel("Borda winner margin")
    fig.suptitle("Joint probability-consensus behavior", x=0.05, ha="left", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def _save_class_confidence_risk(class_risk: pd.DataFrame, output_path: Path, probability_threshold: float) -> Path | None:
    """Save the predicted-class confidence, error, support, and reliability profile."""
    required = {"total", "need_to_check_rate", "median_probability"}
    if class_risk.empty or not required.issubset(class_risk.columns):
        return None
    plot = class_risk.dropna(subset=["need_to_check_rate", "median_probability"]).copy()
    if plot.empty:
        return None
    minimum_class_rows = max(30, int(0.001 * int(plot["total"].sum())))
    supported = plot.loc[plot["total"].ge(minimum_class_rows)].copy()
    if supported.empty:
        supported = plot.copy()
        minimum_class_rows = int(supported["total"].min())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    maximum_support = max(float(supported["total"].max()), 1.0)
    point_sizes = 50 + 850 * np.sqrt(supported["total"] / maximum_support)
    fig, ax = plt.subplots(figsize=(13, 8))
    precision = pd.to_numeric(supported.get("oof_precision"), errors="coerce")
    if precision.notna().any():
        color_values = precision.fillna(precision.median())
        points = ax.scatter(
            supported["median_probability"],
            supported["need_to_check_rate"],
            s=point_sizes,
            c=color_values,
            cmap=sns.light_palette("#2e6f95", as_cmap=True),
            alpha=0.82,
            edgecolor="white",
            linewidth=0.8,
        )
        colorbar = fig.colorbar(points, ax=ax)
        colorbar.set_label("OOF precision of predicted class")
        colorbar.ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
    else:
        ax.scatter(
            supported["median_probability"],
            supported["need_to_check_rate"],
            s=point_sizes,
            color="#2e6f95",
            alpha=0.82,
            edgecolor="white",
            linewidth=0.8,
        )
    overall_error_rate = float(class_risk["need_to_check"].sum() / class_risk["total"].sum())
    ax.axhline(overall_error_rate, color="#555555", linestyle=":", linewidth=1.5)
    ax.axvline(probability_threshold, color="#555555", linestyle="--", linewidth=1.5)
    label_rows = supported.nlargest(min(10, len(supported)), "confident_error_burden")
    for _, row in label_rows.iterrows():
        ax.annotate(
            str(row["predicted_class"]),
            (row["median_probability"], row["need_to_check_rate"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
    ax.set_xlabel("Median maximum predicted probability")
    ax.set_ylabel("Observed needs-check rate")
    ax.set_title(
        f"Predicted-class risk profile (classes with at least {minimum_class_rows:,} parcels)", loc="left", fontweight="bold"
    )
    ax.grid(color="#d9e2e8", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def _save_reason_heatmap(reason_matrix: pd.DataFrame, output_path: Path) -> Path | None:
    """Save the class-reliability by rank-confidence matrix."""
    if reason_matrix.empty:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = int(reason_matrix.to_numpy().sum())
    annotations = reason_matrix.astype(object).copy()
    for row_index in range(reason_matrix.shape[0]):
        for column_index in range(reason_matrix.shape[1]):
            count = int(reason_matrix.iat[row_index, column_index])
            share = count / total if total else 0.0
            annotations.iat[row_index, column_index] = f"{count:,}\n{share:.1%}"
    fig, ax = plt.subplots(figsize=(8.8, 6.6))
    sns.heatmap(
        reason_matrix,
        annot=annotations,
        fmt="",
        cmap=sns.light_palette("#2e6f95", as_cmap=True),
        linewidths=0.7,
        linecolor="white",
        cbar_kws={"label": "Needs-check parcels"},
        ax=ax,
    )
    ax.set_xlabel("Rank / ensemble confidence")
    ax.set_ylabel("Predicted-class reliability")
    ax.set_title("Confidence components of incorrect consensus predictions", loc="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def _save_confusion_heatmap(confusion: pd.DataFrame, output_path: Path) -> Path | None:
    """Save a normalized confusion heatmap with exact count annotations."""
    if confusion.empty:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_share = confusion.div(confusion.sum(axis=1).replace(0, np.nan), axis=0)
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(
        row_share,
        mask=confusion.eq(0),
        annot=confusion,
        fmt=".0f",
        cmap=sns.light_palette("#2e6f95", as_cmap=True),
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Share of displayed errors within true class"},
        ax=ax,
    )
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Need-to-check true-versus-predicted class confusions", loc="left", fontweight="bold")
    ax.tick_params(axis="x", labelrotation=90)
    ax.tick_params(axis="y", labelrotation=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def _save_reason_sankey(need_to_check: pd.DataFrame, output_path: Path) -> Path | None:
    """Save the confidence-component flow as a self-contained Plotly HTML file."""
    required = {
        "predicted_label",
        "prediction_class_reliability_level",
        "prediction_rank_confidence_level",
        "prediction_confidence_level",
    }
    if need_to_check.empty or not required.issubset(need_to_check.columns):
        return None
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    plot = need_to_check.copy()
    top_predicted = plot["predicted_label"].value_counts().head(10).index
    plot["sankey_predicted_label"] = plot["predicted_label"].where(
        plot["predicted_label"].isin(top_predicted), "Other predicted classes"
    )
    stages = (
        ("sankey_predicted_label", "Predicted"),
        ("prediction_class_reliability_level", "Class reliability"),
        ("prediction_rank_confidence_level", "Rank confidence"),
        ("prediction_confidence_level", "Final confidence"),
    )
    node_labels: list[str] = []
    node_colors: list[str] = []
    node_lookup: dict[tuple[str, str], int] = {}
    stage_colors = ("#2e6f95", "#7a9a48", "#d99a2b", "#c66b3d")
    for stage_index, (column, prefix) in enumerate(stages):
        values = plot[column].fillna("Unknown").astype(str).value_counts().index
        for value in values:
            node_lookup[(column, value)] = len(node_labels)
            node_labels.append(f"{prefix}: {value}")
            node_colors.append(stage_colors[stage_index])

    sources: list[int] = []
    targets: list[int] = []
    values: list[int] = []
    for (left_column, _), (right_column, _) in zip(stages[:-1], stages[1:]):
        flows = (
            plot.assign(
                **{
                    left_column: plot[left_column].fillna("Unknown").astype(str),
                    right_column: plot[right_column].fillna("Unknown").astype(str),
                }
            )
            .groupby([left_column, right_column], observed=True)
            .size()
            .reset_index(name="count")
        )
        for _, row in flows.iterrows():
            sources.append(node_lookup[(left_column, str(row[left_column]))])
            targets.append(node_lookup[(right_column, str(row[right_column]))])
            values.append(int(row["count"]))

    figure = go.Figure(
        go.Sankey(
            arrangement="snap",
            node={"label": node_labels, "color": node_colors, "pad": 14, "thickness": 17},
            link={"source": sources, "target": targets, "value": values, "color": "rgba(46,111,149,0.22)"},
        )
    )
    figure.update_layout(
        title="Need-to-check confidence flow",
        height=780,
        font={"family": "Segoe UI, sans-serif", "size": 11, "color": "#1f2933"},
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin={"l": 24, "r": 24, "t": 72, "b": 24},
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(output_path, include_plotlyjs=True, full_html=True)
    return output_path


def export_prediction_confidence_diagnostics(
    predictions: pd.DataFrame,
    target_column: str,
    prediction_column: str,
    output_dir: str | Path,
    *,
    include: Iterable[str] = _AVAILABLE_VISUALS,
    probability_threshold: float = 0.60,
) -> ConfidenceDiagnosticsArtifacts | None:
    """Build source tables and save the requested confidence diagnostics."""
    selected = set(include)
    unknown_visuals = sorted(selected.difference(_AVAILABLE_VISUALS))
    if unknown_visuals:
        raise ValueError(f"Error: Unsupported confidence diagnostic visuals: {unknown_visuals}")
    base_required = {target_column, prediction_column, "prediction_borda_winner"}
    if not base_required.issubset(predictions.columns):
        return None

    diagnostics = prepare_confidence_diagnostics(predictions, target_column, prediction_column)
    if diagnostics.empty:
        return None
    confidence_audit = _confidence_audit(diagnostics)
    need_to_check = confidence_audit.loc[confidence_audit["audit_error"]].copy()
    summary, group_summary = _diagnostic_summary(diagnostics)
    class_risk = _class_risk_table(diagnostics)
    median_summary = _confidence_median_summary(confidence_audit)
    reason_matrix = _reason_matrix(need_to_check)
    confusion = _confusion_matrix(need_to_check)

    root = Path(output_dir)
    plots_dir = root / "plots"
    data_dir = root / "data"
    plots_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "diagnostic_group_summary": group_summary,
        "confidence_metric_median_summary": median_summary,
        "need_to_check_by_predicted_class": class_risk,
        "need_to_check_reason_matrix": reason_matrix,
        "need_to_check_confusion": confusion,
    }
    for name, table in tables.items():
        table.to_csv(data_dir / f"{name}.csv", index=name.endswith("matrix") or name.endswith("confusion"))

    image_builders = {
        "overview": (_save_diagnostic_overview, group_summary, plots_dir / "borda_consensus_correctness_overview.png"),
        "class_rate": (_save_class_risk, class_risk, plots_dir / "need_to_check_rate_by_predicted_class.png"),
        "reason_heatmap": (_save_reason_heatmap, reason_matrix, plots_dir / "need_to_check_confidence_components.png"),
        "confusion": (_save_confusion_heatmap, confusion, plots_dir / "need_to_check_class_confusion.png"),
        "distribution": (_save_distribution_comparison, confidence_audit, plots_dir / "confidence_metric_distributions.png"),
    }
    images: dict[str, Path] = {}
    for name, (builder, table, output_path) in image_builders.items():
        if name in selected:
            saved_path = builder(table, output_path)
            if saved_path is not None:
                images[name] = saved_path

    if "joint_behavior" in selected:
        joint_path = _save_joint_behavior(
            confidence_audit, plots_dir / "probability_borda_joint_behavior.png", probability_threshold
        )
        if joint_path is not None:
            images["joint_behavior"] = joint_path
    if "class_confidence_risk" in selected:
        class_confidence_path = _save_class_confidence_risk(
            class_risk, plots_dir / "class_confidence_risk.png", probability_threshold
        )
        if class_confidence_path is not None:
            images["class_confidence_risk"] = class_confidence_path

    embeds: dict[str, Path] = {}
    if "reason_sankey" in selected:
        sankey_path = _save_reason_sankey(need_to_check, plots_dir / "need_to_check_confidence_flow.html")
        if sankey_path is not None:
            embeds["reason_sankey"] = sankey_path

    return ConfidenceDiagnosticsArtifacts(summary=summary, images=images, embeds=embeds, tables=tables)
