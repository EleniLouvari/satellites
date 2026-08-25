"""HTML rendering for EDA artifacts."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

import pandas as pd

from ..core.config import EDAConfig
from ..core.io import ensure_dir


def write_eda_html_report(artifacts: dict[str, Any], plot_paths: dict[str, Path], config: EDAConfig) -> Path:
    """Render a complete tabbed HTML EDA report."""
    ensure_dir(config.output_dir)
    tabs = _build_report_tabs(artifacts, plot_paths)
    body = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "<meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>{escape(config.report_title)}</title>",
        _style(),
        "</head>",
        "<body>",
        "<main>",
        f"<h1>{escape(config.report_title)}</h1>",
        "<p class='intro'>Automated exploratory analysis for pandas DataFrames and GeoPandas GeoDataFrames.</p>",
        _render_section({"title": "Dataset Summary", "kv": artifacts["summary"]}, config.output_dir),
        (
            "<button class='toggle-button' type='button' aria-expanded='false' "
            "onclick=\"toggleDetails('data-quality-details', this)\">Show Details</button>"
        ),
        "<div id='data-quality-details' class='details-panel hidden'>",
        _render_section(
            {
                "title": "Column Profile and Missing Values",
                "table": artifacts["column_profile"],
                "images": _select_images(plot_paths, ["missing_values"]),
            },
            config.output_dir,
        ),
        _render_section(
            {
                "title": "Data Quality Flags",
                "table": artifacts["data_quality_flags"],
                "images": _select_images(plot_paths, ["data_quality_flags"]),
            },
            config.output_dir,
        ),
        _render_section(
            {
                "title": "Missingness Summary",
                "table": artifacts["missingness_summary"],
                "images": _select_images(plot_paths, ["missingness_heatmap"]),
            },
            config.output_dir,
        ),
        _render_section(
            {
                "title": "Missingness vs Target",
                "table": artifacts["missingness_target_tests"],
                "images": _select_images(plot_paths, ["missingness_target_associations"]),
            },
            config.output_dir,
        ),
        "</div>",
        _render_tabs(tabs, config.output_dir),
        _script(),
        "</main>",
        "</body>",
        "</html>",
    ]
    config.report_path.write_text("\n".join(body), encoding="utf-8")
    return config.report_path


def _build_report_tabs(artifacts: dict[str, Any], plot_paths: dict[str, Path]) -> list[dict[str, Any]]:
    """Build tab definitions grouped by column type."""
    column_profile = artifacts["column_profile"]
    target_summary = artifacts["target_summary"]
    target_tab = _target_tab_key(artifacts)
    max_features_per_plot = artifacts["summary"]["config"]["max_features_per_plot"]

    categorical_sections = [
        {
            "title": "Categorical Column Profile",
            "table": _profile_for_types(column_profile, ["categorical", "boolean", "datetime"]),
        },
        {"title": "Datetime Summary", "table": artifacts["datetime_summary"]},
        {
            "title": "Categorical Summary",
            "table": _display_categorical_summary(artifacts["categorical_summary"]),
            "images": _select_images(plot_paths, ["categorical_distributions"]),
        },
        {
            "title": "Categorical Association Tests",
            "table": artifacts["categorical_associations"],
            "images": _select_images(plot_paths, ["categorical_association_heatmap"]),
        },
        {
            "title": "Categorical vs Target Tests",
            "table": artifacts["categorical_target_tests"],
            "images": _select_images(plot_paths, ["categorical_target_associations"]),
        },
    ]
    if target_tab == "categorical" and not target_summary.empty:
        categorical_sections.append(_target_summary_section(target_summary, plot_paths))

    numerical_subtabs = [
        {
            "id": "numerical-general",
            "label": "General",
            "sections": [
                {"title": "Numerical Column Profile", "table": _profile_for_types(column_profile, ["numeric"])},
                {"title": "Numerical Summary", "table": artifacts["numeric_summary"]},
                {"title": "Robust Numerical Summary", "table": artifacts["robust_numeric_summary"]},
            ],
        },
        {
            "id": "numerical-normality",
            "label": "Normality",
            "sections": [
                {
                    "title": "Normality Tests",
                    "text": (
                        f"The table covers all analyzed numeric columns; the QQ plot shows the top {max_features_per_plot} "
                        "ML-screened features."
                    ),
                    "table": artifacts["normality_tests"],
                    "images": _select_images_with_prefix(plot_paths, "numeric_qq_plots"),
                }
            ],
        },
        {
            "id": "numerical-outliers",
            "label": "Outliers",
            "sections": [
                {
                    "title": "Outlier Summary",
                    "table": artifacts["outlier_summary"],
                    "images": _select_images(plot_paths, ["numeric_distributions"]),
                },
                {
                    "title": "Multivariate Outliers",
                    "text": (
                        "This table contains every row flagged at the 97.5% chi-square threshold, not merely a top-N "
                        "distance preview. "
                        "The final annotated dataset also retains scores and False flags for all scorable rows."
                    ),
                    "table": artifacts["multivariate_outliers"],
                    "show_all_rows": True,
                },
            ],
        },
        {
            "id": "numerical-correlation",
            "label": "Correlation Analysis",
            "sections": [
                {
                    "title": "Strong Pearson Correlation Pairs",
                    "table": artifacts["strong_correlation_pairs"],
                    "images": _select_images(plot_paths, ["pearson_correlation_heatmap", "spearman_correlation_heatmap"]),
                },
                {
                    "title": "Multicollinearity VIF",
                    "table": artifacts["multicollinearity"],
                    "images": _select_images(plot_paths, ["vif_multicollinearity"]),
                },
            ],
        },
        {
            "id": "numerical-multivariate",
            "label": "Multivariate Analysis",
            "sections": [
                {"title": "Numeric Scatter Matrix", "images": _select_images(plot_paths, ["numeric_scatter_matrix"])},
                {
                    "title": "PCA Summary",
                    "table": artifacts["pca_summary"],
                    "images": _select_images(plot_paths, ["pca_explained_variance", "pca_score_plot"]),
                },
                {
                    "title": "Numeric Target Correlations",
                    "table": artifacts["numeric_target_correlations"],
                    "images": _select_images(plot_paths, ["numeric_target_correlations", "numeric_target_scatter"]),
                },
                {
                    "title": "Numeric Target-Class Median Profile",
                    "text": (
                        "Each heatmap cell is the target class median's percentile within the feature's full distribution. "
                        "Values above 50 indicate relatively high class medians; values below 50 indicate relatively low ones."
                    ),
                    "table": artifacts["numeric_target_tests"],
                    "images": _select_images(plot_paths, ["numeric_target_median_percentile_heatmap"]),
                },
                {
                    "title": "Categorical Features vs Numeric Target",
                    "table": artifacts["categorical_numeric_target_tests"],
                    "images": _select_images(plot_paths, ["numeric_target_by_categorical_features"]),
                },
            ],
        },
    ]
    if target_tab == "numerical" and not target_summary.empty:
        numerical_subtabs[0]["sections"].append(_target_summary_section(target_summary, plot_paths))

    tabs = [
        {"id": "categorical", "label": "Categorical", "sections": categorical_sections},
        {"id": "numerical", "label": "Numerical", "subtabs": numerical_subtabs},
    ]
    if target_tab:
        tabs.insert(
            0,
            {
                "id": "feature-selection",
                "label": "Feature Selection",
                "sections": [
                    {
                        "title": "Proposed Initial Feature Set",
                        "text": artifacts["feature_selection_message"],
                        "table": artifacts["feature_selection_proposal"],
                    },
                    {
                        "title": "Ordered Selection Criteria",
                        "text": (
                            "Apply these gates in order; the last model-validation gate must be fitted using training data only."
                        ),
                        "table": artifacts["feature_selection_criteria"],
                    },
                    {
                        "title": "Feature-Target Association Ranking",
                        "text": (
                            "Univariate screening combines effect sizes with false-discovery-rate adjusted p-values. "
                            "Use it to prioritize candidates, then validate them with cross-validation and redundancy checks."
                        ),
                        "table": artifacts["feature_target_associations"],
                        "images": _select_images(plot_paths, ["feature_target_association_ranking"]),
                    },
                    {
                        "title": "Feature Distributions Compared with the Target",
                        "text": (
                            f"Plots show at most {max_features_per_plot} features, ordered by the shared ML-screening priority."
                        ),
                        "images": _select_images(
                            plot_paths,
                            [
                                "numeric_target_distribution_comparison",
                                "categorical_target_composition",
                                "numeric_target_scatter",
                                "numeric_target_by_categorical_features",
                            ],
                        ),
                    },
                ],
            },
        )

    if not artifacts["geospatial_summary"].empty:
        geometry_sections = [
            {"title": "Geometry Column Profile", "table": _profile_for_types(column_profile, ["geometry"])},
            {"title": "Geometry Summary", "table": artifacts["geospatial_summary"]},
        ]
        if "geometry_target_heatmap" in plot_paths:
            geometry_sections.append(
                {"title": "Target Heatmap", "images": _select_images(plot_paths, ["geometry_target_heatmap"])}
            )
        if target_tab == "geometry" and not target_summary.empty:
            geometry_sections.append(_target_summary_section(target_summary, plot_paths))
        tabs.append({"id": "geometry", "label": "Geometry", "sections": geometry_sections})

    return tabs


def _correlation_matrix_table(artifacts: dict[str, Any], method: str) -> pd.DataFrame:
    """Return a display-ready correlation matrix for one method."""
    matrix = artifacts["correlation_matrices"][method]
    return matrix.reset_index(names="column") if not matrix.empty else pd.DataFrame()


def _target_summary_section(target_summary: pd.DataFrame, plot_paths: dict[str, Path]) -> dict[str, Any]:
    """Build the target table/plot section."""
    return {"title": "Target Summary", "table": target_summary, "images": _select_images(plot_paths, ["target_distribution"])}


def _render_tabs(tabs: list[dict[str, Any]], report_dir: Path) -> str:
    """Render tab buttons and tab panels."""
    if not tabs:
        return ""
    parts = ["<section class='tab-shell'>", "<div class='tab-list' role='tablist'>"]
    for index, tab in enumerate(tabs):
        active_class = " active" if index == 0 else ""
        selected = "true" if index == 0 else "false"
        parts.append(
            f"<button class='tab-button{active_class}' role='tab' aria-selected='{selected}' "
            f"aria-controls='panel-{escape(tab['id'])}' id='tab-{escape(tab['id'])}' "
            f"onclick=\"showEdaTab('{escape(tab['id'])}')\">{escape(tab['label'])}</button>"
        )
    parts.append("</div>")
    for index, tab in enumerate(tabs):
        active_class = " active" if index == 0 else ""
        parts.append(
            f"<div class='tab-panel{active_class}' role='tabpanel' id='panel-{escape(tab['id'])}' "
            f"aria-labelledby='tab-{escape(tab['id'])}'>"
        )
        if tab.get("subtabs"):
            parts.append(_render_subtabs(tab["id"], tab["subtabs"], report_dir))
        else:
            for section in tab["sections"]:
                parts.append(_render_section(section, report_dir, nested=True))
        parts.append("</div>")
    parts.append("</section>")
    return "\n".join(parts)


def _render_subtabs(parent_id: str, subtabs: list[dict[str, Any]], report_dir: Path) -> str:
    """Render a second-level tab set inside a top-level report tab."""
    if not subtabs:
        return ""
    parent = escape(parent_id)
    parts = [f"<section class='subtab-shell' data-parent-tab='{parent}'>", "<div class='subtab-list' role='tablist'>"]
    for index, subtab in enumerate(subtabs):
        active_class = " active" if index == 0 else ""
        selected = "true" if index == 0 else "false"
        subtab_id = escape(subtab["id"])
        parts.append(
            f"<button class='subtab-button{active_class}' role='tab' aria-selected='{selected}' "
            f"aria-controls='panel-{subtab_id}' id='tab-{subtab_id}' "
            f"onclick=\"showEdaSubtab('{parent}', '{subtab_id}')\">{escape(subtab['label'])}</button>"
        )
    parts.append("</div>")
    for index, subtab in enumerate(subtabs):
        active_class = " active" if index == 0 else ""
        subtab_id = escape(subtab["id"])
        parts.append(
            f"<div class='subtab-panel{active_class}' role='tabpanel' id='panel-{subtab_id}' aria-labelledby='tab-{subtab_id}'>"
        )
        for section in subtab["sections"]:
            parts.append(_render_section(section, report_dir, nested=True))
        parts.append("</div>")
    parts.append("</section>")
    return "\n".join(parts)


def _section(title: str, **kwargs: Any) -> dict[str, Any]:
    """Build a section dictionary."""
    return {"title": title, **kwargs}


def _render_section(section: dict[str, Any], report_dir: Path, nested: bool = False) -> str:
    """Render a report section."""
    class_name = "subsection" if nested else "report-section"
    parts = [f"<section class='{class_name}'><h2>{escape(section['title'])}</h2>"]
    if section.get("text"):
        parts.append(f"<p>{escape(section['text'])}</p>")
    if section.get("kv"):
        parts.append(_render_key_values(section["kv"]))
    if section.get("table") is not None:
        max_rows = None if section.get("show_all_rows") else 100
        parts.append(_render_table(section["table"], max_rows=max_rows))
    if section.get("images"):
        parts.append(_render_images(section["images"], report_dir))
    parts.append("</section>")
    return "\n".join(parts)


def _render_key_values(values: dict[str, Any]) -> str:
    """Render key-value metadata."""
    rows = []
    rendered_rows = []
    for key, value in values.items():
        if key == "config":
            value = {k: str(v) for k, v in value.items() if k not in {"artifact_names"}}
        key_text = str(key)
        value_text = _stringify(value)
        rendered_rows.append((key_text, value_text))
        rows.append(f"<tr><th>{escape(key_text)}</th><td>{escape(value_text)}</td></tr>")
    key_width = _content_width_ch([row[0] for row in rendered_rows])
    value_width = _content_width_ch([row[1] for row in rendered_rows])
    colgroup = f"<colgroup><col style='width: {key_width}ch;'><col style='width: {value_width}ch;'></colgroup>"
    return "<div class='table-scroll'><table class='kv-table'>" + colgroup + "".join(rows) + "</table></div>"


def _render_table(table: pd.DataFrame, max_rows: int | None = 100) -> str:
    """Render a dataframe preview as HTML."""
    if table is None or table.empty:
        return "<p class='muted'>No rows to display.</p>"
    preview = table.copy()
    if max_rows is not None and len(preview) > max_rows:
        preview = preview.head(max_rows)
    column_widths = _dataframe_column_widths(preview)
    parts = ["<div class='table-scroll'><table class='data-table'><colgroup>"]
    for column in preview.columns:
        parts.append(f"<col style='width: {column_widths[column]}ch;'>")
    parts.append("</colgroup><thead><tr>")
    for column in preview.columns:
        parts.append(f"<th>{escape(str(column))}</th>")
    parts.append("</tr></thead><tbody>")
    for _, row in preview.iterrows():
        parts.append("<tr>")
        for column in preview.columns:
            parts.append(f"<td>{escape(_stringify(row[column]))}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts)


def _render_images(images: dict[str, Path], report_dir: Path) -> str:
    """Render generated plot images."""
    parts = ["<div class='image-grid'>"]
    for title, path in images.items():
        if not Path(path).exists():
            continue
        image_path = Path(path)
        rel_path = (
            image_path.relative_to(report_dir).as_posix() if image_path.is_relative_to(report_dir) else image_path.as_posix()
        )
        parts.append("<figure>")
        parts.append(f"<img src='{escape(rel_path)}' alt='{escape(title)}'>")
        parts.append(f"<figcaption>{escape(title.replace('_', ' ').title())}</figcaption>")
        parts.append("</figure>")
    parts.append("</div>")
    return "".join(parts)


def _select_images(plot_paths: dict[str, Path], keys: list[str]) -> dict[str, Path]:
    """Select a subset of plots for a report group."""
    return {key: plot_paths[key] for key in keys if key in plot_paths}


def _select_images_with_prefix(plot_paths: dict[str, Path], prefix: str) -> dict[str, Path]:
    """Select every ordered plot page belonging to one paginated visual family."""
    return {key: path for key, path in plot_paths.items() if key.startswith(prefix)}


def _profile_for_types(column_profile: pd.DataFrame, logical_types: list[str]) -> pd.DataFrame:
    """Filter column profile rows by logical type."""
    if column_profile.empty or "logical_type" not in column_profile.columns:
        return pd.DataFrame()
    return column_profile[column_profile["logical_type"].isin(logical_types)].reset_index(drop=True)


def _target_tab_key(artifacts: dict[str, Any]) -> str | None:
    """Return the tab where target diagnostics belong."""
    target_column = artifacts["summary"].get("target_column")
    column_profile = artifacts["column_profile"]
    if not target_column or column_profile.empty:
        return None
    target_rows = column_profile[column_profile["column"] == target_column]
    if target_rows.empty:
        return None
    logical_type = str(target_rows.iloc[0]["logical_type"])
    if logical_type == "numeric":
        return "numerical"
    if logical_type == "geometry":
        return "geometry"
    return "categorical"


def _display_categorical_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Make nested top-values dictionaries easier to scan in HTML."""
    if table.empty or "top_values" not in table.columns:
        return table
    output = table.copy()
    output["top_values"] = output["top_values"].apply(_stringify)
    return output


def _stringify(value: Any) -> str:
    """Convert values to readable report strings."""
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, dict):
        return ", ".join(f"{key}: {val}" for key, val in value.items())
    return str(value)


def _dataframe_column_widths(table: pd.DataFrame) -> dict[str, int]:
    """Calculate per-column widths from the longest rendered header or value."""
    widths = {}
    for column in table.columns:
        values = [str(column)] + [_stringify(value) for value in table[column]]
        widths[column] = _content_width_ch(values)
    return widths


def _content_width_ch(values: list[str]) -> int:
    """Estimate HTML column width in ch units from max rendered cell length."""
    if not values:
        return 8
    return max(8, max(len(value) for value in values) + 2)


def _script() -> str:
    """Return lightweight tab behavior JavaScript."""
    return """
<script>
function showEdaTab(tabId) {
    const buttons = document.querySelectorAll('.tab-button');
    const panels = document.querySelectorAll('.tab-panel');
    buttons.forEach((button) => {
        const isActive = button.id === `tab-${tabId}`;
        button.classList.toggle('active', isActive);
        button.setAttribute('aria-selected', isActive ? 'true' : 'false');
    });
    panels.forEach((panel) => {
        panel.classList.toggle('active', panel.id === `panel-${tabId}`);
    });
}

function showEdaSubtab(parentTabId, subtabId) {
    const shell = document.querySelector(`#panel-${parentTabId} .subtab-shell`);
    if (!shell) {
        return;
    }
    const buttons = shell.querySelectorAll('.subtab-button');
    const panels = shell.querySelectorAll('.subtab-panel');
    buttons.forEach((button) => {
        const isActive = button.id === `tab-${subtabId}`;
        button.classList.toggle('active', isActive);
        button.setAttribute('aria-selected', isActive ? 'true' : 'false');
    });
    panels.forEach((panel) => {
        panel.classList.toggle('active', panel.id === `panel-${subtabId}`);
    });
}

function toggleDetails(detailsId, button) {
    const details = document.getElementById(detailsId);
    if (!details) {
        return;
    }
    const isHidden = details.classList.toggle('hidden');
    button.textContent = isHidden ? 'Show Details' : 'Hide Details';
    button.setAttribute('aria-expanded', isHidden ? 'false' : 'true');
}
</script>
"""


def _style() -> str:
    """Return report CSS."""
    return """
<style>
body { margin: 0; background: #f7f8f5; color: #1f2933; font-family: 'Segoe UI', Arial, sans-serif; }
main { max-width: 1320px; margin: 0 auto; padding: 32px 20px 64px; }
h1 { margin: 0 0 8px; color: #16302b; }
h2 { color: #16302b; margin-top: 0; }
.intro { margin-bottom: 24px; color: #52606d; }
.toggle-button { appearance: none; border: 1px solid #1f6f5b; background: #1f6f5b; color: #ffffff;
border-radius: 6px; cursor: pointer; font: inherit; font-weight: 600; padding: 10px 16px; margin: 0 0 4px; }
.toggle-button:hover { background: #185a49; border-color: #185a49; }
.hidden { display: none; }
.details-panel { margin-top: 14px; }
.report-section, .tab-shell { background: #ffffff; border: 1px solid #dde5df; border-radius: 8px; padding: 20px;
margin: 18px 0; box-shadow: 0 8px 22px rgba(35, 48, 43, 0.06); }
.subsection { border: 1px solid #dde5df; border-radius: 8px; padding: 18px; margin: 16px 0; background: #ffffff; }
.subsection h2 { font-size: 1.05rem; }
.tab-list, .subtab-list { display: flex; flex-wrap: wrap; gap: 8px; border-bottom: 1px solid #dde5df;
margin: -2px 0 18px; padding-bottom: 10px; }
.tab-button, .subtab-button { appearance: none; border: 1px solid #b9c8bf; background: #eef3ef; color: #243b35;
border-radius: 6px; cursor: pointer; font: inherit; font-weight: 600; padding: 9px 14px; }
.tab-button:hover, .subtab-button:hover { background: #e0ebe4; }
.tab-button.active, .subtab-button.active { background: #1f6f5b; border-color: #1f6f5b; color: #ffffff; }
.tab-panel, .subtab-panel { display: none; }
.tab-panel.active, .subtab-panel.active { display: block; }
.subtab-shell { margin-top: 4px; }
.table-scroll { overflow: auto; max-height: 70vh; border: 1px solid #e5e7eb; border-radius: 8px; }
table { border-collapse: collapse; width: max-content; min-width: 100%; table-layout: fixed; }
th, td { border-bottom: 1px solid #e5e7eb; padding: 10px 12px; text-align: left; vertical-align: top; white-space: nowrap; }
th { background: #eef3ef; position: sticky; top: 0; z-index: 1; color: #334e48; }
.data-table th, .data-table td, .kv-table th, .kv-table td { overflow: visible; }
.image-grid { display: grid; grid-template-columns: 1fr; gap: 18px; margin-top: 14px; }
figure { margin: 0; background: #f8faf8; border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px; }
img { display: block; max-width: 100%; margin: 0 auto; border-radius: 4px; }
figcaption { margin-top: 8px; color: #52606d; }
.muted { color: #7b8794; }
</style>
"""
