"""Utilities to render structured pipeline outputs into styled HTML reports."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

import pandas as pd

from ..core.persistence import append_log, ensure_dir

# Escape externally supplied text before interpolating it into generated HTML.


def write_html_report(output_path: str | Path, title: str, intro: str, sections: list[dict[str, Any]]) -> None:
    """Render a complete HTML report page from structured section data.

    The function accepts a small, structured representation of report
    sections (key/value blocks, tables, images, embeds, links) and writes a
    standalone HTML file. All external text is escaped to avoid accidental
    HTML injection when reports include user-provided values.
    """
    # Build the page head and body incrementally for predictable output ordering.
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    body = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "<meta charset='utf-8'>",
        f"<title>{escape(title)}</title>",
        _style_block(),
        "</head>",
        "<body>",
        "<main class='page'>",
        f"<h1>{escape(title)}</h1>",
        f"<p class='intro'>{escape(intro)}</p>",
    ]
    for section in sections:
        body.append("<section class='card'>")
        body.append(f"<h2>{escape(section['title'])}</h2>")
        if section.get("text"):
            body.append(f"<p>{escape(section['text'])}</p>")
        if section.get("kv"):
            body.append(_render_key_values(section["kv"]))
        if section.get("table") is not None:
            body.append(
                _render_table(
                    section["table"],
                    highlight_rows_where=section.get("highlight_rows_where"),
                    row_styles_where=section.get("row_styles_where"),
                    column_styles=section.get("column_styles"),
                    numeric_cell_styles=section.get("numeric_cell_styles"),
                    cell_styles_where=section.get("cell_styles_where"),
                    compact_first_column=section.get("compact_first_column", True),
                )
            )
        if section.get("images"):
            body.append(_render_images(section["images"], output_path.parent))
        if section.get("embeds"):
            body.append(_render_embeds(section["embeds"], output_path.parent))
        if section.get("links"):
            body.append(_render_links(section["links"], output_path.parent))
        body.append("</section>")
    body.extend(["</main>", "</body>", "</html>"])
    output_path.write_text("\n".join(body), encoding="utf-8")
    append_log(f"Generated HTML report: {output_path}", level="INFO")
    if section.get("open_html_report", False):
        import webbrowser

        webbrowser.open(output_path.as_uri())


def _render_key_values(values: dict[str, Any]) -> str:
    """Render a key-value dictionary as an HTML table block."""
    # Compute a stable key-column width to improve readability.
    rows = []
    key_lengths = [len(str(key)) for key in values]
    key_width_ch = max(key_lengths, default=12) + 1
    for key, value in values.items():
        rows.append(f"<tr><th>{escape(str(key))}</th><td>{escape(_stringify(value))}</td></tr>")
    return (
        "<div class='table-scroll'>"
        "<table class='kv-table' style='width: max-content; max-width: none;'>"
        f"<colgroup><col style='width: {key_width_ch}ch;'></colgroup>" + "".join(rows) + "</table></div>"
    )


def _render_table(
    table: pd.DataFrame,
    highlight_rows_where: dict[str, Any] | None = None,
    compact_first_column: bool = False,
    row_styles_where: list[dict[str, Any]] | None = None,
    column_styles: dict[str, str] | None = None,
    numeric_cell_styles: dict[str, dict[str, str]] | None = None,
    cell_styles_where: list[dict[str, Any]] | None = None,
) -> str:
    """Render a dataframe as an HTML table with optional semantic row/cell styling."""
    # Limit table preview size to keep reports responsive.
    if table.empty:
        return "<p class='muted'>No rows to display.</p>"
    preview = table.copy()
    if len(preview) > 50:
        preview = preview.head(50)
    highlight_column = highlight_rows_where.get("column") if highlight_rows_where else None
    highlight_values = set(highlight_rows_where.get("values", [])) if highlight_rows_where else set()
    allowed_row_styles = {"success", "muted"}
    allowed_cell_styles = {"danger", "success", "warning"}
    column_widths = []
    for idx, column in enumerate(preview.columns):
        values = [len(str(column))] + [len(_stringify(value)) for value in preview.iloc[:, idx]]
        width_ch = max(values)
        width_ch += 1 if idx > 0 else (1 if compact_first_column else 2)
        column_widths.append(width_ch)

    table_classes = "data-table compact-first-col" if compact_first_column else "data-table"
    parts = [f"<div class='table-scroll table-scroll-data'><table class='{table_classes}'>", "<colgroup>"]
    for idx, _ in enumerate(preview.columns):
        parts.append(f"<col style='width: {column_widths[idx]}ch;'>")
    parts.append("</colgroup><thead><tr>")
    for column in preview.columns:
        parts.append(f"<th>{escape(str(column))}</th>")
    parts.append("</tr></thead><tbody>")
    for _, row in preview.iterrows():
        is_highlighted = highlight_column in preview.columns and row[highlight_column] in highlight_values
        row_class = "row-highlight" if is_highlighted else ""
        if not row_class:
            for style_rule in row_styles_where or []:
                style_column = style_rule.get("column")
                style_values = set(style_rule.get("values", []))
                style_name = style_rule.get("style")
                if (
                    style_name in allowed_row_styles
                    and style_column in preview.columns
                    and row[style_column] in style_values
                ):
                    row_class = f"row-{style_name}"
                    break
        class_attr = f" class='{row_class}'" if row_class else ""
        parts.append(f"<tr{class_attr}>")
        for column in preview.columns:
            cell_style = (column_styles or {}).get(str(column))
            numeric_styles = (numeric_cell_styles or {}).get(str(column), {})
            if numeric_styles and pd.notna(row[column]):
                numeric_value = float(row[column])
                sign = "positive" if numeric_value > 0 else "negative" if numeric_value < 0 else "zero"
                numeric_style = numeric_styles.get(sign)
                if numeric_style in allowed_cell_styles:
                    cell_style = numeric_style
            for style_rule in cell_styles_where or []:
                style_column = style_rule.get("column")
                style_values = set(style_rule.get("values", []))
                target_columns = set(style_rule.get("target_columns", []))
                style_name = style_rule.get("style")
                if (
                    style_name in allowed_cell_styles
                    and style_column in preview.columns
                    and row[style_column] in style_values
                    and str(column) in target_columns
                ):
                    cell_style = style_name
                    break
            cell_class = f" class='cell-{cell_style}'" if cell_style in allowed_cell_styles else ""
            parts.append(f"<td{cell_class}>{escape(_stringify(row[column]))}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts)


def _render_images(images: list[dict[str, str]], report_dir: Path) -> str:
    """Render image cards for existing image paths in a report section."""
    # Skip missing files so report generation never fails on absent visuals.
    parts = ["<div class='image-grid'>"]
    for image in images:
        image_path = Path(image["path"])
        if not image_path.exists():
            continue
        rel_path = _to_report_relative_path(image_path, report_dir)
        parts.append("<figure class='image-card'>")
        parts.append(f"<img src='{escape(rel_path)}' alt='{escape(image['title'])}'>")
        parts.append(f"<figcaption>{escape(image['title'])}</figcaption>")
        parts.append("</figure>")
    parts.append("</div>")
    return "".join(parts)


def _render_links(links: list[dict[str, str]], report_dir: Path) -> str:
    """Render a list of downloadable or navigable report links."""
    # Convert each link path to a report-relative URL for portability.
    items = ["<ul class='link-list'>"]
    for link in links:
        link_path = Path(link["path"])
        rel_path = _to_report_relative_path(link_path, report_dir)
        items.append(f"<li><a href='{escape(rel_path)}' target='_blank'>{escape(link['label'])}</a></li>")
    items.append("</ul>")
    return "".join(items)


def _render_embeds(embeds: list[dict[str, str]], report_dir: Path) -> str:
    """Render iframe embeds for HTML artifacts referenced by a section."""
    # Ignore missing embeds so sections degrade gracefully.
    parts = ["<div class='embed-grid'>"]
    for embed in embeds:
        embed_path = Path(embed["path"])
        if not embed_path.exists():
            continue
        rel_path = _to_report_relative_path(embed_path, report_dir)
        parts.append("<div class='embed-card'>")
        parts.append(f"<iframe src='{escape(rel_path)}' title='{escape(embed['title'])}' loading='lazy'></iframe>")
        parts.append(f"<div class='embed-caption'>{escape(embed['title'])}</div>")
        parts.append("</div>")
    parts.append("</div>")
    return "".join(parts)


def _to_report_relative_path(path: Path, report_dir: Path) -> str:
    """Convert a path to a report-relative POSIX string when possible."""
    # Prefer relative paths so copied report folders remain self-contained.
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.relative_to(report_dir).as_posix()
    except ValueError:
        return path.as_posix()


def _stringify(value: Any) -> str:
    """Convert values to compact display strings for HTML cells."""
    # Use fixed precision for float values in tabular reports.
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _style_block() -> str:
    """Return the shared CSS style block used by generated reports."""
    # Keep report styling centralized to ensure consistent look and feel.
    return """
<style>
body { background: #f4f1ea; color: #1f2933; font-family: 'Segoe UI', Tahoma, sans-serif; margin: 0; }
.page { max-width: 1440px; margin: 0 auto; padding: 32px 20px 64px; }
h1, h2 { color: #12343b; }
.intro { font-size: 1.05rem; max-width: 900px; }
.card { background: #ffffff; border-radius: 18px; padding: 22px 24px; margin: 20px 0;
box-shadow: 0 8px 24px rgba(18, 52, 59, 0.08); }
.kv-table, .data-table { width: 100%; border-collapse: collapse; margin-top: 12px; }
.kv-table th, .kv-table td, .data-table th, .data-table td { border-bottom: 1px solid #e5e7eb;
padding: 12px 16px; text-align: left; vertical-align: top; }
.kv-table th { color: #486581; white-space: nowrap; width: 1%; }
.kv-table td { white-space: nowrap; }
.table-scroll { width: 100%; overflow-x: auto; overflow-y: visible; margin-top: 12px; padding-bottom: 6px; }
.table-scroll-data { max-height: 75vh; overflow-y: auto; border: 1px solid #e5e7eb; border-radius: 12px; }
.table-scroll table { min-width: max-content; margin-top: 0; }
.data-table { table-layout: fixed; }
.compact-first-col { width: max-content; max-width: none; table-layout: auto; }
.data-table thead th { background: #f8fafc; position: sticky; top: 0; white-space: normal;
min-width: 130px; line-height: 1.35; border-right: 1px solid #e5e7eb; }
.data-table thead th:last-child { border-right: none; }
.data-table tbody td { overflow-wrap: anywhere; }
.data-table tbody tr.row-highlight { background-color: #fff3c4; font-weight: 600; }
.data-table tbody tr.row-success { background-color: #dcfce7; font-weight: 600; }
.data-table tbody tr.row-muted { background-color: #f1f5f9; color: #64748b; }
.data-table tbody td.cell-success { background-color: #f0fdf4; color: #166534; font-weight: 700; }
.data-table tbody td.cell-danger { background-color: #fef2f2; color: #b42318; font-weight: 700; }
.data-table tbody td.cell-warning { background-color: #fff3c4; color: #7c5c00; font-weight: 700; }
.compact-first-col thead th:first-child, .compact-first-col tbody td:first-child { white-space: nowrap; }
.compact-first-col thead th:first-child, .compact-first-col tbody td:first-child { width: 1%; }
.compact-first-col thead th, .compact-first-col tbody td { min-width: 0; }
.compact-first-col thead th:not(:first-child), .compact-first-col tbody td:not(:first-child) { white-space: nowrap; }
.image-grid { display: grid; grid-template-columns: 1fr; gap: 22px; margin-top: 14px; }
.image-card { margin: 0; background: #faf7f2; border-radius: 14px; padding: 14px; }
.image-card img { width: 100%; max-width: 1360px; border-radius: 10px; display: block; margin: 0 auto; }
.image-card figcaption { margin-top: 10px; font-size: 0.95rem; color: #52606d; }
.embed-grid { display: grid; grid-template-columns: 1fr; gap: 22px; margin-top: 14px; }
.embed-card { background: #faf7f2; border-radius: 14px; padding: 14px; }
.embed-card iframe { width: 100%; min-height: 720px; border: 0; border-radius: 10px; background: #fff; }
.embed-caption { margin-top: 10px; font-size: 0.95rem; color: #52606d; }
.link-list { margin: 12px 0 0; padding-left: 18px; }
.link-list a { color: #0f766e; text-decoration: none; }
.link-list a:hover { text-decoration: underline; }
.muted { color: #7b8794; }
</style>
"""
