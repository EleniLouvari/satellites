"""Build the project index linking the five step reports and dashboard."""

from __future__ import annotations

from ml_classification.shared.reports.report_html import write_html_report


def write_index_report(config) -> None:
    """Write a top-level index page linking all step reports."""
    # Provide a single navigation entrypoint for generated pipeline reports.
    links = [
        {"label": "01 Check", "path": config.check_dir / "report.html"},
        {"label": "02 Prepare", "path": config.prepare_dir / "report.html"},
        {"label": "03 Train", "path": config.train_dir / "report.html"},
        {"label": "04 Evaluate", "path": config.evaluate_dir / "report.html"},
        {"label": "05 Predict", "path": config.predict_dir / "report.html"},
    ]
    final_dashboard_path = config.final_dashboard_dir / "report.html"
    if final_dashboard_path.exists():
        links.insert(0, {"label": "Final Dashboard", "path": final_dashboard_path})
    write_html_report(
        config.project_dir / "report_index.html",
        "ML Classification Pipeline Report Index",
        "Open any step report below to inspect saved outputs, tables, and plots.",
        sections=[{"title": "Step Reports", "links": links, "open_html_report": getattr(config, "open_html_report", False)}],
    )
