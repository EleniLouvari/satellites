"""Assemble the check step HTML report from its persisted and computed results."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ml_classification.shared.reports.report_html import write_html_report


def write_check_report(config, summary: dict[str, Any], feature_profile: pd.DataFrame) -> None:
    """Write the step-1 check report with summaries and validation visuals.

    Build small preview sections showing validation summaries, a feature
    profile table, and any generated diagnostic plots or interactive map
    embeds that exist in the check output folder.
    """
    # Assemble available plots and optional map embeds for the check report.
    images = [
        {"title": "Missing Values", "path": config.check_dir / "plots" / "missing_values.png"},
        {"title": "Known Target Distribution", "path": config.check_dir / "plots" / "target_distribution.png"},
    ]
    embeds = []
    label_map_path = config.check_dir / "plots" / "known_labels_map.png"
    if label_map_path.exists():
        images.append({"title": "Known Labels Map", "path": label_map_path})
    osm_map_path = config.check_dir / "plots" / "known_labels_map_osm.html"
    if osm_map_path.exists():
        embeds.append({"title": "Known Labels Map with OSM Basemap", "path": osm_map_path})

    write_html_report(
        config.check_dir / "report.html",
        "Step 1 Report: Data Check",
        "Input validation, feature screening, and basic class diagnostics.",
        sections=[
            {"title": "Summary", "kv": summary},
            {
                "title": "Feature Profile",
                "text": "Active features are the ones that move forward into model preparation.",
                "table": feature_profile,
            },
            {"title": "Plots", "images": images, "embeds": embeds},
        ],
    )
