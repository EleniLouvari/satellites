"""Assemble the prepare step HTML report from its persisted and computed results."""

from __future__ import annotations

from typing import Any

import pandas as pd

from satellites.ml_classification.shared.reports.report_html import write_html_report


def write_prepare_report(config, prepare_summary: dict[str, Any], train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Write the step-2 preparation report with split and fold diagnostics."""
    # Build compact summary tables for split statistics and fold sizes.
    summary_kv = {key: value for key, value in prepare_summary.items() if key != "cv_folds"}
    folds_df = pd.DataFrame(
        [
            {
                "fold": fold["fold"],
                "train_rows": len(fold["train_index"]),
                "valid_rows": len(fold["valid_index"]),
                "valid_fraction": len(fold["valid_index"]) / len(train_df),
                "target_valid_fraction": 1 / config.cv_folds,
            }
            for fold in prepare_summary["cv_folds"]
        ]
    )
    split_preview = pd.DataFrame(
        {
            "split": ["train", "test"],
            "rows": [len(train_df), len(test_df)],
            "unique_labels": [train_df[config.target_column].nunique(), test_df[config.target_column].nunique()],
        }
    )
    write_html_report(
        config.prepare_dir / "report.html",
        "Step 2 Report: Data Preparation",
        "Train-test split, fold generation, and preparation outputs for the modeling stage.",
        sections=[
            {"title": "Summary", "kv": summary_kv | {"train_preview_rows": min(5, len(train_df))}},
            {"title": "Split Overview", "table": split_preview},
            {
                "title": "Cross-Validation Folds",
                "text": (
                    "Each validation fold targets 1 / cv_folds of the training rows. "
                    "This matches test_size when test_size = 1 / cv_folds. "
                    "Spatial by_group CV keeps whole cells together, so large cells can prevent balanced fold sizes. "
                    "Spatial by_row CV balances fold sizes and prioritizes class presence and geographic coverage; "
                    "cells may occur in both training and validation. Raw row indices are saved in prepare_summary.json."
                ),
                "table": folds_df,
            },
            {
                "title": "Plots",
                "images": [
                    {"title": "Train-Test Class Balance", "path": config.prepare_dir / "plots" / "train_test_distribution.png"},
                    {
                        "title": "Numeric Correlation Heatmap",
                        "path": config.prepare_dir / "plots" / "numeric_correlation_heatmap.png",
                    },
                    {
                        "title": "Feature Distributions Train vs Test",
                        "path": config.prepare_dir / "plots" / "feature_distributions_train_vs_test.png",
                    },
                    {"title": "Spatial Train Test Split", "path": config.prepare_dir / "plots" / "spatial_train_test_split.png"},
                ],
            },
        ],
    )
