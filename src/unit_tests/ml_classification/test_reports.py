from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from ml_classification.ml_classification_pipeline.reporting import reports as reports_module
from ml_classification.ml_classification_pipeline.reporting.reports import (
    _build_average_metrics_section,
    _prepare_analytical_metrics,
)


def test_build_average_metrics_section_orders_and_labels_special_models():
    metrics_df = pd.DataFrame(
        [
            {"model": "lower_score", "f1_macro": 0.70},
            {"model": "soft_voting", "f1_macro": 0.95},
            {"model": "higher_score", "f1_macro": 0.90},
        ]
    )
    ranking_df = pd.DataFrame(
        [
            {"method": "rank_median", "f1_macro": 0.82, "n_models_used": 3},
            {"method": "probability_average", "f1_macro": 0.84, "n_models_used": 3},
            {"method": "rank_average", "f1_macro": 0.83, "n_models_used": 3},
        ]
    )
    highlighted_models = ["Voting"]

    section = _build_average_metrics_section(
        "Average",
        metrics_df,
        ranking_df,
        "f1_macro",
        highlighted_models,
        "Ranking explanation.",
    )

    assert section["table"]["model"].tolist() == [
        "higher_score",
        "lower_score",
        "Voting",
        "Rank Average",
        "Rank Median",
    ]
    assert section["highlight_rows_where"]["values"] == ["Voting", "Rank Average", "Rank Median"]
    assert section["text"] == "Ranking explanation."
    assert highlighted_models == ["Voting"]
    assert section["table"]["f1_macro"].tolist() == [90.0, 70.0, 95.0, 83.0, 82.0]
    assert metrics_df["f1_macro"].tolist() == [0.70, 0.95, 0.90]


def test_prepare_analytical_metrics_matches_average_order_and_keeps_voting_last():
    average_df = pd.DataFrame(
        [
            {"model": "second", "f1_macro": 0.90},
            {"model": "first", "f1_macro": 0.80},
            {"model": "soft_voting", "f1_macro": 0.95},
        ]
    )
    analytical_df = pd.DataFrame(
        [
            {"model": "first", "f1_macro": 0.80},
            {"model": "Voting", "f1_macro": 0.95},
            {"model": "second", "f1_macro": 0.90},
        ]
    )

    prepared_df = _prepare_analytical_metrics(analytical_df, average_df, "f1_macro")

    assert prepared_df["model"].tolist() == ["second", "first", "Voting"]


def test_write_evaluate_report_assembles_expected_sections_without_artifacts():
    config = SimpleNamespace(evaluate_dir=Path("missing-evaluation-output"), scoring_primary="f1_macro")
    metrics_df = pd.DataFrame([{"model": "base_model", "f1_macro": 0.80}])
    selection = {"selection_type": "single_model", "selected_models": ["base_model"]}

    with patch.object(reports_module, "write_html_report") as write_html_report:
        reports_module.write_evaluate_report(config, metrics_df, metrics_df, selection)

    sections = write_html_report.call_args.kwargs["sections"]
    assert [section["title"] for section in sections] == [
        "Selected Strategy",
        "Metric Definitions",
        "Model Results on Train Set (Analytical per Class)",
        "Model Results on Test Set (Analytical per Class)",
        "Model Results on Train Set (Average)",
        "Model Results on Test Set (Average)",
        "Classification Reports",
        "Evaluation Visuals",
    ]
    train_average_section = next(
        section for section in sections if section["title"] == "Model Results on Train Set (Average)"
    )
    test_average_section = next(
        section for section in sections if section["title"] == "Model Results on Test Set (Average)"
    )
    assert train_average_section["table"].loc[0, "f1_macro"] == 80.0
    assert test_average_section["table"].loc[0, "f1_macro"] == 80.0
