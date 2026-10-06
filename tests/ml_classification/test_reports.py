from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from satellites.ml_classification.shared.reports.report_helpers import _prepare_cv_results_table
from satellites.ml_classification.shared.reports.report_html import _render_table
from satellites.ml_classification.step_03_train.libraries import report as train_reports_module
from satellites.ml_classification.step_04_evaluate.libraries import report as reports_module
from satellites.ml_classification.step_04_evaluate.libraries.report import (
    _build_average_metrics_section,
    _prepare_analytical_metrics,
)


def test_numeric_cell_styles_preserve_missing_value_placeholders():
    table = pd.DataFrame({"delta": ["n/a", None, "2.5", -1, 0]})
    rendered = _render_table(
        table,
        numeric_cell_styles={"delta": {"positive": "success", "negative": "danger", "zero": "warning"}},
    )
    assert "<td>n/a</td>" in rendered
    assert rendered.count("class='cell-success'") == 1
    assert rendered.count("class='cell-danger'") == 1
    assert rendered.count("class='cell-warning'") == 1


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
    config = SimpleNamespace(
        evaluate_dir=Path("missing-evaluation-output"),
        train_dir=Path("missing-training-output"),
        scoring_primary="f1_macro",
        cv_ranking_method="score_minus_std",
        top_voting_models=3,
    )
    metrics_df = pd.DataFrame([{"model": "base_model", "f1_macro": 0.80}])
    selection = {"selection_type": "single_model", "selected_models": ["base_model"]}

    with patch.object(reports_module, "write_html_report") as write_html_report:
        reports_module.write_evaluate_report(config, metrics_df, metrics_df, selection)

    sections = write_html_report.call_args.kwargs["sections"]
    assert [section["title"] for section in sections] == [
        "Model Selection Strategy",
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


def test_write_evaluate_report_separates_oof_evidence_from_holdout_validation():
    config = SimpleNamespace(
        evaluate_dir=Path("missing-evaluation-output"),
        train_dir=Path("missing-training-output"),
        scoring_primary="f1_macro",
        cv_ranking_method="score_minus_std",
        top_voting_models=3,
    )
    metrics_df = pd.DataFrame([{"model": "base_model", "f1_macro": 0.80}])
    confidence_df = pd.DataFrame(
        [{"confidence_level": "HIGH", "parcels": 10, "accuracy": 0.90}]
    )
    selection = {
        "selection_type": "single_model",
        "selected_models": ["base_model"],
        "confidence": {"method": "rank_consensus_with_class_reliability_guard"},
    }

    with patch.object(reports_module, "write_html_report") as write_html_report:
        reports_module.write_evaluate_report(
            config,
            metrics_df,
            metrics_df,
            selection,
            confidence_metrics_df=confidence_df,
            oof_confidence_metrics_df=confidence_df,
        )

    sections = write_html_report.call_args.kwargs["sections"]
    titles = [section["title"] for section in sections]
    oof_title = "OOF Confidence Performance (Descriptive)"
    holdout_title = "Holdout Validation: Rank Consensus with Class Reliability Guard"
    assert titles.index(oof_title) < titles.index(holdout_title)
    assert "not an independent validation result" in next(
        section["text"] for section in sections if section["title"] == oof_title
    )


def test_write_evaluate_report_styles_voting_members_and_other_models():
    config = SimpleNamespace(
        evaluate_dir=Path("missing-evaluation-output"),
        train_dir=Path("missing-training-output"),
        scoring_primary="f1_macro",
        cv_ranking_method="score_minus_std",
        top_voting_models=2,
    )
    metrics_df = pd.DataFrame(
        [
            {"model": "selected_a", "f1_macro": 0.90},
            {"model": "not_selected", "f1_macro": 0.80},
            {"model": "selected_b", "f1_macro": 0.85},
            {"model": "soft_voting", "f1_macro": 0.92},
        ]
    )
    analytical_df = pd.DataFrame(
        [
            {"model": "selected_a", "f1_macro": 0.90},
            {"model": "not_selected", "f1_macro": 0.80},
            {"model": "selected_b", "f1_macro": 0.85},
            {"model": "Voting", "f1_macro": 0.92},
        ]
    )
    selection = {
        "selection_type": "soft_voting",
        "selected_models": ["selected_a", "selected_b"],
    }

    with patch.object(reports_module, "write_html_report") as write_html_report:
        reports_module.write_evaluate_report(
            config,
            metrics_df,
            metrics_df,
            selection,
            geo_train_metrics_df=analytical_df,
            geo_test_metrics_df=analytical_df,
        )

    sections = write_html_report.call_args.kwargs["sections"]
    average_section = next(section for section in sections if section["title"].endswith("Train Set (Average)"))
    analytical_section = next(section for section in sections if section["title"].endswith("Train Set (Analytical per Class)"))
    expected_styles = [
        {"column": "model", "values": ["selected_a", "selected_b"], "style": "success"},
        {"column": "model", "values": ["not_selected"], "style": "muted"},
    ]
    assert average_section["row_styles_where"] == expected_styles
    assert analytical_section["row_styles_where"] == expected_styles
    assert "Green rows" in average_section["text"]
    assert "Green rows" in analytical_section["text"]


def test_prepare_cv_results_table_sorts_and_marks_selected_models():
    training_summary_df = pd.DataFrame(
        [
            {
                "model": "second",
                "best_cv_score": 0.82,
                "cv_score_std": 0.04,
                "cv_ranking_metric": 0.78,
                "cv_ranking_method": "score_minus_std",
                "supports_predict_proba": True,
            },
            {
                "model": "best",
                "best_cv_score": 0.84,
                "cv_score_std": 0.02,
                "cv_ranking_metric": 0.82,
                "cv_ranking_method": "score_minus_std",
                "supports_predict_proba": True,
            },
            {
                "model": "third",
                "best_cv_score": 0.80,
                "cv_score_std": 0.05,
                "cv_ranking_metric": 0.75,
                "cv_ranking_method": "score_minus_std",
                "supports_predict_proba": True,
            },
        ]
    )

    result = _prepare_cv_results_table(training_summary_df, ["best", "second"])

    assert result["cv_rank"].tolist() == [1, 2, 3]
    assert result["model"].tolist() == ["best", "second", "third"]
    assert result["mean_cv_score"].tolist() == [84.0, 82.0, 80.0]
    assert result["cv_selection_score"].tolist() == [82.0, 78.0, 75.0]
    assert result["selected_for_strategy"].tolist() == ["Yes", "Yes", "No"]


def test_write_train_report_explains_strategy_and_ranks_cv_results():
    config = SimpleNamespace(
        train_dir=Path("training-output"),
        scoring_primary="f1_macro",
        cv_ranking_method="score_minus_std",
        selection_type="soft_voting",
        top_voting_models=2,
        train_model_dir=lambda model_name: Path("training-output") / "models" / model_name,
    )
    training_summary_df = pd.DataFrame(
        [
            {
                "model": "second",
                "best_cv_score": 0.82,
                "cv_score_std": 0.04,
                "cv_ranking_metric": 0.78,
                "cv_ranking_method": "score_minus_std",
                "supports_predict_proba": True,
            },
            {
                "model": "best",
                "best_cv_score": 0.84,
                "cv_score_std": 0.02,
                "cv_ranking_metric": 0.82,
                "cv_ranking_method": "score_minus_std",
                "supports_predict_proba": True,
            },
            {
                "model": "third",
                "best_cv_score": 0.80,
                "cv_score_std": 0.05,
                "cv_ranking_metric": 0.75,
                "cv_ranking_method": "score_minus_std",
                "supports_predict_proba": True,
            },
        ]
    )
    model_specs = {model_name: {"best_params": {}} for model_name in ("best", "second", "third")}

    with patch.object(train_reports_module, "write_html_report") as write_html_report:
        train_reports_module.write_train_report(config, training_summary_df, model_specs, pd.DataFrame())

    sections = write_html_report.call_args.kwargs["sections"]
    assert sections[0]["title"] == "Model Selection Strategy"
    assert sections[0]["kv"]["best_cv_model"] == "best"
    assert sections[0]["kv"]["configured_strategy_models"] == ["best", "second"]
    assert sections[1]["title"] == "Cross-Validation Model Ranking"
    assert sections[1]["table"]["model"].tolist() == ["best", "second", "third"]
    assert sections[1]["row_styles_where"][0]["values"] == ["best", "second"]


def test_write_evaluate_report_includes_frozen_cv_selection_table():
    config = SimpleNamespace(
        evaluate_dir=Path("evaluation-output"),
        train_dir=Path("training-output"),
        scoring_primary="f1_macro",
        cv_ranking_method="score_minus_std",
        top_voting_models=2,
    )
    metrics_df = pd.DataFrame([{"model": "best", "f1_macro": 0.80}])
    cv_summary_df = pd.DataFrame(
        [
            {
                "model": "second",
                "best_cv_score": 0.82,
                "cv_score_std": 0.04,
                "cv_ranking_metric": 0.78,
                "cv_ranking_method": "score_minus_std",
                "supports_predict_proba": True,
            },
            {
                "model": "best",
                "best_cv_score": 0.84,
                "cv_score_std": 0.02,
                "cv_ranking_metric": 0.82,
                "cv_ranking_method": "score_minus_std",
                "supports_predict_proba": True,
            },
            {
                "model": "third",
                "best_cv_score": 0.80,
                "cv_score_std": 0.05,
                "cv_ranking_metric": 0.75,
                "cv_ranking_method": "score_minus_std",
                "supports_predict_proba": True,
            },
        ]
    )
    selection = {
        "selection_type": "soft_voting",
        "selection_source": "cross_validation",
        "selected_models": ["best", "second"],
        "selected_score": 0.80,
    }

    def read_optional_csv(path):
        return cv_summary_df if Path(path).name == "training_summary.csv" else None

    with (
        patch.object(reports_module, "_read_optional_csv", side_effect=read_optional_csv),
        patch.object(reports_module, "write_html_report") as write_html_report,
    ):
        reports_module.write_evaluate_report(config, metrics_df, metrics_df, selection)

    sections = write_html_report.call_args.kwargs["sections"]
    strategy_section = sections[0]
    cv_section = sections[1]
    assert strategy_section["title"] == "Model Selection Strategy"
    assert strategy_section["kv"]["cv_ranking_formula"] == "mean CV f1_macro - CV standard deviation"
    assert cv_section["title"] == "Voting Model Selection from Cross-Validation"
    assert cv_section["table"]["model"].tolist() == ["best", "second", "third"]
    assert cv_section["table"]["selected_for_strategy"].tolist() == ["Yes", "Yes", "No"]
    assert cv_section["row_styles_where"][0]["values"] == ["best", "second"]


def test_render_table_applies_semantic_row_styles_with_highlight_precedence():
    table = pd.DataFrame({"model": ["selected", "other", "Voting"], "score": [0.9, 0.8, 0.95]})

    rendered = _render_table(
        table,
        highlight_rows_where={"column": "model", "values": ["Voting"]},
        row_styles_where=[
            {"column": "model", "values": ["selected", "Voting"], "style": "success"},
            {"column": "model", "values": ["other"], "style": "muted"},
        ],
    )

    assert rendered.count("class='row-success'") == 1
    assert rendered.count("class='row-muted'") == 1
    assert rendered.count("class='row-highlight'") == 1
