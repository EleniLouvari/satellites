"""Tests for the final dashboard's compact analytical data model."""

from __future__ import annotations

from pathlib import Path
import shutil
from types import SimpleNamespace
from uuid import uuid4

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from ml_classification.ml_classification_pipeline.reporting.final_dashboard import (
    _build_cv_rows,
    _build_map_html,
    _build_performance_rows,
    _build_phenology_rows,
    _build_seasonal_summary,
    _save_monthly_index_boxplots,
    write_pipeline_final_dashboard,
)
from ml_classification.ml_classification_pipeline.reporting.reports import write_index_report


def test_build_phenology_rows_uses_monthly_medians_per_modeled_class():
    parcels = pd.DataFrame(
        {
            "initial_label_code": ["110", "110", "119"],
            "NDVI_median__20231001": [0.2, 0.4, 0.1],
            "NDVI_median__20231101": [0.5, 0.7, 0.2],
            "NDWI_median__20231001": [-0.2, 0.0, -0.1],
            "NDWI_median__20231101": [0.1, 0.3, 0.0],
        }
    )

    result = _build_phenology_rows(parcels, {"110": "Wheat", "119": "Grass Fodders"})

    wheat_oct_ndvi = result.query("class_code == '110' and index == 'NDVI' and date == '2023-10-01'").iloc[0]
    assert wheat_oct_ndvi["median"] == pytest.approx(0.3)
    assert wheat_oct_ndvi["q25"] == pytest.approx(0.25)
    assert wheat_oct_ndvi["q75"] == pytest.approx(0.35)
    assert wheat_oct_ndvi["lowest"] == pytest.approx(0.2)
    assert wheat_oct_ndvi["highest"] == pytest.approx(0.4)
    assert wheat_oct_ndvi["parcels"] == 2
    assert len(result) == 8

    seasonal = _build_seasonal_summary(result)
    wheat = seasonal.loc[seasonal["class_code"] == "110"].iloc[0]
    assert wheat["ndvi_peak_month"] == "Nov 2023"
    assert wheat["ndvi_peak"] == pytest.approx(0.6)
    assert wheat["ndwi_peak"] == pytest.approx(0.2)


def test_build_cv_rows_ranks_models_and_numbers_folds_for_readers():
    training = pd.DataFrame(
        [
            {"model": "second", "best_cv_score": 0.82, "cv_score_std": 0.04, "cv_ranking_metric": 0.78},
            {"model": "best", "best_cv_score": 0.84, "cv_score_std": 0.02, "cv_ranking_metric": 0.82},
        ]
    )
    folds = pd.DataFrame(
        [
            {"model": "second", "fold": 0, "score": 0.80},
            {"model": "best", "fold": 0, "score": 0.83},
            {"model": "best", "fold": 1, "score": 0.85},
        ]
    )

    fold_rows, ranking = _build_cv_rows(training, folds, {"best"})

    assert ranking["model"].tolist() == ["best", "second"]
    assert ranking["selected"].tolist() == ["Yes", "No"]
    assert fold_rows.loc[fold_rows["model"] == "best", "fold"].tolist() == [1, 2]


def test_build_performance_rows_keeps_exact_train_test_gaps():
    train = pd.DataFrame(
        [
            {"model": "Voting", "f1_macro": 0.80, "balanced_accuracy": 0.79, "accuracy": 0.88},
            {"model": "base", "f1_macro": 0.75, "balanced_accuracy": 0.74, "accuracy": 0.84},
        ]
    )
    test = pd.DataFrame(
        [
            {"model": "Voting", "f1_macro": 0.72, "balanced_accuracy": 0.70, "accuracy": 0.81},
            {"model": "base", "f1_macro": 0.68, "balanced_accuracy": 0.69, "accuracy": 0.78},
        ]
    )

    long, wide = _build_performance_rows(train, test)

    assert set(long["split"]) == {"Train", "Test"}
    voting = wide.loc[wide["model"] == "Voting"].iloc[0]
    assert round(voting["f1_macro_gap"], 8) == 0.08
    assert round(voting["accuracy_gap"], 8) == 0.07
    assert wide.columns.tolist() == [
        "model",
        "train_f1_macro",
        "test_f1_macro",
        "f1_macro_gap",
        "train_balanced_accuracy",
        "test_balanced_accuracy",
        "balanced_accuracy_gap",
        "train_accuracy",
        "test_accuracy",
        "accuracy_gap",
    ]


def test_build_map_html_is_offline_and_links_to_openstreetmap():
    parcels = gpd.GeoDataFrame(
        {"initial_label_code": ["110", "119"]},
        geometry=[Point(22.9, 40.6), Point(23.1, 40.8)],
        crs="EPSG:4326",
    )

    rendered = _build_map_html(parcels, "source.parquet")

    assert "<svg" in rendered
    assert "openstreetmap.org" in rendered
    assert "<iframe" not in rendered
    assert "source.parquet" in rendered


def test_monthly_index_boxplots_use_month_rows_and_ndvi_ndwi_columns(monkeypatch, request):
    distribution = pd.DataFrame(
        {
            "class_code": ["110", "119"] * 4,
            "class_label": ["Wheat", "Grass Fodders"] * 4,
            "index": ["NDVI", "NDVI", "NDWI", "NDWI"] * 2,
            "date": pd.to_datetime(["2023-10-01"] * 4 + ["2023-11-01"] * 4),
            "parcel_value": [0.2, 0.1, -0.2, -0.1, 0.5, 0.3, 0.1, 0.0],
        }
    )
    calls = []
    real_boxplot = __import__("seaborn").boxplot

    def record_boxplot(*args, **kwargs):
        calls.append(kwargs.copy())
        return real_boxplot(*args, **kwargs)

    monkeypatch.setattr(
        "ml_classification.ml_classification_pipeline.reporting.final_dashboard.sns.boxplot",
        record_boxplot,
    )

    output_dir = Path(__file__).parent / f"_monthly_boxplot_{uuid4().hex}"
    request.addfinalizer(lambda: shutil.rmtree(output_dir, ignore_errors=True))
    output = output_dir / "monthly_ndvi_ndwi.png"
    _save_monthly_index_boxplots(distribution, output)

    assert output.exists()
    assert len(calls) == 4
    assert all(call["x"] == "class_label" and call["y"] == "parcel_value" for call in calls)
    assert all(call["order"] == ["Wheat", "Grass Fodders"] for call in calls)
    assert all(call["data"]["date"].nunique() == 1 for call in calls)
    assert all(call["data"]["index"].nunique() == 1 for call in calls)
    assert [call["data"]["index"].iloc[0] for call in calls] == ["NDVI", "NDWI", "NDVI", "NDWI"]


def test_write_pipeline_final_dashboard_exports_report_and_seasonal_peak_graph(monkeypatch, request):
    test_root = Path(__file__).parent / f"_final_dashboard_{uuid4().hex}"
    request.addfinalizer(lambda: shutil.rmtree(test_root, ignore_errors=True))
    project_dir = test_root / "project"
    check_dir = project_dir / "01_check"
    prepare_dir = project_dir / "02_prepare"
    train_dir = project_dir / "03_train"
    evaluate_dir = project_dir / "04_evaluate"
    predict_dir = project_dir / "05_predict"
    final_dashboard_dir = project_dir / "final_dashboard"
    for directory in (
        check_dir / "plots",
        prepare_dir,
        train_dir / "plots",
        evaluate_dir,
        predict_dir,
        final_dashboard_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    (check_dir / "plots" / "known_labels_map_osm.html").write_text("<html>OSM map</html>", encoding="utf-8")
    (check_dir / "plots" / "known_labels_map.png").write_bytes(b"static map must not be embedded")

    (prepare_dir / "prepare_summary.json").write_text(
        '{"train_rows": 3, "test_rows": 1, "target_labels": ["110", "119"]}',
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {"model": "base", "best_cv_score": 0.72, "cv_score_std": 0.03, "cv_ranking_metric": 0.69},
            {"model": "other", "best_cv_score": 0.65, "cv_score_std": 0.04, "cv_ranking_metric": 0.61},
            {"model": "unused", "best_cv_score": 0.60, "cv_score_std": 0.05, "cv_ranking_metric": 0.55},
        ]
    ).to_csv(train_dir / "training_summary.csv", index=False)
    pd.DataFrame(
        [
            {"model": "base", "fold": 0, "score": 0.70},
            {"model": "base", "fold": 1, "score": 0.74},
            {"model": "other", "fold": 0, "score": 0.62},
            {"model": "other", "fold": 1, "score": 0.68},
            {"model": "unused", "fold": 0, "score": 0.58},
            {"model": "unused", "fold": 1, "score": 0.62},
        ]
    ).to_csv(train_dir / "best_cv_fold_scores.csv", index=False)

    parcels = gpd.GeoDataFrame(
        {
            "parcel_code": ["1", "2", "3", "4"],
            "label": ["110", "110", "119", "119"],
            "NDVI_median__20231001": [0.2, 0.3, 0.1, 0.2],
            "NDVI_median__20231101": [0.5, 0.6, 0.3, 0.4],
            "NDWI_median__20231001": [-0.2, -0.1, -0.3, -0.2],
            "NDWI_median__20231101": [0.1, 0.2, -0.1, 0.0],
        },
        geometry=[Point(22.9, 40.6), Point(23.0, 40.6), Point(23.1, 40.7), Point(23.2, 40.7)],
        crs="EPSG:4326",
    )
    monkeypatch.setattr(
        "ml_classification.ml_classification_pipeline.reporting.final_dashboard.load_joblib",
        lambda _path: parcels,
    )
    config = SimpleNamespace(
        project_dir=project_dir,
        check_dir=check_dir,
        prepare_dir=prepare_dir,
        train_dir=train_dir,
        evaluate_dir=evaluate_dir,
        predict_dir=predict_dir,
        final_dashboard_dir=final_dashboard_dir,
        target_column="label",
        prediction_column="label_prediction",
        scoring_primary="f1_macro",
    )
    train_metrics = pd.DataFrame(
        [
            {"model": "base", "f1_macro": 0.80, "accuracy": 0.85},
            {"model": "other", "f1_macro": 0.70, "accuracy": 0.78},
            {"model": "unused", "f1_macro": 0.69, "accuracy": 0.76},
            {"model": "Voting", "f1_macro": 0.82, "accuracy": 0.87},
        ]
    )
    test_metrics = pd.DataFrame(
        [
            {"model": "base", "f1_macro": 0.72, "accuracy": 0.79},
            {"model": "other", "f1_macro": 0.72, "accuracy": 0.70},
            {"model": "unused", "f1_macro": 0.60, "accuracy": 0.68},
            {"model": "Voting", "f1_macro": 0.74, "accuracy": 0.81},
        ]
    )
    selection = {
        "selection_type": "soft_voting",
        "selected_models": ["base", "other"],
        "selected_metric": "cv_score_minus_std",
        "selected_score": 0.69,
    }

    predictions = pd.DataFrame(
        {
            "label": ["110", "110", "119", "119"],
            "label_prediction": ["110", "119", "119", "110"],
            "prediction_borda_winner": ["110", "119", "119", "110"],
            "prediction_class_oof_precision": [0.90, 0.68, 0.84, 0.71],
            "prediction_class_reliability_level": ["HIGH", "MEDIUM", "HIGH", "MEDIUM"],
            "prediction_rank_confidence_level": ["HIGH", "HIGH", "HIGH", "MEDIUM"],
            "prediction_confidence_level": ["HIGH", "MEDIUM", "HIGH", "MEDIUM"],
        }
    )

    output = write_pipeline_final_dashboard(
        config,
        train_metrics,
        test_metrics,
        selection,
        prediction_df=predictions,
    )

    assert output == final_dashboard_dir / "report.html"
    assert output.exists()
    assert (final_dashboard_dir / "plots" / "seasonal_ndvi_ndwi_boxplots_by_month_class.png").exists()
    assert not (final_dashboard_dir / "plots" / "seasonal_ndvi_boxplots_by_class_month.png").exists()
    assert not (final_dashboard_dir / "plots" / "seasonal_ndwi_boxplots_by_class_month.png").exists()
    assert (final_dashboard_dir / "plots" / "seasonal_ndvi_high_low_by_class.png").exists()
    assert (final_dashboard_dir / "plots" / "seasonal_ndwi_high_low_by_class.png").exists()
    assert (final_dashboard_dir / "plots" / "selected_model_cv_folds.png").exists()
    assert (final_dashboard_dir / "plots" / "borda_consensus_correctness_overview.png").exists()
    assert (final_dashboard_dir / "plots" / "need_to_check_rate_by_predicted_class.png").exists()
    assert (final_dashboard_dir / "plots" / "need_to_check_confidence_components.png").exists()
    assert (final_dashboard_dir / "data" / "monthly_index_high_low_by_class.csv").exists()
    assert (final_dashboard_dir / "data" / "need_to_check_by_predicted_class.csv").exists()
    cv_models = set(pd.read_csv(final_dashboard_dir / "data" / "cv_model_ranking.csv")["model"])
    performance_models = set(pd.read_csv(final_dashboard_dir / "data" / "train_test_model_metrics.csv")["model"])
    assert cv_models == {"base", "other"}
    assert performance_models == {"base", "other", "Voting"}
    html = output.read_text(encoding="utf-8")
    assert "one row per month and two columns" in html
    assert "Monthly NDVI and NDWI Distributions by Modeled Class" in html
    assert "Monthly Highest and Lowest NDVI by Modeled Class" in html
    assert "Monthly Highest and Lowest NDWI by Modeled Class" in html
    assert "Train versus Spatial Holdout" in html
    assert "Prediction Confidence Diagnostics" in html
    assert "Unknown fill rows are excluded" in html
    assert "<td class='cell-success'>0.7200</td>" in html
    assert "<td class='cell-warning'>0.7400</td>" in html
    assert "<td class='cell-success'>0.7400</td>" not in html
    assert "<td class='cell-success'>0.8000</td>" not in html
    assert "<td class='cell-danger'>0.0800</td>" in html
    assert "<td class='cell-success'>-0.0200</td>" in html
    assert "known_labels_map_osm.html" in html
    assert "known_labels_map.png" not in html

    write_index_report(config)
    report_index = (project_dir / "report_index.html").read_text(encoding="utf-8")
    assert "Final Dashboard" in report_index
    assert "final_dashboard/report.html" in report_index
