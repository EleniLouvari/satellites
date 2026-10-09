"""Tests for the final dashboard's compact analytical data model."""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from ml_classification.shared.reports.final_dashboard import (
    _build_class_counts_rows,
    _build_cv_rows,
    _build_dashboard_blocks,
    _build_dashboard_charts,
    _build_map_html,
    _build_overview_rows,
    _build_performance_rows,
    _build_phenology_rows,
    _build_probability_rows,
    _build_seasonal_summary,
    _dashboard_source_paths,
    _save_monthly_index_boxplots,
    _sql_quoted_values,
    write_pipeline_final_dashboard,
)
from ml_classification.shared.reports.report_index import write_index_report
from tests.utils import expect_equal, expect_false, expect_in, expect_not_in, expect_true


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
    expect_equal(wheat_oct_ndvi["median"], pytest.approx(0.3))
    expect_equal(wheat_oct_ndvi["q25"], pytest.approx(0.25))
    expect_equal(wheat_oct_ndvi["q75"], pytest.approx(0.35))
    expect_equal(wheat_oct_ndvi["lowest"], pytest.approx(0.2))
    expect_equal(wheat_oct_ndvi["highest"], pytest.approx(0.4))
    expect_equal(wheat_oct_ndvi["parcels"], 2)
    expect_equal(len(result), 8)

    seasonal = _build_seasonal_summary(result)
    wheat = seasonal.loc[seasonal["class_code"] == "110"].iloc[0]
    expect_equal(wheat["ndvi_peak_month"], "Nov 2023")
    expect_equal(wheat["ndvi_peak"], pytest.approx(0.6))
    expect_equal(wheat["ndwi_peak"], pytest.approx(0.2))


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

    expect_equal(ranking["model"].tolist(), ["best", "second"])
    expect_equal(ranking["selected"].tolist(), ["Yes", "No"])
    expect_equal(fold_rows.loc[fold_rows["model"] == "best", "fold"].tolist(), [1, 2])


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

    expect_equal(set(long["split"]), {"Train", "Test"})
    voting = wide.loc[wide["model"] == "Voting"].iloc[0]
    expect_equal(round(voting["f1_macro_gap"], 8), 0.08)
    expect_equal(round(voting["accuracy_gap"], 8), 0.07)
    expect_equal(wide.columns.tolist(), [
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
    ])


def test_build_probability_rows_includes_only_complete_metric_pairs():
    selection = {
        "probability_optimization": {
            "baseline_oof_macro_f1": 0.70,
            "optimized_oof_macro_f1": 0.74,
            "baseline_oof_accuracy": 0.80,
        }
    }

    rows = _build_probability_rows(selection)

    expect_equal(len(rows), 2)
    expect_equal({row["metric"] for row in rows}, {"Macro F1"})
    expect_equal({row["state"] for row in rows}, {"Baseline", "Optimized"})


def test_sql_quoted_values_escapes_single_quotes():
    """SQL IN-list values should be deterministic and SQL-escaped."""
    payload = _sql_quoted_values({"alpha", "m'2"})
    expect_equal(payload, "'alpha', 'm''2'")


def test_build_dashboard_charts_and_blocks_toggle_probability_sections():
    charts_without_probability = _build_dashboard_charts(primary_metric="f1_macro", probability_rows=[])
    blocks_without_probability = _build_dashboard_blocks(
        cards=[{"id": "modeled_parcels"}, {"id": "best_cv"}],
        probability_rows=[],
        modelled_labels={"A", "B"},
    )
    expect_not_in("probability_optimization", [chart["id"] for chart in charts_without_probability])
    expect_not_in("probability", [block["id"] for block in blocks_without_probability])

    probability_rows = [{"metric": "Macro F1", "state": "Baseline", "score": 0.7}]
    charts_with_probability = _build_dashboard_charts(primary_metric="f1_macro", probability_rows=probability_rows)
    blocks_with_probability = _build_dashboard_blocks(
        cards=[{"id": "modeled_parcels"}, {"id": "best_cv"}],
        probability_rows=probability_rows,
        modelled_labels={"A", "B"},
    )
    expect_in("probability_optimization", [chart["id"] for chart in charts_with_probability])
    expect_in("probability", [block["id"] for block in blocks_with_probability])


def test_dashboard_source_paths_use_logical_root_and_project_name():
    """Dashboard source paths should be generated relative to the logical run root."""
    paths = SimpleNamespace(
        parcels=Path("C:/work_dir/satellites/outputs/projectA/1_parcel_stats/ml_ready.parquet"),
        project=Path("C:/work_dir/satellites/outputs/projectA/ml_run"),
    )

    source_paths = _dashboard_source_paths(paths)
    expect_equal(source_paths["satellite_path"], "projectA/1_parcel_stats/ml_ready.parquet")
    expect_equal(source_paths["selection_path"], "projectA/ml_run/04_evaluate/selection_summary.json")


def test_build_overview_rows_and_class_counts_handle_strategy_and_missing_train():
    prepare = {"train_rows": 8, "test_rows": 2}
    selection = {"selection_type": "soft_voting", "selected_models": ["m1", "m2"]}
    modelled_labels = {"110", "119"}
    cv_ranking = pd.DataFrame([{"model": "m1", "mean_cv_score": 0.85}])
    train_metrics = pd.DataFrame([{"model": "m2", "f1_macro": 0.88}])
    test_metrics = pd.DataFrame([{"model": "m1", "f1_macro": 0.80}])

    overview = _build_overview_rows(
        prepare=prepare,
        selection=selection,
        modelled_labels=modelled_labels,
        cv_ranking=cv_ranking,
        strategy_model="m1",
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        primary_metric="f1_macro",
    )[0]
    expect_equal(overview["modeled_parcels"], 10)
    expect_equal(overview["modeled_classes"], 2)
    expect_equal(overview["strategy"], "Soft voting")
    expect_equal(overview["strategy_members"], 2)
    expect_equal(overview["generalization_gap"], None)

    parcels = pd.DataFrame({"initial_label_code": ["110", "119", "110"]})
    counts = _build_class_counts_rows(parcels, {"110": "Wheat", "119": "Grass"})
    expect_equal(counts["class_code"].tolist(), ["110", "119"])
    expect_equal(counts["parcels"].tolist(), [2, 1])


def test_build_map_html_is_offline_and_links_to_openstreetmap():
    parcels = gpd.GeoDataFrame(
        {"initial_label_code": ["110", "119"]},
        geometry=[Point(22.9, 40.6), Point(23.1, 40.8)],
        crs="EPSG:4326",
    )

    rendered = _build_map_html(parcels, "source.parquet")

    expect_in("<svg", rendered)
    expect_in("openstreetmap.org", rendered)
    expect_not_in("<iframe", rendered)
    expect_in("source.parquet", rendered)


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
        "ml_classification.shared.reports.final_dashboard.sns.boxplot",
        record_boxplot,
    )

    output_dir = Path(__file__).parent / f"_monthly_boxplot_{uuid4().hex}"
    request.addfinalizer(lambda: shutil.rmtree(output_dir, ignore_errors=True))
    output = output_dir / "monthly_ndvi_ndwi.png"
    _save_monthly_index_boxplots(distribution, output)

    expect_true(output.exists())
    expect_equal(len(calls), 4)
    expect_true(all(call["x"] == "class_label" and call["y"] == "parcel_value" for call in calls))
    expect_true(all(call["order"] == ["Wheat", "Grass Fodders"] for call in calls))
    expect_true(all(call["data"]["date"].nunique() == 1 for call in calls))
    expect_true(all(call["data"]["index"].nunique() == 1 for call in calls))
    expect_equal([call["data"]["index"].iloc[0] for call in calls], ["NDVI", "NDWI", "NDVI", "NDWI"])


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
        "ml_classification.shared.reports.final_dashboard.load_joblib",
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

    expect_equal(output, final_dashboard_dir / "report.html")
    expect_true(output.exists())
    expect_true((final_dashboard_dir / "plots" / "seasonal_ndvi_ndwi_boxplots_by_month_class.png").exists())
    expect_false((final_dashboard_dir / "plots" / "seasonal_ndvi_boxplots_by_class_month.png").exists())
    expect_false((final_dashboard_dir / "plots" / "seasonal_ndwi_boxplots_by_class_month.png").exists())
    expect_true((final_dashboard_dir / "plots" / "seasonal_ndvi_high_low_by_class.png").exists())
    expect_true((final_dashboard_dir / "plots" / "seasonal_ndwi_high_low_by_class.png").exists())
    expect_true((final_dashboard_dir / "plots" / "selected_model_cv_folds.png").exists())
    expect_true((final_dashboard_dir / "plots" / "borda_consensus_correctness_overview.png").exists())
    expect_true((final_dashboard_dir / "plots" / "need_to_check_rate_by_predicted_class.png").exists())
    expect_true((final_dashboard_dir / "plots" / "need_to_check_confidence_components.png").exists())
    expect_true((final_dashboard_dir / "data" / "monthly_index_high_low_by_class.csv").exists())
    expect_true((final_dashboard_dir / "data" / "need_to_check_by_predicted_class.csv").exists())
    cv_models = set(pd.read_csv(final_dashboard_dir / "data" / "cv_model_ranking.csv")["model"])
    performance_models = set(pd.read_csv(final_dashboard_dir / "data" / "train_test_model_metrics.csv")["model"])
    expect_equal(cv_models, {"base", "other"})
    expect_equal(performance_models, {"base", "other", "Voting"})
    html = output.read_text(encoding="utf-8")
    expect_in("one row per month and two columns", html)
    expect_in("Monthly NDVI and NDWI Distributions by Modeled Class", html)
    expect_in("Monthly Highest and Lowest NDVI by Modeled Class", html)
    expect_in("Monthly Highest and Lowest NDWI by Modeled Class", html)
    expect_in("Train versus Spatial Holdout", html)
    expect_in("Prediction Confidence Diagnostics", html)
    expect_in("Unknown fill rows are excluded", html)
    expect_in("<td class='cell-success'>0.7200</td>", html)
    expect_in("<td class='cell-warning'>0.7400</td>", html)
    expect_not_in("<td class='cell-success'>0.7400</td>", html)
    expect_not_in("<td class='cell-success'>0.8000</td>", html)
    expect_in("<td class='cell-danger'>0.0800</td>", html)
    expect_in("<td class='cell-success'>-0.0200</td>", html)
    expect_in("known_labels_map_osm.html", html)
    expect_not_in("known_labels_map.png", html)

    write_index_report(config)
    report_index = (project_dir / "report_index.html").read_text(encoding="utf-8")
    expect_in("Final Dashboard", report_index)
    expect_in("final_dashboard/report.html", report_index)


@pytest.mark.parametrize("case", ["absent", "empty", "null", "constant", "varied", "ndvi_only"])
def test_export_pipeline_seasonality_handles_available_data(tmp_path, monkeypatch, case):
    """Export only configured seasonal data and flag constant observed indices."""
    from ml_classification.shared.reports import final_dashboard as dashboard

    plots = tmp_path / "plots"
    plots.mkdir()
    stale_names = ["seasonal_peak_by_class.png", "seasonal_ndvi_boxplots_by_class_month.png",
                   "seasonal_ndwi_boxplots_by_class_month.png"]
    for name in stale_names:
        (plots / name).write_bytes(b"old plot")
    values = [None, None] if case == "null" else [0.2, 0.2] if case == "constant" else [0.2, 0.6]
    parcels = pd.DataFrame({"initial_label_code": ["A", "A"], "NDVI_median__20231001": values})
    if case != "ndvi_only":
        parcels["NDWI_median__20231001"] = values
    if case == "absent":
        parcels = parcels[["initial_label_code"]]
    if case == "empty":
        parcels = parcels.iloc[:0]
    plot_calls = []
    monkeypatch.setattr(dashboard, "_save_monthly_index_boxplots", lambda *args: plot_calls.append(args))
    monkeypatch.setattr(dashboard, "_save_monthly_index_range_lines", lambda *args: plot_calls.append(args))
    phenology, images, notes = dashboard._export_pipeline_seasonality(parcels, {"A": "Crop A"}, tmp_path, plots)
    expect_equal(len(images), 0 if case == "absent" else 3)
    expect_equal(len(plot_calls), len(images))
    expect_equal(phenology.empty, case in {"absent", "empty", "null"})
    expect_equal(len(notes), 2 if case == "constant" else 0)
    if notes:
        expect_in("every NDVI value", notes[0])
        expect_in("0.20", notes[0])
        expect_in("every NDWI value", notes[1])
    for name in stale_names:
        expect_equal((plots / name).exists(), case == "absent")
    for name in ["monthly_index_by_class.csv", "monthly_index_high_low_by_class.csv"]:
        expect_equal((tmp_path / name).exists(), case != "absent")
    if not phenology.empty:
        exported = pd.read_csv(tmp_path / "monthly_index_by_class.csv")
        expect_equal(exported["class_label"].unique().tolist(), ["Crop A"])
        expect_equal(set(exported["index"]), {"NDVI"} if case == "ndvi_only" else {"NDVI", "NDWI"})


@pytest.mark.parametrize("failure", ["invalid_date", "write_error"])
def test_export_pipeline_seasonality_propagates_failures(tmp_path, monkeypatch, failure):
    """Invalid source dates and failed writes must not produce success results."""
    from ml_classification.shared.reports import final_dashboard as dashboard

    column = "NDVI_median__20231399" if failure == "invalid_date" else "NDVI_median__20231001"
    parcels = pd.DataFrame({"initial_label_code": ["A"], column: [0.2]})
    if failure == "write_error":
        def fail_write(*args, **kwargs):
            raise OSError("test export failure")
        monkeypatch.setattr(pd.DataFrame, "to_csv", fail_write)
    with pytest.raises(ValueError if failure == "invalid_date" else OSError):
        dashboard._export_pipeline_seasonality(parcels, {"A": "Crop A"}, tmp_path, tmp_path)


@pytest.mark.parametrize("train_score", [0.9, None, float("nan")])
@pytest.mark.parametrize("with_confidence", [True, False])
def test_pipeline_dashboard_summary_handles_missing_scores_and_diagnostics(train_score, with_confidence):
    """Keep missing training scores as NaN and include diagnostics only when supplied."""
    from ml_classification.shared.reports import final_dashboard as dashboard

    train = pd.DataFrame({"model": ["best"], "accuracy": [train_score]})
    if train_score is None:
        train = train.iloc[:0]
    test = pd.DataFrame({"model": ["best"], "accuracy": [0.8]})
    ranking = pd.DataFrame({"model": ["best"], "mean_cv_score": [0.85]})
    diagnostics = SimpleNamespace(summary={
        "validation_rows_with_reference": 10, "prediction_accuracy": 0.8, "borda_agreement_rate": 0.9,
        "need_to_check_rows": 2, "need_to_check_rate_among_borda_agreement": 0.2,
    }) if with_confidence else None
    result = dashboard._pipeline_dashboard_summary(
        {"train_rows": 8, "test_rows": 2, "target_labels": ["A", "B"]},
        {"selection_type": "single_model", "selected_models": ["best"]},
        train, test, "best", "accuracy", ranking, diagnostics,
    )
    expect_equal(result["modeled_parcels"], 10)
    expect_equal(result["train_parcels"], 8)
    expect_equal(result["spatial_holdout_parcels"], 2)
    expect_equal(result["modeled_classes"], 2)
    expect_equal(result["best_cv_model"], "best")
    expect_equal(result["best_mean_cv_score"], 0.85)
    expect_equal(result["test_accuracy"], 0.8)
    if pd.isna(train_score):
        expect_true(pd.isna(result["train_test_accuracy_gap"]))
    else:
        expect_equal(result["train_test_accuracy_gap"], pytest.approx(0.1))
    expect_equal("prediction_accuracy" in result, with_confidence)
    if with_confidence:
        expect_equal(result["prediction_validation_rows"], 10)
        expect_equal(result["prediction_accuracy"], 0.8)
        expect_equal(result["borda_agreement_rate"], 0.9)
        expect_equal(result["need_to_check_rows"], 2)
        expect_equal(result["need_to_check_rate_among_borda_agreement"], 0.2)
    defaults = dashboard._pipeline_dashboard_summary({}, {}, train, test, "best", "accuracy", ranking, None)
    expect_equal(defaults["modeled_parcels"], 0)
    expect_equal(defaults["modeled_classes"], 0)
    expect_equal(defaults["selected_models"], [])
    expect_equal(defaults["final_strategy"], None)


@pytest.mark.parametrize("has_phenology", [True, False])
@pytest.mark.parametrize("has_confidence", [True, False])
def test_pipeline_dashboard_data_links_include_only_available_exports(has_phenology, has_confidence):
    """Keep optional exports and the stable report ordering aligned."""
    from ml_classification.shared.reports import final_dashboard as dashboard

    directory = Path("dashboard/data")
    links = dashboard._pipeline_dashboard_data_links(directory, has_phenology=has_phenology, has_confidence=has_confidence)
    expected = ["cv_model_ranking.csv", "train_test_model_metrics.csv"]
    if has_phenology:
        expected[:0] = ["monthly_index_by_class.csv", "monthly_index_high_low_by_class.csv"]
    if has_confidence:
        expected.extend(["diagnostic_group_summary.csv", "need_to_check_by_predicted_class.csv",
                         "need_to_check_reason_matrix.csv"])
    expect_equal([link["path"] for link in links], [directory / name for name in expected])
    expect_true(all(link["label"] for link in links))


@pytest.mark.parametrize("image_keys", [[], ["overview"], ["overview", "class_rate", "reason_heatmap"]])
def test_pipeline_confidence_section_omits_unavailable_images(monkeypatch, image_keys):
    """Render available plots in canonical order and use the limited display table."""
    from ml_classification.shared.reports import final_dashboard as dashboard

    risk = pd.DataFrame({"class": ["A"]})
    display = pd.DataFrame({"Class": ["Crop A"]})
    calls = []
    def prepare_table(table, limit):
        calls.append((table, limit))
        return display
    monkeypatch.setattr(dashboard, "prepare_class_risk_display_table", prepare_table)
    artifacts = SimpleNamespace(
        images={key: Path(f"{key}.png") for key in image_keys},
        tables={"need_to_check_by_predicted_class": risk},
    )
    section = dashboard._pipeline_confidence_section(artifacts)
    expect_equal([image["path"] for image in section["images"]], [Path(f"{key}.png") for key in image_keys])
    expect_equal(section["title"], "Prediction Confidence Diagnostics")
    expect_in("Unknown fill rows are excluded", section["text"])
    expect_true(section["table"] is display)
    expect_true(calls[0][0] is risk)
    expect_equal(calls[0][1], 15)


@pytest.mark.parametrize("columns", [[], ["model", "train_accuracy", "test_accuracy", "accuracy_gap"],
                                     ["model", "test_f1_macro", "test_accuracy", "accuracy_gap", "notes"]])
def test_pipeline_performance_section_styles_only_score_and_gap_columns(columns):
    """Keep train and metadata columns neutral while highlighting strategy holdout cells."""
    from ml_classification.shared.reports import final_dashboard as dashboard

    table = pd.DataFrame(columns=columns)
    section = dashboard._pipeline_performance_section(table, "Voting", Path("plots"))
    expect_true(section["table"] is table)
    expect_equal(section["column_styles"], {column: "success" for column in columns if column.startswith("test_")})
    expect_equal(section["numeric_cell_styles"],
                 {column: {"positive": "danger", "negative": "success"} for column in columns if column.endswith("_gap")})
    expect_equal(section["cell_styles_where"], [{
        "column": "model", "values": ["Voting"],
        "target_columns": [column for column in columns if column.startswith("test_")], "style": "warning",
    }])
    expect_equal(section["highlight_rows_where"], {"column": "model", "values": ["Voting"]})
    expect_equal(section["images"][0]["path"], Path("plots/train_test_model_comparison.png"))


@pytest.mark.parametrize("configured_metric, available_metric", [
    ("accuracy", "accuracy"), ("unavailable", "f1_macro"), ("unavailable", "accuracy"),
])
def test_pipeline_dashboard_without_optional_sections_uses_available_metric(
    tmp_path, monkeypatch, configured_metric, available_metric
):
    """Exercise report orchestration without seasonality, a map, or predictions."""
    from ml_classification.shared.reports import final_dashboard as dashboard

    config = SimpleNamespace(final_dashboard_dir=tmp_path / "dashboard", prepare_dir=tmp_path,
                             train_dir=tmp_path, check_dir=tmp_path, scoring_primary=configured_metric)
    (tmp_path / "prepare_summary.json").write_text("{}", encoding="utf-8")
    pd.DataFrame({"model": ["best"], "best_cv_score": [0.8], "cv_score_std": [0.0],
                  "cv_ranking_metric": [0.8]}).to_csv(tmp_path / "training_summary.csv", index=False)
    pd.DataFrame({"model": ["best"], "fold": [0], "score": [0.8]}).to_csv(
        tmp_path / "best_cv_fold_scores.csv", index=False)
    monkeypatch.setattr(dashboard, "_pipeline_parcels", lambda *args: (pd.DataFrame(), {}))
    monkeypatch.setattr(dashboard, "_save_selected_cv_fold_plot", lambda *args: None)
    chart_metrics = []
    monkeypatch.setattr(dashboard, "_save_train_test_plot", lambda data, metric, path: chart_metrics.append(metric))
    reports = []
    monkeypatch.setattr(dashboard, "write_html_report", lambda *args: reports.append(args))
    metrics = pd.DataFrame({"model": ["best"], available_metric: [0.8]})
    output = dashboard.write_pipeline_final_dashboard(
        config, metrics, metrics, {"selection_type": "single_model", "selected_models": ["best"]},
    )
    expect_equal(output, config.final_dashboard_dir / "report.html")
    expect_equal(chart_metrics, [available_metric])
    sections = {section["title"]: section for section in reports[0][3]}
    expect_not_in("Prediction Confidence Diagnostics", sections)
    expect_equal(sections["Study Area and Parcel Coverage"]["embeds"], [])
    expect_equal(sections["Seasonal Crop Profiles"]["images"], [])
    expect_equal(len(sections["Dashboard Data"]["links"]), 2)
    expect_equal(sections["Evaluation Snapshot"]["kv"][f"test_{available_metric}"], 0.8)
    expect_equal(sections["Best Model and Voting Result"]["table"].empty, True)


@pytest.mark.parametrize("missing", ["strategy", "ranking", "metric"])
def test_pipeline_dashboard_summary_rejects_incomplete_evaluation(missing):
    """Missing required evaluation evidence must not become a plausible score."""
    from ml_classification.shared.reports import final_dashboard as dashboard

    metrics = pd.DataFrame({"model": ["best"], "accuracy": [0.8]})
    test = metrics.iloc[:0] if missing == "strategy" else metrics
    ranking = pd.DataFrame({"model": ["best"], "mean_cv_score": [0.8]})
    if missing == "ranking":
        ranking = ranking.iloc[:0]
    with pytest.raises(KeyError if missing == "metric" else IndexError):
        dashboard._pipeline_dashboard_summary({}, {}, metrics, test, "best",
                                               "absent" if missing == "metric" else "accuracy", ranking, None)
