"""Regression tests for the renamed EDA package."""

import warnings

import numpy as np
import pandas as pd

from eda_pipeline import EDAConfig, EDAPipeline
from eda_pipeline.core import build_eda_artifacts
from eda_pipeline.visuals import create_eda_plots


def test_build_eda_artifacts_from_a_dataframe() -> None:
    data = pd.DataFrame(
        {"parcel_id": [1, 2, 3, 4, 5, 6], "ndvi": [0.1, 0.2, 0.3, 0.4, 0.5, 2.0], "crop": ["A", "A", "B", "B", "B", "A"]}
    )
    config = EDAConfig(output_dir="unused", include_html_report=False, include_plots=False, include_geospatial=False)

    artifacts = build_eda_artifacts(data, config)

    assert artifacts["summary"]["rows"] == 6
    assert "ndvi" in artifacts["numeric_summary"]["column"].tolist()


def test_classification_feature_screening_includes_effect_sizes_and_fdr() -> None:
    target = np.repeat([0, 1], 30)
    data = pd.DataFrame(
        {
            "target": target,
            "strong_numeric": np.concatenate([np.linspace(0, 1, 30), np.linspace(4, 5, 30)]),
            "weak_numeric": np.tile([0.0, 1.0, 2.0], 20),
            "strong_category": np.where(target == 0, "dry", "wet"),
            "mixed_category": np.tile(["north", "south", "east"], 20),
        }
    )
    config = EDAConfig(
        output_dir="unused",
        target_column="target",
        target_task="classification",
        include_html_report=False,
        include_plots=False,
        include_geospatial=False,
    )

    artifacts = build_eda_artifacts(data, config)

    numeric_tests = artifacts["numeric_target_tests"].set_index("column")
    categorical_tests = artifacts["categorical_target_tests"].set_index("column")
    ranking = artifacts["feature_target_associations"]
    assert artifacts["summary"]["target_task"] == "classification"
    assert numeric_tests.loc["strong_numeric", "epsilon_squared"] > 0.7
    assert numeric_tests.loc["strong_numeric", "qvalue"] <= config.feature_selection_alpha
    assert categorical_tests.loc["strong_category", "cramers_v_bias_corrected"] > 0.8
    assert ranking["selection_note"].str.startswith("priority candidate").any()
    assert "rank_within_method" in ranking.columns
    assert set(ranking["feature_type"]) == {"numeric", "categorical"}


def test_regression_feature_screening_compares_numeric_and_categorical_features() -> None:
    numeric_feature = np.linspace(0, 10, 60)
    category = np.repeat(["low", "high"], 30)
    target = 3 * numeric_feature + np.where(category == "high", 5, 0)
    data = pd.DataFrame(
        {"target": target, "numeric_feature": numeric_feature, "noise": np.sin(numeric_feature), "category": category}
    )
    config = EDAConfig(
        output_dir="unused",
        target_column="target",
        target_task="regression",
        include_html_report=False,
        include_plots=False,
        include_geospatial=False,
    )

    artifacts = build_eda_artifacts(data, config)

    correlations = artifacts["numeric_target_correlations"].set_index("column")
    categorical_tests = artifacts["categorical_numeric_target_tests"].set_index("column")
    assert correlations.loc["numeric_feature", "abs_spearman_correlation"] > 0.9
    assert "kendall_correlation" in correlations.columns
    assert categorical_tests.loc["category", "epsilon_squared"] > 0
    assert set(artifacts["feature_target_associations"]["feature_type"]) == {"numeric", "categorical"}
    assert set(artifacts["target_summary"]["section"]) == {"target_numeric"}


def test_normality_analysis_does_not_emit_scipy_anderson_future_warning() -> None:
    data = pd.DataFrame({"value": np.linspace(-2, 2, 30)})
    config = EDAConfig(output_dir="unused", include_html_report=False, include_plots=False, include_geospatial=False)

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        artifacts = build_eda_artifacts(data, config)

    assert "anderson_pvalue" in artifacts["normality_tests"].columns


def test_normality_analysis_excludes_target_id_and_diagnostic_columns() -> None:
    data = pd.DataFrame(
        {
            "parcel_id": np.arange(30),
            "target": np.tile([0, 1], 15),
            "feature": np.linspace(-2, 2, 30),
            "is_outlier_97_5pct": np.tile([0, 1], 15),
        }
    )
    config = EDAConfig(
        output_dir="unused",
        target_column="target",
        target_task="classification",
        id_column="parcel_id",
        include_html_report=False,
        include_plots=False,
        include_geospatial=False,
    )

    artifacts = build_eda_artifacts(data, config)

    assert artifacts["normality_tests"]["column"].tolist() == ["feature"]


def test_target_comparison_plots_are_created(tmp_path) -> None:
    target = np.repeat(["A", "B"], 20)
    data = pd.DataFrame(
        {
            "target": target,
            "numeric_feature": np.concatenate([np.linspace(0, 1, 20), np.linspace(3, 4, 20)]),
            "category": np.where(target == "A", "north", "south"),
        }
    )
    config = EDAConfig(
        output_dir=tmp_path,
        target_column="target",
        include_html_report=False,
        include_geospatial=False,
        max_features_per_plot=2,
    )
    artifacts = build_eda_artifacts(data, config)

    plot_paths = create_eda_plots(data, artifacts, config)

    expected = {"feature_target_association_ranking", "numeric_target_distribution_comparison", "categorical_target_composition"}
    assert expected.issubset(plot_paths)
    assert all(plot_paths[name].exists() for name in expected)


def test_qq_plot_uses_validity_ranked_global_feature_limit(tmp_path, monkeypatch) -> None:
    target = np.repeat([0, 1], 30)
    data = pd.DataFrame(
        {
            "parcel_id": np.arange(60),
            "target": target,
            "noise_feature": np.tile([0.0, 1.0, 2.0], 20),
            "best_feature": target * 10 + np.tile(np.linspace(0, 1, 30), 2),
            "second_feature": target * 4 + np.tile(np.linspace(0, 2, 30), 2),
        }
    )
    config = EDAConfig(
        output_dir=tmp_path,
        target_column="target",
        target_task="classification",
        id_column="parcel_id",
        max_features_per_plot=2,
        include_html_report=False,
        include_geospatial=False,
    )
    artifacts = build_eda_artifacts(data, config)
    captured_columns = []

    def capture_qq_columns(df, columns, output_path, plot_config) -> None:
        captured_columns.extend(columns)
        output_path.touch()

    monkeypatch.setattr("eda_pipeline.visuals.plots._plot_qq_plots", capture_qq_columns)
    plot_paths = create_eda_plots(data, artifacts, config)

    assert captured_columns == ["best_feature", "second_feature"]
    assert "numeric_qq_plots" in plot_paths
    assert "numeric_qq_plots_002" not in plot_paths
    assert artifacts["outlier_summary"]["column"].tolist() == ["noise_feature", "best_feature", "second_feature"]
    assert set(artifacts["multicollinearity"]["column"]) == {"best_feature", "second_feature"}
    assert len(artifacts["multicollinearity"]) == config.max_features_per_plot


def test_multivariate_outlier_artifact_contains_all_flags_and_annotates_every_row() -> None:
    random = np.random.default_rng(42)
    values = random.normal(size=(120, 2))
    values[-1] = [12, 12]
    data = pd.DataFrame({"parcel_id": np.arange(120), "x": values[:, 0], "y": values[:, 1]})
    data.index = np.repeat(np.arange(60), 2)  # Duplicate labels verify that annotation uses row position, not label alignment.
    config = EDAConfig(
        output_dir="unused",
        id_column="parcel_id",
        include_html_report=False,
        include_plots=False,
        include_geospatial=False,
    )

    artifacts = build_eda_artifacts(data, config)

    annotated = artifacts["annotated_data"]
    outliers = artifacts["multivariate_outliers"]
    assert artifacts["summary"]["multivariate_rows_scored"] == len(data)
    assert artifacts["summary"]["multivariate_outliers_found"] == len(outliers)
    assert artifacts["summary"]["multivariate_features_used"] == ["x", "y"]
    assert annotated["is_outlier_97_5pct"].notna().all()
    assert int(annotated["is_outlier_97_5pct"].sum()) == len(outliers)
    assert not outliers.empty
    assert outliers["is_outlier_97_5pct"].all()


def test_feature_proposal_orders_evidence_and_removes_redundant_candidates() -> None:
    target = np.repeat([0, 1], 30)
    strong = np.concatenate([np.linspace(0, 1, 30), np.linspace(4, 5, 30)])
    data = pd.DataFrame(
        {
            "target": target,
            "strong": strong,
            "strong_copy": 2 * strong,
            "noise": np.tile([0.0, 1.0, 2.0], 20),
            "parcel_id": np.arange(60),
        }
    )
    config = EDAConfig(
        output_dir="unused",
        target_column="target",
        target_task="classification",
        id_column="parcel_id",
        include_html_report=False,
        include_plots=False,
        include_geospatial=False,
    )

    artifacts = build_eda_artifacts(data, config)
    proposal = artifacts["feature_selection_proposal"].set_index("feature")

    assert proposal.loc["strong", "proposed_action"] == "use"
    assert proposal.loc["strong_copy", "proposed_action"] == "review"
    assert proposal.loc["strong_copy", "redundant_with"] == "strong"
    assert proposal.loc["noise", "proposed_action"] == "exclude"
    assert artifacts["plot_feature_order"]["numeric"] == ["strong", "strong_copy", "noise"]
    assert artifacts["feature_selection_criteria"]["criterion_order"].tolist() == [1, 2, 3, 4, 5, 6]
    assert "Proposed initial ML features (1): strong" in artifacts["feature_selection_message"]


def test_pipeline_preserves_geodataframe_and_writes_annotated_geoparquet(tmp_path, capsys) -> None:
    geopandas = __import__("geopandas")
    geometry = geopandas.points_from_xy(np.arange(40), np.arange(40))
    target = np.repeat(["A", "B"], 20)
    data = geopandas.GeoDataFrame(
        {
            "parcel_id": np.arange(40),
            "target": target,
            "x": np.concatenate([np.linspace(0, 1, 20), np.linspace(3, 4, 20)]),
            "y": np.sin(np.arange(40)),
        },
        geometry=geometry,
        crs="EPSG:4326",
    )
    config = EDAConfig(
        output_dir=tmp_path,
        target_column="target",
        target_task="classification",
        id_column="parcel_id",
        include_html_report=False,
        include_plots=False,
    )

    result = EDAPipeline(config).run(data)
    reloaded = geopandas.read_parquet(result["saved_paths"]["annotated_data"])

    assert isinstance(result["annotated_data"], geopandas.GeoDataFrame)
    assert result["annotated_data"].crs == data.crs
    assert "is_outlier_97_5pct" in reloaded.columns
    assert result["proposed_features"] == ["x"]
    assert result["saved_paths"]["numeric_summary"].relative_to(tmp_path).as_posix() == "tables/numeric/numeric_summary.csv"
    assert (
        result["saved_paths"]["feature_selection_proposal"].relative_to(tmp_path).as_posix()
        == "tables/feature_selection/feature_selection_proposal.csv"
    )
    assert result["saved_paths"]["target_summary"].relative_to(tmp_path).as_posix() == "tables/target/target_summary.csv"
    assert "Proposed initial ML features (1): x" in capsys.readouterr().out
