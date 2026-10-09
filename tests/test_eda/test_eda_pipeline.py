"""Regression tests for the renamed EDA package."""

import warnings

import numpy as np
import pandas as pd
import pytest

from eda import EDAConfig, EDAPipeline
from eda.core import build_eda_artifacts
from eda.core.profiling import _build_data_quality_flags, _build_plot_feature_order
from eda.reporting import html as html_module
from eda.visuals import create_eda_plots
from eda.visuals import plots as plots_module
from tests.utils import expect_equal, expect_false, expect_in, expect_isinstance, expect_not_in, expect_subset, expect_true


def test_build_eda_artifacts_from_a_dataframe() -> None:
    data = pd.DataFrame(
        {"parcel_id": [1, 2, 3, 4, 5, 6], "ndvi": [0.1, 0.2, 0.3, 0.4, 0.5, 2.0], "crop": ["A", "A", "B", "B", "B", "A"]}
    )
    config = EDAConfig(output_dir="unused", include_html_report=False, include_plots=False, include_geospatial=False)

    artifacts = build_eda_artifacts(data, config)

    expect_equal(artifacts["summary"]["rows"], 6)
    expect_in("ndvi", artifacts["numeric_summary"]["column"].tolist())


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
    expect_equal(artifacts["summary"]["target_task"], "classification")
    expect_true(numeric_tests.loc["strong_numeric", "epsilon_squared"] > 0.7)
    expect_true(numeric_tests.loc["strong_numeric", "qvalue"] <= config.feature_selection_alpha)
    expect_true(categorical_tests.loc["strong_category", "cramers_v_bias_corrected"] > 0.8)
    expect_true(ranking["selection_note"].str.startswith("priority candidate").any())
    expect_in("rank_within_method", ranking.columns)
    expect_equal(set(ranking["feature_type"]), {"numeric", "categorical"})


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
    expect_true(correlations.loc["numeric_feature", "abs_spearman_correlation"] > 0.9)
    expect_in("kendall_correlation", correlations.columns)
    expect_true(categorical_tests.loc["category", "epsilon_squared"] > 0)
    expect_equal(set(artifacts["feature_target_associations"]["feature_type"]), {"numeric", "categorical"})
    expect_equal(set(artifacts["target_summary"]["section"]), {"target_numeric"})


def test_normality_analysis_does_not_emit_scipy_anderson_future_warning() -> None:
    data = pd.DataFrame({"value": np.linspace(-2, 2, 30)})
    config = EDAConfig(output_dir="unused", include_html_report=False, include_plots=False, include_geospatial=False)

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        artifacts = build_eda_artifacts(data, config)

    expect_in("anderson_pvalue", artifacts["normality_tests"].columns)


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

    expect_equal(artifacts["normality_tests"]["column"].tolist(), ["feature"])


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

    expected = {
        "feature_target_association_ranking",
        "numeric_target_distribution_comparison",
        "numeric_target_median_percentile_heatmap",
        "categorical_target_composition",
    }
    expect_subset(expected, plot_paths)
    expect_true(all(plot_paths[name].exists() for name in expected))


def test_target_comparison_boxplots_match_ecdf_colors(monkeypatch) -> None:
    data = pd.DataFrame(
        {
            "target": ["B", "B", "B", "A", "A"],
            # The feature subset would rank A first; colors must still follow the full-target mapping (B, then A).
            "numeric_feature": [3.0, np.nan, np.nan, 1.0, 2.0],
        }
    )
    boxplot_calls = []
    ecdf_colors = []
    legend_calls = []

    monkeypatch.setattr(plots_module, "_boxplot", lambda **kwargs: boxplot_calls.append(kwargs))
    monkeypatch.setattr("matplotlib.figure.Figure.savefig", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("matplotlib.axes.Axes.legend", lambda *_args, **_kwargs: legend_calls.append(_kwargs))

    def capture_step(_axis, *_args, **kwargs):
        ecdf_colors.append(kwargs["color"])
        return []

    monkeypatch.setattr("matplotlib.axes.Axes.step", capture_step)

    expect_true(plots_module._plot_numeric_target_distribution_comparison(
        data,
        ["numeric_feature"],
        "target",
        10,
        "unused.png",
    ))

    expected_colors = {"B": plots_module.TARGET_PALETTE[0], "A": plots_module.TARGET_PALETTE[1]}
    expect_equal(boxplot_calls[0]["palette"], expected_colors)
    expect_equal(boxplot_calls[0]["hue"], "target")
    expect_equal(boxplot_calls[0]["hue_order"], ["B", "A"])
    expect_equal(boxplot_calls[0]["saturation"], 1)
    expect_equal(ecdf_colors, list(expected_colors.values()))
    expect_equal(legend_calls, [])


def test_vertical_category_labels_are_smaller_and_truncated() -> None:
    fig, axis = plots_module.plt.subplots()
    axis.set_xticks([0, 1], labels=["short", "a_target_label_longer_than_fifteen"])

    plots_module._format_vertical_category_labels(axis)

    labels = axis.get_xticklabels()
    expect_equal([label.get_text() for label in labels], ["short", "a_target_label_"])
    expect_true(all(label.get_rotation() == 90 for label in labels))
    expect_true(all(label.get_fontsize() == 8 for label in labels))
    plots_module.plt.close(fig)


def test_plot_feature_order_prioritizes_proposal_before_flagged_fallbacks() -> None:
    df = pd.DataFrame(
        {
            "target": [0, 1, 1, 0],
            "chosen_feature": [1, 2, 3, 4],
            "high_missing_feature": [1.0, np.nan, np.nan, np.nan],
            "stable_feature": [10.0, 11.0, 12.0, 13.0],
        }
    )
    config = EDAConfig(output_dir="unused", target_column="target", include_html_report=False, include_plots=False, include_geospatial=False)
    column_profile = pd.DataFrame(
        [
            {"column": "chosen_feature", "non_null_count": 4, "missing_percent": 0.0, "unique_percent": 100.0, "is_constant": False, "logical_type": "numeric"},
            {"column": "high_missing_feature", "non_null_count": 1, "missing_percent": 75.0, "unique_percent": 100.0, "is_constant": False, "logical_type": "numeric"},
            {"column": "stable_feature", "non_null_count": 4, "missing_percent": 0.0, "unique_percent": 100.0, "is_constant": False, "logical_type": "numeric"},
        ]
    )
    quality_flags = _build_data_quality_flags(df, column_profile, config)
    proposal = pd.DataFrame({"feature": ["chosen_feature"], "selection_order": [1]})

    ordered = _build_plot_feature_order(df, config, proposal, column_profile, quality_flags)
    expect_equal(ordered["all"][0], "chosen_feature")
    expect_equal(ordered["all"][-1], "high_missing_feature")


def test_categorical_target_colors_are_consistent_across_report_plots(monkeypatch) -> None:
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import box

    data = pd.DataFrame(
        {
            "target": ["B", "B", "B", "A", "A"],
            "numeric_feature": [3.0, np.nan, np.nan, 1.0, 2.0],
            "category": ["x", "x", "y", "x", "y"],
            "geometry": [box(index, 0, index + 0.5, 0.5) for index in range(5)],
        }
    )
    barh_colors = []
    dataframe_plot_calls = []
    map_geometries = []

    def capture_barh(_axis, *_args, **kwargs):
        barh_colors.extend(kwargs["color"])
        return []

    def capture_dataframe_plot(_frame, *_args, **kwargs):
        dataframe_plot_calls.append(kwargs)
        if isinstance(_frame, gpd.GeoDataFrame):
            map_geometries.extend(_frame.geometry)

    monkeypatch.setattr("matplotlib.axes.Axes.barh", capture_barh)
    monkeypatch.setattr("matplotlib.axes.Axes.legend", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("matplotlib.figure.Figure.savefig", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pd.DataFrame, "plot", capture_dataframe_plot)
    monkeypatch.setattr(gpd.GeoDataFrame, "plot", capture_dataframe_plot)

    plots_module._plot_target_distribution(data["target"], "classification", 10, "unused.png")
    expect_true(plots_module._plot_categorical_target_composition(
        data, ["category"], "target", 10, 10, "unused.png"
    ))
    expect_true(plots_module._plot_geometry_target_heatmap(
        data, "target", "classification", 10, "unused.png"
    ))

    expected_colors = {"B": plots_module.TARGET_PALETTE[0], "A": plots_module.TARGET_PALETTE[1]}
    expect_equal(barh_colors, [expected_colors["A"], expected_colors["B"]])
    expect_equal(dataframe_plot_calls[0]["color"], [expected_colors["A"], expected_colors["B"]])
    expect_equal(dataframe_plot_calls[1]["color"].tolist(), [
        expected_colors["B"],
        expected_colors["B"],
        expected_colors["B"],
        expected_colors["A"],
        expected_colors["A"],
    ])
    expect_equal(len(map_geometries), len(data))
    expect_true(all(point.geom_type == "Point" for point in map_geometries))
    expect_true(all(polygon.covers(point) for polygon, point in zip(data["geometry"], map_geometries)))


def test_numeric_target_median_heatmap_uses_global_target_order_and_percentiles(monkeypatch) -> None:
    data = pd.DataFrame(
        {
            "target": ["B", "B", "B", "A", "A"],
            "numeric_feature": [3.0, np.nan, np.nan, 1.0, 2.0],
        }
    )
    heatmap_profiles = []

    def capture_heatmap(profile, **_kwargs):
        heatmap_profiles.append(profile.copy())

    monkeypatch.setattr(plots_module.sns, "heatmap", capture_heatmap)
    monkeypatch.setattr("matplotlib.figure.Figure.savefig", lambda *_args, **_kwargs: None)

    expect_true(plots_module._plot_numeric_target_median_percentile_heatmap(
        data, ["numeric_feature"], "target", 10, "unused.png"
    ))

    profile = heatmap_profiles[0]
    expect_equal(profile.columns.tolist(), ["B", "A"])
    expect_equal(profile.loc["numeric_feature", "B"], pytest.approx(83.333333))
    expect_equal(profile.loc["numeric_feature", "A"], pytest.approx(33.333333))


def test_target_color_mapping_keeps_other_group_stable_in_subsets() -> None:
    full_target = pd.Series(["A"] * 5 + ["B"] * 4 + ["C"] * 3 + ["D"] * 2)

    target_colors = plots_module._target_color_mapping(full_target, 3)
    subset = plots_module._apply_target_color_mapping(pd.Series(["C", "A", None, "D"]), target_colors)

    expect_equal(target_colors, {
        "A": plots_module.TARGET_PALETTE[0],
        "B": plots_module.TARGET_PALETTE[1],
        "Other": plots_module.TARGET_PALETTE[2],
    })
    expect_equal(subset.iloc[0], "Other")
    expect_equal(subset.iloc[1], "A")
    expect_true(pd.isna(subset.iloc[2]))
    expect_equal(subset.iloc[3], "Other")


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

    monkeypatch.setattr("eda.visuals.plots._plot_qq_plots", capture_qq_columns)
    plot_paths = create_eda_plots(data, artifacts, config)

    expect_equal(captured_columns, ["best_feature", "second_feature"])
    expect_in("numeric_qq_plots", plot_paths)
    expect_not_in("numeric_qq_plots_002", plot_paths)
    expect_equal(artifacts["outlier_summary"]["column"].tolist(), ["noise_feature", "best_feature", "second_feature"])
    expect_equal(set(artifacts["multicollinearity"]["column"]), {"best_feature", "second_feature"})
    expect_equal(len(artifacts["multicollinearity"]), config.max_features_per_plot)


def test_create_target_plots_classification_branches(tmp_path, monkeypatch) -> None:
    data = pd.DataFrame({"target": ["A", "B", "A"], "num_feature": [1.0, 2.0, 3.0], "cat_feature": ["x", "y", "x"]})
    config = EDAConfig(
        output_dir=tmp_path,
        target_column="target",
        target_task="classification",
        include_html_report=False,
        include_geospatial=False,
        max_features_per_plot=2,
    )
    artifacts = {
        "numeric_target_tests": pd.DataFrame({"column": ["num_feature"]}),
        "categorical_target_tests": pd.DataFrame({"column": ["cat_feature"]}),
        "numeric_target_correlations": pd.DataFrame({"column": ["num_feature"]}),
        "feature_target_associations": pd.DataFrame({"feature": ["num_feature"], "score": [1.0]}),
    }
    plot_paths = {}
    plots_dir = tmp_path / "plots"
    plots_dir.mkdir(parents=True)
    target_distribution_calls = []

    monkeypatch.setattr(
        plots_module,
        "_plot_target_distribution",
        lambda target_series, task, max_levels, output_path: target_distribution_calls.append(output_path),
    )
    monkeypatch.setattr(
        plots_module, "_plot_numeric_target_median_percentile_heatmap", lambda *_args, **_kwargs: True
    )
    monkeypatch.setattr(
        plots_module, "_plot_numeric_target_distribution_comparison", lambda *_args, **_kwargs: True
    )
    monkeypatch.setattr(plots_module, "_plot_categorical_target_composition", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(plots_module, "_plot_numeric_target_correlations", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(plots_module, "_plot_numeric_target_scatter", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(plots_module, "_plot_feature_target_association_ranking", lambda *_args, **_kwargs: None)

    plots_module._create_target_plots(
        df=data,
        artifacts=artifacts,
        config=config,
        plots_dir=plots_dir,
        plot_paths=plot_paths,
        target_task="classification",
        numeric_feature_columns=["num_feature"],
        categorical_columns=["cat_feature"],
    )

    expect_equal(len(target_distribution_calls), 1)
    expected_keys = {
        "target_distribution",
        "numeric_target_median_percentile_heatmap",
        "numeric_target_distribution_comparison",
        "categorical_target_composition",
        "numeric_target_correlations",
        "numeric_target_scatter",
        "feature_target_association_ranking",
    }
    expect_subset(expected_keys, set(plot_paths))


def test_create_target_plots_regression_branch(tmp_path, monkeypatch) -> None:
    data = pd.DataFrame({"target": [1.0, 2.0, 3.0], "cat_feature": ["x", "y", "x"]})
    config = EDAConfig(
        output_dir=tmp_path,
        target_column="target",
        target_task="regression",
        include_html_report=False,
        include_geospatial=False,
        max_features_per_plot=2,
    )
    artifacts = {
        "categorical_numeric_target_tests": pd.DataFrame({"column": ["cat_feature"]}),
    }
    plot_paths = {}
    plots_dir = tmp_path / "plots"
    plots_dir.mkdir(parents=True)

    monkeypatch.setattr(plots_module, "_plot_target_distribution", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        plots_module, "_plot_numeric_target_by_categorical_features", lambda *_args, **_kwargs: True
    )

    plots_module._create_target_plots(
        df=data,
        artifacts=artifacts,
        config=config,
        plots_dir=plots_dir,
        plot_paths=plot_paths,
        target_task="regression",
        numeric_feature_columns=[],
        categorical_columns=["cat_feature"],
    )

    expect_in("target_distribution", plot_paths)
    expect_in("numeric_target_by_categorical_features", plot_paths)
    expect_not_in("categorical_target_composition", plot_paths)


def test_create_geospatial_plots_adds_expected_outputs(tmp_path, monkeypatch) -> None:
    data = pd.DataFrame({"geometry": [1, 2, 3], "target": ["A", "B", "A"]})
    config = EDAConfig(
        output_dir=tmp_path,
        target_column="target",
        target_task="classification",
        include_html_report=False,
        include_geospatial=True,
    )
    plot_paths = {}
    plots_dir = tmp_path / "plots"
    plots_dir.mkdir(parents=True)

    monkeypatch.setattr(plots_module, "_plot_geometry_overview", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(plots_module, "_plot_geometry_target_heatmap", lambda *_args, **_kwargs: True)

    plots_module._create_geospatial_plots(
        df=data,
        config=config,
        plots_dir=plots_dir,
        plot_paths=plot_paths,
        target_task="classification",
    )

    expect_in("geometry_overview", plot_paths)
    expect_in("geometry_target_heatmap", plot_paths)


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
    expect_equal(artifacts["summary"]["multivariate_rows_scored"], len(data))
    expect_equal(artifacts["summary"]["multivariate_outliers_found"], len(outliers))
    expect_equal(artifacts["summary"]["multivariate_features_used"], ["x", "y"])
    expect_true(annotated["is_outlier_97_5pct"].notna().all())
    expect_equal(int(annotated["is_outlier_97_5pct"].sum()), len(outliers))
    expect_false(outliers.empty)
    expect_true(outliers["is_outlier_97_5pct"].all())


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

    expect_equal(proposal.loc["strong", "proposed_action"], "use")
    expect_equal(proposal.loc["strong_copy", "proposed_action"], "review")
    expect_equal(proposal.loc["strong_copy", "redundant_with"], "strong")
    expect_equal(proposal.loc["noise", "proposed_action"], "exclude")
    expect_equal(artifacts["plot_feature_order"]["numeric"], ["strong", "strong_copy", "noise"])
    expect_equal(artifacts["feature_selection_criteria"]["criterion_order"].tolist(), [1, 2, 3, 4, 5, 6])
    expect_in("Proposed initial ML features (1): strong", artifacts["feature_selection_message"])


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

    expect_isinstance(result["annotated_data"], geopandas.GeoDataFrame)
    expect_equal(result["annotated_data"].crs, data.crs)
    expect_in("is_outlier_97_5pct", reloaded.columns)
    expect_equal(result["proposed_features"], ["x"])
    expect_equal(result["saved_paths"]["numeric_summary"].relative_to(tmp_path).as_posix(), "tables/numeric/numeric_summary.csv")
    expect_equal(result["saved_paths"]["feature_selection_proposal"].relative_to(tmp_path).as_posix(), "tables/feature_selection/feature_selection_proposal.csv")
    expect_equal(result["saved_paths"]["target_summary"].relative_to(tmp_path).as_posix(), "tables/target/target_summary.csv")
    expect_in("Proposed initial ML features (1): x", capsys.readouterr().out)


def test_geometry_report_uses_target_heatmap_without_geometry_overview() -> None:
    data = pd.DataFrame(
        {
            "target": ["A", "A", "B", "B"],
            "feature": [1.0, 2.0, 3.0, 4.0],
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
    artifacts["geospatial_summary"] = pd.DataFrame({"metric": ["geometry_count"], "value": [4]})

    tabs = html_module._build_report_tabs(
        artifacts,
        {
            "geometry_overview": "geometry_overview.png",
            "geometry_target_heatmap": "geometry_target_heatmap.png",
        },
    )

    geometry_sections = next(tab["sections"] for tab in tabs if tab["id"] == "geometry")
    report_image_keys = {
        image_key
        for section in geometry_sections
        for image_key in section.get("images", {})
    }
    expect_not_in("geometry_overview", report_image_keys)
    expect_in("geometry_target_heatmap", report_image_keys)
