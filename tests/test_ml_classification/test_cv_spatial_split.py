"""Unit tests for spatial holdout splitting and cross-validation fold construction."""

from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, box

from ml_classification.step_02_prepare.libraries import cross_validation
from ml_classification.step_02_prepare.libraries.spatial_split import SpatialTrainTestSplitter
from tests.utils import expect_equal, expect_true

FEATURES = ["f1"]


def _grid_frame(cells_per_side=4, points_per_cell=6, crs="EPSG:2100"):
    """Return points spread over a square grid, with both classes present in every cell."""
    rows = []
    for cx in range(cells_per_side):
        for cy in range(cells_per_side):
            for k in range(points_per_cell):
                x = cx * 100 + 10 + k * 10
                y = cy * 100 + 10 + k * 10
                rows.append({"geometry": Point(x, y), "label": "a" if k % 2 == 0 else "b", "f1": float(x + y)})
    frame = gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)
    frame["_target_encoded"] = (frame["label"] == "b").astype(int)
    return frame


def _cv_config(**overrides):
    """Return the configuration attributes consumed by ``build_cv_folds``."""
    values = {"cv_folds": 3, "random_state": 0, "cv_validation_method": "disjoint", "spatial_split_method": "by_group"}
    values.update(overrides)
    return SimpleNamespace(**values)


# --------------------------------------------------------------------------------
# SpatialTrainTestSplitter
# --------------------------------------------------------------------------------


def test_split_rejects_unknown_method():
    """Only ``by_group`` and ``by_row`` are accepted."""
    with pytest.raises(ValueError, match="Error: Spatial split method"):
        SpatialTrainTestSplitter().split(_grid_frame(), "label", 0.25, method="random")


@pytest.mark.parametrize("method", ["by_group", "by_row"])
def test_split_validates_test_size_geometry_column_and_nulls(method):
    """Both split methods validate the test fraction and geometry availability."""
    splitter = SpatialTrainTestSplitter(grid_size=4)
    frame = _grid_frame()
    with pytest.raises(ValueError, match="Error: test_size must be strictly between 0 and 1"):
        splitter.split(frame, "label", 1.0, method=method)
    with pytest.raises(ValueError, match="Error: spatial_split=True requires a 'geometry' column"):
        splitter.split(pd.DataFrame(frame.drop(columns="geometry")), "label", 0.25, method=method)
    nulls = pd.DataFrame({"geometry": [None, None], "label": ["a", "b"]})
    with pytest.raises(ValueError, match="Error: spatial_split=True requires non-null geometries"):
        splitter.split(nulls, "label", 0.25, method=method)


def test_split_by_group_holds_out_whole_cells_and_keeps_all_rows():
    """No grid cell appears in both sets and every row lands in exactly one set."""
    splitter = SpatialTrainTestSplitter(random_state=1, grid_size=4)
    frame = _grid_frame()

    train, test = splitter.split(frame, "label", 0.25, method="by_group")

    expect_equal(len(train) + len(test), len(frame))
    # Re-derive cells on the full extent so train/test labels share one grid.
    groups = splitter.build_spatial_groups(frame)
    cell_by_x_y = {(geom.x, geom.y): cell for geom, cell in zip(frame.geometry, groups)}
    train_cells = {cell_by_x_y[(g.x, g.y)] for g in train.geometry}
    test_cells = {cell_by_x_y[(g.x, g.y)] for g in test.geometry}
    expect_equal(train_cells & test_cells, set())
    expect_equal(set(train["label"]), {"a", "b"})


def test_split_by_group_warns_for_geographic_crs():
    """Gridding geographic coordinates triggers a projection warning."""
    frame = _grid_frame(crs=None)
    frame["geometry"] = [Point(p.x / 1000, p.y / 1000) for p in frame.geometry]
    frame = frame.set_crs("EPSG:4326")

    with pytest.warns(UserWarning, match="geographic CRS"):
        SpatialTrainTestSplitter(grid_size=4).split(frame, "label", 0.25, method="by_group")


def test_split_by_group_requires_two_occupied_cells():
    """A dataset that falls into one cell cannot be held out by group."""
    frame = _grid_frame(cells_per_side=1)
    with pytest.raises(ValueError, match="Error: Spatial holdout requires at least two occupied grid cells"):
        SpatialTrainTestSplitter(grid_size=1).split(frame, "label", 0.25, method="by_group")


def test_split_by_row_keeps_every_class_in_both_sets():
    """Row-wise spatial splitting retains every class and the requested test size."""
    frame = _grid_frame()

    train, test = SpatialTrainTestSplitter(random_state=3, grid_size=4).split(frame, "label", 0.25, method="by_row")

    expect_equal(len(test), int(np.ceil(0.25 * len(frame))))
    expect_equal(len(train) + len(test), len(frame))
    expect_equal(set(train["label"]), {"a", "b"})
    expect_equal(set(test["label"]), {"a", "b"})


def test_split_by_row_warns_when_test_size_is_adjusted():
    """A tiny test fraction is raised so every class still reaches the test set."""
    frame = _grid_frame(cells_per_side=2, points_per_cell=4)
    with pytest.warns(UserWarning, match="Adjusted test rows"):
        _, test = SpatialTrainTestSplitter(grid_size=2).split(frame, "label", 0.01, method="by_row")
    expect_equal(len(test), 2)


def test_split_by_row_rejects_missing_labels_and_singleton_classes():
    """Unknown labels and classes with a single row are rejected."""
    splitter = SpatialTrainTestSplitter(grid_size=2)
    unlabeled = _grid_frame(cells_per_side=2)
    unlabeled.loc[0, "label"] = None
    with pytest.raises(ValueError, match="Error: Row-wise spatial splitting requires non-empty data"):
        splitter.split_by_row(unlabeled, "label", 0.25)
    rare = _grid_frame(cells_per_side=2)
    rare.loc[0, "label"] = "rare"
    with pytest.raises(ValueError, match=r"Unsupported classes: \['rare'\]"):
        splitter.split_by_row(rare, "label", 0.25)


def test_build_spatial_groups_labels_missing_geometries_and_validates_input():
    """Rows without geometry get a fallback cell; unusable inputs are rejected."""
    splitter = SpatialTrainTestSplitter(grid_size=2)
    frame = _grid_frame(cells_per_side=2)
    frame.loc[0, "geometry"] = None

    groups = splitter.build_spatial_groups(frame)

    expect_equal(groups.iloc[0], "cell_unassigned")
    expect_true(groups.iloc[1:].str.match(r"^cell_\d+_\d+$").all(), "Located rows must receive grid cells")
    with pytest.raises(ValueError, match="Error: spatial_split=True requires a 'geometry' column for spatial CV groups"):
        splitter.build_spatial_groups(pd.DataFrame({"x": [1]}))
    with pytest.raises(ValueError, match="Error: Could not compute representative points"):
        splitter.build_spatial_groups(gpd.GeoDataFrame({"geometry": [None]}, geometry="geometry"))


def test_build_spatial_groups_handles_identical_points():
    """Points sharing one coordinate still receive a valid single cell."""
    frame = gpd.GeoDataFrame({"geometry": [Point(5, 5)] * 3}, geometry="geometry", crs="EPSG:2100")
    expect_equal(set(SpatialTrainTestSplitter(grid_size=3).build_spatial_groups(frame)), {"cell_0_0"})


def test_assign_points_to_grid_bins_points_outside_supplied_cells():
    """Points not covered by any supplied cell are binned arithmetically."""
    splitter = SpatialTrainTestSplitter(grid_size=2)
    points = gpd.GeoDataFrame({"geometry": [Point(0, 0), Point(10, 10)]}, geometry="geometry", crs="EPSG:2100")
    only_first_cell = gpd.GeoDataFrame({"cell_id": ["cell_0_0"]}, geometry=[box(0, 0, 5, 5)], crs="EPSG:2100")

    assignments = splitter._assign_points_to_grid(points, only_first_cell)

    expect_equal(assignments.tolist(), ["cell_0_0", "cell_1_1"])


# --------------------------------------------------------------------------------
# build_cv_folds
# --------------------------------------------------------------------------------


def _assert_disjoint_cover(folds, n_rows):
    """Check that validation folds partition the rows and never overlap their training rows."""
    validated = np.concatenate([fold["valid_index"] for fold in folds])
    expect_equal(sorted(validated.tolist()), list(range(n_rows)))
    for fold in folds:
        expect_equal(set(fold["train_index"]) & set(fold["valid_index"]), set())


def test_build_cv_folds_stratified_partitions_rows():
    """Without a spatial splitter, stratified folds validate every row once."""
    frame = _grid_frame()

    folds = cross_validation.build_cv_folds(_cv_config(), frame, FEATURES, spatial_splitter=None)

    expect_equal([fold["fold"] for fold in folds], [0, 1, 2])
    _assert_disjoint_cover(folds, len(frame))


def test_build_cv_folds_by_group_keeps_cells_intact():
    """Grouped spatial folds never share a grid cell between training and validation."""
    frame = _grid_frame()
    splitter = SpatialTrainTestSplitter(grid_size=4)
    groups = splitter.build_spatial_groups(frame)

    folds = cross_validation.build_cv_folds(_cv_config(), frame, FEATURES, splitter)

    _assert_disjoint_cover(folds, len(frame))
    for fold in folds:
        expect_equal(set(groups.iloc[fold["train_index"]]) & set(groups.iloc[fold["valid_index"]]), set())


def test_build_cv_folds_by_row_balances_fold_sizes():
    """Row-wise spatial folds validate every row once with near-equal sizes."""
    frame = _grid_frame()

    folds = cross_validation.build_cv_folds(
        _cv_config(spatial_split_method="by_row"), frame, FEATURES, SpatialTrainTestSplitter(grid_size=4)
    )

    _assert_disjoint_cover(folds, len(frame))
    sizes = [len(fold["valid_index"]) for fold in folds]
    expect_true(max(sizes) - min(sizes) <= 1, f"Fold sizes must be balanced: {sizes}")


@pytest.mark.parametrize(("splitter", "method"), [(None, "by_group"), ("spatial", "by_row"), ("spatial", "by_group")])
def test_build_cv_folds_overlapping_draws_independent_folds(splitter, method):
    """Overlapping validation draws one target-sized validation set per fold."""
    frame = _grid_frame()
    spatial = SpatialTrainTestSplitter(grid_size=4) if splitter else None
    config = _cv_config(spatial_split_method=method)

    folds = cross_validation.build_cv_folds(config, frame, FEATURES, spatial, validation_method="overlapping")

    expect_equal(len(folds), config.cv_folds)
    for fold in folds:
        expect_equal(set(fold["train_index"]) & set(fold["valid_index"]), set())
        expect_equal(len(fold["train_index"]) + len(fold["valid_index"]), len(frame))
        if method == "by_row" or splitter is None:
            expect_equal(len(fold["valid_index"]), round(len(frame) / config.cv_folds))


def test_build_cv_folds_uses_configured_validation_method(monkeypatch):
    """The config's validation method applies when no override is passed."""
    calls = []

    def fake_overlapping(*args):
        calls.append(args)
        return []

    monkeypatch.setattr(cross_validation, "overlapping_cv_splits", fake_overlapping)
    folds = cross_validation.build_cv_folds(_cv_config(cv_validation_method="overlapping"), _grid_frame(), FEATURES, None)

    expect_equal(folds, [])
    expect_equal(len(calls), 1)


def test_build_cv_folds_rejects_spatial_folds_missing_training_classes(monkeypatch):
    """A spatial fold whose training rows lack a class is reported with its fold id."""
    frame = _grid_frame()
    only_class_a = np.flatnonzero(frame["_target_encoded"].to_numpy() == 0)
    only_class_b = np.flatnonzero(frame["_target_encoded"].to_numpy() == 1)
    monkeypatch.setattr(cross_validation, "overlapping_cv_splits", lambda *_args: [(only_class_a, only_class_b)])

    with pytest.raises(ValueError, match=r"Error: Spatial CV fold 0 has no training parcels for encoded classes \[1\]"):
        cross_validation.build_cv_folds(
            _cv_config(), frame, FEATURES, SpatialTrainTestSplitter(grid_size=4), validation_method="overlapping"
        )


def test_grouped_cv_requires_enough_occupied_cells():
    """Grouped CV needs at least ``cv_folds`` occupied cells for both validation methods."""
    frame = _grid_frame(cells_per_side=1)
    groups = pd.Series(["cell_0_0"] * len(frame))
    config = _cv_config(cv_folds=3)
    with pytest.raises(ValueError, match="Error: Spatial CV requires at least cv_folds occupied grid cells"):
        cross_validation.balanced_spatial_cv_splits(config, frame, FEATURES, groups)
    with pytest.raises(ValueError, match="Error: Spatial CV requires at least cv_folds occupied grid cells"):
        cross_validation._overlapping_group_cv_splits(config, frame, FEATURES, frame["_target_encoded"], groups)


def test_grouped_cv_rejects_classes_confined_to_one_cell():
    """A class found in a single cell cannot stay in every training fold."""
    frame = _grid_frame(cells_per_side=2)
    groups = SpatialTrainTestSplitter(grid_size=2).build_spatial_groups(frame)
    confined = groups == groups.iloc[0]
    frame.loc[confined, "_target_encoded"] = 2

    with pytest.raises(ValueError, match="Error: Spatial CV could not retain every class"):
        cross_validation.balanced_spatial_cv_splits(_cv_config(cv_folds=2), frame, FEATURES, groups)
