"""Coverage and class guarantees for row-wise spatial holdouts."""

from itertools import combinations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from satellites.ml_classification.core.spatial_allocation import allocate_spatial_rows
from satellites.ml_classification.core.spatial_split import SpatialTrainTestSplitter


def _frame(sizes, labels=None):
    points = []
    for cell, size in enumerate(sizes):
        points.extend(Point((cell // 2) * 10 + row * 0.001, (cell % 2) * 10 + row * 0.001) for row in range(size))
    return gpd.GeoDataFrame(
        {"row_id": np.arange(sum(sizes)), "label": ["A"] * sum(sizes) if labels is None else labels,
         "tile": np.repeat(np.arange(len(sizes)), sizes)}, geometry=points, crs="EPSG:32634",
    )


def _assert_partition(source, train, test):
    assert set(train.row_id).isdisjoint(test.row_id)
    assert sorted(train.row_id.tolist() + test.row_id.tolist()) == source.row_id.tolist()
    assert set(train.label) == set(test.label) == set(source.label)
    assert train.crs == test.crs == source.crs


def test_holdout_covers_sparse_cells_when_possible():
    source = _frame([2, 2, 18, 18])
    train, test = SpatialTrainTestSplitter(grid_size=2).split(source, "label", 0.2, method="by_row")
    _assert_partition(source, train, test)
    assert (len(train), len(test)) == (32, 8)
    assert train.tile.nunique() == test.tile.nunique() == 4


def test_class_presence_overrides_requested_size():
    source = _frame([2, 2, 18, 18], ["rare"] * 2 + ["common"] * 38)
    with pytest.warns(UserWarning, match="Adjusted test rows"):
        train, test = SpatialTrainTestSplitter(grid_size=2).split_by_row(source, "label", 0.01)
    _assert_partition(source, train, test)
    assert len(test) == 2


def test_coverage_takes_priority_over_class_proportions():
    # Reaching all four cells needs three rare-class test rows, although a
    # strictly proportional split would spend nearly every slot on the common class.
    source = _frame([2, 2, 2, 94], ["rare"] * 6 + ["common"] * 94)
    train, test = SpatialTrainTestSplitter(grid_size=2).split_by_row(source, "label", 0.04)
    _assert_partition(source, train, test)
    assert train.tile.nunique() == test.tile.nunique() == 4
    assert test.label.value_counts().to_dict() == {"rare": 3, "common": 1}


def test_sparse_singletons_do_not_prevent_class_complete_split():
    source = _frame([1, 1, 18, 20], ["rare"] * 2 + ["common"] * 38)
    train, test = SpatialTrainTestSplitter(grid_size=2).split_by_row(source, "label", 0.2)
    _assert_partition(source, train, test)
    assert train.tile.nunique() == test.tile.nunique() == 3


def test_small_test_budget_spreads_across_distant_regions():
    # Four nearby cells and a distant region: both sets can cover the two broad
    # regions, even though test can only visit two of the five fine cells.
    groups = pd.Series(np.repeat(["cell_0_0", "cell_0_1", "cell_1_0", "cell_1_1", "cell_7_7"], 2))
    assignments = allocate_spatial_rows(pd.Series(["A"] * 10), groups, np.array([8, 2]), 42)
    selected = groups[assignments == 1]
    assert selected.nunique() == 2
    assert "cell_7_7" in set(selected)
    assert groups[assignments == 0].nunique() == 5


def test_shared_grid_duplicate_indices_and_reproducibility():
    source = _frame([4, 4, 16, 16], ["A", "B"] * 20)
    source.index = ["duplicate"] * len(source)
    original = source.copy()
    splitter = SpatialTrainTestSplitter(grid_size=2)
    train, test = splitter.split_by_row(source, "label", 0.2)
    repeated_train, repeated_test = splitter.split_by_row(source, "label", 0.2)
    _assert_partition(source, train, test)
    pd.testing.assert_frame_equal(train, repeated_train)
    pd.testing.assert_frame_equal(test, repeated_test)
    pd.testing.assert_frame_equal(source, original)
    assert train.tile.nunique() == test.tile.nunique() == 4


def test_missing_geometry_keeps_rows_without_claiming_area_coverage():
    source = _frame([2, 2, 18, 18], ["rare"] * 2 + ["common"] * 38)
    source.loc[:1, "geometry"] = None
    train, test = SpatialTrainTestSplitter(grid_size=2).split_by_row(source, "label", 0.2)
    _assert_partition(source, train, test)
    assert train.geometry.isna().sum() == test.geometry.isna().sum() == 1
    assert train.loc[train.geometry.notna(), "tile"].nunique() == 3
    assert test.loc[test.geometry.notna(), "tile"].nunique() == 3


def test_singleton_class_is_rejected():
    source = _frame([2, 2], ["rare", "common", "common", "common"])
    with pytest.raises(ValueError, match="at least 2 rows per class"):
        SpatialTrainTestSplitter(grid_size=2).split_by_row(source, "label", 0.2)


@pytest.mark.parametrize("seed", range(5))
def test_joint_coverage_matches_exhaustive_optimum(seed):
    rng = np.random.default_rng(seed)
    target = pd.Series(["A"] * 3 + ["B"] * 5)
    groups = pd.Series(rng.integers(0, 4, 8).astype(str))

    def score(indices):
        mask = np.zeros(8, dtype=bool)
        mask[list(indices)] = True
        coverage = groups[mask].nunique() + groups[~mask].nunique()
        desired = target.value_counts() * mask.mean()
        actual = target[mask].value_counts().reindex(desired.index, fill_value=0)
        return coverage, -float((actual - desired).abs().sum())

    feasible = [indices for indices in combinations(range(8), 3)
                if target.iloc[list(indices)].nunique() == 2]
    assignments = allocate_spatial_rows(target, groups, np.array([5, 3]), seed)
    assert score(np.flatnonzero(assignments == 1)) == max(map(score, feasible))


def test_multiscale_coverage_matches_exhaustive_optimum():
    xy = np.repeat([[0, 0], [0, 1], [1, 0], [1, 1], [7, 7]], 2, axis=0)
    target = pd.Series(list("ABAABBABBB"))
    groups = pd.Series([f"cell_{x}_{y}" for x, y in xy])

    def score(indices):
        mask = np.zeros(10, dtype=bool)
        mask[list(indices)] = True
        coverage = sum(len(set(map(tuple, xy[side] // scale))) for scale in [1, 2, 4] for side in [mask, ~mask])
        desired = target.value_counts() * mask.mean()
        actual = target[mask].value_counts().reindex(desired.index, fill_value=0)
        return coverage, -float((actual - desired).abs().sum())

    feasible = [indices for indices in combinations(range(10), 3) if target.iloc[list(indices)].nunique() == 2]
    assignments = allocate_spatial_rows(target, groups, np.array([7, 3]), 42)
    assert score(np.flatnonzero(assignments == 1)) == max(map(score, feasible))


@pytest.mark.parametrize("test_size", [0.01, 0.99])
def test_two_rows_per_class_are_split_even_at_extreme_ratios(test_size):
    source = _frame([2, 2], ["A", "A", "B", "B"])
    with pytest.warns(UserWarning, match="Adjusted test rows"):
        train, test = SpatialTrainTestSplitter(grid_size=2).split_by_row(source, "label", test_size)
    _assert_partition(source, train, test)
    assert len(train) == len(test) == 2
