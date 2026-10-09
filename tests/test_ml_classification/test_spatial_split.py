"""Coverage and class guarantees for row-wise spatial holdouts."""

from itertools import combinations, product
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from ml_classification.step_02_prepare.libraries.spatial_allocation import (
    _joint_coverage_allocation,
    _minimum_class_deviation,
    allocate_spatial_rows,
)
from ml_classification.step_02_prepare.libraries.spatial_split import SpatialTrainTestSplitter
from tests.utils import expect_equal, expect_in, expect_true

pytestmark = pytest.mark.filterwarnings("error::DeprecationWarning")


def _frame(sizes, labels=None):
    points = []
    for cell, size in enumerate(sizes):
        points.extend(Point((cell // 2) * 10 + row * 0.001, (cell % 2) * 10 + row * 0.001) for row in range(size))
    return gpd.GeoDataFrame(
        {"row_id": np.arange(sum(sizes)), "label": ["A"] * sum(sizes) if labels is None else labels,
         "tile": np.repeat(np.arange(len(sizes)), sizes)}, geometry=points, crs="EPSG:32634",
    )


def _assert_partition(source, train, test):
    expect_true(set(train.row_id).isdisjoint(test.row_id))
    expect_equal(sorted(train.row_id.tolist() + test.row_id.tolist()), source.row_id.tolist())
    expect_true(set(train.label) == set(test.label) == set(source.label))
    expect_true(train.crs == test.crs == source.crs)


def test_holdout_covers_sparse_cells_when_possible():
    source = _frame([2, 2, 18, 18])
    train, test = SpatialTrainTestSplitter(grid_size=2).split(source, "label", 0.2, method="by_row")
    _assert_partition(source, train, test)
    expect_equal((len(train), len(test)), (32, 8))
    expect_true(train.tile.nunique() == test.tile.nunique() == 4)


def test_class_presence_overrides_requested_size():
    source = _frame([2, 2, 18, 18], ["rare"] * 2 + ["common"] * 38)
    with pytest.warns(UserWarning, match="Adjusted test rows"):
        train, test = SpatialTrainTestSplitter(grid_size=2).split_by_row(source, "label", 0.01)
    _assert_partition(source, train, test)
    expect_equal(len(test), 2)


def test_coverage_takes_priority_over_class_proportions():
    # Reaching all four cells needs three rare-class test rows, although a
    # strictly proportional split would spend nearly every slot on the common class.
    source = _frame([2, 2, 2, 94], ["rare"] * 6 + ["common"] * 94)
    train, test = SpatialTrainTestSplitter(grid_size=2).split_by_row(source, "label", 0.04)
    _assert_partition(source, train, test)
    expect_true(train.tile.nunique() == test.tile.nunique() == 4)
    expect_equal(test.label.value_counts().to_dict(), {"rare": 3, "common": 1})


def test_sparse_singletons_do_not_prevent_class_complete_split():
    source = _frame([1, 1, 18, 20], ["rare"] * 2 + ["common"] * 38)
    train, test = SpatialTrainTestSplitter(grid_size=2).split_by_row(source, "label", 0.2)
    _assert_partition(source, train, test)
    expect_true(train.tile.nunique() == test.tile.nunique() == 3)


def test_small_test_budget_spreads_across_distant_regions():
    # Four nearby cells and a distant region: both sets can cover the two broad
    # regions, even though test can only visit two of the five fine cells.
    groups = pd.Series(np.repeat(["cell_0_0", "cell_0_1", "cell_1_0", "cell_1_1", "cell_7_7"], 2))
    assignments = allocate_spatial_rows(pd.Series(["A"] * 10), groups, np.array([8, 2]), 42)
    selected = groups[assignments == 1]
    expect_equal(selected.nunique(), 2)
    expect_in("cell_7_7", set(selected))
    expect_equal(groups[assignments == 0].nunique(), 5)


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
    expect_true(train.tile.nunique() == test.tile.nunique() == 4)


def test_missing_geometry_keeps_rows_without_claiming_area_coverage():
    source = _frame([2, 2, 18, 18], ["rare"] * 2 + ["common"] * 38)
    source.loc[:1, "geometry"] = None
    train, test = SpatialTrainTestSplitter(grid_size=2).split_by_row(source, "label", 0.2)
    _assert_partition(source, train, test)
    expect_true(train.geometry.isna().sum() == test.geometry.isna().sum() == 1)
    expect_equal(train.loc[train.geometry.notna(), "tile"].nunique(), 3)
    expect_equal(test.loc[test.geometry.notna(), "tile"].nunique(), 3)


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
    expect_equal(score(np.flatnonzero(assignments == 1)), max(map(score, feasible)))


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
    expect_equal(score(np.flatnonzero(assignments == 1)), max(map(score, feasible)))


@pytest.mark.parametrize("test_size", [0.01, 0.99])
def test_two_rows_per_class_are_split_even_at_extreme_ratios(test_size):
    source = _frame([2, 2], ["A", "A", "B", "B"])
    with pytest.warns(UserWarning, match="Adjusted test rows"):
        train, test = SpatialTrainTestSplitter(grid_size=2).split_by_row(source, "label", test_size)
    _assert_partition(source, train, test)
    expect_true(len(train) == len(test) == 2)


@pytest.mark.parametrize("count, sizes, minimum, maximum", [
    (6, [1, 1, 1], 1, 6),  # Exact integer targets.
    (5, [1, 1, 1], 1, 5),  # Round upwards.
    (3, [8, 1, 1], 1, 3),  # Minimum support forces rounding down elsewhere.
    (2, [8, 1, 1], 0, 1),  # Rare classes occupy distinct folds.
])
def test_minimum_class_deviation_matches_exhaustive_counts(count, sizes, minimum, maximum):
    """Check the relaxation bound against all feasible integer class counts."""
    desired = count * np.array(sizes) / sum(sizes)
    feasible = (values for values in product(range(minimum, maximum + 1), repeat=len(sizes)) if sum(values) == count)
    expected = min(np.abs(np.array(values) - desired).sum() for values in feasible)
    expect_equal(_minimum_class_deviation(count, desired, minimum, maximum), pytest.approx(expected))


@pytest.mark.parametrize("sizes", [[4, 2], [2, 2, 2]])
@pytest.mark.parametrize("known_regions", [True, False])
def test_joint_solver_helpers_match_exhaustive_coverage_and_balance(sizes, known_regions):
    """Exercise class and coverage helpers through real holdout and CV solves."""
    classes = np.array([0, 0, 1, 1, 1, 1])
    regions = np.array([0, 1, 0, 1, 2, -1]) if known_regions else np.full(6, -1)
    coarse = np.where(regions < 0, -1, regions // 2)
    levels = [regions, coarse]
    class_counts = np.bincount(classes)
    n_parts = len(sizes)

    def score(assignments):
        coverage = 0
        for level in levels:
            for part in range(n_parts):
                coverage += len(set(level[(assignments == part) & (level >= 0)]))
                if n_parts > 2:
                    coverage += len(set(level[(assignments != part) & (level >= 0)]))
        actual = np.array([np.bincount(assignments[classes == label], minlength=n_parts) for label in range(2)])
        desired = class_counts[:, None] * np.array(sizes) / len(classes)
        return coverage, -float(np.abs(actual - desired).sum())

    feasible_scores = []
    for values in product(range(n_parts), repeat=len(classes)):
        assignments = np.array(values)
        if not np.array_equal(np.bincount(assignments, minlength=n_parts), sizes):
            continue
        if any(len(set(assignments[classes == label])) != min(count, n_parts) for label, count in enumerate(class_counts)):
            continue
        feasible_scores.append(score(assignments))

    allocations = _joint_coverage_allocation(np.ones(6, dtype=int), classes, np.arange(6), class_counts, levels, np.array(sizes))
    np.testing.assert_array_equal(allocations.sum(axis=1), np.ones(6))
    np.testing.assert_array_equal(allocations.sum(axis=0), sizes)
    expect_true(np.all(allocations >= 0))
    actual_coverage, actual_balance = score(allocations.argmax(axis=1))
    expected_coverage, expected_balance = max(feasible_scores)
    expect_equal(actual_coverage, expected_coverage)
    expect_equal(actual_balance, pytest.approx(expected_balance))


def test_joint_solver_reports_infeasible_class_support():
    """Preserving both classes is impossible in a one-row partition."""
    with pytest.raises(ValueError, match="required class coverage and sizes"):
        _joint_coverage_allocation(np.array([2, 2]), np.array([0, 1]), np.array([0, 0]),
                                   np.array([2, 2]), [np.array([0])], np.array([3, 1]))


@pytest.mark.parametrize("values, message", [
    ([-1, 3, 2, 0], "source row counts"),
    ([1, 0, 1, 1], "source row counts"),
    ([2, 0, 1, 1], "partition sizes"),
])
def test_joint_solver_rejects_invalid_results(monkeypatch, values, message):
    """Do not materialize a solver result that violates allocation totals."""
    monkeypatch.setattr(
        "ml_classification.step_02_prepare.libraries.spatial_allocation.milp",
        lambda *args, **kwargs: SimpleNamespace(success=True, x=np.array(values)),
    )
    with pytest.raises(RuntimeError, match=message):
        _joint_coverage_allocation(np.array([2, 2]), np.array([0, 1]), np.array([0, 0]),
                                   np.array([2, 2]), [np.array([0])], np.array([2, 2]))
