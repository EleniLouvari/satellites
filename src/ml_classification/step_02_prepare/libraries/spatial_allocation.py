"""Joint row allocation with mandatory class support and multiscale area coverage."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import coo_matrix, vstack


def _coverage_regions(groups: pd.Index) -> list[np.ndarray]:
    """Return occupied fine cells and progressively coarser regions of the same grid.

    Unknown geometries have no spatial reward. Non-grid group labels support fine
    coverage only (for example, callers supplying their own tile identifiers).
    """
    labels = pd.Series(groups.astype(str))
    known = labels.ne("cell_unassigned").to_numpy()
    levels = [np.where(known, np.arange(len(groups)), -1)]
    coordinates = labels.str.extract(r"^cell_(\d+)_(\d+)$")
    if coordinates.loc[known].isna().any().any() or not known.any():
        return levels
    xy = coordinates.fillna(0).to_numpy(dtype=int)
    scale = 2
    while scale <= xy[known].max():
        codes, _ = pd.factorize(pd.MultiIndex.from_arrays((xy[:, 0] // scale, xy[:, 1] // scale)))
        if len(np.unique(codes[known])) == 1:
            break
        levels.append(np.where(known, codes, -1))
        scale *= 2
    return levels


def _maximal_coverage_allocation(counts, stratum_class, stratum_cell, class_counts, levels, sizes):
    """Try transportation problems that attain each region's coverage upper bound.

    These bipartite flow constraints have integral LP solutions. Success attains
    the maximum possible coverage at every scale and optimal class balance. A
    sequential choice can make a later flow infeasible; then return None and let
    the joint optimizer handle the problem without relaxing any requirements.
    """
    n_classes, n_parts, n_strata = len(class_counts), len(sizes), len(counts)
    desired = class_counts[:, None] * sizes / sizes.sum()
    minimum = (class_counts >= n_parts).astype(int)[:, None]
    base = np.maximum(np.floor(desired).astype(int), minimum)
    ceiling = np.maximum(np.ceil(desired).astype(int), minimum)
    ceiling = np.where(class_counts[:, None] < n_parts, 1, ceiling)
    base = np.minimum(base, ceiling)
    # Round class quotas jointly so both class totals and partition sizes agree.
    rows = np.repeat(np.arange(n_classes), n_parts)
    parts = np.tile(np.arange(n_parts), n_classes)
    columns = np.arange(n_classes * n_parts)
    quota_matrix = coo_matrix(
        (np.ones(2 * len(columns)), (np.r_[rows, n_classes + parts], np.r_[columns, columns])),
        shape=(n_classes + n_parts, len(columns)),
    ).tocsc()
    residual = np.r_[class_counts - base.sum(axis=1), sizes - base.sum(axis=0)]
    if np.any(residual < 0):
        return None
    quota_result = linprog(
        (np.abs(base + 1 - desired) - np.abs(base - desired)).ravel(),
        A_eq=quota_matrix,
        b_eq=residual,
        bounds=list(zip(np.zeros(base.size), (ceiling - base).ravel())),
        method="highs",
    )
    if not quota_result.success:
        return None
    quotas = base + np.rint(quota_result.x).astype(int).reshape(base.shape)
    columns = np.arange(n_strata)
    class_matrix = coo_matrix((np.ones(n_strata), (stratum_class, columns)), shape=(n_classes, n_strata)).tocsc()
    region_rows, region_columns = [], []
    n_regions = 0
    for level in levels:
        regions = level[stratum_cell]
        for region in np.unique(regions[regions >= 0]):
            selected = np.flatnonzero(regions == region)
            region_rows.extend([n_regions] * len(selected))
            region_columns.extend(selected)
            n_regions += 1
    # Classes feed fine cells, then nested coarser regions. These laminar
    # regional constraints retain the integral transportation structure.
    region_matrix = coo_matrix((np.ones(len(region_rows)), (region_rows, region_columns)), shape=(n_regions, n_strata)).tocsc()
    remaining = counts.copy()
    allocations = np.zeros((n_strata, n_parts), dtype=int)
    for part in range(n_parts - 1):
        region_counts = np.asarray(region_matrix @ remaining).ravel()
        remaining_parts = n_parts - part
        minimum_coverage = (region_counts >= remaining_parts).astype(int)
        maximum_count = np.where(minimum_coverage, region_counts - remaining_parts + 1, 1)
        result = linprog(
            np.zeros(n_strata),
            A_eq=class_matrix,
            b_eq=quotas[:, part],
            A_ub=vstack((region_matrix, -region_matrix), format="csc"),
            b_ub=np.r_[maximum_count, -minimum_coverage],
            bounds=list(zip(np.zeros(n_strata), remaining)),
            method="highs",
        )
        if not result.success or not np.allclose(result.x, np.rint(result.x), atol=1e-6, rtol=0):
            return None
        allocations[:, part] = np.rint(result.x).astype(int)
        remaining -= allocations[:, part]
    allocations[:, -1] = remaining
    covered = (region_matrix @ allocations > 0).sum(axis=1)
    maximum_coverage = np.minimum(region_matrix @ counts, n_parts)
    if (
        np.any(allocations < 0)
        or not np.array_equal(class_matrix @ allocations, quotas)
        or not np.array_equal(allocations.sum(axis=0), sizes)
        or not np.array_equal(covered, maximum_coverage)
    ):
        return None
    return allocations


def _materialize_allocations(strata, allocations, random_state):
    """Choose source positions without changing optimized class/cell counts."""
    rng = np.random.default_rng(random_state)
    assignments = np.empty(len(strata), dtype=int)
    for stratum, positions in pd.Series(np.arange(len(strata))).groupby(strata, sort=False).indices.items():
        positions = rng.permutation(positions)
        offset = 0
        for part, count in enumerate(allocations[stratum]):
            assignments[positions[offset : offset + count]] = part
            offset += count
    return assignments


def allocate_spatial_rows(target: pd.Series, groups: pd.Series, partition_sizes: np.ndarray, random_state: int) -> np.ndarray:
    """Assign every row to one partition, preserving classes before optimizing coverage.

    Optimize counts per occupied (class, cell), rather than per individual row.
    Class presence and partition sizes are hard constraints. The primary objective
    maximizes occupied regions across fine and coarser grids in every partition
    (and its training complement for CV). Absolute class-count deviations break
    ties only. Sparse cells may belong to just one partition.

    With fewer class samples than partitions, place those samples in distinct
    partitions, maximizing validation class support while retaining the class in
    every training complement. At least two samples per class are required.
    Returns positional partition IDs, independently of dataframe index labels.
    """
    sizes = np.asarray(partition_sizes, dtype=int)
    n_rows, n_parts = len(target), len(sizes)
    if len(groups) != n_rows or n_parts < 2 or np.any(sizes < 1) or sizes.sum() != n_rows:
        raise ValueError("Error: Spatial partition sizes must be positive and sum to the number of rows.")
    if target.isna().any() or groups.isna().any():
        raise ValueError("Error: Spatial allocation requires non-null class and cell labels.")

    classes, labels = pd.factorize(target, sort=False)
    cells, cell_labels = pd.factorize(groups.astype(str), sort=False)
    class_counts = np.bincount(classes)
    if np.any(class_counts < 2):
        rare = labels[class_counts < 2].tolist()
        raise ValueError(f"Error: Row-wise spatial splitting requires at least 2 rows per class. Unsupported classes: {rare}")
    if n_parts > 2 and np.any(class_counts < n_parts):
        warnings.warn(
            "Some classes have fewer samples than cv_folds; they cannot appear in every validation fold. "
            "They will appear in as many validation folds as possible and in every training fold. "
            "Reduce cv_folds for class-complete validation folds.",
            stacklevel=2,
        )

    strata, pairs = pd.factorize(pd.MultiIndex.from_arrays((classes, cells)), sort=False)
    stratum_class = pairs.get_level_values(0).to_numpy()
    stratum_cell = pairs.get_level_values(1).to_numpy()
    counts = np.bincount(strata)
    n_strata = len(counts)
    levels = _coverage_regions(cell_labels)
    allocations = _maximal_coverage_allocation(counts, stratum_class, stratum_cell, class_counts, levels, sizes)
    if allocations is not None:
        return _materialize_allocations(strata, allocations, random_state)

    allocation_ids = np.arange(n_strata * n_parts).reshape(n_strata, n_parts)
    # Integer allocation variables; subsequent coverage/deviation variables can
    # stay continuous because integer counts make coverage min(count, 1) integral.
    costs = [0.0] * allocation_ids.size
    upper_bounds = np.repeat(counts, n_parts).astype(float).tolist()
    integrality = [1] * allocation_ids.size
    row_ids, column_ids, coefficients = [], [], []
    lower, upper = [], []

    def variable(cost: float, bound: float = np.inf) -> int:
        index = len(costs)
        costs.append(cost)
        upper_bounds.append(bound)
        integrality.append(0)
        return index

    def constraint(indices, values, minimum=-np.inf, maximum=np.inf):
        row_ids.extend([len(lower)] * len(indices))
        column_ids.extend(indices)
        coefficients.extend(values)
        lower.append(minimum)
        upper.append(maximum)

    # Every source row is used once, and all partition sizes are exact.
    for ids, count in zip(allocation_ids, counts):
        constraint(ids, np.ones(n_parts), count, count)
    for part, size in enumerate(sizes):
        constraint(allocation_ids[:, part], np.ones(n_strata), size, size)

    # Total absolute deviation is at most 2*N, so its weighted contribution
    # stays below one coverage point. Class proportions cannot override coverage.
    deviation_weight = 0.25 / n_rows
    for label, count in enumerate(class_counts):
        selected = np.flatnonzero(stratum_class == label)
        minimum, maximum = (1, count) if count >= n_parts else (0, 1)
        desired_counts = count * sizes / n_rows
        deviations = []
        for part, size in enumerate(sizes):
            ids = allocation_ids[selected, part].tolist()
            constraint(ids, np.ones(len(ids)), minimum, maximum)
            deviation = variable(deviation_weight)
            deviations.append(deviation)
            desired = count * size / n_rows
            constraint(ids + [deviation], [1.0] * len(ids) + [-1.0], maximum=desired)
            constraint(ids + [deviation], [1.0] * len(ids) + [1.0], minimum=desired)

        # Tighten the relaxation with the best integer class balance possible
        # before considering geography. This avoids expensive branching merely
        # to prove that fractional desired counts cannot be attained exactly.
        rounded = np.clip(np.floor(desired_counts).astype(int), minimum, maximum)
        while rounded.sum() != count:
            direction = 1 if rounded.sum() < count else -1
            eligible = rounded < maximum if direction == 1 else rounded > minimum
            penalty = np.abs(rounded + direction - desired_counts) - np.abs(rounded - desired_counts)
            part = int(np.argmin(np.where(eligible, penalty, np.inf)))
            rounded[part] += direction
        constraint(deviations, np.ones(n_parts), minimum=float(np.abs(rounded - desired_counts).sum()))

    # Joint coverage prevents selecting training rows first and leaving the test
    # set spatially concentrated. Coarse regions reward dispersion when few rows
    # are available, instead of treating adjacent and distant cells identically.
    for level in levels:
        region_strata: dict[int, list[int]] = {}
        for stratum, region in enumerate(level[stratum_cell]):
            if region >= 0:
                region_strata.setdefault(int(region), []).append(stratum)
        for selected in region_strata.values():
            region_count = int(counts[selected].sum())
            for part in range(n_parts):
                ids = allocation_ids[selected, part].tolist()
                coverage = variable(-1.0, 1.0)
                constraint(ids + [coverage], [1.0] * len(ids) + [-1.0], minimum=0)
                if n_parts > 2:
                    train_coverage = variable(-1.0, 1.0)
                    constraint(ids + [train_coverage], [1.0] * len(ids) + [1.0], maximum=region_count)

    matrix = coo_matrix((coefficients, (row_ids, column_ids)), shape=(len(lower), len(costs))).tocsc()
    costs = np.asarray(costs)
    lower, upper = np.asarray(lower), np.asarray(upper)
    result = milp(
        costs,
        integrality=np.asarray(integrality),
        bounds=Bounds(np.zeros(len(costs)), np.asarray(upper_bounds)),
        constraints=LinearConstraint(matrix, lower, upper),
        options={"mip_rel_gap": 0.0},
    )
    if not result.success:
        raise ValueError(f"Error: Could not allocate spatial rows with the required class coverage and sizes: {result.message}")
    allocations = np.rint(result.x[: allocation_ids.size]).astype(int).reshape(n_strata, n_parts)
    if np.any(allocations < 0) or not np.array_equal(allocations.sum(axis=1), counts):
        raise RuntimeError("Error: Spatial allocation did not preserve source row counts.")
    if not np.array_equal(allocations.sum(axis=0), sizes):
        raise RuntimeError("Error: Spatial allocation did not preserve partition sizes.")

    # Randomness chooses the actual rows inside each class/cell, never trims away
    # the optimized class or spatial coverage. Positional IDs handle duplicate indices.
    return _materialize_allocations(strata, allocations, random_state)
