"""Spatially aware train/test splitting utilities based on polygon grid intersections."""

from __future__ import annotations

from dataclasses import dataclass
import warnings
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from shapely.geometry import box


@dataclass(slots=True)
class SpatialTrainTestSplitter:
    """Create reproducible, spatially disjoint holdouts and CV group labels."""

    random_state: int = 42
    grid_size: int = 10

    def split(
        self,
        labeled_df: pd.DataFrame,
        target_column: str,
        test_size: float,
        method: str = "by_group",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Apply the configured row-wise or spatial-group holdout strategy."""
        split_methods = {
            "by_group": self.split_by_group,
            "by_row": self.split_by_row,
        }
        try:
            split_method = split_methods[method]
        except KeyError as exc:
            raise ValueError("Spatial split method must be either 'by_group' or 'by_row'.") from exc
        return split_method(
            labeled_df=labeled_df,
            target_column=target_column,
            test_size=test_size,
        )

    def split_by_group(
        self,
        labeled_df: pd.DataFrame,
        target_column: str,
        test_size: float,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Hold out complete spatial grid cells while approximately stratifying labels."""
        # Validate basic split constraints.
        if not 0.0 < float(test_size) < 1.0:
            raise ValueError("test_size must be strictly between 0 and 1 for spatial split.")
        if "geometry" not in labeled_df.columns:
            raise ValueError("spatial_split=True requires a 'geometry' column.")

        # Validate geometry availability and CRS suitability.
        geometry = labeled_df["geometry"]
        if geometry.isna().all():
            raise ValueError("spatial_split=True requires non-null geometries.")
        crs = getattr(geometry, "crs", None)
        if crs is not None and getattr(crs, "is_geographic", False):
            warnings.warn(
                "Spatial split is using geometry centroids in a geographic CRS. "
                "Consider projecting to a metric CRS for more reliable spatial gridding.",
                stacklevel=2,
            )

        # Treat each occupied grid cell as an indivisible group to prevent spatial leakage.
        groups = self.build_spatial_groups(labeled_df)
        unique_group_count = int(groups.nunique())
        # Approximate the requested test fraction with one held-out fold (for example, 0.2 -> 5 folds).
        requested_splits = max(2, int(round(1.0 / float(test_size))))
        # Never request more folds than there are independently assignable spatial groups.
        n_splits = min(requested_splits, unique_group_count)
        if n_splits < 2:
            raise ValueError("Spatial holdout requires at least two occupied grid cells.")

        # Generate several grouped candidates, then select the one closest to the requested holdout.
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        target = labeled_df[target_column].astype(str)
        # The full-data distribution is the reference used to measure candidate class drift.
        overall_distribution = target.value_counts(normalize=True)
        all_labels = set(overall_distribution.index)
        candidates: list[tuple[float, np.ndarray, np.ndarray]] = []
        for train_idx, test_idx in splitter.split(labeled_df, target, groups=groups):
            train_labels = set(target.iloc[train_idx])
            test_labels = set(target.iloc[test_idx])
            # Missing classes dominate the score because distribution similarity alone cannot expose them.
            missing_label_penalty = len(all_labels - train_labels) + len(all_labels - test_labels)
            test_distribution = target.iloc[test_idx].value_counts(normalize=True).reindex(overall_distribution.index, fill_value=0.0)
            # Balance label-distribution drift against deviation from the configured test fraction.
            class_distribution_error = float((test_distribution - overall_distribution).abs().mean())
            size_error = abs((len(test_idx) / len(labeled_df)) - float(test_size))
            score = (10.0 * missing_label_penalty) + class_distribution_error + size_error
            candidates.append((score, np.asarray(train_idx), np.asarray(test_idx)))

        if not candidates:
            raise ValueError("Spatial holdout could not produce a valid grouped split.")
        # Select deterministically because the candidate order is fixed by random_state.
        _, train_positions, test_positions = min(candidates, key=lambda item: item[0])
        missing_from_train = all_labels - set(target.iloc[train_positions])
        if missing_from_train:
            raise ValueError(
                "Blocked spatial holdout leaves classes absent from training. "
                "Merge unsupported rare classes, enlarge their spatial coverage, or adjust spatial_split_grid_size. "
                f"Missing classes: {sorted(missing_from_train)}"
            )
        # Materialize independent frames so downstream mutation cannot affect the original dataset.
        train_df = labeled_df.iloc[train_positions].copy().reset_index(drop=True)
        test_df = labeled_df.iloc[test_positions].copy().reset_index(drop=True)

        train_groups = set(groups.iloc[train_positions])
        test_groups = set(groups.iloc[test_positions])
        # Assert the core invariant even though the grouped splitter should already guarantee it.
        overlap = train_groups.intersection(test_groups)
        if overlap:
            raise RuntimeError(f"Spatial holdout leaked {len(overlap)} grid cells across train and test.")
        return train_df, test_df


    def split_by_row(
        self,
        labeled_df: pd.DataFrame,
        target_column: str,
        test_size: float,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split labeled data into spatially uniform train/test sets per class."""
        # Validate basic split constraints.
        if not 0.0 < float(test_size) < 1.0:
            raise ValueError("test_size must be strictly between 0 and 1 for spatial split.")
        if "geometry" not in labeled_df.columns:
            raise ValueError("spatial_split=True requires a 'geometry' column.")

        # Validate geometry availability and CRS suitability.
        geometry = labeled_df["geometry"]
        if geometry.isna().all():
            raise ValueError("spatial_split=True requires non-null geometries.")
        crs = getattr(geometry, "crs", None)
        if crs is not None and getattr(crs, "is_geographic", False):
            warnings.warn(
                "Spatial split is using geometry centroids in a geographic CRS. "
                "Consider projecting to a metric CRS for more reliable spatial gridding.",
                stacklevel=2,
            )

        # Build class-wise spatial samples for train.
        split_ratio = 1.0 - float(test_size)
        rng = np.random.default_rng(self.random_state)
        train_indices: list[Any] = []

        for label_value, class_df in labeled_df.groupby(target_column):
            class_idx = class_df.index.to_numpy()
            if len(class_idx) < 2:
                raise ValueError(
                    f"spatial_split=True requires at least 2 rows per class. Class '{label_value}' has {len(class_idx)}."
                )

            target_train_count = int(round(split_ratio * len(class_idx)))
            target_train_count = max(1, min(len(class_idx) - 1, target_train_count))
            selected = self._uniform_spatial_sample_indices(class_df, target_train_count, rng)
            train_indices.extend(selected.tolist())

        # Derive test as complement and randomize row order reproducibly.
        train_indices_arr = np.unique(np.asarray(train_indices, dtype=object))
        all_indices_arr = labeled_df.index.to_numpy(dtype=object)
        test_indices_arr = np.setdiff1d(all_indices_arr, train_indices_arr, assume_unique=False)

        if len(test_indices_arr) == 0:
            raise ValueError("Spatial split produced an empty test split. Reduce test_size or adjust data.")

        train_indices_arr = rng.permutation(train_indices_arr)
        test_indices_arr = rng.permutation(test_indices_arr)
        train_df = labeled_df.loc[train_indices_arr].copy().reset_index(drop=True)
        test_df = labeled_df.loc[test_indices_arr].copy().reset_index(drop=True)
        return train_df, test_df


    def build_spatial_groups(self, df: pd.DataFrame) -> pd.Series:
        """Create spatial grid-cell group labels for grouped cross-validation."""
        # Validate geometry input required for spatial grouping.
        if "geometry" not in df.columns:
            raise ValueError("spatial_split=True requires a 'geometry' column for spatial CV groups.")
        # Convert geometries to representative points and map each row to a grid cell.
        rep_points_gdf = self._to_representative_points(df)
        if rep_points_gdf.empty:
            raise ValueError("Could not compute representative points for spatial CV grouping.")
        # Build the grid from this dataset's extent so group IDs remain locally meaningful.
        grid_gdf = self._create_grid_polygons(rep_points_gdf)
        groups = self._assign_points_to_grid(rep_points_gdf, grid_gdf)
        # Preserve original row alignment and provide fallback label.
        groups = groups.reindex(df.index)
        if groups.isna().any():
            groups = groups.fillna("cell_unassigned")
        return groups.astype(str)

    def _uniform_spatial_sample_indices(
        self,
        class_df: pd.DataFrame,
        total_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample class indices proportionally across intersecting spatial grid cells."""
        # Normalize to point representation for robust point-in-cell checks.
        rep_points_gdf = self._to_representative_points(class_df)
        # Retain source indices because callers use them to select rows from the original class frame.
        valid_indices = rep_points_gdf.index.to_numpy(dtype=object)

        if len(valid_indices) == 0:
            raise ValueError("Could not compute valid representative points for spatial split.")

        grid_gdf = self._create_grid_polygons(rep_points_gdf)
        cell_assignments = self._assign_points_to_grid(rep_points_gdf, grid_gdf)
        cell_ids = cell_assignments.to_numpy(dtype=object)

        # Allocate samples per cell proportional to points in that cell.
        sampled: list[Any] = []
        total_points = len(valid_indices)
        for cell in np.unique(cell_ids):
            cell_idx = valid_indices[cell_ids == cell]
            # Guarantee representation for every occupied cell, subject to its available row count.
            sample_size = max(1, int(round(total_samples * len(cell_idx) / total_points)))
            sample_size = min(sample_size, len(cell_idx))
            chosen = rng.choice(cell_idx, size=sample_size, replace=False)
            sampled.extend(chosen.tolist())

        # Adjust to exact requested count using deterministic random fill/trim.
        sampled_arr = np.unique(np.asarray(sampled, dtype=object))
        sampled_arr = self._adjust_sample_size(sampled_arr, valid_indices, total_samples, rng)

        # Final fallback to full class index pool when needed.
        if len(sampled_arr) < total_samples:
            all_class_idx = class_df.index.to_numpy(dtype=object)
            sampled_arr = self._adjust_sample_size(sampled_arr, all_class_idx, total_samples, rng)

        return sampled_arr

    @staticmethod
    def _to_representative_points(class_df: pd.DataFrame) -> gpd.GeoDataFrame:
        """Convert geometries to representative points while keeping valid rows only."""
        # Create a GeoDataFrame and drop invalid geometries.
        # Preserve CRS metadata because grid geometry must use the same coordinate system as the input.
        gdf = gpd.GeoDataFrame(class_df.copy(), geometry="geometry", crs=getattr(class_df, "crs", None))
        valid = gdf[gdf.geometry.notna()].copy()
        valid = valid[~valid.geometry.is_empty].copy()
        if valid.empty:
            return valid
        # Use representative points to guarantee points are inside polygons.
        valid.geometry = valid.geometry.representative_point()
        return valid

    def _create_grid_polygons(self, points_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Build a regular polygon grid over point extent and keep intersecting cells."""
        # Extract extent and guard against invalid bounds.
        min_x, min_y, max_x, max_y = points_gdf.total_bounds
        if not np.isfinite([min_x, min_y, max_x, max_y]).all():
            raise ValueError("Invalid geometry bounds for spatial split.")

        # Compute cell size from requested grid dimensions.
        span_x = max(max_x - min_x, 1e-12)
        span_y = max(max_y - min_y, 1e-12)
        # Tiny fallback spans allow points sharing one coordinate to receive valid finite cells.
        dx = span_x / self.grid_size
        dy = span_y / self.grid_size

        polygons = []
        poly_ids = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x0 = min_x + i * dx
                y0 = min_y + j * dy
                x1 = min_x + (i + 1) * dx
                y1 = min_y + (j + 1) * dy
                # Shapely boxes include cell boundaries, which the assignment fallback handles safely.
                polygons.append(box(x0, y0, x1, y1))
                poly_ids.append(f"cell_{i}_{j}")

        # Keep only cells that actually intersect the dataset footprint.
        grid_gdf = gpd.GeoDataFrame({"cell_id": poly_ids}, geometry=polygons, crs=points_gdf.crs)
        # Dropping empty cells reduces the assignment loop from the full grid to occupied regions.
        union_geom = points_gdf.geometry.unary_union
        grid_gdf = grid_gdf[grid_gdf.geometry.intersects(union_geom)].copy().reset_index(drop=True)
        if grid_gdf.empty:
            raise ValueError("Spatial grid creation failed: no intersecting cells found.")
        return grid_gdf

    def _assign_points_to_grid(
        self,
        points_gdf: gpd.GeoDataFrame,
        grid_gdf: gpd.GeoDataFrame,
    ) -> pd.Series:
        """Assign each point to a grid cell via geometric intersection checks."""
        # Initialize assignments and iterate cells until all points are mapped.
        assignments = pd.Series(data=None, index=points_gdf.index, dtype="object")
        remaining = points_gdf.index

        for _, cell in grid_gdf.iterrows():
            if len(remaining) == 0:
                break
            poly = cell.geometry
            remaining_points = points_gdf.loc[remaining]
            # `intersects` includes boundary points that strict `within` would otherwise omit.
            mask = remaining_points.geometry.within(poly) | remaining_points.geometry.intersects(poly)
            matched_idx = remaining_points.index[mask]
            if len(matched_idx) > 0:
                assignments.loc[matched_idx] = cell["cell_id"]
                remaining = remaining.difference(matched_idx)

        # Fallback binning for any points left unassigned due to edge cases.
        if len(remaining) > 0:
            centroids = points_gdf.loc[remaining].geometry
            min_x, min_y, max_x, max_y = points_gdf.total_bounds
            span_x = max(max_x - min_x, 1e-12)
            span_y = max(max_y - min_y, 1e-12)
            # Convert coordinates to normalized grid bins without another spatial join.
            x_scaled = (centroids.x.to_numpy(dtype=float) - min_x) / span_x
            y_scaled = (centroids.y.to_numpy(dtype=float) - min_y) / span_y
            x_bin = np.clip(np.floor(x_scaled * self.grid_size).astype(int), 0, self.grid_size - 1)
            y_bin = np.clip(np.floor(y_scaled * self.grid_size).astype(int), 0, self.grid_size - 1)
            fallback_cell_ids = [f"cell_{x}_{y}" for x, y in zip(x_bin, y_bin)]
            assignments.loc[remaining] = fallback_cell_ids

        return assignments

    @staticmethod
    def _adjust_sample_size(
        sampled_indices: np.ndarray,
        pool_indices: np.ndarray,
        total_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Trim or expand sampled indices to exactly match requested sample count."""
        # Deduplicate and either downsample or top up from remaining pool.
        # Uniqueness ensures the returned count always represents distinct source observations.
        sampled = np.unique(sampled_indices)
        if len(sampled) > total_samples:
            return np.sort(rng.choice(sampled, size=total_samples, replace=False))
        if len(sampled) < total_samples:
            remaining = np.setdiff1d(pool_indices, sampled, assume_unique=False)
            if len(remaining) > 0:
                add_n = min(total_samples - len(sampled), len(remaining))
                extra = rng.choice(remaining, size=add_n, replace=False)
                sampled = np.unique(np.concatenate([sampled, extra]))
        return sampled
