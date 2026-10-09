"""Post-statistics feature derivation and ML reshape helpers."""

from __future__ import annotations

import itertools
import math

import geopandas as gpd
import numpy as np
import pandas as pd


class ParcelStatisticsReshapeMixin:
    """Methods for derived features and parcel time-series reshaping."""

    STATIC_RESULT_COLUMNS = (
        "batch_number",
        "intersected_pixel_count",
        "meets_minimum_pixel_count",
        "expected_pixel_count",
        "observed_count",
        "temporal_filled_pixel_count",
        "spatial_filled_pixel_count",
        "temporal_filled_ratio",
        "spatial_filled_ratio",
        "data_reliability_score",
        "pixel_area",
        "geom_area",
        "geom_interior_area_ratio",
        "geom_compactness",
        "geom_perimeter_area_ratio",
        "geom_shape_index",
        "geom_elongation",
        "geom_shape_complexity_score",
    )

    def _add_derived_stats(self, data: pd.DataFrame) -> pd.DataFrame:
        """Append deterministic parcel features in one non-fragmenting operation."""
        metric_parcels = self.parcels.to_crs(epsg=self.working_epsg)
        geom_area = metric_parcels.geometry.area.astype("float64")
        geom_perimeter = metric_parcels.geometry.length.astype("float64")
        positive_area = geom_area.where(geom_area > 0)
        positive_perimeter = geom_perimeter.where(geom_perimeter > 0)
        parcel_metrics = pd.DataFrame(
            {
                self.PARCEL_ID_FIELD: self.parcels[self.PARCEL_ID_FIELD].to_numpy(),
                "geom_area": geom_area.to_numpy(),
                "geom_compactness": (4.0 * np.pi * geom_area / positive_perimeter.pow(2)).to_numpy(),
                "geom_perimeter_area_ratio": (geom_perimeter / positive_area).to_numpy(),
                "geom_shape_index": (geom_perimeter / (2.0 * np.sqrt(np.pi * positive_area))).to_numpy(),
                "geom_elongation": metric_parcels.geometry.map(self._minimum_rotated_rectangle_elongation).to_numpy(),
            }
        )
        enriched = data.merge(parcel_metrics, on=self.PARCEL_ID_FIELD, how="left", validate="many_to_one")
        enriched["geom_interior_area_ratio"] = enriched["pixel_area"] / enriched["geom_area"].where(enriched["geom_area"] > 0)
        enriched["geom_shape_complexity_score"] = self._calculate_geometry_complexity_score(enriched)

        derived_columns: dict[str, pd.Series] = {}

        def add_derived_column(column_name: str, values: pd.Series) -> None:
            if column_name in enriched.columns or column_name in derived_columns:
                return
            derived_columns[column_name] = values

        for variable in self._output_sensor_variables():
            mean = f"{variable}_mean"
            median = f"{variable}_median"
            standard_deviation = f"{variable}_sd"
            minimum = f"{variable}_min"
            maximum = f"{variable}_max"
            p10 = f"{variable}_p10"
            p25 = f"{variable}_p25"
            p75 = f"{variable}_p75"
            p90 = f"{variable}_p90"

            if minimum in enriched and maximum in enriched:
                add_derived_column(f"{variable}_range", enriched[maximum] - enriched[minimum])
            if standard_deviation in enriched:
                add_derived_column(f"{variable}_variance", enriched[standard_deviation] ** 2)
                if mean in enriched:
                    absolute_mean = enriched[mean].abs()
                    add_derived_column(
                        f"{variable}_coefficient_of_variation",
                        enriched[standard_deviation] / absolute_mean.where(absolute_mean > 1e-12),
                    )
            if p25 in enriched and p75 in enriched:
                iqr = enriched[p75] - enriched[p25]
                add_derived_column(f"{variable}_iqr", iqr)
                if median in enriched:
                    add_derived_column(
                        f"{variable}_bowley_skewness",
                        (enriched[p75] + enriched[p25] - 2 * enriched[median]) / iqr.where(iqr.abs() > 1e-12),
                    )
            if p10 in enriched and p90 in enriched:
                add_derived_column(f"{variable}_p90_p10_spread", enriched[p90] - enriched[p10])
        if not derived_columns:
            return enriched
        derived = pd.DataFrame(derived_columns, index=enriched.index)
        return pd.concat([enriched, derived], axis=1)

    @staticmethod
    def _minimum_rotated_rectangle_elongation(geometry) -> float:
        """Return the long/short side ratio of a geometry's rotated rectangle."""
        if geometry is None or geometry.is_empty:
            return np.nan
        rectangle = geometry.minimum_rotated_rectangle
        if rectangle.is_empty or rectangle.geom_type != "Polygon":
            return np.nan
        coordinates = list(rectangle.exterior.coords)
        side_lengths = [math.hypot(x2 - x1, y2 - y1) for (x1, y1), (x2, y2) in itertools.pairwise(coordinates)]
        positive_lengths = [length for length in side_lengths if length > 0]
        if len(positive_lengths) < 2:
            return np.nan
        return float(max(positive_lengths) / min(positive_lengths))

    @staticmethod
    def _calculate_geometry_complexity_score(data: pd.DataFrame) -> pd.Series:
        """Combine raster agreement, boundary irregularity, and elongation."""
        positive_area = data["geom_area"].where(data["geom_area"] > 0)
        interior_ratio = data["geom_interior_area_ratio"].where(data["geom_interior_area_ratio"] > 0)
        compactness = data["geom_compactness"].where(data["geom_compactness"] > 0)
        shape_index = data["geom_shape_index"].where(data["geom_shape_index"] > 0)
        elongation = data["geom_elongation"].where(data["geom_elongation"] > 0)

        raster_agreement_penalty = np.log(interior_ratio).abs()
        compactness_penalty = (-0.5 * np.log(compactness)).clip(lower=0.0)
        equal_area_circle_perimeter_area_ratio = 2.0 * np.sqrt(np.pi / positive_area)
        normalized_perimeter_area_ratio = (data["geom_perimeter_area_ratio"] / equal_area_circle_perimeter_area_ratio).where(
            lambda values: values > 0
        )
        perimeter_penalty = np.log(normalized_perimeter_area_ratio).clip(lower=0.0)
        shape_index_penalty = np.log(shape_index).clip(lower=0.0)
        elongation_penalty = np.log(elongation).clip(lower=0.0)

        boundary_penalty = pd.concat([compactness_penalty, perimeter_penalty, shape_index_penalty], axis=1).mean(
            axis=1, skipna=False
        )
        return pd.concat([raster_agreement_penalty, boundary_penalty, elongation_penalty], axis=1).mean(axis=1, skipna=False)

    def _normalize_reshape_inputs(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize parcel IDs and period labels for temporal reshaping."""
        required = {self.PARCEL_ID_FIELD, "period_start"}
        missing = sorted(required.difference(data.columns))
        if missing:
            raise ValueError(f"Error: Missing columns required for time-series reshaping: {missing}")
        working = data.copy()
        working[self.PARCEL_ID_FIELD] = working[self.PARCEL_ID_FIELD].astype(str)
        parsed_periods = pd.to_datetime(working["period_start"], errors="coerce", utc=True)
        invalid_periods = parsed_periods.isna()
        if invalid_periods.any():
            examples = working.loc[invalid_periods, self.PARCEL_ID_FIELD].head(10).tolist()
            raise ValueError(
                f"Error: 'period_start' contains {int(invalid_periods.sum())} invalid timestamps. Example {self.PARCEL_ID_FIELD} values: {examples}"
            )
        working["_period_label"] = parsed_periods.dt.strftime("%Y%m%d")
        return working

    def _validate_reshape_grain(self, working: pd.DataFrame) -> None:
        """Ensure one row per parcel-period in the source table."""
        duplicate_grain = working.duplicated([self.PARCEL_ID_FIELD, "_period_label"], keep=False)
        if duplicate_grain.any():
            examples = working.loc[duplicate_grain, [self.PARCEL_ID_FIELD, "period_start"]].head(10).to_dict("records")
            raise ValueError(f"Error: Zonal statistics must have at most one row per parcel and period. Duplicate examples: {examples}")

    def _resolve_static_columns_for_reshape(self, working: pd.DataFrame) -> list[str]:
        """Return static columns and verify they are constant per parcel."""
        static_columns = [column for column in self.STATIC_RESULT_COLUMNS if column in working.columns]
        for column in static_columns:
            distinct_counts = working.groupby(self.PARCEL_ID_FIELD, sort=False)[column].nunique(dropna=False)
            inconsistent = distinct_counts[distinct_counts > 1]
            if not inconsistent.empty:
                raise ValueError(
                    f"Error: Static column {column!r} changes across periods for {len(inconsistent)} parcels; examples: {inconsistent.index[:10].tolist()}"
                )
        return static_columns

    def _resolve_temporal_features_for_reshape(self, working: pd.DataFrame, static_columns: list[str]) -> list[str]:
        """Return temporal feature columns that should become dated ML features."""
        excluded = {self.PARCEL_ID_FIELD, "period_start", "period_end", "_period_label", *static_columns}
        temporal_features = [column for column in working.columns if column not in excluded]
        if not temporal_features:
            raise ValueError("Error: No temporal feature columns are available for ML reshaping.")
        return temporal_features

    def _pivot_temporal_features_for_reshape(self, working: pd.DataFrame, temporal_features: list[str]) -> pd.DataFrame:
        """Pivot parcel-period statistics into one row per parcel with dated columns."""
        pivoted = working.pivot(index=self.PARCEL_ID_FIELD, columns="_period_label", values=temporal_features)
        pivoted.columns = [f"{feature}__{period_label}" for feature, period_label in pivoted.columns]
        return pivoted

    def _add_missing_expected_period_columns(self, pivoted: pd.DataFrame, temporal_features: list[str]) -> pd.DataFrame:
        """Pad missing calendar period columns with NaNs to keep partition schemas aligned."""
        _, expected_labels = self._temporal_intervals()
        expected_columns = sorted(
            f"{feature}__{period_label}"
            for feature in temporal_features
            for period_label in [label.replace("-", "") for label in expected_labels]
        )
        missing_columns = [column for column in expected_columns if column not in pivoted.columns]
        if missing_columns:
            self.parcel_logger.warning(
                "Adding %s all-NaN column(s) for calendar period(s) with no valid observations in this partition "
                "(likely 100%% cloud cover). First missing: %s.",
                len(missing_columns),
                missing_columns[0],
            )
            for column in missing_columns:
                pivoted[column] = np.nan
        return pivoted

    def _merge_static_columns_for_reshape(
        self, working: pd.DataFrame, pivoted: pd.DataFrame, static_columns: list[str]
    ) -> pd.DataFrame:
        """Merge validated static columns back into the pivoted table."""
        if not static_columns:
            return pivoted
        static = working[[self.PARCEL_ID_FIELD, *static_columns]].drop_duplicates(self.PARCEL_ID_FIELD)
        return static.merge(pivoted, on=self.PARCEL_ID_FIELD, how="inner", validate="one_to_one")

    def _attach_parcel_attributes_for_reshape(self, pivoted: pd.DataFrame) -> gpd.GeoDataFrame:
        """Attach parcel attributes and geometry after temporal reshaping."""
        parcel_attribute_columns = [column for column in self.parcels.columns if column not in {self.PARCEL_ID_FIELD, "geometry"}]
        collisions = sorted(set(parcel_attribute_columns).intersection(pivoted.columns))
        if collisions:
            raise ValueError(
                f"Error: Parcel attributes collide with generated zonal-statistics columns. Rename these parcel columns before extraction: {collisions}"
            )
        output_parcels = self.parcels.to_crs(epsg=self.working_epsg)
        parcel_attribute_columns = [
            column for column in output_parcels.columns if column not in {self.PARCEL_ID_FIELD, "geometry"}
        ]
        parcel_attributes = output_parcels[[self.PARCEL_ID_FIELD, *parcel_attribute_columns, "geometry"]]
        merged = parcel_attributes.merge(pivoted, on=self.PARCEL_ID_FIELD, how="inner", validate="one_to_one")
        return gpd.GeoDataFrame(merged, geometry="geometry", crs=output_parcels.crs)

    def _reshape_time_series_for_ml(self, data: pd.DataFrame) -> gpd.GeoDataFrame:
        """Pivot parcel-period statistics into one ML-ready row per parcel."""
        working = self._normalize_reshape_inputs(data)
        self._validate_reshape_grain(working)
        static_columns = self._resolve_static_columns_for_reshape(working)
        temporal_features = self._resolve_temporal_features_for_reshape(working, static_columns)
        pivoted = self._pivot_temporal_features_for_reshape(working, temporal_features)
        pivoted = self._add_missing_expected_period_columns(pivoted, temporal_features)
        pivoted = pivoted.sort_index(axis=1).reset_index()
        pivoted = self._merge_static_columns_for_reshape(working, pivoted, static_columns)
        result = self._attach_parcel_attributes_for_reshape(pivoted)
        self.parcel_logger.info(
            "Reshaped %s parcel-period rows into %s ML-ready parcel rows with %s dated features in EPSG:%s.",
            len(data),
            len(result),
            len(pivoted.columns) - len(static_columns) - 1,
            self.working_epsg,
        )
        return result
