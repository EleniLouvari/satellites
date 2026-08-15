"""Helpers for converting parcel-period observations into parcel-level features."""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import pandas as pd

from .config import ClassificationPipelineConfig


# These transformations preserve parcel identity while aggregating observations over time.
def reshape_longitudinal_dataset(
    df: pd.DataFrame,
    config: ClassificationPipelineConfig,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    """Pivot long parcel-period features to exactly one row per parcel.

    The identifier and timestamp are used only to define the table grain and
    feature suffixes. They are never passed to a model as predictors.
    """
    if not config.reshape_time_series:
        return df.copy(), list(config.feature_columns), {
            "reshaped": False,
            "input_rows": int(len(df)),
            "output_entities": int(df[config.id_column].nunique()),
        }

    time_column = config.time_column
    required = [config.id_column, time_column, config.target_column, *config.feature_columns]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns required for longitudinal reshaping: {missing}")

    working = df.copy()
    working[config.id_column] = working[config.id_column].astype(str)
    parsed_time = pd.to_datetime(working[time_column], errors="coerce", utc=True)
    invalid_time = parsed_time.isna()
    if invalid_time.any():
        examples = working.loc[invalid_time, config.id_column].head(10).tolist()
        raise ValueError(
            f"{time_column!r} contains {int(invalid_time.sum())} invalid timestamps. "
            f"Example {config.id_column} values: {examples}"
        )
    working["_model_period"] = parsed_time

    if config.prediction_cutoff:
        cutoff = pd.Timestamp(config.prediction_cutoff)
        if cutoff.tzinfo is None:
            cutoff = cutoff.tz_localize("UTC")
        else:
            cutoff = cutoff.tz_convert("UTC")
        working = working.loc[working["_model_period"] <= cutoff].copy()
        if working.empty:
            raise ValueError(f"No observations remain at prediction_cutoff={config.prediction_cutoff!r}.")

    duplicate_grain = working.duplicated([config.id_column, "_model_period"], keep=False)
    if duplicate_grain.any():
        examples = (
            working.loc[duplicate_grain, [config.id_column, "_model_period"]]
            .head(10)
            .astype(str)
            .to_dict("records")
        )
        raise ValueError(
            "Longitudinal input must have at most one row per parcel and period. "
            f"Duplicate examples: {examples}"
        )

    label_counts = working.groupby(config.id_column, sort=False)[config.target_column].nunique(dropna=True)
    inconsistent = label_counts[label_counts > 1]
    if not inconsistent.empty:
        raise ValueError(
            f"Each parcel must have one stable target label. Found conflicting labels for "
            f"{len(inconsistent)} parcels; examples: {inconsistent.index[:10].tolist()}"
        )

    working["_period_label"] = working["_model_period"].dt.strftime("%Y%m%d")
    pivoted = working.pivot(
        index=config.id_column,
        columns="_period_label",
        values=list(config.feature_columns),
    )
    pivoted.columns = [f"{feature}__{period}" for feature, period in pivoted.columns]
    pivoted = pivoted.sort_index(axis=1)
    model_features = pivoted.columns.tolist()
    pivoted = pivoted.reset_index()

    target_by_parcel = working.groupby(config.id_column, sort=False)[config.target_column].agg(
        lambda values: values.dropna().iloc[0] if values.notna().any() else pd.NA
    )
    result = pivoted.merge(
        target_by_parcel.rename(config.target_column),
        left_on=config.id_column,
        right_index=True,
        how="left",
        validate="one_to_one",
    )

    if "geometry" in working.columns:
        geometry_by_parcel = working.groupby(config.id_column, sort=False)["geometry"].first()
        result = result.merge(
            geometry_by_parcel.rename("geometry"),
            left_on=config.id_column,
            right_index=True,
            how="left",
            validate="one_to_one",
        )
        if isinstance(df, gpd.GeoDataFrame):
            result = gpd.GeoDataFrame(result, geometry="geometry", crs=df.crs)

    metadata = {
        "reshaped": True,
        "input_rows": int(len(df)),
        "rows_after_cutoff": int(len(working)),
        "output_entities": int(len(result)),
        "periods": sorted(working["_period_label"].unique().tolist()),
        "source_features": list(config.feature_columns),
        "generated_features": int(len(model_features)),
        "prediction_cutoff": config.prediction_cutoff,
    }
    return result, model_features, metadata
