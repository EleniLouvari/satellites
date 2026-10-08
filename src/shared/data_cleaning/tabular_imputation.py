"""KNN and text-field imputation for pandas DataFrames."""

from __future__ import annotations

import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

from ._validation import require_columns


def fill_numerical_with_KNN(
    dataframe: pd.DataFrame, fill_col_name: str, features_for_knn: list[str], n_neighbors: int = 3
) -> pd.DataFrame:
    """Fill one numerical column using distance-weighted KNN imputation."""
    # Validate presence of required columns and parameters, then perform a
    # KNN imputation using distance-weighting so closer neighbors influence
    # the imputed value more strongly.
    require_columns(dataframe, [fill_col_name, *features_for_knn])
    if n_neighbors < 1:
        raise ValueError("Error: n_neighbors must be at least 1.")
    result = dataframe.copy()
    columns = [fill_col_name, *features_for_knn]
    imputed = KNNImputer(n_neighbors=min(n_neighbors, len(result)), weights="distance").fit_transform(result[columns])
    result[fill_col_name] = imputed[:, 0]
    return result


def fill_categorical_with_KNN(
    dataframe: pd.DataFrame, fill_col_name: str, features_for_knn: list[str], n_neighbors: int = 3
) -> pd.DataFrame:
    """Fill one categorical column from complete numerical predictor rows."""
    # For categorical columns we train a distance-weighted KNN classifier on
    # rows where the target is present. The label encoder converts string
    # categories to integers for sklearn. Rows without enough predictor
    # information remain unchanged.
    require_columns(dataframe, [fill_col_name, *features_for_knn])
    if n_neighbors < 1:
        raise ValueError("Error: n_neighbors must be at least 1.")
    result = dataframe.copy()
    training = result.loc[result[fill_col_name].notna(), [fill_col_name, *features_for_knn]].dropna()
    prediction = result.loc[result[fill_col_name].isna(), features_for_knn].dropna()
    if prediction.empty:
        return result
    if training.empty:
        raise ValueError(f"Error: Cannot fill {fill_col_name!r}: no complete labeled rows are available.")

    encoder = LabelEncoder()
    labels = encoder.fit_transform(training[fill_col_name])
    classifier = KNeighborsClassifier(n_neighbors=min(n_neighbors, len(training)), weights="distance")
    classifier.fit(training[features_for_knn], labels)
    result.loc[prediction.index, fill_col_name] = encoder.inverse_transform(classifier.predict(prediction))
    return result


def fill_missing_values_in_text_fields(
    dataframe: pd.DataFrame,
    string_null_values_list: list[str],
    cols_to_skip: list[str] | None = None,
    filling_method: str | None = None,
) -> pd.DataFrame:
    """Normalize text columns and replace configured null-like values."""
    # Normalize text (strip, uppercase) and treat configured null-like
    # strings as missing. Supports three strategies: replace with a constant
    # "UNKNOWN", fill with the column mode, or use neighborhood
    # geospatial imputation for spatially-aware datasets.
    if filling_method not in {None, "mode", "neighborhood"}:
        raise ValueError("Error: filling_method must be None, 'mode', or 'neighborhood'.")
    result = dataframe.copy()
    skipped = set(cols_to_skip or [])
    columns = [column for column in result.select_dtypes(include=["object", "string"]).columns if column not in skipped]
    for column in columns:
        # Normalize values to a comparable form before detecting null-like values.
        normalized = result[column].astype("string").str.strip().str.upper()
        normalized = normalized.replace(string_null_values_list, pd.NA)
        if filling_method is None:
            # Simple fallback string for missing textual data.
            result[column] = normalized.fillna("UNKNOWN")
        elif filling_method == "mode":
            modes = normalized.dropna().mode()
            result[column] = normalized.fillna(modes.iloc[0] if not modes.empty else "UNKNOWN")
        else:
            # Use spatial neighborhood imputation when the table contains
            # geometries and nearby records are meaningful donors.
            from .spatial_interpolation import fill_nulls_using_neighborhood_values

            result[column] = normalized
            result = fill_nulls_using_neighborhood_values(result, column, allow_zeros=True, method="mode")
    return result
