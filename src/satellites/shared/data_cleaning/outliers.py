"""Reusable univariate and multivariate outlier detection.

The :class:`OutlierAnalysis` class is the single calculation layer used by the
EDA pipeline and by callers that need row-level outlier flags. All returned
masks retain the input dataframe's index so results can be joined safely.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler


class OutlierAnalysis:
    """Calculate aligned outlier flags and report-ready summaries.

    Args:
        df: Source dataframe. It is copied and is never modified in place.
        iqr_lower_quantile: Lower quantile used to calculate the IQR.
        iqr_upper_quantile: Upper quantile used to calculate the IQR.
        iqr_multiplier: Number of IQRs used to extend each outlier boundary.
        z_score_threshold: Absolute z-score above which a value is an outlier.
        forest_contamination: Expected outlier proportion for Isolation Forest.
        pca_components: Maximum component count for PCA- and PLS-based methods.
        pca_threshold: Upper-tail probability used by the PLS detector.
        random_seed: Seed used by stochastic detection methods.
        max_multivariate_features: Maximum number of columns included in
            multivariate calculations.

    Raises:
        TypeError: If ``df`` is not a pandas dataframe.
        ValueError: If a threshold, quantile, or component setting is invalid.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        iqr_lower_quantile: float = 0.25,
        iqr_upper_quantile: float = 0.75,
        iqr_multiplier: float = 1.5,
        z_score_threshold: float = 3.0,
        forest_contamination: float = 0.05,
        pca_components: int = 2,
        pca_threshold: float = 0.05,
        random_seed: int = 42,
        max_multivariate_features: int = 8,
    ) -> None:
        """Initialize validated outlier settings and defensive data copies."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame.")
        if not 0 <= iqr_lower_quantile < iqr_upper_quantile <= 1:
            raise ValueError("IQR quantiles must satisfy 0 <= lower < upper <= 1.")
        if iqr_multiplier <= 0 or z_score_threshold <= 0:
            raise ValueError("IQR multiplier and z-score threshold must be > 0.")
        if not 0 < forest_contamination <= 0.5:
            raise ValueError("forest_contamination must be in (0, 0.5].")
        if pca_components < 1 or max_multivariate_features < 2:
            raise ValueError("PCA components must be >= 1 and max features >= 2.")
        if not 0 < pca_threshold < 1:
            raise ValueError("pca_threshold must be in (0, 1).")

        # Work on copies so detection never adds flags to the caller's dataframe.
        self.df = df.copy()
        self.results = df.copy()
        self.IQR_lower_quantile = iqr_lower_quantile
        self.IQR_higher_quantile = iqr_upper_quantile
        self.IQR_multiplier = iqr_multiplier
        self.z_score_threshold = z_score_threshold
        self.forest_contamination = forest_contamination
        self.PCA_components = pca_components
        self.PCA_threshold = pca_threshold
        self.random_seed = random_seed
        self.max_multivariate_features = max_multivariate_features
        self.numeric_columns = self._numeric_columns()
        self.figsize = (20, 6)

        # Note: `self.df` and `self.results` intentionally keep the same index
        # as the caller's input so any boolean masks produced by methods can be
        # aligned back to the original rows without reindexing.

    def _numeric_columns(self) -> list[str]:
        """Return numeric, non-geometry columns in their original order."""
        # Select numeric dtypes but exclude a column named 'geometry' which is
        # commonly present in geospatial tables and not a numeric feature.
        return [column for column in self.df.select_dtypes(include=np.number).columns if column != "geometry"]

    def _clean_numeric(self, column: str) -> pd.Series:
        """Return finite numeric values while preserving their source index.

        Raises:
            KeyError: If ``column`` does not exist in the source dataframe.
        """
        if column not in self.df.columns:
            raise KeyError(f"Column not found: {column}")
        # Coerce non-numeric strings to NaN so calculations operate on floats.
        series = pd.to_numeric(self.df[column], errors="coerce")
        # Remove NaN and infinite values since they cannot participate safely in
        # quantiles, standard deviation, or distance computations.
        return series[series.notna() & np.isfinite(series)]

    def check_and_standardize_data(self) -> pd.DataFrame:
        """Replace numeric infinities with missing values.

        Returns:
            A defensive dataframe copy with positive and negative infinity
            replaced by ``NaN`` in numeric columns.
        """
        # Use a defensive copy and convert any infinite numeric values to NaN so
        # subsequent detectors treat them as missing rather than producing
        # arithmetic errors.
        self.df = self.df.copy()
        self.numeric_columns = self._numeric_columns()
        self.df[self.numeric_columns] = self.df[self.numeric_columns].replace([np.inf, -np.inf], np.nan)
        # Reset the results container because previous detection columns may
        # be stale after standardization.
        self.results = self.df.copy()
        return self.df

    def calculate_iqr_bounds(self, column: str) -> tuple[float, float]:
        """Calculate lower and upper IQR fences for one numeric column.

        Args:
            column: Column whose finite values should define the bounds.

        Returns:
            ``(lower_bound, upper_bound)``. Both values are ``NaN`` when the
            column has no finite observations.
        """
        # Compute quartiles and the interquartile range using only finite
        # numeric observations. If no finite values exist return NaNs to
        # indicate bounds are undefined.
        clean = self._clean_numeric(column)
        if clean.empty:
            return np.nan, np.nan
        q1 = clean.quantile(self.IQR_lower_quantile)
        q3 = clean.quantile(self.IQR_higher_quantile)
        iqr = q3 - q1
        return (float(q1 - self.IQR_multiplier * iqr), float(q3 + self.IQR_multiplier * iqr))

    def iqr_outlier_mask(self, column: str) -> pd.Series:
        """Return an IQR outlier mask aligned to every input row.

        Null and infinite observations are represented by ``False`` because
        they are missing/invalid values rather than detected outliers.
        """
        # Build a boolean mask aligned to the original dataframe index. Missing
        # or infinite values are treated as non-outliers (False) because they
        # represent absent data rather than detected anomalies.
        clean = self._clean_numeric(column)
        if clean.empty:
            return pd.Series(False, index=self.df.index, dtype=bool)
        lower, upper = self.calculate_iqr_bounds(column)
        numeric = pd.to_numeric(self.df[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        values = ((numeric < lower) | (numeric > upper)).fillna(False).to_numpy(dtype=bool)
        return pd.Series(values, index=self.df.index, dtype=bool)

    def zscore_outlier_mask(self, column: str) -> pd.Series:
        """Return an aligned absolute z-score outlier mask.

        The sample standard deviation (``ddof=1``, pandas' default) matches the
        historical EDA calculation. Constant and undersized columns yield an
        all-false mask.
        """
        # Use the sample standard deviation consistent with historical EDA
        # calculations. Columns with zero or undefined variance cannot produce
        # meaningful z-scores and therefore yield an all-False mask.
        clean = self._clean_numeric(column)
        standard_deviation = clean.std()
        if clean.empty or pd.isna(standard_deviation) or standard_deviation == 0:
            return pd.Series(False, index=self.df.index, dtype=bool)
        numeric = pd.to_numeric(self.df[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        absolute_z_scores = (numeric - clean.mean()).abs() / standard_deviation
        values = absolute_z_scores.gt(self.z_score_threshold).fillna(False).to_numpy(dtype=bool)
        return pd.Series(values, index=self.df.index, dtype=bool)

    def summarize_univariate(self, columns: Sequence[str] | None = None) -> pd.DataFrame:
        """Summarize IQR and z-score outliers for numeric columns.

        Args:
            columns: Optional ordered subset of columns. All detected numeric
                columns are analyzed when omitted.

        Returns:
            One row per column containing observation counts, IQR bounds, and
            outlier counts and percentages. The schema is stable even when no
            column contains finite values.
        """
        # Build a summary dataframe with one row per numeric feature that
        # includes counts, bounds, and outlier statistics. We align masks by
        # position so that duplicate index labels (if present) do not inflate
        # counts.
        rows = []
        for column in columns if columns is not None else self.numeric_columns:
            clean = self._clean_numeric(column)
            if clean.empty:
                continue
            lower, upper = self.calculate_iqr_bounds(column)
            numeric = pd.to_numeric(self.df[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
            valid_positions = numeric.notna().to_numpy()
            iqr_mask = self.iqr_outlier_mask(column).to_numpy()[valid_positions]
            zscore_mask = self.zscore_outlier_mask(column).to_numpy()[valid_positions]
            rows.append(
                {
                    "column": column,
                    "count": int(len(clean)),
                    "iqr_lower_bound": self._round(lower),
                    "iqr_upper_bound": self._round(upper),
                    "iqr_outlier_count": int(iqr_mask.sum()),
                    "iqr_outlier_percent": self._percent(iqr_mask.sum(), len(clean)),
                    "zscore_threshold": self.z_score_threshold,
                    "zscore_outlier_count": int(zscore_mask.sum()),
                    "zscore_outlier_percent": self._percent(zscore_mask.sum(), len(clean)),
                }
            )
        return pd.DataFrame(rows, columns=self.univariate_summary_columns())

    @staticmethod
    def univariate_summary_columns() -> list[str]:
        """Return the stable column order for univariate summary artifacts."""
        return [
            "column",
            "count",
            "iqr_lower_bound",
            "iqr_upper_bound",
            "iqr_outlier_count",
            "iqr_outlier_percent",
            "zscore_threshold",
            "zscore_outlier_count",
            "zscore_outlier_percent",
        ]

    def prepared_numeric_matrix(self, columns: Sequence[str] | None = None) -> pd.DataFrame:
        """Prepare a standardized matrix for multivariate detection.

        Numeric infinities are treated as missing, all-null and constant
        features are removed, remaining nulls are median-imputed, and each
        retained feature is standardized with population variance.

        Args:
            columns: Optional ordered feature subset. The configured maximum
                feature count is applied after selection.

        Returns:
            A numeric dataframe retaining the input row index.
        """
        # Select features in the requested order but cap to the configured
        # maximum to avoid excessively large covariance matrices.
        selected = list(columns if columns is not None else self.numeric_columns)[: self.max_multivariate_features]
        if not selected:
            return pd.DataFrame(index=self.df.index)

        # Coerce to numeric, drop features that are entirely null, median-impute
        # remaining nulls, and remove constant features which would make the
        # covariance singular.
        numeric_df = self.df[selected].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        numeric_df = numeric_df.dropna(axis=1, how="all")
        if numeric_df.empty:
            return numeric_df
        numeric_df = numeric_df.fillna(numeric_df.median(numeric_only=True))
        numeric_df = numeric_df.loc[:, numeric_df.std(ddof=0) > 0]
        if numeric_df.empty:
            return numeric_df
        # Standardize using population variance to keep scale consistent with
        # other pipeline components.
        return (numeric_df - numeric_df.mean()) / numeric_df.std(ddof=0)

    def detect_multivariate_mahalanobis(
        self, columns: Sequence[str] | None = None, *, confidence: float = 0.975, limit: int | None = 100
    ) -> pd.DataFrame:
        """Rank rows by multivariate Mahalanobis distance.

        Args:
            columns: Optional ordered feature subset.
            confidence: Chi-square confidence level used for the outlier flag.
            limit: Maximum number of highest-distance rows returned. ``None``
                returns all rows.

        Returns:
            Rows ordered by descending squared distance, with chi-square
            p-values and an outlier flag. An empty, schema-stable dataframe is
            returned when there are too few rows or usable features.

        Raises:
            ValueError: If ``confidence`` is not strictly between zero and one.
        """
        if not 0 < confidence < 1:
            raise ValueError("confidence must be in (0, 1).")
        # Prepare a standardized numeric matrix and compute Mahalanobis
        # distances. The output includes p-values and a chi-square based
        # outlier flag for interpretability.
        numeric_df = self.prepared_numeric_matrix(columns)
        output_columns = ["row_index", "mahalanobis_distance", "mahalanobis_distance_squared", "pvalue", "is_outlier_97_5pct"]
        # Covariance needs at least two features and more rows than dimensions.
        if numeric_df.shape[1] < 2 or numeric_df.shape[0] < numeric_df.shape[1] + 2:
            return pd.DataFrame(columns=output_columns)

        # Use a pseudo-inverse for numerical stability when features are
        # correlated or the covariance matrix is rank-deficient.
        values = numeric_df.to_numpy(dtype=float)
        covariance = np.cov(values, rowvar=False)
        inverse_covariance = np.linalg.pinv(covariance)
        difference = values - values.mean(axis=0)
        distances_squared = np.einsum("ij,jk,ik->i", difference, inverse_covariance, difference)
        degrees_of_freedom = numeric_df.shape[1]
        threshold = stats.chi2.ppf(confidence, df=degrees_of_freedom)
        result = pd.DataFrame(
            {
                "row_index": numeric_df.index,
                # Guard against tiny negative values from floating-point error.
                "mahalanobis_distance": np.sqrt(np.maximum(distances_squared, 0)),
                "mahalanobis_distance_squared": distances_squared,
                "pvalue": 1 - stats.chi2.cdf(distances_squared, df=degrees_of_freedom),
                # Name retained for compatibility with the EDA artifact schema.
                "is_outlier_97_5pct": distances_squared > threshold,
            }
        ).sort_values("mahalanobis_distance_squared", ascending=False)
        if limit is not None:
            result = result.head(limit)
        return result.reset_index(drop=True)

    def detect_iqr_outliers(self, column: str) -> pd.Series:
        """Calculate and store an aligned IQR flag column.

        The generated column is named ``IQR_outliers_<column>``.
        """
        mask = self.iqr_outlier_mask(column)
        self.results[f"IQR_outliers_{column}"] = mask
        return mask

    def detect_zscore_outliers(self, column: str) -> pd.Series:
        """Calculate and store an aligned z-score flag column.

        The generated column is named ``Zscore_outliers_<column>``.
        """
        mask = self.zscore_outlier_mask(column)
        self.results[f"Zscore_outliers_{column}"] = mask
        return mask

    def fill_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        """KNN-impute a numeric dataframe while retaining index and columns.

        Up to three neighboring observations are used; smaller inputs
        automatically reduce the neighbor count.
        """
        # Perform a simple KNN-imputation that preserves row/column labels.
        if df.empty:
            return df.copy()
        values = KNNImputer(n_neighbors=min(3, len(df))).fit_transform(df)
        return pd.DataFrame(values, columns=df.columns, index=df.index)

    def detect_isolation_forest_outliers(self, column: str) -> pd.Series:
        """Calculate and store univariate Isolation Forest outlier flags.

        Finite values are standardized before fitting. Null and infinite rows
        remain ``False`` in the aligned result.
        """
        # Isolation Forest is applied to the finite subset of values; rows
        # that were missing remain False in the aligned mask.
        clean = self._clean_numeric(column)
        mask = pd.Series(False, index=self.df.index, dtype=bool)
        if len(clean) >= 2:
            scaled = StandardScaler().fit_transform(clean.to_numpy().reshape(-1, 1))
            predictions = IsolationForest(contamination=self.forest_contamination, random_state=self.random_seed).fit_predict(
                scaled
            )
            # Isolation Forest uses -1 for anomalies.
            mask.loc[clean.index] = predictions == -1
        self.results[f"IForest_outliers_{column}"] = mask
        return mask

    def detect_pca_outliers(self, column: str) -> pd.Series:
        """Calculate and store PCA/DBSCAN multivariate outlier flags.

        PCA reduces the prepared numeric matrix before DBSCAN labels sparse
        observations. The same multivariate labels are stored for each analyzed
        ``column`` to preserve the class's established result-column API.
        """
        # Reduce dimensionality with PCA and apply DBSCAN to identify points
        # that do not belong to dense clusters. The same mask is stored for
        # each requested `column` for compatibility with callers expecting
        # per-column result columns.
        matrix = self.prepared_numeric_matrix(self.numeric_columns)
        mask = pd.Series(False, index=self.df.index, dtype=bool)
        components = min(self.PCA_components, matrix.shape[0], matrix.shape[1])
        if components and len(matrix) >= 5:
            transformed = PCA(n_components=components).fit_transform(matrix)
            labels = DBSCAN(eps=0.3, min_samples=5).fit_predict(transformed)
            # DBSCAN labels noise as -1.
            mask.loc[matrix.index] = labels == -1
        self.results[f"PCA_outliers_{column}"] = mask
        return mask

    def detect_outliers_pls(self, column: str) -> pd.Series:
        """Calculate and store PLS score and reconstruction outlier flags.

        A row is flagged when its Hotelling-style score statistic, score
        distance, or orthogonal reconstruction distance exceeds its configured
        upper-tail cutoff.
        """
        # Use PLS to obtain score-space metrics and flag rows that exceed
        # various upper-tail cutoffs. This combines Hotelling-style and
        # reconstruction-distance criteria to detect multivariate anomalies.
        matrix = self.prepared_numeric_matrix(self.numeric_columns)
        mask = pd.Series(False, index=self.df.index, dtype=bool)
        components = min(self.PCA_components, matrix.shape[1], max(len(matrix) - 1, 0))
        if components and len(matrix) >= 3:
            values = matrix.to_numpy()
            pls = PLSRegression(n_components=components, scale=False)
            scores, _ = pls.fit_transform(values, np.zeros(len(values)))
            reconstructed = pls.inverse_transform(scores)

            # Use robust fallbacks to avoid division by zero when a score
            # component has no variation.
            score_std = np.std(scores, axis=0)
            score_std = np.where(score_std == 0, 1e-8, score_std)
            t_squared = np.sum((scores / score_std) ** 2, axis=1)
            score_distance = np.sqrt(np.sum(scores**2, axis=1))
            orthogonal_distance = np.sqrt(np.sum((values - reconstructed) ** 2, axis=1))
            percentile = 100 * (1 - self.PCA_threshold)
            combined = (
                (t_squared > stats.chi2.ppf(1 - self.PCA_threshold, df=components))
                | (score_distance > np.percentile(score_distance, percentile))
                | (orthogonal_distance > np.percentile(orthogonal_distance, percentile))
            )
            mask.loc[matrix.index] = combined
        self.results[f"PLS_outliers_{column}"] = mask
        return mask

    def detect_all_outliers(self, *, plot: bool = False) -> pd.DataFrame:
        """Run every detector for every numeric column.

        Args:
            plot: Reserved for backward compatibility. Detection is calculation
                only; plotting is handled separately by the EDA visual layer.

        Returns:
            A copy of the input dataframe with one boolean result column per
            detector and numeric source column.
        """
        self.check_and_standardize_data()
        for column in self.numeric_columns:
            self.detect_iqr_outliers(column)
            self.detect_zscore_outliers(column)
            self.detect_isolation_forest_outliers(column)
            self.detect_pca_outliers(column)
            self.detect_outliers_pls(column)
        return self.results

    @staticmethod
    def _round(value: float, digits: int = 6) -> float:
        """Round a finite scalar while preserving missing values."""
        return round(float(value), digits) if pd.notna(value) else np.nan

    @staticmethod
    def _percent(numerator: float, denominator: float) -> float:
        """Return a four-decimal percentage with safe zero division."""
        if not denominator:
            return 0.0
        return round(100 * float(numerator) / float(denominator), 4)
