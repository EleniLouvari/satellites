# Data Analysis

Reusable exploratory analysis, missing-value filling, and outlier-detection
tools for pandas DataFrames and GeoPandas GeoDataFrames.

## Package layout

- `eda/` provides the main end-to-end workflow. It profiles a dataset, runs
  statistical and data-quality checks, creates plots, and writes an HTML report.
- `fill_nulls/` contains tabular and spatial imputation helpers, including KNN,
  nearest-neighbor, inverse-distance weighting, neighborhood aggregation, and
  ordinary kriging.
- `outliers/` exposes `OutlierAnalysis`, the shared calculation layer for
  univariate and multivariate outlier detection. The EDA pipeline also uses
  this layer internally.

Run examples from an environment where the repository's `src` directory is on
`PYTHONPATH`.

## Exploratory data analysis

The EDA pipeline accepts a pandas DataFrame or GeoPandas GeoDataFrame and
returns its generated artifacts and their paths.

```python
from data_analysis.eda import EDAConfig, EDAPipeline

config = EDAConfig(
    output_dir="src/analysis_outputs/parcels",
    target_column="class",
    id_column="parcel_id",
    report_title="Parcel Data Analysis",
)

result = EDAPipeline(config).run(df)
print(result["report_path"])
```

For a shorter one-call interface:

```python
from data_analysis.eda import run_eda

result = run_eda(
    df,
    output_dir="src/analysis_outputs/parcels",
    target_column="class",
    report_title="Parcel Data Analysis",
)
```

The returned dictionary contains:

- `summary`: high-level dataset metadata;
- `artifacts`: all in-memory summary tables and matrices;
- `plot_paths`: generated plot paths;
- `saved_paths`: persisted JSON and CSV paths; and
- `report_path`: the HTML path, or `None` when HTML output is disabled.

### What the EDA covers

The analysis adapts to the columns present and includes, where applicable:

- column types, uniqueness, missingness, constants, and data-quality flags;
- conventional and robust numeric summaries;
- categorical and datetime summaries;
- normality tests and univariate outlier counts;
- Pearson and Spearman correlations and strong-correlation pairs;
- variance inflation factors for multicollinearity;
- missingness relationships and categorical associations;
- numeric or categorical target tests;
- PCA and multivariate Mahalanobis outliers; and
- geometry validity, bounds, coordinate-system information, and geospatial
  plots for GeoDataFrames.

The report and plots are descriptive diagnostics. Statistical significance and
outlier flags should be interpreted in the context of the dataset rather than
treated as automatic instructions to remove data.

### Common EDA options

- `include_html_report` and `include_plots` enable or disable presentation
  artifacts.
- `include_geospatial` controls geometry-specific analysis.
- `correlation_threshold` defines a strong correlation (default `0.75`).
- `high_missing_percent_threshold` sets the high-missingness flag (default
  `40.0`).
- `quasi_constant_threshold` controls quasi-constant detection (default
  `0.98`).
- `outlier_zscore_threshold` and `iqr_multiplier` control univariate outlier
  rules.
- `max_columns_for_plots`, `max_categories`, and the other `max_*` options keep
  wide or high-cardinality reports manageable.
- `random_state` makes sampled and stochastic calculations reproducible.

`reset_output_dir=True` is the default. Running the pipeline therefore clears
the configured output directory before generating a new report. Always use a
dedicated artifact directory, or set `reset_output_dir=False` when preserving
existing files is intentional.

### EDA outputs

The output directory contains `eda_summary.json`, `report.html`, a `plots/`
folder, and CSV artifacts such as:

```text
column_profile.csv
data_quality_flags.csv
numeric_summary.csv
robust_numeric_summary.csv
categorical_summary.csv
datetime_summary.csv
normality_tests.csv
outlier_summary.csv
strong_correlation_pairs.csv
multicollinearity.csv
missingness_summary.csv
categorical_associations.csv
pca_summary.csv
multivariate_outliers.csv
```

Target-specific, geospatial, and correlation-matrix files are written only
when the input and configuration make them applicable. See
[`eda/README.md`](eda/README.md) for the complete artifact list.

## Outlier analysis

Use `OutlierAnalysis` directly when row-level flags or focused calculations are
needed without the complete EDA report.

```python
from data_analysis.outliers import OutlierAnalysis

analysis = OutlierAnalysis(
    df,
    iqr_multiplier=1.5,
    z_score_threshold=3.0,
    forest_contamination=0.05,
    random_seed=42,
)

summary = analysis.summarize_univariate()
iqr_flags = analysis.iqr_outlier_mask("ndvi")
mahalanobis = analysis.detect_multivariate_mahalanobis(
    ["ndvi", "elevation", "rainfall"]
)
all_flags = analysis.detect_all_outliers()
```

The input is copied rather than modified. Boolean masks retain the original
dataframe index, making them safe to align or join. Missing and infinite values
are not classified as outliers; `check_and_standardize_data()` converts numeric
infinities to missing values before combined detection.

Available approaches include IQR, z-score, Isolation Forest, PCA/DBSCAN, PLS,
and Mahalanobis distance. Multivariate methods median-impute usable numeric
features, discard all-null and constant columns, standardize values, and obey
`max_multivariate_features`.

## Filling missing values

Missing-value helpers currently live in the implementation module:

```python
from data_analysis.fill_nulls.fill_nulls_library import (
    fill_numerical_with_KNN,
    fill_null_values_using_interpolation,
)

filled_df = fill_numerical_with_KNN(
    df,
    fill_col_name="elevation",
    features_for_knn=["slope", "rainfall"],
    n_neighbors=3,
)

filled_gdf = fill_null_values_using_interpolation(
    gdf,
    method="idw",
    fill_col_name="soil_value",
    null_value=0,
    max_distance_in_meters=300,
    plot=False,
)
```

The main helpers are:

- `fill_numerical_with_KNN` for numeric feature-based KNN imputation;
- `fill_categorical_with_KNN` for KNN classification of missing categories;
- `fill_missing_values_in_text_fields` for standardizing text and replacing
  configured null-like strings;
- `fill_nulls_using_neighborhood_values` for spatial min, max, median, mean,
  mode, or sum aggregation;
- `fill_null_values_using_interpolation` with `method="nearest"`, `"idw"`, or
  `"kriging"`; and
- lower-level variogram and kriging helpers for specialized geostatistical use.

Spatial interpolation converts geometries to representative points. Operations
that interpret distances in metres require a valid projected CRS; a geographic
latitude/longitude CRS is rejected because its units are angular. IDW requires
`max_distance_in_meters`, while kriging requires
`variogram_lags_max_dist_in_meters`. The module requires the `pykrige`
dependency. Use `plot=False` in automated or headless runs.

Most filling helpers return a copy, but callers should still keep the original
dataset and validate imputed values, class balance, spatial coverage, and data
types before using the result downstream.
