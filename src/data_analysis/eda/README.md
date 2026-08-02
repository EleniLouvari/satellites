# EDA Pipeline

Reusable exploratory data analysis for pandas `DataFrame` and GeoPandas `GeoDataFrame` inputs.

## Minimal usage

```python
from data_analysis.eda import EDAConfig, EDAPipeline

config = EDAConfig(
    output_dir="src/data_analysis/eda/example_outputs",
    target_column="class",
    report_title="My EDA Report",
)

result = EDAPipeline(config).run(df)
result["report_path"]
```

## Outputs

- `eda_summary.json`
- `column_profile.csv`
- `data_quality_flags.csv`
- `numeric_summary.csv`
- `robust_numeric_summary.csv`
- `categorical_summary.csv`
- `datetime_summary.csv`
- `normality_tests.csv`
- `outlier_summary.csv`
- `strong_correlation_pairs.csv`
- `multicollinearity.csv`
- `missingness_summary.csv`
- `missingness_target_tests.csv` when `target_column` is categorical
- `categorical_associations.csv`
- `categorical_target_tests.csv` when `target_column` is categorical
- `numeric_target_tests.csv` when `target_column` is categorical
- `numeric_target_correlations.csv` when `target_column` is numeric
- `pca_summary.csv`
- `multivariate_outliers.csv`
- `target_summary.csv` when `target_column` is provided
- `geospatial_summary.csv` when geometry is available
- `pearson_correlation_matrix.csv` and `spearman_correlation_matrix.csv` for numeric datasets
- `plots/*.png`, including missingness, data-quality, QQ, correlation, VIF, PCA, scatter-matrix, target, categorical-association, target-association, geometry, and geometry-target heatmap plots when applicable
- `report.html`
