# EDA Pipeline

Reusable exploratory data analysis for pandas `DataFrame` and GeoPandas `GeoDataFrame` inputs.

See [FEATURE_SELECTION_GUIDE.md](FEATURE_SELECTION_GUIDE.md) for the full explanation of the ordered ML feature-screening criteria.

## Minimal usage

```python
from eda_pipeline import EDAConfig, EDAPipeline

config = EDAConfig(
    output_dir="outputs/eda",
    target_column="class",
    target_task="classification",
    report_title="My EDA Report",
)

result = EDAPipeline(config).run(df)
result["report_path"]
result["proposed_features"]
ml_ready_gdf = result["annotated_data"]
```

## Outputs

The report, summary, and ML-ready data remain easy to find at the output root:

- `report.html`
- `eda_summary.json`
- `eda_annotated_data.parquet`, preserving GeoParquet metadata and adding the multivariate outlier flag and scores

CSV tables are grouped by analytical scope:

```text
tables/
  overview/           column profile and datetime summary
  data_quality/       quality flags and missingness summary
  numeric/            numeric, robust, normality, and univariate-outlier summaries
  categorical/        categorical summary and pairwise associations
  correlation/        Pearson/Spearman matrices, strong pairs, and VIF
  target/             target summaries and feature-vs-target tests
  feature_selection/  ordered criteria, association ranking, proposal CSV, and printed-message text
  multivariate/       PCA summary and every row flagged at the 97.5% Mahalanobis threshold
  geospatial/         geometry summary
```

Plots remain under `plots/`. Numeric QQ plots exclude the target, configured ID, and prior EDA diagnostics and show at most
`max_features_per_plot` variables under `plots/qq_plots/`.

`max_features_per_plot` is the single display limit for feature-based charts, including numeric/categorical distributions, scatter matrices,
VIF, target comparisons, association rankings, and QQ plots. VIF calculation and display exclude the target and configured ID columns.
It replaces the former `max_columns_for_plots`, `max_target_scatter_features`, and `max_pairplot_features` options.

Feature-based plots use the same ML-screening order as `feature_selection_proposal.csv`: proposed `use` features first, followed by `review`
and `exclude` features in their evidence-ranked order. When target evidence is unavailable, unflagged columns with less missingness come first.
The CSV artifacts retain all analyzed features even when plots display only the configured maximum.

## Target-aware feature screening

Set `target_task="classification"` or `target_task="regression"` when a numeric target contains class labels or when automatic
dtype-based inference would be ambiguous. The default `target_task="auto"` treats numeric targets as regression and other targets as
classification.

The feature-selection tab combines:

- Pearson, Spearman, and Kendall associations for numeric regression features;
- ANOVA and Kruskal-Wallis tests with eta-squared and epsilon-squared effect sizes;
- chi-square tests with conventional and finite-sample bias-corrected Cramér's V;
- Benjamini-Hochberg false-discovery-rate correction; and
- direct distribution comparisons for the strongest numeric and categorical features.

These are univariate screening diagnostics, not a final feature-selection decision. Validate shortlisted variables with leakage checks,
cross-validation, model-based importance, and the correlation/VIF diagnostics because redundant variables can share the same signal.

The automated proposal applies these criteria in order: hard exclusions and leakage configuration, data quality, target association with
false-discovery-rate control, redundancy, and a parsimonious initial shortlist. The shortlist defaults to 20 variables; change it with
`feature_selection_max_features` or set that option to `None`. Configure known post-outcome fields with
`feature_selection_exclude_columns=(...)`. The final criterion remains train-only cross-validation and permutation importance, which belongs
after the train/test split and is intentionally not treated as an EDA result.

The multivariate outlier flag is diagnostic metadata, not a recommended model feature. Do not automatically delete flagged test rows. First
verify whether they are data errors; otherwise keep the test set representative and compare training with and without flagged training rows.
Because this EDA flag is calculated from the complete supplied dataset, do not use it as a predictor in the ML pipeline.
The target, configured ID, and prior EDA diagnostic columns are excluded from Mahalanobis scoring; `eda_summary.json` records the exact
`multivariate_features_used` list.
