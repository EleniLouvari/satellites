# EDA Pipeline

Reusable exploratory data analysis for pandas `DataFrame` and GeoPandas `GeoDataFrame` inputs.

See the [detailed workflow and flowchart](WORKFLOW.md) for execution order,
optional branches, diagnostic subsets, saved outputs, and rerun behavior.

See [FEATURE_SELECTION_GUIDE.md](FEATURE_SELECTION_GUIDE.md) for the full explanation of the ordered ML feature-screening criteria.

## Minimal usage

```python
from eda import EDAConfig, EDAPipeline

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
- `eda_annotated_data.parquet`, preserving GeoParquet metadata and adding the combined outlier flag, per-method votes, vote counts and Mahalanobis scores

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
  multivariate/       PCA summary and method counts and every row flagged by a strict majority of outlier methods
  geospatial/         geometry summary
```

Plots remain under `plots/`. Numeric QQ plots exclude the target, configured ID, and prior EDA diagnostics and show at most
`max_features_per_plot` variables under `plots/qq_plots/`.

`max_features_per_plot` is the single display limit for feature-based charts, including numeric/categorical distributions, scatter matrices,
VIF, target comparisons, association rankings, and QQ plots. VIF calculation and display exclude the target and configured ID columns.
It replaces the former `max_columns_for_plots`, `max_target_scatter_features`, and `max_pairplot_features` options.

Feature-based plots use the same ML-screening order as `feature_selection_proposal.csv`: proposed `use` features first, followed by `review`
and `exclude` features in their evidence-ranked order. When target evidence is unavailable, unflagged columns with less missingness come first.
The CSV artifacts retain computed results even when plots display fewer features.
VIF calculation itself is capped by `max_features_per_plot`; PCA and Mahalanobis
use `max_multivariate_features`. See the [diagnostic subset details](WORKFLOW.md#7-plot-order-vif-pca-and-multivariate-outliers).

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
false-discovery-rate control, redundancy, and a parsimonious initial shortlist. The shortlist defaults to 50 variables; change it with
`feature_selection_max_features` or set that option to `None`. Configure known post-outcome fields with
`feature_selection_exclude_columns=(...)`. The final criterion remains train-only cross-validation and permutation importance, which belongs
after the train/test split and is intentionally not treated as an EDA result.

The combined outlier flag is diagnostic metadata, not a recommended model feature. Do not automatically delete flagged test rows. First
verify whether they are data errors; otherwise keep the test set representative and compare training with and without flagged training rows.
Because this EDA flag is calculated from the complete supplied dataset, do not use it as a predictor in the ML pipeline.
It can be used as downstream inspection metadata: the Neuro classification
workflow joins `is_outlier` by parcel ID, retains it outside `feature_columns`,
and enables `inspection_include_outlier=True`. True outliers then raise the base
inspection priority one band, capped at Very High; Unknown is unchanged.
Confidence and diagnostic scores are unaffected. See the
[inspection policy and workflow](../ml_classification/INSPECTION_PRIORITY_DECISION_TABLE.md).
The target, configured ID, and prior EDA diagnostic columns are excluded from Mahalanobis scoring; `eda_summary.json` records the exact
`multivariate_features_used` list.

## Combined parcel outliers

`OutlierAnalysis.detect_all_outliers()` now supplies `is_outlier` through
one vote each from IQR, z-score, multivariate Isolation Forest, PCA/DBSCAN and
Mahalanobis. A strict majority is required (normally 3 of 5); ties are false.
Unavailable methods abstain, and fewer than two available votes produces a null
flag. Rows with no finite selected features remain unscored. IQR and z-score
vote if any selected feature exceeds their configured threshold.

All five methods use the first `max_multivariate_features` numeric predictors
in input order, excluding target, configured ID and prior EDA diagnostics.
All-null and constant features are removed. Multivariate methods median-impute
remaining gaps and standardize internally; original values are preserved.
Isolation Forest uses 5% contamination and the configured random seed.
PCA/DBSCAN uses up to two components, `eps=0.3` and `min_samples=5`; this density
rule can flag many rows on sparse data, so inspect the method counts.
Mahalanobis uses a 97.5% chi-square cutoff with covariance rank as degrees of
freedom. PLS is excluded: it requires a meaningful response, not an all-zero target.

The combined flag is named `is_outlier`. It does **not** express a
combined 97.5% confidence level or force 2.5% of parcels to be flagged.
`eda_outlier_*` columns retain individual nullable votes, `eda_outlier_votes`
counts positive votes and `eda_outlier_methods_available` counts available votes.
The report and `tables/multivariate/outlier_method_summary.csv` show method
counts. `multivariate_outliers.csv` keeps its filename but now contains majority
outliers. `mahalanobis_distance` and `mahalanobis_pvalue` remain separate diagnostics.

The unused standalone methods `check_and_standardize_data`, `detect_iqr_outliers`,
`detect_zscore_outliers`, `fill_nulls`, `detect_isolation_forest_outliers`,
`detect_pca_outliers`, and `detect_outliers_pls` have been removed, along with the
PLS-only `pca_threshold` constructor option. Use `detect_all_outliers()` for
combined row flags and `summarize_univariate()` for per-feature summaries.
