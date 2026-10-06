# Import migration map

Old imports remain available through compatibility aliases after installation. Use the new paths for maintained code. Existing module-level class names are preserved.

| Previous import | Canonical import |
| --- | --- |
| `common_libraries` | `satellites.shared` |
| `common_libraries.catalog_constants` | `satellites.data_preparation.sources.hub.catalog_constants` |
| `common_libraries.catalog_library` | `satellites.data_preparation.sources.hub.catalog_library` |
| `common_libraries.catalog_s3_checks` | `satellites.data_preparation.sources.hub.catalog_s3_checks` |
| `common_libraries.data_cleaning` | `satellites.shared.data_cleaning` |
| `common_libraries.data_cleaning._validation` | `satellites.shared.data_cleaning._validation` |
| `common_libraries.data_cleaning.outliers` | `satellites.shared.data_cleaning.outliers` |
| `common_libraries.data_cleaning.spatial_interpolation` | `satellites.shared.data_cleaning.spatial_interpolation` |
| `common_libraries.data_cleaning.tabular_imputation` | `satellites.shared.data_cleaning.tabular_imputation` |
| `common_libraries.data_cleaning.variograms` | `satellites.shared.data_cleaning.variograms` |
| `common_libraries.eda_report_class` | `satellites.shared.legacy.eda_report` |
| `common_libraries.generic_library` | `satellites.shared.tabular` |
| `common_libraries.geom_library` | `satellites.shared.geometry` |
| `common_libraries.io_library` | `satellites.shared.io` |
| `common_libraries.logging_library` | `satellites.shared.logging` |
| `common_libraries.parcel_features` | `satellites.data_preparation.features` |
| `common_libraries.parcel_features.elevation` | `satellites.data_preparation.features.elevation` |
| `common_libraries.parcel_features.spatial_context` | `satellites.data_preparation.features.spatial_context` |
| `common_libraries.raster_library` | `satellites.shared.raster` |
| `common_libraries.s3_library` | `satellites.shared.s3` |
| `common_libraries.spatial_stat_library` | `satellites.shared.spatial_statistics` |
| `eda_pipeline` | `satellites.eda` |
| `eda_pipeline.core` | `satellites.eda.core` |
| `eda_pipeline.core.config` | `satellites.eda.core.config` |
| `eda_pipeline.core.io` | `satellites.eda.core.io` |
| `eda_pipeline.core.profiling` | `satellites.eda.core.profiling` |
| `eda_pipeline.pipeline` | `satellites.eda.pipeline` |
| `eda_pipeline.reporting` | `satellites.eda.reporting` |
| `eda_pipeline.reporting.html` | `satellites.eda.reporting.html` |
| `eda_pipeline.visuals` | `satellites.eda.visuals` |
| `eda_pipeline.visuals.plots` | `satellites.eda.visuals.plots` |
| `global_variables` | `satellites.shared.legacy.notebook_settings` |
| `hub_catalog.planet_delivery` | `satellites.data_preparation.sources.planet.delivery` |
| `hub_catalog.planet_parcel_stats` | `satellites.data_preparation.parcel_stats.planet` |
| `import_libraries` | `satellites.shared.legacy.imports` |
| `ml_classification.ml_classification_pipeline` | `satellites.ml_classification` |
| `ml_classification.ml_classification_pipeline.core` | `satellites.ml_classification.core` |
| `ml_classification.ml_classification_pipeline.core.base` | `satellites.ml_classification.core.base` |
| `ml_classification.ml_classification_pipeline.core.class_reliability` | `satellites.ml_classification.core.class_reliability` |
| `ml_classification.ml_classification_pipeline.core.config` | `satellites.ml_classification.core.config` |
| `ml_classification.ml_classification_pipeline.core.inspection` | `satellites.ml_classification.core.inspection` |
| `ml_classification.ml_classification_pipeline.core.iqr` | `satellites.ml_classification.core.iqr` |
| `ml_classification.ml_classification_pipeline.core.metrics` | `satellites.ml_classification.core.metrics` |
| `ml_classification.ml_classification_pipeline.core.models` | `satellites.ml_classification.core.models` |
| `ml_classification.ml_classification_pipeline.core.persistence` | `satellites.ml_classification.core.persistence` |
| `ml_classification.ml_classification_pipeline.core.rank_confidence` | `satellites.ml_classification.core.rank_confidence` |
| `ml_classification.ml_classification_pipeline.core.selection` | `satellites.ml_classification.core.selection` |
| `ml_classification.ml_classification_pipeline.core.spatial_interpolation` | `satellites.ml_classification.core.spatial_interpolation` |
| `ml_classification.ml_classification_pipeline.core.spatial_split` | `satellites.ml_classification.core.spatial_split` |
| `ml_classification.ml_classification_pipeline.core.tensorflow_lstm_models` | `satellites.ml_classification.core.tensorflow_lstm_models` |
| `ml_classification.ml_classification_pipeline.core.tensorflow_models` | `satellites.ml_classification.core.tensorflow_models` |
| `ml_classification.ml_classification_pipeline.pipeline` | `satellites.ml_classification.pipeline` |
| `ml_classification.ml_classification_pipeline.reporting` | `satellites.ml_classification.reporting` |
| `ml_classification.ml_classification_pipeline.reporting.confidence_diagnostics` | `satellites.ml_classification.reporting.confidence_diagnostics` |
| `ml_classification.ml_classification_pipeline.reporting.final_dashboard` | `satellites.ml_classification.reporting.final_dashboard` |
| `ml_classification.ml_classification_pipeline.reporting.html` | `satellites.ml_classification.reporting.html` |
| `ml_classification.ml_classification_pipeline.reporting.inspection_diagnostics` | `satellites.ml_classification.reporting.inspection_diagnostics` |
| `ml_classification.ml_classification_pipeline.reporting.reports` | `satellites.ml_classification.reporting.reports` |
| `ml_classification.ml_classification_pipeline.steps` | `satellites.ml_classification.steps` |
| `ml_classification.ml_classification_pipeline.steps.check_step` | `satellites.ml_classification.steps.check_step` |
| `ml_classification.ml_classification_pipeline.steps.evaluate_step` | `satellites.ml_classification.steps.evaluate_step` |
| `ml_classification.ml_classification_pipeline.steps.predict_step` | `satellites.ml_classification.steps.predict_step` |
| `ml_classification.ml_classification_pipeline.steps.prepare_step` | `satellites.ml_classification.steps.prepare_step` |
| `ml_classification.ml_classification_pipeline.steps.train_step` | `satellites.ml_classification.steps.train_step` |
| `ml_classification.ml_classification_pipeline.visuals` | `satellites.ml_classification.visuals` |
| `ml_classification.ml_classification_pipeline.visuals.interpretability` | `satellites.ml_classification.visuals.interpretability` |
| `ml_classification.ml_classification_pipeline.visuals.maps` | `satellites.ml_classification.visuals.maps` |
| `ml_classification.ml_classification_pipeline.visuals.plots` | `satellites.ml_classification.visuals.plots` |
| `ml_classification.ml_classification_sensitivity` | `satellites.ml_classification.sensitivity` |
| `ml_classification.ml_classification_sensitivity.runner` | `satellites.ml_classification.sensitivity.runner` |
| `ml_classification_pipeline` | `satellites.ml_classification` |
| `ml_classification_sensitivity` | `satellites.ml_classification.sensitivity` |
| `openeo_parcel_stats_pipeline` | `satellites.data_preparation.parcel_stats` |
| `openeo_parcel_stats_pipeline.configuration` | `satellites.data_preparation.parcel_stats.core.configuration` |
| `openeo_parcel_stats_pipeline.crs` | `satellites.data_preparation.parcel_stats.core.crs` |
| `openeo_parcel_stats_pipeline.feature_reduction` | `satellites.data_preparation.features.temporal` |
| `openeo_parcel_stats_pipeline.openeo_pipeline` | `satellites.data_preparation.sources.openeo.cubes` |
| `openeo_parcel_stats_pipeline.parcel_batches` | `satellites.data_preparation.parcel_stats.core.parcel_batches` |
| `openeo_parcel_stats_pipeline.parcel_statistics` | `satellites.data_preparation.parcel_stats.core.parcel_statistics` |
| `openeo_parcel_stats_pipeline.raster_cleaning` | `satellites.data_preparation.parcel_stats.core.raster_cleaning` |
| `openeo_parcel_stats_pipeline.zonal_stats` | `satellites.data_preparation.parcel_stats.openeo` |
| `openeo_parcel_stats_pipeline.zonal_stats_job_manager` | `satellites.data_preparation.parcel_stats.job_manager` |
| `openeo_parcel_stats_pipeline.zonal_stats_multiuser_manager` | `satellites.data_preparation.parcel_stats.multiuser` |
| `parcel_stats_pipeline` | `satellites.data_preparation.parcel_stats` |

## Numbered classification steps

Classification implementations now live in `step_01_check/` through
`step_05_predict/`, each with `libraries/`, and in `shared/`. The older canonical
paths in the table above remain compatibility exports. For new code, use the
[source guide](../src/satellites/ml_classification/README.md) and generated
[file usage index](../src/satellites/ml_classification/FILE_USAGE.md) to locate the
defining module. The public pipeline/config imports remain unchanged.

The classification wrappers now live in
[`ml_classification/to_delete/`](../src/satellites/ml_classification/to_delete/README.md).
An optional package search path keeps their old import names working during
testing. This folder also retains original source snapshots. Delete it only
after testing and resolving any dependence on historical saved class paths.
