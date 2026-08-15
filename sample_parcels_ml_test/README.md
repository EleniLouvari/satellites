# Compact parcel sample for ML pipeline testing

This sample covers approximately 3 km x 3 km in EPSG:2100 and contains 479
unique parcels. It was extracted from the all-zones crop-classification inputs.

## Files

- `sample_parcels_ml_features_long.parquet`: recommended input for the fixed ML
  pipeline. It has 3,832 parcel-period rows, eight dates, geometry, labels,
  predictions, and 15 safe satellite feature families.
- `sample_parcels_ml_features_wide.parquet`: one row per parcel with all 68
  dated source predictors and all retained source metadata.
- `sample_parcels_boundaries.gpkg`: parcel polygons and label/prediction fields;
  also suitable as a small input for testing `zonal_stats.py`.
- `ml_feature_columns.csv`: the 15 safe long-format predictor names.
- `class_distribution.csv`: original-label and model-target counts.
- `sample_summary.json`: source, bounds, schema and QA information.
- `run_sample_pipeline.py`: runnable example for the corrected soft-voting
  pipeline.

## Target labels

`initial_label` is preserved unchanged. `class_ml`/`class` groups original
labels having fewer than five parcels in this local sample into `other_rare`.
This produces 14 testable classes with at least six parcels per class.

Do not use `svm_novelty_full`, holdout fields, initial/predicted labels, parcel
IDs, dates, or batch metadata as predictors. The feature manifest contains only
the satellite predictors intended for modeling.

For this small local area, use `cv_folds=3` and
`spatial_split_grid_size=4`. Larger production datasets should use stronger
spatial and independent-year validation.
