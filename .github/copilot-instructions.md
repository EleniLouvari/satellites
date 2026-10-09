# Copilot Instructions for Satellites Geospatial ML Classification

This repository combines **satellite image processing** (Sentinel-1/2 via openEO) with **machine learning classification** for
agricultural parcels.

## Project Architecture

### Three Core Pipelines

1. **Satellite Zonal Statistics** (`src/data_preparation/parcel_stats/`)
   - Extracts Sentinel-1/2 observations from Copernicus Data Space via openEO
   - Computes temporal statistics for parcel geometries
   - Outputs: one ML-ready row per parcel, dated feature columns (e.g., `NDVI_median__20240701`)
   - Two implementations: `SatelliteZonalStats` (batch-grouped), `JobManagerSatelliteZonalStats` (tile-based)
   - Processing: IQR cleaning → null filling (temporal, 3x3, 5x5) → index calculation → zonal masking → persistence as GeoParquet

2. **ML Classification Pipeline** (`src/ml_classification/`)
   - Five-step restartable workflow: Check → Prepare → Train → Evaluate → Predict
   - Each step writes outputs to numbered folders (`01_check/` through `05_predict/`) and HTML reports
   - Input: one row per entity (DataFrame or GeoDataFrame); output: class predictions with probabilities
   - Handles spatial splitting, class balancing, soft-voting ensembles, probability optimization
   - Supports hyperparameter tuning with configurable model candidates

3. **EDA Pipeline** (`src/eda/`)
   - Exploratory data analysis reports; mostly used in notebooks

### Key Integration Point

Zonal stats pipeline output → ML classification pipeline input. Features must be:
- One row per entity (parcel)
- Numeric or categorical columns
- Can include geometry for spatial splitting
- Target column can have missing values (rows without labels are classified only)

## Developer Workflows

### Running Code Quality Checks
```bash
# Windows: run_code_checks.bat
# Includes: ruff (lint/fix), pydocstyle, bandit (security), pytest with coverage
# Output: outputs/quality/ folder with junit.xml, pydocstyle.txt, bandit.txt
# Coverage HTML: htmlcov/index.html
```

### Notebook Environment Setup

Install the repository into the active kernel environment with `python -m pip install --no-deps --no-build-isolation -e .`.
Use explicit imports from `data_preparation`, `eda`, `ml_classification`, and
`shared`; do not add `sys.path` edits, compatibility aliases, or wildcard imports.

### Pipeline Usage Pattern (See `notebooks/projects/volvi/4_ml_classification.ipynb`)
```python
from ml_classification import (
    GeospatialClassificationPipeline, ClassificationPipelineConfig
)

config = ClassificationPipelineConfig(
    project_dir="output_path",
    target_column="label",
    feature_columns=[...],  # Exclude geometry, target, id
    id_column="parcel_id",
    spatial_split=True,  # For GeoDataFrames
    selection_type="soft_voting",
    selected_models=("random_forest", "extra_trees", ...),
    random_state=SEED_NUMBER,
)

pipeline = GeospatialClassificationPipeline(config)
pipeline.run_check(df)  # Reset project_dir, validate inputs
pipeline.run_prepare()  # Build train/test/CV folds
pipeline.run_train()    # Tune models
pipeline.run_evaluate() # Test set evaluation, choose strategy
pipeline.run_predict()  # Refit + predict all rows
```

## Project-Specific Patterns

### Feature Handling
- **Dated features** from zonal stats: name pattern `METRIC_statistic__YYYYMMDD` (e.g., `NDVI_median__20240701`)
- **Index preservation**: Sentinel-2 indices (NDVI, SAVI, NDWI, etc.) calculated from cleaned bands, not independently
- **Valid-pixel counts** exclude indices that depend on imputed bands
- Common filter: `[x for x in df.columns if any(kw in x for kw in ["_median", "_sd", "_range"]) and x not in meta_cols]`

### Class Distribution Handling
- Rare labels (< 100 samples): often recoded to 9999 or excluded
- Balancing options: `none`, `random_oversample`, `smote`
- CV method: stratified by default; `spatial_split=True` holds out complete grid cells
- Ranking: `cv_ranking_method="score_minus_std"` rewards stability across folds

### Artifact Persistence
- Each pipeline step reads/writes from disk (joblib, JSON, CSV)
- Config controls reset behavior: `reset_project_dir_on_run_check=True` (default) clears prior outputs
- Schema metadata: JSON artifacts include `_schema` block with artifact name + version
- Full predictions in `05_predict/final_predictions.joblib`; CSV preview in `05_predict/final_predictions_preview.csv`

### Random Seed Control
- Default seed in `shared.constants`; notebook setup is explicit (`SEED_NUMBER = 42`)
- TensorFlow, NumPy, random, hash all seeded; CUDA disabled
- OMP/threading limited to 4 intra-op threads for reproducibility
- `ml_classification/sensitivity/` reruns full pipeline over multiple seeds

### Spatial Splitting (GeoDataFrame Projects)
- `spatial_split=True` + `spatial_split_method="by_group"`: hold out complete grid cells
- `spatial_split_method="by_row"`: sample rows while retaining class coverage
- Grid size: `spatial_split_grid_size=10` (10×10 grid)
- Prevents data leakage and enables honest spatial evaluation

### HTML Report Navigation
- Top-level: `report_index.html` (project_dir root)
- Step reports: `01_check/report.html` through `05_predict/report.html`
- Includes summary tables, plots, metrics, and feature importance
- `open_html_report=True` auto-opens browser (useful in dev)

## Common Libraries

All reusable utilities live in `src/shared/`:
- `io.py`: read/write GeoParquet, CSV, NetCDF
- `geometry.py`: geometry repair, CRS transforms
- `raster.py`: IQR cleaning, spatial interpolation (nearest, IDW, kriging)
- `logging.py`: formatted console/file logging
- `tabular.py`: dataframe utilities, null value mapping

## Testing & Validation

- Unit tests: `tests/` (pytest)
- Test markers: `@pytest.mark.slow`, `@pytest.mark.e2e` (integration tests)
- Config: `pytest.ini` (testpaths, python_files, python_classes patterns)
- Coverage: stored as `outputs/quality/coverage.xml` and `outputs/quality/htmlcov/`
- Linting: `.pylintrc` for docstring/style rules

## Cross-Component Data Flow

```
Parcels (GeoParquet)
  ↓
openEO Sentinel retrieval (cloud-masked, 10m resampled)
  ↓
Raster cleaning (IQR, null filling)
  ↓
Index calculation (NDVI, SAVI, NDWI, etc.)
  ↓
Zonal statistics (parcel-masked aggregations)
  ↓
Dated pivot + derived metrics
  ↓
ML-ready GeoParquet (one row per parcel)
  ↓
ML Classification Pipeline (5-step training/inference)
  ↓
Predictions + probabilities + review flags (confidence < threshold)
```

## Key Files to Know

- **Entry config**: `src/shared/constants.py` (seed, null lists, plot themes)
- **Pipeline core**: `src/ml_classification/pipeline.py`
- **Step implementations**: `src/ml_classification/steps/`
- **Zonal stats entry**: `src/data_preparation/parcel_stats/openeo.py`
- **Common utilities**: `src/shared/`


## General Notes
- New code should be accompanied by unit tests in `tests/`.
- All code should adhere to the linting and formatting rules defined in `.pylintrc` and `pyproject.toml`.
- All code must be documented with docstrings and comments where necessary.
- When new functionality is added, update the relevant documentation and README files to reflect the changes.
- When new functions are created first check the existent code to avoid duplication. If a similar function exists,
  consider refactoring or extending it instead of creating a new one.
- Ensure that the new code is covered by unit tests and that the tests are comprehensive, covering edge cases and potential
  failure points.
- When adding new dependencies, ensure they are necessary and do not bloat the project. Update the dependency groups in `pyproject.toml` accordingly.
- The new code should be compatible with the existing codebase and follow the same coding conventions and patterns, i.e.,
  naming conventions, code structure, and documentation style (130 characters max per line).
- When new functionality is added, ensure that it follows the ruff and radon rules. New code should be tested for cyclomatic
  complexity and maintainability index, and refactored if necessary to meet the project's standards.
- When writing new code, ensure it is consistent with the existing project structure and conventions. Always add in-line comments to explain complex or non-obvious code. Always add docstrings for all new functions and classes.
- Always create unit-tests for new code to ensure functionality and maintainability.
- **Code quality metrics**: Ensure that new code maintains or improves the project's cyclomatic complexity, maintainability index, and comment density according to the defined thresholds.
  Run the code_quality.ipynb notebook to check the cyclomatic complexity, maintainability index, and comment density of the new code.
- In all error handling, provide clear and informative messages to aid in debugging and maintenance. Add the 'Error: ' prefix to all error messages.
- Use double quotes for string literals consistently throughout the codebase. Use double quotes for strings in lists, tuples, and dictionaries as well.
- Use f-strings for string formatting instead of concatenation or the `%` operator.
- When new code is added and new unit tests are created, ensure that they are comprehensive and cover edge cases as well as typical usage scenarios.
  Run the run_code_checks.bat script to ensure that all code quality checks pass before committing changes.
- Always ensure that the md files are updated, properly formatted and adhere to the project's markdown style guidelines.

## Repository organization

See `docs/repository_structure.md` for the source boundaries. Notebooks live in `notebooks/`, tests in `tests/`, reference data in `data/reference/`, and generated files in `outputs/`. Shared parcel-statistics code must not depend on an imagery acquisition service.
