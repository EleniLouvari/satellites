# Optional parcel feature enrichment

Feature enrichment adds context to a parcel table between statistics and analysis.
It is available through Python functions and the project notebook; Parcel Studio does
not currently expose a separate feature-enrichment stage.

## Temporal summaries

`data_preparation.features.temporal` reduces dated satellite features into temporal
summaries. The openEO parcel-statistics pipeline already writes a reduced sibling
GeoParquet after the dated table. Choose the representation appropriate to your analysis
rather than applying the same reduction twice.

## Environmental attributes

The feature helpers support:

- **Elevation:** aggregate DEM elevation for parcel geometries.
- **Soil:** attach soil information from the supplied reference source.
- **Nearest lake:** add proximity and lake context using the configured reference data.

These operations require their own reference datasets and spatial configuration.
They do not download Sentinel imagery and are independent of acquisition services.

## Project workflow

The configured example is `notebooks/projects/neuro/2_parcel_features.ipynb`.
Read its input/output paths and configuration cells before running it; the project
can use external work directories. The notebook adds features to a parcel GeoParquet
and uses atomic replacements for its file updates.

Implementation and API docstrings are in:

- `src/data_preparation/features/temporal.py`
- `src/data_preparation/features/elevation.py`
- `src/data_preparation/features/spatial_context.py`

## Handoff to EDA and ML

Preserve one row per parcel and its unique identifier. Keep the target and geometry
columns distinct from candidate predictor columns. Inspect new attributes for missingness,
units and unwanted target information before selecting them for classification.

The Python EDA and ML APIs accept independently prepared tables. The current web app
uses the completed parcel-statistics result for EDA and the completed EDA analysis for ML;
this page does not imply a separate enriched-table upload flow in the app.
