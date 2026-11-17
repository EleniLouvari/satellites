# From satellite observations to inspection

Parcel Studio helps you understand agricultural parcels through satellite observations,
data diagnostics and classification. Use the contents sidebar to follow the processing
order, or open Help from a pipeline page to jump directly to its guide.

## The journey

1. **openEO acquisition:** configure parcels, imagery and accounts; plan and download Sentinel cubes.
2. **Parcel statistics:** clean physical bands, fill gaps, calculate indices and summarize pixels by parcel.
3. **Feature enrichment (optional):** Python and notebook workflows can add temporal and environmental features.
4. **EDA:** inspect data quality, relationships, outliers and candidate features.
5. **ML classification:** Check → Prepare → Train → Evaluate → Predict.
6. **Inspection:** interpret confidence and reliability, then review the prioritized parcels.

Acquisition and local statistics are stages of the same parcel-statistics run in the app.
They have separate guide sections because they do different work and have different completion signals.

## What you need

- A parcel file (GeoPackage, GeoParquet or GeoJSON) with a unique parcel identifier.
- Copernicus accounts and the dates, bands and processing options for imagery acquisition.
- A target column with known labels to train a supervised classifier. Rows without labels can still be predicted.
- One row per entity for classification; geometry is useful for spatial splitting and diagnostics.

In the web app, open EDA from a **completed parcel-statistics run**, then create ML
scenarios from a **completed EDA analysis**. Several ML scenarios can compare different
feature/model choices on the same EDA analysis.

The Python EDA and ML libraries also accept independently prepared DataFrames or
GeoDataFrames. They do not require that their inputs were produced by openEO.

## Read a stage in three passes

Start with its inputs, numbered processing steps and outputs. Then follow the reading
list for the detailed workflow and configuration. Open the HTML workflow reference or
a full-size PNG when you want to see the whole sequence together.

Each article has an in-page table of contents. Previous and Next follow the curated
reading order. Advanced reference covers deployment, command-line use and source code
ownership; it is optional for understanding the main workflow.

## Understand completion and quality

- A finished remote openEO job is not a completed parcel-statistics run. Download,
  local processing and verified publication must also finish.
- Filled pixels are not equivalent to observed pixels. Read the local processing
  guide to interpret provenance and valid-pixel counts.
- EDA feature proposals are diagnostics. Final selection requires train-only validation.
- Inspection priorities support review. Read the decision reasons and quality evidence
  together with the predicted class.

## About this guide

The application builds these pages from the current Markdown guides and workflow
assets in the repository. Existing document locations remain stable. Historical Word
manuals, internal implementation plans and image-generation prompts are not part of
this learning path. References to source files or notebooks that are not published here
are labelled as repository references.
