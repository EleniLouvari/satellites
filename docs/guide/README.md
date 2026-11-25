# From satellite observations to inspection

Parcel Studio helps you understand agricultural parcels through satellite observations,
data diagnostics and classification. Use the contents sidebar to follow the processing
order, or open Help from a pipeline page to jump directly to its guide.

## The journey

1. **Imagery acquisition:** choose one source for each run.
   - **openEO:** configure parcels, imagery and accounts; plan and download Sentinel cubes.
   - **HUB catalog:** read analysis-ready Sentinel-2/Sentinel-1 imagery (`ARD_S2`, `ARD_S1`)
     with the server's catalog access and build local cubes batch by batch.
2. **Parcel statistics:** clean physical bands, fill gaps, calculate indices and summarize pixels by parcel.
3. **Feature enrichment (optional):** Python and notebook workflows can add temporal and environmental features.
4. **EDA:** inspect data quality, relationships, outliers and candidate features.
5. **ML classification:** Check → Prepare → Train → Evaluate → Predict.
6. **Inspection:** interpret confidence and reliability, then review the prioritized parcels.

Acquisition and local statistics are stages of the same parcel-statistics run in the app.
They have separate guide sections because they do different work and have different completion signals.
Both sources feed the same local cleaning, index and statistics pipeline. HUB Sentinel-1
features come from gamma0 backscatter and openEO's from sigma0, so treat them as different
source families when comparing runs.

## What you need

- A parcel file (GeoPackage, GeoParquet or GeoJSON) with a unique parcel identifier.
- The dates, bands and processing options for imagery acquisition, plus either Copernicus
  accounts (openEO) or HUB catalog access configured by the server operator.
- A target column with known labels to train a supervised classifier. Rows without labels can still be predicted.
- One row per entity for classification; geometry is useful for spatial splitting and diagnostics.

In the web app, open EDA from a **completed parcel-statistics run**, then create ML
scenarios from a **completed EDA analysis**. Several ML scenarios can compare different
feature/model choices on the same EDA analysis.

The Python EDA and ML libraries also accept independently prepared DataFrames or
GeoDataFrames. They do not require that their inputs were produced by openEO or HUB.

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
- HUB runs have no remote jobs. Their progress counts HUB batches, which appear once
  the catalog inventory is resolved, and then partition assembly.
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
