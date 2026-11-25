# Running Parcel Studio

Parcel Studio connects the approved React interface to FastAPI, PostgreSQL, and
the existing parcel-statistics pipeline. Registration grants immediate access.
Each user owns their uploads, EO accounts, saved configurations, run history,
events and downloads. The CLI and its batch launcher remain usable independently.

## Windows / Docker Desktop

### Defaults for a first parcel-statistics test

New extraction configurations start with a conservative profile for a slow local disk:

| Setting | Default | Purpose |
| --- | --- | --- |
| Local workers | 1 | Avoid concurrent raster reads and writes. |
| Remote jobs per account | 1 | Reduce simultaneous download pressure; multiple accounts can still download together. |
| Maximum fitted tile width/height | 5,000 m, including buffer | Keep individual raster cubes smaller. |
| Temporal fill block rows | Automatic (blank) | Size chunk-aligned temporal blocks automatically. |
| Scheduling | Shared batches | Process downloaded batches without waiting for an entire partition. |
| Remove NetCDFs after completion | Enabled | Release rasters after durable statistics and reports are saved. |
| Completed-partition reuse | Automatic in the runner | Reuse compatible completed partition results. |

This is a starting profile, not a measured optimum for every dataset. Parcels exceeding the
maximum fitted extent need a larger tile; preflight reports that condition. Sensor selections,
acquisition pooling and statistical/cleaning choices retain their existing defaults. Saved runs,
resumed analyses and runs opened for editing retain their saved settings. To use these defaults,
start a new extraction after rebuilding and restarting the app; this does not submit a run.
In particular, resuming an older run that used partition scheduling keeps that scheduling mode
and its saved worker/tile settings; it does not adopt this new-run profile.

### Docker setup

Use Linux containers. The existing `scripts/.env_docker` holds database connection
settings and the **existing** Fernet key:

```dotenv
POSTGRES_HOST=host.docker.internal
POSTGRES_PORT=5435
DB_NAME=satellites
POSTGRES_USER=your_database_role
POSTGRES_PASSWORD=your_database_password
SATELLITES_CREDENTIAL_KEY=your_existing_key
```

Do not generate a replacement key for previously encrypted accounts. The web
application uses `public.app_users` and `public.eo_users`; legacy custom table
overrides are not supported by this web release.

From the repository root:

```powershell
.\scripts\run_web_app.bat build
.\scripts\run_web_app.bat migrate
.\scripts\run_web_app.bat start
```

Open **http://localhost:8080**. To inspect or stop services:

```powershell
.\scripts\run_web_app.bat logs
.\scripts\run_web_app.bat stop
```

`build` builds the existing pipeline image, API/worker image, and static frontend.
`migrate` explicitly runs the packaged additive migration in one transaction.
It preserves user/account rows, makes password columns TEXT, adds missing active
flags and generated-ID defaults, and creates web tables. It will fail instead of
silently discarding conflicting duplicate usernames. API startup never migrates.
An operator/database-owner role is needed for migration; use a restricted runtime
role for regular operation.

Docker's named volume `parcel-studio_parcel_storage` retains uploaded files,
checkpoints, logs and results through restarts. Back up this volume, PostgreSQL,
and the Fernet key together. Do not run `docker compose down -v` against data you
want to keep. The application does not delete old results automatically.

### Store application files in a Windows folder

Set `PARCEL_HOST_STORAGE` to a Windows folder to use a bind mount instead of the
named volume. Keep `PARCEL_STORAGE_ROOT=/data/parcel-studio`: that is the Linux
path inside the container and is also used by existing saved configurations.

```powershell
$env:PARCEL_HOST_STORAGE = "C:/work_sat/parcel-studio"
.\scripts\run_web_app.bat start
```

For persistence across PowerShell sessions, set `PARCEL_HOST_STORAGE` in the
repository-root Compose `.env` file. `scripts/.env_docker` supplies container
environment variables; it does not supply Compose substitutions for this mount.

This maps `C:/work_sat/parcel-studio` on Windows to `/data/parcel-studio` in the
API, worker, inspector and management containers. Uploaded parcels, statistics,
EDA outputs and logs use this folder. PostgreSQL data stays in its own database
storage.

For an existing installation, finish active jobs, stop the app, and copy the
contents of `parcel-studio_parcel_storage` into the Windows folder **before**
switching. Merely changing the mount does not migrate existing files; otherwise
saved run records would point to files absent from the new folder. Preserve the
original volume until the copied results are verified. No image rebuild is needed.

## Existing application passwords

New registration saves an Argon2id password hash. Existing plaintext application
passwords are deliberately not accepted at login. Set an existing user's password
interactively (the value never appears in command arguments):

```powershell
docker compose -f compose.web.yml run --rm manage -m parcel_app.manage set-password YOUR_USERNAME
```

This changes only that application's password hash and revokes its web sessions;
it does not change Copernicus credentials. Alternatively, register a new user and
add their own Copernicus accounts. Existing EO accounts remain linked to their
original app_user_id and are available when that user signs in.

## Using the application

### HUB Sentinel ARD

Choose **HUB Sentinel ARD** in the extraction wizard's Imagery source step to
read `ARD_S2` and/or `ARD_S1`. HUB uses the server's service identity, so there
is no Copernicus account selection or remote-job scheduling control. A status
message indicates whether the operator has configured access. Existing runs
default to openEO; edits, duplicates and resumes retain their saved source.

Operators: add `CATALOG_URL`, `KEYCLOAK_URL`, `CATALOG_CLIENT_ID` and
`CATALOG_CLIENT_SECRET` to `${PARCEL_ENV_FILE:-./scripts/.env_docker}` before
starting the API and workers. The repository-root notebook `.env` is not the
container environment file. Both API and worker receive these settings; the
API exposes only configured/not-configured status, and workers authenticate.
Use a Keycloak realm URL reachable from the containers, rather than copying a
cluster-internal URL from an item's `auth:schemes` metadata. Write both URLs as
plain `https://...` values: HUB counts as not configured, and saving or starting
a HUB run is rejected with the setting's name, when a URL lacks an http(s) scheme
or contains spaces, quotes or pasted link brackets such as `<https://...>`.
Recreate the API and worker containers after editing the file. Secrets and endpoint
overrides are never browser-editable or stored in run configurations.

Rebuild the application images and apply the additive `006_hub_source.sql`
migration with the existing `run_web_app.bat` commands before starting this
version. The migration preserves existing runs as `openeo`. HUB deletion performs
local cleanup without requesting Copernicus remote-job cleanup.

HUB defaults to EPSG:2100. `hub_read_workers` defaults to 8, capped by the operator's
`PARCEL_MAX_HUB_READ_WORKERS` (default 8); this caps total asset-read concurrency
within each extraction worker. `hub_block_size` defaults to 512 pixels. Local statistics workers keep the
conservative default of 1. The source prunes irrelevant footprints, bounds
in-flight reads, and splits wide parcel groups into smaller spatial batches.
Each completed cube starts local processing as a worker becomes available,
while subsequent cubes are acquired. `batch_workers` caps local processing
processes; `hub_read_workers` independently caps imagery reads. The pending
local work is bounded to `batch_workers` jobs plus one acquired cube waiting
for a slot. With NetCDF cleanup enabled, workers remove each batch's rasters
after its statistics and report are saved. Worker logs use `.worker-<pid>.log`
filenames, even with one worker, and their messages reach the live run events.
With filling disabled it reads only the parcel context needed for masking;
unlimited interpolation can require a wider context when filling is enabled.

The [HUB library guide](../data_preparation/hub/README.md) describes pixel rules,
gamma0 provenance, inventory-aware resume and output files. HUB's gamma0 features
remain a different source family from openEO's sigma0 features.

#### HUB imagery in the inspection map

For a HUB analysis, **Map** in the best completed ML scenario's inspection table
uses the run's recorded HUB acquisitions. The pop-up has two panels: the imagery
controls, timeline, index information and statistics chart on the left, and only the
map on the right (stacked on narrow screens). Choose **Imagery layer** to view
Sentinel-2 true colour, a configured optical index, Sentinel-1 VV/VH false colour,
or a configured radar index. The acquisition selector, timeline and Older/Newer
buttons follow the selected sensor. The **Opacity** slider fades the preview over
the background map (0% transparent, 100% opaque); the setting is kept when you
change date or layer. Index legends use fixed scales across parcels
and dates; values outside the displayed range use its end colours.

When an index layer is selected, the map also gives a short description of what the
index shows and its formula. Below it are the parcel's saved **mean**, **median**,
**SD** and **range** of that index, for the statistics period that contains the
selected acquisition, meaning the latest period starting on or before the acquisition's
date. Only the statistics the analysis calculated are listed (range needs min and max),
and "—" marks a period without valid pixels. These values come from the analysis's
cleaned and filled pixels, so they can differ from the preview. Planet basemap layers
are not analysis inputs and have no statistics.

Under this information, a chart shows the parcel's saved **mean** (line) ± 1 **SD**
(band) of the selected index for every statistics period of the run (for example each
month with `temporal_period` `1M`). The chart changes with the imagery layer only, not
with the selected acquisition. Periods without a mean break the line. Hovering shows a
period's values, and **Show values** lists them as a table.

Previews show observed ARD imagery, without the analysis's local outlier removal,
gap filling or temporal aggregation. True colour retains clouds so inspectors can
see the reason for gaps; index layers apply the source cloud/shadow validity mask.
Radar false colour uses red = VV dB (−20 to 0), green = VH dB (−28 to −8), and
blue = 10 log10(VV/VH) (0 to 15). It is a radar interpretation aid.

Every HUB analysis also offers two Planet monthly basemap layers from the `PL_BSM`
collection: **Planet true colour** (`quad_bandmath` bands 6, 4, 2 as red, green, blue)
and **Planet NDVI** (band 9, as delivered). The basemaps serve as reference imagery
only: they are not statistics inputs and do not appear in the run's item manifests.
The timeline therefore lists every basemap month overlapping the analysis dates. A
month includes the basemap dated on its first day, even when the analysis starts later.
Quads of one month are mosaicked. Band 10 (alpha) hides pixels outside the mosaic data.
True colour converts reflectance × 10,000 to reflectance and uses the Sentinel-2 range
(0 to 0.3). NDVI uses the Sentinel-2 NDVI scale. Basemaps are shown at their native
pixel size of about 4.8 m.

Each preview covers the parcel bounds plus 500 m, using the HUB source's 10 m
sampling. A nearest-neighbour Web Mercator display warp preserves placement on
the map; display oversampling adds no source detail. Parcels exceeding the preview
extent limit (512 source pixels per axis, including buffer) receive a readable
error. Background maps remain available if an image has no coverage or valid pixels.

The API reads small COG windows with server credentials and keeps recent band
windows in a bounded memory cache. Images are cached privately by the browser;
no preview cache is written to disk. API restarts clear the server cache. Preview
loading needs configured HUB access and the run's partition item manifests, even
when extraction NetCDFs were removed. Missing or replaced source items are never
silently substituted with new acquisitions. Planet basemaps are an exception: the
catalog is searched whenever the API process lists a parcel's acquisitions for the first time.
No database migration is needed.

### Existing openEO workflow

1. Register or log in. Registration is immediate; there is no approval step.
2. Add one or more Copernicus accounts. Passwords are encrypted at write time;
   the browser cannot retrieve stored passwords. Connection testing is explicit
   and authenticates without creating a processing job.
3. Create an analysis and upload GeoPackage, GeoParquet or GeoJSON. Wait for
   inspection, choose the layer and unique parcel identifier, and optionally a
   partition column. The map is a simplified preview, limited to 500 polygons.
4. Select accounts and configure the run. Every exposed Python constructor
   option is listed in collapsible Basics, Sentinel imagery, Outlier removal,
   Gap filling, Interpolation, Tiles and execution, and Storage panels.
   IQR controls require outlier removal; temporal/spatial filling and interpolation
   require gap filling. IDW enables search distance; kriging enables variogram
   lags and distance; nearest requires neither. Disabled inputs keep their values
   while editing but are omitted from the saved effective processing request.
   New forms default to `temporal_fill_mode=bidirectional`,
   `require_parcel_observation=true` and `temporal_reducer=none`. The parcel
   observation check is part of Gap filling, so it is sent only when gap filling
   is enabled; it leaves a parcel's date out when none of its pixels were observed. These web defaults do not overwrite existing saved
   selections or change the CLI constructor defaults. Reporting intervals remain
   editable with reducer `none` because they still define the output periods.
   **Temporal fill block rows** in Gap filling accepts a positive maximum row count,
   such as `256`, aligned down to storage chunks where possible. Wide rasters also
   use column blocks. Larger values can use more RAM per local worker; leave blank
   for automatic chunk-aligned sizing with a 64 MiB payload target. This affects
   execution rather than calculated results and does not retune running workers.
   Bands, indices and statistics use supported choices from the Python classes.
   Scheduling is available separately. Import/export processing JSON is supported.
5. Save a draft and validate. Validation runs separately from HTTP requests.
   Editing invalidates the previous validation. Submit the validated revision.
6. Return to My runs for progress, elapsed duration, sanitized stage events,
   configuration, results, cleaning reports and logs. Use **Download all files
   (.zip)** in the completed run header or Files panel. The first click queues
   archive preparation and starts the download when ready; later downloads reuse
   that archive. It includes every published file (results, reports, configuration,
   manifest and sanitized logs), excluding raw caches and private internal files.

Duration comes from the first recorded processing-start event to completion or
failure, and updates while running. Initial queue time is excluded; pauses between
resume attempts are included. Runs without a recorded start show a dash. Existing
run events provide this timing, so this change requires no new database columns.

To change an existing analysis name, open its details and select **Rename**,
enter a name (up to 120 characters), and select **Save name**. Running and completed
analyses can also be renamed. This changes only the display name; validation,
processing settings, results and recorded duration are preserved. Renaming is
unavailable while an analysis is being deleted.

Log redaction recognizes credential fields (including quoted passwords, tokens,
usernames and authorization headers) and URLs containing authentication data.
Ordinary values such as `B11`, `1.5` and `1M` remain readable even when they contain
the same text as a credential. Public backend URLs remain readable; private server
paths are masked separately. Previously saved redacted log text cannot be restored
by this change.

Server paths, credential keys and owner IDs are intentionally not configurable
from the form. The server generates paths and uses the authenticated user ID.
The submitted run keeps its normalized configuration and account order. Adding
accounts later does not change a run that may need to resume. A referenced account
cannot be deleted. Account passwords can be replaced without changing identity.

**Pipeline guide** in the sidebar opens a dedicated reading page at
`/guide/index.html`. **Help** opens the relevant stage in a new tab, preserving
the current form or run view. The contents follow acquisition (openEO or the HUB
catalog) → local statistics → optional feature enrichment → EDA → ML → inspection.
The HUB status message in the Imagery source step links to the HUB acquisition stage. Each stage explains its
inputs, processing order and outputs, followed by a reading list and relevant
workflow diagrams. Articles have an in-page contents list and Previous/Next links.

The guide renders the current Markdown documents, links the HTML workflow references
and displays the PNG diagrams at their relevant stages. The inspection section
includes the detailed evidence/decision-rule diagram and out-of-fold reliability.
Feature enrichment is explicitly identified as Python/notebook functionality;
it is not a separate web-app stage. Historical Word manuals, internal plans and
image-generation prompts are excluded from the learning path.

### Maintaining the pipeline guide

Edit the original Markdown/HTML/PNG in `docs/`. The ordered catalog is
`docs/guide/contents.json`: each section defines its explanation, inputs/outputs,
steps, articles and assets. `web/scripts/build-guide.mjs` generates the pages and
assets under `web/public/guide/`; old `/help/*_workflow.html` URLs are regenerated
as compatibility copies. Generated files are ignored by Git; do not edit them.

`npm run dev`, `npm run build` and `npm test` generate the guide automatically.
After editing docs while the development server is running, run `npm run guide`
from `web/` and refresh the browser. Tests crawl all generated internal links and
anchors, check Markdown rendering and verify workflow assets against the originals.

The frontend Docker build uses the repository root as its context and
`web/Dockerfile.dockerignore` limits the files sent to that build. Compose already
uses this configuration. For a manual frontend-only build, run
`docker build -f web/Dockerfile .` from the repository root. Rebuild the frontend
image to publish documentation changes; no database migration is needed.

## Progress and permanent deletion

Run lists and run details show saved remote-job completion, local batch processing,
and partition assembly separately. Local batches count planned cubes with both
committed statistics and report Parquets, even when completed NetCDFs have been
removed. Partial files, cleaning reports alone and duplicate signatures do not
increase the count. Partition assembly can remain at zero while batches finish.
HUB runs have no remote jobs; they show HUB batches and partition assembly. The
HUB batch total comes from each partition's `hub_item_manifest.parquet`, so it
appears once the catalog is resolved, and a batch counts when its statistics and
report Parquets for the partition's current processing signature are committed.
Remote jobs at 100% does not mean the
download, local processing or final publication has finished. A run is marked
complete only after verified publication. These percentages count work units,
not estimated remaining time; unavailable totals display the current stage.
EDA history shows five processing stages (profiling, charts, tables, report,
ZIP/publication), with 100% only after its artifacts are published. Stage lengths
can differ substantially; disabled optional stages complete immediately.

**Delete** is available in run lists/details and EDA history. Each action opens
a confirmation describing its permanent effects:

- Deleting a run removes its tracked remote openEO jobs/results, including saved
  retry attempts, then its local outputs, logs, caches, EDA analyses and database
  history. The [openEO deletion API](https://api.openeo.org/) stops computation and
  removes remote job data. Cleanup never lists and bulk-deletes an account's jobs.
- The uploaded parcel copy is removed only when no other run references it.
  Copernicus accounts and the user's original source file remain intact.
- Deleting one EDA analysis removes that analysis's report, plots, tables, ZIP,
  logs and configuration; its parent run and sibling analyses remain intact.

Active processing, queued work, EDA and ZIP creation block deletion. Finish the
work first, or cancel an initial queued parcel run. An analysis being deleted
cannot be resumed. Remote progress is checkpointed in PostgreSQL; if credentials,
connectivity or storage prevent cleanup, **Retry deletion** continues safely.
Local files are retained until remote cleanup succeeds. Partial local cleanup
may already have removed some files; the run remains non-resumable until deletion
finishes. Restoring access and retrying is required after such a failure.

The original run folder and its `tile_jobs.parquet` files are needed to identify
remote jobs and owners. Missing or known-incomplete tracking blocks deletion;
restore old-volume files to their original mounted location first. Cleanup uses
the currently configured storage root and the original recorded accounts.

Apply the additive migration before updating the API/worker: it adds
`analysis_runs.deletion_manifest` and `eda_runs.progress`. Existing scientific
results are not changed. No existing run is deleted automatically by deployment.

## Navigating runs, EDA analyses and scenarios

The run lists on Overview and My runs form a tree. A run with EDA analyses shows a
count under its name (for example `2 EDA · 5 scenarios`) and a **▸** toggle that lists
its EDA analyses; an analysis with scenarios has its own toggle that lists them. Select
an analysis or scenario name, or its **Open** button, to go straight to it; an analysis's
**ML** button opens its list of scenarios. The run list (`GET /api/runs`) returns
`eda_count` and `ml_count`, and the EDA list returns each analysis's `ml_count`.

The top bar shows where you are, for example
`Workspace / My runs / <run> / EDA: <analysis> / ML scenarios / <scenario>`. Every earlier
part is a link back to that level. **Back to EDA analysis** on the ML page returns to the
analysis the scenarios belong to.

## Exploratory data analysis (EDA)

The **EDA** action appears beside Open in the run list and in each run's header.
It becomes available only when parcel processing has completed. The server also
checks completion and ownership before exposing columns or accepting an EDA job.

The dialog opens on a start view: **New EDA analysis** opens Setup, and the list of
previous analyses (title, date, column count, live status) opens one analysis's
details. **← All analyses** returns to the start view. Setup reads columns from the registered
final statistics Parquet. Search columns, select all or numeric columns, and
choose exactly which inputs to analyse. Set optional target and ID roles from the
selected columns; numeric class codes should use `target_task=classification`.
Include the geometry column when geospatial summaries are desired.

All analytical/reporting `EDAConfig` options appear in grouped panels. Output
directory, artifact filenames, schema version and `reset_output_dir=False` are
listed under Managed output settings. Each submission receives a separate server
directory; it never clears parcel outputs or an earlier EDA directory. Saved
configurations and selected columns can be reused for another analysis.

An EDA analysis is listed under its **Report title** (default `EDA: <run name>`),
highlighted in the first settings panel.
To tell analyses of the same run apart, select **Rename** in an analysis's details
and enter a name (up to 120 characters). Renaming changes only the display name,
which also becomes the default name of new ML scenarios; the saved settings, the
title inside the HTML report and all results are preserved. Renaming is unavailable
while an analysis is being deleted.

EDA uses the heavy worker queue and continues when the dialog/browser closes.
An analysis's details show queued/running/complete/failed/interrupted status, timestamps,
duration, the saved request and downloads. Results include the HTML report (when
enabled), diagnostic tables, plots, annotated data, configuration and a log.
When an analysis completes while its EDA window is open, the report appears
automatically in the **Report** view. **Open report** in the details opens it directly
in a new browser tab, including for previously completed analyses. These actions
appear only when HTML reporting was enabled. Browser reports embed their plots
and run in a sandbox isolated from application sessions. Ownership checks apply.
The ZIP still includes the original report and plots for offline use: extract it
and open `report.html`. Failed analyses offer a diagnostic log when available;
new failures retain the exception traceback and show its type in the analysis details.

The additive migration creates `eda_runs`, linked to `analysis_runs` and
`app_users`, with display name, input checksum, selected columns, configuration, status,
timestamps and published-file metadata. Existing analyses take their report title as
display name. Apply the migration before starting an
updated worker:

```powershell
.\scripts\run_web_app.bat build
.\scripts\run_web_app.bat migrate
.\scripts\run_web_app.bat start
```

Interrupted/failed EDA analyses are preserved. Use their saved settings to submit
a fresh analysis. EDA proposals are exploratory diagnostics; final feature
selection requires train-only validation, as explained in the
[EDA guide](../eda/README.md).

## ML classification

Every completed EDA analysis has an **ML classification** button in its details. It opens
a page that starts with **New scenario** beside the list of saved scenarios (date, features,
steps done, status); pick one to open it. There you create *scenarios*: one combination of features, models and settings. An EDA
analysis can have any number of scenarios, so different models or search grids can be compared.

1. **Name** the scenario in the highlighted box (default `ML: <EDA name>`). Select **Rename**
   in a scenario's details to change it later (up to 200 characters); settings, steps and
   results are preserved.
2. **Choose the features.** Columns analysed in EDA are listed first and the ones EDA proposed
   are pre-selected. Columns calculated by EDA (for example `is_outlier`) are listed but not
   selected. The remaining parcel-result columns follow. The model input is always the parcel
   result; `is_outlier` is attached for inspection priority even when it is not a feature.
3. **Review the settings.** Every `ClassificationPipelineConfig` option is shown in a group
   with a short explanation and starts at its default. Hyperparameter grids are edited per
   model as JSON in the last group. Settings can be exported and imported as JSON. The web
   app starts from its own defaults for a few settings: spatial split `by_row` (only when the
   result has geometry), the five best models vote (fewer when fewer are selected), and class
   probabilities are optimised over five passes.
4. **Run** all five steps, or only the first one.

Under **Inspection priority ? Thresholds**, **Data reliability** and **Geometry**
have separate controls, mode explanations and worked examples. For data reliability,
**absolute** (default) uses the entered cutoffs; **percentile** uses the smaller of each cutoff
and the reliability score at its chosen percentile in the current prediction run.
For geometry, **absolute** (default) uses complexity cutoffs alone; **percentile** requires both
the complexity cutoff and its percentile rank cutoff. Percentile-only inputs are
disabled in absolute mode. The score-band thresholds remain below these subsections
and only set `inspection_need_base`.

The steps are Check, Prepare, Train, Evaluate and Predict. In a scenario's details each step shows
whether it is done and links to its report. A waiting scenario offers **Run next step** and
**Run all remaining steps**. After a failure or a worker interruption, **Retry step** continues
from the step that stopped; finished steps and already trained models are reused.

An idle scenario can run a later step again with changed settings. The checked data, the
features and the train/test split stay fixed: the settings of Target and parcels, Data
cleaning and Train/test split, the inspection input columns and the Prepare report limits
are locked. Use **Use as a new scenario** to change those.

- **Re-run Train…** (after Prepare) edits the models, their search grids and every setting
  used from Train on, then runs Train, Evaluate and Predict. Each selected model has a
  **retrain** tick box: ticked models are trained afresh, the others reuse their saved
  training. A model is always retrained when it has not been trained yet, when its search
  grid changed, or when a setting shared by all models changed (search candidates, scoring,
  CV ranking, class balancing, rare-category frequency). LSTM settings retrain `keras_lstm`.
- **Re-run Evaluate…** (after Train) keeps every trained model and edits the final model and
  probability tuning, prediction confidence, explanation, inspection priority and output
  settings, then runs Evaluate and Predict. Probability tuning and rank confidence can be
  turned on here only when one of them was already on during Train, because Train saves
  the out-of-fold probabilities they need only then; otherwise re-run Train.
- **Re-run Predict…** (after Evaluate) edits only the inspection priority settings and the
  output column names, then runs Predict.

`POST /api/ml/{id}/rerun` takes `{step, config, retrain}` with `step` one of `train`,
`evaluate` or `predict`. The server rejects locked changes, adds the models that must be
retrained, removes the replaced steps' files from the download list, and queues the steps.
The worker deletes the outputs of the steps that run again and the dashboard (Train's
folder is kept so unticked models are reused) and the folders of the models to retrain
before it starts.

A finished scenario offers the final dashboard, the prediction table (CSV), the full
inspection queue (CSV, every parcel that needs a field check in visit order), a ZIP of all
outputs and the log. ML uses the heavy worker queue, like parcel runs and EDA, so a long
training occupies that queue; it cannot be stopped from the browser. The worker image needs
the `ml` extra (`xgboost`, `lightgbm`, `imbalanced-learn`, `shap`); models whose library is
missing are skipped. `n_jobs` is limited by `PARCEL_MAX_BATCH_WORKERS`.

### Parcel inspection of the best scenario

Once you have compared the scenarios, select **Mark as best scenario** on a finished one.
An EDA analysis has at most one best scenario: marking another moves the mark, and
**Unmark as best** removes it. The best scenario gains a sixth step, **Inspection**; select
**Inspect parcels** (or **Open parcel list** on that step) to open it.

The inspection list shows every parcel with all its prediction columns (except `predicted_class`,
a copy of the configured prediction column such as `crop_prediction`), ordered like the
inspection queue: inspection need first (very high to very low), then inspection score, highest
first. Row colours run from red (needs inspection) to green (does not); grey marks an unknown
need. **Map**, next to the parcel ID, opens a pop-up with the parcel's footprint drawn over a
Sentinel-2 background when it is configured, otherwise Esri satellite (`GET /api/ml/{id}/inspection/geometry?parcel_id=`, read from the parcel
result and returned in WGS84). The background can be switched to Esri satellite or OpenStreetMap.
**Sentinel-2** is true colour from the Copernicus Data Space Sentinel Hub and needs a Sentinel Hub
configuration instance ID: each request uses a random one of the user's Copernicus accounts that has
one (set when adding or editing the account; stored in `eo_users.sen_instance_id`, write-only), which
spreads processing units across accounts, else `PARCEL_SENTINELHUB_INSTANCE_ID`. Every instance needs
a `TRUE_COLOR` layer. The pop-up lists the parcel's Sentinel-2 acquisition dates, oldest
first, within the parcel statistics run's `start_date` to `end_date` (both days included) and
`sentinel2_max_scene_cloud_cover`; granules of the same day count as one date with their lowest
cloud cover (`GET /api/ml/{id}/inspection/sentinel2/dates?parcel_id=`); whether Sentinel-2 is
available, the run's dates and cloud limit and the zoom range come from
`GET /api/ml/{id}/inspection/sentinel2`. It opens on the oldest date; **← Older**, **Newer →**, the
date list and a timeline change it. The timeline spans the run's dates with month labels and a
tick per acquisition date (hover shows its cloud cover); dragging or clicking snaps to the nearest
date and the arrow keys step one date at a time. Images show from zoom 10. The API fetches
the tiles (`GET /api/ml/{id}/inspection/sentinel2/{z}/{x}/{y}.png?date=`), so the instance ID never
reaches the browser; each tile uses Sentinel Hub processing units from the CDSE account's quota. Each row has three editable columns:

- **Checked**: how the parcel was checked: *Not checked* (default), *Satellite image*,
  *Field visit*, or *Satellite image and field visit*.
- **Approved**: tick when the investigation shows the parcel is as the farmer declared
  (default off). Only checked parcels can be approved; setting a parcel back to
  *Not checked* clears it.
- **Comments**: free text up to 2,000 characters, saved when you leave the box.

Changes are saved immediately per parcel, in `ml_inspection_reviews`. Reviews are keyed by
parcel ID, so they survive a re-run of the scenario and remain when it is unmarked; they are
deleted with the scenario. The list is paged 100 parcels at a time and requires the scenario to
have inspection priority columns (`inspection_need`), which Predict writes by default.

`GET /api/ml/{id}/inspection?offset=&limit=` returns one page; `PUT /api/ml/{id}/best` takes
`{best}`; `PUT /api/ml/{id}/inspection/review` takes `{parcel_id, check_status, approved, comment}`.

Run `python -m parcel_app.manage migrate` once to create `ml_runs` (`003_ml.sql`) and the
best-scenario flag and inspection reviews (`004_inspection.sql`).

## Stored history

`analysis_runs` stores each user's draft/submitted request, processing JSON,
scheduling mode, layer/fields, selected account IDs, normalized configuration,
validation revision, summary, status, error and timestamps. `run_accounts` freezes
the submitted identities/order. Related tables are `parcel_uploads`, `web_jobs`,
`run_events`, `run_artifacts`, and `app_sessions`.

The migration SQL files are in [src/parcel_app/migrations/](../../src/parcel_app/migrations/); `migrate` applies them in name order.
Existing CLI run directories are not automatically imported into web history.
All runs created through this application are saved in PostgreSQL.

## Processing and recovery

The run worker handles one heavy analysis at a time. The inspector service handles
uploads, validation and account checks independently, so new drafts remain usable
while imagery processing runs. PostgreSQL advisory locks and shared-filesystem
locks prevent duplicate lane workers and protect orphan child processes.

Closing the browser has no effect on running jobs. Interrupted/failed submitted
runs retain account reservations because their remote jobs may still exist. Resume
the original run to reconcile its checkpoints. Do not edit its uploaded bytes,
paths or saved configuration.

For a running parcel analysis, select **Stop analysis** in its details and confirm.
The button shows **Stopping…** while the supervising worker stops that task and
its local subprocesses. The website and queue service remain available. After
local processes have exited, the run becomes **Interrupted** and **Resume analysis**
is available. Completed checkpoints are preserved; an unfinished stage may repeat.
The worker normally notices a request at its next five-second supervision check;
termination and status refresh can take longer. A stopped worker service cannot
acknowledge new requests until it is restarted.

Stopping is local: already-submitted openEO jobs may continue remotely. Accounts
remain reserved for the original run. Active remote cancellation is not implemented;
**Remove from queue** applies only to never-started queued parcel runs. EDA does not
yet expose this stop action. **Pause tail** pauses log updates only.

Stop requests use the existing queue payload; no database migration is needed.
Deploy the API, worker and frontend together by rebuilding and restarting the app
when active work can be interrupted. Updating only the API cannot make an old
worker honor the new stop request.

Remote JVM out-of-memory failures can now retry with a larger executor heap.
New-run settings under **Tiles and execution** expose `openeo_job_options` as a
JSON object (form default `{"executor-memory":"4G","max-executors":20}`),
`openeo_auto_memory_retry`, `openeo_max_executor_memory` (form default `12G`) and
`openeo_memory_reference_area_km2` (form default 1000; tiles larger than this start
with more executor memory, up to the maximum; blank disables). These form defaults
are larger than the engine defaults used by notebooks and the CLI (2G, 8G, 625). Only
confirmed JVM memory errors increase the heap; generic shuffle failures do not.
The full [retry policy](../data_preparation/openeo/SCHEDULING.md#failure-aware-remote-resources)
explains supported evidence, saved per-batch history and terminal failures.

For an already failed/interrupted run with exhausted memory retries, check
**Allow higher-memory retries for exhausted memory failures**, then select
**Resume analysis**. This grants only higher memory tiers up to its cap, keeps
completed work and original account ownership, and does not change the immutable
configuration snapshot. Ordinary Resume retains the original retry limit. The
owned resume API accepts `{"retry_exhausted_memory":true}`; no database migration
is required for this feature.

Build the updated images while the existing run continues. Before replacing the
worker, allow useful work to finish or stop local processing; already submitted
remote jobs can continue and are reconciled on Resume. If the current deployed
version has no Stop button, `docker compose -f compose.web.yml stop worker`
stops the local run worker. Start the rebuilt services, refresh the UI, then
resume the original interrupted analysis with the recovery checkbox. Do not
duplicate the analysis or manually edit retry counters/configuration files.

The UI shows actual stage events rather than an invented percentage or ETA.
Cleaning, filling and parcel-statistics messages from parallel batch workers are
forwarded to the run process and appear under *local processing*. Some
events can be dropped under excessive log volume; complete sanitized pipeline logs
are published after completion/failure. Native subprocess console output is not
used as a source of downloadable logs. A finished openEO job is not final result
publication: downloads require the runner's verified completion manifest.

## Server configuration

Compose defaults are local-only: `127.0.0.1:8080`, HTTP, and non-Secure cookies for
localhost. Before team deployment, put the app behind HTTPS, restrict it to the
intended private network, set `PARCEL_COOKIE_SECURE=true`, and set
`PARCEL_ALLOWED_ORIGIN=https://your-host`. Change port bindings deliberately;
immediate registration permits anyone who can reach the app to create an account.
The proxy preserves Origin, and the API requires the matching origin plus CSRF
token for authenticated mutations. Do not configure permissive CORS.

Set Compose substitutions in the shell or a separate Compose `--env-file`:

| Variable | Default |
| --- | --- |
| `PARCEL_ENV_FILE` | `./scripts/.env_docker` |
| `PARCEL_HOST_STORAGE` | `parcel_storage` (named volume; set a host folder for a bind mount) |
| `PARCEL_WEB_PORT` | `8080` |
| `PARCEL_ALLOWED_ORIGIN` | `http://localhost:8080` |
| `PARCEL_COOKIE_SECURE` | `false` for localhost |
| `PARCEL_UPLOAD_LIMIT_MB` | `250` |
| `PARCEL_MAX_BATCH_WORKERS` | `4` |
| `PARCEL_WORKER_CPUS` | `4` |
| `PARCEL_WORKER_MEMORY` | `16g` |
| `PARCEL_STORAGE_LIMIT_GB` | `100` |
| `PARCEL_SENTINELHUB_INSTANCE_ID` | empty (only account instance IDs give a Sentinel-2 background on the inspection map); fallback CDSE Sentinel Hub configuration instance with a `TRUE_COLOR` layer |

The nginx upload ceiling is also 250 MiB; update `web/nginx.conf` if increasing
the API ceiling. CPU/memory controls are per service. Storage limits gate new
uploads/submissions; allocate and monitor sufficient filesystem capacity because
processing can grow during an active run. Failed jobs leave checkpoints intact.
Multiple independently launched CLI jobs are outside the application's account
reservation system and should not run concurrently with the same accounts.

## Development and verification

Install the `web`, `database`, `credentials`, `acquisition`, and `interpolation`
extras in a project virtual environment. Use the same PostgreSQL environment
variables and a private `PARCEL_STORAGE_ROOT`. Start the API with
`python -m uvicorn parcel_app.app:app --port 8000`, workers with
`python -m parcel_app.jobs.worker --lane run` and `--lane small`, and frontend with
`npm ci` / `npm run dev` under `web/`. Set the allowed origin to the Vite URL and
disable Secure cookies only for localhost development.

Tests in `tests/test_parcel_app/` default to skipping database integration unless
`PARCEL_TEST_DB=1` is set with **disposable PostgreSQL credentials**. Run migrations
there first. `scripts\run_code_checks.bat db` does this automatically: it starts a throwaway
Docker PostgreSQL, points every database setting at it, migrates, runs all tests and removes it. They cover registration, isolation, CSRF, uploads, validation,
submission idempotency, worker fencing, output redaction, synthetic real local
statistics, authorized downloads/bundles and checkpoint resume. They do not submit
live openEO work. Frontend: `npm test` and `npm run build`.

See [the design](DESIGN.md) and [implementation contracts](IMPLEMENTATION_API.md).
