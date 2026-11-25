# Parcel Stats Studio

## v0.0.1 - 2026-11-23

### Added

- HUB parcel inspection previews in the Map dialog, with native-resolution RGB
  imagery, Sentinel-1 false colour and configured index layers on an acquisition
  timeline.
- Shared preview palettes and legends for supported Sentinel-1 and Sentinel-2
  indices, plus lazy in-memory band-window caching and private browser caching.
- Same-source HUB preview coverage, source and ownership checks, revision-aware
  image URLs, stale-request protection and safe upstream error handling.
- Bounded parcel preview windows with native-grid mosaicking and accurate
  EPSG:3857 placement, keeping the parcel boundary above the imagery.
- HUB Sentinel ARD parcel extraction through `HubZonalStats`, dictionary-configured
  runs, and Parcel Studio's imagery source selector, supporting Sentinel-2
  `ARD_S2` and Sentinel-1 `ARD_S1` collections.
- HUB catalog discovery, authenticated raster-window reads, sensor-specific
  masking, spatial alignment and mosaicking, with bounded asset-read concurrency.
- HUB inventory manifests and completion provenance for resume checks, including
  Sentinel-1 gamma0 calibration metadata distinct from openEO sigma0 features.
- Server-managed HUB credentials and configuration status, source-specific web
  controls and progress, and an additive database migration preserving existing
  runs as openEO.
- A HUB workflow guide, example notebook, and regression coverage for acquisition,
  shared result assembly, runner configuration, web lifecycle and source controls.
- A "HUB catalog acquisition" stage in the in-app pipeline guide, with the HUB
  library guide as its article, and a link to it from the HUB imagery source step.

### Changed

- HUB parcel extraction processes completed batch cubes while later batches are
  acquired, using up to `batch_workers` isolated local processes. Pending work is
  bounded, batch statistics and cleaning reports remain resumable, and NetCDF
  cleanup can run before acquisition finishes. Worker logs reach web run events,
  including with one local worker.
- Parcel result assembly is shared by the openEO and HUB adapters, retaining
  dated and annual GeoParquet outputs and cleaning reports.
- Extraction edits, duplicates, resumes and deletion use the saved imagery source;
  HUB runs use local cleanup without Copernicus remote-job operations.
- Dictionary-configured run failures append exception details and chained causes
  to the run-level `satellite_zonal_stats.log`, available in Parcel Studio run
  events and failure diagnostics; `run.json` retains only the exception type.
- Setup and architecture documentation now cover HUB deployment, concurrency,
  inventory-aware resume, and the distinction between HUB and openEO radar data.

### Fixed

- HUB run progress counts committed batches from each partition manifest
  instead of only completed partitions, so a single-partition run no longer reads
  "0 of 1 completed" while its batches are being processed.

## v0.0.1 - 2026-11-21

### Added

- Shared openEO download recovery for completed jobs: up to three transfer
  attempts with 30/60-second retry delays, scoped connection/read-idle timeouts,
  byte-progress logs, and atomic publication of completed NetCDF files.
- Regression tests for interrupted and truncated downloads, retry limits,
  parcel block-read equivalence and edge cases, and web extraction defaults.

### Changed

- Parcel statistics now read nearby parcel windows in shared spatial blocks,
  reducing repeated checkpoint reads while preserving independent parcel masks,
  existing reducers, provenance counts, and input row order.
- Block reads target a 64 MiB payload with a separate 16 MiB parcel-mask budget.
  Oversized groups split, and oversized individual parcels run alone; these
  budgets do not cap total process memory.
- New web extraction configurations default to one local worker, one remote job
  per account, 5,000 m fitted tiles, and automatic temporal-fill block sizing.
  The default working CRS is EPSG:2100 and temporal-fill windows are `[3, 5]`.
  Saved and resumed runs retain their explicit settings.
- Processing guides now explain block-read memory, download recovery, batch
  versus partition scheduling, and the new-run web defaults.

### Fixed

- Interrupted, empty, and length-mismatched asset transfers cannot publish a
  completed NetCDF filename. Transient retries reuse the completed remote job
  without resubmitting satellite processing or multiplying SDK transport retries.
- Shared batch scheduling passes the acquisition logger to the download manager
  so transfer progress is recorded consistently.
