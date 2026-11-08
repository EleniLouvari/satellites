# Parcel statistics CLI implementation

Build the approved first milestone: a dictionary-based Python runner and a CLI
for validation, execution and restart of the existing openEO parcel pipeline.
The web login, database migrations, Docker image and account-management UI are
later milestones. Application users and their Copernicus accounts will live in
PostgreSQL, with accounts linked to an application user ID.

## Contract

- JSON configuration contains `schema_version`, `input`, `processing`,
  `output_dir`, optional `run_identifier`, and optional `scheduling`.
- Input supports GeoParquet, GeoPackage and GeoJSON; an explicit unique ID
  column is required. An optional partition column groups whole parcels.
- Processing uses the existing constructor parameters and validators; no
  acquisition, cleaning or statistics algorithms are duplicated.
- Credentials are supplied separately, in memory or from PostgreSQL. CLI user
  selection is an operator facility, not application authentication.
- Validation is offline and does not create the configured output directory.
- New runs require an empty output directory. Resume uses saved configuration,
  identical input bytes and the same account identities/order.
- Each run has an exclusive process lock, saved configuration and status.
- Completion requires every partition and exact parcel ID coverage; the final
  GeoParquet is published atomically with an artifact manifest.
- Existing notebooks, the legacy database loader and pipeline APIs stay compatible.

## Implementation tasks

1. Add tests for JSON validation, offline preflight, multi-account execution,
   incomplete results, restart guards, locking and command exit codes.
2. Add `parcel_stats/run_config.py` for input loading and existing-engine preflight.
3. Add `parcel_stats/runner.py` for run state, execution and verified publication.
4. Add `data_preparation/cli.py` and register the `satellites` console command.
5. Add an explicitly user-filtered PostgreSQL credential-loading path without
   changing the existing unfiltered notebook API; use parameterized values.
6. Document configuration, credentials, examples and operational limits.
7. Run focused tests, parcel pipeline regression tests and appropriate static
   checks. Review the complete change before reporting completion.

## Review focus

- No successful download manifest for missing or duplicated parcel results.
- No secrets persisted in configuration, state, errors or manifests.
- No overlapping writers or reuse of a different input/configuration.
- No remote requests during validation or test runs.
- No cross-user account loading in the explicitly scoped database path.

## Implementation record

- Added and tested all three CLI commands and the dictionary API.
- Added the requested `app_users` / `eo_users` schema as an explicit SQL migration;
  it has not been applied to a live database. `eo_users.app_user_id` is the foreign
  key, app passwords are hashes, EO passwords are encrypted, and `is_active`
  permits an administrator to disable an application user.
- Preserved the existing notebook loader through an explicit legacy CLI option.
- Independent review identified a noncanonical geometry-column failure and an
  ID-column whitespace mismatch. Reproduced both with failing tests, then fixed
  preflight normalization/rejection. Added lifecycle and error-redaction tests.
- Documented the command-based exit-code distinction: run/resume errors return 1;
  parser/setup and validation-command errors return 2.
- Verified the real scheduler/local engine using synthetic NetCDF downloads;
  no live PostgreSQL or openEO requests were made.
- Installed the editable package without dependencies or network access into the
  existing `.venv-checks` environment and smoke-tested the console entry point.
- Quality notebook cells executed with UTF-8 output without modifying the notebook.
- Web login, account provisioning UI, cross-run account quotas and live service
  verification remain subsequent deployment/application work, as documented.
