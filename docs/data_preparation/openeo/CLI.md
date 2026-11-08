# Parcel statistics CLI and dictionary runner

Run the existing openEO acquisition and parcel statistics pipeline without a
notebook. This entry point supports multiple Copernicus accounts, offline
validation, durable run metadata and checkpoint-based resume. It does not start
a web server or authenticate application logins.

For a packaged Linux runtime, see [Docker build and execution](DOCKER.md).

## Installation and commands

Use the project's existing environment, or create a dedicated Python 3.12+
virtual environment. Install the package and required extras into that environment:

```powershell
python -m pip install -e ".[acquisition,database,credentials]"
satellites parcel-stats --help
```

The `credentials` extra supplies Fernet encryption/decryption for the new
application-owned account table. Add `interpolation` for kriging. Existing
environments that already contain every dependency can register the command with
`python -m pip install --no-deps --no-build-isolation -e .`.

Copy [cli.example.json](cli.example.json) alongside your parcel file and adapt its
paths, CRS, dates and scientific settings. The example's EPSG:32634 is a regional
choice, not a global default.

```powershell
satellites parcel-stats validate --config C:/work/config.json
satellites parcel-stats run --config C:/work/config.json --app-user-id 42
satellites parcel-stats resume --run-dir C:/work/run --app-user-id 42
```

The equivalent module invocation is `python -m data_preparation.cli parcel-stats ...`.
`validate` never connects to PostgreSQL or openEO. It reads the parcel input and
plans all tiles using temporary files; it does not create the requested output
directory. It reports the normalized configuration and counts of parcels,
partitions and remote batches. Runtime authentication, quota availability and
remote data availability are not tested by offline validation.

CLI stdout contains one JSON summary/manifest on success. Progress goes to stderr
and the existing pipeline log files. Exit codes: `0` success, `2` command/setup or
validation-command errors, `1` run/resume failures, `130` interruption. Secrets
are never accepted as command-line arguments. `.env` is not loaded automatically.

## Configuration contract

| Section/key | Meaning |
| --- | --- |
| `schema_version` | Required integer `1` |
| `input.path` | Required local GeoParquet, GeoPackage or GeoJSON path |
| `input.parcel_id_field` | Required existing unique, non-null ID column |
| `input.layer` | Optional GeoPackage layer name |
| `input.partition_field` | Optional existing grouping column, such as `cluster` |
| `output_dir` | Required dedicated run directory, initially empty |
| `run_identifier` | Optional label of 1–60 characters; otherwise generated |
| `scheduling` | `batches` (default) or `partitions`, as in the existing scheduler |
| `processing` | Existing constructor options described below |

Relative input/output paths resolve against the JSON file's directory. The CLI
`--output-dir` override resolves against the current working directory. For Python
dictionaries, paths resolve against `base_dir` or the current working directory.
Inputs must remain outside the output directory.

IDs are converted to strings and must remain unique. Geometry repair reuses the
existing pipeline, but the CLI rejects inputs that would lose empty or
non-polygon features. It does not invent IDs or silently discard parcels. A
partition column must have a value for every parcel. Without one, all parcels
form one outer partition and the tile planner still creates multiple remote
batches as needed. The CLI does not generate buffer-cluster labels; prepare that
column upstream when reproducing a clustered notebook workflow.

`processing` accepts the named options of `ParcelStatsBase` and
`OpenEOJobManagerZonalStats`, preserving their defaults and validation:

- `start_date` and `end_date` are required ISO calendar dates.
- `working_epsg` is a projected EPSG code. If omitted, the existing UTM estimator
  selects one from the parcel footprint; inspect the resolved value before a run.
- Sensor bands, indices, orbit direction, temporal periods/reducers and spatial
  statistics use the [existing parameter guide](README.md#common-constructor-parameters).
- All IQR, filling, interpolation, quality-threshold and checkpoint-retention
  options remain available. JSON arrays replace Python tuples.
- Tile size, buffer, fitted extents, local workers, remote slots, polling and
  retry controls use the [manager options](README.md#job-manager-tile-workflow).
  Remote concurrency is limited to 1 or 2 jobs per account by this multiuser API.
- `parcels`, credentials, `output_dir`, `run_identifier`, `parcel_id_field`, and
  `resume_completed_partitions` are not processing options in this wrapper.
  Their values come from the request, credential provider and runner.

Unknown fields are rejected. The temporal reducer retains the engine's existing
date-extent semantics; see the [temporal caveat](README.md#temporal-and-statistic-parameters).
Selecting `median` in spatial statistics enables meaningful per-partition annual
summary features. The combined download is the full dated time-statistics table.

## PostgreSQL users and Copernicus accounts

The new schema is provided in
[001_application_accounts.sql](../../../scripts/sql/001_application_accounts.sql).
Apply it explicitly to the intended database using an administrative migration
role. The CLI never creates or alters tables. The script creates:

```text
app_users
  app_user_id  bigint primary key
  username     text, unique ignoring case
  password     text, application password hash
  is_active    boolean, default true

eo_users
  eo_user_id   bigint primary key
  app_user_id  foreign key -> app_users.app_user_id
  username     text, unique ignoring case within its application owner
  password     text, Fernet-encrypted Copernicus password
```

The original `TBL_USERS` account table and its notebook loader are unchanged.
The application login and account-management screens will provision these new
rows in the web milestone. Store a password hash (for example Argon2id) in
`app_users.password`, never plaintext. The CLI does not read that password column.

Set `DB_NAME`, `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_USER` and
`POSTGRES_PASSWORD` as in the [database guide](CREDENTIALS.md). Additionally:

| Environment variable | Meaning |
| --- | --- |
| `TBL_APP_USERS` | Defaults to `public.app_users` |
| `TBL_EO_USERS` | Defaults to `public.eo_users` |
| `SATELLITES_CREDENTIAL_KEY` | Fernet key used to encrypt/decrypt `eo_users.password` |

Generate the key once with `cryptography.fernet.Fernet.generate_key()` and store it
in the deployment's secret store, separate from PostgreSQL and version control.
Provisioning code encrypts an exact password string as follows:

```python
import os
from cryptography.fernet import Fernet

cipher = Fernet(os.environ["SATELLITES_CREDENTIAL_KEY"].encode("ascii"))
encrypted_password = cipher.encrypt(password.encode("utf-8")).decode("ascii")
# Insert encrypted_password using a bound SQL parameter; never log it or password.
```

Preserve the key across deployments and backups. Changing it without re-encrypting
stored credentials makes those accounts unreadable. Passwords are decrypted only
in worker memory. Plaintext legacy rows are not accepted in `eo_users`.

The scoped loader joins active application users to their accounts and binds
`app_user_id` as a SQL parameter. It never falls back to all users. Accounts are
sorted by normalized username. A disabled/missing owner or an empty account list
fails before any openEO submission. App-user deactivation is checked at credential
loading time, not continuously during an already-running extraction.

**The CLI is a trusted operator interface.** `--app-user-id` selects an owner; it
does not prove that the person invoking the CLI owns that ID. A future web API must
derive the ID from its authenticated session, authorize file/run access, and never
accept a client-supplied owner as authorization. Give only trusted operators access
to the database login and encryption key.

For existing notebook infrastructure, explicit legacy operation is available:

```powershell
satellites parcel-stats run --config C:/work/config.json --legacy-accounts-db
```

This reads all accounts in `TBL_USERS`, exactly like the existing notebook. It is
mutually exclusive with `--app-user-id` and is never selected implicitly.

## Python dictionary entry point

```python
from data_preparation.parcel_stats import prepare_run, run_parcel_stats
from data_preparation.parcel_stats.accounts import load_app_openeo_accounts

# config is a dictionary with the same structure as cli.example.json.
summary = prepare_run(config, base_dir="C:/work").summary()
accounts = load_app_openeo_accounts(app_user_id=42)
manifest = run_parcel_stats(config, accounts=accounts, base_dir="C:/work", app_user_id=42)
final_path = manifest["artifacts"]["time_statistics"]
```

Trusted callers can instead supply a list of `(username, password)` pairs directly.
Do not include those credentials in the processing dictionary. Use a normal
`if __name__ == "__main__":` guard when invoking from scripts that enable local
multiprocessing. `resume_parcel_stats(run_dir, accounts=..., app_user_id=42)` is the
corresponding Python restart entry point.

## Outputs and restart

Each run writes resolved `config.json`, durable `run.json`, partition outputs,
existing pipeline logs, and, only after verification, `artifacts.json`. The latter
identifies the final `satellite_parcel_time_stats.geoparquet`, combined cleaning
CSV, parcel count and output checksum. Per-partition annual files remain available;
this version does not merge them into a root annual file.

Completion requires every expected partition's signature, both GeoParquets and
cleaning report, correct projected CRS and exact ID coverage. The final merge does
not hide duplicates. The merged GeoParquet is staged and atomically renamed.
Consumers should require both run state and artifact manifest to say `complete`;
the presence of a GeoParquet or a finished remote job alone is insufficient.

Resume requires the same input file bytes, saved configuration, application owner
and account usernames/order. Passwords may be rotated. Do not edit `config.json`
or relocate a run: existing remote-job records contain absolute paths. For changed
parameters or inputs, use a new output directory. New runs refuse nonempty output
directories; they never reset existing work.

An OS file lock prevents simultaneous writers in the same run directory and is
released if the process exits. The small lock file remains intentionally. A process
crash can leave status `running`; rerun the resume command after the old process has
stopped. The engine reuses complete checkpoints and rebuilds unfinished stages.
Ctrl+C may leave remote jobs running; this milestone does not implement remote
cancellation. Do not rename `.partial` files or manually mark jobs complete.

Run only one application job at a time against a shared Copernicus account pool.
Locks protect a run directory, not global account quotas across separate runs.
The future web worker must coordinate shared account capacity and enforce disk,
memory, upload and retention limits. Web authentication is a separate milestone;
the [Docker image](DOCKER.md) packages this CLI for worker execution.
