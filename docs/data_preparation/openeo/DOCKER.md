# Parcel statistics CLI container

The root [Dockerfile](../../../Dockerfile) packages the existing CLI in a Linux
Python 3.12 image. It includes openEO acquisition, PostgreSQL access, encrypted
credentials and kriging support. It runs as UID/GID `10001:10001`; no database,
credentials, input data or run results are baked into the image.

## Build

Run from the repository root with Docker's Linux container engine available:

```powershell
docker build --pull -t satellites-parcel-stats:local .
docker run --rm satellites-parcel-stats:local
```

The default command prints parcel-statistics help. Image construction checks
dependency consistency and imports the real processing runner. The builder
installs the package from `pyproject.toml`; the final stage contains the installed
environment, without build sources or development/notebook extras. Dependencies
must have binary wheels for the chosen platform. The base image tag and project
dependencies are not a complete version lock; preserve the built image by digest
when deploying or resuming production runs.

## Windows batch launcher

Use [run_docker_process.bat](../../../run_docker_process.bat) to launch the image.
Edit the settings at the top: `INPUT_FOLDER`, `OUTPUT_FOLDER`, `CONFIG_FILE`,
`RUN_NAME`, `APP_USER_ID`, and the CPU/memory limits. Build the image once using
the command above, then start Docker Desktop before running the launcher.

Create `.env_docker_dev` beside the batch file with your actual values:

```dotenv
POSTGRES_HOST=host.docker.internal
POSTGRES_PORT=5432
DB_NAME=satellite_accounts
POSTGRES_USER=parcel_worker
POSTGRES_PASSWORD=replace_with_database_password
SATELLITES_CREDENTIAL_KEY=replace_with_existing_fernet_key
```

Use `NAME=value` lines without enclosing quotes. Use the encryption key that was
used to store the Copernicus passwords. This local credentials file is ignored
by Git and excluded from the Docker build context. Optional table overrides are
`TBL_APP_USERS` and `TBL_EO_USERS`.

From PowerShell in the repository root:

```powershell
.\run_docker_process.bat validate
.\run_docker_process.bat
.\run_docker_process.bat resume
```

Double-clicking the file also starts a run. The window pauses at the end;
set `PAUSE_AT_END=0` for automation. The script returns the container's exit code
and reports failures. Validation runs without network access or the environment
file. Run and resume load only the selected application user's Copernicus accounts.

The launcher overrides JSON `output_dir` with `/data/runs/<RUN_NAME>`; results
remain under `<OUTPUT_FOLDER>\<RUN_NAME>` on Windows. Change `RUN_NAME` for a new
run, or keep it and use `resume` to continue an existing run. Keep the same mounts
and original input files for resume. `CONFIG_FILE` and `RUN_NAME` should be simple
filenames without spaces or shell special characters. Host folder paths may
contain spaces. The script creates the output folder when needed.

## Prepare mounted files

Use separate host directories for configuration/input and persisted runs:

```text
C:/parcel-app/input/
  config.json
  parcels.geoparquet
C:/parcel-app/runs/
```

In `config.json`, use container paths or paths relative to that configuration:

```json
{
  "schema_version": 1,
  "input": {
    "path": "parcels.geoparquet",
    "parcel_id_field": "parcel_id"
  },
  "output_dir": "/data/runs/run-001",
  "processing": {
    "start_date": "2024-01-01",
    "end_date": "2024-01-31",
    "sentinel2_bands": ["B04", "B08"],
    "sentinel2_indices": ["NDVI"],
    "spatial_statistics": ["mean", "median"],
    "fit_tiles_to_parcels": true,
    "batch_workers": 1,
    "remove_nc_after_completion": true
  }
}
```

This example estimates a UTM working CRS from the parcels. Review the resolved
CRS and tile count with `validate` before running. For the full configuration,
see the [CLI guide](CLI.md) and [example JSON](cli.example.json).

## Validate offline

These commands use PowerShell continuation syntax. The host directories must
already exist. `validate` does not require credentials or connect to openEO.

```powershell
docker run --rm --init `
  --mount "type=bind,source=C:/parcel-app/input,target=/data/input,readonly" `
  satellites-parcel-stats:local validate --config /data/input/config.json
```

## Run with PostgreSQL accounts

Set the same database and encryption environment variables as for the CLI in the
host shell or deployment secret manager. Docker's `--env NAME` passes the existing
host value without putting its literal contents in the command. For example,
`POSTGRES_HOST` is the reachable database hostname, not a Windows path:

```powershell
$env:POSTGRES_HOST = "host.docker.internal"
$env:POSTGRES_PORT = "5432"
$env:DB_NAME = "satellite_accounts"
$env:POSTGRES_USER = "parcel_worker"
# Supply POSTGRES_PASSWORD and SATELLITES_CREDENTIAL_KEY securely in this shell.

docker run --rm --init --stop-timeout 60 `
  --mount "type=bind,source=C:/parcel-app/input,target=/data/input,readonly" `
  --mount "type=bind,source=C:/parcel-app/runs,target=/data/runs" `
  --env POSTGRES_HOST --env POSTGRES_PORT --env DB_NAME --env POSTGRES_USER `
  --env POSTGRES_PASSWORD --env SATELLITES_CREDENTIAL_KEY `
  satellites-parcel-stats:local run --config /data/input/config.json --app-user-id 42
```

`host.docker.internal` addresses the host on Docker Desktop. On a Linux Docker
server, use the database's network hostname; if PostgreSQL runs on that same host,
add `--add-host host.docker.internal:host-gateway` to use the example hostname.
`localhost` inside a container means the container itself. PostgreSQL must permit
connections from the Docker network.

The default tables are `public.app_users` and `public.eo_users`. Pass
`--env TBL_APP_USERS --env TBL_EO_USERS` when overriding those names. Account
passwords and the encryption key use the [existing database contract](CLI.md#postgresql-users-and-copernicus-accounts).
The image does not apply the SQL schema or provision users. `--app-user-id` is an
operator selection, not web authentication.

For the legacy notebook account table, replace `--app-user-id 42` with
`--legacy-accounts-db` and pass `--env TBL_USERS`. That path uses the existing
legacy password format and does not require the Fernet key.

On Linux hosts, ensure mounted directories are readable by UID 10001 and the runs
directory is writable by UID 10001. Alternatively, run with your own numeric
UID/GID using `--user "$(id -u):$(id -g)"` in a Linux shell. A bind mount's host
permissions override ownership set in the image. The temporary directory and
Matplotlib cache use `/tmp`.

## Resume and retrieve output

Repeat the run command with the same mounts and environment, replacing its final
arguments with:

```text
satellites-parcel-stats:local resume --run-dir /data/runs/run-001 --app-user-id 42
```

The container paths must remain unchanged: saved configurations and remote job
records contain absolute paths. A run started with Windows paths outside Docker
cannot be resumed directly inside this Linux image. Start a new container run
and retain its paths for subsequent resumes.

Results are available directly under `C:/parcel-app/runs/run-001` on the host,
including `artifacts.json` and the final GeoParquet. Removing a container with
`--rm` does not remove bind-mounted results. Do not keep checkpoints only in the
container's writable layer. Stopping a container preserves completed checkpoints
but may leave remote openEO jobs running; use `resume` to reconnect.

Choose CPU, memory and disk limits for your workloads. Start with one application
run at a time and `batch_workers=1`; independent containers do not coordinate
shared Copernicus account quotas. No HTTP port is exposed because this image
currently runs the CLI worker, not the future web application.

The build uses an allowlisted [.dockerignore](../../../.dockerignore) context;
see Docker's documentation on [build contexts](https://docs.docker.com/build/concepts/context/)
and [image build practices](https://docs.docker.com/build/building/best-practices/).
