# Database-backed openEO accounts

The Neuro `1_parcel_stats_openEO_fill.ipynb` notebook calls
`data_preparation.parcel_stats.multiuser.load_openeo_users_from_db()` before
planning extraction. It reads CDSE account credentials from PostgreSQL using
`shared.postgress.create_db_handler`, then returns a list of `(username, password)`
tuples for `run_parallel_extractions`. The former `load_openeo_users_from_env`
function and semicolon-separated `OPENEO_USERS` setting are no longer used.

## Install and configure

In the notebook's Python environment, install the database dependency group along
with the acquisition dependencies:

```powershell
python -m pip install -e ".[acquisition,database,notebooks]"
```

The `database` extra provides `psycopg2-binary` and SQLAlchemy. Existing geospatial
environments using the README's `--no-deps` installation must already have these
packages installed. Database drivers are imported only when the loader is called.

Set these environment variables in the Python process or before starting Jupyter:

| Variable | Meaning | Default |
| --- | --- | --- |
| `DB_NAME` | PostgreSQL database containing the account table | Required, non-blank |
| `TBL_USERS` | Account table, optionally schema-qualified, such as `public.openeo_accounts` | Required, non-blank |
| `POSTGRES_HOST` | Database server | `localhost` |
| `POSTGRES_PORT` | Database port | `5432` |
| `POSTGRES_USER` | Database login, separate from CDSE usernames | `postgres` |
| `POSTGRES_PASSWORD` | Password for the database login | Empty string |

For example, set the non-secret connection settings in PowerShell:

```powershell
$env:DB_NAME = "satellite_accounts"
$env:TBL_USERS = "public.openeo_accounts"
$env:POSTGRES_HOST = "localhost"
$env:POSTGRES_PORT = "5432"
$env:POSTGRES_USER = "openeo_reader"
```

Provide `POSTGRES_PASSWORD` through your local environment or secret injection.
The loader and notebook do **not** automatically read `.env`. If you maintain
settings there, configure your launcher to load them into the kernel environment;
simply creating or editing the file does not update a running kernel.

## Account table contract

`TBL_USERS` must resolve to a non-empty table or view with these columns:

| Column | Requirement |
| --- | --- |
| `username` | Non-null, non-blank text; unique after trimming surrounding whitespace and ignoring case |
| `password` | Non-null, non-blank text containing the actual CDSE password |

Additional columns are ignored. Every row is loaded; an `active` column does not
automatically filter accounts. To select accounts, point `TBL_USERS` at a view
containing only the intended rows. PostgreSQL access needs `SELECT` on that table
or view and access to its schema; the loader does not create or update records.

The shared `create_tbl_users` helper creates a different, product-oriented table
with blank placeholder credentials and one row per `product_type`. Those rows
cannot be passed directly to this loader unless they form a valid set of unique
CDSE accounts. Use a dedicated account table or a suitable view when the same
login is repeated across products or the table also contains other providers.

Usernames are stripped of surrounding whitespace. Passwords are returned exactly
as stored, including punctuation and leading/trailing spaces; they are never split
on commas or semicolons. Password hashes cannot authenticate to CDSE. Restrict
access to this table and do not print or save `users_list` in notebook output.
The loader prints only the account count and closes its database connection after
reading, including when the read fails.

```python
from data_preparation.parcel_stats.multiuser import load_openeo_users_from_db

users_list = load_openeo_users_from_db()
```

These database account records are separate from the local scheduler's
`tile_jobs.parquet` files. Job files persist the owning `openeo_user`, job IDs and
status, but never the passwords.

## Account ordering and restarting jobs

The loader sorts accounts by case-insensitive username so database row order
does not change scheduling. For shared batch runs with persisted `openeo_user`,
keep the same username spelling and all accounts that own unfinished jobs.

For legacy job files without saved usernames, or for `scheduling="partitions"`,
restore the **original account list and order** before resuming. Alphabetical
order may differ from the old `OPENEO_USERS` order. Reorder the loaded tuples
explicitly using the account names from the original run, for example:

```python
original_usernames = ["second@example.com", "first@example.com"]
accounts_by_name = dict(users_list)
users_list = [(name, accounts_by_name[name]) for name in original_usernames]
```

Use the actual original list; do not infer job ownership from current database
row order. Keep partition order and scheduling mode unchanged as described in
[the scheduling guide](SCHEDULING.md).

## Troubleshooting

- Missing `DB_NAME` or `TBL_USERS`: set both in the kernel environment.
- Connection failure: check `POSTGRES_*`, database availability and login access.
- Table read failure: check the table/schema name and read permissions.
- Empty table, missing columns, blank/non-text credentials, or duplicate usernames:
  correct the account records; the loader rejects the whole list before submission.
- Missing `psycopg2` or `sqlalchemy`: install the `database` extra in the active kernel.
- CDSE authentication failure after loading: database access only retrieves values;
  verify that each account/password is valid for the configured openEO provider.
