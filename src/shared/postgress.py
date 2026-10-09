"""A lightweight utility around `psycopg2` and `SQLAlchemy` for common database tasks.

Tasks: connecting, creating/deleting databases and tables, inserting/updating/deleting records,
reading/writing (Geo)Pandas DataFrames, and creating basic triggers. Geometry-aware
operations are supported when PostGIS is available.

This module favors explicit, side-effect-free operations where possible and prints
succinct informational messages for administrative actions (e.g., database creation).

"""

import os
import re

import geopandas as gpd
import pandas as pd
import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from psycopg2.extras import Json as psycopg2_json
from psycopg2.extras import RealDictCursor
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SQL_DB_EXISTS = "SELECT 1 FROM pg_database WHERE datname = %s;"
_SERIAL_PRIMARY_KEY = "SERIAL PRIMARY KEY"
_UNIQUE_CHAR = "VARCHAR UNIQUE NOT NULL"
_VARCHAR = "VARCHAR"


def _database_exists(cursor, db_name: str) -> bool:
    """Return True if a database with the given name exists."""
    cursor.execute(_SQL_DB_EXISTS, (db_name,))
    return cursor.fetchone() is not None


def _validate_identifier(name: str, *, allow_dot: bool = False) -> None:
    """Validate an identifier to reduce accidental injection in DDL paths.

    When allow_dot=True, allow schema-qualified names like "public.table".
    """
    if allow_dot:
        parts = name.split(".")
        if not parts or any(not _IDENTIFIER_RE.match(p) for p in parts):
            raise ValueError(f"Error: Invalid schema-qualified identifier: {name}")
    else:
        if not _IDENTIFIER_RE.match(name or ""):
            raise ValueError(f"Error: Invalid identifier: {name}")


class PostgresPostGISHandler:
    """
    High-level helper for PostgreSQL/PostGIS administration and CRUD.

    The handler opens a connection on initialization (unless it fails) and exposes
    convenience methods that cover frequent workflows in EO/Geo projects: creating
    databases from a template, managing tables (optionally with a PostGIS geometry
    column), streaming records, and round-tripping (Geo)Pandas DataFrames.
    """

    def __init__(
        self,
        db_name: str = "postgres",
        user: str | None = None,
        password: str | None = None,
        host: str = "localhost",
        port: int | str = 5432,
    ):
        """Initialize the handler and immediately attempt a connection.

        The constructor stores the given parameters and calls :meth:`connect` so the
        instance is ready for use. If the connection fails, the exception is raised.

        Parameters
        ----------
        db_name : str, optional
            Database to connect to. Defaults to "postgres".
        user : str, optional
            Username to authenticate with.
        password : str, optional
            Password for the user.
        host : str, optional
            Database server hostname or IP. Defaults to "localhost".
        port : int | str, optional
            TCP port. Defaults to 5432.

        Raises
        ------
        Exception
            Propagated from :meth:`connect` if the underlying connection fails.

        """
        self.db_name = db_name
        self.template = "postgis_35_sample"
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.connection: psycopg2.extensions.connection | None = None
        self.db_col_geom = "geom"

        # Initialize the connection
        self.connect()
        if self.connection is None:
            raise RuntimeError("Error: Failed to establish database connection")

    @property
    def _conn(self) -> psycopg2.extensions.connection:
        """Return the active connection or raise if not connected.

        Returns
        -------
        psycopg2.extensions.connection
            The active database connection.

        Raises
        ------
        RuntimeError
            If no connection is established.

        """
        self._ensure_connection()
        if self.connection is None:
            raise RuntimeError("Error: No active database connection")
        return self.connection

    def _ensure_connection(self) -> None:
        """Ensure a live database connection exists.

        Reconnect when no connection is set or when the current psycopg2
        connection is already closed/stale.

        Returns
        -------
        None

        Raises
        ------
        Exception
            If reconnecting fails.

        """
        if self.connection is None:
            self.connect()
            return

        # psycopg2: 0 means open, non-zero means closed.
        if int(getattr(self.connection, "closed", 1)) != 0:
            self.connect()

    def _run_with_reconnect_retry(self, operation, operation_name: str, retries: int = 1):
        """Run an operation and retry once after reconnect on transient DB connection errors.

        Parameters
        ----------
        operation : Callable
            Zero-argument callable that executes a DB action.
        operation_name : str
            Human-readable operation label for error messages.
        retries : int, optional
            Number of reconnect retries for transient connection failures.

        Returns
        -------
        Any
            Return value produced by ``operation``.

        Raises
        ------
        RuntimeError
            If retries are exhausted due to connection errors.

        """
        attempt = 0
        while True:
            try:
                return operation()
            except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
                if attempt >= retries:
                    raise RuntimeError(f"Error: Failed to {operation_name}: {e}") from e
                attempt += 1
                self.close_connection()
                self.connect()

    def connect(self):
        """Establish a connection to the PostgreSQL database.

        Returns
        -------
        None

        Raises
        ------
        Exception
            If the underlying `psycopg2.connect` call fails with an OperationalError.

        """
        try:
            self.connection = psycopg2.connect(
                dbname=self.db_name, user=self.user, password=self.password, host=self.host, port=self.port
            )
        except psycopg2.OperationalError as e:
            raise ConnectionError(f"Error: Connection failed: {e}") from e

    def close_connection(self):
        """Close active database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None

    def _get_column_types(self, table_name: str):
        """Return a mapping of column names to their data types for a table.

        Parameters
        ----------
        table_name : str
            Name of the table to inspect (in the public schema).

        Returns
        -------
        dict[str, str]
            Dictionary mapping column names to database-level data type names.

        Raises
        ------
        Exception
            If the metadata query fails.

        """
        _validate_identifier(table_name)
        cursor = self._conn.cursor()
        try:
            cursor.execute(
                """
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = %s;
                """,
                (table_name,),
            )
            return {row[0]: row[1] for row in cursor.fetchall()}
        finally:
            cursor.close()

    def _adapt_jsonb(self, data, table_name):
        """Adapt Python dict/list values to JSONB for the target table's columns.

        Any key in *data* whose corresponding column type is JSONB will be wrapped
        using :class:`psycopg2.extras.Json`. Lists are adapted only when every element
        is a dict (typical for arrays of objects).

        Parameters
        ----------
        data : dict
            Key-value pairs to be inserted/updated.
        table_name : str
            Table name used to introspect column types.

        Returns
        -------
        dict
            The same dict instance with JSONB-compatible values wrapped in a driver
            adapter where appropriate (mutation in place).

        """
        column_types = self._get_column_types(table_name)
        # Handle json
        for key, value in data.items():
            if (
                value
                and column_types.get(key, "").lower() == "jsonb"
                and (isinstance(value, dict) or (isinstance(value, list) and all(isinstance(item, dict) for item in value)))
            ):
                data[key] = psycopg2_json(value)
        return data

    def list_databases(self):
        """Print the non-template databases available on the server.

        Returns
        -------
        None

        Raises
        ------
        Exception
            If listing databases fails.

        """
        try:
            # Connect to the default 'postgres' database to list databases
            conn = psycopg2.connect(dbname="postgres", user=self.user, password=self.password, host=self.host, port=self.port)
            cursor = conn.cursor()

            cursor.execute("SELECT datname FROM pg_database WHERE datistemplate = false;")
            databases = cursor.fetchall()
            if databases:
                print("Databases available:")
                for db in databases:
                    print(db[0])  # Each db is a tuple with the database name in the first position
            else:
                print("No databases found.")

            cursor.close()
            conn.close()
        except Exception as e:
            raise ConnectionError(f"Error: Failed to list databases: {e}") from e

    def create_database(self, new_db_name):
        """Create a new database using the configured template (if it doesn't exist).

        Parameters
        ----------
        new_db_name : str
            Name for the database to create.

        Returns
        -------
        None

        Raises
        ------
        Exception
            If creation fails or privileges are insufficient.

        """
        try:
            # Connect to the default 'postgres' database
            conn = psycopg2.connect(dbname="postgres", user=self.user, password=self.password, host=self.host, port=self.port)
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()

            # Check if the database already exists
            if _database_exists(cursor, new_db_name):
                print(f"Database '{new_db_name}' already exists.")
            else:
                # Create the new database with specified options
                create_db_query = sql.SQL(
                    """
                    CREATE DATABASE {db}
                    WITH
                    OWNER = {owner}
                    TEMPLATE = {tmpl}
                    ENCODING = 'UTF8'
                    TABLESPACE = pg_default
                    CONNECTION LIMIT = -1;
                    """
                ).format(
                    db=sql.Identifier(new_db_name),
                    owner=sql.Identifier(self.user) if self.user else sql.SQL("CURRENT_USER"),
                    tmpl=sql.Identifier(self.template),
                )
                cursor.execute(create_db_query)
                print(f"Database '{new_db_name}' created successfully with the specified options.")

            cursor.close()
            conn.close()

            # Update the object's database name
            self.db_name = new_db_name

        except Exception as e:
            raise RuntimeError(f"Error: Failed to create database: {e}") from e

    def delete_database(self, db_name):
        """Drop a database if it exists.

        Parameters
        ----------
        db_name : str
            Name of the database to drop.

        Returns
        -------
        None

        Raises
        ------
        Exception
            If the drop operation fails.

        """
        try:
            conn = psycopg2.connect(dbname="postgres", user=self.user, password=self.password, host=self.host, port=self.port)
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()
            if _database_exists(cursor, db_name):
                cursor.execute(sql.SQL("DROP DATABASE IF EXISTS {db};").format(db=sql.Identifier(db_name)))
                print(f"Database '{db_name}' deleted successfully.")
            else:
                print(f"Database '{db_name}' does not exist.")
            cursor.close()
            conn.close()

        except Exception as e:
            raise RuntimeError(f"Error: Failed to delete database: {e}") from e

    def list_tables(self):
        """Print all user tables in the current database (public schema).

        Returns
        -------
        None

        Raises
        ------
        Exception
            If listing tables fails.

        """
        self._ensure_connection()

        try:
            cursor = self._conn.cursor()

            cursor.execute(
                """
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public' AND table_type = 'BASE TABLE';
                """
            )
            table_names = []
            tables = cursor.fetchall()
            if tables:
                print("Tables in the current database:")
                for table in tables:
                    print(table[0])  # Each table is a tuple with the table name in the first position
                    table_names.append(table[0])
            else:
                print("No tables found in the current database.")

            cursor.close()
            return table_names
        except Exception as e:
            raise RuntimeError(f"Error: Failed to list tables: {e}") from e

    def create_table(self, table_name, columns, with_geometry=False, srid=4326):
        """Create a table with optional PostGIS geometry column.

        Parameters
        ----------
        table_name : str
            Name of the table to create (in the public schema).
        columns : dict[str, str]
            Mapping of column_name → SQL type (e.g., {"id": "SERIAL PRIMARY KEY"}).
            Do **not** include the geometry column here.
        with_geometry : bool, optional
            If True, a column named geometry with type geometry(GEOMETRY, SRID)
            will be added. Defaults to False.
        srid : int, optional
            Spatial reference identifier used when with_geometry is True. Defaults to 4326.

        Returns
        -------
        None

        Raises
        ------
        Exception
            If the CREATE TABLE statement fails.

        """
        self._ensure_connection()
        _validate_identifier(table_name)
        for c in columns:
            _validate_identifier(c)
        cursor = self._conn.cursor()
        try:
            # Build column fragments: "col_name col_type"
            col_frags = [
                sql.SQL("{} {}").format(sql.Identifier(col_name), sql.SQL(col_type)) for col_name, col_type in columns.items()
            ]

            if with_geometry:
                # Example: generic geometry; ex. geometry(Point, 4326)
                col_frags.append(sql.SQL("geom geometry(Geometry, {})").format(sql.Literal(srid)))

            # CREATE TABLE IF NOT EXISTS table_name (col1 type1, col2 type2, ...)
            stmt = (
                sql.SQL("CREATE TABLE IF NOT EXISTS {} (").format(sql.Identifier(table_name))
                + sql.SQL(", ").join(col_frags)
                + sql.SQL(")")
            )

            cursor.execute(stmt)
            self._conn.commit()
            print(f"Table '{table_name}' created successfully.")
        except Exception as e:
            self._conn.rollback()
            raise RuntimeError(f"Error: Failed to create table: {e}") from e
        finally:
            cursor.close()

    def delete_table(self, table_name):
        """Drop a table (and dependent objects) if it exists.

        Parameters
        ----------
        table_name : str
            Name of the table to drop.

        Returns
        -------
        None

        Raises
        ------
        Exception
            If the operation fails.

        """
        self._ensure_connection()
        cursor = self._conn.cursor()
        try:
            cursor.execute(
                "SELECT 1 FROM information_schema.tables WHERE table_name = %s;",
                (table_name,),
            )
            if cursor.fetchone():
                cursor.execute(sql.SQL("DROP TABLE IF EXISTS {t} CASCADE;").format(t=sql.Identifier(table_name)))
                self._conn.commit()
                print(f"Table '{table_name}' deleted successfully.")
            else:
                print(f"Table '{table_name}' does not exist.")
        except Exception as e:
            self._conn.rollback()
            raise RuntimeError(f"Error: Failed to delete table: {e}") from e
        finally:
            cursor.close()

    def insert_record(self, table_name, data, col_id):
        """Insert a record and return the server-generated identifier.

        JSON/JSONB values are adapted automatically for columns typed as JSONB.

        Parameters
        ----------
        table_name : str
            Target table name.
        data : dict
            Column values to insert ({column: value}).
        col_id : str
            Name of the column to return (e.g., the primary key).

        Returns
        -------
        Any
            The value returned by RETURNING {col_id} for the inserted row.

        Raises
        ------
        Exception
            If the INSERT fails.

        """
        self._ensure_connection()
        _validate_identifier(table_name)
        _validate_identifier(col_id)
        for k in data:
            _validate_identifier(k)

        def _op():
            cursor = self._conn.cursor()
            try:
                adapted_data = self._adapt_jsonb(data.copy(), table_name)
                columns = list(adapted_data.keys())
                placeholders = sql.SQL(", ").join(sql.Placeholder() for _ in columns)
                stmt = sql.SQL("INSERT INTO {t} ({cols}) VALUES ({vals}) RETURNING {id}").format(
                    t=sql.Identifier(table_name),
                    cols=sql.SQL(", ").join(sql.Identifier(c) for c in columns),
                    vals=placeholders,
                    id=sql.Identifier(col_id),
                )
                values = [adapted_data[c] for c in columns]
                cursor.execute(stmt, values)
                new_id = cursor.fetchone()[0]
                self._conn.commit()
                return new_id
            except (psycopg2.OperationalError, psycopg2.InterfaceError):
                self._conn.rollback()
                raise
            except Exception as e:
                print("Insert Error:", str(e))
                self._conn.rollback()
                raise RuntimeError(f"Error: Failed to insert record: {e}") from e
            finally:
                cursor.close()

        return self._run_with_reconnect_retry(_op, operation_name="insert record")

    def update_record(self, table_name, data, condition):
        """Update columns for rows matching a SQL condition.

        For a geometry key with value starting with "ST_" (e.g., "ST_GeomFromText(...)"),
        the expression is injected verbatim into the SQL (not parameterized) to allow PostGIS
        functions; all other values are safely parameterized.

        Parameters
        ----------
        table_name : str
            Target table name.
        data : dict
            Columns and values to update.
        condition : str
            SQL predicate used in the WHERE clause (e.g., "id = 10").

        Returns
        -------
        None

        Raises
        ------
        Exception
            If the UPDATE fails.

        """
        self._ensure_connection()

        def _op():
            cursor = self._conn.cursor()
            try:
                adapted_data = self._adapt_jsonb(data.copy(), table_name)

                # Resolve the actual geometry column name (fallback to configured default)
                actual_geom_col = self.get_geometry_column(table_name) or self.db_col_geom
                geom_aliases = {"geometry", self.db_col_geom}

                updates = []
                values = []

                for key, value in adapted_data.items():
                    # Map alias "geometry" -> actual geometry column
                    col_name = actual_geom_col if key in geom_aliases else key

                    # If caller supplied a raw PostGIS expression (starts with ST_), inject verbatim
                    if col_name == actual_geom_col and isinstance(value, str) and value.lstrip().startswith("ST_"):
                        updates.append(sql.SQL("{} = ").format(sql.Identifier(col_name)).as_string(cursor) + value)
                    else:
                        updates.append(sql.SQL("{} = %s").format(sql.Identifier(col_name)).as_string(cursor))
                        values.append(value)

                stmt = sql.SQL("UPDATE {t} SET {set_clause} WHERE {cond};").format(
                    t=sql.Identifier(table_name),
                    set_clause=sql.SQL(", ").join(sql.SQL(u) for u in updates),
                    cond=sql.SQL(condition),
                )

                cursor.execute(stmt, tuple(values))
                self._conn.commit()
            except (psycopg2.OperationalError, psycopg2.InterfaceError):
                self._conn.rollback()
                raise
            except Exception as e:
                self._conn.rollback()
                raise RuntimeError(f"Error: Failed to update record: {e}") from e
            finally:
                cursor.close()

        self._run_with_reconnect_retry(_op, operation_name="update record")

    def delete_record(self, table_name, condition):
        """Delete rows matching a SQL condition.

        Parameters
        ----------
        table_name : str
            Target table name.
        condition : str
            SQL predicate for the WHERE clause (e.g., "id = 10").

        Returns
        -------
        None

        Raises
        ------
        Exception
            If the DELETE fails.

        """
        self._ensure_connection()
        _validate_identifier(table_name)
        cursor = self._conn.cursor()
        try:
            stmt = sql.SQL("DELETE FROM {t} WHERE {cond};").format(
                t=sql.Identifier(table_name),
                cond=sql.SQL(condition),  # kept as-is for compatibility
            )
            cursor.execute(stmt)
            self._conn.commit()
        except Exception as e:
            self._conn.rollback()
            raise RuntimeError(f"Error: Failed to delete record: {e}") from e
        finally:
            cursor.close()

    def get_geometry_column(self, table_name: str, schema: str | None = None):
        """Delete rows matching a SQL condition.

        Parameters
        ----------
        table_name : str
            Target table name.
        condition : str
            SQL predicate for the WHERE clause (e.g., "id = 10").

        Returns
        -------
        None

        Raises
        ------
        Exception
            If the DELETE fails.

        """
        self._ensure_connection()
        _validate_identifier(table_name)
        cursor = self._conn.cursor(cursor_factory=RealDictCursor)
        try:
            if schema:
                _validate_identifier(schema)
                cursor.execute(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema = %s AND table_name = %s AND udt_name = 'geometry';
                    """,
                    (schema, table_name),
                )
            else:
                cursor.execute(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema = 'public' AND table_name = %s AND udt_name = 'geometry';
                    """,
                    (table_name,),
                )
            rows = cursor.fetchall()
            return rows[0]["column_name"] if rows else None
        finally:
            cursor.close()

    def write_df_to_db(self, df, table_name, if_exists="replace"):
        """Write a (Geo)DataFrame into PostgreSQL (uses SQLAlchemy).

        If *df* is a :class:`geopandas.GeoDataFrame`, it will be written with
        :meth:`GeoDataFrame.to_postgis`; otherwise :meth:`pandas.DataFrame.to_sql`
        is used.

        Parameters
        ----------
        df : pandas.DataFrame | geopandas.GeoDataFrame
            The frame to write.
        table_name : str
            Destination table name.
        if_exists : {"fail", "replace", "append"}, optional
            Behavior when the table already exists. Defaults to "replace".

        Returns
        -------
        None

        Raises
        ------
        Exception
            If writing to the database fails.

        """
        try:
            # Create a SQLAlchemy engine
            connection_string = f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.db_name}"
            engine = create_engine(connection_string)

            # Check if the input is a GeoDataFrame
            if isinstance(df, gpd.GeoDataFrame):
                # Write the GeoDataFrame to PostgreSQL
                df.to_postgis(table_name, engine, if_exists=if_exists)
                print(f"GeoDataFrame written to table '{table_name}' in PostgreSQL database.")
            else:
                # Write the DataFrame to PostgreSQL
                df.to_sql(table_name, engine, if_exists=if_exists, index=False)
                print(f"DataFrame written to table '{table_name}' in PostgreSQL database.")
        except Exception as e:
            raise RuntimeError(f"Error: Failed to write DataFrame to database: {e}") from e

    def _split_schema_name(self, table_name: str) -> tuple[str | None, str, str]:
        """Split 'schema.table' or 'table' into (schema, table) and built sql identifier."""
        if "." in table_name:
            parts = table_name.split(".")
            if len(parts) == 2:
                tbl_sql = sql.SQL("{}.{}").format(sql.Identifier(parts[0]), sql.Identifier(parts[1]))
                return parts[0], parts[1], tbl_sql

        tbl_sql = sql.Identifier(table_name)
        return None, table_name, tbl_sql

    def get_df_from_db(self, table_name: str, condition: str | None = None):
        """Fetch a table as a DataFrame or GeoDataFrame (when geometry is present).

        Detects a geometry-typed column via `get_geometry_column` and uses
        `geopandas.read_postgis` when available; otherwise falls back to `pandas.read_sql_query`.

        Parameters
        ----------
        table_name : str
            Table name to read (optionally schema-qualified, e.g. 'public.tbl_users').
        condition : str, optional
            SQL predicate appended after WHERE (no bind params handled here).

        Returns
        -------
        pandas.DataFrame | geopandas.GeoDataFrame
            A pandas DataFrame if no geometry column is present; otherwise a GeoDataFrame.

        Raises
        ------
        Exception
            If reading from the database fails.

        """
        engine = None
        try:
            self._ensure_connection()

            # Accept schema-qualified identifiers; keep consistent with module style
            _validate_identifier(table_name, allow_dot=True)

            # Preserve special characters in database credentials without URL parsing.
            connection_url = URL.create(
                "postgresql+psycopg2",
                username=self.user,
                password=self.password,
                host=self.host,
                port=int(self.port),
                database=self.db_name,
            )
            engine = create_engine(connection_url)

            # Quote identifier safely (schema-qualified supported)
            schema, name, tbl_sql = self._split_schema_name(table_name)

            # Use the psycopg2 connection for identifier quoting
            base_stmt = sql.SQL("SELECT * FROM {}").format(tbl_sql).as_string(self._conn)

            # Build the final query. If you expect binds, pass them via pandas/gpd params.
            query = text(base_stmt + (f" WHERE {condition}" if condition else ""))

            # Detect geometry correctly (schema-aware)
            geom_col = self.get_geometry_column(name, schema=schema)

            with engine.connect() as conn:
                if geom_col:
                    return gpd.read_postgis(query, conn, geom_col=geom_col)
                else:
                    return pd.read_sql_query(query, conn)

        except Exception as e:
            raise RuntimeError(f"Error: Failed to retrieve table from database: {e}") from e
        finally:
            if engine is not None:
                engine.dispose()

    def record_exists(self, table_name, condition, condition_values=None):
        """Check for the existence of at least one row matching a condition.

        Parameters
        ----------
        table_name : str
            Target table name.
        condition : str
            SQL predicate with optional placeholders (e.g., "id = %s").
        condition_values : tuple | list | None, optional
            Bind values corresponding to placeholders in *condition*.

        Returns
        -------
        bool
            True if at least one row matches; otherwise False.

        Raises
        ------
        Exception
            If the query fails.

        """
        self._ensure_connection()
        _validate_identifier(table_name)
        cursor = self._conn.cursor()
        try:
            stmt = sql.SQL("SELECT EXISTS(SELECT 1 FROM {t} WHERE {cond});").format(
                t=sql.Identifier(table_name),
                cond=sql.SQL(condition),  # keep API; encourage callers to pass placeholders
            )
            cursor.execute(stmt, condition_values if condition_values else ())
            exists = cursor.fetchone()[0]
            return exists
        except Exception as e:
            raise RuntimeError(f"Error: Failed to check record existence: {e}") from e
        finally:
            cursor.close()

    def create_delete_trigger(self, parent_table, child_table, parent_key, child_key):
        """Check for the existence of at least one row matching a condition.

        Parameters
        ----------
        table_name : str
            Target table name.
        condition : str
            SQL predicate with optional placeholders (e.g., "id = %s").
        condition_values : tuple | list | None, optional
            Bind values corresponding to placeholders in *condition*.

        Returns
        -------
        bool
            True if at least one row matches; otherwise False.

        Raises
        ------
        Exception
            If the query fails.

        """
        self._ensure_connection()

        for n in (parent_table, child_table, parent_key, child_key):
            _validate_identifier(n)

        cursor = self._conn.cursor()
        try:
            trigger_function_name = f"delete_{child_table}_on_{parent_table}_delete"
            trigger_name = f"trg_delete_{child_table}_on_{parent_table}"

            # Check if the trigger exists
            cursor.execute(
                """
                SELECT trigger_name FROM information_schema.triggers
                WHERE event_object_table = %s AND trigger_name = %s;
                """,
                (parent_table, trigger_name),
            )
            trigger_exists = cursor.fetchone()

            if trigger_exists:
                print(f"Trigger '{trigger_name}' already exists. Skipping creation.")
                return

            # Create the trigger function and trigger
            func_stmt = sql.SQL(
                """
                CREATE OR REPLACE FUNCTION {fn}()
                RETURNS TRIGGER AS $$
                BEGIN
                DELETE FROM {child} WHERE {ck} = OLD.{pk};
                RETURN OLD;
                END;
                $$ LANGUAGE plpgsql;
                """
            ).format(
                fn=sql.Identifier(trigger_function_name),
                child=sql.Identifier(child_table),
                ck=sql.Identifier(child_key),
                pk=sql.Identifier(parent_key),
            )
            trg_stmt = sql.SQL(
                """
                CREATE TRIGGER {tn}
                BEFORE DELETE ON {pt}
                FOR EACH ROW
                EXECUTE FUNCTION {fn}();
                """
            ).format(
                tn=sql.Identifier(trigger_name),
                pt=sql.Identifier(parent_table),
                fn=sql.Identifier(trigger_function_name),
            )
            cursor.execute(func_stmt)
            cursor.execute(trg_stmt)
            self._conn.commit()
            print(f"Trigger '{trigger_name}' created successfully.")
        except Exception as e:
            raise RuntimeError(f"Error: Failed to create trigger: {e}") from e
        finally:
            cursor.close()

    def clone_database(self, source_db, target_db):
        """Clone a database using PostgreSQL's TEMPLATE mechanism.

        All existing connections to *source_db* (other than the current session) are
        terminated to allow cloning.

        Parameters
        ----------
        source_db : str
            Name of the source database to clone.
        target_db : str
            Name for the new database to be created.

        Returns
        -------
        None

        Raises
        ------
        Exception
            If terminating connections or cloning fails.

        """
        try:
            _validate_identifier(source_db)
            _validate_identifier(target_db)
            self.close_connection()

            # Connect to a neutral DB
            conn = psycopg2.connect(dbname="postgres", user=self.user, password=self.password, host=self.host, port=self.port)
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()

            # Terminate all other connections to the source DB
            cursor.execute(
                """
                SELECT pg_terminate_backend(pid)
                FROM pg_stat_activity
                WHERE datname = %s AND pid <> pg_backend_pid();
                """,
                (source_db,),
            )
            print(f"Terminated active connections to '{source_db}'.")

            # Check if target exists
            if _database_exists(cursor, target_db):
                print(f"Target database '{target_db}' already exists.")
            else:
                # Clone
                cursor.execute(
                    sql.SQL("CREATE DATABASE {tgt} TEMPLATE {src};").format(
                        tgt=sql.Identifier(target_db), src=sql.Identifier(source_db)
                    )
                )
                print(f"Database '{target_db}' successfully cloned from '{source_db}'.")

            cursor.close()
            conn.close()

        except Exception as e:
            raise RuntimeError(f"Error: Failed to clone database '{source_db}' to '{target_db}': {e}") from e

    def convert_column_to_jsonb(self, table_name, column_name, auto_fix=False):
        """Convert a TEXT column to JSONB, optionally fixing common Pythonicity issues.

        If *auto_fix* is True, the function attempts a simple textual cleanup to
        convert Python-style representations (single quotes, None) into valid JSON
        before casting.

        Parameters
        ----------
        table_name : str
            Name of the table containing the column.
        column_name : str
            Name of the column to convert.
        auto_fix : bool, optional
            If True, apply a quick-and-dirty text replacement to improve parse
            success prior to ::jsonb cast. Defaults to False.

        Returns
        -------
        None

        Raises
        ------
        Exception
            If the conversion fails.

        """
        self._ensure_connection()
        _validate_identifier(table_name)
        _validate_identifier(column_name)
        cursor = self._conn.cursor()
        try:
            if auto_fix:
                print(f"Attempting to fix invalid JSON entries in {table_name}.{column_name}...")
                fix_stmt = sql.SQL(
                    """
                    UPDATE {t}
                    SET {c} = REPLACE(
                    REPLACE(
                    REPLACE({c}, '''', '"'),
                    ' None', ' null'
                    ),
                    'None', 'null'
                    )
                    WHERE {c} IS NOT NULL;
                """
                ).format(t=sql.Identifier(table_name), c=sql.Identifier(column_name))
                cursor.execute(fix_stmt)
                self._conn.commit()

            alter_stmt = sql.SQL(
                """
                ALTER TABLE {t}
                ALTER COLUMN {c}
                SET DATA TYPE jsonb
                USING CASE
                WHEN {c} IS NULL THEN NULL
                ELSE {c}::jsonb
                END;
            """
            ).format(t=sql.Identifier(table_name), c=sql.Identifier(column_name))
            cursor.execute(alter_stmt)
            self._conn.commit()
            print(f"Column '{column_name}' in table '{table_name}' converted to JSONB.")
        except Exception as e:
            raise RuntimeError(f"Error: Failed to convert column to JSONB: {e}") from e
        finally:
            cursor.close()

    def create_update_column_on_source_change_trigger(self, tbl_source, tbl_update, col_source, col_update):
        """Create a trigger to update values in one table when another column changes.

        On updates to tbl_source.col_source, any rows in *tbl_update* whose
        *col_update* begins with the old value will have that prefix replaced with
        the new value (simple string replacement).

        Parameters
        ----------
        tbl_source : str
            Table where the source column is updated.
        tbl_update : str
            Table to be updated when a change occurs in *tbl_source*.
        col_source : str
            Column name monitored for changes in *tbl_source*.
        col_update : str
            Column name to update in *tbl_update* by replacing the prefix.

        Returns
        -------
        None

        Raises
        ------
        Exception
            If trigger creation fails.

        """
        self._ensure_connection()
        for n in (tbl_source, tbl_update, col_source, col_update):
            _validate_identifier(n)
        cursor = self._conn.cursor()
        try:
            trigger_function_name = f"update_{tbl_update}_{col_update}_on_{tbl_source}_{col_source}_change"
            trigger_name = f"trg_update_{tbl_update}_{col_update}_on_{tbl_source}_{col_source}"

            # Check if the trigger already exists
            cursor.execute(
                """
                SELECT trigger_name FROM information_schema.triggers
                WHERE event_object_table = %s
                AND trigger_name = %s;
                """,
                (tbl_source, trigger_name),
            )
            if cursor.fetchone():
                print(f"Trigger '{trigger_name}' already exists. Skipping creation.")
                return

            # Construct SQL
            func_stmt = sql.SQL(
                """
                CREATE OR REPLACE FUNCTION {fn}()
                RETURNS TRIGGER AS $$
                BEGIN
                IF NEW.{cs} <> OLD.{cs} THEN
                UPDATE {tu}
                SET {cu} = REPLACE({cu}, OLD.{cs}, NEW.{cs})
                WHERE {cu} LIKE OLD.{cs} || '%';
                END IF;
                RETURN NEW;
                END;
                $$ LANGUAGE plpgsql;
            """
            ).format(
                fn=sql.Identifier(trigger_function_name),
                cs=sql.Identifier(col_source),
                tu=sql.Identifier(tbl_update),
                cu=sql.Identifier(col_update),
            )
            trg_stmt = sql.SQL(
                """
                CREATE TRIGGER {tn}
                AFTER UPDATE OF {cs} ON {ts}
                FOR EACH ROW
                EXECUTE FUNCTION {fn}();
                """
            ).format(
                tn=sql.Identifier(trigger_name),
                cs=sql.Identifier(col_source),
                ts=sql.Identifier(tbl_source),
                fn=sql.Identifier(trigger_function_name),
            )
            cursor.execute(func_stmt)
            cursor.execute(trg_stmt)
            self._conn.commit()
            print(f"Trigger '{trigger_name}' created successfully.")
        except Exception as e:
            raise RuntimeError(f"Error: Failed to create update trigger: {e}") from e
        finally:
            cursor.close()


def create_db_handler(db_name: str) -> PostgresPostGISHandler:
    """Connect to a database using environment variables.

    Environment variables
    ---------------------
    POSTGRES_USER : str, optional
        Defaults to "postgres".
    POSTGRES_PASSWORD : str, optional
        Defaults to an empty string.
    POSTGRES_HOST : str, optional
        Defaults to "localhost".
    POSTGRES_PORT : int, optional
        Defaults to 5432.

    Parameters
    ----------
    db_name : str
        Name of the database to connect to.

    Returns
    -------
    PostgresPostGISHandler
        Connected database handler instance.

    """
    port_env = os.getenv("POSTGRES_PORT")
    port: int | str = port_env if port_env is not None else 5432

    return PostgresPostGISHandler(
        db_name=db_name,
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", ""),
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=port,
    )


def create_tbl_ingestion(db_handler, tbl_ingestion: str):
    """Create the ingestion tracking table if it does not exist.

    The table stores per-product ingestion results, including download and
    processing status, keyed by a unique product title. A geometry column
    is also created via the `with_geometry=True` flag, depending on the
    implementation of `db_handler.create_table`.

    Parameters
    ----------
    db_handler
        Database handler instance exposing a `create_table` method with
        signature compatible to `create_table(table_name, columns, with_geometry)`.
    tbl_ingestion : str
        Name of the ingestion table to create.

    Returns
    -------
    str
        Informational message describing the creation status and table schema.

    Raises
    ------
    Exception
        If the table cannot be created for any reason.

    """
    try:
        msg = f"Table '{tbl_ingestion}' does not exist in the database. Creating..."
        columns = {
            "ingestion_id": _SERIAL_PRIMARY_KEY,
            "date": "TIMESTAMP",
            "collection_name": _VARCHAR,
            "product_type": _VARCHAR,
            "product_title": _VARCHAR,
            "download_result": _VARCHAR,
            "process_result": _VARCHAR,
            "orch_process_id": _VARCHAR,
        }
        db_handler.create_table(table_name=tbl_ingestion, columns=columns, with_geometry=True)
        msg += f"\nSuccess: table '{tbl_ingestion}' created with schema: \n {columns}"
        return msg

    except Exception as e:
        raise RuntimeError(f"Error: Could not create the table '{tbl_ingestion}': {e}") from e


def create_tbl_users(db_handler, tbl_users: str):
    """Create the users/credentials table and insert default product rows.

    The table stores credentials (and optional token) per `product_type`.
    The primary key column name is read from the `COL_USER_ID` environment
    variable to allow project-specific naming conventions.

    Initial placeholder rows are inserted for all supported product types
    (e.g., GRD, SLC, Sentinel-2, AOD, SLSTR, HLSL30, AXIS11, AXIS12) with
    empty credentials that must be updated before use.

    Parameters
    ----------
    db_handler
        Database handler instance exposing `create_table` and `insert_record`
        methods compatible with the calls in this function.
    tbl_users : str
        Name of the users table to create.

    Returns
    -------
    str
        Informational message describing the creation status, table schema,
        and a warning that usernames/passwords must be populated.

    Raises
    ------
    Exception
        If the table cannot be created or the initial records cannot be inserted.

    """
    try:
        col_password = "password"  # nosec # NOSONAR - schema field name, not a credential value

        msg = f"Table '{tbl_users}' does not exist in the database. Creating..."
        col_user_id = os.getenv("COL_USER_ID")
        columns = {
            col_user_id: _SERIAL_PRIMARY_KEY,
            "product_type": _UNIQUE_CHAR,
            "username": _VARCHAR,
            col_password: _VARCHAR,
            "token": _VARCHAR,
        }
        db_handler.create_table(table_name=tbl_users, columns=columns, with_geometry=False)
        msg += f"\nSuccess: table '{tbl_users}' created with schema: \n {columns}"

        for pr in ["GRD", "SLC", "S2MSI2A", "S2MSI1C", "AOD", "SLSTR", "HLSL30", "HOURLY_ERA5", "AXIS11", "AXIS12"]:
            data = {"product_type": pr, "username": "", "password": ""}
            _ = db_handler.insert_record(table_name=tbl_users, data=data, col_id=col_user_id)

        msg += f"Warning: new table: {tbl_users} is created; usernames and passwords must be inserted."
        return msg
    except Exception as e:
        raise RuntimeError(f"Error: Could not create the table '{tbl_users}': {e}") from e
