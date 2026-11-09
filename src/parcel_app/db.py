"""Small transactional PostgreSQL helpers with dictionary rows."""

import os
from contextlib import contextmanager

import psycopg2
from psycopg2.extras import RealDictCursor


@contextmanager
def connection():
    """Commit on success, rollback on failure and always close a connection."""
    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", ""),
        dbname=os.getenv("DB_NAME", os.getenv("POSTGRES_DB", "postgres")),
        cursor_factory=RealDictCursor,
        connect_timeout=8,
    )
    try:
        yield conn
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def _query(query, params, conn, mode):
    """Execute a bound query using either an owned or caller transaction."""
    if conn is None:
        with connection() as owned:
            return _query(query, params, owned, mode)
    with conn.cursor() as cursor:
        cursor.execute(query, params)
        if mode == "one":
            row = cursor.fetchone()
            return dict(row) if row is not None else None
        if mode == "all":
            return [dict(row) for row in cursor.fetchall()]
        return cursor.rowcount


def one(query, params=(), *, conn=None):
    """Return a dictionary row or None."""
    return _query(query, params, conn, "one")


def all_rows(query, params=(), *, conn=None):
    """Return all rows as dictionaries."""
    return _query(query, params, conn, "all")


def execute(query, params=(), *, conn=None):
    """Execute a statement and return its affected row count."""
    return _query(query, params, conn, "execute")
