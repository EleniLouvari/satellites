"""Regression checks for database credentials and read connection cleanup."""

from unittest.mock import MagicMock

import pandas as pd
import pytest
from tests.utils import expect_equal, expect_true

pytest.importorskip("psycopg2")
pytest.importorskip("sqlalchemy")

from shared import postgress


@pytest.mark.parametrize("failure", [None, "connect", "query", "geometry"])
def test_table_read_preserves_url_credentials_and_disposes_engine(monkeypatch, failure):
    """Special characters survive URL construction and every read releases its pool."""
    handler = object.__new__(postgress.PostgresPostGISHandler)
    handler.user = "reader@example"
    handler.password = "p@ss:/?#%word"
    handler.host = "localhost"
    handler.port = "5432"
    handler.db_name = "accounts"
    handler.connection = MagicMock(closed=0)
    monkeypatch.setattr(handler, "get_geometry_column", MagicMock(return_value=None))
    statement = MagicMock()
    statement.format.return_value.as_string.return_value = 'SELECT * FROM "public"."accounts"'
    monkeypatch.setattr(postgress.sql, "SQL", MagicMock(return_value=statement))
    engine = MagicMock()
    engine_factory = MagicMock(return_value=engine)
    monkeypatch.setattr(postgress, "create_engine", engine_factory)
    expected = pd.DataFrame({"username": ["alice"], "password": ["secret"]})
    query = MagicMock(return_value=expected)
    monkeypatch.setattr(postgress.pd, "read_sql_query", query)
    if failure == "connect":
        engine.connect.side_effect = RuntimeError("connection failed")
    elif failure == "query":
        query.side_effect = RuntimeError("read failed")
    elif failure == "geometry":
        handler.get_geometry_column.side_effect = RuntimeError("metadata failed")

    if failure:
        with pytest.raises(RuntimeError, match="Error: Failed to retrieve table"):
            handler.get_df_from_db("public.accounts")
    else:
        expect_true(handler.get_df_from_db("public.accounts").equals(expected))

    url = engine_factory.call_args.args[0]
    expect_equal(url.username, handler.user)
    expect_equal(url.password, handler.password)
    expect_equal(url.port, 5432)
    expect_equal(url.database, "accounts")
    engine.dispose.assert_called_once_with()
