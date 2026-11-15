"""Unit tests for database transactions and reconnects using isolated driver mocks."""

from unittest.mock import MagicMock, Mock

import pytest

pytest.importorskip("psycopg2")
pytest.importorskip("sqlalchemy")

from shared import postgress as pg


@pytest.fixture
def database(monkeypatch):
    """Construct the real handler against a fake connection, never a live database."""
    connection = MagicMock(closed=0)
    cursor = connection.cursor.return_value
    cursor.fetchall.return_value = []
    connect = Mock(return_value=connection)
    monkeypatch.setattr(pg.psycopg2, "connect", connect)
    handler = pg.PostgresPostGISHandler("test_db", user="tester", password="test-only", host="db.invalid", port=5439)
    return handler, connection, cursor, connect


def test_constructor_connects_and_close_is_idempotent(database):
    """Connection options are forwarded and repeated closes release the connection only once."""
    handler, connection, _, connect = database
    connect.assert_called_once_with(dbname="test_db", user="tester", password="test-only", host="db.invalid", port=5439)
    handler.close_connection()
    handler.close_connection()
    connection.close.assert_called_once_with()
    assert handler.connection is None


def test_constructor_wraps_connection_failure(database):
    """Connection errors retain their underlying driver cause."""
    _, _, _, connect = database
    failure = pg.psycopg2.OperationalError("offline")
    connect.side_effect = failure
    with pytest.raises(ConnectionError, match="Error: Connection failed") as error:
        pg.PostgresPostGISHandler()
    assert error.value.__cause__ is failure


@pytest.mark.parametrize("state", [None, 1, 0])
def test_connection_reopens_only_when_missing_or_closed(database, state):
    """Healthy connections are reused while absent and stale connections are replaced."""
    handler, connection, _, connect = database
    if state is None:
        handler.connection = None
    else:
        connection.closed = state
    handler._ensure_connection()
    assert connect.call_count == (1 if state == 0 else 2)
    assert handler.connection is connection


@pytest.mark.parametrize("failure_type", [pg.psycopg2.OperationalError, pg.psycopg2.InterfaceError])
@pytest.mark.parametrize("succeeds", [True, False])
def test_insert_retries_connection_failures_and_releases_cursors(database, failure_type, succeeds):
    """Transient inserts roll back, reconnect once, and either commit or report exhaustion."""
    handler, connection, cursor, connect = database
    failure = failure_type("lost connection")
    # Each insert performs a metadata read before executing its INSERT.
    cursor.execute.side_effect = [None, failure, None, None if succeeds else failure]
    cursor.fetchone.return_value = (42,)
    if succeeds:
        assert handler.insert_record("items", {"title": "sample"}, "id") == 42
        connection.commit.assert_called_once_with()
    else:
        with pytest.raises(RuntimeError, match="Failed to insert record") as error:
            handler.insert_record("items", {"title": "sample"}, "id")
        assert error.value.__cause__ is failure
        connection.commit.assert_not_called()
    assert connection.rollback.call_count == (1 if succeeds else 2)
    assert cursor.close.call_count == 4
    connection.close.assert_called_once_with()
    assert connect.call_count == 2


def test_insert_adapts_jsonb_and_binds_values_without_mutating_input(database):
    """JSON payloads use driver adapters while text remains a bound SQL parameter."""
    handler, connection, cursor, _ = database
    cursor.fetchall.return_value = [("payload", "jsonb"), ("title", "text")]
    cursor.fetchone.return_value = (7,)
    data = {"payload": {"count": 3}, "title": "O'Brien"}
    assert handler.insert_record("items", data, "id") == 7
    statement, values = cursor.execute.call_args.args
    assert isinstance(statement, pg.sql.Composed)
    assert isinstance(values[0], pg.psycopg2_json)
    assert values[0].adapted == {"count": 3}
    assert values[1] == "O'Brien"
    assert data == {"payload": {"count": 3}, "title": "O'Brien"}
    connection.commit.assert_called_once_with()


@pytest.mark.parametrize("value,adapted", [([{"a": 1}], True), ([1, 2], False), ({}, False), (None, False)])
def test_jsonb_adaptation_handles_lists_empty_and_missing_values(database, value, adapted):
    """Only nonempty object payloads are wrapped as JSONB driver values."""
    handler, _, cursor, _ = database
    cursor.fetchall.return_value = [("payload", "JSONB")]
    result = handler._adapt_jsonb({"payload": value}, "items")["payload"]
    assert isinstance(result, pg.psycopg2_json) is adapted
    assert (result.adapted if adapted else result) == value


@pytest.mark.parametrize(
    "method,args",
    [
        ("create_table", ("items", {"id": "INTEGER"}, True, 3857)),
        ("delete_table", ("items",)),
        ("delete_record", ("items", "id = 1")),
        ("insert_record", ("items", {"title": "sample"}, "id")),
    ],
)
@pytest.mark.parametrize("fails", [False, True])
def test_writes_commit_success_or_rollback_failure_and_always_close_cursor(database, method, args, fails):
    """Every transactional write closes cursors and rolls back rejected statements."""
    handler, connection, cursor, connect = database
    cursor.fetchone.return_value = (5,)
    failure = pg.psycopg2.ProgrammingError("rejected statement")
    if fails:
        cursor.execute.side_effect = failure
        with pytest.raises(RuntimeError, match="Error: Failed to") as error:
            getattr(handler, method)(*args)
        assert error.value.__cause__ is failure
        connection.rollback.assert_called_once_with()
        connection.commit.assert_not_called()
    else:
        getattr(handler, method)(*args)
        connection.commit.assert_called_once_with()
        connection.rollback.assert_not_called()
    assert cursor.close.call_count >= 1
    assert connect.call_count == 1


@pytest.mark.parametrize(
    "name,allow_dot", [("items; DROP TABLE users", False), ("", False), ("public..items", True), ("public.items;", True)]
)
def test_invalid_identifiers_are_rejected(name, allow_dot):
    """Malformed identifiers cannot be interpolated into DDL paths."""
    with pytest.raises(ValueError, match="Error: Invalid"):
        pg._validate_identifier(name, allow_dot=allow_dot)


@pytest.mark.parametrize("exists", [True, False])
def test_delete_table_checks_existence_before_dropping(database, exists):
    """Missing tables do not generate a DROP or commit an empty transaction."""
    handler, connection, cursor, _ = database
    cursor.fetchone.return_value = (1,) if exists else None
    handler.delete_table("items")
    assert cursor.execute.call_count == (2 if exists else 1)
    assert connection.commit.call_count == int(exists)
    assert cursor.execute.call_args_list[0].args[1] == ("items",)
    cursor.close.assert_called_once_with()


@pytest.mark.parametrize("schema", [None, "custom"])
@pytest.mark.parametrize("rows,expected", [([], None), ([{"column_name": "shape"}], "shape")])
def test_geometry_metadata_respects_schema_and_missing_columns(database, schema, rows, expected):
    """Geometry discovery uses bound table/schema names and tolerates nonspatial tables."""
    handler, _, cursor, _ = database
    cursor.fetchall.return_value = rows
    assert handler.get_geometry_column("items", schema=schema) == expected
    assert cursor.execute.call_args.args[1] == ((schema, "items") if schema else ("items",))
    cursor.close.assert_called_once_with()


@pytest.mark.parametrize("exists", [True, False])
def test_record_existence_binds_condition_values(database, exists):
    """Existence checks preserve the server result and never interpolate user values."""
    handler, _, cursor, _ = database
    cursor.fetchone.return_value = (exists,)
    assert handler.record_exists("items", "title = %s", ("O'Brien",)) is exists
    assert cursor.execute.call_args.args[1] == ("O'Brien",)
    cursor.close.assert_called_once_with()


@pytest.mark.parametrize("rows", [[], [("items",), ("parcels",)]])
def test_list_tables_returns_names_and_closes_cursor(database, rows):
    """Listing supports both empty and populated schemas."""
    handler, _, cursor, _ = database
    cursor.fetchall.return_value = rows
    assert handler.list_tables() == [row[0] for row in rows]
    cursor.close.assert_called_once_with()


@pytest.mark.parametrize("method", ["create_delete_trigger", "create_update_column_on_source_change_trigger"])
@pytest.mark.parametrize("exists", [True, False])
def test_trigger_creation_is_idempotent(database, method, exists):
    """Existing triggers are skipped; new function/trigger pairs commit together."""
    handler, connection, cursor, _ = database
    cursor.fetchone.return_value = ("trigger",) if exists else None
    getattr(handler, method)("parents", "children", "parent_id", "child_id")
    assert cursor.execute.call_count == (1 if exists else 3)
    assert connection.commit.call_count == (0 if exists else 1)
    cursor.close.assert_called_once_with()


def test_factory_uses_environment_connection_options(database, monkeypatch):
    """The convenience factory forwards explicit environment settings to the driver."""
    _, _, _, connect = database
    for key, value in {
        "POSTGRES_USER": "unit-user",
        "POSTGRES_PASSWORD": "unit-only",
        "POSTGRES_HOST": "unit.invalid",
        "POSTGRES_PORT": "5440",
    }.items():
        monkeypatch.setenv(key, value)
    handler = pg.create_db_handler("unit_db")
    assert handler.db_name == "unit_db"
    assert connect.call_args.kwargs == {
        "dbname": "unit_db",
        "user": "unit-user",
        "password": "unit-only",
        "host": "unit.invalid",
        "port": "5440",
    }


@pytest.mark.parametrize("factory", [pg.create_tbl_ingestion, pg.create_tbl_users])
@pytest.mark.parametrize("fails", [False, True])
def test_table_factories_create_expected_schema_or_report_failure(monkeypatch, factory, fails):
    """Setup helpers forward geometry/schema options and surface table creation failures."""
    monkeypatch.setenv("COL_USER_ID", "account_id")
    handler = Mock()
    if fails:
        handler.create_table.side_effect = ValueError("invalid table")
        with pytest.raises(RuntimeError, match="Could not create the table"):
            factory(handler, "test_table")
        handler.insert_record.assert_not_called()
        return
    assert "Success" in factory(handler, "test_table")
    options = handler.create_table.call_args.kwargs
    assert options["table_name"] == "test_table"
    assert options["with_geometry"] is (factory is pg.create_tbl_ingestion)
    if factory is pg.create_tbl_users:
        assert options["columns"]["account_id"] == "SERIAL PRIMARY KEY"
        records = [entry.kwargs["data"] for entry in handler.insert_record.call_args_list]
        assert len(records) == len({record["product_type"] for record in records}) == 10
        assert all(record["username"] == record["password"] == "" for record in records)
