"""Validate database account loading without connecting to PostgreSQL or openEO."""

import sys
from types import ModuleType
from unittest.mock import Mock

import pandas as pd
import pytest

from data_preparation.parcel_stats.multiuser import load_openeo_users_from_db


@pytest.fixture
def database(monkeypatch):
    """Provide a fake database module even when optional drivers are absent."""
    module = ModuleType("shared.postgress")
    handler = Mock()
    handler.get_df_from_db.return_value = pd.DataFrame({"username": ["user"], "password": ["secret"]})
    module.create_db_handler = Mock(return_value=handler)
    monkeypatch.setitem(sys.modules, "shared.postgress", module)
    monkeypatch.setenv("DB_NAME", "accounts")
    monkeypatch.setenv("TBL_USERS", "public.openeo_accounts")
    return module.create_db_handler, handler


def test_loads_normalized_sorted_accounts_preserving_passwords(database, capsys):
    """Database row order and delimiter characters must not alter credentials."""
    factory, handler = database
    handler.get_df_from_db.return_value = pd.DataFrame(
        {"username": [" zeta ", "Alice"], "password": [" p,@;:/word ", "secret"], "id": [1, 2]}
    )

    assert load_openeo_users_from_db() == [("Alice", "secret"), ("zeta", " p,@;:/word ")]
    factory.assert_called_once_with(db_name="accounts")
    handler.get_df_from_db.assert_called_once_with(table_name="public.openeo_accounts")
    handler.close_connection.assert_called_once_with()
    output = capsys.readouterr().out
    assert "Loaded 2" in output
    assert "secret" not in output
    assert "p,@;:/word" not in output


@pytest.mark.parametrize("name", ["DB_NAME", "TBL_USERS"])
@pytest.mark.parametrize("value", [None, "", "  "])
def test_missing_settings_fail_before_connecting(database, monkeypatch, name, value):
    """Never silently connect to a default database or an unspecified table."""
    factory, _ = database
    if value is None:
        monkeypatch.delenv(name)
    else:
        monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match="Error: Set both DB_NAME and TBL_USERS"):
        load_openeo_users_from_db()
    factory.assert_not_called()


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (pd.DataFrame(columns=["username", "password"]), "table is empty"),
        (pd.DataFrame({"username": ["user"]}), "username and password columns"),
        (pd.DataFrame({"password": ["secret"]}), "username and password columns"),
        (pd.DataFrame({"username": ["same", " SAME "], "password": ["a", "b"]}), "duplicate usernames"),
    ],
)
def test_invalid_table_contents_fail_with_clear_errors(database, rows, message):
    """Malformed tables cannot supply accounts to the scheduler."""
    _, handler = database
    handler.get_df_from_db.return_value = rows
    with pytest.raises(ValueError, match=message):
        load_openeo_users_from_db()
    handler.close_connection.assert_called_once_with()


@pytest.mark.parametrize("column", ["username", "password"])
@pytest.mark.parametrize("invalid", [None, pd.NA, float("nan"), 123, False, "", " \t"])
def test_invalid_credential_values_are_rejected(database, column, invalid):
    """Nulls, non-text values and whitespace-only values must fail early."""
    _, handler = database
    row = {"username": "user", "password": "secret"}
    row[column] = invalid
    handler.get_df_from_db.return_value = pd.DataFrame([row])
    with pytest.raises(ValueError, match="must be non-empty text"):
        load_openeo_users_from_db()
    handler.close_connection.assert_called_once_with()


def test_connection_failure_has_configuration_context(database):
    """Connection errors identify the configuration to check."""
    factory, handler = database
    factory.side_effect = ConnectionError("unavailable")
    with pytest.raises(ValueError, match="check POSTGRES"):
        load_openeo_users_from_db()
    handler.get_df_from_db.assert_not_called()


def test_read_failure_closes_connection(database):
    """Failed table reads must release the already opened connection."""
    _, handler = database
    handler.get_df_from_db.side_effect = RuntimeError("table missing")
    with pytest.raises(ValueError, match="check TBL_USERS and SELECT permissions"):
        load_openeo_users_from_db()
    handler.close_connection.assert_called_once_with()
