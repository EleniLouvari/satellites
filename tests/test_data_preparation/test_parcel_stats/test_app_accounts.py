"""Verify scoped database credentials without accessing a real database."""

import sys
from types import ModuleType
from unittest.mock import MagicMock, Mock

import pytest
from cryptography.fernet import Fernet


@pytest.fixture
def accounts_db(monkeypatch):
    """Provide encrypted rows at the database boundary and a separate runtime key."""
    key = Fernet.generate_key()
    cipher = Fernet(key)
    monkeypatch.setenv("DB_NAME", "app")
    monkeypatch.setenv("SATELLITES_CREDENTIAL_KEY", key.decode())
    monkeypatch.delenv("TBL_APP_USERS", raising=False)
    monkeypatch.delenv("TBL_EO_USERS", raising=False)
    handler = MagicMock()
    cursor = handler.connection.cursor.return_value.__enter__.return_value
    cursor.fetchall.return_value = [
        (" Zeta ", cipher.encrypt(b" p,@word ").decode()),
        ("Alice", cipher.encrypt(b"secret").decode()),
    ]
    module = ModuleType("shared.postgress")
    module.create_db_handler = Mock(return_value=handler)
    monkeypatch.setitem(sys.modules, "shared.postgress", module)
    return handler, cursor, module.create_db_handler


def test_database_scope_and_decryption(accounts_db, capsys):
    """Bind the owner as a value, decrypt only returned rows and release the connection."""
    from data_preparation.parcel_stats.accounts import load_app_openeo_accounts

    handler, cursor, factory = accounts_db
    assert load_app_openeo_accounts(42) == [("Alice", "secret"), ("Zeta", " p,@word ")]
    factory.assert_called_once_with(db_name="app")
    query, values = cursor.execute.call_args.args
    assert values == (42,)
    assert "app_user_id" in str(query)
    assert "WHERE" in str(query)
    assert "JOIN" in str(query)
    handler.close_connection.assert_called_once()
    assert "secret" not in capsys.readouterr().out


@pytest.mark.parametrize("user_id", [None, 0, -1, True, "1 OR 1=1"])
def test_invalid_owner_never_connects(accounts_db, user_id):
    """Missing or malformed owner IDs cannot fall back to an unfiltered account read."""
    from data_preparation.parcel_stats.accounts import load_app_openeo_accounts

    with pytest.raises(ValueError, match="app_user_id"):
        load_app_openeo_accounts(user_id)
    accounts_db[2].assert_not_called()


def test_plaintext_or_wrong_key_fails_closed(accounts_db):
    """New application tables must not silently accept plaintext credentials."""
    from data_preparation.parcel_stats.accounts import load_app_openeo_accounts

    accounts_db[1].fetchall.return_value = [("Alice", "plaintext-secret")]
    with pytest.raises(ValueError, match="decrypt") as exc:
        load_app_openeo_accounts(42)
    assert "plaintext-secret" not in str(exc.value)
    accounts_db[0].close_connection.assert_called_once()


def test_no_accounts_does_not_use_legacy_table(accounts_db):
    """An unknown or disabled user cannot obtain another user's accounts."""
    from data_preparation.parcel_stats.accounts import load_app_openeo_accounts

    accounts_db[1].fetchall.return_value = []
    with pytest.raises(ValueError, match="No Copernicus accounts"):
        load_app_openeo_accounts(42)
    accounts_db[0].get_df_from_db.assert_not_called()


def test_missing_key_never_connects(accounts_db, monkeypatch):
    """Fail before database access if deployment cannot decrypt the credentials."""
    from data_preparation.parcel_stats.accounts import load_app_openeo_accounts

    monkeypatch.delenv("SATELLITES_CREDENTIAL_KEY")
    with pytest.raises(ValueError, match="SATELLITES_CREDENTIAL_KEY"):
        load_app_openeo_accounts(42)
    accounts_db[2].assert_not_called()


def test_database_failure_does_not_expose_credentials(accounts_db):
    """Driver exception payloads must not leak into user-facing error messages."""
    from data_preparation.parcel_stats.accounts import load_app_openeo_accounts

    accounts_db[1].execute.side_effect = RuntimeError("connection password=private")
    with pytest.raises(ValueError, match="database") as exc:
        load_app_openeo_accounts(42)
    assert "private" not in str(exc.value)
    accounts_db[0].close_connection.assert_called_once()
