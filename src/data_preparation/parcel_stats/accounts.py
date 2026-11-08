"""Load a single application user's encrypted Copernicus accounts from PostgreSQL."""

from __future__ import annotations

import os

import pandas as pd

from .multiuser import _validate_openeo_accounts


def _credential_cipher():
    """Load the deployment encryption key independently of database contents."""
    from cryptography.fernet import Fernet

    key = os.getenv("SATELLITES_CREDENTIAL_KEY", "")
    try:
        return Fernet(key.encode("ascii"))
    except (ValueError, UnicodeError):
        raise ValueError("Error: Set SATELLITES_CREDENTIAL_KEY to a valid Fernet key supplied by the deployment.") from None


def load_app_openeo_accounts(app_user_id: int) -> list[tuple[str, str]]:
    """Read only an active owner's accounts, decrypting passwords in memory.

    Uses ``DB_NAME`` and ``POSTGRES_*`` as the existing notebook loader does.
    ``TBL_APP_USERS`` and ``TBL_EO_USERS`` default to ``public.app_users`` and
    ``public.eo_users``. The latter's ``password`` column contains Fernet tokens.
    The caller must authenticate/authorize the owner before invoking this helper;
    an integer ID supplied by a CLI operator is not itself authentication.
    """
    if type(app_user_id) is not int or app_user_id <= 0:
        raise ValueError("Error: app_user_id must be a positive integer.")
    db_name = os.getenv("DB_NAME", "").strip()
    if not db_name:
        raise ValueError("Error: Set DB_NAME before loading application accounts.")
    cipher = _credential_cipher()
    from psycopg2 import sql
    from shared.postgress import create_db_handler

    app_table = os.getenv("TBL_APP_USERS", "public.app_users")
    eo_table = os.getenv("TBL_EO_USERS", "public.eo_users")
    # Identifiers are quoted, while the owner is a bound value; no interpolated WHERE clauses.
    query = sql.SQL(
        "SELECT eo.username, eo.password FROM {} AS eo "
        "JOIN {} AS app ON app.app_user_id = eo.app_user_id "
        "WHERE eo.app_user_id = %s AND app.is_active = TRUE ORDER BY eo.eo_user_id"
    ).format(sql.Identifier(*eo_table.split(".")), sql.Identifier(*app_table.split(".")))
    handler = None
    try:
        handler = create_db_handler(db_name=db_name)
        with handler.connection.cursor() as cursor:
            cursor.execute(query, (app_user_id,))
            rows = cursor.fetchall()
    except Exception:
        raise ValueError("Error: Could not read application accounts; check database settings, tables and permissions.") from None
    finally:
        if handler is not None:
            handler.close_connection()
    if not rows:
        raise ValueError("Error: No Copernicus accounts are available for this active application user.")
    try:
        decrypted = [(username, cipher.decrypt(token.encode("ascii")).decode("utf-8")) for username, token in rows]
    except Exception:
        raise ValueError("Error: Could not decrypt Copernicus credentials; check the encryption key and stored tokens.") from None
    return _validate_openeo_accounts(pd.DataFrame(decrypted, columns=["username", "password"]))
