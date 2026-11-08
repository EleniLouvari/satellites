"""Identity boundaries and password/session security."""
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from parcel_app import auth, security


def client():
    """Build an isolated auth-only app."""
    app = FastAPI()
    app.include_router(auth.router, prefix="/api")
    return TestClient(app)


def test_password_hash_never_accepts_plaintext():
    hashed = security.hash_password("a long test password")
    assert hashed.startswith("$argon2id$")
    assert security.verify_password(hashed, "a long test password")
    assert not security.verify_password("a long test password", "a long test password")
    assert not security.verify_password(hashed, "incorrect")


def test_login_requires_trusted_origin():
    response = client().post("/api/auth/login", json={"username": "alice", "password": "long password"})
    assert response.status_code == 403


def test_login_failure_is_generic(monkeypatch):
    monkeypatch.setenv("PARCEL_ALLOWED_ORIGIN", "https://localhost")
    with patch.object(auth, "throttle"), patch.object(auth.db, "one", return_value=None):
        response = client().post("/api/auth/login", headers={"Origin": "https://localhost"},
                                 json={"username": "alice", "password": "long password"})
    assert response.status_code == 401
    assert response.json() == {"detail": "Error: Invalid username or password"}


def test_csrf_and_owner_come_from_session():
    user = {"app_user_id": 42, "username": "alice", "csrf_token": "secret"}
    app = FastAPI()
    @app.post("/mutate")
    def mutate(request: __import__("fastapi").Request):
        return security.current_user(request)
    with patch.object(security.db, "one", return_value=user):
        with TestClient(app) as test_client:
            test_client.cookies.set("parcel_session", "opaque")
            assert test_client.post("/mutate", headers={"Origin": "https://localhost"}).status_code == 403
            response = test_client.post("/mutate", headers={"Origin": "https://localhost", "X-CSRF-Token": "secret"})
            assert response.json()["app_user_id"] == 42


def test_logout_revokes_server_session():
    app = FastAPI()
    app.include_router(auth.router, prefix="/api")
    app.dependency_overrides[security.current_user] = lambda: {"app_user_id": 42}
    with patch.object(auth.db, "execute") as execute:
        test_client = TestClient(app)
        test_client.cookies.set("parcel_session", "opaque")
        response = test_client.post("/api/auth/logout")
        assert response.status_code == 204
        assert execute.call_args.args[1] == (security.token_hash("opaque"),)

@pytest.mark.skipif(__import__("os").getenv("PARCEL_TEST_DB") != "1", reason="requires disposable PostgreSQL")
def test_database_identity_registration_isolation_and_revocation(monkeypatch):
    """Exercise real SQL, Argon2 storage, owner scopes, CSRF and session logout."""
    from uuid import uuid4
    from cryptography.fernet import Fernet
    from parcel_app import accounts, db
    from parcel_app.manage import migrate
    from parcel_app.settings import get_settings

    monkeypatch.setenv("PARCEL_COOKIE_SECURE", "false")
    monkeypatch.setenv("SATELLITES_CREDENTIAL_KEY", Fernet.generate_key().decode())
    get_settings.cache_clear()
    security.cipher.cache_clear()
    migrate()
    migrate()  # Deployment retries must leave users and identities intact.
    app = FastAPI()
    app.include_router(auth.router, prefix="/api")
    app.include_router(accounts.router, prefix="/api")
    first, second = TestClient(app), TestClient(app)
    origin = {"Origin": "https://localhost"}
    username = f"test_{uuid4().hex}"
    response = first.post("/api/auth/register", headers=origin,
                          json={"username": username, "password": "long test password"})
    assert response.status_code == 201, response.text
    identity = response.json()
    assert first.get("/api/auth/me").json() == identity
    stored = db.one("SELECT password,is_active FROM app_users WHERE app_user_id=%s", (identity["app_user_id"],))
    assert stored["password"].startswith("$argon2id$") and stored["is_active"]
    headers = {**origin, "X-CSRF-Token": identity["csrf_token"]}
    account = first.post("/api/accounts", headers=headers,
                         json={"username": "example@example.org", "password": "EO secret", "label": "My EO"})
    assert account.status_code == 201, account.text
    assert "password" not in account.json()
    account_id = account.json()["eo_user_id"]
    encrypted = db.one("SELECT password FROM eo_users WHERE eo_user_id=%s", (account_id,))["password"]
    assert encrypted != "EO secret"
    assert security.cipher().decrypt(encrypted.encode()) == b"EO secret"
    assert first.patch(f"/api/accounts/{account_id}", headers=origin, json={"label": "forbidden"}).status_code == 403
    other = second.post("/api/auth/register", headers=origin,
                        json={"username": f"test_{uuid4().hex}", "password": "other long password"}).json()
    assert second.get("/api/accounts").json() == []
    assert second.patch(f"/api/accounts/{account_id}", headers={**origin, "X-CSRF-Token": other["csrf_token"]},
                        json={"label": "forbidden"}).status_code == 404
    assert first.post("/api/auth/logout", headers=headers).status_code == 204
    assert first.get("/api/auth/me").status_code == 401
    assert db.one("SELECT count(*) AS n FROM app_sessions WHERE app_user_id=%s", (identity["app_user_id"],))["n"] == 0
    get_settings.cache_clear()
    security.cipher.cache_clear()
