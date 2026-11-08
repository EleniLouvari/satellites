"""Password hashing, encrypted credentials and server-side session checks."""
import hashlib
import secrets
from functools import lru_cache

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError
from cryptography.fernet import Fernet
from fastapi import HTTPException, Request

from . import db
from .settings import get_settings

_PASSWORDS = PasswordHasher()
COOKIE_NAME = "parcel_session"


def hash_password(password: str) -> str:
    """Hash application passwords with Argon2id."""
    return _PASSWORDS.hash(password)


def verify_password(hashed: str, password: str) -> bool:
    """Verify Argon2 hashes only; legacy plaintext always requires reset."""
    if not hashed.startswith("$argon2id$"):
        return False
    try:
        return _PASSWORDS.verify(hashed, password)
    except (InvalidHashError, VerificationError):
        return False


def token_hash(token: str) -> str:
    """Return the non-reversible stored representation of a session token."""
    return hashlib.sha256(token.encode()).hexdigest()


@lru_cache
def cipher() -> Fernet:
    """Load the existing Fernet key; never silently generate another key."""
    return Fernet(get_settings().credential_key.encode())


def check_origin(request: Request) -> None:
    """Reject cross-origin and originless browser mutations, including login."""
    if request.headers.get("origin", "").rstrip("/") != get_settings().allowed_origin:
        raise HTTPException(403, "Error: Untrusted request origin")


def current_user(request: Request) -> dict:
    """Authenticate the session and validate Origin plus CSRF on mutations."""
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        raise HTTPException(401, "Error: Sign in required")
    user = db.one("""SELECT u.app_user_id, u.username, s.csrf_token FROM app_sessions s
                     JOIN app_users u USING(app_user_id)
                     WHERE s.token_hash=%s AND s.expires_at > now() AND u.is_active""", (token_hash(token),))
    if not user:
        raise HTTPException(401, "Error: Sign in required")
    if request.method not in {"GET", "HEAD", "OPTIONS"}:
        check_origin(request)
        supplied = request.headers.get("x-csrf-token", "")
        if not secrets.compare_digest(supplied, user["csrf_token"]):
            raise HTTPException(403, "Error: Invalid CSRF token")
    return user
