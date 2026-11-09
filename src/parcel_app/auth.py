"""Registration, login and revocable cookie sessions."""

import hashlib
import secrets

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field
from psycopg2.errors import UniqueViolation

from . import db
from .security import COOKIE_NAME, check_origin, current_user, hash_password, token_hash, verify_password
from .settings import get_settings

router = APIRouter(prefix="/auth", tags=["identity"])
_DUMMY_HASH = hash_password("not-a-real-user-password")


class Credentials(BaseModel):
    """Bounded login and registration input."""

    username: str = Field(min_length=3, max_length=80, pattern=r"^[A-Za-z0-9_.-]+$")
    password: str = Field(min_length=1, max_length=256)


def throttle(request: Request, username: str) -> None:
    """Persist a bounded per-client/username attempt budget across API restarts."""
    address = request.client.host if request.client else "unknown"
    attempts = []
    for bucket, limit in ((f"client:{address}", 100), (f"login:{username.lower()}", 20)):
        key = hashlib.sha256(bucket.encode()).hexdigest()
        row = db.one(
            """INSERT INTO auth_attempts(bucket_key,window_start,attempts) VALUES (%s,now(),1)
        ON CONFLICT(bucket_key) DO UPDATE SET
        attempts=CASE WHEN auth_attempts.window_start < now()-interval '15 minutes' THEN 1
                      ELSE auth_attempts.attempts+1 END,
        window_start=CASE WHEN auth_attempts.window_start < now()-interval '15 minutes' THEN now()
                          ELSE auth_attempts.window_start END RETURNING attempts""",
            (key,),
        )
        attempts.append(row["attempts"] > limit)
    # Expired keys cannot grow the table indefinitely across distinct login names.
    db.execute("DELETE FROM auth_attempts WHERE window_start < now()-interval '1 day'")
    if any(attempts):
        raise HTTPException(429, "Error: Too many attempts; try again later")


def _session(user: dict, response: Response, *, conn=None) -> dict:
    """Create an opaque session and return only browser-safe user data."""
    token, csrf = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
    settings = get_settings()
    db.execute(
        """INSERT INTO app_sessions(token_hash,app_user_id,csrf_token,expires_at)
                  VALUES (%s,%s,%s,now() + %s * interval '1 hour')""",
        (token_hash(token), user["app_user_id"], csrf, settings.session_hours),
        conn=conn,
    )
    response.set_cookie(
        COOKIE_NAME,
        token,
        max_age=settings.session_hours * 3600,
        httponly=True,
        secure=settings.cookie_secure,
        samesite="strict",
        path="/api",
    )
    return {"app_user_id": user["app_user_id"], "username": user["username"], "csrf_token": csrf}


@router.post("/register", status_code=201)
def register(body: Credentials, request: Request, response: Response):
    """Immediately register an active account with an Argon2id password."""
    check_origin(request)
    throttle(request, body.username)
    if len(body.password) < 12:
        raise HTTPException(422, "Error: Password must contain at least 12 characters")
    try:
        with db.connection() as conn:
            user = db.one(
                "INSERT INTO app_users(username,password,is_active) VALUES (%s,%s,true) RETURNING app_user_id,username",
                (body.username, hash_password(body.password)),
                conn=conn,
            )
            return _session(user, response, conn=conn)
    except UniqueViolation:
        raise HTTPException(409, "Error: Username is unavailable") from None


@router.post("/login")
def login(body: Credentials, request: Request, response: Response):
    """Authenticate using a generic failure response and a persisted attempt limit."""
    check_origin(request)
    throttle(request, body.username)
    user = db.one(
        "SELECT app_user_id,username,password,is_active FROM app_users WHERE lower(username)=lower(%s)", (body.username,)
    )
    valid = verify_password(
        user["password"] if user and user["password"].startswith("$argon2id$") else _DUMMY_HASH, body.password
    )
    if not user or not user["is_active"] or not valid:
        raise HTTPException(401, "Error: Invalid username or password")
    return _session(user, response)


@router.get("/me")
def me(user: dict = Depends(current_user)):
    """Return the active session identity and its CSRF token."""
    return user


@router.post("/logout", status_code=204)
def logout(request: Request, response: Response, user: dict = Depends(current_user)):
    """Revoke the server session and clear the browser cookie."""
    db.execute("DELETE FROM app_sessions WHERE token_hash=%s", (token_hash(request.cookies.get(COOKIE_NAME, "")),))
    response.delete_cookie(COOKIE_NAME, path="/api", secure=get_settings().cookie_secure, httponly=True, samesite="strict")
