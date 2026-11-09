"""Owner-scoped Copernicus account management; passwords are write-only."""

from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from psycopg2.errors import ForeignKeyViolation, UniqueViolation

from . import db
from .security import cipher, current_user

router = APIRouter(prefix="/accounts", tags=["accounts"])
SAFE_COLUMNS = "eo_user_id,username,label,test_status,tested_at"


class AccountInput(BaseModel):
    """Credentials supplied once by the owner."""

    username: str = Field(min_length=1, max_length=320)
    password: str = Field(min_length=1, max_length=4096)
    label: str = Field(default="", max_length=120)


class AccountPatch(BaseModel):
    """Mutable account fields; username identity remains fixed."""

    label: str | None = Field(default=None, max_length=120)
    password: str | None = Field(default=None, min_length=1, max_length=4096)


def owned(account_id: int, user: dict, *, conn=None) -> dict:
    """Find an account owned by the session without disclosing foreign IDs."""
    row = db.one(
        f"SELECT {SAFE_COLUMNS} FROM eo_users WHERE eo_user_id=%s AND app_user_id=%s",
        (account_id, user["app_user_id"]),
        conn=conn,
    )
    if row is None:
        raise HTTPException(404, "Error: Account not found")
    return row


@router.get("")
def list_accounts(user: dict = Depends(current_user)):
    """List only owned accounts with safe credential metadata."""
    return db.all_rows(f"SELECT {SAFE_COLUMNS} FROM eo_users WHERE app_user_id=%s ORDER BY eo_user_id", (user["app_user_id"],))


@router.post("", status_code=201)
def create_account(body: AccountInput, user: dict = Depends(current_user)):
    """Encrypt an EO password before storing it."""
    if not body.username.strip() or body.username != body.username.strip():
        raise HTTPException(422, "Error: Username cannot be blank or have surrounding whitespace")
    encrypted = cipher().encrypt(body.password.encode()).decode()
    try:
        return db.one(
            f"INSERT INTO eo_users(app_user_id,username,password,label) VALUES (%s,%s,%s,%s) RETURNING {SAFE_COLUMNS}",
            (user["app_user_id"], body.username, encrypted, body.label),
        )
    except UniqueViolation:
        raise HTTPException(409, "Error: Account already exists") from None


@router.patch("/{account_id}")
def update_account(account_id: int, body: AccountPatch, user: dict = Depends(current_user)):
    """Replace a password or label without exposing stored credentials."""
    owned(account_id, user)
    password = cipher().encrypt(body.password.encode()).decode() if body.password is not None else None
    return db.one(
        f"""UPDATE eo_users SET label=COALESCE(%s,label),password=COALESCE(%s,password),
        test_status=CASE WHEN %s IS NOT NULL THEN 'untested' ELSE test_status END,
        tested_at=CASE WHEN %s IS NOT NULL THEN NULL ELSE tested_at END
        WHERE eo_user_id=%s AND app_user_id=%s RETURNING {SAFE_COLUMNS}""",
        (body.label, password, password, password, account_id, user["app_user_id"]),
    )


@router.delete("/{account_id}", status_code=204)
def delete_account(account_id: int, user: dict = Depends(current_user)):
    """Delete an unreferenced owned account; preserve all submitted identities."""
    owned(account_id, user)
    try:
        db.execute("DELETE FROM eo_users WHERE eo_user_id=%s AND app_user_id=%s", (account_id, user["app_user_id"]))
    except ForeignKeyViolation:
        raise HTTPException(409, "Error: Account is referenced by an analysis") from None


@router.post("/{account_id}/test", status_code=202)
def test_account(account_id: int, user: dict = Depends(current_user)):
    """Queue authentication only, without creating an openEO processing job."""
    with db.connection() as conn:
        owned(account_id, user, conn=conn)
        job = db.one(
            """INSERT INTO web_jobs(job_id,app_user_id,kind,target_id) VALUES (%s,%s,'test_account',%s)
            ON CONFLICT(kind,target_id) WHERE status IN ('pending','running') DO UPDATE SET target_id=excluded.target_id
            RETURNING job_id""",
            (str(uuid4()), user["app_user_id"], str(account_id)),
            conn=conn,
        )
        db.execute("UPDATE eo_users SET test_status='checking' WHERE eo_user_id=%s", (account_id,), conn=conn)
        return job
