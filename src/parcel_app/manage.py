"""Explicit database migration and interactive operator password reset."""

import argparse
from getpass import getpass
from importlib.resources import files

from parcel_app.core import db
from parcel_app.core.security import hash_password


def migrate():
    """Apply every packaged, idempotent additive migration in name order, in one transaction."""
    scripts = sorted(
        (item for item in files("parcel_app").joinpath("migrations").iterdir() if item.name.endswith(".sql")),
        key=lambda item: item.name,
    )
    with db.connection() as conn, conn.cursor() as cursor:
        for script in scripts:
            cursor.execute(script.read_text(encoding="utf-8"))


def set_password(username: str):
    """Replace one confirmed user's password without command-line secrets."""
    password = getpass("New password (at least 12 characters): ")
    if not 12 <= len(password) <= 256:
        raise ValueError("Error: Password must contain 12 to 256 characters")
    if password != getpass("Confirm new password: "):
        raise ValueError("Error: Passwords do not match")
    with db.connection() as conn:
        user = db.one(
            "UPDATE app_users SET password=%s WHERE lower(username)=lower(%s) RETURNING app_user_id",
            (hash_password(password), username),
            conn=conn,
        )
        if user is None:
            raise ValueError("Error: User not found")
        db.execute("DELETE FROM app_sessions WHERE app_user_id=%s", (user["app_user_id"],), conn=conn)


def main():
    """Run one explicit operator action; never migrate on API startup."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("migrate")
    reset = commands.add_parser("set-password")
    reset.add_argument("username")
    args = parser.parse_args()
    if args.command == "migrate":
        migrate()
    else:
        set_password(args.username)


if __name__ == "__main__":
    main()
