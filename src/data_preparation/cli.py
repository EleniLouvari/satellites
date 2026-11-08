"""Command-line interface for offline validation and unattended parcel statistics."""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    """Define commands without importing optional geospatial or acquisition dependencies."""
    parser = argparse.ArgumentParser(prog="satellites")
    commands = parser.add_subparsers(dest="pipeline", required=True)
    parcel = commands.add_parser("parcel-stats", help="Validate, run or resume openEO parcel statistics")
    actions = parcel.add_subparsers(dest="action", required=True)
    for action in ("validate", "run", "resume"):
        command = actions.add_parser(action)
        if action == "resume":
            command.add_argument("--run-dir", type=Path, required=True)
        else:
            command.add_argument("--config", type=Path, required=True, help="JSON configuration file")
            command.add_argument("--output-dir", type=Path, help="Override output directory (relative to current directory)")
        if action != "validate":
            scope = command.add_mutually_exclusive_group(required=True)
            scope.add_argument("--app-user-id", type=int, help="Load this user's accounts from app_users/eo_users")
            scope.add_argument(
                "--legacy-accounts-db", action="store_true", help="Explicitly use all accounts from legacy TBL_USERS"
            )
    return parser


def _read_config(args) -> dict:
    """Read JSON without executing Python or silently accepting malformed files."""
    try:
        config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        raise ValueError("Error: Cannot read a valid JSON configuration file.") from None
    if not isinstance(config, dict):
        raise ValueError("Error: Configuration must be a JSON object.")
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir.resolve())
    return config


def _error_message(exc: Exception, accounts: list) -> str:
    """Redact known runtime secrets from concise CLI errors; never emit a traceback."""
    message = str(exc)
    secrets = [password for _, password in accounts]
    secrets.extend(os.getenv(name, "") for name in ("POSTGRES_PASSWORD", "SATELLITES_CREDENTIAL_KEY"))
    for secret in sorted(filter(None, secrets), key=len, reverse=True):
        message = message.replace(secret, "[redacted]")
    return message if message.startswith("Error:") else f"Error: {message}"


def main(argv: list[str] | None = None) -> int:
    """Return 0 on success, 2 on invalid input/setup, 1 on run failure or 130 on interrupt."""
    args = _parser().parse_args(argv)
    accounts = []
    running = False
    try:
        from data_preparation.parcel_stats.run_config import prepare_run
        from data_preparation.parcel_stats.runner import resume_parcel_stats, run_parcel_stats

        # Existing pipeline progress is textual. Reserve stdout for one JSON result.
        with redirect_stdout(sys.stderr):
            config = _read_config(args) if args.action != "resume" else None
            if args.action == "validate":
                result = prepare_run(config, base_dir=args.config.resolve().parent).summary()
            else:
                if args.app_user_id is not None:
                    from data_preparation.parcel_stats.accounts import load_app_openeo_accounts

                    accounts = load_app_openeo_accounts(args.app_user_id)
                else:
                    from data_preparation.parcel_stats.multiuser import load_openeo_users_from_db

                    accounts = load_openeo_users_from_db()
                running = True
                if args.action == "resume":
                    result = resume_parcel_stats(args.run_dir, accounts=accounts, app_user_id=args.app_user_id)
                else:
                    result = run_parcel_stats(
                        config, accounts=accounts, base_dir=args.config.resolve().parent, app_user_id=args.app_user_id
                    )
        print(json.dumps(result, indent=2))
        return 0
    except KeyboardInterrupt:
        print("Error: Interrupted. Use resume with the same run directory; remote jobs may still be running.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(_error_message(exc, accounts), file=sys.stderr)
        return 1 if running else 2


if __name__ == "__main__":
    raise SystemExit(main())
