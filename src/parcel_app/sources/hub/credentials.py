"""Operator-owned HUB settings, never serialized into browser configuration."""

import os
from urllib.parse import urlsplit

REQUIRED_SETTINGS = ("CATALOG_URL", "KEYCLOAK_URL", "CATALOG_CLIENT_ID", "CATALOG_CLIENT_SECRET")
URL_SETTINGS = ("CATALOG_URL", "KEYCLOAK_URL")


def _is_service_url(value: str) -> bool:
    """Accept a bare http(s) URL; pasted link markup such as ``<https://...>`` is not one."""
    if any(character.isspace() or character in "<>\"'" for character in value):
        return False
    parts = urlsplit(value)
    return parts.scheme in {"http", "https"} and bool(parts.netloc)


def configuration_error() -> str | None:
    """Return browser-safe text naming missing or malformed settings, never their values."""
    if not all(os.getenv(name, "").strip() for name in REQUIRED_SETTINGS):
        return "Error: HUB catalog access is not configured on this server."
    invalid = [name for name in URL_SETTINGS if not _is_service_url(os.getenv(name, ""))]
    if invalid:
        return (
            f"Error: HUB setting {', '.join(invalid)} must be a plain http(s) URL without "
            "surrounding <>, quotes or spaces; correct the server environment file and restart."
        )
    return None


def configured() -> bool:
    """Check presence and URL form without authenticating or importing acquisition dependencies."""
    return configuration_error() is None


def require_configuration() -> None:
    """Fail with browser-safe text when the operator has not configured HUB correctly."""
    if error := configuration_error():
        raise ValueError(error)


def run_secrets() -> list[str]:
    """Supply secret values only to the existing log-redaction boundary."""
    return [value for name in ("CATALOG_CLIENT_SECRET", "ADMIN_SECRET", "STAGEOUT_SECRET") if (value := os.getenv(name))]
