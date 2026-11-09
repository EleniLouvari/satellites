"""Environment-backed web service settings."""

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Operator-controlled service limits and secrets."""

    storage_root: Path
    credential_key: str
    cookie_secure: bool
    allowed_origin: str
    upload_limit_bytes: int = 250 * 1024**2
    max_batch_workers: int = 4
    storage_limit_bytes: int = 100 * 1024**3
    session_hours: int = 12


@lru_cache
def get_settings() -> Settings:
    """Load settings without connecting to a database or creating directories."""
    return Settings(
        storage_root=Path(os.getenv("PARCEL_STORAGE_ROOT", "./parcel-storage")).resolve(),
        credential_key=os.getenv("SATELLITES_CREDENTIAL_KEY", ""),
        cookie_secure=os.getenv("PARCEL_COOKIE_SECURE", "true").lower() in {"true", "1", "yes"},
        allowed_origin=os.getenv("PARCEL_ALLOWED_ORIGIN", "https://localhost").rstrip("/"),
        upload_limit_bytes=max(1, int(os.getenv("PARCEL_UPLOAD_LIMIT_MB", "250"))) * 1024**2,
        max_batch_workers=max(1, int(os.getenv("PARCEL_MAX_BATCH_WORKERS", "4"))),
        storage_limit_bytes=max(1, int(os.getenv("PARCEL_STORAGE_LIMIT_GB", "100"))) * 1024**3,
    )
