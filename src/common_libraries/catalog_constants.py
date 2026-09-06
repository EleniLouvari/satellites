"""Shared constants for catalog access and retry behavior."""

from __future__ import annotations

import os

import requests

# These names are forwarded to isolated workers when a reduced environment is built.
CATALOG_ENV_VARS = (
    "CATALOG_URL",
    "CATALOG_CLIENT_ID",
    "CATALOG_CLIENT_SECRET",
    "CATALOG_REQUEST_TIMEOUT_SECONDS",
    "KEYCLOAK_URL",
    "REALM_NAME",
)

# Resolve settings at import time to preserve the historical constants API.
BASE_URL = os.getenv("CATALOG_URL")
# Logical client roles map to their corresponding catalog credentials.
USERS = {
    "internal": os.getenv("CATALOG_CLIENT_SECRET"),
    "stageout": os.getenv("STAGEOUT_SECRET"),
    "admin": os.getenv("ADMIN_SECRET"),
}

# Request timeout covers one complete catalog HTTP exchange.
CATALOG_REQUEST_TIMEOUT_SECONDS = float(os.getenv("CATALOG_REQUEST_TIMEOUT_SECONDS", "50"))
# Connection failures and HTTP response failures have separate retry budgets.
MAX_CATALOG_CONNECTION_RETRIES = int(os.getenv("MAX_CATALOG_CONNECTION_RETRIES", "10"))
MAX_CATALOG_REQUEST_RETRIES = int(os.getenv("MAX_CATALOG_REQUEST_RETRIES", "10"))
# Exponential backoff is bounded so prolonged outages remain observable.
RETRY_BASE_SLEEP_SECONDS = float(os.getenv("RETRY_BASE_SLEEP_SECONDS", "1.0"))
RETRY_MAX_SLEEP_SECONDS = float(os.getenv("RETRY_MAX_SLEEP_SECONDS", "60.0"))
# Jitter prevents synchronized workers from retrying simultaneously.
RETRY_JITTER_SECONDS = float(os.getenv("RETRY_JITTER_SECONDS", "0.5"))
# Only transient server/throttling statuses are safe to replay.
RETRYABLE_HTTP_STATUS_CODES = {408, 429, 500, 502, 503, 504}
# Transport failures are retried before an HTTP response exists.
RETRYABLE_REQUEST_EXCEPTIONS = (
    requests.exceptions.Timeout,
    requests.exceptions.ConnectionError,
    requests.exceptions.ChunkedEncodingError,
)
