"""Bounded recovery of completed openEO job assets without resubmitting jobs."""

from __future__ import annotations

import time
from contextlib import contextmanager
from copy import copy
from pathlib import Path

import requests
from openeo.rest import OpenEoApiError, OpenEoApiPlainError

DOWNLOAD_ATTEMPTS = 3
DOWNLOAD_TIMEOUT = (30, 120)
DOWNLOAD_CHUNK_BYTES = 1024**2
PROGRESS_SECONDS = 30
RETRYABLE_HTTP_STATUSES = {408, 429, 500, 502, 503, 504}


@contextmanager
def _transfer_connection(connection):
    """Clone the SDK transport with independent pools and no nested adapter retries.

    Keep SDK authentication and external-domain credential handling, plus session TLS/proxy
    settings. Cloned Requests adapters create their own pools, which are closed here without
    changing the shared connection's adapter retry policy or cookies.
    """
    transfer = copy(connection)
    session = copy(connection.session)
    session.headers = connection.session.headers.copy()
    session.cookies = connection.session.cookies.copy()
    session.proxies = connection.session.proxies.copy()
    session.params = connection.session.params.copy()
    session.hooks = {name: list(hooks) for name, hooks in connection.session.hooks.items()}
    session.adapters = {prefix: copy(adapter) for prefix, adapter in connection.session.adapters.items()}
    for adapter in session.adapters.values():
        adapter.max_retries = requests.adapters.Retry(total=0, read=0, connect=0, status=0)
    transfer.session = session
    with session:
        yield transfer


def _retryable_transfer(error):
    """Retry transport interruptions and transient HTTP errors, excluding TLS failures."""
    if isinstance(error, requests.exceptions.SSLError):
        return False
    if isinstance(
        error, (requests.exceptions.Timeout, requests.exceptions.ConnectionError, requests.exceptions.ChunkedEncodingError)
    ):
        return True
    status = getattr(error, "http_status_code", None)
    if isinstance(error, requests.exceptions.HTTPError) and error.response is not None:
        status = error.response.status_code
    return status in RETRYABLE_HTTP_STATUSES


def _stream_job_asset(job, temporary_path, logger):
    """Stream one asset with scoped idle timeout, progress and unencoded length validation."""
    # Resolve the asset again on retry so expiring result URLs can be refreshed by the SDK.
    asset = job.get_results().get_asset()
    received = 0
    last_progress = time.monotonic()
    # HTTP errors are checked inside the response context so their streams are closed as well.
    with _transfer_connection(job.connection) as connection, connection.get(
        asset.href, stream=True, timeout=DOWNLOAD_TIMEOUT, headers={"Accept-Encoding": "identity"}, check_error=False
    ) as response:
        response.raise_for_status()
        with temporary_path.open("wb") as output:
            for block in response.iter_content(chunk_size=DOWNLOAD_CHUNK_BYTES):
                if not block:
                    continue
                output.write(block)
                received += len(block)
                now = time.monotonic()
                if now - last_progress >= PROGRESS_SECONDS:
                    logger.info(f"Downloading {temporary_path.name}: received {received} bytes.")
                    last_progress = now
        if received == 0:
            raise requests.exceptions.ChunkedEncodingError("Error: Empty asset transfer cannot contain a NetCDF cube.")
        # iter_content decodes content encodings; a compressed Content-Length describes different bytes.
        length = response.headers.get("Content-Length")
        if (
            length is not None
            and response.headers.get("Content-Encoding", "identity").lower() == "identity"
            and received != int(length)
        ):
            raise requests.exceptions.ChunkedEncodingError(
                f"Error: Incomplete asset transfer: received {received} of {length} bytes."
            )
    return received


def download_job_netcdf(job, temporary_path, target_path, logger):
    """Retry one completed job's transfer three times and atomically publish only a complete asset.

    Payload requests have a 30-second connection timeout and a 120-second read-idle timeout.
    Transient failures restart the transfer from byte zero after 30/60 seconds; permanent
    HTTP errors and local filesystem failures propagate immediately. URLs are never logged.
    """
    temporary_path, target_path = Path(temporary_path), Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        temporary_path.unlink(missing_ok=True)
        logger.info(f"Downloading {target_path.name}: attempt {attempt}/{DOWNLOAD_ATTEMPTS}.")
        try:
            received = _stream_job_asset(job, temporary_path, logger)
        except (requests.exceptions.RequestException, OpenEoApiError, OpenEoApiPlainError) as error:
            if not _retryable_transfer(error) or attempt == DOWNLOAD_ATTEMPTS:
                raise
            delay = 30 * 2 ** (attempt - 1)
            # Exception messages can include signed URLs; report only their type and the safe filename.
            logger.warning(f"Transfer of {target_path.name} interrupted ({type(error).__name__}); retrying in {delay} seconds.")
            time.sleep(delay)
            continue
        temporary_path.replace(target_path)
        logger.info(f"Downloaded {target_path.name}: {received} bytes; committed complete asset.")
        return target_path
    raise RuntimeError("Error: Asset download exhausted its attempt budget.")
