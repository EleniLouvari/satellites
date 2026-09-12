"""Library containing functions for s3 connection."""

from typing import Iterator
from typing import Optional
import logging
import os
import time
import boto3
import botocore
from botocore.config import Config
from botocore.exceptions import EndpointConnectionError, ReadTimeoutError

from boto3.s3.transfer import TransferConfig

import satellites.shared.logging as log_l
from botocore.client import BaseClient


def _safe_int_env(env_name: str, default_value: int, min_value: int = 1) -> int:
    """Read an integer environment variable with safe fallback.

    Parameters
    ----------
    env_name : str
        Environment variable name.
    default_value : int
        Fallback value when variable is missing or invalid.
    min_value : int, optional
        Minimum accepted value.

    Returns
    -------
    int
        Parsed integer, clamped to min_value.

    """
    raw_value = os.getenv(env_name)
    if raw_value is None:
        return default_value
    try:
        return max(min_value, int(raw_value))
    except (TypeError, ValueError):
        return default_value


def _safe_bool_env(env_name: str, default_value: bool) -> bool:
    """Read a boolean environment variable with safe fallback."""
    raw_value = os.getenv(env_name)
    if raw_value is None:
        return default_value
    return raw_value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _safe_float_env(env_name: str, default_value: float, min_value: float = 0.0) -> float:
    """Read a float environment variable with safe fallback.

    Parameters
    ----------
    env_name : str
        Environment variable name.
    default_value : float
        Fallback value when variable is missing or invalid.
    min_value : float, optional
        Minimum accepted value.

    Returns
    -------
    float
        Parsed float, clamped to min_value.

    """
    raw_value = os.getenv(env_name)
    if raw_value is None:
        return default_value
    try:
        return max(min_value, float(raw_value))
    except (TypeError, ValueError):
        return default_value


def _is_retryable_upload_exception(exc: Exception) -> bool:
    """Return True when an upload exception is likely transient.

    Parameters
    ----------
    exc : Exception
        Exception raised by S3 upload call.

    Returns
    -------
    bool
        True when the exception should trigger a retry.

    """
    transient_types = (
        ReadTimeoutError,
        EndpointConnectionError,
        botocore.exceptions.ConnectTimeoutError,
        botocore.exceptions.ConnectionClosedError,
    )
    if isinstance(exc, transient_types):
        return True

    msg = str(exc).lower()
    retry_hints = (
        "read timeout",
        "connect timeout",
        "timed out",
        "connection reset",
        "connection aborted",
        "temporarily unavailable",
        "throttl",
        "slowdown",
    )
    return any(hint in msg for hint in retry_hints)


def get_s3_transfer_config() -> TransferConfig:
    """Build a TransferConfig for managed S3 uploads."""
    multipart_threshold_mb = _safe_int_env("S3_MULTIPART_THRESHOLD_MB", default_value=64, min_value=5)
    multipart_chunksize_mb = _safe_int_env("S3_MULTIPART_CHUNKSIZE_MB", default_value=16, min_value=5)
    max_concurrency = _safe_int_env("S3_TRANSFER_MAX_CONCURRENCY", default_value=6, min_value=1)
    use_threads = _safe_bool_env("S3_TRANSFER_USE_THREADS", default_value=True)

    mb = 1024 * 1024
    return TransferConfig(
        multipart_threshold=multipart_threshold_mb * mb,
        multipart_chunksize=multipart_chunksize_mb * mb,
        max_concurrency=max_concurrency,
        use_threads=use_threads,
    )


def get_boto3_client(
    *,
    key: str | None = None,
    secret: str | None = None,
    region: str | None = None,
    endpoint_url: str | None = None,
    session_token: str | None = None,
    addressing_style: str | None = None,  # "virtual" or "path"
    verify: bool | str | None = None,  # True/False or path to CA bundle
) -> BaseClient:
    """Create and return a boto3 S3 client.

    This function initializes a boto3 S3 client using provided parameters or environment variables.
    It supports both AWS S3 and S3-compatible endpoints by allowing custom endpoint URLs and
    addressing styles. The client is configured with reasonable defaults for timeouts and retries.

    Returns
    -------
    boto3.client
        A boto3 S3 client instance configured with the provided credentials.

    """
    key = os.getenv("AWS_ACCESS_KEY_ID") if key is None else key
    secret = os.getenv("AWS_SECRET_ACCESS_KEY") if secret is None else secret
    session_token = os.getenv("AWS_SESSION_TOKEN") if session_token is None else session_token
    endpoint_url = os.getenv("S3_ENDPOINT_URL") if endpoint_url is None else endpoint_url
    region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") if region is None else region
    max_pool_connections = _safe_int_env("S3_MAX_POOL_CONNECTIONS", default_value=32, min_value=1)
    tcp_keepalive = _safe_bool_env("S3_TCP_KEEPALIVE", default_value=True)

    # Safer default for S3-compatible endpoints.
    if addressing_style is None:
        addressing_style = "path" if endpoint_url else "virtual"

    connect_timeout = _safe_int_env("S3_CONNECT_TIMEOUT", default_value=60, min_value=1)
    read_timeout = _safe_int_env("S3_READ_TIMEOUT", default_value=300, min_value=1)
    retry_attempts = _safe_int_env("S3_MAX_RETRY_ATTEMPTS", default_value=10, min_value=1)
    retry_mode = os.getenv("S3_RETRY_MODE", "adaptive")

    cfg = Config(
        signature_version="s3v4",
        s3={"addressing_style": addressing_style},
        connect_timeout=connect_timeout,
        read_timeout=read_timeout,
        retries={"max_attempts": retry_attempts, "mode": retry_mode},
        max_pool_connections=max_pool_connections,
        tcp_keepalive=tcp_keepalive,
    )

    return boto3.client(
        "s3",
        aws_access_key_id=key,
        aws_secret_access_key=secret,
        aws_session_token=session_token,
        region_name=region,
        endpoint_url=endpoint_url,
        verify=True if verify is None else verify,
        config=cfg,
    )


def delete_s3_folder(
    s3_client: BaseClient, bucket_name: str, s3_folder_prefix: str, logger: Optional[logging.Logger] = None
) -> bool:
    """Delete all objects under a specific folder (prefix) in an S3 bucket.

    This function lists and deletes all objects with the given prefix
    (representing a "folder" in S3 terms). It paginates through all
    matching keys and deletes them in batches of 1000, as per the S3 API limit.

    Parameters
    ----------
    s3_client : boto3.client
        An initialized boto3 S3 client.
    bucket_name : str
        Name of the S3 bucket.
    s3_folder_prefix : str
        Prefix (folder path) in the bucket to delete. A trailing slash will be enforced.
    logger : Optional[logging.Logger], optional
        Logger instance for logging messages. If None, logging is skipped or uses defaults.

    Returns
    -------
    bool
        True if all objects were deleted successfully or no objects were found.
        False if an error occurred during deletion.

    """
    try:
        # Ensure the prefix ends with '/'
        if not s3_folder_prefix.endswith("/"):
            s3_folder_prefix += "/"

        # List and delete all objects in the prefix
        objects_to_delete = []
        list_kwargs = {"Bucket": bucket_name, "Prefix": s3_folder_prefix}

        while True:
            response = s3_client.list_objects_v2(**list_kwargs)
            if "Contents" in response:
                objects_to_delete.extend([{"Key": obj["Key"]} for obj in response["Contents"]])

            if response.get("NextContinuationToken"):
                list_kwargs["ContinuationToken"] = response["NextContinuationToken"]
            else:
                break

        if not objects_to_delete:
            log_l.log_message(logger, f"No files found in s3://{bucket_name}/{s3_folder_prefix}")
            return True

        # Delete in batches of 1000 (S3 API limit)
        for i in range(0, len(objects_to_delete), 1000):
            delete_batch = objects_to_delete[i : i + 1000]
            delete_response = s3_client.delete_objects(Bucket=bucket_name, Delete={"Objects": delete_batch})
            if "Errors" in delete_response:
                errors = delete_response["Errors"]
                log_l.log_message(logger, f"Error 4003: Failed to delete {len(errors)} objects: {errors}", type="error")
                return False  # Failure in batch deletion

        log_l.log_message(logger, f"Success: Deleted {len(objects_to_delete)} objects from s3://{bucket_name}/{s3_folder_prefix}")
        return True

    except botocore.exceptions.ClientError as e:
        error_code = e.response["Error"]["Code"]
        error_msg = f"Error: Failed to delete folder s3://{bucket_name}/{s3_folder_prefix}: {e}, Error code: {error_code}"
        log_l.log_message(logger, error_msg)

        return False
    except Exception as e:
        error_msg = f"Error: Unexpected error during folder deletion: {e}"
        log_l.log_message(logger, error_msg, type="error")
        return False


def delete_s3_file(s3_client: BaseClient, bucket_name: str, s3_key: str, logger: Optional[logging.Logger] = None) -> bool:
    """Delete a single object from an S3 bucket.

    Parameters
    ----------
    s3_client : boto3.client
        An initialized boto3 S3 client.
    bucket_name : str
        Name of the S3 bucket.
    s3_key : str
        Full key (path) of the object to delete.
    logger : Optional[logging.Logger], optional
        Logger instance for logging messages. If None, logging is skipped or uses defaults.

    Returns
    -------
    bool
        True if the object was deleted successfully.
        False if an error occurred during deletion.

    """
    try:
        s3_client.delete_object(Bucket=bucket_name, Key=s3_key)
        log_l.log_message(logger, f"Success: Deleted s3://{bucket_name}/{s3_key}")
        return True
    except botocore.exceptions.ClientError as e:
        error_code = e.response["Error"]["Code"]
        error_msg = f"Error: Failed to delete s3://{bucket_name}/{s3_key}: {e}, Error code: {error_code}"
        log_l.log_message(logger, error_msg, type="error")
        return False
    except Exception as e:
        error_msg = f"Error: Unexpected error during file deletion: {e}"
        log_l.log_message(logger, error_msg, type="error")
        return False


def get_s3_object_size(
    bucket_name: str,
    s3_key: str,
    s3_client: Optional[BaseClient] = None,
    logger: Optional[logging.Logger] = None,
    verbose: bool = True,
) -> int | None:
    """Return the size in bytes of an S3 object.

    This function uses the S3 `head_object` operation to retrieve object metadata.
    It returns None when the object does not exist or when the size cannot be
    determined due to an error.

    Parameters
    ----------
    logger : Optional[logging.Logger]
        Logger instance for logging messages. If None, logging will use defaults or be skipped.
    bucket_name : str
        Name of the S3 bucket to check.
    s3_key : str
        Full key (path) of the object in the bucket.
    s3_client : Optional[BaseClient], optional
        Existing boto3 S3 client to use. If None, a new client is created.
    verbose : bool, optional
        If True, print messages to the console.

    Returns
    -------
    int | None
        Object size in bytes when the object exists, otherwise None.

    """
    if s3_client is None:
        s3_client = get_boto3_client()
    try:
        response = s3_client.head_object(Bucket=bucket_name, Key=s3_key)
        content_length = response.get("ContentLength")
        if verbose:
            log_l.log_message(
                logger,
                f"Success: File exists in s3://{bucket_name}/{s3_key} with size {content_length} bytes",
            )
        return int(content_length) if content_length is not None else None
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            if verbose:
                log_l.log_message(logger, f"File does not exist in s3://{bucket_name}/{s3_key}", type="warning")
            return None
        if verbose:
            log_l.log_message(logger, f"Error checking file in s3: {e}", type="error")
        return None
    except Exception as e:
        log_l.log_message(logger, f"Error: Unexpected error while checking file existence: {e}", type="error")
        return None


def check_file_exists_in_s3(
    bucket_name: str,
    s3_key: str,
    s3_client: Optional[BaseClient] = None,
    logger: Optional[logging.Logger] = None,
    verbose: bool = True,
) -> bool:
    """Check if a file (object) exists in an S3 bucket at the specified key.

    Parameters
    ----------
    logger : Optional[logging.Logger]
        Logger instance for logging messages. If None, logging will use defaults or be skipped.
    bucket_name : str
        Name of the S3 bucket to check.
    s3_key : str
        Full key (path) of the object in the bucket.
    s3_client : Optional[BaseClient], optional
        Existing boto3 S3 client to use. If None, a new client is created.
    verbose : bool, optional
        If True, print messages to the console.

    Returns
    -------
    bool
        True if the object exists, False otherwise (including on errors).

    """
    return (
        get_s3_object_size(
            bucket_name=bucket_name,
            s3_key=s3_key,
            s3_client=s3_client,
            logger=logger,
            verbose=verbose,
        )
        is not None
    )


def get_s3_subfolders(s3_client, bucket_name, prefix=None):
    """List immediate subfolders under a given prefix in an S3 bucket.

    Uses the S3 ListObjectsV2 API with a delimiter to emulate folder listing.
    Returns only the "folders" (common prefixes) directly under the given prefix.

    Parameters
    ----------
    s3_client : boto3.client
        An initialized boto3 S3 client.
    bucket_name : str
        Name of the S3 bucket.
    prefix : Optional[str], optional
        Prefix (folder path) to list subfolders under. If None, lists top-level folders.

    Returns
    -------
    list[str]
        A list of subfolder prefixes (each ending with '/').

    """
    list_kwargs = {"Bucket": bucket_name, "Delimiter": "/"}
    if prefix:
        list_kwargs["Prefix"] = prefix

    subfolders: list[str] = []
    while True:
        response = s3_client.list_objects_v2(**list_kwargs)
        subfolders.extend([entry["Prefix"] for entry in response.get("CommonPrefixes", [])])

        next_token = response.get("NextContinuationToken")
        if not next_token:
            break
        list_kwargs["ContinuationToken"] = next_token

    return subfolders


def get_s3_folder_files(s3_client, bucket_name, prefix):
    """List all file keys under a given prefix (folder) in an S3 bucket.

    This function retrieves all objects with the given prefix and filters out
    any "folder" placeholders (keys ending with '/').

    Parameters
    ----------
    s3_client : boto3.client
        An initialized boto3 S3 client.
    bucket_name : str
        Name of the S3 bucket.
    prefix : str
        The prefix (folder path) to list files under.

    Returns
    -------
    list[str]
        A list of object keys representing files under the given prefix.

    """
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    return [obj["Key"] for obj in response.get("Contents", []) if not obj["Key"].endswith("/")]


def ensure_s3_prefix(s3_client: BaseClient, bucket_name: str, prefix: str, logger: Optional[logging.Logger] = None) -> str:
    """Ensure an S3 prefix (folder) exists by creating a placeholder object if needed.

    Parameters
    ----------
    s3_client : boto3.client
        An initialized boto3 S3 client.
    bucket_name : str
        Name of the S3 bucket.
    prefix : str
        Prefix (folder path) to ensure. A trailing slash will be enforced.
    logger : Optional[logging.Logger], optional
        Logger instance for logging messages.

    Returns
    -------
    str
        Normalized prefix (always ending with '/').

    """
    if not prefix.endswith("/"):
        prefix += "/"

    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix, MaxKeys=1)
    if "Contents" not in response:
        s3_client.put_object(Bucket=bucket_name, Key=prefix)
        log_l.log_message(logger, f"Created folder s3://{bucket_name}/{prefix}")
    else:
        log_l.log_message(logger, f"Folder exists s3://{bucket_name}/{prefix}")

    return prefix


def list_s3_contents(
    s3_client: BaseClient,
    bucket_name: str,
    prefix: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    verbose: bool = True,
) -> list[str]:
    """List all object keys under a bucket or prefix.

    This returns every object key found under the given prefix, including any
    folder placeholders (keys ending with '/'), so callers can see folders,
    subfolders, and files.

    Parameters
    ----------
    s3_client : boto3.client
        An initialized boto3 S3 client.
    bucket_name : str
        Name of the S3 bucket.
    prefix : Optional[str], optional
        Prefix (folder path) to list contents under. If None, lists entire bucket.
    logger : Optional[logging.Logger], optional
        Logger instance for logging messages.
    verbose : bool, optional
        If True, print messages to the console about the number of objects found.

    Returns
    -------
    list[str]
        A list of object keys (including folder placeholders if present).

    """
    paginator = s3_client.get_paginator("list_objects_v2")
    list_kwargs = {"Bucket": bucket_name}
    if prefix:
        list_kwargs["Prefix"] = prefix

    keys: list[str] = []
    for page in paginator.paginate(**list_kwargs):
        if "Contents" not in page:
            continue
        keys.extend([obj["Key"] for obj in page["Contents"]])

    if not keys:
        prefix_display = prefix or ""
        if verbose:
            log_l.log_message(logger, f"No objects found in s3://{bucket_name}/{prefix_display}")
    else:
        prefix_display = prefix or ""

        if verbose:
            log_l.log_message(logger, f"Found {len(keys)} objects in s3://{bucket_name}/{prefix_display}")

    return keys


def _upload_file_if_missing(
    *,
    s3_client: BaseClient,
    local_file_path: str,
    bucket_name: str,
    s3_key: str,
    logger: Optional[logging.Logger] = None,
    verbose: bool = True,
    overwrite: bool = False,
    transfer_config: TransferConfig,
) -> None:
    """Upload one local file to S3 unless the remote object already matches it by size.

    Parameters
    ----------
    s3_client : boto3.client
        An initialized boto3 S3 client.
    local_file_path : str
        Full local path to the file.
    bucket_name : str
        Destination S3 bucket name.
    s3_key : str
        Destination key in S3.
    logger : Optional[logging.Logger], optional
        Logger instance for logging messages.
    verbose : bool, optional
        If True, print messages to the console.
    overwrite : bool, optional
        If True, overwrite the file even if it already exists in S3.
    transfer_config : TransferConfig
        TransferConfig object to use for the upload.

    Returns
    -------
    None

    """
    if not overwrite:
        local_size = os.path.getsize(local_file_path)
        remote_size = get_s3_object_size(
            bucket_name=bucket_name,
            s3_key=s3_key,
            s3_client=s3_client,
            logger=logger,
            verbose=verbose,
        )
        if remote_size is not None and remote_size == local_size:
            if verbose:
                log_l.log_message(
                    logger,
                    f"Skipping upload for s3://{bucket_name}/{s3_key}; remote size matches local size ({local_size} bytes)",
                )
            return
        if remote_size is not None and verbose:
            log_l.log_message(
                logger,
                (
                    f"Re-uploading s3://{bucket_name}/{s3_key}; "
                    f"remote size ({remote_size} bytes) differs from local size ({local_size} bytes)"
                ),
                type="warning",
            )

    max_upload_attempts = _safe_int_env("S3_UPLOAD_MAX_ATTEMPTS", default_value=5, min_value=1)
    initial_backoff_s = _safe_float_env("S3_UPLOAD_RETRY_BASE_SECONDS", default_value=3.0, min_value=0.0)
    max_backoff_s = _safe_float_env("S3_UPLOAD_RETRY_MAX_SECONDS", default_value=20.0, min_value=0.0)

    _upload_file_with_retries(
        s3_client=s3_client,
        local_file_path=local_file_path,
        bucket_name=bucket_name,
        s3_key=s3_key,
        logger=logger,
        max_upload_attempts=max_upload_attempts,
        initial_backoff_s=initial_backoff_s,
        max_backoff_s=max_backoff_s,
        transfer_config=transfer_config,
    )

    if verbose:
        log_l.log_message(logger, f"Uploaded file to s3://{bucket_name}/{s3_key}")


def _upload_file_with_retries(
    *,
    s3_client: BaseClient,
    local_file_path: str,
    bucket_name: str,
    s3_key: str,
    logger: Optional[logging.Logger],
    max_upload_attempts: int,
    initial_backoff_s: float,
    max_backoff_s: float,
    transfer_config: TransferConfig,
) -> None:
    """Upload a local file to S3 with retry/backoff for transient failures."""
    for attempt in range(1, max_upload_attempts + 1):
        try:
            s3_client.upload_file(
                local_file_path,
                bucket_name,
                s3_key,
                Config=transfer_config,
            )
            return
        except Exception as exc:
            is_last = attempt == max_upload_attempts
            if is_last or not _is_retryable_upload_exception(exc):
                raise

            backoff_s = min(initial_backoff_s * (2 ** (attempt - 1)), max_backoff_s)
            log_l.log_message(
                logger,
                (
                    f"Retrying upload ({attempt}/{max_upload_attempts}) for s3://{bucket_name}/{s3_key} "
                    f"after transient error: {exc}. Waiting {backoff_s:.1f}s"
                ),
                type="warning",
            )
            if backoff_s > 0:
                time.sleep(backoff_s)


def _iter_local_files(local_path: str) -> Iterator[tuple[str, str]]:
    """Yield local files and their relative paths under a root folder.

    Parameters
    ----------
    local_path : str
        Root local folder.

    Yields
    ------
    tuple[str, str]
        A tuple of (absolute_file_path, relative_path_from_root).

    """
    for root, _, files in os.walk(local_path):
        for file in files:
            local_file_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_file_path, local_path)
            yield local_file_path, relative_path


def upload_to_s3(
    s3_client: BaseClient,
    local_path: str,
    bucket_name: str,
    s3_prefix: str,
    logger: Optional[logging.Logger] = None,
    verbose: bool = True,
    overwrite: bool = False,
) -> None:
    """Upload a local file or folder to an S3 bucket.

    Parameters
    ----------
    s3_client : boto3.client
        An initialized boto3 S3 client.
    local_path : str
        Path to the local file or folder.
    bucket_name : str
        Name of the S3 bucket.
    s3_prefix : str
        S3 key prefix where the data will be uploaded.
    logger : Optional[logging.Logger], optional
        Logger instance for logging messages.
    verbose : bool, optional
        If True, print messages to the console.
    overwrite : bool, optional
        If True, overwrite files in S3 if they already exist. Default is False (skip existing files).

    """
    from tqdm.auto import tqdm

    local_path = os.path.abspath(local_path)
    s3_prefix = s3_prefix.lstrip("/").rstrip("/")

    # Throttle between uploads to avoid S3 rate limits (default 0.0 = no throttling)
    inter_upload_delay_s = _safe_float_env("S3_UPLOAD_INTER_FILE_DELAY_SECONDS", default_value=0.0, min_value=0.0)
    transfer_config = get_s3_transfer_config()

    if os.path.isfile(local_path):
        # Single file upload: if prefix already ends with the filename, avoid duplication
        filename = os.path.basename(local_path)
        if s3_prefix.endswith(f"/{filename}") or s3_prefix == filename:
            s3_prefix = os.path.dirname(s3_prefix)
        normalized_filename = filename.replace(os.sep, "/")
        s3_key = f"{s3_prefix}/{normalized_filename}" if s3_prefix else normalized_filename
        _upload_file_if_missing(
            s3_client=s3_client,
            local_file_path=local_path,
            bucket_name=bucket_name,
            s3_key=s3_key,
            logger=logger,
            verbose=verbose,
            overwrite=overwrite,
            transfer_config=transfer_config,
        )
    else:
        # Directory upload
        total_files = sum(len(files) for _root, _dirs, files in os.walk(local_path))

        with tqdm(total=total_files, desc="Uploading files", unit="file", disable=verbose) as pbar:
            for local_file_path, relative_path in _iter_local_files(local_path):
                # S3 key with prefix
                normalized_relative_path = relative_path.replace(os.sep, "/")
                s3_key = f"{s3_prefix}/{normalized_relative_path}" if s3_prefix else normalized_relative_path
                _upload_file_if_missing(
                    s3_client=s3_client,
                    local_file_path=local_file_path,
                    bucket_name=bucket_name,
                    s3_key=s3_key,
                    logger=logger,
                    verbose=verbose,
                    overwrite=overwrite,
                    transfer_config=transfer_config,
                )
                pbar.update(1)

                # Throttle between uploads to avoid rate limits
                if inter_upload_delay_s > 0:
                    time.sleep(inter_upload_delay_s)


def download_s3_folder(
    s3_client: BaseClient, bucket_name: str, s3_prefix: str, local_folder: str, logger: Optional[logging.Logger] = None
) -> None:
    """Download all objects under the given S3 prefix to a local folder.

    Parameters
    ----------
    s3_client : boto3.client
        An initialized boto3 S3 client.
    bucket_name : str
        Name of the S3 bucket.
    s3_prefix : str
        Prefix (folder in S3) to download, e.g., 'stac/'.
    local_folder : str
        Local destination folder.

    """
    paginator = s3_client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix)

    for page in pages:
        if "Contents" not in page:
            continue

        for obj in page["Contents"]:
            s3_key = obj["Key"]
            # Compute relative path
            relative_path = os.path.relpath(s3_key, s3_prefix)
            local_path = os.path.join(local_folder, relative_path)

            if s3_key.endswith("/"):
                # S3 folders: skip
                continue

            # Ensure local directories exist
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            print(f"Downloading s3://{bucket_name}/{s3_key} -> {local_path}")
            s3_client.download_file(bucket_name, s3_key, local_path)

    log_l.log_message(logger, f"Download complete from s3://{bucket_name}/{s3_prefix}")


def download_s3_file(
    s3_client: BaseClient,
    bucket_name: str,
    s3_key: str,
    local_file_path: str,
    logger: Optional[logging.Logger] = None,
    key: str | None = None,
    secret: str | None = None,
    region: str | None = None,
) -> None:
    """Download a single object from Amazon S3 to a local filesystem path.

    Creates parent directories for local_file_path if they do not exist and then
    downloads the object identified by s3://{bucket_name}/{s3_key} to the given
    local destination. A message is logged on success.

    Parameters
    ----------
    s3_client : botocore.client.BaseClient
        An S3 client instance. (Note: this function currently overrides the provided
        client by calling get_boto3_client() internally.)
    bucket_name : str
        Name of the S3 bucket containing the object.
    s3_key : str
        Object key (path within the bucket) to download.
    local_file_path : str
        Absolute or relative path where the file will be saved locally.
    logger : logging.Logger, optional
        Logger instance for emitting an informational message after download.

    Returns
    -------
    None

    Raises
    ------
    botocore.exceptions.ClientError
        If the S3 download fails (e.g., object not found, access denied).
    OSError
        If local directories cannot be created or the file cannot be written.

    Notes
    -----
    - This function ensures that os.path.dirname(local_file_path) exists
    by calling os.makedirs(..., exist_ok=True).

    """
    # Always use a fresh client to avoid stale credentials/session state.
    # Keep s3_client argument for backward compatibility with existing callers.
    if not s3_client:
        s3_client = get_boto3_client(key=key, secret=secret, region=region)
    parent_dir = os.path.dirname(local_file_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    s3_client.download_file(bucket_name, s3_key, local_file_path)
    log_l.log_message(logger, f"Downloaded s3://{bucket_name}/{s3_key} → {local_file_path}")


def upload_folder_to_s3(s3_client, bucket_name, local_folder, s3_prefix=""):
    """Upload all files in a local folder to S3 under the specified s3_prefix."""
    all_results = []
    for root, _, files in os.walk(local_folder):
        for file in files:
            local_path = os.path.join(root, file).replace("\\", "/")
            item_path = os.path.basename(os.path.normpath(root))
            s3_key = f"{s3_prefix}/{item_path}/{file}".replace("\\", "/")
            try:
                # Upload file to s3 bucket
                s3_client.upload_file(local_path, bucket_name, s3_key)
                print(f"Success: Uploaded {local_path} to s3://{bucket_name}/{s3_key}")
                all_results.append(True)

            except Exception as e:
                print(f"Error: Uploading {local_path} failed: {e}")
                all_results.append(False)
    return all_results


def upload_all_folders(s3_client, bucket_name, folders_to_process) -> bool:
    """Upload data to S3 and logs the process."""
    try:
        all_results = []
        for local_folder, s3_prefix in folders_to_process.items():
            print(f"Uploading data to {bucket_name}/{s3_prefix}...")
            upload_result = upload_folder_to_s3(s3_client, bucket_name, local_folder, s3_prefix)
            all_results.extend(upload_result)

        if all(all_results):
            print("Success: uploaded all products")
            return True
        else:
            errors = [r for r in all_results if not r]
            print(f"Error: Failed to upload {len(errors)} out of {len(all_results)} total files!")
            return False
    except Exception as e:
        print(f"Error: Uploading data to s3 failed: {e}")
    return False
