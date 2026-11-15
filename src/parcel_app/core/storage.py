"""Storage-boundary, checksum and log-redaction helpers shared by every pipeline."""

import hashlib
import re
import shutil
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

from parcel_app.settings import get_settings


def contained(root, value):
    """Resolve and reject files outside a trusted storage boundary."""
    root = Path(root).resolve()
    path = Path(value).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("Error: Artifact is outside its run directory.")
    return path


def checksum(path):
    """Hash a file with bounded memory."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def storage_available(reserve: int = 0) -> bool:
    """Check configured used-storage and physical free-space ceilings."""
    root = get_settings().storage_root
    root.mkdir(parents=True, exist_ok=True)
    used = sum(p.stat().st_size for p in root.rglob("*") if p.is_file() and not p.is_symlink())
    return used + reserve <= get_settings().storage_limit_bytes and shutil.disk_usage(root).free > reserve + 64 * 1024**2


_CREDENTIAL_NAME = (
    # One optional prefix run: a repeated group here backtracks exponentially on long "____" log separators.
    r"(?:[\w-]*[_-])?(?:username|password|passwd|pwd|token|authorization|secret|api[_-]?key|credential[_-]?key|signature)"
)


def _redact_credential_url(match):
    """Hide URLs containing user information or authentication query/fragment keys."""
    value = match.group()
    try:
        url = urlsplit(value)
        fields = parse_qsl(url.query) + parse_qsl(url.fragment)
        sensitive = url.username is not None or any(re.fullmatch(_CREDENTIAL_NAME, key, re.IGNORECASE) for key, _ in fields)
    except ValueError:
        # Malformed authority syntax cannot be safely classified as credential-free.
        sensitive = True
    return "[redacted URL]" if sensitive else value


def sanitize_text(value, secrets=(), *, private_paths=()):
    """Mask credential fields and URLs, never global occurrences of credential values.

    Known values help recognize unquoted credentials containing spaces, but only
    after a credential field label. Private filesystem paths are supplied separately.
    """
    value = re.sub(r"\b[a-z][a-z0-9+.-]*://[^\s\"'<>]+", _redact_credential_url, value, flags=re.IGNORECASE)
    known = "|".join(re.escape(secret) for secret in sorted(set(secrets), key=len, reverse=True) if secret)
    # Quoted values may contain spaces or escaped quotes; auth schemes include the token.
    credential_value = r"\[redacted\]|\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|(?:Bearer|Basic)\s+[^\s,;\"'}\]]+"
    if known:
        credential_value += rf"|(?:{known})(?=$|[\s,;}}\]])"
    credential_value += r"|[^\s,;}\]]+"
    value = re.sub(
        rf"(?<![\w-])([\"']?{_CREDENTIAL_NAME}[\"']?\s*[:=]\s*)(?:{credential_value})",
        r"\1[redacted]",
        value,
        flags=re.IGNORECASE,
    )
    for path in sorted(set(private_paths), key=len, reverse=True):
        if path:
            value = value.replace(path, "[redacted]")
    return value
