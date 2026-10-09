"""Small formatting helpers without scientific or notebook dependencies."""

# Keep this helper focused on a single transformation so the reporting pipeline stays easy to follow.



def human_bytes(n: int) -> str:
    """Print bytes in human readable format."""
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} EB"
