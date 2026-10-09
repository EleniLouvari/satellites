"""Runtime assertion helpers shared across production modules."""

# Keep this helper focused on a single transformation so the reporting pipeline stays easy to follow.



def expect_true(
    cond,
    msg: str = "Expected condition to be true",
) -> None:
    """Raise AssertionError when ``cond`` is falsy."""
    if not cond:
        raise AssertionError(msg)
