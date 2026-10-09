"""Public EDA profiling API composed from focused profiling modules."""

from __future__ import annotations

from typing import Any

from .profiling_artifacts import build_eda_artifacts
from .profiling_feature_selection import _build_feature_selection_criteria, _build_feature_target_associations
from .profiling_plot_order import _build_plot_feature_order
from .profiling_quality import _build_data_quality_flags
from .profiling_statistics import _build_missingness_target_tests, _build_numeric_target_correlations, _build_numeric_target_tests

_COMPAT_MODULES = (
    "profiling_artifacts",
    "profiling_helpers",
    "profiling_quality",
    "profiling_summaries",
    "profiling_statistics",
    "profiling_feature_selection",
    "profiling_multivariate",
    "profiling_constants",
)

__all__ = [
    "_build_data_quality_flags",
    "_build_feature_selection_criteria",
    "_build_feature_target_associations",
    "_build_missingness_target_tests",
    "_build_numeric_target_correlations",
    "_build_numeric_target_tests",
    "_build_plot_feature_order",
    "build_eda_artifacts",
]

# Keep this helper focused on a single transformation so the reporting pipeline stays easy to follow.



def __getattr__(name: str) -> Any:
    """Expose profiling internals from split modules for compatibility."""
    import importlib

    for module_name in _COMPAT_MODULES:
        module = importlib.import_module(f"{__package__}.{module_name}")
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return visible attributes from the split profiling modules."""
    import importlib

    visible = set(globals())
    for module_name in _COMPAT_MODULES:
        module = importlib.import_module(f"{__package__}.{module_name}")
        visible.update(dir(module))
    return sorted(visible)
