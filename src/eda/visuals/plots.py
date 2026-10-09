"""Public EDA plotting API composed from focused plotting modules."""

from __future__ import annotations

import importlib
import sys
import types
from typing import Any

_COMPAT_MODULES = (
    "plots_common",
    "plots_renderers",
    "plots_orchestration",
)

# Keep this helper focused on a single transformation so the reporting pipeline stays easy to follow.



class _CompatibilityModule(types.ModuleType):
    """Mirror monkeypatch updates onto the split plotting modules."""

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        for module_name in _COMPAT_MODULES:
            module = importlib.import_module(f"{__package__}.{module_name}")
            if hasattr(module, name):
                setattr(module, name, value)

    def __delattr__(self, name: str) -> None:
        super().__delattr__(name)
        for module_name in _COMPAT_MODULES:
            module = importlib.import_module(f"{__package__}.{module_name}")
            if hasattr(module, name):
                delattr(module, name)


def __getattr__(name: str) -> Any:
    """Expose internal plotting helpers from split modules for compatibility."""
    for module_name in _COMPAT_MODULES:
        module = importlib.import_module(f"{__package__}.{module_name}")
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return visible attributes from split plotting modules."""
    visible = set(globals())
    for module_name in _COMPAT_MODULES:
        module = importlib.import_module(f"{__package__}.{module_name}")
        visible.update(dir(module))
    return sorted(visible)


sys.modules[__name__].__class__ = _CompatibilityModule
