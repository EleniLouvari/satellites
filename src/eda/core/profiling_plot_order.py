"""Feature ordering helpers for EDA plotting."""

from __future__ import annotations

from .profiling_feature_selection import (
    _build_plot_feature_order as _build_plot_feature_order_impl,
)
from .profiling_feature_selection import (
    _proposal_feature_order as _proposal_feature_order_impl,
)
from .profiling_feature_selection import (
    _sorted_remaining_plot_candidates as _sorted_remaining_plot_candidates_impl,
)

# Re-export the same function objects so plot ordering stays aligned with feature selection.
_build_plot_feature_order = _build_plot_feature_order_impl
_proposal_feature_order = _proposal_feature_order_impl
_sorted_remaining_plot_candidates = _sorted_remaining_plot_candidates_impl
