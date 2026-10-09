"""Shared constants for EDA profiling modules."""

from __future__ import annotations

# These full-dataset diagnostics must be excluded from predictor-only analyses to avoid leakage.
_EDA_DIAGNOSTIC_COLUMNS = {"is_outlier_97_5pct", "mahalanobis_distance", "mahalanobis_pvalue"}
