"""Reporting exports for step-level and index HTML generation helpers.

This package exposes small, explicit helpers used by the ML pipeline to
produce self-contained HTML reports for each step. Consumers should import
only the functions listed in ``__all__``.
"""

from .html import write_html_report
from .reports import (
    write_check_report,
    write_evaluate_report,
    write_index_report,
    write_predict_report,
    write_prepare_report,
    write_train_report,
)

# Public reporting API - keep explicit so tools and IDEs can discover exports.
__all__ = [
    "write_html_report",
    "write_check_report",
    "write_prepare_report",
    "write_train_report",
    "write_evaluate_report",
    "write_predict_report",
    "write_index_report",
]
