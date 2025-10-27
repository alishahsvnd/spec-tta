"""
Tools for experimental protocol evaluation and reporting.

This module provides utilities for:
- Aggregating results across multiple runs
- Computing best-row counts (per dataset×horizon)
- Generating PETSA-style comparison tables
- Parameter efficiency analysis (MSE vs. params plots)
"""

from .protocol_eval import (
    best_row_counts,
    parameter_summary,
    pretty_table,
    load_results,
    save_results,
)

__all__ = [
    'best_row_counts',
    'parameter_summary',
    'pretty_table',
    'load_results',
    'save_results',
]
