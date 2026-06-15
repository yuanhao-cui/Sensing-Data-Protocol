"""Backward-compatible imports for pipeline diagnostics.

Prefer importing these helpers from ``wsdp.diagnostics`` for standalone
pre-pipeline screening.
"""
from wsdp.diagnostics import (
    compare_pipeline_candidates,
    compute_pipeline_metrics,
    compute_phase_diagnostics,
    plot_pipeline_diagnostics,
    run_pipeline_diagnostics,
)

__all__ = [
    "compare_pipeline_candidates",
    "compute_pipeline_metrics",
    "compute_phase_diagnostics",
    "plot_pipeline_diagnostics",
    "run_pipeline_diagnostics",
]
