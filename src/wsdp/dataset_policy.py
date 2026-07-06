"""Dataset-specific preprocessing policy helpers."""

import numpy as np


AMPLITUDE_PRIMARY_DATASETS = {"xrf55"}
PHASE_AMPLITUDE_DATASETS = {"widar", "gait"}


def is_amplitude_primary_dataset(dataset: str) -> bool:
    """Return True for datasets whose downstream features use amplitude values."""
    return dataset in AMPLITUDE_PRIMARY_DATASETS


def uses_phase_amplitude(dataset: str) -> bool:
    """Return True when model inputs should contain both amplitude and phase."""
    return str(dataset).strip().lower() in PHASE_AMPLITUDE_DATASETS


def pipeline_uses_zscore(pipeline_steps) -> bool:
    """Return True when a pipeline config contains z-score normalization."""
    return (
        isinstance(pipeline_steps, dict)
        and pipeline_steps.get("normalize", {}).get("method") == "z-score"
    )


def real_if_negligible_imaginary(csi, dataset: str, atol: float = 1e-10):
    """Drop a near-zero imaginary part for amplitude-primary datasets only."""
    if not is_amplitude_primary_dataset(dataset) or not np.iscomplexobj(csi):
        return csi

    imag_part = np.imag(csi)
    if imag_part.size == 0 or np.max(np.abs(imag_part)) < atol:
        return np.real(csi)
    return csi
