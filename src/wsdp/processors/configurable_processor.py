"""ConfigurableProcessor: run a user-defined algorithm pipeline over CSIData."""

from typing import Any, Dict, List, Tuple

import numpy as np

from wsdp.algorithms import AlgorithmStep, build_steps_from_config, normalize_amplitude
from wsdp.dataset_policy import pipeline_uses_zscore, uses_phase_amplitude
from wsdp.interfaces import Processor
from wsdp.processors.modular_processor import ModularProcessor


class ConfigurableProcessor(Processor):
    """Processor that applies a user-defined algorithm pipeline to each CSI sample.

    Widar and Gait use amplitude-phase model inputs automatically. When their
    pipeline contains z-score normalization, normalization is emitted as
    real-valued ``[signed_normalized_amplitude, phase]`` channels.

    Args:
        pipeline_steps: dict describing the algorithm pipeline, e.g.
            {'denoise': {'method': 'wavelet'},
             'calibrate': {'method': 'stc'},
             'normalize': {'method': 'z-score'}}
    """

    def __init__(self, pipeline_steps: Dict[str, Dict[str, Any]], n_workers: int = 4):
        self.pipeline_steps = pipeline_steps
        self.n_workers = n_workers

    def process(
        self,
        data_list: List[Any],
        **kwargs,
    ) -> Tuple[List[np.ndarray], List[Any], List[Any]]:
        """Process CSIData objects and return processed arrays, labels, and groups."""
        dataset = kwargs.get("dataset", "")
        steps, phase_zscore = self._resolve_steps(dataset)

        processor = ModularProcessor(steps, n_workers=self.n_workers)
        all_data, all_labels, all_groups = processor.process(
            data_list, dataset=dataset
        )

        if phase_zscore:
            all_data = [
                normalize_amplitude(arr, method="z-score", return_phase_channels=True)
                for arr in all_data
            ]

        return all_data, all_labels, all_groups

    def _resolve_steps(
        self, dataset: str
    ) -> Tuple[List[AlgorithmStep], bool]:
        """Build the effective step list and whether to apply phase-zscore."""
        normalize_step = self.pipeline_steps.get("normalize", {})

        phase_zscore = (
            uses_phase_amplitude(dataset)
            and pipeline_uses_zscore(self.pipeline_steps)
        )
        xrf55_skip_normalize = (
            dataset == "xrf55"
            and normalize_step.get("method") in {"z-score", "min-max"}
        )
        skip_normalize = phase_zscore or xrf55_skip_normalize

        effective = dict(self.pipeline_steps)
        if skip_normalize:
            effective.pop("normalize", None)

        steps = build_steps_from_config(effective)
        return steps, phase_zscore


def _process_single_csi_configurable(csi_data, dataset, pipeline_steps):
    """Backward-compatible worker: process one CSIData with a configurable pipeline.

    This function is kept for tests and external callers that import it
    directly. New code should use ``ConfigurableProcessor`` or
    ``ModularProcessor`` instead.
    """
    processor = ConfigurableProcessor(pipeline_steps, n_workers=1)
    data, labels, groups = processor.process([csi_data], dataset=dataset)
    if not data:
        return None, None, None
    return data[0], labels[0], groups[0]
