"""Modular processor that runs configurable algorithm steps per CSI sample."""

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, List, Tuple

import numpy as np

from wsdp.algorithms import AlgorithmStep, execute_algorithm_steps
from wsdp.interfaces import Processor
from wsdp.processors.metadata import parse_file_info_from_filename, select_label_and_group


class ModularProcessor(Processor):
    """Processor that applies a configurable algorithm pipeline to each sample.

    Args:
        steps: Ordered sequence of ``AlgorithmStep`` objects describing the
            per-sample processing chain.
        n_workers: Number of worker processes used to process samples in
            parallel (default 16).
    """

    def __init__(
        self,
        steps: List[AlgorithmStep],
        *,
        n_workers: int = 16,
    ) -> None:
        self.steps = [AlgorithmStep.from_config(s) for s in steps]
        self.n_workers = n_workers

    def process(
        self,
        data_list: List[Any],
        **kwargs,
    ) -> Tuple[List[np.ndarray], List[Any], List[Any]]:
        """Process CSIData objects and return arrays, labels, and groups."""
        dataset = kwargs.get("dataset", "")

        worker_func = partial(
            _process_single_modular,
            dataset=dataset,
            steps=self.steps,
        )

        all_data, all_labels, all_groups = [], [], []
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            results = executor.map(worker_func, data_list)
            for csi, label, group in results:
                if csi is not None:
                    all_data.append(csi)
                    all_labels.append(label)
                    all_groups.append(group)
        return all_data, all_labels, all_groups


def _process_single_modular(
    csi_data: Any,
    dataset: str,
    steps: List[AlgorithmStep],
) -> Tuple[Any, Any, Any]:
    """Worker: parse metadata, stack frames, and run the configured steps."""
    res = parse_file_info_from_filename(csi_data.file_name, dataset)
    label, group = select_label_and_group(res, dataset)

    sorted_frames = sorted(csi_data.frames, key=lambda f: f.timestamp)
    frame_tensors = [f.csi_array for f in sorted_frames]

    if not frame_tensors:
        return None, None, None

    whole_csi = np.stack(frame_tensors, axis=0)
    if whole_csi.ndim == 2:
        whole_csi = np.expand_dims(whole_csi, -1)
    if whole_csi.shape[0] < 2:
        return None, None, None

    state = execute_algorithm_steps(
        whole_csi,
        steps,
        dataset=dataset,
    )
    cleaned_csi = state.get("csi", whole_csi)
    return cleaned_csi, label, group
