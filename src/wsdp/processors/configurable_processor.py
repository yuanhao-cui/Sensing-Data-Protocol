"""ConfigurableProcessor: run a user-defined algorithm pipeline over a list of CSIData."""

from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np

from wsdp.algorithms import execute_pipeline
from wsdp.algorithms.amplitude import normalize_amplitude
from wsdp.dataset_policy import pipeline_uses_zscore, uses_phase_amplitude


class ConfigurableProcessor:
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

    def __init__(self, pipeline_steps):
        self.pipeline_steps = pipeline_steps

    def process(self, data_list, **kwargs):
        """Process CSIData objects and return processed arrays, labels, and groups."""
        dataset = kwargs.get('dataset', '')
        all_data, all_labels, all_groups = [], [], []

        worker_func = partial(
            _process_single_csi_configurable,
            dataset=dataset,
            pipeline_steps=self.pipeline_steps,
        )

        with ProcessPoolExecutor(max_workers=4) as executor:
            results = executor.map(worker_func, data_list)
            for csi, label, group in results:
                if csi is not None:
                    all_data.append(csi)
                    all_labels.append(label)
                    all_groups.append(group)
        return all_data, all_labels, all_groups


def _process_single_csi_configurable(csi_data, dataset, pipeline_steps):
    """Worker: parse one CSIData, build (T, F, A) tensor, run configured pipeline."""
    from wsdp.processors.base_processor import _parse_file_info_from_filename, _selector

    res = _parse_file_info_from_filename(csi_data.file_name, dataset)
    label, group = _selector(res, dataset)

    sorted_frames = sorted(csi_data.frames, key=lambda f: f.timestamp)
    frame_tensors = [f.csi_array for f in sorted_frames]

    if not frame_tensors:
        return None, None, None

    whole_csi = np.stack(frame_tensors, axis=0)
    if whole_csi.ndim == 2:
        whole_csi = np.expand_dims(whole_csi, -1)
    if whole_csi.shape[0] < 2:
        return None, None, None

    phase_zscore = (
        uses_phase_amplitude(dataset)
        and pipeline_uses_zscore(pipeline_steps)
    )

    effective_pipeline_steps = pipeline_steps
    normalize_step = pipeline_steps.get("normalize", {})
    if (
        phase_zscore
        or (
            dataset == "xrf55"
            and normalize_step.get("method") in {"z-score", "min-max"}
        )
    ):
        effective_pipeline_steps = {
            key: value for key, value in pipeline_steps.items()
            if key != "normalize"
        }

    cleaned_csi = execute_pipeline(whole_csi, effective_pipeline_steps, dataset=dataset)

    if phase_zscore:
        cleaned_csi = normalize_amplitude(
            cleaned_csi,
            method="z-score",
            return_phase_channels=True,
        )

    return cleaned_csi, label, group
