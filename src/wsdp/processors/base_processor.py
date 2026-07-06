import os
import re
import logging
import numpy as np

from pathlib import Path
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from typing import List

from wsdp.algorithms import phase_calibration, wavelet_denoise_csi
from wsdp.structure import CSIData
from wsdp.dataset_policy import real_if_negligible_imaginary

logger = logging.getLogger(__name__)


class BaseProcessor:
    def process(self, data_list: List[CSIData], **kwargs):
        dataset = kwargs.get('dataset', '')
        all_data = []
        all_labels = []
        all_groups = []
        worker_func = partial(_process_single_csi, dataset=dataset)
        with ProcessPoolExecutor(max_workers=16) as executor:
            results = executor.map(worker_func, data_list)
            for csi, label, group in results:
                if csi is not None:
                    all_data.append(csi)
                    all_labels.append(label)
                    all_groups.append(group)
        return all_data, all_labels, all_groups


# function for parallel processing
def _process_single_csi(csi_data, dataset):
    res = _parse_file_info_from_filename(csi_data.file_name, dataset)
    label, group = _selector(res, dataset)
    sorted_frames = sorted(csi_data.frames, key=lambda frame: frame.timestamp)
    frame_tensors = []
    for frame in sorted_frames:
        data = frame.csi_array
        frame_tensors.append(data)
    if frame_tensors:
        whole_csi = np.stack(frame_tensors, axis=0)
        # Ensure 3D: (T, F, A) — guard against single-dimension data
        if whole_csi.ndim == 2:
            # (T, F) — single antenna, reshape to (T, F, 1)
            whole_csi = np.expand_dims(whole_csi, -1)
        elif whole_csi.ndim == 1:
            # (T,) — too degenerate, skip
            logger.warning(f"data too degenerate (1D): {csi_data.file_name}")
            return None, None, None
        # discard data with too short time period (1 timestamp)
        if whole_csi.shape[0] < 2:
            logger.warning(f"only one timestamp: {csi_data.file_name}")
            return None, None, None
        whole_csi = phase_calibration(whole_csi, dataset=dataset)
        cleaned_csi = wavelet_denoise_csi(whole_csi)
        cleaned_csi = real_if_negligible_imaginary(cleaned_csi, dataset)
        return cleaned_csi, label, group
    return None, None, None


def _parse_file_info_from_filename(f_name, dataset):
    base = os.path.splitext(os.path.basename(f_name))[0]

    if dataset == 'widar':
        m = re.match(r'user(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-r(\d+)', base)
        if m:
            user_id = int(m.group(1))
            gesture_type = int(m.group(2))
            torso_position = int(m.group(3))
            orientation = int(m.group(4))
            data_serial = int(m.group(5))
            receiver_number = int(m.group(6))
            return user_id, gesture_type, torso_position, orientation, data_serial, receiver_number
        else:
            logger.warning(f"Skipping file {f_name}: Invalid format for Gesture Recognition.")

    elif dataset == 'gait':
        # Parse for Gait User ID (pattern "user{N}-{track}-{repetition}-r{receiver}.dat")
        m = re.search(r'user(\d+)-(\d+)-(\d+)-r(\d+)', base, re.IGNORECASE)
        if m:
            user_id = int(m.group(1))
            track_id = int(m.group(2))
            repetition_id = int(m.group(3))
            receiver_id = int(m.group(4))

            return user_id, track_id, repetition_id, receiver_id, None, None
        else:
            logger.warning(f"Skipping file {f_name}: Invalid format for Activity Recognition.")

    elif dataset == 'xrf55':
        # XRF55 filenames are user_action_trial, e.g. 03_20_08.
        # Keep trial/repetition so the core split can follow the official
        # XRF55-style repetition protocol.
        m = re.search(r'(\d+)_(\d+)_(\d+)', base)
        if m:
            user_id = int(m.group(1))
            action_id = int(m.group(2))
            repetition_id = int(m.group(3))
            return user_id, action_id, repetition_id, None, None, None
        else:
            logger.warning(f"Skipping file {f_name}: Invalid format for xrf55.")

    elif dataset == 'elderAL':
        m = re.search(r"user(\d+)_position(\d+)_activity(\d+)", f_name)
        if m:
            user_id = int(m.group(1))
            position_id = int(m.group(2))
            action_id = int(m.group(3))
            return user_id, position_id, action_id, None, None, None
        else:
            logger.warning(f"Skipping file {f_name}: Invalid format for ElderAL Dataset.")

    elif dataset == 'zte':
        base = _process_file_path(f_name)[0][1]
        m = re.search(r"user(\d+)_pos(\d+)_action(\d+)", base)
        if m:
            user_id = int(m.group(1))
            position_id = int(m.group(2))
            action_id = m.group(3)
            return user_id, position_id, action_id, None, None, None
        else:
            logger.warning(f"Skipping file {f_name}: Invalid format for ZTE Dataset.")

    else:
        logger.error(f"Unknown task type: {dataset}")


def _selector(res, dataset):
    """
    Extract label and group from parsed filename metadata.

    Group variable determines how GroupShuffleSplit partitions data.
    Following standard evaluation protocols:
    - Widar: group=position_id*1000+orientation_id*100+receiver_id for condition split
    - Gait: label=user_id and group=track_id*100+receiver_id for held-out conditions
    - XRF55: label=action_id and group=repetition_id for official-style trial split
    - ElderAL/ZTE: group=position_id

    Args:
        res: Tuple of parsed metadata from _parse_file_info_from_filename.
        dataset: Dataset name string.

    Returns:
        Tuple of (label, group).

    Raises:
        ValueError: If dataset is unknown.
    """
    label = None
    group = None

    if dataset == 'widar':
        label = int(res[1])   # gesture_type
        group = int(res[2]) * 1000 + int(res[3]) * 100 + int(res[5])
    elif dataset == 'gait':
        label = int(res[0])   # user_id
        group = int(res[1]) * 100 + int(res[3])   # track_id + receiver_id
    elif dataset == 'xrf55':
        label = int(res[1])   # action_id
        # XRF55 official-style evaluation splits by repetition/trial:
        # train 01-12, val 13-16, test 17-20.
        group = int(res[2])   # repetition_id
    elif dataset in ('elderAL', 'zte'):
        label = int(res[2])   # action_id
        group = int(res[1])   # position_id
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return label, group


def _process_file_path(f_name):
    """
    process_file_path for cross os
    """
    full_path = Path(f_name)
    path_parts = full_path.parts
    base = full_path.stem
    return path_parts, base
