"""Filename-based metadata extraction for CSI samples."""

import os
import re
import logging
from pathlib import Path
from typing import Any, Tuple

logger = logging.getLogger(__name__)


def parse_file_info_from_filename(f_name: str, dataset: str) -> Tuple[Any, ...]:
    """Parse label/group metadata from a CSI filename."""
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


def select_label_and_group(res, dataset: str) -> Tuple[Any, Any]:
    """Extract label and group from parsed filename metadata.

    Group variable determines how GroupShuffleSplit partitions data.
    Following standard evaluation protocols:
    - Widar: group=position_id*1000+orientation_id*100+receiver_id for condition split
    - Gait: label=user_id and group=track_id*100+receiver_id for held-out conditions
    - XRF55: label=action_id and group=repetition_id for official-style trial split
    - ElderAL/ZTE: group=position_id
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
        group = int(res[2])   # repetition_id
    elif dataset in ('elderAL', 'zte'):
        label = int(res[2])   # action_id
        group = int(res[1])   # position_id
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return label, group


def _process_file_path(f_name: str) -> Tuple[Tuple[str, ...], str]:
    """Cross-platform path splitting helper."""
    full_path = Path(f_name)
    path_parts = full_path.parts
    base = full_path.stem
    return path_parts, base
