import numpy as np
import torch
from torch.utils.data import Dataset

from wsdp.dataset_policy import (
    is_amplitude_primary_dataset,
    pipeline_uses_zscore,
    uses_phase_amplitude,
)


class CSIDataset(Dataset):
    """Convert processed CSI arrays to dataset-specific model inputs."""

    def __init__(self, data_list, labels, dataset_name="", pipeline_steps=None):
        data_array = np.asarray(data_list)

        phase_dataset = uses_phase_amplitude(dataset_name)
        phase_zscore = (
            phase_dataset
            and pipeline_uses_zscore(pipeline_steps)
        )

        if phase_zscore:
            if np.iscomplexobj(data_array):
                raise RuntimeError(
                    "Widar/Gait z-score input must be converted to real "
                    "amplitude-phase channels by ConfigurableProcessor"
                )
            data_list = data_array
        elif phase_dataset and np.iscomplexobj(data_array):
            # First A channels are amplitude; the next A channels are phase.
            amplitude = np.abs(data_array)
            phase = np.angle(data_array)
            data_list = np.concatenate([amplitude, phase], axis=-1)
        elif (
            is_amplitude_primary_dataset(dataset_name)
            and not np.iscomplexobj(data_array)
        ):
            # XRF55 train-split normalization produces signed real values.
            data_list = data_array
        else:
            # Amplitude-only datasets keep the historical magnitude path.
            data_list = np.abs(data_array)

        self.data_list = torch.from_numpy(data_list).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data_list[idx], self.labels[idx]
