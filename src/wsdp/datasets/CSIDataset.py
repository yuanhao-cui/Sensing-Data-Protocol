import torch
import numpy as np

from torch.utils.data import Dataset

class CSIDataset(Dataset):
    def __init__(self, data_list, labels, use_phase=False, preserve_real_sign=False):
        data_array = np.asarray(data_list)
        if use_phase and np.iscomplexobj(data_array):
            # Phase+Amplitude dual-channel: stack |H| and angle(H) along
            # last axis.  PA-CSI (Sensors 2025) shows this outperforms
            # amplitude-only for most sensing tasks.
            amplitude = np.abs(data_array)
            phase = np.angle(data_array)
            data_list = np.concatenate([amplitude, phase], axis=-1)
        elif preserve_real_sign:
            if np.iscomplexobj(data_array):
                imag_part = np.imag(data_array)
                if imag_part.size == 0 or np.max(np.abs(imag_part)) < 1e-10:
                    data_list = np.real(data_array)
                else:
                    data_list = np.abs(data_array)
            else:
                data_list = data_array
        else:
            data_list = np.abs(data_array)
        self.data_list = torch.from_numpy(data_list).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data_list[idx], self.labels[idx]
