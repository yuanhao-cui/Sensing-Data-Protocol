import numpy as np
import pandas as pd

from wsdp.structure import CSIData
from wsdp.structure import BaseFrame


class ZTEReader:

    def __init__(self):
        super().__init__()

    def read_file(self, file_path: str) -> CSIData:
        ret = CSIData(file_path)

        df = pd.read_csv(file_path)

        df = df[df['rx_chain_num'].str.endswith('tx0')].copy()

        if df.empty:
            print("warning: cannot found tx0 in file.")
            return ret

        df['rx_idx'] = df['rx_chain_num'].apply(lambda x: int(x.split('-')[0].replace('rx', '')))

        i_cols = [f'csi_i_{k}' for k in range(512)]
        q_cols = [f'csi_q_{k}' for k in range(512)]

        data_i = df[i_cols].values  # shape: (N_rows, 512)
        data_q = df[q_cols].values  # shape: (N_rows, 512)

        complex_data = data_i + 1j * data_q

        df['complex_vector'] = list(complex_data)

        grouped = df.groupby('timestamp')

        frames_list = []

        for ts, group in grouped:
            frame_matrix = np.zeros((512, 3), dtype=np.complex64)

            for _, row in group.iterrows():
                rx_id = row['rx_idx']
                c_vector = row['complex_vector']

                if 0 <= rx_id < 3:
                    frame_matrix[:, rx_id] = c_vector

            frame = BaseFrame(timestamp=ts, csi_array=frame_matrix)
            frames_list.append(frame)

        ret.frames = frames_list

        return ret