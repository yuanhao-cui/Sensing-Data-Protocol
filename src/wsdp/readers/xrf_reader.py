import numpy as np
import os

from typing import List, Dict, Any
from wsdp.readers.base import BaseReader
from wsdp.structure import CSIData
from wsdp.structure import BaseFrame


class XrfReader(BaseReader):
    """
    XRF55 dataset reader.

    Supports two formats:
    - .npy: Original numpy format (N, 3, 30, 3, 1000) = (samples, rx, sub, ant, time)
    - .dat: Raw binary format from Kaggle download
            40 int16 header + 199 packets × 270 complex values (I/Q pairs)
            CSI shape per packet: (1 Tx, 3 Rx, 3 Ant, 30 Subcarrier)
            Stored as: 199 × 270 × 2 int16 (real, imag)
    """

    XRF55_DAT_HEADER = 40  # int16 values
    XRF55_DAT_PACKETS = 199  # packets per file
    XRF55_DAT_COMPLEX = 270  # complex values per packet = 1*3*3*30

    def __init__(self):
        super().__init__()

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "reader": "XrfReader",
            "format": "npy or dat (xrf55 binary)",
            "description": "XRF55 dataset (numpy or raw .dat binary)",
            "frame_type": "BaseFrame",
            "receivers": 3,
            "subcarriers": 30,
            "time_steps": self.XRF55_DAT_PACKETS,
            "complex": True,
        }

    def sniff(self, file_path) -> bool:
        """Check if file is a supported XRF55 format."""
        try:
            s = str(file_path)
            _, ext = os.path.splitext(s)
            ext = ext.lower()
            if ext == '.npy':
                # Check .npy magic number: \x93NUMPY
                with open(file_path, 'rb') as f:
                    magic = f.read(6)
                return magic[:6] == b'\x93NUMPY'
            elif ext == '.dat':
                # Check file size: 107500 int16 = 215000 bytes
                size = os.path.getsize(file_path)
                # Expected: 107500 int16 (40 header + 199*270*2) = 215000 bytes
                # Also allow slightly different packet counts
                if size % 2 != 0:
                    return False
                int16_count = size // 2
                # Must have header + at least some packets
                if int16_count < self.XRF55_DAT_HEADER + self.XRF55_DAT_COMPLEX * 2:
                    return False
                # Try to read and check header byte pattern
                with open(file_path, 'rb') as f:
                    _ = np.fromfile(f, dtype=np.int16, count=4)
                # Header should be nonzero or structured differently from pure data
                return True
            return False
        except Exception:
            return False

    def read_file(self, file_path) -> List[CSIData]:
        """Read XRF55 file (.npy or .dat) and return list of CSIData."""
        s = str(file_path)
        _, ext = os.path.splitext(s)
        ext = ext.lower()
        if ext == '.dat':
            return self._read_dat(file_path)
        else:
            return self._read_npy(file_path)

    def _read_dat(self, file_path) -> List[CSIData]:
        """Read xrf55 .dat binary format."""
        try:
            data = np.fromfile(file_path, dtype=np.int16)
        except FileNotFoundError:
            print(f"cannot find file: {file_path}")
            return []

        int16_count = len(data)
        expected_payload = self.XRF55_DAT_PACKETS * self.XRF55_DAT_COMPLEX * 2
        if int16_count < self.XRF55_DAT_HEADER + expected_payload:
            print(f"file too small for xrf55 format: {file_path} ({int16_count} int16)")
            return []

        # Skip header, read payload
        payload = data[self.XRF55_DAT_HEADER:]
        n_packets = len(payload) // (self.XRF55_DAT_COMPLEX * 2)
        if n_packets == 0:
            print(f"no complete packets in: {file_path}")
            return []

        # Reshape to (n_packets, 270, 2) = (time, complex_vals, I/Q)
        try:
            pkt_array = payload[:n_packets * self.XRF55_DAT_COMPLEX * 2].reshape(
                n_packets, self.XRF55_DAT_COMPLEX, 2
            )
        except ValueError as e:
            print(f"reshape failed for {file_path}: {e}")
            return []

        I_data = pkt_array[:, :, 0].astype(np.float32)
        Q_data = pkt_array[:, :, 1].astype(np.float32)
        csi_complex = I_data + 1j * Q_data  # (n_packets, 270)

        # Parse labels from path: .../Scene_X/{lb|nb}/YY/YY_AA_BB.dat
        parts = file_path.split(os.sep)
        scene = los = person = action = trial = None
        for i, p in enumerate(parts):
            if p.startswith('Scene_'):
                try:
                    scene = int(p.split('_')[1])
                except (ValueError, IndexError):
                    pass
            if p in ('lb', 'nb'):
                los = p
            # File name pattern: YY_AA_BB.dat
            if '_' in p and p.endswith('.dat'):
                fname = p.replace('.dat', '')
                segs = fname.split('_')
                if len(segs) == 3 and all(s.isdigit() for s in segs):
                    person, action, trial = segs

        # Create one CSIData with all frames (one frame per timestamp)
        # Each frame stores (F, A) = (30, 9) = (subcarriers, rx*ant)
        # After np.stack: (T=199, F=30, A=9) which is what phase_calibration expects
        csi_data = CSIData(file_path)
        for t in range(n_packets):
            # csi_complex[t]: (270,) complex -> reshape to (3, 3, 30) = (Rx, Ant, Sub)
            # Then transpose+reshape to (30, 9) = (F, A)
            csi_3d = csi_complex[t].reshape(3, 3, 30)  # (Rx, Ant, Sub)
            csi_2d = csi_3d.transpose(2, 0, 1).reshape(30, 9)  # (F, Rx*Ant) = (30, 9)
            frame = BaseFrame(
                timestamp=str(t),
                csi_array=csi_2d
            )
            csi_data.add_frame(frame)

        # Store parsed labels on the object
        csi_data._xrf55_labels = {
            'scene': scene,
            'los': los,
            'person': int(person) if person else None,
            'action': int(action) if action else None,
            'trial': int(trial) if trial else None,
        }

        return [csi_data]

    def _read_npy(self, file_path) -> List[CSIData]:
        """Read legacy .npy format."""
        try:
            raw_data = np.load(file_path)
        except FileNotFoundError:
            print(f"cannot find file: {file_path}")
            return []

        try:
            # Legacy shape: (3, 30, 3, 1000) = (rx, sub, ant, time)
            reshaped_data = raw_data.reshape(3, 30, 3, 1000)
        except ValueError as e:
            print(f"reshape failed: {e}")
            return []

        num_time_steps = 1000

        # Old logic: split one .npy file into 3 CSIData samples by Rx.
        # csi_data_list = []
        # num_receivers = 3
        # for rx_idx in range(num_receivers):
        #     csi_data = CSIData(file_path)
        #     current_rx_data = reshaped_data[rx_idx]
        #
        #     for timestamp in range(num_time_steps):
        #         csi_array = current_rx_data[:, :, timestamp]
        #         csi_array = csi_array.copy()
        #         frame = BaseFrame(timestamp=timestamp, csi_array=csi_array)
        #         csi_data.add_frame(frame)
        #
        #     csi_data_list.append(csi_data)
        #
        # return csi_data_list

        # New XRF55 .npy logic: keep all 3 Rx in one CSIData sample.
        # Each frame stores (F, A) = (30, 9), where 9 = 3 Rx * 3 Ant.
        csi_data = CSIData(file_path)
        for timestamp in range(num_time_steps):
            # reshaped_data[:, :, :, timestamp]: (Rx, Sub, Ant)
            # Convert to (Sub, Rx, Ant) -> (30, 9), matching the .dat path.
            csi_array = reshaped_data[:, :, :, timestamp].transpose(1, 0, 2).reshape(30, 9)
            csi_array = csi_array.copy()
            frame = BaseFrame(timestamp=timestamp, csi_array=csi_array)
            csi_data.add_frame(frame)

        return [csi_data]
