"""Example: register a custom reader and algorithm, then run a modular pipeline.

This demonstrates how users can extend WSDP without modifying the core
source code. It runs end-to-end on synthetic in-memory data:

1. ``register_reader`` plugs in a new file-format reader.
2. ``register_algorithm`` plugs in a new algorithm.
3. ``ModularProcessor`` runs a flexible step combination (here A2C1:
   denoise + normalize, skipping calibration) over CSIData samples.

To run a full training pipeline with the same building blocks::

    from wsdp import pipeline
    pipeline(
        input_path="./data/my_dataset",
        output_folder="./output",
        dataset="my_dataset",
        model_name="ResNet1D",
        pipeline_steps={
            "denoise": {"method": "my_denoise", "strength": 0.9},
            "normalize": {"method": "z-score"},
        },
    )
"""

import numpy as np

from wsdp.algorithms import AlgorithmStep, register_algorithm
from wsdp.processors import ModularProcessor
from wsdp.readers import BaseReader, create_reader, register_reader
from wsdp.structure import BaseFrame, CSIData


class MyReader(BaseReader):
    """A tiny example reader that produces synthetic CSI samples."""

    def sniff(self, file_path: str) -> bool:
        return file_path.endswith(".myfmt")

    def read_file(self, file_path: str):
        csi = CSIData(file_path)
        for t in range(10):
            arr = np.random.randn(30, 3) + 1j * np.random.randn(30, 3)
            csi.frames.append(BaseFrame(timestamp=t, csi_array=arr))
        return csi


def my_denoise(csi, strength=1.0, **kwargs):
    """Custom denoising algorithm: simple scaling."""
    return csi * strength


if __name__ == "__main__":
    # 1. Register custom components
    register_reader("my_dataset", MyReader)
    register_algorithm("denoise", "my_denoise", my_denoise)

    reader = create_reader("my_dataset")
    print("Registered reader:", reader.get_metadata())

    # 2. Build a modular pipeline: A2C1 (denoise + normalize, skip calibrate)
    steps = [
        AlgorithmStep(category="denoise", method="my_denoise", params={"strength": 0.9}),
        AlgorithmStep(category="normalize", method="z-score"),
    ]

    # 3. Run it on a synthetic sample (filename follows the xrf55 convention
    #    user_action_trial so label/group parsing works)
    sample = MyReader().read_file("01_02_03.myfmt")
    processor = ModularProcessor(steps, n_workers=1)
    data, labels, groups = processor.process([sample], dataset="xrf55")

    print("Processed shape:", data[0].shape)
    print("Labels:", labels, "Groups:", groups)
