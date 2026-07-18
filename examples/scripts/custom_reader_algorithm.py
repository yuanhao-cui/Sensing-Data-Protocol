"""Example: register a custom reader and algorithm and run a modular pipeline.

This demonstrates how users can extend WSDP without modifying the core
source code.
"""

import numpy as np

from wsdp import pipeline
from wsdp.algorithms import register_algorithm
from wsdp.readers import BaseReader, register_reader
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

    # 2. Build a modular pipeline: A2C1 (denoise + normalize, skip calibrate)
    steps = {
        "denoise": {"method": "my_denoise", "strength": 0.9},
        "normalize": {"method": "z-score"},
    }

    # 3. Run the pipeline
    # NOTE: this example assumes ./data/my_dataset exists with .myfmt files.
    # pipeline(
    #     input_path="./data/my_dataset",
    #     output_folder="./output",
    #     dataset="my_dataset",
    #     model_name="ResNet1D",
    #     pipeline_steps=steps,
    # )

    print("Registered readers:", ["my_dataset"])
    print("Pipeline steps:", steps)
