import logging
from typing import List

from wsdp.algorithms import AlgorithmStep
from wsdp.dataset_policy import real_if_negligible_imaginary
from wsdp.interfaces import Processor
from wsdp.processors.metadata import parse_file_info_from_filename, select_label_and_group
from wsdp.processors.modular_processor import ModularProcessor
from wsdp.structure import CSIData

logger = logging.getLogger(__name__)


class BaseProcessor(Processor):
    """Legacy default processor: phase calibration followed by wavelet denoising.

    This processor is kept for backward compatibility. Internally it builds a
    fixed two-step ``AlgorithmStep`` pipeline and delegates to
    ``ModularProcessor``.
    """

    def __init__(self, *, n_workers: int = 16):
        steps = [
            AlgorithmStep(category="calibrate", method="linear"),
            AlgorithmStep(category="denoise", method="wavelet"),
        ]
        self._modular = ModularProcessor(steps, n_workers=n_workers)

    def process(
        self,
        data_list: List[CSIData],
        **kwargs,
    ):
        dataset = kwargs.get("dataset", "")
        all_data, all_labels, all_groups = self._modular.process(
            data_list, dataset=dataset
        )
        # Preserve historical amplitude-primary cleanup.
        all_data = [
            real_if_negligible_imaginary(arr, dataset) for arr in all_data
        ]
        return all_data, all_labels, all_groups


# Backward-compatible aliases for code that imports these helpers directly.
_parse_file_info_from_filename = parse_file_info_from_filename
_selector = select_label_and_group
