"""Public component interfaces for WSDP.

This module defines the abstract contracts used across readers, processors,
and models. Keeping the contracts in one place makes it easy to swap
implementations without coupling to concrete classes.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import numpy as np
import torch.nn as nn


class Reader(ABC):
    """Dataset reader: turns a file path into one or more CSIData objects."""

    @abstractmethod
    def sniff(self, file_path: str) -> bool:
        """Return True if this reader recognizes the file format."""
        ...

    @abstractmethod
    def read_file(self, file_path: str) -> Any:
        """Read a single file and return a CSIData instance or a list of them."""
        ...

    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata about this reader and its supported format."""
        return {
            "reader": self.__class__.__name__,
            "format": self.__class__.__name__.replace("Reader", "").lower(),
        }


class Processor(ABC):
    """Sample-level processor: converts a list of CSIData into model-ready arrays.

    The returned tuple contains:
        - processed_data: list of numpy arrays, typically shaped (T, F, A)
        - labels: list of integer labels
        - groups: list of group identifiers for subject/condition-aware splitting
    """

    @abstractmethod
    def process(
        self, data_list: List[Any], **kwargs
    ) -> Tuple[List[np.ndarray], List[Any], List[Any]]:
        """Process CSIData objects and return arrays, labels, and groups."""
        ...


class ModelBuilder(ABC):
    """Pluggable model builder: constructs a model from dataset metadata."""

    @abstractmethod
    def build(
        self, num_classes: int, input_shape: Tuple[int, ...], **kwargs
    ) -> nn.Module:
        """Build and return a model instance."""
        ...

    @abstractmethod
    def get_name(self) -> str:
        """Return the model name for records and logging."""
        ...
