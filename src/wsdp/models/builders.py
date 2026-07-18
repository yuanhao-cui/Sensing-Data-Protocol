"""Model builders that turn registry entries into instantiated models."""

from typing import Any, Tuple, Type

import torch.nn as nn

from wsdp.interfaces import ModelBuilder


class ClassModelBuilder(ModelBuilder):
    """Build a model by instantiating a registered ``nn.Module`` subclass."""

    def __init__(
        self,
        name: str,
        model_class: Type[nn.Module],
    ):
        self._name = name
        self.model_class = model_class

    def build(
        self, num_classes: int, input_shape: Tuple[int, ...], **kwargs
    ) -> nn.Module:
        return self.model_class(
            num_classes=num_classes, input_shape=input_shape, **kwargs
        )

    def get_name(self) -> str:
        return self._name


class CustomModelBuilder(ModelBuilder):
    """Build a model loaded from an external Python file."""

    def __init__(self, model_path: str):
        self.model_path = model_path

    def build(
        self, num_classes: int, input_shape: Tuple[int, ...], **kwargs
    ) -> nn.Module:
        from wsdp.utils import load_custom_model
        return load_custom_model(
            self.model_path, num_classes, input_shape=input_shape, model_kwargs=kwargs
        )

    def get_name(self) -> str:
        return f"custom:{self.model_path}"
