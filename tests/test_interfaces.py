"""Tests for the public component interfaces."""

import numpy as np
import pytest
import torch.nn as nn

from wsdp.interfaces import ModelBuilder, Processor, Reader
from wsdp.processors import BaseProcessor, ConfigurableProcessor, ModularProcessor
from wsdp.readers import (
    BaseReader,
    create_reader,
    list_readers,
    register_reader,
    unregister_reader,
)
from wsdp.models import (
    ClassModelBuilder,
    create_model,
    get_model_builder,
    list_models,
    register_model_builder,
    unregister_model,
)


class DummyReader(BaseReader):
    def sniff(self, file_path: str) -> bool:
        return True

    def read_file(self, file_path: str):
        return []


class TestReaderInterface:
    def test_base_reader_is_reader_abc(self):
        assert issubclass(BaseReader, Reader)

    def test_register_and_create_reader(self):
        register_reader("dummy_dataset", DummyReader)
        reader = create_reader("dummy_dataset")
        assert isinstance(reader, DummyReader)

        unregister_reader("dummy_dataset")
        with pytest.raises(ValueError, match="not supported dataset"):
            create_reader("dummy_dataset")

    def test_list_readers_includes_builtins(self):
        readers = list_readers()
        assert "widar" in readers
        assert "xrf55" in readers


class TestProcessorInterface:
    def test_processors_are_abc(self):
        assert issubclass(BaseProcessor, Processor)
        assert issubclass(ConfigurableProcessor, Processor)
        assert issubclass(ModularProcessor, Processor)


class DummyModelBuilder(ModelBuilder):
    def __init__(self):
        self.built = False

    def build(self, num_classes: int, input_shape: tuple, **kwargs) -> nn.Module:
        self.built = True
        return nn.Linear(10, num_classes)

    def get_name(self) -> str:
        return "dummy"


class TestModelBuilderInterface:
    def test_register_custom_builder(self):
        builder = DummyModelBuilder()
        register_model_builder("custom", "dummy_builder", builder)
        model = create_model("dummy_builder", num_classes=3, input_shape=(10, 5, 2))
        assert isinstance(model, nn.Linear)
        assert model.out_features == 3
        assert builder.built

        # Cleanup
        unregister_model("dummy_builder")

    def test_get_model_builder(self):
        builder = get_model_builder("cnn1dmodel")
        assert isinstance(builder, ClassModelBuilder)

    def test_list_models(self):
        models = list_models()
        assert "cnn1dmodel" in models
        assert "mlpmodel" in models
