"""Tests for the public component interfaces."""

import pytest

from wsdp.interfaces import Processor, Reader
from wsdp.processors import BaseProcessor, ConfigurableProcessor, ModularProcessor
from wsdp.readers import (
    BaseReader,
    create_reader,
    list_datasets,
    register_reader,
    unregister_reader,
)
from wsdp.models import create_model, list_models


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
        try:
            reader = create_reader("dummy_dataset")
            assert isinstance(reader, DummyReader)
        finally:
            unregister_reader("dummy_dataset")

        with pytest.raises(ValueError, match="not supported dataset"):
            create_reader("dummy_dataset")

    def test_list_datasets_includes_builtins(self):
        datasets = list_datasets()
        assert "widar" in datasets
        assert "xrf55" in datasets


class TestProcessorInterface:
    def test_processors_are_abc(self):
        assert issubclass(BaseProcessor, Processor)
        assert issubclass(ConfigurableProcessor, Processor)
        assert issubclass(ModularProcessor, Processor)


class TestModelRegistry:
    def test_create_model(self):
        model = create_model("CNN1DModel", num_classes=3, input_shape=(20, 30, 3))
        assert model is not None

    def test_list_models(self):
        models = list_models()
        assert "cnn1dmodel" in models
        assert "mlpmodel" in models
