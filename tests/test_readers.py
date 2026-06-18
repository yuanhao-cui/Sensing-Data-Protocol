"""Tests for reader modules."""

import os

import numpy as np
import pytest

from wsdp.readers import (
    list_datasets, get_reader_class, get_all_reader_metadata,
    _collect_data_files, _extend_csi_data,
    BfeeReader, XrfReader, ElderReader, ZTEReader,
)
from wsdp.structure import CSIData


class TestReaderRegistry:
    def test_list_datasets(self):
        datasets = list_datasets()
        assert isinstance(datasets, list)
        assert len(datasets) >= 5
        assert "widar" in datasets
        assert "gait" in datasets
        assert "xrf55" in datasets
        assert "elderAL" in datasets
        assert "zte" in datasets

    def test_list_datasets_sorted(self):
        datasets = list_datasets()
        assert datasets == sorted(datasets)

    def test_get_reader_class_valid(self):
        cls = get_reader_class("widar")
        assert cls is BfeeReader

        cls = get_reader_class("gait")
        assert cls is BfeeReader

        cls = get_reader_class("xrf55")
        assert cls is XrfReader

    def test_get_reader_class_invalid(self):
        with pytest.raises(ValueError, match="not supported dataset"):
            get_reader_class("nonexistent")


class TestReaderMetadata:
    def test_bfee_metadata(self):
        meta = BfeeReader().get_metadata()
        assert meta["reader"] == "BfeeReader"
        assert meta["format"] == "bfee"
        assert "description" in meta
        assert meta["complex"] is True

    def test_xrf_metadata(self):
        meta = XrfReader().get_metadata()
        assert meta["reader"] == "XrfReader"
        assert meta["receivers"] == 3
        assert meta["subcarriers"] == 30

    def test_elder_metadata(self):
        meta = ElderReader().get_metadata()
        assert meta["reader"] == "ElderReader"
        assert meta["format"] == "csv"
        assert meta["complex"] is False

    def test_zte_metadata(self):
        meta = ZTEReader().get_metadata()
        assert meta["reader"] == "ZTEReader"
        assert meta["subcarriers"] == 512

    def test_get_all_reader_metadata(self):
        for ds in list_datasets():
            meta = get_all_reader_metadata(ds)
            assert "reader" in meta
            assert "format" in meta


class TestReaderLoadingHelpers:
    def test_collect_data_files_excludes_truth_files(self, tmp_path):
        data_file = tmp_path / "sample.dat"
        truth_file = tmp_path / "sample_truth.csv"
        nested_dir = tmp_path / "nested"
        nested_dir.mkdir()
        nested_file = nested_dir / "nested.dat"
        data_file.write_text("data")
        truth_file.write_text("truth")
        nested_file.write_text("nested")

        files = _collect_data_files(tmp_path)

        assert set(files) == {data_file, nested_file}

    def test_extend_csi_data_accepts_single_and_list_items(self):
        target = []
        first = CSIData("first")
        second = CSIData("second")

        _extend_csi_data(target, first)
        _extend_csi_data(target, [second])

        assert target == [first, second]
