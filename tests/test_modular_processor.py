"""Tests for the modular processor and flexible algorithm combinations."""

import numpy as np
import pytest

from wsdp.algorithms import (
    AlgorithmStep,
    execute_algorithm_steps,
    register_algorithm,
    unregister_algorithm,
)
from wsdp.processors import BaseProcessor, ConfigurableProcessor, ModularProcessor
from wsdp.structure import BaseFrame, CSIData


def _make_csi_data(file_name: str, shape=(20, 30, 3)):
    """Create a CSIData object with random complex frames."""
    csi = CSIData(file_name)
    for t in range(shape[0]):
        arr = np.random.randn(*shape[1:]) + 1j * np.random.randn(*shape[1:])
        csi.frames.append(BaseFrame(timestamp=t, csi_array=arr))
    return csi


class TestModularProcessor:
    def test_runs_steps_in_order(self):
        steps = [
            AlgorithmStep(category="denoise", method="wavelet"),
            AlgorithmStep(category="calibrate", method="linear"),
        ]
        processor = ModularProcessor(steps, n_workers=1)
        csi_data = _make_csi_data("user1_2_03.dat", shape=(20, 30, 3))
        data, labels, groups = processor.process([csi_data], dataset="xrf55")

        assert len(data) == 1
        assert data[0].shape == (20, 30, 3)
        assert labels == [2]

    def test_can_skip_step(self):
        """A2C1 combination: denoise + normalize, skip calibrate."""
        steps = [
            AlgorithmStep(category="denoise", method="wavelet"),
            AlgorithmStep(category="normalize", method="z-score"),
        ]
        processor = ModularProcessor(steps, n_workers=1)
        csi_data = _make_csi_data("user1_2_03.dat", shape=(20, 30, 3))
        data, labels, groups = processor.process([csi_data], dataset="xrf55")

        assert len(data) == 1
        assert data[0].shape == (20, 30, 3)

    def test_custom_algorithm_in_modular_processor(self):
        def my_denoise(csi, factor=1.0, **kwargs):
            return csi * factor

        register_algorithm("denoise", "my_test", my_denoise)
        try:
            steps = [
                AlgorithmStep(category="denoise", method="my_test", params={"factor": 0.5}),
                AlgorithmStep(category="calibrate", method="linear"),
            ]
            processor = ModularProcessor(steps, n_workers=1)
            csi_data = _make_csi_data("user1_2_03.dat", shape=(20, 30, 3))
            data, _, _ = processor.process([csi_data], dataset="xrf55")
            assert data[0].shape == (20, 30, 3)
        finally:
            unregister_algorithm("denoise", "my_test")


class TestBaseProcessorBackwardCompatibility:
    def test_base_processor_runs_default_pipeline(self):
        processor = BaseProcessor(n_workers=1)
        csi_data = _make_csi_data("user1_2_03.dat", shape=(20, 30, 3))
        data, labels, groups = processor.process([csi_data], dataset="xrf55")

        assert len(data) == 1
        assert data[0].shape == (20, 30, 3)
        assert labels == [2]


class TestConfigurableProcessorBackwardCompatibility:
    def test_configurable_processor_runs_user_steps(self):
        steps = {
            "denoise": {"method": "wavelet"},
            "calibrate": {"method": "linear"},
        }
        processor = ConfigurableProcessor(steps)
        csi_data = _make_csi_data("user1_2_03.dat", shape=(20, 30, 3))
        data, labels, groups = processor.process([csi_data], dataset="xrf55")

        assert len(data) == 1
        assert data[0].shape == (20, 30, 3)

    def test_configurable_processor_skips_normalize_for_xrf55(self):
        steps = {
            "denoise": {"method": "wavelet"},
            "normalize": {"method": "z-score"},
        }
        processor = ConfigurableProcessor(steps)
        csi_data = _make_csi_data("user1_2_03.dat", shape=(20, 30, 3))
        data, _, _ = processor.process([csi_data], dataset="xrf55")
        assert len(data) == 1


class TestExecuteAlgorithmSteps:
    def test_state_routing(self):
        csi = np.random.randn(20, 30, 3) + 1j * np.random.randn(20, 30, 3)
        steps = [
            AlgorithmStep(
                category="denoise",
                method="wavelet",
                output_key="cleaned",
            ),
            AlgorithmStep(
                category="calibrate",
                method="linear",
                input_key="cleaned",
                output_key="csi",
            ),
        ]
        state = execute_algorithm_steps(csi, steps, dataset="xrf55")
        assert "cleaned" in state
        assert "csi" in state
        assert state["csi"].shape == csi.shape

    def test_disabled_step_is_skipped(self):
        csi = np.random.randn(20, 30, 3) + 1j * np.random.randn(20, 30, 3)
        steps = [
            AlgorithmStep(category="denoise", method="wavelet", enabled=False),
            AlgorithmStep(category="calibrate", method="linear"),
        ]
        state = execute_algorithm_steps(csi, steps, dataset="xrf55")
        assert state["csi"].shape == csi.shape
