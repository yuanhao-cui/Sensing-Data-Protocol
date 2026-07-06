"""Tests for dataset-driven model input representation."""

import inspect
import unittest
from types import SimpleNamespace

import numpy as np

from wsdp.algorithms.amplitude import normalize_amplitude
from wsdp.dataset_policy import pipeline_uses_zscore, uses_phase_amplitude
from wsdp.datasets import CSIDataset
from wsdp.processors.configurable_processor import (
    _process_single_csi_configurable,
)


ZSCORE_STEPS = {"normalize": {"method": "z-score"}}


def _complex_sample():
    amplitude = np.array(
        [
            [[1.0], [4.0]],
            [[2.0], [5.0]],
            [[4.0], [9.0]],
        ]
    )
    phase = np.array(
        [
            [[0.2], [-0.4]],
            [[0.6], [0.9]],
            [[-1.0], [1.4]],
        ]
    )
    return amplitude * np.exp(1j * phase), amplitude, phase


def _csi_data(file_name):
    csi, _, _ = _complex_sample()
    frames = [
        SimpleNamespace(timestamp=index, csi_array=csi[index])
        for index in range(csi.shape[0])
    ]
    return SimpleNamespace(file_name=file_name, frames=frames)


class DatasetRepresentationTests(unittest.TestCase):
    def test_policy_selects_only_widar_and_gait_for_phase(self):
        self.assertTrue(uses_phase_amplitude("widar"))
        self.assertTrue(uses_phase_amplitude("gait"))
        self.assertTrue(uses_phase_amplitude("Gait"))
        self.assertFalse(uses_phase_amplitude("xrf55"))
        self.assertFalse(uses_phase_amplitude("elderAL"))
        self.assertTrue(pipeline_uses_zscore(ZSCORE_STEPS))
        self.assertFalse(
            pipeline_uses_zscore({"normalize": {"method": "min-max"}})
        )
        self.assertFalse(pipeline_uses_zscore(None))

    def test_csidataset_api_has_no_use_phase_switch(self):
        parameters = inspect.signature(CSIDataset.__init__).parameters
        self.assertNotIn("use_phase", parameters)

    def test_processor_zscore_preserves_signed_amplitude_and_true_phase(self):
        cases = (
            ("widar", "user1-1-1-1-1-r1.dat"),
            ("gait", "user1-1-1-r1.dat"),
        )
        csi, amplitude, phase = _complex_sample()
        expected_mean = np.mean(amplitude, axis=0, keepdims=True)
        expected_std = np.std(amplitude, axis=0, keepdims=True)
        expected_norm = (amplitude - expected_mean) / expected_std

        for dataset_name, file_name in cases:
            with self.subTest(dataset_name=dataset_name):
                result, _, _ = _process_single_csi_configurable(
                    _csi_data(file_name),
                    dataset=dataset_name,
                    pipeline_steps=ZSCORE_STEPS,
                )

                antenna_count = csi.shape[-1]
                self.assertFalse(np.iscomplexobj(result))
                self.assertEqual(result.shape[-1], antenna_count * 2)
                np.testing.assert_allclose(
                    result[..., :antenna_count],
                    expected_norm,
                    atol=1e-12,
                )
                np.testing.assert_allclose(
                    result[..., antenna_count:],
                    phase,
                    atol=1e-12,
                )
                self.assertTrue(np.any(result[..., :antenna_count] < 0))
                self.assertTrue(
                    np.any(np.abs(result[..., antenna_count:]) > 0)
                )

    def test_csidataset_automatically_adds_phase_without_zscore(self):
        csi, amplitude, phase = _complex_sample()
        data = csi[np.newaxis, ...]
        labels = np.array([0])

        for dataset_name in ("widar", "gait"):
            with self.subTest(dataset_name=dataset_name):
                dataset = CSIDataset(
                    data,
                    labels,
                    dataset_name=dataset_name,
                )
                result = dataset.data_list.numpy()
                antenna_count = csi.shape[-1]

                self.assertEqual(result.shape[-1], antenna_count * 2)
                np.testing.assert_allclose(
                    result[0, ..., :antenna_count],
                    amplitude,
                    atol=1e-6,
                )
                np.testing.assert_allclose(
                    result[0, ..., antenna_count:],
                    phase,
                    atol=1e-6,
                )

    def test_csidataset_keeps_prepared_phase_zscore_channels_unchanged(self):
        prepared = np.array(
            [[[[-1.5, 0.4], [0.8, -0.7]]]],
            dtype=np.float64,
        )
        labels = np.array([0])

        dataset = CSIDataset(
            prepared,
            labels,
            dataset_name="widar",
            pipeline_steps=ZSCORE_STEPS,
        )

        np.testing.assert_allclose(
            dataset.data_list.numpy(),
            prepared,
            atol=1e-7,
        )
        self.assertLess(dataset.data_list[0, 0, 0, 0].item(), 0)

    def test_amplitude_only_datasets_do_not_add_phase_channels(self):
        real_amplitude = np.array(
            [[[[1.0], [2.0]], [[3.0], [4.0]]]]
        )
        labels = np.array([0])

        for dataset_name in ("xrf55", "elderAL"):
            with self.subTest(dataset_name=dataset_name):
                dataset = CSIDataset(
                    real_amplitude,
                    labels,
                    dataset_name=dataset_name,
                    pipeline_steps=ZSCORE_STEPS,
                )
                self.assertEqual(
                    tuple(dataset.data_list.shape),
                    real_amplitude.shape,
                )

    def test_normalize_amplitude_can_return_phase_channels_directly(self):
        csi, amplitude, phase = _complex_sample()
        expected = (
            amplitude - amplitude.mean(axis=0, keepdims=True)
        ) / amplitude.std(axis=0, keepdims=True)

        result = normalize_amplitude(
            csi,
            method="z-score",
            return_phase_channels=True,
        )

        np.testing.assert_allclose(result[..., :1], expected, atol=1e-12)
        np.testing.assert_allclose(result[..., 1:], phase, atol=1e-12)

    def test_normalize_amplitude_default_complex_output_is_unchanged(self):
        csi, _, _ = _complex_sample()

        result = normalize_amplitude(csi, method="z-score")

        self.assertEqual(result.shape, csi.shape)
        self.assertTrue(np.iscomplexobj(result))


if __name__ == "__main__":
    unittest.main()
