"""Tests for configurable core pipeline wiring."""

from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn

from wsdp.core import _resolve_pipeline_steps, pipeline
from wsdp.utils.cache import get_cache_key


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(12, 2)

    def forward(self, x):
        if torch.is_complex(x):
            x = torch.abs(x)
        return self.fc(x.float().reshape(x.shape[0], -1))


class DummyDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data,
        labels,
        dataset_name="",
        pipeline_steps=None,
    ):
        self.data_list = torch.from_numpy(data).float()
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data_list[idx], torch.tensor(self.labels[idx])


def _split_data():
    train_data = np.ones((4, 4, 3, 1), dtype=np.float32)
    val_data = np.ones((2, 4, 3, 1), dtype=np.float32)
    test_data = np.ones((2, 4, 3, 1), dtype=np.float32)
    train_labels = np.array([0, 1, 0, 1])
    val_labels = np.array([0, 1])
    test_labels = np.array([0, 1])
    return train_data, val_data, test_data, train_labels, val_labels, test_labels


def test_resolve_pipeline_steps_priority():
    explicit = {"denoise": {"method": "wavelet"}}
    assert _resolve_pipeline_steps(
        pipeline_steps=explicit,
        algorithm_config_file="ignored.yaml",
        algorithm_preset="fast",
    ) == explicit


def test_resolve_pipeline_steps_loads_config_file():
    steps = {"calibrate": {"method": "stc"}}
    with patch("wsdp.core.load_algorithm_config", return_value=steps) as load_config:
        assert _resolve_pipeline_steps(algorithm_config_file="algorithms.yaml") == steps
    load_config.assert_called_once_with("algorithms.yaml")


def test_cache_key_includes_preprocess_config(tmp_path):
    data_file = tmp_path / "sample.dat"
    data_file.write_text("sample")

    wavelet_key = get_cache_key(
        tmp_path,
        "xrf55",
        100,
        preprocess_config={"denoise": {"method": "wavelet"}},
    )
    butterworth_key = get_cache_key(
        tmp_path,
        "xrf55",
        100,
        preprocess_config={"denoise": {"method": "butterworth"}},
    )

    assert wavelet_key != butterworth_key


def test_pipeline_uses_registry_model_and_pipeline_steps(tmp_path):
    steps = {"denoise": {"method": "wavelet"}}
    dummy_model = DummyModel()
    checkpoint = tmp_path / "best_checkpoint_123.pth"
    checkpoint.write_text("placeholder")
    processed = (
        np.ones((8, 4, 3, 1), dtype=np.float32),
        np.array([0, 1, 0, 1, 0, 1, 0, 1]),
        np.array([0, 1, 2, 0, 1, 2, 0, 1]),
        [0, 1],
    )

    with patch("wsdp.core.load_params", return_value={"batch": 2, "lr": 0.001, "wd": 0.0, "num_epochs": 1, "padding_length": 4}), \
            patch("wsdp.core._load_and_preprocess", return_value=processed) as load_preprocess, \
            patch("wsdp.core._create_data_split", return_value=_split_data()), \
            patch("wsdp.core.CSIDataset", side_effect=DummyDataset), \
            patch("wsdp.core.create_model", return_value=dummy_model) as create_model, \
            patch("wsdp.core.train_model", return_value=[{"loss": 1.0}]), \
            patch("wsdp.core.random.randint", return_value=123), \
            patch("wsdp.core.torch.load", return_value={"model_state_dict": dummy_model.state_dict()}), \
            patch("wsdp.core._evaluate_model", return_value=([0, 1], [0, 1], 1.0)), \
            patch("wsdp.core.plt.figure"), \
            patch("wsdp.core.sns.heatmap"), \
            patch("wsdp.core.plt.savefig"), \
            patch("wsdp.core.plt.close"):
        pipeline(
            input_path="input",
            output_folder=str(tmp_path),
            dataset="xrf55",
            model_name="cnn1dmodel",
            pipeline_steps=steps,
            num_seeds=1,
            use_cache=False,
        )

    assert load_preprocess.call_args.kwargs["pipeline_steps"] == steps
    assert create_model.call_args.args[0] == "cnn1dmodel"
    assert create_model.call_args.kwargs["num_classes"] == 2
    assert create_model.call_args.kwargs["input_shape"] == (4, 3, 1)
