"""Tests for offline pipeline diagnostic artifacts and metrics."""
import csv
import json
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest


def _synthetic_motion_csi(T=128, F=12, A=2, freq=0.12, noise=0.01):
    rng = np.random.default_rng(7)
    t = np.arange(T)
    base = np.ones((T, F, A), dtype=complex)
    motion = 0.4 * np.exp(1j * 2 * np.pi * freq * t)[:, None, None]
    subcarrier_weight = np.linspace(0.5, 1.5, F)[None, :, None]
    antenna_weight = np.linspace(1.0, 1.2, A)[None, None, :]
    csi = base + motion * subcarrier_weight * antenna_weight
    csi += noise * (rng.standard_normal(csi.shape) + 1j * rng.standard_normal(csi.shape))
    return csi


def test_run_pipeline_diagnostics_writes_png_metrics_and_manifest(tmp_path):
    from wsdp.diagnostics import run_pipeline_diagnostics

    raw = _synthetic_motion_csi()
    processed = raw * 0.8

    result = run_pipeline_diagnostics(
        raw,
        processed,
        tmp_path,
        sample_name="gesture_001",
        include_doppler=True,
        sampling_rate=1.0,
        motion_band=(0.05, 0.2),
        antenna_idx=1,
        n_fft=32,
        hop_length=16,
        pipeline_name="gesture_baseline",
    )

    diagnostic_path = tmp_path / "gesture_001_diagnostic.png"
    metrics_path = tmp_path / "metrics.csv"
    manifest_path = tmp_path / "manifest.json"

    assert result["diagnostic_path"] == str(diagnostic_path)
    assert result["metrics_path"] == str(metrics_path)
    assert result["manifest_path"] == str(manifest_path)
    assert diagnostic_path.exists()
    assert metrics_path.exists()
    assert manifest_path.exists()

    with metrics_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["sample"] == "gesture_001"
    assert float(rows[0]["raw_dynamic_energy"]) > 0
    assert float(rows[0]["processed_dynamic_energy"]) > 0

    manifest = json.loads(manifest_path.read_text())
    assert len(manifest["samples"]) == 1
    sample = manifest["samples"][0]
    assert sample["sample_name"] == "gesture_001"
    assert sample["include_doppler"] is True
    assert sample["raw_shape"] == [128, 12, 2]
    assert sample["antenna_idx"] == 1
    assert sample["n_fft"] == 32
    assert sample["hop_length"] == 16
    assert sample["pipeline_name"] == "gesture_baseline"


def test_compute_pipeline_metrics_detects_over_smoothed_motion_loss():
    from wsdp.diagnostics import compute_pipeline_metrics

    raw = _synthetic_motion_csi()
    over_smoothed = np.repeat(raw.mean(axis=0, keepdims=True), raw.shape[0], axis=0)

    metrics = compute_pipeline_metrics(
        raw,
        over_smoothed,
        sampling_rate=1.0,
        motion_band=(0.05, 0.2),
    )

    assert metrics["processed_dynamic_energy"] < metrics["raw_dynamic_energy"] * 0.05
    assert metrics["processed_motion_band_energy_ratio"] < metrics["raw_motion_band_energy_ratio"]
    assert metrics["signal_preservation_ratio"] < 0.5


def test_pipeline_diagnostics_rejects_mismatched_shapes_without_outputs(tmp_path):
    from wsdp.diagnostics import run_pipeline_diagnostics

    raw = _synthetic_motion_csi(T=64, F=10, A=1)
    processed = _synthetic_motion_csi(T=64, F=12, A=1)

    with pytest.raises(ValueError, match="same shape"):
        run_pipeline_diagnostics(raw, processed, tmp_path, sample_name="bad")

    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("sample_name", ["../escape", "/tmp/escape", "bad/name", "bad\\name", "", "."])
def test_pipeline_diagnostics_rejects_unsafe_sample_names_without_outputs(tmp_path, sample_name):
    from wsdp.diagnostics import run_pipeline_diagnostics

    raw = _synthetic_motion_csi(T=64, F=8, A=1)
    processed = raw * 0.9

    with pytest.raises(ValueError, match="sample_name"):
        run_pipeline_diagnostics(raw, processed, tmp_path, sample_name=sample_name)

    assert not list(tmp_path.iterdir())


def test_run_pipeline_diagnostics_appends_metrics_and_manifest_samples(tmp_path):
    from wsdp.diagnostics import run_pipeline_diagnostics

    raw = _synthetic_motion_csi(T=64, F=8, A=1)
    processed = raw * 0.9

    run_pipeline_diagnostics(raw, processed, tmp_path, sample_name="sample_a")
    run_pipeline_diagnostics(raw, processed, tmp_path, sample_name="sample_b")

    with (tmp_path / "metrics.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert [row["sample"] for row in rows] == ["sample_a", "sample_b"]

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert [sample["sample_name"] for sample in manifest["samples"]] == ["sample_a", "sample_b"]
    assert (tmp_path / "sample_a_diagnostic.png").exists()
    assert (tmp_path / "sample_b_diagnostic.png").exists()


def test_plot_pipeline_diagnostics_uses_independent_raw_processed_scales(tmp_path):
    from wsdp.diagnostics import plot_pipeline_diagnostics

    raw = _synthetic_motion_csi(T=64, F=8, A=1)
    processed = raw * 0.2
    output = tmp_path / "panel.png"

    fig = plot_pipeline_diagnostics(raw, processed, output, include_doppler=True, n_fft=32, hop_length=16)

    image_axes = [ax for ax in fig.axes if ax.images]
    # Raw and processed amplitude panels must auto-scale independently so
    # normalization does not collapse to a single color.
    assert image_axes[0].images[0].get_clim() != image_axes[1].images[0].get_clim()
    assert image_axes[4].images[0].get_clim() != image_axes[5].images[0].get_clim()
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_pipeline_diagnostics_shows_normalized_processed_data(tmp_path):
    from wsdp.diagnostics import plot_pipeline_diagnostics

    raw = np.abs(_synthetic_motion_csi(T=64, F=8, A=1)) * 100.0
    processed = (raw - np.mean(raw, axis=0, keepdims=True)) / (
        np.std(raw, axis=0, keepdims=True) + 1e-12
    )
    output = tmp_path / "panel.png"

    fig = plot_pipeline_diagnostics(raw, processed, output, include_doppler=True, n_fft=32, hop_length=16)
    assert output.exists()

    image_axes = [ax for ax in fig.axes if ax.images]
    raw_clim = image_axes[0].images[0].get_clim()
    proc_clim = image_axes[1].images[0].get_clim()
    # Normalized output should have a much smaller value range than raw.
    assert (proc_clim[1] - proc_clim[0]) < (raw_clim[1] - raw_clim[0]) * 0.5
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_pipeline_diagnostics_can_omit_doppler_panel(tmp_path):
    from wsdp.diagnostics import plot_pipeline_diagnostics

    raw = _synthetic_motion_csi(T=64, F=8, A=1)
    processed = raw * 0.9
    output = tmp_path / "panel.png"

    fig = plot_pipeline_diagnostics(raw, processed, output, include_doppler=False)

    assert output.exists()
    assert len(fig.axes) >= 4
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_diagnostics_metrics_import_is_lightweight():
    code = """
import sys
from wsdp.diagnostics import compute_pipeline_metrics
blocked = {'torch', 'sklearn', 'matplotlib', 'scipy'} & set(sys.modules)
if blocked:
    raise SystemExit(','.join(sorted(blocked)))
assert callable(compute_pipeline_metrics)
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_algorithms_diagnostics_exports_remain_backward_compatible():
    from wsdp.algorithms import compute_pipeline_metrics as legacy_metrics
    from wsdp.diagnostics import compute_pipeline_metrics

    assert legacy_metrics is compute_pipeline_metrics
