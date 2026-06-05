"""Offline diagnostics for comparing raw and processed CSI arrays."""
import csv
import json
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np

ArrayLike = Union[np.ndarray, Sequence[float]]
Band = Optional[Tuple[float, float]]


def _as_valid_pair(raw: ArrayLike, processed: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    raw_arr = np.asarray(raw)
    processed_arr = np.asarray(processed)

    if raw_arr.shape != processed_arr.shape:
        raise ValueError(
            f"raw and processed CSI must have the same shape; got "
            f"{raw_arr.shape} and {processed_arr.shape}"
        )
    if raw_arr.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D CSI arrays, got shape {raw_arr.shape}")
    if raw_arr.size == 0 or raw_arr.shape[0] < 2:
        raise ValueError("CSI arrays must contain at least two time steps")

    return raw_arr, processed_arr


def _safe_sample_name(sample_name: str) -> str:
    if not isinstance(sample_name, str):
        raise ValueError("sample_name must be a string")
    if sample_name in ("", ".", ".."):
        raise ValueError("sample_name must be a non-empty basename")
    if "/" in sample_name or "\\" in sample_name:
        raise ValueError("sample_name must not contain path separators")
    if Path(sample_name).is_absolute() or Path(sample_name).name != sample_name:
        raise ValueError("sample_name must be a basename, not a path")
    return sample_name


def _dynamic_signal(csi: np.ndarray) -> np.ndarray:
    return csi - np.mean(csi, axis=0, keepdims=True)


def _dynamic_energy(csi: np.ndarray) -> float:
    dynamic = _dynamic_signal(csi)
    return float(np.mean(np.abs(dynamic) ** 2))


def _power_spectrum(csi: np.ndarray, sampling_rate: float) -> Tuple[np.ndarray, np.ndarray]:
    if sampling_rate <= 0:
        raise ValueError(f"sampling_rate must be positive, got {sampling_rate}")

    dynamic = _dynamic_signal(csi)
    spectrum = np.fft.fft(dynamic, axis=0)
    power = np.mean(np.abs(spectrum) ** 2, axis=tuple(range(1, spectrum.ndim)))
    freqs = np.fft.fftfreq(csi.shape[0], d=1.0 / sampling_rate)
    return freqs, power


def _motion_band_ratio(csi: np.ndarray, sampling_rate: float, motion_band: Band) -> float:
    freqs, power = _power_spectrum(csi, sampling_rate)
    non_dc = np.abs(freqs) > 1e-12
    total = float(np.sum(power[non_dc]))
    if total <= 1e-12:
        return 0.0

    if motion_band is None:
        band = non_dc
    else:
        low, high = motion_band
        if low < 0 or high <= low:
            raise ValueError(f"motion_band must be (low, high) with high > low, got {motion_band}")
        band = non_dc & (np.abs(freqs) >= low) & (np.abs(freqs) <= high)
    return float(np.sum(power[band]) / total)


def _spectral_concentration(csi: np.ndarray, sampling_rate: float) -> float:
    freqs, power = _power_spectrum(csi, sampling_rate)
    non_dc_power = power[np.abs(freqs) > 1e-12]
    total = float(np.sum(non_dc_power))
    if total <= 1e-12:
        return 0.0
    return float(np.max(non_dc_power) / total)


def compute_pipeline_metrics(
    raw: ArrayLike,
    processed: ArrayLike,
    sampling_rate: float = 1.0,
    motion_band: Band = None,
) -> Dict[str, float]:
    """Compute CPU-only proxy metrics for raw/processed CSI comparison."""
    raw_arr, processed_arr = _as_valid_pair(raw, processed)

    raw_dynamic = _dynamic_energy(raw_arr)
    processed_dynamic = _dynamic_energy(processed_arr)
    raw_motion_ratio = _motion_band_ratio(raw_arr, sampling_rate, motion_band)
    processed_motion_ratio = _motion_band_ratio(processed_arr, sampling_rate, motion_band)

    return {
        "raw_dynamic_energy": raw_dynamic,
        "processed_dynamic_energy": processed_dynamic,
        "dynamic_energy_ratio": processed_dynamic / (raw_dynamic + 1e-12),
        "raw_motion_band_energy_ratio": raw_motion_ratio,
        "processed_motion_band_energy_ratio": processed_motion_ratio,
        "motion_band_energy_ratio_delta": processed_motion_ratio - raw_motion_ratio,
        "raw_spectral_concentration": _spectral_concentration(raw_arr, sampling_rate),
        "processed_spectral_concentration": _spectral_concentration(processed_arr, sampling_rate),
        "signal_preservation_ratio": processed_dynamic / (raw_dynamic + 1e-12),
        "mean_abs_difference": float(np.mean(np.abs(processed_arr - raw_arr))),
    }


def _antenna_slice(csi: np.ndarray, antenna_idx: int) -> np.ndarray:
    if csi.ndim == 2:
        return csi
    if not 0 <= antenna_idx < csi.shape[2]:
        raise ValueError(f"antenna_idx {antenna_idx} out of range for shape {csi.shape}")
    return csi[:, :, antenna_idx]


def _shared_limits(*arrays: np.ndarray) -> Tuple[float, float]:
    values = [np.asarray(array) for array in arrays]
    vmin = float(min(np.min(value) for value in values))
    vmax = float(max(np.max(value) for value in values))
    if np.isclose(vmin, vmax):
        eps = max(abs(vmin), 1.0) * 1e-12
        return vmin - eps, vmax + eps
    return vmin, vmax


def _plot_heatmap(
    ax,
    data: np.ndarray,
    title: str,
    cmap: str = "viridis",
    center: bool = False,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
    kwargs = {"aspect": "auto", "origin": "lower", "cmap": cmap}
    if center:
        centered_vmax = float(np.max(np.abs(data))) + 1e-12
        kwargs.update({"vmin": -centered_vmax, "vmax": centered_vmax})
    elif vmin is not None and vmax is not None:
        kwargs.update({"vmin": vmin, "vmax": vmax})
    im = ax.imshow(data.T, **kwargs)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Subcarrier")
    return im


def _frame_starts(length: int, window: int, hop_length: int) -> np.ndarray:
    if hop_length <= 0:
        raise ValueError(f"hop_length must be positive, got {hop_length}")
    if length <= window:
        return np.array([0], dtype=int)
    return np.arange(0, length - window + 1, hop_length, dtype=int)


def _doppler_summary(csi: np.ndarray, n_fft: int, hop_length: int) -> np.ndarray:
    if n_fft <= 1:
        raise ValueError(f"n_fft must be greater than 1, got {n_fft}")

    signal = _dynamic_signal(csi)
    window = min(n_fft, signal.shape[0])
    starts = _frame_starts(signal.shape[0], window, hop_length)
    weights_1d = np.hanning(window)
    if not np.any(weights_1d):
        weights_1d = np.ones(window)
    weights = weights_1d.reshape((window,) + (1,) * (signal.ndim - 1))

    frames = []
    for start in starts:
        frame = signal[start : start + window]
        padded = np.zeros((n_fft,) + signal.shape[1:], dtype=complex)
        padded[:window] = frame * weights
        spectrum = np.abs(np.fft.fftshift(np.fft.fft(padded, axis=0), axes=0))
        frames.append(np.mean(spectrum, axis=tuple(range(1, spectrum.ndim))))

    if not frames:
        return np.zeros((1, 1))
    return np.stack(frames, axis=1)


def plot_pipeline_diagnostics(
    raw: ArrayLike,
    processed: ArrayLike,
    save_path: Union[str, Path],
    include_doppler: bool = True,
    antenna_idx: int = 0,
    n_fft: int = 64,
    hop_length: int = 32,
    figsize: Optional[Tuple[int, int]] = None,
):
    """Create and save a fixed before/after diagnostic contact sheet."""
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt

    raw_arr, processed_arr = _as_valid_pair(raw, processed)
    raw_view = _antenna_slice(raw_arr, antenna_idx)
    processed_view = _antenna_slice(processed_arr, antenna_idx)
    raw_amplitude = np.abs(raw_view)
    processed_amplitude = np.abs(processed_view)
    amp_vmin, amp_vmax = _shared_limits(raw_amplitude, processed_amplitude)

    if include_doppler:
        figsize = figsize or (15, 8)
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)
        axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]
    else:
        figsize = figsize or (12, 8)
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
        axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]

    panels = [
        (raw_amplitude, "Raw Amplitude", "viridis", False, amp_vmin, amp_vmax),
        (processed_amplitude, "Processed Amplitude", "viridis", False, amp_vmin, amp_vmax),
        (processed_amplitude - raw_amplitude, "Amplitude Difference", "RdBu_r", True, None, None),
        (np.angle(processed_view * np.conj(raw_view)), "Phase Difference", "twilight", False, None, None),
    ]

    for ax, (data, title, cmap, center, vmin, vmax) in zip(axes, panels):
        im = _plot_heatmap(ax, data, title, cmap=cmap, center=center, vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax)

    if include_doppler:
        raw_doppler = _doppler_summary(raw_arr, n_fft=n_fft, hop_length=hop_length)
        processed_doppler = _doppler_summary(processed_arr, n_fft=n_fft, hop_length=hop_length)
        doppler_vmin, doppler_vmax = _shared_limits(raw_doppler, processed_doppler)
        for ax, data, title in (
            (axes[4], raw_doppler, "Raw Doppler Summary"),
            (axes[5], processed_doppler, "Processed Doppler Summary"),
        ):
            im = ax.imshow(
                data,
                aspect="auto",
                origin="lower",
                cmap="magma",
                vmin=doppler_vmin,
                vmax=doppler_vmax,
            )
            ax.set_title(title)
            ax.set_xlabel("STFT Frame")
            ax.set_ylabel("Frequency Bin")
            fig.colorbar(im, ax=ax)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def run_pipeline_diagnostics(
    raw: ArrayLike,
    processed: ArrayLike,
    output_dir: Union[str, Path],
    sample_name: str = "sample",
    include_doppler: bool = True,
    sampling_rate: float = 1.0,
    motion_band: Band = None,
    antenna_idx: int = 0,
    n_fft: int = 64,
    hop_length: int = 32,
    pipeline_name: Optional[str] = None,
    pipeline_config: Optional[Dict[str, object]] = None,
) -> Dict[str, Union[str, Dict[str, float]]]:
    """Write a diagnostic PNG, metrics CSV, and manifest for one CSI sample."""
    import matplotlib.pyplot as plt

    raw_arr, processed_arr = _as_valid_pair(raw, processed)
    sample_name = _safe_sample_name(sample_name)
    metrics = compute_pipeline_metrics(raw_arr, processed_arr, sampling_rate=sampling_rate, motion_band=motion_band)

    output_path = Path(output_dir)
    diagnostic_path = output_path / f"{sample_name}_diagnostic.png"
    metrics_path = output_path / "metrics.csv"
    manifest_path = output_path / "manifest.json"

    output_path.mkdir(parents=True, exist_ok=True)
    fig = plot_pipeline_diagnostics(
        raw_arr,
        processed_arr,
        diagnostic_path,
        include_doppler=include_doppler,
        antenna_idx=antenna_idx,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    plt.close(fig)

    row = {"sample": sample_name, **metrics}
    write_header = not metrics_path.exists()
    with metrics_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    sample_manifest = {
        "sample_name": sample_name,
        "include_doppler": include_doppler,
        "raw_shape": list(raw_arr.shape),
        "processed_shape": list(processed_arr.shape),
        "sampling_rate": sampling_rate,
        "motion_band": list(motion_band) if motion_band is not None else None,
        "antenna_idx": antenna_idx,
        "n_fft": n_fft,
        "hop_length": hop_length,
        "pipeline_name": pipeline_name,
        "pipeline_config": pipeline_config,
        "diagnostic_path": str(diagnostic_path),
        "metrics_path": str(metrics_path),
    }
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if not isinstance(manifest, dict) or not isinstance(manifest.get("samples"), list):
            manifest = {"samples": []}
    else:
        manifest = {"samples": []}
    manifest["samples"].append(sample_manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))

    return {
        "diagnostic_path": str(diagnostic_path),
        "metrics_path": str(metrics_path),
        "manifest_path": str(manifest_path),
        "metrics": metrics,
    }


__all__ = [
    "compute_pipeline_metrics",
    "plot_pipeline_diagnostics",
    "run_pipeline_diagnostics",
]
