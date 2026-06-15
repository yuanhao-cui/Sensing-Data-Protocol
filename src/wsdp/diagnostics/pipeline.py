"""Offline diagnostics for comparing raw and processed CSI arrays."""
import csv
import json
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

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


def _as_valid_csi(csi: ArrayLike) -> np.ndarray:
    arr = np.asarray(csi)
    if arr.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D CSI arrays, got shape {arr.shape}")
    if arr.size == 0 or arr.shape[0] < 2:
        raise ValueError("CSI arrays must contain at least two time steps")
    return arr


def _as_diagnosable_csi(name: str, csi: ArrayLike) -> np.ndarray:
    arr = _as_valid_csi(csi)
    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"{name} CSI must be numeric")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} CSI must contain only finite values")
    return arr


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
    with np.errstate(over="ignore", invalid="ignore"):
        return float(np.mean(np.abs(dynamic) ** 2))


def _phase_stability_score(metrics: Dict[str, float]) -> float:
    return (
        metrics["common_phase_std"]
        + metrics["linear_phase_slope_std"]
        + 0.25 * metrics["phase_fit_residual_rms"]
    )


def _circular_std(angles: Sequence[float]) -> float:
    values = np.asarray(angles, dtype=float)
    resultant = float(np.abs(np.mean(np.exp(1j * values))))
    resultant = float(np.clip(resultant, 1e-12, 1.0))
    return float(np.sqrt(-2.0 * np.log(resultant)))


def _amplitude_outlier_diagnostics(csi: np.ndarray, threshold: float = 6.0) -> Dict[str, float]:
    amplitude = np.abs(csi)
    median = np.median(amplitude, axis=0, keepdims=True)
    mad = np.median(np.abs(amplitude - median), axis=0, keepdims=True)
    robust_z = np.abs(amplitude - median) / (1.4826 * mad + 1e-12)
    return {
        "amplitude_outlier_fraction": float(np.mean(robust_z > threshold)),
        "amplitude_outlier_median_z": float(np.median(robust_z)),
    }


def _ensure_finite_metrics(name: str, metrics: Dict[str, float]) -> Dict[str, float]:
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.number)) and not np.isfinite(float(value)):
            raise ValueError(f"{name}.{key} must be finite, got {value}")
    return metrics


def _validate_json_finite(name: str, value):
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, np.generic):
        value = value.item()
        return _validate_json_finite(name, value)
    if isinstance(value, Mapping):
        validated = {}
        for key, nested in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{name} keys must be strings, got {type(key).__name__}")
            validated[key] = _validate_json_finite(f"{name}.{key}", nested)
        return validated
    if isinstance(value, (list, tuple)):
        return [
            _validate_json_finite(f"{name}[{idx}]", nested)
            for idx, nested in enumerate(value)
        ]
    if isinstance(value, complex):
        raise ValueError(f"{name} must be JSON-compatible, got {type(value).__name__}")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value}")
        return value
    raise ValueError(f"{name} must be JSON-compatible, got {type(value).__name__}")


def _validate_bool_param(name: str, value: bool) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool, got {value}")
    return value


def _validate_optional_string(name: str, value: Optional[str]) -> Optional[str]:
    if value is None or isinstance(value, str):
        return value
    raise ValueError(f"{name} must be a string or None, got {type(value).__name__}")


def _validate_sampling_rate(sampling_rate: float) -> float:
    try:
        value = float(sampling_rate)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"sampling_rate must be numeric, got {sampling_rate}") from exc
    if not np.isfinite(value):
        raise ValueError(f"sampling_rate must be finite, got {sampling_rate}")
    if value <= 0:
        raise ValueError(f"sampling_rate must be positive, got {sampling_rate}")
    return value


def _validate_motion_band(motion_band: Band, sampling_rate: Optional[float] = None) -> Band:
    if motion_band is None:
        return None
    try:
        low, high = motion_band
        low = float(low)
        high = float(high)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"motion_band must be (low, high) numeric values, got {motion_band}"
        ) from exc
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError(f"motion_band values must be finite, got {motion_band}")
    if low < 0 or high <= low:
        raise ValueError(f"motion_band must be (low, high) with high > low, got {motion_band}")
    if sampling_rate is not None:
        nyquist = _validate_sampling_rate(sampling_rate) / 2.0
        if high > nyquist:
            raise ValueError(
                f"motion_band high must not exceed Nyquist frequency {nyquist}, got {high}"
            )
    return (low, high)


def _validate_integer_param(name: str, value: int, min_value: int) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric, got {value}") from exc
    if not np.isfinite(numeric):
        raise ValueError(f"{name} must be finite, got {value}")
    if not numeric.is_integer():
        raise ValueError(f"{name} must be an integer, got {value}")
    integer = int(numeric)
    if integer < min_value:
        raise ValueError(f"{name} must be at least {min_value}, got {value}")
    return integer


def _validate_antenna_idx(csi: np.ndarray, antenna_idx: int) -> int:
    idx = _validate_integer_param("antenna_idx", antenna_idx, min_value=0)
    if csi.ndim == 2 and idx != 0:
        raise ValueError(f"antenna_idx {idx} out of range for 2D CSI")
    if csi.ndim == 3 and idx >= csi.shape[2]:
        raise ValueError(f"antenna_idx {idx} out of range for shape {csi.shape}")
    return idx


def _validate_subcarrier_indices(
    subcarrier_indices: Optional[Sequence[float]],
    subcarrier_count: int,
) -> np.ndarray:
    if subcarrier_indices is None:
        return np.arange(subcarrier_count, dtype=float)

    indices = np.asarray(subcarrier_indices, dtype=float)
    if indices.ndim != 1:
        raise ValueError("subcarrier_indices must be a 1D array")
    if indices.shape != (subcarrier_count,):
        raise ValueError(
            f"subcarrier_indices length ({len(indices)}) != number of "
            f"subcarriers ({subcarrier_count})"
        )
    if not np.all(np.isfinite(indices)):
        raise ValueError("subcarrier_indices must contain only finite values")
    if np.unique(indices).size != indices.size:
        raise ValueError("subcarrier_indices must contain distinct values")
    return indices


def _power_spectrum(csi: np.ndarray, sampling_rate: float) -> Tuple[np.ndarray, np.ndarray]:
    sampling_rate = _validate_sampling_rate(sampling_rate)

    dynamic = _dynamic_signal(csi)
    spectrum = np.fft.fft(dynamic, axis=0)
    with np.errstate(over="ignore", invalid="ignore"):
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
        low, high = _validate_motion_band(motion_band, sampling_rate=sampling_rate)
        band = non_dc & (np.abs(freqs) >= low) & (np.abs(freqs) <= high)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return float(np.sum(power[band]) / total)


def _spectral_concentration(csi: np.ndarray, sampling_rate: float) -> float:
    freqs, power = _power_spectrum(csi, sampling_rate)
    non_dc_power = power[np.abs(freqs) > 1e-12]
    total = float(np.sum(non_dc_power))
    if total <= 1e-12:
        return 0.0
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return float(np.max(non_dc_power) / total)


def compute_pipeline_metrics(
    raw: ArrayLike,
    processed: ArrayLike,
    sampling_rate: float = 1.0,
    motion_band: Band = None,
) -> Dict[str, float]:
    """Compute CPU-only proxy metrics for raw/processed CSI comparison."""
    raw_arr, processed_arr = _as_valid_pair(raw, processed)
    sampling_rate = _validate_sampling_rate(sampling_rate)
    motion_band = _validate_motion_band(motion_band, sampling_rate=sampling_rate)

    raw_dynamic = _dynamic_energy(raw_arr)
    processed_dynamic = _dynamic_energy(processed_arr)
    raw_motion_ratio = _motion_band_ratio(raw_arr, sampling_rate, motion_band)
    processed_motion_ratio = _motion_band_ratio(processed_arr, sampling_rate, motion_band)

    metrics = {
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
    return _ensure_finite_metrics("pipeline_metrics", metrics)


def compute_phase_diagnostics(
    csi: ArrayLike,
    antenna_idx: int = 0,
    subcarrier_indices: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    """Measure packet-to-packet phase stability from linear subcarrier fits."""
    arr = _as_diagnosable_csi("csi", csi)
    view = _antenna_slice(arr, antenna_idx)
    subcarrier_count = view.shape[1]
    if subcarrier_count < 2:
        raise ValueError("phase diagnostics require at least two subcarriers")

    x = _validate_subcarrier_indices(subcarrier_indices, subcarrier_count)

    slopes = []
    intercepts = []
    residuals = []
    for packet in view:
        phase = np.unwrap(np.angle(packet))
        slope, intercept = np.polyfit(x, phase, 1)
        fitted = slope * x + intercept
        slopes.append(slope)
        intercepts.append(intercept)
        residuals.append(float(np.sqrt(np.mean((phase - fitted) ** 2))))

    metrics = {
        "common_phase_std": _circular_std(intercepts),
        "linear_phase_slope_std": float(np.std(slopes)),
        "phase_fit_residual_rms": float(np.mean(residuals)),
    }
    return _ensure_finite_metrics("phase_diagnostics", metrics)


def _antenna_slice(csi: np.ndarray, antenna_idx: int) -> np.ndarray:
    antenna_idx = _validate_antenna_idx(csi, antenna_idx)
    if csi.ndim == 2:
        return csi
    return csi[:, :, antenna_idx]


def _plot_heatmap(
    ax,
    data: np.ndarray,
    title: str,
    cmap: str = "viridis",
    center: bool = False,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
    arr = np.asarray(data, dtype=float)
    # Replace non-finite values so imshow always renders a visible image
    # even when the upstream pipeline emits NaN/Inf.
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    kwargs = {"aspect": "auto", "origin": "lower", "cmap": cmap}
    if center:
        centered_vmax = float(np.max(np.abs(arr))) + 1e-12
        kwargs.update({"vmin": -centered_vmax, "vmax": centered_vmax})
    elif vmin is not None and vmax is not None:
        kwargs.update({"vmin": vmin, "vmax": vmax})
    else:
        data_min = float(np.min(arr))
        data_max = float(np.max(arr))
        if np.isclose(data_min, data_max):
            eps = max(abs(data_min), 1.0) * 1e-12
            data_min -= eps
            data_max += eps
        kwargs.update({"vmin": data_min, "vmax": data_max})

    im = ax.imshow(arr.T, **kwargs)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Subcarrier")
    return im


def _frame_starts(length: int, window: int, hop_length: int) -> np.ndarray:
    hop_length = _validate_integer_param("hop_length", hop_length, min_value=1)
    if length <= window:
        return np.array([0], dtype=int)
    return np.arange(0, length - window + 1, hop_length, dtype=int)


def _doppler_summary(csi: np.ndarray, n_fft: int, hop_length: int) -> np.ndarray:
    n_fft = _validate_integer_param("n_fft", n_fft, min_value=2)
    hop_length = _validate_integer_param("hop_length", hop_length, min_value=1)

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
    n_fft = _validate_integer_param("n_fft", n_fft, min_value=2)
    hop_length = _validate_integer_param("hop_length", hop_length, min_value=1)
    antenna_idx = _validate_antenna_idx(raw_arr, antenna_idx)
    raw_view = _antenna_slice(raw_arr, antenna_idx)
    processed_view = _antenna_slice(processed_arr, antenna_idx)
    raw_amplitude = np.abs(raw_view)
    processed_amplitude = np.abs(processed_view)
    # Use independent color limits per panel. If the processed CSI has been
    # normalized (z-score, min-max, etc.) its amplitude range can be orders of
    # magnitude smaller than the raw CSI; sharing a single scale would make the
    # processed panel appear uniformly black.

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
        (raw_amplitude, "Raw Amplitude", "viridis", False, None, None),
        (processed_amplitude, "Processed Amplitude", "viridis", False, None, None),
        (processed_amplitude - raw_amplitude, "Amplitude Difference", "RdBu_r", True, None, None),
        (np.angle(processed_view * np.conj(raw_view)), "Phase Difference", "twilight", False, None, None),
    ]

    for ax, (data, title, cmap, center, vmin, vmax) in zip(axes, panels):
        im = _plot_heatmap(ax, data, title, cmap=cmap, center=center, vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax)

    if include_doppler:
        raw_doppler = _doppler_summary(raw_arr, n_fft=n_fft, hop_length=hop_length)
        processed_doppler = _doppler_summary(processed_arr, n_fft=n_fft, hop_length=hop_length)
        for ax, data, title in (
            (axes[4], raw_doppler, "Raw Doppler Summary"),
            (axes[5], processed_doppler, "Processed Doppler Summary"),
        ):
            arr = np.asarray(data, dtype=float)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            data_min = float(np.min(arr))
            data_max = float(np.max(arr))
            if np.isclose(data_min, data_max):
                eps = max(abs(data_min), 1.0) * 1e-12
                data_min -= eps
                data_max += eps
            im = ax.imshow(
                arr,
                aspect="auto",
                origin="lower",
                cmap="magma",
                vmin=data_min,
                vmax=data_max,
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
    include_doppler = _validate_bool_param("include_doppler", include_doppler)
    sampling_rate = _validate_sampling_rate(sampling_rate)
    motion_band = _validate_motion_band(motion_band, sampling_rate=sampling_rate)
    n_fft = _validate_integer_param("n_fft", n_fft, min_value=2)
    hop_length = _validate_integer_param("hop_length", hop_length, min_value=1)
    antenna_idx = _validate_antenna_idx(raw_arr, antenna_idx)
    pipeline_name = _validate_optional_string("pipeline_name", pipeline_name)
    pipeline_config = _validate_json_finite("pipeline_config", pipeline_config)
    metrics = compute_pipeline_metrics(raw_arr, processed_arr, sampling_rate=sampling_rate, motion_band=motion_band)

    output_path = Path(output_dir)
    diagnostic_path = output_path / f"{sample_name}_diagnostic.png"
    metrics_path = output_path / "metrics.csv"
    manifest_path = output_path / "manifest.json"

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
    manifest_json = json.dumps(manifest, indent=2, allow_nan=False)

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

    manifest_path.write_text(manifest_json)

    return {
        "diagnostic_path": str(diagnostic_path),
        "metrics_path": str(metrics_path),
        "manifest_path": str(manifest_path),
        "metrics": metrics,
    }


def compare_pipeline_candidates(
    raw: ArrayLike,
    candidates: Mapping[str, ArrayLike],
    output_dir: Union[str, Path],
    include_doppler: bool = True,
    sampling_rate: float = 1.0,
    motion_band: Band = None,
    antenna_idx: int = 0,
    n_fft: int = 64,
    hop_length: int = 32,
) -> Dict[str, object]:
    """Compare candidate CSI processing orders with shared diagnostics."""
    raw_arr = _as_diagnosable_csi("raw", raw)
    if not isinstance(candidates, Mapping):
        raise ValueError("candidates must be a mapping of names to CSI arrays")
    include_doppler = _validate_bool_param("include_doppler", include_doppler)
    sampling_rate = _validate_sampling_rate(sampling_rate)
    motion_band = _validate_motion_band(motion_band, sampling_rate=sampling_rate)
    n_fft = _validate_integer_param("n_fft", n_fft, min_value=2)
    hop_length = _validate_integer_param("hop_length", hop_length, min_value=1)
    antenna_idx = _validate_antenna_idx(raw_arr, antenna_idx)
    if len(candidates) < 2:
        raise ValueError("candidates must contain at least two named CSI arrays")

    validated_candidates = []
    for candidate_name, candidate in candidates.items():
        safe_name = _safe_sample_name(candidate_name)
        _, candidate_arr = _as_valid_pair(raw_arr, candidate)
        candidate_arr = _as_diagnosable_csi(safe_name, candidate_arr)
        validated_candidates.append((safe_name, candidate_arr))

    raw_phase = compute_phase_diagnostics(raw_arr, antenna_idx=antenna_idx)
    raw_phase_score = _phase_stability_score(raw_phase)
    raw_amplitude = _amplitude_outlier_diagnostics(raw_arr)
    raw_outlier_fraction = raw_amplitude["amplitude_outlier_fraction"]

    output_path = Path(output_dir)
    comparison_path = output_path / "candidate_comparison.csv"
    recommendation_path = output_path / "recommendation.json"

    rows = []
    diagnostic_paths = {
        safe_name: str(output_path / f"{safe_name}_diagnostic.png")
        for safe_name, _candidate_arr in validated_candidates
    }
    for safe_name, candidate_arr in validated_candidates:
        metrics = compute_pipeline_metrics(
            raw_arr,
            candidate_arr,
            sampling_rate=sampling_rate,
            motion_band=motion_band,
        )
        phase = compute_phase_diagnostics(candidate_arr, antenna_idx=antenna_idx)
        amplitude = _amplitude_outlier_diagnostics(candidate_arr)

        phase_score = _phase_stability_score(phase)
        phase_improvement = (raw_phase_score - phase_score) / (raw_phase_score + 1e-12)
        amplitude_improvement = (
            (raw_outlier_fraction - amplitude["amplitude_outlier_fraction"])
            / (raw_outlier_fraction + 1e-12)
        )
        signal_ratio = max(metrics["signal_preservation_ratio"], 1e-12)
        preservation_penalty = abs(float(np.log(signal_ratio)))
        quality_score = (
            phase_improvement
            + amplitude_improvement
            + metrics["motion_band_energy_ratio_delta"]
            - 0.15 * preservation_penalty
        )
        _ensure_finite_metrics(
            safe_name,
            {
                **metrics,
                **phase,
                **amplitude,
                "phase_stability_score": phase_score,
                "phase_stability_improvement": phase_improvement,
                "amplitude_outlier_improvement": amplitude_improvement,
                "quality_score": quality_score,
            },
        )

        rows.append(
            {
                "candidate": safe_name,
                **metrics,
                **phase,
                **amplitude,
                "phase_stability_score": phase_score,
                "phase_stability_improvement": phase_improvement,
                "amplitude_outlier_improvement": amplitude_improvement,
                "quality_score": quality_score,
                "diagnostic_path": diagnostic_paths[safe_name],
            }
        )

    rows.sort(key=lambda row: row["quality_score"], reverse=True)
    best = rows[0]
    runner_up = rows[1]
    margin = best["quality_score"] - runner_up["quality_score"]

    if margin < 0.1 or best["quality_score"] < 0.05:
        recommended_candidate = None
        confidence = "inconclusive"
        reasons = ["candidate scores are too close for a reliable recommendation"]
    else:
        recommended_candidate = best["candidate"]
        confidence = "high" if margin >= 0.5 else "medium"
        reasons = []
        if best["phase_stability_improvement"] > 0.2:
            reasons.append(
                f"phase stability improved by {best['phase_stability_improvement']:.3f}"
            )
        if best["amplitude_outlier_improvement"] > 0.2:
            reasons.append(
                "amplitude outlier fraction improved by "
                f"{best['amplitude_outlier_improvement']:.3f}"
            )
        if best["motion_band_energy_ratio_delta"] > 0.05:
            reasons.append(
                "motion-band energy ratio increased by "
                f"{best['motion_band_energy_ratio_delta']:.3f}"
            )
        if best["signal_preservation_ratio"] < 0.2:
            reasons.append(
                "signal preservation is low; inspect diagnostic plots before using this order"
            )
        if not reasons:
            reasons.append("best candidate has the highest combined diagnostic score")

    recommendation = {
        "recommended_candidate": recommended_candidate,
        "confidence": confidence,
        "quality_margin": margin,
        "reasons": reasons,
        "comparison_csv": str(comparison_path),
        "diagnostic_paths": diagnostic_paths,
    }
    recommendation_json = json.dumps(recommendation, indent=2, allow_nan=False)

    output_path.mkdir(parents=True, exist_ok=True)
    for safe_name, candidate_arr in validated_candidates:
        run_pipeline_diagnostics(
            raw_arr,
            candidate_arr,
            output_path,
            sample_name=safe_name,
            include_doppler=include_doppler,
            sampling_rate=sampling_rate,
            motion_band=motion_band,
            antenna_idx=antenna_idx,
            n_fft=n_fft,
            hop_length=hop_length,
            pipeline_name=safe_name,
        )

    fieldnames = list(rows[0].keys())
    with comparison_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    recommendation_path.write_text(recommendation_json)

    return {
        **recommendation,
        "recommendation_json": str(recommendation_path),
        "comparison_rows": rows,
    }


__all__ = [
    "compare_pipeline_candidates",
    "compute_pipeline_metrics",
    "compute_phase_diagnostics",
    "plot_pipeline_diagnostics",
    "run_pipeline_diagnostics",
]
