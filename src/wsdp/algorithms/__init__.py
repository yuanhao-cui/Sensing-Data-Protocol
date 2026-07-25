"""
WSDP Algorithm Library — Unified Processing Pipeline

Provides denoising, phase calibration, amplitude normalization,
interpolation, feature extraction, and activity detection for
WiFi Channel State Information (CSI) data.

## Unified API

    denoise(csi, method='wavelet', **kwargs)
    calibrate(csi, method='linear', **kwargs)
    normalize(csi, method='z-score', **kwargs)
    interpolate(csi, target_K=30, method='cubic', **kwargs)
    extract_features(csi, features=['doppler'], **kwargs)

## Pluggable Architecture

    # List available algorithms
    list_algorithms('denoise')

    # Register custom algorithm
    register_algorithm('denoise', 'my_method', my_func)

    # Use config files
    config = load_config('examples/configs/algorithms_config.yaml')

    # Apply presets
    steps = apply_preset('high_quality')
    result = execute_pipeline(csi, steps)

## Backward-compatible exports

    wavelet_denoise_csi(csi_tensor)
    phase_calibration(csi_data)
"""
# Backward-compatible imports
from .denoising import wavelet_denoise_csi
from .phase_calibration import phase_calibration

# New algorithm imports
from .denoising_butterworth import butterworth_denoise, savgol_denoise
from .phase import polynomial_calibration, stc_calibration, robust_phase_sanitization
from .amplitude import normalize_amplitude, remove_outliers
from .interpolation import interpolate_grid
from .features import doppler_spectrum, entropy_features, csi_ratio, tensor_decomposition
from .detection import detect_activity, change_point_detection
from .visualization import (
    plot_csi_heatmap,
    plot_denoising_comparison,
    plot_phase_calibration,
)

# Registry imports
from .registry import (
    register_algorithm,
    unregister_algorithm,
    get_algorithm,
    list_algorithms,
    is_registered,
    algorithm_info,
    load_config,
    save_config,
    apply_preset,
    register_preset,
    list_presets,
    PRESETS,
)

# Modular pipeline imports
from .pipeline import (
    AlgorithmStep,
    build_steps_from_config,
    execute_algorithm_steps,
    execute_pipeline,
)
from .adapter import FunctionAlgorithm


# ============================================================================
# Unified API
# ============================================================================

def denoise(csi, method='wavelet', **kwargs):
    """
    Unified denoising interface.

    Args:
        csi: CSI array of shape (T, F, A) or (T, F)
        method: Denoising method
            - 'wavelet': Wavelet shrinkage (VisuShrink with db4)
            - 'butterworth': Butterworth low-pass filter
            - 'savgol': Savitzky-Golay polynomial smoothing
            - Or any custom method registered via register_algorithm()
        **kwargs: Method-specific parameters

    Returns:
        np.ndarray: Denoised CSI with same shape

    Examples:
        >>> denoise(csi, method='wavelet')
        >>> denoise(csi, method='butterworth', order=4, cutoff=0.2)
        >>> denoise(csi, method='savgol', window_length=15, polyorder=4)

        >>> # With custom method
        >>> register_algorithm('denoise', 'my_method', my_func)
        >>> denoise(csi, method='my_method', my_param=42)
    """
    func = get_algorithm('denoise', method)
    return func(csi, **kwargs)


def calibrate(csi, method='linear', **kwargs):
    """
    Unified phase calibration interface.

    Args:
        csi: CSI array of shape (T, F, A) — must be complex
        method: Calibration method
            - 'linear': Standard linear phase calibration
            - 'polynomial': Polynomial phase calibration
            - 'stc': Sanitize-then-Calibrate (IEEE TWC 2019)
            - 'robust': Robust phase sanitization (FIMD/MobiCom)
            - Or any custom method registered via register_algorithm()
        **kwargs: Method-specific parameters

    Returns:
        np.ndarray: Phase-calibrated CSI with same shape

    Examples:
        >>> calibrate(csi, method='linear')
        >>> calibrate(csi, method='polynomial', degree=3)
        >>> calibrate(csi, method='stc')
        >>> calibrate(csi, method='robust')
    """
    func = get_algorithm('calibrate', method)
    return func(csi, **kwargs)


def normalize(csi, method='z-score', **kwargs):
    """
    Unified amplitude normalization interface.

    Args:
        csi: CSI array of shape (T, F, A) or (T, F)
        method: Normalization method
            - 'z-score': Zero mean, unit variance per subcarrier
            - 'min-max': Scale to [0, 1] per subcarrier
            - Or any custom method registered via register_algorithm()
        **kwargs: Method-specific parameters

    Returns:
        np.ndarray: Normalized CSI with same shape

    Examples:
        >>> normalize(csi, method='z-score')
        >>> normalize(csi, method='min-max')
    """
    func = get_algorithm('normalize', method)
    return func(csi, method=method, **kwargs)


def interpolate(csi, target_K=30, method='cubic', **kwargs):
    """
    Unified frequency grid interpolation interface.

    Args:
        csi: CSI array of shape (T, F, A)
        target_K: Target number of subcarriers (default: 30)
        method: Interpolation method
            - 'linear': Piecewise linear
            - 'cubic': Cubic spline
            - 'nearest': Nearest-neighbor
        **kwargs: Additional parameters

    Returns:
        np.ndarray: Interpolated CSI with shape (T, target_K, A)

    Examples:
        >>> interpolate(csi, target_K=64, method='cubic')
        >>> interpolate(csi, target_K=15, method='linear')
    """
    func = get_algorithm('interpolate', method)
    return func(csi, target_K=target_K, method=method, **kwargs)


def extract_features(csi, features=None, **kwargs):
    """
    Unified feature extraction interface.

    Extracts one or more features from CSI data and returns them
    in a dictionary.

    Args:
        csi: CSI array of shape (T, F, A) — complex
        features: List of features to extract. Options:
            - 'doppler': Doppler spectrum (STFT)
            - 'entropy': Information entropy per subcarrier
            - 'ratio': CSI antenna ratio
            - 'decomposition': Tensor decomposition
        **kwargs: Feature-specific parameters (passed through)

    Returns:
        dict: Dictionary mapping feature names to extracted values

    Examples:
        >>> extract_features(csi, features=['doppler', 'entropy'])
        >>> extract_features(csi, features=['ratio'], antenna_pairs=[(0, 1)])
        >>> extract_features(csi, features=['decomposition'], rank=5, method='cp')
    """
    if features is None:
        features = ['doppler']

    valid_features = ('doppler', 'entropy', 'ratio', 'decomposition')
    for f in features:
        if f not in valid_features:
            raise ValueError(f"Unknown feature '{f}'. Supported: {list(valid_features)}")

    result = {}

    for feat in features:
        if feat == 'doppler':
            n_fft = kwargs.get('n_fft', 64)
            hop_length = kwargs.get('hop_length', 32)
            result['doppler'] = doppler_spectrum(csi, n_fft=n_fft, hop_length=hop_length)
        elif feat == 'entropy':
            bins = kwargs.get('bins', 50)
            result['entropy'] = entropy_features(csi, bins=bins)
        elif feat == 'ratio':
            antenna_pairs = kwargs.get('antenna_pairs', None)
            result['ratio'] = csi_ratio(csi, antenna_pairs=antenna_pairs)
        elif feat == 'decomposition':
            rank = kwargs.get('rank', 10)
            method = kwargs.get('method', 'cp')
            result['decomposition'] = tensor_decomposition(csi, rank=rank, method=method)

    return result


__all__ = [
    # Backward-compatible
    "wavelet_denoise_csi",
    "phase_calibration",
    # New algorithms
    "butterworth_denoise",
    "savgol_denoise",
    "polynomial_calibration",
    "stc_calibration",
    "robust_phase_sanitization",
    "normalize_amplitude",
    "remove_outliers",
    "interpolate_grid",
    "doppler_spectrum",
    "entropy_features",
    "csi_ratio",
    "tensor_decomposition",
    "detect_activity",
    "change_point_detection",
    # Visualization
    "plot_csi_heatmap",
    "plot_denoising_comparison",
    "plot_phase_calibration",
    # Unified API
    "denoise",
    "calibrate",
    "normalize",
    "interpolate",
    "extract_features",
    # Registry / Pluggable Architecture
    "register_algorithm",
    "unregister_algorithm",
    "get_algorithm",
    "list_algorithms",
    "is_registered",
    "algorithm_info",
    "load_config",
    "save_config",
    "apply_preset",
    "register_preset",
    "list_presets",
    "execute_pipeline",
    "PRESETS",
    # Modular pipeline
    "AlgorithmStep",
    "build_steps_from_config",
    "execute_algorithm_steps",
    "FunctionAlgorithm",
]
