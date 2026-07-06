"""
Amplitude processing algorithms for CSI data.

Provides normalization and outlier removal for CSI amplitude values.
"""
import numpy as np


def normalize_amplitude(csi, method='z-score', return_phase_channels=False):
    """
    Normalize CSI amplitude along the time axis.

    Args:
        csi: CSI array of shape (T, F, A) or (T, F) — complex or real
        method: Normalization method
            - 'z-score': Zero mean, unit variance per subcarrier
            - 'min-max': Scale to [0, 1] per subcarrier
        return_phase_channels: If True, return real-valued
            ``[normalized_amplitude, phase]`` channels instead of rebuilding
            a complex array. This preserves negative z-score amplitudes
            without shifting phase by pi.

    Returns:
        np.ndarray: Normalized CSI with the same shape by default. When
        ``return_phase_channels`` is True, the last dimension is doubled and
        the returned array is real-valued.

    Reference:
        Standard statistical normalization. For CSI-specific usage:
        Ma Y, et al. "PhaseFi: Phase Fingerprinting for Indoor Localization
        with a Deep Learning Approach." IEEE GLOBECOM, 2015.
    """
    if csi.size == 0:
        if return_phase_channels:
            if not np.iscomplexobj(csi):
                raise ValueError(
                    "return_phase_channels=True requires complex CSI data"
                )
            return np.concatenate([np.abs(csi), np.angle(csi)], axis=-1)
        return csi.copy()
    if csi.ndim < 2:
        raise ValueError(f"Expected at least 2D array, got shape {csi.shape}")

    valid_methods = ('z-score', 'min-max')
    if method not in valid_methods:
        raise ValueError(f"Unknown method '{method}'. Supported: {valid_methods}")

    result = np.empty_like(csi)
    amplitude = np.abs(csi)

    # Normalize along time axis (axis=0) per subcarrier
    if method == 'z-score':
        mean = np.mean(amplitude, axis=0, keepdims=True)
        std = np.std(amplitude, axis=0, keepdims=True)
        std = np.where(std < 1e-10, 1.0, std)  # avoid division by zero
        norm_amp = (amplitude - mean) / std
    else:  # min-max
        amin = np.min(amplitude, axis=0, keepdims=True)
        amax = np.max(amplitude, axis=0, keepdims=True)
        range_val = amax - amin
        range_val = np.where(range_val < 1e-10, 1.0, range_val)
        norm_amp = (amplitude - amin) / range_val

    if return_phase_channels:
        if not np.iscomplexobj(csi):
            raise ValueError(
                "return_phase_channels=True requires complex CSI data"
            )
        phase = np.angle(csi)
        return np.concatenate([norm_amp, phase], axis=-1)

    if np.iscomplexobj(csi):
        phase = np.angle(csi)
        result = norm_amp * np.exp(1j * phase)
    else:
        result = norm_amp

    return result


def remove_outliers(csi, method='iqr', factor=1.5):
    """
    Remove or clip outlier amplitudes in CSI data.

    Detects outliers per subcarrier along the time axis and clips them
    to the boundary values. This preserves data shape while mitigating
    the effect of anomalous measurements.

    Args:
        csi: CSI array of shape (T, F, A) or (T, F) — complex or real
        method: Outlier detection method
            - 'iqr': Interquartile Range method
            - 'z-score': Standard deviation based method
        factor: Multiplier for the detection threshold (default: 1.5)
            For IQR: outliers are outside Q1 - factor*IQR to Q3 + factor*IQR
            For z-score: outliers are beyond mean ± factor*std

    Returns:
        np.ndarray: CSI with outliers clipped, same shape and dtype

    Reference:
        Tukey JW. "Exploratory Data Analysis." Addison-Wesley, 1977.
        (IQR method definition)
    """
    if csi.size == 0:
        return csi.copy()
    if csi.ndim < 2:
        raise ValueError(f"Expected at least 2D array, got shape {csi.shape}")

    valid_methods = ('iqr', 'z-score')
    if method not in valid_methods:
        raise ValueError(f"Unknown method '{method}'. Supported: {valid_methods}")
    if factor <= 0:
        raise ValueError(f"factor must be > 0, got {factor}")

    result = csi.copy()
    amplitude = np.abs(csi)

    if method == 'iqr':
        q1 = np.percentile(amplitude, 25, axis=0, keepdims=True)
        q3 = np.percentile(amplitude, 75, axis=0, keepdims=True)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
    else:  # z-score
        mean = np.mean(amplitude, axis=0, keepdims=True)
        std = np.std(amplitude, axis=0, keepdims=True)
        lower = mean - factor * std
        upper = mean + factor * std

    # Clip amplitudes to [lower, upper]
    clipped_amp = np.clip(amplitude, lower, upper)

    if np.iscomplexobj(csi):
        phase = np.angle(csi)
        result = clipped_amp * np.exp(1j * phase)
    else:
        result = clipped_amp

    return result


def agc_compensate(csi, agc_values):
    """
    Compensate for Automatic Gain Control (AGC) in Intel IWL5300 CSI.

    The IWL5300 firmware applies AGC before reporting CSI, attenuating
    strong signals and amplifying weak ones. The true channel amplitude
    is recovered by:

        |H_true| = |H_reported| * 10^(AGC / 20)

    This scales amplitude by the inverse of the AGC gain applied at
    each frame. Phase is preserved unchanged.

    Args:
        csi: CSI array of shape (T, F, A) or (T, F) — complex or real
        agc_values: 1D array of shape (T,) — per-frame AGC gain values
            in dB, as reported by BfeeFrame.agc

    Returns:
        np.ndarray: AGC-compensated CSI, same shape and dtype as input

    Reference:
        Zheng Y, et al. "Optimal Preprocessing of WiFi CSI for Human
        Activity Recognition." arXiv:2307.12126, 2023.
    """
    if csi.size == 0:
        return csi.copy()
    if csi.ndim < 2:
        raise ValueError(f"Expected at least 2D array, got shape {csi.shape}")

    agc_values = np.asarray(agc_values, dtype=np.float64)
    if agc_values.ndim != 1:
        raise ValueError(
            f"agc_values must be 1D array, got shape {agc_values.shape}"
        )
    if agc_values.shape[0] != csi.shape[0]:
        raise ValueError(
            f"agc_values length ({agc_values.shape[0]}) must match "
            f"CSI time dimension ({csi.shape[0]})"
        )

    # Compensation factor: 10^(AGC/20), broadcast over subcarrier/antenna dims
    scale = 10.0 ** (agc_values / 20.0)
    # Reshape for broadcasting: (T,) -> (T, 1) or (T, 1, 1)
    for _ in range(csi.ndim - 1):
        scale = scale[..., np.newaxis]

    if np.iscomplexobj(csi):
        amplitude = np.abs(csi) * scale
        phase = np.angle(csi)
        return (amplitude * np.exp(1j * phase)).astype(csi.dtype)
    else:
        return (csi * scale).astype(csi.dtype)


def hampel_filter(csi, window_size=5, n_sigma=3.0):
    """
    Hampel filter for robust impulse noise removal.

    Uses a sliding window median and Median Absolute Deviation (MAD)
    to detect and replace outlier samples:

        if |x_i - median(window)| > n_sigma * 1.4826 * MAD(window):
            x_i = median(window)

    The constant 1.4826 is the consistency factor relating MAD to
    standard deviation under a Gaussian distribution. Applied
    independently to each subcarrier-antenna time series.

    For complex CSI, real and imaginary parts are filtered separately.

    Args:
        csi: CSI array of shape (T, F, A) or (T, F) — complex or real
        window_size: Half-window size in samples (default: 5).
            Full window = 2 * window_size + 1.
        n_sigma: Threshold multiplier for outlier detection (default: 3.0)

    Returns:
        np.ndarray: Filtered CSI, same shape and dtype as input

    Reference:
        Pearson RK, et al. "Generalized Hampel Filters."
        EURASIP J. Adv. Signal Processing, 2016.
    """
    if csi.size == 0:
        return csi.copy()
    if csi.ndim < 2:
        raise ValueError(f"Expected at least 2D array, got shape {csi.shape}")
    if window_size < 1:
        raise ValueError(f"window_size must be >= 1, got {window_size}")
    if n_sigma <= 0:
        raise ValueError(f"n_sigma must be > 0, got {n_sigma}")

    T = csi.shape[0]
    if T < 3:
        return csi.copy()

    def _hampel_1d(x):
        """Apply Hampel filter to a single 1D real signal."""
        n = len(x)
        y = x.copy()
        for i in range(n):
            lo = max(0, i - window_size)
            hi = min(n, i + window_size + 1)
            window = x[lo:hi]
            med = np.median(window)
            mad = np.median(np.abs(window - med))
            threshold = n_sigma * 1.4826 * mad
            if np.abs(x[i] - med) > threshold:
                y[i] = med
        return y

    result = np.empty_like(csi)

    # Flatten trailing dims for iteration
    if csi.ndim == 2:
        for f in range(csi.shape[1]):
            if np.iscomplexobj(csi):
                result[:, f] = _hampel_1d(np.real(csi[:, f])) + \
                               1j * _hampel_1d(np.imag(csi[:, f]))
            else:
                result[:, f] = _hampel_1d(csi[:, f])
    elif csi.ndim == 3:
        for f in range(csi.shape[1]):
            for a in range(csi.shape[2]):
                if np.iscomplexobj(csi):
                    result[:, f, a] = _hampel_1d(np.real(csi[:, f, a])) + \
                                      1j * _hampel_1d(np.imag(csi[:, f, a]))
                else:
                    result[:, f, a] = _hampel_1d(csi[:, f, a])
    else:
        raise ValueError(f"Expected 2D or 3D array, got shape {csi.shape}")

    return result
