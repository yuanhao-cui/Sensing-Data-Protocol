"""
Butterworth and Savitzky-Golay denoising filters for CSI data.

Literature:
- Butterworth: Butterworth S. "Theory of filter amplifiers." Wireless Engineer, 1930.
- Savitzky-Golay: Savitzky A, Golay MJE. "Smoothing and differentiation of data by 
  simplified least squares procedures." Analytical Chemistry, 1964.
"""
import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter


def butterworth_denoise(csi, order=5, cutoff=0.3):
    """
    Butterworth low-pass filter for CSI denoising.

    Applies a zero-phase Butterworth low-pass filter along the time axis
    for each subcarrier-antenna pair, removing high-frequency noise while
    preserving the underlying signal structure.

    Args:
        csi: CSI array of shape (T, F, A) or (T, F) — complex or real
        order: Filter order (default: 5)
        cutoff: Normalized cutoff frequency in [0, 1], where 1 = Nyquist (default: 0.3)

    Returns:
        np.ndarray: Denoised CSI with same shape and dtype as input

    Reference:
        Butterworth S. "On the theory of filter amplifiers." 
        Wireless Engineer, vol. 7, pp. 536-541, 1930.
    """
    if csi.size == 0:
        return csi.copy()
    if csi.ndim < 2:
        raise ValueError(f"Expected at least 2D array, got shape {csi.shape}")
    if not (0 < cutoff <= 1.0):
        raise ValueError(f"cutoff must be in (0, 1], got {cutoff}")
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")

    T = csi.shape[0]
    if T < 3:
        return csi.copy()  # Not enough samples for filtering

    b, a = butter(order, cutoff, btype='low')
    min_len = max(3 * max(len(a), len(b)), 10)

    if csi.ndim == 2:
        # (T, F)
        result = np.empty_like(csi)
        for f in range(csi.shape[1]):
            if T <= min_len:
                result[:, f] = csi[:, f]
            elif np.iscomplexobj(csi):
                result[:, f] = filtfilt(b, a, np.real(csi[:, f])) + \
                               1j * filtfilt(b, a, np.imag(csi[:, f]))
            else:
                result[:, f] = filtfilt(b, a, csi[:, f])
    elif csi.ndim == 3:
        # (T, F, A)
        result = np.empty_like(csi)
        for f in range(csi.shape[1]):
            for a_idx in range(csi.shape[2]):
                if T <= min_len:
                    result[:, f, a_idx] = csi[:, f, a_idx]
                elif np.iscomplexobj(csi):
                    result[:, f, a_idx] = filtfilt(b, a, np.real(csi[:, f, a_idx])) + \
                                          1j * filtfilt(b, a, np.imag(csi[:, f, a_idx]))
                else:
                    result[:, f, a_idx] = filtfilt(b, a, csi[:, f, a_idx])
    else:
        raise ValueError(f"Expected 2D or 3D array, got shape {csi.shape}")

    return result


def butterworth_bandpass(csi, order=4, low_freq=0.5, high_freq=50.0, fs=1000.0):
    """
    Butterworth bandpass filter for CSI time series.

    Applies a zero-phase Butterworth bandpass filter along the time axis
    for each subcarrier-antenna pair. Used to isolate human motion
    frequency bands from static clutter and high-frequency noise.

    Typical frequency ranges for WiFi sensing:
    - Breathing: 0.1-0.5 Hz
    - Gait/walking: 0.5-3 Hz
    - Hand gestures: 1-50 Hz

    For complex CSI, real and imaginary parts are filtered separately.

    Args:
        csi: CSI array of shape (T, F, A) or (T, F) — complex or real
        order: Filter order (default: 4)
        low_freq: Lower cutoff frequency in Hz (default: 0.5)
        high_freq: Upper cutoff frequency in Hz (default: 50.0)
        fs: Sampling rate in Hz. The default 1000.0 is a fallback; pass
            the dataset/device sampling rate when it is known.

    Returns:
        np.ndarray: Bandpass-filtered CSI, same shape and dtype as input

    Reference:
        Butterworth S. "On the theory of filter amplifiers."
        Wireless Engineer, vol. 7, pp. 536-541, 1930.
    """
    if csi.size == 0:
        return csi.copy()
    if csi.ndim < 2:
        raise ValueError(f"Expected at least 2D array, got shape {csi.shape}")
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")

    nyq = fs / 2.0
    if low_freq <= 0:
        raise ValueError(f"low_freq must be > 0, got {low_freq}")
    if high_freq >= nyq:
        raise ValueError(
            f"high_freq ({high_freq}) must be < Nyquist frequency ({nyq})"
        )
    if low_freq >= high_freq:
        raise ValueError(
            f"low_freq ({low_freq}) must be < high_freq ({high_freq})"
        )

    T = csi.shape[0]
    if T < 3:
        return csi.copy()

    low = low_freq / nyq
    high = high_freq / nyq
    b, a = butter(order, [low, high], btype='band')

    # Minimum length for filtfilt: padlen = 3 * max(len(a), len(b))
    min_len = 3 * max(len(a), len(b)) + 1
    if T < min_len:
        return csi.copy()

    def _apply_filter(signal_1d):
        """Apply bandpass filter to a 1D real signal."""
        return filtfilt(b, a, signal_1d)

    if csi.ndim == 2:
        result = np.empty_like(csi)
        for f in range(csi.shape[1]):
            if np.iscomplexobj(csi):
                result[:, f] = _apply_filter(np.real(csi[:, f])) + \
                               1j * _apply_filter(np.imag(csi[:, f]))
            else:
                result[:, f] = _apply_filter(csi[:, f])
    elif csi.ndim == 3:
        result = np.empty_like(csi)
        for f in range(csi.shape[1]):
            for a_idx in range(csi.shape[2]):
                if np.iscomplexobj(csi):
                    result[:, f, a_idx] = _apply_filter(np.real(csi[:, f, a_idx])) + \
                                          1j * _apply_filter(np.imag(csi[:, f, a_idx]))
                else:
                    result[:, f, a_idx] = _apply_filter(csi[:, f, a_idx])
    else:
        raise ValueError(f"Expected 2D or 3D array, got shape {csi.shape}")

    return result


def savgol_denoise(csi, window_length=11, polyorder=3):
    """
    Savitzky-Golay filter for CSI denoising.

    Applies polynomial smoothing along the time axis, which preserves
    peak shapes and sharp transitions better than simple moving average.
    Ideal for CSI data where activity events produce transient peaks.

    Args:
        csi: CSI array of shape (T, F, A) or (T, F) — complex or real
        window_length: Length of the filter window, must be odd and > polyorder (default: 11)
        polyorder: Order of the polynomial used to fit samples (default: 3)

    Returns:
        np.ndarray: Denoised CSI with same shape and dtype as input

    Reference:
        Savitzky A, Golay MJE. "Smoothing and differentiation of data by
        simplified least squares procedures." Analytical Chemistry,
        vol. 36, no. 8, pp. 1627-1639, 1964.
    """
    if csi.size == 0:
        return csi.copy()
    if csi.ndim < 2:
        raise ValueError(f"Expected at least 2D array, got shape {csi.shape}")

    T = csi.shape[0]

    # Validate parameters
    if window_length < polyorder + 2:
        raise ValueError(f"window_length ({window_length}) must be > polyorder ({polyorder}) + 1")
    if window_length % 2 == 0:
        raise ValueError(f"window_length must be odd, got {window_length}")
    if window_length > T:
        if T >= polyorder + 2:
            window_length = T if T % 2 == 1 else T - 1
        else:
            return csi.copy()

    if csi.ndim == 2:
        result = np.empty_like(csi)
        for f in range(csi.shape[1]):
            if np.iscomplexobj(csi):
                result[:, f] = savgol_filter(np.real(csi[:, f]), window_length, polyorder) + \
                               1j * savgol_filter(np.imag(csi[:, f]), window_length, polyorder)
            else:
                result[:, f] = savgol_filter(csi[:, f], window_length, polyorder)
    elif csi.ndim == 3:
        result = np.empty_like(csi)
        for f in range(csi.shape[1]):
            for a_idx in range(csi.shape[2]):
                if np.iscomplexobj(csi):
                    result[:, f, a_idx] = savgol_filter(np.real(csi[:, f, a_idx]), window_length, polyorder) + \
                                          1j * savgol_filter(np.imag(csi[:, f, a_idx]), window_length, polyorder)
                else:
                    result[:, f, a_idx] = savgol_filter(csi[:, f, a_idx], window_length, polyorder)
    else:
        raise ValueError(f"Expected 2D or 3D array, got shape {csi.shape}")

    return result
