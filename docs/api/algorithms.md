# API Reference - Algorithms

WSDP provides a comprehensive algorithm library for CSI processing with a **pluggable architecture**. All 26 built-in algorithms are organized into 7 categories.

## Quick Reference

| Category | Built-in Algorithms |
|----------|-------------------|
| **Denoising** | `wavelet`, `butterworth`, `savgol`, `bandpass`, `hampel` |
| **Calibration** | `linear`, `polynomial`, `stc`, `robust` |
| **Normalization** | `z-score`, `min-max`, `agc` |
| **Interpolation** | `linear`, `cubic`, `nearest`, `decimate` |
| **Features** | `doppler`, `entropy`, `ratio`, `decomposition`, `conjugate_multiply`, `pca_fusion` |
| **Detection** | `activity`, `change_point` |
| **Outliers** | `iqr`, `z-score` |

## Unified API

```python
from wsdp.algorithms import denoise, calibrate, normalize
from wsdp.algorithms import extract_features

denoised = denoise(csi, method='butterworth', order=5)
calibrated = calibrate(denoised, method='stc')
normalized = normalize(calibrated, method='z-score')
features = extract_features(normalized, features=['doppler', 'entropy'])
```

## Algorithm Details

### Denoising

| Algorithm | Function | Description |
|-----------|----------|-------------|
| `wavelet` | `denoise(csi, method='wavelet')` | Wavelet-based denoising using soft thresholding. Good general-purpose denoiser. |
| `butterworth` | `denoise(csi, method='butterworth', order=5)` | Butterworth low-pass filter. Smooth frequency response with configurable order. |
| `savgol` | `denoise(csi, method='savgol', window=11, polyorder=3)` | Savitzky-Golay filter. Preserves signal shape while removing high-frequency noise. |
| `bandpass` | `denoise(csi, method='bandpass', low_freq=0.5, high_freq=50.0, fs=1000.0)` | Bandpass filter for isolating activity-related frequency components. |
| `hampel` | `denoise(csi, method='hampel', window=5, threshold=3.0)` | Hampel identifier. Replaces outlier samples with local median. Effective against impulse noise. |

### Calibration

| Algorithm | Function | Description |
|-----------|----------|-------------|
| `linear` | `calibrate(csi, method='linear', reference=ref)` | Linear phase calibration using a static reference measurement. |
| `polynomial` | `calibrate(csi, method='polynomial', degree=3)` | Polynomial fitting for phase error correction across subcarriers. |
| `stc` | `calibrate(csi, method='stc')` | Spatial-temporal calibration. Removes phase offsets across antennas and time. |
| `robust` | `calibrate(csi, method='robust')` | Robust calibration using median-based estimation, resilient to outliers. |

### Normalization

| Algorithm | Function | Description |
|-----------|----------|-------------|
| `z-score` | `normalize(csi, method='z-score')` | Standardize to zero mean, unit variance per subcarrier. |
| `min-max` | `normalize(csi, method='min-max')` | Scale to [0, 1] range per subcarrier. |
| `agc` | `normalize(csi, method='agc', target_power=1.0)` | Automatic gain control. Normalizes signal power over sliding windows. |

### Interpolation

| Algorithm | Function | Description |
|-----------|----------|-------------|
| `linear` | `interpolate(csi, method='linear', target_K=30)` | Linear interpolation to target number of subcarriers. |
| `cubic` | `interpolate(csi, method='cubic', target_K=30)` | Cubic spline interpolation. Smoother than linear. |
| `nearest` | `interpolate(csi, method='nearest', target_K=30)` | Nearest-neighbor interpolation. Fastest, no smoothing. |
| `decimate` | `interpolate(csi, method='decimate', target_K=15)` | Downsample with anti-aliasing filter. Reduces to the target number of subcarriers. |

### Feature Extraction

| Algorithm | Function | Description |
|-----------|----------|-------------|
| `doppler` | `extract_features(csi, features=['doppler'])` | Doppler frequency shift estimation via FFT along the time axis. |
| `entropy` | `extract_features(csi, features=['entropy'])` | Shannon entropy of amplitude distribution per subcarrier. |
| `ratio` | `extract_features(csi, features=['ratio'])` | CSI ratio between antenna pairs to remove common-mode noise. |
| `decomposition` | `extract_features(csi, features=['decomposition'])` | PCA/SVD decomposition to extract principal signal components. |
| `conjugate_multiply` | `extract_features(csi, features=['conjugate_multiply'])` | Conjugate multiplication between antenna pairs. Eliminates shared phase noise and extracts differential phase. |
| `pca_fusion` | `extract_features(csi, features=['pca_fusion'])` | PCA-based fusion of multi-antenna streams into a single enhanced representation. |

### Detection

| Algorithm | Function | Description |
|-----------|----------|-------------|
| `activity` | `detect(csi, method='activity', threshold=0.5)` | Activity detection using variance-based thresholding on CSI amplitude. |
| `change_point` | `detect(csi, method='change_point')` | Detects abrupt changes in CSI statistics for segmenting activity boundaries. |

### Outlier Removal

| Algorithm | Function | Description |
|-----------|----------|-------------|
| `iqr` | `remove_outliers(csi, method='iqr', factor=1.5)` | Interquartile range method. Flags samples outside Q1 - 1.5*IQR to Q3 + 1.5*IQR. |
| `z-score` | `remove_outliers(csi, method='z-score', threshold=3.0)` | Flags samples more than N standard deviations from the mean. |

## Pluggable Architecture

### Register Custom Algorithms

```python
from wsdp.algorithms import register_algorithm, denoise

def my_denoise(csi, **kwargs):
    return csi * 0.5

register_algorithm('denoise', 'my_method', my_denoise)
result = denoise(csi, method='my_method')
```

### Use Presets

```python
from wsdp.algorithms import apply_preset, execute_pipeline

steps = apply_preset('high_quality')
processed = execute_pipeline(csi, steps)
```

Available presets:

| Preset | Description |
|--------|-------------|
| `high_quality` | Maximum accuracy: butterworth denoise + STC calibration + z-score normalization |
| `fast` | Speed-optimized: savgol denoise + linear calibration + min-max normalization |
| `robust` | Noisy environments: wavelet denoise + robust calibration + z-score normalization |
| `gesture_recognition` | Gesture tasks: butterworth denoise + STC calibration + z-score normalization + cubic interpolation |
| `activity_detection` | HAR tasks: savgol denoise + polynomial calibration + z-score normalization |
| `localization` | Localization tasks: wavelet denoise + robust calibration + z-score normalization + cubic interpolation |

### Load from Config File

```python
from wsdp.algorithms import load_config, execute_pipeline

config = load_config('examples/configs/algorithms_config.yaml')
processed = execute_pipeline(csi, config)
```

See the [Full API Reference](../API_REFERENCE.md) for complete documentation of all algorithms, parameters, and references.
