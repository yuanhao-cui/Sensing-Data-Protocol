# Algorithm Guide

WSDP includes a comprehensive algorithm library for CSI signal processing. This guide helps you choose and use the right algorithms for your use case.

## Overview

WSDP algorithms are organized into six categories:

```
wsdp.algorithms
├── denoise()          # Remove noise from CSI signals
├── calibrate()        # Correct phase errors
├── normalize()        # Normalize amplitude
├── interpolate()      # Resample frequency grid
├── extract_features() # Extract signal features
└── detect_activity()  # Detect activity (see also change_point_detection())
```

## Choosing the Right Algorithm

### Denoising

| Method | Speed | Quality | Best For |
|--------|:-----:|:-------:|----------|
| **wavelet** | Medium | High | General-purpose, preserves edges |
| **butterworth** | Fast | High | Smooth signals, configurable cutoff |
| **savgol** | Fastest | Medium | Real-time applications, minimal distortion |

**Recommendation:**
- **For research/accuracy**: Use `butterworth` (order=5, cutoff=0.3)
- **For speed**: Use `savgol` (window_length=7-11, polyorder=3)
- **For noisy environments**: Use `wavelet` (automatic thresholding)

### Phase Calibration

| Method | Speed | Quality | Best For |
|--------|:-----:|:-------:|----------|
| **linear** | Fastest | Basic | Simple phase errors |
| **polynomial** | Medium | Good | Non-linear phase distortions |
| **stc** | Medium | High | CFO/SFO errors (IEEE TWC 2019) |
| **robust** | Slowest | Highest | Multipath-heavy environments |

**Recommendation:**
- **Default choice**: `stc` (most robust for commodity WiFi)
- **Clean environments**: `linear` (fast and sufficient)
- **Complex environments**: `robust` (handles outliers)

### Normalization

| Method | Output Range | Best For |
|--------|:------------:|----------|
| **z-score** | ~N(0,1) | Most DL models, standard choice |
| **min-max** | [0, 1] | Sigmoid/tanh activation functions |

### Interpolation

| Method | Smoothness | Best For |
|--------|:----------:|----------|
| **cubic** | Highest | Upsampling, localization |
| **linear** | Medium | General-purpose |
| **nearest** | None | Categorical data, downsampling |

## Quick Examples

### Basic Processing Pipeline

```python
from wsdp.algorithms import denoise, calibrate, normalize

# Chain algorithms manually
denoised = denoise(csi, method='butterworth', order=5, cutoff=0.3)
calibrated = calibrate(denoised, method='stc')
normalized = normalize(calibrated, method='z-score')
```

### Using Presets (Recommended)

```python
from wsdp.algorithms import apply_preset, execute_pipeline

# Use a preset for common use cases
steps = apply_preset('high_quality')
processed = execute_pipeline(csi, steps)

# Available presets:
# 'high_quality'     - Maximum accuracy
# 'fast'             - Speed-optimized
# 'robust'           - Noisy environments
# 'gesture_recognition' - Gesture tasks
# 'activity_detection'  - HAR tasks
# 'localization'        - Localization tasks
```

### Configuration File

Use the example config at `examples/configs/algorithms_config.yaml` (or create your own):

```yaml
denoise:
  method: butterworth
  params:
    order: 5
    cutoff: 0.3

calibrate:
  method: stc

normalize:
  method: z-score
```

Use it:

```python
from wsdp.algorithms import load_config, execute_pipeline

config = load_config('examples/configs/algorithms_config.yaml')
processed = execute_pipeline(csi, config)
```

## Adding Custom Algorithms

WSDP's pluggable architecture makes it easy to add your own algorithms:

```python
import numpy as np
from wsdp.algorithms import register_algorithm, denoise

# 1. Implement your algorithm
def my_median_denoise(csi, kernel_size=5, **kwargs):
    """Median filter denoising for CSI."""
    from scipy.ndimage import median_filter
    result = np.empty_like(csi)
    for f in range(csi.shape[1]):
        for a in range(csi.shape[2]):
            real_part = median_filter(np.real(csi[:, f, a]), size=kernel_size)
            imag_part = median_filter(np.imag(csi[:, f, a]), size=kernel_size)
            result[:, f, a] = real_part + 1j * imag_part
    return result

# 2. Register it
register_algorithm('denoise', 'median', my_median_denoise)

# 3. Use it like any built-in algorithm
result = denoise(csi, method='median', kernel_size=7)
```

### Creating Custom Presets

```python
from wsdp.algorithms import register_preset, execute_pipeline

register_preset('my_workflow', {
    'denoise': {'method': 'median', 'kernel_size': 7},
    'calibrate': {'method': 'stc'},
    'normalize': {'method': 'z-score'},
})

steps = apply_preset('my_workflow')
result = execute_pipeline(csi, steps)
```

## Feature Extraction

Extract signal features for downstream ML:

```python
from wsdp.algorithms import extract_features

features = extract_features(csi, features=['doppler', 'entropy', 'ratio'])

# features['doppler']  — Doppler spectrogram (n_freq, n_time, F, A)
# features['entropy']  — Shannon entropy per subcarrier (F, A)
# features['ratio']    — Antenna pair ratios (T, F, n_pairs)
```

## Activity Detection

Detect motion in CSI data:

```python
from wsdp.algorithms import detect_activity, change_point_detection

# Detect activity via sliding window variance
activity = detect_activity(csi, window=32, threshold=0.1)
# Returns: boolean array (T,) — True = activity detected

# Find transition points
change_points = change_point_detection(csi, method='mean_shift_ratio')
# Returns: array of time indices where transitions occur
```

## Algorithm Performance Tips

1. **Fixed execution order**: Custom configs always run in category order
   (`denoise → outliers → calibrate → normalize → interpolate → extract_features → detect`),
   no matter how you order them in the YAML/dict. (The built-in default chain is the
   one exception: it calibrates first, then denoises.)
2. **Normalization last**: Normalize after all other processing
3. **Match calibration to noise**: STC for CFO-dominant, robust for multipath
4. **Feature extraction**: Do after denoising and calibration for best results

## Full API

See the [Complete API Reference](../API_REFERENCE.md) for all parameters, return values, and academic references.
