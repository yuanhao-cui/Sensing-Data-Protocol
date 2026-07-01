# Python API

WSDP provides a Python API for programmatic usage.

## Core Functions

### `pipeline()`

Run the full training pipeline.

```python
from wsdp import pipeline

pipeline(
    input_path='./data/elderAL',
    output_folder='./output',
    dataset='elderAL',
    learning_rate=1e-3,
    num_epochs=50,
)
```

### `download()`

Download datasets programmatically.

```python
from wsdp import download

download('widar', './data/widar', token='your-jwt-token')
```

### `predict()`

Run inference on CSI data.

```python
from wsdp import predict
import numpy as np

csi = np.random.randn(5, 200, 30, 3) + 1j * np.random.randn(5, 200, 30, 3)
predictions = predict(csi, 'best_checkpoint.pth', num_classes=6)

# With padding to fixed length
predictions = predict(csi, 'best_checkpoint.pth', num_classes=6, padding_length=200)
```

## Using Models

All 19 built-in models are accessible through `create_model()`:

```python
from wsdp.models import create_model

# SOTA models
model = create_model("THAT", num_classes=6, input_shape=(200, 30, 3))
model = create_model("CSITime", num_classes=6, input_shape=(200, 30, 3))
model = create_model("PA_CSI", num_classes=6, input_shape=(200, 30, 3))

# Lightweight models for edge deployment
model = create_model("WiFlexFormer", num_classes=6, input_shape=(200, 30, 3))
model = create_model("AttentionGRU", num_classes=6, input_shape=(200, 30, 3))

# Cross-domain models
model = create_model("EI", num_classes=6, input_shape=(200, 30, 3), num_domains=3)
model = create_model("FewSense", num_classes=6, input_shape=(200, 30, 3), n_support=5)

# Use any model in the pipeline
pipeline(
    input_path='./data/elderAL',
    output_folder='./output',
    dataset='elderAL',
    model_name='THAT',
)
```

## Algorithms

### Unified API

```python
from wsdp.algorithms import denoise, calibrate, normalize, interpolate
from wsdp.algorithms import extract_features, detect, remove_outliers

# Denoising (5 methods)
denoised = denoise(csi, method='butterworth', order=5)
denoised = denoise(csi, method='hampel', window=5, threshold=3.0)
denoised = denoise(csi, method='bandpass', low_freq=0.5, high_freq=50.0, fs=1000.0)

# Calibration
calibrated = calibrate(denoised, method='stc')

# Normalization (including AGC)
normalized = normalize(calibrated, method='agc', target_power=1.0)

# Feature extraction (including new algorithms)
features = extract_features(normalized, features=['conjugate_multiply'])
fused = extract_features(normalized, features=['pca_fusion'])

# Outlier removal
cleaned = remove_outliers(csi, method='iqr', factor=1.5)
cleaned = remove_outliers(csi, method='z-score', threshold=3.0)

# Interpolation (including decimation)
resampled = interpolate(csi, method='decimate', target_K=15)
```

### Algorithm Presets

```python
from wsdp.algorithms import apply_preset, execute_pipeline

# Apply a preset
steps = apply_preset('high_quality')
processed = execute_pipeline(csi, steps)
```

Available presets: `high_quality`, `fast`, `robust`, `gesture_recognition`, `activity_detection`, `localization`.

## Preprocessing Cache

Cache preprocessed data to skip reprocessing on repeated runs:

```python
# First run: preprocesses and caches to disk
pipeline(
    input_path='./data/elderAL',
    output_folder='./output',
    dataset='elderAL',
    use_cache=True,
)

# Second run: loads from cache, much faster
pipeline(
    input_path='./data/elderAL',
    output_folder='./output',
    dataset='elderAL',
    use_cache=True,
    model_name='THAT',  # try a different model on the same data
)
```

## Progress Callback

Monitor training progress programmatically:

```python
def my_callback(epoch, total_epochs, metrics):
    print(f"Epoch {epoch}/{total_epochs} - "
          f"loss: {metrics['loss']:.4f}, acc: {metrics['accuracy']:.4f}")

pipeline(
    input_path='./data/elderAL',
    output_folder='./output',
    dataset='elderAL',
    progress_callback=my_callback,
)
```

## Experiment Tracker

Track and compare experiments across multiple runs:

```python
from wsdp.tracking import ExperimentTracker

tracker = ExperimentTracker('./experiments')

# Log experiment parameters and results
tracker.log_experiment(
    name='THAT_elderAL_v1',
    params={'model': 'THAT', 'lr': 1e-3, 'epochs': 50},
    metrics={'accuracy': 0.95, 'f1': 0.94},
)

# Compare experiments
tracker.compare(['THAT_elderAL_v1', 'CSITime_elderAL_v1'])

# Get best experiment by metric
best = tracker.get_best(metric='accuracy')
```

See [API Reference](../api/core.md) for full documentation.
