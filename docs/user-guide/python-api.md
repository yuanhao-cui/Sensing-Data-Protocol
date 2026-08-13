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
from wsdp.algorithms import extract_features, remove_outliers
from wsdp.algorithms import detect_activity, change_point_detection

# Denoising (5 methods)
denoised = denoise(csi, method='butterworth', order=5)
denoised = denoise(csi, method='hampel', window_size=5, n_sigma=3.0)
denoised = denoise(csi, method='bandpass', low_freq=0.5, high_freq=50.0, fs=1000.0)

# Calibration
calibrated = calibrate(denoised, method='stc')

# Normalization (z-score, min-max, or AGC compensation)
normalized = normalize(calibrated, method='z-score')
# AGC compensation requires the per-frame AGC gain values (shape (T,), from BfeeFrame.agc):
normalized = normalize(calibrated, method='agc', agc_values=agc_values)

# Feature extraction (including new algorithms)
features = extract_features(normalized, features=['conjugate_multiply'])
fused = extract_features(normalized, features=['pca_fusion'])

# Outlier removal (both methods use `factor` as the threshold multiplier)
cleaned = remove_outliers(csi, method='iqr', factor=1.5)
cleaned = remove_outliers(csi, method='z-score', factor=3.0)

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

Available presets: `high_quality`, `fast`, `robust`, `gesture_recognition`, `activity_detection`, `localization`, plus per-dataset presets named after each dataset (`widar`, `gait`, `xrf55`, `elderAL`, `zte`).

### Custom Algorithm Pipeline in `pipeline()`

Pass a flat dict of steps (method + parameters inline) to `pipeline()`:

```python
from wsdp import pipeline

pipeline(
    input_path='./data/elderAL',
    output_folder='./output',
    dataset='elderAL',
    pipeline_steps={
        'denoise': {'method': 'wavelet', 'level': 2},
        'calibrate': {'method': 'linear'},
        'normalize': {'method': 'z-score'},
    },
)
```

Steps execute in a fixed category order (`denoise → calibrate → normalize → ...`),
regardless of dict order. You can also point to a YAML/JSON file with
`algorithm_config_file=` or use `algorithm_preset='high_quality'` — see
[Configuration](configuration.md).

## Custom Readers & Modular Pipeline

Register a reader for a new file format, then select it independently of the
dataset's filename convention:

```python
from wsdp import pipeline
from wsdp.readers import BaseReader, register_reader

class MyReader(BaseReader):
    def sniff(self, file_path): return file_path.endswith('.myfmt')
    def read_file(self, file_path): ...  # parse the file into CSIData

register_reader('my_format', MyReader)
pipeline('./data/my_dataset', './output', 'xrf55', reader='my_format')
```

Compose preprocessing steps freely with `ModularProcessor`:

```python
from wsdp.algorithms import AlgorithmStep
from wsdp.processors import ModularProcessor

steps = [
    AlgorithmStep(category='denoise', method='wavelet', params={'level': 2}),
    AlgorithmStep(category='normalize', method='z-score'),  # calibration skipped
]
data, labels, groups = ModularProcessor(steps).process(csi_data_list, dataset='xrf55')
```

See `examples/scripts/custom_reader_algorithm.py` for a runnable end-to-end example.

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

Track training runs with a local CSV backend (or W&B / MLflow if installed):

```python
from wsdp.utils import ExperimentTracker

tracker = ExperimentTracker(backend='local', project_name='wsdp',
                            run_name='THAT_elderAL_v1', output_dir='./experiments')

# Log hyperparameters and per-epoch metrics
tracker.log_params({'model': 'THAT', 'lr': 1e-3, 'epochs': 50})
tracker.log_metrics({'loss': 0.12, 'accuracy': 0.95}, step=50)

# Optionally attach artifacts (checkpoints, plots)
tracker.log_artifact('./output/best_checkpoint_42.pth')

# Finalise the run (flushes params to CSV for the local backend)
tracker.finish()
```

Backends: `'local'` (CSV, no dependencies), `'wandb'` and `'mlflow'`
(require the corresponding packages; fall back to local CSV if missing).

See [API Reference](../api/core.md) for full documentation.
