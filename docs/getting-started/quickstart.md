# Quick Start

Get from zero to trained model in 5 minutes.

## Installation

```bash
pip install wsdp
```

## 5-Minute Quickstart

### 1. Download a Dataset

```bash
# Create a free account at sdp8.org, then:
wsdp download elderAL ./data --email you@example.com --password yourpassword
```

### 2. Train with Defaults

```bash
wsdp run ./data/elderAL ./output elderAL --lr 0.001 --epochs 50
```

### 3. Check Results

```bash
ls ./output/
# best_checkpoint_<seed>.pth, training_history_<seed>.csv, cm_rs_<seed>.png
# (one set of files per random seed)
```

### 4. Try an Algorithm Preset

Apply optimized preprocessing manually before training:

```python
from wsdp.algorithms import apply_preset, execute_pipeline

# High-quality preprocessing pipeline
steps = apply_preset('high_quality')
processed = execute_pipeline(csi_data, steps)

# Robust preprocessing (good for noisy environments)
steps = apply_preset('robust')
processed = execute_pipeline(csi_data, steps)
```

Available presets: `high_quality`, `fast`, `robust`, `gesture_recognition`, `activity_detection`, `localization`.

### 5. Try a Different Model

WSDP ships with 19 models. Swap models with a single flag:

```bash
# Lightweight model for edge deployment (~62K params)
wsdp run ./data/elderAL ./output elderAL --model WiFlexFormer --epochs 50

# SOTA two-stream Transformer (~300K params)
wsdp run ./data/elderAL ./output elderAL --model THAT --epochs 50

# Phase-amplitude attention model
wsdp run ./data/elderAL ./output elderAL --model PA_CSI --epochs 50
```

## Python API Quickstart

```python
from wsdp import pipeline, predict
import numpy as np
import glob

# Train (uses default CSIModel; num_seeds=1 for a single checkpoint)
pipeline(
    input_path='./data/elderAL',
    output_folder='./output',
    dataset='elderAL',
    num_epochs=50,
    num_seeds=1,
)

# Predict: auto-locate the checkpoint saved by pipeline
checkpoint_path = glob.glob('./output/best_checkpoint_*.pth')[0]
csi = np.random.randn(5, 200, 30, 3) + 1j * np.random.randn(5, 200, 30, 3)
predictions = predict(csi, checkpoint_path, num_classes=6)
```

## What's Next?

- [Model Selection Guide](../models.md) - Compare all 19 models
- [CLI Usage](../user-guide/cli.md) - Full CLI reference
- [Python API](../user-guide/python-api.md) - Programmatic usage
- [Configuration](../user-guide/configuration.md) - YAML configs and algorithm presets
- [Algorithm Reference](../api/algorithms.md) - All 26 built-in algorithms
