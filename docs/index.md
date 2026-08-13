# WSDP Documentation

**Wi-Fi Sensing Data Processing**

> 🌐 **Official Platform**: [SDP8.org](https://sdp8.org) | 📦 **PyPI**: [wsdp](https://pypi.org/project/wsdp/) | 💻 **GitHub**: [Source Code](https://github.com/yuanhao-cui/SDP-Sensing-Data-Protocol-for-Scalable-Wireless-Sensing)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-EE4C2C.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](../LICENSE)
[![SDP8](https://img.shields.io/badge/Platform-SDP8.org-356596)](https://sdp8.org)

A Python library for downloading, processing, analyzing and training on Wi-Fi CSI (Channel State Information) data.

**Published and maintained by [SDP8.org](https://sdp8.org)** - the official platform for reproducible wireless sensing research.

## 🆕 What's New in v0.5.1

- **7 critical scientific bug fixes** -- corrected phase calibration, wavelet boundary handling, Doppler normalization, and more
- **7 new SOTA models** -- THAT, CSITime, PA_CSI, WiFlexFormer, AttentionGRU, EI, FewSense (19 total)
- **6 new preprocessing algorithms** -- conjugate_multiply, agc_compensate, pca_fusion, bandpass, hampel, decimate (26+ total)
- **Benchmark leaderboard** -- standardized model comparison across datasets
- **CI/CD pipeline** -- automated testing, linting, and release workflow
- **Experiment tracking & caching** -- reproducible runs with result caching
- **GroupKFold cross-validation** -- user-aware evaluation to prevent data leakage

## 🚀 Features

- **Multi-dataset support**: Widar, Gait, XRF55, ElderAL, ZTE datasets
- **Intelligent preprocessing**: Wavelet denoising, phase calibration, signal resizing
- **Deep learning pipeline**: End-to-end training with CNN + Transformer architecture
- **Authentication**: JWT Token, email/password, or non-interactive mode
- **Visualization**: Heatmaps, denoising comparison, phase calibration plots
- **Inference API**: Simple `predict()` interface for deployment
- **CLI interface**: Full command-line support for batch operations

## 🧪 Signal Processing Algorithm Library

WSDP provides a comprehensive, pluggable signal preprocessing library with **26+ algorithms in 7 categories**:

| Category | Algorithms | Description |
|----------|-----------|-------------|
| **Denoising** | Wavelet, Butterworth, Savitzky-Golay, Bandpass, Hampel | Remove noise while preserving features |
| **Phase Calibration** | Linear, Polynomial, STC, Robust | Correct hardware phase errors |
| **Amplitude** | Z-Score, Min-Max, IQR Outlier, AGC Compensation | Normalize and clean amplitude |
| **Interpolation** | Linear, Cubic, Nearest, Anti-alias Decimate | Resample to canonical grids |
| **Features** | Doppler, Entropy, CSI Ratio, Tensor, Conjugate Multiply, PCA Fusion | Extract motion/activity features |
| **Detection** | Variance-based, Change Point | Detect activity and transitions |
| **Composition** | Pipeline presets, YAML config | Chain algorithms declaratively |

**Unified API**:
```python
from wsdp.algorithms import denoise, calibrate, normalize, extract_features

denoised = denoise(csi, method='butterworth', order=5, cutoff=0.3)
calibrated = calibrate(csi, method='stc')
features = extract_features(csi, features=['doppler', 'entropy'])
```

See [Algorithm Guide](getting-started/algorithm-guide.md) for details.

## 🧠 Model Zoo (19 Models)

WSDP provides a complete pluggable model library from baselines to SOTA:

| Category | Model | Description |
|:--------:|:-----:|:------------|
| **Baseline** | MLPModel | Fully-connected baseline |
| | CNN1DModel | 1D convolution (temporal) |
| | CNN2DModel | 2D convolution (spectral) |
| | LSTMModel | LSTM temporal modeling |
| **Mainstream** | ResNet1D | 1D residual network |
| | ResNet2D | 2D residual network |
| | BiLSTMAttention | BiLSTM + attention |
| | EfficientNetCSI | Efficient CNN |
| **SOTA** | VisionTransformerCSI | ViT for CSI |
| | MambaCSI | State space model |
| | GraphNeuralCSI | GNN on antenna topology |
| | CSIModel | CNN + Transformer |
| **Specialized** | THAT | Two-stream Transformer (Li et al., 2021, ~300K params) |
| | CSITime | Inception-Time for CSI (Yadav et al., 2023, ~80K params) |
| | PA_CSI | Phase-Amplitude Attention (Sensors 2025, ~292K params) |
| **Lightweight** | WiFlexFormer | Efficient WiFi Transformer (arXiv 2411.04224, ~62K params) |
| | AttentionGRU | Attention + GRU (Sensors 2025, ~52K params) |
| **Cross-Domain** | EI | Environment-Independent (Jiang et al., 2020, ~226K params) |
| | FewSense | Few-shot Prototypical (TMC 2022, ~458K params) |

```python
from wsdp.models import create_model, list_models

# Create any model with a single call
model = create_model("ResNet1D", num_classes=10, input_shape=(20, 30, 3))

# List all available models
print(list_models())
```

### Comparison with Other Libraries

| Feature | SenseFi (2023) | CSIKit | **WSDP** |
|:-------:|:--------------:|:------:|:--------:|
| **Models** | 11 | ❌ | **19** |
| **Preprocessing** | ❌ | Basic | **26+ algorithms** |
| **Leaderboard** | ❌ | ❌ | ✅ **Built-in** |
| **Pluggable** | ❌ | ❌ | ✅ **Registry** |
| **Protocol Abstraction** | ❌ | ❌ | ✅ **Unique** |

See [Model Guide](models.md) for selection recommendations.

## 📦 Installation

```bash
pip install wsdp
```

Or install from source:

```bash
git clone https://github.com/yuanhao-cui/SDP-Sensing-Data-Protocol-for-Scalable-Wireless-Sensing.git
cd SDP-Sensing-Data-Protocol-for-Scalable-Wireless-Sensing
pip install -e .
```

## ⚡ Quick Start

> ⚠️ **Prerequisite**: Create a free account at [SDP8.org](https://sdp8.org) — required for dataset downloads.

### Python API

```python
from wsdp import pipeline, download

# Download dataset (SDP8.org account required)
download('widar', '/data/widar', email='you@example.com', password='yourpassword')

# Run pipeline
pipeline(
    input_path='/data/widar',
    output_folder='/output',
    dataset='widar',
)
```

### CLI

```bash
# Download with SDP8.org credentials
wsdp download widar /data --email you@example.com --password yourpassword

# Or with JWT token (from SDP8.org dashboard)
wsdp download widar /data --token YOUR_JWT_TOKEN

# Run training
wsdp run /data /output widar --lr 0.001 --epochs 50

# List datasets
wsdp list --verbose
```

## 📚 Documentation Sections

- [Getting Started](getting-started/installation.md): Installation and basic usage
- [User Guide](user-guide/cli.md): Detailed guides for CLI and Python API
- [API Reference](api/core.md): Complete API documentation
- [Datasets](datasets/overview.md): Information about supported datasets
- [Development](development/contributing.md): Contributing guidelines and changelog

## 🤝 Contributing

We welcome contributions! See our [Contributing Guide](development/contributing.md) for details.

## 📄 License

This project is licensed under the MIT License.

## 📖 Citation

```bibtex
@ARTICLE{11652923,
  author={Zhang, Di and Huang, Jiawei and Cui, Yuanhao and Cao, Xiaowen and Han, Tony Xiao and Jing, Xiaojun},
  journal={IEEE Transactions on Mobile Computing},
  title={SDP: A Unified Protocol and Benchmarking Framework for Reproducible Wi-Fi Sensing},
  year={2026},
  volume={},
  number={},
  pages={1-14},
  keywords={Wireless fidelity;Modeling;Frequency;Protocols;Training;Streams;Accuracy;Measurement;Tensors;Antennas;Benchmark;canonical representation;channel state information (CSI);integrated sensing and communications (ISAC);reproducibility;wireless sensing},
  doi={10.1109/TMC.2026.3723025}}
```
