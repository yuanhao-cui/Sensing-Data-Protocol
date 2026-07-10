# SDP: Sensing Data Protocol for Scalable Wireless Sensing

<div align="center">

[![SDP Website](https://img.shields.io/badge/🌐_Official_Platform-SDP8.org-356596)](https://sdp8.org/)
[![PyPI](https://img.shields.io/badge/dynamic/toml?url=https://raw.githubusercontent.com/yuanhao-cui/Sensing-Data-Protocol/refs/heads/main/pyproject.toml&query=%24.project.name&logo=pypi&label=pip)](https://pypi.org/project/wsdp/)
[![License](https://img.shields.io/github/license/yuanhao-cui/Sensing-Data-Protocol?color=green)](https://github.com/yuanhao-cui/Sensing-Data-Protocol/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-EE4C2C.svg)](https://pytorch.org)
[![Tests](https://img.shields.io/badge/tests-pytest-blueviolet)](https://docs.pytest.org)
[![Docs](https://img.shields.io/badge/docs-MkDocs-blue.svg)](https://yuanhao-cui.github.io/SDP-Sensing-Data-Protocol-for-Scalable-Wireless-Sensing/)
[![Colab](https://img.shields.io/badge/Colab-Tutorial-yellow.svg)](https://colab.research.google.com/github/yuanhao-cui/SDP-Sensing-Data-Protocol-for-Scalable-Wireless-Sensing/blob/main/examples/wsdp_tutorial.ipynb)

**Published and maintained by [SDP8.org](https://sdp8.org) — the official platform for reproducible wireless sensing.**

</div>

---

## 📖 Citation

If you use SDP in your research, please cite:

```bibtex
@misc{zhang2026sdpunifiedprotocolbenchmarking,
      title={SDP: A Unified Protocol and Benchmarking Framework for Reproducible Wireless Sensing}, 
      author={Di Zhang and Jiawei Huang and Yuanhao Cui and Xiaowen Cao and Tony Xiao Han and Xiaojun Jing and Christos Masouros},
      year={2026},
      eprint={2601.08463},
      archivePrefix={arXiv},
      primaryClass={eess.SP},
      url={https://arxiv.org/abs/2601.08463}, 
}
```

---

<div align="center">

**[🇬🇧 English](#english) | [🇨🇳 中文](#中文)**

</div>

---

<a id="english"></a>
# 🇬🇧 English

## 🆕 What's New in v0.5.1

- **7 critical scientific bug fixes** -- corrected phase calibration, wavelet boundary handling, Doppler normalization, and more
- **7 new SOTA models** -- THAT, CSITime, PA_CSI, WiFlexFormer, AttentionGRU, EI, FewSense (19 total)
- **6 new preprocessing algorithms** -- conjugate_multiply, agc_compensate, pca_fusion, bandpass, hampel, decimate (26+ total)
- **Benchmark leaderboard** -- standardized model comparison across datasets
- **CI/CD pipeline** -- automated testing, linting, and release workflow
- **Experiment tracking & caching** -- reproducible runs with result caching
- **GroupKFold cross-validation** -- user-aware evaluation to prevent data leakage

---

## 🎯 What is SDP?

SDP is a **protocol-level abstraction** and unified benchmark for **reproducible wireless sensing**.

> ⚠️ **SDP is not a new neural network**, but a standardized protocol that unifies CSI representations for fair comparison.

## 🆚 Why Choose WSDP?

| Capability | WSDP | SenseFi (2023) | CSIKit |
|:----------:|:----:|:--------------:|:------:|
| **Built-in Models** | **19 (MLP→Mamba/GNN)** | 11 (MLP→ViT) | ❌ |
| **Preprocessing Algorithms** | **26+ (Wavelet, STC, etc.)** | ❌ | Basic |
| **Datasets** | **5** | 4 | ❌ |
| **Pluggable Architecture** | ✅ **Registry** | ❌ | ❌ |
| **Protocol Abstraction** | ✅ **Unique** | ❌ | ❌ |
| **Training Pipeline** | ✅ | ✅ | ❌ |
| **CLI** | ✅ **Full** | Basic | ✅ |

> *Verified from official GitHub repos on 2026-03-17.*

---

## 🧠 Model Zoo (19 Models, Baseline → SOTA)

<div align="center">

| Category | Models | Use Case |
|:--------:|:------:|:---------|
| **Baseline** | MLP, CNN1D, CNN2D, LSTM | Quick experiments, comparisons |
| **Mainstream** | ResNet1D, ResNet2D, BiLSTM+Attn, EfficientNet | Production use |
| **SOTA** | ViT, Mamba, GNN, CSIModel | Cutting-edge research |
| **Specialized** | THAT, CSITime, PA_CSI | Task-specific architectures |
| **Lightweight** | WiFlexFormer, AttentionGRU | Efficient deployment |
| **Cross-Domain** | EI, FewSense | Domain adaptation & few-shot |

</div>

```python
from wsdp.models import create_model, list_models
model = create_model("ResNet1D", num_classes=10, input_shape=(20, 30, 3))
```

> **⚠️ Baseline Model Architecture Note**
>
> Baseline models (MLP, CNN1D, CNN2D, LSTM) use a **Spatial Encoder** (Conv2d-based)
> to compress the `(F, A)` antenna dimension before temporal processing. This prevents
> parameter explosion from direct `(T, F, A)` flattening. See `CHANGELOG.md` for details.

---

## 🧪 Algorithm Library (26+ Algorithms in 7 Categories)

<div align="center">

| Category | Algorithms | Count |
|:--------:|:----------:|:-----:|
| **Denoising** | Wavelet, Butterworth, Savitzky-Golay, Bandpass, Hampel | 5 |
| **Phase Calibration** | Linear, Polynomial, STC, Robust | 4 |
| **Amplitude** | Z-Score, Min-Max, IQR Outlier, AGC Compensation | 4 |
| **Interpolation** | Linear, Cubic, Nearest, Anti-alias Decimate | 4 |
| **Features** | Doppler, Entropy, CSI Ratio, Tensor, Conjugate Multiply, PCA Fusion | 6 |
| **Detection** | Variance, Change Point | 2 |
| **Composition** | Pipeline presets, YAML config | - |

</div>

```python
from wsdp.algorithms import denoise, calibrate, normalize
denoised = denoise(csi, method='butterworth', order=5, cutoff=0.3)
calibrated = calibrate(csi, method='stc')
```

See [Model Guide](docs/models.md) and [Algorithm Guide](docs/getting-started/algorithm-guide.md).

---

## 🎯 What is SDP? (Cont.)

### The Problem

Wireless sensing research often suffers from:
- ❌ Hardware-specific CSI formats
- ❌ Inconsistent preprocessing pipelines  
- ❌ Unstable training results
- ❌ Large performance variance across random seeds

**Result**: Models cannot be fairly compared.

### The Solution

SDP solves this at the **protocol level**, not the model level:

| Feature | Raw CSI | Other Tools | **SDP** |
|:-------:|:-------:|:-----------:|:-------:|
| **Standardized Format** | ❌ Hardware-specific | ⚠️ Partial | ✅ **Unified CSIFrame** |
| **Multi-Dataset Support** | ❌ Manual parsing | ⚠️ 2-3 datasets | ✅ **5 datasets built-in** |
| **Preprocessing** | ❌ DIY | ⚠️ Basic only | ✅ **Wavelet + Phase Calib** |
| **Reproducibility** | ❌ Random | ⚠️ Varies | ✅ **5-seed standard** |
| **Deep Learning** | ❌ From scratch | ⚠️ Limited | ✅ **CNN+Transformer** |
| **CLI Interface** | ❌ None | ⚠️ Partial | ✅ **Full CLI support** |

SDP projects raw CSI into a fixed **canonical frequency grid (K=30)**, ensuring cross-hardware comparability.

### Performance Highlights

<div align="center">

| Metric | Result |
|:------:|:------:|
| **Accuracy** | SOTA on 5 datasets |
| **Reproducibility** | 5-seed evaluation standard |
| **Stability** | Low variance across runs |

![Accuracy](./img/accuracy.png)
*Figure 1: Accuracy comparison across datasets*

![Reproducibility](./img/reproducibility_and_stability.png)
*Figure 2: Reproducibility and stability analysis*

![Ablation](./img/ablation_rank.png)
*Figure 3: Ablation study results*

</div>

---

## 🚀 Quick Start (3 Steps, 5 Minutes)

### Step 1: Install (30 seconds)

```bash
pip install wsdp
```

Verify installation:
```bash
wsdp --version
```

### Step 2: Download Dataset (2 minutes)

> 🔑 **Required**: Create a free account at **[SDP8.org](https://sdp8.org)** — your account credentials are needed for dataset downloads.
>
> 🌐 **VPN is recommended** for downloading datasets.

**Option A: From CLI (Recommended for testing)**

All datasets hosted on **[SDP8.org](https://sdp8.org)**:

```bash
# elderAL = smallest dataset, fastest for testing
# Use your SDP8.org email/password:
wsdp download elderAL ./data --email you@example.com --password yourpassword

# Or use a JWT token (from SDP8.org dashboard):
wsdp download elderAL ./data --token YOUR_JWT_TOKEN

# Download larger datasets:
# wsdp download widar ./data
# wsdp download gait ./data
# wsdp download xrf55 ./data
# wsdp download zte ./data --email you@example.com --password yourpassword
# ⚠️ zte requires applying for access on the SDP platform first
```

**Option B: From [SDP8.org](https://sdp8.org) Web Interface**

Log in at [sdp8.org](https://sdp8.org) and download datasets manually.

**Required Dataset Structure:**
```
data/
├── elderAL/                    # Dataset name
│   ├── action0_static_new/     # Activity folder
│   │   ├── user0_position1_activity0/  # Sample folder
│   │   │   ├── sample1.csv
│   │   │   └── ...
│   │   └── ...
│   ├── action1_walk_new/
│   └── ...
├── widar/
│
├── gait/
│
├── xrf55/
│   └── WIFI/
│       └── sample.npy
└── zte/
```

### Step 3: Train & Evaluate (2 minutes)

**🐍 Python API (Recommended for research):**

Create `train.py`:
```python
from wsdp import pipeline

# Minimal call - uses default hyperparameters
pipeline("./data/elderAL", "./output", "elderAL")

# Or with custom hyperparameters
pipeline(
    input_path="./data/elderAL",
    output_folder="./output",
    dataset="elderAL",
    learning_rate=1e-3,
    num_epochs=50,
    batch_size=64,
)
```

Run:
```bash
python train.py
```

**💻 CLI (Quick & Simple):**

```bash
# Basic training
wsdp run ./data/elderAL ./output elderAL

# With hyperparameter override
wsdp run ./data/elderAL ./output elderAL --lr 0.001 --epochs 50 --batch-size 64

# With config file
wsdp run ./data/elderAL ./output elderAL --config my_config.yaml
```

**📊 What You Get:**

After training, check `./output/`:
```
output/
├── best_model.pth              # Best model checkpoint
├── confusion_matrix.png        # Evaluation visualization
├── training_curves.png         # Loss & accuracy curves
└── output.log                  # Detailed training logs
```

✅ **If you see these files, SDP is working correctly!**

---

## 📊 Supported Datasets

| Dataset | Format | Subcarriers | Complex | Scenarios | Size |
|:-------:|:------:|:-----------:|:-------:|:---------:|:----:|
| **Widar** | .dat (bfee) | 30 | ✅ | Gesture recognition | ~2GB |
| **Gait** | .dat (bfee, Intel IWL5300) | 30 | ✅ | Gait recognition | ~1GB |
| **XRF55** | .npy | 30 | ✅ | Human activity | ~3GB |
| **ElderAL** | .csv | varies | ❌ | Elderly activity | ~500MB |
| **ZTE** | .csv | 512 | ✅ | CSI with I/Q | ~4GB |

**More datasets coming soon!** See [Roadmap](#roadmap).

---

## 🔬 Research & Customization

### 🧠 Plug in Your Own Model

**Step 1:** Create `custom_model.py`:
```python
import torch
import torch.nn as nn

class YourCustomModel(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        # Your architecture here
        # Input shape: (Batch, Timestamp, Frequency, Antenna)
        
    def forward(self, x):
        # Your forward pass
        return output

# Required: expose model class
model = YourCustomModel
```

**Step 2:** Run with your model:
```bash
wsdp run ./data/elderAL ./output elderAL -m custom_model.py
```

### 📁 Use Your Own Dataset

**Organize your data:**
```
data/
└── my_dataset/
    ├── user0_pos0_action0/
    │   ├── sample1.csv
    │   └── ...
    └── user0_pos0_action1/
        └── ...
```

**Run:**
```bash
wsdp run ./data/my_dataset ./output my_dataset
```

### 🗺️ Codebase Map

Want to go deeper? Here's where to modify:

| Directory | Purpose | What to Modify |
|:---------:|:-------:|:--------------:|
| `models/` | Architectures | Define or compare model architectures |
| `algorithms/` | Signal Processing | Denoising, calibration, etc. |
| `datasets/` | Dataset Wrappers | Add new dataset loaders |
| `readers/` | File Readers | Add new format parsers |
| `structure/` | Data Structures | Modify CSIFrame format |
| `processors/` | Protocol Logic | Adjust canonical projection |

### 🔌 Pluggable Algorithm Architecture

WSDP features a **Registry Pattern** that makes algorithms pluggable:

```python
from wsdp.algorithms import denoise, calibrate, register_algorithm

# Unified API — switch methods with one parameter
denoised = denoise(csi, method='butterworth', order=5)
calibrated = calibrate(csi, method='stc')

# Register your own algorithm
def my_denoise(csi, **kwargs):
    return my_custom_filter(csi)

register_algorithm('denoise', 'my_method', my_denoise)
result = denoise(csi, method='my_method')  # Works like built-in!
```
You can try this in `examples/getting_started.ipynb` or just in your custom pipeline!

**Configuration file support:**

```yaml
# examples/configs/algorithms_config.yaml
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

```python
from wsdp.algorithms import load_config, execute_pipeline
config = load_config('examples/configs/algorithms_config.yaml')
processed = execute_pipeline(csi, config)
```
Or:
```python
pipeline(
    "./data/elderAL",
    "./output",
    "elderAL",
    algorithm_config_file='./examples/configs/algorithms_config.yaml',
)
```

**Pipeline presets:**

```python
from wsdp.algorithms import apply_preset, execute_pipeline

# Choose a preset for your use case
steps = apply_preset('high_quality')  # or 'fast', 'robust', etc.
processed = execute_pipeline(csi, steps)
```

### 📊 Algorithm Library

| Category | Algorithm | Key Function | Reference |
|:--------:|:---------:|:------------:|:---------:|
| **Denoising** | Wavelet | `wavelet_denoise_csi()` | Donoho & Johnstone, 1994 |
| | Butterworth | `butterworth_denoise()` | Butterworth, 1930 |
| | Savitzky-Golay | `savgol_denoise()` | Savitzky & Golay, 1964 |
| | Bandpass | `bandpass_filter()` | Standard DSP |
| | Hampel | `hampel_filter()` | Hampel, 1974 |
| **Phase Calibration** | Linear | `phase_calibration()` | Halperin et al., 2010 |
| | Polynomial | `polynomial_calibration()` | Extension of linear |
| | STC | `stc_calibration()` | Xie et al., IEEE TWC 2019 |
| | Robust | `robust_phase_sanitization()` | Wang et al., ICPADS 2012 |
| **Normalization** | Z-Score | `normalize_amplitude()` | Standard statistical |
| | Min-Max | `normalize_amplitude()` | Standard statistical |
| | AGC Compensation | `agc_compensate()` | AGC gain correction |
| **Interpolation** | Linear/Cubic/Nearest | `interpolate_grid()` | de Boor, 1978 |
| | Anti-alias Decimate | `decimate()` | Anti-alias downsampling |
| **Features** | Doppler | `doppler_spectrum()` | Ali et al., MobiCom 2015 |
| | Entropy | `entropy_features()` | Shannon, 1948 |
| | CSI Ratio | `csi_ratio()` | Halperin et al., 2011 |
| | Tensor Decomposition | `tensor_decomposition()` | Kolda & Bader, SIAM 2009 |
| | Conjugate Multiply | `conjugate_multiply()` | Antenna pair correlation |
| | PCA Fusion | `pca_fusion()` | Dimensionality reduction |
| **Detection** | Activity | `detect_activity()` | Zhou et al., 2013 |
| | Change Point | `change_point_detection()` | Adams & MacKay, 2007 |

**Built-in Presets:**

| Preset | Denoise | Calibrate | Use Case |
|:------:|:-------:|:---------:|:--------:|
| `high_quality` | Butterworth (order=5) | STC | Maximum accuracy |
| `fast` | Savitzky-Golay | Linear | Speed-optimized |
| `robust` | Wavelet | Robust | Noisy environments |
| `gesture_recognition` | Butterworth (order=4) | STC | Gesture tasks |
| `activity_detection` | Savitzky-Golay | Polynomial | HAR tasks |
| `localization` | Wavelet | Robust | Localization tasks |

---

## 🧪 Understanding SDP (10-Min Deep Dive)

### The SDP Pipeline

```
Raw CSI
  ↓
[Deterministic Sanitization]
  - Phase calibration
  - Wavelet denoising
  ↓
[Canonical Tensor Construction]
  - K=30 frequency grid
  - Standardized shape
  ↓
[Deep Learning Model]
  ↓
Prediction
```

### Canonical Tensor Format

After sanitization, SDP constructs a **Canonical CSI Tensor**:

$$X \in \mathbb{C}^{A \times K \times T}$$

Where:
- $A$ = Number of antennas
- $K$ = 30 (fixed frequency grid)
- $T$ = Time samples

This ensures **cross-hardware comparability**.

### Why Deterministic?

Raw CSI contains hardware distortions:
- Phase offsets
- Sampling time offsets  
- Noise fluctuations

SDP enforces **deterministic calibration and denoising**, guaranteeing:
- ✅ Same raw CSI → Same cleaned tensor
- ✅ Reproducibility is enforced, not optional

---

## 📚 Documentation & Resources

### 🎓 Tutorials (Recommended Order)

| # | Resource | What You'll Learn |
|:-:|:---------|:------------------|
| 1 | [**Quickstart Notebook**](examples/quickstart.ipynb) | 5-min intro — registry exploration & processor customization |
| 2 | [**Getting Started Notebook**](examples/getting_started.ipynb) | Algorithm deep-dive — phase calibration & denoising with step-by-step visualizations |
| 3 | [**Full Tutorial Notebook**](examples/wsdp_tutorial.ipynb) [![Colab](https://img.shields.io/badge/Colab-Open-yellow.svg)](https://colab.research.google.com/github/yuanhao-cui/SDP-Sensing-Data-Protocol-for-Scalable-Wireless-Sensing/blob/main/examples/wsdp_tutorial.ipynb) | End-to-end workflow — install → preprocess → train → evaluate → CLI |

### 📘 User Guide

| Resource | Description |
|:---------|:------------|
| [Installation](docs/getting-started/installation.md) | Setup & environment configuration |
| [Quickstart Guide](docs/getting-started/quickstart.md) | First steps with WSDP |
| [Algorithm Guide](docs/getting-started/algorithm-guide.md) | How to choose and chain preprocessing algorithms |
| [Python API](docs/user-guide/python-api.md) | Programmatic usage in detail |
| [CLI Reference](docs/user-guide/cli.md) | Command-line interface usage |
| [Configuration](docs/user-guide/configuration.md) | YAML config files & pipeline presets |

### 📊 Reference

| Resource | Description |
|:---------|:------------|
| [Full Documentation Site](https://yuanhao-cui.github.io/SDP-Sensing-Data-Protocol-for-Scalable-Wireless-Sensing/) | Complete MkDocs documentation |
| [API Reference](docs/API_REFERENCE.md) | All public APIs |
| [Dataset Overview](docs/datasets/overview.md) | Format details & download guide for all 5 datasets |
| [Model Guide](docs/models.md) | All 19 models with architecture details |
| [Leaderboard](docs/leaderboard.md) | Benchmark comparison across models & datasets |
| [Changelog](CHANGELOG.md) | Version history |
| [Contributing](CONTRIBUTING.md) | Development guide & PR process |

---

## 🗺️ Roadmap

- [x] **v0.1** - Initial protocol design
- [x] **v0.2** - 5 datasets support, CLI tool
- [x] **v0.3** - More datasets (WiFi-HAR, CSI-HAR, etc.)
- [x] **v0.4** - 19 models, 26+ algorithms, leaderboard, CI/CD, scientific bug fixes
- [x] **v0.5** - PyPI official release, online demo platform
- [ ] **v1.0** - Full protocol standardization

**Want a specific dataset?** [Open an issue](https://github.com/yuanhao-cui/Sensing-Data-Protocol-for-Scalable-Wireless-Sensing/issues) and let us know!

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Development setup
- Coding guidelines
- Pull request process

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

---

<a id="中文"></a>
# 🇨🇳 中文

## 🆕 v0.5.1 更新内容

- **7 个关键科学 bug 修复** -- 修正相位校准、小波边界处理、多普勒归一化等问题
- **7 个新 SOTA 模型** -- THAT, CSITime, PA_CSI, WiFlexFormer, AttentionGRU, EI, FewSense（共 19 个）
- **6 个新预处理算法** -- conjugate_multiply, agc_compensate, pca_fusion, bandpass, hampel, decimate（共 26+ 个）
- **基准排行榜** -- 跨数据集标准化模型对比
- **CI/CD 流水线** -- 自动化测试、代码检查和发布工作流
- **实验追踪与缓存** -- 可复现运行，支持结果缓存
- **GroupKFold 交叉验证** -- 用户感知评估，防止数据泄漏

---

## 🎯 SDP 是什么？

SDP 是一个**协议级抽象**框架，用于**可复现的无线感知研究**。

> ⚠️ **SDP 不是一个新的神经网络**，而是一个标准化协议，统一 CSI 表示以实现公平比较。

### 问题所在

无线感知研究常面临：
- ❌ 硬件特定的 CSI 格式
- ❌ 不一致的预处理流程
- ❌ 不稳定的训练结果
- ❌ 随机种子间性能方差大

**结果**：模型无法公平比较。

### 解决方案

SDP 在**协议层面**解决问题，而非模型层面：

| 特性 | 原始 CSI | 其他工具 | **SDP** |
|:----:|:--------:|:--------:|:-------:|
| **标准化格式** | ❌ 硬件特定 | ⚠️ 部分支持 | ✅ **统一 CSIFrame** |
| **多数据集支持** | ❌ 手动解析 | ⚠️ 2-3 个 | ✅ **5 个内置数据集** |
| **预处理** | ❌ 自行实现 | ⚠️ 仅基础 | ✅ **小波+相位校准** |
| **可复现性** | ❌ 随机 | ⚠️ 不稳定 | ✅ **5 种子标准** |
| **深度学习** | ❌ 从零开始 | ⚠️ 有限 | ✅ **CNN+Transformer** |
| **CLI 接口** | ❌ 无 | ⚠️ 部分 | ✅ **完整 CLI 支持** |

SDP 将原始 CSI 投影到固定的**规范频率网格 (K=30)**，确保跨硬件可比性。

### 性能亮点

<div align="center">

| 指标 | 结果 |
|:----:|:----:|
| **准确率** | 5 个数据集上达到 SOTA |
| **可复现性** | 5 种子评估标准 |
| **稳定性** | 多次运行方差低 |

![准确率](./img/accuracy.png)
*图 1：跨数据集准确率对比*

![可复现性](./img/reproducibility_and_stability.png)
*图 2：可复现性与稳定性分析*

![消融实验](./img/ablation_rank.png)
*图 3：消融实验结果*

</div>

---

## 🚀 快速开始（3 步，5 分钟）

### 第 1 步：安装（30 秒）

```bash
pip install wsdp
```

验证安装：
```bash
wsdp --version
```

### 第 2 步：下载数据集（2 分钟）

> 🔑 **前提条件**：在 **[SDP8.org](https://sdp8.org)** 注册免费账号 — 下载数据集需要使用账号凭证。
>
> 🌐 **建议使用 VPN 下载**数据集。

**方式 A：命令行下载（测试推荐）**

所有数据集由 **[SDP8.org](https://sdp8.org)** 官方托管：

```bash
# elderAL = 最小数据集，测试最快
# 使用 SDP8.org 的邮箱/密码：
wsdp download elderAL ./data --email you@example.com --password yourpassword

# 或使用 JWT Token（从 SDP8.org 控制台获取）：
wsdp download elderAL ./data --token YOUR_JWT_TOKEN

# 下载其他数据集：
# wsdp download widar ./data
# wsdp download gait ./data
# wsdp download xrf55 ./data
# wsdp download zte ./data --email you@example.com --password yourpassword
# ⚠️ zte requires applying for access on the SDP platform first
```

**方式 B：从 [SDP8.org](https://sdp8.org) 网页下载**

登录 [sdp8.org](https://sdp8.org) 后手动下载数据集。

**必需的数据集结构：**
```
data/
├── elderAL/                    # 数据集名称
│   ├── action0_static_new/     # 活动文件夹
│   │   ├── user0_position1_activity0/  # 样本文件夹
│   │   │   ├── sample1.csv
│   │   │   └── ...
│   │   └── ...
│   ├── action1_walk_new/
│   └── ...
├── widar/
├── gait/
├── xrf55/
└── zte/
```

### 第 3 步：训练与评估（2 分钟）

**🐍 Python API（研究推荐）：**

创建 `train.py`：
```python
from wsdp import pipeline

# 最小调用 - 使用默认超参数
pipeline("./data/elderAL", "./output", "elderAL")

# 或自定义超参数
pipeline(
    input_path="./data/elderAL",
    output_folder="./output",
    dataset="elderAL",
    learning_rate=1e-3,
    num_epochs=50,
    batch_size=64,
)
```

运行：
```bash
python train.py
```

**💻 命令行（快速简单）：**

```bash
# 基础训练
wsdp run ./data/elderAL ./output elderAL

# 自定义超参数
wsdp run ./data/elderAL ./output elderAL --lr 0.001 --epochs 50 --batch-size 64

# 使用配置文件
wsdp run ./data/elderAL ./output elderAL --config my_config.yaml
```

**📊 输出文件：**

训练后，查看 `./output/`：
```
output/
├── best_model.pth              # 最佳模型检查点
├── confusion_matrix.png        # 评估可视化
├── training_curves.png         # 损失和准确率曲线
└── output.log                  # 详细训练日志
```

✅ **如果看到这些文件，说明 SDP 运行正常！**

---

## 📊 支持的数据集

| 数据集 | 格式 | 子载波 | 复数 | 场景 | 大小 |
|:------:|:----:|:------:|:----:|:----:|:----:|
| **Widar** | .dat (bfee) | 30 | ✅ | 手势识别 | ~2GB |
| **Gait** | .dat (bfee) | 30 | ✅ | 步态识别 | ~1GB |
| **XRF55** | .npy | 30 | ✅ | 人体活动 | ~3GB |
| **ElderAL** | .csv | varies | ❌ | 老年人活动 | ~500MB |
| **ZTE** | .csv | 512 | ✅ | I/Q 格式 CSI | ~4GB |

**更多数据集即将推出！** 查看 [路线图](#路线图)。

---

## 🔬 研究与定制

### 🧠 接入你自己的模型

**第 1 步：** 创建 `custom_model.py`：
```python
import torch
import torch.nn as nn

class YourCustomModel(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        # 你的架构代码
        # 输入形状: (Batch, Timestamp, Frequency, Antenna)
        
    def forward(self, x):
        # 你的前向传播
        return output

# 必需：暴露模型类
model = YourCustomModel
```

**第 2 步：** 使用你的模型运行：
```bash
wsdp run ./data/elderAL ./output elderAL -m custom_model.py
```

### 📁 使用你自己的数据集

**组织你的数据：**
```
data/
└── my_dataset/
    ├── user0_pos0_action0/
    │   ├── sample1.csv
    │   └── ...
    └── user0_pos0_action1/
        └── ...
```

**运行：**
```bash
wsdp run ./data/my_dataset ./output my_dataset
```

### 🗺️ 代码结构地图

想深入修改？这里是各目录功能：

| 目录 | 用途 | 修改内容 |
|:----:|:----:|:--------:|
| `models/` | 架构 | 定义或比较模型架构 |
| `algorithms/` | 信号处理 | 去噪、校准等 |
| `datasets/` | 数据集包装 | 添加新数据集加载器 |
| `readers/` | 文件读取器 | 添加新格式解析器 |
| `structure/` | 数据结构 | 修改 CSIFrame 格式 |
| `processors/` | 协议逻辑 | 调整规范投影 |

### 🔌 可插拔算法架构

WSDP 采用**注册表模式**，让算法可以自由切换：

```python
from wsdp.algorithms import denoise, calibrate, register_algorithm

# 统一 API — 一个参数切换方法
denoised = denoise(csi, method='butterworth', order=5)
calibrated = calibrate(csi, method='stc')

# 注册你自己的算法
def my_denoise(csi, **kwargs):
    return my_custom_filter(csi)

register_algorithm('denoise', 'my_method', my_denoise)
result = denoise(csi, method='my_method')  # 像内置算法一样使用！
```

**配置文件支持：**

```yaml
# examples/configs/algorithms_config.yaml
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

```python
from wsdp.algorithms import load_config, execute_pipeline
config = load_config('examples/configs/algorithms_config.yaml')
processed = execute_pipeline(csi, config)
```

**Pipeline 预设：**

```python
from wsdp.algorithms import apply_preset, execute_pipeline

# 选择适合的预设
steps = apply_preset('high_quality')  # 或 'fast', 'robust' 等
processed = execute_pipeline(csi, steps)
```

### 📊 算法库

| 类别 | 算法 | 核心函数 | 参考文献 |
|:----:|:----:|:--------:|:--------:|
| **去噪** | 小波 | `wavelet_denoise_csi()` | Donoho & Johnstone, 1994 |
| | 巴特沃斯 | `butterworth_denoise()` | Butterworth, 1930 |
| | Savitzky-Golay | `savgol_denoise()` | Savitzky & Golay, 1964 |
| | 带通滤波 | `bandpass_filter()` | 标准 DSP |
| | Hampel 滤波 | `hampel_filter()` | Hampel, 1974 |
| **相位校准** | 线性 | `phase_calibration()` | Halperin et al., 2010 |
| | 多项式 | `polynomial_calibration()` | 线性校准的扩展 |
| | STC | `stc_calibration()` | Xie et al., IEEE TWC 2019 |
| | 鲁棒 | `robust_phase_sanitization()` | Wang et al., ICPADS 2012 |
| **归一化** | Z-Score | `normalize_amplitude()` | 标准统计方法 |
| | Min-Max | `normalize_amplitude()` | 标准统计方法 |
| | AGC 补偿 | `agc_compensate()` | AGC 增益校正 |
| **插值** | 线性/三次/最近邻 | `interpolate_grid()` | de Boor, 1978 |
| | 抗混叠降采样 | `decimate()` | 抗混叠降采样 |
| **特征提取** | 多普勒 | `doppler_spectrum()` | Ali et al., MobiCom 2015 |
| | 熵 | `entropy_features()` | Shannon, 1948 |
| | CSI 比率 | `csi_ratio()` | Halperin et al., 2011 |
| | 张量分解 | `tensor_decomposition()` | Kolda & Bader, SIAM 2009 |
| | 共轭乘积 | `conjugate_multiply()` | 天线对相关 |
| | PCA 融合 | `pca_fusion()` | 降维融合 |
| **检测** | 活动 | `detect_activity()` | Zhou et al., 2013 |
| | 变点 | `change_point_detection()` | Adams & MacKay, 2007 |

**内置预设：**

| 预设 | 去噪 | 校准 | 适用场景 |
|:----:|:----:|:----:|:--------:|
| `high_quality` | Butterworth (order=5) | STC | 最高精度 |
| `fast` | Savitzky-Golay | 线性 | 速度优化 |
| `robust` | 小波 | 鲁棒 | 噪声环境 |
| `gesture_recognition` | Butterworth (order=4) | STC | 手势任务 |
| `activity_detection` | Savitzky-Golay | 多项式 | 人体活动识别 |
| `localization` | 小波 | 鲁棒 | 定位任务 |

---

## 🧪 理解 SDP（10 分钟深度阅读）

### SDP 流程

```
原始 CSI
  ↓
[确定性清洗]
  - 相位校准
  - 小波去噪
  ↓
[规范张量构建]
  - K=30 频率网格
  - 标准化形状
  ↓
[深度学习模型]
  ↓
预测
```

### 规范张量格式

清洗后，SDP 构建**规范 CSI 张量**：

$$X \in \mathbb{C}^{A \times K \times T}$$

其中：
- $A$ = 天线数量
- $K$ = 30（固定频率网格）
- $T$ = 时间样本

这确保了**跨硬件可比性**。

### 为什么是确定性的？

原始 CSI 包含硬件失真：
- 相位偏移
- 采样时间偏移
- 噪声波动

SDP 强制执行**确定性校准和去噪**，保证：
- ✅ 相同的原始 CSI → 相同的清洗后张量
- ✅ 可复现性是强制的，不是可选的

---

## 📚 文档与资源

### 🎓 教程（推荐学习顺序）

| # | 资源 | 你将学到 |
|:-:|:-----|:---------|
| 1 | [**快速上手 Notebook**](examples/quickstart.ipynb) | 5 分钟入门 — 注册表探索与处理器定制 |
| 2 | [**入门指南 Notebook**](examples/getting_started.ipynb) | 算法详解 — 相位校准与去噪的逐步可视化演示 |
| 3 | [**完整教程 Notebook**](examples/wsdp_tutorial.ipynb) [![Colab](https://img.shields.io/badge/Colab-打开-yellow.svg)](https://colab.research.google.com/github/yuanhao-cui/SDP-Sensing-Data-Protocol-for-Scalable-Wireless-Sensing/blob/main/examples/wsdp_tutorial.ipynb) | 端到端流程 — 安装 → 预处理 → 训练 → 评估 → CLI |

### 📘 使用指南

| 资源 | 说明 |
|:-----|:-----|
| [安装指南](docs/getting-started/installation.md) | 环境搭建与配置 |
| [快速开始](docs/getting-started/quickstart.md) | WSDP 第一步 |
| [算法指南](docs/getting-started/algorithm-guide.md) | 如何选择和组合预处理算法 |
| [Python API](docs/user-guide/python-api.md) | 编程接口详细用法 |
| [CLI 参考](docs/user-guide/cli.md) | 命令行接口使用说明 |
| [配置文件](docs/user-guide/configuration.md) | YAML 配置与 Pipeline 预设 |

### 📊 参考资料

| 资源 | 说明 |
|:-----|:-----|
| [完整文档站](https://yuanhao-cui.github.io/SDP-Sensing-Data-Protocol-for-Scalable-Wireless-Sensing/) | MkDocs 完整文档 |
| [API 参考](docs/API_REFERENCE.md) | 所有公开 API |
| [数据集总览](docs/datasets/overview.md) | 5 个数据集的格式详情与下载说明 |
| [模型指南](docs/models.md) | 全部 19 个模型的架构详情 |
| [排行榜](docs/leaderboard.md) | 跨模型、跨数据集的基准对比 |
| [更新日志](CHANGELOG.md) | 版本历史 |
| [贡献指南](CONTRIBUTING.md) | 开发规范与 PR 流程 |

---

## 🗺️ 路线图

- [x] **v0.1** - 初始协议设计
- [x] **v0.2** - 5 个数据集支持，CLI 工具
- [x] **v0.3** - 更多数据集（WiFi-HAR、CSI-HAR 等）
- [x] **v0.4** - 19 个模型，26+ 算法，排行榜，CI/CD，科学 bug 修复
- [x] **v0.5** - PyPI 正式发布，在线演示平台
- [ ] **v1.0** - 完整协议标准化

**想要特定数据集？** [提交 issue](https://github.com/yuanhao-cui/Sensing-Data-Protocol-for-Scalable-Wireless-Sensing/issues) 告诉我们！

---

## 🤝 贡献

欢迎贡献！查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解：
- 开发环境搭建
- 编码规范
- Pull Request 流程

---

## 📄 许可证

MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

---

<div align="center">

**Made with ❤️ by the WSDP Team**

[⬆ Back to Top](#sdp-sensing-data-protocol-for-scalable-wireless-sensing)

</div>
