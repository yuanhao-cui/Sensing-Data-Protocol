# SDP: Sensing Data Protocol for Scalable Wireless Sensing（中文文档）

> 🇬🇧 English version: [README.md](../README.md)

---

## 📖 论文引用

如果在研究中使用了 SDP，请引用：

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

---

## 🆕 v0.5.2 更新内容

- **模块化算法流水线** -- 通过 `AlgorithmStep` 与 `execute_algorithm_steps()` 自由组合预处理步骤
- **可插拔 Reader** -- 通过 `register_reader()` 与 `pipeline(..., reader=)`（CLI `--reader`）接入自定义文件格式
- **按数据集的流水线预设** -- `widar`、`gait`、`xrf55`、`elderAL`、`zte`，通过 `--algorithm-preset` 选择
- **兼容性与稳定性修复** -- 算法不支持某数据集时报错更清晰、保证总是保存 checkpoint、修复 Python 3.9 兼容性及 XRF55 子集划分

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

![准确率](../img/accuracy.png)
*图 1：跨数据集准确率对比*

![可复现性](../img/reproducibility_and_stability.png)
*图 2：可复现性与稳定性分析*

![消融实验](../img/ablation_rank.png)
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

# 超参数配置文件（YAML，顶层键为数据集名）
wsdp run ./data/elderAL ./output elderAL --config my_config.yaml

# 换模型：已注册的模型名，或你自己的 .py 文件
wsdp run ./data/elderAL ./output elderAL --model THAT
wsdp run ./data/elderAL ./output elderAL -m custom_model.py

# 自定义预处理：算法配置（YAML/JSON）或预设
wsdp run ./data/elderAL ./output elderAL --algorithm-config my_algorithms.yaml
wsdp run ./data/elderAL ./output elderAL --algorithm-preset high_quality
```

**📊 输出文件：**

训练后，查看 `./output/`（每个随机种子一组文件，默认 5 个种子）：
```
output/
├── best_checkpoint_<seed>.pth    # 每个种子的最佳模型检查点
├── training_history_<seed>.csv   # 每个种子逐 epoch 的损失与准确率
└── cm_rs_<seed>.png              # 每个种子的混淆矩阵
```
训练结束后，终端会打印所有种子 Top-1 准确率的均值与方差。

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

**更多数据集即将推出！**

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

标签与分组依据**文件名**解析，因此请按照某个内置数据集的命名约定命名文件
（例如 XRF55 的 `user_action_trial`），并传入对应的数据集名。若文件是自定义格式，
为它注册一个 reader 即可：

```python
from wsdp import pipeline
from wsdp.readers import BaseReader, register_reader

class MyReader(BaseReader):
    def sniff(self, file_path): return file_path.endswith(".myfmt")
    def read_file(self, file_path): ...  # 将文件解析为 CSIData

register_reader("my_format", MyReader)
# 文件名遵循 XRF55 命名约定 -> dataset="xrf55"，格式用自定义 reader 加载
pipeline("./data/my_dataset", "./output", "xrf55", reader="my_format")
```

端到端示例：[`examples/scripts/custom_reader_algorithm.py`](../examples/scripts/custom_reader_algorithm.py)。

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

**模块化 Pipeline（自由组合或跳过步骤）：**

```python
from wsdp.algorithms import AlgorithmStep
from wsdp.processors import ModularProcessor

steps = [
    AlgorithmStep(category="denoise", method="wavelet", params={"level": 2}),
    AlgorithmStep(category="normalize", method="z-score"),  # 跳过相位校准
]
data, labels, groups = ModularProcessor(steps).process(csi_data_list, dataset="xrf55")
```

配置文件形式见：[`examples/configs/modular_pipeline.yaml`](../examples/configs/modular_pipeline.yaml)。

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

> 另有按数据集命名的预设（`widar`、`gait`、`xrf55`、`elderAL`、`zte`），
> 目前与 legacy 默认链一致（线性相位校准 + 小波去噪）。

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
| 1 | [**快速上手 Notebook**](../examples/quickstart.ipynb) | 5 分钟入门 — 注册表探索与处理器定制 |
| 2 | [**入门指南 Notebook**](../examples/getting_started.ipynb) | 算法详解 — 相位校准与去噪的逐步可视化演示 |
| 3 | [**完整教程 Notebook**](../examples/wsdp_tutorial.ipynb) [![Colab](https://img.shields.io/badge/Colab-打开-yellow.svg)](https://colab.research.google.com/github/yuanhao-cui/SDP-Sensing-Data-Protocol-for-Scalable-Wireless-Sensing/blob/main/examples/wsdp_tutorial.ipynb) | 端到端流程 — 安装 → 预处理 → 训练 → 评估 → CLI |

### 📘 使用指南

| 资源 | 说明 |
|:-----|:-----|
| [安装指南](getting-started/installation.md) | 环境搭建与配置 |
| [快速开始](getting-started/quickstart.md) | WSDP 第一步 |
| [算法指南](getting-started/algorithm-guide.md) | 如何选择和组合预处理算法 |
| [Python API](user-guide/python-api.md) | 编程接口详细用法 |
| [CLI 参考](user-guide/cli.md) | 命令行接口使用说明 |
| [配置文件](user-guide/configuration.md) | YAML 配置与 Pipeline 预设 |

### 📊 参考资料

| 资源 | 说明 |
|:-----|:-----|
| [完整文档站](https://yuanhao-cui.github.io/SDP-Sensing-Data-Protocol-for-Scalable-Wireless-Sensing/) | MkDocs 完整文档 |
| [API 参考](API_REFERENCE.md) | 所有公开 API |
| [数据集总览](datasets/overview.md) | 5 个数据集的格式详情与下载说明 |
| [模型指南](models.md) | 全部 19 个模型的架构详情 |
| [排行榜](leaderboard.md) | 跨模型、跨数据集的基准对比 |
| [更新日志](../CHANGELOG.md) | 版本历史 |
| [贡献指南](../CONTRIBUTING.md) | 开发规范与 PR 流程 |

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

欢迎贡献！查看 [CONTRIBUTING.md](../CONTRIBUTING.md) 了解：
- 开发环境搭建
- 编码规范
- Pull Request 流程

---

## 📄 许可证

MIT 许可证 - 详见 [LICENSE](../LICENSE) 文件。

---

<div align="center">

**Made with ❤️ by the WSDP Team**

[⬆ Back to Top](#sdp-sensing-data-protocol-for-scalable-wireless-sensing)

</div>
