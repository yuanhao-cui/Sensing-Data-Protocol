[![SDP](https://img.shields.io/badge/SDP_Webside-Click_here-356596)](https://sdp8.org/)
[![TOML](https://img.shields.io/badge/dynamic/toml?url=https://raw.githubusercontent.com/yuanhao-cui/Sensing-Data-Protocol/refs/heads/main/pyproject.toml&query=%24.project.name&logo=pypi&label=pip)](https://pypi.org/project/wsdp/
)
[![GitHub](https://img.shields.io/github/license/yuanhao-cui/Sensing-Data-Protocol?color=green
)](https://github.com/yuanhao-cui/Sensing-Data-Protocol/blob/main/LICENSE)
![Welcome to Ask](https://img.shields.io/badge/Welcome_to-Ask-72B063)

# SDP: Sensing Data Protocol for Scalable Wireless Sensing

SDP is a protocol-level abstraction and unified benchmark for reproducible wireless sensing.

SDP is not a new neural network, but a standardized protocol that unifies CSI representations for fair comparison. Instead of improving accuracy through hidden preprocessing tricks, SDP ensures that:

- Every dataset follows the same sanitization rules
- Every model receives the same canonical tensor
- Every experiment is reproducible

SDP acts as a protocol-level middleware between raw CSI and learning models.

---
# 1. Quick Start

## Step 1: Install Dependencies
Create a virtual env in conda or python, then run:
```bash
pip install wsdp
```

## Step 2: Download Dataset
The size of `elderAL` is the smallest. Using it for a quick start is recommended.

Please download needed datasets from [Our SDP Website](http://sdp8.org/) or via command:
```bash
wsdp download eldAL ./data
```
`elderAL` can be changed to `widar`, `gait` or `xrf55`

In the folder of your project, please organize **elderAL** datasets in the structure below for extracting labels:
```
├── data
    ├── elderAL
    │   ├── action0_static_new
    │   │   ├── user0_position1_activity0
    │   │   ├── ...
    │   │
    │   ├── action1_walk_new
    │   ├── ...
    │
    ├── widar
    ├── gait
    ├── xrf55
```

## Step 3: Train and Evaluate
**Function call:**

Create a script, say `script.py`, then copy the code below and paste into the script:
```pycon
from wsdp import pipeline

pipeline("./data/elderAL", "./output", "elderAL")
```
Then run this command in Terminal:
```bash
nohup python script.py >> output.log 2>&1 &
```

**Command call:**

No need to create scripts, just run this command in Terminal:

```bash
wsdp run ./data/elderAL ./output elderAL
```

When running, SDP will automatically:
- Sanitize raw CSI
- Convert it into canonical tensors
- Train a baseline model
- Evaluate performance

After running, besides `output.log`, check `./output`, you can see:
- best_model.pth
- confusion_matrix.png

If you see these files, SDP is working correctly.

---
# 2. Modify & Research (1-Hour Challenge)
**Goal: Modify the model and produce your own results**

You can modify SDP at three levels:
- Replace the model
- Adjust preprocessing
- Add new datasets

## 2.1 Plug in your own models
Create a file: `custom_model.py` then coding for free

All model receive input in the format: `(Batch, Timestamp, Frequency, Antenna)` 

At the last line of your file, the following line should be added:
```python
model = YourCustomModelClassName
```
For more information, please refer to `default_model_template.py` in this project

Then, run:
```python
from wsdp import pipeline

pipeline("./data/elderAL", "./output", "elderAL", "custom_model.py")
```
or:
```bash
wsdp run ./data/elderAL ./output elderAL custom_model.py
```
## 2.2 Codebase Map (Where to modify)

if you want to go further:
- models/ → Define or compare architectures
- algorithms/ → Modify function for signal processing like denoising and calibration
- datasets/ → Add a new dataset
- readers/ → Add new logic for transforming a new format into `CSIFrame`
- structure/CSIFrame → Define the format of your post-process data
- processors/ → Adjust protocol logic (canonical projection, segmentation)


---
# 3. Understanding SDP (10-Min Read)

## 3.1 Why Do We Need SDP?

Wireless sensing research often suffers from:
- Hardware-specific CSI formats
- Inconsistent preprocessing pipelines
- Unstable training results
- Large performance variance across random seeds

As a result, models cannot be fairly compared.

SDP solves this problem at the protocol level, not the model level.

SDP projects raw CSI into a fixed canonical frequency grid (K=30),
ensuring cross-hardware comparability.

## 3.2 The SDP Pipeline

```
Raw CSI
  ↓
Deterministic Sanitization
  ↓
Canonical Tensor Construction
  ↓
Deep Learning Model
  ↓
Prediction
```

## 3.3 Deterministic Sanitization

Raw CSI contains hardware distortions such as:
- Phase offsets
- Sampling time offsets
- Noise fluctuations

SDP enforces deterministic calibration and denoising.

This guarantees:
- The same raw CSI always produces the same cleaned tensor.
- Reproducibility is no longer optional — it is enforced.

## 3.4 Canonical Tensor Construction

After sanitization, SDP constructs a Canonical CSI Tensor.

In the protocol definition, the tensor is expressed as:

$$X \in \mathbb{C}^{A \times K \times T}$$

Where:
- A (Antenna): spatial dimension (Tx–Rx antenna pairs)
- K (Frequency): canonical frequency resolution
- T (Timestamp): temporal samples

### 3.4.1 Canonical Frequency Resolution

SDP projects all raw CSI into a fixed canonical frequency grid:

K = 30

This is a protocol constant, not a hyperparameter.

Regardless of the original hardware (e.g., 56 or 512 subcarriers),
all CSI is interpolated into 30 standardized frequency bins.

This ensures cross-hardware comparability.

### 3.4.2 Deep Learning Input Format

For model training, the tensor is rearranged into:

(Batch, Timestamp, Frequency, Antenna)

This layout is video-like, where:
- Timestamp → time dimension
- Frequency × Antenna → spatial structure

This arrangement allows CNNs, Transformers, and RNNs to operate naturally.

## 3.5 Why This Matters

With SDP:
- Inter-seed variance is significantly reduced
- Model rankings become stable
- Cross-dataset evaluation becomes possible

SDP does not define the model.It defines the rules of the experiment.

---

# 4. Supported Dataset:

**Widar3.0**
 - Dataset Link: [Widar3.0: Wi-Fi-based Hand Gesture Recognition Dataset](http://sdp8.org/Dataset?id=028828f9-1997-48df-895c-9724551a22ae)
 - CSI Shape: (Time, 30, 1, 3)
 - num of classes: 6
 - total num of used samples: 12,000

**GaitID**
 - Dataset Link: [GaitID: Wi-Fi-based Human Gait Recognition Dataset](http://sdp8.org/Dataset?id=87a65da2-18cb-4b8f-a1ec-c9696890172b)
 - CSI Shape: (Time, 30, 1, 3)
 - num of classes: 11
 - total num of used samples: 22,500

 **XRF55**
 - Dataset Link: [XRF55: A Radio Frequency Dataset for Human Indoor Action Analysis](http://sdp8.org/Dataset?id=705e08e7-637e-49a1-aff1-b2f9644467ae)
 - CSI Shape: (270, 1000)
 - num of classes: 55
 - total num of used samples: 9,900

 **ElderAL-CSI**
 - Dataset Link: [ElderAL-CSI](http://sdp8.org/Dataset?id=f144678d-5b4a-4bb9-902c-7aff4916a029)
 - CSI Shape: (Time, 512, 3, 3)
 - num of classes: 6
 - total num of used samples: 2,400


---

# 5. Benchmark Results
**Mean Top-1 accuracy with 95% confidence intervals over five runs**
![accuracy](./img/accuracy.png)

**Performance stability comparison between the baseline and SDP across five random seeds. Boxplots show the distribution of Top-1 accuracy, with scattered dots indicating individual runs.**
![accuracy](./img/reproducibility_and_stability.png)

**Rank consistency heatmap across five random seeds on the ElderAL-CSI dataset. Colors indicate per-seed performance rank (1 = best), with overlaid Top-1 accuracy values. Full SDP exhibits stable top-ranked performance, while ablated variants show higher ranking variability.**
![accuracy](./img/ablation_rank.png)

---

# 6. Academic Reference
If you use SDP in your research, please site:
```
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
