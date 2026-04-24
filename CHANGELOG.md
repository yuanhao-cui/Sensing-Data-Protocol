# Changelog

All notable changes to WSDP are documented here.

## [0.5.0] — 2026-04-24

### 🔧 Bug Fixes

#### Wavelet Denoising 2D Input Support
**Problem**: `wavelet_denoise_csi()` unconditionally unpacked `csi_tensor.shape` into 3 dimensions (`T, S, R`), causing failures on 2D `(T, F)` input — a long-standing gap between test coverage and implementation contract.

**Fix**: Refactored the function to handle both 2D `(T, F)` and 3D `(T, F, A)` inputs. Single-antenna data is correctly processed without requiring an explicit trailing dimension.

**Files changed**: `src/wsdp/algorithms/denoising.py`, `tests/test_all_algorithms_full.py`

#### Dataset Download Reliability
- Added `allow_redirects=True` and `verify=False` to HTTP requests for environments with self-signed certificates or redirect chains.
- Suppressed `InsecureRequestWarning` to avoid noise in logs when downloading datasets.
- Fixed test patching logic for Python < 3.13 where `wsdp.download` function shadowed the module object, breaking `patch('wsdp.download.requests')`.

**Files changed**: `src/wsdp/download.py`, `tests/test_download.py`

#### `interpolate()` Parameter Passing
**Fix**: `interpolate()` now inspects the target function signature before passing `method=...`, preventing `TypeError` on registered interpolators that do not accept a `method` keyword.

**Files changed**: `src/wsdp/algorithms/__init__.py`

### 📖 Documentation & Consistency

- **API docs overhaul**: `docs/api/core.md`, `docs/api/algorithms.md`, and `docs/api/readers.md` updated with full parameter tables, accurate signatures, and usage examples aligned with current code.
- **User guide refresh**: `docs/user-guide/configuration.md` and `docs/getting-started/quickstart.md` rewritten to reflect the 6 built-in presets, custom model loading, and YAML config format.
- **README expansion**: Added structured tutorial directory, user guide links, and reference sections in both English and Chinese.
- **Doc/code sync**: Fixed mismatched function signatures in examples and tutorial notebook.

### ✨ New Features

#### Configurable Pipeline Processor
- Introduced `ConfigurableProcessor` class for user-defined algorithm pipelines:
  ```python
  from configurable_processor import ConfigurableProcessor
  processor = ConfigurableProcessor({'denoise': {'method': 'wavelet'},
                                      'calibrate': {'method': 'stc'}})
  ```
- Ships with `run_full_pipeline.py` — an end-to-end demonstration script using real `xrf55` data, covering data loading, algorithm preprocessing, GroupShuffleSplit, model training, and evaluation. Supports switching algorithms via presets or custom dicts, and swapping models via name string.

### 🧹 Code Quality

- **Repository hygiene**: Archived legacy `wsdp_old/` (29 modules) to `archive/`; removed 70 tracked `site/` MkDocs build artifacts from git.
- **Ruff lint compliance**: Fixed 22 files across `src/wsdp/` — removed unused imports (`torch.nn.functional`, `math`), eliminated dead variables, fixed PEP 8 formatting, and replaced lazy imports with top-level imports in `registry.py`.
- **Processor robustness**: `base_processor._process_single_csi()` shape guards now explicitly protect against degenerate 1D data while preserving `(T, F, 1)` for single-antenna inputs.
- **Full test suite**: Synthetic CSI data generator replaced with a physics-inspired model (static path + dynamic human-motion path + AWGN) for more realistic algorithm validation.

## [0.4.0] — 2026-03-30

### 🔧 Bug Fixes

#### Model Architecture Fix (Critical)
**Problem**: Baseline models (MLPModel, CNN1DModel, CNN2DModel, LSTMModel) had a
dimension explosion bug inherited from an earlier refactoring. Models directly
flattened the full `(T, F, A)` tensor into the first Linear layer:

```python
# BEFORE (buggy):
input_dim = T * F * A * 2  # e.g., 199*30*9*2 = 107,460
x.reshape(B, -1)           # → Linear(107460, 512) = 55M params!
```

This caused severe overfitting on small datasets and was completely untrainable
for most practical CSI shapes.

**Fix**: All baseline models now use a **Spatial Encoder** (Conv2d-based) that
compresses `(F, A)` down to 1024 dimensions per time step before the temporal
processor — exactly matching Huyuochi's original CSIModel architecture:

```python
# AFTER (fixed):
# 1. Spatial encode: (B*T, 1, F, A) → SpatialEncoder → (B*T, 1024)
# 2. Temporal: (B, T, 1024) → CNN/LSTM/Transformer → (B, latent)
# 3. Classify: Linear → (B, num_classes)

# Parameter count comparison:
MLPModel:   55M → 664k  (98.8% reduction)
CNN1DModel: untrainable → 235k
```

**Files changed**: `src/wsdp/models/baselines.py`

**Tests**: 60 forward-pass tests + 268 total tests passing ✅

#### SpatialEncoder Adaptive Padding Fix
**Problem**: Conv2d kernel_size=3 with padding=1 requires input spatial dimensions
≥ (3, 2). For very small antenna arrays (F < 3 or A < 2), the convolution would
fail with "kernel size can't be greater than actual input size".

**Fix**: Added replication padding before the first convolution when
`F < 3 or A < 3`, ensuring the kernel always operates on ≥ (3, 3) spatial
dimensions without altering valid data semantics.

#### Processor squeeze() Guard Fix
**Problem**: `base_processor._process_single_csi()` used `.squeeze()` without
checking, which could silently drop antenna dimensions for single-antenna data.

**Fix**: Replaced bare `.squeeze()` with explicit shape checking that preserves
`(T, F, 1)` for single-antenna data and explicitly guards against degenerate
1D cases.

### 📖 Documentation

- Added architecture overview to `baselines.py` docstring (canonical input shape,
  spatial encoder diagram, adaptive padding explanation)
- Added this CHANGELOG.md

### 🔬 Tests

- **268 tests passing** (all unit, integration, inference, CLI tests)
- **Critical fixes verified** by regression-testing all model forward passes with
  both real and complex input across small/default/large input shapes

---

## [Previous Versions]

See `git log` for full history.
