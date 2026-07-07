# Changelog

All notable changes to WSDP are documented here.

## [0.5.1] — 2026-07-07

### 🔧 Bug Fixes

- **Test compatibility with `CSIDataset` signature**: Updated `DummyDataset` in
  `tests/test_core_configuration.py` to accept the dataset-driven
  `dataset_name` and `pipeline_steps` arguments now passed by `pipeline()`.

### ⚠️ Breaking Changes

- **`CSIDataset` constructor**: Replaced the old
  `(data_list, labels, use_phase=False, preserve_real_sign=False)` signature
  with `(data_list, labels, dataset_name="", pipeline_steps=None)`. Widar and
  Gait now use amplitude+phase channels automatically from `dataset_name` and
  `pipeline_steps`; amplitude-primary datasets such as XRF55 keep their
  dataset-specific representation.

### ✨ New Features

- **Backend presigned download URLs**: `src/wsdp/download.py` now uses
  backend-generated presigned URLs for OVHCloud/MinIO storage and no longer
  resolves S3 regions on the client. Generic HEAD redirect handling is preserved.

## [0.5.0] — 2026-07-07

### ✨ New Features

#### Configurable Core Pipeline
- The `pipeline()` entry point now supports freely configuring the model and algorithm preset via the Python API and CLI.
- Added `ConfigurableProcessor` for user-defined algorithm pipelines:
  ```python
  from wsdp.processors import ConfigurableProcessor
  processor = ConfigurableProcessor({'denoise': {'method': 'wavelet'},
                                      'calibrate': {'method': 'stc'}})
  ```
- Added `test_tools/run_full_pipeline.py` as an end-to-end demonstration script covering data loading, algorithm preprocessing, grouped train/val/test splits, model training, and evaluation.

#### Pipeline Record JSON Output
- `pipeline()` can now emit a JSON record summarizing the executed algorithm steps and configuration, making experiments easier to reproduce and compare.

#### Integration Test Runner
- Added `test_tools/run_integration_tests.py` to exercise four datasets through four execution paths (direct `pipeline()` and CLI, with and without optional model/preset arguments).

### 🔬 Scientific & Evaluation Protocol

#### Dataset-Specific Preprocessing Policy
- Introduced `src/wsdp/dataset_policy.py` with helpers for amplitude-primary datasets (currently `xrf55`).
- `execute_pipeline()` now accepts an optional `dataset` argument and applies `real_if_negligible_imaginary()` for amplitude-primary datasets whose imaginary part is negligible.
- `CSIDataset` now derives its model input representation from `dataset_name` and `pipeline_steps`, enabling automatic amplitude+phase channels for Widar/Gait while preserving dataset-specific amplitude handling for XRF55.

#### XRF55 Official-Style Repetition Split
- Added `_create_xrf55_repetition_split()` in `src/wsdp/core.py`.
- XRF55 filenames are parsed as `user_action_trial` (e.g. `03_20_08`); the trial/repetition id drives the split:
  - Train: repetitions 01–12
  - Validation: repetitions 13–16
  - Test: repetitions 17–20
- When z-score or min-max normalization is configured, statistics are fitted on the training repetitions only and applied to validation/test, preventing data leakage.
- Cache keys for XRF55 now include a protocol marker so old caches are not reused after the switch to repetition-based splits.

#### Condition and Repetition-Based Grouping
- Updated `src/wsdp/processors/base_processor.py` to use dataset-specific split groups aligned with standard evaluation protocols:
  - **Widar**: group = `position_id * 1000 + orientation_id * 100 + receiver_number` for condition-based splits.
  - **Gait**: label = `user_id`, group = `track_id * 100 + receiver_id` for held-out conditions.
  - **XRF55**: label = `action_id`, group = `repetition_id` for official trial splits.
  - **ElderAL / ZTE**: group = `position_id`.

#### XRF55 Reader Consolidates Receivers
- `XrfReader._read_npy()` now keeps all 3 receivers in a single `CSIData` sample.
- Each frame stores `(30, 9)` = (subcarrier, 3 receivers × 3 antennas) over 1000 timestamps, matching the `.dat` path layout.

### 🔧 Bug Fixes

#### Wavelet Denoising 2D Input Support
- `wavelet_denoise_csi()` now handles both 2D `(T, F)` and 3D `(T, F, A)` inputs, fixing failures on single-antenna data.

#### Dataset Download Reliability
- Added `allow_redirects=True` and `verify=False` to HTTP requests for environments with self-signed certificates or redirect chains.
- Suppressed `InsecureRequestWarning` noise during downloads.
- Fixed test patching logic for Python < 3.13 where the `wsdp.download` function shadowed the module object.

#### `interpolate()` Parameter Passing
- `interpolate()` now inspects the target function signature before passing `method=...`, preventing `TypeError` on registered interpolators that do not accept a `method` keyword.

#### Butterworth Filter Guard
- `butterworth_denoise()` length guard changed from `T < min_len` to `T <= min_len` so sequences shorter than or equal to the required padding length are skipped cleanly.

### 🛠 Engineering

- **Dependency bound**: Constrained `kagglehub` to `>=0.1,<1.0` in `pyproject.toml` to avoid incompatible 1.x releases in the supported Python matrix.
- **Benchmark script**: Updated `scripts/benchmark_all_models.py` to pass `dataset=` to `_create_data_split()`.
- **Hyperparameter search**: Updated `src/wsdp/utils/hparam_search.py` to pass `dataset=` to `_create_data_split()`.
- **Integration runner**: Revised `test_tools/run_integration_tests.py` to execute all dataset/case combinations through `main()` and let pipeline/subprocess errors raise directly.

### 🧹 Code Quality & Repository Hygiene

- Archived legacy `wsdp_old/` (29 modules) to `archive/`.
- Removed 70 tracked MkDocs `site/` build artifacts from git.
- Reorganized scattered root-level files into proper subdirectories.
- Fixed Ruff lint errors across `src/wsdp/` — unused imports, dead variables, PEP 8 formatting, and import ordering.
- Upgraded the synthetic CSI data generator to a physics-inspired model (static path + dynamic human-motion path + AWGN) for more realistic algorithm validation.

### 📖 Documentation

- Expanded the Documentation & Resources section in `README.md`.
- Updated API reference pages (`docs/api/core.md`, `docs/api/algorithms.md`, `docs/api/readers.md`) with parameter tables and current signatures.
- Rewrote user guide pages (`docs/user-guide/configuration.md`, `docs/getting-started/quickstart.md`) for the 6 built-in presets, custom model loading, and YAML config format.
- Added the Dataset Split Selectors table to `docs/user-guide/configuration.md` documenting the label/group contract for each built-in dataset.
- Fixed mismatched function signatures in examples and the tutorial notebook.

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
