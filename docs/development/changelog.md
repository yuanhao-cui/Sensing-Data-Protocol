# Changelog

See [CHANGELOG.md](https://github.com/yuanhao-cui/SDP-Sensing-Data-Protocol-for-Scalable-Wireless-Sensing/blob/main/CHANGELOG.md) on GitHub for full version history.

## v0.5.1 (2026-07-07)

### Bug Fixes
- **Test compatibility with `CSIDataset` signature**: Updated `DummyDataset` in
  `tests/test_core_configuration.py` to accept the dataset-driven
  `dataset_name` and `pipeline_steps` arguments.

### Breaking Changes
- **`CSIDataset` constructor**: Replaced
  `(data_list, labels, use_phase=False, preserve_real_sign=False)` with
  `(data_list, labels, dataset_name="", pipeline_steps=None)`. Widar and Gait
  now use amplitude+phase channels automatically from dataset policy.

### New Features
- **Backend presigned download URLs**: `download.py` now uses backend-generated
  presigned URLs for OVHCloud/MinIO storage and no longer resolves S3 regions on
  the client. Generic HEAD redirect handling is preserved.

## v0.5.0 (2026-07-07)

### New Features
- **Configurable pipeline**: `pipeline()` supports free model and algorithm selection; added `ConfigurableProcessor` for dict-based algorithm pipelines.
- **Pipeline record JSON output**: `pipeline()` can emit a JSON summary of the executed configuration.
- **Integration test runner**: `test_tools/run_integration_tests.py` exercises four datasets across four execution paths.
- **Run-full-pipeline demo**: `test_tools/run_full_pipeline.py` demonstrates end-to-end loading, preprocessing, split, training, and evaluation.

### Scientific & Evaluation Protocol
- **Dataset-specific preprocessing policy**: Added `src/wsdp/dataset_policy.py`; `execute_pipeline()` accepts `dataset=` and applies amplitude-primary cleanup for `xrf55`.
- **XRF55 repetition split**: Added `_create_xrf55_repetition_split()` with train 01–12 / val 13–16 / test 17–20 and train-only normalization statistics.
- **Condition/repetition grouping**: Widar uses position/orientation/receiver, Gait uses track/receiver, and XRF55 uses repetition id as the split group.
- **XRF55 reader consolidation**: `.npy` reader returns one `CSIData` sample per file with shape `(1000, 30, 9)` (3 receivers × 3 antennas).
- **`CSIDataset`**: Model inputs are now selected from `dataset_name` and
  `pipeline_steps`, including automatic amplitude+phase channels for Widar/Gait
  and dataset-specific amplitude handling for XRF55.

### Bug Fixes
- **Wavelet denoising 2D input**: `wavelet_denoise_csi()` supports 2D `(T, F)` and 3D `(T, F, A)` inputs.
- **Download reliability**: Fixed redirect/SSL handling and Python < 3.13 test patching for `wsdp.download`.
- **`interpolate()` signature guard**: Avoids passing `method=` to interpolators that do not accept it.
- **Butterworth denoise guard**: Changed `T < min_len` to `T <= min_len` for consistent short-sequence handling.

### Engineering
- Constrained `kagglehub` to `<1.0` in `pyproject.toml`.
- Updated `benchmark_all_models.py` and `hparam_search.py` to pass `dataset=` to `_create_data_split()`.
- Simplified integration runner to raise failures directly instead of swallowing them into status summaries.

### Code Quality
- Archived `wsdp_old/` legacy modules to `archive/`.
- Removed tracked MkDocs `site/` build artifacts.
- Reorganized scattered root-level files.
- Ruff lint fixes across `src/wsdp/`.
- Upgraded synthetic test data to a physics-inspired CSI model.

### Documentation
- README expanded with documentation/resources sections.
- API reference pages updated with parameter tables and current signatures.
- User guide rewritten for presets, custom models, and YAML config.
- Added Dataset Split Selectors table to `docs/user-guide/configuration.md`.

## v0.4.0 (2026-03-30)

### Critical Scientific Fixes (Tier 0)
- **Subcarrier index mapping**: Use real IEEE 802.11n OFDM indices instead of sequential 0..29
- **MambaCSI SSM**: Fixed missing input x in state update equation (h = A*h + B*x)
- **Doppler spectrum**: STFT now operates on complex CSI (phase carries Doppler info)
- **Shannon entropy**: Use probability mass (sum=1) instead of density (integral=1)
- **Data leakage fix**: Widar/Gait grouping changed to user_id for cross-person evaluation
- **Phase preservation**: CSIDataset now supports amplitude+phase dual-channel inputs through dataset-driven Widar/Gait policy.
- **BfeeReader tx/rx**: Antenna index mapping corrected to match Linux CSI Tool

### Engineering Fixes (Tier 1)
- Data split unified to 70/15/15 across both GroupShuffleSplit and simple paths
- Inference padding_length read from checkpoint metadata
- Fixed `_selector()` or-bug for elderAL/zte datasets

### New Preprocessing Algorithms (Tier 2)
- CSI conjugate multiplication (CFO/SFO elimination)
- AGC gain compensation for IWL5300
- PCA subcarrier fusion
- Butterworth bandpass filter (configurable frequency range)
- Hampel filter (robust impulse noise removal)
- Anti-alias decimation for subcarrier downsampling

### New SOTA Models (Tier 3) — 7 models added (total: 19)
- THAT (Two-stream Transformer), CSITime (Inception-Time), PA_CSI (Phase-Amplitude)
- WiFlexFormer (lightweight Transformer), AttentionGRU (52K params)
- EI (domain adaptation), FewSense (few-shot prototypical)

### Leaderboard & Submission System (Tier 4)
- Community benchmark leaderboard with per-dataset tables
- JSON-based submission system with CI auto-verification
- Leaderboard auto-generation from submissions

### Algorithm Accuracy Improvements (Tier 5)
- HiPPO initialization for Mamba A matrix
- GCN symmetric normalization D^{-1/2}AD^{-1/2}
- Wavelet denoising: configurable wavelet/level + BayesShrink option
- Polynomial phase calibration: overfit protection (degree clamp)
- Tensor decomposition: honest HOSVD naming + optional ALS refinement
- Change point detection: renamed from "bayesian" to "mean_shift_ratio"

### Architecture & Code Quality (Tier 6)
- pipeline() refactored into composable helper functions
- num_workers auto-detect (default: min(cpu_count, 8))
- Unified logging (print → logging module)
- Full type annotations on pipeline()

### Usability (Tier 7)
- Training progress callback system
- Preprocessing cache (SHA256 key, npz storage)
- Training checkpoint resume (resume_from parameter)
- Pretrained weights management framework

### Ecosystem (Tier 8)
- GitHub Actions CI/CD (Python 3.9/3.10/3.11 matrix + ruff lint)
- Experiment tracking (local CSV / W&B / MLflow backends)
- GroupKFold cross-validation utility
- Optuna hyperparameter search integration
- Quickstart example notebook

## v0.2.0 (2026-03-17)

- Initial open-source release
- Multi-dataset support (5 datasets)
- CLI with hyperparameter control
- Docker support
- Full documentation site
