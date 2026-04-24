# Changelog

See [CHANGELOG.md](https://github.com/yuanhao-cui/SDP-Sensing-Data-Protocol-for-Scalable-Wireless-Sensing/blob/main/CHANGELOG.md) on GitHub for full version history.

## v0.5.0 (2026-04-24)

### Bug Fixes
- **Wavelet denoising 2D input**: `wavelet_denoise_csi()` now supports both 2D `(T, F)` and 3D `(T, F, A)` CSI inputs, fixing a known mismatch between implementation and test coverage.
- **Download reliability**: Fixed SSL/redirect handling in `download()` for restricted network environments; resolved Python < 3.13 test patching incompatibility.
- **`interpolate()` signature guard**: Prevents `TypeError` when passing `method=` to interpolators that don't accept it.

### Documentation
- Full API reference pages (`core.md`, `algorithms.md`, `readers.md`) updated with parameter tables and current signatures.
- User guide (`configuration.md`, `quickstart.md`) rewritten for 6 presets, custom models, and YAML config.
- README expanded with tutorial directory, user guide links, and bilingual reference sections.
- Docstring and example notebook signatures aligned with actual code.

### New Features
- **ConfigurableProcessor**: User-defined algorithm pipelines via dict config (e.g. `{'denoise': {'method': 'wavelet'}, 'calibrate': {'method': 'stc'}}`).
- **run_full_pipeline.py**: End-to-end demo script on real `xrf55` data with algorithm presets, model switching, GroupShuffleSplit, and full training loop.

### Code Quality
- Archived `wsdp_old/` (29 legacy modules) to `archive/`.
- Removed tracked MkDocs `site/` build artifacts from git.
- Ruff lint fixes across 22 source files: unused imports, dead variables, PEP 8 formatting.
- `base_processor` shape guards improved for single-antenna and degenerate 1D data.
- Test data upgraded to physics-inspired synthetic CSI (static + dynamic motion + AWGN).

## v0.4.0 (2026-03-30)

### Critical Scientific Fixes (Tier 0)
- **Subcarrier index mapping**: Use real IEEE 802.11n OFDM indices instead of sequential 0..29
- **MambaCSI SSM**: Fixed missing input x in state update equation (h = A*h + B*x)
- **Doppler spectrum**: STFT now operates on complex CSI (phase carries Doppler info)
- **Shannon entropy**: Use probability mass (sum=1) instead of density (integral=1)
- **Data leakage fix**: Widar/Gait grouping changed to user_id for cross-person evaluation
- **Phase preservation**: CSIDataset supports amplitude+phase dual-channel via use_phase=True
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
