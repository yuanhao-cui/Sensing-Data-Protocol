# Pipeline Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an offline CSI pipeline diagnostic API that writes fixed before/after visual panels and proxy metrics without services or GPU.

**Architecture:** Add a focused `wsdp.algorithms.pipeline_diagnostics` module for validation, metrics, and artifact writing. Reuse existing `doppler_spectrum()` and matplotlib infrastructure; expose the public API through `wsdp.algorithms.__init__`.

**Tech Stack:** Python, NumPy, SciPy STFT through existing Doppler feature, matplotlib Agg-compatible plotting, pytest temporary directories.

---

### Task 1: Diagnostic Contract Tests

**Files:**
- Create: `tests/test_pipeline_diagnostics.py`
- Modify: none

- [ ] Write tests for artifact generation, invalid shape handling, Doppler panel creation, and bad-pipeline metric degradation.
- [ ] Run `PYTHONPATH=src .venv/bin/python -m pytest tests/test_pipeline_diagnostics.py -q` and confirm failures reference missing diagnostic API.

### Task 2: Minimal Diagnostic Module

**Files:**
- Create: `src/wsdp/algorithms/pipeline_diagnostics.py`
- Modify: `src/wsdp/algorithms/__init__.py`

- [ ] Implement `compute_pipeline_metrics(raw, processed, sampling_rate=1.0, motion_band=None)` returning a flat dict of finite numeric proxy metrics.
- [ ] Implement `plot_pipeline_diagnostics(raw, processed, save_path, include_doppler=True, ...)` returning a matplotlib Figure and writing one PNG.
- [ ] Implement `run_pipeline_diagnostics(raw, processed, output_dir, sample_name='sample', include_doppler=True, ...)` writing `<sample_name>_diagnostic.png`, `metrics.csv`, and `manifest.json`.
- [ ] Export the three functions from `wsdp.algorithms`.

### Task 3: Verification And Cleanup

**Files:**
- Modify tests/code only as needed.

- [ ] Run `PYTHONPATH=src .venv/bin/python -m pytest tests/test_pipeline_diagnostics.py tests/test_visualization.py -q`.
- [ ] Run an independent verification agent on the same tests and a code-review agent on the diff.
- [ ] Commit the accepted spec, plan, tests, and implementation with a conventional commit.
